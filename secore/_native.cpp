// 跨平台（Windows MSVC / Linux GCC & Clang，Intel / AMD / ARM64 通用）状态估计原生微内核。
// 设计原则：
// 1. 仅依赖 Python C API（Buffer Protocol）与 C++17 标准库，零第三方头文件与链接依赖；
// 2. BLAS/LAPACK 函数指针运行时由 SciPy cython_blas / cython_lapack PyCapsule 注入，跨平台零 ABI 冲突；
// 3. 编译选项仅使用通用指令集（不硬编码 AVX512 / -march=native），确保 Intel / AMD / ARM64 全兼容。

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// 标准 Fortran 77 BLAS/LAPACK C 函数指针签名（与 SciPy cython_blas / cython_lapack 一致）
using dgemm_fn_t = void (*)(
    char*, char*, int*, int*, int*,
    double*, double*, int*, double*, int*,
    double*, double*, int*);
using dsyrk_fn_t = void (*)(
    char*, char*, int*, int*,
    double*, double*, int*,
    double*, double*, int*);
using dtrsm_fn_t = void (*)(
    char*, char*, char*, char*, int*, int*,
    double*, double*, int*, double*, int*);
using dtrtri_fn_t = void (*)(
    char*, char*, int*, double*, int*, int*);

struct BlasApi {
    dgemm_fn_t dgemm = nullptr;
    dsyrk_fn_t dsyrk = nullptr;
    dtrsm_fn_t dtrsm = nullptr;
    dtrtri_fn_t dtrtri = nullptr;
    bool ready() const {
        return dgemm != nullptr && dsyrk != nullptr && dtrsm != nullptr && dtrtri != nullptr;
    }
};

static BlasApi g_blas;

// 轻量 RAII Py_buffer 封装，确保任意返回路径安全释放缓冲
struct PyBufGuard {
    Py_buffer view{};
    bool active = false;

    bool acquire(PyObject* obj, int flags) {
        if (PyObject_GetBuffer(obj, &view, flags) != 0) {
            return false;
        }
        active = true;
        return true;
    }

    ~PyBufGuard() {
        if (active) {
            PyBuffer_Release(&view);
        }
    }
};

// 将 int32 或 int64 一维连续缓冲统一读取为 int32 向量（若本身是 int32 则零拷贝返回指针）
struct Int32View {
    const int32_t* ptr = nullptr;
    Py_ssize_t size = 0;
    std::vector<int32_t> owned;

    bool bind(const Py_buffer& buf, const char* name) {
        if (buf.ndim != 1) {
            PyErr_Format(PyExc_ValueError, "%s must be a 1D contiguous array", name);
            return false;
        }
        size = buf.shape[0];
        if (buf.itemsize == 4) {
            ptr = static_cast<const int32_t*>(buf.buf);
            return true;
        }
        if (buf.itemsize == 8) {
            owned.resize(static_cast<size_t>(size));
            const auto* src = static_cast<const int64_t*>(buf.buf);
            for (Py_ssize_t i = 0; i < size; ++i) {
                owned[static_cast<size_t>(i)] = static_cast<int32_t>(src[i]);
            }
            ptr = owned.data();
            return true;
        }
        PyErr_Format(PyExc_ValueError, "%s must have int32 or int64 dtype", name);
        return false;
    }
};

// 将 int32 或 int64 一维连续缓冲统一读取为 int64 向量（适配 > 2^31 索引或 indptr）
struct Int64View {
    const int64_t* ptr = nullptr;
    Py_ssize_t size = 0;
    std::vector<int64_t> owned;

    bool bind(const Py_buffer& buf, const char* name) {
        if (buf.ndim != 1) {
            PyErr_Format(PyExc_ValueError, "%s must be a 1D contiguous array", name);
            return false;
        }
        size = buf.shape[0];
        if (buf.itemsize == 8) {
            ptr = static_cast<const int64_t*>(buf.buf);
            return true;
        }
        if (buf.itemsize == 4) {
            owned.resize(static_cast<size_t>(size));
            const auto* src = static_cast<const int32_t*>(buf.buf);
            for (Py_ssize_t i = 0; i < size; ++i) {
                owned[static_cast<size_t>(i)] = static_cast<int64_t>(src[i]);
            }
            ptr = owned.data();
            return true;
        }
        PyErr_Format(PyExc_ValueError, "%s must have int32 or int64 dtype", name);
        return false;
    }
};

static int resolve_openmp_threads() {
#ifdef _OPENMP
    int max_t = omp_get_max_threads();
    int target = std::max(2, std::min(8, max_t / 2));
    return std::min(max_t, target);
#else
    return 1;
#endif
}

// ============================================================================
// 1. 注册 SciPy BLAS/LAPACK 函数指针（PyCapsule）
// ============================================================================
static void* extract_capsule_ptr(PyObject* cap, const char* label) {
    if (!PyCapsule_CheckExact(cap)) {
        PyErr_Format(PyExc_TypeError, "%s must be a PyCapsule", label);
        return nullptr;
    }
    const char* name = PyCapsule_GetName(cap);
    return PyCapsule_GetPointer(cap, name);
}

static PyObject* py_register_blas_capsules(PyObject*, PyObject* args) {
    PyObject *cap_dgemm = nullptr, *cap_dsyrk = nullptr, *cap_dtrsm = nullptr, *cap_dtrtri = nullptr;
    if (!PyArg_ParseTuple(args, "OOOO", &cap_dgemm, &cap_dsyrk, &cap_dtrsm, &cap_dtrtri)) {
        return nullptr;
    }
    void* p_dgemm = extract_capsule_ptr(cap_dgemm, "dgemm");
    void* p_dsyrk = extract_capsule_ptr(cap_dsyrk, "dsyrk");
    void* p_dtrsm = extract_capsule_ptr(cap_dtrsm, "dtrsm");
    void* p_dtrtri = extract_capsule_ptr(cap_dtrtri, "dtrtri");
    if (!p_dgemm || !p_dsyrk || !p_dtrsm || !p_dtrtri) {
        return nullptr;
    }
    g_blas.dgemm = reinterpret_cast<dgemm_fn_t>(p_dgemm);
    g_blas.dsyrk = reinterpret_cast<dsyrk_fn_t>(p_dsyrk);
    g_blas.dtrsm = reinterpret_cast<dtrsm_fn_t>(p_dtrsm);
    g_blas.dtrtri = reinterpret_cast<dtrtri_fn_t>(p_dtrtri);
    Py_RETURN_TRUE;
}

// ============================================================================
// 2. 超节点 Takahashi 选元求逆（列主序直通 BLAS-3 + O(1) 行映射 + 按需释放）
// ============================================================================
static inline void invert_lower_triangular_small(int w, double* L_col) {
    // 小规模列主序下三角矩阵原位求逆（w <= 16 免 LAPACK 调用开销）
    for (int j = 0; j < w; ++j) {
        double d_inv = 1.0 / L_col[j * w + j];
        L_col[j * w + j] = d_inv;
        for (int i = j + 1; i < w; ++i) {
            double acc = 0.0;
            for (int k = j; k < i; ++k) {
                acc += L_col[k * w + i] * L_col[j * w + k];
            }
            L_col[j * w + i] = -acc / L_col[i * w + i];
        }
    }
}

static inline void dtrsm_right_lower_small(int u, int w, const double* L11, double* B) {
    // 小块原位求解 X @ L11 = B（均为列主序，L11 为 w x w 下三角，B 为 u x w）
    for (int j = w - 1; j >= 0; --j) {
        const double inv_d = 1.0 / L11[j * w + j];
        double* bj = B + static_cast<size_t>(j) * static_cast<size_t>(u);
        for (int i = 0; i < u; ++i) {
            bj[i] *= inv_d;
        }
        for (int k = 0; k < j; ++k) {
            const double l_jk = L11[k * w + j];
            if (l_jk != 0.0) {
                double* bk = B + static_cast<size_t>(k) * static_cast<size_t>(u);
                for (int i = 0; i < u; ++i) {
                    bk[i] -= l_jk * bj[i];
                }
            }
        }
    }
}

static inline void dsyrk_lower_trans_small(int w, const double* A, double* C) {
    // 小块计算 C_lower = A^T @ A（列主序 w x w 下三角，避免调用多线程 BLAS 同步开销）
    for (int j = 0; j < w; ++j) {
        const double* aj = A + static_cast<size_t>(j) * static_cast<size_t>(w);
        double* cj = C + static_cast<size_t>(j) * static_cast<size_t>(w);
        for (int i = j; i < w; ++i) {
            const double* ai = A + static_cast<size_t>(i) * static_cast<size_t>(w);
            double acc = 0.0;
            for (int k = i; k < w; ++k) {
                acc += ai[k] * aj[k];
            }
            cj[i] = acc;
        }
    }
}

static inline void dgemm_nn_update(
    int m, int n, int k,
    double alpha,
    const double* A, int lda,
    const double* B, int ldb,
    double beta,
    double* C, int ldc
) {
    // 小矩阵乘（FLOPs < 16384）直接在 C++ 内联执行，避免 OpenBLAS 线程同步开销
    if (static_cast<int64_t>(m) * n * k < 16384) {
        for (int j = 0; j < n; ++j) {
            double* cj = C + static_cast<size_t>(j) * static_cast<size_t>(ldc);
            const double* bj = B + static_cast<size_t>(j) * static_cast<size_t>(ldb);
            if (beta == 0.0) {
                for (int i = 0; i < m; ++i) cj[i] = 0.0;
            } else if (beta != 1.0) {
                for (int i = 0; i < m; ++i) cj[i] *= beta;
            }
            for (int p = 0; p < k; ++p) {
                const double bpj = alpha * bj[p];
                if (bpj != 0.0) {
                    const double* ap = A + static_cast<size_t>(p) * static_cast<size_t>(lda);
                    for (int i = 0; i < m; ++i) {
                        cj[i] += ap[i] * bpj;
                    }
                }
            }
        }
        return;
    }
    char ch_N = 'N';
    g_blas.dgemm(
        &ch_N, &ch_N, &m, &n, &k,
        &alpha, const_cast<double*>(A), &lda,
        const_cast<double*>(B), &ldb,
        &beta, C, &ldc);
}

static inline void dgemm_tn_update(
    int m, int n, int k,
    double alpha,
    const double* A, int lda,
    const double* B, int ldb,
    double beta,
    double* C, int ldc
) {
    if (static_cast<int64_t>(m) * n * k < 16384) {
        for (int j = 0; j < n; ++j) {
            double* cj = C + static_cast<size_t>(j) * static_cast<size_t>(ldc);
            const double* bj = B + static_cast<size_t>(j) * static_cast<size_t>(ldb);
            for (int i = 0; i < m; ++i) {
                const double* ai = A + static_cast<size_t>(i) * static_cast<size_t>(lda);
                double acc = 0.0;
                for (int p = 0; p < k; ++p) {
                    acc += ai[p] * bj[p];
                }
                if (beta == 0.0) {
                    cj[i] = alpha * acc;
                } else {
                    cj[i] = beta * cj[i] + alpha * acc;
                }
            }
        }
        return;
    }
    char ch_T = 'T', ch_N = 'N';
    g_blas.dgemm(
        &ch_T, &ch_N, &m, &n, &k,
        &alpha, const_cast<double*>(A), &lda,
        const_cast<double*>(B), &ldb,
        &beta, C, &ldc);
}

static bool run_supernodal_selected_inverse(
    int32_t n,
    const int64_t* indptr,
    const int32_t* indices,
    const double* ldata,
    double* zdata
) {
    if (n <= 0) {
        return true;
    }
    const int64_t nnz = indptr[n];
    if (nnz <= 0) {
        return true;
    }

    // 1) 识别连续超节点（利用 memcmp 零拷贝比对相邻列尾部索引）
    std::vector<int32_t> sn_starts;
    sn_starts.reserve(static_cast<size_t>(n) / 2 + 2);
    std::vector<int32_t> col_to_sn(static_cast<size_t>(n));
    std::vector<int32_t> col_to_off(static_cast<size_t>(n));

    sn_starts.push_back(0);
    col_to_sn[0] = 0;
    col_to_off[0] = 0;
    int32_t cur_sn = 0;

    for (int32_t j = 1; j < n; ++j) {
        int64_t s_prev = indptr[j - 1];
        int64_t e_prev = indptr[j];
        int64_t s_cur = e_prev;
        int64_t e_cur = indptr[j + 1];
        int64_t len_prev = e_prev - s_prev;
        int64_t len_cur = e_cur - s_cur;
        bool merge = false;
        if (len_cur > 0 && (len_prev - 1) == len_cur && indices[s_prev + 1] == j) {
            if (indices[e_prev - 1] == indices[e_cur - 1] &&
                std::memcmp(
                    indices + (s_prev + 1),
                    indices + s_cur,
                    static_cast<size_t>(len_cur) * sizeof(int32_t)) == 0) {
                merge = true;
            }
        }
        if (!merge) {
            ++cur_sn;
            sn_starts.push_back(j);
        }
        col_to_sn[j] = cur_sn;
        col_to_off[j] = j - sn_starts[cur_sn];
    }
    sn_starts.push_back(n);
    const int32_t n_sn = static_cast<int32_t>(sn_starts.size()) - 1;

    // 2) 统计每个超节点被上层超节点引用的次数（归零即释放其稠密块）
    std::vector<int32_t> ref_count(static_cast<size_t>(n_sn), 0);
    for (int32_t sid = 0; sid < n_sn; ++sid) {
        int32_t a = sn_starts[sid];
        int32_t w = sn_starts[sid + 1] - a;
        int64_t s0 = indptr[a];
        int64_t e0 = indptr[a + 1];
        int32_t u = static_cast<int32_t>(e0 - s0) - w;
        if (u <= 0) {
            continue;
        }
        if (w == 1 && u == 1) {
            continue;
        }
        int32_t prev_t = -1;
        for (int64_t p = s0 + w; p < e0; ++p) {
            int32_t t_sid = col_to_sn[indices[p]];
            if (t_sid != prev_t) {
                ref_count[t_sid] += 1;
                prev_t = t_sid;
            }
        }
    }

    // 3) 初始化工作区
    std::vector<double> z_diag(static_cast<size_t>(n), 0.0);
    std::vector<std::vector<double>> z_blocks(static_cast<size_t>(n_sn));
    // O(1) 行号映射表：仅当前处理超节点写入有效行偏移，结束时精确清零为 -1
    std::vector<int32_t> row_to_u(static_cast<size_t>(n), -1);

    // 复用临时稠密缓冲，避免高频小堆分配
    std::vector<double> L11_buf;
    std::vector<double> L11_inv_buf;
    std::vector<double> L21_buf;
    std::vector<double> Z21_buf;
    std::vector<double> Z11_buf;
    std::vector<double> sub_buf;
    std::vector<double> prod_buf;
    std::vector<double> prod_t_buf;
    std::vector<int32_t> hit_t_off;
    std::vector<int32_t> hit_u_pos;
    std::vector<int32_t> t_col_offs;

    char ch_L = 'L', ch_N = 'N', ch_R = 'R', ch_T = 'T';
    double one = 1.0, neg_one = -1.0, zero = 0.0;

    // 4) 自底向上超节点 Takahashi 递推
    for (int32_t sid = n_sn - 1; sid >= 0; --sid) {
        const int32_t a = sn_starts[sid];
        const int32_t b = sn_starts[sid + 1];
        const int32_t w = b - a;
        const int64_t s0 = indptr[a];
        const int64_t e0 = indptr[a + 1];
        const int32_t m = static_cast<int32_t>(e0 - s0);
        const int32_t u = m - w;
        const int32_t* rows = indices + s0;

        // 快路径 A：单列零子行（w == 1, u == 0）
        if (w == 1 && u == 0) {
            double inv_l0 = 1.0 / ldata[s0];
            double z11 = inv_l0 * inv_l0;
            zdata[s0] = z11;
            z_diag[a] = z11;
            if (ref_count[sid] > 0) {
                z_blocks[sid].assign(1, z11);
            }
            continue;
        }

        // 快路径 B：单列单子行（w == 1, u == 1）
        if (w == 1 && u == 1) {
            double inv_l0 = 1.0 / ldata[s0];
            double l1 = ldata[s0 + 1] * inv_l0;
            int32_t r1 = rows[1];
            double z21 = -z_diag[r1] * l1;
            double z11 = inv_l0 * inv_l0 - l1 * z21;
            zdata[s0] = z11;
            zdata[s0 + 1] = z21;
            z_diag[a] = z11;
            if (ref_count[sid] > 0) {
                z_blocks[sid] = {z11, z21};
            }
            continue;
        }

        // 快路径 C：单列多子行（w == 1, u >= 2），零分配双指针/查块计算
        if (w == 1) {
            const double inv_l0 = 1.0 / ldata[s0];
            const double inv_l0_sq = inv_l0 * inv_l0;
            L21_buf.resize(static_cast<size_t>(u));
            Z21_buf.assign(static_cast<size_t>(u), 0.0);
            for (int32_t k = 0; k < u; ++k) {
                L21_buf[k] = ldata[s0 + 1 + k] * inv_l0;
                row_to_u[rows[1 + k]] = k;
            }

            int32_t rs = 0;
            while (rs < u) {
                int32_t r_start = rows[1 + rs];
                int32_t t_sid = col_to_sn[r_start];
                int32_t re = rs + 1;
                while (re < u && col_to_sn[rows[1 + re]] == t_sid) {
                    ++re;
                }
                const int32_t t_a = sn_starts[t_sid];
                const int64_t ts0 = indptr[t_a];
                const int64_t te0 = indptr[t_a + 1];
                const int32_t t_m = static_cast<int32_t>(te0 - ts0);
                const int32_t* t_rows = indices + ts0;
                const auto& t_blk = z_blocks[t_sid];
                const double* t_ptr = t_blk.data();

                for (int32_t p = rs; p < re; ++p) {
                    const int32_t r_col = rows[1 + p];
                    const int32_t c_off = col_to_off[r_col];
                    const double lp = L21_buf[p];
                    const double* col_ptr = t_ptr + static_cast<size_t>(c_off) * static_cast<size_t>(t_m);
                    // 对角元
                    Z21_buf[p] -= col_ptr[c_off] * lp;
                    // 严格下三角元
                    double acc_p = 0.0;
                    if (t_m - c_off - 1 <= (u - p - 1) * 4) {
                        for (int32_t tr = c_off + 1; tr < t_m; ++tr) {
                            int32_t u_pos = row_to_u[t_rows[tr]];
                            if (u_pos > p) {
                                double zv = col_ptr[tr];
                                Z21_buf[u_pos] -= zv * lp;
                                acc_p += zv * L21_buf[u_pos];
                            }
                        }
                    } else {
                        // 双指针扫描：当子超节点极高而本超节点很短时更优
                        int32_t tr = c_off + 1;
                        for (int32_t q = p + 1; q < u; ++q) {
                            int32_t target_r = rows[1 + q];
                            while (tr < t_m && t_rows[tr] < target_r) {
                                ++tr;
                            }
                            if (tr < t_m && t_rows[tr] == target_r) {
                                double zv = col_ptr[tr];
                                Z21_buf[q] -= zv * lp;
                                acc_p += zv * L21_buf[q];
                                ++tr;
                            }
                        }
                    }
                    Z21_buf[p] -= acc_p;
                }

                if (--ref_count[t_sid] == 0) {
                    std::vector<double>().swap(z_blocks[t_sid]);
                }
                rs = re;
            }

            double z11 = inv_l0_sq;
            for (int32_t k = 0; k < u; ++k) {
                double zk = Z21_buf[k];
                zdata[s0 + 1 + k] = zk;
                z11 -= L21_buf[k] * zk;
                row_to_u[rows[1 + k]] = -1;
            }
            zdata[s0] = z11;
            z_diag[a] = z11;
            if (ref_count[sid] > 0) {
                auto& blk = z_blocks[sid];
                blk.resize(static_cast<size_t>(m));
                blk[0] = z11;
                std::memcpy(blk.data() + 1, Z21_buf.data(), static_cast<size_t>(u) * sizeof(double));
            }
            continue;
        }

        // 通用超节点路径（w >= 2，全部列主序直通 BLAS-3）
        const size_t ww = static_cast<size_t>(w) * static_cast<size_t>(w);
        L11_buf.assign(ww, 0.0);
        for (int32_t c = 0; c < w; ++c) {
            int64_t cs = indptr[a + c];
            const double* src = ldata + cs;
            double* dst = L11_buf.data() + static_cast<size_t>(c) * static_cast<size_t>(w) + c;
            std::memcpy(dst, src, static_cast<size_t>(w - c) * sizeof(double));
        }

        // 计算 L11^{-1}（下三角列主序）
        L11_inv_buf = L11_buf;
        if (w <= 16) {
            invert_lower_triangular_small(w, L11_inv_buf.data());
        } else {
            int w_i = w;
            int info = 0;
            g_blas.dtrtri(&ch_L, &ch_N, &w_i, L11_inv_buf.data(), &w_i, &info);
            if (info != 0) {
                return false;
            }
        }

        // 计算 Z11 = L11^{-T} L11^{-1}（先算下三角，小块直算免多线程 BLAS 同步）
        Z11_buf.assign(ww, 0.0);
        if (w <= 20) {
            dsyrk_lower_trans_small(w, L11_inv_buf.data(), Z11_buf.data());
        } else {
            int w_i = w;
            g_blas.dsyrk(&ch_L, &ch_T, &w_i, &w_i, &one, L11_inv_buf.data(), &w_i, &zero, Z11_buf.data(), &w_i);
        }

        if (u > 0) {
            const size_t uw = static_cast<size_t>(u) * static_cast<size_t>(w);
            L21_buf.resize(uw);
            for (int32_t c = 0; c < w; ++c) {
                int64_t cs = indptr[a + c];
                const double* src = ldata + cs + (w - c);
                double* dst = L21_buf.data() + static_cast<size_t>(c) * static_cast<size_t>(u);
                std::memcpy(dst, src, static_cast<size_t>(u) * sizeof(double));
            }
            // 原位计算 L21_eff = L21 @ L11^{-1}（求解 X L11 = L21）
            if (static_cast<int64_t>(u) * w * w < 32768) {
                dtrsm_right_lower_small(u, w, L11_buf.data(), L21_buf.data());
            } else {
                int u_i = u, w_i = w;
                g_blas.dtrsm(&ch_R, &ch_L, &ch_N, &ch_N, &u_i, &w_i, &one, L11_buf.data(), &w_i, L21_buf.data(), &u_i);
            }

            // 建立 O(1) 行映射到 [0, u)
            for (int32_t k = 0; k < u; ++k) {
                row_to_u[rows[w + k]] = k;
            }
            Z21_buf.assign(uw, 0.0);

            int32_t rs = 0;
            while (rs < u) {
                int32_t t_sid = col_to_sn[rows[w + rs]];
                int32_t re = rs + 1;
                while (re < u && col_to_sn[rows[w + re]] == t_sid) {
                    ++re;
                }
                const int32_t k_seg = re - rs;
                const int32_t t_a = sn_starts[t_sid];
                const int64_t ts0 = indptr[t_a];
                const int64_t te0 = indptr[t_a + 1];
                const int32_t t_m = static_cast<int32_t>(te0 - ts0);
                const int32_t* t_rows = indices + ts0;
                const double* t_ptr = z_blocks[t_sid].data();

                t_col_offs.resize(static_cast<size_t>(k_seg));
                for (int32_t p = 0; p < k_seg; ++p) {
                    t_col_offs[p] = col_to_off[rows[w + rs + p]];
                }
                const int32_t first_col_off = t_col_offs[0];

                // 找出 t_rows[first_col_off : t_m] 中属于当前超节点 u 行的全部交集
                hit_t_off.clear();
                hit_u_pos.clear();
                if (t_m - first_col_off <= (u - rs) * 4) {
                    for (int32_t tr = first_col_off; tr < t_m; ++tr) {
                        int32_t u_pos = row_to_u[t_rows[tr]];
                        if (u_pos >= rs) {
                            hit_t_off.push_back(tr);
                            hit_u_pos.push_back(u_pos);
                        }
                    }
                } else {
                    int32_t tr = first_col_off;
                    for (int32_t q = rs; q < u; ++q) {
                        int32_t target_r = rows[w + q];
                        while (tr < t_m && t_rows[tr] < target_r) {
                            ++tr;
                        }
                        if (tr < t_m && t_rows[tr] == target_r) {
                            hit_t_off.push_back(tr);
                            hit_u_pos.push_back(q);
                            ++tr;
                        }
                    }
                }

                const int32_t m_sub = static_cast<int32_t>(hit_t_off.size());
                if (m_sub > 0) {
                    // 收集连续列主序子矩阵 sub (m_sub x k_seg)，前 k_seg 行对应对角块（自动对称补齐上三角）
                    sub_buf.resize(static_cast<size_t>(m_sub) * static_cast<size_t>(k_seg));
                    const bool is_contig_rows =
                        (m_sub == (hit_u_pos.back() - rs + 1)) && (hit_u_pos[0] == rs);
                    const bool is_contig_t =
                        (m_sub == (hit_t_off.back() - hit_t_off[0] + 1));

                    for (int32_t p = 0; p < k_seg; ++p) {
                        const double* src_col =
                            t_ptr + static_cast<size_t>(t_col_offs[p]) * static_cast<size_t>(t_m);
                        double* dst_col = sub_buf.data() + static_cast<size_t>(p) * static_cast<size_t>(m_sub);
                        if (is_contig_t) {
                            std::memcpy(dst_col, src_col + hit_t_off[0], static_cast<size_t>(m_sub) * sizeof(double));
                        } else {
                            for (int32_t i = 0; i < m_sub; ++i) {
                                dst_col[i] = src_col[hit_t_off[i]];
                            }
                        }
                    }
                    // 对角块 [0..k_seg) x [0..k_seg) 若在 t_blk 中已对称，可直接用；仍以对称下三角为准镜像保险
                    // （t_blk 的 [0..t_w, 0..t_w) 已对称，故此步已天然满足 sub[i, j] == sub[j, i]）

                    int m_sub_i = m_sub, w_i = w, k_seg_i = k_seg, u_i = u;
                    double* L21_seg = L21_buf.data() + rs;

                    if (is_contig_rows) {
                        // 热路径：行连续，直接原位更新 Z21[rs : rs+m_sub, :] -= sub @ L21_eff[rs:re, :]
                        dgemm_nn_update(
                            m_sub_i, w_i, k_seg_i,
                            neg_one, sub_buf.data(), m_sub_i,
                            L21_seg, u_i,
                            one, Z21_buf.data() + rs, u_i);
                        // 严格下三角部分对称更新 Z21[rs:re, :] -= sub[k_seg:, :]^T @ L21_eff[re : rs+m_sub, :]
                        int m_strict = m_sub - k_seg;
                        if (m_strict > 0) {
                            dgemm_tn_update(
                                k_seg_i, w_i, m_strict,
                                neg_one, sub_buf.data() + k_seg, m_sub_i,
                                L21_buf.data() + re, u_i,
                                one, Z21_buf.data() + rs, u_i);
                        }
                    } else {
                        // 通用散列行路径：先乘进 prod_buf，再回写到 Z21_buf[hit_u_pos]
                        prod_buf.resize(static_cast<size_t>(m_sub) * static_cast<size_t>(w));
                        dgemm_nn_update(
                            m_sub_i, w_i, k_seg_i,
                            neg_one, sub_buf.data(), m_sub_i,
                            L21_seg, u_i,
                            zero, prod_buf.data(), m_sub_i);
                        for (int32_t c = 0; c < w; ++c) {
                            const double* p_col = prod_buf.data() + static_cast<size_t>(c) * static_cast<size_t>(m_sub);
                            double* z_col = Z21_buf.data() + static_cast<size_t>(c) * static_cast<size_t>(u);
                            for (int32_t i = 0; i < m_sub; ++i) {
                                z_col[hit_u_pos[i]] += p_col[i];
                            }
                        }
                        int m_strict = m_sub - k_seg;
                        if (m_strict > 0) {
                            // 将 L21_eff[hit_u_pos[k_seg:]] 紧凑打包后一次矩阵乘
                            prod_t_buf.resize(static_cast<size_t>(m_strict) * static_cast<size_t>(w));
                            for (int32_t c = 0; c < w; ++c) {
                                const double* l_col = L21_buf.data() + static_cast<size_t>(c) * static_cast<size_t>(u);
                                double* pt_col =
                                    prod_t_buf.data() + static_cast<size_t>(c) * static_cast<size_t>(m_strict);
                                for (int32_t i = 0; i < m_strict; ++i) {
                                    pt_col[i] = l_col[hit_u_pos[k_seg + i]];
                                }
                            }
                            dgemm_tn_update(
                                k_seg_i, w_i, m_strict,
                                neg_one, sub_buf.data() + k_seg, m_sub_i,
                                prod_t_buf.data(), m_strict,
                                one, Z21_buf.data() + rs, u_i);
                        }
                    }
                }

                if (--ref_count[t_sid] == 0) {
                    std::vector<double>().swap(z_blocks[t_sid]);
                }
                rs = re;
            }

            // 清理 O(1) 行映射表
            for (int32_t k = 0; k < u; ++k) {
                row_to_u[rows[w + k]] = -1;
            }

            // Z11 -= L21_eff^T @ Z21
            dgemm_tn_update(
                w, w, u,
                neg_one, L21_buf.data(), u,
                Z21_buf.data(), u,
                one, Z11_buf.data(), w);
        }

        // 将 Z11 的下三角对称镜像到上三角（保证供子超节点直读对称对角块）
        for (int32_t c = 0; c < w; ++c) {
            double* col_c = Z11_buf.data() + static_cast<size_t>(c) * static_cast<size_t>(w);
            z_diag[a + c] = col_c[c];
            for (int32_t r = c + 1; r < w; ++r) {
                Z11_buf[static_cast<size_t>(r) * static_cast<size_t>(w) + c] = col_c[r];
            }
        }

        // 写回稀疏输出 zdata（CSC 列主序连续 memcpy）
        for (int32_t c = 0; c < w; ++c) {
            int64_t cs = indptr[a + c];
            const double* z11_col = Z11_buf.data() + static_cast<size_t>(c) * static_cast<size_t>(w) + c;
            std::memcpy(zdata + cs, z11_col, static_cast<size_t>(w - c) * sizeof(double));
            if (u > 0) {
                const double* z21_col = Z21_buf.data() + static_cast<size_t>(c) * static_cast<size_t>(u);
                std::memcpy(zdata + cs + (w - c), z21_col, static_cast<size_t>(u) * sizeof(double));
            }
        }

        // 若仍有上层引用，保存列主序 Z_blk (m x w)
        if (ref_count[sid] > 0) {
            auto& blk = z_blocks[sid];
            blk.resize(static_cast<size_t>(m) * static_cast<size_t>(w));
            for (int32_t c = 0; c < w; ++c) {
                double* dst = blk.data() + static_cast<size_t>(c) * static_cast<size_t>(m);
                const double* z11_col = Z11_buf.data() + static_cast<size_t>(c) * static_cast<size_t>(w);
                std::memcpy(dst, z11_col, static_cast<size_t>(w) * sizeof(double));
                if (u > 0) {
                    const double* z21_col = Z21_buf.data() + static_cast<size_t>(c) * static_cast<size_t>(u);
                    std::memcpy(dst + w, z21_col, static_cast<size_t>(u) * sizeof(double));
                }
            }
        }
    }
    return true;
}

static PyObject* py_supernodal_selected_inverse(PyObject*, PyObject* args) {
    if (!g_blas.ready()) {
        PyErr_SetString(PyExc_RuntimeError, "BLAS/LAPACK capsules not registered");
        return nullptr;
    }
    int n_int = 0;
    PyObject *obj_indptr = nullptr, *obj_indices = nullptr, *obj_ldata = nullptr, *obj_zdata = nullptr;
    if (!PyArg_ParseTuple(args, "iOOOO", &n_int, &obj_indptr, &obj_indices, &obj_ldata, &obj_zdata)) {
        return nullptr;
    }
    PyBufGuard g_indptr, g_indices, g_ldata, g_zdata;
    if (!g_indptr.acquire(obj_indptr, PyBUF_C_CONTIGUOUS) ||
        !g_indices.acquire(obj_indices, PyBUF_C_CONTIGUOUS) ||
        !g_ldata.acquire(obj_ldata, PyBUF_C_CONTIGUOUS) ||
        !g_zdata.acquire(obj_zdata, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS)) {
        return nullptr;
    }
    Int64View indptr;
    Int32View indices;
    if (!indptr.bind(g_indptr.view, "indptr") || !indices.bind(g_indices.view, "indices")) {
        return nullptr;
    }
    if (g_ldata.view.itemsize != 8 || g_zdata.view.itemsize != 8) {
        PyErr_SetString(PyExc_ValueError, "ldata and zdata must be float64");
        return nullptr;
    }
    const double* ldata = static_cast<const double*>(g_ldata.view.buf);
    double* zdata = static_cast<double*>(g_zdata.view.buf);
    bool ok = false;
    Py_BEGIN_ALLOW_THREADS
    ok = run_supernodal_selected_inverse(static_cast<int32_t>(n_int), indptr.ptr, indices.ptr, ldata, zdata);
    Py_END_ALLOW_THREADS
    if (!ok) {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

// ============================================================================
// 3. OpenMP 并行量测杠杆点积 h_i^T Z_low h_i（免构建 H_p 与 SpMM 临时大矩阵）
// ============================================================================
static PyObject* py_leverage_from_selected_inv(PyObject*, PyObject* args) {
    PyObject *obj_hp = nullptr, *obj_hi = nullptr, *obj_hd = nullptr;
    PyObject *obj_invp = nullptr, *obj_zp = nullptr, *obj_zi = nullptr, *obj_zd = nullptr;
    PyObject *obj_lev = nullptr;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOO",
            &obj_hp, &obj_hi, &obj_hd,
            &obj_invp,
            &obj_zp, &obj_zi, &obj_zd,
            &obj_lev)) {
        return nullptr;
    }
    PyBufGuard g_hp, g_hi, g_hd, g_invp, g_zp, g_zi, g_zd, g_lev;
    if (!g_hp.acquire(obj_hp, PyBUF_C_CONTIGUOUS) ||
        !g_hi.acquire(obj_hi, PyBUF_C_CONTIGUOUS) ||
        !g_hd.acquire(obj_hd, PyBUF_C_CONTIGUOUS) ||
        !g_invp.acquire(obj_invp, PyBUF_C_CONTIGUOUS) ||
        !g_zp.acquire(obj_zp, PyBUF_C_CONTIGUOUS) ||
        !g_zi.acquire(obj_zi, PyBUF_C_CONTIGUOUS) ||
        !g_zd.acquire(obj_zd, PyBUF_C_CONTIGUOUS) ||
        !g_lev.acquire(obj_lev, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS)) {
        return nullptr;
    }
    Int64View hp, zp;
    Int32View hi, invp, zi;
    if (!hp.bind(g_hp.view, "H_indptr") ||
        !hi.bind(g_hi.view, "H_indices") ||
        !invp.bind(g_invp.view, "inv_P") ||
        !zp.bind(g_zp.view, "Z_indptr") ||
        !zi.bind(g_zi.view, "Z_indices")) {
        return nullptr;
    }
    const double* hd = static_cast<const double*>(g_hd.view.buf);
    const double* zd = static_cast<const double*>(g_zd.view.buf);
    double* lev = static_cast<double*>(g_lev.view.buf);
    const int64_t n_meas = hp.size - 1;
    int missing_count = 0;

    Py_BEGIN_ALLOW_THREADS
    const int n_threads = (n_meas >= 2048) ? resolve_openmp_threads() : 1;
    #pragma omp parallel num_threads(n_threads) reduction(+:missing_count)
    {
        std::vector<int32_t> cols_buf;
        std::vector<double> vals_buf;
        cols_buf.reserve(32);
        vals_buf.reserve(32);

        #pragma omp for schedule(static, 256)
        for (int64_t r = 0; r < n_meas; ++r) {
            int64_t s = hp.ptr[r];
            int64_t e = hp.ptr[r + 1];
            int64_t deg = e - s;
            if (deg <= 0) {
                lev[r] = 0.0;
                continue;
            }
            if (deg == 1) {
                double v0 = hd[s];
                if (v0 == 0.0) {
                    lev[r] = 0.0;
                } else {
                    int32_t c0 = invp.ptr[hi.ptr[s]];
                    lev[r] = v0 * v0 * zd[zp.ptr[c0]];
                }
                continue;
            }

            cols_buf.clear();
            vals_buf.clear();
            for (int64_t p = s; p < e; ++p) {
                double v = hd[p];
                if (v != 0.0) {
                    cols_buf.push_back(invp.ptr[hi.ptr[p]]);
                    vals_buf.push_back(v);
                }
            }
            const int32_t k = static_cast<int32_t>(cols_buf.size());
            if (k == 0) {
                lev[r] = 0.0;
                continue;
            }
            // 按置换后列号升序做极小插入排序
            for (int32_t i = 1; i < k; ++i) {
                int32_t kc = cols_buf[i];
                double kv = vals_buf[i];
                int32_t j = i - 1;
                while (j >= 0 && cols_buf[j] > kc) {
                    cols_buf[j + 1] = cols_buf[j];
                    vals_buf[j + 1] = vals_buf[j];
                    --j;
                }
                cols_buf[j + 1] = kc;
                vals_buf[j + 1] = kv;
            }
            // 合并可能重复的列索引
            int32_t ku = 0;
            for (int32_t i = 0; i < k; ++i) {
                if (ku > 0 && cols_buf[ku - 1] == cols_buf[i]) {
                    vals_buf[ku - 1] += vals_buf[i];
                } else {
                    cols_buf[ku] = cols_buf[i];
                    vals_buf[ku] = vals_buf[i];
                    ++ku;
                }
            }

            double acc = 0.0;
            for (int32_t a = 0; a < ku; ++a) {
                const int32_t ca = cols_buf[a];
                const double va = vals_buf[a];
                const int64_t zs = zp.ptr[ca];
                const int64_t ze = zp.ptr[ca + 1];
                // Z_low 为下三角已排序 CSC，每列首项即对角元 (ca, ca)
                acc += va * va * zd[zs];
                if (a + 1 < ku) {
                    const int32_t* z_col_idx = zi.ptr + zs + 1;
                    const double* z_col_val = zd + zs + 1;
                    int64_t z_len = ze - zs - 1;
                    int64_t lo = 0;
                    double cross = 0.0;
                    for (int32_t b = a + 1; b < ku; ++b) {
                        const int32_t cb = cols_buf[b];
                        // 指数步进/二分查找（列内单调递增，左边界 lo 持续前进）
                        int64_t hi_b = z_len;
                        while (lo < hi_b) {
                            int64_t mid = lo + ((hi_b - lo) >> 1);
                            if (z_col_idx[mid] < cb) {
                                lo = mid + 1;
                            } else {
                                hi_b = mid;
                            }
                        }
                        if (lo < z_len && z_col_idx[lo] == cb) {
                            cross += vals_buf[b] * z_col_val[lo];
                            ++lo;
                        } else {
                            // 罕见数值抵消导致的缺失项，通知 Python 走精准回退补齐
                            ++missing_count;
                        }
                    }
                    acc += 2.0 * va * cross;
                }
            }
            lev[r] = acc;
        }
    }
    Py_END_ALLOW_THREADS

    return PyLong_FromLong(missing_count);
}

// ============================================================================
// 4. OpenMP 并行固定上三角超集模式 G = A A^T (+ beta I) 数值装配
//    直接接收 A_csc (n x m) -> 跳过零元转 CSR -> 按行并行双指针点积写入 a_out
// ============================================================================
static PyObject* py_assemble_aat_upper_csr(PyObject*, PyObject* args) {
    int n_int = 0;
    double damping = 0.0;
    PyObject *obj_ap = nullptr, *obj_ai = nullptr, *obj_ad = nullptr;
    PyObject *obj_ia = nullptr, *obj_ja = nullptr, *obj_aout = nullptr;
    if (!PyArg_ParseTuple(
            args, "iOOOOOOd",
            &n_int, &obj_ap, &obj_ai, &obj_ad,
            &obj_ia, &obj_ja, &obj_aout, &damping)) {
        return nullptr;
    }
    PyBufGuard g_ap, g_ai, g_ad, g_ia, g_ja, g_aout;
    if (!g_ap.acquire(obj_ap, PyBUF_C_CONTIGUOUS) ||
        !g_ai.acquire(obj_ai, PyBUF_C_CONTIGUOUS) ||
        !g_ad.acquire(obj_ad, PyBUF_C_CONTIGUOUS) ||
        !g_ia.acquire(obj_ia, PyBUF_C_CONTIGUOUS) ||
        !g_ja.acquire(obj_ja, PyBUF_C_CONTIGUOUS) ||
        !g_aout.acquire(obj_aout, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS)) {
        return nullptr;
    }
    Int64View ap;
    Int32View ai, ia, ja;
    if (!ap.bind(g_ap.view, "A_indptr") ||
        !ai.bind(g_ai.view, "A_indices") ||
        !ia.bind(g_ia.view, "ia") ||
        !ja.bind(g_ja.view, "ja")) {
        return nullptr;
    }
    const double* ad = static_cast<const double*>(g_ad.view.buf);
    double* a_out = static_cast<double*>(g_aout.view.buf);
    const int32_t n = static_cast<int32_t>(n_int);
    const int32_t m_cols = static_cast<int32_t>(ap.size - 1);

    Py_BEGIN_ALLOW_THREADS
    // Step 1: 跳过 0.0 权重元素，将 A_csc (n x m) 转为压缩行主序 (r_ptr, r_col, r_val)
    // 列号天然递增有序，每行内列索引无需排序
    std::vector<int64_t> r_ptr(static_cast<size_t>(n) + 1, 0);
    for (int32_t col = 0; col < m_cols; ++col) {
        int64_t s = ap.ptr[col];
        int64_t e = ap.ptr[col + 1];
        for (int64_t p = s; p < e; ++p) {
            if (ad[p] != 0.0) {
                r_ptr[static_cast<size_t>(ai.ptr[p]) + 1] += 1;
            }
        }
    }
    for (int32_t i = 0; i < n; ++i) {
        r_ptr[static_cast<size_t>(i) + 1] += r_ptr[i];
    }
    const int64_t nz_total = r_ptr[n];
    std::vector<int32_t> r_col(static_cast<size_t>(nz_total));
    std::vector<double> r_val(static_cast<size_t>(nz_total));
    std::vector<int64_t> cur_pos = r_ptr;
    for (int32_t col = 0; col < m_cols; ++col) {
        int64_t s = ap.ptr[col];
        int64_t e = ap.ptr[col + 1];
        for (int64_t p = s; p < e; ++p) {
            double v = ad[p];
            if (v != 0.0) {
                int32_t row = ai.ptr[p];
                int64_t dst = cur_pos[row]++;
                r_col[dst] = col;
                r_val[dst] = v;
            }
        }
    }

    // Step 2: OpenMP 按行并行计算上三角固定模式数值（每行写独立切片，零锁零竞争）
    const int n_threads = (n >= 2048) ? resolve_openmp_threads() : 1;
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 128)
    for (int32_t i = 0; i < n; ++i) {
        const int32_t gs = ia.ptr[i];
        const int32_t ge = ia.ptr[i + 1];
        const int64_t is = r_ptr[i];
        const int64_t ie = r_ptr[i + 1];
        const int32_t* ci = r_col.data() + is;
        const double* vi = r_val.data() + is;
        const int64_t li = ie - is;

        for (int32_t gp = gs; gp < ge; ++gp) {
            const int32_t j = ja.ptr[gp];
            if (j == i) {
                double acc = damping;
                for (int64_t k = 0; k < li; ++k) {
                    acc += vi[k] * vi[k];
                }
                a_out[gp] = acc;
            } else {
                const int64_t js = r_ptr[j];
                const int64_t je = r_ptr[j + 1];
                const int32_t* cj = r_col.data() + js;
                const double* vj = r_val.data() + js;
                const int64_t lj = je - js;
                int64_t pi = 0, pj = 0;
                double acc = 0.0;
                while (pi < li && pj < lj) {
                    int32_t a_c = ci[pi];
                    int32_t b_c = cj[pj];
                    if (a_c == b_c) {
                        acc += vi[pi] * vj[pj];
                        ++pi;
                        ++pj;
                    } else if (a_c < b_c) {
                        ++pi;
                    } else {
                        ++pj;
                    }
                }
                a_out[gp] = acc;
            }
        }
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_TRUE;
}

static PyMethodDef NativeMethods[] = {
    {"register_blas_capsules", py_register_blas_capsules, METH_VARARGS,
     "Register SciPy cython_blas/cython_lapack PyCapsule function pointers."},
    {"supernodal_selected_inverse", py_supernodal_selected_inverse, METH_VARARGS,
     "Supernodal Takahashi selected inversion in C++17 + BLAS-3."},
    {"leverage_from_selected_inv", py_leverage_from_selected_inv, METH_VARARGS,
     "OpenMP parallel diag(H G^-1 H^T) from selected inverse Z_low."},
    {"assemble_aat_upper_csr", py_assemble_aat_upper_csr, METH_VARARGS,
     "OpenMP parallel fixed-pattern upper-triangular CSR assembly for G = A A^T (+ beta I)."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef NativeModule = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Cross-platform C++17/OpenMP native numerical micro-kernels for secore.",
    -1,
    NativeMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&NativeModule);
}
