// 跨平台（Windows MSVC / Linux GCC & Clang，Intel / AMD / ARM64 通用）状态估计原生微内核。
// 设计原则：
// 1. 仅依赖 Python C API（Buffer Protocol）与 C++17 标准库，零第三方头文件与链接依赖；
// 2. BLAS/LAPACK 函数指针运行时由 SciPy cython_blas / cython_lapack PyCapsule 注入，跨平台零 ABI 冲突；
// 3. 编译选项仅使用通用指令集（不硬编码 AVX512 / -march=native），确保 Intel / AMD / ARM64 全兼容。

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
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

// ============================================================================
// 5. 静态 CSC->CSR 转置索引映射构建（仅首次固定模式时执行一次）
//    返回 (is_col_sorted) 并原地填充 r_ptr / r_col / r_src_pos
// ============================================================================
static PyObject* py_build_csc_transpose_map(PyObject*, PyObject* args) {
    int n_int = 0;
    PyObject *obj_ap = nullptr, *obj_ai = nullptr;
    PyObject *obj_rptr = nullptr, *obj_rcol = nullptr, *obj_rsrc = nullptr;
    if (!PyArg_ParseTuple(
            args, "iOOOOO",
            &n_int, &obj_ap, &obj_ai, &obj_rptr, &obj_rcol, &obj_rsrc)) {
        return nullptr;
    }
    PyBufGuard g_ap, g_ai, g_rptr, g_rcol, g_rsrc;
    if (!g_ap.acquire(obj_ap, PyBUF_C_CONTIGUOUS) ||
        !g_ai.acquire(obj_ai, PyBUF_C_CONTIGUOUS) ||
        !g_rptr.acquire(obj_rptr, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) ||
        !g_rcol.acquire(obj_rcol, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) ||
        !g_rsrc.acquire(obj_rsrc, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS)) {
        return nullptr;
    }
    Int64View ap;
    Int32View ai;
    if (!ap.bind(g_ap.view, "A_indptr") || !ai.bind(g_ai.view, "A_indices")) {
        return nullptr;
    }
    if (g_rptr.view.itemsize != 8 || g_rcol.view.itemsize != 4 || g_rsrc.view.itemsize != 8) {
        PyErr_SetString(PyExc_TypeError, "r_ptr/r_src must be int64 and r_col must be int32");
        return nullptr;
    }
    const int32_t n = static_cast<int32_t>(n_int);
    const int32_t m_cols = static_cast<int32_t>(ap.size - 1);
    int64_t* r_ptr = static_cast<int64_t*>(g_rptr.view.buf);
    int32_t* r_col = static_cast<int32_t*>(g_rcol.view.buf);
    int64_t* r_src = static_cast<int64_t*>(g_rsrc.view.buf);

    int is_sorted = 1;
    Py_BEGIN_ALLOW_THREADS
    std::fill(r_ptr, r_ptr + static_cast<size_t>(n) + 1, int64_t(0));
    for (int32_t col = 0; col < m_cols; ++col) {
        const int64_t s = ap.ptr[col];
        const int64_t e = ap.ptr[col + 1];
        int32_t prev_row = -1;
        for (int64_t p = s; p < e; ++p) {
            const int32_t row = ai.ptr[p];
            if (row < prev_row) {
                is_sorted = 0;
            }
            prev_row = row;
            r_ptr[static_cast<size_t>(row) + 1] += 1;
        }
    }
    for (int32_t i = 0; i < n; ++i) {
        r_ptr[static_cast<size_t>(i) + 1] += r_ptr[i];
    }
    std::vector<int64_t> cur_pos(r_ptr, r_ptr + static_cast<size_t>(n) + 1);
    for (int32_t col = 0; col < m_cols; ++col) {
        const int64_t s = ap.ptr[col];
        const int64_t e = ap.ptr[col + 1];
        for (int64_t p = s; p < e; ++p) {
            const int32_t row = ai.ptr[p];
            const int64_t dst = cur_pos[static_cast<size_t>(row)]++;
            r_col[dst] = col;
            r_src[dst] = p;
        }
    }
    Py_END_ALLOW_THREADS

    return PyBool_FromLong(is_sorted);
}

// ============================================================================
// 6. 固定模式 Gustavson OpenMP 并行装配 G = A A^T (+ beta I)
//    复用预计算转置映射 (r_ptr, r_col, r_src)，零转置、零重分配、零双指针归并
// ============================================================================
static PyObject* py_assemble_aat_upper_csr_cached(PyObject*, PyObject* args) {
    int n_int = 0;
    int col_sorted = 0;
    double damping = 0.0;
    PyObject *obj_ap = nullptr, *obj_ai = nullptr, *obj_ad = nullptr;
    PyObject *obj_rptr = nullptr, *obj_rcol = nullptr, *obj_rsrc = nullptr;
    PyObject *obj_ia = nullptr, *obj_ja = nullptr, *obj_aout = nullptr;
    if (!PyArg_ParseTuple(
            args, "iOOOOOOOOOdi",
            &n_int, &obj_ap, &obj_ai, &obj_ad,
            &obj_rptr, &obj_rcol, &obj_rsrc,
            &obj_ia, &obj_ja, &obj_aout, &damping, &col_sorted)) {
        return nullptr;
    }
    PyBufGuard g_ap, g_ai, g_ad, g_rptr, g_rcol, g_rsrc, g_ia, g_ja, g_aout;
    if (!g_ap.acquire(obj_ap, PyBUF_C_CONTIGUOUS) ||
        !g_ai.acquire(obj_ai, PyBUF_C_CONTIGUOUS) ||
        !g_ad.acquire(obj_ad, PyBUF_C_CONTIGUOUS) ||
        !g_rptr.acquire(obj_rptr, PyBUF_C_CONTIGUOUS) ||
        !g_rcol.acquire(obj_rcol, PyBUF_C_CONTIGUOUS) ||
        !g_rsrc.acquire(obj_rsrc, PyBUF_C_CONTIGUOUS) ||
        !g_ia.acquire(obj_ia, PyBUF_C_CONTIGUOUS) ||
        !g_ja.acquire(obj_ja, PyBUF_C_CONTIGUOUS) ||
        !g_aout.acquire(obj_aout, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS)) {
        return nullptr;
    }
    Int64View ap, rptr, rsrc;
    Int32View ai, rcol, ia, ja;
    if (!ap.bind(g_ap.view, "A_indptr") ||
        !ai.bind(g_ai.view, "A_indices") ||
        !rptr.bind(g_rptr.view, "r_ptr") ||
        !rcol.bind(g_rcol.view, "r_col") ||
        !rsrc.bind(g_rsrc.view, "r_src") ||
        !ia.bind(g_ia.view, "ia") ||
        !ja.bind(g_ja.view, "ja")) {
        return nullptr;
    }
    const double* ad = static_cast<const double*>(g_ad.view.buf);
    double* a_out = static_cast<double*>(g_aout.view.buf);
    const int32_t n = static_cast<int32_t>(n_int);

    int pattern_miss = 0;
    Py_BEGIN_ALLOW_THREADS
    const int n_threads = (n >= 2048) ? resolve_openmp_threads() : 1;
    #pragma omp parallel num_threads(n_threads) reduction(+:pattern_miss)
    {
        std::vector<int32_t> slot(static_cast<size_t>(n), -1);
        #pragma omp for schedule(dynamic, 128)
        for (int32_t i = 0; i < n; ++i) {
            const int32_t gs = ia.ptr[i];
            const int32_t ge = ia.ptr[i + 1];
            for (int32_t gp = gs; gp < ge; ++gp) {
                const int32_t j = ja.ptr[gp];
                a_out[gp] = (j == i) ? damping : 0.0;
                slot[j] = gp;
            }
            const int64_t rs = rptr.ptr[i];
            const int64_t re = rptr.ptr[i + 1];
            if (col_sorted) {
                for (int64_t rk = rs; rk < re; ++rk) {
                    const int64_t p0 = rsrc.ptr[rk];
                    const double vi = ad[p0];
                    if (vi == 0.0) {
                        continue;
                    }
                    const int32_t col = rcol.ptr[rk];
                    const int64_t ce = ap.ptr[col + 1];
                    for (int64_t p = p0; p < ce; ++p) {
                        const double vj = ad[p];
                        if (vj != 0.0) {
                            const int32_t gp = slot[ai.ptr[p]];
                            if (gp >= 0) {
                                a_out[gp] += vi * vj;
                            } else {
                                pattern_miss += 1;
                            }
                        }
                    }
                }
            } else {
                for (int64_t rk = rs; rk < re; ++rk) {
                    const int64_t p0 = rsrc.ptr[rk];
                    const double vi = ad[p0];
                    if (vi == 0.0) {
                        continue;
                    }
                    const int32_t col = rcol.ptr[rk];
                    const int64_t cs = ap.ptr[col];
                    const int64_t ce = ap.ptr[col + 1];
                    for (int64_t p = cs; p < ce; ++p) {
                        const int32_t j = ai.ptr[p];
                        if (j >= i) {
                            const double vj = ad[p];
                            if (vj != 0.0) {
                                const int32_t gp = slot[j];
                                if (gp >= 0) {
                                    a_out[gp] += vi * vj;
                                } else {
                                    pattern_miss += 1;
                                }
                            }
                        }
                    }
                }
            }
            for (int32_t gp = gs; gp < ge; ++gp) {
                slot[ja.ptr[gp]] = -1;
            }
        }
    }
    Py_END_ALLOW_THREADS

    if (pattern_miss > 0) {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

// ============================================================================
// 7. OpenMP 并行复电压计算 voltage * exp(1j * theta)
// ============================================================================
static PyObject* py_complex_voltage(PyObject*, PyObject* args) {
    PyObject *obj_theta = nullptr, *obj_voltage = nullptr, *obj_out = nullptr;
    if (!PyArg_ParseTuple(args, "OOO", &obj_theta, &obj_voltage, &obj_out)) {
        return nullptr;
    }
    PyBufGuard g_theta, g_voltage, g_out;
    if (!g_theta.acquire(obj_theta, PyBUF_C_CONTIGUOUS) ||
        !g_voltage.acquire(obj_voltage, PyBUF_C_CONTIGUOUS) ||
        !g_out.acquire(obj_out, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS)) {
        return nullptr;
    }
    if (g_theta.view.itemsize != 8 || g_voltage.view.itemsize != 8 || g_out.view.itemsize != 16) {
        PyErr_SetString(PyExc_ValueError, "theta/voltage must be float64 and out must be complex128");
        return nullptr;
    }
    const int64_t n = static_cast<int64_t>(g_theta.view.len / 8);
    if (static_cast<int64_t>(g_voltage.view.len / 8) != n || static_cast<int64_t>(g_out.view.len / 16) != n) {
        PyErr_SetString(PyExc_ValueError, "array length mismatch in complex_voltage");
        return nullptr;
    }
    const double* theta = static_cast<const double*>(g_theta.view.buf);
    const double* voltage = static_cast<const double*>(g_voltage.view.buf);
    double* out = static_cast<double*>(g_out.view.buf);

    Py_BEGIN_ALLOW_THREADS
    // 轻量内存带宽受限核保持单线程执行，避免唤醒 MSVC vcomp140 线程池与 MKL libiomp5md 自旋争抢 CPU
    for (int64_t i = 0; i < n; ++i) {
        const double v = voltage[i];
        const double th = theta[i];
        out[2 * i] = v * std::cos(th);
        out[2 * i + 1] = v * std::sin(th);
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_TRUE;
}

// 通用输出通道：支持按槽位直接散射写入目标稀疏 CSR 缓冲或连续写入独立数组
struct ScatterChannel {
    double* dst = nullptr;
    const int64_t* slots = nullptr;
    int64_t dst_len = 0;

    inline void write(int64_t k, double val) const noexcept {
        if (dst == nullptr) {
            return;
        }
        if (slots != nullptr) {
            const int64_t s = slots[k];
            if (s >= 0) {
                dst[s] = val;
            }
        } else {
            dst[k] = val;
        }
    }
};

static bool bind_scatter_channel(
    PyObject* obj_dst,
    PyObject* obj_slots,
    int64_t expected_k,
    PyBufGuard& g_dst,
    PyBufGuard& g_slots,
    ScatterChannel& ch) {
    ch.dst = nullptr;
    ch.slots = nullptr;
    ch.dst_len = 0;
    if (obj_dst == nullptr || obj_dst == Py_None) {
        return true;
    }
    if (!g_dst.acquire(obj_dst, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS)) {
        return false;
    }
    if (g_dst.view.itemsize != 8) {
        PyErr_SetString(PyExc_ValueError, "scatter destination must be float64");
        return false;
    }
    ch.dst = static_cast<double*>(g_dst.view.buf);
    ch.dst_len = static_cast<int64_t>(g_dst.view.len / 8);
    if (obj_slots != nullptr && obj_slots != Py_None) {
        if (!g_slots.acquire(obj_slots, PyBUF_C_CONTIGUOUS)) {
            return false;
        }
        if (g_slots.view.itemsize != 8) {
            PyErr_SetString(PyExc_ValueError, "scatter slots must be int64");
            return false;
        }
        const int64_t slen = static_cast<int64_t>(g_slots.view.len / 8);
        if (slen == 0) {
            ch.dst = nullptr;
            return true;
        }
        if (slen != expected_k) {
            PyErr_SetString(PyExc_ValueError, "scatter slots length mismatch");
            return false;
        }
        ch.slots = static_cast<const int64_t*>(g_slots.view.buf);
    } else if (ch.dst_len < expected_k) {
        PyErr_SetString(PyExc_ValueError, "output buffer too short");
        return false;
    }
    return true;
}

// ============================================================================
// 8. OpenMP 并行 AC 支路/变压器功率量测求值
// ============================================================================
static PyObject* py_eval_ac_branch_power_values(PyObject*, PyObject* args) {
    PyObject *obj_values = nullptr, *obj_rows = nullptr;
    PyObject *obj_own = nullptr, *obj_other = nullptr, *obj_third = nullptr;
    PyObject *obj_is_p = nullptr;
    PyObject *obj_ys = nullptr, *obj_ym = nullptr, *obj_yt = nullptr;
    PyObject *obj_vc = nullptr;
    int has_third = 0;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOOi",
            &obj_values, &obj_rows, &obj_own, &obj_other, &obj_third,
            &obj_is_p, &obj_ys, &obj_ym, &obj_yt, &obj_vc, &has_third)) {
        return nullptr;
    }
    PyBufGuard g_val, g_rows, g_own, g_other, g_third, g_isp, g_ys, g_ym, g_yt, g_vc;
    if (!g_val.acquire(obj_values, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) ||
        !g_rows.acquire(obj_rows, PyBUF_C_CONTIGUOUS) ||
        !g_own.acquire(obj_own, PyBUF_C_CONTIGUOUS) ||
        !g_other.acquire(obj_other, PyBUF_C_CONTIGUOUS) ||
        !g_isp.acquire(obj_is_p, PyBUF_C_CONTIGUOUS) ||
        !g_ys.acquire(obj_ys, PyBUF_C_CONTIGUOUS) ||
        !g_ym.acquire(obj_ym, PyBUF_C_CONTIGUOUS) ||
        !g_vc.acquire(obj_vc, PyBUF_C_CONTIGUOUS)) {
        return nullptr;
    }
    if (has_third) {
        if (!g_third.acquire(obj_third, PyBUF_C_CONTIGUOUS) ||
            !g_yt.acquire(obj_yt, PyBUF_C_CONTIGUOUS)) {
            return nullptr;
        }
    }
    Int64View rows, own, other, third;
    if (!rows.bind(g_rows.view, "rows") ||
        !own.bind(g_own.view, "own") ||
        !other.bind(g_other.view, "other")) {
        return nullptr;
    }
    if (has_third && !third.bind(g_third.view, "third")) {
        return nullptr;
    }
    if (g_val.view.itemsize != 8 || g_isp.view.itemsize != 1 ||
        g_ys.view.itemsize != 16 || g_ym.view.itemsize != 16 || g_vc.view.itemsize != 16 ||
        (has_third && g_yt.view.itemsize != 16)) {
        PyErr_SetString(PyExc_ValueError, "invalid dtype in eval_ac_branch_power_values");
        return nullptr;
    }
    const int64_t m = rows.size;
    double* values = static_cast<double*>(g_val.view.buf);
    const uint8_t* is_p = static_cast<const uint8_t*>(g_isp.view.buf);
    const double* ys = static_cast<const double*>(g_ys.view.buf);
    const double* ym = static_cast<const double*>(g_ym.view.buf);
    const double* yt = has_third ? static_cast<const double*>(g_yt.view.buf) : nullptr;
    const double* vc = static_cast<const double*>(g_vc.view.buf);

    Py_BEGIN_ALLOW_THREADS
    // 轻量内存带宽核单线程执行，零线程唤醒且不干扰 MKL PARDISO 线程池
    for (int64_t k = 0; k < m; ++k) {
        const int64_t io = own.ptr[k];
        const int64_t it = other.ptr[k];
        const double vo_r = vc[2 * io];
        const double vo_i = vc[2 * io + 1];
        const double vt_r = vc[2 * it];
        const double vt_i = vc[2 * it + 1];

        const double ys_r = ys[2 * k];
        const double ys_i = ys[2 * k + 1];
        const double ym_r = ym[2 * k];
        const double ym_i = ym[2 * k + 1];

        double cur_r = (ys_r * vo_r - ys_i * vo_i) + (ym_r * vt_r - ym_i * vt_i);
        double cur_i = (ys_r * vo_i + ys_i * vo_r) + (ym_r * vt_i + ym_i * vt_r);
        if (has_third) {
            const int64_t i3 = third.ptr[k];
            const double v3_r = vc[2 * i3];
            const double v3_i = vc[2 * i3 + 1];
            const double yt_r = yt[2 * k];
            const double yt_i = yt[2 * k + 1];
            cur_r += yt_r * v3_r - yt_i * v3_i;
            cur_i += yt_r * v3_i + yt_i * v3_r;
        }
        // S = v_own * conj(I)
        if (is_p[k]) {
            values[rows.ptr[k]] = vo_r * cur_r + vo_i * cur_i;
        } else {
            values[rows.ptr[k]] = vo_i * cur_r - vo_r * cur_i;
        }
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_TRUE;
}

// ============================================================================
// 9. OpenMP 并行 AC 支路/变压器功率雅可比计算（支持固定模式直写目标 CSR 缓冲）
// ============================================================================
static PyObject* py_fill_ac_branch_power_jacobian(PyObject*, PyObject* args) {
    PyObject *obj_own = nullptr, *obj_other = nullptr, *obj_third = nullptr;
    PyObject *obj_is_p = nullptr;
    PyObject *obj_ysc = nullptr, *obj_ymc = nullptr, *obj_ytc = nullptr;
    PyObject *obj_v = nullptr, *obj_vc = nullptr;
    int has_third = 0;
    PyObject *d0 = nullptr, *s0 = nullptr, *d1 = nullptr, *s1 = nullptr;
    PyObject *d2 = nullptr, *s2 = nullptr, *d3 = nullptr, *s3 = nullptr;
    PyObject *d4 = nullptr, *s4 = nullptr, *d5 = nullptr, *s5 = nullptr;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOiOOOOOOOOOOOO",
            &obj_own, &obj_other, &obj_third, &obj_is_p,
            &obj_ysc, &obj_ymc, &obj_ytc, &obj_v, &obj_vc, &has_third,
            &d0, &s0, &d1, &s1, &d2, &s2, &d3, &s3, &d4, &s4, &d5, &s5)) {
        return nullptr;
    }
    PyBufGuard g_own, g_other, g_third, g_isp, g_ysc, g_ymc, g_ytc, g_v, g_vc;
    if (!g_own.acquire(obj_own, PyBUF_C_CONTIGUOUS) ||
        !g_other.acquire(obj_other, PyBUF_C_CONTIGUOUS) ||
        !g_isp.acquire(obj_is_p, PyBUF_C_CONTIGUOUS) ||
        !g_ysc.acquire(obj_ysc, PyBUF_C_CONTIGUOUS) ||
        !g_ymc.acquire(obj_ymc, PyBUF_C_CONTIGUOUS) ||
        !g_v.acquire(obj_v, PyBUF_C_CONTIGUOUS) ||
        !g_vc.acquire(obj_vc, PyBUF_C_CONTIGUOUS)) {
        return nullptr;
    }
    if (has_third) {
        if (!g_third.acquire(obj_third, PyBUF_C_CONTIGUOUS) ||
            !g_ytc.acquire(obj_ytc, PyBUF_C_CONTIGUOUS)) {
            return nullptr;
        }
    }
    Int64View own, other, third;
    if (!own.bind(g_own.view, "own") || !other.bind(g_other.view, "other")) {
        return nullptr;
    }
    if (has_third && !third.bind(g_third.view, "third")) {
        return nullptr;
    }
    if (g_isp.view.itemsize != 1 || g_ysc.view.itemsize != 16 || g_ymc.view.itemsize != 16 ||
        g_v.view.itemsize != 8 || g_vc.view.itemsize != 16 ||
        (has_third && g_ytc.view.itemsize != 16)) {
        PyErr_SetString(PyExc_ValueError, "invalid dtype in fill_ac_branch_power_jacobian");
        return nullptr;
    }
    const int64_t m = own.size;
    PyBufGuard gd0, gs0, gd1, gs1, gd2, gs2, gd3, gs3, gd4, gs4, gd5, gs5;
    ScatterChannel c0, c1, c2, c3, c4, c5;
    if (!bind_scatter_channel(d0, s0, m, gd0, gs0, c0) ||
        !bind_scatter_channel(d1, s1, m, gd1, gs1, c1) ||
        !bind_scatter_channel(d2, s2, m, gd2, gs2, c2) ||
        !bind_scatter_channel(d3, s3, m, gd3, gs3, c3) ||
        !bind_scatter_channel(d4, s4, m, gd4, gs4, c4) ||
        !bind_scatter_channel(d5, s5, m, gd5, gs5, c5)) {
        return nullptr;
    }

    const uint8_t* is_p = static_cast<const uint8_t*>(g_isp.view.buf);
    const double* ysc = static_cast<const double*>(g_ysc.view.buf);
    const double* ymc = static_cast<const double*>(g_ymc.view.buf);
    const double* ytc = has_third ? static_cast<const double*>(g_ytc.view.buf) : nullptr;
    const double* voltage = static_cast<const double*>(g_v.view.buf);
    const double* vc = static_cast<const double*>(g_vc.view.buf);

    Py_BEGIN_ALLOW_THREADS
    // 轻量内存带宽核单线程执行，零线程唤醒且不干扰 MKL PARDISO 线程池
    for (int64_t k = 0; k < m; ++k) {
        const int64_t io = own.ptr[k];
        const int64_t it = other.ptr[k];
        const double vo_r = vc[2 * io];
        const double vo_i = vc[2 * io + 1];
        const double vt_r = vc[2 * it];
        const double vt_i = vc[2 * it + 1];

        // u_other = v_own * conj(v_other)
        const double uo_r = vo_r * vt_r + vo_i * vt_i;
        const double uo_i = vo_i * vt_r - vo_r * vt_i;
        const double ymc_r = ymc[2 * k];
        const double ymc_i = ymc[2 * k + 1];
        const double oo_r = ymc_r * uo_r - ymc_i * uo_i;
        const double oo_i = ymc_r * uo_i + ymc_i * uo_r;

        double off_r = oo_r;
        double off_i = oo_i;
        double o3_r = 0.0;
        double o3_i = 0.0;
        if (has_third) {
            const int64_t i3 = third.ptr[k];
            const double v3_r = vc[2 * i3];
            const double v3_i = vc[2 * i3 + 1];
            const double u3_r = vo_r * v3_r + vo_i * v3_i;
            const double u3_i = vo_i * v3_r - vo_r * v3_i;
            const double ytc_r = ytc[2 * k];
            const double ytc_i = ytc[2 * k + 1];
            o3_r = ytc_r * u3_r - ytc_i * u3_i;
            o3_i = ytc_r * u3_i + ytc_i * u3_r;
            off_r += o3_r;
            off_i += o3_i;
        }

        const double v_own_mag = voltage[io];
        const double v_other_mag = voltage[it];
        const double inv_vo = 1.0 / (v_own_mag > 1e-12 ? v_own_mag : 1e-12);
        const double inv_vt = 1.0 / (v_other_mag > 1e-12 ? v_other_mag : 1e-12);

        const double dvo_r = 2.0 * ysc[2 * k] * v_own_mag + off_r * inv_vo;
        const double dvo_i = 2.0 * ysc[2 * k + 1] * v_own_mag + off_i * inv_vo;
        const double dvt_r = oo_r * inv_vt;
        const double dvt_i = oo_i * inv_vt;

        if (is_p[k]) {
            c0.write(k, -off_i);
            c1.write(k, oo_i);
            c3.write(k, dvo_r);
            c4.write(k, dvt_r);
            if (has_third) {
                const double v3_mag = voltage[third.ptr[k]];
                const double inv_v3 = 1.0 / (v3_mag > 1e-12 ? v3_mag : 1e-12);
                c2.write(k, o3_i);
                c5.write(k, o3_r * inv_v3);
            }
        } else {
            c0.write(k, off_r);
            c1.write(k, -oo_r);
            c3.write(k, dvo_i);
            c4.write(k, dvt_i);
            if (has_third) {
                const double v3_mag = voltage[third.ptr[k]];
                const double inv_v3 = 1.0 / (v3_mag > 1e-12 ? v3_mag : 1e-12);
                c2.write(k, -o3_r);
                c5.write(k, o3_i * inv_v3);
            }
        }
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_TRUE;
}

// ============================================================================
// 10. OpenMP 并行 AC 节点功率平衡雅可比计算（单遍融合归约 + 固定模式原位直写）
// ============================================================================
static PyObject* py_fill_ac_balance_jacobian(PyObject*, PyObject* args) {
    PyObject *obj_bal_pos = nullptr, *obj_y_bal_ptr = nullptr;
    PyObject *obj_y_nodes = nullptr, *obj_y_pos = nullptr;
    PyObject *obj_y_conj = nullptr, *obj_y_off_idx = nullptr;
    PyObject *obj_y_diag_conj = nullptr;
    PyObject *obj_v = nullptr, *obj_vc = nullptr;
    int64_t n_off = 0;
    PyObject *do0 = nullptr, *so0 = nullptr, *do1 = nullptr, *so1 = nullptr;
    PyObject *do2 = nullptr, *so2 = nullptr, *do3 = nullptr, *so3 = nullptr;
    PyObject *dw0 = nullptr, *sw0 = nullptr, *dw1 = nullptr, *sw1 = nullptr;
    PyObject *dw2 = nullptr, *sw2 = nullptr, *dw3 = nullptr, *sw3 = nullptr;
    if (!PyArg_ParseTuple(
            args, "OOOOOOOOOLOOOOOOOOOOOOOOOO",
            &obj_bal_pos, &obj_y_bal_ptr, &obj_y_nodes, &obj_y_pos,
            &obj_y_conj, &obj_y_off_idx, &obj_y_diag_conj, &obj_v, &obj_vc, &n_off,
            &do0, &so0, &do1, &so1, &do2, &so2, &do3, &so3,
            &dw0, &sw0, &dw1, &sw1, &dw2, &sw2, &dw3, &sw3)) {
        return nullptr;
    }
    PyBufGuard g_bp, g_ptr, g_yn, g_yp, g_yc, g_oidx, g_ydc, g_v, g_vc;
    if (!g_bp.acquire(obj_bal_pos, PyBUF_C_CONTIGUOUS) ||
        !g_ptr.acquire(obj_y_bal_ptr, PyBUF_C_CONTIGUOUS) ||
        !g_yn.acquire(obj_y_nodes, PyBUF_C_CONTIGUOUS) ||
        !g_yp.acquire(obj_y_pos, PyBUF_C_CONTIGUOUS) ||
        !g_yc.acquire(obj_y_conj, PyBUF_C_CONTIGUOUS) ||
        !g_oidx.acquire(obj_y_off_idx, PyBUF_C_CONTIGUOUS) ||
        !g_ydc.acquire(obj_y_diag_conj, PyBUF_C_CONTIGUOUS) ||
        !g_v.acquire(obj_v, PyBUF_C_CONTIGUOUS) ||
        !g_vc.acquire(obj_vc, PyBUF_C_CONTIGUOUS)) {
        return nullptr;
    }
    Int64View bal_pos, y_bal_ptr, y_nodes, y_pos;
    Int32View y_off_idx;
    if (!bal_pos.bind(g_bp.view, "balance_pos") ||
        !y_bal_ptr.bind(g_ptr.view, "y_bal_ptr") ||
        !y_nodes.bind(g_yn.view, "y_nodes") ||
        !y_pos.bind(g_yp.view, "y_pos") ||
        !y_off_idx.bind(g_oidx.view, "y_off_idx")) {
        return nullptr;
    }
    if (g_yc.view.itemsize != 16 || g_ydc.view.itemsize != 16 ||
        g_v.view.itemsize != 8 || g_vc.view.itemsize != 16) {
        PyErr_SetString(PyExc_ValueError, "invalid dtype in fill_ac_balance_jacobian");
        return nullptr;
    }
    const int64_t n_bal = bal_pos.size;
    PyBufGuard gdo0, gso0, gdo1, gso1, gdo2, gso2, gdo3, gso3;
    PyBufGuard gdw0, gsw0, gdw1, gsw1, gdw2, gsw2, gdw3, gsw3;
    ScatterChannel co0, co1, co2, co3, cw0, cw1, cw2, cw3;
    if (!bind_scatter_channel(do0, so0, n_off, gdo0, gso0, co0) ||
        !bind_scatter_channel(do1, so1, n_off, gdo1, gso1, co1) ||
        !bind_scatter_channel(do2, so2, n_off, gdo2, gso2, co2) ||
        !bind_scatter_channel(do3, so3, n_off, gdo3, gso3, co3) ||
        !bind_scatter_channel(dw0, sw0, n_bal, gdw0, gsw0, cw0) ||
        !bind_scatter_channel(dw1, sw1, n_bal, gdw1, gsw1, cw1) ||
        !bind_scatter_channel(dw2, sw2, n_bal, gdw2, gsw2, cw2) ||
        !bind_scatter_channel(dw3, sw3, n_bal, gdw3, gsw3, cw3)) {
        return nullptr;
    }

    const double* y_conj = static_cast<const double*>(g_yc.view.buf);
    const double* y_diag_conj = static_cast<const double*>(g_ydc.view.buf);
    const double* voltage = static_cast<const double*>(g_v.view.buf);
    const double* vc = static_cast<const double*>(g_vc.view.buf);

    Py_BEGIN_ALLOW_THREADS
    // 单遍紧凑循环零临时数组，避免 vcomp140 线程池与 MKL libiomp5md 自旋争抢核
    for (int64_t b = 0; b < n_bal; ++b) {
        const int64_t s = y_bal_ptr.ptr[b];
        const int64_t e = y_bal_ptr.ptr[b + 1];
        double off_sum_r = 0.0;
        double off_sum_i = 0.0;
        double sum_all_r = 0.0;
        double sum_all_i = 0.0;

        for (int64_t k = s; k < e; ++k) {
            const int64_t p = y_pos.ptr[k];
            const int64_t nd = y_nodes.ptr[k];
            const double vp_mag = voltage[p];
            const double inv_vp = 1.0 / (vp_mag > 1e-12 ? vp_mag : 1e-12);
            const double vu_r = vc[2 * p] * inv_vp;
            const double vu_i = vc[2 * p + 1] * inv_vp;
            const double vn_r = vc[2 * nd];
            const double vn_i = vc[2 * nd + 1];

            // u = v_unit[p] * conj(V[nd])
            const double u_r = vu_r * vn_r + vu_i * vn_i;
            const double u_i = vu_i * vn_r - vu_r * vn_i;
            const double yc_r = y_conj[2 * k];
            const double yc_i = y_conj[2 * k + 1];
            const double yv_r = yc_r * u_r - yc_i * u_i;
            const double yv_i = yc_r * u_i + yc_i * u_r;

            sum_all_r += yv_r;
            sum_all_i += yv_i;

            const int32_t o = y_off_idx.ptr[k];
            if (o >= 0) {
                const double term_r = yv_r * vp_mag;
                const double term_i = yv_i * vp_mag;
                off_sum_r += term_r;
                off_sum_i += term_i;

                const double vn_mag = voltage[nd];
                const double inv_vn = 1.0 / (vn_mag > 1e-12 ? vn_mag : 1e-12);
                // theta_values = -1j * term = (term_i, -term_r)
                co0.write(o, term_i);
                co1.write(o, -term_r);
                co2.write(o, term_r * inv_vn);
                co3.write(o, term_i * inv_vn);
            }
        }

        const int64_t bp = bal_pos.ptr[b];
        const double vbp = voltage[bp];
        // own_theta = 1j * off_sum = (-off_sum_i, off_sum_r)
        cw0.write(b, -off_sum_i);
        cw1.write(b, off_sum_r);
        cw2.write(b, y_diag_conj[2 * bp] * vbp + sum_all_r);
        cw3.write(b, y_diag_conj[2 * bp + 1] * vbp + sum_all_i);
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_TRUE;
}

// ============================================================================
// 11. C++17 单遍高速解析 <Measurement> 标准无引号字节块
// ============================================================================
static inline bool parse_int64_sv(std::string_view sv, int64_t& out) noexcept {
    if (sv.empty()) return false;
    const char* p = sv.data();
    const char* e = p + sv.size();
    if (*p == '+') {
        ++p;
        if (p == e) return false;
    }
    auto res = std::from_chars(p, e, out);
    return res.ec == std::errc{} && res.ptr == e;
}

static inline bool parse_double_sv(std::string_view sv, double& out) noexcept {
    if (sv.empty()) return false;
    const char* p = sv.data();
    const char* e = p + sv.size();
    if (*p == '+') {
        ++p;
        if (p == e) return false;
    }
#if defined(_MSC_VER) || (defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L)
    auto res = std::from_chars(p, e, out);
    return res.ec == std::errc{} && res.ptr == e;
#else
    char buf[128];
    const size_t n = static_cast<size_t>(e - p);
    if (n >= sizeof(buf)) return false;
    std::memcpy(buf, p, n);
    buf[n] = '\0';
    char* endp = nullptr;
    out = std::strtod(buf, &endp);
    return endp == buf + n;
#endif
}

static bool extract_bytes_code_map(PyObject* dict_obj, std::unordered_map<std::string, int16_t>& out_map) {
    if (!PyDict_Check(dict_obj)) {
        PyErr_SetString(PyExc_TypeError, "code map must be a dict");
        return false;
    }
    PyObject *key = nullptr, *val = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(dict_obj, &pos, &key, &val)) {
        if (!PyBytes_Check(key)) {
            PyErr_SetString(PyExc_TypeError, "code map keys must be bytes");
            return false;
        }
        char* kbuf = nullptr;
        Py_ssize_t klen = 0;
        if (PyBytes_AsStringAndSize(key, &kbuf, &klen) != 0) {
            return false;
        }
        long v = PyLong_AsLong(val);
        if (v == -1 && PyErr_Occurred()) {
            return false;
        }
        out_map.emplace(std::string(kbuf, static_cast<size_t>(klen)), static_cast<int16_t>(v));
    }
    return true;
}

static PyObject* py_parse_meas_bulk_bytes(PyObject*, PyObject* args) {
    PyObject *obj_data = nullptr;
    int ncols = 0, idx_pos = 0, dt_pos = 0, dn_pos = 0, mt_pos = 0;
    int w_pos = 0, valid_pos = 0, val_pos = 0, st_pos = -1;
    int status_normal = 0, status_invalid = 1, status_removed = 2;
    PyObject *obj_dt_map = nullptr, *obj_mt_map = nullptr, *obj_st_map = nullptr;
    PyObject *obj_idx = nullptr, *obj_w = nullptr, *obj_val = nullptr;
    PyObject *obj_valid = nullptr, *obj_st = nullptr;
    PyObject *obj_dt = nullptr, *obj_mt = nullptr, *obj_dn_id = nullptr;
    if (!PyArg_ParseTuple(
            args, "OiiiiiiiiiiiiOOOOOOOOOOO",
            &obj_data,
            &ncols, &idx_pos, &dt_pos, &dn_pos, &mt_pos,
            &w_pos, &valid_pos, &val_pos, &st_pos,
            &status_normal, &status_invalid, &status_removed,
            &obj_dt_map, &obj_mt_map, &obj_st_map,
            &obj_idx, &obj_w, &obj_val, &obj_valid, &obj_st,
            &obj_dt, &obj_mt, &obj_dn_id)) {
        return nullptr;
    }

    std::unordered_map<std::string, int16_t> dt_map, mt_map, st_map;
    if (!extract_bytes_code_map(obj_dt_map, dt_map) ||
        !extract_bytes_code_map(obj_mt_map, mt_map) ||
        !extract_bytes_code_map(obj_st_map, st_map)) {
        return nullptr;
    }

    PyBufGuard g_data, g_idx, g_w, g_val, g_valid, g_st, g_dt, g_mt, g_dnid;
    if (!g_data.acquire(obj_data, PyBUF_C_CONTIGUOUS) ||
        !g_idx.acquire(obj_idx, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) ||
        !g_w.acquire(obj_w, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) ||
        !g_val.acquire(obj_val, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) ||
        !g_valid.acquire(obj_valid, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) ||
        !g_st.acquire(obj_st, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) ||
        !g_dt.acquire(obj_dt, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) ||
        !g_mt.acquire(obj_mt, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) ||
        !g_dnid.acquire(obj_dn_id, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS)) {
        return nullptr;
    }
    if (g_idx.view.itemsize != 8 || g_w.view.itemsize != 8 || g_val.view.itemsize != 8 ||
        g_valid.view.itemsize != 1 || g_st.view.itemsize != 2 ||
        g_dt.view.itemsize != 2 || g_mt.view.itemsize != 2 || g_dnid.view.itemsize != 4) {
        PyErr_SetString(PyExc_ValueError, "invalid output dtypes in parse_meas_bulk_bytes");
        return nullptr;
    }

    const int64_t n_rows = static_cast<int64_t>(g_idx.view.len / 8);
    const char* raw = static_cast<const char*>(g_data.view.buf);
    const size_t raw_len = static_cast<size_t>(g_data.view.len);
    int64_t* idx_out = static_cast<int64_t*>(g_idx.view.buf);
    double* w_out = static_cast<double*>(g_w.view.buf);
    double* val_out = static_cast<double*>(g_val.view.buf);
    uint8_t* valid_out = static_cast<uint8_t*>(g_valid.view.buf);
    int16_t* st_out = static_cast<int16_t*>(g_st.view.buf);
    int16_t* dt_out = static_cast<int16_t*>(g_dt.view.buf);
    int16_t* mt_out = static_cast<int16_t*>(g_mt.view.buf);
    int32_t* dnid_out = static_cast<int32_t*>(g_dnid.view.buf);

    std::string unique_names_joined;
    int64_t parsed_rows = 0;
    bool ok = true;

    Py_BEGIN_ALLOW_THREADS
    // 用 string_view 映射加速（底层 key 存储在 dt_map / mt_map / st_map 与 raw 中生命周期全程有效）
    std::unordered_map<std::string_view, int16_t> dt_sv, mt_sv, st_sv;
    dt_sv.reserve(dt_map.size() * 2 + 8);
    mt_sv.reserve(mt_map.size() * 2 + 8);
    st_sv.reserve(st_map.size() * 2 + 8);
    for (const auto& kv : dt_map) dt_sv.emplace(kv.first, kv.second);
    for (const auto& kv : mt_map) mt_sv.emplace(kv.first, kv.second);
    for (const auto& kv : st_map) st_sv.emplace(kv.first, kv.second);

    // 扁平开放寻址字符串去重表（FNV-1a 64-bit + 线性探测），消除 std::unordered_map 10 万次链表节点堆分配
    struct FlatSvSlot {
        uint64_t hash = 0;
        const char* ptr = nullptr;
        uint32_t len = 0;
        int32_t id = -1;
    };
    size_t ht_cap = 1024;
    const size_t min_cap = static_cast<size_t>(n_rows) + 256;
    while (ht_cap < min_cap) {
        ht_cap <<= 1;
    }
    std::vector<FlatSvSlot> ht(ht_cap);
    size_t ht_mask = ht_cap - 1;
    int32_t unique_count = 0;
    auto fnv1a_sv = [](std::string_view sv) noexcept -> uint64_t {
        uint64_t h = 14695981039346656037ULL;
        const unsigned char* p = reinterpret_cast<const unsigned char*>(sv.data());
        for (size_t i = 0; i < sv.size(); ++i) {
            h ^= static_cast<uint64_t>(p[i]);
            h *= 1099511628211ULL;
        }
        return h ? h : 1ULL;
    };
    auto ht_rehash = [&]() {
        const size_t new_cap = ht_cap << 1;
        const size_t new_mask = new_cap - 1;
        std::vector<FlatSvSlot> nh(new_cap);
        for (const auto& s : ht) {
            if (s.id >= 0) {
                size_t idx = static_cast<size_t>(s.hash) & new_mask;
                while (nh[idx].id >= 0) {
                    idx = (idx + 1) & new_mask;
                }
                nh[idx] = s;
            }
        }
        ht.swap(nh);
        ht_cap = new_cap;
        ht_mask = new_mask;
    };
    unique_names_joined.reserve(raw_len / 4 + 64);

    std::vector<std::string_view> row_toks;
    row_toks.reserve(static_cast<size_t>(ncols + 2));

    std::string_view last_dt_tok, last_mt_tok, last_st_tok, last_dn_tok;
    int16_t last_dt_code = 0, last_mt_code = 0, last_st_code = 0;
    int32_t last_dn_id = -1;
    bool has_last_dt = false, has_last_mt = false, has_last_st = false, has_last_dn = false;
    std::string upper_tmp;
    upper_tmp.reserve(32);

    size_t pos = 0;
    while (pos < raw_len) {
        size_t line_end = pos;
        while (line_end < raw_len && raw[line_end] != '\n') {
            ++line_end;
        }
        size_t cur = pos;
        pos = (line_end < raw_len) ? (line_end + 1) : raw_len;

        // 切分当前行 token（按 ASCII 空白）
        row_toks.clear();
        while (cur < line_end) {
            while (cur < line_end && static_cast<unsigned char>(raw[cur]) <= ' ') {
                ++cur;
            }
            if (cur >= line_end) break;
            const size_t t0 = cur;
            while (cur < line_end && static_cast<unsigned char>(raw[cur]) > ' ') {
                ++cur;
            }
            row_toks.emplace_back(raw + t0, cur - t0);
        }
        if (static_cast<int>(row_toks.size()) != ncols ||
            row_toks[0].size() != 1 || row_toks[0][0] != '#' ||
            parsed_rows >= n_rows) {
            ok = false;
            break;
        }

        int64_t idx_v = 0;
        double w_v = 0.0, val_v = 0.0;
        if (!parse_int64_sv(row_toks[static_cast<size_t>(idx_pos)], idx_v) ||
            !parse_double_sv(row_toks[static_cast<size_t>(w_pos)], w_v) ||
            !parse_double_sv(row_toks[static_cast<size_t>(val_pos)], val_v)) {
            ok = false;
            break;
        }
        idx_out[parsed_rows] = idx_v;
        w_out[parsed_rows] = w_v;
        val_out[parsed_rows] = val_v;

        const std::string_view valid_tok = row_toks[static_cast<size_t>(valid_pos)];
        uint8_t is_valid = (valid_tok.size() == 1 && valid_tok[0] == '1') ? 1 : 0;

        if (st_pos >= 0) {
            const std::string_view st_tok = row_toks[static_cast<size_t>(st_pos)];
            int16_t sc = 0;
            if (has_last_st && st_tok == last_st_tok) {
                sc = last_st_code;
            } else {
                auto it = st_sv.find(st_tok);
                if (it == st_sv.end()) {
                    upper_tmp.resize(st_tok.size());
                    for (size_t i = 0; i < st_tok.size(); ++i) {
                        unsigned char ch = static_cast<unsigned char>(st_tok[i]);
                        upper_tmp[i] = static_cast<char>((ch >= 'a' && ch <= 'z') ? (ch - 32) : ch);
                    }
                    it = st_sv.find(upper_tmp);
                    if (it == st_sv.end()) {
                        ok = false;
                        break;
                    }
                }
                sc = it->second;
                last_st_tok = st_tok;
                last_st_code = sc;
                has_last_st = true;
            }
            st_out[parsed_rows] = sc;
            if (sc == status_invalid || sc == status_removed) {
                is_valid = 0;
            }
        } else {
            st_out[parsed_rows] = is_valid ? static_cast<int16_t>(status_normal) : static_cast<int16_t>(status_invalid);
        }
        valid_out[parsed_rows] = is_valid;

        const std::string_view dt_tok = row_toks[static_cast<size_t>(dt_pos)];
        if (has_last_dt && dt_tok == last_dt_tok) {
            dt_out[parsed_rows] = last_dt_code;
        } else {
            auto it = dt_sv.find(dt_tok);
            const int16_t code = (it != dt_sv.end()) ? it->second : 0;
            dt_out[parsed_rows] = code;
            last_dt_tok = dt_tok;
            last_dt_code = code;
            has_last_dt = true;
        }

        const std::string_view mt_tok = row_toks[static_cast<size_t>(mt_pos)];
        if (has_last_mt && mt_tok == last_mt_tok) {
            mt_out[parsed_rows] = last_mt_code;
        } else {
            auto it = mt_sv.find(mt_tok);
            int16_t code = 0;
            if (it != mt_sv.end()) {
                code = it->second;
            } else {
                upper_tmp.resize(mt_tok.size());
                for (size_t i = 0; i < mt_tok.size(); ++i) {
                    unsigned char ch = static_cast<unsigned char>(mt_tok[i]);
                    upper_tmp[i] = static_cast<char>((ch >= 'a' && ch <= 'z') ? (ch - 32) : ch);
                }
                auto it2 = mt_sv.find(upper_tmp);
                code = (it2 != mt_sv.end()) ? it2->second : 0;
            }
            mt_out[parsed_rows] = code;
            last_mt_tok = mt_tok;
            last_mt_code = code;
            has_last_mt = true;
        }

        const std::string_view dn_tok = row_toks[static_cast<size_t>(dn_pos)];
        if (has_last_dn && dn_tok == last_dn_tok) {
            dnid_out[parsed_rows] = last_dn_id;
        } else {
            if (static_cast<size_t>(unique_count) * 10 >= ht_cap * 7) {
                ht_rehash();
            }
            const uint64_t h = fnv1a_sv(dn_tok);
            size_t slot_idx = static_cast<size_t>(h) & ht_mask;
            int32_t found_id = -1;
            const uint32_t dlen = static_cast<uint32_t>(dn_tok.size());
            while (true) {
                auto& sl = ht[slot_idx];
                if (sl.id < 0) {
                    found_id = unique_count++;
                    sl.hash = h;
                    sl.ptr = dn_tok.data();
                    sl.len = dlen;
                    sl.id = found_id;
                    if (!unique_names_joined.empty()) {
                        unique_names_joined.push_back('\n');
                    }
                    unique_names_joined.append(dn_tok.data(), dn_tok.size());
                    break;
                }
                if (sl.hash == h && sl.len == dlen && std::memcmp(sl.ptr, dn_tok.data(), dlen) == 0) {
                    found_id = sl.id;
                    break;
                }
                slot_idx = (slot_idx + 1) & ht_mask;
            }
            dnid_out[parsed_rows] = found_id;
            last_dn_tok = dn_tok;
            last_dn_id = found_id;
            has_last_dn = true;
        }

        ++parsed_rows;
    }
    if (ok && parsed_rows != n_rows) {
        ok = false;
    }
    Py_END_ALLOW_THREADS

    if (!ok) {
        Py_RETURN_NONE;
    }
    return PyBytes_FromStringAndSize(unique_names_joined.data(), static_cast<Py_ssize_t>(unique_names_joined.size()));
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
    {"build_csc_transpose_map", py_build_csc_transpose_map, METH_VARARGS,
     "Build static CSC->CSR row transpose map (r_ptr, r_col, r_src) once per fixed pattern."},
    {"assemble_aat_upper_csr_cached", py_assemble_aat_upper_csr_cached, METH_VARARGS,
     "OpenMP Gustavson fixed-pattern upper-triangular CSR assembly reusing static transpose map."},
    {"complex_voltage", py_complex_voltage, METH_VARARGS,
     "OpenMP parallel complex voltage calculation V * exp(1j * theta)."},
    {"eval_ac_branch_power_values", py_eval_ac_branch_power_values, METH_VARARGS,
     "OpenMP parallel AC branch/transformer P/Q measurement evaluation."},
    {"fill_ac_branch_power_jacobian", py_fill_ac_branch_power_jacobian, METH_VARARGS,
     "OpenMP parallel AC branch/transformer P/Q Jacobian with optional direct CSR scatter."},
    {"fill_ac_balance_jacobian", py_fill_ac_balance_jacobian, METH_VARARGS,
     "OpenMP parallel AC nodal power-balance Jacobian with fused reduction and direct CSR scatter."},
    {"parse_meas_bulk_bytes", py_parse_meas_bulk_bytes, METH_VARARGS,
     "Single-pass C++17 std::from_chars bulk parser for standard unquoted <Measurement> blocks."},
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
