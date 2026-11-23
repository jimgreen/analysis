"""跨平台（Windows / Linux，Intel / AMD x86_64）Intel MKL PARDISO 纯 ctypes 稀疏正定求解后端。

设计目标：
1. 零自编译依赖：运行时按平台自动探测 ``mkl_rt`` 动态库（支持 conda 与 ``pip install mkl``）。
2. 全架构与非 Intel CPU 兼容护栏：
   - 非 ``x86_64 / AMD64`` 架构（如 ARM64）直接跳过加载，零报错回退 CHOLMOD / SciPy；
   - 检测 CPU 厂商与 MKL 版本号：对于非 Intel CPU（如 AMD ``AuthenticAMD``），仅在 oneMKL >= 2021
     （已移除旧版 SSE2 降级陷阱，原生支持 AMD AVX2 / AVX-512）时自动启用；旧版 MKL 自动回退 CHOLMOD；
   - 首次加载强制执行 5x5 对称正定矩阵全流程冒烟验证（``phase 11 -> 22 -> 33 -> 331 -> -1``），
     校验线性解与半边前代杠杆恒等式误差 ``< 1e-11``，失败则永久静默降级。
3. 线程治理：默认使用物理核数 ``min(8, max(2, cpu_count // 2))``，支持 ``SE_MKL_NUM_THREADS``
   与 ``SE_LINEAR_SOLVER=auto|pardiso|cholmod|scipy`` 显式控制。
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csc_matrix as SP_CSC_MATRIX
from scipy.sparse import csr_matrix as SP_CSR_MATRIX
from scipy.sparse import eye as SP_EYE
from scipy.sparse import triu as SP_TRIU

try:
    from sksparse.cholmod import analyze as CHOLMOD_ANALYZE
    from sksparse.cholmod import analyze_AAt as CHOLMOD_ANALYZE_AAT
    from sksparse.cholmod import cholesky as CHOLMOD_CHOLESKY
    from sksparse.cholmod import cholesky_AAt as CHOLMOD_CHOLESKY_AAT
except Exception:
    CHOLMOD_ANALYZE = None
    CHOLMOD_ANALYZE_AAT = None
    CHOLMOD_CHOLESKY = None
    CHOLMOD_CHOLESKY_AAT = None


try:
    from .native_kernels import assemble_aat_upper_csr_native as NATIVE_ASSEMBLE_AAT_UPPER_CSR
except Exception:
    NATIVE_ASSEMBLE_AAT_UPPER_CSR = None


_C_INT32_P = ctypes.POINTER(ctypes.c_int32)
_C_DBL_P = ctypes.POINTER(ctypes.c_double)
_C_VOID_P = ctypes.c_void_p


class _MKLVersion(ctypes.Structure):
    _fields_ = [
        ("MajorVersion", ctypes.c_int),
        ("MinorVersion", ctypes.c_int),
        ("UpdateVersion", ctypes.c_int),
        ("ProductStatus", ctypes.c_char_p),
        ("Build", ctypes.c_char_p),
        ("Processor", ctypes.c_char_p),
        ("Platform", ctypes.c_char_p),
    ]


_MKL_PROBE_DONE = False
_MKL_LIB = None
_MKL_PARDISO_FN = None
_MKL_INFO: dict = {
    "available": False,
    "reason": "unprobed",
    "library_path": None,
    "mkl_major": 0,
    "mkl_minor": 0,
    "cpu_vendor": "",
    "arch": "",
    "threads": 0,
}
_DLL_DIR_HANDLES: List[object] = []


def get_linear_solver_preference() -> str:
    """读取环境变量 ``SE_LINEAR_SOLVER``，返回 ``auto`` / ``pardiso`` / ``cholmod`` / ``scipy``。"""
    raw = str(os.environ.get("SE_LINEAR_SOLVER", "auto") or "auto").strip().lower()
    if raw in ("auto", "pardiso", "mkl", "cholmod", "scipy"):
        return "pardiso" if raw == "mkl" else raw
    return "auto"


def _detect_cpu_arch() -> str:
    return str(platform.machine() or "").strip().lower()


def _is_x86_64_arch(arch: Optional[str] = None) -> bool:
    a = _detect_cpu_arch() if arch is None else str(arch).strip().lower()
    return a in ("x86_64", "amd64", "x64", "i686", "x86")


def _detect_cpu_vendor() -> str:
    """轻量探测 CPU 厂商标识（用于识别 Intel / AMD），不引入外部依赖。"""
    if sys.platform == "win32":
        ident = str(os.environ.get("PROCESSOR_IDENTIFIER", "") or "")
        if "GenuineIntel" in ident or "Intel" in ident:
            return "GenuineIntel"
        if "AuthenticAMD" in ident or "AMD" in ident:
            return "AuthenticAMD"
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                vendor, _ = winreg.QueryValueEx(key, "VendorIdentifier")
                return str(vendor or "").strip()
        except Exception:
            return ident.strip()
    else:
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    if line.lower().startswith("vendor_id"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            return parts[1].strip()
        except Exception:
            pass
    return str(platform.processor() or "").strip()


def _should_allow_mkl_on_cpu(
    *,
    arch: str,
    cpu_vendor: str,
    mkl_major: int,
    preference: str = "auto",
) -> Tuple[bool, str]:
    """根据架构、CPU 厂商、MKL 主版本与用户偏好判断是否启用 MKL PARDISO。"""
    if preference in ("cholmod", "scipy"):
        return False, f"disabled_by_env:{preference}"
    if not _is_x86_64_arch(arch):
        return False, f"unsupported_arch:{arch}"
    vendor_l = str(cpu_vendor or "").lower()
    is_amd_or_non_intel = ("amd" in vendor_l) or (bool(vendor_l) and ("intel" not in vendor_l))
    if is_amd_or_non_intel and preference == "auto" and 0 < int(mkl_major) < 2021:
        # 旧版 MKL (< 2021) 在非 Intel CPU 上存在历史 SSE2 降级问题，自动模式安全回退 CHOLMOD
        return False, f"legacy_mkl_on_non_intel:{cpu_vendor}:mkl_{mkl_major}"
    return True, "ok"


def _candidate_mkl_directories() -> List[Path]:
    dirs: List[Path] = []
    seen: set = set()

    def _add(p: Optional[Path]) -> None:
        if p is None:
            return
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp).lower() if sys.platform == "win32" else str(rp)
        if key not in seen and rp.is_dir():
            seen.add(key)
            dirs.append(rp)

    env_prefixes = [
        os.environ.get("CONDA_PREFIX"),
        os.environ.get("MKLROOT"),
        sys.prefix,
        sys.base_prefix,
    ]
    sub_dirs = ("Library/bin", "bin", "DLLs") if sys.platform == "win32" else ("lib", "lib64", "lib/intel64")
    for pref in env_prefixes:
        if not pref:
            continue
        base = Path(pref)
        for sub in sub_dirs:
            _add(base / sub)

    # pip install mkl 在虚拟环境 / site-packages 中的常见布局
    for sp in list(sys.path):
        if not sp:
            continue
        sp_path = Path(sp)
        if sp_path.name == "site-packages":
            if sys.platform == "win32":
                _add(sp_path.parent.parent / "Library" / "bin")
                _add(sp_path / "mkl" / "lib")
            else:
                _add(sp_path.parent.parent / "lib")
                _add(sp_path / "mkl" / "lib")
    return dirs


def _load_mkl_rt_library() -> Tuple[Optional[ctypes.CDLL], Optional[str]]:
    if sys.platform == "win32":
        lib_names = ("mkl_rt.3.dll", "mkl_rt.2.dll", "mkl_rt.dll")
    elif sys.platform.startswith("linux"):
        lib_names = ("libmkl_rt.so.2", "libmkl_rt.so", "libmkl_rt.so.1")
    elif sys.platform == "darwin":
        lib_names = ("libmkl_rt.2.dylib", "libmkl_rt.dylib")
    else:
        return None, None

    cand_dirs = _candidate_mkl_directories()
    # 先在已知目录按精确路径加载（避免依赖全局 PATH / LD_LIBRARY_PATH）
    for d in cand_dirs:
        for name in lib_names:
            full_path = d / name
            if not full_path.is_file():
                continue
            try:
                if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
                    try:
                        _DLL_DIR_HANDLES.append(os.add_dll_directory(str(d)))
                    except Exception:
                        pass
                lib = ctypes.CDLL(str(full_path))
                if hasattr(lib, "pardiso"):
                    return lib, str(full_path)
            except Exception:
                continue

    # 再尝试系统默认动态库搜索路径
    for name in lib_names:
        try:
            lib = ctypes.CDLL(name)
            if hasattr(lib, "pardiso"):
                return lib, name
        except Exception:
            continue
    return None, None


def _query_mkl_version(lib: ctypes.CDLL) -> Tuple[int, int]:
    fn = getattr(lib, "mkl_get_version", None) or getattr(lib, "MKL_Get_Version", None)
    if fn is None:
        return 0, 0
    try:
        fn.argtypes = [ctypes.POINTER(_MKLVersion)]
        fn.restype = None
        ver = _MKLVersion()
        fn(ctypes.byref(ver))
        return int(ver.MajorVersion), int(ver.MinorVersion)
    except Exception:
        return 0, 0


def _resolve_mkl_num_threads() -> int:
    raw = os.environ.get("SE_MKL_NUM_THREADS")
    if raw is not None and str(raw).strip():
        try:
            val = int(str(raw).strip())
            if val >= 1:
                return val
        except ValueError:
            pass
    # 默认使用物理核估算值（逻辑核一半，钳位在 [2, 8]），避免与 OpenBLAS 争抢及笔记本满载降频
    logical = int(os.cpu_count() or 4)
    return max(2, min(8, logical // 2))


def _configure_mkl_threads(lib: ctypes.CDLL, n_threads: int) -> None:
    fn = (
        getattr(lib, "MKL_Set_Num_Threads_Local", None)
        or getattr(lib, "mkl_set_num_threads_local", None)
        or getattr(lib, "MKL_Set_Num_Threads", None)
        or getattr(lib, "mkl_set_num_threads", None)
    )
    if fn is None:
        return
    try:
        fn.argtypes = [ctypes.c_int]
        fn.restype = ctypes.c_int
        fn(int(n_threads))
    except Exception:
        pass


def _smoke_test_pardiso(pardiso_fn) -> Tuple[bool, str]:
    """使用 5x5 非对角 SPD 矩阵执行完整 phase 11/22/33/331/-1 冒烟验证。"""
    try:
        A_dense = np.array(
            [
                [10.0, -2.0, 0.0, 1.5, 0.0],
                [-2.0, 9.0, -1.0, 0.0, 0.5],
                [0.0, -1.0, 8.0, -2.0, 1.0],
                [1.5, 0.0, -2.0, 11.0, -1.5],
                [0.0, 0.5, 1.0, -1.5, 7.0],
            ],
            dtype=np.float64,
        )
        Au = SP_TRIU(SP_CSR_MATRIX(A_dense), format="csr")
        Au.sort_indices()
        n = 5
        ia = np.ascontiguousarray(Au.indptr, dtype=np.int32)
        ja = np.ascontiguousarray(Au.indices, dtype=np.int32)
        a = np.ascontiguousarray(Au.data, dtype=np.float64)

        pt = np.zeros(64, dtype=np.int64)
        iparm = np.zeros(64, dtype=np.int32)
        iparm[0] = 1   # 非默认参数
        iparm[1] = 2   # METIS 嵌套剖分
        iparm[9] = 8   # SPD 小主元扰动阈值 1e-8
        iparm[17] = -1
        iparm[18] = 0
        iparm[26] = 0
        iparm[34] = 1  # 0-based 索引

        maxfct = ctypes.c_int32(1)
        mnum = ctypes.c_int32(1)
        mtype = ctypes.c_int32(2)
        n_c = ctypes.c_int32(n)
        msglvl = ctypes.c_int32(0)
        err = ctypes.c_int32(0)
        perm = np.zeros(n, dtype=np.int32)

        def _call(phase_val: int, nrhs: int, b_arr: np.ndarray, x_arr: np.ndarray) -> int:
            ph = ctypes.c_int32(phase_val)
            nr = ctypes.c_int32(nrhs)
            pardiso_fn(
                pt.ctypes.data_as(_C_VOID_P),
                ctypes.byref(maxfct),
                ctypes.byref(mnum),
                ctypes.byref(mtype),
                ctypes.byref(ph),
                ctypes.byref(n_c),
                a.ctypes.data_as(_C_DBL_P),
                ia.ctypes.data_as(_C_INT32_P),
                ja.ctypes.data_as(_C_INT32_P),
                perm.ctypes.data_as(_C_INT32_P),
                ctypes.byref(nr),
                iparm.ctypes.data_as(_C_INT32_P),
                ctypes.byref(msglvl),
                b_arr.ctypes.data_as(_C_DBL_P),
                x_arr.ctypes.data_as(_C_DBL_P),
                ctypes.byref(err),
            )
            return int(err.value)

        b1 = np.array([1.0, -2.0, 3.0, 0.5, -1.5], dtype=np.float64)
        x1 = np.zeros(n, dtype=np.float64)
        try:
            for ph in (11, 22, 33):
                code = _call(ph, 1, b1, x1)
                if code != 0:
                    return False, f"smoke_phase_{ph}_err_{code}"
            ref1 = np.linalg.solve(A_dense, b1)
            if float(np.linalg.norm(x1 - ref1) / max(1.0, float(np.linalg.norm(ref1)))) > 1e-11:
                return False, "smoke_solve_mismatch"

            # 验证半边前代 phase=331 满足 ||y||_2^2 == b^T A^-1 b（用于精确候选杠杆计算）
            B = np.asfortranarray(
                np.array(
                    [
                        [1.0, 0.0, -1.5],
                        [0.0, 2.0, 0.5],
                        [-0.5, 1.0, 0.0],
                        [2.0, -1.0, 1.0],
                        [0.0, 0.5, -2.0],
                    ],
                    dtype=np.float64,
                )
            )
            Y = np.asfortranarray(np.zeros_like(B))
            code = _call(331, B.shape[1], B, Y)
            if code != 0:
                return False, f"smoke_phase_331_err_{code}"
            lev_331 = np.einsum("ij,ij->j", Y, Y)
            lev_ref = np.einsum("ij,ij->j", B, np.linalg.solve(A_dense, B))
            max_rel = float(np.max(np.abs(lev_331 - lev_ref) / np.maximum(1.0, np.abs(lev_ref))))
            if not np.all(np.isfinite(lev_331)) or max_rel > 1e-11:
                return False, f"smoke_331_lev_mismatch:{max_rel:.2e}"
        finally:
            _call(-1, 1, b1, x1)
        return True, "ok"
    except Exception as exc:
        return False, f"smoke_exception:{type(exc).__name__}"


def probe_mkl_pardiso(*, force_reprobe: bool = False) -> dict:
    """探测并初始化 MKL PARDISO 后端（进程级缓存，线程安全回退）。"""
    global _MKL_PROBE_DONE, _MKL_LIB, _MKL_PARDISO_FN, _MKL_INFO
    pref = get_linear_solver_preference()
    if pref in ("cholmod", "scipy"):
        return {
            **_MKL_INFO,
            "available": False,
            "reason": f"disabled_by_env:{pref}",
        }
    if _MKL_PROBE_DONE and not force_reprobe:
        return dict(_MKL_INFO)

    arch = _detect_cpu_arch()
    vendor = _detect_cpu_vendor()
    info = {
        "available": False,
        "reason": "unavailable",
        "library_path": None,
        "mkl_major": 0,
        "mkl_minor": 0,
        "cpu_vendor": vendor,
        "arch": arch,
        "threads": 0,
    }
    if not _is_x86_64_arch(arch):
        info["reason"] = f"unsupported_arch:{arch}"
        _MKL_INFO = info
        _MKL_PROBE_DONE = True
        return dict(info)

    lib, lib_path = _load_mkl_rt_library()
    if lib is None:
        info["reason"] = "mkl_rt_not_found"
        _MKL_INFO = info
        _MKL_PROBE_DONE = True
        return dict(info)

    major, minor = _query_mkl_version(lib)
    info["library_path"] = lib_path
    info["mkl_major"] = major
    info["mkl_minor"] = minor

    allowed, reason = _should_allow_mkl_on_cpu(
        arch=arch,
        cpu_vendor=vendor,
        mkl_major=major,
        preference=pref,
    )
    if not allowed:
        info["reason"] = reason
        _MKL_INFO = info
        _MKL_PROBE_DONE = True
        return dict(info)

    n_threads = _resolve_mkl_num_threads()
    _configure_mkl_threads(lib, n_threads)
    pardiso_fn = lib.pardiso
    ok, smoke_reason = _smoke_test_pardiso(pardiso_fn)
    if not ok:
        info["reason"] = smoke_reason
        _MKL_INFO = info
        _MKL_PROBE_DONE = True
        return dict(info)

    info["available"] = True
    info["reason"] = "ok"
    info["threads"] = n_threads
    _MKL_LIB = lib
    _MKL_PARDISO_FN = pardiso_fn
    _MKL_INFO = info
    _MKL_PROBE_DONE = True
    return dict(info)


def is_mkl_pardiso_available() -> bool:
    """判断当前环境与偏好设置下是否允许并可用 MKL PARDISO。"""
    pref = get_linear_solver_preference()
    if pref in ("cholmod", "scipy"):
        return False
    return bool(probe_mkl_pardiso().get("available", False))


def _upper_csr_from_symmetric_csr(
    G_csr: SP_CSR_MATRIX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从对称 CSR 矩阵向量化提取 0-based 上三角 CSR（ia, ja, a, upper_mask, sup_keys），免走 COO 重排。"""
    n = int(G_csr.shape[0])
    if not G_csr.has_sorted_indices:
        G_csr.sort_indices()
    indptr = G_csr.indptr
    indices = G_csr.indices
    data = G_csr.data
    row_nnz = np.diff(indptr)
    row_idx = np.repeat(np.arange(n, dtype=indices.dtype), row_nnz)
    upper_mask = indices >= row_idx
    u_rows = row_idx[upper_mask]
    u_cols = indices[upper_mask]
    ja = np.ascontiguousarray(u_cols, dtype=np.int32)
    a = np.ascontiguousarray(data[upper_mask], dtype=np.float64)
    sup_keys = u_rows.astype(np.int64) * np.int64(n) + u_cols.astype(np.int64)
    ia = np.empty(n + 1, dtype=np.int32)
    ia[0] = 0
    if upper_mask.size:
        # 每行上三角非零数通过 cumsum 差分一次得到
        csum = np.empty(upper_mask.size + 1, dtype=np.int64)
        csum[0] = 0
        np.cumsum(upper_mask, out=csum[1:])
        ia[1:] = (csum[indptr[1:]] - csum[indptr[:-1]]).cumsum(dtype=np.int32)
    else:
        ia[1:] = 0
    return ia, ja, a, upper_mask, sup_keys


def build_upper_csr_from_aat(
    A_csc,
    *,
    damping: float = 0.0,
    cached_pattern: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """构造 ``G = A A^T (+ beta I)`` 的 0-based 上三角 CSR；支持同模式快速抽取及删行子集复用超集符号模式。"""
    beta = float(damping)
    A_csc_use = A_csc if getattr(A_csc, "format", None) == "csc" else A_csc.tocsc()

    # 超快路径 0（OpenMP 原生 C++ 内核）：当 A_csc 稀疏拓扑与初始超集一致（覆盖全部同轮 WLS 迭代
    # 及 masked-weight 删行），直接跳过 A.tocsr()、A@A.T 与 searchsorted，按固定超集 (ia, ja) 并行装配
    if (
        NATIVE_ASSEMBLE_AAT_UPPER_CSR is not None
        and cached_pattern is not None
        and cached_pattern.get("a_shape") == A_csc_use.shape
        and cached_pattern.get("a_nnz") == int(A_csc_use.nnz)
        and (beta == 0.0 or bool(cached_pattern.get("has_full_diag", False)))
        and memoryview(cached_pattern["a_indptr"]).tobytes() == memoryview(A_csc_use.indptr).tobytes()
        and memoryview(cached_pattern["a_indices"]).tobytes() == memoryview(A_csc_use.indices).tobytes()
    ):
        a_native = NATIVE_ASSEMBLE_AAT_UPPER_CSR(
            A_csc_use,
            cached_pattern["ia"],
            cached_pattern["ja"],
            damping=beta,
        )
        if a_native is not None:
            return cached_pattern["ia"], cached_pattern["ja"], a_native, cached_pattern

    A_csr = A_csc_use.tocsr()
    A_t_csc = A_csc_use.T
    G_csr = A_csr @ A_t_csc
    if getattr(G_csr, "format", None) != "csr":
        G_csr = G_csr.tocsr()
    if beta > 0.0:
        G_csr = (G_csr + SP_EYE(G_csr.shape[0], format="csr", dtype=np.float64) * beta).tocsr()
    if not G_csr.has_sorted_indices:
        G_csr.sort_indices()

    if cached_pattern is not None and cached_pattern.get("shape") == G_csr.shape:
        # 快速路径 1：稀疏拓扑完全一致（同轮 WLS 迭代），零拷贝抽取上三角数值
        if (
            cached_pattern.get("nnz") == int(G_csr.nnz)
            and memoryview(cached_pattern["g_indptr"]).tobytes() == memoryview(G_csr.indptr).tobytes()
            and memoryview(cached_pattern["g_indices"]).tobytes() == memoryview(G_csr.indices).tobytes()
        ):
            a = np.ascontiguousarray(G_csr.data[cached_pattern["upper_mask"]], dtype=np.float64)
            return cached_pattern["ia"], cached_pattern["ja"], a, cached_pattern

        # 快速路径 1B：命中最近一次删行子集映射（同轮多次 WLS 迭代免重复 searchsorted）
        last_sub = cached_pattern.get("last_sub")
        if (
            last_sub is not None
            and last_sub["nnz"] == int(G_csr.nnz)
            and memoryview(last_sub["g_indptr"]).tobytes() == memoryview(G_csr.indptr).tobytes()
            and memoryview(last_sub["g_indices"]).tobytes() == memoryview(G_csr.indices).tobytes()
        ):
            a_sup = np.zeros(cached_pattern["sup_keys"].size, dtype=np.float64)
            a_sup[last_sub["pos"]] = G_csr.data[last_sub["u_mask"]]
            return cached_pattern["ia"], cached_pattern["ja"], a_sup, cached_pattern

        # 快速路径 2：删行后的稀疏子集映射回初始超集模式（缺失位填 0.0），彻底免除重复 phase=11 符号分析
        sup_keys = cached_pattern.get("sup_keys")
        if sup_keys is not None and int(G_csr.nnz) <= int(cached_pattern.get("nnz", 0)):
            n = int(G_csr.shape[0])
            row_nnz = np.diff(G_csr.indptr)
            row_idx = np.repeat(np.arange(n, dtype=G_csr.indices.dtype), row_nnz)
            u_mask = G_csr.indices >= row_idx
            sub_keys = row_idx[u_mask].astype(np.int64) * np.int64(n) + G_csr.indices[u_mask].astype(np.int64)
            if sub_keys.size == 0:
                a_sup = np.zeros(sup_keys.size, dtype=np.float64)
                return cached_pattern["ia"], cached_pattern["ja"], a_sup, cached_pattern
            pos = np.searchsorted(sup_keys, sub_keys)
            if int(pos[-1]) < sup_keys.size and np.array_equal(sup_keys[pos], sub_keys):
                cached_pattern["last_sub"] = {
                    "nnz": int(G_csr.nnz),
                    "g_indptr": G_csr.indptr.copy(),
                    "g_indices": G_csr.indices.copy(),
                    "u_mask": u_mask,
                    "pos": pos,
                }
                a_sup = np.zeros(sup_keys.size, dtype=np.float64)
                a_sup[pos] = G_csr.data[u_mask]
                return cached_pattern["ia"], cached_pattern["ja"], a_sup, cached_pattern

    ia, ja, a, upper_mask, sup_keys = _upper_csr_from_symmetric_csr(G_csr)
    n_rows = int(G_csr.shape[0])
    # 检查上三角每行首项是否包含全部对角元 (i, i)（用于 damping > 0 安全判定）
    has_full_diag = bool(
        n_rows == 0
        or (np.all(ia[1:] > ia[:-1]) and np.array_equal(ja[ia[:-1]], np.arange(n_rows, dtype=np.int32)))
    )
    new_pattern = {
        "shape": G_csr.shape,
        "nnz": int(G_csr.nnz),
        "g_indptr": G_csr.indptr.copy(),
        "g_indices": G_csr.indices.copy(),
        "upper_mask": upper_mask,
        "sup_keys": sup_keys,
        "ia": ia,
        "ja": ja,
        "a_shape": A_csc_use.shape,
        "a_nnz": int(A_csc_use.nnz),
        "a_indptr": A_csc_use.indptr.copy(),
        "a_indices": A_csc_use.indices.copy(),
        "has_full_diag": has_full_diag,
    }
    return ia, ja, a, new_pattern


class MKLPardisoFactor:
    """封装 Intel oneMKL PARDISO 实对称正定（``mtype=2``）持久因子，兼容 CHOLMOD 因子调用协议。"""

    is_mkl_pardiso_factor: bool = True

    def __init__(
        self,
        n: int,
        ia: np.ndarray,
        ja: np.ndarray,
        *,
        parallel_metis: bool = True,
        upper_pattern: Optional[dict] = None,
    ):
        if not is_mkl_pardiso_available() or _MKL_PARDISO_FN is None:
            raise RuntimeError("MKL PARDISO is not available")
        self._pardiso = _MKL_PARDISO_FN
        self._lib = _MKL_LIB
        self._n = int(n)
        self._parallel_metis = bool(parallel_metis)
        self._ia = np.ascontiguousarray(ia, dtype=np.int32)
        self._ja = np.ascontiguousarray(ja, dtype=np.int32)
        self._upper_pattern = upper_pattern
        self._a: Optional[np.ndarray] = None
        self._pt = np.zeros(64, dtype=np.int64)
        self._iparm = np.zeros(64, dtype=np.int32)
        self._iparm[0] = 1
        # 大矩阵优先并行 OpenMP METIS (3)，小中矩阵串行 METIS (2) 更低开销
        self._iparm[1] = 3 if (parallel_metis and self._n >= 8000) else 2
        self._iparm[9] = 8
        self._iparm[17] = -1
        self._iparm[18] = 0
        self._iparm[26] = 0
        self._iparm[34] = 1  # 0-based CSR 索引
        self._perm = np.zeros(self._n, dtype=np.int32)
        self._maxfct = ctypes.c_int32(1)
        self._mnum = ctypes.c_int32(1)
        self._mtype = ctypes.c_int32(2)  # Real SPD
        self._n_c = ctypes.c_int32(self._n)
        self._msglvl = ctypes.c_int32(0)
        self._dummy = np.zeros(1, dtype=np.float64)
        self._closed = False
        self._factorized = False
        self._last_aat_operand = None
        self._last_damping: float = 0.0
        self._cholmod_materialized = None

        # 执行符号分析 phase=11
        err = self._call_pardiso(11, 1, self._dummy, self._dummy, a_arr=self._dummy)
        if err != 0:
            self._closed = True
            raise RuntimeError(f"MKL PARDISO symbolic analysis failed with error={err}")

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._n, self._n)

    def _call_pardiso(
        self,
        phase_val: int,
        nrhs: int,
        b_arr: np.ndarray,
        x_arr: np.ndarray,
        *,
        a_arr: Optional[np.ndarray] = None,
    ) -> int:
        if self._closed:
            raise RuntimeError("MKLPardisoFactor has already been closed")
        a_use = self._a if a_arr is None else a_arr
        if a_use is None:
            a_use = self._dummy
        ph = ctypes.c_int32(phase_val)
        nr = ctypes.c_int32(int(nrhs))
        err = ctypes.c_int32(0)
        self._pardiso(
            self._pt.ctypes.data_as(_C_VOID_P),
            ctypes.byref(self._maxfct),
            ctypes.byref(self._mnum),
            ctypes.byref(self._mtype),
            ctypes.byref(ph),
            ctypes.byref(self._n_c),
            a_use.ctypes.data_as(_C_DBL_P),
            self._ia.ctypes.data_as(_C_INT32_P),
            self._ja.ctypes.data_as(_C_INT32_P),
            self._perm.ctypes.data_as(_C_INT32_P),
            ctypes.byref(nr),
            self._iparm.ctypes.data_as(_C_INT32_P),
            ctypes.byref(self._msglvl),
            b_arr.ctypes.data_as(_C_DBL_P),
            x_arr.ctypes.data_as(_C_DBL_P),
            ctypes.byref(err),
        )
        return int(err.value)

    def factorize_inplace(self, a: np.ndarray, *, aat_operand=None, damping: float = 0.0) -> None:
        """复用已缓存的符号分析与置换，原位执行多线程数值 Cholesky 分解（``phase=22``）。"""
        a_c = np.ascontiguousarray(a, dtype=np.float64)
        self._a = a_c
        self._last_aat_operand = aat_operand
        self._last_damping = float(damping)
        self._cholmod_materialized = None
        err = self._call_pardiso(22, 1, self._dummy, self._dummy, a_arr=a_c)
        if err != 0:
            self._factorized = False
            raise RuntimeError(f"MKL PARDISO numerical factorization failed with error={err}")
        self._factorized = True

    def cholesky_AAt_inplace(self, A_csc, beta: float = 0.0) -> None:
        """兼容 CHOLMOD ``factor.cholesky_AAt_inplace(A, beta=...)`` 协议，供冻结因子终轮原位刷新复用。"""
        prev_pattern = self._upper_pattern
        ia, ja, a, new_pattern = build_upper_csr_from_aat(
            A_csc,
            damping=float(beta),
            cached_pattern=prev_pattern,
        )
        same_struct = (
            (new_pattern is prev_pattern)
            or (
                ia.shape == self._ia.shape
                and ja.shape == self._ja.shape
                and memoryview(ia).tobytes() == memoryview(self._ia).tobytes()
                and memoryview(ja).tobytes() == memoryview(self._ja).tobytes()
            )
        )
        if not same_struct:
            # 稀疏结构变化时原位重建 PARDISO 符号分析上下文
            if not self._closed:
                try:
                    self._call_pardiso(-1, 1, self._dummy, self._dummy, a_arr=self._dummy)
                except Exception:
                    pass
            self._n = int(A_csc.shape[0])
            self._ia = np.ascontiguousarray(ia, dtype=np.int32)
            self._ja = np.ascontiguousarray(ja, dtype=np.int32)
            self._pt = np.zeros(64, dtype=np.int64)
            self._iparm = np.zeros(64, dtype=np.int32)
            self._iparm[0] = 1
            self._iparm[1] = 3 if (self._parallel_metis and self._n >= 8000) else 2
            self._iparm[9] = 8
            self._iparm[17] = -1
            self._iparm[18] = 0
            self._iparm[26] = 0
            self._iparm[34] = 1
            self._perm = np.zeros(self._n, dtype=np.int32)
            self._n_c = ctypes.c_int32(self._n)
            self._closed = False
            err = self._call_pardiso(11, 1, self._dummy, self._dummy, a_arr=self._dummy)
            if err != 0:
                self._closed = True
                raise RuntimeError(f"MKL PARDISO symbolic re-analysis failed with error={err}")
        self._upper_pattern = new_pattern
        self.factorize_inplace(a, aat_operand=A_csc, damping=float(beta))

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """执行多线程稀疏前代+回代（``phase=33``），支持 1D ``(n,)`` 或 2D ``(n, k)`` 右端。"""
        if not self._factorized:
            raise RuntimeError("MKLPardisoFactor is not factorized")
        rhs_arr = np.asarray(rhs, dtype=np.float64)
        if rhs_arr.ndim == 1:
            if rhs_arr.shape[0] != self._n:
                raise ValueError(f"Expected rhs length {self._n}, got {rhs_arr.shape[0]}")
            b_c = np.ascontiguousarray(rhs_arr, dtype=np.float64)
            x_c = np.empty(self._n, dtype=np.float64)
            err = self._call_pardiso(33, 1, b_c, x_c)
            if err != 0:
                raise RuntimeError(f"MKL PARDISO solve failed with error={err}")
            return x_c
        if rhs_arr.ndim == 2:
            if rhs_arr.shape[0] != self._n:
                raise ValueError(f"Expected rhs rows {self._n}, got {rhs_arr.shape[0]}")
            k = int(rhs_arr.shape[1])
            if k == 0:
                return np.empty((self._n, 0), dtype=np.float64)
            b_f = np.asfortranarray(rhs_arr, dtype=np.float64)
            x_f = np.empty((self._n, k), dtype=np.float64, order="F")
            err = self._call_pardiso(33, k, b_f, x_f)
            if err != 0:
                raise RuntimeError(f"MKL PARDISO multi-rhs solve failed with error={err}")
            return x_f
        raise ValueError(f"Unsupported rhs ndim={rhs_arr.ndim}")

    def solve_L_unpermuted(self, rhs: np.ndarray) -> np.ndarray:
        """执行多线程半边前代（``phase=331``：求解 ``L y = P b``），供 ``||y||_2^2 = b^T G^-1 b`` 快速杠杆计算。"""
        if not self._factorized:
            raise RuntimeError("MKLPardisoFactor is not factorized")
        rhs_arr = np.asarray(rhs, dtype=np.float64)
        if rhs_arr.ndim == 1:
            b_c = np.ascontiguousarray(rhs_arr, dtype=np.float64)
            y_c = np.empty(self._n, dtype=np.float64)
            err = self._call_pardiso(331, 1, b_c, y_c)
            if err != 0:
                raise RuntimeError(f"MKL PARDISO phase=331 failed with error={err}")
            return y_c
        if rhs_arr.ndim == 2:
            k = int(rhs_arr.shape[1])
            if k == 0:
                return np.empty((self._n, 0), dtype=np.float64)
            b_f = np.asfortranarray(rhs_arr, dtype=np.float64)
            y_f = np.empty((self._n, k), dtype=np.float64, order="F")
            err = self._call_pardiso(331, k, b_f, y_f)
            if err != 0:
                raise RuntimeError(f"MKL PARDISO phase=331 multi-rhs failed with error={err}")
            return y_f
        raise ValueError(f"Unsupported rhs ndim={rhs_arr.ndim}")

    def __call__(self, rhs: np.ndarray) -> np.ndarray:
        return self.solve(rhs)

    def materialize_cholmod_factor(self, *, ordering_method: str = "nesdis"):
        """仅在需要整网 Takahashi 选元求逆时，按需将当前数值系统物化为 CHOLMOD 因子以提取 ``L``。"""
        if self._cholmod_materialized is not None:
            return self._cholmod_materialized
        if CHOLMOD_ANALYZE is None and CHOLMOD_CHOLESKY is None:
            return None
        try:
            # 优先直接利用缓存的上三角 CSR 还原对称 CSC 矩阵（比 AAt 快 ~24%）
            if self._a is not None:
                Gu = SP_CSR_MATRIX(
                    (self._a, self._ja.astype(np.intp, copy=False), self._ia.astype(np.intp, copy=False)),
                    shape=(self._n, self._n),
                )
                # 对称化：G = Gu + Gu.T - diag(Gu)
                diag_v = Gu.diagonal()
                G_full = (Gu + Gu.T).tocsc()
                if diag_v.size:
                    G_full.setdiag(diag_v)
                if CHOLMOD_ANALYZE is not None:
                    try:
                        fac = CHOLMOD_ANALYZE(G_full, ordering_method=ordering_method)
                    except Exception:
                        fac = CHOLMOD_ANALYZE(G_full)
                    fac.cholesky_inplace(G_full)
                else:
                    fac = CHOLMOD_CHOLESKY(G_full)
                self._cholmod_materialized = fac
                return fac
            if self._last_aat_operand is not None and CHOLMOD_ANALYZE_AAT is not None:
                try:
                    fac = CHOLMOD_ANALYZE_AAT(self._last_aat_operand, ordering_method=ordering_method)
                except Exception:
                    fac = CHOLMOD_ANALYZE_AAT(self._last_aat_operand)
                if self._last_damping > 0.0:
                    fac.cholesky_AAt_inplace(self._last_aat_operand, beta=self._last_damping)
                else:
                    fac.cholesky_AAt_inplace(self._last_aat_operand)
                self._cholmod_materialized = fac
                return fac
        except Exception:
            return None
        return None

    def close(self) -> None:
        if not self._closed:
            try:
                self._call_pardiso(-1, 1, self._dummy, self._dummy, a_arr=self._dummy)
            except Exception:
                pass
            self._closed = True
            self._factorized = False
            self._cholmod_materialized = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
