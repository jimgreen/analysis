"""跨平台原生微内核（``secore._native``）薄封装与自动回退加载层。

设计目标：
- 若已编译 ``secore._native``（Windows ``.pyd`` / Linux ``.so``）且环境变量 ``SE_NATIVE_KERNELS``
  不为 ``off``，自动绑定 SciPy 内置 BLAS/LAPACK 函数指针并启用高速 C++17/OpenMP 内核；
- 若部署环境未编译扩展或显式关闭，100% 静默回退至现有纯 Python/NumPy 实现，绝不中断运行。
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

_NATIVE_MOD = None
_NATIVE_READY = False
_NATIVE_INIT_DONE = False


def get_native_kernels_preference() -> str:
    """读取环境变量 ``SE_NATIVE_KERNELS``，支持 ``auto`` / ``on`` / ``off``。"""
    raw = str(os.environ.get("SE_NATIVE_KERNELS", "auto") or "auto").strip().lower()
    if raw in ("auto", "on", "off", "1", "0", "true", "false"):
        if raw in ("on", "1", "true"):
            return "on"
        if raw in ("off", "0", "false"):
            return "off"
    return "auto"


def _init_native_module() -> bool:
    global _NATIVE_MOD, _NATIVE_READY, _NATIVE_INIT_DONE
    if _NATIVE_INIT_DONE:
        return _NATIVE_READY
    _NATIVE_INIT_DONE = True
    try:
        from scipy.linalg import cython_blas as cb
        from scipy.linalg import cython_lapack as cl

        from . import _native as native_mod

        cap_dgemm = cb.__pyx_capi__["dgemm"]
        cap_dsyrk = cb.__pyx_capi__["dsyrk"]
        cap_dtrsm = cb.__pyx_capi__["dtrsm"]
        cap_dtrtri = cl.__pyx_capi__["dtrtri"]
        ok = bool(native_mod.register_blas_capsules(cap_dgemm, cap_dsyrk, cap_dtrsm, cap_dtrtri))
        if ok:
            _NATIVE_MOD = native_mod
            _NATIVE_READY = True
    except Exception:
        _NATIVE_MOD = None
        _NATIVE_READY = False
    return _NATIVE_READY


def is_native_kernels_available() -> bool:
    """判断当前环境与环境变量配置下是否启用原生 C++17/OpenMP 微内核。"""
    if get_native_kernels_preference() == "off":
        return False
    return _init_native_module()


def supernodal_selected_inverse_native(L_csc) -> Optional[np.ndarray]:
    """调用 C++17 + BLAS-3 超节点 Takahashi 选元求逆，成功返回 ``zdata``，失败返回 ``None``。"""
    if not is_native_kernels_available() or _NATIVE_MOD is None:
        return None
    try:
        n = int(L_csc.shape[0])
        indptr = np.ascontiguousarray(L_csc.indptr)
        indices = np.ascontiguousarray(L_csc.indices)
        ldata = np.ascontiguousarray(L_csc.data, dtype=np.float64)
        zdata = np.empty_like(ldata)
        ok = bool(_NATIVE_MOD.supernodal_selected_inverse(n, indptr, indices, ldata, zdata))
        if not ok or not np.all(np.isfinite(zdata)):
            return None
        return zdata
    except Exception:
        return None


def leverage_from_selected_inv_native(
    H_csr,
    Z_low_csc,
    inv_P: np.ndarray,
) -> Optional[np.ndarray]:
    """调用 OpenMP 并行原生内核计算精确量测杠杆 ``diag(H G^-1 H^T)``；若检测到缺失项返回 ``None`` 走回退补齐。"""
    if not is_native_kernels_available() or _NATIVE_MOD is None:
        return None
    try:
        n_meas = int(H_csr.shape[0])
        if n_meas == 0:
            return np.empty(0, dtype=np.float64)
        h_indptr = np.ascontiguousarray(H_csr.indptr)
        h_indices = np.ascontiguousarray(H_csr.indices)
        h_data = np.ascontiguousarray(H_csr.data, dtype=np.float64)
        inv_p_arr = np.ascontiguousarray(inv_P, dtype=np.int32)
        z_indptr = np.ascontiguousarray(Z_low_csc.indptr)
        z_indices = np.ascontiguousarray(Z_low_csc.indices)
        z_data = np.ascontiguousarray(Z_low_csc.data, dtype=np.float64)
        lev = np.empty(n_meas, dtype=np.float64)
        missing = int(
            _NATIVE_MOD.leverage_from_selected_inv(
                h_indptr, h_indices, h_data, inv_p_arr, z_indptr, z_indices, z_data, lev
            )
        )
        if missing > 0 or not np.all(np.isfinite(lev)):
            return None
        return lev
    except Exception:
        return None


def assemble_aat_upper_csr_native(
    A_csc,
    ia: np.ndarray,
    ja: np.ndarray,
    *,
    damping: float = 0.0,
) -> Optional[np.ndarray]:
    """调用 OpenMP 并行原生内核按固定超集上三角模式 ``(ia, ja)`` 直接装配 ``G = A A^T (+ beta I)`` 数值数组。"""
    if not is_native_kernels_available() or _NATIVE_MOD is None:
        return None
    try:
        n = int(A_csc.shape[0])
        ap = np.ascontiguousarray(A_csc.indptr)
        ai = np.ascontiguousarray(A_csc.indices)
        ad = np.ascontiguousarray(A_csc.data, dtype=np.float64)
        ia_c = np.ascontiguousarray(ia, dtype=np.int32)
        ja_c = np.ascontiguousarray(ja, dtype=np.int32)
        a_out = np.empty(int(ja_c.size), dtype=np.float64)
        ok = bool(_NATIVE_MOD.assemble_aat_upper_csr(n, ap, ai, ad, ia_c, ja_c, a_out, float(damping)))
        if not ok:
            return None
        return a_out
    except Exception:
        return None
