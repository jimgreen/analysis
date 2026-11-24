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


def build_csc_transpose_map_native(A_csc):
    """一次性构建 ``A_csc`` 的静态行主序转置索引映射 ``(r_ptr, r_col, r_src, col_sorted)``。"""
    if not is_native_kernels_available() or _NATIVE_MOD is None:
        return None
    fn = getattr(_NATIVE_MOD, "build_csc_transpose_map", None)
    if fn is None:
        return None
    try:
        n = int(A_csc.shape[0])
        nnz = int(A_csc.nnz)
        ap = np.ascontiguousarray(A_csc.indptr, dtype=np.int64)
        ai = np.ascontiguousarray(A_csc.indices, dtype=np.int32)
        r_ptr = np.empty(n + 1, dtype=np.int64)
        r_col = np.empty(nnz, dtype=np.int32)
        r_src = np.empty(nnz, dtype=np.int64)
        col_sorted = bool(fn(n, ap, ai, r_ptr, r_col, r_src))
        return r_ptr, r_col, r_src, col_sorted
    except Exception:
        return None


def assemble_aat_upper_csr_native(
    A_csc,
    ia: np.ndarray,
    ja: np.ndarray,
    *,
    damping: float = 0.0,
    transpose_map=None,
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
        cached_fn = getattr(_NATIVE_MOD, "assemble_aat_upper_csr_cached", None)
        if transpose_map is not None and cached_fn is not None:
            r_ptr, r_col, r_src, col_sorted = transpose_map
            ok = bool(
                cached_fn(
                    n,
                    ap,
                    ai,
                    ad,
                    r_ptr,
                    r_col,
                    r_src,
                    ia_c,
                    ja_c,
                    a_out,
                    float(damping),
                    1 if col_sorted else 0,
                )
            )
            if ok:
                return a_out
        ok = bool(_NATIVE_MOD.assemble_aat_upper_csr(n, ap, ai, ad, ia_c, ja_c, a_out, float(damping)))
        if not ok:
            return None
        return a_out
    except Exception:
        return None


def complex_voltage_native(theta: np.ndarray, voltage: np.ndarray) -> Optional[np.ndarray]:
    """调用 OpenMP 并行原生核计算复电压 ``voltage * exp(1j * theta)``。"""
    if not is_native_kernels_available() or _NATIVE_MOD is None:
        return None
    fn = getattr(_NATIVE_MOD, "complex_voltage", None)
    if fn is None:
        return None
    try:
        th = np.ascontiguousarray(theta, dtype=np.float64)
        v = np.ascontiguousarray(voltage, dtype=np.float64)
        out = np.empty(th.shape[0], dtype=np.complex128)
        if fn(th, v, out):
            return out
        return None
    except Exception:
        return None


def eval_ac_branch_power_values_native(
    values: np.ndarray,
    rows: np.ndarray,
    own: np.ndarray,
    other: np.ndarray,
    third: Optional[np.ndarray],
    is_p: np.ndarray,
    y_self: np.ndarray,
    y_mutual: np.ndarray,
    y_third: Optional[np.ndarray],
    voltage_complex: np.ndarray,
    has_third: bool,
) -> bool:
    """调用 OpenMP 并行原生核计算 AC 支路/变压器 P/Q 量测预测值并直接写入 ``values[rows]``。"""
    if not is_native_kernels_available() or _NATIVE_MOD is None:
        return False
    fn = getattr(_NATIVE_MOD, "eval_ac_branch_power_values", None)
    if fn is None:
        return False
    try:
        return bool(
            fn(
                values,
                np.ascontiguousarray(rows, dtype=np.int64),
                np.ascontiguousarray(own, dtype=np.int64),
                np.ascontiguousarray(other, dtype=np.int64),
                np.ascontiguousarray(third, dtype=np.int64) if has_third and third is not None else None,
                np.ascontiguousarray(is_p, dtype=np.bool_),
                np.ascontiguousarray(y_self, dtype=np.complex128),
                np.ascontiguousarray(y_mutual, dtype=np.complex128),
                np.ascontiguousarray(y_third, dtype=np.complex128) if has_third and y_third is not None else None,
                np.ascontiguousarray(voltage_complex, dtype=np.complex128),
                1 if has_third else 0,
            )
        )
    except Exception:
        return False


def fill_ac_branch_power_jacobian_native(
    own: np.ndarray,
    other: np.ndarray,
    third: Optional[np.ndarray],
    is_p: np.ndarray,
    y_self_conj: np.ndarray,
    y_mutual_conj: np.ndarray,
    y_third_conj: Optional[np.ndarray],
    voltage: np.ndarray,
    voltage_complex: np.ndarray,
    has_third: bool,
    channels,
) -> bool:
    """调用 OpenMP 并行原生核计算 AC 支路/变压器 P/Q 雅可比并按 ``channels=[(dst, slots), ...]`` 写入。"""
    if not is_native_kernels_available() or _NATIVE_MOD is None:
        return False
    fn = getattr(_NATIVE_MOD, "fill_ac_branch_power_jacobian", None)
    if fn is None:
        return False
    try:
        (d0, s0), (d1, s1), (d2, s2), (d3, s3), (d4, s4), (d5, s5) = channels
        return bool(
            fn(
                np.ascontiguousarray(own, dtype=np.int64),
                np.ascontiguousarray(other, dtype=np.int64),
                np.ascontiguousarray(third, dtype=np.int64) if has_third and third is not None else None,
                np.ascontiguousarray(is_p, dtype=np.bool_),
                np.ascontiguousarray(y_self_conj, dtype=np.complex128),
                np.ascontiguousarray(y_mutual_conj, dtype=np.complex128),
                np.ascontiguousarray(y_third_conj, dtype=np.complex128) if has_third and y_third_conj is not None else None,
                np.ascontiguousarray(voltage, dtype=np.float64),
                np.ascontiguousarray(voltage_complex, dtype=np.complex128),
                1 if has_third else 0,
                d0, s0, d1, s1, d2, s2, d3, s3, d4, s4, d5, s5,
            )
        )
    except Exception:
        return False


def fill_ac_balance_jacobian_native(
    balance_pos: np.ndarray,
    y_bal_ptr: np.ndarray,
    y_nodes: np.ndarray,
    y_pos: np.ndarray,
    y_conj: np.ndarray,
    y_off_idx: np.ndarray,
    y_diag_conj: np.ndarray,
    voltage: np.ndarray,
    voltage_complex: np.ndarray,
    n_off: int,
    channels,
) -> bool:
    """调用 OpenMP 并行原生核单遍计算 AC 节点功率平衡雅可比（合并 off_sum/sum_all 归约）并按 ``channels`` 写入。"""
    if not is_native_kernels_available() or _NATIVE_MOD is None:
        return False
    fn = getattr(_NATIVE_MOD, "fill_ac_balance_jacobian", None)
    if fn is None:
        return False
    try:
        (do0, so0), (do1, so1), (do2, so2), (do3, so3), (dw0, sw0), (dw1, sw1), (dw2, sw2), (dw3, sw3) = channels
        return bool(
            fn(
                np.ascontiguousarray(balance_pos, dtype=np.int64),
                np.ascontiguousarray(y_bal_ptr, dtype=np.int64),
                np.ascontiguousarray(y_nodes, dtype=np.int64),
                np.ascontiguousarray(y_pos, dtype=np.int64),
                np.ascontiguousarray(y_conj, dtype=np.complex128),
                np.ascontiguousarray(y_off_idx, dtype=np.int32),
                np.ascontiguousarray(y_diag_conj, dtype=np.complex128),
                np.ascontiguousarray(voltage, dtype=np.float64),
                np.ascontiguousarray(voltage_complex, dtype=np.complex128),
                int(n_off),
                do0, so0, do1, so1, do2, so2, do3, so3,
                dw0, sw0, dw1, sw1, dw2, sw2, dw3, sw3,
            )
        )
    except Exception:
        return False


def parse_meas_bulk_bytes_native(
    data_bytes: bytes,
    ncols: int,
    idx_pos: int,
    dev_type_pos: int,
    dev_name_pos: int,
    meas_type_pos: int,
    weight_pos: int,
    valid_pos: int,
    value_pos: int,
    status_pos: int,
    status_normal: int,
    status_invalid: int,
    status_removed: int,
    dt_code_map: dict,
    mt_code_map: dict,
    status_code_map: dict,
    idx_array: np.ndarray,
    weight_values: np.ndarray,
    value_values: np.ndarray,
    valid_values: np.ndarray,
    status_values: np.ndarray,
    device_type_code_array: np.ndarray,
    meas_type_code_array: np.ndarray,
    device_name_id: np.ndarray,
) -> Optional[bytes]:
    """调用 C++17 单遍解析核高速解析标准无引号 ``<Measurement>`` 字节块，成功返回 ``\\n`` 分隔唯一设备名字节串。"""
    if not is_native_kernels_available() or _NATIVE_MOD is None:
        return None
    fn = getattr(_NATIVE_MOD, "parse_meas_bulk_bytes", None)
    if fn is None:
        return None
    try:
        res = fn(
            data_bytes,
            int(ncols),
            int(idx_pos),
            int(dev_type_pos),
            int(dev_name_pos),
            int(meas_type_pos),
            int(weight_pos),
            int(valid_pos),
            int(value_pos),
            int(status_pos),
            int(status_normal),
            int(status_invalid),
            int(status_removed),
            dt_code_map,
            mt_code_map,
            status_code_map,
            idx_array,
            weight_values,
            value_values,
            valid_values,
            status_values,
            device_type_code_array,
            meas_type_code_array,
            device_name_id,
        )
        return res if isinstance(res, bytes) else None
    except Exception:
        return None
