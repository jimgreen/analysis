"""状态估计量测预过滤核心引擎。

在正式状态估计（可观测性分析、伪量测补全及 WLS 迭代）之前，统一对原始量测执行向量化预过滤：
1. 原始量测状态异常（status != 0）一律置为不可用，不参与估计；
2. 死岛/停运设备量测（特别是电压/功率/电流/流量明显不为 0）置为不可用并精准标记原因；
3. 状态正常但数值非有限（NaN/Inf）或明显越过用户可配置物理上下限的极端异常量测置为不可用。

所有数值上下限与零容差均直接从 ``StateEstimationParameters`` 输入参数动态读取，不在本模块写死阈值。
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from model.meas_model import (
    MEAS_STATUS_INVALID,
    MEAS_STATUS_NORMAL,
    MeasurementTable,
    measurement_table_status_code,
)
from model.meas_type import (
    DEVICE_TYPE_DCACConverter,
    DEVICE_TYPE_DCBranch,
    DEVICE_TYPE_DCBreak,
    DEVICE_TYPE_DCBreakConstraint,
    DEVICE_TYPE_DCDCConverter,
    DEVICE_TYPE_DCGenerator,
    DEVICE_TYPE_DCLoad,
    DEVICE_TYPE_DCNode,
    DEVICE_TYPE_DCSwitch,
    DEVICE_TYPE_DCSwitchConstraint,
    DEVICE_TYPE_DCZeroBranch,
    DEVICE_TYPE_DCZeroBranchConstraint,
    MEAS_TYPE_CODES,
    MEAS_TYPE_I_AC,
    MEAS_TYPE_I_DC,
    MEAS_TYPE_I_FROM,
    MEAS_TYPE_I_GEN,
    MEAS_TYPE_I_LOAD,
    MEAS_TYPE_I_THIRD,
    MEAS_TYPE_I_TO,
    MEAS_TYPE_P_AC,
    MEAS_TYPE_P_DC,
    MEAS_TYPE_P_FROM,
    MEAS_TYPE_P_GEN,
    MEAS_TYPE_P_IN,
    MEAS_TYPE_P_LOAD,
    MEAS_TYPE_P_OUT,
    MEAS_TYPE_P_THIRD,
    MEAS_TYPE_P_TO,
    MEAS_TYPE_Q_AC,
    MEAS_TYPE_Q_FROM,
    MEAS_TYPE_Q_GEN,
    MEAS_TYPE_Q_LOAD,
    MEAS_TYPE_Q_THIRD,
    MEAS_TYPE_Q_TO,
    MEAS_TYPE_V,
    MEAS_TYPE_V_AC,
    MEAS_TYPE_V_DC,
    MEAS_TYPE_V_FROM,
    MEAS_TYPE_V_GEN,
    MEAS_TYPE_V_LOAD,
    MEAS_TYPE_V_THIRD,
    MEAS_TYPE_V_TO,
)


# 预过滤标准原因常量
REASON_ABNORMAL_STATUS = "abnormal_status"
REASON_NONFINITE_VALUE = "nonfinite_value"
REASON_ZERO_WEIGHT = "zero weight"
REASON_INVALID = "invalid"
REASON_DEAD_ISLAND_NONZERO = "dead_island_nonzero"
REASON_DEAD_ISLAND = "dead_island"
REASON_VOLTAGE_OUT_OF_RANGE = "voltage_out_of_range"
REASON_POWER_OUT_OF_RANGE = "power_out_of_range"
REASON_CURRENT_OUT_OF_RANGE = "current_out_of_range"
REASON_PRESSURE_OUT_OF_RANGE = "pressure_out_of_range"
REASON_FLOW_OUT_OF_RANGE = "flow_out_of_range"
REASON_TEMPERATURE_OUT_OF_RANGE = "temperature_out_of_range"
REASON_ENTHALPY_OUT_OF_RANGE = "enthalpy_out_of_range"
REASON_HEAT_OUT_OF_RANGE = "heat_out_of_range"


_ELECTRIC_VOLTAGE_CODES = np.asarray(
    (
        MEAS_TYPE_V,
        MEAS_TYPE_V_FROM,
        MEAS_TYPE_V_TO,
        MEAS_TYPE_V_THIRD,
        MEAS_TYPE_V_GEN,
        MEAS_TYPE_V_LOAD,
        MEAS_TYPE_V_AC,
        MEAS_TYPE_V_DC,
    ),
    dtype=np.int16,
)

_ELECTRIC_POWER_CODES = np.asarray(
    (
        MEAS_TYPE_P_FROM,
        MEAS_TYPE_Q_FROM,
        MEAS_TYPE_P_TO,
        MEAS_TYPE_Q_TO,
        MEAS_TYPE_P_THIRD,
        MEAS_TYPE_Q_THIRD,
        MEAS_TYPE_P_GEN,
        MEAS_TYPE_Q_GEN,
        MEAS_TYPE_P_LOAD,
        MEAS_TYPE_Q_LOAD,
        MEAS_TYPE_P_AC,
        MEAS_TYPE_Q_AC,
        MEAS_TYPE_P_DC,
        MEAS_TYPE_P_IN,
        MEAS_TYPE_P_OUT,
    ),
    dtype=np.int16,
)

_ELECTRIC_CURRENT_CODES = np.asarray(
    (
        MEAS_TYPE_I_FROM,
        MEAS_TYPE_I_TO,
        MEAS_TYPE_I_THIRD,
        MEAS_TYPE_I_GEN,
        MEAS_TYPE_I_LOAD,
        MEAS_TYPE_I_AC,
        MEAS_TYPE_I_DC,
    ),
    dtype=np.int16,
)

_DC_DEVICE_CODES = np.asarray(
    (
        DEVICE_TYPE_DCNode,
        DEVICE_TYPE_DCBranch,
        DEVICE_TYPE_DCBreak,
        DEVICE_TYPE_DCZeroBranch,
        DEVICE_TYPE_DCSwitch,
        DEVICE_TYPE_DCGenerator,
        DEVICE_TYPE_DCLoad,
        DEVICE_TYPE_DCDCConverter,
        DEVICE_TYPE_DCZeroBranchConstraint,
        DEVICE_TYPE_DCBreakConstraint,
        DEVICE_TYPE_DCSwitchConstraint,
    ),
    dtype=np.int16,
)

_FLUID_PRESSURE_NAMES = frozenset(("PRESSURE", "PRESSURE_FROM", "PRESSURE_TO", "P_FROM", "P_TO"))
_FLUID_FLOW_NAMES = frozenset(("FLOW", "FLOW_FROM", "FLOW_TO"))
_FLUID_TEMP_NAMES = frozenset(
    ("TEMPERATURE", "T_SUPPLY", "T_RETURN", "T_FROM", "T_TO", "TEMP_SUPPLY", "TEMP_RETURN", "TEMP_FROM", "TEMP_TO")
)
_FLUID_ENTHALPY_NAMES = frozenset(("ENTHALPY", "H_FROM", "H_TO"))
_FLUID_HEAT_NAMES = frozenset(("HEAT",))


def _codes_for_names(names: Iterable[str]) -> np.ndarray:
    codes = [int(MEAS_TYPE_CODES[nm]) for nm in names if nm in MEAS_TYPE_CODES]
    return np.asarray(sorted(set(codes)), dtype=np.int16)


_FLUID_PRESSURE_CODES = _codes_for_names(_FLUID_PRESSURE_NAMES)
_FLUID_FLOW_CODES = _codes_for_names(_FLUID_FLOW_NAMES)
_FLUID_TEMP_CODES = _codes_for_names(_FLUID_TEMP_NAMES)
_FLUID_ENTHALPY_CODES = _codes_for_names(_FLUID_ENTHALPY_NAMES)
_FLUID_HEAT_CODES = _codes_for_names(_FLUID_HEAT_NAMES)


def _opt_finite_limit(params: object, attr: str) -> Optional[float]:
    """从参数对象读取可选数值上下限；未设置、为 None 或非有限值时返回 None（表示不限）。"""
    if params is None:
        return None
    raw = getattr(params, attr, None)
    if raw is None:
        return None
    val = float(raw)
    if not math.isfinite(val):
        return None
    return val


def _check_bounds_violation(
    values: np.ndarray,
    lower: Optional[float],
    upper: Optional[float],
) -> np.ndarray:
    """检查数值是否越过用户配置的闭区间上下限 [lower, upper]；任一侧为 None 时不设限。"""
    violated = np.zeros(values.shape, dtype=bool)
    if lower is not None:
        violated |= values < lower
    if upper is not None:
        violated |= values > upper
    return violated


def ensure_prefilter_reason_array(table: MeasurementTable) -> np.ndarray:
    """获取或初始化与 MeasurementTable 等长的逐行预过滤原因数组。"""
    n_rows = int(np.asarray(table.idx).size)
    existing = getattr(table, "prefilter_reason", None)
    if isinstance(existing, np.ndarray) and int(existing.size) == n_rows:
        return existing
    reasons = np.full(n_rows, "", dtype=object)
    setattr(table, "prefilter_reason", reasons)
    return reasons


def set_prefilter_reasons(
    reasons: np.ndarray,
    mask: np.ndarray,
    reason_text: str,
    *,
    overwrite: bool = False,
) -> None:
    """将指定掩码对应的预过滤原因写入原因数组（默认保留已有先触发原因）。"""
    if mask is None or not np.any(mask):
        return
    if overwrite:
        target = mask
    else:
        target = mask & (reasons == "")
    if np.any(target):
        reasons[target] = str(reason_text)


def resolve_table_device_names_for_rows(table: MeasurementTable, rows: np.ndarray) -> Sequence[str]:
    """按需解析 MeasurementTable 指定行的设备名称，兼容压缩词典列与对象字符串列。"""
    rows = np.asarray(rows, dtype=np.intp)
    if rows.size == 0:
        return ()
    dev_name_arr = getattr(table, "device_name", None)
    if dev_name_arr is not None and int(np.asarray(dev_name_arr).size) > int(rows.max()):
        return [str(x) for x in np.asarray(dev_name_arr)[rows].tolist()]
    dn_id = getattr(table, "device_name_id", None)
    dn_vocab = getattr(table, "device_names", None)
    if (
        dn_id is not None
        and dn_vocab is not None
        and int(np.asarray(dn_id).size) > int(rows.max())
        and int(np.asarray(dn_vocab).size) > 0
    ):
        vocab = [str(x) for x in np.asarray(dn_vocab, dtype=object).tolist()]
        n_vocab = len(vocab)
        ids = np.asarray(dn_id, dtype=np.int64)[rows].tolist()
        return [vocab[i] if 0 <= i < n_vocab else "" for i in ids]
    return [""] * int(rows.size)


def match_dead_island_rows(
    table: MeasurementTable,
    candidate_rows: np.ndarray,
    device_type_code: np.ndarray,
    dead_device_names_by_type: Optional[Mapping[int, Set[str]]],
    dead_device_scales_by_type: Optional[Mapping[int, Mapping[str, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """针对候选不可用行按需匹配死岛/停运设备集合，返回匹配掩码与对应设备标幺基准尺度。"""
    rows = np.asarray(candidate_rows, dtype=np.int64)
    n_cand = int(rows.size)
    matched = np.zeros(n_cand, dtype=bool)
    scales = np.ones(n_cand, dtype=np.float64)
    if n_cand == 0 or not dead_device_names_by_type:
        return matched, scales
    names = resolve_table_device_names_for_rows(table, rows)
    codes = np.asarray(device_type_code, dtype=np.int16)[rows.astype(np.intp, copy=False)].tolist()
    for pos, (code, name) in enumerate(zip(codes, names)):
        if not name:
            continue
        dead_set = dead_device_names_by_type.get(int(code))
        if dead_set and name in dead_set:
            matched[pos] = True
            if dead_device_scales_by_type:
                scale_map = dead_device_scales_by_type.get(int(code))
                if scale_map:
                    sc = scale_map.get(name)
                    if sc is not None and math.isfinite(float(sc)) and abs(float(sc)) > 1e-12:
                        scales[pos] = abs(float(sc))
    return matched, scales


def prefilter_electric_measurements(
    table: MeasurementTable,
    params: object,
    *,
    available_mask: Optional[np.ndarray] = None,
    raw_values: Optional[np.ndarray] = None,
    dead_device_names_by_type: Optional[Mapping[int, Set[str]]] = None,
    dead_device_scales_by_type: Optional[Mapping[int, Mapping[str, float]]] = None,
    dead_row_mask: Optional[np.ndarray] = None,
    dead_power_scale: float = 1.0,
    side: str = "ac",
    row_subset_mask: Optional[np.ndarray] = None,
    reasons: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """对电侧（AC / DC / Hybrid）标幺化后量测表执行统一预过滤。

    返回 ``(invalidated_mask, reasons)``：
    - ``invalidated_mask`` 为本次预过滤新置为不可用的布尔掩码；
    - ``reasons`` 为整张表对应的逐行预过滤原因字符串数组，并挂接至 ``table.prefilter_reason``。
    """
    value = np.asarray(table.value, dtype=np.float64)
    valid = np.asarray(table.valid, dtype=bool)
    weight = np.asarray(table.weight, dtype=np.float64)
    status = measurement_table_status_code(table)
    n_rows = int(value.size)

    if reasons is None:
        reasons = ensure_prefilter_reason_array(table)
    else:
        setattr(table, "prefilter_reason", reasons)

    invalidated = np.zeros(n_rows, dtype=bool)
    if n_rows == 0:
        return invalidated, reasons

    scope = np.ones(n_rows, dtype=bool) if row_subset_mask is None else np.asarray(row_subset_mask, dtype=bool)
    dt_code = np.asarray(getattr(table, "device_type_code", np.zeros(n_rows, dtype=np.int16)), dtype=np.int16)
    mt_code = np.asarray(getattr(table, "meas_type_code", np.zeros(n_rows, dtype=np.int16)), dtype=np.int16)
    raw_val_arr = value if raw_values is None else np.asarray(raw_values, dtype=np.float64)

    # 1. 规则一：原始量测状态本身不正常（status != 0）一律不参与估计
    non_normal_status = scope & (status != MEAS_STATUS_NORMAL)
    # 明确异常状态（status != 0 且（原 valid=True 或状态码不是默认 1/INVALID））记录为 abnormal_status
    explicit_abnormal = non_normal_status & (valid | (status != MEAS_STATUS_INVALID))
    if np.any(explicit_abnormal):
        set_prefilter_reasons(reasons, explicit_abnormal, REASON_ABNORMAL_STATUS)
    still_valid_abnormal = non_normal_status & valid
    if np.any(still_valid_abnormal):
        valid[still_valid_abnormal] = False
        invalidated[still_valid_abnormal] = True

    # 原始 valid=False 且未标注原因的量测保持 invalid
    raw_invalid_no_reason = scope & (~valid) & (reasons == "")
    if np.any(raw_invalid_no_reason):
        set_prefilter_reasons(reasons, raw_invalid_no_reason, REASON_INVALID)

    # 2. 权重非有限或 <= 0
    bad_weight_nonfinite = scope & valid & (~np.isfinite(weight))
    if np.any(bad_weight_nonfinite):
        valid[bad_weight_nonfinite] = False
        status[bad_weight_nonfinite] = MEAS_STATUS_INVALID
        invalidated[bad_weight_nonfinite] = True
        set_prefilter_reasons(reasons, bad_weight_nonfinite, REASON_NONFINITE_VALUE)

    zero_weight = scope & valid & np.isfinite(weight) & (weight <= 0.0)
    if np.any(zero_weight):
        set_prefilter_reasons(reasons, zero_weight, REASON_ZERO_WEIGHT)

    enable_pf = bool(getattr(params, "enable_measurement_prefilter", True)) if params is not None else True
    dead_tol = float(getattr(params, "prefilter_dead_island_zero_tol", 1e-3)) if params is not None else 1e-3
    if not math.isfinite(dead_tol) or dead_tol < 0.0:
        dead_tol = 0.0

    # 3. 规则三：死岛 / 停运设备检查（仅针对当前状态正常且有正权重的不可用行）
    active_candidate = scope & valid & (weight > 0.0)
    if available_mask is not None:
        avail = np.asarray(available_mask, dtype=bool)
        unavail_cand = active_candidate & (~avail)
    else:
        unavail_cand = np.zeros(n_rows, dtype=bool)

    if dead_row_mask is not None:
        unavail_cand = unavail_cand | (active_candidate & np.asarray(dead_row_mask, dtype=bool))

    if np.any(unavail_cand):
        cand_rows = np.flatnonzero(unavail_cand)
        dead_hit = np.zeros(cand_rows.size, dtype=bool)
        dead_scales = np.ones(cand_rows.size, dtype=np.float64)
        if dead_row_mask is not None:
            dead_hit |= np.asarray(dead_row_mask, dtype=bool)[cand_rows]
        if dead_device_names_by_type:
            name_hit, name_scales = match_dead_island_rows(
                table,
                cand_rows,
                dt_code,
                dead_device_names_by_type,
                dead_device_scales_by_type,
            )
            dead_hit |= name_hit
            dead_scales = np.where(name_hit, name_scales, dead_scales)

        if np.any(dead_hit):
            d_rows = cand_rows[dead_hit]
            d_mtypes = mt_code[d_rows]
            d_raw = raw_val_arr[d_rows]
            # 对死岛/停运设备：电压按设备基准电压归一，功率按 p_base 归一，其余按原始值检查是否明显不为 0
            d_scales = np.ones(d_rows.size, dtype=np.float64)
            is_v = np.isin(d_mtypes, _ELECTRIC_VOLTAGE_CODES)
            if np.any(is_v):
                d_scales[is_v] = dead_scales[dead_hit][is_v]
            is_p = np.isin(d_mtypes, _ELECTRIC_POWER_CODES)
            if np.any(is_p) and math.isfinite(float(dead_power_scale)) and abs(float(dead_power_scale)) > 1e-12:
                d_scales[is_p] = abs(float(dead_power_scale))
            d_norm = np.divide(d_raw, d_scales, out=d_raw.copy(), where=np.abs(d_scales) > 1e-12)
            # 电压/功率/电流若明显不为 0（或非有限）则标 dead_island_nonzero，否则标 dead_island
            nonzero_dead = (~np.isfinite(d_norm)) | (np.abs(d_norm) > dead_tol)
            if enable_pf and np.any(nonzero_dead):
                nz_mask = np.zeros(n_rows, dtype=bool)
                nz_mask[d_rows[nonzero_dead]] = True
                set_prefilter_reasons(reasons, nz_mask, REASON_DEAD_ISLAND_NONZERO)
            zero_dead_rows = d_rows[~nonzero_dead] if enable_pf else d_rows
            if zero_dead_rows.size:
                zd_mask = np.zeros(n_rows, dtype=bool)
                zd_mask[zero_dead_rows] = True
                set_prefilter_reasons(reasons, zd_mask, REASON_DEAD_ISLAND)

        valid[cand_rows] = False
        status[cand_rows] = MEAS_STATUS_INVALID
        invalidated[cand_rows] = True
        set_prefilter_reasons(reasons, unavail_cand, REASON_INVALID)

    # 4. 非有限量测值检查（始终生效，杜绝 NaN/Inf 污染求解矩阵）
    processable = scope & valid & (weight > 0.0)
    if not np.any(processable):
        return invalidated, reasons

    nonfinite = processable & (~np.isfinite(value))
    if np.any(nonfinite):
        valid[nonfinite] = False
        status[nonfinite] = MEAS_STATUS_INVALID
        invalidated[nonfinite] = True
        set_prefilter_reasons(reasons, nonfinite, REASON_NONFINITE_VALUE)
        processable = processable & (~nonfinite)

    if not enable_pf or not np.any(processable):
        return invalidated, reasons

    # 5. 规则二：物理上下限检查（全部限值从 params 动态读取，无硬编码）
    v_min = _opt_finite_limit(params, "prefilter_voltage_min_pu")
    v_max = _opt_finite_limit(params, "prefilter_voltage_max_pu")
    if v_min is not None or v_max is not None:
        v_mask = processable & np.isin(mt_code, _ELECTRIC_VOLTAGE_CODES)
        if np.any(v_mask):
            v_rows = np.flatnonzero(v_mask)
            v_bad = _check_bounds_violation(value[v_rows], v_min, v_max)
            if np.any(v_bad):
                bad_rows = v_rows[v_bad]
                valid[bad_rows] = False
                status[bad_rows] = MEAS_STATUS_INVALID
                invalidated[bad_rows] = True
                bad_mask = np.zeros(n_rows, dtype=bool)
                bad_mask[bad_rows] = True
                set_prefilter_reasons(reasons, bad_mask, REASON_VOLTAGE_OUT_OF_RANGE)
                processable[bad_rows] = False

    p_min = _opt_finite_limit(params, "prefilter_power_min_pu")
    p_max = _opt_finite_limit(params, "prefilter_power_max_pu")
    if p_min is not None or p_max is not None:
        p_mask = processable & np.isin(mt_code, _ELECTRIC_POWER_CODES)
        if np.any(p_mask):
            p_rows = np.flatnonzero(p_mask)
            p_bad = _check_bounds_violation(value[p_rows], p_min, p_max)
            if np.any(p_bad):
                bad_rows = p_rows[p_bad]
                valid[bad_rows] = False
                status[bad_rows] = MEAS_STATUS_INVALID
                invalidated[bad_rows] = True
                bad_mask = np.zeros(n_rows, dtype=bool)
                bad_mask[bad_rows] = True
                set_prefilter_reasons(reasons, bad_mask, REASON_POWER_OUT_OF_RANGE)
                processable[bad_rows] = False

    i_ac_min = _opt_finite_limit(params, "prefilter_ac_current_min_pu")
    i_dc_min = _opt_finite_limit(params, "prefilter_dc_current_min_pu")
    i_max = _opt_finite_limit(params, "prefilter_current_max_pu")
    if i_ac_min is not None or i_dc_min is not None or i_max is not None:
        i_mask = processable & np.isin(mt_code, _ELECTRIC_CURRENT_CODES)
        if np.any(i_mask):
            if side == "dc":
                dc_curr = i_mask
                ac_curr = np.zeros(n_rows, dtype=bool)
            elif side == "ac":
                ac_curr = i_mask
                dc_curr = np.zeros(n_rows, dtype=bool)
            else:
                is_dc_curr = (mt_code == MEAS_TYPE_I_DC) | (
                    np.isin(dt_code, _DC_DEVICE_CODES) & (mt_code != MEAS_TYPE_I_AC)
                )
                dc_curr = i_mask & is_dc_curr
                ac_curr = i_mask & (~is_dc_curr)

            if np.any(ac_curr) and (i_ac_min is not None or i_max is not None):
                rows_ac = np.flatnonzero(ac_curr)
                bad_ac = _check_bounds_violation(value[rows_ac], i_ac_min, i_max)
                if np.any(bad_ac):
                    bad_rows = rows_ac[bad_ac]
                    valid[bad_rows] = False
                    status[bad_rows] = MEAS_STATUS_INVALID
                    invalidated[bad_rows] = True
                    bad_mask = np.zeros(n_rows, dtype=bool)
                    bad_mask[bad_rows] = True
                    set_prefilter_reasons(reasons, bad_mask, REASON_CURRENT_OUT_OF_RANGE)
                    processable[bad_rows] = False

            if np.any(dc_curr) and (i_dc_min is not None or i_max is not None):
                rows_dc = np.flatnonzero(dc_curr)
                bad_dc = _check_bounds_violation(value[rows_dc], i_dc_min, i_max)
                if np.any(bad_dc):
                    bad_rows = rows_dc[bad_dc]
                    valid[bad_rows] = False
                    status[bad_rows] = MEAS_STATUS_INVALID
                    invalidated[bad_rows] = True
                    bad_mask = np.zeros(n_rows, dtype=bool)
                    bad_mask[bad_rows] = True
                    set_prefilter_reasons(reasons, bad_mask, REASON_CURRENT_OUT_OF_RANGE)
                    processable[bad_rows] = False

    return invalidated, reasons


def evaluate_fluid_measurement_prefilter(
    *,
    values: np.ndarray,
    weights: np.ndarray,
    raw_valid: np.ndarray,
    raw_status: np.ndarray,
    meas_type_code: np.ndarray,
    params: object,
    is_dead_or_inactive: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """对流体域（Gas/Hydro/Heat/Steam）量测数组执行向量化预过滤。

    返回 ``(keep_mask, reasons)``，其中 ``keep_mask`` 仅表示通过状态、死岛、非有限值及物理上下限检查的掩码，
    调用方应与网络归属、设备匹配、支持类型掩码相与。
    """
    vals = np.asarray(values, dtype=np.float64)
    wgts = np.asarray(weights, dtype=np.float64)
    vld = np.asarray(raw_valid, dtype=bool)
    st = np.asarray(raw_status, dtype=np.int16)
    mt = np.asarray(meas_type_code, dtype=np.int16)
    n_rows = int(vals.size)
    reasons = np.full(n_rows, "", dtype=object)
    keep = np.ones(n_rows, dtype=bool)
    if n_rows == 0:
        return keep, reasons

    enable_pf = bool(getattr(params, "enable_measurement_prefilter", True)) if params is not None else True
    dead_tol = float(getattr(params, "prefilter_dead_island_zero_tol", 1e-3)) if params is not None else 1e-3
    if not math.isfinite(dead_tol) or dead_tol < 0.0:
        dead_tol = 0.0

    # 1. 状态非 0 检查
    abnormal = st != MEAS_STATUS_NORMAL
    if np.any(abnormal):
        keep[abnormal] = False
        explicit_abn = abnormal & (vld | (st != MEAS_STATUS_INVALID))
        set_prefilter_reasons(reasons, explicit_abn, REASON_ABNORMAL_STATUS)
        set_prefilter_reasons(reasons, abnormal, REASON_INVALID)

    inv_mask = keep & (~vld)
    if np.any(inv_mask):
        keep[inv_mask] = False
        set_prefilter_reasons(reasons, inv_mask, REASON_INVALID)

    # 2. 权重检查
    nf_w = keep & (~np.isfinite(wgts))
    if np.any(nf_w):
        keep[nf_w] = False
        set_prefilter_reasons(reasons, nf_w, REASON_NONFINITE_VALUE)

    zw = keep & (wgts <= 0.0)
    if np.any(zw):
        keep[zw] = False
        set_prefilter_reasons(reasons, zw, REASON_ZERO_WEIGHT)

    # 3. 死岛 / 停运流体设备检查
    if is_dead_or_inactive is not None:
        dead_m = keep & np.asarray(is_dead_or_inactive, dtype=bool)
        if np.any(dead_m):
            keep[dead_m] = False
            if enable_pf:
                nz = dead_m & ((~np.isfinite(vals)) | (np.abs(vals) > dead_tol))
                set_prefilter_reasons(reasons, nz, REASON_DEAD_ISLAND_NONZERO)
            set_prefilter_reasons(reasons, dead_m, REASON_DEAD_ISLAND)

    # 4. 非有限量测值检查
    nf_v = keep & (~np.isfinite(vals))
    if np.any(nf_v):
        keep[nf_v] = False
        set_prefilter_reasons(reasons, nf_v, REASON_NONFINITE_VALUE)

    if not enable_pf or not np.any(keep):
        return keep, reasons

    # 5. 流体物理上下限检查（全部从 params 动态读取，统一按闭区间 [min, max] 判定）
    checks = (
        (_FLUID_PRESSURE_CODES, "prefilter_pressure_min", "prefilter_pressure_max", REASON_PRESSURE_OUT_OF_RANGE),
        (_FLUID_FLOW_CODES, "prefilter_flow_min", "prefilter_flow_max", REASON_FLOW_OUT_OF_RANGE),
        (_FLUID_TEMP_CODES, "prefilter_temperature_min", "prefilter_temperature_max", REASON_TEMPERATURE_OUT_OF_RANGE),
        (_FLUID_ENTHALPY_CODES, "prefilter_enthalpy_min", "prefilter_enthalpy_max", REASON_ENTHALPY_OUT_OF_RANGE),
        (_FLUID_HEAT_CODES, "prefilter_heat_min", "prefilter_heat_max", REASON_HEAT_OUT_OF_RANGE),
    )
    for code_set, min_attr, max_attr, reason_tag in checks:
        if code_set.size == 0:
            continue
        lo = _opt_finite_limit(params, min_attr)
        hi = _opt_finite_limit(params, max_attr)
        if lo is None and hi is None:
            continue
        group_mask = keep & np.isin(mt, code_set)
        if not np.any(group_mask):
            continue
        rows = np.flatnonzero(group_mask)
        bad = _check_bounds_violation(vals[rows], lo, hi)
        if np.any(bad):
            bad_rows = rows[bad]
            keep[bad_rows] = False
            bad_mask = np.zeros(n_rows, dtype=bool)
            bad_mask[bad_rows] = True
            set_prefilter_reasons(reasons, bad_mask, reason_tag)

    return keep, reasons


def check_single_fluid_measurement_prefilter(
    *,
    value: float,
    weight: float,
    valid: bool,
    status: int,
    meas_type: str,
    params: object,
    is_dead_or_inactive: bool = False,
) -> str:
    """对象回退路径下的单条流体量测预过滤判定，返回空串表示通过，否则返回具体原因。"""
    st = int(status)
    if st != MEAS_STATUS_NORMAL:
        if bool(valid) or st != MEAS_STATUS_INVALID:
            return REASON_ABNORMAL_STATUS
        return REASON_INVALID
    if not bool(valid):
        return REASON_INVALID
    w = float(weight)
    if not math.isfinite(w):
        return REASON_NONFINITE_VALUE
    if w <= 0.0:
        return REASON_ZERO_WEIGHT

    enable_pf = bool(getattr(params, "enable_measurement_prefilter", True)) if params is not None else True
    dead_tol = float(getattr(params, "prefilter_dead_island_zero_tol", 1e-3)) if params is not None else 1e-3
    if not math.isfinite(dead_tol) or dead_tol < 0.0:
        dead_tol = 0.0

    val = float(value)
    if is_dead_or_inactive:
        if enable_pf and ((not math.isfinite(val)) or abs(val) > dead_tol):
            return REASON_DEAD_ISLAND_NONZERO
        return REASON_DEAD_ISLAND
    if not math.isfinite(val):
        return REASON_NONFINITE_VALUE
    if not enable_pf:
        return ""

    mt = str(meas_type).upper()
    if mt in _FLUID_PRESSURE_NAMES:
        lo = _opt_finite_limit(params, "prefilter_pressure_min")
        hi = _opt_finite_limit(params, "prefilter_pressure_max")
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            return REASON_PRESSURE_OUT_OF_RANGE
    elif mt in _FLUID_FLOW_NAMES:
        lo = _opt_finite_limit(params, "prefilter_flow_min")
        hi = _opt_finite_limit(params, "prefilter_flow_max")
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            return REASON_FLOW_OUT_OF_RANGE
    elif mt in _FLUID_TEMP_NAMES:
        lo = _opt_finite_limit(params, "prefilter_temperature_min")
        hi = _opt_finite_limit(params, "prefilter_temperature_max")
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            return REASON_TEMPERATURE_OUT_OF_RANGE
    elif mt in _FLUID_ENTHALPY_NAMES:
        lo = _opt_finite_limit(params, "prefilter_enthalpy_min")
        hi = _opt_finite_limit(params, "prefilter_enthalpy_max")
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            return REASON_ENTHALPY_OUT_OF_RANGE
    elif mt in _FLUID_HEAT_NAMES:
        lo = _opt_finite_limit(params, "prefilter_heat_min")
        hi = _opt_finite_limit(params, "prefilter_heat_max")
        if (lo is not None and val < lo) or (hi is not None and val > hi):
            return REASON_HEAT_OUT_OF_RANGE
    return ""


def tag_dead_island_unavailable_rows(
    table: MeasurementTable,
    unavailable_rows: np.ndarray,
    params: object,
    *,
    dead_device_names_by_type: Optional[Mapping[int, Set[str]]],
    dead_device_scales_by_type: Optional[Mapping[int, Mapping[str, float]]] = None,
    dead_power_scale: float = 1.0,
) -> np.ndarray:
    """对即将置为不可用的设备行按需识别死岛/停运设备，并标注 dead_island_nonzero / dead_island。"""
    reasons = ensure_prefilter_reason_array(table)
    rows = np.asarray(unavailable_rows, dtype=np.int64)
    if rows.size == 0 or not dead_device_names_by_type:
        return reasons
    n_rows = int(np.asarray(table.idx).size)
    dt_code = np.asarray(getattr(table, "device_type_code", np.zeros(n_rows, dtype=np.int16)), dtype=np.int16)
    mt_code = np.asarray(getattr(table, "meas_type_code", np.zeros(n_rows, dtype=np.int16)), dtype=np.int16)
    status = measurement_table_status_code(table)
    valid = np.asarray(table.valid, dtype=bool)
    raw_vals = np.asarray(table.value, dtype=np.float64)

    # 仅对当前状态正常且原本有效的不可用行做死岛/停运异常归类
    normal_sel = valid[rows] & (status[rows] == MEAS_STATUS_NORMAL)
    if not np.any(normal_sel):
        return reasons
    cand_rows = rows[normal_sel]
    dead_hit, dead_scales = match_dead_island_rows(
        table,
        cand_rows,
        dt_code,
        dead_device_names_by_type,
        dead_device_scales_by_type,
    )
    if not np.any(dead_hit):
        return reasons

    enable_pf = bool(getattr(params, "enable_measurement_prefilter", True)) if params is not None else True
    dead_tol = float(getattr(params, "prefilter_dead_island_zero_tol", 1e-3)) if params is not None else 1e-3
    if not math.isfinite(dead_tol) or dead_tol < 0.0:
        dead_tol = 0.0

    d_rows = cand_rows[dead_hit]
    d_mtypes = mt_code[d_rows]
    d_raw = raw_vals[d_rows]
    d_scales = np.ones(d_rows.size, dtype=np.float64)
    is_v = np.isin(d_mtypes, _ELECTRIC_VOLTAGE_CODES)
    if np.any(is_v):
        d_scales[is_v] = dead_scales[dead_hit][is_v]
    is_p = np.isin(d_mtypes, _ELECTRIC_POWER_CODES)
    if np.any(is_p) and math.isfinite(float(dead_power_scale)) and abs(float(dead_power_scale)) > 1e-12:
        d_scales[is_p] = abs(float(dead_power_scale))
    d_norm = np.divide(d_raw, d_scales, out=d_raw.copy(), where=np.abs(d_scales) > 1e-12)
    nonzero_dead = (~np.isfinite(d_norm)) | (np.abs(d_norm) > dead_tol)

    if enable_pf and np.any(nonzero_dead):
        nz_mask = np.zeros(n_rows, dtype=bool)
        nz_mask[d_rows[nonzero_dead]] = True
        set_prefilter_reasons(reasons, nz_mask, REASON_DEAD_ISLAND_NONZERO)
    zero_dead_rows = d_rows[~nonzero_dead] if enable_pf else d_rows
    if zero_dead_rows.size:
        zd_mask = np.zeros(n_rows, dtype=bool)
        zd_mask[zero_dead_rows] = True
        set_prefilter_reasons(reasons, zd_mask, REASON_DEAD_ISLAND)
    return reasons
