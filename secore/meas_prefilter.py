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
    DEVICE_TYPE_ACACConverter,
    DEVICE_TYPE_ACBranch,
    DEVICE_TYPE_ACBreak,
    DEVICE_TYPE_ACGenerator,
    DEVICE_TYPE_ACLoad,
    DEVICE_TYPE_ACNode,
    DEVICE_TYPE_ACSwitch,
    DEVICE_TYPE_ACThreeWindingTransformer,
    DEVICE_TYPE_ACTransformer,
    DEVICE_TYPE_ACZeroBranch,
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
    MEAS_TYPE_ENTHALPY,
    MEAS_TYPE_FLOW,
    MEAS_TYPE_FLOW_FROM,
    MEAS_TYPE_FLOW_TO,
    MEAS_TYPE_H_FROM,
    MEAS_TYPE_H_TO,
    MEAS_TYPE_HEAT,
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
    MEAS_TYPE_PRESSURE,
    MEAS_TYPE_PRESSURE_FROM,
    MEAS_TYPE_PRESSURE_TO,
    MEAS_TYPE_Q_AC,
    MEAS_TYPE_Q_FROM,
    MEAS_TYPE_Q_GEN,
    MEAS_TYPE_Q_LOAD,
    MEAS_TYPE_Q_THIRD,
    MEAS_TYPE_Q_TO,
    MEAS_TYPE_T_FROM,
    MEAS_TYPE_T_RETURN,
    MEAS_TYPE_T_SUPPLY,
    MEAS_TYPE_T_TO,
    MEAS_TYPE_TEMPERATURE,
    MEAS_TYPE_TR_FROM,
    MEAS_TYPE_TR_TO,
    MEAS_TYPE_TS_FROM,
    MEAS_TYPE_TS_TO,
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
REASON_DEVICE_LIMIT_EXCEEDED = "device_limit_exceeded"


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


def resolve_device_limit_factor(params: object) -> Optional[float]:
    """解析设备状态限值预过滤系数；开关关闭或配置为 None / <=0 时返回 None（表示关闭）。"""
    if params is None:
        return 2.0
    if not bool(getattr(params, "enable_measurement_prefilter", True)):
        return None
    if not bool(getattr(params, "enable_device_limit_prefilter", True)):
        return None
    raw = getattr(params, "prefilter_device_limit_factor", 2.0)
    if raw is None:
        return None
    val = float(raw)
    if not math.isfinite(val) or val <= 0.0:
        return None
    return max(val, 1.0)


def check_device_bounds_violation(
    values: np.ndarray,
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
    factor: float,
    *,
    symmetric_upper: bool = False,
) -> np.ndarray:
    """向量化判定量测值是否大幅超出设备对应状态限值区间 [lower, upper]。

    规则（``k = max(factor, 1.0)``）：
    1. 双侧均有限且 ``lo <= hi``（跳过 ``lo == hi == 0`` 占位）：
       ``span = hi - lo``（若 ``span <= 1e-12`` 取 ``max(|lo|, |hi|)``），
       允许闭区间 ``[lo - (k - 1) * span, hi + (k - 1) * span]``；
    2. 仅上限有限且 ``|hi| > 1e-12``：
       - 若 ``symmetric_upper=True`` 且 ``hi > 0``（双向功率/直流电流/双向流量额定上限），
         允许 ``[-k * hi, k * hi]``；
       - 否则仅检查上界 ``val > (k * hi if hi > 0 else hi + (k - 1) * |hi|)``；
    3. 仅下限有限且 ``|lo| > 1e-12``：
       仅检查下界 ``val < (k * lo if lo < 0 else lo / k)``。
    """
    vals = np.asarray(values, dtype=np.float64)
    n = int(vals.size)
    violated = np.zeros(n, dtype=bool)
    if n == 0 or (lower is None and upper is None):
        return violated
    k = max(float(factor), 1.0)
    lo = np.full(n, np.nan, dtype=np.float64) if lower is None else np.asarray(lower, dtype=np.float64)
    hi = np.full(n, np.nan, dtype=np.float64) if upper is None else np.asarray(upper, dtype=np.float64)
    has_lo = np.isfinite(lo)
    has_hi = np.isfinite(hi)

    both = has_lo & has_hi & (lo <= hi + 1e-12)
    if np.any(both):
        lo_b = lo[both]
        hi_b = hi[both]
        val_b = vals[both]
        non_placeholder = (np.abs(lo_b) > 1e-12) | (np.abs(hi_b) > 1e-12)
        if np.any(non_placeholder):
            span = hi_b - lo_b
            degenerate = span <= 1e-12
            if np.any(degenerate):
                span = np.where(degenerate, np.maximum(np.abs(lo_b), np.abs(hi_b)), span)
            margin = (k - 1.0) * span
            lo_eff = lo_b - margin
            hi_eff = hi_b + margin
            bad_b = non_placeholder & ((val_b < lo_eff) | (val_b > hi_eff))
            if np.any(bad_b):
                violated[np.flatnonzero(both)[bad_b]] = True

    only_hi = (~has_lo) & has_hi & (np.abs(hi) > 1e-12)
    if np.any(only_hi):
        hi_o = hi[only_hi]
        val_o = vals[only_hi]
        hi_eff = np.where(hi_o > 0.0, k * hi_o, hi_o + (k - 1.0) * np.abs(hi_o))
        if symmetric_upper:
            pos_hi = hi_o > 0.0
            bad_o = np.where(pos_hi, (val_o < -k * hi_o) | (val_o > hi_eff), val_o > hi_eff)
        else:
            bad_o = val_o > hi_eff
        if np.any(bad_o):
            violated[np.flatnonzero(only_hi)[bad_o]] = True

    only_lo = has_lo & (~has_hi) & (np.abs(lo) > 1e-12)
    if np.any(only_lo):
        lo_o = lo[only_lo]
        val_o = vals[only_lo]
        lo_eff = np.where(lo_o < 0.0, k * lo_o, lo_o / k)
        bad_o = val_o < lo_eff
        if np.any(bad_o):
            violated[np.flatnonzero(only_lo)[bad_o]] = True

    return violated


def _apply_device_limits_lookup(
    *,
    values: np.ndarray,
    processable: np.ndarray,
    dt_code: np.ndarray,
    mt_code: np.ndarray,
    device_pos: Optional[np.ndarray],
    device_limits_lookup: Optional[Mapping[Tuple[int, int], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]]],
    factor: Optional[float],
) -> np.ndarray:
    """按 (device_type_code, meas_type_code) 查找表批量执行设备状态限值越限检查，返回越限布尔掩码。"""
    n_rows = int(values.size)
    exceeded = np.zeros(n_rows, dtype=bool)
    if (
        n_rows == 0
        or factor is None
        or not device_limits_lookup
        or device_pos is None
        or not np.any(processable)
    ):
        return exceeded
    dpos = np.asarray(device_pos, dtype=np.int64)
    if int(dpos.size) != n_rows:
        return exceeded
    base_mask = processable & (dpos >= 0)
    if not np.any(base_mask):
        return exceeded

    for (dt_k, mt_k), (lo_vec, hi_vec, sym_upper) in device_limits_lookup.items():
        if lo_vec is None and hi_vec is None:
            continue
        sel = base_mask & (dt_code == int(dt_k)) & (mt_code == int(mt_k))
        if not np.any(sel):
            continue
        rows = np.flatnonzero(sel)
        pos_sel = dpos[rows]
        lo_vals = None
        hi_vals = None
        if lo_vec is not None:
            lo_arr = np.asarray(lo_vec, dtype=np.float64)
            lo_vals = np.full(rows.size, np.nan, dtype=np.float64)
            ok_lo = pos_sel < int(lo_arr.size)
            if np.any(ok_lo):
                lo_vals[ok_lo] = lo_arr[pos_sel[ok_lo].astype(np.intp, copy=False)]
        if hi_vec is not None:
            hi_arr = np.asarray(hi_vec, dtype=np.float64)
            hi_vals = np.full(rows.size, np.nan, dtype=np.float64)
            ok_hi = pos_sel < int(hi_arr.size)
            if np.any(ok_hi):
                hi_vals[ok_hi] = hi_arr[pos_sel[ok_hi].astype(np.intp, copy=False)]
        bad = check_device_bounds_violation(
            values[rows],
            lo_vals,
            hi_vals,
            factor,
            symmetric_upper=bool(sym_upper),
        )
        if np.any(bad):
            exceeded[rows[bad]] = True
    return exceeded


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
    device_limits_lookup: Optional[Mapping[Tuple[int, int], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]]] = None,
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

    # 6. 规则六：设备状态限值超倍预过滤（超过设备对应上下限区间系数则置为设备越限排除）
    dev_limit_factor = resolve_device_limit_factor(params)
    if dev_limit_factor is not None and device_limits_lookup and np.any(processable):
        dev_exceeded = _apply_device_limits_lookup(
            values=value,
            processable=processable,
            dt_code=dt_code,
            mt_code=mt_code,
            device_pos=getattr(table, "device_pos", None),
            device_limits_lookup=device_limits_lookup,
            factor=dev_limit_factor,
        )
        if np.any(dev_exceeded):
            valid[dev_exceeded] = False
            status[dev_exceeded] = MEAS_STATUS_INVALID
            invalidated[dev_exceeded] = True
            set_prefilter_reasons(reasons, dev_exceeded, REASON_DEVICE_LIMIT_EXCEEDED)
            processable[dev_exceeded] = False

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
    device_type_code: Optional[np.ndarray] = None,
    device_pos: Optional[np.ndarray] = None,
    device_limits_lookup: Optional[Mapping[Tuple[int, int], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]]] = None,
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

    # 6. 流体设备状态限值超倍预过滤
    dev_limit_factor = resolve_device_limit_factor(params)
    if dev_limit_factor is not None and device_limits_lookup and device_type_code is not None and np.any(keep):
        dev_exceeded = _apply_device_limits_lookup(
            values=vals,
            processable=keep,
            dt_code=np.asarray(device_type_code, dtype=np.int16),
            mt_code=mt,
            device_pos=device_pos,
            device_limits_lookup=device_limits_lookup,
            factor=dev_limit_factor,
        )
        if np.any(dev_exceeded):
            keep[dev_exceeded] = False
            set_prefilter_reasons(reasons, dev_exceeded, REASON_DEVICE_LIMIT_EXCEEDED)

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
    device_limit_lo: Optional[float] = None,
    device_limit_hi: Optional[float] = None,
    symmetric_upper: bool = False,
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

    dev_limit_factor = resolve_device_limit_factor(params)
    if dev_limit_factor is not None and (device_limit_lo is not None or device_limit_hi is not None):
        lo_arr = np.asarray([np.nan if device_limit_lo is None else float(device_limit_lo)], dtype=np.float64)
        hi_arr = np.asarray([np.nan if device_limit_hi is None else float(device_limit_hi)], dtype=np.float64)
        if bool(
            check_device_bounds_violation(
                np.asarray([val], dtype=np.float64),
                lo_arr,
                hi_arr,
                dev_limit_factor,
                symmetric_upper=bool(symmetric_upper),
            )[0]
        ):
            return REASON_DEVICE_LIMIT_EXCEEDED
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


def _slice_limit_vec(
    lim_dict: Optional[Mapping[str, np.ndarray]],
    key: str,
    rows: np.ndarray,
) -> Optional[np.ndarray]:
    if not lim_dict or key not in lim_dict or rows.size == 0:
        return None
    src = np.asarray(lim_dict[key], dtype=np.float64)
    if src.size == 0:
        return None
    out = np.full(rows.size, np.nan, dtype=np.float64)
    ok = (rows >= 0) & (rows < src.size)
    if np.any(ok):
        out[ok] = src[rows[ok].astype(np.intp, copy=False)]
    return out if np.any(np.isfinite(out)) else None


def _coalesce_limit_vecs(*vecs: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """按优先级合并多个同长限值数组（前者非有限位置由后者回退补齐）。"""
    out: Optional[np.ndarray] = None
    for v in vecs:
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float64)
        if out is None:
            out = arr.copy()
        else:
            fill = (~np.isfinite(out)) & np.isfinite(arr)
            if np.any(fill):
                out[fill] = arr[fill]
    if out is not None and np.any(np.isfinite(out)):
        return out
    return None


def _node_limit_by_node_ids(
    bus_table: Optional[np.ndarray],
    bus_limits: Optional[Mapping[str, np.ndarray]],
    key: str,
    query_node_ids: np.ndarray,
) -> Optional[np.ndarray]:
    if bus_table is None or not bus_limits or key not in bus_limits or query_node_ids.size == 0:
        return None
    bus = np.asarray(bus_table, dtype=np.float64)
    src = np.asarray(bus_limits[key], dtype=np.float64)
    if bus.ndim != 2 or bus.shape[0] == 0 or src.size == 0 or not np.any(np.isfinite(src)):
        return None
    n_bus = min(bus.shape[0], src.size)
    ids = bus[:n_bus, 0].astype(np.int64, copy=False)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    q = np.asarray(query_node_ids, dtype=np.int64)
    loc = np.searchsorted(sorted_ids, q)
    ok = (loc >= 0) & (loc < sorted_ids.size)
    out = np.full(q.size, np.nan, dtype=np.float64)
    if np.any(ok):
        ok_idx = np.flatnonzero(ok)
        matched = sorted_ids[loc[ok_idx]] == q[ok_idx]
        if np.any(matched):
            m_idx = ok_idx[matched]
            out[m_idx] = src[order[loc[m_idx]]]
    return out if np.any(np.isfinite(out)) else None


def _register_lookup(
    lookup: Dict[Tuple[int, int], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]],
    dt_code: int,
    mt_codes: Sequence[int],
    lo: Optional[np.ndarray],
    hi: Optional[np.ndarray],
    *,
    symmetric_upper: bool = False,
) -> None:
    if lo is None and hi is None:
        return
    for mt in mt_codes:
        lookup[(int(dt_code), int(mt))] = (lo, hi, bool(symmetric_upper))


def build_ac_device_limits_lookup(
    ppc: Optional[Mapping[str, object]],
    plan_rows_by_type_code: Optional[Mapping[int, np.ndarray]],
) -> Dict[Tuple[int, int], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]]:
    """根据 AC PPC 与活网设备映射构建 (device_type_code, meas_type_code) -> (lo, hi, symmetric_upper) 查找表。"""
    if not isinstance(ppc, Mapping) or not plan_rows_by_type_code:
        return {}
    dev_limits = ppc.get("_device_limits")
    if not isinstance(dev_limits, Mapping) or not dev_limits:
        return {}
    lookup: Dict[Tuple[int, int], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]] = {}
    bus_table = ppc.get("bus")
    bus_lim = dev_limits.get("bus")

    node_rows = np.asarray(plan_rows_by_type_code.get(DEVICE_TYPE_ACNode, ()), dtype=np.int64)
    if node_rows.size and bus_lim:
        _register_lookup(
            lookup,
            DEVICE_TYPE_ACNode,
            (MEAS_TYPE_V,),
            _slice_limit_vec(bus_lim, "v_min", node_rows),
            _slice_limit_vec(bus_lim, "v_max", node_rows),
        )

    def _node_v_limits(table_key: str, rows: np.ndarray, node_col: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not bus_lim or rows.size == 0:
            return None, None
        arr = ppc.get(table_key)
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[0] == 0:
            return None, None
        ok = (rows >= 0) & (rows < arr.shape[0])
        if not np.any(ok):
            return None, None
        nids = np.full(rows.size, -1, dtype=np.int64)
        nids[ok] = arr[rows[ok].astype(np.intp, copy=False), node_col].astype(np.int64, copy=False)
        return (
            _node_limit_by_node_ids(bus_table, bus_lim, "v_min", nids),
            _node_limit_by_node_ids(bus_table, bus_lim, "v_max", nids),
        )

    # 双端元件：支路 / 变压器 / 零阻 / 开关 / 断路器 / ACAC 换流器
    two_term_specs = (
        (DEVICE_TYPE_ACBranch, "branch", 1, 2),
        (DEVICE_TYPE_ACTransformer, "transformer", 1, 2),
        (DEVICE_TYPE_ACZeroBranch, "zero_branch", 1, 2),
        (DEVICE_TYPE_ACSwitch, "switch", 1, 2),
        (DEVICE_TYPE_ACBreak, "break", 1, 2),
        (DEVICE_TYPE_ACACConverter, "acac", 1, 2),
    )
    for dt_code, tbl_key, i_col, j_col in two_term_specs:
        rows = np.asarray(plan_rows_by_type_code.get(dt_code, ()), dtype=np.int64)
        if rows.size == 0:
            continue
        lim = dev_limits.get(tbl_key)
        s_max = _slice_limit_vec(lim, "s_max", rows)
        p_lo = _slice_limit_vec(lim, "p_min", rows)
        p_hi = _coalesce_limit_vecs(_slice_limit_vec(lim, "p_max", rows), s_max)
        q_lo = _slice_limit_vec(lim, "q_min", rows)
        q_hi = _coalesce_limit_vecs(_slice_limit_vec(lim, "q_max", rows), s_max)
        i_lo = _slice_limit_vec(lim, "i_min", rows)
        i_hi = _slice_limit_vec(lim, "i_max", rows)
        _register_lookup(lookup, dt_code, (MEAS_TYPE_P_FROM, MEAS_TYPE_P_TO), p_lo, p_hi, symmetric_upper=True)
        _register_lookup(lookup, dt_code, (MEAS_TYPE_Q_FROM, MEAS_TYPE_Q_TO), q_lo, q_hi, symmetric_upper=True)
        _register_lookup(lookup, dt_code, (MEAS_TYPE_I_FROM, MEAS_TYPE_I_TO), i_lo, i_hi, symmetric_upper=False)
        dev_v_lo = _slice_limit_vec(lim, "v_min", rows)
        dev_v_hi = _slice_limit_vec(lim, "v_max", rows)
        ni_lo, ni_hi = _node_v_limits(tbl_key, rows, i_col)
        nj_lo, nj_hi = _node_v_limits(tbl_key, rows, j_col)
        _register_lookup(lookup, dt_code, (MEAS_TYPE_V_FROM,), _coalesce_limit_vecs(dev_v_lo, ni_lo), _coalesce_limit_vecs(dev_v_hi, ni_hi))
        _register_lookup(lookup, dt_code, (MEAS_TYPE_V_TO,), _coalesce_limit_vecs(dev_v_lo, nj_lo), _coalesce_limit_vecs(dev_v_hi, nj_hi))

    # 三绕组变压器
    rows3 = np.asarray(plan_rows_by_type_code.get(DEVICE_TYPE_ACThreeWindingTransformer, ()), dtype=np.int64)
    if rows3.size:
        lim3 = dev_limits.get("three_winding_transformer")
        s3 = _slice_limit_vec(lim3, "s_max", rows3)
        _register_lookup(
            lookup,
            DEVICE_TYPE_ACThreeWindingTransformer,
            (MEAS_TYPE_P_FROM, MEAS_TYPE_P_TO, MEAS_TYPE_P_THIRD),
            _slice_limit_vec(lim3, "p_min", rows3),
            _coalesce_limit_vecs(_slice_limit_vec(lim3, "p_max", rows3), s3),
            symmetric_upper=True,
        )
        _register_lookup(
            lookup,
            DEVICE_TYPE_ACThreeWindingTransformer,
            (MEAS_TYPE_Q_FROM, MEAS_TYPE_Q_TO, MEAS_TYPE_Q_THIRD),
            _slice_limit_vec(lim3, "q_min", rows3),
            _coalesce_limit_vecs(_slice_limit_vec(lim3, "q_max", rows3), s3),
            symmetric_upper=True,
        )
        _register_lookup(
            lookup,
            DEVICE_TYPE_ACThreeWindingTransformer,
            (MEAS_TYPE_I_FROM, MEAS_TYPE_I_TO, MEAS_TYPE_I_THIRD),
            _slice_limit_vec(lim3, "i_min", rows3),
            _slice_limit_vec(lim3, "i_max", rows3),
            symmetric_upper=False,
        )
        dv3_lo = _slice_limit_vec(lim3, "v_min", rows3)
        dv3_hi = _slice_limit_vec(lim3, "v_max", rows3)
        for mt_v, ncol in ((MEAS_TYPE_V_FROM, 1), (MEAS_TYPE_V_TO, 2), (MEAS_TYPE_V_THIRD, 3)):
            nv_lo, nv_hi = _node_v_limits("three_winding_transformer", rows3, ncol)
            _register_lookup(
                lookup,
                DEVICE_TYPE_ACThreeWindingTransformer,
                (mt_v,),
                _coalesce_limit_vecs(dv3_lo, nv_lo),
                _coalesce_limit_vecs(dv3_hi, nv_hi),
            )

    # 发电机与负荷
    for dt_code, tbl_key, p_mt, q_mt, v_mt, i_mt, sym_p in (
        (DEVICE_TYPE_ACGenerator, "gen", MEAS_TYPE_P_GEN, MEAS_TYPE_Q_GEN, MEAS_TYPE_V_GEN, MEAS_TYPE_I_GEN, False),
        (DEVICE_TYPE_ACLoad, "load", MEAS_TYPE_P_LOAD, MEAS_TYPE_Q_LOAD, MEAS_TYPE_V_LOAD, MEAS_TYPE_I_LOAD, True),
    ):
        rows = np.asarray(plan_rows_by_type_code.get(dt_code, ()), dtype=np.int64)
        if rows.size == 0:
            continue
        lim = dev_limits.get(tbl_key)
        s_max = _slice_limit_vec(lim, "s_max", rows)
        _register_lookup(
            lookup,
            dt_code,
            (p_mt,),
            _slice_limit_vec(lim, "p_min", rows),
            _coalesce_limit_vecs(_slice_limit_vec(lim, "p_max", rows), s_max),
            symmetric_upper=sym_p,
        )
        _register_lookup(
            lookup,
            dt_code,
            (q_mt,),
            _slice_limit_vec(lim, "q_min", rows),
            _coalesce_limit_vecs(_slice_limit_vec(lim, "q_max", rows), s_max),
            symmetric_upper=True,
        )
        _register_lookup(
            lookup,
            dt_code,
            (i_mt,),
            _slice_limit_vec(lim, "i_min", rows),
            _slice_limit_vec(lim, "i_max", rows),
            symmetric_upper=False,
        )
        nv_lo, nv_hi = _node_v_limits(tbl_key, rows, 1)
        _register_lookup(
            lookup,
            dt_code,
            (v_mt,),
            _coalesce_limit_vecs(_slice_limit_vec(lim, "v_min", rows), nv_lo),
            _coalesce_limit_vecs(_slice_limit_vec(lim, "v_max", rows), nv_hi),
        )

    return lookup


def build_dc_device_limits_lookup(
    ppc: Optional[Mapping[str, object]],
    plan_rows_by_type_code: Optional[Mapping[int, np.ndarray]],
) -> Dict[Tuple[int, int], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]]:
    """根据 DC PPC 与活网设备映射构建设备状态限值查找表。"""
    if not isinstance(ppc, Mapping) or not plan_rows_by_type_code:
        return {}
    dev_limits = ppc.get("_device_limits")
    if not isinstance(dev_limits, Mapping) or not dev_limits:
        return {}
    lookup: Dict[Tuple[int, int], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]] = {}
    bus_table = ppc.get("bus")
    bus_lim = dev_limits.get("bus")

    node_rows = np.asarray(plan_rows_by_type_code.get(DEVICE_TYPE_DCNode, ()), dtype=np.int64)
    if node_rows.size and bus_lim:
        _register_lookup(
            lookup,
            DEVICE_TYPE_DCNode,
            (MEAS_TYPE_V,),
            _slice_limit_vec(bus_lim, "v_min", node_rows),
            _slice_limit_vec(bus_lim, "v_max", node_rows),
        )

    def _node_v_limits(table_key: str, rows: np.ndarray, node_col: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not bus_lim or rows.size == 0:
            return None, None
        arr = ppc.get(table_key)
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[0] == 0:
            return None, None
        ok = (rows >= 0) & (rows < arr.shape[0])
        if not np.any(ok):
            return None, None
        nids = np.full(rows.size, -1, dtype=np.int64)
        nids[ok] = arr[rows[ok].astype(np.intp, copy=False), node_col].astype(np.int64, copy=False)
        return (
            _node_limit_by_node_ids(bus_table, bus_lim, "v_min", nids),
            _node_limit_by_node_ids(bus_table, bus_lim, "v_max", nids),
        )

    for dt_code, tbl_key in (
        (DEVICE_TYPE_DCBranch, "branch"),
        (DEVICE_TYPE_DCZeroBranch, "zero_branch"),
        (DEVICE_TYPE_DCSwitch, "switch"),
        (DEVICE_TYPE_DCBreak, "break"),
        (DEVICE_TYPE_DCDCConverter, "dcdc"),
    ):
        rows = np.asarray(plan_rows_by_type_code.get(dt_code, ()), dtype=np.int64)
        if rows.size == 0:
            continue
        lim = dev_limits.get(tbl_key)
        s_max = _slice_limit_vec(lim, "s_max", rows)
        _register_lookup(
            lookup,
            dt_code,
            (MEAS_TYPE_P_FROM, MEAS_TYPE_P_TO, MEAS_TYPE_P_IN, MEAS_TYPE_P_OUT),
            _slice_limit_vec(lim, "p_min", rows),
            _coalesce_limit_vecs(_slice_limit_vec(lim, "p_max", rows), s_max),
            symmetric_upper=True,
        )
        _register_lookup(
            lookup,
            dt_code,
            (MEAS_TYPE_I_FROM, MEAS_TYPE_I_TO),
            _slice_limit_vec(lim, "i_min", rows),
            _slice_limit_vec(lim, "i_max", rows),
            symmetric_upper=True,
        )
        dv_lo = _slice_limit_vec(lim, "v_min", rows)
        dv_hi = _slice_limit_vec(lim, "v_max", rows)
        ni_lo, ni_hi = _node_v_limits(tbl_key, rows, 1)
        nj_lo, nj_hi = _node_v_limits(tbl_key, rows, 2)
        _register_lookup(lookup, dt_code, (MEAS_TYPE_V_FROM,), _coalesce_limit_vecs(dv_lo, ni_lo), _coalesce_limit_vecs(dv_hi, ni_hi))
        _register_lookup(lookup, dt_code, (MEAS_TYPE_V_TO,), _coalesce_limit_vecs(dv_lo, nj_lo), _coalesce_limit_vecs(dv_hi, nj_hi))

    for dt_code, tbl_key, p_mt, v_mt, i_mt, sym_p in (
        (DEVICE_TYPE_DCGenerator, "gen", MEAS_TYPE_P_GEN, MEAS_TYPE_V_GEN, MEAS_TYPE_I_GEN, False),
        (DEVICE_TYPE_DCLoad, "load", MEAS_TYPE_P_LOAD, MEAS_TYPE_V_LOAD, MEAS_TYPE_I_LOAD, True),
    ):
        rows = np.asarray(plan_rows_by_type_code.get(dt_code, ()), dtype=np.int64)
        if rows.size == 0:
            continue
        lim = dev_limits.get(tbl_key)
        s_max = _slice_limit_vec(lim, "s_max", rows)
        _register_lookup(
            lookup,
            dt_code,
            (p_mt,),
            _slice_limit_vec(lim, "p_min", rows),
            _coalesce_limit_vecs(_slice_limit_vec(lim, "p_max", rows), s_max),
            symmetric_upper=sym_p,
        )
        _register_lookup(
            lookup,
            dt_code,
            (i_mt,),
            _slice_limit_vec(lim, "i_min", rows),
            _slice_limit_vec(lim, "i_max", rows),
            symmetric_upper=True,
        )
        nv_lo, nv_hi = _node_v_limits(tbl_key, rows, 1)
        _register_lookup(
            lookup,
            dt_code,
            (v_mt,),
            _coalesce_limit_vecs(_slice_limit_vec(lim, "v_min", rows), nv_lo),
            _coalesce_limit_vecs(_slice_limit_vec(lim, "v_max", rows), nv_hi),
        )

    return lookup


def build_hybrid_dcac_device_limits_lookup(
    hybrid_ppc: Optional[Mapping[str, object]],
    ac_ppc: Optional[Mapping[str, object]],
    dc_ppc: Optional[Mapping[str, object]],
    dcac_converters: Sequence[object],
) -> Dict[Tuple[int, int], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]]:
    """根据交直流混合 PPC 构建 DCACConverter 设备状态限值查找表（支持端口电压回退 AC/DC 节点限值）。"""
    if not dcac_converters:
        return {}
    h_dev_lim = (hybrid_ppc.get("_device_limits") if isinstance(hybrid_ppc, Mapping) else None) or {}
    dcac_lim = h_dev_lim.get("dcac") if isinstance(h_dev_lim, Mapping) else None
    ac_bus_lim = ((ac_ppc.get("_device_limits") if isinstance(ac_ppc, Mapping) else None) or {}).get("bus")
    dc_bus_lim = ((dc_ppc.get("_device_limits") if isinstance(dc_ppc, Mapping) else None) or {}).get("bus")
    if not dcac_lim and not ac_bus_lim and not dc_bus_lim:
        return {}

    n_conv = len(dcac_converters)
    rows = np.full(n_conv, -1, dtype=np.int64)
    dcac_arr = hybrid_ppc.get("dcac") if isinstance(hybrid_ppc, Mapping) else None
    if isinstance(dcac_arr, np.ndarray) and dcac_arr.ndim == 2 and dcac_arr.shape[0] > 0:
        idx_to_row = {int(round(dcac_arr[r, 0])): r for r in range(dcac_arr.shape[0])}
        for pos, conv in enumerate(dcac_converters):
            rows[pos] = idx_to_row.get(int(getattr(conv, "idx", -1)), -1)

    ac_nodes = np.asarray([int(getattr(conv, "ac_node", -1)) for conv in dcac_converters], dtype=np.int64)
    dc_nodes = np.asarray([int(getattr(conv, "dc_node", -1)) for conv in dcac_converters], dtype=np.int64)

    lookup: Dict[Tuple[int, int], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]] = {}
    s_max = _slice_limit_vec(dcac_lim, "s_max", rows)
    p_lo = _slice_limit_vec(dcac_lim, "p_min", rows)
    p_hi = _coalesce_limit_vecs(_slice_limit_vec(dcac_lim, "p_max", rows), s_max)
    _register_lookup(
        lookup,
        DEVICE_TYPE_DCACConverter,
        (MEAS_TYPE_P_AC,),
        _coalesce_limit_vecs(_slice_limit_vec(dcac_lim, "p_ac_min", rows), p_lo),
        _coalesce_limit_vecs(_slice_limit_vec(dcac_lim, "p_ac_max", rows), p_hi),
        symmetric_upper=True,
    )
    _register_lookup(
        lookup,
        DEVICE_TYPE_DCACConverter,
        (MEAS_TYPE_P_DC,),
        _coalesce_limit_vecs(_slice_limit_vec(dcac_lim, "p_dc_min", rows), p_lo),
        _coalesce_limit_vecs(_slice_limit_vec(dcac_lim, "p_dc_max", rows), p_hi),
        symmetric_upper=True,
    )
    _register_lookup(
        lookup,
        DEVICE_TYPE_DCACConverter,
        (MEAS_TYPE_Q_AC,),
        _coalesce_limit_vecs(_slice_limit_vec(dcac_lim, "q_ac_min", rows), _slice_limit_vec(dcac_lim, "q_min", rows)),
        _coalesce_limit_vecs(_slice_limit_vec(dcac_lim, "q_ac_max", rows), _slice_limit_vec(dcac_lim, "q_max", rows), s_max),
        symmetric_upper=True,
    )
    ac_bus_tbl = ac_ppc.get("bus") if isinstance(ac_ppc, Mapping) else None
    dc_bus_tbl = dc_ppc.get("bus") if isinstance(dc_ppc, Mapping) else None
    _register_lookup(
        lookup,
        DEVICE_TYPE_DCACConverter,
        (MEAS_TYPE_V_AC,),
        _coalesce_limit_vecs(
            _slice_limit_vec(dcac_lim, "v_ac_min", rows),
            _slice_limit_vec(dcac_lim, "v_min", rows),
            _node_limit_by_node_ids(ac_bus_tbl, ac_bus_lim, "v_min", ac_nodes),
        ),
        _coalesce_limit_vecs(
            _slice_limit_vec(dcac_lim, "v_ac_max", rows),
            _slice_limit_vec(dcac_lim, "v_max", rows),
            _node_limit_by_node_ids(ac_bus_tbl, ac_bus_lim, "v_max", ac_nodes),
        ),
    )
    _register_lookup(
        lookup,
        DEVICE_TYPE_DCACConverter,
        (MEAS_TYPE_V_DC,),
        _coalesce_limit_vecs(
            _slice_limit_vec(dcac_lim, "v_dc_min", rows),
            _node_limit_by_node_ids(dc_bus_tbl, dc_bus_lim, "v_min", dc_nodes),
        ),
        _coalesce_limit_vecs(
            _slice_limit_vec(dcac_lim, "v_dc_max", rows),
            _node_limit_by_node_ids(dc_bus_tbl, dc_bus_lim, "v_max", dc_nodes),
        ),
    )
    _register_lookup(
        lookup,
        DEVICE_TYPE_DCACConverter,
        (MEAS_TYPE_I_AC,),
        _coalesce_limit_vecs(_slice_limit_vec(dcac_lim, "i_ac_min", rows), _slice_limit_vec(dcac_lim, "i_min", rows)),
        _coalesce_limit_vecs(_slice_limit_vec(dcac_lim, "i_ac_max", rows), _slice_limit_vec(dcac_lim, "i_max", rows)),
        symmetric_upper=False,
    )
    _register_lookup(
        lookup,
        DEVICE_TYPE_DCACConverter,
        (MEAS_TYPE_I_DC,),
        _coalesce_limit_vecs(_slice_limit_vec(dcac_lim, "i_dc_min", rows), _slice_limit_vec(dcac_lim, "i_min", rows)),
        _coalesce_limit_vecs(_slice_limit_vec(dcac_lim, "i_dc_max", rows), _slice_limit_vec(dcac_lim, "i_max", rows)),
        symmetric_upper=True,
    )
    return lookup


def build_fluid_device_limits_lookup(
    network: object,
    *,
    node_dt: int,
    pipe_dt: int,
    valve_dt: int,
    ctrl_dt: int,
    src_dt: int,
    sto_dt: int,
    load_dt: int,
    exch_dt: int = -1,
) -> Dict[Tuple[int, int], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]]:
    """根据 FluidNetwork 构建流体域 (device_type_code, meas_type_code) 设备状态限值查找表。"""
    dev_limits = getattr(network, "device_limits", None)
    if not isinstance(dev_limits, Mapping) or not any(dev_limits.values()):
        return {}
    lookup: Dict[Tuple[int, int], Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]] = {}
    node_lim = dev_limits.get("node") or {}
    src_lim = dev_limits.get("source") or {}
    load_lim = dev_limits.get("load") or {}
    edge_lim = dev_limits.get("edge") or {}
    exch_lim = dev_limits.get("exchanger") or {}

    def _by_pos(lim_map: Mapping[str, np.ndarray], key: str, pos_arr: np.ndarray) -> Optional[np.ndarray]:
        if key not in lim_map or pos_arr.size == 0:
            return None
        src = np.asarray(lim_map[key], dtype=np.float64)
        out = np.full(pos_arr.size, np.nan, dtype=np.float64)
        ok = (pos_arr >= 0) & (pos_arr < src.size)
        if np.any(ok):
            out[ok] = src[pos_arr[ok].astype(np.intp, copy=False)]
        return out if np.any(np.isfinite(out)) else None

    # 1. 节点限值
    if node_dt >= 0 and node_lim:
        _register_lookup(lookup, node_dt, (MEAS_TYPE_PRESSURE,), node_lim.get("pressure_min"), node_lim.get("pressure_max"))
        _register_lookup(lookup, node_dt, (MEAS_TYPE_TEMPERATURE,), node_lim.get("temperature_min"), node_lim.get("temperature_max"))
        _register_lookup(
            lookup,
            node_dt,
            (MEAS_TYPE_T_SUPPLY,),
            _coalesce_limit_vecs(node_lim.get("supply_temperature_min"), node_lim.get("temperature_min")),
            _coalesce_limit_vecs(node_lim.get("supply_temperature_max"), node_lim.get("temperature_max")),
        )
        _register_lookup(
            lookup,
            node_dt,
            (MEAS_TYPE_T_RETURN,),
            _coalesce_limit_vecs(node_lim.get("return_temperature_min"), node_lim.get("temperature_min")),
            _coalesce_limit_vecs(node_lim.get("return_temperature_max"), node_lim.get("temperature_max")),
        )
        _register_lookup(lookup, node_dt, (MEAS_TYPE_ENTHALPY,), node_lim.get("enthalpy_min"), node_lim.get("enthalpy_max"))

    # 2. 边限值（Pipe / Valve / Pump / Compressor / PressureReducer 共享 edge_pos）
    edge_i = np.asarray(getattr(network, "edge_i", ()), dtype=np.int64)
    edge_j = np.asarray(getattr(network, "edge_j", ()), dtype=np.int64)
    edge_dts = tuple(c for c in (pipe_dt, valve_dt, ctrl_dt) if c >= 0)
    if edge_dts and edge_i.size:
        ef_lo, ef_hi = edge_lim.get("flow_min"), edge_lim.get("flow_max")
        ep_lo, ep_hi = edge_lim.get("pressure_min"), edge_lim.get("pressure_max")
        et_lo, et_hi = edge_lim.get("temperature_min"), edge_lim.get("temperature_max")
        eh_lo, eh_hi = edge_lim.get("enthalpy_min"), edge_lim.get("enthalpy_max")
        eq_lo, eq_hi = edge_lim.get("heat_min"), edge_lim.get("heat_max")

        p_from_lo = _coalesce_limit_vecs(ep_lo, _by_pos(node_lim, "pressure_min", edge_i))
        p_from_hi = _coalesce_limit_vecs(ep_hi, _by_pos(node_lim, "pressure_max", edge_i))
        p_to_lo = _coalesce_limit_vecs(ep_lo, _by_pos(node_lim, "pressure_min", edge_j))
        p_to_hi = _coalesce_limit_vecs(ep_hi, _by_pos(node_lim, "pressure_max", edge_j))

        t_from_lo = _coalesce_limit_vecs(et_lo, _by_pos(node_lim, "temperature_min", edge_i))
        t_from_hi = _coalesce_limit_vecs(et_hi, _by_pos(node_lim, "temperature_max", edge_i))
        t_to_lo = _coalesce_limit_vecs(et_lo, _by_pos(node_lim, "temperature_min", edge_j))
        t_to_hi = _coalesce_limit_vecs(et_hi, _by_pos(node_lim, "temperature_max", edge_j))

        ts_from_lo = _coalesce_limit_vecs(et_lo, _by_pos(node_lim, "supply_temperature_min", edge_i), _by_pos(node_lim, "temperature_min", edge_i))
        ts_from_hi = _coalesce_limit_vecs(et_hi, _by_pos(node_lim, "supply_temperature_max", edge_i), _by_pos(node_lim, "temperature_max", edge_i))
        ts_to_lo = _coalesce_limit_vecs(et_lo, _by_pos(node_lim, "supply_temperature_min", edge_j), _by_pos(node_lim, "temperature_min", edge_j))
        ts_to_hi = _coalesce_limit_vecs(et_hi, _by_pos(node_lim, "supply_temperature_max", edge_j), _by_pos(node_lim, "temperature_max", edge_j))

        tr_from_lo = _coalesce_limit_vecs(et_lo, _by_pos(node_lim, "return_temperature_min", edge_i), _by_pos(node_lim, "temperature_min", edge_i))
        tr_from_hi = _coalesce_limit_vecs(et_hi, _by_pos(node_lim, "return_temperature_max", edge_i), _by_pos(node_lim, "temperature_max", edge_i))
        tr_to_lo = _coalesce_limit_vecs(et_lo, _by_pos(node_lim, "return_temperature_min", edge_j), _by_pos(node_lim, "temperature_min", edge_j))
        tr_to_hi = _coalesce_limit_vecs(et_hi, _by_pos(node_lim, "return_temperature_max", edge_j), _by_pos(node_lim, "temperature_max", edge_j))

        h_from_lo = _coalesce_limit_vecs(eh_lo, _by_pos(node_lim, "enthalpy_min", edge_i))
        h_from_hi = _coalesce_limit_vecs(eh_hi, _by_pos(node_lim, "enthalpy_max", edge_i))
        h_to_lo = _coalesce_limit_vecs(eh_lo, _by_pos(node_lim, "enthalpy_min", edge_j))
        h_to_hi = _coalesce_limit_vecs(eh_hi, _by_pos(node_lim, "enthalpy_max", edge_j))

        for edt in edge_dts:
            _register_lookup(lookup, edt, (MEAS_TYPE_FLOW, MEAS_TYPE_FLOW_FROM, MEAS_TYPE_FLOW_TO), ef_lo, ef_hi, symmetric_upper=True)
            _register_lookup(lookup, edt, (MEAS_TYPE_PRESSURE_FROM, MEAS_TYPE_P_FROM), p_from_lo, p_from_hi)
            _register_lookup(lookup, edt, (MEAS_TYPE_PRESSURE_TO, MEAS_TYPE_P_TO), p_to_lo, p_to_hi)
            _register_lookup(lookup, edt, (MEAS_TYPE_T_FROM,), t_from_lo, t_from_hi)
            _register_lookup(lookup, edt, (MEAS_TYPE_T_TO,), t_to_lo, t_to_hi)
            _register_lookup(lookup, edt, (MEAS_TYPE_TS_FROM,), ts_from_lo, ts_from_hi)
            _register_lookup(lookup, edt, (MEAS_TYPE_TS_TO,), ts_to_lo, ts_to_hi)
            _register_lookup(lookup, edt, (MEAS_TYPE_TR_FROM,), tr_from_lo, tr_from_hi)
            _register_lookup(lookup, edt, (MEAS_TYPE_TR_TO,), tr_to_lo, tr_to_hi)
            _register_lookup(lookup, edt, (MEAS_TYPE_H_FROM,), h_from_lo, h_from_hi)
            _register_lookup(lookup, edt, (MEAS_TYPE_H_TO,), h_to_lo, h_to_hi)
            _register_lookup(lookup, edt, (MEAS_TYPE_HEAT,), eq_lo, eq_hi, symmetric_upper=True)

    # 3. 源与储（Source / Storage 共享 source_pos）
    src_nodes = np.asarray(getattr(network, "source_node_pos", ()), dtype=np.int64)
    src_dts = tuple(c for c in (src_dt, sto_dt) if c >= 0)
    if src_dts and src_nodes.size:
        sp_lo = _coalesce_limit_vecs(src_lim.get("pressure_min"), _by_pos(node_lim, "pressure_min", src_nodes))
        sp_hi = _coalesce_limit_vecs(src_lim.get("pressure_max"), _by_pos(node_lim, "pressure_max", src_nodes))
        st_lo = _coalesce_limit_vecs(src_lim.get("temperature_min"), _by_pos(node_lim, "temperature_min", src_nodes))
        st_hi = _coalesce_limit_vecs(src_lim.get("temperature_max"), _by_pos(node_lim, "temperature_max", src_nodes))
        sts_lo = _coalesce_limit_vecs(src_lim.get("supply_temperature_min"), src_lim.get("temperature_min"), _by_pos(node_lim, "supply_temperature_min", src_nodes), _by_pos(node_lim, "temperature_min", src_nodes))
        sts_hi = _coalesce_limit_vecs(src_lim.get("supply_temperature_max"), src_lim.get("temperature_max"), _by_pos(node_lim, "supply_temperature_max", src_nodes), _by_pos(node_lim, "temperature_max", src_nodes))
        str_lo = _coalesce_limit_vecs(src_lim.get("return_temperature_min"), src_lim.get("temperature_min"), _by_pos(node_lim, "return_temperature_min", src_nodes), _by_pos(node_lim, "temperature_min", src_nodes))
        str_hi = _coalesce_limit_vecs(src_lim.get("return_temperature_max"), src_lim.get("temperature_max"), _by_pos(node_lim, "return_temperature_max", src_nodes), _by_pos(node_lim, "temperature_max", src_nodes))
        sh_lo = _coalesce_limit_vecs(src_lim.get("enthalpy_min"), _by_pos(node_lim, "enthalpy_min", src_nodes))
        sh_hi = _coalesce_limit_vecs(src_lim.get("enthalpy_max"), _by_pos(node_lim, "enthalpy_max", src_nodes))
        for sdt in src_dts:
            _register_lookup(lookup, sdt, (MEAS_TYPE_FLOW,), src_lim.get("flow_min"), src_lim.get("flow_max"), symmetric_upper=False)
            _register_lookup(lookup, sdt, (MEAS_TYPE_PRESSURE,), sp_lo, sp_hi)
            _register_lookup(lookup, sdt, (MEAS_TYPE_TEMPERATURE,), st_lo, st_hi)
            _register_lookup(lookup, sdt, (MEAS_TYPE_T_SUPPLY,), sts_lo, sts_hi)
            _register_lookup(lookup, sdt, (MEAS_TYPE_T_RETURN,), str_lo, str_hi)
            _register_lookup(lookup, sdt, (MEAS_TYPE_ENTHALPY,), sh_lo, sh_hi)
            _register_lookup(lookup, sdt, (MEAS_TYPE_HEAT,), src_lim.get("heat_min"), src_lim.get("heat_max"), symmetric_upper=True)

    # 4. 负荷
    load_nodes = np.asarray(getattr(network, "load_node_pos", ()), dtype=np.int64)
    if load_dt >= 0 and load_nodes.size:
        _register_lookup(lookup, load_dt, (MEAS_TYPE_FLOW,), load_lim.get("flow_min"), load_lim.get("flow_max"), symmetric_upper=False)
        _register_lookup(
            lookup,
            load_dt,
            (MEAS_TYPE_PRESSURE,),
            _coalesce_limit_vecs(load_lim.get("pressure_min"), _by_pos(node_lim, "pressure_min", load_nodes)),
            _coalesce_limit_vecs(load_lim.get("pressure_max"), _by_pos(node_lim, "pressure_max", load_nodes)),
        )
        _register_lookup(
            lookup,
            load_dt,
            (MEAS_TYPE_TEMPERATURE,),
            _coalesce_limit_vecs(load_lim.get("temperature_min"), _by_pos(node_lim, "temperature_min", load_nodes)),
            _coalesce_limit_vecs(load_lim.get("temperature_max"), _by_pos(node_lim, "temperature_max", load_nodes)),
        )
        _register_lookup(
            lookup,
            load_dt,
            (MEAS_TYPE_T_SUPPLY,),
            _coalesce_limit_vecs(load_lim.get("supply_temperature_min"), load_lim.get("temperature_min"), _by_pos(node_lim, "supply_temperature_min", load_nodes), _by_pos(node_lim, "temperature_min", load_nodes)),
            _coalesce_limit_vecs(load_lim.get("supply_temperature_max"), load_lim.get("temperature_max"), _by_pos(node_lim, "supply_temperature_max", load_nodes), _by_pos(node_lim, "temperature_max", load_nodes)),
        )
        _register_lookup(
            lookup,
            load_dt,
            (MEAS_TYPE_T_RETURN,),
            _coalesce_limit_vecs(load_lim.get("return_temperature_min"), load_lim.get("temperature_min"), _by_pos(node_lim, "return_temperature_min", load_nodes), _by_pos(node_lim, "temperature_min", load_nodes)),
            _coalesce_limit_vecs(load_lim.get("return_temperature_max"), load_lim.get("temperature_max"), _by_pos(node_lim, "return_temperature_max", load_nodes), _by_pos(node_lim, "temperature_max", load_nodes)),
        )
        _register_lookup(
            lookup,
            load_dt,
            (MEAS_TYPE_ENTHALPY,),
            _coalesce_limit_vecs(load_lim.get("enthalpy_min"), _by_pos(node_lim, "enthalpy_min", load_nodes)),
            _coalesce_limit_vecs(load_lim.get("enthalpy_max"), _by_pos(node_lim, "enthalpy_max", load_nodes)),
        )
        _register_lookup(lookup, load_dt, (MEAS_TYPE_HEAT,), load_lim.get("heat_min"), load_lim.get("heat_max"), symmetric_upper=True)

    # 5. 换热器
    if exch_dt >= 0 and exch_lim:
        _register_lookup(lookup, exch_dt, (MEAS_TYPE_FLOW, MEAS_TYPE_FLOW_FROM, MEAS_TYPE_FLOW_TO), exch_lim.get("flow_min"), exch_lim.get("flow_max"), symmetric_upper=True)
        _register_lookup(lookup, exch_dt, (MEAS_TYPE_HEAT,), exch_lim.get("heat_min"), exch_lim.get("heat_max"), symmetric_upper=True)
        _register_lookup(lookup, exch_dt, (MEAS_TYPE_TEMPERATURE, MEAS_TYPE_T_FROM, MEAS_TYPE_T_TO), exch_lim.get("temperature_min"), exch_lim.get("temperature_max"))

    return lookup
