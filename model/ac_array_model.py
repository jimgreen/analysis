import mmap
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from efile_read import _read_efile_rows, efile_factory_from_file, efile_factory_from_rows
from paths import resolve_project_file
from unit_system import ac_current_base_ka

MODEL_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODEL_DIR.parent
for path in (MODEL_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from topology import (
    GridTopologyInput,
    SingleDeviceTopologyInput,
    TerminalDeviceTopologyInput,
    _EMPTY_BOOL,
    _EMPTY_INT,
    _map_node_positions,
    build_ac_topology_input_ppc,
)
from model.effective_state import propagate_composite_run_states
from model.setpoint_validation import (
    MAX_VOLTAGE_SETPOINT,
    MIN_VOLTAGE_SETPOINT,
    append_setpoint_corrections,
    sanitize_fixed_setpoints,
)
from model.switching_status import switching_status_columns
from model.array_common import (
    LazyNameArray,
    _assign_current_if_present,
    _assign_power_if_present,
    _base_from_rows,
    _build_node_float_lookup,
    _cell,
    _code_column,
    _code_value,
    _current_scale_by_node,
    _empty,
    _fill_float_column_if_present,
    _float_cell,
    _float_column,
    _float_value,
    _has_value,
    _int_cell,
    _int_column,
    _int_value,
    _name_array,
    _names_from_rows,
    _optional_ppc_vector,
    _preload_numeric_columns,
    _raw_vbase_by_node,
    _rows_for,
    _scale_by_node,
    _value,
    _voltage_set_column,
    file_cache_key as _file_cache_key,
)


CTRL_PQ = 0
CTRL_P = 1
CTRL_PV = 2
CTRL_SLACK = 3
CTRL_CODE = {
    "PQ": CTRL_PQ,
    "P": CTRL_P,
    "PV": CTRL_PV,
    "V": CTRL_SLACK,
    "SLACK": CTRL_SLACK,
    "PH": CTRL_SLACK,
}

SHUNT_Q = 0
SHUNT_V = 1
SHUNT_B = 2
SHUNT_Z = 3
SHUNT_CODE = {
    "Q": SHUNT_Q,
    "V": SHUNT_V,
    "B": SHUNT_B,
    "Z": SHUNT_Z,
}

BUS_COLS = {
    "idx": 0,
    "vbase": 1,
    "voltage": 2,
    "angle": 3,
    "isl": 4,
    "run_stat": 5,
}
BRANCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "r": 3,
    "x": 4,
    "b": 5,
    "run_stat": 6,
    "i_p": 7,
    "i_q": 8,
    "i_c": 9,
    "j_p": 10,
    "j_q": 11,
    "j_c": 12,
}
TRANSFORMER_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "r": 3,
    "x": 4,
    "gt": 5,
    "bt": 6,
    "tap": 7,
    "shift": 8,
    "run_stat": 9,
    "i_p": 10,
    "i_q": 11,
    "i_c": 12,
    "j_p": 13,
    "j_q": 14,
    "j_c": 15,
}
THREE_WINDING_TRANSFORMER_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "k_node": 3,
    "i_r": 4,
    "i_x": 5,
    "j_r": 6,
    "j_x": 7,
    "k_r": 8,
    "k_x": 9,
    "gt": 10,
    "bt": 11,
    "i_tap": 12,
    "i_shift": 13,
    "j_tap": 14,
    "j_shift": 15,
    "k_tap": 16,
    "k_shift": 17,
    "run_stat": 18,
    "i_p": 19,
    "i_q": 20,
    "i_c": 21,
    "j_p": 22,
    "j_q": 23,
    "j_c": 24,
    "k_p": 25,
    "k_q": 26,
    "k_c": 27,
}
GEN_COLS = {
    "idx": 0,
    "node": 1,
    "control_type": 2,
    "p_set": 3,
    "q_set": 4,
    "v_set": 5,
    "alpha": 6,
    "run_stat": 7,
    "p": 8,
    "q": 9,
    "current": 10,
    "p_max": 11,
}

AC_GEN_LIMIT_KEYS = ("_gen_p_min", "_gen_q_min", "_gen_q_max")


def ensure_ac_ppc_gen_columns(ppc: Dict) -> Dict:
    """Upgrade legacy ac_ppc_v1 generator arrays with the optional p_max column."""
    gen = np.asarray(ppc.get("gen", _empty(len(GEN_COLS))), dtype=np.float64)
    required_width = len(GEN_COLS)
    if gen.ndim != 2:
        raise ValueError(f"AC PPC gen array must be two-dimensional, got shape {gen.shape}")
    if gen.shape[1] >= required_width:
        return ppc
    if gen.shape[0] and gen.shape[1] != GEN_COLS["p_max"]:
        raise ValueError(
            f"Legacy AC PPC gen array must have {GEN_COLS['p_max']} columns, got {gen.shape[1]}"
        )
    upgraded = np.full((gen.shape[0], required_width), np.nan, dtype=np.float64)
    if gen.shape[1]:
        upgraded[:, :gen.shape[1]] = gen
    ppc["gen"] = upgraded
    return ppc
LOAD_COLS = {
    "idx": 0,
    "node": 1,
    "pbase": 2,
    "pv0": 3,
    "pv1": 4,
    "pv2": 5,
    "qbase": 6,
    "qv0": 7,
    "qv1": 8,
    "qv2": 9,
    "run_stat": 10,
    "p": 11,
    "q": 12,
    "current": 13,
}
SHUNT_COLS = {
    "idx": 0,
    "node": 1,
    "control_type": 2,
    "q_set": 3,
    "g_set": 4,
    "b_set": 5,
    "v_set": 6,
    "run_stat": 7,
    "p": 8,
    "q": 9,
    "current": 10,
}
ZERO_BRANCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "run_stat": 3,
    "p": 4,
    "q": 5,
    "current": 6,
}
SWITCH_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "status": 3,
    "run_stat": 4,
    "p": 5,
    "q": 6,
    "current": 7,
    "closed_status_set": 8,
    "closed_status": 9,
}
BREAK_COLS = SWITCH_COLS
ACAC_LEGACY_CONTROL_CODE = {"PQQ": 0, "PVQ": 1, "PQV": 2, "PVV": 3}
ACAC_LEGACY_CONTROL_LABEL = {value: key for key, value in ACAC_LEGACY_CONTROL_CODE.items()}
# Preserve the historical Q/V numeric codes while exposing canonical AC labels.
ACAC_SIDE_CONTROL_CODE = {"PQ": 0, "PV": 1, "PH": 2, "NONE": 3}
ACAC_SIDE_CONTROL_LABEL = {value: key for key, value in ACAC_SIDE_CONTROL_CODE.items()}
ACAC_SIDE_CONTROL_PARSE_CODE = {
    **ACAC_SIDE_CONTROL_CODE,
    "CTRL_PQ": ACAC_SIDE_CONTROL_CODE["PQ"],
    "Q": ACAC_SIDE_CONTROL_CODE["PQ"],
    "CTRL_PV": ACAC_SIDE_CONTROL_CODE["PV"],
    "V": ACAC_SIDE_CONTROL_CODE["PV"],
    "CTRL_PH": ACAC_SIDE_CONTROL_CODE["PH"],
    "CTRL_NONE": ACAC_SIDE_CONTROL_CODE["NONE"],
    "UNSPEC": ACAC_SIDE_CONTROL_CODE["NONE"],
    "UNDEFINED": ACAC_SIDE_CONTROL_CODE["NONE"],
    "NA": ACAC_SIDE_CONTROL_CODE["NONE"],
    "不定": ACAC_SIDE_CONTROL_CODE["NONE"],
}
ACAC_LEGACY_TO_PAIR = {
    "PQQ": ("PQ", "PQ"),
    "PVQ": ("PV", "PQ"),
    "PQV": ("PQ", "PV"),
    "PVV": ("PV", "PV"),
}
ACAC_PAIR_TO_LEGACY = {value: key for key, value in ACAC_LEGACY_TO_PAIR.items()}


def acac_control_pair_from_legacy(control_type):
    label = str(control_type or "PQQ").upper()
    if label not in ACAC_LEGACY_TO_PAIR:
        raise ValueError(f"未知 ACACConverter 控制模式: {control_type}")
    return ACAC_LEGACY_TO_PAIR[label]


def acac_legacy_control_label(i_control_type, j_control_type):
    i_code = ACAC_SIDE_CONTROL_PARSE_CODE.get(str(i_control_type or "PQ").upper())
    j_code = ACAC_SIDE_CONTROL_PARSE_CODE.get(str(j_control_type or "PQ").upper())
    if i_code is None or j_code is None:
        raise ValueError(
            f"未知 ACACConverter 交流端控制组合: ({i_control_type}, {j_control_type})"
        )
    i_label = ACAC_SIDE_CONTROL_LABEL[i_code]
    j_label = ACAC_SIDE_CONTROL_LABEL[j_code]
    legacy = ACAC_PAIR_TO_LEGACY.get((i_label, j_label))
    if legacy is None:
        raise ValueError(f"不支持的 ACACConverter 控制组合: ({i_label}, {j_label})")
    return legacy


def acac_legacy_control_code(i_control_type, j_control_type):
    return ACAC_LEGACY_CONTROL_CODE[acac_legacy_control_label(i_control_type, j_control_type)]


def acac_combined_control_code(i_control_type, j_control_type):
    return acac_legacy_control_code(i_control_type, j_control_type)
ACAC_COLS = {
    "idx": 0,
    "i_node": 1,
    "j_node": 2,
    "r1": 3,
    "r2": 4,
    "i_control_type": 5,
    "j_control_type": 6,
    "p_set": 7,
    "i_q_set": 8,
    "j_q_set": 9,
    "i_v_set": 10,
    "j_v_set": 11,
    "run_stat": 12,
    "i_p": 13,
    "i_q": 14,
    "j_p": 15,
    "j_q": 16,
    "i_i": 17,
    "j_i": 18,
}


def sanitize_ac_voltage_setpoints(ppc: Dict) -> list[dict]:
    """Normalize every active AC voltage-control setpoint in place."""

    corrections = []
    gen = np.asarray(ppc.get("gen", _empty(len(GEN_COLS))), dtype=np.float64)
    if gen.size:
        control = gen[:, GEN_COLS["control_type"]].astype(np.int8, copy=False)
        active = gen[:, GEN_COLS["run_stat"]] == 1
        voltage_control = active & np.isin(control, (CTRL_PV, CTRL_SLACK))
        auto_slack = np.asarray(ppc.get("_auto_slack_gen_rows", ()), dtype=np.int64)
        if auto_slack.size:
            valid = auto_slack[(auto_slack >= 0) & (auto_slack < gen.shape[0])]
            voltage_control[valid] = active[valid]
        _, items = sanitize_fixed_setpoints(
            gen[:, GEN_COLS["v_set"]],
            voltage_control,
            device_type="ACGenerator",
            field="v_set",
            indices=gen[:, GEN_COLS["idx"]],
        )
        corrections.extend(items)

    shunt = np.asarray(ppc.get("shunt", _empty(len(SHUNT_COLS))), dtype=np.float64)
    if shunt.size:
        voltage_control = (
            (shunt[:, SHUNT_COLS["run_stat"]] == 1)
            & (shunt[:, SHUNT_COLS["control_type"]].astype(np.int8, copy=False) == SHUNT_V)
        )
        _, items = sanitize_fixed_setpoints(
            shunt[:, SHUNT_COLS["v_set"]],
            voltage_control,
            device_type="ACShuntCompensator",
            field="v_set",
            indices=shunt[:, SHUNT_COLS["idx"]],
        )
        corrections.extend(items)

    acac = np.asarray(ppc.get("acac", _empty(len(ACAC_COLS))), dtype=np.float64)
    if acac.size:
        active = acac[:, ACAC_COLS["run_stat"]] == 1
        voltage_codes = (ACAC_SIDE_CONTROL_CODE["PV"], ACAC_SIDE_CONTROL_CODE["PH"])
        for side in ("i", "j"):
            voltage_control = active & np.isin(
                acac[:, ACAC_COLS[f"{side}_control_type"]].astype(np.int8, copy=False),
                voltage_codes,
            )
            _, items = sanitize_fixed_setpoints(
                acac[:, ACAC_COLS[f"{side}_v_set"]],
                voltage_control,
                device_type="ACACConverter",
                field=f"{side}_v_set",
                indices=acac[:, ACAC_COLS["idx"]],
                side=side,
            )
            corrections.extend(items)

    append_setpoint_corrections(ppc, corrections)
    return corrections

MP_BUS_I = 0
MP_BUS_TYPE = 1
MP_PD = 2
MP_QD = 3
MP_GS = 4
MP_BS = 5
MP_BUS_AREA = 6
MP_VM = 7
MP_VA = 8
MP_BASE_KV = 9
MP_ZONE = 10
MP_VMAX = 11
MP_VMIN = 12

MP_GEN_BUS = 0
MP_PG = 1
MP_QG = 2
MP_QMAX = 3
MP_QMIN = 4
MP_VG = 5
MP_MBASE = 6
MP_GEN_STATUS = 7
MP_PMAX = 8
MP_PMIN = 9

MP_F_BUS = 0
MP_T_BUS = 1
MP_BR_R = 2
MP_BR_X = 3
MP_BR_B = 4
MP_RATE_A = 5
MP_RATE_B = 6
MP_RATE_C = 7
MP_TAP = 8
MP_SHIFT = 9
MP_BR_STATUS = 10
MP_ANGMIN = 11
MP_ANGMAX = 12

MP_PQ = 1
MP_PV = 2
MP_REF = 3
MP_NONE = 4


def _first_existing_column(columns: Dict[str, int], *names: str):
    for name in names:
        if name in columns:
            return name
    return None


def _aliased_float_column(table_rows, columns, names, default=0.0):
    name = _first_existing_column(columns, *names)
    if name is None:
        return np.full(len(table_rows), float(default), dtype=np.float64)
    return _float_column(table_rows, columns, name, default)


def _three_winding_star_impedances(table_rows, columns):
    """Return star-equivalent winding impedances.

    Canonical E columns are ``i_r/i_x``, ``j_r/j_x`` and ``k_r/k_x``.
    For utility data that stores pairwise short-circuit impedances, the
    aliases ``ij_*``, ``ik_*`` and ``jk_*`` are also accepted and converted.
    """
    direct = all(
        _first_existing_column(columns, f"{terminal}_{part}", f"{part}_{terminal}") is not None
        for terminal in ("i", "j", "k")
        for part in ("r", "x")
    )
    if direct:
        return tuple(
            _aliased_float_column(table_rows, columns, (f"{terminal}_{part}", f"{part}_{terminal}"))
            for terminal in ("i", "j", "k")
            for part in ("r", "x")
        )

    z_ij = _aliased_float_column(table_rows, columns, ("ij_r", "r_ij")) + 1j * _aliased_float_column(
        table_rows, columns, ("ij_x", "x_ij")
    )
    z_ik = _aliased_float_column(table_rows, columns, ("ik_r", "ki_r", "r_ik", "r_ki")) + 1j * _aliased_float_column(
        table_rows, columns, ("ik_x", "ki_x", "x_ik", "x_ki")
    )
    z_jk = _aliased_float_column(table_rows, columns, ("jk_r", "kj_r", "r_jk", "r_kj")) + 1j * _aliased_float_column(
        table_rows, columns, ("jk_x", "kj_x", "x_jk", "x_kj")
    )
    z_i = 0.5 * (z_ij + z_ik - z_jk)
    z_j = 0.5 * (z_ij + z_jk - z_ik)
    z_k = 0.5 * (z_ik + z_jk - z_ij)
    return z_i.real, z_i.imag, z_j.real, z_j.imag, z_k.real, z_k.imag


def _build_switch_like_from_rows(
    table_rows,
    columns,
    current_scale_by_node: Dict[int, float],
    *,
    prefix: str,
    scale_optional_power: bool = True,
    p_base: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(table_rows), len(SWITCH_COLS)), dtype=np.float64)
    if not table_rows:
        return out, np.asarray([], dtype=object)
    out[:, SWITCH_COLS["idx"]] = _int_column(table_rows, columns, "idx")
    out[:, SWITCH_COLS["i_node"]] = _int_column(table_rows, columns, "i_node")
    out[:, SWITCH_COLS["j_node"]] = _int_column(table_rows, columns, "j_node")
    status, closed_status_set, closed_status = switching_status_columns(table_rows, columns)
    out[:, SWITCH_COLS["status"]] = status
    out[:, SWITCH_COLS["closed_status_set"]] = closed_status_set
    out[:, SWITCH_COLS["closed_status"]] = closed_status
    out[:, SWITCH_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
    if scale_optional_power:
        _assign_power_if_present(out, SWITCH_COLS["p"], table_rows, columns, "p", p_base)
        _assign_power_if_present(out, SWITCH_COLS["q"], table_rows, columns, "q", p_base)
        _assign_current_if_present(
            out,
            SWITCH_COLS["current"],
            table_rows,
            columns,
            "current",
            out[:, SWITCH_COLS["i_node"]],
            current_scale_by_node,
        )
    else:
        if "p" in columns:
            out[:, SWITCH_COLS["p"]] = _float_column(table_rows, columns, "p")
        if "q" in columns:
            out[:, SWITCH_COLS["q"]] = _float_column(table_rows, columns, "q")
        if "current" in columns:
            out[:, SWITCH_COLS["current"]] = _float_column(table_rows, columns, "current")
    return out, _names_from_rows(table_rows, columns, prefix, out[:, SWITCH_COLS["idx"]])


def _build_acac_from_rows(
    table_rows,
    columns,
    raw_vbase_by_idx: Dict[int, float],
    current_scale_by_node,
    p_base: float,
) -> Tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(table_rows), len(ACAC_COLS)), dtype=np.float64)
    if not table_rows:
        return out, np.asarray([], dtype=object)
    out[:, ACAC_COLS["idx"]] = _int_column(table_rows, columns, "idx")
    out[:, ACAC_COLS["i_node"]] = _int_column(table_rows, columns, "i_node")
    out[:, ACAC_COLS["j_node"]] = _int_column(table_rows, columns, "j_node")
    out[:, ACAC_COLS["r1"]] = _float_column(table_rows, columns, "r1")
    out[:, ACAC_COLS["r2"]] = _float_column(table_rows, columns, "r2")
    if "i_control_type" in columns or "j_control_type" in columns:
        out[:, ACAC_COLS["i_control_type"]] = _code_column(
            table_rows, columns, "i_control_type", ACAC_SIDE_CONTROL_PARSE_CODE, "PQ"
        )
        out[:, ACAC_COLS["j_control_type"]] = _code_column(
            table_rows, columns, "j_control_type", ACAC_SIDE_CONTROL_PARSE_CODE, "PQ"
        )
    else:
        pairs = [acac_control_pair_from_legacy(_cell(row, columns.get("control_type"), "PQQ")) for row in table_rows]
        out[:, ACAC_COLS["i_control_type"]] = np.asarray(
            [ACAC_SIDE_CONTROL_CODE[i_ctrl] for i_ctrl, _j_ctrl in pairs],
            dtype=np.float64,
        )
        out[:, ACAC_COLS["j_control_type"]] = np.asarray(
            [ACAC_SIDE_CONTROL_CODE[j_ctrl] for _i_ctrl, j_ctrl in pairs],
            dtype=np.float64,
        )
    out[:, ACAC_COLS["p_set"]] = _float_column(table_rows, columns, "p_set") / p_base
    out[:, ACAC_COLS["i_q_set"]] = _float_column(table_rows, columns, "i_q_set") / p_base
    out[:, ACAC_COLS["j_q_set"]] = _float_column(table_rows, columns, "j_q_set") / p_base
    out[:, ACAC_COLS["i_v_set"]] = _voltage_set_column(
        table_rows,
        columns,
        "i_v_set",
        out[:, ACAC_COLS["i_node"]],
        raw_vbase_by_idx,
    )
    out[:, ACAC_COLS["j_v_set"]] = _voltage_set_column(
        table_rows,
        columns,
        "j_v_set",
        out[:, ACAC_COLS["j_node"]],
        raw_vbase_by_idx,
    )
    out[:, ACAC_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
    for attr in ("i_p", "i_q", "j_p", "j_q"):
        _assign_power_if_present(out, ACAC_COLS[attr], table_rows, columns, attr, p_base)
    _assign_current_if_present(
        out,
        ACAC_COLS["i_i"],
        table_rows,
        columns,
        "i_i",
        out[:, ACAC_COLS["i_node"]],
        current_scale_by_node,
    )
    _assign_current_if_present(
        out,
        ACAC_COLS["j_i"],
        table_rows,
        columns,
        "j_i",
        out[:, ACAC_COLS["j_node"]],
        current_scale_by_node,
    )
    return out, _names_from_rows(table_rows, columns, "acac", out[:, ACAC_COLS["idx"]])


def _build_ac_ppc_from_rows_dict(rows: Dict, source) -> Dict:
    p_base, u_scale, p_scale, i_scale, p_base_kW = _base_from_rows(rows)

    columns, table_rows = _rows_for(rows, "ACNode")
    bus = np.zeros((len(table_rows), len(BUS_COLS)), dtype=np.float64, order="F")
    if table_rows:
        _preload_numeric_columns(table_rows, columns, ("idx", "vbase", "voltage", "angle", "isl", "run_stat"))
        raw_vbase = _float_column(table_rows, columns, "vbase")
        bus[:, BUS_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        bus[:, BUS_COLS["vbase"]] = raw_vbase / u_scale
        if "voltage" in columns:
            raw_voltage = _float_column(table_rows, columns, "voltage", np.nan)
            missing_voltage = ~np.isfinite(raw_voltage)
            if np.any(missing_voltage):
                raw_voltage = raw_voltage.copy()
                raw_voltage[missing_voltage] = raw_vbase[missing_voltage]
            bus[:, BUS_COLS["voltage"]] = np.divide(
                raw_voltage,
                raw_vbase,
                out=np.ones_like(raw_voltage),
                where=np.abs(raw_vbase) > 1e-12,
            )
        else:
            # 节点表未提供 voltage 列时，默认按标幺 1.0 p.u.（物理电压等于基准电压）初始化
            bus[:, BUS_COLS["voltage"]] = 1.0
        bus[:, BUS_COLS["angle"]] = np.deg2rad(_float_column(table_rows, columns, "angle", 0.0))
        bus[:, BUS_COLS["isl"]] = _float_column(table_rows, columns, "isl", 0.0)
        bus[:, BUS_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
    bus_names = _names_from_rows(table_rows, columns, "bus", bus[:, BUS_COLS["idx"]])
    raw_vbase_by_idx = _build_node_float_lookup(
        bus[:, BUS_COLS["idx"]],
        raw_vbase if table_rows else np.empty(0, dtype=np.float64),
        0.0,
    )
    current_scale_by_node = None

    def get_current_scale_by_node():
        nonlocal current_scale_by_node
        if current_scale_by_node is None:
            current_scale_by_node = _current_scale_by_node(
                bus[:, BUS_COLS["idx"]], bus[:, BUS_COLS["vbase"]], p_base_kW, i_scale, ac=True,
            )
        return current_scale_by_node

    columns, table_rows = _rows_for(rows, "ACBranch")
    branch = np.zeros((len(table_rows), len(BRANCH_COLS)), dtype=np.float64, order="F")
    if table_rows:
        _preload_numeric_columns(
            table_rows, columns, ("idx", "i_node", "j_node", "r", "x", "b", "run_stat", "i_p", "i_q", "j_p", "j_q", "i_c", "j_c")
        )
        branch[:, BRANCH_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        branch[:, BRANCH_COLS["i_node"]] = _int_column(table_rows, columns, "i_node")
        branch[:, BRANCH_COLS["j_node"]] = _int_column(table_rows, columns, "j_node")
        branch[:, BRANCH_COLS["r"]] = _float_column(table_rows, columns, "r")
        branch[:, BRANCH_COLS["x"]] = _float_column(table_rows, columns, "x")
        branch[:, BRANCH_COLS["b"]] = _float_column(table_rows, columns, "b")
        branch[:, BRANCH_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        for attr in ("i_p", "i_q", "j_p", "j_q"):
            _assign_power_if_present(branch, BRANCH_COLS[attr], table_rows, columns, attr, p_base)
        _assign_current_if_present(
            branch,
            BRANCH_COLS["i_c"],
            table_rows,
            columns,
            "i_c",
            branch[:, BRANCH_COLS["i_node"]],
            get_current_scale_by_node,
        )
        _assign_current_if_present(
            branch,
            BRANCH_COLS["j_c"],
            table_rows,
            columns,
            "j_c",
            branch[:, BRANCH_COLS["j_node"]],
            get_current_scale_by_node,
        )
    branch_names = _names_from_rows(table_rows, columns, "branch", branch[:, BRANCH_COLS["idx"]])

    columns, table_rows = _rows_for(rows, "ACTransformer")
    transformer = np.zeros((len(table_rows), len(TRANSFORMER_COLS)), dtype=np.float64, order="F")
    if table_rows:
        _preload_numeric_columns(
            table_rows,
            columns,
            ("idx", "i_node", "j_node", "r", "x", "gt", "bt", "b", "tap", "shift", "run_stat", "i_p", "i_q", "j_p", "j_q", "i_c", "j_c"),
        )
        transformer[:, TRANSFORMER_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        transformer[:, TRANSFORMER_COLS["i_node"]] = _int_column(table_rows, columns, "i_node")
        transformer[:, TRANSFORMER_COLS["j_node"]] = _int_column(table_rows, columns, "j_node")
        transformer[:, TRANSFORMER_COLS["r"]] = _float_column(table_rows, columns, "r")
        transformer[:, TRANSFORMER_COLS["x"]] = _float_column(table_rows, columns, "x")
        transformer[:, TRANSFORMER_COLS["gt"]] = _float_column(table_rows, columns, "gt")
        if "bt" in columns:
            transformer[:, TRANSFORMER_COLS["bt"]] = _float_column(table_rows, columns, "bt")
        elif "b" in columns:
            transformer[:, TRANSFORMER_COLS["bt"]] = _float_column(table_rows, columns, "b") / 2.0
        transformer[:, TRANSFORMER_COLS["tap"]] = _float_column(table_rows, columns, "tap", 1.0)
        transformer[:, TRANSFORMER_COLS["shift"]] = _float_column(table_rows, columns, "shift")
        transformer[:, TRANSFORMER_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        for attr in ("i_p", "i_q", "j_p", "j_q"):
            _assign_power_if_present(transformer, TRANSFORMER_COLS[attr], table_rows, columns, attr, p_base)
        _assign_current_if_present(
            transformer,
            TRANSFORMER_COLS["i_c"],
            table_rows,
            columns,
            "i_c",
            transformer[:, TRANSFORMER_COLS["i_node"]],
            get_current_scale_by_node,
        )
        _assign_current_if_present(
            transformer,
            TRANSFORMER_COLS["j_c"],
            table_rows,
            columns,
            "j_c",
            transformer[:, TRANSFORMER_COLS["j_node"]],
            get_current_scale_by_node,
        )
    transformer_names = _names_from_rows(table_rows, columns, "transformer", transformer[:, TRANSFORMER_COLS["idx"]])

    columns, table_rows = _rows_for(rows, "ACThreeWindingTransformer")
    if not table_rows:
        columns, table_rows = _rows_for(rows, "AC3WTransformer")
    three_winding_transformer = np.zeros(
        (len(table_rows), len(THREE_WINDING_TRANSFORMER_COLS)),
        dtype=np.float64,
    )
    if table_rows:
        cols = THREE_WINDING_TRANSFORMER_COLS
        three_winding_transformer[:, cols["idx"]] = _int_column(table_rows, columns, "idx")
        three_winding_transformer[:, cols["i_node"]] = _int_column(table_rows, columns, "i_node")
        three_winding_transformer[:, cols["j_node"]] = _int_column(table_rows, columns, "j_node")
        three_winding_transformer[:, cols["k_node"]] = _int_column(table_rows, columns, "k_node")
        i_r, i_x, j_r, j_x, k_r, k_x = _three_winding_star_impedances(table_rows, columns)
        for attr, values in (
            ("i_r", i_r),
            ("i_x", i_x),
            ("j_r", j_r),
            ("j_x", j_x),
            ("k_r", k_r),
            ("k_x", k_x),
        ):
            three_winding_transformer[:, cols[attr]] = values
        three_winding_transformer[:, cols["gt"]] = _aliased_float_column(table_rows, columns, ("gt", "g"))
        three_winding_transformer[:, cols["bt"]] = _aliased_float_column(table_rows, columns, ("bt", "b"))
        for terminal in ("i", "j", "k"):
            three_winding_transformer[:, cols[f"{terminal}_tap"]] = _aliased_float_column(
                table_rows,
                columns,
                (f"{terminal}_tap", f"tap_{terminal}"),
                1.0,
            )
            three_winding_transformer[:, cols[f"{terminal}_shift"]] = _aliased_float_column(
                table_rows,
                columns,
                (f"{terminal}_shift", f"shift_{terminal}"),
            )
        three_winding_transformer[:, cols["run_stat"]] = _float_column(
            table_rows,
            columns,
            "run_stat",
            1.0,
        )
        for attr in ("i_p", "i_q", "j_p", "j_q", "k_p", "k_q"):
            _assign_power_if_present(
                three_winding_transformer,
                cols[attr],
                table_rows,
                columns,
                attr,
                p_base,
            )
        for terminal in ("i", "j", "k"):
            _assign_current_if_present(
                three_winding_transformer,
                cols[f"{terminal}_c"],
                table_rows,
                columns,
                f"{terminal}_c",
                three_winding_transformer[:, cols[f"{terminal}_node"]],
                get_current_scale_by_node,
            )
    three_winding_transformer_names = _names_from_rows(
        table_rows,
        columns,
        "three_winding_transformer",
        three_winding_transformer[:, THREE_WINDING_TRANSFORMER_COLS["idx"]],
    )

    columns, table_rows = _rows_for(rows, "ACGenerator")
    gen = np.zeros((len(table_rows), len(GEN_COLS)), dtype=np.float64, order="F")
    rated_capacity = np.empty(0, dtype=np.float64)
    if table_rows:
        _preload_numeric_columns(
            table_rows,
            columns,
            ("idx", "node", "p_set", "q_set", "v_set", "alpha", "run_stat", "p_max", "p_min", "q_max", "q_min", "rated_capacity", "p", "q", "current"),
        )
        gen[:, GEN_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        gen[:, GEN_COLS["node"]] = _int_column(table_rows, columns, "node")
        gen[:, GEN_COLS["control_type"]] = _code_column(table_rows, columns, "control_type", CTRL_CODE, "PQ")
        gen[:, GEN_COLS["p_set"]] = _float_column(table_rows, columns, "p_set") / p_base
        gen[:, GEN_COLS["q_set"]] = _float_column(table_rows, columns, "q_set") / p_base
        gen[:, GEN_COLS["v_set"]] = _voltage_set_column(table_rows, columns, "v_set", gen[:, GEN_COLS["node"]], raw_vbase_by_idx)
        gen[:, GEN_COLS["alpha"]] = _float_column(table_rows, columns, "alpha", 1.0)
        gen[:, GEN_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        gen_p_max = _float_column(table_rows, columns, "p_max", np.nan) / p_base
        rated_capacity = _float_column(table_rows, columns, "rated_capacity", np.nan) / p_base
        p_max_placeholder = (
            np.isfinite(gen_p_max)
            & (np.abs(gen_p_max) <= 1e-12)
            & np.isfinite(rated_capacity)
            & (rated_capacity > 0.0)
        )
        if np.any(p_max_placeholder):
            gen_p_max = gen_p_max.copy()
            gen_p_max[p_max_placeholder] = rated_capacity[p_max_placeholder]
        gen[:, GEN_COLS["p_max"]] = gen_p_max
        _assign_power_if_present(gen, GEN_COLS["p"], table_rows, columns, "p", p_base)
        _assign_power_if_present(gen, GEN_COLS["q"], table_rows, columns, "q", p_base)
        _assign_current_if_present(
            gen,
            GEN_COLS["current"],
            table_rows,
            columns,
            "current",
            gen[:, GEN_COLS["node"]],
            get_current_scale_by_node,
        )
    gen_names = _names_from_rows(table_rows, columns, "gen", gen[:, GEN_COLS["idx"]])
    gen_p_min = _float_column(table_rows, columns, "p_min", np.nan) / p_base
    gen_q_min = _float_column(table_rows, columns, "q_min", np.nan) / p_base
    gen_q_max = _float_column(table_rows, columns, "q_max", np.nan) / p_base
    # 复用已解析的 rated_capacity，避免重复转换
    q_limit_placeholder = (
        np.isfinite(gen_q_min)
        & np.isfinite(gen_q_max)
        & (np.abs(gen_q_min) <= 1e-12)
        & (np.abs(gen_q_max) <= 1e-12)
        & np.isfinite(rated_capacity)
        & (rated_capacity > 0.0)
    )
    if np.any(q_limit_placeholder):
        gen_q_min = gen_q_min.copy()
        gen_q_max = gen_q_max.copy()
        gen_q_min[q_limit_placeholder] = -rated_capacity[q_limit_placeholder]
        gen_q_max[q_limit_placeholder] = rated_capacity[q_limit_placeholder]

    columns, table_rows = _rows_for(rows, "ACLoad")
    load = np.zeros((len(table_rows), len(LOAD_COLS)), dtype=np.float64, order="F")
    if table_rows:
        _preload_numeric_columns(
            table_rows,
            columns,
            ("idx", "node", "p_set", "pbase", "pv0", "pv1", "pv2", "qbase", "qv0", "qv1", "qv2", "run_stat", "p", "q", "current"),
        )
        load[:, LOAD_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        load[:, LOAD_COLS["node"]] = _int_column(table_rows, columns, "node")
        load[:, LOAD_COLS["pbase"]] = _aliased_float_column(
            table_rows,
            columns,
            ("p_set", "pbase"),
            1.0,
        ) / p_base
        load[:, LOAD_COLS["pv0"]] = _float_column(table_rows, columns, "pv0")
        load[:, LOAD_COLS["pv1"]] = _float_column(table_rows, columns, "pv1")
        load[:, LOAD_COLS["pv2"]] = _float_column(table_rows, columns, "pv2")
        load[:, LOAD_COLS["qbase"]] = _float_column(table_rows, columns, "qbase", 1.0) / p_base
        load[:, LOAD_COLS["qv0"]] = _float_column(table_rows, columns, "qv0")
        load[:, LOAD_COLS["qv1"]] = _float_column(table_rows, columns, "qv1")
        load[:, LOAD_COLS["qv2"]] = _float_column(table_rows, columns, "qv2")
        load[:, LOAD_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        _assign_power_if_present(load, LOAD_COLS["p"], table_rows, columns, "p", p_base)
        _assign_power_if_present(load, LOAD_COLS["q"], table_rows, columns, "q", p_base)
        _assign_current_if_present(
            load,
            LOAD_COLS["current"],
            table_rows,
            columns,
            "current",
            load[:, LOAD_COLS["node"]],
            get_current_scale_by_node,
        )
    load_names = _names_from_rows(table_rows, columns, "load", load[:, LOAD_COLS["idx"]])

    columns, table_rows = _rows_for(rows, "ACShuntCompensator")
    shunt = np.zeros((len(table_rows), len(SHUNT_COLS)), dtype=np.float64, order="F")
    if table_rows:
        _preload_numeric_columns(
            table_rows,
            columns,
            ("idx", "node", "q_set", "g_set", "b_set", "v_set", "run_stat", "p", "q", "current"),
        )
        shunt[:, SHUNT_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        shunt[:, SHUNT_COLS["node"]] = _int_column(table_rows, columns, "node")
        shunt[:, SHUNT_COLS["control_type"]] = _code_column(table_rows, columns, "control_type", SHUNT_CODE, "Q")
        shunt[:, SHUNT_COLS["q_set"]] = _float_column(table_rows, columns, "q_set") / p_base
        shunt[:, SHUNT_COLS["g_set"]] = _float_column(table_rows, columns, "g_set")
        shunt[:, SHUNT_COLS["b_set"]] = _float_column(table_rows, columns, "b_set")
        shunt[:, SHUNT_COLS["v_set"]] = _voltage_set_column(table_rows, columns, "v_set", shunt[:, SHUNT_COLS["node"]], raw_vbase_by_idx)
        shunt[:, SHUNT_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        _assign_power_if_present(shunt, SHUNT_COLS["p"], table_rows, columns, "p", p_base)
        _assign_power_if_present(shunt, SHUNT_COLS["q"], table_rows, columns, "q", p_base)
        _assign_current_if_present(
            shunt,
            SHUNT_COLS["current"],
            table_rows,
            columns,
            "current",
            shunt[:, SHUNT_COLS["node"]],
            get_current_scale_by_node,
        )
    shunt_names = _names_from_rows(table_rows, columns, "shunt", shunt[:, SHUNT_COLS["idx"]])

    columns, table_rows = _rows_for(rows, "ACZeroBranch")
    zero_branch = np.zeros((len(table_rows), len(ZERO_BRANCH_COLS)), dtype=np.float64, order="F")
    if table_rows:
        _preload_numeric_columns(table_rows, columns, ("idx", "i_node", "j_node", "run_stat", "p", "q", "current"))
        zero_branch[:, ZERO_BRANCH_COLS["idx"]] = _int_column(table_rows, columns, "idx")
        zero_branch[:, ZERO_BRANCH_COLS["i_node"]] = _int_column(table_rows, columns, "i_node")
        zero_branch[:, ZERO_BRANCH_COLS["j_node"]] = _int_column(table_rows, columns, "j_node")
        zero_branch[:, ZERO_BRANCH_COLS["run_stat"]] = _float_column(table_rows, columns, "run_stat", 1.0)
        _assign_power_if_present(zero_branch, ZERO_BRANCH_COLS["p"], table_rows, columns, "p", p_base)
        _assign_power_if_present(zero_branch, ZERO_BRANCH_COLS["q"], table_rows, columns, "q", p_base)
        _assign_current_if_present(
            zero_branch,
            ZERO_BRANCH_COLS["current"],
            table_rows,
            columns,
            "current",
            zero_branch[:, ZERO_BRANCH_COLS["i_node"]],
            get_current_scale_by_node,
        )
    zero_branch_names = _names_from_rows(table_rows, columns, "zero_branch", zero_branch[:, ZERO_BRANCH_COLS["idx"]])

    sw_columns, sw_rows = _rows_for(rows, "ACSwitch")
    switch, switch_names = _build_switch_like_from_rows(
        sw_rows,
        sw_columns,
        get_current_scale_by_node,
        prefix="switch",
        p_base=p_base,
    )
    br_columns, br_rows = _rows_for(rows, "ACBreak")
    breaker, breaker_names = _build_switch_like_from_rows(
        br_rows,
        br_columns,
        get_current_scale_by_node,
        prefix="break",
        scale_optional_power=False,
        p_base=p_base,
    )

    acac_columns, acac_rows = _rows_for(rows, "ACACConverter")
    acac, acac_names = _build_acac_from_rows(
        acac_rows,
        acac_columns,
        raw_vbase_by_idx,
        get_current_scale_by_node,
        p_base,
    )

    ppc = {
        "format": "ac_ppc_v1",
        "source": str(source),
        "base": {
            "p_base": float(p_base),
            "u_scale": float(u_scale),
            "p_scale": float(p_scale),
            "i_scale": float(i_scale),
            "p_base_kW": float(p_base_kW),
        },
        "bus": bus,
        "branch": branch,
        "transformer": transformer,
        "three_winding_transformer": three_winding_transformer,
        "gen": gen,
        "_gen_p_min": gen_p_min,
        "_gen_q_min": gen_q_min,
        "_gen_q_max": gen_q_max,
        "load": load,
        "shunt": shunt,
        "zero_branch": zero_branch,
        "switch": switch,
        "break": breaker,
        "acac": acac,
        "bus_name": bus_names,
        "bus_cols": BUS_COLS,
        "branch_cols": BRANCH_COLS,
        "transformer_cols": TRANSFORMER_COLS,
        "three_winding_transformer_cols": THREE_WINDING_TRANSFORMER_COLS,
        "gen_cols": GEN_COLS,
        "load_cols": LOAD_COLS,
        "shunt_cols": SHUNT_COLS,
        "zero_branch_cols": ZERO_BRANCH_COLS,
        "switch_cols": SWITCH_COLS,
        "break_cols": BREAK_COLS,
        "acac_cols": ACAC_COLS,
        "ctrl": {"PQ": CTRL_PQ, "P": CTRL_P, "PV": CTRL_PV, "SLACK": CTRL_SLACK},
        "shunt_ctrl": {"Q": SHUNT_Q, "V": SHUNT_V, "B": SHUNT_B, "Z": SHUNT_Z},
        "branch_name": branch_names,
        "transformer_name": transformer_names,
        "three_winding_transformer_name": three_winding_transformer_names,
        "gen_name": gen_names,
        "load_name": load_names,
        "shunt_name": shunt_names,
        "zero_branch_name": zero_branch_names,
        "switch_name": switch_names,
        "break_name": breaker_names,
        "acac_name": acac_names,
    }
    # 常规全合法设定点快速短路，异常时走完整 sanitize_ac_voltage_setpoints
    need_sanitize = bool(acac.shape[0])
    if not need_sanitize and gen.shape[0]:
        gv = gen[:, GEN_COLS["v_set"]]
        need_sanitize = not bool(np.all((gv >= MIN_VOLTAGE_SETPOINT) & (gv <= MAX_VOLTAGE_SETPOINT)))
    if not need_sanitize and shunt.shape[0]:
        sv_ctrl = shunt[:, SHUNT_COLS["control_type"]] == SHUNT_V
        if np.any(sv_ctrl):
            sv = shunt[sv_ctrl, SHUNT_COLS["v_set"]]
            need_sanitize = not bool(np.all((sv >= MIN_VOLTAGE_SETPOINT) & (sv <= MAX_VOLTAGE_SETPOINT)))
    if need_sanitize:
        sanitize_ac_voltage_setpoints(ppc)
    ppc["_setpoints_preverified"] = True
    if (
        three_winding_transformer.shape[0] == 0
        and zero_branch.shape[0] == 0
        and switch.shape[0] == 0
        and breaker.shape[0] == 0
        and acac.shape[0] == 0
    ):
        bus_ids_32 = bus[:, BUS_COLS["idx"]].astype(np.int32, copy=False)
        bus_run_col = bus[:, BUS_COLS["run_stat"]]
        bus_active = (bus_run_col >= 1.0) & (bus_run_col < 2.0)
        node_lookup, _ = _make_node_lookup_and_unique(bus_ids_32)
        empty_term = TerminalDeviceTopologyInput(_EMPTY_INT, _EMPTY_INT, _EMPTY_BOOL)
        br_term = (
            TerminalDeviceTopologyInput(
                _map_node_positions(branch[:, BRANCH_COLS["i_node"]], node_lookup),
                _map_node_positions(branch[:, BRANCH_COLS["j_node"]], node_lookup),
                (branch[:, BRANCH_COLS["run_stat"]] >= 1.0) & (branch[:, BRANCH_COLS["run_stat"]] < 2.0),
            )
            if branch.shape[0]
            else empty_term
        )
        xf_term = (
            TerminalDeviceTopologyInput(
                _map_node_positions(transformer[:, TRANSFORMER_COLS["i_node"]], node_lookup),
                _map_node_positions(transformer[:, TRANSFORMER_COLS["j_node"]], node_lookup),
                (transformer[:, TRANSFORMER_COLS["run_stat"]] >= 1.0)
                & (transformer[:, TRANSFORMER_COLS["run_stat"]] < 2.0),
            )
            if transformer.shape[0]
            else empty_term
        )
        gen_single = (
            SingleDeviceTopologyInput(
                _map_node_positions(gen[:, GEN_COLS["node"]], node_lookup),
                (gen[:, GEN_COLS["run_stat"]] >= 1.0) & (gen[:, GEN_COLS["run_stat"]] < 2.0),
            )
            if gen.shape[0]
            else SingleDeviceTopologyInput(_EMPTY_INT, _EMPTY_BOOL)
        )
        load_single = (
            SingleDeviceTopologyInput(
                _map_node_positions(load[:, LOAD_COLS["node"]], node_lookup),
                (load[:, LOAD_COLS["run_stat"]] >= 1.0) & (load[:, LOAD_COLS["run_stat"]] < 2.0),
            )
            if load.shape[0]
            else SingleDeviceTopologyInput(_EMPTY_INT, _EMPTY_BOOL)
        )
        shunt_single = (
            SingleDeviceTopologyInput(
                _map_node_positions(shunt[:, SHUNT_COLS["node"]], node_lookup),
                (shunt[:, SHUNT_COLS["run_stat"]] >= 1.0) & (shunt[:, SHUNT_COLS["run_stat"]] < 2.0),
            )
            if shunt.shape[0]
            else SingleDeviceTopologyInput(_EMPTY_INT, _EMPTY_BOOL)
        )
        ppc["_topology_input"] = GridTopologyInput(
            bus_ids_32,
            bus_active,
            node_lookup,
            terminals={
                "branch": br_term,
                "transformer": xf_term,
                "three_winding_transformer_ij": empty_term,
                "three_winding_transformer_ik": empty_term,
                "zero_branch": empty_term,
                "switch": empty_term,
                "break": empty_term,
                "acac": empty_term,
            },
            singles={"gen": gen_single, "load": load_single, "shunt": shunt_single},
        )
    else:
        ppc["_topology_input"] = build_ac_topology_input_ppc(ppc)
    return ppc


def build_ac_ppc_from_e_file(file_path) -> Dict:
    """Build a NumPy dictionary for AC power-flow fast paths.

    The layout is MATPOWER-like for buses, branches, generators, and loads, but
    transformer rows use this project's T-type model: ``gt`` and ``bt`` are
    single-ended i-side shunt admittance terms, not MATPOWER ``BR_B`` charging.
    The returned arrays are in pu/radians and should be treated as read-only by
    callers.
    """
    source = resolve_project_file(file_path).resolve()
    return build_ac_ppc_from_efile_rows(source, _read_efile_rows(source))


def build_ac_ppc_from_efile_rows(file_path, rows, *, run_states_propagated: bool = False) -> Dict:
    """Build AC ppc from E rows that are already loaded in memory."""
    source = resolve_project_file(file_path).resolve()
    effective_rows, overrides = (rows, []) if run_states_propagated else propagate_composite_run_states(rows)
    ppc = _build_ac_ppc_from_rows_dict(effective_rows, source)
    ppc["source"] = str(source)
    ppc["_effective_run_state_overrides"] = overrides
    return ppc


def _base_mva_from_ac_ppc(ppc: Dict) -> float:
    base = ppc.get("base", {})
    if isinstance(base, dict):
        return float(base.get("p_base", 1.0))
    arr = np.asarray(base, dtype=np.float64).ravel()
    return float(arr[0]) if arr.size else 1.0


def _matpower_empty(cols: int) -> np.ndarray:
    return np.zeros((0, int(cols)), dtype=np.float64)


def _matpower_matrix(value: Any, min_cols: int, *, required: bool = False) -> np.ndarray:
    if value is None:
        if required:
            raise KeyError("MATPOWER case is missing a required matrix")
        return _matpower_empty(min_cols)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return _matpower_empty(min_cols)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    else:
        arr = np.atleast_2d(arr)
    if arr.shape[1] < min_cols:
        if required:
            raise ValueError(f"MATPOWER matrix has {arr.shape[1]} columns; expected at least {min_cols}")
        padded = np.zeros((arr.shape[0], min_cols), dtype=np.float64)
        padded[:, : arr.shape[1]] = arr
        arr = padded
    return np.asarray(arr, dtype=np.float64)


def _matpower_scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(np.asarray(arr, dtype=np.float64).ravel()[0])


def _matpower_struct_to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray) and value.dtype.names:
        item = value.reshape(-1)[0]
        return {name: item[name] for name in value.dtype.names}
    if isinstance(value, np.ndarray) and value.size == 1:
        return _matpower_struct_to_dict(value.reshape(-1)[0])
    if isinstance(value, np.void) and value.dtype.names:
        return {name: value[name] for name in value.dtype.names}
    field_names = getattr(value, "_fieldnames", None)
    if field_names:
        return {name: getattr(value, name) for name in field_names}
    raise ValueError("MAT file does not contain a MATPOWER/PYPOWER ppc structure")


def _load_matpower_case_from_mat_file(file_path, variable_name: str = "mpc") -> Dict[str, Any]:
    from scipy.io import loadmat

    path = Path(file_path)
    data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if variable_name in data:
        return _matpower_struct_to_dict(data[variable_name])
    if {"baseMVA", "bus", "gen", "branch"}.issubset(data.keys()):
        return data
    for key, value in data.items():
        if key.startswith("__"):
            continue
        try:
            candidate = _matpower_struct_to_dict(value)
        except ValueError:
            continue
        if {"baseMVA", "bus", "gen", "branch"}.issubset(candidate.keys()):
            return candidate
    raise KeyError(f"MAT file {path} does not contain variable {variable_name!r} or ppc fields")


def _strip_matpower_comment(line: str) -> str:
    in_quote = False
    out = []
    for char in line:
        if char == "'":
            in_quote = not in_quote
        if char == "%" and not in_quote:
            break
        out.append(char)
    return "".join(out)


def _parse_matpower_numeric_matrix(text: str, name: str) -> np.ndarray:
    import re

    pattern = re.compile(rf"mpc\.{re.escape(name)}\s*=\s*\[(.*?)\]\s*;", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if match is None:
        return _matpower_empty(0)
    rows = []
    for raw_row in match.group(1).split(";"):
        cleaned = " ".join(_strip_matpower_comment(line) for line in raw_row.splitlines()).replace(",", " ").strip()
        if not cleaned:
            continue
        rows.append([float(value) for value in cleaned.split()])
    if not rows:
        return _matpower_empty(0)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"MATPOWER mpc.{name} matrix rows are not rectangular")
    return np.asarray(rows, dtype=np.float64)


def _load_matpower_case_from_m_file(file_path, variable_name: str = "mpc") -> Dict[str, Any]:
    import re

    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    if variable_name != "mpc":
        text = re.sub(rf"\b{re.escape(variable_name)}\.", "mpc.", text)
    no_comments = "\n".join(_strip_matpower_comment(line) for line in text.splitlines())
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([^;]+);", no_comments, re.IGNORECASE)
    if base_match is None:
        raise KeyError(f"MATPOWER file {path} does not define mpc.baseMVA")
    version_match = re.search(r"mpc\.version\s*=\s*'([^']+)'", no_comments, re.IGNORECASE)
    return {
        "version": version_match.group(1) if version_match is not None else "2",
        "baseMVA": float(base_match.group(1).strip()),
        "bus": _parse_matpower_numeric_matrix(text, "bus"),
        "gen": _parse_matpower_numeric_matrix(text, "gen"),
        "branch": _parse_matpower_numeric_matrix(text, "branch"),
    }


def _active_rows(rows: np.ndarray, run_col: int) -> np.ndarray:
    if rows.size == 0:
        return rows
    return rows[rows[:, run_col] == 1]


class _MatpowerDSU:
    def __init__(self, size: int):
        self.parent = np.arange(int(size), dtype=np.int64)

    def find(self, value: int) -> int:
        value = int(value)
        while int(self.parent[value]) != value:
            self.parent[value] = self.parent[int(self.parent[value])]
            value = int(self.parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _build_matpower_components(ac_ppc: Dict) -> Tuple[np.ndarray, List[List[int]], Dict[int, int], np.ndarray]:
    bus = np.asarray(ac_ppc["bus"], dtype=np.float64)
    zero = np.asarray(ac_ppc.get("zero_branch", _empty(len(ZERO_BRANCH_COLS))), dtype=np.float64)
    switch = np.asarray(ac_ppc.get("switch", _empty(len(SWITCH_COLS))), dtype=np.float64)
    breaker = np.asarray(ac_ppc.get("break", _empty(len(BREAK_COLS))), dtype=np.float64)

    node_ids = bus[:, BUS_COLS["idx"]].astype(np.int64, copy=False)
    row_by_node = {int(node): row for row, node in enumerate(node_ids)}
    active_bus = bus[:, BUS_COLS["run_stat"]] == 1
    dsu = _MatpowerDSU(bus.shape[0])

    for row in _active_rows(zero, ZERO_BRANCH_COLS["run_stat"]):
        left = row_by_node.get(int(row[ZERO_BRANCH_COLS["i_node"]]))
        right = row_by_node.get(int(row[ZERO_BRANCH_COLS["j_node"]]))
        if left is not None and right is not None and active_bus[left] and active_bus[right]:
            dsu.union(left, right)

    if switch.size:
        live = (switch[:, SWITCH_COLS["run_stat"]] == 1) & (
            switch[:, SWITCH_COLS["closed_status"]] == 1
        )
        for row in switch[live]:
            left = row_by_node.get(int(row[SWITCH_COLS["i_node"]]))
            right = row_by_node.get(int(row[SWITCH_COLS["j_node"]]))
            if left is not None and right is not None and active_bus[left] and active_bus[right]:
                dsu.union(left, right)

    if breaker.size:
        live = (breaker[:, BREAK_COLS["run_stat"]] == 1) & (
            breaker[:, BREAK_COLS["closed_status"]] == 1
        )
        for row in breaker[live]:
            left = row_by_node.get(int(row[BREAK_COLS["i_node"]]))
            right = row_by_node.get(int(row[BREAK_COLS["j_node"]]))
            if left is not None and right is not None and active_bus[left] and active_bus[right]:
                dsu.union(left, right)

    root_to_comp: Dict[int, int] = {}
    comp_rows: List[List[int]] = []
    for row in np.flatnonzero(active_bus):
        root = dsu.find(int(row))
        if root not in root_to_comp:
            root_to_comp[root] = len(comp_rows)
            comp_rows.append([])
        comp_rows[root_to_comp[root]].append(int(row))

    row_to_comp = np.full(bus.shape[0], -1, dtype=np.int64)
    for comp, rows in enumerate(comp_rows):
        row_to_comp[rows] = comp
    comp_to_bus_id = np.arange(1, len(comp_rows) + 1, dtype=np.int64)
    row_to_bus_id = np.where(row_to_comp >= 0, comp_to_bus_id[np.maximum(row_to_comp, 0)], -1)
    return row_to_comp, comp_rows, row_by_node, row_to_bus_id


def build_matpower_ppc_from_ac_ppc(ac_ppc: Dict) -> Dict[str, Any]:
    """Project this project's AC ppc into a MATPOWER/PYPOWER v2 ppc.

    Zero branches, closed switches, and closed breakers are collapsed into one
    MATPOWER bus. ZIP loads are exported as static P/Q at V=1. Transformer
    ``gt/bt`` grounding admittance is converted to an equivalent i-side bus
    shunt referred through the tap magnitude. ACAC converter terminal powers
    are projected to MATPOWER bus loads or generator rows: positive terminal
    P/Q is a PQ load, while injected or voltage-controlled terminals become
    PQ/PV/reference generator rows.
    """

    ensure_ac_ppc_gen_columns(ac_ppc)
    base_mva = _base_mva_from_ac_ppc(ac_ppc)
    bus0 = np.asarray(ac_ppc["bus"], dtype=np.float64)
    branch0 = np.asarray(ac_ppc.get("branch", _empty(len(BRANCH_COLS))), dtype=np.float64)
    transformer0 = np.asarray(ac_ppc.get("transformer", _empty(len(TRANSFORMER_COLS))), dtype=np.float64)
    three_winding_transformer0 = np.asarray(
        ac_ppc.get(
            "three_winding_transformer",
            _empty(len(THREE_WINDING_TRANSFORMER_COLS)),
        ),
        dtype=np.float64,
    )
    gen0 = np.asarray(ac_ppc.get("gen", _empty(len(GEN_COLS))), dtype=np.float64)
    gen_p_min0 = _optional_ppc_vector(ac_ppc, "_gen_p_min", gen0.shape[0])
    gen_q_min0 = _optional_ppc_vector(ac_ppc, "_gen_q_min", gen0.shape[0])
    gen_q_max0 = _optional_ppc_vector(ac_ppc, "_gen_q_max", gen0.shape[0])
    load0 = np.asarray(ac_ppc.get("load", _empty(len(LOAD_COLS))), dtype=np.float64)
    shunt0 = np.asarray(ac_ppc.get("shunt", _empty(len(SHUNT_COLS))), dtype=np.float64)
    acac0 = np.asarray(ac_ppc.get("acac", _empty(len(ACAC_COLS))), dtype=np.float64)

    row_to_comp, comp_rows, row_by_node, row_to_bus_id = _build_matpower_components(ac_ppc)
    comp_count = len(comp_rows)
    comp_to_bus_id = np.arange(1, comp_count + 1, dtype=np.int64)

    pd = np.zeros(comp_count, dtype=np.float64)
    qd = np.zeros(comp_count, dtype=np.float64)
    gs = np.zeros(comp_count, dtype=np.float64)
    bs = np.zeros(comp_count, dtype=np.float64)
    bus_type = np.full(comp_count, MP_PQ, dtype=np.float64)
    base_kv = np.zeros(comp_count, dtype=np.float64)
    vm0 = np.ones(comp_count, dtype=np.float64)
    va0 = np.zeros(comp_count, dtype=np.float64)

    for comp, rows in enumerate(comp_rows):
        first = rows[0]
        base_kv[comp] = bus0[first, BUS_COLS["vbase"]]
        vm0[comp] = bus0[first, BUS_COLS["voltage"]]
        va0[comp] = np.degrees(bus0[first, BUS_COLS["angle"]])

    for row in _active_rows(load0, LOAD_COLS["run_stat"]):
        bus_row = row_by_node.get(int(row[LOAD_COLS["node"]]))
        if bus_row is None or row_to_comp[bus_row] < 0:
            continue
        comp = int(row_to_comp[bus_row])
        pd[comp] += row[LOAD_COLS["pbase"]] * (row[LOAD_COLS["pv0"]] + row[LOAD_COLS["pv1"]] + row[LOAD_COLS["pv2"]]) * base_mva
        qd[comp] += row[LOAD_COLS["qbase"]] * (row[LOAD_COLS["qv0"]] + row[LOAD_COLS["qv1"]] + row[LOAD_COLS["qv2"]]) * base_mva

    for row in _active_rows(shunt0, SHUNT_COLS["run_stat"]):
        bus_row = row_by_node.get(int(row[SHUNT_COLS["node"]]))
        if bus_row is None or row_to_comp[bus_row] < 0:
            continue
        comp = int(row_to_comp[bus_row])
        gs[comp] += row[SHUNT_COLS["g_set"]] * base_mva
        bs[comp] += row[SHUNT_COLS["b_set"]] * base_mva

    for row in _active_rows(transformer0, TRANSFORMER_COLS["run_stat"]):
        bus_row = row_by_node.get(int(row[TRANSFORMER_COLS["i_node"]]))
        if bus_row is None or row_to_comp[bus_row] < 0:
            continue
        comp = int(row_to_comp[bus_row])
        tap = row[TRANSFORMER_COLS["tap"]]
        tap_mag = tap if abs(tap) > 1e-12 else 1.0
        scale = 1.0 / (tap_mag * tap_mag)
        gs[comp] += row[TRANSFORMER_COLS["gt"]] * scale * base_mva
        bs[comp] += row[TRANSFORMER_COLS["bt"]] * scale * base_mva

    active_three_winding = []
    zero_arm_three_winding = []
    for row in _active_rows(
        three_winding_transformer0,
        THREE_WINDING_TRANSFORMER_COLS["run_stat"],
    ):
        cols3 = THREE_WINDING_TRANSFORMER_COLS
        terminal_bus_rows = [
            row_by_node.get(int(row[cols3[f"{terminal}_node"]]))
            for terminal in ("i", "j", "k")
        ]
        if any(bus_row is None or row_to_comp[bus_row] < 0 for bus_row in terminal_bus_rows):
            continue
        terminal_bus_ids = [int(row_to_bus_id[bus_row]) for bus_row in terminal_bus_rows]
        i_comp = int(row_to_comp[terminal_bus_rows[0]])
        i_tap = float(row[cols3["i_tap"]])
        i_tap = i_tap if abs(i_tap) > 1e-12 else 1.0
        scale = 1.0 / (i_tap * i_tap)
        gs[i_comp] += row[cols3["gt"]] * scale * base_mva
        bs[i_comp] += row[cols3["bt"]] * scale * base_mva

        winding_z = np.asarray(
            [
                complex(row[cols3[f"{terminal}_r"]], row[cols3[f"{terminal}_x"]])
                for terminal in ("i", "j", "k")
            ],
            dtype=np.complex128,
        )
        zero_arm = np.abs(winding_z) <= 1e-12
        zero_count = int(np.count_nonzero(zero_arm))
        if zero_count > 1:
            raise ValueError(
                "ACThreeWindingTransformer cannot contain more than one zero-impedance winding "
                f"during MATPOWER export; device idx={int(row[cols3['idx']])}"
            )
        if zero_count == 1:
            zero_arm_three_winding.append(
                (row, terminal_bus_ids, winding_z, int(np.flatnonzero(zero_arm)[0]))
            )
            continue

        star_bus_id = comp_count + len(active_three_winding) + 1
        active_three_winding.append((row, terminal_bus_rows, terminal_bus_ids, star_bus_id))

    for row in _active_rows(gen0, GEN_COLS["run_stat"]):
        bus_row = row_by_node.get(int(row[GEN_COLS["node"]]))
        if bus_row is None or row_to_comp[bus_row] < 0:
            continue
        comp = int(row_to_comp[bus_row])
        control = int(row[GEN_COLS["control_type"]])
        if control == CTRL_SLACK:
            bus_type[comp] = MP_REF
        elif bus_type[comp] != MP_REF and control in (CTRL_PV, CTRL_P):
            bus_type[comp] = MP_PV

    acac_gen_rows = []

    def add_acac_terminal_power(node_id: float, p: float, q: float, control_kind: str, voltage_set: float) -> None:
        bus_row = row_by_node.get(int(node_id))
        if bus_row is None or row_to_comp[bus_row] < 0:
            return
        comp = int(row_to_comp[bus_row])
        kind = str(control_kind).upper()
        is_voltage_source = kind in {"PV", "PH"}
        is_source = is_voltage_source or p < -1e-12
        if is_source:
            gen_row = np.zeros(21, dtype=np.float64)
            gen_row[MP_GEN_BUS] = row_to_bus_id[bus_row]
            gen_row[MP_PG] = -p * base_mva
            gen_row[MP_QG] = -q * base_mva
            gen_row[MP_QMAX] = 1e9
            gen_row[MP_QMIN] = -1e9
            gen_row[MP_VG] = voltage_set if abs(voltage_set) > 1e-12 else vm0[comp]
            gen_row[MP_MBASE] = base_mva
            gen_row[MP_GEN_STATUS] = 1
            gen_row[MP_PMAX] = 1e9
            gen_row[MP_PMIN] = -1e9
            acac_gen_rows.append(gen_row)
            if kind == "PH":
                bus_type[comp] = MP_REF
            elif kind == "PV" and bus_type[comp] != MP_REF:
                bus_type[comp] = MP_PV
            return
        if abs(p) <= 1e-12 and abs(q) <= 1e-12:
            return
        pd[comp] += p * base_mva
        qd[comp] += q * base_mva

    for row in _active_rows(acac0, ACAC_COLS["run_stat"]):
        i_control = ACAC_SIDE_CONTROL_LABEL.get(int(row[ACAC_COLS["i_control_type"]]), "PQ")
        j_control = ACAC_SIDE_CONTROL_LABEL.get(int(row[ACAC_COLS["j_control_type"]]), "PQ")
        i_p = float(row[ACAC_COLS["i_p"]])
        i_q = float(row[ACAC_COLS["i_q"]])
        j_p = float(row[ACAC_COLS["j_p"]])
        j_q = float(row[ACAC_COLS["j_q"]])
        if abs(i_p) <= 1e-12 and abs(j_p) <= 1e-12 and abs(row[ACAC_COLS["p_set"]]) > 1e-12:
            i_p = float(row[ACAC_COLS["p_set"]])
            j_p = -i_p
        if i_control == "PQ" and abs(i_q) <= 1e-12 and abs(row[ACAC_COLS["i_q_set"]]) > 1e-12:
            i_q = float(row[ACAC_COLS["i_q_set"]])
        if j_control == "PQ" and abs(j_q) <= 1e-12 and abs(row[ACAC_COLS["j_q_set"]]) > 1e-12:
            j_q = float(row[ACAC_COLS["j_q_set"]])
        add_acac_terminal_power(
            row[ACAC_COLS["i_node"]],
            i_p,
            i_q,
            i_control,
            float(row[ACAC_COLS["i_v_set"]]),
        )
        add_acac_terminal_power(
            row[ACAC_COLS["j_node"]],
            j_p,
            j_q,
            j_control,
            float(row[ACAC_COLS["j_v_set"]]),
        )

    bus = np.zeros((comp_count, 13), dtype=np.float64)
    bus[:, MP_BUS_I] = comp_to_bus_id
    bus[:, MP_BUS_TYPE] = bus_type
    bus[:, MP_PD] = pd
    bus[:, MP_QD] = qd
    bus[:, MP_GS] = gs
    bus[:, MP_BS] = bs
    bus[:, MP_BUS_AREA] = 1
    bus[:, MP_VM] = vm0
    bus[:, MP_VA] = va0
    bus[:, MP_BASE_KV] = base_kv
    bus[:, MP_ZONE] = 1
    bus[:, MP_VMAX] = 1.2
    bus[:, MP_VMIN] = 0.8
    if active_three_winding:
        star_rows = np.zeros((len(active_three_winding), 13), dtype=np.float64)
        for pos, (_row, terminal_bus_rows, _terminal_bus_ids, star_bus_id) in enumerate(active_three_winding):
            source_comp = int(row_to_comp[terminal_bus_rows[0]])
            star_rows[pos, MP_BUS_I] = star_bus_id
            star_rows[pos, MP_BUS_TYPE] = MP_PQ
            star_rows[pos, MP_BUS_AREA] = 1
            star_rows[pos, MP_VM] = vm0[source_comp]
            star_rows[pos, MP_VA] = va0[source_comp]
            star_rows[pos, MP_BASE_KV] = base_kv[source_comp]
            star_rows[pos, MP_ZONE] = 1
            star_rows[pos, MP_VMAX] = 1.2
            star_rows[pos, MP_VMIN] = 0.8
        bus = np.vstack((bus, star_rows))

    gen_rows = []
    for gen_pos, row in enumerate(gen0):
        if int(row[GEN_COLS["run_stat"]]) != 1:
            continue
        bus_row = row_by_node.get(int(row[GEN_COLS["node"]]))
        if bus_row is None or row_to_comp[bus_row] < 0:
            continue
        gen_row = np.zeros(21, dtype=np.float64)
        gen_row[MP_GEN_BUS] = row_to_bus_id[bus_row]
        gen_row[MP_PG] = row[GEN_COLS["p_set"]] * base_mva
        gen_row[MP_QG] = row[GEN_COLS["q_set"]] * base_mva
        q_max = gen_q_max0[gen_pos]
        q_min = gen_q_min0[gen_pos]
        gen_row[MP_QMAX] = q_max * base_mva if np.isfinite(q_max) else 1e9
        gen_row[MP_QMIN] = q_min * base_mva if np.isfinite(q_min) else -1e9
        gen_row[MP_VG] = row[GEN_COLS["v_set"]]
        gen_row[MP_MBASE] = base_mva
        gen_row[MP_GEN_STATUS] = 1
        p_max = row[GEN_COLS["p_max"]]
        gen_row[MP_PMAX] = p_max * base_mva if np.isfinite(p_max) else 1e9
        p_min = gen_p_min0[gen_pos]
        gen_row[MP_PMIN] = p_min * base_mva if np.isfinite(p_min) else -1e9
        gen_rows.append(gen_row)
    gen_rows.extend(acac_gen_rows)
    gen = np.vstack(gen_rows) if gen_rows else np.zeros((0, 21), dtype=np.float64)

    branch_rows = []

    def add_branch_devices(devices: np.ndarray, cols: Dict[str, int], is_transformer: bool) -> None:
        for row in devices:
            if int(row[cols["run_stat"]]) != 1:
                continue
            i_row = row_by_node.get(int(row[cols["i_node"]]))
            j_row = row_by_node.get(int(row[cols["j_node"]]))
            if i_row is None or j_row is None or row_to_comp[i_row] < 0 or row_to_comp[j_row] < 0:
                continue
            i_bus = row_to_bus_id[i_row]
            j_bus = row_to_bus_id[j_row]
            if i_bus == j_bus:
                continue
            branch_row = np.zeros(13, dtype=np.float64)
            branch_row[MP_F_BUS] = i_bus
            branch_row[MP_T_BUS] = j_bus
            branch_row[MP_BR_R] = row[cols["r"]]
            branch_row[MP_BR_X] = row[cols["x"]]
            branch_row[MP_BR_B] = 0.0 if is_transformer else row[cols["b"]]
            branch_row[MP_RATE_A] = 0.0
            branch_row[MP_RATE_B] = 0.0
            branch_row[MP_RATE_C] = 0.0
            if is_transformer:
                tap = row[cols["tap"]]
                branch_row[MP_TAP] = 0.0 if abs(tap - 1.0) < 1e-12 else tap
                branch_row[MP_SHIFT] = row[cols["shift"]]
            branch_row[MP_BR_STATUS] = 1
            branch_row[MP_ANGMIN] = -360.0
            branch_row[MP_ANGMAX] = 360.0
            branch_rows.append(branch_row)

    add_branch_devices(branch0, BRANCH_COLS, False)
    add_branch_devices(transformer0, TRANSFORMER_COLS, True)
    for row, _terminal_bus_rows, terminal_bus_ids, star_bus_id in active_three_winding:
        cols3 = THREE_WINDING_TRANSFORMER_COLS
        for terminal, terminal_bus_id in zip(("i", "j", "k"), terminal_bus_ids):
            branch_row = np.zeros(13, dtype=np.float64)
            branch_row[MP_F_BUS] = terminal_bus_id
            branch_row[MP_T_BUS] = star_bus_id
            branch_row[MP_BR_R] = row[cols3[f"{terminal}_r"]]
            branch_row[MP_BR_X] = row[cols3[f"{terminal}_x"]]
            tap = row[cols3[f"{terminal}_tap"]]
            branch_row[MP_TAP] = 0.0 if abs(tap - 1.0) < 1e-12 else tap
            branch_row[MP_SHIFT] = row[cols3[f"{terminal}_shift"]]
            branch_row[MP_BR_STATUS] = 1
            branch_row[MP_ANGMIN] = -360.0
            branch_row[MP_ANGMAX] = 360.0
            branch_rows.append(branch_row)
    for row, terminal_bus_ids, winding_z, anchor in zero_arm_three_winding:
        cols3 = THREE_WINDING_TRANSFORMER_COLS
        terminal_labels = ("i", "j", "k")
        tap_complex = []
        for terminal in terminal_labels:
            tap = float(row[cols3[f"{terminal}_tap"]])
            tap = tap if abs(tap) > 1e-12 else 1.0
            shift = float(row[cols3[f"{terminal}_shift"]])
            tap_complex.append(tap * np.exp(1j * np.deg2rad(shift)))
        anchor_tap = tap_complex[anchor]
        for terminal in range(3):
            if terminal == anchor:
                continue
            equivalent_z = winding_z[terminal] * abs(tap_complex[terminal]) ** 2
            equivalent_tap = anchor_tap / tap_complex[terminal]
            branch_row = np.zeros(13, dtype=np.float64)
            branch_row[MP_F_BUS] = terminal_bus_ids[anchor]
            branch_row[MP_T_BUS] = terminal_bus_ids[terminal]
            branch_row[MP_BR_R] = equivalent_z.real
            branch_row[MP_BR_X] = equivalent_z.imag
            branch_row[MP_RATE_A] = 0.0
            branch_row[MP_RATE_B] = 0.0
            branch_row[MP_RATE_C] = 0.0
            tap_magnitude = abs(equivalent_tap)
            branch_row[MP_TAP] = 0.0 if abs(tap_magnitude - 1.0) < 1e-12 else tap_magnitude
            branch_row[MP_SHIFT] = np.degrees(np.angle(equivalent_tap))
            branch_row[MP_BR_STATUS] = 1
            branch_row[MP_ANGMIN] = -360.0
            branch_row[MP_ANGMAX] = 360.0
            branch_rows.append(branch_row)
    branch = np.vstack(branch_rows) if branch_rows else np.zeros((0, 13), dtype=np.float64)
    return {"version": "2", "baseMVA": base_mva, "bus": bus, "gen": gen, "branch": branch}


_MAT_V5_MI_INT8 = 1
_MAT_V5_MI_UINT8 = 2
_MAT_V5_MI_INT16 = 3
_MAT_V5_MI_UINT16 = 4
_MAT_V5_MI_INT32 = 5
_MAT_V5_MI_UINT32 = 6
_MAT_V5_MI_SINGLE = 7
_MAT_V5_MI_DOUBLE = 9
_MAT_V5_MI_INT64 = 12
_MAT_V5_MI_UINT64 = 13
_MAT_V5_MI_MATRIX = 14
_MAT_V5_MI_COMPRESSED = 15

_MAT_V5_MX_STRUCT_CLASS = 2
_MAT_V5_NUMERIC_MX_CLASSES = frozenset((6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
_MAT_V5_REQUIRED_FIELDS = frozenset(("baseMVA", "bus", "gen", "branch"))
_MAT_V5_DTYPE_BY_MI = {
    _MAT_V5_MI_INT8: np.dtype("<i1"),
    _MAT_V5_MI_UINT8: np.dtype("<u1"),
    _MAT_V5_MI_INT16: np.dtype("<i2"),
    _MAT_V5_MI_UINT16: np.dtype("<u2"),
    _MAT_V5_MI_INT32: np.dtype("<i4"),
    _MAT_V5_MI_UINT32: np.dtype("<u4"),
    _MAT_V5_MI_SINGLE: np.dtype("<f4"),
    _MAT_V5_MI_DOUBLE: np.dtype("<f8"),
    _MAT_V5_MI_INT64: np.dtype("<i8"),
    _MAT_V5_MI_UINT64: np.dtype("<u8"),
}
_EMPTY_OBJ_ARRAY = np.asarray([], dtype=object)
_EMPTY_OBJ_ARRAY.setflags(write=False)


def _read_mat_v5_tag(buf, pos: int, limit: int):
    """解析 MATLAB Level-5 数据元素标签，返回 (dtype_code, nbytes, data_start, next_pos)。"""
    if pos + 8 > limit:
        return None
    word0, word1 = struct.unpack_from("<II", buf, pos)
    upper = word0 >> 16
    if upper != 0:
        # 小数据元素（Small Data Element）格式
        dtype_code = word0 & 0xFFFF
        nbytes = upper
        if nbytes > 4:
            return None
        return dtype_code, nbytes, pos + 4, pos + 8
    dtype_code = word0
    nbytes = word1
    data_start = pos + 8
    next_pos = data_start + ((nbytes + 7) & ~7)
    if data_start + nbytes > limit or next_pos > limit:
        return None
    return dtype_code, nbytes, data_start, next_pos


def _parse_mat_v5_matrix(buf, data_start: int, data_end: int):
    """解析未压缩小端 Level-5 miMATRIX，仅处理标量/实数值矩阵与 1x1 MATPOWER 结构体。"""
    tag = _read_mat_v5_tag(buf, data_start, data_end)
    if tag is None:
        return None
    dt, nb, ds, pos = tag
    if dt != _MAT_V5_MI_UINT32 or nb < 8:
        return None
    flags0 = struct.unpack_from("<I", buf, ds)[0]
    mx_class = flags0 & 0xFF
    if flags0 & 0x0800:  # 复数矩阵回退 scipy.io.loadmat
        return None

    tag = _read_mat_v5_tag(buf, pos, data_end)
    if tag is None:
        return None
    dt, nb, ds, pos = tag
    if dt != _MAT_V5_MI_INT32 or nb == 0 or (nb & 3) != 0:
        return None
    dims = struct.unpack_from(f"<{nb // 4}i", buf, ds)
    if any(d < 0 for d in dims):
        return None

    tag = _read_mat_v5_tag(buf, pos, data_end)
    if tag is None:
        return None
    dt, nb, ds, pos = tag
    if dt not in (_MAT_V5_MI_INT8, _MAT_V5_MI_UINT8):
        return None
    name = bytes(buf[ds : ds + nb]).decode("ascii", errors="ignore")

    if mx_class in _MAT_V5_NUMERIC_MX_CLASSES:
        tag = _read_mat_v5_tag(buf, pos, data_end)
        if tag is None:
            return None
        dt, nb, ds, _ = tag
        np_dt = _MAT_V5_DTYPE_BY_MI.get(dt)
        if np_dt is None:
            return None
        count = 1
        for dim in dims:
            count *= dim
        if nb < count * np_dt.itemsize:
            return None
        arr = np.frombuffer(buf, dtype=np_dt, count=count, offset=ds).reshape(dims, order="F")
        return name, arr

    if mx_class == _MAT_V5_MX_STRUCT_CLASS:
        if any(dim != 1 for dim in dims):
            return None
        tag = _read_mat_v5_tag(buf, pos, data_end)
        if tag is None:
            return None
        dt, nb, ds, pos = tag
        if dt != _MAT_V5_MI_INT32 or nb != 4:
            return None
        field_name_len = struct.unpack_from("<i", buf, ds)[0]
        if field_name_len <= 0:
            return None

        tag = _read_mat_v5_tag(buf, pos, data_end)
        if tag is None:
            return None
        dt, nb, ds, pos = tag
        if dt not in (_MAT_V5_MI_INT8, _MAT_V5_MI_UINT8) or (nb % field_name_len) != 0:
            return None
        n_fields = nb // field_name_len
        field_names = [
            bytes(buf[ds + i * field_name_len : ds + (i + 1) * field_name_len])
            .split(b"\x00", 1)[0]
            .decode("ascii", errors="ignore")
            for i in range(n_fields)
        ]

        result: Dict[str, Any] = {}
        for fname in field_names:
            ftag = _read_mat_v5_tag(buf, pos, data_end)
            if ftag is None:
                return None
            fdt, fnb, fds, pos = ftag
            if fname not in _MAT_V5_REQUIRED_FIELDS:
                continue
            if fdt != _MAT_V5_MI_MATRIX:
                return None
            parsed = _parse_mat_v5_matrix(buf, fds, fds + fnb)
            if parsed is None or not isinstance(parsed[1], np.ndarray):
                return None
            result[fname] = parsed[1]
        return name, result

    # 无关顶层变量（如字符串 version 等）标记跳过
    return name, None


def _make_node_lookup_and_unique(bus_ids_32: np.ndarray) -> Tuple[Any, bool]:
    """一次构建拓扑节点位置查找表，同时检测母线 ID 是否严格唯一。"""
    n_bus = int(bus_ids_32.size)
    if n_bus == 0:
        return np.array([], dtype=np.int32), True
    min_id = int(bus_ids_32.min())
    max_id = int(bus_ids_32.max())
    if min_id >= 0 and max_id <= max(1024, n_bus * 4):
        lookup = np.full(max_id + 1, -1, dtype=np.int32)
        lookup[bus_ids_32] = np.arange(n_bus, dtype=np.int32)
        is_unique = int(np.count_nonzero(lookup >= 0)) == n_bus
        return lookup, is_unique
    mapping = {int(node_id): pos for pos, node_id in enumerate(bus_ids_32.tolist())}
    return mapping, len(mapping) == n_bus


def _assemble_ac_ppc_from_matpower_dict(mpc: Dict[str, Any], source_str: str) -> Dict:
    """将 MATPOWER/PYPOWER 字典向量化组装为本项目 AC PPC（列主序分配并融合拓扑输入）。"""
    base_mva = _matpower_scalar(mpc.get("baseMVA"), 1.0)
    bus_mp = _matpower_matrix(mpc.get("bus"), 13, required=True)
    gen_mp = _matpower_matrix(mpc.get("gen"), 10)
    branch_mp = _matpower_matrix(mpc.get("branch"), 13)

    # 1. 母线表（列主序分配：利用 OS 虚拟零页并使所有按列读写完全连续）
    n_bus = bus_mp.shape[0]
    bus = np.zeros((n_bus, len(BUS_COLS)), dtype=np.float64, order="F")
    if n_bus:
        bus_ids = bus_mp[:, MP_BUS_I].astype(np.int64)
        bus_ids_32 = bus_ids.astype(np.int32)
        bus_types = bus_mp[:, MP_BUS_TYPE].astype(np.int8)
        bus_active = bus_types != MP_NONE
        bus[:, BUS_COLS["idx"]] = bus_ids
        bus[:, BUS_COLS["vbase"]] = bus_mp[:, MP_BASE_KV]
        bus[:, BUS_COLS["voltage"]] = bus_mp[:, MP_VM]
        np.deg2rad(bus_mp[:, MP_VA], out=bus[:, BUS_COLS["angle"]])
        bus[:, BUS_COLS["run_stat"]] = bus_active
    else:
        bus_ids = np.empty(0, dtype=np.int64)
        bus_ids_32 = _EMPTY_INT
        bus_types = np.empty(0, dtype=np.int8)
        bus_active = _EMPTY_BOOL
    bus_names = LazyNameArray(None, bus_ids, "bus")

    # 一次构建 node_lookup 并同时供发电机控制类型与 GridTopologyInput 共用
    node_lookup, bus_ids_unique = _make_node_lookup_and_unique(bus_ids_32)

    # 2. 负荷表（跳过停运母线及零负荷母线）
    if n_bus:
        pd_col = bus_mp[:, MP_PD]
        qd_col = bus_mp[:, MP_QD]
        load_pos = np.flatnonzero(bus_active & ((np.abs(pd_col) > 1e-12) | (np.abs(qd_col) > 1e-12)))
        n_load = int(load_pos.size)
    else:
        load_pos = _EMPTY_INT
        n_load = 0
    if n_load:
        load_idx = np.arange(1, n_load + 1, dtype=np.int64)
        load = np.zeros((n_load, len(LOAD_COLS)), dtype=np.float64, order="F")
        load_nodes = bus_ids[load_pos]
        load[:, LOAD_COLS["idx"]] = load_idx
        load[:, LOAD_COLS["node"]] = load_nodes
        np.divide(pd_col[load_pos], base_mva, out=load[:, LOAD_COLS["pbase"]])
        load[:, LOAD_COLS["pv0"]] = 1.0
        np.divide(qd_col[load_pos], base_mva, out=load[:, LOAD_COLS["qbase"]])
        load[:, LOAD_COLS["qv0"]] = 1.0
        load[:, LOAD_COLS["run_stat"]] = 1.0
    else:
        load_idx = np.empty(0, dtype=np.int64)
        load = _empty(len(LOAD_COLS))
        load_nodes = np.empty(0, dtype=np.int64)
    load_names = LazyNameArray(None, load_idx, "load")

    # 3. 母线并联索引（稍后与变压器末端充电电纳并联合并分配）
    if n_bus:
        gs_col = bus_mp[:, MP_GS]
        bs_col = bus_mp[:, MP_BS]
        bus_shunt_pos = np.flatnonzero(
            bus_active & ((np.abs(gs_col) > 1e-12) | (np.abs(bs_col) > 1e-12))
        )
        n_bus_shunt = int(bus_shunt_pos.size)
    else:
        bus_shunt_pos = _EMPTY_INT
        n_bus_shunt = 0

    # 4. 发电机表（复用 node_lookup 查母线类型并同步产出 gen_node_pos）
    n_gen = gen_mp.shape[0]
    if n_gen:
        gen = np.zeros((n_gen, len(GEN_COLS)), dtype=np.float64, order="F")
        gen_idx = np.arange(1, n_gen + 1, dtype=np.int64)
        gen_bus_raw = gen_mp[:, MP_GEN_BUS]
        if n_bus:
            gen_node_pos = _map_node_positions(gen_bus_raw, node_lookup)
            valid_g = gen_node_pos >= 0
            if valid_g.all():
                gen_bus_types = bus_types[gen_node_pos]
            else:
                gen_bus_types = np.full(n_gen, MP_PQ, dtype=np.int8)
                if valid_g.any():
                    gen_bus_types[valid_g] = bus_types[gen_node_pos[valid_g]]
        else:
            gen_node_pos = np.full(n_gen, -1, dtype=np.int32)
            gen_bus_types = np.full(n_gen, MP_PQ, dtype=np.int8)

        gen_ctrl = gen[:, GEN_COLS["control_type"]]
        gen_ctrl.fill(CTRL_PQ)
        gen_ctrl[gen_bus_types == MP_PV] = CTRL_PV
        gen_ctrl[gen_bus_types == MP_REF] = CTRL_SLACK

        p_pu = gen[:, GEN_COLS["p_set"]]
        q_pu = gen[:, GEN_COLS["q_set"]]
        np.divide(gen_mp[:, MP_PG], base_mva, out=p_pu)
        np.divide(gen_mp[:, MP_QG], base_mva, out=q_pu)
        vg_col = gen_mp[:, MP_VG]

        gen[:, GEN_COLS["idx"]] = gen_idx
        gen[:, GEN_COLS["node"]] = gen_bus_raw
        gen[:, GEN_COLS["v_set"]] = np.where(np.abs(vg_col) > 1e-12, vg_col, 1.0)
        gen[:, GEN_COLS["alpha"]] = 1.0
        if gen_mp.shape[1] > MP_GEN_STATUS:
            gen[:, GEN_COLS["run_stat"]] = gen_mp[:, MP_GEN_STATUS]
        else:
            gen[:, GEN_COLS["run_stat"]] = 1.0
        gen[:, GEN_COLS["p"]] = p_pu
        gen[:, GEN_COLS["q"]] = q_pu
        if gen_mp.shape[1] > MP_PMAX:
            np.divide(gen_mp[:, MP_PMAX], base_mva, out=gen[:, GEN_COLS["p_max"]])
        else:
            gen[:, GEN_COLS["p_max"]] = np.nan
    else:
        gen = _empty(len(GEN_COLS))
        gen_idx = np.empty(0, dtype=np.int64)
        gen_node_pos = _EMPTY_INT

    gen_p_min = (
        gen_mp[:, MP_PMIN] / base_mva
        if gen_mp.shape[1] > MP_PMIN
        else np.full(n_gen, np.nan, dtype=np.float64)
    )
    gen_q_min = (
        gen_mp[:, MP_QMIN] / base_mva
        if gen_mp.shape[1] > MP_QMIN
        else np.full(n_gen, np.nan, dtype=np.float64)
    )
    gen_q_max = (
        gen_mp[:, MP_QMAX] / base_mva
        if gen_mp.shape[1] > MP_QMAX
        else np.full(n_gen, np.nan, dtype=np.float64)
    )
    gen_names = LazyNameArray(None, gen_idx, "gen")

    # 5. 普通线路、变压器与变压器末端充电电纳并联（先切 1D 连续列，再用 flatnonzero + take）
    n_br = branch_mp.shape[0]
    if n_br:
        br_ncols = branch_mp.shape[1]
        f_bus_col = branch_mp[:, MP_F_BUS]
        t_bus_col = branch_mp[:, MP_T_BUS]
        r_col = branch_mp[:, MP_BR_R]
        x_col = branch_mp[:, MP_BR_X]
        status_col = (
            branch_mp[:, MP_BR_STATUS]
            if br_ncols > MP_BR_STATUS
            else np.ones(n_br, dtype=np.float64)
        )
        if br_ncols > MP_TAP:
            raw_tap = branch_mp[:, MP_TAP]
            tap_col = np.where(np.abs(raw_tap) > 1e-12, raw_tap, 1.0)
        else:
            tap_col = np.ones(n_br, dtype=np.float64)
        shift_col = (
            branch_mp[:, MP_SHIFT]
            if br_ncols > MP_SHIFT
            else np.zeros(n_br, dtype=np.float64)
        )
        charging_col = (
            branch_mp[:, MP_BR_B]
            if br_ncols > MP_BR_B
            else np.zeros(n_br, dtype=np.float64)
        )

        xfmr_mask = (np.abs(tap_col - 1.0) > 1e-12) | (np.abs(shift_col) > 1e-12)
        xfmr_pos = np.flatnonzero(xfmr_mask)
        n_xfmr = int(xfmr_pos.size)
        n_line = n_br - n_xfmr

        if n_line:
            branch_idx = np.arange(1, n_line + 1, dtype=np.int64)
            branch = np.zeros((n_line, len(BRANCH_COLS)), dtype=np.float64, order="F")
            branch[:, BRANCH_COLS["idx"]] = branch_idx
            if n_xfmr == 0:
                branch[:, BRANCH_COLS["i_node"]] = f_bus_col
                branch[:, BRANCH_COLS["j_node"]] = t_bus_col
                branch[:, BRANCH_COLS["r"]] = r_col
                branch[:, BRANCH_COLS["x"]] = x_col
                branch[:, BRANCH_COLS["b"]] = charging_col
                branch[:, BRANCH_COLS["run_stat"]] = status_col
            else:
                line_pos = np.flatnonzero(~xfmr_mask)
                np.take(f_bus_col, line_pos, out=branch[:, BRANCH_COLS["i_node"]])
                np.take(t_bus_col, line_pos, out=branch[:, BRANCH_COLS["j_node"]])
                np.take(r_col, line_pos, out=branch[:, BRANCH_COLS["r"]])
                np.take(x_col, line_pos, out=branch[:, BRANCH_COLS["x"]])
                np.take(charging_col, line_pos, out=branch[:, BRANCH_COLS["b"]])
                np.take(status_col, line_pos, out=branch[:, BRANCH_COLS["run_stat"]])
        else:
            branch_idx = np.empty(0, dtype=np.int64)
            branch = _empty(len(BRANCH_COLS))

        if n_xfmr:
            xfmr_idx = np.arange(1, n_xfmr + 1, dtype=np.int64)
            transformer = np.zeros((n_xfmr, len(TRANSFORMER_COLS)), dtype=np.float64, order="F")
            transformer[:, TRANSFORMER_COLS["idx"]] = xfmr_idx
            np.take(f_bus_col, xfmr_pos, out=transformer[:, TRANSFORMER_COLS["i_node"]])
            np.take(t_bus_col, xfmr_pos, out=transformer[:, TRANSFORMER_COLS["j_node"]])
            np.take(r_col, xfmr_pos, out=transformer[:, TRANSFORMER_COLS["r"]])
            np.take(x_col, xfmr_pos, out=transformer[:, TRANSFORMER_COLS["x"]])
            xf_ch = charging_col[xfmr_pos]
            np.multiply(xf_ch, 0.5, out=transformer[:, TRANSFORMER_COLS["bt"]])
            np.take(tap_col, xfmr_pos, out=transformer[:, TRANSFORMER_COLS["tap"]])
            np.take(shift_col, xfmr_pos, out=transformer[:, TRANSFORMER_COLS["shift"]])
            np.take(status_col, xfmr_pos, out=transformer[:, TRANSFORMER_COLS["run_stat"]])

            xf_shunt_sub = np.flatnonzero(np.abs(xf_ch) > 1e-12)
            n_xfmr_shunt = int(xf_shunt_sub.size)
        else:
            xfmr_idx = np.empty(0, dtype=np.int64)
            transformer = _empty(len(TRANSFORMER_COLS))
            xf_ch = np.empty(0, dtype=np.float64)
            xf_shunt_sub = _EMPTY_INT
            n_xfmr_shunt = 0
    else:
        branch_idx = np.empty(0, dtype=np.int64)
        branch = _empty(len(BRANCH_COLS))
        xfmr_idx = np.empty(0, dtype=np.int64)
        transformer = _empty(len(TRANSFORMER_COLS))
        xf_ch = np.empty(0, dtype=np.float64)
        xf_shunt_sub = _EMPTY_INT
        n_xfmr_shunt = 0

    # 6. 合并分配并联表
    n_shunt = n_bus_shunt + n_xfmr_shunt
    if n_shunt:
        shunt_idx = np.arange(1, n_shunt + 1, dtype=np.int64)
        shunt = np.zeros((n_shunt, len(SHUNT_COLS)), dtype=np.float64, order="F")
        shunt[:, SHUNT_COLS["idx"]] = shunt_idx
        shunt[:, SHUNT_COLS["control_type"]] = SHUNT_B
        if n_bus_shunt:
            shunt[:n_bus_shunt, SHUNT_COLS["node"]] = bus_ids[bus_shunt_pos]
            np.divide(gs_col[bus_shunt_pos], base_mva, out=shunt[:n_bus_shunt, SHUNT_COLS["g_set"]])
            np.divide(bs_col[bus_shunt_pos], base_mva, out=shunt[:n_bus_shunt, SHUNT_COLS["b_set"]])
            shunt[:n_bus_shunt, SHUNT_COLS["run_stat"]] = 1.0
        if n_xfmr_shunt:
            shunt[n_bus_shunt:, SHUNT_COLS["node"]] = transformer[xf_shunt_sub, TRANSFORMER_COLS["j_node"]]
            np.multiply(xf_ch[xf_shunt_sub], 0.5, out=shunt[n_bus_shunt:, SHUNT_COLS["b_set"]])
            shunt[n_bus_shunt:, SHUNT_COLS["run_stat"]] = transformer[xf_shunt_sub, TRANSFORMER_COLS["run_stat"]]
    else:
        shunt_idx = np.empty(0, dtype=np.int64)
        shunt = _empty(len(SHUNT_COLS))

    ppc = {
        "format": "ac_ppc_v1",
        "source": source_str,
        "base": {
            "p_base": float(base_mva),
            "u_scale": 1.0,
            "p_scale": 0.001,
            "i_scale": 1.0,
            "p_base_kW": float(base_mva / 0.001),
        },
        "bus": bus,
        "branch": branch,
        "transformer": transformer,
        "three_winding_transformer": _empty(len(THREE_WINDING_TRANSFORMER_COLS)),
        "gen": gen,
        "_gen_p_min": gen_p_min,
        "_gen_q_min": gen_q_min,
        "_gen_q_max": gen_q_max,
        "load": load,
        "shunt": shunt,
        "zero_branch": _empty(len(ZERO_BRANCH_COLS)),
        "switch": _empty(len(SWITCH_COLS)),
        "break": _empty(len(BREAK_COLS)),
        "acac": _empty(len(ACAC_COLS)),
        "bus_name": bus_names,
        "branch_name": LazyNameArray(None, branch_idx, "branch"),
        "transformer_name": LazyNameArray(None, xfmr_idx, "transformer"),
        "three_winding_transformer_name": _EMPTY_OBJ_ARRAY,
        "gen_name": gen_names,
        "load_name": load_names,
        "shunt_name": LazyNameArray(None, shunt_idx, "shunt"),
        "zero_branch_name": _EMPTY_OBJ_ARRAY,
        "switch_name": _EMPTY_OBJ_ARRAY,
        "break_name": _EMPTY_OBJ_ARRAY,
        "acac_name": _EMPTY_OBJ_ARRAY,
        "bus_cols": BUS_COLS,
        "branch_cols": BRANCH_COLS,
        "transformer_cols": TRANSFORMER_COLS,
        "three_winding_transformer_cols": THREE_WINDING_TRANSFORMER_COLS,
        "gen_cols": GEN_COLS,
        "load_cols": LOAD_COLS,
        "shunt_cols": SHUNT_COLS,
        "zero_branch_cols": ZERO_BRANCH_COLS,
        "switch_cols": SWITCH_COLS,
        "break_cols": BREAK_COLS,
        "acac_cols": ACAC_COLS,
        "ctrl": {"PQ": CTRL_PQ, "P": CTRL_P, "PV": CTRL_PV, "SLACK": CTRL_SLACK},
        "shunt_ctrl": {"Q": SHUNT_Q, "V": SHUNT_V, "B": SHUNT_B, "Z": SHUNT_Z},
    }

    # 7. 电压设定点校验快速短路（shunt 全为 SHUNT_B、acac 为空，仅需检查 gen v_set 合法区间）
    if n_gen:
        v_set_col = gen[:, GEN_COLS["v_set"]]
        if not bool(np.all((v_set_col >= MIN_VOLTAGE_SETPOINT) & (v_set_col <= MAX_VOLTAGE_SETPOINT))):
            sanitize_ac_voltage_setpoints(ppc)
    ppc["_setpoints_preverified"] = True

    # 8. 融合构造 GridTopologyInput（复用已有 node_lookup，唯一母线 ID 时 load/shunt 零查表）
    empty_term = TerminalDeviceTopologyInput(_EMPTY_INT, _EMPTY_INT, _EMPTY_BOOL)
    if branch.shape[0]:
        br_run_col = branch[:, BRANCH_COLS["run_stat"]]
        br_term = TerminalDeviceTopologyInput(
            _map_node_positions(branch[:, BRANCH_COLS["i_node"]], node_lookup),
            _map_node_positions(branch[:, BRANCH_COLS["j_node"]], node_lookup),
            (br_run_col >= 1.0) & (br_run_col < 2.0),
        )
    else:
        br_term = empty_term

    if transformer.shape[0]:
        xf_j_pos = _map_node_positions(transformer[:, TRANSFORMER_COLS["j_node"]], node_lookup)
        xf_run_col = transformer[:, TRANSFORMER_COLS["run_stat"]]
        xf_term = TerminalDeviceTopologyInput(
            _map_node_positions(transformer[:, TRANSFORMER_COLS["i_node"]], node_lookup),
            xf_j_pos,
            (xf_run_col >= 1.0) & (xf_run_col < 2.0),
        )
    else:
        xf_j_pos = _EMPTY_INT
        xf_term = empty_term

    if n_gen:
        gen_run_col = gen[:, GEN_COLS["run_stat"]]
        gen_single = SingleDeviceTopologyInput(
            gen_node_pos,
            (gen_run_col >= 1.0) & (gen_run_col < 2.0),
        )
    else:
        gen_single = SingleDeviceTopologyInput(_EMPTY_INT, _EMPTY_BOOL)

    if n_load:
        load_node_pos = (
            load_pos.astype(np.int32)
            if bus_ids_unique
            else _map_node_positions(load_nodes, node_lookup)
        )
        load_single = SingleDeviceTopologyInput(load_node_pos, np.ones(n_load, dtype=bool))
    else:
        load_single = SingleDeviceTopologyInput(_EMPTY_INT, _EMPTY_BOOL)

    if n_shunt:
        if bus_ids_unique:
            if n_xfmr_shunt == 0:
                sh_node_pos = bus_shunt_pos.astype(np.int32)
                sh_run_mask = np.ones(n_bus_shunt, dtype=bool)
            elif n_bus_shunt == 0:
                sh_node_pos = xf_j_pos[xf_shunt_sub]
                sh_run_col = shunt[:, SHUNT_COLS["run_stat"]]
                sh_run_mask = (sh_run_col >= 1.0) & (sh_run_col < 2.0)
            else:
                sh_node_pos = np.empty(n_shunt, dtype=np.int32)
                sh_node_pos[:n_bus_shunt] = bus_shunt_pos
                sh_node_pos[n_bus_shunt:] = xf_j_pos[xf_shunt_sub]
                sh_run_col = shunt[:, SHUNT_COLS["run_stat"]]
                sh_run_mask = (sh_run_col >= 1.0) & (sh_run_col < 2.0)
        else:
            sh_node_pos = _map_node_positions(shunt[:, SHUNT_COLS["node"]], node_lookup)
            sh_run_col = shunt[:, SHUNT_COLS["run_stat"]]
            sh_run_mask = (sh_run_col >= 1.0) & (sh_run_col < 2.0)
        shunt_single = SingleDeviceTopologyInput(sh_node_pos, sh_run_mask)
    else:
        shunt_single = SingleDeviceTopologyInput(_EMPTY_INT, _EMPTY_BOOL)

    ppc["_topology_input"] = GridTopologyInput(
        bus_ids_32,
        bus_active,
        node_lookup,
        terminals={
            "branch": br_term,
            "transformer": xf_term,
            "three_winding_transformer_ij": empty_term,
            "three_winding_transformer_ik": empty_term,
            "zero_branch": empty_term,
            "switch": empty_term,
            "break": empty_term,
            "acac": empty_term,
        },
        singles={
            "gen": gen_single,
            "load": load_single,
            "shunt": shunt_single,
        },
    )
    return ppc


def _try_assemble_ac_ppc_from_mat_v5(path: Path, variable_name: str = "mpc") -> Any:
    """对标准未压缩小端 MATLAB Level-5 .mat 文件执行 mmap 零拷贝直读并组装 PPC；不匹配时返回 None 回退。"""
    try:
        file_size = path.stat().st_size
    except OSError:
        return None
    if file_size < 128:
        return None

    with open(path, "rb") as fp:
        with mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            version, endian = struct.unpack_from("<H2s", mm, 124)
            if version != 0x0100 or endian != b"IM":
                return None

            pos = 128
            selected_mpc = None
            top_arrays: Dict[str, np.ndarray] = {}
            fallback_structs: List[Dict[str, Any]] = []
            parsed = None
            vval = None
            try:
                while pos + 8 <= file_size:
                    tag = _read_mat_v5_tag(mm, pos, file_size)
                    if tag is None:
                        return None
                    dt, nb, ds, next_pos = tag
                    if dt == _MAT_V5_MI_COMPRESSED:
                        return None
                    if dt == _MAT_V5_MI_MATRIX:
                        parsed = _parse_mat_v5_matrix(mm, ds, ds + nb)
                        if parsed is None:
                            return None
                        vname, vval = parsed
                        if vname == variable_name and isinstance(vval, dict):
                            if "bus" not in vval:
                                return None
                            selected_mpc = vval
                            break
                        if isinstance(vval, dict):
                            if _MAT_V5_REQUIRED_FIELDS.issubset(vval.keys()):
                                fallback_structs.append(vval)
                        elif isinstance(vval, np.ndarray) and vname in _MAT_V5_REQUIRED_FIELDS:
                            top_arrays[vname] = vval
                    pos = next_pos

                if selected_mpc is None:
                    if _MAT_V5_REQUIRED_FIELDS.issubset(top_arrays.keys()):
                        selected_mpc = top_arrays
                    elif fallback_structs:
                        selected_mpc = fallback_structs[0]

                if selected_mpc is None:
                    return None

                # 在 mmap 有效期内完成全部独立数组组装，退出前通过 finally 清理全部 buffer view
                return _assemble_ac_ppc_from_matpower_dict(selected_mpc, str(path))
            finally:
                selected_mpc = None
                top_arrays.clear()
                fallback_structs.clear()
                parsed = None
                vval = None


def build_ac_ppc_from_mat_file(file_path, *, variable_name: str = "mpc") -> Dict:
    """Read a MATPOWER/PYPOWER ``.mat`` case and build this project's AC ppc.

    The input must provide ``baseMVA``, ``bus``, ``gen`` and ``branch`` either
    inside a MATLAB struct named by ``variable_name`` or as top-level MAT
    variables. Text MATPOWER ``.m`` files using ``mpc.*`` assignments are also
    accepted. Bus loads and shunts are converted into ACLoad/ACShunt rows.
    MATPOWER branches with tap or phase shift are converted into ACTransformer
    rows; ordinary branches stay in the ACBranch table.
    """

    path = Path(file_path).resolve()
    if path.suffix.lower() != ".m":
        fast_ppc = _try_assemble_ac_ppc_from_mat_v5(path, variable_name=variable_name)
        if fast_ppc is not None:
            return fast_ppc
        mpc = _load_matpower_case_from_mat_file(path, variable_name=variable_name)
    else:
        mpc = _load_matpower_case_from_m_file(path, variable_name=variable_name)
    return _assemble_ac_ppc_from_matpower_dict(mpc, str(path))


def save_ac_ppc_to_mat_file(ppc: Dict, file_path, *, variable_name: str = "mpc") -> Path:
    """Save this project's AC ppc as a MATPOWER/PYPOWER ``.mat`` case."""

    from scipy.io import savemat

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matpower_ppc = build_matpower_ppc_from_ac_ppc(ppc)
    savemat(str(path), {variable_name: matpower_ppc}, do_compression=True, oned_as="row")
    return path


def build_ac_ppc_from_network(network) -> Dict:
    """Build an AC ppc dictionary from an already loaded ACPowerNetwork."""
    nodes = list(getattr(network, "nodes", []))
    branches = list(getattr(network, "branches", []))
    transformers = list(getattr(network, "transformers", []))
    three_winding_transformers = list(getattr(network, "three_winding_transformers", []))
    generators = list(getattr(network, "generators", []))
    loads = list(getattr(network, "loads", []))
    shunts = list(getattr(network, "shunt_compensators", []))
    zero_branches = list(getattr(network, "zero_branches", []))
    switches = list(getattr(network, "switches", []))
    breakers = list(getattr(network, "breakers", []))
    acac_converters = list(getattr(network, "acac_converters", []))

    p_base = float(getattr(network, "p_base", 1.0))
    u_scale = float(getattr(network, "u_scale", 1.0))
    p_scale = float(getattr(network, "p_scale", 1.0))
    i_scale = float(getattr(network, "i_scale", 1.0))

    bus = np.zeros((len(nodes), len(BUS_COLS)), dtype=np.float64)
    for row, node in enumerate(nodes):
        bus[row, BUS_COLS["idx"]] = _int_value(node, "idx")
        bus[row, BUS_COLS["vbase"]] = _float_value(node, "vbase")
        bus[row, BUS_COLS["voltage"]] = _float_value(node, "voltage", 1.0)
        bus[row, BUS_COLS["angle"]] = _float_value(node, "angle", 0.0)
        bus[row, BUS_COLS["isl"]] = _float_value(node, "isl", 0.0)
        bus[row, BUS_COLS["run_stat"]] = _float_value(node, "run_stat", 1.0)
    bus_names = _name_array(nodes, "bus")

    branch = np.zeros((len(branches), len(BRANCH_COLS)), dtype=np.float64)
    for row, dev in enumerate(branches):
        branch[row, BRANCH_COLS["idx"]] = _int_value(dev, "idx")
        branch[row, BRANCH_COLS["i_node"]] = _int_value(dev, "i_node")
        branch[row, BRANCH_COLS["j_node"]] = _int_value(dev, "j_node")
        branch[row, BRANCH_COLS["r"]] = _float_value(dev, "r")
        branch[row, BRANCH_COLS["x"]] = _float_value(dev, "x")
        branch[row, BRANCH_COLS["b"]] = _float_value(dev, "b")
        branch[row, BRANCH_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
    for attr in ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c"):
        _fill_float_column_if_present(branch, branches, BRANCH_COLS[attr], attr)

    transformer = np.zeros((len(transformers), len(TRANSFORMER_COLS)), dtype=np.float64)
    for row, dev in enumerate(transformers):
        transformer[row, TRANSFORMER_COLS["idx"]] = _int_value(dev, "idx")
        transformer[row, TRANSFORMER_COLS["i_node"]] = _int_value(dev, "i_node")
        transformer[row, TRANSFORMER_COLS["j_node"]] = _int_value(dev, "j_node")
        transformer[row, TRANSFORMER_COLS["r"]] = _float_value(dev, "r")
        transformer[row, TRANSFORMER_COLS["x"]] = _float_value(dev, "x")
        transformer[row, TRANSFORMER_COLS["gt"]] = _float_value(dev, "gt")
        transformer[row, TRANSFORMER_COLS["bt"]] = _float_value(dev, "bt", _float_value(dev, "b") / 2.0)
        transformer[row, TRANSFORMER_COLS["tap"]] = _float_value(dev, "tap", 1.0)
        transformer[row, TRANSFORMER_COLS["shift"]] = _float_value(dev, "shift")
        transformer[row, TRANSFORMER_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
    for attr in ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c"):
        _fill_float_column_if_present(transformer, transformers, TRANSFORMER_COLS[attr], attr)

    three_winding_transformer = np.zeros(
        (len(three_winding_transformers), len(THREE_WINDING_TRANSFORMER_COLS)),
        dtype=np.float64,
    )
    tw_cols = THREE_WINDING_TRANSFORMER_COLS
    for row, dev in enumerate(three_winding_transformers):
        for attr in ("idx", "i_node", "j_node", "k_node"):
            three_winding_transformer[row, tw_cols[attr]] = _int_value(dev, attr)
        for attr in (
            "i_r",
            "i_x",
            "j_r",
            "j_x",
            "k_r",
            "k_x",
            "gt",
            "bt",
            "i_shift",
            "j_shift",
            "k_shift",
        ):
            three_winding_transformer[row, tw_cols[attr]] = _float_value(dev, attr)
        for attr in ("i_tap", "j_tap", "k_tap"):
            three_winding_transformer[row, tw_cols[attr]] = _float_value(dev, attr, 1.0)
        three_winding_transformer[row, tw_cols["run_stat"]] = _float_value(dev, "run_stat", 1.0)
    for attr in ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c", "k_p", "k_q", "k_c"):
        _fill_float_column_if_present(
            three_winding_transformer,
            three_winding_transformers,
            tw_cols[attr],
            attr,
        )

    gen = np.zeros((len(generators), len(GEN_COLS)), dtype=np.float64)
    gen_p_min = np.full(len(generators), np.nan, dtype=np.float64)
    gen_q_min = np.full(len(generators), np.nan, dtype=np.float64)
    gen_q_max = np.full(len(generators), np.nan, dtype=np.float64)
    for row, dev in enumerate(generators):
        gen[row, GEN_COLS["idx"]] = _int_value(dev, "idx")
        gen[row, GEN_COLS["node"]] = _int_value(dev, "node")
        gen[row, GEN_COLS["control_type"]] = _code_value(_value(dev, "control_type", "PQ"), CTRL_CODE, "PQ")
        gen[row, GEN_COLS["p_set"]] = _float_value(dev, "p_set")
        gen[row, GEN_COLS["q_set"]] = _float_value(dev, "q_set")
        gen[row, GEN_COLS["v_set"]] = _float_value(dev, "v_set", 1.0)
        gen[row, GEN_COLS["alpha"]] = _float_value(dev, "alpha", 1.0)
        gen[row, GEN_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
        rated_capacity = _float_value(dev, "rated_capacity", np.nan)
        p_max = _float_value(dev, "p_max", np.nan)
        if np.isfinite(p_max) and abs(p_max) <= 1e-12 and np.isfinite(rated_capacity) and rated_capacity > 0.0:
            p_max = rated_capacity
        gen[row, GEN_COLS["p_max"]] = p_max
        gen_p_min[row] = _float_value(dev, "p_min", np.nan)
        q_min = _float_value(dev, "q_min", np.nan)
        q_max = _float_value(dev, "q_max", np.nan)
        if (
            np.isfinite(q_min)
            and np.isfinite(q_max)
            and abs(q_min) <= 1e-12
            and abs(q_max) <= 1e-12
            and np.isfinite(rated_capacity)
            and rated_capacity > 0.0
        ):
            q_min = -rated_capacity
            q_max = rated_capacity
        gen_q_min[row] = q_min
        gen_q_max[row] = q_max
    for attr in ("p", "q", "current"):
        _fill_float_column_if_present(gen, generators, GEN_COLS[attr], attr)

    load = np.zeros((len(loads), len(LOAD_COLS)), dtype=np.float64)
    for row, dev in enumerate(loads):
        load[row, LOAD_COLS["idx"]] = _int_value(dev, "idx")
        load[row, LOAD_COLS["node"]] = _int_value(dev, "node")
        load[row, LOAD_COLS["pbase"]] = _float_value(
            dev,
            "p_set",
            _float_value(dev, "pbase", 1.0),
        )
        load[row, LOAD_COLS["pv0"]] = _float_value(dev, "pv0")
        load[row, LOAD_COLS["pv1"]] = _float_value(dev, "pv1")
        load[row, LOAD_COLS["pv2"]] = _float_value(dev, "pv2")
        load[row, LOAD_COLS["qbase"]] = _float_value(dev, "qbase", 1.0)
        load[row, LOAD_COLS["qv0"]] = _float_value(dev, "qv0")
        load[row, LOAD_COLS["qv1"]] = _float_value(dev, "qv1")
        load[row, LOAD_COLS["qv2"]] = _float_value(dev, "qv2")
        load[row, LOAD_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
    for attr in ("p", "q", "current"):
        _fill_float_column_if_present(load, loads, LOAD_COLS[attr], attr)

    shunt = np.zeros((len(shunts), len(SHUNT_COLS)), dtype=np.float64)
    for row, dev in enumerate(shunts):
        shunt[row, SHUNT_COLS["idx"]] = _int_value(dev, "idx")
        shunt[row, SHUNT_COLS["node"]] = _int_value(dev, "node")
        shunt[row, SHUNT_COLS["control_type"]] = _code_value(_value(dev, "control_type", "Q"), SHUNT_CODE, "Q")
        shunt[row, SHUNT_COLS["q_set"]] = _float_value(dev, "q_set")
        shunt[row, SHUNT_COLS["g_set"]] = _float_value(dev, "g_set")
        shunt[row, SHUNT_COLS["b_set"]] = _float_value(dev, "b_set")
        shunt[row, SHUNT_COLS["v_set"]] = _float_value(dev, "v_set", 1.0)
        shunt[row, SHUNT_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
    for attr in ("p", "q", "current"):
        _fill_float_column_if_present(shunt, shunts, SHUNT_COLS[attr], attr)

    zero_branch = np.zeros((len(zero_branches), len(ZERO_BRANCH_COLS)), dtype=np.float64)
    for row, dev in enumerate(zero_branches):
        zero_branch[row, ZERO_BRANCH_COLS["idx"]] = _int_value(dev, "idx")
        zero_branch[row, ZERO_BRANCH_COLS["i_node"]] = _int_value(dev, "i_node")
        zero_branch[row, ZERO_BRANCH_COLS["j_node"]] = _int_value(dev, "j_node")
        zero_branch[row, ZERO_BRANCH_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
    for attr in ("p", "q", "current"):
        _fill_float_column_if_present(zero_branch, zero_branches, ZERO_BRANCH_COLS[attr], attr)

    def build_switch_like(devices):
        out = np.zeros((len(devices), len(SWITCH_COLS)), dtype=np.float64)
        for row, dev in enumerate(devices):
            out[row, SWITCH_COLS["idx"]] = _int_value(dev, "idx")
            out[row, SWITCH_COLS["i_node"]] = _int_value(dev, "i_node")
            out[row, SWITCH_COLS["j_node"]] = _int_value(dev, "j_node")
            status = _float_value(dev, "status", 1.0)
            out[row, SWITCH_COLS["status"]] = status
            out[row, SWITCH_COLS["closed_status_set"]] = _float_value(
                dev,
                "closed_status_set",
                status,
            )
            out[row, SWITCH_COLS["closed_status"]] = _float_value(
                dev,
                "closed_status",
                status,
            )
            out[row, SWITCH_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
        for attr in ("p", "q", "current"):
            _fill_float_column_if_present(out, devices, SWITCH_COLS[attr], attr)
        return out

    acac = np.zeros((len(acac_converters), len(ACAC_COLS)), dtype=np.float64)
    for row, dev in enumerate(acac_converters):
        acac[row, ACAC_COLS["idx"]] = _int_value(dev, "idx")
        acac[row, ACAC_COLS["i_node"]] = _int_value(dev, "i_node")
        acac[row, ACAC_COLS["j_node"]] = _int_value(dev, "j_node")
        acac[row, ACAC_COLS["r1"]] = _float_value(dev, "r1")
        acac[row, ACAC_COLS["r2"]] = _float_value(dev, "r2")
        i_ctrl = _value(dev, "i_control_type", None)
        j_ctrl = _value(dev, "j_control_type", None)
        if i_ctrl in (None, "") or j_ctrl in (None, ""):
            i_ctrl, j_ctrl = acac_control_pair_from_legacy(_value(dev, "control_type", "PQQ"))
        acac[row, ACAC_COLS["i_control_type"]] = _code_value(
            i_ctrl, ACAC_SIDE_CONTROL_PARSE_CODE, "PQ"
        )
        acac[row, ACAC_COLS["j_control_type"]] = _code_value(
            j_ctrl, ACAC_SIDE_CONTROL_PARSE_CODE, "PQ"
        )
        acac[row, ACAC_COLS["p_set"]] = _float_value(dev, "p_set")
        acac[row, ACAC_COLS["i_q_set"]] = _float_value(dev, "i_q_set")
        acac[row, ACAC_COLS["j_q_set"]] = _float_value(dev, "j_q_set")
        acac[row, ACAC_COLS["i_v_set"]] = _float_value(dev, "i_v_set")
        acac[row, ACAC_COLS["j_v_set"]] = _float_value(dev, "j_v_set")
        acac[row, ACAC_COLS["run_stat"]] = _float_value(dev, "run_stat", 1.0)
    for attr in ("i_p", "i_q", "j_p", "j_q", "i_i", "j_i"):
        _fill_float_column_if_present(acac, acac_converters, ACAC_COLS[attr], attr)

    ppc = {
        "format": "ac_ppc_v1",
        "source": str(getattr(network, "source", getattr(network, "file_name", "<network>"))),
        "base": {
            "p_base": float(p_base),
            "u_scale": float(u_scale),
            "p_scale": float(p_scale),
            "i_scale": float(i_scale),
            "p_base_kW": float(getattr(network, "p_base_kW", p_base / p_scale)),
        },
        "bus": bus,
        "branch": branch,
        "transformer": transformer,
        "three_winding_transformer": three_winding_transformer,
        "gen": gen,
        "_gen_p_min": gen_p_min,
        "_gen_q_min": gen_q_min,
        "_gen_q_max": gen_q_max,
        "load": load,
        "shunt": shunt,
        "zero_branch": zero_branch,
        "switch": build_switch_like(switches),
        "break": build_switch_like(breakers),
        "acac": acac,
        "bus_name": bus_names,
        "bus_cols": BUS_COLS,
        "branch_cols": BRANCH_COLS,
        "transformer_cols": TRANSFORMER_COLS,
        "three_winding_transformer_cols": THREE_WINDING_TRANSFORMER_COLS,
        "gen_cols": GEN_COLS,
        "load_cols": LOAD_COLS,
        "shunt_cols": SHUNT_COLS,
        "zero_branch_cols": ZERO_BRANCH_COLS,
        "switch_cols": SWITCH_COLS,
        "break_cols": BREAK_COLS,
        "acac_cols": ACAC_COLS,
        "ctrl": {"PQ": CTRL_PQ, "P": CTRL_P, "PV": CTRL_PV, "SLACK": CTRL_SLACK},
        "shunt_ctrl": {"Q": SHUNT_Q, "V": SHUNT_V, "B": SHUNT_B, "Z": SHUNT_Z},
    }
    ppc.update(
        branch_name=_name_array(branches, "branch"),
        transformer_name=_name_array(transformers, "transformer"),
        three_winding_transformer_name=_name_array(
            three_winding_transformers,
            "three_winding_transformer",
        ),
        gen_name=_name_array(generators, "gen"),
        load_name=_name_array(loads, "load"),
        shunt_name=_name_array(shunts, "shunt"),
        zero_branch_name=_name_array(zero_branches, "zero_branch"),
        switch_name=_name_array(switches, "switch"),
        break_name=_name_array(breakers, "break"),
        acac_name=_name_array(acac_converters, "acac"),
    )
    sanitize_ac_voltage_setpoints(ppc)
    ppc["_setpoints_preverified"] = True
    ppc["_topology_input"] = build_ac_topology_input_ppc(ppc)
    return ppc


def _build_ac_ppc_from_model(model, *, units_already_normalized: bool = False):
    from ac_model import ACPowerNetwork

    network = ACPowerNetwork()
    network.model = model
    network._load_from_model(units_already_normalized=units_already_normalized)
    ppc = build_ac_ppc_from_network(network)
    network.ppc = ppc
    return network, ppc


def build_ac_ppc_from_model(model):
    return _build_ac_ppc_from_model(model)


def _list_ppc_names(ppc: Dict, key: str, prefix: str, count: int) -> List[str]:
    values = ppc.get(key)
    if values is None:
        return [f"{prefix}_{idx}" for idx in range(count)]
    return [str(value) for value in values]


def build_ac_network_from_ppc(ppc: Dict):
    ensure_ac_ppc_gen_columns(ppc)
    from ac_model import (
        ACBreak,
        ACBranch,
        ACACConverter,
        ACGenerator,
        ACLoad,
        ACNode,
        ACPowerNetwork,
        ACShuntCompensator,
        ACSwitch,
        ACThreeWindingTransformer,
        ACTransformer,
        ACZeroBranch,
    )

    ctrl_name = {CTRL_PQ: "PQ", CTRL_P: "P", CTRL_PV: "PV", CTRL_SLACK: "V"}
    shunt_ctrl_name = {SHUNT_Q: "Q", SHUNT_V: "V", SHUNT_B: "B", SHUNT_Z: "Z"}
    bus_names = _list_ppc_names(ppc, "bus_name", "bus", ppc["bus"].shape[0])
    branch_names = _list_ppc_names(ppc, "branch_name", "branch", ppc["branch"].shape[0])
    transformer_names = _list_ppc_names(ppc, "transformer_name", "transformer", ppc["transformer"].shape[0])
    three_winding_table = np.asarray(
        ppc.get("three_winding_transformer", _empty(len(THREE_WINDING_TRANSFORMER_COLS))),
        dtype=np.float64,
    )
    three_winding_transformer_names = _list_ppc_names(
        ppc,
        "three_winding_transformer_name",
        "three_winding_transformer",
        three_winding_table.shape[0],
    )
    gen_names = _list_ppc_names(ppc, "gen_name", "gen", ppc["gen"].shape[0])
    gen_p_min = _optional_ppc_vector(ppc, "_gen_p_min", len(gen_names))
    gen_q_min = _optional_ppc_vector(ppc, "_gen_q_min", len(gen_names))
    gen_q_max = _optional_ppc_vector(ppc, "_gen_q_max", len(gen_names))
    load_names = _list_ppc_names(ppc, "load_name", "load", ppc["load"].shape[0])
    shunt_names = _list_ppc_names(ppc, "shunt_name", "shunt", ppc["shunt"].shape[0])
    zero_branch_names = _list_ppc_names(ppc, "zero_branch_name", "zero_branch", ppc["zero_branch"].shape[0])
    switch_names = _list_ppc_names(ppc, "switch_name", "switch", ppc["switch"].shape[0])
    break_names = _list_ppc_names(ppc, "break_name", "break", ppc.get("break", _empty(len(BREAK_COLS))).shape[0])
    acac_table = ppc.get("acac", _empty(len(ACAC_COLS)))
    acac_names = _list_ppc_names(ppc, "acac_name", "acac", acac_table.shape[0])

    network = ACPowerNetwork()
    base = ppc["base"]
    network.ppc = ppc
    network.p_base = float(base["p_base"])
    network.u_scale = float(base["u_scale"])
    network.p_scale = float(base["p_scale"])
    network.i_scale = float(base["i_scale"])
    network.p_base_kW = float(base["p_base_kW"])

    network.nodes = [
        ACNode(
            int(row[BUS_COLS["idx"]]),
            float(row[BUS_COLS["vbase"]]),
            float(row[BUS_COLS["voltage"]]),
            float(row[BUS_COLS["angle"]]),
            int(row[BUS_COLS["run_stat"]]),
        )
        for row in ppc["bus"]
    ]
    for node in network.nodes:
        node.isl = None
        node.isl_obj = None
        node.v_gens = []
        node.generators = []
        node.loads = []
        node.branches = []
        node.switches = []
        node.breakers = []
        node.zero_branches = []
        node.transformers = []
        node.three_winding_transformers = []
        node.shunt_compensators = []
        node.acac_converters = []

    network.branches = [
        ACBranch(
            int(row[BRANCH_COLS["idx"]]),
            int(row[BRANCH_COLS["i_node"]]),
            int(row[BRANCH_COLS["j_node"]]),
            float(row[BRANCH_COLS["r"]]),
            float(row[BRANCH_COLS["x"]]),
            float(row[BRANCH_COLS["b"]]),
            int(row[BRANCH_COLS["run_stat"]]),
        )
        for row in ppc["branch"]
    ]
    network.transformers = [
        # ACTransformer expects gt/bt as i-side single-ended shunt admittance.
        # Older E files using b are normalized during array construction.
        ACTransformer(
            int(row[TRANSFORMER_COLS["idx"]]),
            int(row[TRANSFORMER_COLS["i_node"]]),
            int(row[TRANSFORMER_COLS["j_node"]]),
            float(row[TRANSFORMER_COLS["r"]]),
            float(row[TRANSFORMER_COLS["x"]]),
            float(row[TRANSFORMER_COLS["tap"]]),
            float(row[TRANSFORMER_COLS["shift"]]),
            float(row[TRANSFORMER_COLS["gt"]]),
            float(row[TRANSFORMER_COLS["bt"]]),
            int(row[TRANSFORMER_COLS["run_stat"]]),
        )
        for row in ppc["transformer"]
    ]
    tw_cols = THREE_WINDING_TRANSFORMER_COLS
    network.three_winding_transformers = [
        ACThreeWindingTransformer(
            int(row[tw_cols["idx"]]),
            int(row[tw_cols["i_node"]]),
            int(row[tw_cols["j_node"]]),
            int(row[tw_cols["k_node"]]),
            float(row[tw_cols["i_r"]]),
            float(row[tw_cols["i_x"]]),
            float(row[tw_cols["j_r"]]),
            float(row[tw_cols["j_x"]]),
            float(row[tw_cols["k_r"]]),
            float(row[tw_cols["k_x"]]),
            float(row[tw_cols["i_tap"]]),
            float(row[tw_cols["i_shift"]]),
            float(row[tw_cols["j_tap"]]),
            float(row[tw_cols["j_shift"]]),
            float(row[tw_cols["k_tap"]]),
            float(row[tw_cols["k_shift"]]),
            float(row[tw_cols["gt"]]),
            float(row[tw_cols["bt"]]),
            int(row[tw_cols["run_stat"]]),
        )
        for row in three_winding_table
    ]
    network.generators = [
        ACGenerator(
            int(row[GEN_COLS["idx"]]),
            int(row[GEN_COLS["node"]]),
            ctrl_name.get(int(row[GEN_COLS["control_type"]]), "PQ"),
            float(row[GEN_COLS["p_set"]]),
            float(row[GEN_COLS["q_set"]]),
            float(row[GEN_COLS["v_set"]]),
            float(row[GEN_COLS["alpha"]]),
            int(row[GEN_COLS["run_stat"]]),
            float(row[GEN_COLS["p_max"]]) if np.isfinite(row[GEN_COLS["p_max"]]) else None,
            float(gen_p_min[pos]) if np.isfinite(gen_p_min[pos]) else None,
            float(gen_q_min[pos]) if np.isfinite(gen_q_min[pos]) else None,
            float(gen_q_max[pos]) if np.isfinite(gen_q_max[pos]) else None,
        )
        for pos, row in enumerate(ppc["gen"])
    ]
    network.loads = [
        ACLoad(
            int(row[LOAD_COLS["idx"]]),
            int(row[LOAD_COLS["node"]]),
            float(row[LOAD_COLS["pbase"]]),
            float(row[LOAD_COLS["pv0"]]),
            float(row[LOAD_COLS["pv1"]]),
            float(row[LOAD_COLS["pv2"]]),
            float(row[LOAD_COLS["qbase"]]),
            float(row[LOAD_COLS["qv0"]]),
            float(row[LOAD_COLS["qv1"]]),
            float(row[LOAD_COLS["qv2"]]),
            int(row[LOAD_COLS["run_stat"]]),
        )
        for row in ppc["load"]
    ]
    network.shunt_compensators = [
        ACShuntCompensator(
            int(row[SHUNT_COLS["idx"]]),
            int(row[SHUNT_COLS["node"]]),
            shunt_ctrl_name.get(int(row[SHUNT_COLS["control_type"]]), "Q"),
            float(row[SHUNT_COLS["q_set"]]),
            float(row[SHUNT_COLS["g_set"]]),
            float(row[SHUNT_COLS["b_set"]]),
            float(row[SHUNT_COLS["v_set"]]),
            int(row[SHUNT_COLS["run_stat"]]),
        )
        for row in ppc["shunt"]
    ]
    network.zero_branches = [
        ACZeroBranch(
            int(row[ZERO_BRANCH_COLS["idx"]]),
            int(row[ZERO_BRANCH_COLS["i_node"]]),
            int(row[ZERO_BRANCH_COLS["j_node"]]),
            int(row[ZERO_BRANCH_COLS["run_stat"]]),
        )
        for row in ppc["zero_branch"]
    ]
    network.switches = [
        ACSwitch(
            int(row[SWITCH_COLS["idx"]]),
            int(row[SWITCH_COLS["i_node"]]),
            int(row[SWITCH_COLS["j_node"]]),
            int(row[SWITCH_COLS["status"]]),
            int(row[SWITCH_COLS["run_stat"]]),
            closed_status_set=int(row[SWITCH_COLS["closed_status_set"]]),
            closed_status=int(row[SWITCH_COLS["closed_status"]]),
        )
        for row in ppc["switch"]
    ]
    network.breakers = [
        ACBreak(
            int(row[BREAK_COLS["idx"]]),
            int(row[BREAK_COLS["i_node"]]),
            int(row[BREAK_COLS["j_node"]]),
            int(row[BREAK_COLS["status"]]),
            int(row[BREAK_COLS["run_stat"]]),
            closed_status_set=int(row[BREAK_COLS["closed_status_set"]]),
            closed_status=int(row[BREAK_COLS["closed_status"]]),
        )
        for row in ppc.get("break", _empty(len(BREAK_COLS)))
    ]
    network.acac_converters = [
        ACACConverter(
            int(row[ACAC_COLS["idx"]]),
            int(row[ACAC_COLS["i_node"]]),
            int(row[ACAC_COLS["j_node"]]),
            float(row[ACAC_COLS["r1"]]),
            float(row[ACAC_COLS["r2"]]),
            ACAC_SIDE_CONTROL_LABEL.get(int(row[ACAC_COLS["i_control_type"]]), "PQ"),
            ACAC_SIDE_CONTROL_LABEL.get(int(row[ACAC_COLS["j_control_type"]]), "PQ"),
            float(row[ACAC_COLS["p_set"]]),
            float(row[ACAC_COLS["i_q_set"]]),
            float(row[ACAC_COLS["j_q_set"]]),
            float(row[ACAC_COLS["i_v_set"]]),
            float(row[ACAC_COLS["j_v_set"]]),
            int(row[ACAC_COLS["run_stat"]]),
        )
        for row in acac_table
    ]
    network.node_dict = {}
    network.switch_dict = {}
    network.break_dict = {}
    network.load_dict = {}
    network.generator_dict = {}
    network.zero_branch_dict = {}
    network.branch_dict = {}
    network.transformer_dict = {}
    network.three_winding_transformer_dict = {}
    network.shunt_compensator_dict = {}
    network.acac_converter_dict = {}
    network.islands = []
    for obj, name in zip(network.nodes, bus_names):
        obj.name = name
        obj.is_alive = False
    for obj, row, name in zip(network.branches, ppc["branch"], branch_names):
        obj.name = name
        obj.i_p = float(row[BRANCH_COLS["i_p"]])
        obj.i_q = float(row[BRANCH_COLS["i_q"]])
        obj.i_c = float(row[BRANCH_COLS["i_c"]])
        obj.j_p = float(row[BRANCH_COLS["j_p"]])
        obj.j_q = float(row[BRANCH_COLS["j_q"]])
        obj.j_c = float(row[BRANCH_COLS["j_c"]])
        obj.is_alive = False
    for obj, row, name in zip(network.transformers, ppc["transformer"], transformer_names):
        obj.name = name
        obj.i_p = float(row[TRANSFORMER_COLS["i_p"]])
        obj.i_q = float(row[TRANSFORMER_COLS["i_q"]])
        obj.i_c = float(row[TRANSFORMER_COLS["i_c"]])
        obj.j_p = float(row[TRANSFORMER_COLS["j_p"]])
        obj.j_q = float(row[TRANSFORMER_COLS["j_q"]])
        obj.j_c = float(row[TRANSFORMER_COLS["j_c"]])
        obj.is_alive = False
    for obj, row, name in zip(
        network.three_winding_transformers,
        three_winding_table,
        three_winding_transformer_names,
    ):
        obj.name = name
        for attr in ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c", "k_p", "k_q", "k_c"):
            setattr(obj, attr, float(row[tw_cols[attr]]))
        obj.is_alive = False
    for obj, row, name in zip(network.generators, ppc["gen"], gen_names):
        obj.name = name
        obj.p = float(row[GEN_COLS["p"]])
        obj.q = float(row[GEN_COLS["q"]])
        obj.current = float(row[GEN_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.loads, ppc["load"], load_names):
        obj.name = name
        obj.p = float(row[LOAD_COLS["p"]])
        obj.q = float(row[LOAD_COLS["q"]])
        obj.current = float(row[LOAD_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.shunt_compensators, ppc["shunt"], shunt_names):
        obj.name = name
        obj.p = float(row[SHUNT_COLS["p"]])
        obj.q = float(row[SHUNT_COLS["q"]])
        obj.current = float(row[SHUNT_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.zero_branches, ppc["zero_branch"], zero_branch_names):
        obj.name = name
        obj.p = float(row[ZERO_BRANCH_COLS["p"]])
        obj.q = float(row[ZERO_BRANCH_COLS["q"]])
        obj.current = float(row[ZERO_BRANCH_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.switches, ppc["switch"], switch_names):
        obj.name = name
        obj.p = float(row[SWITCH_COLS["p"]])
        obj.q = float(row[SWITCH_COLS["q"]])
        obj.current = float(row[SWITCH_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.breakers, ppc.get("break", _empty(len(BREAK_COLS))), break_names):
        obj.name = name
        obj.p = float(row[BREAK_COLS["p"]])
        obj.q = float(row[BREAK_COLS["q"]])
        obj.current = float(row[BREAK_COLS["current"]])
        obj.is_alive = False
    for obj, row, name in zip(network.acac_converters, acac_table, acac_names):
        obj.name = name
        obj.i_p = float(row[ACAC_COLS["i_p"]])
        obj.i_q = float(row[ACAC_COLS["i_q"]])
        obj.j_p = float(row[ACAC_COLS["j_p"]])
        obj.j_q = float(row[ACAC_COLS["j_q"]])
        obj.i_i = float(row[ACAC_COLS["i_i"]])
        obj.j_i = float(row[ACAC_COLS["j_i"]])
        obj.is_alive = False
    return network


