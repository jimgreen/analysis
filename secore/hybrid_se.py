import argparse
import sys
import time
import warnings
from copy import copy, deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import block_diag, coo_matrix, csr_matrix, diags, eye as sparse_eye, hstack, vstack


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithm_parameters import (
    DEFAULT_SE_PARAMETER_FILE,
    StateEstimationParameters,
    load_se_parameters,
    resolve_bad_data_limits,
)
from paths import measurement_file, model_file
from efile_read import _read_efile_rows
from hybrid_lf import HybridPowerNetwork
from model.multi_energy_model import (
    MultiEnergyContext,
    attach_multi_energy_context,
    build_multi_energy_context_from_rows,
    electric_heat_balance_residual,
    gas_electric_balance_residual,
    gas_electric_dependent_value,
    gas_heat_balance_residual,
    hydrogen_electric_balance_residual,
    hydrogen_electric_dependent_value,
    steam_electric_balance_residual,
    steam_electric_dependent_value,
)
from model.ac_array_model import (
    BRANCH_COLS as AC_BRANCH_COLS,
    BUS_COLS as AC_BUS_COLS,
    GEN_COLS as AC_GEN_COLS,
    LOAD_COLS as AC_LOAD_COLS,
    SWITCH_COLS as AC_SWITCH_COLS,
    TRANSFORMER_COLS as AC_TRANSFORMER_COLS,
    ZERO_BRANCH_COLS as AC_ZERO_BRANCH_COLS,
)
from model.hybrid_array_model import (
    DCAC_COLS,
    DCAC_AC_CONTROL_LABEL,
    DCAC_DC_CONTROL_LABEL,
    DCAC_DEVICE_TYPE_LABEL,
)
from model.dc_array_model import (
    BRANCH_COLS as DC_BRANCH_COLS,
    BUS_COLS as DC_BUS_COLS,
    CTRL_V as DC_CTRL_V,
    GEN_COLS as DC_GEN_COLS,
    LOAD_COLS as DC_LOAD_COLS,
    SWITCH_COLS as DC_SWITCH_COLS,
    ZERO_BRANCH_COLS as DC_ZERO_BRANCH_COLS,
)
from model.ppc_topology import (
    build_ac_ppc_with_topology_from_efile_rows,
    build_dc_ppc_with_topology_from_efile_rows,
    build_hybrid_ppc_with_topology_from_efile_rows,
)
from model.meas_array_model import (
    attach_device_pos_from_name_arrays,
    build_meas_ppc_from_e_file,
    copy_meas_ppc,
    measurement_list_from_meas_ppc,
    sync_meas_ppc_from_measurement_table,
)
from model.meas_type import (
    DEVICE_TYPE_ACBreak,
    DEVICE_TYPE_ACBreakConstraint,
    DEVICE_TYPE_ACBranch,
    DEVICE_TYPE_ACGenerator,
    DEVICE_TYPE_ACLoad,
    DEVICE_TYPE_ACNode,
    DEVICE_TYPE_ACACConverter,
    DEVICE_TYPE_ACPowerBalance,
    DEVICE_TYPE_ACSwitch,
    DEVICE_TYPE_ACSwitchConstraint,
    DEVICE_TYPE_ACTransformer,
    DEVICE_TYPE_ACThreeWindingTransformer,
    DEVICE_TYPE_ACZeroBranch,
    DEVICE_TYPE_ACZeroBranchConstraint,
    DEVICE_TYPE_DCBreak,
    DEVICE_TYPE_DCBreakConstraint,
    DEVICE_TYPE_DCBranch,
    DEVICE_TYPE_DCDCConverter,
    DEVICE_TYPE_DCACConverter,
    DEVICE_TYPE_DCGenerator,
    DEVICE_TYPE_DCLoad,
    DEVICE_TYPE_DCNode,
    DEVICE_TYPE_DCSwitch,
    DEVICE_TYPE_DCSwitchConstraint,
    DEVICE_TYPE_DCZeroBranch,
    DEVICE_TYPE_DCZeroBranchConstraint,
    DEVICE_TYPE_NAMES,
    MEAS_TYPE_I_AC,
    MEAS_TYPE_I_DC,
    MEAS_TYPE_I_FROM,
    MEAS_TYPE_I_GEN,
    MEAS_TYPE_I_LOAD,
    MEAS_TYPE_I_TO,
    MEAS_TYPE_I_THIRD,
    MEAS_TYPE_P_AC,
    MEAS_TYPE_P_DC,
    MEAS_TYPE_P_FROM,
    MEAS_TYPE_P_GEN,
    MEAS_TYPE_P_LOAD,
    MEAS_TYPE_P_TO,
    MEAS_TYPE_P_THIRD,
    MEAS_TYPE_Q_AC,
    MEAS_TYPE_Q_FROM,
    MEAS_TYPE_Q_GEN,
    MEAS_TYPE_Q_LOAD,
    MEAS_TYPE_Q_TO,
    MEAS_TYPE_Q_THIRD,
    MEAS_TYPE_V,
    MEAS_TYPE_V_AC,
    MEAS_TYPE_V_DC,
    MEAS_TYPE_V_FROM,
    MEAS_TYPE_V_TO,
    MEAS_TYPE_V_THIRD,
    MEAS_TYPE_V_GEN,
    MEAS_TYPE_V_LOAD,
    MEAS_TYPE_CODES,
    MEAS_TYPE_NAMES,
)
from model.meas_model import (
    BadDataItem,
    DEVICE_TYPE_CODES,
    EstimateResult,
    MEAS_STATUS_INVALID,
    MEAS_STATUS_NORMAL,
    MEAS_STATUS_PSEUDO,
    VOLTAGE_PSEUDO_MEAS_TYPE_CODES,
    Measurement,
    MeasurementList,
    MeasurementTable,
    MeasurementTableView,
    ObservabilityResult,
    TableBackedMeasurementList,
    is_pseudo_measurement,
    mark_measurement_pseudo,
    measurement_from_table_row,
    measurement_table_bad_data_protected_mask,
    measurement_table_from_measurements,
    measurement_table_status_code,
    normalize_pseudo_measurement_arrays,
    print_iteration as _print_iteration,
    print_iteration_header as _print_iteration_header,
)
from secore.ac_se import (
    ACStateEstimator,
    _build_ac_se_ppc_namespace,
)
from secore.dc_se import DCStateEstimator, _build_dc_se_ppc_namespace
from secore.gas_se import GasStateEstimator
from secore.heat_se import HeatStateEstimator
from secore.hydro_se import HydroStateEstimator
from secore.steam_se import SteamStateEstimator
from secore.se_array_plan import (
    MeasurementPartitions,
    append_active_measurement_view,
    build_active_measurement_view,
    build_measurement_plan_table,
    concat_measurement_tables,
    copy_measurement_view,
    extend_measurement_partitions,
    measurement_table_take,
    partition_measurements_by_code,
    rows_by_device_type_code,
    take_measurement_view,
)
from secore.se_math import (
    AndersonAccelerator,
    NielsenTrustRegionController,
    nielsen_lm_damping_update,
    projected_box_kkt_residual,
    CHOLMOD_ANALYZE_AAT,
    CHOLMOD_CHOLESKY,
    CHOLMOD_CHOLESKY_AAT,
    CholmodAAtNormalEquationPlan,
    CholmodAAtNormalEquationSolver,
    LowerNormalEquationCscPlan,
    NormalEquationSolver,
    SEScaleProfile,
    SparseJacobianBuilder,
    _normal_equation_structural_pattern,
    build_normal_equations,
    build_switch_current_curvature_columns,
    check_active_structural_full_rank,
    compute_active_column_degrees,
    compute_adaptive_bad_threshold,
    compute_normalized_residuals_for_bad_data,
    detect_active_box_pinned_cols,
    filter_batch_preserving_column_coverage,
    full_normal_equation_from_lower,
    is_critical_leverage_row,
    is_sparse_matrix,
    inverse_gain_for_bad_data,
    matrix_is_empty,
    measurement_leverage,
    measurement_leverage_for_bad_data,
    observability_rank_details,
    observability_weak_direction,
    run_huber_irls_warmup,
    screen_bad_data_after_removal,
    screen_initial_bad_data_fast,
    select_independent_bad_data_batch,
    solve_box_pinned_normal_equations,
    solve_woodbury_after_row_removal,
    sparse_structural_rank,
    targeted_redundancy_count,
    unanchored_angle_state_indices,
)
from secore.se_result import (
    SEResult,
    build_seresult_full_from_table,
    build_seresult_summary,
    build_seresult_summary_from_table,
    normalize_seresult_result_mode,
)
from secore.state_metadata import StateMeta, state_labels_from_metadata, state_meta_at
from unit_system import ac_current_base_ka, dc_current_base_ka


DEFAULT_CASE = model_file("hybrid", "qinling.e")
DEFAULT_MEAS = measurement_file("hybrid", "qinling.meas")


def _measurement_table_from_measurements(measurements: Sequence[Measurement]):
    return measurement_table_from_measurements(
        measurements,
        device_type_codes=DEVICE_TYPE_CODES,
    )


class _SELightweightHybridNetwork(SimpleNamespace):
    @property
    def total_nodes(self) -> int:
        return _side_alive_node_count(getattr(self, "ac", None)) + _side_alive_node_count(getattr(self, "dc", None))

    def prepare(self, verbose: bool = True):
        return [], [], [], []

    def check_topology(self):
        return [], [], [], []


def _array_device(idx, name=None, **values):
    return SimpleNamespace(idx=int(idx), name=str(name if name is not None else idx), **values)


def _name_at(names, pos: int, prefix: str, idx: int) -> str:
    if names is None or pos >= len(names):
        return f"{prefix}_{idx}"
    return str(names[pos])


def _side_topology(side):
    if side is None:
        return None
    topology = getattr(side, "topology", None) or getattr(side, "_topology_arrays", None)
    if topology is not None:
        return topology
    ppc = getattr(side, "ppc", None)
    return ppc.get("_topology_arrays") if isinstance(ppc, dict) else None


def _side_alive_node_count(side) -> int:
    topology = _side_topology(side)
    if topology is not None:
        bus_alive = getattr(topology, "bus_alive_mask", None)
        if bus_alive is not None:
            return int(np.count_nonzero(np.asarray(bus_alive, dtype=bool)))
        node_alive = getattr(topology, "node_alive_mask", None)
        if node_alive is not None:
            return int(np.count_nonzero(np.asarray(node_alive, dtype=bool)))
    nodes = getattr(side, "nodes", None)
    if nodes is not None:
        return sum(1 for node in nodes if getattr(node, "is_alive", True))
    ppc = getattr(side, "ppc", None)
    bus = ppc.get("bus") if isinstance(ppc, dict) else None
    return int(np.asarray(bus).shape[0]) if isinstance(bus, np.ndarray) else 0


def _side_has_alive_nodes(side) -> bool:
    return _side_alive_node_count(side) > 0


def _side_alive_node_lookup(side) -> Dict[int, bool]:
    topology = _side_topology(side)
    if topology is not None:
        node_ids = getattr(topology, "node_ids", None)
        node_alive = getattr(topology, "node_alive_mask", None)
        if node_ids is not None and node_alive is not None:
            ids = np.asarray(node_ids, dtype=np.int64)
            alive = np.asarray(node_alive, dtype=bool)
            if ids.size == alive.size:
                return {int(idx): bool(flag) for idx, flag in zip(ids, alive)}
    node_dict = getattr(side, "node_dict", None)
    if isinstance(node_dict, dict):
        return {int(idx): bool(getattr(node, "is_alive", True)) for idx, node in node_dict.items()}
    return {}


def _build_se_dcac_converters(ppc: Dict) -> List[SimpleNamespace]:
    rows = np.asarray(ppc.get("dcac", np.zeros((0, len(DCAC_COLS)), dtype=np.float64)))
    n_rows = int(rows.shape[0]) if rows.ndim == 2 else 0
    if n_rows == 0:
        return []
    names = ppc.get("dcac_name")
    has_names = names is not None and len(names) == n_rows
    c_idx = rows[:, DCAC_COLS["idx"]].astype(np.int64, copy=False).tolist()
    c_ac_node = rows[:, DCAC_COLS["ac_node"]].astype(np.int64, copy=False).tolist()
    c_dc_node = rows[:, DCAC_COLS["dc_node"]].astype(np.int64, copy=False).tolist()
    c_r1 = rows[:, DCAC_COLS["r1"]].astype(np.float64, copy=False).tolist()
    c_r2 = rows[:, DCAC_COLS["r2"]].astype(np.float64, copy=False).tolist()
    c_ac_ctrl = rows[:, DCAC_COLS["ac_control_type"]].astype(np.int64, copy=False).tolist()
    c_dc_ctrl = rows[:, DCAC_COLS["dc_control_type"]].astype(np.int64, copy=False).tolist()
    c_p_ac_set = rows[:, DCAC_COLS["p_ac_set"]].astype(np.float64, copy=False).tolist()
    c_p_dc_set = rows[:, DCAC_COLS["p_dc_set"]].astype(np.float64, copy=False).tolist()
    c_q_ac_set = rows[:, DCAC_COLS["q_ac_set"]].astype(np.float64, copy=False).tolist()
    c_v_ac_set = rows[:, DCAC_COLS["v_ac_set"]].astype(np.float64, copy=False).tolist()
    c_v_dc_set = rows[:, DCAC_COLS["v_dc_set"]].astype(np.float64, copy=False).tolist()
    c_run_stat = rows[:, DCAC_COLS["run_stat"]].astype(np.int64, copy=False).tolist()
    c_dc_p = rows[:, DCAC_COLS["dc_p"]].astype(np.float64, copy=False).tolist()
    c_ac_p = rows[:, DCAC_COLS["ac_p"]].astype(np.float64, copy=False).tolist()
    c_ac_q = rows[:, DCAC_COLS["ac_q"]].astype(np.float64, copy=False).tolist()
    c_dc_i = rows[:, DCAC_COLS["dc_i"]].astype(np.float64, copy=False).tolist()
    c_ac_i = rows[:, DCAC_COLS["ac_i"]].astype(np.float64, copy=False).tolist()
    c_dev_type = rows[:, DCAC_COLS["dev_type"]].astype(np.int64, copy=False).tolist()
    name_list = (
        [str(v) if v is not None and str(v) else f"dcac_{idx}" for v, idx in zip(names, c_idx)]
        if has_names
        else [f"dcac_{idx}" for idx in c_idx]
    )
    ac_ctrl_get = DCAC_AC_CONTROL_LABEL.get
    dc_ctrl_get = DCAC_DC_CONTROL_LABEL.get
    dev_type_get = DCAC_DEVICE_TYPE_LABEL.get
    return [
        SimpleNamespace(
            idx=idx,
            name=name,
            ac_node=ac_node,
            dc_node=dc_node,
            r1=r1,
            r2=r2,
            ac_control_type=ac_ctrl_get(ac_ctrl, "NONE"),
            dc_control_type=dc_ctrl_get(dc_ctrl, "NONE"),
            p_ac_set=p_ac_set,
            p_dc_set=p_dc_set,
            q_ac_set=q_ac_set,
            v_ac_set=v_ac_set,
            v_dc_set=v_dc_set,
            run_stat=run_stat,
            dc_p=dc_p,
            ac_p=ac_p,
            ac_q=ac_q,
            dc_i=dc_i,
            ac_i=ac_i,
            dev_type=dev_type_get(dev_type, "DCACConverter"),
            ac_node_obj=None,
            dc_node_obj=None,
            ac_isl_obj=None,
            dc_isl_obj=None,
            hybrid_isl=0,
            hybrid_isl_obj=None,
            is_alive=False,
        )
        for (
            idx, name, ac_node, dc_node, r1, r2, ac_ctrl, dc_ctrl,
            p_ac_set, p_dc_set, q_ac_set, v_ac_set, v_dc_set,
            run_stat, dc_p, ac_p, ac_q, dc_i, ac_i, dev_type,
        ) in zip(
            c_idx, name_list, c_ac_node, c_dc_node, c_r1, c_r2, c_ac_ctrl, c_dc_ctrl,
            c_p_ac_set, c_p_dc_set, c_q_ac_set, c_v_ac_set, c_v_dc_set,
            c_run_stat, c_dc_p, c_ac_p, c_ac_q, c_dc_i, c_ac_i, c_dev_type,
        )
    ]


def _assign_se_converter_topology(network: _SELightweightHybridNetwork) -> None:
    # Converter node/island object links are unused by the array-mode SE path
    # (only the load-flow path consumes them, and it wires its own). Alive flags
    # are derived from the topology arrays via the side alive lookups, so this
    # avoids touching ``network.dc.node_dict`` and keeps the DC object graph
    # unmaterialized for hybrid cold starts.
    ac_alive = _side_alive_node_lookup(network.ac)
    dc_alive = _side_alive_node_lookup(network.dc)
    for conv in network.dcac_converters:
        conv.ac_node_obj = None
        conv.dc_node_obj = None
        conv.ac_isl_obj = None
        conv.dc_isl_obj = None
        conv.is_alive = bool(
            int(getattr(conv, "run_stat", 1)) == 1
            and ac_alive.get(int(conv.ac_node), False)
            and dc_alive.get(int(conv.dc_node), False)
        )


def _build_hybrid_se_network_from_ppc(ppc: Dict) -> _SELightweightHybridNetwork:
    ac_network = (
        _build_ac_se_ppc_namespace(ppc["ac"], Path(ppc["source"]) if ppc.get("source") else None)
        if ppc.get("ac") is not None else _empty_se_side(ppc["base"])
    )
    dc_network = (
        _build_dc_se_ppc_namespace(ppc["dc"])
        if ppc.get("dc") is not None else _empty_se_side(ppc["base"])
    )
    network = _SELightweightHybridNetwork(
        _se_lightweight=True,
        ac=ac_network,
        dc=dc_network,
        dcac_converters=_build_se_dcac_converters(ppc),
        hybrid_islands=[],
        ppc=ppc,
        _ac_ppc=ppc["ac"],
        _dc_ppc=ppc["dc"],
    )
    base = ppc["base"]
    network.p_base = float(base["p_base"])
    network.u_scale = float(base["u_scale"])
    network.p_scale = float(base["p_scale"])
    network.i_scale = float(base["i_scale"])
    network.p_base_kW = float(base["p_base_kW"])
    _assign_se_converter_topology(network)
    return network


def _detect_se_rows_kind(rows) -> str:
    has_ac = bool(rows.get("ACNode", {}).get("rows"))
    has_dc = bool(rows.get("DCNode", {}).get("rows"))
    has_dcac = bool(rows.get("DCACConverter", {}).get("rows"))
    if has_dcac or (has_ac and has_dc):
        return "hybrid"
    if has_ac:
        return "ac"
    if has_dc:
        return "dc"
    if any(
        bool(rows.get(f"{prefix}Node", {}).get("rows"))
        for prefix in ("Heat", "Gas", "Hydro", "Steam")
    ):
        return "fluid"
    return "hybrid"


def _empty_se_side(base: Dict) -> SimpleNamespace:
    return SimpleNamespace(
        _se_lightweight=True,
        ppc=None,
        base=base,
        topology=None,
        _topology_arrays=None,
        p_base=float(base.get("p_base", 1.0)),
        p_base_kW=float(base.get("p_base_kW", base.get("p_base", 1.0))),
        u_scale=float(base.get("u_scale", 1.0)),
        p_scale=float(base.get("p_scale", 1.0)),
        i_scale=float(base.get("i_scale", 1.0)),
        nodes=[],
        node_dict={},
    )


def _build_se_network_from_fluid_only(file_name) -> _SELightweightHybridNetwork:
    base = {
        "p_base": 1.0,
        "p_base_kW": 1.0,
        "u_scale": 1.0,
        "p_scale": 1.0,
        "i_scale": 1.0,
    }
    return _SELightweightHybridNetwork(
        _se_lightweight=True,
        ac=_empty_se_side(base),
        dc=_empty_se_side(base),
        dcac_converters=[],
        hybrid_islands=[],
        ppc={
            "format": "hybrid_ppc_v1",
            "source": str(Path(file_name).resolve()),
            "base": base,
            "ac": None,
            "dc": None,
        },
        _ac_ppc=None,
        _dc_ppc=None,
        p_base=1.0,
        p_base_kW=1.0,
        u_scale=1.0,
        p_scale=1.0,
        i_scale=1.0,
    )


class _HybridStateLabelSlice:
    __slots__ = ("_parent", "_start", "_stop", "_step")

    def __init__(self, parent, start: int, stop: int, step: int):
        self._parent = parent
        self._start = int(start)
        self._stop = int(stop)
        self._step = int(step)

    def __len__(self) -> int:
        return len(range(self._start, self._stop, self._step))

    def __iter__(self):
        for pos in range(self._start, self._stop, self._step):
            yield self._parent[pos]

    def __getitem__(self, key):
        values = range(self._start, self._stop, self._step)
        if isinstance(key, slice):
            sub = values[key]
            return _HybridStateLabelSlice(self._parent, sub.start, sub.stop, sub.step)
        return self._parent[values[int(key)]]

    def __contains__(self, value) -> bool:
        return any(label == value for label in self)

    def __eq__(self, other) -> bool:
        return list(self) == list(other)

    def index(self, value) -> int:
        for pos, label in enumerate(self):
            if label == value:
                return pos
        raise ValueError(f"{value!r} is not in state labels")


class _HybridStateLabelView:
    __slots__ = ("ac_n", "dc_n", "dcac_names", "dcac_start", "n_state", "_ac_arrays", "_dc_arrays")

    def __init__(self, ac_n: int, dc_n: int, dcac_names: Sequence[str]):
        self.ac_n = int(ac_n)
        self.dc_n = int(dc_n)
        self.dcac_names = tuple(str(name) for name in dcac_names)
        self.dcac_start = self.ac_n + self.dc_n
        self.n_state = self.dcac_start + 3 * len(self.dcac_names)
        self._ac_arrays = None
        self._dc_arrays = None

    def __len__(self) -> int:
        return self.n_state

    def _label_at(self, pos: int) -> str:
        if pos < self.ac_n:
            if self._ac_arrays is not None:
                return self._sub_label(self._ac_arrays, pos, "ac")
            return f"AC_STATE:{pos}"
        if pos < self.dcac_start:
            if self._dc_arrays is not None:
                return self._sub_label(self._dc_arrays, pos - self.ac_n, "dc")
            return f"DC_STATE:{pos - self.ac_n}"
        local = pos - self.dcac_start
        name = self.dcac_names[local // 3]
        kind = local % 3
        if kind == 0:
            return f"DCAC_P_DC:{name}"
        if kind == 1:
            return f"DCAC_P_AC:{name}"
        return f"DCAC_Q_AC:{name}"

    @staticmethod
    def _sub_label(arrays, pos: int, side: str) -> str:
        kind = str(arrays["kind"][pos])
        prefixes = (HybridStateEstimator._AC_STATE_META_PREFIX_BY_KIND if side == "ac"
                    else HybridStateEstimator._DC_STATE_META_PREFIX_BY_KIND)
        prefix = prefixes.get(kind)
        if prefix is None:
            if side == "ac" and kind in ("zero_current", "break_current"):
                prefix = "AC_I_RE" if arrays["component"][pos] == "re" else "AC_I_IM"
            else:
                prefix = f"{side.upper()}_{kind.upper()}"
        return f"{prefix}:{arrays['device_name'][pos]}"

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(self.n_state)[key]
            return _HybridStateLabelSlice(self, indices.start, indices.stop, indices.step)
        pos = int(key)
        if pos < 0:
            pos += self.n_state
        if pos < 0 or pos >= self.n_state:
            raise IndexError("state label index out of range")
        return self._label_at(pos)

    def __iter__(self):
        for pos in range(self.n_state):
            yield self._label_at(pos)

    def __contains__(self, value) -> bool:
        return any(label == value for label in self)

    def __eq__(self, other) -> bool:
        return list(self) == list(other)

    def index(self, value) -> int:
        for pos, label in enumerate(self):
            if label == value:
                return pos
        raise ValueError(f"{value!r} is not in state labels")


class _HybridStateSideView:
    __slots__ = ("ac_n", "dc_n", "hybrid_n", "n_state")

    def __init__(self, ac_n: int, dc_n: int, hybrid_n: int):
        self.ac_n = int(ac_n)
        self.dc_n = int(dc_n)
        self.hybrid_n = int(hybrid_n)
        self.n_state = self.ac_n + self.dc_n + self.hybrid_n

    def __len__(self) -> int:
        return self.n_state

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[pos] for pos in range(self.n_state)[key]]
        pos = int(key)
        if pos < 0:
            pos += self.n_state
        if pos < 0 or pos >= self.n_state:
            raise IndexError("state side index out of range")
        if pos < self.ac_n:
            return "ac"
        if pos < self.ac_n + self.dc_n:
            return "dc"
        if pos < self.n_state:
            return "hybrid"
        raise IndexError("state side index out of range")

    def __iter__(self):
        for _ in range(self.ac_n):
            yield "ac"
        for _ in range(self.dc_n):
            yield "dc"
        for _ in range(self.hybrid_n):
            yield "hybrid"


class _DelegatedSequenceView:
    __slots__ = ("_delegate", "_attr")

    def __init__(self, delegate, attr: str):
        self._delegate = delegate
        self._attr = str(attr)

    def _target(self):
        return getattr(self._delegate, self._attr)

    def __len__(self) -> int:
        n_state = getattr(self._delegate, "n_state", None)
        if self._attr in {"state_labels", "state_meta"} and n_state is not None:
            return int(n_state)
        return len(self._target())

    def __getitem__(self, key):
        return self._target()[key]

    def __iter__(self):
        return iter(self._target())

    def __contains__(self, value) -> bool:
        return value in self._target()

    def __eq__(self, other) -> bool:
        try:
            return list(self._target()) == list(other)
        except TypeError:
            return False


def _state_meta_from_arrays(arrays: Optional[Dict[str, np.ndarray]], pos: int) -> Optional[StateMeta]:
    if arrays is None:
        return None
    pos = int(pos)
    kind = np.asarray(arrays.get("kind", np.asarray([], dtype=object)), dtype=object)
    if not 0 <= pos < kind.size:
        return None
    return StateMeta(
        str(np.asarray(arrays.get("side", np.asarray([], dtype=object)), dtype=object)[pos]),
        str(kind[pos]),
        str(np.asarray(arrays.get("device_type", np.asarray([], dtype=object)), dtype=object)[pos]),
        str(np.asarray(arrays.get("device_name", np.asarray([], dtype=object)), dtype=object)[pos]),
        terminal=str(np.asarray(arrays.get("terminal", np.asarray([], dtype=object)), dtype=object)[pos]),
        component=str(np.asarray(arrays.get("component", np.asarray([], dtype=object)), dtype=object)[pos]),
        legacy_label=str(np.asarray(arrays.get("legacy_label", np.asarray([], dtype=object)), dtype=object)[pos]),
        device_pos=int(np.asarray(arrays.get("device_pos", np.asarray([], dtype=np.int64)), dtype=np.int64)[pos]),
        device_type_code=int(np.asarray(arrays.get("device_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)[pos]),
        meas_type_code=int(np.asarray(arrays.get("meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)[pos]),
    )


def _state_meta_from_estimator_arrays(estimator, pos: int) -> Optional[StateMeta]:
    if estimator is None:
        return None
    arrays_ref = getattr(estimator, "_state_meta_arrays_ref", None)
    if callable(arrays_ref):
        meta = _state_meta_from_arrays(arrays_ref(), pos)
        if meta is not None:
            return meta
    return state_meta_at(getattr(estimator, "state_meta", ()), int(pos))


class _HybridStateMetaView:
    __slots__ = ("_labels", "_ac_estimator", "_dc_estimator", "_ac_arrays", "_dc_arrays")

    def __init__(self, labels: _HybridStateLabelView, ac_estimator=None, dc_estimator=None):
        self._labels = labels
        self._ac_estimator = ac_estimator
        self._dc_estimator = dc_estimator
        ac_arrays_ref = getattr(ac_estimator, "_state_meta_arrays_ref", None)
        dc_arrays_ref = getattr(dc_estimator, "_state_meta_arrays_ref", None)
        self._ac_arrays = ac_arrays_ref() if callable(ac_arrays_ref) else None
        self._dc_arrays = dc_arrays_ref() if callable(dc_arrays_ref) else None

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[pos] for pos in range(len(self))[key]]
        pos = int(key)
        labels = self._labels
        if pos < 0:
            pos += len(labels)
        if pos < 0 or pos >= len(labels):
            raise IndexError("state metadata index out of range")
        if pos < labels.ac_n:
            meta = _state_meta_from_arrays(self._ac_arrays, pos)
            if meta is None:
                meta = _state_meta_from_estimator_arrays(self._ac_estimator, pos)
            if meta is not None:
                return HybridStateEstimator._ac_sub_state_meta_to_hybrid(meta)
            return StateMeta("ac", "sub_state", "ACSubsystem", str(pos), legacy_label=labels[pos])
        if pos < labels.dcac_start:
            local = pos - labels.ac_n
            meta = _state_meta_from_arrays(self._dc_arrays, local)
            if meta is None:
                meta = _state_meta_from_estimator_arrays(self._dc_estimator, local)
            if meta is not None:
                return HybridStateEstimator._dc_sub_state_meta_to_hybrid(meta)
            return StateMeta("dc", "sub_state", "DCSubsystem", str(local), legacy_label=labels[pos])
        local = pos - labels.dcac_start
        conv_pos = local // 3
        name = labels.dcac_names[conv_pos]
        kind = local % 3
        if kind == 0:
            return StateMeta("hybrid", "dcac_p_dc", "DCACConverter", name, terminal="dc", component="p", legacy_label=labels[pos], device_pos=conv_pos, device_type_code=DEVICE_TYPE_DCACConverter, meas_type_code=MEAS_TYPE_P_DC)
        if kind == 1:
            return StateMeta("hybrid", "dcac_p_ac", "DCACConverter", name, terminal="ac", component="p", legacy_label=labels[pos], device_pos=conv_pos, device_type_code=DEVICE_TYPE_DCACConverter, meas_type_code=MEAS_TYPE_P_AC)
        return StateMeta("hybrid", "dcac_q_ac", "DCACConverter", name, terminal="ac", component="q", legacy_label=labels[pos], device_pos=conv_pos, device_type_code=DEVICE_TYPE_DCACConverter, meas_type_code=MEAS_TYPE_Q_AC)

    def __iter__(self):
        for pos in range(len(self)):
            yield self[pos]


@dataclass
class MultiEnergySEResult:
    electric: Optional[SEResult] = None
    fluids: Dict[str, SEResult] = field(default_factory=dict)
    fluid_estimators: Dict[str, object] = field(default_factory=dict)
    couplings: List[object] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    converged: bool = False
    observable: bool = False

    @property
    def bad_data_count(self) -> int:
        return sum(len(getattr(estimator, "bad_data", ())) for estimator in self.fluid_estimators.values())


@dataclass(frozen=True)
class _MultiEnergySEEndpoint:
    provider: str
    row: int
    domain: str
    device_type: str
    device_idx: int
    device_type_code: int
    meas_type_code: int
    device_pos: int


@dataclass(frozen=True)
class _MultiEnergySECouplingPlan:
    coupling: object
    t1: _MultiEnergySEEndpoint
    t2: _MultiEnergySEEndpoint
    t1_scale: float
    t2_scale: float
    normalization: float
    measurement_row: int
    control_col: int = -1
    prior_measurement_row: int = -1
    prior_state_col: int = -1
    controlled_prior_measurement_row: int = -1
    controlled_prior_state_col: int = -1
    electric_heat: bool = False
    gas_heat: bool = False
    hydrogen_electric: bool = False
    gas_electric: bool = False
    steam_electric: bool = False
    controlled: Optional[_MultiEnergySEEndpoint] = None
    dependent: Optional[_MultiEnergySEEndpoint] = None
    electric: Optional[_MultiEnergySEEndpoint] = None
    gas: Optional[_MultiEnergySEEndpoint] = None
    heat_flow: Optional[_MultiEnergySEEndpoint] = None
    heat_temperature: Optional[_MultiEnergySEEndpoint] = None
    controlled_setpoint: float = 0.0
    controlled_measurement_rows: Tuple[int, ...] = ()
    dependent_measurement_rows: Tuple[int, ...] = ()
    heat_supply_measurement_rows: Tuple[int, ...] = ()
    heat_power_measurement_rows: Tuple[int, ...] = ()
    electric_setpoint: float = 0.0
    gas_setpoint: float = 0.0
    return_temperature_col: int = -1
    supply_temperature_set: float = 0.0
    heat_capacity: float = 1.0
    power_scale: float = 1.0
    steam_enthalpy_col: int = -1
    condensate_enthalpy: float = 0.0


class HybridStateEstimator:
    """AC/DC and fluid-network state-estimation orchestrator.

    AC and DC device rows are normalized, evaluated and differentiated by
    ACStateEstimator/DCStateEstimator.  This class owns only measurement
    partitioning, global WLS assembly and converter-side rows. Heat, gas,
    hydrogen and steam measurements are partitioned by device ownership and
    delegated to their specialized sparse WLS estimators.

    For a coupled hybrid network, the side estimators are measurement-model
    providers only: they do not run independent WLS iterations.  All AC, DC
    and converter states are assembled into one state vector, one sparse H,
    and one global normal equation solved jointly in each iteration.
    Multi-energy coupling rows remain in this layer for endpoint validation
    and aggregate result reporting.
    """

    _AC_MEASUREMENT_DEVICE_TYPES = frozenset(
        (
            "ACNode",
            "ACBranch",
            "ACTransformer",
            "ACThreeWindingTransformer",
            "AC3WTransformer",
            "ACBreak",
            "ACZeroBranch",
            "ACGenerator",
            "ACLoad",
            "ACPowerBalance",
            "ACZeroBranchConstraint",
            "ACBreakConstraint",
            "ACACConverter",
        )
    )
    _DC_MEASUREMENT_DEVICE_TYPES = frozenset(
        (
            "DCNode",
            "DCBranch",
            "DCBreak",
            "DCZeroBranch",
            "DCZeroBranchConstraint",
            "DCBreakConstraint",
            "DCGenerator",
            "DCLoad",
            "DCDCConverter",
            "DCPowerBalance",
        )
    )
    _HYBRID_MEASUREMENT_DEVICE_TYPES = frozenset(("DCACConverter",))
    _VOLTAGE_MEASUREMENT_TYPES = frozenset(
        ("V", "V_FROM", "V_TO", "V_THIRD", "V_GEN", "V_LOAD", "V_DC", "V_AC")
    )
    _VOLTAGE_MEASUREMENT_TYPE_TUPLE = tuple(_VOLTAGE_MEASUREMENT_TYPES)
    _MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE = {
        DEVICE_TYPE_ACNode: "ac",
        DEVICE_TYPE_ACBranch: "ac",
        DEVICE_TYPE_ACTransformer: "ac",
        DEVICE_TYPE_ACThreeWindingTransformer: "ac",
        DEVICE_TYPE_ACBreak: "ac",
        DEVICE_TYPE_ACZeroBranch: "ac",
        DEVICE_TYPE_ACGenerator: "ac",
        DEVICE_TYPE_ACLoad: "ac",
        DEVICE_TYPE_ACPowerBalance: "ac",
        DEVICE_TYPE_ACZeroBranchConstraint: "ac",
        DEVICE_TYPE_ACBreakConstraint: "ac",
        DEVICE_TYPE_ACACConverter: "ac",
        DEVICE_TYPE_DCNode: "dc",
        DEVICE_TYPE_DCBranch: "dc",
        DEVICE_TYPE_DCBreak: "dc",
        DEVICE_TYPE_DCZeroBranch: "dc",
        DEVICE_TYPE_DCZeroBranchConstraint: "dc",
        DEVICE_TYPE_DCBreakConstraint: "dc",
        DEVICE_TYPE_DCGenerator: "dc",
        DEVICE_TYPE_DCLoad: "dc",
        DEVICE_TYPE_DCDCConverter: "dc",
        DEVICE_TYPE_DCACConverter: "hybrid",
    }
    _DCAC_MEASUREMENT_CODE = {
        MEAS_TYPE_P_DC: 1,
        MEAS_TYPE_P_AC: 2,
        MEAS_TYPE_Q_AC: 3,
        MEAS_TYPE_V_DC: 4,
        MEAS_TYPE_I_DC: 5,
        MEAS_TYPE_V_AC: 6,
        MEAS_TYPE_I_AC: 7,
    }
    _SUB_MEASUREMENT_SUMMARY_ATTRS = (
        "_active_device_key_cache",
        "_active_measurement_key_cache",
        "_max_measurement_idx",
        "_node_voltage_measurement_cache",
        "_real_voltage_observation_node_cache",
        "_real_power_measurement_seed_cache",
        "_has_valid_angle_measurements",
    )

    @dataclass(frozen=True)
    class _MeasurementSideBlock:
        rows: np.ndarray
        measurements: Sequence[Measurement]
        plan_tables: Optional[object] = None

    @dataclass(frozen=True)
    class _SubEstimatorMeasurementContext:
        measurements: Sequence[Measurement]
        summary_attrs: Optional[Dict[str, object]]

    @dataclass(frozen=True)
    class _HybridCodeBucket:
        """Pre-sliced state/jacobian info for a single (device-type, measurement-code) pair.

        Built once at plan construction; consumed every iteration by
        `_evaluate_hybrid_measurements` and `_append_hybrid_jacobian_plan` so they
        avoid per-iteration `codes == k` masks, column-validity filtering and
        default-template copies.
        """
        rows: np.ndarray         # int32, global measurement rows
        hx_p: np.ndarray         # int32, primary hybrid_x index (e.g. 3*pos for DCAC P_DC)
        hx_q: np.ndarray         # int32, secondary hybrid_x index (used by hypot codes)
        jcol_p: np.ndarray       # int32, jacobian col for the primary power state
        jcol_q: np.ndarray       # int32, jacobian col for the secondary power state
        v_default: np.ndarray    # float64, len == rows.size, default voltage per row
        v_use_x: np.ndarray      # bool,    len == rows.size, True where voltage comes from x
        v_x_index: np.ndarray    # int32,   filtered (len == v_use_x.sum()), index into x
        v_jcol: np.ndarray       # int32,   len == rows.size, voltage jacobian col (only valid where v_use_x)
        v_any_x: bool            # True iff v_use_x has any True entries

    @dataclass(frozen=True)
    class _HybridMeasurementPlan:
        dcac_rows: np.ndarray
        dcac_codes: np.ndarray
        dcac_pos: np.ndarray
        dcac_dc_v_col: np.ndarray
        dcac_ac_v_col: np.ndarray
        dcac_dc_v_default: np.ndarray
        dcac_ac_v_default: np.ndarray
        # Per-code buckets (indexed by 1-based measurement code).
        # DCAC: 1=P_DC 2=P_AC 3=Q_AC 4=V_DC 5=I_DC 6=V_AC 7=I_AC
        dcac_p_dc: "HybridStateEstimator._HybridCodeBucket"
        dcac_p_ac: "HybridStateEstimator._HybridCodeBucket"
        dcac_q_ac: "HybridStateEstimator._HybridCodeBucket"
        dcac_v_dc: "HybridStateEstimator._HybridCodeBucket"
        dcac_i_dc: "HybridStateEstimator._HybridCodeBucket"
        dcac_v_ac: "HybridStateEstimator._HybridCodeBucket"
        dcac_i_ac: "HybridStateEstimator._HybridCodeBucket"

    @dataclass(frozen=True)
    class _HybridSeedMeasurementPlan:
        measurement_row: np.ndarray
        state_col: np.ndarray

    @dataclass(frozen=True)
    class _HybridConverterMeasurementSpec:
        device_type: str
        device_name: str
        meas_type: str
        value: float
        device_type_code: int = 0
        device_pos: int = -1
        meas_type_code: int = 0

    @dataclass(frozen=True)
    class _MeasurementActivitySummary:
        max_idx: int
        measured_devices: set
        active_keys: set

    _ACTIVE_KEY_MEAS_BITS = 16
    _ACTIVE_KEY_POS_BITS = 32

    @staticmethod
    def _packed_device_key(device_type_code: int, device_pos: int) -> int:
        if int(device_type_code) <= 0 or int(device_pos) < 0:
            return -1
        return (int(device_type_code) << HybridStateEstimator._ACTIVE_KEY_POS_BITS) | int(device_pos)

    @staticmethod
    def _packed_measurement_key(device_type_code: int, device_pos: int, meas_type_code: int) -> int:
        if int(device_type_code) <= 0 or int(device_pos) < 0 or int(meas_type_code) <= 0:
            return -1
        return (
            (int(device_type_code) << (HybridStateEstimator._ACTIVE_KEY_POS_BITS + HybridStateEstimator._ACTIVE_KEY_MEAS_BITS))
            | (int(device_pos) << HybridStateEstimator._ACTIVE_KEY_MEAS_BITS)
            | int(meas_type_code)
        )

    @staticmethod
    def _packed_device_key_array(device_type_code: np.ndarray, device_pos: np.ndarray) -> np.ndarray:
        type_values = np.asarray(device_type_code, dtype=np.int64)
        pos_values = np.asarray(device_pos, dtype=np.int64)
        return (type_values << HybridStateEstimator._ACTIVE_KEY_POS_BITS) | pos_values

    @staticmethod
    def _packed_measurement_key_array(
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
        meas_type_code: np.ndarray,
    ) -> np.ndarray:
        type_values = np.asarray(device_type_code, dtype=np.int64)
        pos_values = np.asarray(device_pos, dtype=np.int64)
        meas_values = np.asarray(meas_type_code, dtype=np.int64)
        return (
            (type_values << (HybridStateEstimator._ACTIVE_KEY_POS_BITS + HybridStateEstimator._ACTIVE_KEY_MEAS_BITS))
            | (pos_values << HybridStateEstimator._ACTIVE_KEY_MEAS_BITS)
            | meas_values
        )

    def _warn_missing_device_pos(
        self,
        context: str,
        device_type_code: int,
        device_name: object = "",
        meas_type_code: int = 0,
    ) -> None:
        key = (str(context), int(device_type_code), int(meas_type_code))
        seen = getattr(self, "_missing_device_pos_warning_keys", None)
        if seen is None:
            seen = set()
            self._missing_device_pos_warning_keys = seen
        if key in seen:
            return
        seen.add(key)
        warnings.warn(
            "Hybrid SE measurement row missing device_pos; string/name-id fallback is disabled "
            f"(context={context}, device_type_code={int(device_type_code)}, "
            f"device_name={device_name}, meas_type_code={int(meas_type_code)}).",
            RuntimeWarning,
            stacklevel=2,
        )

    def __init__(
        self,
        e_file: Path = DEFAULT_CASE,
        meas_file: Path = DEFAULT_MEAS,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        diff_step: Optional[float] = None,
        flat_start: Optional[bool] = None,
        parameter_file: Path = DEFAULT_SE_PARAMETER_FILE,
        parameters: Optional[StateEstimationParameters] = None,
        profile: bool = False,
        network: Optional[object] = None,
        measurements: Optional[object] = None,
        prepare_active_measurements: bool = True,
        defer_prepare_finalize: bool = False,
        auto_prepare: bool = True,
        matrix_dump_dir: Optional[Path] = None,
        power_flow_linear_solver: Optional[str] = None,
        voltage_min: Optional[float] = None,
        voltage_max: Optional[float] = None,
        angle_min_deg: Optional[float] = None,
        angle_max_deg: Optional[float] = None,
        pseudo_measurement_weight_v: Optional[float] = None,
        pseudo_measurement_weight_pq: Optional[float] = None,
        max_remove: Optional[int] = None,
        max_remove_per_round: Optional[int] = None,
    ):
        self.profile_enabled = bool(profile)
        self.profile_times: Dict[str, float] = {}
        self.matrix_dump_dir = None if matrix_dump_dir is None else Path(matrix_dump_dir)
        override_kwargs = dict(
            tol=tol,
            max_iter=max_iter,
            diff_step=diff_step,
            flat_start=flat_start,
            pseudo_measurement_weight_v=pseudo_measurement_weight_v,
            pseudo_measurement_weight_pq=pseudo_measurement_weight_pq,
            voltage_min=voltage_min,
            voltage_max=voltage_max,
            angle_min_deg=angle_min_deg,
            angle_max_deg=angle_max_deg,
        )
        if max_remove is not None:
            override_kwargs["max_remove"] = max_remove
        if max_remove_per_round is not None:
            override_kwargs["max_remove_per_round"] = max_remove_per_round
        self.params = (parameters or load_se_parameters(parameter_file)).with_overrides(**override_kwargs)
        self.e_file = Path(e_file)
        self.meas_file = Path(meas_file)
        self.tol = self.params.tol
        self.max_iter = self.params.max_iter
        self.diff_step = self.params.diff_step
        self.flat_start = self.params.flat_start
        self.pseudo_measurement_weight_v = float(self.params.pseudo_measurement_weight_v)
        self.pseudo_measurement_weight_pq = float(self.params.pseudo_measurement_weight_pq)
        self.pseudo_measurement_weight = self.pseudo_measurement_weight_pq
        self.targeted_pseudo_measurement_max = self.params.targeted_pseudo_measurement_max
        self.targeted_pseudo_measurement_redundancy_ratio = (
            self.params.targeted_pseudo_measurement_redundancy_ratio
        )
        self.targeted_pseudo_measurement_step = self.params.targeted_pseudo_measurement_step
        self.voltage_floor = self.params.voltage_floor
        self.voltage_min = float(self.params.voltage_min)
        self.voltage_max = float(self.params.voltage_max)
        self.angle_min_deg = float(self.params.angle_min_deg)
        self.angle_max_deg = float(self.params.angle_max_deg)
        self.angle_min_rad = float(np.deg2rad(self.angle_min_deg))
        self.angle_max_rad = float(np.deg2rad(self.angle_max_deg))
        self.min_current_voltage = self.params.min_current_voltage
        solver_name = str(power_flow_linear_solver).strip().lower() if power_flow_linear_solver is not None else ""
        self.power_flow_linear_solver = solver_name or None

        self._prepared = False
        self._prepare_network = network
        self._prepare_measurements = measurements
        self._prepare_active_measurements = bool(prepare_active_measurements)
        self._prepare_defer_finalize = bool(defer_prepare_finalize)
        self.observability_result = None
        self.estimate_result = None
        self.converged = False
        self.iterations = 0
        self.removed_bad_data: List[BadDataItem] = []
        self.bad_items: List[BadDataItem] = []
        self.normalized_residual = np.array([], dtype=np.float64)
        self.se_result = None
        self._last_bad_data_cache = None
        self._real_voltage_seed_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._hybrid_seed_measurement_plan_cache: Dict[
            Tuple[int, int], HybridStateEstimator._HybridSeedMeasurementPlan
        ] = {}
        self.power_flow_state = np.array([], dtype=np.float64)
        self.flat_state = np.array([], dtype=np.float64)
        self._initial_state_cache_ready = False
        self.multi_energy = MultiEnergyContext()
        self._fluid_only = False
        self.fluid_estimators: Dict[str, object] = {}
        self.fluid_se_rc: Dict[str, int] = {}
        self.fluid_se_errors: Dict[str, str] = {}
        self.multi_energy_result = MultiEnergySEResult()
        self.multi_energy_se_coupling_plans: List[_MultiEnergySECouplingPlan] = []
        self.multi_energy_estimate_result: Optional[EstimateResult] = None
        self.multi_energy_observability_result: Optional[ObservabilityResult] = None
        self._multi_energy_se_layout_ready = False
        self._state_labels_cache: Sequence[str] = []
        self._state_meta_cache: Sequence[StateMeta] = []
        self._last_estimate_cholmod_factor = None
        self._compatibility_device_views_ready = False
        if auto_prepare:
            self.prepare()

    _COMPAT_VIEW_ATTRS = frozenset(
        {
            "ac_nodes",
            "dc_nodes",
            "ac_node_by_name",
            "ac_node_by_idx",
            "dc_node_by_name",
            "dc_node_by_idx",
            "ac_branches",
            "ac_transformers",
            "ac_zero_branches",
            "ac_breakers",
            "ac_generators",
            "ac_loads",
            "dc_branches",
            "dc_zero_branches",
            "dc_breakers",
            "dc_generators",
            "dc_loads",
            "ac_reference_nodes",
            "dc_reference_nodes",
            "ac_branch_by_name",
            "ac_zero_branch_by_name",
            "ac_generator_by_name",
            "ac_load_by_name",
            "dc_zero_branch_by_name",
            "dc_generator_by_name",
            "dc_load_by_name",
            "dcac_by_name",
        }
    )

    def __getattr__(self, name: str):
        # 兼容视图属性改为按需懒加载，避免在 prepare() 阶段无谓构造全网 SimpleNamespace 对象
        if name in self._COMPAT_VIEW_ATTRS:
            self._install_compatibility_device_views()
            return self.__dict__[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    @property
    def state_labels(self):
        return getattr(self, "_state_labels_cache", [])

    @state_labels.setter
    def state_labels(self, value) -> None:
        self._state_labels_cache = value

    @property
    def state_meta(self):
        return getattr(self, "_state_meta_cache", [])

    @state_meta.setter
    def state_meta(self, value) -> None:
        self._state_meta_cache = value

    def prepare(
        self,
        *,
        network: Optional[object] = None,
        measurements: Optional[object] = None,
        prepare_active_measurements: Optional[bool] = None,
        defer_prepare_finalize: Optional[bool] = None,
    ) -> "HybridStateEstimator":
        """Prepare arrays using the same staged options as AC/DC estimators.

        ``defer_prepare_finalize`` stops before state layout; finish that stage
        with ``finalize_prepare``. ``prepare_active_measurements=False`` builds
        the state layout but leaves the active measurement runtime empty.
        """
        if (
            self._prepared
            and network is None
            and measurements is None
            and prepare_active_measurements is None
            and defer_prepare_finalize is None
        ):
            return self
        if self._prepared:
            # Explicit overrides rebuild into a fresh instance. Publish only a
            # fully prepared replacement, so old topology/measurement plans do
            # not survive a changed input (or an unsuccessful preparation).
            replacement = type(self)(
                e_file=self.e_file, meas_file=self.meas_file, parameters=self.params,
                profile=self.profile_enabled, auto_prepare=False,
                power_flow_linear_solver=self.power_flow_linear_solver,
                network=network if network is not None else self._prepare_network,
                measurements=measurements if measurements is not None else self._prepare_measurements,
                prepare_active_measurements=(self._prepare_active_measurements if prepare_active_measurements is None
                                             else prepare_active_measurements),
                defer_prepare_finalize=(self._prepare_defer_finalize if defer_prepare_finalize is None
                                        else defer_prepare_finalize),
            )
            replacement.prepare()
            self.__dict__.clear()
            self.__dict__.update(replacement.__dict__)
            return self
        if network is not None:
            self._prepare_network = network
        if measurements is not None:
            self._prepare_measurements = measurements
        if prepare_active_measurements is not None:
            self._prepare_active_measurements = bool(prepare_active_measurements)
        if defer_prepare_finalize is not None:
            self._prepare_defer_finalize = bool(defer_prepare_finalize)
        result = self._prepare_electric()
        if self._prepared and self._prepare_active_measurements:
            stage_start = time.perf_counter()
            self._prepare_fluid_estimators()
            self._record_profile_time("init.prepare_fluid_estimators", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self._prepare_multi_energy_se_layout()
            self._record_profile_time("init.multi_energy_se_layout", time.perf_counter() - stage_start)
        return result

    def _prepare_measurement_source(self):
        """Return a private, table-backed measurement source for preparation."""
        source = getattr(self, "_prepare_measurements", None)
        if source is None:
            meas_ppc = build_meas_ppc_from_e_file(
                self.meas_file,
                include_strings=False,
                include_matrix=False,
            )
            return meas_ppc, measurement_list_from_meas_ppc(meas_ppc)
        if isinstance(source, dict):
            meas_ppc = copy_meas_ppc(source)
            return meas_ppc, measurement_list_from_meas_ppc(meas_ppc)
        if isinstance(source, MeasurementTable):
            table = source
        else:
            table = getattr(source, "table", None)
            if table is None:
                table = _measurement_table_from_measurements(list(source))
        measurements = TableBackedMeasurementList(
            deepcopy(table),
            normalized=bool(getattr(source, "normalized", False)),
        )
        return None, measurements

    @staticmethod
    def _snapshot_measurement_table(table: MeasurementTable) -> MeasurementTable:
        """Copy only runtime-mutated columns for the fluid-side source.

        Electric preparation changes validity, weights, values and status
        codes.  The topology/name/code arrays are immutable during prepare and
        can be shared, avoiding a second full object/string-array copy while
        keeping the fluid source isolated from electric-side mutations.
        """
        snapshot = copy(table)
        for field_name in ("weight", "valid", "value"):
            values = getattr(table, field_name, None)
            if values is not None:
                setattr(snapshot, field_name, np.asarray(values).copy())
        status_code = getattr(table, "status_code", None)
        if status_code is not None:
            snapshot.status_code = np.asarray(status_code).copy()
        return snapshot

    def _prepare_electric(self) -> "HybridStateEstimator":
        if self._prepared:
            return self
        profile_start = time.perf_counter()
        stage_start = time.perf_counter()
        supplied_network = getattr(self, "_prepare_network", None)
        if supplied_network is not None:
            self.network = supplied_network
            if isinstance(supplied_network, dict):
                if supplied_network.get('format') != 'hybrid_ppc_v1':
                    raise TypeError('Hybrid SE network dictionary must be a hybrid PPC')
                self.network = _build_hybrid_se_network_from_ppc(deepcopy(supplied_network))
            if not hasattr(self.network, 'ac') or not hasattr(self.network, 'dc'):
                raise TypeError('Hybrid SE network must expose AC and DC sides')
            has_ac = _side_has_alive_nodes(getattr(self.network, "ac", None))
            has_dc = _side_has_alive_nodes(getattr(self.network, "dc", None))
            has_coupling = bool(getattr(self.network, "dcac_converters", None))
            if not has_ac and not has_dc and not has_coupling:
                self._fluid_only = True
                rows_kind = "fluid"
            elif (has_ac and has_dc) or has_coupling:
                rows_kind = "hybrid"
            elif has_ac:
                rows_kind = "ac"
            else:
                rows_kind = "dc"
            self.multi_energy = getattr(self.network, "multi_energy", MultiEnergyContext())
        else:
            efile_rows = _read_efile_rows(self.e_file)
            self.multi_energy = build_multi_energy_context_from_rows(
                efile_rows,
                source=self.e_file,
            )
            rows_kind = _detect_se_rows_kind(efile_rows)
            if rows_kind == "ac":
                ac_ppc = build_ac_ppc_with_topology_from_efile_rows(self.e_file, efile_rows)
                base = ac_ppc["base"]
                self.network = _SELightweightHybridNetwork(
                    _se_lightweight=True,
                    ac=_build_ac_se_ppc_namespace(ac_ppc, self.e_file),
                    dc=_empty_se_side(base),
                    dcac_converters=[],
                    hybrid_islands=[],
                    ppc={"format": "hybrid_ppc_v1", "source": str(Path(self.e_file).resolve()), "base": base, "ac": ac_ppc, "dc": None},
                    _ac_ppc=ac_ppc,
                    _dc_ppc=None,
                )
            elif rows_kind == "dc":
                dc_ppc = build_dc_ppc_with_topology_from_efile_rows(self.e_file, efile_rows)
                base = dc_ppc["base"]
                self.network = _SELightweightHybridNetwork(
                    _se_lightweight=True,
                    ac=_empty_se_side(base),
                    dc=_build_dc_se_ppc_namespace(dc_ppc),
                    dcac_converters=[],
                    hybrid_islands=[],
                    ppc={"format": "hybrid_ppc_v1", "source": str(Path(self.e_file).resolve()), "base": base, "ac": None, "dc": dc_ppc},
                    _ac_ppc=None,
                    _dc_ppc=dc_ppc,
                )
            elif rows_kind == "fluid":
                self.network = _build_se_network_from_fluid_only(self.e_file)
                self._fluid_only = True
            else:
                self.network = _build_hybrid_se_network_from_ppc(
                    build_hybrid_ppc_with_topology_from_efile_rows(self.e_file, efile_rows)
                )
        attach_multi_energy_context(self.network, self.multi_energy)
        if rows_kind in ("ac", "dc"):
            base = self.network.ppc["base"]
            self.network.p_base = float(base["p_base"])
            self.network.u_scale = float(base["u_scale"])
            self.network.p_scale = float(base["p_scale"])
            self.network.i_scale = float(base["i_scale"])
            self.network.p_base_kW = float(base["p_base_kW"])
        self._record_profile_time("init.load_network", time.perf_counter() - stage_start)
        self.p_base = float(getattr(self.network, "p_base", 1.0))
        self.p_base_kW = float(getattr(self.network, "p_base_kW", self.p_base))
        self.u_scale = float(getattr(self.network, "u_scale", 1.0))
        self.p_scale = float(getattr(self.network, "p_scale", 1.0))
        self.i_scale = float(getattr(self.network, "i_scale", 1.0))
        # Keep the original table-backed measurement source available for
        # fluid estimators.  It is owned by this estimator and lets pure-fluid
        # and coupled cases share one parse without depending on the later
        # active/pseudo-measurement table.
        if not hasattr(self, "measurements"):
            stage_start = time.perf_counter()
            self.meas_ppc, self.measurements = self._prepare_measurement_source()
            # Keep an estimator-private snapshot.  Electric-side preparation
            # mutates ``self.measurements.table`` (status/value/active rows),
            # while fluid preparation must see the original measurement
            # validity and names.  The snapshot removes the old second file
            # parse without allowing those mutations to leak across domains.
            self._source_measurement_table = self._snapshot_measurement_table(self.measurements.table)
            self._record_profile_time("init.load_measurements", time.perf_counter() - stage_start)
        else:
            source_table = getattr(self.measurements, "table", None)
            self._source_measurement_table = (
                None if source_table is None else self._snapshot_measurement_table(source_table)
            )
        if self._fluid_only:
            self._ac_sub_estimator = None
            self._dc_sub_estimator = None
            self._delegate_estimator = None
            self._prepared = not self._prepare_defer_finalize
            self._defer_prepare_finalize_pending = self._prepare_defer_finalize
            self._record_profile_time("init.total", time.perf_counter() - profile_start)
            return self
        if rows_kind in ("ac", "dc"):
            return self._prepare_uncoupled_direct_delegate(rows_kind, profile_start)
        if not hasattr(self, "measurements"):
            stage_start = time.perf_counter()
            self.meas_ppc, self.measurements = self._prepare_measurement_source()
            self._source_measurement_table = self._snapshot_measurement_table(self.measurements.table)
            self._record_profile_time("init.load_measurements", time.perf_counter() - stage_start)
        self._sub_measurement_sources_by_side = None
        self._sub_measurement_source_rows_by_side = None
        self._sub_measurement_summary_attrs_by_side = {}
        self._sub_measurement_global_summary_attrs = None
        self._sub_measurements_converted_by_side = {"ac": False, "dc": False}
        self._measurements_normalized = bool(getattr(self.measurements, "normalized", False))

        self._sub_estimators_enabled = False
        self._delegate_estimator = None
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        stage_start = time.perf_counter()
        self._ac_sub_estimator = self._build_ac_sub_estimator()
        self._dc_sub_estimator = self._build_dc_sub_estimator()
        self._record_profile_time("init.prepare_sub_estimators", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        # Compatibility object views are only needed after the unified state
        # layout is known. Avoid constructing the large legacy object lists
        # before that point; _finish_state_layout_arrays() installs them once.
        self._build_device_maps(install_views=False)
        self._record_profile_time("init.device_maps", time.perf_counter() - stage_start)
        self._populate_measurement_device_pos_from_sub_estimators(force=True)

        if self._prepare_defer_finalize:
            self._defer_prepare_finalize_pending = True
            self._record_profile_time('init.total', time.perf_counter() - profile_start)
            return self
        if self._try_delegate_uncoupled_single_side():
            self._prepared = True
            self._record_profile_time("init.total", time.perf_counter() - profile_start)
            return self
        return self._finalize_electric_prepare(profile_start)

    def finalize_prepare(
        self, *, prepare_active_measurements: bool = True,
        measurements_already_normalized: bool = False,
    ) -> "HybridStateEstimator":
        """Complete a deferred preparation without reading network files again."""
        self._prepare_active_measurements = bool(prepare_active_measurements)
        self._prepare_defer_finalize = False
        if not hasattr(self, 'network'):
            return self.prepare()
        if measurements_already_normalized and not self._fluid_only:
            self.measurements.normalized = True
            self._measurements_normalized = True
            if isinstance(self.meas_ppc, dict):
                self.meas_ppc['normalized'] = True
        if getattr(self, '_direct_delegate_fast_path', False):
            side = self._direct_delegate_side
            delegate = self._ac_sub_estimator if side == 'ac' else self._dc_sub_estimator
            delegate.finalize_prepare(
                prepare_active_measurements=prepare_active_measurements,
                measurements_already_normalized=measurements_already_normalized,
            )
            self._adopt_delegate(delegate, side)
            self.meas_ppc = getattr(delegate, 'meas_ppc', self.meas_ppc)
            self._prepared = True
        elif not self._fluid_only:
            self._finalize_electric_prepare(time.perf_counter())
        else:
            self._prepared = True
        self._defer_prepare_finalize_pending = False
        if prepare_active_measurements:
            stage_start = time.perf_counter()
            self._prepare_fluid_estimators()
            self._record_profile_time("init.prepare_fluid_estimators", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            self._prepare_multi_energy_se_layout()
            self._record_profile_time("init.multi_energy_se_layout", time.perf_counter() - stage_start)
        return self

    def _finalize_electric_prepare(self, profile_start: float) -> "HybridStateEstimator":
        stage_start = time.perf_counter()
        self._disable_ac_current_measurements()
        self._disable_unavailable_measurements()
        self._convert_measurements_to_pu()
        self._record_profile_time("init.convert_measurements_to_pu", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self._finalize_sub_estimators_after_measurement_prepare()
        self._build_device_maps(install_views=False)
        self._populate_measurement_device_pos_from_sub_estimators(force=True)
        self._record_profile_time("init.finalize_sub_estimators", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        if self._prepare_active_measurements:
            self._add_pseudo_power_measurements()
            self._populate_measurement_device_pos_from_sub_estimators()
        self._record_profile_time("init.add_pseudo_measurements", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self._build_state_layout()
        self._record_profile_time("init.state_layout", time.perf_counter() - stage_start)
        self.calc = self._calc_adapter()
        self.targeted_observability_pseudo_count = 0
        if not self._prepare_active_measurements:
            source = self.measurements
            self.measurements = take_measurement_view(source, np.array([], dtype=np.int64))
            try:
                self._refresh_active_measurement_state_layout()
            finally:
                self.measurements = source
            self._prepared = True
            self._defer_prepare_finalize_pending = False
            self._record_profile_time('init.total', time.perf_counter() - profile_start)
            return self
        stage_start = time.perf_counter()
        self._refresh_active_measurement_state_layout()
        self._record_profile_time("init.refresh_active_measurements", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        fast_observability = self._fast_active_observability_certificate()
        if fast_observability is not None:
            self._initial_observability_cache = fast_observability
            self.required_pseudo_measurement_count = 0
        else:
            self.required_pseudo_measurement_count = self._add_required_missing_pseudo_measurements()
        self._record_profile_time("init.required_missing_pseudo", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self.targeted_observability_pseudo_count = self._add_targeted_observability_pseudo_measurements()
        self._record_profile_time("init.targeted_observability_pseudo", time.perf_counter() - stage_start)
        self.ac_angle_reference_anchor_count = 0
        self._prepared = True
        self._defer_prepare_finalize_pending = False
        self._record_profile_time("init.total", time.perf_counter() - profile_start)
        return self

    def _prepare_fluid_estimators(self) -> None:
        if self.fluid_estimators or not self.multi_energy.fluid_networks:
            return
        # Reuse the already parsed table-backed source.  The array parser keeps
        # a compact device-name dictionary, so fluid estimators can materialize
        # only their rows without a second file read or a second full object
        # parse.  This is per-estimator data, not a process-wide file cache.
        source_table = getattr(self, "_source_measurement_table", None)
        if source_table is None:
            source_table = getattr(getattr(self, "measurements", None), "table", None)
        if source_table is None:
            _, source = self._prepare_measurement_source()
            source_table = source.table
        all_measurements = [
            measurement_from_table_row(source_table, row)
            for row in range(int(source_table.idx.size))
        ]
        estimator_types = {
            "heat": HeatStateEstimator,
            "gas": GasStateEstimator,
            "hydro": HydroStateEstimator,
            "steam": SteamStateEstimator,
        }
        for name, network in self.multi_energy.fluid_networks.items():
            prefix = network.prefix
            measurements = [
                measurement
                for measurement in all_measurements
                if str(measurement.device_type).startswith(prefix)
            ]
            estimator = estimator_types[name](
                network,
                measurements=measurements,
                parameters=self.params,
                flat_start=self.flat_start,
                tol=self.tol,
                max_iter=self.max_iter,
                verbose=False,
            )
            estimator.prepare()
            self.fluid_estimators[name] = estimator

    def _add_multi_energy_warning(self, message: str) -> None:
        if message not in self.multi_energy.warnings:
            self.multi_energy.warnings.append(message)

    def _electric_endpoint_measurement(self, terminal) -> Tuple[Optional[Measurement], str]:
        domain = str(terminal.domain)
        device_type = str(terminal.device_type)
        is_load = device_type.endswith("Load")
        is_source = device_type.endswith(("Generator", "Unit", "Source"))
        if domain not in {"ac", "dc"} or (not is_load and not is_source):
            return None, f"unsupported electric endpoint type {device_type}"

        ppc = getattr(self.network, "ppc", {}) or {}
        side_ppc = ppc.get(domain)
        if not isinstance(side_ppc, dict):
            return None, f"{domain} endpoint has no active estimator block"
        if domain == "ac":
            estimator = self._ac_sub_estimator
            table_key = "load" if is_load else "gen"
            columns = AC_LOAD_COLS if is_load else AC_GEN_COLS
            device_type_code = DEVICE_TYPE_ACLoad if is_load else DEVICE_TYPE_ACGenerator
            meas_type = "P_LOAD" if is_load else "P_GEN"
            meas_type_code = MEAS_TYPE_P_LOAD if is_load else MEAS_TYPE_P_GEN
        else:
            estimator = self._dc_sub_estimator
            table_key = "load" if is_load else "gen"
            columns = DC_LOAD_COLS if is_load else DC_GEN_COLS
            device_type_code = DEVICE_TYPE_DCLoad if is_load else DEVICE_TYPE_DCGenerator
            meas_type = "P_LOAD" if is_load else "P_GEN"
            meas_type_code = MEAS_TYPE_P_LOAD if is_load else MEAS_TYPE_P_GEN
        if estimator is None:
            return None, f"{domain} endpoint has no active state estimator"

        table = np.asarray(side_ppc.get(table_key, ()), dtype=np.float64)
        if table.ndim != 2 or table.size == 0:
            return None, f"missing {device_type} idx={int(terminal.device_idx)}"
        rows = np.flatnonzero(
            table[:, columns["idx"]].astype(np.int64, copy=False)
            == int(terminal.device_idx)
        )
        if rows.size == 0:
            return None, f"missing {device_type} idx={int(terminal.device_idx)}"
        table_row = int(rows[0])
        run_stat_col = columns.get("run_stat")
        if run_stat_col is not None and int(table[table_row, run_stat_col]) != 1:
            return None, f"inactive {device_type} idx={int(terminal.device_idx)}"

        if domain == "ac":
            positions = estimator._plan_pos_for_ppc_rows(
                device_type_code,
                np.asarray([table_row], dtype=np.int64),
                table.shape[0],
            )
            device_pos = int(positions[0]) if positions.size else -1
        else:
            active_rows = np.asarray(
                estimator._load_rows if is_load else estimator._generator_rows,
                dtype=np.int64,
            )
            positions = np.flatnonzero(active_rows == table_row)
            device_pos = int(positions[0]) if positions.size else -1
        if device_pos < 0:
            return None, f"{device_type} idx={int(terminal.device_idx)} is outside the live topology"

        measurement = Measurement(
            0,
            f"coupling_endpoint_{domain}_{device_type_code}_{device_pos}",
            "ACLoad" if domain == "ac" and is_load else
            "ACGenerator" if domain == "ac" else
            "DCLoad" if is_load else "DCGenerator",
            "",
            meas_type,
            1.0,
            True,
            0.0,
            status=MEAS_STATUS_PSEUDO,
            device_type_code=device_type_code,
            meas_type_code=meas_type_code,
            device_pos=device_pos,
        )
        return measurement, ""

    def _fluid_endpoint_measurement(self, terminal) -> Tuple[Optional[Measurement], str]:
        domain = str(terminal.domain)
        estimator = self.fluid_estimators.get(domain)
        if estimator is None:
            return None, f"{domain} endpoint has no active state estimator"
        device_type = str(terminal.device_type)
        is_load = device_type.endswith("Load")
        is_storage = device_type.endswith("Storage")
        is_source = device_type.endswith(("Source", "Unit", "Generator")) or is_storage
        if not is_load and not is_source:
            return None, f"unsupported fluid endpoint type {device_type}"
        if is_load:
            positions = [
                pos
                for pos, device in enumerate(estimator.network.loads)
                if int(device.idx) == int(terminal.device_idx)
            ]
        else:
            positions = [
                pos
                for pos, device in enumerate(estimator.network.sources)
                if bool(estimator.network.source_is_storage[pos]) == is_storage
                and int(device.idx) == int(terminal.device_idx)
            ]
        if not positions:
            return None, f"missing {device_type} idx={int(terminal.device_idx)}"
        device_pos = int(positions[0])
        runtime_device_type = (
            f"{estimator.network.prefix}Load"
            if is_load
            else f"{estimator.network.prefix}Storage"
            if is_storage
            else f"{estimator.network.prefix}Source"
        )
        measurement = Measurement(
            0,
            f"coupling_endpoint_{domain}_{device_pos}",
            runtime_device_type,
            "",
            "FLOW",
            1.0,
            True,
            0.0,
            status=MEAS_STATUS_PSEUDO,
            device_type_code=int(DEVICE_TYPE_CODES.get(runtime_device_type, 0)),
            meas_type_code=int(MEAS_TYPE_CODES.get("FLOW", 0)),
            device_pos=device_pos,
        )
        return measurement, ""

    def _heat_source_temperature_measurement(
        self,
        terminal,
    ) -> Tuple[Optional[Measurement], str]:
        if str(terminal.domain) != "heat" or not str(terminal.device_type).endswith(
            ("Source", "Storage")
        ):
            return None, "electric heating requires a HeatSource or HeatStorage endpoint"
        flow_measurement, reason = self._fluid_endpoint_measurement(terminal)
        if flow_measurement is None:
            return None, reason
        measurement = Measurement(
            0,
            f"coupling_endpoint_heat_temperature_{int(flow_measurement.device_pos)}",
            str(flow_measurement.device_type),
            "",
            "T_SUPPLY",
            1.0,
            True,
            0.0,
            status=MEAS_STATUS_PSEUDO,
            device_type_code=int(flow_measurement.device_type_code),
            meas_type_code=int(MEAS_TYPE_CODES.get("T_SUPPLY", 0)),
            device_pos=int(flow_measurement.device_pos),
        )
        return measurement, ""

    def _resolve_multi_energy_se_endpoint(self, terminal) -> Tuple[Optional[Measurement], str]:
        if str(terminal.domain) in {"ac", "dc"}:
            return self._electric_endpoint_measurement(terminal)
        return self._fluid_endpoint_measurement(terminal)

    def _multi_energy_terminal_setpoint(self, terminal) -> float:
        domain = str(terminal.domain)
        device_type = str(terminal.device_type)
        is_load = device_type.endswith("Load")
        is_storage = device_type.endswith("Storage")
        if domain in {"ac", "dc"}:
            side_ppc = (getattr(self.network, "ppc", {}) or {}).get(domain, {})
            table_key = "load" if is_load else "gen"
            columns = (
                AC_LOAD_COLS if domain == "ac" and is_load else
                AC_GEN_COLS if domain == "ac" else
                DC_LOAD_COLS if is_load else DC_GEN_COLS
            )
            table = np.asarray(side_ppc.get(table_key, ()), dtype=np.float64)
            if table.ndim != 2 or table.size == 0:
                raise ValueError(f"missing {device_type} idx={int(terminal.device_idx)}")
            rows = np.flatnonzero(
                table[:, columns["idx"]].astype(np.int64, copy=False)
                == int(terminal.device_idx)
            )
            if rows.size == 0:
                raise ValueError(f"missing {device_type} idx={int(terminal.device_idx)}")
            column = "pbase" if is_load else "p_set"
            return float(table[int(rows[0]), columns[column]])

        estimator = self.fluid_estimators[domain]
        network = estimator.network
        if is_load:
            positions = [
                pos
                for pos, device in enumerate(network.loads)
                if int(device.idx) == int(terminal.device_idx)
            ]
            if not positions:
                raise ValueError(f"missing {device_type} idx={int(terminal.device_idx)}")
            return float(network.load_flow_set[int(positions[0])])
        positions = [
            pos
            for pos, device in enumerate(network.sources)
            if bool(network.source_is_storage[pos]) == is_storage
            and int(device.idx) == int(terminal.device_idx)
        ]
        if not positions:
            raise ValueError(f"missing {device_type} idx={int(terminal.device_idx)}")
        return float(network.source_flow_set[int(positions[0])])

    def _multi_energy_endpoint_measurement_rows(
        self,
        endpoint: _MultiEnergySEEndpoint,
        meas_type_code: Optional[int] = None,
    ) -> Tuple[int, ...]:
        if endpoint.provider == "electric":
            measurements = self.active_measurements
            measurement_slice = self.electric_measurement_slice
        else:
            measurements = self.fluid_estimators[endpoint.provider].active_measurements
            measurement_slice = self.fluid_measurement_slices[endpoint.provider]
        table = measurements.table
        mask = (
            (np.asarray(table.device_type_code, dtype=np.int64) == endpoint.device_type_code)
            & (
                np.asarray(table.meas_type_code, dtype=np.int64)
                == (
                    endpoint.meas_type_code
                    if meas_type_code is None
                    else int(meas_type_code)
                )
            )
            & (np.asarray(table.device_pos, dtype=np.int64) == endpoint.device_pos)
        )
        return tuple(
            (measurement_slice.start + np.flatnonzero(mask)).astype(np.int64).tolist()
        )

    def _multi_energy_local_measurement_value(self, row: int) -> float:
        row = int(row)
        if self.electric_measurement_slice.start <= row < self.electric_measurement_slice.stop:
            return float(
                self.active_measurements.table.value[
                    row - self.electric_measurement_slice.start
                ]
            )
        for name, measurement_slice in self.fluid_measurement_slices.items():
            if measurement_slice.start <= row < measurement_slice.stop:
                return float(
                    self.fluid_estimators[name].active_measurements.table.value[
                        row - measurement_slice.start
                    ]
                )
        raise IndexError(f"measurement row {row} is outside local measurement slices")

    def _multi_energy_measured_state_prior_value(
        self,
        state_col: int,
        fallback: float,
    ) -> float:
        """Return a direct electric-state measurement without using device names."""
        state_col = int(state_col)
        if not self.electric_state_slice.start <= state_col < self.electric_state_slice.stop:
            return float(fallback)
        meta = state_meta_at(
            self.state_meta,
            state_col - self.electric_state_slice.start,
        )
        if meta is None:
            return float(fallback)
        device_type_code = int(getattr(meta, "device_type_code", -1))
        device_pos = int(getattr(meta, "device_pos", -1))
        meas_type_code = int(getattr(meta, "meas_type_code", -1))
        if min(device_type_code, device_pos, meas_type_code) < 0:
            return float(fallback)

        table = self.active_measurements.table
        table_meas_type = getattr(table, "meas_type_code", None)
        table_device_pos = getattr(table, "device_pos", None)
        if table_meas_type is None or table_device_pos is None:
            return float(fallback)
        mask = (
            np.asarray(table.valid, dtype=bool)
            & (np.asarray(table.device_type_code, dtype=np.int64) == device_type_code)
            & (np.asarray(table_device_pos, dtype=np.int64) == device_pos)
            & (np.asarray(table_meas_type, dtype=np.int64) == meas_type_code)
            & np.isfinite(np.asarray(table.value, dtype=np.float64))
        )
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            return float(fallback)
        best_row = int(rows[np.argmax(np.asarray(table.weight, dtype=np.float64)[rows])])
        return float(table.value[best_row])

    def _prepare_multi_energy_se_layout(self) -> None:
        if self._multi_energy_se_layout_ready or not self.fluid_estimators:
            return

        for estimator in self.fluid_estimators.values():
            estimator.analyze_observability(add_pseudo=True)

        self.electric_state_count = 0 if self._fluid_only else int(self.n_state)
        self.electric_state_slice = slice(0, self.electric_state_count)
        state_offset = self.electric_state_count
        self.fluid_state_slices: Dict[str, slice] = {}
        for name, estimator in self.fluid_estimators.items():
            self.fluid_state_slices[name] = slice(
                state_offset,
                state_offset + int(estimator.state_count),
            )
            state_offset += int(estimator.state_count)
        self.multi_energy_state_count = state_offset

        self.electric_measurement_count = 0 if self._fluid_only else len(self.active_measurements)
        self.electric_measurement_slice = slice(0, self.electric_measurement_count)
        measurement_offset = self.electric_measurement_count
        self.fluid_measurement_slices: Dict[str, slice] = {}
        for name, estimator in self.fluid_estimators.items():
            count = len(estimator.active_measurements)
            self.fluid_measurement_slices[name] = slice(
                measurement_offset,
                measurement_offset + count,
            )
            measurement_offset += count

        endpoint_measurements: Dict[str, MeasurementList] = {
            "electric": MeasurementList()
        }
        endpoint_measurements.update(
            {name: MeasurementList() for name in self.fluid_estimators}
        )
        endpoint_cache: Dict[Tuple[str, str, int, str], _MultiEnergySEEndpoint] = {}

        def register(
            terminal,
            endpoint_kind: str = "flow",
        ) -> Tuple[Optional[_MultiEnergySEEndpoint], str]:
            key = (
                str(terminal.domain),
                str(terminal.device_type),
                int(terminal.device_idx),
                str(endpoint_kind),
            )
            cached = endpoint_cache.get(key)
            if cached is not None:
                return cached, ""
            measurement, reason = (
                self._heat_source_temperature_measurement(terminal)
                if endpoint_kind == "temperature"
                else self._resolve_multi_energy_se_endpoint(terminal)
            )
            if measurement is None:
                return None, reason
            provider = "electric" if str(terminal.domain) in {"ac", "dc"} else str(terminal.domain)
            rows = endpoint_measurements[provider]
            endpoint = _MultiEnergySEEndpoint(
                provider=provider,
                row=len(rows),
                domain=str(terminal.domain),
                device_type=str(terminal.device_type),
                device_idx=int(terminal.device_idx),
                device_type_code=int(measurement.device_type_code),
                meas_type_code=int(measurement.meas_type_code),
                device_pos=int(measurement.device_pos),
            )
            rows.append(measurement)
            endpoint_cache[key] = endpoint
            return endpoint, ""

        pending = []
        for coupling in self.multi_energy.couplings:
            if not coupling.active or not coupling.supports_energy_balance:
                continue
            electric_scale = float(getattr(self.network, "p_base_kW", self.p_base_kW))
            if coupling.is_electric_heat_control:
                electric_terminal = coupling.electric_terminal
                heat_terminal = coupling.heat_terminal
                if electric_terminal is None or heat_terminal is None:
                    self._add_multi_energy_warning(
                        f"{coupling.table_name}:{coupling.name}: missing electric or heat endpoint"
                    )
                    continue
                electric, reason = register(electric_terminal)
                if electric is None:
                    self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                    continue
                heat_flow, reason = register(heat_terminal)
                if heat_flow is None:
                    self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                    continue
                heat_temperature, reason = register(heat_terminal, "temperature")
                if heat_temperature is None:
                    self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                    continue
                heat_estimator = self.fluid_estimators["heat"]
                heat_network = heat_estimator.network
                source_pos = int(heat_flow.device_pos)
                return_node = int(heat_network.source_return_node_pos[source_pos])
                return_temperature_col = (
                    self.fluid_state_slices["heat"].start
                    + heat_estimator.base_temperature
                    + int(heat_network.return_temperature_state_by_node[return_node])
                )
                electric_setpoint = self._multi_energy_terminal_setpoint(electric_terminal)
                supply_temperature_set = float(
                    heat_network.source_supply_temperature_set[source_pos]
                )
                controlled = electric if coupling.control_type == "P" else heat_temperature
                dependent = heat_temperature if coupling.control_type == "P" else electric
                t1 = electric if coupling.t1 == electric_terminal else heat_temperature
                t2 = heat_temperature if coupling.t2 == heat_terminal else electric
                pending.append(
                    SimpleNamespace(
                        coupling=coupling,
                        t1=t1,
                        t2=t2,
                        t1_scale=1.0,
                        t2_scale=1.0,
                        electric_heat=True,
                        gas_heat=False,
                        hydrogen_electric=False,
                        gas_electric=False,
                        steam_electric=False,
                        controlled=controlled,
                        dependent=dependent,
                        electric=electric,
                        gas=None,
                        heat_flow=heat_flow,
                        heat_temperature=heat_temperature,
                        electric_setpoint=float(electric_setpoint),
                        gas_setpoint=0.0,
                        supply_temperature_set=supply_temperature_set,
                        controlled_setpoint=(
                            float(electric_setpoint)
                            if coupling.control_type == "P"
                            else supply_temperature_set
                        ),
                        dependent_setpoint=0.0,
                        return_temperature_col=return_temperature_col,
                        heat_capacity=max(
                            float(heat_network.medium.heat_capacity),
                            1.0e-12,
                        ),
                        power_scale=electric_scale,
                        steam_enthalpy_col=-1,
                        condensate_enthalpy=0.0,
                    )
                )
                continue
            if coupling.is_gas_heat_control:
                gas_terminal = coupling.gas_terminal
                heat_terminal = coupling.heat_terminal
                if gas_terminal is None or heat_terminal is None:
                    self._add_multi_energy_warning(
                        f"{coupling.table_name}:{coupling.name}: missing gas or heat endpoint"
                    )
                    continue
                gas, reason = register(gas_terminal)
                if gas is None:
                    self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                    continue
                heat_flow, reason = register(heat_terminal)
                if heat_flow is None:
                    self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                    continue
                heat_temperature, reason = register(heat_terminal, "temperature")
                if heat_temperature is None:
                    self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                    continue
                heat_estimator = self.fluid_estimators["heat"]
                heat_network = heat_estimator.network
                source_pos = int(heat_flow.device_pos)
                return_node = int(heat_network.source_return_node_pos[source_pos])
                return_temperature_col = (
                    self.fluid_state_slices["heat"].start
                    + heat_estimator.base_temperature
                    + int(heat_network.return_temperature_state_by_node[return_node])
                )
                gas_setpoint = self._multi_energy_terminal_setpoint(gas_terminal)
                supply_temperature_set = float(
                    heat_network.source_supply_temperature_set[source_pos]
                )
                controlled = gas if coupling.control_type == "FLOW" else heat_temperature
                dependent = heat_temperature if coupling.control_type == "FLOW" else gas
                t1 = gas if coupling.t1 == gas_terminal else heat_temperature
                t2 = heat_temperature if coupling.t2 == heat_terminal else gas
                pending.append(
                    SimpleNamespace(
                        coupling=coupling,
                        t1=t1,
                        t2=t2,
                        t1_scale=1.0,
                        t2_scale=1.0,
                        electric_heat=False,
                        gas_heat=True,
                        hydrogen_electric=False,
                        gas_electric=False,
                        steam_electric=False,
                        controlled=controlled,
                        dependent=dependent,
                        electric=None,
                        gas=gas,
                        heat_flow=heat_flow,
                        heat_temperature=heat_temperature,
                        gas_setpoint=float(gas_setpoint),
                        controlled_setpoint=(
                            float(gas_setpoint)
                            if coupling.control_type == "FLOW"
                            else supply_temperature_set
                        ),
                        dependent_setpoint=0.0,
                        return_temperature_col=return_temperature_col,
                        supply_temperature_set=supply_temperature_set,
                        heat_capacity=max(
                            float(heat_network.medium.heat_capacity),
                            1.0e-12,
                        ),
                        power_scale=1.0,
                        electric_setpoint=0.0,
                        steam_enthalpy_col=-1,
                        condensate_enthalpy=0.0,
                    )
                )
                continue
            t1, reason = register(coupling.t1)
            if t1 is None:
                self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                continue
            t2, reason = register(coupling.t2)
            if t2 is None:
                self._add_multi_energy_warning(f"{coupling.table_name}:{coupling.name}: {reason}")
                continue
            if (
                coupling.is_hydrogen_electric_control
                or coupling.is_gas_electric_control
                or coupling.is_steam_electric_control
            ):
                controlled_terminal = coupling.controlled_terminal
                dependent_terminal = coupling.dependent_terminal
                controlled = t1 if coupling.t1 == controlled_terminal else t2
                dependent = t1 if coupling.t1 == dependent_terminal else t2
                controlled_setpoint = self._multi_energy_terminal_setpoint(
                    controlled_terminal
                )
                steam_enthalpy_col = -1
                condensate_enthalpy = 0.0
                if coupling.is_hydrogen_electric_control:
                    dependent_setpoint = hydrogen_electric_dependent_value(
                        coupling,
                        controlled_setpoint,
                        electric_scale,
                    )
                elif coupling.is_gas_electric_control:
                    dependent_setpoint = gas_electric_dependent_value(
                        coupling,
                        controlled_setpoint,
                        electric_scale,
                    )
                else:
                    steam_endpoint = t1 if t1.domain == "steam" else t2
                    steam_estimator = self.fluid_estimators["steam"]
                    steam_network = steam_estimator.network
                    device_pos = int(steam_endpoint.device_pos)
                    if steam_endpoint.device_type.endswith("Load"):
                        steam_node = int(steam_network.load_supply_node_pos[device_pos])
                        condensate_enthalpy = float(
                            steam_network.load_condensate_enthalpy[device_pos]
                        )
                    else:
                        steam_node = int(steam_network.source_supply_node_pos[device_pos])
                        condensate_enthalpy = float(
                            steam_network.medium.feedwater_enthalpy
                        )
                    steam_enthalpy_col = (
                        self.fluid_state_slices["steam"].start
                        + steam_estimator.base_enthalpy
                        + steam_node
                    )
                    dependent_setpoint = steam_electric_dependent_value(
                        coupling,
                        controlled_setpoint,
                        electric_scale,
                        float(steam_network.node_enthalpy[steam_node]),
                        condensate_enthalpy,
                    )
                pending.append(
                    SimpleNamespace(
                        coupling=coupling,
                        t1=t1,
                        t2=t2,
                        t1_scale=1.0,
                        t2_scale=1.0,
                        electric_heat=False,
                        gas_heat=False,
                        hydrogen_electric=coupling.is_hydrogen_electric_control,
                        gas_electric=coupling.is_gas_electric_control,
                        steam_electric=coupling.is_steam_electric_control,
                        controlled=controlled,
                        dependent=dependent,
                        controlled_setpoint=float(controlled_setpoint),
                        dependent_setpoint=float(dependent_setpoint),
                        electric=None,
                        gas=None,
                        heat_flow=None,
                        heat_temperature=None,
                        electric_setpoint=0.0,
                        gas_setpoint=0.0,
                        return_temperature_col=-1,
                        supply_temperature_set=0.0,
                        heat_capacity=1.0,
                        power_scale=electric_scale,
                        steam_enthalpy_col=steam_enthalpy_col,
                        condensate_enthalpy=condensate_enthalpy,
                    )
                )
                continue
            factor = float(coupling.energy_factor)
            pending.append(
                SimpleNamespace(
                    coupling=coupling,
                    t1=t1,
                    t2=t2,
                    t1_scale=electric_scale if t1.domain in {"ac", "dc"} else factor,
                    t2_scale=electric_scale if t2.domain in {"ac", "dc"} else factor,
                    electric_heat=False,
                    gas_heat=False,
                    hydrogen_electric=False,
                    gas_electric=False,
                    steam_electric=False,
                    controlled=None,
                    dependent=None,
                    controlled_setpoint=0.0,
                    dependent_setpoint=0.0,
                    electric=None,
                    gas=None,
                    heat_flow=None,
                    heat_temperature=None,
                    electric_setpoint=0.0,
                    gas_setpoint=0.0,
                    return_temperature_col=-1,
                    supply_temperature_set=0.0,
                    heat_capacity=1.0,
                    power_scale=electric_scale,
                    steam_enthalpy_col=-1,
                    condensate_enthalpy=0.0,
                )
            )

        for measurements in endpoint_measurements.values():
            measurements.table = _measurement_table_from_measurements(measurements)
        self._multi_energy_endpoint_measurements = endpoint_measurements
        self._multi_energy_se_layout_ready = True

        initial_state = self._multi_energy_initial_state()
        endpoint_values, endpoint_jacobians = self._multi_energy_endpoint_runtime(
            initial_state,
            with_jacobian=True,
        )

        def direct_endpoint_state_col(endpoint: _MultiEnergySEEndpoint) -> int:
            """Return the existing state column for an identity endpoint row."""
            endpoint_row = endpoint_jacobians[endpoint.provider].getrow(endpoint.row)
            if endpoint_row.nnz != 1 or not np.isclose(
                endpoint_row.data[0],
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            ):
                return -1
            state_slice = (
                self.electric_state_slice
                if endpoint.provider == "electric"
                else self.fluid_state_slices[endpoint.provider]
            )
            state_col = state_slice.start + int(endpoint_row.indices[0])
            endpoint_value = float(endpoint_values[endpoint.provider][endpoint.row])
            return (
                state_col
                if np.isclose(
                    initial_state[state_col],
                    endpoint_value,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
                else -1
            )

        local_weight_max = 1.0
        for table in self._multi_energy_local_measurement_tables():
            if table.weight.size:
                local_weight_max = max(local_weight_max, float(np.max(table.weight)))
        self.multi_energy_coupling_weight = max(1.0e6, 100.0 * local_weight_max)
        self.multi_energy_coupling_prior_weight = local_weight_max
        control_measurements = MeasurementList()
        control_initial = []
        prepared = []
        for item in pending:
            coupling = item.coupling
            t1 = item.t1
            t2 = item.t2
            t1_scale = item.t1_scale
            t2_scale = item.t2_scale
            if item.electric_heat:
                source_flow = float(
                    endpoint_values[item.heat_flow.provider][item.heat_flow.row]
                )
                t_return = float(initial_state[item.return_temperature_col])
                coefficient = float(coupling.e2h_coeff)
                if coupling.control_type == "P":
                    dependent_setpoint = (
                        t_return
                        + item.electric_setpoint
                        * item.power_scale
                        * coefficient
                        / (source_flow * item.heat_capacity)
                        if abs(source_flow) > 1.0e-12
                        else item.supply_temperature_set
                    )
                    thermal_gap = dependent_setpoint - t_return
                else:
                    dependent_setpoint = (
                        source_flow
                        * item.heat_capacity
                        * (item.supply_temperature_set - t_return)
                        / (item.power_scale * coefficient)
                    )
                    thermal_gap = item.supply_temperature_set - t_return
                normalization = max(
                    1.0,
                    abs(item.electric_setpoint * item.power_scale * coefficient),
                    abs(source_flow * item.heat_capacity * thermal_gap),
                )
                control_col = (
                    direct_endpoint_state_col(item.dependent)
                    if coupling.control_type == "T_OUT"
                    else -1
                )
                if control_col < 0:
                    control_col = self.multi_energy_state_count + len(control_initial)
                    control_initial.append(float(dependent_setpoint))
                controlled_measurement_rows = (
                    self._multi_energy_endpoint_measurement_rows(item.controlled)
                )
                dependent_measurement_rows = (
                    self._multi_energy_endpoint_measurement_rows(item.dependent)
                )
                heat_supply_measurement_rows = (
                    self._multi_energy_endpoint_measurement_rows(item.heat_temperature)
                )
                heat_power_measurement_rows = (
                    self._multi_energy_endpoint_measurement_rows(
                        item.heat_flow,
                        MEAS_TYPE_CODES["HEAT"],
                    )
                )
                prior_measurement_row = -1
                if not dependent_measurement_rows:
                    prior_measurement_row = measurement_offset + len(control_measurements)
                    control_measurements.append(
                        Measurement(
                            prior_measurement_row + 1,
                            f"coupling_state_prior_{coupling.table_name}_{coupling.idx}",
                            str(coupling.table_name),
                            str(coupling.name),
                            "COUPLING_STATE_PRIOR",
                            self.multi_energy_coupling_prior_weight,
                            True,
                            float(dependent_setpoint),
                            status=MEAS_STATUS_PSEUDO,
                        )
                    )
                controlled_prior_measurement_row = -1
                controlled_prior_state_col = -1
                if controlled_measurement_rows:
                    # 清空受控端量测行雅可比时，若受控端对应子域独立单位状态列（如 AC_LOAD_P），
                    # 必须追加弱权重观测先验锚定该列，防止盲区下该状态列沦为全零不可观测列。
                    ctrl_row = endpoint_jacobians[item.controlled.provider].getrow(
                        item.controlled.row
                    )
                    if (
                        ctrl_row.nnz == 1
                        and abs(float(ctrl_row.data[0]) - 1.0) <= 1.0e-12
                    ):
                        ctrl_slice = (
                            self.electric_state_slice
                            if item.controlled.provider == "electric"
                            else self.fluid_state_slices[item.controlled.provider]
                        )
                        cand_ctrl_col = ctrl_slice.start + int(ctrl_row.indices[0])
                        if cand_ctrl_col != control_col:
                            controlled_prior_state_col = cand_ctrl_col
                            controlled_prior_measurement_row = measurement_offset + len(
                                control_measurements
                            )
                            ctrl_prior_val = self._multi_energy_measured_state_prior_value(
                                controlled_prior_state_col,
                                float(item.controlled_setpoint),
                            )
                            control_measurements.append(
                                Measurement(
                                    controlled_prior_measurement_row + 1,
                                    f"coupling_ctrl_obs_prior_{coupling.table_name}_{coupling.idx}",
                                    str(coupling.table_name),
                                    str(coupling.name),
                                    "OBSERVABILITY_STATE_PRIOR",
                                    min(self.multi_energy_coupling_prior_weight, 1.0e-6),
                                    True,
                                    float(ctrl_prior_val),
                                    status=MEAS_STATUS_PSEUDO,
                                )
                            )
                prepared.append(
                    SimpleNamespace(
                        coupling=coupling,
                        t1=t1,
                        t2=t2,
                        t1_scale=1.0,
                        t2_scale=1.0,
                        normalization=float(normalization),
                        control_col=control_col,
                        prior_measurement_row=prior_measurement_row,
                        controlled_prior_measurement_row=controlled_prior_measurement_row,
                        controlled_prior_state_col=controlled_prior_state_col,
                        electric_heat=True,
                        gas_heat=False,
                        hydrogen_electric=False,
                        gas_electric=False,
                        steam_electric=False,
                        controlled=item.controlled,
                        dependent=item.dependent,
                        electric=item.electric,
                        gas=None,
                        heat_flow=item.heat_flow,
                        heat_temperature=item.heat_temperature,
                        controlled_setpoint=float(item.controlled_setpoint),
                        controlled_measurement_rows=controlled_measurement_rows,
                        dependent_measurement_rows=dependent_measurement_rows,
                        heat_supply_measurement_rows=heat_supply_measurement_rows,
                        heat_power_measurement_rows=heat_power_measurement_rows,
                        return_temperature_col=item.return_temperature_col,
                        supply_temperature_set=item.supply_temperature_set,
                        heat_capacity=item.heat_capacity,
                        power_scale=item.power_scale,
                        electric_setpoint=item.electric_setpoint,
                        gas_setpoint=0.0,
                        steam_enthalpy_col=-1,
                        condensate_enthalpy=0.0,
                    )
                )
                continue
            if item.gas_heat:
                heat_flow = float(
                    endpoint_values[item.heat_flow.provider][item.heat_flow.row]
                )
                t_return = float(initial_state[item.return_temperature_col])
                coefficient = float(coupling.g2h_coeff)
                if coupling.control_type == "FLOW":
                    dependent_setpoint = (
                        t_return
                        + item.gas_setpoint
                        * coefficient
                        / (heat_flow * item.heat_capacity)
                        if abs(heat_flow) > 1.0e-12
                        else item.supply_temperature_set
                    )
                    thermal_gap = dependent_setpoint - t_return
                else:
                    dependent_setpoint = (
                        heat_flow
                        * item.heat_capacity
                        * (item.supply_temperature_set - t_return)
                        / coefficient
                    )
                    thermal_gap = item.supply_temperature_set - t_return
                normalization = max(
                    1.0,
                    abs(item.gas_setpoint * coefficient),
                    abs(heat_flow * item.heat_capacity * thermal_gap),
                )
                control_col = (
                    direct_endpoint_state_col(item.dependent)
                    if coupling.control_type == "T_OUT"
                    else -1
                )
                if control_col < 0:
                    control_col = self.multi_energy_state_count + len(control_initial)
                    control_initial.append(float(dependent_setpoint))
                controlled_measurement_rows = (
                    self._multi_energy_endpoint_measurement_rows(item.controlled)
                )
                dependent_measurement_rows = (
                    self._multi_energy_endpoint_measurement_rows(item.dependent)
                )
                heat_supply_measurement_rows = (
                    self._multi_energy_endpoint_measurement_rows(item.heat_temperature)
                )
                heat_power_measurement_rows = (
                    self._multi_energy_endpoint_measurement_rows(
                        item.heat_flow,
                        MEAS_TYPE_CODES["HEAT"],
                    )
                )
                prior_measurement_row = -1
                if not dependent_measurement_rows:
                    prior_measurement_row = measurement_offset + len(control_measurements)
                    control_measurements.append(
                        Measurement(
                            prior_measurement_row + 1,
                            f"coupling_state_prior_{coupling.table_name}_{coupling.idx}",
                            str(coupling.table_name),
                            str(coupling.name),
                            "COUPLING_STATE_PRIOR",
                            self.multi_energy_coupling_prior_weight,
                            True,
                            float(dependent_setpoint),
                            status=MEAS_STATUS_PSEUDO,
                        )
                    )
                controlled_prior_measurement_row = -1
                controlled_prior_state_col = -1
                if controlled_measurement_rows:
                    ctrl_row = endpoint_jacobians[item.controlled.provider].getrow(
                        item.controlled.row
                    )
                    if (
                        ctrl_row.nnz == 1
                        and abs(float(ctrl_row.data[0]) - 1.0) <= 1.0e-12
                    ):
                        ctrl_slice = (
                            self.electric_state_slice
                            if item.controlled.provider == "electric"
                            else self.fluid_state_slices[item.controlled.provider]
                        )
                        cand_ctrl_col = ctrl_slice.start + int(ctrl_row.indices[0])
                        if cand_ctrl_col != control_col:
                            controlled_prior_state_col = cand_ctrl_col
                            controlled_prior_measurement_row = measurement_offset + len(
                                control_measurements
                            )
                            ctrl_prior_val = self._multi_energy_measured_state_prior_value(
                                controlled_prior_state_col,
                                float(item.controlled_setpoint),
                            )
                            control_measurements.append(
                                Measurement(
                                    controlled_prior_measurement_row + 1,
                                    f"coupling_ctrl_obs_prior_{coupling.table_name}_{coupling.idx}",
                                    str(coupling.table_name),
                                    str(coupling.name),
                                    "OBSERVABILITY_STATE_PRIOR",
                                    min(self.multi_energy_coupling_prior_weight, 1.0e-6),
                                    True,
                                    float(ctrl_prior_val),
                                    status=MEAS_STATUS_PSEUDO,
                                )
                            )
                prepared.append(
                    SimpleNamespace(
                        coupling=coupling,
                        t1=item.t1,
                        t2=item.t2,
                        t1_scale=1.0,
                        t2_scale=1.0,
                        normalization=float(normalization),
                        control_col=control_col,
                        prior_measurement_row=prior_measurement_row,
                        controlled_prior_measurement_row=controlled_prior_measurement_row,
                        controlled_prior_state_col=controlled_prior_state_col,
                        electric_heat=False,
                        gas_heat=True,
                        hydrogen_electric=False,
                        gas_electric=False,
                        steam_electric=False,
                        controlled=item.controlled,
                        dependent=item.dependent,
                        electric=None,
                        gas=item.gas,
                        heat_flow=item.heat_flow,
                        heat_temperature=item.heat_temperature,
                        controlled_setpoint=float(item.controlled_setpoint),
                        controlled_measurement_rows=controlled_measurement_rows,
                        dependent_measurement_rows=dependent_measurement_rows,
                        heat_supply_measurement_rows=heat_supply_measurement_rows,
                        heat_power_measurement_rows=heat_power_measurement_rows,
                        return_temperature_col=item.return_temperature_col,
                        supply_temperature_set=item.supply_temperature_set,
                        heat_capacity=item.heat_capacity,
                        power_scale=1.0,
                        electric_setpoint=0.0,
                        gas_setpoint=item.gas_setpoint,
                        steam_enthalpy_col=-1,
                        condensate_enthalpy=0.0,
                    )
                )
                continue
            t1_value = float(endpoint_values[t1.provider][t1.row])
            t2_value = float(endpoint_values[t2.provider][t2.row])
            direct_fluid_electric = (
                item.hydrogen_electric
                or item.gas_electric
                or item.steam_electric
            )
            dependent_setpoint = float(item.dependent_setpoint)
            if item.steam_electric:
                dependent_setpoint = steam_electric_dependent_value(
                    coupling,
                    item.controlled_setpoint,
                    item.power_scale,
                    float(initial_state[item.steam_enthalpy_col]),
                    item.condensate_enthalpy,
                )
            endpoint = item.dependent if direct_fluid_electric else t1
            endpoint_value = (
                float(endpoint_values[endpoint.provider][endpoint.row])
                if direct_fluid_electric
                else t1_value
            )
            if item.steam_electric:
                electric_value = (
                    item.controlled_setpoint
                    if item.controlled.domain in {"ac", "dc"}
                    else dependent_setpoint
                )
                steam_flow = (
                    item.controlled_setpoint
                    if item.controlled.domain == "steam"
                    else dependent_setpoint
                )
                steam_kw = (
                    steam_flow
                    * (
                        float(initial_state[item.steam_enthalpy_col])
                        - item.condensate_enthalpy
                    )
                    / 3.6
                )
                normalization = max(
                    1.0,
                    abs(electric_value * item.power_scale),
                    abs(steam_kw * float(coupling.steam_electric_coeff)),
                )
            elif direct_fluid_electric:
                normalization = max(1.0, abs(endpoint_value), abs(dependent_setpoint))
            else:
                normalization = max(
                    1.0,
                    abs(t1_value * t1_scale),
                    abs(t2_value * t2_scale),
                )
            endpoint_row = endpoint_jacobians[endpoint.provider].getrow(endpoint.row)
            control_col = -1
            prior_measurement_row = -1
            reused_unit_state_col = -1
            if (
                direct_fluid_electric
                and endpoint_row.nnz == 1
                and abs(float(endpoint_row.data[0]) - 1.0) <= 1.0e-12
            ):
                # 当受控电端（如 AC+DC 混合模式下的 ACGenerator P_GEN / DCGenerator P_GEN）
                # 在子域状态向量中已有专属单位状态列时，直接复用该列作为耦合控制列，
                # 避免新建重复控制列并覆写 dependent_measurement_rows 导致原子域列沦为全零不可观测列。
                endpoint_slice = (
                    self.electric_state_slice
                    if endpoint.provider == "electric"
                    else self.fluid_state_slices[endpoint.provider]
                )
                reused_unit_state_col = endpoint_slice.start + int(endpoint_row.indices[0])
            if reused_unit_state_col >= 0:
                control_col = reused_unit_state_col
            elif direct_fluid_electric or endpoint_row.nnz == 0:
                control_col = self.multi_energy_state_count + len(control_initial)
                control_initial.append(
                    dependent_setpoint if direct_fluid_electric else t1_value
                )
                if not direct_fluid_electric:
                    prior_measurement_row = measurement_offset + len(control_measurements)
                    control_measurements.append(
                        Measurement(
                            prior_measurement_row + 1,
                            f"coupling_control_{coupling.table_name}_{coupling.idx}",
                            str(coupling.table_name),
                            str(coupling.name),
                            "T1_CONTROL",
                            self.multi_energy_coupling_prior_weight,
                            True,
                            t1_value,
                            status=MEAS_STATUS_PSEUDO,
                        )
                    )
            controlled_measurement_rows = (
                self._multi_energy_endpoint_measurement_rows(item.controlled)
                if direct_fluid_electric else ()
            )
            dependent_measurement_rows = (
                self._multi_energy_endpoint_measurement_rows(item.dependent)
                if direct_fluid_electric else ()
            )
            prior_state_col = -1
            if (
                item.hydrogen_electric
                and coupling.hydrogen_electric_direction == "E2H"
                and coupling.control_type == "P"
            ):
                controlled_row = endpoint_jacobians[item.controlled.provider].getrow(
                    item.controlled.row
                )
                if controlled_row.nnz:
                    local_pos = int(
                        np.argmax(np.abs(controlled_row.data))
                    )
                    state_slice = (
                        self.electric_state_slice
                        if item.controlled.provider == "electric"
                        else self.fluid_state_slices[item.controlled.provider]
                    )
                    prior_state_col = (
                        state_slice.start
                        + int(controlled_row.indices[local_pos])
                    )
                if prior_state_col >= 0:
                    prior_measurement_row = measurement_offset + len(control_measurements)
                    prior_value = self._multi_energy_measured_state_prior_value(
                        prior_state_col,
                        float(initial_state[prior_state_col]),
                    )
                    control_measurements.append(
                        Measurement(
                            prior_measurement_row + 1,
                            f"coupling_state_prior_{coupling.table_name}_{coupling.idx}",
                            str(coupling.table_name),
                            str(coupling.name),
                            "OBSERVABILITY_STATE_PRIOR",
                            min(self.multi_energy_coupling_prior_weight, 1.0e-6),
                            True,
                            prior_value,
                            status=MEAS_STATUS_PSEUDO,
                        )
                    )
            prepared.append(
                SimpleNamespace(
                    coupling=coupling,
                    t1=t1,
                    t2=t2,
                    t1_scale=float(t1_scale),
                    t2_scale=float(t2_scale),
                    normalization=float(normalization),
                    control_col=control_col,
                    prior_measurement_row=prior_measurement_row,
                    prior_state_col=prior_state_col,
                    electric_heat=False,
                    gas_heat=False,
                    hydrogen_electric=item.hydrogen_electric,
                    gas_electric=item.gas_electric,
                    steam_electric=item.steam_electric,
                    controlled=item.controlled,
                    dependent=item.dependent,
                    controlled_setpoint=item.controlled_setpoint,
                    controlled_measurement_rows=controlled_measurement_rows,
                    dependent_measurement_rows=dependent_measurement_rows,
                    electric=None,
                    gas=None,
                    heat_flow=None,
                    heat_temperature=None,
                    return_temperature_col=-1,
                    supply_temperature_set=0.0,
                    heat_capacity=1.0,
                    power_scale=item.power_scale,
                    electric_setpoint=0.0,
                    gas_setpoint=0.0,
                    steam_enthalpy_col=item.steam_enthalpy_col,
                    condensate_enthalpy=item.condensate_enthalpy,
                )
            )

        self._multi_energy_coupling_state_initial = np.asarray(control_initial, dtype=np.float64)
        self.multi_energy_coupling_state_slice = slice(
            self.multi_energy_state_count,
            self.multi_energy_state_count + len(control_initial),
        )
        self.multi_energy_state_count = self.multi_energy_coupling_state_slice.stop
        self.multi_energy_control_measurement_slice = slice(
            measurement_offset,
            measurement_offset + len(control_measurements),
        )
        energy_measurement_offset = self.multi_energy_control_measurement_slice.stop
        energy_measurements = MeasurementList()
        self.multi_energy_se_coupling_plans = []
        for item in prepared:
            coupling = item.coupling
            measurement_row = energy_measurement_offset + len(energy_measurements)
            self.multi_energy_se_coupling_plans.append(
                _MultiEnergySECouplingPlan(
                    coupling=coupling,
                    t1=item.t1,
                    t2=item.t2,
                    t1_scale=item.t1_scale,
                    t2_scale=item.t2_scale,
                    normalization=item.normalization,
                    measurement_row=measurement_row,
                    control_col=item.control_col,
                    prior_measurement_row=item.prior_measurement_row,
                    prior_state_col=getattr(item, "prior_state_col", -1),
                    controlled_prior_measurement_row=getattr(
                        item, "controlled_prior_measurement_row", -1
                    ),
                    controlled_prior_state_col=getattr(
                        item, "controlled_prior_state_col", -1
                    ),
                    electric_heat=item.electric_heat,
                    gas_heat=item.gas_heat,
                    hydrogen_electric=item.hydrogen_electric,
                    gas_electric=item.gas_electric,
                    steam_electric=item.steam_electric,
                    controlled=item.controlled,
                    dependent=item.dependent,
                    electric=item.electric,
                    gas=item.gas,
                    heat_flow=item.heat_flow,
                    heat_temperature=item.heat_temperature,
                    controlled_setpoint=item.controlled_setpoint,
                    controlled_measurement_rows=item.controlled_measurement_rows,
                    dependent_measurement_rows=item.dependent_measurement_rows,
                    heat_supply_measurement_rows=getattr(
                        item,
                        "heat_supply_measurement_rows",
                        (),
                    ),
                    heat_power_measurement_rows=getattr(
                        item,
                        "heat_power_measurement_rows",
                        (),
                    ),
                    electric_setpoint=item.electric_setpoint,
                    gas_setpoint=item.gas_setpoint,
                    return_temperature_col=item.return_temperature_col,
                    supply_temperature_set=item.supply_temperature_set,
                    heat_capacity=item.heat_capacity,
                    power_scale=item.power_scale,
                    steam_enthalpy_col=item.steam_enthalpy_col,
                    condensate_enthalpy=item.condensate_enthalpy,
                )
            )
            energy_measurements.append(
                Measurement(
                    measurement_row + 1,
                    f"coupling_{coupling.table_name}_{coupling.idx}",
                    str(coupling.table_name),
                    str(coupling.name),
                    "ENERGY_BALANCE",
                    self.multi_energy_coupling_weight,
                    True,
                    0.0,
                    status=MEAS_STATUS_PSEUDO,
                )
            )
        coupling_measurements = MeasurementList([*control_measurements, *energy_measurements])
        coupling_measurements.table = _measurement_table_from_measurements(coupling_measurements)
        self._multi_energy_coupling_measurements = coupling_measurements
        self.multi_energy_coupling_measurement_slice = slice(
            energy_measurement_offset,
            energy_measurement_offset + len(energy_measurements),
        )
        self.multi_energy_measurement_count = energy_measurement_offset + len(energy_measurements)
        self._multi_energy_measurement_table = self._build_multi_energy_measurement_table()
        self._multi_energy_z = np.asarray(self._multi_energy_measurement_table.value, dtype=np.float64)
        self._multi_energy_weight = np.asarray(self._multi_energy_measurement_table.weight, dtype=np.float64)
        voltage_cols = []
        if self.electric_state_count:
            raw_me_v = np.asarray(self.voltage_cols, dtype=np.int64)
            if raw_me_v.size:
                voltage_cols.extend(np.unique(raw_me_v[raw_me_v >= 0]).tolist())
        self._multi_energy_voltage_cols = np.asarray(voltage_cols, dtype=np.int64)
        self._multi_energy_fluid_potential_cols = np.concatenate(
            [
                np.arange(state_slice.start, state_slice.start + estimator.n_potential, dtype=np.int64)
                for name, estimator in self.fluid_estimators.items()
                for state_slice in (self.fluid_state_slices[name],)
            ]
        ) if self.fluid_estimators else np.asarray([], dtype=np.int64)
        self._multi_energy_fluid_compressible_potential_cols = np.concatenate(
            [
                np.arange(state_slice.start, state_slice.start + estimator.n_potential, dtype=np.int64)
                for name, estimator in self.fluid_estimators.items()
                if float(estimator.network.potential_power) > 1.0
                for state_slice in (self.fluid_state_slices[name],)
            ]
        ) if any(float(est.network.potential_power) > 1.0 for est in self.fluid_estimators.values()) else np.asarray([], dtype=np.int64)
        electric_labels = [] if self._fluid_only else list(self.state_labels)
        self.multi_energy_state_labels = electric_labels + [
            label
            for name, estimator in self.fluid_estimators.items()
            for label in estimator.state_labels
        ] + [
            f"Coupling:{plan.coupling.table_name}:{plan.coupling.name}:"
            f"{'TEMPERATURE' if (plan.electric_heat and plan.coupling.control_type == 'P') or (plan.gas_heat and plan.coupling.control_type == 'FLOW') else 'POWER' if plan.electric_heat else 'GAS_FLOW' if plan.gas_heat else 'DEPENDENT' if plan.hydrogen_electric or plan.gas_electric or plan.steam_electric else 'T1_CONTROL'}"
            for plan in self.multi_energy_se_coupling_plans
            if self.multi_energy_coupling_state_slice.start
            <= plan.control_col
            < self.multi_energy_coupling_state_slice.stop
        ]
        self._multi_energy_observability_cache = None

    def _multi_energy_local_measurement_tables(self) -> List[MeasurementTable]:
        tables = []
        if not self._fluid_only:
            tables.append(
                self._measurement_table_with_string_columns(
                    self._measurement_table(self.active_measurements)
                )
            )
        tables.extend(
            estimator.active_measurements.table
            for estimator in self.fluid_estimators.values()
        )
        return tables

    def _build_multi_energy_measurement_table(self) -> MeasurementTable:
        tables = self._multi_energy_local_measurement_tables()
        tables.append(self._multi_energy_coupling_measurements.table)
        combined = _measurement_table_from_measurements(MeasurementList())
        for table in tables:
            combined = concat_measurement_tables(combined, table)
        combined.idx = np.arange(1, combined.idx.size + 1, dtype=np.int64)
        return combined

    def _multi_energy_initial_state(self) -> np.ndarray:
        parts = []
        if not self._fluid_only:
            parts.append(np.asarray(self.initial_state(), dtype=np.float64))
        parts.extend(
            np.asarray(estimator.initial_state(), dtype=np.float64)
            for estimator in self.fluid_estimators.values()
        )
        coupling_initial = np.asarray(
            getattr(self, "_multi_energy_coupling_state_initial", np.asarray([], dtype=np.float64)),
            dtype=np.float64,
        )
        if coupling_initial.size:
            parts.append(coupling_initial.copy())
        return np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)

    def _multi_energy_endpoint_runtime(self, x: np.ndarray, *, with_jacobian: bool):
        values = {}
        jacobians = {}
        for provider, measurements in self._multi_energy_endpoint_measurements.items():
            if not measurements:
                values[provider] = np.asarray([], dtype=np.float64)
                if with_jacobian:
                    state_slice = (
                        self.electric_state_slice
                        if provider == "electric"
                        else self.fluid_state_slices[provider]
                    )
                    jacobians[provider] = csr_matrix((0, state_slice.stop - state_slice.start))
                continue
            if provider == "electric":
                local_x = x[self.electric_state_slice]
                values[provider] = self.evaluate(local_x, measurements)
                if with_jacobian:
                    jacobians[provider] = self.jacobian_sparse(local_x, measurements)
            else:
                estimator = self.fluid_estimators[provider]
                local_x = x[self.fluid_state_slices[provider]]
                values[provider] = estimator.evaluate(local_x, measurements)
                if with_jacobian:
                    jacobians[provider] = estimator.jacobian_sparse(local_x, measurements)
        return values, jacobians

    def _evaluate_multi_energy(self, x: np.ndarray) -> np.ndarray:
        self._prepare_multi_energy_se_layout()
        state = np.asarray(x, dtype=np.float64)
        values = np.empty(self.multi_energy_measurement_count, dtype=np.float64)
        if self.electric_measurement_count:
            values[self.electric_measurement_slice] = self.evaluate(
                state[self.electric_state_slice],
                self.active_measurements,
            )
        for name, estimator in self.fluid_estimators.items():
            values[self.fluid_measurement_slices[name]] = estimator.evaluate(
                state[self.fluid_state_slices[name]],
            )
        endpoint_values, _endpoint_jacobians = self._multi_energy_endpoint_runtime(
            state,
            with_jacobian=False,
        )
        for plan in self.multi_energy_se_coupling_plans:
            if plan.electric_heat:
                dependent_value = float(state[plan.control_col])
                if plan.prior_measurement_row >= 0:
                    values[plan.prior_measurement_row] = dependent_value
                if plan.controlled_prior_measurement_row >= 0 and plan.controlled_prior_state_col >= 0:
                    values[plan.controlled_prior_measurement_row] = float(
                        state[plan.controlled_prior_state_col]
                    )
                source_flow = float(
                    endpoint_values[plan.heat_flow.provider][plan.heat_flow.row]
                )
                t_return = float(state[plan.return_temperature_col])
                if plan.coupling.control_type == "P":
                    electric_power = plan.electric_setpoint
                    t_out = dependent_value
                else:
                    electric_power = dependent_value
                    t_out = plan.supply_temperature_set
                if plan.heat_supply_measurement_rows:
                    values[
                        np.asarray(plan.heat_supply_measurement_rows, dtype=np.int64)
                    ] = t_out
                if plan.heat_power_measurement_rows:
                    values[
                        np.asarray(plan.heat_power_measurement_rows, dtype=np.int64)
                    ] = (
                        source_flow
                        * plan.heat_capacity
                        * (t_out - t_return)
                    )
                if plan.controlled_measurement_rows:
                    values[np.asarray(plan.controlled_measurement_rows, dtype=np.int64)] = (
                        plan.controlled_setpoint
                    )
                if plan.dependent_measurement_rows:
                    values[np.asarray(plan.dependent_measurement_rows, dtype=np.int64)] = (
                        dependent_value
                    )
                values[plan.measurement_row] = electric_heat_balance_residual(
                    plan.coupling,
                    electric_power,
                    source_flow,
                    t_out,
                    t_return,
                    plan.heat_capacity,
                    plan.power_scale,
                ) / plan.normalization
                continue
            if plan.gas_heat:
                dependent_value = float(state[plan.control_col])
                if plan.prior_measurement_row >= 0:
                    values[plan.prior_measurement_row] = dependent_value
                if plan.controlled_prior_measurement_row >= 0 and plan.controlled_prior_state_col >= 0:
                    values[plan.controlled_prior_measurement_row] = float(
                        state[plan.controlled_prior_state_col]
                    )
                heat_flow = float(
                    endpoint_values[plan.heat_flow.provider][plan.heat_flow.row]
                )
                t_return = float(state[plan.return_temperature_col])
                if plan.coupling.control_type == "FLOW":
                    gas_flow = plan.gas_setpoint
                    t_out = dependent_value
                else:
                    gas_flow = dependent_value
                    t_out = plan.supply_temperature_set
                if plan.heat_supply_measurement_rows:
                    values[
                        np.asarray(plan.heat_supply_measurement_rows, dtype=np.int64)
                    ] = t_out
                if plan.heat_power_measurement_rows:
                    values[
                        np.asarray(plan.heat_power_measurement_rows, dtype=np.int64)
                    ] = (
                        heat_flow
                        * plan.heat_capacity
                        * (t_out - t_return)
                    )
                if plan.controlled_measurement_rows:
                    values[np.asarray(plan.controlled_measurement_rows, dtype=np.int64)] = (
                        plan.controlled_setpoint
                    )
                if plan.dependent_measurement_rows:
                    values[np.asarray(plan.dependent_measurement_rows, dtype=np.int64)] = (
                        dependent_value
                    )
                values[plan.measurement_row] = gas_heat_balance_residual(
                    plan.coupling,
                    gas_flow,
                    heat_flow,
                    t_out,
                    t_return,
                    plan.heat_capacity,
                ) / plan.normalization
                continue
            if plan.hydrogen_electric or plan.gas_electric or plan.steam_electric:
                dependent_value = (
                    float(state[plan.control_col])
                    if plan.control_col >= 0
                    else float(
                        endpoint_values[plan.dependent.provider][plan.dependent.row]
                    )
                )
                if plan.prior_measurement_row >= 0:
                    prior_col = (
                        plan.prior_state_col
                        if plan.prior_state_col >= 0
                        else plan.control_col
                    )
                    values[plan.prior_measurement_row] = state[prior_col]
                if plan.hydrogen_electric and plan.controlled_measurement_rows:
                    values[np.asarray(plan.controlled_measurement_rows, dtype=np.int64)] = (
                        plan.controlled_setpoint
                    )
                if plan.dependent_measurement_rows:
                    values[np.asarray(plan.dependent_measurement_rows, dtype=np.int64)] = (
                        dependent_value
                    )
                if plan.hydrogen_electric:
                    expected = hydrogen_electric_dependent_value(
                        plan.coupling,
                        plan.controlled_setpoint,
                        plan.power_scale,
                    )
                    coupling_residual = dependent_value - expected
                elif plan.gas_electric:
                    expected = gas_electric_dependent_value(
                        plan.coupling,
                        plan.controlled_setpoint,
                        plan.power_scale,
                    )
                    coupling_residual = dependent_value - expected
                else:
                    expected = steam_electric_dependent_value(
                        plan.coupling,
                        plan.controlled_setpoint,
                        plan.power_scale,
                        float(state[plan.steam_enthalpy_col]),
                        plan.condensate_enthalpy,
                    )
                    coupling_residual = dependent_value - expected
                values[plan.measurement_row] = coupling_residual / plan.normalization
                continue
            if plan.control_col >= 0:
                t1_value = float(state[plan.control_col])
                values[plan.prior_measurement_row] = t1_value
            else:
                t1_value = float(endpoint_values[plan.t1.provider][plan.t1.row])
            t2_value = float(endpoint_values[plan.t2.provider][plan.t2.row])
            values[plan.measurement_row] = (
                abs(t2_value) * plan.t2_scale
                - float(plan.coupling.efficiency) * abs(t1_value) * plan.t1_scale
            ) / plan.normalization
        return values

    @staticmethod
    def _append_multi_energy_endpoint_jacobian(
        rows: List[int],
        cols: List[int],
        data: List[float],
        *,
        target_row: int,
        endpoint: _MultiEnergySEEndpoint,
        endpoint_jacobians: Dict[str, csr_matrix],
        state_slice: slice,
        scale: float,
    ) -> None:
        jacobian_row = endpoint_jacobians[endpoint.provider].getrow(endpoint.row)
        start, stop = jacobian_row.indptr[0], jacobian_row.indptr[1]
        if stop <= start:
            return
        rows.extend([target_row] * (stop - start))
        cols.extend((state_slice.start + jacobian_row.indices[start:stop]).tolist())
        data.extend((scale * jacobian_row.data[start:stop]).tolist())

    def _multi_energy_jacobian_sparse(self, x: np.ndarray) -> csr_matrix:
        self._prepare_multi_energy_se_layout()
        state = np.asarray(x, dtype=np.float64)
        local_blocks = []
        if not self._fluid_only:
            local_blocks.append(
                self.jacobian_sparse(state[self.electric_state_slice], self.active_measurements)
            )
        local_blocks.extend(
            estimator.jacobian_sparse(state[self.fluid_state_slices[name]])
            for name, estimator in self.fluid_estimators.items()
        )
        local_jacobian = (
            block_diag(local_blocks, format="csr")
            if local_blocks
            else csr_matrix((0, self.multi_energy_state_count), dtype=np.float64)
        )
        if local_jacobian.shape[1] < self.multi_energy_state_count:
            local_jacobian = hstack(
                (
                    local_jacobian,
                    csr_matrix(
                        (
                            local_jacobian.shape[0],
                            self.multi_energy_state_count - local_jacobian.shape[1],
                        ),
                        dtype=np.float64,
                    ),
                ),
                format="csr",
            )
        if any(
            plan.hydrogen_electric
            or plan.gas_electric
            or plan.steam_electric
            or plan.electric_heat
            or plan.gas_heat
            for plan in self.multi_energy_se_coupling_plans
        ):
            local_jacobian = local_jacobian.tolil()
            for plan in self.multi_energy_se_coupling_plans:
                if not (
                    plan.hydrogen_electric
                    or plan.gas_electric
                    or plan.steam_electric
                    or plan.electric_heat
                    or plan.gas_heat
                ):
                    continue
                if plan.electric_heat or plan.gas_heat or plan.hydrogen_electric:
                    for row in plan.controlled_measurement_rows:
                        local_jacobian.rows[row] = []
                        local_jacobian.data[row] = []
                if plan.control_col < 0:
                    continue
                for row in plan.dependent_measurement_rows:
                    local_jacobian.rows[row] = [plan.control_col]
                    local_jacobian.data[row] = [1.0]
            local_jacobian = local_jacobian.tocsr()
        endpoint_values, endpoint_jacobians = self._multi_energy_endpoint_runtime(
            state,
            with_jacobian=True,
        )
        if any(
            plan.heat_power_measurement_rows
            for plan in self.multi_energy_se_coupling_plans
        ):
            local_jacobian = local_jacobian.tolil()
            for plan in self.multi_energy_se_coupling_plans:
                if not plan.heat_power_measurement_rows:
                    continue
                source_flow = float(
                    endpoint_values[plan.heat_flow.provider][plan.heat_flow.row]
                )
                t_return = float(state[plan.return_temperature_col])
                variable_outlet = (
                    plan.electric_heat and plan.coupling.control_type == "P"
                ) or (
                    plan.gas_heat and plan.coupling.control_type == "FLOW"
                )
                t_out = (
                    float(state[plan.control_col])
                    if variable_outlet
                    else plan.supply_temperature_set
                )
                flow_row = endpoint_jacobians[plan.heat_flow.provider].getrow(
                    plan.heat_flow.row
                )
                flow_slice = self.fluid_state_slices[plan.heat_flow.provider]
                flow_start, flow_stop = flow_row.indptr[0], flow_row.indptr[1]
                flow_cols = (
                    flow_slice.start + flow_row.indices[flow_start:flow_stop]
                ).tolist()
                flow_data = (
                    plan.heat_capacity
                    * (t_out - t_return)
                    * flow_row.data[flow_start:flow_stop]
                ).tolist()
                for row in plan.heat_power_measurement_rows:
                    cols = [*flow_cols, plan.return_temperature_col]
                    row_data = [
                        *flow_data,
                        -source_flow * plan.heat_capacity,
                    ]
                    if variable_outlet:
                        cols.append(plan.control_col)
                        row_data.append(source_flow * plan.heat_capacity)
                    local_jacobian.rows[row] = cols
                    local_jacobian.data[row] = row_data
            local_jacobian = local_jacobian.tocsr()
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        added_row_offset = self.multi_energy_control_measurement_slice.start
        for plan in self.multi_energy_se_coupling_plans:
            local_row = plan.measurement_row - added_row_offset
            if plan.electric_heat:
                if plan.prior_measurement_row >= 0:
                    rows.append(plan.prior_measurement_row - added_row_offset)
                    cols.append(
                        plan.prior_state_col
                        if plan.prior_state_col >= 0
                        else plan.control_col
                    )
                    data.append(1.0)
                if plan.controlled_prior_measurement_row >= 0 and plan.controlled_prior_state_col >= 0:
                    rows.append(plan.controlled_prior_measurement_row - added_row_offset)
                    cols.append(plan.controlled_prior_state_col)
                    data.append(1.0)
                source_flow = float(
                    endpoint_values[plan.heat_flow.provider][plan.heat_flow.row]
                )
                t_return = float(state[plan.return_temperature_col])
                t_out = (
                    float(state[plan.control_col])
                    if plan.coupling.control_type == "P"
                    else plan.supply_temperature_set
                )
                flow_slice = self.fluid_state_slices[plan.heat_flow.provider]
                self._append_multi_energy_endpoint_jacobian(
                    rows,
                    cols,
                    data,
                    target_row=local_row,
                    endpoint=plan.heat_flow,
                    endpoint_jacobians=endpoint_jacobians,
                    state_slice=flow_slice,
                    scale=(
                        -plan.heat_capacity
                        * (t_out - t_return)
                        / plan.normalization
                    ),
                )
                rows.append(local_row)
                cols.append(plan.return_temperature_col)
                data.append(source_flow * plan.heat_capacity / plan.normalization)
                rows.append(local_row)
                cols.append(plan.control_col)
                data.append(
                    -source_flow * plan.heat_capacity / plan.normalization
                    if plan.coupling.control_type == "P"
                    else plan.power_scale
                    * float(plan.coupling.e2h_coeff)
                    / plan.normalization
                )
                continue
            if plan.gas_heat:
                if plan.prior_measurement_row >= 0:
                    rows.append(plan.prior_measurement_row - added_row_offset)
                    cols.append(plan.control_col)
                    data.append(1.0)
                if plan.controlled_prior_measurement_row >= 0 and plan.controlled_prior_state_col >= 0:
                    rows.append(plan.controlled_prior_measurement_row - added_row_offset)
                    cols.append(plan.controlled_prior_state_col)
                    data.append(1.0)
                heat_flow = float(
                    endpoint_values[plan.heat_flow.provider][plan.heat_flow.row]
                )
                t_return = float(state[plan.return_temperature_col])
                t_out = (
                    float(state[plan.control_col])
                    if plan.coupling.control_type == "FLOW"
                    else plan.supply_temperature_set
                )
                heat_slice = self.fluid_state_slices[plan.heat_flow.provider]
                self._append_multi_energy_endpoint_jacobian(
                    rows,
                    cols,
                    data,
                    target_row=local_row,
                    endpoint=plan.heat_flow,
                    endpoint_jacobians=endpoint_jacobians,
                    state_slice=heat_slice,
                    scale=(
                        -plan.heat_capacity
                        * (t_out - t_return)
                        / plan.normalization
                    ),
                )
                rows.append(local_row)
                cols.append(plan.return_temperature_col)
                data.append(heat_flow * plan.heat_capacity / plan.normalization)
                rows.append(local_row)
                cols.append(plan.control_col)
                data.append(
                    -heat_flow * plan.heat_capacity / plan.normalization
                    if plan.coupling.control_type == "FLOW"
                    else float(plan.coupling.g2h_coeff) / plan.normalization
                )
                continue
            if plan.hydrogen_electric or plan.gas_electric or plan.steam_electric:
                if plan.prior_measurement_row >= 0:
                    rows.append(plan.prior_measurement_row - added_row_offset)
                    cols.append(
                        plan.prior_state_col
                        if plan.prior_state_col >= 0
                        else plan.control_col
                    )
                    data.append(1.0)
                if plan.control_col >= 0:
                    rows.append(local_row)
                    cols.append(plan.control_col)
                    if not plan.steam_electric:
                        data.append(1.0 / plan.normalization)
                    else:
                        enthalpy_gap = (
                            float(state[plan.steam_enthalpy_col])
                            - plan.condensate_enthalpy
                        )
                        expected = steam_electric_dependent_value(
                            plan.coupling,
                            plan.controlled_setpoint,
                            plan.power_scale,
                            float(state[plan.steam_enthalpy_col]),
                            plan.condensate_enthalpy,
                        )
                        data.append(1.0 / plan.normalization)
                        rows.append(local_row)
                        cols.append(plan.steam_enthalpy_col)
                        enthalpy_derivative = (
                            expected / enthalpy_gap
                            if plan.coupling.control_type == "P"
                            else -expected / enthalpy_gap
                        )
                        data.append(enthalpy_derivative / plan.normalization)
                else:
                    dependent_slice = (
                        self.electric_state_slice
                        if plan.dependent.provider == "electric"
                        else self.fluid_state_slices[plan.dependent.provider]
                    )
                    self._append_multi_energy_endpoint_jacobian(
                        rows,
                        cols,
                        data,
                        target_row=local_row,
                        endpoint=plan.dependent,
                        endpoint_jacobians=endpoint_jacobians,
                        state_slice=dependent_slice,
                        scale=1.0 / plan.normalization,
                    )
                continue
            if plan.control_col >= 0:
                t1_value = float(state[plan.control_col])
                prior_row = plan.prior_measurement_row - added_row_offset
                rows.append(prior_row)
                cols.append(plan.control_col)
                data.append(1.0)
            else:
                t1_value = float(endpoint_values[plan.t1.provider][plan.t1.row])
            t2_value = float(endpoint_values[plan.t2.provider][plan.t2.row])
            t1_sign = 1.0 if t1_value >= 0.0 else -1.0
            t2_sign = 1.0 if t2_value >= 0.0 else -1.0
            t1_slice = (
                self.electric_state_slice
                if plan.t1.provider == "electric"
                else self.fluid_state_slices[plan.t1.provider]
            )
            t2_slice = (
                self.electric_state_slice
                if plan.t2.provider == "electric"
                else self.fluid_state_slices[plan.t2.provider]
            )
            if plan.control_col >= 0:
                rows.append(local_row)
                cols.append(plan.control_col)
                data.append(
                    -float(plan.coupling.efficiency)
                    * t1_sign
                    * plan.t1_scale
                    / plan.normalization
                )
            else:
                self._append_multi_energy_endpoint_jacobian(
                    rows,
                    cols,
                    data,
                    target_row=local_row,
                    endpoint=plan.t1,
                    endpoint_jacobians=endpoint_jacobians,
                    state_slice=t1_slice,
                    scale=(
                        -float(plan.coupling.efficiency)
                        * t1_sign
                        * plan.t1_scale
                        / plan.normalization
                    ),
                )
            self._append_multi_energy_endpoint_jacobian(
                rows,
                cols,
                data,
                target_row=local_row,
                endpoint=plan.t2,
                endpoint_jacobians=endpoint_jacobians,
                state_slice=t2_slice,
                scale=t2_sign * plan.t2_scale / plan.normalization,
            )
        coupling_jacobian = coo_matrix(
            (
                np.asarray(data, dtype=np.float64),
                (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
            ),
            shape=(
                self.multi_energy_measurement_count - added_row_offset,
                self.multi_energy_state_count,
            ),
        ).tocsr()
        return vstack((local_jacobian, coupling_jacobian), format="csr")

    def _apply_multi_energy_state_bounds(
        self,
        x: np.ndarray,
        ref_x: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self._multi_energy_voltage_cols.size:
            x[self._multi_energy_voltage_cols] = np.maximum(
                x[self._multi_energy_voltage_cols],
                self.voltage_floor,
            )
        if self._multi_energy_fluid_potential_cols.size:
            x[self._multi_energy_fluid_potential_cols] = np.maximum(
                x[self._multi_energy_fluid_potential_cols],
                1.0e-12,
            )
        comp_cols = getattr(self, "_multi_energy_fluid_compressible_potential_cols", None)
        if ref_x is not None and comp_cols is not None and comp_cols.size:
            # 可压缩流体（Steam/Gas, u = P^2）防穿零逐分量保正下界，避免多压级节点首步跌入 1e-12 真空奇点
            x[comp_cols] = np.maximum(
                x[comp_cols],
                np.maximum(1.0e-12, 0.25 * ref_x[comp_cols]),
            )
        steam_estimator = self.fluid_estimators.get("steam")
        if steam_estimator is not None:
            state_slice = self.fluid_state_slices["steam"]
            enthalpy_start = state_slice.start + steam_estimator.base_enthalpy
            minimum_enthalpy = max(
                float(steam_estimator.network.medium.feedwater_enthalpy),
                float(np.max(steam_estimator.network.load_condensate_enthalpy))
                if steam_estimator.network.load_condensate_enthalpy.size
                else -np.inf,
            ) + 1.0e-6
            maximum_enthalpy = max(
                float(np.max(steam_estimator.network.source_enthalpy_set))
                if steam_estimator.network.source_enthalpy_set.size
                else -np.inf,
                float(np.max(steam_estimator.network.node_enthalpy))
                if steam_estimator.network.node_enthalpy.size
                else -np.inf,
            )
            # 仅防物理发散（2 倍参考上限），避免将带正常正向波动（如 +5%）的蒸汽焓状态锁死在设定值上导致线搜索停滞
            upper_enthalpy = max(maximum_enthalpy * 2.0, minimum_enthalpy + 1000.0) if np.isfinite(maximum_enthalpy) else np.inf
            x[enthalpy_start:state_slice.stop] = np.clip(
                x[enthalpy_start:state_slice.stop],
                minimum_enthalpy,
                upper_enthalpy,
            )
        return x

    def _multi_energy_measurement_residual(
        self,
        z_est: np.ndarray,
        *,
        measurement_rows: Optional[np.ndarray] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        rows = None if measurement_rows is None else np.asarray(measurement_rows, dtype=np.int64)
        z = self._multi_energy_z if rows is None else self._multi_energy_z[rows]
        if out is None:
            return z - z_est
        return np.subtract(z, z_est, out=out)

    def _multi_energy_observability_analysis(
        self,
        x: Optional[np.ndarray] = None,
        H=None,
        measurement_rows: Optional[np.ndarray] = None,
    ) -> ObservabilityResult:
        self._prepare_multi_energy_se_layout()
        state = self._multi_energy_initial_state() if x is None else np.asarray(x, dtype=np.float64)
        rows = (
            np.arange(self.multi_energy_measurement_count, dtype=np.int64)
            if measurement_rows is None
            else np.asarray(measurement_rows, dtype=np.int64)
        )
        jacobian = self._multi_energy_jacobian_sparse(state) if H is None else H
        if jacobian.shape[0] != rows.size:
            jacobian = jacobian[rows]
        measurement_count = int(rows.size)
        if matrix_is_empty(jacobian):
            result = ObservabilityResult(
                False,
                0,
                self.multi_energy_state_count,
                measurement_count,
                self.multi_energy_state_count,
                np.asarray([], dtype=np.float64),
                [],
            )
        else:
            structural_rank = sparse_structural_rank(jacobian)
            if structural_rank == self.multi_energy_state_count:
                result = ObservabilityResult(
                    True,
                    self.multi_energy_state_count,
                    self.multi_energy_state_count,
                    measurement_count,
                    0,
                    np.asarray([], dtype=np.float64),
                    [],
                )
            else:
                rank, deficiency, singular_values, weak_states = observability_rank_details(
                    jacobian,
                    self.multi_energy_state_count,
                )
                result = ObservabilityResult(
                    rank == self.multi_energy_state_count,
                    rank,
                    self.multi_energy_state_count,
                    measurement_count,
                    max(0, deficiency),
                    singular_values,
                    weak_states,
                )
        self.multi_energy_observability_result = result
        self._multi_energy_observability_cache = {
            "result": result,
            "x": state.copy(),
            "H": jacobian,
            "measurement_rows": rows.copy(),
        }
        return result

    def _multi_energy_state_base_scales(self, x_ref: Optional[np.ndarray] = None) -> np.ndarray:
        """构造电-气-热-水-蒸汽多能联合状态基准向量 D_x（电侧保持 1.0，流体与耦合变量按物理特征值标幺化）。"""
        n_total = int(self.multi_energy_state_count)
        scales = np.ones(n_total, dtype=np.float64)
        if n_total <= 0:
            return scales
        # 各流体子域直接复用各自物理基准（势能 p^k、受控流量、供回水温度、蒸汽比焓），电侧已是标幺值保持 1.0
        for name, estimator in self.fluid_estimators.items():
            sl = self.fluid_state_slices.get(name)
            if sl is None:
                continue
            sub_fn = getattr(estimator, "state_base_scales", None)
            if callable(sub_fn):
                sub_scales = np.asarray(sub_fn(), dtype=np.float64)
                if sub_scales.size == (sl.stop - sl.start):
                    scales[sl] = np.maximum(sub_scales, 1e-6)
        # 耦合控制状态按初值特征幅值归一（下限 1.0）
        c_sl = getattr(self, "multi_energy_coupling_state_slice", slice(0, 0))
        if c_sl.stop > c_sl.start:
            c_init = getattr(self, "_multi_energy_coupling_state_initial", None)
            if c_init is not None and np.asarray(c_init).size == (c_sl.stop - c_sl.start):
                scales[c_sl] = np.maximum(np.abs(np.asarray(c_init, dtype=np.float64)), 1.0)
            elif x_ref is not None and np.asarray(x_ref).size >= c_sl.stop:
                scales[c_sl] = np.maximum(np.abs(np.asarray(x_ref[c_sl], dtype=np.float64)), 1.0)
        return scales

    def _estimate_multi_energy(
        self,
        *,
        x0: Optional[np.ndarray] = None,
        measurement_rows: Optional[np.ndarray] = None,
        verbose: bool = False,
        final_diagnostics: bool = True,
        observability: Optional[ObservabilityResult] = None,
        _hint_lm_damping: float = 0.0,
    ) -> EstimateResult:
        self._prepare_multi_energy_se_layout()
        x = self._multi_energy_initial_state() if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        if x.size != self.multi_energy_state_count:
            raise ValueError(
                f"multi-energy initial state has {x.size} values; expected {self.multi_energy_state_count}"
            )
        rows = (
            np.arange(self.multi_energy_measurement_count, dtype=np.int64)
            if measurement_rows is None
            else np.asarray(measurement_rows, dtype=np.int64)
        )
        observability = observability or self._multi_energy_observability_analysis(
            x,
            measurement_rows=rows,
        )
        cached_H = None
        cache = self._multi_energy_observability_cache
        if cache is not None and cache.get("result") is observability:
            cached_x = np.asarray(cache.get("x"), dtype=np.float64)
            cached_rows = np.asarray(cache.get("measurement_rows"), dtype=np.int64)
            if (
                cached_x.shape == x.shape
                and np.array_equal(cached_x, x)
                and np.array_equal(cached_rows, rows)
            ):
                cached_H = cache.get("H")

        n_me_state = int(self.multi_energy_state_count)
        weight = np.asarray(self._multi_energy_weight[rows], dtype=np.float64)
        uniform_weight = self._uniform_weight(weight)
        weights_are_uniform = uniform_weight is not None
        weighted_residual = None if weights_are_uniform else np.empty_like(weight)
        use_aat = n_me_state >= 64
        aat_solver = (
            CholmodAAtNormalEquationSolver(assume_fixed_pattern=False, ordering_method="auto", multi_solve=True)
            if use_aat
            else None
        )
        aat_plan: Optional[CholmodAAtNormalEquationPlan] = None
        normal_solver = NormalEquationSolver(assume_fixed_pattern=False)
        normal_pattern = None
        z_est = np.empty(rows.size, dtype=np.float64)
        residual = np.empty_like(z_est)
        candidate_z_est = np.empty_like(z_est)
        candidate_residual = np.empty_like(z_est)
        converged = False
        objective = np.inf
        residual_inf = np.inf
        max_correction = np.inf
        scaled_correction = np.inf
        iteration = 0
        H = None
        gain = None
        evaluated_at_final_x = False
        z_est_valid = False
        if verbose:
            _print_iteration_header()

        me_v_cols = np.asarray(getattr(self, "_multi_energy_voltage_cols", np.asarray([], dtype=np.int64)), dtype=np.int64)
        me_raw_a_cols = np.asarray(getattr(self, "ac_theta_state_col", np.asarray([], dtype=np.int32)), dtype=np.int32)
        me_a_cols = me_raw_a_cols[me_raw_a_cols >= 0] if me_raw_a_cols.size else np.asarray([], dtype=np.int32)
        me_v_min = float(self.voltage_min)
        me_v_max = float(self.voltage_max)
        me_a_min = float(self.angle_min_rad)
        me_a_max = float(self.angle_max_rad)

        def _me_state_out_of_bounds(state_vec: np.ndarray) -> bool:
            if me_v_cols.size:
                v_part = state_vec[me_v_cols]
                if not np.all(np.isfinite(v_part)) or float(v_part.min()) < me_v_min or float(v_part.max()) > me_v_max:
                    return True
            if me_a_cols.size:
                a_part = state_vec[me_a_cols]
                if not np.all(np.isfinite(a_part)) or float(a_part.min()) < me_a_min or float(a_part.max()) > me_a_max:
                    return True
            return False

        me_state_scales = self._multi_energy_state_base_scales(x)

        def _solve_me_step(H_cur, res_cur: np.ndarray, damping_val: float):
            """在电-气-热-水统一无量纲标幺状态空间 x_pu = x / D_x 下组装求解正规方程，再还原物理增量 dx = D_x * dx_pu。"""
            nonlocal aat_plan, normal_pattern, gain
            if use_aat and aat_solver is not None and is_sparse_matrix(H_cur) and H_cur.nnz > 0:
                H_csr = H_cur if getattr(H_cur, "format", None) == "csr" else H_cur.tocsr()
                if not H_csr.has_sorted_indices:
                    H_csr.sort_indices()
                if aat_plan is None or not aat_plan.matches(H_csr):
                    aat_plan = CholmodAAtNormalEquationPlan.from_jacobian(H_csr)
                A_csc, rhs_vec = aat_plan.assemble(
                    H_csr,
                    res_cur,
                    weight,
                    assume_fixed_weights=False,
                    copy_rhs=False,
                    assume_pattern_matches=True,
                )
                # 统一标幺化：A_pu = D_x * A_csc，rhs_pu = D_x * rhs_vec（无阻尼步与阻尼步全程消除跨能域尺度差）
                A_pu = A_csc.copy()
                A_pu.data = A_csc.data * me_state_scales[A_csc.indices]
                rhs_pu = rhs_vec * me_state_scales
                diag_pu = np.bincount(
                    A_pu.indices,
                    weights=A_pu.data * A_pu.data,
                    minlength=n_me_state,
                )
                diag_g = diag_pu / np.maximum(me_state_scales * me_state_scales, 1e-24)
                if damping_val > 0.0:
                    # 在标幺正规方程上叠加尺度不变 Jacobi 列均衡 Marquardt 相对阻尼
                    col_scale = 1.0 / np.sqrt(np.maximum(diag_pu, 1e-12))
                    A_scaled = A_pu.copy()
                    A_scaled.data = A_pu.data * col_scale[A_pu.indices]
                    rhs_scaled = rhs_pu * col_scale
                    dx_scaled, _ = aat_solver.solve(
                        A_scaled,
                        rhs_scaled,
                        return_factor_diag=False,
                        damping=damping_val,
                    )
                    dx_pu = np.asarray(dx_scaled, dtype=np.float64) * col_scale
                else:
                    dx_raw, _ = aat_solver.solve(
                        A_pu,
                        rhs_pu,
                        return_factor_diag=False,
                        damping=0.0,
                    )
                    dx_pu = np.asarray(dx_raw, dtype=np.float64)
                dx_out = dx_pu * me_state_scales
                return dx_out, rhs_vec, diag_g

            # 中小规模同样在标幺状态空间 H_pu = H * D_x 组装良态正规方程
            if is_sparse_matrix(H_cur):
                H_csr_s = H_cur if getattr(H_cur, "format", None) == "csr" else H_cur.tocsr()
                H_pu = H_csr_s.copy()
                if H_pu.nnz > 0:
                    H_pu.data = H_csr_s.data * me_state_scales[H_csr_s.indices]
            else:
                H_pu = np.asarray(H_cur, dtype=np.float64) * me_state_scales[None, :]
            if weighted_residual is not None:
                np.multiply(weight, res_cur, out=weighted_residual)
            gain_pu, rhs_pu = build_normal_equations(
                H_pu,
                res_cur,
                weight,
                uniform_weight=uniform_weight,
                weights_are_uniform=weights_are_uniform,
                weighted_residual=weighted_residual,
                normal_pattern=None,
                assume_normal_pattern_matches=False,
            )
            rhs_vec = rhs_pu / np.maximum(me_state_scales, 1e-12)
            gain = None
            if is_sparse_matrix(gain_pu):
                diag_pu = np.asarray(gain_pu.diagonal(), dtype=np.float64)
                diag_g = diag_pu / np.maximum(me_state_scales * me_state_scales, 1e-24)
                zero_diag_mask = diag_pu <= 1e-12
                if damping_val > 0.0 or bool(np.any(zero_diag_mask)):
                    col_scale = np.where(zero_diag_mask, 1.0, 1.0 / np.sqrt(np.maximum(diag_pu, 1e-12)))
                    D_scale = diags(col_scale, format="csc")
                    scaled_g = (D_scale @ gain_pu.tocsc() @ D_scale).tocsc()
                    eff_damp = max(damping_val, 1e-6)
                    diag_add = np.where(zero_diag_mask, 1.0 + eff_damp, eff_damp)
                    damped_g = (scaled_g + diags(diag_add, format="csc")).tocsc()
                    rhs_scaled = np.where(zero_diag_mask, 0.0, rhs_pu * col_scale)
                    dx_scaled, _ = normal_solver.solve(damped_g, rhs_scaled, return_factor_diag=False)
                    dx_pu = np.where(zero_diag_mask, 0.0, np.asarray(dx_scaled, dtype=np.float64) * col_scale)
                else:
                    dx_pu, _ = normal_solver.solve(gain_pu, rhs_pu, return_factor_diag=False)
            else:
                diag_pu = np.asarray(np.diag(gain_pu), dtype=np.float64)
                diag_g = diag_pu / np.maximum(me_state_scales * me_state_scales, 1e-24)
                zero_diag_mask = diag_pu <= 1e-12
                if damping_val > 0.0 or bool(np.any(zero_diag_mask)):
                    col_scale = np.where(zero_diag_mask, 1.0, 1.0 / np.sqrt(np.maximum(diag_pu, 1e-12)))
                    scaled_g = np.asarray(gain_pu, dtype=np.float64) * col_scale[:, None] * col_scale[None, :]
                    eff_damp = max(damping_val, 1e-6)
                    scaled_g[np.diag_indices_from(scaled_g)] += np.where(zero_diag_mask, 1.0 + eff_damp, eff_damp)
                    rhs_scaled = np.where(zero_diag_mask, 0.0, rhs_pu * col_scale)
                    dx_scaled, _ = normal_solver.solve(scaled_g, rhs_scaled, return_factor_diag=False)
                    dx_pu = np.where(zero_diag_mask, 0.0, np.asarray(dx_scaled, dtype=np.float64) * col_scale)
                else:
                    dx_pu, _ = normal_solver.solve(gain_pu, rhs_pu, return_factor_diag=False)
            dx_raw = np.asarray(dx_pu, dtype=np.float64) * me_state_scales
            return dx_raw, rhs_vec, diag_g

        force_lm_damping = max(0.0, float(_hint_lm_damping))
        last_used_damping = 0.0
        backtrack_streak = 0
        fullstep_osc_streak = 0
        smooth_fullstep_streak = 0
        stagnation_streak = 0
        high_damp_stagnation_streak = 0
        high_damp_slow_streak = 0
        low_progress_streak = 0
        damp_relaxed_flag = False
        damp_cycle_count = 0
        min_cycle_damping = 0.0
        cycle_slow_streak = 0
        curvature_damping_active = force_lm_damping > 0.0
        tried_undamped_escape = False
        prev_scaled_correction = np.inf
        self._last_estimate_cholmod_factor = None

        for iteration in range(1, self.max_iter + 1):
            if _me_state_out_of_bounds(x):
                break
            if not z_est_valid:
                z_est[:] = self._evaluate_multi_energy(x)[rows]
                self._multi_energy_measurement_residual(
                    z_est,
                    measurement_rows=rows,
                    out=residual,
                )
                objective = self._weighted_objective(weight, residual)
                residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
                z_est_valid = True
            if iteration == 1 and cached_H is not None:
                H = cached_H
                cached_H = None
            else:
                H = self._multi_energy_jacobian_sparse(x)[rows]

            step_damping = float(force_lm_damping)
            dx = None
            rhs = None
            diag_g = None
            try:
                dx, rhs, diag_g = _solve_me_step(H, residual, step_damping)
            except Exception:
                dx = None
            if dx is None or (dx.size and not np.all(np.isfinite(dx))):
                for beta_try in (1e-4, 1e-2, 1e-1, 1.0, 10.0):
                    if beta_try <= step_damping:
                        continue
                    try:
                        dx_try, rhs_try, diag_try = _solve_me_step(H, residual, beta_try)
                        if dx_try.size == n_me_state and np.all(np.isfinite(dx_try)):
                            dx, rhs, diag_g = dx_try, rhs_try, diag_try
                            step_damping = beta_try
                            force_lm_damping = beta_try
                            break
                    except Exception:
                        continue
            if dx is None or (dx.size and not np.all(np.isfinite(dx))):
                break

            last_used_damping = step_damping
            # 有效上下界约束投影：已紧贴 voltage_floor 下界或 voltage_max 上界且搜索方向继续指向墙外的电压分量，
            # 其投影步长与投影梯度均为 0，防止单节点贴墙锁死误判未收敛
            eff_dx = dx
            eff_rhs = rhs
            if me_v_cols.size and dx.size:
                v_at_floor = x[me_v_cols] <= (float(self.voltage_floor) + 1e-5)
                v_at_ceil = x[me_v_cols] >= (float(self.voltage_max) - 1e-5)
                if bool(np.any(v_at_floor)) or bool(np.any(v_at_ceil)):
                    pinned_dx = me_v_cols[(v_at_floor & (dx[me_v_cols] < 0.0)) | (v_at_ceil & (dx[me_v_cols] > 0.0))]
                    if pinned_dx.size:
                        eff_dx = dx.copy()
                        eff_dx[pinned_dx] = 0.0
                    if rhs is not None and rhs.size == dx.size:
                        pinned_rhs = me_v_cols[
                            (v_at_floor & (rhs[me_v_cols] < 0.0)) | (v_at_ceil & (rhs[me_v_cols] > 0.0))
                        ]
                        if pinned_rhs.size:
                            eff_rhs = rhs.copy()
                            eff_rhs[pinned_rhs] = 0.0
            max_correction = float(np.max(np.abs(eff_dx))) if eff_dx.size else 0.0
            scaled_correction = (
                float(np.max(np.abs(eff_dx) / np.maximum(1.0, np.abs(x))))
                if eff_dx.size
                else 0.0
            )
            kkt_inf = (
                float(np.max(np.abs(eff_rhs) / np.maximum(diag_g, 1.0)))
                if (eff_rhs is not None and diag_g is not None and eff_rhs.size > 0)
                else np.inf
            )
            if step_damping == 0.0 and scaled_correction < self.tol:
                converged = True
                evaluated_at_final_x = True
                if verbose:
                    _print_iteration(iteration, objective, residual_inf, max_correction, None, True)
                break
            if (
                step_damping > 0.0
                and scaled_correction < max(1e-5, 10.0 * self.tol)
                and kkt_inf <= max(1e-4, 100.0 * self.tol)
            ):
                converged = True
                evaluated_at_final_x = True
                if verbose:
                    _print_iteration(iteration, objective, residual_inf, max_correction, None, True)
                break

            accepted = False
            stationary = False
            step_scale = 1.0
            if me_v_cols.size and dx.size:
                max_v_dx = float(np.max(np.abs(dx[me_v_cols])))
                if max_v_dx > 0.35:
                    step_scale = min(step_scale, 0.35 / max_v_dx)
            if me_a_cols.size and dx.size:
                max_a_dx = float(np.max(np.abs(dx[me_a_cols])))
                if max_a_dx > 0.75:
                    step_scale = min(step_scale, 0.75 / max_a_dx)

            previous_objective = objective
            obj_tol = max(1e-12, 1e-10 * (abs(objective) + 1.0))
            ls_slope = -float(np.dot(rhs, dx)) if (rhs is not None and rhs.size == dx.size) else 0.0
            oob_trials = 0
            for _attempt in range(20):
                raw_candidate = x + step_scale * dx
                if not np.all(np.isfinite(raw_candidate)):
                    oob_trials += 1
                    if oob_trials >= 3:
                        break
                    step_scale *= 0.5
                    continue
                # 先投影下界（如 voltage_floor / 流体势能防穿零下界 / 蒸汽焓下界），避免电压贴近下界轻碰时在 it=1 误判越界
                candidate = self._apply_multi_energy_state_bounds(raw_candidate, ref_x=x)
                if _me_state_out_of_bounds(candidate):
                    oob_trials += 1
                    if oob_trials >= 3:
                        break
                    step_scale *= 0.5
                    continue
                candidate_z_est[:] = self._evaluate_multi_energy(candidate)[rows]
                self._multi_energy_measurement_residual(
                    candidate_z_est,
                    measurement_rows=rows,
                    out=candidate_residual,
                )
                candidate_objective = self._weighted_objective(weight, candidate_residual)
                if np.isfinite(candidate_objective) and candidate_objective <= objective + obj_tol:
                    relative_objective_change = abs(objective - candidate_objective) / max(1.0, abs(objective))
                    actual_scaled_dx = (
                        float(np.max(np.abs(candidate - x) / np.maximum(np.abs(x), 1.0)))
                        if x.size
                        else 0.0
                    )
                    x = candidate
                    z_est, candidate_z_est = candidate_z_est, z_est
                    residual, candidate_residual = candidate_residual, residual
                    objective = candidate_objective
                    residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
                    accepted = True
                    stationary = bool(
                        step_damping == 0.0
                        and relative_objective_change < self.tol
                        and scaled_correction < np.sqrt(self.tol)
                    )
                    break
                if ls_slope < 0.0 and np.isfinite(candidate_objective):
                    denom = 2.0 * (candidate_objective - objective - ls_slope * step_scale)
                    if np.isfinite(denom) and denom > 1e-30:
                        s_quad = (-ls_slope * step_scale * step_scale) / denom
                        step_scale = float(np.clip(s_quad, 0.1 * step_scale, 0.5 * step_scale))
                    else:
                        step_scale *= 0.5
                else:
                    step_scale *= 0.5

            if not accepted:
                if tried_undamped_escape:
                    # 连续平底谷停滞后尝试无阻尼脱困仍无下降，确认已到达局部极小驻点
                    converged = True
                    evaluated_at_final_x = True
                    break
                if force_lm_damping < 100.0:
                    force_lm_damping = min(100.0, max(force_lm_damping * 10.0, 1e-2))
                    curvature_damping_active = True
                    backtrack_streak = 0
                    continue
                if kkt_inf <= max(1e-3, 1000.0 * self.tol) and scaled_correction < 1e-3:
                    converged = True
                evaluated_at_final_x = True
                break

            obj_change = abs(previous_objective - objective)
            # actual_scaled_dx 已在投影后按 ||candidate - x|| 精确计算，准确识别贴墙零位移驻点
            # 自适应阻尼调节：识别香蕉谷交替回溯与满步慢发散，并在满步显著压降目标时果断放松阻尼避免蒸汽焓等线性斜坡焊死重阻尼
            if step_scale >= 0.5:
                backtrack_streak = max(0, backtrack_streak - 1)
                if (
                    scaled_correction > max(self.tol, 1e-5)
                    and scaled_correction >= 0.95 * prev_scaled_correction
                    and obj_change <= 1e-4 * (abs(previous_objective) + 1.0)
                ):
                    fullstep_osc_streak += 1
                    if fullstep_osc_streak >= 4:
                        force_lm_damping = min(100.0, max(force_lm_damping * 5.0, 0.1))
                        curvature_damping_active = True
                        fullstep_osc_streak = 0
                else:
                    fullstep_osc_streak = 0
                if force_lm_damping > 0.0:
                    if curvature_damping_active:
                        if scaled_correction < prev_scaled_correction and objective < previous_objective:
                            smooth_fullstep_streak += 1
                        else:
                            smooth_fullstep_streak = 0
                        if (
                            scaled_correction < 0.98 * prev_scaled_correction
                            or obj_change > 1e-3 * (abs(previous_objective) + 1.0)
                            or smooth_fullstep_streak >= 3
                        ):
                            next_damp = force_lm_damping * 0.5
                            smooth_fullstep_streak = 0
                            if min_cycle_damping > 0.0 and next_damp < min_cycle_damping:
                                force_lm_damping = min_cycle_damping
                            else:
                                force_lm_damping = next_damp
                                damp_relaxed_flag = True
                                if force_lm_damping < 1e-4:
                                    force_lm_damping = 0.0
                                    curvature_damping_active = False
                    else:
                        smooth_fullstep_streak = 0
                        next_damp = force_lm_damping * 0.25
                        if min_cycle_damping > 0.0 and next_damp < min_cycle_damping:
                            force_lm_damping = min_cycle_damping
                        else:
                            force_lm_damping = next_damp
                            damp_relaxed_flag = True
                            if force_lm_damping < 1e-6:
                                force_lm_damping = 0.0
            else:
                fullstep_osc_streak = 0
                smooth_fullstep_streak = 0
                backtrack_streak += 2
                if backtrack_streak >= 4 or step_scale < 1e-2:
                    if damp_relaxed_flag:
                        damp_cycle_count += 1
                        damp_relaxed_flag = False
                        if damp_cycle_count >= 2:
                            # 阻尼放松后反复撞回溯形成周期震荡，锁住温和防跌零下限(1e-3)防止跌零撞墙且不截断有效热网推进区间
                            min_cycle_damping = min(1.0, max(min_cycle_damping, 1e-3))
                    force_lm_damping = min(100.0, max(force_lm_damping * 10.0, max(1e-2, min_cycle_damping)))
                    curvature_damping_active = True
                    backtrack_streak = 0
            prev_scaled_correction = scaled_correction

            # 带阻尼稳态停滞、平底谷与高阻尼非光滑脊线脱困检测（与单域流体求解器保持一致）
            rel_obj_tol_high_damp = max(1e-12, 1e-5 * (abs(previous_objective) + 1.0))
            rel_obj_tol_slow = max(1e-12, 5e-4 * (abs(previous_objective) + 1.0))
            rel_obj_tol_flat = max(1e-12, 1e-6 * (abs(previous_objective) + 1.0))
            if obj_change <= rel_obj_tol_flat:
                low_progress_streak += 1
            else:
                low_progress_streak = 0
            if min_cycle_damping > 0.0 and obj_change <= max(1e-10, 2e-3 * (abs(previous_objective) + 1.0)):
                cycle_slow_streak += 1
            else:
                cycle_slow_streak = 0
            if tried_undamped_escape:
                if (
                    obj_change <= max(1e-10, 5e-4 * (abs(previous_objective) + 1.0))
                    or (step_scale < 0.5 and obj_change <= max(1e-10, 5e-3 * (abs(previous_objective) + 1.0)))
                ):
                    # 无阻尼脱困试探未能令目标函数实质下降（或需大幅回溯且仅挤出极微降幅），确认已到达非光滑局部极小驻点
                    stationary = True
                else:
                    tried_undamped_escape = False
                    stagnation_streak = 0
                    high_damp_stagnation_streak = 0
                    high_damp_slow_streak = 0
                    low_progress_streak = 0
                    cycle_slow_streak = 0
            elif (
                step_damping > 0.0
                and scaled_correction < 1e-3
                and kkt_inf <= max(5e-4, 500.0 * self.tol)
                and obj_change <= max(1e-12, 1e-7 * (abs(previous_objective) + 1.0))
            ):
                stationary = True
            elif (
                obj_change <= max(1e-12, 1e-8 * (abs(previous_objective) + 1.0))
                and actual_scaled_dx <= max(1e-5, 10.0 * self.tol)
            ):
                stagnation_streak += 1
                if stagnation_streak >= 3:
                    if step_damping > 0.0:
                        force_lm_damping = 0.0
                        curvature_damping_active = False
                        tried_undamped_escape = True
                        stagnation_streak = 0
                        high_damp_stagnation_streak = 0
                        high_damp_slow_streak = 0
                    else:
                        stationary = True
            elif step_damping >= 1.0 and obj_change <= rel_obj_tol_high_damp:
                stagnation_streak = 0
                high_damp_stagnation_streak += 1
                high_damp_slow_streak += 1
                if high_damp_stagnation_streak >= 6:
                    force_lm_damping = 0.0
                    curvature_damping_active = False
                    tried_undamped_escape = True
                    high_damp_stagnation_streak = 0
                    high_damp_slow_streak = 0
            elif (step_damping >= 1.0 or (curvature_damping_active and step_damping >= 0.1)) and obj_change <= rel_obj_tol_slow:
                stagnation_streak = 0
                high_damp_stagnation_streak = 0
                high_damp_slow_streak += 1
                slow_limit = 12 if step_damping >= 1.0 else 16
                if high_damp_slow_streak >= slow_limit:
                    force_lm_damping = 0.0
                    curvature_damping_active = False
                    tried_undamped_escape = True
                    high_damp_slow_streak = 0
            elif low_progress_streak >= 10 or cycle_slow_streak >= 15:
                # 连续 10 步目标函数 6 位以上无实质变化，或周期阻尼护栏下连续 15 步缓慢停滞，触发脱困确认释放外层 BDR
                low_progress_streak = 0
                cycle_slow_streak = 0
                if step_damping > 0.0:
                    force_lm_damping = 0.0
                    curvature_damping_active = False
                    tried_undamped_escape = True
                else:
                    stationary = True
            else:
                stagnation_streak = 0
                high_damp_stagnation_streak = 0
                high_damp_slow_streak = 0

            if stationary:
                converged = True
                if verbose:
                    _print_iteration(iteration, objective, residual_inf, max_correction, step_scale, True)
                break
            if verbose:
                _print_iteration(iteration, objective, residual_inf, max_correction, step_scale, False)

        if not z_est_valid:
            z_est[:] = self._evaluate_multi_energy(x)[rows]
            self._multi_energy_measurement_residual(
                z_est,
                measurement_rows=rows,
                out=residual,
            )
            objective = self._weighted_objective(weight, residual)
            residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
        if scaled_correction < self.tol:
            converged = True

        # 若未在最终点求雅可比且需要诊断，则在最终点补求 H 与 gain（无阻尼收敛时直接复用 aat 因子）
        if not evaluated_at_final_x:
            if final_diagnostics or H is None:
                H = self._multi_energy_jacobian_sparse(x)[rows]
                gain = None
        if final_diagnostics and gain is None and H is not None:
            if not (use_aat and aat_solver is not None and last_used_damping == 0.0 and evaluated_at_final_x and aat_solver.last_cholmod_factor is not None):
                normal_pattern = _normal_equation_structural_pattern(H)
                if weighted_residual is not None:
                    np.multiply(weight, residual, out=weighted_residual)
                gain, _rhs = build_normal_equations(
                    H,
                    residual,
                    weight,
                    uniform_weight=uniform_weight,
                    weights_are_uniform=weights_are_uniform,
                    weighted_residual=weighted_residual,
                    normal_pattern=normal_pattern,
                    assume_normal_pattern_matches=False,
                )
        if use_aat and aat_solver is not None and last_used_damping == 0.0 and evaluated_at_final_x and aat_solver.last_cholmod_factor is not None and H is not None:
            self._last_estimate_cholmod_factor = (id(H), aat_solver.last_cholmod_factor)
            self._last_estimate_aat_runtime = (aat_solver, aat_plan)
        else:
            self._last_estimate_cholmod_factor = None

        result = EstimateResult(
            converged=bool(converged and observability.observable),
            iterations=int(iteration),
            objective=float(objective),
            max_correction=float(max_correction),
            residual_inf=float(residual_inf),
            x=x,
            z_est=z_est.copy(),
            residual=residual.copy(),
            H=H,
            gain=gain,
            measurements=[],
            observability=observability,
            measurement_table=measurement_table_take(self._multi_energy_measurement_table, rows),
        )
        result.multi_energy_rows = rows.copy()
        result._final_lm_damping = float(force_lm_damping)
        result._state_base_scales = me_state_scales
        self.multi_energy_estimate_result = result
        return result

    @staticmethod
    def _local_observability_result(H, state_count: int, measurement_count: int) -> ObservabilityResult:
        if state_count == 0:
            return ObservabilityResult(True, 0, 0, measurement_count, 0, np.asarray([]), [])
        rank, deficiency, singular_values, weak_states = observability_rank_details(H, state_count)
        return ObservabilityResult(
            rank == state_count,
            rank,
            state_count,
            measurement_count,
            max(0, deficiency),
            singular_values,
            weak_states,
        )

    def _multi_energy_local_result(
        self,
        global_result: EstimateResult,
        measurement_slice: slice,
        state_slice: slice,
        measurement_table: MeasurementTable,
    ) -> EstimateResult:
        selected_rows = np.asarray(
            getattr(
                global_result,
                "multi_energy_rows",
                np.arange(self.multi_energy_measurement_count, dtype=np.int64),
            ),
            dtype=np.int64,
        )
        selected_positions = np.flatnonzero(
            (selected_rows >= measurement_slice.start)
            & (selected_rows < measurement_slice.stop)
        )
        local_rows = selected_rows[selected_positions] - measurement_slice.start
        local_table = measurement_table_take(measurement_table, local_rows)
        H = global_result.H[selected_positions, state_slice].tocsr()
        weights = np.asarray(local_table.weight, dtype=np.float64)
        gain, _rhs = build_normal_equations(
            H,
            global_result.residual[selected_positions],
            weights,
        )
        observability = self._local_observability_result(
            H,
            state_slice.stop - state_slice.start,
            int(selected_positions.size),
        )
        residual = global_result.residual[selected_positions].copy()
        return EstimateResult(
            converged=bool(global_result.converged and observability.observable),
            iterations=global_result.iterations,
            objective=self._weighted_objective(weights, residual),
            max_correction=global_result.max_correction,
            residual_inf=float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0,
            x=global_result.x[state_slice].copy(),
            z_est=global_result.z_est[selected_positions].copy(),
            residual=residual,
            H=H,
            gain=gain,
            measurements=[],
            observability=observability,
            measurement_table=local_table,
        )

    def _build_result_from_table(
        self,
        result: EstimateResult,
        *,
        bad_items: Sequence[BadDataItem],
        normalized_residual: np.ndarray,
        result_mode: str,
        all_measurement_table: MeasurementTable,
    ) -> Optional[SEResult]:
        mode = normalize_seresult_result_mode(result_mode)
        if mode in {"none", "array"}:
            return None
        if mode == "summary":
            return build_seresult_summary_from_table(
                result,
                bad_items=bad_items,
                all_measurement_table=all_measurement_table,
            )
        return build_seresult_full_from_table(
            result,
            bad_items=bad_items,
            normalized_residual=normalized_residual,
            all_measurement_table=all_measurement_table,
        )

    def _commit_multi_energy_results(
        self,
        result: EstimateResult,
        *,
        result_mode: str,
        skip_bad_data: bool,
        threshold: float,
        removed_bad_data: Optional[Sequence[BadDataItem]] = None,
    ) -> Optional[SEResult]:
        self.multi_energy_estimate_result = result
        self.estimate_result = result
        self.converged = bool(result.converged)
        self.iterations = int(result.iterations)
        self.observability_result = result.observability
        self.multi_energy_observability_result = result.observability
        self.removed_bad_data = list(removed_bad_data or ())
        if skip_bad_data:
            bad_items = []
            normalized = np.asarray([], dtype=np.float64)
        else:
            bad_items, normalized = self.identify_bad_data(result, threshold)
        self.bad_items = list(bad_items)
        self.normalized_residual = np.asarray(normalized, dtype=np.float64)
        self.se_result = self._build_result_from_table(
            result,
            bad_items=bad_items,
            normalized_residual=self.normalized_residual,
            result_mode=result_mode,
            all_measurement_table=self._multi_energy_measurement_table,
        )

        electric_se_result = None
        if not self._fluid_only:
            electric_table = self._measurement_table(self.active_measurements)
            electric_estimate = self._multi_energy_local_result(
                result,
                self.electric_measurement_slice,
                self.electric_state_slice,
                electric_table,
            )
            self.electric_estimate_result = electric_estimate
            electric_se_result = self._build_result_from_table(
                electric_estimate,
                bad_items=[],
                normalized_residual=np.asarray([], dtype=np.float64),
                result_mode=result_mode,
                all_measurement_table=electric_table,
            )
            self.electric_se_result = electric_se_result

        self.fluid_se_rc = {}
        self.fluid_se_errors = {}
        for name, estimator in self.fluid_estimators.items():
            local_result = self._multi_energy_local_result(
                result,
                self.fluid_measurement_slices[name],
                self.fluid_state_slices[name],
                estimator.active_measurements.table,
            )
            estimator.result = local_result
            estimator.observability = local_result.observability
            estimator._write_back_state(local_result.x)
            if skip_bad_data:
                estimator.bad_data = []
                estimator.normalized_residual = np.asarray([], dtype=np.float64)
            else:
                local_bad_data, local_normalized = self.identify_bad_data(
                    local_result,
                    threshold,
                )
                estimator.bad_data = local_bad_data
                estimator.normalized_residual = local_normalized
            estimator.se_result = self._build_result_from_table(
                local_result,
                bad_items=estimator.bad_data,
                normalized_residual=estimator.normalized_residual,
                result_mode=result_mode,
                all_measurement_table=estimator.active_measurements.table,
            )
            self.fluid_se_rc[name] = 0 if local_result.converged else 1
            if not local_result.converged:
                self.fluid_se_errors[name] = (
                    f"fluid state estimation residual={float(local_result.residual_inf):.6e}"
                )

        self._build_multi_energy_result(electric_se_result)
        return None if self._fluid_only else self.se_result

    def _run_multi_energy(
        self,
        *,
        result_mode: str,
        remove_bad_data: bool,
        bad_threshold: Optional[float],
        max_remove: Optional[int],
        max_remove_per_round: Optional[int] = None,
        skip_bad_data: bool,
        verbose: bool,
        final_diagnostics: bool,
        observability: Optional[ObservabilityResult],
    ) -> Optional[SEResult]:
        threshold = self.params.bad_threshold if bad_threshold is None else float(bad_threshold)
        removed: List[BadDataItem] = []
        if remove_bad_data:
            result, removed = self._estimate_multi_energy_with_bad_data_removal(
                threshold=threshold,
                max_remove=max_remove,
                max_remove_per_round=max_remove_per_round,
                verbose=verbose,
                observability=observability,
            )
        else:
            result = self._estimate_multi_energy(
                verbose=verbose,
                final_diagnostics=final_diagnostics or not skip_bad_data,
                observability=observability,
            )
        return self._commit_multi_energy_results(
            result,
            result_mode=result_mode,
            skip_bad_data=skip_bad_data,
            threshold=threshold,
            removed_bad_data=removed,
        )

    def _estimate_multi_energy_with_bad_data_removal(
        self,
        *,
        threshold: float,
        max_remove: Optional[int],
        max_remove_per_round: Optional[int] = None,
        verbose: bool,
        observability: Optional[ObservabilityResult] = None,
    ) -> Tuple[EstimateResult, List[BadDataItem]]:
        remove_limit, per_round_limit = resolve_bad_data_limits(
            int(self.multi_energy_control_measurement_slice.start),
            max_remove=max_remove,
            max_remove_per_round=max_remove_per_round,
            param_max_remove=self.params.max_remove,
            param_max_remove_per_round=getattr(self.params, "max_remove_per_round", None),
        )
        rows = np.arange(self.multi_energy_measurement_count, dtype=np.int64)
        ctrl_start = int(self.multi_energy_control_measurement_slice.start)
        current_observability = observability
        result = self._estimate_multi_energy(
            x0=None,
            measurement_rows=rows,
            verbose=verbose,
            final_diagnostics=True,
            observability=current_observability,
        )
        removed: List[BadDataItem] = []
        if remove_limit <= 0 or (not result.converged and not (result.observability.observable and np.isfinite(result.objective))):
            return result, removed

        blocked_global_rows: set = set()
        need_identify = True
        bad_items: List[BadDataItem] = []
        normalized = np.empty(0, dtype=np.float64)
        col_degrees: Optional[np.ndarray] = None
        local_to_item: Dict[int, BadDataItem] = {}
        unconverged_streak = 0 if result.converged else 1
        unconverged_total_rounds = 0 if result.converged else 1
        failed_trial_solves = 0
        me_rescue_rounds = 0
        v_cols_me = np.asarray(getattr(self, "_multi_energy_voltage_cols", np.asarray([], dtype=np.int64)), dtype=np.int64)
        orig_me_weights = np.asarray(self._multi_energy_weight, dtype=np.float64).copy()
        orig_me_z = np.asarray(self._multi_energy_z, dtype=np.float64).copy()
        me_mt_arr = np.asarray(self._multi_energy_measurement_table.meas_type_code, dtype=np.int16)
        me_is_pseudo = measurement_table_bad_data_protected_mask(self._multi_energy_measurement_table)

        def _warm_start_is_healthy(state_vec: Optional[np.ndarray]) -> bool:
            if state_vec is None or not state_vec.size or not np.all(np.isfinite(state_vec)):
                return False
            if v_cols_me.size:
                vp = state_vec[v_cols_me]
                if float(vp.min()) < 0.5 or float(vp.max()) > 1.5:
                    return False
            return True

        def _try_me_boundary_node_rescue() -> bool:
            """多能耦合贴墙节点自救：对拉死电压至上/下界的极端离群真实 V 测点安全降权至 1e-4 并平启动重估。"""
            nonlocal result, need_identify, unconverged_streak, unconverged_total_rounds, failed_trial_solves
            if (
                result.converged
                or not v_cols_me.size
                or result.x is None
                or result.H is None
                or result.residual is None
                or not is_sparse_matrix(result.H)
            ):
                return False
            vx = result.x[v_cols_me]
            v_lo_p = float(self.voltage_min)
            v_hi_p = float(self.voltage_max)
            ext_v_cols = v_cols_me[(vx <= v_lo_p + 0.02) | (vx >= v_hi_p - 0.02)]
            if not ext_v_cols.size:
                return False
            H_csc = result.H.tocsc()
            downweighted = False
            for c_val in ext_v_cols.tolist():
                c_i = int(c_val)
                if c_i < 0 or c_i >= H_csc.shape[1]:
                    continue
                for r_l in H_csc.indices[H_csc.indptr[c_i] : H_csc.indptr[c_i + 1]]:
                    l_i = int(r_l)
                    if l_i < 0 or l_i >= rows.size:
                        continue
                    g_i = int(rows[l_i])
                    if g_i >= ctrl_start or me_is_pseudo[g_i]:
                        continue
                    if int(me_mt_arr[g_i]) not in VOLTAGE_PSEUDO_MEAS_TYPE_CODES:
                        continue
                    w_cur = float(self._multi_energy_weight[g_i])
                    if w_cur <= 1e-3:
                        continue
                    z_cur = (
                        float(result.z_est[l_i]) + float(result.residual[l_i])
                        if (result.z_est is not None and l_i < result.z_est.size)
                        else 1.0
                    )
                    phys_oob = (z_cur < v_lo_p + 0.02) or (z_cur > v_hi_p - 0.02)
                    if phys_oob or (
                        abs(float(result.residual[l_i])) * np.sqrt(w_cur) > max(2.0 * float(threshold), 8.0)
                    ):
                        # 退化为极弱标称电压伪量测 (V=1.0 pu, w=1e-4)，既保持列覆盖满秩又解除贴墙拉扯
                        self._multi_energy_weight[g_i] = 1e-4
                        self._multi_energy_z[g_i] = 1.0
                        downweighted = True
            if not downweighted:
                return False
            rescue_res = self._estimate_multi_energy(
                x0=None,
                measurement_rows=rows,
                verbose=verbose,
                final_diagnostics=True,
                observability=current_observability,
                _hint_lm_damping=0.0,
            )
            if rescue_res.converged or (
                np.isfinite(rescue_res.objective)
                and (not np.isfinite(result.objective) or rescue_res.objective < result.objective)
            ):
                result = rescue_res
                need_identify = True
                unconverged_streak = 0
                unconverged_total_rounds = 0
                failed_trial_solves = 0
                return True
            return False

        try:
            while len(removed) < remove_limit:
                if need_identify:
                    bad_items, normalized = self.identify_bad_data(result, threshold)
                    cur_w = np.asarray(self._multi_energy_weight[rows], dtype=np.float64)
                    cur_act = np.ones(rows.size, dtype=bool)
                    col_degrees = (
                        compute_active_column_degrees(result.H, cur_act, cur_w)
                        if result.H is not None
                        else None
                    )
                    local_to_item = {int(it.row_pos): it for it in bad_items}
                    need_identify = False
                if not bad_items:
                    if (not result.converged) and me_rescue_rounds < 4 and _try_me_boundary_node_rescue():
                        me_rescue_rounds += 1
                        continue
                    break

                cand_locals: List[int] = []
                for item in bad_items:
                    lpos = int(item.row_pos)
                    if 0 <= lpos < rows.size:
                        grow = int(rows[lpos])
                        if grow < ctrl_start and grow not in blocked_global_rows:
                            cand_locals.append(lpos)
                if not cand_locals:
                    if (not result.converged) and me_rescue_rounds < 4 and _try_me_boundary_node_rescue():
                        me_rescue_rounds += 1
                        continue
                    break

                remaining = remove_limit - len(removed)
                batch_limit = max(1, min(per_round_limit, remaining))
                if result.H is not None and len(cand_locals) >= 2 and batch_limit >= 2:
                    cand_norms = [float(normalized[p]) for p in cand_locals]
                    sel_idx = select_independent_bad_data_batch(
                        result.H,
                        cand_locals,
                        cand_norms,
                        threshold,
                        min_batch_norm=max(float(threshold), 4.5),
                        max_batch=batch_limit,
                        min_hops=3,
                    )
                    selected_locals = [cand_locals[k] for k in sel_idx] if sel_idx else [cand_locals[0]]
                else:
                    selected_locals = [cand_locals[0]]

                # 列覆盖可观测性保护：禁止把任意状态列的全部支撑量测删空
                if result.H is not None and col_degrees is not None:
                    safe_pos, _ = filter_batch_preserving_column_coverage(result.H, selected_locals, col_degrees)
                    safe_locals = [selected_locals[k] for k in safe_pos]
                    if not safe_locals:
                        probe = cand_locals[: min(len(cand_locals), 20)]
                        p_pos, _ = filter_batch_preserving_column_coverage(result.H, probe, col_degrees)
                        if p_pos:
                            safe_locals = [probe[p_pos[0]]]
                else:
                    safe_locals = selected_locals
                if not safe_locals:
                    if (not result.converged) and me_rescue_rounds < 4 and _try_me_boundary_node_rescue():
                        me_rescue_rounds += 1
                        continue
                    break

                # 结构满秩前置拦截：若删批后导致降秩则折半收缩或跳过临界行
                trial_locals = list(safe_locals)
                if result.H is not None:
                    cur_w = np.asarray(self._multi_energy_weight[rows], dtype=np.float64)
                    while trial_locals:
                        t_mask = np.ones(rows.size, dtype=bool)
                        t_w = cur_w.copy()
                        t_mask[trial_locals] = False
                        t_w[trial_locals] = 0.0
                        if check_active_structural_full_rank(result.H, t_mask, t_w):
                            break
                        if len(trial_locals) == 1:
                            blocked_global_rows.add(int(rows[trial_locals[0]]))
                            trial_locals = []
                            break
                        trial_locals = trial_locals[: max(1, len(trial_locals) // 2)]
                if not trial_locals:
                    if len(blocked_global_rows) >= 32:
                        if (not result.converged) and me_rescue_rounds < 4 and _try_me_boundary_node_rescue():
                            me_rescue_rounds += 1
                            blocked_global_rows.clear()
                            continue
                        break
                    continue

                # 带回滚与折半退避的重估：防止误删隐式临界测点导致发散或目标严重恶化
                accepted_trial = False
                while trial_locals:
                    drop_mask = np.ones(rows.size, dtype=bool)
                    drop_mask[trial_locals] = False
                    trial_rows = rows[drop_mask]
                    trial_obs = self._multi_energy_observability_analysis(
                        result.x,
                        measurement_rows=trial_rows,
                    )
                    if not trial_obs.observable:
                        if len(trial_locals) == 1:
                            blocked_global_rows.add(int(rows[trial_locals[0]]))
                            break
                        trial_locals = trial_locals[: max(1, len(trial_locals) // 2)]
                        continue
                    use_warm = _warm_start_is_healthy(result.x)
                    hint_damp = float(getattr(result, "_final_lm_damping", 0.0)) if use_warm else 0.0
                    trial_res = self._estimate_multi_energy(
                        x0=result.x if use_warm else None,
                        measurement_rows=trial_rows,
                        verbose=verbose,
                        final_diagnostics=True,
                        observability=trial_obs,
                        _hint_lm_damping=hint_damp,
                    )
                    # 热启动未收敛或极早异常退出时，自动以健康平启动重试一次以脱离坏初值陷阱
                    if use_warm and (not trial_res.converged):
                        flat_res = self._estimate_multi_energy(
                            x0=None,
                            measurement_rows=trial_rows,
                            verbose=verbose,
                            final_diagnostics=True,
                            observability=trial_obs,
                            _hint_lm_damping=0.0,
                        )
                        if flat_res.converged or (
                            np.isfinite(flat_res.objective)
                            and (not np.isfinite(trial_res.objective) or flat_res.objective < trial_res.objective)
                        ):
                            trial_res = flat_res
                    trial_ok = (
                        np.isfinite(trial_res.objective)
                        and (
                            trial_res.converged
                            or (not result.converged and trial_res.objective < result.objective)
                        )
                        and (
                            not result.converged
                            or trial_res.objective <= result.objective * 1.05 + 1e-6
                        )
                    )
                    if trial_ok:
                        prev_obj_accepted = float(result.objective)
                        for lp in trial_locals:
                            removed.append(local_to_item[lp])
                        rows = trial_rows
                        result = trial_res
                        current_observability = trial_obs
                        if result.converged:
                            unconverged_streak = 0
                            unconverged_total_rounds = 0
                        else:
                            unconverged_total_rounds += 1
                            if np.isfinite(prev_obj_accepted) and result.objective <= 0.85 * prev_obj_accepted:
                                # 删坏后目标函数显著下降，说明正在精准剔除真坏数据，重置连续未收敛停滞计数
                                unconverged_streak = 0
                            else:
                                unconverged_streak += 1
                        failed_trial_solves = 0
                        need_identify = True
                        accepted_trial = True
                        break
                    if not trial_res.converged:
                        failed_trial_solves += 1
                        if failed_trial_solves >= 4:
                            # 连续多次删坏重估均不收敛，立即熔断回滚至最优已收敛解，防止连跑数十次满步迭代
                            break
                    if len(trial_locals) == 1:
                        blocked_global_rows.add(int(rows[trial_locals[0]]))
                        break
                    trial_locals = trial_locals[: max(1, len(trial_locals) // 2)]

                if not accepted_trial:
                    if (not result.converged) and me_rescue_rounds < 4 and _try_me_boundary_node_rescue():
                        me_rescue_rounds += 1
                        continue
                    if not result.converged or len(blocked_global_rows) >= 16 or failed_trial_solves >= 4:
                        break
                    continue
                if unconverged_streak >= 3 or unconverged_total_rounds >= 10:
                    if (not result.converged) and me_rescue_rounds < 4 and _try_me_boundary_node_rescue():
                        me_rescue_rounds += 1
                        continue
                    break
            if (not result.converged) and me_rescue_rounds < 4 and _try_me_boundary_node_rescue():
                me_rescue_rounds += 1
        finally:
            self._multi_energy_weight[:] = orig_me_weights
            self._multi_energy_z[:] = orig_me_z

        return result, removed

    def _prepare_uncoupled_direct_delegate(self, side: str, profile_start: float) -> "HybridStateEstimator":
        stage_start = time.perf_counter()
        self.meas_ppc, self.measurements = self._prepare_measurement_source()
        if isinstance(self.meas_ppc, dict):
            self.meas_ppc["_mutable_runtime_arrays"] = True
        self._record_profile_time("init.load_measurements", time.perf_counter() - stage_start)
        self._sub_measurement_sources_by_side = None
        self._sub_measurement_source_rows_by_side = None
        self._sub_measurement_summary_attrs_by_side = {}
        self._sub_measurement_global_summary_attrs = None
        self._sub_measurements_converted_by_side = {"ac": False, "dc": False}
        self._measurements_normalized = False
        self._sub_estimators_enabled = False
        self._delegate_estimator = None
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        self.dcac_converters = []

        stage_start = time.perf_counter()
        self._direct_delegate_fast_path = True
        self._direct_delegate_side = side
        if side == "ac":
            self._dc_sub_estimator = None
            self._ac_sub_estimator = ACStateEstimator(
                e_file=self.e_file,
                meas_file=self.meas_file,
                tol=self.tol,
                max_iter=self.max_iter,
                diff_step=self.diff_step,
                flat_start=self.flat_start,
                parameters=self.params,
                network=self.network.ac,
                matrix_dump_dir=None if self.matrix_dump_dir is None else self.matrix_dump_dir / "ac",
                measurements=self.meas_ppc if isinstance(self.meas_ppc, dict) else self.measurements,
                prepare_active_measurements=self._prepare_active_measurements,
                defer_prepare_finalize=self._prepare_defer_finalize,
                profile=self.profile_enabled,
                auto_prepare=False,
                power_flow_linear_solver=self._sub_power_flow_linear_solver(),
            )
            self._ac_sub_estimator.prepare()
            delegate = self._ac_sub_estimator
        elif side == "dc":
            self._ac_sub_estimator = None
            self._dc_sub_estimator = DCStateEstimator(
                e_file=self.e_file,
                meas_file=self.meas_file,
                tol=self.tol,
                max_iter=self.max_iter,
                diff_step=self.diff_step,
                flat_start=self.flat_start,
                parameters=self.params,
                network=self.network.dc,
                matrix_dump_dir=None if self.matrix_dump_dir is None else self.matrix_dump_dir / "dc",
                measurements=self.meas_ppc if isinstance(self.meas_ppc, dict) else self.measurements,
                prepare_active_measurements=self._prepare_active_measurements,
                defer_prepare_finalize=self._prepare_defer_finalize,
                profile=self.profile_enabled,
                auto_prepare=False,
                power_flow_linear_solver=self._sub_power_flow_linear_solver(),
            )
            self._dc_sub_estimator.prepare()
            delegate = self._dc_sub_estimator
        else:
            raise ValueError(f"Unsupported direct delegate side: {side}")
        self._record_profile_time("init.prepare_sub_estimators", time.perf_counter() - stage_start)
        if self._prepare_defer_finalize:
            self._defer_prepare_finalize_pending = True
            self._record_profile_time('init.total', time.perf_counter() - profile_start)
            return self
        stage_start = time.perf_counter()
        self._build_device_maps(install_views=False)
        self._record_profile_time("init.device_maps", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        self._adopt_delegate(delegate, side)
        self.meas_ppc = getattr(delegate, "meas_ppc", self.meas_ppc)
        self._record_profile_time("init.adopt_delegate", time.perf_counter() - stage_start)
        self._prepared = True
        self._record_profile_time("init.total", time.perf_counter() - profile_start)
        return self

    def _require_prepared(self, action: str) -> None:
        if not self._prepared:
            raise RuntimeError(f"Call prepare() before {action}.")

    def _record_profile_time(self, name: str, elapsed: float) -> None:
        if self.profile_enabled:
            self.profile_times[name] = self.profile_times.get(name, 0.0) + float(elapsed)

    @staticmethod
    def _load_network(e_file: Path) -> HybridPowerNetwork:
        efile_rows = _read_efile_rows(e_file)
        ppc = build_hybrid_ppc_with_topology_from_efile_rows(e_file, efile_rows)
        return _build_hybrid_se_network_from_ppc(ppc)

    def _initial_measurement_sources_by_side(self) -> Dict[str, List[Measurement]]:
        sources_by_side = getattr(self, "_sub_measurement_sources_by_side", None)
        rows_by_side = getattr(self, "_sub_measurement_source_rows_by_side", None)
        if sources_by_side is None or rows_by_side is None:
            partitions = partition_measurements_by_code(
                self.measurements,
                self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
                table_builder=_measurement_table_from_measurements,
                as_view=False,
                sides=("ac", "dc", "hybrid"),
            )
            sources_by_side = partitions.measurements
            self._sub_measurement_sources_by_side = sources_by_side
            self._sub_measurement_source_rows_by_side = partitions.rows
            self._sub_measurement_source_count = len(self.measurements)
        return sources_by_side

    def _measurements_for_sub_estimator(self, side: str, share_measurements: bool):
        if share_measurements and self._is_uncoupled_single_side(side):
            meas_ppc = getattr(self, "meas_ppc", None)
            if isinstance(meas_ppc, dict):
                return meas_ppc
            return self.measurements
        sources = self._initial_measurement_sources_by_side().get(side, ())
        defer_finalize = getattr(self, "network", None) is not None and self._defer_sub_prepare_finalize()
        if share_measurements or defer_finalize:
            return sources
        return copy_measurement_view(sources)

    def _defer_sub_prepare_finalize(self) -> bool:
        if getattr(self, '_prepare_defer_finalize', False):
            return True
        if self._is_uncoupled_single_side("ac") or self._is_uncoupled_single_side("dc"):
            return False
        has_ac = _side_has_alive_nodes(getattr(self.network, "ac", None))
        has_dc = _side_has_alive_nodes(getattr(self.network, "dc", None))
        return bool(has_ac and has_dc)

    def _is_uncoupled_single_side(self, side: str) -> bool:
        if getattr(self.network, "dcac_converters", None):
            return False
        has_ac = _side_has_alive_nodes(getattr(self.network, "ac", None))
        has_dc = _side_has_alive_nodes(getattr(self.network, "dc", None))
        if side == "ac":
            return has_ac and not has_dc
        if side == "dc":
            return has_dc and not has_ac
        return False

    def _defer_sub_active_measurement_preparation(self) -> bool:
        if not getattr(self, '_prepare_active_measurements', True):
            return True
        has_ac = _side_has_alive_nodes(getattr(self.network, "ac", None))
        has_dc = _side_has_alive_nodes(getattr(self.network, "dc", None))
        has_coupling = bool(getattr(self.network, "dcac_converters", None))
        return bool(self.flat_start and (has_coupling or (has_ac and has_dc)))

    def _sub_power_flow_linear_solver(self) -> Optional[str]:
        if self.power_flow_linear_solver:
            return self.power_flow_linear_solver
        has_dc = _side_has_alive_nodes(getattr(self.network, "dc", None))
        has_coupling = bool(getattr(self.network, "dcac_converters", None))
        return "umfpack" if has_dc or has_coupling else None

    def _build_ac_sub_estimator(self) -> Optional[ACStateEstimator]:
        if not _side_has_alive_nodes(getattr(self.network, "ac", None)):
            return None
        reuse_loaded = True
        defer_active = reuse_loaded and self._defer_sub_active_measurement_preparation()
        defer_finalize = reuse_loaded and self._defer_sub_prepare_finalize()
        share_measurements = defer_active or (reuse_loaded and self._is_uncoupled_single_side("ac"))
        try:
            estimator = ACStateEstimator(
                e_file=self.e_file,
                meas_file=self.meas_file,
                tol=self.tol,
                max_iter=self.max_iter,
                diff_step=self.diff_step,
                flat_start=self.flat_start,
                parameters=self.params,
                network=self.network.ac if reuse_loaded else None,
                matrix_dump_dir=None if self.matrix_dump_dir is None else self.matrix_dump_dir / "ac",
                measurements=self._measurements_for_sub_estimator("ac", share_measurements) if reuse_loaded else None,
                prepare_active_measurements=not defer_active,
                defer_prepare_finalize=defer_finalize,
                profile=self.profile_enabled,
                auto_prepare=False,
                power_flow_linear_solver=self._sub_power_flow_linear_solver(),
            )
            estimator.prepare()
            return estimator
        except RuntimeError as exc:
            if "No alive AC nodes" in str(exc):
                return None
            raise

    def _build_dc_sub_estimator(self) -> Optional[DCStateEstimator]:
        if not _side_has_alive_nodes(getattr(self.network, "dc", None)):
            return None
        reuse_loaded = True
        defer_active = reuse_loaded and self._defer_sub_active_measurement_preparation()
        defer_finalize = reuse_loaded and self._defer_sub_prepare_finalize()
        share_measurements = defer_active or (reuse_loaded and self._is_uncoupled_single_side("dc"))
        try:
            estimator = DCStateEstimator(
                e_file=self.e_file,
                meas_file=self.meas_file,
                tol=self.tol,
                max_iter=self.max_iter,
                diff_step=self.diff_step,
                flat_start=self.flat_start,
                parameters=self.params,
                network=self.network.dc if reuse_loaded else None,
                matrix_dump_dir=None if self.matrix_dump_dir is None else self.matrix_dump_dir / "dc",
                measurements=self._measurements_for_sub_estimator("dc", share_measurements) if reuse_loaded else None,
                prepare_active_measurements=not defer_active,
                defer_prepare_finalize=defer_finalize,
                profile=self.profile_enabled,
                auto_prepare=False,
                power_flow_linear_solver=self._sub_power_flow_linear_solver(),
            )
            estimator.prepare()
            return estimator
        except RuntimeError as exc:
            if "No alive DC nodes" in str(exc):
                return None
            raise

    def _build_device_maps(self, *, install_views: bool = True) -> None:
        if hasattr(self, "_hybrid_converter_device_key_array_cache"):
            delattr(self, "_hybrid_converter_device_key_array_cache")
        self.dcac_converters = sorted(
            (
                conv
                for conv in getattr(self.network, "dcac_converters", [])
                if getattr(conv, "is_alive", False)
            ),
            key=lambda item: item.idx,
        )
        if install_views:
            self._link_converter_endpoint_nodes()

    def _link_converter_endpoint_nodes(self) -> None:
        """仅为交直流换流器端点挂载轻量节点视图，避免急切构建全网兼容设备列表。"""
        if not self.dcac_converters:
            return
        if getattr(self, "_compatibility_device_views_ready", False):
            ac_map = self.ac_node_by_idx
            dc_map = self.dc_node_by_idx
            for conv in self.dcac_converters:
                conv.ac_node_obj = ac_map.get(int(conv.ac_node))
                conv.dc_node_obj = dc_map.get(int(conv.dc_node))
            return
        net = getattr(self, "network", None)
        ac_ppc = getattr(net, "_ac_ppc", None)
        dc_ppc = getattr(net, "_dc_ppc", None)

        def _build_selected_nodes(ppc, cols, wanted_ids, ac_side: bool) -> Dict[int, SimpleNamespace]:
            if not isinstance(ppc, dict) or not wanted_ids:
                return {}
            table = np.asarray(ppc.get("bus", np.zeros((0, len(cols)))), dtype=np.float64)
            if table.ndim != 2 or table.shape[0] == 0:
                return {}
            bus_ids = table[:, cols["idx"]].astype(np.int64, copy=False)
            wanted_arr = np.unique(np.asarray(wanted_ids, dtype=np.int64))
            matches = np.flatnonzero(np.isin(bus_ids, wanted_arr))
            if matches.size == 0:
                return {}
            sel_ids = bus_ids[matches].tolist()
            sel_vbase = table[matches, cols["vbase"]].tolist()
            sel_v = table[matches, cols["voltage"]].tolist()
            sel_stat = table[matches, cols["run_stat"]].astype(np.int64, copy=False).tolist()
            raw_names = ppc.get("bus_name", ())
            if len(raw_names) == int(table.shape[0]):
                sel_names = [
                    str(raw_names[pos]) if raw_names[pos] is not None else f"bus_{idx_val}"
                    for pos, idx_val in zip(matches.tolist(), sel_ids)
                ]
            else:
                sel_names = [f"bus_{idx_val}" for idx_val in sel_ids]
            if ac_side:
                sel_ang = table[matches, cols["angle"]].tolist()
                return {
                    idx_val: SimpleNamespace(
                        idx=idx_val,
                        name=name_val,
                        vbase=vb_val,
                        voltage=v_val,
                        run_stat=st_val,
                        is_alive=(st_val == 1),
                        angle=ang_val,
                    )
                    for idx_val, name_val, vb_val, v_val, st_val, ang_val in zip(
                        sel_ids, sel_names, sel_vbase, sel_v, sel_stat, sel_ang
                    )
                }
            return {
                idx_val: SimpleNamespace(
                    idx=idx_val,
                    name=name_val,
                    vbase=vb_val,
                    voltage=v_val,
                    run_stat=st_val,
                    is_alive=(st_val == 1),
                )
                for idx_val, name_val, vb_val, v_val, st_val in zip(
                    sel_ids, sel_names, sel_vbase, sel_v, sel_stat
                )
            }

        ac_wanted = [int(conv.ac_node) for conv in self.dcac_converters]
        dc_wanted = [int(conv.dc_node) for conv in self.dcac_converters]
        ac_map = _build_selected_nodes(ac_ppc, AC_BUS_COLS, ac_wanted, ac_side=True)
        dc_map = _build_selected_nodes(dc_ppc, DC_BUS_COLS, dc_wanted, ac_side=False)
        for conv in self.dcac_converters:
            conv.ac_node_obj = ac_map.get(int(conv.ac_node))
            conv.dc_node_obj = dc_map.get(int(conv.dc_node))

    @staticmethod
    def _compat_names(ppc: Dict, name_key: str, table: np.ndarray, prefix: str) -> np.ndarray:
        names = np.asarray(ppc.get(name_key, ()), dtype=object)
        count = int(table.shape[0]) if table.ndim == 2 else 0
        if names.size == count:
            return names.astype(str, copy=False)
        idx = table[:, 0].astype(np.int64, copy=False) if count else np.asarray([], dtype=np.int64)
        return np.asarray([f"{prefix}_{int(value)}" for value in idx], dtype=object)

    @classmethod
    def _compat_devices(
        cls,
        ppc: Dict,
        table_key: str,
        name_key: str,
        cols: Dict[str, int],
        prefix: str,
        attr_cols: Dict[str, str],
    ) -> List[SimpleNamespace]:
        table = np.asarray(ppc.get(table_key, np.zeros((0, len(cols)))), dtype=np.float64)
        if table.ndim != 2 or table.shape[0] == 0:
            return []
        names = cls._compat_names(ppc, name_key, table, prefix)
        # Convert each column once, instead of indexing/casting every field of
        # every device in Python. Lists keep the legacy Python scalar types.
        integer_attrs = {"idx", "node", "i_node", "j_node", "run_stat", "status"}
        attrs = tuple(attr_cols)
        columns = [
            (table[:, cols[attr_cols[attr]]] if attr_cols[attr] in cols else np.zeros(table.shape[0]))
            .astype(np.int64 if attr in integer_attrs else np.float64).tolist()
            for attr in attrs
        ]
        devices = []
        for name, row_values in zip(names, zip(*columns)):
            values = dict(zip(attrs, row_values))
            values["is_alive"] = bool(
                values.get("run_stat", 1) == 1
                and values.get("status", 1) == 1
            )
            values["name"] = str(name if name is not None else values["idx"])
            devices.append(SimpleNamespace(**values))
        return devices

    @classmethod
    def _compat_nodes(
        cls,
        ppc: Dict,
        cols: Dict[str, int],
        *,
        ac_side: bool,
    ) -> List[SimpleNamespace]:
        table = np.asarray(ppc.get("bus", np.zeros((0, len(cols)))), dtype=np.float64)
        if table.ndim != 2 or table.shape[0] == 0:
            return []
        names = cls._compat_names(ppc, "bus_name", table, "bus")
        nodes = []
        ids = table[:, cols['idx']].astype(np.int64).tolist()
        vbases = table[:, cols['vbase']].tolist()
        voltages = table[:, cols['voltage']].tolist()
        statuses = table[:, cols['run_stat']].astype(np.int64).tolist()
        angles = table[:, cols['angle']].tolist() if ac_side else [0.0] * len(ids)
        for idx, name, vbase, voltage, status, angle in zip(ids, names, vbases, voltages, statuses, angles):
            values = {
                "idx": idx,
                "name": str(name if name is not None else idx),
                "vbase": vbase,
                "voltage": voltage,
                "run_stat": status,
                "is_alive": status == 1,
            }
            if ac_side:
                values["angle"] = angle
            nodes.append(SimpleNamespace(**values))
        return nodes

    @staticmethod
    def _link_compat_nodes(devices: Sequence[SimpleNamespace], nodes_by_idx: Dict[int, SimpleNamespace]) -> None:
        for device in devices:
            if hasattr(device, "node"):
                device.node_obj = nodes_by_idx.get(int(device.node))
            if hasattr(device, "i_node"):
                device.i_node_obj = nodes_by_idx.get(int(device.i_node))
            if hasattr(device, "j_node"):
                device.j_node_obj = nodes_by_idx.get(int(device.j_node))

    @staticmethod
    def _compat_reference_nodes(estimator, node_ids: np.ndarray, nodes_by_idx: Dict[int, SimpleNamespace]):
        if estimator is None:
            return []
        references = np.asarray(getattr(estimator, "references", ()), dtype=np.int64)
        valid = references[(references >= 0) & (references < node_ids.size)]
        return [
            nodes_by_idx[int(node_ids[int(pos)])]
            for pos in valid
            if int(node_ids[int(pos)]) in nodes_by_idx
        ]

    def _install_compatibility_device_views(self) -> None:
        """Expose lightweight legacy lookup views without rebuilding network objects."""
        if getattr(self, "_compatibility_device_views_ready", False):
            return
        ac_ppc = getattr(getattr(self, "network", None), "_ac_ppc", None)
        dc_ppc = getattr(getattr(self, "network", None), "_dc_ppc", None)

        self.ac_nodes = self._compat_nodes(ac_ppc, AC_BUS_COLS, ac_side=True) if isinstance(ac_ppc, dict) else []
        self.dc_nodes = self._compat_nodes(dc_ppc, DC_BUS_COLS, ac_side=False) if isinstance(dc_ppc, dict) else []
        self.ac_node_by_name = {node.name: node for node in self.ac_nodes}
        self.ac_node_by_idx = {node.idx: node for node in self.ac_nodes}
        self.dc_node_by_name = {node.name: node for node in self.dc_nodes}
        self.dc_node_by_idx = {node.idx: node for node in self.dc_nodes}

        if isinstance(ac_ppc, dict):
            self.ac_branches = self._compat_devices(
                ac_ppc, "branch", "branch_name", AC_BRANCH_COLS, "branch",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "run_stat": "run_stat"},
            )
            self.ac_transformers = self._compat_devices(
                ac_ppc, "transformer", "transformer_name", AC_TRANSFORMER_COLS, "transformer",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "run_stat": "run_stat"},
            )
            self.ac_zero_branches = self._compat_devices(
                ac_ppc, "zero_branch", "zero_branch_name", AC_ZERO_BRANCH_COLS, "zero_branch",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "run_stat": "run_stat", "p": "p", "q": "q"},
            )
            self.ac_breakers = self._compat_devices(
                ac_ppc, "break", "break_name", AC_SWITCH_COLS, "break",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "status": "status", "run_stat": "run_stat", "p": "p", "q": "q"},
            )
            self.ac_generators = self._compat_devices(
                ac_ppc, "gen", "gen_name", AC_GEN_COLS, "generator",
                {"idx": "idx", "node": "node", "run_stat": "run_stat", "p": "p", "q": "q"},
            )
            self.ac_loads = self._compat_devices(
                ac_ppc, "load", "load_name", AC_LOAD_COLS, "load",
                {"idx": "idx", "node": "node", "run_stat": "run_stat", "p": "p", "q": "q"},
            )
        else:
            self.ac_branches = self.ac_transformers = self.ac_zero_branches = self.ac_breakers = []
            self.ac_generators = self.ac_loads = []

        if isinstance(dc_ppc, dict):
            self.dc_branches = self._compat_devices(
                dc_ppc, "branch", "branch_name", DC_BRANCH_COLS, "branch",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "run_stat": "run_stat"},
            )
            self.dc_zero_branches = self._compat_devices(
                dc_ppc, "zero_branch", "zero_branch_name", DC_ZERO_BRANCH_COLS, "zero_branch",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "run_stat": "run_stat", "p": "p"},
            )
            self.dc_breakers = self._compat_devices(
                dc_ppc, "break", "break_name", DC_SWITCH_COLS, "break",
                {"idx": "idx", "i_node": "i_node", "j_node": "j_node", "status": "status", "run_stat": "run_stat", "p": "p"},
            )
            self.dc_generators = self._compat_devices(
                dc_ppc, "gen", "gen_name", DC_GEN_COLS, "generator",
                {"idx": "idx", "node": "node", "run_stat": "run_stat", "p": "p"},
            )
            self.dc_loads = self._compat_devices(
                dc_ppc, "load", "load_name", DC_LOAD_COLS, "load",
                {"idx": "idx", "node": "node", "run_stat": "run_stat", "p": "p"},
            )
        else:
            self.dc_branches = self.dc_zero_branches = self.dc_breakers = []
            self.dc_generators = self.dc_loads = []

        for devices, nodes in (
            (self.ac_branches, self.ac_node_by_idx),
            (self.ac_transformers, self.ac_node_by_idx),
            (self.ac_zero_branches, self.ac_node_by_idx),
            (self.ac_breakers, self.ac_node_by_idx),
            (self.ac_generators, self.ac_node_by_idx),
            (self.ac_loads, self.ac_node_by_idx),
            (self.dc_branches, self.dc_node_by_idx),
            (self.dc_zero_branches, self.dc_node_by_idx),
            (self.dc_breakers, self.dc_node_by_idx),
            (self.dc_generators, self.dc_node_by_idx),
            (self.dc_loads, self.dc_node_by_idx),
        ):
            self._link_compat_nodes(devices, nodes)

        ac = getattr(self, "_ac_sub_estimator", None)
        dc = getattr(self, "_dc_sub_estimator", None)
        ac_ids = np.asarray(getattr(ac, "_ac_node_ids", ()), dtype=np.int64)
        dc_ids = np.asarray(getattr(dc, "_node_idx_by_pos", ()), dtype=np.int64)
        self.ac_reference_nodes = self._compat_reference_nodes(ac, ac_ids, self.ac_node_by_idx)
        self.dc_reference_nodes = self._compat_reference_nodes(dc, dc_ids, self.dc_node_by_idx)

        self.ac_branch_by_name = {item.name: item for item in self.ac_branches}
        self.ac_zero_branch_by_name = {item.name: item for item in self.ac_zero_branches}
        self.ac_generator_by_name = {item.name: item for item in self.ac_generators}
        self.ac_load_by_name = {item.name: item for item in self.ac_loads}
        self.dc_zero_branch_by_name = {item.name: item for item in self.dc_zero_branches}
        self.dc_generator_by_name = {item.name: item for item in self.dc_generators}
        self.dc_load_by_name = {item.name: item for item in self.dc_loads}
        all_dcac = getattr(getattr(self, "network", None), "dcac_converters", self.dcac_converters)
        self.dcac_by_name = {item.name: item for item in all_dcac}

        if getattr(self, "network", None) is not None:
            if getattr(self.network, "ac", None) is not None:
                self.network.ac.alive_buses = [node for node in self.ac_nodes if node.is_alive]
            if getattr(self.network, "dc", None) is not None:
                self.network.dc.alive_buses = [node for node in self.dc_nodes if node.is_alive]

        for conv in self.dcac_converters:
            conv.ac_node_obj = self.ac_node_by_idx.get(int(conv.ac_node))
            conv.dc_node_obj = self.dc_node_by_idx.get(int(conv.dc_node))
        self._compatibility_device_views_ready = True

    def _try_delegate_uncoupled_single_side(self) -> bool:
        if self.dcac_converters:
            return False
        if self._ac_sub_estimator is not None and self._dc_sub_estimator is None:
            self._adopt_delegate(self._ac_sub_estimator, "ac")
            return True
        if self._dc_sub_estimator is not None and self._ac_sub_estimator is None:
            self._adopt_delegate(self._dc_sub_estimator, "dc")
            return True
        return False

    def _adopt_delegate(self, delegate, side: str) -> None:
        direct_delegate = bool(getattr(self, "_direct_delegate_fast_path", False))
        self._delegate_estimator = delegate
        self._sub_estimators_enabled = True
        self.calc = self._calc_adapter()
        self.measurements = delegate.measurements
        self.active_measurements = delegate.active_measurements
        self.active_z = delegate.active_z
        self.active_weight = delegate.active_weight
        delegate_state_labels = _DelegatedSequenceView(delegate, "state_labels") if direct_delegate else delegate.state_labels
        delegate_state_meta = _DelegatedSequenceView(delegate, "state_meta") if direct_delegate else getattr(delegate, "state_meta", [])
        self.state_meta = delegate_state_meta
        self.state_labels = delegate_state_labels
        self.ac_state_labels = delegate_state_labels if side == "ac" else []
        self.dc_state_labels = delegate_state_labels if side == "dc" else []
        self.ac_state_layout = (
            {"state_labels": delegate_state_labels, "n_state": int(delegate.n_state)}
            if direct_delegate and side == "ac"
            else delegate.state_layout() if side == "ac" else {"state_labels": [], "n_state": 0}
        )
        self.dc_state_layout = (
            {
                "state_labels": delegate_state_labels,
                "voltage_col": getattr(delegate, "voltage_col", np.array([], dtype=np.int32)),
                "n_state": delegate.n_state,
                "references": getattr(delegate, "references", []),
            }
            if side == "dc"
            else {"state_labels": [], "n_state": 0}
        )
        self.n_state = int(delegate.n_state)
        self.ac_n_state = int(delegate.n_state) if side == "ac" else 0
        self.dc_n_state = int(delegate.n_state) if side == "dc" else 0
        self.hybrid_n_state = 0
        self.dc_state_start = self.ac_n_state
        self.hybrid_state_start = self.ac_n_state + self.dc_n_state
        self.state_sides = _HybridStateSideView(self.ac_n_state, self.dc_n_state, 0)
        raw_delegate_v_cols = np.asarray(
            getattr(delegate, "voltage_cols", getattr(delegate, "voltage_col", [])),
            dtype=np.int32,
        )
        self.voltage_cols = (
            np.unique(raw_delegate_v_cols[raw_delegate_v_cols >= 0]).astype(np.int32, copy=False)
            if raw_delegate_v_cols.size
            else np.array([], dtype=np.int32)
        )
        self.power_flow_state = np.array([], dtype=np.float64)
        self.flat_state = np.array([], dtype=np.float64)
        self._initial_state_cache_ready = False
        self.dc_reference_nodes = getattr(delegate, "references", []) if side == "dc" else []
        self.ac_reference_nodes = getattr(delegate, "references", []) if side == "ac" else []
        self.dc_node_voltage_measurements = getattr(delegate, "node_voltage_measurements", {}) if side == "dc" else {}
        self.ac_node_voltage_measurements = getattr(delegate, "_node_voltage_measurement_cache", {}) if side == "ac" else {}
        self.ac_theta_state_col = getattr(delegate, "angle_col", np.array([], dtype=np.int32)) if side == "ac" else np.array([], dtype=np.int32)
        self.ac_voltage_state_col = getattr(delegate, "voltage_col", np.array([], dtype=np.int32)) if side == "ac" else np.array([], dtype=np.int32)
        self.dc_voltage_state_col = getattr(delegate, "voltage_col", np.array([], dtype=np.int32)) if side == "dc" else np.array([], dtype=np.int32)
        self.targeted_observability_pseudo_count = getattr(delegate, "targeted_observability_pseudo_count", 0)
        if direct_delegate:
            empty_rows = np.array([], dtype=np.int32)
            self.ac_meas_rows = empty_rows
            self.dc_meas_rows = empty_rows
            self.hybrid_meas_rows = empty_rows
            self.ac_meas = []
            self.dc_meas = []
            self.hybrid_meas = []
            self._active_ac_sub_measurements = []
            self._active_dc_sub_measurements = []
            self._active_ac_sub_rows = empty_rows
            self._active_dc_sub_rows = empty_rows
            self._active_ac_hybrid_rows = empty_rows
            self._active_dc_hybrid_rows = empty_rows
            self._active_ac_delegated_row_mask = np.array([], dtype=bool)
            self._active_dc_delegated_row_mask = np.array([], dtype=bool)
            self._jacobian_static_skip = np.array([], dtype=bool)
        else:
            self._adopt_delegate_active_partition(side)
        if side == "ac":
            self.ac_state_cols = np.arange(self.n_state, dtype=np.int32)
            self.dc_state_cols = np.array([], dtype=np.int32)
            self.hybrid_state_cols = np.array([], dtype=np.int32)
            self.ac_vars = self.state_labels
            self.dc_vars = []
            self.hybrid_vars = []
            self.ac_state_slice = slice(0, self.n_state)
            self.dc_state_slice = slice(self.n_state, self.n_state)
            self.hybrid_state_slice = slice(self.n_state, self.n_state)
        else:
            self.ac_state_cols = np.array([], dtype=np.int32)
            self.dc_state_cols = np.arange(self.n_state, dtype=np.int32)
            self.hybrid_state_cols = np.array([], dtype=np.int32)
            self.ac_vars = []
            self.dc_vars = self.state_labels
            self.hybrid_vars = []
            self.ac_state_slice = slice(0, 0)
            self.dc_state_slice = slice(0, self.n_state)
            self.hybrid_state_slice = slice(self.n_state, self.n_state)
        self._link_converter_endpoint_nodes()

    def _adopt_delegate_active_partition(self, side: str) -> None:
        count = len(self.active_measurements)
        rows = np.arange(count, dtype=np.int32)
        empty_rows = np.array([], dtype=np.int32)
        if side == "ac":
            self.ac_meas_rows = rows
            self.dc_meas_rows = empty_rows
            self.hybrid_meas_rows = empty_rows
            self.ac_meas = self.active_measurements
            self.dc_meas = []
            self.hybrid_meas = []
            self._active_ac_hybrid_rows = rows.copy()
            self._active_dc_hybrid_rows = empty_rows
        elif side == "dc":
            self.ac_meas_rows = empty_rows
            self.dc_meas_rows = rows
            self.hybrid_meas_rows = empty_rows
            self.ac_meas = []
            self.dc_meas = self.active_measurements
            self.hybrid_meas = []
            self._active_ac_hybrid_rows = empty_rows
            self._active_dc_hybrid_rows = rows.copy()
        else:
            self._partition_active_measurements()
            return
        self._active_ac_sub_measurements = self.ac_meas
        self._active_dc_sub_measurements = self.dc_meas
        self._active_ac_sub_rows = np.arange(len(self.ac_meas), dtype=np.int32)
        self._active_dc_sub_rows = np.arange(len(self.dc_meas), dtype=np.int32)
        self._active_ac_delegated_row_mask = np.zeros(count, dtype=bool)
        self._active_dc_delegated_row_mask = np.zeros(count, dtype=bool)
        if side == "ac" and count:
            self._active_ac_delegated_row_mask[:] = True
        elif side == "dc" and count:
            self._active_dc_delegated_row_mask[:] = True
        self._jacobian_static_skip = np.zeros(count, dtype=bool)

    def _delegate(self):
        return getattr(self, "_delegate_estimator", None) if getattr(self, "_sub_estimators_enabled", False) else None

    class _ACCalcAdapter:
        def __init__(self, estimator: ACStateEstimator):
            self.estimator = estimator
            ids = np.asarray(getattr(estimator, "_ac_node_id_lookup_ids", ()), dtype=np.int64)
            pos = np.asarray(getattr(estimator, "_ac_node_id_lookup_pos", ()), dtype=np.int64)
            self.node_pos = {int(idx): int(value) for idx, value in zip(ids, pos) if int(value) >= 0}
            self.N = int(getattr(estimator, "n_nodes", len(self.node_pos)))

        def _extract_state_vars(self, x, update_cache=False):
            theta, voltage = self.estimator._unpack_state(np.asarray(x, dtype=np.float64))
            return theta, voltage, np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    class _DCCalcAdapter:
        def __init__(self, estimator: DCStateEstimator):
            self.estimator = estimator
            ids = np.asarray(getattr(estimator, "_node_idx_lookup_ids", ()), dtype=np.int64)
            pos = np.asarray(getattr(estimator, "_node_idx_lookup_pos", ()), dtype=np.int64)
            self.node_pos = {int(idx): int(value) for idx, value in zip(ids, pos) if int(value) >= 0}
            self.alive_node_dict = self.node_pos
            self.N = int(getattr(estimator, "n_nodes", len(self.node_pos)))

    def _calc_adapter(self):
        ac_calc = self._ACCalcAdapter(self._ac_sub_estimator) if self._ac_sub_estimator is not None else None
        dc_calc = self._DCCalcAdapter(self._dc_sub_estimator) if self._dc_sub_estimator is not None else None
        return SimpleNamespace(
            ac_calc=ac_calc,
            dc_calc=dc_calc,
            ac_size=int(getattr(self._ac_sub_estimator, "n_state", 0) or 0),
            dc_size=int(getattr(self._dc_sub_estimator, "n_state", 0) or 0),
            total_vars=int(getattr(self._ac_sub_estimator, "n_state", 0) or 0)
            + int(getattr(self._dc_sub_estimator, "n_state", 0) or 0)
            + 3 * len(getattr(self, "dcac_converters", []))
        )

    def _partition_active_measurements(self) -> None:
        partitions = partition_measurements_by_code(
            self.active_measurements,
            self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
            table_builder=_measurement_table_from_measurements,
            as_view=False,
            sides=("ac", "dc", "hybrid"),
        )
        self.ac_meas_rows = partitions.rows["ac"].astype(np.int32, copy=False)
        self.dc_meas_rows = partitions.rows["dc"].astype(np.int32, copy=False)
        self.hybrid_meas_rows = partitions.rows["hybrid"].astype(np.int32, copy=False)
        self.ac_meas = partitions.measurements["ac"]
        self.dc_meas = partitions.measurements["dc"]
        self.hybrid_meas = partitions.measurements["hybrid"]
        self._active_measurement_blocks = {
            "ac": self._MeasurementSideBlock(
                self.ac_meas_rows,
                self.ac_meas,
                self._sub_measurement_plan_tables(self._ac_sub_estimator, self.ac_meas),
            ),
            "dc": self._MeasurementSideBlock(
                self.dc_meas_rows,
                self.dc_meas,
                self._sub_measurement_plan_tables(self._dc_sub_estimator, self.dc_meas),
            ),
            "hybrid": self._MeasurementSideBlock(self.hybrid_meas_rows, self.hybrid_meas),
        }
        self._active_ac_hybrid_rows = self.ac_meas_rows.copy()
        self._active_dc_hybrid_rows = self.dc_meas_rows.copy()
        self._active_ac_sub_measurements = list(self.ac_meas)
        self._active_dc_sub_measurements = list(self.dc_meas)
        self._active_ac_sub_rows = np.arange(len(self.ac_meas), dtype=np.int32)
        self._active_dc_sub_rows = np.arange(len(self.dc_meas), dtype=np.int32)
        self._ac_sub_to_hybrid_cols = np.arange(getattr(self, "ac_n_state", 0), dtype=np.int32)
        self._dc_sub_to_hybrid_cols = (
            np.arange(getattr(self, "dc_n_state", 0), dtype=np.int32) + int(getattr(self, "dc_state_start", 0))
        )
        self._active_ac_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_dc_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        if self._active_ac_hybrid_rows.size:
            self._active_ac_delegated_row_mask[self._active_ac_hybrid_rows] = True
        if self._active_dc_hybrid_rows.size:
            self._active_dc_delegated_row_mask[self._active_dc_hybrid_rows] = True
        self._jacobian_static_skip = np.zeros(len(self.active_measurements), dtype=bool)

    def _measurement_blocks_for(
        self,
        measurements: Sequence[Measurement],
    ) -> Dict[str, "HybridStateEstimator._MeasurementSideBlock"]:
        if measurements is self.active_measurements:
            return self._active_measurement_blocks
        partitions = partition_measurements_by_code(
            measurements,
            self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
            table_builder=_measurement_table_from_measurements,
            as_view=False,
            sides=("ac", "dc", "hybrid"),
        )
        return {
            "ac": self._MeasurementSideBlock(
                partitions.rows["ac"].astype(np.int32, copy=False),
                partitions.measurements["ac"],
            ),
            "dc": self._MeasurementSideBlock(
                partitions.rows["dc"].astype(np.int32, copy=False),
                partitions.measurements["dc"],
            ),
            "hybrid": self._MeasurementSideBlock(
                partitions.rows["hybrid"].astype(np.int32, copy=False),
                partitions.measurements["hybrid"],
            ),
        }

    @staticmethod
    def _sub_measurement_plan_tables(estimator, measurements: Sequence[Measurement]):
        if estimator is None or len(measurements) == 0:
            return None
        return estimator.install_measurement_runtime(measurements)

    @staticmethod
    def _sub_measurement_runtime_input(block: "HybridStateEstimator._MeasurementSideBlock"):
        return block.plan_tables if block.plan_tables is not None else block.measurements

    def _partition_state_variables(self) -> None:
        if (
            getattr(self, "ac_n_state", None) is not None
            and getattr(self, "dc_n_state", None) is not None
            and getattr(self, "hybrid_n_state", None) is not None
        ):
            ac_n = int(self.ac_n_state)
            dc_n = int(self.dc_n_state)
            hybrid_n = int(self.hybrid_n_state)
            dc_start = int(getattr(self, "dc_state_start", ac_n))
            hybrid_start = int(getattr(self, "hybrid_state_start", dc_start + dc_n))
            self.ac_state_cols = np.arange(ac_n, dtype=np.int32)
            self.dc_state_cols = dc_start + np.arange(dc_n, dtype=np.int32)
            self.hybrid_state_cols = hybrid_start + np.arange(hybrid_n, dtype=np.int32)
            if getattr(self, "_lazy_state_layout", False):
                self.ac_vars = self.state_labels[:ac_n]
                self.dc_vars = self.state_labels[dc_start : dc_start + dc_n]
                self.hybrid_vars = self.state_labels[hybrid_start : hybrid_start + hybrid_n]
            else:
                self.ac_vars = list(self.state_labels[:ac_n])
                self.dc_vars = list(self.state_labels[dc_start : dc_start + dc_n])
                self.hybrid_vars = list(self.state_labels[hybrid_start : hybrid_start + hybrid_n])
            self.ac_state_slice = slice(0, ac_n)
            self.dc_state_slice = slice(dc_start, dc_start + dc_n)
            self.hybrid_state_slice = slice(hybrid_start, hybrid_start + hybrid_n)
            return
        self.ac_state_cols = np.asarray(
            [idx for idx, side in enumerate(self.state_sides) if side == "ac"],
            dtype=np.int32,
        )
        self.dc_state_cols = np.asarray(
            [idx for idx, side in enumerate(self.state_sides) if side == "dc"],
            dtype=np.int32,
        )
        self.hybrid_state_cols = np.asarray(
            [idx for idx, side in enumerate(self.state_sides) if side == "hybrid"],
            dtype=np.int32,
        )
        self.ac_vars = [self.state_labels[idx] for idx in self.ac_state_cols]
        self.dc_vars = [self.state_labels[idx] for idx in self.dc_state_cols]
        self.hybrid_vars = [self.state_labels[idx] for idx in self.hybrid_state_cols]
        self.ac_state_slice = self._cols_to_slice(self.ac_state_cols)
        self.dc_state_slice = self._cols_to_slice(self.dc_state_cols)
        self.hybrid_state_slice = self._cols_to_slice(self.hybrid_state_cols)

    @staticmethod
    def _cols_to_slice(cols: np.ndarray) -> slice:
        if cols.size == 0:
            return slice(0, 0)
        return slice(int(cols[0]), int(cols[-1]) + 1)

    def _disable_ac_current_measurements(self) -> None:
        """Disable unsupported AC-subsystem currents.

        DCAC ``I_AC`` belongs to the coupling block and has its own direct
        measurement/Jacobian plan, so it must remain active here.
        """
        table = getattr(self.measurements, "table", None)
        if table is None:
            warnings.warn(
                "Hybrid SE AC-current disable requires table-backed measurements; Measurement object iteration is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        row_count = int(np.asarray(getattr(table, "idx", np.asarray([]))).size)
        if row_count == 0:
            return
        device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
        meas_type_code = np.asarray(getattr(table, "meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
        if device_type_code.size != row_count or meas_type_code.size != row_count:
            return
        ac_terminal_devices = np.asarray(
            (
                DEVICE_TYPE_ACBranch,
                DEVICE_TYPE_ACTransformer,
                DEVICE_TYPE_ACThreeWindingTransformer,
            ),
            dtype=np.int16,
        )
        ac_single_devices = np.asarray((DEVICE_TYPE_ACLoad, DEVICE_TYPE_ACGenerator), dtype=np.int16)
        ac_current_mask = (
            np.isin(device_type_code, ac_terminal_devices)
            & np.isin(
                meas_type_code,
                np.asarray((MEAS_TYPE_I_FROM, MEAS_TYPE_I_TO, MEAS_TYPE_I_THIRD), dtype=np.int16),
            )
        )
        ac_current_mask |= (
            np.isin(device_type_code, ac_single_devices)
            & np.isin(meas_type_code, np.asarray((MEAS_TYPE_I_LOAD, MEAS_TYPE_I_GEN), dtype=np.int16))
        )
        if not np.any(ac_current_mask):
            return
        table.valid[ac_current_mask] = False
        measurement_table_status_code(table)[ac_current_mask] = MEAS_STATUS_INVALID
        self._invalidate_measurement_activity_summary()

    def _measurement_device_count_by_type_code(self) -> Dict[int, int]:
        ac = getattr(self, "_ac_sub_estimator", None)
        dc = getattr(self, "_dc_sub_estimator", None)
        counts: Dict[int, int] = {}
        if ac is not None:
            counts.update(ac.measurement_device_counts())
        if dc is not None:
            counts.update(dc.measurement_device_counts())
        counts[DEVICE_TYPE_DCACConverter] = int(len(getattr(self, "dcac_converters", ())))
        return counts

    def _dead_or_inactive_device_info(self) -> Tuple[Dict[int, set], Dict[int, Dict[str, float]]]:
        """按需聚合 AC / DC 子估计器及换流器中处于死岛或停运状态的设备名集合与电压基准。"""
        dead_names: Dict[int, set] = {}
        dead_scales: Dict[int, Dict[str, float]] = {}
        for sub in (getattr(self, "_ac_sub_estimator", None), getattr(self, "_dc_sub_estimator", None)):
            if sub is None:
                continue
            fn = getattr(sub, "_dead_or_inactive_device_info", None)
            if callable(fn):
                sub_names, sub_scales = fn()
                for code, name_set in sub_names.items():
                    dead_names.setdefault(int(code), set()).update(name_set)
                for code, scale_map in sub_scales.items():
                    dead_scales.setdefault(int(code), {}).update(scale_map)
        all_convs = getattr(getattr(self, "network", None), "dcac_converters", ()) or ()
        if all_convs:
            alive_names = {str(conv.name) for conv in getattr(self, "dcac_converters", ())}
            dead_conv_names = {str(conv.name) for conv in all_convs if str(conv.name) not in alive_names}
            if dead_conv_names:
                dead_names.setdefault(int(DEVICE_TYPE_DCACConverter), set()).update(dead_conv_names)
        return dead_names, dead_scales

    def _disable_unavailable_measurements(self) -> None:
        device_count_by_code = self._measurement_device_count_by_type_code()
        unsupported_switch_codes = {
            DEVICE_TYPE_ACSwitch,
            DEVICE_TYPE_ACSwitchConstraint,
            DEVICE_TYPE_DCSwitch,
            DEVICE_TYPE_DCSwitchConstraint,
        }
        table = getattr(self.measurements, "table", None)
        if table is not None:
            table_size = int(table.idx.size)
            disabled_rows = []
            if table_size:
                valid = np.asarray(table.valid, dtype=bool)
                weight = np.asarray(table.weight, dtype=np.float64)
                active = valid & (weight > 0.0)
                known = np.zeros(table_size, dtype=bool)
                rows_by_code = getattr(table, "rows_by_device_type_code", None)
                if rows_by_code is None:
                    rows_by_code = rows_by_device_type_code(table)
                device_pos = getattr(table, "device_pos", None)
                has_device_pos = device_pos is not None and np.asarray(device_pos).size == table_size
                if has_device_pos:
                    device_pos = np.asarray(device_pos, dtype=np.int64)

                for code in unsupported_switch_codes:
                    rows = rows_by_code.get(int(code))
                    rows = np.asarray(rows, dtype=np.int64) if rows is not None else np.asarray([], dtype=np.int64)
                    if rows.size:
                        rows = rows[active[rows]]
                        if rows.size:
                            known[rows] = True
                            disabled_rows.append(rows)

                for code, device_count in device_count_by_code.items():
                    rows = rows_by_code.get(int(code))
                    rows = np.asarray(rows, dtype=np.int64) if rows is not None else np.asarray([], dtype=np.int64)
                    if rows.size == 0:
                        continue
                    rows = rows[active[rows]]
                    if rows.size == 0:
                        continue
                    known[rows] = True
                    if int(device_count) <= 0 or not has_device_pos:
                        disabled_rows.append(rows)
                        continue
                    pos = device_pos[rows.astype(np.intp, copy=False)]
                    available = (pos >= 0) & (pos < int(device_count))
                    if not np.all(available):
                        disabled_rows.append(rows[~available])

                unknown_active = np.flatnonzero(active & ~known)
                if unknown_active.size:
                    disabled_rows.append(unknown_active)

                if disabled_rows:
                    rows = np.concatenate(disabled_rows).astype(np.int64, copy=False)
                    from secore.meas_prefilter import tag_dead_island_unavailable_rows

                    dead_names_by_type, dead_scales_by_type = self._dead_or_inactive_device_info()
                    tag_dead_island_unavailable_rows(
                        table,
                        rows,
                        self.params,
                        dead_device_names_by_type=dead_names_by_type,
                        dead_device_scales_by_type=dead_scales_by_type,
                        dead_power_scale=float(self._power_file_base()),
                    )
                    table.valid[rows] = False
                    measurement_table_status_code(table)[rows] = MEAS_STATUS_INVALID
            return

    def _sub_attrs_snapshot(self, estimator) -> Dict[str, object]:
        names = (
            "measurements",
            "measurement_table",
            "_active_device_key_cache",
            "_active_measurement_key_cache",
            "_max_measurement_idx",
            "_node_voltage_measurement_cache",
            "_real_power_measurement_seed_cache",
            "_has_valid_angle_measurements",
        )
        return {name: getattr(estimator, name) for name in names if hasattr(estimator, name)}

    @staticmethod
    def _restore_sub_attrs(estimator, snapshot: Dict[str, object]) -> None:
        for name, value in snapshot.items():
            setattr(estimator, name, value)

    @staticmethod
    def _max_measurement_idx_fast(measurements: Sequence[Measurement]) -> int:
        table = getattr(measurements, "table", None)
        if table is None or int(np.asarray(getattr(table, "idx", np.asarray([]))).size) != len(measurements):
            warnings.warn(
                "Hybrid SE requires table-backed measurements for max idx; Measurement object scan is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0
        return int(np.asarray(table.idx, dtype=np.int64).max()) if int(table.idx.size) else 0

    @staticmethod
    def _activity_key_sets_from_table(table, active_mask: np.ndarray) -> Tuple[set, set]:
        row_count = int(np.asarray(getattr(table, "idx", np.asarray([]))).size)
        device_pos = getattr(table, "device_pos", None)
        meas_type_code = getattr(table, "meas_type_code", None)
        if device_pos is not None and meas_type_code is not None:
            device_type_code = np.asarray(table.device_type_code, dtype=np.int64)
            device_pos = np.asarray(device_pos, dtype=np.int64)
            meas_type_code = np.asarray(meas_type_code, dtype=np.int64)
            if device_type_code.size == row_count and device_pos.size == row_count and meas_type_code.size == row_count:
                active = np.asarray(active_mask, dtype=bool)
                valid_device = active & (device_type_code > 0) & (device_pos >= 0)
                measured_devices = set(
                    HybridStateEstimator._packed_device_key_array(
                        device_type_code[valid_device],
                        device_pos[valid_device],
                    ).tolist()
                )
                valid_measurement = valid_device & (meas_type_code > 0)
                active_keys = set(
                    HybridStateEstimator._packed_measurement_key_array(
                        device_type_code[valid_measurement],
                        device_pos[valid_measurement],
                        meas_type_code[valid_measurement],
                    ).tolist()
                )
                return measured_devices, active_keys
        return set(), set()

    def _sub_measurement_summary_attrs(
        self,
        measurements: Sequence[Measurement],
        max_idx: Optional[int] = None,
        voltage_node_mapper=None,
    ) -> Dict[str, object]:
        cached_attrs = self._cached_sub_measurement_summary_attrs(
            measurements,
            max_idx=max_idx,
            voltage_node_mapper=voltage_node_mapper,
        )
        if cached_attrs is not None:
            return cached_attrs
        active_device_keys = set()
        active_measurement_keys = set()
        voltage_best: Dict[int, Tuple[float, float]] = {}
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            active = table.valid & (table.weight > 0.0)
            active_device_keys, active_measurement_keys = self._activity_key_sets_from_table(table, active)
            if voltage_node_mapper is not None and np.any(active):
                status_code = measurement_table_status_code(table)
                meas_type_code = np.asarray(getattr(table, "meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
                device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
                device_pos = self._hybrid_measurement_device_pos_array(table)
                if meas_type_code.size != table.idx.size or device_pos.size != table.idx.size:
                    self._warn_missing_device_pos("sub_summary_voltage", 0)
                    meas_type_code = np.asarray([], dtype=np.int16)
                voltage_codes = np.asarray(
                    [MEAS_TYPE_V, MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO, MEAS_TYPE_V_THIRD, MEAS_TYPE_V_GEN, MEAS_TYPE_V_LOAD, MEAS_TYPE_V_DC, MEAS_TYPE_V_AC],
                    dtype=np.int16,
                )
                voltage_mask = (
                    active
                    & (status_code != MEAS_STATUS_PSEUDO)
                    & np.isin(meas_type_code, voltage_codes)
                    & (device_pos >= 0)
                )
                for row in np.flatnonzero(voltage_mask):
                    node_idx = voltage_node_mapper(
                        int(device_type_code[int(row)]),
                        int(device_pos[int(row)]),
                        int(meas_type_code[int(row)]),
                    )
                    if node_idx is not None:
                        weight = float(table.weight[row])
                        current = voltage_best.get(int(node_idx))
                        if current is None or weight > current[0]:
                            voltage_best[int(node_idx)] = (weight, float(table.value[row]))
        else:
            self._warn_missing_device_pos("sub_summary_no_table", 0)
        attrs = {
            "_active_device_key_cache": active_device_keys,
            "_active_measurement_key_cache": active_measurement_keys,
            "_max_measurement_idx": int(max_idx) if max_idx is not None else self._max_measurement_idx_fast(self.measurements),
        }
        if voltage_node_mapper is not None:
            attrs["_real_voltage_observation_node_cache"] = {
                node_idx: value for node_idx, (_weight, value) in voltage_best.items()
            }
        return attrs

    @staticmethod
    def _copy_summary_attrs(attrs: Dict[str, object]) -> Dict[str, object]:
        copied = {}
        for name, value in attrs.items():
            if name == "_real_voltage_observation_node_cache":
                copied[name] = value
            elif isinstance(value, set):
                copied[name] = set(value)
            elif isinstance(value, dict):
                copied[name] = dict(value)
            else:
                copied[name] = value
        return copied

    def _cache_sub_measurement_summary_attrs(self, side: str, attrs: Dict[str, object]) -> None:
        if not attrs:
            return
        cache = getattr(self, "_sub_measurement_summary_attrs_by_side", None)
        if cache is None:
            cache = {}
            self._sub_measurement_summary_attrs_by_side = cache
        cache[side] = self._copy_summary_attrs(attrs)

    def _cached_sub_measurement_summary_attrs(
        self,
        measurements: Sequence[Measurement],
        *,
        max_idx: Optional[int] = None,
        voltage_node_mapper=None,
    ) -> Optional[Dict[str, object]]:
        cache_by_side = getattr(self, "_sub_measurement_summary_attrs_by_side", None)
        sources_by_side = getattr(self, "_sub_measurement_sources_by_side", None)
        if not cache_by_side or not sources_by_side:
            return None
        side = None
        for candidate_side, source in sources_by_side.items():
            if measurements is source:
                side = candidate_side
                break
        if side is None:
            return None
        side_attrs = cache_by_side.get(side, {})
        if voltage_node_mapper is not None and "_real_voltage_observation_node_cache" not in side_attrs:
            return None
        summary_attrs = getattr(self, "_sub_measurement_global_summary_attrs", None)
        if summary_attrs is None:
            summary_attrs = {
                "source_count": len(self.measurements),
            }
            self._sub_measurement_global_summary_attrs = summary_attrs
        attrs = {
            "_active_device_key_cache": set(),
            "_active_measurement_key_cache": set(),
            "_max_measurement_idx": (
                int(max_idx)
                if max_idx is not None
                else self._max_measurement_idx_fast(self.measurements)
            ),
        }
        if "_real_voltage_observation_node_cache" in side_attrs:
            attrs["_real_voltage_observation_node_cache"] = side_attrs["_real_voltage_observation_node_cache"]
        return attrs

    def _sub_measurement_context(
        self,
        measurements: Sequence[Measurement],
        *,
        refresh_summary: bool = True,
        summary_measurements: Optional[Sequence[Measurement]] = None,
        summary_max_idx: Optional[int] = None,
        summary_voltage_node_mapper=None,
    ) -> "HybridStateEstimator._SubEstimatorMeasurementContext":
        summary_attrs = None
        if summary_measurements is not None:
            summary_attrs = self._sub_measurement_summary_attrs(
                summary_measurements,
                summary_max_idx,
                voltage_node_mapper=summary_voltage_node_mapper,
            )
        elif not refresh_summary:
            summary_attrs = {}
        return self._SubEstimatorMeasurementContext(measurements=measurements, summary_attrs=summary_attrs)

    def _apply_sub_measurement_context(
        self,
        estimator,
        context: "HybridStateEstimator._SubEstimatorMeasurementContext",
    ) -> None:
        estimator.measurements = context.measurements
        if context.summary_attrs is None:
            refresh = getattr(estimator, "_refresh_measurement_summary_cache", None)
            if callable(refresh):
                refresh()
            return
        for name, value in context.summary_attrs.items():
            setattr(estimator, name, value)

    def _invoke_sub_estimator_methods(
        self,
        estimator,
        context: "HybridStateEstimator._SubEstimatorMeasurementContext",
        method_names: Sequence[str],
        *args,
        preserve_max_idx: bool = False,
        capture_attrs: Sequence[str] = (),
        captured_attrs: Optional[Dict[str, object]] = None,
    ):
        if estimator is None:
            return None
        snapshot = self._sub_attrs_snapshot(estimator)
        max_idx = None
        result = None
        try:
            self._apply_sub_measurement_context(estimator, context)
            for index, method_name in enumerate(method_names):
                method = getattr(estimator, method_name, None)
                if not callable(method):
                    continue
                if index == 0:
                    result = method(*args)
                else:
                    method()
            if captured_attrs is not None:
                captured_attrs.clear()
                captured_attrs.update(
                    {
                        name: getattr(estimator, name)
                        for name in capture_attrs
                        if hasattr(estimator, name)
                    }
                )
            max_idx = getattr(estimator, "_max_measurement_idx", None)
            return result
        finally:
            self._restore_sub_attrs(estimator, snapshot)
            if preserve_max_idx and max_idx is not None:
                estimator._max_measurement_idx = max_idx

    def _call_sub_with_measurements(
        self,
        estimator,
        measurements: List[Measurement],
        method_name: str,
        *args,
        refresh_summary: bool = True,
        preserve_max_idx: bool = False,
        summary_measurements: Optional[Sequence[Measurement]] = None,
        summary_max_idx: Optional[int] = None,
        capture_attrs: Sequence[str] = (),
        captured_attrs: Optional[Dict[str, object]] = None,
    ):
        context = self._sub_measurement_context(
            measurements,
            refresh_summary=refresh_summary,
            summary_measurements=summary_measurements,
            summary_max_idx=summary_max_idx,
        )
        return self._invoke_sub_estimator_methods(
            estimator,
            context,
            (method_name,),
            *args,
            preserve_max_idx=preserve_max_idx,
            capture_attrs=capture_attrs,
            captured_attrs=captured_attrs,
        )

    def _call_sub_sequence_with_measurements(
        self,
        estimator,
        measurements: List[Measurement],
        method_names: Sequence[str],
        *,
        refresh_summary: bool = True,
        preserve_max_idx: bool = False,
        summary_measurements: Optional[Sequence[Measurement]] = None,
        summary_max_idx: Optional[int] = None,
        summary_voltage_node_mapper=None,
        capture_attrs: Sequence[str] = (),
        captured_attrs: Optional[Dict[str, object]] = None,
    ) -> None:
        context = self._sub_measurement_context(
            measurements,
            refresh_summary=refresh_summary,
            summary_measurements=summary_measurements,
            summary_max_idx=summary_max_idx,
            summary_voltage_node_mapper=summary_voltage_node_mapper,
        )
        self._invoke_sub_estimator_methods(
            estimator,
            context,
            tuple(method_names),
            preserve_max_idx=preserve_max_idx,
            capture_attrs=capture_attrs,
            captured_attrs=captured_attrs,
        )

    def _adopt_sub_measurement_update(self, captured_attrs: Dict[str, object]) -> bool:
        """Adopt a sub-estimator MeasurementList when it replaced the shared view."""
        updated = captured_attrs.get("measurements")
        if updated is None or updated is self.measurements:
            return False
        table = captured_attrs.get("measurement_table")
        if table is None:
            table = getattr(updated, "table", None)
        self.measurements = updated
        if table is not None:
            self.measurement_table = table
            try:
                self.measurements.table = table
            except AttributeError:
                pass
        self._invalidate_measurement_activity_summary()
        return True

    def _meas_ppc_for_source_rows(self, source_rows: np.ndarray) -> Dict:
        meas_ppc = getattr(self, "meas_ppc", None)
        if not isinstance(meas_ppc, dict):
            return {}
        rows = np.asarray(source_rows, dtype=np.int64)
        row_count = int(rows.size)
        out = dict(meas_ppc)
        full_count = int(np.asarray(meas_ppc.get("idx_array", np.asarray([]))).size)
        for key in (
            "idx_array",
            "weight_array",
            "valid_array",
            "value_array",
            "status_array",
            "device_type_code_array",
            "device_name_id_array",
            "meas_type_code_array",
            "angle_mask_array",
            "device_pos",
            "scale",
            "from_pos",
            "to_pos",
            "available",
            "name",
            "device_type",
            "device_name",
            "meas_type",
        ):
            value = meas_ppc.get(key)
            if isinstance(value, np.ndarray) and int(value.shape[0]) == full_count:
                out[key] = value[rows.astype(np.intp, copy=False)].copy()
        meas = meas_ppc.get("meas")
        if isinstance(meas, np.ndarray) and meas.ndim == 2 and int(meas.shape[0]) == full_count:
            out["meas"] = meas[rows.astype(np.intp, copy=False), :].copy()
        device_type_code = out.get("device_type_code_array")
        if isinstance(device_type_code, np.ndarray) and int(device_type_code.size) == row_count:
            out["rows_by_device_type_code"] = {
                int(code): np.flatnonzero(device_type_code == code).astype(np.int64, copy=False)
                for code in np.unique(device_type_code)
            }
        else:
            out["rows_by_device_type_code"] = {}
        out["_mutable_runtime_arrays"] = True
        out["normalized"] = bool(meas_ppc.get("normalized", False))
        return out

    @staticmethod
    def _sub_voltage_node_mapper(estimator):
        mapper = getattr(estimator, "voltage_measurement_node_idx", None)
        if not callable(mapper):
            return None

        def voltage_node_mapper(device_type_code: int, device_pos: int, meas_type_code: int):
            node_idx = mapper(device_type_code, device_pos, meas_type_code)
            return None if node_idx is None else int(node_idx)

        return voltage_node_mapper

    def _convert_measurements_to_pu(self) -> None:
        """Delegate AC/DC normalization and normalize only converter rows locally."""
        self._invalidate_measurement_activity_summary()
        if getattr(self, "_measurements_normalized", False):
            return
        sources = self._initial_measurement_sources_by_side()
        converted_by_sub = getattr(self, "_sub_measurements_converted_by_side", {})
        if not converted_by_sub.get("ac", False):
            captured_summary: Dict[str, object] = {}
            self._call_sub_with_measurements(
                self._ac_sub_estimator,
                sources["ac"],
                "_convert_measurements_to_pu",
                refresh_summary=False,
                capture_attrs=self._SUB_MEASUREMENT_SUMMARY_ATTRS,
                captured_attrs=captured_summary,
            )
            self._cache_sub_measurement_summary_attrs("ac", captured_summary)
            if not self._sub_estimator_updates_measurement_table(self._ac_sub_estimator):
                self._sync_partition_table_from_objects(sources["ac"])
            converted_by_sub["ac"] = True
            if hasattr(sources["ac"], "normalized"):
                sources["ac"].normalized = True
            if self._ac_sub_estimator is not None and hasattr(self._ac_sub_estimator.measurements, "normalized"):
                self._ac_sub_estimator.measurements.normalized = True
        if not converted_by_sub.get("dc", False):
            captured_summary = {}
            dc_rows = (
                np.asarray(self._sub_measurement_source_rows_by_side.get("dc", np.asarray([], dtype=np.int64)), dtype=np.int64)
                if isinstance(getattr(self, "_sub_measurement_source_rows_by_side", None), dict)
                else np.asarray([], dtype=np.int64)
            )
            dc_table = getattr(sources["dc"], "table", None)
            normalized_by_ppc = False
            if self._dc_sub_estimator is not None and dc_table is not None and dc_rows.size == int(dc_table.idx.size):
                dc_meas_ppc = self._meas_ppc_for_source_rows(dc_rows)
                normalize_from_ppc = getattr(self._dc_sub_estimator, "_normalize_measurements_to_pu_from_meas_ppc", None)
                if callable(normalize_from_ppc) and dc_meas_ppc:
                    self._dc_sub_estimator.meas_ppc = dc_meas_ppc
                    self._dc_sub_estimator.measurements = sources["dc"]
                    normalized_by_ppc = bool(normalize_from_ppc(dc_table, dc_meas_ppc))
                    captured_summary = {
                        name: getattr(self._dc_sub_estimator, name)
                        for name in self._SUB_MEASUREMENT_SUMMARY_ATTRS
                        if hasattr(self._dc_sub_estimator, name)
                    }
            if not normalized_by_ppc:
                self._call_sub_with_measurements(
                    self._dc_sub_estimator,
                    sources["dc"],
                    "_convert_measurements_to_pu",
                    refresh_summary=False,
                    capture_attrs=self._SUB_MEASUREMENT_SUMMARY_ATTRS,
                    captured_attrs=captured_summary,
                )
            self._cache_sub_measurement_summary_attrs("dc", captured_summary)
            if not self._sub_estimator_updates_measurement_table(self._dc_sub_estimator):
                self._sync_partition_table_from_objects(sources["dc"])
            converted_by_sub["dc"] = True
            if hasattr(sources["dc"], "normalized"):
                sources["dc"].normalized = True
            if self._dc_sub_estimator is not None and hasattr(self._dc_sub_estimator.measurements, "normalized"):
                self._dc_sub_estimator.measurements.normalized = True
        self._convert_hybrid_measurements_to_pu(sources["hybrid"])
        self._sync_measurement_table_from_partition_tables(sources)
        if hasattr(self.measurements, "normalized"):
            self.measurements.normalized = True
        self._measurements_normalized = True
        if getattr(self, "meas_ppc", None) is not None and getattr(self.measurements, "table", None) is not None:
            self.meas_ppc["normalized"] = True
            sync_meas_ppc_from_measurement_table(self.meas_ppc, self.measurements.table)

    def _measurement_name_arrays_by_type_code(self, requested_codes=None) -> Dict[int, np.ndarray]:
        requested = None if requested_codes is None else {int(code) for code in requested_codes}
        out: Dict[int, np.ndarray] = {}
        ac = getattr(self, "_ac_sub_estimator", None)
        if ac is not None:
            out.update(ac.measurement_device_names(requested))
        dc = getattr(self, "_dc_sub_estimator", None)
        if dc is not None:
            out.update(dc.measurement_device_names(requested))
        if requested is None or DEVICE_TYPE_DCACConverter in requested:
            out[DEVICE_TYPE_DCACConverter] = np.asarray([conv.name for conv in self.dcac_converters], dtype=object)
        return out

    def _populate_measurement_device_pos_from_sub_estimators(
        self,
        *,
        force: bool = False,
        name_attach_codes=None,
    ) -> None:
        table = getattr(self.measurements, "table", None)
        if table is None or len(table.idx) != len(self.measurements):
            return
        n_rows = int(table.idx.size)
        if n_rows == 0:
            table.device_pos = np.asarray([], dtype=np.int64)
            return
        meas_ppc = getattr(self, "meas_ppc", None)
        existing_device_pos = getattr(table, "device_pos", None)
        if existing_device_pos is not None and np.asarray(existing_device_pos).size == n_rows:
            device_pos = np.asarray(existing_device_pos, dtype=np.int64).copy()
        else:
            ppc_device_pos = meas_ppc.get("device_pos") if isinstance(meas_ppc, dict) else None
            if isinstance(ppc_device_pos, np.ndarray) and int(ppc_device_pos.size) == n_rows:
                device_pos = np.asarray(ppc_device_pos, dtype=np.int64).copy()
            else:
                device_pos = np.full(n_rows, -1, dtype=np.int64)

        sources_by_side = getattr(self, "_sub_measurement_sources_by_side", None)
        rows_by_side = getattr(self, "_sub_measurement_source_rows_by_side", None)
        source_count = int(getattr(self, "_sub_measurement_source_count", -1))
        can_reuse_partitions = (
            isinstance(sources_by_side, dict)
            and isinstance(rows_by_side, dict)
            and source_count == n_rows
        )
        if can_reuse_partitions:
            estimators = {"ac": self._ac_sub_estimator, "dc": self._dc_sub_estimator}
            for side, source in sources_by_side.items():
                estimator = estimators.get(side)
                estimator_measurements = getattr(estimator, "measurements", None) if estimator is not None else None
                source_table = getattr(estimator_measurements, "table", None)
                if source_table is None:
                    source_table = getattr(source, "table", None)
                rows = rows_by_side.get(side)
                source_pos = getattr(source_table, "device_pos", None) if source_table is not None else None
                if source_table is None or rows is None or source_pos is None:
                    continue
                rows = np.asarray(rows, dtype=np.int64)
                source_pos = np.asarray(source_pos, dtype=np.int64)
                if rows.size != source_pos.size or rows.size != int(getattr(source_table, "idx", np.asarray([])).size):
                    continue
                valid_rows = (rows >= 0) & (rows < n_rows) & (source_pos >= 0)
                if np.any(valid_rows):
                    device_pos[rows[valid_rows].astype(np.intp, copy=False)] = source_pos[
                        valid_rows
                    ].astype(np.int64, copy=False)

        previous_count = int(getattr(self, "_measurement_device_pos_populated_count", -1))
        unresolved = device_pos < 0
        name_attach_done_count = int(getattr(self, "_measurement_device_pos_name_attach_count", -1))
        should_attach = bool(force and np.any(unresolved) and name_attach_done_count != n_rows)
        if not should_attach and previous_count < 0:
            should_attach = bool(np.any(unresolved))

        if should_attach and isinstance(meas_ppc, dict):
            meas_ppc["device_pos"] = device_pos.copy()
            codes = np.asarray(getattr(table, "device_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
            unresolved_codes = (
                np.unique(codes[unresolved])
                if codes.size == n_rows and np.any(unresolved)
                else np.asarray([], dtype=np.int16)
            )
            if name_attach_codes is not None:
                allowed_codes = np.asarray(tuple(int(code) for code in name_attach_codes), dtype=np.int16)
                unresolved_codes = unresolved_codes[np.isin(unresolved_codes, allowed_codes)]
            device_pos = attach_device_pos_from_name_arrays(
                meas_ppc,
                self._measurement_name_arrays_by_type_code(unresolved_codes),
            )
            if name_attach_codes is None:
                self._measurement_device_pos_name_attach_count = n_rows

        table.device_pos = device_pos
        cache = getattr(table, "_device_pos_plan_cache", None)
        if cache is not None:
            cache.clear()
        if can_reuse_partitions:
            for side, source in sources_by_side.items():
                source_table = getattr(source, "table", None)
                rows = rows_by_side.get(side)
                if source_table is None or rows is None:
                    continue
                rows = np.asarray(rows, dtype=np.int64)
                if rows.size == int(getattr(source_table, "idx", np.asarray([])).size):
                    source_table.device_pos = device_pos[rows.astype(np.intp, copy=False)].astype(np.int64, copy=True)
                    source_cache = getattr(source_table, "_device_pos_plan_cache", None)
                    if source_cache is not None:
                        source_cache.clear()
        if isinstance(getattr(self, "meas_ppc", None), dict):
            self.meas_ppc["device_pos"] = device_pos
        if not can_reuse_partitions:
            self._sub_measurement_sources_by_side = None
            self._sub_measurement_source_rows_by_side = None
        self._sub_measurement_source_count = n_rows
        self._measurement_device_pos_populated_count = n_rows

    def _attach_converter_voltage_samples_to_sub_estimators(self) -> None:
        """提取 DCACConverter 上真实有效的 V_AC / V_DC 标幺量测并透传给 AC/DC 子估计器参与定电压母线竞选与求平均。"""
        table = getattr(self.measurements, "table", None)
        if table is None or int(getattr(table.idx, "size", 0)) == 0:
            return
        status_code = measurement_table_status_code(table)
        valid_mask = (
            np.asarray(table.valid, dtype=bool)
            & (status_code == MEAS_STATUS_NORMAL)
            & (np.asarray(table.weight, dtype=np.float64) > 0.0)
        )
        if not np.any(valid_mask):
            return
        code = np.asarray(table.device_type_code, dtype=np.int16)
        meas_code = np.asarray(getattr(table, "meas_type_code", np.empty(0, dtype=np.int16)), dtype=np.int16)
        if meas_code.size != code.size:
            return
        dev_pos = self._hybrid_measurement_device_pos_array(table)
        val_arr = np.asarray(table.value, dtype=np.float64)
        dcac_valid = valid_mask & (code == DEVICE_TYPE_DCACConverter) & (dev_pos >= 0)
        if not np.any(dcac_valid):
            return
        self._ensure_hybrid_converter_node_cache()
        dcac_ac_node = np.asarray(getattr(self, "_dcac_ac_node", np.empty(0, dtype=np.int64)), dtype=np.int64)
        dcac_dc_node = np.asarray(getattr(self, "_dcac_dc_node", np.empty(0, dtype=np.int64)), dtype=np.int64)

        if self._ac_sub_estimator is not None and dcac_ac_node.size:
            ac_rows = np.flatnonzero(dcac_valid & (meas_code == MEAS_TYPE_V_AC))
            if ac_rows.size:
                d_idx = dev_pos[ac_rows]
                in_r = (d_idx >= 0) & (d_idx < dcac_ac_node.size)
                if np.any(in_r):
                    ac_rows = ac_rows[in_r]
                    node_ids = dcac_ac_node[d_idx[in_r].astype(np.intp, copy=False)]
                    ac = self._ac_sub_estimator
                    ids = np.asarray(getattr(ac, "_ac_node_id_lookup_ids", np.empty(0, dtype=np.int64)), dtype=np.int64)
                    pos = np.asarray(getattr(ac, "_ac_node_id_lookup_pos", np.empty(0, dtype=np.int64)), dtype=np.int64)
                    if ids.size and pos.size == ids.size:
                        loc = np.searchsorted(ids, node_ids)
                        ok = (loc >= 0) & (loc < ids.size)
                        if np.any(ok):
                            ok_locs = loc[ok].astype(np.intp, copy=False)
                            ok[ok] = ids[ok_locs] == node_ids[ok]
                        if np.any(ok):
                            ac._extra_voltage_sample_pos = pos[loc[ok].astype(np.intp, copy=False)].astype(np.int32, copy=False)
                            ac._extra_voltage_sample_val = val_arr[ac_rows[ok]].astype(np.float64, copy=False)

        if self._dc_sub_estimator is not None and dcac_dc_node.size:
            dc_rows = np.flatnonzero(dcac_valid & (meas_code == MEAS_TYPE_V_DC))
            if dc_rows.size:
                d_idx = dev_pos[dc_rows]
                in_r = (d_idx >= 0) & (d_idx < dcac_dc_node.size)
                if np.any(in_r):
                    dc_rows = dc_rows[in_r]
                    node_ids = dcac_dc_node[d_idx[in_r].astype(np.intp, copy=False)]
                    dc = self._dc_sub_estimator
                    dc_ids = np.asarray(getattr(dc, "_node_idx_by_pos", np.empty(0, dtype=np.int64)), dtype=np.int64)
                    if dc_ids.size:
                        order = np.argsort(dc_ids)
                        sorted_ids = dc_ids[order]
                        loc = np.searchsorted(sorted_ids, node_ids)
                        ok = (loc >= 0) & (loc < sorted_ids.size)
                        if np.any(ok):
                            ok_locs = loc[ok].astype(np.intp, copy=False)
                            ok[ok] = sorted_ids[ok_locs] == node_ids[ok]
                        if np.any(ok):
                            dc._extra_voltage_sample_pos = order[loc[ok].astype(np.intp, copy=False)].astype(np.int32, copy=False)
                            dc._extra_voltage_sample_val = val_arr[dc_rows[ok]].astype(np.float64, copy=False)

    def _finalize_sub_estimators_after_measurement_prepare(self) -> None:
        self._attach_converter_voltage_samples_to_sub_estimators()
        if self._ac_sub_estimator is not None and (
            getattr(self._ac_sub_estimator, "_defer_prepare_finalize_pending", False)
            or getattr(self._ac_sub_estimator, "_prepare_defer_finalize", False)
            or not hasattr(self._ac_sub_estimator, "voltage_col")
        ):
            self._ac_sub_estimator.finalize_prepare(
                prepare_active_measurements=False,
                measurements_already_normalized=True,
            )
        if self._dc_sub_estimator is not None and (
            getattr(self._dc_sub_estimator, "_defer_prepare_finalize_pending", False)
            or getattr(self._dc_sub_estimator, "_prepare_defer_finalize", False)
            or not hasattr(self._dc_sub_estimator, "voltage_col")
        ):
            self._dc_sub_estimator.finalize_prepare(
                prepare_active_measurements=False,
                measurements_already_normalized=True,
            )

    @staticmethod
    def _sub_estimator_updates_measurement_table(estimator) -> bool:
        return isinstance(estimator, (ACStateEstimator, DCStateEstimator))

    @staticmethod
    def _sync_partition_table_from_objects(measurements: Sequence[Measurement]) -> bool:
        table = getattr(measurements, "table", None)
        if table is None or len(table.idx) != len(measurements):
            return False
        for pos, meas in enumerate(measurements):
            table.valid[pos] = bool(meas.valid)
            table.weight[pos] = float(meas.weight)
            table.value[pos] = float(meas.value)
        return True

    def _hybrid_measurement_device_pos_array(self, table) -> np.ndarray:
        row_count = int(np.asarray(getattr(table, "idx", np.asarray([]))).size)
        device_pos = getattr(table, "device_pos", None)
        if device_pos is not None and np.asarray(device_pos).size == row_count:
            return np.asarray(device_pos, dtype=np.int64)
        meas_ppc = getattr(self, "meas_ppc", None)
        ppc_device_pos = meas_ppc.get("device_pos") if isinstance(meas_ppc, dict) else None
        if isinstance(ppc_device_pos, np.ndarray) and int(ppc_device_pos.size) == row_count:
            device_pos = np.asarray(ppc_device_pos, dtype=np.int64)
            table.device_pos = device_pos
            return device_pos
        device_pos = np.full(row_count, -1, dtype=np.int64)
        table.device_pos = device_pos
        return device_pos

    def _convert_hybrid_measurements_to_pu(self, measurements: Sequence[Measurement]) -> None:
        table = getattr(measurements, "table", None)
        has_table = table is not None and len(table.idx) == len(measurements)
        if not has_table:
            self._warn_missing_device_pos("convert_hybrid_measurements_no_table", DEVICE_TYPE_DCACConverter)
            return

        status = measurement_table_status_code(table)
        active = (
            np.asarray(table.valid, dtype=bool)
            & (np.asarray(table.weight, dtype=np.float64) > 0.0)
            & (status == MEAS_STATUS_NORMAL)
        )
        if np.any(active):
            scale = np.ones(table.value.size, dtype=np.float64)
            code = np.asarray(table.device_type_code, dtype=np.int16)
            meas_code = np.asarray(getattr(table, "meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
            if meas_code.size != table.value.size:
                meas_code = np.zeros(table.value.size, dtype=np.int16)
            device_pos = self._hybrid_measurement_device_pos_array(table)
            self._ensure_hybrid_converter_scale_cache()

            dcac_rows = np.flatnonzero(active & (code == DEVICE_TYPE_DCACConverter) & (device_pos >= 0))
            if dcac_rows.size:
                dcac_pos = device_pos[dcac_rows].astype(np.intp, copy=False)
                scale_vals = np.ones(dcac_rows.size, dtype=np.float64)
                dcac_meas = meas_code[dcac_rows]
                power_mask = (dcac_meas == MEAS_TYPE_P_DC) | (dcac_meas == MEAS_TYPE_P_AC) | (dcac_meas == MEAS_TYPE_Q_AC)
                scale_vals[power_mask] = self._power_file_base()
                mask = dcac_meas == MEAS_TYPE_V_DC
                scale_vals[mask] = self._dcac_dc_v_file_scale_by_pos[dcac_pos[mask]]
                mask = dcac_meas == MEAS_TYPE_I_DC
                scale_vals[mask] = self._dcac_dc_i_file_scale_by_pos[dcac_pos[mask]]
                mask = dcac_meas == MEAS_TYPE_V_AC
                scale_vals[mask] = self._dcac_ac_v_file_scale_by_pos[dcac_pos[mask]]
                mask = dcac_meas == MEAS_TYPE_I_AC
                scale_vals[mask] = self._dcac_ac_i_file_scale_by_pos[dcac_pos[mask]]
                scale[dcac_rows] = scale_vals

            table.value[active] = np.divide(
                table.value[active],
                scale[active],
                out=table.value[active].copy(),
                where=np.abs(scale[active]) > 1e-12,
            )

        from secore.meas_prefilter import prefilter_electric_measurements

        prefilter_electric_measurements(
            table,
            self.params,
            raw_values=table.value,
            dead_power_scale=float(self._power_file_base()),
            side="hybrid",
        )
        object_count = list.__len__(measurements) if isinstance(measurements, list) else 0
        if object_count == table.value.size and object_count > 0:
            for pos, meas in enumerate(list.__iter__(measurements)):
                meas.value = float(table.value[pos])
                meas.valid = bool(table.valid[pos])
                meas.status = int(status[pos])
        else:
            source = getattr(measurements, "source", None)
            source_rows = getattr(measurements, "rows", None)
            source_count = list.__len__(source) if isinstance(source, list) else 0
            if source_count > 0 and source_rows is not None:
                for pos, source_row in enumerate(source_rows):
                    source[int(source_row)].value = float(table.value[pos])
                    source[int(source_row)].valid = bool(table.valid[pos])
                    source[int(source_row)].status = int(status[pos])

    def _sync_measurement_table_from_partition_tables(
        self,
        sources: Dict[str, Sequence[Measurement]],
    ) -> None:
        master_table = getattr(self.measurements, "table", None)
        if master_table is None or len(master_table.idx) != len(self.measurements):
            self._sync_measurement_table_from_objects()
            return
        rows_by_side = getattr(self, "_sub_measurement_source_rows_by_side", None)
        if rows_by_side is None:
            self._sync_measurement_table_from_objects()
            return
        for side, measurements in sources.items():
            rows = np.asarray(rows_by_side.get(side, ()), dtype=np.int64)
            table = getattr(measurements, "table", None)
            if table is None or len(table.idx) != len(measurements) or rows.size != len(measurements):
                self._sync_measurement_table_from_objects()
                return
            if rows.size == 0:
                continue
            # A side partition may reject additional rows, but it must not
            # reactivate rows already rejected by Hybrid-level ownership or
            # device-position validation.
            prev_valid = np.asarray(master_table.valid[rows], dtype=bool)
            part_valid = np.asarray(table.valid, dtype=bool)
            master_table.valid[rows] = prev_valid & part_valid
            master_table.weight[rows] = table.weight
            master_table.value[rows] = table.value
            master_status = measurement_table_status_code(master_table)
            part_status = measurement_table_status_code(table)
            newly_rejected = prev_valid & (~part_valid)
            if np.any(newly_rejected):
                master_status[rows[newly_rejected]] = part_status[newly_rejected]
            part_reasons = getattr(table, "prefilter_reason", None)
            if isinstance(part_reasons, np.ndarray) and int(part_reasons.size) == rows.size:
                from secore.meas_prefilter import ensure_prefilter_reason_array

                master_reasons = ensure_prefilter_reason_array(master_table)
                fill_mask = (master_reasons[rows] == "") & (part_reasons != "")
                if np.any(fill_mask):
                    master_reasons[rows[fill_mask]] = part_reasons[fill_mask]
        master_reasons_final = getattr(master_table, "prefilter_reason", None)
        if master_reasons_final is not None:
            self._prefilter_reasons = master_reasons_final

    def _sync_measurement_table_from_objects(self) -> None:
        table = getattr(self.measurements, "table", None)
        if table is None or len(table.idx) != len(self.measurements):
            return
        for pos, meas in enumerate(self.measurements):
            table.valid[pos] = bool(meas.valid)
            table.weight[pos] = float(meas.weight)
            table.value[pos] = float(meas.value)

    def _ac_voltage_base(self, node_idx: int) -> float:
        estimator = getattr(self, "_ac_sub_estimator", None)
        if estimator is not None:
            value = self._lookup_node_value_by_idx(
                int(node_idx),
                getattr(estimator, "_ac_node_id_lookup_ids", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_ac_node_id_lookup_pos", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_ac_node_vbase_by_pos", np.asarray([], dtype=np.float64)),
            )
            if value is not None:
                return value
        node_dict = getattr(getattr(self.network, "ac", None), "node_dict", {})
        if int(node_idx) in node_dict:
            return float(node_dict[int(node_idx)].vbase)
        return 1.0

    def _dc_voltage_base(self, node_idx: int) -> float:
        estimator = getattr(self, "_dc_sub_estimator", None)
        if estimator is not None:
            value = self._lookup_node_value_by_idx(
                int(node_idx),
                getattr(estimator, "_node_idx_lookup_ids", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_node_idx_lookup_pos", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_node_vbase_by_pos", np.asarray([], dtype=np.float64)),
            )
            if value is not None:
                return value
        node_dict = getattr(getattr(self.network, "dc", None), "node_dict", {})
        if int(node_idx) in node_dict:
            return float(node_dict[int(node_idx)].vbase)
        return 1.0

    @staticmethod
    def _lookup_node_values_by_idx(
        node_idx: np.ndarray,
        ids: np.ndarray,
        pos: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.ones(nodes.size, dtype=np.float64)
        matched_rows = np.zeros(nodes.size, dtype=bool)
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.asarray(pos, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if nodes.size == 0 or ids.size == 0 or pos.size == 0 or values.size == 0:
            return out, matched_rows
        loc = np.searchsorted(ids, nodes)
        in_range = loc < ids.size
        if not np.any(in_range):
            return out, matched_rows
        rows = np.flatnonzero(in_range)
        loc_rows = loc[rows].astype(np.intp, copy=False)
        matched = ids[loc_rows] == nodes[rows]
        if not np.any(matched):
            return out, matched_rows
        matched_row_idx = rows[matched]
        value_pos = pos[loc[matched_row_idx].astype(np.intp, copy=False)]
        valid = (value_pos >= 0) & (value_pos < values.size)
        if np.any(valid):
            final_rows = matched_row_idx[valid]
            out[final_rows] = values[value_pos[valid].astype(np.intp, copy=False)]
            matched_rows[final_rows] = True
        return out, matched_rows

    def _ac_voltage_base_array(self, node_idx: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.ones(nodes.size, dtype=np.float64)
        matched = np.zeros(nodes.size, dtype=bool)
        estimator = getattr(self, "_ac_sub_estimator", None)
        if estimator is not None and nodes.size:
            out, matched = self._lookup_node_values_by_idx(
                nodes,
                getattr(estimator, "_ac_node_id_lookup_ids", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_ac_node_id_lookup_pos", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_ac_node_vbase_by_pos", np.asarray([], dtype=np.float64)),
            )
        if nodes.size and not np.all(matched):
            node_dict = getattr(getattr(self.network, "ac", None), "node_dict", {})
            for row in np.flatnonzero(~matched):
                node = node_dict.get(int(nodes[int(row)]))
                if node is not None:
                    out[int(row)] = float(getattr(node, "vbase", 1.0) or 1.0)
        return out

    def _dc_voltage_base_array(self, node_idx: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.ones(nodes.size, dtype=np.float64)
        matched = np.zeros(nodes.size, dtype=bool)
        estimator = getattr(self, "_dc_sub_estimator", None)
        if estimator is not None and nodes.size:
            out, matched = self._lookup_node_values_by_idx(
                nodes,
                getattr(estimator, "_node_idx_lookup_ids", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_node_idx_lookup_pos", np.asarray([], dtype=np.int64)),
                getattr(estimator, "_node_vbase_by_pos", np.asarray([], dtype=np.float64)),
            )
        if nodes.size and not np.all(matched):
            node_dict = getattr(getattr(self.network, "dc", None), "node_dict", {})
            for row in np.flatnonzero(~matched):
                node = node_dict.get(int(nodes[int(row)]))
                if node is not None:
                    out[int(row)] = float(getattr(node, "vbase", 1.0) or 1.0)
        return out

    def _ac_current_base_array(self, node_idx: np.ndarray) -> np.ndarray:
        voltage_base = self._ac_voltage_base_array(node_idx)
        current = np.ones(voltage_base.size, dtype=np.float64)
        valid = np.abs(voltage_base) > 1e-12
        if np.any(valid):
            current[valid] = self.p_base_kW / (1000.0 * np.sqrt(3.0) * np.abs(voltage_base[valid]))
        return self.i_scale * current

    def _dc_current_base_array(self, node_idx: np.ndarray) -> np.ndarray:
        voltage_base = self._dc_voltage_base_array(node_idx)
        current = np.ones(voltage_base.size, dtype=np.float64)
        valid = np.abs(voltage_base) > 1e-12
        if np.any(valid):
            current[valid] = self.p_base_kW / (1000.0 * np.abs(voltage_base[valid]))
        return self.i_scale * current

    def _ensure_hybrid_converter_node_cache(self) -> None:
        dcac_count = len(self.dcac_converters)
        if (
            getattr(self, "_dcac_count", None) == dcac_count
            and hasattr(self, "_dcac_dc_node")
            and hasattr(self, "_dcac_ac_node")
        ):
            return
        self._dcac_count = dcac_count
        self._dcac_dc_node = np.asarray([int(conv.dc_node) for conv in self.dcac_converters], dtype=np.int64)
        self._dcac_ac_node = np.asarray([int(conv.ac_node) for conv in self.dcac_converters], dtype=np.int64)
        if hasattr(self, "_hybrid_converter_scale_cache_key"):
            delattr(self, "_hybrid_converter_scale_cache_key")

    def _ensure_hybrid_converter_scale_cache(self) -> None:
        self._ensure_hybrid_converter_node_cache()
        cache_key = (
            int(self._dcac_count),
            float(getattr(self, "p_base_kW", 1.0)),
            float(getattr(self, "u_scale", 1.0)),
            float(getattr(self, "i_scale", 1.0)),
        )
        if getattr(self, "_hybrid_converter_scale_cache_key", None) == cache_key:
            return
        dcac_dc_v = self._dc_voltage_base_array(self._dcac_dc_node)
        dcac_ac_v = self._ac_voltage_base_array(self._dcac_ac_node)
        self._dcac_dc_v_file_scale_by_pos = self.u_scale * dcac_dc_v
        self._dcac_ac_v_file_scale_by_pos = self.u_scale * dcac_ac_v
        self._dcac_dc_i_file_scale_by_pos = self._dc_current_base_array(self._dcac_dc_node)
        self._dcac_ac_i_file_scale_by_pos = self._ac_current_base_array(self._dcac_ac_node)
        self._hybrid_converter_scale_cache_key = cache_key

    def _ac_current_base(self, node_idx: int) -> float:
        return self.i_scale * ac_current_base_ka(self.p_base_kW, self._ac_voltage_base(node_idx))

    def _dc_current_base(self, node_idx: int) -> float:
        return self.i_scale * dc_current_base_ka(self.p_base_kW, self._dc_voltage_base(node_idx))

    def _ac_voltage_file_base(self, node_idx: int) -> float:
        return self.u_scale * self._ac_voltage_base(node_idx)

    def _dc_voltage_file_base(self, node_idx: int) -> float:
        return self.u_scale * self._dc_voltage_base(node_idx)

    def _power_file_base(self) -> float:
        return self.p_base

    @staticmethod
    def _lookup_node_value_by_idx(node_idx: int, ids: np.ndarray, pos: np.ndarray, values: np.ndarray) -> Optional[float]:
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.asarray(pos, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if ids.size == 0 or pos.size == 0 or values.size == 0:
            return None
        loc = int(np.searchsorted(ids, int(node_idx)))
        if loc < 0 or loc >= ids.size or int(ids[loc]) != int(node_idx):
            return None
        value_pos = int(pos[loc])
        if value_pos < 0 or value_pos >= values.size:
            return None
        return float(values[value_pos])

    def _append_pseudo_measurement(
        self,
        next_idx: int,
        name: str,
        device_type: str,
        device_name: str,
        meas_type: str,
        value: float,
        *,
        device_type_code: int = 0,
        device_pos: int = -1,
        meas_type_code: int = 0,
    ) -> int:
        device_type_code = int(device_type_code)
        meas_type_code = int(meas_type_code)
        device_pos = int(device_pos)
        if device_type_code <= 0 or meas_type_code <= 0 or device_pos < 0:
            self._warn_missing_device_pos("append_pseudo", device_type_code, device_name, meas_type_code)
            return next_idx
        measurement = Measurement.__new__(Measurement)
        measurement.idx = next_idx
        measurement.name = name
        measurement.device_type = device_type
        measurement.device_name = device_name
        measurement.meas_type = meas_type
        measurement.weight = self.pseudo_measurement_weight_pq
        measurement.valid = True
        measurement.value = float(value)
        mark_measurement_pseudo(measurement)
        measurement.device_type_code = device_type_code
        measurement.meas_type_code = meas_type_code
        measurement.device_pos = device_pos
        self._append_pseudo_measurement_objects((measurement,))
        return next_idx + 1

    def _measurement_spec(
        self,
        device_type_code: int,
        device_pos: int,
        meas_type_code: int,
        value: float,
        device_name: str = "",
    ) -> "_HybridConverterMeasurementSpec":
        device_type_code = int(device_type_code)
        meas_type_code = int(meas_type_code)
        pos_value = int(device_pos)
        if device_type_code <= 0 or meas_type_code <= 0 or pos_value < 0:
            self._warn_missing_device_pos("measurement_spec", device_type_code, device_name, meas_type_code)
        device_type = DEVICE_TYPE_NAMES.get(device_type_code, "")
        meas_type = MEAS_TYPE_NAMES.get(meas_type_code, "")
        return self._HybridConverterMeasurementSpec(
            device_type,
            device_name,
            meas_type,
            float(value),
            device_type_code,
            pos_value,
            meas_type_code,
        )

    def _pseudo_measurement_table_from_objects(
        self,
        measurements: Sequence[Measurement],
    ) -> MeasurementTable:
        count = len(measurements)
        idx = np.empty(count, dtype=np.int64)
        name = np.empty(count, dtype=object)
        device_type = np.empty(count, dtype=object)
        device_name = np.empty(count, dtype=object)
        meas_type = np.empty(count, dtype=object)
        weight = np.empty(count, dtype=np.float64)
        valid = np.ones(count, dtype=bool)
        value = np.empty(count, dtype=np.float64)
        device_type_code = np.empty(count, dtype=np.int16)
        meas_type_code = np.empty(count, dtype=np.int16)
        device_pos = np.empty(count, dtype=np.int64)
        angle_mask = np.zeros(count, dtype=bool)
        status_code = np.full(count, MEAS_STATUS_PSEUDO, dtype=np.int16)
        name_ids = np.full(count, -1, dtype=np.int64)
        rows_by_code: Dict[int, List[int]] = {}
        for pos, measurement in enumerate(measurements):
            idx[pos] = int(measurement.idx)
            name[pos] = measurement.name
            device_type[pos] = measurement.device_type
            device_name[pos] = measurement.device_name
            meas_type[pos] = measurement.meas_type
            weight[pos] = float(measurement.weight)
            valid[pos] = bool(measurement.valid)
            value[pos] = float(measurement.value)
            type_code = int(getattr(measurement, "device_type_code", 0))
            kind_code = int(getattr(measurement, "meas_type_code", 0))
            pos_value = int(getattr(measurement, "device_pos", -1))
            if type_code <= 0 or kind_code <= 0 or pos_value < 0:
                self._warn_missing_device_pos(
                    "pseudo_table",
                    type_code,
                    getattr(measurement, "device_name", ""),
                    kind_code,
                )
            device_type_code[pos] = type_code
            meas_type_code[pos] = kind_code
            device_pos[pos] = pos_value
            rows_by_code.setdefault(type_code, []).append(pos)
        value, weight = normalize_pseudo_measurement_arrays(
            meas_type_code,
            value,
            weight,
            weight_v=self.pseudo_measurement_weight_v,
            weight_pq=self.pseudo_measurement_weight_pq,
            preserve_explicit_pq_weight=False,
        )
        return MeasurementTable(
            idx=idx,
            name=name,
            device_type=device_type,
            device_name=device_name,
            meas_type=meas_type,
            weight=weight,
            valid=valid,
            value=value,
            device_type_code=device_type_code,
            angle_mask=angle_mask,
            status_code=status_code,
            rows_by_device_type_code={
                int(code): np.asarray(rows, dtype=np.int64)
                for code, rows in rows_by_code.items()
            },
            device_name_id=name_ids,
            meas_type_code=meas_type_code,
            device_pos=device_pos,
        )

    def _append_pseudo_measurement_objects(
        self,
        measurements: Sequence[Measurement],
    ) -> MeasurementTable:
        if not measurements:
            return MeasurementTable(
                idx=np.asarray([], dtype=np.int64),
                name=np.asarray([], dtype=object),
                device_type=np.asarray([], dtype=object),
                device_name=np.asarray([], dtype=object),
                meas_type=np.asarray([], dtype=object),
                weight=np.asarray([], dtype=np.float64),
                valid=np.asarray([], dtype=bool),
                value=np.asarray([], dtype=np.float64),
                device_type_code=np.asarray([], dtype=np.int16),
                angle_mask=np.asarray([], dtype=bool),
                status_code=np.asarray([], dtype=np.int16),
                device_name_id=np.asarray([], dtype=np.int64),
                meas_type_code=np.asarray([], dtype=np.int16),
                device_pos=np.asarray([], dtype=np.int64),
            )
        tail_table = self._pseudo_measurement_table_from_objects(measurements)
        master_table = getattr(self.measurements, "table", None)
        if master_table is None or int(master_table.idx.size) != len(self.measurements):
            master_table = _measurement_table_from_measurements(self.measurements)
        new_table = concat_measurement_tables(master_table, tail_table)
        self.measurement_table = new_table
        self.measurements = TableBackedMeasurementList(
            new_table,
            normalized=getattr(self.measurements, "normalized", False),
        )
        self._invalidate_measurement_activity_summary(preserve_real_voltage_cache=True)
        return tail_table

    def _measurement_table_with_string_columns(self, table: MeasurementTable) -> MeasurementTable:
        count = int(table.idx.size)
        if (
            int(np.asarray(table.name, dtype=object).size) == count
            and int(np.asarray(table.device_type, dtype=object).size) == count
            and int(np.asarray(table.device_name, dtype=object).size) == count
            and int(np.asarray(table.meas_type, dtype=object).size) == count
        ):
            return table
        idx = np.asarray(table.idx, dtype=np.int64)
        device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
        meas_type_code = np.asarray(getattr(table, "meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
        device_pos = np.asarray(getattr(table, "device_pos", np.asarray([], dtype=np.int64)), dtype=np.int64)
        if meas_type_code.size != count:
            meas_type_code = np.zeros(count, dtype=np.int16)
        if device_pos.size != count:
            device_pos = np.full(count, -1, dtype=np.int64)
        name = (
            np.asarray(table.name, dtype=object)
            if int(np.asarray(table.name, dtype=object).size) == count
            else np.asarray([f"m{int(value)}" for value in idx], dtype=object)
        )
        device_type = (
            np.asarray(table.device_type, dtype=object)
            if int(np.asarray(table.device_type, dtype=object).size) == count
            else np.asarray([DEVICE_TYPE_NAMES.get(int(code), "") for code in device_type_code], dtype=object)
        )
        device_name = (
            np.asarray(table.device_name, dtype=object)
            if int(np.asarray(table.device_name, dtype=object).size) == count
            else self._device_name_array_from_table(table, device_type_code, device_pos)
        )
        meas_type = (
            np.asarray(table.meas_type, dtype=object)
            if int(np.asarray(table.meas_type, dtype=object).size) == count
            else np.asarray([MEAS_TYPE_NAMES.get(int(code), "") for code in meas_type_code], dtype=object)
        )
        return MeasurementTable(
            idx=idx,
            name=name,
            device_type=device_type,
            device_name=device_name,
            meas_type=meas_type,
            weight=np.asarray(table.weight, dtype=np.float64),
            valid=np.asarray(table.valid, dtype=bool),
            value=np.asarray(table.value, dtype=np.float64),
            device_type_code=device_type_code,
            angle_mask=np.asarray(table.angle_mask, dtype=bool),
            status_code=measurement_table_status_code(table).copy(),
            rows_by_device_type_code=getattr(table, "rows_by_device_type_code", None),
            device_name_id=getattr(table, "device_name_id", None),
            meas_type_code=meas_type_code,
            device_pos=device_pos,
        )

    def _device_name_array_from_table(
        self,
        table: MeasurementTable,
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
    ) -> np.ndarray:
        device_name_id = getattr(table, "device_name_id", None)
        meas_ppc = getattr(self, "meas_ppc", {})
        device_names = meas_ppc.get("device_names") if isinstance(meas_ppc, dict) else None
        if device_name_id is not None and isinstance(device_names, np.ndarray):
            ids = np.asarray(device_name_id, dtype=np.int64)
            names = np.asarray(device_names, dtype=object)
            if ids.size == int(device_type_code.size) and names.size:
                out = np.empty(ids.size, dtype=object)
                out[:] = ""
                valid = (ids >= 0) & (ids < names.size)
                if np.any(valid):
                    out[valid] = names[ids[valid].astype(np.intp, copy=False)]
                missing = ~valid
                if np.any(missing):
                    fallback = self._device_name_array_from_type_pos(device_type_code[missing], device_pos[missing])
                    out[missing] = fallback
                return out
        return self._device_name_array_from_type_pos(device_type_code, device_pos)

    def _device_name_array_from_type_pos(
        self,
        device_type_code: np.ndarray,
        device_pos: np.ndarray,
    ) -> np.ndarray:
        names = np.empty(int(device_type_code.size), dtype=object)
        names[:] = ""
        type_codes = np.asarray(device_type_code, dtype=np.int16)
        positions = np.asarray(device_pos, dtype=np.int64)
        names_by_code = self._measurement_name_arrays_by_type_code(np.unique(type_codes))
        for code in np.unique(type_codes):
            rows = np.flatnonzero(type_codes == code)
            code_names = np.asarray(names_by_code.get(int(code), ()), dtype=object)
            row_pos = positions[rows]
            valid = (row_pos >= 0) & (row_pos < code_names.size)
            if np.any(valid):
                names[rows[valid]] = code_names[row_pos[valid].astype(np.intp, copy=False)]
            if np.any(~valid):
                names[rows[~valid]] = np.asarray(
                    [f"pos:{int(value)}" for value in row_pos[~valid]],
                    dtype=object,
                )
        return names

    def _hybrid_converter_spec_device_key(self, spec: "_HybridConverterMeasurementSpec") -> int:
        device_type_code = int(spec.device_type_code)
        device_pos = int(spec.device_pos)
        if device_type_code <= 0 or device_pos < 0:
            self._warn_missing_device_pos("converter_spec_key", device_type_code, spec.device_name, spec.meas_type_code)
        return self._packed_device_key(device_type_code, device_pos)

    def _add_pseudo_power_measurements(self) -> None:
        """Delegate AC/DC pseudo rows, then add missing converter-device pseudo rows."""
        if self._ac_sub_estimator is not None:
            add_balance = getattr(self._ac_sub_estimator, "_add_power_balance_constraint_measurements", None)
            methods = ["_add_pseudo_power_measurements"]
            if callable(add_balance):
                methods.append("_add_power_balance_constraint_measurements")
            max_idx = self._max_measurement_idx_fast(self.measurements)
            self._call_sub_sequence_with_measurements(
                self._ac_sub_estimator,
                self.measurements,
                methods,
                preserve_max_idx=True,
                summary_measurements=self._initial_measurement_sources_by_side()["ac"],
                summary_max_idx=max_idx,
                summary_voltage_node_mapper=self._sub_voltage_node_mapper(self._ac_sub_estimator),
                capture_attrs=("measurements", "measurement_table", "_max_measurement_idx"),
                captured_attrs=(captured_attrs := {}),
            )
            self._adopt_sub_measurement_update(captured_attrs)
            if callable(add_balance):
                self._disable_coupled_ac_power_balance_rows()
        if self._dc_sub_estimator is not None:
            add_constraints = getattr(self._dc_sub_estimator, "_add_zero_branch_constraint_measurements", None)
            methods = ["_add_pseudo_power_measurements"]
            if callable(add_constraints):
                methods.append("_add_zero_branch_constraint_measurements")
            max_idx = self._max_measurement_idx_fast(self.measurements)
            self._call_sub_sequence_with_measurements(
                self._dc_sub_estimator,
                self.measurements,
                methods,
                summary_measurements=self._initial_measurement_sources_by_side()["dc"],
                summary_max_idx=max_idx,
                summary_voltage_node_mapper=self._sub_voltage_node_mapper(self._dc_sub_estimator),
                capture_attrs=("measurements", "measurement_table", "_max_measurement_idx"),
                captured_attrs=(captured_attrs := {}),
            )
            self._adopt_sub_measurement_update(captured_attrs)
        self._add_hybrid_pseudo_measurements()

    def _disable_coupled_ac_power_balance_rows(self) -> None:
        coupled_pos = self._converter_coupled_ac_device_positions()
        if coupled_pos.size == 0:
            return
        table = getattr(self.measurements, "table", None)
        if table is not None:
            table_size = int(table.idx.size)
            if table_size:
                rows_by_code = getattr(table, "rows_by_device_type_code", None) or {}
                balance_code = int(DEVICE_TYPE_ACPowerBalance)
                balance_rows = rows_by_code.get(balance_code)
                if balance_rows is None:
                    balance_rows = np.flatnonzero(np.asarray(table.device_type_code, dtype=np.int16) == balance_code)
                else:
                    balance_rows = np.asarray(balance_rows, dtype=np.int64)
                if balance_rows.size:
                    matched = np.zeros(balance_rows.size, dtype=bool)
                    device_pos = getattr(table, "device_pos", None)
                    if device_pos is not None and np.asarray(device_pos).size == table_size:
                        matched |= np.isin(
                            np.asarray(device_pos, dtype=np.int64)[balance_rows.astype(np.intp, copy=False)],
                            coupled_pos,
                        )
                    else:
                        self._warn_missing_device_pos("disable_coupled_ac_power_balance", balance_code)
                    if np.any(matched):
                        disable_rows = balance_rows[matched]
                        table.valid[disable_rows] = False
                        measurement_table_status_code(table)[disable_rows] = MEAS_STATUS_INVALID
                        self._invalidate_measurement_activity_summary()
            return
        self._warn_missing_device_pos("disable_coupled_ac_power_balance_no_table", DEVICE_TYPE_ACPowerBalance)

    def _add_hybrid_pseudo_measurements(self) -> None:
        n_conv = len(self.dcac_converters)
        if n_conv == 0:
            return
        measured_devices = self._active_hybrid_converter_measurement_devices()
        converter_keys = self._hybrid_converter_device_key_array()
        if converter_keys.size == n_conv and measured_devices:
            if len(measured_devices) >= n_conv and all(int(key) in measured_devices for key in converter_keys):
                return
            measured_sorted = np.fromiter(measured_devices, dtype=np.int64, count=len(measured_devices))
            if measured_sorted.size > 1:
                measured_sorted.sort()
            pos_hit = np.searchsorted(measured_sorted, converter_keys)
            in_measured = (pos_hit < measured_sorted.size) & (
                measured_sorted[np.minimum(pos_hit, measured_sorted.size - 1)] == converter_keys
            )
            missing_pos = np.flatnonzero(~in_measured)
        else:
            missing_pos = np.arange(n_conv, dtype=np.int64)
        if missing_pos.size == 0:
            return

        # 预取真实电压观测节点，对 V_DC / V_AC 做 O(1) 覆盖判定，避免逐条构建 spec 与重复扫描
        ac_obs_nodes = self._real_voltage_observation_nodes("ac")
        dc_obs_nodes = self._real_voltage_observation_nodes("dc")
        ac_has_zero_ties = bool(getattr(self._ac_sub_estimator, "_has_zero_impedance_ties", False)) if self._ac_sub_estimator is not None else False
        dc_has_zero_ties = bool(getattr(self._dc_sub_estimator, "_has_zero_impedance_ties", False)) if self._dc_sub_estimator is not None else False
        dcac_dc_nodes = np.asarray(getattr(self, "_dcac_dc_node", ()), dtype=np.int64)
        dcac_ac_nodes = np.asarray(getattr(self, "_dcac_ac_node", ()), dtype=np.int64)
        voltage_covered = self._voltage_pseudo_is_covered_by_code

        converters = self.dcac_converters
        dp_list: List[int] = []
        mt_list: List[int] = []
        val_list: List[float] = []
        for pos in missing_pos.tolist():
            conv = converters[pos]
            p_dc = float(getattr(conv, "dc_p", 0.0) or 0.0)
            p_ac = float(getattr(conv, "ac_p", 0.0) or 0.0)
            q_ac = float(getattr(conv, "ac_q", 0.0) or 0.0)
            dp_list.extend((pos, pos, pos))
            mt_list.extend((MEAS_TYPE_P_DC, MEAS_TYPE_P_AC, MEAS_TYPE_Q_AC))
            val_list.extend((p_dc, p_ac, q_ac))
            if not dc_has_zero_ties and pos < dcac_dc_nodes.size:
                v_dc_cov = int(dcac_dc_nodes[pos]) in dc_obs_nodes
            else:
                v_dc_cov = voltage_covered(DEVICE_TYPE_DCACConverter, pos, MEAS_TYPE_V_DC)
            if not v_dc_cov:
                dc_node_obj = getattr(conv, "dc_node_obj", None)
                v_dc = float(getattr(dc_node_obj, "voltage", 1.0) or 1.0)
                dp_list.append(pos)
                mt_list.append(MEAS_TYPE_V_DC)
                val_list.append(v_dc)
            if not ac_has_zero_ties and pos < dcac_ac_nodes.size:
                v_ac_cov = int(dcac_ac_nodes[pos]) in ac_obs_nodes
            else:
                v_ac_cov = voltage_covered(DEVICE_TYPE_DCACConverter, pos, MEAS_TYPE_V_AC)
            if not v_ac_cov:
                ac_node_obj = getattr(conv, "ac_node_obj", None)
                v_ac = float(getattr(ac_node_obj, "voltage", 1.0) or 1.0)
                dp_list.append(pos)
                mt_list.append(MEAS_TYPE_V_AC)
                val_list.append(v_ac)

        count = len(dp_list)
        if count == 0:
            return
        next_idx = self._max_measurement_idx_fast(self.measurements) + 1
        mt_arr = np.asarray(mt_list, dtype=np.int16)
        dp_arr = np.asarray(dp_list, dtype=np.int64)
        dt_arr = np.full(count, DEVICE_TYPE_DCACConverter, dtype=np.int16)
        values_arr, weights_arr = normalize_pseudo_measurement_arrays(
            mt_arr,
            np.asarray(val_list, dtype=np.float64),
            np.full(count, self.pseudo_measurement_weight_pq, dtype=np.float64),
            weight_v=self.pseudo_measurement_weight_v,
            weight_pq=self.pseudo_measurement_weight_pq,
            preserve_explicit_pq_weight=False,
        )
        master_table = getattr(self.measurements, "table", None)
        if master_table is None or int(master_table.idx.size) != len(self.measurements):
            master_table = _measurement_table_from_measurements(self.measurements)
        need_strings = int(np.asarray(master_table.name, dtype=object).size) == int(master_table.idx.size) and int(master_table.idx.size) > 0
        if need_strings:
            mt_label_map = {
                MEAS_TYPE_P_DC: "P_DC",
                MEAS_TYPE_P_AC: "P_AC",
                MEAS_TYPE_Q_AC: "Q_AC",
                MEAS_TYPE_V_DC: "V_DC",
                MEAS_TYPE_V_AC: "V_AC",
            }
            conv_names = [converters[p].name for p in dp_list]
            mt_labels = [mt_label_map[m] for m in mt_list]
            name_arr = np.asarray([f"pseudo_{ml.lower()}_{cn}" for ml, cn in zip(mt_labels, conv_names)], dtype=object)
            dt_str_arr = np.full(count, "DCACConverter", dtype=object)
            dn_str_arr = np.asarray(conv_names, dtype=object)
            mt_str_arr = np.asarray(mt_labels, dtype=object)
        else:
            empty_obj = np.asarray([], dtype=object)
            name_arr = dt_str_arr = dn_str_arr = mt_str_arr = empty_obj
        tail_table = MeasurementTable(
            idx=np.arange(next_idx, next_idx + count, dtype=np.int64),
            name=name_arr,
            device_type=dt_str_arr,
            device_name=dn_str_arr,
            meas_type=mt_str_arr,
            weight=weights_arr,
            valid=np.ones(count, dtype=bool),
            value=values_arr,
            device_type_code=dt_arr,
            angle_mask=np.zeros(count, dtype=bool),
            status_code=np.full(count, MEAS_STATUS_PSEUDO, dtype=np.int16),
            rows_by_device_type_code={int(DEVICE_TYPE_DCACConverter): np.arange(count, dtype=np.int64)},
            device_name_id=np.full(count, -1, dtype=np.int64),
            meas_type_code=mt_arr,
            device_pos=dp_arr,
        )
        new_table = concat_measurement_tables(master_table, tail_table)
        self.measurement_table = new_table
        self.measurements = TableBackedMeasurementList(
            new_table,
            normalized=getattr(self.measurements, "normalized", False),
        )
        self._invalidate_measurement_activity_summary(preserve_real_voltage_cache=True)

    def _hybrid_converter_device_key_array(self) -> np.ndarray:
        cached = getattr(self, "_hybrid_converter_device_key_array_cache", None)
        expected_size = len(self.dcac_converters)
        if cached is not None and int(np.asarray(cached).size) == expected_size:
            return cached
        parts = []
        if self.dcac_converters:
            parts.append(
                self._packed_device_key_array(
                    np.full(len(self.dcac_converters), DEVICE_TYPE_DCACConverter, dtype=np.int64),
                    np.arange(len(self.dcac_converters), dtype=np.int64),
                )
            )
        keys = np.concatenate(parts).astype(np.int64, copy=False) if parts else np.empty(0, dtype=np.int64)
        self._hybrid_converter_device_key_array_cache = keys
        return keys

    def _active_hybrid_converter_measurement_devices(self) -> set:
        table = getattr(self.measurements, "table", None)
        if table is not None and len(table.idx) == len(self.measurements):
            source_count = int(getattr(self, "_sub_measurement_source_count", len(self.measurements)))
            return self._active_hybrid_converter_measurement_devices_from_table(table, row_limit=source_count)
        sources_by_side = getattr(self, "_sub_measurement_sources_by_side", None)
        if sources_by_side is None:
            return self._active_hybrid_converter_measurement_devices_from(self.measurements)
        return self._active_hybrid_converter_measurement_devices_from(sources_by_side.get("hybrid", ()))

    def _active_hybrid_converter_measurement_devices_from_table(self, table, row_limit: Optional[int] = None) -> set:
        row_count = int(table.idx.size)
        limit = row_count if row_limit is None else max(0, min(int(row_limit), row_count))
        if limit <= 0:
            return set()
        active = (
            np.asarray(table.valid[:limit], dtype=bool)
            & (np.asarray(table.weight[:limit], dtype=np.float64) > 0.0)
        )
        if not np.any(active):
            return set()
        code = np.asarray(table.device_type_code[:limit], dtype=np.int64)
        device_pos = self._hybrid_measurement_device_pos_array(table)[:limit]
        converter_rows = active & (code == DEVICE_TYPE_DCACConverter)
        if not np.any(converter_rows):
            return set()
        converter_mask = converter_rows & (device_pos >= 0)
        if np.any(converter_mask):
            return set(self._packed_device_key_array(code[converter_mask], device_pos[converter_mask]).tolist())
        self._warn_missing_device_pos("active_hybrid_converter_measurement_devices", DEVICE_TYPE_DCACConverter)
        return set()

    def _active_hybrid_converter_measurement_devices_from(self, measurements: Sequence[Measurement]) -> set:
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            return self._active_hybrid_converter_measurement_devices_from_table(table)
        self._warn_missing_device_pos("active_hybrid_converter_measurement_devices_sequence", DEVICE_TYPE_DCACConverter)
        return set()

    # Module-style constants reused across the 60K+ per-converter state-meta
    # rebuilds during prepare; previously the dicts were re-created on every
    # call which dominated state-layout construction for large grids.
    _AC_STATE_META_PREFIX_BY_KIND = {
        "angle": "AC_THETA",
        "voltage": "AC_V",
        "generator_p": "AC_GEN_P",
        "generator_q": "AC_GEN_Q",
        "load_p": "AC_LOAD_P",
        "load_q": "AC_LOAD_Q",
        "shunt_q": "AC_Q_SHUNT",
    }
    _DC_STATE_META_PREFIX_BY_KIND = {
        "voltage": "DC_V",
        "zero_current": "DC_I",
        "break_current": "DC_I",
        "dcdc_p_from": "DCDC_P_FROM",
        "dcdc_p_to": "DCDC_P_TO",
        "v_generator_p": "DC_VGEN_P",
    }

    @staticmethod
    def _ac_sub_state_meta_to_hybrid(meta: StateMeta) -> StateMeta:
        kind = meta.kind
        prefix = HybridStateEstimator._AC_STATE_META_PREFIX_BY_KIND.get(kind)
        if prefix is None:
            if kind == "zero_current" or kind == "break_current":
                prefix = "AC_I_RE" if meta.component == "re" else "AC_I_IM"
            else:
                prefix = f"AC_{kind.upper()}"
        device_name = meta.device_name
        # Construct StateMeta inline (rather than going through
        # `with_legacy_label`) to skip one Python-level function call per state.
        return StateMeta(
            "ac",
            kind,
            meta.device_type,
            device_name,
            terminal=meta.terminal,
            component=meta.component,
            legacy_label=f"{prefix}:{device_name}",
            device_pos=meta.device_pos,
            device_type_code=meta.device_type_code,
            meas_type_code=meta.meas_type_code,
        )

    @staticmethod
    def _dc_sub_state_meta_to_hybrid(meta: StateMeta) -> StateMeta:
        kind = meta.kind
        prefix = HybridStateEstimator._DC_STATE_META_PREFIX_BY_KIND.get(kind)
        if prefix is None:
            prefix = f"DC_{kind.upper()}"
        device_name = meta.device_name
        return StateMeta(
            "dc",
            kind,
            meta.device_type,
            device_name,
            terminal=meta.terminal,
            component=meta.component,
            legacy_label=f"{prefix}:{device_name}",
            device_pos=meta.device_pos,
            device_type_code=meta.device_type_code,
            meas_type_code=meta.meas_type_code,
        )

    def _build_state_layout(self) -> None:
        ac_n = int(getattr(self._ac_sub_estimator, "n_state", 0) or 0)
        dc_n = int(getattr(self._dc_sub_estimator, "n_state", 0) or 0)
        dcac_count = len(self.dcac_converters)
        hybrid_n = 3 * dcac_count
        self.ac_n_state = ac_n
        self.dc_n_state = dc_n
        self.hybrid_n_state = hybrid_n
        self.dc_state_start = ac_n
        self.hybrid_state_start = ac_n + dc_n
        self.dcac_state_start = ac_n + dc_n
        self.n_state = ac_n + dc_n + hybrid_n
        # Keep the public metadata sequence lazy for every non-empty Hybrid
        # layout.  Materializing one StateMeta plus several string/NumPy
        # conversions per AC/DC state made medium cases pay the same O(n)
        # object cost as large cases; the array-backed view preserves indexed
        # access and only materializes metadata that a caller actually reads.
        lazy_state_layout = self.n_state > 0
        self._lazy_state_layout = bool(lazy_state_layout)
        if lazy_state_layout:
            labels = _HybridStateLabelView(
                ac_n,
                dc_n,
                (conv.name for conv in self.dcac_converters),
            )
            self.state_labels = labels
            self.state_meta = _HybridStateMetaView(labels, self._ac_sub_estimator, self._dc_sub_estimator)
            # Retain the detailed labels exposed by the former eager path.
            # Large layouts keep their existing compact AC_STATE/DC_STATE labels.
            if self.n_state <= 20000:
                labels._ac_arrays = self.state_meta._ac_arrays
                labels._dc_arrays = self.state_meta._dc_arrays
            self.state_sides = _HybridStateSideView(ac_n, dc_n, hybrid_n)
            self.ac_state_labels = labels[:ac_n]
            self.dc_state_labels = labels[ac_n : ac_n + dc_n]
            self.ac_state_layout = {"state_labels": self.ac_state_labels, "n_state": ac_n}
            self.dc_state_layout = {"state_labels": self.dc_state_labels, "n_state": dc_n}
            self.dcac_p_dc_state_col = self.dcac_state_start + 3 * np.arange(dcac_count, dtype=np.int32)
            self.dcac_p_ac_state_col = self.dcac_p_dc_state_col + 1
            self.dcac_q_ac_state_col = self.dcac_p_dc_state_col + 2
            self._finish_state_layout_arrays()
            return
        state_meta: List[StateMeta] = []
        for idx in range(ac_n):
            meta = _state_meta_from_estimator_arrays(self._ac_sub_estimator, idx)
            if meta is None:
                state_meta.append(StateMeta("ac", "sub_state", "ACSubsystem", str(idx), legacy_label=f"AC_STATE:{idx}"))
            else:
                state_meta.append(self._ac_sub_state_meta_to_hybrid(meta))
        for idx in range(dc_n):
            meta = _state_meta_from_estimator_arrays(self._dc_sub_estimator, idx)
            if meta is None:
                state_meta.append(StateMeta("dc", "sub_state", "DCSubsystem", str(idx), legacy_label=f"DC_STATE:{idx}"))
            else:
                state_meta.append(self._dc_sub_state_meta_to_hybrid(meta))
        for pos, conv in enumerate(self.dcac_converters):
            state_meta.extend(
                (
                    StateMeta("hybrid", "dcac_p_dc", "DCACConverter", conv.name, terminal="dc", component="p", legacy_label=f"DCAC_P_DC:{conv.name}", device_pos=pos, device_type_code=DEVICE_TYPE_DCACConverter, meas_type_code=MEAS_TYPE_P_DC),
                    StateMeta("hybrid", "dcac_p_ac", "DCACConverter", conv.name, terminal="ac", component="p", legacy_label=f"DCAC_P_AC:{conv.name}", device_pos=pos, device_type_code=DEVICE_TYPE_DCACConverter, meas_type_code=MEAS_TYPE_P_AC),
                    StateMeta("hybrid", "dcac_q_ac", "DCACConverter", conv.name, terminal="ac", component="q", legacy_label=f"DCAC_Q_AC:{conv.name}", device_pos=pos, device_type_code=DEVICE_TYPE_DCACConverter, meas_type_code=MEAS_TYPE_Q_AC),
                )
            )
        self.n_state = len(state_meta)
        self.state_meta = state_meta
        labels = [meta.legacy_label for meta in state_meta]
        self.state_labels = labels if all(labels) else state_labels_from_metadata(self.state_meta)
        self.state_sides = ["ac"] * ac_n + ["dc"] * dc_n + ["hybrid"] * self.hybrid_n_state
        self.ac_state_labels = self.state_labels[:ac_n]
        self.dc_state_labels = self.state_labels[ac_n : ac_n + dc_n]
        self.ac_state_layout = {"state_labels": self.ac_state_labels, "n_state": ac_n}
        self.dc_state_layout = {"state_labels": self.dc_state_labels, "n_state": dc_n}
        self.dcac_p_dc_state_col = np.asarray(
            [self.dcac_state_start + 3 * idx for idx in range(len(self.dcac_converters))],
            dtype=np.int32,
        )
        self.dcac_p_ac_state_col = self.dcac_p_dc_state_col + 1
        self.dcac_q_ac_state_col = self.dcac_p_dc_state_col + 2
        self._finish_state_layout_arrays()

    def _finish_state_layout_arrays(self) -> None:
        self.ac_reference_nodes = getattr(self._ac_sub_estimator, "references", []) if self._ac_sub_estimator is not None else []
        self.dc_reference_nodes = getattr(self._dc_sub_estimator, "references", []) if self._dc_sub_estimator is not None else []
        self.ac_node_voltage_measurements = (
            getattr(self._ac_sub_estimator, "_node_voltage_measurement_cache", {})
            if self._ac_sub_estimator is not None
            else {}
        )
        self.dc_node_voltage_measurements = (
            getattr(self._dc_sub_estimator, "node_voltage_measurements", {})
            if self._dc_sub_estimator is not None
            else {}
        )
        self.ac_theta_state_col = (
            np.asarray(getattr(self._ac_sub_estimator, "angle_col", []), dtype=np.int32)
            if self._ac_sub_estimator is not None
            else np.array([], dtype=np.int32)
        )
        self.ac_voltage_state_col = (
            np.asarray(getattr(self._ac_sub_estimator, "voltage_col", []), dtype=np.int32)
            if self._ac_sub_estimator is not None
            else np.array([], dtype=np.int32)
        )
        if self._dc_sub_estimator is not None:
            dc_voltage_col = np.asarray(getattr(self._dc_sub_estimator, "voltage_col", []), dtype=np.int32)
            self.dc_voltage_state_col = np.where(dc_voltage_col >= 0, dc_voltage_col + self.dc_state_start, -1).astype(
                np.int32,
                copy=False,
            )
        else:
            self.dc_voltage_state_col = np.array([], dtype=np.int32)
        voltage_col_parts = []
        if self.ac_voltage_state_col.size:
            voltage_col_parts.append(self.ac_voltage_state_col[self.ac_voltage_state_col >= 0].astype(np.int32, copy=False))
        if self.dc_voltage_state_col.size:
            voltage_col_parts.append(self.dc_voltage_state_col[self.dc_voltage_state_col >= 0].astype(np.int32, copy=False))
        self.voltage_cols = np.concatenate(voltage_col_parts) if voltage_col_parts else np.array([], dtype=np.int32)
        self._build_hybrid_converter_array_cache()
        self._partition_state_variables()
        self._link_converter_endpoint_nodes()

    def _build_hybrid_converter_array_cache(self) -> None:
        self._ensure_hybrid_converter_node_cache()
        self._dcac_dc_v_col_by_pos = self._dc_voltage_cols_for_nodes(self._dcac_dc_node)
        self._dcac_ac_v_col_by_pos = self._ac_voltage_cols_for_nodes(self._dcac_ac_node)
        self._dcac_dc_v_default_by_pos = self._dc_voltage_defaults_for_nodes(self._dcac_dc_node)
        self._dcac_ac_v_default_by_pos = self._ac_voltage_defaults_for_nodes(self._dcac_ac_node)
        flat_parts = []
        nonflat_parts = []
        if self._dcac_count:
            dcac_flat = np.zeros(3 * self._dcac_count, dtype=np.float64)
            # Terminal powers are always positive into the converter. A
            # lossless seed therefore has P_DC == -P_AC.
            p_ac_set = np.asarray(
                [float(getattr(conv, "p_ac_set", 0.0) or 0.0) for conv in self.dcac_converters],
                dtype=np.float64,
            )
            p_dc_set = np.asarray(
                [float(getattr(conv, "p_dc_set", 0.0) or 0.0) for conv in self.dcac_converters],
                dtype=np.float64,
            )
            dc_p_control = np.asarray(
                [str(getattr(conv, "dc_control_type", "NONE")).upper() == "P" for conv in self.dcac_converters],
                dtype=bool,
            )
            dcac_flat[0::3] = np.where(dc_p_control, p_dc_set, -p_ac_set)
            dcac_flat[1::3] = np.where(dc_p_control, -p_dc_set, p_ac_set)
            dcac_flat[2::3] = np.asarray(
                [float(getattr(conv, "q_ac_set", 0.0) or 0.0) for conv in self.dcac_converters],
                dtype=np.float64,
            )
            dcac_nonflat = np.empty(3 * self._dcac_count, dtype=np.float64)
            dcac_nonflat[0::3] = np.asarray(
                [float(getattr(conv, "dc_p", 0.0) or 0.0) for conv in self.dcac_converters],
                dtype=np.float64,
            )
            dcac_nonflat[1::3] = np.asarray(
                [float(getattr(conv, "ac_p", 0.0) or 0.0) for conv in self.dcac_converters],
                dtype=np.float64,
            )
            dcac_nonflat[2::3] = np.asarray(
                [float(getattr(conv, "ac_q", 0.0) or 0.0) for conv in self.dcac_converters],
                dtype=np.float64,
            )
            flat_parts.append(dcac_flat)
            nonflat_parts.append(dcac_nonflat)
        self._hybrid_seed_flat_base = (
            np.concatenate(flat_parts).astype(np.float64, copy=False)
            if flat_parts
            else np.zeros(self.hybrid_n_state, dtype=np.float64)
        )
        self._hybrid_seed_nonflat_base = (
            np.concatenate(nonflat_parts).astype(np.float64, copy=False)
            if nonflat_parts
            else np.zeros(self.hybrid_n_state, dtype=np.float64)
        )

    def _refresh_active_measurement_state_layout(self) -> None:
        self._invalidate_measurement_activity_summary()
        active_view = build_active_measurement_view(
            self.measurements,
            table_builder=_measurement_table_from_measurements,
            materialize_measurements=False,
        )
        partitions = partition_measurements_by_code(
            active_view.measurements,
            self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
            table_builder=_measurement_table_from_measurements,
            rows_by_device_type_code=active_view.rows_by_device_type_code,
            as_view=True,
            sides=("ac", "dc", "hybrid"),
        )

        self.active_measurements = active_view.measurements
        self._active_measurement_source_count = len(self.measurements)
        self._active_measurement_source_table = active_view.source_table
        self.ac_meas_rows = partitions.rows["ac"].astype(np.int32, copy=False)
        self.dc_meas_rows = partitions.rows["dc"].astype(np.int32, copy=False)
        self.hybrid_meas_rows = partitions.rows["hybrid"].astype(np.int32, copy=False)
        self.ac_meas = partitions.measurements["ac"]
        self.dc_meas = partitions.measurements["dc"]
        self.hybrid_meas = partitions.measurements["hybrid"]
        self._active_ac_hybrid_rows = self.ac_meas_rows.copy()
        self._active_dc_hybrid_rows = self.dc_meas_rows.copy()
        self._active_ac_sub_measurements = self.ac_meas
        self._active_dc_sub_measurements = self.dc_meas
        self._active_ac_sub_rows = np.arange(len(self.ac_meas), dtype=np.int32)
        self._active_dc_sub_rows = np.arange(len(self.dc_meas), dtype=np.int32)
        self._ac_sub_to_hybrid_cols = np.arange(getattr(self, "ac_n_state", 0), dtype=np.int32)
        self._dc_sub_to_hybrid_cols = (
            self.dc_state_start + np.arange(getattr(self, "dc_n_state", 0), dtype=np.int32)
        )
        self._active_ac_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_dc_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        if self._active_ac_hybrid_rows.size:
            self._active_ac_delegated_row_mask[self._active_ac_hybrid_rows] = True
        if self._active_dc_hybrid_rows.size:
            self._active_dc_delegated_row_mask[self._active_dc_hybrid_rows] = True
        self._jacobian_static_skip = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_measurement_blocks = {
            "ac": self._MeasurementSideBlock(
                self.ac_meas_rows,
                self.ac_meas,
                self._sub_measurement_plan_tables(self._ac_sub_estimator, self.ac_meas),
            ),
            "dc": self._MeasurementSideBlock(
                self.dc_meas_rows,
                self.dc_meas,
                self._sub_measurement_plan_tables(self._dc_sub_estimator, self.dc_meas),
            ),
            "hybrid": self._MeasurementSideBlock(self.hybrid_meas_rows, self.hybrid_meas),
        }
        self._active_hybrid_measurement_plan = self._build_hybrid_measurement_plan(
            self._active_measurement_blocks["hybrid"],
        )
        self.active_z = active_view.z
        self.active_weight = active_view.weight
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        self._active_normal_pattern = None
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._sub_jacobian_stamp_cache = {}
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        self._invalidate_real_voltage_seed_cache()
        self._rebuild_initial_state_cache()

    @staticmethod
    def _preserve_incremental_appended_string_fields(
        table: MeasurementTable,
        base_count: int,
        appended_table: MeasurementTable,
    ) -> None:
        total_count = int(table.idx.size)
        base_count = int(base_count)
        appended_count = int(appended_table.idx.size)
        if base_count < 0 or base_count + appended_count != total_count:
            return
        for field_name in ("name", "device_type", "device_name", "meas_type"):
            values = np.asarray(getattr(table, field_name), dtype=object)
            if values.size == total_count:
                continue
            appended_values = np.asarray(getattr(appended_table, field_name), dtype=object)
            if appended_values.size != appended_count:
                continue
            merged = np.empty(total_count, dtype=object)
            merged[:base_count] = ""
            merged[base_count:] = appended_values
            setattr(table, field_name, merged)

    def _incremental_update_active_measurement_state_layout(
        self,
        appended_measurements: Sequence[Measurement],
    ) -> bool:
        if not appended_measurements:
            return True
        if not hasattr(self, "active_measurements"):
            return False
        if any((not meas.valid) or float(meas.weight) <= 0.0 for meas in appended_measurements):
            return False
        master_table = getattr(self.measurements, "table", None)
        active_table = getattr(self.active_measurements, "table", None)
        if master_table is None or active_table is None:
            return False
        appended_count = len(appended_measurements)
        source_row_start = len(self.measurements) - appended_count
        if source_row_start < 0:
            return False
        if len(active_table.idx) != len(self.active_measurements):
            return False
        master_size = int(master_table.idx.size)
        if master_size == len(self.measurements):
            appended_rows = np.arange(source_row_start, master_size, dtype=np.int64)
            appended_table = measurement_table_take(master_table, appended_rows)
            base_table = measurement_table_take(master_table, np.arange(source_row_start, dtype=np.int64))
        elif master_size == source_row_start:
            appended_table = _measurement_table_from_measurements(appended_measurements)
            base_table = master_table
        else:
            return False

        appended_list = MeasurementList(
            list(appended_measurements),
            appended_table,
            normalized=getattr(self.measurements, "normalized", False),
        )
        self.measurements.table = (
            master_table
            if master_size == len(self.measurements)
            else concat_measurement_tables(base_table, appended_list.table)
        )

        active_start = len(self.active_measurements)
        active_view = append_active_measurement_view(
            build_active_measurement_view(
                self.active_measurements,
                table_builder=_measurement_table_from_measurements,
                materialize_measurements=False,
            ),
            appended_list,
            source_row_start=source_row_start,
            table_builder=_measurement_table_from_measurements,
            materialize_measurements=False,
        )
        self._preserve_incremental_appended_string_fields(active_view.table, active_start, appended_list.table)
        self.active_measurements = active_view.measurements
        self._active_measurement_source_count = len(self.measurements)
        self._active_measurement_source_table = self.measurements.table
        partitions = extend_measurement_partitions(
            self._active_measurement_blocks_as_partitions(),
            list(appended_list),
            self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
            row_offset=active_start,
            table_builder=_measurement_table_from_measurements,
            sides=("ac", "dc", "hybrid"),
        )
        self.ac_meas_rows = np.asarray(partitions.rows["ac"], dtype=np.int32)
        self.dc_meas_rows = np.asarray(partitions.rows["dc"], dtype=np.int32)
        self.hybrid_meas_rows = np.asarray(partitions.rows["hybrid"], dtype=np.int32)
        self.ac_meas = partitions.measurements["ac"]
        self.dc_meas = partitions.measurements["dc"]
        self.hybrid_meas = partitions.measurements["hybrid"]
        self._active_ac_hybrid_rows = self.ac_meas_rows.copy()
        self._active_dc_hybrid_rows = self.dc_meas_rows.copy()
        self._active_ac_sub_measurements = self.ac_meas
        self._active_dc_sub_measurements = self.dc_meas
        self._active_ac_sub_rows = np.arange(len(self.ac_meas), dtype=np.int32)
        self._active_dc_sub_rows = np.arange(len(self.dc_meas), dtype=np.int32)
        self._active_ac_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_dc_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        if self._active_ac_hybrid_rows.size:
            self._active_ac_delegated_row_mask[self._active_ac_hybrid_rows] = True
        if self._active_dc_hybrid_rows.size:
            self._active_dc_delegated_row_mask[self._active_dc_hybrid_rows] = True
        self._jacobian_static_skip = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_measurement_blocks = {
            "ac": self._MeasurementSideBlock(
                self.ac_meas_rows,
                self.ac_meas,
                self._sub_measurement_plan_tables(self._ac_sub_estimator, self.ac_meas),
            ),
            "dc": self._MeasurementSideBlock(
                self.dc_meas_rows,
                self.dc_meas,
                self._sub_measurement_plan_tables(self._dc_sub_estimator, self.dc_meas),
            ),
            "hybrid": self._MeasurementSideBlock(self.hybrid_meas_rows, self.hybrid_meas),
        }
        self._active_hybrid_measurement_plan = self._build_hybrid_measurement_plan(
            self._active_measurement_blocks["hybrid"],
        )
        self.active_z = np.asarray(self.active_measurements.table.value, dtype=np.float64)
        self.active_weight = np.asarray(self.active_measurements.table.weight, dtype=np.float64)
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        self._active_normal_pattern = None
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._sub_jacobian_stamp_cache = {}
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        if self._has_real_voltage_seed_measurement(appended_measurements):
            self._invalidate_real_voltage_seed_cache()
            self._invalidate_initial_state_cache()
        self._invalidate_measurement_activity_summary()
        return True

    def _active_measurement_blocks_as_partitions(self):
        return MeasurementPartitions(
            measurements={"ac": self.ac_meas, "dc": self.dc_meas, "hybrid": self.hybrid_meas},
            rows={
                "ac": np.asarray(self.ac_meas_rows, dtype=np.int64),
                "dc": np.asarray(self.dc_meas_rows, dtype=np.int64),
                "hybrid": np.asarray(self.hybrid_meas_rows, dtype=np.int64),
            },
        )

    @staticmethod
    def _shrink_partition_rows(rows: Sequence[int], removed_pos: int) -> Tuple[np.ndarray, np.ndarray]:
        row_array = np.asarray(rows, dtype=np.int64)
        keep = row_array != int(removed_pos)
        shrunk = row_array[keep]
        shrunk = shrunk - (shrunk > int(removed_pos))
        return shrunk.astype(np.int32, copy=False), keep

    def _shrink_active_measurement_state_layout(self, removed_pos: int) -> MeasurementList:
        active_table_before = getattr(self.active_measurements, "table", None)
        removed_measurement = (
            measurement_from_table_row(active_table_before, int(removed_pos))
            if active_table_before is not None and int(active_table_before.idx.size) > int(removed_pos)
            else None
        )
        keep_rows = np.concatenate(
            (
                np.arange(int(removed_pos), dtype=np.int64),
                np.arange(int(removed_pos) + 1, len(self.active_measurements), dtype=np.int64),
            )
        )
        active_table = getattr(self.active_measurements, "table", None)
        if active_table is None:
            raise RuntimeError("Hybrid SE active measurement shrink requires an active MeasurementTable")
        self.active_measurements = MeasurementTableView(
            measurement_table_take(active_table, keep_rows),
            normalized=getattr(self.active_measurements, "normalized", False),
        )
        source_measurements = getattr(self, "measurements", self.active_measurements)
        self._active_measurement_source_count = len(source_measurements)
        self._active_measurement_source_table = getattr(source_measurements, "table", None)
        if not hasattr(self, "ac_meas_rows"):
            if removed_measurement is not None and self._has_real_voltage_seed_measurement([removed_measurement]):
                self._invalidate_real_voltage_seed_cache()
                self._invalidate_initial_state_cache()
            return self.active_measurements
        self.ac_meas_rows, ac_keep = self._shrink_partition_rows(self.ac_meas_rows, removed_pos)
        self.dc_meas_rows, dc_keep = self._shrink_partition_rows(self.dc_meas_rows, removed_pos)
        self.hybrid_meas_rows, hybrid_keep = self._shrink_partition_rows(self.hybrid_meas_rows, removed_pos)
        self.ac_meas = take_measurement_view(self.ac_meas, np.flatnonzero(ac_keep).astype(np.int64, copy=False))
        self.dc_meas = take_measurement_view(self.dc_meas, np.flatnonzero(dc_keep).astype(np.int64, copy=False))
        self.hybrid_meas = take_measurement_view(
            self.hybrid_meas,
            np.flatnonzero(hybrid_keep).astype(np.int64, copy=False),
        )
        self._active_ac_hybrid_rows = self.ac_meas_rows.copy()
        self._active_dc_hybrid_rows = self.dc_meas_rows.copy()
        self._active_ac_sub_measurements = self.ac_meas
        self._active_dc_sub_measurements = self.dc_meas
        self._active_ac_sub_rows = np.arange(len(self.ac_meas), dtype=np.int32)
        self._active_dc_sub_rows = np.arange(len(self.dc_meas), dtype=np.int32)
        self._active_ac_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_dc_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        if self._active_ac_hybrid_rows.size:
            self._active_ac_delegated_row_mask[self._active_ac_hybrid_rows] = True
        if self._active_dc_hybrid_rows.size:
            self._active_dc_delegated_row_mask[self._active_dc_hybrid_rows] = True
        self._jacobian_static_skip = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_measurement_blocks = {
            "ac": self._MeasurementSideBlock(
                self.ac_meas_rows,
                self.ac_meas,
                self._sub_measurement_plan_tables(self._ac_sub_estimator, self.ac_meas),
            ),
            "dc": self._MeasurementSideBlock(
                self.dc_meas_rows,
                self.dc_meas,
                self._sub_measurement_plan_tables(self._dc_sub_estimator, self.dc_meas),
            ),
            "hybrid": self._MeasurementSideBlock(self.hybrid_meas_rows, self.hybrid_meas),
        }
        self._active_hybrid_measurement_plan = self._build_hybrid_measurement_plan(
            self._active_measurement_blocks["hybrid"],
        )
        active_table = self.active_measurements.table
        self.active_z = np.asarray(active_table.value, dtype=np.float64)
        self.active_weight = np.asarray(active_table.weight, dtype=np.float64)
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        if not hasattr(self, "n_state"):
            return self.active_measurements
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._active_normal_pattern = None
        self._sub_jacobian_stamp_cache = {}
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        if removed_measurement is not None and self._has_real_voltage_seed_measurement([removed_measurement]):
            self._invalidate_real_voltage_seed_cache()
            self._invalidate_initial_state_cache()
        self._invalidate_measurement_activity_summary()
        return self.active_measurements

    @staticmethod
    def _empty_hybrid_code_bucket() -> "HybridStateEstimator._HybridCodeBucket":
        return HybridStateEstimator._HybridCodeBucket(
            rows=np.array([], dtype=np.int32),
            hx_p=np.array([], dtype=np.int32),
            hx_q=np.array([], dtype=np.int32),
            jcol_p=np.array([], dtype=np.int32),
            jcol_q=np.array([], dtype=np.int32),
            v_default=np.array([], dtype=np.float64),
            v_use_x=np.array([], dtype=bool),
            v_x_index=np.array([], dtype=np.int32),
            v_jcol=np.array([], dtype=np.int32),
            v_any_x=False,
        )

    def _empty_hybrid_measurement_plan(self) -> "HybridStateEstimator._HybridMeasurementPlan":
        empty_i32 = np.array([], dtype=np.int32)
        empty_i8 = np.array([], dtype=np.int8)
        empty_f64 = np.array([], dtype=np.float64)
        buckets = [self._empty_hybrid_code_bucket() for _ in range(7)]
        return self._HybridMeasurementPlan(
            dcac_rows=empty_i32,
            dcac_codes=empty_i8,
            dcac_pos=empty_i32,
            dcac_dc_v_col=empty_i32,
            dcac_ac_v_col=empty_i32,
            dcac_dc_v_default=empty_f64,
            dcac_ac_v_default=empty_f64,
            dcac_p_dc=buckets[0],
            dcac_p_ac=buckets[1],
            dcac_q_ac=buckets[2],
            dcac_v_dc=buckets[3],
            dcac_i_dc=buckets[4],
            dcac_v_ac=buckets[5],
            dcac_i_ac=buckets[6],
        )

    def _build_hybrid_measurement_plan(
        self,
        block: "HybridStateEstimator._MeasurementSideBlock",
    ) -> "HybridStateEstimator._HybridMeasurementPlan":
        if len(block.measurements) == 0:
            return self._empty_hybrid_measurement_plan()
        dcac_code = int(DEVICE_TYPE_DCACConverter)
        meas_kind_code_by_type_code = {
            dcac_code: self._measurement_kind_code_lookup(self._DCAC_MEASUREMENT_CODE),
        }
        plan_table = build_measurement_plan_table(
            block.measurements,
            device_pos_by_type_code={},
            meas_kind_by_type_code={
                dcac_code: self._DCAC_MEASUREMENT_CODE,
            },
            meas_kind_code_by_type_code=meas_kind_code_by_type_code,
            require_index_arrays=True,
            table_builder=_measurement_table_from_measurements,
        )

        dcac_mask = plan_table.handled & (plan_table.device_type_code == dcac_code)
        dcac_local_rows = plan_table.row[dcac_mask].astype(np.int64, copy=False)
        dcac_pos = plan_table.device_pos[dcac_mask].astype(np.int32, copy=False)
        dcac_rows = block.rows[dcac_local_rows].astype(np.int32, copy=False)
        dcac_codes = plan_table.meas_kind[dcac_mask].astype(np.int8, copy=False)
        dcac_dc_v_col = self._dcac_dc_v_col_by_pos[dcac_pos] if dcac_pos.size else np.array([], dtype=np.int32)
        dcac_ac_v_col = self._dcac_ac_v_col_by_pos[dcac_pos] if dcac_pos.size else np.array([], dtype=np.int32)
        dcac_dc_v_default = (
            self._dcac_dc_v_default_by_pos[dcac_pos] if dcac_pos.size else np.array([], dtype=np.float64)
        )
        dcac_ac_v_default = (
            self._dcac_ac_v_default_by_pos[dcac_pos] if dcac_pos.size else np.array([], dtype=np.float64)
        )

        # Per-code buckets pre-slice all per-iteration indexing arrays so that
        # evaluate/jacobian can run without recomputing `codes == k` masks or
        # filtering invalid voltage columns. v_col-related fields default to
        # empty for codes that don't read a voltage state.
        dc_state_start = int(self.dc_state_start)

        def _empty_bucket() -> "HybridStateEstimator._HybridCodeBucket":
            return self._HybridCodeBucket(
                rows=np.array([], dtype=np.int32),
                hx_p=np.array([], dtype=np.int32),
                hx_q=np.array([], dtype=np.int32),
                jcol_p=np.array([], dtype=np.int32),
                jcol_q=np.array([], dtype=np.int32),
                v_default=np.array([], dtype=np.float64),
                v_use_x=np.array([], dtype=bool),
                v_x_index=np.array([], dtype=np.int32),
                v_jcol=np.array([], dtype=np.int32),
                v_any_x=False,
            )

        def _voltage_fields(
            cols_subset: np.ndarray,
            defaults_subset: np.ndarray,
            offset: int,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
            """Pack voltage-gather info: (default, use_x, x_index, jcol, any_x)."""
            if cols_subset.size == 0:
                return (
                    np.array([], dtype=np.float64),
                    np.array([], dtype=bool),
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32),
                    False,
                )
            cols_int = cols_subset.astype(np.int32, copy=False)
            use_x = cols_int >= 0
            any_x = bool(use_x.any())
            x_index = (cols_int[use_x] - int(offset)).astype(np.int32, copy=False) if any_x else np.array([], dtype=np.int32)
            return (
                defaults_subset.astype(np.float64, copy=False),
                use_x,
                x_index,
                cols_int,
                any_x,
            )

        def _build_dcac_linear_power_bucket(code: int, hx_offset: int, state_col_arr: np.ndarray) -> "HybridStateEstimator._HybridCodeBucket":
            """For DCAC codes 1/2/3: identity observation of a hybrid power state."""
            if dcac_pos.size == 0:
                return _empty_bucket()
            mask = dcac_codes == code
            if not mask.any():
                return _empty_bucket()
            pos_k = dcac_pos[mask]
            rows_k = dcac_rows[mask]
            hx_p = (3 * pos_k.astype(np.int32, copy=False) + hx_offset).astype(np.int32, copy=False)
            jcol_p = state_col_arr[pos_k].astype(np.int32, copy=False)
            return self._HybridCodeBucket(
                rows=rows_k,
                hx_p=hx_p,
                hx_q=np.array([], dtype=np.int32),
                jcol_p=jcol_p,
                jcol_q=np.array([], dtype=np.int32),
                v_default=np.array([], dtype=np.float64),
                v_use_x=np.array([], dtype=bool),
                v_x_index=np.array([], dtype=np.int32),
                v_jcol=np.array([], dtype=np.int32),
                v_any_x=False,
            )

        def _build_voltage_only_bucket(
            codes_arr: np.ndarray,
            rows_arr: np.ndarray,
            v_col_full: np.ndarray,
            v_default_full: np.ndarray,
            code: int,
            offset: int,
        ) -> "HybridStateEstimator._HybridCodeBucket":
            """Build a direct DCAC terminal-voltage observation bucket."""
            if codes_arr.size == 0:
                return _empty_bucket()
            mask = codes_arr == code
            if not mask.any():
                return _empty_bucket()
            rows_k = rows_arr[mask]
            v_default, v_use_x, v_x_index, v_jcol, v_any_x = _voltage_fields(
                v_col_full[mask], v_default_full[mask], offset,
            )
            return self._HybridCodeBucket(
                rows=rows_k,
                hx_p=np.array([], dtype=np.int32),
                hx_q=np.array([], dtype=np.int32),
                jcol_p=np.array([], dtype=np.int32),
                jcol_q=np.array([], dtype=np.int32),
                v_default=v_default,
                v_use_x=v_use_x,
                v_x_index=v_x_index,
                v_jcol=v_jcol,
                v_any_x=v_any_x,
            )

        def _build_dcac_i_dc_bucket() -> "HybridStateEstimator._HybridCodeBucket":
            """DCAC code 5: i_dc = p_dc / v_dc."""
            if dcac_pos.size == 0:
                return _empty_bucket()
            mask = dcac_codes == 5
            if not mask.any():
                return _empty_bucket()
            pos_k = dcac_pos[mask]
            rows_k = dcac_rows[mask]
            hx_p = (3 * pos_k.astype(np.int32, copy=False)).astype(np.int32, copy=False)
            jcol_p = self.dcac_p_dc_state_col[pos_k].astype(np.int32, copy=False)
            v_default, v_use_x, v_x_index, v_jcol, v_any_x = _voltage_fields(
                dcac_dc_v_col[mask], dcac_dc_v_default[mask], dc_state_start,
            )
            return self._HybridCodeBucket(
                rows=rows_k, hx_p=hx_p, hx_q=np.array([], dtype=np.int32),
                jcol_p=jcol_p, jcol_q=np.array([], dtype=np.int32),
                v_default=v_default, v_use_x=v_use_x, v_x_index=v_x_index,
                v_jcol=v_jcol, v_any_x=v_any_x,
            )

        def _build_dcac_i_ac_bucket() -> "HybridStateEstimator._HybridCodeBucket":
            """DCAC code 7: i_ac = hypot(p_ac, q_ac) / v_ac."""
            if dcac_pos.size == 0:
                return _empty_bucket()
            mask = dcac_codes == 7
            if not mask.any():
                return _empty_bucket()
            pos_k = dcac_pos[mask]
            rows_k = dcac_rows[mask]
            hx_p = (3 * pos_k.astype(np.int32, copy=False) + 1).astype(np.int32, copy=False)
            hx_q = (3 * pos_k.astype(np.int32, copy=False) + 2).astype(np.int32, copy=False)
            jcol_p = self.dcac_p_ac_state_col[pos_k].astype(np.int32, copy=False)
            jcol_q = self.dcac_q_ac_state_col[pos_k].astype(np.int32, copy=False)
            v_default, v_use_x, v_x_index, v_jcol, v_any_x = _voltage_fields(
                dcac_ac_v_col[mask], dcac_ac_v_default[mask], 0,
            )
            return self._HybridCodeBucket(
                rows=rows_k, hx_p=hx_p, hx_q=hx_q,
                jcol_p=jcol_p, jcol_q=jcol_q,
                v_default=v_default, v_use_x=v_use_x, v_x_index=v_x_index,
                v_jcol=v_jcol, v_any_x=v_any_x,
            )

        dcac_p_dc_bucket = _build_dcac_linear_power_bucket(1, 0, self.dcac_p_dc_state_col)
        dcac_p_ac_bucket = _build_dcac_linear_power_bucket(2, 1, self.dcac_p_ac_state_col)
        dcac_q_ac_bucket = _build_dcac_linear_power_bucket(3, 2, self.dcac_q_ac_state_col)
        dcac_v_dc_bucket = _build_voltage_only_bucket(
            dcac_codes, dcac_rows, dcac_dc_v_col, dcac_dc_v_default, 4, dc_state_start,
        )
        dcac_i_dc_bucket = _build_dcac_i_dc_bucket()
        dcac_v_ac_bucket = _build_voltage_only_bucket(
            dcac_codes, dcac_rows, dcac_ac_v_col, dcac_ac_v_default, 6, 0,
        )
        dcac_i_ac_bucket = _build_dcac_i_ac_bucket()

        return self._HybridMeasurementPlan(
            dcac_rows=dcac_rows,
            dcac_codes=dcac_codes,
            dcac_pos=dcac_pos,
            dcac_dc_v_col=dcac_dc_v_col,
            dcac_ac_v_col=dcac_ac_v_col,
            dcac_dc_v_default=dcac_dc_v_default,
            dcac_ac_v_default=dcac_ac_v_default,
            dcac_p_dc=dcac_p_dc_bucket,
            dcac_p_ac=dcac_p_ac_bucket,
            dcac_q_ac=dcac_q_ac_bucket,
            dcac_v_dc=dcac_v_dc_bucket,
            dcac_i_dc=dcac_i_dc_bucket,
            dcac_v_ac=dcac_v_ac_bucket,
            dcac_i_ac=dcac_i_ac_bucket,
        )

    def _hybrid_measurement_plan_for(
        self,
        measurements: Sequence[Measurement],
        blocks: Optional[Dict[str, "HybridStateEstimator._MeasurementSideBlock"]] = None,
    ) -> "HybridStateEstimator._HybridMeasurementPlan":
        if measurements is self.active_measurements:
            return self._active_hybrid_measurement_plan
        side_blocks = self._measurement_blocks_for(measurements) if blocks is None else blocks
        return self._build_hybrid_measurement_plan(side_blocks["hybrid"])

    def _build_hybrid_seed_measurement_plan(
        self,
        measurements: Sequence[Measurement],
    ) -> "HybridStateEstimator._HybridSeedMeasurementPlan":
        dcac_code = int(DEVICE_TYPE_DCACConverter)
        kind_lookup = np.full(max(MEAS_TYPE_P_DC, MEAS_TYPE_P_AC, MEAS_TYPE_Q_AC) + 1, -1, dtype=np.int16)
        kind_lookup[MEAS_TYPE_P_DC] = 0
        kind_lookup[MEAS_TYPE_P_AC] = 1
        kind_lookup[MEAS_TYPE_Q_AC] = 2
        plan_table = build_measurement_plan_table(
            measurements,
            device_pos_by_type_code={},
            meas_kind_by_type_code={},
            meas_kind_code_by_type_code={dcac_code: kind_lookup},
            require_index_arrays=True,
            table_builder=_measurement_table_from_measurements,
        )
        active_mask = (
            plan_table.handled
            & np.asarray(plan_table.table.valid, dtype=bool)
            & (np.asarray(plan_table.table.weight, dtype=np.float64) > 0.0)
        )
        dcac_mask = active_mask & (plan_table.device_type_code == dcac_code)

        dcac_rows = plan_table.row[dcac_mask].astype(np.int64, copy=False)
        dcac_state_col = (
            3 * plan_table.device_pos[dcac_mask].astype(np.int64, copy=False)
            + plan_table.meas_kind[dcac_mask].astype(np.int64, copy=False)
        )
        return self._HybridSeedMeasurementPlan(
            measurement_row=dcac_rows,
            state_col=dcac_state_col,
        )

    def _hybrid_converter_measurement_specs(
        self,
        source: str = "flow",
    ) -> List["_HybridConverterMeasurementSpec"]:
        specs: List[HybridStateEstimator._HybridConverterMeasurementSpec] = []
        def spec(device_type_code: int, device_name: str, meas_type_code: int, value: float, device_pos: int) -> "HybridStateEstimator._HybridConverterMeasurementSpec":
            return self._measurement_spec(device_type_code, int(device_pos), meas_type_code, value, str(device_name))

        for pos, conv in enumerate(self.dcac_converters):
            p_dc = float(getattr(conv, "dc_p", 0.0) or 0.0) if source == "pseudo" else float(getattr(conv, "i_p", 0.0) or 0.0)
            p_ac = float(getattr(conv, "ac_p", 0.0) or 0.0) if source == "pseudo" else float(getattr(conv, "j_p", 0.0) or 0.0)
            q_ac = float(getattr(conv, "ac_q", 0.0) or 0.0) if source == "pseudo" else float(getattr(conv, "j_q", 0.0) or 0.0)
            specs.extend(
                (
                    spec(DEVICE_TYPE_DCACConverter, conv.name, MEAS_TYPE_P_DC, p_dc, pos),
                    spec(DEVICE_TYPE_DCACConverter, conv.name, MEAS_TYPE_P_AC, p_ac, pos),
                    spec(DEVICE_TYPE_DCACConverter, conv.name, MEAS_TYPE_Q_AC, q_ac, pos),
                    spec(
                        DEVICE_TYPE_DCACConverter,
                        conv.name,
                        MEAS_TYPE_V_DC,
                        float(getattr(getattr(conv, "dc_node_obj", None), "voltage", 1.0) or 1.0),
                        pos,
                    ),
                    spec(
                        DEVICE_TYPE_DCACConverter,
                        conv.name,
                        MEAS_TYPE_V_AC,
                        float(getattr(getattr(conv, "ac_node_obj", None), "voltage", 1.0) or 1.0),
                        pos,
                    ),
                )
            )
        return specs

    def state_layout(self) -> Dict[str, object]:
        return {
            "state_labels": self.state_labels,
            "state_meta": self.state_meta,
            "state_sides": self.state_sides,
            "ac_state_slice": self.ac_state_slice,
            "dc_state_slice": self.dc_state_slice,
            "hybrid_state_slice": self.hybrid_state_slice,
            "n_state": self.n_state,
        }

    def _hybrid_seed_vector(self, flat: Optional[bool] = None) -> np.ndarray:
        flat = self.flat_start if flat is None else bool(flat)
        base = self._hybrid_seed_flat_base if flat else self._hybrid_seed_nonflat_base
        x = base.copy()
        self._seed_hybrid_power_states_from_measurements(x)
        return x

    def _seed_hybrid_power_states_from_measurements(self, x: np.ndarray) -> None:
        measurements = getattr(self, "hybrid_meas", None) or self.measurements
        if measurements is getattr(self, "hybrid_meas", None) and hasattr(self, "_active_hybrid_measurement_plan"):
            plan = self._active_hybrid_measurement_plan
            dcac_mask = (plan.dcac_codes >= 1) & (plan.dcac_codes <= 3)
            if not np.any(dcac_mask):
                return
            state_col = (
                3 * plan.dcac_pos[dcac_mask].astype(np.int64, copy=False)
                + plan.dcac_codes[dcac_mask].astype(np.int64, copy=False)
                - 1
            )
            rows = plan.dcac_rows[dcac_mask].astype(np.int64, copy=False)
            x[state_col] = self.active_z[rows]
            return
        seed_plan = self._hybrid_seed_measurement_plan_for(measurements)
        if seed_plan.measurement_row.size == 0:
            return
        table = getattr(measurements, "table", None)
        if table is None or len(table.idx) != len(measurements):
            table = _measurement_table_from_measurements(measurements)
            try:
                measurements.table = table
            except AttributeError:
                pass
        x[seed_plan.state_col] = np.asarray(table.value, dtype=np.float64)[seed_plan.measurement_row]

    def _hybrid_seed_measurement_plan_for(
        self,
        measurements: Sequence[Measurement],
    ) -> "HybridStateEstimator._HybridSeedMeasurementPlan":
        key = (id(measurements), len(measurements))
        cached = self._hybrid_seed_measurement_plan_cache.get(key)
        if cached is not None:
            return cached
        plan = self._build_hybrid_seed_measurement_plan(measurements)
        self._hybrid_seed_measurement_plan_cache[key] = plan
        return plan

    # Keep the opt-in diagnostic format identical to the standalone estimators.
    _write_sparse_triplet_file = staticmethod(ACStateEstimator._write_sparse_triplet_file)
    _sparse_triplets_from_matrix = staticmethod(ACStateEstimator._sparse_triplets_from_matrix)
    _sparse_matrix_stores_lower_triangle = staticmethod(ACStateEstimator._sparse_matrix_stores_lower_triangle)
    _dump_iteration_matrices = ACStateEstimator._dump_iteration_matrices

    def initial_state(self, flat: Optional[bool] = None) -> np.ndarray:
        delegate = self._delegate()
        if delegate is not None:
            return delegate.initial_state(flat=flat)
        flat = self.flat_start if flat is None else bool(flat)
        if self._initial_state_cache_ready:
            cached = self.flat_state if flat else self.power_flow_state
            if cached.size == getattr(self, "n_state", 0):
                return cached.copy()
        if hasattr(self, "active_measurements") and getattr(self, "n_state", 0) > 0:
            self._rebuild_initial_state_cache(flat=flat)
            cached = self.flat_state if flat else self.power_flow_state
            if cached.size == getattr(self, "n_state", 0):
                return cached.copy()
        return self._build_initial_state(flat=flat)

    def _build_initial_state(self, flat: bool) -> np.ndarray:
        parts = []
        if self._ac_sub_estimator is not None:
            parts.append(self._ac_sub_estimator.initial_state(flat=flat))
        if self._dc_sub_estimator is not None:
            parts.append(self._dc_sub_estimator.initial_state(flat=flat))
        parts.append(self._hybrid_seed_vector(flat=flat))
        x = np.concatenate(parts) if parts else np.array([], dtype=np.float64)
        if not flat:
            self._seed_state_from_real_voltage_measurements(x)
        return x

    def _rebuild_initial_state_cache(self, flat: Optional[bool] = None) -> None:
        self._initial_state_cache_ready = False
        flat = self.flat_start if flat is None else bool(flat)
        if flat:
            self.flat_state = self._build_initial_state(flat=True)
        else:
            self.power_flow_state = self._build_initial_state(flat=False)
        self._initial_state_cache_ready = True

    def _invalidate_initial_state_cache(self) -> None:
        self._initial_state_cache_ready = False
        self._hybrid_seed_measurement_plan_cache = {}

    def _seed_state_from_real_voltage_measurements(self, x: np.ndarray) -> None:
        """Seed coupled voltage states from converter-side real voltage rows."""
        measurements = getattr(self, "hybrid_meas", None)
        if measurements is None:
            measurements = getattr(self, "active_measurements", self.measurements)
        cols, values = self._real_voltage_seed_arrays(measurements)
        if cols.size:
            x[cols] = values

    def _invalidate_real_voltage_seed_cache(self) -> None:
        self._real_voltage_seed_cache = {}

    @staticmethod
    def _has_real_voltage_seed_measurement(measurements: Sequence[Measurement]) -> bool:
        voltage_codes = {
            MEAS_TYPE_V,
            MEAS_TYPE_V_FROM,
            MEAS_TYPE_V_TO,
            MEAS_TYPE_V_THIRD,
            MEAS_TYPE_V_GEN,
            MEAS_TYPE_V_LOAD,
            MEAS_TYPE_V_DC,
            MEAS_TYPE_V_AC,
        }
        for meas in measurements:
            if (
                bool(getattr(meas, "valid", True))
                and float(getattr(meas, "weight", 0.0)) > 0.0
                and not is_pseudo_measurement(meas)
                and int(getattr(meas, "meas_type_code", 0)) in voltage_codes
            ):
                return True
        return False

    def _real_voltage_seed_arrays(self, measurements: Sequence[Measurement]) -> Tuple[np.ndarray, np.ndarray]:
        key = (id(measurements), len(measurements))
        cached = self._real_voltage_seed_cache.get(key)
        if cached is not None:
            return cached
        if measurements is getattr(self, "hybrid_meas", None) and hasattr(self, "_active_hybrid_measurement_plan"):
            result = self._real_voltage_seed_arrays_from_active_hybrid_plan()
            self._real_voltage_seed_cache[key] = result
            return result
        best: Dict[int, Tuple[float, float]] = {}
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            status_code = measurement_table_status_code(table)
            active_real = (
                np.asarray(table.valid, dtype=bool)
                & (np.asarray(table.weight, dtype=np.float64) > 0.0)
                & (status_code != MEAS_STATUS_PSEUDO)
            )
            meas_type_code = np.asarray(getattr(table, "meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
            device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
            device_pos = self._hybrid_measurement_device_pos_array(table)
            if meas_type_code.size == table.idx.size and device_pos.size == table.idx.size:
                voltage_codes = np.asarray(
                    [MEAS_TYPE_V, MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO, MEAS_TYPE_V_THIRD, MEAS_TYPE_V_GEN, MEAS_TYPE_V_LOAD, MEAS_TYPE_V_DC, MEAS_TYPE_V_AC],
                    dtype=np.int16,
                )
                rows = np.flatnonzero(active_real & np.isin(meas_type_code, voltage_codes) & (device_pos >= 0))
                for row in rows:
                    side, node_idx = self._voltage_measurement_side_node_idx_from_code(
                        int(device_type_code[int(row)]),
                        int(device_pos[int(row)]),
                        int(meas_type_code[int(row)]),
                    )
                    if side == "ac" and node_idx is not None:
                        col = self._ac_voltage_col_for_node(int(node_idx))
                    elif side == "dc" and node_idx is not None:
                        col = self._dc_voltage_col_for_node(int(node_idx))
                    else:
                        continue
                    if col < 0:
                        continue
                    weight = float(table.weight[int(row)])
                    current = best.get(col)
                    if current is None or weight > current[0]:
                        best[col] = (weight, float(table.value[int(row)]))
        else:
            self._warn_missing_device_pos("real_voltage_seed_no_table", DEVICE_TYPE_ACNode)
        if best:
            cols = np.fromiter(best.keys(), dtype=np.int32, count=len(best))
            values = np.fromiter((item[1] for item in best.values()), dtype=np.float64, count=len(best))
        else:
            cols = np.array([], dtype=np.int32)
            values = np.array([], dtype=np.float64)
        self._real_voltage_seed_cache[key] = (cols, values)
        return cols, values

    def _real_voltage_seed_arrays_from_active_hybrid_plan(self) -> Tuple[np.ndarray, np.ndarray]:
        plan = self._active_hybrid_measurement_plan
        active_table = getattr(self.active_measurements, "table", None)
        status_code = (
            measurement_table_status_code(active_table)
            if active_table is not None and len(active_table.idx) == len(self.active_measurements)
            else None
        )
        row_chunks = []
        col_chunks = []

        def append_rows(mask: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> None:
            if status_code is not None and np.any(mask):
                selected_rows = rows[mask]
                mask_indices = np.flatnonzero(mask)
                real_mask = status_code[selected_rows.astype(np.int64, copy=False)] != MEAS_STATUS_PSEUDO
                selected_rows = selected_rows[real_mask]
                selected_cols = cols[mask_indices[real_mask]]
            else:
                selected_rows = rows[mask]
                selected_cols = cols[mask]
            valid_col = selected_cols >= 0
            if np.any(valid_col):
                row_chunks.append(selected_rows[valid_col].astype(np.int64, copy=False))
                col_chunks.append(selected_cols[valid_col].astype(np.int32, copy=False))

        append_rows(plan.dcac_codes == self._DCAC_MEASUREMENT_CODE[MEAS_TYPE_V_DC], plan.dcac_rows, plan.dcac_dc_v_col)
        append_rows(plan.dcac_codes == self._DCAC_MEASUREMENT_CODE[MEAS_TYPE_V_AC], plan.dcac_rows, plan.dcac_ac_v_col)
        if not row_chunks:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

        rows = np.concatenate(row_chunks).astype(np.int64, copy=False)
        cols = np.concatenate(col_chunks).astype(np.int32, copy=False)
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        cols = cols[order]
        values = self.active_z[rows]
        weights = self.active_weight[rows]
        best: Dict[int, Tuple[float, float]] = {}
        for col, weight, value in zip(cols, weights, values):
            col_int = int(col)
            current = best.get(col_int)
            if current is None or float(weight) > current[0]:
                best[col_int] = (float(weight), float(value))
        if not best:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
        return (
            np.fromiter(best.keys(), dtype=np.int32, count=len(best)),
            np.fromiter((item[1] for item in best.values()), dtype=np.float64, count=len(best)),
        )

    def _split_state(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        ac_x = x[: self.ac_n_state]
        dc_x = x[self.dc_state_start : self.dc_state_start + self.dc_n_state]
        hybrid_x = x[self.hybrid_state_start :]
        return ac_x, dc_x, hybrid_x

    def _expand_state(self, x: np.ndarray) -> np.ndarray:
        delegate = self._delegate()
        if isinstance(delegate, DCStateEstimator):
            compact = np.asarray(x, dtype=np.float64)
            voltage, _, _, _ = delegate._unpack_state(compact)
            return np.concatenate((voltage, compact[delegate.n_voltage :]))
        return np.asarray(x, dtype=np.float64).copy()

    def _expand_state_mapped_only(self, x: np.ndarray) -> np.ndarray:
        return self._expand_state(x)

    def _ac_sub_state_from_hybrid(self, x: np.ndarray) -> Optional[np.ndarray]:
        if self._ac_sub_estimator is None:
            return None
        return np.asarray(x, dtype=np.float64)[: self.ac_n_state].copy()

    def _dc_sub_state_from_hybrid(self, x: np.ndarray) -> Optional[np.ndarray]:
        if self._dc_sub_estimator is None:
            return None
        start = self.dc_state_start
        return np.asarray(x, dtype=np.float64)[start : start + self.dc_n_state].copy()

    @staticmethod
    def _append_sparse_rows_unchecked(H, rows, cols, values) -> None:
        if rows is None or cols is None or values is None:
            return
        if hasattr(H, "rows") and hasattr(H, "cols") and hasattr(H, "data"):
            rr = np.asarray(rows, dtype=np.int32)
            cc = np.asarray(cols, dtype=np.int32)
            vv = np.asarray(values, dtype=np.float64)
            mask = (cc >= 0) & (vv != 0.0)
            if rr.size == 1 and cc.size:
                H.rows.extend([int(rr[0])] * int(np.count_nonzero(mask)))
                H.cols.extend(cc[mask].tolist())
                H.data.extend(vv[mask].tolist())
            elif rr.size == cc.size:
                H.rows.extend(rr[mask].tolist())
                H.cols.extend(cc[mask].tolist())
                H.data.extend(vv[mask].tolist())

    def _converter_coupled_ac_node_names(self) -> set:
        names = set()
        for conv in self.dcac_converters:
            node = getattr(conv, "ac_node_obj", None)
            if node is not None:
                names.add(node.name)
        return names

    def _converter_coupled_ac_device_positions(self) -> np.ndarray:
        node_ids = []
        dcac_ac_node = np.asarray(getattr(self, "_dcac_ac_node", np.asarray([], dtype=np.int64)), dtype=np.int64)
        if dcac_ac_node.size:
            node_ids.extend(dcac_ac_node.tolist())
        else:
            node_ids.extend(int(conv.ac_node) for conv in self.dcac_converters)
        if not node_ids or self._ac_sub_estimator is None:
            return np.asarray([], dtype=np.int64)
        ac = self._ac_sub_estimator
        ids = np.asarray(getattr(ac, "_ac_node_id_lookup_ids", np.asarray([], dtype=np.int64)), dtype=np.int64)
        pos = np.asarray(getattr(ac, "_ac_node_id_lookup_pos", np.asarray([], dtype=np.int64)), dtype=np.int64)
        if ids.size == 0 or pos.size != ids.size:
            return np.asarray([], dtype=np.int64)
        query = np.unique(np.asarray(node_ids, dtype=np.int64))
        loc = np.searchsorted(ids, query)
        valid = (loc >= 0) & (loc < ids.size)
        if np.any(valid):
            valid_locs = loc[valid].astype(np.intp, copy=False)
            valid[valid] = ids[valid_locs] == query[valid]
        if not np.any(valid):
            return np.asarray([], dtype=np.int64)
        solver_pos = pos[loc[valid].astype(np.intp, copy=False)].astype(np.int64, copy=False)
        plan_pos = np.asarray(getattr(ac, "_ac_node_plan_pos", np.asarray([], dtype=np.int64)), dtype=np.int64)
        node_count = int(getattr(ac, "n_nodes", 0) or 0)
        if plan_pos.size and node_count > 0:
            solver_to_plan = np.full(node_count, -1, dtype=np.int64)
            valid_plan = (plan_pos >= 0) & (plan_pos < node_count)
            if np.any(valid_plan):
                solver_to_plan[plan_pos[valid_plan].astype(np.intp, copy=False)] = np.nonzero(valid_plan)[0]
                in_range = (solver_pos >= 0) & (solver_pos < solver_to_plan.size)
                mapped = np.full(solver_pos.size, -1, dtype=np.int64)
                mapped[in_range] = solver_to_plan[solver_pos[in_range].astype(np.intp, copy=False)]
                mapped = mapped[mapped >= 0]
                if mapped.size:
                    return np.unique(mapped)
        return np.unique(solver_pos)

    def _real_voltage_observation_nodes(self, side: str) -> Dict[int, float]:
        """Return side nodes covered by real usable voltage measurements on any device."""
        cache_name = f"_{side}_real_voltage_observation_node_cache"
        cache = getattr(self, cache_name, None)
        if cache is not None:
            return cache
        if side == "ac":
            sub_cache = getattr(self._ac_sub_estimator, "_real_voltage_observation_node_cache", None)
        else:
            sub_cache = getattr(self._dc_sub_estimator, "_real_voltage_observation_node_cache", None)
        best: Dict[int, Tuple[float, float]] = {}
        if isinstance(sub_cache, dict):
            for node_idx, value in sub_cache.items():
                best[int(node_idx)] = (float("inf"), float(value))
        table = getattr(self.measurements, "table", None)
        if table is not None and len(table.idx) == len(self.measurements):
            status_code = measurement_table_status_code(table)
            active_real = (
                np.asarray(table.valid, dtype=bool)
                & (np.asarray(table.weight, dtype=np.float64) > 0.0)
                & (status_code != MEAS_STATUS_PSEUDO)
            )
            device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
            meas_type_code = np.asarray(getattr(table, "meas_type_code", np.asarray([], dtype=np.int16)), dtype=np.int16)
            device_pos = self._hybrid_measurement_device_pos_array(table)
            if meas_type_code.size == table.idx.size and device_pos.size == table.idx.size:
                voltage_codes = np.asarray(
                    [MEAS_TYPE_V, MEAS_TYPE_V_FROM, MEAS_TYPE_V_TO, MEAS_TYPE_V_THIRD, MEAS_TYPE_V_GEN, MEAS_TYPE_V_LOAD, MEAS_TYPE_V_DC, MEAS_TYPE_V_AC],
                    dtype=np.int16,
                )
                voltage_rows = np.flatnonzero(active_real & np.isin(meas_type_code, voltage_codes) & (device_pos >= 0))
                if voltage_rows.size:
                    row_dev = device_type_code[voltage_rows]
                    row_meas = meas_type_code[voltage_rows]
                    row_pos = device_pos[voltage_rows]
                    row_weights = np.asarray(table.weight, dtype=np.float64)[voltage_rows]
                    row_values = np.asarray(table.value, dtype=np.float64)[voltage_rows]

                    # 向量化处理最常见的 ACNode/DCNode 电压量测，仅对非节点电压量测走逐行回退
                    handled = np.zeros(voltage_rows.size, dtype=bool)
                    if side == "ac" and self._ac_sub_estimator is not None:
                        fast_mask = (row_dev == DEVICE_TYPE_ACNode) & (row_meas == MEAS_TYPE_V)
                        if np.any(fast_mask):
                            handled |= fast_mask
                            plan_pos = np.asarray(getattr(self._ac_sub_estimator, "_ac_node_plan_pos", ()), dtype=np.int64)
                            node_ids = np.asarray(getattr(self._ac_sub_estimator, "_ac_node_ids", ()), dtype=np.int64)
                            f_pos = row_pos[fast_mask]
                            in_plan = f_pos < plan_pos.size
                            if np.any(in_plan):
                                solver_pos = plan_pos[f_pos[in_plan].astype(np.intp, copy=False)]
                                in_node = (solver_pos >= 0) & (solver_pos < node_ids.size)
                                if np.any(in_node):
                                    n_ids = node_ids[solver_pos[in_node].astype(np.intp, copy=False)]
                                    f_w = row_weights[fast_mask][in_plan][in_node]
                                    f_v = row_values[fast_mask][in_plan][in_node]
                                    if not best and n_ids.size > 0:
                                        # 绝大多数场景每个节点仅 1 条节点电压量测：直接检测无重复后零排序快写
                                        if n_ids.size == 1 or np.unique(n_ids).size == n_ids.size:
                                            best = {
                                                nid: (w_val, v_val)
                                                for nid, w_val, v_val in zip(n_ids.tolist(), f_w.tolist(), f_v.tolist())
                                            }
                                        else:
                                            order = np.argsort(f_w, kind="mergesort")
                                            for nid, w_val, v_val in zip(n_ids[order].tolist(), f_w[order].tolist(), f_v[order].tolist()):
                                                best[nid] = (w_val, v_val)
                                    else:
                                        order = np.argsort(f_w, kind="mergesort")
                                        for nid, w_val, v_val in zip(n_ids[order].tolist(), f_w[order].tolist(), f_v[order].tolist()):
                                            cur = best.get(nid)
                                            if cur is None or w_val > cur[0]:
                                                best[nid] = (w_val, v_val)
                        # 向量化处理 DCACConverter 的 V_AC 电压量测，跳过肯定属于 DC 侧的 DCNode/V_DC
                        conv_ac_mask = (row_dev == DEVICE_TYPE_DCACConverter) & (row_meas == MEAS_TYPE_V_AC)
                        if np.any(conv_ac_mask):
                            handled |= conv_ac_mask
                            conv_ac_nodes = np.asarray(getattr(self, "_dcac_ac_node", ()), dtype=np.int64)
                            c_pos = row_pos[conv_ac_mask]
                            in_conv = c_pos < conv_ac_nodes.size
                            if np.any(in_conv):
                                n_ids = conv_ac_nodes[c_pos[in_conv].astype(np.intp, copy=False)]
                                c_w = row_weights[conv_ac_mask][in_conv]
                                c_v = row_values[conv_ac_mask][in_conv]
                                for nid, w_val, v_val in zip(n_ids.tolist(), c_w.tolist(), c_v.tolist()):
                                    cur = best.get(nid)
                                    if cur is None or w_val > cur[0]:
                                        best[nid] = (w_val, v_val)
                        handled |= (row_dev == DEVICE_TYPE_DCNode) | ((row_dev == DEVICE_TYPE_DCACConverter) & (row_meas == MEAS_TYPE_V_DC))
                    elif side == "dc" and self._dc_sub_estimator is not None:
                        fast_mask = (row_dev == DEVICE_TYPE_DCNode) & (row_meas == MEAS_TYPE_V)
                        if np.any(fast_mask):
                            handled |= fast_mask
                            node_ids = np.asarray(getattr(self._dc_sub_estimator, "_raw_node_idx_alive", ()), dtype=np.int64)
                            f_pos = row_pos[fast_mask]
                            in_node = f_pos < node_ids.size
                            if np.any(in_node):
                                n_ids = node_ids[f_pos[in_node].astype(np.intp, copy=False)]
                                f_w = row_weights[fast_mask][in_node]
                                f_v = row_values[fast_mask][in_node]
                                order = np.argsort(f_w, kind="mergesort")
                                for nid, w_val, v_val in zip(n_ids[order].tolist(), f_w[order].tolist(), f_v[order].tolist()):
                                    cur = best.get(nid)
                                    if cur is None or w_val > cur[0]:
                                        best[nid] = (w_val, v_val)
                        # 向量化处理 DCACConverter 的 V_DC 电压量测，跳过肯定属于 AC 侧的 ACNode/V_AC
                        conv_dc_mask = (row_dev == DEVICE_TYPE_DCACConverter) & (row_meas == MEAS_TYPE_V_DC)
                        if np.any(conv_dc_mask):
                            handled |= conv_dc_mask
                            conv_dc_nodes = np.asarray(getattr(self, "_dcac_dc_node", ()), dtype=np.int64)
                            c_pos = row_pos[conv_dc_mask]
                            in_conv = c_pos < conv_dc_nodes.size
                            if np.any(in_conv):
                                n_ids = conv_dc_nodes[c_pos[in_conv].astype(np.intp, copy=False)]
                                c_w = row_weights[conv_dc_mask][in_conv]
                                c_v = row_values[conv_dc_mask][in_conv]
                                for nid, w_val, v_val in zip(n_ids.tolist(), c_w.tolist(), c_v.tolist()):
                                    cur = best.get(nid)
                                    if cur is None or w_val > cur[0]:
                                        best[nid] = (w_val, v_val)
                        handled |= (row_dev == DEVICE_TYPE_ACNode) | ((row_dev == DEVICE_TYPE_DCACConverter) & (row_meas == MEAS_TYPE_V_AC))

                    rem_idx = np.flatnonzero(~handled)
                    for idx in rem_idx.tolist():
                        side_value, node_idx = self._voltage_measurement_side_node_idx_from_code(
                            int(row_dev[idx]),
                            int(row_pos[idx]),
                            int(row_meas[idx]),
                        )
                        if side_value != side or node_idx is None:
                            continue
                        w_val = float(row_weights[idx])
                        cur = best.get(int(node_idx))
                        if cur is None or w_val > cur[0]:
                            best[int(node_idx)] = (w_val, float(row_values[idx]))
        else:
            self._warn_missing_device_pos("real_voltage_observation_no_table", DEVICE_TYPE_ACNode if side == "ac" else DEVICE_TYPE_DCNode)
        cache = {node_idx: value for node_idx, (_weight, value) in best.items()}
        setattr(self, cache_name, cache)
        return cache

    def _real_voltage_observation_value_for_side_node(
        self,
        side: str,
        node_idx: Optional[int],
    ) -> Optional[float]:
        """Return a real side voltage on the node or its compressed zero-tie component."""
        if node_idx is None:
            return None
        observed = self._real_voltage_observation_nodes(side)
        if node_idx in observed:
            return float(observed[node_idx])
        estimator = self._ac_sub_estimator if side == "ac" else self._dc_sub_estimator
        node_pos = getattr(estimator, "node_pos", {})
        pos = node_pos.get(node_idx)
        component_by_pos = getattr(estimator, "zero_tie_component_by_pos", None)
        components = getattr(estimator, "zero_tie_components", None)
        nodes = getattr(estimator, "nodes", None)
        if pos is None or component_by_pos is None or not components or nodes is None:
            return None
        component = components[int(component_by_pos[int(pos)])]
        for member_pos in component:
            member_idx = nodes[int(member_pos)].idx
            if member_idx in observed:
                return float(observed[member_idx])
        return None

    def _voltage_measurement_side_node_idx_from_code(
        self,
        device_type_code: int,
        device_pos: int,
        meas_type_code: int,
    ) -> Tuple[Optional[str], Optional[int]]:
        pos = int(device_pos)
        if pos < 0:
            return None, None
        code = int(device_type_code)
        meas_code = int(meas_type_code)
        if code == DEVICE_TYPE_DCACConverter:
            if meas_code == MEAS_TYPE_V_AC:
                nodes = np.asarray(getattr(self, "_dcac_ac_node", ()), dtype=np.int64)
                return ("ac", int(nodes[pos])) if pos < nodes.size else (None, None)
            if meas_code == MEAS_TYPE_V_DC:
                nodes = np.asarray(getattr(self, "_dcac_dc_node", ()), dtype=np.int64)
                return ("dc", int(nodes[pos])) if pos < nodes.size else (None, None)
            return None, None

        side = self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE.get(code)
        estimator = self._ac_sub_estimator if side == "ac" else self._dc_sub_estimator if side == "dc" else None
        if estimator is None:
            return None, None
        node_idx = estimator.voltage_measurement_node_idx(code, pos, meas_code)
        if node_idx is None:
            return None, None
        return side, int(node_idx)

    def voltage_measurement_node_idx(
        self,
        device_type_code: int,
        device_pos: int,
        meas_type_code: int,
    ) -> Optional[int]:
        """返回电压量测对应的节点编号（与 AC/DC SE 接口保持一致）。"""
        delegate = self._delegate()
        if delegate is not None:
            return delegate.voltage_measurement_node_idx(device_type_code, device_pos, meas_type_code)
        _, node_idx = self._voltage_measurement_side_node_idx_from_code(
            device_type_code,
            device_pos,
            meas_type_code,
        )
        return node_idx

    def _voltage_pseudo_is_covered_by_code(
        self,
        device_type_code: int,
        device_pos: int,
        meas_type_code: int,
    ) -> bool:
        side, node_idx = self._voltage_measurement_side_node_idx_from_code(
            device_type_code,
            device_pos,
            meas_type_code,
        )
        if side is None or node_idx is None:
            return False
        return self._real_voltage_observation_value_for_side_node(side, node_idx) is not None

    def _active_measurement_keys(self) -> set:
        return set(self._measurement_activity_summary().active_keys)

    def _invalidate_measurement_activity_summary(self, *, preserve_real_voltage_cache: bool = False) -> None:
        if hasattr(self, "_measurement_activity_summary_cache"):
            delattr(self, "_measurement_activity_summary_cache")
        if hasattr(self, "_sub_measurement_global_summary_attrs"):
            self._sub_measurement_global_summary_attrs = None
        if hasattr(self, "_observability_pseudo_candidate_cache"):
            delattr(self, "_observability_pseudo_candidate_cache")
        if not preserve_real_voltage_cache:
            if hasattr(self, "_ac_real_voltage_observation_node_cache"):
                delattr(self, "_ac_real_voltage_observation_node_cache")
            if hasattr(self, "_dc_real_voltage_observation_node_cache"):
                delattr(self, "_dc_real_voltage_observation_node_cache")

    def _measurement_activity_summary(self) -> "_MeasurementActivitySummary":
        cache = getattr(self, "_measurement_activity_summary_cache", None)
        if cache is not None:
            return cache
        table = getattr(self.measurements, "table", None)
        if table is None or len(table.idx) != len(self.measurements):
            table = _measurement_table_from_measurements(self.measurements)
            try:
                self.measurements.table = table
            except AttributeError:
                pass
        active_table = self._current_active_measurement_table(table)
        if active_table is not None:
            max_idx = int(np.asarray(table.idx, dtype=np.int64).max()) if table.idx.size else 0
            active_mask = np.asarray(active_table.valid, dtype=bool) & (
                np.asarray(active_table.weight, dtype=np.float64) > 0.0
            )
            measured_devices, active_keys = self._activity_key_sets_from_table(active_table, active_mask)
            cache = self._MeasurementActivitySummary(
                max_idx=max_idx,
                measured_devices=measured_devices,
                active_keys=active_keys,
            )
            self._measurement_activity_summary_cache = cache
            return cache
        idx = np.asarray(table.idx, dtype=np.int64)
        valid = np.asarray(table.valid, dtype=bool)
        weight = np.asarray(table.weight, dtype=np.float64)
        active_mask = valid & (weight > 0.0)
        max_idx = int(idx.max()) if idx.size else 0
        measured_devices, active_keys = self._activity_key_sets_from_table(table, active_mask)
        cache = self._MeasurementActivitySummary(max_idx=max_idx, measured_devices=measured_devices, active_keys=active_keys)
        self._measurement_activity_summary_cache = cache
        return cache

    def materialize_measurements(
        self,
        measurements: Optional[object] = None,
    ) -> MeasurementList:
        """Materialize array-backed rows for diagnostics and compatibility only."""
        source = self.active_measurements if measurements is None else measurements
        table = getattr(source, "table", None)
        if table is None or int(table.idx.size) != len(source):
            return MeasurementList(list(source), normalized=getattr(source, "normalized", False))
        codes = np.asarray(table.device_type_code, dtype=np.int16)
        positions = self._hybrid_measurement_device_pos_array(table)
        meas_codes = np.asarray(
            getattr(table, "meas_type_code", np.zeros(table.idx.size, dtype=np.int16)),
            dtype=np.int16,
        )
        name_arrays = self._measurement_name_arrays_by_type_code(np.unique(codes))
        status = measurement_table_status_code(table)
        rows = []
        for row in range(int(table.idx.size)):
            measurement = measurement_from_table_row(table, row)
            type_code = int(codes[row])
            meas_code = int(meas_codes[row]) if meas_codes.size == table.idx.size else 0
            device_pos = int(positions[row]) if positions.size == table.idx.size else -1
            measurement.device_type = DEVICE_TYPE_NAMES.get(type_code, measurement.device_type)
            measurement.meas_type = MEAS_TYPE_NAMES.get(meas_code, measurement.meas_type)
            names = np.asarray(name_arrays.get(type_code, ()), dtype=object)
            if 0 <= device_pos < names.size:
                measurement.device_name = str(names[device_pos])
            if not measurement.name or measurement.name.startswith("m"):
                prefix = "pseudo" if int(status[row]) == MEAS_STATUS_PSEUDO else "measurement"
                measurement.name = f"{prefix}_{measurement.meas_type.lower()}_{measurement.device_name}"
            rows.append(measurement)
        return MeasurementList(rows, table=table, normalized=getattr(source, "normalized", False))

    def _current_active_measurement_table(self, source_table):
        active_measurements = getattr(self, "active_measurements", None)
        active_table = getattr(active_measurements, "table", None)
        if active_measurements is None or active_table is None:
            return None
        if len(active_table.idx) != len(active_measurements):
            return None
        if getattr(self, "_active_measurement_source_count", None) != len(self.measurements):
            return None
        if getattr(self, "_active_measurement_source_table", None) is not source_table:
            return None
        return active_table

    def _next_measurement_idx(self) -> int:
        return int(self._measurement_activity_summary().max_idx) + 1

    @staticmethod
    def _converter_meas_type_code_for_state_meta(meta: StateMeta) -> int:
        mapping = {
            "dcac_p_dc": MEAS_TYPE_P_DC,
            "dcac_p_ac": MEAS_TYPE_P_AC,
            "dcac_q_ac": MEAS_TYPE_Q_AC,
        }
        return int(mapping.get(meta.kind, 0))

    def _targeted_converter_specs_for_state(
        self,
        meta: StateMeta,
        source: str = "flow",
    ) -> List["_HybridConverterMeasurementSpec"]:
        meas_type_code = self._converter_meas_type_code_for_state_meta(meta)
        if meas_type_code <= 0:
            return []
        device_type_code = int(meta.device_type_code)
        device_pos = int(meta.device_pos)
        return [
            spec
            for spec in self._hybrid_converter_measurement_specs(source=source)
            if int(spec.device_type_code) == device_type_code
            and int(spec.device_pos) == device_pos
            and int(spec.meas_type_code) == meas_type_code
        ]

    def _hybrid_direct_state_coverage_ok(self) -> bool:
        hybrid_n = int(getattr(self, "hybrid_n_state", 0))
        if hybrid_n <= 0:
            return True
        plan = getattr(self, "_active_hybrid_measurement_plan", None)
        if plan is None:
            return False
        start = int(getattr(self, "hybrid_state_start", 0))
        stop = start + hybrid_n
        covered = np.zeros(hybrid_n, dtype=bool)
        for bucket in (
            plan.dcac_p_dc,
            plan.dcac_p_ac,
            plan.dcac_q_ac,
        ):
            cols = np.asarray(bucket.jcol_p, dtype=np.int64)
            valid = (cols >= start) & (cols < stop)
            if np.any(valid):
                covered[(cols[valid] - start).astype(np.intp, copy=False)] = True
        return bool(np.all(covered))

    def _sub_block_observable_fast_certificate(
        self,
        estimator,
        block: "HybridStateEstimator._MeasurementSideBlock",
    ) -> bool:
        sub_n_state = int(getattr(estimator, "n_state", 0) or 0) if estimator is not None else 0
        if sub_n_state <= 0:
            return True
        if block is None or block.plan_tables is None or len(block.measurements) < sub_n_state:
            return False
        cached = getattr(estimator, "_initial_observability_cache", None)
        if (
            cached is not None
            and bool(getattr(cached, "observable", False))
            and int(getattr(cached, "state_count", -1)) == sub_n_state
        ):
            return True
        fast_builder = getattr(estimator, "_fast_active_observability_certificate", None)
        if callable(fast_builder):
            fast_result = fast_builder()
            if (
                fast_result is not None
                and bool(getattr(fast_result, "observable", False))
                and int(getattr(fast_result, "state_count", -1)) == sub_n_state
            ):
                return True
        if isinstance(estimator, DCStateEstimator) and self._dc_sub_direct_observability_certificate(estimator, block):
            return True
        try:
            result = estimator.observability_analysis(
                estimator.initial_state(),
                self._sub_measurement_runtime_input(block),
            )
        except Exception:
            return False
        return bool(getattr(result, "observable", False)) and int(getattr(result, "rank", -1)) == sub_n_state

    @staticmethod
    def _dc_sub_direct_observability_certificate(
        estimator: DCStateEstimator,
        block: "HybridStateEstimator._MeasurementSideBlock",
    ) -> bool:
        n_state = int(getattr(estimator, "n_state", 0) or 0)
        if n_state <= 0:
            return True
        if block is None or block.plan_tables is None or len(block.measurements) < n_state:
            return False
        try:
            plan = estimator._vector_plans_for_measurement_plan_tables(block.plan_tables)["simple"]
        except Exception:
            return False
        handled = np.asarray(plan.get("handled_mask", ()), dtype=bool)
        if handled.size != len(block.measurements) or not np.all(handled):
            return False
        covered = np.zeros(n_state, dtype=bool)

        def mark(cols, mask=None) -> None:
            values = np.asarray(cols, dtype=np.int64)
            valid = (values >= 0) & (values < n_state)
            if mask is not None:
                mask_values = np.asarray(mask, dtype=bool)
                if mask_values.size != values.size:
                    return
                valid &= mask_values
            if np.any(valid):
                covered[values[valid].astype(np.intp, copy=False)] = True

        def mask_at(masks, code: int, size: int) -> np.ndarray:
            if isinstance(masks, (tuple, list)) and 0 <= int(code) < len(masks):
                values = np.asarray(masks[int(code)], dtype=bool)
                if values.size == int(size):
                    return values
            return np.zeros(int(size), dtype=bool)

        mark(plan.get("node_col", ()))
        branch_size = np.asarray(plan.get("branch_i_col", ())).size
        branch_kind_masks = plan.get("branch_kind_masks", ())
        mark(plan.get("branch_i_col", ()), mask_at(branch_kind_masks, MEAS_TYPE_V_FROM, branch_size))
        mark(plan.get("branch_j_col", ()), mask_at(branch_kind_masks, MEAS_TYPE_V_TO, branch_size))
        load_size = np.asarray(plan.get("load_col", ())).size
        mark(plan.get("load_col", ()), mask_at(plan.get("load_kind_masks", ()), MEAS_TYPE_V_LOAD, load_size))
        gen_size = np.asarray(plan.get("gen_col", ())).size
        gen_kind_masks = plan.get("gen_kind_masks", ())
        gen_ctrl_masks = plan.get("gen_ctrl_masks", ())
        mark(plan.get("gen_col", ()), mask_at(gen_kind_masks, MEAS_TYPE_V_GEN, gen_size))
        mark(
            plan.get("gen_p_col", ()),
            mask_at(gen_kind_masks, MEAS_TYPE_P_GEN, gen_size) & mask_at(gen_ctrl_masks, DC_CTRL_V, gen_size),
        )
        switch_size = np.asarray(plan.get("switch_col", ())).size
        switch_kind_masks = plan.get("switch_kind_masks", ())
        switch_i_mask = mask_at(switch_kind_masks, MEAS_TYPE_I_FROM, switch_size) | mask_at(
            switch_kind_masks,
            MEAS_TYPE_I_TO,
            switch_size,
        )
        mark(plan.get("switch_col", ()), switch_i_mask)
        mark(plan.get("switch_i_col", ()), mask_at(switch_kind_masks, MEAS_TYPE_V_FROM, switch_size))
        mark(plan.get("switch_j_col", ()), mask_at(switch_kind_masks, MEAS_TYPE_V_TO, switch_size))
        dcdc_size = np.asarray(plan.get("dcdc_p_col", ())).size
        dcdc_kind_masks = plan.get("dcdc_kind_masks", ())
        mark(plan.get("dcdc_p_col", ()), mask_at(dcdc_kind_masks, MEAS_TYPE_P_FROM, dcdc_size))
        mark(plan.get("dcdc_q_col", ()), mask_at(dcdc_kind_masks, MEAS_TYPE_P_TO, dcdc_size))
        mark(plan.get("dcdc_i_col", ()), mask_at(dcdc_kind_masks, MEAS_TYPE_V_FROM, dcdc_size))
        mark(plan.get("dcdc_j_col", ()), mask_at(dcdc_kind_masks, MEAS_TYPE_V_TO, dcdc_size))
        return bool(np.all(covered))

    def _fast_active_observability_certificate(self) -> Optional[ObservabilityResult]:
        if not hasattr(self, "active_measurements") or not hasattr(self, "_active_measurement_blocks"):
            return None
        n_state = int(getattr(self, "n_state", 0) or 0)
        if n_state <= 0:
            return None
        measurement_count = len(self.active_measurements)
        if measurement_count < n_state:
            return None
        if targeted_redundancy_count(
            n_state,
            getattr(self, "targeted_pseudo_measurement_redundancy_ratio", 0.0),
        ) > 0:
            return None
        if not self._hybrid_direct_state_coverage_ok():
            return None
        blocks = self._active_measurement_blocks
        if not self._sub_block_observable_fast_certificate(self._ac_sub_estimator, blocks.get("ac")):
            return None
        if not self._sub_block_observable_fast_certificate(self._dc_sub_estimator, blocks.get("dc")):
            return None
        return ObservabilityResult(
            observable=True,
            rank=n_state,
            state_count=n_state,
            measurement_count=measurement_count,
            deficiency=0,
            singular_values=np.array([], dtype=np.float64),
            weak_states=[],
        )

    def _add_required_missing_pseudo_measurements(self) -> int:
        """Add all required pseudo rows for missing P/Q/V topology states without a max-count cap."""
        delegate = self._delegate()
        if delegate is not None:
            return 0
        active_keys_raw = self._measurement_activity_summary().active_keys
        existing_sorted = np.fromiter(active_keys_raw, dtype=np.int64, count=len(active_keys_raw))
        if existing_sorted.size > 1:
            existing_sorted.sort()
        # AC 侧、DC 侧与 DCACConverter 换流器设备类型码完全互斥，各自内部已完成保序去重，
        # 无需将数万条 AC/DC 键装箱进 Python set 再跨侧全量排序查重。
        conv_added_keys: set = set()
        allowed_codes_by_device = {
            DEVICE_TYPE_ACGenerator: (MEAS_TYPE_P_GEN, MEAS_TYPE_Q_GEN),
            DEVICE_TYPE_ACLoad: (MEAS_TYPE_P_LOAD, MEAS_TYPE_Q_LOAD),
            DEVICE_TYPE_ACZeroBranch: (MEAS_TYPE_P_FROM, MEAS_TYPE_Q_FROM),
            DEVICE_TYPE_ACBreak: (MEAS_TYPE_P_FROM, MEAS_TYPE_Q_FROM),
            DEVICE_TYPE_DCGenerator: (MEAS_TYPE_P_GEN,),
            DEVICE_TYPE_DCLoad: (MEAS_TYPE_P_LOAD,),
            DEVICE_TYPE_DCZeroBranch: (MEAS_TYPE_P_FROM,),
            DEVICE_TYPE_DCBreak: (MEAS_TYPE_P_FROM,),
            DEVICE_TYPE_DCACConverter: (MEAS_TYPE_P_DC, MEAS_TYPE_P_AC, MEAS_TYPE_Q_AC),
            DEVICE_TYPE_ACNode: (MEAS_TYPE_V,),
            DEVICE_TYPE_DCNode: (MEAS_TYPE_V,),
        }

        dt_chunks: List[np.ndarray] = []
        dp_chunks: List[np.ndarray] = []
        mt_chunks: List[np.ndarray] = []
        val_chunks: List[np.ndarray] = []
        next_idx = self._next_measurement_idx()
        pack_key = self._packed_measurement_key
        pack_key_array = self._packed_measurement_key_array
        voltage_covered = self._voltage_pseudo_is_covered_by_code
        # 预取真实电压观测集合并转为排序数组，供后续全向量化查重
        ac_obs_nodes = self._real_voltage_observation_nodes("ac")
        dc_obs_nodes = self._real_voltage_observation_nodes("dc")
        ac_obs_sorted = np.fromiter(ac_obs_nodes, dtype=np.int64, count=len(ac_obs_nodes))
        if ac_obs_sorted.size > 1:
            ac_obs_sorted.sort()
        dc_obs_sorted = np.fromiter(dc_obs_nodes, dtype=np.int64, count=len(dc_obs_nodes))
        if dc_obs_sorted.size > 1:
            dc_obs_sorted.sort()
        ac_plan_pos = (
            np.asarray(getattr(self._ac_sub_estimator, "_ac_node_plan_pos", ()), dtype=np.int64)
            if self._ac_sub_estimator is not None
            else np.empty(0, dtype=np.int64)
        )
        ac_node_ids = (
            np.asarray(getattr(self._ac_sub_estimator, "_ac_node_ids", ()), dtype=np.int64)
            if self._ac_sub_estimator is not None
            else np.empty(0, dtype=np.int64)
        )
        dc_node_ids = (
            np.asarray(getattr(self._dc_sub_estimator, "_raw_node_idx_alive", ()), dtype=np.int64)
            if self._dc_sub_estimator is not None
            else np.empty(0, dtype=np.int64)
        )
        ac_has_zero_ties = bool(getattr(self._ac_sub_estimator, "_has_zero_impedance_ties", False)) if self._ac_sub_estimator is not None else False
        dc_has_zero_ties = bool(getattr(self._dc_sub_estimator, "_has_zero_impedance_ties", False)) if self._dc_sub_estimator is not None else False

        allowed_device_type_codes = frozenset(allowed_codes_by_device.keys())

        def append_from_side_estimator(side: str, estimator) -> None:
            if estimator is None:
                return
            candidates = estimator._observability_pseudo_candidate_measurements(
                allowed_device_type_codes=allowed_device_type_codes,
                allowed_meas_types_by_device=allowed_codes_by_device,
            )
            table = getattr(candidates, "table", None)
            if table is None:
                raise RuntimeError(f"{side.upper()} observability candidates must expose an array measurement table")
            row_count = int(table.idx.size)
            if row_count == 0:
                return
            dt_arr = np.asarray(table.device_type_code, dtype=np.int16)
            dp_arr = np.asarray(table.device_pos, dtype=np.int64)
            mt_arr = np.asarray(table.meas_type_code, dtype=np.int16)
            val_arr = np.asarray(table.value, dtype=np.float64)
            if not (dt_arr.size == dp_arr.size == mt_arr.size == val_arr.size == row_count):
                raise RuntimeError(f"{side.upper()} observability candidate arrays have inconsistent sizes")
            keep = np.zeros(row_count, dtype=bool)
            for dt_code, mt_codes in allowed_codes_by_device.items():
                dt_mask = dt_arr == dt_code
                if not np.any(dt_mask):
                    continue
                mt_mask = np.zeros(row_count, dtype=bool)
                for mt_code in mt_codes:
                    mt_mask |= mt_arr == mt_code
                keep |= dt_mask & mt_mask
            keep &= (dt_arr > 0) & (mt_arr > 0) & (dp_arr >= 0)
            selected = np.flatnonzero(keep)
            if selected.size == 0:
                return
            sel_dt = dt_arr[selected]
            sel_dp = dp_arr[selected]
            sel_mt = mt_arr[selected]
            sel_val = val_arr[selected]
            cand_keys = pack_key_array(sel_dt, sel_dp, sel_mt)
            valid_mask = cand_keys >= 0
            # 向量化排除已存在的有效量测键（AC/DC 两侧设备类型码完全互斥，无需跨侧查重）
            if existing_sorted.size > 0 and np.any(valid_mask):
                pos = np.searchsorted(existing_sorted, cand_keys)
                in_existing = (pos < existing_sorted.size) & (
                    existing_sorted[np.minimum(pos, existing_sorted.size - 1)] == cand_keys
                )
                valid_mask &= ~in_existing
            if not np.any(valid_mask):
                return
            sel_dt = sel_dt[valid_mask]
            sel_dp = sel_dp[valid_mask]
            sel_mt = sel_mt[valid_mask]
            sel_val = sel_val[valid_mask]
            cand_keys = cand_keys[valid_mask]
            # 向量化过滤已被真实电压量测覆盖的节点电压伪量测
            v_rows = np.flatnonzero(sel_mt == MEAS_TYPE_V)
            if v_rows.size > 0:
                v_keep = np.ones(sel_dt.size, dtype=bool)
                v_dt = sel_dt[v_rows]
                v_dp = sel_dp[v_rows]
                ac_v_local = np.flatnonzero(v_dt == DEVICE_TYPE_ACNode)
                if ac_v_local.size > 0:
                    if not ac_has_zero_ties:
                        if ac_obs_sorted.size > 0:
                            dp_sub = v_dp[ac_v_local]
                            in_range = (dp_sub >= 0) & (dp_sub < ac_plan_pos.size)
                            if np.any(in_range):
                                ir_idx = np.flatnonzero(in_range)
                                sp = ac_plan_pos[dp_sub[ir_idx]]
                                sp_ok = (sp >= 0) & (sp < ac_node_ids.size)
                                if np.any(sp_ok):
                                    nids = ac_node_ids[sp[sp_ok]]
                                    pos_n = np.searchsorted(ac_obs_sorted, nids)
                                    hit_n = (pos_n < ac_obs_sorted.size) & (
                                        ac_obs_sorted[np.minimum(pos_n, ac_obs_sorted.size - 1)] == nids
                                    )
                                    v_keep[v_rows[ac_v_local[ir_idx[sp_ok][hit_n]]]] = False
                    else:
                        for loc in ac_v_local.tolist():
                            if voltage_covered(DEVICE_TYPE_ACNode, int(v_dp[loc]), MEAS_TYPE_V):
                                v_keep[v_rows[loc]] = False
                dc_v_local = np.flatnonzero(v_dt == DEVICE_TYPE_DCNode)
                if dc_v_local.size > 0:
                    if not dc_has_zero_ties:
                        if dc_obs_sorted.size > 0:
                            dp_sub = v_dp[dc_v_local]
                            in_range = (dp_sub >= 0) & (dp_sub < dc_node_ids.size)
                            if np.any(in_range):
                                ir_idx = np.flatnonzero(in_range)
                                nids = dc_node_ids[dp_sub[ir_idx]]
                                pos_n = np.searchsorted(dc_obs_sorted, nids)
                                hit_n = (pos_n < dc_obs_sorted.size) & (
                                    dc_obs_sorted[np.minimum(pos_n, dc_obs_sorted.size - 1)] == nids
                                )
                                v_keep[v_rows[dc_v_local[ir_idx[hit_n]]]] = False
                    else:
                        for loc in dc_v_local.tolist():
                            if voltage_covered(DEVICE_TYPE_DCNode, int(v_dp[loc]), MEAS_TYPE_V):
                                v_keep[v_rows[loc]] = False
                other_v_local = np.flatnonzero((v_dt != DEVICE_TYPE_ACNode) & (v_dt != DEVICE_TYPE_DCNode))
                for loc in other_v_local.tolist():
                    if voltage_covered(int(v_dt[loc]), int(v_dp[loc]), MEAS_TYPE_V):
                        v_keep[v_rows[loc]] = False
                if not np.all(v_keep):
                    sel_dt = sel_dt[v_keep]
                    sel_dp = sel_dp[v_keep]
                    sel_mt = sel_mt[v_keep]
                    sel_val = sel_val[v_keep]
                    cand_keys = cand_keys[v_keep]
            if cand_keys.size == 0:
                return
            # 候选表内保序去重
            if cand_keys.size > 1:
                _, first_idx = np.unique(cand_keys, return_index=True)
                if first_idx.size < cand_keys.size:
                    first_idx.sort()
                    sel_dt = sel_dt[first_idx]
                    sel_dp = sel_dp[first_idx]
                    sel_mt = sel_mt[first_idx]
                    sel_val = sel_val[first_idx]
                    cand_keys = cand_keys[first_idx]
            dt_chunks.append(sel_dt)
            dp_chunks.append(sel_dp)
            mt_chunks.append(sel_mt)
            val_chunks.append(sel_val)

        append_from_side_estimator("ac", self._ac_sub_estimator)
        append_from_side_estimator("dc", self._dc_sub_estimator)

        n_conv = len(self.dcac_converters)
        if n_conv > 0:
            # 向量化构造换流器 (P_DC, P_AC, Q_AC) 缺失伪量测，免除 5.5 万个 spec 对象构建与逐条查重
            c_dp = np.repeat(np.arange(n_conv, dtype=np.int64), 3)
            c_dt = np.full(n_conv * 3, DEVICE_TYPE_DCACConverter, dtype=np.int16)
            c_mt = np.tile(np.asarray([MEAS_TYPE_P_DC, MEAS_TYPE_P_AC, MEAS_TYPE_Q_AC], dtype=np.int16), n_conv)
            c_keys = pack_key_array(c_dt, c_dp, c_mt)
            c_keep = c_keys >= 0
            if existing_sorted.size > 0 and np.any(c_keep):
                c_pos = np.searchsorted(existing_sorted, c_keys)
                c_in = (c_pos < existing_sorted.size) & (
                    existing_sorted[np.minimum(c_pos, existing_sorted.size - 1)] == c_keys
                )
                c_keep &= ~c_in
            missing_conv_idx = np.flatnonzero(c_keep)
            if missing_conv_idx.size > 0:
                sel_dp = c_dp[missing_conv_idx]
                sel_mt = c_mt[missing_conv_idx]
                converters = self.dcac_converters
                sel_val = np.empty(missing_conv_idx.size, dtype=np.float64)
                for k, (p_i, m_i) in enumerate(zip(sel_dp.tolist(), sel_mt.tolist())):
                    conv = converters[p_i]
                    if m_i == MEAS_TYPE_P_DC:
                        sel_val[k] = float(getattr(conv, "dc_p", 0.0) or 0.0)
                    elif m_i == MEAS_TYPE_P_AC:
                        sel_val[k] = float(getattr(conv, "ac_p", 0.0) or 0.0)
                    else:
                        sel_val[k] = float(getattr(conv, "ac_q", 0.0) or 0.0)
                dt_chunks.append(c_dt[missing_conv_idx])
                dp_chunks.append(sel_dp)
                mt_chunks.append(sel_mt)
                val_chunks.append(sel_val)
        if not dt_chunks:
            return 0
        device_type_arr = dt_chunks[0] if len(dt_chunks) == 1 else np.concatenate(dt_chunks)
        device_pos_arr = dp_chunks[0] if len(dp_chunks) == 1 else np.concatenate(dp_chunks)
        meas_type_arr = mt_chunks[0] if len(mt_chunks) == 1 else np.concatenate(mt_chunks)
        values_arr = val_chunks[0] if len(val_chunks) == 1 else np.concatenate(val_chunks)
        count = int(device_type_arr.size)
        if count == 0:
            return 0
        values_arr, weights_arr = normalize_pseudo_measurement_arrays(
            meas_type_arr,
            values_arr.astype(np.float64, copy=False),
            np.full(count, self.pseudo_measurement_weight_pq, dtype=np.float64),
            weight_v=self.pseudo_measurement_weight_v,
            weight_pq=self.pseudo_measurement_weight_pq,
            preserve_explicit_pq_weight=False,
        )
        tail_table = MeasurementTable(
            idx=np.arange(next_idx, next_idx + count, dtype=np.int64),
            name=np.asarray([], dtype=object),
            device_type=np.asarray([], dtype=object),
            device_name=np.asarray([], dtype=object),
            meas_type=np.asarray([], dtype=object),
            weight=weights_arr,
            valid=np.ones(count, dtype=bool),
            value=values_arr,
            device_type_code=device_type_arr.astype(np.int16, copy=False),
            angle_mask=np.zeros(count, dtype=bool),
            status_code=np.full(count, MEAS_STATUS_PSEUDO, dtype=np.int16),
            device_name_id=np.full(count, -1, dtype=np.int64),
            meas_type_code=meas_type_arr.astype(np.int16, copy=False),
            device_pos=device_pos_arr.astype(np.int64, copy=False),
        )
        master_table = getattr(self.measurements, "table", None)
        if master_table is None or int(master_table.idx.size) != len(self.measurements):
            master_table = _measurement_table_from_measurements(self.measurements)
        self.measurement_table = concat_measurement_tables(master_table, tail_table)
        self.measurements = TableBackedMeasurementList(
            self.measurement_table,
            normalized=getattr(self.measurements, "normalized", False),
        )
        self._invalidate_measurement_activity_summary(preserve_real_voltage_cache=True)
        self._refresh_active_measurement_state_layout()
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        return count

    def _add_targeted_observability_pseudo_measurements(self) -> int:
        """Patch remaining hybrid rank deficiencies via weak-direction targeted pseudo measurements."""
        delegate = self._delegate()
        if delegate is not None:
            return int(getattr(delegate, "targeted_observability_pseudo_count", 0))
        fast_observability = self._fast_active_observability_certificate()
        if fast_observability is not None:
            self._initial_observability_cache = fast_observability
            return 0
        observability = self.observability_analysis()
        if observability.observable:
            self._initial_observability_cache = observability
            return 0

        total_added = 0
        # 优先按零空间弱方向分批精准定向补盲（消除辐射叶节点等局部缺秩而不引入数百条多余伪量测）
        for _round in range(12):
            self._observability_pseudo_candidate_cache = None
            candidates = self._observability_pseudo_candidate_measurements()
            if not candidates:
                break
            deficiency = max(1, int(getattr(observability, "deficiency", 1) or 1))
            batch_limit = max(4, min(64, 2 * deficiency))
            picked = self._select_weak_direction_pseudo_candidates(observability, candidates, batch_limit)
            if not picked:
                break
            existing_keys = set(self._active_measurement_keys())
            next_idx = self._next_measurement_idx()
            selected = []
            for candidate in picked:
                device_type_code = int(getattr(candidate, "device_type_code", 0))
                meas_type_code = int(getattr(candidate, "meas_type_code", 0))
                device_pos = int(getattr(candidate, "device_pos", -1))
                if device_type_code <= 0 or meas_type_code <= 0 or device_pos < 0:
                    continue
                key = self._packed_measurement_key(device_type_code, device_pos, meas_type_code)
                if key < 0 or key in existing_keys:
                    continue
                if self._voltage_pseudo_is_covered_by_code(device_type_code, device_pos, meas_type_code):
                    continue
                candidate.idx = next_idx
                next_idx += 1
                candidate.device_type_code = device_type_code
                candidate.meas_type_code = meas_type_code
                candidate.device_pos = device_pos
                mark_measurement_pseudo(candidate)
                selected.append(candidate)
                existing_keys.add(key)
            if not selected:
                break
            measurement_count_before = len(self.measurements)
            self._append_pseudo_measurement_objects(selected)
            appended = self.measurements[measurement_count_before:]
            if not self._incremental_update_active_measurement_state_layout(appended):
                self._refresh_active_measurement_state_layout()
            total_added += len(selected)
            self._observability_matrix_cache = None
            self._observability_pseudo_candidate_cache = None
            observability = self.observability_analysis()
            if observability.observable:
                self._initial_observability_cache = observability
                return total_added

        # 若多轮定向补盲后仍未满秩且尚有剩余候选，全量补齐兜底
        if not observability.observable:
            self._observability_pseudo_candidate_cache = None
            candidates = self._observability_pseudo_candidate_measurements()
            existing_keys = set(self._active_measurement_keys())
            next_idx = self._next_measurement_idx()
            selected = []
            for candidate in candidates:
                device_type_code = int(getattr(candidate, "device_type_code", 0))
                meas_type_code = int(getattr(candidate, "meas_type_code", 0))
                device_pos = int(getattr(candidate, "device_pos", -1))
                if device_type_code <= 0 or meas_type_code <= 0 or device_pos < 0:
                    continue
                key = self._packed_measurement_key(device_type_code, device_pos, meas_type_code)
                if key < 0 or key in existing_keys:
                    continue
                if self._voltage_pseudo_is_covered_by_code(device_type_code, device_pos, meas_type_code):
                    continue
                candidate.idx = next_idx
                next_idx += 1
                candidate.device_type_code = device_type_code
                candidate.meas_type_code = meas_type_code
                candidate.device_pos = device_pos
                mark_measurement_pseudo(candidate)
                selected.append(candidate)
                existing_keys.add(key)
            if selected:
                measurement_count_before = len(self.measurements)
                self._append_pseudo_measurement_objects(selected)
                appended = self.measurements[measurement_count_before:]
                if not self._incremental_update_active_measurement_state_layout(appended):
                    self._refresh_active_measurement_state_layout()
                total_added += len(selected)
                self._observability_matrix_cache = None
                self._observability_pseudo_candidate_cache = None
                observability = self.observability_analysis()
        self._initial_observability_cache = observability
        return total_added

    def _add_weak_direction_observability_pseudo_measurements(
        self,
        observability: ObservabilityResult,
        max_add: int,
        refresh: bool = True,
    ) -> int:
        """Add the candidate pseudo rows that best observe the current weak direction."""
        if max_add <= 0:
            return 0
        candidates = self._observability_pseudo_candidate_measurements()
        if not candidates:
            return 0
        if len(candidates) <= max_add:
            selected = list(candidates[:max_add])
        else:
            selected = self._select_weak_direction_pseudo_candidates(observability, candidates, max_add)
        if not selected:
            return 0
        next_idx = self._next_measurement_idx()
        valid_selected: List[Measurement] = []
        for candidate in selected:
            candidate.device_type_code = int(getattr(candidate, "device_type_code", 0))
            candidate.meas_type_code = int(getattr(candidate, "meas_type_code", 0))
            device_pos = int(getattr(candidate, "device_pos", -1))
            if candidate.device_type_code <= 0 or candidate.meas_type_code <= 0 or device_pos < 0:
                self._warn_missing_device_pos(
                    "weak_direction_candidate",
                    candidate.device_type_code,
                    candidate.device_name,
                    candidate.meas_type_code,
                )
                continue
            candidate.idx = next_idx
            next_idx += 1
            candidate.device_pos = device_pos
            mark_measurement_pseudo(candidate)
            valid_selected.append(candidate)
        if not valid_selected:
            return 0
        measurement_count_before = len(self.measurements)
        self._append_pseudo_measurement_objects(valid_selected)
        if refresh:
            appended = self.measurements[measurement_count_before:]
            if not self._incremental_update_active_measurement_state_layout(appended):
                self._refresh_active_measurement_state_layout()
        return len(valid_selected)

    def _add_redundant_observability_pseudo_measurements(self, max_add: int) -> int:
        observability = self.observability_analysis()
        return self._add_weak_direction_observability_pseudo_measurements(observability, max_add)

    def _observability_pseudo_candidate_measurements(self) -> List[Measurement]:
        """Build low-weight candidate pseudo rows for weak-direction observability repair."""
        cache = getattr(self, "_observability_pseudo_candidate_cache", None)
        if cache is not None:
            return cache
        existing_keys = self._active_measurement_keys()
        candidates: List[Measurement] = []

        def add(
            device_name: str,
            value: float,
            device_type_code: int,
            device_pos: int = -1,
            meas_type_code: int = 0,
        ) -> None:
            device_type_code = int(device_type_code)
            meas_type_code = int(meas_type_code)
            device_pos = int(device_pos)
            if device_type_code <= 0 or meas_type_code <= 0 or device_pos < 0:
                self._warn_missing_device_pos("observability_candidate", device_type_code, device_name, meas_type_code)
                return
            key = self._packed_measurement_key(device_type_code, int(device_pos), meas_type_code)
            device_type = DEVICE_TYPE_NAMES.get(device_type_code, "")
            meas_type = MEAS_TYPE_NAMES.get(meas_type_code, "")
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{device_name}"
            if self._voltage_pseudo_is_covered_by_code(device_type_code, device_pos, meas_type_code):
                return
            if key < 0 or key in existing_keys:
                return
            candidates.append(
                Measurement(
                    0,
                    pseudo_name,
                    device_type,
                    device_name,
                    meas_type,
                    self.pseudo_measurement_weight_pq,
                    True,
                    float(value),
                    MEAS_STATUS_PSEUDO,
                    device_type_code=device_type_code,
                    meas_type_code=meas_type_code,
                    device_pos=device_pos,
                )
            )
            existing_keys.add(key)

        for spec in self._side_observability_pseudo_specs("ac"):
            add(
                spec.device_name,
                spec.value,
                spec.device_type_code,
                spec.device_pos,
                spec.meas_type_code,
            )
        for spec in self._side_observability_pseudo_specs("dc"):
            add(
                spec.device_name,
                spec.value,
                spec.device_type_code,
                spec.device_pos,
                spec.meas_type_code,
            )

        for spec in self._hybrid_converter_measurement_specs(source="flow"):
            add(
                spec.device_name,
                spec.value,
                spec.device_type_code,
                spec.device_pos,
                spec.meas_type_code,
            )

        self._observability_pseudo_candidate_cache = candidates
        return candidates

    def _select_weak_direction_pseudo_candidates(
        self,
        observability: ObservabilityResult,
        candidates: Sequence[Measurement],
        max_add: int,
    ) -> List[Measurement]:
        if max_add <= 0 or not candidates:
            return []
        x = self.initial_state()
        cache = self._observability_matrix_cache_for(observability, self.active_measurements, x)
        H = cache.get("H") if cache is not None else self.jacobian_sparse(x, self.active_measurements)
        direction = observability_weak_direction(H, self.n_state, observability.weak_states)
        if direction.size != self.n_state or not np.any(direction):
            return list(candidates[:max_add])
        candidate_h = None
        if cache is not None:
            candidate_cache = cache.setdefault("candidate_jacobians", {})
            signature = self._measurement_signature(candidates)
            candidate_h = candidate_cache.get(signature)
        if candidate_h is None:
            candidate_h = self.jacobian_sparse(x, candidates)
            if cache is not None:
                candidate_cache[signature] = candidate_h
        scores = np.abs(candidate_h @ direction)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size != len(candidates) or not np.any(scores > 0.0):
            return list(candidates[:max_add])
        order = np.argsort(-scores, kind="stable")
        selected = [candidates[int(pos)] for pos in order[:max_add] if scores[int(pos)] > 0.0]
        return selected or list(candidates[:max_add])

    def _side_observability_pseudo_specs(
        self,
        side: str,
    ) -> List["_HybridConverterMeasurementSpec"]:
        estimator = self._ac_sub_estimator if side == "ac" else self._dc_sub_estimator if side == "dc" else None
        if estimator is None:
            return []
        candidates = estimator._observability_pseudo_candidate_measurements()
        table = getattr(candidates, "table", None)
        if table is None:
            raise RuntimeError(f"{side.upper()} observability candidates must expose an array measurement table")
        row_count = int(table.idx.size)
        if row_count == 0:
            return []
        device_type_code = np.asarray(table.device_type_code, dtype=np.int16)
        device_pos = np.asarray(table.device_pos, dtype=np.int64)
        meas_type_code = np.asarray(table.meas_type_code, dtype=np.int16)
        values = np.asarray(table.value, dtype=np.float64)
        if not all(array.size == row_count for array in (device_type_code, device_pos, meas_type_code, values)):
            raise RuntimeError(f"{side.upper()} observability candidate arrays have inconsistent sizes")
        names_by_code = self._measurement_name_arrays_by_type_code(np.unique(device_type_code))
        table_names = np.asarray(table.device_name, dtype=object)
        specs: List[HybridStateEstimator._HybridConverterMeasurementSpec] = []
        for row in range(row_count):
            code = int(device_type_code[row])
            pos = int(device_pos[row])
            names = np.asarray(names_by_code.get(code, ()), dtype=object)
            if 0 <= pos < names.size:
                device_name = str(names[pos])
            elif table_names.size == row_count:
                device_name = str(table_names[row])
            else:
                device_name = ""
            specs.append(
                self._measurement_spec(
                    code,
                    pos,
                    int(meas_type_code[row]),
                    float(values[row]),
                    device_name,
                )
            )
        return specs

    def _targeted_side_specs_for_state(
        self,
        meta: StateMeta,
        existing_keys: set,
    ) -> List["_HybridConverterMeasurementSpec"]:
        name = meta.device_name
        device_type_code = int(meta.device_type_code)
        device_pos = int(meta.device_pos)
        if device_type_code <= 0 or device_pos < 0:
            self._warn_missing_device_pos("targeted_side_state", device_type_code, name, meta.meas_type_code)
            return []
        if meta.side == "ac" and meta.kind in ("zero_current", "break_current"):
            if device_type_code not in (DEVICE_TYPE_ACZeroBranch, DEVICE_TYPE_ACBreak):
                return []
            p_from_key = self._packed_measurement_key(device_type_code, device_pos, MEAS_TYPE_P_FROM)
            q_from_key = self._packed_measurement_key(device_type_code, device_pos, MEAS_TYPE_Q_FROM)
            p_type_code = MEAS_TYPE_P_TO if p_from_key in existing_keys else MEAS_TYPE_P_FROM
            q_type_code = MEAS_TYPE_Q_TO if q_from_key in existing_keys else MEAS_TYPE_Q_FROM
            return [
                self._measurement_spec(device_type_code, device_pos, p_type_code, 0.0, name),
                self._measurement_spec(device_type_code, device_pos, q_type_code, 0.0, name),
            ]
        if meta.side == "dc" and meta.kind in ("zero_current", "break_current"):
            if device_type_code not in (DEVICE_TYPE_DCZeroBranch, DEVICE_TYPE_DCBreak):
                return []
            return [
                self._measurement_spec(
                    device_type_code,
                    device_pos,
                    MEAS_TYPE_I_FROM,
                    0.0,
                    name,
                )
            ]
        return []

    def _append_targeted_observability_pseudo(
        self,
        next_idx: int,
        state_idx: int,
        existing_keys: set,
        max_add: int,
    ) -> Tuple[int, int]:
        if max_add <= 0:
            return next_idx, 0
        meta = state_meta_at(self.state_meta, state_idx)
        if meta is None:
            return next_idx, 0
        name = meta.device_name
        if meta.side == "ac" and meta.kind == "angle":
            return next_idx, 0
        added = 0

        def add(
            device_type_code: int,
            meas_type_code: int,
            value: float,
            device_pos: int = -1,
        ) -> None:
            nonlocal next_idx, added
            if added >= max_add:
                return
            device_type_code = int(device_type_code)
            meas_type_code = int(meas_type_code)
            device_pos = int(device_pos if int(device_pos) >= 0 else meta.device_pos)
            if device_type_code <= 0 or meas_type_code <= 0 or device_pos < 0:
                self._warn_missing_device_pos("targeted_add", device_type_code, name, meas_type_code)
                return
            key = self._packed_measurement_key(device_type_code, device_pos, meas_type_code)
            device_type = DEVICE_TYPE_NAMES.get(device_type_code, "")
            meas_type = MEAS_TYPE_NAMES.get(meas_type_code, "")
            pseudo_name = f"pseudo_obs_{meas_type.lower()}_{name}"
            if self._voltage_pseudo_is_covered_by_code(device_type_code, device_pos, meas_type_code):
                return
            if key < 0 or key in existing_keys:
                return
            new_next_idx = self._append_pseudo_measurement(
                next_idx,
                pseudo_name,
                device_type,
                name,
                meas_type,
                value,
                device_type_code=device_type_code,
                device_pos=device_pos,
                meas_type_code=meas_type_code,
            )
            if new_next_idx == next_idx:
                return
            next_idx = new_next_idx
            existing_keys.add(key)
            added += 1

        if meta.kind == "voltage" and int(meta.device_type_code) == DEVICE_TYPE_ACNode and int(meta.device_pos) >= 0:
            ac = self._ac_sub_estimator
            solver_pos = np.asarray(getattr(ac, "_ac_node_plan_pos", np.asarray([], dtype=np.int64)), dtype=np.int64)
            values = np.asarray(getattr(ac, "file_voltage", np.asarray([], dtype=np.float64)), dtype=np.float64)
            device_pos = int(meta.device_pos)
            voltage = 1.0
            if device_pos < solver_pos.size:
                node_pos = int(solver_pos[device_pos])
                if 0 <= node_pos < values.size:
                    voltage = float(values[node_pos])
            add(
                meta.device_type_code,
                MEAS_TYPE_V,
                voltage,
                meta.device_pos,
            )
            return next_idx, added
        if meta.kind == "voltage" and int(meta.device_type_code) == DEVICE_TYPE_DCNode and int(meta.device_pos) >= 0:
            dc = self._dc_sub_estimator
            solver_pos = np.asarray(getattr(dc, "_raw_node_solver_pos_alive", np.asarray([], dtype=np.int64)), dtype=np.int64)
            values = np.asarray(getattr(dc, "_node_voltage_by_pos", np.asarray([], dtype=np.float64)), dtype=np.float64)
            device_pos = int(meta.device_pos)
            voltage = 1.0
            if device_pos < solver_pos.size:
                node_pos = int(solver_pos[device_pos])
                if 0 <= node_pos < values.size:
                    voltage = float(values[node_pos])
            add(
                meta.device_type_code,
                MEAS_TYPE_V,
                voltage,
                meta.device_pos,
            )
            return next_idx, added
        converter_specs = self._targeted_converter_specs_for_state(meta, source="flow")
        if converter_specs:
            for spec in converter_specs:
                add(
                    spec.device_type_code,
                    spec.meas_type_code,
                    spec.value,
                    spec.device_pos,
                )
            return next_idx, added
        side_specs = self._targeted_side_specs_for_state(meta, existing_keys)
        if side_specs:
            for spec in side_specs:
                add(
                    spec.device_type_code,
                    spec.meas_type_code,
                    spec.value,
                    spec.device_pos,
                )
        return next_idx, added

    def _normalize_measurements(self, measurements: Optional[Sequence[Measurement]]) -> List[Measurement]:
        if measurements is None:
            return self.active_measurements
        table = getattr(measurements, "table", None)
        if table is not None and int(np.asarray(getattr(table, "idx", np.asarray([]))).size) == len(measurements):
            return measurements
        if isinstance(measurements, list):
            return measurements
        raise RuntimeError("Hybrid SE requires table-backed measurements; Measurement object iteration is disabled.")

    def _measurement_table(self, measurements: Sequence[Measurement]):
        table = getattr(measurements, "table", None)
        if table is not None and len(table.idx) == len(measurements):
            return table
        table = _measurement_table_from_measurements(measurements)
        if isinstance(measurements, MeasurementList):
            measurements.table = table
        return table

    def _measurement_vectors(self, measurements: Sequence[Measurement]) -> Tuple[np.ndarray, np.ndarray]:
        if measurements is self.active_measurements:
            return self.active_z, self.active_weight
        table = self._measurement_table(measurements)
        return np.asarray(table.value, dtype=np.float64), np.asarray(table.weight, dtype=np.float64)

    @staticmethod
    def _measurement_signature(measurements: Sequence[Measurement]) -> Tuple[Tuple[int, int, int, float, float, bool], ...]:
        return tuple(
            (
                int(getattr(meas, "device_type_code", 0)),
                int(getattr(meas, "device_pos", -1)),
                int(getattr(meas, "meas_type_code", 0)),
                float(meas.value),
                float(meas.weight),
                bool(meas.valid),
            )
            for meas in measurements
        )

    @staticmethod
    def _uniform_weight(weight: np.ndarray) -> Optional[float]:
        if weight.size == 0:
            return None
        first_weight = float(weight[0])
        return first_weight if bool(np.all(weight == first_weight)) else None

    def _cache_observability_matrix(
        self,
        result: ObservabilityResult,
        x: np.ndarray,
        measurements: Sequence[Measurement],
        H,
    ) -> None:
        self._observability_matrix_cache = {
            "result": result,
            "measurements": measurements,
            "x": np.asarray(x, dtype=np.float64).copy(),
            "H": H,
            "normal_pattern": None,
            "candidate_jacobians": {},
        }

    def _observability_matrix_cache_for(
        self,
        result: Optional[ObservabilityResult],
        measurements: Sequence[Measurement],
        x: np.ndarray,
    ):
        cache = getattr(self, "_observability_matrix_cache", None)
        if cache is None or cache.get("result") is not result:
            return None
        if cache.get("measurements") is not measurements:
            return None
        cached_x = cache.get("x")
        x_array = np.asarray(x, dtype=np.float64)
        if cached_x is None or cached_x.shape != x_array.shape or not np.array_equal(cached_x, x_array):
            return None
        return cache

    @staticmethod
    def _weighted_objective(weight: np.ndarray, residual: np.ndarray) -> float:
        return 0.5 * float(np.einsum("i,i,i->", weight, residual, residual, optimize=False))

    def _measurement_residual(
        self,
        z: np.ndarray,
        z_est: np.ndarray,
        measurements: Sequence[Measurement],
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if out is None:
            return np.subtract(z, z_est, dtype=np.float64)
        np.subtract(z, z_est, out=out)
        return out

    def _safe_current(self, p: np.ndarray, q: Optional[np.ndarray], v: np.ndarray) -> np.ndarray:
        power = np.abs(p) if q is None else np.hypot(p, q)
        return np.divide(power, v, out=np.zeros_like(power, dtype=np.float64), where=np.abs(v) > self.min_current_voltage)

    @staticmethod
    def _state_values_from_cols(
        x: np.ndarray,
        cols: np.ndarray,
        defaults: np.ndarray,
        offset: int = 0,
    ) -> np.ndarray:
        values = defaults.copy()
        mask = cols >= 0
        if np.any(mask):
            values[mask] = x[cols[mask] - int(offset)]
        return values

    @staticmethod
    def _lookup_node_pos_by_idx(node_idx: int, ids: np.ndarray, pos: np.ndarray) -> int:
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.asarray(pos, dtype=np.int64)
        if ids.size == 0 or pos.size == 0:
            return -1
        loc = int(np.searchsorted(ids, int(node_idx)))
        if loc >= ids.size or int(ids[loc]) != int(node_idx):
            return -1
        value = int(pos[loc])
        return value if value >= 0 else -1

    @staticmethod
    def _lookup_node_pos_array_by_idx(node_idx: np.ndarray, ids: np.ndarray, pos: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.full(nodes.size, -1, dtype=np.int64)
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.asarray(pos, dtype=np.int64)
        if nodes.size == 0 or ids.size == 0 or pos.size == 0:
            return out
        loc = np.searchsorted(ids, nodes)
        in_range = loc < ids.size
        if np.any(in_range):
            rows = np.flatnonzero(in_range)
            matched = ids[loc[rows].astype(np.intp, copy=False)] == nodes[rows]
            if np.any(matched):
                matched_rows = rows[matched]
                values = pos[loc[matched_rows].astype(np.intp, copy=False)]
                valid = values >= 0
                out[matched_rows[valid]] = values[valid]
        return out

    def _ac_node_pos_for_idx(self, node_idx: int) -> int:
        ac = self._ac_sub_estimator
        if ac is None:
            return -1
        return self._lookup_node_pos_by_idx(
            int(node_idx),
            getattr(ac, "_ac_node_id_lookup_ids", np.asarray([], dtype=np.int64)),
            getattr(ac, "_ac_node_id_lookup_pos", np.asarray([], dtype=np.int64)),
        )

    def _ac_node_pos_for_idx_array(self, node_idx: np.ndarray) -> np.ndarray:
        ac = self._ac_sub_estimator
        if ac is None:
            return np.full(np.asarray(node_idx, dtype=np.int64).size, -1, dtype=np.int64)
        return self._lookup_node_pos_array_by_idx(
            node_idx,
            getattr(ac, "_ac_node_id_lookup_ids", np.asarray([], dtype=np.int64)),
            getattr(ac, "_ac_node_id_lookup_pos", np.asarray([], dtype=np.int64)),
        )

    @staticmethod
    def _measurement_kind_code_lookup(kind_by_code: Dict[int, int]) -> np.ndarray:
        if not kind_by_code:
            return np.asarray([], dtype=np.int16)
        valid_codes = [int(code) for code in kind_by_code if int(code) >= 0]
        if not valid_codes:
            return np.asarray([], dtype=np.int16)
        lookup = np.empty(max(valid_codes) + 1, dtype=np.int16)
        lookup.fill(-1)
        for code, kind in kind_by_code.items():
            code = int(code)
            if code >= 0:
                lookup[code] = int(kind)
        return lookup

    def _dc_node_pos_for_idx(self, node_idx: int) -> int:
        dc = self._dc_sub_estimator
        if dc is None:
            return -1
        return self._lookup_node_pos_by_idx(
            int(node_idx),
            getattr(dc, "_node_idx_lookup_ids", np.asarray([], dtype=np.int64)),
            getattr(dc, "_node_idx_lookup_pos", np.asarray([], dtype=np.int64)),
        )

    def _dc_node_pos_for_idx_array(self, node_idx: np.ndarray) -> np.ndarray:
        dc = self._dc_sub_estimator
        if dc is None:
            return np.full(np.asarray(node_idx, dtype=np.int64).size, -1, dtype=np.int64)
        return self._lookup_node_pos_array_by_idx(
            node_idx,
            getattr(dc, "_node_idx_lookup_ids", np.asarray([], dtype=np.int64)),
            getattr(dc, "_node_idx_lookup_pos", np.asarray([], dtype=np.int64)),
        )

    def _ac_voltage_col_for_node(self, node_idx: int) -> int:
        ac = self._ac_sub_estimator
        pos = self._ac_node_pos_for_idx(int(node_idx))
        if ac is None or pos < 0:
            return -1
        col = int(ac.voltage_col[pos])
        return col if col >= 0 else -1

    def _ac_voltage_cols_for_nodes(self, node_idx: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.full(nodes.size, -1, dtype=np.int32)
        ac = self._ac_sub_estimator
        if ac is None or nodes.size == 0:
            return out
        pos = self._ac_node_pos_for_idx_array(nodes)
        voltage_col = np.asarray(getattr(ac, "voltage_col", np.asarray([], dtype=np.int32)), dtype=np.int32)
        rows = np.flatnonzero((pos >= 0) & (pos < voltage_col.size))
        if rows.size:
            cols = voltage_col[pos[rows].astype(np.intp, copy=False)]
            valid = cols >= 0
            out[rows[valid]] = cols[valid]
        return out

    def _ac_voltage_default_for_node(self, node_idx: int) -> float:
        ac = self._ac_sub_estimator
        pos = self._ac_node_pos_for_idx(int(node_idx))
        if ac is not None and pos >= 0:
            if pos in getattr(ac, "ref_voltages", {}):
                return float(ac.ref_voltages[pos])
            file_voltage = getattr(ac, "file_voltage", None)
            if file_voltage is not None and pos < int(np.asarray(file_voltage).size):
                value = float(np.asarray(file_voltage, dtype=np.float64)[pos])
                return value if value != 0.0 else 1.0
        node = getattr(getattr(self.network, "ac", None), "node_dict", {}).get(int(node_idx))
        return float(getattr(node, "voltage", 1.0) or 1.0)

    def _ac_voltage_defaults_for_nodes(self, node_idx: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.ones(nodes.size, dtype=np.float64)
        ac = self._ac_sub_estimator
        if ac is None or nodes.size == 0:
            return out
        pos = self._ac_node_pos_for_idx_array(nodes)
        valid_pos = pos >= 0
        file_voltage = getattr(ac, "file_voltage", None)
        if file_voltage is not None:
            voltage = np.asarray(file_voltage, dtype=np.float64)
            rows = np.flatnonzero(valid_pos & (pos < voltage.size))
            if rows.size:
                values = voltage[pos[rows].astype(np.intp, copy=False)]
                out[rows] = np.where(values != 0.0, values, 1.0)
        self._apply_reference_voltages(pos, getattr(ac, "ref_voltages", {}), out)
        invalid_rows = np.flatnonzero(~valid_pos)
        node_dict = getattr(getattr(self.network, "ac", None), "node_dict", {})
        for row in invalid_rows:
            node = node_dict.get(int(nodes[row]))
            out[row] = float(getattr(node, "voltage", 1.0) or 1.0)
        return out

    @staticmethod
    def _apply_reference_voltages(pos: np.ndarray, references, out: np.ndarray) -> None:
        """Apply reference overrides in one batch, without one scan per reference."""
        if not references or pos.size == 0:
            return
        reference_pos = np.fromiter(references, dtype=np.int64, count=len(references))
        values = np.fromiter(references.values(), dtype=np.float64, count=len(references))
        order = np.argsort(reference_pos)
        reference_pos = reference_pos[order]
        loc = np.searchsorted(reference_pos, pos)
        rows = np.flatnonzero((pos >= 0) & (loc < reference_pos.size))
        matched = reference_pos[loc[rows]] == pos[rows]
        rows = rows[matched]
        out[rows] = values[order[loc[rows]]]

    def _dc_voltage_default_for_node(self, node_idx: int) -> float:
        dc = self._dc_sub_estimator
        pos = self._dc_node_pos_for_idx(int(node_idx))
        if dc is not None and pos >= 0:
            if pos in getattr(dc, "ref_voltages", {}):
                return float(dc.ref_voltages[pos])
            node_voltage = getattr(dc, "_node_voltage_by_pos", None)
            if node_voltage is not None and pos < int(np.asarray(node_voltage).size):
                value = float(np.asarray(node_voltage, dtype=np.float64)[pos])
                return value if value != 0.0 else 1.0
        node = getattr(getattr(self.network, "dc", None), "node_dict", {}).get(int(node_idx))
        return float(getattr(node, "voltage", 1.0) or 1.0)

    def _dc_voltage_defaults_for_nodes(self, node_idx: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.ones(nodes.size, dtype=np.float64)
        dc = self._dc_sub_estimator
        if dc is None or nodes.size == 0:
            return out
        pos = self._dc_node_pos_for_idx_array(nodes)
        valid_pos = pos >= 0
        node_voltage = getattr(dc, "_node_voltage_by_pos", None)
        if node_voltage is not None:
            voltage = np.asarray(node_voltage, dtype=np.float64)
            rows = np.flatnonzero(valid_pos & (pos < voltage.size))
            if rows.size:
                values = voltage[pos[rows].astype(np.intp, copy=False)]
                out[rows] = np.where(values != 0.0, values, 1.0)
        self._apply_reference_voltages(pos, getattr(dc, "ref_voltages", {}), out)
        invalid_rows = np.flatnonzero(~valid_pos)
        if invalid_rows.size:
            node_dict = getattr(getattr(self.network, "dc", None), "node_dict", {})
            for row in invalid_rows:
                node = node_dict.get(int(nodes[row]))
                out[row] = float(getattr(node, "voltage", 1.0) or 1.0)
        return out

    def _dc_voltage_col_for_node(self, node_idx: int) -> int:
        dc = self._dc_sub_estimator
        pos = self._dc_node_pos_for_idx(int(node_idx))
        if dc is None or pos < 0:
            return -1
        col = int(dc.voltage_col[pos])
        return self.dc_state_start + col if col >= 0 else -1

    def _dc_voltage_cols_for_nodes(self, node_idx: np.ndarray) -> np.ndarray:
        nodes = np.asarray(node_idx, dtype=np.int64)
        out = np.full(nodes.size, -1, dtype=np.int32)
        dc = self._dc_sub_estimator
        if dc is None or nodes.size == 0:
            return out
        pos = self._dc_node_pos_for_idx_array(nodes)
        voltage_col = np.asarray(getattr(dc, "voltage_col", np.asarray([], dtype=np.int32)), dtype=np.int32)
        rows = np.flatnonzero((pos >= 0) & (pos < voltage_col.size))
        if rows.size:
            cols = voltage_col[pos[rows].astype(np.intp, copy=False)]
            valid = cols >= 0
            out[rows[valid]] = (self.dc_state_start + cols[valid]).astype(np.int32, copy=False)
        return out

    def _evaluate_active_hybrid_measurements(
        self,
        values: np.ndarray,
        ac_x: np.ndarray,
        dc_x: np.ndarray,
        hybrid_x: np.ndarray,
    ) -> None:
        self._evaluate_hybrid_measurements(values, ac_x, dc_x, hybrid_x, self._active_hybrid_measurement_plan)

    def _evaluate_hybrid_measurements(
        self,
        values: np.ndarray,
        ac_x: np.ndarray,
        dc_x: np.ndarray,
        hybrid_x: np.ndarray,
        plan: "HybridStateEstimator._HybridMeasurementPlan",
    ) -> None:
        min_v = self.min_current_voltage

        def _gather_voltage(x: np.ndarray, bucket) -> np.ndarray:
            out = bucket.v_default.copy()
            if bucket.v_any_x:
                out[bucket.v_use_x] = x[bucket.v_x_index]
            return out

        # DCAC identity codes (1 P_DC, 2 P_AC, 3 Q_AC).
        for bucket in (plan.dcac_p_dc, plan.dcac_p_ac, plan.dcac_q_ac):
            if bucket.rows.size:
                values[bucket.rows] = hybrid_x[bucket.hx_p]

        # DCAC voltage-only codes (4 V_DC, 6 V_AC).
        for bucket, source_x in (
            (plan.dcac_v_dc, dc_x),
            (plan.dcac_v_ac, ac_x),
        ):
            if bucket.rows.size:
                values[bucket.rows] = _gather_voltage(source_x, bucket)

        # DCAC code 5 — I_DC = P_DC / V_DC.
        bucket = plan.dcac_i_dc
        if bucket.rows.size:
            p_dc_k = hybrid_x[bucket.hx_p]
            v_dc_k = _gather_voltage(dc_x, bucket)
            out = np.zeros(bucket.rows.size, dtype=np.float64)
            np.divide(p_dc_k, v_dc_k, out=out, where=np.abs(v_dc_k) > min_v)
            values[bucket.rows] = out

        # DCAC code 7 — I_AC = hypot(P_AC, Q_AC) / V_AC.
        bucket = plan.dcac_i_ac
        if bucket.rows.size:
            p_ac_k = hybrid_x[bucket.hx_p]
            q_ac_k = hybrid_x[bucket.hx_q]
            v_ac_k = _gather_voltage(ac_x, bucket)
            out = np.zeros(bucket.rows.size, dtype=np.float64)
            np.divide(np.hypot(p_ac_k, q_ac_k), v_ac_k, out=out, where=np.abs(v_ac_k) > min_v)
            values[bucket.rows] = out

    def evaluate(
        self,
        x: np.ndarray,
        measurement_plan_tables=None,
        *,
        out: Optional[np.ndarray] = None,
        measurements: Optional[object] = None,
    ) -> np.ndarray:
        # Keep the AC/DC legacy plan keyword and the common measurements
        # keyword interchangeable.  Rejecting both avoids silently evaluating
        # one plan while the caller believes the other was selected.
        if measurements is not None:
            if measurement_plan_tables is not None:
                raise TypeError("Pass either measurements or measurement_plan_tables, not both")
            measurement_plan_tables = measurements
        measurements = measurement_plan_tables
        delegate = self._delegate()
        if delegate is not None:
            return delegate.evaluate(x, measurements, out=out)
        measurements = self._normalize_measurements(measurements)
        ac_x, dc_x, hybrid_x = self._split_state(x)
        n_meas = len(measurements)
        if out is None:
            values = np.zeros(n_meas, dtype=np.float64)
        else:
            values = out
            values.fill(0.0)
        blocks = self._measurement_blocks_for(measurements)
        ac_block = blocks["ac"]
        dc_block = blocks["dc"]
        if ac_block.measurements and self._ac_sub_estimator is not None:
            values[ac_block.rows] = self._ac_sub_estimator.evaluate(
                ac_x,
                self._sub_measurement_runtime_input(ac_block),
            )
        if dc_block.measurements and self._dc_sub_estimator is not None:
            values[dc_block.rows] = self._dc_sub_estimator.evaluate(
                dc_x,
                self._sub_measurement_runtime_input(dc_block),
            )
        self._evaluate_hybrid_measurements(
            values,
            ac_x,
            dc_x,
            hybrid_x,
            self._hybrid_measurement_plan_for(measurements, blocks),
        )
        return values

    def jacobian(self, x: np.ndarray, measurements: Optional[object] = None) -> np.ndarray:
        return self.jacobian_sparse(x, measurements).toarray()

    def jacobian_sparse(self, x: np.ndarray, measurements: Optional[object] = None):
        delegate = self._delegate()
        if delegate is not None:
            return delegate.jacobian_sparse(x, measurements)
        measurements = self._normalize_measurements(measurements)
        ac_x, dc_x, hybrid_x = self._split_state(x)
        shape = (len(measurements), self.n_state)
        is_active = measurements is self.active_measurements
        H = self._jacobian_builder if is_active else SparseJacobianBuilder(shape)
        H.reset()
        if is_active:
            obs_cache = getattr(self, "_observability_matrix_cache", None)
            if obs_cache is not None:
                obs_cache["H"] = None
        blocks = self._measurement_blocks_for(measurements)
        self._append_sub_jacobian(H, self._ac_sub_estimator, ac_x, blocks["ac"], 0, "ac" if is_active else None)
        self._append_sub_jacobian(H, self._dc_sub_estimator, dc_x, blocks["dc"], self.dc_state_start, "dc" if is_active else None)
        self._append_hybrid_jacobian_plan(
            H,
            ac_x,
            dc_x,
            hybrid_x,
            self._hybrid_measurement_plan_for(measurements, blocks),
        )
        return H.to_csr()

    def _append_sub_jacobian(
        self,
        target,
        estimator,
        sub_x: np.ndarray,
        block: "HybridStateEstimator._MeasurementSideBlock",
        col_offset: int,
        cache_key: Optional[str] = None,
    ) -> None:
        """Stamp the sub-estimator's CSR jacobian into ``target``.

        When ``cache_key`` is provided (i.e. the parent is operating on its
        active measurements), the remapped row/column index arrays are cached
        after the first call. Subsequent calls skip the CSR→COO conversion,
        the row gather via ``block.rows[csr_rows]`` and the column offset
        arithmetic — only the freshly-computed ``sub_csr.data`` is stamped.

        Cache invalidation is driven by ``_refresh_active_measurement_state_layout``
        and friends, which reset ``self._sub_jacobian_stamp_cache`` whenever the
        active partition changes. We additionally guard against unexpected
        pattern shifts by checking the child CSR structure on every reuse.
        """
        if estimator is None or len(block.measurements) == 0:
            return
        # The child estimator installs a fixed-pattern SparseJacobianBuilder
        # for this delegated measurement block. Keep that mode enabled: after
        # the first iteration the child refreshes only CSR data while retaining
        # its symbolic sparse structure. Forcing dynamic mode here rebuilt the
        # COO/CSR pattern on every Hybrid WLS iteration.
        sub_csr = estimator.jacobian_sparse(sub_x, self._sub_measurement_runtime_input(block))
        if sub_csr.nnz == 0:
            return
        cache_store = self._sub_jacobian_stamp_cache if cache_key is not None else None
        cached = cache_store.get(cache_key) if cache_store is not None else None
        if cached is not None and self._sub_jacobian_pattern_matches(cached, sub_csr):
            # Valid CSR columns and the checked stamp already guarantee the
            # fixed chunk has no negative/masked indices. Refresh values only.
            if not target._try_append_fixed_data_chunk(sub_csr.data):
                target.add_many(cached["parent_rows"], cached["parent_cols"], sub_csr.data)
            return

        # Cold path: derive row indices from indptr without going through tocoo().
        indptr = np.asarray(sub_csr.indptr)
        indices = np.asarray(sub_csr.indices)
        sub_rows = np.repeat(
            np.arange(sub_csr.shape[0], dtype=np.int64),
            np.diff(indptr).astype(np.int64, copy=False),
        )
        parent_rows = block.rows[sub_rows].astype(np.int32, copy=False)
        parent_cols = (indices.astype(np.int32, copy=False) + np.int32(col_offset))
        target.add_many(parent_rows, parent_cols, sub_csr.data)
        if cache_store is not None:
            cache_store[cache_key] = {
                "parent_rows": parent_rows,
                "parent_cols": parent_cols,
                "nnz": int(sub_csr.nnz),
                "shape": tuple(int(item) for item in sub_csr.shape),
                "indptr": sub_csr.indptr,
                "indices": sub_csr.indices,
            }

    @staticmethod
    def _sub_jacobian_pattern_matches(cached: Dict[str, object], sub_csr) -> bool:
        """Return whether a cached child-to-parent stamp is still valid."""
        if int(cached.get("nnz", -1)) != int(sub_csr.nnz):
            return False
        if tuple(cached.get("shape", ())) != tuple(int(item) for item in sub_csr.shape):
            return False
        cached_indptr = cached.get("indptr")
        cached_indices = cached.get("indices")
        if cached_indptr is sub_csr.indptr and cached_indices is sub_csr.indices:
            return True
        return bool(
            cached_indptr is not None
            and cached_indices is not None
            and np.array_equal(cached_indptr, sub_csr.indptr)
            and np.array_equal(cached_indices, sub_csr.indices)
        )

    @staticmethod
    def _append_vector_jacobian_entries(
        target,
        row_array: np.ndarray,
        col_array: np.ndarray,
        value_array,
        mask: np.ndarray,
    ) -> None:
        if not np.any(mask):
            return
        selected_cols = np.asarray(col_array, dtype=np.int32)[mask]
        valid = selected_cols >= 0
        if not np.any(valid):
            return
        selected_rows = row_array[mask][valid]
        selected_cols = selected_cols[valid]
        if np.isscalar(value_array):
            selected_values = np.full(selected_rows.size, float(value_array), dtype=np.float64)
        else:
            selected_values = np.asarray(value_array, dtype=np.float64)[mask][valid]
            valid_value = selected_values != 0.0
            if not np.any(valid_value):
                return
            selected_rows = selected_rows[valid_value]
            selected_cols = selected_cols[valid_value]
            selected_values = selected_values[valid_value]
        target.add_many(
            selected_rows.astype(np.int32, copy=False),
            selected_cols.astype(np.int32, copy=False),
            selected_values.astype(np.float64, copy=False),
        )

    @staticmethod
    def _masked_divide(numerator, denominator: np.ndarray, mask: np.ndarray) -> np.ndarray:
        denominator = np.asarray(denominator, dtype=np.float64)
        values = np.zeros_like(denominator, dtype=np.float64)
        np.divide(numerator, denominator, out=values, where=mask)
        return values

    def _append_active_hybrid_jacobian_entries(
        self,
        target,
        ac_x: np.ndarray,
        dc_x: np.ndarray,
        hybrid_x: np.ndarray,
    ) -> None:
        self._append_hybrid_jacobian_plan(target, ac_x, dc_x, hybrid_x, self._active_hybrid_measurement_plan)

    def _append_hybrid_jacobian_plan(
        self,
        target,
        ac_x: np.ndarray,
        dc_x: np.ndarray,
        hybrid_x: np.ndarray,
        plan: "HybridStateEstimator._HybridMeasurementPlan",
    ) -> None:
        min_v = self.min_current_voltage

        def _gather_voltage(x: np.ndarray, bucket) -> np.ndarray:
            out = bucket.v_default.copy()
            if bucket.v_any_x:
                out[bucket.v_use_x] = x[bucket.v_x_index]
            return out

        def _stamp_identity_power(bucket) -> None:
            """Identity derivative: 1.0 at jcol_p for every row."""
            if bucket.rows.size:
                target.add_many(
                    bucket.rows,
                    bucket.jcol_p,
                    np.ones(bucket.rows.size, dtype=np.float64),
                )

        def _stamp_identity_voltage(bucket) -> None:
            """Identity derivative: 1.0 at v_jcol where v_use_x (cols<0 filtered out)."""
            if bucket.rows.size and bucket.v_any_x:
                mask = bucket.v_use_x
                target.add_many(
                    bucket.rows[mask],
                    bucket.v_jcol[mask],
                    np.ones(int(mask.sum()), dtype=np.float64),
                )

        def _stamp_current_dc(bucket, x_for_v) -> None:
            """I_DC = p_dc / v_dc. Two jacobian terms."""
            if not bucket.rows.size:
                return
            p_dc_k = hybrid_x[bucket.hx_p]
            v_dc_k = _gather_voltage(x_for_v, bucket)
            valid = np.abs(v_dc_k) > min_v
            if not valid.any():
                return
            target.add_many(
                bucket.rows[valid],
                bucket.jcol_p[valid],
                1.0 / v_dc_k[valid],
            )
            v_mask = bucket.v_use_x & valid
            if v_mask.any():
                v_dc_sel = v_dc_k[v_mask]
                target.add_many(
                    bucket.rows[v_mask],
                    bucket.v_jcol[v_mask],
                    -p_dc_k[v_mask] / (v_dc_sel * v_dc_sel),
                )

        def _stamp_current_ac(bucket, x_for_v) -> None:
            """I = hypot(p, q) / v. Three jacobian terms."""
            if not bucket.rows.size:
                return
            p_k = hybrid_x[bucket.hx_p]
            q_k = hybrid_x[bucket.hx_q]
            v_k = _gather_voltage(x_for_v, bucket)
            s_k = np.hypot(p_k, q_k)
            valid = (np.abs(v_k) > min_v) & (s_k > 1e-12)
            if not valid.any():
                return
            s_v = s_k[valid] * v_k[valid]
            rows_v = bucket.rows[valid]
            target.add_many(rows_v, bucket.jcol_p[valid], p_k[valid] / s_v)
            target.add_many(rows_v, bucket.jcol_q[valid], q_k[valid] / s_v)
            v_mask = bucket.v_use_x & valid
            if v_mask.any():
                v_sel = v_k[v_mask]
                target.add_many(
                    bucket.rows[v_mask],
                    bucket.v_jcol[v_mask],
                    -s_k[v_mask] / (v_sel * v_sel),
                )

        # DCAC linear codes (1/2/3 — identity on power state; 4/6 — identity on voltage state).
        _stamp_identity_power(plan.dcac_p_dc)
        _stamp_identity_power(plan.dcac_p_ac)
        _stamp_identity_power(plan.dcac_q_ac)
        _stamp_identity_voltage(plan.dcac_v_dc)
        _stamp_identity_voltage(plan.dcac_v_ac)
        # DCAC nonlinear codes.
        _stamp_current_dc(plan.dcac_i_dc, dc_x)
        _stamp_current_ac(plan.dcac_i_ac, ac_x)

    def observability_analysis(
        self,
        x: Optional[np.ndarray] = None,
        measurements: Optional[object] = None,
        H: Optional[np.ndarray] = None,
        normal_matrix: Optional[np.ndarray] = None,
        normal_factor_diag: Optional[np.ndarray] = None,
    ) -> ObservabilityResult:
        if self.fluid_estimators and measurements is None and normal_matrix is None and normal_factor_diag is None:
            return self._multi_energy_observability_analysis(x=x, H=H)
        delegate = self._delegate()
        if delegate is not None:
            return delegate.observability_analysis(x, measurements, H, normal_matrix, normal_factor_diag)
        default_active_request = (
            x is None
            and measurements is None
            and H is None
            and normal_matrix is None
            and normal_factor_diag is None
        )
        if (
            default_active_request
            and self._initial_observability_cache is not None
        ):
            return self._initial_observability_cache
        measurements = self._normalize_measurements(measurements)
        x = self.initial_state() if x is None else x
        H = self.jacobian_sparse(x, measurements) if H is None else H
        if matrix_is_empty(H):
            result = ObservabilityResult(False, 0, self.n_state, 0, self.n_state, np.array([]), [])
            if default_active_request:
                self._initial_observability_cache = result
            return result
        if (
            normal_matrix is None
            and normal_factor_diag is None
            and is_sparse_matrix(H)
            and int(H.shape[0]) >= int(self.n_state)
        ):
            structural_rank = sparse_structural_rank(H)
            if structural_rank == self.n_state:
                ac_angle_cols = np.asarray(getattr(self, "ac_theta_state_col", np.asarray([], dtype=np.int32)), dtype=np.int32)
                unanchored_angles = unanchored_angle_state_indices(H, ac_angle_cols[ac_angle_cols >= 0])
                if not unanchored_angles:
                    result = ObservabilityResult(
                        observable=True,
                        rank=self.n_state,
                        state_count=self.n_state,
                        measurement_count=len(measurements),
                        deficiency=0,
                        singular_values=np.array([], dtype=np.float64),
                        weak_states=[],
                    )
                    self._cache_observability_matrix(result, x, measurements, H)
                    if default_active_request:
                        self._initial_observability_cache = result
                    return result
        rank, deficiency, s, weak_states = observability_rank_details(
            H,
            self.n_state,
            normal_matrix=normal_matrix,
            normal_factor_diag=normal_factor_diag,
        )
        result = ObservabilityResult(
            observable=rank == self.n_state,
            rank=rank,
            state_count=self.n_state,
            measurement_count=len(measurements),
            deficiency=max(0, deficiency),
            singular_values=s,
            weak_states=weak_states,
        )
        self._cache_observability_matrix(result, x, measurements, H)
        if default_active_request:
            self._initial_observability_cache = result
        return result

    def estimate(
        self,
        measurements: Optional[object] = None,
        x0: Optional[np.ndarray] = None,
        verbose: bool = False,
        final_diagnostics: bool = True,
        observability: Optional[ObservabilityResult] = None,
        _defer_observability: bool = False,
        _weight_override: Optional[np.ndarray] = None,
        _value_override: Optional[np.ndarray] = None,
        _shared_aat_solver: Optional[CholmodAAtNormalEquationSolver] = None,
        _shared_aat_plan: Optional[CholmodAAtNormalEquationPlan] = None,
        _woodbury_removed_rows: Optional[Sequence[int]] = None,
        _woodbury_prev_weights: Optional[np.ndarray] = None,
        _woodbury_prev_H: Optional[object] = None,
        _coarse_bad_data_threshold: Optional[float] = None,
        _defer_factor_refresh: bool = False,
        _woodbury_u_cache: Optional[Dict] = None,
        _hint_lm_damping: float = 0.0,
    ) -> EstimateResult:
        if not self._prepared:
            self.prepare()
        if self.fluid_estimators and measurements is None:
            return self._estimate_multi_energy(
                x0=x0,
                verbose=verbose,
                final_diagnostics=final_diagnostics,
                observability=observability,
            )
        solve_profile_start = time.perf_counter() if self.profile_enabled else None
        delegate = self._delegate()
        if delegate is not None:
            array_only_result = bool(getattr(self, "_array_only_estimate_result", False))
            previous_delegate_array_only = bool(getattr(delegate, "_array_only_estimate_result", False))
            delegate._array_only_estimate_result = array_only_result
            try:
                try:
                    result = delegate.estimate(
                        measurements,
                        x0,
                        verbose=verbose,
                        final_diagnostics=final_diagnostics,
                        observability=observability,
                    )
                except TypeError as exc:
                    if "final_diagnostics" not in str(exc):
                        raise
                    result = delegate.estimate(
                        measurements,
                        x0,
                        verbose=verbose,
                        observability=observability,
                    )
            finally:
                delegate._array_only_estimate_result = previous_delegate_array_only
            if array_only_result:
                result.measurements = []
            if solve_profile_start is not None:
                self._record_profile_time("solve.total", time.perf_counter() - solve_profile_start)
            return result
        measurements = self._normalize_measurements(measurements)
        n_meas = len(measurements)
        effective_n_meas = (
            int(np.count_nonzero(np.asarray(_weight_override, dtype=np.float64) > 0.0))
            if _weight_override is not None
            else n_meas
        )
        if effective_n_meas < self.n_state:
            raise RuntimeError(f"Not enough valid measurements: {effective_n_meas} < {self.n_state}")
        x = self.initial_state() if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        if observability is None and not _defer_observability:
            start = time.perf_counter() if self.profile_enabled else None
            if measurements is self.active_measurements and x0 is None and _weight_override is None:
                observability = self.observability_analysis()
            else:
                observability = self.observability_analysis(x, measurements)
            if start is not None:
                self._record_profile_time("solve.observability", time.perf_counter() - start)
        observability_cache = (
            self._observability_matrix_cache_for(observability, measurements, x)
            if _weight_override is None
            else None
        )
        cached_initial_H = observability_cache.get("H") if observability_cache is not None else None
        z, weight = self._measurement_vectors(measurements)
        if _value_override is not None:
            z = np.asarray(_value_override, dtype=np.float64)
        weight_masked = _weight_override is not None
        if weight_masked:
            weight = np.asarray(_weight_override, dtype=np.float64)
        active_measurement_run = measurements is self.active_measurements
        uniform_weight = (
            None
            if weight_masked
            else (self.active_uniform_weight if active_measurement_run else self._uniform_weight(weight))
        )
        weights_are_uniform = (
            False
            if weight_masked
            else (self.active_weights_are_uniform if active_measurement_run else uniform_weight is not None)
        )
        weighted_residual = None if weights_are_uniform else np.empty_like(weight)
        normal_solver = NormalEquationSolver(assume_fixed_pattern=active_measurement_run)
        normal_pattern = self._active_normal_pattern if active_measurement_run else None
        if observability_cache is not None and observability_cache.get("normal_pattern") is not None:
            normal_pattern = observability_cache["normal_pattern"]
        use_aat_normal_solver = (
            getattr(self, "matrix_dump_dir", None) is None
            and (CHOLMOD_ANALYZE_AAT is not None or CHOLMOD_CHOLESKY_AAT is not None)
            and self.n_state >= 16
        )
        aat_solver = (
            _shared_aat_solver
            if (use_aat_normal_solver and _shared_aat_solver is not None and not getattr(_shared_aat_solver, "_cholmod_disabled", False))
            else (CholmodAAtNormalEquationSolver(assume_fixed_pattern=active_measurement_run) if use_aat_normal_solver else None)
        )
        aat_plan = _shared_aat_plan if use_aat_normal_solver else None
        if aat_plan is not None:
            aat_plan.prepare_fixed_weights(weight)
        if verbose:
            _print_iteration_header()
        converged = False
        objective = np.inf
        max_correction = np.inf
        residual_inf = np.inf
        H = None
        gain = None
        normal_factor_diag = None
        iteration = 0
        # Pre-allocate evaluation/residual buffers and reuse them across iterations
        # and line-search candidates. On acceptance we swap pointers so the accepted
        # vectors become the "main" buffers without copying.
        z_est = np.empty(n_meas, dtype=np.float64)
        residual = np.empty(n_meas, dtype=np.float64)
        cand_z_est = np.empty(n_meas, dtype=np.float64)
        cand_residual = np.empty(n_meas, dtype=np.float64)
        scale_profile = SEScaleProfile.for_problem(self.n_state, n_meas)
        wb_max_rank = scale_profile.woodbury_max_rank
        z_est_dirty = True
        final_quantities_current = False
        last_solved_via_aat = False
        self._last_estimate_cholmod_factor = None
        frozen_steps = 0
        prev_max_correction = np.inf
        prev_rms_correction = np.inf
        prev_step_accepted = False
        last_factor_x_disp = 0.0
        woodbury_tried = False
        wb_solver = None
        wb_steps = 0
        wb_local_shrunk = False
        factor_is_current_weights = True
        force_lm_damping = 0.0
        micro_step_streak = 0
        slow_micro_streak = 0
        high_damp_flat_streak = 0
        backtrack_streak = 0
        curvature_damping_active = False
        fullstep_osc_streak = 0
        aa_accel = AndersonAccelerator(m=5)
        tr_ctrl = NielsenTrustRegionController()
        curv_stagnation_streak = 0
        curv_stagnation_best_g = np.inf
        flat_valley_streak = 0
        box_kkt_stagnation_streak = 0
        cur_rhs_inf = np.inf
        last_undamped_spd_failed = False
        active_weight_mask = (weight > 0.0) if weight_masked else None
        coarse_stopped = False
        coarse_lb_sq: Optional[float] = None
        coarse_non_pseudo_mask: Optional[np.ndarray] = None
        if _coarse_bad_data_threshold is not None and float(_coarse_bad_data_threshold) > 0.0:
            coarse_lb_sq = max(2.0 * float(_coarse_bad_data_threshold), 8.0) ** 2
            coarse_table = getattr(measurements, "table", None)
            if coarse_table is not None and int(coarse_table.idx.size) == n_meas:
                coarse_non_pseudo_mask = (
                    (~measurement_table_bad_data_protected_mask(coarse_table))
                    & (weight > 0.0)
                )
            else:
                coarse_non_pseudo_mask = weight > 0.0

        v_cols = np.asarray(getattr(self, "voltage_cols", np.asarray([], dtype=np.int32)), dtype=np.int32)
        raw_ac_theta_cols = np.asarray(getattr(self, "ac_theta_state_col", np.asarray([], dtype=np.int32)), dtype=np.int32)
        a_cols = raw_ac_theta_cols[raw_ac_theta_cols >= 0] if raw_ac_theta_cols.size else np.asarray([], dtype=np.int32)
        v_min_bound = float(self.voltage_min)
        v_max_bound = float(self.voltage_max)
        a_min_bound = float(self.angle_min_rad)
        a_max_bound = float(self.angle_max_rad)

        # 预提取交流子系统零阻抗支路/断路器电流幅值测点，供负残差锥尖附近精确秩 1 半正定二阶曲率修正使用
        sw_curv_rows = np.asarray([], dtype=np.int64)
        sw_curv_col_re = np.asarray([], dtype=np.int64)
        sw_curv_col_im = np.asarray([], dtype=np.int64)
        ac_sub = getattr(self, "_ac_sub_estimator", None)
        if ac_sub is not None and int(getattr(ac_sub, "n_switch_current", 0)) > 0:
            ac_block = self._measurement_blocks_for(measurements)["ac"]
            if len(ac_block.measurements) > 0:
                ac_plan_tables = ac_sub._measurement_plan_tables_for(self._sub_measurement_runtime_input(ac_block))
                zc_plan = ac_sub._vector_plans_for_measurement_plan_tables(ac_plan_tables).get("zero_current")
                if zc_plan is not None and zc_plan["current_rows"].size > 0:
                    sw_curv_rows = np.asarray(ac_block.rows, dtype=np.int64)[
                        np.asarray(zc_plan["current_rows"], dtype=np.int64)
                    ]
                    sw_cidx = np.asarray(zc_plan["current_idx"], dtype=np.int64)
                    sw_curv_col_re = int(ac_sub.base_switch_re) + sw_cidx
                    sw_curv_col_im = int(ac_sub.base_switch_im) + sw_cidx

        def _hybrid_dx_needs_lm_guard(dx_vec: np.ndarray, *, strict_step: bool = True) -> bool:
            """判断交直流混合无阻尼步是否非有限或幅值剧烈过冲；健康幅值的贴墙跨界交由线搜索与活跃集投影处理。"""
            if dx_vec.size == 0:
                return False
            if not np.all(np.isfinite(dx_vec)):
                return True
            if v_cols.size:
                dv_slice = dx_vec[v_cols]
                max_dv = float(np.max(np.abs(dv_slice)))
                if max_dv > (0.50 if strict_step else 1.2):
                    return True
            if a_cols.size:
                da_slice = dx_vec[a_cols]
                max_da = float(np.max(np.abs(da_slice)))
                if max_da > (1.0 if strict_step else 2.0):
                    return True
            return False

        def _hybrid_state_out_of_bounds(state_vec: np.ndarray) -> bool:
            """检查交直流混合电网电压标幺值与交流相角是否超出物理允许范围。"""
            if v_cols.size:
                v_part = state_vec[v_cols]
                if not np.all(np.isfinite(v_part)):
                    return True
                if float(v_part.min()) < v_min_bound or float(v_part.max()) > v_max_bound:
                    return True
            if a_cols.size:
                a_part = state_vec[a_cols]
                if not np.all(np.isfinite(a_part)):
                    return True
                if float(a_part.min()) < a_min_bound or float(a_part.max()) > a_max_bound:
                    return True
            return False

        box_pin_cache: Dict = {}
        boundary_proj_streak = 0

        for iteration in range(1, self.max_iter + 1):
            if _hybrid_state_out_of_bounds(x):
                final_quantities_current = False
                break
            step_damping = 0.0
            # An accepted line-search candidate already contains h(x) and its
            # residual. Re-evaluate only after an untested fallback step.
            if z_est_dirty:
                start = time.perf_counter() if self.profile_enabled else None
                self.evaluate(x, measurements, out=z_est)
                if start is not None:
                    self._record_profile_time("solve.evaluate", time.perf_counter() - start)
                start = time.perf_counter() if self.profile_enabled else None
                self._measurement_residual(z, z_est, measurements, out=residual)
                objective = self._weighted_objective(weight, residual)
                if start is not None:
                    self._record_profile_time("solve.residual_objective", time.perf_counter() - start)
                z_est_dirty = False
            if active_weight_mask is not None:
                residual_inf = float(np.linalg.norm(residual[active_weight_mask], np.inf)) if effective_n_meas else 0.0
            else:
                residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0

            used_frozen_step = False
            used_woodbury_step = False
            used_sw_curv = False
            used_box_pin_step = False
            pinned_cols = np.empty(0, dtype=np.intp)
            cur_rhs_inf = np.inf
            rhs = None
            solved_via_aat = False
            if (
                iteration == 1
                and not woodbury_tried
                and use_aat_normal_solver
                and aat_solver is not None
                and aat_solver.last_cholmod_factor is not None
                and _woodbury_removed_rows is not None
                and _woodbury_prev_weights is not None
                and _woodbury_prev_H is not None
                and 1 <= len(_woodbury_removed_rows) <= wb_max_rank
                and is_sparse_matrix(_woodbury_prev_H)
                and int(_woodbury_prev_H.shape[0]) == n_meas
            ):
                woodbury_tried = True
                start = time.perf_counter() if self.profile_enabled else None
                rhs_wb = np.asarray(_woodbury_prev_H.T @ (weight * residual), dtype=np.float64).ravel()
                dx_wb, wb_solver_cand = solve_woodbury_after_row_removal(
                    _woodbury_prev_H,
                    _woodbury_prev_weights,
                    _woodbury_removed_rows,
                    rhs_wb,
                    aat_solver.last_cholmod_factor,
                    max_batch=wb_max_rank,
                    return_solver=True,
                    u_col_cache=_woodbury_u_cache,
                )
                if start is not None:
                    self._record_profile_time("solve.linear_solve", time.perf_counter() - start)
                if dx_wb is not None and (dx_wb.size == 0 or np.all(np.isfinite(dx_wb))):
                    corr_wb = float(np.max(np.abs(dx_wb))) if dx_wb.size else 0.0
                    dx = dx_wb
                    max_correction = corr_wb
                    H = _woodbury_prev_H
                    wb_solver = wb_solver_cand
                    wb_steps = 1
                    wb_local_shrunk = False
                    last_factor_x_disp = corr_wb
                    factor_is_current_weights = False
                    used_woodbury_step = True
                    used_frozen_step = True
                    solved_via_aat = True
            elif (
                wb_solver is not None
                and H is not None
                and wb_steps < scale_profile.woodbury_max_steps
                and (
                    (prev_max_correction < 5e-2 and (last_factor_x_disp < 5e-2 or (wb_local_shrunk and last_factor_x_disp < 0.4)))
                    or (wb_steps == 1 and prev_rms_correction < 2e-3 and 5e-2 <= prev_max_correction < 0.35 and last_factor_x_disp < 0.35)
                )
            ):
                is_local_probe = (wb_steps == 1 and prev_max_correction >= 5e-2)
                start = time.perf_counter() if self.profile_enabled else None
                rhs_wb = np.asarray(H.T @ (weight * residual), dtype=np.float64).ravel()
                dx_wb = wb_solver.solve(rhs_wb)
                if start is not None:
                    self._record_profile_time("solve.linear_solve", time.perf_counter() - start)
                if dx_wb is not None and (dx_wb.size == 0 or np.all(np.isfinite(dx_wb))):
                    corr_wb = float(np.max(np.abs(dx_wb))) if dx_wb.size else 0.0
                    shrink_ok = (
                        (corr_wb < self.tol or (corr_wb <= 0.25 * prev_max_correction and corr_wb < 5e-2))
                        if is_local_probe
                        else (corr_wb < self.tol or corr_wb <= 0.45 * prev_max_correction)
                    )
                    if shrink_ok:
                        dx = dx_wb
                        max_correction = corr_wb
                        wb_steps += 1
                        if is_local_probe:
                            wb_local_shrunk = True
                        last_factor_x_disp += corr_wb
                        used_woodbury_step = True
                        used_frozen_step = True
                        solved_via_aat = True
                    else:
                        wb_solver = None
                else:
                    wb_solver = None
            local_warm_frozen = (
                weight_masked
                and prev_rms_correction < scale_profile.local_warm_frozen_max_rms
                and prev_max_correction < scale_profile.local_warm_frozen_max_dx
                and frozen_steps < scale_profile.local_warm_frozen_max_steps
            )
            if (
                not used_woodbury_step
                and use_aat_normal_solver
                and aat_solver is not None
                and aat_plan is not None
                and H is not None
                and aat_solver.last_cholmod_factor is not None
                and prev_step_accepted
                and (
                    (prev_max_correction < 5e-2 and last_factor_x_disp < 0.15 and frozen_steps < scale_profile.frozen_max_steps)
                    or local_warm_frozen
                )
            ):
                start = time.perf_counter() if self.profile_enabled else None
                rhs_try = aat_plan.assemble_rhs_only(
                    residual,
                    weight,
                    uniform_weight=uniform_weight,
                    weights_are_uniform=weights_are_uniform,
                    assume_fixed_weights=True,
                    copy_rhs=False,
                )
                dx_try = aat_solver.solve_with_cached_factor(rhs_try)
                if start is not None:
                    self._record_profile_time("solve.linear_solve", time.perf_counter() - start)
                if dx_try is not None and (dx_try.size == 0 or np.all(np.isfinite(dx_try))):
                    corr_try = float(np.max(np.abs(dx_try))) if dx_try.size else 0.0
                    shrink_gate = 0.35 if (local_warm_frozen and prev_max_correction >= 5e-2) else 0.45
                    if corr_try < self.tol or corr_try <= shrink_gate * prev_max_correction:
                        dx = dx_try
                        max_correction = corr_try
                        used_frozen_step = True
                        solved_via_aat = True
                        frozen_steps += 1
                        last_factor_x_disp += corr_try

            if not used_frozen_step:
                if iteration == 1 and cached_initial_H is not None:
                    H = cached_initial_H
                    cached_initial_H = None
                else:
                    start = time.perf_counter() if self.profile_enabled else None
                    H = self.jacobian_sparse(x, measurements)
                    if start is not None:
                        self._record_profile_time("solve.jacobian", time.perf_counter() - start)

                if use_aat_normal_solver and aat_solver is not None and is_sparse_matrix(H):
                    start = time.perf_counter() if self.profile_enabled else None
                    if aat_plan is not None and not active_measurement_run and not aat_plan.matches(H):
                        aat_plan = None
                    if aat_plan is not None and active_measurement_run and (
                        tuple(H.shape) != aat_plan.shape or int(H.nnz) != int(aat_plan.h_indices.size)
                    ):
                        aat_plan = None
                    if aat_plan is None:
                        aat_plan = CholmodAAtNormalEquationPlan.from_jacobian(H)
                        aat_plan.prepare_fixed_weights(weight)
                    normal_operand, rhs = aat_plan.assemble(
                        H,
                        residual,
                        weight,
                        uniform_weight=uniform_weight,
                        weights_are_uniform=weights_are_uniform,
                        weighted_residual=None,
                        assume_fixed_weights=True,
                        copy_rhs=False,
                        assume_pattern_matches=active_measurement_run,
                    )
                    gain = None
                    if start is not None:
                        self._record_profile_time("solve.normal_equations", time.perf_counter() - start)
                    start = time.perf_counter() if self.profile_enabled else None
                    cur_rhs_inf = float(np.max(np.abs(rhs))) if rhs.size else 0.0
                    step_damping = float(force_lm_damping)
                    pinned_cols = detect_active_box_pinned_cols(
                        x, rhs, v_cols, v_min_bound, v_max_bound, margin=1e-4
                    )
                    try:
                        solve_operand = normal_operand
                        if sw_curv_rows.size > 0:
                            V_curv, has_sw_curv = build_switch_current_curvature_columns(
                                self.n_state,
                                sw_curv_rows,
                                sw_curv_col_re,
                                sw_curv_col_im,
                                x,
                                residual,
                                weight,
                            )
                            if has_sw_curv and V_curv is not None:
                                solve_operand = hstack([normal_operand, V_curv], format="csc")
                                used_sw_curv = True
                        if pinned_cols.size > 0:
                            dx_pin, pin_ok = solve_box_pinned_normal_equations(
                                solve_operand,
                                None,
                                rhs,
                                pinned_cols,
                                damping=step_damping,
                                solver_cache=box_pin_cache,
                            )
                            if pin_ok and step_damping == 0.0 and _hybrid_dx_needs_lm_guard(dx_pin, strict_step=True):
                                for beta_try in (1e-2, 1e-1, 1.0, 10.0):
                                    dx_lm, lm_ok = solve_box_pinned_normal_equations(
                                        solve_operand,
                                        None,
                                        rhs,
                                        pinned_cols,
                                        damping=beta_try,
                                        solver_cache=box_pin_cache,
                                    )
                                    if lm_ok:
                                        dx_pin = dx_lm
                                        step_damping = beta_try
                                        if not _hybrid_dx_needs_lm_guard(dx_pin, strict_step=False):
                                            break
                            if not pin_ok:
                                for beta_try in (1e-2, 1e-1, 1.0, 10.0):
                                    dx_lm, lm_ok = solve_box_pinned_normal_equations(
                                        solve_operand,
                                        None,
                                        rhs,
                                        pinned_cols,
                                        damping=beta_try,
                                        solver_cache=box_pin_cache,
                                    )
                                    if lm_ok:
                                        dx_pin = dx_lm
                                        pin_ok = True
                                        step_damping = beta_try
                                        break
                            if pin_ok:
                                dx = dx_pin
                                normal_factor_diag = None
                                solved_via_aat = True
                                used_box_pin_step = True
                            else:
                                pinned_cols = np.empty(0, dtype=np.intp)
                        if not used_box_pin_step:
                            use_fast_lm_handoff = step_damping == 0.0 and self.n_state >= 500
                            dx, normal_factor_diag = aat_solver.solve(
                                solve_operand,
                                rhs,
                                return_factor_diag=False,
                                damping=step_damping,
                                allow_slow_fallback=not use_fast_lm_handoff,
                            )
                            if dx.size == 0 or np.all(np.isfinite(dx)):
                                solved_via_aat = True
                            if step_damping == 0.0:
                                undamped_ok = (
                                    (dx.size == 0 or bool(np.all(np.isfinite(dx))))
                                    and getattr(aat_solver, "_active_backend", None) in ("pardiso", "cholmod")
                                )
                                last_undamped_spd_failed = not undamped_ok
                            # 若无阻尼牛顿步出现非有限值、显著过冲或试图把弱节点推入低/过电压危险区，提前采用分级 LM 阻尼扶正方向
                            if (solved_via_aat or use_fast_lm_handoff) and step_damping == 0.0 and _hybrid_dx_needs_lm_guard(dx, strict_step=True):
                                for beta_try in (1e-2, 1e-1, 1.0, 10.0):
                                    dx_lm, _ = aat_solver.solve(
                                        solve_operand,
                                        rhs,
                                        return_factor_diag=False,
                                        damping=beta_try,
                                        allow_slow_fallback=(self.n_state < 500 or beta_try >= 10.0),
                                    )
                                    if dx_lm.size == dx.size and np.all(np.isfinite(dx_lm)):
                                        dx = dx_lm
                                        step_damping = beta_try
                                        solved_via_aat = True
                                        if not _hybrid_dx_needs_lm_guard(dx, strict_step=False):
                                            break
                    except Exception:
                        use_aat_normal_solver = False
                        aat_solver = None
                        aat_plan = None
                        used_sw_curv = False
                        used_box_pin_step = False
                        pinned_cols = np.empty(0, dtype=np.intp)
                    if start is not None:
                        self._record_profile_time("solve.linear_solve", time.perf_counter() - start)

                if not solved_via_aat:
                    if normal_pattern is None and is_sparse_matrix(H):
                        normal_pattern = _normal_equation_structural_pattern(H)
                        if active_measurement_run:
                            self._active_normal_pattern = normal_pattern
                        if observability_cache is not None:
                            observability_cache["normal_pattern"] = normal_pattern
                    if weighted_residual is not None:
                        np.multiply(weight, residual, out=weighted_residual)
                    start = time.perf_counter() if self.profile_enabled else None
                    gain, rhs = build_normal_equations(
                        H,
                        residual,
                        weight,
                        uniform_weight=uniform_weight,
                        weights_are_uniform=weights_are_uniform,
                        weighted_residual=weighted_residual,
                        normal_pattern=normal_pattern,
                        assume_normal_pattern_matches=active_measurement_run,
                    )
                    if start is not None:
                        self._record_profile_time("solve.normal_equations", time.perf_counter() - start)
                    if getattr(self, "matrix_dump_dir", None) is not None:
                        self._dump_iteration_matrices(iteration, H, weight, gain)
                    start = time.perf_counter() if self.profile_enabled else None
                    cur_rhs_inf = float(np.max(np.abs(rhs))) if rhs.size else 0.0
                    solve_gain = gain
                    if sw_curv_rows.size > 0 and solve_gain is not None:
                        V_curv, has_sw_curv = build_switch_current_curvature_columns(
                            self.n_state,
                            sw_curv_rows,
                            sw_curv_col_re,
                            sw_curv_col_im,
                            x,
                            residual,
                            weight,
                        )
                        if has_sw_curv and V_curv is not None:
                            curv_term = V_curv @ V_curv.T
                            solve_gain = (
                                (solve_gain + curv_term).tocsc()
                                if is_sparse_matrix(solve_gain)
                                else (solve_gain + curv_term.toarray())
                            )
                            used_sw_curv = True
                    dx, normal_factor_diag = normal_solver.solve(solve_gain, rhs, return_factor_diag=final_diagnostics)
                    if start is not None:
                        self._record_profile_time("solve.linear_solve", time.perf_counter() - start)
                    if dx.size and not np.all(np.isfinite(dx)):
                        diag_scale = 1.0
                        if is_sparse_matrix(gain):
                            diag_values = np.asarray(gain.diagonal(), dtype=np.float64)
                            finite_diag = diag_values[np.isfinite(diag_values) & (diag_values > 0.0)]
                            if finite_diag.size:
                                diag_scale = float(np.median(finite_diag))
                            identity = sparse_eye(gain.shape[0], format="csc", dtype=np.float64)
                            for damping_factor in (1e-10, 1e-8, 1e-6, 1e-4):
                                damped_gain = gain + identity * (diag_scale * damping_factor)
                                dx_candidate, normal_factor_diag = NormalEquationSolver().solve(
                                    damped_gain,
                                    rhs,
                                    return_factor_diag=final_diagnostics,
                                )
                                if dx_candidate.size == 0 or np.all(np.isfinite(dx_candidate)):
                                    dx = dx_candidate
                                    self._record_profile_time("solve.damped_normal_fallback", damping_factor)
                                    break
                        else:
                            diag_values = np.diag(gain)
                            finite_diag = diag_values[np.isfinite(diag_values) & (diag_values > 0.0)]
                            if finite_diag.size:
                                diag_scale = float(np.median(finite_diag))
                            for damping_factor in (1e-10, 1e-8, 1e-6, 1e-4):
                                damped_gain = np.array(gain, dtype=np.float64, copy=True)
                                damped_gain[np.diag_indices_from(damped_gain)] += diag_scale * damping_factor
                                dx_candidate, normal_factor_diag = NormalEquationSolver().solve(
                                    damped_gain,
                                    rhs,
                                    return_factor_diag=final_diagnostics,
                                )
                                if dx_candidate.size == 0 or np.all(np.isfinite(dx_candidate)):
                                    dx = dx_candidate
                                    self._record_profile_time("solve.damped_normal_fallback", damping_factor)
                                    break
                frozen_steps = 0
                wb_solver = None
                factor_is_current_weights = (not used_sw_curv) and (not used_box_pin_step)
                if dx.size and not np.all(np.isfinite(dx)):
                    final_quantities_current = False
                    break
                max_correction = float(np.max(np.abs(dx))) if dx.size else 0.0
                last_factor_x_disp = max_correction

            last_solved_via_aat = solved_via_aat
            damped_kkt_ok = False
            if (
                not used_frozen_step
                and step_damping > 0.0
                and max_correction < 1e-3
                and rhs is not None
                and rhs.size > 0
            ):
                diag_g = None
                if solved_via_aat and normal_operand is not None:
                    diag_g = (
                        aat_solver.unregularized_diagonal()
                        if (aat_solver is not None and not used_box_pin_step)
                        else None
                    )
                    if diag_g is None:
                        diag_g = np.asarray(normal_operand.power(2).sum(axis=1), dtype=np.float64).ravel()
                elif gain is not None:
                    diag_g = np.asarray(gain.diagonal() if is_sparse_matrix(gain) else np.diag(gain), dtype=np.float64).ravel()
                if diag_g is not None and diag_g.size == rhs.size:
                    rhs_kkt = rhs.copy() if pinned_cols.size > 0 else rhs
                    if pinned_cols.size > 0:
                        rhs_kkt[pinned_cols] = 0.0
                    cur_rhs_inf = float(np.max(np.abs(rhs_kkt) / np.maximum(diag_g, 1.0)))
                damped_kkt_ok = cur_rhs_inf <= max(1e-4, 100.0 * self.tol)
            if max_correction < self.tol and (used_frozen_step or step_damping == 0.0 or damped_kkt_ok):
                converged = True
                final_quantities_current = factor_is_current_weights and ((not used_frozen_step) or (last_factor_x_disp <= 5e-2))
                if verbose:
                    _print_iteration(iteration, objective, residual_inf, max_correction, None, True)
                break
            # 物理信赖域步长限幅：防止单步过冲把盲节点电压或相角推入非物理发散区域
            step_scale = 1.0
            if v_cols.size and dx.size:
                max_v_dx = float(np.max(np.abs(dx[v_cols])))
                if max_v_dx > 0.35:
                    step_scale = min(step_scale, 0.35 / max_v_dx)
            if a_cols.size and dx.size:
                max_a_dx = float(np.max(np.abs(dx[a_cols])))
                if max_a_dx > 0.75:
                    step_scale = min(step_scale, 0.75 / max_a_dx)
            base_step_scale = step_scale
            accepted = False
            accepted_step = None
            used_boundary_proj = False
            added_wall_nodes = False
            nonfinite_candidates = 0
            oob_candidates = 0
            previous_objective = objective
            x_before_step = x
            ls_slope = (
                -float(np.dot(rhs, dx))
                if (not used_frozen_step and rhs is not None and rhs.size == dx.size)
                else 0.0
            )
            max_ls_trials = 1 if (used_frozen_step and not used_woodbury_step) else 16
            for _attempt in range(max_ls_trials):
                candidate = x + step_scale * dx
                if _hybrid_state_out_of_bounds(candidate):
                    oob_candidates += 1
                    # 严禁为避开单个越界节点把全局步长压成贴墙微步空转：最多折半 3 次即尝试投影接入活跃集或切向 LM 阻尼
                    if oob_candidates >= 3:
                        if (
                            boundary_proj_streak < 6
                            and v_cols.size
                        ):
                            proj_scales = (base_step_scale, step_scale) if base_step_scale > step_scale * 1.01 else (step_scale,)
                            for ps in proj_scales:
                                proj_cand = x + ps * dx
                                proj_cand[v_cols] = np.clip(proj_cand[v_cols], v_min_bound + 1e-6, v_max_bound - 1e-6)
                                if not _hybrid_state_out_of_bounds(proj_cand):
                                    self.evaluate(proj_cand, measurements, out=cand_z_est)
                                    self._measurement_residual(z, cand_z_est, measurements, out=cand_residual)
                                    proj_obj = self._weighted_objective(weight, cand_residual)
                                    if np.isfinite(proj_obj) and proj_obj < objective - max(1e-12, 1e-5 * (abs(objective) + 1.0)):
                                        prev_v_s = x[v_cols]
                                        new_v_s = proj_cand[v_cols]
                                        prev_wall = int(np.count_nonzero((prev_v_s <= v_min_bound + 1e-4) | (prev_v_s >= v_max_bound - 1e-4)))
                                        new_wall = int(np.count_nonzero((new_v_s <= v_min_bound + 1e-4) | (new_v_s >= v_max_bound - 1e-4)))
                                        x = proj_cand
                                        objective = proj_obj
                                        z_est, cand_z_est = cand_z_est, z_est
                                        residual, cand_residual = cand_residual, residual
                                        if active_weight_mask is not None:
                                            residual_inf = float(np.linalg.norm(residual[active_weight_mask], np.inf)) if effective_n_meas else 0.0
                                        else:
                                            residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
                                        accepted_step = ps
                                        used_boundary_proj = True
                                        added_wall_nodes = new_wall > prev_wall
                                        boundary_proj_streak = 0 if added_wall_nodes else (boundary_proj_streak + 1)
                                        accepted = True
                                        break
                        break
                    step_scale *= 0.5
                    continue
                start = time.perf_counter() if self.profile_enabled else None
                self.evaluate(candidate, measurements, out=cand_z_est)
                if start is not None:
                    self._record_profile_time("solve.evaluate", time.perf_counter() - start)
                start = time.perf_counter() if self.profile_enabled else None
                self._measurement_residual(z, cand_z_est, measurements, out=cand_residual)
                candidate_objective = self._weighted_objective(weight, cand_residual)
                if start is not None:
                    self._record_profile_time("solve.residual_objective", time.perf_counter() - start)
                objective_tol = max(1e-12, 1e-10 * (abs(objective) + 1.0))
                if np.isfinite(candidate_objective) and candidate_objective <= objective + objective_tol:
                    x = candidate
                    objective = candidate_objective
                    # Swap the accepted candidate buffers into the main role.
                    z_est, cand_z_est = cand_z_est, z_est
                    residual, cand_residual = cand_residual, residual
                    if active_weight_mask is not None:
                        residual_inf = float(np.linalg.norm(residual[active_weight_mask], np.inf)) if effective_n_meas else 0.0
                    else:
                        residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
                    accepted_step = step_scale
                    accepted = True
                    break
                if not np.isfinite(candidate_objective):
                    nonfinite_candidates += 1
                    if nonfinite_candidates >= 4:
                        break
                    step_scale *= 0.5
                    continue
                if ls_slope < 0.0:
                    denom = 2.0 * (candidate_objective - objective - ls_slope * step_scale)
                    if np.isfinite(denom) and denom > 1e-30:
                        s_quad = (-ls_slope * step_scale * step_scale) / denom
                        step_scale = float(np.clip(s_quad, 0.1 * step_scale, 0.5 * step_scale))
                    else:
                        step_scale *= 0.5
                else:
                    step_scale *= 0.5
            if not accepted:
                aa_accel.reset()
                if used_frozen_step:
                    prev_step_accepted = False
                    if used_woodbury_step:
                        wb_solver = None
                    else:
                        frozen_steps = max(scale_profile.frozen_max_steps, scale_profile.local_warm_frozen_max_steps)
                    continue
                if force_lm_damping < 100.0 and solved_via_aat and aat_solver is not None:
                    force_lm_damping = max(force_lm_damping * 10.0, 1e-2)
                    backtrack_streak = 0
                    prev_step_accepted = False
                    continue
                # 阻尼已达上限 (100.0) 且任意微步均无法再降目标函数：若核心电压/相角处于健康边界内
                # 且有效无约束方向的归一化梯度极小（扣除紧贴 voltage_floor 且指向墙外的有效不等式约束分量），
                # 确认已到达局部极小驻点并优雅收敛
                if (
                    not _hybrid_state_out_of_bounds(x)
                    and max_correction <= 5e-2
                    and rhs is not None
                    and rhs.size > 0
                ):
                    diag_g_st = aat_solver.unregularized_diagonal() if (solved_via_aat and aat_solver is not None) else None
                    if diag_g_st is None and solved_via_aat and normal_operand is not None:
                        diag_g_st = np.asarray(normal_operand.power(2).sum(axis=1), dtype=np.float64).ravel()
                    if diag_g_st is not None and diag_g_st.size == rhs.size:
                        eff_rhs_st = rhs.copy()
                        if v_cols.size:
                            v_pin_lo = v_cols[(x[v_cols] <= float(self.voltage_floor) + 1e-3) & (eff_rhs_st[v_cols] < 0.0)]
                            v_pin_hi = v_cols[(x[v_cols] >= float(self.voltage_max) - 1e-3) & (eff_rhs_st[v_cols] > 0.0)]
                            if v_pin_lo.size:
                                eff_rhs_st[v_pin_lo] = 0.0
                            if v_pin_hi.size:
                                eff_rhs_st[v_pin_hi] = 0.0
                        scaled_vec_st = np.abs(eff_rhs_st) / np.maximum(diag_g_st, 1.0)
                        va_g_st = 0.0
                        if v_cols.size:
                            va_g_st = max(va_g_st, float(np.max(scaled_vec_st[v_cols])))
                        if a_cols.size:
                            va_g_st = max(va_g_st, float(np.max(scaled_vec_st[a_cols])))
                        if va_g_st <= max(5e-3, 5000.0 * self.tol):
                            converged = True
                final_quantities_current = factor_is_current_weights
                if verbose:
                    _print_iteration(iteration, objective, residual_inf, max_correction, step_scale, converged)
                break
            if _hybrid_state_out_of_bounds(x):
                final_quantities_current = False
                break
            if not used_boundary_proj:
                boundary_proj_streak = 0
            prev_step_accepted = (
                False
                if (used_woodbury_step or used_sw_curv or used_box_pin_step)
                else (True if (used_frozen_step or step_damping == 0.0) else False)
            )
            last_step_max_correction = prev_max_correction
            prev_max_correction = max_correction
            prev_rms_correction = float(np.sqrt(float(np.dot(dx, dx)) / dx.size)) if dx.size else 0.0
            objective_change = abs(previous_objective - objective)
            stagnation_tol = max(1e-14, 1e-10 * (abs(previous_objective) + 1.0))
            gn_actual_drop = previous_objective - objective
            if (
                accepted_step is not None
                and accepted_step >= 0.95
                and not used_boundary_proj
                and not used_frozen_step
                and max_correction > 10.0 * self.tol
            ):
                aa_ratio = (
                    (max_correction / last_step_max_correction)
                    if (np.isfinite(last_step_max_correction) and last_step_max_correction > 0.0)
                    else 0.0
                )
                x_aa = aa_accel.update_and_propose(
                    x_before_step,
                    x,
                    step_ratio=aa_ratio,
                    damping=step_damping,
                    allow_propose=(max_correction < 0.5),
                )
                if x_aa is not None and not _hybrid_state_out_of_bounds(x_aa):
                    self.evaluate(x_aa, measurements, out=cand_z_est)
                    self._measurement_residual(z, cand_z_est, measurements, out=cand_residual)
                    aa_obj = self._weighted_objective(weight, cand_residual)
                    if np.isfinite(aa_obj) and aa_obj < objective - max(1e-12, 1e-10 * (abs(objective) + 1.0)):
                        aa_jump = float(np.max(np.abs(x_aa - x))) if x_aa.size else 0.0
                        x = x_aa
                        z_est, cand_z_est = cand_z_est, z_est
                        residual, cand_residual = cand_residual, residual
                        objective = aa_obj
                        objective_change = abs(previous_objective - objective)
                        if active_weight_mask is not None:
                            residual_inf = float(np.linalg.norm(residual[active_weight_mask], np.inf)) if effective_n_meas else 0.0
                        else:
                            residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
                        prev_step_accepted = False
                        aa_accel.on_accepted(step_norm=aa_jump)
                    else:
                        aa_accel.on_rejected()
                elif x_aa is not None:
                    aa_accel.on_rejected()
            else:
                aa_accel.reset()

            # 支柱 1：带退避倍率 ν 与反极限环下界记忆的 Nielsen (1999) 连续阻尼更新律（严格使用纯 GN/LM 单步降幅）
            if curvature_damping_active and step_damping > 0.0:
                step_solver_lm = box_pin_cache.get("solver") if used_box_pin_step else (aat_solver if solved_via_aat else None)
                diag_reg_lm = (
                    step_solver_lm.unregularized_diagonal()
                    if (step_solver_lm is not None and bool(getattr(step_solver_lm, "_last_factor_scaled", False)))
                    else None
                )
                new_d, _rho_lm = tr_ctrl.update(
                    step_damping,
                    rhs,
                    dx,
                    gn_actual_drop,
                    accepted_step if accepted_step is not None else 0.0,
                    current_obj=objective,
                    half_obj=True,
                    diag_reg=diag_reg_lm,
                    min_damping=0.0,
                    max_damping=10.0,
                    zero_cutoff=1e-3,
                    dx_growing=(max_correction > 1.2 * last_step_max_correction),
                )
                force_lm_damping = new_d
                if force_lm_damping <= 0.0:
                    curvature_damping_active = False
                    backtrack_streak = 0
                    fullstep_osc_streak = 0
                if abs(new_d - step_damping) > max(1e-12, 0.05 * max(new_d, step_damping)):
                    prev_step_accepted = False
            if accepted_step is not None and accepted_step < 1e-3 and max_correction > 1e-2:
                micro_step_streak += 1
                fullstep_osc_streak = 0
                if (accepted_step < 1e-4 and objective_change <= stagnation_tol) or micro_step_streak >= 3:
                    if force_lm_damping >= 100.0:
                        final_quantities_current = factor_is_current_weights
                        break
                    force_lm_damping = min(100.0, max(force_lm_damping * 10.0, 1e-2))
                    curvature_damping_active = True
                    tr_ctrl.notify_damping_activated(force_lm_damping, objective)
                    micro_step_streak = 0
                    backtrack_streak = 0
                    prev_step_accepted = False
            elif accepted_step is not None and accepted_step < 0.5 and max_correction > 10.0 * self.tol and not added_wall_nodes:
                micro_step_streak = 0
                fullstep_osc_streak = 0
                if accepted_step < 0.05 and objective_change <= max(1e-12, 1e-4 * (abs(previous_objective) + 1.0)):
                    slow_micro_streak += 1
                    if slow_micro_streak >= 10 and solved_via_aat and aat_solver is not None:
                        force_lm_damping = min(10.0, max(float(force_lm_damping) * 5.0, 1.0))
                        curvature_damping_active = True
                        tr_ctrl.notify_damping_activated(force_lm_damping, objective)
                        slow_micro_streak = 0
                        prev_step_accepted = False
                else:
                    slow_micro_streak = 0
                backtrack_streak += 4
                if backtrack_streak >= 12 and solved_via_aat and aat_solver is not None:
                    if force_lm_damping < 10.0:
                        hint_base = min(2.0, max(0.1, float(_hint_lm_damping))) if float(_hint_lm_damping) > 0.0 else 0.1
                        force_lm_damping = min(10.0, max(force_lm_damping * 5.0, hint_base))
                        curvature_damping_active = True
                        tr_ctrl.notify_damping_activated(force_lm_damping, objective)
                        prev_step_accepted = False
                    backtrack_streak = 0
            elif accepted_step is not None and accepted_step >= 0.5:
                micro_step_streak = 0
                if max_correction < 0.5 * last_step_max_correction:
                    backtrack_streak = 0
                    fullstep_osc_streak = 0
                else:
                    backtrack_streak = max(0, backtrack_streak - 1)
                    if (
                        (not curvature_damping_active)
                        and max_correction > 1e-3
                        and max_correction >= 0.85 * last_step_max_correction
                        and objective_change <= 1e-3 * (abs(previous_objective) + 1.0)
                    ):
                        fullstep_osc_streak += 1
                        if fullstep_osc_streak >= 4 and solved_via_aat and aat_solver is not None:
                            hint_base = min(2.0, max(0.1, float(_hint_lm_damping))) if float(_hint_lm_damping) > 0.0 else 0.1
                            force_lm_damping = min(10.0, max(force_lm_damping * 5.0, hint_base))
                            curvature_damping_active = True
                            tr_ctrl.notify_damping_activated(force_lm_damping, objective)
                            backtrack_streak = 0
                            fullstep_osc_streak = 0
                            prev_step_accepted = False
                    else:
                        fullstep_osc_streak = 0
                if not curvature_damping_active:
                    force_lm_damping = 0.0
            elif accepted_step is not None and accepted_step >= 1e-2:
                micro_step_streak = 0
                if not curvature_damping_active:
                    force_lm_damping = 0.0
            damped_kkt_tol = max(5e-4, 500.0 * self.tol)
            damped_obj_tol = max(1e-12, 1e-7 * (abs(previous_objective) + 1.0))
            damped_kkt_stagnation_ok = (
                curvature_damping_active
                and step_damping > 0.0
                and cur_rhs_inf <= damped_kkt_tol
                and max_correction < 1e-3
                and objective_change <= damped_obj_tol
            )
            if (
                curvature_damping_active
                and max_correction < 1e-3
                and cur_rhs_inf > damped_kkt_tol
            ):
                if cur_rhs_inf < 0.85 * curv_stagnation_best_g:
                    curv_stagnation_best_g = cur_rhs_inf
                    curv_stagnation_streak = 0
                else:
                    curv_stagnation_streak += 1
                    if (
                        curv_stagnation_streak >= 12
                        and cur_rhs_inf <= max(1e-3, 1000.0 * self.tol)
                        and max_correction < 1e-3
                        and objective_change <= damped_obj_tol
                    ):
                        damped_kkt_stagnation_ok = True
                    elif curv_stagnation_streak >= (12 if cur_rhs_inf > 2e-3 else 40):
                        final_quantities_current = factor_is_current_weights
                        break
            elif max_correction >= 1e-3:
                curv_stagnation_streak = 0
                curv_stagnation_best_g = np.inf
            # 极高盲区平底谷驻点检测
            if (
                not damped_kkt_stagnation_ok
                and curvature_damping_active
                and not used_frozen_step
                and step_damping > 0.0
                and max_correction < 5e-2
                and prev_rms_correction <= 1e-3
                and objective_change <= max(1e-12, 1e-5 * (abs(previous_objective) + 1.0))
                and rhs is not None
                and rhs.size > 0
            ):
                va_max_dx = 0.0
                if v_cols.size and dx.size:
                    va_max_dx = max(va_max_dx, float(np.max(np.abs(dx[v_cols]))))
                if a_cols.size and dx.size:
                    va_max_dx = max(va_max_dx, float(np.max(np.abs(dx[a_cols]))))
                if not v_cols.size and not a_cols.size:
                    va_max_dx = max_correction
                diag_g_fv = None
                if solved_via_aat and normal_operand is not None:
                    diag_g_fv = aat_solver.unregularized_diagonal() if aat_solver is not None else None
                    if diag_g_fv is None:
                        diag_g_fv = np.asarray(normal_operand.power(2).sum(axis=1), dtype=np.float64).ravel()
                elif gain is not None:
                    diag_g_fv = np.asarray(gain.diagonal() if is_sparse_matrix(gain) else np.diag(gain), dtype=np.float64).ravel()
                if diag_g_fv is not None and diag_g_fv.size == rhs.size:
                    scaled_vec = np.abs(rhs) / np.maximum(diag_g_fv, 1.0)
                    scaled_g = float(np.max(scaled_vec))
                    va_scaled_g = 0.0
                    if v_cols.size:
                        va_scaled_g = max(va_scaled_g, float(np.max(scaled_vec[v_cols])))
                    if a_cols.size:
                        va_scaled_g = max(va_scaled_g, float(np.max(scaled_vec[a_cols])))
                    if not v_cols.size and not a_cols.size:
                        va_scaled_g = scaled_g
                else:
                    scaled_g = cur_rhs_inf
                    va_scaled_g = cur_rhs_inf
                if (
                    va_max_dx <= 1e-3
                    and va_scaled_g <= max(1e-4, 100.0 * self.tol)
                    and scaled_g <= max(5e-3, 5000.0 * self.tol)
                ):
                    flat_valley_streak += 1
                    if flat_valley_streak >= 6:
                        damped_kkt_stagnation_ok = True
                else:
                    flat_valley_streak = 0
            else:
                flat_valley_streak = 0
            # 高阻尼微步平底谷检测
            if (
                (not damped_kkt_stagnation_ok)
                and step_damping >= 1.0
                and accepted_step is not None
                and (accepted_step * max_correction) <= 1e-2
                and objective_change <= max(1e-12, 1e-5 * (abs(previous_objective) + 1.0))
                and not _hybrid_state_out_of_bounds(x)
            ):
                high_damp_flat_streak += 1
                if high_damp_flat_streak >= 8:
                    damped_kkt_stagnation_ok = True
            else:
                high_damp_flat_streak = 0
            # 支柱 3：统一 Projected KKT 一阶最优性早停（无论是否带阻尼或贴墙，循环内外采用同一准则）
            va_max_dx_bk = 0.0
            if v_cols.size and dx.size:
                va_max_dx_bk = max(va_max_dx_bk, float(np.max(np.abs(dx[v_cols]))))
            if a_cols.size and dx.size:
                va_max_dx_bk = max(va_max_dx_bk, float(np.max(np.abs(dx[a_cols]))))
            if not v_cols.size and not a_cols.size:
                va_max_dx_bk = max_correction
            box_kkt_obj_tol = max(1e-12, 1e-5 * (abs(previous_objective) + 1.0))
            if (
                not damped_kkt_stagnation_ok
                and not used_frozen_step
                and va_max_dx_bk <= 2e-2
                and max_correction <= 0.1
                and objective_change <= box_kkt_obj_tol
                and rhs is not None
                and rhs.size == x.size
                and not _hybrid_state_out_of_bounds(x)
            ):
                box_kkt_stagnation_streak += 1
                need_bk_streak = 3 if objective_change <= damped_obj_tol else 5
                if box_kkt_stagnation_streak >= need_bk_streak:
                    diag_g_bk = None
                    if solved_via_aat and normal_operand is not None:
                        diag_g_bk = aat_solver.unregularized_diagonal() if aat_solver is not None else None
                        if diag_g_bk is None:
                            diag_g_bk = np.asarray(normal_operand.power(2).sum(axis=1), dtype=np.float64).ravel()
                    elif gain is not None:
                        diag_g_bk = np.asarray(gain.diagonal() if is_sparse_matrix(gain) else np.diag(gain), dtype=np.float64).ravel()
                    if diag_g_bk is not None and diag_g_bk.size == rhs.size:
                        core_cols_bk = (
                            np.concatenate([v_cols, a_cols])
                            if (v_cols.size and a_cols.size)
                            else (v_cols if v_cols.size else (a_cols if a_cols.size else None))
                        )
                        va_g_bk, _ = projected_box_kkt_residual(
                            x, rhs, diag_g_bk, v_cols, float(self.voltage_floor), float(self.voltage_max), core_cols=core_cols_bk
                        )
                        if va_g_bk <= max(1e-4, 100.0 * self.tol):
                            damped_kkt_stagnation_ok = True
            else:
                box_kkt_stagnation_streak = 0
            practically_converged = (
                (used_frozen_step or step_damping == 0.0 or damped_kkt_ok)
                and max_correction < 10.0 * self.tol
                and objective_change <= stagnation_tol
            ) or damped_kkt_stagnation_ok
            if verbose:
                _print_iteration(iteration, objective, residual_inf, max_correction, accepted_step, False)
            if (max_correction < self.tol and (used_frozen_step or step_damping == 0.0 or damped_kkt_ok)) or practically_converged:
                converged = True
                if (
                    factor_is_current_weights
                    and last_solved_via_aat
                    and aat_solver is not None
                    and aat_solver.last_cholmod_factor is not None
                    and last_factor_x_disp <= 5e-2
                ):
                    final_quantities_current = True
                break
            coarse_v_healthy = True
            if v_cols.size > 0:
                xv_c = x[v_cols]
                coarse_v_healthy = (
                    float(np.min(xv_c)) > float(self.voltage_floor) + 0.05
                    and float(np.max(xv_c)) < float(self.voltage_max) - 0.05
                )
            if (
                accepted
                and coarse_lb_sq is not None
                and coarse_non_pseudo_mask is not None
                and coarse_v_healthy
                and iteration >= 2
                and not used_woodbury_step
                and prev_rms_correction < 2e-2
                and H is not None
                and (
                    used_box_pin_step
                    or (
                        last_solved_via_aat
                        and aat_solver is not None
                        and (aat_solver.last_cholmod_factor is not None or bool(getattr(aat_solver, "_last_factor_scaled", False)))
                    )
                )
            ):
                # 前期仍存在大批显性大坏数据时，主体已稳定即可安全提前结束注定被废弃的尾部磨步
                strong_bad_cnt = int(
                    np.count_nonzero(
                        (weight * (residual * residual) >= coarse_lb_sq) & coarse_non_pseudo_mask
                    )
                )
                if strong_bad_cnt >= 10:
                    coarse_stopped = True
                    final_quantities_current = not used_box_pin_step
                    break

        if z_est_dirty:
            start = time.perf_counter() if self.profile_enabled else None
            self.evaluate(x, measurements, out=z_est)
            if start is not None:
                self._record_profile_time("solve.evaluate", time.perf_counter() - start)
            start = time.perf_counter() if self.profile_enabled else None
            self._measurement_residual(z, z_est, measurements, out=residual)
            objective = self._weighted_objective(weight, residual)
            residual_inf = float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0
            if start is not None:
                self._record_profile_time("solve.final_residual_objective", time.perf_counter() - start)
        # 有界约束 KKT 驻点核验：扣除紧贴 voltage_floor 且指向墙外的有效下界约束分量后，
        # 若全部核心电压/相角归一化梯度均已满足容差，确认已收敛至有界约束最优解
        if (
            (not converged)
            and (not coarse_stopped)
            and iteration >= 3
            and np.isfinite(objective)
            and max_correction <= 0.1
            and H is not None
            and not _hybrid_state_out_of_bounds(x)
        ):
            if H.shape[0] == residual.size and weight.size == residual.size:
                rhs_chk = np.asarray(H.T @ (weight * residual), dtype=np.float64)
                if is_sparse_matrix(H):
                    dg_chk = np.asarray(H.power(2).T @ weight, dtype=np.float64).ravel()
                else:
                    dg_chk = np.asarray(np.sum((H * H) * weight[:, None], axis=0), dtype=np.float64).ravel()
                if rhs_chk.size == x.size and dg_chk.size == x.size:
                    if v_cols.size:
                        v_pin_lo = v_cols[(x[v_cols] <= float(self.voltage_floor) + 1e-4) & (rhs_chk[v_cols] < 0.0)]
                        v_pin_hi = v_cols[(x[v_cols] >= float(self.voltage_max) - 1e-4) & (rhs_chk[v_cols] > 0.0)]
                        if v_pin_lo.size:
                            rhs_chk[v_pin_lo] = 0.0
                        if v_pin_hi.size:
                            rhs_chk[v_pin_hi] = 0.0
                    sv_chk = np.abs(rhs_chk) / np.maximum(dg_chk, 1.0)
                    va_g_chk = 0.0
                    if v_cols.size:
                        va_g_chk = max(va_g_chk, float(np.max(sv_chk[v_cols])))
                    if a_cols.size:
                        va_g_chk = max(va_g_chk, float(np.max(sv_chk[a_cols])))
                    if va_g_chk <= max(1e-4, 100.0 * self.tol):
                        converged = True
        if box_pin_cache.get("solver") is not None:
            try:
                box_pin_cache["solver"]._reset_pardiso()
            except Exception:
                pass
            box_pin_cache.clear()
        cached_cholmod_factor = None
        factor_deferred = False
        inplace_factor_modified = bool(factor_is_current_weights)
        if final_diagnostics:
            if not final_quantities_current or H is None:
                start = time.perf_counter() if self.profile_enabled else None
                H = self.jacobian_sparse(x, measurements)
                if start is not None:
                    self._record_profile_time("solve.jacobian", time.perf_counter() - start)
                if (
                    _defer_factor_refresh
                    and converged
                    and not factor_is_current_weights
                    and wb_solver is not None
                    and last_factor_x_disp <= 5e-2
                    and last_solved_via_aat
                    and aat_solver is not None
                    and aat_solver.last_cholmod_factor is not None
                ):
                    # 跨轮持久锚点低秩 Woodbury：延迟 CHOLMOD 数值分解，保留锚点原封不动，交付低秩求解器供筛查复用
                    gain = None
                    factor_deferred = True
                    cached_cholmod_factor = wb_solver
                    self._last_estimate_cholmod_factor = (id(H), cached_cholmod_factor)
                elif (
                    coarse_lb_sq is not None
                    and coarse_non_pseudo_mask is not None
                    and converged
                    and factor_is_current_weights
                    and prev_rms_correction < 2e-2
                    and last_solved_via_aat
                    and aat_solver is not None
                    and aat_solver.last_cholmod_factor is not None
                    and int(
                        np.count_nonzero(
                            (weight * (residual * residual) >= coarse_lb_sq) & coarse_non_pseudo_mask
                        )
                    )
                    >= 10
                ):
                    # 前期批删轮已收敛但仍余大批显性大坏数据：当前因子下一轮立刻被批删丢弃，跳过冗余终末重分解
                    gain = None
                    coarse_stopped = True
                    cached_cholmod_factor = aat_solver.last_cholmod_factor
                    self._last_estimate_cholmod_factor = (id(H), cached_cholmod_factor)
                elif (
                    last_solved_via_aat
                    and aat_solver is not None
                    and aat_plan is not None
                    and aat_solver.last_cholmod_factor is not None
                ):
                    try:
                        start = time.perf_counter() if self.profile_enabled else None
                        normal_operand, _ = aat_plan.assemble(
                            H,
                            residual,
                            weight,
                            uniform_weight=uniform_weight,
                            weights_are_uniform=weights_are_uniform,
                            weighted_residual=None,
                            assume_fixed_weights=True,
                            copy_rhs=False,
                            assume_pattern_matches=active_measurement_run,
                        )
                        aat_solver.last_cholmod_factor.cholesky_AAt_inplace(normal_operand)
                        aat_solver._last_normal_operand = normal_operand
                        cached_cholmod_factor = aat_solver.last_cholmod_factor
                        inplace_factor_modified = True
                        self._last_estimate_cholmod_factor = (id(H), cached_cholmod_factor)
                        if start is not None:
                            self._record_profile_time("solve.linear_solve", time.perf_counter() - start)
                    except Exception:
                        pass
            elif (
                last_solved_via_aat
                and aat_solver is not None
                and aat_solver.last_cholmod_factor is not None
            ):
                cached_cholmod_factor = aat_solver.last_cholmod_factor
                self._last_estimate_cholmod_factor = (id(H), cached_cholmod_factor)
            if gain is None and cached_cholmod_factor is None and self.n_state <= 5000:
                if normal_pattern is None and is_sparse_matrix(H):
                    normal_pattern = _normal_equation_structural_pattern(H)
                    if active_measurement_run:
                        self._active_normal_pattern = normal_pattern
                if weighted_residual is not None:
                    np.multiply(weight, residual, out=weighted_residual)
                start = time.perf_counter() if self.profile_enabled else None
                gain, _rhs = build_normal_equations(
                    H,
                    residual,
                    weight,
                    uniform_weight=uniform_weight,
                    weights_are_uniform=weights_are_uniform,
                    weighted_residual=weighted_residual,
                    normal_pattern=normal_pattern,
                    assume_normal_pattern_matches=active_measurement_run,
                )
                if start is not None:
                    self._record_profile_time("solve.normal_equations", time.perf_counter() - start)
        else:
            H = None
            gain = None
        if observability is None:
            if (converged or coarse_stopped) and aat_solver is not None and aat_solver.last_cholmod_factor is not None:
                observability = ObservabilityResult(
                    observable=True,
                    rank=self.n_state,
                    state_count=self.n_state,
                    measurement_count=effective_n_meas,
                    deficiency=0,
                    singular_values=np.array([], dtype=np.float64),
                    weak_states=[],
                )
            else:
                start = time.perf_counter() if self.profile_enabled else None
                obs_meas = (
                    take_measurement_view(measurements, np.flatnonzero(active_weight_mask).astype(np.int64, copy=False))
                    if active_weight_mask is not None
                    else measurements
                )
                observability = self.observability_analysis(x, obs_meas)
                if start is not None:
                    self._record_profile_time("solve.observability", time.perf_counter() - start)
        if solve_profile_start is not None:
            self._record_profile_time("solve.total", time.perf_counter() - solve_profile_start)
        self._last_estimate_aat_runtime = (
            (aat_solver, aat_plan) if (aat_solver is not None and aat_plan is not None) else None
        )
        array_only_result = bool(getattr(self, "_array_only_estimate_result", False))
        result_measurements = []
        if not array_only_result:
            table = getattr(measurements, "table", None)
            result_measurements = measurements if table is not None and int(table.idx.size) == len(measurements) else list(measurements)
        # Copy the buffer-backed vectors so callers (and the next estimate() invocation)
        # can't accidentally see them mutate as line-search scratch space later on.
        estimate_result = EstimateResult(
            converged=converged,
            iterations=iteration,
            objective=objective,
            max_correction=max_correction,
            residual_inf=residual_inf,
            x=x,
            z_est=z_est.copy(),
            residual=residual.copy(),
            H=H,
            gain=gain,
            measurements=result_measurements,
            observability=observability,
            measurement_table=getattr(measurements, "table", None),
        )
        if coarse_stopped:
            estimate_result._coarse_stopped = True
        if last_undamped_spd_failed:
            estimate_result._undamped_spd_failed = True
        if factor_deferred:
            estimate_result._factor_deferred = True
        if curvature_damping_active and force_lm_damping > 0.0:
            estimate_result._final_lm_damping = float(force_lm_damping)
        estimate_result._inplace_factor_modified = inplace_factor_modified
        return estimate_result

    def identify_bad_data(
        self,
        result: EstimateResult,
        threshold: Optional[float] = None,
    ) -> Tuple[List[BadDataItem], np.ndarray]:
        profile_start = time.perf_counter() if self.profile_enabled else None
        # 多能耦合模式的结果（含全局联合结果与各流体子网局部切片）严禁委派给单侧 AC/DC 子估计器
        has_fluid_subnets = bool(getattr(self, "fluid_estimators", None))
        is_multi_energy_result = has_fluid_subnets and (
            getattr(result, "multi_energy_rows", None) is not None
            or int(getattr(result.x, "size", -1)) == int(getattr(self, "multi_energy_state_count", -2))
            or getattr(result, "measurement_plan_tables", None) is None
        )
        delegate = None if is_multi_energy_result else self._delegate()
        if delegate is not None:
            return delegate.identify_bad_data(result, threshold)
        base_threshold = float(self.params.bad_threshold if threshold is None else threshold)
        measurement_table = getattr(result, "measurement_table", None)
        table_weight = getattr(measurement_table, "weight", None)
        if table_weight is not None and np.asarray(table_weight).size == result.residual.size:
            weights = np.asarray(table_weight, dtype=np.float64)
        else:
            _z, weights = self._measurement_vectors(result.measurements)
        is_pseudo = None
        if measurement_table is not None and int(measurement_table.idx.size) == result.residual.size:
            is_pseudo = measurement_table_bad_data_protected_mask(measurement_table)
        n_eff_init = (
            int(np.count_nonzero((~is_pseudo) & (weights > 0.0)))
            if is_pseudo is not None
            else int(np.count_nonzero(weights > 0.0))
        )
        eff_threshold, _ = compute_adaptive_bad_threshold(base_threshold, n_eff_init)
        if result.residual.size and eff_threshold > 0.0:
            normalized_upper_bound = np.abs(result.residual) / np.sqrt(1e-12)
            if float(normalized_upper_bound.max()) <= float(eff_threshold):
                self._record_profile_time("bad_data.fast_residual_bound", 0.0)
                if profile_start is not None:
                    self._record_profile_time("bad_data.total", time.perf_counter() - profile_start)
                self._last_adaptive_bad_threshold = (id(result), float(eff_threshold), None)
                return [], normalized_upper_bound
        cached_factor_entry = getattr(self, "_last_estimate_cholmod_factor", None)
        cached_factor = (
            cached_factor_entry[1]
            if (
                cached_factor_entry is not None
                and result.H is not None
                and cached_factor_entry[0] == id(result.H)
                and not getattr(cached_factor_entry[1], "is_woodbury_solver", False)
            )
            else None
        )
        effective_n_state = (
            int(result.H.shape[1])
            if result.H is not None
            else int(self.multi_energy_state_count if is_multi_energy_result else self.n_state)
        )
        sparse_leverage_available = effective_n_state <= SEScaleProfile.for_problem(effective_n_state).sparse_leverage_max_state and (
            cached_factor is not None or CHOLMOD_CHOLESKY is not None
        )
        if not sparse_leverage_available and effective_n_state > 5000 and (
            result.H is None or (result.gain is None and cached_factor is None)
        ):
            leverage = None
        else:
            if result.H is not None:
                H = result.H
            elif is_multi_energy_result:
                me_rows = getattr(result, "multi_energy_rows", None)
                H_full = self._multi_energy_jacobian_sparse(result.x)
                H = H_full[me_rows] if me_rows is not None else H_full
            else:
                jacobian_measurements = (
                    result.measurements
                    if len(result.measurements) == result.residual.size
                    else getattr(self, "active_measurements", result.measurements)
                )
                H = self.jacobian_sparse(result.x, jacobian_measurements)
            gain = result.gain
            if gain is None and cached_factor is None:
                gain, _rhs = build_normal_equations(H, result.residual, weights)
            gain_inverse_start = time.perf_counter() if self.profile_enabled else None
            cached_aat_rt = getattr(self, "_last_estimate_aat_runtime", None)
            cached_aat_op = (
                cached_aat_rt[0].last_normal_operand
                if (
                    cached_factor is not None
                    and isinstance(cached_aat_rt, tuple)
                    and len(cached_aat_rt) == 2
                    and cached_aat_rt[0] is not None
                    and cached_aat_rt[0].last_cholmod_factor is cached_factor
                )
                else None
            )
            me_scales = getattr(result, "_state_base_scales", None)
            use_pu_lev = (
                is_multi_energy_result
                and me_scales is not None
                and np.asarray(me_scales).size == int(H.shape[1])
            )
            if use_pu_lev:
                # 杠杆率尺度不变恒等式：H G^-1 H^T == H_pu G_pu^-1 H_pu^T，在良态标幺空间求逆/复用缓存因子
                s_vec = np.asarray(me_scales, dtype=np.float64)
                if is_sparse_matrix(H):
                    H_csr_l = H if getattr(H, "format", None) == "csr" else H.tocsr()
                    H_lev = H_csr_l.copy()
                    if H_lev.nnz > 0:
                        H_lev.data = H_csr_l.data * s_vec[H_csr_l.indices]
                else:
                    H_lev = np.asarray(H, dtype=np.float64) * s_vec[None, :]
                if gain is not None:
                    if is_sparse_matrix(gain):
                        D_s = diags(s_vec, format="csc")
                        gain_lev = (D_s @ gain.tocsc() @ D_s).tocsr()
                    else:
                        gain_lev = np.asarray(gain, dtype=np.float64) * s_vec[:, None] * s_vec[None, :]
                else:
                    gain_lev = None
            else:
                H_lev = H
                gain_lev = gain
            leverage = measurement_leverage_for_bad_data(
                H_lev,
                gain_lev,
                cholmod_factor=cached_factor,
                aat_operand=cached_aat_op,
                dense_state_limit=256 if is_multi_energy_result else 5000,
                sparse_threshold=256 if is_multi_energy_result else None,
                sparse_state_limit=500_000 if is_multi_energy_result else None,
            )
            if leverage is None and gain_lev is None and effective_n_state <= (256 if is_multi_energy_result else 5000):
                # 极少数极端情况稀疏杠杆回退时再兜底构造良态标幺增益矩阵
                gain_fb, _rhs = build_normal_equations(H_lev, result.residual, weights)
                leverage = measurement_leverage_for_bad_data(H_lev, gain_fb)
            if gain_inverse_start is not None:
                self._record_profile_time("bad_data.gain_inverse", time.perf_counter() - gain_inverse_start)
        normalized, critical_mask = compute_normalized_residuals_for_bad_data(
            result.residual,
            weights,
            leverage,
            is_pseudo=is_pseudo,
        )
        n_eff_nc = (
            int(np.count_nonzero((~is_pseudo) & (weights > 0.0) & (~critical_mask)))
            if is_pseudo is not None
            else int(np.count_nonzero((weights > 0.0) & (~critical_mask)))
        )
        eff_threshold, per_row_thr = compute_adaptive_bad_threshold(
            base_threshold,
            n_eff_nc if n_eff_nc > 0 else n_eff_init,
            normalized=normalized,
            weights=weights,
            active_mask=(weights > 0.0),
            protected_mask=is_pseudo,
            critical_mask=critical_mask,
        )
        self._last_adaptive_bad_threshold = (id(result), float(eff_threshold), per_row_thr)
        bad_items: List[BadDataItem] = []
        hit_indices = (
            np.where(normalized > per_row_thr)[0]
            if per_row_thr is not None
            else np.where(normalized > eff_threshold)[0]
        )
        for idx in hit_indices:
            row_pos = int(idx)
            if is_pseudo is not None and row_pos < is_pseudo.size and bool(is_pseudo[row_pos]):
                continue
            meas = (
                measurement_from_table_row(measurement_table, row_pos)
                if measurement_table is not None
                else result.measurements[row_pos]
            )
            measured_value = (
                float(measurement_table.value[row_pos])
                if measurement_table is not None
                else float(meas.value)
            )
            bad_items.append(
                BadDataItem(
                    measurement=meas,
                    residual=float(result.residual[idx]),
                    normalized_residual=float(normalized[idx]),
                    estimated_value=float(result.z_est[idx]),
                    measured_value=measured_value,
                    row_pos=row_pos,
                )
            )
        bad_items.sort(key=lambda item: item.normalized_residual, reverse=True)
        self._last_leverage_cache = (id(result), leverage, cached_factor)
        if profile_start is not None:
            self._record_profile_time("bad_data.total", time.perf_counter() - profile_start)
        return bad_items, normalized

    def build_se_result(
        self,
        result: EstimateResult,
        bad_items: Optional[Sequence[BadDataItem]] = None,
        normalized_residual: Optional[Sequence[float]] = None,
        threshold: Optional[float] = None,
        result_mode: str = "full",
    ) -> Optional[SEResult]:
        """Build the structured state-estimation result snapshot after WLS."""
        mode = normalize_seresult_result_mode(result_mode)
        if self.fluid_estimators and result is self.multi_energy_estimate_result:
            if bad_items is None or normalized_residual is None:
                computed_bad_items, computed_normalized = self.identify_bad_data(result, threshold)
                if bad_items is None:
                    bad_items = computed_bad_items
                if normalized_residual is None:
                    normalized_residual = computed_normalized
            self.se_result = self._build_result_from_table(
                result,
                bad_items=bad_items,
                normalized_residual=np.asarray(normalized_residual, dtype=np.float64),
                result_mode=mode,
                all_measurement_table=self._multi_energy_measurement_table,
            )
            return self.se_result
        if mode in ("none", "array"):
            self.se_result = None
            return None
        if bad_items is None or normalized_residual is None:
            computed_bad_items, computed_normalized = self.identify_bad_data(result, threshold)
            if bad_items is None:
                bad_items = computed_bad_items
            if normalized_residual is None:
                normalized_residual = computed_normalized
        all_table = getattr(self.measurements, "table", None)
        prefilter_reasons = getattr(all_table, "prefilter_reason", getattr(self, "_prefilter_reasons", None))
        if mode == "summary":
            self.se_result = build_seresult_summary_from_table(
                result,
                bad_items=bad_items,
                all_measurement_table=all_table,
                prefilter_reasons=prefilter_reasons,
            )
            return self.se_result
        self.se_result = build_seresult_full_from_table(
            result,
            bad_items=bad_items,
            normalized_residual=normalized_residual,
            all_measurement_table=all_table,
            prefilter_reasons=prefilter_reasons,
        )
        return self.se_result

    def apply_state(self, x: np.ndarray) -> None:
        """Apply an estimated hybrid state vector back to the underlying network arrays."""
        if not self._prepared:
            self.prepare()
        x_arr = np.asarray(x, dtype=np.float64)
        delegate = self._delegate()
        if delegate is not None:
            delegate.apply_state(x_arr)
            return
        if self._ac_sub_estimator is not None and self.ac_n_state > 0:
            self._ac_sub_estimator.apply_state(x_arr[: self.ac_n_state])
        if self._dc_sub_estimator is not None and self.dc_n_state > 0:
            self._dc_sub_estimator.apply_state(
                x_arr[self.dc_state_start : self.dc_state_start + self.dc_n_state]
            )

    def install_measurement_runtime(self, measurements):
        """安装混合量测视图为当前活跃数组运行时（与 AC/DC SE 接口保持一致）。"""
        if not self._prepared:
            self.prepare()
        delegate = self._delegate()
        if delegate is not None:
            return delegate.install_measurement_runtime(measurements)
        normalized = self._normalize_measurements(measurements)
        table = getattr(normalized, "table", None)
        if table is None or int(table.idx.size) != len(normalized):
            table = _measurement_table_from_measurements(normalized)
            normalized = MeasurementTableView(table, normalized=True)
        by_code = rows_by_device_type_code(table)
        partitions = partition_measurements_by_code(
            normalized,
            self._MEASUREMENT_SIDE_BY_DEVICE_TYPE_CODE,
            table_builder=_measurement_table_from_measurements,
            rows_by_device_type_code=by_code,
            as_view=True,
            sides=("ac", "dc", "hybrid"),
        )
        self.active_measurements = normalized
        self.ac_meas_rows = partitions.rows["ac"].astype(np.int32, copy=False)
        self.dc_meas_rows = partitions.rows["dc"].astype(np.int32, copy=False)
        self.hybrid_meas_rows = partitions.rows["hybrid"].astype(np.int32, copy=False)
        self.ac_meas = partitions.measurements["ac"]
        self.dc_meas = partitions.measurements["dc"]
        self.hybrid_meas = partitions.measurements["hybrid"]
        self._active_ac_hybrid_rows = self.ac_meas_rows.copy()
        self._active_dc_hybrid_rows = self.dc_meas_rows.copy()
        self._active_ac_sub_measurements = self.ac_meas
        self._active_dc_sub_measurements = self.dc_meas
        self._active_ac_sub_rows = np.arange(len(self.ac_meas), dtype=np.int32)
        self._active_dc_sub_rows = np.arange(len(self.dc_meas), dtype=np.int32)
        self._active_ac_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_dc_delegated_row_mask = np.zeros(len(self.active_measurements), dtype=bool)
        if self._active_ac_hybrid_rows.size:
            self._active_ac_delegated_row_mask[self._active_ac_hybrid_rows] = True
        if self._active_dc_hybrid_rows.size:
            self._active_dc_delegated_row_mask[self._active_dc_hybrid_rows] = True
        self._jacobian_static_skip = np.zeros(len(self.active_measurements), dtype=bool)
        self._active_measurement_blocks = {
            "ac": self._MeasurementSideBlock(
                self.ac_meas_rows,
                self.ac_meas,
                self._sub_measurement_plan_tables(self._ac_sub_estimator, self.ac_meas),
            ),
            "dc": self._MeasurementSideBlock(
                self.dc_meas_rows,
                self.dc_meas,
                self._sub_measurement_plan_tables(self._dc_sub_estimator, self.dc_meas),
            ),
            "hybrid": self._MeasurementSideBlock(self.hybrid_meas_rows, self.hybrid_meas),
        }
        self._active_hybrid_measurement_plan = self._build_hybrid_measurement_plan(
            self._active_measurement_blocks["hybrid"],
        )
        self.active_z = np.asarray(table.value, dtype=np.float64)
        self.active_weight = np.asarray(table.weight, dtype=np.float64)
        self.active_uniform_weight = self._uniform_weight(self.active_weight)
        self.active_weights_are_uniform = self.active_uniform_weight is not None
        self._active_normal_pattern = None
        self._jacobian_builder = SparseJacobianBuilder((len(self.active_measurements), self.n_state))
        self._jacobian_builder._assume_fixed_pattern = True
        self._sub_jacobian_stamp_cache = {}
        self._initial_observability_cache = None
        self._observability_matrix_cache = None
        return self._active_measurement_blocks

    def measurement_device_counts(self) -> Dict[int, int]:
        """Return measurement device counts keyed by device_type_code, matching AC/DC SE."""
        delegate = self._delegate()
        if delegate is not None:
            return delegate.measurement_device_counts()
        counts: Dict[int, int] = {}
        if self._ac_sub_estimator is not None:
            counts.update(self._ac_sub_estimator.measurement_device_counts())
        if self._dc_sub_estimator is not None:
            counts.update(self._dc_sub_estimator.measurement_device_counts())
        counts[DEVICE_TYPE_DCACConverter] = int(getattr(self, "n_dcac", 0))
        return counts

    def measurement_device_names(self, requested_codes=None) -> Dict[int, np.ndarray]:
        """Return measurement device name arrays keyed by device_type_code, matching AC/DC SE."""
        delegate = self._delegate()
        if delegate is not None:
            return delegate.measurement_device_names(requested_codes)
        if requested_codes is None:
            codes = list(self.measurement_device_counts().keys())
        else:
            codes = [int(code) for code in requested_codes]
        return self._measurement_name_arrays_by_type_code(codes)

    def write_back(
        self,
        result: EstimateResult,
        *,
        bad_items: Optional[Sequence[BadDataItem]] = None,
        normalized_residual: Optional[Sequence[float]] = None,
        threshold: Optional[float] = None,
        result_mode: str = "full",
    ) -> Optional[SEResult]:
        """Store estimate_result / bad-data / se_result after run()."""
        self.estimate_result = result
        self.converged = bool(result.converged)
        self.iterations = int(result.iterations)
        if getattr(result, "observability", None) is not None:
            self.observability_result = result.observability
        if bad_items is None or normalized_residual is None:
            computed_bad_items, computed_normalized = self.identify_bad_data(result, threshold)
            if bad_items is None:
                bad_items = computed_bad_items
            if normalized_residual is None:
                normalized_residual = computed_normalized
        self.bad_items = list(bad_items)
        self.normalized_residual = np.asarray(normalized_residual, dtype=np.float64)
        return self.build_se_result(
            result,
            bad_items=bad_items,
            normalized_residual=normalized_residual,
            threshold=threshold,
            result_mode=result_mode,
        )

    def run(
        self,
        *,
        result_mode: str = "full",
        remove_bad_data: bool = False,
        bad_threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        max_remove_per_round: Optional[int] = None,
        skip_bad_data: bool = False,
        verbose: bool = False,
        final_diagnostics: bool = True,
        observability: Optional[ObservabilityResult] = None,
    ) -> Optional[SEResult]:
        if not self._prepared:
            self.prepare()
        if self.fluid_estimators:
            return self._run_multi_energy(
                result_mode=result_mode,
                remove_bad_data=remove_bad_data,
                bad_threshold=bad_threshold,
                max_remove=max_remove,
                max_remove_per_round=max_remove_per_round,
                skip_bad_data=skip_bad_data,
                verbose=verbose,
                final_diagnostics=final_diagnostics,
                observability=observability,
            )
        electric_result = self._run_electric(
            result_mode=result_mode,
            remove_bad_data=remove_bad_data,
            bad_threshold=bad_threshold,
            max_remove=max_remove,
            max_remove_per_round=max_remove_per_round,
            skip_bad_data=skip_bad_data,
            verbose=verbose,
            final_diagnostics=final_diagnostics,
            observability=observability,
        )
        self._run_fluid_estimators()
        self._build_multi_energy_result(electric_result)
        return electric_result

    def _run_electric(
        self,
        *,
        result_mode: str = "full",
        remove_bad_data: bool = False,
        bad_threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        max_remove_per_round: Optional[int] = None,
        skip_bad_data: bool = False,
        verbose: bool = False,
        final_diagnostics: bool = True,
        observability: Optional[ObservabilityResult] = None,
    ) -> Optional[SEResult]:
        if not self._prepared:
            self.prepare()
        if self._fluid_only:
            return None
        mode = normalize_seresult_result_mode(result_mode)
        threshold = self.params.bad_threshold if bad_threshold is None else bad_threshold
        array_only = mode == "array"
        delegate = self._delegate()
        if delegate is not None:
            result = delegate.run(
                result_mode=mode,
                remove_bad_data=remove_bad_data,
                bad_threshold=threshold,
                max_remove=max_remove,
                max_remove_per_round=max_remove_per_round,
                skip_bad_data=skip_bad_data,
                verbose=verbose,
                final_diagnostics=final_diagnostics,
                observability=observability,
            )
            self.observability_result = getattr(delegate, "observability_result", None)
            self.estimate_result = getattr(delegate, "estimate_result", None)
            self.converged = bool(getattr(delegate, "converged", getattr(self.estimate_result, "converged", False)))
            self.iterations = int(getattr(delegate, "iterations", getattr(self.estimate_result, "iterations", 0)))
            self.removed_bad_data = list(getattr(delegate, "removed_bad_data", []) or [])
            self.bad_items = list(getattr(delegate, "bad_items", []) or [])
            self.normalized_residual = np.asarray(
                getattr(delegate, "normalized_residual", np.array([], dtype=np.float64)),
                dtype=np.float64,
            )
            self.se_result = getattr(delegate, "se_result", None)
            return result
        needs_bad_data = not skip_bad_data
        if observability is None:
            observability = self.observability_analysis()
        self.observability_result = observability
        removed: List[BadDataItem] = []
        previous_array_only = bool(getattr(self, "_array_only_estimate_result", False))
        previous_fast_verify = bool(getattr(self, "_bad_data_fast_final_verify", False))
        self._array_only_estimate_result = array_only
        self._bad_data_fast_final_verify = mode in ("array", "none")
        try:
            if remove_bad_data:
                result, removed = self.estimate_with_bad_data_removal(
                    threshold=threshold,
                    max_remove=max_remove,
                    max_remove_per_round=max_remove_per_round,
                    verbose=verbose,
                )
            else:
                result = self.estimate(
                    verbose=verbose,
                    final_diagnostics=final_diagnostics and needs_bad_data,
                    observability=observability,
                )
        finally:
            self._array_only_estimate_result = previous_array_only
            self._bad_data_fast_final_verify = previous_fast_verify
        self.removed_bad_data = removed
        if skip_bad_data:
            bad_items = []
            normalized = np.array([], dtype=np.float64)
        else:
            cached_bad = getattr(self, "_last_bad_data_cache", None)
            if (
                isinstance(cached_bad, tuple)
                and len(cached_bad) == 4
                and cached_bad[0] == id(result)
                and float(cached_bad[1]) == float(threshold)
            ):
                bad_items, normalized = list(cached_bad[2]), cached_bad[3]
            else:
                bad_items, normalized = self.identify_bad_data(result, threshold)
        return self.write_back(
            result,
            bad_items=bad_items,
            normalized_residual=normalized,
            threshold=threshold,
            result_mode=mode,
        )

    def _run_fluid_estimators(self) -> None:
        self.fluid_se_rc = {}
        self.fluid_se_errors = {}
        for name, estimator in self.fluid_estimators.items():
            try:
                rc = int(estimator.run())
            except (RuntimeError, ValueError, ArithmeticError) as exc:
                rc = 1
                self.fluid_se_errors[name] = str(exc)
            self.fluid_se_rc[name] = rc
            if rc != 0 and name not in self.fluid_se_errors:
                result = getattr(estimator, "result", None)
                self.fluid_se_errors[name] = (
                    "fluid state estimation did not converge"
                    if result is None
                    else f"fluid state estimation residual={float(result.residual_inf):.6e}"
                )

    def _build_multi_energy_coupling_results(self) -> List[object]:
        plan_by_key = {
            (plan.coupling.table_name, int(plan.coupling.idx)): plan
            for plan in self.multi_energy_se_coupling_plans
        }
        endpoint_values = {}
        if self.multi_energy_estimate_result is not None and self._multi_energy_se_layout_ready:
            endpoint_values, _endpoint_jacobians = self._multi_energy_endpoint_runtime(
                self.multi_energy_estimate_result.x,
                with_jacobian=False,
            )
        results = []
        for coupling in self.multi_energy.couplings:
            plan = plan_by_key.get((coupling.table_name, int(coupling.idx)))
            domains_ready = all(
                terminal.domain in {"ac", "dc"}
                or terminal.domain in self.fluid_estimators
                for terminal in (coupling.t1, coupling.t2)
            )
            t1_value = None
            t2_value = None
            residual = None
            source_flow = None
            t_out = None
            t_return = None
            if not coupling.active:
                status = "inactive"
            elif plan is not None and endpoint_values:
                if plan.electric_heat:
                    dependent_value = float(
                        self.multi_energy_estimate_result.x[plan.control_col]
                    )
                    source_flow = float(
                        endpoint_values[plan.heat_flow.provider][plan.heat_flow.row]
                    )
                    t_return = float(
                        self.multi_energy_estimate_result.x[plan.return_temperature_col]
                    )
                    if coupling.control_type == "P":
                        electric_value = plan.electric_setpoint
                        t_out = dependent_value
                    else:
                        electric_value = dependent_value
                        t_out = plan.supply_temperature_set
                    if plan.t1 == plan.electric:
                        t1_value, t2_value = electric_value, t_out
                    else:
                        t1_value, t2_value = t_out, electric_value
                    residual = electric_heat_balance_residual(
                        coupling,
                        electric_value,
                        source_flow,
                        t_out,
                        t_return,
                        plan.heat_capacity,
                        plan.power_scale,
                    )
                elif plan.gas_heat:
                    dependent_value = float(
                        self.multi_energy_estimate_result.x[plan.control_col]
                    )
                    source_flow = float(
                        endpoint_values[plan.heat_flow.provider][plan.heat_flow.row]
                    )
                    t_return = float(
                        self.multi_energy_estimate_result.x[plan.return_temperature_col]
                    )
                    if coupling.control_type == "FLOW":
                        gas_value = plan.gas_setpoint
                        t_out = dependent_value
                    else:
                        gas_value = dependent_value
                        t_out = plan.supply_temperature_set
                    if plan.t1 == plan.gas:
                        t1_value, t2_value = gas_value, t_out
                    else:
                        t1_value, t2_value = t_out, gas_value
                    residual = gas_heat_balance_residual(
                        coupling,
                        gas_value,
                        source_flow,
                        t_out,
                        t_return,
                        plan.heat_capacity,
                    )
                elif plan.hydrogen_electric or plan.gas_electric or plan.steam_electric:
                    dependent_value = (
                        float(self.multi_energy_estimate_result.x[plan.control_col])
                        if plan.control_col >= 0
                        else float(
                            endpoint_values[plan.dependent.provider][plan.dependent.row]
                        )
                    )
                    controlled_value = float(plan.controlled_setpoint)
                    if plan.t1 == plan.controlled:
                        t1_value, t2_value = controlled_value, dependent_value
                    else:
                        t1_value, t2_value = dependent_value, controlled_value
                    electric_value = (
                        t1_value if plan.t1.domain in {"ac", "dc"} else t2_value
                    )
                    if plan.hydrogen_electric:
                        fluid_value = t1_value if plan.t1.domain == "hydro" else t2_value
                        residual = hydrogen_electric_balance_residual(
                            coupling,
                            electric_value,
                            fluid_value,
                            plan.power_scale,
                        )
                    elif plan.gas_electric:
                        fluid_value = t1_value if plan.t1.domain == "gas" else t2_value
                        residual = gas_electric_balance_residual(
                            coupling,
                            electric_value,
                            fluid_value,
                            plan.power_scale,
                        )
                    else:
                        fluid_value = t1_value if plan.t1.domain == "steam" else t2_value
                        residual = steam_electric_balance_residual(
                            coupling,
                            electric_value,
                            fluid_value,
                            float(
                                self.multi_energy_estimate_result.x[
                                    plan.steam_enthalpy_col
                                ]
                            ),
                            plan.condensate_enthalpy,
                            plan.power_scale,
                        )
                else:
                    t1_value = (
                        float(self.multi_energy_estimate_result.x[plan.control_col])
                        if plan.control_col >= 0
                        else float(endpoint_values[plan.t1.provider][plan.t1.row])
                    )
                    t2_value = float(endpoint_values[plan.t2.provider][plan.t2.row])
                    residual = (
                        abs(t2_value) * plan.t2_scale
                        - float(coupling.efficiency) * abs(t1_value) * plan.t1_scale
                    )
                normalized_tolerance = max(
                    10.0 * self.tol,
                    3.0 / np.sqrt(max(self.multi_energy_coupling_weight, 1.0)),
                )
                tolerance = max(1.0e-7, normalized_tolerance * plan.normalization)
                status = "balanced" if abs(residual) <= tolerance else "mismatch"
            elif domains_ready:
                status = "associated"
            else:
                status = "unavailable"
            results.append(
                SimpleNamespace(
                    coupling=coupling,
                    table_name=coupling.table_name,
                    idx=coupling.idx,
                    name=coupling.name,
                    active=coupling.active,
                    control_type=coupling.control_type,
                    e2h_coeff=coupling.e2h_coeff,
                    h2e_coeff=coupling.h2e_coeff,
                    e2g_coeff=coupling.e2g_coeff,
                    g2e_coeff=coupling.g2e_coeff,
                    e2s_coeff=coupling.e2s_coeff,
                    s2e_coeff=coupling.s2e_coeff,
                    g2h_coeff=coupling.g2h_coeff,
                    efficiency=coupling.efficiency,
                    energy_factor=coupling.energy_factor,
                    t1_value=t1_value,
                    t2_value=t2_value,
                    source_flow=(
                        source_flow
                        if plan is not None and (plan.electric_heat or plan.gas_heat) and endpoint_values
                        else None
                    ),
                    supply_temperature=(
                        t_out
                        if plan is not None and (plan.electric_heat or plan.gas_heat) and endpoint_values
                        else None
                    ),
                    return_temperature=(
                        t_return
                        if plan is not None and (plan.electric_heat or plan.gas_heat) and endpoint_values
                        else None
                    ),
                    residual=residual,
                    status=status,
                )
            )
        return results

    def _build_multi_energy_result(self, electric_result: Optional[SEResult]) -> MultiEnergySEResult:
        fluid_results = {
            name: estimator.se_result
            for name, estimator in self.fluid_estimators.items()
            if getattr(estimator, "se_result", None) is not None
        }
        electric_converged = self._fluid_only or bool(
            self.estimate_result is not None and self.estimate_result.converged
        )
        electric_observable = self._fluid_only or bool(
            self.observability_result is not None and self.observability_result.observable
        )
        fluid_converged = all(
            self.fluid_se_rc.get(name, 1) == 0
            and getattr(estimator, "result", None) is not None
            and estimator.result.converged
            for name, estimator in self.fluid_estimators.items()
        )
        fluid_observable = all(
            getattr(estimator, "result", None) is not None
            and estimator.result.observability.observable
            for estimator in self.fluid_estimators.values()
        )
        result = MultiEnergySEResult(
            electric=electric_result,
            fluids=fluid_results,
            fluid_estimators=dict(self.fluid_estimators),
            couplings=self._build_multi_energy_coupling_results(),
            warnings=list(self.multi_energy.warnings),
            errors=dict(self.fluid_se_errors),
            converged=bool(electric_converged and fluid_converged),
            observable=bool(electric_observable and fluid_observable),
        )
        self.multi_energy_result = result
        if self.se_result is not None:
            self.se_result.multi_energy = result
        return result

    def estimate_with_bad_data_removal(
        self,
        threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        max_remove_per_round: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[EstimateResult, List[BadDataItem]]:
        if self.fluid_estimators:
            threshold_value = self.params.bad_threshold if threshold is None else float(threshold)
            return self._estimate_multi_energy_with_bad_data_removal(
                threshold=threshold_value,
                max_remove=max_remove,
                max_remove_per_round=max_remove_per_round,
                verbose=verbose,
            )
        delegate = self._delegate()
        if delegate is not None:
            return delegate.estimate_with_bad_data_removal(
                threshold=threshold,
                max_remove=max_remove,
                max_remove_per_round=max_remove_per_round,
                verbose=verbose,
            )
        base_threshold = float(self.params.bad_threshold if threshold is None else threshold)
        base_measurements = self.active_measurements
        base_table = getattr(base_measurements, "table", None)
        base_weights = (
            np.asarray(base_table.weight, dtype=np.float64)
            if base_table is not None and int(base_table.idx.size) == len(base_measurements)
            else self._measurement_vectors(base_measurements)[1]
        )
        base_is_pseudo = (
            measurement_table_bad_data_protected_mask(base_table)
            if base_table is not None and int(base_table.idx.size) == len(base_measurements)
            else None
        )
        n_base = int(base_weights.size)
        n_real_meas = (
            int(np.count_nonzero((~base_is_pseudo) & (base_weights > 0.0)))
            if base_is_pseudo is not None
            else int(np.count_nonzero(base_weights > 0.0))
        )
        threshold, _ = compute_adaptive_bad_threshold(
            base_threshold,
            n_real_meas if n_real_meas > 0 else n_base,
        )
        cur_row_thr: Optional[np.ndarray] = None

        def _filter_fast_bad_items(
            items_in: List[BadDataItem],
            res_obj: EstimateResult,
            w_cur: np.ndarray,
            act_cur: np.ndarray,
        ) -> List[BadDataItem]:
            """在快速筛查路径上按异构权值分桶保守下界过滤失配高权桶假阳性。"""
            if not items_in or base_threshold <= 0.0:
                return items_in
            row_thr_use = cur_row_thr
            if row_thr_use is None and res_obj.residual.size == n_base:
                lb_vec = np.abs(np.asarray(res_obj.residual, dtype=np.float64)) * np.sqrt(np.maximum(w_cur, 0.0))
                n_act_eff = (
                    int(np.count_nonzero((~base_is_pseudo) & act_cur & (w_cur > 0.0)))
                    if base_is_pseudo is not None
                    else int(np.count_nonzero(act_cur & (w_cur > 0.0)))
                )
                _, row_thr_use = compute_adaptive_bad_threshold(
                    base_threshold,
                    n_act_eff,
                    normalized=lb_vec,
                    weights=w_cur,
                    active_mask=act_cur,
                    protected_mask=base_is_pseudo,
                )
            if row_thr_use is None:
                return items_in
            return [
                it
                for it in items_in
                if it.row_pos is not None
                and 0 <= int(it.row_pos) < row_thr_use.size
                and float(it.normalized_residual) > float(row_thr_use[int(it.row_pos)])
            ]
        max_remove, max_remove_per_round = resolve_bad_data_limits(
            n_real_meas if n_real_meas > 0 else n_base,
            max_remove=max_remove,
            max_remove_per_round=max_remove_per_round,
            param_max_remove=self.params.max_remove,
            param_max_remove_per_round=getattr(self.params, "max_remove_per_round", None),
        )
        effective_base_weights = base_weights.copy()
        active_mask = np.ones(n_base, dtype=bool)
        masked_weights = base_weights.copy()
        masked_values: Optional[np.ndarray] = None
        weight_boosted = False
        removed: List[BadDataItem] = []
        protected_keys: set = set()
        fast_final_verify = bool(getattr(self, "_bad_data_fast_final_verify", False))
        x0 = self.initial_state()
        initial_obs = self.observability_result if self.observability_result is not None else None
        init_profile = SEScaleProfile.for_problem(self.n_state, n_base, multi_solve=(max_remove > 0))
        # 对超大规模交直流网络（n_state >= 2000）与 AC 一致启用首轮/中间批删粗估早停，避免携带大量显性坏数据时无谓精磨
        coarse_thr = float(threshold) if (self.n_state >= 2000 and max_remove > 0) else None
        init_aat_solver = (
            CholmodAAtNormalEquationSolver(assume_fixed_pattern=True, ordering_method="auto", multi_solve=True)
            if (
                init_profile.use_aat_normal_solver
                and (CHOLMOD_ANALYZE_AAT is not None or CHOLMOD_CHOLESKY_AAT is not None)
                and max_remove > 0
            )
            else None
        )
        result = self.estimate(
            measurements=base_measurements,
            x0=x0,
            verbose=verbose,
            observability=initial_obs,
            _shared_aat_solver=init_aat_solver,
            _coarse_bad_data_threshold=coarse_thr,
        )
        # 严重污染热启动（flat_start=False）首轮未收敛时，立即在原始全量测集上试探 1 次平启动，
        # 避免陷入错误局部盆地后在 BDR 中耗尽配额误删健康量测
        if (
            (not self.flat_start)
            and (not result.converged)
            and (not bool(getattr(result, "_coarse_stopped", False)))
        ):
            x_flat_init = self.initial_state(flat=True)
            flat_init_res = self.estimate(
                measurements=base_measurements,
                x0=x_flat_init,
                verbose=verbose,
                observability=initial_obs,
                _shared_aat_solver=init_aat_solver,
                _coarse_bad_data_threshold=coarse_thr,
            )
            prev_obj_i = float(result.objective) if np.isfinite(result.objective) else float("inf")
            flat_obj_i = float(flat_init_res.objective) if np.isfinite(flat_init_res.objective) else float("inf")
            if flat_init_res.converged or (np.isfinite(flat_obj_i) and flat_obj_i < 0.5 * prev_obj_i):
                result = flat_init_res
                x0 = x_flat_init.copy()
        huber_warmup_items: Optional[List[BadDataItem]] = None
        if (
            not result.converged
            and not bool(getattr(result, "_coarse_stopped", False))
            and max_remove > 0
            and int(self.max_iter) >= 5
            and n_base > 0
        ):
            base_z = (
                np.asarray(base_table.value, dtype=np.float64)
                if base_table is not None
                else np.asarray([float(m.value) for m in base_measurements], dtype=np.float64)
            )
            v_cols_arr = np.asarray(getattr(self, "voltage_cols", np.asarray([], dtype=np.int32)), dtype=np.intp)
            raw_ac_th = np.asarray(getattr(self, "ac_theta_state_col", np.asarray([], dtype=np.int32)), dtype=np.int32)
            a_cols_arr = raw_ac_th[raw_ac_th >= 0].astype(np.intp, copy=False) if raw_ac_th.size else np.empty(0, dtype=np.intp)
            init_rt = getattr(self, "_last_estimate_aat_runtime", None)
            hub_solver = (
                init_rt[0]
                if (isinstance(init_rt, tuple) and len(init_rt) == 2 and init_rt[0] is not None)
                else init_aat_solver
            )
            hub_plan = (
                init_rt[1]
                if (isinstance(init_rt, tuple) and len(init_rt) == 2)
                else None
            )
            warmup = run_huber_irls_warmup(
                lambda xv, out_z: self.evaluate(xv, base_measurements, out=out_z),
                lambda ze, out_r: self._measurement_residual(base_z, ze, base_measurements, out=out_r),
                lambda xv: self.jacobian_sparse(xv, base_measurements),
                x0,
                base_weights,
                base_is_pseudo,
                v_cols=v_cols_arr,
                a_cols=a_cols_arr,
                v_min=float(self.voltage_min),
                v_max=float(self.voltage_max),
                a_min_rad=float(self.angle_min_rad),
                a_max_rad=float(self.angle_max_rad),
                max_iter=min(15, int(self.max_iter)),
                huber_k=3.0,
                shared_aat_solver=hub_solver,
                shared_aat_plan=hub_plan,
            )
            if warmup is not None:
                x_hub, r_hub, z_hub = warmup
                v_hub_min = float(np.min(x_hub[v_cols_arr])) if v_cols_arr.size else 1.0
                v_hub_max = float(np.max(x_hub[v_cols_arr])) if v_cols_arr.size else 1.0
                hub_healthy = (
                    bool(np.all(np.isfinite(x_hub)))
                    and v_hub_min >= float(self.voltage_min) + 0.10
                    and v_hub_max <= float(self.voltage_max) - 0.10
                )
                if hub_healthy:
                    warm_res = self.estimate(
                        measurements=base_measurements,
                        x0=x_hub,
                        verbose=verbose,
                        observability=initial_obs,
                        _shared_aat_solver=init_aat_solver,
                        _coarse_bad_data_threshold=coarse_thr,
                    )
                    if warm_res.converged or (
                        np.isfinite(warm_res.objective)
                        and (not np.isfinite(result.objective) or warm_res.objective < result.objective)
                    ):
                        result = warm_res
                        if warm_res.converged:
                            x0 = warm_res.x.copy()
                    if not result.converged and not bool(getattr(result, "_coarse_stopped", False)):
                        wres_h = np.sqrt(np.maximum(base_weights, 0.0)) * np.abs(r_hub)
                        cand_h = np.flatnonzero(
                            (~base_is_pseudo) & (base_weights > 0.0) & (wres_h > max(float(threshold), 4.0))
                        )
                        if cand_h.size:
                            order_h = cand_h[np.argsort(-wres_h[cand_h], kind="stable")]
                            huber_warmup_items = []
                            for rp in order_h.tolist():
                                rp_i = int(rp)
                                meas = (
                                    measurement_from_table_row(base_table, rp_i)
                                    if base_table is not None
                                    else base_measurements[rp_i]
                                )
                                measured_val = float(base_table.value[rp_i]) if base_table is not None else float(meas.value)
                                huber_warmup_items.append(
                                    BadDataItem(
                                        measurement=meas,
                                        residual=float(r_hub[rp_i]),
                                        normalized_residual=float(wres_h[rp_i]),
                                        estimated_value=float(z_hub[rp_i]),
                                        measured_value=measured_val,
                                        row_pos=rp_i,
                                    )
                                )
        shared_runtime = getattr(self, "_last_estimate_aat_runtime", None)
        shared_aat_solver = shared_runtime[0] if isinstance(shared_runtime, tuple) and len(shared_runtime) == 2 else None
        shared_aat_plan = shared_runtime[1] if isinstance(shared_runtime, tuple) and len(shared_runtime) == 2 else None
        rm_profile = SEScaleProfile.for_problem(
            self.n_state,
            n_base,
            int(result.H.nnz) if is_sparse_matrix(result.H) else 0,
            multi_solve=True,
        )
        wb_max_rank = rm_profile.woodbury_max_rank

        fac_entry = getattr(self, "_last_estimate_cholmod_factor", None)
        cur_factor = (
            fac_entry[1]
            if (fac_entry is not None and result.H is not None and fac_entry[0] == id(result.H))
            else None
        )
        anchor_factor = None
        anchor_H = None
        anchor_weights = None
        anchor_x = None
        anchor_leverage = None
        anchor_removed_rows: List[int] = []
        anchor_u_cache: Dict = {}

        def _reset_anchor(res_obj: EstimateResult, exact_fac, lev_arr) -> None:
            nonlocal anchor_factor, anchor_H, anchor_weights, anchor_x, anchor_leverage, anchor_removed_rows
            anchor_u_cache.clear()
            if exact_fac is not None and not getattr(exact_fac, "is_woodbury_solver", False) and res_obj.H is not None:
                anchor_factor = exact_fac
                anchor_H = res_obj.H
                anchor_weights = masked_weights.copy()
                anchor_x = res_obj.x.copy()
                anchor_leverage = lev_arr
                anchor_removed_rows = []
            else:
                anchor_factor = None
                anchor_H = None
                anchor_weights = None
                anchor_x = None
                anchor_leverage = None
                anchor_removed_rows = []

        def _refresh_exact_factor_for_current() -> bool:
            """当当前解处于延迟低秩 Woodbury 状态时，在最新状态点补做 1 次精确 CHOLMOD 数值分解。"""
            nonlocal result, cur_factor, shared_aat_solver, shared_aat_plan
            if not getattr(result, "_factor_deferred", False) and not getattr(cur_factor, "is_woodbury_solver", False):
                return True
            if (
                result.H is not None
                and shared_aat_solver is not None
                and shared_aat_plan is not None
                and shared_aat_solver.last_cholmod_factor is not None
                and is_sparse_matrix(result.H)
                and tuple(result.H.shape) == shared_aat_plan.shape
                and int(result.H.nnz) == int(shared_aat_plan.h_indices.size)
            ):
                try:
                    t_ref = time.perf_counter() if self.profile_enabled else None
                    normal_operand, _ = shared_aat_plan.assemble(
                        result.H,
                        result.residual,
                        masked_weights,
                        uniform_weight=None,
                        weights_are_uniform=False,
                        weighted_residual=None,
                        assume_fixed_weights=False,
                        copy_rhs=False,
                        assume_pattern_matches=True,
                    )
                    shared_aat_solver.last_cholmod_factor.cholesky_AAt_inplace(normal_operand)
                    shared_aat_solver._last_normal_operand = normal_operand
                    cur_factor = shared_aat_solver.last_cholmod_factor
                    result._factor_deferred = False
                    result._inplace_factor_modified = True
                    self._last_estimate_cholmod_factor = (id(result.H), cur_factor)
                    if t_ref is not None:
                        self._record_profile_time("solve.linear_solve", time.perf_counter() - t_ref)
                    return True
                except Exception:
                    pass
            refined = self.estimate(
                measurements=base_measurements,
                x0=result.x,
                verbose=verbose,
                _defer_observability=need_slice_finalize,
                _weight_override=masked_weights if need_slice_finalize else None,
                _value_override=masked_values,
                _shared_aat_solver=shared_aat_solver,
                _shared_aat_plan=shared_aat_plan,
            )
            result = refined
            ref_fac_entry = getattr(self, "_last_estimate_cholmod_factor", None)
            cur_factor = (
                ref_fac_entry[1]
                if (ref_fac_entry is not None and result.H is not None and ref_fac_entry[0] == id(result.H))
                else None
            )
            return cur_factor is not None

        def _ensure_exact_result() -> bool:
            """若当前解来自前期粗收敛早停，则立即按原 tol 严格精算，确保终轮零精度损失。"""
            nonlocal result, cur_factor, cur_leverage, full_normalized_valid, normalized
            if not getattr(result, "_coarse_stopped", False):
                return False
            refined = self.estimate(
                measurements=base_measurements,
                x0=result.x,
                verbose=verbose,
                _defer_observability=need_slice_finalize,
                _weight_override=masked_weights if need_slice_finalize else None,
                _value_override=masked_values,
                _shared_aat_solver=shared_aat_solver,
                _shared_aat_plan=shared_aat_plan,
            )
            result = refined
            ref_fac_entry = getattr(self, "_last_estimate_cholmod_factor", None)
            cur_factor = (
                ref_fac_entry[1]
                if (ref_fac_entry is not None and result.H is not None and ref_fac_entry[0] == id(result.H))
                else None
            )
            cur_leverage = None
            normalized = None
            full_normalized_valid = False
            _reset_anchor(result, cur_factor, None)
            return True

        need_slice_finalize = False
        cur_leverage = None
        normalized = None
        full_normalized_valid = False
        init_fast_done = False
        if huber_warmup_items:
            bad_items = huber_warmup_items
            normalized = None
            cur_leverage = None
            full_normalized_valid = False
            init_fast_done = True
        elif rm_profile.fast_init_screen_enabled and max_remove > 0 and cur_factor is not None and result.H is not None:
            init_screen_start = time.perf_counter() if self.profile_enabled else None
            init_status, init_rows, init_norms = screen_initial_bad_data_fast(
                result.residual,
                result.H,
                base_weights,
                base_is_pseudo,
                cur_factor,
                threshold,
            )
            if init_status == "fast_clean" and not getattr(result, "_coarse_stopped", False):
                bad_items = []
                normalized = init_norms
                cur_leverage = None
                full_normalized_valid = True
                init_fast_done = True
                if init_screen_start is not None:
                    self._record_profile_time("bad_data.total", time.perf_counter() - init_screen_start)
            elif init_status == "candidates":
                bad_items = []
                for r_pos, r_norm in zip(init_rows.tolist(), init_norms.tolist()):
                    meas = (
                        measurement_from_table_row(base_table, r_pos)
                        if base_table is not None
                        else base_measurements[r_pos]
                    )
                    measured_value = (
                        float(base_table.value[r_pos])
                        if base_table is not None
                        else float(meas.value)
                    )
                    bad_items.append(
                        BadDataItem(
                            measurement=meas,
                            residual=float(result.residual[r_pos]),
                            normalized_residual=float(r_norm),
                            estimated_value=float(result.z_est[r_pos]),
                            measured_value=measured_value,
                            row_pos=r_pos,
                        )
                    )
                bad_items = _filter_fast_bad_items(bad_items, result, base_weights, active_mask)
                normalized = None
                cur_leverage = None
                full_normalized_valid = False
                init_fast_done = bool(bad_items)
                if init_screen_start is not None:
                    dt = time.perf_counter() - init_screen_start
                    self._record_profile_time("bad_data.gain_inverse", dt)
                    self._record_profile_time("bad_data.total", dt)
        if not init_fast_done:
            _ensure_exact_result()
            bad_items, normalized = self.identify_bad_data(result, base_threshold)
            lev_pair = getattr(self, "_last_leverage_cache", None)
            cur_leverage = lev_pair[1] if (isinstance(lev_pair, tuple) and len(lev_pair) == 3 and lev_pair[0] == id(result)) else None
            thr_pair = getattr(self, "_last_adaptive_bad_threshold", None)
            if isinstance(thr_pair, tuple) and len(thr_pair) == 3 and thr_pair[0] == id(result):
                cur_row_thr = thr_pair[2]
            full_normalized_valid = True
        _reset_anchor(result, cur_factor, cur_leverage)

        def _prepare_woodbury_args(add_rows: Sequence[int], prev_x: np.ndarray, prev_H, prev_weights: np.ndarray):
            """决定本轮是否复用跨轮持久锚点低秩 Woodbury（累计移除 <=24 且相对锚点位移 <=5e-2）。"""
            add_list = [int(r) for r in add_rows]
            if (
                anchor_factor is not None
                and anchor_H is not None
                and anchor_weights is not None
                and anchor_x is not None
                and shared_aat_solver is not None
                and shared_aat_solver.last_cholmod_factor is anchor_factor
                and 1 <= len(anchor_removed_rows) + len(add_list) <= wb_max_rank
                and prev_x.size == anchor_x.size
            ):
                diff_anc = prev_x - anchor_x
                disp = float(np.max(np.abs(diff_anc))) if diff_anc.size else 0.0
                rms_disp = float(np.sqrt(float(np.dot(diff_anc, diff_anc)) / diff_anc.size)) if diff_anc.size else 0.0
                if disp <= 5e-2:
                    return anchor_removed_rows + add_list, anchor_weights, anchor_H, True, True
                if rms_disp <= 2e-3 and disp <= 0.35:
                    return anchor_removed_rows + add_list, anchor_weights, anchor_H, False, True
            if (
                cur_factor is not None
                and not getattr(cur_factor, "is_woodbury_solver", False)
                and prev_H is not None
                and 1 <= len(add_list) <= wb_max_rank
                and shared_aat_solver is not None
                and shared_aat_solver.last_cholmod_factor is cur_factor
            ):
                return add_list, prev_weights, prev_H, True, False
            return None, None, None, False, False

        def _finalize_masked_result(cur_res: EstimateResult, cur_fac, norm_arr):
            if not need_slice_finalize:
                return cur_res, norm_arr
            keep_rows = np.flatnonzero(active_mask).astype(np.int64, copy=False)
            sliced_meas = take_measurement_view(base_measurements, keep_rows)
            sliced_table = getattr(sliced_meas, "table", None)
            if weight_boosted and sliced_table is not None:
                sliced_table.weight[:] = masked_weights[keep_rows]
            sliced_H = (
                cur_res.H.tocsr()[keep_rows]
                if is_sparse_matrix(cur_res.H)
                else (cur_res.H[keep_rows] if cur_res.H is not None else None)
            )
            sliced_res_vec = cur_res.residual[keep_rows].copy()
            sliced_z_est = cur_res.z_est[keep_rows].copy()
            obs = cur_res.observability
            if obs is not None and obs.measurement_count != int(keep_rows.size):
                obs = ObservabilityResult(
                    observable=obs.observable,
                    rank=obs.rank,
                    state_count=obs.state_count,
                    measurement_count=int(keep_rows.size),
                    deficiency=obs.deficiency,
                    singular_values=obs.singular_values,
                    weak_states=obs.weak_states,
                )
            array_only_result = bool(getattr(self, "_array_only_estimate_result", False))
            final_res = EstimateResult(
                converged=cur_res.converged,
                iterations=cur_res.iterations,
                objective=cur_res.objective,
                max_correction=cur_res.max_correction,
                residual_inf=float(np.linalg.norm(sliced_res_vec, np.inf)) if sliced_res_vec.size else 0.0,
                x=cur_res.x,
                z_est=sliced_z_est,
                residual=sliced_res_vec,
                H=sliced_H,
                gain=cur_res.gain,
                measurements=[] if array_only_result else sliced_meas,
                observability=obs,
                measurement_table=sliced_table,
            )
            if (
                sliced_H is not None
                and cur_fac is not None
                and not getattr(cur_fac, "is_woodbury_solver", False)
            ):
                self._last_estimate_cholmod_factor = (id(sliced_H), cur_fac)
            else:
                self._last_estimate_cholmod_factor = None
            sliced_norm = norm_arr[keep_rows] if (norm_arr is not None and np.asarray(norm_arr).size == n_base) else norm_arr
            return final_res, sliced_norm

        dynamic_batch_cap: Optional[int] = None
        batch_cooldown = 0
        batch_fail_streak = 0
        batch_ok_streak = 0
        unconverged_stall_rounds = 0
        unconverged_total_rounds = 0
        rescue_rounds = 0
        min_unconv_obj_drop = max(4.0, 0.5 * float(threshold) * float(threshold))

        def _unconv_obj_improved(prev_v: float, cur_v: float) -> bool:
            return bool(
                np.isfinite(cur_v)
                and (cur_v < 0.995 * prev_v or (prev_v - cur_v) >= min_unconv_obj_drop)
            )

        def _removal_causes_singularity(trial_res: EstimateResult, prev_res: EstimateResult) -> bool:
            """判定交直流混合坏数据剔除试探是否导致原问题结构或数值奇异。"""
            t_obs = getattr(trial_res, "observability", None)
            if t_obs is not None and not t_obs.observable:
                return True
            prev_spd_bad = bool(getattr(prev_res, "_undamped_spd_failed", False))
            trial_spd_bad = bool(getattr(trial_res, "_undamped_spd_failed", False))
            if trial_spd_bad or (not trial_res.converged):
                H_chk = trial_res.H if trial_res.H is not None else prev_res.H
                if H_chk is not None and int(H_chk.shape[0]) == n_base:
                    if not check_active_structural_full_rank(H_chk, active_mask, masked_weights):
                        return True
            if trial_spd_bad and not prev_spd_bad:
                return True
            return False

        def _is_critical_leverage_row(row_pos: int) -> bool:
            return is_critical_leverage_row(cur_leverage, masked_weights, row_pos)

        def _can_warm_start_from(res_obj: EstimateResult) -> bool:
            """仅当当前混合电网状态健康且无贴墙电压/相角节点（即使已收敛）时才允许原样延续温启动或刷新锚点 x0。"""
            if not np.isfinite(float(res_obj.objective)):
                return False
            is_coarse = bool(getattr(res_obj, "_coarse_stopped", False))
            if not res_obj.converged and not is_coarse and not (float(res_obj.max_correction) < 1e-2):
                return False
            xv = res_obj.x
            if xv is None or xv.size == 0 or not np.all(np.isfinite(xv)):
                return False
            v_cols = np.asarray(getattr(self, "voltage_cols", np.asarray([], dtype=np.int32)), dtype=np.int32)
            if v_cols.size:
                vp = xv[v_cols]
                if float(vp.min()) < float(self.voltage_min) + 0.03 or float(vp.max()) > float(self.voltage_max) - 0.03:
                    return False
            raw_ac_th = np.asarray(getattr(self, "ac_theta_state_col", np.asarray([], dtype=np.int32)), dtype=np.int32)
            a_cols = raw_ac_th[raw_ac_th >= 0] if raw_ac_th.size else np.empty(0, dtype=np.int32)
            if a_cols.size:
                ap = xv[a_cols]
                if float(ap.min()) < float(self.angle_min_rad) + 0.1 or float(ap.max()) > float(self.angle_max_rad) - 0.1:
                    return False
            return True

        def _warm_start_x0(res_obj: EstimateResult, fallback_x0: np.ndarray) -> Tuple[np.ndarray, bool]:
            """构造混合电网删坏后重算初值：若存在贴墙电压节点则仅将贴墙节点复位为标称 1.0，跳出低电压下支盆地。"""
            if _can_warm_start_from(res_obj):
                return res_obj.x, False
            xv = res_obj.x
            if (
                xv is None
                or xv.size == 0
                or not np.all(np.isfinite(xv))
                or not np.isfinite(float(res_obj.objective))
            ):
                return fallback_x0, True
            is_coarse = bool(getattr(res_obj, "_coarse_stopped", False))
            if not res_obj.converged and not is_coarse and not (float(res_obj.max_correction) < 1e-2):
                return fallback_x0, True
            raw_ac_th = np.asarray(getattr(self, "ac_theta_state_col", np.asarray([], dtype=np.int32)), dtype=np.int32)
            a_cols = raw_ac_th[raw_ac_th >= 0] if raw_ac_th.size else np.empty(0, dtype=np.int32)
            if a_cols.size:
                ap = xv[a_cols]
                if float(ap.min()) < float(self.angle_min_rad) + 0.1 or float(ap.max()) > float(self.angle_max_rad) - 0.1:
                    return fallback_x0, True
            v_cols = np.asarray(getattr(self, "voltage_cols", np.asarray([], dtype=np.int32)), dtype=np.int32)
            if v_cols.size:
                vp = xv[v_cols]
                pin_mask = (vp < float(self.voltage_min) + 0.03) | (vp > float(self.voltage_max) - 0.03)
                if np.any(pin_mask):
                    clean_x = xv.copy()
                    v_nom = float(np.clip(1.0, float(self.voltage_min) + 0.05, float(self.voltage_max) - 0.05))
                    clean_x[v_cols[pin_mask]] = v_nom
                    return clean_x, True
            return fallback_x0, True

        def _try_hybrid_boundary_node_rescue_once() -> bool:
            """当交直流混合常规坏数据剔除结束仍未收敛且存在贴边电压节点时，执行两级自救：
            1) 沿贴边节点 1 跳失衡交流功率平衡方程反查高杠杆真实坏数据并安全剔除；
            2) 若无高杠杆坏数据失衡，仅对贴边盲节点自身的电压伪量测局部抬升权重（×10，上限 10.0）重算。
            """
            nonlocal result, cur_factor, cur_leverage, bad_items, normalized, full_normalized_valid
            nonlocal need_slice_finalize, weight_boosted, shared_aat_solver, masked_values
            nonlocal unconverged_stall_rounds, unconverged_total_rounds
            if (
                result.converged
                or int(self.max_iter) < 5
                or result.H is None
                or not is_sparse_matrix(result.H)
                or int(result.H.shape[0]) != n_base
                or base_table is None
            ):
                return False
            v_cols = np.asarray(getattr(self, "voltage_cols", np.asarray([], dtype=np.int32)), dtype=np.int32)
            if v_cols.size == 0:
                return False
            vx = result.x[v_cols]
            ext_cols = v_cols[
                (vx <= float(self.voltage_min) + 0.02) | (vx >= float(self.voltage_max) - 0.02)
            ].tolist()
            if not ext_cols:
                return False
            if (
                result.x is not None
                and result.x.size == x0.size
                and np.all(np.isfinite(result.x))
                and np.isfinite(float(result.objective))
            ):
                rescue_x0 = result.x.copy()
                rescue_x0[ext_cols] = 1.0
            else:
                rescue_x0 = x0
            H_csc = result.H.tocsc()
            H_csr = result.H.tocsr()
            dt_arr = np.asarray(base_table.device_type_code, dtype=np.int16)
            mt_arr = np.asarray(base_table.meas_type_code, dtype=np.int16)
            st_arr = measurement_table_status_code(base_table)
            val_arr = np.asarray(base_table.value, dtype=np.float64)

            if len(removed) < max_remove and base_is_pseudo is not None and self._ac_sub_estimator is not None:
                ac_gen_start = int(getattr(self._ac_sub_estimator, "base_gen_p", self.ac_n_state))
                per_bus_top: Dict[int, Tuple[float, int, Measurement]] = {}
                all_cand_map: Dict[int, Tuple[float, int, Measurement]] = {}
                for c_ext in ext_cols:
                    local_bal = []
                    for r in H_csc.indices[H_csc.indptr[c_ext] : H_csc.indptr[c_ext + 1]]:
                        r_i = int(r)
                        if (
                            active_mask[r_i]
                            and int(dt_arr[r_i]) == DEVICE_TYPE_ACPowerBalance
                            and abs(float(result.residual[r_i])) > 0.05
                        ):
                            local_bal.append(r_i)
                    if not local_bal:
                        continue
                    nb_cols = set()
                    for br in local_bal:
                        nb_cols.update(H_csr.indices[H_csr.indptr[br] : H_csr.indptr[br + 1]].tolist())
                    best_local: Optional[Tuple[float, int, Measurement]] = None
                    for c in nb_cols:
                        if not (ac_gen_start <= c < self.ac_n_state):
                            continue
                        for r in H_csc.indices[H_csc.indptr[c] : H_csc.indptr[c + 1]]:
                            r_i = int(r)
                            if (
                                active_mask[r_i]
                                and (not base_is_pseudo[r_i])
                                and masked_weights[r_i] > 0.0
                                and not _is_critical_leverage_row(r_i)
                            ):
                                meas_cand = measurement_from_table_row(base_table, r_i)
                                if (int(meas_cand.idx), str(meas_cand.name)) not in protected_keys:
                                    score = abs(float(val_arr[r_i])) * float(np.sqrt(masked_weights[r_i]))
                                    item_t = (score, r_i, meas_cand)
                                    if best_local is None or score > best_local[0]:
                                        best_local = item_t
                                    prev_t = all_cand_map.get(r_i)
                                    if prev_t is None or score > prev_t[0]:
                                        all_cand_map[r_i] = item_t
                    if best_local is not None:
                        prev_b = per_bus_top.get(best_local[1])
                        if prev_b is None or best_local[0] > prev_b[0]:
                            per_bus_top[best_local[1]] = best_local
                if all_cand_map:
                    col_deg_rescue = compute_active_column_degrees(H_csr, active_mask, masked_weights)
                    prev_obj = float(result.objective) if np.isfinite(result.objective) else float("inf")
                    rem_quota = min(max_remove_per_round, max_remove - len(removed))
                    batch_cands = sorted(per_bus_top.values(), key=lambda item: item[0], reverse=True)[:rem_quota]
                    if len(batch_cands) >= 2 and col_deg_rescue is not None:
                        safe_b_idx, _ = filter_batch_preserving_column_coverage(
                            H_csr, [item[1] for item in batch_cands], col_deg_rescue
                        )
                        batch_cands = [batch_cands[p] for p in safe_b_idx]
                    if len(batch_cands) >= 2 and (int(np.count_nonzero(active_mask)) - len(batch_cands)) >= self.n_state:
                        b_rows = [item[1] for item in batch_cands]
                        for r_i in b_rows:
                            masked_weights[r_i] = 0.0
                            active_mask[r_i] = False
                        trial_res = self.estimate(
                            measurements=base_measurements,
                            x0=rescue_x0,
                            verbose=verbose,
                            _defer_observability=True,
                            _weight_override=masked_weights,
                            _value_override=masked_values,
                            _shared_aat_solver=shared_aat_solver,
                            _shared_aat_plan=shared_aat_plan,
                        )
                        trial_obj = float(trial_res.objective) if np.isfinite(trial_res.objective) else float("inf")
                        rejected = _removal_causes_singularity(trial_res, result) or (
                            (not trial_res.converged) and (not _unconv_obj_improved(prev_obj, trial_obj))
                        )
                        if not rejected:
                            for _, r_i, meas_cand in batch_cands:
                                w_i = float(effective_base_weights[r_i])
                                r_val = float(result.residual[r_i])
                                removed.append(
                                    BadDataItem(
                                        measurement=meas_cand,
                                        residual=r_val,
                                        normalized_residual=float(abs(r_val) * np.sqrt(max(w_i, 0.0))),
                                        estimated_value=float(result.z_est[r_i]),
                                        measured_value=float(val_arr[r_i]),
                                        row_pos=r_i,
                                    )
                                )
                            need_slice_finalize = True
                            result = trial_res
                            fac_e = getattr(self, "_last_estimate_cholmod_factor", None)
                            cur_factor = (
                                fac_e[1]
                                if (fac_e is not None and result.H is not None and fac_e[0] == id(result.H))
                                else None
                            )
                            cur_leverage = None
                            _reset_anchor(result, cur_factor, None)
                            bad_items = []
                            normalized = None
                            full_normalized_valid = False
                            unconverged_stall_rounds = 0
                            unconverged_total_rounds = 0
                            return True
                        for r_i in b_rows:
                            masked_weights[r_i] = effective_base_weights[r_i]
                            active_mask[r_i] = True
                        if shared_aat_solver is not None and getattr(shared_aat_solver, "_cholmod_disabled", False):
                            shared_aat_solver = CholmodAAtNormalEquationSolver(assume_fixed_pattern=True)
                    cand_real = sorted(all_cand_map.values(), key=lambda item: item[0], reverse=True)
                    for _, r_i, meas_cand in cand_real[:6]:
                        if int(np.count_nonzero(active_mask)) - 1 < self.n_state:
                            protected_keys.add((int(meas_cand.idx), str(meas_cand.name)))
                            continue
                        if col_deg_rescue is not None:
                            safe_s, _ = filter_batch_preserving_column_coverage(H_csr, [r_i], col_deg_rescue)
                            if not safe_s:
                                protected_keys.add((int(meas_cand.idx), str(meas_cand.name)))
                                continue
                        masked_weights[r_i] = 0.0
                        active_mask[r_i] = False
                        trial_res = self.estimate(
                            measurements=base_measurements,
                            x0=rescue_x0,
                            verbose=verbose,
                            _defer_observability=True,
                            _weight_override=masked_weights,
                            _value_override=masked_values,
                            _shared_aat_solver=shared_aat_solver,
                            _shared_aat_plan=shared_aat_plan,
                        )
                        trial_obj = float(trial_res.objective) if np.isfinite(trial_res.objective) else float("inf")
                        rejected = _removal_causes_singularity(trial_res, result) or (
                            (not trial_res.converged) and (not _unconv_obj_improved(prev_obj, trial_obj))
                        )
                        if rejected:
                            masked_weights[r_i] = effective_base_weights[r_i]
                            active_mask[r_i] = True
                            protected_keys.add((int(meas_cand.idx), str(meas_cand.name)))
                            if shared_aat_solver is not None and getattr(shared_aat_solver, "_cholmod_disabled", False):
                                shared_aat_solver = CholmodAAtNormalEquationSolver(assume_fixed_pattern=True)
                            continue
                        w_i = float(effective_base_weights[r_i])
                        r_val = float(result.residual[r_i])
                        removed.append(
                            BadDataItem(
                                measurement=meas_cand,
                                residual=r_val,
                                normalized_residual=float(abs(r_val) * np.sqrt(max(w_i, 0.0))),
                                estimated_value=float(result.z_est[r_i]),
                                measured_value=float(val_arr[r_i]),
                                row_pos=r_i,
                            )
                        )
                        need_slice_finalize = True
                        result = trial_res
                        fac_e = getattr(self, "_last_estimate_cholmod_factor", None)
                        cur_factor = (
                            fac_e[1]
                            if (fac_e is not None and result.H is not None and fac_e[0] == id(result.H))
                            else None
                        )
                        cur_leverage = None
                        _reset_anchor(result, cur_factor, None)
                        bad_items = []
                        normalized = None
                        full_normalized_valid = False
                        unconverged_stall_rounds = 0
                        unconverged_total_rounds = 0
                        return True

            v_lo_phys = float(self.voltage_min)
            v_hi_phys = float(self.voltage_max)
            boosted = False
            for c in ext_cols:
                col_boosted = False
                col_rows = H_csc.indices[H_csc.indptr[c] : H_csc.indptr[c + 1]]
                for r in col_rows:
                    r_i = int(r)
                    if (
                        active_mask[r_i]
                        and int(st_arr[r_i]) == MEAS_STATUS_PSEUDO
                        and int(mt_arr[r_i]) in VOLTAGE_PSEUDO_MEAS_TYPE_CODES
                    ):
                        cur_w = float(masked_weights[r_i])
                        new_w = min(10.0, max(cur_w * 10.0, 1.0))
                        if new_w > cur_w + 1e-12:
                            masked_weights[r_i] = new_w
                            effective_base_weights[r_i] = new_w
                            col_boosted = True
                            boosted = True
                # 若贴墙节点无电压伪量测可提权，但存在把节点电压直接拉死在物理极限墙上（如 vp_spike 上/下界攻击）
                # 的极端离群真实 V 测点，将其安全降权至极弱（1e-4，保持列覆盖且不突破 max_remove 配额）以解救发散节点
                if not col_boosted and result.residual is not None:
                    for r in col_rows:
                        r_i = int(r)
                        if (
                            active_mask[r_i]
                            and (base_is_pseudo is None or (not base_is_pseudo[r_i]))
                            and int(mt_arr[r_i]) in VOLTAGE_PSEUDO_MEAS_TYPE_CODES
                            and masked_weights[r_i] > 1e-3
                        ):
                            w_r = float(masked_weights[r_i])
                            z_r = (
                                float(result.z_est[r_i]) + float(result.residual[r_i])
                                if (result.z_est is not None and r_i < result.z_est.size)
                                else 1.0
                            )
                            phys_oob = (z_r < v_lo_phys + 0.02) or (z_r > v_hi_phys - 0.02)
                            if phys_oob or (
                                abs(float(result.residual[r_i])) * np.sqrt(w_r) > max(2.0 * float(threshold), 8.0)
                            ):
                                # 退化为极弱标称电压伪量测 (V=1.0 pu, w=1e-4)：保持单度节点列覆盖满秩并解除贴墙拉扯
                                masked_weights[r_i] = 1e-4
                                effective_base_weights[r_i] = 1e-4
                                if masked_values is None:
                                    base_z_vec, _ = self._measurement_vectors(base_measurements)
                                    masked_values = np.asarray(base_z_vec, dtype=np.float64).copy()
                                masked_values[r_i] = 1.0
                                boosted = True
            if not boosted:
                return False
            weight_boosted = True
            need_slice_finalize = True
            trial_res = self.estimate(
                measurements=base_measurements,
                x0=rescue_x0,
                verbose=verbose,
                _defer_observability=True,
                _weight_override=masked_weights,
                _value_override=masked_values,
                _shared_aat_solver=shared_aat_solver,
                _shared_aat_plan=shared_aat_plan,
            )
            result = trial_res
            fac_e = getattr(self, "_last_estimate_cholmod_factor", None)
            cur_factor = (
                fac_e[1]
                if (fac_e is not None and result.H is not None and fac_e[0] == id(result.H))
                else None
            )
            cur_leverage = None
            _reset_anchor(result, cur_factor, None)
            bad_items = []
            normalized = None
            full_normalized_valid = False
            unconverged_stall_rounds = 0
            unconverged_total_rounds = 0
            return True

        def _try_hybrid_boundary_node_rescue() -> bool:
            """单次混合电网贴边自救会话：最多连续推进 3 个子步直至收敛或全网无贴边，并刷新安全锚定初值 x0。"""
            nonlocal x0
            any_progressed = False
            for _sub in range(3):
                if not _try_hybrid_boundary_node_rescue_once():
                    break
                any_progressed = True
                if result.converged:
                    break
            if any_progressed and _can_warm_start_from(result):
                x0 = result.x.copy()
            return any_progressed

        flat_fallback_tried = bool(self.flat_start)
        warm_cont_tried = False

        def _try_warm_continuation() -> bool:
            """若在当前活跃量测集上跑满 max_iter 仍未收敛但状态健康，以当前解清零阻尼温启动续跑一次。"""
            nonlocal result, cur_factor, cur_leverage, bad_items, normalized, full_normalized_valid
            nonlocal x0, warm_cont_tried, unconverged_stall_rounds, unconverged_total_rounds
            if (
                result.converged
                or warm_cont_tried
                or int(getattr(result, "iterations", 0)) < min(20, int(self.max_iter))
                or not _can_warm_start_from(result)
            ):
                return False
            warm_cont_tried = True
            cont_res = self.estimate(
                measurements=base_measurements,
                x0=result.x.copy(),
                verbose=verbose,
                _defer_observability=True,
                _weight_override=masked_weights,
                _value_override=masked_values,
                _shared_aat_solver=shared_aat_solver,
                _shared_aat_plan=shared_aat_plan,
                _hint_lm_damping=0.0,
            )
            prev_obj = float(result.objective) if np.isfinite(result.objective) else float("inf")
            cont_obj = float(cont_res.objective) if np.isfinite(cont_res.objective) else float("inf")
            if not (cont_res.converged or (np.isfinite(cont_obj) and cont_obj < 0.95 * prev_obj)):
                return False
            result = cont_res
            if _can_warm_start_from(result):
                x0 = result.x.copy()
            fac_e = getattr(self, "_last_estimate_cholmod_factor", None)
            cur_factor = (
                fac_e[1]
                if (fac_e is not None and result.H is not None and fac_e[0] == id(result.H))
                else None
            )
            cur_leverage = None
            _reset_anchor(result, cur_factor, None)
            bad_items = []
            normalized = None
            full_normalized_valid = False
            unconverged_stall_rounds = 0
            unconverged_total_rounds = 0
            return True

        def _try_flat_start_fallback() -> bool:
            """热启动叠加坏数据落入局部极小未收敛时，在当前活跃量测集上执行一次平启动兜底重算。"""
            nonlocal result, cur_factor, cur_leverage, bad_items, normalized, full_normalized_valid
            nonlocal x0, flat_fallback_tried, unconverged_stall_rounds, unconverged_total_rounds
            if result.converged or flat_fallback_tried:
                return False
            flat_fallback_tried = True
            x_flat = self.initial_state(flat=True)
            flat_res = self.estimate(
                measurements=base_measurements,
                x0=x_flat,
                verbose=verbose,
                _defer_observability=True,
                _weight_override=masked_weights,
                _value_override=masked_values,
                _shared_aat_solver=shared_aat_solver,
                _shared_aat_plan=shared_aat_plan,
            )
            prev_obj = float(result.objective) if np.isfinite(result.objective) else float("inf")
            flat_obj = float(flat_res.objective) if np.isfinite(flat_res.objective) else float("inf")
            if not (flat_res.converged or (np.isfinite(flat_obj) and flat_obj < 0.8 * prev_obj)):
                return False
            result = flat_res
            if _can_warm_start_from(result):
                x0 = result.x.copy()
            fac_e = getattr(self, "_last_estimate_cholmod_factor", None)
            cur_factor = (
                fac_e[1]
                if (fac_e is not None and result.H is not None and fac_e[0] == id(result.H))
                else None
            )
            cur_leverage = None
            _reset_anchor(result, cur_factor, None)
            bad_items = []
            normalized = None
            full_normalized_valid = False
            unconverged_stall_rounds = 0
            unconverged_total_rounds = 0
            return True

        def _prioritize_unconverged_candidates(cands: List[BadDataItem]) -> List[BadDataItem]:
            """未收敛态下抑制近临界健康测点因 1-h_ii -> 0 放大的假阳性，优先按加权残差 sqrt(w)*|r| 剔除显性坏数据。"""
            if result.converged or len(cands) <= 1:
                return cands
            scored = []
            for item in cands:
                rp = int(getattr(item, "row_pos", -1))
                w_i = float(masked_weights[rp]) if 0 <= rp < n_base else float(getattr(item.measurement, "weight", 0.0))
                r_i = (
                    float(result.residual[rp])
                    if (result.residual is not None and 0 <= rp < result.residual.size)
                    else float(item.residual)
                )
                scored.append((abs(r_i) * float(np.sqrt(max(w_i, 0.0))), item))
            strong = [pair for pair in scored if pair[0] > float(threshold)]
            chosen = strong if strong else scored
            chosen.sort(key=lambda pair: pair[0], reverse=True)
            return [pair[1] for pair in chosen]

        while True:
            if len(removed) >= max_remove:
                if rescue_rounds < 6 and _try_hybrid_boundary_node_rescue():
                    rescue_rounds += 1
                    continue
                if _try_flat_start_fallback():
                    continue
                break
            if (
                (not result.converged)
                and (not bool(getattr(result, "_coarse_stopped", False)))
                and rescue_rounds < 6
                and (unconverged_total_rounds >= 3 or unconverged_stall_rounds >= 2)
                and _try_hybrid_boundary_node_rescue()
            ):
                rescue_rounds += 1
                continue
            if (
                (not result.converged)
                and (not bool(getattr(result, "_coarse_stopped", False)))
                and (unconverged_stall_rounds >= 8 or unconverged_total_rounds >= 20)
            ):
                if rescue_rounds < 6 and _try_hybrid_boundary_node_rescue():
                    rescue_rounds += 1
                    continue
                break
            candidates = _prioritize_unconverged_candidates([
                item
                for item in bad_items
                if (int(item.measurement.idx), str(item.measurement.name)) not in protected_keys
                and not (
                    base_is_pseudo is not None
                    and item.row_pos is not None
                    and 0 <= int(item.row_pos) < n_base
                    and bool(base_is_pseudo[int(item.row_pos)])
                )
            ])
            if not candidates:
                _ensure_exact_result()
                if not full_normalized_valid:
                    _refresh_exact_factor_for_current()
                    final_res, _ = _finalize_masked_result(result, cur_factor, None)
                    bad_items, normalized = self.identify_bad_data(final_res, base_threshold)
                    full_normalized_valid = True
                    keep_rows_cur = np.flatnonzero(active_mask)
                    lev_pair = getattr(self, "_last_leverage_cache", None)
                    if isinstance(lev_pair, tuple) and len(lev_pair) == 3 and lev_pair[0] == id(final_res) and lev_pair[1] is not None:
                        full_lev = np.zeros(n_base, dtype=np.float64)
                        full_lev[keep_rows_cur] = lev_pair[1]
                        cur_leverage = full_lev
                    thr_pair = getattr(self, "_last_adaptive_bad_threshold", None)
                    if isinstance(thr_pair, tuple) and len(thr_pair) == 3 and thr_pair[0] == id(final_res):
                        if thr_pair[2] is not None:
                            full_thr = np.full(n_base, float(thr_pair[1]), dtype=np.float64)
                            full_thr[keep_rows_cur] = thr_pair[2]
                            cur_row_thr = full_thr
                        else:
                            cur_row_thr = None
                    _reset_anchor(result, cur_factor, cur_leverage)
                    for item in bad_items:
                        item.row_pos = int(keep_rows_cur[int(item.row_pos)])
                    candidates = _prioritize_unconverged_candidates([
                        item
                        for item in bad_items
                        if (int(item.measurement.idx), str(item.measurement.name)) not in protected_keys
                        and not (
                            base_is_pseudo is not None
                            and item.row_pos is not None
                            and 0 <= int(item.row_pos) < n_base
                            and bool(base_is_pseudo[int(item.row_pos)])
                        )
                    ])
                    if not candidates:
                        if result.converged:
                            self._last_bad_data_cache = (id(final_res), float(base_threshold), list(bad_items), normalized)
                            return final_res, removed
                        if rescue_rounds < 6 and _try_hybrid_boundary_node_rescue():
                            rescue_rounds += 1
                            continue
                        break
                else:
                    if rescue_rounds < 6 and _try_hybrid_boundary_node_rescue():
                        rescue_rounds += 1
                        continue
                    break

            def _apply_screen_outcome(
                add_rows: Sequence[int],
                prev_x_in: np.ndarray,
                prev_H_in,
                prev_weights_in: np.ndarray,
                trial_coarse: bool,
                used_anchor_wb: bool,
                wb_cum_rows: Optional[List[int]],
            ):
                """对已接受的批删/单删结果执行低成本筛查，并在延迟 Woodbury 遇终轮认证时自动刷新精确因子。"""
                nonlocal result, cur_factor, cur_leverage, bad_items, normalized, full_normalized_valid
                nonlocal anchor_factor, anchor_H, anchor_weights, anchor_x, anchor_leverage, anchor_removed_rows
                trial_fac_entry = getattr(self, "_last_estimate_cholmod_factor", None)
                trial_factor = (
                    trial_fac_entry[1]
                    if (trial_fac_entry is not None and result.H is not None and trial_fac_entry[0] == id(result.H))
                    else None
                )
                trial_deferred = (
                    bool(getattr(result, "_factor_deferred", False))
                    and not bool(getattr(result, "_inplace_factor_modified", True))
                    and getattr(trial_factor, "is_woodbury_solver", False)
                )
                if trial_deferred:
                    if used_anchor_wb and wb_cum_rows is not None:
                        anchor_removed_rows = list(wb_cum_rows)
                    else:
                        anchor_factor = cur_factor
                        anchor_H = prev_H_in
                        anchor_weights = prev_weights_in.copy()
                        anchor_x = prev_x_in.copy()
                        anchor_leverage = cur_leverage
                        anchor_removed_rows = [int(r) for r in add_rows]
                state_delta = float(np.max(np.abs(result.x - prev_x_in))) if prev_x_in.size else 0.0
                rem_arg = int(add_rows[0]) if len(add_rows) == 1 else list(add_rows)
                use_anc_direct_prop = (
                    trial_deferred
                    and anchor_leverage is not None
                    and anchor_factor is not None
                    and anchor_H is not None
                    and anchor_weights is not None
                    and anchor_x is not None
                    and len(anchor_removed_rows) > 0
                    and getattr(trial_factor, "_factor", None) is anchor_factor
                    and anchor_x.size == result.x.size
                )
                screen_start = time.perf_counter() if self.profile_enabled else None
                if use_anc_direct_prop:
                    anc_dx_vec = result.x - anchor_x
                    anc_dx_inf = float(np.max(np.abs(anc_dx_vec))) if anc_dx_vec.size else 0.0
                    anc_rem_arg = int(anchor_removed_rows[0]) if len(anchor_removed_rows) == 1 else list(anchor_removed_rows)
                    status, prop_lev, hit_rows, hit_norms = screen_bad_data_after_removal(
                        anchor_leverage,
                        anchor_H,
                        anchor_weights,
                        anc_rem_arg,
                        anchor_factor,
                        result.residual,
                        result.H,
                        masked_weights,
                        base_is_pseudo,
                        trial_factor,
                        threshold,
                        state_delta_inf=anc_dx_inf,
                        state_delta_vec=anc_dx_vec,
                        active_mask=active_mask,
                        keep_shape=True,
                        allow_fast_final_verify=fast_final_verify,
                        u_col_cache=anchor_u_cache,
                    )
                else:
                    status, prop_lev, hit_rows, hit_norms = screen_bad_data_after_removal(
                        cur_leverage,
                        prev_H_in,
                        prev_weights_in,
                        rem_arg,
                        cur_factor,
                        result.residual,
                        result.H,
                        masked_weights,
                        base_is_pseudo,
                        trial_factor,
                        threshold,
                        state_delta_inf=state_delta,
                        state_delta_vec=(result.x - prev_x_in) if prev_x_in.size else None,
                        active_mask=active_mask,
                        keep_shape=True,
                        allow_fast_final_verify=fast_final_verify,
                    )
                if status == "need_full" and trial_deferred:
                    saved_anc_lev = anchor_leverage
                    saved_anc_H = anchor_H
                    saved_anc_w = anchor_weights
                    saved_anc_x = anchor_x
                    saved_anc_rows = list(anchor_removed_rows)
                    cur_factor = trial_factor
                    if _refresh_exact_factor_for_current():
                        trial_factor = cur_factor
                        trial_deferred = False
                        if (
                            saved_anc_lev is not None
                            and saved_anc_H is not None
                            and saved_anc_w is not None
                            and saved_anc_x is not None
                            and 1 <= len(saved_anc_rows) <= wb_max_rank
                        ):
                            anc_delta_vec = (result.x - saved_anc_x) if saved_anc_x.size == result.x.size else None
                            anc_delta_inf = float(np.max(np.abs(anc_delta_vec))) if (anc_delta_vec is not None and anc_delta_vec.size) else 0.0
                            anc_rem_arg = int(saved_anc_rows[0]) if len(saved_anc_rows) == 1 else saved_anc_rows
                            status, prop_lev, hit_rows, hit_norms = screen_bad_data_after_removal(
                                saved_anc_lev,
                                saved_anc_H,
                                saved_anc_w,
                                anc_rem_arg,
                                trial_factor,
                                result.residual,
                                result.H,
                                masked_weights,
                                base_is_pseudo,
                                trial_factor,
                                threshold,
                                state_delta_inf=anc_delta_inf,
                                state_delta_vec=anc_delta_vec,
                                active_mask=active_mask,
                                keep_shape=True,
                                allow_fast_final_verify=fast_final_verify,
                            )
                if status in ("fast_clean", "verified_clean") and not trial_coarse:
                    bad_items = []
                    normalized = hit_norms
                    cur_leverage = prop_lev
                    cur_factor = trial_factor
                    full_normalized_valid = True
                    if not trial_deferred:
                        _reset_anchor(result, cur_factor, cur_leverage)
                    if screen_start is not None:
                        dt = time.perf_counter() - screen_start
                        if status == "verified_clean":
                            self._record_profile_time("bad_data.gain_inverse", dt)
                        self._record_profile_time("bad_data.total", dt)
                    return None
                if status == "candidates":
                    bad_items = []
                    for r_pos, r_norm in zip(hit_rows.tolist(), hit_norms.tolist()):
                        meas = (
                            measurement_from_table_row(base_table, r_pos)
                            if base_table is not None
                            else base_measurements[r_pos]
                        )
                        measured_value = (
                            float(base_table.value[r_pos])
                            if base_table is not None
                            else float(meas.value)
                        )
                        bad_items.append(
                            BadDataItem(
                                measurement=meas,
                                residual=float(result.residual[r_pos]),
                                normalized_residual=float(r_norm),
                                estimated_value=float(result.z_est[r_pos]),
                                measured_value=measured_value,
                                row_pos=r_pos,
                            )
                        )
                    bad_items = _filter_fast_bad_items(bad_items, result, masked_weights, active_mask)
                    normalized = None
                    cur_leverage = prop_lev
                    cur_factor = trial_factor
                    full_normalized_valid = False
                    if not trial_deferred:
                        _reset_anchor(result, cur_factor, cur_leverage)
                    if screen_start is not None:
                        dt = time.perf_counter() - screen_start
                        self._record_profile_time("bad_data.gain_inverse", dt)
                        self._record_profile_time("bad_data.total", dt)
                    return None
                if _ensure_exact_result():
                    trial_factor = cur_factor
                else:
                    cur_factor = trial_factor
                    if _refresh_exact_factor_for_current():
                        trial_factor = cur_factor
                keep_rows_cur = np.flatnonzero(active_mask)
                sliced_res, _ = _finalize_masked_result(result, trial_factor, None)
                bad_items, normalized_sliced = self.identify_bad_data(sliced_res, base_threshold)
                lev_pair = getattr(self, "_last_leverage_cache", None)
                if isinstance(lev_pair, tuple) and len(lev_pair) == 3 and lev_pair[0] == id(sliced_res) and lev_pair[1] is not None:
                    full_lev = np.zeros(n_base, dtype=np.float64)
                    full_lev[keep_rows_cur] = lev_pair[1]
                    cur_leverage = full_lev
                else:
                    cur_leverage = None
                thr_pair = getattr(self, "_last_adaptive_bad_threshold", None)
                if isinstance(thr_pair, tuple) and len(thr_pair) == 3 and thr_pair[0] == id(sliced_res):
                    if thr_pair[2] is not None:
                        full_thr = np.full(n_base, float(thr_pair[1]), dtype=np.float64)
                        full_thr[keep_rows_cur] = thr_pair[2]
                        cur_row_thr = full_thr
                    else:
                        cur_row_thr = None
                cur_factor = trial_factor
                _reset_anchor(result, cur_factor, cur_leverage)
                if not bad_items:
                    if result.converged:
                        self._last_bad_data_cache = (id(sliced_res), float(base_threshold), [], normalized_sliced)
                        return sliced_res
                    normalized = None
                    full_normalized_valid = True
                    return None
                for item in bad_items:
                    item.row_pos = int(keep_rows_cur[int(item.row_pos)])
                normalized = np.zeros(n_base, dtype=np.float64)
                normalized[keep_rows_cur] = normalized_sliced
                full_normalized_valid = True
                return None
            accepted_removal = False
            remaining_quota = max_remove - len(removed)
            cur_H_csr = (
                (result.H if getattr(result.H, "format", None) == "csr" else result.H.tocsr())
                if (result.H is not None and is_sparse_matrix(result.H) and int(result.H.shape[0]) == n_base)
                else None
            )
            cur_col_degrees = (
                compute_active_column_degrees(cur_H_csr, active_mask, masked_weights)
                if cur_H_csr is not None
                else None
            )
            if batch_cooldown > 0:
                batch_cooldown -= 1
            elif len(candidates) >= 2 and remaining_quota >= 2 and max_remove_per_round >= 2 and result.H is not None:
                valid_cands = [
                    item
                    for item in candidates
                    if item.row_pos is not None
                    and 0 <= int(item.row_pos) < n_base
                    and active_mask[int(item.row_pos)]
                    and not _is_critical_leverage_row(int(item.row_pos))
                ]
                if len(valid_cands) >= 2:
                    cands_norms = [float(it.normalized_residual) for it in valid_cands]
                    batch_min_norm = float(threshold)
                    batch_hops = 3
                    batch_max = min(max_remove_per_round, remaining_quota)
                    if dynamic_batch_cap is not None:
                        batch_max = min(batch_max, dynamic_batch_cap)
                    batch_local = (
                        select_independent_bad_data_batch(
                            result.H,
                            [int(it.row_pos) for it in valid_cands],
                            cands_norms,
                            threshold,
                            min_batch_norm=batch_min_norm,
                            max_batch=batch_max,
                            min_hops=batch_hops,
                        )
                        if batch_max >= 2
                        else []
                    )
                    if len(batch_local) >= 2 and cur_H_csr is not None and cur_col_degrees is not None:
                        safe_pos, _ = filter_batch_preserving_column_coverage(
                            cur_H_csr,
                            [int(valid_cands[i].row_pos) for i in batch_local],
                            cur_col_degrees,
                        )
                        batch_local = [batch_local[p] for p in safe_pos]
                    if len(batch_local) >= 2 and (int(np.count_nonzero(active_mask)) - len(batch_local)) >= self.n_state:
                        trial_batch_items = [valid_cands[i] for i in batch_local]
                        # 批删试探失败时自动折半重试一次，再失败则收缩上限并进入退避冷却，避免反复整批撞墙死循环
                        for half_attempt in range(2):
                            batch_items = trial_batch_items
                            batch_rows = [int(it.row_pos) for it in batch_items]
                            prev_H = result.H
                            prev_weights = masked_weights.copy()
                            prev_x = result.x
                            for rp in batch_rows:
                                masked_weights[rp] = 0.0
                                active_mask[rp] = False
                            trial_x0, trial_sanitized = _warm_start_x0(result, x0)
                            if result.converged and not trial_sanitized:
                                wb_rows, wb_prev_w, wb_prev_H, wb_defer, wb_used_anc = _prepare_woodbury_args(
                                    batch_rows, prev_x, prev_H, prev_weights
                                )
                            else:
                                wb_rows, wb_prev_w, wb_prev_H, wb_defer, wb_used_anc = None, None, None, False, False
                            # 剩余配额与待删显性强坏数据仍充足时，批删中间轮次安全复用粗收敛早停，避免无谓精磨
                            batch_coarse_thr = (
                                coarse_thr
                                if (
                                    coarse_thr is not None
                                    and half_attempt == 0
                                    and (remaining_quota - len(batch_rows)) >= 2
                                    and (len(valid_cands) - len(batch_rows)) >= 10
                                )
                                else None
                            )
                            trial_result = self.estimate(
                                measurements=base_measurements,
                                x0=trial_x0,
                                verbose=verbose,
                                _defer_observability=True,
                                _weight_override=masked_weights,
                                _value_override=masked_values,
                                _shared_aat_solver=shared_aat_solver,
                                _shared_aat_plan=shared_aat_plan,
                                _woodbury_removed_rows=wb_rows,
                                _woodbury_prev_weights=wb_prev_w,
                                _woodbury_prev_H=wb_prev_H,
                                _coarse_bad_data_threshold=batch_coarse_thr,
                                _defer_factor_refresh=wb_defer and len(batch_rows) < 8,
                                _woodbury_u_cache=anchor_u_cache,
                                _hint_lm_damping=0.0 if trial_sanitized else float(getattr(result, "_final_lm_damping", 0.0)),
                            )
                            trial_obs = getattr(trial_result, "observability", None)
                            trial_coarse_stopped = bool(getattr(trial_result, "_coarse_stopped", False))
                            prev_obj = float(result.objective) if np.isfinite(result.objective) else float("inf")
                            cur_obj = float(trial_result.objective) if np.isfinite(trial_result.objective) else float("inf")
                            trial_conv_rejected = (
                                (not trial_result.converged and not trial_coarse_stopped)
                                if result.converged
                                else (
                                    (not trial_result.converged and not trial_coarse_stopped)
                                    and (not _unconv_obj_improved(prev_obj, cur_obj))
                                )
                            )
                            if _removal_causes_singularity(trial_result, result) or trial_conv_rejected:
                                for rp in batch_rows:
                                    masked_weights[rp] = effective_base_weights[rp]
                                    active_mask[rp] = True
                                if getattr(trial_result, "_inplace_factor_modified", True):
                                    _reset_anchor(result, None, None)
                                if shared_aat_solver is not None and getattr(shared_aat_solver, "_cholmod_disabled", False):
                                    shared_aat_solver = CholmodAAtNormalEquationSolver(assume_fixed_pattern=True)
                                batch_ok_streak = 0
                                next_half = len(batch_items) // 2
                                dynamic_batch_cap = max(2, next_half)
                                if half_attempt == 0 and next_half >= 2:
                                    trial_batch_items = batch_items[:next_half]
                                    continue
                                batch_fail_streak += 1
                                batch_cooldown = min(6, batch_fail_streak + (1 if next_half < 2 else 0))
                                break
                            batch_fail_streak = 0
                            batch_ok_streak += 1
                            if dynamic_batch_cap is not None and batch_ok_streak >= 2:
                                next_cap = dynamic_batch_cap * 2
                                dynamic_batch_cap = None if next_cap >= max_remove_per_round else next_cap
                                batch_ok_streak = 0
                            if trial_result.converged:
                                unconverged_stall_rounds = 0
                                unconverged_total_rounds = 0
                                if _can_warm_start_from(trial_result):
                                    x0 = trial_result.x.copy()
                            elif trial_coarse_stopped:
                                # 大批量粗早停仅重置停滞计数，不重置未收敛累计轮次，避免拖延尾段贴边自救
                                unconverged_stall_rounds = 0
                            else:
                                significant_batch_drop = np.isfinite(cur_obj) and cur_obj < 0.90 * prev_obj
                                if not significant_batch_drop:
                                    unconverged_total_rounds += 1
                                unconverged_stall_rounds = (
                                    0
                                    if (
                                        np.isfinite(cur_obj)
                                        and (cur_obj < 0.95 * prev_obj or (prev_obj - cur_obj) >= min_unconv_obj_drop)
                                    )
                                    else (unconverged_stall_rounds + 1)
                                )
                            removed.extend(batch_items)
                            need_slice_finalize = True
                            result = trial_result
                            early_done = _apply_screen_outcome(
                                batch_rows,
                                prev_x,
                                prev_H,
                                prev_weights,
                                trial_coarse_stopped,
                                wb_used_anc,
                                wb_rows,
                            )
                            if early_done is not None:
                                return early_done, removed
                            accepted_removal = True
                            break
            if accepted_removal:
                continue
            single_rejects = 0
            for worst in candidates:
                if worst.row_pos is not None and 0 <= int(worst.row_pos) < n_base and active_mask[int(worst.row_pos)]:
                    remove_pos = int(worst.row_pos)
                else:
                    remove_pos = base_measurements.index(worst.measurement)
                if int(np.count_nonzero(active_mask)) - 1 < self.n_state or _is_critical_leverage_row(remove_pos):
                    protected_keys.add((int(worst.measurement.idx), str(worst.measurement.name)))
                    continue
                if cur_H_csr is not None and cur_col_degrees is not None:
                    safe_single, _ = filter_batch_preserving_column_coverage(cur_H_csr, [remove_pos], cur_col_degrees)
                    if not safe_single:
                        protected_keys.add((int(worst.measurement.idx), str(worst.measurement.name)))
                        continue
                prev_H = result.H
                prev_weights = masked_weights.copy()
                prev_x = result.x
                masked_weights[remove_pos] = 0.0
                active_mask[remove_pos] = False
                trial_x0, trial_sanitized = _warm_start_x0(result, x0)
                if result.converged and not trial_sanitized:
                    wb_rows, wb_prev_w, wb_prev_H, wb_defer, wb_used_anc = _prepare_woodbury_args(
                        [remove_pos], prev_x, prev_H, prev_weights
                    )
                else:
                    wb_rows, wb_prev_w, wb_prev_H, wb_defer, wb_used_anc = None, None, None, False, False
                trial_result = self.estimate(
                    measurements=base_measurements,
                    x0=trial_x0,
                    verbose=verbose,
                    _defer_observability=True,
                    _weight_override=masked_weights,
                    _value_override=masked_values,
                    _shared_aat_solver=shared_aat_solver,
                    _shared_aat_plan=shared_aat_plan,
                    _woodbury_removed_rows=wb_rows,
                    _woodbury_prev_weights=wb_prev_w,
                    _woodbury_prev_H=wb_prev_H,
                    _defer_factor_refresh=wb_defer,
                    _woodbury_u_cache=anchor_u_cache,
                    _hint_lm_damping=0.0 if trial_sanitized else float(getattr(result, "_final_lm_damping", 0.0)),
                )
                trial_obs = getattr(trial_result, "observability", None)
                trial_coarse_stopped = bool(getattr(trial_result, "_coarse_stopped", False))
                prev_obj = float(result.objective) if np.isfinite(result.objective) else float("inf")
                cur_obj = float(trial_result.objective) if np.isfinite(trial_result.objective) else float("inf")
                single_conv_rejected = (
                    (not trial_result.converged and not trial_coarse_stopped)
                    if result.converged
                    else (
                        (not trial_result.converged and not trial_coarse_stopped)
                        and (not _unconv_obj_improved(prev_obj, cur_obj))
                    )
                )
                if _removal_causes_singularity(trial_result, result) or single_conv_rejected:
                    # O(1) 零拷贝回滚临界量测并标记保护，严防排除坏数据导致原问题奇异
                    masked_weights[remove_pos] = effective_base_weights[remove_pos]
                    active_mask[remove_pos] = True
                    protected_keys.add((int(worst.measurement.idx), str(worst.measurement.name)))
                    if getattr(trial_result, "_inplace_factor_modified", True):
                        _reset_anchor(result, None, None)
                    if shared_aat_solver is not None and getattr(shared_aat_solver, "_cholmod_disabled", False):
                        shared_aat_solver = CholmodAAtNormalEquationSolver(assume_fixed_pattern=True)
                    single_rejects += 1
                    if single_rejects >= 12:
                        break
                    continue
                if trial_result.converged or trial_coarse_stopped:
                    unconverged_stall_rounds = 0
                    unconverged_total_rounds = 0
                    if trial_result.converged and _can_warm_start_from(trial_result):
                        x0 = trial_result.x.copy()
                else:
                    unconverged_total_rounds += 1
                    unconverged_stall_rounds = (
                        0
                        if (
                            np.isfinite(cur_obj)
                            and (cur_obj < 0.95 * prev_obj or (prev_obj - cur_obj) >= min_unconv_obj_drop)
                        )
                        else (unconverged_stall_rounds + 1)
                    )
                removed.append(worst)
                need_slice_finalize = True
                result = trial_result
                early_done = _apply_screen_outcome(
                    [remove_pos],
                    prev_x,
                    prev_H,
                    prev_weights,
                    trial_coarse_stopped,
                    wb_used_anc,
                    wb_rows,
                )
                if early_done is not None:
                    return early_done, removed
                accepted_removal = True
                break
            if not accepted_removal:
                if not full_normalized_valid:
                    bad_items = []
                    continue
                if rescue_rounds < 6 and _try_hybrid_boundary_node_rescue():
                    rescue_rounds += 1
                    continue
                if _try_warm_continuation():
                    continue
                if _try_flat_start_fallback():
                    continue
                break
        _ensure_exact_result()
        if not result.converged and _try_warm_continuation():
            _ensure_exact_result()
        if not result.converged and _try_flat_start_fallback():
            _ensure_exact_result()
        if not full_normalized_valid:
            _refresh_exact_factor_for_current()
        final_res, final_norm = _finalize_masked_result(result, cur_factor, normalized)
        self._last_bad_data_cache = (
            (id(final_res), float(base_threshold), list(bad_items), final_norm)
            if full_normalized_valid
            else None
        )
        return final_res, removed

    def print_state(self, x: np.ndarray, limit: int = 20) -> None:
        delegate = self._delegate()
        if delegate is not None:
            delegate.print_state(x, limit=limit)
            return
        ac_x, dc_x, hybrid_x = self._split_state(x)
        if self._ac_sub_estimator is not None:
            print("Estimated AC subsystem state:")
            self._ac_sub_estimator.print_state(ac_x, limit=limit)
        if self._dc_sub_estimator is not None:
            print("Estimated DC subsystem state:")
            self._dc_sub_estimator.print_state(dc_x, limit=limit)
        if self.dcac_converters:
            print("Estimated DCAC converter states:")
            for k, conv in enumerate(self.dcac_converters[:limit]):
                base = 3 * k
                print(
                    f"  {conv.name:14s} "
                    f"P_DC={hybrid_x[base]:.9f} P_AC={hybrid_x[base + 1]:.9f} Q_AC={hybrid_x[base + 2]:.9f}"
                )


def _print_observability(result: ObservabilityResult) -> None:
    print(
        "Observability: "
        f"observable={result.observable}, rank={result.rank}/{result.state_count}, "
        f"measurements={result.measurement_count}, deficiency={result.deficiency}"
    )
    if result.weak_states:
        print("Weak states:")
        for label, score in result.weak_states[:10]:
            print(f"  {label}: {score:.3e}")


def _print_bad_data(items: Sequence[BadDataItem], normalized: np.ndarray, threshold: float, top: int = 10) -> None:
    max_norm = float(np.max(normalized)) if normalized.size else 0.0
    print(f"Bad data: threshold={threshold:.3f}, max_normalized_residual={max_norm:.3e}, count={len(items)}")
    for item in list(items)[:top]:
        meas = item.measurement
        print(
            f"  idx={meas.idx} name={meas.name} type={meas.meas_type} device={meas.device} "
            f"z={item.measured_value:.9g} h={item.estimated_value:.9g} "
            f"res={item.residual:.3e} rn={item.normalized_residual:.3e}"
        )


def _profile_time_items(estimator: HybridStateEstimator) -> List[Tuple[str, float]]:
    items = list(getattr(estimator, "profile_times", {}).items())
    delegate = estimator._delegate()
    if delegate is not None:
        items.extend(
            (f"delegate.{name}", value)
            for name, value in getattr(delegate, "profile_times", {}).items()
        )
    else:
        for side, sub_estimator in (
            ("ac", getattr(estimator, "_ac_sub_estimator", None)),
            ("dc", getattr(estimator, "_dc_sub_estimator", None)),
        ):
            items.extend(
                (f"{side}.{name}", value)
                for name, value in getattr(sub_estimator, "profile_times", {}).items()
            )
    for side, sub_estimator in estimator.fluid_estimators.items():
        items.extend(
            (f"{side}.{name}", value)
            for name, value in getattr(sub_estimator, "profile_times", {}).items()
        )
    return sorted(items)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid electric/heat/gas/hydrogen/steam weighted least-squares state estimation."
    )
    parser.add_argument("--case", default=str(DEFAULT_CASE), help="Hybrid network E file.")
    parser.add_argument("--meas", default=str(DEFAULT_MEAS), help="Measurement E file.")
    parser.add_argument("--para", default=str(DEFAULT_SE_PARAMETER_FILE), help="State-estimation algorithm parameter file.")
    parser.add_argument("--tol", type=float, default=None, help="Override state correction convergence tolerance.")
    parser.add_argument("--max-iter", type=int, default=None, help="Override maximum WLS iterations.")
    parser.add_argument("--diff-step", type=float, default=None, help="Override derivative check step parameter.")
    parser.add_argument("--bad-threshold", type=float, default=None, help="Override normalized residual bad-data threshold.")
    parser.add_argument("--max-remove", type=int, default=None, help="Override maximum removed bad data count.")
    parser.add_argument(
        "--max-remove-per-round",
        type=int,
        default=None,
        help="Override maximum bad data removed per identification round.",
    )
    parser.add_argument("--flat-start", action="store_true", default=None, help="Use flat hybrid state.")
    parser.add_argument("--remove-bad-data", action="store_true", help="Iteratively remove the largest bad datum.")
    parser.add_argument("--print-state", action="store_true", help="Print estimated states.")
    parser.add_argument("--quiet", action="store_true", help="Suppress WLS iteration process output.")
    parser.add_argument("--profile", action="store_true", help="Print initialization profile timings.")
    parser.add_argument(
        "--result-mode",
        default=None,
        choices=("full", "summary", "array", "none"),
        help="SEResult payload mode: full, summary, array, or none.",
    )
    parser.add_argument(
        "--power-flow-linear-solver",
        default=None,
        help="Sparse solver used by AC/DC load-flow seeds. Default: umfpack for hybrid cases with DC/coupling.",
    )
    args = parser.parse_args(argv)

    estimator = HybridStateEstimator(
        e_file=Path(args.case),
        meas_file=Path(args.meas),
        tol=args.tol,
        max_iter=args.max_iter,
        diff_step=args.diff_step,
        flat_start=args.flat_start,
        parameter_file=Path(args.para),
        profile=args.profile,
        auto_prepare=False,
        power_flow_linear_solver=args.power_flow_linear_solver,
    )
    estimator.prepare()
    bad_threshold = estimator.params.bad_threshold if args.bad_threshold is None else args.bad_threshold
    estimator.run(
        result_mode=args.result_mode if args.result_mode is not None else "none",
        remove_bad_data=args.remove_bad_data,
        bad_threshold=bad_threshold,
        max_remove=args.max_remove,
        max_remove_per_round=args.max_remove_per_round,
        verbose=not args.quiet,
    )
    if estimator.observability_result is not None:
        _print_observability(estimator.observability_result)
    result = estimator.estimate_result
    removed = estimator.removed_bad_data
    if removed:
        print("Removed bad data:")
        for item in removed:
            print(f"  idx={item.measurement.idx} name={item.measurement.name} rn={item.normalized_residual:.3e}")
    bad_items = estimator.bad_items
    normalized = estimator.normalized_residual
    if result is not None:
        print(
            "State estimation: "
            f"converged={result.converged}, iterations={result.iterations}, "
            f"objective={result.objective:.6e}, max_dx={result.max_correction:.3e}, "
            f"residual_inf={result.residual_inf:.3e}"
        )
        _print_bad_data(bad_items, normalized, bad_threshold)
    if estimator.fluid_estimators:
        print("Fluid state estimation:")
        for name, fluid_estimator in estimator.fluid_estimators.items():
            fluid_result = fluid_estimator.result
            print(
                f"  {name}: converged={fluid_result.converged}, "
                f"observable={fluid_result.observability.observable}, "
                f"rank={fluid_result.observability.rank}/{fluid_result.observability.state_count}, "
                f"iterations={fluid_result.iterations}, residual={fluid_result.residual_inf:.6e}, "
                f"bad_data={len(fluid_estimator.bad_data)}"
            )
    if estimator.multi_energy_result.couplings:
        print("Multi-energy couplings:")
        for item in estimator.multi_energy_result.couplings:
            print(f"  {item.table_name}:{item.name}: status={item.status}")
    print(
        "Multi-energy result: "
        f"converged={estimator.multi_energy_result.converged}, "
        f"observable={estimator.multi_energy_result.observable}"
    )
    if args.print_state and result is not None:
        estimator.print_state(result.x)
    if args.profile:
        print("Profile:")
        for name, value in _profile_time_items(estimator):
            print(f"  {name}={value:.6f}s")
    return 0 if estimator.multi_energy_result.converged and estimator.multi_energy_result.observable else 1


if __name__ == "__main__":
    raise SystemExit(main())
