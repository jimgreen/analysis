"""Shared sparse WLS state estimator for heat, gas, and hydrogen networks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import sys

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import maximum_bipartite_matching, structural_rank
from scipy.sparse.linalg import MatrixRankWarning


ROOT_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, ROOT_DIR / "model", ROOT_DIR / "lfcore", ROOT_DIR / "secore"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithm_parameters import (
    DEFAULT_SE_PARAMETER_FILE,
    StateEstimationParameters,
    load_se_parameters,
    resolve_bad_data_limits,
)
from lfcore.common import allocate_limited_residual
from model.fluid_model import FluidNetwork
from model.meas_array_model import build_meas_ppc_from_e_file
from model.meas_model import (
    MEAS_STATUS_INVALID,
    MEAS_STATUS_NORMAL,
    MEAS_STATUS_PSEUDO,
    MEAS_STATUS_REMOVED,
    BadDataItem,
    EstimateResult,
    Measurement,
    MeasurementList,
    MeasurementTable,
    ObservabilityResult,
    measurement_table_from_measurements,
)
from model.meas_type import DEVICE_TYPE_CODES, DEVICE_TYPE_NAMES, MEAS_TYPE_CODES, MEAS_TYPE_NAMES
from secore.se_math import (
    CholmodAAtNormalEquationPlan,
    CholmodAAtNormalEquationSolver,
    NormalEquationSolver,
    compute_active_column_degrees,
    compute_normalized_residuals_for_bad_data,
    filter_batch_preserving_column_coverage,
    measurement_leverage_for_bad_data,
    select_independent_bad_data_batch,
)
from secore.se_result import SEResult


SUPPORTED_MEASUREMENT_TYPES = frozenset(
    {
        "PRESSURE",
        "PRESSURE_FROM",
        "PRESSURE_TO",
        "FLOW",
        "FLOW_FROM",
        "FLOW_TO",
        "T_SUPPLY",
        "T_RETURN",
        "TS_FROM",
        "TS_TO",
        "TR_FROM",
        "TR_TO",
        "HEAT",
        "ENTHALPY",
        "TEMPERATURE",
        "H_FROM",
        "H_TO",
        "T_FROM",
        "T_TO",
    }
)
_DEVICE_TYPE_ALIASES = {
    "HeatSource2": "HeatSource",
    "HeatLoad2": "HeatLoad",
}


class FluidStateEstimator:
    """Estimate fluid-network pressure/flow and optional thermal states."""

    def __init__(
        self,
        network: FluidNetwork,
        measurement_file=None,
        *,
        measurements: Optional[Sequence[Measurement]] = None,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        flat_start: Optional[bool] = None,
        bad_threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        max_remove_per_round: Optional[int] = None,
        parameter_file=DEFAULT_SE_PARAMETER_FILE,
        parameters: Optional[StateEstimationParameters] = None,
        verbose: bool = False,
    ):
        if not hasattr(network, "prepare") or not hasattr(network, "potential_power"):
            raise ValueError("FluidStateEstimator requires a FluidNetwork-compatible input")
        self.network = network
        self.measurement_file = None if measurement_file is None else Path(measurement_file)
        self._provided_measurements = measurements
        override_kwargs = dict(
            tol=tol,
            max_iter=max_iter,
            flat_start=flat_start,
            bad_threshold=bad_threshold,
        )
        if max_remove is not None:
            override_kwargs["max_remove"] = max_remove
        if max_remove_per_round is not None:
            override_kwargs["max_remove_per_round"] = max_remove_per_round
        self.params = (parameters or load_se_parameters(parameter_file)).with_overrides(**override_kwargs)
        self.tol = float(self.params.tol)
        self.max_iter = int(self.params.max_iter)
        self.bad_threshold = float(self.params.bad_threshold)
        self.verbose = bool(verbose)
        self.prepared = False
        self.measurements: MeasurementList = MeasurementList()
        self.active_measurements: MeasurementList = MeasurementList()
        self.prefiltered_measurements: List[Tuple[Measurement, str]] = []
        self.measurement_groups: Dict[Tuple[str, str], np.ndarray] = {}
        self.state_labels: List[str] = []
        self.result: Optional[EstimateResult] = None
        self.bad_data: List[BadDataItem] = []
        self.normalized_residual = np.empty(0, dtype=np.float64)
        self.se_result = SEResult()

    def prepare(self) -> "FluidStateEstimator":
        net = self.network.prepare()
        self.n_potential = len(net.nodes)
        self.base_regulated_flow = self.n_potential
        self.n_regulated = int(net.regulated_edge_pos.size)
        self.base_supply_temperature = self.base_regulated_flow + self.n_regulated
        self.base_temperature = self.base_supply_temperature
        self.base_return_temperature = self.base_temperature + (
            len(net.nodes) if net.thermal else 0
        )
        self.base_enthalpy = self.base_temperature + (
            net.temperature_state_count if net.thermal else 0
        )
        self.state_count = self.base_enthalpy + (len(net.nodes) if net.steam else 0)
        self.regulated_state_by_edge = np.full(len(net.edges), -1, dtype=np.int64)
        if net.regulated_edge_pos.size:
            self.regulated_state_by_edge[net.regulated_edge_pos] = (
                self.base_regulated_flow + np.arange(net.regulated_edge_pos.size, dtype=np.int64)
            )
        self.state_labels = [f"{net.prefix}Node:{name}:PRESSURE" for name in net.node_name.tolist()]
        self.state_labels.extend(
            f"{self._controller_device_type()}:{net.edge_name[pos]}:FLOW"
            for pos in net.regulated_edge_pos.tolist()
        )
        if net.thermal:
            self.state_labels.extend(
                f"HeatNode:{name}:{'TEMPERATURE' if net.node_explicit_return[pos] else 'T_SUPPLY'}"
                for pos, name in enumerate(net.node_name.tolist())
            )
            self.state_labels.extend(
                f"HeatNode:{net.node_name[pos]}:T_RETURN"
                for pos in np.flatnonzero(~net.node_explicit_return).tolist()
            )
        if net.steam:
            self.state_labels.extend(f"SteamNode:{name}:ENTHALPY" for name in net.node_name.tolist())
        self.measurements = self._load_measurements()
        self._install_measurement_runtime()
        self.prepared = True
        return self

    def _load_measurements(self) -> MeasurementList:
        if self._provided_measurements is not None:
            return MeasurementList(list(self._provided_measurements))
        if self.measurement_file is None:
            raise ValueError("measurement_file or measurements is required")
        # 优先走 C++17 单遍 .meas 高速解析核，遇非标准格式自动回退逐行解析
        try:
            ppc = build_meas_ppc_from_e_file(
                self.measurement_file,
                include_strings=False,
                include_matrix=False,
            )
            if ppc is not None and "idx_array" in ppc:
                idx_arr = ppc["idx_array"]
                n_rows = int(idx_arr.size)
                if n_rows == 0:
                    return MeasurementList()
                dt_codes = ppc["device_type_code_array"].tolist()
                mt_codes = ppc["meas_type_code_array"].tolist()
                dn_ids = ppc["device_name_id_array"].tolist()
                dev_names = ppc["device_names"].tolist()
                weights = ppc["weight_array"].tolist()
                valids = ppc["valid_array"].tolist()
                values = ppc["value_array"].tolist()
                statuses = ppc["status_array"].tolist()
                idxs = idx_arr.tolist()
                dt_name_get = DEVICE_TYPE_NAMES.get
                mt_name_get = MEAS_TYPE_NAMES.get
                loaded = [
                    Measurement(
                        idx=idx_v,
                        name=f"m_{idx_v}",
                        device_type=dt_name_get(dtc, ""),
                        device_name=str(dev_names[dnid]) if 0 <= dnid < len(dev_names) else "",
                        meas_type=mt_name_get(mtc, ""),
                        weight=w_v,
                        valid=bool(vld_v),
                        value=val_v,
                        status=st_v,
                        device_type_code=dtc,
                        meas_type_code=mtc,
                    )
                    for idx_v, dtc, dnid, mtc, w_v, vld_v, val_v, st_v in zip(
                        idxs, dt_codes, dn_ids, mt_codes, weights, valids, values, statuses
                    )
                ]
                return MeasurementList(loaded)
        except Exception:
            pass
        loaded = Measurement.read_from_file(self.measurement_file)
        return MeasurementList(list(loaded))

    def _device_position(self, measurement: Measurement) -> int:
        net = self.network
        device_type = _DEVICE_TYPE_ALIASES.get(
            str(measurement.device_type),
            str(measurement.device_type),
        )
        name = str(measurement.device_name)
        prefix = net.prefix
        if device_type == f"{prefix}Node":
            return int(net.node_pos_by_name.get(name, -1))
        if device_type in {
            f"{prefix}Pipe",
            f"{prefix}Valve",
            self._controller_device_type(),
        }:
            return int(net.edge_pos_by_name.get(name, -1))
        if device_type == f"{prefix}Source":
            return int(net.source_pos_by_name.get(name, -1))
        if device_type == f"{prefix}Storage":
            return int(net.storage_pos_by_name.get(name, -1))
        if device_type == f"{prefix}Load":
            return int(net.load_pos_by_name.get(name, -1))
        if net.thermal and device_type == "HeatExchanger":
            return int(net.exchanger_pos_by_name.get(name, -1))
        return -1

    def _controller_device_type(self) -> str:
        if self.network.thermal:
            return "HeatPump"
        if self.network.steam:
            return "SteamPressureReducer"
        return f"{self.network.prefix}Compressor"

    def _source_device_types(self) -> frozenset[str]:
        prefix = self.network.prefix
        return frozenset((f"{prefix}Source", f"{prefix}Storage"))

    def _install_measurement_runtime(self) -> None:
        prefix = self.network.prefix
        active = []
        prefiltered = []
        for measurement in self.measurements:
            measurement.device_type = _DEVICE_TYPE_ALIASES.get(
                str(measurement.device_type),
                str(measurement.device_type),
            )
            measurement.meas_type = str(measurement.meas_type).upper()
            measurement.device_type_code = int(DEVICE_TYPE_CODES.get(str(measurement.device_type), 0))
            measurement.meas_type_code = int(MEAS_TYPE_CODES.get(measurement.meas_type, 0))
            measurement.device_pos = self._device_position(measurement)
            reason = ""
            if not str(measurement.device_type).startswith(prefix):
                reason = "foreign network measurement"
            elif measurement.device_pos < 0:
                reason = "unknown device"
            elif measurement.meas_type not in SUPPORTED_MEASUREMENT_TYPES:
                reason = "unsupported measurement type"
            elif not bool(measurement.valid):
                reason = "invalid"
            elif float(measurement.weight) <= 0.0:
                reason = "zero weight"
            if reason:
                measurement.valid = False
                prefiltered.append((measurement, reason))
            else:
                active.append(measurement)
        self.prefiltered_measurements = prefiltered
        self.active_measurements = MeasurementList(active)
        self._rebuild_measurement_groups()

    def _rebuild_measurement_groups(self) -> None:
        groups = defaultdict(list)
        for row, measurement in enumerate(self.active_measurements):
            groups[(str(measurement.device_type), str(measurement.meas_type))].append(row)
        self.measurement_groups = {
            key: np.asarray(rows, dtype=np.int64) for key, rows in groups.items()
        }
        self.active_measurements.table = measurement_table_from_measurements(self.active_measurements)
        self._compiled_meas_plan_key = None
        self._compiled_meas_plan = None

    def initial_state(self, flat_start: Optional[bool] = None) -> np.ndarray:
        if not self.prepared:
            self.prepare()
        net = self.network
        use_flat = self.params.flat_start if flat_start is None else bool(flat_start)
        if use_flat:
            pressure = np.empty(len(net.nodes), dtype=np.float64)
            for island in range(int(net.island_count)):
                island_nodes = np.flatnonzero(net.node_island == island)
                fixed_rows = np.flatnonzero(
                    net.node_island[net.fixed_node_pos] == island
                )
                if fixed_rows.size:
                    pressure_value = float(np.mean(net.fixed_pressure[fixed_rows]))
                else:
                    pressure_value = float(np.mean(net.node_pressure[island_nodes]))
                pressure[island_nodes] = max(pressure_value, 1e-6)
        else:
            pressure = np.maximum(net.node_pressure, 1e-6).copy()
        x = np.empty(self.state_count, dtype=np.float64)
        x[: self.n_potential] = pressure ** int(net.potential_power)
        if self.n_regulated:
            x[self.base_regulated_flow : self.base_supply_temperature] = net.edge_flow_set[
                net.regulated_edge_pos
            ]
        if net.thermal:
            if use_flat:
                source_rows = np.flatnonzero(net.source_supply_temperature > 0.0)
                supply = (
                    float(np.mean(net.source_supply_temperature[source_rows]))
                    if source_rows.size
                    else float(np.mean(net.initial_temperature_state))
                )
                return_value = float(np.mean(net.node_return_temperature))
                temperature = np.full(
                    net.temperature_state_count, supply, dtype=np.float64
                )
                implicit_return_states = np.unique(
                    net.return_temperature_state_by_node[~net.node_explicit_return]
                )
                temperature[implicit_return_states] = return_value
                if net.fixed_temperature_state_pos.size:
                    temperature[
                        net.fixed_temperature_state_pos
                    ] = net.fixed_temperature
                x[self.base_temperature : self.base_enthalpy] = temperature
            else:
                x[self.base_temperature : self.base_enthalpy] = net.initial_temperature_state
        elif net.steam:
            if use_flat:
                source_rows = np.flatnonzero(net.source_enthalpy_set > 0.0)
                enthalpy = (
                    float(np.mean(net.source_enthalpy_set[source_rows]))
                    if source_rows.size
                    else float(np.mean(net.node_enthalpy))
                )
                x[self.base_enthalpy :] = enthalpy
            else:
                x[self.base_enthalpy :] = net.node_enthalpy
        return x

    def _flow_state_arrays(self, x: np.ndarray, *, need_derivative: bool = True):
        net = self.network
        potential = np.asarray(x[: self.n_potential], dtype=np.float64)
        flow = net.edge_flow_set.copy()
        passive = net.passive_edge_pos
        passive_deriv = None
        if passive.size:
            pi = net.edge_i[passive]
            pj = net.edge_j[passive]
            delta = potential[pi] - potential[pj]
            abs_delta = np.abs(delta)
            conductance = net.edge_conductance[passive]
            flow[passive] = conductance * np.sign(delta) * np.sqrt(abs_delta)
            if need_derivative:
                passive_deriv = 0.5 * conductance / np.sqrt(np.maximum(abs_delta, 1e-10))
        regulated = net.regulated_edge_pos
        if regulated.size:
            flow[regulated] = x[self.regulated_state_by_edge[regulated]]
        return potential, flow, passive_deriv

    def _build_flow_derivative_csr(self, passive_deriv: Optional[np.ndarray]) -> csr_matrix:
        net = self.network
        row_parts = []
        col_parts = []
        data_parts = []
        passive = net.passive_edge_pos
        if passive.size and passive_deriv is not None:
            p_rows = np.concatenate((passive, passive)).astype(np.int32, copy=False)
            p_cols = np.concatenate((net.edge_i[passive], net.edge_j[passive])).astype(np.int32, copy=False)
            p_data = np.concatenate((passive_deriv, -passive_deriv))
            row_parts.append(p_rows)
            col_parts.append(p_cols)
            data_parts.append(p_data)
        regulated = net.regulated_edge_pos
        if regulated.size:
            row_parts.append(regulated.astype(np.int32, copy=False))
            col_parts.append(self.regulated_state_by_edge[regulated].astype(np.int32, copy=False))
            data_parts.append(np.ones(regulated.size, dtype=np.float64))
        if not row_parts:
            return csr_matrix((len(net.edges), self.state_count), dtype=np.float64)
        return coo_matrix(
            (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
            shape=(len(net.edges), self.state_count),
        ).tocsr()

    def _flow_state(self, x: np.ndarray):
        potential, flow, passive_deriv = self._flow_state_arrays(x, need_derivative=True)
        derivative = self._build_flow_derivative_csr(passive_deriv)
        return potential, flow, derivative

    def _non_group_pressure_source_nodes(self):
        net = self.network
        cached = getattr(self, "_cached_non_group_pressure_sources", None)
        if cached is not None and cached[0] == id(net):
            return cached[1]
        mask = np.asarray(net.source_is_pressure_controlled, dtype=bool)
        if net.thermal and mask.size:
            mask = mask & ~np.asarray(net.source_explicit_return, dtype=bool)
        positions = np.flatnonzero(mask)
        groups = []
        if positions.size:
            by_node = defaultdict(list)
            for pos in positions.tolist():
                by_node[int(net.source_node_pos[pos])].append(pos)
            for node_pos in sorted(by_node.keys()):
                groups.append((node_pos, np.asarray(by_node[node_pos], dtype=np.int64)))
        self._cached_non_group_pressure_sources = (id(net), groups)
        return groups

    def _source_flow_and_derivative(
        self,
        x: np.ndarray,
        flow: np.ndarray,
        flow_derivative: Optional[csr_matrix],
        *,
        need_derivative: bool = True,
    ):
        net = self.network
        source_flow = net.source_flow_set.copy()
        non_group_nodes = self._non_group_pressure_source_nodes()
        has_pressure_sources = bool(net.pressure_source_groups) or bool(non_group_nodes)
        if not has_pressure_sources:
            if not need_derivative:
                return source_flow, None
            return source_flow, [csr_matrix((1, self.state_count), dtype=np.float64) for _ in net.sources]

        source_derivatives = (
            [csr_matrix((1, self.state_count), dtype=np.float64) for _ in net.sources]
            if need_derivative
            else None
        )
        node_edge_balance = np.asarray(net.incidence @ flow, dtype=np.float64)
        node_target_derivative = (
            -(net.incidence @ flow_derivative).tocsr()
            if (need_derivative and flow_derivative is not None)
            else None
        )
        for source_positions in net.pressure_source_groups:
            node_pos = int(net.source_supply_node_pos[source_positions[0]])
            target = float(
                net.demand[node_pos]
                - net.fixed_injection[node_pos]
                - node_edge_balance[node_pos]
            )
            allocated = allocate_limited_residual(
                net.source_flow_set[source_positions],
                target,
                lower=net.source_flow_min[source_positions],
                upper=net.source_flow_max[source_positions],
                alpha=net.source_alpha[source_positions],
            )
            source_flow[source_positions] = allocated
            if need_derivative and node_target_derivative is not None:
                step = max(1e-7, abs(target) * 1e-7)
                allocated_plus = allocate_limited_residual(
                    net.source_flow_set[source_positions],
                    target + step,
                    lower=net.source_flow_min[source_positions],
                    upper=net.source_flow_max[source_positions],
                    alpha=net.source_alpha[source_positions],
                )
                shares = (allocated_plus - allocated) / step
                target_row = node_target_derivative.getrow(node_pos)
                for local, source_pos in enumerate(source_positions.tolist()):
                    source_derivatives[source_pos] = target_row * float(shares[local])
        for node_pos, pressure_positions in non_group_nodes:
            target = float(net.demand[node_pos] - node_edge_balance[node_pos] - net.fixed_injection[node_pos])
            allocated = allocate_limited_residual(
                net.source_flow_set[pressure_positions],
                target,
                lower=net.source_flow_min[pressure_positions],
                upper=net.source_flow_max[pressure_positions],
                alpha=net.source_alpha[pressure_positions],
            )
            source_flow[pressure_positions] = allocated
            if need_derivative and node_target_derivative is not None:
                step = max(1e-7, abs(target) * 1e-7)
                allocated_plus = allocate_limited_residual(
                    net.source_flow_set[pressure_positions],
                    target + step,
                    lower=net.source_flow_min[pressure_positions],
                    upper=net.source_flow_max[pressure_positions],
                    alpha=net.source_alpha[pressure_positions],
                )
                shares = (allocated_plus - allocated) / step
                target_row = node_target_derivative.getrow(node_pos)
                for local, source_pos in enumerate(pressure_positions.tolist()):
                    source_derivatives[source_pos] = target_row * float(shares[local])
        return source_flow, source_derivatives

    def _get_compiled_meas_plan(self, active: Sequence[Measurement]):
        use_cache = active is self.active_measurements
        if use_cache:
            key = (id(active), len(active), id(self.network))
            if getattr(self, "_compiled_meas_plan_key", None) == key and getattr(self, "_compiled_meas_plan", None) is not None:
                return self._compiled_meas_plan
        net = self.network
        node_type = f"{net.prefix}Node"
        edge_types = {f"{net.prefix}Pipe", f"{net.prefix}Valve", self._controller_device_type()}
        source_types = self._source_device_types()
        load_type = f"{net.prefix}Load"

        passive_lookup = np.full(len(net.edges), -1, dtype=np.int64)
        if net.passive_edge_pos.size:
            passive_lookup[net.passive_edge_pos] = np.arange(net.passive_edge_pos.size, dtype=np.int64)

        # 统一压力通道（合并 Node/Edge/Source/Load 全部节点压力量测）
        p_rows: List[int] = []
        p_nodes: List[int] = []
        # 管段流量通道
        edge_flow_rows: List[int] = []
        edge_flow_pos: List[int] = []
        edge_flow_sign: List[float] = []
        edge_flow_passive_rows: List[int] = []
        edge_flow_passive_idx: List[int] = []
        edge_flow_passive_i: List[int] = []
        edge_flow_passive_j: List[int] = []
        edge_flow_passive_sign: List[float] = []
        edge_flow_reg_rows: List[int] = []
        edge_flow_reg_col: List[int] = []
        edge_flow_reg_sign: List[float] = []
        # 线性状态直测通道：val = x[col] * scale + offset，导数 = scale
        lin_rows: List[int] = []
        lin_cols: List[int] = []
        lin_scale: List[float] = []
        lin_offset: List[float] = []
        # 管段热损上游追踪通道（覆盖蒸汽网 H/T_FROM/TO 与显式回水热网 T_FROM/TO）
        adv_rows: List[int] = []
        adv_pos: List[int] = []
        adv_i: List[int] = []
        adv_j: List[int] = []
        adv_is_from: List[bool] = []
        adv_state_i: List[int] = []
        adv_state_j: List[int] = []
        adv_loss: List[float] = []
        adv_amb: List[float] = []
        adv_out_scale: List[float] = []
        adv_out_offset: List[float] = []
        adv_passive_idx: List[int] = []
        adv_reg_col: List[int] = []
        # 常量通道（如固定 Load FLOW / 非显式回水 HeatLoad HEAT）
        const_rows: List[int] = []
        const_vals: List[float] = []

        fallback_items = []
        has_source_meas = False
        fallback_needs_flow_deriv = False

        cp = max(float(net.medium.heat_capacity), 1e-12) if (net.thermal or net.steam) else 1.0
        steam_t_scale = 1.0 / cp
        steam_t_offset = (
            float(net.medium.reference_temperature) - float(net.medium.reference_enthalpy) / cp
            if net.steam
            else 0.0
        )
        amb_temp = float(net.medium.ambient_temperature) if net.thermal else 0.0
        amb_enth = float(net.medium.ambient_enthalpy) if net.steam else 0.0

        for row, measurement in enumerate(active):
            device_type = str(measurement.device_type)
            meas_type = str(measurement.meas_type)
            pos = int(measurement.device_pos)
            if device_type == node_type:
                if meas_type == "PRESSURE":
                    p_rows.append(row)
                    p_nodes.append(pos)
                elif net.thermal and meas_type in ("T_SUPPLY", "TEMPERATURE"):
                    lin_rows.append(row)
                    lin_cols.append(self.base_temperature + int(net.supply_temperature_state_by_node[pos]))
                    lin_scale.append(1.0)
                    lin_offset.append(0.0)
                elif net.thermal and meas_type == "T_RETURN":
                    lin_rows.append(row)
                    lin_cols.append(self.base_temperature + int(net.return_temperature_state_by_node[pos]))
                    lin_scale.append(1.0)
                    lin_offset.append(0.0)
                elif net.steam and meas_type == "ENTHALPY":
                    lin_rows.append(row)
                    lin_cols.append(self.base_enthalpy + pos)
                    lin_scale.append(1.0)
                    lin_offset.append(0.0)
                elif net.steam and meas_type == "TEMPERATURE":
                    lin_rows.append(row)
                    lin_cols.append(self.base_enthalpy + pos)
                    lin_scale.append(steam_t_scale)
                    lin_offset.append(steam_t_offset)
                else:
                    fallback_items.append((row, measurement))
            elif device_type in edge_types:
                ei = int(net.edge_i[pos])
                ej = int(net.edge_j[pos])
                if meas_type in ("FLOW", "FLOW_FROM", "FLOW_TO"):
                    sign = -1.0 if meas_type == "FLOW_TO" else 1.0
                    edge_flow_rows.append(row)
                    edge_flow_pos.append(pos)
                    edge_flow_sign.append(sign)
                    p_idx = int(passive_lookup[pos])
                    if p_idx >= 0:
                        edge_flow_passive_rows.append(row)
                        edge_flow_passive_idx.append(p_idx)
                        edge_flow_passive_i.append(ei)
                        edge_flow_passive_j.append(ej)
                        edge_flow_passive_sign.append(sign)
                    else:
                        reg_col = int(self.regulated_state_by_edge[pos])
                        if reg_col >= 0:
                            edge_flow_reg_rows.append(row)
                            edge_flow_reg_col.append(reg_col)
                            edge_flow_reg_sign.append(sign)
                elif meas_type == "PRESSURE_FROM":
                    p_rows.append(row)
                    p_nodes.append(ei)
                elif meas_type == "PRESSURE_TO":
                    p_rows.append(row)
                    p_nodes.append(ej)
                elif net.thermal and not net.node_explicit_return[ei] and meas_type in ("TS_FROM", "TS_TO", "TR_FROM", "TR_TO"):
                    nd = ei if meas_type.endswith("FROM") else ej
                    smap = net.supply_temperature_state_by_node if meas_type.startswith("TS") else net.return_temperature_state_by_node
                    lin_rows.append(row)
                    lin_cols.append(self.base_temperature + int(smap[nd]))
                    lin_scale.append(1.0)
                    lin_offset.append(0.0)
                elif net.thermal and net.node_explicit_return[ei] and meas_type in ("T_FROM", "T_TO"):
                    adv_rows.append(row)
                    adv_pos.append(pos)
                    adv_i.append(ei)
                    adv_j.append(ej)
                    adv_is_from.append(meas_type == "T_FROM")
                    adv_state_i.append(self.base_temperature + int(net.supply_temperature_state_by_node[ei]))
                    adv_state_j.append(self.base_temperature + int(net.supply_temperature_state_by_node[ej]))
                    adv_loss.append(float(net.edge_heat_loss[pos]))
                    adv_amb.append(amb_temp)
                    adv_out_scale.append(1.0)
                    adv_out_offset.append(0.0)
                    adv_passive_idx.append(int(passive_lookup[pos]))
                    adv_reg_col.append(int(self.regulated_state_by_edge[pos]))
                elif net.steam and meas_type in ("H_FROM", "H_TO", "T_FROM", "T_TO"):
                    is_t = meas_type.startswith("T")
                    adv_rows.append(row)
                    adv_pos.append(pos)
                    adv_i.append(ei)
                    adv_j.append(ej)
                    adv_is_from.append(meas_type.endswith("FROM"))
                    adv_state_i.append(self.base_enthalpy + ei)
                    adv_state_j.append(self.base_enthalpy + ej)
                    adv_loss.append(float(net.edge_heat_loss[pos]))
                    adv_amb.append(amb_enth)
                    adv_out_scale.append(steam_t_scale if is_t else 1.0)
                    adv_out_offset.append(steam_t_offset if is_t else 0.0)
                    adv_passive_idx.append(int(passive_lookup[pos]))
                    adv_reg_col.append(int(self.regulated_state_by_edge[pos]))
                else:
                    fallback_items.append((row, measurement))
                    fallback_needs_flow_deriv = True
            elif device_type in source_types:
                sn = int(net.source_supply_node_pos[pos])
                rn = int(net.source_return_node_pos[pos])
                if meas_type in ("PRESSURE", "PRESSURE_TO"):
                    p_rows.append(row)
                    p_nodes.append(sn)
                elif meas_type == "PRESSURE_FROM":
                    p_rows.append(row)
                    p_nodes.append(rn)
                elif net.thermal and meas_type == "T_SUPPLY":
                    lin_rows.append(row)
                    lin_cols.append(self.base_temperature + int(net.supply_temperature_state_by_node[sn]))
                    lin_scale.append(1.0)
                    lin_offset.append(0.0)
                elif net.thermal and meas_type == "T_RETURN":
                    lin_rows.append(row)
                    lin_cols.append(self.base_temperature + int(net.return_temperature_state_by_node[rn]))
                    lin_scale.append(1.0)
                    lin_offset.append(0.0)
                elif net.steam and meas_type == "ENTHALPY":
                    lin_rows.append(row)
                    lin_cols.append(self.base_enthalpy + sn)
                    lin_scale.append(1.0)
                    lin_offset.append(0.0)
                elif net.steam and meas_type == "TEMPERATURE":
                    lin_rows.append(row)
                    lin_cols.append(self.base_enthalpy + sn)
                    lin_scale.append(steam_t_scale)
                    lin_offset.append(steam_t_offset)
                else:
                    has_source_meas = True
                    fallback_needs_flow_deriv = True
                    fallback_items.append((row, measurement))
            elif device_type == load_type:
                sn = int(net.load_supply_node_pos[pos])
                rn = int(net.load_return_node_pos[pos])
                if meas_type == "FLOW":
                    const_rows.append(row)
                    const_vals.append(float(net.load_flow_set[pos]))
                elif meas_type in ("PRESSURE", "PRESSURE_FROM"):
                    p_rows.append(row)
                    p_nodes.append(sn)
                elif meas_type == "PRESSURE_TO":
                    p_rows.append(row)
                    p_nodes.append(rn)
                elif net.thermal and meas_type == "T_SUPPLY":
                    lin_rows.append(row)
                    lin_cols.append(self.base_temperature + int(net.supply_temperature_state_by_node[sn]))
                    lin_scale.append(1.0)
                    lin_offset.append(0.0)
                elif net.thermal and meas_type == "T_RETURN":
                    lin_rows.append(row)
                    lin_cols.append(self.base_temperature + int(net.return_temperature_state_by_node[rn]))
                    lin_scale.append(1.0)
                    lin_offset.append(0.0)
                elif net.thermal and meas_type == "HEAT" and not net.load_explicit_return[pos]:
                    const_rows.append(row)
                    const_vals.append(float(net.load_heat_power[pos]))
                elif net.steam and meas_type == "ENTHALPY":
                    lin_rows.append(row)
                    lin_cols.append(self.base_enthalpy + sn)
                    lin_scale.append(1.0)
                    lin_offset.append(0.0)
                elif net.steam and meas_type == "TEMPERATURE":
                    lin_rows.append(row)
                    lin_cols.append(self.base_enthalpy + sn)
                    lin_scale.append(steam_t_scale)
                    lin_offset.append(steam_t_offset)
                elif net.steam and meas_type == "HEAT":
                    lf_val = float(net.load_flow_set[pos])
                    lin_rows.append(row)
                    lin_cols.append(self.base_enthalpy + sn)
                    lin_scale.append(lf_val)
                    lin_offset.append(-lf_val * float(net.load_condensate_enthalpy[pos]))
                else:
                    fallback_items.append((row, measurement))
            else:
                fallback_items.append((row, measurement))

        plan = {
            "p_rows": np.asarray(p_rows, dtype=np.int32),
            "p_nodes": np.asarray(p_nodes, dtype=np.int32),
            "edge_flow_rows": np.asarray(edge_flow_rows, dtype=np.int64),
            "edge_flow_pos": np.asarray(edge_flow_pos, dtype=np.int64),
            "edge_flow_sign": np.asarray(edge_flow_sign, dtype=np.float64),
            "edge_flow_passive_rows": np.asarray(edge_flow_passive_rows, dtype=np.int32),
            "edge_flow_passive_idx": np.asarray(edge_flow_passive_idx, dtype=np.int64),
            "edge_flow_passive_i": np.asarray(edge_flow_passive_i, dtype=np.int32),
            "edge_flow_passive_j": np.asarray(edge_flow_passive_j, dtype=np.int32),
            "edge_flow_passive_sign": np.asarray(edge_flow_passive_sign, dtype=np.float64),
            "edge_flow_reg_rows": np.asarray(edge_flow_reg_rows, dtype=np.int32),
            "edge_flow_reg_col": np.asarray(edge_flow_reg_col, dtype=np.int32),
            "edge_flow_reg_sign": np.asarray(edge_flow_reg_sign, dtype=np.float64),
            "lin_rows": np.asarray(lin_rows, dtype=np.int32),
            "lin_cols": np.asarray(lin_cols, dtype=np.int32),
            "lin_scale": np.asarray(lin_scale, dtype=np.float64),
            "lin_offset": np.asarray(lin_offset, dtype=np.float64),
            "adv_rows": np.asarray(adv_rows, dtype=np.int32),
            "adv_pos": np.asarray(adv_pos, dtype=np.int64),
            "adv_i": np.asarray(adv_i, dtype=np.int32),
            "adv_j": np.asarray(adv_j, dtype=np.int32),
            "adv_is_from": np.asarray(adv_is_from, dtype=bool),
            "adv_state_i": np.asarray(adv_state_i, dtype=np.int32),
            "adv_state_j": np.asarray(adv_state_j, dtype=np.int32),
            "adv_loss": np.asarray(adv_loss, dtype=np.float64),
            "adv_amb": np.asarray(adv_amb, dtype=np.float64),
            "adv_out_scale": np.asarray(adv_out_scale, dtype=np.float64),
            "adv_out_offset": np.asarray(adv_out_offset, dtype=np.float64),
            "adv_passive_idx": np.asarray(adv_passive_idx, dtype=np.int64),
            "adv_reg_col": np.asarray(adv_reg_col, dtype=np.int32),
            "const_rows": np.asarray(const_rows, dtype=np.int32),
            "const_vals": np.asarray(const_vals, dtype=np.float64),
            "fallback_items": fallback_items,
            "has_source_meas": has_source_meas,
            "fallback_needs_flow_deriv": fallback_needs_flow_deriv,
        }
        if use_cache:
            self._compiled_meas_plan_key = (id(active), len(active), id(self.network))
            self._compiled_meas_plan = plan
        return plan

    def evaluate(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> np.ndarray:
        if not self.prepared:
            self.prepare()
        active = self.active_measurements if measurements is None else measurements
        plan = self._get_compiled_meas_plan(active)
        potential, flow, _ = self._flow_state_arrays(x, need_derivative=False)
        pressure = np.maximum(potential, 1e-12) ** (1.0 / self.network.potential_power)
        values = np.empty(len(active), dtype=np.float64)

        p_rows = plan["p_rows"]
        if p_rows.size:
            values[p_rows] = pressure[plan["p_nodes"]]
        edge_flow_rows = plan["edge_flow_rows"]
        if edge_flow_rows.size:
            values[edge_flow_rows] = flow[plan["edge_flow_pos"]] * plan["edge_flow_sign"]
        lin_rows = plan["lin_rows"]
        if lin_rows.size:
            values[lin_rows] = x[plan["lin_cols"]] * plan["lin_scale"] + plan["lin_offset"]
        const_rows = plan["const_rows"]
        if const_rows.size:
            values[const_rows] = plan["const_vals"]
        adv_rows = plan["adv_rows"]
        if adv_rows.size:
            f_val = flow[plan["adv_pos"]]
            is_pos = f_val >= 0.0
            up_col = np.where(is_pos, plan["adv_state_i"], plan["adv_state_j"])
            is_up_end = plan["adv_is_from"] == is_pos
            mass = np.maximum(np.abs(f_val), 1e-9)
            factor = np.exp(-plan["adv_loss"] / mass)
            u_up = x[up_col]
            amb = plan["adv_amb"]
            u_val = np.where(is_up_end, u_up, amb + (u_up - amb) * factor)
            values[adv_rows] = u_val * plan["adv_out_scale"] + plan["adv_out_offset"]

        fallback_items = plan["fallback_items"]
        if not fallback_items:
            return values

        source_flow = (
            self._source_flow_and_derivative(x, flow, None, need_derivative=False)[0]
            if plan["has_source_meas"]
            else self.network.source_flow_set
        )
        if self.network.thermal:
            temperature_state = x[self.base_temperature : self.base_enthalpy]
            supply = temperature_state[
                self.network.supply_temperature_state_by_node
            ]
            return_temperature = temperature_state[
                self.network.return_temperature_state_by_node
            ]
        else:
            supply = None
            return_temperature = None
        exchanger_values = self._heat_exchanger_state(x) if self.network.thermal else None
        enthalpy = x[self.base_enthalpy :] if self.network.steam else None
        steam_temperature = self._steam_temperature(enthalpy) if enthalpy is not None else None
        net = self.network
        for row, measurement in fallback_items:
            device_type = str(measurement.device_type)
            meas_type = str(measurement.meas_type)
            pos = int(measurement.device_pos)
            if device_type == f"{net.prefix}Node":
                if meas_type == "PRESSURE":
                    values[row] = pressure[pos]
                elif meas_type == "T_SUPPLY" and supply is not None:
                    values[row] = supply[pos]
                elif meas_type == "T_RETURN" and return_temperature is not None:
                    values[row] = return_temperature[pos]
                elif meas_type == "TEMPERATURE" and supply is not None:
                    values[row] = supply[pos]
                elif meas_type == "ENTHALPY" and enthalpy is not None:
                    values[row] = enthalpy[pos]
                elif meas_type == "TEMPERATURE" and steam_temperature is not None:
                    values[row] = steam_temperature[pos]
                else:
                    values[row] = np.nan
            elif device_type in {
                f"{net.prefix}Pipe",
                f"{net.prefix}Valve",
                self._controller_device_type(),
            }:
                i = int(net.edge_i[pos])
                j = int(net.edge_j[pos])
                edge_values = {
                    "FLOW": flow[pos],
                    "FLOW_FROM": flow[pos],
                    "FLOW_TO": -flow[pos],
                    "PRESSURE_FROM": pressure[i],
                    "PRESSURE_TO": pressure[j],
                }
                if supply is not None:
                    if net.node_explicit_return[i]:
                        factor = float(
                            np.exp(
                                -net.edge_heat_loss[pos]
                                / max(abs(float(flow[pos])), 1e-9)
                            )
                        )
                        ambient = float(net.medium.ambient_temperature)
                        if flow[pos] >= 0.0:
                            t_from = supply[i]
                            t_to = ambient + (supply[i] - ambient) * factor
                        else:
                            t_to = supply[j]
                            t_from = ambient + (supply[j] - ambient) * factor
                        edge_values.update(T_FROM=t_from, T_TO=t_to)
                    else:
                        edge_values.update(
                            TS_FROM=supply[i],
                            TS_TO=supply[j],
                            TR_FROM=return_temperature[i],
                            TR_TO=return_temperature[j],
                        )
                if enthalpy is not None:
                    factor = float(
                        np.exp(-net.edge_heat_loss[pos] / max(abs(float(flow[pos])), 1e-9))
                    )
                    ambient = float(net.medium.ambient_enthalpy)
                    if flow[pos] >= 0.0:
                        h_from = enthalpy[i]
                        h_to = ambient + (enthalpy[i] - ambient) * factor
                    else:
                        h_to = enthalpy[j]
                        h_from = ambient + (enthalpy[j] - ambient) * factor
                    edge_values.update(
                        H_FROM=h_from,
                        H_TO=h_to,
                        T_FROM=self._steam_temperature([h_from])[0],
                        T_TO=self._steam_temperature([h_to])[0],
                    )
                values[row] = edge_values.get(meas_type, np.nan)
            elif device_type in self._source_device_types():
                supply_node = int(net.source_supply_node_pos[pos])
                return_node = int(net.source_return_node_pos[pos])
                if meas_type == "FLOW":
                    values[row] = source_flow[pos]
                elif meas_type == "PRESSURE":
                    values[row] = pressure[supply_node]
                elif meas_type == "PRESSURE_FROM":
                    values[row] = pressure[return_node]
                elif meas_type == "PRESSURE_TO":
                    values[row] = pressure[supply_node]
                elif meas_type == "T_SUPPLY" and supply is not None:
                    values[row] = supply[supply_node]
                elif meas_type == "T_RETURN" and return_temperature is not None:
                    values[row] = return_temperature[return_node]
                elif (
                    meas_type == "HEAT"
                    and supply is not None
                    and return_temperature is not None
                ):
                    values[row] = (
                        source_flow[pos]
                        * float(net.medium.heat_capacity)
                        * (supply[supply_node] - return_temperature[return_node])
                    )
                elif meas_type == "ENTHALPY" and enthalpy is not None:
                    values[row] = enthalpy[supply_node]
                elif meas_type == "TEMPERATURE" and steam_temperature is not None:
                    values[row] = steam_temperature[supply_node]
                else:
                    values[row] = np.nan
            elif device_type == f"{net.prefix}Load":
                supply_node = int(net.load_supply_node_pos[pos])
                return_node = int(net.load_return_node_pos[pos])
                if meas_type == "FLOW":
                    values[row] = net.load_flow_set[pos]
                elif meas_type == "PRESSURE":
                    values[row] = pressure[supply_node]
                elif meas_type == "PRESSURE_FROM":
                    values[row] = pressure[supply_node]
                elif meas_type == "PRESSURE_TO":
                    values[row] = pressure[return_node]
                elif meas_type == "T_SUPPLY" and supply is not None:
                    values[row] = supply[supply_node]
                elif meas_type == "T_RETURN" and return_temperature is not None:
                    values[row] = return_temperature[return_node]
                elif meas_type == "HEAT" and supply is not None:
                    if net.load_explicit_return[pos]:
                        values[row] = (
                            net.load_flow_set[pos]
                            * net.medium.heat_capacity
                            * (supply[supply_node] - return_temperature[return_node])
                        )
                    else:
                        values[row] = net.load_heat_power[pos]
                elif meas_type == "ENTHALPY" and enthalpy is not None:
                    values[row] = enthalpy[supply_node]
                elif meas_type == "TEMPERATURE" and steam_temperature is not None:
                    values[row] = steam_temperature[supply_node]
                elif meas_type == "HEAT" and enthalpy is not None:
                    values[row] = net.load_flow_set[pos] * (
                        enthalpy[supply_node] - net.load_condensate_enthalpy[pos]
                    )
                else:
                    values[row] = np.nan
            elif device_type == "HeatExchanger" and exchanger_values is not None:
                primary_supply = int(net.exchanger_primary_supply[pos])
                primary_return = int(net.exchanger_primary_return[pos])
                secondary_return = int(net.exchanger_secondary_return[pos])
                secondary_supply = int(net.exchanger_secondary_supply[pos])
                primary_heat, secondary_heat, primary_out, secondary_out = exchanger_values
                exchanger_measurements = {
                    "FLOW_FROM": net.exchanger_primary_flow[pos],
                    "FLOW_TO": net.exchanger_secondary_flow[pos],
                    "PRESSURE_FROM": pressure[primary_supply],
                    "PRESSURE_TO": pressure[secondary_return],
                    "TS_FROM": supply[primary_supply],
                    "TR_FROM": primary_out[pos],
                    "TR_TO": return_temperature[secondary_return],
                    "TS_TO": secondary_out[pos],
                    "HEAT": primary_heat[pos],
                }
                values[row] = exchanger_measurements.get(meas_type, np.nan)
            else:
                values[row] = np.nan
        return values

    def _steam_temperature(self, enthalpy) -> np.ndarray:
        net = self.network
        cp = max(float(net.medium.heat_capacity), 1e-12)
        return (
            float(net.medium.reference_temperature)
            + (np.asarray(enthalpy, dtype=np.float64) - float(net.medium.reference_enthalpy)) / cp
        )

    def _heat_exchanger_state(self, x: np.ndarray):
        net = self.network
        temperature_state = x[self.base_temperature : self.base_enthalpy]
        supply = temperature_state[net.supply_temperature_state_by_node]
        return_temperature = temperature_state[net.return_temperature_state_by_node]
        count = int(net.exchanger_i.size)
        primary_heat = np.zeros(count, dtype=np.float64)
        secondary_heat = np.zeros(count, dtype=np.float64)
        primary_out = np.zeros(count, dtype=np.float64)
        secondary_out = np.zeros(count, dtype=np.float64)
        cp = max(float(net.medium.heat_capacity), 1e-12)
        for pos in range(count):
            primary_supply = int(net.exchanger_primary_supply[pos])
            secondary_return = int(net.exchanger_secondary_return[pos])
            primary_mass = max(float(net.exchanger_primary_flow[pos]), 1e-12)
            secondary_mass = max(float(net.exchanger_secondary_flow[pos]), 1e-12)
            if str(net.exchanger_control_type[pos]) == "EFFECTIVENESS":
                primary_heat[pos] = (
                    net.exchanger_effectiveness[pos]
                    * min(primary_mass, secondary_mass)
                    * cp
                    * (supply[primary_supply] - return_temperature[secondary_return])
                )
            else:
                primary_heat[pos] = net.exchanger_heat_set[pos]
            secondary_heat[pos] = primary_heat[pos] * (1.0 - net.exchanger_heat_loss[pos])
            primary_out[pos] = (
                supply[primary_supply] - primary_heat[pos] / (primary_mass * cp)
            )
            secondary_out[pos] = (
                return_temperature[secondary_return]
                + secondary_heat[pos] / (secondary_mass * cp)
            )
        return primary_heat, secondary_heat, primary_out, secondary_out

    def jacobian_sparse(self, x: np.ndarray, measurements: Optional[Sequence[Measurement]] = None) -> csr_matrix:
        active = self.active_measurements if measurements is None else measurements
        plan = self._get_compiled_meas_plan(active)
        net = self.network
        potential, flow, passive_deriv = self._flow_state_arrays(x, need_derivative=True)
        pressure_derivative = (1.0 / net.potential_power) * np.maximum(potential, 1e-12) ** (
            1.0 / net.potential_power - 1.0
        )

        row_parts = []
        col_parts = []
        data_parts = []

        p_rows = plan["p_rows"]
        if p_rows.size:
            p_nodes = plan["p_nodes"]
            row_parts.append(p_rows)
            col_parts.append(p_nodes)
            data_parts.append(pressure_derivative[p_nodes])

        efp_rows = plan["edge_flow_passive_rows"]
        if efp_rows.size and passive_deriv is not None:
            dvals = passive_deriv[plan["edge_flow_passive_idx"]] * plan["edge_flow_passive_sign"]
            row_parts.append(efp_rows)
            col_parts.append(plan["edge_flow_passive_i"])
            data_parts.append(dvals)
            row_parts.append(efp_rows)
            col_parts.append(plan["edge_flow_passive_j"])
            data_parts.append(-dvals)

        efr_rows = plan["edge_flow_reg_rows"]
        if efr_rows.size:
            row_parts.append(efr_rows)
            col_parts.append(plan["edge_flow_reg_col"])
            data_parts.append(plan["edge_flow_reg_sign"])

        lin_rows = plan["lin_rows"]
        if lin_rows.size:
            row_parts.append(lin_rows)
            col_parts.append(plan["lin_cols"])
            data_parts.append(plan["lin_scale"])

        adv_rows = plan["adv_rows"]
        if adv_rows.size:
            f_val = flow[plan["adv_pos"]]
            is_pos = f_val >= 0.0
            up_col = np.where(is_pos, plan["adv_state_i"], plan["adv_state_j"]).astype(np.int32, copy=False)
            is_up_end = plan["adv_is_from"] == is_pos
            mass = np.maximum(np.abs(f_val), 1e-9)
            loss = plan["adv_loss"]
            factor = np.exp(-loss / mass)
            out_scale = plan["adv_out_scale"]
            row_parts.append(adv_rows)
            col_parts.append(up_col)
            data_parts.append(out_scale * np.where(is_up_end, 1.0, factor))

            # 下游端且存在热损时向量化写入对流量的偏导（彻底替代逐行 flow_derivative.getrow）
            dn_mask = (~is_up_end) & (loss > 0.0)
            if np.any(dn_mask):
                dn_idx = np.flatnonzero(dn_mask)
                f_dn = f_val[dn_idx]
                mass_dn = mass[dn_idx]
                sign_dn = np.where(f_dn == 0.0, 1.0, np.sign(f_dn))
                u_up_dn = x[up_col[dn_idx]]
                flow_scale = (
                    (u_up_dn - plan["adv_amb"][dn_idx])
                    * factor[dn_idx]
                    * loss[dn_idx]
                    * sign_dn
                    / (mass_dn * mass_dn)
                    * out_scale[dn_idx]
                )
                p_idx_dn = plan["adv_passive_idx"][dn_idx]
                if passive_deriv is not None:
                    pm = p_idx_dn >= 0
                    if np.any(pm):
                        pi_sel = dn_idx[pm]
                        r_p = adv_rows[pi_sel]
                        dv_p = flow_scale[pm] * passive_deriv[p_idx_dn[pm]]
                        row_parts.append(r_p)
                        col_parts.append(plan["adv_i"][pi_sel])
                        data_parts.append(dv_p)
                        row_parts.append(r_p)
                        col_parts.append(plan["adv_j"][pi_sel])
                        data_parts.append(-dv_p)
                reg_dn = plan["adv_reg_col"][dn_idx]
                rm = reg_dn >= 0
                if np.any(rm):
                    row_parts.append(adv_rows[dn_idx[rm]])
                    col_parts.append(reg_dn[rm])
                    data_parts.append(flow_scale[rm])

        fallback_items = plan["fallback_items"]
        if not fallback_items:
            if not row_parts:
                return csr_matrix((len(active), self.state_count), dtype=np.float64)
            return coo_matrix(
                (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
                shape=(len(active), self.state_count),
            ).tocsr()

        flow_derivative = (
            self._build_flow_derivative_csr(passive_deriv)
            if plan["fallback_needs_flow_deriv"]
            else None
        )
        source_flow, source_derivatives = (
            self._source_flow_and_derivative(x, flow, flow_derivative, need_derivative=True)
            if plan["has_source_meas"]
            else (net.source_flow_set, None)
        )
        rows = []
        cols = []
        data = []

        def append_sparse_row(row_pos: int, sparse_row, scale: float = 1.0) -> None:
            sparse_row = sparse_row.tocsr()
            start, stop = sparse_row.indptr[0], sparse_row.indptr[1]
            count = stop - start
            if count:
                rows.extend([row_pos] * count)
                cols.extend(sparse_row.indices[start:stop].tolist())
                data.extend((sparse_row.data[start:stop] * scale).tolist())

        for row, measurement in fallback_items:
            device_type = str(measurement.device_type)
            meas_type = str(measurement.meas_type)
            pos = int(measurement.device_pos)
            if device_type == f"{net.prefix}Node":
                if meas_type == "PRESSURE":
                    rows.append(row)
                    cols.append(pos)
                    data.append(float(pressure_derivative[pos]))
                elif meas_type == "T_SUPPLY" and net.thermal:
                    rows.append(row)
                    cols.append(
                        self.base_temperature
                        + int(net.supply_temperature_state_by_node[pos])
                    )
                    data.append(1.0)
                elif meas_type == "T_RETURN" and net.thermal:
                    rows.append(row)
                    cols.append(
                        self.base_temperature
                        + int(net.return_temperature_state_by_node[pos])
                    )
                    data.append(1.0)
                elif meas_type == "TEMPERATURE" and net.thermal:
                    rows.append(row)
                    cols.append(
                        self.base_temperature
                        + int(net.supply_temperature_state_by_node[pos])
                    )
                    data.append(1.0)
                elif meas_type == "ENTHALPY" and net.steam:
                    rows.append(row)
                    cols.append(self.base_enthalpy + pos)
                    data.append(1.0)
                elif meas_type == "TEMPERATURE" and net.steam:
                    rows.append(row)
                    cols.append(self.base_enthalpy + pos)
                    data.append(1.0 / max(float(net.medium.heat_capacity), 1e-12))
                continue
            if device_type in {
                f"{net.prefix}Pipe",
                f"{net.prefix}Valve",
                self._controller_device_type(),
            }:
                i = int(net.edge_i[pos])
                j = int(net.edge_j[pos])
                if meas_type in {"FLOW", "FLOW_FROM"}:
                    append_sparse_row(row, flow_derivative.getrow(pos))
                elif meas_type == "FLOW_TO":
                    append_sparse_row(row, flow_derivative.getrow(pos), -1.0)
                elif meas_type == "PRESSURE_FROM":
                    rows.append(row)
                    cols.append(i)
                    data.append(float(pressure_derivative[i]))
                elif meas_type == "PRESSURE_TO":
                    rows.append(row)
                    cols.append(j)
                    data.append(float(pressure_derivative[j]))
                elif net.thermal and meas_type in {
                    "TS_FROM",
                    "TS_TO",
                    "TR_FROM",
                    "TR_TO",
                    "T_FROM",
                    "T_TO",
                }:
                    if net.node_explicit_return[i]:
                        terminal_from = meas_type.endswith("FROM")
                        flow_value = float(flow[pos])
                        upstream_node = i if flow_value >= 0.0 else j
                        upstream_terminal_is_from = flow_value >= 0.0
                        upstream_col = (
                            self.base_temperature
                            + int(net.supply_temperature_state_by_node[upstream_node])
                        )
                        if terminal_from == upstream_terminal_is_from:
                            rows.append(row)
                            cols.append(upstream_col)
                            data.append(1.0)
                        else:
                            mass = max(abs(flow_value), 1e-9)
                            loss = float(net.edge_heat_loss[pos])
                            factor = float(np.exp(-loss / mass))
                            rows.append(row)
                            cols.append(upstream_col)
                            data.append(factor)
                            if loss > 0.0:
                                temperature_state = x[
                                    self.base_temperature : self.base_enthalpy
                                ]
                                upstream_temperature = temperature_state[
                                    net.supply_temperature_state_by_node[upstream_node]
                                ]
                                da_dq = (
                                    factor
                                    * loss
                                    * np.sign(flow_value or 1.0)
                                    / (mass * mass)
                                )
                                append_sparse_row(
                                    row,
                                    flow_derivative.getrow(pos),
                                    (upstream_temperature - net.medium.ambient_temperature)
                                    * da_dq,
                                )
                    else:
                        node_pos = i if meas_type.endswith("FROM") else j
                        state_map = (
                            net.supply_temperature_state_by_node
                            if meas_type.startswith("TS")
                            else net.return_temperature_state_by_node
                        )
                        rows.append(row)
                        cols.append(
                            self.base_temperature + int(state_map[node_pos])
                        )
                        data.append(1.0)
                elif net.steam and meas_type in {"H_FROM", "H_TO", "T_FROM", "T_TO"}:
                    terminal_from = meas_type.endswith("FROM")
                    flow_value = float(flow[pos])
                    upstream_node = i if flow_value >= 0.0 else j
                    upstream_terminal_is_from = flow_value >= 0.0
                    output_scale = 1.0 if meas_type.startswith("H") else 1.0 / max(
                        float(net.medium.heat_capacity), 1e-12
                    )
                    if terminal_from == upstream_terminal_is_from:
                        rows.append(row)
                        cols.append(self.base_enthalpy + upstream_node)
                        data.append(output_scale)
                    else:
                        mass = max(abs(flow_value), 1e-9)
                        loss = float(net.edge_heat_loss[pos])
                        factor = float(np.exp(-loss / mass))
                        rows.append(row)
                        cols.append(self.base_enthalpy + upstream_node)
                        data.append(factor * output_scale)
                        if loss > 0.0:
                            enthalpy = x[self.base_enthalpy :]
                            da_dq = factor * loss * np.sign(flow_value or 1.0) / (mass * mass)
                            flow_scale = (
                                (float(enthalpy[upstream_node]) - float(net.medium.ambient_enthalpy))
                                * da_dq
                                * output_scale
                            )
                            append_sparse_row(row, flow_derivative.getrow(pos), flow_scale)
                continue
            if device_type in self._source_device_types():
                supply_node = int(net.source_supply_node_pos[pos])
                return_node = int(net.source_return_node_pos[pos])
                if meas_type == "FLOW":
                    append_sparse_row(row, source_derivatives[pos])
                elif meas_type == "PRESSURE":
                    rows.append(row)
                    cols.append(supply_node)
                    data.append(float(pressure_derivative[supply_node]))
                elif meas_type == "PRESSURE_FROM":
                    rows.append(row)
                    cols.append(return_node)
                    data.append(float(pressure_derivative[return_node]))
                elif meas_type == "PRESSURE_TO":
                    rows.append(row)
                    cols.append(supply_node)
                    data.append(float(pressure_derivative[supply_node]))
                elif net.thermal and meas_type in {"T_SUPPLY", "T_RETURN"}:
                    node_pos = supply_node if meas_type == "T_SUPPLY" else return_node
                    state_map = (
                        net.supply_temperature_state_by_node
                        if meas_type == "T_SUPPLY"
                        else net.return_temperature_state_by_node
                    )
                    rows.append(row)
                    cols.append(self.base_temperature + int(state_map[node_pos]))
                    data.append(1.0)
                elif net.thermal and meas_type == "HEAT":
                    supply_col = (
                        self.base_temperature
                        + int(net.supply_temperature_state_by_node[supply_node])
                    )
                    return_col = (
                        self.base_temperature
                        + int(net.return_temperature_state_by_node[return_node])
                    )
                    source_value = float(source_flow[pos])
                    heat_capacity = float(net.medium.heat_capacity)
                    temperature = x[self.base_temperature : self.base_enthalpy]
                    supply_state = int(
                        net.supply_temperature_state_by_node[supply_node]
                    )
                    return_state = int(
                        net.return_temperature_state_by_node[return_node]
                    )
                    temperature_gap = float(
                        temperature[supply_state] - temperature[return_state]
                    )
                    append_sparse_row(
                        row,
                        source_derivatives[pos],
                        heat_capacity * temperature_gap,
                    )
                    rows.extend((row, row))
                    cols.extend((supply_col, return_col))
                    data.extend(
                        (
                            source_value * heat_capacity,
                            -source_value * heat_capacity,
                        )
                    )
                elif net.steam and meas_type in {"ENTHALPY", "TEMPERATURE"}:
                    rows.append(row)
                    cols.append(self.base_enthalpy + supply_node)
                    data.append(
                        1.0
                        if meas_type == "ENTHALPY"
                        else 1.0 / max(float(net.medium.heat_capacity), 1e-12)
                    )
                continue
            if device_type == f"{net.prefix}Load":
                supply_node = int(net.load_supply_node_pos[pos])
                return_node = int(net.load_return_node_pos[pos])
                if meas_type == "PRESSURE":
                    rows.append(row)
                    cols.append(supply_node)
                    data.append(float(pressure_derivative[supply_node]))
                elif meas_type == "PRESSURE_FROM":
                    rows.append(row)
                    cols.append(supply_node)
                    data.append(float(pressure_derivative[supply_node]))
                elif meas_type == "PRESSURE_TO":
                    rows.append(row)
                    cols.append(return_node)
                    data.append(float(pressure_derivative[return_node]))
                elif net.thermal and meas_type in {"T_SUPPLY", "T_RETURN"}:
                    node_pos = supply_node if meas_type == "T_SUPPLY" else return_node
                    state_map = (
                        net.supply_temperature_state_by_node
                        if meas_type == "T_SUPPLY"
                        else net.return_temperature_state_by_node
                    )
                    rows.append(row)
                    cols.append(self.base_temperature + int(state_map[node_pos]))
                    data.append(1.0)
                elif net.thermal and meas_type == "HEAT" and net.load_explicit_return[pos]:
                    scale = float(net.load_flow_set[pos] * net.medium.heat_capacity)
                    rows.extend((row, row))
                    cols.extend(
                        (
                            self.base_temperature
                            + int(net.supply_temperature_state_by_node[supply_node]),
                            self.base_temperature
                            + int(net.return_temperature_state_by_node[return_node]),
                        )
                    )
                    data.extend((scale, -scale))
                elif net.steam and meas_type in {"ENTHALPY", "TEMPERATURE"}:
                    rows.append(row)
                    cols.append(self.base_enthalpy + supply_node)
                    data.append(
                        1.0
                        if meas_type == "ENTHALPY"
                        else 1.0 / max(float(net.medium.heat_capacity), 1e-12)
                    )
                elif net.steam and meas_type == "HEAT":
                    rows.append(row)
                    cols.append(self.base_enthalpy + supply_node)
                    data.append(float(net.load_flow_set[pos]))
                # Fixed HeatLoad heat-power measurements do not add a state derivative.
                continue
            if device_type == "HeatExchanger" and net.thermal:
                primary_supply = int(net.exchanger_primary_supply[pos])
                secondary_return = int(net.exchanger_secondary_return[pos])
                primary_mass = max(float(net.exchanger_primary_flow[pos]), 1e-12)
                secondary_mass = max(float(net.exchanger_secondary_flow[pos]), 1e-12)
                loss_factor = 1.0 - float(net.exchanger_heat_loss[pos])
                if str(net.exchanger_control_type[pos]) == "EFFECTIVENESS":
                    heat_gradient = (
                        net.exchanger_effectiveness[pos]
                        * min(primary_mass, secondary_mass)
                        * net.medium.heat_capacity
                    )
                else:
                    heat_gradient = 0.0
                ts_i = self.base_temperature + int(
                    net.supply_temperature_state_by_node[primary_supply]
                )
                tr_j = self.base_temperature + int(
                    net.return_temperature_state_by_node[secondary_return]
                )
                if meas_type == "PRESSURE_FROM":
                    rows.append(row)
                    cols.append(primary_supply)
                    data.append(float(pressure_derivative[primary_supply]))
                elif meas_type == "PRESSURE_TO":
                    rows.append(row)
                    cols.append(secondary_return)
                    data.append(float(pressure_derivative[secondary_return]))
                elif meas_type == "TS_FROM":
                    rows.append(row)
                    cols.append(ts_i)
                    data.append(1.0)
                elif meas_type == "TR_TO":
                    rows.append(row)
                    cols.append(tr_j)
                    data.append(1.0)
                elif meas_type == "HEAT" and heat_gradient:
                    rows.extend((row, row))
                    cols.extend((ts_i, tr_j))
                    data.extend((heat_gradient, -heat_gradient))
                elif meas_type == "TR_FROM":
                    scale = heat_gradient / (primary_mass * net.medium.heat_capacity)
                    rows.extend((row, row))
                    cols.extend((ts_i, tr_j))
                    data.extend((1.0 - scale, scale))
                elif meas_type == "TS_TO":
                    scale = loss_factor * heat_gradient / (secondary_mass * net.medium.heat_capacity)
                    rows.extend((row, row))
                    cols.extend((ts_i, tr_j))
                    data.extend((scale, 1.0 - scale))
        if rows:
            row_parts.append(np.asarray(rows, dtype=np.int32))
            col_parts.append(np.asarray(cols, dtype=np.int32))
            data_parts.append(np.asarray(data, dtype=np.float64))
        if not row_parts:
            return csr_matrix((len(active), self.state_count), dtype=np.float64)
        return coo_matrix(
            (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
            shape=(len(active), self.state_count),
        ).tocsr()

    @staticmethod
    def _rank(H: csr_matrix) -> int:
        if H.shape[1] == 0:
            return 0
        if max(H.shape) <= 300:
            return int(np.linalg.matrix_rank(H.toarray()))
        return int(structural_rank(H))

    def _observability_result(self, H: csr_matrix) -> ObservabilityResult:
        rank = self._rank(H)
        state_count = int(H.shape[1])
        column_norm = np.sqrt(np.asarray(H.power(2).sum(axis=0)).reshape(-1))
        weak_order = np.argsort(column_norm)
        weak = [(int(pos), float(column_norm[pos])) for pos in weak_order[: min(20, state_count)]]
        return ObservabilityResult(
            observable=rank == state_count,
            rank=rank,
            state_count=state_count,
            measurement_count=int(H.shape[0]),
            deficiency=max(0, state_count - rank),
            singular_values=np.asarray([], dtype=np.float64),
            weak_states=weak,
        )

    def _pseudo_candidates(self) -> List[Measurement]:
        net = self.network
        existing = {
            (str(item.device_type), str(item.device_name), str(item.meas_type))
            for item in self.active_measurements
        }
        candidates = []
        next_idx = max((int(item.idx) for item in self.measurements), default=0) + 1

        def add(device_type: str, device_name: str, meas_type: str, value: float, device_pos: int) -> None:
            nonlocal next_idx
            if (device_type, device_name, meas_type) in existing:
                return
            candidates.append(
                Measurement(
                    next_idx,
                    f"pseudo_{device_name}_{meas_type.lower()}",
                    device_type,
                    device_name,
                    meas_type,
                    self.params.pseudo_measurement_weight_pq,
                    True,
                    float(value),
                    status=MEAS_STATUS_PSEUDO,
                    device_type_code=DEVICE_TYPE_CODES[device_type],
                    meas_type_code=MEAS_TYPE_CODES[meas_type],
                    device_pos=device_pos,
                )
            )
            next_idx += 1

        for pos, name in enumerate(net.node_name.tolist()):
            add(f"{net.prefix}Node", str(name), "PRESSURE", net.node_pressure[pos], pos)
        for edge_pos in net.regulated_edge_pos.tolist():
            edge = net.edges[edge_pos]
            device_type = self._controller_device_type()
            add(device_type, edge.name, "FLOW_FROM", net.edge_flow_set[edge_pos], edge_pos)
        if net.thermal:
            for pos, name in enumerate(net.node_name.tolist()):
                if net.node_explicit_return[pos]:
                    add(
                        "HeatNode",
                        str(name),
                        "TEMPERATURE",
                        net.node_temperature[pos],
                        pos,
                    )
                else:
                    add(
                        "HeatNode",
                        str(name),
                        "T_SUPPLY",
                        net.node_supply_temperature[pos],
                        pos,
                    )
                    add(
                        "HeatNode",
                        str(name),
                        "T_RETURN",
                        net.node_return_temperature[pos],
                        pos,
                    )
        if net.steam:
            for pos, name in enumerate(net.node_name.tolist()):
                add("SteamNode", str(name), "ENTHALPY", net.node_enthalpy[pos], pos)
        return candidates

    def _build_pseudo_measurement_for_state_col(
        self,
        col: int,
        next_idx: int,
        existing: set,
        *,
        temp_state_to_spec: Optional[Dict[int, Tuple[str, int, float]]] = None,
    ) -> Optional[Measurement]:
        """O(1) 为指定亏秩状态列构造对应的定向伪量测。"""
        net = self.network
        prefix = net.prefix
        if col < self.n_potential:
            pos = int(col)
            dt = f"{prefix}Node"
            dn = str(net.node_name[pos])
            mt = "PRESSURE"
            if (dt, dn, mt) in existing:
                return None
            return Measurement(
                next_idx,
                f"pseudo_{dn}_pressure",
                dt,
                dn,
                mt,
                self.params.pseudo_measurement_weight_pq,
                True,
                float(net.node_pressure[pos]),
                status=MEAS_STATUS_PSEUDO,
                device_type_code=DEVICE_TYPE_CODES[dt],
                meas_type_code=MEAS_TYPE_CODES[mt],
                device_pos=pos,
            )
        if col < self.base_supply_temperature:
            reg_k = int(col - self.base_regulated_flow)
            if 0 <= reg_k < net.regulated_edge_pos.size:
                edge_pos = int(net.regulated_edge_pos[reg_k])
                edge = net.edges[edge_pos]
                dt = self._controller_device_type()
                dn = str(edge.name)
                mt = "FLOW_FROM"
                if (dt, dn, mt) in existing:
                    return None
                return Measurement(
                    next_idx,
                    f"pseudo_{dn}_flow_from",
                    dt,
                    dn,
                    mt,
                    self.params.pseudo_measurement_weight_pq,
                    True,
                    float(net.edge_flow_set[edge_pos]),
                    status=MEAS_STATUS_PSEUDO,
                    device_type_code=DEVICE_TYPE_CODES[dt],
                    meas_type_code=MEAS_TYPE_CODES[mt],
                    device_pos=edge_pos,
                )
            return None
        if net.thermal and col < self.base_enthalpy:
            t_state = int(col - self.base_temperature)
            if temp_state_to_spec is not None and t_state in temp_state_to_spec:
                mt, pos, val = temp_state_to_spec[t_state]
                dt = "HeatNode"
                dn = str(net.node_name[pos])
                if (dt, dn, mt) in existing:
                    return None
                return Measurement(
                    next_idx,
                    f"pseudo_{dn}_{mt.lower()}",
                    dt,
                    dn,
                    mt,
                    self.params.pseudo_measurement_weight_pq,
                    True,
                    float(val),
                    status=MEAS_STATUS_PSEUDO,
                    device_type_code=DEVICE_TYPE_CODES[dt],
                    meas_type_code=MEAS_TYPE_CODES[mt],
                    device_pos=pos,
                )
            return None
        if net.steam and col >= self.base_enthalpy:
            pos = int(col - self.base_enthalpy)
            if 0 <= pos < len(net.nodes):
                dt = "SteamNode"
                dn = str(net.node_name[pos])
                mt = "ENTHALPY"
                if (dt, dn, mt) in existing:
                    return None
                return Measurement(
                    next_idx,
                    f"pseudo_{dn}_enthalpy",
                    dt,
                    dn,
                    mt,
                    self.params.pseudo_measurement_weight_pq,
                    True,
                    float(net.node_enthalpy[pos]),
                    status=MEAS_STATUS_PSEUDO,
                    device_type_code=DEVICE_TYPE_CODES[dt],
                    meas_type_code=MEAS_TYPE_CODES[mt],
                    device_pos=pos,
                )
        return None

    def analyze_observability(self, *, add_pseudo: bool = True) -> ObservabilityResult:
        if not self.prepared:
            self.prepare()
        x = self.initial_state()
        H = self.jacobian_sparse(x)
        result = self._observability_result(H)
        if result.observable or not add_pseudo:
            self.observability = result
            return result

        max_add = int(self.params.targeted_pseudo_measurement_max)
        added = 0
        # 1. O(nnz) 二分图最大匹配一次性定位全部结构亏秩列并定向批量补齐，彻底消除 O(N^2) 重建 H 瓶颈
        H_nz = H.copy()
        H_nz.eliminate_zeros()
        match_cols = maximum_bipartite_matching(H_nz, perm_type="row")
        unmatched_cols = np.flatnonzero(match_cols < 0)

        if unmatched_cols.size > 0 and max_add > 0:
            net = self.network
            existing = {
                (str(item.device_type), str(item.device_name), str(item.meas_type))
                for item in self.active_measurements
            }
            next_idx = max((int(item.idx) for item in self.measurements), default=0) + 1
            temp_state_to_spec: Optional[Dict[int, Tuple[str, int, float]]] = None
            if net.thermal:
                temp_state_to_spec = {}
                for pos in range(len(net.nodes)):
                    s_st = int(net.supply_temperature_state_by_node[pos])
                    if net.node_explicit_return[pos]:
                        temp_state_to_spec.setdefault(s_st, ("TEMPERATURE", pos, float(net.node_temperature[pos])))
                    else:
                        temp_state_to_spec.setdefault(s_st, ("T_SUPPLY", pos, float(net.node_supply_temperature[pos])))
                        r_st = int(net.return_temperature_state_by_node[pos])
                        temp_state_to_spec.setdefault(r_st, ("T_RETURN", pos, float(net.node_return_temperature[pos])))

            new_pseudos: List[Measurement] = []
            for col in unmatched_cols.tolist():
                if added >= max_add:
                    break
                cand = self._build_pseudo_measurement_for_state_col(
                    int(col),
                    next_idx,
                    existing,
                    temp_state_to_spec=temp_state_to_spec,
                )
                if cand is None:
                    continue
                existing.add((str(cand.device_type), str(cand.device_name), str(cand.meas_type)))
                new_pseudos.append(cand)
                next_idx += 1
                added += 1

            if new_pseudos:
                self.measurements.extend(new_pseudos)
                self.active_measurements.extend(new_pseudos)
                self._rebuild_measurement_groups()
                H = self.jacobian_sparse(x)
                result = self._observability_result(H)

        # 2. 若小规模模型仍存在数值相关性导致的残余秩亏（结构满秩但数值欠定），做少量回退补齐
        if not result.observable and added < max_add and max(H.shape) <= 300:
            current_rank = result.rank
            for candidate in self._pseudo_candidates():
                trial = MeasurementList([*self.active_measurements, candidate])
                trial_H = self.jacobian_sparse(x, trial)
                rank = self._rank(trial_H)
                if rank <= current_rank:
                    continue
                self.measurements.append(candidate)
                self.active_measurements.append(candidate)
                current_rank = rank
                added += 1
                if current_rank == self.state_count or added >= max_add:
                    break
            self._rebuild_measurement_groups()
            H = self.jacobian_sparse(x)
            result = self._observability_result(H)

        self.observability = result
        return result

    def estimate(
        self,
        *,
        x0: Optional[np.ndarray] = None,
        observability: Optional[ObservabilityResult] = None,
        _weight_override: Optional[np.ndarray] = None,
        _hint_lm_damping: float = 0.0,
    ) -> EstimateResult:
        if not self.prepared:
            self.prepare()
        obs = observability or self.analyze_observability(add_pseudo=True)
        x = self.initial_state() if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        table = getattr(self.active_measurements, "table", None)
        if table is not None and int(table.value.size) == len(self.active_measurements):
            z = np.asarray(table.value, dtype=np.float64)
            base_weight = np.asarray(table.weight, dtype=np.float64)
        else:
            z = np.asarray([item.value for item in self.active_measurements], dtype=np.float64)
            base_weight = np.asarray([item.weight for item in self.active_measurements], dtype=np.float64)
        weight = (
            np.asarray(_weight_override, dtype=np.float64)
            if _weight_override is not None
            else base_weight
        )
        converged = False
        max_correction = np.inf
        objective = np.inf
        iteration = 0
        H = csr_matrix((len(z), self.state_count), dtype=np.float64)
        gain = csr_matrix((self.state_count, self.state_count), dtype=np.float64)
        minimum = 1e-12
        use_aat = self.state_count >= 64
        aat_solver = (
            CholmodAAtNormalEquationSolver(assume_fixed_pattern=False, ordering_method="auto", multi_solve=True)
            if use_aat
            else None
        )
        aat_plan: Optional[CholmodAAtNormalEquationPlan] = None
        normal_solver = NormalEquationSolver(assume_fixed_pattern=False)
        cached_z_est = None
        cached_residual = None
        cached_objective = None
        evaluated_at_final_x = False
        lm_damping = max(0.0, float(_hint_lm_damping))
        backtrack_streak = 0
        stagnation_streak = 0
        tried_undamped_escape = False
        last_used_damping = 0.0
        self._last_aat_solver = None
        self._last_weight_used = weight

        def _solve_step(H_cur: csr_matrix, res_cur: np.ndarray, damping_val: float):
            nonlocal aat_plan, gain
            if use_aat and aat_solver is not None and H_cur.nnz > 0:
                if not H_cur.has_sorted_indices:
                    H_cur.sort_indices()
                if aat_plan is None or not aat_plan.matches(H_cur):
                    aat_plan = CholmodAAtNormalEquationPlan.from_jacobian(H_cur)
                A_csc, rhs_vec = aat_plan.assemble(
                    H_cur,
                    res_cur,
                    weight,
                    assume_fixed_weights=False,
                    copy_rhs=False,
                    assume_pattern_matches=True,
                )
                diag_g = np.bincount(
                    A_csc.indices,
                    weights=A_csc.data * A_csc.data,
                    minlength=self.state_count,
                )
                if damping_val > 0.0:
                    # Jacobi 列均衡 + Marquardt 相对阻尼，保证压力/流量/温度/焓值跨量纲尺度不变
                    col_scale = 1.0 / np.sqrt(np.maximum(diag_g, 1e-12))
                    A_scaled = A_csc.copy()
                    A_scaled.data = A_csc.data * col_scale[A_csc.indices]
                    rhs_scaled = rhs_vec * col_scale
                    dx_scaled, _ = aat_solver.solve(
                        A_scaled,
                        rhs_scaled,
                        return_factor_diag=False,
                        damping=damping_val,
                    )
                    dx_out = np.asarray(dx_scaled, dtype=np.float64) * col_scale
                else:
                    dx_raw, _ = aat_solver.solve(
                        A_csc,
                        rhs_vec,
                        return_factor_diag=False,
                        damping=0.0,
                    )
                    dx_out = np.asarray(dx_raw, dtype=np.float64)
                return dx_out, rhs_vec, diag_g
            weighted_H = diags(weight) @ H_cur
            gain_cur = (H_cur.T @ weighted_H).tocsc()
            rhs_vec = np.asarray(H_cur.T @ (weight * res_cur), dtype=np.float64)
            diag_g = np.asarray(gain_cur.diagonal(), dtype=np.float64)
            if damping_val > 0.0:
                gain_reg = (gain_cur + diags(damping_val * np.maximum(diag_g, 1e-12), format="csc")).tocsc()
                dx_raw, _ = normal_solver.solve(gain_reg, rhs_vec, return_factor_diag=False)
            else:
                dx_raw, _ = normal_solver.solve(gain_cur, rhs_vec, return_factor_diag=False)
            gain = gain_cur
            return np.asarray(dx_raw, dtype=np.float64), rhs_vec, diag_g

        for iteration in range(1, self.max_iter + 1):
            if cached_z_est is not None:
                z_est = cached_z_est
                residual = cached_residual
                objective = cached_objective
                cached_z_est = cached_residual = cached_objective = None
            else:
                z_est = self.evaluate(x)
                residual = z - z_est
                objective = float(np.dot(weight, residual * residual))
            H = self.jacobian_sparse(x)

            dx = None
            rhs = None
            diag_g = None
            step_damping = lm_damping
            try:
                dx, rhs, diag_g = _solve_step(H, residual, step_damping)
            except (MatrixRankWarning, RuntimeError, ValueError):
                dx = None
            if dx is None or (dx.size and not np.all(np.isfinite(dx))):
                # 数值病态或近零流量奇异时分级启用 Marquardt 相对阻尼扶正方向
                for beta_try in (1e-4, 1e-2, 1e-1, 1.0, 10.0):
                    if beta_try <= step_damping:
                        continue
                    try:
                        dx_try, rhs_try, diag_try = _solve_step(H, residual, beta_try)
                        if dx_try.size == self.state_count and np.all(np.isfinite(dx_try)):
                            dx, rhs, diag_g = dx_try, rhs_try, diag_try
                            step_damping = beta_try
                            lm_damping = beta_try
                            break
                    except (MatrixRankWarning, RuntimeError, ValueError):
                        continue
            if dx is None or (dx.size and not np.all(np.isfinite(dx))):
                break

            last_used_damping = step_damping
            max_correction = float(np.max(np.abs(dx))) if dx.size else 0.0
            rel_correction = (
                float(np.max(np.abs(dx) / np.maximum(np.abs(x), 1.0))) if dx.size else 0.0
            )
            if self.verbose:
                print(
                    f"Iter {iteration}: objective={objective:.6e}, "
                    f"max_dx={max_correction:.6e}, rel_dx={rel_correction:.3e}, "
                    f"damp={step_damping:.2e}, residual={np.linalg.norm(residual, np.inf):.6e}"
                )
            kkt_inf = (
                float(np.max(np.abs(rhs) / np.maximum(diag_g, 1e-12)))
                if (rhs is not None and diag_g is not None and rhs.size > 0)
                else np.inf
            )
            if step_damping == 0.0 and (
                max_correction < self.tol
                or (rel_correction < self.tol and max_correction < 1e-3)
            ):
                converged = True
                evaluated_at_final_x = True
                break
            if (
                step_damping > 0.0
                and rel_correction < max(1e-5, 10.0 * self.tol)
                and kkt_inf <= max(1e-4, 100.0 * self.tol)
            ):
                converged = True
                evaluated_at_final_x = True
                break

            accepted = False
            step = 1.0
            accepted_step = 0.0
            prev_objective = objective
            # J = sum(w * r^2) 的梯度为 -2 * H^T W r = -2 * rhs
            ls_slope = -2.0 * float(np.dot(rhs, dx)) if (rhs is not None and rhs.size == dx.size) else 0.0
            obj_tol = max(1e-14, 1e-11 * (abs(objective) + 1.0))
            for _ in range(20):
                candidate = x + step * dx
                candidate[: self.n_potential] = np.maximum(candidate[: self.n_potential], minimum)
                candidate_z_est = self.evaluate(candidate)
                candidate_residual = z - candidate_z_est
                candidate_objective = float(np.dot(weight, candidate_residual * candidate_residual))
                if np.isfinite(candidate_objective) and candidate_objective <= objective + obj_tol:
                    x = candidate
                    cached_z_est = candidate_z_est
                    cached_residual = candidate_residual
                    cached_objective = candidate_objective
                    accepted_step = step
                    accepted = True
                    break
                if ls_slope < 0.0 and np.isfinite(candidate_objective):
                    denom = 2.0 * (candidate_objective - objective - ls_slope * step)
                    if np.isfinite(denom) and denom > 1e-30:
                        s_quad = (-ls_slope * step * step) / denom
                        step = float(np.clip(s_quad, 0.1 * step, 0.5 * step))
                    else:
                        step *= 0.5
                else:
                    step *= 0.5
            if not accepted:
                if tried_undamped_escape:
                    # 连续停滞后尝试无阻尼脱困仍无法下降，确认已到达非光滑边界局部极小驻点
                    converged = True
                    evaluated_at_final_x = True
                    break
                if lm_damping < 100.0:
                    lm_damping = min(100.0, max(lm_damping * 10.0, 1e-2))
                    backtrack_streak = 0
                    continue
                if kkt_inf <= max(1e-3, 1000.0 * self.tol) and rel_correction < 1e-3:
                    converged = True
                evaluated_at_final_x = True
                break

            obj_change = abs(prev_objective - cached_objective)
            actual_rel_dx = accepted_step * rel_correction
            # 自适应阻尼调节：满步成功时温和放松，连续回溯/微步时抬升阻尼抑制非光滑零流量震荡
            if accepted_step >= 0.5:
                backtrack_streak = max(0, backtrack_streak - 1)
                if lm_damping > 0.0:
                    lm_damping = lm_damping * 0.25
                    if lm_damping < 1e-6:
                        lm_damping = 0.0
            else:
                backtrack_streak += 1
                if backtrack_streak >= 2 or accepted_step < 1e-2:
                    lm_damping = min(10.0, max(lm_damping * 10.0, 1e-3))
                    backtrack_streak = 0

            # 带阻尼且已达稳态停滞（KKT 梯度与目标函数 7 位双满足）提前优雅收敛
            if (
                step_damping > 0.0
                and rel_correction < 1e-3
                and kkt_inf <= max(1e-4, 100.0 * self.tol)
                and obj_change <= max(1e-12, 1e-7 * (abs(prev_objective) + 1.0))
            ):
                converged = True
                break

            # 非光滑边界（如上游追踪热损 q=0 翻转台阶）局部极小驻点检测：
            # 若目标函数 9 位有效数字不变且实际相对位移 <= max(tol, 1e-6)，先做 1 次无阻尼脱困试探，
            # 试探后仍停滞则确认到达不可微边界极小驻点并优雅退出。
            if (
                obj_change <= max(1e-12, 1e-9 * (abs(prev_objective) + 1.0))
                and actual_rel_dx <= max(1e-6, self.tol)
            ):
                stagnation_streak += 1
                if tried_undamped_escape:
                    converged = True
                    break
                if stagnation_streak >= 3:
                    if step_damping > 0.0:
                        lm_damping = 0.0
                        tried_undamped_escape = True
                        stagnation_streak = 0
                    else:
                        converged = True
                        break
            else:
                stagnation_streak = 0
                tried_undamped_escape = False

        if not evaluated_at_final_x:
            if cached_z_est is not None:
                z_est = cached_z_est
                residual = cached_residual
                objective = cached_objective
            else:
                z_est = self.evaluate(x)
                residual = z - z_est
                objective = float(np.dot(weight, residual * residual))
            if max_correction < self.tol:
                converged = True
            H = self.jacobian_sparse(x)
            gain = (H.T @ (diags(weight) @ H)).tocsr()
        else:
            if gain is None or getattr(gain, "format", None) != "csr" or gain.shape != (self.state_count, self.state_count) or gain.nnz == 0:
                gain = (H.T @ (diags(weight) @ H)).tocsr()

        # 仅当终轮无阻尼且使用 aat_solver 成功分解时，缓存求解器供坏数据杠杆复用
        if use_aat and aat_solver is not None and last_used_damping == 0.0 and evaluated_at_final_x:
            self._last_aat_solver = aat_solver
        else:
            self._last_aat_solver = None

        active_table = getattr(self.active_measurements, "table", None)
        if active_table is None or int(active_table.idx.size) != len(self.active_measurements):
            active_table = measurement_table_from_measurements(self.active_measurements)
        result = EstimateResult(
            converged=bool(converged and obs.observable),
            iterations=int(iteration),
            objective=objective,
            max_correction=float(max_correction),
            residual_inf=float(np.linalg.norm(residual, np.inf)) if residual.size else 0.0,
            x=x,
            z_est=z_est,
            residual=residual,
            H=H,
            gain=gain,
            measurements=list(self.active_measurements),
            observability=obs,
            measurement_table=active_table,
        )
        result._final_lm_damping = float(lm_damping)
        self.result = result
        self._write_back_state(x)
        return result

    def identify_bad_data(
        self,
        result: Optional[EstimateResult] = None,
        threshold: Optional[float] = None,
        *,
        weight_override: Optional[np.ndarray] = None,
    ) -> Tuple[List[BadDataItem], np.ndarray]:
        result = result or self.result
        if result is None:
            raise RuntimeError("state estimation has not run")
        threshold_value = self.bad_threshold if threshold is None else float(threshold)
        n_meas = len(result.measurements)
        if n_meas == 0:
            self.bad_data = []
            self.normalized_residual = np.empty(0, dtype=np.float64)
            return [], self.normalized_residual

        table = getattr(result, "measurement_table", None)
        if weight_override is not None:
            weights = np.asarray(weight_override, dtype=np.float64)
        elif table is not None and int(table.weight.size) == n_meas:
            weights = np.asarray(table.weight, dtype=np.float64)
        else:
            weights = np.asarray([item.weight for item in result.measurements], dtype=np.float64)

        if table is not None and getattr(table, "status_code", None) is not None and int(table.status_code.size) == n_meas:
            is_pseudo = np.asarray(table.status_code, dtype=np.int16) == MEAS_STATUS_PSEUDO
        else:
            is_pseudo = np.asarray(
                [int(getattr(item, "status", MEAS_STATUS_NORMAL)) == MEAS_STATUS_PSEUDO for item in result.measurements],
                dtype=bool,
            )

        # 使用精确杠杆 h_ii = diag(H G^-1 H^T) 计算标准化残差 r_N = |r_i| / sqrt(1/w_i - h_ii)
        leverage = None
        if result.H is not None and result.gain is not None and result.H.shape[0] == n_meas and self.state_count > 0:
            cached_solver = getattr(self, "_last_aat_solver", None)
            cached_fac = cached_solver.last_cholmod_factor if cached_solver is not None else None
            cached_op = cached_solver.last_normal_operand if cached_solver is not None else None
            try:
                leverage = measurement_leverage_for_bad_data(
                    result.H,
                    result.gain,
                    cholmod_factor=cached_fac,
                    aat_operand=cached_op,
                    sparse_threshold=256,
                    sparse_state_limit=100000,
                )
            except Exception:
                leverage = None

        if leverage is not None and leverage.size == n_meas:
            normalized, _ = compute_normalized_residuals_for_bad_data(
                np.asarray(result.residual, dtype=np.float64),
                weights,
                leverage,
                is_pseudo=is_pseudo,
            )
            # 掩蔽权重为 0（已剔除）的行不再上报
            normalized = np.where(weights > 0.0, normalized, 0.0)
        else:
            normalized = np.where(
                (weights > 0.0) & (~is_pseudo),
                np.sqrt(np.maximum(weights, 0.0)) * np.abs(result.residual),
                0.0,
            )

        hit_pos = np.flatnonzero(normalized > threshold_value)
        if hit_pos.size > 1:
            order = np.argsort(-normalized[hit_pos], kind="mergesort")
            hit_pos = hit_pos[order]

        bad = []
        for pos in hit_pos.tolist():
            measurement = result.measurements[pos]
            bad.append(
                BadDataItem(
                    measurement=measurement,
                    residual=float(result.residual[pos]),
                    normalized_residual=float(normalized[pos]),
                    estimated_value=float(result.z_est[pos]),
                    measured_value=float(measurement.value),
                    row_pos=pos,
                )
            )
        self.bad_data = bad
        self.normalized_residual = normalized
        return bad, normalized

    def estimate_with_bad_data_removal(
        self,
        *,
        threshold: Optional[float] = None,
        max_remove: Optional[int] = None,
        max_remove_per_round: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> Tuple[EstimateResult, List[BadDataItem]]:
        """先完整收敛再按标准化残差降序迭代剔除坏数据（受双上限与列覆盖可观测性保护）。"""
        if not self.prepared:
            self.prepare()
        obs = self.analyze_observability(add_pseudo=True)
        threshold_value = self.bad_threshold if threshold is None else float(threshold)
        prev_verbose = self.verbose
        if verbose is not None:
            self.verbose = bool(verbose)
        try:
            table = getattr(self.active_measurements, "table", None)
            n_meas = len(self.active_measurements)
            if table is not None and int(table.weight.size) == n_meas:
                base_weight = np.asarray(table.weight, dtype=np.float64).copy()
                is_pseudo = (
                    np.asarray(table.status_code, dtype=np.int16) == MEAS_STATUS_PSEUDO
                    if getattr(table, "status_code", None) is not None and int(table.status_code.size) == n_meas
                    else np.zeros(n_meas, dtype=bool)
                )
            else:
                base_weight = np.asarray([item.weight for item in self.active_measurements], dtype=np.float64)
                is_pseudo = np.asarray(
                    [int(getattr(item, "status", MEAS_STATUS_NORMAL)) == MEAS_STATUS_PSEUDO for item in self.active_measurements],
                    dtype=bool,
                )

            n_real_valid = int(np.count_nonzero((base_weight > 0.0) & (~is_pseudo)))
            eff_max_remove = self.params.max_remove if max_remove is None else max_remove
            eff_max_per_round = (
                self.params.max_remove_per_round if max_remove_per_round is None else max_remove_per_round
            )
            total_cap, round_cap = resolve_bad_data_limits(
                n_real_valid,
                max_remove=eff_max_remove,
                max_remove_per_round=eff_max_per_round,
            )

            cur_weight = base_weight.copy()
            active_mask = cur_weight > 0.0
            removed_items: List[BadDataItem] = []
            removed_rows_set = set()

            res = self.estimate(observability=obs, _weight_override=cur_weight)
            if not res.converged or total_cap <= 0:
                self.identify_bad_data(res, threshold=threshold_value, weight_override=cur_weight)
                return res, removed_items

            while len(removed_items) < total_cap:
                bad_list, norm_res = self.identify_bad_data(
                    res,
                    threshold=threshold_value,
                    weight_override=cur_weight,
                )
                if not bad_list:
                    break
                cand_rows = [
                    int(it.row_pos)
                    for it in bad_list
                    if int(it.row_pos) not in removed_rows_set
                    and not bool(is_pseudo[int(it.row_pos)])
                ]
                if not cand_rows:
                    break
                cand_norms = [float(norm_res[r]) for r in cand_rows]
                remaining = total_cap - len(removed_items)
                batch_limit = min(round_cap, remaining)
                # 选择图上相互独立的坏数据子批，避免单条坏数据连带邻域引发误报
                selected_pos = select_independent_bad_data_batch(
                    res.H,
                    cand_rows,
                    cand_norms,
                    threshold_value,
                    min_batch_norm=max(threshold_value, 4.5),
                    max_batch=batch_limit,
                    min_hops=3,
                )
                selected = [cand_rows[k] for k in selected_pos] if selected_pos else [cand_rows[0]]

                # 严格保护状态列覆盖可观测性：不得把任意状态变量的全部支撑量测删空
                col_degrees = compute_active_column_degrees(res.H, active_mask, cur_weight)
                safe_local, _skipped = filter_batch_preserving_column_coverage(res.H, selected, col_degrees)
                safe_rows = [selected[k] for k in safe_local]
                if not safe_rows:
                    # 若首批均涉关键列，尝试剩余候选中首条安全行
                    probe_cands = cand_rows[: min(len(cand_rows), 20)]
                    probe_local, _ = filter_batch_preserving_column_coverage(res.H, probe_cands, col_degrees)
                    if probe_local:
                        safe_rows = [probe_cands[probe_local[0]]]
                if not safe_rows:
                    break

                row_to_item = {int(it.row_pos): it for it in bad_list}
                for r_idx in safe_rows:
                    cur_weight[r_idx] = 0.0
                    active_mask[r_idx] = False
                    removed_rows_set.add(r_idx)
                    item = row_to_item[r_idx]
                    item.measurement.valid = False
                    item.measurement.status = MEAS_STATUS_REMOVED
                    removed_items.append(item)

                hint_damp = float(getattr(res, "_final_lm_damping", 0.0))
                res = self.estimate(
                    x0=res.x,
                    observability=obs,
                    _weight_override=cur_weight,
                    _hint_lm_damping=hint_damp,
                )
                if not res.converged:
                    break

            self.bad_data = removed_items
            self.identify_bad_data(res, threshold=threshold_value, weight_override=cur_weight)
            self.bad_data = removed_items
            return res, removed_items
        finally:
            self.verbose = prev_verbose

    def build_se_result(self, result: Optional[EstimateResult] = None) -> SEResult:
        result = result or self.result
        if result is None:
            raise RuntimeError("state estimation has not run")
        self.se_result = SEResult.from_estimate_result(
            result,
            bad_items=self.bad_data,
            normalized_residual=self.normalized_residual,
            prefiltered_measurements=self.prefiltered_measurements,
            all_measurements=self.measurements,
        )
        return self.se_result

    def run(self) -> int:
        if not self.prepared:
            self.prepare()
        observability = self.analyze_observability(add_pseudo=True)
        result = self.estimate(observability=observability)
        self.identify_bad_data(result)
        self.build_se_result(result)
        return 0 if result.converged else 1

    def _write_back_state(self, x: np.ndarray) -> None:
        net = self.network
        pressure = np.maximum(x[: self.n_potential], 1e-12) ** (1.0 / net.potential_power)
        temperature_state = (
            x[self.base_temperature : self.base_enthalpy] if net.thermal else None
        )
        for pos, node in enumerate(net.nodes):
            node.pressure = float(pressure[pos])
            if net.thermal:
                node.supply_temperature = float(
                    temperature_state[net.supply_temperature_state_by_node[pos]]
                )
                node.return_temperature = float(
                    temperature_state[net.return_temperature_state_by_node[pos]]
                )
                if net.node_explicit_return[pos]:
                    node.temperature = node.supply_temperature
            elif net.steam:
                node.enthalpy = float(x[self.base_enthalpy + pos])
                node.temperature = float(self._steam_temperature([node.enthalpy])[0])


def print_fluid_se_result(estimator: FluidStateEstimator, rc: int) -> None:
    result = estimator.result
    if result is None:
        print(f"{estimator.network.prefix} state estimation did not run")
        return
    obs = result.observability
    print(f"{estimator.network.prefix} state estimation: {'converged' if rc == 0 else 'not converged'}")
    print(
        f"  states={obs.state_count}, measurements={obs.measurement_count}, rank={obs.rank}, "
        f"pseudo={estimator.se_result.statistics.pseudo_measurement_count}"
    )
    print(
        f"  iterations={result.iterations}, objective={result.objective:.6e}, "
        f"residual={result.residual_inf:.6e}, bad_data={len(estimator.bad_data)}"
    )
