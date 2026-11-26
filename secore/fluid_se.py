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
from model.meas_array_model import build_meas_ppc_from_e_file, measurement_list_from_meas_ppc
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
    TableBackedMeasurementList,
    measurement_from_table_row,
    measurement_table_from_measurements,
    measurement_table_status_code,
)
from model.meas_type import DEVICE_TYPE_CODES, DEVICE_TYPE_NAMES, MEAS_TYPE_CODES, MEAS_TYPE_NAMES
from secore.se_math import (
    AndersonAccelerator,
    nielsen_lm_damping_update,
    CholmodAAtNormalEquationPlan,
    CholmodAAtNormalEquationSolver,
    NormalEquationSolver,
    check_active_structural_full_rank,
    compute_active_column_degrees,
    compute_adaptive_bad_threshold,
    compute_normalized_residuals_for_bad_data,
    filter_batch_preserving_column_coverage,
    measurement_leverage_for_bad_data,
    select_independent_bad_data_batch,
    solve_box_pinned_normal_equations,
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
        # 避免重复执行 FluidNetwork.prepare()（5w~10w 节点单次耗 0.7~1.4s）
        net = self.network if getattr(self.network, "prepared", False) else self.network.prepare()
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
        # 优先走 C++17 单遍 .meas 高速解析核并直接返回延迟物化 TableBackedMeasurementList（0 对象装箱）
        try:
            ppc = build_meas_ppc_from_e_file(
                self.measurement_file,
                include_strings=False,
                include_matrix=False,
            )
            if ppc is not None and "idx_array" in ppc:
                if int(ppc["idx_array"].size) == 0:
                    return MeasurementList()
                return measurement_list_from_meas_ppc(ppc)
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
        net = self.network
        prefix = net.prefix
        raw_table = getattr(self.measurements, "table", None)
        if (
            isinstance(self.measurements, TableBackedMeasurementList)
            and raw_table is not None
            and int(raw_table.idx.size) == len(self.measurements)
            and getattr(raw_table, "device_name_id", None) is not None
            and getattr(raw_table, "device_names", None) is not None
            and getattr(raw_table, "meas_type_code", None) is not None
        ):
            # 全向量化零对象绑定与预过滤快路径（适用于 10 ~ 100k+ 节点超大 .meas 文件）
            n_all = int(raw_table.idx.size)
            dt_code = np.asarray(raw_table.device_type_code, dtype=np.int16).copy()
            mt_code = np.asarray(raw_table.meas_type_code, dtype=np.int16)
            dn_id = np.asarray(raw_table.device_name_id, dtype=np.int32)
            dev_names_list = [str(x) for x in np.asarray(raw_table.device_names, dtype=object).tolist()]
            n_uniq_names = len(dev_names_list)

            # 别名设备类型码归一化（HeatSource2 -> HeatSource, HeatLoad2 -> HeatLoad）
            for alias_name, canon_name in _DEVICE_TYPE_ALIASES.items():
                a_code = DEVICE_TYPE_CODES.get(alias_name)
                c_code = DEVICE_TYPE_CODES.get(canon_name)
                if a_code is not None and c_code is not None and a_code != c_code:
                    dt_code[dt_code == a_code] = c_code

            node_dt = int(DEVICE_TYPE_CODES.get(f"{prefix}Node", -1))
            pipe_dt = int(DEVICE_TYPE_CODES.get(f"{prefix}Pipe", -1))
            valve_dt = int(DEVICE_TYPE_CODES.get(f"{prefix}Valve", -1))
            ctrl_dt = int(DEVICE_TYPE_CODES.get(self._controller_device_type(), -1))
            src_dt = int(DEVICE_TYPE_CODES.get(f"{prefix}Source", -1))
            sto_dt = int(DEVICE_TYPE_CODES.get(f"{prefix}Storage", -1))
            load_dt = int(DEVICE_TYPE_CODES.get(f"{prefix}Load", -1))
            exch_dt = int(DEVICE_TYPE_CODES.get("HeatExchanger", -1)) if net.thermal else -1

            device_pos = np.full(n_all, -1, dtype=np.int64)
            valid_dn = (dn_id >= 0) & (dn_id < n_uniq_names)

            def _bind_group(mask: np.ndarray, lookup: Dict[str, int]) -> None:
                if not lookup or not np.any(mask):
                    return
                idx_rows = np.flatnonzero(mask & valid_dn)
                if idx_rows.size == 0:
                    return
                # 先在去重设备名表上查一次字典，再 O(1) 向量化索引写回
                mapped_ids = np.fromiter((lookup.get(nm, -1) for nm in dev_names_list), dtype=np.int64, count=n_uniq_names)
                device_pos[idx_rows] = mapped_ids[dn_id[idx_rows]]

            _bind_group(dt_code == node_dt, net.node_pos_by_name)
            edge_mask = (dt_code == pipe_dt) | (dt_code == valve_dt) | (dt_code == ctrl_dt)
            _bind_group(edge_mask, net.edge_pos_by_name)
            _bind_group(dt_code == src_dt, net.source_pos_by_name)
            _bind_group(dt_code == sto_dt, net.storage_pos_by_name)
            _bind_group(dt_code == load_dt, net.load_pos_by_name)
            if exch_dt >= 0:
                _bind_group(dt_code == exch_dt, net.exchanger_pos_by_name)

            allowed_dt_codes = [c for c in (node_dt, pipe_dt, valve_dt, ctrl_dt, src_dt, sto_dt, load_dt, exch_dt) if c >= 0]
            supported_mt_codes = np.asarray(
                [int(MEAS_TYPE_CODES[m]) for m in SUPPORTED_MEASUREMENT_TYPES if m in MEAS_TYPE_CODES],
                dtype=np.int16,
            )
            dt_ok = np.isin(dt_code, np.asarray(allowed_dt_codes, dtype=np.int16))
            mt_ok = np.isin(mt_code, supported_mt_codes)
            raw_valid = np.asarray(raw_table.valid, dtype=bool)
            raw_weight = np.asarray(raw_table.weight, dtype=np.float64)
            raw_value = np.asarray(raw_table.value, dtype=np.float64)
            raw_status = measurement_table_status_code(raw_table)

            # 按需识别停运/关闭/死岛流体设备（仅扫描属于本网但未进入活网表的行）
            inactive_by_type = getattr(net, "inactive_device_names_by_type", None) or {}
            is_dead_or_inactive = np.zeros(n_all, dtype=bool)
            if inactive_by_type:
                dead_candidates = np.flatnonzero(dt_ok & (device_pos < 0) & valid_dn)
                if dead_candidates.size:
                    inactive_by_code = {
                        int(DEVICE_TYPE_CODES[dtype]): name_set
                        for dtype, name_set in inactive_by_type.items()
                        if dtype in DEVICE_TYPE_CODES and name_set
                    }
                    if inactive_by_code:
                        for r in dead_candidates.tolist():
                            dset = inactive_by_code.get(int(dt_code[r]))
                            if dset and dev_names_list[int(dn_id[r])] in dset:
                                is_dead_or_inactive[r] = True

            from secore.meas_prefilter import evaluate_fluid_measurement_prefilter

            pf_keep, pf_reasons = evaluate_fluid_measurement_prefilter(
                values=raw_value,
                weights=raw_weight,
                raw_valid=raw_valid,
                raw_status=raw_status,
                meas_type_code=mt_code,
                params=self.params,
                is_dead_or_inactive=is_dead_or_inactive,
            )
            active_mask = dt_ok & (device_pos >= 0) & mt_ok & pf_keep

            raw_table.device_type_code = dt_code
            raw_table.device_pos = device_pos
            raw_table.valid = active_mask.copy()
            setattr(raw_table, "prefilter_reason", pf_reasons)

            act_rows = np.flatnonzero(active_mask)
            active_table = MeasurementTable(
                idx=np.asarray(raw_table.idx, dtype=np.int64)[act_rows],
                name=np.empty(0, dtype=object),
                device_type=np.empty(0, dtype=object),
                device_name=np.empty(0, dtype=object),
                meas_type=np.empty(0, dtype=object),
                weight=raw_weight[act_rows],
                valid=np.ones(act_rows.size, dtype=bool),
                value=raw_value[act_rows],
                device_type_code=dt_code[act_rows],
                angle_mask=np.zeros(act_rows.size, dtype=bool),
                status_code=raw_status[act_rows].copy(),
                meas_type_code=mt_code[act_rows],
                device_pos=device_pos[act_rows],
                device_name_id=dn_id[act_rows],
                device_names=np.asarray(raw_table.device_names, dtype=object),
            )
            self.active_measurements = TableBackedMeasurementList(active_table, normalized=True)

            bad_rows = np.flatnonzero(~active_mask)
            if bad_rows.size == 0:
                self.prefiltered_measurements = []
            elif bad_rows.size <= 2048:
                prefiltered = []
                for r in bad_rows.tolist():
                    if not bool(dt_ok[r]):
                        reason = "foreign network measurement"
                    elif bool(is_dead_or_inactive[r]):
                        reason = str(pf_reasons[r]) or "dead_island"
                    elif int(device_pos[r]) < 0:
                        reason = "unknown device"
                    elif not bool(mt_ok[r]):
                        reason = "unsupported measurement type"
                    elif str(pf_reasons[r]):
                        reason = str(pf_reasons[r])
                    elif not bool(raw_valid[r]):
                        reason = "invalid"
                    else:
                        reason = "zero weight"
                    m_obj = measurement_from_table_row(raw_table, r)
                    m_obj.valid = False
                    prefiltered.append((m_obj, reason))
                self.prefiltered_measurements = prefiltered
            else:
                # 超大规模下跳过数十万条无效行的对象物化，防止 prepare() 退化
                self.prefiltered_measurements = []
            self._rebuild_measurement_groups()
            return

        from secore.meas_prefilter import check_single_fluid_measurement_prefilter

        inactive_by_type = getattr(net, "inactive_device_names_by_type", None) or {}
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
            is_dead = (
                measurement.device_pos < 0
                and str(measurement.device_name) in inactive_by_type.get(str(measurement.device_type), ())
            )
            reason = ""
            if not str(measurement.device_type).startswith(prefix):
                reason = "foreign network measurement"
            elif is_dead:
                reason = check_single_fluid_measurement_prefilter(
                    value=float(measurement.value),
                    weight=float(measurement.weight),
                    valid=bool(measurement.valid),
                    status=int(getattr(measurement, "status", 0)),
                    meas_type=measurement.meas_type,
                    params=self.params,
                    is_dead_or_inactive=True,
                ) or "dead_island"
            elif measurement.device_pos < 0:
                reason = "unknown device"
            elif measurement.meas_type not in SUPPORTED_MEASUREMENT_TYPES:
                reason = "unsupported measurement type"
            else:
                reason = check_single_fluid_measurement_prefilter(
                    value=float(measurement.value),
                    weight=float(measurement.weight),
                    valid=bool(measurement.valid),
                    status=int(getattr(measurement, "status", 0)),
                    meas_type=measurement.meas_type,
                    params=self.params,
                    is_dead_or_inactive=False,
                )
            if reason:
                measurement.valid = False
                prefiltered.append((measurement, reason))
            else:
                active.append(measurement)
        self.prefiltered_measurements = prefiltered
        self.active_measurements = MeasurementList(active)
        self._rebuild_measurement_groups()

    def _rebuild_measurement_groups(self) -> None:
        table = measurement_table_from_measurements(self.active_measurements)
        self.active_measurements.table = table
        n_rows = int(table.idx.size)
        if (
            n_rows > 0
            and getattr(table, "device_type_code", None) is not None
            and getattr(table, "meas_type_code", None) is not None
            and int(table.device_type_code.size) == n_rows
            and int(table.meas_type_code.size) == n_rows
        ):
            dt_arr = np.asarray(table.device_type_code, dtype=np.int32)
            mt_arr = np.asarray(table.meas_type_code, dtype=np.int32)
            pair_keys = (dt_arr << 16) | (mt_arr & 0xFFFF)
            order = np.argsort(pair_keys, kind="mergesort")
            sorted_keys = pair_keys[order]
            uniq_keys, split_pos = np.unique(sorted_keys, return_index=True)
            chunks = np.split(order, split_pos[1:])
            groups: Dict[Tuple[str, str], np.ndarray] = {}
            for uk, chunk in zip(uniq_keys.tolist(), chunks):
                dtc = int(uk >> 16)
                mtc = int(uk & 0xFFFF)
                groups[(DEVICE_TYPE_NAMES.get(dtc, ""), MEAS_TYPE_NAMES.get(mtc, ""))] = np.asarray(
                    chunk, dtype=np.int64
                )
            self.measurement_groups = groups
        elif n_rows == 0:
            self.measurement_groups = {}
        else:
            groups_py = defaultdict(list)
            for row, measurement in enumerate(self.active_measurements):
                groups_py[(str(measurement.device_type), str(measurement.meas_type))].append(row)
            self.measurement_groups = {
                key: np.asarray(rows, dtype=np.int64) for key, rows in groups_py.items()
            }
        self._compiled_meas_plan_key = None
        self._compiled_meas_plan = None

    def initial_state(self, flat_start: Optional[bool] = None) -> np.ndarray:
        if not self.prepared:
            self.prepare()
        net = self.network
        use_flat = self.params.flat_start if flat_start is None else bool(flat_start)
        if use_flat:
            n_isl = int(net.island_count)
            node_isl = np.asarray(net.node_island, dtype=np.int64)
            # O(N) 按岛统计锚点平均压力与节点平均压力，避免 O(islands * N) 重复全扫
            isl_sum = np.bincount(node_isl, weights=net.node_pressure, minlength=n_isl)
            isl_cnt = np.maximum(np.bincount(node_isl, minlength=n_isl), 1)
            isl_pressure = isl_sum / isl_cnt
            if net.fixed_node_pos.size:
                fix_isl = node_isl[net.fixed_node_pos]
                fix_sum = np.bincount(fix_isl, weights=net.fixed_pressure, minlength=n_isl)
                fix_cnt = np.bincount(fix_isl, minlength=n_isl)
                has_fix = fix_cnt > 0
                isl_pressure = np.where(has_fix, fix_sum / np.maximum(fix_cnt, 1), isl_pressure)
            pressure = np.maximum(isl_pressure[node_isl], 1e-6)
        else:
            pressure = np.maximum(net.node_pressure, 1e-6).copy()
        x = np.empty(self.state_count, dtype=np.float64)
        x[: self.n_potential] = pressure ** int(net.potential_power)
        if self.n_regulated:
            reg_edges = net.regulated_edge_pos
            reg_flow = np.asarray(net.edge_flow_set[reg_edges], dtype=np.float64).copy()
            # 显式调节支路（如 RATIO 模式 SteamPressureReducer，flow_set=0）若以精确 0 流量起步，
            # 会恰好骑在热损/焓值上游追踪 flow=0 阶跃不连续悬崖上；按端点压差方向给予小非零物理初值
            zero_mask = np.abs(reg_flow) < 1e-3
            if np.any(zero_mask):
                z_edges = reg_edges[zero_mask]
                dp = pressure[net.edge_i[z_edges]] - pressure[net.edge_j[z_edges]]
                reg_flow[zero_mask] = np.where(dp >= 0.0, 0.1, -0.1)
            x[self.base_regulated_flow : self.base_supply_temperature] = reg_flow
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

        # 全向量化表驱动编译快路径（0 Python 对象物化，50w 量测编译 < 20ms）
        active_tbl = getattr(active, "table", None)
        if (
            active_tbl is not None
            and int(active_tbl.idx.size) == len(active)
            and getattr(active_tbl, "device_type_code", None) is not None
            and getattr(active_tbl, "meas_type_code", None) is not None
            and getattr(active_tbl, "device_pos", None) is not None
            and int(active_tbl.device_type_code.size) == len(active)
            and int(active_tbl.meas_type_code.size) == len(active)
            and int(active_tbl.device_pos.size) == len(active)
        ):
            n_act = len(active)
            dt_arr = np.asarray(active_tbl.device_type_code, dtype=np.int16)
            mt_arr = np.asarray(active_tbl.meas_type_code, dtype=np.int16)
            pos_arr = np.asarray(active_tbl.device_pos, dtype=np.int64)
            handled = np.zeros(n_act, dtype=bool)

            node_dtc = int(DEVICE_TYPE_CODES.get(node_type, -1))
            edge_dtcs = np.asarray(
                [int(DEVICE_TYPE_CODES[t]) for t in edge_types if t in DEVICE_TYPE_CODES],
                dtype=np.int16,
            )
            src_dtcs = np.asarray(
                [int(DEVICE_TYPE_CODES[t]) for t in source_types if t in DEVICE_TYPE_CODES],
                dtype=np.int16,
            )
            load_dtc = int(DEVICE_TYPE_CODES.get(load_type, -1))

            mt_p = int(MEAS_TYPE_CODES.get("PRESSURE", -1))
            mt_pf = int(MEAS_TYPE_CODES.get("PRESSURE_FROM", -1))
            mt_pt = int(MEAS_TYPE_CODES.get("PRESSURE_TO", -1))
            mt_f = int(MEAS_TYPE_CODES.get("FLOW", -1))
            mt_ff = int(MEAS_TYPE_CODES.get("FLOW_FROM", -1))
            mt_ft = int(MEAS_TYPE_CODES.get("FLOW_TO", -1))
            mt_ts = int(MEAS_TYPE_CODES.get("T_SUPPLY", -1))
            mt_tr = int(MEAS_TYPE_CODES.get("T_RETURN", -1))
            mt_temp = int(MEAS_TYPE_CODES.get("TEMPERATURE", -1))
            mt_tsf = int(MEAS_TYPE_CODES.get("TS_FROM", -1))
            mt_tst = int(MEAS_TYPE_CODES.get("TS_TO", -1))
            mt_trf = int(MEAS_TYPE_CODES.get("TR_FROM", -1))
            mt_trt = int(MEAS_TYPE_CODES.get("TR_TO", -1))
            mt_tf = int(MEAS_TYPE_CODES.get("T_FROM", -1))
            mt_tt = int(MEAS_TYPE_CODES.get("T_TO", -1))
            mt_h = int(MEAS_TYPE_CODES.get("ENTHALPY", -1))
            mt_hf = int(MEAS_TYPE_CODES.get("H_FROM", -1))
            mt_ht = int(MEAS_TYPE_CODES.get("H_TO", -1))
            mt_heat = int(MEAS_TYPE_CODES.get("HEAT", -1))

            p_rows_chunks: List[np.ndarray] = []
            p_nodes_chunks: List[np.ndarray] = []
            lin_rows_chunks: List[np.ndarray] = []
            lin_cols_chunks: List[np.ndarray] = []
            lin_scale_chunks: List[np.ndarray] = []
            lin_offset_chunks: List[np.ndarray] = []
            adv_rows_chunks: List[np.ndarray] = []
            adv_pos_chunks: List[np.ndarray] = []
            adv_i_chunks: List[np.ndarray] = []
            adv_j_chunks: List[np.ndarray] = []
            adv_is_from_chunks: List[np.ndarray] = []
            adv_si_chunks: List[np.ndarray] = []
            adv_sj_chunks: List[np.ndarray] = []
            adv_loss_chunks: List[np.ndarray] = []
            adv_amb_chunks: List[np.ndarray] = []
            adv_os_chunks: List[np.ndarray] = []
            adv_oo_chunks: List[np.ndarray] = []
            adv_pi_chunks: List[np.ndarray] = []
            adv_rc_chunks: List[np.ndarray] = []
            const_rows_chunks: List[np.ndarray] = []
            const_vals_chunks: List[np.ndarray] = []

            is_node = dt_arr == node_dtc
            is_edge = np.isin(dt_arr, edge_dtcs) if edge_dtcs.size else np.zeros(n_act, dtype=bool)
            is_src = np.isin(dt_arr, src_dtcs) if src_dtcs.size else np.zeros(n_act, dtype=bool)
            is_load = dt_arr == load_dtc

            # 1. Node 量测
            if np.any(is_node):
                m = is_node & (mt_arr == mt_p)
                if np.any(m):
                    r = np.flatnonzero(m)
                    p_rows_chunks.append(r.astype(np.int32))
                    p_nodes_chunks.append(pos_arr[r].astype(np.int32))
                    handled[r] = True
                if net.thermal:
                    m = is_node & ((mt_arr == mt_ts) | (mt_arr == mt_temp))
                    if np.any(m):
                        r = np.flatnonzero(m)
                        lin_rows_chunks.append(r.astype(np.int32))
                        lin_cols_chunks.append((self.base_temperature + net.supply_temperature_state_by_node[pos_arr[r]]).astype(np.int32))
                        lin_scale_chunks.append(np.ones(r.size, dtype=np.float64))
                        lin_offset_chunks.append(np.zeros(r.size, dtype=np.float64))
                        handled[r] = True
                    m = is_node & (mt_arr == mt_tr)
                    if np.any(m):
                        r = np.flatnonzero(m)
                        lin_rows_chunks.append(r.astype(np.int32))
                        lin_cols_chunks.append((self.base_temperature + net.return_temperature_state_by_node[pos_arr[r]]).astype(np.int32))
                        lin_scale_chunks.append(np.ones(r.size, dtype=np.float64))
                        lin_offset_chunks.append(np.zeros(r.size, dtype=np.float64))
                        handled[r] = True
                elif net.steam:
                    m = is_node & (mt_arr == mt_h)
                    if np.any(m):
                        r = np.flatnonzero(m)
                        lin_rows_chunks.append(r.astype(np.int32))
                        lin_cols_chunks.append((self.base_enthalpy + pos_arr[r]).astype(np.int32))
                        lin_scale_chunks.append(np.ones(r.size, dtype=np.float64))
                        lin_offset_chunks.append(np.zeros(r.size, dtype=np.float64))
                        handled[r] = True
                    m = is_node & (mt_arr == mt_temp)
                    if np.any(m):
                        r = np.flatnonzero(m)
                        lin_rows_chunks.append(r.astype(np.int32))
                        lin_cols_chunks.append((self.base_enthalpy + pos_arr[r]).astype(np.int32))
                        lin_scale_chunks.append(np.full(r.size, steam_t_scale, dtype=np.float64))
                        lin_offset_chunks.append(np.full(r.size, steam_t_offset, dtype=np.float64))
                        handled[r] = True

            # 2. Edge 量测
            ef_rows = np.empty(0, dtype=np.int64)
            ef_pos = np.empty(0, dtype=np.int64)
            ef_sign = np.empty(0, dtype=np.float64)
            ef_p_rows = np.empty(0, dtype=np.int32)
            ef_p_idx = np.empty(0, dtype=np.int64)
            ef_p_i = np.empty(0, dtype=np.int32)
            ef_p_j = np.empty(0, dtype=np.int32)
            ef_p_sign = np.empty(0, dtype=np.float64)
            ef_r_rows = np.empty(0, dtype=np.int32)
            ef_r_col = np.empty(0, dtype=np.int32)
            ef_r_sign = np.empty(0, dtype=np.float64)
            if np.any(is_edge):
                m_ef = is_edge & ((mt_arr == mt_f) | (mt_arr == mt_ff) | (mt_arr == mt_ft))
                if np.any(m_ef):
                    r = np.flatnonzero(m_ef)
                    ep = pos_arr[r]
                    sgn = np.where(mt_arr[r] == mt_ft, -1.0, 1.0)
                    ef_rows = r.astype(np.int64)
                    ef_pos = ep
                    ef_sign = sgn
                    pl = passive_lookup[ep]
                    pm = pl >= 0
                    if np.any(pm):
                        ef_p_rows = r[pm].astype(np.int32)
                        ef_p_idx = pl[pm]
                        ef_p_i = net.edge_i[ep[pm]].astype(np.int32)
                        ef_p_j = net.edge_j[ep[pm]].astype(np.int32)
                        ef_p_sign = sgn[pm]
                    rc = self.regulated_state_by_edge[ep]
                    rm = (~pm) & (rc >= 0)
                    if np.any(rm):
                        ef_r_rows = r[rm].astype(np.int32)
                        ef_r_col = rc[rm].astype(np.int32)
                        ef_r_sign = sgn[rm]
                    handled[r] = True
                m_epf = is_edge & (mt_arr == mt_pf)
                if np.any(m_epf):
                    r = np.flatnonzero(m_epf)
                    p_rows_chunks.append(r.astype(np.int32))
                    p_nodes_chunks.append(net.edge_i[pos_arr[r]].astype(np.int32))
                    handled[r] = True
                m_ept = is_edge & (mt_arr == mt_pt)
                if np.any(m_ept):
                    r = np.flatnonzero(m_ept)
                    p_rows_chunks.append(r.astype(np.int32))
                    p_nodes_chunks.append(net.edge_j[pos_arr[r]].astype(np.int32))
                    handled[r] = True
                if net.thermal:
                    m_imp_t = is_edge & ((mt_arr == mt_tsf) | (mt_arr == mt_tst) | (mt_arr == mt_trf) | (mt_arr == mt_trt))
                    if np.any(m_imp_t):
                        r_cand = np.flatnonzero(m_imp_t)
                        ei_c = net.edge_i[pos_arr[r_cand]]
                        keep = ~net.node_explicit_return[ei_c]
                        if np.any(keep):
                            r = r_cand[keep]
                            ep = pos_arr[r]
                            mt_r = mt_arr[r]
                            nd = np.where((mt_r == mt_tsf) | (mt_r == mt_trf), net.edge_i[ep], net.edge_j[ep])
                            st_col = np.where(
                                (mt_r == mt_tsf) | (mt_r == mt_tst),
                                net.supply_temperature_state_by_node[nd],
                                net.return_temperature_state_by_node[nd],
                            )
                            lin_rows_chunks.append(r.astype(np.int32))
                            lin_cols_chunks.append((self.base_temperature + st_col).astype(np.int32))
                            lin_scale_chunks.append(np.ones(r.size, dtype=np.float64))
                            lin_offset_chunks.append(np.zeros(r.size, dtype=np.float64))
                            handled[r] = True
                    m_exp_t = is_edge & ((mt_arr == mt_tf) | (mt_arr == mt_tt))
                    if np.any(m_exp_t):
                        r_cand = np.flatnonzero(m_exp_t)
                        ei_c = net.edge_i[pos_arr[r_cand]]
                        keep = net.node_explicit_return[ei_c]
                        if np.any(keep):
                            r = r_cand[keep]
                            ep = pos_arr[r]
                            ei = net.edge_i[ep]
                            ej = net.edge_j[ep]
                            adv_rows_chunks.append(r.astype(np.int32))
                            adv_pos_chunks.append(ep)
                            adv_i_chunks.append(ei.astype(np.int32))
                            adv_j_chunks.append(ej.astype(np.int32))
                            adv_is_from_chunks.append(mt_arr[r] == mt_tf)
                            adv_si_chunks.append((self.base_temperature + net.supply_temperature_state_by_node[ei]).astype(np.int32))
                            adv_sj_chunks.append((self.base_temperature + net.supply_temperature_state_by_node[ej]).astype(np.int32))
                            adv_loss_chunks.append(net.edge_heat_loss[ep].astype(np.float64))
                            adv_amb_chunks.append(np.full(r.size, amb_temp, dtype=np.float64))
                            adv_os_chunks.append(np.ones(r.size, dtype=np.float64))
                            adv_oo_chunks.append(np.zeros(r.size, dtype=np.float64))
                            adv_pi_chunks.append(passive_lookup[ep])
                            adv_rc_chunks.append(self.regulated_state_by_edge[ep].astype(np.int32))
                            handled[r] = True
                elif net.steam:
                    m_s_adv = is_edge & ((mt_arr == mt_hf) | (mt_arr == mt_ht) | (mt_arr == mt_tf) | (mt_arr == mt_tt))
                    if np.any(m_s_adv):
                        r = np.flatnonzero(m_s_adv)
                        ep = pos_arr[r]
                        ei = net.edge_i[ep]
                        ej = net.edge_j[ep]
                        mt_r = mt_arr[r]
                        is_t = (mt_r == mt_tf) | (mt_r == mt_tt)
                        adv_rows_chunks.append(r.astype(np.int32))
                        adv_pos_chunks.append(ep)
                        adv_i_chunks.append(ei.astype(np.int32))
                        adv_j_chunks.append(ej.astype(np.int32))
                        adv_is_from_chunks.append((mt_r == mt_hf) | (mt_r == mt_tf))
                        adv_si_chunks.append((self.base_enthalpy + ei).astype(np.int32))
                        adv_sj_chunks.append((self.base_enthalpy + ej).astype(np.int32))
                        adv_loss_chunks.append(net.edge_heat_loss[ep].astype(np.float64))
                        adv_amb_chunks.append(np.full(r.size, amb_enth, dtype=np.float64))
                        adv_os_chunks.append(np.where(is_t, steam_t_scale, 1.0))
                        adv_oo_chunks.append(np.where(is_t, steam_t_offset, 0.0))
                        adv_pi_chunks.append(passive_lookup[ep])
                        adv_rc_chunks.append(self.regulated_state_by_edge[ep].astype(np.int32))
                        handled[r] = True

            # 3. Source / Storage 量测
            if np.any(is_src):
                m = is_src & ((mt_arr == mt_p) | (mt_arr == mt_pt))
                if np.any(m):
                    r = np.flatnonzero(m)
                    p_rows_chunks.append(r.astype(np.int32))
                    p_nodes_chunks.append(net.source_supply_node_pos[pos_arr[r]].astype(np.int32))
                    handled[r] = True
                m = is_src & (mt_arr == mt_pf)
                if np.any(m):
                    r = np.flatnonzero(m)
                    p_rows_chunks.append(r.astype(np.int32))
                    p_nodes_chunks.append(net.source_return_node_pos[pos_arr[r]].astype(np.int32))
                    handled[r] = True
                if net.thermal:
                    m = is_src & (mt_arr == mt_ts)
                    if np.any(m):
                        r = np.flatnonzero(m)
                        sn = net.source_supply_node_pos[pos_arr[r]]
                        lin_rows_chunks.append(r.astype(np.int32))
                        lin_cols_chunks.append((self.base_temperature + net.supply_temperature_state_by_node[sn]).astype(np.int32))
                        lin_scale_chunks.append(np.ones(r.size, dtype=np.float64))
                        lin_offset_chunks.append(np.zeros(r.size, dtype=np.float64))
                        handled[r] = True
                    m = is_src & (mt_arr == mt_tr)
                    if np.any(m):
                        r = np.flatnonzero(m)
                        rn = net.source_return_node_pos[pos_arr[r]]
                        lin_rows_chunks.append(r.astype(np.int32))
                        lin_cols_chunks.append((self.base_temperature + net.return_temperature_state_by_node[rn]).astype(np.int32))
                        lin_scale_chunks.append(np.ones(r.size, dtype=np.float64))
                        lin_offset_chunks.append(np.zeros(r.size, dtype=np.float64))
                        handled[r] = True
                elif net.steam:
                    m = is_src & (mt_arr == mt_h)
                    if np.any(m):
                        r = np.flatnonzero(m)
                        sn = net.source_supply_node_pos[pos_arr[r]]
                        lin_rows_chunks.append(r.astype(np.int32))
                        lin_cols_chunks.append((self.base_enthalpy + sn).astype(np.int32))
                        lin_scale_chunks.append(np.ones(r.size, dtype=np.float64))
                        lin_offset_chunks.append(np.zeros(r.size, dtype=np.float64))
                        handled[r] = True
                    m = is_src & (mt_arr == mt_temp)
                    if np.any(m):
                        r = np.flatnonzero(m)
                        sn = net.source_supply_node_pos[pos_arr[r]]
                        lin_rows_chunks.append(r.astype(np.int32))
                        lin_cols_chunks.append((self.base_enthalpy + sn).astype(np.int32))
                        lin_scale_chunks.append(np.full(r.size, steam_t_scale, dtype=np.float64))
                        lin_offset_chunks.append(np.full(r.size, steam_t_offset, dtype=np.float64))
                        handled[r] = True

            # 4. Load 量测
            if np.any(is_load):
                m = is_load & (mt_arr == mt_f)
                if np.any(m):
                    r = np.flatnonzero(m)
                    const_rows_chunks.append(r.astype(np.int32))
                    const_vals_chunks.append(net.load_flow_set[pos_arr[r]].astype(np.float64))
                    handled[r] = True
                m = is_load & ((mt_arr == mt_p) | (mt_arr == mt_pf))
                if np.any(m):
                    r = np.flatnonzero(m)
                    p_rows_chunks.append(r.astype(np.int32))
                    p_nodes_chunks.append(net.load_supply_node_pos[pos_arr[r]].astype(np.int32))
                    handled[r] = True
                m = is_load & (mt_arr == mt_pt)
                if np.any(m):
                    r = np.flatnonzero(m)
                    p_rows_chunks.append(r.astype(np.int32))
                    p_nodes_chunks.append(net.load_return_node_pos[pos_arr[r]].astype(np.int32))
                    handled[r] = True
                if net.thermal:
                    m = is_load & (mt_arr == mt_ts)
                    if np.any(m):
                        r = np.flatnonzero(m)
                        sn = net.load_supply_node_pos[pos_arr[r]]
                        lin_rows_chunks.append(r.astype(np.int32))
                        lin_cols_chunks.append((self.base_temperature + net.supply_temperature_state_by_node[sn]).astype(np.int32))
                        lin_scale_chunks.append(np.ones(r.size, dtype=np.float64))
                        lin_offset_chunks.append(np.zeros(r.size, dtype=np.float64))
                        handled[r] = True
                    m = is_load & (mt_arr == mt_tr)
                    if np.any(m):
                        r = np.flatnonzero(m)
                        rn = net.load_return_node_pos[pos_arr[r]]
                        lin_rows_chunks.append(r.astype(np.int32))
                        lin_cols_chunks.append((self.base_temperature + net.return_temperature_state_by_node[rn]).astype(np.int32))
                        lin_scale_chunks.append(np.ones(r.size, dtype=np.float64))
                        lin_offset_chunks.append(np.zeros(r.size, dtype=np.float64))
                        handled[r] = True
                    m = is_load & (mt_arr == mt_heat)
                    if np.any(m):
                        r_cand = np.flatnonzero(m)
                        keep = ~net.load_explicit_return[pos_arr[r_cand]]
                        if np.any(keep):
                            r = r_cand[keep]
                            const_rows_chunks.append(r.astype(np.int32))
                            const_vals_chunks.append(net.load_heat_power[pos_arr[r]].astype(np.float64))
                            handled[r] = True
                elif net.steam:
                    m = is_load & (mt_arr == mt_h)
                    if np.any(m):
                        r = np.flatnonzero(m)
                        sn = net.load_supply_node_pos[pos_arr[r]]
                        lin_rows_chunks.append(r.astype(np.int32))
                        lin_cols_chunks.append((self.base_enthalpy + sn).astype(np.int32))
                        lin_scale_chunks.append(np.ones(r.size, dtype=np.float64))
                        lin_offset_chunks.append(np.zeros(r.size, dtype=np.float64))
                        handled[r] = True
                    m = is_load & (mt_arr == mt_temp)
                    if np.any(m):
                        r = np.flatnonzero(m)
                        sn = net.load_supply_node_pos[pos_arr[r]]
                        lin_rows_chunks.append(r.astype(np.int32))
                        lin_cols_chunks.append((self.base_enthalpy + sn).astype(np.int32))
                        lin_scale_chunks.append(np.full(r.size, steam_t_scale, dtype=np.float64))
                        lin_offset_chunks.append(np.full(r.size, steam_t_offset, dtype=np.float64))
                        handled[r] = True
                    m = is_load & (mt_arr == mt_heat)
                    if np.any(m):
                        r = np.flatnonzero(m)
                        lp = pos_arr[r]
                        sn = net.load_supply_node_pos[lp]
                        lf_v = net.load_flow_set[lp].astype(np.float64)
                        lin_rows_chunks.append(r.astype(np.int32))
                        lin_cols_chunks.append((self.base_enthalpy + sn).astype(np.int32))
                        lin_scale_chunks.append(lf_v)
                        lin_offset_chunks.append(-lf_v * net.load_condensate_enthalpy[lp].astype(np.float64))
                        handled[r] = True

            rem_rows = np.flatnonzero(~handled)
            if rem_rows.size:
                fallback_items = [(int(r), active[int(r)]) for r in rem_rows.tolist()]
                has_source_meas = bool(np.any(is_src[rem_rows]))
                fallback_needs_flow_deriv = bool(np.any(is_edge[rem_rows]) or has_source_meas)

            def _cat(chunks, dtype):
                if not chunks:
                    return np.empty(0, dtype=dtype)
                return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

            plan = {
                "p_rows": _cat(p_rows_chunks, np.int32),
                "p_nodes": _cat(p_nodes_chunks, np.int32),
                "edge_flow_rows": ef_rows,
                "edge_flow_pos": ef_pos,
                "edge_flow_sign": ef_sign,
                "edge_flow_passive_rows": ef_p_rows,
                "edge_flow_passive_idx": ef_p_idx,
                "edge_flow_passive_i": ef_p_i,
                "edge_flow_passive_j": ef_p_j,
                "edge_flow_passive_sign": ef_p_sign,
                "edge_flow_reg_rows": ef_r_rows,
                "edge_flow_reg_col": ef_r_col,
                "edge_flow_reg_sign": ef_r_sign,
                "lin_rows": _cat(lin_rows_chunks, np.int32),
                "lin_cols": _cat(lin_cols_chunks, np.int32),
                "lin_scale": _cat(lin_scale_chunks, np.float64),
                "lin_offset": _cat(lin_offset_chunks, np.float64),
                "adv_rows": _cat(adv_rows_chunks, np.int32),
                "adv_pos": _cat(adv_pos_chunks, np.int64),
                "adv_i": _cat(adv_i_chunks, np.int32),
                "adv_j": _cat(adv_j_chunks, np.int32),
                "adv_is_from": _cat(adv_is_from_chunks, bool),
                "adv_state_i": _cat(adv_si_chunks, np.int32),
                "adv_state_j": _cat(adv_sj_chunks, np.int32),
                "adv_loss": _cat(adv_loss_chunks, np.float64),
                "adv_amb": _cat(adv_amb_chunks, np.float64),
                "adv_out_scale": _cat(adv_os_chunks, np.float64),
                "adv_out_offset": _cat(adv_oo_chunks, np.float64),
                "adv_passive_idx": _cat(adv_pi_chunks, np.int64),
                "adv_reg_col": _cat(adv_rc_chunks, np.int32),
                "const_rows": _cat(const_rows_chunks, np.int32),
                "const_vals": _cat(const_vals_chunks, np.float64),
                "fallback_items": fallback_items,
                "has_source_meas": has_source_meas,
                "fallback_needs_flow_deriv": fallback_needs_flow_deriv,
            }
            if use_cache:
                self._compiled_meas_plan_key = (id(active), len(active), id(self.network))
                self._compiled_meas_plan = plan
            return plan

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
            # 热损质量流量下界 1e-3：避免零流量叶管处 exp(-L/|q|) 导数爆炸击穿信息矩阵条件数
            mass = np.maximum(np.abs(f_val), 1e-3)
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
                                / max(abs(float(flow[pos])), 1e-3)
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
                        np.exp(-net.edge_heat_loss[pos] / max(abs(float(flow[pos])), 1e-3))
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
            abs_f = np.abs(f_val)
            mass = np.maximum(abs_f, 1e-3)
            loss = plan["adv_loss"]
            factor = np.exp(-loss / mass)
            out_scale = plan["adv_out_scale"]
            row_parts.append(adv_rows)
            col_parts.append(up_col)
            data_parts.append(out_scale * np.where(is_up_end, 1.0, factor))

            # 下游端且存在热损时向量化写入对流量的偏导（|f|<=1e-3 平台段导数严格为 0，与前向公式一致）
            dn_mask = (~is_up_end) & (loss > 0.0) & (abs_f > 1e-3)
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
                            abs_flow = abs(flow_value)
                            mass = max(abs_flow, 1e-3)
                            loss = float(net.edge_heat_loss[pos])
                            factor = float(np.exp(-loss / mass))
                            rows.append(row)
                            cols.append(upstream_col)
                            data.append(factor)
                            if loss > 0.0 and abs_flow > 1e-3:
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
                        abs_flow = abs(flow_value)
                        mass = max(abs_flow, 1e-3)
                        loss = float(net.edge_heat_loss[pos])
                        factor = float(np.exp(-loss / mass))
                        rows.append(row)
                        cols.append(self.base_enthalpy + upstream_node)
                        data.append(factor * output_scale)
                        if loss > 0.0 and abs_flow > 1e-3:
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
            act_tbl = getattr(self.active_measurements, "table", None)
            all_tbl = getattr(self.measurements, "table", None)
            if (
                act_tbl is not None
                and int(act_tbl.idx.size) == len(self.active_measurements)
                and getattr(act_tbl, "device_pos", None) is not None
                and getattr(act_tbl, "meas_type_code", None) is not None
                and int(act_tbl.device_pos.size) == int(act_tbl.idx.size)
            ):
                # 零对象表级去重：用 (dt_code, device_pos, mt_code) 整型三元组替代物化全表对象
                existing_codes = set(
                    zip(
                        act_tbl.device_type_code.tolist(),
                        act_tbl.device_pos.tolist(),
                        act_tbl.meas_type_code.tolist(),
                    )
                )
                existing: set = set()
            else:
                existing_codes = None
                existing = {
                    (str(item.device_type), str(item.device_name), str(item.meas_type))
                    for item in self.active_measurements
                }
            if all_tbl is not None and int(all_tbl.idx.size) > 0:
                next_idx = int(np.max(all_tbl.idx)) + 1
                if len(self.measurements) > int(all_tbl.idx.size):
                    for tail_pos in range(int(all_tbl.idx.size), len(self.measurements)):
                        next_idx = max(next_idx, int(self.measurements[tail_pos].idx) + 1)
            else:
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
                if existing_codes is not None:
                    ck = (int(cand.device_type_code), int(cand.device_pos), int(cand.meas_type_code))
                    if ck in existing_codes:
                        continue
                    existing_codes.add(ck)
                else:
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

    def state_base_scales(self) -> np.ndarray:
        """构造流体状态向量的物理特征基准向量 D_x（用于求解空间内生无量纲标幺化，外部输入输出保持有名值不变）。"""
        cached = getattr(self, "_cached_state_base_scales", None)
        net = self.network
        cache_key = (id(net), int(self.state_count))
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        scales = np.ones(int(self.state_count), dtype=np.float64)
        if self.state_count <= 0:
            self._cached_state_base_scales = (cache_key, scales)
            return scales

        # 1. 节点压力势能基准：p_base^potential_power（使势能标幺值归一化至 O(1)）
        if self.n_potential > 0:
            p_vals = np.asarray(net.node_pressure, dtype=np.float64)
            p_pos = p_vals[np.isfinite(p_vals) & (p_vals > 1e-6)]
            p_base = float(np.median(p_pos)) if p_pos.size else 1.0
            pot_base = max(p_base ** int(net.potential_power), 1e-6)
            scales[: self.n_potential] = pot_base

        # 2. 受控管段流量基准：取管段设定/源/负荷特征流量中位数（下限 1.0 防零流除零）
        if self.n_regulated > 0:
            f_samples = []
            for arr in (
                getattr(net, "edge_flow_set", None),
                getattr(net, "source_flow_set", None),
                getattr(net, "demand", None),
            ):
                if arr is not None:
                    a = np.abs(np.asarray(arr, dtype=np.float64))
                    a = a[np.isfinite(a) & (a > 1e-6)]
                    if a.size:
                        f_samples.append(a)
            f_base = float(np.median(np.concatenate(f_samples))) if f_samples else 1.0
            f_base = max(f_base, 1.0)
            scales[self.base_regulated_flow : self.base_supply_temperature] = f_base

        # 3. 热网供/回水温度基准：T_base
        if net.thermal and self.base_enthalpy > self.base_temperature:
            t_arr = np.abs(np.asarray(net.initial_temperature_state, dtype=np.float64))
            t_pos = t_arr[np.isfinite(t_arr) & (t_arr > 1.0)]
            t_base = max(float(np.median(t_pos)) if t_pos.size else 50.0, 1.0)
            scales[self.base_temperature : self.base_enthalpy] = t_base

        # 4. 蒸汽网比焓基准：h_base（典型值 ~2800-3000 kJ/kg，标幺后 O(1)）
        if net.steam and self.state_count > self.base_enthalpy:
            h_arr = np.abs(np.asarray(net.node_enthalpy, dtype=np.float64))
            h_pos = h_arr[np.isfinite(h_arr) & (h_arr > 1.0)]
            h_base = max(float(np.median(h_pos)) if h_pos.size else 2500.0, 1.0)
            scales[self.base_enthalpy :] = h_base

        self._cached_state_base_scales = (cache_key, scales)
        return scales

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
        high_damp_stagnation_streak = 0
        high_damp_slow_streak = 0
        low_progress_streak = 0
        fullstep_osc_streak = 0
        curvature_damping_active = False
        singular_fail_count = 0
        singular_min_damping = 0.0
        aa_accel = AndersonAccelerator(m=4)
        prev_max_correction = np.inf
        tried_undamped_escape = False
        pre_escape_max_correction = np.inf
        last_used_damping = 0.0
        self._last_aat_solver = None
        self._last_weight_used = weight
        state_scales = self.state_base_scales()

        def _solve_step(H_cur: csr_matrix, res_cur: np.ndarray, damping_val: float):
            """在无量纲标幺状态空间 x_pu = x / D_x 下组装并求解正规方程，再还原物理增量 dx = D_x * dx_pu。"""
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
                # 标幺化：A_pu = D_x * A_csc, rhs_pu = D_x * rhs_vec（对无阻尼步与阻尼步均消除跨物理量尺度差）
                A_pu = A_csc.copy()
                A_pu.data = A_csc.data * state_scales[A_csc.indices]
                rhs_pu = rhs_vec * state_scales
                diag_pu = np.bincount(
                    A_pu.indices,
                    weights=A_pu.data * A_pu.data,
                    minlength=self.state_count,
                )
                diag_g = diag_pu / np.maximum(state_scales * state_scales, 1e-24)
                pinned_pot = (
                    np.flatnonzero((x[: self.n_potential] <= minimum + 1e-9) & (rhs_vec[: self.n_potential] < 0.0))
                    if self.n_potential > 0
                    else np.empty(0, dtype=np.intp)
                )
                if damping_val > 0.0:
                    # 在标幺正规方程上叠加 Jacobi 列均衡 Marquardt 相对阻尼
                    col_scale = 1.0 / np.sqrt(np.maximum(diag_pu, 1e-12))
                    A_scaled = A_pu.copy()
                    A_scaled.data = A_pu.data * col_scale[A_pu.indices]
                    rhs_scaled = rhs_pu * col_scale
                    if pinned_pot.size > 0:
                        dx_scaled, ok_pin = solve_box_pinned_normal_equations(
                            A_scaled, None, rhs_scaled, pinned_pot, damping=damping_val
                        )
                        if not ok_pin:
                            raise RuntimeError("Pinned fluid AAt solve failed")
                    else:
                        dx_scaled, _ = aat_solver.solve(
                            A_scaled,
                            rhs_scaled,
                            return_factor_diag=False,
                            damping=damping_val,
                        )
                    dx_pu = np.asarray(dx_scaled, dtype=np.float64) * col_scale
                else:
                    if pinned_pot.size > 0:
                        dx_raw, ok_pin = solve_box_pinned_normal_equations(
                            A_pu, None, rhs_pu, pinned_pot, damping=0.0
                        )
                        if not ok_pin:
                            raise RuntimeError("Pinned fluid AAt solve failed")
                    else:
                        dx_raw, _ = aat_solver.solve(
                            A_pu,
                            rhs_pu,
                            return_factor_diag=False,
                            damping=0.0,
                        )
                    dx_pu = np.asarray(dx_raw, dtype=np.float64)
                if pinned_pot.size > 0:
                    dx_pu[pinned_pot] = 0.0
                dx_out = dx_pu * state_scales
                return dx_out, rhs_vec, diag_g, pinned_pot
            # 中小规模路径同样在标幺状态空间 H_pu = H * D_x 组装良态正规方程
            H_pu = H_cur.copy()
            if H_pu.nnz > 0:
                H_pu.data = H_cur.data * state_scales[H_cur.indices]
            weighted_H_pu = diags(weight) @ H_pu
            gain_pu = (H_pu.T @ weighted_H_pu).tocsc()
            rhs_pu = np.asarray(H_pu.T @ (weight * res_cur), dtype=np.float64)
            rhs_vec = rhs_pu / np.maximum(state_scales, 1e-12)
            diag_pu = np.asarray(gain_pu.diagonal(), dtype=np.float64)
            diag_g = diag_pu / np.maximum(state_scales * state_scales, 1e-24)
            pinned_pot = (
                np.flatnonzero((x[: self.n_potential] <= minimum + 1e-9) & (rhs_vec[: self.n_potential] < 0.0))
                if self.n_potential > 0
                else np.empty(0, dtype=np.intp)
            )
            zero_diag_mask = diag_pu <= 1e-12
            if damping_val > 0.0 or bool(np.any(zero_diag_mask)):
                col_scale = np.where(zero_diag_mask, 1.0, 1.0 / np.sqrt(np.maximum(diag_pu, 1e-12)))
                D_scale = diags(col_scale, format="csc")
                scaled_g = (D_scale @ gain_pu @ D_scale).tocsc()
                eff_damp = max(damping_val, 1e-6)
                diag_add = np.where(zero_diag_mask, 1.0 + eff_damp, eff_damp)
                gain_reg = (scaled_g + diags(diag_add, format="csc")).tocsc()
                rhs_scaled = np.where(zero_diag_mask, 0.0, rhs_pu * col_scale)
                if pinned_pot.size > 0:
                    dx_scaled, ok_pin = solve_box_pinned_normal_equations(
                        None, gain_reg, rhs_scaled, pinned_pot, damping=0.0
                    )
                    if not ok_pin:
                        raise RuntimeError("Pinned fluid normal solve failed")
                else:
                    dx_scaled, _ = normal_solver.solve(gain_reg, rhs_scaled, return_factor_diag=False)
                dx_pu = np.where(zero_diag_mask, 0.0, np.asarray(dx_scaled, dtype=np.float64) * col_scale)
            else:
                if pinned_pot.size > 0:
                    dx_pu, ok_pin = solve_box_pinned_normal_equations(
                        None, gain_pu, rhs_pu, pinned_pot, damping=0.0
                    )
                    if not ok_pin:
                        raise RuntimeError("Pinned fluid normal solve failed")
                else:
                    dx_pu, _ = normal_solver.solve(gain_pu, rhs_pu, return_factor_diag=False)
            if pinned_pot.size > 0:
                dx_pu[pinned_pot] = 0.0
            dx_raw = np.asarray(dx_pu, dtype=np.float64) * state_scales
            gain = None
            return dx_raw, rhs_vec, diag_g, pinned_pot

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
            pinned_pot = np.empty(0, dtype=np.intp)
            step_damping = lm_damping
            try:
                dx, rhs, diag_g, pinned_pot = _solve_step(H, residual, step_damping)
            except (MatrixRankWarning, RuntimeError, ValueError):
                dx = None
            if dx is None or (dx.size and not np.all(np.isfinite(dx))):
                if step_damping == 0.0:
                    singular_fail_count += 1
                    if singular_fail_count >= 2:
                        # 多次无阻尼分解奇异说明当前结构秩亏，锁定微幅安全底线防止反复归零撞奇异
                        singular_min_damping = max(singular_min_damping, 1e-6)
                # 数值病态或近零流量奇异时分级启用 Marquardt 相对阻尼扶正方向
                for beta_try in (1e-4, 1e-2, 1e-1, 1.0, 10.0):
                    if beta_try <= step_damping:
                        continue
                    try:
                        dx_try, rhs_try, diag_try, pin_try = _solve_step(H, residual, beta_try)
                        if dx_try.size == self.state_count and np.all(np.isfinite(dx_try)):
                            dx, rhs, diag_g, pinned_pot = dx_try, rhs_try, diag_try, pin_try
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
            if rhs is not None and diag_g is not None and rhs.size > 0:
                rhs_kkt = rhs.copy() if pinned_pot.size > 0 else rhs
                if pinned_pot.size > 0:
                    rhs_kkt[pinned_pot] = 0.0
                kkt_inf = float(np.max(np.abs(rhs_kkt) / np.maximum(diag_g, 1e-12)))
            else:
                kkt_inf = np.inf
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
            x_before_step = x
            # J = sum(w * r^2) 的梯度为 -2 * H^T W r = -2 * rhs
            ls_slope = -2.0 * float(np.dot(rhs, dx)) if (rhs is not None and rhs.size == dx.size) else 0.0
            obj_tol = max(1e-14, 1e-11 * (abs(objective) + 1.0))
            # 可压缩流体（potential_power > 1.0，如蒸汽/天然气 u = P^2）采用内点法防穿零保正下界：
            # 单步压势最多缩减至当前的 1/4，避免高压平启动首步将下游多压级节点拍入 1e-12 真空奇点（dP/du -> ∞）
            pot_floor = (
                np.maximum(minimum, 0.25 * x[: self.n_potential])
                if (self.n_potential > 0 and float(self.network.potential_power) > 1.0)
                else minimum
            )
            for _ in range(20):
                candidate = x + step * dx
                candidate[: self.n_potential] = np.maximum(candidate[: self.n_potential], pot_floor)
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
            if not accepted and self.n_regulated > 0:
                # 若热/汽显式调节支路恰好跨越 flow=0 上游追踪阶跃悬崖导致全部线搜索失败，
                # 试探一次同侧锁止步（防止微步跨零触发焓值/温度阶跃暴涨）
                reg_s = self.base_regulated_flow
                reg_e = self.base_supply_temperature
                x_reg = x[reg_s:reg_e]
                dx_reg = dx[reg_s:reg_e]
                cross_mask = ((x_reg >= 0.0) & (x_reg + dx_reg < 0.0)) | ((x_reg < 0.0) & (x_reg + dx_reg >= 0.0))
                if np.any(cross_mask):
                    clamp_step = 1.0
                    for _ in range(8):
                        cand_c = x + clamp_step * dx
                        cand_c[: self.n_potential] = np.maximum(cand_c[: self.n_potential], pot_floor)
                        reg_c = cand_c[reg_s:reg_e]
                        pos_cross = (x_reg >= 0.0) & (reg_c < 1e-4)
                        neg_cross = (x_reg < 0.0) & (reg_c > -1e-4)
                        if np.any(pos_cross):
                            reg_c[pos_cross] = 1e-4
                        if np.any(neg_cross):
                            reg_c[neg_cross] = -1e-4
                        cand_c_z = self.evaluate(cand_c)
                        cand_c_r = z - cand_c_z
                        cand_c_obj = float(np.dot(weight, cand_c_r * cand_c_r))
                        if np.isfinite(cand_c_obj) and cand_c_obj <= objective + obj_tol:
                            x = cand_c
                            cached_z_est = cand_c_z
                            cached_residual = cand_c_r
                            cached_objective = cand_c_obj
                            accepted_step = clamp_step
                            accepted = True
                            break
                        clamp_step *= 0.5
            if not accepted:
                if tried_undamped_escape:
                    # 连续停滞后尝试无阻尼脱困仍无法下降，确认已到达非光滑边界局部极小驻点并还原脱困前有效步长
                    if np.isfinite(pre_escape_max_correction):
                        max_correction = min(max_correction, pre_escape_max_correction)
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
            # 支柱 2：连续满步慢线性收敛时尝试 Type-II Anderson 加速（严格目标单调保护）
            if accepted_step >= 0.95 and max_correction > 10.0 * self.tol:
                aa_ratio = (
                    (max_correction / prev_max_correction)
                    if (np.isfinite(prev_max_correction) and prev_max_correction > 0.0)
                    else 0.0
                )
                x_aa = aa_accel.update_and_propose(
                    x_before_step,
                    x,
                    step_ratio=aa_ratio,
                    damping=step_damping,
                    allow_propose=(rel_correction < 0.5),
                )
                if x_aa is not None and (
                    self.n_potential == 0 or bool(np.all(x_aa[: self.n_potential] >= minimum))
                ):
                    aa_z = self.evaluate(x_aa)
                    aa_r = z - aa_z
                    aa_obj = float(np.dot(weight, aa_r * aa_r))
                    if np.isfinite(aa_obj) and aa_obj < cached_objective - max(1e-12, 1e-10 * (abs(cached_objective) + 1.0)):
                        x = x_aa
                        cached_z_est = aa_z
                        cached_residual = aa_r
                        cached_objective = aa_obj
                        obj_change = abs(prev_objective - cached_objective)
                        aa_accel.on_accepted()
                    else:
                        aa_accel.on_rejected()
                elif x_aa is not None:
                    aa_accel.on_rejected()
            else:
                aa_accel.reset()

            # 支柱 1：基于精确 LM 增益比 ρ 的 Nielsen (1999) 连续光滑阻尼更新律（全步长统一决策）
            if accepted_step >= 0.5:
                backtrack_streak = max(0, backtrack_streak - 1)
                if (
                    max_correction > max(self.tol, 1e-5)
                    and (
                        max_correction >= 0.95 * prev_max_correction
                        or obj_change <= max(1e-12, 1e-9 * (abs(prev_objective) + 1.0))
                    )
                    and obj_change <= 1e-4 * (abs(prev_objective) + 1.0)
                ):
                    fullstep_osc_streak += 1
                    if fullstep_osc_streak >= 4:
                        lm_damping = min(100.0, max(lm_damping * 5.0, 0.1))
                        curvature_damping_active = True
                        fullstep_osc_streak = 0
                else:
                    fullstep_osc_streak = 0
            else:
                if (
                    max_correction > max(self.tol, 1e-5)
                    and obj_change <= max(1e-12, 1e-9 * (abs(prev_objective) + 1.0))
                ):
                    fullstep_osc_streak += 1
                    if fullstep_osc_streak >= 4:
                        lm_damping = min(100.0, max(lm_damping * 5.0, 0.1))
                        curvature_damping_active = True
                        fullstep_osc_streak = 0
                else:
                    fullstep_osc_streak = 0
                backtrack_streak += 1
                if lm_damping <= 0.0 and (backtrack_streak >= 2 or accepted_step < 1e-2):
                    lm_damping = min(100.0, max(1e-3, singular_min_damping))
                    backtrack_streak = 0
            if lm_damping > 0.0:
                lm_damping, _rho_lm = nielsen_lm_damping_update(
                    lm_damping,
                    rhs,
                    dx,
                    prev_objective - cached_objective,
                    accepted_step,
                    half_obj=False,
                    diag_reg=diag_g,
                    min_damping=singular_min_damping,
                    max_damping=100.0,
                    zero_cutoff=1e-4 if curvature_damping_active else 1e-6,
                    dx_growing=(max_correction > 1.2 * prev_max_correction),
                )
                if lm_damping <= 0.0:
                    curvature_damping_active = False
                    backtrack_streak = 0
            prev_max_correction = max_correction

            # 带阻尼且已达稳态停滞（KKT 梯度与目标函数 7 位双满足）提前优雅收敛
            if (
                step_damping > 0.0
                and rel_correction < 1e-3
                and kkt_inf <= max(1e-4, 100.0 * self.tol)
                and obj_change <= max(1e-12, 1e-7 * (abs(prev_objective) + 1.0))
            ):
                converged = True
                break

            # 非光滑边界（如零流量管段上游追踪热损/焓值翻转台阶）局部极小驻点与高阻尼锁死脱困检测：
            # 1) 若上一步刚做完无阻尼脱困试探但目标函数未见实质下降，确认已到达非光滑边界极小驻点并优雅退出；
            # 2) 若目标函数 9 位有效数字不变且实际相对位移极小，或在高阻尼 (>=1.0) 下连续多步相对降幅极微 (<1e-5)，
            #    先做 1 次无阻尼脱困试探，试探后仍停滞则优雅收敛，释放外层坏数据剔除流程。
            rel_obj_tol_high_damp = max(1e-12, 1e-5 * (abs(prev_objective) + 1.0))
            rel_obj_tol_slow = max(1e-12, 5e-4 * (abs(prev_objective) + 1.0))
            rel_obj_tol_flat = max(1e-12, 1e-6 * (abs(prev_objective) + 1.0))
            if obj_change <= rel_obj_tol_flat:
                low_progress_streak += 1
            else:
                low_progress_streak = 0
            if tried_undamped_escape:
                if obj_change <= max(1e-10, 5e-4 * (abs(prev_objective) + 1.0)):
                    if np.isfinite(pre_escape_max_correction):
                        max_correction = min(max_correction, pre_escape_max_correction)
                    converged = True
                    break
                tried_undamped_escape = False
                pre_escape_max_correction = np.inf
                stagnation_streak = 0
                high_damp_stagnation_streak = 0
                high_damp_slow_streak = 0
                low_progress_streak = 0
            elif (
                obj_change <= max(1e-12, 1e-9 * (abs(prev_objective) + 1.0))
                and actual_rel_dx <= max(1e-6, self.tol)
            ):
                stagnation_streak += 1
                if stagnation_streak >= 3:
                    if step_damping > 0.0:
                        lm_damping = 0.0
                        curvature_damping_active = False
                        tried_undamped_escape = True
                        pre_escape_max_correction = max_correction
                        stagnation_streak = 0
                        high_damp_stagnation_streak = 0
                        high_damp_slow_streak = 0
                    else:
                        converged = True
                        break
            elif step_damping >= 1.0 and obj_change <= rel_obj_tol_high_damp:
                stagnation_streak = 0
                high_damp_stagnation_streak += 1
                high_damp_slow_streak += 1
                if high_damp_stagnation_streak >= 6:
                    lm_damping = 0.0
                    curvature_damping_active = False
                    tried_undamped_escape = True
                    pre_escape_max_correction = max_correction
                    high_damp_stagnation_streak = 0
                    high_damp_slow_streak = 0
            elif step_damping >= 1.0 and obj_change <= rel_obj_tol_slow:
                stagnation_streak = 0
                high_damp_stagnation_streak = 0
                high_damp_slow_streak += 1
                if high_damp_slow_streak >= 12:
                    lm_damping = 0.0
                    curvature_damping_active = False
                    tried_undamped_escape = True
                    pre_escape_max_correction = max_correction
                    high_damp_slow_streak = 0
            elif low_progress_streak >= 12:
                # 中低阻尼非光滑脊线微幅抖动（连续 12 步目标 6 位无实质进展）：触发无阻尼脱困确认，释放外层 BDR 剔坏
                low_progress_streak = 0
                stagnation_streak = 0
                high_damp_stagnation_streak = 0
                high_damp_slow_streak = 0
                if step_damping > 0.0:
                    lm_damping = 0.0
                    curvature_damping_active = False
                    tried_undamped_escape = True
                    pre_escape_max_correction = max_correction
                else:
                    converged = True
                    break
            else:
                stagnation_streak = 0
                high_damp_stagnation_streak = 0
                high_damp_slow_streak = 0

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
            measurements=self.active_measurements,
            observability=obs,
            measurement_table=active_table,
        )
        result._final_lm_damping = float(lm_damping)
        result._state_base_scales = state_scales
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

        # 伪量测（MEAS_STATUS_PSEUDO）允许参与坏数据辨识与排除；唯一补盲的临界量测由相对冗余度保护自动拦护
        is_protected = np.zeros(n_meas, dtype=bool)

        # 使用精确杠杆 h_ii = diag(H G^-1 H^T) = diag(H_pu G_pu^-1 H_pu^T) 在标幺良态空间计算标准化残差
        leverage = None
        if result.H is not None and result.gain is not None and result.H.shape[0] == n_meas and self.state_count > 0:
            cached_solver = getattr(self, "_last_aat_solver", None)
            cached_fac = cached_solver.last_cholmod_factor if cached_solver is not None else None
            cached_op = cached_solver.last_normal_operand if cached_solver is not None else None
            state_scales = getattr(result, "_state_base_scales", None)
            if state_scales is None or np.asarray(state_scales).size != int(result.H.shape[1]):
                state_scales = self.state_base_scales()
            state_scales = np.asarray(state_scales, dtype=np.float64)
            H_csr = result.H if getattr(result.H, "format", None) == "csr" else result.H.tocsr()
            H_pu = H_csr.copy()
            if H_pu.nnz > 0:
                H_pu.data = H_csr.data * state_scales[H_csr.indices]
            D_s = diags(state_scales, format="csc")
            gain_pu = (D_s @ result.gain.tocsc() @ D_s).tocsr()
            try:
                leverage = measurement_leverage_for_bad_data(
                    H_pu,
                    gain_pu,
                    cholmod_factor=cached_fac,
                    aat_operand=cached_op,
                    dense_state_limit=256,
                    sparse_threshold=256,
                    sparse_state_limit=500_000,
                )
            except Exception:
                leverage = None

        if leverage is not None and leverage.size == n_meas:
            normalized, critical_mask = compute_normalized_residuals_for_bad_data(
                np.asarray(result.residual, dtype=np.float64),
                weights,
                leverage,
                is_pseudo=is_protected,
            )
            # 掩蔽权重为 0（已剔除）的行不再上报
            normalized = np.where(weights > 0.0, normalized, 0.0)
        else:
            critical_mask = np.zeros(n_meas, dtype=bool)
            normalized = np.where(
                weights > 0.0,
                np.sqrt(np.maximum(weights, 0.0)) * np.abs(result.residual),
                0.0,
            )

        n_eff = int(np.count_nonzero((weights > 0.0) & (~is_protected) & (~critical_mask)))
        eff_threshold, per_row_thr = compute_adaptive_bad_threshold(
            threshold_value,
            n_eff if n_eff > 0 else int(np.count_nonzero(weights > 0.0)),
            normalized=normalized,
            weights=weights,
            active_mask=(weights > 0.0),
            protected_mask=is_protected,
            critical_mask=critical_mask,
        )
        hit_pos = (
            np.flatnonzero(normalized > per_row_thr)
            if per_row_thr is not None
            else np.flatnonzero(normalized > eff_threshold)
        )
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
            else:
                base_weight = np.asarray([item.weight for item in self.active_measurements], dtype=np.float64)
            is_protected = np.zeros(n_meas, dtype=bool)
            n_real_valid = int(np.count_nonzero((base_weight > 0.0) & (~is_protected)))
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
            blocked_rows_set = set()

            res = self.estimate(observability=obs, _weight_override=cur_weight)
            if not res.converged and not self.params.flat_start:
                flat_init_res = self.estimate(
                    x0=self.initial_state(flat_start=True),
                    observability=obs,
                    _weight_override=cur_weight,
                    _hint_lm_damping=0.0,
                )
                if flat_init_res.converged or (
                    np.isfinite(flat_init_res.objective)
                    and (not np.isfinite(res.objective) or flat_init_res.objective < res.objective)
                ):
                    res = flat_init_res
            if total_cap <= 0 or (not res.converged and not (obs.observable and np.isfinite(res.objective))):
                self.identify_bad_data(res, threshold=threshold_value, weight_override=cur_weight)
                return res, removed_items

            unconverged_streak = 0 if res.converged else 1
            unconverged_total_rounds = 0 if res.converged else 1
            # 缓存当前状态下的坏数据识别与列覆盖度，跳过临界行或回滚重试时免去重复矩阵求逆
            need_identify = True
            bad_list: List[BadDataItem] = []
            norm_res = np.empty(0, dtype=np.float64)
            col_degrees: Optional[np.ndarray] = None
            row_to_item: Dict[int, BadDataItem] = {}
            while len(removed_items) < total_cap:
                if need_identify:
                    bad_list, norm_res = self.identify_bad_data(
                        res,
                        threshold=threshold_value,
                        weight_override=cur_weight,
                    )
                    col_degrees = compute_active_column_degrees(res.H, active_mask, cur_weight)
                    row_to_item = {int(it.row_pos): it for it in bad_list}
                    need_identify = False
                if not bad_list:
                    break
                cand_rows = [
                    int(it.row_pos)
                    for it in bad_list
                    if int(it.row_pos) not in removed_rows_set
                    and int(it.row_pos) not in blocked_rows_set
                    and not bool(is_protected[int(it.row_pos)])
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
                safe_local, _skipped = filter_batch_preserving_column_coverage(res.H, selected, col_degrees)
                safe_rows = [selected[k] for k in safe_local]
                if not safe_rows:
                    probe_cands = cand_rows[: min(len(cand_rows), 20)]
                    probe_local, _ = filter_batch_preserving_column_coverage(res.H, probe_cands, col_degrees)
                    if probe_local:
                        safe_rows = [probe_cands[probe_local[0]]]
                if not safe_rows:
                    break

                # 结构满秩前置拦截：若删批后导致系统结构降秩（临界对/孤立支路），折半收缩或跳过临界行
                trial_rows = list(safe_rows)
                while trial_rows:
                    t_mask = active_mask.copy()
                    t_w = cur_weight.copy()
                    t_mask[trial_rows] = False
                    t_w[trial_rows] = 0.0
                    if check_active_structural_full_rank(res.H, t_mask, t_w):
                        break
                    if len(trial_rows) == 1:
                        blocked_rows_set.add(trial_rows[0])
                        trial_rows = []
                        break
                    trial_rows = trial_rows[: max(1, len(trial_rows) // 2)]
                if not trial_rows:
                    if len(blocked_rows_set) >= 32:
                        break
                    continue

                # 带回滚与折半退避的重估：防止误删隐式临界测点导致压力坍塌、单点压力爆炸或发散
                pw = max(float(self.network.potential_power), 1e-12)
                if self.n_potential > 0:
                    prev_pot = np.maximum(res.x[: self.n_potential], 1e-12)
                    prev_p = prev_pot ** (1.0 / pw)
                    prev_min_p = float(np.min(prev_p))
                    prev_max_p = float(np.max(prev_p))
                    prev_med_p = float(np.median(prev_p))
                else:
                    prev_min_p = 1.0
                    prev_max_p = 1.0
                    prev_med_p = 1.0
                accepted_trial = False
                while trial_rows:
                    t_weight = cur_weight.copy()
                    t_weight[trial_rows] = 0.0
                    hint_damp = float(getattr(res, "_final_lm_damping", 0.0))
                    trial_res = self.estimate(
                        x0=res.x,
                        observability=obs,
                        _weight_override=t_weight,
                        _hint_lm_damping=hint_damp,
                    )
                    # 热启动未能收敛时自动以平启动重试一次，避免被高阻尼坏初值锁死
                    if not trial_res.converged:
                        flat_res = self.estimate(
                            x0=self.initial_state(flat_start=True),
                            observability=obs,
                            _weight_override=t_weight,
                            _hint_lm_damping=0.0,
                        )
                        if flat_res.converged or (
                            np.isfinite(flat_res.objective)
                            and (not np.isfinite(trial_res.objective) or flat_res.objective < trial_res.objective)
                        ):
                            trial_res = flat_res
                    if self.n_potential > 0:
                        trial_pot = np.maximum(trial_res.x[: self.n_potential], 1e-12)
                        trial_min_p = float(np.min(trial_pot)) ** (1.0 / pw)
                        trial_max_p = float(np.max(trial_pot)) ** (1.0 / pw)
                    else:
                        trial_min_p = 1.0
                        trial_max_p = 1.0
                    pressure_abnormal = (
                        (prev_min_p > 1e-3 and trial_min_p < 0.25 * prev_min_p)
                        or (prev_max_p > 1e-3 and trial_max_p > 3.0 * prev_max_p)
                        or (
                            prev_med_p > 1e-3
                            and trial_max_p > 3.0 * prev_med_p
                            and trial_max_p > 1.25 * prev_max_p
                        )
                    )
                    # 允许首轮未收敛后通过删坏恢复收敛；已收敛后若删坏导致发散、压力坍塌/爆炸或目标严重恶化则回滚折半
                    trial_ok = (
                        np.isfinite(trial_res.objective)
                        and not pressure_abnormal
                        and (
                            trial_res.converged
                            or (not res.converged and trial_res.objective < res.objective)
                        )
                        and (
                            not res.converged
                            or trial_res.objective <= res.objective * 1.05 + 1e-6
                        )
                    )
                    if trial_ok:
                        prev_obj_accepted = float(res.objective)
                        cur_weight = t_weight
                        active_mask[trial_rows] = False
                        act_tbl = getattr(self.active_measurements, "table", None)
                        tbl_status = measurement_table_status_code(act_tbl) if act_tbl is not None else None
                        for r_idx in trial_rows:
                            removed_rows_set.add(r_idx)
                            item = row_to_item[r_idx]
                            item.measurement.valid = False
                            item.measurement.status = MEAS_STATUS_REMOVED
                            if act_tbl is not None and r_idx < int(act_tbl.idx.size):
                                act_tbl.valid[r_idx] = False
                                if tbl_status is not None:
                                    tbl_status[r_idx] = MEAS_STATUS_REMOVED
                            removed_items.append(item)
                        res = trial_res
                        if res.converged:
                            unconverged_streak = 0
                            unconverged_total_rounds = 0
                        else:
                            unconverged_total_rounds += 1
                            if np.isfinite(prev_obj_accepted) and res.objective <= 0.85 * prev_obj_accepted:
                                unconverged_streak = 0
                            else:
                                unconverged_streak += 1
                        need_identify = True
                        accepted_trial = True
                        break
                    if len(trial_rows) == 1:
                        blocked_rows_set.add(trial_rows[0])
                        break
                    trial_rows = trial_rows[: max(1, len(trial_rows) // 2)]

                if not accepted_trial:
                    if not res.converged or len(blocked_rows_set) >= 16:
                        break
                    continue
                if unconverged_streak >= 3 or unconverged_total_rounds >= 10:
                    break

            self.bad_data = removed_items
            if need_identify:
                self.identify_bad_data(res, threshold=threshold_value, weight_override=cur_weight)
            self.bad_data = removed_items
            self.result = res
            self._write_back_state(res.x)
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
