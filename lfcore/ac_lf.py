import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags, hstack, vstack
from scipy.sparse.linalg import spsolve

try:
    from scipy.sparse.linalg import use_solver as _scipy_use_solver
    _scipy_use_solver(useUmfpack=False)
except Exception:
    pass

from collections import deque
from typing import List, Tuple, Dict, Optional
import warnings
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
for path in (ROOT_DIR, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from _sparse_pattern import (
        apply_raw_sum_plan,
        build_compressed_pattern_from_raw_coords,
        build_raw_sum_plan,
    )
except ImportError:  # pragma: no cover - package import path
    from ._sparse_pattern import (
        apply_raw_sum_plan,
        build_compressed_pattern_from_raw_coords,
        build_raw_sum_plan,
    )
from algorithm_parameters import DEFAULT_LF_PARAMETER_FILE, PowerFlowParameters, load_lf_parameters
from paths import model_file
from ac_array_model import (
    ACAC_COLS,
    ACAC_SIDE_CONTROL_CODE,
    ACAC_SIDE_CONTROL_LABEL,
    BREAK_COLS,
    BRANCH_COLS,
    BUS_COLS,
    CTRL_P,
    CTRL_PQ,
    CTRL_PV,
    CTRL_SLACK,
    GEN_COLS,
    LOAD_COLS,
    SHUNT_B,
    SHUNT_COLS,
    SHUNT_Q,
    SHUNT_V,
    SHUNT_Z,
    SWITCH_COLS,
    THREE_WINDING_TRANSFORMER_COLS,
    TRANSFORMER_COLS,
    ZERO_BRANCH_COLS,
    acac_combined_control_code,
    build_ac_ppc_from_e_file,
    build_ac_ppc_from_mat_file,
    build_ac_ppc_from_network,
    sanitize_ac_voltage_setpoints,
)
from ac_model import ACAC_SIDE_CONTROL_TYPES
from model.ppc_topology import build_ac_ppc_with_topology_from_e_file, ensure_ac_ppc_topology
from model.switching_status import (
    apply_lf_closed_status_boundaries,
    finalize_lf_closed_status_results,
)
from itertools import starmap
from model.topology import build_lf_control_bus_groups
try:
    from lfcore.common import (
        ACNodeResult,
        ACSinglePortResult,
        ACSwitchResult,
        ACThreePortResult,
        ACTwoPortResult,
        LazyResultDict,
        allocate_grouped_residual,
        allocate_limited_residual,
        build_group_share_metadata as _build_group_share_metadata,
        build_result_voltage_lookup,
        build_zero_branch_components_fast,
        find_spanning_tree_edges,
        normalize_result_mode as _normalize_lf_result_mode,
        optional_ppc_vector,
        result_name_list,
    )
except ImportError:  # pragma: no cover - direct script import path
    from common import (
        ACNodeResult,
        ACSinglePortResult,
        ACSwitchResult,
        ACThreePortResult,
        ACTwoPortResult,
        LazyResultDict,
        allocate_grouped_residual,
        allocate_limited_residual,
        build_group_share_metadata as _build_group_share_metadata,
        build_result_voltage_lookup,
        build_zero_branch_components_fast,
        find_spanning_tree_edges,
        normalize_result_mode as _normalize_lf_result_mode,
        optional_ppc_vector,
        result_name_list,
    )
try:
    from lfcore.solver_common import (
        AdaptiveNewtonFactor as _AdaptiveNewtonFactor,
        OPTIONAL_SPARSE_MISSING as _OPTIONAL_SPARSE_MISSING,
        OPTIONAL_SPARSE_SOLVERS as _OPTIONAL_SPARSE_SOLVERS,
        factor_jacobian as _factor_jacobian,
        make_reusable_factorizer as _make_reusable_factorizer,
        resolve_linear_solver as _resolve_linear_solver,
    )
except ImportError:  # pragma: no cover - direct script import path
    from solver_common import (
        AdaptiveNewtonFactor as _AdaptiveNewtonFactor,
        OPTIONAL_SPARSE_MISSING as _OPTIONAL_SPARSE_MISSING,
        OPTIONAL_SPARSE_SOLVERS as _OPTIONAL_SPARSE_SOLVERS,
        factor_jacobian as _factor_jacobian,
        make_reusable_factorizer as _make_reusable_factorizer,
        resolve_linear_solver as _resolve_linear_solver,
    )


AC_NODE_TYPE_PQ = np.int8(1)
AC_NODE_TYPE_PV = np.int8(2)
AC_NODE_TYPE_SLACK = np.int8(3)
AC_NODE_TYPE_LABELS = {
    int(AC_NODE_TYPE_PQ): "PQ",
    int(AC_NODE_TYPE_PV): "PV",
    int(AC_NODE_TYPE_SLACK): "SLACK",
}


def ac_node_type_label(node_type) -> str:
    return AC_NODE_TYPE_LABELS.get(int(node_type), str(int(node_type)))


@dataclass
class ACLFResult:
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    acac_converters: Dict[str, SimpleNamespace] = field(default_factory=dict)
    branches: Dict[str, SimpleNamespace] = field(default_factory=dict)
    transformers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    three_winding_transformers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    nodes: Dict[str, SimpleNamespace] = field(default_factory=dict)
    zero_branches: Dict[str, SimpleNamespace] = field(default_factory=dict)
    breakers: Dict[str, SimpleNamespace] = field(default_factory=dict)
    generators: Dict[str, SimpleNamespace] = field(default_factory=dict)
    loads: Dict[str, SimpleNamespace] = field(default_factory=dict)


def load_ac_ppc_from_e_file(file_name) -> Dict:
    """Read an AC E/MATPOWER file into PPC with topology arrays attached."""
    path = Path(file_name)
    suffix = path.suffix.lower()
    if suffix in {".m", ".mat"}:
        ppc = build_ac_ppc_from_mat_file(path)
        ppc["source"] = str(path.resolve())
        return ensure_ac_ppc_topology(ppc)
    return build_ac_ppc_with_topology_from_e_file(path)


def _build_csr_pattern_from_raw_coords(raw_rows, raw_cols, n_rows: int):
    """Build a CSR sparsity pattern and raw-entry-to-CSR-position map.

    ``raw_rows``/``raw_cols`` may contain duplicate coordinates.  The returned
    ``raw_to_csr`` array maps each raw coordinate to the corresponding entry in
    the unique CSR pattern so runtime values can be accumulated with
    a precomputed copy/reduce plan.
    """
    return build_compressed_pattern_from_raw_coords(raw_rows, raw_cols, n_rows)


class _PPCNode:
    __slots__ = ("idx", "name", "vbase", "voltage", "angle")

    def __init__(self, idx, name, vbase, voltage, angle):
        self.idx = idx
        self.name = name
        self.vbase = vbase
        self.voltage = voltage
        self.angle = angle


def safe_division(a, b, default=0.0):
    """安全除法，避免除零错误"""
    try:
        return a / b if abs(b) > 1e-12 else default
    except Exception:
        return default


def matpower_branch_stamp(r: float, x: float, b: float = 0.0, tap: float = 1.0, shift: float = 0.0):
    """Return MATPOWER-compatible branch admittance entries Yff, Yft, Ytf, Ytt."""
    y = safe_division(1.0, complex(r, x))
    tap_mag = tap if abs(tap) > 1e-12 else 1.0
    tap_complex = tap_mag * np.exp(1j * np.deg2rad(shift))
    y_sh = 1j * b / 2.0
    y_series_shunt = y + y_sh
    return (
        y_series_shunt / (tap_complex * np.conj(tap_complex)),
        -y / np.conj(tap_complex),
        -y / tap_complex,
        y_series_shunt,
    )


def matpower_branch_stamp_vectorized(r, x, b=0.0, tap=1.0, shift=0.0):
    """Vectorized MATPOWER-compatible branch admittance stamp for branch batches.

    ``b`` remains a branch parameter, not a transformer parameter. It is the
    total shunt susceptance and is applied as ``j*b/2`` at each end.
    """
    r = np.asarray(r, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    b = np.broadcast_to(np.asarray(b, dtype=np.float64), r.shape)
    tap = np.broadcast_to(np.asarray(tap, dtype=np.float64), r.shape)
    shift = np.broadcast_to(np.asarray(shift, dtype=np.float64), r.shape)

    z = r + 1j * x
    y = np.divide(
        1.0,
        z,
        out=np.zeros(r.shape, dtype=np.complex128),
        where=np.abs(z) > 1e-12,
    )
    tap_mag = np.where(np.abs(tap) > 1e-12, tap, 1.0)
    tap_complex = tap_mag * np.exp(1j * np.deg2rad(shift))
    y_sh = 1j * b / 2.0
    y_series_shunt = y + y_sh
    return (
        y_series_shunt / (tap_complex * np.conj(tap_complex)),
        -y / np.conj(tap_complex),
        -y / tap_complex,
        y_series_shunt,
    )


def matpower_transformer_stamp_vectorized(r, x, gt=0.0, bt=0.0, tap=1.0, shift=0.0):
    """Vectorized T-type transformer stamp with one grounding branch on the i side.

    ``gt + j*bt`` is a single shunt admittance on the i side before the ideal
    tap/phase-shift transformer. It is not MATPOWER's symmetric ``BR_B`` line
    charging and must not be split across both terminals.
    """
    r = np.asarray(r, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    gt = np.broadcast_to(np.asarray(gt, dtype=np.float64), r.shape)
    bt = np.broadcast_to(np.asarray(bt, dtype=np.float64), r.shape)
    tap = np.broadcast_to(np.asarray(tap, dtype=np.float64), r.shape)
    shift = np.broadcast_to(np.asarray(shift, dtype=np.float64), r.shape)

    z = r + 1j * x
    y = np.divide(
        1.0,
        z,
        out=np.zeros(r.shape, dtype=np.complex128),
        where=np.abs(z) > 1e-12,
    )
    tap_mag = np.where(np.abs(tap) > 1e-12, tap, 1.0)
    tap_complex = tap_mag * np.exp(1j * np.deg2rad(shift))
    # Transformer shunt is single-ended: add it only to the i-side
    # self-admittance, then refer it through the complex tap.
    y_ground = gt + 1j * bt
    return (
        (y + y_ground) / (tap_complex * np.conj(tap_complex)),
        -y / np.conj(tap_complex),
        -y / tap_complex,
        y,
    )


def three_winding_transformer_stamp_vectorized(
    i_r,
    i_x,
    j_r,
    j_x,
    k_r,
    k_x,
    gt=0.0,
    bt=0.0,
    i_tap=1.0,
    i_shift=0.0,
    j_tap=1.0,
    j_shift=0.0,
    k_tap=1.0,
    k_shift=0.0,
):
    """Return the Kron-reduced 3x3 terminal admittance stamp.

    The three leakage impedances connect each terminal to an internal star
    point. The star voltage is analytically eliminated, so no synthetic bus is
    exposed to load flow or state estimation. The result shape is ``(n, 3, 3)``.
    """
    inputs = np.broadcast_arrays(
        np.asarray(i_r, dtype=np.float64),
        np.asarray(i_x, dtype=np.float64),
        np.asarray(j_r, dtype=np.float64),
        np.asarray(j_x, dtype=np.float64),
        np.asarray(k_r, dtype=np.float64),
        np.asarray(k_x, dtype=np.float64),
        np.asarray(gt, dtype=np.float64),
        np.asarray(bt, dtype=np.float64),
        np.asarray(i_tap, dtype=np.float64),
        np.asarray(i_shift, dtype=np.float64),
        np.asarray(j_tap, dtype=np.float64),
        np.asarray(j_shift, dtype=np.float64),
        np.asarray(k_tap, dtype=np.float64),
        np.asarray(k_shift, dtype=np.float64),
    )
    (
        i_r,
        i_x,
        j_r,
        j_x,
        k_r,
        k_x,
        gt,
        bt,
        i_tap,
        i_shift,
        j_tap,
        j_shift,
        k_tap,
        k_shift,
    ) = inputs
    shape = i_r.shape
    z = np.stack((i_r + 1j * i_x, j_r + 1j * j_x, k_r + 1j * k_x), axis=-1)
    tap = np.stack((i_tap, j_tap, k_tap), axis=-1)
    tap = np.where(np.abs(tap) > 1e-12, tap, 1.0)
    shift = np.stack((i_shift, j_shift, k_shift), axis=-1)
    tapc = tap * np.exp(1j * np.deg2rad(shift))
    z_flat = z.reshape(-1, 3)
    tapc_flat = tapc.reshape(-1, 3)
    zero_arm = np.abs(z_flat) <= 1e-12
    zero_count = np.count_nonzero(zero_arm, axis=1)
    invalid = np.flatnonzero(zero_count > 1)
    if invalid.size:
        raise ValueError(
            "ACThreeWindingTransformer cannot contain more than one zero-impedance winding; "
            f"invalid flattened row(s): {invalid.tolist()}"
        )

    y_flat = np.divide(
        1.0,
        z_flat,
        out=np.zeros(z_flat.shape, dtype=np.complex128),
        where=~zero_arm,
    )
    stamp_flat = np.zeros((z_flat.shape[0], 3, 3), dtype=np.complex128)

    regular = zero_count == 0
    if np.any(regular):
        y_regular = y_flat[regular]
        tap_regular = tapc_flat[regular]
        y_sum = np.sum(y_regular, axis=1)
        inv_y_sum = np.divide(
            1.0,
            y_sum,
            out=np.zeros(y_sum.shape, dtype=np.complex128),
            where=np.abs(y_sum) > 1e-12,
        )
        from_star = y_regular / np.conj(tap_regular)
        to_star = y_regular / tap_regular
        regular_stamp = -from_star[:, :, None] * to_star[:, None, :] * inv_y_sum[:, None, None]
        diag = np.arange(3)
        regular_stamp[:, diag, diag] += y_regular / (tap_regular * np.conj(tap_regular))
        stamp_flat[regular] = regular_stamp

    single_zero = zero_count == 1
    for anchor in range(3):
        rows = np.flatnonzero(single_zero & zero_arm[:, anchor])
        if rows.size == 0:
            continue
        anchor_tap = tapc_flat[rows, anchor]
        for terminal in range(3):
            if terminal == anchor:
                continue
            terminal_y = y_flat[rows, terminal]
            terminal_tap = tapc_flat[rows, terminal]
            stamp_flat[rows, anchor, anchor] += terminal_y / (anchor_tap * np.conj(anchor_tap))
            stamp_flat[rows, terminal, terminal] += terminal_y / (terminal_tap * np.conj(terminal_tap))
            stamp_flat[rows, anchor, terminal] -= terminal_y / (np.conj(anchor_tap) * terminal_tap)
            stamp_flat[rows, terminal, anchor] -= terminal_y / (np.conj(terminal_tap) * anchor_tap)

    gt_flat = np.asarray(gt, dtype=np.float64).reshape(-1)
    bt_flat = np.asarray(bt, dtype=np.float64).reshape(-1)
    stamp_flat[:, 0, 0] += (gt_flat + 1j * bt_flat) / (
        tapc_flat[:, 0] * np.conj(tapc_flat[:, 0])
    )
    return stamp_flat.reshape(shape + (3, 3))


# ==============================================================================
# 核心潮流计算类（精简极速版）
# ==============================================================================
class ACPowerFlowCalc:
    """交流潮流计算类（极坐标牛顿-拉夫逊法）。

    求解变量按 ``theta_unknown``、``V_unknown``、零阻抗支路辅助电位
    ``phi_re/phi_im`` 排列。节点 P/Q 平衡、PV/Slack 电压约束、零阻抗
    支路电压相等约束和 phi 参考约束共同组成 Newton 方程组。

    这里的 LF 还承担母线控制语义：同一 AC 母线上多个定压设备时，
    允许按平均 ``v_set`` 建立等效电压目标，并按既定能力系数分摊
    有功/无功失衡。SE 侧暂不强制复用这套硬聚合规则。
    """

    def __init__(
        self,
        network,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        min_voltage: Optional[float] = None,
        island=None,
        parameter_file=DEFAULT_LF_PARAMETER_FILE,
        parameters: Optional[PowerFlowParameters] = None,
        keep_node_objects: bool = True,
        linear_solver: str = "pyklu",
        result_mode: str = "full",
        verbose: bool = False,
    ):
        self.params = (parameters or load_lf_parameters(parameter_file)).with_overrides(
            tol=tol,
            max_iter=max_iter,
            min_voltage=min_voltage,
        )
        self._network_writeback = None
        if isinstance(network, dict) and network.get("format") == "ac_ppc_v1":
            self.ppc = network
        elif island is None and hasattr(network, "nodes"):
            self._network_writeback = network
            existing_ppc = getattr(network, "ppc", None)
            if isinstance(existing_ppc, dict) and existing_ppc.get("format") == "ac_ppc_v1":
                self.ppc = existing_ppc
            else:
                self.ppc = build_ac_ppc_from_network(network)
            if hasattr(network, "source") and "source" not in self.ppc:
                self.ppc["source"] = str(getattr(network, "source"))
        else:
            raise ValueError("ACPowerFlowCalc requires ac_ppc_v1 or ACPowerNetwork input")
        self.result_mode = self._normalize_result_mode(result_mode)
        self.keep_node_objects = False
        # NR 主循环仅使用 CSC 格式；CSR 模式改为在 get_jacobi(x) 首次需要时懒构建
        self._cache_csr_jacobian_pattern = False
        self.tol = self.params.tol
        self.max_iter = self.params.max_iter
        self.min_voltage = self.params.min_voltage
        # 用户请求的求解器名原样保留，便于上层日志/测试断言；实际 callable
        # 由 _resolve_linear_solver 决定，未安装时回退 SuperLU。
        self.linear_solver = str(linear_solver or "pyklu").strip().lower()
        self._linear_solver_resolved, self._linear_solver_fn = _resolve_linear_solver(self.linear_solver)
        # 实例级可选求解器黑名单: 失败时只在本实例回退 scipy, 不污染模块缓存。
        self._instance_solver_blacklist: set = set()
        self.verbose = bool(verbose)
        self.target_island = island
        self.skipped_islands: List = []
        self.calc_islands: List = []

        # 状态变量
        self.converged: bool = False
        self.iterations: int = 0
        self.normF: float = np.inf
        self.failure_reason: str = ""
        self.lf_result = None

        # 核心变量初始化（精简合并）
        self.N: int = 0
        self.node_list: List = []
        self.node_pos: Dict[int, int] = {}
        self.ppc_node_idx = np.array([], dtype=np.int64)
        self.ppc_node_name = np.array([], dtype=object)
        self.ppc_node_vbase = np.array([], dtype=np.float64)
        self.ppc_node_voltage = np.array([], dtype=np.float64)
        self.ppc_node_angle = np.array([], dtype=np.float64)
        self._ppc_node_row_lookup = np.array([], dtype=np.int32)
        self._active_node_solver_lookup = np.array([], dtype=np.int32)
        self.active_node_rows = np.array([], dtype=np.int32)
        self.active_node_solver_pos = np.array([], dtype=np.int32)
        self._topology_bus_to_solver_pos = np.array([], dtype=np.int32)
        self._lf_control_bus_groups = None
        self.isl = None
        self.Y: Optional[csr_matrix] = None

        # 节点参数（使用更简洁的初始化方式）
        self.node_type: np.ndarray = np.array([])
        self.V_spec: np.ndarray = np.array([])
        self.theta_spec: np.ndarray = np.array([])
        self.P_spec: np.ndarray = np.array([])
        self.Q_spec: np.ndarray = np.array([])
        self._slack_mask = np.array([], dtype=bool)
        self._fixed_voltage_mask = np.array([], dtype=bool)
        self.load_info: List = []
        self.slack_node: int = -1

        # 零阻抗支路相关。
        # 零阻抗支路不进入常规 Y 矩阵；这里把每个零阻抗连通分量映射为
        # 一组 phi 辅助变量，支路电流由两端 phi 差表示。
        self.zero_edges = []
        self.comp_nodes: List[List[int]] = []
        self.comp_tree_edges: List[List[int]] = []
        self.phi_node: List[int] = []
        self.phi_comp: List[int] = []
        self.ref_phi_idx: List[int] = []
        self.zero_branch_info: np.ndarray = np.array([])
        self.N_phi: int = 0

        # 变量/方程索引。
        # theta_idx/V_idx 指向 Newton 状态向量中的列号；后续构造残差和
        # 雅可比时只使用这些预计算索引，避免在迭代中重复查表。
        self.theta_unknown: np.ndarray = np.array([])
        self.V_unknown: np.ndarray = np.array([])
        self.theta_idx = np.array([], dtype=np.int32)
        self.V_idx = np.array([], dtype=np.int32)
        self.base_phi_re: int = 0
        self.base_phi_im: int = 0
        self.n_theta: int = 0
        self.n_V: int = 0
        self.total_vars: int = 0
        self.total_eq: int = 0

        # 缓存变量（精简键名）
        self._cache = dict(theta=None, V=None, Vc=None, cos_theta=None, sin_theta=None, P_load=None, Q_load=None)

        # 迭代变量
        self.x: np.ndarray = np.array([])
        self.load_pos = np.array([], dtype=np.int32)
        self.load_pv0 = np.array([], dtype=np.float64)
        self.load_pv1 = np.array([], dtype=np.float64)
        self.load_pv2 = np.array([], dtype=np.float64)
        self.load_qv0 = np.array([], dtype=np.float64)
        self.load_qv1 = np.array([], dtype=np.float64)
        self.load_qv2 = np.array([], dtype=np.float64)
        self.zero_idx = np.array([], dtype=np.int32)
        self.zero_type = np.array([], dtype=np.int32)
        self.zero_a = np.array([], dtype=np.int32)
        self.zero_b = np.array([], dtype=np.int32)
        self.zero_phi_a = np.array([], dtype=np.int32)
        self.zero_phi_b = np.array([], dtype=np.int32)
        self.zero_top_left_rows = np.array([], dtype=np.int32)
        self.zero_top_left_cols = np.array([], dtype=np.int32)
        self.zero_top_left_edge = np.array([], dtype=np.intp)
        self.zero_top_left_kind = np.array([], dtype=np.int8)
        self.zero_top_left_data = np.array([], dtype=np.float64)
        self.zero_top_right_rows = np.array([], dtype=np.int32)
        self.zero_top_right_cols = np.array([], dtype=np.int32)
        self.zero_top_right_edge = np.array([], dtype=np.intp)
        self.zero_top_right_kind = np.array([], dtype=np.int8)
        self.zero_top_right_data = np.array([], dtype=np.float64)
        self.zero_bottom_left_rows = np.array([], dtype=np.int32)
        self.zero_bottom_left_cols = np.array([], dtype=np.int32)
        self.zero_bottom_left_node = np.array([], dtype=np.int32)
        self.zero_bottom_left_kind = np.array([], dtype=np.int8)
        self.zero_bottom_left_data = np.array([], dtype=np.float64)
        self.zero_bottom_right_rows = np.array([], dtype=np.int32)
        self.zero_bottom_right_cols = np.array([], dtype=np.int32)
        self.zero_bottom_right_data = np.array([], dtype=np.float64)
        self.zero_top_left_pos_by_kind = ()
        self.zero_top_left_edge_by_kind = ()
        self.zero_top_right_pos_by_kind = ()
        self.zero_top_right_edge_by_kind = ()
        self.zero_bottom_left_pos_by_kind = ()
        self.zero_bottom_left_node_by_kind = ()
        self._zero_I_re = np.array([], dtype=np.float64)
        self._zero_I_im = np.array([], dtype=np.float64)
        self._zero_Va = np.array([], dtype=np.float64)
        self._zero_Vb = np.array([], dtype=np.float64)
        self._zero_cos_a = np.array([], dtype=np.float64)
        self._zero_sin_a = np.array([], dtype=np.float64)
        self._zero_cos_b = np.array([], dtype=np.float64)
        self._zero_sin_b = np.array([], dtype=np.float64)
        self._zero_a_pv = np.array([], dtype=np.float64)
        self._zero_a_pt = np.array([], dtype=np.float64)
        self._zero_b_pv = np.array([], dtype=np.float64)
        self._zero_b_pt = np.array([], dtype=np.float64)
        self._zero_va_cos = np.array([], dtype=np.float64)
        self._zero_va_sin = np.array([], dtype=np.float64)
        self._zero_vb_cos = np.array([], dtype=np.float64)
        self._zero_vb_sin = np.array([], dtype=np.float64)
        self._zero_tmp = np.array([], dtype=np.float64)
        self.pq_theta_rows = np.array([], dtype=np.int32)
        self.pq_v_cols = np.array([], dtype=np.int32)
        self.Y_diag = np.array([], dtype=np.complex128)
        self.Y_jac_rows = np.array([], dtype=np.int32)
        self.Y_jac_cols = np.array([], dtype=np.int32)
        self.Y_jac_data = np.array([], dtype=np.complex128)
        self.Y_jac_g = np.array([], dtype=np.float64)
        self.Y_jac_b = np.array([], dtype=np.float64)
        self.Y_jac_diag = np.array([], dtype=np.complex128)
        self.Y_offdiag_indices: List[np.ndarray] = []
        self.Y_offdiag_data: List[np.ndarray] = []
        self.theta_col_by_node = np.array([], dtype=np.int32)
        self.v_col_by_node = np.array([], dtype=np.int32)
        self.p_row_by_node = np.array([], dtype=np.int32)
        self.q_row_by_node = np.array([], dtype=np.int32)
        self.standard_jac_csr_indices = np.array([], dtype=np.int32)
        self.standard_jac_csr_indptr = np.array([], dtype=np.int32)
        self.standard_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
        self.standard_jac_csr_data = np.array([], dtype=np.float64)
        self.standard_jac_csr_sum_plan = build_raw_sum_plan(self.standard_jac_raw_to_csr_pos, 0)
        self.standard_jac_raw_to_csc_pos = np.array([], dtype=np.intp)
        self.standard_jac_csc_indices = np.array([], dtype=np.int32)
        self.standard_jac_csc_indptr = np.array([], dtype=np.int32)
        self.standard_jac_csc_data = np.array([], dtype=np.float64)
        self._standard_jac_csc_matrix = None
        self.standard_jac_csc_sum_plan = build_raw_sum_plan(self.standard_jac_raw_to_csc_pos, 0)
        self.full_jac_raw_data = np.array([], dtype=np.float64)
        self._full_jac_raw_rows = np.array([], dtype=np.int32)
        self._full_jac_raw_cols = np.array([], dtype=np.int32)
        self.full_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
        self.full_jac_csr_indices = np.array([], dtype=np.int32)
        self.full_jac_csr_indptr = np.array([], dtype=np.int32)
        self.full_jac_csr_data = np.array([], dtype=np.float64)
        self.full_jac_csr_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csr_pos, 0)
        self.full_jac_raw_to_csc_pos = np.array([], dtype=np.intp)
        self.full_jac_csc_indices = np.array([], dtype=np.int32)
        self.full_jac_csc_indptr = np.array([], dtype=np.int32)
        self.full_jac_csc_data = np.array([], dtype=np.float64)
        self._full_jac_csc_matrix = None
        self.full_jac_csc_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csc_pos, 0)
        self.full_jac_standard_slice = slice(0, 0)
        self.full_jac_zero_top_left_slice = slice(0, 0)
        self.full_jac_zero_top_right_slice = slice(0, 0)
        self.full_jac_zero_bottom_left_slice = slice(0, 0)
        self.full_jac_zero_bottom_right_slice = slice(0, 0)
        self.std_jac_load_nodes = np.array([], dtype=np.int32)
        self.std_jac_load_extra_nodes = np.array([], dtype=np.int32)
        self.std_jac_load_p_pos = np.array([], dtype=np.intp)
        self.std_jac_load_q_pos = np.array([], dtype=np.intp)
        self._jac_w_work = np.array([], dtype=np.complex128)
        self._jac_tmp = np.array([], dtype=np.float64)
        self._state_cos = np.array([], dtype=np.float64)
        self._state_sin = np.array([], dtype=np.float64)
        self._state_u_conj = np.array([], dtype=np.complex128)
        self._has_voltage_dependent_load = False
        self._const_p_load = np.array([], dtype=np.float64)
        self._const_q_load = np.array([], dtype=np.float64)
        self.live_gens: List = []
        self.live_loads: List = []
        self.live_shunts: List = []
        self.live_branches: List = []
        self.live_transformers: List = []
        self.live_zero_branches: List = []
        self.live_switches: List = []
        self.gen_pos = np.array([], dtype=np.int32)
        self.gen_share = np.array([], dtype=np.float64)
        self.load_obj_pos = np.array([], dtype=np.int32)
        self.shunt_pos = np.array([], dtype=np.int32)
        self.branch_i = np.array([], dtype=np.int32)
        self.branch_j = np.array([], dtype=np.int32)
        self.branch_yff = np.array([], dtype=np.complex128)
        self.branch_yft = np.array([], dtype=np.complex128)
        self.branch_ytf = np.array([], dtype=np.complex128)
        self.branch_ytt = np.array([], dtype=np.complex128)
        self.transformer_i = np.array([], dtype=np.int32)
        self.transformer_j = np.array([], dtype=np.int32)
        self.transformer_yff = np.array([], dtype=np.complex128)
        self.transformer_yft = np.array([], dtype=np.complex128)
        self.transformer_ytf = np.array([], dtype=np.complex128)
        self.transformer_ytt = np.array([], dtype=np.complex128)
        self.active_bus_rows = np.array([], dtype=np.int32)
        self.row_to_pos = np.array([], dtype=np.int32)
        self.ppc_gen_rows = np.array([], dtype=np.int32)
        self.ppc_gen_pos = np.array([], dtype=np.int32)
        self.ppc_gen_share = np.array([], dtype=np.float64)
        self.ppc_gen_p_min = np.array([], dtype=np.float64)
        self.ppc_gen_q_min = np.array([], dtype=np.float64)
        self.ppc_gen_q_max = np.array([], dtype=np.float64)
        self.ppc_auto_slack_mask = np.array([], dtype=bool)
        self._external_ac_p_injection = None
        self._external_ac_q_injection = None
        self.N_acac = 0
        self.acac_start = 0
        self.acac_eq_start = 0
        self.ac_core_vars = 0
        self.ac_core_eq = 0
        self.ppc_load_rows = np.array([], dtype=np.int32)
        self.ppc_load_pos = np.array([], dtype=np.int32)
        self.ppc_shunt_rows = np.array([], dtype=np.int32)
        self.ppc_shunt_pos = np.array([], dtype=np.int32)
        self.ppc_branch_rows = np.array([], dtype=np.int32)
        self.ppc_transformer_rows = np.array([], dtype=np.int32)
        self._state_theta = np.array([], dtype=np.float64)
        self._state_voltage = np.array([], dtype=np.float64)
        self._state_vc = np.array([], dtype=np.complex128)
        self._empty_phi = np.array([], dtype=np.float64)
        self._state_x_obj = None
        self._power_x_obj = None
        self._last_Ibus = None
        self._last_Sbus = None
        self._load_p_work = np.array([], dtype=np.float64)
        self._load_q_work = np.array([], dtype=np.float64)
        self._load_dp_work = np.array([], dtype=np.float64)
        self._load_dq_work = np.array([], dtype=np.float64)
        self._load_vm_work = np.array([], dtype=np.float64)
        self._load_value_work = np.array([], dtype=np.float64)
        self._load_aux_work = np.array([], dtype=np.float64)
        self._residual_work = np.array([], dtype=np.float64)
        self.result: Dict = {}
        self._clear_acac_arrays()

    @classmethod
    def from_file_fast(
        cls,
        file_name,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        min_voltage: Optional[float] = None,
        parameter_file=DEFAULT_LF_PARAMETER_FILE,
        parameters: Optional[PowerFlowParameters] = None,
        linear_solver: Optional[str] = None,
        result_mode: str = "array",
        verbose: bool = False,
    ) -> "ACPowerFlowCalc":
        """Build an AC PPC-backed solver directly from file.

        Accepted inputs:
        - `.e` named-unit/network files
        - `.m` / `.mat` MATPOWER cases
        """
        path = Path(file_name)
        suffix = path.suffix.lower()
        if suffix in {".m", ".mat"}:
            ppc = build_ac_ppc_from_mat_file(path)
        elif suffix == ".e":
            ppc = build_ac_ppc_from_e_file(path)
        else:
            raise ValueError(f"ACPowerFlowCalc.from_file_fast() only supports .e/.m/.mat files, got: {path}")
        return cls(
            ppc,
            tol=tol,
            max_iter=max_iter,
            min_voltage=min_voltage,
            parameter_file=parameter_file,
            parameters=parameters,
            linear_solver=linear_solver,
            result_mode=result_mode,
            verbose=verbose,
        )

    @staticmethod
    def _normalize_result_mode(result_mode: str) -> str:
        return _normalize_lf_result_mode(result_mode, "AC")

    def _cache_node_type_masks(self):
        self._slack_mask = self.node_type == AC_NODE_TYPE_SLACK
        self._fixed_voltage_mask = self._slack_mask | (self.node_type == AC_NODE_TYPE_PV)

    # --------------------------------------------------------------------------
    # 私有辅助函数（精简版）
    # --------------------------------------------------------------------------
    def _extract_state_vars(self, x: np.ndarray, update_cache: bool = True) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """将扁平 Newton 向量还原为 theta、V 和零阻抗支路的 phi 变量。"""
        theta = self._state_theta
        V = self._state_voltage

        # 已知节点（Slack/PV）的设定值已在 prepare 阶段预填，此处仅更新未知节点切片
        theta[self.theta_unknown] = x[:self.n_theta]
        V[self.V_unknown] = x[self.n_theta:self.n_theta + self.n_V]

        # 提取 phi 变量
        phi_re = x[self.base_phi_re:self.base_phi_re + self.N_phi] if self.N_phi > 0 else self._empty_phi
        phi_im = x[self.base_phi_im:self.base_phi_im + self.N_phi] if self.N_phi > 0 else self._empty_phi

        # 更新缓存（原地复用预分配三角函数和复电压缓冲区）
        if update_cache:
            cos_theta = self._state_cos
            sin_theta = self._state_sin
            np.cos(theta, out=cos_theta)
            np.sin(theta, out=sin_theta)
            Vc = self._state_vc
            np.multiply(V, cos_theta, out=Vc.real)
            np.multiply(V, sin_theta, out=Vc.imag)
            u_conj = self._state_u_conj
            u_conj.real = cos_theta
            np.negative(sin_theta, out=u_conj.imag)
            cache = self._cache
            cache['theta'] = theta
            cache['V'] = V
            cache['cos_theta'] = cos_theta
            cache['sin_theta'] = sin_theta
            cache['Vc'] = Vc
            self._state_x_obj = x
            self._power_x_obj = None
            self._last_Ibus = None
            self._last_Sbus = None

        return theta, V, phi_re, phi_im

    def _calc_power_balance(self, theta: np.ndarray, V: np.ndarray, phi_re: np.ndarray, phi_im: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        """计算每个节点的 P/Q 不平衡量，包含 Y 矩阵注入、负荷和零阻抗电流。"""
        # 从缓存获取预计算值
        cache = self._cache
        Vc = cache['Vc']

        # 导纳注入功率
        I_y = self.Y.dot(Vc)
        S_y = Vc * np.conj(I_y)
        self._power_x_obj = self._state_x_obj
        self._last_Ibus = I_y
        self._last_Sbus = S_y

        # 负荷功率
        P_load, Q_load = self._calc_load_power(V)
        cache['P_load'] = P_load
        cache['Q_load'] = Q_load

        # 功率不平衡量（无零阻抗支路时跳过零数组分配与加法）
        if self.N_phi > 0 and self.zero_a.size:
            P_zero, Q_zero = self._calc_zero_branch_power(Vc, phi_re, phi_im)
            return S_y.real + P_zero - self.P_spec + P_load, S_y.imag + Q_zero - self.Q_spec + Q_load
        return S_y.real - self.P_spec + P_load, S_y.imag - self.Q_spec + Q_load

    def _calc_load_power(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算各节点负荷功率，恒功率负荷直接复用预聚合常量数组。"""
        if not self._has_voltage_dependent_load:
            return self._const_p_load, self._const_q_load

        P_load = self._load_p_work
        Q_load = self._load_q_work
        P_load.fill(0.0)
        Q_load.fill(0.0)

        if self.load_pos.size:
            vm_vals = self._load_vm_work
            values = self._load_value_work
            aux = self._load_aux_work
            np.take(V, self.load_pos, out=vm_vals)

            # 向量化计算负荷功率
            np.multiply(vm_vals, vm_vals, out=values)
            values *= self.load_pv2
            np.multiply(self.load_pv1, vm_vals, out=aux)
            values += aux
            values += self.load_pv0

            # 累加至对应节点（bincount 比 np.add.at 快 5-10×）。
            P_load[:] = np.bincount(self.load_pos, weights=values, minlength=self.N)

            np.multiply(vm_vals, vm_vals, out=values)
            values *= self.load_qv2
            np.multiply(self.load_qv1, vm_vals, out=aux)
            values += aux
            values += self.load_qv0
            Q_load[:] = np.bincount(self.load_pos, weights=values, minlength=self.N)

        return P_load, Q_load

    def _calc_zero_branch_power(self, Vc: np.ndarray, phi_re: np.ndarray, phi_im: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        """把零阻抗支路的显式电流变量转换为两端节点功率注入。"""
        P_zero = np.zeros(self.N, dtype=np.float64)
        Q_zero = np.zeros(self.N, dtype=np.float64)

        if self.N_phi > 0 and self.zero_a.size:
            I_ab_re = phi_re[self.zero_phi_a] - phi_re[self.zero_phi_b]
            I_ab_im = phi_im[self.zero_phi_a] - phi_im[self.zero_phi_b]

            # 向量化计算功率（精简计算公式）
            Vc_a, Vc_b = Vc[self.zero_a], Vc[self.zero_b]
            Sa = Vc_a * (I_ab_re - 1j * I_ab_im)
            Sb = Vc_b * (-I_ab_re + 1j * I_ab_im)

            # bincount 比 np.add.at 快得多；分两端调用避免引入 np.concatenate。
            P_zero = np.bincount(self.zero_a, weights=Sa.real, minlength=self.N)
            P_zero += np.bincount(self.zero_b, weights=Sb.real, minlength=self.N)
            Q_zero = np.bincount(self.zero_a, weights=Sa.imag, minlength=self.N)
            Q_zero += np.bincount(self.zero_b, weights=Sb.imag, minlength=self.N)

        return P_zero, Q_zero

    def get_current(self, v: float, p: float, q: float):
        return abs(np.conj(complex(p, q)) / v) if abs(v) > self.min_voltage else 0.0

    def _ppc_node_rows(self, node_ids) -> np.ndarray:
        node_ids = np.asarray(node_ids, dtype=np.int64)
        if self._ppc_sequential_node_ids:
            return node_ids.astype(np.int32)
        lookup = getattr(self, "_ppc_node_row_lookup", None)
        if isinstance(lookup, np.ndarray) and lookup.size:
            rows = np.full(node_ids.shape, -1, dtype=np.int32)
            valid = (node_ids >= 0) & (node_ids < lookup.size)
            if np.any(valid):
                rows[valid] = lookup[node_ids[valid].astype(np.intp, copy=False)]
            if np.any(rows < 0):
                missing = int(node_ids[np.flatnonzero(rows < 0)[0]])
                raise KeyError(missing)
            return rows
        return np.fromiter(
            (self._ppc_node_row_by_id[int(node_id)] for node_id in node_ids),
            dtype=np.int32,
            count=node_ids.size,
        )

    def _bind_ppc_nodes_to_network(self) -> None:
        """Use original ACNode objects for network-input array solves."""
        network = getattr(self, "_network_writeback", None)
        if network is None or not self.keep_node_objects:
            return
        node_by_idx = {int(node.idx): node for node in getattr(network, "nodes", [])}
        if not node_by_idx:
            return
        bound_nodes = []
        for pos, node_idx in enumerate(self.ppc_node_idx):
            node = node_by_idx.get(int(node_idx))
            if node is None:
                node = self.node_list[pos] if pos < len(self.node_list) else _PPCNode(
                    int(node_idx),
                    str(self.ppc_node_name[pos]) if pos < self.ppc_node_name.size else f"bus_{int(node_idx)}",
                    float(self.ppc_node_vbase[pos]),
                    float(self.ppc_node_voltage[pos]),
                    float(self.ppc_node_angle[pos]),
                )
            bound_nodes.append(node)
        self.node_list = bound_nodes
        self.node_pos = {int(node_idx): pos for pos, node_idx in enumerate(self.ppc_node_idx)}

    def _prepare_from_ppc(self):
        """Prepare a Newton system from an already arrayized AC ppc dictionary."""
        ppc = self.ppc
        apply_lf_closed_status_boundaries(ppc, SWITCH_COLS)
        bus = np.asarray(ppc["bus"], dtype=np.float64)
        branch = np.asarray(ppc.get("branch", np.zeros((0, len(BRANCH_COLS)))), dtype=np.float64)
        transformer = np.asarray(ppc.get("transformer", np.zeros((0, len(TRANSFORMER_COLS)))), dtype=np.float64)
        three_winding_transformer = np.asarray(
            ppc.get(
                "three_winding_transformer",
                np.zeros((0, len(THREE_WINDING_TRANSFORMER_COLS))),
            ),
            dtype=np.float64,
        )
        gen = np.asarray(ppc.get("gen", np.zeros((0, len(GEN_COLS)))), dtype=np.float64)
        load = np.asarray(ppc.get("load", np.zeros((0, len(LOAD_COLS)))), dtype=np.float64)
        shunt = np.asarray(ppc.get("shunt", np.zeros((0, len(SHUNT_COLS)))), dtype=np.float64)
        zero_branch = np.asarray(ppc.get("zero_branch", np.zeros((0, len(ZERO_BRANCH_COLS)))), dtype=np.float64)
        switch = np.asarray(ppc.get("switch", np.zeros((0, len(SWITCH_COLS)))), dtype=np.float64)
        breaker = np.asarray(ppc.get("break", np.zeros((0, len(SWITCH_COLS)))), dtype=np.float64)
        acac = np.asarray(ppc.get("acac", np.zeros((0, len(ACAC_COLS)))), dtype=np.float64)

        n_bus_all = bus.shape[0]
        bus_ids = bus[:, BUS_COLS["idx"]].astype(np.int64)
        self._ppc_sequential_node_ids = bool(np.array_equal(bus_ids, np.arange(n_bus_all)))
        self._ppc_node_row_lookup = np.array([], dtype=np.int32)
        self._ppc_node_row_by_id = {}
        if not self._ppc_sequential_node_ids:
            if bus_ids.size and np.all(bus_ids >= 0):
                self._ppc_node_row_lookup = np.full(int(np.max(bus_ids)) + 1, -1, dtype=np.int32)
                self._ppc_node_row_lookup[bus_ids.astype(np.intp, copy=False)] = np.arange(
                    n_bus_all,
                    dtype=np.int32,
                )
            else:
                self._ppc_node_row_by_id = {int(node_id): pos for pos, node_id in enumerate(bus_ids)}
        if not (ppc.pop("_topology_preverified", False) and ppc.get("_topology_arrays") is not None):
            ensure_ac_ppc_topology(ppc)
        if not ppc.pop("_setpoints_preverified", False):
            sanitize_ac_voltage_setpoints(ppc)
        topology = ppc["_topology_arrays"]
        self._ppc_topology = topology
        active_node = np.asarray(topology.node_alive_mask, dtype=bool)
        if not np.any(active_node):
            raise RuntimeError("电网中无带平衡节点的存活拓扑岛，无法进行潮流计算")

        self._prepare_ppc_solver_bus_mapping(topology, bus, gen, shunt, acac)

        active_rows = self.active_bus_rows
        self.ppc_node_idx = bus[active_rows, BUS_COLS["idx"]].astype(np.int64, copy=True)
        self.ppc_node_vbase = bus[active_rows, BUS_COLS["vbase"]].astype(np.float64, copy=True)
        self.ppc_node_voltage = self._solver_initial_voltage.astype(np.float64, copy=True)
        self.ppc_node_angle = self._solver_initial_angle.astype(np.float64, copy=True)
        active_node_ids = bus[self.active_node_rows, BUS_COLS["idx"]].astype(np.int64, copy=False)
        if active_node_ids.size and np.all(active_node_ids >= 0):
            self._active_node_solver_lookup = np.full(int(np.max(active_node_ids)) + 1, -1, dtype=np.int32)
            self._active_node_solver_lookup[active_node_ids.astype(np.intp, copy=False)] = self.active_node_solver_pos
        else:
            self._active_node_solver_lookup = np.array([], dtype=np.int32)
        if self.keep_node_objects:
            bus_names = ppc.get("bus_name", np.asarray([f"bus_{int(idx)}" for idx in bus_ids], dtype=object))
            self.ppc_node_name = np.asarray(bus_names[active_rows], dtype=object)
            self.node_list = [
                _PPCNode(
                    idx=int(idx),
                    name=str(name),
                    vbase=float(vbase),
                    voltage=float(voltage),
                    angle=float(angle),
                )
                for idx, name, vbase, voltage, angle in zip(
                    self.ppc_node_idx,
                    self.ppc_node_name,
                    self.ppc_node_vbase,
                    self.ppc_node_voltage,
                    self.ppc_node_angle,
                )
            ]
            self.node_pos = {node.idx: pos for pos, node in enumerate(self.node_list)}
        else:
            self.ppc_node_name = np.empty(self.N, dtype=object)
            self.node_list = []
            self.node_pos = {}
        self._bind_ppc_nodes_to_network()

        self.node_type = np.full(self.N, AC_NODE_TYPE_PQ, dtype=np.int8)
        self.V_spec = np.full(self.N, np.nan, dtype=np.float64)
        self.theta_spec = np.zeros(self.N, dtype=np.float64)
        self.P_spec = np.zeros(self.N, dtype=np.float64)
        self.Q_spec = np.zeros(self.N, dtype=np.float64)
        self.load_info = []
        self.slack_node = -1

        self._prepare_ppc_devices(branch, transformer, gen, load, shunt, zero_branch, switch, active_node)
        self._prepare_ppc_y_matrix(branch, transformer, three_winding_transformer, shunt)
        self._prepare_ppc_zero_edges(zero_branch, switch, active_node, breaker)
        has_acac_rows = bool(acac is not None and acac.size > 0)
        self._finalize_prepared_arrays(defer_full_pattern=has_acac_rows)
        self._prepare_acac_converters(acac)
        if has_acac_rows and self.N_acac == 0:
            # 若 ACAC 表非空但无有效在运设备，补齐标准/全量雅可比稀疏模式
            if self.N_phi > 0:
                self._cache_full_jacobian_pattern()
            elif not getattr(self, "_defer_csc_pattern", False):
                self._ensure_standard_csc_pattern()
        if self.verbose:
            print(f"预处理完成：节点数 {self.N}, 变量数 {self.total_vars}, 方程数 {self.total_eq}")

    def _prepare_ppc_solver_bus_mapping(self, topology, bus, gen, shunt, acac):
        """Map existing topology buses to LF buses after control-aware ideal-tie grouping."""
        controlled_bus_parts = []
        gen_topology = topology.devices.get("gen")
        if gen.size and gen_topology is not None:
            controls = gen[:, GEN_COLS["control_type"]].astype(np.int32, copy=False)
            alive = np.asarray(gen_topology.alive_mask, dtype=bool)
            controlled = alive & ((controls == CTRL_SLACK) | (controls == CTRL_PV))
            if np.any(controlled):
                controlled_bus_parts.append(
                    np.asarray(gen_topology.bus_pos, dtype=np.int32)[controlled]
                )

        shunt_topology = topology.devices.get("shunt")
        if shunt.size and shunt_topology is not None:
            controls = shunt[:, SHUNT_COLS["control_type"]].astype(np.int32, copy=False)
            alive = np.asarray(shunt_topology.alive_mask, dtype=bool)
            controlled = alive & (controls == SHUNT_V)
            if np.any(controlled):
                controlled_bus_parts.append(
                    np.asarray(shunt_topology.bus_pos, dtype=np.int32)[controlled]
                )

        if acac.size:
            run = acac[:, ACAC_COLS["run_stat"]].astype(np.int8, copy=False) == 1
            voltage_codes = (ACAC_SIDE_CONTROL_CODE["PV"], ACAC_SIDE_CONTROL_CODE["PH"])
            for node_column, control_column in (
                ("i_node", "i_control_type"),
                ("j_node", "j_control_type"),
            ):
                control = acac[:, ACAC_COLS[control_column]].astype(np.int8, copy=False)
                mask = run & np.isin(control, voltage_codes)
                if not np.any(mask):
                    continue
                node_rows = self._ppc_node_rows(acac[mask, ACAC_COLS[node_column]])
                valid = (node_rows >= 0) & (node_rows < topology.node_to_bus_pos.size)
                if np.any(valid):
                    controlled_bus_parts.append(
                        topology.node_to_bus_pos[node_rows[valid]].astype(np.int32, copy=False)
                    )

        for key in ("_external_angle_reference_node_ids", "_external_voltage_control_node_ids"):
            node_ids = np.asarray(self.ppc.get(key, ()), dtype=np.int64)
            if not node_ids.size:
                continue
            node_rows = self._ppc_node_rows(node_ids)
            valid = (node_rows >= 0) & (node_rows < topology.node_to_bus_pos.size)
            if np.any(valid):
                controlled_bus_parts.append(
                    topology.node_to_bus_pos[node_rows[valid]].astype(np.int32, copy=False)
                )

        controlled_bus_positions = (
            np.concatenate(controlled_bus_parts).astype(np.int32, copy=False)
            if controlled_bus_parts
            else np.asarray([], dtype=np.int32)
        )
        groups = build_lf_control_bus_groups(topology, controlled_bus_positions)
        self._lf_control_bus_groups = groups

        active_topology_bus = np.flatnonzero(topology.bus_alive_mask).astype(np.int32, copy=False)
        active_group = np.unique(groups.bus_to_group_pos[active_topology_bus]).astype(np.int32, copy=False)
        group_to_solver = np.full(groups.group_ids.size, -1, dtype=np.int32)
        group_to_solver[active_group] = np.arange(active_group.size, dtype=np.int32)
        self._topology_bus_to_solver_pos = group_to_solver[groups.bus_to_group_pos]

        node_bus_pos = np.asarray(topology.node_to_bus_pos, dtype=np.int32)
        self.row_to_pos = np.full(bus.shape[0], -1, dtype=np.int32)
        valid_node = (
            np.asarray(topology.node_alive_mask, dtype=bool)
            & (node_bus_pos >= 0)
            & (node_bus_pos < self._topology_bus_to_solver_pos.size)
        )
        self.row_to_pos[valid_node] = self._topology_bus_to_solver_pos[node_bus_pos[valid_node]]
        self.active_node_rows = np.flatnonzero(self.row_to_pos >= 0).astype(np.int32, copy=False)
        self.active_node_solver_pos = self.row_to_pos[self.active_node_rows]
        self.N = int(active_group.size)

        order = np.lexsort(
            (
                bus[self.active_node_rows, BUS_COLS["idx"]],
                self.active_node_solver_pos,
            )
        )
        sorted_rows = self.active_node_rows[order]
        sorted_solver = self.active_node_solver_pos[order]
        first = np.concatenate((np.asarray([True]), sorted_solver[1:] != sorted_solver[:-1]))
        self.active_bus_rows = sorted_rows[first].astype(np.int32, copy=False)

        count = np.bincount(self.active_node_solver_pos, minlength=self.N)
        voltage_sum = np.bincount(
            self.active_node_solver_pos,
            weights=bus[self.active_node_rows, BUS_COLS["voltage"]],
            minlength=self.N,
        )
        angle_sum = np.bincount(
            self.active_node_solver_pos,
            weights=bus[self.active_node_rows, BUS_COLS["angle"]],
            minlength=self.N,
        )
        self._solver_initial_voltage = voltage_sum / count
        self._solver_initial_angle = angle_sum / count

        vbase = bus[self.active_node_rows, BUS_COLS["vbase"]]
        vbase_min = np.full(self.N, np.inf, dtype=np.float64)
        vbase_max = np.full(self.N, -np.inf, dtype=np.float64)
        np.minimum.at(vbase_min, self.active_node_solver_pos, vbase)
        np.maximum.at(vbase_max, self.active_node_solver_pos, vbase)
        bad_vbase = np.flatnonzero(np.abs(vbase_max - vbase_min) > 1e-9)
        if bad_vbase.size:
            pos = int(bad_vbase[0])
            raise ValueError(
                "零阻抗等值母线包含不一致的 AC 电压基值: "
                f"solver_bus={pos}, min={vbase_min[pos]}, max={vbase_max[pos]}"
            )

    def _prepare_ppc_devices(self, branch, transformer, gen, load, shunt, zero_branch, switch, active_bus):
        """Convert live ppc device rows into compact node positions and specification arrays."""
        v_set_sum = np.zeros(self.N, dtype=np.float64)
        v_set_count = np.zeros(self.N, dtype=np.int32)
        theta_set_sum = np.zeros(self.N, dtype=np.float64)
        theta_set_count = np.zeros(self.N, dtype=np.int32)
        if gen.size:
            gen_rows = self._ppc_node_rows(gen[:, GEN_COLS["node"]])
            gen_live = (gen[:, GEN_COLS["run_stat"]] == 1) & active_bus[gen_rows]
            self.ppc_gen_rows = np.where(gen_live)[0].astype(np.int32)
            self.ppc_gen_pos = self.row_to_pos[gen_rows[self.ppc_gen_rows]]
            self.ppc_gen_share = np.ones(self.ppc_gen_rows.size, dtype=np.float64)
            live_gen = gen[self.ppc_gen_rows]
            p_min = optional_ppc_vector(self.ppc, "_gen_p_min", gen.shape[0])
            q_min = optional_ppc_vector(self.ppc, "_gen_q_min", gen.shape[0])
            q_max = optional_ppc_vector(self.ppc, "_gen_q_max", gen.shape[0])
            self.ppc_gen_p_min = p_min[self.ppc_gen_rows]
            self.ppc_gen_q_min = q_min[self.ppc_gen_rows]
            self.ppc_gen_q_max = q_max[self.ppc_gen_rows]
            controls = live_gen[:, GEN_COLS["control_type"]].astype(np.int32, copy=False)

            auto_slack_rows = np.asarray(self.ppc.get("_auto_slack_gen_rows", ()), dtype=np.int32)
            auto_slack_mask = (
                np.isin(self.ppc_gen_rows, auto_slack_rows)
                if auto_slack_rows.size
                else np.zeros(self.ppc_gen_rows.size, dtype=bool)
            )
            self.ppc_auto_slack_mask = auto_slack_mask
            slack_mask = (controls == CTRL_SLACK) | auto_slack_mask
            if np.any(slack_mask):
                slack_pos = self.ppc_gen_pos[slack_mask]
                self.node_type[slack_pos] = AC_NODE_TYPE_SLACK
                self.slack_node = int(slack_pos[-1])
                np.add.at(v_set_sum, slack_pos, live_gen[slack_mask, GEN_COLS["v_set"]])
                np.add.at(v_set_count, slack_pos, 1)
                slack_gen_rows = gen_rows[self.ppc_gen_rows[slack_mask]]
                np.add.at(
                    theta_set_sum,
                    slack_pos,
                    self.ppc["bus"][slack_gen_rows, BUS_COLS["angle"]],
                )
                np.add.at(theta_set_count, slack_pos, 1)

            pv_control_mask = controls == CTRL_PV
            if np.any(pv_control_mask):
                pv_pos = self.ppc_gen_pos[pv_control_mask]
                np.add.at(self.P_spec, pv_pos, live_gen[pv_control_mask, GEN_COLS["p_set"]])
                np.add.at(v_set_sum, pv_pos, live_gen[pv_control_mask, GEN_COLS["v_set"]])
                np.add.at(v_set_count, pv_pos, 1)
                pv_node_mask = self.node_type[pv_pos] != AC_NODE_TYPE_SLACK
                if np.any(pv_node_mask):
                    self.node_type[pv_pos[pv_node_mask]] = AC_NODE_TYPE_PV

            pq_mask = (controls == CTRL_PQ) | (controls == CTRL_P)
            if np.any(pq_mask):
                pq_pos = self.ppc_gen_pos[pq_mask]
                np.add.at(self.P_spec, pq_pos, live_gen[pq_mask, GEN_COLS["p_set"]])
                np.add.at(self.Q_spec, pq_pos, live_gen[pq_mask, GEN_COLS["q_set"]])

            alpha = live_gen[:, GEN_COLS["alpha"]]
            alpha_sum = np.bincount(self.ppc_gen_pos, weights=alpha, minlength=self.N)
            gen_count = np.bincount(self.ppc_gen_pos, minlength=self.N)
            positive_alpha = alpha_sum[self.ppc_gen_pos] > 0.0
            self.ppc_gen_share[positive_alpha] = alpha[positive_alpha] / alpha_sum[self.ppc_gen_pos[positive_alpha]]
            no_alpha = ~positive_alpha
            if np.any(no_alpha):
                self.ppc_gen_share[no_alpha] = 1.0 / gen_count[self.ppc_gen_pos[no_alpha]]
            slack_count = np.bincount(self.ppc_gen_pos[slack_mask], minlength=self.N) if np.any(slack_mask) else np.zeros(self.N, dtype=np.int32)
            self.ppc_multi_slack_node_mask = slack_count > 1
            self.ppc_gen_control = controls.copy()
        else:
            self.ppc_multi_slack_node_mask = np.zeros(self.N, dtype=bool)
            self.ppc_gen_control = np.array([], dtype=np.int32)

        if shunt.size:
            shunt_rows = self._ppc_node_rows(shunt[:, SHUNT_COLS["node"]])
            shunt_live = (shunt[:, SHUNT_COLS["run_stat"]] == 1) & active_bus[shunt_rows]
            self.ppc_shunt_rows = np.where(shunt_live)[0].astype(np.int32)
            self.ppc_shunt_pos = self.row_to_pos[shunt_rows[self.ppc_shunt_rows]]
            live_shunt = shunt[self.ppc_shunt_rows]
            controls = live_shunt[:, SHUNT_COLS["control_type"]].astype(np.int32, copy=False)
            q_mask = controls == SHUNT_Q
            if np.any(q_mask):
                np.add.at(self.Q_spec, self.ppc_shunt_pos[q_mask], live_shunt[q_mask, SHUNT_COLS["q_set"]])
            v_control_mask = controls == SHUNT_V
            if np.any(v_control_mask):
                v_pos = self.ppc_shunt_pos[v_control_mask]
                v_set_values = live_shunt[v_control_mask, SHUNT_COLS["v_set"]]
                np.add.at(v_set_sum, v_pos, v_set_values)
                np.add.at(v_set_count, v_pos, 1)
                pv_node_mask = self.node_type[v_pos] != AC_NODE_TYPE_SLACK
                if np.any(pv_node_mask):
                    self.node_type[v_pos[pv_node_mask]] = AC_NODE_TYPE_PV

        v_mask = v_set_count > 0
        if np.any(v_mask):
            self.V_spec[v_mask] = v_set_sum[v_mask] / v_set_count[v_mask]
        theta_mask = theta_set_count > 0
        if np.any(theta_mask):
            self.theta_spec[theta_mask] = theta_set_sum[theta_mask] / theta_set_count[theta_mask]

        if load.size:
            load_rows = self._ppc_node_rows(load[:, LOAD_COLS["node"]])
            load_live = (load[:, LOAD_COLS["run_stat"]] == 1) & active_bus[load_rows]
            self.ppc_load_rows = np.where(load_live)[0].astype(np.int32)
            self.ppc_load_pos = self.row_to_pos[load_rows[self.ppc_load_rows]]
            self.load_info = np.column_stack(
                (
                    self.ppc_load_pos,
                    load[self.ppc_load_rows, LOAD_COLS["pbase"]] * load[self.ppc_load_rows, LOAD_COLS["pv0"]],
                    load[self.ppc_load_rows, LOAD_COLS["pbase"]] * load[self.ppc_load_rows, LOAD_COLS["pv1"]],
                    load[self.ppc_load_rows, LOAD_COLS["pbase"]] * load[self.ppc_load_rows, LOAD_COLS["pv2"]],
                    load[self.ppc_load_rows, LOAD_COLS["qbase"]] * load[self.ppc_load_rows, LOAD_COLS["qv0"]],
                    load[self.ppc_load_rows, LOAD_COLS["qbase"]] * load[self.ppc_load_rows, LOAD_COLS["qv1"]],
                    load[self.ppc_load_rows, LOAD_COLS["qbase"]] * load[self.ppc_load_rows, LOAD_COLS["qv2"]],
                )
            )

        external_refs = np.asarray(self.ppc.get("_external_angle_reference_node_ids", []), dtype=np.int64)
        if self.slack_node == -1 and external_refs.size == 0:
            pv_indices = np.where(self.node_type == AC_NODE_TYPE_PV)[0]
            if pv_indices.size:
                self.slack_node = int(pv_indices[0])
                self.node_type[self.slack_node] = AC_NODE_TYPE_SLACK
                self.theta_spec[self.slack_node] = self.ppc["bus"][self.active_bus_rows[self.slack_node], BUS_COLS["angle"]]
        if self.slack_node == -1:
            if external_refs.size == 0:
                raise RuntimeError("电网中无平衡节点，无法进行潮流计算")

    def _prepare_ppc_y_matrix(self, branch, transformer, three_winding_transformer, shunt):
        row_parts, col_parts, data_parts = [], [], []
        self.ppc_three_winding_transformer_rows = np.asarray([], dtype=np.int32)
        self.three_winding_transformer_i = np.asarray([], dtype=np.int32)
        self.three_winding_transformer_j = np.asarray([], dtype=np.int32)
        self.three_winding_transformer_k = np.asarray([], dtype=np.int32)
        self.three_winding_transformer_y = np.zeros((0, 3, 3), dtype=np.complex128)
        if branch.size:
            i_rows = self._ppc_node_rows(branch[:, BRANCH_COLS["i_node"]])
            j_rows = self._ppc_node_rows(branch[:, BRANCH_COLS["j_node"]])
            live = (branch[:, BRANCH_COLS["run_stat"]] == 1) & (self.row_to_pos[i_rows] >= 0) & (self.row_to_pos[j_rows] >= 0)
            self.ppc_branch_rows = np.where(live)[0].astype(np.int32)
            self.branch_i = self.row_to_pos[i_rows[self.ppc_branch_rows]]
            self.branch_j = self.row_to_pos[j_rows[self.ppc_branch_rows]]
            self.branch_yff, self.branch_yft, self.branch_ytf, self.branch_ytt = matpower_branch_stamp_vectorized(
                branch[self.ppc_branch_rows, BRANCH_COLS["r"]],
                branch[self.ppc_branch_rows, BRANCH_COLS["x"]],
                branch[self.ppc_branch_rows, BRANCH_COLS["b"]],
            )
            row_parts.append(np.column_stack((self.branch_i, self.branch_i, self.branch_j, self.branch_j)).ravel())
            col_parts.append(np.column_stack((self.branch_i, self.branch_j, self.branch_i, self.branch_j)).ravel())
            data_parts.append(np.column_stack((self.branch_yff, self.branch_yft, self.branch_ytf, self.branch_ytt)).ravel())

        if transformer.size:
            i_rows = self._ppc_node_rows(transformer[:, TRANSFORMER_COLS["i_node"]])
            j_rows = self._ppc_node_rows(transformer[:, TRANSFORMER_COLS["j_node"]])
            live = (transformer[:, TRANSFORMER_COLS["run_stat"]] == 1) & (self.row_to_pos[i_rows] >= 0) & (self.row_to_pos[j_rows] >= 0)
            self.ppc_transformer_rows = np.where(live)[0].astype(np.int32)
            self.transformer_i = self.row_to_pos[i_rows[self.ppc_transformer_rows]]
            self.transformer_j = self.row_to_pos[j_rows[self.ppc_transformer_rows]]
            (
                self.transformer_yff,
                self.transformer_yft,
                self.transformer_ytf,
                self.transformer_ytt,
            ) = matpower_transformer_stamp_vectorized(
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["r"]],
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["x"]],
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["gt"]],
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["bt"]],
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["tap"]],
                transformer[self.ppc_transformer_rows, TRANSFORMER_COLS["shift"]],
            )
            row_parts.append(
                np.column_stack((self.transformer_i, self.transformer_i, self.transformer_j, self.transformer_j)).ravel()
            )
            col_parts.append(
                np.column_stack((self.transformer_i, self.transformer_j, self.transformer_i, self.transformer_j)).ravel()
            )
            data_parts.append(
                np.column_stack((self.transformer_yff, self.transformer_yft, self.transformer_ytf, self.transformer_ytt)).ravel()
            )

        if three_winding_transformer.size:
            cols3 = THREE_WINDING_TRANSFORMER_COLS
            i_rows = self._ppc_node_rows(three_winding_transformer[:, cols3["i_node"]])
            j_rows = self._ppc_node_rows(three_winding_transformer[:, cols3["j_node"]])
            k_rows = self._ppc_node_rows(three_winding_transformer[:, cols3["k_node"]])
            live = (
                (three_winding_transformer[:, cols3["run_stat"]] == 1)
                & (self.row_to_pos[i_rows] >= 0)
                & (self.row_to_pos[j_rows] >= 0)
                & (self.row_to_pos[k_rows] >= 0)
            )
            self.ppc_three_winding_transformer_rows = np.where(live)[0].astype(np.int32)
            rows3 = self.ppc_three_winding_transformer_rows
            self.three_winding_transformer_i = self.row_to_pos[i_rows[rows3]]
            self.three_winding_transformer_j = self.row_to_pos[j_rows[rows3]]
            self.three_winding_transformer_k = self.row_to_pos[k_rows[rows3]]
            table = three_winding_transformer[rows3]
            self.three_winding_transformer_y = three_winding_transformer_stamp_vectorized(
                table[:, cols3["i_r"]],
                table[:, cols3["i_x"]],
                table[:, cols3["j_r"]],
                table[:, cols3["j_x"]],
                table[:, cols3["k_r"]],
                table[:, cols3["k_x"]],
                table[:, cols3["gt"]],
                table[:, cols3["bt"]],
                table[:, cols3["i_tap"]],
                table[:, cols3["i_shift"]],
                table[:, cols3["j_tap"]],
                table[:, cols3["j_shift"]],
                table[:, cols3["k_tap"]],
                table[:, cols3["k_shift"]],
            )
            terminal_pos = np.column_stack(
                (
                    self.three_winding_transformer_i,
                    self.three_winding_transformer_j,
                    self.three_winding_transformer_k,
                )
            )
            row_parts.append(np.repeat(terminal_pos, 3, axis=1).ravel())
            col_parts.append(np.tile(terminal_pos, (1, 3)).ravel())
            data_parts.append(self.three_winding_transformer_y.reshape(-1))

        if shunt.size and self.ppc_shunt_rows.size:
            shunt_live = shunt[self.ppc_shunt_rows]
            control = shunt_live[:, SHUNT_COLS["control_type"]].astype(np.int32)
            y_sh = shunt_live[:, SHUNT_COLS["g_set"]] + 1j * shunt_live[:, SHUNT_COLS["b_set"]]
            mask = ((control == SHUNT_B) | (control == SHUNT_Z) | (shunt_live[:, SHUNT_COLS["g_set"]] != 0.0)) & (y_sh != 0.0)
            if np.any(mask):
                shunt_pos = self.ppc_shunt_pos[mask]
                row_parts.append(shunt_pos)
                col_parts.append(shunt_pos)
                data_parts.append(y_sh[mask])

        if row_parts:
            rows = np.concatenate(row_parts).astype(np.int32, copy=False)
            cols = np.concatenate(col_parts).astype(np.int32, copy=False)
            data = np.concatenate(data_parts).astype(np.complex128, copy=False)
        else:
            rows = cols = np.array([], dtype=np.int32)
            data = np.array([], dtype=np.complex128)
        self.Y = csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        self.Y.sum_duplicates()

    def _prepare_ppc_zero_edges(self, zero_branch, switch, active_bus, breaker=None):
        idx_parts = []
        type_parts = []
        a_parts = []
        b_parts = []

        def collect_table(table, cols, type_code: int, require_closed: bool):
            if table is None or not table.size:
                return
            i_rows = self._ppc_node_rows(table[:, cols["i_node"]])
            j_rows = self._ppc_node_rows(table[:, cols["j_node"]])
            live = (table[:, cols["run_stat"]] == 1) & (i_rows >= 0) & (j_rows >= 0)
            if require_closed:
                live &= table[:, cols["closed_status"]] == 1
            if not np.any(live):
                return
            cand_rows = np.flatnonzero(live)
            ci = i_rows[cand_rows]
            cj = j_rows[cand_rows]
            act = active_bus[ci] & active_bus[cj]
            if not np.any(act):
                return
            cand_rows = cand_rows[act]
            a_arr = self.row_to_pos[ci[act]].astype(np.int32, copy=False)
            b_arr = self.row_to_pos[cj[act]].astype(np.int32, copy=False)
            keep = a_arr != b_arr
            if self.node_type.size and np.any(keep):
                both_slack = (
                    (self.node_type[a_arr] == AC_NODE_TYPE_SLACK)
                    & (self.node_type[b_arr] == AC_NODE_TYPE_SLACK)
                )
                if np.any(both_slack):
                    redundant = (
                        both_slack
                        & (np.abs(self.V_spec[a_arr] - self.V_spec[b_arr]) <= 1e-10)
                        & (np.abs(self.theta_spec[a_arr] - self.theta_spec[b_arr]) <= 1e-10)
                    )
                    keep &= ~redundant
            if np.any(keep):
                idx_parts.append(cand_rows[keep].astype(np.int32, copy=False))
                type_parts.append(np.full(int(np.count_nonzero(keep)), type_code, dtype=np.int32))
                a_parts.append(a_arr[keep])
                b_parts.append(b_arr[keep])

        collect_table(zero_branch, ZERO_BRANCH_COLS, 0, False)
        collect_table(breaker, SWITCH_COLS, 2, True)
        collect_table(switch, SWITCH_COLS, 1, True)

        if idx_parts:
            self._zero_edge_idx_arr = idx_parts[0] if len(idx_parts) == 1 else np.concatenate(idx_parts)
            self._zero_edge_type_arr = type_parts[0] if len(type_parts) == 1 else np.concatenate(type_parts)
            self._zero_edge_a_arr = a_parts[0] if len(a_parts) == 1 else np.concatenate(a_parts)
            self._zero_edge_b_arr = b_parts[0] if len(b_parts) == 1 else np.concatenate(b_parts)
        else:
            empty_i32 = np.empty(0, dtype=np.int32)
            self._zero_edge_idx_arr = empty_i32
            self._zero_edge_type_arr = empty_i32
            self._zero_edge_a_arr = empty_i32
            self._zero_edge_b_arr = empty_i32

        if self.keep_node_objects and self._zero_edge_idx_arr.size:
            self.zero_edges = list(
                zip(
                    self._zero_edge_idx_arr.tolist(),
                    self._zero_edge_type_arr.tolist(),
                    self._zero_edge_a_arr.tolist(),
                    self._zero_edge_b_arr.tolist(),
                )
            )
        else:
            self.zero_edges = []
        self._prepare_zero_branch_components()

    def _finalize_prepared_arrays(self, *, defer_full_pattern: bool = False):
        """Finalize variable/equation indices after node specs, Y and zero edges are ready."""
        self.theta_unknown = np.where(self.node_type != AC_NODE_TYPE_SLACK)[0].astype(np.int32)
        self.V_unknown = np.where(self.node_type == AC_NODE_TYPE_PQ)[0].astype(np.int32)
        self.n_theta = len(self.theta_unknown)
        self.n_V = len(self.V_unknown)
        self.theta_idx = np.full(self.N, -1, dtype=np.int32)
        self.V_idx = np.full(self.N, -1, dtype=np.int32)
        self.theta_idx[self.theta_unknown] = np.arange(self.n_theta, dtype=np.int32)
        self.V_idx[self.V_unknown] = np.arange(self.n_V, dtype=np.int32)
        self.base_phi_re = self.n_theta + self.n_V
        self.base_phi_im = self.base_phi_re + self.N_phi
        self.total_vars = self.base_phi_im + self.N_phi
        n_tree = int(getattr(self, "_zero_tree_a", np.empty(0, dtype=np.int32)).size)
        n_phi_fix = 2 * int(getattr(self, "_n_zero_comp", len(self.comp_nodes)))
        self.total_eq = self.n_theta + self.n_V + 2 * n_tree + n_phi_fix
        self.ac_core_vars = self.total_vars
        self.ac_core_eq = self.total_eq
        self.acac_start = self.ac_core_vars
        self.acac_eq_start = self.ac_core_eq
        self.x = np.zeros(self.total_vars, dtype=np.float64)
        self.x[self.n_theta:self.n_theta + self.n_V] = 1.0
        self._state_theta = np.zeros(self.N, dtype=np.float64)
        self._state_voltage = np.ones(self.N, dtype=np.float64)
        self._state_cos = np.empty(self.N, dtype=np.float64)
        self._state_sin = np.empty(self.N, dtype=np.float64)
        self._state_vc = np.empty(self.N, dtype=np.complex128)
        self._state_u_conj = np.empty(self.N, dtype=np.complex128)
        self._empty_phi = np.array([], dtype=np.float64)
        self._load_p_work = np.zeros(self.N, dtype=np.float64)
        self._load_q_work = np.zeros(self.N, dtype=np.float64)
        self._load_dp_work = np.zeros(self.N, dtype=np.float64)
        self._load_dq_work = np.zeros(self.N, dtype=np.float64)
        self._load_vm_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._load_value_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._load_aux_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._residual_work = np.zeros(self.total_eq, dtype=np.float64)
        # 残差里零阻抗树边直接使用向量化构造的 (a, b) 数组。
        self._zero_residual_a = getattr(self, "_zero_tree_a", np.empty(0, dtype=np.int32))
        self._zero_residual_b = getattr(self, "_zero_tree_b", np.empty(0, dtype=np.int32))
        self._ref_phi_idx_arr = (
            np.asarray(self.ref_phi_idx, dtype=np.int32)
            if len(self.ref_phi_idx)
            else np.empty(0, dtype=np.int32)
        )
        self._cache_static_numeric_arrays(defer_full_pattern=defer_full_pattern)
        if self.total_vars != self.total_eq:
            warnings.warn(f"变量数({self.total_vars})与方程数({self.total_eq})不匹配！")

    def _clear_acac_arrays(self):
        """Reset ACAC converter metadata and sparse-assembly plans."""
        self.acac_row_pos = np.array([], dtype=np.int32)
        self.acac_idx = np.array([], dtype=np.int32)
        self.acac_names = np.array([], dtype=object)
        self.acac_i_pos = np.array([], dtype=np.int32)
        self.acac_j_pos = np.array([], dtype=np.int32)
        self.acac_ctrl_code = np.array([], dtype=np.int8)
        self.acac_r1 = np.array([], dtype=np.float64)
        self.acac_r2 = np.array([], dtype=np.float64)
        self.acac_p_set = np.array([], dtype=np.float64)
        self.acac_i_q_set = np.array([], dtype=np.float64)
        self.acac_j_q_set = np.array([], dtype=np.float64)
        self.acac_i_v_set = np.array([], dtype=np.float64)
        self.acac_j_v_set = np.array([], dtype=np.float64)
        self.acac_i_p_row = np.array([], dtype=np.int32)
        self.acac_i_q_row = np.array([], dtype=np.int32)
        self.acac_j_p_row = np.array([], dtype=np.int32)
        self.acac_j_q_row = np.array([], dtype=np.int32)
        self.acac_i_p_eq_mask = np.array([], dtype=bool)
        self.acac_i_q_eq_mask = np.array([], dtype=bool)
        self.acac_j_p_eq_mask = np.array([], dtype=bool)
        self.acac_j_q_eq_mask = np.array([], dtype=bool)
        self.acac_i_v_col = np.array([], dtype=np.int32)
        self.acac_j_v_col = np.array([], dtype=np.int32)
        self.acac_i_p_col = np.array([], dtype=np.int32)
        self.acac_i_q_col = np.array([], dtype=np.int32)
        self.acac_j_p_col = np.array([], dtype=np.int32)
        self.acac_j_q_col = np.array([], dtype=np.int32)
        self.acac_eq_loss = np.array([], dtype=np.int32)
        self.acac_eq_ctrl_1 = np.array([], dtype=np.int32)
        self.acac_eq_ctrl_2 = np.array([], dtype=np.int32)
        self.acac_eq_ctrl_3 = np.array([], dtype=np.int32)
        self.acac_loss_rows = np.array([], dtype=np.int32)
        self.acac_loss_cols = np.array([], dtype=np.int32)
        self.acac_q_i_mask = np.array([], dtype=bool)
        self.acac_v_i_mask = np.array([], dtype=bool)
        self.acac_q_j_mask = np.array([], dtype=bool)
        self.acac_v_j_mask = np.array([], dtype=bool)
        self.acac_i_v_avg_set = np.array([], dtype=np.float64)
        self.acac_i_v_rep_mask = np.array([], dtype=bool)
        self.acac_i_v_share_mask = np.array([], dtype=bool)
        self.acac_i_v_ref_idx = np.array([], dtype=np.int32)
        self.acac_j_v_avg_set = np.array([], dtype=np.float64)
        self.acac_j_v_rep_mask = np.array([], dtype=bool)
        self.acac_j_v_share_mask = np.array([], dtype=bool)
        self.acac_j_v_ref_idx = np.array([], dtype=np.int32)
        self.acac_ctrl_rows = np.array([], dtype=np.int32)
        self.acac_ctrl_cols = np.array([], dtype=np.int32)
        self.acac_ctrl_data = np.array([], dtype=np.float64)
        self.full_jac_acac_i_p_slice = slice(0, 0)
        self.full_jac_acac_i_q_slice = slice(0, 0)
        self.full_jac_acac_j_p_slice = slice(0, 0)
        self.full_jac_acac_j_q_slice = slice(0, 0)
        self.full_jac_acac_loss_slice = slice(0, 0)
        self.full_jac_acac_ctrl_slice = slice(0, 0)

    def _validate_acac_terminal_control(self, converter_idx, side, ac_pos, control_type):
        if control_type not in {"PV", "PH"}:
            return
        node_type = self.node_type[int(ac_pos)]
        if node_type == AC_NODE_TYPE_PQ:
            return
        raise ValueError(
            f"ACACConverter[{converter_idx}] 的 {side} 侧 {control_type} 控制与"
            f"拓扑后母线已有的 {ac_node_type_label(node_type)} 控制重复"
        )

    def _prepare_acac_converters(self, table):
        """Attach ACAC states and equations directly to the AC Newton block."""
        self.N_acac = 0
        self._clear_acac_arrays()
        if table is None or table.size == 0:
            return

        run_rows = np.flatnonzero(table[:, ACAC_COLS["run_stat"]].astype(np.int64, copy=False) == 1)
        if run_rows.size == 0:
            return
        cand = table[run_rows]
        i_rows = self._ppc_node_rows(cand[:, ACAC_COLS["i_node"]])
        j_rows = self._ppc_node_rows(cand[:, ACAC_COLS["j_node"]])
        valid_ij = (i_rows >= 0) & (j_rows >= 0)
        if not np.any(valid_ij):
            return
        run_rows = run_rows[valid_ij]
        cand = cand[valid_ij]
        i_sol = self.row_to_pos[i_rows[valid_ij]].astype(np.int32, copy=False)
        j_sol = self.row_to_pos[j_rows[valid_ij]].astype(np.int32, copy=False)
        valid_sol = (i_sol >= 0) & (j_sol >= 0)
        if not np.any(valid_sol):
            return
        run_rows = run_rows[valid_sol]
        cand = cand[valid_sol]
        i_sol = i_sol[valid_sol]
        j_sol = j_sol[valid_sol]

        same = i_sol == j_sol
        if np.any(same):
            bad_pos = int(np.flatnonzero(same)[0])
            idx = int(cand[bad_pos, ACAC_COLS["idx"]])
            raise ValueError(f"ACACConverter[{idx}] 两端不能连接同一个 AC 节点")

        i_code = cand[:, ACAC_COLS["i_control_type"]].astype(np.int32, copy=False)
        j_code = cand[:, ACAC_COLS["j_control_type"]].astype(np.int32, copy=False)
        pq_code = int(ACAC_SIDE_CONTROL_CODE["PQ"])
        pv_code = int(ACAC_SIDE_CONTROL_CODE["PV"])
        standard_codes = (
            ((i_code == pq_code) | (i_code == pv_code))
            & ((j_code == pq_code) | (j_code == pv_code))
        )
        if bool(np.all(standard_codes)):
            i_is_pv = i_code == pv_code
            j_is_pv = j_code == pv_code
            if np.any(i_is_pv):
                bad_i = i_is_pv & (self.node_type[i_sol] != AC_NODE_TYPE_PQ)
                if np.any(bad_i):
                    p = int(np.flatnonzero(bad_i)[0])
                    self._validate_acac_terminal_control(
                        int(cand[p, ACAC_COLS["idx"]]), "i", int(i_sol[p]), "PV"
                    )
            if np.any(j_is_pv):
                bad_j = j_is_pv & (self.node_type[j_sol] != AC_NODE_TYPE_PQ)
                if np.any(bad_j):
                    p = int(np.flatnonzero(bad_j)[0])
                    self._validate_acac_terminal_control(
                        int(cand[p, ACAC_COLS["idx"]]), "j", int(j_sol[p]), "PV"
                    )
            ctrl_arr = (i_is_pv.astype(np.int8) + 2 * j_is_pv.astype(np.int8)).astype(np.int8, copy=False)
        else:
            ctrl_list = []
            for p in range(run_rows.size):
                idx = int(cand[p, ACAC_COLS["idx"]])
                i_ctrl = ACAC_SIDE_CONTROL_LABEL.get(int(i_code[p]), "PQ")
                j_ctrl = ACAC_SIDE_CONTROL_LABEL.get(int(j_code[p]), "PQ")
                if i_ctrl not in ACAC_SIDE_CONTROL_TYPES:
                    raise ValueError(f"未知 ACACConverter i_control_type: {i_ctrl}")
                if j_ctrl not in ACAC_SIDE_CONTROL_TYPES:
                    raise ValueError(f"未知 ACACConverter j_control_type: {j_ctrl}")
                self._validate_acac_terminal_control(idx, "i", int(i_sol[p]), i_ctrl)
                self._validate_acac_terminal_control(idx, "j", int(j_sol[p]), j_ctrl)
                ctrl_list.append(acac_combined_control_code(i_ctrl, j_ctrl))
            ctrl_arr = np.asarray(ctrl_list, dtype=np.int8)

        self.N_acac = int(run_rows.size)
        self.acac_row_pos = run_rows.astype(np.int32, copy=False)
        active = cand
        self.acac_idx = active[:, ACAC_COLS["idx"]].astype(np.int32, copy=True)
        names = np.asarray(self.ppc.get("acac_name", ()), dtype=object)
        self.acac_names = (
            names[self.acac_row_pos].copy()
            if names.size == table.shape[0]
            else np.asarray([str(idx) for idx in self.acac_idx], dtype=object)
        )
        self.acac_i_pos = i_sol
        self.acac_j_pos = j_sol
        self.acac_ctrl_code = ctrl_arr
        self.acac_r1 = active[:, ACAC_COLS["r1"]].astype(np.float64, copy=True)
        self.acac_r2 = active[:, ACAC_COLS["r2"]].astype(np.float64, copy=True)
        self.acac_p_set = active[:, ACAC_COLS["p_set"]].astype(np.float64, copy=True)
        self.acac_i_q_set = active[:, ACAC_COLS["i_q_set"]].astype(np.float64, copy=True)
        self.acac_j_q_set = active[:, ACAC_COLS["j_q_set"]].astype(np.float64, copy=True)
        self.acac_i_v_set = active[:, ACAC_COLS["i_v_set"]].astype(np.float64, copy=True)
        self.acac_j_v_set = active[:, ACAC_COLS["j_v_set"]].astype(np.float64, copy=True)

        self.acac_i_p_row = self.p_row_by_node[self.acac_i_pos]
        self.acac_i_q_row = self.q_row_by_node[self.acac_i_pos]
        self.acac_j_p_row = self.p_row_by_node[self.acac_j_pos]
        self.acac_j_q_row = self.q_row_by_node[self.acac_j_pos]
        self.acac_i_p_eq_mask = self.acac_i_p_row >= 0
        self.acac_i_q_eq_mask = self.acac_i_q_row >= 0
        self.acac_j_p_eq_mask = self.acac_j_p_row >= 0
        self.acac_j_q_eq_mask = self.acac_j_q_row >= 0
        self.acac_i_v_col = self.v_col_by_node[self.acac_i_pos]
        self.acac_j_v_col = self.v_col_by_node[self.acac_j_pos]

        self.acac_start = self.ac_core_vars
        self.acac_eq_start = self.ac_core_eq
        idx = np.arange(self.N_acac, dtype=np.int32)
        self.acac_i_p_col = self.acac_start + 4 * idx
        self.acac_i_q_col = self.acac_i_p_col + 1
        self.acac_j_p_col = self.acac_i_p_col + 2
        self.acac_j_q_col = self.acac_i_p_col + 3
        self.acac_eq_loss = self.acac_eq_start + 4 * idx
        self.acac_eq_ctrl_1 = self.acac_eq_loss + 1
        self.acac_eq_ctrl_2 = self.acac_eq_loss + 2
        self.acac_eq_ctrl_3 = self.acac_eq_loss + 3
        self.total_vars = self.ac_core_vars + 4 * self.N_acac
        self.total_eq = self.ac_core_eq + 4 * self.N_acac
        self.x = np.concatenate((self.x, self._initial_acac_x()))
        self._residual_work = np.zeros(self.total_eq, dtype=np.float64)

        variable_i_v = self.acac_i_v_col >= 0
        variable_j_v = self.acac_j_v_col >= 0
        self.acac_loss_rows = np.concatenate(
            (
                self.acac_eq_loss,
                self.acac_eq_loss,
                self.acac_eq_loss,
                self.acac_eq_loss,
                self.acac_eq_loss[variable_i_v],
                self.acac_eq_loss[variable_j_v],
            )
        ).astype(np.int32, copy=False)
        self.acac_loss_cols = np.concatenate(
            (
                self.acac_i_p_col,
                self.acac_i_q_col,
                self.acac_j_p_col,
                self.acac_j_q_col,
                self.acac_i_v_col[variable_i_v],
                self.acac_j_v_col[variable_j_v],
            )
        ).astype(np.int32, copy=False)

        self.acac_q_i_mask = (self.acac_ctrl_code == 0) | (self.acac_ctrl_code == 2)
        self.acac_v_i_mask = ~self.acac_q_i_mask
        self.acac_q_j_mask = (self.acac_ctrl_code == 0) | (self.acac_ctrl_code == 1)
        self.acac_v_j_mask = ~self.acac_q_j_mask
        (
            self.acac_i_v_avg_set,
            self.acac_i_v_rep_mask,
            self.acac_i_v_share_mask,
            self.acac_i_v_ref_idx,
        ) = _build_group_share_metadata(self.acac_i_pos, self.acac_i_v_set, self.acac_v_i_mask)
        (
            self.acac_j_v_avg_set,
            self.acac_j_v_rep_mask,
            self.acac_j_v_share_mask,
            self.acac_j_v_ref_idx,
        ) = _build_group_share_metadata(self.acac_j_pos, self.acac_j_v_set, self.acac_v_j_mask)
        self._build_acac_control_jacobian_plan()
        self._cache_full_jacobian_pattern()

    def _initial_acac_x(self):
        x = np.zeros(self.N_acac * 4, dtype=np.float64)
        x[0::4] = self.acac_p_set
        x[1::4] = self.acac_i_q_set
        x[2::4] = -self.acac_p_set
        x[3::4] = self.acac_j_q_set
        return x

    def _build_acac_control_jacobian_plan(self):
        ctrl_rows = [self.acac_eq_ctrl_1]
        ctrl_cols = [self.acac_i_p_col]
        ctrl_data = [np.ones(self.N_acac, dtype=np.float64)]
        for q_mask, rep_mask, share_mask, ref_idx, eq_rows, q_cols, v_cols in (
            (
                self.acac_q_i_mask,
                self.acac_i_v_rep_mask,
                self.acac_i_v_share_mask,
                self.acac_i_v_ref_idx,
                self.acac_eq_ctrl_2,
                self.acac_i_q_col,
                self.acac_i_v_col,
            ),
            (
                self.acac_q_j_mask,
                self.acac_j_v_rep_mask,
                self.acac_j_v_share_mask,
                self.acac_j_v_ref_idx,
                self.acac_eq_ctrl_3,
                self.acac_j_q_col,
                self.acac_j_v_col,
            ),
        ):
            if np.any(q_mask):
                selected = np.flatnonzero(q_mask).astype(np.int32, copy=False)
                ctrl_rows.append(eq_rows[selected])
                ctrl_cols.append(q_cols[selected])
                ctrl_data.append(np.ones(selected.size, dtype=np.float64))
            if np.any(rep_mask):
                selected = np.flatnonzero(rep_mask).astype(np.int32, copy=False)
                ctrl_rows.append(eq_rows[selected])
                ctrl_cols.append(v_cols[selected])
                ctrl_data.append(np.ones(selected.size, dtype=np.float64))
            if np.any(share_mask):
                selected = np.flatnonzero(share_mask).astype(np.int32, copy=False)
                ref = ref_idx[selected]
                rows = np.repeat(eq_rows[selected], 2)
                cols = np.empty(2 * selected.size, dtype=np.int32)
                data = np.empty(2 * selected.size, dtype=np.float64)
                cols[0::2] = q_cols[selected]
                cols[1::2] = q_cols[ref]
                data[0::2] = 1.0
                data[1::2] = -1.0
                ctrl_rows.append(rows)
                ctrl_cols.append(cols)
                ctrl_data.append(data)
        self.acac_ctrl_rows = np.concatenate(ctrl_rows).astype(np.int32, copy=False)
        self.acac_ctrl_cols = np.concatenate(ctrl_cols).astype(np.int32, copy=False)
        self.acac_ctrl_data = np.concatenate(ctrl_data).astype(np.float64, copy=False)

    def _cache_static_numeric_arrays(self, *, defer_full_pattern: bool = False):
        """Cache numeric arrays used by Newton iterations without object-model access."""
        self._cache_node_type_masks()
        # 预填已知节点的相角和电压幅值设定，迭代时无需重复清零与回填
        self._state_theta.fill(0.0)
        self._state_voltage.fill(1.0)
        if np.any(self._slack_mask):
            self._state_theta[self._slack_mask] = self.theta_spec[self._slack_mask]
        if np.any(self._fixed_voltage_mask):
            self._state_voltage[self._fixed_voltage_mask] = self.V_spec[self._fixed_voltage_mask]
        if len(self.load_info):
            load_data = np.asarray(self.load_info, dtype=np.float64)
            self.load_pos = load_data[:, 0].astype(np.int32)
            self.load_pv0 = load_data[:, 1]
            self.load_pv1 = load_data[:, 2]
            self.load_pv2 = load_data[:, 3]
            self.load_qv0 = load_data[:, 4]
            self.load_qv1 = load_data[:, 5]
            self.load_qv2 = load_data[:, 6]
        else:
            self.load_pos = np.array([], dtype=np.int32)
            self.load_pv0 = self.load_pv1 = self.load_pv2 = np.array([], dtype=np.float64)
            self.load_qv0 = self.load_qv1 = self.load_qv2 = np.array([], dtype=np.float64)
        self._has_voltage_dependent_load = bool(
            self.load_pos.size
            and (
                np.any(self.load_pv1 != 0.0)
                or np.any(self.load_pv2 != 0.0)
                or np.any(self.load_qv1 != 0.0)
                or np.any(self.load_qv2 != 0.0)
            )
        )
        if not self._has_voltage_dependent_load:
            if self.load_pos.size:
                self._const_p_load = np.bincount(self.load_pos, weights=self.load_pv0, minlength=self.N)
                self._const_q_load = np.bincount(self.load_pos, weights=self.load_qv0, minlength=self.N)
            else:
                self._const_p_load = np.zeros(self.N, dtype=np.float64)
                self._const_q_load = np.zeros(self.N, dtype=np.float64)
        else:
            self._const_p_load = np.array([], dtype=np.float64)
            self._const_q_load = np.array([], dtype=np.float64)
        self._load_vm_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._load_value_work = np.empty(self.load_pos.size, dtype=np.float64)
        self._load_aux_work = np.empty(self.load_pos.size, dtype=np.float64)
        if self.N_phi > 0 and len(self.zero_branch_info) > 0:
            self.zero_idx = self.zero_branch_info[:, 0].astype(np.int32)
            self.zero_type = self.zero_branch_info[:, 1].astype(np.int32)
            self.zero_a = self.zero_branch_info[:, 2].astype(np.int32)
            self.zero_b = self.zero_branch_info[:, 3].astype(np.int32)
            self.zero_phi_a = self.zero_branch_info[:, 4].astype(np.int32)
            self.zero_phi_b = self.zero_branch_info[:, 5].astype(np.int32)
        else:
            self.zero_idx = self.zero_type = self.zero_a = self.zero_b = np.array([], dtype=np.int32)
            self.zero_phi_a = self.zero_phi_b = np.array([], dtype=np.int32)
        self.pq_theta_rows = self.theta_idx[self.V_unknown].astype(np.int32, copy=True)
        self.pq_v_cols = np.arange(self.V_unknown.size, dtype=np.int32)
        self.theta_col_by_node = np.full(self.N, -1, dtype=np.int32)
        self.v_col_by_node = np.full(self.N, -1, dtype=np.int32)
        self.p_row_by_node = np.full(self.N, -1, dtype=np.int32)
        self.q_row_by_node = np.full(self.N, -1, dtype=np.int32)
        self.theta_col_by_node[self.theta_unknown] = np.arange(self.n_theta, dtype=np.int32)
        self.p_row_by_node[self.theta_unknown] = self.theta_col_by_node[self.theta_unknown]
        self.v_col_by_node[self.V_unknown] = self.n_theta + np.arange(self.n_V, dtype=np.int32)
        self.q_row_by_node[self.V_unknown] = self.n_theta + np.arange(self.n_V, dtype=np.int32)
        y_csr = self.Y.tocsr()
        self.Y_jac_rows = np.repeat(np.arange(self.N, dtype=np.int32), np.diff(y_csr.indptr))
        self.Y_jac_cols = y_csr.indices.astype(np.int32, copy=True)
        self.Y_jac_data = y_csr.data.astype(np.complex128, copy=True)
        self.Y_jac_conj = np.conj(self.Y_jac_data)
        self.Y_jac_g = self.Y_jac_data.real
        self.Y_jac_b = self.Y_jac_data.imag
        self.Y_jac_diag = y_csr.diagonal().astype(np.complex128, copy=False)
        self.Y_jac_diag_idx = np.flatnonzero(self.Y_jac_rows == self.Y_jac_cols)
        self.Y_jac_diag_nodes = self.Y_jac_rows[self.Y_jac_diag_idx]
        self.Y_jac_diag_g = self.Y_jac_diag[self.Y_jac_diag_nodes].real
        self.Y_jac_diag_b = self.Y_jac_diag[self.Y_jac_diag_nodes].imag
        self._cache_standard_jacobian_pattern(build_csc=not defer_full_pattern)
        self._cache_zero_jacobian_pattern()
        if defer_full_pattern:
            self._full_jac_raw_rows = np.array([], dtype=np.int32)
            self._full_jac_raw_cols = np.array([], dtype=np.int32)
            self.full_jac_raw_data = np.array([], dtype=np.float64)
            self.full_jac_raw_to_csc_pos = np.array([], dtype=np.intp)
            self.full_jac_csc_indices = np.array([], dtype=np.int32)
            self.full_jac_csc_indptr = np.zeros(self.total_vars + 1, dtype=np.int32)
            self.full_jac_csc_data = np.array([], dtype=np.float64)
            self._full_jac_csc_matrix = None
            self.full_jac_csc_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csc_pos, 0)
            self.full_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
            self.full_jac_csr_indices = np.array([], dtype=np.int32)
            self.full_jac_csr_indptr = np.zeros(self.total_eq + 1, dtype=np.int32)
            self.full_jac_csr_data = np.array([], dtype=np.float64)
            self.full_jac_csr_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csr_pos, 0)
        else:
            self._cache_full_jacobian_pattern()
        self.Y_diag = np.array([], dtype=np.complex128)
        self.Y_offdiag_indices = []
        self.Y_offdiag_data = []

    def _cache_standard_jacobian_pattern(self, *, build_csc: bool = True):
        """Cache fixed COO coordinates for the standard P/Q Jacobian."""
        self.standard_jac_rows = np.array([], dtype=np.int32)
        self.standard_jac_cols = np.array([], dtype=np.int32)
        self.standard_jac_data = np.array([], dtype=np.float64)
        self.std_jac_p_theta_idx = self.std_jac_p_vm_idx = np.array([], dtype=np.intp)
        self.std_jac_q_theta_idx = self.std_jac_q_vm_idx = np.array([], dtype=np.intp)
        self.std_jac_p_theta_slice = self.std_jac_p_vm_slice = slice(0, 0)
        self.std_jac_q_theta_slice = self.std_jac_q_vm_slice = slice(0, 0)
        self.std_jac_load_p_slice = self.std_jac_load_q_slice = slice(0, 0)
        self.standard_jac_csr_indices = np.array([], dtype=np.int32)
        self.standard_jac_csr_indptr = np.array([], dtype=np.int32)
        self.standard_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
        self.standard_jac_csr_data = np.array([], dtype=np.float64)
        self.standard_jac_csr_sum_plan = build_raw_sum_plan(self.standard_jac_raw_to_csr_pos, 0)
        self.standard_jac_raw_to_csc_pos = np.array([], dtype=np.intp)
        self.standard_jac_csc_indices = np.array([], dtype=np.int32)
        self.standard_jac_csc_indptr = np.array([], dtype=np.int32)
        self.standard_jac_csc_data = np.array([], dtype=np.float64)
        self._standard_jac_csc_matrix = None
        self.standard_jac_csc_sum_plan = build_raw_sum_plan(self.standard_jac_raw_to_csc_pos, 0)
        self.std_jac_load_nodes = np.array([], dtype=np.int32)
        self.std_jac_load_extra_nodes = np.array([], dtype=np.int32)
        self.std_jac_load_p_pos = np.array([], dtype=np.intp)
        self.std_jac_load_q_pos = np.array([], dtype=np.intp)

        if not self.Y_jac_rows.size:
            return

        p_rows = self.p_row_by_node[self.Y_jac_rows]
        q_rows = self.q_row_by_node[self.Y_jac_rows]
        theta_cols = self.theta_col_by_node[self.Y_jac_cols]
        v_cols = self.v_col_by_node[self.Y_jac_cols]

        self.std_jac_p_theta_idx = np.flatnonzero((p_rows >= 0) & (theta_cols >= 0))
        self.std_jac_p_vm_idx = np.flatnonzero((p_rows >= 0) & (v_cols >= 0))
        self.std_jac_q_theta_idx = np.flatnonzero((q_rows >= 0) & (theta_cols >= 0))
        self.std_jac_q_vm_idx = np.flatnonzero((q_rows >= 0) & (v_cols >= 0))

        rows_parts = [
            p_rows[self.std_jac_p_theta_idx],
            p_rows[self.std_jac_p_vm_idx],
            q_rows[self.std_jac_q_theta_idx],
            q_rows[self.std_jac_q_vm_idx],
        ]
        cols_parts = [
            theta_cols[self.std_jac_p_theta_idx],
            v_cols[self.std_jac_p_vm_idx],
            theta_cols[self.std_jac_q_theta_idx],
            v_cols[self.std_jac_q_vm_idx],
        ]
        cursor = 0
        self.std_jac_p_theta_slice = slice(cursor, cursor + self.std_jac_p_theta_idx.size)
        cursor = self.std_jac_p_theta_slice.stop
        self.std_jac_p_vm_slice = slice(cursor, cursor + self.std_jac_p_vm_idx.size)
        cursor = self.std_jac_p_vm_slice.stop
        self.std_jac_q_theta_slice = slice(cursor, cursor + self.std_jac_q_theta_idx.size)
        cursor = self.std_jac_q_theta_slice.stop
        self.std_jac_q_vm_slice = slice(cursor, cursor + self.std_jac_q_vm_idx.size)
        cursor = self.std_jac_q_vm_slice.stop

        if self.V_unknown.size:
            diag_idx_by_node = np.full(self.N, -1, dtype=np.intp)
            diag_idx_by_node[self.Y_jac_diag_nodes] = self.Y_jac_diag_idx
            load_diag_idx = diag_idx_by_node[self.V_unknown]
            yidx_to_p_vm_pos = np.full(self.Y_jac_rows.size, -1, dtype=np.intp)
            yidx_to_q_vm_pos = np.full(self.Y_jac_rows.size, -1, dtype=np.intp)
            yidx_to_p_vm_pos[self.std_jac_p_vm_idx] = np.arange(self.std_jac_p_vm_idx.size, dtype=np.intp)
            yidx_to_q_vm_pos[self.std_jac_q_vm_idx] = np.arange(self.std_jac_q_vm_idx.size, dtype=np.intp)
            p_local_pos = np.where(load_diag_idx >= 0, yidx_to_p_vm_pos[load_diag_idx], -1)
            q_local_pos = np.where(load_diag_idx >= 0, yidx_to_q_vm_pos[load_diag_idx], -1)
            direct_mask = (p_local_pos >= 0) & (q_local_pos >= 0)
            self.std_jac_load_nodes = self.V_unknown[direct_mask]
            self.std_jac_load_p_pos = self.std_jac_p_vm_slice.start + p_local_pos[direct_mask]
            self.std_jac_load_q_pos = self.std_jac_q_vm_slice.start + q_local_pos[direct_mask]

            self.std_jac_load_extra_nodes = self.V_unknown[~direct_mask]
            if self.std_jac_load_extra_nodes.size:
                v_load_cols = self.v_col_by_node[self.std_jac_load_extra_nodes]
                rows_parts.extend((self.p_row_by_node[self.std_jac_load_extra_nodes], self.q_row_by_node[self.std_jac_load_extra_nodes]))
                cols_parts.extend((v_load_cols, v_load_cols))
                self.std_jac_load_p_slice = slice(cursor, cursor + self.std_jac_load_extra_nodes.size)
                cursor = self.std_jac_load_p_slice.stop
                self.std_jac_load_q_slice = slice(cursor, cursor + self.std_jac_load_extra_nodes.size)
                cursor = self.std_jac_load_q_slice.stop

        self.standard_jac_rows = np.concatenate(rows_parts).astype(np.int32, copy=False)
        self.standard_jac_cols = np.concatenate(cols_parts).astype(np.int32, copy=False)
        self.standard_jac_data = np.empty(cursor, dtype=np.float64)
        if self.std_jac_load_p_slice.stop > self.std_jac_load_p_slice.start:
            self.standard_jac_data[self.std_jac_load_p_slice] = 0.0
            self.standard_jac_data[self.std_jac_load_q_slice] = 0.0
        if cursor:
            n_dim = self.n_theta + self.n_V
            cache_standard_csc = (
                build_csc
                and self.N_phi == 0
                and self.N_acac == 0
                and not getattr(self, "_defer_csc_pattern", False)
            )
            # CSR 模式统一懒构建；混合耦合模式下 CSC 模式亦按需懒构建
            if cache_standard_csc:
                (
                    self.standard_jac_csc_indices,
                    self.standard_jac_csc_indptr,
                    self.standard_jac_raw_to_csc_pos,
                ) = build_compressed_pattern_from_raw_coords(self.standard_jac_cols, self.standard_jac_rows, n_dim)
                self.standard_jac_csc_data = np.empty(self.standard_jac_csc_indices.size, dtype=np.float64)
                self.standard_jac_csc_sum_plan = build_raw_sum_plan(
                    self.standard_jac_raw_to_csc_pos,
                    self.standard_jac_csc_data.size,
                )
            y_nnz = self.Y_jac_rows.size
            self._jac_w_work = np.empty(y_nnz, dtype=np.complex128)
            self._jac_tmp = np.empty(y_nnz, dtype=np.float64)

    def _ensure_standard_csc_pattern(self):
        """按需懒构建标准 P/Q 雅可比的 CSC 压缩模式并缓存。"""
        if self.standard_jac_raw_to_csc_pos.size or not self.standard_jac_rows.size:
            return
        n_dim = self.n_theta + self.n_V
        (
            self.standard_jac_csc_indices,
            self.standard_jac_csc_indptr,
            self.standard_jac_raw_to_csc_pos,
        ) = build_compressed_pattern_from_raw_coords(self.standard_jac_cols, self.standard_jac_rows, n_dim)
        self.standard_jac_csc_data = np.empty(self.standard_jac_csc_indices.size, dtype=np.float64)
        self.standard_jac_csc_sum_plan = build_raw_sum_plan(
            self.standard_jac_raw_to_csc_pos,
            self.standard_jac_csc_data.size,
        )

    def _ensure_standard_csr_pattern(self):
        """按需懒构建标准 P/Q 雅可比的 CSR 压缩模式并缓存。"""
        if self.standard_jac_raw_to_csr_pos.size or not self.standard_jac_rows.size:
            return
        n_dim = self.n_theta + self.n_V
        (
            self.standard_jac_csr_indices,
            self.standard_jac_csr_indptr,
            self.standard_jac_raw_to_csr_pos,
        ) = build_compressed_pattern_from_raw_coords(self.standard_jac_rows, self.standard_jac_cols, n_dim)
        self.standard_jac_csr_data = np.empty(self.standard_jac_csr_indices.size, dtype=np.float64)
        self.standard_jac_csr_sum_plan = build_raw_sum_plan(
            self.standard_jac_raw_to_csr_pos,
            self.standard_jac_csr_data.size,
        )

    @staticmethod
    def _group_indices_by_kind(kind, source_index, n_kind):
        """Group data positions and source indices once, avoiding per-iteration masks."""
        empty_groups = tuple(np.array([], dtype=np.intp) for _ in range(n_kind))
        if kind.size == 0:
            return empty_groups, empty_groups
        pos_groups = []
        index_groups = []
        for code in range(n_kind):
            pos = np.flatnonzero(kind == code).astype(np.intp, copy=False)
            pos_groups.append(pos)
            if pos.size:
                index_groups.append(source_index[pos].astype(np.intp, copy=False))
            else:
                index_groups.append(np.array([], dtype=np.intp))
        return tuple(pos_groups), tuple(index_groups)

    def _cache_zero_jacobian_runtime_arrays(self):
        """Cache runtime grouping and work arrays for zero-impedance Jacobian refresh."""
        self.zero_top_left_pos_by_kind, self.zero_top_left_edge_by_kind = self._group_indices_by_kind(
            self.zero_top_left_kind,
            self.zero_top_left_edge,
            8,
        )
        self.zero_top_right_pos_by_kind, self.zero_top_right_edge_by_kind = self._group_indices_by_kind(
            self.zero_top_right_kind,
            self.zero_top_right_edge,
            16,
        )
        self.zero_bottom_left_pos_by_kind, self.zero_bottom_left_node_by_kind = self._group_indices_by_kind(
            self.zero_bottom_left_kind,
            self.zero_bottom_left_node,
            8,
        )

        n_edge = int(self.zero_a.size)
        if n_edge == 0:
            self._zero_I_re = np.array([], dtype=np.float64)
            self._zero_I_im = np.array([], dtype=np.float64)
            self._zero_Va = np.array([], dtype=np.float64)
            self._zero_Vb = np.array([], dtype=np.float64)
            self._zero_cos_a = np.array([], dtype=np.float64)
            self._zero_sin_a = np.array([], dtype=np.float64)
            self._zero_cos_b = np.array([], dtype=np.float64)
            self._zero_sin_b = np.array([], dtype=np.float64)
            self._zero_a_pv = np.array([], dtype=np.float64)
            self._zero_a_pt = np.array([], dtype=np.float64)
            self._zero_b_pv = np.array([], dtype=np.float64)
            self._zero_b_pt = np.array([], dtype=np.float64)
            self._zero_va_cos = np.array([], dtype=np.float64)
            self._zero_va_sin = np.array([], dtype=np.float64)
            self._zero_vb_cos = np.array([], dtype=np.float64)
            self._zero_vb_sin = np.array([], dtype=np.float64)
            self._zero_tmp = np.array([], dtype=np.float64)
            return

        self._zero_I_re = np.empty(n_edge, dtype=np.float64)
        self._zero_I_im = np.empty(n_edge, dtype=np.float64)
        self._zero_Va = np.empty(n_edge, dtype=np.float64)
        self._zero_Vb = np.empty(n_edge, dtype=np.float64)
        self._zero_cos_a = np.empty(n_edge, dtype=np.float64)
        self._zero_sin_a = np.empty(n_edge, dtype=np.float64)
        self._zero_cos_b = np.empty(n_edge, dtype=np.float64)
        self._zero_sin_b = np.empty(n_edge, dtype=np.float64)
        self._zero_a_pv = np.empty(n_edge, dtype=np.float64)
        self._zero_a_pt = np.empty(n_edge, dtype=np.float64)
        self._zero_b_pv = np.empty(n_edge, dtype=np.float64)
        self._zero_b_pt = np.empty(n_edge, dtype=np.float64)
        self._zero_va_cos = np.empty(n_edge, dtype=np.float64)
        self._zero_va_sin = np.empty(n_edge, dtype=np.float64)
        self._zero_vb_cos = np.empty(n_edge, dtype=np.float64)
        self._zero_vb_sin = np.empty(n_edge, dtype=np.float64)
        self._zero_tmp = np.empty(n_edge, dtype=np.float64)

    def _cache_zero_jacobian_pattern(self):
        """Cache fixed zero-impedance Jacobian coordinates for Newton iterations."""
        self.zero_top_left_rows = np.array([], dtype=np.int32)
        self.zero_top_left_cols = np.array([], dtype=np.int32)
        self.zero_top_left_edge = np.array([], dtype=np.intp)
        self.zero_top_left_kind = np.array([], dtype=np.int8)
        self.zero_top_left_data = np.array([], dtype=np.float64)
        self.zero_top_right_rows = np.array([], dtype=np.int32)
        self.zero_top_right_cols = np.array([], dtype=np.int32)
        self.zero_top_right_edge = np.array([], dtype=np.intp)
        self.zero_top_right_kind = np.array([], dtype=np.int8)
        self.zero_top_right_data = np.array([], dtype=np.float64)
        self.zero_bottom_left_rows = np.array([], dtype=np.int32)
        self.zero_bottom_left_cols = np.array([], dtype=np.int32)
        self.zero_bottom_left_node = np.array([], dtype=np.int32)
        self.zero_bottom_left_kind = np.array([], dtype=np.int8)
        self.zero_bottom_left_data = np.array([], dtype=np.float64)
        self.zero_bottom_right_rows = np.array([], dtype=np.int32)
        self.zero_bottom_right_cols = np.array([], dtype=np.int32)
        self.zero_bottom_right_data = np.array([], dtype=np.float64)
        self._cache_zero_jacobian_runtime_arrays()

        if self.N_phi <= 0:
            return

        # Each edge contributes the same small stencil, filtered by terminal
        # bus type. Flatten in edge-major order to preserve the original raw
        # entry order (and therefore duplicate summation/rounding).
        a, b = self.zero_a, self.zero_b
        ta, tb = self.theta_idx[a], self.theta_idx[b]
        va, vb = self.n_theta + self.V_idx[a], self.n_theta + self.V_idx[b]
        ap, bp = self.node_type[a] != AC_NODE_TYPE_SLACK, self.node_type[b] != AC_NODE_TYPE_SLACK
        aq, bq = self.node_type[a] == AC_NODE_TYPE_PQ, self.node_type[b] == AC_NODE_TYPE_PQ

        mask = np.column_stack((ap, aq, aq, aq, bp, bq, bq, bq))
        edge, kind = np.nonzero(mask)
        self.zero_top_left_rows = np.column_stack((ta, ta, va, va, tb, tb, vb, vb))[mask].astype(np.int32)
        self.zero_top_left_cols = np.column_stack((ta, va, ta, va, tb, vb, tb, vb))[mask].astype(np.int32)
        self.zero_top_left_edge = edge
        self.zero_top_left_kind = kind.astype(np.int8)
        self.zero_top_left_data = np.empty(edge.size, dtype=np.float64)

        mask = np.column_stack((ap, aq, bp, bq))
        edge, group = np.nonzero(mask)
        phi_cols = np.column_stack((
            self.zero_phi_a, self.N_phi + self.zero_phi_a,
            self.zero_phi_b, self.N_phi + self.zero_phi_b,
        ))
        self.zero_top_right_rows = np.repeat(np.column_stack((ta, va, tb, vb))[mask], 4).astype(np.int32)
        self.zero_top_right_cols = phi_cols[edge].reshape(-1).astype(np.int32)
        self.zero_top_right_edge = np.repeat(edge, 4)
        self.zero_top_right_kind = (group[:, None] * 4 + np.arange(4)).reshape(-1).astype(np.int8)
        self.zero_top_right_data = np.empty(self.zero_top_right_rows.size, dtype=np.float64)

        # 直接使用向量化构造的生成树端点数组
        a = getattr(self, "_zero_tree_a", np.empty(0, dtype=np.int32))
        b = getattr(self, "_zero_tree_b", np.empty(0, dtype=np.int32))
        ta, tb = self.theta_idx[a], self.theta_idx[b]
        va, vb = self.n_theta + self.V_idx[a], self.n_theta + self.V_idx[b]
        mask4 = np.column_stack((
            self.node_type[a] != AC_NODE_TYPE_SLACK, self.node_type[a] == AC_NODE_TYPE_PQ,
            self.node_type[b] != AC_NODE_TYPE_SLACK, self.node_type[b] == AC_NODE_TYPE_PQ,
        ))
        mask = np.tile(mask4, (1, 2))
        edge, kind = np.nonzero(mask)
        self.zero_bottom_left_rows = (edge * 2 + kind // 4).astype(np.int32)
        self.zero_bottom_left_cols = np.tile(np.column_stack((ta, va, tb, vb)), (1, 2))[mask].astype(np.int32)
        self.zero_bottom_left_node = np.tile(np.column_stack((a, a, b, b)), (1, 2))[mask].astype(np.int32)
        self.zero_bottom_left_kind = kind.astype(np.int8)
        self.zero_bottom_left_data = np.empty(edge.size, dtype=np.float64)

        ref_phi = np.asarray(self.ref_phi_idx, dtype=np.int32)
        self.zero_bottom_right_rows = 2 * int(a.size) + np.arange(2 * ref_phi.size, dtype=np.int32)
        self.zero_bottom_right_cols = np.column_stack((ref_phi, self.N_phi + ref_phi)).reshape(-1)
        self.zero_bottom_right_data = np.ones(self.zero_bottom_right_rows.size, dtype=np.float64)
        self._cache_zero_jacobian_runtime_arrays()

    def _cache_full_jacobian_pattern(self):
        """Precompute the full AC Jacobian CSR pattern for standard and zero-branch blocks."""
        self._full_jac_raw_rows = np.array([], dtype=np.int32)
        self._full_jac_raw_cols = np.array([], dtype=np.int32)
        self.full_jac_raw_data = np.array([], dtype=np.float64)
        self.full_jac_raw_to_csr_pos = np.array([], dtype=np.intp)
        self.full_jac_csr_indices = np.array([], dtype=np.int32)
        self.full_jac_csr_indptr = np.zeros(self.total_eq + 1, dtype=np.int32)
        self.full_jac_csr_data = np.array([], dtype=np.float64)
        self.full_jac_csr_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csr_pos, 0)
        self.full_jac_raw_to_csc_pos = np.array([], dtype=np.intp)
        self.full_jac_csc_indices = np.array([], dtype=np.int32)
        self.full_jac_csc_indptr = np.zeros(self.total_vars + 1, dtype=np.int32)
        self.full_jac_csc_data = np.array([], dtype=np.float64)
        self._full_jac_csc_matrix = None
        self.full_jac_csc_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csc_pos, 0)
        self.full_jac_standard_slice = slice(0, 0)
        self.full_jac_zero_top_left_slice = slice(0, 0)
        self.full_jac_zero_top_right_slice = slice(0, 0)
        self.full_jac_zero_bottom_left_slice = slice(0, 0)
        self.full_jac_zero_bottom_right_slice = slice(0, 0)
        self.full_jac_acac_i_p_slice = slice(0, 0)
        self.full_jac_acac_i_q_slice = slice(0, 0)
        self.full_jac_acac_j_p_slice = slice(0, 0)
        self.full_jac_acac_j_q_slice = slice(0, 0)
        self.full_jac_acac_loss_slice = slice(0, 0)
        self.full_jac_acac_ctrl_slice = slice(0, 0)

        if self.N_phi == 0 and self.N_acac == 0:
            return
        if self.total_eq == 0 or self.total_vars == 0:
            return

        rows_parts = []
        cols_parts = []
        raw_count = 0

        def add_part(name, rows, cols):
            nonlocal raw_count
            rows = np.asarray(rows, dtype=np.int32)
            cols = np.asarray(cols, dtype=np.int32)
            if rows.size != cols.size:
                raise ValueError(f"AC Jacobian pattern part {name!r} has mismatched row/column lengths")
            part_slice = slice(raw_count, raw_count + rows.size)
            setattr(self, f"full_jac_{name}_slice", part_slice)
            raw_count += rows.size
            if rows.size:
                rows_parts.append(rows)
                cols_parts.append(cols)

        std_eq = self.n_theta + self.n_V
        std_vars = std_eq
        add_part("standard", self.standard_jac_rows, self.standard_jac_cols)
        if self.N_phi:
            add_part("zero_top_left", self.zero_top_left_rows, self.zero_top_left_cols)
            add_part("zero_top_right", self.zero_top_right_rows, self.zero_top_right_cols + std_vars)
            add_part("zero_bottom_left", self.zero_bottom_left_rows + std_eq, self.zero_bottom_left_cols)
            add_part("zero_bottom_right", self.zero_bottom_right_rows + std_eq, self.zero_bottom_right_cols + std_vars)
        if self.N_acac:
            add_part(
                "acac_i_p",
                self.acac_i_p_row[self.acac_i_p_eq_mask],
                self.acac_i_p_col[self.acac_i_p_eq_mask],
            )
            add_part(
                "acac_i_q",
                self.acac_i_q_row[self.acac_i_q_eq_mask],
                self.acac_i_q_col[self.acac_i_q_eq_mask],
            )
            add_part(
                "acac_j_p",
                self.acac_j_p_row[self.acac_j_p_eq_mask],
                self.acac_j_p_col[self.acac_j_p_eq_mask],
            )
            add_part(
                "acac_j_q",
                self.acac_j_q_row[self.acac_j_q_eq_mask],
                self.acac_j_q_col[self.acac_j_q_eq_mask],
            )
            add_part("acac_loss", self.acac_loss_rows, self.acac_loss_cols)
            add_part("acac_ctrl", self.acac_ctrl_rows, self.acac_ctrl_cols)

        self.full_jac_raw_data = np.empty(raw_count, dtype=np.float64)
        if raw_count == 0:
            return

        raw_rows = np.concatenate(rows_parts)
        raw_cols = np.concatenate(cols_parts)
        self._full_jac_raw_rows = raw_rows
        self._full_jac_raw_cols = raw_cols
        # 独立求解时预构建 CSC 模式；混合耦合模式（_defer_csc_pattern=True）直接向混合层输出 raw COO
        if not getattr(self, "_defer_csc_pattern", False):
            (
                self.full_jac_csc_indices,
                self.full_jac_csc_indptr,
                self.full_jac_raw_to_csc_pos,
            ) = build_compressed_pattern_from_raw_coords(raw_cols, raw_rows, self.total_vars)
            self.full_jac_csc_data = np.empty(self.full_jac_csc_indices.size, dtype=np.float64)
            self.full_jac_csc_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csc_pos, self.full_jac_csc_data.size)

    def _ensure_full_csc_pattern(self):
        """按需懒构建完整 AC 雅可比的 CSC 压缩模式并缓存。"""
        if self.full_jac_raw_to_csc_pos.size or not self._full_jac_raw_rows.size:
            return
        (
            self.full_jac_csc_indices,
            self.full_jac_csc_indptr,
            self.full_jac_raw_to_csc_pos,
        ) = build_compressed_pattern_from_raw_coords(self._full_jac_raw_cols, self._full_jac_raw_rows, self.total_vars)
        self.full_jac_csc_data = np.empty(self.full_jac_csc_indices.size, dtype=np.float64)
        self.full_jac_csc_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csc_pos, self.full_jac_csc_data.size)

    def _ensure_full_csr_pattern(self):
        """按需懒构建完整 AC 雅可比的 CSR 压缩模式并缓存。"""
        if self.full_jac_raw_to_csr_pos.size or not self._full_jac_raw_rows.size:
            return
        (
            self.full_jac_csr_indices,
            self.full_jac_csr_indptr,
            self.full_jac_raw_to_csr_pos,
        ) = _build_csr_pattern_from_raw_coords(self._full_jac_raw_rows, self._full_jac_raw_cols, self.total_eq)
        self.full_jac_csr_data = np.empty(self.full_jac_csr_indices.size, dtype=np.float64)
        self.full_jac_csr_sum_plan = build_raw_sum_plan(self.full_jac_raw_to_csr_pos, self.full_jac_csr_data.size)

    # --------------------------------------------------------------------------
    # 预处理阶段（精简版）
    # --------------------------------------------------------------------------
    def prepare(self) -> None:
        """预处理并确保拓扑/PPC 就绪。

        调用方无需先对 ACPowerNetwork 显式执行 `topo()`；prepare() 会基于
        当前 PPC 数据自动补齐所需的拓扑信息。
        """
        self._prepare_from_ppc()

    def _prepare_zero_branch_components(self):
        """基于稀疏连通分量分析与无环快路径建立零阻抗连通块、生成树约束和 phi 映射。"""
        idx_arr = getattr(self, "_zero_edge_idx_arr", None)
        if idx_arr is None or (len(self.zero_edges) > 0 and idx_arr.size != len(self.zero_edges)):
            if self.zero_edges:
                arr = np.asarray(self.zero_edges, dtype=np.int32)
                idx_arr = arr[:, 0]
                type_arr = arr[:, 1]
                a_arr = arr[:, 2]
                b_arr = arr[:, 3]
            else:
                empty_i32 = np.empty(0, dtype=np.int32)
                idx_arr = type_arr = a_arr = b_arr = empty_i32
            self._zero_edge_idx_arr = idx_arr
            self._zero_edge_type_arr = type_arr
            self._zero_edge_a_arr = a_arr
            self._zero_edge_b_arr = b_arr
        else:
            type_arr = self._zero_edge_type_arr
            a_arr = self._zero_edge_a_arr
            b_arr = self._zero_edge_b_arr

        if idx_arr.size == 0:
            empty_i32 = np.empty(0, dtype=np.int32)
            self.comp_nodes = []
            self.comp_tree_edges = []
            self.phi_node = []
            self.phi_comp = []
            self.ref_phi_idx = []
            self.zero_branch_info = np.zeros((0, 6))
            self._zero_tree_a = empty_i32
            self._zero_tree_b = empty_i32
            self._n_zero_comp = 0
            self.N_phi = 0
            return

        comp_res = build_zero_branch_components_fast(
            a_arr,
            b_arr,
            build_lists=self.keep_node_objects,
        )
        self.comp_nodes = comp_res["comp_nodes"]
        self.comp_tree_edges = comp_res["comp_tree_edges"]
        self._n_zero_comp = int(comp_res["n_comp"])
        self.N_phi = int(comp_res["phi_node"].size)
        self.ref_phi_idx = comp_res["ref_phi_idx"]
        tree_idx = comp_res["tree_edge_indices"]
        self._zero_tree_a = a_arr[tree_idx] if tree_idx.size else np.empty(0, dtype=np.int32)
        self._zero_tree_b = b_arr[tree_idx] if tree_idx.size else np.empty(0, dtype=np.int32)
        if self.keep_node_objects:
            self.phi_node = comp_res["phi_node"].tolist()
            self.phi_comp = comp_res["phi_comp"].tolist()
        else:
            self.phi_node = comp_res["phi_node"]
            self.phi_comp = comp_res["phi_comp"]

        self.zero_branch_info = np.column_stack(
            (
                idx_arr,
                type_arr,
                a_arr,
                b_arr,
                comp_res["phi_a"],
                comp_res["phi_b"],
            )
        ).astype(np.float64, copy=False)

    def _is_redundant_slack_zero_edge(self, a: int, b: int) -> bool:
        """Skip ideal ties between equal fixed-voltage slack nodes in Newton equations.

        Such ties are still part of topology, but their voltage equality is already
        enforced by the fixed slack phasors. Adding zero-current phi variables for
        them creates singular rows at flat start without changing the solved state.
        """
        if self.node_type.size == 0:
            return False
        if self.node_type[a] != AC_NODE_TYPE_SLACK or self.node_type[b] != AC_NODE_TYPE_SLACK:
            return False
        if abs(self.V_spec[a] - self.V_spec[b]) > 1e-10:
            return False
        if abs(self.theta_spec[a] - self.theta_spec[b]) > 1e-10:
            return False
        return True

    def get_f(self, x: np.ndarray) -> np.ndarray:
        """计算残差向量 F"""
        theta, V, phi_re, phi_im = self._extract_state_vars(x)
        dP, dQ = self._calc_power_balance(theta, V, phi_re, phi_im)
        return self._fill_residual(theta, V, phi_re, phi_im, dP, dQ)

    def _acac_state_view(self, x=None):
        if self.N_acac == 0:
            return np.empty((0, 4), dtype=np.float64)
        source = self._state_x_obj if x is None else x
        return source[self.acac_start:self.total_vars].reshape(self.N_acac, 4)

    def _append_acac_residuals(self, ac_f, acac, voltage, out):
        i_p = acac[:, 0]
        i_q = acac[:, 1]
        j_p = acac[:, 2]
        j_q = acac[:, 3]
        for rows, mask, values in (
            (self.acac_i_p_row, self.acac_i_p_eq_mask, i_p),
            (self.acac_i_q_row, self.acac_i_q_eq_mask, i_q),
            (self.acac_j_p_row, self.acac_j_p_eq_mask, j_p),
            (self.acac_j_q_row, self.acac_j_q_eq_mask, j_q),
        ):
            if np.any(mask):
                ac_f += np.bincount(rows[mask], weights=values[mask], minlength=ac_f.size)

        vi = voltage[self.acac_i_pos]
        vj = voltage[self.acac_j_pos]
        vi2 = vi * vi
        vj2 = vj * vj
        out[0::4] = (
            vi2 * vj2 * (i_p + j_p)
            - self.acac_r1 * (i_p * i_p + i_q * i_q) * vj2
            - self.acac_r2 * (j_p * j_p + j_q * j_q) * vi2
        )
        out[1::4] = i_p - self.acac_p_set
        i_control = out[2::4]
        j_control = out[3::4]
        i_control[self.acac_q_i_mask] = i_q[self.acac_q_i_mask] - self.acac_i_q_set[self.acac_q_i_mask]
        j_control[self.acac_q_j_mask] = j_q[self.acac_q_j_mask] - self.acac_j_q_set[self.acac_q_j_mask]
        i_control[self.acac_i_v_rep_mask] = (
            vi[self.acac_i_v_rep_mask] - self.acac_i_v_avg_set[self.acac_i_v_rep_mask]
        )
        if np.any(self.acac_i_v_share_mask):
            selected = np.flatnonzero(self.acac_i_v_share_mask).astype(np.int32, copy=False)
            i_control[selected] = i_q[selected] - i_q[self.acac_i_v_ref_idx[selected]]
        j_control[self.acac_j_v_rep_mask] = (
            vj[self.acac_j_v_rep_mask] - self.acac_j_v_avg_set[self.acac_j_v_rep_mask]
        )
        if np.any(self.acac_j_v_share_mask):
            selected = np.flatnonzero(self.acac_j_v_share_mask).astype(np.int32, copy=False)
            j_control[selected] = j_q[selected] - j_q[self.acac_j_v_ref_idx[selected]]

    def _fill_residual(self, theta, V, phi_re, phi_im, dP, dQ) -> np.ndarray:
        """Fill the preallocated Newton residual using already computed balances."""
        F = self._residual_work
        F.fill(0.0)
        # 方程顺序必须与 get_jacobi() 保持一致：P、Q、零阻抗电压约束、phi参考。
        n_theta = self.n_theta
        n_V = self.n_V

        # 有功 / 无功不平衡量
        F[:n_theta] = dP[self.theta_unknown]
        F[n_theta:n_theta + n_V] = dQ[self.V_unknown]

        # 零阻抗树边的电压约束：cos/sin 块按 stride=2 一次写入，避免 Python 循环。
        a = self._zero_residual_a
        eq_idx = n_theta + n_V
        if a.size:
            cos_theta = self._cache['cos_theta']
            sin_theta = self._cache['sin_theta']
            b = self._zero_residual_b
            n = a.size
            F[eq_idx:eq_idx + 2 * n:2] = V[a] * cos_theta[a] - V[b] * cos_theta[b]
            F[eq_idx + 1:eq_idx + 2 * n:2] = V[a] * sin_theta[a] - V[b] * sin_theta[b]
            eq_idx += 2 * n

        # phi 参考固定
        ref_phi_arr = self._ref_phi_idx_arr
        if ref_phi_arr.size:
            span = 2 * ref_phi_arr.size
            F[eq_idx:eq_idx + span:2] = phi_re[ref_phi_arr]
            F[eq_idx + 1:eq_idx + span:2] = phi_im[ref_phi_arr]

        if self.N_acac:
            self._append_acac_residuals(
                F[:self.ac_core_eq],
                self._acac_state_view(),
                V,
                F[self.acac_eq_start:self.total_eq],
            )

        return F

    def _calc_load_power_derivatives(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dP = self._load_dp_work
        dQ = self._load_dq_work
        if not self._has_voltage_dependent_load:
            dP.fill(0.0)
            dQ.fill(0.0)
            return dP, dQ
        dP.fill(0.0)
        dQ.fill(0.0)
        if self.load_pos.size:
            vm = self._load_vm_work
            values = self._load_value_work
            np.take(V, self.load_pos, out=vm)
            np.multiply(vm, self.load_pv2, out=values)
            values *= 2.0
            values += self.load_pv1
            dP[:] = np.bincount(self.load_pos, weights=values, minlength=self.N)

            np.multiply(vm, self.load_qv2, out=values)
            values *= 2.0
            values += self.load_qv1
            dQ[:] = np.bincount(self.load_pos, weights=values, minlength=self.N)
        return dP, dQ

    def _get_standard_jacobi_sparse(self, V: np.ndarray, Sbus=None, *, matrix_format="csr", build_matrix=True):
        """标准P/Q方程对theta/V变量的稀疏雅可比，采用MATPOWER矩阵化公式。"""
        if self.Y_jac_rows.size:
            return self._get_standard_jacobi_direct(
                V,
                Sbus=Sbus,
                matrix_format=matrix_format,
                build_matrix=build_matrix,
            )

        Vc = self._cache['Vc']
        Ibus = self.Y.dot(Vc)
        Vnorm = np.divide(Vc, V, out=np.zeros_like(Vc), where=V != 0)

        # 等价于 MATPOWER dSbus_dV 公式，但直接缩放 Y 的列/行，避免构造
        # diag(V)、diag(V/|V|) 后再做稀疏矩阵乘法。
        dS_dVm = self.Y.multiply(Vnorm).conjugate().multiply(Vc[:, None]).tocsr()
        dS_dVm.setdiag(dS_dVm.diagonal() + np.conj(Ibus) * Vnorm)

        dS_dVa = self.Y.multiply(Vc).conjugate().multiply((-1j * Vc)[:, None]).tocsr()
        dS_dVa.setdiag(dS_dVa.diagonal() + 1j * Vc * np.conj(Ibus))

        dPload_dV, dQload_dV = self._calc_load_power_derivatives(V)
        pvpq = self.theta_unknown
        pq = self.V_unknown

        J11 = dS_dVa.real[pvpq, :][:, pvpq]
        J12 = dS_dVm.real[pvpq, :][:, pq]
        J21 = dS_dVa.imag[pq, :][:, pvpq]
        J22 = dS_dVm.imag[pq, :][:, pq]

        if pq.size:
            J12 = J12 + coo_matrix((dPload_dV[pq], (self.pq_theta_rows, self.pq_v_cols)), shape=(self.n_theta, self.n_V)).tocsr()
            J22 = J22 + diags(dQload_dV[pq], 0, shape=(pq.size, pq.size), format='csr')

        jac = vstack((hstack((J11, J12), format='csr'), hstack((J21, J22), format='csr')), format='csr')
        if matrix_format == "csc":
            jac = jac.tocsc()
        return jac if build_matrix else None

    def _fill_standard_jacobian_data(self, V: np.ndarray, Sbus=None):
        """基于节点复电压无三角函数刷新标准 P/Q 雅可比原始数据。"""
        Vc = self._cache['Vc']
        if Sbus is not None:
            pass
        elif self._power_x_obj is self._state_x_obj and self._last_Sbus is not None:
            Sbus = self._last_Sbus
        else:
            Ibus = self.Y.dot(Vc)
            Sbus = Vc * np.conj(Ibus)
            self._power_x_obj = self._state_x_obj
            self._last_Ibus = Ibus
            self._last_Sbus = Sbus
        P = Sbus.real
        Q = Sbus.imag

        i = self.Y_jac_rows
        j = self.Y_jac_cols
        # 利用已缓存的 Vc 与单位共轭复电压 U_conj = cos(theta) - j*sin(theta)：
        # W_ij = Vc[i] * conj(Y_ij) * U_conj[j] = Vi*(G*cos(d)+B*sin(d)) + j*Vi*(G*sin(d)-B*cos(d))
        # 完全消除对全部 nnz(Y) 非零元的 cos/sin 超越函数调用，且无除零风险。
        w = self._jac_w_work
        np.multiply(self.Y_jac_conj, self._state_u_conj[j], out=w)
        w *= Vc[i]
        w_re = w.real
        w_im = w.imag
        Vj = V[j]
        tmp = self._jac_tmp

        diag_idx = self.Y_jac_diag_idx
        if diag_idx.size:
            diag_nodes = self.Y_jac_diag_nodes
            Gii = self.Y_jac_diag_g
            Bii = self.Y_jac_diag_b
            Vdiag = V[diag_nodes]
            Vdiag2 = Vdiag * Vdiag

        data = self.standard_jac_data
        # dP/dtheta: 非对角 = Vj * Im(W_ij)，对角 = -Q_i - B_ii * V_i^2
        np.multiply(Vj, w_im, out=tmp)
        if diag_idx.size:
            tmp[diag_idx] = -Q[diag_nodes] - Bii * Vdiag2
        data[self.std_jac_p_theta_slice] = tmp[self.std_jac_p_theta_idx]

        # dQ/dtheta: 非对角 = -Vj * Re(W_ij)，对角 = P_i - G_ii * V_i^2
        np.multiply(Vj, w_re, out=tmp)
        tmp *= -1.0
        if diag_idx.size:
            tmp[diag_idx] = P[diag_nodes] - Gii * Vdiag2
        data[self.std_jac_q_theta_slice] = tmp[self.std_jac_q_theta_idx]

        # dP/dV: 非对角 = Re(W_ij)，对角 = P_i / V_i + G_ii * V_i
        tmp[:] = w_re
        if diag_idx.size:
            tmp[diag_idx] = np.divide(
                P[diag_nodes],
                Vdiag,
                out=np.zeros_like(Vdiag),
                where=np.abs(Vdiag) > 1e-12,
            ) + Gii * Vdiag
        data[self.std_jac_p_vm_slice] = tmp[self.std_jac_p_vm_idx]

        # dQ/dV: 非对角 = Im(W_ij)，对角 = Q_i / V_i - B_ii * V_i
        tmp[:] = w_im
        if diag_idx.size:
            tmp[diag_idx] = np.divide(
                Q[diag_nodes],
                Vdiag,
                out=np.zeros_like(Vdiag),
                where=np.abs(Vdiag) > 1e-12,
            ) - Bii * Vdiag
        data[self.std_jac_q_vm_slice] = tmp[self.std_jac_q_vm_idx]

        if self.V_unknown.size and self._has_voltage_dependent_load:
            dPload_dV, dQload_dV = self._calc_load_power_derivatives(V)
            if self.std_jac_load_nodes.size:
                data[self.std_jac_load_p_pos] += dPload_dV[self.std_jac_load_nodes]
                data[self.std_jac_load_q_pos] += dQload_dV[self.std_jac_load_nodes]
            if self.std_jac_load_p_slice.stop > self.std_jac_load_p_slice.start:
                data[self.std_jac_load_p_slice] = dPload_dV[self.std_jac_load_extra_nodes]
                data[self.std_jac_load_q_slice] = dQload_dV[self.std_jac_load_extra_nodes]

        return data

    def _get_standard_jacobi_direct(self, V: np.ndarray, Sbus=None, *, matrix_format="csr", build_matrix=True):
        """Build the standard P/Q Jacobian directly from Y nonzeros."""
        data = self._fill_standard_jacobian_data(V, Sbus=Sbus)
        if matrix_format == "raw":
            return data
        shape = (self.n_theta + self.n_V, self.n_theta + self.n_V)
        if matrix_format == "csc":
            self._ensure_standard_csc_pattern()
            if self.standard_jac_raw_to_csc_pos.size:
                apply_raw_sum_plan(self.standard_jac_csc_data, data, self.standard_jac_csc_sum_plan)
            if not build_matrix:
                return self.standard_jac_csc_data
            if not self.standard_jac_csc_indptr.size:
                return csc_matrix(shape, dtype=np.float64)
            if self._standard_jac_csc_matrix is None:
                self._standard_jac_csc_matrix = csc_matrix(
                    (self.standard_jac_csc_data, self.standard_jac_csc_indices, self.standard_jac_csc_indptr),
                    shape=shape,
                    copy=False,
                )
            return self._standard_jac_csc_matrix

        self._ensure_standard_csr_pattern()
        if self.standard_jac_raw_to_csr_pos.size:
            apply_raw_sum_plan(self.standard_jac_csr_data, data, self.standard_jac_csr_sum_plan)
            if not build_matrix:
                return self.standard_jac_csr_data
            return csr_matrix(
                (self.standard_jac_csr_data, self.standard_jac_csr_indices, self.standard_jac_csr_indptr),
                shape=shape,
                copy=False,
            )
        if not build_matrix:
            return self.standard_jac_csr_data
        return coo_matrix(
            (data, (self.standard_jac_rows, self.standard_jac_cols)),
            shape=shape,
        ).tocsr()

    def _refresh_zero_edge_jacobian_work(self, V, phi_re, phi_im, cos_theta, sin_theta):
        """Refresh zero-edge workspace arrays shared by top-left and top-right blocks."""
        if not self.zero_a.size:
            return
        I_re = self._zero_I_re
        I_im = self._zero_I_im
        tmp = self._zero_tmp

        np.take(phi_re, self.zero_phi_a, out=I_re)
        np.take(phi_re, self.zero_phi_b, out=tmp)
        I_re -= tmp
        np.take(phi_im, self.zero_phi_a, out=I_im)
        np.take(phi_im, self.zero_phi_b, out=tmp)
        I_im -= tmp

        Va = self._zero_Va
        Vb = self._zero_Vb
        cos_a = self._zero_cos_a
        sin_a = self._zero_sin_a
        cos_b = self._zero_cos_b
        sin_b = self._zero_sin_b
        np.take(V, self.zero_a, out=Va)
        np.take(V, self.zero_b, out=Vb)
        np.take(cos_theta, self.zero_a, out=cos_a)
        np.take(sin_theta, self.zero_a, out=sin_a)
        np.take(cos_theta, self.zero_b, out=cos_b)
        np.take(sin_theta, self.zero_b, out=sin_b)

        a_pv = self._zero_a_pv
        a_pt = self._zero_a_pt
        b_pv = self._zero_b_pv
        b_pt = self._zero_b_pt
        np.multiply(cos_a, I_re, out=a_pv)
        np.multiply(sin_a, I_im, out=tmp)
        a_pv += tmp
        np.multiply(sin_a, I_re, out=a_pt)
        a_pt *= -1.0
        np.multiply(cos_a, I_im, out=tmp)
        a_pt += tmp
        np.multiply(cos_b, I_re, out=b_pv)
        np.multiply(sin_b, I_im, out=tmp)
        b_pv += tmp
        np.multiply(sin_b, I_re, out=b_pt)
        np.multiply(cos_b, I_im, out=tmp)
        b_pt -= tmp

        np.multiply(Va, cos_a, out=self._zero_va_cos)
        np.multiply(Va, sin_a, out=self._zero_va_sin)
        np.multiply(Vb, cos_b, out=self._zero_vb_cos)
        np.multiply(Vb, sin_b, out=self._zero_vb_sin)

    def _fill_zero_top_jacobian_data(self, V, phi_re, phi_im, cos_theta, sin_theta):
        if not self.zero_a.size:
            return
        self._refresh_zero_edge_jacobian_work(V, phi_re, phi_im, cos_theta, sin_theta)
        Va = self._zero_Va
        Vb = self._zero_Vb
        a_pv = self._zero_a_pv
        a_pt = self._zero_a_pt
        b_pv = self._zero_b_pv
        b_pt = self._zero_b_pt

        if self.zero_top_left_data.size:
            data = self.zero_top_left_data
            pos = self.zero_top_left_pos_by_kind
            edge = self.zero_top_left_edge_by_kind
            if pos[0].size:
                data[pos[0]] = Va[edge[0]] * a_pt[edge[0]]
            if pos[1].size:
                data[pos[1]] = a_pv[edge[1]]
            if pos[2].size:
                data[pos[2]] = Va[edge[2]] * a_pv[edge[2]]
            if pos[3].size:
                data[pos[3]] = -a_pt[edge[3]]
            if pos[4].size:
                data[pos[4]] = Vb[edge[4]] * b_pt[edge[4]]
            if pos[5].size:
                data[pos[5]] = -b_pv[edge[5]]
            if pos[6].size:
                data[pos[6]] = -Vb[edge[6]] * b_pv[edge[6]]
            if pos[7].size:
                data[pos[7]] = -b_pt[edge[7]]
        if self.zero_top_right_data.size:
            data = self.zero_top_right_data
            pos = self.zero_top_right_pos_by_kind
            edge = self.zero_top_right_edge_by_kind
            Va_cos = self._zero_va_cos
            Va_sin = self._zero_va_sin
            Vb_cos = self._zero_vb_cos
            Vb_sin = self._zero_vb_sin
            if pos[0].size:
                data[pos[0]] = Va_cos[edge[0]]
            if pos[1].size:
                data[pos[1]] = Va_sin[edge[1]]
            if pos[2].size:
                data[pos[2]] = -Va_cos[edge[2]]
            if pos[3].size:
                data[pos[3]] = -Va_sin[edge[3]]
            if pos[4].size:
                data[pos[4]] = Va_sin[edge[4]]
            if pos[5].size:
                data[pos[5]] = -Va_cos[edge[5]]
            if pos[6].size:
                data[pos[6]] = -Va_sin[edge[6]]
            if pos[7].size:
                data[pos[7]] = Va_cos[edge[7]]
            if pos[8].size:
                data[pos[8]] = -Vb_cos[edge[8]]
            if pos[9].size:
                data[pos[9]] = -Vb_sin[edge[9]]
            if pos[10].size:
                data[pos[10]] = Vb_cos[edge[10]]
            if pos[11].size:
                data[pos[11]] = Vb_sin[edge[11]]
            if pos[12].size:
                data[pos[12]] = -Vb_sin[edge[12]]
            if pos[13].size:
                data[pos[13]] = Vb_cos[edge[13]]
            if pos[14].size:
                data[pos[14]] = Vb_sin[edge[14]]
            if pos[15].size:
                data[pos[15]] = -Vb_cos[edge[15]]

    def _fill_zero_bottom_jacobian_data(self, V, cos_theta, sin_theta):
        if not self.zero_bottom_left_data.size:
            return
        data = self.zero_bottom_left_data
        pos = self.zero_bottom_left_pos_by_kind
        node = self.zero_bottom_left_node_by_kind
        if pos[0].size:
            data[pos[0]] = -V[node[0]] * sin_theta[node[0]]
        if pos[1].size:
            data[pos[1]] = cos_theta[node[1]]
        if pos[2].size:
            data[pos[2]] = V[node[2]] * sin_theta[node[2]]
        if pos[3].size:
            data[pos[3]] = -cos_theta[node[3]]
        if pos[4].size:
            data[pos[4]] = V[node[4]] * cos_theta[node[4]]
        if pos[5].size:
            data[pos[5]] = sin_theta[node[5]]
        if pos[6].size:
            data[pos[6]] = -V[node[6]] * cos_theta[node[6]]
        if pos[7].size:
            data[pos[7]] = -sin_theta[node[7]]

    @staticmethod
    def _slice_len(value):
        return int(value.stop - value.start)

    def _fill_acac_jacobian_data(self, raw, voltage):
        if self.N_acac == 0:
            return
        acac = self._acac_state_view()
        i_p = acac[:, 0]
        i_q = acac[:, 1]
        j_p = acac[:, 2]
        j_q = acac[:, 3]
        for part_slice in (
            self.full_jac_acac_i_p_slice,
            self.full_jac_acac_i_q_slice,
            self.full_jac_acac_j_p_slice,
            self.full_jac_acac_j_q_slice,
        ):
            if self._slice_len(part_slice):
                raw[part_slice] = 1.0
        if self._slice_len(self.full_jac_acac_ctrl_slice):
            raw[self.full_jac_acac_ctrl_slice] = self.acac_ctrl_data

        vi = voltage[self.acac_i_pos]
        vj = voltage[self.acac_j_pos]
        vi2 = vi * vi
        vj2 = vj * vj
        i_s2 = i_p * i_p + i_q * i_q
        j_s2 = j_p * j_p + j_q * j_q
        d_vi = 2.0 * vi * vj2 * (i_p + j_p) - 2.0 * self.acac_r2 * j_s2 * vi
        d_vj = 2.0 * vj * vi2 * (i_p + j_p) - 2.0 * self.acac_r1 * i_s2 * vj
        cursor = int(self.full_jac_acac_loss_slice.start)
        for values in (
            vi2 * vj2 - 2.0 * self.acac_r1 * i_p * vj2,
            -2.0 * self.acac_r1 * i_q * vj2,
            vi2 * vj2 - 2.0 * self.acac_r2 * j_p * vi2,
            -2.0 * self.acac_r2 * j_q * vi2,
            d_vi[self.acac_i_v_col >= 0],
            d_vj[self.acac_j_v_col >= 0],
        ):
            next_cursor = cursor + int(values.size)
            raw[cursor:next_cursor] = values
            cursor = next_cursor

    def get_jacobi(self, x: np.ndarray) -> csr_matrix:
        """计算雅可比矩阵。标准AC部分使用稀疏矩阵批量公式，零阻抗扩展按块追加。"""
        if self._state_x_obj is x:
            theta = self._cache['theta']
            V = self._cache['V']
            phi_re = x[self.base_phi_re:self.base_phi_re + self.N_phi] if self.N_phi > 0 else self._empty_phi
            phi_im = x[self.base_phi_im:self.base_phi_im + self.N_phi] if self.N_phi > 0 else self._empty_phi
        else:
            theta, V, phi_re, phi_im = self._extract_state_vars(x, update_cache=True)
        return self._get_jacobi_from_cached_state(theta, V, phi_re, phi_im)

    def _get_jacobi_from_precomputed_pattern(
        self,
        V,
        phi_re,
        phi_im,
        cos_theta,
        sin_theta,
        Sbus=None,
        *,
        matrix_format="csr",
        build_matrix=True,
    ):
        if self.full_jac_raw_data.size == 0:
            return None

        raw = self.full_jac_raw_data
        raw[self.full_jac_standard_slice] = self._fill_standard_jacobian_data(V, Sbus=Sbus)
        if self.N_phi:
            self._fill_zero_top_jacobian_data(V, phi_re, phi_im, cos_theta, sin_theta)
            self._fill_zero_bottom_jacobian_data(V, cos_theta, sin_theta)
            raw[self.full_jac_zero_top_left_slice] = self.zero_top_left_data
            raw[self.full_jac_zero_top_right_slice] = self.zero_top_right_data
            raw[self.full_jac_zero_bottom_left_slice] = self.zero_bottom_left_data
            raw[self.full_jac_zero_bottom_right_slice] = self.zero_bottom_right_data
        self._fill_acac_jacobian_data(raw, V)

        if matrix_format == "raw":
            return raw

        if matrix_format == "csc":
            self._ensure_full_csc_pattern()
            apply_raw_sum_plan(self.full_jac_csc_data, raw, self.full_jac_csc_sum_plan)
            if not build_matrix:
                return self.full_jac_csc_data
            # The sparsity arrays are immutable after prepare(); retain one
            # CSC wrapper and refresh only its data values each Newton step.
            # This avoids allocating a new sparse matrix object on every
            # iteration while preserving the existing numeric factorization
            # and solver fallback behavior.
            if self._full_jac_csc_matrix is None:
                self._full_jac_csc_matrix = csc_matrix(
                    (self.full_jac_csc_data, self.full_jac_csc_indices, self.full_jac_csc_indptr),
                    shape=(self.total_eq, self.total_vars),
                    copy=False,
                )
            return self._full_jac_csc_matrix
        self._ensure_full_csr_pattern()
        if not self.full_jac_raw_to_csr_pos.size:
            return None

        apply_raw_sum_plan(self.full_jac_csr_data, raw, self.full_jac_csr_sum_plan)
        if not build_matrix:
            return self.full_jac_csr_data
        return csr_matrix(
            (self.full_jac_csr_data, self.full_jac_csr_indices, self.full_jac_csr_indptr),
            shape=(self.total_eq, self.total_vars),
            copy=False,
        )

    def _get_jacobi_from_cached_state(
        self,
        theta,
        V,
        phi_re,
        phi_im,
        Sbus=None,
        *,
        matrix_format="csr",
        build_matrix=True,
    ) -> csr_matrix:
        """Build Jacobian using state arrays already extracted for the current Newton step."""
        cos_theta, sin_theta = self._cache['cos_theta'], self._cache['sin_theta']

        precomputed = self._get_jacobi_from_precomputed_pattern(
            V,
            phi_re,
            phi_im,
            cos_theta,
            sin_theta,
            Sbus=Sbus,
            matrix_format=matrix_format,
            build_matrix=build_matrix,
        )
        if precomputed is not None:
            return precomputed

        J_standard = self._get_standard_jacobi_sparse(
            V,
            Sbus=Sbus,
            matrix_format=matrix_format,
            build_matrix=build_matrix,
        )
        if self.N_phi == 0:
            return J_standard

        # 标准 P/Q 子块保持 CSR，不再转成 Python list 后重建整矩阵。
        # 零阻抗支路只生成三个小稀疏块，再与标准块拼接。
        std_eq = self.n_theta + self.n_V
        std_vars = std_eq
        zero_eq = self.total_eq - std_eq
        phi_vars = 2 * self.N_phi

        self._fill_zero_top_jacobian_data(V, phi_re, phi_im, cos_theta, sin_theta)
        if self.zero_top_left_rows.size:
            J_standard = J_standard + coo_matrix(
                (self.zero_top_left_data, (self.zero_top_left_rows, self.zero_top_left_cols)),
                shape=(std_eq, std_vars),
            ).tocsr()
        J_phi_top = coo_matrix(
            (self.zero_top_right_data, (self.zero_top_right_rows, self.zero_top_right_cols)),
            shape=(std_eq, phi_vars),
        ).tocsr()

        self._fill_zero_bottom_jacobian_data(V, cos_theta, sin_theta)
        J_zero_left = coo_matrix(
            (self.zero_bottom_left_data, (self.zero_bottom_left_rows, self.zero_bottom_left_cols)),
            shape=(zero_eq, std_vars),
        ).tocsr()
        J_zero_right = coo_matrix(
            (self.zero_bottom_right_data, (self.zero_bottom_right_rows, self.zero_bottom_right_cols)),
            shape=(zero_eq, phi_vars),
        ).tocsr()
        jac = vstack(
            (
                hstack((J_standard, J_phi_top), format='csr'),
                hstack((J_zero_left, J_zero_right), format='csr'),
            ),
            format='csr',
        )
        if matrix_format == "csc":
            jac = jac.tocsc()
        return jac if build_matrix else None

    def _build_newton_system(self, x: np.ndarray, *, return_jacobian=True, jacobian_format="csc"):
        """Compute residual and Jacobian together for one Newton iteration."""
        theta, V, phi_re, phi_im = self._extract_state_vars(x, update_cache=True)
        dP, dQ = self._calc_power_balance(theta, V, phi_re, phi_im)
        F = self._fill_residual(theta, V, phi_re, phi_im, dP, dQ)
        J = self._get_jacobi_from_cached_state(
            theta,
            V,
            phi_re,
            phi_im,
            Sbus=self._last_Sbus,
            matrix_format=jacobian_format,
            build_matrix=return_jacobian,
        )
        return F, J

    # --------------------------------------------------------------------------
    # 迭代求解
    # --------------------------------------------------------------------------
    def run(self, result_mode=None) -> int:
        """执行所选潮流算法。"""
        if result_mode is not None:
            self.result_mode = self._normalize_result_mode(result_mode)
            self._cache_csr_jacobian_pattern = False
            self.keep_node_objects = False
            self.node_list = []
            self.node_pos = {}
        if self.x.size == 0:
            self.prepare()
        return self._run_newton_raphson()

    def _run_newton_raphson(self) -> int:
        """执行牛顿-拉夫逊迭代求解"""
        self.converged = False
        self.iterations = 0
        self.failure_reason = ""
        x = self.x.copy()

        # 若本实例已把 KLU/UMFPACK 加入黑名单, 直接走 scipy, 避免重复触发失败路径。
        if self._instance_solver_blacklist and self._linear_solver_resolved in self._instance_solver_blacklist:
            self._linear_solver_resolved, self._linear_solver_fn = "scipy", spsolve
        resolved_name = self._linear_solver_resolved
        solver_fn = self._linear_solver_fn
        reusable_factorizer = None
        adaptive_factor = _AdaptiveNewtonFactor()
        n_vars = int(x.size)

        for it in range(self.max_iter):
            self.iterations += 1

            # 先计算状态与残差并判收敛，未收敛再视收缩情况复用 LU 或组装 Jacobian。
            theta, V, phi_re, phi_im = self._extract_state_vars(x, update_cache=True)
            dP, dQ = self._calc_power_balance(theta, V, phi_re, phi_im)
            F = self._fill_residual(theta, V, phi_re, phi_im, dP, dQ)
            self.normF = float(np.linalg.norm(F, np.inf)) if F.size else 0.0

            rollback_x = adaptive_factor.check_rollback(self.normF)
            if rollback_x is not None:
                x = rollback_x
                theta, V, phi_re, phi_im = self._extract_state_vars(x, update_cache=True)
                dP, dQ = self._calc_power_balance(theta, V, phi_re, phi_im)
                F = self._fill_residual(theta, V, phi_re, phi_im, dP, dQ)
                self.normF = float(np.linalg.norm(F, np.inf)) if F.size else 0.0

            if self.verbose:
                print(f"Iter {it + 1}: |F| = {self.normF:.2e}")

            # 收敛判断
            if self.normF < self.tol:
                if self.verbose:
                    print(f"收敛于第 {it + 1} 次迭代")
                self.converged = True
                self.x = x
                self._write_back()
                return 0

            # 发散 / 非有限残差提前终止，避免空跑后续迭代
            if not np.isfinite(self.normF) or self.normF > self.params.divergence_threshold:
                self.failure_reason = f"第 {it + 1} 次迭代残差发散 (|F|={self.normF:.2e})"
                if self.verbose:
                    print(self.failure_reason)
                self.x = x
                self._write_back()
                return -1

            if adaptive_factor.can_reuse(self.normF, n_vars):
                try:
                    delta = adaptive_factor.solve_reused(F, x, self.normF)
                    if np.all(np.isfinite(delta)):
                        x -= delta
                        continue
                except (RuntimeError, ValueError, ArithmeticError):
                    pass
                adaptive_factor.reset()

            J = self._get_jacobi_from_cached_state(
                theta,
                V,
                phi_re,
                phi_im,
                Sbus=self._last_Sbus,
                matrix_format="csc",
                build_matrix=True,
            )

            try:
                if reusable_factorizer is None:
                    reusable_factorizer = _make_reusable_factorizer(J, resolved_name)
                if reusable_factorizer is not None:
                    factor = reusable_factorizer.factor(J)
                else:
                    # 纯/近似对称 AC 雅可比在中高密度网格上使用 MMD_AT_PLUS_A 可大幅降低 SuperLU 填充元
                    ac_permc = (
                        "MMD_AT_PLUS_A"
                        if (
                            resolved_name in {"scipy", "superlu", "default"}
                            and self.N_acac == 0
                            and J.shape[0] >= 512
                            and J.nnz >= 8 * J.shape[0]
                            and 2 * self.N_phi <= max(64, int(0.05 * J.shape[0]))
                        )
                        else None
                    )
                    factor = _factor_jacobian(J, resolved_name, solver_fn, permc_spec=ac_permc)
                delta = factor.solve(F)
                adaptive_factor.update_factor(factor, self.normF)
            except (RuntimeError, ValueError, ArithmeticError) as exc:
                # 因子分解或解算失败时, 仅本实例回退到 SuperLU spsolve 单步路径,
                # 避免污染模块级缓存 (同进程内其他 calc 仍可继续尝试 KLU)。
                if resolved_name not in {"scipy", "superlu", "default"}:
                    self._instance_solver_blacklist.add(resolved_name)
                    if self.verbose:
                        print(f"[ac_lf] 可选稀疏求解器 {resolved_name!r} 失败，回退到 scipy: {exc}")
                reusable_factorizer = None
                adaptive_factor.reset()
                resolved_name, solver_fn = "scipy", spsolve
                self._linear_solver_resolved, self._linear_solver_fn = "scipy", spsolve
                delta = spsolve(J, F)
            # 方程定义为 F(x)=0，这里使用 x_new = x - J^{-1}F。
            x -= delta

        # 未收敛
        self.failure_reason = f"未在最大迭代次数 {self.max_iter} 内收敛 (|F|={self.normF:.2e})"
        if self.verbose:
            print(f"达到最大迭代次数 {self.max_iter}，未收敛")
        self.x = x
        self._write_back()
        return -1

    def _summary_node_ids(self):
        if self.ppc_node_idx.size == self.N:
            return self.ppc_node_idx.copy()
        if self.node_list:
            return np.asarray([int(getattr(node, "idx", pos)) for pos, node in enumerate(self.node_list)], dtype=np.int64)
        return np.arange(self.N, dtype=np.int64)

    def _write_summary_result(self):
        theta, V, _phi_re, _phi_im = self._extract_state_vars(self.x)
        self.result = {
            "node_id": self._summary_node_ids(),
            "voltage": V.copy(),
            "angle": theta.copy(),
            "summary": {
                "converged": bool(self.converged),
                "iterations": int(self.iterations),
                "normF": float(self.normF),
            },
        }
        self.lf_result = None

    def _write_back(self):
        """结果回填；数值计算批量完成，Python 循环只负责对象属性赋值。"""
        self._write_back_ppc()
        if self.result_mode == "none":
            self.result = {}
            self.lf_result = None
            return
        if self.result_mode == "summary":
            self._write_summary_result()
            return
        if self.result_mode != "array":
            self._write_ppc_result_to_network()
        if self.result_mode != "array" and not getattr(self, "skip_lf_result", False):
            self.lf_result = self._build_lf_result()
        else:
            self.lf_result = None
        return

    def _build_lf_result(self) -> ACLFResult:
        return self._build_lf_result_from_ppc()

    def _build_lf_result_from_ppc(self) -> ACLFResult:
        """基于 PPC 结果数组构建按需物化的 ACLFResult（全部字段保持 dict 兼容）。"""
        result = ACLFResult()
        result.arrays = dict(self.result)
        bus = result.arrays.get("bus")
        if bus is None or len(bus) == 0:
            return result

        ppc = self.ppc
        voltage_lookup_box = [None]

        def _get_voltage_lookup():
            fn = voltage_lookup_box[0]
            if fn is None:
                fn = build_result_voltage_lookup(bus, BUS_COLS["idx"], BUS_COLS["voltage"])
                voltage_lookup_box[0] = fn
            return fn

        bus_names_ref = ppc.get("bus_name")

        def _build_nodes():
            names = result_name_list(bus_names_ref, len(bus))
            return zip(
                names,
                starmap(
                    ACNodeResult,
                    zip(bus[:, BUS_COLS["voltage"]].tolist(), bus[:, BUS_COLS["angle"]].tolist()),
                ),
            )

        result.nodes = LazyResultDict(_build_nodes, len(bus))

        def _make_two_port_dict(rows, name_key, cols):
            if rows is None or len(rows) == 0:
                return {}
            raw_names = ppc.get(name_key)

            def _builder():
                lookup = _get_voltage_lookup()
                names = result_name_list(raw_names, len(rows))
                return zip(
                    names,
                    starmap(
                        ACTwoPortResult,
                        zip(
                            rows[:, cols["i_p"]].tolist(),
                            rows[:, cols["i_q"]].tolist(),
                            rows[:, cols["i_c"]].tolist(),
                            lookup(rows[:, cols["i_node"]]).tolist(),
                            rows[:, cols["j_p"]].tolist(),
                            rows[:, cols["j_q"]].tolist(),
                            rows[:, cols["j_c"]].tolist(),
                            lookup(rows[:, cols["j_node"]]).tolist(),
                        ),
                    ),
                )

            return LazyResultDict(_builder, len(rows))

        result.branches = _make_two_port_dict(result.arrays.get("branch"), "branch_name", BRANCH_COLS)
        result.transformers = _make_two_port_dict(result.arrays.get("transformer"), "transformer_name", TRANSFORMER_COLS)

        three_rows = result.arrays.get("three_winding_transformer")
        if three_rows is not None and len(three_rows):
            three_names_ref = ppc.get("three_winding_transformer_name")

            def _build_three_winding():
                cols3 = THREE_WINDING_TRANSFORMER_COLS
                lookup = _get_voltage_lookup()
                names = result_name_list(three_names_ref, len(three_rows))
                return zip(
                    names,
                    starmap(
                        ACThreePortResult,
                        zip(
                            three_rows[:, cols3["i_p"]].tolist(),
                            three_rows[:, cols3["i_q"]].tolist(),
                            three_rows[:, cols3["i_c"]].tolist(),
                            lookup(three_rows[:, cols3["i_node"]]).tolist(),
                            three_rows[:, cols3["j_p"]].tolist(),
                            three_rows[:, cols3["j_q"]].tolist(),
                            three_rows[:, cols3["j_c"]].tolist(),
                            lookup(three_rows[:, cols3["j_node"]]).tolist(),
                            three_rows[:, cols3["k_p"]].tolist(),
                            three_rows[:, cols3["k_q"]].tolist(),
                            three_rows[:, cols3["k_c"]].tolist(),
                            lookup(three_rows[:, cols3["k_node"]]).tolist(),
                        ),
                    ),
                )

            result.three_winding_transformers = LazyResultDict(_build_three_winding, len(three_rows))

        acac_rows = result.arrays.get("acac")
        if acac_rows is not None and len(acac_rows):
            acac_names_ref = ppc.get("acac_name")

            def _build_acac():
                lookup = _get_voltage_lookup()
                names = result_name_list(acac_names_ref, len(acac_rows))
                return zip(
                    names,
                    starmap(
                        ACTwoPortResult,
                        zip(
                            acac_rows[:, ACAC_COLS["i_p"]].tolist(),
                            acac_rows[:, ACAC_COLS["i_q"]].tolist(),
                            acac_rows[:, ACAC_COLS["i_i"]].tolist(),
                            lookup(acac_rows[:, ACAC_COLS["i_node"]]).tolist(),
                            acac_rows[:, ACAC_COLS["j_p"]].tolist(),
                            acac_rows[:, ACAC_COLS["j_q"]].tolist(),
                            acac_rows[:, ACAC_COLS["j_i"]].tolist(),
                            lookup(acac_rows[:, ACAC_COLS["j_node"]]).tolist(),
                        ),
                    ),
                )

            result.acac_converters = LazyResultDict(_build_acac, len(acac_rows))

        def _make_single_port_dict(rows, name_key, p_col, q_col, c_col, node_col, *, status_cols=None):
            if rows is None or len(rows) == 0:
                return {}
            raw_names = ppc.get(name_key)

            def _builder():
                lookup = _get_voltage_lookup()
                names = result_name_list(raw_names, len(rows))
                i_p = rows[:, p_col].tolist()
                i_q = rows[:, q_col].tolist()
                i_c = rows[:, c_col].tolist()
                i_v = lookup(rows[:, node_col]).tolist()
                if status_cols is None:
                    return zip(names, starmap(ACSinglePortResult, zip(i_p, i_q, i_c, i_v)))
                st_set = rows[:, status_cols[0]].astype(np.int64, copy=False).tolist()
                st_cur = rows[:, status_cols[1]].astype(np.int64, copy=False).tolist()
                return zip(names, starmap(ACSwitchResult, zip(i_p, i_q, i_c, i_v, st_set, st_cur)))

            return LazyResultDict(_builder, len(rows))

        result.zero_branches = _make_single_port_dict(
            result.arrays.get("zero_branch"),
            "zero_branch_name",
            ZERO_BRANCH_COLS["p"],
            ZERO_BRANCH_COLS["q"],
            ZERO_BRANCH_COLS["current"],
            ZERO_BRANCH_COLS["i_node"],
        )
        result.breakers = _make_single_port_dict(
            result.arrays.get("break"),
            "break_name",
            SWITCH_COLS["p"],
            SWITCH_COLS["q"],
            SWITCH_COLS["current"],
            SWITCH_COLS["i_node"],
            status_cols=(
                SWITCH_COLS["closed_status_set"],
                SWITCH_COLS["closed_status"],
            ),
        )
        result.generators = _make_single_port_dict(
            result.arrays.get("gen"),
            "gen_name",
            GEN_COLS["p"],
            GEN_COLS["q"],
            GEN_COLS["current"],
            GEN_COLS["node"],
        )
        result.loads = _make_single_port_dict(
            result.arrays.get("load"),
            "load_name",
            LOAD_COLS["p"],
            LOAD_COLS["q"],
            LOAD_COLS["current"],
            LOAD_COLS["node"],
        )
        return result

    def write_array_result_to_network(self, network=None) -> None:
        """Copy array results to an AC network without rebuilding solver state."""
        if network is None:
            network = getattr(self, "_network_writeback", None)
        if network is None or not self.result:
            return
        if getattr(network, "_lf_lightweight", False):
            network.result = self.result
            return

        topology = getattr(self, "_ppc_topology", None)

        def alive_by_idx(table_name, topology_key, columns):
            source_rows = np.asarray(
                self.ppc.get(
                    table_name,
                    np.zeros((0, max(columns.values()) + 1), dtype=np.float64),
                ),
                dtype=np.float64,
            )
            if topology is None:
                return {}
            if topology_key is None:
                alive_mask = np.asarray(topology.node_alive_mask, dtype=bool)
            else:
                device_topology = topology.devices.get(topology_key)
                if device_topology is None:
                    return {}
                alive_mask = np.asarray(device_topology.alive_mask, dtype=bool)
            if len(source_rows) != len(alive_mask):
                return {}
            return {
                int(row[columns["idx"]]): bool(alive)
                for row, alive in zip(source_rows, alive_mask)
            }

        def rows_by_idx(key, idx_col):
            rows = self.result.get(key)
            if rows is None or len(rows) == 0:
                return {}
            return {int(row[idx_col]): row for row in rows}

        bus_by_idx = rows_by_idx("bus", BUS_COLS["idx"])
        node_alive_by_idx = alive_by_idx("bus", None, BUS_COLS)
        for node in getattr(network, "nodes", []):
            row = bus_by_idx.get(int(getattr(node, "idx", -1)))
            if row is None:
                continue
            node.voltage = float(row[BUS_COLS["voltage"]])
            node.angle = float(row[BUS_COLS["angle"]])
            node.is_alive = node_alive_by_idx.get(int(row[BUS_COLS["idx"]]), False)

        for bus in getattr(network, "buses", []):
            members = [node for node in getattr(bus, "nodes", []) if getattr(node, "voltage", None) is not None]
            if not members:
                row = bus_by_idx.get(int(getattr(bus, "idx", -1)))
                if row is None:
                    continue
                bus.voltage = float(row[BUS_COLS["voltage"]])
                bus.angle = float(row[BUS_COLS["angle"]])
                continue
            bus.voltage = float(getattr(members[0], "voltage", 0.0) or 0.0)
            bus.angle = float(getattr(members[0], "angle", 0.0) or 0.0)

        def copy_fields(devices, row_map, fields, alive_map=None):
            if not row_map:
                return
            for dev in devices:
                row = row_map.get(int(getattr(dev, "idx", -1)))
                if row is None:
                    continue
                for attr, col in fields:
                    setattr(dev, attr, float(row[col]))
                if alive_map is not None:
                    dev.is_alive = alive_map.get(int(row[0]), False)

        copy_fields(
            getattr(network, "generators", []),
            rows_by_idx("gen", GEN_COLS["idx"]),
            (("p", GEN_COLS["p"]), ("q", GEN_COLS["q"]), ("current", GEN_COLS["current"])),
            alive_by_idx("gen", "gen", GEN_COLS),
        )
        copy_fields(
            getattr(network, "loads", []),
            rows_by_idx("load", LOAD_COLS["idx"]),
            (("p", LOAD_COLS["p"]), ("q", LOAD_COLS["q"]), ("current", LOAD_COLS["current"])),
            alive_by_idx("load", "load", LOAD_COLS),
        )
        copy_fields(
            getattr(network, "shunt_compensators", []),
            rows_by_idx("shunt", SHUNT_COLS["idx"]),
            (("p", SHUNT_COLS["p"]), ("q", SHUNT_COLS["q"]), ("current", SHUNT_COLS["current"])),
            alive_by_idx("shunt", "shunt", SHUNT_COLS),
        )
        copy_fields(
            getattr(network, "branches", []),
            rows_by_idx("branch", BRANCH_COLS["idx"]),
            (
                ("i_p", BRANCH_COLS["i_p"]),
                ("i_q", BRANCH_COLS["i_q"]),
                ("i_c", BRANCH_COLS["i_c"]),
                ("j_p", BRANCH_COLS["j_p"]),
                ("j_q", BRANCH_COLS["j_q"]),
                ("j_c", BRANCH_COLS["j_c"]),
            ),
            alive_by_idx("branch", "branch", BRANCH_COLS),
        )
        copy_fields(
            getattr(network, "transformers", []),
            rows_by_idx("transformer", TRANSFORMER_COLS["idx"]),
            (
                ("i_p", TRANSFORMER_COLS["i_p"]),
                ("i_q", TRANSFORMER_COLS["i_q"]),
                ("i_c", TRANSFORMER_COLS["i_c"]),
                ("j_p", TRANSFORMER_COLS["j_p"]),
                ("j_q", TRANSFORMER_COLS["j_q"]),
                ("j_c", TRANSFORMER_COLS["j_c"]),
            ),
            alive_by_idx("transformer", "transformer", TRANSFORMER_COLS),
        )
        copy_fields(
            getattr(network, "three_winding_transformers", []),
            rows_by_idx(
                "three_winding_transformer",
                THREE_WINDING_TRANSFORMER_COLS["idx"],
            ),
            tuple(
                (attr, THREE_WINDING_TRANSFORMER_COLS[attr])
                for attr in ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c", "k_p", "k_q", "k_c")
            ),
            alive_by_idx(
                "three_winding_transformer",
                "three_winding_transformer",
                THREE_WINDING_TRANSFORMER_COLS,
            ),
        )
        copy_fields(
            getattr(network, "zero_branches", []),
            rows_by_idx("zero_branch", ZERO_BRANCH_COLS["idx"]),
            (("p", ZERO_BRANCH_COLS["p"]), ("q", ZERO_BRANCH_COLS["q"]), ("current", ZERO_BRANCH_COLS["current"])),
            alive_by_idx("zero_branch", "zero_branch", ZERO_BRANCH_COLS),
        )
        copy_fields(
            getattr(network, "switches", []),
            rows_by_idx("switch", SWITCH_COLS["idx"]),
            (
                ("p", SWITCH_COLS["p"]),
                ("q", SWITCH_COLS["q"]),
                ("current", SWITCH_COLS["current"]),
                ("closed_status", SWITCH_COLS["closed_status"]),
            ),
            alive_by_idx("switch", "switch", SWITCH_COLS),
        )
        copy_fields(
            getattr(network, "breakers", []),
            rows_by_idx("break", SWITCH_COLS["idx"]),
            (
                ("p", SWITCH_COLS["p"]),
                ("q", SWITCH_COLS["q"]),
                ("current", SWITCH_COLS["current"]),
                ("closed_status", SWITCH_COLS["closed_status"]),
            ),
            alive_by_idx("break", "break", BREAK_COLS),
        )
        copy_fields(
            getattr(network, "acac_converters", []),
            rows_by_idx("acac", ACAC_COLS["idx"]),
            (
                ("i_p", ACAC_COLS["i_p"]),
                ("i_q", ACAC_COLS["i_q"]),
                ("i_i", ACAC_COLS["i_i"]),
                ("i_c", ACAC_COLS["i_i"]),
                ("j_p", ACAC_COLS["j_p"]),
                ("j_q", ACAC_COLS["j_q"]),
                ("j_i", ACAC_COLS["j_i"]),
                ("j_c", ACAC_COLS["j_i"]),
            ),
            alive_by_idx("acac", "acac", ACAC_COLS),
        )

    def _write_ppc_result_to_network(self) -> None:
        self.write_array_result_to_network()

    def _acac_node_power_injections(self, acac=None):
        p = np.zeros(self.N, dtype=np.float64)
        q = np.zeros(self.N, dtype=np.float64)
        if self.N_acac == 0:
            return p, q
        state = self._acac_state_view(self.x) if acac is None else acac
        p += np.bincount(self.acac_i_pos, weights=state[:, 0], minlength=self.N)
        q += np.bincount(self.acac_i_pos, weights=state[:, 1], minlength=self.N)
        p += np.bincount(self.acac_j_pos, weights=state[:, 2], minlength=self.N)
        q += np.bincount(self.acac_j_pos, weights=state[:, 3], minlength=self.N)
        return p, q

    def _write_back_ppc(self):
        """Write array-mode results to self.result without mutating the input ppc."""
        # ``summary`` and ``none`` never expose device result tables.  Keep the
        # compatibility call to this method, but avoid copying and rebuilding
        # every PPC table when only the state summary (or no result) is needed.
        if self.result_mode in {"summary", "none"}:
            self.result = {}
            return
        if (
            self._state_x_obj is self.x
            and self._power_x_obj is self.x
            and self._last_Sbus is not None
        ):
            theta = self._cache['theta']
            V = self._cache['V']
            Vc = self._cache['Vc']
            phi_re = self.x[self.base_phi_re:self.base_phi_re + self.N_phi] if self.N_phi > 0 else self._empty_phi
            phi_im = self.x[self.base_phi_im:self.base_phi_im + self.N_phi] if self.N_phi > 0 else self._empty_phi
            S_y = self._last_Sbus
            P_load, Q_load = self._cache['P_load'], self._cache['Q_load']
            if P_load is None or Q_load is None:
                P_load, Q_load = self._calc_load_power(V)
        else:
            theta, V, phi_re, phi_im = self._extract_state_vars(self.x)
            Vc = self._cache['Vc']
            P_load, Q_load = self._calc_load_power(V)
            I_y = self.Y.dot(Vc)
            S_y = Vc * np.conj(I_y)
        if self.N_phi > 0 and self.zero_a.size:
            I_ab = (phi_re[self.zero_phi_a] - phi_re[self.zero_phi_b]) + 1j * (
                phi_im[self.zero_phi_a] - phi_im[self.zero_phi_b]
            )
            Sa = Vc[self.zero_a] * np.conj(I_ab)
            Sb = Vc[self.zero_b] * np.conj(-I_ab)
            P_zero = np.bincount(self.zero_a, weights=Sa.real, minlength=self.N)
            P_zero += np.bincount(self.zero_b, weights=Sb.real, minlength=self.N)
            Q_zero = np.bincount(self.zero_a, weights=Sa.imag, minlength=self.N)
            Q_zero += np.bincount(self.zero_b, weights=Sb.imag, minlength=self.N)
        else:
            I_ab = np.array([], dtype=np.complex128)
            P_zero = np.zeros(self.N, dtype=np.float64)
            Q_zero = np.zeros(self.N, dtype=np.float64)

        P_gen = S_y.real + P_zero + P_load
        Q_gen = S_y.imag + Q_zero + Q_load
        external_p = self._external_ac_p_injection
        external_q = self._external_ac_q_injection
        if external_p is not None:
            external_p = np.asarray(external_p, dtype=np.float64)
            if external_p.shape == P_gen.shape:
                P_gen = P_gen + external_p
        if external_q is not None:
            external_q = np.asarray(external_q, dtype=np.float64)
            if external_q.shape == Q_gen.shape:
                Q_gen = Q_gen + external_q
        acac_p, acac_q = self._acac_node_power_injections()
        P_gen = P_gen + acac_p
        Q_gen = Q_gen + acac_q

        bus = self.ppc["bus"].copy()
        if bus.size:
            bus[:, [BUS_COLS["voltage"], BUS_COLS["angle"]]] = 0.0
        bus[self.active_node_rows, BUS_COLS["voltage"]] = V[self.active_node_solver_pos]
        bus[self.active_node_rows, BUS_COLS["angle"]] = theta[self.active_node_solver_pos]
        for pos, node in enumerate(self.node_list):
            node.voltage = float(V[pos])
            node.angle = float(theta[pos])

        shunt = self.ppc["shunt"].copy()
        if shunt.size:
            shunt[:, [SHUNT_COLS["p"], SHUNT_COLS["q"], SHUNT_COLS["current"]]] = 0.0
        shunt_p = np.zeros(self.ppc_shunt_rows.size, dtype=np.float64)
        shunt_q = np.zeros(self.ppc_shunt_rows.size, dtype=np.float64)
        shunt_control_q = np.zeros(self.ppc_shunt_rows.size, dtype=np.float64)
        shunt_control = np.array([], dtype=np.int32)
        shunt_q_mask = np.array([], dtype=bool)
        shunt_v_mask = np.array([], dtype=bool)
        if self.ppc_shunt_rows.size:
            rows = self.ppc_shunt_rows
            vm = V[self.ppc_shunt_pos]
            vm2 = vm * vm
            shunt_control = shunt[rows, SHUNT_COLS["control_type"]].astype(np.int32, copy=False)
            g_set = shunt[rows, SHUNT_COLS["g_set"]]
            b_set = shunt[rows, SHUNT_COLS["b_set"]]
            admittance_mask = (
                (shunt_control == SHUNT_B)
                | (shunt_control == SHUNT_Z)
                | (g_set != 0.0)
            )
            admittance_p = np.zeros(rows.size, dtype=np.float64)
            admittance_q = np.zeros(rows.size, dtype=np.float64)
            admittance_p[admittance_mask] = vm2[admittance_mask] * g_set[admittance_mask]
            admittance_q[admittance_mask] = -vm2[admittance_mask] * b_set[admittance_mask]

            admittance_control = (shunt_control == SHUNT_B) | (shunt_control == SHUNT_Z)
            shunt_p[admittance_control] = admittance_p[admittance_control]
            shunt_q[admittance_control] = admittance_q[admittance_control]

            injection_control = ~admittance_control
            shunt_p[injection_control] = -admittance_p[injection_control]
            shunt_q_mask = shunt_control == SHUNT_Q
            shunt_v_mask = shunt_control == SHUNT_V
            shunt_control_q[shunt_q_mask] = shunt[rows[shunt_q_mask], SHUNT_COLS["q_set"]]
            shunt_q[injection_control] = -admittance_q[injection_control]

        gen = self.ppc["gen"].copy()
        if gen.size:
            gen[:, [GEN_COLS["p"], GEN_COLS["q"], GEN_COLS["current"]]] = 0.0
        if self.ppc_gen_rows.size:
            live_gen = gen[self.ppc_gen_rows]
            controls = self.ppc_gen_control
            slack_mask = (controls == CTRL_SLACK) | self.ppc_auto_slack_mask
            pv_mask = controls == CTRL_PV
            pq_mask = (controls == CTRL_PQ) | (controls == CTRL_P)
            gen_p = np.zeros(self.ppc_gen_rows.size, dtype=np.float64)
            gen_q = np.zeros(self.ppc_gen_rows.size, dtype=np.float64)
            if np.any(pq_mask):
                gen_p[pq_mask] = live_gen[pq_mask, GEN_COLS["p_set"]]
                gen_q[pq_mask] = live_gen[pq_mask, GEN_COLS["q_set"]]
            if np.any(pv_mask):
                gen_p[pv_mask] = live_gen[pv_mask, GEN_COLS["p_set"]]
            # Group participants once instead of scanning every device for each bus.
            voltage_gen = slack_mask | pv_mask
            fixed_gen_q = np.bincount(
                self.ppc_gen_pos[~voltage_gen], weights=gen_q[~voltage_gen], minlength=self.N,
            )
            fixed_shunt_q = np.bincount(
                self.ppc_shunt_pos[shunt_q_mask], weights=shunt_control_q[shunt_q_mask], minlength=self.N,
            )
            shunt_count = np.count_nonzero(shunt_v_mask)
            gen_count = np.count_nonzero(voltage_gen)
            allocated_q = allocate_grouped_residual(
                np.r_[self.ppc_gen_pos[voltage_gen], self.ppc_shunt_pos[shunt_v_mask]],
                np.r_[live_gen[voltage_gen, GEN_COLS["q_set"]],
                      shunt[self.ppc_shunt_rows[shunt_v_mask], SHUNT_COLS["q_set"]]],
                Q_gen - fixed_gen_q - fixed_shunt_q,
                lower=np.r_[self.ppc_gen_q_min[voltage_gen], np.full(shunt_count, -np.inf)],
                upper=np.r_[self.ppc_gen_q_max[voltage_gen], np.full(shunt_count, np.inf)],
            )
            gen_q[voltage_gen] = allocated_q[:gen_count]
            shunt_control_q[shunt_v_mask] = allocated_q[gen_count:]
            fixed_gen_p = np.bincount(
                self.ppc_gen_pos[~slack_mask], weights=gen_p[~slack_mask], minlength=self.N,
            )
            gen_p[slack_mask] = allocate_grouped_residual(
                self.ppc_gen_pos[slack_mask], live_gen[slack_mask, GEN_COLS["p_set"]],
                P_gen - fixed_gen_p,
                lower=self.ppc_gen_p_min[slack_mask],
                upper=live_gen[slack_mask, GEN_COLS["p_max"]],
                alpha=live_gen[slack_mask, GEN_COLS["alpha"]],
            )
            gen_v = V[self.ppc_gen_pos]
            gen_current = np.divide(
                np.hypot(gen_p, gen_q),
                gen_v,
                out=np.zeros_like(gen_p),
                where=gen_v > self.min_voltage,
            )
            gen[self.ppc_gen_rows, GEN_COLS["p"]] = gen_p
            gen[self.ppc_gen_rows, GEN_COLS["q"]] = gen_q
            gen[self.ppc_gen_rows, GEN_COLS["current"]] = gen_current

        load = self.ppc["load"].copy()
        if load.size:
            load[:, [LOAD_COLS["p"], LOAD_COLS["q"], LOAD_COLS["current"]]] = 0.0
        if self.ppc_load_rows.size:
            vm = V[self.ppc_load_pos]
            load_p = self.load_pv0 + self.load_pv1 * vm + self.load_pv2 * vm ** 2
            load_q = self.load_qv0 + self.load_qv1 * vm + self.load_qv2 * vm ** 2
            load_current = np.divide(
                np.hypot(load_p, load_q),
                vm,
                out=np.zeros_like(load_p),
                where=vm > self.min_voltage,
            )
            load[self.ppc_load_rows, LOAD_COLS["p"]] = load_p
            load[self.ppc_load_rows, LOAD_COLS["q"]] = load_q
            load[self.ppc_load_rows, LOAD_COLS["current"]] = load_current

        if self.ppc_shunt_rows.size:
            rows = self.ppc_shunt_rows
            vm = V[self.ppc_shunt_pos]
            injection_control = (shunt_control == SHUNT_Q) | (shunt_control == SHUNT_V)
            shunt_q[injection_control] += shunt_control_q[injection_control]
            current = np.divide(
                np.hypot(shunt_p, shunt_q),
                vm,
                out=np.zeros_like(shunt_p),
                where=vm > self.min_voltage,
            )
            shunt[rows, SHUNT_COLS["p"]] = shunt_p
            shunt[rows, SHUNT_COLS["q"]] = shunt_q
            shunt[rows, SHUNT_COLS["current"]] = current

        branch = self.ppc["branch"].copy()
        if branch.size:
            branch[:, [
                BRANCH_COLS["i_p"],
                BRANCH_COLS["i_q"],
                BRANCH_COLS["i_c"],
                BRANCH_COLS["j_p"],
                BRANCH_COLS["j_q"],
                BRANCH_COLS["j_c"],
            ]] = 0.0
        if self.ppc_branch_rows.size:
            Vi = Vc[self.branch_i]
            Vj = Vc[self.branch_j]
            I_ij = self.branch_yff * Vi + self.branch_yft * Vj
            I_ji = self.branch_ytf * Vi + self.branch_ytt * Vj
            S_ij = Vi * np.conj(I_ij)
            S_ji = Vj * np.conj(I_ji)
            rows = self.ppc_branch_rows
            branch[rows, BRANCH_COLS["i_p"]] = S_ij.real
            branch[rows, BRANCH_COLS["i_q"]] = S_ij.imag
            branch[rows, BRANCH_COLS["i_c"]] = np.abs(I_ij)
            branch[rows, BRANCH_COLS["j_p"]] = S_ji.real
            branch[rows, BRANCH_COLS["j_q"]] = S_ji.imag
            branch[rows, BRANCH_COLS["j_c"]] = np.abs(I_ji)

        transformer = self.ppc["transformer"].copy()
        if transformer.size:
            transformer[:, [
                TRANSFORMER_COLS["i_p"],
                TRANSFORMER_COLS["i_q"],
                TRANSFORMER_COLS["i_c"],
                TRANSFORMER_COLS["j_p"],
                TRANSFORMER_COLS["j_q"],
                TRANSFORMER_COLS["j_c"],
            ]] = 0.0
        if self.ppc_transformer_rows.size:
            Vi = Vc[self.transformer_i]
            Vj = Vc[self.transformer_j]
            I_ij = self.transformer_yff * Vi + self.transformer_yft * Vj
            I_ji = self.transformer_ytf * Vi + self.transformer_ytt * Vj
            S_ij = Vi * np.conj(I_ij)
            S_ji = Vj * np.conj(I_ji)
            rows = self.ppc_transformer_rows
            transformer[rows, TRANSFORMER_COLS["i_p"]] = S_ij.real
            transformer[rows, TRANSFORMER_COLS["i_q"]] = S_ij.imag
            transformer[rows, TRANSFORMER_COLS["i_c"]] = np.abs(I_ij)
            transformer[rows, TRANSFORMER_COLS["j_p"]] = S_ji.real
            transformer[rows, TRANSFORMER_COLS["j_q"]] = S_ji.imag
            transformer[rows, TRANSFORMER_COLS["j_c"]] = np.abs(I_ji)

        three_winding_transformer = self.ppc.get(
            "three_winding_transformer",
            np.zeros((0, len(THREE_WINDING_TRANSFORMER_COLS)), dtype=np.float64),
        ).copy()
        cols3 = THREE_WINDING_TRANSFORMER_COLS
        three_result_cols = [
            cols3[attr]
            for attr in ("i_p", "i_q", "i_c", "j_p", "j_q", "j_c", "k_p", "k_q", "k_c")
        ]
        if three_winding_transformer.size:
            three_winding_transformer[:, three_result_cols] = 0.0
        if self.ppc_three_winding_transformer_rows.size:
            rows = self.ppc_three_winding_transformer_rows
            terminal_pos = np.column_stack(
                (
                    self.three_winding_transformer_i,
                    self.three_winding_transformer_j,
                    self.three_winding_transformer_k,
                )
            )
            terminal_voltage = Vc[terminal_pos]
            terminal_current = np.einsum(
                "nij,nj->ni",
                self.three_winding_transformer_y,
                terminal_voltage,
            )
            terminal_power = terminal_voltage * np.conj(terminal_current)
            for terminal_pos_idx, terminal in enumerate(("i", "j", "k")):
                three_winding_transformer[rows, cols3[f"{terminal}_p"]] = terminal_power[:, terminal_pos_idx].real
                three_winding_transformer[rows, cols3[f"{terminal}_q"]] = terminal_power[:, terminal_pos_idx].imag
                three_winding_transformer[rows, cols3[f"{terminal}_c"]] = np.abs(
                    terminal_current[:, terminal_pos_idx]
                )

        zero_branch = self.ppc["zero_branch"].copy()
        switch = finalize_lf_closed_status_results(self.ppc["switch"].copy(), SWITCH_COLS)
        breaker = finalize_lf_closed_status_results(
            self.ppc.get(
                "break",
                np.zeros((0, len(SWITCH_COLS)), dtype=np.float64),
            ).copy(),
            SWITCH_COLS,
        )
        if zero_branch.size:
            zero_branch[:, [
                ZERO_BRANCH_COLS["p"],
                ZERO_BRANCH_COLS["q"],
                ZERO_BRANCH_COLS["current"],
            ]] = 0.0
        if switch.size:
            switch[:, [SWITCH_COLS["p"], SWITCH_COLS["q"], SWITCH_COLS["current"]]] = 0.0
        if breaker.size:
            breaker[:, [SWITCH_COLS["p"], SWITCH_COLS["q"], SWITCH_COLS["current"]]] = 0.0
        if self.zero_idx.size and I_ab.size:
            s_from = Vc[self.zero_a] * np.conj(I_ab)
            i_abs = np.abs(I_ab)
            for edge_code, table, p_col, q_col, c_col in (
                (0, zero_branch, ZERO_BRANCH_COLS["p"], ZERO_BRANCH_COLS["q"], ZERO_BRANCH_COLS["current"]),
                (2, breaker, SWITCH_COLS["p"], SWITCH_COLS["q"], SWITCH_COLS["current"]),
                (1, switch, SWITCH_COLS["p"], SWITCH_COLS["q"], SWITCH_COLS["current"]),
            ):
                mask = self.zero_type == edge_code
                if np.any(mask):
                    dev_rows = self.zero_idx[mask]
                    table[dev_rows, p_col] = s_from.real[mask]
                    table[dev_rows, q_col] = s_from.imag[mask]
                    table[dev_rows, c_col] = i_abs[mask]

        acac = self.ppc.get("acac", np.zeros((0, len(ACAC_COLS)), dtype=np.float64)).copy()
        result_columns = [
            ACAC_COLS[name]
            for name in ("i_p", "i_q", "j_p", "j_q", "i_i", "j_i")
        ]
        if acac.size:
            acac[:, result_columns] = 0.0
        if self.N_acac:
            state = self._acac_state_view(self.x)
            vi = V[self.acac_i_pos]
            vj = V[self.acac_j_pos]
            i_current = np.divide(
                np.hypot(state[:, 0], state[:, 1]),
                vi,
                out=np.zeros(self.N_acac, dtype=np.float64),
                where=np.abs(vi) > self.min_voltage,
            )
            j_current = np.divide(
                np.hypot(state[:, 2], state[:, 3]),
                vj,
                out=np.zeros(self.N_acac, dtype=np.float64),
                where=np.abs(vj) > self.min_voltage,
            )
            rows = self.acac_row_pos
            acac[rows, ACAC_COLS["i_p"]] = state[:, 0]
            acac[rows, ACAC_COLS["i_q"]] = state[:, 1]
            acac[rows, ACAC_COLS["j_p"]] = state[:, 2]
            acac[rows, ACAC_COLS["j_q"]] = state[:, 3]
            acac[rows, ACAC_COLS["i_i"]] = i_current
            acac[rows, ACAC_COLS["j_i"]] = j_current

        self.result = {
            "bus": bus,
            "gen": gen,
            "load": load,
            "shunt": shunt,
            "branch": branch,
            "transformer": transformer,
            "three_winding_transformer": three_winding_transformer,
            "zero_branch": zero_branch,
            "switch": switch,
            "break": breaker,
            "acac": acac,
        }

def print_ac_result(calc: ACPowerFlowCalc, rc: int) -> None:
    for isl in calc.skipped_islands:
        print(f"跳过岛屿 {isl.idx}: 无可用平衡节点或定电压源")

    def _names(key, count):
        values = calc.ppc.get(key)
        if values is None:
            return np.asarray([str(i) for i in range(count)], dtype=object)
        return values

    # 结果输出（精简）
    print("\n=== 潮流计算结果 ===")

    # 节点电压
    print("\n1. 节点电压 (pu):")
    bus = calc.result["bus"]
    bus_names = _names("bus_name", bus.shape[0])
    slack_node_ids = set(calc.ppc_node_idx[calc.node_type == AC_NODE_TYPE_SLACK].tolist())
    active_node_ids = set(calc.ppc_node_idx.tolist())
    for row, name in zip(bus, bus_names):
        node_idx = int(row[BUS_COLS["idx"]])
        flags = []
        if node_idx in slack_node_ids:
            flags.append("松弛节点")
        if node_idx not in active_node_ids:
            flags.append("未参与计算")
        suffix = f" ({', '.join(flags)})" if flags else ""
        print(
            f"   节点 {node_idx} {name}: "
            f"{row[BUS_COLS['voltage']]:.6f}  {row[BUS_COLS['angle']]:.6f}{suffix}"
        )

    # 支路信息
    branch = calc.result["branch"]
    branch_names = _names("branch_name", branch.shape[0])
    p_loss_br = float(np.sum(branch[:, BRANCH_COLS["i_p"]] + branch[:, BRANCH_COLS["j_p"]])) if branch.size else 0.0
    print("\n2. 支路信息:")
    for row, name in zip(branch, branch_names):
        loss = row[BRANCH_COLS["i_p"]] + row[BRANCH_COLS["j_p"]]
        print(f"   支路 {int(row[BRANCH_COLS['idx']])} {name} ({int(row[BRANCH_COLS['i_node']])}->{int(row[BRANCH_COLS['j_node']])}):")
        print(
            f"     送端功率: {row[BRANCH_COLS['i_p']]:.6f} + j {row[BRANCH_COLS['i_q']]:.6f} pu, "
            f"受端功率: {row[BRANCH_COLS['j_p']]:.6f} + j{row[BRANCH_COLS['j_q']]:.6f} pu"
        )
        print(f"     有功损耗: {loss:.6f} pu")

    # 变压器信息
    transformer = calc.result["transformer"]
    transformer_names = _names("transformer_name", transformer.shape[0])
    p_loss_tr = (
        float(np.sum(transformer[:, TRANSFORMER_COLS["i_p"]] + transformer[:, TRANSFORMER_COLS["j_p"]]))
        if transformer.size
        else 0.0
    )
    print("\n3. 变压器信息:")
    for row, name in zip(transformer, transformer_names):
        loss = row[TRANSFORMER_COLS["i_p"]] + row[TRANSFORMER_COLS["j_p"]]
        print(
            f"   变压器 {int(row[TRANSFORMER_COLS['idx']])} {name} "
            f"({int(row[TRANSFORMER_COLS['i_node']])}->{int(row[TRANSFORMER_COLS['j_node']])}):"
        )
        print(
            f"     送端功率: {row[TRANSFORMER_COLS['i_p']]:.6f} + j {row[TRANSFORMER_COLS['i_q']]:.6f} pu, "
            f"受端功率: {row[TRANSFORMER_COLS['j_p']]:.6f} + j{row[TRANSFORMER_COLS['j_q']]:.6f} pu"
        )
        print(f"     有功损耗: {loss:.6f} pu")

    three_winding_transformer = calc.result.get(
        "three_winding_transformer",
        np.zeros((0, len(THREE_WINDING_TRANSFORMER_COLS)), dtype=np.float64),
    )
    three_winding_names = _names(
        "three_winding_transformer_name",
        three_winding_transformer.shape[0],
    )
    p_loss_tr3 = (
        float(
            np.sum(
                three_winding_transformer[:, THREE_WINDING_TRANSFORMER_COLS["i_p"]]
                + three_winding_transformer[:, THREE_WINDING_TRANSFORMER_COLS["j_p"]]
                + three_winding_transformer[:, THREE_WINDING_TRANSFORMER_COLS["k_p"]]
            )
        )
        if three_winding_transformer.size
        else 0.0
    )
    print("\n3.1 三绕组主变信息:")
    for row, name in zip(three_winding_transformer, three_winding_names):
        cols3 = THREE_WINDING_TRANSFORMER_COLS
        loss = row[cols3["i_p"]] + row[cols3["j_p"]] + row[cols3["k_p"]]
        print(
            f"   三绕组主变 {int(row[cols3['idx']])} {name} "
            f"({int(row[cols3['i_node']])}/{int(row[cols3['j_node']])}/{int(row[cols3['k_node']])}):"
        )
        for terminal, label in (("i", "i端"), ("j", "j端"), ("k", "k端")):
            print(
                f"     {label}功率: {row[cols3[f'{terminal}_p']]:.6f} "
                f"+ j {row[cols3[f'{terminal}_q']]:.6f} pu, "
                f"电流: {row[cols3[f'{terminal}_c']]:.6f} pu"
            )
        print(f"     有功损耗: {loss:.6f} pu")

    # 其他设备信息
    zero_branch = calc.result["zero_branch"]
    zero_names = _names("zero_branch_name", zero_branch.shape[0])
    print("\n4. 零阻抗支路信息:")
    for row, name in zip(zero_branch, zero_names):
        print(
            f"   零阻抗支路 {int(row[ZERO_BRANCH_COLS['idx']])} {name}: "
            f"电流={row[ZERO_BRANCH_COLS['current']]:.6f} pu, "
            f"功率={row[ZERO_BRANCH_COLS['p']]:.6f} + j {row[ZERO_BRANCH_COLS['q']]:.6f} pu"
        )

    switch = calc.result["switch"]
    switch_names = _names("switch_name", switch.shape[0])
    print("\n5. 开关信息:")
    for row, name in zip(switch, switch_names):
        status = "闭合" if int(row[SWITCH_COLS["closed_status"]]) == 1 else "断开"
        print(
            f"   开关 {int(row[SWITCH_COLS['idx']])} {name} "
            f"(状态:{status}): 电流={row[SWITCH_COLS['current']]:.6f} pu"
        )

    breaker = calc.result["break"]
    breaker_names = _names("break_name", breaker.shape[0])
    if breaker.shape[0]:
        print("\n5.1 刀闸信息:")
        for row, name in zip(breaker, breaker_names):
            status = "闭合" if int(row[BREAK_COLS["closed_status"]]) == 1 else "断开"
            print(
                f"   刀闸 {int(row[BREAK_COLS['idx']])} {name} "
                f"(状态:{status}): 电流={row[BREAK_COLS['current']]:.6f} pu, "
                f"功率={row[BREAK_COLS['p']]:.6f} + j {row[BREAK_COLS['q']]:.6f} pu"
            )

    load = calc.result["load"]
    load_names = _names("load_name", load.shape[0])
    print("\n6. 负荷信息:")
    for row, name in zip(load, load_names):
        print(
            f"   负荷 {int(row[LOAD_COLS['idx']])} {name}: "
            f"消耗功率={row[LOAD_COLS['p']]:.6f} + j {row[LOAD_COLS['q']]:.6f} pu"
        )

    gen = calc.result["gen"]
    gen_names = _names("gen_name", gen.shape[0])
    print("\n7. 发电机信息:")
    for row, name in zip(gen, gen_names):
        print(
            f"   发电机 {int(row[GEN_COLS['idx']])} {name}: "
            f"送出功率={row[GEN_COLS['p']]:.6f} + j {row[GEN_COLS['q']]:.6f} pu"
        )

    # 收敛信息和功率平衡
    print("\n8. 计算收敛信息:")
    calc_scope = f"全网 (参与节点 {calc.N}/{bus.shape[0]})"
    print(
        f"   {calc_scope}: {'✓ 已收敛' if calc.converged else '✗ 未收敛'}, "
        f"返回码: {rc}, 迭代次数: {calc.iterations}, 最终残差: {calc.normF:.2e}"
    )

    total_gen_power = float(np.sum(gen[:, GEN_COLS["p"]])) if gen.size else 0.0
    total_load_power = float(np.sum(load[:, LOAD_COLS["p"]])) if load.size else 0.0
    total_loss = total_gen_power - total_load_power
    shunt = calc.result["shunt"]
    p_loss_gs = float(np.sum(shunt[:, SHUNT_COLS["p"]])) if shunt.size else 0.0
    print("\n9. 功率平衡校验:")
    print(f"   总发电功率: {total_gen_power:.6f} pu")
    print(f"   总负荷功率: {total_load_power:.6f} pu")
    print(
        f"   总网损: {total_loss:.6f} pu "
        f"(支路: {p_loss_br:.6f} pu, 双绕组主变: {p_loss_tr:.6f} pu, "
        f"三绕组主变: {p_loss_tr3:.6f} pu, 并联电导: {p_loss_gs:.6f} pu)"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AC power flow")
    parser.add_argument("file", nargs="?", default=str(model_file("ac", "ieee300.e")), help="AC E file path")
    parser.add_argument("--para", default=str(DEFAULT_LF_PARAMETER_FILE), help="Power-flow algorithm parameter file.")
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--min-voltage", type=float, default=None)
    parser.add_argument("--linear-solver", default="pyklu")
    parser.add_argument("--result-mode", choices=("full", "array", "summary", "none"), default="full")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    ppc = load_ac_ppc_from_e_file(args.file)
    calc = ACPowerFlowCalc(
        ppc,
        parameter_file=args.para,
        tol=args.tol,
        max_iter=args.max_iter,
        min_voltage=args.min_voltage,
        linear_solver=args.linear_solver,
        result_mode=args.result_mode,
        verbose=not args.quiet,
    )
    rc = calc.run()
    if not args.quiet and calc.result_mode == "full":
        print_ac_result(calc, rc)
    elif not args.quiet:
        print(f"收敛状态: {'已收敛' if calc.converged else '未收敛'}, iter={calc.iterations}, normF={calc.normF:.3e}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
