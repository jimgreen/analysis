from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(slots=True)
class ACNodeResult:
    volt: float
    angle: float


@dataclass(slots=True)
class ACTwoPortResult:
    i_p: float
    i_q: float
    i_c: float
    i_v: float
    j_p: float
    j_q: float
    j_c: float
    j_v: float


@dataclass(slots=True)
class ACThreePortResult:
    i_p: float
    i_q: float
    i_c: float
    i_v: float
    j_p: float
    j_q: float
    j_c: float
    j_v: float
    k_p: float
    k_q: float
    k_c: float
    k_v: float


@dataclass(slots=True)
class ACSinglePortResult:
    i_p: float
    i_q: float
    i_c: float
    i_v: float


@dataclass(slots=True)
class ACSwitchResult:
    i_p: float
    i_q: float
    i_c: float
    i_v: float
    closed_status_set: int
    closed_status: int


@dataclass(slots=True)
class DCNodeResult:
    volt: float


@dataclass(slots=True)
class DCTwoPortResult:
    i_p: float
    i_c: float
    i_v: float
    j_p: float
    j_c: float
    j_v: float


@dataclass(slots=True)
class DCSinglePortResult:
    i_p: float
    i_c: float
    i_v: float


@dataclass(slots=True)
class DCSwitchResult:
    i_p: float
    i_c: float
    i_v: float
    closed_status_set: int
    closed_status: int


@dataclass(slots=True)
class DCACPortResult:
    i_p: float
    i_c: float
    i_v: float
    j_p: float
    j_q: float
    j_c: float
    j_v: float


class LazyResultDict(dict):
    """按需物化的 dict 子类，兼容 isinstance(..., dict) 与全部字典操作。

    在 result_mode='full' 构建结果对象时仅保存数组快照与构造闭包，首次访问某张
    设备表时一次性批量物化，避免为未读取的设备表创建数十万个 Python 对象。
    """

    __slots__ = ("_builder", "_count", "_materialized")

    def __init__(self, builder, count: int) -> None:
        super().__init__()
        self._count = int(count)
        if self._count <= 0 or builder is None:
            self._builder = None
            self._materialized = True
        else:
            self._builder = builder
            self._materialized = False

    def _ensure(self) -> None:
        if not self._materialized:
            builder = self._builder
            self._builder = None
            self._materialized = True
            if builder is not None:
                super().update(builder())

    def __bool__(self) -> bool:
        if not self._materialized:
            return self._count > 0
        return super().__len__() > 0

    def __len__(self) -> int:
        self._ensure()
        return super().__len__()

    def __getitem__(self, key):
        self._ensure()
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        self._ensure()
        super().__setitem__(key, value)

    def __delitem__(self, key) -> None:
        self._ensure()
        super().__delitem__(key)

    def __contains__(self, key) -> bool:
        self._ensure()
        return super().__contains__(key)

    def __iter__(self):
        self._ensure()
        return super().__iter__()

    def __reversed__(self):
        self._ensure()
        return super().__reversed__()

    def get(self, key, default=None):
        self._ensure()
        return super().get(key, default)

    def keys(self):
        self._ensure()
        return super().keys()

    def values(self):
        self._ensure()
        return super().values()

    def items(self):
        self._ensure()
        return super().items()

    def pop(self, *args):
        self._ensure()
        return super().pop(*args)

    def popitem(self):
        self._ensure()
        return super().popitem()

    def clear(self) -> None:
        self._builder = None
        self._count = 0
        self._materialized = True
        super().clear()

    def update(self, *args, **kwargs) -> None:
        self._ensure()
        super().update(*args, **kwargs)

    def setdefault(self, key, default=None):
        self._ensure()
        return super().setdefault(key, default)

    def copy(self) -> dict:
        self._ensure()
        return super().copy()

    def __repr__(self) -> str:
        self._ensure()
        return super().__repr__()

    def __eq__(self, other) -> bool:
        if isinstance(other, LazyResultDict):
            other._ensure()
        self._ensure()
        return super().__eq__(other)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __or__(self, other):
        if isinstance(other, LazyResultDict):
            other._ensure()
        self._ensure()
        return super().__or__(other)

    def __ror__(self, other):
        self._ensure()
        return super().__ror__(other)

    def __ior__(self, other):
        if isinstance(other, LazyResultDict):
            other._ensure()
        self._ensure()
        return super().__ior__(other)


def build_result_voltage_lookup(bus_rows, idx_col: int, voltage_col: int):
    """Build a local node-id to voltage lookup for full LF result assembly.

    PPC tables produced by the fast loaders normally contain rows ordered by
    consecutive node IDs.  In that case direct indexing avoids sorting and a
    ``searchsorted`` call for every device table.  Imported/custom PPC tables
    keep the previous sorted lookup as a correctness-preserving fallback.
    The returned closure owns only this run's arrays; nothing is cached on the
    network or shared between runs.
    """
    rows = np.asarray(bus_rows)
    if rows.ndim != 2 or rows.shape[0] == 0:
        def empty_lookup(col_array):
            values = np.asarray(col_array)
            return np.zeros(values.shape, dtype=np.float64)

        return empty_lookup

    bus_idx = rows[:, idx_col].astype(np.int64, copy=False)
    bus_volt = rows[:, voltage_col]
    if bus_idx.size and (
        bus_idx.size == 1
        or np.all(bus_idx[1:] == bus_idx[:-1] + 1)
    ):
        offset = int(bus_idx[0])
        width = int(bus_idx.size)

        def direct_lookup(col_array):
            values = np.asarray(col_array)
            if values.size == 0:
                return np.zeros(values.shape, dtype=np.float64)
            node_ids = values.astype(np.int64, copy=False)
            relative = node_ids - offset
            valid = (relative >= 0) & (relative < width)
            result = np.zeros(node_ids.shape, dtype=np.float64)
            if np.any(valid):
                result[valid] = bus_volt[relative[valid]]
            return result

        return direct_lookup

    order = np.argsort(bus_idx, kind="stable")
    sorted_idx = bus_idx[order]

    def sorted_lookup(col_array):
        values = np.asarray(col_array)
        if values.size == 0:
            return np.zeros(values.shape, dtype=np.float64)
        node_ids = values.astype(np.int64, copy=False)
        if sorted_idx.size == 0:
            return np.zeros(node_ids.shape, dtype=np.float64)
        pos = np.searchsorted(sorted_idx, node_ids)
        pos_clip = np.clip(pos, 0, sorted_idx.size - 1)
        hit = sorted_idx[pos_clip] == node_ids
        bus_row = np.where(hit, order[pos_clip], 0)
        return np.where(hit, bus_volt[bus_row], 0.0)

    return sorted_lookup


def result_name_list(names, count: int):
    """Return result keys as Python strings without needless dtype conversion."""
    if names is None or len(names) != count:
        return [str(i) for i in range(count)]
    if isinstance(names, np.ndarray):
        if names.dtype == object:
            vals = names.tolist()
            if all(type(v) is str for v in vals):
                return vals
            return [str(v) for v in vals]
        return names.astype(str).tolist()
    from model.array_common import LazyNameArray

    if isinstance(names, LazyNameArray):
        raw_list = names.raw_str_list()
        if raw_list is not None:
            return raw_list
        return names.tolist()
    values = names.tolist() if hasattr(names, "tolist") else list(names)
    return [str(value) for value in values]


def device_key(device) -> str:
    """Return the stable result-map key for a device object."""
    return str(getattr(device, "name", "") or getattr(device, "idx", id(device)))


def find_spanning_tree_edges(edges, n_nodes: int):
    """Return edge indices selected by Kruskal union-find for a spanning forest."""
    parent = np.arange(n_nodes, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[ry] = rx
        return True

    tree_indices = []
    for idx, (u, v) in enumerate(edges):
        if union(int(u), int(v)):
            tree_indices.append(idx)
    return tree_indices


def normalize_result_mode(result_mode: str, context: str) -> str:
    mode = str(result_mode or "full").strip().lower()
    if mode not in {"full", "summary", "array", "none"}:
        raise ValueError(f"Unsupported {context} result_mode: {result_mode!r}")
    return mode


def optional_ppc_vector(ppc, key: str, size: int, default=np.nan) -> np.ndarray:
    """Return a length-``size`` optional PPC vector without shape surprises."""
    values = np.asarray((ppc or {}).get(key, ()), dtype=np.float64).reshape(-1)
    result = np.full(int(size), float(default), dtype=np.float64)
    if values.size:
        result[: min(result.size, values.size)] = values[: result.size]
    return result


def allocate_grouped_residual(positions, baseline, targets, *, lower=None, upper=None, alpha=None):
    """Allocate each bus total with stable device order and no network cache.

    Singleton buses take their total directly, as in allocate_limited_residual.
    Only buses with multiple participants need the bounded allocation loop.
    All indices are built for this call; published result arrays stay independent.
    """
    positions = np.asarray(positions, dtype=np.intp).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if positions.size != baseline.size:
        raise ValueError('positions and baseline must have the same length')
    limits = {}
    for name, values in (('lower', lower), ('upper', upper), ('alpha', alpha)):
        if values is not None:
            values = np.asarray(values, dtype=np.float64).reshape(-1)
            if values.size != baseline.size:
                raise ValueError(f'{name} and baseline must have the same length')
            limits[name] = values
    result = baseline.copy()
    if not positions.size:
        return result
    targets = np.asarray(targets, dtype=np.float64)
    order = np.argsort(positions, kind='stable')
    sorted_pos = positions[order]
    starts = np.r_[0, np.flatnonzero(sorted_pos[1:] != sorted_pos[:-1]) + 1]
    stops = np.r_[starts[1:], order.size]
    single = (stops - starts) == 1
    single_indices = order[starts[single]]
    result[single_indices] = targets[positions[single_indices]]
    for start, stop in zip(starts[~single], stops[~single]):
        indices = order[start:stop]
        result[indices] = allocate_limited_residual(
            baseline[indices], targets[positions[indices[0]]],
            **{name: values[indices] for name, values in limits.items()},
        )
    return result


def allocate_limited_residual(
    baseline,
    target_total: float,
    *,
    lower=None,
    upper=None,
    alpha=None,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Apply setpoints first, then distribute the residual in two stages.

    The first stage respects directional operating limits. Finite headroom is
    multiplied by ``alpha``; saturated devices are removed and the residual is
    redistributed in subsequent passes. Missing limits remain unbounded and
    fall back to the alpha weight, preserving legacy E files without limits.

    If all available headroom is exhausted before the requested total is met,
    the second stage deliberately exceeds the limits and allocates the remaining
    residual by normalized ``alpha``. This keeps the solved network power-balanced
    while leaving the limit violation visible to higher-level controls.
    """
    result = np.asarray(baseline, dtype=np.float64).reshape(-1).copy()
    if result.size == 0:
        return result

    # A large LF result write-back workload consists of one balancing device
    # per bus.  The general allocator below is intentionally defensive and
    # allocates several temporary arrays for every call; for one participant
    # the final overflow stage necessarily assigns the complete target to that
    # participant, regardless of its finite headroom.  Keep the size checks so
    # malformed optional vectors still fail with the same contract, then take
    # the scalar fast path.
    if result.size == 1:
        for values, label in (
            (lower, "下限"),
            (upper, "上限"),
            (alpha, "alpha"),
        ):
            if values is not None:
                checked = np.asarray(values, dtype=np.float64).reshape(-1)
                if checked.size != 1:
                    raise ValueError(f"分配 {label} 必须与基准设定数组长度一致")
        return np.asarray([float(target_total)], dtype=np.float64)

    lower_values = (
        np.full(result.size, -np.inf, dtype=np.float64)
        if lower is None
        else np.asarray(lower, dtype=np.float64).reshape(-1)
    )
    upper_values = (
        np.full(result.size, np.inf, dtype=np.float64)
        if upper is None
        else np.asarray(upper, dtype=np.float64).reshape(-1)
    )
    if lower_values.size != result.size or upper_values.size != result.size:
        raise ValueError("分配上下限必须与基准设定数组长度一致")
    # NaN denotes an omitted optional limit. Keep explicit infinities intact.
    lower_values = np.where(np.isnan(lower_values), -np.inf, lower_values)
    upper_values = np.where(np.isnan(upper_values), np.inf, upper_values)

    if alpha is None:
        alpha_values = np.ones(result.size, dtype=np.float64)
    else:
        alpha_values = np.asarray(alpha, dtype=np.float64).reshape(-1)
        if alpha_values.size != result.size:
            raise ValueError("分配 alpha 必须与基准设定数组长度一致")
    alpha_values = np.nan_to_num(alpha_values, nan=0.0, posinf=0.0, neginf=0.0)
    alpha_values = np.maximum(alpha_values, 0.0)
    if not np.any(alpha_values > 0.0):
        alpha_values.fill(1.0)

    remaining = float(target_total) - float(np.sum(result))

    # With no finite limits there can be no saturation pass.  This is common
    # for legacy E files and reduces the allocation to one weighted vector
    # operation while preserving the exact residual correction semantics.
    if not np.any(np.isfinite(lower_values) | np.isfinite(upper_values)):
        if abs(remaining) <= tolerance:
            return result
        weights = alpha_values
        result += remaining * weights / float(np.sum(weights))
        correction = float(target_total) - float(np.sum(result))
        if abs(correction) > tolerance:
            participating = np.flatnonzero(weights > 0.0)
            if participating.size:
                result[int(participating[-1])] += correction
        return result

    active = np.ones(result.size, dtype=bool)
    for _ in range(result.size + 1):
        if abs(remaining) <= tolerance:
            break
        if remaining > 0.0:
            headroom = upper_values - result
        else:
            headroom = result - lower_values
        usable = active & (headroom > tolerance)
        if not np.any(usable):
            break

        finite_headroom = usable & np.isfinite(headroom)
        weights = np.zeros(result.size, dtype=np.float64)
        weights[finite_headroom] = headroom[finite_headroom] * alpha_values[finite_headroom]
        unbounded = usable & ~finite_headroom
        weights[unbounded] = alpha_values[unbounded]
        if not np.any(weights > 0.0):
            weights[usable] = 1.0
        proposals = abs(remaining) * weights / float(np.sum(weights))
        saturated = finite_headroom & (proposals >= headroom - tolerance)
        applied = proposals.copy()
        applied[saturated] = headroom[saturated]
        applied[~usable] = 0.0
        if remaining > 0.0:
            result += applied
        else:
            result -= applied
        remaining -= float(np.sum(applied)) if remaining > 0.0 else -float(np.sum(applied))
        active[saturated] = False
        if not np.any(saturated) and np.all(applied <= tolerance):
            break

    remaining = float(target_total) - float(np.sum(result))
    if abs(remaining) > tolerance:
        overflow_weights = alpha_values.copy()
        if not np.any(overflow_weights > 0.0):
            overflow_weights.fill(1.0)
        result += remaining * overflow_weights / float(np.sum(overflow_weights))

        # Remove the final floating-point accumulation error without changing
        # the intended participation set.
        correction = float(target_total) - float(np.sum(result))
        if abs(correction) > tolerance:
            participating = np.flatnonzero(overflow_weights > 0.0)
            result[int(participating[-1])] += correction
    return result


def build_group_share_metadata(positions, setpoints, active_mask):
    """纯 NumPy 向量化计算同母线多设备控制分组均值、代表掩码与均分映射。"""
    positions = np.asarray(positions, dtype=np.int32)
    setpoints = np.asarray(setpoints, dtype=np.float64)
    active_mask = np.asarray(active_mask, dtype=bool)
    n = int(positions.size)
    avg_set = setpoints.copy()
    rep_mask = np.zeros(n, dtype=bool)
    share_mask = np.zeros(n, dtype=bool)
    ref_idx = np.arange(n, dtype=np.int32)
    if n == 0 or not np.any(active_mask):
        return avg_set, rep_mask, share_mask, ref_idx

    active_idx = np.flatnonzero(active_mask).astype(np.int32, copy=False)
    pos_active = positions[active_idx]
    order = np.argsort(pos_active, kind="stable")
    sorted_idx = active_idx[order]
    sorted_pos = pos_active[order]

    is_new = np.empty(sorted_pos.size, dtype=bool)
    is_new[0] = True
    if sorted_pos.size > 1:
        is_new[1:] = sorted_pos[1:] != sorted_pos[:-1]
    starts = np.flatnonzero(is_new)
    ends = np.empty(starts.size, dtype=np.intp)
    if starts.size > 1:
        ends[:-1] = starts[1:]
    ends[-1] = sorted_pos.size
    counts = ends - starts

    group_avgs = np.add.reduceat(setpoints[sorted_idx], starts) / counts
    group_id = np.repeat(np.arange(starts.size, dtype=np.intp), counts)
    reps = sorted_idx[starts]

    avg_set[sorted_idx] = group_avgs[group_id]
    rep_mask[reps] = True
    share_mask[sorted_idx] = True
    share_mask[reps] = False
    ref_idx[sorted_idx] = reps[group_id]
    return avg_set, rep_mask, share_mask, ref_idx


def build_zero_branch_components_fast(edge_a, edge_b, *, build_lists: bool = False):
    """基于稀疏连通分量分析与无环快路径构建零阻抗支路 phi 映射及生成树边。"""
    a = np.asarray(edge_a, dtype=np.int32)
    b = np.asarray(edge_b, dtype=np.int32)
    m = int(a.size)
    empty_i32 = np.empty(0, dtype=np.int32)
    if m == 0:
        return {
            "phi_node": empty_i32,
            "phi_comp": empty_i32,
            "ref_phi_idx": empty_i32,
            "phi_a": empty_i32,
            "phi_b": empty_i32,
            "tree_edge_indices": empty_i32,
            "n_comp": 0,
            "comp_nodes": [],
            "comp_tree_edges": [],
        }

    endpoints = np.concatenate((a, b))
    uniq_nodes, inv = np.unique(endpoints, return_inverse=True)
    k = int(uniq_nodes.size)
    la = inv[:m].astype(np.int32, copy=False)
    lb = inv[m:].astype(np.int32, copy=False)

    adj_rows = np.concatenate((la, lb))
    adj_cols = np.concatenate((lb, la))
    adj = csr_matrix(
        (np.ones(adj_rows.size, dtype=np.int8), (adj_rows, adj_cols)),
        shape=(k, k),
    )
    n_comp, comp_labels = connected_components(adj, directed=False)
    comp_labels = np.asarray(comp_labels, dtype=np.int32)

    node_order = np.argsort(comp_labels, kind="stable")
    phi_node = uniq_nodes[node_order].astype(np.int32, copy=False)
    phi_comp = comp_labels[node_order].astype(np.int32, copy=False)

    node_counts = np.bincount(comp_labels, minlength=n_comp)
    ref_phi_idx = np.empty(n_comp, dtype=np.int32)
    ref_phi_idx[0] = 0
    if n_comp > 1:
        ref_phi_idx[1:] = np.cumsum(node_counts[:-1], dtype=np.int32)

    local_to_phi = np.empty(k, dtype=np.int32)
    local_to_phi[node_order] = np.arange(k, dtype=np.int32)
    phi_a = local_to_phi[la]
    phi_b = local_to_phi[lb]

    edge_comp = comp_labels[la]
    edge_counts = np.bincount(edge_comp, minlength=n_comp)
    is_tree_comp = edge_counts == (node_counts - 1)

    if bool(np.all(is_tree_comp)):
        tree_edge_indices = np.arange(m, dtype=np.int32)
    else:
        tree_edge_mask = is_tree_comp[edge_comp]
        tree_parts = []
        if bool(np.any(tree_edge_mask)):
            tree_parts.append(np.flatnonzero(tree_edge_mask).astype(np.int32, copy=False))
        cyc_edge_idx = np.flatnonzero(~tree_edge_mask).astype(np.int32, copy=False)
        cyc_comp = edge_comp[cyc_edge_idx]
        c_order = np.argsort(cyc_comp, kind="stable")
        cyc_edge_sorted = cyc_edge_idx[c_order]
        cyc_comp_sorted = cyc_comp[c_order]
        is_new_c = np.empty(cyc_comp_sorted.size, dtype=bool)
        is_new_c[0] = True
        if cyc_comp_sorted.size > 1:
            is_new_c[1:] = cyc_comp_sorted[1:] != cyc_comp_sorted[:-1]
        c_starts = np.flatnonzero(is_new_c)
        c_ends = np.empty(c_starts.size, dtype=np.intp)
        if c_starts.size > 1:
            c_ends[:-1] = c_starts[1:]
        c_ends[-1] = cyc_comp_sorted.size
        for s_idx, e_idx in zip(c_starts.tolist(), c_ends.tolist()):
            c_id = int(cyc_comp_sorted[s_idx])
            base_phi = int(ref_phi_idx[c_id])
            c_n_nodes = int(node_counts[c_id])
            c_edges = cyc_edge_sorted[s_idx:e_idx]
            local_edges = list(
                zip(
                    (phi_a[c_edges] - base_phi).tolist(),
                    (phi_b[c_edges] - base_phi).tolist(),
                )
            )
            chosen_local = find_spanning_tree_edges(local_edges, c_n_nodes)
            if chosen_local:
                tree_parts.append(c_edges[np.asarray(chosen_local, dtype=np.intp)])
        tree_edge_indices = (
            np.concatenate(tree_parts).astype(np.int32, copy=False)
            if tree_parts
            else empty_i32
        )

    comp_nodes = []
    comp_tree_edges = []
    if build_lists:
        node_ends = np.empty(n_comp, dtype=np.intp)
        if n_comp > 1:
            node_ends[:-1] = ref_phi_idx[1:]
        node_ends[-1] = k
        phi_node_list = phi_node.tolist()
        for s_idx, e_idx in zip(ref_phi_idx.tolist(), node_ends.tolist()):
            comp_nodes.append(phi_node_list[s_idx:e_idx])
        if tree_edge_indices.size:
            t_comp = edge_comp[tree_edge_indices]
            t_order = np.argsort(t_comp, kind="stable")
            t_sorted = tree_edge_indices[t_order].tolist()
            t_comp_sorted = t_comp[t_order]
            comp_tree_edges = [[] for _ in range(n_comp)]
            for e_i, c_i in zip(t_sorted, t_comp_sorted.tolist()):
                comp_tree_edges[c_i].append(e_i)
        else:
            comp_tree_edges = [[] for _ in range(n_comp)]

    return {
        "phi_node": phi_node,
        "phi_comp": phi_comp,
        "ref_phi_idx": ref_phi_idx,
        "phi_a": phi_a,
        "phi_b": phi_b,
        "tree_edge_indices": tree_edge_indices,
        "n_comp": int(n_comp),
        "comp_nodes": comp_nodes,
        "comp_tree_edges": comp_tree_edges,
    }
