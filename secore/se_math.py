import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scipy.linalg import cho_factor as CHO_FACTOR
from scipy.linalg import cho_solve as CHO_SOLVE
from scipy.linalg import solve_triangular as SOLVE_TRIANGULAR
from scipy.linalg.lapack import dposv as DPOSV
from scipy.linalg.lapack import dpotrf as DPOTRF
from scipy.linalg.lapack import dpotri as DPOTRI
from scipy.linalg.lapack import dtrtri as DTRTRI
from scipy.sparse import coo_matrix as SP_COO_MATRIX
from scipy.sparse import csc_matrix as SP_CSC_MATRIX
from scipy.sparse import csr_matrix as SP_CSR_MATRIX
from scipy.sparse import eye as SP_EYE
from scipy.sparse import hstack as SP_HSTACK
from scipy.sparse import issparse as SP_ISSPARSE
from scipy.sparse import tril as SP_TRIL
from scipy.sparse.csgraph import connected_components as SP_CONNECTED_COMPONENTS
from scipy.sparse.csgraph import maximum_bipartite_matching as SP_MAX_BIPARTITE_MATCHING
from scipy.sparse.csgraph import structural_rank as SP_STRUCTURAL_RANK
from scipy.sparse.linalg import MatrixRankWarning as SP_MATRIX_RANK_WARNING
from scipy.sparse.linalg import eigsh as SP_EIGSH
from scipy.sparse.linalg import splu as SP_SPLU
from scipy.sparse.linalg import spsolve as SP_SPSOLVE

try:
    from sksparse.cholmod import cholesky as CHOLMOD_CHOLESKY
    from sksparse.cholmod import analyze as CHOLMOD_ANALYZE
except Exception:
    CHOLMOD_CHOLESKY = None
    CHOLMOD_ANALYZE = None

try:
    from sksparse.cholmod import cholesky_AAt as CHOLMOD_CHOLESKY_AAT
    from sksparse.cholmod import analyze_AAt as CHOLMOD_ANALYZE_AAT
except Exception:
    CHOLMOD_CHOLESKY_AAT = None
    CHOLMOD_ANALYZE_AAT = None

try:
    from .mkl_pardiso import (
        MKLPardisoFactor,
        MKLPardisoFactorizationError,
        build_upper_csr_from_aat,
        get_linear_solver_preference,
        is_mkl_pardiso_available,
    )
except Exception:
    MKLPardisoFactor = None
    MKLPardisoFactorizationError = None
    build_upper_csr_from_aat = None
    get_linear_solver_preference = lambda: "auto"  # noqa: E731
    is_mkl_pardiso_available = lambda: False  # noqa: E731

try:
    from .native_kernels import (
        leverage_from_selected_inv_native as NATIVE_LEVERAGE_FROM_SELECTED_INV,
        supernodal_selected_inverse_native as NATIVE_SUPERNODAL_SELECTED_INVERSE,
    )
except Exception:
    NATIVE_LEVERAGE_FROM_SELECTED_INV = None
    NATIVE_SUPERNODAL_SELECTED_INVERSE = None

# MKL PARDISO 符号分析存在固定初始化开销，仅在中大规模状态数上启用，小网保留 CHOLMOD/LAPACK 极速路径
_PARDISO_MIN_STATE_COUNT = 2000


_NORMAL_EQUATION_PATTERN_CACHE = {}
_NORMAL_EQUATION_PATTERN_IDENTITY_CACHE = []
_SPARSE_PATTERN_EXPANSION_CACHE = {}
_SPARSE_PATTERN_LINEAR_INDEX_CACHE = {}
_SPARSE_CSR_ROW_INDEX_CACHE = {}
_MAX_DIRECT_NORMAL_ASSEMBLY_PAIRS = 2_000_000


@dataclass(frozen=True)
class SEScaleProfile:
    """状态估计按规模与稀疏度自适应调节的内部调度参数包。

    依据状态维数 ``n_state``、量测行数 ``n_meas``、雅可比非零元数 ``nnz_h``（平均每状态非零元
    ``avg_nnz = nnz_h / max(1, n_state)``）及是否多轮重复分解 ``multi_solve``，自动推导全套
    数值分解、杠杆计算、低秩 Woodbury 与坏数据批删参数，兼顾 20 节点至 100,000+ 节点及
    极稀疏直流网 / 稠密交流与混联电网。
    """

    n_state: int
    n_meas: int
    avg_nnz_per_state: float
    use_aat_normal_solver: bool
    aat_ordering_method: str
    leverage_eye_max_state: int
    sparse_leverage_max_state: int
    exact_leverage_chunk_size: int
    exact_leverage_use_sparse_rhs: bool
    selinv_supernodal_rebuild_min_state: int
    woodbury_max_rank: int
    woodbury_self_cert_min_state: int
    batch_max_base: int
    fast_init_screen_enabled: bool
    direct_strong_peaks_enabled: bool
    coarse_early_stop_enabled: bool
    screen_primary_cap: int
    screen_verify_cap: int
    screen_verify_cap_rich: int
    screen_selinv_switch_k: int
    frozen_max_steps: int
    local_warm_frozen_max_dx: float
    local_warm_frozen_max_rms: float
    local_warm_frozen_max_steps: int
    woodbury_max_steps: int

    @classmethod
    def for_problem(
        cls,
        n_state: int,
        n_meas: int = 0,
        nnz_h: int = 0,
        *,
        multi_solve: bool = False,
    ) -> "SEScaleProfile":
        """根据问题规模、雅可比稀疏度与求解模式推导自适应策略参数。"""
        ns = max(0, int(n_state))
        nm = max(0, int(n_meas))
        nnz = max(0, int(nnz_h))
        avg_nnz = (float(nnz) / float(ns)) if (ns > 0 and nnz > 0) else 0.0
        is_ultra_sparse = (avg_nnz > 0.0) and (avg_nnz < 8.0)

        # 大网稠密耦合系统在多轮分解或超大规模时选用嵌套剖分 nesdis，大幅降低 L 填充；
        # 小中规模与稀疏直流网（如 DC 20k 满测 avg_nnz ≈ 9.4 < 10.5）保持默认 AMD，避免多余的符号分析耗时。
        if (multi_solve and ns >= 10000 and avg_nnz >= 10.5) or (ns >= 30000 and avg_nnz >= 10.5):
            ordering = "nesdis"
        else:
            ordering = "default"

        # L3 Cache 字节预算（约 24 MB）自适应决定 solve_L 稠密分块列数，钳位在 [16, 256]
        if ns > 0:
            chunk = max(16, min(256, int((24 * 1024 * 1024) // (8 * ns))))
        else:
            chunk = 64

        # 稀疏图上 fac(I) 与超节点选元求逆的交叉点更高
        lev_eye_max = 768 if is_ultra_sparse else 512

        # 跨轮持久锚点低秩 Woodbury 最大秩随 sqrt(n) 平滑缩放：小网不做重型高秩更新，大网拉长免分解窗口
        wb_max_rank = min(48, max(4, int(round(math.sqrt(max(1, ns)) / 4.0))))

        # 高置信度独立批删单轮上限与初筛局部峰值/回代上限按状态数自适应缩放（覆盖 10 ~ 100k+ 节点）
        batch_max = max(16, min(8192, ns // 16)) if ns >= 384 else max(16, ns // 8)
        primary_cap = min(16384, max(256, ns // 10))
        # 单轮精确回代上限与全网选元求逆切换点解耦：
        # - 大网回代上限 384 可单轮清空中等盲区中段 200~384 条候选，避免触发冗余的提前全网选元求逆；
        # - 仅当无足够免回代强峰且未截断待回代局部峰值 >= 512（与全网选元求逆实测交叉点齐平）时自动切全网选元求逆；
        # - 当免回代强峰已足够丰富时，大网/超大网将附加回代压缩至 1 个缓存分块（16~32 条）。
        verify_cap = 384 if ns > 5000 else min(192, max(128, ns // 1024))
        selinv_switch_k = 512
        if ns > 60000:
            verify_cap_rich = 16
        elif ns > 5000:
            verify_cap_rich = 32
        else:
            verify_cap_rich = 128

        return cls(
            n_state=ns,
            n_meas=nm,
            avg_nnz_per_state=avg_nnz,
            use_aat_normal_solver=(ns >= 16),
            aat_ordering_method=ordering,
            leverage_eye_max_state=lev_eye_max,
            sparse_leverage_max_state=500_000,
            exact_leverage_chunk_size=chunk,
            exact_leverage_use_sparse_rhs=(ns <= 20000 and avg_nnz > 0.0 and avg_nnz < 10.0),
            selinv_supernodal_rebuild_min_state=2000,
            woodbury_max_rank=wb_max_rank,
            woodbury_self_cert_min_state=16,
            batch_max_base=batch_max,
            fast_init_screen_enabled=(ns > 512),
            direct_strong_peaks_enabled=(ns > 5000),
            coarse_early_stop_enabled=(ns > 5000),
            screen_primary_cap=primary_cap,
            screen_verify_cap=verify_cap,
            screen_verify_cap_rich=verify_cap_rich,
            screen_selinv_switch_k=selinv_switch_k,
            frozen_max_steps=5,
            local_warm_frozen_max_dx=0.60,
            local_warm_frozen_max_rms=2e-2,
            local_warm_frozen_max_steps=4,
            woodbury_max_steps=6,
        )

def targeted_redundancy_count(state_count: int, ratio: float) -> int:
    """Return the configured pseudo-measurement redundancy target as a row count."""
    return max(0, int(math.ceil(max(0.0, float(ratio)) * max(0, int(state_count)))))


def observability_weak_direction(
    H,
    state_count: int,
    weak_states: Sequence[Tuple[int, float]] = (),
    dense_svd_limit: int = 256,
) -> np.ndarray:
    """Return a normalized state-space direction that is currently weakest."""
    state_count = int(state_count)
    if state_count <= 0:
        return np.array([], dtype=np.float64)

    direction = np.zeros(state_count, dtype=np.float64)
    if weak_states:
        valid_positions: List[int] = []
        for state_idx, score in weak_states:
            try:
                pos = int(state_idx)
            except (TypeError, ValueError):
                continue
            if 0 <= pos < state_count:
                valid_positions.append(pos)
                direction[pos] = max(abs(float(score)), direction[pos])
        norm = float(np.linalg.norm(direction))
        if norm > 0.0:
            return direction / norm
        # 若弱状态列表全为零覆盖状态（score == 0.0），直接指向这些状态以避免退化到稠密 SVD
        if valid_positions:
            direction[np.asarray(valid_positions, dtype=np.intp)] = 1.0
            return direction / float(np.linalg.norm(direction))

    if H is None or H.shape[0] == 0:
        direction[0] = 1.0
        return direction

    try:
        if state_count <= dense_svd_limit:
            H_arr = H.toarray() if is_sparse_matrix(H) else np.asarray(H, dtype=np.float64)
            if H_arr.size:
                _u, _s, vh = np.linalg.svd(H_arr, full_matrices=False)
                if vh.size:
                    direction = np.asarray(vh[-1, :], dtype=np.float64)
                    norm = float(np.linalg.norm(direction))
                    if norm > 0.0:
                        return direction / norm
        gram = H.T @ H
        if is_sparse_matrix(gram):
            values, vectors = SP_EIGSH(gram.tocsc(), k=1, which="SM", tol=1e-6, maxiter=500)
            if values.size and vectors.size:
                direction = np.asarray(vectors[:, 0], dtype=np.float64)
                norm = float(np.linalg.norm(direction))
                if norm > 0.0:
                    return direction / norm
        else:
            values, vectors = np.linalg.eigh(np.asarray(gram, dtype=np.float64))
            if values.size and vectors.size:
                direction = np.asarray(vectors[:, int(np.argmin(values))], dtype=np.float64)
                norm = float(np.linalg.norm(direction))
                if norm > 0.0:
                    return direction / norm
    except Exception:
        pass

    gram = H.T @ H
    diag = np.asarray(gram.diagonal() if is_sparse_matrix(gram) else np.diag(gram), dtype=np.float64)
    if diag.size:
        direction[int(np.argmin(diag))] = 1.0
    else:
        direction[0] = 1.0
    return direction


def _as_array_dtype(values, dtype):
    if isinstance(values, np.ndarray) and values.dtype == dtype:
        return values
    return np.asarray(values, dtype=dtype)


def measurement_residual(
    z: np.ndarray,
    z_est: np.ndarray,
) -> np.ndarray:
    """Return z - h(x)."""
    return np.asarray(z, dtype=np.float64) - np.asarray(z_est, dtype=np.float64)


class SparseJacobianBuilder:
    """Collect Jacobian entries as COO triplets while preserving ndarray-style writes.

    The state estimators already express every analytical derivative as sparse
    row/column writes. This adapter lets those writes create sparse triplets
    directly instead of first allocating a mostly-zero dense measurement matrix.
    """

    def __init__(self, shape: Tuple[int, int]):
        self.shape = tuple(int(item) for item in shape)
        self.size = self.shape[0] * self.shape[1]
        self._cached_pattern_linear: Optional[np.ndarray] = None
        self._cached_pattern_indptr: Optional[np.ndarray] = None
        self._cached_pattern_indices: Optional[np.ndarray] = None
        self._cached_slot_positions: Optional[np.ndarray] = None
        self._cached_unique_slot_mask: Optional[np.ndarray] = None
        self._cached_duplicate_slot_mask: Optional[np.ndarray] = None
        self._cached_unique_slots: Optional[np.ndarray] = None
        self._cached_unique_data_positions: Optional[np.ndarray] = None
        self._cached_duplicate_slots: Optional[np.ndarray] = None
        self._cached_duplicate_data_positions: Optional[np.ndarray] = None
        self._cached_chunk_slices: Optional[List[Tuple[int, int]]] = None
        self._cached_chunk_slot_plans: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None
        self._cached_direct_chunk_slots: Optional[List[Optional[np.ndarray]]] = None
        self._cached_expanded_direct_slots: Dict[int, np.ndarray] = {}
        self._cached_duplicate_dest_slots: Optional[np.ndarray] = None
        self._cached_slots_cover_pattern = False
        self._cached_has_unique_slots = False
        self._cached_has_duplicate_slots = False
        self._cached_csr_data: Optional[np.ndarray] = None
        self._cached_csr_matrix = None
        self._data_buffer: Optional[np.ndarray] = None
        self._assume_fixed_pattern = False
        self._data_only_refresh_enabled = False
        self._data_only_refresh_active = False
        self._data_only_direct_active = False
        self._data_only_direct_initialized = False
        self._data_only_chunk_cursor = 0
        self._recorded_add_many_specs: List[Tuple[object, object, int, Optional[np.ndarray]]] = []
        self._cached_add_many_specs: Optional[List[Tuple[object, object, int, Optional[np.ndarray]]]] = None
        self._add_many_call_cursor = 0
        self._add_many_fast_path_ok = False
        self.reset()

    def _set_cached_slot_positions(self, slot: np.ndarray) -> None:
        self._cached_slot_positions = slot
        self._cached_chunk_slices = self._current_chunk_slices()
        self._cached_add_many_specs = list(self._recorded_add_many_specs) if self._recorded_add_many_specs else None
        self._cached_chunk_slot_plans = []
        self._cached_direct_chunk_slots = []
        self._cached_expanded_direct_slots = {}
        if slot.size == 0:
            self._cached_unique_slot_mask = np.array([], dtype=bool)
            self._cached_duplicate_slot_mask = np.array([], dtype=bool)
            self._cached_unique_slots = np.array([], dtype=np.int64)
            self._cached_unique_data_positions = np.array([], dtype=np.int64)
            self._cached_duplicate_slots = np.array([], dtype=np.int64)
            self._cached_duplicate_data_positions = np.array([], dtype=np.int64)
            self._cached_duplicate_dest_slots = np.array([], dtype=np.int64)
            self._cached_slots_cover_pattern = self._cached_pattern_linear is not None and self._cached_pattern_linear.size == 0
            self._cached_has_unique_slots = False
            self._cached_has_duplicate_slots = False
            return
        counts = np.bincount(slot, minlength=int(slot.max()) + 1)
        self._cached_unique_slot_mask = counts[slot] == 1
        self._cached_duplicate_slot_mask = ~self._cached_unique_slot_mask
        self._cached_has_unique_slots = bool(np.any(self._cached_unique_slot_mask))
        self._cached_has_duplicate_slots = bool(np.any(self._cached_duplicate_slot_mask))
        if self._cached_has_unique_slots:
            self._cached_unique_data_positions = np.nonzero(self._cached_unique_slot_mask)[0].astype(np.int64, copy=False)
            self._cached_unique_slots = slot[self._cached_unique_data_positions]
        else:
            self._cached_unique_data_positions = np.array([], dtype=np.int64)
            self._cached_unique_slots = np.array([], dtype=np.int64)
        if self._cached_has_duplicate_slots:
            self._cached_duplicate_data_positions = np.nonzero(self._cached_duplicate_slot_mask)[0].astype(
                np.int64,
                copy=False,
            )
            self._cached_duplicate_slots = slot[self._cached_duplicate_data_positions]
            self._cached_duplicate_dest_slots = np.flatnonzero(counts > 1).astype(np.int64, copy=False)
        else:
            self._cached_duplicate_data_positions = np.array([], dtype=np.int64)
            self._cached_duplicate_slots = np.array([], dtype=np.int64)
            self._cached_duplicate_dest_slots = np.array([], dtype=np.int64)
        self._cached_slots_cover_pattern = (
            self._cached_pattern_linear is not None
            and np.count_nonzero(counts) == int(self._cached_pattern_linear.size)
        )
        for start, end in self._cached_chunk_slices:
            if start == end:
                empty = np.array([], dtype=np.int64)
                self._cached_chunk_slot_plans.append((empty, empty, empty, empty))
                self._cached_direct_chunk_slots.append(empty)
                continue
            chunk_unique_mask = self._cached_unique_slot_mask[start:end]
            unique_pos = np.nonzero(chunk_unique_mask)[0].astype(np.int64, copy=False)
            duplicate_pos = np.nonzero(~chunk_unique_mask)[0].astype(np.int64, copy=False)
            chunk_slots = slot[start:end]
            unique_slots = chunk_slots[unique_pos].astype(np.int64, copy=False)
            duplicate_slots = chunk_slots[duplicate_pos].astype(np.int64, copy=False)
            self._cached_chunk_slot_plans.append((unique_pos, unique_slots, duplicate_pos, duplicate_slots))
            if duplicate_pos.size == 0 and unique_pos.size == chunk_slots.size:
                self._cached_direct_chunk_slots.append(chunk_slots.astype(np.int64, copy=False))
            else:
                self._cached_direct_chunk_slots.append(None)

    def _current_chunk_slices(self) -> List[Tuple[int, int]]:
        slices: List[Tuple[int, int]] = []
        cursor = len(self.data)
        for chunk in self._data_chunks:
            size = int(chunk.size)
            slices.append((cursor, cursor + size))
            cursor += size
        return slices

    def reset(self) -> None:
        self.rows = []
        self.cols = []
        self.data = []
        self._row_chunks = []
        self._col_chunks = []
        self._data_chunks = []
        self._data_only_chunk_cursor = 0
        self._recorded_add_many_specs = []
        self._add_many_call_cursor = 0
        self._data_only_refresh_active = bool(
            self._data_only_refresh_enabled
            and self._assume_fixed_pattern
            and self._cached_pattern_indptr is not None
            and self._cached_pattern_indices is not None
            and self._cached_chunk_slices is not None
        )
        self._add_many_fast_path_ok = bool(
            self._data_only_refresh_active and self._cached_add_many_specs is not None
        )
        first_chunk_starts_after_scalars = (
            self._cached_chunk_slices is not None
            and len(self._cached_chunk_slices) > 0
            and int(self._cached_chunk_slices[0][0]) == 0
        )
        self._data_only_direct_active = bool(self._data_only_refresh_active and first_chunk_starts_after_scalars)
        self._data_only_direct_initialized = False

    def _initialize_data_only_direct_buffer(self) -> Optional[np.ndarray]:
        if not self._data_only_direct_active:
            return None
        pattern = self._cached_pattern_linear
        if pattern is None:
            return None
        if self._cached_csr_data is None or self._cached_csr_data.size != pattern.size:
            self._cached_csr_data = np.zeros(pattern.size, dtype=np.float64)
            self._cached_csr_matrix = None
        values = self._cached_csr_data
        if not self._data_only_direct_initialized:
            if not self._cached_slots_cover_pattern:
                values.fill(0.0)
            else:
                duplicate_dest_slots = self._cached_duplicate_dest_slots
                if duplicate_dest_slots is not None and duplicate_dest_slots.size:
                    values[duplicate_dest_slots] = 0.0
            self._data_only_direct_initialized = True
        return values

    def _append_data_only_chunk(self, values: np.ndarray) -> bool:
        if not self._data_only_refresh_active:
            return False
        chunk_slices = self._cached_chunk_slices
        if chunk_slices is None or self._data_only_chunk_cursor >= len(chunk_slices):
            raise RuntimeError("SparseJacobianBuilder fixed CSR data refresh received too many chunks")
        start, end = chunk_slices[self._data_only_chunk_cursor]
        expected = int(end) - int(start)
        if int(values.size) != expected:
            raise RuntimeError(
                "SparseJacobianBuilder fixed CSR data refresh pattern changed: "
                f"expected chunk size {expected}, got {int(values.size)}"
            )
        values = _as_array_dtype(values, np.float64)
        target = self._initialize_data_only_direct_buffer()
        if target is not None:
            direct_slots = (
                self._cached_direct_chunk_slots[self._data_only_chunk_cursor]
                if self._cached_direct_chunk_slots is not None
                and self._data_only_chunk_cursor < len(self._cached_direct_chunk_slots)
                else None
            )
            chunk_plan = (
                self._cached_chunk_slot_plans[self._data_only_chunk_cursor]
                if self._cached_chunk_slot_plans is not None
                and self._data_only_chunk_cursor < len(self._cached_chunk_slot_plans)
                else None
            )
            if direct_slots is not None:
                if direct_slots.size:
                    target[direct_slots] = values
            elif chunk_plan is not None:
                unique_pos, unique_slots, duplicate_pos, duplicate_slots = chunk_plan
                if unique_pos.size:
                    target[unique_slots] = values[unique_pos]
                if duplicate_pos.size:
                    np.add.at(target, duplicate_slots, values[duplicate_pos])
            else:
                slot = self._cached_slot_positions[int(start) : int(end)]
                np.add.at(target, slot, values)
        else:
            self._data_chunks.append(values)
        self._data_only_chunk_cursor += 1
        return True

    def _try_append_fixed_data_chunk(self, values) -> bool:
        """Fast path for fixed CSR refresh when the current block shape is unchanged."""
        if not self._data_only_refresh_active:
            return False
        chunk_slices = self._cached_chunk_slices
        if chunk_slices is None or self._data_only_chunk_cursor >= len(chunk_slices):
            return False
        start, end = chunk_slices[self._data_only_chunk_cursor]
        expected = int(end) - int(start)
        value_size = int(values.size) if isinstance(values, np.ndarray) else int(np.asarray(values).size)
        if value_size != expected:
            return False
        return self._append_data_only_chunk(values)

    def _append_arrays(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        """Keep vectorized writes as NumPy chunks until final COO/CSR construction."""
        if rows.size == 0:
            return
        rows = _as_array_dtype(rows, np.int32)
        cols = _as_array_dtype(cols, np.int32)
        values = _as_array_dtype(values, np.float64)
        mask = cols >= 0
        if not mask.any():
            return
        if mask.all():
            if self._append_data_only_chunk(values):
                return
            self._row_chunks.append(rows)
            self._col_chunks.append(cols)
            self._data_chunks.append(values)
            return
        masked_values = values[mask].astype(np.float64, copy=False)
        if self._append_data_only_chunk(masked_values):
            return
        self._row_chunks.append(rows[mask].astype(np.int32, copy=False))
        self._col_chunks.append(cols[mask].astype(np.int32, copy=False))
        self._data_chunks.append(masked_values)

    def _coo_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.rows and not self._row_chunks:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
            )
        scalar_count = len(self.rows)
        chunk_sizes = [int(chunk.size) for chunk in self._row_chunks]
        total_count = scalar_count + sum(chunk_sizes)
        if total_count == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
            )
        out_rows = np.empty(total_count, dtype=np.int32)
        out_cols = np.empty(total_count, dtype=np.int32)
        out_data = np.empty(total_count, dtype=np.float64)
        cursor = 0
        if scalar_count:
            end = scalar_count
            out_rows[:end] = np.asarray(self.rows, dtype=np.int32)
            out_cols[:end] = np.asarray(self.cols, dtype=np.int32)
            out_data[:end] = np.asarray(self.data, dtype=np.float64)
            cursor = end
        for rows, cols, data, size in zip(self._row_chunks, self._col_chunks, self._data_chunks, chunk_sizes):
            if size == 0:
                continue
            end = cursor + size
            out_rows[cursor:end] = rows
            out_cols[cursor:end] = cols
            out_data[cursor:end] = data
            cursor = end
        if cursor != total_count:
            return out_rows[:cursor], out_cols[:cursor], out_data[:cursor]
        return out_rows, out_cols, out_data

    def _data_array(self) -> np.ndarray:
        if not self.data and not self._data_chunks:
            return np.array([], dtype=np.float64)
        scalar_count = len(self.data)
        chunk_sizes = [int(chunk.size) for chunk in self._data_chunks]
        total_count = scalar_count + sum(chunk_sizes)
        if self._data_buffer is None or self._data_buffer.size < total_count:
            self._data_buffer = np.empty(total_count, dtype=np.float64)
        out_data = self._data_buffer
        cursor = 0
        if scalar_count:
            out_data[:scalar_count] = np.asarray(self.data, dtype=np.float64)
            cursor = scalar_count
        for data, size in zip(self._data_chunks, chunk_sizes):
            if size == 0:
                continue
            end = cursor + size
            out_data[cursor:end] = data
            cursor = end
        if cursor != total_count:
            return out_data[:cursor]
        return out_data[:total_count]

    def _refresh_fixed_pattern_values(self, values: np.ndarray) -> None:
        scalar_count = len(self.data)
        if scalar_count or not self._cached_slots_cover_pattern:
            values.fill(0.0)
        else:
            duplicate_dest_slots = self._cached_duplicate_dest_slots
            if duplicate_dest_slots is not None and duplicate_dest_slots.size:
                values[duplicate_dest_slots] = 0.0
        if scalar_count:
            scalar_values = np.asarray(self.data, dtype=np.float64)
            np.add.at(values, self._cached_slot_positions[:scalar_count], scalar_values)
        chunk_slices = self._cached_chunk_slices
        chunk_slot_plans = self._cached_chunk_slot_plans
        if chunk_slices is None or len(chunk_slices) != len(self._data_chunks):
            data = self._data_array()
            unique_mask = self._cached_unique_slot_mask
            if unique_mask is None or unique_mask.size != self._cached_slot_positions.size:
                np.add.at(values, self._cached_slot_positions, data)
                return
            if self._cached_has_unique_slots:
                values[self._cached_unique_slots] = data[self._cached_unique_data_positions]
            if self._cached_has_duplicate_slots:
                np.add.at(values, self._cached_duplicate_slots, data[self._cached_duplicate_data_positions])
            return
        if chunk_slot_plans is not None and len(chunk_slot_plans) == len(self._data_chunks):
            direct_chunk_slots = self._cached_direct_chunk_slots
            if direct_chunk_slots is not None and len(direct_chunk_slots) == len(self._data_chunks):
                for chunk, direct_slots, (unique_pos, unique_slots, duplicate_pos, duplicate_slots) in zip(
                    self._data_chunks,
                    direct_chunk_slots,
                    chunk_slot_plans,
                ):
                    if direct_slots is not None:
                        if direct_slots.size:
                            values[direct_slots] = chunk
                        continue
                    if unique_pos.size:
                        values[unique_slots] = chunk[unique_pos]
                    if duplicate_pos.size:
                        np.add.at(values, duplicate_slots, chunk[duplicate_pos])
                return
            for chunk, (unique_pos, unique_slots, duplicate_pos, duplicate_slots) in zip(
                self._data_chunks,
                chunk_slot_plans,
            ):
                if unique_pos.size:
                    values[unique_slots] = chunk[unique_pos]
                if duplicate_pos.size:
                    np.add.at(values, duplicate_slots, chunk[duplicate_pos])
            return
        for chunk, (start, end) in zip(self._data_chunks, chunk_slices):
            if start == end:
                continue
            np.add.at(values, self._cached_slot_positions[start:end], chunk)

    def __getitem__(self, key):
        row_key, col_key = key
        if isinstance(col_key, slice):
            if col_key != slice(None):
                raise NotImplementedError("SparseJacobianBuilder only supports full-column slices")
            return np.zeros(self.shape[1], dtype=np.float64)

        rows = np.asarray(row_key)
        cols = np.asarray(col_key)
        if rows.ndim == 0 and cols.ndim == 0:
            return 0.0
        broadcast_rows, _broadcast_cols = np.broadcast_arrays(rows, cols)
        return np.zeros(broadcast_rows.shape, dtype=np.float64)

    def __setitem__(self, key, value) -> None:
        row_key, col_key = key
        if isinstance(col_key, slice):
            if col_key != slice(None):
                raise NotImplementedError("SparseJacobianBuilder only supports full-column slices")
            values = np.asarray(value, dtype=np.float64)
            if values.ndim != 1 or values.size != self.shape[1]:
                raise ValueError("Full-row sparse assignment requires a vector with n_state entries")
            cols = np.nonzero(values)[0].astype(np.int32)
            if cols.size:
                self._append_arrays(
                    np.full(cols.size, int(row_key), dtype=np.int32),
                    cols,
                    values[cols].astype(np.float64, copy=False),
                )
            return

        if np.isscalar(row_key) and np.isscalar(col_key) and np.isscalar(value):
            col = int(col_key)
            if col >= 0:
                self.rows.append(int(row_key))
                self.cols.append(col)
                self.data.append(float(value))
            return

        rows = np.asarray(row_key)
        cols = np.asarray(col_key)
        values = np.asarray(value, dtype=np.float64)
        broadcast_rows, broadcast_cols, broadcast_values = np.broadcast_arrays(rows, cols, values)
        flat_rows = broadcast_rows.ravel().astype(np.int32)
        flat_cols = broadcast_cols.ravel().astype(np.int32)
        flat_values = broadcast_values.ravel().astype(np.float64)
        mask = flat_cols >= 0
        if np.any(mask):
            self._append_arrays(flat_rows[mask], flat_cols[mask], flat_values[mask])

    def add(self, row: int, col: int, value: float) -> None:
        if col >= 0:
            self.rows.append(int(row))
            self.cols.append(int(col))
            self.data.append(float(value))

    def add_many(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray, mask: np.ndarray = None) -> None:
        # 固定稀疏模式刷新快路径：若 cols 与 mask 对象身份与录制期一致，直接跳过类型转换与 cols>=0/mask 过滤
        if self._add_many_fast_path_ok:
            specs = self._cached_add_many_specs
            cursor = self._add_many_call_cursor
            if specs is not None and cursor < len(specs):
                rec_cols, rec_mask, kind, rec_idx = specs[cursor]
                self._add_many_call_cursor = cursor + 1
                if cols is rec_cols and mask is rec_mask:
                    if kind == 0:
                        return
                    if kind == 1:
                        if self._append_data_only_chunk(values):
                            return
                    elif kind == 2 and rec_idx is not None:
                        val_arr = _as_array_dtype(values, np.float64)
                        if self._append_data_only_chunk(val_arr[rec_idx]):
                            return
            else:
                self._add_many_fast_path_ok = False

        raw_cols = cols
        raw_mask = mask
        rows = _as_array_dtype(rows, np.int32)
        cols = _as_array_dtype(cols, np.int32)
        values = _as_array_dtype(values, np.float64)
        if mask is None:
            mask = cols >= 0
            if mask.all() and self._try_append_fixed_data_chunk(values):
                return
        else:
            mask = np.asarray(mask, dtype=bool) & (cols >= 0)
        if mask.any():
            if mask.all():
                if not self._data_only_refresh_active:
                    self._recorded_add_many_specs.append((raw_cols, raw_mask, 1, None))
                if self._append_data_only_chunk(values):
                    return
                self._row_chunks.append(rows)
                self._col_chunks.append(cols)
                self._data_chunks.append(values)
                return
            if not self._data_only_refresh_active:
                rec_idx = np.flatnonzero(mask).astype(np.intp, copy=False)
                self._recorded_add_many_specs.append((raw_cols, raw_mask, 2, rec_idx))
                masked_values = values[rec_idx].astype(np.float64, copy=False)
                if self._append_data_only_chunk(masked_values):
                    return
                self._row_chunks.append(rows[rec_idx].astype(np.int32, copy=False))
                self._col_chunks.append(cols[rec_idx].astype(np.int32, copy=False))
                self._data_chunks.append(masked_values)
                return
            masked_values = values[mask].astype(np.float64, copy=False)
            if self._append_data_only_chunk(masked_values):
                return
            self._row_chunks.append(rows[mask].astype(np.int32, copy=False))
            self._col_chunks.append(cols[mask].astype(np.int32, copy=False))
            self._data_chunks.append(masked_values)
        elif not self._data_only_refresh_active:
            self._recorded_add_many_specs.append((raw_cols, raw_mask, 0, None))

    def try_claim_direct_scatter(
        self,
        block_checks: Sequence[Tuple[object, object, int]],
    ) -> Optional[Tuple[np.ndarray, List[np.ndarray]]]:
        """固定稀疏模式直写协商：若接下来连续 K 块均无重复槽位且模式与录制期一致，
        直接返回 ``(target_csr_data, [slots_0, ..., slots_{K-1}])`` 供原生核原位写入并推进游标。
        若任一条件不满足则保持游标不变并返回 ``None``，调用方无缝回退常规路径。
        """
        if not self._add_many_fast_path_ok or not self._data_only_direct_active:
            return None
        specs = self._cached_add_many_specs
        direct_list = self._cached_direct_chunk_slots
        chunk_slices = self._cached_chunk_slices
        if specs is None or direct_list is None or chunk_slices is None:
            return None
        n_blocks = len(block_checks)
        spec_cur = self._add_many_call_cursor
        if spec_cur + n_blocks > len(specs):
            return None
        chunk_cur = self._data_only_chunk_cursor
        slots_out: List[np.ndarray] = []
        empty_slots = np.empty(0, dtype=np.int64)
        for i in range(n_blocks):
            cols_ref, mask_ref, arr_len = block_checks[i]
            rec_cols, rec_mask, kind, rec_idx = specs[spec_cur + i]
            if cols_ref is not rec_cols or mask_ref is not rec_mask:
                return None
            if kind == 0:
                slots_out.append(empty_slots)
                continue
            if chunk_cur >= len(direct_list) or chunk_cur >= len(chunk_slices):
                return None
            d_slots = direct_list[chunk_cur]
            if d_slots is None:
                return None
            start, end = chunk_slices[chunk_cur]
            expected = int(end) - int(start)
            if kind == 1:
                if int(arr_len) != expected or int(d_slots.size) != int(arr_len):
                    return None
                slots_out.append(d_slots)
            elif kind == 2:
                if rec_idx is None or int(rec_idx.size) != expected or int(d_slots.size) != expected:
                    return None
                expanded = self._cached_expanded_direct_slots.get(chunk_cur)
                if expanded is None or int(expanded.size) != int(arr_len):
                    expanded = np.full(int(arr_len), -1, dtype=np.int64)
                    expanded[rec_idx] = d_slots
                    self._cached_expanded_direct_slots[chunk_cur] = expanded
                slots_out.append(expanded)
            else:
                return None
            chunk_cur += 1

        target = self._initialize_data_only_direct_buffer()
        if target is None:
            return None
        self._add_many_call_cursor = spec_cur + n_blocks
        self._data_only_chunk_cursor = chunk_cur
        return target, slots_out

    def to_csr(self):
        if self._assume_fixed_pattern and self._cached_slot_positions is not None and self._cached_pattern_indptr is not None and self._cached_pattern_indices is not None:
            if self._data_only_refresh_active and self._cached_chunk_slices is not None:
                expected_chunks = len(self._cached_chunk_slices)
                if self._data_only_chunk_cursor != expected_chunks:
                    raise RuntimeError(
                        "SparseJacobianBuilder fixed CSR data refresh pattern changed: "
                        f"expected {expected_chunks} chunks, got {self._data_only_chunk_cursor}"
                    )
                if self._data_only_direct_initialized and self._cached_csr_data is not None:
                    if self.rows:
                        n_cols = int(self.shape[1])
                        rows = np.asarray(self.rows, dtype=np.int64)
                        cols = np.asarray(self.cols, dtype=np.int64)
                        linear = rows * np.int64(n_cols) + cols
                        slot = np.searchsorted(self._cached_pattern_linear, linear)
                        if (
                            slot.size
                            and int(slot.max()) < self._cached_pattern_linear.size
                            and np.array_equal(self._cached_pattern_linear[slot], linear)
                        ):
                            np.add.at(self._cached_csr_data, slot, np.asarray(self.data, dtype=np.float64))
                        elif slot.size:
                            raise RuntimeError("SparseJacobianBuilder fixed CSR data refresh scalar pattern changed")
                    if self._cached_csr_matrix is None:
                        self._cached_csr_matrix = SP_CSR_MATRIX(
                            (self._cached_csr_data, self._cached_pattern_indices, self._cached_pattern_indptr),
                            shape=self.shape,
                            copy=False,
                        )
                    return self._cached_csr_matrix
            if self._cached_csr_data is None or self._cached_csr_data.size != self._cached_pattern_linear.size:
                self._cached_csr_data = np.zeros(self._cached_pattern_linear.size, dtype=np.float64)
                self._cached_csr_matrix = None
            values = self._cached_csr_data
            self._refresh_fixed_pattern_values(values)
            if self._cached_csr_matrix is None:
                self._cached_csr_matrix = SP_CSR_MATRIX(
                    (values, self._cached_pattern_indices, self._cached_pattern_indptr),
                    shape=self.shape,
                    copy=False,
                )
            return self._cached_csr_matrix
        rows, cols, data = self._coo_arrays()
        if self._cached_pattern_linear is not None and self._cached_pattern_indptr is not None and self._cached_pattern_indices is not None:
            n_cols = int(self.shape[1])
            linear = rows.astype(np.int64, copy=False) * n_cols + cols.astype(np.int64, copy=False)
            slot = np.searchsorted(self._cached_pattern_linear, linear)
            if slot.size and int(slot.max()) < self._cached_pattern_linear.size and np.array_equal(
                self._cached_pattern_linear[slot],
                linear,
            ):
                self._set_cached_slot_positions(slot)
                values = np.zeros(self._cached_pattern_linear.size, dtype=np.float64)
                np.add.at(values, slot, data)
                return SP_CSR_MATRIX(
                    (values, self._cached_pattern_indices, self._cached_pattern_indptr),
                    shape=self.shape,
                    copy=False,
                )
        n_rows = int(self.shape[0])
        n_cols = int(self.shape[1])
        if self._assume_fixed_pattern and rows.size > 0 and n_cols > 0:
            linear = rows.astype(np.int64, copy=False) * np.int64(n_cols) + cols.astype(np.int64, copy=False)
            order = np.argsort(linear, kind="quicksort")
            sorted_linear = linear[order]
            is_new = np.empty(rows.size, dtype=bool)
            is_new[0] = True
            is_new[1:] = sorted_linear[1:] != sorted_linear[:-1]
            pattern_linear = sorted_linear[is_new]
            slot = np.empty(rows.size, dtype=np.int64)
            slot[order] = np.cumsum(is_new, dtype=np.int64) - 1
            unique_rows = (pattern_linear // np.int64(n_cols)).astype(np.intp, copy=False)
            indices = (pattern_linear - unique_rows.astype(np.int64, copy=False) * np.int64(n_cols)).astype(
                np.int32,
                copy=False,
            )
            indptr = np.empty(n_rows + 1, dtype=np.int32)
            indptr[0] = 0
            indptr[1:] = np.cumsum(np.bincount(unique_rows, minlength=n_rows), dtype=np.int64)
            csr_data = np.bincount(slot, weights=data, minlength=indices.size)
            csr = SP_CSR_MATRIX((csr_data, indices, indptr), shape=self.shape, copy=False)
            csr.has_sorted_indices = True
            self._cached_pattern_linear = pattern_linear
            self._cached_pattern_indptr = indptr
            self._cached_pattern_indices = indices
            self._set_cached_slot_positions(slot)
            self._cached_csr_data = csr_data
            self._cached_csr_matrix = csr
            self._data_only_refresh_enabled = True
            return csr

        csr = SP_COO_MATRIX((data, (rows, cols)), shape=self.shape).tocsr(copy=False)
        indptr = csr.indptr.astype(np.int32, copy=True)
        indices = csr.indices.astype(np.int32, copy=True)
        pattern_rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(indptr).astype(np.int64, copy=False))
        pattern_linear = pattern_rows * np.int64(n_cols) + indices.astype(np.int64, copy=False)
        self._cached_pattern_linear = pattern_linear
        self._cached_pattern_indptr = indptr
        self._cached_pattern_indices = indices
        self._cached_csr_matrix = None
        if self._assume_fixed_pattern:
            linear = rows.astype(np.int64, copy=False) * np.int64(n_cols) + cols.astype(np.int64, copy=False)
            slot = np.searchsorted(pattern_linear, linear)
            if slot.size and int(slot.max()) < pattern_linear.size and np.array_equal(pattern_linear[slot], linear):
                self._set_cached_slot_positions(slot)
                self._cached_csr_data = csr.data
                self._cached_csr_matrix = csr
                self._data_only_refresh_enabled = True
        return csr


def is_sparse_matrix(matrix) -> bool:
    return bool(SP_ISSPARSE(matrix))


def sparse_structural_rank(matrix) -> Optional[int]:
    """Return sparse structural rank for sparse matrices."""
    if not is_sparse_matrix(matrix):
        return None
    return int(SP_STRUCTURAL_RANK(matrix))


def matrix_is_empty(matrix) -> bool:
    return matrix.shape[0] == 0 or matrix.shape[1] == 0


def full_normal_equation_from_lower(matrix):
    """Expand a lower-triangular normal-equation matrix into full symmetry."""
    if is_sparse_matrix(matrix):
        lower = matrix if getattr(matrix, "format", None) == "csc" else matrix.tocsc()
        full = lower + lower.T
        full.setdiag(lower.diagonal())
        return full.tocsc()
    lower = np.tril(np.asarray(matrix, dtype=np.float64))
    return lower + lower.T - np.diag(np.diag(lower))


def _normal_equation_structural_pattern(H, triangular: Optional[str] = None):
    """Return the structural pattern implied by H.T @ H, retaining zero entries."""
    if not is_sparse_matrix(H):
        return None
    triangular_mode = None if triangular is None else str(triangular).lower()
    if triangular_mode not in (None, "lower"):
        raise ValueError(f"Unsupported normal-equation triangular mode: {triangular!r}")
    # 若 H 的稀疏结构数组对象身份未变（如固定模式刷新），跳过 tocsc() 与 blake2b 哈希
    h_fmt = getattr(H, "format", None)
    h_indptr = getattr(H, "indptr", None) if h_fmt in ("csr", "csc") else None
    h_indices = getattr(H, "indices", None) if h_fmt in ("csr", "csc") else None
    h_shape = H.shape
    h_nnz = int(H.nnz)
    if h_indptr is not None and h_indices is not None:
        for c_fmt, c_shape, c_nnz, c_tri, c_indptr, c_indices, c_pattern in _NORMAL_EQUATION_PATTERN_IDENTITY_CACHE:
            if (
                c_fmt == h_fmt
                and c_shape == h_shape
                and c_nnz == h_nnz
                and c_tri == triangular_mode
                and c_indptr is h_indptr
                and c_indices is h_indices
            ):
                return c_pattern

    H_csc = H if h_fmt == "csc" else H.tocsc()
    digest = hashlib.blake2b(
        H_csc.indptr.tobytes() + H_csc.indices.tobytes(),
        digest_size=16,
    ).digest()
    key = (H_csc.shape, int(H_csc.nnz), digest, triangular_mode)
    cached = _NORMAL_EQUATION_PATTERN_CACHE.get(key)
    if cached is not None:
        if h_indptr is not None and h_indices is not None:
            if len(_NORMAL_EQUATION_PATTERN_IDENTITY_CACHE) >= 8:
                _NORMAL_EQUATION_PATTERN_IDENTITY_CACHE.pop(0)
            _NORMAL_EQUATION_PATTERN_IDENTITY_CACHE.append(
                (h_fmt, h_shape, h_nnz, triangular_mode, h_indptr, h_indices, cached)
            )
        return cached
    H_pattern = H_csc.copy()
    H_pattern.data = np.ones(int(H_pattern.nnz), dtype=np.float64)
    pattern = (H_pattern.T @ H_pattern).tocsc()
    if triangular_mode == "lower":
        pattern = SP_TRIL(pattern, format="csc")
    pattern.data = np.zeros(int(pattern.nnz), dtype=np.float64)
    if len(_NORMAL_EQUATION_PATTERN_CACHE) > 16:
        _NORMAL_EQUATION_PATTERN_CACHE.clear()
    _NORMAL_EQUATION_PATTERN_CACHE[key] = pattern
    if h_indptr is not None and h_indices is not None:
        if len(_NORMAL_EQUATION_PATTERN_IDENTITY_CACHE) >= 8:
            _NORMAL_EQUATION_PATTERN_IDENTITY_CACHE.pop(0)
        _NORMAL_EQUATION_PATTERN_IDENTITY_CACHE.append(
            (h_fmt, h_shape, h_nnz, triangular_mode, h_indptr, h_indices, pattern)
        )
    return pattern


def _expand_sparse_matrix_to_pattern(matrix, pattern):
    """Return matrix on the union pattern, keeping explicit structural zeros."""
    if pattern is None:
        return matrix
    matrix_csc = matrix if getattr(matrix, "format", None) == "csc" else matrix.tocsc()
    pattern_csc = pattern if getattr(pattern, "format", None) == "csc" else pattern.tocsc()
    if not matrix_csc.has_sorted_indices:
        matrix_csc.sort_indices()
    if not pattern_csc.has_sorted_indices:
        pattern_csc.sort_indices()
    if (
        matrix_csc.shape == pattern_csc.shape
        and int(matrix_csc.nnz) == int(pattern_csc.nnz)
        and np.array_equal(matrix_csc.indptr, pattern_csc.indptr)
        and np.array_equal(matrix_csc.indices, pattern_csc.indices)
    ):
        return matrix_csc

    pattern_key = id(pattern_csc)
    cache = _SPARSE_PATTERN_EXPANSION_CACHE.get(pattern_key)
    if (
        cache is not None
        and cache["shape"] == matrix_csc.shape
        and cache["nnz"] == int(matrix_csc.nnz)
        and np.array_equal(cache["indptr"], matrix_csc.indptr)
        and np.array_equal(cache["indices"], matrix_csc.indices)
    ):
        target_positions = cache["target_positions"]
    else:
        n_rows = int(pattern_csc.shape[0])
        pattern_cache_key = (id(pattern_csc), pattern_csc.shape, int(pattern_csc.nnz))
        pattern_linear = _SPARSE_PATTERN_LINEAR_INDEX_CACHE.get(pattern_cache_key)
        if pattern_linear is None:
            pattern_counts = np.diff(pattern_csc.indptr).astype(np.int64, copy=False)
            pattern_cols = np.repeat(np.arange(int(pattern_csc.shape[1]), dtype=np.int64), pattern_counts)
            pattern_linear = pattern_cols * n_rows + pattern_csc.indices.astype(np.int64, copy=False)
            if len(_SPARSE_PATTERN_LINEAR_INDEX_CACHE) > 16:
                _SPARSE_PATTERN_LINEAR_INDEX_CACHE.clear()
            _SPARSE_PATTERN_LINEAR_INDEX_CACHE[pattern_cache_key] = pattern_linear
        matrix_counts = np.diff(matrix_csc.indptr).astype(np.int64, copy=False)
        matrix_cols = np.repeat(np.arange(int(matrix_csc.shape[1]), dtype=np.int64), matrix_counts)
        matrix_linear = matrix_cols * n_rows + matrix_csc.indices.astype(np.int64, copy=False)
        target_positions = np.nonzero(np.isin(pattern_linear, matrix_linear, assume_unique=True))[0].astype(
            np.int64,
            copy=False,
        )
        _SPARSE_PATTERN_EXPANSION_CACHE[pattern_key] = {
            "shape": matrix_csc.shape,
            "nnz": int(matrix_csc.nnz),
            "indptr": matrix_csc.indptr.copy(),
            "indices": matrix_csc.indices.copy(),
            "target_positions": target_positions,
        }
    data = np.zeros(int(pattern_csc.nnz), dtype=np.float64)
    data[target_positions] = matrix_csc.data
    return SP_CSC_MATRIX((data, pattern_csc.indices, pattern_csc.indptr), shape=pattern_csc.shape, copy=False)


def _csr_row_index_for_data(matrix_csr) -> np.ndarray:
    """Return the row index for every CSR data slot, cached by row pointer pattern."""
    indptr = matrix_csr.indptr
    digest = hashlib.blake2b(indptr.tobytes(), digest_size=16).digest()
    key = (matrix_csr.shape[0], int(matrix_csr.nnz), digest)
    cached = _SPARSE_CSR_ROW_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    counts = np.diff(indptr).astype(np.intp, copy=False)
    row_index = np.repeat(np.arange(matrix_csr.shape[0], dtype=np.intp), counts)
    if len(_SPARSE_CSR_ROW_INDEX_CACHE) > 16:
        _SPARSE_CSR_ROW_INDEX_CACHE.clear()
    _SPARSE_CSR_ROW_INDEX_CACHE[key] = row_index
    return row_index


def _row_weighted_sparse_matrix(matrix, weight: np.ndarray):
    """Return W*H with the same CSR structure, avoiding scipy's broadcast multiply path."""
    matrix_csr = matrix if getattr(matrix, "format", None) == "csr" else matrix.tocsr()
    weight = np.asarray(weight, dtype=np.float64)
    row_index = _csr_row_index_for_data(matrix_csr)
    weighted_data = np.empty(int(matrix_csr.nnz), dtype=np.float64)
    np.multiply(matrix_csr.data, weight[row_index], out=weighted_data)
    return SP_CSR_MATRIX(
        (weighted_data, matrix_csr.indices, matrix_csr.indptr),
        shape=matrix_csr.shape,
        copy=False,
    )


class NormalEquationAssemblyPlan:
    """Precomputed sparse assembly plan for repeated ``H.T W H`` builds.

    State-estimation iterations keep the active measurement layout fixed, so the
    Jacobian sparsity pattern is normally stable.  This plan turns the row-wise
    Jacobian pattern into raw normal-equation coordinate slots once, then each
    iteration only refreshes numeric values with vectorized reductions.
    """

    def __init__(
        self,
        *,
        shape: Tuple[int, int],
        h_indptr: np.ndarray,
        h_indices: np.ndarray,
        h_rows: np.ndarray,
        h_cols: np.ndarray,
        pair_left_pos: np.ndarray,
        pair_right_pos: np.ndarray,
        pair_rows: np.ndarray,
        pair_slot: np.ndarray,
        gain_indptr: np.ndarray,
        gain_indices: np.ndarray,
        pair_count: int,
        use_direct_assembly: bool,
    ):
        self.shape = tuple(int(item) for item in shape)
        self.h_indptr = np.asarray(h_indptr, dtype=np.int32)
        self.h_indices = np.asarray(h_indices, dtype=np.int32)
        self.h_rows = np.asarray(h_rows, dtype=np.int32)
        self.h_cols = np.asarray(h_cols, dtype=np.int32)
        self.pair_left_pos = np.asarray(pair_left_pos, dtype=np.int32)
        self.pair_right_pos = np.asarray(pair_right_pos, dtype=np.int32)
        self.pair_rows = np.asarray(pair_rows, dtype=np.int32)
        self.pair_slot = np.asarray(pair_slot, dtype=np.int32)
        self.gain_indptr = np.asarray(gain_indptr, dtype=np.int32)
        self.gain_indices = np.asarray(gain_indices, dtype=np.int32)
        self.gain_shape = (self.shape[1], self.shape[1])
        self.gain_nnz = int(self.gain_indices.size)
        self.pair_count = int(pair_count)
        self.use_direct_assembly = bool(use_direct_assembly)

    @classmethod
    def _row_pair_count(cls, H) -> int:
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        counts = np.diff(H_csr.indptr).astype(np.int64, copy=False)
        return int(np.dot(counts, counts))

    @classmethod
    def direct_assembly_is_reasonable(cls, H, max_pair_count: int = _MAX_DIRECT_NORMAL_ASSEMBLY_PAIRS) -> bool:
        if not is_sparse_matrix(H):
            return False
        return cls._row_pair_count(H) <= int(max_pair_count)

    @classmethod
    def from_jacobian(
        cls,
        H,
        *,
        max_direct_pair_count: int = _MAX_DIRECT_NORMAL_ASSEMBLY_PAIRS,
    ) -> "NormalEquationAssemblyPlan":
        if not is_sparse_matrix(H):
            raise TypeError("NormalEquationAssemblyPlan requires a sparse Jacobian")
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            H_csr = H_csr.copy()
            H_csr.sort_indices()
        indptr = H_csr.indptr.astype(np.int32, copy=True)
        indices = H_csr.indices.astype(np.int32, copy=True)
        counts = np.diff(indptr).astype(np.int32, copy=False)
        pair_count = int(np.dot(counts.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)))
        h_rows = np.repeat(np.arange(H_csr.shape[0], dtype=np.int32), counts)
        h_cols = indices.copy()

        left_chunks = []
        right_chunks = []
        row_chunks = []
        for count in np.unique(counts):
            k = int(count)
            if k <= 0:
                continue
            rows = np.nonzero(counts == k)[0].astype(np.int32, copy=False)
            if rows.size == 0:
                continue
            positions = indptr[rows, None] + np.arange(k, dtype=np.int32)
            left = np.repeat(positions, k, axis=1).ravel()
            right = np.tile(positions, (1, k)).ravel()
            left_chunks.append(left.astype(np.int32, copy=False))
            right_chunks.append(right.astype(np.int32, copy=False))
            row_chunks.append(np.repeat(rows, k * k).astype(np.int32, copy=False))

        if left_chunks:
            pair_left_pos = np.concatenate(left_chunks).astype(np.int32, copy=False)
            pair_right_pos = np.concatenate(right_chunks).astype(np.int32, copy=False)
            pair_rows = np.concatenate(row_chunks).astype(np.int32, copy=False)
            n_state = int(H_csr.shape[1])
            gain_rows = indices[pair_left_pos].astype(np.int64, copy=False)
            gain_cols = indices[pair_right_pos].astype(np.int64, copy=False)
            raw_linear = gain_cols * np.int64(n_state) + gain_rows
            gain_linear, pair_slot = np.unique(raw_linear, return_inverse=True)
            gain_indices = (gain_linear % np.int64(n_state)).astype(np.int32, copy=False)
            gain_cols_unique = (gain_linear // np.int64(n_state)).astype(np.int64, copy=False)
            gain_indptr = np.zeros(n_state + 1, dtype=np.int32)
            np.add.at(gain_indptr, gain_cols_unique + 1, 1)
            np.cumsum(gain_indptr, out=gain_indptr)
            pair_slot = pair_slot.astype(np.int32, copy=False)
        else:
            pair_left_pos = np.array([], dtype=np.int32)
            pair_right_pos = np.array([], dtype=np.int32)
            pair_rows = np.array([], dtype=np.int32)
            pair_slot = np.array([], dtype=np.int32)
            gain_indices = np.array([], dtype=np.int32)
            gain_indptr = np.zeros(H_csr.shape[1] + 1, dtype=np.int32)

        return cls(
            shape=H_csr.shape,
            h_indptr=indptr,
            h_indices=indices,
            h_rows=h_rows,
            h_cols=h_cols,
            pair_left_pos=pair_left_pos,
            pair_right_pos=pair_right_pos,
            pair_rows=pair_rows,
            pair_slot=pair_slot,
            gain_indptr=gain_indptr,
            gain_indices=gain_indices,
            pair_count=pair_count,
            use_direct_assembly=pair_count <= int(max_direct_pair_count),
        )

    def matches(self, H) -> bool:
        if not is_sparse_matrix(H) or tuple(H.shape) != self.shape:
            return False
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            return False
        return (
            int(H_csr.nnz) == int(self.h_indices.size)
            and H_csr.indptr.shape == self.h_indptr.shape
            and H_csr.indices.shape == self.h_indices.shape
            and np.array_equal(H_csr.indptr, self.h_indptr)
            and np.array_equal(H_csr.indices, self.h_indices)
        )

    def assemble(
        self,
        H,
        residual: np.ndarray,
        weight: np.ndarray,
        *,
        uniform_weight: Optional[float] = None,
        weights_are_uniform: Optional[bool] = None,
        weighted_residual: Optional[np.ndarray] = None,
        dense_gain_limit: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        data = np.asarray(H_csr.data, dtype=np.float64)
        residual = np.asarray(residual, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)

        if weight.size == 0:
            row_weight = None
            weighted_residual_values = residual
            scale = 1.0
        elif weights_are_uniform is True or uniform_weight is not None:
            scale = float(weight[0] if uniform_weight is None else uniform_weight)
            row_weight = None
            weighted_residual_values = residual if scale == 1.0 else residual * scale
        else:
            if weights_are_uniform is None and weight.size:
                first_weight = float(weight[0])
                if np.all(weight == first_weight):
                    scale = first_weight
                    row_weight = None
                    weighted_residual_values = residual if scale == 1.0 else residual * scale
                else:
                    scale = 1.0
                    row_weight = weight
                    weighted_residual_values = weight * residual if weighted_residual is None else weighted_residual
            else:
                scale = 1.0
                row_weight = weight
                weighted_residual_values = weight * residual if weighted_residual is None else weighted_residual

        if self.pair_slot.size:
            gain_values = data[self.pair_left_pos] * data[self.pair_right_pos]
            if row_weight is not None:
                gain_values = gain_values * row_weight[self.pair_rows]
            elif scale != 1.0:
                gain_values = gain_values * scale
            gain_data = np.bincount(
                self.pair_slot,
                weights=gain_values,
                minlength=self.gain_nnz,
            ).astype(np.float64, copy=False)
        else:
            gain_data = np.array([], dtype=np.float64)

        rhs_weights = data * weighted_residual_values[self.h_rows] if data.size else np.array([], dtype=np.float64)
        rhs = np.bincount(self.h_cols, weights=rhs_weights, minlength=self.shape[1]).astype(np.float64, copy=False)
        gain = SP_CSC_MATRIX(
            (gain_data, self.gain_indices, self.gain_indptr),
            shape=self.gain_shape,
            copy=False,
        )
        if gain.shape[0] <= dense_gain_limit:
            return gain.toarray(), rhs
        return gain, rhs


class LowerNormalEquationCscPlan:
    """Solver-ready lower-CSC assembly plan for repeated ``H.T W H`` builds.

    The plan owns the CSC ``indptr``/``indices`` and a reusable ``data`` buffer.
    Iterations refresh only numeric values for the lower triangle; they do not
    build the full normal matrix, call ``tril()``, or expand into a cached
    structural pattern.
    """

    def __init__(
        self,
        *,
        shape: Tuple[int, int],
        h_indptr: np.ndarray,
        h_indices: np.ndarray,
        h_rows: np.ndarray,
        h_cols: np.ndarray,
        pair_left_pos: np.ndarray,
        pair_right_pos: np.ndarray,
        pair_rows: np.ndarray,
        pair_slot: np.ndarray,
        gain_indptr: np.ndarray,
        gain_indices: np.ndarray,
    ):
        self.shape = tuple(int(item) for item in shape)
        self.h_indptr = np.asarray(h_indptr, dtype=np.int32)
        self.h_indices = np.asarray(h_indices, dtype=np.int32)
        self.h_rows = np.asarray(h_rows, dtype=np.int32)
        self.h_cols = np.asarray(h_cols, dtype=np.int32)
        self.pair_left_pos = np.asarray(pair_left_pos, dtype=np.int32)
        self.pair_right_pos = np.asarray(pair_right_pos, dtype=np.int32)
        self.pair_rows = np.asarray(pair_rows, dtype=np.int32)
        self.pair_slot = np.asarray(pair_slot, dtype=np.int32)
        self.gain_indptr = np.asarray(gain_indptr, dtype=np.int32)
        self.gain_indices = np.asarray(gain_indices, dtype=np.int32)
        self.gain_shape = (self.shape[1], self.shape[1])
        self.gain_nnz = int(self.gain_indices.size)
        self.gain_data = np.zeros(self.gain_nnz, dtype=np.float64)
        self.rhs = np.zeros(self.shape[1], dtype=np.float64)
        self._pair_values = np.empty(int(self.pair_slot.size), dtype=np.float64)
        self._rhs_values = np.empty(int(self.h_cols.size), dtype=np.float64)
        self._fixed_pair_weight = None
        self._fixed_h_weight = None
        self.gain = SP_CSC_MATRIX(
            (self.gain_data, self.gain_indices, self.gain_indptr),
            shape=self.gain_shape,
            copy=False,
        )
        self.pair_count = int(self.pair_slot.size)

    @classmethod
    def from_jacobian(cls, H) -> "LowerNormalEquationCscPlan":
        if not is_sparse_matrix(H):
            raise TypeError("LowerNormalEquationCscPlan requires a sparse Jacobian")
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            H_csr = H_csr.copy()
            H_csr.sort_indices()
        indptr = H_csr.indptr.astype(np.int32, copy=True)
        indices = H_csr.indices.astype(np.int32, copy=True)
        counts = np.diff(indptr).astype(np.int32, copy=False)
        h_rows = np.repeat(np.arange(H_csr.shape[0], dtype=np.int32), counts)
        h_cols = indices.copy()

        pair_counts = (counts.astype(np.int64, copy=False) * (counts.astype(np.int64, copy=False) + 1)) // 2
        total_pairs = int(pair_counts.sum())
        if total_pairs:
            pair_left_pos = np.empty(total_pairs, dtype=np.int32)
            pair_right_pos = np.empty(total_pairs, dtype=np.int32)
            pair_rows = np.empty(total_pairs, dtype=np.int32)
            nonzero_rows = np.flatnonzero(counts).astype(np.int32, copy=False)
            width_order = np.argsort(counts[nonzero_rows], kind="stable")
            rows_by_width = nonzero_rows[width_order]
            widths_by_row = counts[rows_by_width]
            width_breaks = np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.nonzero(widths_by_row[1:] != widths_by_row[:-1])[0] + 1,
                    np.array([rows_by_width.size], dtype=np.int64),
                )
            )
            offset = 0
            for group_start, group_stop in zip(width_breaks[:-1], width_breaks[1:]):
                rows = rows_by_width[int(group_start) : int(group_stop)]
                k = int(counts[rows[0]])
                per_row_pairs = (k * (k + 1)) // 2
                out_stop = offset + int(rows.size) * per_row_pairs
                if k == 1:
                    source = indptr[rows]
                    pair_left_pos[offset:out_stop] = source
                    pair_right_pos[offset:out_stop] = source
                    pair_rows[offset:out_stop] = rows
                else:
                    lower_left, lower_right = np.tril_indices(k)
                    lower_left = lower_left.astype(np.int32, copy=False)
                    lower_right = lower_right.astype(np.int32, copy=False)
                    starts = indptr[rows]
                    pair_left_pos[offset:out_stop] = (starts[:, None] + lower_left[None, :]).ravel()
                    pair_right_pos[offset:out_stop] = (starts[:, None] + lower_right[None, :]).ravel()
                    pair_rows[offset:out_stop].reshape(rows.size, per_row_pairs)[:] = rows[:, None]
                offset = out_stop
            n_state = int(H_csr.shape[1])
            gain_rows = indices[pair_left_pos].astype(np.int64, copy=False)
            gain_cols = indices[pair_right_pos].astype(np.int64, copy=False)
            raw_linear = gain_cols * np.int64(n_state) + gain_rows
            gain_linear, pair_slot = np.unique(raw_linear, return_inverse=True)
            gain_indices = (gain_linear % np.int64(n_state)).astype(np.int32, copy=False)
            gain_cols_unique = (gain_linear // np.int64(n_state)).astype(np.int64, copy=False)
            gain_indptr = np.zeros(n_state + 1, dtype=np.int32)
            np.add.at(gain_indptr, gain_cols_unique + 1, 1)
            np.cumsum(gain_indptr, out=gain_indptr)
            pair_slot = pair_slot.astype(np.int32, copy=False)
        else:
            pair_left_pos = np.array([], dtype=np.int32)
            pair_right_pos = np.array([], dtype=np.int32)
            pair_rows = np.array([], dtype=np.int32)
            pair_slot = np.array([], dtype=np.int32)
            gain_indices = np.array([], dtype=np.int32)
            gain_indptr = np.zeros(H_csr.shape[1] + 1, dtype=np.int32)

        return cls(
            shape=H_csr.shape,
            h_indptr=indptr,
            h_indices=indices,
            h_rows=h_rows,
            h_cols=h_cols,
            pair_left_pos=pair_left_pos,
            pair_right_pos=pair_right_pos,
            pair_rows=pair_rows,
            pair_slot=pair_slot,
            gain_indptr=gain_indptr,
            gain_indices=gain_indices,
        )

    def prepare_fixed_weights(self, weight: np.ndarray) -> None:
        """Cache ``weight[pair_rows]`` for repeated fixed-weight assemblies."""
        weight = np.asarray(weight, dtype=np.float64)
        if self.pair_rows.size:
            self._fixed_pair_weight = weight[self.pair_rows].copy()
        else:
            self._fixed_pair_weight = np.array([], dtype=np.float64)
        if self.h_rows.size:
            self._fixed_h_weight = weight[self.h_rows].copy()
        else:
            self._fixed_h_weight = np.array([], dtype=np.float64)

    def clear_fixed_weights(self) -> None:
        self._fixed_pair_weight = None
        self._fixed_h_weight = None

    def matches(self, H) -> bool:
        if not is_sparse_matrix(H) or tuple(H.shape) != self.shape:
            return False
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            return False
        return (
            int(H_csr.nnz) == int(self.h_indices.size)
            and H_csr.indptr.shape == self.h_indptr.shape
            and H_csr.indices.shape == self.h_indices.shape
            and np.array_equal(H_csr.indptr, self.h_indptr)
            and np.array_equal(H_csr.indices, self.h_indices)
        )

    def assemble(
        self,
        H,
        residual: np.ndarray,
        weight: np.ndarray,
        *,
        uniform_weight: Optional[float] = None,
        weights_are_uniform: Optional[bool] = None,
        weighted_residual: Optional[np.ndarray] = None,
        dense_gain_limit: int = 1000,
        assume_fixed_weights: bool = False,
        copy_rhs: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        data = np.asarray(H_csr.data, dtype=np.float64)
        residual = np.asarray(residual, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)

        if weight.size == 0:
            row_weight = None
            weighted_residual_values = residual
            scale = 1.0
        elif weights_are_uniform is True or uniform_weight is not None:
            scale = float(weight[0] if uniform_weight is None else uniform_weight)
            row_weight = None
            weighted_residual_values = residual if scale == 1.0 else residual * scale
        else:
            if weights_are_uniform is None and weight.size:
                first_weight = float(weight[0])
                if np.all(weight == first_weight):
                    scale = first_weight
                    row_weight = None
                    weighted_residual_values = residual if scale == 1.0 else residual * scale
                else:
                    scale = 1.0
                    row_weight = weight
                    weighted_residual_values = None if (assume_fixed_weights and weighted_residual is None) else (
                        weight * residual if weighted_residual is None else weighted_residual
                    )
            else:
                scale = 1.0
                row_weight = weight
                weighted_residual_values = None if (assume_fixed_weights and weighted_residual is None) else (
                    weight * residual if weighted_residual is None else weighted_residual
                )

        if self.pair_slot.size:
            np.multiply(data[self.pair_left_pos], data[self.pair_right_pos], out=self._pair_values)
            if row_weight is not None:
                if assume_fixed_weights:
                    if self._fixed_pair_weight is None or self._fixed_pair_weight.shape[0] != self.pair_rows.size:
                        self.prepare_fixed_weights(row_weight)
                    np.multiply(self._pair_values, self._fixed_pair_weight, out=self._pair_values)
                else:
                    np.multiply(self._pair_values, row_weight[self.pair_rows], out=self._pair_values)
            elif scale != 1.0:
                self._pair_values *= scale
            self.gain_data[:] = np.bincount(
                self.pair_slot,
                weights=self._pair_values,
                minlength=self.gain_nnz,
            )
        else:
            self.gain_data.fill(0.0)

        if data.size:
            if weighted_residual_values is None and row_weight is not None and assume_fixed_weights:
                if self._fixed_h_weight is None or self._fixed_h_weight.shape[0] != self.h_rows.size:
                    self.prepare_fixed_weights(row_weight)
                np.multiply(data, residual[self.h_rows], out=self._rhs_values)
                np.multiply(self._rhs_values, self._fixed_h_weight, out=self._rhs_values)
            else:
                np.multiply(data, weighted_residual_values[self.h_rows], out=self._rhs_values)
            self.rhs[:] = np.bincount(
                self.h_cols,
                weights=self._rhs_values,
                minlength=self.shape[1],
            )
        else:
            self.rhs.fill(0.0)
        rhs = self.rhs.copy() if copy_rhs else self.rhs
        if self.gain.shape[0] <= dense_gain_limit:
            return self.gain.toarray(), rhs
        return self.gain, rhs


class CholmodAAtNormalEquationPlan:
    """Prototype CHOLMOD ``A A.T`` plan for ``H.T W H`` normal equations.

    ``A`` is the weighted transpose of the measurement Jacobian:
    ``A = H.T * sqrt(W)``.  CHOLMOD's AAt factorization then solves the same
    system as ``H.T W H`` without explicitly assembling the lower normal CSC.
    The plan keeps the transposed CSC structure fixed and refreshes only data.
    """

    def __init__(
        self,
        *,
        shape: Tuple[int, int],
        h_indptr: np.ndarray,
        h_indices: np.ndarray,
        h_rows: np.ndarray,
        h_cols: np.ndarray,
    ):
        self.shape = tuple(int(item) for item in shape)
        self.h_indptr = np.asarray(h_indptr, dtype=np.int32)
        self.h_indices = np.asarray(h_indices, dtype=np.int32)
        self.h_rows = np.asarray(h_rows, dtype=np.int32)
        self.h_cols = np.asarray(h_cols, dtype=np.int32)
        self.a_indptr = self.h_indptr
        self.a_indices = self.h_indices
        self.a_shape = (self.shape[1], self.shape[0])
        self.a_data = np.zeros(int(self.h_indices.size), dtype=np.float64)
        self.rhs = np.zeros(self.shape[1], dtype=np.float64)
        self._rhs_values = np.empty(int(self.h_cols.size), dtype=np.float64)
        self._sqrt_weighted_residual = np.empty(self.shape[0], dtype=np.float64)
        self._fixed_sqrt_weight = None
        self._fixed_sqrt_row_weight = None
        self._dyn_weight_snapshot = None
        self._dyn_sqrt_row_weight = None
        self._dyn_sqrt_weight = None
        self._assembly_version = 0
        self.A = SP_CSC_MATRIX(
            (self.a_data, self.a_indices, self.a_indptr),
            shape=self.a_shape,
            copy=False,
        )
        self.A._aat_assembly_version = 0

    @classmethod
    def from_jacobian(cls, H) -> "CholmodAAtNormalEquationPlan":
        if not is_sparse_matrix(H):
            raise TypeError("CholmodAAtNormalEquationPlan requires a sparse Jacobian")
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            H_csr = H_csr.copy()
            H_csr.sort_indices()
        indptr = H_csr.indptr.astype(np.int32, copy=True)
        indices = H_csr.indices.astype(np.int32, copy=True)
        counts = np.diff(indptr).astype(np.int32, copy=False)
        h_rows = np.repeat(np.arange(H_csr.shape[0], dtype=np.int32), counts)
        h_cols = indices.copy()
        return cls(
            shape=H_csr.shape,
            h_indptr=indptr,
            h_indices=indices,
            h_rows=h_rows,
            h_cols=h_cols,
        )

    def prepare_fixed_weights(self, weight: np.ndarray) -> None:
        weight = np.asarray(weight, dtype=np.float64)
        self._fixed_sqrt_row_weight = np.sqrt(weight).astype(np.float64, copy=True)
        if self.h_rows.size:
            self._fixed_sqrt_weight = self._fixed_sqrt_row_weight[self.h_rows].astype(np.float64, copy=True)
        else:
            self._fixed_sqrt_weight = np.array([], dtype=np.float64)

    def clear_fixed_weights(self) -> None:
        self._fixed_sqrt_weight = None
        self._fixed_sqrt_row_weight = None
        self._dyn_weight_snapshot = None
        self._dyn_sqrt_row_weight = None
        self._dyn_sqrt_weight = None

    def matches(self, H) -> bool:
        if not is_sparse_matrix(H) or tuple(H.shape) != self.shape:
            return False
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            return False
        return (
            int(H_csr.nnz) == int(self.h_indices.size)
            and H_csr.indptr.shape == self.h_indptr.shape
            and H_csr.indices.shape == self.h_indices.shape
            and np.array_equal(H_csr.indptr, self.h_indptr)
            and np.array_equal(H_csr.indices, self.h_indices)
        )

    def assemble(
        self,
        H,
        residual: np.ndarray,
        weight: np.ndarray,
        *,
        uniform_weight: Optional[float] = None,
        weights_are_uniform: Optional[bool] = None,
        weighted_residual: Optional[np.ndarray] = None,
        assume_fixed_weights: bool = False,
        copy_rhs: bool = False,
        assume_pattern_matches: bool = False,
    ):
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        if not H_csr.has_sorted_indices:
            H_csr = H_csr.copy()
            H_csr.sort_indices()
        if not assume_pattern_matches and not self.matches(H_csr):
            raise ValueError("Jacobian sparse pattern does not match the AAt plan")

        data = np.asarray(H_csr.data, dtype=np.float64)
        residual = np.asarray(residual, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)

        if weight.size == 0:
            row_weight = None
            sqrt_scale = 1.0
            weighted_residual_values = residual
        elif weights_are_uniform is True or uniform_weight is not None:
            scale = float(weight[0] if uniform_weight is None else uniform_weight)
            row_weight = None
            sqrt_scale = math.sqrt(scale)
            weighted_residual_values = residual if scale == 1.0 else residual * scale
        else:
            if weights_are_uniform is None and weight.size:
                first_weight = float(weight[0])
                if np.all(weight == first_weight):
                    row_weight = None
                    sqrt_scale = math.sqrt(first_weight)
                    weighted_residual_values = residual if first_weight == 1.0 else residual * first_weight
                else:
                    row_weight = weight
                    sqrt_scale = 1.0
                    weighted_residual_values = None if (assume_fixed_weights and weighted_residual is None) else (
                        weight * residual if weighted_residual is None else weighted_residual
                    )
            else:
                row_weight = weight
                sqrt_scale = 1.0
                weighted_residual_values = None if (assume_fixed_weights and weighted_residual is None) else (
                    weight * residual if weighted_residual is None else weighted_residual
                )

        if data.size:
            dyn_sqrt_row_w = None
            if row_weight is not None:
                if assume_fixed_weights:
                    if self._fixed_sqrt_weight is None or self._fixed_sqrt_weight.shape[0] != self.h_rows.size:
                        self.prepare_fixed_weights(row_weight)
                    np.multiply(data, self._fixed_sqrt_weight, out=self.a_data)
                else:
                    # 同一次 estimate() 多轮迭代期间动态权重向量不变，缓存 sqrt(w)[h_rows] 免去每步百万次开方与花式索引
                    if (
                        self._dyn_weight_snapshot is None
                        or self._dyn_weight_snapshot.shape != row_weight.shape
                        or not np.array_equal(self._dyn_weight_snapshot, row_weight)
                    ):
                        self._dyn_weight_snapshot = row_weight.copy()
                        self._dyn_sqrt_row_weight = np.sqrt(np.maximum(row_weight, 0.0))
                        self._dyn_sqrt_weight = self._dyn_sqrt_row_weight[self.h_rows]
                    dyn_sqrt_row_w = self._dyn_sqrt_row_weight
                    np.multiply(data, self._dyn_sqrt_weight, out=self.a_data)
            elif sqrt_scale != 1.0:
                np.multiply(data, sqrt_scale, out=self.a_data)
            else:
                self.a_data[:] = data

            if weighted_residual_values is None and row_weight is not None and assume_fixed_weights:
                sqrt_row_weight = self._fixed_sqrt_row_weight
                if sqrt_row_weight is None or sqrt_row_weight.shape[0] != residual.shape[0]:
                    self.prepare_fixed_weights(row_weight)
                    sqrt_row_weight = self._fixed_sqrt_row_weight
                if self._sqrt_weighted_residual.shape[0] != residual.shape[0]:
                    self._sqrt_weighted_residual = np.empty(residual.shape[0], dtype=np.float64)
                np.multiply(residual, sqrt_row_weight, out=self._sqrt_weighted_residual)
                self.rhs[:] = self.A @ self._sqrt_weighted_residual
            elif dyn_sqrt_row_w is not None and weighted_residual is None:
                if self._sqrt_weighted_residual.shape[0] != residual.shape[0]:
                    self._sqrt_weighted_residual = np.empty(residual.shape[0], dtype=np.float64)
                np.multiply(residual, dyn_sqrt_row_w, out=self._sqrt_weighted_residual)
                self.rhs[:] = self.A @ self._sqrt_weighted_residual
            else:
                np.multiply(data, weighted_residual_values[self.h_rows], out=self._rhs_values)
                self.rhs[:] = np.bincount(
                    self.h_cols,
                    weights=self._rhs_values,
                    minlength=self.shape[1],
                )
        else:
            self.a_data.fill(0.0)
            self.rhs.fill(0.0)

        self._assembly_version += 1
        self.A._aat_assembly_version = self._assembly_version
        rhs = self.rhs.copy() if copy_rhs else self.rhs
        return self.A, rhs

    def assemble_rhs_only(
        self,
        residual: np.ndarray,
        weight: np.ndarray,
        *,
        uniform_weight: Optional[float] = None,
        weights_are_uniform: Optional[bool] = None,
        assume_fixed_weights: bool = False,
        copy_rhs: bool = False,
    ) -> np.ndarray:
        """在冻结雅可比与 CHOLMOD 因子精修步中仅快速计算右端项 H^T W r。"""
        residual = np.asarray(residual, dtype=np.float64)
        if not self.a_data.size:
            self.rhs.fill(0.0)
            return self.rhs.copy() if copy_rhs else self.rhs
        if weights_are_uniform is True or uniform_weight is not None:
            scale = float(weight[0] if uniform_weight is None else uniform_weight)
            sqrt_scale = math.sqrt(scale)
            if sqrt_scale == 1.0:
                self.rhs[:] = self.A @ residual
            else:
                if self._sqrt_weighted_residual.shape[0] != residual.shape[0]:
                    self._sqrt_weighted_residual = np.empty(residual.shape[0], dtype=np.float64)
                np.multiply(residual, sqrt_scale, out=self._sqrt_weighted_residual)
                self.rhs[:] = self.A @ self._sqrt_weighted_residual
        else:
            if not assume_fixed_weights or self._fixed_sqrt_row_weight is None or self._fixed_sqrt_row_weight.shape[0] != residual.shape[0]:
                self.prepare_fixed_weights(weight)
            sqrt_row_weight = self._fixed_sqrt_row_weight
            if self._sqrt_weighted_residual.shape[0] != residual.shape[0]:
                self._sqrt_weighted_residual = np.empty(residual.shape[0], dtype=np.float64)
            np.multiply(residual, sqrt_row_weight, out=self._sqrt_weighted_residual)
            self.rhs[:] = self.A @ self._sqrt_weighted_residual
        return self.rhs.copy() if copy_rhs else self.rhs


class CholmodAAtNormalEquationSolver:
    """正规方程稀疏正定求解器：中大规模优先使用多线程 MKL PARDISO，小规模或无 MKL 自动回退 CHOLMOD ``A A.T`` / SciPy。"""

    def __init__(
        self,
        assume_fixed_pattern: bool = False,
        ordering_method: str = "auto",
        multi_solve: bool = False,
    ):
        self._cholmod_factor = None
        self._cholmod_pattern = None
        self._cholmod_disabled = False
        self._pardiso_factor = None
        self._pardiso_upper_pattern = None
        self._pardiso_last_a_key = None
        self._pardiso_disabled = False
        self._active_backend = None
        self._last_normal_operand = None
        self._last_factor_scaled = False
        self._last_unscaled_diag = None
        self.assume_fixed_pattern = bool(assume_fixed_pattern)
        self.ordering_method = str(ordering_method or "auto")
        self.multi_solve = bool(multi_solve)
        self._fallback_solver = NormalEquationSolver(assume_fixed_pattern=assume_fixed_pattern)

    def _reset_pardiso(self) -> None:
        # 仅解除当前求解器对因子的引用，避免提前 close() 销毁外部（如 prev_cholmod_factor / WoodburyRemovalSolver）
        # 仍在复用的有效句柄；当引用计数归零时由 MKLPardisoFactor.__del__ 自动调用 phase=-1 释放内存。
        self._pardiso_factor = None
        self._pardiso_upper_pattern = None
        self._pardiso_last_a_key = None

    def __del__(self) -> None:
        self._reset_pardiso()

    def _should_use_pardiso(self, n_state: int, nnz_a: int = 0) -> bool:
        if self._pardiso_disabled or MKLPardisoFactor is None or build_upper_csr_from_aat is None:
            return False
        pref = get_linear_solver_preference()
        if pref in ("cholmod", "scipy"):
            return False
        if pref == "pardiso":
            return bool(is_mkl_pardiso_available())
        # auto 模式：极小稀疏网（ns < 3000 且 avg_nnz < 10.5）CHOLMOD AMD 更轻快；
        # 中大规模网络（含高盲区导致 avg_nnz 下降的 1k~100k 节点 AC/Hybrid/Fluid）启用多线程 MKL PARDISO
        ns = int(n_state)
        if ns < _PARDISO_MIN_STATE_COUNT:
            return False
        if nnz_a > 0 and ns < 3000 and (float(nnz_a) / float(ns)) < 10.5:
            return False
        return bool(is_mkl_pardiso_available())

    def _solve_sparse_pardiso(self, A_csc, rhs: np.ndarray, damping: float = 0.0):
        beta = float(damping)
        n_state = int(A_csc.shape[0])
        a_version = getattr(A_csc, "_aat_assembly_version", None)
        a_key = (id(A_csc), a_version) if a_version is not None else None
        # 同一步迭代中若 A_csc 未重新装配、仅调整 LM 阻尼，直接原位更新对角元，免去整矩阵 A·A^T 重算
        if (
            a_key is not None
            and a_key == self._pardiso_last_a_key
            and self._pardiso_factor is not None
            and self._pardiso_factor.can_update_damping_inplace()
        ):
            self._pardiso_factor.update_damping_and_factorize(beta, aat_operand=A_csc)
            self._last_normal_operand = A_csc
            self._active_backend = "pardiso"
            return self._pardiso_factor.solve(rhs), None

        prev_pattern = self._pardiso_upper_pattern
        ia, ja, a, new_pattern = build_upper_csr_from_aat(
            A_csc,
            damping=beta,
            cached_pattern=prev_pattern,
        )
        need_new_factor = (self._pardiso_factor is None) or (new_pattern is not prev_pattern)
        if need_new_factor:
            self._reset_pardiso()
            self._pardiso_factor = MKLPardisoFactor(n_state, ia, ja, upper_pattern=new_pattern)
            self._pardiso_upper_pattern = new_pattern
        self._pardiso_last_a_key = a_key
        self._pardiso_factor.factorize_inplace(a, aat_operand=A_csc, damping=beta)
        self._last_normal_operand = A_csc
        self._active_backend = "pardiso"
        return self._pardiso_factor.solve(rhs), None

    @staticmethod
    def _sparse_pattern(matrix) -> Tuple[Tuple[int, int], int, np.ndarray, np.ndarray]:
        csc = matrix if getattr(matrix, "format", None) == "csc" else matrix.tocsc()
        return (
            csc.shape,
            int(csc.nnz),
            csc.indptr.astype(np.int64, copy=True),
            csc.indices.astype(np.int64, copy=True),
        )

    @staticmethod
    def _same_sparse_pattern(pattern, matrix) -> bool:
        if pattern is None:
            return False
        csc = matrix if getattr(matrix, "format", None) == "csc" else matrix.tocsc()
        shape, nnz, indptr, indices = pattern
        return (
            shape == csc.shape
            and nnz == int(csc.nnz)
            and indptr.shape == csc.indptr.shape
            and indices.shape == csc.indices.shape
            and np.array_equal(indptr, csc.indptr)
            and np.array_equal(indices, csc.indices)
        )

    def _solve_sparse_cholmod_aat(self, A_csc, rhs: np.ndarray, damping: float = 0.0):
        beta = float(damping)
        if CHOLMOD_ANALYZE_AAT is not None:
            if self._cholmod_factor is None or (
                not self.assume_fixed_pattern and not self._same_sparse_pattern(self._cholmod_pattern, A_csc)
            ):
                eff_ordering = (
                    SEScaleProfile.for_problem(
                        int(A_csc.shape[0]),
                        int(A_csc.shape[1]),
                        int(A_csc.nnz),
                        multi_solve=self.multi_solve,
                    ).aat_ordering_method
                    if self.ordering_method == "auto"
                    else self.ordering_method
                )
                if eff_ordering and eff_ordering != "default":
                    try:
                        self._cholmod_factor = CHOLMOD_ANALYZE_AAT(A_csc, ordering_method=eff_ordering)
                    except Exception:
                        self._cholmod_factor = CHOLMOD_ANALYZE_AAT(A_csc)
                else:
                    self._cholmod_factor = CHOLMOD_ANALYZE_AAT(A_csc)
                if not self.assume_fixed_pattern:
                    self._cholmod_pattern = self._sparse_pattern(A_csc)
            if beta > 0.0:
                self._cholmod_factor.cholesky_AAt_inplace(A_csc, beta=beta)
            else:
                self._cholmod_factor.cholesky_AAt_inplace(A_csc)
            self._last_normal_operand = A_csc
            return self._cholmod_factor(rhs), None
        if CHOLMOD_CHOLESKY_AAT is not None:
            factor = CHOLMOD_CHOLESKY_AAT(A_csc, beta=beta) if beta > 0.0 else CHOLMOD_CHOLESKY_AAT(A_csc)
            self._last_normal_operand = A_csc
            return factor(rhs), None
        raise RuntimeError("CHOLMOD AAt is not available")

    def solve(
        self,
        A,
        rhs: np.ndarray,
        return_factor_diag: bool = False,
        damping: float = 0.0,
        allow_slow_fallback: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not is_sparse_matrix(A):
            raise TypeError("CholmodAAtNormalEquationSolver requires a sparse weighted transpose matrix")
        A_csc = A if getattr(A, "format", None) == "csc" else A.tocsc()
        # 高异构权重（法方程对角元跨度 > 1e4）加阻尼时，自动切换尺度不变 Jacobi 列均衡 Marquardt 相对阻尼，
        # 避免绝对阻尼 β·I 对高权枢纽节点形同虚设、对低权开关支路过度压制导致 200 步慢爬。
        col_scale = None
        rhs_solve = rhs
        A_solve = A_csc
        self._last_factor_scaled = False
        self._last_unscaled_diag = None
        if float(damping) > 0.0 and int(A_csc.shape[0]) > 0 and A_csc.nnz > 0:
            diag_g = np.bincount(
                A_csc.indices,
                weights=A_csc.data * A_csc.data,
                minlength=int(A_csc.shape[0]),
            )
            nz_diag = diag_g[diag_g > 1e-12]
            diag_ratio = (float(nz_diag.max()) / float(nz_diag.min())) if nz_diag.size >= 2 else 1.0
            # 仅在真正极端异构权重（对角元跨度 > 5e7）或重阻尼慢爬（跨度 > 5e5 且 damping >= 0.5）时
            # 切入 Jacobi 列均衡 Marquardt，并交由 CHOLMOD 稳定执行，避免与 PARDISO 未缩放句柄并发占内存或跨尺度污染
            if diag_ratio > 5e7 or (diag_ratio > 5e5 and float(damping) >= 0.5):
                col_scale = 1.0 / np.sqrt(np.maximum(diag_g, 1e-12))
                A_sc = A_csc.copy()
                A_sc.data = A_csc.data * col_scale[A_csc.indices]
                A_solve = A_sc
                rhs_solve = np.asarray(rhs, dtype=np.float64) * col_scale
                self._last_factor_scaled = True
                self._last_unscaled_diag = diag_g
        pref = get_linear_solver_preference()
        if col_scale is None and self._should_use_pardiso(int(A_solve.shape[0]), int(A_solve.nnz)):
            try:
                return self._solve_sparse_pardiso(A_solve, rhs_solve, damping=damping)
            except (TypeError, AttributeError):
                self._reset_pardiso()
                self._pardiso_disabled = True
            except Exception as exc:
                # 若仅是 phase=22 数值分解非正定失败，严禁销毁已缓存的 phase=11 符号分析与上三角数值缓冲，
                # 供紧随其后的加阻尼步零开销原位更新对角元并复用符号分析。
                if MKLPardisoFactorizationError is not None and isinstance(exc, MKLPardisoFactorizationError):
                    self._active_backend = None
                else:
                    self._reset_pardiso()
                    self._active_backend = None
        if (
            pref != "scipy"
            and not self._cholmod_disabled
            and (CHOLMOD_ANALYZE_AAT is not None or CHOLMOD_CHOLESKY_AAT is not None)
        ):
            try:
                sol, diag = self._solve_sparse_cholmod_aat(A_solve, rhs_solve, damping=damping)
                self._active_backend = "cholmod"
                if col_scale is not None:
                    sol = np.asarray(sol, dtype=np.float64) * col_scale
                    self._last_normal_operand = A_csc
                return sol, diag
            except (TypeError, AttributeError):
                # 仅在接口不兼容时禁用 CHOLMOD；数值奇异/病态仅本次回退，避免后续健康矩阵永久退化到慢速 SuperLU
                self._cholmod_factor = None
                self._cholmod_pattern = None
                self._last_normal_operand = None
                self._active_backend = None
                self._cholmod_disabled = True
            except Exception:
                self._cholmod_factor = None
                self._cholmod_pattern = None
                self._last_normal_operand = None
                self._active_backend = None
        if not allow_slow_fallback:
            self._active_backend = None
            return np.full(np.asarray(rhs, dtype=np.float64).shape, np.nan, dtype=np.float64), None
        gain = A_solve @ A_solve.T
        if float(damping) > 0.0:
            gain = gain + SP_EYE(gain.shape[0], format="csc", dtype=np.float64) * float(damping)
        self._active_backend = "fallback"
        sol, diag = self._fallback_solver.solve(gain.tocsc(), rhs_solve, return_factor_diag=return_factor_diag)
        if col_scale is not None:
            sol = np.asarray(sol, dtype=np.float64) * col_scale
        return sol, diag

    @property
    def last_cholmod_factor(self):
        """返回最近一次成功分解的稀疏正定因子对象（PARDISO 或 CHOLMOD，供坏数据杠杆计算与 Woodbury 复用）。"""
        if self._last_factor_scaled:
            return None
        if self._active_backend == "pardiso" and self._pardiso_factor is not None:
            return self._pardiso_factor
        return self._cholmod_factor

    @property
    def last_normal_operand(self):
        """返回最近一次成功分解的稀疏加权转置操作数 A_csc（供极稀疏选元求逆按需切换 supernodal）。"""
        return self._last_normal_operand

    def unregularized_diagonal(self) -> Optional[np.ndarray]:
        """O(n) 返回最近一次 PARDISO 分解对应的无阻尼对角元 diag(A·A^T)，失败返回 None。"""
        if self._last_factor_scaled and self._last_unscaled_diag is not None:
            return self._last_unscaled_diag
        if self._active_backend == "pardiso" and self._pardiso_factor is not None:
            return self._pardiso_factor.unregularized_diagonal()
        return None

    def solve_with_cached_factor(self, rhs: np.ndarray) -> Optional[np.ndarray]:
        """直接复用当前 PARDISO 或 CHOLMOD 数值因子执行稀疏前代/回代，不重新分解。"""
        if self._active_backend == "pardiso" and self._pardiso_factor is not None and not self._pardiso_disabled:
            try:
                return np.asarray(self._pardiso_factor.solve(rhs), dtype=np.float64)
            except Exception:
                return None
        if self._cholmod_factor is None or self._cholmod_disabled:
            return None
        try:
            return np.asarray(self._cholmod_factor(rhs), dtype=np.float64)
        except Exception:
            return None

    def solve_from_plan(
        self,
        plan: CholmodAAtNormalEquationPlan,
        H,
        residual: np.ndarray,
        weight: np.ndarray,
        *,
        uniform_weight: Optional[float] = None,
        weights_are_uniform: Optional[bool] = None,
        weighted_residual: Optional[np.ndarray] = None,
        assume_fixed_weights: bool = False,
        return_factor_diag: bool = False,
        assume_pattern_matches: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        A, rhs = plan.assemble(
            H,
            residual,
            weight,
            uniform_weight=uniform_weight,
            weights_are_uniform=weights_are_uniform,
            weighted_residual=weighted_residual,
            assume_fixed_weights=assume_fixed_weights,
            copy_rhs=False,
            assume_pattern_matches=assume_pattern_matches,
        )
        return self.solve(A, rhs, return_factor_diag=return_factor_diag)


def _simplicial_selected_inverse_fast(L_csc):
    """纯单列零分配双指针 Takahashi 选元求逆（专攻极稀疏 simplicial Cholesky 因子）。"""
    n = int(L_csc.shape[0])
    indptr = L_csc.indptr.tolist()
    indices = L_csc.indices.tolist()
    ldata = L_csc.data.tolist()
    zdata = [0.0] * len(ldata)
    z_diag = [0.0] * n
    for j in range(n - 1, -1, -1):
        s0 = indptr[j]
        e0 = indptr[j + 1]
        u = e0 - s0 - 1
        inv_l0 = 1.0 / ldata[s0]
        inv_l0_sq = inv_l0 * inv_l0
        if u == 0:
            zdata[s0] = inv_l0_sq
            z_diag[j] = inv_l0_sq
            continue
        if u == 1:
            r1 = indices[s0 + 1]
            l1 = ldata[s0 + 1] * inv_l0
            z21 = -z_diag[r1] * l1
            z11 = inv_l0_sq - l1 * z21
            zdata[s0] = z11
            zdata[s0 + 1] = z21
            z_diag[j] = z11
            continue
        l_vec = [ldata[p] * inv_l0 for p in range(s0 + 1, e0)]
        r_vec = indices[s0 + 1 : e0]
        z_vec = [0.0] * u
        for a in range(u):
            col_r = r_vec[a]
            la = l_vec[a]
            z_vec[a] -= z_diag[col_r] * la
            if a + 1 < u:
                p = indptr[col_r] + 1
                te = indptr[col_r + 1]
                acc_a = 0.0
                for b in range(a + 1, u):
                    rb = r_vec[b]
                    while p < te and indices[p] < rb:
                        p += 1
                    if p < te and indices[p] == rb:
                        zv = zdata[p]
                        z_vec[b] -= zv * la
                        acc_a += zv * l_vec[b]
                        p += 1
                z_vec[a] -= acc_a
        z11 = inv_l0_sq
        for a in range(u):
            za = z_vec[a]
            zdata[s0 + 1 + a] = za
            z11 -= l_vec[a] * za
        zdata[s0] = z11
        z_diag[j] = z11
    return SP_CSC_MATRIX((np.asarray(zdata, dtype=np.float64), L_csc.indices, L_csc.indptr), shape=L_csc.shape, copy=False)


def _supernodal_selected_inverse_from_cholmod(factor, *, aat_operand=None, _allow_supernodal_rebuild: bool = True):
    """基于 CHOLMOD 原生超节点 Cholesky 因子 L 执行超节点 Takahashi 选元求逆，精确计算 (P G P^T)^-1 在 L+L^T 稀疏模式上的元素。"""
    L = factor.L()
    if getattr(L, "format", None) != "csc":
        L = L.tocsc()
    if not getattr(L, "has_sorted_indices", False):
        L.sort_indices()
    n = int(L.shape[0])
    if n == 0:
        empty = SP_CSC_MATRIX((0, 0), dtype=np.float64)
        return empty, np.empty(0, dtype=np.int32)
    # 极端高填充病态图的内存安全护栏（100k AC 约 41M 非零元，60M 为安全上限）
    if int(L.nnz) > 60_000_000:
        return None, None

    indptr = L.indptr
    indices = L.indices
    ldata = L.data
    l_diag = np.asarray(L.diagonal(), dtype=np.float64)
    if l_diag.size != n or np.any(l_diag <= 0.0) or not np.all(np.isfinite(l_diag)):
        return None, None

    # 优先走 C++17 + BLAS-3 原生超节点 Takahashi 选元求逆（零外部依赖，失败或关闭时自动回退纯 Python）
    if NATIVE_SUPERNODAL_SELECTED_INVERSE is not None:
        zdata_native = NATIVE_SUPERNODAL_SELECTED_INVERSE(L)
        if zdata_native is not None:
            Z_low = SP_CSC_MATRIX((zdata_native, indices, indptr), shape=(n, n), copy=False)
            Z_low.has_sorted_indices = True
            return Z_low, factor.P()

    # 向量化识别尾部稀疏模式相同的连续列作为超节点（factor.L() 原生保留 supernodal 块结构）
    col_len = np.diff(indptr)
    is_start = np.ones(n, dtype=bool)
    if n > 1:
        same_len = (col_len[:-1] - 1) == col_len[1:]
        has_sub = col_len[1:] > 0
        cand = np.flatnonzero(same_len & has_sub) + 1
        if cand.size:
            # 必要条件：前一列的次对角行号必须等于当前列号 j，且末尾元素一致
            s_p = indptr[cand - 1]
            e_p = indptr[cand]
            e_c = indptr[cand + 1]
            first_sub = indices[s_p + 1]
            cand = cand[(first_sub == cand) & (indices[e_p - 1] == indices[e_c - 1])]
            idx_mv = memoryview(indices)
            for j in cand.tolist():
                s_prev = int(indptr[j - 1])
                e_prev = int(indptr[j])
                e_cur = int(indptr[j + 1])
                if idx_mv[s_prev + 1 : e_prev] == idx_mv[e_prev:e_cur]:
                    is_start[j] = False
    sn_starts_arr = np.flatnonzero(is_start).astype(np.int32, copy=False)
    sn_starts = sn_starts_arr.tolist()
    sn_starts.append(n)
    n_sn = len(sn_starts) - 1

    # 极稀疏高碎裂 simplicial 因子（如 20k DC 电网：n_sn/n > 0.75 且平均列长 < 15）：
    # 1) 若提供了稀疏操作数 aat_operand，临时构造一次 mode="supernodal" 因子可将选元求逆提速 2.5x~3x；
    # 2) 否则走零分配双指针 simplicial 选元求逆，消除上万次小块 NumPy 分配与搜索开销。
    avg_l_col = float(L.nnz) / float(n)
    if n >= 2000 and avg_l_col < 15.0 and n_sn > int(0.75 * n):
        if _allow_supernodal_rebuild and aat_operand is not None and CHOLMOD_ANALYZE_AAT is not None:
            try:
                A_csc = aat_operand if getattr(aat_operand, "format", None) == "csc" else aat_operand.tocsc()
                if int(A_csc.shape[0]) == n:
                    sup_fac = CHOLMOD_ANALYZE_AAT(A_csc, mode="supernodal")
                    sup_fac.cholesky_AAt_inplace(A_csc)
                    Z_sup, P_sup = _supernodal_selected_inverse_from_cholmod(
                        sup_fac, aat_operand=None, _allow_supernodal_rebuild=False
                    )
                    if Z_sup is not None and P_sup is not None:
                        return Z_sup, P_sup
            except Exception:
                pass
        if int(L.nnz) <= 1_000_000:
            Z_simp = _simplicial_selected_inverse_fast(L)
            P_simp = np.asarray(factor.P(), dtype=np.int32)
            return Z_simp, P_simp

    col_to_sn = np.cumsum(is_start, dtype=np.int32) - 1
    col_to_off = (np.arange(n, dtype=np.int32) - sn_starts_arr[col_to_sn]).astype(np.int32, copy=False)
    col_to_sn_list = col_to_sn.tolist()
    col_to_off_list = col_to_off.tolist()
    indptr_list = indptr.tolist()
    # 对极稀疏因子（nnz <= 1M，如 DC 电网）一次性转 Python list，避免数万个小超节点反复触发 NumPy 标量拆箱；
    # 对超大规模稠密因子（nnz > 1M，如 AC/Hybrid）跳过全量转 list 以免除大内存分配与拷贝。
    indices_list = indices.tolist() if int(indices.size) <= 1_000_000 else None
    ldata_list = ldata.tolist() if int(ldata.size) <= 1_000_000 else None

    zdata = np.empty_like(ldata)
    z_diag = np.empty(n, dtype=np.float64)
    z_blocks = [None] * n_sn
    sn_rows = [None] * n_sn
    sn_contig = [False] * n_sn
    sn_is_1d = [False] * n_sn
    sn_row0 = [0] * n_sn
    sn_row_map = [None] * n_sn

    # 仅按超节点首列统计后续子超节点引用次数，避免对全量 9M indices 调用 .tolist()
    ref_count = [0] * n_sn
    for sid in range(n_sn):
        a = sn_starts[sid]
        b = sn_starts[sid + 1]
        w = b - a
        s0 = indptr_list[a]
        e0 = indptr_list[a + 1]
        u = (e0 - s0) - w
        if u <= 0 or (w == 1 and u == 1):
            continue
        if w == 1 and u == 2:
            # w == 1, u == 2 直算分支仅需从 r1 所在超节点查 z_{r2, r1}，r1/r2 对角元直接读 z_diag
            r1_idx = indices_list[s0 + 1] if indices_list is not None else int(indices[s0 + 1])
            ref_count[col_to_sn_list[r1_idx]] += 1
            continue
        if indices_list is not None:
            prev_t = -1
            for r in indices_list[s0 + w : e0]:
                t_sid = col_to_sn_list[r]
                if t_sid != prev_t:
                    ref_count[t_sid] += 1
                    prev_t = t_sid
        elif u <= 32:
            prev_t = -1
            for r in indices[s0 + w : e0].tolist():
                t_sid = col_to_sn_list[r]
                if t_sid != prev_t:
                    ref_count[t_sid] += 1
                    prev_t = t_sid
        else:
            sub_sn = col_to_sn[indices[s0 + w : e0]]
            ref_count[int(sub_sn[0])] += 1
            chg = np.flatnonzero(sub_sn[1:] != sub_sn[:-1]) + 1
            for k in chg.tolist():
                ref_count[int(sub_sn[k])] += 1

    for sid in range(n_sn - 1, -1, -1):
        a = sn_starts[sid]
        b = sn_starts[sid + 1]
        w = b - a
        s0 = indptr_list[a]
        e0 = indptr_list[a + 1]
        rows = indices[s0:e0]
        m = e0 - s0
        u = m - w
        sn_rows[sid] = rows
        if indices_list is not None:
            r_first = indices_list[s0] if m > 0 else 0
            sn_row0[sid] = r_first
            sn_contig[sid] = m > 0 and (indices_list[e0 - 1] - r_first == m - 1)
        else:
            r_first = int(rows[0]) if m > 0 else 0
            sn_row0[sid] = r_first
            sn_contig[sid] = m > 0 and (int(rows[m - 1]) - r_first == m - 1)

        if w == 1:
            sn_is_1d[sid] = True
            l0 = ldata_list[s0] if ldata_list is not None else float(ldata[s0])
            inv_l0 = 1.0 / l0
            inv_l0_sq = inv_l0 * inv_l0
            if u == 0:
                zdata[s0] = inv_l0_sq
                z_diag[a] = inv_l0_sq
                if ref_count[sid] > 0:
                    z_blocks[sid] = zdata[s0:e0]
                continue
            if u == 1:
                r0 = indices_list[s0 + 1] if indices_list is not None else int(rows[1])
                l1 = (ldata_list[s0 + 1] if ldata_list is not None else float(ldata[s0 + 1])) * inv_l0
                z21 = -float(z_diag[r0]) * l1
                z11 = inv_l0_sq - l1 * z21
                zdata[s0] = z11
                zdata[s0 + 1] = z21
                z_diag[a] = z11
                if ref_count[sid] > 0:
                    z_blocks[sid] = zdata[s0:e0]
                continue
            if u == 2:
                if indices_list is not None:
                    r1 = indices_list[s0 + 1]
                    r2 = indices_list[s0 + 2]
                    l1 = ldata_list[s0 + 1] * inv_l0
                    l2 = ldata_list[s0 + 2] * inv_l0
                else:
                    r1 = int(rows[1])
                    r2 = int(rows[2])
                    l1 = float(ldata[s0 + 1]) * inv_l0
                    l2 = float(ldata[s0 + 2]) * inv_l0
                t_sid = col_to_sn_list[r1]
                t_Z = z_blocks[t_sid]
                r0_t = sn_row0[t_sid]
                if sn_contig[t_sid] or r2 < sn_starts[t_sid + 1]:
                    pos2 = r2 - r0_t
                else:
                    row_map = sn_row_map[t_sid]
                    if row_map is None:
                        t_rows = sn_rows[t_sid]
                        if ref_count[t_sid] >= 2:
                            span = int(t_rows[-1]) - r0_t + 1
                            row_map = np.empty(span, dtype=np.int32)
                            row_map[t_rows - r0_t] = np.arange(t_rows.size, dtype=np.int32)
                            sn_row_map[t_sid] = row_map
                            pos2 = int(row_map[r2 - r0_t])
                        else:
                            pos2 = int(t_rows.searchsorted(r2))
                    else:
                        pos2 = int(row_map[r2 - r0_t])
                if sn_is_1d[t_sid]:
                    z21_off = float(t_Z[pos2])
                else:
                    z21_off = float(t_Z[pos2, col_to_off_list[r1]])
                rem_ref = ref_count[t_sid] - 1
                ref_count[t_sid] = rem_ref
                if rem_ref == 0:
                    sn_row_map[t_sid] = None
                    z_blocks[t_sid] = None
                    sn_rows[t_sid] = None
                z_r1 = float(z_diag[r1])
                z_r2 = float(z_diag[r2])
                z21_0 = -(z_r1 * l1 + z21_off * l2)
                z21_1 = -(z21_off * l1 + z_r2 * l2)
                z11 = inv_l0_sq - (l1 * z21_0 + l2 * z21_1)
                zdata[s0] = z11
                zdata[s0 + 1] = z21_0
                zdata[s0 + 2] = z21_1
                z_diag[a] = z11
                if ref_count[sid] > 0:
                    z_blocks[sid] = zdata[s0:e0]
                continue
            l_21 = ldata[s0 + 1 : e0] * inv_l0
            upd_rows = rows[1:]
            if u <= 32:
                upd_rows_list = upd_rows.tolist()
                u_sn_list = [col_to_sn_list[r] for r in upd_rows_list]
                bounds = [0]
                prev_s = u_sn_list[0]
                for idx_u in range(1, u):
                    cur_s = u_sn_list[idx_u]
                    if cur_s != prev_s:
                        bounds.append(idx_u)
                        prev_s = cur_s
                bounds.append(u)
                last_upd_row = upd_rows_list[u - 1]
                u_sn_get = u_sn_list
                upd_get = upd_rows_list
            else:
                u_sn = col_to_sn[upd_rows]
                chg = np.flatnonzero(u_sn[1:] != u_sn[:-1]) + 1
                bounds = [0] + chg.tolist() + [u]
                last_upd_row = int(upd_rows[u - 1])
                u_sn_get = u_sn
                upd_get = upd_rows
            z_21 = np.zeros(u, dtype=np.float64)
            for bi in range(len(bounds) - 1):
                rs = bounds[bi]
                re = bounds[bi + 1]
                target_sid = int(u_sn_get[rs])
                t_rows = sn_rows[target_sid]
                t_Z = z_blocks[target_sid]
                r0_t = sn_row0[target_sid]
                first_q = int(upd_get[rs])
                exact_match = (u - rs) == t_rows.size and first_q == r0_t
                if not exact_match:
                    query = upd_rows[rs:]
                    if sn_contig[target_sid] or last_upd_row < sn_starts[target_sid + 1]:
                        r_pos = query - r0_t
                    else:
                        row_map = sn_row_map[target_sid]
                        if row_map is None:
                            if ref_count[target_sid] >= 2:
                                span = int(t_rows[-1]) - r0_t + 1
                                row_map = np.empty(span, dtype=np.int32)
                                row_map[t_rows - r0_t] = np.arange(t_rows.size, dtype=np.int32)
                                sn_row_map[target_sid] = row_map
                                r_pos = row_map[query - r0_t]
                            else:
                                r_pos = t_rows.searchsorted(query)
                        else:
                            r_pos = row_map[query - r0_t]
                if sn_is_1d[target_sid]:
                    col_vec = t_Z if exact_match else t_Z[r_pos]
                    z_21[rs:] -= col_vec * float(l_21[rs])
                    if rs + 1 < u:
                        z_21[rs] -= float(np.dot(col_vec[1:], l_21[rs + 1 :]))
                elif re == rs + 1:
                    c0 = col_to_off_list[first_q]
                    col_vec = t_Z[:, c0] if exact_match else t_Z[r_pos, c0]
                    z_21[rs:] -= col_vec * float(l_21[rs])
                    if rs + 1 < u:
                        z_21[rs] -= float(np.dot(col_vec[1:], l_21[rs + 1 :]))
                else:
                    c0 = col_to_off_list[first_q]
                    c1 = col_to_off_list[int(upd_get[re - 1])] + 1
                    if c1 - c0 == re - rs:
                        sub = t_Z[:, c0:c1] if exact_match else t_Z[r_pos, c0:c1]
                    else:
                        t_cols = col_to_off[upd_rows[rs:re]]
                        sub = t_Z[:, t_cols] if exact_match else t_Z[r_pos[:, None], t_cols]
                    z_21[rs:] -= sub @ l_21[rs:re]
                    if re < u:
                        z_21[rs:re] -= sub[re - rs :, :].T @ l_21[re:]
                rem_ref = ref_count[target_sid] - 1
                ref_count[target_sid] = rem_ref
                if rem_ref == 0:
                    sn_row_map[target_sid] = None
                    z_blocks[target_sid] = None
                    sn_rows[target_sid] = None
            z11 = inv_l0_sq - float(np.dot(l_21, z_21))
            zdata[s0] = z11
            zdata[s0 + 1 : e0] = z_21
            z_diag[a] = z11
            if ref_count[sid] > 0:
                z_blocks[sid] = zdata[s0:e0]
            continue

        L_blk = np.zeros((m, w), dtype=np.float64)
        for col_off in range(w):
            cs = indptr_list[a + col_off]
            ce = indptr_list[a + col_off + 1]
            L_blk[col_off:, col_off] = ldata[cs:ce]
        L_11 = L_blk[:w, :]
        L_21 = L_blk[w:, :]
        # 小超节点直接用解析公式求下三角逆，较大超节点调用 LAPACK dtrtri 绕过 SciPy 包装开销
        if w == 2:
            d0 = 1.0 / L_11[0, 0]
            d1 = 1.0 / L_11[1, 1]
            L11_inv = np.array([[d0, 0.0], [-L_11[1, 0] * d0 * d1, d1]], dtype=np.float64)
        elif w == 3:
            d0 = 1.0 / L_11[0, 0]
            d1 = 1.0 / L_11[1, 1]
            d2 = 1.0 / L_11[2, 2]
            l10 = -L_11[1, 0] * d0 * d1
            l21 = -L_11[2, 1] * d1 * d2
            l20 = -(L_11[2, 0] * d0 + L_11[2, 1] * l10) * d2
            L11_inv = np.array(
                [
                    [d0, 0.0, 0.0],
                    [l10, d1, 0.0],
                    [l20, l21, d2],
                ],
                dtype=np.float64,
            )
        else:
            # L_11 严格上三角已由 L_blk 零初始化，DTRTRI 仅写入下三角，无需再调用 np.tril
            inv_raw, info = DTRTRI(L_11, lower=1, unitdiag=0, overwrite_c=0)
            if info == 0:
                L11_inv = inv_raw
            else:
                L11_inv = SOLVE_TRIANGULAR(
                    L_11,
                    np.eye(w, dtype=np.float64),
                    lower=True,
                    unit_diagonal=False,
                    check_finite=False,
                )
        Z_blk = np.empty((m, w), dtype=np.float64)
        if u > 0:
            L21_eff = L_21 @ L11_inv
            upd_rows = rows[w:]
            if u <= 32:
                upd_rows_list = upd_rows.tolist()
                u_sn_list = [col_to_sn_list[r] for r in upd_rows_list]
                bounds = [0]
                prev_s = u_sn_list[0]
                for idx_u in range(1, u):
                    cur_s = u_sn_list[idx_u]
                    if cur_s != prev_s:
                        bounds.append(idx_u)
                        prev_s = cur_s
                bounds.append(u)
                last_upd_row = upd_rows_list[u - 1]
                u_sn_get = u_sn_list
                upd_get = upd_rows_list
            else:
                u_sn = col_to_sn[upd_rows]
                chg = np.flatnonzero(u_sn[1:] != u_sn[:-1]) + 1
                bounds = [0] + chg.tolist() + [u]
                last_upd_row = int(upd_rows[u - 1])
                u_sn_get = u_sn
                upd_get = upd_rows
            # 混合策略：小超节点 (w < 16) 直接累加至 Z_21 免建 Z_22；大超节点 (w >= 16) 汇总 Z_22 走单次多线程 BLAS-3 dgemm
            use_z22 = w >= 16
            if use_z22:
                Z_22 = np.empty((u, u), dtype=np.float64)
            else:
                Z_21 = np.zeros((u, w), dtype=np.float64)
            for bi in range(len(bounds) - 1):
                rs = bounds[bi]
                re = bounds[bi + 1]
                target_sid = int(u_sn_get[rs])
                t_rows = sn_rows[target_sid]
                t_Z = z_blocks[target_sid]
                r0_t = sn_row0[target_sid]
                first_q = int(upd_get[rs])
                exact_match = (u - rs) == t_rows.size and first_q == r0_t
                if not exact_match:
                    query = upd_rows[rs:]
                    if sn_contig[target_sid] or last_upd_row < sn_starts[target_sid + 1]:
                        r_pos = query - r0_t
                    else:
                        row_map = sn_row_map[target_sid]
                        if row_map is None:
                            if ref_count[target_sid] >= 2:
                                span = int(t_rows[-1]) - r0_t + 1
                                row_map = np.empty(span, dtype=np.int32)
                                row_map[t_rows - r0_t] = np.arange(t_rows.size, dtype=np.int32)
                                sn_row_map[target_sid] = row_map
                                r_pos = row_map[query - r0_t]
                            else:
                                r_pos = t_rows.searchsorted(query)
                        else:
                            r_pos = row_map[query - r0_t]
                if sn_is_1d[target_sid]:
                    col_vec = t_Z if exact_match else t_Z[r_pos]
                    if use_z22:
                        Z_22[rs:, rs] = col_vec
                        if rs + 1 < u:
                            Z_22[rs, rs + 1 :] = col_vec[1:]
                    else:
                        Z_21[rs:] -= col_vec[:, None] * L21_eff[rs]
                        if rs + 1 < u:
                            Z_21[rs] -= col_vec[1:] @ L21_eff[rs + 1 :]
                elif re == rs + 1:
                    c0 = col_to_off_list[first_q]
                    col_vec = t_Z[:, c0] if exact_match else t_Z[r_pos, c0]
                    if use_z22:
                        Z_22[rs:, rs] = col_vec
                        if rs + 1 < u:
                            Z_22[rs, rs + 1 :] = col_vec[1:]
                    else:
                        Z_21[rs:] -= col_vec[:, None] * L21_eff[rs]
                        if rs + 1 < u:
                            Z_21[rs] -= col_vec[1:] @ L21_eff[rs + 1 :]
                else:
                    c0 = col_to_off_list[first_q]
                    c1 = col_to_off_list[int(upd_get[re - 1])] + 1
                    if c1 - c0 == re - rs:
                        sub = t_Z[:, c0:c1] if exact_match else t_Z[r_pos, c0:c1]
                    else:
                        t_cols = col_to_off[upd_rows[rs:re]]
                        sub = t_Z[:, t_cols] if exact_match else t_Z[r_pos[:, None], t_cols]
                    if use_z22:
                        Z_22[rs:, rs:re] = sub
                        if re < u:
                            Z_22[rs:re, re:] = sub[re - rs :, :].T
                    else:
                        Z_21[rs:] -= sub @ L21_eff[rs:re]
                        if re < u:
                            Z_21[rs:re] -= sub[re - rs :, :].T @ L21_eff[re:]
                rem_ref = ref_count[target_sid] - 1
                ref_count[target_sid] = rem_ref
                if rem_ref == 0:
                    sn_row_map[target_sid] = None
                    z_blocks[target_sid] = None
                    sn_rows[target_sid] = None
            if use_z22:
                Z_21 = -(Z_22 @ L21_eff)
            Z_11 = L11_inv.T @ L11_inv - L21_eff.T @ Z_21
            Z_blk[:w, :] = Z_11
            Z_blk[w:, :] = Z_21
        else:
            Z_11 = L11_inv.T @ L11_inv
            Z_blk[:w, :] = Z_11
        z_diag[a:b] = Z_11.diagonal()
        for col_off in range(w):
            cs = indptr_list[a + col_off]
            ce = indptr_list[a + col_off + 1]
            zdata[cs:ce] = Z_blk[col_off:, col_off]
        if ref_count[sid] > 0:
            z_blocks[sid] = Z_blk

    Z_low = SP_CSC_MATRIX((zdata, indices, indptr), shape=(n, n), copy=False)
    Z_low.has_sorted_indices = True
    return Z_low, factor.P()


def _measurement_leverage_from_selected_inv(
    H,
    Z_low,
    P,
    *,
    cholmod_factor=None,
    _force_spmm: bool = False,
) -> Optional[np.ndarray]:
    """基于下三角选择逆 Z_low 计算精确 diag(H G^-1 H^T)；极少数数值抵消缺失项用 CHOLMOD 回代补齐。"""
    H_csr = H.tocsr() if getattr(H, "format", None) != "csr" else H
    inv_P = np.empty(P.size, dtype=np.int32)
    inv_P[P] = np.arange(P.size, dtype=np.int32)

    # 优先走 OpenMP 并行原生内核直接计算 h_i^T Z_low h_i（免构建 H_p 与 SpMM 临时大矩阵；
    # 若极罕见遇到数值抵消缺失项，自动返回 None 并无缝落入下方 SpMM + CHOLMOD 回代补齐路径）
    if not _force_spmm and NATIVE_LEVERAGE_FROM_SELECTED_INV is not None:
        lev_native = NATIVE_LEVERAGE_FROM_SELECTED_INV(H_csr, Z_low, inv_P)
        if lev_native is not None:
            return lev_native

    hp_indices = inv_P[H_csr.indices]
    # 复制 data/indptr 以避免 sort_indices() 原地改动调用方的 H
    H_p = SP_CSR_MATRIX(
        (H_csr.data.copy(), hp_indices, H_csr.indptr.copy()),
        shape=H_csr.shape,
        copy=False,
    )
    H_p.sort_indices()

    indptr = H_p.indptr
    indices = H_p.indices
    hdata = H_p.data
    row_nnz = np.diff(indptr)
    max_nnz = int(row_nnz.max()) if row_nnz.size else 0
    z_diag = Z_low.diagonal()

    # 对宽行大电网（如 AC/Hybrid，max_nnz >= 4 且行数 >= 4096）采用稀疏裁剪 + 半矩阵 SpMM：
    # 利用恒等式 diag(H_p (D + Z_sub + Z_sub^T) H_p^T) = (H_p^2) z_diag + 2 * row_sum(H_p * (H_p @ Z_sub))
    # 彻底免除百万级状态对拼接、去重排序与二分查找，原生支持任意行宽且与全量求逆机器精度等价。
    if max_nnz >= 2 and (_force_spmm or (max_nnz >= 4 and int(H_p.shape[0]) >= 4096)):
        H_nz = H_p.copy()
        H_nz.eliminate_zeros()
        n = int(Z_low.shape[0])
        if H_nz.nnz == 0:
            return np.zeros(H_p.shape[0], dtype=np.float64)
        H_pat = SP_CSR_MATRIX(
            (np.ones(H_nz.nnz, dtype=np.float64), H_nz.indices, H_nz.indptr),
            shape=H_nz.shape,
            copy=False,
        )
        # CSC.T 为 CSR，使 CSR @ CSR 全程在 SciPy C++ 内核中高速完成
        Q_pat = SP_TRIL(H_pat.tocsc().T @ H_pat, k=-1, format="csc")
        if Q_pat.nnz > 0:
            Q_pat.data[:] = 1.0
            Z_sub = Z_low.multiply(Q_pat).tocsc()
            if Z_sub.nnz < Q_pat.nnz:
                Z_sub_pat = SP_CSC_MATRIX(
                    (np.ones(Z_sub.nnz, dtype=np.float64), Z_sub.indices, Z_sub.indptr),
                    shape=Z_sub.shape,
                    copy=False,
                )
                M_miss = (Q_pat - Z_sub_pat).tocoo()
                M_miss.eliminate_zeros()
                if M_miss.nnz > 0:
                    if cholmod_factor is None:
                        return None
                    miss_ca = M_miss.col.astype(np.intp, copy=False)
                    miss_cb = M_miss.row.astype(np.intp, copy=False)
                    uniq_ca, inv_ca = np.unique(miss_ca, return_inverse=True)
                    if uniq_ca.size > 64:
                        return None
                    rhs = np.zeros((n, uniq_ca.size), dtype=np.float64)
                    rhs[P[uniq_ca], np.arange(uniq_ca.size)] = 1.0
                    cols_inv = np.asarray(cholmod_factor(rhs), dtype=np.float64)
                    if cols_inv.ndim == 1:
                        cols_inv = cols_inv[:, None]
                    z_miss = cols_inv[P[miss_cb], inv_ca]
                    Z_miss = SP_CSC_MATRIX((z_miss, (miss_cb, miss_ca)), shape=Z_sub.shape)
                    Z_sub = (Z_sub + Z_miss).tocsr()
                else:
                    Z_sub = Z_sub.tocsr()
            else:
                Z_sub = Z_sub.tocsr()
            H_sq = SP_CSR_MATRIX(
                (H_nz.data * H_nz.data, H_nz.indices, H_nz.indptr),
                shape=H_nz.shape,
                copy=False,
            )
            lev_diag = np.asarray(H_sq @ z_diag, dtype=np.float64).ravel()
            lev_off = np.asarray(H_nz.multiply(H_nz @ Z_sub).sum(axis=1), dtype=np.float64).ravel()
            return lev_diag + 2.0 * lev_off
        H_sq = SP_CSR_MATRIX(
            (H_nz.data * H_nz.data, H_nz.indices, H_nz.indptr),
            shape=H_nz.shape,
            copy=False,
        )
        return np.asarray(H_sq @ z_diag, dtype=np.float64).ravel()

    lev = np.zeros(H_p.shape[0], dtype=np.float64)
    pair_rows_list = []
    pair_ca_list = []
    pair_cb_list = []
    pair_prod_list = []
    pair_limit = 64
    for k in range(1, min(max_nnz + 1, pair_limit)):
        r_idx = np.flatnonzero(row_nnz == k)
        if r_idx.size == 0:
            continue
        starts = indptr[r_idx]
        pos = starts[:, None] + np.arange(k, dtype=np.intp)[None, :]
        cols = indices[pos]
        vals = hdata[pos]
        lev[r_idx] = np.sum((vals * vals) * z_diag[cols], axis=1)
        if k >= 2:
            ia, ib = np.triu_indices(k, k=1)
            pair_rows_list.append(np.repeat(r_idx, ia.size))
            pair_ca_list.append(cols[:, ia].ravel())
            pair_cb_list.append(cols[:, ib].ravel())
            pair_prod_list.append((2.0 * vals[:, ia] * vals[:, ib]).ravel())

    if pair_rows_list:
        p_r = np.concatenate(pair_rows_list)
        p_ca = np.concatenate(pair_ca_list)
        p_cb = np.concatenate(pair_cb_list)
        p_prod = np.concatenate(pair_prod_list)
        # 跳过由固定稀疏模式显式零元产生的零乘积对，避免误报缺失
        nz_mask = p_prod != 0.0
        if not np.all(nz_mask):
            p_r = p_r[nz_mask]
            p_ca = p_ca[nz_mask]
            p_cb = p_cb[nz_mask]
            p_prod = p_prod[nz_mask]
        if p_r.size:
            n = Z_low.shape[0]
            q_keys = (p_ca.astype(np.int64) << 32) | p_cb.astype(np.int64)
            # 先对重复状态对去重并按升序查找，显著减少二分查找次数并提升缓存局部性
            if q_keys.size >= 4096:
                uniq_q, inv_q = np.unique(q_keys, return_inverse=True)
                if int(Z_low.nnz) > 2_000_000:
                    # 超大 Z_low（如 50k~100k AC/Hybrid，17M~41M nnz）走按列向量化并行二分查找，
                    # 彻底免除 300~650 MB 的全量 z_cols / z_keys int64 临时数组分配。
                    u_ca = (uniq_q >> 32).astype(np.intp, copy=False)
                    u_cb = (uniq_q & 0xFFFFFFFF).astype(Z_low.indices.dtype, copy=False)
                    z_indptr = Z_low.indptr
                    z_indices = Z_low.indices
                    lo = z_indptr[u_ca].astype(np.int64, copy=True)
                    hi = z_indptr[u_ca + 1].astype(np.int64, copy=True)
                    col_hi = hi.copy()
                    max_span = int(np.max(hi - lo)) if lo.size else 0
                    steps = int(max_span.bit_length()) if max_span > 0 else 0
                    for _ in range(steps):
                        mid = (lo + hi) >> 1
                        safe_mid = np.minimum(mid, col_hi - 1)
                        active_step = lo < hi
                        lt = active_step & (z_indices[safe_mid] < u_cb)
                        lo = np.where(lt, mid + 1, lo)
                        hi = np.where(active_step & ~lt, mid, hi)
                    u_pos = lo
                    u_hit = u_pos < col_hi
                    if np.any(u_hit):
                        u_hit_idx = np.flatnonzero(u_hit)
                        u_hit[u_hit_idx] = z_indices[u_pos[u_hit_idx]] == u_cb[u_hit_idx]
                else:
                    z_cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(Z_low.indptr))
                    z_keys = (z_cols << 32) | Z_low.indices.astype(np.int64)
                    u_pos = z_keys.searchsorted(uniq_q)
                    u_hit = u_pos < z_keys.size
                    if np.any(u_hit):
                        u_hit_idx = np.flatnonzero(u_hit)
                        u_hit[u_hit_idx] = z_keys[u_pos[u_hit_idx]] == uniq_q[u_hit_idx]
                if np.all(u_hit):
                    u_vals = Z_low.data[u_pos]
                    lev += np.bincount(p_r, weights=p_prod * u_vals[inv_q], minlength=lev.size)
                else:
                    if cholmod_factor is None:
                        return None
                    u_vals = np.empty(uniq_q.size, dtype=np.float64)
                    if np.any(u_hit):
                        u_vals[u_hit] = Z_low.data[u_pos[u_hit]]
                    u_miss = ~u_hit
                    miss_keys = uniq_q[u_miss]
                    miss_ca = (miss_keys >> 32).astype(np.intp, copy=False)
                    miss_cb = (miss_keys & 0xFFFFFFFF).astype(np.intp, copy=False)
                    uniq_ca, inv_ca = np.unique(miss_ca, return_inverse=True)
                    if uniq_ca.size > 64:
                        return None
                    rhs = np.zeros((n, uniq_ca.size), dtype=np.float64)
                    rhs[P[uniq_ca], np.arange(uniq_ca.size)] = 1.0
                    cols_inv = np.asarray(cholmod_factor(rhs), dtype=np.float64)
                    if cols_inv.ndim == 1:
                        cols_inv = cols_inv[:, None]
                    u_vals[u_miss] = cols_inv[P[miss_cb], inv_ca]
                    lev += np.bincount(p_r, weights=p_prod * u_vals[inv_q], minlength=lev.size)
            else:
                z_cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(Z_low.indptr))
                z_keys = (z_cols << 32) | Z_low.indices.astype(np.int64)
                q_pos = z_keys.searchsorted(q_keys)
                hit = q_pos < z_keys.size
                if np.any(hit):
                    hit_idx = np.flatnonzero(hit)
                    hit[hit_idx] = z_keys[q_pos[hit_idx]] == q_keys[hit_idx]
                if np.all(hit):
                    lev += np.bincount(p_r, weights=p_prod * Z_low.data[q_pos], minlength=lev.size)
                else:
                    if cholmod_factor is None:
                        return None
                    miss_mask = ~hit
                    miss_ca = p_ca[miss_mask]
                    miss_cb = p_cb[miss_mask]
                    uniq_ca, inv_ca = np.unique(miss_ca, return_inverse=True)
                    if uniq_ca.size > 64:
                        return None
                    if np.any(hit):
                        lev += np.bincount(
                            p_r[hit],
                            weights=p_prod[hit] * Z_low.data[q_pos[hit]],
                            minlength=lev.size,
                        )
                    rhs = np.zeros((n, uniq_ca.size), dtype=np.float64)
                    rhs[P[uniq_ca], np.arange(uniq_ca.size)] = 1.0
                    cols_inv = np.asarray(cholmod_factor(rhs), dtype=np.float64)
                    if cols_inv.ndim == 1:
                        cols_inv = cols_inv[:, None]
                    z_miss = cols_inv[P[miss_cb], inv_ca]
                    lev += np.bincount(
                        p_r[miss_mask],
                        weights=p_prod[miss_mask] * z_miss,
                        minlength=lev.size,
                    )

    big_rows = np.flatnonzero(row_nnz >= pair_limit)
    if big_rows.size:
        if cholmod_factor is not None and big_rows.size <= 64:
            exact_big = exact_leverage_for_candidate_rows(H_csr, big_rows, cholmod_factor)
            if exact_big is not None and exact_big.size == big_rows.size:
                lev[big_rows] = exact_big
                return lev
        Z_sym = (Z_low + SP_TRIL(Z_low, k=-1).T).tocsr()
        H_big = H_p[big_rows]
        proj_big = H_big @ Z_sym
        lev[big_rows] = np.asarray(H_big.multiply(proj_big).sum(axis=1)).ravel()
    return lev


def measurement_leverage(H, gain_inv: np.ndarray) -> np.ndarray:
    """Compute diag(H gain_inv H.T) for dense or sparse measurement Jacobians."""
    projected = H @ gain_inv
    if is_sparse_matrix(H):
        return np.asarray(H.multiply(projected).sum(axis=1)).ravel()
    return np.sum(projected * H, axis=1)


def inverse_gain_for_bad_data(gain: np.ndarray, dense_state_limit: int = 5000):
    """Return gain inverse for bad-data leverage, or None when full inverse is too large.

    Normalized-residual bad-data detection needs diag(H G^-1 H.T). A full inverse is
    acceptable for the named examples, but multi-10k-state systems would require
    tens of GB if forced dense. For those large runs the caller can fall back to a
    residual-only score instead of failing after the state estimate has converged.
    """
    if is_sparse_matrix(gain):
        if gain.shape[0] > dense_state_limit:
            return None
        gain = gain.toarray()
    else:
        gain = np.asarray(gain, dtype=np.float64)
    if gain.ndim == 2 and gain.shape[0] == gain.shape[1] and gain.shape[0] > 0:
        try:
            chol, info = DPOTRF(gain, lower=1, overwrite_a=0)
            if info == 0:
                inv_l, info2 = DPOTRI(chol, lower=1, overwrite_c=1)
                if info2 == 0:
                    iu = np.triu_indices(inv_l.shape[0], k=1)
                    inv_l[iu] = inv_l[(iu[1], iu[0])]
                    return inv_l
        except Exception:
            pass
    try:
        return np.linalg.inv(gain)
    except np.linalg.LinAlgError:
        # 禁止中大型矩阵跌落 O(N^3) 稠密 SVD pinv（单次 2000 阶即耗时 >6s）
        if gain.shape[0] > 512:
            return None
        return np.linalg.pinv(gain)


def measurement_leverage_for_bad_data(
    H,
    gain,
    *,
    cholmod_factor=None,
    aat_operand=None,
    dense_state_limit: int = 5000,
    sparse_threshold: Optional[int] = None,
    sparse_state_limit: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Compute exact diag(H G^-1 H^T) using sparse selected inversion for medium/large sparse SPD gains.

    稀疏 CHOLMOD 选元求逆仅依赖稀疏因子的非零模式，默认支持至 ``sparse_state_limit``（未显式传参时：
    若保持默认 ``dense_state_limit == 5000`` 则放宽至 ``100000``；若调用方显式压低 ``dense_state_limit``
    则沿用该上限以保持兼容）。小中规模（按 ``SEScaleProfile`` 依稀疏度自适应，默认稀疏图 ``<= 768``、
    稠密图 ``<= 512``）且已缓存 ``cholmod_factor`` 时，直接利用 ``cholmod_factor(I)`` 求精确逆矩阵，
    免去重新组装 ``gain`` 与重复分解。
    """
    n_state = int(H.shape[1])
    if n_state == 0:
        return np.zeros(int(H.shape[0]), dtype=np.float64)
    h_nnz = int(H.nnz) if is_sparse_matrix(H) else 0
    scale_prof = SEScaleProfile.for_problem(n_state, int(H.shape[0]), h_nnz)
    if sparse_threshold is None:
        eff_sparse_threshold = scale_prof.leverage_eye_max_state
    else:
        eff_sparse_threshold = int(sparse_threshold)
    effective_sparse_limit = (
        int(sparse_state_limit)
        if sparse_state_limit is not None
        else (scale_prof.sparse_leverage_max_state if int(dense_state_limit) == 5000 else int(dense_state_limit))
    )
    if (
        n_state <= eff_sparse_threshold
        and n_state <= dense_state_limit
        and cholmod_factor is not None
        and not getattr(cholmod_factor, "is_woodbury_solver", False)
    ):
        try:
            gain_inv = np.asarray(cholmod_factor(np.eye(n_state, dtype=np.float64)), dtype=np.float64)
            if gain_inv.shape == (n_state, n_state) and np.all(np.isfinite(gain_inv)):
                leverage = measurement_leverage(H, gain_inv)
                if np.all(np.isfinite(leverage)):
                    return leverage
        except Exception:
            pass
    if (
        is_sparse_matrix(H)
        and eff_sparse_threshold < n_state <= effective_sparse_limit
        and (cholmod_factor is not None or CHOLMOD_CHOLESKY is not None)
    ):
        try:
            factor = cholmod_factor
            if factor is not None and getattr(factor, "is_mkl_pardiso_factor", False):
                factor = factor.materialize_cholmod_factor(ordering_method=scale_prof.aat_ordering_method)
            if factor is None:
                if gain is None or CHOLMOD_CHOLESKY is None:
                    return None
                gain_csc = gain if getattr(gain, "format", None) == "csc" else gain.tocsc()
                factor = CHOLMOD_CHOLESKY(gain_csc)
            Z_low, P = _supernodal_selected_inverse_from_cholmod(factor, aat_operand=aat_operand)
            if Z_low is not None and P is not None:
                leverage = _measurement_leverage_from_selected_inv(
                    H, Z_low, P, cholmod_factor=factor
                )
                if leverage is not None and np.all(np.isfinite(leverage)):
                    return leverage
        except Exception:
            pass
    if n_state > dense_state_limit:
        return None
    if gain is None:
        return None
    gain_inv = inverse_gain_for_bad_data(gain, dense_state_limit=dense_state_limit)
    if gain_inv is None:
        return None
    return measurement_leverage(H, gain_inv)


DEFAULT_CRITICAL_REDUNDANCY_RATIO = 1e-4


def is_critical_leverage_row(
    cur_leverage: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    row_pos: int,
    *,
    critical_ratio: float = DEFAULT_CRITICAL_REDUNDANCY_RATIO,
) -> bool:
    """按无量纲相对冗余度 ``S_ii = 1 - w_i * H_ii`` 判定单行是否为临界/准临界量测。

    ``cur_leverage`` 存储带量纲的 ``H_ii = diag(H G^-1 H^T)``（量纲同 ``R_ii = 1/w_i``），
    必须乘以权重 ``w_i`` 还原为无量纲杠杆率 ``h_ii = w_i * H_ii in [0, 1]``，
    才能避免高权量测漏判、低权量测误判。
    """
    if cur_leverage is None or weights is None:
        return False
    pos = int(row_pos)
    if pos < 0 or pos >= cur_leverage.size or pos >= weights.size:
        return False
    w_i = float(weights[pos])
    if not np.isfinite(w_i) or w_i <= 0.0:
        return False
    lev_i = float(cur_leverage[pos])
    if not np.isfinite(lev_i):
        return False
    return (1.0 - w_i * lev_i) <= float(critical_ratio)


def compute_adaptive_bad_threshold(
    base_threshold: float,
    n_effective: int,
    *,
    normalized: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    active_mask: Optional[np.ndarray] = None,
    protected_mask: Optional[np.ndarray] = None,
    critical_mask: Optional[np.ndarray] = None,
    ref_count: int = 200,
    min_bucket_size: int = 30,
    hetero_weight_ratio: float = 1e3,
) -> Tuple[float, Optional[np.ndarray]]:
    """计算多重检验规模自适应与异构权值分桶校准的坏数据判定阈值 ``(scalar_thr, per_row_thr)``。

    设计原则：
    1. **下界守恒**：绝不低于用户给定的 ``base_threshold``；当 ``n_effective <= ref_count``（默认 200）时
       标量阈值严格等于 ``base_threshold``，保障中小网与既有单元测试零回归；
    2. **规模 FWER 守恒**：当 ``n_effective > ref_count`` 时，按
       ``tau_scale = sqrt(tau_base^2 + 2 * ln(n_effective / ref_count))`` 自动抬高基准阈值，
       消除万~十万节点大网下的多重检验假阳性爆发；
    3. **异构权值分桶稳健尺度校准**：当传入全量 ``normalized`` 与 ``weights`` 且有效权值跨度
       ``w_max / w_min >= hetero_weight_ratio`` 且 ``n_effective > ref_count`` 时，按 ``floor(log10(w))``
       分桶，对样本数 ``>= min_bucket_size`` 的桶使用 75% 分位数（25% 崩溃点，对常规坏数据免疫）
       估计该桶的噪声方差膨胀倍数 ``s_b = max(1.0, p75(r^N_b) / 1.15034938)``，仅向上防误删，
       不压低正常桶阈值。未发生明显膨胀时返回 ``per_row_thr=None`` 以免额外数组开销。
    """
    tau_base = float(base_threshold)
    if not np.isfinite(tau_base) or tau_base <= 0.0:
        return max(0.0, tau_base) if np.isfinite(tau_base) else 0.0, None
    m_eff = max(1, int(n_effective))
    ref_m = max(1, int(ref_count))
    if m_eff > ref_m:
        tau_scale = float(math.sqrt(tau_base * tau_base + 2.0 * math.log(float(m_eff) / float(ref_m))))
    else:
        tau_scale = tau_base

    if normalized is None or weights is None or m_eff <= ref_m:
        return tau_scale, None

    norm_arr = np.asarray(normalized, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    if norm_arr.size == 0 or norm_arr.size != w_arr.size:
        return tau_scale, None

    valid = np.isfinite(norm_arr) & np.isfinite(w_arr) & (w_arr > 0.0)
    if active_mask is not None:
        a_arr = np.asarray(active_mask, dtype=bool)
        if a_arr.size == valid.size:
            valid &= a_arr
    if protected_mask is not None:
        p_arr = np.asarray(protected_mask, dtype=bool)
        if p_arr.size == valid.size:
            valid &= ~p_arr
    if critical_mask is not None:
        c_arr = np.asarray(critical_mask, dtype=bool)
        if c_arr.size == valid.size:
            valid &= ~c_arr

    n_valid = int(np.count_nonzero(valid))
    if n_valid <= ref_m:
        return tau_scale, None

    w_valid = w_arr[valid]
    w_min = float(w_valid.min())
    w_max = float(w_valid.max())
    if w_min <= 0.0 or (w_max / w_min) < float(hetero_weight_ratio):
        return tau_scale, None

    # 标准正态 |Z| 的 75% 分位数为 Phi^-1(0.875) = 1.1503493803760079
    _HALF_NORM_Q75 = 1.1503493803760079
    buckets = np.floor(np.log10(np.maximum(w_arr, 1e-12))).astype(np.int32)
    valid_buckets = np.unique(buckets[valid])
    per_row_thr: Optional[np.ndarray] = None
    min_b = max(10, int(min_bucket_size))
    for b_val in valid_buckets.tolist():
        bm = valid & (buckets == int(b_val))
        if int(np.count_nonzero(bm)) < min_b:
            continue
        q75 = float(np.percentile(norm_arr[bm], 75.0))
        scale_b = q75 / _HALF_NORM_Q75
        if scale_b > 1.15:
            if per_row_thr is None:
                per_row_thr = np.full(norm_arr.size, tau_scale, dtype=np.float64)
            per_row_thr[bm] = tau_scale * scale_b
    return tau_scale, per_row_thr


def compute_normalized_residuals_for_bad_data(
    residual: np.ndarray,
    weights: np.ndarray,
    leverage: Optional[np.ndarray],
    *,
    is_pseudo: Optional[np.ndarray] = None,
    critical_ratio: float = DEFAULT_CRITICAL_REDUNDANCY_RATIO,
    critical_noise_sigma: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算坏数据辨识归一化残差并返回临界量测掩码 ``(normalized, critical_mask)``。

    - 对于普通冗余量测，按 ``|r| / sqrt(max(R_ii - H_ii, 1e-12))`` 计算；
    - 对于临界/准临界量测（相对冗余度 ``(R_ii - H_ii) / R_ii <= critical_ratio``，默认 ``1e-4``），
      因删去必致不可观测或灵敏度放大超百倍引发虚假误报，直接将归一化残差置 ``0.0``；
    - 对于受保护的硬物理等式约束（``is_pseudo`` 传入的保护掩码），归一化残差一律置 ``0.0``。
    """
    res = np.asarray(residual, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    R_diag = 1.0 / np.maximum(w, 1e-12)
    pseudo_arr = None
    if is_pseudo is not None:
        p_tmp = np.asarray(is_pseudo, dtype=bool)
        if p_tmp.size == res.size:
            pseudo_arr = p_tmp
    if leverage is None:
        omega_diag = np.maximum(R_diag, 1e-12)
        normalized = np.abs(res) / np.sqrt(omega_diag)
        if pseudo_arr is not None and np.any(pseudo_arr):
            normalized[pseudo_arr] = 0.0
        return normalized, np.zeros(res.size, dtype=bool)

    lev = np.asarray(leverage, dtype=np.float64)
    raw_omega = R_diag - lev
    critical_mask = raw_omega <= (float(critical_ratio) * R_diag)
    omega_diag = np.maximum(raw_omega, 1e-12)
    normalized = np.abs(res) / np.sqrt(omega_diag)
    if np.any(critical_mask):
        # 临界/准临界量测不可辨识且删除会破坏可观测性，统一归零屏蔽防数值放大爆炸
        normalized[critical_mask] = 0.0
    if pseudo_arr is not None and np.any(pseudo_arr):
        # 硬物理等式约束（零注入/功率平衡/开关零阻等式）绝不参与坏数据排除
        normalized[pseudo_arr] = 0.0
    return normalized, critical_mask


def update_leverage_after_row_removal(
    prev_leverage: Optional[np.ndarray],
    prev_H,
    prev_weights: np.ndarray,
    remove_pos: int,
    prev_cholmod_factor,
    *,
    keep_shape: bool = False,
    factor_is_post_removal: bool = False,
) -> Optional[np.ndarray]:
    """利用 Sherman-Morrison 秩 1 更新公式快速传播剔除单条量测后的杠杆率。

    - ``factor_is_post_removal=False``（传入 G_old 因子）：
      u = G_old^-1 h_k^T，H_ii^new = H_ii^old + (w_k / (1 - w_k h_k u)) * (h_i u)^2；
    - ``factor_is_post_removal=True``（传入 G_new 因子，正定逆向更新，适配原地覆写 CHOLMOD 因子）：
      u = G_new^-1 h_k^T，H_ii^new = H_ii^old + (w_k / (1 + w_k h_k u)) * (h_i u)^2。
    """
    if prev_leverage is None or prev_H is None or prev_cholmod_factor is None:
        return None
    m = int(prev_leverage.size)
    pos = int(remove_pos)
    if pos < 0 or pos >= m or int(prev_H.shape[0]) != m:
        return None
    w_k = float(prev_weights[pos])
    if not np.isfinite(w_k) or w_k <= 0.0:
        return None
    try:
        H_csr = prev_H if getattr(prev_H, "format", None) == "csr" else prev_H.tocsr()
        n = int(H_csr.shape[1])
        rhs = np.zeros(n, dtype=np.float64)
        rs = int(H_csr.indptr[pos])
        re = int(H_csr.indptr[pos + 1])
        cols_k = H_csr.indices[rs:re]
        vals_k = H_csr.data[rs:re]
        rhs[cols_k] = vals_k
        u = np.asarray(prev_cholmod_factor(rhs), dtype=np.float64).ravel()
        if u.size != n or not np.all(np.isfinite(u)):
            return None
        h_k_u = float(np.dot(vals_k, u[cols_k]))
        denom = (1.0 + w_k * h_k_u) if factor_is_post_removal else (1.0 - w_k * h_k_u)
        if not np.isfinite(denom) or denom <= 1e-9:
            return None
        alpha = w_k / denom
        Hu = np.asarray(H_csr @ u, dtype=np.float64).ravel()
        updated = prev_leverage + alpha * (Hu * Hu)
        if keep_shape:
            updated[pos] = 0.0
            return updated
        if pos == 0:
            return updated[1:].copy()
        if pos == m - 1:
            return updated[:-1].copy()
        return np.concatenate((updated[:pos], updated[pos + 1 :]))
    except Exception:
        return None


def exact_leverage_for_candidate_rows(
    H,
    candidate_rows: np.ndarray,
    cholmod_factor,
    *,
    chunk_size: Optional[int] = None,
) -> Optional[np.ndarray]:
    """使用当前 CHOLMOD 因子对候选量测行执行半边前代或回代，返回精确杠杆率。

    对于原生对称正定 CHOLMOD 因子 ``G = P^T L L^T P``，利用恒等式 ``h_j^T G^-1 h_j = ||L^-1 (P h_j)||_2^2``，
    仅执行半边下三角前代 ``solve_L``（免除后代 ``solve_Lt`` 与反置换 ``apply_Pt``）：
    - 极稀疏系统（平均每状态非零元 < 10）且候选数 >= 16 时走稀疏 CSC 右端消除树前代；
    - 常规超节点系统走列主序（Fortran order）L3 Cache 自适应分块稠密前代 + ``einsum``；
    - 对低秩 ``WoodburyRemovalSolver`` 或异常场景自动回退 ``cholmod_factor(rhs)``。
    """
    if H is None or cholmod_factor is None:
        return None
    rows = np.asarray(candidate_rows, dtype=np.intp).ravel()
    if rows.size == 0:
        return np.empty(0, dtype=np.float64)
    try:
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        H_sub = H_csr[rows]
        n = int(H_csr.shape[1])
        k = int(rows.size)
        profile = SEScaleProfile.for_problem(n, int(H_csr.shape[0]), int(H_csr.nnz))
        step = max(int(chunk_size) if chunk_size is not None else int(profile.exact_leverage_chunk_size), 1)
        indptr = H_sub.indptr
        indices = H_sub.indices
        data = H_sub.data
        lev = np.empty(k, dtype=np.float64)

        # 快速路径 1：低秩 WoodburyRemovalSolver 半边前代恒等式（实测提速 ~2.6x，免分配 (n, k) 大矩阵）：
        # h_j^T (G_0^-1 + U M^-1 U^T) h_j = ||L_0^-1 (P h_j)||_2^2 + (h_j^T U) M^-1 (h_j^T U)^T
        if getattr(cholmod_factor, "is_woodbury_solver", False):
            base_fac = getattr(cholmod_factor, "_factor", None)
            wb_U = getattr(cholmod_factor, "_U", None)
            if (
                base_fac is not None
                and wb_U is not None
                and (
                    (hasattr(base_fac, "solve_L") and hasattr(base_fac, "P"))
                    or getattr(base_fac, "is_mkl_pardiso_factor", False)
                )
            ):
                base_lev = exact_leverage_for_candidate_rows(H_csr, rows, base_fac, chunk_size=chunk_size)
                if base_lev is not None and base_lev.size == k:
                    if getattr(cholmod_factor, "_is_single", False):
                        v_wb = np.asarray(H_sub @ wb_U, dtype=np.float64).ravel()
                        lev_wb = base_lev + float(getattr(cholmod_factor, "_single_scale", 0.0)) * (v_wb * v_wb)
                    else:
                        m_chol_wb = getattr(cholmod_factor, "_m_chol", None)
                        if m_chol_wb is not None:
                            V_wb = np.asarray(H_sub @ wb_U, dtype=np.float64)
                            X_wb = CHO_SOLVE(m_chol_wb, V_wb.T, check_finite=False).T
                            lev_wb = base_lev + np.sum(V_wb * X_wb, axis=1)
                        else:
                            lev_wb = None
                    if lev_wb is not None and np.all(np.isfinite(lev_wb)):
                        return lev_wb

        # 快速路径 2A：多线程 MKL PARDISO 因子半边前代（phase=331 直接解 L y = P b，免手动置换）
        if getattr(cholmod_factor, "is_mkl_pardiso_factor", False):
            try:
                p_step = max(step, 64) if chunk_size is None else step
                rhs_buf_f = np.zeros((n, min(k, p_step)), dtype=np.float64, order="F")
                ok_p = True
                for start in range(0, k, p_step):
                    end = min(start + p_step, k)
                    b = end - start
                    rhs = rhs_buf_f if b == rhs_buf_f.shape[1] else rhs_buf_f[:, :b]
                    rhs.fill(0.0)
                    for col in range(b):
                        j = start + col
                        s = int(indptr[j])
                        e = int(indptr[j + 1])
                        if e > s:
                            rhs[indices[s:e], col] = data[s:e]
                    y = np.asarray(cholmod_factor.solve_L_unpermuted(rhs), dtype=np.float64)
                    if y.ndim == 1:
                        y = y[:, None]
                    if y.shape != (n, b) or not np.all(np.isfinite(y)):
                        ok_p = False
                        break
                    lev[start:end] = np.einsum("ij,ij->j", y, y)
                if ok_p:
                    return lev
            except Exception:
                pass

        # 快速路径 2B：原生 CHOLMOD 因子半边前代 solve_L（省去一半三角代与反置换）
        if (
            not getattr(cholmod_factor, "is_woodbury_solver", False)
            and hasattr(cholmod_factor, "solve_L")
            and hasattr(cholmod_factor, "P")
        ):
            try:
                P = np.asarray(cholmod_factor.P(), dtype=np.intp)
                if P.size == n:
                    invP = np.empty(n, dtype=np.intp)
                    invP[P] = np.arange(n, dtype=np.intp)
                    if profile.exact_leverage_use_sparse_rhs and k >= 16:
                        Ht_coo = H_sub.T.tocoo()
                        B_perm = SP_CSC_MATRIX(
                            (Ht_coo.data, (invP[Ht_coo.row], Ht_coo.col)),
                            shape=(n, k),
                            dtype=np.float64,
                        )
                        Y_sp = cholmod_factor.solve_L(B_perm, use_LDLt_decomposition=False)
                        if is_sparse_matrix(Y_sp):
                            Y_csc = Y_sp if getattr(Y_sp, "format", None) == "csc" else Y_sp.tocsc()
                            if Y_csc.shape == (n, k) and np.all(np.isfinite(Y_csc.data)):
                                y_sq = Y_csc.data * Y_csc.data
                                csum = np.empty(y_sq.size + 1, dtype=np.float64)
                                csum[0] = 0.0
                                if y_sq.size:
                                    np.cumsum(y_sq, out=csum[1:])
                                lev[:] = csum[Y_csc.indptr[1:]] - csum[Y_csc.indptr[:-1]]
                                return lev
                        else:
                            y_arr = np.asarray(Y_sp, dtype=np.float64)
                            if y_arr.ndim == 1:
                                y_arr = y_arr[:, None]
                            if y_arr.shape == (n, k) and np.all(np.isfinite(y_arr)):
                                lev[:] = np.einsum("ij,ij->j", y_arr, y_arr)
                                return lev
                    rhs_buf_f = np.zeros((n, min(k, step)), dtype=np.float64, order="F")
                    ok_l = True
                    for start in range(0, k, step):
                        end = min(start + step, k)
                        b = end - start
                        rhs = rhs_buf_f if b == rhs_buf_f.shape[1] else rhs_buf_f[:, :b]
                        rhs.fill(0.0)
                        for col in range(b):
                            j = start + col
                            s = int(indptr[j])
                            e = int(indptr[j + 1])
                            if e > s:
                                rhs[invP[indices[s:e]], col] = data[s:e]
                        y = np.asarray(cholmod_factor.solve_L(rhs, use_LDLt_decomposition=False), dtype=np.float64)
                        if y.ndim == 1:
                            y = y[:, None]
                        if y.shape != (n, b) or not np.all(np.isfinite(y)):
                            ok_l = False
                            break
                        lev[start:end] = np.einsum("ij,ij->j", y, y)
                    if ok_l:
                        return lev
            except Exception:
                pass

        rhs_buf = np.zeros((n, min(k, step)), dtype=np.float64)
        for start in range(0, k, step):
            end = min(start + step, k)
            b = end - start
            rhs = rhs_buf if b == rhs_buf.shape[1] else rhs_buf[:, :b]
            rhs.fill(0.0)
            for col in range(b):
                j = start + col
                s = int(indptr[j])
                e = int(indptr[j + 1])
                if e > s:
                    rhs[indices[s:e], col] = data[s:e]
            sol = np.asarray(cholmod_factor(rhs), dtype=np.float64)
            if sol.ndim == 1:
                sol = sol[:, None]
            if sol.shape != (n, b) or not np.all(np.isfinite(sol)):
                return None
            for col in range(b):
                j = start + col
                s = int(indptr[j])
                e = int(indptr[j + 1])
                lev[j] = float(np.dot(data[s:e], sol[indices[s:e], col])) if e > s else 0.0
        return lev
    except Exception:
        return None


def _get_cached_csc_for_graph(H_csr):
    """复用同一稀疏 CSR 雅可比矩阵上的 CSC 视图，避免单轮坏数据筛查与批选重复转置。"""
    cached = getattr(H_csr, "_se_cached_csc_graph", None)
    if (
        cached is not None
        and cached.shape == H_csr.shape
        and cached.nnz == H_csr.nnz
    ):
        return cached
    H_csc = H_csr.tocsc()
    try:
        setattr(H_csr, "_se_cached_csc_graph", H_csc)
    except Exception:
        pass
    return H_csc


def select_independent_bad_data_batch(
    H,
    candidate_rows: Sequence[int],
    candidate_norms: Sequence[float],
    threshold: float,
    *,
    min_batch_norm: float = 5.0,
    max_batch: int = 32,
    min_hops: int = 3,
) -> List[int]:
    """基于雅可比稀疏二分图选择拓扑隔离（>= min_hops 跳，默认 3 跳）的高置信度坏数据批次。

    输入 ``candidate_rows`` 与 ``candidate_norms`` 应按归一化残差严格降序排列。
    返回入选批次在输入候选序列中的下标列表（始终至少包含 ``[0]``）。
    每选入一条量测后封锁其关联状态列的 ``min_hops - 1`` 跳状态邻域，确保同批坏数据无残差相互污染。
    """
    n_cand = len(candidate_rows)
    if n_cand <= 1 or max_batch <= 1 or H is None or not is_sparse_matrix(H):
        return [0] if n_cand >= 1 else []
    norm_gate = max(float(threshold), float(min_batch_norm))
    if float(candidate_norms[0]) < norm_gate:
        return [0]

    H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
    H_csc = _get_cached_csc_for_graph(H_csr)
    n_rows, n_cols = int(H_csr.shape[0]), int(H_csr.shape[1])
    csr_indptr = H_csr.indptr
    csr_indices = H_csr.indices
    csc_indptr = H_csc.indptr
    csc_indices = H_csc.indices

    blocked_cols = np.zeros(n_cols, dtype=bool)
    selected: List[int] = []
    block_hops = max(2, int(min_hops) - 1)

    def _gather_adj(indptr_arr: np.ndarray, indices_arr: np.ndarray, idx_arr: np.ndarray) -> np.ndarray:
        if idx_arr.size == 0:
            return np.empty(0, dtype=indices_arr.dtype)
        starts = indptr_arr[idx_arr]
        ends = indptr_arr[idx_arr + 1]
        if idx_arr.size == 1:
            return indices_arr[int(starts[0]) : int(ends[0])]
        chunks = [indices_arr[s:e] for s, e in zip(starts.tolist(), ends.tolist()) if e > s]
        if not chunks:
            return np.empty(0, dtype=indices_arr.dtype)
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    def _block_two_hop(row_idx: int) -> None:
        rs = int(csr_indptr[row_idx])
        re = int(csr_indptr[row_idx + 1])
        if re <= rs:
            return
        frontier = csr_indices[rs:re]
        blocked_cols[frontier] = True
        for hop in range(block_hops):
            r_adj = _gather_adj(csc_indptr, csc_indices, frontier)
            if r_adj.size == 0:
                return
            if r_adj.size > 1:
                r_adj = np.unique(r_adj)
            c_all = _gather_adj(csr_indptr, csr_indices, r_adj)
            if c_all.size == 0:
                return
            if hop + 1 < block_hops:
                c_new = c_all[~blocked_cols[c_all]]
                if c_new.size == 0:
                    return
                if c_new.size > 1:
                    c_new = np.unique(c_new)
                blocked_cols[c_new] = True
                frontier = c_new
            else:
                blocked_cols[c_all] = True

    for idx in range(n_cand):
        if len(selected) >= int(max_batch):
            break
        if idx > 0 and float(candidate_norms[idx]) < norm_gate:
            break
        r_pos = int(candidate_rows[idx])
        if r_pos < 0 or r_pos >= n_rows:
            if not selected:
                selected.append(idx)
            continue
        rs = int(csr_indptr[r_pos])
        re = int(csr_indptr[r_pos + 1])
        row_cols = csr_indices[rs:re]
        if selected and row_cols.size and np.any(blocked_cols[row_cols]):
            continue
        selected.append(idx)
        _block_two_hop(r_pos)

    return selected if selected else [0]


def _woodbury_get_cached_u_columns(
    H_csr,
    pos_arr: np.ndarray,
    base_factor,
    u_col_cache: Optional[Dict],
    *,
    extra_rhs: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """按锚点 (factor, H) 身份增量缓存并拼接 U = G_base^-1 H_K^T；可选顺带求解首列 extra_rhs。"""
    n = int(H_csr.shape[1])
    b = int(pos_arr.size)
    pos_list = [int(p) for p in pos_arr.tolist()]
    cache_tag = (id(base_factor), id(H_csr))
    use_cache = isinstance(u_col_cache, dict)
    if use_cache:
        if u_col_cache.get("_tag") != cache_tag:
            u_col_cache.clear()
            u_col_cache["_tag"] = cache_tag
        miss_local = [j for j, rp in enumerate(pos_list) if rp not in u_col_cache]
    else:
        miss_local = list(range(b))

    n_miss = len(miss_local)
    has_extra = extra_rhs is not None
    u0 = None
    if has_extra or n_miss > 0:
        n_cols = (1 if has_extra else 0) + n_miss
        mat = np.zeros((n, n_cols), dtype=np.float64)
        col_off = 0
        if has_extra:
            mat[:, 0] = extra_rhs
            col_off = 1
        if n_miss > 0:
            miss_rows = np.asarray([pos_list[j] for j in miss_local], dtype=np.intp)
            H_miss = H_csr[miss_rows]
            m_indptr = H_miss.indptr
            m_indices = H_miss.indices
            m_data = H_miss.data
            for k_i in range(n_miss):
                s = int(m_indptr[k_i])
                e = int(m_indptr[k_i + 1])
                if e > s:
                    mat[m_indices[s:e], col_off + k_i] = m_data[s:e]
        sol = np.asarray(base_factor(mat[:, 0] if n_cols == 1 else mat), dtype=np.float64)
        if sol.ndim == 1:
            sol = sol[:, None]
        if sol.shape != (n, n_cols) or not np.all(np.isfinite(sol)):
            return None, None
        if has_extra:
            u0 = sol[:, 0].copy()
        if use_cache:
            for k_i, j_loc in enumerate(miss_local):
                u_col_cache[pos_list[j_loc]] = sol[:, col_off + k_i].copy()
            U = np.empty((n, b), dtype=np.float64)
            for j, rp in enumerate(pos_list):
                U[:, j] = u_col_cache[rp]
        else:
            U = sol[:, col_off:]
    else:
        U = np.empty((n, b), dtype=np.float64)
        for j, rp in enumerate(pos_list):
            U[:, j] = u_col_cache[rp]
    return u0, U


def update_leverage_after_batch_removal(
    prev_leverage: Optional[np.ndarray],
    prev_H,
    prev_weights: np.ndarray,
    remove_positions: Sequence[int],
    prev_cholmod_factor,
    *,
    keep_shape: bool = False,
    factor_is_post_removal: bool = False,
    u_col_cache: Optional[Dict] = None,
) -> Optional[np.ndarray]:
    """利用 Block Woodbury 秩 k 矩阵求逆引理批量传播剔除多条量测后的全量杠杆率。

    - ``factor_is_post_removal=False``：U = G_old^-1 H_K^T，M = diag(w_K^-1) - H_K U；
    - ``factor_is_post_removal=True``（正定逆向更新）：U = G_new^-1 H_K^T，M = diag(w_K^-1) + H_K U，
      严格正定且适配原地覆写 CHOLMOD 因子。
    """
    pos_arr = np.asarray(remove_positions, dtype=np.intp).ravel()
    if pos_arr.size == 0:
        return None if prev_leverage is None else prev_leverage.copy()
    if pos_arr.size == 1 and (u_col_cache is None or factor_is_post_removal):
        return update_leverage_after_row_removal(
            prev_leverage,
            prev_H,
            prev_weights,
            int(pos_arr[0]),
            prev_cholmod_factor,
            keep_shape=keep_shape,
            factor_is_post_removal=factor_is_post_removal,
        )
    if prev_leverage is None or prev_H is None or prev_cholmod_factor is None:
        return None
    m = int(prev_leverage.size)
    if int(prev_H.shape[0]) != m or np.any(pos_arr < 0) or np.any(pos_arr >= m):
        return None
    w_k = np.asarray(prev_weights[pos_arr], dtype=np.float64)
    if not np.all(np.isfinite(w_k)) or np.any(w_k <= 0.0):
        return None
    try:
        H_csr = prev_H if getattr(prev_H, "format", None) == "csr" else prev_H.tocsr()
        H_k = H_csr[pos_arr]
        eff_cache = u_col_cache if (not factor_is_post_removal) else None
        _, U = _woodbury_get_cached_u_columns(H_csr, pos_arr, prev_cholmod_factor, eff_cache)
        if U is None:
            return None
        H_k_U = np.asarray(H_k @ U, dtype=np.float64)
        M = (np.diag(1.0 / w_k) + H_k_U) if factor_is_post_removal else (np.diag(1.0 / w_k) - H_k_U)
        M = 0.5 * (M + M.T)
        # 检查对称正定性（防止批次中隐含临界量测或临界量测对）
        if np.any(np.diag(M) <= 1e-9 / np.maximum(w_k, 1e-12)):
            return None
        m_chol = CHO_FACTOR(M, lower=True, check_finite=False)
        V = np.asarray(H_csr @ U, dtype=np.float64)
        X = CHO_SOLVE(m_chol, V.T, check_finite=False).T
        delta = np.sum(V * X, axis=1)
        if not np.all(np.isfinite(delta)):
            return None
        updated = prev_leverage + delta
        if keep_shape:
            updated[pos_arr] = 0.0
            return updated
        keep = np.ones(m, dtype=bool)
        keep[pos_arr] = False
        return updated[keep].copy()
    except Exception:
        return None


class WoodburyRemovalSolver:
    """缓存删行 Exact Block Woodbury 辅助因子，支持同一删行批次的单/多右端低成本线性回代。"""

    is_woodbury_solver: bool = True

    def __init__(
        self,
        prev_cholmod_factor,
        H_k,
        U: np.ndarray,
        *,
        m_chol=None,
        single_cols: Optional[np.ndarray] = None,
        single_vals: Optional[np.ndarray] = None,
        single_scale: float = 0.0,
    ):
        self._factor = prev_cholmod_factor
        self._H_k = H_k
        self._U = U
        self._m_chol = m_chol
        self._single_cols = single_cols
        self._single_vals = single_vals
        self._single_scale = float(single_scale)
        self._is_single = m_chol is None

    def solve(self, rhs: np.ndarray) -> Optional[np.ndarray]:
        """利用缓存的 G_old 因子与秩 k 修正，支持 1D 向量或 2D 多右端矩阵回代计算 G_new^-1 rhs。"""
        try:
            rhs_arr = np.asarray(rhs, dtype=np.float64)
            is_1d = rhs_arr.ndim == 1
            if is_1d:
                u0 = np.asarray(self._factor(rhs_arr), dtype=np.float64).ravel()
                if not np.all(np.isfinite(u0)):
                    return None
                if self._is_single:
                    h_k_u0 = (
                        float(np.dot(self._single_vals, u0[self._single_cols]))
                        if self._single_cols is not None and self._single_cols.size > 0
                        else 0.0
                    )
                    dx = u0 + (self._single_scale * h_k_u0) * self._U
                else:
                    H_k_u0 = np.asarray(self._H_k @ u0, dtype=np.float64)
                    y = CHO_SOLVE(self._m_chol, H_k_u0, check_finite=False)
                    dx = u0 + self._U @ y
            else:
                u0 = np.asarray(self._factor(rhs_arr), dtype=np.float64)
                if u0.ndim == 1:
                    u0 = u0[:, None]
                if not np.all(np.isfinite(u0)):
                    return None
                if self._is_single:
                    if self._single_cols is not None and self._single_cols.size > 0:
                        h_k_u0 = self._single_vals @ u0[self._single_cols, :]
                        dx = u0 + np.outer(self._U, self._single_scale * h_k_u0)
                    else:
                        dx = u0
                else:
                    H_k_u0 = np.asarray(self._H_k @ u0, dtype=np.float64)
                    y = CHO_SOLVE(self._m_chol, H_k_u0, check_finite=False)
                    dx = u0 + self._U @ y
            if not np.all(np.isfinite(dx)):
                return None
            return dx
        except Exception:
            return None

    def __call__(self, rhs: np.ndarray) -> np.ndarray:
        """兼容 CHOLMOD 因子调用协议，可直接用于杠杆率传播与候选行精确回代。"""
        out = self.solve(rhs)
        if out is None:
            raise ValueError("WoodburyRemovalSolver failed to solve linear system")
        return out


def solve_woodbury_after_row_removal(
    prev_H,
    prev_weights: np.ndarray,
    remove_positions: Sequence[int],
    rhs: np.ndarray,
    prev_cholmod_factor,
    *,
    max_batch: int = 32,
    return_solver: bool = False,
    u_col_cache: Optional[Dict] = None,
):
    """基于上一轮 G_old 缓存 CHOLMOD 因子，用 Exact Block Woodbury 精确求解删行后的正规方程。

    给定 G_new = G_old - H_K^T W_K H_K（1 <= |K| <= max_batch），利用同锚点增量列缓存 ``u_col_cache``
    仅对新增移除行回代解出 U = G_old^-1 H_K^T，再通过小矩阵 M = W_K^-1 - H_K U 精确算出：
      dx = u0 + U M^-1 (H_K u0) = G_new^-1 rhs
    若 M 非正定或接近奇异（说明被删量测含临界量测导致失观），返回 None 交由标准数值分解处理。
    当 ``return_solver=True`` 时返回 ``(dx, WoodburyRemovalSolver)`` 以供同轮后续微步复用。
    """
    none_ret = (None, None) if return_solver else None
    if prev_H is None or prev_cholmod_factor is None or not is_sparse_matrix(prev_H):
        return none_ret
    pos_arr = np.asarray(remove_positions, dtype=np.intp).ravel()
    b = int(pos_arr.size)
    if b == 0 or b > int(max_batch):
        return none_ret
    m, n = int(prev_H.shape[0]), int(prev_H.shape[1])
    rhs_arr = np.asarray(rhs, dtype=np.float64).ravel()
    if rhs_arr.size != n or np.any(pos_arr < 0) or np.any(pos_arr >= m):
        return none_ret
    w_k = np.asarray(prev_weights[pos_arr], dtype=np.float64)
    if not np.all(np.isfinite(w_k)) or np.any(w_k <= 0.0):
        return none_ret
    try:
        H_csr = prev_H if getattr(prev_H, "format", None) == "csr" else prev_H.tocsr()
        H_k = H_csr[pos_arr]
        k_indptr = H_k.indptr
        k_indices = H_k.indices
        k_data = H_k.data
        u0, U = _woodbury_get_cached_u_columns(
            H_csr, pos_arr, prev_cholmod_factor, u_col_cache, extra_rhs=rhs_arr
        )
        if u0 is None or U is None:
            return none_ret
        solver_obj = None
        if b == 1:
            u1 = U[:, 0]
            s0 = int(k_indptr[0])
            e0 = int(k_indptr[1])
            cols = k_indices[s0:e0]
            vals = k_data[s0:e0]
            h_k_u = float(np.dot(vals, u1[cols])) if e0 > s0 else 0.0
            denom = 1.0 - float(w_k[0]) * h_k_u
            if not np.isfinite(denom) or denom <= 1e-7:
                return none_ret
            h_k_u0 = float(np.dot(vals, u0[cols])) if e0 > s0 else 0.0
            scale = float(w_k[0]) / denom
            dx = u0 + (scale * h_k_u0) * u1
            if return_solver:
                solver_obj = WoodburyRemovalSolver(
                    prev_cholmod_factor,
                    H_k,
                    u1,
                    single_cols=cols.copy(),
                    single_vals=vals.copy(),
                    single_scale=scale,
                )
        else:
            H_k_U = np.asarray(H_k @ U, dtype=np.float64)
            M = np.diag(1.0 / w_k) - H_k_U
            M = 0.5 * (M + M.T)
            if np.any(np.diag(M) <= 1e-7 / np.maximum(w_k, 1e-12)):
                return none_ret
            m_chol = CHO_FACTOR(M, lower=True, check_finite=False)
            H_k_u0 = np.asarray(H_k @ u0, dtype=np.float64)
            y = CHO_SOLVE(m_chol, H_k_u0, check_finite=False)
            dx = u0 + U @ y
            if return_solver:
                solver_obj = WoodburyRemovalSolver(
                    prev_cholmod_factor,
                    H_k,
                    U,
                    m_chol=m_chol,
                )
        if not np.all(np.isfinite(dx)):
            return none_ret
        return (dx, solver_obj) if return_solver else dx
    except Exception:
        return none_ret


def _strict_upper_bound_bad_data_cert(
    cur_residual: np.ndarray,
    cur_H,
    cur_weights: np.ndarray,
    cur_is_pseudo: Optional[np.ndarray],
    cur_cholmod_factor,
    threshold: float,
    *,
    active_mask: Optional[np.ndarray] = None,
    max_exact_rows: int = 256,
) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    """基于严格残差方差下限的零误差稀疏坏数据认证。

    对于任意量测 i，与 ``compute_normalized_residuals_for_bad_data`` 严格一致：
    - 临界/准临界量测（``(R_ii - H_ii) / R_ii <= DEFAULT_CRITICAL_REDUNDANCY_RATIO``）与硬物理保护行均归零；
    - 非临界量测必满足残差方差 ``Omega_ii > max(DEFAULT_CRITICAL_REDUNDANCY_RATIO * R_ii, 1e-12)``，
      故 ``r_Ni < |r_i| / sqrt(max(DEFAULT_CRITICAL_REDUNDANCY_RATIO * R_ii, 1e-12))``。
    因此仅需对可能超阈值的稀疏集合 ``S`` 做 CHOLMOD 精确回代：
    - ``|S| == 0``：0 ms 严格认证全网无坏数据（``"fast_clean"``）；
    - ``0 < |S| <= max_exact_rows``：对 ``S`` 精确求杠杆率与归一化残差，若全不超阈则 100% 严格认证无坏数据
      （``"verified_clean"``），否则精确返回全部超阈值坏数据（``"candidates"``）。
    """
    res = np.asarray(cur_residual, dtype=np.float64)
    w_arr = np.asarray(cur_weights, dtype=np.float64)
    thr = float(threshold)
    if res.size == 0 or thr <= 0.0 or w_arr.size != res.size:
        return "skip", None, None
    active_w = w_arr > 0.0
    if active_mask is not None:
        active_w &= np.asarray(active_mask, dtype=bool)
    if cur_is_pseudo is not None:
        pseudo_arr = np.asarray(cur_is_pseudo, dtype=bool)
        if pseudo_arr.size == res.size:
            active_w &= ~pseudo_arr
    if not np.any(active_w):
        return "fast_clean", np.empty(0, dtype=np.intp), np.zeros(res.size, dtype=np.float64)
    abs_r = np.abs(res)
    R_diag = 1.0 / np.maximum(w_arr, 1e-12)
    ub = abs_r / np.sqrt(np.maximum(DEFAULT_CRITICAL_REDUNDANCY_RATIO * R_diag, 1e-12))
    possible_rows = np.flatnonzero(active_w & (ub > thr))
    if possible_rows.size == 0:
        ub_out = np.where(active_w, ub, 0.0)
        return "fast_clean", np.empty(0, dtype=np.intp), ub_out
    if (
        possible_rows.size > int(max_exact_rows)
        or cur_H is None
        or cur_cholmod_factor is None
        or not is_sparse_matrix(cur_H)
        or int(cur_H.shape[0]) != int(res.size)
    ):
        return "skip", None, None
    exact_lev = exact_leverage_for_candidate_rows(cur_H, possible_rows, cur_cholmod_factor)
    if exact_lev is None or exact_lev.size != possible_rows.size:
        return "skip", None, None
    cand_pseudo = None if cur_is_pseudo is None else np.asarray(cur_is_pseudo, dtype=bool)[possible_rows]
    exact_norm, _ = compute_normalized_residuals_for_bad_data(
        res[possible_rows],
        w_arr[possible_rows],
        exact_lev,
        is_pseudo=cand_pseudo,
    )
    hit_local = np.flatnonzero(exact_norm > thr)
    if hit_local.size == 0:
        if getattr(cur_cholmod_factor, "is_woodbury_solver", False):
            return "skip", None, None
        ub_out = np.where(active_w, ub, 0.0)
        ub_out[possible_rows] = exact_norm
        return "verified_clean", np.empty(0, dtype=np.intp), ub_out
    hit_rows = possible_rows[hit_local]
    hit_norms = exact_norm[hit_local]
    order = np.argsort(hit_norms)[::-1]
    return "candidates", hit_rows[order], hit_norms[order]


def screen_bad_data_after_removal(
    prev_leverage: Optional[np.ndarray],
    prev_H,
    prev_weights: np.ndarray,
    remove_pos,
    prev_cholmod_factor,
    cur_residual: np.ndarray,
    cur_H,
    cur_weights: np.ndarray,
    cur_is_pseudo: Optional[np.ndarray],
    cur_cholmod_factor,
    threshold: float,
    *,
    state_delta_inf: float = 0.0,
    state_delta_vec: Optional[np.ndarray] = None,
    max_state_delta: float = 0.25,
    max_candidates: int = 256,
    top_extra: int = 24,
    active_mask: Optional[np.ndarray] = None,
    keep_shape: bool = False,
    allow_fast_final_verify: bool = False,
    final_verify_top: int = 32,
    u_col_cache: Optional[Dict] = None,
) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """坏数据剔除中间轮的快速精确筛查（严格上界稀疏认证 + Block Woodbury 传播 + 多右端精确回代）。

    返回 ``(status, updated_leverage, bad_rows, bad_norm_values)``：
    - ``"fast_clean"``：满足严格残差上界，零成本确认无坏数据；
    - ``"verified_clean"``：严格上界候选集（<=256 行）经当前因子精确回代全不超阈，或
      ``allow_fast_final_verify=True`` 时 Top 候选远低于阈值，安全免除终轮全矩阵选元求逆；
    - ``"candidates"``：已用 ``cur_cholmod_factor`` 精确算出超过阈值的候选坏数据行及精确归一化残差（按降序排列）；
    - ``"need_full"``：准备终止需要终轮全量精确认证，或候选过多/非线性偏移较大，回退到完整 ``identify_bad_data``。
    """
    res = np.asarray(cur_residual, dtype=np.float64)
    if active_mask is not None:
        mask_arr = np.asarray(active_mask, dtype=bool)
    else:
        mask_arr = None
    cert_status, cert_rows, cert_norms = _strict_upper_bound_bad_data_cert(
        res,
        cur_H,
        cur_weights,
        cur_is_pseudo,
        cur_cholmod_factor,
        threshold,
        active_mask=mask_arr,
    )
    if cert_status in ("fast_clean", "verified_clean", "candidates"):
        return cert_status, None, cert_rows, cert_norms

    def _try_lower_bound_screen():
        if cur_H is None or cur_cholmod_factor is None:
            return None
        lb_gate = max(float(threshold), 5.0)
        st, h_rows, h_norms = screen_initial_bad_data_fast(
            res,
            cur_H,
            cur_weights,
            cur_is_pseudo,
            cur_cholmod_factor,
            threshold,
            max_candidates=384,
            min_truncated_norm=lb_gate,
            active_mask=mask_arr,
            min_lower_bound=lb_gate,
            skip_upper_cert=True,
        )
        if st == "candidates" and h_rows is not None and h_rows.size > 0:
            return "candidates", None, h_rows, h_norms
        return None

    # 区分全网剧烈偏移与极少数孤立盲区/坏点节点的局部偏移：
    # 仅统计连接到至少 1 条有效非伪量测的状态列（纯伪量测盲节点不参与任何真实量测残差判定，避免 50% 大盲区团块联动触发空跑全量求逆）；
    # 排除已单独纳入精确回代的动点列 moved_cols 后，若未动区真实状态 RMS <= 1e-2 且动点列 <= 64，
    # 允许继续走 Block Woodbury + 动点关联行精确回代 + 未动区实测杠杆误差自检。
    moved_cols = np.empty(0, dtype=np.intp)
    local_only_shift = False
    if state_delta_vec is not None and cur_H is not None:
        dx_arr = np.asarray(state_delta_vec, dtype=np.float64).ravel()
        n_cols_cur = int(cur_H.shape[1])
        if dx_arr.size == n_cols_cur and dx_arr.size > 0:
            abs_dx = np.abs(dx_arr)
            eval_cols_mask: Optional[np.ndarray] = None
            if cur_is_pseudo is not None and is_sparse_matrix(cur_H):
                pseudo_arr = np.asarray(cur_is_pseudo, dtype=bool)
                if pseudo_arr.size == int(cur_H.shape[0]) and np.any(pseudo_arr):
                    w_pos_mask = np.asarray(cur_weights, dtype=np.float64) > 0.0
                    real_row_mask = w_pos_mask & (~pseudo_arr)
                    if mask_arr is not None:
                        real_row_mask &= mask_arr
                    if np.any(real_row_mask):
                        H_csr_tmp = cur_H if getattr(cur_H, "format", None) == "csr" else cur_H.tocsr()
                        real_rows = np.flatnonzero(real_row_mask)
                        eval_cols_mask = np.zeros(n_cols_cur, dtype=bool)
                        eval_cols_mask[H_csr_tmp[real_rows].indices] = True
            moved_mask = abs_dx > 5e-2
            if eval_cols_mask is not None:
                moved_mask &= eval_cols_mask
            moved_cols = np.flatnonzero(moved_mask)
            moved_col_cap = max(64, min(512, n_cols_cur // 64))
            if moved_cols.size <= moved_col_cap:
                if eval_cols_mask is not None:
                    unmoved_mask_cols = eval_cols_mask & (~moved_mask)
                else:
                    unmoved_mask_cols = ~moved_mask
                unmoved_dx = abs_dx[unmoved_mask_cols]
                unmoved_rms = (
                    float(np.sqrt(float(np.dot(unmoved_dx, unmoved_dx)) / unmoved_dx.size))
                    if unmoved_dx.size > 0
                    else 0.0
                )
                local_only_shift = unmoved_rms <= 1e-2

    batch_too_large_for_woodbury = (not isinstance(remove_pos, (int, np.integer))) and len(remove_pos) > 128
    if (
        prev_leverage is None
        or prev_H is None
        or cur_H is None
        or cur_cholmod_factor is None
        or batch_too_large_for_woodbury
        or (float(state_delta_inf) > float(max_state_delta) and not local_only_shift)
    ):
        lb_hit = _try_lower_bound_screen()
        if lb_hit is not None:
            return lb_hit
        return "need_full", None, None, None
    # 当 cur_cholmod_factor 为绑定原锚点因子的 WoodburyRemovalSolver 且提供了 u_col_cache 时，
    # 直接用锚点因子 + 增量缓存的 U 列走前向 Woodbury 传播（factor_is_post_removal=False），免除重复回代；
    # 否则使用当前 G_new 最新因子做正定逆向 Woodbury 传播（适配共享因子原地覆写）。
    use_anchor_cache_prop = (
        u_col_cache is not None
        and getattr(cur_cholmod_factor, "is_woodbury_solver", False)
        and getattr(cur_cholmod_factor, "_factor", None) is not None
        and getattr(cur_cholmod_factor, "_factor", None) is prev_cholmod_factor
    )
    if use_anchor_cache_prop:
        rem_seq = [int(remove_pos)] if isinstance(remove_pos, (int, np.integer)) else remove_pos
        prop_lev = update_leverage_after_batch_removal(
            prev_leverage,
            prev_H,
            prev_weights,
            rem_seq,
            prev_cholmod_factor,
            keep_shape=keep_shape,
            factor_is_post_removal=False,
            u_col_cache=u_col_cache,
        )
    elif isinstance(remove_pos, (int, np.integer)):
        prop_lev = update_leverage_after_row_removal(
            prev_leverage,
            prev_H,
            prev_weights,
            int(remove_pos),
            cur_cholmod_factor,
            keep_shape=keep_shape,
            factor_is_post_removal=True,
        )
    else:
        prop_lev = update_leverage_after_batch_removal(
            prev_leverage,
            prev_H,
            prev_weights,
            remove_pos,
            cur_cholmod_factor,
            keep_shape=keep_shape,
            factor_is_post_removal=True,
        )
    if prop_lev is None or prop_lev.size != res.size:
        lb_hit = _try_lower_bound_screen()
        if lb_hit is not None:
            return lb_hit
        return "need_full", None, None, None
    w_arr = np.asarray(cur_weights, dtype=np.float64)
    approx_norm, _ = compute_normalized_residuals_for_bad_data(
        res,
        w_arr,
        prop_lev,
        is_pseudo=cur_is_pseudo,
    )
    if mask_arr is not None:
        approx_norm = approx_norm.copy()
        approx_norm[~mask_arr] = 0.0
    max_approx = float(approx_norm.max()) if approx_norm.size else 0.0
    is_wb_factor = bool(getattr(cur_cholmod_factor, "is_woodbury_solver", False))
    auto_self_cert = (bool(allow_fast_final_verify) or (int(cur_H.shape[1]) >= 16)) and (not is_wb_factor)
    small_global_shift = (float(state_delta_inf) <= 5e-2) or local_only_shift
    # 若存在少量位移 > 5e-2 的局部状态列，强制将其关联的激活非伪量测行纳入精确回代集合（零局部非线性误差）
    moved_rows = np.empty(0, dtype=np.intp)
    if moved_cols.size > 0 and is_sparse_matrix(cur_H):
        H_csc_cur = cur_H if getattr(cur_H, "format", None) == "csc" else cur_H.tocsc()
        c_indptr = H_csc_cur.indptr
        c_indices = H_csc_cur.indices
        m_chunks = [c_indices[int(c_indptr[c]) : int(c_indptr[c + 1])] for c in moved_cols.tolist()]
        m_chunks = [ch for ch in m_chunks if ch.size > 0]
        if m_chunks:
            moved_rows = np.unique(m_chunks[0] if len(m_chunks) == 1 else np.concatenate(m_chunks)).astype(np.intp, copy=False)
            if mask_arr is not None:
                moved_rows = moved_rows[mask_arr[moved_rows]]
            if cur_is_pseudo is not None and moved_rows.size > 0:
                pseudo_arr_m = np.asarray(cur_is_pseudo, dtype=bool)
                if pseudo_arr_m.size == int(cur_H.shape[0]):
                    moved_rows = moved_rows[~pseudo_arr_m[moved_rows]]
    eff_max_candidates = (
        max(int(max_candidates), 320)
        if moved_rows.size > 0
        else int(max_candidates)
    )

    if max_approx <= 0.5 * float(threshold):
        if auto_self_cert and approx_norm.size > 0 and (allow_fast_final_verify or small_global_shift):
            active_idx = np.flatnonzero(mask_arr) if mask_arr is not None else np.arange(approx_norm.size, dtype=np.intp)
            if active_idx.size > 0:
                k_ver = min(int(final_verify_top), int(active_idx.size))
                sub_norm = approx_norm[active_idx]
                top_local = np.argpartition(sub_norm, -k_ver)[-k_ver:]
                ver_rows = active_idx[top_local]
                if moved_rows.size > 0:
                    ver_rows = np.union1d(ver_rows, moved_rows)
                if ver_rows.size <= eff_max_candidates:
                    exact_ver_lev = exact_leverage_for_candidate_rows(cur_H, ver_rows, cur_cholmod_factor)
                    if exact_ver_lev is not None and exact_ver_lev.size == ver_rows.size:
                        if moved_rows.size > 0:
                            unmoved_mask = ~np.isin(ver_rows, moved_rows)
                            ver_rel_err = (
                                float(
                                    np.max(
                                        np.abs(exact_ver_lev[unmoved_mask] - prop_lev[ver_rows[unmoved_mask]])
                                        * np.maximum(w_arr[ver_rows[unmoved_mask]], 0.0)
                                    )
                                )
                                if np.any(unmoved_mask)
                                else 0.0
                            )
                        else:
                            ver_rel_err = float(
                                np.max(np.abs(exact_ver_lev - prop_lev[ver_rows]) * np.maximum(w_arr[ver_rows], 0.0))
                            )
                        prop_lev[ver_rows] = exact_ver_lev
                        ver_pseudo = None if cur_is_pseudo is None else np.asarray(cur_is_pseudo, dtype=bool)[ver_rows]
                        exact_ver_norm, _ = compute_normalized_residuals_for_bad_data(
                            res[ver_rows],
                            w_arr[ver_rows],
                            exact_ver_lev,
                            is_pseudo=ver_pseudo,
                        )
                        approx_norm[ver_rows] = exact_ver_norm
                        if float(exact_ver_norm.max()) <= 0.8 * float(threshold) and (
                            allow_fast_final_verify or ver_rel_err <= 5e-3
                        ):
                            return "verified_clean", prop_lev, np.empty(0, dtype=np.intp), approx_norm
        # 估算已无坏数据，强制走终轮完整精确认证以杜绝漏检并产出全量精确归一化残差
        return "need_full", prop_lev, None, None
    # 带噪大电网中仅筛选真正逼近阈值的候选 (>= 0.85 * threshold) 与 Top-K 候选（含局部动点关联行），避免 2sigma 正常噪声触发无意义回退
    fast_mid_wb = is_wb_factor and max_approx > 1.25 * float(threshold)
    cand_floor = (
        max(0.95 * float(threshold), 0.75 * max_approx)
        if fast_mid_wb
        else max(0.85 * float(threshold), 0.7 * max_approx)
    )
    cand_rows = np.flatnonzero(approx_norm >= cand_floor)
    eff_top_extra = min(int(top_extra), 2) if fast_mid_wb else int(top_extra)
    if eff_top_extra > 0 and approx_norm.size > 0:
        if mask_arr is not None:
            active_idx = np.flatnonzero(mask_arr)
            if active_idx.size > 0:
                k_top = min(eff_top_extra, int(active_idx.size))
                top_rows = active_idx[np.argpartition(approx_norm[active_idx], -k_top)[-k_top:]]
                cand_rows = np.union1d(cand_rows, top_rows)
        else:
            k_top = min(eff_top_extra, int(approx_norm.size))
            top_rows = np.argpartition(approx_norm, -k_top)[-k_top:]
            cand_rows = np.union1d(cand_rows, top_rows)
    if moved_rows.size > 0:
        cand_rows = np.union1d(cand_rows, moved_rows)
    if cand_rows.size > eff_max_candidates:
        lb_hit = _try_lower_bound_screen()
        if lb_hit is not None:
            return lb_hit
        return "need_full", prop_lev, None, None
    exact_cand_lev = exact_leverage_for_candidate_rows(cur_H, cand_rows, cur_cholmod_factor)
    if exact_cand_lev is None or exact_cand_lev.size != cand_rows.size:
        return "need_full", prop_lev, None, None
    if moved_rows.size > 0:
        unmoved_mask = ~np.isin(cand_rows, moved_rows)
        prop_rel_err = (
            float(
                np.max(
                    np.abs(exact_cand_lev[unmoved_mask] - prop_lev[cand_rows[unmoved_mask]])
                    * np.maximum(w_arr[cand_rows[unmoved_mask]], 0.0)
                )
            )
            if np.any(unmoved_mask)
            else 0.0
        )
    else:
        prop_rel_err = float(
            np.max(np.abs(exact_cand_lev - prop_lev[cand_rows]) * np.maximum(w_arr[cand_rows], 0.0))
        )
    prop_lev[cand_rows] = exact_cand_lev
    cand_pseudo = None if cur_is_pseudo is None else np.asarray(cur_is_pseudo, dtype=bool)[cand_rows]
    exact_cand_norm, _ = compute_normalized_residuals_for_bad_data(
        res[cand_rows],
        w_arr[cand_rows],
        exact_cand_lev,
        is_pseudo=cand_pseudo,
    )
    hit_local = np.flatnonzero(exact_cand_norm > float(threshold))
    if hit_local.size == 0:
        if (
            auto_self_cert
            and small_global_shift
            and prop_rel_err <= 5e-3
            and float(exact_cand_norm.max()) <= float(threshold)
        ) or (
            (not is_wb_factor)
            and allow_fast_final_verify
            and float(exact_cand_norm.max()) <= 0.8 * float(threshold)
        ):
            approx_norm[cand_rows] = exact_cand_norm
            return "verified_clean", prop_lev, np.empty(0, dtype=np.intp), approx_norm
        # 精确验证候选后均未超阈但传播误差/位移偏大或仍处于延迟 Woodbury 因子状态，安全回退精确因子认证
        return "need_full", prop_lev, None, None
    hit_rows = cand_rows[hit_local]
    hit_norms = exact_cand_norm[hit_local]
    order = np.argsort(hit_norms)[::-1]
    return "candidates", prop_lev, hit_rows[order], hit_norms[order]


def screen_initial_bad_data_fast(
    cur_residual: np.ndarray,
    cur_H,
    cur_weights: np.ndarray,
    cur_is_pseudo: Optional[np.ndarray],
    cur_cholmod_factor,
    threshold: float,
    *,
    max_candidates: int = 384,
    min_truncated_norm: float = 5.0,
    active_mask: Optional[np.ndarray] = None,
    min_lower_bound: Optional[float] = None,
    skip_upper_cert: bool = False,
    allow_direct_strong_peaks: Optional[bool] = None,
) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    """首轮及中间轮大偏差坏数据精准筛查（基于免费严格下界 + CHOLMOD 分块多右端精确回代）。

    数学依据：对任意权重 w_i > 0 的量测，残差方差满足 Omega_ii = 1/w_i - h_ii <= 1/w_i，
    故精确归一化残差恒满足严格下界 lb_i = |r_i| * sqrt(w_i) <= r_{N,i}。
    仅对触发下界 ``lb_i > lb_gate`` 的极少数候选行做 CHOLMOD 分块多右端精确回代求杠杆率，
    免除存在显性坏数据时的全矩阵选元求逆；若无下界命中则直接返回 ``"need_full"`` 交由全量精确认证，
    保障零精度损失与零漏检。

    返回 ``(status, hit_rows, hit_norms)``：
    - ``"fast_clean"``：满足严格残差上界，零成本确认全网无坏数据；
    - ``"candidates"``：已算出超过阈值的精确坏数据行号与精确归一化残差（按降序排列）；
    - ``"need_full"``：无下界候选或因子不可用，回退到完整 ``identify_bad_data``。
    """
    res = np.asarray(cur_residual, dtype=np.float64)
    w_arr = np.asarray(cur_weights, dtype=np.float64)
    thr = float(threshold)
    if res.size == 0 or thr <= 0.0:
        return "need_full", None, None
    mask_arr = None if active_mask is None else np.asarray(active_mask, dtype=bool)
    if not skip_upper_cert:
        # 免费全局上界 + 稀疏精确上界认证短路
        cert_status, cert_rows, cert_norms = _strict_upper_bound_bad_data_cert(
            res,
            cur_H,
            w_arr,
            cur_is_pseudo,
            cur_cholmod_factor,
            thr,
            active_mask=mask_arr,
        )
        if cert_status == "fast_clean" or cert_status == "verified_clean":
            return "fast_clean", cert_rows, cert_norms
        if cert_status == "candidates":
            return "candidates", cert_rows, cert_norms
    if cur_H is None or cur_cholmod_factor is None or not is_sparse_matrix(cur_H):
        return "need_full", None, None
    if int(cur_H.shape[0]) != int(res.size) or int(w_arr.size) != int(res.size):
        return "need_full", None, None

    active_w = np.maximum(w_arr, 0.0)
    lb_gate = thr if min_lower_bound is None else max(thr, float(min_lower_bound))
    lower_bound = np.abs(res) * np.sqrt(active_w)
    cand_mask = (active_w > 0.0) & (lower_bound > lb_gate)
    if cur_is_pseudo is not None:
        cand_mask &= ~np.asarray(cur_is_pseudo, dtype=bool)
    if mask_arr is not None:
        cand_mask &= mask_arr
    cand_rows = np.flatnonzero(cand_mask)
    if cand_rows.size == 0:
        return "need_full", None, None

    truncated = False
    n_state_cur = int(cur_H.shape[1])
    scale_prof = SEScaleProfile.for_problem(n_state_cur, int(res.size), int(cur_H.nnz))
    is_pardiso_fac = bool(getattr(cur_cholmod_factor, "is_mkl_pardiso_factor", False))
    # 多线程 MKL PARDISO 半边前代快 2.3~4.1x，单轮回代上限与全网选元求逆切换点相应放宽至 768 / 1536
    eff_verify_cap = 768 if (is_pardiso_fac and n_state_cur > 5000) else scale_prof.screen_verify_cap
    eff_selinv_switch_k = 1536 if (is_pardiso_fac and n_state_cur > 5000) else scale_prof.screen_selinv_switch_k
    # 回代预算严格控制在全网选元求逆交叉点以下；若外部显式传入更小上限则尊重外部上限
    req_cap = max(int(max_candidates), 1)
    cap = min(req_cap, eff_verify_cap) if req_cap >= 128 else req_cap
    direct_enabled = (
        scale_prof.direct_strong_peaks_enabled
        if allow_direct_strong_peaks is None
        else bool(allow_direct_strong_peaks)
    )
    primary_cap = max(cap, scale_prof.screen_primary_cap) if direct_enabled else cap
    direct_lb_gate = max(2.0 * thr, 8.0, float(min_truncated_norm))
    direct_rows_list: List[int] = []
    direct_norms_list: List[float] = []
    sub_lb = lower_bound[cand_rows]
    has_strong_lb = direct_enabled and bool(np.any(sub_lb >= direct_lb_gate))

    if cand_rows.size > 64 or has_strong_lb:
        # 按下界降序做 1 跳邻域局部峰值优先采样：
        # 1) 每选入一个局部最高下界峰值后，将其关联列及 1 跳直连状态列标记为已覆盖，
        #    使强坏数据周围的邻接支路与 1 跳邻居节点电压等连带陪跑量测自动让位；
        # 2) 对非伪量测恒有严格下界 lb_i = |r_i|*sqrt(w_i) <= r_{N,i}：
        #    若局部峰值满足 lb_i >= direct_lb_gate 且 1 跳邻域内无势均力敌竞争候选
        #    (lb_j > 0.6 * lb_i)，直接免 CHOLMOD 回代确认并记保守严格下界 lb_i；
        # 3) 仅将有 1 跳竞争对手（连同其最强竞争对手）或中等下界的极少数候选送入精确回代。
        H_csr = cur_H if getattr(cur_H, "format", None) == "csr" else cur_H.tocsr()
        H_csc = _get_cached_csc_for_graph(H_csr)
        h_indptr = H_csr.indptr
        h_indices = H_csr.indices
        c_indptr = H_csc.indptr
        c_indices = H_csc.indices

        order_desc = np.argsort(sub_lb)[::-1]
        ordered_rows = cand_rows[order_desc]
        ordered_lbs = sub_lb[order_desc]

        # 定长数组按降序单遍填充列级 Top-2 下界倒排表（消除数十万次 Python dict 哈希分配）：
        # 因 ordered_rows 严格降序，每列第 1 次命中必为 top1，第 2 次命中必为 top2。
        col_top1_row = None
        col_top1_lb = None
        col_top2_row = None
        col_top2_lb = None
        if has_strong_lb:
            col_top1_row = np.full(n_state_cur, -1, dtype=np.intp)
            col_top1_lb = np.zeros(n_state_cur, dtype=np.float64)
            col_top2_row = np.full(n_state_cur, -1, dtype=np.intp)
            col_top2_lb = np.zeros(n_state_cur, dtype=np.float64)
            # 向量化抽取候选子矩阵的全部 (row_pos_in_ordered, col) 非零对，仅对每列前两次出现赋值
            H_ord = H_csr[ordered_rows]
            nnz_ord = np.diff(H_ord.indptr)
            if H_ord.nnz > 0:
                entry_ord_idx = np.repeat(np.arange(ordered_rows.size, dtype=np.intp), nnz_ord)
                entry_cols = H_ord.indices
                # 按列稳定排序，同一列内保持 ordered_rows 的降序顺序（第 0、1 项即 top1、top2）
                sort_c = np.argsort(entry_cols, kind="mergesort")
                sc_cols = entry_cols[sort_c]
                sc_ord = entry_ord_idx[sort_c]
                first_mask = np.empty(sc_cols.size, dtype=bool)
                first_mask[0] = True
                if sc_cols.size > 1:
                    first_mask[1:] = sc_cols[1:] != sc_cols[:-1]
                f_pos = np.flatnonzero(first_mask)
                f_cols = sc_cols[f_pos]
                f_ord = sc_ord[f_pos]
                col_top1_row[f_cols] = ordered_rows[f_ord]
                col_top1_lb[f_cols] = ordered_lbs[f_ord]
                if sc_cols.size > 1:
                    sec_pos = f_pos + 1
                    has_sec = (sec_pos < sc_cols.size) & (sc_cols[np.minimum(sec_pos, sc_cols.size - 1)] == f_cols)
                    if np.any(has_sec):
                        s_pos = sec_pos[has_sec]
                        s_cols = f_cols[has_sec]
                        s_ord = sc_ord[s_pos]
                        col_top2_row[s_cols] = ordered_rows[s_ord]
                        col_top2_lb[s_cols] = ordered_lbs[s_ord]

        col_taken = np.zeros(n_state_cur, dtype=bool)
        verify_strong_rival: List[int] = []
        verify_strong_pairs: List[Tuple[int, int]] = []
        verify_medium: List[int] = []
        rival_rows: List[int] = []
        rival_seen: set = set()
        secondary: List[int] = []
        total_primary = 0
        uncapped_verify_count = 0
        for r_idx in ordered_rows.tolist():
            rs = int(h_indptr[r_idx])
            re = int(h_indptr[r_idx + 1])
            cols = h_indices[rs:re]
            if cols.size == 0 or not np.any(col_taken[cols]):
                total_primary += 1
                nb_cols_arr = cols
                if cols.size:
                    col_taken[cols] = True
                    # 扩展 1 跳直连状态列：与 cols 相连的量测行所触碰的状态列
                    cs = c_indptr[cols]
                    ce = c_indptr[cols + 1]
                    if cols.size == 1:
                        nb_rows = c_indices[int(cs[0]) : int(ce[0])]
                    else:
                        chunks = [c_indices[s:e] for s, e in zip(cs.tolist(), ce.tolist()) if e > s]
                        nb_rows = (
                            chunks[0]
                            if len(chunks) == 1
                            else (np.concatenate(chunks) if chunks else np.empty(0, dtype=c_indices.dtype))
                        )
                    if nb_rows.size:
                        if nb_rows.size > 1:
                            nb_rows = np.unique(nb_rows)
                        rs_nb = h_indptr[nb_rows]
                        re_nb = h_indptr[nb_rows + 1]
                        nb_chunks = [h_indices[s:e] for s, e in zip(rs_nb.tolist(), re_nb.tolist()) if e > s]
                        if nb_chunks:
                            nb_cols_arr = nb_chunks[0] if len(nb_chunks) == 1 else np.concatenate(nb_chunks)
                            col_taken[nb_cols_arr] = True

                lb_i = float(lower_bound[r_idx])
                if direct_enabled and lb_i >= direct_lb_gate and cols.size > 0 and col_top1_row is not None:
                    rival_gate = 0.6 * lb_i
                    best_rival_lb = 0.0
                    best_rival_row = -1
                    check_cols = np.unique(nb_cols_arr) if nb_cols_arr.size > 1 else nb_cols_arr
                    t1_rows = col_top1_row[check_cols]
                    has_t1 = t1_rows >= 0
                    if np.any(has_t1):
                        c_sub = check_cols[has_t1]
                        t1_sub = t1_rows[has_t1]
                        is_self = t1_sub == r_idx
                        r_lbs = np.where(is_self, col_top2_lb[c_sub], col_top1_lb[c_sub])
                        r_rows = np.where(is_self, col_top2_row[c_sub], t1_sub)
                        valid_r = r_rows >= 0
                        if np.any(valid_r):
                            vr_lbs = r_lbs[valid_r]
                            max_k = int(np.argmax(vr_lbs))
                            best_rival_lb = float(vr_lbs[max_k])
                            best_rival_row = int(r_rows[valid_r][max_k])
                    if best_rival_lb <= rival_gate:
                        direct_rows_list.append(r_idx)
                        direct_norms_list.append(lb_i)
                    else:
                        uncapped_verify_count += 2 if best_rival_row >= 0 else 1
                        if len(verify_strong_rival) < cap:
                            verify_strong_rival.append(r_idx)
                            verify_strong_pairs.append((r_idx, int(best_rival_row)))
                        if best_rival_row >= 0 and best_rival_row not in rival_seen:
                            rival_seen.add(best_rival_row)
                            rival_rows.append(best_rival_row)
                else:
                    uncapped_verify_count += 1
                    if len(verify_medium) < cap:
                        verify_medium.append(r_idx)

                if total_primary >= primary_cap:
                    break
            elif len(secondary) < cap:
                secondary.append(r_idx)

        # 大网且无大量免回代强峰时，仅当待精确回代的局部峰值候选达到全网选元求逆交叉点，
        # 且当前环境具备 CHOLMOD 选元求逆能力时，才返回 "need_full" 走 1 次全网超节点选元求逆。
        rich_gate = 64 if n_state_cur > 20000 else 128
        can_sparse_selinv = (not is_pardiso_fac) or (CHOLMOD_ANALYZE is not None or CHOLMOD_CHOLESKY is not None)
        if (
            n_state_cur > 5000
            and n_state_cur <= scale_prof.sparse_leverage_max_state
            and can_sparse_selinv
            and len(direct_rows_list) < rich_gate
            and uncapped_verify_count >= eff_selinv_switch_k
            and cap >= 256
            and not getattr(cur_cholmod_factor, "is_woodbury_solver", False)
        ):
            return "need_full", None, None

        # 合并待精确回代的行：统一按降序成对纳入 (峰值, 最强竞争对手) 杜绝单边比较截断；
        # 若本轮已通过免回代锁定足够多明确强主导坏数据，按规模收紧附加回代预算至 screen_verify_cap_rich
        # （超大网仅需 1 个缓存分块 = 16 条），避免耗费数秒对受污染次要候选执行冗余 CHOLMOD 回代。
        verify_cap = min(cap, scale_prof.screen_verify_cap_rich) if len(direct_rows_list) >= rich_gate else cap
        verify_list: List[int] = []
        verify_set: set = set()
        direct_set = set(direct_rows_list)
        for p_row, r_row in verify_strong_pairs:
            need_slots = (1 if (p_row not in verify_set and p_row not in direct_set) else 0) + (
                1 if (r_row >= 0 and r_row not in verify_set and r_row not in direct_set and r_row != p_row) else 0
            )
            if len(verify_list) + need_slots > verify_cap:
                break
            if p_row not in verify_set and p_row not in direct_set:
                verify_list.append(p_row)
                verify_set.add(p_row)
            if r_row >= 0 and r_row not in verify_set and r_row not in direct_set:
                verify_list.append(r_row)
                verify_set.add(r_row)
        for r_idx in verify_medium:
            if len(verify_list) >= verify_cap:
                break
            if r_idx not in verify_set and r_idx not in direct_set:
                verify_list.append(r_idx)
                verify_set.add(r_idx)
        # 若已经有大量强主导峰值被免回代锁定，则无需额外向次级陪跑行追加 CHOLMOD 回代预算
        if not direct_rows_list:
            extra_cap = 16 if len(verify_list) >= 64 else 64
            extra_budget = min(len(secondary), max(0, min(cap - len(verify_list), extra_cap)))
            if extra_budget > 0:
                for r_idx in secondary:
                    if extra_budget <= 0:
                        break
                    if r_idx not in verify_set and r_idx not in direct_set:
                        verify_list.append(r_idx)
                        verify_set.add(r_idx)
                        extra_budget -= 1
        elif len(verify_list) == 0 and len(secondary) > 0 and len(direct_rows_list) < 4:
            # 极少数强峰值场景，适当核验最多 16 条次级候选以防掩蔽
            for r_idx in secondary[:16]:
                if r_idx not in direct_set:
                    verify_list.append(r_idx)
                    verify_set.add(r_idx)

        if (len(direct_rows_list) + len(verify_list)) < cand_rows.size:
            truncated = True
        verify_rows = np.sort(np.asarray(verify_list, dtype=np.intp))
    else:
        verify_rows = cand_rows

    hit_row_parts: List[np.ndarray] = []
    hit_norm_parts: List[np.ndarray] = []
    if direct_rows_list:
        hit_row_parts.append(np.asarray(direct_rows_list, dtype=np.intp))
        hit_norm_parts.append(np.asarray(direct_norms_list, dtype=np.float64))

    if verify_rows.size > 0:
        exact_lev = exact_leverage_for_candidate_rows(cur_H, verify_rows, cur_cholmod_factor)
        if exact_lev is None or exact_lev.size != verify_rows.size:
            if not hit_row_parts:
                return "need_full", None, None
        else:
            cand_pseudo = None if cur_is_pseudo is None else np.asarray(cur_is_pseudo, dtype=bool)[verify_rows]
            exact_norm, _ = compute_normalized_residuals_for_bad_data(
                res[verify_rows],
                w_arr[verify_rows],
                exact_lev,
                is_pseudo=cand_pseudo,
            )
            accept_gate = max(thr, float(min_truncated_norm)) if (truncated or min_lower_bound is not None) else thr
            hit_local = np.flatnonzero(exact_norm > accept_gate)
            if hit_local.size > 0:
                hit_row_parts.append(verify_rows[hit_local])
                hit_norm_parts.append(exact_norm[hit_local])

    if not hit_row_parts:
        return "need_full", None, None
    hit_rows = hit_row_parts[0] if len(hit_row_parts) == 1 else np.concatenate(hit_row_parts)
    hit_norms = hit_norm_parts[0] if len(hit_norm_parts) == 1 else np.concatenate(hit_norm_parts)
    order = np.argsort(hit_norms)[::-1]
    return "candidates", hit_rows[order], hit_norms[order]


def _rank_from_singular_values(singular_values: np.ndarray, shape: Tuple[int, int]) -> Tuple[int, float]:
    """Return numerical rank using NumPy's standard SVD tolerance rule."""
    tol = max(shape) * np.finfo(float).eps * (singular_values[0] if singular_values.size else 1.0)
    rank = int(np.sum(singular_values > tol))
    return rank, float(tol)


def observability_rank_details(
    H: np.ndarray,
    state_count: int,
    normal_matrix: np.ndarray = None,
    normal_factor_diag: np.ndarray = None,
    dense_svd_limit: int = 256,
) -> Tuple[int, int, np.ndarray, List[Tuple[int, float]]]:
    """Fast rank analysis for tall state-estimation Jacobians.

    Well-conditioned observable systems are the common path; a Cholesky factor of
    a positive-definite normal matrix proves full column rank without computing a
    full spectral decomposition. `estimate()` passes its already assembled
    H.T @ W @ H here, which has the same rank test outcome when measurement
    weights are positive. For medium/large deficient sparse systems
    (``state_count > dense_svd_limit``), an O(nnz) two-stage sparse diagnostic
    (normal diagonal + maximum bipartite matching) locates blind and
    structurally unmatched states without dense SVD blowup.
    """
    state_count = int(state_count)
    measurement_count = int(H.shape[0])
    min_deficiency = max(0, state_count - measurement_count)
    full_column_rank_possible = measurement_count >= state_count
    gram = H.T @ H if normal_matrix is None else normal_matrix
    weak_states: List[Tuple[int, float]] = []

    if is_sparse_matrix(gram):
        diag_values = gram.diagonal()
    else:
        diag_values = np.diag(gram) if gram.size else np.array([], dtype=np.float64)
    scale = float(np.sqrt(np.max(diag_values))) if diag_values.size else 1.0
    tol = max(H.shape) * np.finfo(float).eps * max(scale, 1.0)
    tol_sq = max(tol * tol, 1e-24)
    has_zero_diag = bool(diag_values.size and np.any(diag_values <= tol_sq))
    if full_column_rank_possible and not has_zero_diag and normal_factor_diag is not None:
        diag = np.asarray(normal_factor_diag, dtype=np.float64)
        if diag.size == state_count and float(np.min(diag)) > tol:
            return state_count, 0, np.array([], dtype=np.float64), weak_states
    elif full_column_rank_possible and not has_zero_diag and is_sparse_matrix(gram) and CHOLMOD_CHOLESKY is not None:
        try:
            CHOLMOD_CHOLESKY(gram.tocsc())
            return state_count, 0, np.array([], dtype=np.float64), weak_states
        except Exception:
            pass
    if full_column_rank_possible and not has_zero_diag and normal_factor_diag is None and is_sparse_matrix(gram):
        gram_csc = gram.tocsc()
        try:
            lu = SP_SPLU(gram_csc, diag_pivot_thresh=0.0, permc_spec="MMD_AT_PLUS_A")
            diag = np.abs(lu.U.diagonal())
            if diag.size == state_count and float(np.min(diag)) > tol:
                return state_count, 0, np.array([], dtype=np.float64), weak_states
        except Exception:
            pass
        try:
            lu = SP_SPLU(gram_csc)
            diag = np.abs(lu.U.diagonal())
            if diag.size == state_count and float(np.min(diag)) > tol:
                return state_count, 0, np.array([], dtype=np.float64), weak_states
        except Exception:
            pass
    elif full_column_rank_possible and not has_zero_diag and normal_factor_diag is None:
        chol, info = DPOTRF(gram, lower=True, clean=False, overwrite_a=False)
        if info == 0:
            diag = np.diag(chol)
            if diag.size == state_count and float(np.min(diag)) > tol:
                return state_count, 0, np.array([], dtype=np.float64), weak_states

    use_sparse_structural = state_count > dense_svd_limit
    zero_cols_pre = np.empty(0, dtype=np.int64)
    unmatched_cols_pre = np.empty(0, dtype=np.int64)
    if not use_sparse_structural and is_sparse_matrix(H) and diag_values.size:
        zero_cols_pre = np.flatnonzero(diag_values <= tol_sq)
        if measurement_count > 0:
            try:
                H_csc_pre = H.tocsc()
                if H_csc_pre.nnz > 0:
                    H_csc_pre.eliminate_zeros()
                if H_csc_pre.nnz > 0:
                    col_match_pre = SP_MAX_BIPARTITE_MATCHING(H_csc_pre, perm_type="row")
                    unmatched_cols_pre = np.flatnonzero(col_match_pre < 0)
            except Exception:
                unmatched_cols_pre = np.empty(0, dtype=np.int64)
        # 小规模稀疏系统若存在明确结构零覆盖或二分图未匹配列，直接一次性返回全部结构弱状态，免去多轮稠密 SVD
        if zero_cols_pre.size > 0 or unmatched_cols_pre.size > 0:
            use_sparse_structural = True

    if use_sparse_structural:
        # 稀疏结构缺秩快速路径：先用对角线定位零覆盖盲区，再用二分图最大匹配 O(nnz) 精准找出所有结构未匹配状态列。
        if is_sparse_matrix(gram):
            diag_values = np.asarray(gram.diagonal(), dtype=np.float64)
        else:
            diag_values = np.asarray(np.diag(gram), dtype=np.float64) if gram.size else np.array([], dtype=np.float64)
        if diag_values.size:
            tol_sq = max(tol * tol, 1e-24)
            if state_count > dense_svd_limit:
                zero_cols = np.flatnonzero(diag_values <= tol_sq)
                unmatched_cols = np.empty(0, dtype=np.int64)
                if is_sparse_matrix(H) and measurement_count > 0:
                    try:
                        H_csc = H.tocsc()
                        if H_csc.nnz > 0:
                            H_csc.eliminate_zeros()
                        if H_csc.nnz > 0:
                            col_match = SP_MAX_BIPARTITE_MATCHING(H_csc, perm_type="row")
                            unmatched_cols = np.flatnonzero(col_match < 0)
                    except Exception:
                        unmatched_cols = np.empty(0, dtype=np.int64)
            else:
                zero_cols = zero_cols_pre
                unmatched_cols = unmatched_cols_pre
            # 向量化保序合并全部结构零覆盖列与二分图未匹配列，取消 4096 截断以消除超大规模盲区 O(n^2) 循环修补
            seen_mask = np.zeros(diag_values.size, dtype=bool)
            cand_parts: List[np.ndarray] = []
            if zero_cols.size > 0:
                z_arr = np.asarray(zero_cols, dtype=np.int64)
                seen_mask[z_arr] = True
                cand_parts.append(z_arr)
            if unmatched_cols.size > 0:
                u_arr = np.asarray(unmatched_cols, dtype=np.int64)
                u_new = u_arr[~seen_mask[u_arr]]
                if u_new.size > 0:
                    seen_mask[u_new] = True
                    cand_parts.append(u_new)
            weak_arr = (
                cand_parts[0]
                if len(cand_parts) == 1
                else (np.concatenate(cand_parts) if cand_parts else np.empty(0, dtype=np.int64))
            )
            if weak_arr.size < 10 and state_count > dense_svd_limit:
                extra_top = np.argsort(diag_values)[: min(10 + int(weak_arr.size), diag_values.size)]
                extra_new = extra_top[~seen_mask[extra_top]][: max(0, 10 - int(weak_arr.size))]
                if extra_new.size > 0:
                    weak_arr = np.concatenate((weak_arr, extra_new.astype(np.int64, copy=False)))
            weak_states = list(zip(weak_arr.tolist(), diag_values[weak_arr].tolist())) if weak_arr.size > 0 else []
            deficiency = max(1, int(zero_cols.size), int(unmatched_cols.size), min_deficiency)
        else:
            deficiency = state_count
        return max(0, state_count - deficiency), deficiency, np.array([], dtype=np.float64), weak_states

    if is_sparse_matrix(gram):
        gram = gram.toarray()
    H_for_svd = H.toarray() if is_sparse_matrix(H) else H
    _, singular_values, vh = np.linalg.svd(H_for_svd, full_matrices=False)
    rank, _ = _rank_from_singular_values(singular_values, H.shape)
    deficiency = state_count - rank
    if deficiency > 0 and vh.size:
        null_vectors = vh[-deficiency:, :]
        scores = np.max(np.abs(null_vectors), axis=0)
        top = np.argsort(scores)[::-1][: min(10, scores.size)]
        weak_states = [(int(idx), float(scores[idx])) for idx in top]

    return rank, max(0, deficiency), singular_values, weak_states


def unanchored_voltage_island_state_indices(
    H,
    voltage_cols: Sequence[int],
    anchored_voltage_cols: Sequence[int],
) -> List[int]:
    """O(nnz) 检测量测图中不含任何电压量测锚点的纯功率孤立连通分量，每岛返回 1 个代表电压状态列。

    在直流或混合网络高比例缺测/盲区下，若干节点可能仅通过支路功率量测互连而缺失任何绝对电压量测，
    导致正规矩阵在整体电压标度方向高度病态。对每个 ``size >= 2`` 的无电压锚定连通分量补 1 条
    低权重伪电压量测即可从物理根源消除缩放奇异性。
    """
    v_cols = np.asarray(voltage_cols, dtype=np.int64).ravel()
    n_v = int(v_cols.size)
    if n_v < 2 or H is None or not is_sparse_matrix(H):
        return []
    H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
    if H_csr.nnz == 0:
        return []
    contiguous = bool(np.array_equal(v_cols, np.arange(n_v, dtype=np.int64)))
    if contiguous:
        mask = H_csr.indices < n_v
        if not np.any(mask):
            return []
        row_nnz = np.diff(H_csr.indptr).astype(np.int64, copy=False)
        entry_rows = np.repeat(np.arange(H_csr.shape[0], dtype=np.int64), row_nnz)
        v_rows = entry_rows[mask]
        v_local = H_csr.indices[mask].astype(np.int64, copy=False)
    else:
        n_total = int(H_csr.shape[1])
        col_map = np.full(n_total, -1, dtype=np.int64)
        col_map[v_cols] = np.arange(n_v, dtype=np.int64)
        mapped = col_map[H_csr.indices]
        mask = mapped >= 0
        if not np.any(mask):
            return []
        row_nnz = np.diff(H_csr.indptr).astype(np.int64, copy=False)
        entry_rows = np.repeat(np.arange(H_csr.shape[0], dtype=np.int64), row_nnz)
        v_rows = entry_rows[mask]
        v_local = mapped[mask]

    # 对每行多电压非零元，将首列与同行其余电压列连成星形边，O(nnz) 等价于完整团连通性
    nnz_by_row = np.bincount(v_rows, minlength=H_csr.shape[0])
    starts = np.empty(H_csr.shape[0] + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(nnz_by_row, out=starts[1:])
    pos = np.arange(v_local.size, dtype=np.int64)
    first_pos = starts[v_rows]
    nonfirst = pos > first_pos
    if not np.any(nonfirst):
        return []
    edge_u = v_local[first_pos[nonfirst]]
    edge_v = v_local[nonfirst]
    adj = SP_CSR_MATRIX(
        (np.ones(edge_u.size, dtype=np.int8), (edge_u, edge_v)),
        shape=(n_v, n_v),
    )
    n_comp, labels = SP_CONNECTED_COMPONENTS(adj, directed=False, return_labels=True)
    comp_sizes = np.bincount(labels, minlength=n_comp)
    anchored_mask = np.zeros(n_comp, dtype=bool)
    anch = np.asarray(anchored_voltage_cols, dtype=np.int64).ravel()
    if anch.size:
        if contiguous:
            valid_a = anch[(anch >= 0) & (anch < n_v)]
            if valid_a.size:
                anchored_mask[labels[valid_a]] = True
        else:
            valid_a = col_map[anch[(anch >= 0) & (anch < col_map.size)]]
            valid_a = valid_a[valid_a >= 0]
            if valid_a.size:
                anchored_mask[labels[valid_a]] = True
    bad_comps = np.flatnonzero((comp_sizes >= 2) & (~anchored_mask))
    if bad_comps.size == 0:
        return []
    # 每个无电压锚定孤岛取最小状态列作为确定性代表
    order = np.argsort(labels, kind="mergesort")
    sorted_labels = labels[order]
    first_idx = np.searchsorted(sorted_labels, bad_comps)
    rep_local = order[first_idx]
    return [int(v_cols[i]) for i in rep_local.tolist()]


def unanchored_angle_state_indices(
    H: np.ndarray,
    angle_cols: Sequence[int],
    tol: float = 1e-12,
) -> List[int]:
    """Find AC angle components whose measurement rows only define relative angles.

    Large sparse cases can have a tiny rank deficiency even though every node has
    local P/Q rows. In that situation the weak-state diagonal heuristic often
    points at arbitrary low-diagonal columns. This structural pass looks only at
    angle columns of H: rows with zero coefficient sum tie angles relatively,
    while rows with a nonzero sum provide an absolute anchor through a reference
    angle or a direct angle measurement.
    """
    angle_cols = [int(idx) for idx in angle_cols]
    if not angle_cols:
        return []
    angle_cols_array = np.asarray(angle_cols, dtype=np.int64)

    if is_sparse_matrix(H):
        H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
        n = len(angle_cols)
        if np.array_equal(angle_cols_array, np.arange(n, dtype=np.int64)):
            mask = H_csr.indices < n
            if np.any(mask):
                row_nnz = np.diff(H_csr.indptr).astype(np.int64, copy=False)
                entry_rows = np.repeat(np.arange(H_csr.shape[0], dtype=np.int64), row_nnz)
                angle_rows = entry_rows[mask]
                angle_indices = H_csr.indices[mask].astype(np.int64, copy=False)
                angle_data = H_csr.data[mask]
                nnz_by_row = np.bincount(angle_rows, minlength=H_csr.shape[0])
                active_rows = np.flatnonzero(nnz_by_row).astype(np.int64, copy=False)
                degree = np.bincount(angle_indices, minlength=n)
                filtered_indptr = np.empty(H_csr.shape[0] + 1, dtype=np.int64)
                filtered_indptr[0] = 0
                np.cumsum(nnz_by_row, out=filtered_indptr[1:])
                positions = np.arange(angle_indices.size, dtype=np.int64)
                row_start_for_entry = filtered_indptr[angle_rows]
                nonfirst = positions > row_start_for_entry
                if np.any(nonfirst):
                    left = angle_indices[row_start_for_entry[nonfirst]]
                    right = angle_indices[positions[nonfirst]]
                    graph = SP_COO_MATRIX(
                        (
                            np.ones(left.size * 2, dtype=np.int8),
                            (np.concatenate((left, right)), np.concatenate((right, left))),
                        ),
                        shape=(n, n),
                    ).tocsr()
                else:
                    graph = SP_CSR_MATRIX((n, n), dtype=np.int8)
                n_components, labels = SP_CONNECTED_COMPONENTS(graph, directed=False, return_labels=True)
                anchored = np.zeros(int(n_components), dtype=bool)
                if active_rows.size:
                    starts = filtered_indptr[active_rows]
                    row_sums = np.add.reduceat(angle_data, starts)
                    anchor_rows = np.abs(row_sums) > tol
                    if np.any(anchor_rows):
                        first_cols = angle_indices[starts]
                        anchored[labels[first_cols[anchor_rows]]] = True
            else:
                degree = np.zeros(n, dtype=np.int64)
                n_components = n
                labels = np.arange(n, dtype=np.int32)
                anchored = np.zeros(n, dtype=bool)
            state_indices = []
            for component in np.flatnonzero(~anchored).tolist():
                local_cols = np.flatnonzero(labels == int(component))
                if local_cols.size == 0:
                    continue
                local_degree = degree[local_cols]
                best = local_cols[local_degree == int(np.max(local_degree))]
                representative = int(best[np.argmin(angle_cols_array[best])])
                state_indices.append((int(local_cols.size), int(angle_cols_array[representative])))
            state_indices.sort(key=lambda item: (-item[0], item[1]))
            return [idx for _size, idx in state_indices]

        sub = H[:, angle_cols].tocsr()
        nnz_by_row = np.diff(sub.indptr)
        active_rows = np.flatnonzero(nnz_by_row).astype(np.int64, copy=False)
        degree = np.bincount(sub.indices, minlength=len(angle_cols)) if sub.indices.size else np.zeros(len(angle_cols), dtype=np.int64)
        if sub.indices.size:
            positions = np.arange(sub.indices.size, dtype=np.int64)
            entry_rows = np.repeat(np.arange(sub.shape[0], dtype=np.int64), nnz_by_row)
            row_start_for_entry = sub.indptr[entry_rows]
            nonfirst = positions > row_start_for_entry
            if np.any(nonfirst):
                left = sub.indices[row_start_for_entry[nonfirst]]
                right = sub.indices[positions[nonfirst]]
                graph = SP_COO_MATRIX(
                    (
                        np.ones(left.size * 2, dtype=np.int8),
                        (np.concatenate((left, right)), np.concatenate((right, left))),
                    ),
                    shape=(n, n),
                ).tocsr()
            else:
                graph = SP_CSR_MATRIX((n, n), dtype=np.int8)
            n_components, labels = SP_CONNECTED_COMPONENTS(graph, directed=False, return_labels=True)
            anchored = np.zeros(int(n_components), dtype=bool)
            if active_rows.size:
                starts = sub.indptr[active_rows]
                row_sums = np.add.reduceat(sub.data, starts)
                anchor_rows = np.abs(row_sums) > tol
                if np.any(anchor_rows):
                    first_cols = sub.indices[starts]
                    anchored[labels[first_cols[anchor_rows]]] = True
        else:
            n_components = n
            labels = np.arange(n, dtype=np.int32)
            anchored = np.zeros(n, dtype=bool)
        state_indices = []
        for component in np.flatnonzero(~anchored).tolist():
            local_cols = np.flatnonzero(labels == int(component))
            if local_cols.size == 0:
                continue
            local_degree = degree[local_cols]
            best = local_cols[local_degree == int(np.max(local_degree))]
            representative = int(best[np.argmin(angle_cols_array[best])])
            state_indices.append((int(local_cols.size), int(angle_cols_array[representative])))
        state_indices.sort(key=lambda item: (-item[0], item[1]))
        return [idx for _size, idx in state_indices]
    else:
        matrix = np.asarray(H)
        sub = matrix[:, angle_cols]
        active_rows = None
        degree = np.count_nonzero(np.abs(sub) > tol, axis=0)

    n = len(angle_cols)
    parent = np.arange(n, dtype=np.int32)
    anchored = np.zeros(n, dtype=bool)

    def find(pos: int) -> int:
        while int(parent[pos]) != pos:
            parent[pos] = parent[int(parent[pos])]
            pos = int(parent[pos])
        return int(pos)

    def union(left: int, right: int) -> int:
        root_l = find(left)
        root_r = find(right)
        if root_l != root_r:
            parent[root_r] = root_l
            anchored[root_l] = anchored[root_l] or anchored[root_r]
        return root_l

    for row in range(sub.shape[0]):
        vals = sub[row, :]
        cols = np.flatnonzero(np.abs(vals) > tol)
        if cols.size == 0:
            continue
        root = int(cols[0])
        for col in cols[1:]:
            root = union(root, int(col))
        root = find(root)
        if abs(float(np.sum(vals[cols]))) > tol:
            anchored[root] = True

    components = {}
    for local_col in range(n):
        components.setdefault(find(local_col), []).append(local_col)

    state_indices = []
    for root, local_cols in components.items():
        root = find(root)
        if anchored[root]:
            continue
        representative = max(local_cols, key=lambda col: (int(degree[col]), -int(angle_cols[col])))
        state_indices.append((len(local_cols), int(angle_cols[representative])))
    state_indices.sort(key=lambda item: (-item[0], item[1]))
    return [idx for _, idx in state_indices]


class NormalEquationSolver:
    """Solve repeated normal equations, reusing CHOLMOD symbolic analysis when available."""

    def __init__(self, assume_fixed_pattern: bool = False):
        self._cholmod_factor = None
        self._cholmod_pattern = None
        self._cholmod_disabled = False
        self.assume_fixed_pattern = bool(assume_fixed_pattern)

    @staticmethod
    def _sparse_pattern(matrix) -> Tuple[Tuple[int, int], int, np.ndarray, np.ndarray]:
        return (
            matrix.shape,
            int(matrix.nnz),
            matrix.indptr.astype(np.int64, copy=True),
            matrix.indices.astype(np.int64, copy=True),
        )

    @staticmethod
    def _same_sparse_pattern(pattern, matrix) -> bool:
        if pattern is None:
            return False
        shape, nnz, indptr, indices = pattern
        return (
            shape == matrix.shape
            and nnz == int(matrix.nnz)
            and indptr.shape == matrix.indptr.shape
            and indices.shape == matrix.indices.shape
            and np.array_equal(indptr, matrix.indptr)
            and np.array_equal(indices, matrix.indices)
        )

    def _solve_sparse_cholmod(self, gain_csc, rhs: np.ndarray):
        if CHOLMOD_ANALYZE is not None:
            if self._cholmod_factor is None or (
                not self.assume_fixed_pattern and not self._same_sparse_pattern(self._cholmod_pattern, gain_csc)
            ):
                self._cholmod_factor = CHOLMOD_ANALYZE(gain_csc)
                if not self.assume_fixed_pattern:
                    self._cholmod_pattern = self._sparse_pattern(gain_csc)
            self._cholmod_factor.cholesky_inplace(gain_csc)
            return self._cholmod_factor(rhs), None
        if CHOLMOD_CHOLESKY is not None:
            factor = CHOLMOD_CHOLESKY(gain_csc)
            return factor(rhs), None
        raise RuntimeError("CHOLMOD is not available")

    def solve(
        self,
        gain: np.ndarray,
        rhs: np.ndarray,
        return_factor_diag: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve normal equations, preferring sparse SPD Cholesky when available."""
        if is_sparse_matrix(gain):
            gain_csc = gain if getattr(gain, "format", None) == "csc" else gain.tocsc()
            if (
                get_linear_solver_preference() != "scipy"
                and not self._cholmod_disabled
                and (CHOLMOD_ANALYZE is not None or CHOLMOD_CHOLESKY is not None)
            ):
                try:
                    return self._solve_sparse_cholmod(gain_csc, rhs)
                except Exception:
                    self._cholmod_factor = None
                    self._cholmod_pattern = None
                    self._cholmod_disabled = True
            return _solve_normal_equations_no_cholmod(gain_csc, rhs, return_factor_diag)
        return _solve_dense_normal_equations(gain, rhs, return_factor_diag)


def _solve_normal_equations_no_cholmod(
    gain_csc,
    rhs: np.ndarray,
    return_factor_diag: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        lu = SP_SPLU(gain_csc, diag_pivot_thresh=0.0, permc_spec="MMD_AT_PLUS_A")
        factor_diag = np.abs(lu.U.diagonal()) if return_factor_diag else None
        return lu.solve(rhs), factor_diag
    except Exception:
        pass
    try:
        lu = SP_SPLU(gain_csc)
        factor_diag = np.abs(lu.U.diagonal()) if return_factor_diag else None
        return lu.solve(rhs), factor_diag
    except Exception:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SP_MATRIX_RANK_WARNING)
            dx = SP_SPSOLVE(gain_csc, rhs)
        return dx, None
    except Exception:
        pass
    return _solve_dense_normal_equations(gain_csc.toarray(), rhs, return_factor_diag)


def _solve_dense_normal_equations(
    gain: np.ndarray,
    rhs: np.ndarray,
    return_factor_diag: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        factor, dx, info = DPOSV(gain, rhs.copy(), lower=True, overwrite_a=False, overwrite_b=True)
        if info == 0:
            factor_diag = np.diag(factor).copy() if return_factor_diag else None
            return dx, factor_diag
    except Exception:
        pass
    try:
        factor = CHO_FACTOR(gain, lower=True, check_finite=False)
        dx = CHO_SOLVE(factor, rhs, check_finite=False)
        factor_diag = np.diag(factor[0]).copy() if return_factor_diag else None
        return dx, factor_diag
    except Exception:
        pass
    try:
        return np.linalg.solve(gain, rhs), None
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gain) @ rhs, None


def solve_normal_equations_with_factor(
    gain: np.ndarray,
    rhs: np.ndarray,
    return_factor_diag: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve normal equations and return the Cholesky diagonal when available."""
    return NormalEquationSolver().solve(gain, rhs, return_factor_diag=return_factor_diag)


def build_normal_equations(
    H: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    dense_gain_limit: int = 1000,
    uniform_weight: Optional[float] = None,
    weights_are_uniform: Optional[bool] = None,
    weighted_residual: Optional[np.ndarray] = None,
    normal_pattern=None,
    assume_normal_pattern_matches: bool = False,
    normal_assembly_plan: Optional[NormalEquationAssemblyPlan] = None,
    triangular: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build WLS normal equations while avoiding WH allocation for uniform weights."""
    triangular_mode = None if triangular is None else str(triangular).lower()
    if triangular_mode not in (None, "lower"):
        raise ValueError(f"Unsupported normal-equation triangular mode: {triangular!r}")
    if is_sparse_matrix(H):
        if (
            normal_assembly_plan is not None
            and normal_assembly_plan.use_direct_assembly
            and normal_assembly_plan.matches(H)
        ):
            return normal_assembly_plan.assemble(
                H,
                residual,
                weight,
                uniform_weight=uniform_weight,
                weights_are_uniform=weights_are_uniform,
                weighted_residual=weighted_residual,
                dense_gain_limit=dense_gain_limit,
            )
        if weight.size == 0:
            gain = H.T @ H
            rhs = H.T @ residual
        elif weights_are_uniform is True or uniform_weight is not None:
            if uniform_weight is None:
                uniform_weight = float(weight[0])
            gain = H.T @ H
            rhs = H.T @ residual
            if uniform_weight != 1.0:
                gain = uniform_weight * gain
                rhs = uniform_weight * rhs
        elif weights_are_uniform is False:
            weighted_residual = weight * residual if weighted_residual is None else weighted_residual
            weighted_H = _row_weighted_sparse_matrix(H, weight)
            gain = H.T @ weighted_H
            rhs = H.T @ weighted_residual
        else:
            first_weight = float(weight[0])
            if np.all(weight == first_weight):
                gain = H.T @ H
                rhs = H.T @ residual
                if first_weight != 1.0:
                    gain = first_weight * gain
                    rhs = first_weight * rhs
            else:
                weighted_residual = weight * residual if weighted_residual is None else weighted_residual
                weighted_H = _row_weighted_sparse_matrix(H, weight)
                gain = H.T @ weighted_H
                rhs = H.T @ weighted_residual
        if is_sparse_matrix(gain):
            if triangular_mode == "lower":
                gain = SP_TRIL(gain, format="csc")
                if normal_pattern is not None:
                    normal_pattern = SP_TRIL(normal_pattern, format="csc")
            if assume_normal_pattern_matches:
                gain = gain.tocsc() if getattr(gain, "format", None) != "csc" else gain
            else:
                if normal_pattern is None:
                    normal_pattern = _normal_equation_structural_pattern(H, triangular=triangular_mode)
                gain = _expand_sparse_matrix_to_pattern(gain, normal_pattern)
            gain = gain.toarray() if gain.shape[0] <= dense_gain_limit else gain.tocsc()
        return gain, np.asarray(rhs, dtype=np.float64).ravel()

    if weight.size == 0:
        gain = H.T @ H
        rhs = H.T @ residual
        if triangular_mode == "lower":
            gain = np.tril(gain)
        return gain, rhs

    if weights_are_uniform is True or uniform_weight is not None:
        if uniform_weight is None:
            uniform_weight = float(weight[0])
        gain = H.T @ H
        rhs = H.T @ residual
        if uniform_weight != 1.0:
            gain = uniform_weight * gain
            rhs = uniform_weight * rhs
        if triangular_mode == "lower":
            gain = np.tril(gain)
        return gain, rhs

    if weights_are_uniform is False:
        weighted_residual = weight * residual if weighted_residual is None else weighted_residual
        WH = weight[:, None] * H
        gain = H.T @ WH
        rhs = H.T @ weighted_residual
        if triangular_mode == "lower":
            gain = np.tril(gain)
        return gain, rhs

    first_weight = float(weight[0])
    if np.all(weight == first_weight):
        gain = H.T @ H
        rhs = H.T @ residual
        if first_weight != 1.0:
            gain = first_weight * gain
            rhs = first_weight * rhs
        if triangular_mode == "lower":
            gain = np.tril(gain)
        return gain, rhs

    WH = weight[:, None] * H
    gain = H.T @ WH
    weighted_residual = weight * residual if weighted_residual is None else weighted_residual
    rhs = H.T @ weighted_residual
    if triangular_mode == "lower":
        gain = np.tril(gain)
    return gain, rhs


def select_island_fixed_voltage_reference_buses(
    sample_pos: np.ndarray,
    sample_val: np.ndarray,
    component_by_pos: np.ndarray,
    n_components: int,
    node_island_pos: np.ndarray,
    island_alive_mask: Optional[np.ndarray] = None,
    node_degree: Optional[np.ndarray] = None,
    node_ids: Optional[np.ndarray] = None,
    *,
    voltage_min: float = 0.1,
    voltage_max: float = 1.9,
    healthy_min: float = 0.8,
    healthy_max: float = 1.2,
    outlier_tol: float = 0.15,
) -> Dict[int, float]:
    """按电气拓扑岛选取合理有效电压量测数最多的拓扑母线，并计算其稳健均值作为定电压参考。

    参数说明：
    - ``sample_pos`` / ``sample_val``: 全部真实有效电压量测（含母线节点 ``V`` 及关联支路/变压器/开关/
      断路器/发负荷/变流器等端口电压量测）对应的求解节点位置 ``solver_pos`` 与标幺电压值。
    - ``component_by_pos``: 节点 ``solver_pos`` 到零阻抗拓扑母线编号（``0 .. n_components-1``）的映射。
    - ``node_island_pos`` / ``island_alive_mask``: 节点所属电气岛编号与活动岛掩码。
    - 合理性筛选优先采用 ``[max(voltage_min, healthy_min), min(voltage_max, healthy_max)]``（默认 ``[0.8, 1.2]``）；
      若某电气岛无落入健康区间的样本，则自动回退到 ``[voltage_min, voltage_max]``。
    - 当选定母线的合理有效电压量测数 ``>= 3`` 时，先剔除与中位数偏差超过 ``outlier_tol`` 的离群坏数据后再取算术平均，
      既拉平随机偏差，又免疫坏数据污染定电压基准。
    """
    n_comp = int(n_components)
    comp_by_pos = np.asarray(component_by_pos, dtype=np.int64)
    n_nodes = int(comp_by_pos.size)
    if n_comp <= 0 or n_nodes == 0:
        return {}
    pos_arr = np.asarray(sample_pos, dtype=np.int64)
    val_arr = np.asarray(sample_val, dtype=np.float64)
    if pos_arr.size == 0 or val_arr.size != pos_arr.size:
        return {}

    isl_by_pos = np.asarray(node_island_pos, dtype=np.int64)
    if isl_by_pos.size != n_nodes:
        return {}
    v_lo = float(voltage_min)
    v_hi = float(voltage_max)
    if v_lo > v_hi:
        v_lo, v_hi = v_hi, v_lo

    # 1. 基础有效性与物理上下界过滤
    valid = (
        (pos_arr >= 0)
        & (pos_arr < n_nodes)
        & np.isfinite(val_arr)
        & (val_arr >= v_lo)
        & (val_arr <= v_hi)
    )
    if not np.any(valid):
        return {}
    pos_v = pos_arr[valid]
    val_v = val_arr[valid]
    isl_v = isl_by_pos[pos_v]
    isl_valid = isl_v >= 0
    if island_alive_mask is not None:
        alive_arr = np.asarray(island_alive_mask, dtype=bool)
        isl_valid &= (isl_v < alive_arr.size)
        if np.any(isl_valid):
            isl_valid[isl_valid] &= alive_arr[isl_v[isl_valid]]
    if not np.any(isl_valid):
        return {}
    pos_v = pos_v[isl_valid]
    val_v = val_v[isl_valid]
    isl_v = isl_v[isl_valid]
    comp_v = comp_by_pos[pos_v]
    comp_ok = (comp_v >= 0) & (comp_v < n_comp)
    if not np.any(comp_ok):
        return {}
    pos_v = pos_v[comp_ok]
    val_v = val_v[comp_ok]
    isl_v = isl_v[comp_ok]
    comp_v = comp_v[comp_ok]

    # 2. 按岛优先采用健康运行区间 [h_lo, h_hi]；若某岛无健康命中则回退至 [v_lo, v_hi]
    h_lo = max(v_lo, float(healthy_min))
    h_hi = min(v_hi, float(healthy_max))
    if h_lo <= h_hi:
        healthy_mask = (val_v >= h_lo) & (val_v <= h_hi)
        if np.any(healthy_mask):
            max_isl = int(isl_v.max())
            isl_has_healthy = np.zeros(max_isl + 1, dtype=bool)
            isl_has_healthy[isl_v[healthy_mask]] = True
            keep_mask = healthy_mask | (~isl_has_healthy[isl_v])
            pos_v = pos_v[keep_mask]
            val_v = val_v[keep_mask]
            isl_v = isl_v[keep_mask]
            comp_v = comp_v[keep_mask]

    if comp_v.size == 0:
        return {}

    # 3. 向量化统计每条拓扑母线的合理有效电压量测数、总关联度与最小节点 ID
    comp_counts = np.bincount(comp_v, minlength=n_comp)
    cand_comps = np.flatnonzero(comp_counts > 0).astype(np.int64, copy=False)
    if cand_comps.size == 0:
        return {}

    valid_node_mask = (comp_by_pos >= 0) & (comp_by_pos < n_comp) & (isl_by_pos >= 0)
    node_idx_range = np.flatnonzero(valid_node_mask)
    node_comps = comp_by_pos[node_idx_range]
    comp_isl = np.full(n_comp, -1, dtype=np.int64)
    comp_isl[node_comps] = isl_by_pos[node_idx_range]

    comp_deg = np.zeros(n_comp, dtype=np.int64)
    if node_degree is not None:
        deg_arr = np.asarray(node_degree, dtype=np.int64)
        if deg_arr.size == n_nodes:
            np.add.at(comp_deg, node_comps, deg_arr[node_idx_range])

    comp_min_id = np.full(n_comp, np.iinfo(np.int64).max, dtype=np.int64)
    if node_ids is not None:
        id_arr = np.asarray(node_ids, dtype=np.int64)
        if id_arr.size == n_nodes:
            np.minimum.at(comp_min_id, node_comps, id_arr[node_idx_range])
        else:
            np.minimum.at(comp_min_id, node_comps, node_idx_range)
    else:
        np.minimum.at(comp_min_id, node_comps, node_idx_range)

    cand_isl = comp_isl[cand_comps]
    cand_ok = cand_isl >= 0
    if not np.any(cand_ok):
        return {}
    cand_comps = cand_comps[cand_ok]
    cand_isl = cand_isl[cand_ok]
    cand_counts = comp_counts[cand_comps]
    cand_deg = comp_deg[cand_comps]
    cand_min_id = comp_min_id[cand_comps]

    # 按岛分组，优先量测数最多、次之拓扑母线总关联度最高、再按最小节点 ID 确定性破平
    order = np.lexsort((cand_min_id, -cand_deg, -cand_counts, cand_isl))
    sorted_isl = cand_isl[order]
    sorted_comps = cand_comps[order]
    first_in_island = np.empty(sorted_isl.size, dtype=bool)
    first_in_island[0] = True
    first_in_island[1:] = sorted_isl[1:] != sorted_isl[:-1]
    chosen_comps = sorted_comps[first_in_island]

    # 4. 为每岛选定的定电压母线计算稳健均值（>=3 条时先剔除偏离中位数 > outlier_tol 的离群坏数据）
    #    使用一次 O(K log K) 排序切片替代 O(K^2) 逐岛全表扫描，彻底消除万岛场景初始化瓶颈
    chosen_lut = np.zeros(n_comp, dtype=bool)
    chosen_lut[chosen_comps] = True
    chosen_mask = chosen_lut[comp_v]
    sel_comps = comp_v[chosen_mask]
    sel_vals = val_v[chosen_mask]
    if sel_comps.size == 0:
        return {}

    order_v = np.argsort(sel_comps, kind="mergesort")
    sc = sel_comps[order_v]
    sv = sel_vals[order_v]
    change = np.empty(sc.size, dtype=bool)
    change[0] = True
    change[1:] = sc[1:] != sc[:-1]
    starts = np.flatnonzero(change)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = sc.size
    group_comps = sc[starts]
    group_counts = ends - starts

    out: Dict[int, float] = {}
    tol = float(outlier_tol)
    # 快速向量化路径：样本数 < 3（或 tol <= 0）直接通过前缀和一步算均值，零 Python np.mean 循环
    small_mask = (group_counts < 3) if tol > 0.0 else np.ones(group_counts.size, dtype=bool)
    if np.any(small_mask):
        csum = np.empty(sv.size + 1, dtype=np.float64)
        csum[0] = 0.0
        np.cumsum(sv, out=csum[1:])
        s_idx = np.flatnonzero(small_mask)
        s_starts = starts[s_idx]
        s_ends = ends[s_idx]
        s_means = (csum[s_ends] - csum[s_starts]) / group_counts[s_idx]
        np.clip(s_means, v_lo, v_hi, out=s_means)
        for c_id, ref_v in zip(group_comps[s_idx].tolist(), s_means.tolist()):
            out[int(c_id)] = float(ref_v)

    if tol > 0.0 and not np.all(small_mask):
        large_idx = np.flatnonzero(~small_mask)
        for gi in large_idx.tolist():
            vals = sv[int(starts[gi]) : int(ends[gi])]
            med = float(np.median(vals))
            inliers = vals[np.abs(vals - med) <= tol]
            ref_v = float(np.mean(inliers)) if inliers.size > 0 else med
            out[int(group_comps[gi])] = min(max(ref_v, v_lo), v_hi)
    return out


def run_huber_irls_warmup(
    evaluate_fn,
    residual_fn,
    jacobian_fn,
    x0: np.ndarray,
    base_weights: np.ndarray,
    protected_mask: np.ndarray,
    *,
    v_cols: Optional[np.ndarray] = None,
    a_cols: Optional[np.ndarray] = None,
    v_min: float = 0.1,
    v_max: float = 1.9,
    a_min_rad: float = -float(np.deg2rad(270.0)),
    a_max_rad: float = float(np.deg2rad(270.0)),
    max_iter: int = 15,
    huber_k: float = 3.0,
    tol: float = 1e-4,
    shared_aat_solver: Optional[CholmodAAtNormalEquationSolver] = None,
    shared_aat_plan: Optional[CholmodAAtNormalEquationPlan] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """当初始纯 WLS 未收敛时，执行轻量 Huber IRLS 稳健热身，抑制大坏数据拉偏并产出稳健状态与残差。

    仅对非保护真实量测（``~protected_mask``）按 Huber 准则降权，物理等式约束与伪量测权重保持不变；
    求解完成后交回精确纯 WLS 迭代获取严格的最小二乘解。
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    w_base = np.asarray(base_weights, dtype=np.float64)
    prot = np.asarray(protected_mask, dtype=bool)
    n_meas = int(w_base.size)
    if x.size == 0 or n_meas == 0:
        return None
    vc = np.asarray(v_cols, dtype=np.intp) if v_cols is not None else np.empty(0, dtype=np.intp)
    ac = np.asarray(a_cols, dtype=np.intp) if a_cols is not None else np.empty(0, dtype=np.intp)

    def _oob(vec: np.ndarray) -> bool:
        if vc.size:
            vp = vec[vc]
            if (not np.all(np.isfinite(vp))) or float(vp.min()) < v_min or float(vp.max()) > v_max:
                return True
        if ac.size:
            ap = vec[ac]
            if (not np.all(np.isfinite(ap))) or float(ap.min()) < a_min_rad or float(ap.max()) > a_max_rad:
                return True
        return False

    if _oob(x):
        return None

    z_est = np.empty(n_meas, dtype=np.float64)
    res = np.empty(n_meas, dtype=np.float64)
    cand_z = np.empty(n_meas, dtype=np.float64)
    cand_r = np.empty(n_meas, dtype=np.float64)
    sqrt_w = np.sqrt(np.maximum(w_base, 0.0))

    evaluate_fn(x, z_est)
    residual_fn(z_est, res)
    if not np.all(np.isfinite(res)):
        return None

    def _huber_weights(r_vec: np.ndarray) -> np.ndarray:
        s = sqrt_w * np.abs(r_vec)
        big = (~prot) & (w_base > 0.0) & (s > huber_k)
        if not np.any(big):
            return w_base
        w_eff = w_base.copy()
        w_eff[big] = w_base[big] * (huber_k / s[big])
        return w_eff

    use_aat = True
    aat_solver = (
        shared_aat_solver
        if shared_aat_solver is not None
        else CholmodAAtNormalEquationSolver(assume_fixed_pattern=True, ordering_method="auto", multi_solve=True)
    )
    aat_plan = shared_aat_plan
    normal_solver = NormalEquationSolver()
    lam = 0.0

    for _it in range(1, max(1, int(max_iter)) + 1):
        if _oob(x):
            break
        H = jacobian_fn(x)
        w_cur = _huber_weights(res)
        obj = float(np.dot(w_cur, res * res))
        if not np.isfinite(obj):
            break
        norm_op = None
        gain = None
        if use_aat and is_sparse_matrix(H):
            try:
                if aat_plan is None or tuple(H.shape) != aat_plan.shape or int(H.nnz) != int(aat_plan.h_indices.size):
                    aat_plan = CholmodAAtNormalEquationPlan.from_jacobian(H)
                norm_op, rhs = aat_plan.assemble(
                    H,
                    res,
                    w_cur,
                    uniform_weight=None,
                    weights_are_uniform=False,
                    weighted_residual=None,
                    assume_fixed_weights=False,
                    copy_rhs=False,
                    assume_pattern_matches=True,
                )
            except Exception:
                use_aat = False
                aat_plan = None
        if norm_op is None:
            gain, rhs = build_normal_equations(H, res, w_cur)

        accepted = False
        try_lam = lam
        for _ls in range(10):
            if norm_op is not None:
                allow_slow = (x.size < 500) or (try_lam >= 10.0)
                dx, _ = aat_solver.solve(
                    norm_op,
                    rhs,
                    return_factor_diag=False,
                    damping=try_lam,
                    allow_slow_fallback=allow_slow,
                )
            else:
                if try_lam > 0.0:
                    if is_sparse_matrix(gain):
                        g_solve = gain + SP_EYE(gain.shape[0], format="csc", dtype=np.float64) * try_lam
                    else:
                        g_solve = np.array(gain, dtype=np.float64, copy=True)
                        g_solve[np.diag_indices_from(g_solve)] += try_lam
                else:
                    g_solve = gain
                dx, _ = normal_solver.solve(g_solve, rhs, return_factor_diag=False)
            if dx.size == 0 or not np.all(np.isfinite(dx)):
                try_lam = max(try_lam * 10.0, 1e-3)
                if try_lam > 1e5:
                    break
                continue
            max_dx = float(np.max(np.abs(dx)))
            if max_dx < tol and try_lam == 0.0:
                return x, res, z_est
            if _it >= 4 and max_dx < 1e-2 and try_lam == 0.0 and not _oob(x):
                # Huber 仅负责将发散初值拉回稳健吸引域，达到 1e-2 即提前交回精确 WLS 完成二次收敛
                return x, res, z_est
            s = 1.0
            if vc.size:
                dv_m = float(np.max(np.abs(dx[vc])))
                if dv_m > 0.35:
                    s = min(s, 0.35 / dv_m)
            if ac.size:
                da_m = float(np.max(np.abs(dx[ac])))
                if da_m > 0.75:
                    s = min(s, 0.75 / da_m)
            cand = x + s * dx
            if _oob(cand):
                try_lam = max(try_lam * 10.0, 1e-3)
                if try_lam > 1e5:
                    break
                continue
            evaluate_fn(cand, cand_z)
            residual_fn(cand_z, cand_r)
            cand_obj = float(np.dot(w_cur, cand_r * cand_r))
            if np.isfinite(cand_obj) and cand_obj <= obj + max(1e-12, 1e-10 * (abs(obj) + 1.0)):
                x = cand
                z_est, cand_z = cand_z, z_est
                res, cand_r = cand_r, res
                accepted = True
                if s == 1.0:
                    lam = 0.0 if try_lam <= 1e-5 else try_lam * 0.25
                else:
                    lam = max(try_lam, 1e-3)
                if max_dx < tol and try_lam > 0.0:
                    lam = 0.0
                break
            try_lam = max(try_lam * 5.0, 1e-3)
            if try_lam > 1e5:
                break
        if not accepted:
            break
    return x, res, z_est


def compute_active_column_degrees(
    H,
    active_mask: np.ndarray,
    weights: np.ndarray,
) -> Optional[np.ndarray]:
    """统计当前活跃（有效且权重大于零）量测在稀疏雅可比矩阵各状态列上的非零覆盖度。"""
    if H is None or not is_sparse_matrix(H):
        return None
    n_rows, n_cols = int(H.shape[0]), int(H.shape[1])
    if n_cols == 0:
        return np.empty(0, dtype=np.int32)
    mask_arr = np.asarray(active_mask, dtype=bool)
    w_arr = np.asarray(weights, dtype=np.float64)
    if mask_arr.size != n_rows or w_arr.size != n_rows:
        return None
    act = mask_arr & (w_arr > 0.0)
    H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
    act_rows = np.flatnonzero(act)
    if act_rows.size == 0:
        return np.zeros(n_cols, dtype=np.int32)
    sub = H_csr[act_rows]
    return np.bincount(sub.indices, minlength=n_cols).astype(np.int32, copy=False)


def filter_batch_preserving_column_coverage(
    H_csr,
    candidate_rows: Sequence[int],
    col_degrees: Optional[np.ndarray],
) -> Tuple[List[int], List[int]]:
    """按候选顺序贪心筛选不会使任何状态列活跃覆盖度归零的安全删除子批。

    返回 ``(safe_local_indices, blocked_local_indices)``，均为 ``candidate_rows`` 的局部下标。
    单条或多条删除若会导致原本有支撑的状态列覆盖度归零，必然引起正规方程严格奇异，
    此处在 O(nnz) 微秒级前置拦截，避免删除临界量测或同时删除临界量测对。
    """
    n_cand = len(candidate_rows)
    if n_cand == 0 or H_csr is None or col_degrees is None:
        return list(range(n_cand)), []
    indptr = H_csr.indptr
    indices = H_csr.indices
    n_rows = int(H_csr.shape[0])
    n_cols = int(col_degrees.size)
    pending_dec: Dict[int, int] = {}
    safe_idx: List[int] = []
    blocked_idx: List[int] = []
    for local_i, raw_row in enumerate(candidate_rows):
        r = int(raw_row)
        if not (0 <= r < n_rows):
            safe_idx.append(local_i)
            continue
        cols = indices[indptr[r] : indptr[r + 1]]
        if cols.size == 0:
            safe_idx.append(local_i)
            continue
        # 去重以防稀疏矩阵存在重复列项
        uniq_cols = cols if cols.size <= 1 else np.unique(cols)
        would_zero = False
        for c_val in uniq_cols:
            c = int(c_val)
            if 0 <= c < n_cols:
                deg = int(col_degrees[c])
                if deg > 0 and (deg - pending_dec.get(c, 0)) <= 1:
                    would_zero = True
                    break
        if would_zero:
            blocked_idx.append(local_i)
        else:
            safe_idx.append(local_i)
            for c_val in uniq_cols:
                c = int(c_val)
                if 0 <= c < n_cols and int(col_degrees[c]) > 0:
                    pending_dec[c] = pending_dec.get(c, 0) + 1
    return safe_idx, blocked_idx


def check_active_structural_full_rank(
    H,
    active_mask: np.ndarray,
    weights: np.ndarray,
) -> bool:
    """检查剔除坏数据后的活跃雅可比子矩阵是否保持结构列满秩（防临界对/临界环致奇异）。"""
    if H is None or not is_sparse_matrix(H):
        return True
    n_rows, n_cols = int(H.shape[0]), int(H.shape[1])
    if n_cols == 0:
        return True
    mask_arr = np.asarray(active_mask, dtype=bool)
    w_arr = np.asarray(weights, dtype=np.float64)
    if mask_arr.size != n_rows or w_arr.size != n_rows:
        return True
    act_rows = np.flatnonzero(mask_arr & (w_arr > 0.0))
    if act_rows.size < n_cols:
        return False
    H_csr = H if getattr(H, "format", None) == "csr" else H.tocsr()
    sub = H_csr[act_rows]
    # 快速短路：任意列覆盖度为 0 则必然结构秩亏
    if np.any(np.bincount(sub.indices, minlength=n_cols) == 0):
        return False
    return int(sparse_structural_rank(sub)) >= n_cols


def build_switch_current_curvature_columns(
    n_state: int,
    sw_rows: np.ndarray,
    col_re: np.ndarray,
    col_im: np.ndarray,
    x: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    *,
    min_curv: float = 1e-2,
) -> Tuple[Optional[SP_CSC_MATRIX], bool]:
    """为零阻抗支路/断路器电流幅值负残差测点构造精确秩 1 半正定二阶曲率列矩阵 V_curv。

    对电流幅值量测 h(I) = sqrt(I_re^2 + I_im^2) 且残差 r = z - |I| < 0：
        -w * r * nabla^2 |I| = (w * (-r) / |I|) * t * t^T >= 0,
    其中切向单位向量 t = (-I_im / |I|, I_re / |I|)^T。
    将该精确半正定二阶曲率项以额外稀疏列 V_curv 追加到 A_csc 右侧（使 A A^T + V V^T 成为
    精确半正定二阶牛顿海森矩阵），其稀疏非零模式是原正规方程的严格子集，能彻底消除轻载零阻抗
    支路在 I ≈ 0 锥尖附近的 50~100 倍 Gauss-Newton 切向过冲振荡。
    为保证相邻迭代可复用符号分析，V_curv 固定保留 sw_rows.size 列与每列 2 个排序后的非零位置，
    仅在存在满足 curv > min_curv 的活跃负残差测点时返回 (V_csc, True)，否则短路返回 (None, False)。
    """
    k = int(sw_rows.size)
    if k <= 0 or n_state <= 0:
        return None, False
    w_s = weight[sw_rows]
    r_s = residual[sw_rows]
    neg_mask = (w_s > 0.0) & (r_s < 0.0)
    if not np.any(neg_mask):
        return None, False
    ire = x[col_re]
    iim = x[col_im]
    imag = np.hypot(ire, iim)
    active = neg_mask & (imag > 1e-12)
    if not np.any(active):
        return None, False
    curv = np.zeros(k, dtype=np.float64)
    curv[active] = (w_s[active] * (-r_s[active])) / imag[active]
    sig_mask = curv > float(min_curv)
    if not np.any(sig_mask):
        return None, False
    amp = np.zeros(k, dtype=np.float64)
    amp[sig_mask] = np.sqrt(curv[sig_mask]) / imag[sig_mask]
    v_re = -amp * iim
    v_im = amp * ire
    # 保证每列行号严格升序（通常 col_re < col_im，此处显式兼容一般顺序）
    swap = col_re > col_im
    row0 = np.where(swap, col_im, col_re).astype(np.int32, copy=False)
    row1 = np.where(swap, col_re, col_im).astype(np.int32, copy=False)
    val0 = np.where(swap, v_im, v_re)
    val1 = np.where(swap, v_re, v_im)
    indices = np.empty(2 * k, dtype=np.int32)
    indices[0::2] = row0
    indices[1::2] = row1
    data = np.empty(2 * k, dtype=np.float64)
    data[0::2] = val0
    data[1::2] = val1
    indptr = np.arange(0, 2 * k + 1, 2, dtype=np.int32)
    return SP_CSC_MATRIX((data, indices, indptr), shape=(int(n_state), k)), True


def detect_active_box_pinned_cols(
    x: np.ndarray,
    rhs: np.ndarray,
    v_cols: np.ndarray,
    v_min: float,
    v_max: float,
    *,
    margin: float = 1e-4,
) -> np.ndarray:
    """识别已紧贴电压盒约束边界且无约束下降方向指向墙外的活跃约束列索引集合 A。"""
    if v_cols.size == 0 or x.size == 0 or rhs.size == 0:
        return np.empty(0, dtype=np.intp)
    v_vals = x[v_cols]
    r_vals = rhs[v_cols]
    lo_thr = float(v_min) + float(margin)
    hi_thr = float(v_max) - float(margin)
    pin_mask = ((v_vals <= lo_thr) & (r_vals < 0.0)) | ((v_vals >= hi_thr) & (r_vals > 0.0))
    if not np.any(pin_mask):
        return np.empty(0, dtype=np.intp)
    return np.asarray(v_cols[pin_mask], dtype=np.intp)


def solve_box_pinned_normal_equations(
    normal_operand,
    gain,
    rhs: np.ndarray,
    pinned_cols: np.ndarray,
    *,
    damping: float = 0.0,
    solver_cache: Optional[Dict] = None,
) -> Tuple[np.ndarray, bool]:
    """在活跃盒约束集合 A 上施加精确等式约束 dx[A] = 0，求解自由子空间约化牛顿步。

    构造块解耦系统 A_pin = [A_zeroed, E_A]，令 A_pin A_pin^T = blkdiag(G_FF, I_A) 且 rhs_pin[A] = 0，
    从而精确得到 dx_F = (G_FF + beta I)^{-1} g_F、dx_A = 0：
    - 无大数罚项条件数劣化，无论 beta == 0 或 beta > 0 均严格对称正定；
    - 使用独立局部 solver_cache 复用同活跃集符号分析，绝不污染外层 BDR 持久因子。
    """
    n_state = int(rhs.size)
    rhs_pin = np.asarray(rhs, dtype=np.float64).copy()
    if pinned_cols.size == 0 or n_state == 0:
        return np.full(n_state, np.nan, dtype=np.float64), False
    rhs_pin[pinned_cols] = 0.0
    beta = float(damping)
    k = int(pinned_cols.size)

    if normal_operand is not None and SP_ISSPARSE(normal_operand):
        A_csc = normal_operand if getattr(normal_operand, "format", None) == "csc" else normal_operand.tocsc()
        pin_mask = np.zeros(n_state, dtype=bool)
        pin_mask[pinned_cols] = True
        a_data = A_csc.data.copy()
        a_data[pin_mask[A_csc.indices]] = 0.0
        A_zero = SP_CSC_MATRIX((a_data, A_csc.indices, A_csc.indptr), shape=A_csc.shape, copy=False)
        e_cols = np.arange(k + 1, dtype=np.int32)
        e_rows = pinned_cols.astype(np.int32, copy=False)
        e_data = np.ones(k, dtype=np.float64)
        E_pin = SP_CSC_MATRIX((e_data, e_rows, e_cols), shape=(n_state, k))
        A_pin = SP_HSTACK([A_zero, E_pin], format="csc")

        pin_key = (tuple(int(c) for c in pinned_cols.tolist()), A_csc.shape, int(A_csc.nnz))
        pin_solver = None
        if solver_cache is not None:
            if solver_cache.get("key") == pin_key and solver_cache.get("solver") is not None:
                pin_solver = solver_cache["solver"]
            else:
                prev_s = solver_cache.get("solver")
                if prev_s is not None and hasattr(prev_s, "_reset_pardiso"):
                    try:
                        prev_s._reset_pardiso()
                    except Exception:
                        pass
                pin_solver = CholmodAAtNormalEquationSolver(assume_fixed_pattern=True, ordering_method="amd")
                pin_solver._pardiso_disabled = True
                solver_cache["key"] = pin_key
                solver_cache["solver"] = pin_solver
        else:
            pin_solver = CholmodAAtNormalEquationSolver(assume_fixed_pattern=False, ordering_method="amd")
            pin_solver._pardiso_disabled = True

        dx, _ = pin_solver.solve(
            A_pin,
            rhs_pin,
            return_factor_diag=False,
            damping=beta,
            allow_slow_fallback=(n_state < 500 or beta >= 10.0),
        )
        if dx.size == n_state and np.all(np.isfinite(dx)):
            dx[pinned_cols] = 0.0
            return dx, True
        return np.full(n_state, np.nan, dtype=np.float64), False

    if gain is not None:
        if SP_ISSPARSE(gain):
            G_coo = gain.tocoo()
            pin_mask = np.zeros(n_state, dtype=bool)
            pin_mask[pinned_cols] = True
            keep = (~pin_mask[G_coo.row]) & (~pin_mask[G_coo.col])
            rows = np.concatenate([G_coo.row[keep], pinned_cols])
            cols = np.concatenate([G_coo.col[keep], pinned_cols])
            vals = np.concatenate([G_coo.data[keep], np.ones(k, dtype=np.float64)])
            G_pin = SP_CSC_MATRIX((vals, (rows, cols)), shape=(n_state, n_state))
            if beta > 0.0:
                G_pin = G_pin + SP_EYE(n_state, format="csc", dtype=np.float64) * beta
        else:
            G_pin = np.array(gain, dtype=np.float64, copy=True)
            G_pin[pinned_cols, :] = 0.0
            G_pin[:, pinned_cols] = 0.0
            G_pin[pinned_cols, pinned_cols] = 1.0
            if beta > 0.0:
                G_pin[np.diag_indices_from(G_pin)] += beta
        dx, _ = NormalEquationSolver().solve(G_pin, rhs_pin, return_factor_diag=False)
        if dx.size == n_state and np.all(np.isfinite(dx)):
            dx[pinned_cols] = 0.0
            return dx, True
    return np.full(n_state, np.nan, dtype=np.float64), False


class AndersonAccelerator:
    """Type-II Anderson 不动点加速器，用于加速大残差 Gauss-Newton 慢线性收敛。

    将相同阻尼下连续满步迭代视为不动点映射 ``G(x_k) = x_{k+1} = x_k + f_k``：
    仅在连续满步呈现慢线性收缩（步长收缩比处于 ``[0.55, 1.05]``）时求解极小
    正则最小二乘 ``min ||f_k - dF @ gamma||^2 + reg ||gamma||^2`` 并提出外推点
    ``x_aa = x_{k+1} - (dX + dF) @ gamma``。外推点是否采纳由调用方执行严格的物理
    状态边界检查与目标函数单调下降检验 ``J(x_aa) < J(x_{k+1})`` 决定，确保零回退。
    """

    def __init__(
        self,
        m: int = 4,
        reg: float = 1e-4,
        max_gamma_norm: float = 1e6,
        min_slow_streak: int = 3,
    ) -> None:
        self.m = max(2, int(m))
        self.reg = float(reg)
        self.max_gamma_norm = float(max_gamma_norm)
        self.min_slow_streak = max(2, int(min_slow_streak))
        self._last_x: Optional[np.ndarray] = None
        self._last_f: Optional[np.ndarray] = None
        self._last_damping: Optional[float] = None
        self._dx_hist: List[np.ndarray] = []
        self._df_hist: List[np.ndarray] = []
        self._slow_streak: int = 0

    def reset(self) -> None:
        """清空历史切空间窗口与慢收缩计数。"""
        self._last_x = None
        self._last_f = None
        self._last_damping = None
        self._dx_hist.clear()
        self._df_hist.clear()
        self._slow_streak = 0

    def update_and_propose(
        self,
        x_prev: np.ndarray,
        x_next: np.ndarray,
        *,
        step_ratio: float,
        damping: float = 0.0,
        allow_propose: bool = True,
    ) -> Optional[np.ndarray]:
        """记录本步满步转移并按需通过列归一化薄 QR 构造 Anderson 外推候选点。"""
        if x_prev.size == 0 or x_prev.size != x_next.size:
            self.reset()
            return None
        d_cur = float(damping)
        if self._last_damping is not None and abs(d_cur - self._last_damping) > max(1e-12, 0.05 * max(d_cur, self._last_damping)):
            # 阻尼发生显著变化时不动点映射已改变，必须重置切空间窗口
            self.reset()
        self._last_damping = d_cur

        f_cur = np.asarray(x_next - x_prev, dtype=np.float64)
        if not np.all(np.isfinite(f_cur)):
            self.reset()
            return None
        if self._last_x is not None and self._last_f is not None:
            dx_pair = np.asarray(x_prev - self._last_x, dtype=np.float64)
            df_pair = np.asarray(f_cur - self._last_f, dtype=np.float64)
            if np.all(np.isfinite(dx_pair)) and np.all(np.isfinite(df_pair)):
                self._dx_hist.append(dx_pair)
                self._df_hist.append(df_pair)
                if len(self._dx_hist) > self.m:
                    self._dx_hist.pop(0)
                    self._df_hist.pop(0)
        self._last_x = np.asarray(x_prev, dtype=np.float64).copy()
        self._last_f = f_cur

        if allow_propose and (0.55 <= float(step_ratio) <= 1.05):
            self._slow_streak += 1
        else:
            self._slow_streak = 0
            return None

        k = len(self._df_hist)
        if self._slow_streak < self.min_slow_streak or k < 2:
            return None

        dF = np.column_stack(self._df_hist)
        col_norms = np.linalg.norm(dF, axis=0)
        if not np.all(np.isfinite(col_norms)) or float(np.max(col_norms)) <= 1e-15:
            return None
        valid_cols = col_norms > max(1e-15, 1e-12 * float(np.max(col_norms)))
        if not np.any(valid_cols):
            return None
        dF_v = dF[:, valid_cols]
        norms_v = col_norms[valid_cols]
        dF_scaled = dF_v / norms_v[None, :]
        try:
            Q, R = np.linalg.qr(dF_scaled, mode="reduced")
            gamma_s, *_ = np.linalg.lstsq(R, Q.T @ f_cur, rcond=max(self.reg, 1e-6))
        except np.linalg.LinAlgError:
            return None
        gamma = gamma_s / norms_v
        if (not np.all(np.isfinite(gamma))) or float(np.linalg.norm(gamma)) > self.max_gamma_norm:
            return None

        dX = np.column_stack(self._dx_hist)[:, valid_cols]
        extrap_disp = -((dX + dF_v) @ gamma)
        f_inf = float(np.max(np.abs(f_cur)))
        if (not np.all(np.isfinite(extrap_disp))) or float(np.max(np.abs(extrap_disp))) > max(2.0, 500.0 * f_inf):
            return None
        x_aa = np.asarray(x_next, dtype=np.float64) + extrap_disp
        if not np.all(np.isfinite(x_aa)):
            return None
        return x_aa

    def on_accepted(self, step_norm: Optional[float] = None) -> None:
        """大跨度外推采纳后重置旧切空间；局部线性化半径内（<=0.02）小步外推保留滚动窗口以形成多维 Krylov 加速。"""
        if step_norm is None or (not np.isfinite(float(step_norm))) or float(step_norm) > 0.02:
            self.reset()
        else:
            # 仅清空上一锚点对，防止外推跳转向量被误算作不动点差分，同时保留已累积的高维切空间
            self._last_x = None
            self._last_f = None

    def on_rejected(self) -> None:
        """外推点未降低目标函数时清空窗口并重新积累，防止连续无效试探。"""
        self.reset()


class NielsenTrustRegionController:
    """带 Nielsen (1999) 退避倍率 ν 与反极限环安全下界记忆的信赖域阻尼控制器。

    解决无状态阻尼更新律在强非凸/大残差临界阻尼附近出现的“1/3 猛降 → 撞壁回溯/归零弹回 → 再猛降”周期极限环：
    1. 完整实现 Nielsen 退避倍率 ``nu``（每次回溯 ``lam *= nu; nu = min(16, 2*nu)``，良态满步重置 ``nu=2``）；
    2. 对流体等非光滑悬崖极微步（``alpha < cliff_step``）支持单步跃迁（``cliff_bump``）；
    3. 当探测到 2 次“降阻尼/归零 → 回溯或严重失配弹回”反弹时，锁定临界阻尼安全下界 ``bounce_floor``，
       让系统稳停在最优曲率阻尼台阶并使 Anderson 加速窗口稳定生效；
    4. 一旦目标函数相对反弹点显著下降（默认 >= 10%），说明已离开高曲率区，自动释放 ``bounce_floor`` 恢复二阶收敛。
    """

    def __init__(self) -> None:
        self.nu: float = 2.0
        self.bounce_floor: float = 0.0
        self.bounce_ref_obj: float = np.inf
        self.bounce_count: int = 0
        self._was_reduced: bool = False
        self._just_zeroed: bool = False
        self._last_safe_nonzero: float = 0.0
        self._good_streak: int = 0
        self._zero_bounce_count: int = 0

    def reset(self) -> None:
        self.nu = 2.0
        self.bounce_floor = 0.0
        self.bounce_ref_obj = np.inf
        self.bounce_count = 0
        self._was_reduced = False
        self._just_zeroed = False
        self._last_safe_nonzero = 0.0
        self._good_streak = 0
        self._zero_bounce_count = 0

    def notify_damping_activated(self, new_lam: float, current_obj: float = np.inf) -> None:
        """外部由无阻尼态重新激活阻尼时调用，检测“归零即撞壁弹回”极限环。"""
        lam_v = max(0.0, float(new_lam))
        if self._just_zeroed and lam_v > 0.0:
            self._zero_bounce_count += 1
            self.bounce_count += 1
            if self.bounce_count >= 2:
                self.bounce_floor = max(self.bounce_floor, max(self._last_safe_nonzero, 1e-3))
                if np.isfinite(current_obj):
                    self.bounce_ref_obj = min(self.bounce_ref_obj, float(current_obj))
        self._just_zeroed = False
        self._was_reduced = False
        if lam_v > 0.0:
            self._last_safe_nonzero = max(self._last_safe_nonzero, lam_v)

    def update(
        self,
        damping: float,
        rhs: Optional[np.ndarray],
        dx: Optional[np.ndarray],
        actual_drop: float,
        step_scale: float = 1.0,
        *,
        current_obj: float = np.inf,
        half_obj: bool = True,
        diag_reg: Optional[np.ndarray] = None,
        min_damping: float = 0.0,
        max_damping: float = 100.0,
        zero_cutoff: float = 1e-4,
        dx_growing: bool = False,
        cliff_step: float = 0.0,
        cliff_bump: float = 10.0,
    ) -> Tuple[float, float]:
        lam = max(0.0, float(damping))
        alpha = float(step_scale)
        d_min = max(0.0, float(min_damping))
        d_max = max(d_min, float(max_damping))

        # 目标函数相对上次反弹点下降 >= 10% 或连续 8 步完美二次匹配，自动释放下界记忆恢复二阶收敛
        if self.bounce_floor > 0.0 and (
            (
                np.isfinite(current_obj)
                and np.isfinite(self.bounce_ref_obj)
                and float(current_obj) <= 0.90 * self.bounce_ref_obj
            )
            or self._good_streak >= 8
        ):
            self.bounce_floor = 0.0
            self.bounce_count = 0
            self.bounce_ref_obj = np.inf
            self.nu = 2.0
            self._good_streak = 0

        eff_min = min(d_max, max(d_min, self.bounce_floor))
        eff_zero_cut = float(zero_cutoff)
        if self._zero_bounce_count > 0 and eff_zero_cut > 0.0:
            eff_zero_cut = max(1e-8, eff_zero_cut * (0.01 ** min(self._zero_bounce_count, 3)))
        base_new, rho = nielsen_lm_damping_update(
            lam,
            rhs,
            dx,
            actual_drop,
            alpha,
            half_obj=half_obj,
            diag_reg=diag_reg,
            min_damping=eff_min,
            max_damping=d_max,
            zero_cutoff=eff_zero_cut,
            dx_growing=dx_growing,
        )
        if lam <= 0.0:
            return 0.0, float(rho)

        rho_eff = (rho if rho <= 1.0 else max(0.0, 2.0 - rho)) if np.isfinite(rho) else 0.0
        is_backtrack = (alpha < 0.5) or (np.isfinite(rho) and rho <= 0.0)
        is_poor_model = (alpha >= 0.8) and (rho_eff < 0.45)

        if is_backtrack or is_poor_model:
            self._good_streak = 0
            if self._was_reduced:
                # 严格仅对“缩减阻尼后立即碰壁弹回”的真实极限环计数并记录安全下界（上限 <= 0.25 * d_max）
                self.bounce_count += 1
                if self.bounce_count >= 2:
                    floor_cap = max(d_min, 0.25 * d_max)
                    self.bounce_floor = min(floor_cap, max(self.bounce_floor, 1.25 * lam))
                    if np.isfinite(current_obj):
                        self.bounce_ref_obj = min(self.bounce_ref_obj, float(current_obj))
            if is_backtrack:
                if cliff_step > 0.0 and alpha < float(cliff_step):
                    bump = max(self.nu, float(cliff_bump))
                else:
                    bump = self.nu
                new_lam = min(d_max, max(eff_min, lam * bump, max(1e-3, eff_min)))
                self.nu = min(4.0, self.nu * 2.0)
            else:
                bump = max(1.5, 0.5 * self.nu) if rho_eff >= 0.25 else max(2.0, self.nu)
                new_lam = min(d_max, max(eff_min, lam * bump))
            self._was_reduced = False
            self._just_zeroed = False
            self._last_safe_nonzero = new_lam
            return float(new_lam), float(rho)

        # 良态步：重置退避倍率，受 eff_min 保护不跌穿已探明临界阻尼下界
        if alpha >= 0.8:
            self.nu = 2.0
            if self.bounce_floor > 0.0 and (0.85 <= rho <= 1.15) and not dx_growing:
                self._good_streak += 1
            else:
                self._good_streak = 0
        else:
            self._good_streak = 0
        new_lam = float(base_new)
        if eff_min > 0.0 and new_lam < eff_min:
            new_lam = eff_min
        if new_lam <= 0.0:
            self._just_zeroed = True
            self._last_safe_nonzero = max(lam, float(zero_cutoff))
            self._was_reduced = True
        else:
            self._just_zeroed = False
            if new_lam < lam - 1e-15:
                self._was_reduced = True
            elif new_lam >= 0.99 * lam:
                self._last_safe_nonzero = new_lam
        return float(new_lam), float(rho)


def nielsen_lm_damping_update(
    damping: float,
    rhs: Optional[np.ndarray],
    dx: Optional[np.ndarray],
    actual_drop: float,
    step_scale: float = 1.0,
    *,
    half_obj: bool = True,
    diag_reg: Optional[np.ndarray] = None,
    min_damping: float = 0.0,
    max_damping: float = 100.0,
    zero_cutoff: float = 1e-4,
    dx_growing: bool = False,
) -> Tuple[float, float]:
    """基于大残差对称增益比 |ρ - 1| 的 Nielsen (1999) 连续信赖域阻尼更新律。

    对于正则正规方程 ``(G + λ D) Δx = g``（其中 ``g = rhs = H^T W r``）：
    - 目标函数 ``J = c * r^T W r``（``c = 0.5`` 当 ``half_obj=True``，否则 ``c = 1.0``）；
    - 二次模型在步长 ``α = step_scale`` 处的精确预测下降量为：
      ``ΔJ_pred = c * [α(2 - α) (g^T Δx) + α^2 λ (Δx^T D Δx)]``；
    - 对称匹配度 ``ρ_eff = 1 - |ρ - 1|``（既识别 ``ρ < 0.5`` 的高估，也识别大残差非凸 ``ρ > 1.5`` 失配）：
      * ``ρ ≈ 1``（且步长未放大）时按 ``max(1/3, 1 - (2ρ_eff - 1)^3)`` 平滑缩减阻尼；
      * ``ρ ≈ 0.5`` 或 ``1.5`` 时因子趋近 ``1.0``，自动稳停在最优曲率阻尼台阶；
      * ``ρ < 0.25`` 或 ``ρ > 1.75`` 时提前温和抬升阻尼，防止撞回溯或过冲震荡。

    返回 ``(new_damping, rho)``。
    """
    lam = max(0.0, float(damping))
    alpha = float(step_scale)
    d_min = max(0.0, float(min_damping))
    d_max = max(d_min, float(max_damping))

    if lam <= 0.0:
        return 0.0, 1.0

    rho = np.nan
    if rhs is not None and dx is not None and rhs.size > 0 and rhs.size == dx.size and alpha > 0.0:
        g_dot_dx = float(np.dot(rhs, dx))
        if diag_reg is not None and diag_reg.size == dx.size:
            reg_quad = float(np.dot(dx * dx, np.maximum(diag_reg, 1e-12)))
        else:
            reg_quad = float(np.dot(dx, dx))
        pred_raw = alpha * (2.0 - alpha) * g_dot_dx + (alpha * alpha) * lam * reg_quad
        pred_drop = (0.5 * pred_raw) if half_obj else pred_raw
        if np.isfinite(pred_drop) and pred_drop > 1e-30:
            rho = float(actual_drop) / pred_drop

    if alpha >= 0.8 and np.isfinite(rho) and rho > 0.0:
        # 大残差双向对称信赖域更新律（Nielsen 1999 连续缩减 + 临界阻尼防死区抬升）：
        # - rho_eff > 0.50 且步长未放大时按 Nielsen 原始连续律 max(1/3, 1 - (2*rho_eff - 1)^3) 平滑缩减，
        #   消除 [0.50, 0.70) 人为锁死带导致平滑峡谷阻尼降不下去的问题；
        # - rho_eff in [0.45, 0.50] 稳停；
        # - rho_eff < 0.45（实际降幅不足模型一半或严重非凸失配）果断按 1.5x~2.0x 越过临界阻尼。
        rho_eff = rho if rho <= 1.0 else max(0.0, 2.0 - rho)
        if rho_eff > 0.50:
            factor = max(1.0 / 3.0, 1.0 - (2.0 * rho_eff - 1.0) ** 3)
            if dx_growing and factor < 1.0:
                factor = 1.0
        elif rho_eff >= 0.45:
            factor = 1.0
        else:
            factor = 2.0 if rho_eff < 0.25 else 1.5
        new_lam = lam * factor
        # 超高电纳/刚性网络在 Jacobi 相对阻尼模式下，若主体刚度（90 分位 diag_reg）> 5e7，则标称 zero_cutoff=1e-3
        # 对应的主体绝对正则项 λ·d_90 仍高达 5e4~1e6，过早一刀归零会导致弱模态瞬间过冲并反复弹回重开阻尼；
        # 采用 90 分位而非 max 避免被单个零注入/换流器极端权重节点误触发，对普通网络保持 100% 零扰动。
        eff_zero_cutoff = float(zero_cutoff)
        if diag_reg is not None and diag_reg.size > 0 and eff_zero_cutoff > 0.0:
            d_reg_scale = float(np.percentile(diag_reg, 90))
            if np.isfinite(d_reg_scale) and d_reg_scale > 5e7:
                eff_zero_cutoff = min(eff_zero_cutoff, max(1e-7, 5e2 / d_reg_scale))
        if new_lam < eff_zero_cutoff and d_min <= 0.0 and (0.75 <= rho <= 1.25) and not dx_growing:
            new_lam = 0.0
        else:
            new_lam = min(d_max, max(d_min, new_lam))
        return float(new_lam), float(rho)

    if alpha < 0.5 or (np.isfinite(rho) and rho <= 0.0):
        new_lam = min(d_max, max(lam * 2.0, max(1e-3, d_min)))
        return float(new_lam), float(rho) if np.isfinite(rho) else 0.0

    return float(min(d_max, max(d_min, lam))), float(rho) if np.isfinite(rho) else 0.5


def projected_box_kkt_residual(
    x: np.ndarray,
    rhs: np.ndarray,
    diag_g: np.ndarray,
    v_cols: np.ndarray,
    v_min: float,
    v_max: float,
    *,
    core_cols: Optional[np.ndarray] = None,
    margin: float = 1e-4,
) -> Tuple[float, float]:
    """计算带盒约束非线性最小二乘的归一化投影 KKT 一阶最优性残差。

    扣除已紧贴电压下界/上界且梯度继续指向可行域外侧的活跃不等式约束分量后，
    按 ``|rhs_i| / max(diag_g_i, 1.0)`` 返回 ``(core_kkt, all_kkt)``。
    """
    if rhs.size == 0 or diag_g.size != rhs.size or x.size != rhs.size:
        return np.inf, np.inf
    eff_rhs = np.asarray(rhs, dtype=np.float64).copy()
    if v_cols.size > 0:
        vc = np.asarray(v_cols, dtype=np.intp)
        xv = x[vc]
        rv = eff_rhs[vc]
        pin = ((xv <= float(v_min) + margin) & (rv < 0.0)) | ((xv >= float(v_max) - margin) & (rv > 0.0))
        if np.any(pin):
            eff_rhs[vc[pin]] = 0.0
    scaled = np.abs(eff_rhs) / np.maximum(np.asarray(diag_g, dtype=np.float64), 1.0)
    all_kkt = float(np.max(scaled)) if scaled.size else 0.0
    if core_cols is not None and core_cols.size > 0:
        core_kkt = float(np.max(scaled[np.asarray(core_cols, dtype=np.intp)]))
    else:
        core_kkt = all_kkt
    return core_kkt, all_kkt






