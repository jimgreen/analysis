import importlib

import numpy as np
from scipy.sparse.linalg import splu, spsolve


OPTIONAL_SPARSE_SOLVERS = {}
OPTIONAL_SPARSE_MISSING = set()
OPTIONAL_SOLVER_CANDIDATES = {
    "pypardiso": ("pypardiso", "spsolve"),
    "umfpack": ("scikits.umfpack", "spsolve"),
    "sksparse.klu.klu_solve": ("sksparse.klu", "klu_solve"),
    "klu_solve": ("sksparse.klu", "klu_solve"),
    "pyklu": ("PyKLU", "Klu"),
    "klu": ("sksparse.klu", "spsolve"),
    "klu_alt": ("klu", "solve"),
}


def as_solver_csc(matrix):
    return matrix if getattr(matrix, "format", None) == "csc" else matrix.tocsc()


def load_named_sparse_solver(solver_name):
    """Return a named optional sparse solver when installed."""
    solver_name = str(solver_name).strip().lower()
    if solver_name in OPTIONAL_SPARSE_SOLVERS:
        return OPTIONAL_SPARSE_SOLVERS[solver_name]
    if solver_name in OPTIONAL_SPARSE_MISSING:
        return None

    candidate_names = (
        ("pyklu", "sksparse.klu.klu_solve", "klu", "klu_alt", "pypardiso")
        if solver_name == "auto"
        else (solver_name,)
    )
    for candidate_name in candidate_names:
        module_name, func_name = OPTIONAL_SOLVER_CANDIDATES.get(candidate_name, (None, None))
        if module_name is None:
            continue
        try:
            if importlib.util.find_spec(module_name) is None:
                continue
        except (ImportError, ValueError):
            continue
        try:
            module = importlib.import_module(module_name)
            solver = getattr(module, func_name, None)
        except Exception:
            continue
        if solver is not None:
            if candidate_name == "pyklu":
                klu_cls = solver

                def solver(matrix, rhs, _klu_cls=klu_cls):
                    return _klu_cls(as_solver_csc(matrix)).solve(rhs)

            OPTIONAL_SPARSE_SOLVERS[solver_name] = solver
            return solver

    OPTIONAL_SPARSE_MISSING.add(solver_name)
    return None


def resolve_linear_solver(solver_name):
    """Return (resolved_name, callable) for a requested linear solver."""
    name = str(solver_name or "scipy").strip().lower()
    if name in {"scipy", "superlu", "default"}:
        return "scipy", spsolve
    solver = load_named_sparse_solver(name)
    if solver is None:
        return "scipy", spsolve
    return name, solver


class CallableFactor:
    """Wrap a (matrix, rhs)->solution callable into a .solve(rhs) factor object."""

    __slots__ = ("_matrix", "_fn")

    def __init__(self, matrix, fn):
        self._matrix = matrix
        self._fn = fn

    def solve(self, rhs):
        return self._fn(self._matrix, rhs)


class ReusableUmfpackFactor:
    """Reuse UMFPACK symbolic analysis when repeated matrices share a pattern."""

    __slots__ = ("_ctx", "_matrix", "_umfpack")

    def __init__(self, matrix):
        self._umfpack = importlib.import_module("scikits.umfpack")
        matrix = as_solver_csc(matrix)
        family = self._family_for_matrix(matrix)
        self._ctx = self._umfpack.UmfpackContext(family)
        self._matrix = None
        self._ctx.symbolic(matrix)

    @staticmethod
    def _family_for_matrix(matrix):
        real = not np.iscomplexobj(matrix.data)
        int32_index = matrix.indices.dtype == np.int32 and matrix.indptr.dtype == np.int32
        if real:
            return "di" if int32_index else "dl"
        return "zi" if int32_index else "zl"

    def factor(self, matrix):
        matrix = as_solver_csc(matrix)
        self._ctx.numeric(matrix)
        self._matrix = matrix
        return self

    def solve(self, rhs):
        return self._ctx.solve(self._umfpack.UMFPACK_A, self._matrix, rhs, autoTranspose=True)


def make_reusable_factorizer(matrix, resolved_name):
    """Return a reusable factorizer for solvers that support fixed-pattern reuse."""
    if resolved_name != "umfpack":
        return None
    return ReusableUmfpackFactor(matrix)


def get_pyklu_cls():
    """Lazily import PyKLU.Klu and cache the class, or None when unavailable."""
    cache = OPTIONAL_SPARSE_SOLVERS
    if "__pyklu_cls__" in cache:
        return cache["__pyklu_cls__"]
    try:
        from PyKLU import Klu as klu_cls  # type: ignore
    except Exception:
        klu_cls = None
    cache["__pyklu_cls__"] = klu_cls
    return klu_cls


def factor_jacobian(matrix, resolved_name, solver_fn, *, permc_spec=None):
    """构建支持 `.solve(b)` 的稀疏矩阵分解器。"""
    if resolved_name in {"scipy", "superlu", "default"}:
        csc = as_solver_csc(matrix)
        if permc_spec is not None:
            return splu(csc, permc_spec=permc_spec)
        return splu(csc)
    if resolved_name in {"pyklu", "auto"}:
        klu_cls = get_pyklu_cls()
        if klu_cls is not None:
            return klu_cls(as_solver_csc(matrix))
    return CallableFactor(matrix, solver_fn)


class AdaptiveNewtonFactor:
    """牛顿-拉夫逊潮流循环的安全自适应 Shamanskii LU 因子复用器。

    当残差较小且保持快速单调收缩时，复用已缓存的 LU 因子执行低开销回代；
    当收缩变慢、达到最大复用次数，或复用步未能降低残差范数时，自动回滚并重新分解。
    """

    __slots__ = (
        "_factor",
        "_reuses",
        "_prev_norm",
        "_last_was_reuse",
        "_x_before_reuse",
        "_norm_before_reuse",
        "_reuse_threshold",
        "_contract_ratio",
        "_max_reuses",
        "_fast_tail_max_reuses",
        "_fast_tail_ratio",
        "_min_size",
    )

    def __init__(
        self,
        *,
        reuse_threshold: float = 0.10,
        contract_ratio: float = 0.50,
        max_reuses: int = 6,
        fast_tail_max_reuses: int = 10,
        fast_tail_ratio: float = 0.25,
        min_size: int = 128,
    ) -> None:
        self._factor = None
        self._reuses = 0
        self._prev_norm = np.inf
        self._last_was_reuse = False
        self._x_before_reuse = None
        self._norm_before_reuse = np.inf
        self._reuse_threshold = float(reuse_threshold)
        self._contract_ratio = float(contract_ratio)
        self._max_reuses = int(max_reuses)
        self._fast_tail_max_reuses = max(int(max_reuses), int(fast_tail_max_reuses))
        self._fast_tail_ratio = float(fast_tail_ratio)
        self._min_size = int(min_size)

    def reset(self) -> None:
        self._factor = None
        self._reuses = 0
        self._prev_norm = np.inf
        self._last_was_reuse = False
        self._x_before_reuse = None
        self._norm_before_reuse = np.inf

    def check_rollback(self, norm_f: float):
        """Return backup x to restore if the previous reused step failed to reduce norm_f."""
        if self._last_was_reuse and (not np.isfinite(norm_f) or norm_f >= self._norm_before_reuse):
            backup = self._x_before_reuse
            self.reset()
            return backup
        return None

    def can_reuse(self, norm_f: float, n_vars: int) -> bool:
        if self._factor is None or n_vars < self._min_size:
            return False
        if not np.isfinite(norm_f) or norm_f >= self._reuse_threshold:
            return False
        if self._reuses < self._max_reuses:
            return norm_f < self._contract_ratio * self._prev_norm
        # 达到常规上限后，仅在尾部保持高速收缩时继续复用，避免在临近收敛时重做整次 LU 分解
        if self._reuses < self._fast_tail_max_reuses:
            return norm_f < self._fast_tail_ratio * self._prev_norm
        return False

    def solve_reused(self, rhs: np.ndarray, x_current: np.ndarray, norm_f: float) -> np.ndarray:
        self._x_before_reuse = x_current.copy()
        self._norm_before_reuse = float(norm_f)
        self._last_was_reuse = True
        self._reuses += 1
        self._prev_norm = float(norm_f)
        return self._factor.solve(rhs)

    def update_factor(self, factor, norm_f: float) -> None:
        self._factor = None if isinstance(factor, CallableFactor) else factor
        self._reuses = 0
        self._prev_norm = float(norm_f)
        self._last_was_reuse = False
        self._x_before_reuse = None
        self._norm_before_reuse = np.inf
