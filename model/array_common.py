from pathlib import Path
import math
import operator
from typing import Dict, Tuple

import numpy as np

from paths import resolve_project_file

_P_UNIT_TO_SCALE = {"W": 1000.0, "kW": 1.0, "MW": 0.001}
_U_UNIT_TO_SCALE = {"mV": 1_000_000.0, "V": 1000.0, "kV": 1.0}
_I_UNIT_TO_SCALE = {"mA": 1_000_000.0, "A": 1000.0, "kA": 1.0}


def file_cache_key(file_path) -> Tuple[Path, int, int]:
    path = resolve_project_file(file_path).resolve()
    stat = path.stat()
    return path, stat.st_mtime_ns, stat.st_size


def _empty(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.float64)


def _optional_ppc_vector(ppc, key: str, size: int, default=np.nan) -> np.ndarray:
    """Return an optional PPC vector padded to the requested device count."""
    values = np.asarray((ppc or {}).get(key, ()), dtype=np.float64).reshape(-1)
    result = np.full(int(size), float(default), dtype=np.float64)
    if values.size:
        result[: min(result.size, values.size)] = values[: result.size]
    return result


class EFileTableRows:
    """List-compatible E table rows with cached column access."""

    __slots__ = ("rows", "columns", "_matrix", "_raw_cache")

    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self._matrix = None
        self._raw_cache = {}

    def __len__(self):
        return len(self.rows)

    def __bool__(self):
        return bool(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, item):
        return self.rows[item]

    def _as_matrix(self):
        if self._matrix is not None:
            return self._matrix
        if not self.rows:
            self._matrix = np.empty((0, 0), dtype=object)
            return self._matrix
        matrix = np.asarray(self.rows, dtype=object)
        self._matrix = matrix if matrix.ndim == 2 else False
        return self._matrix

    def float_column(self, col: int, default: float = 0.0) -> np.ndarray:
        n = len(self.rows)
        if col is None or n == 0:
            return np.full(n, float(default), dtype=np.float64) if n else np.empty(0, dtype=np.float64)
        col_idx = int(col)
        col_tokens = self.rows.column_tokens(col_idx) if hasattr(self.rows, "column_tokens") else None
        try:
            if col_tokens is not None:
                return np.fromiter(map(float, col_tokens), dtype=np.float64, count=n)
            return np.fromiter(
                map(float, map(operator.itemgetter(col_idx), self.rows)),
                dtype=np.float64,
                count=n,
            )
        except (IndexError, TypeError, ValueError):
            return self.raw_column(col, default).astype(np.float64, copy=False)

    def int_column(self, col: int, default: int = 0) -> np.ndarray:
        n = len(self.rows)
        if col is None or n == 0:
            return np.full(n, float(default), dtype=np.float64) if n else np.empty(0, dtype=np.float64)
        col_idx = int(col)
        col_tokens = self.rows.column_tokens(col_idx) if hasattr(self.rows, "column_tokens") else None
        try:
            if col_tokens is not None:
                return np.fromiter(map(int, col_tokens), dtype=np.float64, count=n)
            return np.fromiter(
                map(int, map(operator.itemgetter(col_idx), self.rows)),
                dtype=np.float64,
                count=n,
            )
        except (IndexError, TypeError, ValueError):
            return (
                self.raw_column(col, default)
                .astype(np.float64, copy=False)
                .astype(np.int64, copy=False)
                .astype(np.float64)
            )

    def flat_str_column(self, col):
        """当底层为批量扁平 token 视图时，直接返回非空字符串列 list，避免对象数组拷贝。"""
        n = len(self.rows)
        if col is None or n == 0:
            return None
        if hasattr(self.rows, "column_tokens"):
            return self.rows.column_tokens(int(col))
        return None

    def raw_column(self, col, default="") -> np.ndarray:
        n = len(self.rows)
        if col is None or n == 0:
            return np.full(n, default, dtype=object) if n else np.empty(0, dtype=object)
        key = (int(col), default)
        cached = self._raw_cache.get(key)
        if cached is not None:
            return cached

        col_idx = int(col)
        col_tokens = self.rows.column_tokens(col_idx) if hasattr(self.rows, "column_tokens") else None
        if col_tokens is not None:
            values = np.array(col_tokens, dtype=object)
            missing = (values == "") | (values == None)
            if np.any(missing):
                values[missing] = default
            self._raw_cache[key] = values
            return values
        try:
            values = np.array(list(map(operator.itemgetter(col_idx), self.rows)), dtype=object)
            missing = (values == "") | (values == None)
            if np.any(missing):
                values[missing] = default
        except IndexError:
            values = np.empty(n, dtype=object)
            for i, row in enumerate(self.rows):
                if col_idx >= len(row):
                    values[i] = default
                else:
                    value = row[col_idx]
                    values[i] = default if value in (None, "") else value
        self._raw_cache[key] = values
        return values


class LazyNameArray:
    """按需物化的序列兼容名称数组，支持扁平字符串列零拷贝快速路径。"""

    __slots__ = ("_raw_names", "_raw_list", "_idx_values", "_prefix", "_array")

    def __init__(self, raw_names, idx_values: np.ndarray, prefix: str, *, verified_str_list: bool = False):
        if verified_str_list and isinstance(raw_names, list):
            self._raw_list = raw_names
            self._raw_names = None
        else:
            self._raw_list = None
            self._raw_names = None if raw_names is None else np.asarray(raw_names, dtype=object).copy()
        self._idx_values = np.asarray(idx_values)
        self._prefix = str(prefix)
        self._array = None

    def __len__(self):
        return int(self._idx_values.shape[0])

    def _materialize(self) -> np.ndarray:
        if self._array is not None:
            return self._array
        if self._raw_list is not None:
            names = np.asarray(self._raw_list, dtype=object)
        elif self._raw_names is None:
            names = np.asarray([f"{self._prefix}_{int(idx)}" for idx in self._idx_values.tolist()], dtype=object)
        else:
            raw = self._raw_names
            prefix = self._prefix
            names = np.asarray(
                [
                    str(v) if v not in (None, "") else f"{prefix}_{int(idx)}"
                    for v, idx in zip(raw.tolist(), self._idx_values.tolist())
                ],
                dtype=object,
            )
        self._array = names
        return names

    def __getitem__(self, item):
        return self._materialize()[item]

    def __iter__(self):
        if self._raw_list is not None and self._array is None:
            return iter(self._raw_list)
        return iter(self._materialize())

    def __array__(self, dtype=None, copy=None):
        array = self._materialize()
        if dtype is not None:
            return array.astype(dtype, copy=bool(copy) if copy is not None else False)
        return array.copy() if copy else array

    @property
    def shape(self):
        return (len(self),)

    @property
    def dtype(self):
        return np.dtype(object)

    def astype(self, dtype, copy=True):
        return self._materialize().astype(dtype, copy=copy)

    def tolist(self):
        if self._raw_list is not None and self._array is None:
            return list(self._raw_list)
        return self._materialize().tolist()

    def raw_str_list(self):
        """返回只读底层字符串列表（若已就绪），供内部只读装配避免额外拷贝。"""
        if self._raw_list is not None and self._array is None:
            return self._raw_list
        return None


def _rows_for(data: Dict, table_name: str):
    table = data.get(table_name)
    if not table:
        return {}, []
    columns = {str(name): pos for pos, name in enumerate(table.get("header_list", []))}
    return columns, EFileTableRows(table.get("rows", []), columns)


def _cell(row, col, default=""):
    if col is None or col >= len(row):
        return default
    value = row[col]
    return default if value in (None, "") else value


def _float_cell(row, col, default: float = 0.0) -> float:
    return float(_cell(row, col, default))


def _int_cell(row, col, default: int = 0) -> int:
    return int(float(_cell(row, col, default)))


def _float_column(table_rows, columns, attr: str, default: float = 0.0) -> np.ndarray:
    col = columns.get(attr)
    n = len(table_rows)
    if col is None or n == 0:
        return np.full(n, float(default), dtype=np.float64) if n else np.empty(0, dtype=np.float64)
    if hasattr(table_rows, "float_column"):
        return table_rows.float_column(col, default)
    if hasattr(table_rows, "raw_column"):
        return table_rows.raw_column(col, default).astype(np.float64, copy=False)
    values = [None] * n
    for i in range(n):
        row = table_rows[i]
        if col >= len(row):
            values[i] = default
        else:
            v = row[col]
            values[i] = default if v in (None, "") else float(v)
    return np.asarray(values, dtype=np.float64)


def _int_column(table_rows, columns, attr: str, default: int = 0) -> np.ndarray:
    col = columns.get(attr)
    n = len(table_rows)
    if col is None or n == 0:
        return np.full(n, float(default), dtype=np.float64) if n else np.empty(0, dtype=np.float64)
    if hasattr(table_rows, "int_column"):
        return table_rows.int_column(col, default)
    if hasattr(table_rows, "raw_column"):
        return (
            table_rows.raw_column(col, default)
            .astype(np.float64, copy=False)
            .astype(np.int64, copy=False)
            .astype(np.float64)
        )
    values = [None] * n
    for i in range(n):
        row = table_rows[i]
        if col >= len(row):
            values[i] = default
        else:
            v = row[col]
            values[i] = default if v in (None, "") else int(float(v))
    return np.asarray(values, dtype=np.float64)


def _code_value(value, mapping: Dict[str, int], default_label: str) -> int:
    if value in (None, ""):
        return mapping[default_label]
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return mapping.get(str(value).upper(), mapping[default_label])


def _code_column(table_rows, columns, attr: str, mapping: Dict[str, int], default_label: str) -> np.ndarray:
    col = columns.get(attr)
    n = len(table_rows)
    default = mapping[default_label]
    if col is None or n == 0:
        return np.full(n, float(default), dtype=np.float64) if n else np.empty(0, dtype=np.float64)
    if hasattr(table_rows, "flat_str_column"):
        flat_tokens = table_rows.flat_str_column(col)
        if flat_tokens is not None:
            # 仅对列中极少数唯一字符串做一次归一化映射，避免数万次重复 .upper() 与类型判断
            val_map = {tok: float(_code_value(tok, mapping, default_label)) for tok in dict.fromkeys(flat_tokens)}
            return np.fromiter(map(val_map.__getitem__, flat_tokens), dtype=np.float64, count=n)
    if hasattr(table_rows, "raw_column"):
        raw = table_rows.raw_column(col, default_label)
        raw_list = raw.tolist()
        val_map = {tok: float(_code_value(tok, mapping, default_label)) for tok in dict.fromkeys(raw_list)}
        return np.fromiter(map(val_map.__getitem__, raw_list), dtype=np.float64, count=n)
    return np.asarray(
        [_code_value(_cell(row, col, default_label), mapping, default_label) for row in table_rows],
        dtype=np.float64,
    )


def _names_from_rows(table_rows, columns, prefix: str, idx_values: np.ndarray) -> np.ndarray:
    name_col = columns.get("name")
    if name_col is None:
        if hasattr(table_rows, "raw_column"):
            return LazyNameArray(None, idx_values, prefix)
        return np.asarray([f"{prefix}_{int(idx)}" for idx in idx_values], dtype=object)
    if hasattr(table_rows, "flat_str_column"):
        flat_names = table_rows.flat_str_column(name_col)
        if flat_names is not None:
            return LazyNameArray(flat_names, idx_values, prefix, verified_str_list=True)
    if hasattr(table_rows, "raw_column"):
        raw = table_rows.raw_column(name_col, "")
        return LazyNameArray(raw, idx_values, prefix)
    return np.asarray(
        [
            str(_cell(row, name_col, "") or f"{prefix}_{int(idx_values[pos])}")
            for pos, row in enumerate(table_rows)
        ],
        dtype=object,
    )


def _base_from_rows(data: Dict) -> Tuple[float, float, float, float, float]:
    block_name = "Model"
    columns, table_rows = _rows_for(data, block_name)
    if not table_rows:
        block_name = "PowerBase"
        columns, table_rows = _rows_for(data, block_name)
    if not table_rows:
        raise RuntimeError("E file must define <Model> with p_base, u_unit, p_unit, and i_unit")
    row = table_rows[0]
    if "p_base" not in columns:
        raise RuntimeError(f"E file <{block_name}> must define p_base")
    p_base = float(_cell(row, columns["p_base"], 0.0))
    if p_base <= 0.0:
        raise RuntimeError(f"Invalid p_base in <{block_name}>: {p_base}")

    def unit_or_legacy_scale(unit_attr: str, scale_attr: str, mapping, default_unit: str) -> float:
        if unit_attr in columns:
            unit = str(_cell(row, columns[unit_attr], default_unit)).strip()
            if unit not in mapping:
                allowed = "/".join(mapping.keys())
                raise RuntimeError(f"Invalid {unit_attr} in <{block_name}>: {unit!r}; expected one of {allowed}")
            return float(mapping[unit])
        if scale_attr not in columns:
            raise RuntimeError(f"E file <{block_name}> must define p_base, u_unit, p_unit, and i_unit")
        value = float(_cell(row, columns[scale_attr], 0.0))
        if value <= 0.0:
            raise RuntimeError(f"Invalid {scale_attr} in <{block_name}>: {value}")
        return value

    u_scale = unit_or_legacy_scale("u_unit", "u_scale", _U_UNIT_TO_SCALE, "kV")
    p_scale = unit_or_legacy_scale("p_unit", "p_scale", _P_UNIT_TO_SCALE, "kW")
    i_scale = unit_or_legacy_scale("i_unit", "i_scale", _I_UNIT_TO_SCALE, "kA")
    return p_base, u_scale, p_scale, i_scale, p_base / p_scale


def _scale_by_node(node_values: np.ndarray, scales_by_idx: Dict[int, float]) -> np.ndarray:
    nodes = np.asarray(node_values, dtype=np.int64)
    n = int(nodes.size)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    getter = scales_by_idx.get
    return np.fromiter((getter(node, 1.0) for node in nodes.tolist()), dtype=np.float64, count=n)


def _current_scale_by_node(node_ids, voltage_bases, power_base_kW, i_scale, *, ac=False):
    """Compute unit bases by column, preserving scalar operation order.

    The dictionary is local to input conversion, including last-row-wins for
    duplicate IDs. No network or result is retained across calls.
    """
    voltage = np.abs(np.asarray(voltage_bases, dtype=np.float64))
    factor = 1000.0 * math.sqrt(3.0) if ac else 1000.0
    current = np.divide(
        power_base_kW, factor * voltage, out=np.ones_like(voltage),
        where=~(voltage <= 1e-12),
    )
    current *= i_scale
    return dict(zip(map(int, np.asarray(node_ids).tolist()), current.tolist()))


class _DenseNodeFloatLookup:
    """稠密节点 ID 到浮点值的向量化查表结构，兼容器字典 .get() 接口。"""

    __slots__ = ("_values", "_present", "_default")

    def __init__(self, values: np.ndarray, present: np.ndarray, default: float = 0.0):
        self._values = values
        self._present = present
        self._default = float(default)

    def get(self, node_id, default=None):
        fallback = self._default if default is None else default
        idx = int(node_id)
        if 0 <= idx < self._values.size and self._present[idx]:
            return float(self._values[idx])
        return fallback

    def lookup_array(self, node_values: np.ndarray, default: float = 0.0) -> np.ndarray:
        nodes = np.asarray(node_values, dtype=np.int64)
        n = int(nodes.size)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        out = np.full(n, float(default), dtype=np.float64)
        in_range = (nodes >= 0) & (nodes < self._values.size)
        if np.any(in_range):
            valid_nodes = nodes[in_range]
            present = self._present[valid_nodes]
            if np.all(present):
                out[in_range] = self._values[valid_nodes]
            elif np.any(present):
                hit_pos = np.flatnonzero(in_range)[present]
                out[hit_pos] = self._values[valid_nodes[present]]
        return out


def _build_node_float_lookup(node_ids: np.ndarray, values: np.ndarray, default: float = 0.0):
    """构建节点 ID 到浮点基准值的快速查找表（保留重复 ID 后行覆盖语义）。"""
    ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(ids.size)
    if n == 0:
        return _DenseNodeFloatLookup(np.empty(0, dtype=np.float64), np.empty(0, dtype=bool), default)
    min_id = int(ids.min())
    max_id = int(ids.max())
    if min_id >= 0 and max_id <= max(4 * n, 65536):
        dense = np.full(max_id + 1, float(default), dtype=np.float64)
        present = np.zeros(max_id + 1, dtype=bool)
        dense[ids] = vals
        present[ids] = True
        return _DenseNodeFloatLookup(dense, present, default)
    return dict(zip(ids.tolist(), vals.tolist()))


def _raw_vbase_by_node(node_values: np.ndarray, raw_vbase_by_idx) -> np.ndarray:
    if isinstance(raw_vbase_by_idx, _DenseNodeFloatLookup):
        return raw_vbase_by_idx.lookup_array(node_values, 0.0)
    nodes = np.asarray(node_values, dtype=np.int64)
    n = int(nodes.size)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    getter = raw_vbase_by_idx.get
    return np.fromiter((getter(node, 0.0) for node in nodes.tolist()), dtype=np.float64, count=n)


def _assign_power_if_present(out: np.ndarray, col: int, table_rows, columns, attr: str, p_base: float) -> None:
    if attr in columns:
        out[:, col] = _float_column(table_rows, columns, attr) / p_base


def _assign_current_if_present(
    out: np.ndarray,
    col: int,
    table_rows,
    columns,
    attr: str,
    node_values: np.ndarray,
    current_scale_by_node: Dict[int, float],
) -> None:
    if attr not in columns:
        return
    if callable(current_scale_by_node):
        current_scale_by_node = current_scale_by_node()
    scales = _scale_by_node(node_values.astype(np.int64, copy=False), current_scale_by_node)
    raw = _float_column(table_rows, columns, attr)
    out[:, col] = np.divide(raw, scales, out=np.zeros_like(raw), where=np.abs(scales) > 1e-12)


def _voltage_set_column(table_rows, columns, attr: str, node_values: np.ndarray, raw_vbase_by_idx: Dict[int, float]) -> np.ndarray:
    if attr not in columns:
        return np.ones(len(table_rows), dtype=np.float64)
    raw = _float_column(table_rows, columns, attr, 1.0)
    raw_vbase = _raw_vbase_by_node(node_values.astype(np.int64, copy=False), raw_vbase_by_idx)
    return np.divide(raw, raw_vbase, out=np.ones_like(raw), where=np.abs(raw_vbase) > 1e-12)


def _value(obj, attr: str, default=0.0):
    value = getattr(obj, attr, default)
    return default if value in (None, "") else value


def _float_value(obj, attr: str, default: float = 0.0) -> float:
    return float(_value(obj, attr, default))


def _int_value(obj, attr: str, default: int = 0) -> int:
    return int(float(_value(obj, attr, default)))


def _has_value(devices, attr: str) -> bool:
    return any(getattr(dev, attr, None) not in (None, "") for dev in devices)


def _fill_float_column_if_present(out: np.ndarray, devices, col: int, attr: str) -> None:
    if not _has_value(devices, attr):
        return
    for row, dev in enumerate(devices):
        out[row, col] = _float_value(dev, attr)


def _name_array(devices, prefix: str) -> np.ndarray:
    return np.asarray(
        [str(getattr(dev, "name", "") or f"{prefix}_{_int_value(dev, 'idx', pos)}") for pos, dev in enumerate(devices)],
        dtype=object,
    )
