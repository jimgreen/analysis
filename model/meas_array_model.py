import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from efile_read import _split_data_row
from paths import resolve_project_file
from model.meas_type import DEVICE_TYPE_CODES, MEAS_TYPE_CODES, MEAS_TYPE_NAMES
from model.meas_model import (
    MEAS_STATUS_INVALID,
    MEAS_STATUS_NORMAL,
    MEAS_STATUS_PSEUDO,
    MEAS_STATUS_REMOVED,
    MeasurementTable,
    TableBackedMeasurementList,
    normalize_measurement_status,
    measurement_status_is_active,
)

try:
    from secore.native_kernels import parse_meas_bulk_bytes_native as _NATIVE_PARSE_MEAS_BULK_BYTES
except Exception:
    _NATIVE_PARSE_MEAS_BULK_BYTES = None


MEAS_COLS = {
    "idx": 0,
    "device_type_code": 1,
    "device_name_id": 2,
    "meas_type_code": 3,
    "weight": 4,
    "valid": 5,
    "value": 6,
    "status": 7,
    "angle_mask": 8,
    "source_row": 9,
}

_DEVICE_TYPE_CODES_BYTES = {name.encode("utf8"): int(code) for name, code in DEVICE_TYPE_CODES.items()}
_MEAS_TYPE_CODES_BYTES = {name.encode("utf8"): int(code) for name, code in MEAS_TYPE_CODES.items()}
_STATUS_BYTES_FAST_MAP = {
    b"0": MEAS_STATUS_NORMAL,
    b"NORMAL": MEAS_STATUS_NORMAL,
    b"1": MEAS_STATUS_INVALID,
    b"INVALID": MEAS_STATUS_INVALID,
    b"2": MEAS_STATUS_PSEUDO,
    b"PSEUDO": MEAS_STATUS_PSEUDO,
    b"3": MEAS_STATUS_REMOVED,
    b"REMOVED": MEAS_STATUS_REMOVED,
}
_BULK_CHUNK_BYTES = 262144
_REQUIRED_COLUMNS = ("idx", "name", "dev_type", "dev_name", "meas_type", "weight", "valid", "value")
_REQUIRED_COLUMNS_BYTES = tuple(name.encode("utf8") for name in _REQUIRED_COLUMNS)

# Parsed measurement files are immutable source data.  Estimator preparation
# still receives a private copy (see ``copy_meas_ppc``), so runtime status,
# values and topology-position arrays can never leak between estimators.
#
# Deliberately do not keep a process-wide file cache here.  State-estimation
# callers use a fresh network/measurement load per estimator; a global cache
# makes cold-start timings depend on an earlier request and can retain large
# PPC arrays for unrelated cases.  Reuse inside one estimator is handled by
# its table-backed measurement plan instead.
_MEAS_PARSE_CHUNK_SIZE = 16384


def _file_cache_key(file_path) -> Tuple[Path, int, int]:
    path = resolve_project_file(file_path).resolve()
    stat = path.stat()
    return path, int(stat.st_mtime_ns), int(stat.st_size)


def _device_type_codes_from_names(device_type_values: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    device_type_code_values = np.zeros(device_type_values.shape[0], dtype=np.int16)
    rows_by_code: Dict[int, np.ndarray] = {}
    for device_type, code in DEVICE_TYPE_CODES.items():
        rows = np.flatnonzero(device_type_values == device_type)
        if rows.size:
            code_int = int(code)
            device_type_code_values[rows] = code_int
            rows_by_code[code_int] = rows.astype(np.int64, copy=False)
    unknown_rows = np.flatnonzero(device_type_code_values == 0)
    if unknown_rows.size:
        rows_by_code[0] = unknown_rows.astype(np.int64, copy=False)
    return device_type_code_values, rows_by_code


def _meas_type_codes_from_names(meas_type_values: np.ndarray) -> np.ndarray:
    if meas_type_values.size == 0:
        return np.empty(0, dtype=np.int16)
    code_get = MEAS_TYPE_CODES.get
    return np.asarray([code_get(str(name).upper(), 0) for name in meas_type_values.tolist()], dtype=np.int16)


def _encode_device_names(device_name_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if device_name_values.size == 0:
        return np.empty(0, dtype=np.int32), np.asarray([], dtype=object)
    ids = np.empty(int(device_name_values.size), dtype=np.int32)
    names = []
    lookup = {}
    for pos, name in enumerate(device_name_values.astype(object, copy=False)):
        name_id = lookup.get(name)
        if name_id is None:
            name_id = len(names)
            lookup[name] = name_id
            names.append(name)
        ids[pos] = name_id
    return ids, np.asarray(names, dtype=object)


def _build_ppc(
    *,
    source: Path,
    idx: np.ndarray,
    name: Optional[np.ndarray],
    device_type: Optional[np.ndarray],
    device_name: Optional[np.ndarray],
    meas_type: Optional[np.ndarray],
    weight: np.ndarray,
    valid: np.ndarray,
    value: np.ndarray,
    status: np.ndarray,
    include_strings: bool = True,
    rows_by_device_type_code: Optional[Dict[int, np.ndarray]] = None,
    device_type_code: Optional[np.ndarray] = None,
    meas_type_code: Optional[np.ndarray] = None,
    device_name_id: Optional[np.ndarray] = None,
    device_names: Optional[np.ndarray] = None,
    device_name_id_by_name: Optional[Dict[object, int]] = None,
    device_name_id_by_bytes: Optional[Dict[bytes, int]] = None,
    angle_mask: Optional[np.ndarray] = None,
    include_matrix: bool = True,
) -> Dict:
    count = int(idx.size)
    if device_type_code is None:
        if device_type is None:
            raise RuntimeError("device_type_code is required when device_type strings are omitted")
        device_type_code, computed_rows_by_code = _device_type_codes_from_names(device_type)
    else:
        device_type_code = np.asarray(device_type_code, dtype=np.int16)
        computed_rows_by_code = None
    if rows_by_device_type_code is None:
        if computed_rows_by_code is None:
            rows_by_device_type_code = {
                int(code): np.flatnonzero(device_type_code == int(code)).astype(np.int64, copy=False)
                for code in np.unique(device_type_code)
            }
        else:
            rows_by_device_type_code = computed_rows_by_code
    if meas_type_code is None:
        if meas_type is None:
            raise RuntimeError("meas_type_code is required when meas_type strings are omitted")
        meas_type_code = _meas_type_codes_from_names(meas_type)
    else:
        meas_type_code = np.asarray(meas_type_code, dtype=np.int16)
    if device_name_id is None or device_names is None:
        if device_name is None:
            raise RuntimeError("device_name_id and device_names are required when device_name strings are omitted")
        device_name_id, device_names = _encode_device_names(device_name)
    else:
        device_name_id = np.asarray(device_name_id, dtype=np.int32)
        device_names = np.asarray(device_names, dtype=object)
    weight = np.asarray(weight, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    value = np.asarray(value, dtype=np.float64)
    status = np.asarray(status, dtype=np.int16)
    if include_strings:
        name = np.asarray(name if name is not None else (), dtype=object)
        device_type = np.asarray(device_type if device_type is not None else (), dtype=object)
        device_name = np.asarray(device_name if device_name is not None else (), dtype=object)
        meas_type = np.asarray(meas_type if meas_type is not None else (), dtype=object)
        if device_name.size != count and device_name_id.size == count and device_names.size:
            device_name = device_names[device_name_id.astype(np.intp, copy=False)]
    else:
        name = np.asarray([], dtype=object)
        device_type = np.asarray([], dtype=object)
        device_name = np.asarray([], dtype=object)
        meas_type = np.asarray([], dtype=object)
    if device_name_id_by_name is None and (include_strings or device_name_id_by_bytes is None):
        device_name_id_by_name = dict(
            zip(device_names.astype(object, copy=False).tolist(), range(int(device_names.size)))
        )
    if angle_mask is None:
        angle_mask = np.zeros(count, dtype=bool)
    if include_matrix:
        meas = np.zeros((count, len(MEAS_COLS)), dtype=np.float64, order="F")
        if count:
            meas[:, MEAS_COLS["idx"]] = idx.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["device_type_code"]] = device_type_code.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["device_name_id"]] = device_name_id.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["meas_type_code"]] = meas_type_code.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["weight"]] = weight
            meas[:, MEAS_COLS["valid"]] = valid.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["value"]] = value
            meas[:, MEAS_COLS["status"]] = status.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["angle_mask"]] = angle_mask.astype(np.float64, copy=False)
            meas[:, MEAS_COLS["source_row"]] = np.arange(count, dtype=np.float64)
    else:
        meas = None
    return {
        "format": "meas_ppc_v1",
        "source": str(source),
        "meas": meas,
        "meas_cols": MEAS_COLS,
        "meas_type_codes": MEAS_TYPE_CODES,
        "meas_type_names": MEAS_TYPE_NAMES,
        "device_type_codes": DEVICE_TYPE_CODES,
        "device_type_names": {code: name for name, code in DEVICE_TYPE_CODES.items()},
        "idx_array": idx,
        "weight_array": weight,
        "valid_array": valid,
        "value_array": value,
        "status_array": status,
        "device_type_code_array": device_type_code,
        "device_name_id_array": device_name_id,
        "meas_type_code_array": meas_type_code,
        "angle_mask_array": angle_mask,
        "name": name,
        "device_type": device_type,
        "device_name": device_name,
        "device_names": device_names,
        "device_name_id_by_name": device_name_id_by_name,
        "device_name_id_by_bytes": device_name_id_by_bytes,
        "meas_type": meas_type,
        "rows_by_device_type_code": rows_by_device_type_code,
        "normalized": False,
    }


def attach_device_pos_from_name_arrays(
    ppc: Dict,
    name_arrays_by_type_code: Dict[int, np.ndarray],
) -> np.ndarray:
    """Attach per-row device_pos to a meas PPC using typed PPC name arrays.

    The parser keeps device names compacted as ``device_names`` plus
    ``device_name_id_array``.  SE runtimes should consume the final
    ``device_pos`` array instead of rebuilding name-id lookup maps.
    """
    idx = ppc.get("idx_array")
    meas = ppc.get("meas")
    if isinstance(idx, np.ndarray):
        row_count = int(idx.size)
    elif isinstance(meas, np.ndarray) and meas.ndim == 2:
        row_count = int(meas.shape[0])
    else:
        row_count = 0
    existing = ppc.get("device_pos")
    if isinstance(existing, np.ndarray) and int(existing.size) == row_count:
        device_pos = existing.astype(np.int64, copy=True)
    else:
        device_pos = np.full(row_count, -1, dtype=np.int64)
    if row_count == 0:
        ppc["device_pos"] = device_pos
        return device_pos

    device_name_id = ppc.get("device_name_id_array")
    device_names = ppc.get("device_names")
    rows_by_code = ppc.get("rows_by_device_type_code")
    if not isinstance(device_name_id, np.ndarray) or int(device_name_id.size) != row_count:
        ppc["device_pos"] = device_pos
        return device_pos
    if not isinstance(device_names, np.ndarray) or int(device_names.size) == 0:
        ppc["device_pos"] = device_pos
        return device_pos
    if not isinstance(rows_by_code, dict):
        ppc["device_pos"] = device_pos
        return device_pos

    device_name_id = np.asarray(device_name_id, dtype=np.int64)
    device_names = np.asarray(device_names, dtype=object)
    device_name_id_by_name = ppc.get("device_name_id_by_name")
    device_name_id_by_bytes = ppc.get("device_name_id_by_bytes")
    lookup_size = int(device_names.size)
    shared_lookup: Optional[np.ndarray] = None
    get_name_id = device_name_id_by_name.get if isinstance(device_name_id_by_name, dict) else None
    get_bytes_id = device_name_id_by_bytes.get if isinstance(device_name_id_by_bytes, dict) else None
    for code, plan_names in name_arrays_by_type_code.items():
        rows = rows_by_code.get(int(code))
        if rows is None:
            continue
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            continue
        if int(rows.min()) < 0 or int(rows.max()) >= row_count:
            rows = rows[(rows >= 0) & (rows < row_count)]
            if rows.size == 0:
                continue
        rows_idx = rows.astype(np.intp, copy=False)
        unresolved = device_pos[rows_idx] < 0
        if not np.any(unresolved):
            continue
        if not np.all(unresolved):
            rows = rows[unresolved]
            rows_idx = rows.astype(np.intp, copy=False)

        # 优先使用零解码 bytes 名称直连（模型侧与量测侧均免去 UTF-8 解码与二次哈希）
        active_getter = None
        query_list = None
        if get_bytes_id is not None and hasattr(plan_names, "raw_bytes_list"):
            raw_bytes = plan_names.raw_bytes_list()
            if raw_bytes is not None:
                active_getter = get_bytes_id
                query_list = raw_bytes
        if active_getter is None:
            if get_name_id is None and get_bytes_id is not None:
                device_name_id_by_name = dict(
                    zip(device_names.astype(object, copy=False).tolist(), range(lookup_size))
                )
                ppc["device_name_id_by_name"] = device_name_id_by_name
                get_name_id = device_name_id_by_name.get
            if get_name_id is not None:
                if isinstance(plan_names, (list, tuple)):
                    query_list = plan_names
                elif hasattr(plan_names, "raw_str_list"):
                    raw_list = plan_names.raw_str_list()
                    query_list = raw_list if raw_list is not None else plan_names.tolist()
                elif hasattr(plan_names, "tolist"):
                    query_list = plan_names.tolist()
                else:
                    query_list = list(plan_names)
                active_getter = get_name_id

        if active_getter is not None:
            if not query_list:
                continue
            ids = device_name_id[rows_idx]
            all_ids_valid = int(ids.min()) >= 0 and int(ids.max()) < lookup_size
            if all_ids_valid:
                query_ids = ids.astype(np.intp, copy=False)
            else:
                valid_ids = (ids >= 0) & (ids < lookup_size)
                if not np.any(valid_ids):
                    continue
                valid_row_pos = np.flatnonzero(valid_ids)
                query_ids = ids[valid_ids].astype(np.intp, copy=False)
            if shared_lookup is None:
                shared_lookup = np.full(lookup_size, -1, dtype=np.int64)

            mapped = list(map(active_getter, query_list))
            n_q = len(mapped)
            if None not in mapped:
                touched_ids_arr = np.asarray(mapped, dtype=np.intp)
                if touched_ids_arr.size and int(touched_ids_arr.min()) >= 0 and int(touched_ids_arr.max()) < lookup_size:
                    shared_lookup[touched_ids_arr] = np.arange(n_q, dtype=np.int64)
                else:
                    valid_m = (touched_ids_arr >= 0) & (touched_ids_arr < lookup_size)
                    touched_ids_arr = touched_ids_arr[valid_m]
                    if touched_ids_arr.size == 0:
                        continue
                    shared_lookup[touched_ids_arr] = np.flatnonzero(valid_m).astype(np.int64, copy=False)
            else:
                touched_ids = []
                touched_pos = []
                append_id = touched_ids.append
                append_pos = touched_pos.append
                for pos, name_id in enumerate(mapped):
                    if name_id is None and active_getter is get_name_id:
                        item = query_list[pos]
                        if not isinstance(item, str):
                            name_id = active_getter(str(item))
                    if name_id is not None and 0 <= name_id < lookup_size:
                        append_id(name_id)
                        append_pos(pos)
                if not touched_ids:
                    continue
                touched_ids_arr = np.asarray(touched_ids, dtype=np.intp)
                shared_lookup[touched_ids_arr] = np.asarray(touched_pos, dtype=np.int64)

            target_pos = shared_lookup[query_ids]
            shared_lookup[touched_ids_arr] = -1
            matched = target_pos >= 0
            if np.any(matched):
                if all_ids_valid:
                    if np.all(matched):
                        device_pos[rows_idx] = target_pos
                    else:
                        device_pos[rows_idx[matched]] = target_pos[matched]
                else:
                    target_rows = rows[valid_row_pos[matched]].astype(np.intp, copy=False)
                    device_pos[target_rows] = target_pos[matched]
            continue
        names = np.asarray(plan_names, dtype=object)
        if names.size == 0:
            continue
        ids = device_name_id[rows.astype(np.intp, copy=False)]
        valid_ids = (ids >= 0) & (ids < device_names.size)
        if not np.any(valid_ids):
            continue
        rows_valid = rows[valid_ids]
        query = device_names[ids[valid_ids].astype(np.intp, copy=False)]
        order = np.argsort(names, kind="stable")
        sorted_names = names[order.astype(np.intp, copy=False)]
        loc = np.searchsorted(sorted_names, query)
        in_range = loc < sorted_names.size
        if not np.any(in_range):
            continue
        valid_loc_idx = np.flatnonzero(in_range)
        matched = sorted_names[loc[valid_loc_idx].astype(np.intp, copy=False)] == query[valid_loc_idx]
        if np.any(matched):
            target_rows = rows_valid[valid_loc_idx[matched]].astype(np.intp, copy=False)
            target_pos = order[loc[valid_loc_idx[matched]].astype(np.intp, copy=False)].astype(np.int64, copy=False)
            device_pos[target_rows] = target_pos
    ppc["device_pos"] = device_pos
    return device_pos


def _empty_meas_ppc(source: Path, *, include_matrix: bool = True) -> Dict:
    return _build_ppc(
        source=source,
        idx=np.asarray([], dtype=np.int64),
        name=np.asarray([], dtype=object),
        device_type=np.asarray([], dtype=object),
        device_name=np.asarray([], dtype=object),
        meas_type=np.asarray([], dtype=object),
        weight=np.asarray([], dtype=np.float64),
        valid=np.asarray([], dtype=bool),
        value=np.asarray([], dtype=np.float64),
        status=np.asarray([], dtype=np.int16),
        rows_by_device_type_code={},
        device_type_code=np.asarray([], dtype=np.int16),
        meas_type_code=np.asarray([], dtype=np.int16),
        device_name_id=np.asarray([], dtype=np.int32),
        device_names=np.asarray([], dtype=object),
        include_matrix=include_matrix,
    )


def _build_meas_ppc_from_rows_dict(rows_dict: Dict, source: Path) -> Dict:
    block = rows_dict.get("Measurement") if isinstance(rows_dict, dict) else None
    if block is None:
        raise RuntimeError(f"{source} does not contain a <Measurement> block")
    header = tuple(block.get("header_list") or ())
    if not header:
        raise RuntimeError(f"{source} Measurement block does not contain a header")
    header_index = {name: idx for idx, name in enumerate(header)}
    missing = [name for name in _REQUIRED_COLUMNS if name not in header_index]
    if missing:
        raise RuntimeError(f"{source} Measurement header is missing columns: {missing}")
    rows = block.get("rows") or ()
    count = len(rows)
    if count == 0:
        return _empty_meas_ppc(source)
    idx_col = header_index["idx"]
    name_col = header_index["name"]
    device_type_col = header_index["dev_type"]
    device_name_col = header_index["dev_name"]
    meas_type_col = header_index["meas_type"]
    weight_col = header_index["weight"]
    valid_col = header_index["valid"]
    value_col = header_index["value"]
    status_col = header_index.get("status", -1)
    required_max_col = max(
        idx_col,
        name_col,
        device_type_col,
        device_name_col,
        meas_type_col,
        weight_col,
        valid_col,
        value_col,
    )
    allow_missing_trailing_value = value_col == required_max_col
    intern = sys.intern
    try:
        idx_values = np.fromiter((int(fields[idx_col]) for fields in rows), dtype=np.int64, count=count)
        name_values = np.asarray([fields[name_col] for fields in rows], dtype=object)
        device_type_list = [intern(fields[device_type_col]) for fields in rows]
        device_type_values = np.asarray(device_type_list, dtype=object)
        device_name_values = np.asarray([fields[device_name_col] for fields in rows], dtype=object)
        meas_type_list = [intern(fields[meas_type_col].upper()) for fields in rows]
        meas_type_values = np.asarray(meas_type_list, dtype=object)
        weight_values = np.fromiter((float(fields[weight_col]) for fields in rows), dtype=np.float64, count=count)
        valid_values = np.asarray(
            [
                (not (allow_missing_trailing_value and len(fields) == value_col))
                and (fields[valid_col] == "1" or fields[valid_col] == 1 or fields[valid_col] is True)
                for fields in rows
            ],
            dtype=bool,
        )
        value_values = np.fromiter(
            (
                np.nan
                if (allow_missing_trailing_value and len(fields) == value_col)
                else float(fields[value_col])
                for fields in rows
            ),
            dtype=np.float64,
            count=count,
        )
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed Measurement row in {source}") from exc
    if status_col >= 0:
        status_values = np.empty(count, dtype=np.int16)
        for row_pos, fields in enumerate(rows):
            status = (
                normalize_measurement_status(fields[status_col], valid=bool(valid_values[row_pos]))
                if len(fields) > status_col
                else MEAS_STATUS_NORMAL if valid_values[row_pos] else MEAS_STATUS_INVALID
            )
            if status != MEAS_STATUS_NORMAL:
                valid_values[row_pos] = False
            status_values[row_pos] = status
    else:
        status_values = np.where(valid_values, MEAS_STATUS_NORMAL, MEAS_STATUS_INVALID).astype(np.int16, copy=False)
    device_type_code_values = np.asarray(
        [DEVICE_TYPE_CODES.get(device_type, 0) for device_type in device_type_list],
        dtype=np.int16,
    )
    meas_type_code_values = np.asarray(
        [MEAS_TYPE_CODES.get(meas_type, 0) for meas_type in meas_type_list],
        dtype=np.int16,
    )
    rows_by_code = {
        int(code): np.flatnonzero(device_type_code_values == int(code)).astype(np.int64, copy=False)
        for code in np.unique(device_type_code_values)
    }
    device_name_id, device_names = _encode_device_names(device_name_values)
    return _build_ppc(
        source=source,
        idx=idx_values,
        name=name_values,
        device_type=device_type_values,
        device_name=device_name_values,
        meas_type=meas_type_values,
        weight=weight_values,
        valid=valid_values,
        value=value_values,
        status=status_values,
        rows_by_device_type_code=rows_by_code,
        device_type_code=device_type_code_values,
        meas_type_code=meas_type_code_values,
        device_name_id=device_name_id,
        device_names=device_names,
    )


def _normalize_status_text(status_text, valid: bool) -> int:
    if isinstance(status_text, str):
        text = status_text.strip().upper()
        if not text:
            return MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
        if text == "0":
            return MEAS_STATUS_NORMAL
        if text == "1":
            return MEAS_STATUS_INVALID
        if text == "2":
            return MEAS_STATUS_PSEUDO
        if text == "3":
            return MEAS_STATUS_REMOVED
    return normalize_measurement_status(status_text, valid=valid)


def _normalize_status_bytes(status_text: bytes, valid: bool) -> int:
    text = status_text.strip().upper()
    if not text:
        return MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
    if text == b"0":
        return MEAS_STATUS_NORMAL
    if text == b"1":
        return MEAS_STATUS_INVALID
    if text == b"2":
        return MEAS_STATUS_PSEUDO
    if text == b"3":
        return MEAS_STATUS_REMOVED
    return _normalize_status_text(text.decode("utf8"), valid)


def _measurement_block_name(line: str) -> Optional[str]:
    if not line.startswith("<") or not line.endswith(">"):
        return None
    name = line[1:-1].strip()
    if name.startswith("/"):
        name = name[1:].strip()
    if " lv " in name:
        name = name.split(" lv ", 1)[0].strip()
    return name


def _measurement_block_name_bytes(line: bytes) -> Optional[bytes]:
    if not line.startswith(b"<") or not line.endswith(b">"):
        return None
    name = line[1:-1].strip()
    if name.startswith(b"/"):
        name = name[1:].strip()
    marker = b" lv "
    if marker in name:
        name = name.split(marker, 1)[0].strip()
    return name


def _initial_measurement_capacity(source: Path) -> int:
    """Estimate a growable row capacity without scanning the measurement file twice."""
    try:
        return max(1024, int(source.stat().st_size // 96) + 128)
    except OSError:
        return 1024


def _grow_measurement_arrays(arrays: Tuple[np.ndarray, ...], new_size: int) -> Tuple[np.ndarray, ...]:
    grown = []
    for array in arrays:
        array.resize(new_size, refcheck=False)
        grown.append(array)
    return tuple(grown)


def _rows_by_device_type_code_fast(device_type_code_array: np.ndarray) -> Dict[int, np.ndarray]:
    """一次分组构造按设备类型码划分的行号索引，避免多类型时反复全表扫描。"""
    n = int(device_type_code_array.size)
    if n == 0:
        return {}
    order = np.argsort(device_type_code_array, kind="stable").astype(np.int64, copy=False)
    sorted_codes = device_type_code_array[order]
    change = np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1
    starts = np.empty(change.size + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = change
    stops = np.empty(change.size + 1, dtype=np.int64)
    stops[:-1] = change
    stops[-1] = n
    return {
        int(sorted_codes[int(s)]): order[int(s) : int(e)]
        for s, e in zip(starts.tolist(), stops.tolist())
    }


def _try_parse_meas_ppc_bulk_bytes(
    source: Path,
    *,
    include_strings: bool = False,
    include_matrix: bool = True,
) -> Optional[Dict]:
    """标准无引号 <Measurement> 块的分块高速字节批量解析器（支持数组模式、全字符串模式与 status 列）。

    按约 256 KB 分块切分并即时释放临时 bytes token，保持工作集在 L2/L3 Cache 内，
    同时触发块级常量短路。遇到引号、块内空行/注释、未知 status 文本或列宽不一致时
    返回 None 自动回退逐行解析器。
    """
    try:
        raw = source.read_bytes()
    except OSError:
        return None
    if b"'" in raw:
        return None
    start_tag = raw.find(b"<Measurement>")
    if start_tag < 0:
        return None
    end_tag = raw.find(b"</Measurement>", start_tag + 13)
    if end_tag < 0:
        return None
    body = raw[start_tag + 13 : end_tag]
    at_pos = body.find(b"@")
    if at_pos < 0:
        return None
    header_end = body.find(b"\n", at_pos)
    if header_end < 0:
        return None
    header = tuple(body[at_pos + 1 : header_end].split())
    if not header:
        return None
    header_index = {name: idx for idx, name in enumerate(header)}
    if any(name not in header_index for name in _REQUIRED_COLUMNS_BYTES):
        return None
    if include_strings and b"name" not in header_index:
        return None
    data_bytes = body[header_end + 1 :].strip()
    del raw, body
    if not data_bytes:
        return _empty_meas_ppc(source, include_matrix=include_matrix)
    if b"\n\n" in data_bytes or b"\r\n\r\n" in data_bytes or b"//" in data_bytes:
        return None

    n_rows = data_bytes.count(b"\n") + 1
    ncols = len(header) + 1
    idx_pos = header_index[b"idx"] + 1
    dev_type_pos = header_index[b"dev_type"] + 1
    dev_name_pos = header_index[b"dev_name"] + 1
    meas_type_pos = header_index[b"meas_type"] + 1
    weight_pos = header_index[b"weight"] + 1
    valid_pos = header_index[b"valid"] + 1
    value_pos = header_index[b"value"] + 1
    status_pos = header_index[b"status"] + 1 if b"status" in header_index else -1
    name_col_pos = header_index[b"name"] + 1 if include_strings else -1

    idx_array = np.empty(n_rows, dtype=np.int64)
    weight_values = np.empty(n_rows, dtype=np.float64)
    value_values = np.empty(n_rows, dtype=np.float64)
    valid_values = np.empty(n_rows, dtype=bool)
    status_values = np.empty(n_rows, dtype=np.int16)
    device_type_code_array = np.empty(n_rows, dtype=np.int16)
    meas_type_code_array = np.empty(n_rows, dtype=np.int16)
    device_name_id = np.empty(n_rows, dtype=np.int32)

    dt_code_map = dict(_DEVICE_TYPE_CODES_BYTES)
    mt_code_map = dict(_MEAS_TYPE_CODES_BYTES)
    name_to_id: Dict[bytes, int] = {}
    unique_name_bytes = []

    # 非字符串数组模式优先调用 C++17 单遍 std::from_chars 高速解析核，失败自动回退下方分块路径
    native_unique_names = None
    if not include_strings and _NATIVE_PARSE_MEAS_BULK_BYTES is not None:
        native_unique_names = _NATIVE_PARSE_MEAS_BULK_BYTES(
            data_bytes,
            ncols,
            idx_pos,
            dev_type_pos,
            dev_name_pos,
            meas_type_pos,
            weight_pos,
            valid_pos,
            value_pos,
            status_pos,
            int(MEAS_STATUS_NORMAL),
            int(MEAS_STATUS_INVALID),
            int(MEAS_STATUS_REMOVED),
            dt_code_map,
            mt_code_map,
            _STATUS_BYTES_FAST_MAP,
            idx_array,
            weight_values,
            value_values,
            valid_values,
            status_values,
            device_type_code_array,
            meas_type_code_array,
            device_name_id,
        )

    if native_unique_names is not None:
        if status_pos >= 0:
            non_normal = status_values != MEAS_STATUS_NORMAL
            if np.any(non_normal):
                valid_values &= ~non_normal
        uniq_bytes_list = native_unique_names.split(b"\n") if native_unique_names else []
        name_to_id = dict(zip(uniq_bytes_list, range(len(uniq_bytes_list))))
        device_names_list = native_unique_names.decode("utf8").split("\n") if native_unique_names else []
        device_names_arr = np.asarray(device_names_list, dtype=object)
        name_arr = device_type_arr = device_name_arr = meas_type_arr = None
        device_name_id_by_name = None
    else:
        if include_strings:
            name_chunks = []
            dt_local_id_arr = np.empty(n_rows, dtype=np.int32)
            mt_local_id_arr = np.empty(n_rows, dtype=np.int32)
            dt_tok_to_local: Dict[bytes, int] = {}
            dt_local_tokens = []
            mt_tok_to_local: Dict[bytes, int] = {}
            mt_local_tokens = []

        data_len = len(data_bytes)
        pos = 0
        row_off = 0
        chunk_size = _BULK_CHUNK_BYTES
        while pos < data_len:
            end = pos + chunk_size
            if end < data_len:
                nl = data_bytes.find(b"\n", end)
                end = data_len if nl < 0 else nl
            else:
                end = data_len
            chunk = data_bytes[pos:end]
            pos = end + 1
            if not chunk:
                continue
            c_rows = chunk.count(b"\n") + 1
            tokens = chunk.split()
            if len(tokens) != c_rows * ncols or tokens[0::ncols].count(b"#") != c_rows:
                return None
            row_end = row_off + c_rows
            if row_end > n_rows:
                return None

            try:
                idx_array[row_off:row_end] = np.fromiter(
                    map(int, tokens[idx_pos::ncols]), dtype=np.int64, count=c_rows
                )
                weight_toks = tokens[weight_pos::ncols]
                first_weight = weight_toks[0]
                if weight_toks.count(first_weight) == c_rows:
                    weight_values[row_off:row_end] = float(first_weight)
                else:
                    weight_values[row_off:row_end] = np.fromiter(
                        map(float, weight_toks), dtype=np.float64, count=c_rows
                    )
                value_values[row_off:row_end] = np.fromiter(
                    map(float, tokens[value_pos::ncols]), dtype=np.float64, count=c_rows
                )
            except ValueError:
                return None

            valid_toks = tokens[valid_pos::ncols]
            all_valid_chunk = valid_toks.count(b"1") == c_rows
            if all_valid_chunk:
                valid_values[row_off:row_end] = True
            else:
                valid_values[row_off:row_end] = np.fromiter(
                    map(b"1".__eq__, valid_toks), dtype=bool, count=c_rows
                )

            if status_pos >= 0:
                st_toks = tokens[status_pos::ncols]
                st0 = st_toks[0]
                sc0 = _STATUS_BYTES_FAST_MAP.get(st0)
                if sc0 is not None and st_toks.count(st0) == c_rows:
                    status_values[row_off:row_end] = sc0
                    if sc0 != MEAS_STATUS_NORMAL:
                        valid_values[row_off:row_end] = False
                else:
                    st_local_map = {}
                    for tok in dict.fromkeys(st_toks):
                        code = _STATUS_BYTES_FAST_MAP.get(tok)
                        if code is None:
                            code = _STATUS_BYTES_FAST_MAP.get(tok.upper())
                        if code is None:
                            return None
                        st_local_map[tok] = code
                    sc_arr = np.fromiter(map(st_local_map.__getitem__, st_toks), dtype=np.int16, count=c_rows)
                    status_values[row_off:row_end] = sc_arr
                    inactive = sc_arr != MEAS_STATUS_NORMAL
                    if np.any(inactive):
                        valid_values[row_off:row_end] &= ~inactive
            else:
                if all_valid_chunk:
                    status_values[row_off:row_end] = MEAS_STATUS_NORMAL
                else:
                    status_values[row_off:row_end] = np.where(
                        valid_values[row_off:row_end], MEAS_STATUS_NORMAL, MEAS_STATUS_INVALID
                    )

            dev_toks = tokens[dev_type_pos::ncols]
            dt0 = dev_toks[0]
            if dev_toks.count(dt0) == c_rows:
                dt_c0 = dt_code_map.get(dt0, 0)
                device_type_code_array[row_off:row_end] = dt_c0
                if include_strings:
                    lid = dt_tok_to_local.get(dt0)
                    if lid is None:
                        lid = len(dt_local_tokens)
                        dt_tok_to_local[dt0] = lid
                        dt_local_tokens.append(dt0)
                    dt_local_id_arr[row_off:row_end] = lid
            else:
                dt_uniq_chunk = list(dict.fromkeys(dev_toks))
                for tok in dt_uniq_chunk:
                    if tok not in dt_code_map:
                        dt_code_map[tok] = 0
                device_type_code_array[row_off:row_end] = np.fromiter(
                    map(dt_code_map.__getitem__, dev_toks), dtype=np.int16, count=c_rows
                )
                if include_strings:
                    for tok in dt_uniq_chunk:
                        if tok not in dt_tok_to_local:
                            dt_tok_to_local[tok] = len(dt_local_tokens)
                            dt_local_tokens.append(tok)
                    dt_local_id_arr[row_off:row_end] = np.fromiter(
                        map(dt_tok_to_local.__getitem__, dev_toks), dtype=np.int32, count=c_rows
                    )

            meas_toks = tokens[meas_type_pos::ncols]
            mt_uniq_chunk = list(dict.fromkeys(meas_toks))
            for tok in mt_uniq_chunk:
                code = mt_code_map.get(tok)
                if code is None:
                    code = _MEAS_TYPE_CODES_BYTES.get(tok.upper(), 0)
                    mt_code_map[tok] = code
            meas_type_code_array[row_off:row_end] = np.fromiter(
                map(mt_code_map.__getitem__, meas_toks), dtype=np.int16, count=c_rows
            )
            if include_strings:
                for tok in mt_uniq_chunk:
                    if tok not in mt_tok_to_local:
                        mt_tok_to_local[tok] = len(mt_local_tokens)
                        mt_local_tokens.append(tok)
                mt_local_id_arr[row_off:row_end] = np.fromiter(
                    map(mt_tok_to_local.__getitem__, meas_toks), dtype=np.int32, count=c_rows
                )

            name_toks = tokens[dev_name_pos::ncols]
            new_names = [tok for tok in dict.fromkeys(name_toks) if tok not in name_to_id]
            if new_names:
                base_id = len(unique_name_bytes)
                unique_name_bytes.extend(new_names)
                name_to_id.update(zip(new_names, range(base_id, base_id + len(new_names))))
            device_name_id[row_off:row_end] = np.fromiter(
                map(name_to_id.__getitem__, name_toks), dtype=np.int32, count=c_rows
            )

            if include_strings:
                name_chunks.append(b"\n".join(tokens[name_col_pos::ncols]))

            row_off = row_end

        if row_off != n_rows:
            return None

        # 仅解码唯一设备名（经 bytes.split 切分必定不含换行符，可整块一次性 UTF-8 解码）
        device_names_list = b"\n".join(unique_name_bytes).decode("utf8").split("\n")
        device_names_arr = np.asarray(device_names_list, dtype=object)

        if include_strings:
            name_arr = np.asarray(b"\n".join(name_chunks).decode("utf8").split("\n"), dtype=object)
            dt_labels = np.asarray([tok.decode("utf8") for tok in dt_local_tokens], dtype=object)
            device_type_arr = dt_labels[dt_local_id_arr]
            device_name_arr = device_names_arr[device_name_id]
            mt_labels = np.asarray(
                [
                    tok.decode("utf8") if _MEAS_TYPE_CODES_BYTES.get(tok, 0) != 0 else tok.decode("utf8").upper()
                    for tok in mt_local_tokens
                ],
                dtype=object,
            )
            meas_type_arr = mt_labels[mt_local_id_arr]
            device_name_id_by_name = dict(zip(device_names_list, range(len(device_names_list))))
        else:
            name_arr = device_type_arr = device_name_arr = meas_type_arr = None
            device_name_id_by_name = None

    # 若按设备类型已分块连续，直接 arange 切片免去 argsort
    change = np.flatnonzero(device_type_code_array[1:] != device_type_code_array[:-1]) + 1
    if change.size == 0:
        rows_by_code = {int(device_type_code_array[0]): np.arange(n_rows, dtype=np.int64)}
    else:
        starts = np.empty(change.size + 1, dtype=np.int64)
        starts[0] = 0
        starts[1:] = change
        block_codes = device_type_code_array[starts].tolist()
        if len(set(block_codes)) == len(block_codes):
            stops = np.empty(change.size + 1, dtype=np.int64)
            stops[:-1] = change
            stops[-1] = n_rows
            rows_by_code = {
                int(code): np.arange(int(s0), int(e0), dtype=np.int64)
                for code, s0, e0 in zip(block_codes, starts.tolist(), stops.tolist())
            }
        else:
            rows_by_code = _rows_by_device_type_code_fast(device_type_code_array)

    angle_mask = np.zeros(n_rows, dtype=bool)
    return _build_ppc(
        source=source,
        idx=idx_array,
        name=name_arr,
        device_type=device_type_arr,
        device_name=device_name_arr,
        meas_type=meas_type_arr,
        weight=weight_values,
        valid=valid_values,
        value=value_values,
        status=status_values,
        include_strings=include_strings,
        rows_by_device_type_code=rows_by_code,
        device_type_code=device_type_code_array,
        meas_type_code=meas_type_code_array,
        device_name_id=device_name_id,
        device_names=device_names_arr,
        device_name_id_by_name=device_name_id_by_name,
        device_name_id_by_bytes=name_to_id,
        angle_mask=angle_mask,
        include_matrix=include_matrix,
    )


def _build_meas_ppc_from_measurement_file_array_only(
    source: Path,
    *,
    include_matrix: bool = True,
) -> Dict:
    """Parse numeric columns in bounded batches during one byte scan.

    Converting whole columns avoids Python float/int conversion followed by a
    NumPy scalar write for every row. Batches bound temporary Python objects;
    no parsed network, measurement result, or previous solve is reused here.
    """
    bulk_result = _try_parse_meas_ppc_bulk_bytes(source, include_matrix=include_matrix)
    if bulk_result is not None:
        return bulk_result
    header = ()
    idx_col = device_type_col = device_name_col = meas_type_col = -1
    weight_col = valid_col = value_col = -1
    status_col = -1
    required_max_col = -1
    split_limit = -1
    raw_split_limit = -1
    raw_idx_col = raw_device_type_col = raw_device_name_col = raw_meas_type_col = -1
    raw_weight_col = raw_valid_col = raw_value_col = raw_status_col = -1
    standard_raw_no_status = False
    has_status = False
    idx_values, weight_values, valid_values, value_values = [], [], [], []
    status_values, device_type_code_values, meas_type_code_values, device_name_id = [], [], [], []
    columns = (idx_values, weight_values, valid_values, value_values, status_values,
               device_type_code_values, meas_type_code_values, device_name_id)
    dtypes = (np.int64, np.float64, bool, np.float64, np.int16, np.int16, np.int16, np.int32)
    capacity = _initial_measurement_capacity(source)
    arrays = tuple(np.empty(capacity, dtype=dtype) for dtype in dtypes)
    written = 0
    chunk_size = max(1, int(_MEAS_PARSE_CHUNK_SIZE))
    next_flush = chunk_size

    def flush_columns():
        nonlocal capacity, written
        if not idx_values:
            return
        end = written + len(idx_values)
        if end > capacity:
            capacity = max(2 * capacity, end)
            _grow_measurement_arrays(arrays, capacity)
        for array, column in zip(arrays, columns):
            array[written:end] = column
            column.clear()
        written = end
    row_count = 0
    device_names_list = []
    device_name_lookup = {}

    intern = sys.intern
    device_code_get = _DEVICE_TYPE_CODES_BYTES.get
    meas_code_get = _MEAS_TYPE_CODES_BYTES.get
    device_name_lookup_get = device_name_lookup.get
    device_names_list_append = device_names_list.append
    status_is_active = measurement_status_is_active
    last_device_type = None
    last_device_type_code = 0
    last_device_name = None
    last_device_name_id = -1
    last_meas_type = None
    last_meas_type_code = 0
    in_measurement = False

    try:
        with open(source, mode="rb") as fp:
            for line_no, raw_line in enumerate(fp, start=1):
                first = raw_line[:1]
                if first == b"#" and in_measurement:
                    text = None
                else:
                    line = raw_line.strip()
                    if not line:
                        continue
                    first = line[:1]
                    if first == b"<":
                        block_name = _measurement_block_name_bytes(line)
                        if not line.startswith(b"</") and block_name == b"Measurement":
                            in_measurement = True
                            header = ()
                        elif line.startswith(b"</") and block_name == b"Measurement" and in_measurement:
                            break
                        elif in_measurement and line.startswith(b"</"):
                            break
                        continue
                    if first == b"@" and in_measurement:
                        header = tuple(line[1:].split())
                        header_index = {name: idx for idx, name in enumerate(header)}
                        missing = [name.decode("utf8") for name in _REQUIRED_COLUMNS_BYTES if name not in header_index]
                        if missing:
                            raise RuntimeError(f"{source} Measurement header is missing columns: {missing}")
                        idx_col = header_index[b"idx"]
                        device_type_col = header_index[b"dev_type"]
                        device_name_col = header_index[b"dev_name"]
                        meas_type_col = header_index[b"meas_type"]
                        weight_col = header_index[b"weight"]
                        valid_col = header_index[b"valid"]
                        value_col = header_index[b"value"]
                        status_col = header_index.get(b"status", -1)
                        has_status = status_col >= 0
                        required_max_col = max(
                            idx_col,
                            device_type_col,
                            device_name_col,
                            meas_type_col,
                            weight_col,
                            valid_col,
                            value_col,
                        )
                        split_limit = status_col + 1 if has_status else required_max_col
                        raw_split_limit = split_limit + 1
                        raw_idx_col = idx_col + 1
                        raw_device_type_col = device_type_col + 1
                        raw_device_name_col = device_name_col + 1
                        raw_meas_type_col = meas_type_col + 1
                        raw_weight_col = weight_col + 1
                        raw_valid_col = valid_col + 1
                        raw_value_col = value_col + 1
                        raw_status_col = status_col + 1
                        standard_raw_no_status = (
                            not has_status
                            and idx_col == 0
                            and device_type_col == 2
                            and device_name_col == 3
                            and meas_type_col == 4
                            and weight_col == 5
                            and valid_col == 6
                            and value_col == 7
                        )
                        continue
                    if first != b"#" or not in_measurement:
                        continue
                    text = line[1:]

                if not header:
                    raise RuntimeError(f"{source} Measurement data appears before header at line {line_no}")
                if standard_raw_no_status and first == b"#" and raw_line[:1] == b"#":
                    fields = raw_line.split(None, 8)
                    if len(fields) < 8:
                        raise RuntimeError(f"Malformed Measurement row at line {line_no} in {source}")
                    if row_count >= next_flush:
                        flush_columns()
                        next_flush += chunk_size

                    idx_values.append(fields[1])
                    device_type = fields[3]
                    device_type_code_values.append(device_code_get(device_type, 0))

                    device_name = fields[4]
                    if device_name == last_device_name:
                        name_id = last_device_name_id
                    else:
                        name_id = device_name_lookup_get(device_name)
                        if name_id is None:
                            device_name_text = intern(device_name.decode("utf8"))
                            name_id = len(device_names_list)
                            device_name_lookup[device_name] = name_id
                            device_names_list_append(device_name_text)
                        last_device_name = device_name
                        last_device_name_id = name_id
                    device_name_id.append(name_id)

                    meas_type = fields[5]
                    meas_code = meas_code_get(meas_type, 0)
                    if meas_code == 0:
                        meas_code = meas_code_get(meas_type.upper(), 0)
                    meas_type_code_values.append(meas_code)

                    weight_field = fields[6]
                    weight_values.append(1.0 if weight_field == b"1.0" or weight_field == b"1" else weight_field)
                    if len(fields) == 8:
                        value_values.append(np.nan)
                        valid_values.append(False)
                        status_values.append(MEAS_STATUS_INVALID)
                    else:
                        valid = fields[7] == b"1"
                        value_values.append(fields[8])
                        valid_values.append(valid)
                        status_values.append(MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID)
                    row_count += 1
                    continue
                if first == b"#" and raw_line[:1] == b"#":
                    fields = raw_line.split(None, raw_split_limit)
                    idx_pos = raw_idx_col
                    device_type_pos = raw_device_type_col
                    device_name_pos = raw_device_name_col
                    meas_type_pos = raw_meas_type_col
                    weight_pos = raw_weight_col
                    valid_pos = raw_valid_col
                    value_pos = raw_value_col
                    status_pos = raw_status_col
                    required_len = required_max_col + 1
                else:
                    if text is None:
                        text = raw_line[1:]
                    fields = text.split(None, split_limit)
                    idx_pos = idx_col
                    device_type_pos = device_type_col
                    device_name_pos = device_name_col
                    meas_type_pos = meas_type_col
                    weight_pos = weight_col
                    valid_pos = valid_col
                    value_pos = value_col
                    status_pos = status_col
                    required_len = required_max_col
                missing_trailing_value = value_col == required_max_col and len(fields) == value_pos
                if len(fields) <= required_len and not missing_trailing_value:
                    raise RuntimeError(f"Malformed Measurement row at line {line_no} in {source}")
                if row_count >= next_flush:
                    flush_columns()
                    next_flush += chunk_size

                idx_values.append(fields[idx_pos])
                device_type = fields[device_type_pos]
                if device_type == last_device_type:
                    device_code = last_device_type_code
                else:
                    device_code = device_code_get(device_type, 0)
                    last_device_type = device_type
                    last_device_type_code = device_code
                device_type_code_values.append(device_code)
                device_name = fields[device_name_pos]
                if device_name == last_device_name:
                    name_id = last_device_name_id
                else:
                    name_id = device_name_lookup_get(device_name)
                    if name_id is None:
                        device_name_text = intern(device_name.decode("utf8"))
                        name_id = len(device_names_list)
                        device_name_lookup[device_name] = name_id
                        device_names_list_append(device_name_text)
                    last_device_name = device_name
                    last_device_name_id = name_id
                device_name_id.append(name_id)
                meas_type = fields[meas_type_pos]
                if meas_type == last_meas_type:
                    meas_code = last_meas_type_code
                else:
                    meas_code = meas_code_get(meas_type, 0)
                    if meas_code == 0:
                        meas_code = meas_code_get(meas_type.upper(), 0)
                    last_meas_type = meas_type
                    last_meas_type_code = meas_code
                meas_type_code_values.append(meas_code)
                weight_field = fields[weight_pos]
                weight_values.append(1.0 if weight_field == b"1.0" or weight_field == b"1" else weight_field)
                if missing_trailing_value:
                    valid = False
                    value_values.append(np.nan)
                    status = MEAS_STATUS_INVALID
                else:
                    valid = fields[valid_pos] == b"1"
                    value_values.append(fields[value_pos])
                    if has_status and len(fields) > status_pos:
                        status = _normalize_status_bytes(fields[status_pos], valid)
                        if status != MEAS_STATUS_NORMAL:
                            valid = False
                    else:
                        status = MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
                valid_values.append(valid)
                status_values.append(status)
                row_count += 1
            flush_columns()
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed Measurement row in {source}") from exc

    if not header:
        raise RuntimeError(f"{source} does not contain a <Measurement> block")
    if row_count == 0:
        return _empty_meas_ppc(source, include_matrix=include_matrix)
    (idx_array, weight_values, valid_values, value_values, status_values,
     device_type_code_array, meas_type_code_array, device_name_id) = tuple(
         array[:row_count] for array in arrays
     )
    rows_by_code = {
        int(code): np.flatnonzero(device_type_code_array == int(code)).astype(np.int64, copy=False)
        for code in np.unique(device_type_code_array)
    }
    device_name_id_by_name = dict(zip(device_names_list, range(len(device_names_list))))
    return _build_ppc(
        source=source,
        idx=idx_array,
        name=None,
        device_type=None,
        device_name=None,
        meas_type=None,
        weight=weight_values,
        valid=valid_values,
        value=value_values,
        status=status_values,
        include_strings=False,
        rows_by_device_type_code=rows_by_code,
        device_type_code=device_type_code_array,
        meas_type_code=meas_type_code_array,
        device_name_id=device_name_id,
        device_names=np.asarray(device_names_list, dtype=object),
        device_name_id_by_name=device_name_id_by_name,
        include_matrix=include_matrix,
    )


def _build_meas_ppc_from_measurement_file(
    source: Path,
    *,
    include_strings: bool = True,
    include_matrix: bool = True,
) -> Dict:
    if not include_strings:
        return _build_meas_ppc_from_measurement_file_array_only(source, include_matrix=include_matrix)
    bulk_result = _try_parse_meas_ppc_bulk_bytes(
        source,
        include_strings=True,
        include_matrix=include_matrix,
    )
    if bulk_result is not None:
        return bulk_result
    header = ()
    idx_col = name_col = device_type_col = device_name_col = meas_type_col = -1
    weight_col = valid_col = value_col = -1
    status_col = -1
    required_max_col = -1
    idx_values = []
    name_values = [] if include_strings else None
    device_type_values = [] if include_strings else None
    device_name_values = [] if include_strings else None
    meas_type_values = [] if include_strings else None
    weight_values = []
    valid_values = []
    value_values = []
    status_values = []
    device_type_code_values = []
    meas_type_code_values = []
    device_name_id = []
    device_names_list = []
    device_name_lookup = {}

    intern = sys.intern
    to_int = int
    to_float = float
    split_data_row = _split_data_row
    device_code_get = DEVICE_TYPE_CODES.get
    meas_code_get = MEAS_TYPE_CODES.get
    idx_append = idx_values.append
    name_append = name_values.append if include_strings else None
    device_type_append = device_type_values.append if include_strings else None
    device_name_append = device_name_values.append if include_strings else None
    meas_type_append = meas_type_values.append if include_strings else None
    weight_append = weight_values.append
    valid_append = valid_values.append
    value_append = value_values.append
    status_append = status_values.append
    device_type_code_append = device_type_code_values.append
    meas_type_code_append = meas_type_code_values.append
    device_name_id_append = device_name_id.append
    device_name_lookup_get = device_name_lookup.get
    device_names_list_append = device_names_list.append
    status_is_active = measurement_status_is_active
    in_measurement = False

    try:
        with open(source, mode="rt", encoding="utf8") as fp:
            for line_no, raw_line in enumerate(fp, start=1):
                first = raw_line[0] if raw_line else ""
                if first == "#" and in_measurement:
                    text = raw_line[1:]
                else:
                    line = raw_line.strip()
                    if not line:
                        continue
                    first = line[0]
                    if first == "<":
                        block_name = _measurement_block_name(line)
                        if not line.startswith("</") and block_name == "Measurement":
                            in_measurement = True
                            header = ()
                        elif line.startswith("</") and block_name == "Measurement" and in_measurement:
                            break
                        elif in_measurement and line.startswith("</"):
                            break
                        continue
                    if first == "@" and in_measurement:
                        header = tuple(line[1:].split())
                        header_index = {name: idx for idx, name in enumerate(header)}
                        missing = [name for name in _REQUIRED_COLUMNS if name not in header_index]
                        if missing:
                            raise RuntimeError(f"{source} Measurement header is missing columns: {missing}")
                        idx_col = header_index["idx"]
                        name_col = header_index["name"]
                        device_type_col = header_index["dev_type"]
                        device_name_col = header_index["dev_name"]
                        meas_type_col = header_index["meas_type"]
                        weight_col = header_index["weight"]
                        valid_col = header_index["valid"]
                        value_col = header_index["value"]
                        status_col = header_index.get("status", -1)
                        required_max_col = max(
                            idx_col,
                            name_col,
                            device_type_col,
                            device_name_col,
                            meas_type_col,
                            weight_col,
                            valid_col,
                            value_col,
                        )
                        continue
                    if first != "#" or not in_measurement:
                        continue
                    text = line[1:]

                fields = text.split() if "'" not in text else split_data_row(text)
                if not header:
                    raise RuntimeError(f"{source} Measurement data appears before header at line {line_no}")
                missing_trailing_value = value_col == required_max_col and len(fields) == value_col
                if len(fields) <= required_max_col and not missing_trailing_value:
                    raise RuntimeError(f"Malformed Measurement row at line {line_no} in {source}")
                idx_append(to_int(fields[idx_col]))
                if include_strings:
                    name_append(fields[name_col])
                device_type = fields[device_type_col]
                device_code = device_code_get(device_type, 0)
                if include_strings:
                    device_type_append(intern(device_type))
                device_name = fields[device_name_col]
                name_id = device_name_lookup_get(device_name)
                if name_id is None:
                    device_name = intern(device_name)
                    name_id = len(device_names_list)
                    device_name_lookup[device_name] = name_id
                    device_names_list_append(device_name)
                else:
                    device_name = device_names_list[name_id]
                if include_strings:
                    device_name_append(device_name)
                meas_type = fields[meas_type_col]
                meas_code = meas_code_get(meas_type, 0)
                if meas_code == 0:
                    meas_type = meas_type.upper()
                    meas_code = meas_code_get(meas_type, 0)
                if include_strings:
                    meas_type_append(intern(meas_type))
                weight_append(to_float(fields[weight_col]))
                if missing_trailing_value:
                    valid = False
                    value_append(np.nan)
                    status = MEAS_STATUS_INVALID
                else:
                    valid = fields[valid_col] == "1"
                    value_append(to_float(fields[value_col]))
                    if status_col >= 0 and len(fields) > status_col:
                        status = _normalize_status_text(fields[status_col], valid)
                        if status != MEAS_STATUS_NORMAL:
                            valid = False
                    else:
                        status = MEAS_STATUS_NORMAL if valid else MEAS_STATUS_INVALID
                valid_append(valid)
                status_append(status)
                device_type_code_append(device_code)
                meas_type_code_append(meas_code)
                device_name_id_append(name_id)
    except (IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed Measurement row in {source}") from exc

    if not header:
        raise RuntimeError(f"{source} does not contain a <Measurement> block")
    if not idx_values:
        return _empty_meas_ppc(source, include_matrix=include_matrix)
    idx_array = np.asarray(idx_values, dtype=np.int64)
    device_type_code_array = np.asarray(device_type_code_values, dtype=np.int16)
    meas_type_code_array = np.asarray(meas_type_code_values, dtype=np.int16)
    rows_by_code = {
        int(code): np.flatnonzero(device_type_code_array == int(code)).astype(np.int64, copy=False)
        for code in np.unique(device_type_code_array)
    }
    return _build_ppc(
        source=source,
        idx=idx_array,
        name=np.asarray(name_values, dtype=object) if include_strings else None,
        device_type=np.asarray(device_type_values, dtype=object) if include_strings else None,
        device_name=np.asarray(device_name_values, dtype=object) if include_strings else None,
        meas_type=np.asarray(meas_type_values, dtype=object) if include_strings else None,
        weight=np.asarray(weight_values, dtype=np.float64),
        valid=np.asarray(valid_values, dtype=bool),
        value=np.asarray(value_values, dtype=np.float64),
        status=np.asarray(status_values, dtype=np.int16),
        include_strings=include_strings,
        rows_by_device_type_code=rows_by_code,
        device_type_code=device_type_code_array,
        meas_type_code=meas_type_code_array,
        device_name_id=np.asarray(device_name_id, dtype=np.int32),
        device_names=np.asarray(device_names_list, dtype=object),
        device_name_id_by_name=device_name_lookup,
        include_matrix=include_matrix,
    )


def build_meas_ppc_from_efile_rows(file_path, rows) -> Dict:
    """Build measurement ppc from E rows that are already loaded in memory."""
    path = resolve_project_file(file_path).resolve()
    return _build_meas_ppc_from_rows_dict(rows, path)


def build_meas_ppc_from_e_file(
    file_path,
    *,
    include_strings: bool = True,
    include_matrix: bool = True,
) -> Dict:
    """Read a measurement file directly into a PPC-style NumPy dictionary."""
    source = resolve_project_file(file_path).resolve()
    include_strings = bool(include_strings)
    include_matrix = bool(include_matrix)
    # 全新解析结果已拥有独立数组，无需再次深拷贝
    return _build_meas_ppc_from_measurement_file(
        source,
        include_strings=include_strings,
        include_matrix=include_matrix,
    )


def copy_meas_ppc(ppc: Dict) -> Dict:
    """Return a PPC copy whose source and runtime arrays are independent."""
    copied = dict(ppc)
    for key in (
        "meas",
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
        "device_names",
        "meas_type",
    ):
        value = ppc.get(key)
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
    rows_by_code = ppc.get("rows_by_device_type_code")
    if isinstance(rows_by_code, dict):
        copied["rows_by_device_type_code"] = {
            code: rows.copy() if isinstance(rows, np.ndarray) else rows
            for code, rows in rows_by_code.items()
        }
    name_ids = ppc.get("device_name_id_by_name")
    if isinstance(name_ids, dict):
        copied["device_name_id_by_name"] = dict(name_ids)
    byte_name_ids = ppc.get("device_name_id_by_bytes")
    if isinstance(byte_name_ids, dict):
        copied["device_name_id_by_bytes"] = dict(byte_name_ids)
    return copied


def measurement_table_from_meas_ppc(ppc: Dict, *, include_strings: bool = True) -> MeasurementTable:
    meas = ppc.get("meas")
    cols = ppc.get("meas_cols", MEAS_COLS)
    has_meas = isinstance(meas, np.ndarray) and meas.ndim == 2
    idx_array = ppc.get("idx_array")
    if has_meas:
        row_count = int(meas.shape[0])
        if not isinstance(idx_array, np.ndarray) or int(idx_array.size) != row_count:
            idx_array = meas[:, cols["idx"]].astype(np.int64, copy=False)
    else:
        if not isinstance(idx_array, np.ndarray):
            raise RuntimeError("meas PPC requires idx_array when meas matrix is omitted")
        row_count = int(idx_array.size)
    device_type_code = ppc.get("device_type_code_array")
    if not isinstance(device_type_code, np.ndarray) or int(device_type_code.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires device_type_code_array when meas matrix is omitted")
        device_type_code = meas[:, cols["device_type_code"]].astype(np.int16, copy=False)
    angle_mask = ppc.get("angle_mask_array")
    if not isinstance(angle_mask, np.ndarray) or int(angle_mask.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires angle_mask_array when meas matrix is omitted")
        angle_mask = meas[:, cols["angle_mask"]].astype(bool, copy=False)
    device_name_id = ppc.get("device_name_id_array")
    if not isinstance(device_name_id, np.ndarray) or int(device_name_id.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires device_name_id_array when meas matrix is omitted")
        device_name_id = meas[:, cols["device_name_id"]].astype(np.int64, copy=False)
    meas_type_code = ppc.get("meas_type_code_array")
    if not isinstance(meas_type_code, np.ndarray) or int(meas_type_code.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires meas_type_code_array when meas matrix is omitted")
        meas_type_code = meas[:, cols["meas_type_code"]].astype(np.int16, copy=False)
    weight = ppc.get("weight_array")
    if not isinstance(weight, np.ndarray) or int(weight.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires weight_array when meas matrix is omitted")
        weight = meas[:, cols["weight"]]
    value = ppc.get("value_array")
    if not isinstance(value, np.ndarray) or int(value.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires value_array when meas matrix is omitted")
        value = meas[:, cols["value"]]
    valid = ppc.get("valid_array")
    if not isinstance(valid, np.ndarray) or int(valid.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires valid_array when meas matrix is omitted")
        valid = meas[:, cols["valid"]].astype(bool, copy=False)
    status = ppc.get("status_array")
    if not isinstance(status, np.ndarray) or int(status.size) != row_count:
        if not has_meas:
            raise RuntimeError("meas PPC requires status_array when meas matrix is omitted")
        status = meas[:, cols["status"]].astype(np.int16, copy=False)
    device_pos = ppc.get("device_pos")
    if isinstance(device_pos, np.ndarray) and int(device_pos.size) == row_count:
        device_pos = device_pos.astype(np.int64, copy=False)
    else:
        device_pos = None
    if include_strings:
        name = np.asarray(ppc.get("name", ()), dtype=object)
        device_type = np.asarray(ppc.get("device_type", ()), dtype=object)
        device_name = np.asarray(ppc.get("device_name", ()), dtype=object)
        meas_type = np.asarray(ppc.get("meas_type", ()), dtype=object)
    else:
        name = np.asarray([], dtype=object)
        device_type = np.asarray([], dtype=object)
        device_name = np.asarray([], dtype=object)
        meas_type = np.asarray([], dtype=object)
    table = MeasurementTable(
        idx=idx_array,
        name=name,
        device_type=device_type,
        device_name=device_name,
        meas_type=meas_type,
        weight=np.asarray(weight, dtype=np.float64),
        valid=np.asarray(valid, dtype=bool),
        value=np.asarray(value, dtype=np.float64),
        device_type_code=device_type_code.astype(np.int16, copy=False),
        angle_mask=angle_mask.astype(bool, copy=False),
        status_code=np.asarray(status, dtype=np.int16),
        rows_by_device_type_code=ppc.get("rows_by_device_type_code"),
        device_name_id=device_name_id.astype(np.int64, copy=False),
        meas_type_code=meas_type_code.astype(np.int16, copy=False),
        device_pos=device_pos,
        device_names=np.asarray(ppc.get("device_names", ()), dtype=object),
    )
    for key in ("scale", "from_pos", "to_pos", "available"):
        value = ppc.get(key)
        if isinstance(value, np.ndarray) and int(value.size) == row_count:
            setattr(table, key, value)
    return table


def measurement_list_from_meas_ppc(ppc: Dict) -> TableBackedMeasurementList:
    return TableBackedMeasurementList(
        measurement_table_from_meas_ppc(ppc),
        normalized=bool(ppc.get("normalized", False)),
    )


def sync_meas_ppc_from_measurement_table(ppc: Dict, table: MeasurementTable) -> None:
    meas = ppc.get("meas")
    cols = ppc.get("meas_cols", MEAS_COLS)
    if isinstance(meas, np.ndarray) and meas.ndim == 2 and meas.shape[0] == table.idx.size:
        meas[:, cols["weight"]] = np.asarray(table.weight, dtype=np.float64)
        meas[:, cols["valid"]] = np.asarray(table.valid, dtype=np.float64)
        meas[:, cols["value"]] = np.asarray(table.value, dtype=np.float64)
        meas[:, cols["status"]] = np.asarray(table.status_code, dtype=np.float64)
        meas[:, cols["angle_mask"]] = np.asarray(table.angle_mask, dtype=np.float64)
    ppc["weight_array"] = np.asarray(table.weight, dtype=np.float64)
    ppc["valid_array"] = np.asarray(table.valid, dtype=bool)
    ppc["value_array"] = np.asarray(table.value, dtype=np.float64)
    ppc["status_array"] = np.asarray(table.status_code, dtype=np.int16)
    ppc["normalized"] = bool(getattr(table, "normalized", ppc.get("normalized", False)))


def meas_ppc_active_mask(ppc: Dict) -> np.ndarray:
    meas = ppc.get("meas")
    if isinstance(meas, np.ndarray) and meas.ndim == 2:
        valid = meas[:, MEAS_COLS["valid"]] != 0.0
        weight = meas[:, MEAS_COLS["weight"]]
        status = meas[:, MEAS_COLS["status"]].astype(np.int16, copy=False)
    else:
        valid = np.asarray(ppc.get("valid_array", ()), dtype=bool)
        weight = np.asarray(ppc.get("weight_array", ()), dtype=np.float64)
        status = np.asarray(ppc.get("status_array", ()), dtype=np.int16)
    return (
        valid
        & (weight > 0.0)
        & ((status == MEAS_STATUS_NORMAL) | (status == MEAS_STATUS_PSEUDO))
    )
