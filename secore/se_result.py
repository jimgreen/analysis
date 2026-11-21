from dataclasses import asdict, dataclass, field
from itertools import repeat, starmap
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from model.meas_model import (
    MEAS_STATUS_PSEUDO,
    BadDataItem,
    EstimateResult,
    Measurement,
    MeasurementTable,
    is_pseudo_measurement,
    measurement_table_status_code,
)
from model.meas_type import DEVICE_TYPE_NAMES, MEAS_TYPE_NAMES


def normalize_seresult_result_mode(result_mode: str) -> str:
    mode = str(result_mode or "full").strip().lower()
    aliases = {
        "all": "full",
        "full": "full",
        "complete": "full",
        "array": "array",
        "arrays": "array",
        "ppc": "array",
        "summary": "summary",
        "brief": "summary",
        "minimal": "summary",
        "none": "none",
        "skip": "none",
        "raw": "none",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported SEResult result_mode: {result_mode!r}")
    return aliases[mode]


def build_seresult_summary(
    result: EstimateResult,
    *,
    bad_items: Optional[Sequence[BadDataItem]] = None,
    all_measurements: Optional[Iterable[Measurement]] = None,
) -> "SEResult":
    if not result.measurements and result.measurement_table is not None:
        all_table = getattr(all_measurements, "table", None)
        return build_seresult_summary_from_table(
            result,
            bad_items=bad_items,
            all_measurement_table=all_table,
        )
    se_result = SEResult()
    bad_items = list(bad_items or ())
    active_ids = {id(meas) for meas in result.measurements}
    prefiltered_count = 0
    if all_measurements is not None:
        for measurement in all_measurements:
            if id(measurement) not in active_ids and SEResult._prefiltered_reason(measurement):
                prefiltered_count += 1
    pseudo_count = sum(1 for measurement in result.measurements if SEResult._is_pseudo_measurement(measurement))
    bad_count = len(bad_items)
    normal_count = max(0, len(result.measurements) - pseudo_count - bad_count)
    obs = result.observability
    se_result.statistics = SEResult.StatisticsTable(
        converged=bool(result.converged),
        iterations=int(result.iterations),
        objective=float(result.objective),
        max_correction=float(result.max_correction),
        residual_inf=float(result.residual_inf),
        observable=bool(obs.observable),
        rank=int(obs.rank),
        state_count=int(obs.state_count),
        measurement_count=int(obs.measurement_count),
        deficiency=int(obs.deficiency),
        prefiltered_measurement_count=prefiltered_count,
        pseudo_measurement_count=pseudo_count,
        bad_data_count=bad_count,
        normal_measurement_count=normal_count,
    )
    return se_result


def _sanitize_e_text(text: str) -> str:
    """将字符串转换为 E 文件单个 token（空串转 '-'，含空白符时以 '_' 连接）。"""
    if not text:
        return "-"
    if " " not in text and "\t" not in text and "\n" not in text and "\r" not in text:
        return text
    return "_".join(text.split())


# 预先清洗静态设备类型与量测类型映射表，供大批量 E 文件导出直接查表
_SANITIZED_DEVICE_TYPE_NAMES = {code: _sanitize_e_text(name) for code, name in DEVICE_TYPE_NAMES.items()}
_SANITIZED_MEAS_TYPE_NAMES = {code: _sanitize_e_text(name) for code, name in MEAS_TYPE_NAMES.items()}


def _extract_table_columns_chunk(
    table: MeasurementTable,
    positions: np.ndarray,
    *,
    estimated: Optional[np.ndarray],
    residual: Optional[np.ndarray],
    normalized: Optional[np.ndarray],
):
    """按批抽取测量结果表各列 Python 列表，供行物化、to_dicts 与 E 文本导出共享。"""
    device_pos = getattr(table, "device_pos", None)
    meas_codes = getattr(table, "meas_type_code", None)
    max_pos = int(positions[-1])

    def text_column(values, fallback):
        if values.size > max_pos:
            selected = values[positions]
            if values.dtype.kind in "OUS":
                selected = selected.tolist()
                # 绝大多数元素已经是 str，避免不必要的逐个 str() 调用
                if all(type(x) is str for x in selected):
                    return selected
            return list(map(str, selected))
        return [str(values[p]) if p < values.size else fallback(int(p)) for p in positions.tolist()]

    def coded_text(values, codes, names_map):
        if values.size == 0 and codes is not None and len(codes) > max_pos:
            lookup = names_map.get
            return [lookup(code, "") for code in codes[positions].tolist()]
        return text_column(
            values,
            lambda p: names_map.get(int(codes[p]), "") if codes is not None and len(codes) > p else "",
        )

    def optional_column(values):
        if values is None or values.size == 0:
            return None
        if values.size > max_pos:
            return values[positions].tolist()
        n_val = int(values.size)
        return [float(values[p]) if p < n_val else None for p in positions.tolist()]

    def missing_device_name(p):
        return f"pos:{int(device_pos[p])}" if device_pos is not None and len(device_pos) > p and int(device_pos[p]) >= 0 else ""

    ids = table.idx[positions].astype(np.int64, copy=False).tolist()
    if table.name.size == 0:
        names = [f"m{idx}" for idx in ids]
    else:
        names = text_column(table.name, lambda p: f"m{int(table.idx[p])}")
    device_types = coded_text(table.device_type, table.device_type_code, DEVICE_TYPE_NAMES)
    if table.device_name.size == 0 and device_pos is not None and len(device_pos) > max_pos:
        device_names = [f"pos:{pos}" if pos >= 0 else "" for pos in device_pos[positions].tolist()]
    else:
        device_names = text_column(table.device_name, missing_device_name)
    meas_types = coded_text(table.meas_type, meas_codes, MEAS_TYPE_NAMES)
    weights = table.weight[positions].astype(np.float64, copy=False).tolist()
    valid = table.valid[positions].astype(bool, copy=False).tolist()
    values = table.value[positions].astype(np.float64, copy=False).tolist()
    estimates = optional_column(estimated)
    residuals = optional_column(residual)
    normalized_values = optional_column(normalized)
    return (
        ids,
        names,
        device_types,
        device_names,
        meas_types,
        weights,
        valid,
        values,
        estimates,
        residuals,
        normalized_values,
    )


class _ColumnarMeasurementBatch:
    """不可变列式测量结果视图，延迟物化 SEResult.MeasurementRow 对象。"""

    __slots__ = ("table", "positions", "estimated", "residual", "normalized", "source", "reasons")

    def __init__(
        self,
        table: MeasurementTable,
        positions: np.ndarray,
        *,
        estimated: Optional[np.ndarray] = None,
        residual: Optional[np.ndarray] = None,
        normalized: Optional[np.ndarray] = None,
        source: str = "",
        reasons: Optional[List[str]] = None,
    ) -> None:
        self.table = table
        self.positions = np.asarray(positions, dtype=np.intp)
        self.estimated = None if estimated is None else np.asarray(estimated, dtype=np.float64)
        self.residual = None if residual is None else np.asarray(residual, dtype=np.float64)
        self.normalized = None if normalized is None else np.asarray(normalized, dtype=np.float64)
        self.source = str(source)
        self.reasons = reasons


class _MeasurementResultTable:
    __slots__ = ("_rows", "_columnar")

    def __init__(self, rows: Optional[Iterable["SEResult.MeasurementRow"]] = None) -> None:
        self._rows: Optional[List["SEResult.MeasurementRow"]] = list(rows or ())
        self._columnar: Optional[_ColumnarMeasurementBatch] = None

    @classmethod
    def _from_columnar(cls, batch: _ColumnarMeasurementBatch):
        inst = cls.__new__(cls)
        if batch.positions.size == 0:
            inst._rows = []
            inst._columnar = None
        else:
            inst._rows = None
            inst._columnar = batch
        return inst

    def _materialize(self) -> List["SEResult.MeasurementRow"]:
        rows = self._rows
        if rows is None:
            batch = self._columnar
            if batch is None or batch.positions.size == 0:
                rows = []
            else:
                rows = _measurement_rows_from_table(
                    batch.table,
                    batch.positions,
                    estimated=batch.estimated,
                    residual=batch.residual,
                    normalized=batch.normalized,
                    source=batch.source,
                    reasons=batch.reasons,
                )
            self._rows = rows
            self._columnar = None
        return rows

    @property
    def rows(self) -> List["SEResult.MeasurementRow"]:
        return self._materialize()

    @rows.setter
    def rows(self, value: Iterable["SEResult.MeasurementRow"]) -> None:
        self._rows = list(value)
        self._columnar = None

    def __len__(self) -> int:
        if self._rows is not None:
            return len(self._rows)
        return int(self._columnar.positions.size) if self._columnar is not None else 0

    def __iter__(self):
        return iter(self._materialize())

    def __getitem__(self, index: int):
        return self._materialize()[index]

    def append(
        self,
        measurement: Measurement,
        *,
        estimated_value: Optional[float] = None,
        residual: Optional[float] = None,
        normalized_residual: Optional[float] = None,
        reason: str = "",
        source: str = "",
    ) -> None:
        self._materialize().append(
            SEResult.MeasurementRow.from_measurement(
                measurement,
                estimated_value=estimated_value,
                residual=residual,
                normalized_residual=normalized_residual,
                reason=reason,
                source=source,
            )
        )

    def append_row(self, row: "SEResult.MeasurementRow") -> None:
        self._materialize().append(row)

    def extend(self, rows: Iterable["SEResult.MeasurementRow"]) -> None:
        self._materialize().extend(rows)

    def to_dicts(self) -> List[dict]:
        """高速导出字典列表（跳过 dataclasses.asdict 的递归 deepcopy 开销）。"""
        if self._columnar is not None and self._rows is None:
            batch = self._columnar
            positions_all = batch.positions
            n_total = int(positions_all.size)
            if n_total == 0:
                return []
            src = batch.source
            reasons_all = batch.reasons
            out: List[dict] = []
            out_extend = out.extend
            for start in range(0, n_total, 16384):
                pos_chunk = positions_all[start : start + 16384]
                chunk_len = int(pos_chunk.size)
                (
                    ids,
                    names,
                    device_types,
                    device_names,
                    meas_types,
                    weights,
                    valid,
                    values,
                    estimates,
                    residuals,
                    normalized_values,
                ) = _extract_table_columns_chunk(
                    batch.table,
                    pos_chunk,
                    estimated=batch.estimated,
                    residual=batch.residual,
                    normalized=batch.normalized,
                )
                est_iter = repeat(None, chunk_len) if estimates is None else estimates
                res_iter = repeat(None, chunk_len) if residuals is None else residuals
                nrm_iter = repeat(None, chunk_len) if normalized_values is None else normalized_values
                rsn_iter = repeat("", chunk_len) if reasons_all is None else reasons_all[start : start + chunk_len]
                out_extend(
                    {
                        "idx": idx,
                        "name": name,
                        "device_type": dt,
                        "device_name": dn,
                        "meas_type": mt,
                        "weight": w,
                        "valid": vld,
                        "value": val,
                        "estimated_value": est,
                        "residual": res,
                        "normalized_residual": nrm,
                        "reason": rsn,
                        "source": src,
                    }
                    for idx, name, dt, dn, mt, w, vld, val, est, res, nrm, rsn in zip(
                        ids,
                        names,
                        device_types,
                        device_names,
                        meas_types,
                        weights,
                        valid,
                        values,
                        est_iter,
                        res_iter,
                        nrm_iter,
                        rsn_iter,
                    )
                )
            return out
        return [
            {
                "idx": r.idx,
                "name": r.name,
                "device_type": r.device_type,
                "device_name": r.device_name,
                "meas_type": r.meas_type,
                "weight": r.weight,
                "valid": r.valid,
                "value": r.value,
                "estimated_value": r.estimated_value,
                "residual": r.residual,
                "normalized_residual": r.normalized_residual,
                "reason": r.reason,
                "source": r.source,
            }
            for r in self._rows
        ]


@dataclass
class SEResult:
    @dataclass(slots=True)
    class StatisticsTable:
        converged: bool = False
        iterations: int = 0
        objective: float = 0.0
        max_correction: float = 0.0
        residual_inf: float = 0.0
        observable: bool = False
        rank: int = 0
        state_count: int = 0
        measurement_count: int = 0
        deficiency: int = 0
        prefiltered_measurement_count: int = 0
        pseudo_measurement_count: int = 0
        bad_data_count: int = 0
        normal_measurement_count: int = 0

    @dataclass(slots=True)
    class MeasurementRow:
        idx: int
        name: str
        device_type: str
        device_name: str
        meas_type: str
        weight: float
        valid: bool
        value: float
        estimated_value: Optional[float] = None
        residual: Optional[float] = None
        normalized_residual: Optional[float] = None
        reason: str = ""
        source: str = ""

        @classmethod
        def from_measurement(
            cls,
            measurement: Measurement,
            *,
            estimated_value: Optional[float] = None,
            residual: Optional[float] = None,
            normalized_residual: Optional[float] = None,
            reason: str = "",
            source: str = "",
        ) -> "SEResult.MeasurementRow":
            return cls(
                idx=int(measurement.idx),
                name=str(measurement.name),
                device_type=str(measurement.device_type),
                device_name=str(measurement.device_name),
                meas_type=str(measurement.meas_type),
                weight=float(measurement.weight),
                valid=bool(measurement.valid),
                value=float(measurement.value),
                estimated_value=None if estimated_value is None else float(estimated_value),
                residual=None if residual is None else float(residual),
                normalized_residual=None if normalized_residual is None else float(normalized_residual),
                reason=str(reason),
                source=str(source),
            )

        @classmethod
        def from_table_row(
            cls,
            table: MeasurementTable,
            row: int,
            *,
            estimated_value: Optional[float] = None,
            residual: Optional[float] = None,
            normalized_residual: Optional[float] = None,
            reason: str = "",
            source: str = "",
        ) -> "SEResult.MeasurementRow":
            pos = int(row)
            idx = int(table.idx[pos])
            name = str(table.name[pos]) if int(table.name.size) > pos else f"m{idx}"
            if int(table.device_type.size) > pos:
                device_type = str(table.device_type[pos])
            else:
                device_type = DEVICE_TYPE_NAMES.get(int(table.device_type_code[pos]), "")
            if int(table.device_name.size) > pos:
                device_name = str(table.device_name[pos])
            else:
                device_pos = getattr(table, "device_pos", None)
                device_pos = None if device_pos is None else np.asarray(device_pos)
                device_name = (
                    f"pos:{int(device_pos[pos])}"
                    if device_pos is not None and int(device_pos.size) > pos and int(device_pos[pos]) >= 0
                    else ""
                )
            if int(table.meas_type.size) > pos:
                meas_type = str(table.meas_type[pos])
            else:
                meas_type_code = getattr(table, "meas_type_code", None)
                meas_type_code = None if meas_type_code is None else np.asarray(meas_type_code)
                meas_type = (
                    MEAS_TYPE_NAMES.get(int(meas_type_code[pos]), "")
                    if meas_type_code is not None and int(meas_type_code.size) > pos
                    else ""
                )
            return cls(
                idx=idx,
                name=name,
                device_type=device_type,
                device_name=device_name,
                meas_type=meas_type,
                weight=float(table.weight[pos]),
                valid=bool(table.valid[pos]),
                value=float(table.value[pos]),
                estimated_value=None if estimated_value is None else float(estimated_value),
                residual=None if residual is None else float(residual),
                normalized_residual=None if normalized_residual is None else float(normalized_residual),
                reason=str(reason),
                source=str(source),
            )

    class PrefilteredMeasurementTable(_MeasurementResultTable):
        pass

    class PseudoMeasurementTable(_MeasurementResultTable):
        pass

    class BadDataTable(_MeasurementResultTable):
        def append_bad_data_item(self, item: BadDataItem) -> None:
            self.append(
                item.measurement,
                estimated_value=item.estimated_value,
                residual=item.residual,
                normalized_residual=item.normalized_residual,
                source="bad_data",
            )

    class NormalMeasurementTable(_MeasurementResultTable):
        pass

    statistics: StatisticsTable = field(default_factory=lambda: SEResult.StatisticsTable())
    prefiltered_measurements: PrefilteredMeasurementTable = field(
        default_factory=lambda: SEResult.PrefilteredMeasurementTable()
    )
    pseudo_measurements: PseudoMeasurementTable = field(default_factory=lambda: SEResult.PseudoMeasurementTable())
    bad_data: BadDataTable = field(default_factory=lambda: SEResult.BadDataTable())
    normal_measurements: NormalMeasurementTable = field(default_factory=lambda: SEResult.NormalMeasurementTable())

    _STATISTICS_COLUMNS = (
        "converged",
        "iterations",
        "objective",
        "max_correction",
        "residual_inf",
        "observable",
        "rank",
        "state_count",
        "measurement_count",
        "deficiency",
        "prefiltered_measurement_count",
        "pseudo_measurement_count",
        "bad_data_count",
        "normal_measurement_count",
    )
    _MEASUREMENT_COLUMNS = (
        "idx",
        "name",
        "device_type",
        "device_name",
        "meas_type",
        "weight",
        "valid",
        "value",
        "estimated_value",
        "residual",
        "normalized_residual",
        "reason",
        "source",
    )

    @staticmethod
    def _is_pseudo_measurement(measurement: Measurement) -> bool:
        return is_pseudo_measurement(measurement)

    @staticmethod
    def _prefiltered_reason(measurement: Measurement) -> str:
        if not bool(measurement.valid):
            return "invalid"
        if float(measurement.weight) <= 0.0:
            return "zero weight"
        return ""

    @classmethod
    def from_estimate_result(
        cls,
        result: EstimateResult,
        *,
        bad_items: Optional[Sequence[BadDataItem]] = None,
        normalized_residual: Optional[Sequence[float]] = None,
        prefiltered_measurements: Optional[Iterable[object]] = None,
        all_measurements: Optional[Iterable[Measurement]] = None,
    ) -> "SEResult":
        se_result = cls()
        bad_items = list(bad_items or ())
        normalized = np.asarray(normalized_residual, dtype=np.float64) if normalized_residual is not None else None
        measurement_ids = {id(meas) for meas in result.measurements}
        bad_ids = {id(item.measurement) for item in bad_items}
        bad_indexes = {int(item.measurement.idx) for item in bad_items}

        prefiltered_rows = cls._normalize_prefiltered_rows(
            prefiltered_measurements,
            all_measurements,
            measurement_ids,
        )
        for measurement, reason in prefiltered_rows:
            se_result.prefiltered_measurements.append(measurement, reason=reason, source="prefiltered")

        for pos, measurement in enumerate(result.measurements):
            estimated = cls._array_value(result.z_est, pos)
            residual = cls._array_value(result.residual, pos)
            normalized_value = cls._array_value(normalized, pos)
            if cls._is_pseudo_measurement(measurement):
                se_result.pseudo_measurements.append(
                    measurement,
                    estimated_value=estimated,
                    residual=residual,
                    normalized_residual=normalized_value,
                    source="pseudo",
                )
                continue
            if id(measurement) in bad_ids or int(measurement.idx) in bad_indexes:
                continue
            se_result.normal_measurements.append(
                measurement,
                estimated_value=estimated,
                residual=residual,
                normalized_residual=normalized_value,
                source="normal",
            )

        for item in bad_items:
            se_result.bad_data.append_bad_data_item(item)

        obs = result.observability
        se_result.statistics = cls.StatisticsTable(
            converged=bool(result.converged),
            iterations=int(result.iterations),
            objective=float(result.objective),
            max_correction=float(result.max_correction),
            residual_inf=float(result.residual_inf),
            observable=bool(obs.observable),
            rank=int(obs.rank),
            state_count=int(obs.state_count),
            measurement_count=int(obs.measurement_count),
            deficiency=int(obs.deficiency),
            prefiltered_measurement_count=len(se_result.prefiltered_measurements),
            pseudo_measurement_count=len(se_result.pseudo_measurements),
            bad_data_count=len(se_result.bad_data),
            normal_measurement_count=len(se_result.normal_measurements),
        )
        return se_result

    @classmethod
    def _normalize_prefiltered_rows(
        cls,
        prefiltered_measurements: Optional[Iterable[object]],
        all_measurements: Optional[Iterable[Measurement]],
        active_measurement_ids: set,
    ) -> List[Tuple[Measurement, str]]:
        if prefiltered_measurements is not None:
            rows = []
            for item in prefiltered_measurements:
                if isinstance(item, tuple):
                    measurement, reason = item
                else:
                    measurement = item
                    reason = cls._prefiltered_reason(measurement)
                rows.append((measurement, str(reason)))
            return rows

        if all_measurements is None:
            return []
        rows = []
        for measurement in all_measurements:
            if id(measurement) in active_measurement_ids:
                continue
            reason = cls._prefiltered_reason(measurement)
            if reason:
                rows.append((measurement, reason))
        return rows

    @staticmethod
    def _array_value(values, pos: int) -> Optional[float]:
        if values is None:
            return None
        if pos >= len(values):
            return None
        return float(values[pos])

    def to_e_text(self) -> str:
        blocks = [
            self._statistics_block_to_e(),
            self._measurement_block_to_e("SEResultPrefilteredMeasurement", self.prefiltered_measurements),
            self._measurement_block_to_e("SEResultPseudoMeasurement", self.pseudo_measurements),
            self._measurement_block_to_e("SEResultBadData", self.bad_data),
            self._measurement_block_to_e("SEResultNormalMeasurement", self.normal_measurements),
        ]
        return "\n\n".join(blocks) + "\n"

    def write_e_file(self, file_path: Path) -> None:
        """分块高速写入 E 格式状态估计结果文件，避免超大结果单字符串峰值拷贝。"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table_specs = (
            ("SEResultPrefilteredMeasurement", self.prefiltered_measurements),
            ("SEResultPseudoMeasurement", self.pseudo_measurements),
            ("SEResultBadData", self.bad_data),
            ("SEResultNormalMeasurement", self.normal_measurements),
        )
        with open(path, mode="wt", encoding="utf-8", buffering=1024 * 1024) as fp:
            fp.write(self._statistics_block_to_e())
            for block_name, table in table_specs:
                fp.write("\n\n")
                for chunk in self._iter_measurement_block_e_chunks(block_name, table):
                    fp.write(chunk)
            fp.write("\n")

    def _statistics_block_to_e(self) -> str:
        columns = self._STATISTICS_COLUMNS
        values = [getattr(self.statistics, name) for name in columns]
        return self._block_to_e("SEResultStatistics", columns, [values])

    @classmethod
    def _iter_measurement_block_e_chunks(cls, block_name: str, table: _MeasurementResultTable):
        header_line = f"<{block_name}>\n@ " + " ".join(cls._MEASUREMENT_COLUMNS)
        n_rows = len(table)
        if n_rows == 0:
            yield f"{header_line}\n</{block_name}>"
            return
        yield header_line + "\n"
        sanitize = _sanitize_e_text
        if table._columnar is not None and table._rows is None:
            batch = table._columnar
            tbl = batch.table
            positions_all = batch.positions
            src_tok = sanitize(batch.source)
            reasons_all = batch.reasons
            dt_lookup = _SANITIZED_DEVICE_TYPE_NAMES.get
            mt_lookup = _SANITIZED_MEAS_TYPE_NAMES.get
            device_pos_all = getattr(tbl, "device_pos", None)
            meas_codes_all = getattr(tbl, "meas_type_code", None)
            dev_codes_all = getattr(tbl, "device_type_code", None)
            name_empty = tbl.name.size == 0
            dev_name_empty = tbl.device_name.size == 0
            dev_type_empty = tbl.device_type.size == 0
            meas_type_empty = tbl.meas_type.size == 0

            def _extract_sanitized_text_col(values, pos_chunk, max_pos, fallback_fn):
                if values.size > max_pos:
                    selected = values[pos_chunk]
                    if values.dtype.kind in "OUS":
                        raw_list = selected.tolist()
                        if not all(type(x) is str for x in raw_list):
                            raw_list = list(map(str, raw_list))
                    else:
                        raw_list = list(map(str, selected))
                else:
                    raw_list = [str(values[p]) if p < values.size else fallback_fn(int(p)) for p in pos_chunk.tolist()]
                uniq = {s: sanitize(s) for s in dict.fromkeys(raw_list)}
                return list(map(uniq.__getitem__, raw_list))

            def _optional_float_col(values, pos_chunk, max_pos):
                if values is None or values.size == 0:
                    return None
                if values.size > max_pos:
                    return values[pos_chunk].tolist()
                n_val = int(values.size)
                return [float(values[p]) if p < n_val else None for p in pos_chunk.tolist()]

            for start in range(0, n_rows, 16384):
                pos_chunk = positions_all[start : start + 16384]
                chunk_len = int(pos_chunk.size)
                max_pos = int(pos_chunk[-1])

                ids = tbl.idx[pos_chunk].astype(np.int64, copy=False).tolist()
                if dev_type_empty and dev_codes_all is not None and len(dev_codes_all) > max_pos:
                    dt_s = [dt_lookup(code, "-") for code in dev_codes_all[pos_chunk].tolist()]
                else:
                    dt_s = _extract_sanitized_text_col(
                        tbl.device_type,
                        pos_chunk,
                        max_pos,
                        lambda p: DEVICE_TYPE_NAMES.get(int(dev_codes_all[p]), "")
                        if dev_codes_all is not None and len(dev_codes_all) > p
                        else "",
                    )
                if meas_type_empty and meas_codes_all is not None and len(meas_codes_all) > max_pos:
                    mt_s = [mt_lookup(code, "-") for code in meas_codes_all[pos_chunk].tolist()]
                else:
                    mt_s = _extract_sanitized_text_col(
                        tbl.meas_type,
                        pos_chunk,
                        max_pos,
                        lambda p: MEAS_TYPE_NAMES.get(int(meas_codes_all[p]), "")
                        if meas_codes_all is not None and len(meas_codes_all) > p
                        else "",
                    )

                dp_list = None
                dn_s = None
                if dev_name_empty and device_pos_all is not None and len(device_pos_all) > max_pos:
                    dp_arr = device_pos_all[pos_chunk]
                    if bool(np.all(dp_arr >= 0)):
                        dp_list = dp_arr.tolist()
                    else:
                        dn_s = [f"pos:{pos}" if pos >= 0 else "-" for pos in dp_arr.tolist()]
                else:
                    dn_s = _extract_sanitized_text_col(
                        tbl.device_name,
                        pos_chunk,
                        max_pos,
                        lambda p: f"pos:{int(device_pos_all[p])}"
                        if device_pos_all is not None and len(device_pos_all) > p and int(device_pos_all[p]) >= 0
                        else "",
                    )

                weights = tbl.weight[pos_chunk].astype(np.float64, copy=False).tolist()
                valid_arr = tbl.valid[pos_chunk].astype(bool, copy=False)
                all_valid_true = bool(valid_arr.all())
                values = tbl.value[pos_chunk].astype(np.float64, copy=False).tolist()
                estimates = _optional_float_col(batch.estimated, pos_chunk, max_pos)
                residuals = _optional_float_col(batch.residual, pos_chunk, max_pos)
                normalized_values = _optional_float_col(batch.normalized, pos_chunk, max_pos)

                all_est = estimates is not None and (batch.estimated is not None and batch.estimated.size > max_pos)
                all_res = residuals is not None and (batch.residual is not None and batch.residual.size > max_pos)
                all_nrm = normalized_values is not None and (batch.normalized is not None and batch.normalized.size > max_pos)

                if reasons_all is None and all_est and all_res and all_nrm:
                    if name_empty and dp_list is not None and all_valid_true:
                        lines = [
                            f"# {idx} m{idx} {dt} pos:{dp} {mt} {w:.12g} 1 {val:.12g} {est:.12g} {res:.12g} {nrm:.12g} - {src_tok}"
                            for idx, dt, dp, mt, w, val, est, res, nrm in zip(
                                ids, dt_s, dp_list, mt_s, weights, values, estimates, residuals, normalized_values
                            )
                        ]
                    else:
                        names_s = (
                            [f"m{idx}" for idx in ids]
                            if name_empty
                            else _extract_sanitized_text_col(tbl.name, pos_chunk, max_pos, lambda p: f"m{int(tbl.idx[p])}")
                        )
                        if dn_s is None:
                            dn_s = [f"pos:{dp}" for dp in dp_list]
                        if all_valid_true:
                            lines = [
                                f"# {idx} {nm} {dt} {dn} {mt} {w:.12g} 1 {val:.12g} {est:.12g} {res:.12g} {nrm:.12g} - {src_tok}"
                                for idx, nm, dt, dn, mt, w, val, est, res, nrm in zip(
                                    ids, names_s, dt_s, dn_s, mt_s, weights, values, estimates, residuals, normalized_values
                                )
                            ]
                        else:
                            valid = valid_arr.tolist()
                            lines = [
                                f"# {idx} {nm} {dt} {dn} {mt} {w:.12g} {'1' if vld else '0'} {val:.12g} {est:.12g} {res:.12g} {nrm:.12g} - {src_tok}"
                                for idx, nm, dt, dn, mt, w, vld, val, est, res, nrm in zip(
                                    ids, names_s, dt_s, dn_s, mt_s, weights, valid, values, estimates, residuals, normalized_values
                                )
                            ]
                elif reasons_all is None and all_est and all_res and normalized_values is None:
                    if name_empty and dp_list is not None and all_valid_true:
                        lines = [
                            f"# {idx} m{idx} {dt} pos:{dp} {mt} {w:.12g} 1 {val:.12g} {est:.12g} {res:.12g} - - {src_tok}"
                            for idx, dt, dp, mt, w, val, est, res in zip(
                                ids, dt_s, dp_list, mt_s, weights, values, estimates, residuals
                            )
                        ]
                    else:
                        names_s = (
                            [f"m{idx}" for idx in ids]
                            if name_empty
                            else _extract_sanitized_text_col(tbl.name, pos_chunk, max_pos, lambda p: f"m{int(tbl.idx[p])}")
                        )
                        if dn_s is None:
                            dn_s = [f"pos:{dp}" for dp in dp_list]
                        valid = valid_arr.tolist()
                        lines = [
                            f"# {idx} {nm} {dt} {dn} {mt} {w:.12g} {'1' if vld else '0'} {val:.12g} {est:.12g} {res:.12g} - - {src_tok}"
                            for idx, nm, dt, dn, mt, w, vld, val, est, res in zip(
                                ids, names_s, dt_s, dn_s, mt_s, weights, valid, values, estimates, residuals
                            )
                        ]
                elif estimates is None and residuals is None and normalized_values is None:
                    all_valid_false = (not all_valid_true) and (not bool(valid_arr.any()))
                    if name_empty and dp_list is not None:
                        if reasons_all is None:
                            if all_valid_false:
                                lines = [
                                    f"# {idx} m{idx} {dt} pos:{dp} {mt} {w:.12g} 0 {val:.12g} - - - - {src_tok}"
                                    for idx, dt, dp, mt, w, val in zip(ids, dt_s, dp_list, mt_s, weights, values)
                                ]
                            elif all_valid_true:
                                lines = [
                                    f"# {idx} m{idx} {dt} pos:{dp} {mt} {w:.12g} 1 {val:.12g} - - - - {src_tok}"
                                    for idx, dt, dp, mt, w, val in zip(ids, dt_s, dp_list, mt_s, weights, values)
                                ]
                            else:
                                valid = valid_arr.tolist()
                                lines = [
                                    f"# {idx} m{idx} {dt} pos:{dp} {mt} {w:.12g} {'1' if vld else '0'} {val:.12g} - - - - {src_tok}"
                                    for idx, dt, dp, mt, w, vld, val in zip(ids, dt_s, dp_list, mt_s, weights, valid, values)
                                ]
                        else:
                            rsn_raw = reasons_all[start : start + chunk_len]
                            rsn_map = {s: sanitize(s) for s in dict.fromkeys(rsn_raw)}
                            rsn_chunk = list(map(rsn_map.__getitem__, rsn_raw))
                            if all_valid_false:
                                lines = [
                                    f"# {idx} m{idx} {dt} pos:{dp} {mt} {w:.12g} 0 {val:.12g} - - - {rsn} {src_tok}"
                                    for idx, dt, dp, mt, w, val, rsn in zip(ids, dt_s, dp_list, mt_s, weights, values, rsn_chunk)
                                ]
                            elif all_valid_true:
                                lines = [
                                    f"# {idx} m{idx} {dt} pos:{dp} {mt} {w:.12g} 1 {val:.12g} - - - {rsn} {src_tok}"
                                    for idx, dt, dp, mt, w, val, rsn in zip(ids, dt_s, dp_list, mt_s, weights, values, rsn_chunk)
                                ]
                            else:
                                valid = valid_arr.tolist()
                                lines = [
                                    f"# {idx} m{idx} {dt} pos:{dp} {mt} {w:.12g} {'1' if vld else '0'} {val:.12g} - - - {rsn} {src_tok}"
                                    for idx, dt, dp, mt, w, vld, val, rsn in zip(
                                        ids, dt_s, dp_list, mt_s, weights, valid, values, rsn_chunk
                                    )
                                ]
                    else:
                        names_s = (
                            [f"m{idx}" for idx in ids]
                            if name_empty
                            else _extract_sanitized_text_col(tbl.name, pos_chunk, max_pos, lambda p: f"m{int(tbl.idx[p])}")
                        )
                        if dn_s is None:
                            dn_s = [f"pos:{dp}" for dp in dp_list]
                        valid = valid_arr.tolist()
                        if reasons_all is None:
                            lines = [
                                f"# {idx} {nm} {dt} {dn} {mt} {w:.12g} {'1' if vld else '0'} {val:.12g} - - - - {src_tok}"
                                for idx, nm, dt, dn, mt, w, vld, val in zip(
                                    ids, names_s, dt_s, dn_s, mt_s, weights, valid, values
                                )
                            ]
                        else:
                            rsn_raw = reasons_all[start : start + chunk_len]
                            rsn_map = {s: sanitize(s) for s in dict.fromkeys(rsn_raw)}
                            rsn_chunk = list(map(rsn_map.__getitem__, rsn_raw))
                            lines = [
                                f"# {idx} {nm} {dt} {dn} {mt} {w:.12g} {'1' if vld else '0'} {val:.12g} - - - {rsn} {src_tok}"
                                for idx, nm, dt, dn, mt, w, vld, val, rsn in zip(
                                    ids, names_s, dt_s, dn_s, mt_s, weights, valid, values, rsn_chunk
                                )
                            ]
                else:
                    names_s = (
                        [f"m{idx}" for idx in ids]
                        if name_empty
                        else _extract_sanitized_text_col(tbl.name, pos_chunk, max_pos, lambda p: f"m{int(tbl.idx[p])}")
                    )
                    if dn_s is None:
                        dn_s = [f"pos:{dp}" for dp in dp_list]
                    valid = valid_arr.tolist()
                    est_iter = repeat(None, chunk_len) if estimates is None else estimates
                    res_iter = repeat(None, chunk_len) if residuals is None else residuals
                    nrm_iter = repeat(None, chunk_len) if normalized_values is None else normalized_values
                    rsn_iter = (
                        repeat("-", chunk_len)
                        if reasons_all is None
                        else [sanitize(s) for s in reasons_all[start : start + chunk_len]]
                    )
                    lines = [
                        (
                            f"# {idx} {nm} {dt} {dn} {mt} {w:.12g} {'1' if vld else '0'} {val:.12g} "
                            f"{'-' if est is None else format(est, '.12g')} "
                            f"{'-' if res is None else format(res, '.12g')} "
                            f"{'-' if nrm is None else format(nrm, '.12g')} {rsn} {src_tok}"
                        )
                        for idx, nm, dt, dn, mt, w, vld, val, est, res, nrm, rsn in zip(
                            ids, names_s, dt_s, dn_s, mt_s, weights, valid, values, est_iter, res_iter, nrm_iter, rsn_iter
                        )
                    ]
                yield "\n".join(lines) + "\n"
            yield f"</{block_name}>"
            return

        # 已物化 MeasurementRow 列表同样走直接属性访问 + f-string 快速路径
        rows_list = table._rows or ()
        for start in range(0, len(rows_list), 16384):
            chunk_rows = rows_list[start : start + 16384]
            lines = [
                (
                    f"# {int(r.idx)} {sanitize(str(r.name))} {sanitize(str(r.device_type))} "
                    f"{sanitize(str(r.device_name))} {sanitize(str(r.meas_type))} "
                    f"{float(r.weight):.12g} {'1' if r.valid else '0'} {float(r.value):.12g} "
                    f"{'-' if r.estimated_value is None else format(float(r.estimated_value), '.12g')} "
                    f"{'-' if r.residual is None else format(float(r.residual), '.12g')} "
                    f"{'-' if r.normalized_residual is None else format(float(r.normalized_residual), '.12g')} "
                    f"{sanitize(str(r.reason))} {sanitize(str(r.source))}"
                )
                for r in chunk_rows
            ]
            yield "\n".join(lines) + "\n"
        yield f"</{block_name}>"

    def _measurement_block_to_e(self, block_name: str, table: _MeasurementResultTable) -> str:
        return "".join(self._iter_measurement_block_e_chunks(block_name, table))

    @classmethod
    def _block_to_e(cls, block_name: str, columns: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
        lines = [f"<{block_name}>", "@ " + " ".join(columns)]
        for row in rows:
            lines.append("# " + " ".join(cls._e_cell(value) for value in row))
        lines.append(f"</{block_name}>")
        return "\n".join(lines)

    @staticmethod
    def _e_cell(value) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (np.bool_,)):
            return "1" if bool(value) else "0"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return format(float(value), ".12g")
        text = str(value)
        if not text:
            return "-"
        return "_".join(text.split())


def _table_prefiltered_reason(table: MeasurementTable, row: int) -> str:
    pos = int(row)
    if not bool(table.valid[pos]):
        return "invalid"
    if float(table.weight[pos]) <= 0.0:
        return "zero weight"
    return ""


def _active_source_mask(table: MeasurementTable, active_source_rows: Optional[Sequence[int]]) -> np.ndarray:
    active = np.zeros(int(table.idx.size), dtype=bool)
    if active_source_rows is None:
        return active
    rows = np.asarray(active_source_rows, dtype=np.int64)
    if rows.size == 0:
        return active
    rows = rows[(rows >= 0) & (rows < active.size)]
    if rows.size:
        active[rows.astype(np.intp, copy=False)] = True
    return active


def _bad_row_positions(bad_items: Optional[Sequence[BadDataItem]], row_count: int) -> set:
    positions = set()
    for item in bad_items or ():
        row_pos = int(getattr(item, "row_pos", -1))
        if 0 <= row_pos < row_count:
            positions.add(row_pos)
    return positions


def _seresult_statistics_from_table(
    result: EstimateResult,
    *,
    bad_items: Optional[Sequence[BadDataItem]],
    all_measurement_table: Optional[MeasurementTable],
    active_source_rows: Optional[Sequence[int]],
) -> "SEResult.StatisticsTable":
    table = result.measurement_table
    status_code = measurement_table_status_code(table)
    pseudo_count = int(np.count_nonzero(status_code == MEAS_STATUS_PSEUDO))
    bad_count = len(bad_items or ())
    normal_count = max(0, int(table.idx.size) - pseudo_count - bad_count)
    prefiltered_count = 0
    if all_measurement_table is not None:
        if active_source_rows is None and int(all_measurement_table.idx.size) == int(table.idx.size):
            active_mask = np.ones(int(all_measurement_table.idx.size), dtype=bool)
        else:
            active_mask = _active_source_mask(all_measurement_table, active_source_rows)
        prefiltered = (
            (~active_mask)
            & (
                (~np.asarray(all_measurement_table.valid, dtype=bool))
                | (np.asarray(all_measurement_table.weight, dtype=np.float64) <= 0.0)
            )
        )
        prefiltered_count = int(np.count_nonzero(prefiltered))
    obs = result.observability
    return SEResult.StatisticsTable(
        converged=bool(result.converged),
        iterations=int(result.iterations),
        objective=float(result.objective),
        max_correction=float(result.max_correction),
        residual_inf=float(result.residual_inf),
        observable=bool(obs.observable),
        rank=int(obs.rank),
        state_count=int(obs.state_count),
        measurement_count=int(obs.measurement_count),
        deficiency=int(obs.deficiency),
        prefiltered_measurement_count=prefiltered_count,
        pseudo_measurement_count=pseudo_count,
        bad_data_count=bad_count,
        normal_measurement_count=normal_count,
    )


def build_seresult_summary_from_table(
    result: EstimateResult,
    *,
    bad_items: Optional[Sequence[BadDataItem]] = None,
    all_measurement_table: Optional[MeasurementTable] = None,
    active_source_rows: Optional[Sequence[int]] = None,
) -> "SEResult":
    """Build a summary result directly from measurement arrays."""
    if result.measurement_table is None:
        raise RuntimeError("SEResult summary requires result.measurement_table")
    se_result = SEResult()
    se_result.statistics = _seresult_statistics_from_table(
        result,
        bad_items=bad_items,
        all_measurement_table=all_measurement_table,
        active_source_rows=active_source_rows,
    )
    return se_result


def _measurement_rows_from_table(
    table: MeasurementTable, rows, *, estimated=None, residual=None,
    normalized=None, source="", reasons=None,
):
    """Materialize ordinary result rows in bounded column batches.

    The public row/list types stay unchanged. Only the conversion work moves
    outside the per-row constructor; no table or result survives this call.
    """
    rows = np.asarray(rows, dtype=np.intp)
    row_type = SEResult.MeasurementRow
    device_pos = getattr(table, "device_pos", None)
    meas_codes = getattr(table, "meas_type_code", None)

    def text_column(values, positions, fallback):
        if values.size > int(positions[-1]):
            selected = values[positions]
            # Object/string arrays retain the same string conversion after
            # tolist; numerical metadata keeps NumPy scalar formatting.
            if values.dtype.kind in "OUS":
                selected = selected.tolist()
            return list(map(str, selected))
        return [str(values[p]) if p < values.size else fallback(int(p)) for p in positions]

    def coded_text(values, codes, positions, names):
        if values.size == 0 and codes is not None and len(codes) > int(positions[-1]):
            lookup = names.get
            return [lookup(code, "") for code in codes[positions].tolist()]
        return text_column(values, positions, lambda p: names.get(int(codes[p]), "") if codes is not None and len(codes) > p else "")

    def optional_column(values, positions):
        if values is None:
            return [None] * len(positions)
        values = np.asarray(values, dtype=np.float64)
        if values.size > int(positions[-1]):
            return values[positions].tolist()
        return [float(values[p]) if p < len(values) else None for p in positions]

    def missing_device_name(p):
        return f"pos:{int(device_pos[p])}" if device_pos is not None and len(device_pos) > p and int(device_pos[p]) >= 0 else ""

    rows_out = []
    for start in range(0, rows.size, 8192):
        positions = rows[start:start + 8192]
        ids = table.idx[positions].astype(np.int64).tolist()
        names = ([f"m{idx}" for idx in ids] if table.name.size == 0 else
                 text_column(table.name, positions, lambda p: f"m{int(table.idx[p])}"))
        device_types = coded_text(table.device_type, table.device_type_code, positions, DEVICE_TYPE_NAMES)
        if table.device_name.size == 0 and device_pos is not None and len(device_pos) > int(positions[-1]):
            device_names = [f"pos:{pos}" if pos >= 0 else "" for pos in device_pos[positions].tolist()]
        else:
            device_names = text_column(table.device_name, positions, missing_device_name)
        meas_types = coded_text(table.meas_type, meas_codes, positions, MEAS_TYPE_NAMES)
        weights = table.weight[positions].astype(np.float64, copy=False).tolist()
        valid = table.valid[positions].astype(bool, copy=False).tolist()
        values = table.value[positions].astype(np.float64, copy=False).tolist()
        estimates = optional_column(estimated, positions)
        residuals = optional_column(residual, positions)
        normalized_values = optional_column(normalized, positions)
        reason_values = repeat("", len(positions)) if reasons is None else reasons[start:start + 8192]
        rows_out.extend(
            starmap(
                row_type,
                zip(
                    ids,
                    names,
                    device_types,
                    device_names,
                    meas_types,
                    weights,
                    valid,
                    values,
                    estimates,
                    residuals,
                    normalized_values,
                    reason_values,
                    repeat(source, len(positions)),
                ),
            )
        )
    return rows_out


def build_seresult_full_from_table(
    result: EstimateResult,
    *,
    bad_items: Optional[Sequence[BadDataItem]] = None,
    normalized_residual: Optional[Sequence[float]] = None,
    all_measurement_table: Optional[MeasurementTable] = None,
    active_source_rows: Optional[Sequence[int]] = None,
) -> "SEResult":
    """Build a full SEResult from array-backed measurement tables with lazy row materialization."""
    table = result.measurement_table
    if table is None:
        raise RuntimeError("SEResult full output requires result.measurement_table")
    se_result = SEResult()
    bad_items = list(bad_items or ())
    normalized = np.asarray(normalized_residual, dtype=np.float64) if normalized_residual is not None else None
    n_active = int(table.idx.size)
    bad_positions = _bad_row_positions(bad_items, n_active)
    status_code = measurement_table_status_code(table)

    prefiltered_count = 0
    if all_measurement_table is not None:
        if active_source_rows is None and int(all_measurement_table.idx.size) == n_active:
            active_mask = np.ones(int(all_measurement_table.idx.size), dtype=bool)
        else:
            active_mask = _active_source_mask(all_measurement_table, active_source_rows)
        prefiltered = np.flatnonzero(
            (~active_mask)
            & (
                (~np.asarray(all_measurement_table.valid, dtype=bool))
                | (np.asarray(all_measurement_table.weight, dtype=np.float64) <= 0.0)
            )
        )
        prefiltered_count = int(prefiltered.size)
        if prefiltered_count:
            reasons = np.where(all_measurement_table.valid[prefiltered], "zero weight", "invalid").tolist()
            se_result.prefiltered_measurements = SEResult.PrefilteredMeasurementTable._from_columnar(
                _ColumnarMeasurementBatch(
                    all_measurement_table,
                    prefiltered,
                    source="prefiltered",
                    reasons=reasons,
                )
            )

    pseudo_mask = status_code == MEAS_STATUS_PSEUDO
    pseudo_rows = np.flatnonzero(pseudo_mask)
    pseudo_count = int(pseudo_rows.size)
    normal_mask = ~pseudo_mask
    if bad_positions:
        normal_mask[list(bad_positions)] = False
    normal_rows = np.flatnonzero(normal_mask)

    if pseudo_count:
        se_result.pseudo_measurements = SEResult.PseudoMeasurementTable._from_columnar(
            _ColumnarMeasurementBatch(
                table,
                pseudo_rows,
                estimated=result.z_est,
                residual=result.residual,
                normalized=normalized,
                source="pseudo",
            )
        )
    if normal_rows.size:
        se_result.normal_measurements = SEResult.NormalMeasurementTable._from_columnar(
            _ColumnarMeasurementBatch(
                table,
                normal_rows,
                estimated=result.z_est,
                residual=result.residual,
                normalized=normalized,
                source="normal",
            )
        )

    for item in bad_items:
        row_pos = int(getattr(item, "row_pos", -1))
        if 0 <= row_pos < n_active:
            row = SEResult.MeasurementRow.from_table_row(
                table,
                row_pos,
                estimated_value=getattr(item, "estimated_value", None),
                residual=getattr(item, "residual", None),
                normalized_residual=getattr(item, "normalized_residual", None),
                source="bad_data",
            )
            se_result.bad_data.append_row(row)
        elif getattr(item, "measurement", None) is not None:
            se_result.bad_data.append_bad_data_item(item)

    # 直接复用上面已计算的掩码计数，避免重复扫描整张量测表
    bad_count = len(bad_items)
    normal_count = max(0, n_active - pseudo_count - bad_count)
    obs = result.observability
    se_result.statistics = SEResult.StatisticsTable(
        converged=bool(result.converged),
        iterations=int(result.iterations),
        objective=float(result.objective),
        max_correction=float(result.max_correction),
        residual_inf=float(result.residual_inf),
        observable=bool(obs.observable),
        rank=int(obs.rank),
        state_count=int(obs.state_count),
        measurement_count=int(obs.measurement_count),
        deficiency=int(obs.deficiency),
        prefiltered_measurement_count=prefiltered_count,
        pseudo_measurement_count=pseudo_count,
        bad_data_count=bad_count,
        normal_measurement_count=normal_count,
    )
    return se_result
