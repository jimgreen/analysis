from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from efile_read import EBook
from paths import LF_PARAMETER_DIR, PROJECT_ROOT, SE_PARAMETER_DIR, resolve_project_file


ROOT_DIR = PROJECT_ROOT
DEFAULT_LF_PARAMETER_FILE = LF_PARAMETER_DIR / "lf.para"
DEFAULT_SE_PARAMETER_FILE = SE_PARAMETER_DIR / "se.para"

DEFAULT_MAX_BAD_DATA_PER_ROUND_CAP = 20
DEFAULT_MAX_BAD_DATA_PER_ROUND_RATIO = 0.001
DEFAULT_MAX_BAD_DATA_TOTAL_CAP = 100
DEFAULT_MAX_BAD_DATA_TOTAL_RATIO = 0.005

_UNSET = object()


def resolve_bad_data_limits(
    measurement_count: int,
    max_remove: Optional[int] = None,
    max_remove_per_round: Optional[int] = None,
    *,
    param_max_remove: Optional[int] = None,
    param_max_remove_per_round: Optional[int] = None,
) -> Tuple[int, int]:
    """根据量测总数与用户覆盖参数，推导坏数据排除总数上限与每轮排除上限。

    默认规则（当对应参数未显式指定正整数时生效）：
    - 每轮最大排除数量：``max(1, min(20, int(N * 0.001)))``（``N == 0`` 时为 ``0``）；
    - 总最大排除数量：``max(1, min(100, int(N * 0.005)))``（``N == 0`` 时为 ``0``）。
    显式传入 ``max_remove=0`` 表示禁止排除坏数据。
    """
    n_meas = max(0, int(measurement_count))
    default_total = (
        max(1, min(DEFAULT_MAX_BAD_DATA_TOTAL_CAP, int(n_meas * DEFAULT_MAX_BAD_DATA_TOTAL_RATIO)))
        if n_meas > 0
        else 0
    )
    default_per_round = (
        max(1, min(DEFAULT_MAX_BAD_DATA_PER_ROUND_CAP, int(n_meas * DEFAULT_MAX_BAD_DATA_PER_ROUND_RATIO)))
        if n_meas > 0
        else 0
    )

    if max_remove is not None:
        raw_total = int(max_remove)
        eff_total = 0 if raw_total == 0 else (raw_total if raw_total > 0 else default_total)
    elif param_max_remove is not None:
        raw_total = int(param_max_remove)
        eff_total = 0 if raw_total == 0 else (raw_total if raw_total > 0 else default_total)
    else:
        eff_total = default_total

    if max_remove_per_round is not None and int(max_remove_per_round) > 0:
        eff_per_round = int(max_remove_per_round)
    elif param_max_remove_per_round is not None and int(param_max_remove_per_round) > 0:
        eff_per_round = int(param_max_remove_per_round)
    else:
        eff_per_round = default_per_round

    if eff_total <= 0:
        return 0, 0
    return eff_total, min(eff_per_round, eff_total)


def _to_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean parameter value: {value}")


def _load_key_value_block(path: Path, block_name: str) -> Dict[str, str]:
    data = EBook(path).to_dict()
    if block_name not in data:
        raise RuntimeError(f"{path} does not contain a <{block_name}> block")
    values = {}
    for row in data[block_name]["data"]:
        values[str(row["name"])] = str(row["value"])
    return values


def _require(values: Dict[str, str], key: str) -> str:
    if key not in values:
        raise RuntimeError(f"Missing required algorithm parameter: {key}")
    return values[key]


def _override_value(current, value):
    return current if value is None else value


@dataclass(frozen=True)
class PowerFlowParameters:
    tol: float
    max_iter: int
    min_voltage: float
    divergence_threshold: float

    def with_overrides(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        min_voltage: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
    ) -> "PowerFlowParameters":
        return replace(
            self,
            tol=float(_override_value(self.tol, tol)),
            max_iter=int(_override_value(self.max_iter, max_iter)),
            min_voltage=float(_override_value(self.min_voltage, min_voltage)),
            divergence_threshold=float(_override_value(self.divergence_threshold, divergence_threshold)),
        )


@dataclass(frozen=True)
class StateEstimationParameters:
    tol: float
    max_iter: int
    diff_step: float
    flat_start: bool
    bad_threshold: float
    max_remove: Optional[int]
    targeted_pseudo_measurement_max: int
    targeted_pseudo_measurement_redundancy_ratio: float
    targeted_pseudo_measurement_step: int
    voltage_floor: float
    min_current_voltage: float
    power_flow_tol: float
    power_flow_max_iter: int
    power_flow_min_voltage: float
    pseudo_measurement_weight_v: float = 0.1
    pseudo_measurement_weight_pq: float = 0.01
    voltage_min: float = 0.1
    voltage_max: float = 1.9
    angle_min_deg: float = -270.0
    angle_max_deg: float = 270.0
    max_remove_per_round: Optional[int] = None

    @property
    def pseudo_measurement_weight(self) -> float:
        """向后兼容只读属性，默认返回有功/无功伪量测权重。"""
        return self.pseudo_measurement_weight_pq

    def with_overrides(
        self,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        diff_step: Optional[float] = None,
        flat_start: Optional[bool] = None,
        bad_threshold: Optional[float] = None,
        max_remove: Any = _UNSET,
        targeted_pseudo_measurement_max: Optional[int] = None,
        targeted_pseudo_measurement_redundancy_ratio: Optional[float] = None,
        targeted_pseudo_measurement_step: Optional[int] = None,
        pseudo_measurement_weight_v: Optional[float] = None,
        pseudo_measurement_weight_pq: Optional[float] = None,
        voltage_min: Optional[float] = None,
        voltage_max: Optional[float] = None,
        angle_min_deg: Optional[float] = None,
        angle_max_deg: Optional[float] = None,
        max_remove_per_round: Any = _UNSET,
    ) -> "StateEstimationParameters":
        resolved_max_remove = (
            self.max_remove
            if max_remove is _UNSET
            else (None if max_remove is None else int(max_remove))
        )
        resolved_per_round = (
            self.max_remove_per_round
            if max_remove_per_round is _UNSET
            else (None if max_remove_per_round is None else int(max_remove_per_round))
        )
        return replace(
            self,
            tol=float(_override_value(self.tol, tol)),
            max_iter=int(_override_value(self.max_iter, max_iter)),
            diff_step=float(_override_value(self.diff_step, diff_step)),
            flat_start=bool(_override_value(self.flat_start, flat_start)),
            bad_threshold=float(_override_value(self.bad_threshold, bad_threshold)),
            max_remove=resolved_max_remove,
            targeted_pseudo_measurement_max=int(
                _override_value(self.targeted_pseudo_measurement_max, targeted_pseudo_measurement_max)
            ),
            targeted_pseudo_measurement_redundancy_ratio=float(
                _override_value(
                    self.targeted_pseudo_measurement_redundancy_ratio,
                    targeted_pseudo_measurement_redundancy_ratio,
                )
            ),
            targeted_pseudo_measurement_step=int(
                _override_value(self.targeted_pseudo_measurement_step, targeted_pseudo_measurement_step)
            ),
            pseudo_measurement_weight_v=float(
                _override_value(self.pseudo_measurement_weight_v, pseudo_measurement_weight_v)
            ),
            pseudo_measurement_weight_pq=float(
                _override_value(self.pseudo_measurement_weight_pq, pseudo_measurement_weight_pq)
            ),
            voltage_min=float(_override_value(self.voltage_min, voltage_min)),
            voltage_max=float(_override_value(self.voltage_max, voltage_max)),
            angle_min_deg=float(_override_value(self.angle_min_deg, angle_min_deg)),
            angle_max_deg=float(_override_value(self.angle_max_deg, angle_max_deg)),
            max_remove_per_round=resolved_per_round,
        )


def load_lf_parameters(path: Optional[Path] = None) -> PowerFlowParameters:
    parameter_file = resolve_project_file(path) if path is not None else DEFAULT_LF_PARAMETER_FILE
    values = _load_key_value_block(parameter_file, "PowerFlowParameter")
    return PowerFlowParameters(
        tol=float(_require(values, "tol")),
        max_iter=int(_require(values, "max_iter")),
        min_voltage=float(_require(values, "min_voltage")),
        divergence_threshold=float(_require(values, "divergence_threshold")),
    )


def load_se_parameters(path: Optional[Path] = None) -> StateEstimationParameters:
    parameter_file = resolve_project_file(path) if path is not None else DEFAULT_SE_PARAMETER_FILE
    values = _load_key_value_block(parameter_file, "StateEstimationParameter")
    raw_max_remove = values.get("max_remove")
    parsed_max_remove = (
        int(raw_max_remove)
        if (raw_max_remove is not None and str(raw_max_remove).strip() != "" and int(raw_max_remove) >= 0)
        else None
    )
    raw_max_remove_per_round = values.get("max_remove_per_round")
    parsed_max_remove_per_round = (
        int(raw_max_remove_per_round)
        if (
            raw_max_remove_per_round is not None
            and str(raw_max_remove_per_round).strip() != ""
            and int(raw_max_remove_per_round) > 0
        )
        else None
    )
    return StateEstimationParameters(
        tol=float(_require(values, "tol")),
        max_iter=int(values.get("max_iter", "200")),
        diff_step=float(_require(values, "diff_step")),
        flat_start=_to_bool(_require(values, "flat_start")),
        bad_threshold=float(_require(values, "bad_threshold")),
        max_remove=parsed_max_remove,
        targeted_pseudo_measurement_max=int(_require(values, "targeted_pseudo_measurement_max")),
        targeted_pseudo_measurement_redundancy_ratio=float(
            values.get("targeted_pseudo_measurement_redundancy_ratio", "0")
        ),
        targeted_pseudo_measurement_step=int(values.get("targeted_pseudo_measurement_step", "10")),
        voltage_floor=float(_require(values, "voltage_floor")),
        min_current_voltage=float(_require(values, "min_current_voltage")),
        power_flow_tol=float(_require(values, "power_flow_tol")),
        power_flow_max_iter=int(_require(values, "power_flow_max_iter")),
        power_flow_min_voltage=float(_require(values, "power_flow_min_voltage")),
        pseudo_measurement_weight_v=float(values.get("pseudo_measurement_weight_v", "0.1")),
        pseudo_measurement_weight_pq=float(
            values.get("pseudo_measurement_weight_pq", values.get("pseudo_measurement_weight", "0.01"))
        ),
        voltage_min=float(values.get("voltage_min", "0.1")),
        voltage_max=float(values.get("voltage_max", "1.9")),
        angle_min_deg=float(values.get("angle_min_deg", "-270.0")),
        angle_max_deg=float(values.get("angle_max_deg", "270.0")),
        max_remove_per_round=parsed_max_remove_per_round,
    )
