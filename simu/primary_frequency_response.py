"""Primary frequency response calculation for power deficit disturbances."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SystemFrequencyModel:
    f_nom_hz: float = 50.0
    s_base_mw: float = 10.0
    inertia_s: float = 4.0
    damping_mw_per_hz: float = 0.0


@dataclass(frozen=True)
class DieselGovernor:
    reserve_mw: float
    droop_mw_per_hz: float
    time_constant_s: float
    ramp_mw_per_s: float
    deadband_hz: float = 0.03
    name: str = ""


@dataclass(frozen=True)
class GridFormingStorage:
    discharge_limit_mw: float
    charge_limit_mw: float
    droop_mw_per_hz: float
    inertia_mw_s_per_hz: float = 0.0
    response_time_s: float = 0.1
    energy_mwh: float = 1.0
    initial_soc: float = 0.5
    min_soc: float = 0.1
    max_soc: float = 0.9
    deadband_hz: float = 0.01
    name: str = ""


@dataclass(frozen=True)
class Disturbance:
    start_s: float
    deficit_mw: float
    end_s: float | None = None


@dataclass(frozen=True)
class FrequencyResponseResult:
    time_s: list[float]
    frequency_hz: list[float]
    delta_frequency_hz: list[float]
    diesel_power_mw: list[float]
    storage_power_mw: list[float]
    storage_soc: list[float]
    power_deficit_mw: list[float]
    diesel_unit_names: list[str] = field(default_factory=list)
    diesel_units_power_mw: list[list[float]] = field(default_factory=list)
    storage_unit_names: list[str] = field(default_factory=list)
    storage_units_power_mw: list[list[float]] = field(default_factory=list)
    storage_units_soc: list[list[float]] = field(default_factory=list)

    @property
    def nadir_hz(self) -> float:
        return min(self.frequency_hz)

    @property
    def nadir_time_s(self) -> float:
        return self.time_s[self.frequency_hz.index(self.nadir_hz)]

    @property
    def final_frequency_hz(self) -> float:
        return self.frequency_hz[-1]

    @property
    def final_delta_frequency_hz(self) -> float:
        return self.delta_frequency_hz[-1]

    @property
    def max_diesel_power_mw(self) -> float:
        return max(self.diesel_power_mw)

    @property
    def max_storage_power_mw(self) -> float:
        return max(self.storage_power_mw)


def simulate_primary_frequency_response(
    *,
    system: SystemFrequencyModel,
    disturbance: Disturbance,
    duration_s: float,
    dt_s: float,
    diesel: DieselGovernor | None = None,
    storage: GridFormingStorage | None = None,
    diesels: list[DieselGovernor] | None = None,
    storages: list[GridFormingStorage] | None = None,
) -> FrequencyResponseResult:
    """Simulate primary frequency response after a power deficit disturbance."""

    diesel_units, storage_units = _normalize_units(diesel, storage, diesels, storages)
    _validate_inputs(system, diesel_units, storage_units, disturbance, duration_s, dt_s)
    steps = int(round(duration_s / dt_s))
    n_points = steps + 1
    f_nom = system.f_nom_hz
    f_hz = f_nom
    damping = system.damping_mw_per_hz
    dfdt_scale = f_nom / (2.0 * system.inertia_s * system.s_base_mw)
    dist_start = disturbance.start_s
    dist_end = disturbance.end_s
    dist_deficit = disturbance.deficit_mw
    dt_h = dt_s / 3600.0

    n_diesel = len(diesel_units)
    n_storage = len(storage_units)
    diesel_powers = [0.0] * n_diesel
    storage_powers = [0.0] * n_storage
    storage_socs = [unit.initial_soc for unit in storage_units]
    previous_dfdt = 0.0
    diesel_names = _unit_names(diesel_units, "diesel")
    storage_names = _unit_names(storage_units, "bess")

    time_s: list[float] = [0.0] * n_points
    frequency_hz: list[float] = [0.0] * n_points
    delta_frequency_hz: list[float] = [0.0] * n_points
    diesel_power_mw: list[float] = [0.0] * n_points
    storage_power_mw: list[float] = [0.0] * n_points
    storage_soc_values: list[float] = [0.0] * n_points
    power_deficit_mw: list[float] = [0.0] * n_points
    diesel_units_power_mw: list[list[float]] = [[0.0] * n_points for _ in range(n_diesel)]
    storage_units_power_mw: list[list[float]] = [[0.0] * n_points for _ in range(n_storage)]
    storage_units_soc: list[list[float]] = [[0.0] * n_points for _ in range(n_storage)]

    # Pre-extract per-unit constants and output buffers to avoid attribute lookups and helper calls in the hot loop.
    diesel_specs = [
        (
            u_idx,
            unit.reserve_mw,
            unit.droop_mw_per_hz,
            dt_s / unit.time_constant_s,
            unit.ramp_mw_per_s * dt_s,
            unit.deadband_hz,
            diesel_units_power_mw[u_idx],
        )
        for u_idx, unit in enumerate(diesel_units)
    ]
    storage_specs = [
        (
            u_idx,
            unit.discharge_limit_mw,
            -unit.charge_limit_mw,
            unit.droop_mw_per_hz,
            unit.inertia_mw_s_per_hz,
            1.0 if dt_s >= unit.response_time_s else dt_s / unit.response_time_s,
            unit.energy_mwh,
            unit.energy_mwh / dt_h,
            dt_h / unit.energy_mwh,
            unit.min_soc,
            unit.max_soc,
            unit.deadband_hz,
            storage_units_power_mw[u_idx],
            storage_units_soc[u_idx],
        )
        for u_idx, unit in enumerate(storage_units)
    ]

    total_storage_energy = sum(unit.energy_mwh for unit in storage_units)
    inv_total_storage_energy = (1.0 / total_storage_energy) if total_storage_energy > 0.0 else 0.0

    diesel_total = 0.0
    storage_total = 0.0
    weighted_storage_soc = (
        sum(soc * unit.energy_mwh for soc, unit in zip(storage_socs, storage_units)) * inv_total_storage_energy
        if n_storage
        else 0.0
    )

    for idx in range(n_points):
        t_s = round(idx * dt_s, 12)
        if t_s < dist_start or (dist_end is not None and t_s >= dist_end):
            deficit = 0.0
        else:
            deficit = dist_deficit

        delta_f = f_hz - f_nom
        time_s[idx] = t_s
        frequency_hz[idx] = f_hz
        delta_frequency_hz[idx] = delta_f
        diesel_power_mw[idx] = diesel_total
        storage_power_mw[idx] = storage_total
        storage_soc_values[idx] = weighted_storage_soc
        power_deficit_mw[idx] = deficit

        for u_idx in range(n_diesel):
            diesel_units_power_mw[u_idx][idx] = diesel_powers[u_idx]
        for u_idx in range(n_storage):
            storage_units_power_mw[u_idx][idx] = storage_powers[u_idx]
            storage_units_soc[u_idx][idx] = storage_socs[u_idx]

        if idx == steps:
            break

        abs_df = -delta_f if delta_f < 0.0 else delta_f

        new_diesel_total = 0.0
        for u_idx, reserve, droop, dt_over_tau, max_step, deadband, _out_p in diesel_specs:
            if abs_df <= deadband:
                eff_df = 0.0
            elif delta_f > 0.0:
                eff_df = delta_f - deadband
            else:
                eff_df = delta_f + deadband
            ref = -droop * eff_df
            if ref < 0.0:
                ref = 0.0
            elif ref > reserve:
                ref = reserve
            cur_p = diesel_powers[u_idx]
            step_p = (ref - cur_p) * dt_over_tau
            if step_p < -max_step:
                step_p = -max_step
            elif step_p > max_step:
                step_p = max_step
            new_p = cur_p + step_p
            if new_p < 0.0:
                new_p = 0.0
            elif new_p > reserve:
                new_p = reserve
            diesel_powers[u_idx] = new_p
            new_diesel_total += new_p
        diesel_total = new_diesel_total

        new_storage_total = 0.0
        weighted_soc_num = 0.0
        for (
            u_idx,
            discharge_lim,
            neg_charge_lim,
            droop,
            inertia_coeff,
            alpha,
            energy_mwh,
            energy_over_dt_h,
            dt_h_over_energy,
            min_soc,
            max_soc,
            deadband,
            _out_p,
            _out_s,
        ) in storage_specs:
            if abs_df <= deadband:
                eff_df = 0.0
            elif delta_f > 0.0:
                eff_df = delta_f - deadband
            else:
                eff_df = delta_f + deadband
            ref = -droop * eff_df - inertia_coeff * previous_dfdt
            cur_p = storage_powers[u_idx]
            p = cur_p + (ref - cur_p) * alpha
            if p < neg_charge_lim:
                p = neg_charge_lim
            elif p > discharge_lim:
                p = discharge_lim
            soc = storage_socs[u_idx]
            if p > 0.0:
                max_dis = (soc - min_soc) * energy_over_dt_h
                if max_dis < 0.0:
                    max_dis = 0.0
                if p > max_dis:
                    p = max_dis
            elif p < 0.0:
                max_chg = (max_soc - soc) * energy_over_dt_h
                if max_chg < 0.0:
                    max_chg = 0.0
                if p < -max_chg:
                    p = -max_chg
            storage_powers[u_idx] = p
            soc = soc - p * dt_h_over_energy
            if soc < min_soc:
                soc = min_soc
            elif soc > max_soc:
                soc = max_soc
            storage_socs[u_idx] = soc
            new_storage_total += p
            weighted_soc_num += soc * energy_mwh
        storage_total = new_storage_total
        weighted_storage_soc = weighted_soc_num * inv_total_storage_energy

        power_balance = diesel_total + storage_total - deficit - damping * delta_f
        previous_dfdt = dfdt_scale * power_balance
        f_hz = f_hz + previous_dfdt * dt_s

    return FrequencyResponseResult(
        time_s=time_s,
        frequency_hz=frequency_hz,
        delta_frequency_hz=delta_frequency_hz,
        diesel_power_mw=diesel_power_mw,
        storage_power_mw=storage_power_mw,
        storage_soc=storage_soc_values,
        power_deficit_mw=power_deficit_mw,
        diesel_unit_names=diesel_names,
        diesel_units_power_mw=diesel_units_power_mw,
        storage_unit_names=storage_names,
        storage_units_power_mw=storage_units_power_mw,
        storage_units_soc=storage_units_soc,
    )


def _disturbance_power(disturbance: Disturbance, t_s: float) -> float:
    if t_s < disturbance.start_s:
        return 0.0
    if disturbance.end_s is not None and t_s >= disturbance.end_s:
        return 0.0
    return disturbance.deficit_mw


def _validate_inputs(
    system: SystemFrequencyModel,
    diesels: list[DieselGovernor],
    storages: list[GridFormingStorage],
    disturbance: Disturbance,
    duration_s: float,
    dt_s: float,
) -> None:
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if duration_s < 0.0:
        raise ValueError("duration_s must be non-negative")
    if system.f_nom_hz <= 0.0:
        raise ValueError("system.f_nom_hz must be positive")
    if system.s_base_mw <= 0.0:
        raise ValueError("system.s_base_mw must be positive")
    if system.inertia_s <= 0.0:
        raise ValueError("system.inertia_s must be positive")
    if system.damping_mw_per_hz < 0.0:
        raise ValueError("system.damping_mw_per_hz must be non-negative")
    for idx, diesel in enumerate(diesels, start=1):
        prefix = f"diesels[{idx}]"
        if diesel.reserve_mw < 0.0:
            raise ValueError(f"{prefix}.reserve_mw must be non-negative")
        if diesel.droop_mw_per_hz < 0.0:
            raise ValueError(f"{prefix}.droop_mw_per_hz must be non-negative")
        if diesel.time_constant_s <= 0.0:
            raise ValueError(f"{prefix}.time_constant_s must be positive")
        if diesel.ramp_mw_per_s < 0.0:
            raise ValueError(f"{prefix}.ramp_mw_per_s must be non-negative")
        if diesel.deadband_hz < 0.0:
            raise ValueError(f"{prefix}.deadband_hz must be non-negative")
    for idx, storage in enumerate(storages, start=1):
        prefix = f"storages[{idx}]"
        if storage.discharge_limit_mw < 0.0:
            raise ValueError(f"{prefix}.discharge_limit_mw must be non-negative")
        if storage.charge_limit_mw < 0.0:
            raise ValueError(f"{prefix}.charge_limit_mw must be non-negative")
        if storage.droop_mw_per_hz < 0.0:
            raise ValueError(f"{prefix}.droop_mw_per_hz must be non-negative")
        if storage.inertia_mw_s_per_hz < 0.0:
            raise ValueError(f"{prefix}.inertia_mw_s_per_hz must be non-negative")
        if storage.response_time_s <= 0.0:
            raise ValueError(f"{prefix}.response_time_s must be positive")
        if storage.energy_mwh <= 0.0:
            raise ValueError(f"{prefix}.energy_mwh must be positive")
        if not 0.0 <= storage.min_soc <= storage.initial_soc <= storage.max_soc <= 1.0:
            raise ValueError(f"{prefix} SOC values must satisfy 0 <= min_soc <= initial_soc <= max_soc <= 1")
        if storage.deadband_hz < 0.0:
            raise ValueError(f"{prefix}.deadband_hz must be non-negative")
    if disturbance.start_s < 0.0:
        raise ValueError("disturbance.start_s must be non-negative")
    if disturbance.end_s is not None and disturbance.end_s < disturbance.start_s:
        raise ValueError("disturbance.end_s must be greater than or equal to start_s")


def _normalize_units(
    diesel: DieselGovernor | None,
    storage: GridFormingStorage | None,
    diesels: list[DieselGovernor] | None,
    storages: list[GridFormingStorage] | None,
) -> tuple[list[DieselGovernor], list[GridFormingStorage]]:
    if diesel is not None and diesels is not None:
        raise ValueError("Use either diesel or diesels, not both")
    if storage is not None and storages is not None:
        raise ValueError("Use either storage or storages, not both")
    diesel_units = list(diesels) if diesels is not None else ([diesel] if diesel is not None else [])
    storage_units = list(storages) if storages is not None else ([storage] if storage is not None else [])
    return diesel_units, storage_units


def _unit_names(units: list[DieselGovernor] | list[GridFormingStorage], prefix: str) -> list[str]:
    names: list[str] = []
    counts: dict[str, int] = {}
    for idx, unit in enumerate(units, start=1):
        base_name = unit.name.strip() if unit.name else f"{prefix}_{idx}"
        count = counts.get(base_name, 0) + 1
        counts[base_name] = count
        names.append(base_name if count == 1 else f"{base_name}_{count}")
    return names


def _weighted_storage_soc(socs: list[float], storages: list[GridFormingStorage]) -> float:
    if not socs:
        return 0.0
    total_energy = sum(storage.energy_mwh for storage in storages)
    if total_energy <= 0.0:
        return 0.0
    return sum(soc * storage.energy_mwh for soc, storage in zip(socs, storages)) / total_energy


def _apply_storage_energy_limits(
    power_mw: float,
    soc: float,
    storage: GridFormingStorage,
    dt_s: float,
) -> float:
    dt_h = dt_s / 3600.0
    if power_mw > 0.0:
        max_discharge_mw = max(0.0, (soc - storage.min_soc) * storage.energy_mwh / dt_h)
        return min(power_mw, max_discharge_mw)
    if power_mw < 0.0:
        max_charge_mw = max(0.0, (storage.max_soc - soc) * storage.energy_mwh / dt_h)
        return max(power_mw, -max_charge_mw)
    return power_mw


def _apply_deadband(delta_f_hz: float, deadband_hz: float) -> float:
    if abs(delta_f_hz) <= deadband_hz:
        return 0.0
    if delta_f_hz > 0.0:
        return delta_f_hz - deadband_hz
    return delta_f_hz + deadband_hz


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
