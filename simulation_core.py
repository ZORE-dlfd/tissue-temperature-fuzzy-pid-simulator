from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

import numpy as np

from fuzzy_rules import FuzzyScales, FuzzyTuner


def _require_finite(**values: float) -> None:
    invalid = [name for name, value in values.items() if not math.isfinite(value)]
    if invalid:
        raise ValueError(f"{', '.join(invalid)} must be finite")


@dataclass(frozen=True)
class ThermalParams:
    ambient_temp: float = 37.0
    tau: float = 45.0
    heat_gain: float = 0.30

    def __post_init__(self) -> None:
        _require_finite(
            ambient_temp=self.ambient_temp, tau=self.tau, heat_gain=self.heat_gain
        )
        if self.tau <= 0:
            raise ValueError("tau must be greater than zero")
        if self.heat_gain < 0:
            raise ValueError("heat_gain cannot be negative")


def thermal_step(
    temperature: float,
    power: float,
    dt: float,
    params: ThermalParams,
) -> float:
    _require_finite(temperature=temperature, power=power, dt=dt)
    if dt <= 0:
        raise ValueError("dt must be greater than zero")
    if not 0.0 <= power <= 1.0:
        raise ValueError("power must be between 0 and 1")
    decay = math.exp(-dt / params.tau)
    equilibrium = params.ambient_temp + params.tau * params.heat_gain * power
    return equilibrium + (temperature - equilibrium) * decay


def rmse(values: np.ndarray, reference: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if values.shape != reference.shape or values.size == 0:
        raise ValueError("rmse inputs must be non-empty arrays with equal shapes")
    return float(np.sqrt(np.mean(np.square(values - reference))))


class ScalarThermalKalman:
    def __init__(
        self,
        initial_temp: float,
        initial_variance: float,
        process_variance: float,
        measurement_variance: float,
    ) -> None:
        _require_finite(
            initial_temp=initial_temp,
            initial_variance=initial_variance,
            process_variance=process_variance,
            measurement_variance=measurement_variance,
        )
        if min(initial_variance, process_variance, measurement_variance) < 0:
            raise ValueError("Kalman variances cannot be negative")
        if measurement_variance == 0:
            raise ValueError("measurement_variance must be greater than zero")
        self.temperature = float(initial_temp)
        self.variance = float(initial_variance)
        self.process_variance = float(process_variance)
        self.measurement_variance = float(measurement_variance)

    def predict(self, power: float, dt: float, params: ThermalParams) -> float:
        transition = math.exp(-dt / params.tau)
        self.temperature = thermal_step(self.temperature, power, dt, params)
        self.variance = transition * transition * self.variance + self.process_variance
        return self.temperature

    def update(self, measurement: float) -> float:
        gain = self.variance / (self.variance + self.measurement_variance)
        self.temperature += gain * (measurement - self.temperature)
        self.variance *= 1.0 - gain
        return self.temperature


@dataclass(frozen=True)
class PIDConfig:
    kp: float = 0.18
    ki: float = 0.012
    kd: float = 0.20
    output_min: float = 0.0
    output_max: float = 1.0
    derivative_filter: float = 0.85
    integral_limit: float = 200.0

    def __post_init__(self) -> None:
        _require_finite(
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            output_min=self.output_min,
            output_max=self.output_max,
            derivative_filter=self.derivative_filter,
            integral_limit=self.integral_limit,
        )
        if min(self.kp, self.ki, self.kd) < 0:
            raise ValueError("PID gains cannot be negative")
        if not 0.0 <= self.output_min < self.output_max <= 1.0:
            raise ValueError("PID output limits must stay between 0 and 1")
        if not 0.0 <= self.derivative_filter < 1.0:
            raise ValueError("derivative_filter must be in [0, 1)")
        if self.integral_limit <= 0:
            raise ValueError("integral_limit must be greater than zero")


class PIDController:
    def __init__(self, config: PIDConfig) -> None:
        self.config = config
        self.integral = 0.0
        self.previous_measurement: float | None = None
        self.filtered_derivative = 0.0
        self.last_gains = (config.kp, config.ki, config.kd)

    def update(
        self,
        error: float,
        dt: float,
        measurement: float | None = None,
        gains: tuple[float, float, float] | None = None,
    ) -> float:
        if dt <= 0:
            raise ValueError("dt must be greater than zero")
        kp, ki, kd = gains or (self.config.kp, self.config.ki, self.config.kd)
        self.last_gains = (kp, ki, kd)

        derivative = 0.0
        if measurement is not None and self.previous_measurement is not None:
            derivative = -(measurement - self.previous_measurement) / dt
        self.previous_measurement = measurement
        alpha = self.config.derivative_filter
        self.filtered_derivative = alpha * self.filtered_derivative + (1.0 - alpha) * derivative

        candidate_integral = self.integral + error * dt
        candidate_integral = min(
            self.config.integral_limit,
            max(-self.config.integral_limit, candidate_integral),
        )
        candidate = kp * error + ki * candidate_integral + kd * self.filtered_derivative
        drives_further_high = candidate > self.config.output_max and error > 0
        drives_further_low = candidate < self.config.output_min and error < 0
        if not (drives_further_high or drives_further_low):
            self.integral = candidate_integral

        output = kp * error + ki * self.integral + kd * self.filtered_derivative
        return min(self.config.output_max, max(self.config.output_min, output))


class FuzzyPIDController(PIDController):
    def __init__(self, config: PIDConfig, scales: FuzzyScales) -> None:
        super().__init__(config)
        self.tuner = FuzzyTuner(scales)
        self.previous_error: float | None = None

    def update(
        self,
        error: float,
        dt: float,
        measurement: float | None = None,
        gains: tuple[float, float, float] | None = None,
    ) -> float:
        if dt <= 0:
            raise ValueError("dt must be greater than zero")
        error_rate = 0.0 if self.previous_error is None else (error - self.previous_error) / dt
        self.previous_error = error
        dkp, dki, dkd = self.tuner.corrections(error, error_rate)
        adaptive = (
            min(self.config.kp + self.tuner.scales.delta_kp_max, max(0.0, self.config.kp + dkp)),
            min(self.config.ki + self.tuner.scales.delta_ki_max, max(0.0, self.config.ki + dki)),
            min(self.config.kd + self.tuner.scales.delta_kd_max, max(0.0, self.config.kd + dkd)),
        )
        return super().update(error, dt, measurement, adaptive)


@dataclass(frozen=True)
class SimulationConfig:
    duration: float = 180.0
    dt: float = 0.1
    initial_temp: float = 37.0
    target_temp: float = 43.0
    measurement_noise_std: float = 0.45
    kalman_q: float = 0.0025
    kalman_r: float = 0.2025
    kalman_initial_variance: float = 1.0
    use_kalman: bool = True
    controller_mode: str = "fuzzy"
    random_seed: int = 20260817
    thermal: ThermalParams = field(default_factory=ThermalParams)
    pid: PIDConfig = field(default_factory=PIDConfig)
    fuzzy: FuzzyScales = field(default_factory=FuzzyScales)

    def __post_init__(self) -> None:
        _require_finite(
            duration=self.duration,
            dt=self.dt,
            initial_temp=self.initial_temp,
            target_temp=self.target_temp,
            measurement_noise_std=self.measurement_noise_std,
            kalman_q=self.kalman_q,
            kalman_r=self.kalman_r,
            kalman_initial_variance=self.kalman_initial_variance,
        )
        if self.duration <= 0 or self.dt <= 0:
            raise ValueError("duration and dt must be greater than zero")
        step_ratio = self.duration / self.dt
        rounded_steps = round(step_ratio)
        if not math.isclose(step_ratio, rounded_steps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("duration must be an integer multiple of dt")
        if rounded_steps + 1 > 200_001:
            raise ValueError("simulation steps exceed 200001")
        if self.measurement_noise_std < 0 or self.kalman_q < 0 or self.kalman_r <= 0:
            raise ValueError("noise parameters are outside valid ranges")
        if self.controller_mode not in {"classic", "fuzzy"}:
            raise ValueError("controller_mode must be classic or fuzzy")
        if not 35.0 <= self.initial_temp <= 45.0:
            raise ValueError("initial_temp must be between 35 and 45 C")
        if not 37.0 <= self.target_temp <= 55.0:
            raise ValueError("target_temp must be between 37 and 55 C")
        if self.target_temp < self.initial_temp:
            raise ValueError("target_temp must not be below initial_temp in warming mode")


@dataclass(frozen=True)
class SimulationResult:
    time: np.ndarray
    target_temp: np.ndarray
    true_temp: np.ndarray
    measurement_temp: np.ndarray
    estimated_temp: np.ndarray
    feedback_temp: np.ndarray
    error: np.ndarray
    power: np.ndarray
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray
    config: SimulationConfig


def run_simulation(
    config: SimulationConfig,
    noise_sequence: np.ndarray | None = None,
) -> SimulationResult:
    steps = int(round(config.duration / config.dt)) + 1
    time = np.arange(steps, dtype=float) * config.dt
    target = np.full(steps, config.target_temp)
    true_temp = np.empty(steps)
    measurement = np.empty(steps)
    estimate = np.empty(steps)
    feedback = np.empty(steps)
    error = np.empty(steps)
    power = np.zeros(steps)
    kp = np.empty(steps)
    ki = np.empty(steps)
    kd = np.empty(steps)

    if noise_sequence is None:
        rng = np.random.default_rng(config.random_seed)
        noise = rng.normal(0.0, config.measurement_noise_std, steps)
    else:
        noise = np.asarray(noise_sequence, dtype=float)
        if noise.shape != (steps,):
            raise ValueError("noise_sequence length does not match simulation steps")
    if not np.all(np.isfinite(noise)):
        raise ValueError("noise_sequence must contain only finite values")

    controller: PIDController
    if config.controller_mode == "fuzzy":
        controller = FuzzyPIDController(config.pid, config.fuzzy)
    else:
        controller = PIDController(config.pid)
    kalman = ScalarThermalKalman(
        config.initial_temp,
        config.kalman_initial_variance,
        config.kalman_q,
        config.kalman_r,
    )

    true_temp[0] = config.initial_temp
    measurement[0] = true_temp[0] + noise[0]
    estimate[0] = kalman.update(measurement[0])
    feedback[0] = estimate[0] if config.use_kalman else measurement[0]
    error[0] = target[0] - feedback[0]
    power[0] = controller.update(error[0], config.dt, feedback[0])
    kp[0], ki[0], kd[0] = controller.last_gains

    for index in range(1, steps):
        true_temp[index] = thermal_step(
            true_temp[index - 1], power[index - 1], config.dt, config.thermal
        )
        measurement[index] = true_temp[index] + noise[index]
        kalman.predict(power[index - 1], config.dt, config.thermal)
        estimate[index] = kalman.update(measurement[index])
        feedback[index] = estimate[index] if config.use_kalman else measurement[index]
        error[index] = target[index] - feedback[index]
        power[index] = controller.update(error[index], config.dt, feedback[index])
        kp[index], ki[index], kd[index] = controller.last_gains

    values = (true_temp, measurement, estimate, feedback, error, power, kp, ki, kd)
    if not all(np.all(np.isfinite(value)) for value in values):
        raise FloatingPointError("simulation produced a non-finite value")

    return SimulationResult(
        time,
        target,
        true_temp,
        measurement,
        estimate,
        feedback,
        error,
        power,
        kp,
        ki,
        kd,
        config,
    )


def calculate_metrics(result: SimulationResult) -> dict[str, float | None]:
    target_rise = result.config.target_temp - result.config.initial_temp
    peak = float(np.max(result.true_temp))
    overshoot_c = max(0.0, peak - result.config.target_temp)
    overshoot_percent = 0.0 if target_rise <= 0 else 100.0 * overshoot_c / target_rise
    threshold = result.config.initial_temp + 0.9 * target_rise
    reached = np.flatnonzero(result.true_temp >= threshold)
    rise_time = None if reached.size == 0 else float(result.time[reached[0]])
    tail_count = max(1, int(round(0.2 * result.time.size)))
    steady_error = float(
        np.mean(result.target_temp[-tail_count:] - result.true_temp[-tail_count:])
    )
    control_mae = float(np.mean(np.abs(result.target_temp - result.true_temp)))
    return {
        "overshoot_c": overshoot_c,
        "overshoot_percent": overshoot_percent,
        "rise_time_s": rise_time,
        "steady_state_error_c": steady_error,
        "control_mae_c": control_mae,
        "measurement_rmse_c": rmse(result.measurement_temp, result.true_temp),
        "kalman_rmse_c": rmse(result.estimated_temp, result.true_temp),
        "normalized_energy_s": float(
            np.sum(result.power[:-1], dtype=float) * result.config.dt
        ),
    }


def export_data_files(result: SimulationResult, directory: Path) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "timeseries.csv"
    json_path = directory / "parameters.json"
    readme_path = directory / "README.txt"
    existing = (csv_path, json_path, readme_path)
    if any(path.exists() for path in existing):
        raise FileExistsError("export directory already contains result files")
    header = [
        "time_s",
        "target_temp_c",
        "true_temp_c",
        "measurement_temp_c",
        "estimated_temp_c",
        "feedback_temp_c",
        "error_c",
        "power_percent",
        "kp",
        "ki",
        "kd",
    ]
    columns = [
        result.time,
        result.target_temp,
        result.true_temp,
        result.measurement_temp,
        result.estimated_temp,
        result.feedback_temp,
        result.error,
        result.power * 100.0,
        result.kp,
        result.ki,
        result.kd,
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(zip(*columns, strict=True))

    payload = {
        "provenance": "教学仿真参数，不是论文原始实验参数",
        "config": asdict(result.config),
        "metrics": calculate_metrics(result),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    readme_path.write_text(
        "本结果来自一阶集总组织热模型。默认 PID、模糊规则和热模型参数"
        "是教学仿真假设，不是论文原始实验参数，也不能替代真实治疗实验。\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "json": json_path, "readme": readme_path}
