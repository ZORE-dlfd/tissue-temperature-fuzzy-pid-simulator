# Tissue Temperature Fuzzy PID Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows desktop simulator that combines a first-order tissue thermal model, classical PID, fuzzy self-tuning PID, noisy thermometry, model-based Kalman estimation, result metrics, and GitHub-ready exports.

**Architecture:** Keep all deterministic numerical behavior in `simulation_core.py` and all fuzzy inference data in `fuzzy_rules.py`. `app.py` owns only Tkinter widgets, Matplotlib drawing, input collection, and user-facing error messages. Tests call the numerical API directly without creating a GUI, while one final manual smoke test verifies the real window.

**Tech Stack:** Python 3.13, NumPy, Matplotlib, Tkinter 8.6, standard-library `unittest`, CSV and JSON, Git.

---

## File map

Files created under `C:\Users\老白给\Desktop\PID仿真`:

- `simulation_core.py`: parameter dataclasses, thermal step, PID controllers, Kalman filter, simulation loop, metrics, validation, and CSV/JSON/text export.
- `fuzzy_rules.py`: five-set membership calculation, 5x5 rule matrices, bounded gain corrections.
- `app.py`: Tkinter controls, embedded four-panel Matplotlib figure, comparison table, and export command.
- `requirements.txt`: NumPy and Matplotlib versions.
- `run_simulator.bat`: one-click virtual-environment startup.
- `README.md`: model explanation, setup, controls, limitations, and result interpretation.
- `outputs/.gitkeep`: preserves the export directory without committing generated runs.
- `tests/test_thermal_model.py`: heating and cooling behavior.
- `tests/test_kalman.py`: scalar prediction/update and noise reduction.
- `tests/test_pid.py`: saturation, anti-windup, and derivative filtering.
- `tests/test_fuzzy_pid.py`: membership, rule direction, and gain bounds.
- `tests/test_simulation.py`: repeatability, feedback selection, comparison fairness, and metrics.
- `tests/test_export.py`: exported schema, provenance warning, and PNG readability.
- `docs/images/example_result.png`: verified example figure for the GitHub README.

## Task 1: Bootstrap the Python project

**Files:**
- Create: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `outputs/.gitkeep`

- [ ] **Step 1: Create dependency declarations**

Write `requirements.txt`:

```text
numpy>=2.1,<3
matplotlib>=3.10,<4
```

- [ ] **Step 2: Create package markers**

Create an empty `tests/__init__.py` and an empty `outputs/.gitkeep`.

- [ ] **Step 3: Create and populate the virtual environment**

Run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Expected: installation exits with code 0 and installs NumPy and Matplotlib.

- [ ] **Step 4: Verify imports and Tkinter**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import numpy, matplotlib, tkinter; print(numpy.__version__, matplotlib.__version__, tkinter.TkVersion)"
```

Expected: one line containing NumPy 2.x, Matplotlib 3.10.x or later, and Tk 8.6.

- [ ] **Step 5: Commit the bootstrap**

```powershell
git add requirements.txt tests/__init__.py outputs/.gitkeep
git commit -m "chore: bootstrap simulator environment"
```

## Task 2: Implement the first-order tissue thermal model

**Files:**
- Create: `tests/test_thermal_model.py`
- Create: `simulation_core.py`

- [ ] **Step 1: Write failing thermal-model tests**

Create `tests/test_thermal_model.py`:

```python
import unittest

from simulation_core import ThermalParams, thermal_step


class ThermalModelTests(unittest.TestCase):
    def test_hot_tissue_cools_toward_ambient_without_laser(self):
        params = ThermalParams(ambient_temp=37.0, tau=40.0, heat_gain=0.25)
        next_temp = thermal_step(45.0, power=0.0, dt=1.0, params=params)
        self.assertLess(next_temp, 45.0)
        self.assertGreater(next_temp, 37.0)

    def test_positive_laser_power_heats_tissue_at_ambient(self):
        params = ThermalParams(ambient_temp=37.0, tau=40.0, heat_gain=0.25)
        next_temp = thermal_step(37.0, power=0.5, dt=1.0, params=params)
        self.assertAlmostEqual(next_temp, 37.125, places=6)

    def test_invalid_time_constant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tau"):
            ThermalParams(ambient_temp=37.0, tau=0.0, heat_gain=0.25)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_thermal_model -v
```

Expected: FAIL or ERROR because `simulation_core` does not exist.

- [ ] **Step 3: Add the minimal thermal implementation**

Create `simulation_core.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalParams:
    ambient_temp: float = 37.0
    tau: float = 45.0
    heat_gain: float = 0.30

    def __post_init__(self) -> None:
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
    if dt <= 0:
        raise ValueError("dt must be greater than zero")
    if not 0.0 <= power <= 1.0:
        raise ValueError("power must be between 0 and 1")
    derivative = -((temperature - params.ambient_temp) / params.tau)
    derivative += params.heat_gain * power
    return temperature + dt * derivative
```

- [ ] **Step 4: Run the thermal tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_thermal_model -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit the thermal model**

```powershell
git add simulation_core.py tests/test_thermal_model.py
git commit -m "feat: add first-order tissue thermal model"
```

## Task 3: Implement model-based scalar Kalman estimation

**Files:**
- Create: `tests/test_kalman.py`
- Modify: `simulation_core.py`

- [ ] **Step 1: Write failing Kalman tests**

Create `tests/test_kalman.py`:

```python
import unittest

import numpy as np

from simulation_core import ScalarThermalKalman, ThermalParams, rmse


class KalmanTests(unittest.TestCase):
    def test_measurement_update_moves_estimate_toward_measurement(self):
        params = ThermalParams()
        kalman = ScalarThermalKalman(
            initial_temp=37.0,
            initial_variance=1.0,
            process_variance=0.01,
            measurement_variance=0.25,
        )
        predicted = kalman.predict(power=0.0, dt=0.1, params=params)
        updated = kalman.update(measurement=38.0)
        self.assertGreater(updated, predicted)
        self.assertLess(updated, 38.0)

    def test_filter_reduces_rmse_for_constant_temperature(self):
        rng = np.random.default_rng(20260817)
        truth = np.full(300, 42.0)
        measurements = truth + rng.normal(0.0, 0.5, truth.size)
        kalman = ScalarThermalKalman(42.0, 1.0, 0.001, 0.25)
        estimates = []
        for value in measurements:
            kalman.predict(power=0.0, dt=0.1, params=ThermalParams(42.0, 45.0, 0.0))
            estimates.append(kalman.update(value))
        self.assertLess(rmse(np.asarray(estimates), truth), rmse(measurements, truth))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify the new tests fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_kalman -v
```

Expected: FAIL because `ScalarThermalKalman` and `rmse` are missing.

- [ ] **Step 3: Add Kalman and RMSE behavior**

Append to `simulation_core.py`:

```python
import numpy as np


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
        if min(initial_variance, process_variance, measurement_variance) < 0:
            raise ValueError("Kalman variances cannot be negative")
        if measurement_variance == 0:
            raise ValueError("measurement_variance must be greater than zero")
        self.temperature = float(initial_temp)
        self.variance = float(initial_variance)
        self.process_variance = float(process_variance)
        self.measurement_variance = float(measurement_variance)

    def predict(self, power: float, dt: float, params: ThermalParams) -> float:
        transition = 1.0 - dt / params.tau
        self.temperature = thermal_step(self.temperature, power, dt, params)
        self.variance = transition * transition * self.variance + self.process_variance
        return self.temperature

    def update(self, measurement: float) -> float:
        gain = self.variance / (self.variance + self.measurement_variance)
        self.temperature += gain * (measurement - self.temperature)
        self.variance *= 1.0 - gain
        return self.temperature
```

- [ ] **Step 4: Run Kalman and thermal tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_thermal_model tests.test_kalman -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit the estimator**

```powershell
git add simulation_core.py tests/test_kalman.py
git commit -m "feat: add model-based Kalman temperature estimator"
```

## Task 4: Implement bounded classical PID control

**Files:**
- Create: `tests/test_pid.py`
- Modify: `simulation_core.py`

- [ ] **Step 1: Write failing PID tests**

Create `tests/test_pid.py`:

```python
import unittest

from simulation_core import PIDConfig, PIDController


class PIDTests(unittest.TestCase):
    def test_output_is_limited_to_laser_range(self):
        controller = PIDController(PIDConfig(kp=2.0, ki=0.0, kd=0.0))
        self.assertEqual(controller.update(10.0, 0.1), 1.0)
        self.assertEqual(controller.update(-10.0, 0.1), 0.0)

    def test_integral_stops_growing_during_positive_saturation(self):
        controller = PIDController(PIDConfig(kp=1.0, ki=1.0, kd=0.0))
        for _ in range(100):
            controller.update(10.0, 0.1)
        self.assertAlmostEqual(controller.integral, 0.0, places=8)

    def test_derivative_filter_rejects_invalid_coefficient(self):
        with self.assertRaisesRegex(ValueError, "derivative_filter"):
            PIDConfig(kp=1.0, ki=0.1, kd=0.1, derivative_filter=1.5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify PID tests fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_pid -v
```

Expected: FAIL because PID classes are missing.

- [ ] **Step 3: Add PID configuration and controller**

Append to `simulation_core.py`:

```python
@dataclass(frozen=True)
class PIDConfig:
    kp: float = 0.18
    ki: float = 0.012
    kd: float = 0.20
    output_min: float = 0.0
    output_max: float = 1.0
    derivative_filter: float = 0.85

    def __post_init__(self) -> None:
        if min(self.kp, self.ki, self.kd) < 0:
            raise ValueError("PID gains cannot be negative")
        if self.output_min >= self.output_max:
            raise ValueError("output_min must be less than output_max")
        if not 0.0 <= self.derivative_filter < 1.0:
            raise ValueError("derivative_filter must be in [0, 1)")


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
        candidate = kp * error + ki * candidate_integral + kd * self.filtered_derivative
        saturated = min(self.config.output_max, max(self.config.output_min, candidate))
        drives_further_high = candidate > self.config.output_max and error > 0
        drives_further_low = candidate < self.config.output_min and error < 0
        if not (drives_further_high or drives_further_low):
            self.integral = candidate_integral
        output = kp * error + ki * self.integral + kd * self.filtered_derivative
        return min(self.config.output_max, max(self.config.output_min, output))
```

- [ ] **Step 4: Run PID tests and the full current suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit classical PID**

```powershell
git add simulation_core.py tests/test_pid.py
git commit -m "feat: add bounded PID laser controller"
```

## Task 5: Implement explainable fuzzy gain scheduling

**Files:**
- Create: `tests/test_fuzzy_pid.py`
- Create: `fuzzy_rules.py`
- Modify: `simulation_core.py`

- [ ] **Step 1: Write failing fuzzy-rule tests**

Create `tests/test_fuzzy_pid.py`:

```python
import unittest

from fuzzy_rules import FuzzyScales, FuzzyTuner, memberships
from simulation_core import FuzzyPIDController, PIDConfig


class FuzzyPIDTests(unittest.TestCase):
    def test_memberships_sum_to_one_in_normalized_domain(self):
        for value in (-1.0, -0.75, -0.2, 0.0, 0.4, 0.9, 1.0):
            self.assertAlmostEqual(sum(memberships(value)), 1.0, places=8)

    def test_large_error_increases_proportional_gain_and_reduces_integral(self):
        tuner = FuzzyTuner(FuzzyScales())
        delta_kp, delta_ki, _ = tuner.corrections(error=6.0, error_rate=0.0)
        self.assertGreater(delta_kp, 0.0)
        self.assertLess(delta_ki, 0.0)

    def test_online_gains_remain_inside_configured_bounds(self):
        controller = FuzzyPIDController(PIDConfig(), FuzzyScales())
        for error in (-100.0, -5.0, 0.0, 5.0, 100.0):
            controller.update(error, 0.1, measurement=37.0)
            kp, ki, kd = controller.last_gains
            self.assertGreaterEqual(kp, 0.0)
            self.assertGreaterEqual(ki, 0.0)
            self.assertGreaterEqual(kd, 0.0)
            self.assertLessEqual(kp, 1.0)
            self.assertLessEqual(ki, 0.2)
            self.assertLessEqual(kd, 2.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify fuzzy tests fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_fuzzy_pid -v
```

Expected: FAIL because fuzzy modules are missing.

- [ ] **Step 3: Implement membership and rule inference**

Create `fuzzy_rules.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CENTERS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

DKP_RULES = np.array([
    [1.0, 1.0, 0.8, 0.6, 0.4],
    [0.8, 0.6, 0.4, 0.2, 0.0],
    [-0.2, -0.3, -0.4, -0.3, -0.2],
    [0.0, 0.2, 0.4, 0.6, 0.8],
    [0.4, 0.6, 0.8, 1.0, 1.0],
])

DKI_RULES = np.array([
    [-1.0, -1.0, -0.8, -1.0, -1.0],
    [-0.6, -0.4, -0.2, -0.4, -0.6],
    [0.5, 0.8, 1.0, 0.8, 0.5],
    [-0.6, -0.4, -0.2, -0.4, -0.6],
    [-1.0, -1.0, -0.8, -1.0, -1.0],
])

DKD_RULES = np.array([
    [1.0, 0.8, 0.5, 0.0, -0.2],
    [0.8, 0.6, 0.3, 0.0, -0.2],
    [0.2, 0.1, 0.0, 0.1, 0.2],
    [-0.2, 0.0, 0.3, 0.6, 0.8],
    [-0.2, 0.0, 0.5, 0.8, 1.0],
])


@dataclass(frozen=True)
class FuzzyScales:
    error_scale: float = 6.0
    error_rate_scale: float = 0.5
    delta_kp_max: float = 0.08
    delta_ki_max: float = 0.006
    delta_kd_max: float = 0.12

    def __post_init__(self) -> None:
        if self.error_scale <= 0 or self.error_rate_scale <= 0:
            raise ValueError("fuzzy input scales must be greater than zero")
        if min(self.delta_kp_max, self.delta_ki_max, self.delta_kd_max) < 0:
            raise ValueError("fuzzy output scales cannot be negative")


def memberships(value: float) -> np.ndarray:
    value = float(np.clip(value, -1.0, 1.0))
    weights = np.maximum(0.0, 1.0 - np.abs(value - CENTERS) / 0.5)
    if value <= -1.0:
        weights[0] = 1.0
    if value >= 1.0:
        weights[-1] = 1.0
    return weights / weights.sum()


class FuzzyTuner:
    def __init__(self, scales: FuzzyScales) -> None:
        self.scales = scales

    def corrections(self, error: float, error_rate: float) -> tuple[float, float, float]:
        error_weights = memberships(error / self.scales.error_scale)
        rate_weights = memberships(error_rate / self.scales.error_rate_scale)
        firing = np.outer(error_weights, rate_weights)
        denominator = firing.sum()
        dkp = float((firing * DKP_RULES).sum() / denominator) * self.scales.delta_kp_max
        dki = float((firing * DKI_RULES).sum() / denominator) * self.scales.delta_ki_max
        dkd = float((firing * DKD_RULES).sum() / denominator) * self.scales.delta_kd_max
        return dkp, dki, dkd
```

- [ ] **Step 4: Add the fuzzy PID wrapper**

Append to `simulation_core.py`:

```python
from fuzzy_rules import FuzzyScales, FuzzyTuner


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
        error_rate = 0.0 if self.previous_error is None else (error - self.previous_error) / dt
        self.previous_error = error
        dkp, dki, dkd = self.tuner.corrections(error, error_rate)
        adaptive = (
            min(1.0, max(0.0, self.config.kp + dkp)),
            min(0.2, max(0.0, self.config.ki + dki)),
            min(2.0, max(0.0, self.config.kd + dkd)),
        )
        return super().update(error, dt, measurement, adaptive)
```

- [ ] **Step 5: Run the fuzzy tests and full suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: 11 tests pass.

- [ ] **Step 6: Commit fuzzy gain scheduling**

```powershell
git add fuzzy_rules.py simulation_core.py tests/test_fuzzy_pid.py
git commit -m "feat: add explainable fuzzy PID gain scheduling"
```

## Task 6: Integrate the closed-loop simulation and metrics

**Files:**
- Create: `tests/test_simulation.py`
- Modify: `simulation_core.py`

- [ ] **Step 1: Write failing integration tests**

Create `tests/test_simulation.py`:

```python
import unittest

import numpy as np

from simulation_core import SimulationConfig, calculate_metrics, run_simulation


class SimulationTests(unittest.TestCase):
    def test_fixed_seed_makes_simulation_reproducible(self):
        config = SimulationConfig(duration=20.0, dt=0.1, random_seed=17)
        first = run_simulation(config)
        second = run_simulation(config)
        np.testing.assert_array_equal(first.measurement_temp, second.measurement_temp)
        np.testing.assert_array_equal(first.true_temp, second.true_temp)

    def test_kalman_feedback_uses_estimate_instead_of_measurement(self):
        config = SimulationConfig(duration=5.0, dt=0.1, use_kalman=True)
        result = run_simulation(config)
        np.testing.assert_array_equal(result.feedback_temp, result.estimated_temp)

    def test_raw_feedback_uses_measurement_when_kalman_is_disabled(self):
        config = SimulationConfig(duration=5.0, dt=0.1, use_kalman=False)
        result = run_simulation(config)
        np.testing.assert_array_equal(result.feedback_temp, result.measurement_temp)

    def test_default_kalman_estimate_beats_raw_measurement_rmse(self):
        result = run_simulation(SimulationConfig())
        metrics = calculate_metrics(result)
        self.assertLess(metrics["kalman_rmse_c"], metrics["measurement_rmse_c"])

    def test_excessive_number_of_steps_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "steps"):
            SimulationConfig(duration=10000.0, dt=0.0001)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify integration tests fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_simulation -v
```

Expected: FAIL because simulation dataclasses and functions are missing.

- [ ] **Step 3: Add simulation parameter and result dataclasses**

Add to `simulation_core.py`:

```python
from dataclasses import asdict, field


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
        if self.duration <= 0 or self.dt <= 0:
            raise ValueError("duration and dt must be greater than zero")
        if int(round(self.duration / self.dt)) + 1 > 200_001:
            raise ValueError("simulation steps exceed 200001")
        if self.measurement_noise_std < 0 or self.kalman_q < 0 or self.kalman_r <= 0:
            raise ValueError("noise parameters are outside valid ranges")
        if self.controller_mode not in {"classic", "fuzzy"}:
            raise ValueError("controller_mode must be classic or fuzzy")
        if not 35.0 <= self.initial_temp <= 45.0:
            raise ValueError("initial_temp must be between 35 and 45 C")
        if not 37.0 <= self.target_temp <= 55.0:
            raise ValueError("target_temp must be between 37 and 55 C")


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
```

- [ ] **Step 4: Implement the deterministic simulation loop**

Add to `simulation_core.py`:

```python
def run_simulation(
    config: SimulationConfig,
    noise_sequence: np.ndarray | None = None,
) -> SimulationResult:
    steps = int(round(config.duration / config.dt)) + 1
    time = np.linspace(0.0, config.duration, steps)
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

    return SimulationResult(
        time, target, true_temp, measurement, estimate, feedback,
        error, power, kp, ki, kd, config
    )
```

- [ ] **Step 5: Implement metrics with explicit edge behavior**

Add to `simulation_core.py`:

```python
def calculate_metrics(result: SimulationResult) -> dict[str, float | None]:
    target_rise = result.config.target_temp - result.config.initial_temp
    peak = float(np.max(result.true_temp))
    overshoot_c = max(0.0, peak - result.config.target_temp)
    overshoot_percent = 0.0 if target_rise <= 0 else 100.0 * overshoot_c / target_rise
    threshold = result.config.initial_temp + 0.9 * target_rise
    reached = np.flatnonzero(result.true_temp >= threshold)
    rise_time = None if reached.size == 0 else float(result.time[reached[0]])
    tail_count = max(1, int(round(0.2 * result.time.size)))
    steady_error = float(np.mean(result.target_temp[-tail_count:] - result.true_temp[-tail_count:]))
    control_mae = float(np.mean(np.abs(result.target_temp - result.true_temp)))
    return {
        "overshoot_c": overshoot_c,
        "overshoot_percent": overshoot_percent,
        "rise_time_s": rise_time,
        "steady_state_error_c": steady_error,
        "control_mae_c": control_mae,
        "measurement_rmse_c": rmse(result.measurement_temp, result.true_temp),
        "kalman_rmse_c": rmse(result.estimated_temp, result.true_temp),
        "normalized_energy_s": float(np.trapezoid(result.power, result.time)),
    }
```

- [ ] **Step 6: Run integration and full tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: 16 tests pass.

- [ ] **Step 7: Inspect default controller behavior numerically**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from simulation_core import *; r=run_simulation(SimulationConfig()); print(calculate_metrics(r)); print(r.true_temp[-1], r.power.min(), r.power.max())"
```

Expected: arrays stay finite, power stays inside 0 to 1, the final temperature approaches 43 °C, and Kalman RMSE is below measurement RMSE. If the default PID is unstable or cannot approach the target, add a failing behavior test before changing the default gains.

- [ ] **Step 8: Commit simulation integration**

```powershell
git add simulation_core.py tests/test_simulation.py
git commit -m "feat: integrate closed-loop thermal simulation"
```

## Task 7: Implement validated result export

**Files:**
- Create: `tests/test_export.py`
- Modify: `simulation_core.py`

- [ ] **Step 1: Write failing export tests**

Create `tests/test_export.py`:

```python
import csv
import json
import tempfile
import unittest
from pathlib import Path

from simulation_core import SimulationConfig, export_data_files, run_simulation


class ExportTests(unittest.TestCase):
    def test_export_contains_required_data_and_provenance_warning(self):
        result = run_simulation(SimulationConfig(duration=2.0, dt=0.1))
        with tempfile.TemporaryDirectory() as directory:
            paths = export_data_files(result, Path(directory))
            self.assertTrue(paths["csv"].exists())
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["readme"].exists())
            with paths["csv"].open(encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle))
            self.assertEqual(
                header,
                ["time_s", "target_temp_c", "true_temp_c", "measurement_temp_c",
                 "estimated_temp_c", "feedback_temp_c", "error_c", "power_percent",
                 "kp", "ki", "kd"],
            )
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertIn("metrics", payload)
            warning = paths["readme"].read_text(encoding="utf-8")
            self.assertIn("不是论文原始实验参数", warning)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify export tests fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_export -v
```

Expected: FAIL because `export_data_files` is missing.

- [ ] **Step 3: Add CSV, JSON, and warning export**

Add imports and function to `simulation_core.py`:

```python
import csv
import json
from pathlib import Path


def _config_to_dict(config: SimulationConfig) -> dict[str, object]:
    return asdict(config)


def export_data_files(result: SimulationResult, directory: Path) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "timeseries.csv"
    json_path = directory / "parameters.json"
    readme_path = directory / "README.txt"
    header = [
        "time_s", "target_temp_c", "true_temp_c", "measurement_temp_c",
        "estimated_temp_c", "feedback_temp_c", "error_c", "power_percent",
        "kp", "ki", "kd",
    ]
    columns = [
        result.time, result.target_temp, result.true_temp, result.measurement_temp,
        result.estimated_temp, result.feedback_temp, result.error,
        result.power * 100.0, result.kp, result.ki, result.kd,
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(zip(*columns, strict=True))
    payload = {
        "provenance": "教学仿真参数，不是论文原始实验参数",
        "config": _config_to_dict(result.config),
        "metrics": calculate_metrics(result),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    readme_path.write_text(
        "本结果来自一阶集总组织热模型。默认 PID、模糊规则和热模型参数"
        "是教学仿真假设，不是论文原始实验参数，也不能替代真实治疗实验。\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "json": json_path, "readme": readme_path}
```

- [ ] **Step 4: Run export and full tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: 17 tests pass.

- [ ] **Step 5: Commit data export**

```powershell
git add simulation_core.py tests/test_export.py
git commit -m "feat: export simulation data and parameter provenance"
```

## Task 8: Build the Tkinter and Matplotlib desktop application

**Files:**
- Create: `app.py`

- [ ] **Step 1: Add a headless import smoke test before GUI code**

Add this test to `tests/test_simulation.py`:

```python
    def test_app_module_imports_without_creating_a_window(self):
        import app
        self.assertTrue(callable(app.main))
```

- [ ] **Step 2: Verify the smoke test fails**

Run:

```powershell
$env:MPLBACKEND='Agg'
.\.venv\Scripts\python.exe -m unittest tests.test_simulation.SimulationTests.test_app_module_imports_without_creating_a_window -v
```

Expected: FAIL because `app.py` does not exist.

- [ ] **Step 3: Create the GUI module without import side effects**

Create `app.py` with these concrete components:

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from fuzzy_rules import FuzzyScales
from simulation_core import (
    PIDConfig,
    SimulationConfig,
    SimulationResult,
    ThermalParams,
    calculate_metrics,
    export_data_files,
    run_simulation,
)


PARAMETER_FIELDS = (
    ("target_temp", "目标温度 (°C)", "43.0"),
    ("duration", "仿真时长 (s)", "180.0"),
    ("dt", "时间步长 (s)", "0.1"),
    ("tau", "热时间常数 (s)", "45.0"),
    ("heat_gain", "激光加热增益 (°C/s)", "0.30"),
    ("kp", "基础 Kp", "0.18"),
    ("ki", "基础 Ki", "0.012"),
    ("kd", "基础 Kd", "0.20"),
    ("noise_std", "测量噪声标准差 (°C)", "0.45"),
    ("kalman_q", "卡尔曼 Q", "0.0025"),
    ("kalman_r", "卡尔曼 R", "0.2025"),
    ("error_scale", "模糊误差尺度", "6.0"),
    ("error_rate_scale", "模糊误差变化率尺度", "0.5"),
    ("delta_kp", "最大 ΔKp", "0.08"),
    ("delta_ki", "最大 ΔKi", "0.006"),
    ("delta_kd", "最大 ΔKd", "0.12"),
    ("seed", "随机种子", "20260817"),
)

SLIDER_RANGES = {
    "target_temp": (37.0, 55.0, 0.1),
    "kp": (0.0, 1.0, 0.005),
    "ki": (0.0, 0.2, 0.001),
    "kd": (0.0, 2.0, 0.01),
}


class SimulatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("组织升温与模糊 PID 激光功率控制仿真")
        self.root.geometry("1500x900")
        self.root.minsize(1180, 720)
        self.entries: dict[str, ttk.Entry] = {}
        self.scales: dict[str, tk.Scale] = {}
        self.controller_mode = tk.StringVar(value="fuzzy")
        self.use_kalman = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="调整参数后点击运行。默认值是教学仿真参数。")
        self.current_result: SimulationResult | None = None
        self.comparison_result: SimulationResult | None = None
        self._build_layout()
        self.run_current()

    def _build_layout(self) -> None:
        container = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(container, padding=10)
        charts = ttk.Frame(container, padding=(0, 10, 10, 10))
        container.add(controls, weight=0)
        container.add(charts, weight=1)

        ttk.Label(controls, text="控制与模型参数", font=("Microsoft YaHei UI", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        ttk.Label(controls, text="控制器").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            controls, textvariable=self.controller_mode,
            values=("classic", "fuzzy"), state="readonly", width=15,
        ).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Checkbutton(controls, text="使用卡尔曼估计反馈", variable=self.use_kalman).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=2
        )
        for row, (name, label, default) in enumerate(PARAMETER_FIELDS, start=3):
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=1)
            entry = ttk.Entry(controls, width=14)
            entry.insert(0, default)
            entry.grid(row=row, column=1, sticky="ew", pady=1)
            self.entries[name] = entry
            if name in SLIDER_RANGES:
                lower, upper, resolution = SLIDER_RANGES[name]
                scale = tk.Scale(
                    controls, from_=lower, to=upper, resolution=resolution,
                    orient=tk.HORIZONTAL, showvalue=False, length=150,
                    command=lambda value, key=name: self._sync_entry(key, value),
                )
                scale.set(float(default))
                scale.grid(row=row, column=2, sticky="ew", padx=(6, 0), pady=1)
                self.scales[name] = scale
        button_row = 3 + len(PARAMETER_FIELDS)
        ttk.Button(controls, text="运行当前参数", command=self.run_current).grid(
            row=button_row, column=0, columnspan=3, sticky="ew", pady=(10, 2)
        )
        ttk.Button(controls, text="经典 PID 与模糊 PID 对比", command=self.run_comparison).grid(
            row=button_row + 1, column=0, columnspan=3, sticky="ew", pady=2
        )
        ttk.Button(controls, text="恢复默认参数", command=self.reset_defaults).grid(
            row=button_row + 2, column=0, columnspan=3, sticky="ew", pady=2
        )
        ttk.Button(controls, text="导出结果", command=self.export_current).grid(
            row=button_row + 3, column=0, columnspan=3, sticky="ew", pady=2
        )
        controls.columnconfigure(1, weight=1)

        self.figure = Figure(figsize=(10, 7), dpi=100, constrained_layout=True)
        self.axes = self.figure.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=charts)
        toolbar = NavigationToolbar2Tk(self.canvas, charts, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.metrics_text = tk.Text(charts, height=5, wrap="word", font=("Consolas", 10))
        self.metrics_text.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(self.root, textvariable=self.status, anchor="w", padding=5).pack(fill=tk.X)

    def _sync_entry(self, name: str, value: str) -> None:
        entry = self.entries[name]
        entry.delete(0, tk.END)
        resolution = SLIDER_RANGES[name][2]
        digits = max(0, len(str(resolution).split(".")[-1].rstrip("0")))
        entry.insert(0, f"{float(value):.{digits}f}")

    def _number(self, name: str) -> float:
        try:
            return float(self.entries[name].get())
        except ValueError as exc:
            raise ValueError(f"{name} 必须是数字") from exc

    def read_config(self, mode: str | None = None) -> SimulationConfig:
        thermal = ThermalParams(37.0, self._number("tau"), self._number("heat_gain"))
        pid = PIDConfig(self._number("kp"), self._number("ki"), self._number("kd"))
        fuzzy = FuzzyScales(
            self._number("error_scale"), self._number("error_rate_scale"),
            self._number("delta_kp"), self._number("delta_ki"), self._number("delta_kd"),
        )
        return SimulationConfig(
            duration=self._number("duration"), dt=self._number("dt"),
            target_temp=self._number("target_temp"),
            measurement_noise_std=self._number("noise_std"),
            kalman_q=self._number("kalman_q"), kalman_r=self._number("kalman_r"),
            use_kalman=self.use_kalman.get(),
            controller_mode=mode or self.controller_mode.get(),
            random_seed=int(self._number("seed")), thermal=thermal, pid=pid, fuzzy=fuzzy,
        )

    def run_current(self) -> None:
        try:
            self.current_result = run_simulation(self.read_config())
            self.comparison_result = None
            self.draw_results(self.current_result)
            self.status.set("仿真完成。默认模型用于教学，不代表真实治疗参数。")
        except Exception as exc:
            messagebox.showerror("参数或计算错误", str(exc), parent=self.root)

    def run_comparison(self) -> None:
        try:
            classic_config = self.read_config("classic")
            fuzzy_config = self.read_config("fuzzy")
            classic = run_simulation(classic_config)
            noise = classic.measurement_temp - classic.true_temp
            fuzzy = run_simulation(fuzzy_config, noise_sequence=noise)
            self.current_result = classic
            self.comparison_result = fuzzy
            self.draw_results(classic, fuzzy)
            self.status.set("已使用相同组织参数和噪声完成控制器对比。")
        except Exception as exc:
            messagebox.showerror("参数或计算错误", str(exc), parent=self.root)

    def draw_results(self, primary: SimulationResult, secondary: SimulationResult | None = None) -> None:
        for axis in self.axes.flat:
            axis.clear()
            axis.grid(True, alpha=0.25)
        temp_ax, power_ax, error_ax, gain_ax = self.axes.flat
        temp_ax.plot(primary.time, primary.target_temp, "k--", label="目标温度")
        temp_ax.plot(primary.time, primary.true_temp, color="#0072B2", label="真实温度")
        temp_ax.scatter(primary.time[::10], primary.measurement_temp[::10], s=7, alpha=0.2, label="带噪测量")
        temp_ax.plot(primary.time, primary.estimated_temp, color="#009E73", label="卡尔曼估计")
        power_ax.plot(primary.time, primary.power * 100.0, color="#D55E00", label=primary.config.controller_mode)
        error_ax.plot(primary.time, primary.error, color="#CC79A7", label=primary.config.controller_mode)
        gain_ax.plot(primary.time, primary.kp, label="Kp")
        gain_ax.plot(primary.time, primary.ki, label="Ki")
        gain_ax.plot(primary.time, primary.kd, label="Kd")
        if secondary is not None:
            temp_ax.plot(secondary.time, secondary.true_temp, color="#E69F00", label="模糊 PID 真实温度")
            power_ax.plot(secondary.time, secondary.power * 100.0, color="#56B4E9", label="fuzzy")
            error_ax.plot(secondary.time, secondary.error, color="#009E73", label="fuzzy")
            gain_ax.plot(secondary.time, secondary.kp, "--", label="fuzzy Kp")
            gain_ax.plot(secondary.time, secondary.ki, "--", label="fuzzy Ki")
            gain_ax.plot(secondary.time, secondary.kd, "--", label="fuzzy Kd")
        temp_ax.set(title="温度响应", xlabel="时间 (s)", ylabel="温度 (°C)")
        power_ax.set(title="激光功率", xlabel="时间 (s)", ylabel="功率 (%)", ylim=(-2, 102))
        error_ax.set(title="反馈温度误差", xlabel="时间 (s)", ylabel="误差 (°C)")
        gain_ax.set(title="PID 实时增益", xlabel="时间 (s)", ylabel="增益")
        for axis in self.axes.flat:
            axis.legend(loc="best", fontsize=8)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, self._format_metrics(primary, secondary))
        self.canvas.draw_idle()

    def _format_metrics(self, primary: SimulationResult, secondary: SimulationResult | None) -> str:
        def line(label: str, result: SimulationResult) -> str:
            m = calculate_metrics(result)
            rise = "未达到" if m["rise_time_s"] is None else f'{m["rise_time_s"]:.1f} s'
            return (
                f'{label}: 超调 {m["overshoot_c"]:.3f} °C | 上升时间 {rise} | '
                f'稳态误差 {m["steady_state_error_c"]:.3f} °C | '
                f'测量 RMSE {m["measurement_rmse_c"]:.3f} °C | '
                f'卡尔曼 RMSE {m["kalman_rmse_c"]:.3f} °C | '
                f'能量 {m["normalized_energy_s"]:.2f}\n'
            )
        text = line(primary.config.controller_mode, primary)
        if secondary is not None:
            text += line(secondary.config.controller_mode, secondary)
        return text

    def reset_defaults(self) -> None:
        defaults = {name: default for name, _, default in PARAMETER_FIELDS}
        for name, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, defaults[name])
            if name in self.scales:
                self.scales[name].set(float(defaults[name]))
        self.controller_mode.set("fuzzy")
        self.use_kalman.set(True)
        self.run_current()

    def export_current(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("没有结果", "请先运行仿真。", parent=self.root)
            return
        base = filedialog.askdirectory(initialdir=Path(__file__).parent / "outputs", parent=self.root)
        if not base:
            return
        directory = Path(base) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        export_data_files(self.current_result, directory)
        self.figure.savefig(directory / "curves.png", dpi=180, facecolor="white")
        self.status.set(f"结果已导出到 {directory}")


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    SimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
```

Do not create a Tk root at import time. The import smoke test depends on this behavior.

- [ ] **Step 4: Run the import smoke test and full tests**

Run:

```powershell
$env:MPLBACKEND='Agg'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Remove-Item Env:MPLBACKEND
```

Expected: 18 tests pass and no window opens during tests.

- [ ] **Step 5: Commit the initial desktop application**

```powershell
git add app.py tests/test_simulation.py
git commit -m "feat: add interactive desktop simulator"
```

## Task 9: Add PNG verification and GitHub documentation

**Files:**
- Modify: `tests/test_export.py`
- Create: `README.md`
- Create: `run_simulator.bat`
- Create: `docs/images/example_result.png`

- [ ] **Step 1: Add a failing PNG readability test**

Append to `tests/test_export.py`:

```python
    def test_matplotlib_can_write_a_nonempty_png(self):
        from matplotlib.figure import Figure

        result = run_simulation(SimulationConfig(duration=2.0, dt=0.1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curves.png"
            figure = Figure(figsize=(4, 3))
            axis = figure.subplots()
            axis.plot(result.time, result.true_temp)
            figure.savefig(path, dpi=100)
            self.assertGreater(path.stat().st_size, 1_000)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
```

- [ ] **Step 2: Run the PNG test**

Run:

```powershell
$env:MPLBACKEND='Agg'
.\.venv\Scripts\python.exe -m unittest tests.test_export.ExportTests.test_matplotlib_can_write_a_nonempty_png -v
Remove-Item Env:MPLBACKEND
```

Expected: PASS because Matplotlib is already installed. This test establishes the export dependency before documentation and visual verification.

- [ ] **Step 3: Add the one-click launcher**

Create `run_simulator.bat`:

```bat
@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment not found.
  echo Run: py -3.13 -m venv .venv
  echo Then: .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
```

- [ ] **Step 4: Write the GitHub README**

Create `README.md` with these sections and facts:

````markdown
# 组织升温与模糊 PID 激光功率控制仿真

这是一个面向控制与光声测温入门的 Windows 桌面仿真。程序用一阶热模型描述组织升温和散热，对比经典 PID 与模糊自整定 PID，并演示带噪测温经过模型驱动卡尔曼滤波后的反馈控制。

> 本项目参考相关论文公开摘要中的“温度估计 + 模糊 PID 闭环控制”结构。默认热模型参数、PID 增益和模糊规则是教学仿真假设，不是论文原始实验参数，不能用于真实治疗决策。

## 功能

- 调节目标温度、组织热时间常数和激光加热增益
- 调节 Kp、Ki、Kd 和模糊增益修正范围
- 开关卡尔曼温度估计并调节 Q、R 和测量噪声
- 比较经典 PID 与模糊 PID 的温度、功率、误差和实时增益
- 计算超调、上升时间、稳态误差、测温 RMSE 和累计激光能量
- 导出 PNG、CSV、JSON 和参数来源声明

## 界面示例

![仿真结果](docs/images/example_result.png)

## 安装

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行

双击 `run_simulator.bat`，或者执行：

```powershell
.\.venv\Scripts\python.exe app.py
```

调整参数后点击“运行当前参数”。“经典 PID 与模糊 PID 对比”会使用相同组织参数和相同噪声序列，避免随机噪声影响比较。

## 模型

组织温度采用一阶集总模型：

\[
\frac{dT}{dt}=-\frac{T-T_a}{\tau}+K_hu
\]

该模型适合解释闭环控制，不描述肿瘤内部空间温度场。真实研究还需要组织光学参数、空间热扩散、血流灌注、探针分布和实验标定。

## RMSE 怎么看

原始测量 RMSE 衡量带噪测温与模型真实温度之间的差距。卡尔曼 RMSE 衡量估计温度与真实温度之间的差距。卡尔曼 RMSE 更低，说明在当前仿真假设下滤波降低了测量误差。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖热模型、PID 限幅和抗饱和、模糊规则、卡尔曼估计、随机可重复性、指标计算与结果导出。
````

- [ ] **Step 5: Launch the real GUI and export a result**

Run:

```powershell
.\.venv\Scripts\python.exe app.py
```

Manual checks:

- Window opens without an exception.
- All 17 parameter rows are visible at 1500x900 and remain reachable at 1180x720.
- Chinese text renders without squares.
- The four plots do not overlap controls or each other.
- Current mode and comparison mode both update curves and metrics.
- Invalid `tau=0` produces a readable dialog and does not crash the window.
- Export creates `curves.png`, `timeseries.csv`, `parameters.json`, and `README.txt`.

- [ ] **Step 6: Preserve a verified example image**

Create `docs/images/` and copy the verified default-run PNG to `docs/images/example_result.png`. Open it with an image viewer and confirm the temperature, power, error, and gain panels contain nonblank curves, legible axes, and complete legends.

- [ ] **Step 7: Run the complete automated suite**

Run:

```powershell
$env:MPLBACKEND='Agg'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Remove-Item Env:MPLBACKEND
```

Expected: 19 tests pass with no failures, errors, or warnings.

- [ ] **Step 8: Commit documentation and verified example**

```powershell
git add README.md run_simulator.bat tests/test_export.py docs/images/example_result.png
git commit -m "docs: add simulator guide and verified example"
```

## Task 10: Final repository verification

**Files:**
- Inspect all tracked project files

- [ ] **Step 1: Check source syntax**

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py simulation_core.py fuzzy_rules.py tests
```

Expected: exit code 0 with no output.

- [ ] **Step 2: Run all tests again with fresh output**

Run:

```powershell
$env:MPLBACKEND='Agg'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Remove-Item Env:MPLBACKEND
```

Expected: 19 tests pass, 0 failures, 0 errors.

- [ ] **Step 3: Verify repository hygiene**

Run:

```powershell
git status --short
git ls-files
```

Expected: clean status. Tracked files include source, tests, documentation, launcher, example image, design, and implementation plan. `.venv`, `__pycache__`, and generated `outputs/run_*` directories are absent from tracked files.

- [ ] **Step 4: Check the commit history**

Run:

```powershell
git log --oneline --decorate -10
```

Expected: separate commits for design, bootstrap, thermal model, Kalman estimator, PID, fuzzy PID, simulation integration, export, GUI, and documentation.

- [ ] **Step 5: Record actual final values**

In the task handoff, report the exact test count, Python version, NumPy version, Matplotlib version, Git commit ID, project path, and any remaining scientific limitations. Do not claim that defaults reproduce a paper.
