import unittest

import numpy as np

from simulation_core import SimulationConfig, calculate_metrics, run_simulation


class SimulationTests(unittest.TestCase):
    def test_app_module_imports_without_creating_a_window(self):
        import app

        self.assertTrue(callable(app.main))

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

    def test_unreachable_target_reports_no_rise_time(self):
        result = run_simulation(
            SimulationConfig(duration=1.0, dt=0.1, target_temp=55.0)
        )
        self.assertIsNone(calculate_metrics(result)["rise_time_s"])

    def test_supplied_noise_sequence_is_used_exactly(self):
        config = SimulationConfig(duration=1.0, dt=0.1)
        noise = np.linspace(-0.2, 0.2, 11)
        result = run_simulation(config, noise_sequence=noise)
        np.testing.assert_allclose(result.measurement_temp - result.true_temp, noise)

    def test_nonintegral_duration_step_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            SimulationConfig(duration=1.0, dt=0.3)

    def test_warming_simulator_rejects_target_below_initial_temperature(self):
        with self.assertRaisesRegex(ValueError, "target_temp"):
            SimulationConfig(initial_temp=43.0, target_temp=40.0)

    def test_simulation_rejects_nonfinite_noise_parameter(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            SimulationConfig(measurement_noise_std=float("inf"))

    def test_normalized_energy_uses_applied_zero_order_hold_power(self):
        config = SimulationConfig(
            duration=1.0,
            dt=0.5,
            target_temp=43.0,
            measurement_noise_std=0.0,
        )
        result = run_simulation(config)
        metrics = calculate_metrics(result)
        self.assertAlmostEqual(
            metrics["normalized_energy_s"],
            float(np.sum(result.power[:-1]) * config.dt),
        )


if __name__ == "__main__":
    unittest.main()
