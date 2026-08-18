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

    def test_large_base_gain_is_not_silently_replaced(self):
        controller = FuzzyPIDController(PIDConfig(kp=5.0), FuzzyScales())
        controller.update(0.0, 0.1, measurement=37.0)
        self.assertGreater(controller.last_gains[0], 4.0)

    def test_rising_temperature_below_target_increases_derivative_braking(self):
        tuner = FuzzyTuner(FuzzyScales())
        _, _, delta_kd = tuner.corrections(error=6.0, error_rate=-1.0)
        self.assertGreater(delta_kd, 0.0)


if __name__ == "__main__":
    unittest.main()
