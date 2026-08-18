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

    def test_integral_is_limited_even_without_output_saturation(self):
        config = PIDConfig(
            kp=0.0,
            ki=0.1,
            kd=0.0,
            integral_limit=2.0,
        )
        controller = PIDController(config)
        for _ in range(10):
            controller.update(1.0, 1.0)
        self.assertEqual(controller.integral, 2.0)

    def test_nonpositive_integral_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integral_limit"):
            PIDConfig(integral_limit=0.0)

    def test_output_range_cannot_exceed_normalized_laser_power(self):
        with self.assertRaisesRegex(ValueError, "output"):
            PIDConfig(output_max=1.1)

    def test_derivative_filter_rejects_invalid_coefficient(self):
        with self.assertRaisesRegex(ValueError, "derivative_filter"):
            PIDConfig(kp=1.0, ki=0.1, kd=0.1, derivative_filter=1.5)


if __name__ == "__main__":
    unittest.main()
