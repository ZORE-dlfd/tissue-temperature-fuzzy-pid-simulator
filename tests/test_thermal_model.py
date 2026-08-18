import math
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
        expected = 37.0 + 40.0 * 0.25 * 0.5 * (1.0 - math.exp(-1.0 / 40.0))
        self.assertAlmostEqual(next_temp, expected, places=6)

    def test_invalid_time_constant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tau"):
            ThermalParams(ambient_temp=37.0, tau=0.0, heat_gain=0.25)

    def test_exact_step_does_not_cross_ambient_at_large_time_step(self):
        params = ThermalParams(ambient_temp=37.0, tau=45.0, heat_gain=0.25)
        next_temp = thermal_step(45.0, power=0.0, dt=100.0, params=params)
        self.assertGreater(next_temp, 37.0)
        self.assertLess(next_temp, 45.0)

    def test_nonfinite_thermal_parameter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            ThermalParams(ambient_temp=float("nan"))


if __name__ == "__main__":
    unittest.main()
