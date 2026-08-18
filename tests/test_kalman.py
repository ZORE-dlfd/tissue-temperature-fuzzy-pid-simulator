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
        params = ThermalParams(ambient_temp=42.0, tau=45.0, heat_gain=0.0)
        for value in measurements:
            kalman.predict(power=0.0, dt=0.1, params=params)
            estimates.append(kalman.update(value))
        self.assertLess(rmse(np.asarray(estimates), truth), rmse(measurements, truth))


if __name__ == "__main__":
    unittest.main()
