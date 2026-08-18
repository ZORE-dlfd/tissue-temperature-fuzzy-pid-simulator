from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


CENTERS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

# Rows are error (NB to PB); columns are error rate (NB to PB).
# These are explainable teaching rules, not values copied from a paper.
DKP_RULES = np.array(
    [
        [1.0, 1.0, 0.8, 0.6, 0.4],
        [0.8, 0.6, 0.4, 0.2, 0.0],
        [-0.2, -0.3, -0.4, -0.3, -0.2],
        [0.0, 0.2, 0.4, 0.6, 0.8],
        [0.4, 0.6, 0.8, 1.0, 1.0],
    ]
)

DKI_RULES = np.array(
    [
        [-1.0, -1.0, -0.8, -1.0, -1.0],
        [-0.6, -0.4, -0.2, -0.4, -0.6],
        [0.5, 0.8, 1.0, 0.8, 0.5],
        [-0.6, -0.4, -0.2, -0.4, -0.6],
        [-1.0, -1.0, -0.8, -1.0, -1.0],
    ]
)

DKD_RULES = np.array(
    [
        [1.0, 0.8, 0.5, 0.0, -0.2],
        [0.8, 0.6, 0.3, 0.0, -0.2],
        [0.2, 0.1, 0.0, 0.1, 0.2],
        [-0.2, 0.0, 0.3, 0.6, 0.8],
        [1.0, 0.0, 0.5, 0.8, 1.0],
    ]
)


@dataclass(frozen=True)
class FuzzyScales:
    error_scale: float = 6.0
    error_rate_scale: float = 0.5
    delta_kp_max: float = 0.08
    delta_ki_max: float = 0.006
    delta_kd_max: float = 0.12

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                self.error_scale,
                self.error_rate_scale,
                self.delta_kp_max,
                self.delta_ki_max,
                self.delta_kd_max,
            )
        ):
            raise ValueError("fuzzy scales must be finite")
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
