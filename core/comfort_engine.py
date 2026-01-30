"""
Expanded Comfort Inference Engine

Penalty-based, zone-aware, temporally sensitive comfort modeling.
"""

import numpy as np
from core.constants import MAX_CELL_PRESSURE, COMFORT_WEIGHTS


def _clip01(x):
    return max(0.0, min(x, 1.0))


def compute_comfort(pressure_grid, previous_grid):
    rows, cols = pressure_grid.shape
    mean_p = max(np.mean(pressure_grid), 1e-8)

    # ---- 1. Peak pressure ----
    peak_penalty = _clip01(np.max(pressure_grid) / MAX_CELL_PRESSURE)

    # ---- 2. High-pressure area ----
    area_penalty = _clip01(
        np.sum(pressure_grid > 0.7 * MAX_CELL_PRESSURE) / (rows * cols)
    )

    # ---- 3. Zone bias (heel vs forefoot) ----
    heel = pressure_grid[int(0.7 * rows):, :]
    forefoot = pressure_grid[:int(0.3 * rows), :]
    zone_penalty = _clip01(
        abs(np.mean(heel) - np.mean(forefoot)) / mean_p
    )

    # ---- 4. Left-right asymmetry ----
    left = pressure_grid[:, :cols // 2]
    right = pressure_grid[:, cols // 2:]
    asymmetry_penalty = _clip01(
        abs(np.mean(left) - np.mean(right)) / mean_p
    )

    # ---- 5. Temporal volatility ----
    if previous_grid is None:
        temporal_penalty = 0.0
    else:
        temporal_penalty = _clip01(
            np.mean(np.abs(pressure_grid - previous_grid)) / mean_p
        )

    # ---- 6. Pressure persistence ----
    persistence_penalty = 0.0
    if temporal_penalty < 0.2:
        persistence_penalty = _clip01(mean_p / MAX_CELL_PRESSURE)

    # ---- Weighted sum ----
    total_penalty = (
        COMFORT_WEIGHTS["pressure_peak"] * peak_penalty +
        COMFORT_WEIGHTS["high_pressure_area"] * area_penalty +
        COMFORT_WEIGHTS["zone_bias"] * zone_penalty +
        COMFORT_WEIGHTS["asymmetry"] * asymmetry_penalty +
        COMFORT_WEIGHTS["temporal_variation"] * temporal_penalty +
        COMFORT_WEIGHTS["pressure_persistence"] * persistence_penalty
    )

    comfort_index = int(round(100 * (1 - _clip01(total_penalty))))

    return {
        "comfort_index": comfort_index,
        "penalties": {
            "pressure_peak": round(peak_penalty, 3),
            "high_pressure_area": round(area_penalty, 3),
            "zone_bias": round(zone_penalty, 3),
            "asymmetry": round(asymmetry_penalty, 3),
            "temporal_variation": round(temporal_penalty, 3),
            "pressure_persistence": round(persistence_penalty, 3)
        }
    }
