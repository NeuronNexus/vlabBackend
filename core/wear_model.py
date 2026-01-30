"""
Enhanced Wear Accumulation Model

Deterministic, bias-aware wear accumulation driven by:
- Pressure magnitude (nonlinear peak sensitivity)
- Temporal persistence
- Spatial extent
- Activity intensity
- Material durability
"""

import numpy as np
from core.constants import (
    WEAR_RATE,
    MAX_WEAR,
    MAX_CELL_PRESSURE,
    WEAR_NONLINEARITY
)


def accumulate_wear(
    previous_wear: np.ndarray,
    pressure_grid: np.ndarray,
    previous_pressure: np.ndarray,
    durability_factor: float,
    activity_wear_rate: float,
    time_step: int = 1
) -> np.ndarray:
    """
    Accumulate wear with explicit, explainable modifiers.
    """

    rows, cols = pressure_grid.shape
    dt = max(time_step, 0)

    # ---- Material response ----
    durability = max(min(durability_factor, 1.0), 0.0)
    material_response = (1.0 - durability) ** 2

    # ---- Area modifier ----
    high_pressure_fraction = np.sum(
        pressure_grid > 0.6 * MAX_CELL_PRESSURE
    ) / (rows * cols)

    area_modifier = 1.0 + high_pressure_fraction

    # ---- Persistence modifier ----
    if previous_pressure is None:
        persistence_modifier = 1.0
    else:
        delta = np.mean(np.abs(pressure_grid - previous_pressure))
        persistence = 1.0 - min(delta / MAX_CELL_PRESSURE, 1.0)
        persistence_modifier = 1.0 + 0.5 * persistence

    # ---- Zone modifier ----
    zone_modifier = np.ones_like(pressure_grid)

    heel_start = int(0.7 * rows)
    forefoot_end = int(0.3 * rows)

    zone_modifier[:forefoot_end, :] *= 1.1
    zone_modifier[heel_start:, :] *= 1.1
    zone_modifier[forefoot_end:heel_start, :] *= 0.9

    # ---- Wear increment (NONLINEAR, PEAK-SENSITIVE) ----
    wear_increment = (
        (pressure_grid / MAX_CELL_PRESSURE) ** WEAR_NONLINEARITY *
        WEAR_RATE *
        dt *
        activity_wear_rate *
        material_response *
        area_modifier *
        persistence_modifier *
        zone_modifier
    )

    # ---- Accumulate & saturate ----
    new_wear = previous_wear + wear_increment
    new_wear = np.clip(new_wear, 0.0, MAX_WEAR)

    return new_wear
