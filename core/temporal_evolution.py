"""
Temporal Evolution Module

Evolves pressure fields over time using:
- Deterministic longitudinal load shift
- Temporal persistence
- Activity-controlled variation

No randomness. No physics. No ML.
"""

import numpy as np
from core.constants import GRID_ROWS


def _longitudinal_load_shift(step: int, variation: float) -> np.ndarray:
    """
    Deterministic heel-to-toe load shift profile.
    Variation controls amplitude.
    """
    x = np.linspace(0, np.pi, GRID_ROWS)
    phase = step * 0.1

    return variation * np.sin(x + phase)


def evolve_pressure_field(
    previous_grid: np.ndarray,
    base_grid: np.ndarray,
    step: int,
    activity_variation: float,
    relaxation_rate: float = 0.15
) -> np.ndarray:
    """
    Evolve pressure field with temporal memory.

    Parameters
    ----------
    previous_grid : np.ndarray
        Pressure grid from previous timestep
    base_grid : np.ndarray
        Base spatial pressure field
    step : int
        Current simulation step
    activity_variation : float
        Controls temporal instability (0–1)
    relaxation_rate : float
        Controls inertia (0–1)
    """

    # --- 1. Deterministic longitudinal modulation ---
    shift_profile = _longitudinal_load_shift(step, activity_variation)
    modulation = np.tile(
        shift_profile.reshape(-1, 1),
        (1, base_grid.shape[1])
    )

    # --- 2. Target pressure ---
    target_grid = base_grid * (1.0 + modulation)

    # --- 3. Temporal relaxation ---
    alpha = max(min(relaxation_rate, 1.0), 0.0)

    evolved_grid = (
        (1.0 - alpha) * previous_grid +
        alpha * target_grid
    )

    return evolved_grid
