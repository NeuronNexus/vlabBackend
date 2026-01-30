"""
Spatial Pressure Field Module

Generates a bounded 2D pressure distribution over the shoe sole.
Now includes lateral (medial–lateral) pressure variation.
"""

import numpy as np
from core.constants import GRID_ROWS, GRID_COLS


def _base_longitudinal_profile():
    """
    Heel-to-toe base pressure profile.
    Heel > Forefoot > Toe
    """
    x = np.linspace(0, 1, GRID_ROWS)
    return 1.4 * np.exp(-3.5 * x) + 0.35


def _apply_arch_bias(profile, arch_bias):
    """
    Reduce pressure in midfoot for higher arches.
    """
    mid_start = int(0.35 * GRID_ROWS)
    mid_end = int(0.65 * GRID_ROWS)

    profile[mid_start:mid_end] *= (1.0 - arch_bias)
    return profile


def _lateral_weight(row_idx):
    """
    Generate medial-lateral pressure distribution.
    - Heel: center-biased
    - Midfoot: narrow contact
    - Forefoot: dual lobes
    """
    x = np.linspace(-1, 1, GRID_COLS)

    row_ratio = row_idx / GRID_ROWS

    # Heel
    if row_ratio < 0.35:
        sigma = 0.35
        return np.exp(-(x ** 2) / (2 * sigma ** 2))

    # Midfoot (arch void)
    if row_ratio < 0.6:
        sigma = 0.18
        return np.exp(-(x ** 2) / (2 * sigma ** 2)) * 0.6

    # Forefoot (split pressure heads)
    left = np.exp(-((x + 0.35) ** 2) / 0.08)
    right = np.exp(-((x - 0.35) ** 2) / 0.08)
    return left + right


def _expand_to_grid(longitudinal_profile):
    """
    Expand 1D profile into full 2D pressure field.
    """
    grid = np.zeros((GRID_ROWS, GRID_COLS))

    for i in range(GRID_ROWS):
        lateral = _lateral_weight(i)
        grid[i, :] = longitudinal_profile[i] * lateral

    return grid


def _apply_contact_capacity(grid, capacity):
    """
    Shape pressure concentration.
    """
    exponent = max(0.6, 1.4 - capacity)
    return np.power(grid, exponent)


def _smooth_grid_bounded(grid, stiffness_factor, iterations=4):
    """
    Bounded smoothing with zero-flux edges.
    """
    steps = int((1.0 - stiffness_factor) * iterations)
    rows, cols = grid.shape

    for _ in range(steps):
        new_grid = grid.copy()

        for i in range(rows):
            for j in range(cols):
                neighbors = [grid[i, j]]

                if i > 0:
                    neighbors.append(grid[i - 1, j])
                if i < rows - 1:
                    neighbors.append(grid[i + 1, j])
                if j > 0:
                    neighbors.append(grid[i, j - 1])
                if j < cols - 1:
                    neighbors.append(grid[i, j + 1])

                new_grid[i, j] = sum(neighbors) / len(neighbors)

        grid = new_grid

    return grid


def generate_pressure_field(params: dict) -> np.ndarray:
    """
    Generate 2D pressure field with realistic foot physics.
    """

    profile = _base_longitudinal_profile()
    profile = _apply_arch_bias(profile, params["arch_bias"])

    grid = _expand_to_grid(profile)

    grid = _apply_contact_capacity(
        grid,
        params["contact_capacity"]
    )

    grid = _smooth_grid_bounded(
        grid,
        params["stiffness_factor"]
    )

    return grid
