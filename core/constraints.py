"""
Constraint Enforcement Module (Refined)

Enforces hard invariants with capacity awareness and
monotonic behavior. Degrades safely when force targets
are unrepresentable.
"""

import numpy as np
from core.constants import MAX_CELL_PRESSURE, MIN_CELL_PRESSURE

EPSILON = 1e-8


def _max_representable_force(shape):
    return MAX_CELL_PRESSURE * shape[0] * shape[1]


def _safe_uniform(shape, target_force):
    """
    Uniform fallback that ALWAYS respects bounds.
    """
    max_force = _max_representable_force(shape)
    effective_force = min(target_force, max_force)

    value = effective_force / (shape[0] * shape[1])
    value = min(value, MAX_CELL_PRESSURE)

    return np.full(shape, value)


def enforce_finite(grid):
    return np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)


def enforce_bounds(grid):
    return np.clip(grid, MIN_CELL_PRESSURE, MAX_CELL_PRESSURE)


def enforce_force_with_capacity(grid, target_force):
    """
    Conserves force ONLY if representable.
    Otherwise saturates safely.
    """
    max_force = _max_representable_force(grid.shape)

    if target_force > max_force:
        return _safe_uniform(grid.shape, target_force)

    current_force = grid.sum()

    if current_force < EPSILON:
        return _safe_uniform(grid.shape, target_force)

    scale = target_force / current_force
    scaled = grid * scale

    clipped = enforce_bounds(scaled)
    clipped_force = clipped.sum()

    if clipped_force < EPSILON:
        return _safe_uniform(grid.shape, target_force)

    # Final soft correction (monotonic)
    correction = target_force / clipped_force
    return clipped * correction


def apply_constraints(grid, target_total_force):
    """
    Apply constraints in a priority-aware order.

    Guarantees:
    - Finite values
    - Bounded per-cell pressure
    - Force conservation when representable
    """

    grid = enforce_finite(grid)
    grid = enforce_bounds(grid)
    grid = enforce_force_with_capacity(grid, target_total_force)
    grid = enforce_bounds(grid)

    return grid
