"""
Input Normalization Module

Maps bounded user inputs into bounded, dimensionless
simulation control parameters.

No physics. No inference.
"""

from core.constants import ARCH_BIAS, ACTIVITY_PROFILE


def _normalize_range(value, min_val, max_val):
    """Clamp and normalize value to [0, 1]."""
    value = max(min(value, max_val), min_val)
    return (value - min_val) / (max_val - min_val)


def normalize_inputs(inputs: dict) -> dict:
    """
    Normalize raw user inputs into simulation parameters.

    All outputs are dimensionless, bounded, and explainable.
    """

    # -----------------------------
    # LOAD / FORCE CONTROL
    # -----------------------------
    # Controls total injected load (abstract, not physical)
    load_factor = _normalize_range(
        inputs["body_weight"],
        min_val=40.0,
        max_val=120.0
    )

    # -----------------------------
    # CONTACT AREA CAPACITY
    # -----------------------------
    # Controls spatial spread capacity
    contact_capacity = _normalize_range(
        inputs["foot_size"],
        min_val=36.0,
        max_val=48.0
    )

    # -----------------------------
    # STRUCTURAL BIAS
    # -----------------------------
    # Small redistribution bias (arch shape proxy)
    arch_bias = ARCH_BIAS.get(inputs["arch_type"], 0.0)

    # -----------------------------
    # ACTIVITY DYNAMICS (DECOMPOSED)
    # -----------------------------
    activity = ACTIVITY_PROFILE.get(
        inputs["activity_mode"],
        ACTIVITY_PROFILE["walking"]  # safe fallback
    )

    activity_load = activity["load_multiplier"]
    activity_variation = activity["variation"]
    activity_wear_rate = activity["wear_rate"]

    # -----------------------------
    # MATERIAL CONTROLS
    # -----------------------------
    stiffness_factor = max(min(inputs["sole_stiffness"], 1.0), 0.0)
    durability_factor = max(min(inputs["material_durability"], 1.0), 0.0)

    return {
        # Load & spread
        "load_factor": load_factor,
        "contact_capacity": contact_capacity,

        # Structural modifiers
        "arch_bias": arch_bias,

        # Activity dynamics
        "activity_load": activity_load,
        "activity_variation": activity_variation,
        "activity_wear_rate": activity_wear_rate,

        # Material response
        "stiffness_factor": stiffness_factor,
        "durability_factor": durability_factor
    }
