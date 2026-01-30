"""
Input Validation Utilities

Responsible for validating and sanitizing user inputs
before they enter the simulation core.

No defaults. No inference. Fail fast.
"""

ALLOWED_ARCH_TYPES = {"flat", "normal", "high"}
ALLOWED_ACTIVITY_MODES = {
    "standing",
    "walking",
    "running",
    "stairs",
    "jumping"
}


def validate_numeric(name, value, min_val=None, max_val=None):
    if not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a number.")

    if min_val is not None and value < min_val:
        raise ValueError(f"'{name}' must be >= {min_val}.")

    if max_val is not None and value > max_val:
        raise ValueError(f"'{name}' must be <= {max_val}.")


def validate_enum(name, value, allowed):
    if not isinstance(value, str):
        raise ValueError(f"'{name}' must be a string.")

    value = value.lower()
    if value not in allowed:
        raise ValueError(
            f"'{name}' must be one of {sorted(allowed)}."
        )

    return value


def validate_simulation_inputs(payload: dict) -> dict:
    """
    Validate and normalize simulation inputs.
    Returns a sanitized copy of inputs.
    """

    required = [
        "body_weight",
        "foot_size",
        "arch_type",
        "activity_mode",
        "sole_stiffness",
        "material_durability",
    ]

    for key in required:
        if key not in payload:
            raise ValueError(f"Missing required field: '{key}'")

    validate_numeric("body_weight", payload["body_weight"], 20, 300)
    validate_numeric("foot_size", payload["foot_size"], 30, 50)
    validate_numeric("sole_stiffness", payload["sole_stiffness"], 0.0, 1.0)
    validate_numeric("material_durability", payload["material_durability"], 0.0, 1.0)

    arch = validate_enum(
        "arch_type",
        payload["arch_type"],
        ALLOWED_ARCH_TYPES
    )

    activity = validate_enum(
        "activity_mode",
        payload["activity_mode"],
        ALLOWED_ACTIVITY_MODES
    )

    return {
        "body_weight": payload["body_weight"],
        "foot_size": payload["foot_size"],
        "arch_type": arch,
        "activity_mode": activity,
        "sole_stiffness": payload["sole_stiffness"],
        "material_durability": payload["material_durability"],
    }
