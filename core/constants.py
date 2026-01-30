"""
SoleSense Simulation Constants

Abstract, dimensionless constants defining the simulation universe.
No real-world biomechanical meaning is implied.
"""

# =========================
# GRID CONFIGURATION
# =========================

GRID_ROWS = 20   # heel → toe
GRID_COLS = 10   # left → right

# =========================
# PRESSURE LIMITS
# =========================

MAX_CELL_PRESSURE = 0.15
MIN_CELL_PRESSURE = 0.0

# =========================
# TIME CONFIGURATION
# =========================

TIME_STEP = 1
DEFAULT_SIM_STEPS = 10000

# =========================
# ACTIVITY PROFILES
# =========================

ACTIVITY_PROFILE = {
    "standing": {
        "load_multiplier": 1.0,
        "variation": 0.05,
        "wear_rate": 0.6
    },
    "walking": {
        "load_multiplier": 1.1,
        "variation": 0.15,
        "wear_rate": 1.0
    },
    "running": {
        "load_multiplier": 1.35,
        "variation": 0.35,
        "wear_rate": 1.8
    },
    "stairs": {
        "load_multiplier": 1.25,
        "variation": 0.25,
        "wear_rate": 1.5
    },
    "jumping": {
        "load_multiplier": 1.6,
        "variation": 0.5,
        "wear_rate": 2.2
    }
}

# =========================
# ARCH DISTRIBUTION BIAS
# =========================

ARCH_BIAS = {
    "flat": 0.15,
    "normal": 0.0,
    "high": -0.15
}

# =========================
# COMFORT PENALTY WEIGHTS
# =========================

COMFORT_WEIGHTS = {
    "pressure_peak": 0.20,
    "high_pressure_area": 0.20,
    "zone_bias": 0.15,
    "asymmetry": 0.15,
    "temporal_variation": 0.15,
    "pressure_persistence": 0.15
}

# =========================
# WEAR CONFIGURATION
# =========================

WEAR_RATE = 0.00001
MAX_WEAR = 1.0

# =========================
# WEAR VISIBILITY & CONTRAST
# =========================

# Amplifies peak wear deterministically (no randomness)
WEAR_NONLINEARITY = 1.3

# Presentation-only scaling for human interpretability
# Does NOT affect comfort, pressure, or scenario classification
WEAR_VISIBILITY_GAIN = 50.0
