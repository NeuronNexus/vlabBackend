"""
Simulation Orchestrator (Refined)

Runs the SoleSense simulation and performs
post-simulation analysis to classify system behavior.
"""

import numpy as np

from core.normalization import normalize_inputs
from core.pressure_field import generate_pressure_field
from core.temporal_evolution import evolve_pressure_field
from core.constraints import apply_constraints
from core.comfort_engine import compute_comfort
from core.wear_model import accumulate_wear
from core.constants import (
    DEFAULT_SIM_STEPS,
    WEAR_VISIBILITY_GAIN
)
from core.scenario_compare import compare_scenarios


# ============================================================
# POST-SIMULATION ANALYSIS
# ============================================================

def _analyze_trends(comfort_history, wear_history, pressure_history):
    comfort_values = [c["comfort_index"] for c in comfort_history]
    comfort_slope = comfort_values[-1] - comfort_values[0]

    wear_means = [np.mean(w) for w in wear_history]
    mid = len(wear_means) // 2

    early_wear_rate = wear_means[mid] - wear_means[0]
    late_wear_rate = wear_means[-1] - wear_means[mid]
    wear_accelerating = late_wear_rate > early_wear_rate * 1.2

    if len(pressure_history) < 2:
        pressure_delta = 0.0
    else:
        deltas = [
            np.mean(np.abs(pressure_history[i] - pressure_history[i - 1]))
            for i in range(1, len(pressure_history))
        ]
        pressure_delta = np.mean(deltas[-5:])

    penalty_totals = {}
    for c in comfort_history:
        for k, v in c["penalties"].items():
            penalty_totals[k] = penalty_totals.get(k, 0.0) + v

    dominant_factors = sorted(
        penalty_totals.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "comfort_slope": comfort_slope,
        "wear_accelerating": wear_accelerating,
        "pressure_delta": pressure_delta,
        "dominant_factors": [k for k, _ in dominant_factors[:2]]
    }


def _classify_scenario(analysis):
    if analysis["comfort_slope"] < -20:
        stability = "degrading"
    elif analysis["pressure_delta"] < 1e-3:
        stability = "stable"
    else:
        stability = "saturated"

    if "asymmetry" in analysis["dominant_factors"]:
        scenario_type = "imbalance-driven"
    elif "high_pressure_area" in analysis["dominant_factors"]:
        scenario_type = "fatigue-driven"
    elif "pressure_peak" in analysis["dominant_factors"]:
        scenario_type = "overload-driven"
    else:
        scenario_type = "stable"

    return {
        "scenario_type": scenario_type,
        "stability": stability,
        "dominant_factors": analysis["dominant_factors"],
        "explanation": (
            f"This scenario is {scenario_type.replace('-', ' ')}, "
            f"with system behavior classified as {stability}."
        )
    }


def _align_comfort_and_wear(comfort_history, wear_history):
    comfort_values = [c["comfort_index"] for c in comfort_history]
    wear_means = [np.mean(w) for w in wear_history]

    comfort_drop = comfort_values[0] - comfort_values[-1]
    wear_growth = wear_means[-1] - wear_means[0]

    comfort_drop_norm = comfort_drop / max(comfort_values[0], 1)
    wear_growth_norm = wear_growth / max(wear_means[-1], 1e-6)

    if comfort_drop_norm > 0.3 and wear_growth_norm < 0.2:
        regime = "transient_discomfort"
        explanation = (
            "Comfort decreases without significant material wear. "
            "Discomfort is likely due to pressure distribution rather than degradation."
        )
    elif comfort_drop_norm > 0.2 and wear_growth_norm > 0.3:
        regime = "fatigue_driven_degradation"
        explanation = (
            "Comfort declines alongside accelerating wear. "
            "Sustained pressure is degrading the sole material over time."
        )
    elif comfort_drop_norm < 0.1 and wear_growth_norm > 0.3:
        regime = "hidden_wear_risk"
        explanation = (
            "Material wear accumulates despite acceptable comfort levels. "
            "Potential long-term degradation without immediate discomfort."
        )
    else:
        regime = "balanced"
        explanation = (
            "Comfort and wear evolve proportionally with no dominant risk pattern."
        )

    return {
        "alignment_regime": regime,
        "comfort_drop_normalized": round(comfort_drop_norm, 3),
        "wear_growth_normalized": round(wear_growth_norm, 3),
        "interpretation": explanation
    }


def _model_assumptions():
    return {
        "scope": {
            "modeled": [
                "relative pressure distribution",
                "deterministic temporal evolution",
                "penalty-based comfort inference",
                "pressure-driven material wear"
            ],
            "not_modeled": [
                "human biomechanics",
                "medical conditions",
                "real gait cycles",
                "material fatigue physics"
            ]
        },
        "determinism": {
            "randomness": False,
            "repeatable": True,
            "same_input_same_output": True
        },
        "interpretation_limits": {
            "comfort_index": "comparative comfort indicator, not a diagnosis",
            "wear": "relative material degradation, not lifespan prediction"
        },
        "simplifications": [
            "2D discretized sole grid",
            "abstract force units",
            "bounded nonlinear wear accumulation",
            "zone-based heuristics instead of anatomy"
        ]
    }


# ============================================================
# MAIN SIMULATION
# ============================================================

def run_simulation(raw_inputs, steps=DEFAULT_SIM_STEPS):
    params = normalize_inputs(raw_inputs)

    base_grid = generate_pressure_field(params)

    pressure_grid = base_grid.copy()
    previous_pressure = None
    wear_grid = np.zeros_like(base_grid)

    comfort_history = []
    pressure_history = []
    wear_history = []
    wear_history_visible = []

    target_force = params["load_factor"] * params["activity_load"]

    for step in range(steps):
        evolved = evolve_pressure_field(
            previous_grid=pressure_grid,
            base_grid=base_grid,
            step=step,
            activity_variation=params["activity_variation"]
        )

        constrained = apply_constraints(
            evolved,
            target_total_force=target_force
        )

        comfort = compute_comfort(constrained, previous_pressure)

        wear_grid = accumulate_wear(
            previous_wear=wear_grid,
            pressure_grid=constrained,
            previous_pressure=previous_pressure,
            durability_factor=params["durability_factor"],
            activity_wear_rate=params["activity_wear_rate"],
            time_step=1
        )

        comfort_history.append(comfort)
        pressure_history.append(constrained.copy())
        wear_history.append(wear_grid.copy())
        wear_history_visible.append(wear_grid * WEAR_VISIBILITY_GAIN)

        previous_pressure = constrained
        pressure_grid = constrained

    analysis = _analyze_trends(
        comfort_history,
        wear_history,
        pressure_history
    )

    scenario_summary = _classify_scenario(analysis)
    alignment_summary = _align_comfort_and_wear(
        comfort_history,
        wear_history
    )

    return {
        "final_pressure": pressure_history[-1],
        "final_wear": wear_history[-1],
        "comfort_history": comfort_history,
        "wear_history": wear_history,
        "wear_history_visible": wear_history_visible,
        "scenario_summary": scenario_summary,
        "alignment_summary": alignment_summary,
        "model_assumptions": _model_assumptions(),
    }


# ============================================================
# SCENARIO COMPARISON (RESTORED)
# ============================================================

def run_scenario_comparison(
    baseline_inputs: dict,
    variant_inputs: dict,
    steps: int = DEFAULT_SIM_STEPS
) -> dict:

    baseline = run_simulation(baseline_inputs, steps)
    variant = run_simulation(variant_inputs, steps)

    comparison = compare_scenarios(baseline, variant)

    return {
        "baseline": {
            "scenario_summary": baseline["scenario_summary"],
            "alignment_summary": baseline["alignment_summary"],
            "final_comfort": baseline["comfort_history"][-1]["comfort_index"],
            "mean_wear": float(baseline["final_wear"].mean()),
            "max_wear": float(baseline["final_wear"].max())
        },
        "variant": {
            "scenario_summary": variant["scenario_summary"],
            "alignment_summary": variant["alignment_summary"],
            "final_comfort": variant["comfort_history"][-1]["comfort_index"],
            "mean_wear": float(variant["final_wear"].mean()),
            "max_wear": float(variant["final_wear"].max())
        },
        "what_if_analysis": comparison,
        "model_assumptions": baseline["model_assumptions"]
    }
