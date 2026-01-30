"""
Scenario Comparison Engine (Refined)

Performs causal, tradeoff-aware comparison between
two SoleSense simulation runs.
"""

import numpy as np


def compare_scenarios(baseline: dict, variant: dict) -> dict:
    # -----------------------------
    # 1. Raw outcome deltas
    # -----------------------------
    base_comfort_end = baseline["comfort_history"][-1]["comfort_index"]
    var_comfort_end = variant["comfort_history"][-1]["comfort_index"]
    comfort_delta = var_comfort_end - base_comfort_end

    base_wear = baseline["final_wear"]
    var_wear = variant["final_wear"]
    mean_wear_delta = np.mean(var_wear) - np.mean(base_wear)
    max_wear_delta = np.max(var_wear) - np.max(base_wear)

    base_alignment = baseline["alignment_summary"]["alignment_regime"]
    var_alignment = variant["alignment_summary"]["alignment_regime"]

    # -----------------------------
    # 2. Mechanism attribution
    # -----------------------------
    base_factors = set(baseline["scenario_summary"]["dominant_factors"])
    var_factors = set(variant["scenario_summary"]["dominant_factors"])

    factor_shift = {
        "reduced": list(base_factors - var_factors),
        "introduced": list(var_factors - base_factors)
    }

    mechanism_notes = []

    if mean_wear_delta < 0:
        mechanism_notes.append(
            "Wear reduction driven by material or pressure persistence effects."
        )

    if comfort_delta > 0:
        mechanism_notes.append(
            "Comfort improvement linked to reduced dominant pressure penalties."
        )

    if base_alignment != var_alignment:
        mechanism_notes.append(
            f"Alignment regime changed from {base_alignment} to {var_alignment}."
        )

    # -----------------------------
    # 3. Tradeoff analysis
    # -----------------------------
    if comfort_delta >= 0 and mean_wear_delta <= 0:
        tradeoff_type = "no_tradeoff"
        tradeoff_summary = (
            "Variant improves or preserves comfort while reducing wear."
        )
    elif comfort_delta < 0 and mean_wear_delta < 0:
        tradeoff_type = "durability_tradeoff"
        tradeoff_summary = (
            "Variant reduces wear at the cost of comfort."
        )
    elif comfort_delta > 0 and mean_wear_delta > 0:
        tradeoff_type = "comfort_tradeoff"
        tradeoff_summary = (
            "Variant improves comfort but increases wear."
        )
    else:
        tradeoff_type = "neutral"
        tradeoff_summary = (
            "Variant does not materially change comfort or wear."
        )

    # -----------------------------
    # 4. Verdict
    # -----------------------------
    if tradeoff_type == "no_tradeoff":
        verdict = "strictly_better"
        verdict_reason = (
            "Variant dominates baseline on both experience and durability."
        )
    elif tradeoff_type in ("durability_tradeoff", "comfort_tradeoff"):
        verdict = "tradeoff"
        verdict_reason = tradeoff_summary
    elif abs(comfort_delta) < 5 and abs(mean_wear_delta) < 0.01:
        verdict = "equivalent"
        verdict_reason = (
            "Variant behaves similarly to baseline within tolerance."
        )
    else:
        verdict = "worse"
        verdict_reason = (
            "Variant degrades outcomes without compensating benefits."
        )

    # -----------------------------
    # 5. Structured output
    # -----------------------------
    return {
        "outcome_deltas": {
            "comfort_delta": comfort_delta,
            "mean_wear_delta": round(mean_wear_delta, 4),
            "max_wear_delta": round(max_wear_delta, 4),
            "alignment_change": {
                "from": base_alignment,
                "to": var_alignment
            }
        },
        "mechanism_attribution": {
            "dominant_factor_shift": factor_shift,
            "notes": mechanism_notes
        },
        "tradeoff_analysis": {
            "type": tradeoff_type,
            "summary": tradeoff_summary
        },
        "verdict": {
            "classification": verdict,
            "rationale": verdict_reason
        }
    }
