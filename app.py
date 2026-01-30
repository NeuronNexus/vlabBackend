from flask import Flask, request, jsonify
from typing import Dict, Any
import numpy as np
import os
from flask_cors import CORS
from core.orchestrator import (
    run_simulation,
    run_scenario_comparison
)
from utils.validators import validate_simulation_inputs

app = Flask(__name__)

# ============================================================
# CORS CONFIGURATION (FIXED)
# ============================================================

allowed_origins = [
    # Local development
    "http://localhost:8080",
    "http://localhost:5173",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5173",

    # Production (Vercel)
    "https://solesenses-alpha.vercel.app",
]

# Optional: allow dynamic frontend via env (Render-safe)
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    allowed_origins.append(frontend_url.rstrip("/"))

CORS(
    app,
    origins=allowed_origins,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ============================================================
# JSON Serialization Boundary
# ============================================================

def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


# ============================================================
# Response Construction
# ============================================================

def _build_simulation_response(sim_result: Dict[str, Any]) -> Dict[str, Any]:
    comfort_start = sim_result["comfort_history"][0]["comfort_index"]
    comfort_end = sim_result["comfort_history"][-1]["comfort_index"]

    return _json_safe({
        "overview": {
            "scenario_type": sim_result["scenario_summary"]["scenario_type"],
            "stability": sim_result["scenario_summary"]["stability"],
            "alignment_regime": sim_result["alignment_summary"]["alignment_regime"],
            "comfort_change": comfort_end - comfort_start,
        },
        "key_drivers": {
            "dominant_pressure_factors": sim_result["scenario_summary"]["dominant_factors"],
            "scenario_explanation": sim_result["scenario_summary"]["explanation"],
            "alignment_interpretation": sim_result["alignment_summary"]["interpretation"],
        },
        "evidence": {
            "final_comfort": comfort_end,
            "mean_wear": float(sim_result["final_wear"].mean()),
            "max_wear": float(sim_result["final_wear"].max()),
            "comfort_drop_normalized": sim_result["alignment_summary"]["comfort_drop_normalized"],
            "wear_growth_normalized": sim_result["alignment_summary"]["wear_growth_normalized"],
        },
        "raw": sim_result,
    })


def _build_comparison_response(compare_result: Dict[str, Any]) -> Dict[str, Any]:
    verdict = compare_result["what_if_analysis"]["verdict"]

    return _json_safe({
        "overview": {
            "decision": verdict["classification"],
            "rationale": verdict["rationale"],
        },
        "baseline": compare_result["baseline"],
        "variant": compare_result["variant"],
        "analysis": compare_result["what_if_analysis"],
        "model_assumptions": compare_result["model_assumptions"],
    })


# ============================================================
# Routes
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "SoleSense API",
        "version": "1.0.0"
    })


@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        payload = request.get_json(force=True)
        steps = int(payload.get("steps", 50))
        sim_inputs = validate_simulation_inputs(payload)
        result = run_simulation(sim_inputs, steps=steps)
        return jsonify(_build_simulation_response(result))
    except Exception as e:
        return jsonify({"error": "Simulation failed", "message": str(e)}), 400


@app.route("/compare", methods=["POST"])
def compare():
    try:
        payload = request.get_json(force=True)
        steps = int(payload.get("steps", 50))

        baseline = validate_simulation_inputs(payload["baseline"])
        variant = validate_simulation_inputs(payload["variant"])

        result = run_scenario_comparison(
            baseline_inputs=baseline,
            variant_inputs=variant,
            steps=steps
        )

        return jsonify(_build_comparison_response(result))
    except Exception as e:
        return jsonify({"error": "Scenario comparison failed", "message": str(e)}), 400


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=os.getenv("FLASK_ENV") == "development",
        use_reloader=False
    )
