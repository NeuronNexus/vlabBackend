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

# Configure CORS - allow requests from frontend
allowed_origins = [
    "http://localhost:8080",
    "http://localhost:5173",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5173",
]

# Allow setting the frontend origin via environment variable (no trailing slash)
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    frontend_url = frontend_url.rstrip('/')
    allowed_origins.append(frontend_url)

CORS(app, resources={
    r"/*": {
        "origins": allowed_origins,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
}, supports_credentials=False)


# Fallback: ensure CORS headers are present on all responses
@app.after_request
def _ensure_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin and origin.rstrip('/') in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        # Do not set credentials unless you explicitly need them
        # response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

#done
# ============================================================
# JSON Serialization Boundary
# ============================================================

def _json_safe(obj):
    """
    Recursively convert NumPy objects into JSON-safe types.
    """
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
# Response Construction (Narrative Layer)
# ============================================================

def _build_simulation_response(sim_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct a semantically rich response without
    reinterpreting or modifying core outputs.
    """

    comfort_start = sim_result["comfort_history"][0]["comfort_index"]
    comfort_end = sim_result["comfort_history"][-1]["comfort_index"]

    overview = {
        "scenario_type": sim_result["scenario_summary"]["scenario_type"],
        "stability": sim_result["scenario_summary"]["stability"],
        "alignment_regime": sim_result["alignment_summary"]["alignment_regime"],
        "comfort_change": comfort_end - comfort_start,
    }

    key_drivers = {
        "dominant_pressure_factors":
            sim_result["scenario_summary"]["dominant_factors"],
        "scenario_explanation":
            sim_result["scenario_summary"]["explanation"],
        "alignment_interpretation":
            sim_result["alignment_summary"]["interpretation"],
    }

    evidence = {
        "final_comfort": comfort_end,
        "mean_wear": float(sim_result["final_wear"].mean()),
        "max_wear": float(sim_result["final_wear"].max()),
        "comfort_drop_normalized":
            sim_result["alignment_summary"]["comfort_drop_normalized"],
        "wear_growth_normalized":
            sim_result["alignment_summary"]["wear_growth_normalized"],
    }

    return _json_safe({
        "overview": overview,
        "key_drivers": key_drivers,
        "evidence": evidence,
        "raw": sim_result,
    })


def _build_comparison_response(compare_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a decision-grade what-if response.
    """

    verdict = compare_result["what_if_analysis"]["verdict"]

    overview = {
        "decision": verdict["classification"],
        "rationale": verdict["rationale"],
    }

    return _json_safe({
        "overview": overview,
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
    """
    Health check endpoint for monitoring backend availability
    """
    return jsonify({
        "status": "healthy",
        "service": "SoleSense API",
        "version": "1.0.0"
    })


@app.route("/simulate", methods=["POST", "OPTIONS"])
def simulate():
    """
    Run a single SoleSense simulation.
    """
    if request.method == "OPTIONS":
        return "", 200
    try:
        payload = request.get_json(force=True)

        steps = int(payload.get("steps", 50))
        sim_inputs = validate_simulation_inputs(payload)

        result = run_simulation(sim_inputs, steps=steps)
        response = _build_simulation_response(result)

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": "Simulation failed",
            "message": str(e)
        }), 400


@app.route("/compare", methods=["POST", "OPTIONS"])
def compare():
    """
    Run a deterministic what-if comparison between two scenarios.
    """
    if request.method == "OPTIONS":
        return "", 200
    try:
        payload = request.get_json(force=True)

        if "baseline" not in payload or "variant" not in payload:
            raise ValueError(
                "Request must contain both 'baseline' and 'variant' objects."
            )

        steps = int(payload.get("steps", 50))

        baseline = validate_simulation_inputs(payload["baseline"])
        variant = validate_simulation_inputs(payload["variant"])

        result = run_scenario_comparison(
            baseline_inputs=baseline,
            variant_inputs=variant,
            steps=steps
        )

        response = _build_comparison_response(result)
        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": "Scenario comparison failed",
            "message": str(e)
        }), 400


# ============================================================
# Error Handlers
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint does not exist"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }), 500


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 SoleSense Backend API Starting...")
    print("=" * 60)
    print(f"📍 API URL: http://0.0.0.0:5000")
    print(f"🔍 Health Check: http://0.0.0.0:5000/health")
    print(f"📊 Simulate: POST http://0.0.0.0:5000/simulate")
    print(f"⚖️  Compare: POST http://0.0.0.0:5000/compare")
    print("=" * 60)
    debug_mode = os.getenv("FLASK_ENV") == "development"
    app.run(debug=debug_mode, host='0.0.0.0', use_reloader=False, port=int(os.getenv("PORT", 5000)))