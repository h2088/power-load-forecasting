from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.forecast_logic import ForecastRequestError, get_model_info, run_forecast


app = Flask(__name__)
app.json.ensure_ascii = False


@app.get("/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200


@app.get("/api/model-info")
def model_info() -> tuple["Response", int]:
    return jsonify(get_model_info()), 200


@app.get("/api/forecast/latest")
def latest_forecast():
    return jsonify(run_forecast()), 200


@app.post("/api/predict")
def predict_route():
    payload = request.get_json(silent=True) or {}
    target_date = payload.get("target_date") or request.args.get("target_date")
    return jsonify(run_forecast(target_date)), 200


@app.errorhandler(ForecastRequestError)
def handle_forecast_error(error: ForecastRequestError):
    return jsonify({"error": str(error)}), 400


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    app.logger.exception("Unexpected backend failure", exc_info=error)
    return jsonify({"error": "Unexpected backend failure"}), 500


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug, use_reloader=False)
