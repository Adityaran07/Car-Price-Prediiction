"""
app.py — Flask REST API for Car Price Prediction
Endpoints:
  POST /predict       → predict price for a single car
  POST /predict/batch → predict prices for multiple cars
  GET  /health        → health check
"""

from flask import Flask, request, jsonify
from src.car_price_model import CarPricePredictor
import logging

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
predictor = CarPricePredictor()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": predictor.model_name})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data provided"}), 400
    try:
        price = predictor.predict(data)
        return jsonify({"predicted_price_inr": round(price, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON list of car objects"}), 400
    try:
        prices = predictor.predict_batch(data)
        return jsonify({"predictions": [round(p, 2) for p in prices]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
