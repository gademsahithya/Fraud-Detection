"""
app.py  -  Flask backend for the Fraud Detection UI

Run:
    python app.py
then open http://127.0.0.1:5000
"""
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

bundle = joblib.load("model.joblib")
MODEL = bundle["model"]
FEATURES = bundle["features"]
THRESHOLD = bundle["threshold"]

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

NUMERIC_FIELDS = [
    "step", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "origin_transaction_count", "origin_avg_amount", "origin_max_amount",
    "destination_transaction_count", "destination_unique_origins",
]


def safe_ratio(numerator, denominator):
    """Same rule as the notebook: ratio only when the denominator is > 0."""
    return numerator / denominator if denominator > 0 else 0


def build_row(data: dict) -> dict:
    """Turn the form values into the exact 17 features the model was trained on."""
    v = {k: float(data[k]) for k in NUMERIC_FIELDS}

    row = {
        "step": v["step"],
        "amount": v["amount"],
        "origin_transaction_count": v["origin_transaction_count"],
        "origin_avg_amount": v["origin_avg_amount"],
        "origin_max_amount": v["origin_max_amount"],
        "amount_vs_origin_avg": safe_ratio(v["amount"], v["origin_avg_amount"]),
        "destination_unique_origins": v["destination_unique_origins"],
        "destination_transaction_count": v["destination_transaction_count"],
        "origin_amount_ratio": safe_ratio(v["amount"], v["oldbalanceOrg"]),
        "origin_remaining_ratio": safe_ratio(v["newbalanceOrig"], v["oldbalanceOrg"]),
        "destination_amount_ratio": safe_ratio(v["amount"], v["oldbalanceDest"]),
        "destination_balance_ratio": safe_ratio(v["newbalanceDest"], v["oldbalanceDest"]),
    }
    for t in TYPES:
        row[f"type_{t}"] = 1 if data["type"] == t else 0

    # The notebook trained on df.astype(int), so we truncate the same way.
    return {k: int(val) for k, val in row.items()}


@app.route("/")
def home():
    return render_template("index.html", threshold=THRESHOLD)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    if data.get("type") not in TYPES:
        return jsonify(error="Choose a transaction type."), 400
    missing = [f for f in NUMERIC_FIELDS if data.get(f) in (None, "")]
    if missing:
        return jsonify(error=f"Missing values: {', '.join(missing)}"), 400
    try:
        row = build_row(data)
    except (TypeError, ValueError):
        return jsonify(error="All numeric fields must contain numbers."), 400
    if any(float(data[f]) < 0 for f in NUMERIC_FIELDS):
        return jsonify(error="Values cannot be negative."), 400

    X = pd.DataFrame([row])[FEATURES]
    probability = float(MODEL.predict_proba(X)[0, 1])

    if probability >= THRESHOLD:
        level = "high"
    elif probability >= THRESHOLD / 2:
        level = "elevated"
    else:
        level = "low"

    return jsonify(
        probability=probability,
        threshold=THRESHOLD,
        is_fraud=probability >= THRESHOLD,
        level=level,
        features=row,
    )


if __name__ == "__main__":
    app.run()
