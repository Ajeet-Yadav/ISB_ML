from flask import Flask, request, jsonify
import joblib
import pandas as pd

MODEL_PATH = "bank_term_deposit_model.joblib"
model = joblib.load(MODEL_PATH)

EXPECTED_FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

NUMERIC_FEATURES = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON body provided"}), 400

    missing = [f for f in EXPECTED_FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing features", "missing": missing}), 400

    row = [[data[feat] for feat in EXPECTED_FEATURES]]
    df_input = pd.DataFrame(row, columns=EXPECTED_FEATURES)

    # Ensure numeric columns are numeric
    for col in NUMERIC_FEATURES:
        df_input[col] = pd.to_numeric(df_input[col])

    proba = model.predict_proba(df_input)[0, 1]
    pred_class = int(proba >= 0.5)
    label = "yes" if pred_class == 1 else "no"

    response = {
        "prediction": label,
        "prediction_proba": float(proba),
        "details": {
            "threshold": 0.5,
        },
    }

    return jsonify(response), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
