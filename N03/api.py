import os, joblib
import numpy as np
from flask import Flask, request, jsonify

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
PORT = int(os.environ.get("PORT", "8080"))

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    features = data.get("features")
    if features is None:
        return jsonify({"error": "Se espera JSON con 'features': [[...], [...]]"}), 400
    X = np.array(features, dtype=float)
    y_pred = model.predict(X).tolist()
    return jsonify({"predictions": y_pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
