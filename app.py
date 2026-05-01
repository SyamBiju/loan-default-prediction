from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("models/model.pkl")

@app.route("/")
def home():
    return "Loan Default Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    df = pd.DataFrame([data])
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "default_probability": float(probability)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)