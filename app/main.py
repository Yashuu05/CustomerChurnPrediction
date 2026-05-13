import os
import sys
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify

# Add project root to sys.path to allow loading custom pipeline steps
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

app = Flask(__name__)

# Load the model once at startup
MODEL_PATH = os.path.join(project_root, "models", "best_model_random_forest.joblib")
model = None
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

def get_customer_segment(tenure):
    """Segmentation logic based on tenure, aligning with DBSCAN training."""
    tenure = float(tenure)
    if tenure <= 12:
        return "Short Term"
    elif 13 <= tenure <= 48:
        return "Established"
    else:
        return "Loyalists"

@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Real prediction API using Random Forest and Segmentation logic."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    
    # 1. Map frontend fields to model features (19 features expected by X_res.csv)
    # Handle Phone and MultipleLines logic
    phone_service = "Yes" if data.get('phone') != "No" else "No"
    multiple_lines = "Yes" if data.get('phone') == "Multiple lines" else ("No" if data.get('phone') == "Single line" else "No phone service")
    
    # Handle Internet related services
    internet = data.get('internet')
    def get_service_val(key):
        if internet == "No":
            return "No internet service"
        return "Yes" if data.get(key) else "No"

    input_data = {
        "gender": data.get('gender'),
        "SeniorCitizen": 1 if data.get('senior') else 0,
        "Partner": "Yes" if data.get('partner') else "No",
        "Dependents": "Yes" if data.get('dependents') else "No",
        "tenure": float(data.get('tenure', 0)),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": get_service_val('security'),
        "OnlineBackup": get_service_val('backup'),
        "DeviceProtection": "No", # Defaulting as not in form
        "TechSupport": get_service_val('support'),
        "StreamingTV": get_service_val('streaming'),
        "StreamingMovies": get_service_val('streaming'),
        "Contract": data.get('contract'),
        "PaperlessBilling": "Yes", # Defaulting
        "PaymentMethod": data.get('payment'),
        "MonthlyCharges": float(data.get('monthly', 0)),
        "TotalCharges": float(data.get('total', 0))
    }

    # 2. Perform Prediction
    X_input = pd.DataFrame([input_data])
    
    # Get churn probability
    prob = model.predict_proba(X_input)[0][1]
    churn_probability = int(prob * 100)
    is_churn = prob > 0.5
    
    # 3. Get Customer Segment
    segment = get_customer_segment(data.get('tenure', 0))
    
    # 4. Generate dynamic drivers (Key Drivers)
    # We can use feature importance from the model, but for simplicity and UX,
    # we'll pick top features known to affect churn in this dataset.
    drivers = [
        {"feature": "Contract Type", "impact": 25 if data.get('contract') == "Month-to-month" else -15},
        {"feature": "Internet Service", "impact": 18 if data.get('internet') == "Fiber optic" else -10},
        {"feature": "Tenure", "impact": -20 if float(data.get('tenure', 0)) > 24 else 15},
        {"feature": "Monthly Charges", "impact": 12 if float(data.get('monthly', 0)) > 70 else -5}
    ]
    
    results = {
        "status": "CHURN LIKELY" if is_churn else "RETAINED",
        "probability": churn_probability,
        "segment": segment,
        "risk_level": "High" if churn_probability > 70 else ("Medium" if churn_probability > 30 else "Low"),
        "drivers": drivers,
        "insight": "Customer is at high risk due to short tenure and high monthly charges. Recommend switching to a long-term contract." if is_churn else "Customer shows high loyalty indicators. Continue engagement with value-added services."
    }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)