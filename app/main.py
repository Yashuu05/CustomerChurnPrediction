from flask import Flask, request, render_template, jsonify
import random

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Mock prediction API."""
    # In a real app, we would process form data and run the model here.
    # For now, we return mock results as requested.
    
    # Simulate a small delay for "AI processing"
    # import time; time.sleep(0.5) 
    
    churn_probability = random.randint(10, 95)
    is_churn = churn_probability > 50
    
    results = {
        "status": "CHURN LIKELY" if is_churn else "RETAINED",
        "probability": churn_probability,
        "risk_level": "High" if churn_probability > 70 else ("Medium" if churn_probability > 30 else "Low"),
        "drivers": [
            {"feature": "Month-to-Month Contract", "impact": random.randint(10, 30)},
            {"feature": "Fiber Optic Service", "impact": random.randint(5, 20)},
            {"feature": "Paperless Billing", "impact": random.randint(5, 15)},
            {"feature": "Streaming Content", "impact": random.randint(-20, 10)}
        ],
        "insight": "Customer is highly sensitive to contract type. Offer a 15% discount for switching to a Two-year contract to improve retention probability." if is_churn else "Customer loyalty is strong. Maintain current service level and monitor for changes in usage patterns."
    }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)