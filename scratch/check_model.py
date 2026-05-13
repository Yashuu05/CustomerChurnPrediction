import joblib
import os
import sys
project_root = r'd:\projects\ChurnPrediction'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
model_path = os.path.join(project_root, 'models', 'best_model_random_forest.joblib')
model = joblib.load(model_path)
print(f"Model type: {type(model)}")
if hasattr(model, 'steps'):
    print(f"Pipeline steps: {[s[0] for s in model.steps]}")
