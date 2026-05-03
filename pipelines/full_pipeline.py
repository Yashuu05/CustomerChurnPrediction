import json
import os
from pipelines.model_pipeline import build_model_pipelines
from sklearn.model_selection import RandomizedSearchCV

def build_full_pipeline(categorical_cols: list, numerical_cols: list) -> dict:
    """
    Builds a dictionary of RandomizedSearchCV objects for each model.
    """
    # Fix the file path and load hyperparameters correctly
    hyperparams_path = os.path.join('configs', 'hyperparameters.json')
    with open(hyperparams_path, 'r') as file:
        hyperparams_data = json.load(file)

    # Get the dictionary of model pipelines
    model_pipelines = build_model_pipelines(cat_cols=categorical_cols, num_cols=numerical_cols)

    grid_searches = {}

    # Dynamically setup RandomizedSearchCV for each model
    for model_name, pipeline in model_pipelines.items():
        if model_name in hyperparams_data:
            grid_searches[model_name] = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=hyperparams_data[model_name],
                cv=5,
                scoring='f1',  # Corrected from 'f1_score' to 'f1'
                n_iter=10,      # Added n_iter for RandomizedSearchCV
                n_jobs=-1,
                verbose=2,
                random_state=42
            )
        else:
            print(f"Warning: No hyperparameters found for model '{model_name}'. Skipping grid search setup.")

    return grid_searches