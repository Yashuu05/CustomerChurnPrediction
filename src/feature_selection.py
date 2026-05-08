import pandas as pd
import os
import sys 
import json
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.logger import logging
from utils.data_utils import save_dataset
SAVE_FILE_PATH = os.path.join(project_root, "outputs", "perm_imp.csv")

def get_low_importance_features(importance_df, threshold=0.0):
    """
    Identifies features with importance scores at or below the threshold.
    
    Args:
        importance_df (pd.DataFrame): DataFrame with 'Feature' and 'Importance' columns.
        threshold (float): Importance threshold. Features <= threshold will be returned.
        
    Returns:
        pd.DataFrame: Features to consider dropping.
    """
    low_importance = importance_df[importance_df['Importance'] <= threshold]
    logging.info(f"Identified {len(low_importance)} features with importance <= {threshold}")
    return low_importance

def suggest_original_feature_drops(importance_df):
    """
    Groups one-hot encoded features and suggests original features to drop 
    if all their categories have low importance.
    """
    # Extract original feature names (assuming 'cat__Feature_Category' format)
    importance_df['OriginalFeature'] = importance_df['Feature'].apply(
        lambda x: x.split('__')[1].split('_')[0] if '__' in x else x.split('__')[-1]
    )
    
    feature_stats = importance_df.groupby('OriginalFeature')['Importance'].agg(['max', 'mean']).reset_index()
    
    # Suggest dropping if the maximum importance across all categories is very low
    suggested_drops = feature_stats[feature_stats['max'] <= 0.001]
    
    return suggested_drops

if __name__ == "__main__":
    # Example usage (simulated data)
    data = {
        'Feature': [
            "cat__Contract_Month-to-month",
            "cat__InternetService_Fiber optic",	
            "cat__OnlineSecurity_No",	
            "cat__Contract_Two year",	
            "cat__PhoneService_No",
            "cat__StreamingMovies_Yes",	
            "cat__TechSupport_No",	
            "cat__Contract_One year",	
            "num__tenure",  
            "cat__OnlineBackup_No",	
            "cat__StreamingMovies_No",	
            "cat__DeviceProtection_Yes",	
            "cat__PaymentMethod_Electronic check",
            "cat__MultipleLines_No",	
            "remainder__SeniorCitizen",	
            "num__MonthlyCharges",	
            "num__TotalCharges",	
            "cat__PaperlessBilling_No",	
            "cat__MultipleLines_Yes",
            "cat__PaymentMethod_Bank transfer (automatic)",
            "cat__StreamingTV_Yes",	
            "cat__PaymentMethod_Credit card (automatic)",
            "cat__Dependents_No",
            "cat__gender_Female",	
            "cat__OnlineBackup_Yes",
            "cat__PaymentMethod_Mailed check",	
            "cat__DeviceProtection_No",
            "cat__Partner_No",
            "cat__StreamingTV_No",	
            "cat__InternetService_DSL",	
            "cat__TechSupport_Yes",	
            "cat__OnlineSecurity_Yes",	
            "cat__MultipleLines_No phone service",	
            "cat__gender_Male",	
            "cat__PhoneService_Yes",
            "cat__Partner_Yes",	
            "cat__Dependents_Yes",	
            "cat__TechSupport_No internet service",	
            "cat__DeviceProtection_No internet service",	
            "cat__OnlineBackup_No internet service",	
            "cat__OnlineSecurity_No internet service",	
            "cat__InternetService_No",	
            "cat__StreamingTV_No internet service",	
            "cat__StreamingMovies_No internet service",	
            "cat__PaperlessBilling_Yes",	
        ],
        'Importance': [
            0.426870,
            0.174917,
            0.034330,
            0.031609,
            0.024113,
            0.022608,
            0.020701,
            0.018461,
            0.016337,
            0.014211,
            0.012710,
            0.012155,
            0.011823,
            0.011771,
            0.011582,
            0.011384,
            0.011360,
            0.010709,
            0.010405,
            0.010156,
            0.010135,
            0.009606,
            0.009527,
            0.009251,
            0.009027,
            0.009015,
            0.008548,
            0.008452,
            0.008172,
            0.007369,
            0.006845,
            0.005843,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000
        ]
    }
    df = pd.DataFrame(data)
    print("features with low importance:")
    result = get_low_importance_features(importance_df=df, threshold=0.005)
    print(result)
    
    print("\nSuggested Drops")
    suggested_drops = suggest_original_feature_drops(importance_df=df)
    print(suggested_drops)
    
    # Save the dataframe
    #os.makedirs(os.path.dirname(SAVE_FILE_PATH), exist_ok=True)
    #save_dataset(data=df, file_path=SAVE_FILE_PATH)
    #logging.info(f"Saved permutation importance to {SAVE_FILE_PATH}")
    
    # Save features with low importance in congig/
    CONFIG_PATH = os.path.join(project_root, "configs", "suggested_drops.json")
    drops_list = result['Feature'].tolist()
    with open(CONFIG_PATH, 'w') as f:
        json.dump(drops_list, f, indent=4)
    logging.info(f"Saved suggested drops to {CONFIG_PATH}")