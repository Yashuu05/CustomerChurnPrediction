import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from src.logger import logging
from utils.data_utils import load_dataset
from utils.model_utils import save_model
from CustomerSegmentation.evaluation import evaluate_segmentation
from CustomerSegmentation.visualization import plot_customer_segments

def feature_engineering(df):
    """
    Create target_segment based on tenure logic.
    """
    def get_segment(tenure):
        if tenure <= 12:
            return "Short Term"
        elif 13 <= tenure <= 48:
            return "Established"
        else:
            return "Loyalists"
    
    df['target_segment'] = df['tenure'].apply(get_segment)
    return df

def train_segmentation():
    # Paths
    DATA_PATH = "data/processed/X_res.csv"
    MODEL_SAVE_PATH = "models/dbscan_segmentation_model.pkl"
    PLOT_SAVE_PATH = "outputs/customer_segmentation_scatter.png"
    EVAL_SAVE_PATH = "outputs/segmentation_evaluation.json"

    # MLflow Setup
    mlflow.set_experiment("Customer_Segmentation_DBSCAN")

    with mlflow.start_run():
        logging.info("Starting Customer Segmentation training...")

        # 1. Load Data
        df = load_dataset(DATA_PATH)
        if df is None:
            return

        # 2. Feature Engineering
        df = feature_engineering(df)
        logging.info("Feature engineering completed.")

        # 3. Preprocessing for DBSCAN
        # Using tenure and Contract to ensure 3 distinct clusters based on contract type
        # which correlates strongly with customer loyalty/tenure.
        X_raw = df[['tenure', 'Contract']]
        X_encoded = pd.get_dummies(X_raw)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # 4. Train DBSCAN
        # eps=0.5 works well with one-hot encoded Contract to find 3 clusters
        eps = 0.5 
        min_samples = 10
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        df['cluster'] = clusters

        # 5. Map clusters to labels
        cluster_mapping = {}
        unique_clusters = set(clusters)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
            cluster_mapping[-1] = "Noise"

        if len(unique_clusters) > 0:
            cluster_means = df[df['cluster'] != -1].groupby('cluster')['tenure'].mean().sort_values()
            
            # Assign labels based on mean tenure rank
            labels = ["Short Term", "Established", "Loyalists"]
            for i, (cluster_id, mean_tenure) in enumerate(cluster_means.items()):
                if i < len(labels):
                    cluster_mapping[cluster_id] = labels[i]
                else:
                    cluster_mapping[cluster_id] = f"Cluster_{cluster_id}"
        else:
            logging.warning("No clusters found!")

        df['customer_segment'] = df['cluster'].map(cluster_mapping)

        # 6. Evaluation
        eval_results = evaluate_segmentation(X_scaled, clusters, EVAL_SAVE_PATH, df)
        
        # 7. Visualization
        plot_customer_segments(df, 'tenure', 'MonthlyCharges', 'customer_segment', PLOT_SAVE_PATH)

        # 8. MLflow Logging
        mlflow.log_param("eps", eps)
        mlflow.log_param("min_samples", min_samples)
        mlflow.log_param("features", ['tenure', 'Contract'])
        if eval_results:
            mlflow.log_metric("silhouette_score", eval_results['silhouette_score'])
            mlflow.log_metric("num_clusters", eval_results['num_clusters'])
        
        mlflow.sklearn.log_model(dbscan, "dbscan_model")
        
        # 9. Save Model locally
        save_model(MODEL_SAVE_PATH, dbscan)
        
        logging.info("Customer Segmentation training and tracking completed.")
        print(f"Success! Training completed with {len(unique_clusters)} clusters.")

if __name__ == "__main__":
    train_segmentation()
