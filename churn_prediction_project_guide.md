# Customer Churn Prediction: Industry-Level Project Structure

---

## **1. PROJECT FOLDER STRUCTURE**

```
customer-churn-prediction/
│
├── README.md                          # Project overview, setup instructions
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
├── config.yaml                        # Configuration (paths, hyperparameters)
│
├── data/
│   ├── raw/                           # Original, untouched data
│   │   └── telecom_churn.csv
│   ├── processed/                     # Cleaned, transformed data
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   └── features_metadata.json     # Feature descriptions
│   └── external/                      # External reference data (if any)
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb       # EDA & initial insights
│   ├── 02_feature_engineering.ipynb             # Feature creation & transformation
│   ├── 03_model_development.ipynb               # Model training & comparison
│   └── 04_model_interpretation.ipynb            # SHAP, feature importance
│
├── src/                               # Source code (reusable modules)
│   ├── __init__.py
│   ├── data_loader.py                 # Load & validate data
│   ├── preprocessing.py               # Data cleaning, scaling, imputation
│   ├── feature_engineering.py         # Create new features (RFM, behavioral)
│   ├── models.py                      # Model training & evaluation
│   ├── evaluation.py                  # Metrics, confusion matrix, ROC curves
│   ├── interpretation.py              # SHAP, feature importance
│   └── utils.py                       # Helper functions
│
├── models/                            # Trained model artifacts
│   ├── best_model.pkl                 # Serialized trained model
│   ├── scaler.pkl                     # Feature scaler
│   ├── encoder.pkl                    # Categorical encoder
│   └── model_metadata.json            # Model version, training date, metrics
│
├── reports/                           # Generated reports & visualizations
│   ├── eda_report.html                # Interactive EDA report
│   ├── model_performance_report.pdf   # Model evaluation metrics
│   ├── feature_importance.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   └── business_recommendations.md    # Actionable insights
│
├── scripts/
│   ├── train_pipeline.py              # End-to-end training script
│   ├── predict_pipeline.py            # Make predictions on new data
│   ├── evaluate_model.py              # Performance evaluation
│   └── deploy_preparation.py          # Prepare model for production
│
├── tests/                             # Unit tests & validation
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   └── test_utils.py
│
├── configs/                           # Configuration files
│   ├── default_config.yaml            # Default hyperparameters
│   ├── hyperparameters.yaml           # Optimized hyperparameters
│   └── features_config.json           # Feature list & types
│
└── outputs/                           # Final predictions & results
    ├── predictions.csv                # Model predictions on test set
    ├── churn_scores.csv               # Churn probability for customers
    └── model_comparison.csv           # Performance across models

```

---

## **2. KEY TOOLS & LIBRARIES**

### **Core ML Stack**
```
scikit-learn==1.3.2          # RandomForest, preprocessing, metrics
xgboost==2.0.0               # XGBoost classifier
pandas==2.1.0                # Data manipulation
numpy==1.24.0                # Numerical computing
matplotlib==3.8.0            # Basic plotting
seaborn==0.13.0              # Statistical visualization
plotly==5.18.0               # Interactive visualizations
```

### **Advanced Tools (NEW - Learn These!)**
```
imbalanced-learn==0.11.0     # SMOTE for handling class imbalance
shap==0.43.0                 # Model interpretation (SHAP values)
optuna==3.14.0               # Hyperparameter optimization (Bayesian)
category-encoders==2.6.0     # Advanced categorical encoding
scikit-optimize==0.9.0       # Hyperparameter tuning
```

### **Data Profiling & EDA**
```
pandas-profiling==3.6.1      # Automated EDA reports
sweetviz==2.2.1              # Interactive visualization
missingno==0.5.2             # Missing data visualization
```

### **Production & Deployment**
```
joblib==1.3.2                # Save/load trained models
pyyaml==6.0.1                # Configuration management
python-dotenv==1.0.0         # Environment variables
```

### **Monitoring & Logging**
```
python-json-logger==2.0.7    # JSON logging
```

---

## **3. INDUSTRY-LEVEL ALGORITHMS & CONCEPTS**

### **A. CLASSIFICATION ALGORITHMS** (Compare these)

| Algorithm | Why Use It | When to Use | Pros | Cons |
|-----------|-----------|-----------|------|------|
| **Logistic Regression** | Baseline, interpretable | First pass, explainability needed | Fast, interpretable | Linear boundaries only |
| **Decision Tree** | Baseline, tree-based | Feature interaction exploration | Interpretable, handles non-linearity | Prone to overfitting |
| **Random Forest** | Ensemble, robust | Strong baseline | Robust, handles non-linearity | Less interpretable |
| **XGBoost** | SOTA, gradient boosting | When accuracy matters most | Highest accuracy, feature importance | Slower training, harder to tune |
| **LightGBM** | Fast gradient boosting | Large datasets | Faster than XGBoost | May overfit on small data |
| **Gradient Boosting** | Ensemble, powerful | High accuracy needed | Powerful, good generalization | Slow to train |
| **SVM (SVC)** | Non-linear boundaries | You've used it! | Handles high dimensions | Slow on large data, hard to tune |

**Recommendation:** Train **Logistic Regression → Random Forest → XGBoost** pipeline, compare using ROC-AUC.

---

### **B. CRITICAL CONCEPTS TO MASTER**

#### **1. Class Imbalance Handling** ⭐ CRITICAL
Most churn datasets are imbalanced (e.g., 80% retained, 20% churned).

**Techniques:**
```
a) SMOTE (Synthetic Minority Over-sampling Technique)
   - Generate synthetic minority samples
   - Use: from imblearn.over_sampling import SMOTE
   
b) Class Weights
   - Give more penalty to minority class
   - XGBoost: scale_pos_weight parameter
   
c) Stratified Sampling
   - Ensure balanced train-test split
   - Use: train_test_split(stratify=y)
   
d) Threshold Tuning
   - Adjust decision boundary (default 0.5)
   - Optimize for business metric (precision vs recall)
```

#### **2. Feature Engineering for Churn** ⭐ KEY DIFFERENTIATOR
Raw features → Business-relevant features

**RFM Analysis (Recency, Frequency, Monetary):**
```python
# Recency: Days since last purchase
# Frequency: Number of transactions in period
# Monetary: Total spend in period

# Example:
recency = (today - last_transaction_date).days
frequency = count_of_transactions
monetary = total_revenue
```

**Behavioral Features:**
```python
# Customer engagement metrics
- Contract length
- Monthly charges trend
- Service usage patterns (calls, data usage)
- Customer service interactions
- Internet service type combinations
- Payment method changes
- Services adopted over time
```

**Temporal Features:**
```python
# Time-based patterns
- Days as customer (tenure)
- Months since contract start
- Service tenure ratio
- Usage trend (increasing/decreasing)
```

#### **3. Evaluation Metrics** ⭐ MORE THAN ACCURACY

```
1. Precision: Of predicted churners, how many actually churn?
   - Important: Avoid wasting retention budget on false positives
   
2. Recall: Of actual churners, how many did we catch?
   - Important: Don't miss customers we could have saved
   
3. F1-Score: Harmonic mean of precision & recall
   - Balanced metric for imbalanced data
   
4. ROC-AUC: Area under ROC curve
   - Best for imbalanced classification
   - Threshold-independent evaluation
   
5. PR-AUC: Area under Precision-Recall curve
   - Better than ROC-AUC for imbalanced data
   
6. Confusion Matrix:
   - TP, TN, FP, FN visualization
   
7. Cost-Sensitive Metrics:
   - Cost of false positive (wrong retention offer)
   - Cost of false negative (lost customer)
   - Business-driven evaluation
```

**For this project:** Use **ROC-AUC** as primary, **PR-AUC** as secondary.

#### **4. Model Interpretability** ⭐ REQUIRED FOR PRODUCTION

**SHAP (SHapley Additive exPlanations):**
```python
import shap

# Explain individual predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot: Why did model predict this customer will churn?
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Plot: Feature importance across dataset
shap.summary_plot(shap_values, X_test)
```

**Benefits:**
- Explains individual predictions (not just global importance)
- Shows feature contribution direction (+/- impact)
- Builds stakeholder trust

#### **5. Hyperparameter Optimization** ⭐ MAXIMIZE PERFORMANCE

**Methods:**
```
a) Grid Search
   - Try all parameter combinations
   - Slow but thorough
   
b) Random Search
   - Random sampling of parameters
   - Faster than grid search
   
c) Bayesian Optimization (Optuna)
   - Smart sampling based on past results
   - Efficient, faster convergence
   - RECOMMENDED: Optuna
```

**Example (Optuna):**
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model.score(X_val, y_val)

study = optuna.create_study()
study.optimize(objective, n_trials=100)
best_params = study.best_params
```

#### **6. Cross-Validation Strategy**

```python
# Use StratifiedKFold for imbalanced data
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Train model on fold
```

#### **7. Feature Selection** (Remove irrelevant features)

```python
# Method 1: Feature importance from tree models
feature_importance = model.feature_importances_

# Method 2: Permutation importance
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(model, X_test, y_test)

# Method 3: Recursive Feature Elimination
from sklearn.feature_selection import RFE
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X_train, y_train)
```

---

## **4. STEP-BY-STEP IMPLEMENTATION PIPELINE**

### **Phase 1: Data Loading & Exploration (Week 1)**

```python
# 1. Load data
df = pd.read_csv('data/raw/telecom_churn.csv')

# 2. Basic exploration
print(f"Shape: {df.shape}")
print(f"Churn distribution: {df['Churn'].value_counts()}")
print(f"Missing values: {df.isnull().sum()}")

# 3. Visualize
import seaborn as sns
sns.histplot(data=df, x='MonthlyCharges', hue='Churn')
sns.heatmap(df.corr(), cmap='coolwarm')

# 4. Automated EDA report
from pandas_profiling import ProfileReport
report = ProfileReport(df)
report.to_file("reports/eda_report.html")
```

### **Phase 2: Data Preprocessing (Week 1-2)**

```python
# 1. Handle missing values
df['column_name'].fillna(df['column_name'].median(), inplace=True)

# 2. Handle outliers
from scipy import stats
z_scores = stats.zscore(df[numeric_cols])
df = df[(z_scores < 3).all(axis=1)]

# 3. Encode categorical variables
from category_encoders import TargetEncoder
encoder = TargetEncoder()
df[categorical_cols] = encoder.fit_transform(df[categorical_cols], df['Churn'])

# 4. Scale numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 5. Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### **Phase 3: Feature Engineering (Week 2)**

```python
# Create RFM features (if temporal data available)
def create_rfm_features(df):
    df['tenure_months'] = df['tenure_days'] / 30
    df['monthly_to_lifetime_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['contract_flexibility'] = df['contract_length']  # Proxy
    return df

# Create behavioral features
def create_behavioral_features(df):
    df['num_services'] = df[['InternetService', 'OnlineSecurity', 
                              'OnlineBackup', 'DeviceProtection']].notna().sum(axis=1)
    df['has_support'] = ((df['TechSupport'] == 'Yes') | 
                         (df['CustomerService'] == 'Yes')).astype(int)
    return df

X_train_engineered = create_rfm_features(X_train)
X_train_engineered = create_behavioral_features(X_train_engineered)
```

### **Phase 4: Handle Class Imbalance (Week 2)**

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# SMOTE to balance training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {y_train.value_counts()}")
print(f"After SMOTE: {y_train_balanced.value_counts()}")
```

### **Phase 5: Model Training & Comparison (Week 3)**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15),
    'XGBoost': xgb.XGBClassifier(scale_pos_weight=4, random_state=42)  # Weight for imbalance
}

results = {}

for name, model in models.items():
    # Train with cross-validation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                cv=5, scoring='roc_auc')
    
    # Fit on full training data
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'cv_scores': cv_scores,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"{name}: ROC-AUC = {roc_auc:.4f}, CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Select best model
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_model = results[best_model_name]['model']
```

### **Phase 6: Hyperparameter Tuning (Week 3)**

```python
import optuna
from optuna.samplers import TPESampler

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    
    model = xgb.XGBClassifier(**params, scale_pos_weight=4, random_state=42)
    
    # Use cross-validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X_train_balanced, y_train_balanced, 
                            cv=5, scoring='roc_auc')
    
    return scores.mean()

sampler = TPESampler(seed=42)
study = optuna.create_study(sampler=sampler, direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

best_params = study.best_params
best_model = xgb.XGBClassifier(**best_params, scale_pos_weight=4, random_state=42)
best_model.fit(X_train_balanced, y_train_balanced)
```

### **Phase 7: Model Interpretation (Week 4)**

```python
import shap

# Initialize SHAP explainer
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig('reports/shap_feature_importance.png')

# Individual prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], show=False)
plt.savefig('reports/shap_individual_prediction.png')

# Dependence plots
shap.dependence_plot("MonthlyCharges", shap_values, X_test, show=False)
plt.savefig('reports/shap_dependence.png')
```

### **Phase 8: Evaluation & Business Metrics (Week 4)**

```python
from sklearn.metrics import (confusion_matrix, classification_report, 
                            precision_recall_curve, auc, roc_curve, roc_auc_score)
import matplotlib.pyplot as plt

# 1. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('reports/roc_curve.png')

# 2. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('reports/pr_curve.png')

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.savefig('reports/confusion_matrix.png')

# 4. Classification Report
print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
```

### **Phase 9: Threshold Tuning for Business Impact (Week 4)**

```python
# Default threshold is 0.5, but adjust based on business needs
# Example: Prioritize recall (catch more churners) over precision

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find threshold that maximizes F1-score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

# Apply optimized threshold
y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)

print(f"Best threshold: {best_threshold:.3f}")
print(f"Precision: {precision[best_threshold_idx]:.3f}, Recall: {recall[best_threshold_idx]:.3f}")
```

### **Phase 10: Generate Business Recommendations (Week 5)**

```python
# Identify high-risk churn segments
high_churn_risk = X_test[y_pred_proba > 0.7]

# Top features driving churn for this segment
churn_drivers = high_churn_risk[top_features].describe()

# Generate actionable recommendations
recommendations = f"""
CUSTOMER CHURN PREDICTION: BUSINESS RECOMMENDATIONS

1. HIGH-RISK SEGMENT (Churn Probability > 70%):
   - {len(high_churn_risk)} customers identified
   - Primary churn drivers: Monthly charges, contract length
   - Recommended action: Immediate retention offers

2. MEDIUM-RISK SEGMENT (Churn Probability 40-70%):
   - {len(X_test[(y_pred_proba > 0.4) & (y_pred_proba <= 0.7)])} customers
   - Recommended action: Loyalty programs, service upgrades

3. TARGET CUSTOMER PROFILE FOR RETENTION:
   - Monthly charges: > $65
   - Tenure: < 12 months
   - Services: Internet service without support

4. COST-BENEFIT ANALYSIS:
   - Retention offer cost: $50/customer
   - Average customer lifetime value: $2,500
   - Expected customers saved: {int(len(high_churn_risk) * recall[best_threshold_idx])}
   - ROI: {(int(len(high_churn_risk) * recall[best_threshold_idx]) * 2500 - len(high_churn_risk) * 50) / (len(high_churn_risk) * 50):.2f}x
"""

with open('reports/business_recommendations.md', 'w') as f:
    f.write(recommendations)
```

---

## **5. MODEL SERIALIZATION & DEPLOYMENT PREPARATION**

```python
import joblib
import json
from datetime import datetime

# Save trained model
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoder, 'models/encoder.pkl')

# Save model metadata
model_metadata = {
    'model_type': 'XGBoost',
    'training_date': datetime.now().isoformat(),
    'train_samples': len(X_train_balanced),
    'test_auc': float(roc_auc_score(y_test, y_pred_proba)),
    'hyperparameters': best_params,
    'features': X_train.columns.tolist(),
    'threshold': float(best_threshold),
    'version': '1.0'
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=4)
```

### **Production Prediction Script**

```python
# predict_pipeline.py
import joblib
import pandas as pd
import numpy as np

class ChurnPredictor:
    def __init__(self, model_path='models/best_model.pkl',
                 scaler_path='models/scaler.pkl',
                 metadata_path='models/model_metadata.json'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.threshold = self.metadata['threshold']
    
    def predict(self, customer_data):
        """
        Input: DataFrame with customer features
        Output: DataFrame with predictions and probabilities
        """
        # Preprocess
        X_processed = self.scaler.transform(customer_data)
        
        # Predict
        churn_proba = self.model.predict_proba(X_processed)[:, 1]
        churn_pred = (churn_proba >= self.threshold).astype(int)
        
        results = pd.DataFrame({
            'customer_id': customer_data.index,
            'churn_probability': churn_proba,
            'churn_prediction': churn_pred,
            'risk_segment': pd.cut(churn_proba, bins=[0, 0.4, 0.7, 1.0],
                                  labels=['Low', 'Medium', 'High'])
        })
        
        return results

# Usage
predictor = ChurnPredictor()
new_customers = pd.read_csv('data/new_customers.csv')
predictions = predictor.predict(new_customers)
predictions.to_csv('outputs/churn_predictions.csv', index=False)
```

---

## **6. EVALUATION CHECKLIST**

- [ ] EDA completed with insights documented
- [ ] Missing values handled appropriately
- [ ] Outliers detected and addressed
- [ ] Features engineered (RFM, behavioral)
- [ ] Class imbalance handled (SMOTE applied)
- [ ] Train-test split stratified
- [ ] Multiple models trained & compared
- [ ] Hyperparameters optimized (Optuna)
- [ ] Cross-validation performed (StratifiedKFold)
- [ ] Model evaluated with ROC-AUC and PR-AUC
- [ ] SHAP interpretability analysis done
- [ ] Business recommendations generated
- [ ] Threshold tuned for business objectives
- [ ] Model serialized and saved
- [ ] Prediction pipeline created
- [ ] Code documented and tested
- [ ] Reports generated (visualizations, metrics)

---

## **7. TIMELINE**

| Week | Task | Deliverable |
|------|------|-------------|
| **1** | EDA + Preprocessing | Cleaned data, insight document |
| **2** | Feature Engineering + SMOTE | Engineered features, balanced dataset |
| **3** | Model Development + Tuning | 3+ models trained, best model selected |
| **4** | Interpretation + Evaluation | SHAP analysis, metrics, ROC curves |
| **5** | Business Recommendations + Deployment Prep | Final report, prediction pipeline |

---

## **8. ADVANCED CONCEPTS TO EXPLORE**

1. **Ensemble Methods:** Stacking, Voting, Blending
2. **Feature Interactions:** Polynomial features, interaction terms
3. **Temporal Validation:** Time-based cross-validation if data is temporal
4. **Fairness & Bias:** Ensure model doesn't discriminate unfairly
5. **Model Monitoring:** Track performance drift in production
6. **A/B Testing:** Validate business impact before full deployment

---

## **GITHUB REPOSITORY STRUCTURE (Professional)**

```
README.md (setup, quick start, project overview)
LICENSE (choose appropriate license)
.github/
    ├── workflows/
    │   ├── ci.yml (continuous integration tests)
    │   └── deploy.yml (automated deployment)
CONTRIBUTING.md (contribution guidelines)
```

---

This is a **production-grade structure** used in real companies!

