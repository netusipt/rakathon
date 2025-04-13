from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading and preprocessing onco_data.csv...")
# Load the data
df = pd.read_csv("/home/petrn/python/rackathon/main/viktor/new/rakathon/onco_data.csv")

# Create relapse target variable (as in your current code)
df["relapse"] = ((df[["je_disp", "je_nl"]].sum(axis=1) == 2) & (df.iloc[:, 150] / 365 > 1)).astype(int)

# Basic dataset info
print(f"Dataset shape: {df.shape}")
print(f"Relapse rate: {df['relapse'].mean():.2%}")

# Identify columns to exclude
exclude_cols = ['id', 'relapse'] 

# Prepare feature dataframe
X_all = df.drop(exclude_cols, axis=1)
column_names = X_all.columns.tolist()
y = df['relapse']

# Process features - handle categorical and numeric data
numeric_cols = []
categorical_cols = []

# Process each column
for col in X_all.columns:
    # Check if column has non-numeric data
    if X_all[col].dtype == 'object':
        categorical_cols.append(col)
    else:
        numeric_cols.append(col)

print(f"Found {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")

# Create preprocessing pipeline
X_processed = X_all.copy()

# Handle categorical features
print("Processing categorical features...")
for col in categorical_cols:
    # Fill missing values with most frequent value
    X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0])
    
    # Encode categorical variables
    encoder = LabelEncoder()
    X_processed[col] = encoder.fit_transform(X_processed[col])

# Handle numeric features
print("Processing numeric features...")
numeric_imputer = SimpleImputer(strategy='median')
X_processed[numeric_cols] = numeric_imputer.fit_transform(X_processed[numeric_cols])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# Train a random forest model
print("\nTraining Random Forest model with all features...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1  # Use all available cores
)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nModel AUC on test data: {auc:.4f}")

# Get feature importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print top 50 feature importances
print("\nTop 50 Most Important Features:")
for i in range(min(100, len(column_names))):
    idx = indices[i]
    print(f"{i+1}. {column_names[idx]}: {importances[idx]:.4f}")

# Save feature importances to CSV
feature_importance_df = pd.DataFrame({
    'Feature': X_processed.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

feature_importance_df.to_csv('feature_importance.csv', index=False)
print("Feature importances saved to feature_importance.csv")

# Plot top 20 feature importances
plt.figure(figsize=(12, 10))
plt.title("Top 20 Feature Importances")
plt.barh(range(20), importances[indices[:20]], align="center")
plt.yticks(range(20), [column_names[i] for i in indices[:20]])
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as feature_importance.png")

# Save the model
joblib.dump(rf_model, 'rf_all_features_model.pkl')
print("Model saved as rf_all_features_model.pkl")

print("\nComplete!")