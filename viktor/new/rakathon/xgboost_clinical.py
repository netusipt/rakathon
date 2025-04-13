from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib

# Load your clinical data
df = pd.read_csv("/home/petrn/python/rackathon/main/viktor/new/rakathon/onco_data.csv")

# Data preprocessing functions
def map_age(age_category):
    if pd.isna(age_category):
        return 2  # Default middle age category if missing
    category = str(age_category).strip()
    if category.startswith('10'):  # Age categories like 101, 102, 103, etc.
        return int(category[2:]) - 1  # Convert to 0-based index
    return 2  # Default if not recognized

def map_grading(grade):
    if pd.isna(grade):
        return 2  # Default to middle grade if missing
    try:
        return int(grade) - 1  # Convert to 0-based index
    except:
        return 2  # Default if not convertible

def tumour_map(t_code):
    if pd.isna(t_code):
        return 2  # Default
    
    # Extract numeric part if available
    t_str = str(t_code).strip()
    if t_str.startswith('0'):
        return 0
    elif t_str.startswith('1'):
        return 1
    elif t_str.startswith('2'):
        return 2
    elif t_str.startswith('3'):
        return 3
    elif t_str.startswith('4'):
        return 4
    return 2  # Default medium size

def stadium_map(stadium):
    if pd.isna(stadium):
        return 2
    try:
        if stadium in ["0", "1", "2", "3", "4", "X", "Y"]:
            stadium_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "X": 2, "Y": 2}
            return stadium_dict[stadium]
        else:
            return 2
    except:
        return 2

# Create new dataframe with needed features
clinical_df = pd.DataFrame()

# Extract and transform required features
clinical_df["vek"] = df["vekova_kategorie_10let_dg"].apply(map_age)
clinical_df["er_status"] = df["je_pl_hormo"].fillna(0).astype(int)  # Fill NA with 0
clinical_df["tumour_size"] = df["tnm_klasifikace_t_kod"].apply(tumour_map)
clinical_df["grading"] = df["grading"].apply(map_grading)
clinical_df["lym"] = df["tnm_klasifikace_metastazy_lym"].fillna(0).astype(int)
clinical_df["stadium"] = df["stadium"].apply(stadium_map)

# Add pl_delka parameter (treatment length in years)
clinical_df["pl_delka"] = df.iloc[:, 150].fillna(0) / 365  # Convert days to years
clinical_df["pl_mamo"] = df["pl_mamo"].fillna(0).astype(int)

modality = [75, 77, 79, 81, 83, 85, 87, 89, 91, 93]
symbols = ['O', 'R', 'T', 'C', 'H', 'I']
for symbol in symbols:
    clinical_df[symbol + '_count'] = df.iloc[:, modality].apply(lambda row: list(row).count(symbol), axis=1)

# Create target variable - relapse
clinical_df["relapse"] = (
        (df[["je_disp", "je_nl"]].sum(axis=1) == 2) & (df.iloc[:, 150] / 365 > 1)
    ).astype(int)

# Print dataset information
print("Dataset shape:", clinical_df.shape)
print(f"Relapse rate: {clinical_df['relapse'].mean():.2%}")

# Set up features and target
feature_cols = [col for col in clinical_df.columns if col != "relapse"]
X = clinical_df[feature_cols]
y = clinical_df["relapse"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining XGBoost model...")
# Calculate scale_pos_weight for imbalanced class
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

# Try different learning rates
for learning_rate in [0.05, 0.1, 0.2]:
    print(f"\n--- Learning rate: {learning_rate} ---")
    
    # Train XGBoost model
    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    xgb.fit(X_train, y_train)
    
    # Evaluate
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["no_relapse", "relapse"]))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

# Train final model with optimal learning rate
best_lr = 0.1  # Change this based on results
final_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=best_lr,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    scale_pos_weight=scale_pos,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

final_model.fit(X_train, y_train)

# Feature importance for final model
print("\nFeature Importance:")
for feature, importance in zip(feature_cols, final_model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Save the model
joblib.dump(final_model, 'xgboost_clinical_model.pkl')
print("\nModel saved as 'xgboost_clinical_model.pkl'")

# Plot ROC curve
try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    
    y_proba = final_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_proba):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('XGBoost ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('xgboost_roc_curve.png')
    print("ROC curve saved as 'xgboost_roc_curve.png'")
    
    # Feature importance plot
    plt.figure(figsize=(12, 10))
    sorted_idx = np.argsort(final_model.feature_importances_)[::-1]
    plt.barh(range(len(feature_cols)), 
             final_model.feature_importances_[sorted_idx])
    plt.yticks(range(len(feature_cols)), 
               [feature_cols[i] for i in sorted_idx])
    plt.title('XGBoost Feature Importance')
    plt.xlabel('F Score')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')
    print("Feature importance plot saved as 'xgboost_feature_importance.png'")
except:
    print("Could not generate plots (matplotlib might be missing)")

print("\nComplete!")