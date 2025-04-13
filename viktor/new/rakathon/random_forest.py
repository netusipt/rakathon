from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib

# Load data
df = pd.read_csv("/home/petrn/python/rackathon/main/viktor/processed_dataset.csv")

# Convert one-hot encoded features back to categorical
def one_hot_to_categorical(df, prefix):
    # Find all columns with this prefix
    cols = [col for col in df.columns if col.startswith(prefix + '_')]
    # Create a new column with categorical values
    df[prefix] = np.argmax(df[cols].values, axis=1)
    # Drop the original one-hot columns
    df = df.drop(columns=cols)
    return df

# Convert each one-hot encoded group to a categorical feature
df = one_hot_to_categorical(df, 'age')
df = one_hot_to_categorical(df, 'er_status')
df = one_hot_to_categorical(df, 'tumour_size')
df = one_hot_to_categorical(df, 'grading')

# Create target variable
df["relapse"] = (~(df["no_relaps"] == 1.0)).astype(int)

# Print the first few rows to verify the transformation
print("Transformed data sample:")
print(df.head())

# Set up features and target
feature_cols = ['age', 'er_status', 'tumour_size', 'grading']
X = df[feature_cols]
y = df["relapse"]

print("\nFeature distribution:")
for col in feature_cols:
    print(f"\n{col} value counts:")
    print(df[col].value_counts().sort_index())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with class weights
print("\nTraining Random Forest model...")
clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["no_relapse", "relapse"]))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Feature importance
print("\nFeature Importance:")
for feature, importance in zip(feature_cols, clf.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Optionally save the model
joblib.dump(clf, 'random_forest_model.pkl') 