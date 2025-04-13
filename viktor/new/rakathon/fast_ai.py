import pandas as pd
import numpy as np
import torch
from fastai.tabular.all import *
from sklearn.metrics import classification_report

# === Load and prepare dataset ===
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

# Create binary label: relapse (1) vs. no relapse (0)
df["relapse"] = (~(df["no_relaps"] == 1.0)).astype(int)

# Print class distribution
print(f"Class distribution: {df['relapse'].value_counts().sort_index()}")
print(f"0: No relapse, 1: Relapse")

# Print categorical feature distribution
feature_cols = ['age', 'er_status', 'tumour_size', 'grading']
print("\nFeature distribution:")
for col in feature_cols:
    print(f"\n{col} value counts:")
    print(df[col].value_counts().sort_index())

# Drop original target columns
df = df.drop(columns=["relaps_1_5", "relaps_5_10", "relaps_10_15", "no_relaps"])

# Set up features and dependent variable
cat_names = feature_cols  # All features are now categorical 
cont_names = []  # No continuous features
dep_var = "relapse"

# Calculate class ratios for balancing
total = len(df)
neg_count = (df[dep_var] == 0).sum()
pos_count = (df[dep_var] == 1).sum()
print(f"\nPositive examples: {pos_count}, Negative examples: {neg_count}, Ratio: 1:{neg_count/pos_count:.1f}")

# Create a balanced version of the dataset
neg_indices = df[df[dep_var] == 0].index
pos_indices = df[df[dep_var] == 1].index

# Undersample the majority class (no relapse)
undersampling_ratio = 2  # Keep twice as many negative examples as positive
n_neg_samples = min(len(neg_indices), int(pos_count * undersampling_ratio))
sampled_neg_indices = np.random.choice(neg_indices, n_neg_samples, replace=False)

# Combine with all positive examples
balanced_indices = np.concatenate([sampled_neg_indices, pos_indices])
df_balanced = df.loc[balanced_indices].copy().reset_index(drop=True)

print(f"\nBalanced dataset class distribution:")
print(f"{df_balanced[dep_var].value_counts().sort_index()}")

# Create FastAI TabularDataLoaders with the balanced dataset
splits = RandomSplitter(valid_pct=0.2)(range_of(df_balanced))

# Set up the TabularPandas with categorical variables
procs = [Categorify]  # Use Categorify for categorical variables
to = TabularPandas(df_balanced, procs=procs, cat_names=cat_names, cont_names=cont_names, 
                  y_names=dep_var, splits=splits)

# Create standard dataloaders (now with balanced classes)
dls = to.dataloaders(bs=128)

# === Create and train the model ===
# Create the model with tabular config
config = tabular_config(ps=[0.2, 0.1])  # Use dropout for regularization
learn = tabular_learner(dls, layers=[64, 32], metrics=[accuracy, F1Score(), Precision(), Recall()], 
                        config=config)

# Train for 10 epochs with a fixed learning rate
print("\nTraining model...")
learn.fit(10, lr=1e-3)

# === Evaluate the model ===
# Get predictions on validation set
valid_preds, valid_targets = learn.get_preds()
valid_pred_class = valid_preds.argmax(dim=1)

# Print classification report
print("\nValidation Set Results:")
print(classification_report(
    valid_targets, valid_pred_class,
    target_names=["no_relapse", "relapse"]
))

# Try with adjusted threshold for improved recall
valid_probs = valid_preds[:, 1]  # Probability of relapse
threshold = 0.3  # Lower threshold to increase recall for minority class
valid_pred_adjusted = (valid_probs > threshold).int()

print("\nValidation Set Results with adjusted threshold (0.3):")
print(classification_report(
    valid_targets, valid_pred_adjusted,
    target_names=["no_relapse", "relapse"]
))

# Save the model
learn.save('cancer_relapse_model')

# Prepare model to be used on the full dataset (for real predictions)
print("\nCreating full dataset model...")
# Create dataloaders for the complete dataset
splits_full = RandomSplitter(valid_pct=0.1)(range_of(df))
to_full = TabularPandas(df, procs=procs, cat_names=cat_names, cont_names=cont_names, 
                        y_names=dep_var, splits=splits_full)
dls_full = to_full.dataloaders(bs=256)

# Create a new model with the same parameters
learn_full = tabular_learner(dls_full, layers=[64, 32], config=config)

# Apply trained weights from the balanced model
learn_full.model.load_state_dict(learn.model.state_dict())

# Save full model for future inference
learn_full.save('cancer_relapse_full_model')

print("\nComplete! Models saved as 'cancer_relapse_model' and 'cancer_relapse_full_model'")

# To make predictions on new data:
# test_df = pd.read_csv("path_to_test_data.csv")
# test_dl = learn.dls.test_dl(test_df)
# preds, _ = learn.get_preds(dl=test_dl)
# pred_class = preds.argmax(dim=1)
