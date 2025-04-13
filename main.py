import pandas as pd
import numpy as np
from cox_model import BreastCancerRecurrencePredictor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def analyze_breast_cancer_recurrence():
    # Load data
    df = pd.read_csv('data.csv')
    total_patients = len(df)

    
    
    # Additional relapse analysis based on time periods
    # Create specific relapse indicator based on je_disp and je_nl columns
    if 'je_disp' in df.columns and 'je_nl' in df.columns:
        df["relaps"] = (df[["je_disp", "je_nl"]].sum(axis=1) == 2).astype(int)
        print(f"Relapses percentage: {df['relaps'].sum() / total_patients * 100}%")
        
        # Assuming column at index 150 contains days since diagnosis or treatment
        if df.shape[1] > 150:
            # Create time-specific relapse indicators
            df["relaps_0_2"] = ((df["relaps"] == 1) & (df.iloc[:, 150] / 365 <= 2)).astype(int)
            df["relaps_2_5"] = ((df["relaps"] == 1) & (df.iloc[:, 150] / 365 > 2) & (df.iloc[:, 150] / 365 <= 5)).astype(int)
            df["relaps_5_10"] = ((df["relaps"] == 1) & (df.iloc[:, 150] / 365 > 5) & (df.iloc[:, 150] / 365 <= 10)).astype(int)
            df["relaps_10_15"] = ((df["relaps"] == 1) & (df.iloc[:, 150] / 365 > 10) & (df.iloc[:, 150] / 365 <= 15)).astype(int)
            
            # Print time-based relapse statistics
            print(f"Relapses within 0-2 years: {df['relaps_0_2'].sum()}")
            print(f"Relapses within 2-5 years: {df['relaps_2_5'].sum()}")
            print(f"Relapses within 5-10 years: {df['relaps_5_10'].sum()}")
            print(f"Relapses within 10-15 years: {df['relaps_10_15'].sum()}")
        else:
            print("Warning: Column used for time calculations not found at index 150")
    else:
        print("Warning: Required columns 'je_disp' and 'je_nl' not found for detailed relapse analysis")
    
    # Calculate percentage of patients with relapse

    
    relapse_percentage = ((df['relaps'].sum() / total_patients) * 100)
    print(f"Relapse percentage: {relapse_percentage}%")
    
    # Save results to a new CSV file
    # relapse_patients.to_csv('patients_with_relapse.csv', index=False)
    
    # After creating relaps columns and before returning
    if 'relaps' in df.columns:
        # Train Cox model if we have relapse data
        cox_model = train_cox_model(df)
    
    return df["relaps"]

def train_cox_model(df):
    """
    Train a Cox proportional hazards model on the breast cancer data.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Breast cancer dataset with recurrence information
        
    Returns:
    --------
    BreastCancerRecurrencePredictor
        Fitted model
    """
    print("Training Cox proportional hazards model...")
    
    # Verify we have the necessary columns
    if 'relaps' not in df.columns:
        print("Error: 'relaps' column not found in dataset.")
        return None
    
    # Find the time column (column at index 150)
    if df.shape[1] > 150:
        time_col = df.columns[150]
    else:
        print("Error: Time column not available (expected at index 150).")
        return None
    
    # Use selected features (manually specified)
    selected_features = ['vekova_kategorie_10let_dg', 'je_pl_hormo', 'grading']
    print(f"Using {len(selected_features)} features: {', '.join(selected_features)}")
    
    # Optionally add TNM-related features if you still want them
    if 'tnm_klasifikace_t_kod' in df.columns:
        # Check if it has mostly numeric values
        numeric_values = pd.to_numeric(df['tnm_klasifikace_t_kod'], errors='coerce')
        if numeric_values.notna().mean() > 0.5:  # More than 50% can be converted to numeric
            selected_features.append('tnm_klasifikace_t_kod')
            print("Added tnm_klasifikace_t_kod as feature")
    
    # Evaluate model with proper data splitting
    results = evaluate_cox_model_performance(df, time_col, selected_features)
    
    # Save predictions to CSV
    save_risk_predictions(results['model'], df, time_col, selected_features)
    
    return results['model']

def evaluate_cox_model_performance(df, time_col, selected_features):
    """
    Split data into training/validation/test sets and evaluate model performance
    
    Parameters:
    -----------
    df : pandas DataFrame
        The complete dataset with recurrence information
    time_col : str
        Name of the column containing time to event data
    selected_features : list
        Features to use in the model
        
    Returns:
    --------
    dict
        Performance metrics
    """
    print("\n--- Model Evaluation with Data Splitting ---")
    
    # First, drop rows with missing values in critical columns
    critical_cols = selected_features + [time_col, 'relaps']
    valid_df = df.dropna(subset=critical_cols).copy()
    
    print(f"Using {len(valid_df)} valid samples out of {len(df)} total")
    print(f"Recurrence rate in filtered dataset: {valid_df['relaps'].mean() * 100:.1f}%")
    
    # Save the selected features and outcome to CSV
    export_columns = selected_features + [time_col, 'relaps']
    if 'relaps_0_2' in valid_df.columns:
        export_columns += ['relaps_0_2', 'relaps_2_5', 'relaps_5_10', 'relaps_10_15']
    
    export_df = valid_df[export_columns].copy()
    export_filename = 'selected_features_data.csv'
    export_df.to_csv(export_filename, index=False)
    print(f"Saved selected features data to {export_filename}")
    
    # Split into train and temp sets (70% / 30%)
    train_df, temp_df = train_test_split(valid_df, test_size=0.3, random_state=42, stratify=valid_df['relaps'])
    
    # Split temp into validation and test sets (15% / 15% of total)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['relaps'])
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Training distribution
    train_pos = train_df['relaps'].sum()
    train_neg = len(train_df) - train_pos
    print(f"Training set: {train_pos} recurrences ({train_pos/len(train_df)*100:.1f}%), {train_neg} non-recurrences")
    
    # Create and train model
    model = BreastCancerRecurrencePredictor()
    
    # Train on training set
    model.fit(
        train_df, 
        duration_col=time_col, 
        event_col='relaps',
        feature_cols=selected_features,
        show_progress=True
    )
    
    # Validate on validation set
    val_c_index = model.evaluate(val_df, duration_col=time_col, event_col='relaps')
    print(f"Validation C-index: {val_c_index:.3f}")
    
    # Final evaluation on test set
    test_c_index = model.evaluate(test_df, duration_col=time_col, event_col='relaps')
    print(f"Test C-index: {test_c_index:.3f}")
    
    # Get feature importance
    importance = model.feature_importance()
    print("\nFeature Importance:")
    print(importance[['coef', 'exp(coef)', 'p']].sort_values('p').head(len(selected_features)))
    
    # Plot survival curves for different risk groups
    create_stratified_survival_curves(model, test_df, time_col, selected_features)
    
    return {
        'validation_c_index': val_c_index,
        'test_c_index': test_c_index,
        'feature_importance': importance,
        'model': model
    }

def create_stratified_survival_curves(model, test_df, time_col, features):
    """
    Create survival curves for different risk groups
    """
    # Calculate risk scores for test set
    risk_scores = model.predict_risk(test_df)
    
    # Create risk groups (low, medium, high)
    test_df_with_risk = test_df.copy()
    test_df_with_risk['risk_score'] = risk_scores
    
    # Define risk quantiles
    low_risk = test_df_with_risk['risk_score'].quantile(0.33)
    high_risk = test_df_with_risk['risk_score'].quantile(0.67)
    
    # Create risk groups
    low_risk_group = test_df_with_risk[test_df_with_risk['risk_score'] <= low_risk]
    medium_risk_group = test_df_with_risk[(test_df_with_risk['risk_score'] > low_risk) & 
                                         (test_df_with_risk['risk_score'] <= high_risk)]
    high_risk_group = test_df_with_risk[test_df_with_risk['risk_score'] > high_risk]
    
    # Plot survival curves
    plt.figure(figsize=(12, 8))
    
    # Plot each risk group
    if len(low_risk_group) > 0:
        model.plot_survival_curves(low_risk_group, label=f"Low Risk (n={len(low_risk_group)})")
    if len(medium_risk_group) > 0:
        model.plot_survival_curves(medium_risk_group, label=f"Medium Risk (n={len(medium_risk_group)})")
    if len(high_risk_group) > 0:
        model.plot_survival_curves(high_risk_group, label=f"High Risk (n={len(high_risk_group)})")
    
    plt.title("Survival Curves by Risk Group")
    plt.xlabel("Time (days)")
    plt.ylabel("Recurrence-free Probability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('risk_stratified_survival.png')
    plt.close()
    print("Risk-stratified survival curves saved to 'risk_stratified_survival.png'")

def save_risk_predictions(model, df, time_col, selected_features):
    """
    Apply the trained model to the dataset and save the risk predictions to CSV.
    
    Parameters:
    -----------
    model : BreastCancerRecurrencePredictor
        Trained Cox model
    df : pandas DataFrame
        Dataset with features
    time_col : str
        Name of the time column
    selected_features : list
        Features used for prediction
    """
    # Filter to only rows with complete data for prediction
    valid_mask = df[selected_features + [time_col, 'relaps']].notna().all(axis=1)
    prediction_df = df.loc[valid_mask].copy()
    
    # Make risk predictions
    risk_scores = model.predict_risk(prediction_df)
    
    # Add predictions to dataframe
    prediction_df['predicted_risk_score'] = risk_scores
    
    # Add risk categories (low, medium, high)
    low_risk = risk_scores.quantile(0.33)
    high_risk = risk_scores.quantile(0.67)
    
    # Create risk category
    prediction_df['risk_category'] = 'medium'
    prediction_df.loc[risk_scores <= low_risk, 'risk_category'] = 'low'
    prediction_df.loc[risk_scores > high_risk, 'risk_category'] = 'high'
    
    # Select columns to export
    export_cols = selected_features + [time_col, 'relaps', 'predicted_risk_score', 'risk_category']
    if 'relaps_0_2' in prediction_df.columns:
        export_cols += ['relaps_0_2', 'relaps_2_5', 'relaps_5_10', 'relaps_10_15']
    
    # Export to CSV
    output_file = 'breast_cancer_risk_predictions.csv'
    prediction_df[export_cols].to_csv(output_file, index=False)
    print(f"Saved risk predictions to {output_file}")
    
    return prediction_df

if __name__ == "__main__":
    relapse_patients = analyze_breast_cancer_recurrence()
    # print("Analysis completed. Results saved to file 'patients_with_relapse.csv'")
