import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import xgboost as xgb
from lightgbm import LGBMClassifier
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

def preprocess_processed_dataset(data_path='/home/petrn/python/rackathon/main/viktor/processed_dataset.csv'):
    """
    Preprocess the one-hot encoded processed dataset
    """
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded data with {len(df)} rows and {df.shape[1]} columns")
    
    # Create a single target variable for overall recurrence
    # A patient has a recurrence if any of the period-specific recurrences are 1
    df['relaps'] = ((df['relaps_1_5'] > 0) | (df['relaps_5_10'] > 0) | (df['relaps_10_15'] > 0)).astype(int)
    
    # Identify feature groups based on prefix
    age_cols = [col for col in df.columns if col.startswith('age_')]
    er_status_cols = [col for col in df.columns if col.startswith('er_status_')]
    tumour_size_cols = [col for col in df.columns if col.startswith('tumour_size_')]
    grading_cols = [col for col in df.columns if col.startswith('grading_')]
    
    # Target columns
    target_cols = ['relaps', 'relaps_1_5', 'relaps_5_10', 'relaps_10_15', 'no_relaps']
    
    # All feature columns
    feature_cols = age_cols + er_status_cols + tumour_size_cols + grading_cols
    
    print(f"Feature groups:")
    print(f"  Age features: {len(age_cols)}")
    print(f"  ER status features: {len(er_status_cols)}")
    print(f"  Tumour size features: {len(tumour_size_cols)}")
    print(f"  Grading features: {len(grading_cols)}")
    print(f"Total features: {len(feature_cols)}")
    
    # Split data into features and target
    X = df[feature_cols]
    y = df['relaps']
    
    # Split data into train, validation and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_cols,
        'original_data': df
    }

def train_xgboost_model_simple(preprocessed_data):
    """
    Train a simpler XGBoost model without full grid search.
    """
    print("\n--- Training XGBoost Model (Simple Version) ---")
    X_train = preprocessed_data['X_train']
    y_train = preprocessed_data['y_train']
    X_val = preprocessed_data['X_val']
    y_val = preprocessed_data['y_val']
    X_test = preprocessed_data['X_test']
    y_test = preprocessed_data['y_test']
    
    # Create model with reasonable defaults
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Evaluate model
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"AUC: {auc:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_names = preprocessed_data['feature_names']
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Feature Importance:")
    print(importance_df.head(10))
    
    # Plot feature importance for top 20 features
    plt.figure(figsize=(12, 8))
    importance_df.head(20).plot(kind='bar', x='Feature', y='Importance', figsize=(12, 8))
    plt.title('XGBoost Feature Importance (Top 20)')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance_processed.png')
    
    return {
        'model': model,
        'auc': auc,
        'accuracy': accuracy,
        'f1': f1,
        'feature_importance': importance_df
    }

def train_time_specific_models(preprocessed_data):
    """
    Train separate models for different recurrence time periods.
    """
    print("\n--- Training Time-Specific Models ---")
    original_data = preprocessed_data['original_data']
    
    # Define time periods and corresponding columns
    time_periods = {
        '1-5 years': 'relaps_1_5',
        '5-10 years': 'relaps_5_10',
        '10-15 years': 'relaps_10_15'
    }
    
    results = {}
    
    for period_name, target_col in time_periods.items():
        print(f"\nTraining model for {period_name} recurrence")
        
        # Get target for this time period
        y_train = original_data.loc[preprocessed_data['X_train'].index, target_col]
        y_val = original_data.loc[preprocessed_data['X_val'].index, target_col]
        y_test = original_data.loc[preprocessed_data['X_test'].index, target_col]
        
        # Check if we have enough positive examples
        positive_count = y_train.sum()
        if positive_count < 20:
            print(f"Not enough positive examples for {period_name} ({positive_count}), skipping")
            continue
            
        print(f"Training with {positive_count} positive examples ({positive_count/len(y_train)*100:.1f}%)")
        
        # Create LightGBM classifier
        model = LGBMClassifier(
            objective='binary',
            metric='auc',
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Train model
        model.fit(
            preprocessed_data['X_train'], y_train,
            eval_set=[(preprocessed_data['X_val'], y_val)],
            # verbose=False
        )
        
        # Make predictions
        y_pred_proba = model.predict_proba(preprocessed_data['X_test'])[:, 1]
        y_pred = model.predict(preprocessed_data['X_test'])
        
        # Evaluate model
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"AUC: {auc:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        # Store results
        results[period_name] = {
            'model': model,
            'auc': auc,
            'accuracy': accuracy,
            'f1': f1
        }
    
    return results

def evaluate_models_by_timeframe(preprocessed_data, models_dict):
    """
    Evaluate models for each recurrence time frame
    """
    print("\n--- Time-Specific Model Evaluation ---")
    original_data = preprocessed_data['original_data']
    
    # Define time periods and corresponding columns
    time_periods = {
        'Overall': 'relaps',
        '1-5 years': 'relaps_1_5',
        '5-10 years': 'relaps_5_10',
        '10-15 years': 'relaps_10_15'
    }
    
    # Dictionary to store results
    results = {}
    
    # Test indices 
    test_indices = preprocessed_data['X_test'].index
    
    # For each model
    for model_name, model_info in models_dict.items():
        model = model_info['model']
        print(f"\nEvaluating {model_name} model across time periods")
        
        # Get model predictions on test set
        X_test = preprocessed_data['X_test']
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Evaluate for each time period
        period_results = {}
        for period_name, target_col in time_periods.items():
            if target_col not in original_data.columns:
                continue
                
            # Get target for this time period
            y_test_period = original_data.loc[test_indices, target_col]
            
            # Skip if all negative (no positive cases in this time period)
            if y_test_period.sum() == 0:
                print(f"  No positive cases for {period_name} in test set, skipping")
                period_results[period_name] = {
                    'auc': None,
                    'accuracy': None,
                    'f1': None,
                    'positive_cases': 0,
                    'total_cases': len(y_test_period)
                }
                continue
            
            # Evaluate metrics
            try:
                auc = roc_auc_score(y_test_period, y_pred_proba)
                auc_str = f"{auc:.3f}"
            except:
                auc = None
                auc_str = "N/A"
                
            accuracy = accuracy_score(y_test_period, y_pred)
            f1 = f1_score(y_test_period, y_pred, zero_division=0)
            
            # Store results
            period_results[period_name] = {
                'auc': auc,
                'accuracy': accuracy,
                'f1': f1,
                'positive_cases': y_test_period.sum(),
                'total_cases': len(y_test_period)
            }
            
            print(f"  {period_name}: AUC={auc_str}, Accuracy={accuracy:.3f}, F1={f1:.3f}")
            print(f"    Positive cases: {y_test_period.sum()} / {len(y_test_period)} ({y_test_period.mean()*100:.1f}%)")
        
        # Store all period results for this model
        results[model_name] = period_results
    
    # Create a summary DataFrame
    summary_rows = []
    for model_name, period_results in results.items():
        for period_name, metrics in period_results.items():
            summary_rows.append({
                'Model': model_name,
                'Time Period': period_name,
                'AUC': metrics['auc'],
                'Accuracy': metrics['accuracy'],
                'F1': metrics['f1'],
                'Positive Rate': metrics['positive_cases'] / metrics['total_cases'] if metrics['total_cases'] > 0 else 0
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary to CSV
    summary_df.to_csv('time_specific_model_performance_processed.csv', index=False)
    print("\nTime-specific performance metrics saved to 'time_specific_model_performance_processed.csv'")
    
    # Create visualization of performance by time period
    plt.figure(figsize=(12, 8))
    
    # Filter to only rows with AUC values
    plot_df = summary_df[summary_df['AUC'].notna()]
    
    # Plot AUC by time period for each model
    for model_name in results.keys():
        model_data = plot_df[plot_df['Model'] == model_name]
        plt.plot(model_data['Time Period'], model_data['AUC'], marker='o', label=f'{model_name} (AUC)')
    
    plt.title('Model Performance by Recurrence Time Period')
    plt.xlabel('Time Period')
    plt.ylabel('AUC Score')
    plt.ylim(0.5, 1.0)  # AUC ranges from 0.5 (random) to 1.0 (perfect)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('time_specific_performance_processed.png')
    print("Time-specific performance visualization saved to 'time_specific_performance_processed.png'")
    
    return {
        'results': results,
        'summary': summary_df
    }

def train_random_forest_with_shap(preprocessed_data):
    """
    Train Random Forest model with SHAP explanations.
    """
    print("\n--- Training Random Forest with SHAP Explanations ---")
    X_train = preprocessed_data['X_train']
    y_train = preprocessed_data['y_train']
    X_test = preprocessed_data['X_test']
    y_test = preprocessed_data['y_test']
    
    # Create and train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    y_pred = rf_model.predict(X_test)
    
    # Evaluate model
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"AUC: {auc:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Calculate permutation feature importance
    perm_importance = permutation_importance(
        rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    # Create importance dataframe
    feature_names = preprocessed_data['feature_names']
    perm_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Permutation Feature Importance:")
    print(perm_imp_df.head(10))
    
    # Calculate SHAP values for explainability
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test.iloc[:100])
        
        # Plot SHAP summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[1], X_test.iloc[:100], plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('rf_shap_importance_processed.png')
        
        # Plot detailed SHAP values
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values[1], X_test.iloc[:100], show=False)
        plt.tight_layout()
        plt.savefig('rf_shap_summary_processed.png')
        print("SHAP plots saved to 'rf_shap_importance_processed.png' and 'rf_shap_summary_processed.png'")
    except Exception as e:
        print(f"Error generating SHAP plots: {e}")
    
    return {
        'model': rf_model,
        'auc': auc,
        'accuracy': accuracy,
        'f1': f1,
        'feature_importance': perm_imp_df
    }

def main():
    """
    Run all survival analysis models and compare results.
    """
    # Preprocess the data
    preprocessed_data = preprocess_processed_dataset()
    
    # Train XGBoost model
    xgb_results = train_xgboost_model_simple(preprocessed_data)
    
    # Train Random Forest with SHAP
    rf_results = train_random_forest_with_shap(preprocessed_data)
    
    # Train time-specific models
    time_models = train_time_specific_models(preprocessed_data)
    
    # Compare model performance
    print("\n--- Model Comparison ---")
    models = {
        'XGBoost': xgb_results,
        'Random Forest': rf_results
    }
    
    comparison_df = pd.DataFrame({
        'Model': list(models.keys()),
        'AUC': [models[m]['auc'] for m in models],
        'Accuracy': [models[m]['accuracy'] for m in models],
        'F1 Score': [models[m]['f1'] for m in models]
    })
    
    print(comparison_df)
    
    # Evaluate models by timeframe
    timeframe_results = evaluate_models_by_timeframe(preprocessed_data, models)
    
    # Save all models and results
    import pickle
    with open('processed_survival_models.pkl', 'wb') as f:
        pickle.dump({
            'xgboost': xgb_results,
            'random_forest': rf_results,
            'time_models': time_models,
            'timeframe_results': timeframe_results,
            'feature_names': preprocessed_data['feature_names']
        }, f)
    
    print("\nModels saved to 'processed_survival_models.pkl'")

if __name__ == "__main__":
    main()