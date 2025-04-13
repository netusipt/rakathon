import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BreastCancerRecurrencePredictor:
    """
    A predictor for breast cancer recurrence using Cox Proportional Hazards model.
    """
    
    def __init__(self):
        self.model = CoxPHFitter()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = None
        
    def preprocess_data(self, df, duration_col, event_col, feature_cols=None):
        """
        Preprocess the data for Cox model.
        
        Parameters:
        -----------
        df : pandas DataFrame
            The data frame containing patient data
        duration_col : str
            Column name for time to event (recurrence or last follow-up)
        event_col : str
            Column name for event indicator (1 for recurrence, 0 for censored)
        feature_cols : list of str, optional
            Feature columns to use for prediction. If None, will use all numeric features.
            
        Returns:
        --------
        pandas DataFrame
            Preprocessed data ready for modeling
        """
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # If feature columns not specified, use all numeric columns except duration and event
        if feature_cols is None:
            # Get all numeric columns
            numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
            
            # Remove duration and event columns if they're in the list
            if duration_col in numeric_cols:
                numeric_cols.remove(duration_col)
            if event_col in numeric_cols:
                numeric_cols.remove(event_col)
                
            feature_cols = numeric_cols
        
        self.feature_columns = feature_cols
        
        # Handle missing values - avoid inplace with chained assignment
        for col in feature_cols + [duration_col, event_col]:
            if col in processed_df.columns and processed_df[col].isnull().any():
                if processed_df[col].dtype.kind in 'iuf':  # for numeric
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                else:  # for categorical
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
        
        # Scale features
        processed_df[feature_cols] = self.scaler.fit_transform(processed_df[feature_cols])
        
        return processed_df
    
    def fit(self, df, duration_col, event_col, feature_cols=None, **kwargs):
        """
        Fit the Cox Proportional Hazards model.
        
        Parameters:
        -----------
        df : pandas DataFrame
            The data frame containing patient data
        duration_col : str
            Column name for time to event (recurrence or last follow-up)
        event_col : str
            Column name for event indicator (1 for recurrence, 0 for censored)
        feature_cols : list of str, optional
            Feature columns to use for prediction
        **kwargs : dict
            Additional parameters to pass to CoxPHFitter's fit method
            
        Returns:
        --------
        self
        """
        # Preprocess data
        processed_df = self.preprocess_data(df, duration_col, event_col, feature_cols)
        
        # Select relevant columns for modeling
        model_df = processed_df[[duration_col, event_col] + self.feature_columns]
        
        # Fit the model
        self.model.fit(model_df, duration_col=duration_col, event_col=event_col, **kwargs)
        self.is_fitted = True
        
        return self
    
    def evaluate(self, df, duration_col, event_col):
        """
        Evaluate the model using concordance index on validation data.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Validation data frame
        duration_col : str
            Column name for time to event
        event_col : str
            Column name for event indicator
            
        Returns:
        --------
        float
            Concordance index (C-index) score
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Preprocess validation data
        processed_df = df.copy()
        
        # Handle missing values in feature columns
        for col in self.feature_columns:
            if processed_df[col].isnull().any():
                if processed_df[col].dtype.kind in 'iuf':  # for numeric
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                else:  # for categorical
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
        
        # Handle missing values in duration and event columns
        if processed_df[duration_col].isnull().any():
            processed_df[duration_col] = processed_df[duration_col].fillna(processed_df[duration_col].median())
        if processed_df[event_col].isnull().any():
            processed_df[event_col] = processed_df[event_col].fillna(0)  # Assume no event for missing values
        
        # Scale features using the already fitted scaler
        processed_df[self.feature_columns] = self.scaler.transform(processed_df[self.feature_columns])
        
        # Make predictions
        predictions = self.model.predict_partial_hazard(processed_df[self.feature_columns])
        
        # Ensure no NaN values in predictions or inputs to concordance index
        valid_mask = ~(predictions.isna() | processed_df[duration_col].isna() | processed_df[event_col].isna())
        
        if valid_mask.sum() == 0:
            print("Warning: No valid samples for evaluation after removing NaNs")
            return float('nan')
        
        # Calculate concordance index using only valid rows
        c_index = concordance_index(
            processed_df[duration_col][valid_mask], 
            -predictions[valid_mask], 
            processed_df[event_col][valid_mask]
        )
        
        return c_index
    
    def predict_risk(self, X):
        """
        Predict the hazard ratio for new samples.
        
        Parameters:
        -----------
        X : pandas DataFrame
            New samples to predict
            
        Returns:
        --------
        pandas Series
            Predicted hazard ratios
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Make sure X has all required features
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")
        
        # Preprocess features
        X_processed = X[self.feature_columns].copy()
        
        # Apply categorical mappings if available
        if hasattr(self, 'categorical_mappings'):
            for col, mapping in self.categorical_mappings.items():
                if col in X_processed.columns:
                    # For values not seen during training, assign median of known values
                    X_processed[col] = X_processed[col].map(mapping)
                    unknown_mask = X_processed[col].isna() & X[col].notna()
                    if unknown_mask.any():
                        median_value = np.median(list(mapping.values()))
                        X_processed.loc[unknown_mask, col] = median_value
        
        # Handle missing values
        for col in self.feature_columns:
            if X_processed[col].isnull().any():
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        
        # Scale numeric features
        numeric_features = [col for col in self.feature_columns if pd.api.types.is_numeric_dtype(X_processed[col])]
        if numeric_features:
            X_processed[numeric_features] = self.scaler.transform(X_processed[numeric_features])
        
        # Predict hazard ratio
        hazard_ratios = self.model.predict_partial_hazard(X_processed)
        
        return hazard_ratios
    
    def plot_survival_curves(self, X, times=None, label=None, ax=None):
        """
        Plot survival curves for new samples.
        
        Parameters:
        -----------
        X : pandas DataFrame
            New samples to plot survival curves for
        times : array-like, optional
            Times to evaluate the survival function at
        label : str, optional
            Label for the survival curve
        ax : matplotlib Axes, optional
            Axes to plot on
            
        Returns:
        --------
        matplotlib Axes
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate hazard ratios
        hazard_ratios = self.predict_risk(X)
        
        # Plot the survival function
        if times is None:
            times = np.linspace(0, 10 * 365, 100)  # 10 years in days
        
        surv_func = self.model.predict_survival_function(X, times=times)
        
        # If X has multiple rows, average the survival functions
        if len(X) > 1:
            avg_surv = surv_func.mean(axis=1)
            if label:
                ax.plot(times, avg_surv, label=label)
            else:
                ax.plot(times, avg_surv, label="Average Survival")
        else:
            if label:
                ax.plot(times, surv_func.iloc[:, 0], label=label)
            else:
                ax.plot(times, surv_func.iloc[:, 0], label="Survival Function")
        
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival Probability")
        ax.set_title("Breast Cancer Recurrence-Free Survival")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    def feature_importance(self):
        """
        Get feature importance from the fitted model.
        
        Returns:
        --------
        pandas DataFrame
            Feature importance summary
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
            
        return self.model.summary 