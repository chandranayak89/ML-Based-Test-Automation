"""
Module for selecting the most important features for ML models.
This module provides functions for feature selection using various methods
including correlation analysis, feature importance, and dimensionality reduction.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"feature_selection_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def remove_low_variance_features(df, threshold=0.01):
    """
    Remove features with low variance.
    
    Args:
        df (pd.DataFrame): DataFrame containing features.
        threshold (float): Variance threshold for feature selection.
    
    Returns:
        pd.DataFrame: DataFrame with low-variance features removed.
    """
    logger.info(f"Removing low variance features with threshold {threshold}")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for low variance removal")
        return df, []
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate variance for each numerical column
    numeric_cols = [col for col in result_df.columns if pd.api.types.is_numeric_dtype(result_df[col])]
    variances = result_df[numeric_cols].var()
    
    # Identify low-variance features
    low_var_features = variances[variances < threshold].index.tolist()
    
    if low_var_features:
        logger.info(f"Removing {len(low_var_features)} low variance features")
        result_df = result_df.drop(columns=low_var_features)
    else:
        logger.info("No low variance features found")
    
    return result_df, low_var_features

def remove_highly_correlated_features(df, threshold=0.95, target_col=None):
    """
    Remove highly correlated features.
    
    Args:
        df (pd.DataFrame): DataFrame containing features.
        threshold (float): Correlation threshold for feature removal.
        target_col (str, optional): Name of the target column to preserve.
    
    Returns:
        pd.DataFrame: DataFrame with highly correlated features removed.
        list: List of removed features.
    """
    logger.info(f"Removing highly correlated features with threshold {threshold}")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for correlation analysis")
        return df, []
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Identify numerical columns
    numeric_cols = [col for col in result_df.columns if pd.api.types.is_numeric_dtype(result_df[col])]
    
    # If target_col is provided, make sure it's excluded from removal candidates
    if target_col and target_col in numeric_cols:
        features_to_check = [col for col in numeric_cols if col != target_col]
    else:
        features_to_check = numeric_cols
    
    # Calculate correlation matrix
    correlation_matrix = result_df[features_to_check].corr().abs()
    
    # Create mask for upper triangle of correlation matrix
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    # Find columns to remove (highly correlated)
    to_remove = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if to_remove:
        logger.info(f"Removing {len(to_remove)} highly correlated features: {', '.join(to_remove)}")
        result_df = result_df.drop(columns=to_remove)
    else:
        logger.info("No highly correlated features found")
    
    return result_df, to_remove

def select_features_by_importance(X, y, n_features=20, method='random_forest'):
    """
    Select top features based on their importance in prediction.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        n_features (int): Number of features to select.
        method (str): Method to use for feature importance ('random_forest' or 'mutual_info').
    
    Returns:
        list: List of selected feature names.
        dict: Dictionary of feature importance scores.
    """
    logger.info(f"Selecting top {n_features} features using {method}")
    
    if X.empty or y.empty:
        logger.warning("Empty data provided for feature importance selection")
        return [], {}
    
    # Preprocess data
    # Fill missing values in features
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Method selection
    if method == 'random_forest':
        # Use Random Forest importance
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_imputed, y)
        
        # Get feature importances
        importances = model.feature_importances_
    
    elif method == 'mutual_info':
        # Use mutual information
        selector = SelectKBest(score_func=mutual_info_classif, k='all')
        selector.fit(X_imputed, y)
        
        # Get feature importances
        importances = selector.scores_
    
    elif method == 'anova':
        # Use ANOVA F-value
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X_imputed, y)
        
        # Get feature importances
        importances = selector.scores_
    
    else:
        logger.error(f"Unknown method: {method}")
        return [], {}
    
    # Create dictionary of feature importances
    feature_importances = dict(zip(X.columns, importances))
    
    # Sort by importance
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    
    # Select top features
    top_features = list(sorted_importances.keys())[:n_features]
    
    logger.info(f"Selected {len(top_features)} top features")
    return top_features, sorted_importances

def select_features_with_rfe(X, y, n_features=20):
    """
    Select features using Recursive Feature Elimination (RFE).
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        n_features (int): Number of features to select.
    
    Returns:
        list: List of selected feature names.
    """
    logger.info(f"Selecting {n_features} features using RFE")
    
    if X.empty or y.empty:
        logger.warning("Empty data provided for RFE feature selection")
        return []
    
    # Preprocess data
    # Create a pipeline with imputation and scaling
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    X_processed = pd.DataFrame(preprocessor.fit_transform(X), columns=X.columns)
    
    # Create RFE with Random Forest
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    
    # Fit RFE
    selector.fit(X_processed, y)
    
    # Get selected features
    selected_features = X.columns[selector.support_].tolist()
    
    logger.info(f"Selected {len(selected_features)} features using RFE")
    return selected_features

def select_features_with_pca(X, n_components=0.95):
    """
    Reduce dimensionality using Principal Component Analysis (PCA).
    
    Args:
        X (pd.DataFrame): Feature matrix.
        n_components (float or int): Number of components or variance to retain.
    
    Returns:
        pd.DataFrame: DataFrame with PCA components.
        object: Fitted PCA object for transforming new data.
    """
    logger.info(f"Reducing dimensionality with PCA (n_components={n_components})")
    
    if X.empty:
        logger.warning("Empty data provided for PCA")
        return pd.DataFrame(), None
    
    # Create a pipeline with imputation and scaling
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(X_processed)
    
    # Create DataFrame with principal components
    if isinstance(n_components, float):
        # If n_components is a float (variance ratio), get the actual number of components
        n_actual_components = pca.n_components_
    else:
        n_actual_components = n_components
    
    column_names = [f'PC{i+1}' for i in range(n_actual_components)]
    principalDf = pd.DataFrame(data=principalComponents, columns=column_names)
    
    logger.info(f"PCA reduced features from {X.shape[1]} to {principalDf.shape[1]} components")
    logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    return principalDf, pca

def plot_feature_importances(importances, top_n=20, output_file=None):
    """
    Plot feature importances.
    
    Args:
        importances (dict): Dictionary of feature importance scores.
        top_n (int): Number of top features to plot.
        output_file (str, optional): Path to save the plot. If None, the plot is not saved.
    
    Returns:
        None
    """
    logger.info(f"Plotting top {top_n} feature importances")
    
    if not importances:
        logger.warning("No importances provided for plotting")
        return
    
    # Sort importances
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True)[:top_n])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_importances)), list(sorted_importances.values()), align='center')
    plt.yticks(range(len(sorted_importances)), list(sorted_importances.keys()))
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Saved feature importance plot to {output_file}")
    
    plt.close()

def select_features(df, target_col='test_passed', methods=['correlation', 'importance'], n_features=20):
    """
    Select features using a combination of methods.
    
    Args:
        df (pd.DataFrame): DataFrame containing features and target.
        target_col (str): Name of the target column.
        methods (list): List of methods to use for feature selection.
        n_features (int): Number of features to select.
    
    Returns:
        pd.DataFrame: DataFrame with selected features and target.
        dict: Dictionary of feature importance scores.
    """
    logger.info(f"Selecting features using methods: {', '.join(methods)}")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for feature selection")
        return df, {}
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Separate features and target
    if target_col in result_df.columns:
        X = result_df.drop(columns=[target_col])
        y = result_df[target_col]
    else:
        logger.warning(f"Target column '{target_col}' not found in DataFrame")
        X = result_df
        y = pd.Series()
    
    # Remove non-numeric columns
    numeric_X = X.select_dtypes(include=['number'])
    
    # Apply each method
    all_importances = {}
    selected_features = set()
    
    if 'low_variance' in methods:
        # Remove low variance features
        result_df, low_var_features = remove_low_variance_features(result_df)
    
    if 'correlation' in methods:
        # Remove highly correlated features
        result_df, correlated_features = remove_highly_correlated_features(result_df, target_col=target_col)
    
    if 'importance' in methods and not y.empty:
        # Select features by importance
        top_features, importances = select_features_by_importance(numeric_X, y, n_features)
        selected_features.update(top_features)
        all_importances.update(importances)
        
        # Plot feature importances
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = os.path.join(config.REPORTS_DIR, f"feature_importance_{timestamp}.png")
        plot_feature_importances(importances, top_n=min(n_features, len(importances)), output_file=plot_file)
    
    if 'rfe' in methods and not y.empty:
        # Select features with RFE
        rfe_features = select_features_with_rfe(numeric_X, y, n_features)
        selected_features.update(rfe_features)
    
    if 'pca' in methods:
        # Apply PCA
        # This is handled separately as it transforms the features rather than selecting them
        logger.info("PCA method is handled separately, not included in final feature selection")
    
    # Add target column back to selected features
    if selected_features:
        selected_columns = list(selected_features)
        if target_col in df.columns:
            selected_columns.append(target_col)
        
        final_df = df[selected_columns].copy()
        logger.info(f"Selected {len(selected_columns)} features")
    else:
        final_df = result_df
        logger.info("No features selected, returning original DataFrame")
    
    return final_df, all_importances

def main(input_file=None, output_file=None, target_col='test_passed', methods=None, n_features=20):
    """
    Main function to select features from data.
    
    Args:
        input_file (str, optional): Path to the input CSV file. If None, will use 
                                     the latest features file.
        output_file (str, optional): Path to save the output CSV file. If None, will 
                                     generate a default name.
        target_col (str): Name of the target column.
        methods (list, optional): List of methods to use for feature selection.
        n_features (int): Number of features to select.
    
    Returns:
        str: Path to the output file.
    """
    try:
        logger.info("Starting feature selection")
        
        if methods is None:
            methods = ['low_variance', 'correlation', 'importance']
        
        # Find input file if not provided
        if input_file is None:
            # Look in interim directory for features files
            feature_files = [f for f in os.listdir(config.INTERIM_DATA_DIR) 
                            if f.endswith('.csv') and f.startswith('features_')]
            
            if feature_files:
                # Sort by file name (which contains timestamp)
                feature_files.sort(reverse=True)
                input_file = os.path.join(config.INTERIM_DATA_DIR, feature_files[0])
            else:
                # If no feature files, look for processed data
                processed_files = [f for f in os.listdir(config.PROCESSED_DATA_DIR) 
                                 if f.endswith('.csv') and f.startswith('test_data_')]
                
                if processed_files:
                    # Sort by file name (which contains timestamp)
                    processed_files.sort(reverse=True)
                    input_file = os.path.join(config.PROCESSED_DATA_DIR, processed_files[0])
                else:
                    logger.error("No input files found")
                    return None
        
        logger.info(f"Loading data from: {input_file}")
        df = pd.read_csv(input_file)
        
        if df.empty:
            logger.warning("Input file contains no data")
            return None
        
        # Select features
        selected_df, importances = select_features(df, target_col, methods, n_features)
        
        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(config.INTERIM_DATA_DIR, f"selected_features_{timestamp}.csv")
        
        selected_df.to_csv(output_file, index=False)
        logger.info(f"Saved selected features to: {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.exception(f"Error during feature selection: {str(e)}")
        return None

if __name__ == "__main__":
    main() 