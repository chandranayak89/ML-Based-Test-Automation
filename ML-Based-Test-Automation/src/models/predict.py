"""
Module for making predictions with trained models.
This module provides functionality to load trained models and use them
to predict test failures.
"""
import os
import sys
import logging
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, List, Union, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.models.train_model import load_model, calculate_metrics, find_latest_features_file

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"model_prediction_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_best_model(metric: str = 'f1') -> str:
    """
    Find the best model based on a specific metric.
    
    Args:
        metric (str): Metric to use for selecting the best model ('f1', 'precision', 'recall', etc.).
        
    Returns:
        str: Path to the best model file.
    """
    logger.info(f"Finding best model based on {metric}")
    
    # Search in models directory
    if not os.path.exists(config.MODELS_DIR):
        logger.error(f"Models directory not found: {config.MODELS_DIR}")
        return None
    
    # Find all model metadata files
    import glob
    metadata_files = glob.glob(os.path.join(config.MODELS_DIR, '**/*_metadata.json'), recursive=True)
    
    if not metadata_files:
        logger.warning("No model metadata files found")
        return None
    
    # Load metadata and find best model
    best_score = -1
    best_model_path = None
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Extract score from evaluation metrics
            if 'evaluation' in metadata and metric in metadata['evaluation']:
                score = float(metadata['evaluation'][metric])
                
                # Update best model if this one is better
                if score > best_score:
                    best_score = score
                    model_file = metadata_file.replace('_metadata.json', '.pkl')
                    if os.path.exists(model_file):
                        best_model_path = model_file
        
        except Exception as e:
            logger.warning(f"Error processing metadata file {metadata_file}: {str(e)}")
    
    if best_model_path:
        logger.info(f"Found best model with {metric}={best_score}: {best_model_path}")
    else:
        logger.warning(f"No model found with {metric} score")
    
    return best_model_path

def load_model_metadata(model_path: str) -> Dict[str, Any]:
    """
    Load metadata for a model.
    
    Args:
        model_path (str): Path to the model file.
        
    Returns:
        dict: Model metadata.
    """
    logger.info(f"Loading metadata for model {model_path}")
    
    # Construct metadata path
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        return {}

def preprocess_test_features(test_data: pd.DataFrame, model_metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Preprocess test features to match the format expected by the model.
    
    Args:
        test_data (pd.DataFrame): Test data to preprocess.
        model_metadata (dict, optional): Model metadata containing feature names.
        
    Returns:
        pd.DataFrame: Preprocessed test features.
    """
    logger.info(f"Preprocessing test data with shape {test_data.shape}")
    
    # Create a copy to avoid modifying the original
    data = test_data.copy()
    
    # Keep only numeric columns
    numeric_data = data.select_dtypes(include=['number'])
    
    if len(numeric_data.columns) < len(data.columns):
        non_numeric_cols = set(data.columns) - set(numeric_data.columns)
        logger.warning(f"Removed {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
        data = numeric_data
    
    # If model metadata is provided, ensure features match what the model expects
    if model_metadata and 'feature_names' in model_metadata:
        expected_features = set(model_metadata['feature_names'])
        available_features = set(data.columns)
        
        # Features in the model but not in the data
        missing_features = expected_features - available_features
        for feature in missing_features:
            logger.warning(f"Missing feature: {feature}. Adding with zeros.")
            data[feature] = 0
        
        # Only keep features that the model expects
        features_to_keep = list(expected_features)
        data = data[features_to_keep]
    
    return data

def predict_test_failures(
    model: Any,
    test_data: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Predict test failures for a set of tests.
    
    Args:
        model: Trained model for prediction.
        test_data (pd.DataFrame): Test data.
        threshold (float): Threshold for probability to classify as failure.
        
    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    logger.info(f"Predicting test failures for {len(test_data)} tests with threshold {threshold}")
    
    # Make a copy of the input data
    result = test_data.copy()
    
    try:
        # Get probability predictions if available
        if hasattr(model, 'predict_proba'):
            # Predict probabilities of class 1 (failure)
            failure_probs = model.predict_proba(test_data)[:, 1]
            result['failure_probability'] = failure_probs
            result['predicted_result'] = (failure_probs >= threshold).astype(int)
        else:
            # Binary predictions only
            predictions = model.predict(test_data)
            result['predicted_result'] = predictions
        
        logger.info(f"Prediction complete. Predicted failures: {sum(result['predicted_result'])}")
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
    
    return result

def evaluate_predictions(
    predictions: pd.DataFrame,
    actual_results_col: str = 'test_passed',
    pred_results_col: str = 'predicted_result'
) -> Dict[str, float]:
    """
    Evaluate predictions against actual results.
    
    Args:
        predictions (pd.DataFrame): DataFrame with predictions and actual results.
        actual_results_col (str): Column name for actual results.
        pred_results_col (str): Column name for predicted results.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    if actual_results_col not in predictions.columns:
        logger.warning(f"Actual results column '{actual_results_col}' not found in predictions")
        return {}
    
    if pred_results_col not in predictions.columns:
        logger.warning(f"Predicted results column '{pred_results_col}' not found in predictions")
        return {}
    
    # Extract actual and predicted values
    y_true = predictions[actual_results_col]
    y_pred = predictions[pred_results_col]
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    logger.info(f"Prediction metrics: {metrics}")
    
    return metrics

def predict_on_new_data(
    model_path: Optional[str] = None,
    data_file: Optional[str] = None,
    output_file: Optional[str] = None,
    threshold: float = 0.5,
    target_col: Optional[str] = 'test_passed'
) -> pd.DataFrame:
    """
    Make predictions on new data using a trained model.
    
    Args:
        model_path (str, optional): Path to the model to use. If None, uses the best model.
        data_file (str, optional): Path to the data file. If None, finds the most recent file.
        output_file (str, optional): Path to save predictions.
        threshold (float): Threshold for probability to classify as failure.
        target_col (str, optional): Name of the target column for evaluation.
                                    If None, no evaluation is performed.
        
    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    try:
        # Find best model if not provided
        if model_path is None:
            model_path = find_best_model(metric='f1')
            if model_path is None:
                logger.error("No model found for prediction")
                return pd.DataFrame()
        
        # Load model and metadata
        model = load_model(model_path)
        metadata = load_model_metadata(model_path)
        
        # Find latest data file if not provided
        if data_file is None:
            data_file = find_latest_features_file()
            if data_file is None:
                logger.error("No data file found for prediction")
                return pd.DataFrame()
        
        # Load data
        logger.info(f"Loading data from {data_file}")
        data = pd.read_csv(data_file)
        
        # Separate features and target if present
        if target_col and target_col in data.columns:
            X = data.drop(columns=[target_col])
            y = data[target_col]
            has_target = True
        else:
            X = data
            has_target = False
        
        # Preprocess features
        X_processed = preprocess_test_features(X, metadata)
        
        # Make predictions
        predictions = predict_test_failures(model, X_processed, threshold)
        
        # Combine with original data
        result = pd.concat([data, predictions[['failure_probability', 'predicted_result']]], axis=1)
        
        # Evaluate if target is available
        if has_target:
            metrics = evaluate_predictions(result, target_col, 'predicted_result')
            logger.info(f"Prediction evaluation: {metrics}")
        
        # Save predictions if output file is provided
        if output_file:
            logger.info(f"Saving predictions to {output_file}")
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            result.to_csv(output_file, index=False)
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in prediction process: {str(e)}")
        return pd.DataFrame()

def predict_prioritized_tests(
    model_path: Optional[str] = None,
    test_metadata_file: Optional[str] = None,
    output_file: Optional[str] = None,
    n_top_tests: int = 10
) -> pd.DataFrame:
    """
    Predict and prioritize tests based on failure probability.
    
    Args:
        model_path (str, optional): Path to the model to use. If None, uses the best model.
        test_metadata_file (str, optional): Path to the test metadata file.
                                         If None, finds the most recent file.
        output_file (str, optional): Path to save prioritized tests.
        n_top_tests (int): Number of top priority tests to highlight.
        
    Returns:
        pd.DataFrame: DataFrame with prioritized tests.
    """
    # Make predictions
    predictions = predict_on_new_data(
        model_path=model_path,
        data_file=test_metadata_file,
        output_file=None
    )
    
    if predictions.empty:
        return predictions
    
    # Sort by failure probability in descending order
    if 'failure_probability' in predictions.columns:
        prioritized = predictions.sort_values('failure_probability', ascending=False)
    else:
        # If probabilities aren't available, sort by predicted result
        prioritized = predictions.sort_values('predicted_result', ascending=False)
    
    # Add rank column
    prioritized['priority_rank'] = range(1, len(prioritized) + 1)
    
    # Add priority level
    def priority_level(rank):
        if rank <= n_top_tests:
            return 'High'
        elif rank <= len(prioritized) // 3:
            return 'Medium'
        else:
            return 'Low'
    
    prioritized['priority_level'] = prioritized['priority_rank'].apply(priority_level)
    
    # Save prioritized tests if output file is provided
    if output_file:
        logger.info(f"Saving prioritized tests to {output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        prioritized.to_csv(output_file, index=False)
    
    return prioritized

def main(
    model_path: Optional[str] = None,
    data_file: Optional[str] = None,
    output_file: Optional[str] = None,
    threshold: float = 0.5,
    prioritize: bool = True,
    n_top_tests: int = 10
) -> pd.DataFrame:
    """
    Main function to make predictions with a trained model.
    
    Args:
        model_path (str, optional): Path to the model to use. If None, uses the best model.
        data_file (str, optional): Path to the data file. If None, finds the most recent file.
        output_file (str, optional): Path to save predictions.
        threshold (float): Threshold for probability to classify as failure.
        prioritize (bool): Whether to prioritize tests by failure probability.
        n_top_tests (int): Number of top priority tests to highlight.
        
    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    logger.info("Starting prediction process")
    
    if prioritize:
        result = predict_prioritized_tests(
            model_path=model_path,
            test_metadata_file=data_file,
            output_file=output_file,
            n_top_tests=n_top_tests
        )
    else:
        result = predict_on_new_data(
            model_path=model_path,
            data_file=data_file,
            output_file=output_file,
            threshold=threshold
        )
    
    logger.info("Prediction process completed")
    return result

if __name__ == "__main__":
    main() 