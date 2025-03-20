"""
Module for training machine learning models to predict test failures.
This module handles loading data, preprocessing, model training, hyperparameter tuning,
and model persistence.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Tuple, List, Union, Optional

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.models.baseline_models import ModelFactory, get_all_model_types, get_recommended_models_for_test_data

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"model_training_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    logger.info(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Loaded data with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_train_test_data(
    data: pd.DataFrame, 
    target_col: str = 'test_passed',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for training and testing.
    
    Args:
        data (pd.DataFrame): Input data.
        target_col (str): Name of the target column.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Preparing train/test data with test_size={test_size}")
    
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Remove non-numeric columns (they won't work with scikit-learn)
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        logger.warning(f"Removing {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
        X = X.select_dtypes(include=['number'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    logger.info(f"Target distribution - Train: {y_train.mean():.2f}, Test: {y_test.mean():.2f}")
    
    return X_train, X_test, y_train, y_test

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    params: Optional[Dict[str, Any]] = None,
    cv: int = 5,
    n_jobs: int = -1,
    use_grid_search: bool = True
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    """
    Train a model with optional hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        model_type (str): Type of model to train.
        params (dict, optional): Model parameters to override defaults.
        cv (int): Number of cross-validation folds.
        n_jobs (int): Number of parallel jobs (-1 for all available cores).
        use_grid_search (bool): Whether to use grid search for hyperparameter tuning.
        
    Returns:
        tuple: (trained model pipeline, best parameters, training metrics)
    """
    logger.info(f"Training {model_type} model")
    
    # Get model and parameter grid
    pipeline, param_grid = ModelFactory.get_model(model_type, params)
    
    if use_grid_search:
        logger.info(f"Using GridSearchCV with {cv} folds")
        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_obj,
            scoring='f1',
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Get cross-validation results
        cv_results = {
            'best_score': grid_search.best_score_,
            'best_params': best_params,
            'cv_results': {
                'mean_train_score': grid_search.cv_results_['mean_train_score'].tolist(),
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_train_score': grid_search.cv_results_['std_train_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist()
            }
        }
        
        logger.info(f"Best cv f1 score: {grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {best_params}")
    
    else:
        logger.info("Training single model without grid search")
        model = pipeline
        model.fit(X_train, y_train)
        best_params = params or {}
        cv_results = {}
    
    # Get feature importances if available
    feature_importances = {}
    try:
        # For models that have a feature_importances_ attribute (e.g., tree-based models)
        if hasattr(model[-1], 'feature_importances_'):
            importances = model[-1].feature_importances_
            feature_importances = dict(zip(X_train.columns, importances))
            feature_importances = dict(sorted(
                feature_importances.items(), 
                key=lambda item: item[1], 
                reverse=True
            ))
    except Exception as e:
        logger.warning(f"Could not extract feature importances: {str(e)}")
    
    # Calculate training metrics
    y_train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    
    # Combine metrics and CV results
    training_results = {
        'train_metrics': train_metrics,
        'cv_results': cv_results,
        'feature_importances': feature_importances
    }
    
    return model, best_params, training_results

def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model (Pipeline): Trained model pipeline.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        output_dir (str, optional): Directory to save evaluation plots.
        
    Returns:
        dict: Evaluation metrics.
    """
    logger.info("Evaluating model on test data")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # If the model can predict probabilities, get them
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = None
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    logger.info(f"Test metrics: {metrics}")
    
    # Generate plots if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
        plt.yticks(tick_marks, ['Negative', 'Positive'])
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # ROC curve
        if y_pred_proba is not None:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["roc_auc"]:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
            plt.close()
            
            # Precision-Recall curve
            plt.figure(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
            plt.close()
    
    return metrics

def calculate_metrics(
    y_true: pd.Series, 
    y_pred: np.ndarray, 
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true (pd.Series): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_pred_proba (np.ndarray, optional): Predicted probabilities for the positive class.
        
    Returns:
        dict: Dictionary of metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Add AUC if probabilities are available
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics

def save_model(
    model: Pipeline,
    model_type: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: str = None,
    model_name: str = None
) -> str:
    """
    Save a trained model and its metadata.
    
    Args:
        model (Pipeline): Trained model pipeline.
        model_type (str): Type of the model.
        params (dict): Model parameters.
        metrics (dict): Model evaluation metrics.
        output_dir (str, optional): Directory to save the model. 
                                    Defaults to config.MODELS_DIR.
        model_name (str, optional): Name of the model file. 
                                   Defaults to f"{model_type}_{timestamp}.pkl".
        
    Returns:
        str: Path to the saved model.
    """
    if output_dir is None:
        output_dir = config.MODELS_DIR
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate model name if not provided
    if model_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{model_type}_{timestamp}"
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'metrics': metrics,
        'feature_names': list(metrics.get('feature_importances', {}).keys())
    }
    
    metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metadata saved to {metadata_path}")
    
    return model_path

def load_model(model_path: str) -> Pipeline:
    """
    Load a saved model.
    
    Args:
        model_path (str): Path to the saved model.
        
    Returns:
        Pipeline: Loaded model pipeline.
    """
    logger.info(f"Loading model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def train_and_evaluate_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type: str = 'random_forest',
    params: Optional[Dict[str, Any]] = None,
    save: bool = True,
    output_dir: Optional[str] = None
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train and evaluate a model, optionally saving it.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.
        model_type (str): Type of model to train.
        params (dict, optional): Model parameters to override defaults.
        save (bool): Whether to save the model.
        output_dir (str, optional): Directory to save the model and evaluation results.
        
    Returns:
        tuple: (trained model, combined training and evaluation results)
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(config.MODELS_DIR, f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    model, best_params, training_results = train_model(
        X_train, y_train, model_type=model_type, params=params
    )
    
    # Evaluate model
    evaluation_metrics = evaluate_model(
        model, X_test, y_test, output_dir=output_dir
    )
    
    # Combine results
    results = {
        'training': training_results,
        'evaluation': evaluation_metrics,
        'model_type': model_type,
        'parameters': best_params
    }
    
    # Save model if requested
    if save:
        model_path = save_model(
            model=model,
            model_type=model_type,
            params=best_params,
            metrics=results,
            output_dir=output_dir
        )
        results['model_path'] = model_path
    
    return model, results

def train_multiple_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_types: Optional[List[str]] = None,
    save: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate multiple models.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.
        model_types (list, optional): List of model types to train. 
                                    If None, use recommended models.
        save (bool): Whether to save the models.
        output_dir (str, optional): Directory to save the models and evaluation results.
        
    Returns:
        dict: Dictionary of model results keyed by model type.
    """
    # Set default model types if not provided
    if model_types is None:
        model_types = get_recommended_models_for_test_data()
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(config.MODELS_DIR, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train each model
    results = {}
    for model_type in model_types:
        logger.info(f"Training model: {model_type}")
        model_output_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_output_dir, exist_ok=True)
        
        try:
            _, model_results = train_and_evaluate_model(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_type=model_type,
                save=save,
                output_dir=model_output_dir
            )
            results[model_type] = model_results
        except Exception as e:
            logger.error(f"Error training {model_type} model: {str(e)}")
            results[model_type] = {'error': str(e)}
    
    # Save comparison results
    comparison = {model_type: results[model_type]['evaluation'] for model_type in results}
    comparison_path = os.path.join(output_dir, "model_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Create comparison plot
    plot_model_comparison(comparison, os.path.join(output_dir, "model_comparison.png"))
    
    return results

def plot_model_comparison(comparison: Dict[str, Dict[str, float]], output_path: str) -> None:
    """
    Create a bar plot comparing model performances.
    
    Args:
        comparison (dict): Dictionary of model evaluation metrics keyed by model type.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 8))
    
    # Metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    if 'roc_auc' in list(comparison.values())[0]:
        metrics.append('roc_auc')
    
    # Extract values
    model_types = list(comparison.keys())
    values = {metric: [comparison[model].get(metric, 0) for model in model_types] for metric in metrics}
    
    # Set up bar positions
    bar_width = 0.15
    positions = np.arange(len(model_types))
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        plt.bar(
            positions + i * bar_width, 
            values[metric], 
            width=bar_width,
            label=metric.replace('_', ' ').title()
        )
    
    # Set up plot
    plt.xlabel('Model Type')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(positions + bar_width * (len(metrics) - 1) / 2, model_types)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Model comparison plot saved to {output_path}")

def find_latest_features_file() -> str:
    """
    Find the most recent features file to use for model training.
    
    Returns:
        str: Path to the latest features file.
    """
    # Directory priorities for finding feature files
    search_dirs = [
        config.INTERIM_DATA_DIR, 
        config.PROCESSED_DATA_DIR,
        os.path.join(config.DATA_DIR, 'sample')
    ]
    
    # Patterns to search for, in order of preference
    file_patterns = [
        "engineered_features_*.csv",
        "selected_features_*.csv",
        "features_*.csv",
        "model_ready_data_*.csv",
        "test_data_*.csv",
        "test_metadata.csv"
    ]
    
    for directory in search_dirs:
        if not os.path.exists(directory):
            continue
        
        for pattern in file_patterns:
            import glob
            files = glob.glob(os.path.join(directory, pattern))
            if files:
                # Sort by modification time, newest first
                files.sort(key=os.path.getmtime, reverse=True)
                return files[0]
    
    return None

def main(
    data_file: Optional[str] = None,
    target_col: str = 'test_passed',
    model_types: Optional[List[str]] = None,
    test_size: float = 0.2
) -> Dict[str, Dict[str, Any]]:
    """
    Main function to train and evaluate models.
    
    Args:
        data_file (str, optional): Path to the data file.
        target_col (str): Name of the target column.
        model_types (list, optional): List of model types to train.
        test_size (float): Proportion of data to use for testing.
        
    Returns:
        dict: Dictionary of model results.
    """
    try:
        logger.info("Starting model training process")
        
        # Find data file if not provided
        if data_file is None:
            data_file = find_latest_features_file()
            if data_file is None:
                logger.error("No data file found for model training")
                return {}
        
        # Load data
        data = load_data(data_file)
        
        # Prepare train/test data
        X_train, X_test, y_train, y_test = prepare_train_test_data(
            data, target_col=target_col, test_size=test_size
        )
        
        # Train and evaluate models
        results = train_multiple_models(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_types=model_types,
            save=True
        )
        
        logger.info("Model training process completed successfully")
        return results
        
    except Exception as e:
        logger.exception(f"Error in model training process: {str(e)}")
        return {}

if __name__ == "__main__":
    main() 