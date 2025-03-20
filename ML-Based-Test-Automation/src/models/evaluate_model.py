"""
Module for evaluating and analyzing machine learning model performance for test failure prediction.
This module provides functions to evaluate model accuracy, analyze feature importance,
and generate visualizations for model interpretability.
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.inspection import permutation_importance

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.models.train_model import load_model, find_latest_features_file, load_data, prepare_train_test_data
from src.models.predict import load_model_metadata, predict_test_failures

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"model_evaluation_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def evaluate_classifier(
    model_path: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a classifier model and generate performance metrics and visualizations.
    
    Args:
        model_path (str): Path to the model file.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        output_dir (str, optional): Directory to save evaluation results and plots.
        
    Returns:
        dict: Dictionary of evaluation metrics and file paths.
    """
    logger.info(f"Evaluating classifier model from {model_path}")
    
    # Load model
    model = load_model(model_path)
    model_metadata = load_model_metadata(model_path)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
    
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics['classification_report'] = report
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Create plots if output directory is provided
    if output_dir:
        # Confusion matrix plot
        plot_confusion_matrix(
            cm, 
            os.path.join(output_dir, 'confusion_matrix.png'),
            ['Negative (Pass)', 'Positive (Fail)']
        )
        
        # ROC curve if probabilities are available
        if y_pred_proba is not None:
            plot_roc_curve(
                y_test, 
                y_pred_proba, 
                os.path.join(output_dir, 'roc_curve.png')
            )
            
            # Precision-recall curve
            plot_precision_recall_curve(
                y_test, 
                y_pred_proba, 
                os.path.join(output_dir, 'precision_recall_curve.png')
            )
        
        # Save metrics to JSON
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            # Convert numpy values to Python types for JSON serialization
            clean_metrics = {}
            for key, value in metrics.items():
                if key not in ['confusion_matrix', 'classification_report']:
                    clean_metrics[key] = float(value)
                else:
                    clean_metrics[key] = value
            
            json.dump(clean_metrics, f, indent=2)
    
    logger.info(f"Model evaluation metrics: {metrics}")
    return metrics

def analyze_feature_importance(
    model_path: str,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Optional[str] = None,
    n_top_features: int = 20,
    use_permutation: bool = True
) -> Dict[str, float]:
    """
    Analyze feature importance for a model.
    
    Args:
        model_path (str): Path to the model file.
        X (pd.DataFrame): Features data.
        y (pd.Series): Target data.
        output_dir (str, optional): Directory to save plots.
        n_top_features (int): Number of top features to display.
        use_permutation (bool): Whether to use permutation importance.
        
    Returns:
        dict: Dictionary of feature importance scores.
    """
    logger.info(f"Analyzing feature importance for model {model_path}")
    
    # Load model
    model = load_model(model_path)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    feature_names = X.columns
    importances = {}
    
    # Extract built-in feature importances if available
    if hasattr(model[-1], 'feature_importances_'):
        logger.info("Using built-in feature importances")
        raw_importances = model[-1].feature_importances_
        importances = dict(zip(feature_names, raw_importances))
    
    # Use permutation importance if requested or built-in not available
    if use_permutation or not importances:
        logger.info("Computing permutation importance")
        try:
            # Compute permutation importance
            perm_importance = permutation_importance(
                model, X, y, n_repeats=10, random_state=42, n_jobs=-1
            )
            
            # Average importance across repeats
            mean_importances = perm_importance.importances_mean
            importances = dict(zip(feature_names, mean_importances))
            
        except Exception as e:
            logger.error(f"Error computing permutation importance: {str(e)}")
    
    # Sort by importance
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    
    # Plot feature importances
    if output_dir and sorted_importances:
        # Limit to top N features
        top_importances = dict(list(sorted_importances.items())[:n_top_features])
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_importances)), list(top_importances.values()), align='center')
        plt.yticks(range(len(top_importances)), list(top_importances.keys()))
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save plot
        importance_plot_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(importance_plot_path)
        plt.close()
        
        # Save importance values
        importance_path = os.path.join(output_dir, 'feature_importance.json')
        with open(importance_path, 'w') as f:
            # Convert numpy values to Python types for JSON serialization
            clean_importances = {k: float(v) for k, v in sorted_importances.items()}
            json.dump(clean_importances, f, indent=2)
    
    return sorted_importances

def plot_learning_curve(
    model_path: str,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: str,
    cv: int = 5,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10)
) -> None:
    """
    Generate and plot a learning curve for a model.
    
    Args:
        model_path (str): Path to the model file.
        X (pd.DataFrame): Features data.
        y (pd.Series): Target data.
        output_path (str): Path to save the plot.
        cv (int): Number of cross-validation folds.
        train_sizes (np.ndarray): Array of training set sizes to evaluate.
    """
    logger.info(f"Generating learning curve for model {model_path}")
    
    # Load model
    model = load_model(model_path)
    
    try:
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=X,
            y=y,
            train_sizes=train_sizes,
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training F1')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        
        plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5, label='Validation F1')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('F1 Score')
        plt.title('Learning Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        # Save plot
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating learning curve: {str(e)}")

def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str,
    class_names: List[str]
) -> None:
    """
    Plot a confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix.
        output_path (str): Path to save the plot.
        class_names (list): List of class names.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Save plot
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    output_path: str
) -> None:
    """
    Plot a ROC curve.
    
    Args:
        y_true (pd.Series): True labels.
        y_pred_proba (np.ndarray): Predicted probabilities.
        output_path (str): Path to save the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Save plot
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_precision_recall_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    output_path: str
) -> None:
    """
    Plot a precision-recall curve.
    
    Args:
        y_true (pd.Series): True labels.
        y_pred_proba (np.ndarray): Predicted probabilities.
        output_path (str): Path to save the plot.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    # Save plot
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def analyze_model_performance(
    model_path: str,
    test_data_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    target_col: str = 'test_passed'
) -> Dict[str, Any]:
    """
    Analyze model performance on test data.
    
    Args:
        model_path (str): Path to the model file.
        test_data_file (str, optional): Path to test data file.
        output_dir (str, optional): Directory to save analysis results.
        target_col (str): Name of the target column.
        
    Returns:
        dict: Dictionary with analysis results.
    """
    logger.info(f"Analyzing model performance for {model_path}")
    
    # Set output directory
    if output_dir is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_dir = os.path.join(config.REPORTS_DIR, f"model_analysis_{model_name}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find test data if not provided
    if test_data_file is None:
        test_data_file = find_latest_features_file()
        if test_data_file is None:
            logger.error("No test data file found")
            return {}
    
    # Load data
    data = load_data(test_data_file)
    
    # Prepare train/test data
    X_train, X_test, y_train, y_test = prepare_train_test_data(
        data, target_col=target_col, test_size=0.3
    )
    
    # Evaluate model
    metrics = evaluate_classifier(
        model_path=model_path,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir
    )
    
    # Analyze feature importance
    importances = analyze_feature_importance(
        model_path=model_path,
        X=X_test,
        y=y_test,
        output_dir=output_dir
    )
    
    # Generate learning curve
    plot_learning_curve(
        model_path=model_path,
        X=data.drop(columns=[target_col]),
        y=data[target_col],
        output_path=os.path.join(output_dir, 'learning_curve.png')
    )
    
    # Save summary to JSON
    summary = {
        'model_path': model_path,
        'test_data_file': test_data_file,
        'metrics': metrics,
        'top_features': list(importances.keys())[:10],
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Model analysis complete. Results saved to {output_dir}")
    return summary

def compare_models(
    model_paths: List[str],
    test_data_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    target_col: str = 'test_passed'
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models on the same test data.
    
    Args:
        model_paths (list): List of paths to model files.
        test_data_file (str, optional): Path to test data file.
        output_dir (str, optional): Directory to save comparison results.
        target_col (str): Name of the target column.
        
    Returns:
        dict: Dictionary with comparison results for each model.
    """
    logger.info(f"Comparing {len(model_paths)} models")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(config.REPORTS_DIR, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find test data if not provided
    if test_data_file is None:
        test_data_file = find_latest_features_file()
        if test_data_file is None:
            logger.error("No test data file found")
            return {}
    
    # Load data
    data = load_data(test_data_file)
    
    # Prepare train/test data
    X_train, X_test, y_train, y_test = prepare_train_test_data(
        data, target_col=target_col, test_size=0.3
    )
    
    # Evaluate each model
    results = {}
    metrics_summary = {}
    
    for model_path in model_paths:
        try:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            logger.info(f"Evaluating model: {model_name}")
            
            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Evaluate model
            metrics = evaluate_classifier(
                model_path=model_path,
                X_test=X_test,
                y_test=y_test,
                output_dir=model_output_dir
            )
            
            # Analyze feature importance
            importances = analyze_feature_importance(
                model_path=model_path,
                X=X_test,
                y=y_test,
                output_dir=model_output_dir
            )
            
            # Store results
            results[model_name] = {
                'metrics': metrics,
                'importances': dict(list(importances.items())[:10])  # Top 10 features
            }
            
            # Add to metrics summary
            metrics_summary[model_name] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            }
            
            if 'roc_auc' in metrics:
                metrics_summary[model_name]['roc_auc'] = metrics['roc_auc']
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_path}: {str(e)}")
    
    # Create comparison visualizations
    if metrics_summary:
        # Bar chart comparing metrics
        plot_model_comparison(
            metrics_summary,
            os.path.join(output_dir, 'model_comparison.png')
        )
    
    # Save summary to JSON
    summary = {
        'models': model_paths,
        'test_data_file': test_data_file,
        'metrics_summary': metrics_summary,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(output_dir, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Model comparison complete. Results saved to {output_dir}")
    return results

def plot_model_comparison(
    metrics_summary: Dict[str, Dict[str, float]],
    output_path: str
) -> None:
    """
    Create a bar chart comparing model metrics.
    
    Args:
        metrics_summary (dict): Dictionary of model metrics.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 8))
    
    # Get metrics and models
    models = list(metrics_summary.keys())
    metrics = list(next(iter(metrics_summary.values())).keys())
    
    # Set up bar positions
    bar_width = 0.15
    index = np.arange(len(models))
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        values = [metrics_summary[model][metric] for model in models]
        plt.bar(
            index + i * bar_width, 
            values, 
            width=bar_width,
            label=metric.replace('_', ' ').title()
        )
    
    # Set up plot
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(index + bar_width * (len(metrics) - 1) / 2, models, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def find_all_models() -> List[str]:
    """
    Find all model files in the models directory.
    
    Returns:
        list: List of paths to model files.
    """
    if not os.path.exists(config.MODELS_DIR):
        logger.error(f"Models directory not found: {config.MODELS_DIR}")
        return []
    
    # Find all model files
    import glob
    model_files = glob.glob(os.path.join(config.MODELS_DIR, '**/*.pkl'), recursive=True)
    
    return model_files

def main(
    model_path: Optional[str] = None,
    test_data_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    compare_all: bool = False
) -> Dict[str, Any]:
    """
    Main function to evaluate models.
    
    Args:
        model_path (str, optional): Path to model file.
        test_data_file (str, optional): Path to test data file.
        output_dir (str, optional): Directory to save evaluation results.
        compare_all (bool): Whether to compare all available models.
        
    Returns:
        dict: Dictionary with evaluation results.
    """
    try:
        logger.info("Starting model evaluation")
        
        if compare_all:
            # Find all models
            model_paths = find_all_models()
            if not model_paths:
                logger.error("No models found")
                return {}
            
            # Compare models
            results = compare_models(
                model_paths=model_paths,
                test_data_file=test_data_file,
                output_dir=output_dir
            )
            
        else:
            # Find model if not provided
            if model_path is None:
                from src.models.predict import find_best_model
                model_path = find_best_model(metric='f1')
                if model_path is None:
                    logger.error("No model found for evaluation")
                    return {}
            
            # Analyze model
            results = analyze_model_performance(
                model_path=model_path,
                test_data_file=test_data_file,
                output_dir=output_dir
            )
        
        logger.info("Model evaluation complete")
        return results
        
    except Exception as e:
        logger.exception(f"Error in model evaluation: {str(e)}")
        return {}

if __name__ == "__main__":
    main() 