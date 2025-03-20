"""
Module containing baseline machine learning models for predicting test failures.
These models serve as a foundation for the test failure prediction system.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, List, Union, Optional

# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

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

class ModelFactory:
    """
    Factory class to create different types of machine learning models for test failure prediction.
    """
    
    @staticmethod
    def get_model(model_type: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Get a model pipeline based on the specified type and parameters.
        
        Args:
            model_type (str): Type of model to create (e.g., 'random_forest', 'logistic_regression').
            params (dict, optional): Dictionary of model-specific parameters to override defaults.
                                    
        Returns:
            tuple: (sklearn Pipeline, param_grid for hyperparameter tuning)
        """
        if params is None:
            params = {}
        
        # Common steps for all models
        steps = [
            ('scaler', StandardScaler())
        ]
        
        # Get the model and its parameter grid
        if model_type == 'dummy':
            model, param_grid = ModelFactory._get_dummy_classifier(params)
        elif model_type == 'logistic_regression':
            model, param_grid = ModelFactory._get_logistic_regression(params)
        elif model_type == 'random_forest':
            model, param_grid = ModelFactory._get_random_forest(params)
        elif model_type == 'gradient_boosting':
            model, param_grid = ModelFactory._get_gradient_boosting(params)
        elif model_type == 'svm':
            model, param_grid = ModelFactory._get_svm(params)
        elif model_type == 'decision_tree':
            model, param_grid = ModelFactory._get_decision_tree(params)
        elif model_type == 'naive_bayes':
            model, param_grid = ModelFactory._get_naive_bayes(params)
        elif model_type == 'knn':
            model, param_grid = ModelFactory._get_knn(params)
        elif model_type == 'neural_network':
            model, param_grid = ModelFactory._get_neural_network(params)
        else:
            logger.warning(f"Unknown model type: {model_type}. Using Random Forest.")
            model, param_grid = ModelFactory._get_random_forest(params)
        
        # Add model to pipeline
        steps.append(('model', model))
        
        return Pipeline(steps), param_grid
    
    @staticmethod
    def _get_dummy_classifier(params: Dict[str, Any]) -> Tuple[DummyClassifier, Dict[str, List[Any]]]:
        """
        Create a dummy classifier as a baseline.
        
        Args:
            params (dict): Model parameters.
            
        Returns:
            tuple: (DummyClassifier, parameter grid)
        """
        defaults = {
            'strategy': 'most_frequent'
        }
        defaults.update(params)
        
        model = DummyClassifier(
            strategy=defaults['strategy'],
            random_state=42
        )
        
        param_grid = {
            'model__strategy': ['most_frequent', 'stratified', 'prior', 'uniform']
        }
        
        return model, param_grid
    
    @staticmethod
    def _get_logistic_regression(params: Dict[str, Any]) -> Tuple[LogisticRegression, Dict[str, List[Any]]]:
        """
        Create a logistic regression model.
        
        Args:
            params (dict): Model parameters.
            
        Returns:
            tuple: (LogisticRegression, parameter grid)
        """
        defaults = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'liblinear'
        }
        defaults.update(params)
        
        model = LogisticRegression(
            C=defaults['C'],
            penalty=defaults['penalty'],
            solver=defaults['solver'],
            random_state=42,
            max_iter=1000
        )
        
        param_grid = {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear']
        }
        
        return model, param_grid
    
    @staticmethod
    def _get_random_forest(params: Dict[str, Any]) -> Tuple[RandomForestClassifier, Dict[str, List[Any]]]:
        """
        Create a random forest classifier.
        
        Args:
            params (dict): Model parameters.
            
        Returns:
            tuple: (RandomForestClassifier, parameter grid)
        """
        defaults = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
        defaults.update(params)
        
        model = RandomForestClassifier(
            n_estimators=defaults['n_estimators'],
            max_depth=defaults['max_depth'],
            min_samples_split=defaults['min_samples_split'],
            min_samples_leaf=defaults['min_samples_leaf'],
            random_state=42,
            n_jobs=-1
        )
        
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
        
        return model, param_grid
    
    @staticmethod
    def _get_gradient_boosting(params: Dict[str, Any]) -> Tuple[GradientBoostingClassifier, Dict[str, List[Any]]]:
        """
        Create a gradient boosting classifier.
        
        Args:
            params (dict): Model parameters.
            
        Returns:
            tuple: (GradientBoostingClassifier, parameter grid)
        """
        defaults = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2
        }
        defaults.update(params)
        
        model = GradientBoostingClassifier(
            n_estimators=defaults['n_estimators'],
            learning_rate=defaults['learning_rate'],
            max_depth=defaults['max_depth'],
            min_samples_split=defaults['min_samples_split'],
            random_state=42
        )
        
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5, 10]
        }
        
        return model, param_grid
    
    @staticmethod
    def _get_svm(params: Dict[str, Any]) -> Tuple[SVC, Dict[str, List[Any]]]:
        """
        Create a support vector machine classifier.
        
        Args:
            params (dict): Model parameters.
            
        Returns:
            tuple: (SVC, parameter grid)
        """
        defaults = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True
        }
        defaults.update(params)
        
        model = SVC(
            C=defaults['C'],
            kernel=defaults['kernel'],
            gamma=defaults['gamma'],
            probability=defaults['probability'],
            random_state=42
        )
        
        param_grid = {
            'model__C': [0.1, 1, 10, 100],
            'model__kernel': ['linear', 'rbf', 'poly'],
            'model__gamma': ['scale', 'auto', 0.01, 0.1, 1]
        }
        
        return model, param_grid
    
    @staticmethod
    def _get_decision_tree(params: Dict[str, Any]) -> Tuple[DecisionTreeClassifier, Dict[str, List[Any]]]:
        """
        Create a decision tree classifier.
        
        Args:
            params (dict): Model parameters.
            
        Returns:
            tuple: (DecisionTreeClassifier, parameter grid)
        """
        defaults = {
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'criterion': 'gini'
        }
        defaults.update(params)
        
        model = DecisionTreeClassifier(
            max_depth=defaults['max_depth'],
            min_samples_split=defaults['min_samples_split'],
            min_samples_leaf=defaults['min_samples_leaf'],
            criterion=defaults['criterion'],
            random_state=42
        )
        
        param_grid = {
            'model__max_depth': [None, 5, 10, 15, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__criterion': ['gini', 'entropy']
        }
        
        return model, param_grid
    
    @staticmethod
    def _get_naive_bayes(params: Dict[str, Any]) -> Tuple[GaussianNB, Dict[str, List[Any]]]:
        """
        Create a Gaussian Naive Bayes classifier.
        
        Args:
            params (dict): Model parameters.
            
        Returns:
            tuple: (GaussianNB, parameter grid)
        """
        defaults = {
            'var_smoothing': 1e-9
        }
        defaults.update(params)
        
        model = GaussianNB(
            var_smoothing=defaults['var_smoothing']
        )
        
        param_grid = {
            'model__var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
        }
        
        return model, param_grid
    
    @staticmethod
    def _get_knn(params: Dict[str, Any]) -> Tuple[KNeighborsClassifier, Dict[str, List[Any]]]:
        """
        Create a k-nearest neighbors classifier.
        
        Args:
            params (dict): Model parameters.
            
        Returns:
            tuple: (KNeighborsClassifier, parameter grid)
        """
        defaults = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto'
        }
        defaults.update(params)
        
        model = KNeighborsClassifier(
            n_neighbors=defaults['n_neighbors'],
            weights=defaults['weights'],
            algorithm=defaults['algorithm'],
            n_jobs=-1
        )
        
        param_grid = {
            'model__n_neighbors': [3, 5, 7, 9, 11],
            'model__weights': ['uniform', 'distance'],
            'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        
        return model, param_grid
    
    @staticmethod
    def _get_neural_network(params: Dict[str, Any]) -> Tuple[MLPClassifier, Dict[str, List[Any]]]:
        """
        Create a multi-layer perceptron (neural network) classifier.
        
        Args:
            params (dict): Model parameters.
            
        Returns:
            tuple: (MLPClassifier, parameter grid)
        """
        defaults = {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'alpha': 0.0001,
            'learning_rate': 'constant',
            'max_iter': 500
        }
        defaults.update(params)
        
        model = MLPClassifier(
            hidden_layer_sizes=defaults['hidden_layer_sizes'],
            activation=defaults['activation'],
            alpha=defaults['alpha'],
            learning_rate=defaults['learning_rate'],
            max_iter=defaults['max_iter'],
            random_state=42
        )
        
        param_grid = {
            'model__hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
            'model__activation': ['relu', 'tanh', 'logistic'],
            'model__alpha': [0.0001, 0.001, 0.01],
            'model__learning_rate': ['constant', 'adaptive']
        }
        
        return model, param_grid

def get_all_model_types() -> List[str]:
    """
    Get a list of all available model types.
    
    Returns:
        list: List of model type strings.
    """
    return [
        'dummy',
        'logistic_regression',
        'random_forest', 
        'gradient_boosting',
        'svm',
        'decision_tree',
        'naive_bayes',
        'knn',
        'neural_network'
    ]
    
def get_recommended_models_for_test_data() -> List[str]:
    """
    Get a list of recommended models for test data.
    
    Returns:
        list: List of recommended model type strings.
    """
    return [
        'random_forest',
        'gradient_boosting', 
        'logistic_regression',
        'decision_tree',
        'neural_network'
    ] 