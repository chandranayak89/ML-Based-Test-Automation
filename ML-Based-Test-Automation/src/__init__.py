"""
ML-Based Test Automation Framework
=================================

A comprehensive framework for ML-based test automation, prediction, and optimization.
"""

from .version import __version__, __author__, __email__, __license__

# Import main components for convenience
# Data handling
from src.data.collect_data import collect_data
from src.data.preprocess_data import preprocess_data

# Feature engineering
from src.features.build_features import build_features

# Model components
from src.models.train_model import train_model, load_model
from src.models.evaluate_model import evaluate_model
from src.models.predict import predict_single_test, predict_batch, find_best_model

# Execution
from src.execution.test_scheduler import TestScheduler
from src.execution.suite_optimizer import TestSuiteOptimizer
from src.execution.root_cause_analyzer import RootCauseAnalyzer
from src.execution.impact_analyzer import ImpactAnalyzer

# Reporting
from src.reporting.dashboard import TestReportingDashboard

# Integration
from src.integration.cicd_integration import CICDPipeline

__all__ = [
    # Version info
    '__version__', '__author__', '__email__', '__license__',
    
    # Data
    'collect_data', 'preprocess_data',
    
    # Features
    'build_features',
    
    # Models
    'train_model', 'load_model', 'evaluate_model', 
    'predict_single_test', 'predict_batch', 'find_best_model',
    
    # Execution
    'TestScheduler', 'TestSuiteOptimizer', 'RootCauseAnalyzer', 'ImpactAnalyzer',
    
    # Reporting
    'TestReportingDashboard',
    
    # Integration
    'CICDPipeline',
]
