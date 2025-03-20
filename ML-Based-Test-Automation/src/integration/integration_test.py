"""
Integration Test Module for ML-Based Test Automation.
This module provides integration tests that verify the end-to-end workflows
of the ML-Based Test Automation framework.
"""

import os
import sys
import json
import time
import logging
import unittest
import pandas as pd
import requests
from typing import Dict, List, Any, Optional
import tempfile
import shutil
import subprocess

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from src.models.predict import predict
from src.execution.test_scheduler import TestScheduler
from src.execution.suite_optimizer import TestSuiteOptimizer
from src.execution.root_cause_analyzer import RootCauseAnalyzer
from src.execution.impact_analyzer import ImpactAnalyzer

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, "integration_test.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegrationTest(unittest.TestCase):
    """
    Integration test class for verifying end-to-end workflows of the ML-Based Test Automation framework.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test resources before running tests.
        """
        logger.info("Setting up integration test environment")
        
        # Create temporary directories for test data
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_data_dir = os.path.join(cls.temp_dir, "data")
        cls.test_models_dir = os.path.join(cls.temp_dir, "models")
        cls.test_results_dir = os.path.join(cls.temp_dir, "results")
        
        os.makedirs(cls.test_data_dir, exist_ok=True)
        os.makedirs(cls.test_models_dir, exist_ok=True)
        os.makedirs(cls.test_results_dir, exist_ok=True)
        
        # Generate test data
        cls._generate_test_data()
        
        # Start API server if needed for integration tests
        # This is commented out because it's better to mock or use a test server in most cases
        # cls._start_api_server()
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up test resources after tests are complete.
        """
        logger.info("Cleaning up integration test environment")
        
        # Remove temporary directories
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
        
        # Stop API server if it was started
        # cls._stop_api_server()
    
    @classmethod
    def _generate_test_data(cls):
        """
        Generate test data for integration tests.
        """
        logger.info("Generating test data for integration tests")
        
        # Create a sample test data DataFrame
        test_metadata = pd.DataFrame({
            'test_id': [f'test_{i}' for i in range(1, 101)],
            'test_name': [f'Test Case {i}' for i in range(1, 101)],
            'execution_time': [10 + i % 20 for i in range(1, 101)],
            'last_status': ['PASS' if i % 4 != 0 else 'FAIL' for i in range(1, 101)],
            'priority': ['P1' if i % 10 == 0 else 'P2' if i % 5 == 0 else 'P3' for i in range(1, 101)],
            'component': [f'component_{(i % 5) + 1}' for i in range(1, 101)],
            'author': [f'user_{(i % 3) + 1}' for i in range(1, 101)],
            'last_execution': [f'2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}' for i in range(1, 101)],
            'failure_count': [i % 10 for i in range(1, 101)],
            'pass_count': [50 - (i % 10) for i in range(1, 101)],
        })
        
        # Add features for ML
        test_metadata['failure_rate'] = test_metadata['failure_count'] / (test_metadata['failure_count'] + test_metadata['pass_count'])
        test_metadata['days_since_last_execution'] = 30  # Mock value
        
        # Save test metadata
        test_metadata_path = os.path.join(cls.test_data_dir, "test_metadata.csv")
        test_metadata.to_csv(test_metadata_path, index=False)
        logger.info(f"Test metadata saved to {test_metadata_path}")
        
        # Generate some test results
        test_results = []
        for i in range(1, 31):
            for test_id in test_metadata['test_id'].sample(50).tolist():
                # 80% pass, 20% fail
                status = 'PASS' if i % 5 != 0 and test_id != f'test_{i}' else 'FAIL'
                test_results.append({
                    'test_id': test_id,
                    'execution_date': f'2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}',
                    'status': status,
                    'execution_time': float(test_metadata[test_metadata['test_id'] == test_id]['execution_time'].values[0]),
                    'environment': 'TEST',
                    'version': f'1.{i}.0'
                })
        
        # Save test results
        test_results_df = pd.DataFrame(test_results)
        test_results_path = os.path.join(cls.test_data_dir, "test_results.csv")
        test_results_df.to_csv(test_results_path, index=False)
        logger.info(f"Test results saved to {test_results_path}")
    
    def test_model_training_and_evaluation(self):
        """
        Test model training and evaluation workflow.
        """
        logger.info("Testing model training and evaluation workflow")
        
        # Set paths for test
        data_path = os.path.join(self.test_data_dir, "test_metadata.csv")
        
        # Train a model
        model, model_info = train_model(
            data_path=data_path,
            model_type="random_forest",
            output_dir=self.test_models_dir,
            save_model=True
        )
        
        # Verify model was trained
        self.assertIsNotNone(model, "Model training failed")
        self.assertIn('model_path', model_info, "Model path not found in model info")
        self.assertTrue(os.path.exists(model_info['model_path']), "Model file not found")
        
        # Evaluate the model
        evaluation = evaluate_model(
            model=model,
            data_path=data_path,
            output_dir=self.test_results_dir
        )
        
        # Verify evaluation metrics
        self.assertIn('accuracy', evaluation, "Accuracy not found in evaluation results")
        self.assertIn('precision', evaluation, "Precision not found in evaluation results")
        self.assertIn('recall', evaluation, "Recall not found in evaluation results")
        self.assertIn('f1', evaluation, "F1 score not found in evaluation results")
        
        logger.info(f"Model evaluation results: {evaluation}")
    
    def test_test_scheduling_and_optimization(self):
        """
        Test the test scheduling and optimization workflow.
        """
        logger.info("Testing test scheduling and optimization workflow")
        
        # Set paths for test
        data_path = os.path.join(self.test_data_dir, "test_metadata.csv")
        
        # Initialize test scheduler
        scheduler = TestScheduler(
            output_dir=self.test_results_dir,
            model_path=None  # We don't have a model yet, will use default fallback
        )
        
        # Generate execution plan
        execution_plan = scheduler.schedule_tests_from_metadata(
            metadata_path=data_path,
            time_constraint=600  # 10 minutes constraint
        )
        
        # Verify execution plan
        self.assertIsNotNone(execution_plan, "Execution plan generation failed")
        self.assertIn('tests', execution_plan, "Tests not found in execution plan")
        self.assertIn('total_time', execution_plan, "Total time not found in execution plan")
        self.assertIn('priority_distribution', execution_plan, "Priority distribution not found in execution plan")
        
        # Initialize test suite optimizer
        optimizer = TestSuiteOptimizer(
            min_coverage_threshold=0.8,
            max_redundancy_threshold=0.3
        )
        
        # Generate optimized suite
        optimized_suite = optimizer.generate_optimized_suite(
            test_metadata_path=data_path,
            output_path=os.path.join(self.test_results_dir, "optimized_suite.json")
        )
        
        # Verify optimized suite
        self.assertIsNotNone(optimized_suite, "Optimized suite generation failed")
        self.assertIn('tests_to_keep', optimized_suite, "Tests to keep not found in optimized suite")
        self.assertIn('tests_to_remove', optimized_suite, "Tests to remove not found in optimized suite")
        self.assertIn('execution_time_reduction', optimized_suite, "Execution time reduction not found in optimized suite")
        
        logger.info(f"Optimized suite metrics: {optimized_suite}")
    
    def test_root_cause_and_impact_analysis(self):
        """
        Test the root cause and impact analysis workflow.
        """
        logger.info("Testing root cause and impact analysis workflow")
        
        # Set paths for test
        results_path = os.path.join(self.test_data_dir, "test_results.csv")
        
        # Initialize root cause analyzer
        analyzer = RootCauseAnalyzer()
        
        # Analyze test failures
        failure_analysis = analyzer.analyze_failures(
            results_path=results_path,
            output_path=os.path.join(self.test_results_dir, "failure_analysis.json")
        )
        
        # Verify failure analysis
        self.assertIsNotNone(failure_analysis, "Failure analysis failed")
        self.assertIn('common_failures', failure_analysis, "Common failures not found in analysis")
        self.assertIn('failure_patterns', failure_analysis, "Failure patterns not found in analysis")
        
        # Initialize impact analyzer
        impact_analyzer = ImpactAnalyzer()
        
        # Mock some changed files for impact analysis
        changed_files = [
            "src/models/train_model.py",
            "src/execution/test_scheduler.py"
        ]
        
        # Analyze impact
        impact_analysis = impact_analyzer.analyze_impact(
            changed_files=changed_files,
            test_metadata_path=os.path.join(self.test_data_dir, "test_metadata.csv"),
            output_path=os.path.join(self.test_results_dir, "impact_analysis.json")
        )
        
        # Verify impact analysis
        self.assertIsNotNone(impact_analysis, "Impact analysis failed")
        self.assertIn('affected_tests', impact_analysis, "Affected tests not found in analysis")
        self.assertIn('impact_score', impact_analysis, "Impact score not found in analysis")
        
        logger.info(f"Impact analysis results: {impact_analysis}")
    
    def test_end_to_end_workflow(self):
        """
        Test the complete end-to-end workflow.
        """
        logger.info("Testing end-to-end workflow")
        
        # 1. Start with test data
        data_path = os.path.join(self.test_data_dir, "test_metadata.csv")
        test_data = pd.read_csv(data_path)
        
        # 2. Train a model
        model, model_info = train_model(
            data_path=data_path,
            model_type="random_forest",
            output_dir=self.test_models_dir,
            save_model=True
        )
        
        # 3. Schedule tests based on model predictions
        scheduler = TestScheduler(
            output_dir=self.test_results_dir,
            model_path=model_info['model_path']
        )
        
        execution_plan = scheduler.schedule_tests_from_metadata(
            metadata_path=data_path,
            time_constraint=600  # 10 minutes constraint
        )
        
        # 4. Optimize the test suite
        optimizer = TestSuiteOptimizer()
        optimized_suite = optimizer.generate_optimized_suite(
            test_metadata_path=data_path,
            output_path=os.path.join(self.test_results_dir, "optimized_suite.json")
        )
        
        # 5. Run a subset of tests from the optimized suite (simulated)
        executed_tests = test_data[test_data['test_id'].isin(optimized_suite['tests_to_keep'][:10])]
        
        # 6. Analyze root causes of any failures
        # Simulate some failures
        failure_results = [
            {"test_id": executed_tests.iloc[0]['test_id'], "status": "FAIL", "error_message": "Connection timeout"},
            {"test_id": executed_tests.iloc[2]['test_id'], "status": "FAIL", "error_message": "Assertion error"}
        ]
        
        failure_df = pd.DataFrame(failure_results)
        failure_path = os.path.join(self.test_results_dir, "failures.csv")
        failure_df.to_csv(failure_path, index=False)
        
        analyzer = RootCauseAnalyzer()
        failure_analysis = analyzer.analyze_specific_failures(
            failures_path=failure_path,
            test_metadata_path=data_path,
            output_path=os.path.join(self.test_results_dir, "specific_failure_analysis.json")
        )
        
        # 7. Perform impact analysis for code changes
        impact_analyzer = ImpactAnalyzer()
        impact_analysis = impact_analyzer.analyze_impact(
            changed_files=["src/models/predict.py"],
            test_metadata_path=data_path,
            output_path=os.path.join(self.test_results_dir, "impact_analysis.json")
        )
        
        # 8. Verify all steps completed successfully
        self.assertTrue(os.path.exists(model_info['model_path']), "Model file not found")
        self.assertGreater(len(execution_plan['tests']), 0, "No tests in execution plan")
        self.assertGreater(len(optimized_suite['tests_to_keep']), 0, "No tests to keep in optimized suite")
        self.assertIn('recommendations', failure_analysis, "No recommendations in failure analysis")
        self.assertIn('affected_tests', impact_analysis, "No affected tests in impact analysis")
        
        logger.info("End-to-end workflow completed successfully")


def run_integration_tests():
    """
    Run all integration tests.
    """
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(IntegrationTest))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    results = runner.run(test_suite)
    
    return results.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1) 