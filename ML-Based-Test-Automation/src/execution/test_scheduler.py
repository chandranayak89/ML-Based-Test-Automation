"""
Test Scheduler Module for ML-Based Test Automation.
This module provides functionality to prioritize and schedule tests based on 
ML model predictions, execution time, and other relevant factors.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.models.predict import find_best_model, load_model, predict_test_failures

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"test_scheduler_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestPrioritizer:
    """
    Class responsible for prioritizing tests based on ML model predictions and other factors.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        prediction_threshold: float = 0.5,
        time_weight: float = 0.3,
        failure_weight: float = 0.5,
        priority_weight: float = 0.2,
        time_constraint: Optional[int] = None
    ):
        """
        Initialize the TestPrioritizer.
        
        Args:
            model_path (str, optional): Path to the ML model to use for predictions.
                                       If None, will use the best available model.
            prediction_threshold (float): Threshold for classifying a test as likely to fail.
            time_weight (float): Weight for execution time in priority calculation.
            failure_weight (float): Weight for failure probability in priority calculation.
            priority_weight (float): Weight for test priority in priority calculation.
            time_constraint (int, optional): Maximum total execution time in minutes.
                                           If provided, will optimize for this time constraint.
        """
        self.prediction_threshold = prediction_threshold
        self.time_weight = time_weight
        self.failure_weight = failure_weight
        self.priority_weight = priority_weight
        self.time_constraint = time_constraint
        
        # Load model
        if model_path is None:
            model_path = find_best_model(metric='f1')
            if model_path is None:
                raise ValueError("No suitable model found for test prioritization")
        
        self.model_path = model_path
        self.model = load_model(model_path)
        
        logger.info(f"TestPrioritizer initialized with model: {os.path.basename(model_path)}")
        logger.info(f"Weights: time={time_weight}, failure={failure_weight}, priority={priority_weight}")
        if time_constraint:
            logger.info(f"Time constraint: {time_constraint} minutes")
    
    def prioritize_tests(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prioritize tests based on ML predictions and other factors.
        
        Args:
            test_data (pd.DataFrame): DataFrame containing test metadata.
                                    Must include columns for test_id, test_name,
                                    and avg_execution_time.
        
        Returns:
            pd.DataFrame: The input DataFrame with additional columns:
                         - failure_probability: Probability of test failing
                         - priority_score: Calculated priority score
                         - priority_rank: Rank based on priority score
        """
        logger.info(f"Prioritizing {len(test_data)} tests")
        
        # Ensure required columns exist
        required_cols = ['test_id', 'test_name', 'avg_execution_time']
        missing_cols = [col for col in required_cols if col not in test_data.columns]
        if missing_cols:
            raise ValueError(f"Test data missing required columns: {missing_cols}")
        
        # Create a copy to avoid modifying the input
        result = test_data.copy()
        
        # Get failure predictions
        try:
            prediction_result = predict_test_failures(
                model=self.model,
                test_data=result,
                threshold=self.prediction_threshold
            )
            
            # Ensure we got back predictions
            if 'failure_probability' not in prediction_result.columns:
                logger.warning("No failure probabilities in prediction result")
                result['failure_probability'] = 0.0
            else:
                result['failure_probability'] = prediction_result['failure_probability']
        
        except Exception as e:
            logger.error(f"Error predicting test failures: {str(e)}")
            result['failure_probability'] = 0.0
        
        # Calculate priority score
        result = self._calculate_priority_scores(result)
        
        # Sort by priority score and add rank
        result = result.sort_values('priority_score', ascending=False)
        result['priority_rank'] = range(1, len(result) + 1)
        
        logger.info(f"Tests prioritized. Top priority test: {result.iloc[0]['test_id']}")
        return result
    
    def _calculate_priority_scores(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate priority scores for tests based on multiple factors.
        
        Args:
            test_data (pd.DataFrame): DataFrame with test data and failure probabilities.
            
        Returns:
            pd.DataFrame: DataFrame with priority_score column added.
        """
        # Normalize execution time (invert so faster tests get higher scores)
        if test_data['avg_execution_time'].max() > 0:
            max_time = test_data['avg_execution_time'].max()
            time_score = 1 - (test_data['avg_execution_time'] / max_time)
        else:
            time_score = 1.0
        
        # Use test priority if available, otherwise default to medium (0.5)
        if 'priority' in test_data.columns:
            # Convert string priorities to numeric scores
            priority_map = {
                'critical': 1.0,
                'high': 0.75,
                'medium': 0.5,
                'low': 0.25
            }
            # Handle case sensitivity and missing values
            test_data['priority_str'] = test_data['priority'].astype(str).str.lower()
            priority_score = test_data['priority_str'].map(
                lambda p: priority_map.get(p, 0.5)
            )
        else:
            priority_score = 0.5
        
        # Combine factors using weights
        test_data['priority_score'] = (
            (self.time_weight * time_score) +
            (self.failure_weight * test_data['failure_probability']) +
            (self.priority_weight * priority_score)
        )
        
        # Normalize to 0-1 range
        min_score = test_data['priority_score'].min()
        max_score = test_data['priority_score'].max()
        if max_score > min_score:
            test_data['priority_score'] = (test_data['priority_score'] - min_score) / (max_score - min_score)
            
        return test_data
    
    def optimize_for_time_constraint(self, prioritized_tests: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize the test execution schedule for a given time constraint.
        
        Args:
            prioritized_tests (pd.DataFrame): DataFrame with prioritized tests.
            
        Returns:
            pd.DataFrame: DataFrame with optimized selection of tests.
        """
        if not self.time_constraint:
            logger.info("No time constraint set, returning all tests")
            return prioritized_tests
        
        logger.info(f"Optimizing for time constraint: {self.time_constraint} minutes")
        
        # Convert time constraint to seconds for comparison
        time_constraint_seconds = self.time_constraint * 60
        
        # Ensure we have execution time data
        if 'avg_execution_time' not in prioritized_tests.columns:
            logger.warning("No execution time data, cannot optimize for time constraint")
            return prioritized_tests
        
        # Sort by priority score (highest first)
        sorted_tests = prioritized_tests.sort_values('priority_score', ascending=False)
        
        # Select tests within time constraint
        selected_tests = []
        total_time = 0
        
        for _, test in sorted_tests.iterrows():
            test_time = test['avg_execution_time']
            
            if total_time + test_time <= time_constraint_seconds:
                selected_tests.append(test)
                total_time += test_time
            else:
                # If we're under 95% of the time constraint, try to fit more tests
                if total_time < 0.95 * time_constraint_seconds:
                    # Find tests that would fit in the remaining time
                    remaining_time = time_constraint_seconds - total_time
                    candidate_tests = sorted_tests[
                        (sorted_tests['avg_execution_time'] <= remaining_time) &
                        (~sorted_tests.index.isin([t.name for t in selected_tests]))
                    ]
                    
                    if not candidate_tests.empty:
                        # Get the highest priority test that fits
                        next_test = candidate_tests.iloc[0]
                        selected_tests.append(next_test)
                        total_time += next_test['avg_execution_time']
                        
                        # Continue looking for more tests to fit
                        continue
                
                # We've reached the constraint or can't fit more tests
                break
        
        # Convert back to DataFrame
        result = pd.DataFrame(selected_tests)
        
        total_tests = len(prioritized_tests)
        selected_count = len(result)
        total_priority = result['priority_score'].sum()
        
        logger.info(f"Selected {selected_count}/{total_tests} tests within time constraint")
        logger.info(f"Total execution time: {total_time/60:.2f} minutes")
        logger.info(f"Total priority score: {total_priority:.2f}")
        
        return result
    
    def generate_execution_plan(
        self, 
        test_data: pd.DataFrame,
        optimize_for_time: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a complete test execution plan.
        
        Args:
            test_data (pd.DataFrame): DataFrame containing test metadata.
            optimize_for_time (bool): Whether to optimize for time constraint.
            
        Returns:
            dict: Execution plan containing:
                - prioritized_tests: DataFrame of prioritized tests
                - total_execution_time: Estimated total execution time
                - predicted_failures: List of tests predicted to fail
                - metadata: General information about the plan
        """
        logger.info("Generating test execution plan")
        
        # Prioritize tests
        prioritized_tests = self.prioritize_tests(test_data)
        
        # Optimize for time constraint if requested and constraint exists
        if optimize_for_time and self.time_constraint:
            selected_tests = self.optimize_for_time_constraint(prioritized_tests)
        else:
            selected_tests = prioritized_tests
        
        # Calculate total execution time
        total_time = selected_tests['avg_execution_time'].sum()
        
        # Get tests predicted to fail
        predicted_failures = selected_tests[selected_tests['failure_probability'] >= self.prediction_threshold]
        
        # Create execution plan
        execution_plan = {
            'prioritized_tests': selected_tests.to_dict(orient='records'),
            'total_execution_time': float(total_time),
            'estimated_duration_minutes': float(total_time / 60),
            'predicted_failures': predicted_failures['test_id'].tolist(),
            'predicted_failure_count': len(predicted_failures),
            'total_test_count': len(selected_tests),
            'metadata': {
                'model_used': os.path.basename(self.model_path),
                'prediction_threshold': self.prediction_threshold,
                'time_constraint_minutes': self.time_constraint,
                'generated_at': datetime.now().isoformat(),
                'weights': {
                    'time_weight': self.time_weight,
                    'failure_weight': self.failure_weight,
                    'priority_weight': self.priority_weight
                }
            }
        }
        
        logger.info(f"Execution plan generated with {len(selected_tests)} tests")
        logger.info(f"Estimated duration: {total_time/60:.2f} minutes")
        logger.info(f"Predicted failures: {len(predicted_failures)}/{len(selected_tests)}")
        
        return execution_plan

    def save_execution_plan(self, execution_plan: Dict[str, Any], output_path: str) -> str:
        """
        Save the execution plan to a file.
        
        Args:
            execution_plan (dict): Execution plan to save.
            output_path (str): Path to save the plan.
            
        Returns:
            str: Path to the saved file.
        """
        logger.info(f"Saving execution plan to {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Convert DataFrame to list of dictionaries
        if isinstance(execution_plan['prioritized_tests'], pd.DataFrame):
            execution_plan['prioritized_tests'] = execution_plan['prioritized_tests'].to_dict(orient='records')
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(execution_plan, f, indent=2)
        
        logger.info(f"Execution plan saved to {output_path}")
        return output_path

class TestScheduler:
    """
    Class responsible for scheduling and executing tests based on prioritization.
    """
    
    def __init__(
        self,
        prioritizer: Optional[TestPrioritizer] = None,
        execution_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the TestScheduler.
        
        Args:
            prioritizer (TestPrioritizer, optional): TestPrioritizer instance for prioritizing tests.
                                                   If None, a new instance will be created.
            execution_config (dict, optional): Configuration for test execution.
        """
        self.prioritizer = prioritizer or TestPrioritizer()
        self.execution_config = execution_config or {}
        
        # Default configuration
        self.default_config = {
            'max_parallel_tests': 4,
            'retry_failed_tests': True,
            'max_retries': 1,
            'retry_only_likely_flaky': True,
            'generate_reports': True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.execution_config}
        
        logger.info("TestScheduler initialized with configuration:")
        for key, value in self.config.items():
            logger.info(f"  {key}: {value}")
    
    def schedule_from_metadata(
        self, 
        metadata_file: str,
        output_dir: Optional[str] = None,
        optimize_for_time: bool = True
    ) -> Dict[str, Any]:
        """
        Schedule tests based on a metadata file.
        
        Args:
            metadata_file (str): Path to the test metadata file (CSV).
            output_dir (str, optional): Directory to save the execution plan.
            optimize_for_time (bool): Whether to optimize for time constraint.
            
        Returns:
            dict: Execution plan.
        """
        logger.info(f"Scheduling tests from metadata file: {metadata_file}")
        
        # Load test metadata
        test_data = pd.read_csv(metadata_file)
        logger.info(f"Loaded {len(test_data)} tests from metadata file")
        
        # Generate execution plan
        execution_plan = self.prioritizer.generate_execution_plan(
            test_data=test_data,
            optimize_for_time=optimize_for_time
        )
        
        # Save execution plan if output directory is provided
        if output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f"execution_plan_{timestamp}.json")
            self.prioritizer.save_execution_plan(execution_plan, output_path)
        
        return execution_plan
    
    def schedule_from_test_ids(
        self,
        test_ids: List[str],
        metadata_source: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Schedule specific tests by their IDs.
        
        Args:
            test_ids (list): List of test IDs to schedule.
            metadata_source (str): Path to the test metadata file.
            output_dir (str, optional): Directory to save the execution plan.
            
        Returns:
            dict: Execution plan.
        """
        logger.info(f"Scheduling {len(test_ids)} specific tests")
        
        # Load test metadata
        test_data = pd.read_csv(metadata_source)
        
        # Filter to only the requested tests
        filtered_data = test_data[test_data['test_id'].isin(test_ids)]
        
        if len(filtered_data) < len(test_ids):
            missing_ids = set(test_ids) - set(filtered_data['test_id'])
            logger.warning(f"Could not find metadata for {len(missing_ids)} tests: {missing_ids}")
        
        # Generate execution plan without time optimization
        execution_plan = self.prioritizer.generate_execution_plan(
            test_data=filtered_data,
            optimize_for_time=False
        )
        
        # Save execution plan if output directory is provided
        if output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f"execution_plan_custom_{timestamp}.json")
            self.prioritizer.save_execution_plan(execution_plan, output_path)
        
        return execution_plan
    
    def schedule_for_changed_files(
        self,
        changed_files: List[str],
        impact_mapping_file: str,
        metadata_source: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Schedule tests that are affected by changed files.
        
        Args:
            changed_files (list): List of files that have changed.
            impact_mapping_file (str): Path to the file mapping files to tests.
            metadata_source (str): Path to the test metadata file.
            output_dir (str, optional): Directory to save the execution plan.
            
        Returns:
            dict: Execution plan.
        """
        logger.info(f"Scheduling tests affected by {len(changed_files)} changed files")
        
        # Load impact mapping
        with open(impact_mapping_file, 'r') as f:
            impact_mapping = json.load(f)
        
        # Find affected tests
        affected_tests = set()
        for file_path in changed_files:
            # Normalize file path
            normalized_path = file_path.replace('\\', '/')
            
            # Find tests impacted by this file
            for mapping_path, tests in impact_mapping.items():
                if mapping_path.endswith(normalized_path) or normalized_path.endswith(mapping_path):
                    affected_tests.update(tests)
        
        # Convert to list
        affected_test_ids = list(affected_tests)
        
        if not affected_test_ids:
            logger.warning("No tests affected by the changed files")
            return {
                'prioritized_tests': [],
                'total_execution_time': 0,
                'estimated_duration_minutes': 0,
                'predicted_failures': [],
                'predicted_failure_count': 0,
                'total_test_count': 0,
                'metadata': {
                    'model_used': os.path.basename(self.prioritizer.model_path),
                    'prediction_threshold': self.prioritizer.prediction_threshold,
                    'time_constraint_minutes': self.prioritizer.time_constraint,
                    'generated_at': datetime.now().isoformat(),
                    'changed_files': changed_files,
                    'affected_tests': affected_test_ids
                }
            }
        
        logger.info(f"Found {len(affected_test_ids)} tests affected by changes")
        
        # Schedule the affected tests
        return self.schedule_from_test_ids(
            test_ids=affected_test_ids,
            metadata_source=metadata_source,
            output_dir=output_dir
        )
    
    def generate_incremental_schedule(
        self,
        previous_results_file: str,
        metadata_source: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an incremental schedule based on previous test results.
        Prioritizes previously failed tests and tests not run in the last execution.
        
        Args:
            previous_results_file (str): Path to the previous test results file.
            metadata_source (str): Path to the test metadata file.
            output_dir (str, optional): Directory to save the execution plan.
            
        Returns:
            dict: Execution plan.
        """
        logger.info(f"Generating incremental schedule based on previous results: {previous_results_file}")
        
        # Load previous results
        with open(previous_results_file, 'r') as f:
            previous_results = json.load(f)
        
        # Extract executed and failed tests
        executed_tests = set()
        failed_tests = set()
        
        for test_result in previous_results.get('test_results', []):
            test_id = test_result.get('test_id')
            if test_id:
                executed_tests.add(test_id)
                if test_result.get('status') == 'FAIL':
                    failed_tests.add(test_id)
        
        # Load all tests
        all_tests = pd.read_csv(metadata_source)
        all_test_ids = set(all_tests['test_id'])
        
        # Find tests not executed in the previous run
        not_executed = all_test_ids - executed_tests
        
        # Combine failed and not executed tests
        priority_tests = list(failed_tests.union(not_executed))
        
        logger.info(f"Prioritizing {len(failed_tests)} failed tests and {len(not_executed)} not executed tests")
        
        # Schedule the priority tests
        return self.schedule_from_test_ids(
            test_ids=priority_tests,
            metadata_source=metadata_source,
            output_dir=output_dir
        )
    
    def mock_test_execution(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock the execution of a test plan (for demonstration purposes).
        
        Args:
            execution_plan (dict): Execution plan to mock.
            
        Returns:
            dict: Mock execution results.
        """
        logger.info("Mocking test execution for demonstration")
        
        # Extract test information
        tests = execution_plan.get('prioritized_tests', [])
        if isinstance(tests, pd.DataFrame):
            tests = tests.to_dict(orient='records')
        
        # Mock execution results
        results = []
        for test in tests:
            # Simulate test execution
            test_id = test.get('test_id', 'unknown')
            test_name = test.get('test_name', 'unknown')
            execution_time = test.get('avg_execution_time', 1.0)
            failure_prob = test.get('failure_probability', 0.0)
            
            # Determine result based on failure probability
            import random
            is_failure = random.random() < failure_prob
            
            # Add some randomness to execution time
            actual_time = execution_time * random.uniform(0.8, 1.2)
            
            results.append({
                'test_id': test_id,
                'test_name': test_name,
                'status': 'FAIL' if is_failure else 'PASS',
                'execution_time': actual_time,
                'timestamp': datetime.now().isoformat()
            })
        
        # Create execution results
        execution_results = {
            'execution_id': f"mock_execution_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'start_time': datetime.now().isoformat(),
            'end_time': (datetime.now() + timedelta(seconds=sum(test.get('avg_execution_time', 0) for test in tests))).isoformat(),
            'total_tests': len(tests),
            'passed_tests': sum(1 for r in results if r['status'] == 'PASS'),
            'failed_tests': sum(1 for r in results if r['status'] == 'FAIL'),
            'total_execution_time': sum(r['execution_time'] for r in results),
            'test_results': results,
            'metadata': execution_plan.get('metadata', {})
        }
        
        logger.info(f"Mock execution completed: {execution_results['passed_tests']} passed, {execution_results['failed_tests']} failed")
        
        return execution_results
    
    def save_execution_results(self, execution_results: Dict[str, Any], output_path: str) -> str:
        """
        Save execution results to a file.
        
        Args:
            execution_results (dict): Execution results to save.
            output_path (str): Path to save the results.
            
        Returns:
            str: Path to the saved file.
        """
        logger.info(f"Saving execution results to {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(execution_results, f, indent=2)
        
        logger.info(f"Execution results saved to {output_path}")
        return output_path

def main():
    """
    Main function to demonstrate the test scheduler.
    """
    # Set up paths
    metadata_file = os.path.join(config.DATA_DIR, 'sample', 'test_metadata.csv')
    output_dir = os.path.join(config.REPORTS_DIR, 'execution_plans')
    
    # Create a test prioritizer with a time constraint
    prioritizer = TestPrioritizer(
        time_constraint=60,  # 60 minutes time constraint
        failure_weight=0.6,
        time_weight=0.3,
        priority_weight=0.1
    )
    
    # Create a test scheduler
    scheduler = TestScheduler(prioritizer=prioritizer)
    
    # Generate execution plan
    execution_plan = scheduler.schedule_from_metadata(
        metadata_file=metadata_file,
        output_dir=output_dir
    )
    
    # Mock test execution
    execution_results = scheduler.mock_test_execution(execution_plan)
    
    # Save execution results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_dir, f"execution_results_{timestamp}.json")
    scheduler.save_execution_results(execution_results, results_path)
    
    print(f"Test scheduling and mock execution completed. Results saved to {results_path}")

if __name__ == "__main__":
    main() 