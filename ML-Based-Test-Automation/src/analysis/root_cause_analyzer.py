"""
Root Cause Analysis Module for ML-Based Test Automation.
This module helps identify common failure patterns in test runs and provides
insights to help diagnose and address test failures more efficiently.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from collections import defaultdict, Counter

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"root_cause_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RootCauseAnalyzer:
    """
    Class for analyzing test failures to identify common patterns and root causes.
    """
    
    def __init__(
        self,
        error_pattern_similarity_threshold: float = 0.7,
        max_patterns_per_category: int = 10,
        max_suggestions_per_failure: int = 3,
        time_window_days: int = 30
    ):
        """
        Initialize the RootCauseAnalyzer.
        
        Args:
            error_pattern_similarity_threshold (float): Threshold for considering error patterns similar
            max_patterns_per_category (int): Maximum number of patterns to track per category
            max_suggestions_per_failure (int): Maximum number of suggestions to provide per failure
            time_window_days (int): Time window in days for historical analysis
        """
        self.error_pattern_similarity_threshold = error_pattern_similarity_threshold
        self.max_patterns_per_category = max_patterns_per_category
        self.max_suggestions_per_failure = max_suggestions_per_failure
        self.time_window_days = time_window_days
        
        # Initialize pattern database
        self.error_patterns = defaultdict(list)
        
        logger.info(f"RootCauseAnalyzer initialized with similarity threshold: {error_pattern_similarity_threshold}")
    
    def analyze_test_runs(
        self,
        test_results: List[Dict[str, Any]],
        test_logs: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze test runs to identify failure patterns.
        
        Args:
            test_results (list): List of test result dictionaries
            test_logs (dict, optional): Dictionary mapping test_id to log content
            
        Returns:
            dict: Analysis results including failure patterns and statistics
        """
        logger.info(f"Analyzing {len(test_results)} test results")
        
        # Extract failed tests
        failed_tests = [r for r in test_results if r.get('status') == 'FAIL']
        logger.info(f"Found {len(failed_tests)} failed tests")
        
        # Create analysis result structure
        analysis_result = {
            'total_tests': len(test_results),
            'failed_tests': len(failed_tests),
            'failure_rate': len(failed_tests) / len(test_results) if test_results else 0,
            'failure_patterns': [],
            'component_failures': defaultdict(int),
            'test_specific_failures': defaultdict(list),
            'common_root_causes': [],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Analyze component-level failures
        for result in test_results:
            if result.get('status') == 'FAIL':
                component = result.get('component', 'unknown')
                analysis_result['component_failures'][component] += 1
                
                test_id = result.get('test_id', 'unknown')
                analysis_result['test_specific_failures'][test_id].append(result)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        analysis_result['component_failures'] = dict(analysis_result['component_failures'])
        analysis_result['test_specific_failures'] = dict(analysis_result['test_specific_failures'])
        
        # Analyze error patterns from logs if available
        if test_logs:
            logger.info("Analyzing test logs for error patterns")
            
            failure_patterns = self._extract_error_patterns(failed_tests, test_logs)
            analysis_result['failure_patterns'] = failure_patterns
            
            # Identify common root causes
            root_causes = self._identify_root_causes(failure_patterns)
            analysis_result['common_root_causes'] = root_causes
        
        # Calculate additional statistics
        if failed_tests:
            # Calculate average failure duration if available
            if all('execution_time' in result for result in failed_tests):
                avg_duration = sum(result['execution_time'] for result in failed_tests) / len(failed_tests)
                analysis_result['avg_failure_duration'] = avg_duration
            
            # Most common failing components
            common_components = sorted(
                analysis_result['component_failures'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            analysis_result['most_failing_components'] = common_components[:5]
        
        logger.info("Test run analysis completed")
        return analysis_result
    
    def _extract_error_patterns(
        self,
        failed_tests: List[Dict[str, Any]],
        test_logs: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Extract error patterns from test logs.
        
        Args:
            failed_tests (list): List of failed test results
            test_logs (dict): Dictionary mapping test_id to log content
            
        Returns:
            list: List of identified error patterns
        """
        # Initialize pattern extraction
        error_patterns = []
        processed_logs = set()
        
        # Common regex patterns for errors
        error_regexes = [
            # Java/JVM stack traces
            r'Exception in thread "[^"]*" ([A-Za-z0-9_.]+Exception)[^\n]*\n([ \t]+at [^\n]+\n)+',
            # Python tracebacks
            r'Traceback \(most recent call last\):\n([ \t]+File "[^"]+", line \d+, in [^\n]+\n)+([A-Za-z0-9_.]+Error|[A-Za-z0-9_.]+Exception): (.+?)(?=\n\n|\n[^\s]|$)',
            # Generic error messages
            r'ERROR[:\s]+(.+?)(?=\n\n|\n[^\s]|$)',
            r'FAIL[:\s]+(.+?)(?=\n\n|\n[^\s]|$)',
            # Assertion errors
            r'AssertionError[:\s]+(.+?)(?=\n\n|\n[^\s]|$)',
            # General errors
            r'Error[:\s]+(.+?)(?=\n\n|\n[^\s]|$)'
        ]
        
        for test in failed_tests:
            test_id = test.get('test_id', 'unknown')
            
            # Skip if no logs or already processed
            if test_id not in test_logs or test_id in processed_logs:
                continue
            
            log_content = test_logs[test_id]
            processed_logs.add(test_id)
            
            # Extract error patterns using regex
            errors_found = []
            for regex in error_regexes:
                matches = re.finditer(regex, log_content, re.MULTILINE)
                for match in matches:
                    error_text = match.group(0)
                    
                    # Limit length for readability
                    if len(error_text) > 500:
                        error_text = error_text[:500] + "..."
                    
                    # Check if similar to existing patterns
                    similar_pattern = None
                    for pattern in error_patterns:
                        if self._calculate_similarity(error_text, pattern['pattern']) > self.error_pattern_similarity_threshold:
                            similar_pattern = pattern
                            break
                    
                    if similar_pattern:
                        # Update existing pattern
                        similar_pattern['occurrences'] += 1
                        similar_pattern['test_ids'].add(test_id)
                        similar_pattern['components'].add(test.get('component', 'unknown'))
                    else:
                        # Create new pattern
                        error_type = self._categorize_error(error_text)
                        new_pattern = {
                            'pattern': error_text,
                            'error_type': error_type,
                            'occurrences': 1,
                            'test_ids': {test_id},
                            'components': {test.get('component', 'unknown')},
                            'first_seen': datetime.now().isoformat()
                        }
                        error_patterns.append(new_pattern)
                        errors_found.append(new_pattern)
            
            # If no errors found using regex, extract using heuristics
            if not errors_found:
                # Look for lines containing keywords
                error_keywords = ['error', 'exception', 'fail', 'failure', 'timeout', 'crash', 'abort', 'terminated']
                potential_errors = []
                
                for line in log_content.splitlines():
                    if any(keyword in line.lower() for keyword in error_keywords):
                        potential_errors.append(line)
                
                if potential_errors:
                    # Use the most relevant error line (shortest one with most error keywords)
                    potential_errors.sort(
                        key=lambda l: (
                            -sum(keyword in l.lower() for keyword in error_keywords),
                            len(l)
                        )
                    )
                    error_text = potential_errors[0]
                    
                    # Create a new pattern
                    error_type = 'generic_error'
                    new_pattern = {
                        'pattern': error_text,
                        'error_type': error_type,
                        'occurrences': 1,
                        'test_ids': {test_id},
                        'components': {test.get('component', 'unknown')},
                        'first_seen': datetime.now().isoformat()
                    }
                    error_patterns.append(new_pattern)
        
        # Convert sets to lists for JSON serialization
        for pattern in error_patterns:
            pattern['test_ids'] = list(pattern['test_ids'])
            pattern['components'] = list(pattern['components'])
        
        # Sort patterns by occurrence count
        error_patterns.sort(key=lambda p: p['occurrences'], reverse=True)
        
        logger.info(f"Extracted {len(error_patterns)} distinct error patterns")
        return error_patterns
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two error messages.
        
        Args:
            text1 (str): First error text
            text2 (str): Second error text
            
        Returns:
            float: Similarity score (0-1)
        """
        # Simple method: compare word sets
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity coefficient
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _categorize_error(self, error_text: str) -> str:
        """
        Categorize an error based on its text.
        
        Args:
            error_text (str): Error text to categorize
            
        Returns:
            str: Error category
        """
        error_text_lower = error_text.lower()
        
        # Define category patterns
        categories = {
            'null_pointer': ['nullpointerexception', 'none', 'null'],
            'assertion': ['assertionerror', 'assert', 'expected'],
            'timeout': ['timeout', 'timed out', 'wait', 'delay'],
            'connection': ['connection', 'connect', 'socket', 'network', 'http', 'url'],
            'io_error': ['ioexception', 'file', 'path', 'directory', 'permission'],
            'data_format': ['format', 'parse', 'syntax', 'json', 'xml', 'csv'],
            'memory': ['memory', 'outofmemory', 'heap', 'stack overflow'],
            'concurrency': ['concurrent', 'deadlock', 'race condition', 'synchronization'],
            'authentication': ['auth', 'login', 'credentials', 'password', 'token'],
            'environment': ['environment', 'config', 'parameter', 'system', 'version'],
            'dependency': ['dependency', 'import', 'module', 'library', 'package']
        }
        
        # Match error text against category patterns
        for category, keywords in categories.items():
            if any(keyword in error_text_lower for keyword in keywords):
                return category
        
        # Check for specific exception types
        exception_match = re.search(r'([A-Za-z0-9_.]+)(Exception|Error)', error_text)
        if exception_match:
            return exception_match.group(0).lower()
        
        return 'other'
    
    def _identify_root_causes(self, failure_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify common root causes from failure patterns.
        
        Args:
            failure_patterns (list): List of failure patterns
            
        Returns:
            list: List of identified root causes with suggestions
        """
        # Group patterns by error type
        patterns_by_type = defaultdict(list)
        for pattern in failure_patterns:
            patterns_by_type[pattern['error_type']].append(pattern)
        
        # Identify root causes
        root_causes = []
        
        for error_type, patterns in patterns_by_type.items():
            # Sort by occurrence count
            patterns.sort(key=lambda p: p['occurrences'], reverse=True)
            
            # Take top patterns
            top_patterns = patterns[:self.max_patterns_per_category]
            
            # Get affected components
            affected_components = set()
            for pattern in top_patterns:
                affected_components.update(pattern['components'])
            
            # Get error examples
            examples = [p['pattern'] for p in top_patterns[:3]]
            
            # Generate suggestions based on error type
            suggestions = self._generate_suggestions(error_type, examples)
            
            root_cause = {
                'error_type': error_type,
                'frequency': sum(p['occurrences'] for p in top_patterns),
                'affected_components': list(affected_components),
                'examples': examples,
                'suggestions': suggestions
            }
            root_causes.append(root_cause)
        
        # Sort by frequency
        root_causes.sort(key=lambda rc: rc['frequency'], reverse=True)
        
        logger.info(f"Identified {len(root_causes)} potential root causes")
        return root_causes
    
    def _generate_suggestions(self, error_type: str, examples: List[str]) -> List[str]:
        """
        Generate suggestions for addressing a particular error type.
        
        Args:
            error_type (str): Type of error
            examples (list): Example error messages
            
        Returns:
            list: List of suggestions
        """
        # Common suggestion templates based on error type
        suggestion_templates = {
            'null_pointer': [
                "Check for null values before accessing object methods or properties",
                "Verify that external data is properly initialized before use",
                "Add defensive null checks in the code"
            ],
            'assertion': [
                "Review test assertions and expected values",
                "Ensure test data matches expectations",
                "Check for recent changes that might affect expected behavior"
            ],
            'timeout': [
                "Investigate performance issues or bottlenecks",
                "Consider increasing timeout thresholds for slower operations",
                "Check for deadlocks or resource contention"
            ],
            'connection': [
                "Verify that required services/endpoints are running and accessible",
                "Check network connectivity and firewall settings",
                "Add retry logic for transient connection issues"
            ],
            'io_error': [
                "Ensure file paths and permissions are correct",
                "Verify that required files exist before tests run",
                "Add proper cleanup of file resources"
            ],
            'data_format': [
                "Validate input data format against expected schema",
                "Check for changed API contracts or data structures",
                "Add more robust error handling for malformed data"
            ],
            'memory': [
                "Optimize resource usage to reduce memory consumption",
                "Look for memory leaks or resource hoarding",
                "Consider increasing memory limits or optimizing test data"
            ],
            'concurrency': [
                "Review synchronization mechanisms in concurrent code",
                "Check for race conditions or improper thread safety",
                "Consider using thread-safe collections or patterns"
            ],
            'authentication': [
                "Verify that test credentials are valid and not expired",
                "Check for changes in authentication requirements",
                "Ensure proper setup of authentication for tests"
            ],
            'environment': [
                "Validate environment configuration and variables",
                "Check for compatibility with system versions",
                "Ensure consistent environment setup across test runs"
            ],
            'dependency': [
                "Verify that all dependencies are properly installed",
                "Check for version conflicts or compatibility issues",
                "Update dependencies to compatible versions"
            ]
        }
        
        # Get suggestions for specific error type
        if error_type in suggestion_templates:
            return suggestion_templates[error_type]
        
        # Generic suggestions for other error types
        return [
            "Review the error messages for specific details on the failure cause",
            "Check for recent code changes that might have introduced the issue",
            "Consider adding more logging around the failure point for better diagnosis"
        ]
    
    def track_test_stability(
        self,
        historical_results: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze test stability over time and identify flaky tests.
        
        Args:
            historical_results (pd.DataFrame): DataFrame with historical test results
            
        Returns:
            dict: Stability analysis results
        """
        logger.info("Analyzing test stability from historical results")
        
        # Check required columns
        required_cols = ['test_id', 'execution_id', 'status', 'timestamp']
        if not all(col in historical_results.columns for col in required_cols):
            missing = [col for col in required_cols if col not in historical_results.columns]
            logger.error(f"Missing required columns in historical data: {missing}")
            return {'error': f"Missing columns: {missing}"}
        
        # Convert timestamp to datetime if string
        if historical_results['timestamp'].dtype == 'object':
            historical_results['timestamp'] = pd.to_datetime(
                historical_results['timestamp'], errors='coerce'
            )
        
        # Filter for recent results if timestamp is available
        if 'timestamp' in historical_results.columns:
            now = pd.Timestamp.now()
            cutoff = now - pd.Timedelta(days=self.time_window_days)
            recent_results = historical_results[historical_results['timestamp'] >= cutoff]
            if len(recent_results) > 0:
                historical_results = recent_results
                logger.info(f"Using {len(recent_results)} results from the last {self.time_window_days} days")
        
        # Prepare analysis results
        stability_results = {
            'total_tests': len(historical_results['test_id'].unique()),
            'total_executions': len(historical_results['execution_id'].unique()),
            'flaky_tests': [],
            'stable_tests': [],
            'test_stability_scores': {},
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Calculate stability metrics for each test
        test_metrics = {}
        
        for test_id, group in historical_results.groupby('test_id'):
            # Count pass/fail occurrences
            statuses = group['status'].value_counts()
            total_runs = len(group)
            pass_count = statuses.get('PASS', 0)
            fail_count = statuses.get('FAIL', 0)
            
            # Calculate pass rate
            pass_rate = pass_count / total_runs if total_runs > 0 else 0
            
            # Check for transitions (pass->fail or fail->pass)
            transitions = 0
            if total_runs > 1:
                # Sort by timestamp if available
                if 'timestamp' in group.columns:
                    sorted_runs = group.sort_values('timestamp')
                    statuses = sorted_runs['status'].tolist()
                else:
                    statuses = group['status'].tolist()
                
                # Count transitions
                for i in range(1, len(statuses)):
                    if statuses[i] != statuses[i-1]:
                        transitions += 1
            
            # Calculate flakiness score (0-1, higher means more flaky)
            # Based on pass rate variability and transition frequency
            transition_rate = transitions / (total_runs - 1) if total_runs > 1 else 0
            stability_score = 1.0 - (transition_rate * 0.7 + abs(0.5 - pass_rate) * 0.3)
            
            # Higher score means more stable (less flaky)
            normalized_stability = min(1.0, max(0.0, stability_score))
            
            test_metrics[test_id] = {
                'runs': total_runs,
                'pass_count': pass_count,
                'fail_count': fail_count,
                'pass_rate': pass_rate,
                'transitions': transitions,
                'transition_rate': transition_rate,
                'stability_score': normalized_stability
            }
        
        # Identify flaky and stable tests
        for test_id, metrics in test_metrics.items():
            if metrics['runs'] >= 5:  # Only consider tests with enough data
                if metrics['stability_score'] < 0.7 and metrics['transition_rate'] > 0.2:
                    stability_results['flaky_tests'].append({
                        'test_id': test_id,
                        'stability_score': metrics['stability_score'],
                        'pass_rate': metrics['pass_rate'],
                        'runs': metrics['runs'],
                        'transitions': metrics['transitions']
                    })
                elif metrics['stability_score'] > 0.9:
                    stability_results['stable_tests'].append({
                        'test_id': test_id,
                        'stability_score': metrics['stability_score'],
                        'pass_rate': metrics['pass_rate'],
                        'runs': metrics['runs'],
                        'transitions': metrics['transitions']
                    })
        
        # Sort by stability score
        stability_results['flaky_tests'].sort(key=lambda x: x['stability_score'])
        stability_results['stable_tests'].sort(key=lambda x: x['stability_score'], reverse=True)
        
        # Add stability scores for all tests
        stability_results['test_stability_scores'] = {
            test_id: metrics['stability_score'] for test_id, metrics in test_metrics.items()
        }
        
        logger.info(f"Identified {len(stability_results['flaky_tests'])} flaky tests and "
                   f"{len(stability_results['stable_tests'])} stable tests")
        
        return stability_results
    
    def analyze_failure_clusters(
        self,
        test_results: List[Dict[str, Any]],
        max_clusters: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identify clusters of related test failures.
        
        Args:
            test_results (list): List of test result dictionaries
            max_clusters (int): Maximum number of clusters to identify
            
        Returns:
            list: List of failure clusters
        """
        logger.info(f"Analyzing failure clusters in {len(test_results)} test results")
        
        # Extract failed tests
        failed_tests = [r for r in test_results if r.get('status') == 'FAIL']
        if not failed_tests:
            return []
        
        # Group failures by various attributes
        clusters = []
        
        # 1. Cluster by component
        component_clusters = defaultdict(list)
        for test in failed_tests:
            component = test.get('component', 'unknown')
            component_clusters[component].append(test)
        
        significant_component_clusters = [
            {
                'cluster_type': 'component',
                'identifier': component,
                'test_count': len(tests),
                'tests': [t.get('test_id', 'unknown') for t in tests],
                'significance': len(tests) / len(failed_tests)
            }
            for component, tests in component_clusters.items()
            if len(tests) > 1  # Only include clusters with multiple tests
        ]
        
        # 2. Cluster by timestamp (time-based clusters)
        if all('timestamp' in test for test in failed_tests):
            # Convert timestamps to datetime
            for test in failed_tests:
                if isinstance(test['timestamp'], str):
                    test['timestamp'] = pd.to_datetime(test['timestamp'])
            
            # Sort by timestamp
            sorted_by_time = sorted(failed_tests, key=lambda t: t['timestamp'])
            
            # Define time window (e.g., 1 hour)
            from datetime import timedelta
            time_window = timedelta(hours=1)
            
            time_clusters = []
            current_cluster = [sorted_by_time[0]]
            cluster_start_time = sorted_by_time[0]['timestamp']
            
            for test in sorted_by_time[1:]:
                if test['timestamp'] - cluster_start_time <= time_window:
                    current_cluster.append(test)
                else:
                    if len(current_cluster) > 1:
                        time_clusters.append(current_cluster)
                    current_cluster = [test]
                    cluster_start_time = test['timestamp']
            
            # Add the last cluster if it's significant
            if len(current_cluster) > 1:
                time_clusters.append(current_cluster)
            
            # Create cluster objects
            timestamp_clusters = [
                {
                    'cluster_type': 'time_window',
                    'identifier': cluster[0]['timestamp'].isoformat(),
                    'test_count': len(cluster),
                    'tests': [t.get('test_id', 'unknown') for t in cluster],
                    'significance': len(cluster) / len(failed_tests),
                    'start_time': cluster[0]['timestamp'].isoformat(),
                    'end_time': cluster[-1]['timestamp'].isoformat()
                }
                for cluster in time_clusters
            ]
            
            clusters.extend(timestamp_clusters)
        
        # 3. Combine all cluster types and sort by significance
        clusters.extend(significant_component_clusters)
        clusters.sort(key=lambda c: c['significance'], reverse=True)
        
        # Limit to max_clusters
        top_clusters = clusters[:max_clusters]
        
        logger.info(f"Identified {len(top_clusters)} significant failure clusters")
        return top_clusters
    
    def suggest_fixes(
        self,
        failure_analysis: Dict[str, Any],
        component_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Suggest potential fixes for identified failure patterns.
        
        Args:
            failure_analysis (dict): Analysis results from analyze_test_runs
            component_info (dict, optional): Additional information about components
            
        Returns:
            dict: Dictionary mapping test_id to suggested fixes
        """
        logger.info("Generating fix suggestions for test failures")
        
        suggestions = {}
        
        # Extract failure patterns
        failure_patterns = failure_analysis.get('failure_patterns', [])
        root_causes = failure_analysis.get('common_root_causes', [])
        
        # Create a mapping from error types to suggestions
        error_type_suggestions = {}
        for cause in root_causes:
            error_type_suggestions[cause['error_type']] = cause['suggestions']
        
        # Process each failure pattern
        for pattern in failure_patterns:
            # Get test IDs affected by this pattern
            test_ids = pattern.get('test_ids', [])
            error_type = pattern.get('error_type', 'unknown')
            
            # Get suggestions for this error type
            pattern_suggestions = error_type_suggestions.get(
                error_type, 
                ["Review the error message and recent code changes."]
            )
            
            # Create suggestion object
            suggestion = {
                'error_pattern': pattern['pattern'],
                'error_type': error_type,
                'suggestions': pattern_suggestions[:self.max_suggestions_per_failure],
                'confidence': min(1.0, pattern['occurrences'] / 10)  # Confidence based on occurrences
            }
            
            # Add component-specific suggestions if available
            if component_info and 'components' in pattern:
                for component in pattern['components']:
                    if component in component_info:
                        component_suggestions = component_info[component].get('common_fixes', [])
                        if component_suggestions:
                            suggestion['component_specific_suggestions'] = component_suggestions
                            break
            
            # Add suggestion to each affected test
            for test_id in test_ids:
                if test_id not in suggestions:
                    suggestions[test_id] = []
                
                suggestions[test_id].append(suggestion)
        
        logger.info(f"Generated suggestions for {len(suggestions)} failed tests")
        return suggestions
    
    def save_analysis_result(
        self,
        analysis_result: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Save the analysis results to a file.
        
        Args:
            analysis_result (dict): Analysis results to save
            output_path (str): Path to save the results
            
        Returns:
            str: Path to the saved file
        """
        logger.info(f"Saving analysis results to {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(analysis_result, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_path}")
        return output_path

def load_test_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load test results from a file.
    
    Args:
        file_path (str): Path to the test results file
        
    Returns:
        list: List of test result dictionaries
    """
    logger.info(f"Loading test results from {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if it's a single test run with test_results array
        if 'test_results' in data and isinstance(data['test_results'], list):
            results = data['test_results']
        # Check if it's an array of test results
        elif isinstance(data, list):
            results = data
        else:
            logger.error("Unexpected format for test results file")
            return []
        
        logger.info(f"Loaded {len(results)} test results")
        return results
    
    except Exception as e:
        logger.error(f"Error loading test results: {str(e)}")
        return []

def load_test_logs(directory_path: str) -> Dict[str, str]:
    """
    Load test logs from a directory.
    
    Args:
        directory_path (str): Path to the directory containing test logs
        
    Returns:
        dict: Dictionary mapping test_id to log content
    """
    logger.info(f"Loading test logs from {directory_path}")
    
    test_logs = {}
    
    try:
        if not os.path.exists(directory_path):
            logger.error(f"Log directory does not exist: {directory_path}")
            return {}
        
        # Iterate through log files
        log_count = 0
        for filename in os.listdir(directory_path):
            if not (filename.endswith('.log') or filename.endswith('.txt')):
                continue
                
            file_path = os.path.join(directory_path, filename)
            
            # Extract test_id from filename
            test_id = os.path.splitext(filename)[0]
            
            # Read log content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                test_logs[test_id] = content
                log_count += 1
        
        logger.info(f"Loaded {log_count} test logs")
        return test_logs
    
    except Exception as e:
        logger.error(f"Error loading test logs: {str(e)}")
        return {}

def main():
    """
    Main function to demonstrate the root cause analyzer.
    """
    # Set up paths
    results_file = os.path.join(config.REPORTS_DIR, 'execution_results', 'latest_results.json')
    logs_dir = os.path.join(config.DATA_DIR, 'logs')
    output_dir = os.path.join(config.REPORTS_DIR, 'root_cause_analysis')
    
    # Create a root cause analyzer
    analyzer = RootCauseAnalyzer(
        error_pattern_similarity_threshold=0.7,
        max_patterns_per_category=10,
        max_suggestions_per_failure=3,
        time_window_days=30
    )
    
    # Load test results
    test_results = load_test_results(results_file)
    if not test_results:
        logger.error("No test results loaded. Exiting.")
        return
    
    # Load test logs
    test_logs = load_test_logs(logs_dir)
    
    # Analyze test runs
    analysis_result = analyzer.analyze_test_runs(test_results, test_logs)
    
    # Save analysis results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f"root_cause_analysis_{timestamp}.json")
    analyzer.save_analysis_result(analysis_result, output_path)
    
    # Generate suggestions for fixes
    suggestions = analyzer.suggest_fixes(analysis_result)
    
    # Print summary
    print(f"Root cause analysis completed. Results saved to {output_path}")
    print(f"Analyzed {analysis_result['total_tests']} tests with {analysis_result['failed_tests']} failures")
    print(f"Identified {len(analysis_result['failure_patterns'])} failure patterns")
    print(f"Generated suggestions for {len(suggestions)} failed tests")

if __name__ == "__main__":
    main() 