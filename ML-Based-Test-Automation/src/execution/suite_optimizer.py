"""
Test Suite Optimizer Module for ML-Based Test Automation.
This module provides functionality to analyze and optimize test suites by
identifying redundant tests, maximizing coverage, and minimizing execution time.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from collections import defaultdict

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"suite_optimizer_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestSuiteOptimizer:
    """
    Class for analyzing and optimizing test suites to reduce redundancy and execution time
    while maintaining coverage and effectiveness.
    """
    
    def __init__(
        self,
        coverage_weight: float = 0.5,
        execution_time_weight: float = 0.3,
        failure_detection_weight: float = 0.2,
        min_coverage_threshold: float = 0.95,
        max_redundancy_threshold: float = 0.8,
        max_suite_reduction: float = 0.3
    ):
        """
        Initialize the TestSuiteOptimizer.
        
        Args:
            coverage_weight (float): Weight given to code coverage in optimization
            execution_time_weight (float): Weight given to execution time in optimization
            failure_detection_weight (float): Weight given to failure detection in optimization
            min_coverage_threshold (float): Minimum coverage to maintain (0-1)
            max_redundancy_threshold (float): Maximum acceptable redundancy level (0-1)
            max_suite_reduction (float): Maximum test suite reduction allowed (0-1)
        """
        self.coverage_weight = coverage_weight
        self.execution_time_weight = execution_time_weight
        self.failure_detection_weight = failure_detection_weight
        self.min_coverage_threshold = min_coverage_threshold
        self.max_redundancy_threshold = max_redundancy_threshold
        self.max_suite_reduction = max_suite_reduction
        
        logger.info(f"TestSuiteOptimizer initialized with weights: "
                   f"coverage={coverage_weight}, time={execution_time_weight}, "
                   f"failure={failure_detection_weight}")
        logger.info(f"Thresholds: min_coverage={min_coverage_threshold}, "
                   f"max_redundancy={max_redundancy_threshold}, "
                   f"max_reduction={max_suite_reduction}")
    
    def analyze_test_redundancy(
        self,
        test_data: pd.DataFrame,
        coverage_data: Optional[Dict[str, List[str]]] = None,
        test_results_history: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Analyze test suite to identify redundancy patterns.
        
        Args:
            test_data (pd.DataFrame): DataFrame containing test metadata
            coverage_data (dict, optional): Dictionary mapping test_id to covered components/files
            test_results_history (pd.DataFrame, optional): Historical test results
            
        Returns:
            dict: Analysis results including redundancy scores and patterns
        """
        logger.info(f"Analyzing test redundancy for {len(test_data)} tests")
        
        # Ensure required columns exist
        required_cols = ['test_id', 'test_name', 'component']
        missing_cols = [col for col in required_cols if col not in test_data.columns]
        if missing_cols:
            logger.warning(f"Test data missing columns: {missing_cols}")
            # Add default values for missing columns
            for col in missing_cols:
                if col == 'component':
                    test_data[col] = 'unknown'
                else:
                    test_data[col] = test_data.index if col == 'test_id' else 'unknown'
        
        # Create analysis result structure
        analysis_result = {
            'redundancy_groups': [],
            'redundancy_by_test': {},
            'overall_redundancy_score': 0.0,
            'optimization_targets': [],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Calculate component-based redundancy
        component_groups = test_data.groupby('component')['test_id'].apply(list).to_dict()
        
        component_redundancy = []
        for component, tests in component_groups.items():
            if len(tests) > 1:
                avg_time = test_data[test_data['test_id'].isin(tests)]['avg_execution_time'].mean() \
                    if 'avg_execution_time' in test_data.columns else 0
                
                component_redundancy.append({
                    'component': component,
                    'test_count': len(tests),
                    'test_ids': tests,
                    'avg_execution_time': float(avg_time),
                    'redundancy_type': 'component-based'
                })
        
        # Calculate coverage-based redundancy if coverage data is provided
        coverage_redundancy = []
        if coverage_data:
            logger.info("Analyzing coverage-based redundancy")
            
            # Invert coverage mapping (file -> tests that cover it)
            file_to_tests = defaultdict(set)
            for test_id, covered_files in coverage_data.items():
                for file in covered_files:
                    file_to_tests[file].add(test_id)
            
            # Find tests that cover similar files
            coverage_overlap = defaultdict(dict)
            test_ids = list(coverage_data.keys())
            
            for i, test_id1 in enumerate(test_ids):
                files1 = set(coverage_data[test_id1])
                if not files1:
                    continue
                    
                for test_id2 in test_ids[i+1:]:
                    files2 = set(coverage_data[test_id2])
                    if not files2:
                        continue
                    
                    # Calculate Jaccard similarity of coverage
                    intersection = len(files1.intersection(files2))
                    union = len(files1.union(files2))
                    
                    if union > 0:
                        similarity = intersection / union
                        if similarity > self.max_redundancy_threshold:
                            coverage_overlap[test_id1][test_id2] = similarity
            
            # Group tests by coverage overlap
            processed_tests = set()
            for test_id, overlaps in coverage_overlap.items():
                if test_id in processed_tests:
                    continue
                    
                # Find all tests that have high overlap with this test
                group = {test_id}
                group.update(overlaps.keys())
                processed_tests.update(group)
                
                if len(group) > 1:
                    avg_time = test_data[test_data['test_id'].isin(group)]['avg_execution_time'].mean() \
                        if 'avg_execution_time' in test_data.columns else 0
                    
                    coverage_redundancy.append({
                        'test_ids': list(group),
                        'test_count': len(group),
                        'avg_execution_time': float(avg_time),
                        'average_similarity': float(sum(overlaps.values()) / len(overlaps)) if overlaps else 0,
                        'redundancy_type': 'coverage-based'
                    })
        
        # Calculate result-based redundancy if history is provided
        result_redundancy = []
        if test_results_history is not None and not test_results_history.empty:
            logger.info("Analyzing result-based redundancy")
            
            # Ensure required columns exist in history
            history_cols = ['test_id', 'execution_id', 'status']
            if all(col in test_results_history.columns for col in history_cols):
                # Create a pivot table of test results
                result_pivot = test_results_history.pivot_table(
                    index='test_id',
                    columns='execution_id',
                    values='status',
                    fill_value='NOT_RUN'
                )
                
                # Calculate correlation between test results
                try:
                    # Convert status to numeric (1 for PASS, 0 for FAIL)
                    numeric_results = result_pivot.applymap(lambda x: 1 if x == 'PASS' else 0)
                    
                    # Calculate correlation matrix
                    corr_matrix = numeric_results.T.corr()
                    
                    # Find highly correlated tests
                    highly_correlated = {}
                    for test_id in corr_matrix.index:
                        correlated_tests = corr_matrix[test_id][
                            (corr_matrix[test_id] > self.max_redundancy_threshold) & 
                            (corr_matrix[test_id] < 1.0)
                        ].index.tolist()
                        
                        if correlated_tests:
                            highly_correlated[test_id] = correlated_tests
                    
                    # Group correlated tests
                    processed_tests = set()
                    for test_id, correlated in highly_correlated.items():
                        if test_id in processed_tests:
                            continue
                            
                        group = {test_id}
                        group.update(correlated)
                        processed_tests.update(group)
                        
                        if len(group) > 1:
                            avg_time = test_data[test_data['test_id'].isin(group)]['avg_execution_time'].mean() \
                                if 'avg_execution_time' in test_data.columns else 0
                            
                            result_redundancy.append({
                                'test_ids': list(group),
                                'test_count': len(group),
                                'avg_execution_time': float(avg_time),
                                'redundancy_type': 'result-based'
                            })
                except Exception as e:
                    logger.error(f"Error calculating result-based redundancy: {str(e)}")
        
        # Combine all redundancy groups
        all_redundancy = component_redundancy + coverage_redundancy + result_redundancy
        analysis_result['redundancy_groups'] = all_redundancy
        
        # Calculate per-test redundancy scores
        redundancy_by_test = {}
        for test_id in test_data['test_id']:
            count = sum(1 for group in all_redundancy if test_id in group.get('test_ids', []))
            redundancy_by_test[test_id] = min(1.0, count / max(1, len(all_redundancy)))
        
        analysis_result['redundancy_by_test'] = redundancy_by_test
        
        # Calculate overall redundancy score
        if redundancy_by_test:
            overall_score = sum(redundancy_by_test.values()) / len(redundancy_by_test)
            analysis_result['overall_redundancy_score'] = overall_score
            logger.info(f"Overall redundancy score: {overall_score:.2f}")
        
        # Identify optimization targets
        if 'avg_execution_time' in test_data.columns:
            # Find redundant groups with high execution time
            high_impact_groups = sorted(
                [g for g in all_redundancy if g['test_count'] > 1],
                key=lambda g: g['avg_execution_time'] * g['test_count'],
                reverse=True
            )[:5]  # Top 5 high-impact groups
            
            for group in high_impact_groups:
                analysis_result['optimization_targets'].append({
                    'group': group,
                    'potential_time_saved': group['avg_execution_time'] * (group['test_count'] - 1),
                    'recommendation': 'Consider consolidating these tests or running a subset'
                })
        
        return analysis_result
    
    def generate_optimized_suite(
        self,
        test_data: pd.DataFrame,
        redundancy_analysis: Dict[str, Any],
        coverage_data: Optional[Dict[str, List[str]]] = None,
        historical_effectiveness: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate an optimized test suite based on redundancy analysis.
        
        Args:
            test_data (pd.DataFrame): DataFrame containing test metadata
            redundancy_analysis (dict): Results from analyze_test_redundancy
            coverage_data (dict, optional): Dictionary mapping test_id to covered components/files
            historical_effectiveness (dict, optional): Dict of test_id to historical effectiveness score
            
        Returns:
            dict: Optimized test suite and optimization metrics
        """
        logger.info(f"Generating optimized test suite from {len(test_data)} tests")
        
        # Create a copy of the test data for optimization
        optimized_suite = test_data.copy()
        
        # Create optimization result structure
        optimization_result = {
            'original_test_count': len(test_data),
            'original_execution_time': 0,
            'optimized_test_count': 0,
            'optimized_execution_time': 0,
            'removed_tests': [],
            'kept_tests': [],
            'coverage_impact': 0,
            'optimization_metrics': {},
            'optimization_timestamp': datetime.now().isoformat()
        }
        
        # Calculate original execution time
        if 'avg_execution_time' in test_data.columns:
            optimization_result['original_execution_time'] = float(test_data['avg_execution_time'].sum())
        
        # Define scoring function for tests based on multiple factors
        def calculate_test_score(test_id):
            score = 0.0
            
            # Factor 1: Redundancy (lower is better)
            redundancy = redundancy_analysis['redundancy_by_test'].get(test_id, 0)
            redundancy_score = 1.0 - redundancy  # Invert so lower redundancy gets higher score
            
            # Factor 2: Execution time (lower is better)
            time_score = 0.0
            if 'avg_execution_time' in test_data.columns:
                test_time = test_data.loc[test_data['test_id'] == test_id, 'avg_execution_time'].iloc[0]
                max_time = test_data['avg_execution_time'].max()
                time_score = 1.0 - (test_time / max_time) if max_time > 0 else 1.0
            
            # Factor 3: Historical effectiveness (higher is better)
            effectiveness_score = 0.5  # Default middle value
            if historical_effectiveness and test_id in historical_effectiveness:
                effectiveness_score = historical_effectiveness[test_id]
            
            # Factor 4: Coverage (higher is better)
            coverage_score = 0.5  # Default middle value
            if coverage_data and test_id in coverage_data:
                coverage_breadth = len(coverage_data[test_id])
                coverage_score = min(1.0, coverage_breadth / 100)  # Normalize, assuming max of 100 files
            
            # Combine scores using weights
            score = (
                (self.coverage_weight * coverage_score) +
                (self.execution_time_weight * time_score) +
                (self.failure_detection_weight * effectiveness_score) +
                # Add redundancy as a bonus factor
                (0.2 * redundancy_score)
            )
            
            return score
        
        # Calculate test scores
        test_scores = {test_id: calculate_test_score(test_id) for test_id in test_data['test_id']}
        
        # Process redundancy groups to select tests to keep and remove
        removed_tests = set()
        redundancy_groups = redundancy_analysis.get('redundancy_groups', [])
        
        for group in redundancy_groups:
            test_ids = group.get('test_ids', [])
            if len(test_ids) <= 1:
                continue
                
            # Calculate scores for tests in this group
            group_scores = {test_id: test_scores.get(test_id, 0) for test_id in test_ids}
            
            # Sort by score (descending)
            sorted_tests = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Keep the top test(s) depending on group size and redundancy type
            keep_count = max(1, int(len(test_ids) * (1 - self.max_suite_reduction)))
            tests_to_keep = [t[0] for t in sorted_tests[:keep_count]]
            tests_to_remove = [t[0] for t in sorted_tests[keep_count:]]
            
            # Don't remove tests that are already in the keep list from another group
            tests_to_remove = [t for t in tests_to_remove if t not in tests_to_keep]
            
            # Update removed tests set
            removed_tests.update(tests_to_remove)
        
        # Apply the optimization
        kept_tests = [t for t in test_data['test_id'] if t not in removed_tests]
        optimized_suite = test_data[test_data['test_id'].isin(kept_tests)]
        
        # Calculate optimization metrics
        optimization_result['optimized_test_count'] = len(optimized_suite)
        if 'avg_execution_time' in optimized_suite.columns:
            optimization_result['optimized_execution_time'] = float(optimized_suite['avg_execution_time'].sum())
        
        optimization_result['removed_tests'] = list(removed_tests)
        optimization_result['kept_tests'] = kept_tests
        
        # Calculate coverage impact if coverage data is provided
        if coverage_data:
            # Calculate original coverage
            all_covered_files = set()
            for test_id, files in coverage_data.items():
                if test_id in test_data['test_id'].values:
                    all_covered_files.update(files)
            
            # Calculate optimized coverage
            optimized_covered_files = set()
            for test_id, files in coverage_data.items():
                if test_id in kept_tests:
                    optimized_covered_files.update(files)
            
            # Calculate coverage ratio
            if all_covered_files:
                coverage_ratio = len(optimized_covered_files) / len(all_covered_files)
                optimization_result['coverage_impact'] = coverage_ratio
                logger.info(f"Coverage maintained: {coverage_ratio:.2%}")
        
        # Calculate additional metrics
        time_reduction = 0
        if optimization_result['original_execution_time'] > 0:
            time_reduction = 1 - (optimization_result['optimized_execution_time'] / 
                                 optimization_result['original_execution_time'])
        
        test_reduction = 1 - (len(optimized_suite) / len(test_data)) if len(test_data) > 0 else 0
        
        optimization_result['optimization_metrics'] = {
            'execution_time_reduction': float(time_reduction),
            'test_count_reduction': float(test_reduction),
            'time_saved_seconds': float(optimization_result['original_execution_time'] - 
                                       optimization_result['optimized_execution_time'])
        }
        
        logger.info(f"Optimization complete: "
                   f"Removed {len(removed_tests)}/{len(test_data)} tests "
                   f"({test_reduction:.2%}), "
                   f"Time reduction: {time_reduction:.2%}")
        
        return {
            'optimized_suite': optimized_suite,
            'optimization_result': optimization_result
        }
    
    def save_optimization_result(
        self, 
        optimization_result: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Save the optimization results to a file.
        
        Args:
            optimization_result (dict): Optimization results to save
            output_path (str): Path to save the results
            
        Returns:
            str: Path to the saved file
        """
        logger.info(f"Saving optimization results to {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Convert DataFrame to dict if present
        result_copy = optimization_result.copy()
        if 'optimized_suite' in result_copy and isinstance(result_copy['optimized_suite'], pd.DataFrame):
            result_copy['optimized_suite'] = result_copy['optimized_suite'].to_dict(orient='records')
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(result_copy, f, indent=2)
        
        logger.info(f"Optimization results saved to {output_path}")
        return output_path
    
    def generate_test_subset(
        self,
        test_data: pd.DataFrame,
        coverage_data: Optional[Dict[str, List[str]]] = None,
        time_budget: Optional[float] = None,
        count_limit: Optional[int] = None,
        prioritize_coverage: bool = True
    ) -> pd.DataFrame:
        """
        Generate a subset of tests optimized for coverage, execution time, or test count.
        
        Args:
            test_data (pd.DataFrame): DataFrame containing test metadata
            coverage_data (dict, optional): Dictionary mapping test_id to covered components/files
            time_budget (float, optional): Maximum execution time budget in seconds
            count_limit (int, optional): Maximum number of tests to include
            prioritize_coverage (bool): Whether to prioritize coverage over other factors
            
        Returns:
            pd.DataFrame: Optimized subset of tests
        """
        logger.info(f"Generating test subset from {len(test_data)} tests")
        
        # Check for execution time column if time_budget is specified
        if time_budget and 'avg_execution_time' not in test_data.columns:
            logger.warning("avg_execution_time column not found in test data. Cannot optimize for time budget.")
            time_budget = None
        
        # Create scoring criteria based on coverage and other factors
        if coverage_data and prioritize_coverage:
            logger.info("Prioritizing coverage in subset generation")
            
            # Calculate coverage per test
            all_files = set()
            for files in coverage_data.values():
                all_files.update(files)
            
            logger.info(f"Total files that can be covered: {len(all_files)}")
            
            # Greedy algorithm to maximize coverage
            remaining_tests = test_data.copy()
            selected_tests = pd.DataFrame(columns=test_data.columns)
            covered_files = set()
            total_time = 0
            
            while not remaining_tests.empty:
                # Calculate added coverage for each remaining test
                best_test = None
                best_score = -1
                
                for idx, row in remaining_tests.iterrows():
                    test_id = row['test_id']
                    
                    # Skip tests that exceed budget or count limit
                    if time_budget and total_time + row.get('avg_execution_time', 0) > time_budget:
                        continue
                        
                    if count_limit and len(selected_tests) >= count_limit:
                        break
                    
                    # Calculate new coverage
                    if test_id in coverage_data:
                        new_files = set(coverage_data[test_id]) - covered_files
                        new_coverage = len(new_files)
                        
                        # Score = new_coverage / execution_time (if available)
                        execution_time = row.get('avg_execution_time', 1)
                        score = new_coverage / execution_time if execution_time > 0 else new_coverage
                        
                        if score > best_score:
                            best_score = score
                            best_test = idx
                
                # If we couldn't find a test that adds coverage or we've hit limits, exit
                if best_test is None or best_score <= 0:
                    break
                
                # Add the best test to the selected set
                test_row = remaining_tests.loc[best_test]
                selected_tests = pd.concat([selected_tests, pd.DataFrame([test_row])], ignore_index=True)
                
                # Update covered files and total time
                test_id = test_row['test_id']
                if test_id in coverage_data:
                    covered_files.update(coverage_data[test_id])
                
                total_time += test_row.get('avg_execution_time', 0)
                
                # Remove the selected test from remaining tests
                remaining_tests = remaining_tests.drop(best_test)
                
                # Log progress periodically
                if len(selected_tests) % 10 == 0:
                    coverage_pct = len(covered_files) / len(all_files) if all_files else 0
                    logger.info(f"Selected {len(selected_tests)} tests, "
                               f"coverage: {coverage_pct:.2%}, "
                               f"time: {total_time:.2f}s")
                
                # Check if we've reached our limits
                if count_limit and len(selected_tests) >= count_limit:
                    logger.info(f"Reached test count limit: {count_limit}")
                    break
                    
                if time_budget and total_time >= time_budget:
                    logger.info(f"Reached time budget: {time_budget}s")
                    break
            
            coverage_pct = len(covered_files) / len(all_files) if all_files else 0
            logger.info(f"Final subset: {len(selected_tests)} tests, "
                       f"coverage: {coverage_pct:.2%}, "
                       f"time: {total_time:.2f}s")
            
            return selected_tests
            
        else:
            # If no coverage data or not prioritizing coverage, use simpler heuristics
            logger.info("Using simple heuristics for subset generation")
            
            # Sort by priority or effectiveness if available
            if 'priority' in test_data.columns:
                # Map priority strings to numeric values
                priority_map = {
                    'critical': 4,
                    'high': 3,
                    'medium': 2,
                    'low': 1
                }
                
                # Create a numeric priority column
                test_data['priority_value'] = test_data['priority'].str.lower().map(
                    lambda p: priority_map.get(p, 2)  # Default to medium
                )
                
                sorted_tests = test_data.sort_values('priority_value', ascending=False)
                
            elif 'recent_failure_rate' in test_data.columns:
                # Sort by failure rate (higher first)
                sorted_tests = test_data.sort_values('recent_failure_rate', ascending=False)
                
            else:
                # Default sorting by test_id
                sorted_tests = test_data
            
            # Apply time budget if specified
            if time_budget:
                selected_tests = pd.DataFrame(columns=test_data.columns)
                total_time = 0
                
                for idx, row in sorted_tests.iterrows():
                    if total_time + row.get('avg_execution_time', 0) <= time_budget:
                        selected_tests = pd.concat([selected_tests, pd.DataFrame([row])], ignore_index=True)
                        total_time += row.get('avg_execution_time', 0)
                    
                    if count_limit and len(selected_tests) >= count_limit:
                        break
                
                logger.info(f"Selected {len(selected_tests)} tests within time budget: {total_time:.2f}s")
                return selected_tests
                
            # Apply count limit if specified
            elif count_limit:
                return sorted_tests.head(count_limit)
            
            # If no constraints, return all tests in priority order
            return sorted_tests
    
    def optimize_test_order(
        self,
        test_data: pd.DataFrame,
        coverage_data: Optional[Dict[str, List[str]]] = None,
        dependency_data: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Optimize the order of tests to maximize early failures and respect dependencies.
        
        Args:
            test_data (pd.DataFrame): DataFrame containing test metadata
            coverage_data (dict, optional): Dictionary mapping test_id to covered components/files
            dependency_data (dict, optional): Dictionary mapping test_id to dependencies
            
        Returns:
            pd.DataFrame: Reordered test data
        """
        logger.info(f"Optimizing test order for {len(test_data)} tests")
        
        # First, handle dependencies if provided
        if dependency_data:
            logger.info("Ordering tests based on dependencies")
            
            # Create a directed graph of dependencies
            import networkx as nx
            G = nx.DiGraph()
            
            # Add all tests as nodes
            for test_id in test_data['test_id']:
                G.add_node(test_id)
            
            # Add dependency edges
            for test_id, deps in dependency_data.items():
                if test_id in test_data['test_id'].values:
                    for dep in deps:
                        if dep in test_data['test_id'].values:
                            G.add_edge(dep, test_id)  # dep must run before test_id
            
            try:
                # Perform topological sort to respect dependencies
                ordered_tests = list(nx.topological_sort(G))
                
                # Reorder the test DataFrame
                # Create a position mapping for sorting
                pos_map = {test_id: i for i, test_id in enumerate(ordered_tests)}
                
                # Add tests that are not in the dependency graph
                missing_tests = set(test_data['test_id']) - set(ordered_tests)
                for test_id in missing_tests:
                    pos_map[test_id] = len(pos_map)
                
                # Create position column for sorting
                test_data['dep_position'] = test_data['test_id'].map(pos_map)
                dependency_ordered = test_data.sort_values('dep_position')
                
                # Remove temporary column
                dependency_ordered = dependency_ordered.drop(columns=['dep_position'])
                
                logger.info(f"Ordered {len(ordered_tests)} tests based on dependencies")
                
            except nx.NetworkXUnfeasible:
                logger.warning("Dependency graph contains cycles. Cannot produce a valid order.")
                # Fall back to original order
                dependency_ordered = test_data
                
        else:
            # No dependencies, use original order
            dependency_ordered = test_data
        
        # Next, optimize the order within dependency constraints
        # Prioritize tests more likely to fail and with shorter execution time
        reordered = dependency_ordered.copy()
        
        # Create a composite score for ordering
        if 'recent_failure_rate' in reordered.columns:
            # Higher score = run earlier
            reordered['order_score'] = reordered['recent_failure_rate']
            
            # Factor in execution time if available (prioritize shorter tests)
            if 'avg_execution_time' in reordered.columns:
                max_time = reordered['avg_execution_time'].max()
                if max_time > 0:
                    time_factor = 1 - (reordered['avg_execution_time'] / max_time)
                    # Combine failure rate and time factor
                    reordered['order_score'] = (0.7 * reordered['order_score']) + (0.3 * time_factor)
            
            # Sort by order score (descending)
            final_ordered = reordered.sort_values('order_score', ascending=False)
            
            # Remove temporary column
            final_ordered = final_ordered.drop(columns=['order_score'])
            
        else:
            # If no failure rate data, just use the dependency order
            final_ordered = reordered
        
        logger.info("Test order optimization complete")
        return final_ordered

def load_coverage_data(file_path: str) -> Dict[str, List[str]]:
    """
    Load test coverage data from a JSON file.
    
    Args:
        file_path (str): Path to the coverage data file
        
    Returns:
        dict: Dictionary mapping test_id to covered components/files
    """
    logger.info(f"Loading coverage data from {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            coverage_data = json.load(f)
        
        return coverage_data
    except Exception as e:
        logger.error(f"Error loading coverage data: {str(e)}")
        return {}

def load_test_history(file_path: str) -> pd.DataFrame:
    """
    Load historical test results.
    
    Args:
        file_path (str): Path to the historical test results file
        
    Returns:
        pd.DataFrame: DataFrame with historical test results
    """
    logger.info(f"Loading test history from {file_path}")
    
    try:
        # Try JSON format first
        try:
            with open(file_path, 'r') as f:
                history_data = json.load(f)
            
            # Convert to DataFrame
            rows = []
            for execution in history_data:
                execution_id = execution.get('execution_id', 'unknown')
                timestamp = execution.get('timestamp', 'unknown')
                
                for result in execution.get('test_results', []):
                    rows.append({
                        'test_id': result.get('test_id', 'unknown'),
                        'execution_id': execution_id,
                        'timestamp': timestamp,
                        'status': result.get('status', 'unknown'),
                        'execution_time': result.get('execution_time', 0)
                    })
            
            history_df = pd.DataFrame(rows)
            
        except:
            # Try CSV format
            history_df = pd.read_csv(file_path)
        
        return history_df
    
    except Exception as e:
        logger.error(f"Error loading test history: {str(e)}")
        return pd.DataFrame()

def main():
    """
    Main function to demonstrate the test suite optimizer.
    """
    # Set up paths
    metadata_file = os.path.join(config.DATA_DIR, 'sample', 'test_metadata.csv')
    output_dir = os.path.join(config.REPORTS_DIR, 'suite_optimization')
    
    # Create a test suite optimizer
    optimizer = TestSuiteOptimizer(
        coverage_weight=0.5,
        execution_time_weight=0.3,
        failure_detection_weight=0.2,
        min_coverage_threshold=0.95,
        max_redundancy_threshold=0.7,
        max_suite_reduction=0.3
    )
    
    # Load test data
    test_data = pd.read_csv(metadata_file)
    logger.info(f"Loaded {len(test_data)} tests from metadata file")
    
    # Analyze test redundancy
    redundancy_analysis = optimizer.analyze_test_redundancy(test_data)
    
    # Generate optimized test suite
    optimization_result = optimizer.generate_optimized_suite(
        test_data=test_data,
        redundancy_analysis=redundancy_analysis
    )
    
    # Save optimization results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f"optimization_result_{timestamp}.json")
    optimizer.save_optimization_result(optimization_result, output_path)
    
    # Generate test subset
    if 'avg_execution_time' in test_data.columns:
        total_time = test_data['avg_execution_time'].sum()
        time_budget = total_time * 0.5  # 50% time budget
        
        subset = optimizer.generate_test_subset(
            test_data=test_data,
            time_budget=time_budget,
            count_limit=50
        )
        
        logger.info(f"Generated test subset with {len(subset)} tests")
    
    # Optimize test order
    reordered_tests = optimizer.optimize_test_order(test_data)
    
    print(f"Test suite optimization completed. Results saved to {output_path}")
    print(f"Original test count: {len(test_data)}")
    print(f"Optimized test count: {optimization_result['optimization_result']['optimized_test_count']}")
    print(f"Execution time reduction: {optimization_result['optimization_result']['optimization_metrics']['execution_time_reduction']:.2%}")

if __name__ == "__main__":
    main() 