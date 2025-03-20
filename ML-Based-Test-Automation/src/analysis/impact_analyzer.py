"""
Impact Analysis Module for ML-Based Test Automation.
This module analyzes code changes to determine which tests are affected and should be prioritized
for execution, enabling more efficient testing strategies for incremental code changes.
"""

import os
import sys
import json
import logging
import subprocess
import re
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
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"impact_analysis_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImpactAnalyzer:
    """
    Class for analyzing the impact of code changes on the test suite.
    Determines which tests should be run based on code changes.
    """
    
    def __init__(
        self,
        code_coverage_data_path: Optional[str] = None,
        dependency_mapping_path: Optional[str] = None,
        git_repo_path: Optional[str] = None,
        test_metadata_path: Optional[str] = None,
        min_confidence_threshold: float = 0.5
    ):
        """
        Initialize the ImpactAnalyzer.
        
        Args:
            code_coverage_data_path (str, optional): Path to code coverage data
            dependency_mapping_path (str, optional): Path to dependency mapping file
            git_repo_path (str, optional): Path to git repository
            test_metadata_path (str, optional): Path to test metadata file
            min_confidence_threshold (float): Minimum confidence threshold for impacted test selection
        """
        self.code_coverage_data_path = code_coverage_data_path
        self.dependency_mapping_path = dependency_mapping_path
        self.git_repo_path = git_repo_path or os.getcwd()
        self.test_metadata_path = test_metadata_path
        self.min_confidence_threshold = min_confidence_threshold
        
        # Initialize mappings
        self.file_to_tests_map = {}  # Maps source files to tests that use them
        self.component_to_tests_map = {}  # Maps components to tests that test them
        self.dependency_graph = defaultdict(set)  # Maps modules to their dependencies
        self.historical_impact_data = {}  # Maps change patterns to historically affected tests
        
        # Load dependency and coverage data if provided
        if dependency_mapping_path and os.path.exists(dependency_mapping_path):
            self._load_dependency_mapping()
            
        if code_coverage_data_path and os.path.exists(code_coverage_data_path):
            self._load_code_coverage_data()
            
        if test_metadata_path and os.path.exists(test_metadata_path):
            self._load_test_metadata()
            
        logger.info(f"ImpactAnalyzer initialized with min confidence threshold: {min_confidence_threshold}")
    
    def _load_dependency_mapping(self):
        """
        Load dependency mapping data from a file.
        """
        logger.info(f"Loading dependency mapping from {self.dependency_mapping_path}")
        
        try:
            with open(self.dependency_mapping_path, 'r') as f:
                dependency_data = json.load(f)
            
            # Build dependency graph
            for module, dependencies in dependency_data.items():
                self.dependency_graph[module] = set(dependencies)
            
            logger.info(f"Loaded dependency data for {len(self.dependency_graph)} modules")
        
        except Exception as e:
            logger.error(f"Error loading dependency mapping: {str(e)}")
    
    def _load_code_coverage_data(self):
        """
        Load code coverage data to map source files to tests.
        """
        logger.info(f"Loading code coverage data from {self.code_coverage_data_path}")
        
        try:
            with open(self.code_coverage_data_path, 'r') as f:
                coverage_data = json.load(f)
            
            # Build file-to-tests mapping
            for test_id, covered_files in coverage_data.items():
                for file_path in covered_files:
                    if file_path not in self.file_to_tests_map:
                        self.file_to_tests_map[file_path] = set()
                    self.file_to_tests_map[file_path].add(test_id)
            
            logger.info(f"Loaded coverage data mapping {len(self.file_to_tests_map)} files to tests")
        
        except Exception as e:
            logger.error(f"Error loading code coverage data: {str(e)}")
    
    def _load_test_metadata(self):
        """
        Load test metadata to map components to tests.
        """
        logger.info(f"Loading test metadata from {self.test_metadata_path}")
        
        try:
            with open(self.test_metadata_path, 'r') as f:
                test_metadata = json.load(f)
            
            # Build component-to-tests mapping
            for test in test_metadata:
                test_id = test.get('test_id')
                component = test.get('component')
                
                if test_id and component:
                    if component not in self.component_to_tests_map:
                        self.component_to_tests_map[component] = set()
                    self.component_to_tests_map[component].add(test_id)
            
            # Convert sets to lists for serialization
            self.component_to_tests_map = {k: list(v) for k, v in self.component_to_tests_map.items()}
            
            logger.info(f"Loaded test metadata mapping {len(self.component_to_tests_map)} components to tests")
        
        except Exception as e:
            logger.error(f"Error loading test metadata: {str(e)}")
    
    def get_changed_files(
        self,
        base_commit: Optional[str] = None,
        current_commit: str = 'HEAD',
        file_pattern: Optional[str] = None
    ) -> List[str]:
        """
        Get the list of files changed between two commits.
        
        Args:
            base_commit (str, optional): Base commit hash or branch
            current_commit (str): Current commit hash or branch
            file_pattern (str, optional): File pattern to filter changes
            
        Returns:
            list: List of changed file paths
        """
        logger.info(f"Getting changed files between {base_commit or 'default'} and {current_commit}")
        
        try:
            # Change to the git repository directory
            os.chdir(self.git_repo_path)
            
            # Build the git diff command
            if base_commit:
                cmd = ['git', 'diff', '--name-only', base_commit, current_commit]
            else:
                # If no base commit specified, get changes in the working tree
                cmd = ['git', 'diff', '--name-only', 'HEAD']
            
            # Add file pattern if specified
            if file_pattern:
                cmd.append(file_pattern)
            
            # Execute the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error executing git diff: {result.stderr}")
                return []
            
            # Split the output by lines and remove empty lines
            changed_files = [f for f in result.stdout.strip().split('\n') if f]
            
            logger.info(f"Found {len(changed_files)} changed files")
            return changed_files
        
        except Exception as e:
            logger.error(f"Error getting changed files: {str(e)}")
            return []
    
    def get_code_changes(
        self,
        file_path: str,
        base_commit: Optional[str] = None,
        current_commit: str = 'HEAD'
    ) -> Dict[str, Any]:
        """
        Get detailed information about changes in a specific file.
        
        Args:
            file_path (str): Path to the file
            base_commit (str, optional): Base commit hash or branch
            current_commit (str): Current commit hash or branch
            
        Returns:
            dict: Detailed change information
        """
        logger.info(f"Getting code changes for {file_path}")
        
        try:
            # Change to the git repository directory
            os.chdir(self.git_repo_path)
            
            # Build the git diff command
            if base_commit:
                cmd = ['git', 'diff', '-U0', base_commit, current_commit, '--', file_path]
            else:
                cmd = ['git', 'diff', '-U0', 'HEAD', '--', file_path]
            
            # Execute the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error executing git diff for {file_path}: {result.stderr}")
                return {}
            
            # Extract change information
            diff_output = result.stdout
            
            # Parse the diff output to extract changed lines
            added_lines = []
            removed_lines = []
            current_line_number = 0
            
            # Regular expression to match diff chunk headers
            chunk_pattern = re.compile(r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@')
            
            for line in diff_output.split('\n'):
                if line.startswith('@@'):
                    # Extract line numbers from chunk headers
                    match = chunk_pattern.match(line)
                    if match:
                        current_line_number = int(match.group(2))
                        
                elif line.startswith('+') and not line.startswith('+++'):
                    # Added line
                    added_lines.append({
                        'line_number': current_line_number,
                        'content': line[1:]
                    })
                    current_line_number += 1
                    
                elif line.startswith('-') and not line.startswith('---'):
                    # Removed line
                    removed_lines.append({
                        'line_number': current_line_number,
                        'content': line[1:]
                    })
                    
                elif not line.startswith('+') and not line.startswith('-') and not line.startswith('@@'):
                    # Unchanged line
                    current_line_number += 1
            
            # Extract changed functions/methods if possible
            changed_functions = self._extract_changed_functions(file_path, added_lines, removed_lines)
            
            change_info = {
                'file_path': file_path,
                'added_lines': len(added_lines),
                'removed_lines': len(removed_lines),
                'line_details': {
                    'added': added_lines,
                    'removed': removed_lines
                },
                'changed_functions': changed_functions
            }
            
            return change_info
        
        except Exception as e:
            logger.error(f"Error getting code changes for {file_path}: {str(e)}")
            return {}
    
    def _extract_changed_functions(
        self,
        file_path: str,
        added_lines: List[Dict[str, Any]],
        removed_lines: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract names of functions/methods that were changed.
        
        Args:
            file_path (str): Path to the file
            added_lines (list): List of added lines
            removed_lines (list): List of removed lines
            
        Returns:
            list: List of changed function/method names
        """
        changed_functions = set()
        
        try:
            # Determine file type based on extension
            _, ext = os.path.splitext(file_path)
            
            # Define patterns for function/method definitions based on file type
            function_patterns = {
                '.py': r'^\s*def\s+([a-zA-Z0-9_]+)\s*\(',  # Python functions
                '.java': r'^\s*(public|private|protected|static|\s)+[\w\<\>\[\]]+\s+([a-zA-Z0-9_]+)\s*\(',  # Java methods
                '.js': r'^\s*(?:function\s+([a-zA-Z0-9_]+)\s*\(|(?:let|var|const)\s+([a-zA-Z0-9_]+)\s*=\s*function\s*\()',  # JavaScript functions
                '.cpp': r'^\s*[\w\<\>\[\]]+\s+([a-zA-Z0-9_]+)\s*\(',  # C++ functions
                '.c': r'^\s*[\w\<\>\[\]]+\s+([a-zA-Z0-9_]+)\s*\(',  # C functions
            }
            
            # Get appropriate pattern for this file type
            pattern = function_patterns.get(ext.lower())
            
            if not pattern:
                return list(changed_functions)
            
            # Compile the pattern
            function_pattern = re.compile(pattern)
            
            # Extract function names from added and removed lines
            for line_info in added_lines + removed_lines:
                match = function_pattern.match(line_info['content'])
                if match:
                    # Get the function name from the appropriate capture group
                    if len(match.groups()) > 1 and match.group(2):
                        function_name = match.group(2)
                    else:
                        function_name = match.group(1)
                    
                    if function_name:
                        changed_functions.add(function_name)
        
        except Exception as e:
            logger.error(f"Error extracting changed functions: {str(e)}")
        
        return list(changed_functions)
    
    def analyze_code_changes(
        self,
        changed_files: List[str],
        base_commit: Optional[str] = None,
        current_commit: str = 'HEAD'
    ) -> Dict[str, Any]:
        """
        Analyze code changes to understand their nature and scope.
        
        Args:
            changed_files (list): List of changed file paths
            base_commit (str, optional): Base commit hash or branch
            current_commit (str): Current commit hash or branch
            
        Returns:
            dict: Analysis of code changes
        """
        logger.info(f"Analyzing {len(changed_files)} changed files")
        
        analysis_result = {
            'total_files_changed': len(changed_files),
            'files_by_type': defaultdict(int),
            'changed_components': set(),
            'high_risk_changes': [],
            'change_scope': 'unknown',
            'file_details': []
        }
        
        total_lines_added = 0
        total_lines_removed = 0
        
        # Analyze each changed file
        for file_path in changed_files:
            # Get file extension
            _, ext = os.path.splitext(file_path)
            if ext:
                analysis_result['files_by_type'][ext.lower()] += 1
            
            # Get detailed changes
            file_changes = self.get_code_changes(file_path, base_commit, current_commit)
            
            if file_changes:
                # Update line counts
                total_lines_added += file_changes.get('added_lines', 0)
                total_lines_removed += file_changes.get('removed_lines', 0)
                
                # Add to file details
                analysis_result['file_details'].append(file_changes)
                
                # Identify component from file path
                component = self._identify_component_from_path(file_path)
                if component:
                    analysis_result['changed_components'].add(component)
                
                # Check if this is a high-risk change
                if self._is_high_risk_change(file_path, file_changes):
                    analysis_result['high_risk_changes'].append({
                        'file_path': file_path,
                        'reason': 'Core functionality or high-risk area'
                    })
        
        # Update total line counts
        analysis_result['total_lines_added'] = total_lines_added
        analysis_result['total_lines_removed'] = total_lines_removed
        
        # Determine change scope
        if len(changed_files) > 20 or total_lines_added + total_lines_removed > 500:
            analysis_result['change_scope'] = 'large'
        elif len(changed_files) > 5 or total_lines_added + total_lines_removed > 100:
            analysis_result['change_scope'] = 'medium'
        else:
            analysis_result['change_scope'] = 'small'
        
        # Convert sets to lists for serialization
        analysis_result['changed_components'] = list(analysis_result['changed_components'])
        analysis_result['files_by_type'] = dict(analysis_result['files_by_type'])
        
        logger.info(f"Code change analysis completed. Scope: {analysis_result['change_scope']}")
        return analysis_result
    
    def _identify_component_from_path(self, file_path: str) -> Optional[str]:
        """
        Identify which component a file belongs to based on its path.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str: Component name or None if unknown
        """
        # Simple component identification using directory structure
        # This would be more sophisticated in a real implementation
        
        path_parts = file_path.split(os.path.sep)
        
        # Look for common component directory names
        component_indicators = ['api', 'ui', 'db', 'auth', 'core', 'util', 'service']
        
        for part in path_parts:
            if part.lower() in component_indicators:
                return part.lower()
        
        # Check if part of a known component based on path patterns
        common_prefixes = {
            'src/api/': 'api',
            'src/ui/': 'ui',
            'src/db/': 'database',
            'src/auth/': 'authentication',
            'src/core/': 'core'
        }
        
        for prefix, component in common_prefixes.items():
            if file_path.startswith(prefix):
                return component
        
        return None
    
    def _is_high_risk_change(self, file_path: str, change_info: Dict[str, Any]) -> bool:
        """
        Determine if a change is high-risk based on file and change details.
        
        Args:
            file_path (str): Path to the file
            change_info (dict): Change information
            
        Returns:
            bool: True if the change is high-risk, False otherwise
        """
        # Check if this is a core file
        high_risk_paths = [
            'src/core/', 
            'src/auth/', 
            'config/', 
            'src/db/models',
            'src/api/'
        ]
        
        if any(file_path.startswith(path) for path in high_risk_paths):
            return True
        
        # Check for critical operations in changes
        critical_patterns = [
            r'password', 
            r'security', 
            r'authenticate', 
            r'authorize',
            r'delete', 
            r'remove',
            r'drop\s+table',
            r'truncate'
        ]
        
        for line_info in change_info.get('line_details', {}).get('added', []):
            content = line_info.get('content', '')
            if any(re.search(pattern, content, re.IGNORECASE) for pattern in critical_patterns):
                return True
        
        # Check if many functions were changed
        if len(change_info.get('changed_functions', [])) > 5:
            return True
        
        return False
    
    def identify_impacted_tests(
        self,
        changed_files: List[str],
        change_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Identify tests that are impacted by code changes.
        
        Args:
            changed_files (list): List of changed file paths
            change_analysis (dict, optional): Analysis of code changes
            
        Returns:
            dict: Impacted tests with confidence scores
        """
        logger.info(f"Identifying tests impacted by changes to {len(changed_files)} files")
        
        impacted_tests = {}
        impacted_test_ids = set()
        test_confidence = {}
        
        # Use coverage data to identify affected tests
        if self.file_to_tests_map:
            for file_path in changed_files:
                # Normalize path to match coverage data format
                normalized_path = self._normalize_path(file_path)
                
                # Find tests that cover this file
                tests_for_file = self.file_to_tests_map.get(normalized_path, [])
                
                # Add to impacted tests with high confidence
                for test_id in tests_for_file:
                    impacted_test_ids.add(test_id)
                    test_confidence[test_id] = max(test_confidence.get(test_id, 0.0), 0.9)
                    
                    # Record the reason why this test is affected
                    if test_id not in impacted_tests:
                        impacted_tests[test_id] = {
                            'confidence': 0.9,
                            'reason': f"Direct coverage of changed file: {file_path}",
                            'changed_files': [file_path]
                        }
                    else:
                        impacted_tests[test_id]['changed_files'].append(file_path)
        
        # Use component mapping to identify affected tests
        if self.component_to_tests_map and change_analysis and 'changed_components' in change_analysis:
            for component in change_analysis['changed_components']:
                tests_for_component = self.component_to_tests_map.get(component, [])
                
                # Add to impacted tests with medium confidence
                for test_id in tests_for_component:
                    if test_id not in impacted_test_ids:
                        impacted_test_ids.add(test_id)
                        test_confidence[test_id] = max(test_confidence.get(test_id, 0.0), 0.7)
                        
                        # Record the reason
                        impacted_tests[test_id] = {
                            'confidence': 0.7,
                            'reason': f"Component affected: {component}",
                            'changed_files': []
                        }
        
        # Use dependency graph to identify affected tests
        if self.dependency_graph:
            # Identify changed modules
            changed_modules = set()
            for file_path in changed_files:
                module = self._file_path_to_module(file_path)
                if module:
                    changed_modules.add(module)
            
            # Find modules that depend on changed modules
            affected_modules = set(changed_modules)
            for module in changed_modules:
                for dependent, dependencies in self.dependency_graph.items():
                    if module in dependencies and dependent not in affected_modules:
                        affected_modules.add(dependent)
            
            # Map affected modules to tests
            for module in affected_modules:
                module_files = self._module_to_file_paths(module)
                for file_path in module_files:
                    tests_for_file = self.file_to_tests_map.get(file_path, [])
                    
                    # Add to impacted tests with medium confidence
                    for test_id in tests_for_file:
                        if test_id not in impacted_test_ids:
                            impacted_test_ids.add(test_id)
                            test_confidence[test_id] = max(test_confidence.get(test_id, 0.0), 0.6)
                            
                            # Record the reason
                            impacted_tests[test_id] = {
                                'confidence': 0.6,
                                'reason': f"Dependency affected: {module}",
                                'changed_files': []
                            }
        
        # Update confidence scores
        for test_id in impacted_tests:
            impacted_tests[test_id]['confidence'] = test_confidence[test_id]
        
        # Filter out tests below the confidence threshold
        filtered_impacted_tests = {
            test_id: info for test_id, info in impacted_tests.items()
            if info['confidence'] >= self.min_confidence_threshold
        }
        
        logger.info(f"Identified {len(filtered_impacted_tests)} impacted tests with confidence >= {self.min_confidence_threshold}")
        
        return {
            'impacted_tests': filtered_impacted_tests,
            'total_impacted': len(filtered_impacted_tests),
            'all_potentially_impacted': len(impacted_tests),
            'confidence_threshold': self.min_confidence_threshold
        }
    
    def _normalize_path(self, file_path: str) -> str:
        """
        Normalize a file path to match the format in coverage data.
        
        Args:
            file_path (str): Original file path
            
        Returns:
            str: Normalized file path
        """
        # This implementation depends on how paths are stored in coverage data
        # For example, convert absolute paths to relative or vice versa
        
        # Simple implementation: ensure the path uses forward slashes
        normalized = file_path.replace('\\', '/')
        
        # Remove leading './' if present
        if normalized.startswith('./'):
            normalized = normalized[2:]
        
        return normalized
    
    def _file_path_to_module(self, file_path: str) -> Optional[str]:
        """
        Convert a file path to a module name.
        
        Args:
            file_path (str): File path
            
        Returns:
            str: Module name or None if conversion fails
        """
        # Simple implementation for Python files
        if file_path.endswith('.py'):
            # Remove extension
            module_path = file_path[:-3]
            
            # Convert path separators to dots
            module = module_path.replace('/', '.').replace('\\', '.')
            
            # Remove leading dots
            if module.startswith('.'):
                module = module[1:]
            
            return module
        
        return None
    
    def _module_to_file_paths(self, module: str) -> List[str]:
        """
        Convert a module name to possible file paths.
        
        Args:
            module (str): Module name
            
        Returns:
            list: List of possible file paths
        """
        # Convert dots to path separators
        path = module.replace('.', '/')
        
        # Add possible extensions
        paths = [f"{path}.py", f"{path}/__init__.py"]
        
        return paths
    
    def generate_test_selection_plan(
        self,
        test_metadata: List[Dict[str, Any]],
        impacted_tests: Dict[str, Any],
        time_budget_minutes: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a test selection plan based on impacted tests and constraints.
        
        Args:
            test_metadata (list): List of test metadata dictionaries
            impacted_tests (dict): Dictionary with impacted test information
            time_budget_minutes (float, optional): Time budget for test execution in minutes
            
        Returns:
            dict: Test selection plan
        """
        logger.info("Generating test selection plan")
        
        # Create a mapping from test_id to metadata
        test_map = {test['test_id']: test for test in test_metadata if 'test_id' in test}
        
        # Get the list of impacted test IDs with their confidence scores
        impacted_test_list = [
            {
                'test_id': test_id,
                'confidence': info['confidence'],
                'reason': info.get('reason', 'Unknown'),
                'execution_time': test_map.get(test_id, {}).get('execution_time', 1.0)
            }
            for test_id, info in impacted_tests.get('impacted_tests', {}).items()
        ]
        
        # Sort tests by confidence (highest first)
        impacted_test_list.sort(key=lambda t: t['confidence'], reverse=True)
        
        # Calculate total execution time for impacted tests
        total_execution_time = sum(test.get('execution_time', 1.0) for test in impacted_test_list)
        
        selection_plan = {
            'total_candidate_tests': len(impacted_test_list),
            'total_execution_time_minutes': total_execution_time / 60,
            'selected_tests': impacted_test_list,
            'selection_strategy': 'confidence_based'
        }
        
        # Apply time budget if specified
        if time_budget_minutes and total_execution_time / 60 > time_budget_minutes:
            logger.info(f"Optimizing test selection for time budget of {time_budget_minutes} minutes")
            
            # Initialize selection variables
            selected_tests = []
            current_time = 0
            
            # Add tests in order of confidence until budget is reached
            for test in impacted_test_list:
                test_time = test.get('execution_time', 1.0)
                
                if (current_time + test_time) / 60 <= time_budget_minutes:
                    selected_tests.append(test)
                    current_time += test_time
                else:
                    # If this test would exceed the budget, consider skipping it
                    # unless it has very high confidence
                    if test['confidence'] > 0.9:
                        selected_tests.append(test)
                        current_time += test_time
            
            # Update selection plan
            selection_plan['selected_tests'] = selected_tests
            selection_plan['total_execution_time_minutes'] = current_time / 60
            selection_plan['selection_strategy'] = 'time_constrained'
            selection_plan['time_budget_minutes'] = time_budget_minutes
            selection_plan['coverage_percentage'] = len(selected_tests) / len(impacted_test_list) * 100
        
        logger.info(f"Test selection plan generated with {len(selection_plan['selected_tests'])} tests")
        return selection_plan
    
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
        logger.info(f"Saving impact analysis results to {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(analysis_result, f, indent=2)
        
        logger.info(f"Impact analysis results saved to {output_path}")
        return output_path

def load_test_metadata(file_path: str) -> List[Dict[str, Any]]:
    """
    Load test metadata from a file.
    
    Args:
        file_path (str): Path to the test metadata file
        
    Returns:
        list: List of test metadata dictionaries
    """
    logger.info(f"Loading test metadata from {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, list):
            metadata = data
        elif isinstance(data, dict) and 'tests' in data:
            metadata = data['tests']
        else:
            logger.error("Unexpected format for test metadata file")
            return []
        
        logger.info(f"Loaded metadata for {len(metadata)} tests")
        return metadata
    
    except Exception as e:
        logger.error(f"Error loading test metadata: {str(e)}")
        return []

def main():
    """
    Main function to demonstrate the impact analyzer.
    """
    # Set up paths
    code_coverage_path = os.path.join(config.DATA_DIR, 'coverage', 'coverage_mapping.json')
    dependency_path = os.path.join(config.DATA_DIR, 'dependencies', 'module_dependencies.json')
    test_metadata_path = os.path.join(config.DATA_DIR, 'metadata', 'test_metadata.json')
    git_repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(config.REPORTS_DIR, 'impact_analysis')
    
    # Create an impact analyzer
    analyzer = ImpactAnalyzer(
        code_coverage_data_path=code_coverage_path,
        dependency_mapping_path=dependency_path,
        git_repo_path=git_repo_path,
        test_metadata_path=test_metadata_path,
        min_confidence_threshold=0.5
    )
    
    # Get changed files
    changed_files = analyzer.get_changed_files()
    if not changed_files:
        print("No changed files detected. Please specify commits to compare.")
        return
    
    # Analyze code changes
    change_analysis = analyzer.analyze_code_changes(changed_files)
    
    # Identify impacted tests
    impacted_tests = analyzer.identify_impacted_tests(changed_files, change_analysis)
    
    # Load test metadata
    test_metadata = load_test_metadata(test_metadata_path)
    
    # Generate test selection plan
    selection_plan = analyzer.generate_test_selection_plan(
        test_metadata, 
        impacted_tests,
        time_budget_minutes=30  # Example time budget
    )
    
    # Combine results
    analysis_result = {
        'timestamp': datetime.now().isoformat(),
        'code_changes': change_analysis,
        'impacted_tests': impacted_tests,
        'test_selection_plan': selection_plan
    }
    
    # Save analysis results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f"impact_analysis_{timestamp}.json")
    analyzer.save_analysis_result(analysis_result, output_path)
    
    # Print summary
    print("\nImpact Analysis Summary:")
    print(f"- Changed Files: {len(changed_files)}")
    print(f"- Impacted Tests: {impacted_tests['total_impacted']} (of {impacted_tests['all_potentially_impacted']} potential impacts)")
    print(f"- Selected Tests: {len(selection_plan['selected_tests'])}")
    print(f"- Estimated Execution Time: {selection_plan['total_execution_time_minutes']:.2f} minutes")
    print(f"- Results saved to: {output_path}")

if __name__ == "__main__":
    main() 