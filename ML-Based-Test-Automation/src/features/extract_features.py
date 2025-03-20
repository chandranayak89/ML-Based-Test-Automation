"""
Module for extracting advanced features from test execution data.
This module goes beyond basic preprocessing to create specialized features
for test failure prediction and optimization.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.data.preprocess_data import clean_test_data

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"feature_extraction_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_time_based_features(df):
    """
    Extract time-based features from test execution data.
    
    Args:
        df (pd.DataFrame): DataFrame containing test execution data.
        
    Returns:
        pd.DataFrame: DataFrame with additional time-based features.
    """
    logger.info("Extracting time-based features")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for time-based feature extraction")
        return df
    
    result_df = df.copy()
    
    # Ensure we have datetime columns
    for col in ['start_time', 'end_time']:
        if col in result_df.columns and not pd.api.types.is_datetime64_dtype(result_df[col]):
            try:
                result_df[col] = pd.to_datetime(result_df[col])
            except Exception as e:
                logger.warning(f"Could not convert {col} to datetime: {str(e)}")
    
    # Time-based features
    if 'start_time' in result_df.columns and pd.api.types.is_datetime64_dtype(result_df['start_time']):
        # Time of day features
        result_df['hour_of_day'] = result_df['start_time'].dt.hour
        result_df['is_business_hours'] = ((result_df['hour_of_day'] >= 9) & 
                                          (result_df['hour_of_day'] < 17)).astype(int)
        
        # Day of week features
        result_df['day_of_week'] = result_df['start_time'].dt.dayofweek
        result_df['is_weekend'] = (result_df['day_of_week'] >= 5).astype(int)
        
        # Month and quarter features
        result_df['month'] = result_df['start_time'].dt.month
        result_df['quarter'] = result_df['start_time'].dt.quarter
        
        # Execution time of day bins
        result_df['time_of_day'] = pd.cut(
            result_df['hour_of_day'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        # Create dummy variables for time of day
        time_dummies = pd.get_dummies(result_df['time_of_day'], prefix='time')
        result_df = pd.concat([result_df, time_dummies], axis=1)
        
    return result_df

def extract_error_pattern_features(df):
    """
    Extract features from error messages and patterns.
    
    Args:
        df (pd.DataFrame): DataFrame containing test execution data.
        
    Returns:
        pd.DataFrame: DataFrame with additional error pattern features.
    """
    logger.info("Extracting error pattern features")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for error pattern feature extraction")
        return df
    
    result_df = df.copy()
    
    if 'errors' in result_df.columns:
        # Common error types
        error_patterns = {
            'timeout': r'timeout|timed out|Timeout',
            'connection': r'connect|Connection|network|Network',
            'authentication': r'auth|login|credential|permission|Access denied',
            'assertion': r'assert|Assert|expect|Expect|should be|should have',
            'null_reference': r'null|undefined|None|NullPointer',
            'database': r'database|query|sql|table|DB|Database',
            'memory': r'memory|heap|stack|overflow|out of memory',
            'file_system': r'file|path|directory|io|IO|File|Path',
            'syntax': r'syntax|invalid|malformed|parse',
            'configuration': r'config|setting|property|environment'
        }
        
        # Create binary indicators for each error pattern
        for error_type, pattern in error_patterns.items():
            col_name = f'has_{error_type}_error'
            result_df[col_name] = result_df['errors'].str.contains(
                pattern, case=False, na=False, regex=True
            ).astype(int)
        
        # Count the number of distinct error types
        result_df['distinct_error_types'] = result_df[[f'has_{error_type}_error' 
                                                      for error_type in error_patterns.keys()]].sum(axis=1)
        
        # Extract error severity indicators
        result_df['has_critical_error'] = result_df['errors'].str.contains(
            r'critical|severe|fatal|crash|exception', case=False, na=False, regex=True
        ).astype(int)
        
        # Extract numeric values from errors (like error codes)
        def extract_error_codes(error_str):
            if pd.isna(error_str):
                return np.nan
            matches = re.findall(r'(\d{3,4})', error_str)
            if matches:
                return int(matches[0])
            return np.nan
        
        result_df['error_code'] = result_df['errors'].apply(extract_error_codes)
    
    return result_df

def extract_test_metadata_features(df):
    """
    Extract features from test metadata.
    
    Args:
        df (pd.DataFrame): DataFrame containing test execution data with metadata.
        
    Returns:
        pd.DataFrame: DataFrame with additional metadata-based features.
    """
    logger.info("Extracting test metadata features")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for metadata feature extraction")
        return df
    
    result_df = df.copy()
    
    # Test age feature (if last_modified_date is available)
    if 'last_modified_date' in result_df.columns:
        try:
            result_df['last_modified_date'] = pd.to_datetime(result_df['last_modified_date'])
            
            if 'start_time' in result_df.columns and pd.api.types.is_datetime64_dtype(result_df['start_time']):
                # Calculate days since last modification
                result_df['days_since_modified'] = (result_df['start_time'] - result_df['last_modified_date']).dt.days
                
                # Create buckets for test age
                result_df['test_age_category'] = pd.cut(
                    result_df['days_since_modified'],
                    bins=[-1, 7, 30, 90, float('inf')],
                    labels=['very_recent', 'recent', 'moderate', 'old'],
                    include_lowest=True
                )
                
                # Create dummy variables for test age
                age_dummies = pd.get_dummies(result_df['test_age_category'], prefix='age')
                result_df = pd.concat([result_df, age_dummies], axis=1)
        except Exception as e:
            logger.warning(f"Could not process last_modified_date: {str(e)}")
    
    # Convert priority to numerical values if it's categorical
    if 'priority' in result_df.columns and result_df['priority'].dtype == 'object':
        priority_map = {
            'critical': 1,
            'high': 2,
            'medium': 3,
            'low': 4
        }
        result_df['priority_level'] = result_df['priority'].map(priority_map).fillna(3)
    
    # Create component-based features
    if 'component' in result_df.columns:
        # Create dummy variables for component
        component_dummies = pd.get_dummies(result_df['component'], prefix='component')
        result_df = pd.concat([result_df, component_dummies], axis=1)
    
    # Create suite-based features
    if 'suite_name' in result_df.columns:
        # Count tests per suite
        suite_counts = result_df['suite_name'].value_counts().to_dict()
        result_df['tests_in_suite'] = result_df['suite_name'].map(suite_counts)
        
        # Create dummy variables for test suite
        suite_dummies = pd.get_dummies(result_df['suite_name'], prefix='suite')
        result_df = pd.concat([result_df, suite_dummies], axis=1)
    
    return result_df

def extract_historical_performance_features(df):
    """
    Extract features based on historical test performance.
    
    Args:
        df (pd.DataFrame): DataFrame containing test execution data.
        
    Returns:
        pd.DataFrame: DataFrame with additional historical performance features.
    """
    logger.info("Extracting historical performance features")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for historical performance feature extraction")
        return df
    
    result_df = df.copy()
    
    # Make sure we have a datetime column for sorting
    if 'start_time' in result_df.columns and not pd.api.types.is_datetime64_dtype(result_df['start_time']):
        try:
            result_df['start_time'] = pd.to_datetime(result_df['start_time'])
        except Exception as e:
            logger.warning(f"Could not convert start_time to datetime: {str(e)}")
    
    # Sort by test name and time
    if 'test_name' in result_df.columns and 'start_time' in result_df.columns:
        result_df = result_df.sort_values(['test_name', 'start_time'])
        
        # Group by test name to calculate historical statistics
        historical_stats = result_df.groupby('test_name').agg({
            'execution_time': ['mean', 'std', 'min', 'max', 'count'],
            'result': lambda x: (x == 'passed').mean()  # Pass rate
        })
        
        # Flatten the column hierarchy
        historical_stats.columns = ['_'.join(col).strip() for col in historical_stats.columns.values]
        historical_stats = historical_stats.reset_index()
        
        # Rename the columns for clarity
        historical_stats = historical_stats.rename(columns={
            'execution_time_mean': 'avg_historical_execution_time',
            'execution_time_std': 'std_historical_execution_time',
            'execution_time_min': 'min_historical_execution_time',
            'execution_time_max': 'max_historical_execution_time',
            'execution_time_count': 'historical_execution_count',
            'result_<lambda>': 'historical_pass_rate'
        })
        
        # Merge with the original DataFrame
        result_df = pd.merge(result_df, historical_stats, on='test_name', how='left')
        
        # Calculate execution time deviation from historical mean
        result_df['execution_time_deviation'] = (
            (result_df['execution_time'] - result_df['avg_historical_execution_time']) / 
            result_df['avg_historical_execution_time']
        ).fillna(0)
        
        # Create relative performance indicators
        result_df['is_slower_than_average'] = (
            result_df['execution_time'] > result_df['avg_historical_execution_time']
        ).astype(int)
        
        # Calculate z-score of execution time
        result_df['execution_time_zscore'] = (
            (result_df['execution_time'] - result_df['avg_historical_execution_time']) / 
            result_df['std_historical_execution_time']
        ).fillna(0)
        
        # Flag tests with unusual execution times
        result_df['unusual_execution_time'] = (
            result_df['execution_time_zscore'].abs() > 2
        ).astype(int)
    
    return result_df

def extract_execution_context_features(df):
    """
    Extract features from test execution context.
    
    Args:
        df (pd.DataFrame): DataFrame containing test execution data.
        
    Returns:
        pd.DataFrame: DataFrame with additional execution context features.
    """
    logger.info("Extracting execution context features")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for execution context feature extraction")
        return df
    
    result_df = df.copy()
    
    # Create features based on sequential test execution
    if 'start_time' in result_df.columns and pd.api.types.is_datetime64_dtype(result_df['start_time']):
        # Sort by start time
        result_df = result_df.sort_values('start_time')
        
        # Calculate time since previous test execution
        result_df['time_since_previous_test'] = result_df['start_time'].diff().dt.total_seconds().fillna(0)
        
        # Flag tests that started immediately after another (potential dependency)
        result_df['quick_follow_up_test'] = (result_df['time_since_previous_test'] < 2).astype(int)
    
    # Group by test file to create file-level features
    if 'file_name' in result_df.columns:
        # Calculate stats per file
        file_stats = result_df.groupby('file_name').agg({
            'execution_time': 'sum',
            'result': lambda x: (x == 'passed').mean()
        }).reset_index()
        
        file_stats.columns = ['file_name', 'total_file_execution_time', 'file_pass_rate']
        
        # Merge back with the main DataFrame
        result_df = pd.merge(result_df, file_stats, on='file_name', how='left')
        
        # Calculate test's proportion of file execution time
        result_df['proportion_of_file_time'] = (
            result_df['execution_time'] / result_df['total_file_execution_time']
        ).fillna(0)
    
    # Create test complexity indicator
    if 'execution_time' in result_df.columns and 'error_count' in result_df.columns:
        # Normalize execution time and error count
        max_time = result_df['execution_time'].max()
        max_errors = result_df['error_count'].max() if result_df['error_count'].max() > 0 else 1
        
        normalized_time = result_df['execution_time'] / max_time if max_time > 0 else 0
        normalized_errors = result_df['error_count'] / max_errors
        
        # Calculate complexity score (weighted sum)
        result_df['test_complexity_score'] = (0.7 * normalized_time + 0.3 * normalized_errors).fillna(0)
        
        # Categorize complexity
        result_df['complexity_category'] = pd.cut(
            result_df['test_complexity_score'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['low', 'medium', 'high', 'very_high'],
            include_lowest=True
        )
        
        # Create dummy variables for complexity
        complexity_dummies = pd.get_dummies(result_df['complexity_category'], prefix='complexity')
        result_df = pd.concat([result_df, complexity_dummies], axis=1)
    
    return result_df

def extract_all_features(df):
    """
    Apply all feature extraction functions to create a comprehensive set of features.
    
    Args:
        df (pd.DataFrame): Raw or preprocessed test execution data.
        
    Returns:
        pd.DataFrame: DataFrame with all extracted features.
    """
    logger.info(f"Extracting all features from dataset with {len(df)} records")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for feature extraction")
        return df
    
    # First, ensure the data is cleaned
    clean_df = clean_test_data(df)
    
    # Apply each feature extraction function
    with_time_features = extract_time_based_features(clean_df)
    with_error_features = extract_error_pattern_features(with_time_features)
    with_metadata_features = extract_test_metadata_features(with_error_features)
    with_historical_features = extract_historical_performance_features(with_metadata_features)
    result_df = extract_execution_context_features(with_historical_features)
    
    logger.info(f"Feature extraction complete. Original columns: {len(df.columns)}, "
                f"New columns: {len(result_df.columns)}")
    
    return result_df

def main(input_file=None, output_file=None):
    """
    Main function to extract features from test data.
    
    Args:
        input_file (str, optional): Path to the input CSV file. If None, will use 
                                     the latest processed file.
        output_file (str, optional): Path to save the output CSV file. If None, will 
                                     generate a default name.
    
    Returns:
        str: Path to the output file.
    """
    try:
        logger.info("Starting feature extraction")
        
        # Find input file if not provided
        if input_file is None:
            # Look in processed directory first
            processed_files = [f for f in os.listdir(config.PROCESSED_DATA_DIR) 
                              if f.endswith('.csv') and f.startswith('test_data_')]
            
            if processed_files:
                # Sort by file name (which contains timestamp)
                processed_files.sort(reverse=True)
                input_file = os.path.join(config.PROCESSED_DATA_DIR, processed_files[0])
            else:
                # If no processed files, look in sample directory
                sample_dir = os.path.join(config.DATA_DIR, 'sample')
                if os.path.exists(os.path.join(sample_dir, 'test_metadata.csv')):
                    input_file = os.path.join(sample_dir, 'test_metadata.csv')
                else:
                    logger.error("No input files found")
                    return None
        
        logger.info(f"Loading data from: {input_file}")
        df = pd.read_csv(input_file)
        
        if df.empty:
            logger.warning("Input file contains no data")
            return None
        
        # Extract features
        featured_df = extract_all_features(df)
        
        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(config.INTERIM_DATA_DIR, f"features_{timestamp}.csv")
        
        featured_df.to_csv(output_file, index=False)
        logger.info(f"Saved featured data to: {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.exception(f"Error during feature extraction: {str(e)}")
        return None

if __name__ == "__main__":
    main() 