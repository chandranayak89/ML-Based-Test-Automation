"""
Module for cleaning and preprocessing test data for ML model training.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"data_preprocessing_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_test_data(df):
    """
    Clean and prepare test data by handling missing values, outliers, and data type conversions.
    
    Args:
        df (pd.DataFrame): Raw test execution data.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logger.info(f"Cleaning test data with {len(df)} records")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for cleaning")
        return df
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Convert timestamp strings to datetime objects
    for col in ['start_time', 'end_time']:
        if col in cleaned_df.columns:
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col])
            except Exception as e:
                logger.warning(f"Could not convert {col} to datetime: {str(e)}")
    
    # Create a binary target column for test result
    if 'result' in cleaned_df.columns:
        cleaned_df['test_passed'] = (cleaned_df['result'] == 'passed').astype(int)
    
    # Handle missing values
    for col in ['execution_time', 'error_count']:
        if col in cleaned_df.columns:
            # Fill missing numeric values with column median
            median_value = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_value)
    
    # Handle missing categorical values
    for col in ['suite_name', 'component', 'priority']:
        if col in cleaned_df.columns:
            # Fill missing categorical values with the most common value
            most_common = cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else "unknown"
            cleaned_df[col] = cleaned_df[col].fillna(most_common)
    
    # Remove outliers in execution time (values beyond 3 standard deviations)
    if 'execution_time' in cleaned_df.columns:
        mean = cleaned_df['execution_time'].mean()
        std = cleaned_df['execution_time'].std()
        min_val = max(0, mean - 3 * std)  # Ensure non-negative
        max_val = mean + 3 * std
        
        # Cap outliers rather than removing
        cleaned_df['execution_time_original'] = cleaned_df['execution_time']
        cleaned_df['execution_time'] = cleaned_df['execution_time'].clip(min_val, max_val)
        
        outlier_count = (cleaned_df['execution_time_original'] != cleaned_df['execution_time']).sum()
        logger.info(f"Capped {outlier_count} outliers in execution_time")
    
    # Extract additional time-related features
    if 'start_time' in cleaned_df.columns and pd.api.types.is_datetime64_dtype(cleaned_df['start_time']):
        cleaned_df['hour_of_day'] = cleaned_df['start_time'].dt.hour
        cleaned_df['day_of_week'] = cleaned_df['start_time'].dt.dayofweek
        cleaned_df['is_weekend'] = (cleaned_df['day_of_week'] >= 5).astype(int)
    
    # Drop any duplicate rows
    original_len = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    logger.info(f"Removed {original_len - len(cleaned_df)} duplicate rows")
    
    return cleaned_df

def engineer_features(df):
    """
    Engineer additional features from test execution data.
    
    Args:
        df (pd.DataFrame): Cleaned test execution data.
    
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    logger.info(f"Engineering features for {len(df)} records")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for feature engineering")
        return df
    
    # Create a copy to avoid modifying the original
    featured_df = df.copy()
    
    # Calculate test complexity based on execution time and error count
    if 'execution_time' in featured_df.columns and 'error_count' in featured_df.columns:
        execution_time_norm = (featured_df['execution_time'] - featured_df['execution_time'].min()) / \
                             (featured_df['execution_time'].max() - featured_df['execution_time'].min())
        
        error_count_norm = featured_df['error_count'] / featured_df['error_count'].max() if featured_df['error_count'].max() > 0 else 0
        
        # Weighted sum of normalized values
        featured_df['complexity_score'] = (0.7 * execution_time_norm + 0.3 * error_count_norm).fillna(0)
    
    # Extract error patterns if available
    if 'errors' in featured_df.columns:
        # Flag for common error types
        featured_df['has_timeout_error'] = featured_df['errors'].str.contains('timeout|Timeout', na=False, case=False).astype(int)
        featured_df['has_connection_error'] = featured_df['errors'].str.contains('connection|Connection', na=False, case=False).astype(int)
        featured_df['has_assertion_error'] = featured_df['errors'].str.contains('assert|Assert', na=False, case=False).astype(int)
        featured_df['has_memory_error'] = featured_df['errors'].str.contains('memory|Memory', na=False, case=False).astype(int)
    
    # Create features for test execution history
    if 'test_name' in featured_df.columns and 'start_time' in featured_df.columns:
        # Sort by test name and time
        featured_df = featured_df.sort_values(['test_name', 'start_time'])
        
        # Group by test name and calculate historical metrics
        test_stats = featured_df.groupby('test_name').agg({
            'test_passed': ['mean', 'std', 'count'],
            'execution_time': ['mean', 'std']
        })
        
        # Flatten the column hierarchy
        test_stats.columns = ['_'.join(col).strip() for col in test_stats.columns.values]
        test_stats = test_stats.reset_index()
        
        # Rename for clarity
        test_stats = test_stats.rename(columns={
            'test_passed_mean': 'historical_pass_rate',
            'test_passed_std': 'historical_pass_volatility',
            'test_passed_count': 'execution_count',
            'execution_time_mean': 'average_execution_time',
            'execution_time_std': 'execution_time_volatility'
        })
        
        # Merge back with original data
        featured_df = pd.merge(featured_df, test_stats, on='test_name', how='left')
        
        # Calculate relative execution time compared to historical average
        featured_df['relative_execution_time'] = featured_df['execution_time'] / featured_df['average_execution_time']
        featured_df['relative_execution_time'] = featured_df['relative_execution_time'].fillna(1.0)
    
    # Create priority level numerical mapping
    if 'priority' in featured_df.columns:
        priority_map = {
            'critical': 1,
            'high': 2,
            'medium': 3,
            'low': 4
        }
        featured_df['priority_level'] = featured_df['priority'].map(priority_map).fillna(3)
    
    return featured_df

def encode_categorical_features(df, categorical_cols=None):
    """
    Encode categorical features for ML model consumption.
    
    Args:
        df (pd.DataFrame): DataFrame with features.
        categorical_cols (list, optional): List of categorical columns to encode.
                                          If None, will use default columns.
    
    Returns:
        pd.DataFrame: DataFrame with encoded features.
        dict: Dictionary of fitted encoders for future use.
    """
    logger.info("Encoding categorical features")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for encoding")
        return df, {}
    
    # Define categorical columns if not provided
    if categorical_cols is None:
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col not in 
                           ['test_name', 'errors', 'file_name', 'file_path', 'start_time', 'end_time', 'raw_log']]
    
    logger.info(f"Encoding {len(categorical_cols)} categorical columns: {', '.join(categorical_cols)}")
    
    # Create a copy to avoid modifying the original
    encoded_df = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        if col in encoded_df.columns:
            # Fill nulls with placeholder
            encoded_df[col] = encoded_df[col].fillna('unknown')
            
            # Create one-hot encoder
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_values = encoder.fit_transform(encoded_df[[col]])
            
            # Create new column names
            categories = encoder.categories_[0].tolist()
            encoded_col_names = [f"{col}_{cat}" for cat in categories]
            
            # Create DataFrame with encoded values
            encoded_cols_df = pd.DataFrame(encoded_values, columns=encoded_col_names, index=encoded_df.index)
            
            # Concatenate with main DataFrame
            encoded_df = pd.concat([encoded_df, encoded_cols_df], axis=1)
            
            # Store encoder for future use
            encoders[col] = encoder
            
            # Drop original column
            encoded_df = encoded_df.drop(col, axis=1)
    
    return encoded_df, encoders

def prepare_data_for_modeling(df, target_col='test_passed'):
    """
    Prepare data for ML modeling by scaling numerical features and splitting into features and target.
    
    Args:
        df (pd.DataFrame): DataFrame with all features.
        target_col (str): Name of the target column.
    
    Returns:
        tuple: (X, y, scaler) where X is the feature matrix, y is the target vector,
               and scaler is the fitted StandardScaler for future use.
    """
    logger.info("Preparing data for modeling")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for modeling preparation")
        return None, None, None
    
    # Identify numerical columns (exclude object, datetime, and the target)
    numerical_cols = [col for col in df.columns 
                     if df[col].dtype in ['int64', 'float64'] 
                     and col != target_col
                     and col not in ['test_id']]
    
    logger.info(f"Scaling {len(numerical_cols)} numerical features")
    
    # Create X (features) and y (target)
    X = df.drop([col for col in [target_col, 'test_name', 'errors', 'file_name', 'file_path', 'start_time', 'end_time', 'raw_log'] 
                if col in df.columns], axis=1)
    
    # Handle missing values in numerical features
    if numerical_cols:
        imputer = SimpleImputer(strategy='median')
        X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
        
        # Scale numerical features
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    else:
        scaler = None
    
    # Extract target if available
    if target_col in df.columns:
        y = df[target_col]
    else:
        y = None
        logger.warning(f"Target column '{target_col}' not found in DataFrame")
    
    return X, y, scaler

def main(input_file=None):
    """
    Main function to preprocess test data.
    
    Args:
        input_file (str, optional): Path to the input CSV file. If None, will use the latest file in processed_data_dir.
    
    Returns:
        str: Path to the output file.
    """
    try:
        logger.info("Starting data preprocessing")
        
        # Find input file if not provided
        if input_file is None:
            processed_files = [f for f in os.listdir(config.PROCESSED_DATA_DIR) 
                              if f.endswith('.csv') and f.startswith('test_data_')]
            
            if not processed_files:
                logger.error("No processed data files found")
                return None
            
            # Sort by file name (which contains timestamp)
            processed_files.sort(reverse=True)
            input_file = os.path.join(config.PROCESSED_DATA_DIR, processed_files[0])
        
        logger.info(f"Loading data from: {input_file}")
        df = pd.read_csv(input_file)
        
        if df.empty:
            logger.warning("Input file contains no data")
            return None
        
        # Clean data
        cleaned_df = clean_test_data(df)
        
        # Engineer features
        featured_df = engineer_features(cleaned_df)
        
        # Encode categorical features
        encoded_df, encoders = encode_categorical_features(featured_df)
        
        # Prepare for modeling
        X, y, scaler = prepare_data_for_modeling(encoded_df)
        
        # Save processed data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(config.INTERIM_DATA_DIR, f"model_ready_data_{timestamp}.csv")
        
        # Save the full processed dataset
        encoded_df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to: {output_file}")
        
        # Also save X and y separately for convenience
        X_file = os.path.join(config.INTERIM_DATA_DIR, f"X_data_{timestamp}.csv")
        y_file = os.path.join(config.INTERIM_DATA_DIR, f"y_data_{timestamp}.csv")
        
        X.to_csv(X_file, index=False)
        if y is not None:
            pd.DataFrame(y, columns=[y.name]).to_csv(y_file, index=False)
        
        logger.info(f"Preprocessing complete. Output files: {output_file}, {X_file}, {y_file}")
        
        return output_file
        
    except Exception as e:
        logger.exception(f"Error during data preprocessing: {str(e)}")
        return None

if __name__ == "__main__":
    main() 