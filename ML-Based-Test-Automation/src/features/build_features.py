"""
Main feature engineering module that orchestrates the extraction and selection
of features for test failure prediction models.
"""
import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.features.extract_features import extract_all_features
from src.features.select_features import select_features

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"feature_engineering_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def build_features(input_data, target_col='test_passed', n_features=20, 
                   extraction_methods=None, selection_methods=None):
    """
    Build features for test failure prediction from raw or preprocessed data.
    
    Args:
        input_data (pd.DataFrame or str): Input data as DataFrame or path to CSV file.
        target_col (str): Name of the target column.
        n_features (int): Number of features to select.
        extraction_methods (list, optional): List of feature extraction methods to use.
        selection_methods (list, optional): List of feature selection methods to use.
    
    Returns:
        pd.DataFrame: DataFrame with selected engineered features.
        dict: Dictionary of feature importances.
    """
    logger.info("Starting feature engineering process")
    
    # Set default methods if not provided
    if selection_methods is None:
        selection_methods = ['low_variance', 'correlation', 'importance']
    
    # Load data if a file path is provided
    if isinstance(input_data, str):
        logger.info(f"Loading data from {input_data}")
        try:
            data = pd.read_csv(input_data)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame(), {}
    else:
        data = input_data
    
    # Check if data is empty
    if data.empty:
        logger.warning("Empty data provided for feature engineering")
        return pd.DataFrame(), {}
    
    # 1. Extract all features
    logger.info("Extracting features")
    extracted_features = extract_all_features(data)
    
    # 2. Select the most important features
    logger.info(f"Selecting top {n_features} features")
    selected_features_df, importances = select_features(
        extracted_features, 
        target_col=target_col,
        methods=selection_methods,
        n_features=n_features
    )
    
    # Log the shape of the resulting DataFrame
    logger.info(f"Feature engineering complete. Input shape: {data.shape}, Output shape: {selected_features_df.shape}")
    
    return selected_features_df, importances

def save_engineered_features(df, output_file=None, importance_dict=None):
    """
    Save engineered features to file.
    
    Args:
        df (pd.DataFrame): DataFrame with engineered features.
        output_file (str, optional): Path to save the features. If None, generates a default name.
        importance_dict (dict, optional): Dictionary of feature importance scores to save.
    
    Returns:
        str: Path to the saved file.
    """
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(config.INTERIM_DATA_DIR, f"engineered_features_{timestamp}.csv")
    
    # Save features
    logger.info(f"Saving engineered features to {output_file}")
    df.to_csv(output_file, index=False)
    
    # Save feature importances if provided
    if importance_dict:
        importance_file = output_file.replace('.csv', '_importances.csv')
        pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).to_csv(importance_file, index=False)
        logger.info(f"Saved feature importances to {importance_file}")
    
    return output_file

def main(input_file=None, output_file=None, target_col='test_passed', 
         n_features=20, selection_methods=None):
    """
    Main function to build and save engineered features.
    
    Args:
        input_file (str, optional): Path to input data file. If None, looks for the latest file.
        output_file (str, optional): Path to save output features. If None, generates a default name.
        target_col (str): Name of the target column.
        n_features (int): Number of features to select.
        selection_methods (list, optional): List of feature selection methods to use.
        
    Returns:
        str: Path to the output file.
    """
    try:
        logger.info("Starting feature engineering pipeline")
        
        # Find input file if not provided
        if input_file is None:
            # Try to find the latest processed data file
            processed_files = []
            
            # First check interim directory for previously preprocessed data
            if os.path.exists(config.INTERIM_DATA_DIR):
                interim_files = [f for f in os.listdir(config.INTERIM_DATA_DIR) 
                               if f.endswith('.csv') and f.startswith(('model_ready_data_', 'test_data_'))]
                processed_files.extend([os.path.join(config.INTERIM_DATA_DIR, f) for f in interim_files])
            
            # Then check processed directory
            if os.path.exists(config.PROCESSED_DATA_DIR):
                proc_files = [f for f in os.listdir(config.PROCESSED_DATA_DIR) 
                             if f.endswith('.csv') and f.startswith('test_data_')]
                processed_files.extend([os.path.join(config.PROCESSED_DATA_DIR, f) for f in proc_files])
            
            # Finally check sample directory
            sample_dir = os.path.join(config.DATA_DIR, 'sample')
            if os.path.exists(sample_dir):
                if os.path.exists(os.path.join(sample_dir, 'test_metadata.csv')):
                    processed_files.append(os.path.join(sample_dir, 'test_metadata.csv'))
            
            if not processed_files:
                logger.error("No input files found")
                return None
            
            # Sort by modification time (newest first)
            processed_files.sort(key=os.path.getmtime, reverse=True)
            input_file = processed_files[0]
        
        # Build features
        logger.info(f"Building features from {input_file}")
        engineered_df, importances = build_features(
            input_file,
            target_col=target_col,
            n_features=n_features,
            selection_methods=selection_methods
        )
        
        if engineered_df.empty:
            logger.warning("No features were generated")
            return None
        
        # Save results
        output_path = save_engineered_features(engineered_df, output_file, importances)
        
        logger.info(f"Feature engineering complete. Output saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.exception(f"Error in feature engineering pipeline: {str(e)}")
        return None

if __name__ == "__main__":
    main() 