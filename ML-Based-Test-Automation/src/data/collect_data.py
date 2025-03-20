"""
Script for collecting and ingesting test log data into the system.
"""
import os
import sys
import logging
import re
import glob
from datetime import datetime
import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"data_collection_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def scan_test_logs(source_dir=None):
    """
    Scan and identify test log files for ingestion.
    
    Args:
        source_dir (str, optional): Directory to scan for test logs. 
                                    Defaults to None (uses configured dir).
    
    Returns:
        list: List of log file paths.
    """
    if source_dir is None:
        source_dir = config.RAW_DATA_DIR
    
    logger.info(f"Scanning for test logs in: {source_dir}")
    
    # Find all log files matching the pattern
    pattern = os.path.join(source_dir, config.TEST_LOG_PATTERN)
    log_files = glob.glob(pattern)
    
    # If no files found in the raw directory, check the sample directory
    if not log_files and os.path.exists(os.path.join(config.DATA_DIR, "sample")):
        sample_pattern = os.path.join(config.DATA_DIR, "sample", config.TEST_LOG_PATTERN)
        log_files = glob.glob(sample_pattern)
        logger.info(f"No files found in raw directory, using sample files instead.")
    
    logger.info(f"Found {len(log_files)} log files for processing")
    return log_files

def parse_test_log_line(line):
    """
    Parse a single line from a test log file.
    
    Args:
        line (str): Line from test log file.
    
    Returns:
        dict: Parsed data from the line or None if not a valid log entry.
    """
    # Define regex patterns for different log entry types
    timestamp_pattern = r'\[(.*?)\]'
    log_level_pattern = r']\s+(INFO|WARNING|ERROR|DEBUG):'
    test_start_pattern = r'Starting test: (.*?)$'
    test_end_pattern = r'Test (passed|failed): (.*?) - Execution time: (.*?)s'
    error_pattern = r'ERROR: (.*?)$'
    suite_pattern = r'Test suite (started|completed) - (.*?)( - .*)?$'
    
    # Extract timestamp
    timestamp_match = re.search(timestamp_pattern, line)
    if not timestamp_match:
        return None
    
    timestamp = timestamp_match.group(1)
    
    # Extract log level
    log_level_match = re.search(log_level_pattern, line)
    if not log_level_match:
        return None
    
    log_level = log_level_match.group(1)
    
    # Base data dictionary
    data = {
        'timestamp': timestamp,
        'log_level': log_level,
        'raw_log': line
    }
    
    # Check for test start
    test_start_match = re.search(test_start_pattern, line)
    if test_start_match:
        data['event_type'] = 'test_start'
        data['test_name'] = test_start_match.group(1)
        return data
    
    # Check for test end
    test_end_match = re.search(test_end_pattern, line)
    if test_end_match:
        data['event_type'] = 'test_end'
        data['test_result'] = test_end_match.group(1)
        data['test_name'] = test_end_match.group(2)
        data['execution_time'] = float(test_end_match.group(3))
        return data
    
    # Check for error
    error_match = re.search(error_pattern, line)
    if error_match and log_level == 'ERROR':
        data['event_type'] = 'error'
        data['error_message'] = error_match.group(1)
        return data
    
    # Check for suite start/end
    suite_match = re.search(suite_pattern, line)
    if suite_match:
        data['event_type'] = f"suite_{suite_match.group(1)}"
        data['suite_name'] = suite_match.group(2)
        
        # Extract test results summary if available
        if suite_match.group(3):
            results_match = re.search(r'(\d+)/(\d+) tests passed', suite_match.group(3))
            if results_match:
                data['tests_passed'] = int(results_match.group(1))
                data['tests_total'] = int(results_match.group(2))
        
        return data
    
    # Default to generic event
    data['event_type'] = 'other'
    return data

def parse_test_log_file(file_path):
    """
    Parse a single test log file into structured data.
    
    Args:
        file_path (str): Path to the log file.
    
    Returns:
        list: List of dictionaries containing parsed log entries.
    """
    logger.info(f"Parsing log file: {file_path}")
    entries = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                entry = parse_test_log_line(line)
                if entry:
                    # Add file metadata
                    entry['file_name'] = os.path.basename(file_path)
                    entry['file_path'] = file_path
                    entries.append(entry)
    
    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {str(e)}")
    
    logger.info(f"Extracted {len(entries)} log entries from {file_path}")
    return entries

def extract_test_executions(log_entries):
    """
    Extract test execution information from parsed log entries.
    
    Args:
        log_entries (list): List of parsed log entries.
    
    Returns:
        pd.DataFrame: DataFrame containing test execution data.
    """
    logger.info("Extracting test execution data")
    
    # Group log entries by test file and test name
    test_executions = []
    current_suite = None
    current_test = None
    test_start_time = None
    errors = []
    
    for entry in log_entries:
        if entry['event_type'] == 'suite_started':
            current_suite = entry['suite_name']
            
        elif entry['event_type'] == 'test_start':
            current_test = entry['test_name']
            test_start_time = entry['timestamp']
            errors = []
            
        elif entry['event_type'] == 'error':
            errors.append(entry['error_message'])
            
        elif entry['event_type'] == 'test_end' and current_test == entry['test_name']:
            # Create a test execution record
            test_execution = {
                'suite_name': current_suite,
                'test_name': current_test,
                'start_time': test_start_time,
                'end_time': entry['timestamp'],
                'execution_time': entry['execution_time'],
                'result': entry['test_result'],
                'errors': '; '.join(errors) if errors else None,
                'error_count': len(errors),
                'file_name': entry['file_name']
            }
            
            test_executions.append(test_execution)
            
            # Reset for next test
            current_test = None
            test_start_time = None
            errors = []
    
    # Convert to DataFrame
    if test_executions:
        return pd.DataFrame(test_executions)
    else:
        return pd.DataFrame()

def parse_test_logs(log_files):
    """
    Parse test log files into structured data.
    
    Args:
        log_files (list): List of log file paths to parse.
    
    Returns:
        pd.DataFrame: DataFrame containing structured test data.
    """
    logger.info(f"Parsing {len(log_files)} log files")
    
    all_log_entries = []
    
    # Parse each log file
    for file_path in log_files:
        entries = parse_test_log_file(file_path)
        all_log_entries.extend(entries)
    
    if not all_log_entries:
        logger.warning("No log entries found in any files")
        return pd.DataFrame()
    
    # Extract test execution data
    test_executions = extract_test_executions(all_log_entries)
    
    # Try to load and merge with test metadata if available
    try:
        metadata_path = os.path.join(config.DATA_DIR, "sample", "test_metadata.csv")
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            test_executions = pd.merge(
                test_executions, 
                metadata, 
                on="test_name", 
                how="left"
            )
            logger.info(f"Merged test metadata from {metadata_path}")
    except Exception as e:
        logger.warning(f"Could not load or merge test metadata: {str(e)}")
    
    return test_executions

def save_processed_data(data, output_file=None):
    """
    Save processed test data to file.
    
    Args:
        data (pd.DataFrame): DataFrame containing structured test data.
        output_file (str, optional): Output file path. Defaults to None.
    
    Returns:
        str: Path to the saved file.
    """
    if output_file is None:
        output_file = os.path.join(
            config.PROCESSED_DATA_DIR, 
            f"test_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    
    logger.info(f"Saving processed data to: {output_file}")
    data.to_csv(output_file, index=False)
    
    return output_file

def main():
    """Main data collection function."""
    try:
        logger.info("Starting test data collection")
        
        # Scan for log files
        log_files = scan_test_logs()
        
        if not log_files:
            logger.warning("No log files found for processing")
            return
        
        # Parse log files into structured data
        data = parse_test_logs(log_files)
        
        if data.empty:
            logger.warning("No data extracted from log files")
            return
        
        # Save processed data
        output_file = save_processed_data(data)
        
        logger.info(f"Data collection completed successfully. Output: {output_file}")
        logger.info(f"Processed {len(data)} test executions")
    
    except Exception as e:
        logger.exception(f"Error during data collection: {str(e)}")

if __name__ == "__main__":
    main() 