"""
Main entry point for the ML-Based Test Automation project.
"""
import os
import sys
import logging
import argparse
from datetime import datetime

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"main_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_argparse():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(description="ML-Based Test Automation Tool")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Data collection command
    collect_parser = subparsers.add_parser("collect", help="Collect and process test data")
    collect_parser.add_argument("--source", help="Source directory for test logs")
    
    # Model training command
    train_parser = subparsers.add_parser("train", help="Train prediction models")
    train_parser.add_argument("--data", help="Path to training data")
    train_parser.add_argument("--model-type", choices=["rf", "xgb", "nn"], default="rf",
                          help="Model type to train (rf=Random Forest, xgb=XGBoost, nn=Neural Network)")
    
    # Test scheduling command
    schedule_parser = subparsers.add_parser("schedule", help="Schedule test execution")
    schedule_parser.add_argument("--test-dir", help="Directory containing test files")
    
    # Analysis command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze test results")
    analyze_parser.add_argument("--results", help="Path to test results data")
    
    return parser

def main():
    """Main function to run the ML-Based Test Automation tool."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        logger.info(f"Starting ML-Based Test Automation - Command: {args.command}")
        
        if args.command == "collect":
            # Import here to avoid circular imports
            from src.data.collect_data import main as collect_main
            collect_main()
            
        elif args.command == "train":
            logger.info("Training models - Not implemented yet")
            # TODO: Implement model training
            
        elif args.command == "schedule":
            logger.info("Scheduling tests - Not implemented yet")
            # TODO: Implement test scheduling
            
        elif args.command == "analyze":
            logger.info("Analyzing results - Not implemented yet")
            # TODO: Implement results analysis
            
        logger.info(f"Completed {args.command} command successfully")
            
    except Exception as e:
        logger.exception(f"Error during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 