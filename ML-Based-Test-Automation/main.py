#!/usr/bin/env python3
"""
ML-Based Test Automation - Main Entry Point

This script provides a unified command-line interface to access all the main functionality
of the ML-Based Test Automation framework.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

# Import core modules
from src.data import collect_data, preprocess_data
from src.features import build_features
from src.models import train_model, evaluate_model, predict
from src.execution import test_scheduler, suite_optimizer, root_cause_analyzer, impact_analyzer
from src.reporting import dashboard
from src.integration import cicd_integration
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

# Define command-line arguments
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="ML-Based Test Automation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect and process test data
  python main.py data --source path/to/logs

  # Train a new model
  python main.py train --data path/to/data

  # Predict test failures
  python main.py predict --test-id TEST-1001

  # Generate an optimized test execution plan
  python main.py schedule --metadata path/to/metadata.csv

  # Start the dashboard
  python main.py dashboard
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Data collection and preprocessing
    data_parser = subparsers.add_parser("data", help="Collect and preprocess test data")
    data_parser.add_argument("--source", help="Path to source test logs")
    data_parser.add_argument("--output", help="Output path for processed data")
    data_parser.add_argument("--preprocess-only", action="store_true", help="Skip collection, only preprocess")
    
    # Feature engineering
    features_parser = subparsers.add_parser("features", help="Extract and build features")
    features_parser.add_argument("--data", help="Path to raw data")
    features_parser.add_argument("--output", help="Output path for feature data")
    
    # Model training
    train_parser = subparsers.add_parser("train", help="Train a prediction model")
    train_parser.add_argument("--data", help="Path to training data")
    train_parser.add_argument("--model-type", choices=["random_forest", "gradient_boosting", "neural_network"], 
                            default="random_forest", help="Type of model to train")
    train_parser.add_argument("--output-dir", help="Directory to save trained model")
    train_parser.add_argument("--evaluate", action="store_true", help="Evaluate model after training")
    
    # Model evaluation
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", help="Path to model file")
    eval_parser.add_argument("--data", help="Path to evaluation data")
    eval_parser.add_argument("--output-dir", help="Directory to save evaluation results")
    
    # Prediction
    predict_parser = subparsers.add_parser("predict", help="Make test failure predictions")
    predict_parser.add_argument("--model", help="Path to model file (uses best model if not specified)")
    predict_parser.add_argument("--test-id", help="ID of the test to predict")
    predict_parser.add_argument("--test-metadata", help="Path to test metadata file for batch prediction")
    predict_parser.add_argument("--output", help="Output path for predictions")
    
    # Test scheduling
    schedule_parser = subparsers.add_parser("schedule", help="Generate optimized test execution plan")
    schedule_parser.add_argument("--metadata", help="Path to test metadata file")
    schedule_parser.add_argument("--output-dir", help="Directory to save execution plan")
    schedule_parser.add_argument("--time-constraint", type=int, help="Time constraint in minutes")
    schedule_parser.add_argument("--model", help="Path to model file (uses best model if not specified)")
    
    # Suite optimization
    optimize_parser = subparsers.add_parser("optimize", help="Optimize test suite")
    optimize_parser.add_argument("--metadata", help="Path to test metadata file")
    optimize_parser.add_argument("--output", help="Output path for optimized suite")
    optimize_parser.add_argument("--coverage", type=float, default=0.8, help="Minimum coverage threshold (0-1)")
    optimize_parser.add_argument("--redundancy", type=float, default=0.3, help="Maximum redundancy threshold (0-1)")
    
    # Root cause analysis
    root_cause_parser = subparsers.add_parser("root-cause", help="Analyze test failures")
    root_cause_parser.add_argument("--results", help="Path to test results file")
    root_cause_parser.add_argument("--output", help="Output path for analysis results")
    root_cause_parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    # Impact analysis
    impact_parser = subparsers.add_parser("impact", help="Analyze code change impact")
    impact_parser.add_argument("--changes", nargs="+", help="List of changed files")
    impact_parser.add_argument("--metadata", help="Path to test metadata file")
    impact_parser.add_argument("--output", help="Output path for impact analysis")
    
    # Dashboard
    dashboard_parser = subparsers.add_parser("dashboard", help="Start the reporting dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8050, help="Port to run dashboard on")
    dashboard_parser.add_argument("--results-dir", help="Directory containing test results")
    
    # CI/CD
    cicd_parser = subparsers.add_parser("cicd", help="CI/CD integration utilities")
    cicd_parser.add_argument("--action", choices=["generate", "test", "train", "deploy"], 
                           required=True, help="CI/CD action to perform")
    cicd_parser.add_argument("--type", choices=["github", "jenkins", "gitlab"], 
                           default="github", help="CI/CD platform type")
    cicd_parser.add_argument("--output", help="Output path for CI/CD configuration")
    cicd_parser.add_argument("--env", choices=["staging", "production"], 
                           default="staging", help="Deployment environment")
    
    # API server
    api_parser = subparsers.add_parser("api", help="Start the prediction API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to run API on")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to run API on")
    
    return parser.parse_args()

def main():
    """Main entry point for the application"""
    args = parse_args()
    
    # If no command provided, show help
    if not args.command:
        print("No command specified. Use -h for help.")
        return 1
    
    try:
        # Execute the selected command
        if args.command == "data":
            # Data collection and preprocessing
            if not args.preprocess_only:
                logger.info("Collecting test data...")
                collect_data.collect_data(source_path=args.source, output_path=args.output)
            
            logger.info("Preprocessing test data...")
            preprocess_data.preprocess_data(input_path=args.output or args.source, output_path=args.output)
            
            logger.info("Data processing completed successfully")
        
        elif args.command == "features":
            # Feature engineering
            logger.info("Building features...")
            build_features.build_features(data_path=args.data, output_path=args.output)
            
            logger.info("Feature engineering completed successfully")
        
        elif args.command == "train":
            # Model training
            logger.info(f"Training {args.model_type} model...")
            model, model_info = train_model.train_model(
                data_path=args.data,
                model_type=args.model_type,
                output_dir=args.output_dir,
                save_model=True
            )
            
            if args.evaluate and model:
                logger.info("Evaluating trained model...")
                evaluation = evaluate_model.evaluate_model(
                    model=model,
                    data_path=args.data,
                    output_dir=args.output_dir
                )
                
                logger.info(f"Model evaluation results: {evaluation}")
            
            logger.info(f"Model training completed: {model_info}")
        
        elif args.command == "evaluate":
            # Model evaluation
            logger.info("Evaluating model...")
            model = train_model.load_model(args.model) if args.model else None
            
            evaluation = evaluate_model.evaluate_model(
                model=model,
                data_path=args.data,
                output_dir=args.output_dir
            )
            
            logger.info(f"Model evaluation results: {evaluation}")
        
        elif args.command == "predict":
            # Prediction
            logger.info("Making test failure predictions...")
            if args.test_id:
                # Single test prediction
                result = predict.predict_single_test(
                    test_id=args.test_id,
                    model_path=args.model
                )
                logger.info(f"Prediction result: {result}")
            
            elif args.test_metadata:
                # Batch prediction
                results = predict.predict_batch(
                    metadata_path=args.test_metadata,
                    model_path=args.model,
                    output_path=args.output
                )
                logger.info(f"Batch prediction completed: {len(results)} predictions")
            
            else:
                logger.error("Either --test-id or --test-metadata must be specified")
                return 1
        
        elif args.command == "schedule":
            # Test scheduling
            logger.info("Generating optimized test execution plan...")
            scheduler = test_scheduler.TestScheduler(
                model_path=args.model,
                output_dir=args.output_dir
            )
            
            execution_plan = scheduler.schedule_tests_from_metadata(
                metadata_path=args.metadata,
                time_constraint=args.time_constraint
            )
            
            logger.info(f"Test execution plan generated: {len(execution_plan['tests'])} tests")
        
        elif args.command == "optimize":
            # Test suite optimization
            logger.info("Optimizing test suite...")
            optimizer = suite_optimizer.TestSuiteOptimizer(
                min_coverage_threshold=args.coverage,
                max_redundancy_threshold=args.redundancy
            )
            
            optimized_suite = optimizer.generate_optimized_suite(
                test_metadata_path=args.metadata,
                output_path=args.output
            )
            
            logger.info(f"Test suite optimization completed: {len(optimized_suite['tests_to_keep'])} tests to keep")
        
        elif args.command == "root-cause":
            # Root cause analysis
            logger.info("Analyzing test failures...")
            analyzer = root_cause_analyzer.RootCauseAnalyzer()
            
            analysis = analyzer.analyze_failures(
                results_path=args.results,
                output_path=args.output,
                generate_visualizations=args.visualize
            )
            
            logger.info(f"Root cause analysis completed: {len(analysis['common_failures'])} common failure patterns identified")
        
        elif args.command == "impact":
            # Impact analysis
            logger.info("Analyzing code change impact...")
            impact_analyzer = impact_analyzer.ImpactAnalyzer()
            
            impact = impact_analyzer.analyze_impact(
                changed_files=args.changes,
                test_metadata_path=args.metadata,
                output_path=args.output
            )
            
            logger.info(f"Impact analysis completed: {len(impact['affected_tests'])} affected tests identified")
        
        elif args.command == "dashboard":
            # Start dashboard
            logger.info(f"Starting dashboard on port {args.port}...")
            reporting_dashboard = dashboard.TestReportingDashboard(
                results_dir=args.results_dir,
                port=args.port
            )
            
            reporting_dashboard.launch_dashboard()
        
        elif args.command == "cicd":
            # CI/CD integration
            logger.info(f"Executing CI/CD action: {args.action}")
            pipeline = cicd_integration.CICDPipeline(pipeline_type=args.type)
            
            if args.action == "generate":
                output_path = pipeline.generate_cicd_config(args.output)
                logger.info(f"CI/CD configuration generated: {output_path}")
            
            elif args.action == "test":
                success = pipeline.run_tests()
                logger.info(f"CI/CD tests {'passed' if success else 'failed'}")
                if not success:
                    return 1
            
            elif args.action == "train":
                results = pipeline.train_and_evaluate_model()
                logger.info(f"CI/CD model training completed: {results}")
            
            elif args.action == "deploy":
                results = pipeline.deploy_model(environment=args.env)
                logger.info(f"CI/CD deployment completed: {results}")
        
        elif args.command == "api":
            # Start API server
            logger.info(f"Starting API server on {args.host}:{args.port}...")
            from src.api.prediction_api import app
            import uvicorn
            
            uvicorn.run(app, host=args.host, port=args.port)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error executing command {args.command}: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 