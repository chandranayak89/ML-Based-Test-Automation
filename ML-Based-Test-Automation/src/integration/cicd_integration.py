"""
CI/CD Integration Module for ML-Based Test Automation.
This module provides utilities and helpers for integrating the framework with CI/CD pipelines,
automating testing, model training, and deployment workflows.
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"cicd_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CICDPipeline:
    """
    Class for handling CI/CD pipeline integration tasks such as automated testing,
    model training, evaluation, and deployment steps.
    """
    
    def __init__(
        self, 
        workspace_dir: Optional[str] = None,
        config_file: Optional[str] = None,
        pipeline_type: str = 'github'
    ):
        """
        Initialize the CI/CD pipeline handler.
        
        Args:
            workspace_dir (str, optional): Path to the workspace directory
            config_file (str, optional): Path to the CI/CD configuration file
            pipeline_type (str): Type of CI/CD pipeline (github, jenkins, gitlab, etc.)
        """
        self.workspace_dir = workspace_dir or os.getcwd()
        self.config_file = config_file
        self.pipeline_type = pipeline_type.lower()
        
        # Load CI/CD configuration if provided
        self.config = {}
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading CI/CD configuration: {str(e)}")
        
        logger.info(f"Initialized CI/CD pipeline ({self.pipeline_type}) in workspace: {self.workspace_dir}")
    
    def run_tests(self, test_path: str = 'tests/', pytest_args: Optional[List[str]] = None) -> bool:
        """
        Run automated tests using pytest.
        
        Args:
            test_path (str): Path to the test directory or file
            pytest_args (list, optional): Additional pytest arguments
            
        Returns:
            bool: True if tests pass, False otherwise
        """
        logger.info(f"Running tests from {test_path}")
        
        # Construct pytest command
        cmd = ['pytest', test_path, '-v']
        
        # Add optional pytest arguments
        if pytest_args:
            cmd.extend(pytest_args)
        
        logger.info(f"Test command: {' '.join(cmd)}")
        
        try:
            # Run pytest
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Log and check results
            logger.info(f"Test exit code: {result.returncode}")
            if result.stdout:
                logger.info(f"Test output:\n{result.stdout}")
            if result.stderr:
                logger.error(f"Test errors:\n{result.stderr}")
            
            return result.returncode == 0
        
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            return False
    
    def train_and_evaluate_model(
        self,
        data_path: Optional[str] = None,
        model_type: str = 'random_forest',
        evaluate: bool = True,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train and evaluate a model as part of the CI/CD pipeline.
        
        Args:
            data_path (str, optional): Path to the training data
            model_type (str): Type of model to train
            evaluate (bool): Whether to evaluate the model after training
            save_model (bool): Whether to save the trained model
            
        Returns:
            dict: Training and evaluation results
        """
        from src.models.train_model import train_model, evaluate_model
        
        logger.info(f"Training {model_type} model for CI/CD pipeline")
        
        try:
            # Train the model
            model, model_info = train_model(
                data_path=data_path,
                model_type=model_type,
                save_model=save_model
            )
            
            results = {
                'model_type': model_type,
                'training_time': model_info.get('training_time', 0),
                'model_path': model_info.get('model_path', ''),
                'timestamp': datetime.now().isoformat()
            }
            
            # Evaluate the model if requested
            if evaluate and model:
                evaluation = evaluate_model(model, data_path=data_path)
                results['evaluation'] = evaluation
            
            logger.info(f"Model training completed: {results}")
            return results
        
        except Exception as e:
            logger.error(f"Error in model training pipeline: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def deploy_model(
        self,
        model_path: Optional[str] = None,
        environment: str = 'staging',
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deploy a trained model to the specified environment.
        
        Args:
            model_path (str, optional): Path to the model to deploy
            environment (str): Target environment (staging, production)
            version (str, optional): Version tag for the deployment
            
        Returns:
            dict: Deployment results
        """
        logger.info(f"Deploying model to {environment} environment")
        
        # If no model path is provided, find the latest model
        if not model_path:
            from src.models.predict import find_best_model
            model_path = find_best_model()
            if not model_path:
                logger.error("No model found for deployment")
                return {
                    'error': 'No model found for deployment',
                    'timestamp': datetime.now().isoformat()
                }
        
        # Generate version if not provided
        if not version:
            version = datetime.now().strftime('%Y%m%d%H%M%S')
        
        try:
            # Placeholder for actual deployment logic
            # In a real implementation, this would copy the model to a deployment location,
            # update configuration, restart services, etc.
            
            # Simulate deployment
            deployment_dir = os.path.join(config.MODELS_DIR, 'deployed', environment)
            os.makedirs(deployment_dir, exist_ok=True)
            
            # Create deployment record
            deployment_record = {
                'model_path': model_path,
                'environment': environment,
                'version': version,
                'deployed_by': os.environ.get('USER', 'ci_system'),
                'deployed_at': datetime.now().isoformat(),
                'status': 'success'
            }
            
            # Save deployment record
            record_path = os.path.join(
                deployment_dir, 
                f"deployment_{environment}_{version}.json"
            )
            with open(record_path, 'w') as f:
                json.dump(deployment_record, f, indent=2)
            
            logger.info(f"Model deployed to {environment}: {record_path}")
            return deployment_record
        
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            return {
                'error': str(e),
                'environment': environment,
                'model_path': model_path,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed'
            }
    
    def generate_cicd_config(self, output_path: Optional[str] = None) -> str:
        """
        Generate CI/CD configuration files based on the pipeline type.
        
        Args:
            output_path (str, optional): Path to save the generated configuration
            
        Returns:
            str: Path to the generated configuration file
        """
        logger.info(f"Generating CI/CD configuration for {self.pipeline_type}")
        
        # Set default output path if not provided
        if not output_path:
            if self.pipeline_type == 'github':
                output_dir = os.path.join(self.workspace_dir, '.github', 'workflows')
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, 'ml_automation.yml')
            elif self.pipeline_type == 'jenkins':
                output_path = os.path.join(self.workspace_dir, 'Jenkinsfile')
            elif self.pipeline_type == 'gitlab':
                output_path = os.path.join(self.workspace_dir, '.gitlab-ci.yml')
            else:
                output_path = os.path.join(self.workspace_dir, f"cicd_{self.pipeline_type}.yml")
        
        try:
            # Generate configuration based on pipeline type
            if self.pipeline_type == 'github':
                config_content = self._generate_github_workflow()
            elif self.pipeline_type == 'jenkins':
                config_content = self._generate_jenkinsfile()
            elif self.pipeline_type == 'gitlab':
                config_content = self._generate_gitlab_config()
            else:
                logger.warning(f"Unsupported pipeline type: {self.pipeline_type}")
                return ''
            
            # Write configuration to file
            with open(output_path, 'w') as f:
                f.write(config_content)
            
            logger.info(f"CI/CD configuration saved to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating CI/CD configuration: {str(e)}")
            return ''
    
    def _generate_github_workflow(self) -> str:
        """
        Generate a GitHub Actions workflow configuration.
        
        Returns:
            str: GitHub Actions workflow YAML content
        """
        workflow = """name: ML-Based Test Automation Pipeline

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sundays

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install pytest pytest-cov
    - name: Run tests
      run: |
        pytest tests/ --cov=src/ --cov-report=xml
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: false

  train_model:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master')
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Train and evaluate model
      run: |
        python -m src.models.train_model
        python -m src.models.evaluate_model
    - name: Archive model artifacts
      uses: actions/upload-artifact@v3
      with:
        name: ml-models
        path: models/

  deploy_staging:
    runs-on: ubuntu-latest
    needs: train_model
    if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master')
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Download model artifacts
      uses: actions/download-artifact@v3
      with:
        name: ml-models
        path: models/
    - name: Deploy to staging
      run: |
        python -m src.integration.cicd_integration deploy --environment staging
"""
        return workflow
    
    def _generate_jenkinsfile(self) -> str:
        """
        Generate a Jenkinsfile for Jenkins pipeline.
        
        Returns:
            str: Jenkinsfile content
        """
        jenkinsfile = """pipeline {
    agent {
        docker {
            image 'python:3.9'
        }
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'mkdir -p logs models/deployed'
            }
        }
        
        stage('Test') {
            steps {
                sh 'pytest tests/ --cov=src/ --junitxml=results.xml'
                junit 'results.xml'
            }
        }
        
        stage('Train Model') {
            when {
                branch 'main'
            }
            steps {
                sh 'python -m src.models.train_model'
                sh 'python -m src.models.evaluate_model'
            }
            post {
                success {
                    archiveArtifacts artifacts: 'models/*.pkl', fingerprint: true
                }
            }
        }
        
        stage('Deploy to Staging') {
            when {
                branch 'main'
            }
            steps {
                sh 'python -m src.integration.cicd_integration deploy --environment staging'
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            input {
                message "Deploy to production?"
                ok "Yes, deploy it!"
            }
            steps {
                sh 'python -m src.integration.cicd_integration deploy --environment production'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'logs/*.log', fingerprint: true
        }
    }
}
"""
        return jenkinsfile
    
    def _generate_gitlab_config(self) -> str:
        """
        Generate a GitLab CI configuration.
        
        Returns:
            str: GitLab CI YAML content
        """
        gitlab_ci = """stages:
  - test
  - train
  - deploy_staging
  - deploy_production

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.pip-cache"

cache:
  paths:
    - .pip-cache/

test:
  stage: test
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - pytest tests/ --cov=src/ --junitxml=results.xml
  artifacts:
    reports:
      junit: results.xml
    paths:
      - results.xml

train_model:
  stage: train
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - python -m src.models.train_model
    - python -m src.models.evaluate_model
  artifacts:
    paths:
      - models/
  only:
    - main
    - master

deploy_staging:
  stage: deploy_staging
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - python -m src.integration.cicd_integration deploy --environment staging
  only:
    - main
    - master

deploy_production:
  stage: deploy_production
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - python -m src.integration.cicd_integration deploy --environment production
  when: manual
  only:
    - main
    - master
"""
        return gitlab_ci

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='CI/CD Integration for ML-Based Test Automation')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # generate command
    generate_parser = subparsers.add_parser('generate', help='Generate CI/CD configuration')
    generate_parser.add_argument('--type', choices=['github', 'jenkins', 'gitlab'], 
                                 default='github', help='CI/CD platform type')
    generate_parser.add_argument('--output', help='Output path for the configuration file')
    
    # test command
    test_parser = subparsers.add_parser('test', help='Run automated tests')
    test_parser.add_argument('--path', default='tests/', help='Path to test directory or file')
    test_parser.add_argument('--args', nargs='+', help='Additional pytest arguments')
    
    # train command
    train_parser = subparsers.add_parser('train', help='Train and evaluate model')
    train_parser.add_argument('--data', help='Path to training data')
    train_parser.add_argument('--model', default='random_forest', help='Model type to train')
    train_parser.add_argument('--no-evaluate', action='store_true', help='Skip evaluation')
    
    # deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy model')
    deploy_parser.add_argument('--model', help='Path to model to deploy')
    deploy_parser.add_argument('--environment', default='staging', 
                               choices=['staging', 'production'], help='Target environment')
    deploy_parser.add_argument('--version', help='Version tag for deployment')
    
    return parser.parse_args()

def main():
    """
    Main function to run CI/CD integration tasks from the command line.
    """
    args = parse_args()
    
    # Initialize CI/CD pipeline
    pipeline = CICDPipeline()
    
    # Execute the requested command
    if args.command == 'generate':
        pipeline.pipeline_type = args.type
        output_path = pipeline.generate_cicd_config(args.output)
        print(f"CI/CD configuration generated: {output_path}")
        
    elif args.command == 'test':
        success = pipeline.run_tests(args.path, args.args)
        print(f"Tests {'passed' if success else 'failed'}")
        if not success:
            sys.exit(1)
        
    elif args.command == 'train':
        results = pipeline.train_and_evaluate_model(
            data_path=args.data,
            model_type=args.model,
            evaluate=not args.no_evaluate
        )
        print(f"Model training results: {json.dumps(results, indent=2)}")
        
    elif args.command == 'deploy':
        results = pipeline.deploy_model(
            model_path=args.model,
            environment=args.environment,
            version=args.version
        )
        print(f"Deployment results: {json.dumps(results, indent=2)}")
    
    else:
        print("No command specified. Use --help for usage information.")

if __name__ == "__main__":
    main() 