# ML-Based Test Automation - Deployment Guide

This guide provides detailed instructions for deploying the ML-Based Test Automation framework in various environments, from local development to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [CI/CD Integration](#cicd-integration)
6. [Environment Configuration](#environment-configuration)
7. [Production Considerations](#production-considerations)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying the ML-Based Test Automation framework, ensure you have the following installed:

- Python 3.8 or higher
- pip
- Docker and Docker Compose (for containerized deployment)
- Kubernetes CLI (kubectl) and access to a Kubernetes cluster (for Kubernetes deployment)
- Git

## Local Development Setup

For local development and testing, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-org/ML-Based-Test-Automation.git
   cd ML-Based-Test-Automation
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```
   mkdir -p data/raw data/processed models logs results
   ```

4. Run the tests to verify the setup:
   ```
   pytest tests/
   ```

5. Train a model:
   ```
   python -m src.models.train_model --save-model
   ```

6. Start the prediction API:
   ```
   python -m src.api.prediction_api
   ```

The API will be available at http://localhost:8000. You can access the API documentation at http://localhost:8000/docs.

## Docker Deployment

For a more isolated and reproducible deployment, use Docker and Docker Compose:

1. Build and start the services using Docker Compose:
   ```
   cd deployment
   docker-compose up -d
   ```

2. To view logs:
   ```
   docker-compose logs -f
   ```

3. To stop the services:
   ```
   docker-compose down
   ```

Docker Compose will start the following services:
- API service (port 8000)
- Training service (runs once and exits)
- Optimizer service (runs once and exits)
- Monitoring dashboard (port 8050)

Each service uses the same Docker image but with different commands and configurations.

## Kubernetes Deployment

For production-grade deployment with scaling and high availability, use Kubernetes:

1. Build and push the Docker image to a registry:
   ```
   docker build -t your-registry/ml-test-automation:latest -f deployment/Dockerfile .
   docker push your-registry/ml-test-automation:latest
   ```

2. Update the image reference in the Kubernetes manifest files:
   ```
   sed -i 's|ml-test-automation:latest|your-registry/ml-test-automation:latest|g' deployment/kubernetes/*.yaml
   ```

3. Apply the Kubernetes manifests:
   ```
   kubectl apply -f deployment/kubernetes/
   ```

4. Check the deployment status:
   ```
   kubectl get pods -l app=ml-test-automation
   ```

5. Set up port forwarding to access the API:
   ```
   kubectl port-forward service/ml-test-api-service 8000:8000
   ```

The API will be available at http://localhost:8000.

## CI/CD Integration

The framework provides built-in support for CI/CD integration with GitHub Actions, Jenkins, and GitLab CI:

### GitHub Actions

1. Generate GitHub Actions workflow:
   ```
   python -m src.integration.cicd_integration generate --type github
   ```

2. Commit and push the generated workflow file:
   ```
   git add .github/workflows/ml_automation.yml
   git commit -m "Add GitHub Actions workflow"
   git push
   ```

### Jenkins

1. Generate Jenkinsfile:
   ```
   python -m src.integration.cicd_integration generate --type jenkins
   ```

2. Commit and push the generated Jenkinsfile:
   ```
   git add Jenkinsfile
   git commit -m "Add Jenkinsfile"
   git push
   ```

### GitLab CI

1. Generate GitLab CI configuration:
   ```
   python -m src.integration.cicd_integration generate --type gitlab
   ```

2. Commit and push the generated configuration:
   ```
   git add .gitlab-ci.yml
   git commit -m "Add GitLab CI configuration"
   git push
   ```

## Environment Configuration

The framework uses environment variables for configuration. You can set these variables in your environment, in a `.env` file, or in the deployment configuration.

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (INFO, DEBUG, etc.) | INFO |
| `MODEL_PATH` | Path to model directory | ./models |
| `DATA_PATH` | Path to data directory | ./data |
| `RESULTS_PATH` | Path to results directory | ./results |

## Production Considerations

When deploying to production, consider the following:

1. **Security**:
   - Use HTTPS with a valid SSL certificate
   - Implement authentication and authorization
   - Run containers with non-root users (already configured in the Dockerfile)
   - Secure sensitive data in environment variables or secrets

2. **Scalability**:
   - Configure horizontal pod autoscaling in Kubernetes
   - Monitor resource usage and adjust resource limits
   - Use a distributed storage solution for models and data

3. **Monitoring**:
   - Set up monitoring for the API and other services
   - Configure alerts for critical failures
   - Implement logging aggregation

4. **Backup and Recovery**:
   - Implement regular backups of models and data
   - Define disaster recovery procedures
   - Test recovery scenarios

## Troubleshooting

Common issues and solutions:

1. **API service won't start**:
   - Check logs: `docker logs ml-test-api`
   - Verify models directory exists and contains trained models
   - Ensure ports are not already in use

2. **Model training fails**:
   - Check data format and availability
   - Verify Python dependencies are correctly installed
   - Check logs for specific error messages

3. **Kubernetes pods in CrashLoopBackOff**:
   - Check logs: `kubectl logs <pod-name>`
   - Verify resource limits are appropriate
   - Check persistent volume claims are bound

4. **CI/CD pipeline fails**:
   - Check workflow logs in the CI/CD platform
   - Verify environment variables are correctly set
   - Ensure tests are passing locally before pushing 