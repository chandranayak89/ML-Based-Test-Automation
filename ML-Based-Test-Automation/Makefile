.PHONY: setup test lint train deploy-local deploy-k8s clean help generate-cicd

# Variables
VERSION ?= latest
DOCKER_IMAGE = ml-test-automation
PYTHON = python3
PIP = pip3
PYTEST = pytest
DOCKER = docker
DOCKER_COMPOSE = docker-compose
KUBERNETES = kubectl

# Help command
help:
	@echo "ML-Based Test Automation Framework"
	@echo ""
	@echo "Usage:"
	@echo "  make setup              Install dependencies"
	@echo "  make test               Run tests"
	@echo "  make lint               Run linting checks"
	@echo "  make train              Train a new model"
	@echo "  make deploy-local       Deploy locally using Docker Compose"
	@echo "  make deploy-k8s         Deploy to Kubernetes"
	@echo "  make generate-cicd      Generate CI/CD configuration files"
	@echo "  make clean              Clean up workspace"
	@echo ""
	@echo "Options:"
	@echo "  VERSION=1.0.0           Set version for deployment (default: latest)"

# Setup environment and dependencies
setup:
	$(PIP) install -r requirements.txt
	mkdir -p data/raw data/processed models logs results

# Run tests
test:
	$(PYTEST) tests/ -v --cov=src --cov-report=term --cov-report=xml:coverage.xml

# Run linting
lint:
	flake8 src tests
	mypy src --ignore-missing-imports

# Train a new model
train:
	$(PYTHON) -m src.models.train_model --save-model

# Deploy locally using Docker Compose
deploy-local:
	cd deployment && \
	VERSION=$(VERSION) $(DOCKER_COMPOSE) up -d

# Deploy to Kubernetes
deploy-k8s:
	$(DOCKER) build -t $(DOCKER_IMAGE):$(VERSION) -f deployment/Dockerfile .
	$(KUBERNETES) apply -f deployment/kubernetes/api-deployment.yaml

# Generate CI/CD configuration
generate-cicd:
	$(PYTHON) -m src.integration.cicd_integration generate

# Clean up
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .mypy_cache/ 