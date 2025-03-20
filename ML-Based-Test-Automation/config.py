"""
Configuration settings for the ML-Based Test Automation project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data Configuration
TEST_LOG_PATTERN = "*.log"  # Pattern for test log files
HISTORICAL_DATA_LIMIT = 90  # Days of historical data to consider

# Model Configuration
MODEL_VERSION = "0.1.0"
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.1
MODEL_METRICS = ["accuracy", "precision", "recall", "f1"]

# Feature Engineering Configuration
FEATURE_SELECTION_METHOD = "mutual_info"  # Options: mutual_info, chi2, rfe
MAX_FEATURES = 50

# Test Execution Configuration
TEST_PRIORITY_LEVELS = {
    "critical": 1,
    "high": 2,
    "medium": 3,
    "low": 4
}
MAX_PARALLEL_TESTS = 5
TEST_TIMEOUT = 300  # seconds

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# CI/CD Integration
CI_PLATFORM = os.getenv("CI_PLATFORM", "github")  # Options: github, jenkins, gitlab
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# API Configuration
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", 5000))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true" 