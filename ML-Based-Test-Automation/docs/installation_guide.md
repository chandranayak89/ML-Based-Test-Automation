# ML-Based Test Automation - Installation Guide

This guide provides detailed instructions for installing and setting up the ML-Based Test Automation framework.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Methods](#installation-methods)
   - [Method 1: Install from Source](#method-1-install-from-source)
   - [Method 2: Install Using Pip](#method-2-install-using-pip)
   - [Method 3: Using Docker](#method-3-using-docker)
3. [Environment Setup](#environment-setup)
4. [Verifying Installation](#verifying-installation)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

Before installing ML-Based Test Automation, ensure you have the following:

- **Python**: Version 3.8 or higher
- **Pip**: Latest version recommended
- **Git**: For cloning the repository (if installing from source)
- **Docker**: (Optional) For containerized deployment
- **Disk Space**: At least 2GB of free disk space
- **Memory**: Minimum 4GB RAM recommended

For GPU acceleration (optional but recommended for larger datasets):
- CUDA-compatible GPU
- CUDA Toolkit 11.2 or higher
- cuDNN 8.1 or higher

## Installation Methods

### Method 1: Install from Source

This is the recommended method for development or customization.

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ML-Based-Test-Automation.git
   cd ML-Based-Test-Automation
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Using venv
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

   This installs the package in editable mode, allowing you to modify the source code without reinstalling.

### Method 2: Install Using Pip

For users who want to use the framework without modifying the code:

```bash
pip install ml-test-automation
```

### Method 3: Using Docker

For isolated, containerized deployment:

1. Pull the Docker image:
   ```bash
   docker pull your-registry/ml-test-automation:latest
   ```

   Or build from the Dockerfile:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ML-Based-Test-Automation.git
   cd ML-Based-Test-Automation
   docker build -t ml-test-automation:latest -f deployment/Dockerfile .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 -p 8050:8050 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models ml-test-automation:latest
   ```

## Environment Setup

After installation, set up your environment:

1. Configure the data paths:

   Create a `.env` file in your project root (or wherever you'll run the commands from):
   ```
   DATA_PATH=/path/to/your/data
   MODELS_PATH=/path/to/store/models
   RESULTS_PATH=/path/to/store/results
   LOG_LEVEL=INFO
   ```

2. Create the necessary directories:
   ```bash
   mkdir -p data/raw data/processed models logs results
   ```

3. (Optional) Configure GPU support:

   For TensorFlow:
   ```bash
   pip install tensorflow-gpu==2.9.1
   ```

   For PyTorch (if used in custom models):
   ```bash
   pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
   ```

## Verifying Installation

Verify your installation by running:

```bash
# If installed as a package
ml-test-automation --help

# Or using the main script directly
python main.py --help
```

You should see a list of available commands and options.

Run a simple test:

```bash
python main.py data --source examples/data --preprocess-only
```

## Troubleshooting

### Common Issues:

1. **Missing Dependencies**:
   - Error: `ModuleNotFoundError: No module named 'xyz'`
   - Solution: Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **CUDA/GPU Issues**:
   - Error: `Could not load dynamic library 'cudart64_110.dll'`
   - Solution: Verify CUDA installation and ensure compatible versions of TensorFlow/CUDA

3. **Permission Errors**:
   - Error: `Permission denied: '/path/to/dir'`
   - Solution: Check file/directory permissions or run with elevated privileges

4. **Memory Errors**:
   - Error: `MemoryError` or `OutOfMemoryError`
   - Solution: Reduce batch size, use data chunking, or increase system memory

5. **Import Errors After Installation**:
   - Error: `ImportError: cannot import name 'xyz' from 'src'`
   - Solution: Ensure you're in the correct Python environment and the package is installed properly

For more detailed troubleshooting, refer to the [Troubleshooting Guide](troubleshooting.md) or [open an issue](https://github.com/YOUR_USERNAME/ML-Based-Test-Automation/issues) on GitHub. 