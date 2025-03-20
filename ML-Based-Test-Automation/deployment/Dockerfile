# Base Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y gcc build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/logs /app/results

# Create a non-root user
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "src.api.prediction_api"] 