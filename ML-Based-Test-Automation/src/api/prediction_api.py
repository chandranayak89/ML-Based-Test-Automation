"""
Prediction API for ML-Based Test Automation
This module provides a REST API for making predictions on test failures and prioritizing tests.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Depends, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.models.predict import (
    find_best_model, 
    load_model, 
    preprocess_test_features, 
    predict_test_failures,
    predict_prioritized_tests
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, f"prediction_api_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ML-Based Test Automation API",
    description="API for predicting test failures and prioritizing tests",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class TestData(BaseModel):
    test_id: str
    test_name: str
    component: str
    priority: str
    avg_execution_time: float
    recent_failure_rate: float = Field(..., ge=0.0, le=1.0)
    recent_runs: int
    code_churn: Optional[int] = None
    test_age_days: Optional[int] = None
    dependencies: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "test_id": "TEST-1001",
                "test_name": "test_login_system",
                "component": "authentication",
                "priority": "high",
                "avg_execution_time": 15.2,
                "recent_failure_rate": 0.25,
                "recent_runs": 20,
                "code_churn": 150,
                "test_age_days": 95,
                "dependencies": 3
            }
        }

class TestBatch(BaseModel):
    tests: List[TestData]

class PredictionResponse(BaseModel):
    test_id: str
    test_name: str
    failure_probability: float
    predicted_outcome: str
    priority_score: float
    
    class Config:
        schema_extra = {
            "example": {
                "test_id": "TEST-1001",
                "test_name": "test_login_system",
                "failure_probability": 0.78,
                "predicted_outcome": "FAIL", 
                "priority_score": 0.92
            }
        }

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    model_name: str
    model_version: str
    timestamp: str

class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    model_version: str
    training_date: str
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]

def get_model_path():
    """
    Get the path to the best model.
    Returns a model path or raises an exception if no model is found.
    """
    model_path = find_best_model(metric='f1')
    if not model_path:
        raise HTTPException(status_code=500, detail="No model found for prediction")
    return model_path

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint that returns basic API information."""
    return {
        "name": "ML-Based Test Automation API",
        "version": "1.0.0",
        "description": "API for predicting test failures and prioritizing tests",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API information"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/model/info", "method": "GET", "description": "Get model information"},
            {"path": "/predict/single", "method": "POST", "description": "Predict single test"},
            {"path": "/predict/batch", "method": "POST", "description": "Predict batch of tests"},
            {"path": "/predict/file", "method": "POST", "description": "Predict from file upload"}
        ]
    }

@app.get("/health", response_class=JSONResponse)
async def health():
    """Health check endpoint."""
    try:
        # Check if model exists
        model_path = get_model_path()
        
        return {
            "status": "healthy",
            "model": os.path.basename(model_path),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the current model."""
    try:
        model_path = get_model_path()
        
        # Load model metadata
        metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=500, detail="Model metadata not found")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get feature importance if available
        feature_importance = {}
        importance_path = os.path.join(
            os.path.dirname(model_path), 
            "feature_importance.json"
        )
        
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                feature_importance = json.load(f)
        
        return {
            "model_name": metadata.get("model_name", os.path.basename(model_path)),
            "model_type": metadata.get("model_type", "unknown"),
            "model_version": metadata.get("version", "1.0.0"),
            "training_date": metadata.get("trained_at", "unknown"),
            "metrics": metadata.get("metrics", {}),
            "feature_importance": feature_importance
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single_test(test: TestData):
    """
    Predict failure probability for a single test.
    """
    try:
        logger.info(f"Predicting for test: {test.test_id}")
        
        # Convert to DataFrame for prediction
        test_df = pd.DataFrame([test.dict()])
        
        # Get model path
        model_path = get_model_path()
        model = load_model(model_path)
        
        # Make prediction
        result_df = predict_prioritized_tests(
            model=model,
            test_data=test_df,
            threshold=0.5
        )
        
        if result_df.empty:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Get first row of results
        result = result_df.iloc[0]
        
        return {
            "test_id": test.test_id,
            "test_name": test.test_name,
            "failure_probability": float(result['failure_probability']),
            "predicted_outcome": "FAIL" if result['predicted_result'] else "PASS",
            "priority_score": float(result['priority_score'])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error predicting test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(tests: TestBatch):
    """
    Predict failure probabilities for a batch of tests and return prioritized results.
    """
    try:
        logger.info(f"Batch prediction requested for {len(tests.tests)} tests")
        
        # Convert to DataFrame
        test_data = pd.DataFrame([test.dict() for test in tests.tests])
        
        # Get model
        model_path = get_model_path()
        model = load_model(model_path)
        
        # Get model metadata
        metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
        model_name = os.path.basename(model_path)
        model_version = "1.0.0"
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                model_name = metadata.get("model_name", model_name)
                model_version = metadata.get("version", model_version)
        
        # Make prediction
        result_df = predict_prioritized_tests(
            model=model,
            test_data=test_data,
            threshold=0.5
        )
        
        if result_df.empty:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Format response
        predictions = []
        for _, row in result_df.iterrows():
            predictions.append({
                "test_id": row.get('test_id', ''),
                "test_name": row.get('test_name', ''),
                "failure_probability": float(row['failure_probability']),
                "predicted_outcome": "FAIL" if row['predicted_result'] else "PASS",
                "priority_score": float(row['priority_score'])
            })
        
        # Sort by priority score descending
        predictions.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return {
            "predictions": predictions,
            "model_name": model_name,
            "model_version": model_version,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/file", response_model=BatchPredictionResponse)
async def predict_from_file(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Probability threshold for failure prediction")
):
    """
    Predict test failures from a CSV or JSON file containing test data.
    """
    try:
        logger.info(f"File prediction requested: {file.filename}")
        
        # Save uploaded file
        temp_dir = os.path.join(config.TEMP_DIR, "uploads")
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load data based on file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext == '.csv':
            data = pd.read_csv(file_path)
        elif file_ext == '.json':
            data = pd.read_json(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or JSON.")
        
        # Check for required columns
        required_cols = ['test_id', 'test_name']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_cols)}"
            )
        
        # Get model
        model_path = get_model_path()
        model = load_model(model_path)
        
        # Get model metadata
        metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
        model_name = os.path.basename(model_path)
        model_version = "1.0.0"
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                model_name = metadata.get("model_name", model_name)
                model_version = metadata.get("version", model_version)
        
        # Make prediction
        result_df = predict_prioritized_tests(
            model=model,
            test_data=data,
            threshold=threshold
        )
        
        if result_df.empty:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Format response
        predictions = []
        for _, row in result_df.iterrows():
            predictions.append({
                "test_id": row.get('test_id', ''),
                "test_name": row.get('test_name', ''),
                "failure_probability": float(row['failure_probability']),
                "predicted_outcome": "FAIL" if row['predicted_result'] else "PASS",
                "priority_score": float(row['priority_score'])
            })
        
        # Sort by priority score descending
        predictions.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Clean up
        os.remove(file_path)
        
        return {
            "predictions": predictions,
            "model_name": model_name,
            "model_version": model_version,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in file prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File prediction error: {str(e)}")

@app.post("/prediction/bulk", response_class=JSONResponse)
async def schedule_bulk_prediction(
    background_tasks: BackgroundTasks,
    input_path: str = Form(...),
    output_path: str = Form(...),
    threshold: float = Form(0.5)
):
    """
    Schedule a bulk prediction job to run in the background.
    
    This endpoint is designed for large datasets that would timeout in a regular request.
    """
    try:
        # Validate paths
        if not os.path.exists(input_path):
            raise HTTPException(status_code=400, detail=f"Input path does not exist: {input_path}")
        
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Schedule task
        task_id = f"bulk_prediction_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        background_tasks.add_task(
            run_bulk_prediction,
            task_id=task_id,
            input_path=input_path,
            output_path=output_path,
            threshold=threshold
        )
        
        return {
            "status": "scheduled",
            "task_id": task_id,
            "message": f"Bulk prediction scheduled. Results will be saved to {output_path}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error scheduling bulk prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scheduling task: {str(e)}")

async def run_bulk_prediction(
    task_id: str,
    input_path: str,
    output_path: str,
    threshold: float = 0.5
):
    """
    Background task to run bulk prediction.
    """
    logger.info(f"Starting bulk prediction task {task_id}")
    try:
        from src.models.predict import predict_on_new_data
        
        # Get model path
        model_path = find_best_model(metric='f1')
        if not model_path:
            logger.error(f"Task {task_id}: No model found for prediction")
            return
        
        # Run prediction
        predict_on_new_data(
            model_path=model_path,
            data_file=input_path,
            output_file=output_path,
            threshold=threshold
        )
        
        logger.info(f"Task {task_id}: Bulk prediction completed. Results saved to {output_path}")
        
    except Exception as e:
        logger.exception(f"Task {task_id}: Error in bulk prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run("src.api.prediction_api:app", host=host, port=port, reload=True) 