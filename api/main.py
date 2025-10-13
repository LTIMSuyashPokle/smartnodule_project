# ========================================================================
# 1. FASTAPI INFERENCE ENGINE WITH ADVANCED FEATURES
# ========================================================================

# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import asyncio
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import json
import io
from PIL import Image
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from contextlib import asynccontextmanager
import uuid
from fastapi.responses import JSONResponse
from api.models import (
    AnalysisRequest, AnalysisResponse, BatchAnalysisRequest, 
    ModelPerformanceMetrics, SystemHealthResponse
)
from api.inference_engine import SmartNoduleInferenceEngine
from api.middleware import authenticate_user, log_request, validate_image
from api.config import get_settings
from mlops.experiment_tracker import MLflowTracker
from active_learning.uncertainty_queue import UncertaintyQueue
from monitoring.performance_metrics import PerformanceMonitor

# Global variables for model and dependencies
inference_engine = None
mlflow_tracker = None
uncertainty_queue = None
performance_monitor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global inference_engine, mlflow_tracker, uncertainty_queue, performance_monitor
    
    # Startup
    logging.info("Starting SmartNodule API Server...")
    
    # Initialize core components
    settings = get_settings()
    inference_engine = SmartNoduleInferenceEngine(
        model_path=settings.MODEL_PATH,
        device=settings.DEVICE
    )
    
    mlflow_tracker = MLflowTracker(
        tracking_uri=settings.MLFLOW_TRACKING_URI,
        experiment_name="SmartNodule_Production"
    )
    
    uncertainty_queue = UncertaintyQueue(
        db_path=settings.UNCERTAINTY_DB_PATH,
        threshold=settings.UNCERTAINTY_THRESHOLD
    )
    
    performance_monitor = PerformanceMonitor(
        metrics_db=settings.METRICS_DB_PATH
    )
    
    logging.info("✅ SmartNodule API Server Ready!")
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down SmartNodule API Server...")

# Create FastAPI application with advanced configuration
app = FastAPI(
    title="SmartNodule API",
    description="Production-grade AI API for Pulmonary Nodule Detection",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# ========================================================================
# CORE API ENDPOINTS
# ========================================================================

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_chest_xray(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    clinical_history: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Advanced chest X-ray analysis with uncertainty quantification
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    try:
        # Authentication and logging
        user_id = authenticate_user(credentials.credentials)
        log_request(request_id, user_id, "analyze", start_time)
        
        # Validate and process image
        image = validate_image(await file.read())
        
        # AI Inference with uncertainty quantification
        results = await inference_engine.analyze_with_uncertainty(
            image=image,
            patient_id=patient_id,
            clinical_history=clinical_history
        )
        
        # Performance monitoring
        processing_time = (datetime.now() - start_time).total_seconds()
        performance_monitor.log_inference(
            request_id=request_id,
            processing_time=processing_time,
            prediction_confidence=results['confidence'],
            uncertainty_level=results['uncertainty_level']
        )
        
        # Queue uncertain cases for expert review
        if results['uncertainty_level'] in ['Medium', 'High']:
            background_tasks.add_task(
                uncertainty_queue.add_case,
                request_id=request_id,
                image_data=image,
                prediction=results,
                priority=2 if results['uncertainty_level'] == 'High' else 1
            )
        
        # MLflow experiment logging
        background_tasks.add_task(
            mlflow_tracker.log_prediction,
            request_id=request_id,
            prediction=results,
            processing_time=processing_time
        )
        
        return AnalysisResponse(
            request_id=request_id,
            prediction=results,
            processing_time=processing_time,
            model_version=inference_engine.model_version,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logging.error(f"Analysis failed for request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/batch-analyze")
async def batch_analyze_chest_xrays(
    background_tasks: BackgroundTasks,
    request: BatchAnalysisRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Batch processing for multiple chest X-rays
    """
    batch_id = str(uuid.uuid4())
    user_id = authenticate_user(credentials.credentials)
    
    try:
        # Process batch asynchronously
        background_tasks.add_task(
            #process_batch,
            batch_id=batch_id,
            image_paths=request.image_paths,
            user_id=user_id
        )
        
        return {
            "batch_id": batch_id,
            "status": "processing",
            "estimated_completion": "5-10 minutes",
            "images_count": len(request.image_paths)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/api/v1/model-performance", response_model=ModelPerformanceMetrics)
async def get_model_performance(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get real-time model performance metrics
    """
    user_id = authenticate_user(credentials.credentials)
    
    try:
        metrics = performance_monitor.get_current_metrics()
        return ModelPerformanceMetrics(**metrics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/api/v1/health", response_model=SystemHealthResponse)
async def health_check():
    """
    Comprehensive system health check
    """
    try:
        health_status = {
            "api_status": "healthy",
            "model_status": "loaded" if inference_engine.is_ready() else "error",
            "mlflow_status": "connected" if mlflow_tracker.is_connected() else "disconnected",
            "database_status": "connected" if uncertainty_queue.is_connected() else "error",
            "timestamp": datetime.now().isoformat(),
            "uptime": performance_monitor.get_uptime(),
            "memory_usage": performance_monitor.get_memory_usage(),
            "gpu_usage": performance_monitor.get_gpu_usage() if torch.cuda.is_available() else None
        }
        
        overall_status = "healthy" if all([
            health_status["model_status"] == "loaded",
            health_status["database_status"] == "connected"
        ]) else "unhealthy"
        
        return SystemHealthResponse(
            status=overall_status,
            components=health_status
        )
        
    except Exception as e:
        return SystemHealthResponse(
            status="unhealthy",
            components={"error": str(e)}
        )

@app.get("/api/v1/uncertain-cases")
async def get_uncertain_cases(
    limit: int = 20,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get cases queued for expert annotation
    """
    user_id = authenticate_user(credentials.credentials)
    
    try:
        cases = uncertainty_queue.get_pending_cases(limit=limit)
        return {
            "cases": cases,
            "total_pending": uncertainty_queue.get_pending_count(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get uncertain cases: {str(e)}")