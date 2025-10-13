# CONFIGURATION FILES FOR MODULE 4
# ========================================================================
# api/config.py - Configuration Management
# ========================================================================

import os
from pathlib import Path
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings with environment variable support
    """
    
    # Application settings
    APP_NAME: str = "SmartNodule API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Model settings
    MODEL_PATH: str = "smartnodule_memory_optimized_best.pth"
    DEVICE: str = "auto"  # auto, cpu, cuda, mps
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "SmartNodule_Production"
    MLFLOW_ARTIFACT_ROOT: str = "./mlflow-artifacts"
    
    # Database settings
    UNCERTAINTY_DB_PATH: str = "uncertain_cases.db"
    METRICS_DB_PATH: str = "performance_metrics.db"
    SQLITE_DB_PATH: str = "smartnodule_database.db"
    
    # Active learning settings
    UNCERTAINTY_THRESHOLD: float = 0.15
    MIN_CASES_FOR_RETRAIN: int = 50
    RETRAIN_INTERVAL_HOURS: int = 24
    
    # Case retrieval settings
    FAISS_INDEX_PATH: str = "case_retrieval/faiss_index.idx"
    CASE_METADATA_PATH: str = "case_retrieval/case_metadata.pkl"
    
    # Security settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "smartnodule_api.log"
    
    # Performance settings
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 300  # 5 minutes
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings"""
    return settings

# ========================================================================
# api/models.py - Pydantic Models for API
# ========================================================================

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

class UncertaintyLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class PredictionResult(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0, description="Nodule probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    uncertainty_std: float = Field(..., ge=0.0, description="Uncertainty standard deviation")
    uncertainty_level: UncertaintyLevel = Field(..., description="Uncertainty level")
    predicted_class: str = Field(..., description="Predicted class")
    processing_time: float = Field(..., gt=0, description="Processing time in seconds")
    explanation: Optional[List[List[float]]] = Field(None, description="Grad-CAM explanation")
    similar_cases: Optional[List[Dict[str, Any]]] = Field(None, description="Similar cases")
    model_version: str = Field(..., description="Model version")
    mc_samples: int = Field(..., gt=0, description="Monte Carlo samples")

class AnalysisRequest(BaseModel):
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    clinical_history: Optional[str] = Field(None, description="Clinical history")

class AnalysisResponse(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    prediction: PredictionResult = Field(..., description="Prediction results")
    processing_time: float = Field(..., description="Total processing time")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Analysis timestamp")

class BatchAnalysisRequest(BaseModel):
    image_paths: List[str] = Field(..., min_items=1, max_items=100, description="List of image paths")
    patient_ids: Optional[List[str]] = Field(None, description="List of patient IDs")
    clinical_histories: Optional[List[str]] = Field(None, description="List of clinical histories")

class ModelPerformanceMetrics(BaseModel):
    accuracy: float = Field(..., description="Current accuracy")
    sensitivity: float = Field(..., description="Current sensitivity")
    specificity: float = Field(..., description="Current specificity")
    avg_processing_time: float = Field(..., description="Average processing time")
    total_predictions: int = Field(..., description="Total predictions made")
    uncertain_cases_count: int = Field(..., description="Number of uncertain cases")
    last_updated: str = Field(..., description="Last update timestamp")

class SystemHealthResponse(BaseModel):
    status: str = Field(..., description="Overall system status")
    components: Dict[str, Any] = Field(..., description="Component status details")

class AnnotationRequest(BaseModel):
    case_id: str = Field(..., description="Case identifier")
    has_nodule: bool = Field(..., description="Expert annotation - nodule present")
    nodule_locations: Optional[List[Dict[str, float]]] = Field(None, description="Nodule locations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Annotator confidence")
    notes: Optional[str] = Field(None, description="Additional notes")

class AnnotationResponse(BaseModel):
    success: bool = Field(..., description="Annotation submission success")
    annotation_id: str = Field(..., description="Annotation identifier")
    quality_score: float = Field(..., description="Annotation quality score")

# ========================================================================
# api/middleware.py - Authentication and Logging Middleware
# ========================================================================

import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
import hashlib
import uuid
from PIL import Image
import io
import numpy as np

# Simple authentication (replace with proper auth in production)
VALID_TOKENS = {
    "demo-token-123": {"user_id": "demo_user", "role": "admin"},
    "clinical-token-456": {"user_id": "clinical_user", "role": "clinician"},
    "research-token-789": {"user_id": "research_user", "role": "researcher"}
}

def authenticate_user(token: str) -> str:
    """
    Authenticate user token (simplified for demo)
    In production, use proper JWT validation
    """
    if token in VALID_TOKENS:
        return VALID_TOKENS[token]["user_id"]
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )

def log_request(request_id: str, user_id: str, endpoint: str, timestamp: datetime):
    """Log API request for audit trail"""
    logging.info(f"API_REQUEST | {request_id} | {user_id} | {endpoint} | {timestamp.isoformat()}")

def validate_image(image_data: bytes) -> np.ndarray:
    """
    Validate and process uploaded image
    """
    try:
        # Open image
        image = Image.open(io.BytesIO(image_data))
        
        # Validate format
        if image.format not in ['JPEG', 'PNG', 'TIFF']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported image format: {image.format}"
            )
        
        # Validate size
        if image.size[0] < 100 or image.size[1] < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image too small (minimum 100x100 pixels)"
            )
        
        if image.size[0] > 2048 or image.size[1] > 2048:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image too large (maximum 2048x2048 pixels)"
            )
        
        # Convert to numpy array
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image data: {str(e)}"
        )

def generate_request_id() -> str:
    """Generate unique request ID"""
    return str(uuid.uuid4())

# ========================================================================
# requirements.txt - Dependencies
# ========================================================================

REQUIREMENTS_TXT = '''
# Core ML/AI
torch>=1.12.0
torchvision>=0.13.0
efficientnet-pytorch>=0.7.1
timm>=0.6.12
numpy>=1.21.0
opencv-python>=4.6.0
pillow>=9.2.0
scikit-image>=0.19.3
scikit-learn>=1.1.0
albumentations>=1.3.0

# API and Web
fastapi>=0.95.0
uvicorn[standard]>=0.20.0
python-multipart>=0.0.6
aiofiles>=22.1.0
streamlit>=1.28.0
requests>=2.28.0

# MLOps
mlflow>=2.3.0
pandas>=1.5.0
pydantic>=1.10.0

# Database
sqlite3
faiss-cpu>=1.7.3

# Monitoring and Logging
schedule>=1.2.0
psutil>=5.9.0

# Development
pytest>=7.2.0
black>=22.0.0
flake8>=5.0.0

# Security
python-jose[cryptography]>=3.3.0
python-multipart>=0.0.6

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.14.0
'''

# ========================================================================
# docker-compose.yml (if Docker becomes available later)
# ========================================================================

DOCKER_COMPOSE = '''
version: '3.8'

services:
  smartnodule-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./models:/app/models
      - ./case_retrieval:/app/case_retrieval
      - ./logs:/app/logs
    depends_on:
      - mlflow
    restart: unless-stopped

  mlflow:
    image: python:3.9-slim
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow:/mlflow
    command: >
      sh -c "pip install mlflow &&
             mlflow server 
             --backend-store-uri sqlite:///mlflow/mlflow.db
             --default-artifact-root /mlflow/artifacts
             --host 0.0.0.0
             --port 5000"
    restart: unless-stopped

  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./:/app
    command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
    depends_on:
      - smartnodule-api
    restart: unless-stopped

volumes:
  mlflow-data:
  model-data:
'''

# ========================================================================
# .env - Environment Variables (Template)
# ========================================================================

ENV_TEMPLATE = '''
# SmartNodule Configuration

# Application
APP_NAME=SmartNodule API
DEBUG=false

# Server
HOST=0.0.0.0
PORT=8000

# Model
MODEL_PATH=smartnodule_memory_optimized_best.pth
DEVICE=auto

# MLflow
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=SmartNodule_Production

# Security (CHANGE IN PRODUCTION!)
SECRET_KEY=your-very-secret-key-change-this-in-production

# Active Learning
UNCERTAINTY_THRESHOLD=0.15
MIN_CASES_FOR_RETRAIN=50
RETRAIN_INTERVAL_HOURS=24

# Logging
LOG_LEVEL=INFO
'''

# ========================================================================
# setup_module4.py - Automated Setup Script
# ========================================================================

#SETUP_SCRIPT = '''
#!/usr/bin/env python3
"""
Automated setup script for SmartNodule Module 4
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create required directory structure"""
    directories = [
        "api",
        "mlops", 
        "active_learning",
        "monitoring",
        "deployment",
        "logs",
        "mlflow-artifacts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        # Create __init__.py files
        (Path(directory) / "__init__.py").touch()
    
    print("✅ Directory structure created")

def create_config_files():
    """Create configuration files"""
    
    # Create requirements.txt
    with open("requirements.txt", "w") as f:
        f.write(REQUIREMENTS_TXT)
    
    # Create .env template
    with open(".env.template", "w") as f:
        f.write(ENV_TEMPLATE)
    
    # Create docker-compose.yml
    with open("docker-compose.yml", "w") as f:
        f.write(DOCKER_COMPOSE)
    
    print("✅ Configuration files created")

def setup_logging():
    """Setup logging directory and files"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create log files
    (logs_dir / "smartnodule_api.log").touch()
    (logs_dir / "mlflow.log").touch()
    (logs_dir / "uncertainty_queue.log").touch()
    
    print("✅ Logging setup completed")

def main():
    """Main setup function"""
    print("🚀 Setting up SmartNodule Module 4...")
    
    create_directory_structure()
    create_config_files()
    setup_logging()
    
    print("""
🎉 Module 4 setup completed!

Next steps:
1. Copy your trained model to: smartnodule_memory_optimized_best.pth
2. Install dependencies: pip install -r requirements.txt
3. Copy .env.template to .env and update settings
4. Start MLflow: mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
5. Start API: python -m api.main
6. Start Streamlit: streamlit run app.py

Access points:
- API docs: http://localhost:8000/api/docs
- MLflow UI: http://localhost:5000
- Streamlit: http://localhost:8501
""")

if __name__ == "__main__":
    main()
'''

print("Configuration files created! Here's what I've prepared for you:")
print("\n📁 Configuration Files:")
print("✅ api/config.py - Application settings")
print("✅ api/models.py - Pydantic models for API")
print("✅ api/middleware.py - Authentication and validation")
print("✅ requirements.txt - All dependencies")
print("✅ .env template - Environment variables")
print("✅ docker-compose.yml - Container orchestration")
print("✅ setup_module4.py - Automated setup script")