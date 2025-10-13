# ========================================================================
# api/config.py - Configuration Management
# ========================================================================

import os
from pathlib import Path
#from pydantic import BaseSettings
from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import ConfigDict

class Settings(BaseSettings):
    """
    Application settings with environment variable support
    """
    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra="allow")
    #model_config = ConfigDict(extra="allow")
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
    SECRET_KEY: str = "suyashpoklesmartnodule18"   #
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "smartnodule_api.log"
    
    # Performance settings
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 300  # 5 minutes
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    
    # class Config:
    #     env_file = ".env"
    #     case_sensitive = True

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings"""
    return settings