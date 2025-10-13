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