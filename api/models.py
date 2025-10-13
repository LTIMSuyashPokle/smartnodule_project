# ========================================================================
# FIXED api/models.py - Complete Models with Proper Config Classes
# ========================================================================

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import torch
import torch.nn as nn
import timm
from torchvision import models
from pydantic_settings import BaseSettings

# ========================================================================
# PYTORCH MODEL CONFIG CLASSES (For Model Loading)
# ========================================================================

class MemoryOptimizedConfig:
    """
    Configuration class that was used during model training
    This needs to be available for PyTorch checkpoint loading
    """
    def __init__(self, *args, **kwargs):
        # Model architecture parameters
        self.backbone = kwargs.get('backbone', 'efficientnet_b3')
        self.num_classes = kwargs.get('num_classes', 1)
        self.dropout_rate = kwargs.get('dropout_rate', 0.3)
        self.pretrained = kwargs.get('pretrained', True)
        
        # Training parameters
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        
        # Memory optimization settings
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
        self.mixed_precision = kwargs.get('mixed_precision', True)
        self.pin_memory = kwargs.get('pin_memory', True)
        
        # Data augmentation
        self.use_mixup = kwargs.get('use_mixup', True)
        self.use_cutmix = kwargs.get('use_cutmix', True)
        self.augmentation_strength = kwargs.get('augmentation_strength', 0.5)

class TrainingConfig:
    """Extended training configuration"""
    def __init__(self, *args, **kwargs):
        self.epochs = kwargs.get('epochs', 100)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        self.scheduler = kwargs.get('scheduler', 'cosine')
        self.warmup_epochs = kwargs.get('warmup_epochs', 5)

# ========================================================================
# PYDANTIC MODELS (API Request/Response Models)  
# ========================================================================

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
# PYTORCH MODEL CLASSES (Actual AI Models)
# ========================================================================

class SmartNoduleModel(nn.Module):
    """
    SmartNodule PyTorch Model for Pulmonary Nodule Detection
    """
    
    def __init__(self, backbone='efficientnet_b3', num_classes=1, pretrained=True):
        super(SmartNoduleModel, self).__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Initialize backbone
        if 'efficientnet' in backbone:
            self.encoder = timm.create_model(backbone, pretrained=pretrained)
            # Get feature dimension
            if hasattr(self.encoder, 'classifier'):
                in_features = self.encoder.classifier.in_features
                self.encoder.classifier = nn.Identity()
            else:
                in_features = self.encoder.fc.in_features
                self.encoder.fc = nn.Identity()
        
        elif 'resnet' in backbone:
            if backbone == 'resnet50':
                self.encoder = models.resnet50(pretrained=pretrained)
            elif backbone == 'resnet101':
                self.encoder = models.resnet101(pretrained=pretrained)
            else:
                self.encoder = models.resnet18(pretrained=pretrained)
            
            in_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classifier head with dropout for uncertainty estimation
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        features = self.encoder(x)
        if len(features.shape) > 2:
            features = torch.mean(features, dim=[2, 3])  # Global average pooling
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x):
        """Extract features without classification"""
        with torch.no_grad():
            features = self.encoder(x)
            if len(features.shape) > 2:
                features = torch.mean(features, dim=[2, 3])
        return features

class EnsembleModel(nn.Module):
    """
    Ensemble of multiple SmartNodule models for improved performance
    """
    
    def __init__(self, model_configs: List[Dict[str, Any]]):
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList()
        for config in model_configs:
            model = SmartNoduleModel(**config)
            self.models.append(model)
    
    def forward(self, x):
        """Forward pass through ensemble"""
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Average ensemble predictions
        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        return ensemble_output
    
    def get_predictions_with_uncertainty(self, x):
        """Get predictions with uncertainty from ensemble"""
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(torch.sigmoid(output))
        
        predictions = torch.stack(outputs)  # [num_models, batch_size, num_classes]
        
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return mean_pred, std_pred

# ========================================================================
# LEGACY COMPATIBILITY CLASSES
# ========================================================================

# For backward compatibility with old API structure
class PredictionRequest(BaseModel):
    """Legacy prediction request model"""
    image_data: str = Field(..., description="Base64 encoded chest X-ray image")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    include_similar_cases: bool = Field(True, description="Include similar case retrieval")
    include_explainability: bool = Field(True, description="Include explainability heatmap")

class SimilarCase(BaseModel):
    """Model for similar case information"""
    case_id: str
    similarity_score: float
    diagnosis: str
    confidence: float
    image_path: Optional[str] = None

class PredictionResponse(BaseModel):
    """Legacy prediction response model"""
    prediction: str
    confidence: float
    nodule_detected: bool
    bounding_boxes: List[List[float]] = []
    explainability_heatmap: Optional[str] = None
    similar_cases: List[SimilarCase] = []
    processing_time: float
    timestamp: datetime
    patient_id: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: datetime


# # Top-level placeholder for MemoryOptimizedConfig to support PyTorch checkpoint loading
# class MemoryOptimizedConfig:
#     def __init__(self, *args, **kwargs):
#         pass
# # ========================================================================
# # api/models.py - Complete Models (Pydantic + PyTorch)
# # ========================================================================

# from pydantic import BaseModel, Field
# from typing import Dict, List, Any, Optional
# from datetime import datetime
# from enum import Enum
# import torch
# import torch.nn as nn
# import timm
# from torchvision import models
# from pydantic_settings import BaseSettings
# # ========================================================================
# # PYDANTIC MODELS (API Request/Response Models)
# # ========================================================================

# class UncertaintyLevel(str, Enum):
#     LOW = "Low"
#     MEDIUM = "Medium"
#     HIGH = "High"

# class PredictionResult(BaseModel):
#     probability: float = Field(..., ge=0.0, le=1.0, description="Nodule probability")
#     confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
#     uncertainty_std: float = Field(..., ge=0.0, description="Uncertainty standard deviation")
#     uncertainty_level: UncertaintyLevel = Field(..., description="Uncertainty level")
#     predicted_class: str = Field(..., description="Predicted class")
#     processing_time: float = Field(..., gt=0, description="Processing time in seconds")
#     explanation: Optional[List[List[float]]] = Field(None, description="Grad-CAM explanation")
#     similar_cases: Optional[List[Dict[str, Any]]] = Field(None, description="Similar cases")
#     model_version: str = Field(..., description="Model version")
#     mc_samples: int = Field(..., gt=0, description="Monte Carlo samples")

# class AnalysisRequest(BaseModel):
#     patient_id: Optional[str] = Field(None, description="Patient identifier")
#     clinical_history: Optional[str] = Field(None, description="Clinical history")

# class AnalysisResponse(BaseModel):
#     request_id: str = Field(..., description="Unique request identifier")
#     prediction: PredictionResult = Field(..., description="Prediction results")
#     processing_time: float = Field(..., description="Total processing time")
#     model_version: str = Field(..., description="Model version used")
#     timestamp: str = Field(..., description="Analysis timestamp")

# class BatchAnalysisRequest(BaseModel):
#     image_paths: List[str] = Field(..., min_items=1, max_items=100, description="List of image paths")
#     patient_ids: Optional[List[str]] = Field(None, description="List of patient IDs")
#     clinical_histories: Optional[List[str]] = Field(None, description="List of clinical histories")

# class ModelPerformanceMetrics(BaseModel):
#     accuracy: float = Field(..., description="Current accuracy")
#     sensitivity: float = Field(..., description="Current sensitivity")
#     specificity: float = Field(..., description="Current specificity")
#     avg_processing_time: float = Field(..., description="Average processing time")
#     total_predictions: int = Field(..., description="Total predictions made")
#     uncertain_cases_count: int = Field(..., description="Number of uncertain cases")
#     last_updated: str = Field(..., description="Last update timestamp")

# class SystemHealthResponse(BaseModel):
#     status: str = Field(..., description="Overall system status")
#     components: Dict[str, Any] = Field(..., description="Component status details")

# class AnnotationRequest(BaseModel):
#     case_id: str = Field(..., description="Case identifier")
#     has_nodule: bool = Field(..., description="Expert annotation - nodule present")
#     nodule_locations: Optional[List[Dict[str, float]]] = Field(None, description="Nodule locations")
#     confidence: float = Field(..., ge=0.0, le=1.0, description="Annotator confidence")
#     notes: Optional[str] = Field(None, description="Additional notes")

# class AnnotationResponse(BaseModel):
#     success: bool = Field(..., description="Annotation submission success")
#     annotation_id: str = Field(..., description="Annotation identifier")
#     quality_score: float = Field(..., description="Annotation quality score")

# # ========================================================================
# # PYTORCH MODEL CLASSES (Actual AI Models)
# # ========================================================================

# class SmartNoduleModel(nn.Module):
#     """
#     SmartNodule PyTorch Model for Pulmonary Nodule Detection
#     """
    
#     def __init__(self, backbone='efficientnet-b3', num_classes=1, pretrained=True):
#         super(SmartNoduleModel, self).__init__()
        
#         self.backbone = backbone
#         self.num_classes = num_classes
        
#         # Initialize backbone
#         if 'efficientnet' in backbone:
#             self.encoder = timm.create_model(backbone, pretrained=pretrained)
#             # Get feature dimension
#             if hasattr(self.encoder, 'classifier'):
#                 in_features = self.encoder.classifier.in_features
#                 self.encoder.classifier = nn.Identity()
#             else:
#                 in_features = self.encoder.fc.in_features
#                 self.encoder.fc = nn.Identity()
        
#         elif 'resnet' in backbone:
#             if backbone == 'resnet50':
#                 self.encoder = models.resnet50(pretrained=pretrained)
#             elif backbone == 'resnet101':
#                 self.encoder = models.resnet101(pretrained=pretrained)
#             else:
#                 self.encoder = models.resnet18(pretrained=pretrained)
            
#             in_features = self.encoder.fc.in_features
#             self.encoder.fc = nn.Identity()
        
#         else:
#             raise ValueError(f"Unsupported backbone: {backbone}")
        
#         # Classifier head with dropout for uncertainty estimation
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(in_features, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(256, num_classes)
#         )
        
#         # Initialize weights
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         """Initialize classifier weights"""
#         for m in self.classifier.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0)
    
#     def forward(self, x):
#         """Forward pass"""
#         features = self.encoder(x)
#         if len(features.shape) > 2:
#             features = torch.mean(features, dim=[2, 3])  # Global average pooling
#         logits = self.classifier(features)
#         return logits
    
#     def extract_features(self, x):
#         """Extract features without classification"""
#         with torch.no_grad():
#             features = self.encoder(x)
#             if len(features.shape) > 2:
#                 features = torch.mean(features, dim=[2, 3])
#         return features

# class EnsembleModel(nn.Module):
#     """
#     Ensemble of multiple SmartNodule models for improved performance
#     """
    
#     def __init__(self, model_configs: List[Dict[str, Any]]):
#         super(EnsembleModel, self).__init__()
        
#         self.models = nn.ModuleList()
#         for config in model_configs:
#             model = SmartNoduleModel(**config)
#             self.models.append(model)
    
#     def forward(self, x):
#         """Forward pass through ensemble"""
#         outputs = []
#         for model in self.models:
#             output = model(x)
#             outputs.append(output)
        
#         # Average ensemble predictions
#         ensemble_output = torch.mean(torch.stack(outputs), dim=0)
#         return ensemble_output
    
#     def get_predictions_with_uncertainty(self, x):
#         """Get predictions with uncertainty from ensemble"""
#         outputs = []
#         for model in self.models:
#             output = model(x)
#             outputs.append(torch.sigmoid(output))
        
#         predictions = torch.stack(outputs)  # [num_models, batch_size, num_classes]
        
#         mean_pred = torch.mean(predictions, dim=0)
#         std_pred = torch.std(predictions, dim=0)
        
#         return mean_pred, std_pred

# # ========================================================================
# # LEGACY COMPATIBILITY CLASSES
# # ========================================================================

# # For backward compatibility with old API structure
# class PredictionRequest(BaseModel):
#     """Legacy prediction request model"""
#     image_data: str = Field(..., description="Base64 encoded chest X-ray image")
#     patient_id: Optional[str] = Field(None, description="Patient identifier")
#     include_similar_cases: bool = Field(True, description="Include similar case retrieval")
#     include_explainability: bool = Field(True, description="Include explainability heatmap")

# class SimilarCase(BaseModel):
#     """Model for similar case information"""
#     case_id: str
#     similarity_score: float
#     diagnosis: str
#     confidence: float
#     image_path: Optional[str] = None

# class PredictionResponse(BaseModel):
#     """Legacy prediction response model"""
#     prediction: str
#     confidence: float
#     nodule_detected: bool
#     bounding_boxes: List[List[float]] = []
#     explainability_heatmap: Optional[str] = None
#     similar_cases: List[SimilarCase] = []
#     processing_time: float
#     timestamp: datetime
#     patient_id: Optional[str] = None

# class HealthResponse(BaseModel):
#     """Health check response"""
#     status: str
#     model_loaded: bool
#     database_connected: bool
#     timestamp: datetime





# # ========================================================================
# # api/models.py - Pydantic Models for API
# # ========================================================================

# from pydantic import BaseModel, Field
# from typing import Dict, List, Any, Optional
# from datetime import datetime
# from enum import Enum

# class UncertaintyLevel(str, Enum):
#     LOW = "Low"
#     MEDIUM = "Medium"
#     HIGH = "High"

# class PredictionResult(BaseModel):
#     probability: float = Field(..., ge=0.0, le=1.0, description="Nodule probability")
#     confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
#     uncertainty_std: float = Field(..., ge=0.0, description="Uncertainty standard deviation")
#     uncertainty_level: UncertaintyLevel = Field(..., description="Uncertainty level")
#     predicted_class: str = Field(..., description="Predicted class")
#     processing_time: float = Field(..., gt=0, description="Processing time in seconds")
#     explanation: Optional[List[List[float]]] = Field(None, description="Grad-CAM explanation")
#     similar_cases: Optional[List[Dict[str, Any]]] = Field(None, description="Similar cases")
#     model_version: str = Field(..., description="Model version")
#     mc_samples: int = Field(..., gt=0, description="Monte Carlo samples")

# class AnalysisRequest(BaseModel):
#     patient_id: Optional[str] = Field(None, description="Patient identifier")
#     clinical_history: Optional[str] = Field(None, description="Clinical history")

# class AnalysisResponse(BaseModel):
#     request_id: str = Field(..., description="Unique request identifier")
#     prediction: PredictionResult = Field(..., description="Prediction results")
#     processing_time: float = Field(..., description="Total processing time")
#     model_version: str = Field(..., description="Model version used")
#     timestamp: str = Field(..., description="Analysis timestamp")

# class BatchAnalysisRequest(BaseModel):
#     image_paths: List[str] = Field(..., min_items=1, max_items=100, description="List of image paths")
#     patient_ids: Optional[List[str]] = Field(None, description="List of patient IDs")
#     clinical_histories: Optional[List[str]] = Field(None, description="List of clinical histories")

# class ModelPerformanceMetrics(BaseModel):
#     accuracy: float = Field(..., description="Current accuracy")
#     sensitivity: float = Field(..., description="Current sensitivity")
#     specificity: float = Field(..., description="Current specificity")
#     avg_processing_time: float = Field(..., description="Average processing time")
#     total_predictions: int = Field(..., description="Total predictions made")
#     uncertain_cases_count: int = Field(..., description="Number of uncertain cases")
#     last_updated: str = Field(..., description="Last update timestamp")

# class SystemHealthResponse(BaseModel):
#     status: str = Field(..., description="Overall system status")
#     components: Dict[str, Any] = Field(..., description="Component status details")

# class AnnotationRequest(BaseModel):
#     case_id: str = Field(..., description="Case identifier")
#     has_nodule: bool = Field(..., description="Expert annotation - nodule present")
#     nodule_locations: Optional[List[Dict[str, float]]] = Field(None, description="Nodule locations")
#     confidence: float = Field(..., ge=0.0, le=1.0, description="Annotator confidence")
#     notes: Optional[str] = Field(None, description="Additional notes")

# class AnnotationResponse(BaseModel):
#     success: bool = Field(..., description="Annotation submission success")
#     annotation_id: str = Field(..., description="Annotation identifier")
#     quality_score: float = Field(..., description="Annotation quality score")
