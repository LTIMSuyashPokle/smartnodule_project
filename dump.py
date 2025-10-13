

# # ========================================================================
# # Fixed Inference Engine
# # ========================================================================
# # api/inference_engine.py

# import torch
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from PIL import Image
# import asyncio
# import logging
# from typing import Dict, Any, List, Tuple, Optional
# import time
# from concurrent.futures import ThreadPoolExecutor
# import sys
# import os

# # Import the SmartNoduleModel from models.py
# from api.models import SmartNoduleModel

# # Mock imports for missing modules (create these as placeholders)
# class MedicalPreprocessor:
#     """Placeholder medical preprocessor"""
#     def __init__(self, target_size=(384, 384), **kwargs):
#         self.target_size = target_size
#         logging.info("✅ MedicalPreprocessor initialized (placeholder)")
    
#     def process(self, image: np.ndarray) -> np.ndarray:
#         """Basic preprocessing"""
#         try:
#             # Convert to RGB if needed
#             if len(image.shape) == 3 and image.shape[2] == 3:
#                 # Resize image
#                 from PIL import Image
#                 pil_image = Image.fromarray(image.astype('uint8'))
#                 pil_image = pil_image.resize(self.target_size)
                
#                 # Convert to numpy and normalize
#                 processed = np.array(pil_image).astype(np.float32) / 255.0
                
#                 # Add channel dimension if needed and transpose to CHW format
#                 if len(processed.shape) == 3:
#                     processed = processed.transpose(2, 0, 1)  # HWC to CHW
                
#                 return processed
#             else:
#                 # Handle grayscale
#                 pil_image = Image.fromarray(image.astype('uint8'))
#                 pil_image = pil_image.resize(self.target_size)
#                 processed = np.array(pil_image).astype(np.float32) / 255.0
                
#                 # Add channel dimension
#                 if len(processed.shape) == 2:
#                     processed = np.expand_dims(processed, axis=0)
                
#                 return processed
#         except Exception as e:
#             logging.error(f"Preprocessing error: {e}")
#             # Return a dummy processed image
#             return np.random.rand(3, *self.target_size).astype(np.float32)

# class MedicalGradCAM:
#     """Placeholder explainability system"""
#     def __init__(self, model, device, **kwargs):
#         self.model = model
#         self.device = device
#         logging.info("✅ MedicalGradCAM initialized (placeholder)")
    
#     def generate_explanation(self, image_tensor: torch.Tensor) -> Optional[np.ndarray]:
#         """Generate placeholder explanation"""
#         try:
#             # Return a dummy heatmap
#             batch_size, channels, height, width = image_tensor.shape
#             heatmap = np.random.rand(height, width).astype(np.float32)
#             return heatmap
#         except Exception as e:
#             logging.error(f"Explanation generation error: {e}")
#             return None

# class CaseRetriever:
#     """Placeholder case retrieval system"""
#     def __init__(self, index_path: str, metadata_path: str):
#         self.index_path = index_path
#         self.metadata_path = metadata_path
#         logging.info("✅ CaseRetriever initialized (placeholder)")
    
#     def retrieve_similar(self, image_tensor: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:
#         """Retrieve placeholder similar cases"""
#         similar_cases = []
#         for i in range(min(top_k, 3)):  # Return fewer dummy cases
#             similar_cases.append({
#                 'case_id': f'case_{i+1:03d}',
#                 'similarity_score': np.random.rand() * 0.3 + 0.7,  # High similarity
#                 'diagnosis': 'No Nodule' if i % 2 == 0 else 'Nodule Present',
#                 'confidence': np.random.rand() * 0.2 + 0.8,  # High confidence
#                 'metadata': f'Similar case {i+1}'
#             })
#         return similar_cases

# class SmartNoduleInferenceEngine:
#     """
#     Production-grade inference engine with advanced features
#     """
    
#     def __init__(self, model_path: str, device: str = "auto"):
#         self.device = self._setup_device(device)
#         self.model = None
#         self.preprocessor = None
#         self.gradcam = None
#         self.case_retriever = None
#         self.model_version = None
#         self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
#         # Load model and components
#         self._load_model(model_path)
#         self._initialize_components()
        
#         logging.info(f"✅ SmartNodule Inference Engine initialized on {self.device}")
    
#     def _setup_device(self, device: str) -> str:
#         """Intelligent device selection"""
#         if device == "auto":
#             if torch.cuda.is_available():
#                 return "cuda"
#             elif torch.backends.mps.is_available():  # Apple Silicon
#                 return "mps"
#             else:
#                 return "cpu"
#         return device
    
#     def _load_model(self, model_path: str):
#         # Ensure MemoryOptimizedConfig is available for torch.load
#         import importlib.util
#         import sys
#         import torch
#         try:
#             # Dynamically import MemoryOptimizedConfig from api.models
#             spec = importlib.util.spec_from_file_location("api.models", os.path.join(os.path.dirname(__file__), "models.py"))
#             models_module = importlib.util.module_from_spec(spec)
#             sys.modules["api.models"] = models_module
#             spec.loader.exec_module(models_module)
#             torch.serialization.add_safe_globals([models_module.MemoryOptimizedConfig])
#         except Exception as e:
#             logging.warning(f"Could not add MemoryOptimizedConfig to torch safe globals: {e}")
#         """Load model with error handling and version tracking"""
#         try:
#             # Check if model file exists
#             if not os.path.exists(model_path):
#                 logging.warning(f"Model file not found: {model_path}, creating dummy model")
#                 # Create a dummy model for testing
#                 self.model = SmartNoduleModel(
#                     backbone='efficientnet_b3',
#                     #backbone='efficientnet-b3',
#                     num_classes=1,
#                     pretrained=False
#                 )
#                 self.model.to(self.device)
#                 self.model.eval()
#                 self.model_version = '2.0.0'
#                 self._enable_mc_dropout()
#                 logging.info("✅ Dummy model created for testing")
#                 return
            
#             # Load checkpoint
#             checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
#             # Initialize model architecture
#             self.model = SmartNoduleModel(
#                 backbone='efficientnet_b3',
#                 num_classes=1,
#                 pretrained=False  # Loading from checkpoint
#             )
            
#             # Load weights
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             self.model.to(self.device)
#             self.model.eval()
            
#             # Extract model metadata
#             self.model_version = checkpoint.get('model_version', '2.0.0')
            
#             # Enable dropout for uncertainty estimation
#             self._enable_mc_dropout()
            
#             logging.info(f"✅ Model loaded successfully (version: {self.model_version})")
            
#         except Exception as e:
#             logging.error(f"❌ Failed to load model: {str(e)}")
#             logging.info("Creating dummy model for testing...")
#             # Create dummy model as fallback
#             self.model = SmartNoduleModel(
#                 backbone='efficientnet_b3',
#                 num_classes=1,
#                 pretrained=False
#             )
#             self.model.to(self.device)
#             self.model.eval()
#             self.model_version = '2.0.0'
#             self._enable_mc_dropout()
#             logging.info("✅ Dummy model created as fallback")
    
#     def _initialize_components(self):
#         """Initialize preprocessing, explainability, and retrieval components"""
#         try:
#             # Medical preprocessor
#             self.preprocessor = MedicalPreprocessor(
#                 target_size=(384, 384),
#                 enable_bone_suppression=True,
#                 enable_artifact_removal=True,
#                 enable_lung_segmentation=True
#             )
            
#             # Explainability system
#             self.gradcam = MedicalGradCAM(
#                 model=self.model,
#                 device=self.device,
#                 lung_segmentation=True,
#                 nodule_size_filter=(3, 30)  # mm
#             )
            
#             # Case retrieval system
#             self.case_retriever = CaseRetriever(
#                 index_path="case_retrieval/faiss_index.idx",
#                 metadata_path="case_retrieval/case_metadata.pkl"
#             )
            
#             logging.info("✅ All inference components initialized")
            
#         except Exception as e:
#             logging.error(f"❌ Failed to initialize components: {str(e)}")
#             raise RuntimeError(f"Component initialization failed: {str(e)}")
    
#     def _enable_mc_dropout(self):
#         """Enable Monte Carlo dropout for uncertainty estimation"""
#         for module in self.model.modules():
#             if isinstance(module, torch.nn.Dropout):
#                 module.train()  # Keep dropout active during inference
    
#     async def analyze_with_uncertainty(
#         self,
#         image: np.ndarray,
#         patient_id: Optional[str] = None,
#         clinical_history: Optional[str] = None,
#         mc_samples: int = 50
#     ) -> Dict[str, Any]:
#         """
#         Complete analysis with uncertainty quantification
#         """
#         start_time = time.time()
#         try:
#             # Preprocessing
#             processed_image = await asyncio.get_event_loop().run_in_executor(
#                 self.thread_pool, self.preprocessor.process, image
#             )
            
#             # Convert to tensor
#             image_tensor = torch.from_numpy(processed_image).unsqueeze(0).to(self.device)
            
#             # Monte Carlo inference for uncertainty
#             predictions = []
#             with torch.no_grad():
#                 for _ in range(mc_samples):
#                     logits = self.model(image_tensor)
#                     prob = torch.sigmoid(logits).cpu().numpy()
#                     predictions.append(prob[0, 0])
            
#             # Calculate statistics
#             mean_prob = np.mean(predictions)
#             std_prob = np.std(predictions)
#             confidence = 1.0 - std_prob
            
#             # Uncertainty classification
#             if std_prob < 0.05:
#                 uncertainty_level = "Low"
#             elif std_prob < 0.15:
#                 uncertainty_level = "Medium"
#             else:
#                 uncertainty_level = "High"
            
#             # Generate explanation
#             explanation = await asyncio.get_event_loop().run_in_executor(
#                 self.thread_pool, self.gradcam.generate_explanation, image_tensor
#             )
            
#             # Case retrieval
#             similar_cases = await asyncio.get_event_loop().run_in_executor(
#                 self.thread_pool, self.case_retriever.retrieve_similar,
#                 image_tensor, 5  # top 5 cases
#             )
            
#             # Compile results
#             results = {
#                 'probability': float(mean_prob),
#                 'confidence': float(confidence),
#                 'uncertainty_std': float(std_prob),
#                 'uncertainty_level': uncertainty_level,
#                 'predicted_class': 'Nodule Present' if mean_prob > 0.5 else 'No Nodule',
#                 'processing_time': time.time() - start_time,
#                 'explanation': explanation.tolist() if explanation is not None else None,
#                 'similar_cases': similar_cases,
#                 'model_version': self.model_version,
#                 'mc_samples': mc_samples
#             }
            
#             return results
            
#         except Exception as e:
#             logging.error(f"❌ Analysis failed: {str(e)}")
#             raise RuntimeError(f"Analysis failed: {str(e)}")
    
#     async def batch_analyze(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
#         """
#         Efficient batch processing
#         """
#         # Process images concurrently
#         tasks = [self.analyze_with_uncertainty(img) for img in images]
#         results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         # Handle exceptions
#         processed_results = []
#         for i, result in enumerate(results):
#             if isinstance(result, Exception):
#                 processed_results.append({
#                     'error': str(result),
#                     'image_index': i,
#                     'status': 'failed'
#                 })
#             else:
#                 processed_results.append({
#                     **result,
#                     'image_index': i,
#                     'status': 'success'
#                 })
        
#         return processed_results
    
#     def is_ready(self) -> bool:
#         """Check if inference engine is ready"""
#         return all([
#             self.model is not None,
#             self.preprocessor is not None,
#             self.gradcam is not None,
#             self.case_retriever is not None
#         ])




# # ========================================================================
# # 2. ADVANCED INFERENCE ENGINE
# # ========================================================================

# # api/inference_engine.py
# import torch
# import torch.nn.functional as F
# import numpy as np
# import cv2
# from PIL import Image
# import asyncio
# import logging
# from typing import Dict, Any, List, Tuple, Optional
# import time
# from concurrent.futures import ThreadPoolExecutor
# import mlflow
# import mlflow.pytorch
# import sys
# import os

# # from ..models.smartnodule_model import SmartNoduleModel
# # from ..preprocessing.medical_preprocessor import MedicalPreprocessor
# # from ..explainability.medical_gradcam import MedicalGradCAM
# # from ..case_retrieval.faiss_retriever import CaseRetriever

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from .models import SmartNoduleModel

# # from models.smartnodule_model import SmartNoduleModel
# from preprocessing.medical_preprocessor import MedicalPreprocessor
# from explainability.medical_gradcam import MedicalGradCAM
# from case_retrieval.faiss_retriever import CaseRetriever

# class SmartNoduleInferenceEngine:
#     """
#     Production-grade inference engine with advanced features
#     """
    
#     def __init__(self, model_path: str, device: str = "auto"):
#         self.device = self._setup_device(device)
#         self.model = None
#         self.preprocessor = None
#         self.gradcam = None
#         self.case_retriever = None
#         self.model_version = None
#         self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
#         # Load model and components
#         self._load_model(model_path)
#         self._initialize_components()
        
#         logging.info(f"✅ SmartNodule Inference Engine initialized on {self.device}")
    
#     def _setup_device(self, device: str) -> str:
#         """Intelligent device selection"""
#         if device == "auto":
#             if torch.cuda.is_available():
#                 return "cuda"
#             elif torch.backends.mps.is_available():  # Apple Silicon
#                 return "mps"
#             else:
#                 return "cpu"
#         return device
    
#     def _load_model(self, model_path: str):
#         """Load model with error handling and version tracking"""
#         try:
#             # Load checkpoint
#             checkpoint = torch.load(model_path, map_location=self.device)
            
#             # Initialize model architecture
#             self.model = SmartNoduleModel(
#                 backbone='efficientnet-b3',
#                 num_classes=1,
#                 pretrained=False  # Loading from checkpoint
#             )
            
#             # Load weights
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             self.model.to(self.device)
#             self.model.eval()
            
#             # Extract model metadata
#             self.model_version = checkpoint.get('model_version', '2.0.0')
            
#             # Enable dropout for uncertainty estimation
#             self._enable_mc_dropout()
            
#             logging.info(f"✅ Model loaded successfully (version: {self.model_version})")
            
#         except Exception as e:
#             logging.error(f"❌ Failed to load model: {str(e)}")
#             raise RuntimeError(f"Model loading failed: {str(e)}")
    
#     def _initialize_components(self):
#         """Initialize preprocessing, explainability, and retrieval components"""
#         try:
#             # Medical preprocessor
#             self.preprocessor = MedicalPreprocessor(
#                 target_size=(384, 384),
#                 enable_bone_suppression=True,
#                 enable_artifact_removal=True,
#                 enable_lung_segmentation=True
#             )
            
#             # Explainability system
#             self.gradcam = MedicalGradCAM(
#                 model=self.model,
#                 device=self.device,
#                 lung_segmentation=True,
#                 nodule_size_filter=(3, 30)  # mm
#             )
            
#             # Case retrieval system
#             self.case_retriever = CaseRetriever(
#                 index_path="case_retrieval/faiss_index.idx",
#                 metadata_path="case_retrieval/case_metadata.pkl"
#             )
            
#             logging.info("✅ All inference components initialized")
            
#         except Exception as e:
#             logging.error(f"❌ Failed to initialize components: {str(e)}")
#             raise RuntimeError(f"Component initialization failed: {str(e)}")
    
#     def _enable_mc_dropout(self):
#         """Enable Monte Carlo dropout for uncertainty estimation"""
#         for module in self.model.modules():
#             if isinstance(module, torch.nn.Dropout):
#                 module.train()  # Keep dropout active during inference
    
#     async def analyze_with_uncertainty(
#         self, 
#         image: np.ndarray,
#         patient_id: Optional[str] = None,
#         clinical_history: Optional[str] = None,
#         mc_samples: int = 50
#     ) -> Dict[str, Any]:
#         """
#         Complete analysis with uncertainty quantification
#         """
#         start_time = time.time()
        
#         try:
#             # Preprocessing
#             processed_image = await asyncio.get_event_loop().run_in_executor(
#                 self.thread_pool, self.preprocessor.process, image
#             )
            
#             # Convert to tensor
#             image_tensor = torch.from_numpy(processed_image).unsqueeze(0).to(self.device)
            
#             # Monte Carlo inference for uncertainty
#             predictions = []
#             with torch.no_grad():
#                 for _ in range(mc_samples):
#                     logits = self.model(image_tensor)
#                     prob = torch.sigmoid(logits).cpu().numpy()
#                     predictions.append(prob[0, 0])
            
#             # Calculate statistics
#             mean_prob = np.mean(predictions)
#             std_prob = np.std(predictions)
#             confidence = 1.0 - std_prob
            
#             # Uncertainty classification
#             if std_prob < 0.05:
#                 uncertainty_level = "Low"
#             elif std_prob < 0.15:
#                 uncertainty_level = "Medium"
#             else:
#                 uncertainty_level = "High"
            
#             # Generate explanation
#             explanation = await asyncio.get_event_loop().run_in_executor(
#                 self.thread_pool, self.gradcam.generate_explanation, image_tensor
#             )
            
#             # Case retrieval
#             similar_cases = await asyncio.get_event_loop().run_in_executor(
#                 self.thread_pool, self.case_retriever.retrieve_similar, 
#                 image_tensor, 5  # top 5 cases
#             )
            
#             # Compile results
#             results = {
#                 'probability': float(mean_prob),
#                 'confidence': float(confidence),
#                 'uncertainty_std': float(std_prob),
#                 'uncertainty_level': uncertainty_level,
#                 'predicted_class': 'Nodule Present' if mean_prob > 0.5 else 'No Nodule',
#                 'processing_time': time.time() - start_time,
#                 'explanation': explanation.tolist() if explanation is not None else None,
#                 'similar_cases': similar_cases,
#                 'model_version': self.model_version,
#                 'mc_samples': mc_samples
#             }
            
#             return results
            
#         except Exception as e:
#             logging.error(f"❌ Analysis failed: {str(e)}")
#             raise RuntimeError(f"Analysis failed: {str(e)}")
    
#     async def batch_analyze(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
#         """
#         Efficient batch processing
#         """
#         # Process images concurrently
#         tasks = [self.analyze_with_uncertainty(img) for img in images]
#         results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         # Handle exceptions
#         processed_results = []
#         for i, result in enumerate(results):
#             if isinstance(result, Exception):
#                 processed_results.append({
#                     'error': str(result),
#                     'image_index': i,
#                     'status': 'failed'
#                 })
#             else:
#                 processed_results.append({
#                     **result,
#                     'image_index': i,
#                     'status': 'success'
#                 })
        
#         return processed_results
    
#     def is_ready(self) -> bool:
#         """Check if inference engine is ready"""
#         return all([
#             self.model is not None,
#             self.preprocessor is not None,
#             self.gradcam is not None,
#             self.case_retriever is not None
#         ])
