# ========================================================================
# mlops/data_validator.py
# ========================================================================

#DATA_VALIDATOR = '''
"""
Comprehensive Data Quality Validation for Medical Images
"""

import numpy as np
import cv2
from PIL import Image
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
from datetime import datetime
import json
import sqlite3
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy import stats
import hashlib

class MedicalDataValidator:
    """
    Advanced data quality validation for medical imaging data
    """
    
    def __init__(self, validation_db: str = "data_validation.db"):
        self.validation_db = validation_db
        self.quality_standards = {
            'min_resolution': (256, 256),
            'max_resolution': (2048, 2048),
            'min_contrast': 0.1,
            'max_noise_level': 0.3,
            'required_bit_depth': 8,
            'accepted_formats': ['PNG', 'JPEG', 'TIFF', 'DICOM']
        }
        
        self._initialize_validation_db()
        logging.info("✅ Medical data validator initialized")
    
    def _initialize_validation_db(self):
        """Initialize validation results database"""
        try:
            with sqlite3.connect(self.validation_db) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS validation_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        file_path TEXT,
                        file_hash TEXT,
                        validation_type TEXT,
                        passed BOOLEAN,
                        quality_score REAL,
                        issues TEXT,
                        metrics TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS data_drift_detection (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        batch_id TEXT,
                        drift_score REAL,
                        drift_detected BOOLEAN,
                        reference_stats TEXT,
                        current_stats TEXT
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Validation DB initialization failed: {str(e)}")
            raise
    
    def validate_medical_image(
        self, 
        image_data: Union[np.ndarray, str, Path],
        image_type: str = "chest_xray"
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of medical image quality
        """
        try:
            # Load image if path provided
            if isinstance(image_data, (str, Path)):
                image = cv2.imread(str(image_data))
                if image is None:
                    return self._create_validation_result(False, 0.0, ["Failed to load image"])
                file_path = str(image_data)
                file_hash = self._calculate_file_hash(file_path)
            else:
                image = image_data
                file_path = "in_memory"
                file_hash = self._calculate_array_hash(image)
            
            # Initialize validation results
            issues = []
            quality_metrics = {}
            quality_score = 0.0
            
            # 1. Basic format validation
            format_result = self._validate_format(image)
            quality_metrics['format'] = format_result
            if not format_result['valid']:
                issues.extend(format_result['issues'])
            else:
                quality_score += 15
            
            # 2. Resolution validation
            resolution_result = self._validate_resolution(image)
            quality_metrics['resolution'] = resolution_result
            if not resolution_result['valid']:
                issues.extend(resolution_result['issues'])
            else:
                quality_score += 15
            
            # 3. Image quality assessment
            quality_result = self._assess_image_quality(image)
            quality_metrics['quality'] = quality_result
            quality_score += quality_result['score']
            if quality_result['score'] < 30:
                issues.append(f"Poor image quality (score: {quality_result['score']:.1f}/50)")
            
            # 4. Medical-specific validation
            if image_type == "chest_xray":
                medical_result = self._validate_chest_xray(image)
                quality_metrics['medical'] = medical_result
                quality_score += medical_result['score']
                if medical_result['score'] < 15:
                    issues.extend(medical_result['issues'])
            
            # 5. Artifact detection
            artifact_result = self._detect_artifacts(image)
            quality_metrics['artifacts'] = artifact_result
            if artifact_result['artifacts_detected']:
                issues.extend(artifact_result['issues'])
            else:
                quality_score += 5
            
            # Final validation result
            passed = len(issues) == 0 and quality_score >= 70
            
            validation_result = self._create_validation_result(
                passed, quality_score, issues, quality_metrics
            )
            
            # Log validation result
            self._log_validation_result(
                file_path, file_hash, "medical_image", 
                passed, quality_score, issues, quality_metrics
            )
            
            return validation_result
            
        except Exception as e:
            logging.error(f"❌ Image validation failed: {str(e)}")
            return self._create_validation_result(False, 0.0, [f"Validation error: {str(e)}"])
    
    def validate_batch_consistency(
        self, 
        image_batch: List[np.ndarray],
        metadata_batch: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Validate consistency across a batch of medical images
        """
        try:
            if len(image_batch) == 0:
                return {"valid": False, "issues": ["Empty batch"]}
            
            issues = []
            batch_metrics = {}
            
            # Extract image statistics
            image_stats = []
            for i, image in enumerate(image_batch):
                stats = self._extract_image_statistics(image)
                image_stats.append(stats)
            
            # Check consistency
            consistency_result = self._check_batch_consistency(image_stats)
            batch_metrics['consistency'] = consistency_result
            
            if not consistency_result['consistent']:
                issues.extend(consistency_result['issues'])
            
            # Check for data drift if reference available
            drift_result = self._detect_batch_drift(image_stats)
            batch_metrics['drift'] = drift_result
            
            if drift_result['drift_detected']:
                issues.append(f"Data drift detected (score: {drift_result['drift_score']:.3f})")
            
            # Metadata validation if provided
            if metadata_batch:
                metadata_result = self._validate_metadata_batch(metadata_batch)
                batch_metrics['metadata'] = metadata_result
                if not metadata_result['valid']:
                    issues.extend(metadata_result['issues'])
            
            batch_valid = len(issues) == 0
            
            # Log batch validation
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._log_batch_validation(batch_id, batch_valid, issues, batch_metrics)
            
            return {
                "valid": batch_valid,
                "batch_size": len(image_batch),
                "issues": issues,
                "metrics": batch_metrics,
                "batch_id": batch_id
            }
            
        except Exception as e:
            logging.error(f"❌ Batch validation failed: {str(e)}")
            return {"valid": False, "issues": [f"Batch validation error: {str(e)}"]}
    
    def _validate_format(self, image: np.ndarray) -> Dict[str, Any]:
        """Validate image format and basic properties"""
        try:
            if image is None:
                return {"valid": False, "issues": ["Image is None"]}
            
            if len(image.shape) < 2:
                return {"valid": False, "issues": ["Invalid image dimensions"]}
            
            # Check bit depth
            if image.dtype not in [np.uint8, np.uint16]:
                return {"valid": False, "issues": ["Unsupported bit depth"]}
            
            # Check channels
            if len(image.shape) == 3 and image.shape[2] not in [1, 3]:
                return {"valid": False, "issues": ["Invalid number of channels"]}
            
            return {
                "valid": True,
                "shape": image.shape,
                "dtype": str(image.dtype),
                "channels": image.shape[2] if len(image.shape) == 3 else 1
            }
            
        except Exception as e:
            return {"valid": False, "issues": [f"Format validation error: {str(e)}"]}
    
    def _validate_resolution(self, image: np.ndarray) -> Dict[str, Any]:
        """Validate image resolution"""
        try:
            height, width = image.shape[:2]
            min_h, min_w = self.quality_standards['min_resolution']
            max_h, max_w = self.quality_standards['max_resolution']
            
            issues = []
            
            if height < min_h or width < min_w:
                issues.append(f"Resolution too low: {width}x{height}, minimum: {min_w}x{min_h}")
            
            if height > max_h or width > max_w:
                issues.append(f"Resolution too high: {width}x{height}, maximum: {max_w}x{max_h}")
            
            return {
                "valid": len(issues) == 0,
                "resolution": (width, height),
                "issues": issues
            }
            
        except Exception as e:
            return {"valid": False, "issues": [f"Resolution validation error: {str(e)}"]}
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Comprehensive image quality assessment"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            quality_score = 0.0
            metrics = {}
            
            # 1. Contrast assessment (0-15 points)
            contrast = gray.std() / 255.0
            metrics['contrast'] = float(contrast)
            if contrast >= 0.2:
                quality_score += 15
            elif contrast >= 0.1:
                quality_score += 10
            else:
                quality_score += 5
            
            # 2. Sharpness assessment (0-15 points)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = laplacian_var / 1000.0  # Normalize
            metrics['sharpness'] = float(sharpness)
            if sharpness >= 50:
                quality_score += 15
            elif sharpness >= 20:
                quality_score += 10
            else:
                quality_score += 5
            
            # 3. Noise assessment (0-10 points)
            noise_level = self._estimate_noise_level(gray)
            metrics['noise_level'] = float(noise_level)
            if noise_level <= 0.1:
                quality_score += 10
            elif noise_level <= 0.2:
                quality_score += 7
            else:
                quality_score += 3
            
            # 4. Dynamic range (0-10 points)
            dynamic_range = (gray.max() - gray.min()) / 255.0
            metrics['dynamic_range'] = float(dynamic_range)
            if dynamic_range >= 0.8:
                quality_score += 10
            elif dynamic_range >= 0.6:
                quality_score += 7
            else:
                quality_score += 4
            
            return {
                "score": quality_score,
                "max_score": 50,
                "metrics": metrics
            }
            
        except Exception as e:
            logging.error(f"❌ Quality assessment failed: {str(e)}")
            return {"score": 0.0, "max_score": 50, "metrics": {}}
    
    def _validate_chest_xray(self, image: np.ndarray) -> Dict[str, Any]:
        """Medical-specific validation for chest X-rays"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            score = 0.0
            issues = []
            
            # 1. Check for lung regions (basic shape analysis)
            lung_regions = self._detect_lung_regions(gray)
            if lung_regions['detected']:
                score += 10
            else:
                issues.append("Lung regions not clearly visible")
            
            # 2. Check orientation (portrait expected for chest X-ray)
            height, width = gray.shape
            if height > width:
                score += 5
            else:
                issues.append("Unexpected image orientation (landscape)")
            
            # 3. Check for proper centering
            center_quality = self._assess_centering(gray)
            if center_quality > 0.7:
                score += 5
            elif center_quality > 0.5:
                score += 3
            else:
                issues.append("Poor image centering")
            
            return {
                "score": score,
                "max_score": 20,
                "issues": issues,
                "lung_regions": lung_regions,
                "center_quality": center_quality
            }
            
        except Exception as e:
            logging.error(f"❌ Chest X-ray validation failed: {str(e)}")
            return {"score": 0.0, "max_score": 20, "issues": [f"Medical validation error: {str(e)}"]}
    
    def _detect_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect common imaging artifacts"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            artifacts = []
            artifact_scores = {}
            
            # 1. Grid artifact detection (FFT-based)
            grid_detected = self._detect_grid_artifact(gray)
            artifact_scores['grid'] = grid_detected
            if grid_detected > 0.3:
                artifacts.append("Grid artifact detected")
            
            # 2. Motion blur detection
            motion_blur = self._detect_motion_blur(gray)
            artifact_scores['motion_blur'] = motion_blur
            if motion_blur > 0.4:
                artifacts.append("Motion blur detected")
            
            # 3. Saturation detection
            saturation = self._detect_saturation(gray)
            artifact_scores['saturation'] = saturation
            if saturation > 0.1:
                artifacts.append("Image saturation detected")
            
            # 4. Foreign object detection (basic)
            foreign_objects = self._detect_foreign_objects(gray)
            artifact_scores['foreign_objects'] = foreign_objects
            if foreign_objects > 0.2:
                artifacts.append("Potential foreign objects detected")
            
            return {
                "artifacts_detected": len(artifacts) > 0,
                "artifacts": artifacts,
                "scores": artifact_scores,
                "issues": artifacts
            }
            
        except Exception as e:
            logging.error(f"❌ Artifact detection failed: {str(e)}")
            return {
                "artifacts_detected": False,
                "artifacts": [],
                "issues": [f"Artifact detection error: {str(e)}"]
            }
    
    def _extract_image_statistics(self, image: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from image"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            return {
                'mean': float(gray.mean()),
                'std': float(gray.std()),
                'min': float(gray.min()),
                'max': float(gray.max()),
                'median': float(np.median(gray)),
                'skewness': float(stats.skew(gray.flatten())),
                'kurtosis': float(stats.kurtosis(gray.flatten()))
            }
            
        except Exception as e:
            logging.error(f"❌ Failed to extract image statistics: {str(e)}")
            return {}
    
    def _check_batch_consistency(self, image_stats: List[Dict[str, float]]) -> Dict[str, Any]:
        """Check consistency across batch statistics"""
        try:
            if len(image_stats) < 2:
                return {"consistent": True, "issues": []}
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(image_stats)
            
            issues = []
            consistency_metrics = {}
            
            # Check coefficient of variation for each metric
            for metric in df.columns:
                cv = df[metric].std() / df[metric].mean()
                consistency_metrics[f'{metric}_cv'] = float(cv)
                
                # High variation thresholds
                if metric in ['mean', 'median'] and cv > 0.3:
                    issues.append(f"High variation in {metric} (CV: {cv:.3f})")
                elif metric in ['std'] and cv > 0.5:
                    issues.append(f"High variation in {metric} (CV: {cv:.3f})")
            
            return {
                "consistent": len(issues) == 0,
                "issues": issues,
                "metrics": consistency_metrics
            }
            
        except Exception as e:
            logging.error(f"❌ Batch consistency check failed: {str(e)}")
            return {"consistent": False, "issues": [f"Consistency check error: {str(e)}"]}
    
    def _detect_batch_drift(self, current_stats: List[Dict[str, float]]) -> Dict[str, Any]:
        """Detect data drift in current batch vs reference"""
        try:
            # This is a simplified implementation
            # In production, you would compare against stored reference statistics
            
            if len(current_stats) == 0:
                return {"drift_detected": False, "drift_score": 0.0}
            
            # For now, just check if current batch statistics are within expected ranges
            df = pd.DataFrame(current_stats)
            
            # Define expected ranges (these should be learned from reference data)
            expected_ranges = {
                'mean': (50, 200),
                'std': (20, 100),
                'skewness': (-1, 1),
                'kurtosis': (-2, 5)
            }
            
            drift_score = 0.0
            drift_detected = False
            
            for metric, (min_val, max_val) in expected_ranges.items():
                if metric in df.columns:
                    batch_mean = df[metric].mean()
                    if batch_mean < min_val or batch_mean > max_val:
                        drift_score += 0.25
                        drift_detected = True
            
            return {
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "batch_stats": df.mean().to_dict()
            }
            
        except Exception as e:
            logging.error(f"❌ Drift detection failed: {str(e)}")
            return {"drift_detected": False, "drift_score": 0.0}
    
    # Helper methods for specific validation tasks
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in image"""
        # Simplified noise estimation using high-frequency content
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(image.astype(np.float32), -1, kernel)
        noise_level = np.std(filtered) / 255.0
        return min(1.0, noise_level)
    
    def _detect_lung_regions(self, image: np.ndarray) -> Dict[str, Any]:
        """Basic lung region detection"""
        # Simplified implementation - would use more sophisticated methods in production
        # Apply threshold to find dark lung regions
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for large dark regions that could be lungs
        large_regions = [c for c in contours if cv2.contourArea(c) > image.size * 0.05]
        
        return {
            "detected": len(large_regions) >= 1,
            "regions_found": len(large_regions)
        }
    
    def _assess_centering(self, image: np.ndarray) -> float:
        """Assess if anatomical structures are properly centered"""
        # Simplified centering assessment
        height, width = image.shape
        center_region = image[height//4:3*height//4, width//4:3*width//4]
        
        # Higher intensity in center region suggests better centering
        center_intensity = center_region.mean()
        total_intensity = image.mean()
        
        centering_ratio = center_intensity / (total_intensity + 1e-6)
        return min(1.0, max(0.0, centering_ratio))
    
    def _detect_grid_artifact(self, image: np.ndarray) -> float:
        """Detect grid artifacts using FFT"""
        # Simplified grid detection
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Look for periodic patterns in frequency domain
        periodic_energy = np.std(magnitude_spectrum)
        return min(1.0, periodic_energy / 5.0)  # Normalize
    
    def _detect_motion_blur(self, image: np.ndarray) -> float:
        """Detect motion blur using variance of Laplacian"""
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        # Lower variance suggests more blur
        blur_score = max(0.0, 1.0 - laplacian_var / 1000.0)
        return min(1.0, blur_score)
    
    def _detect_saturation(self, image: np.ndarray) -> float:
        """Detect pixel saturation"""
        saturated_pixels = np.sum((image == 0) | (image == 255))
        saturation_ratio = saturated_pixels / image.size
        return min(1.0, saturation_ratio)
    
    def _detect_foreign_objects(self, image: np.ndarray) -> float:
        """Basic foreign object detection"""
        # Look for very bright regions that might be metal objects
        bright_threshold = np.percentile(image, 98)
        bright_pixels = np.sum(image > bright_threshold)
        bright_ratio = bright_pixels / image.size
        return min(1.0, bright_ratio * 10)  # Amplify for detection
    
    def _validate_metadata_batch(self, metadata_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate metadata consistency in batch"""
        issues = []
        
        # Check for required fields
        required_fields = ['patient_id', 'study_date', 'modality']
        for i, metadata in enumerate(metadata_batch):
            for field in required_fields:
                if field not in metadata:
                    issues.append(f"Missing {field} in image {i}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def _create_validation_result(
        self, 
        passed: bool, 
        quality_score: float, 
        issues: List[str],
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create standardized validation result"""
        return {
            "passed": passed,
            "quality_score": quality_score,
            "max_score": 100.0,
            "quality_grade": self._get_quality_grade(quality_score),
            "issues": issues,
            "issue_count": len(issues),
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _calculate_array_hash(self, array: np.ndarray) -> str:
        """Calculate hash of numpy array"""
        return hashlib.sha256(array.tobytes()).hexdigest()
    
    def _log_validation_result(
        self,
        file_path: str,
        file_hash: str,
        validation_type: str,
        passed: bool,
        quality_score: float,
        issues: List[str],
        metrics: Dict[str, Any]
    ):
        """Log validation result to database"""
        try:
            with sqlite3.connect(self.validation_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO validation_results 
                    (timestamp, file_path, file_hash, validation_type, 
                     passed, quality_score, issues, metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    file_path,
                    file_hash,
                    validation_type,
                    passed,
                    quality_score,
                    json.dumps(issues),
                    json.dumps(metrics)
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"❌ Failed to log validation result: {str(e)}")
    
    def _log_batch_validation(
        self,
        batch_id: str,
        valid: bool,
        issues: List[str],
        metrics: Dict[str, Any]
    ):
        """Log batch validation result"""
        try:
            drift_score = metrics.get('drift', {}).get('drift_score', 0.0)
            drift_detected = metrics.get('drift', {}).get('drift_detected', False)
            
            with sqlite3.connect(self.validation_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO data_drift_detection 
                    (timestamp, batch_id, drift_score, drift_detected, 
                     reference_stats, current_stats)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    batch_id,
                    drift_score,
                    drift_detected,
                    json.dumps({}),  # Would store reference stats
                    json.dumps(metrics)
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"❌ Failed to log batch validation: {str(e)}")