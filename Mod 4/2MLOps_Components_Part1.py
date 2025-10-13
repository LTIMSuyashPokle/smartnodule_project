# SMARTNODULE MODULE 4: MISSING IMPLEMENTATION FILES
# Complete Implementation of All Missing Components
# ========================================================================
# MLOPS COMPONENTS
# ========================================================================

# mlops/model_registry.py
#MODEL_REGISTRY = '''
"""
Advanced Model Registry for SmartNodule with versioning and deployment management
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import json
from datetime import datetime
from pathlib import Path
import shutil
import hashlib

class SmartNoduleModelRegistry:
    """
    Production-grade model registry with versioning, staging, and deployment management
    """
    
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db", model_name: str = "SmartNodule"):
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.client = MlflowClient(tracking_uri)
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create registered model if it doesn't exist
        try:
            self.client.create_registered_model(model_name)
            logging.info(f"✅ Created new registered model: {model_name}")
        except Exception:
            logging.info(f"✅ Using existing registered model: {model_name}")
    
    def register_model(
        self, 
        run_id: str,
        model_path: str,
        model_metrics: Dict[str, float],
        model_metadata: Dict[str, Any],
        stage: str = "Staging"
    ) -> str:
        """
        Register a new model version
        """
        try:
            # Create model version from run
            model_version = mlflow.register_model(
                f"runs:/{run_id}/model",
                self.model_name,
                description=f"SmartNodule v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Add metrics and metadata as tags
            for metric, value in model_metrics.items():
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=model_version.version,
                    key=f"metric_{metric}",
                    value=str(value)
                )
            
            for key, value in model_metadata.items():
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=model_version.version,
                    key=f"metadata_{key}",
                    value=str(value)
                )
            
            # Copy model file to registry
            registry_path = f"model_registry/{self.model_name}/v{model_version.version}"
            Path(registry_path).mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_path, f"{registry_path}/model.pth")
            
            # Calculate model checksum for integrity
            checksum = self._calculate_checksum(model_path)
            self.client.set_model_version_tag(
                name=self.model_name,
                version=model_version.version,
                key="checksum",
                value=checksum
            )
            
            # Set stage
            self.transition_model_stage(model_version.version, stage)
            
            logging.info(f"✅ Registered model version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logging.error(f"❌ Model registration failed: {str(e)}")
            raise
    
    def transition_model_stage(self, version: str, stage: str) -> bool:
        """
        Transition model to different stage (Staging, Production, Archived)
        """
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage
            )
            
            logging.info(f"✅ Model v{version} transitioned to {stage}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Stage transition failed: {str(e)}")
            return False
    
    def get_production_model(self) -> Optional[Dict[str, Any]]:
        """
        Get current production model information
        """
        try:
            versions = self.client.get_latest_versions(
                name=self.model_name,
                stages=["Production"]
            )
            
            if versions:
                version = versions[0]
                return {
                    "version": version.version,
                    "run_id": version.run_id,
                    "status": version.status,
                    "tags": version.tags,
                    "creation_timestamp": version.creation_timestamp,
                    "model_uri": f"models:/{self.model_name}/{version.version}"
                }
            else:
                return None
                
        except Exception as e:
            logging.error(f"❌ Failed to get production model: {str(e)}")
            return None
    
    def compare_models(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions
        """
        try:
            v1 = self.client.get_model_version(self.model_name, version1)
            v2 = self.client.get_model_version(self.model_name, version2)
            
            # Extract metrics from tags
            v1_metrics = {k.replace('metric_', ''): float(v) for k, v in v1.tags.items() if k.startswith('metric_')}
            v2_metrics = {k.replace('metric_', ''): float(v) for k, v in v2.tags.items() if k.startswith('metric_')}
            
            comparison = {
                "version_1": {
                    "version": version1,
                    "metrics": v1_metrics,
                    "stage": v1.current_stage,
                    "created": v1.creation_timestamp
                },
                "version_2": {
                    "version": version2,
                    "metrics": v2_metrics,
                    "stage": v2.current_stage,
                    "created": v2.creation_timestamp
                },
                "improvements": {}
            }
            
            # Calculate improvements
            for metric in v1_metrics:
                if metric in v2_metrics:
                    improvement = v2_metrics[metric] - v1_metrics[metric]
                    comparison["improvements"][metric] = improvement
            
            return comparison
            
        except Exception as e:
            logging.error(f"❌ Model comparison failed: {str(e)}")
            return {}
    
    def list_models(self, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all model versions, optionally filtered by stage
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(self.model_name, stages=[stage])
            else:
                versions = self.client.search_model_versions(f"name='{self.model_name}'")
            
            models = []
            for version in versions:
                model_info = {
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "status": version.status,
                    "creation_timestamp": version.creation_timestamp,
                    "tags": version.tags
                }
                models.append(model_info)
            
            return sorted(models, key=lambda x: int(x["version"]), reverse=True)
            
        except Exception as e:
            logging.error(f"❌ Failed to list models: {str(e)}")
            return []
    
    def rollback_to_previous_production(self) -> bool:
        """
        Rollback to previous production model in case of issues
        """
        try:
            # Get all production versions
            all_versions = self.client.search_model_versions(
                f"name='{self.model_name}' and run_id != ''"
            )
            
            # Find previous production model
            production_versions = [v for v in all_versions if v.current_stage == "Production"]
            
            if len(production_versions) < 2:
                logging.warning("⚠️ No previous production model to rollback to")
                return False
            
            # Sort by creation time
            production_versions.sort(key=lambda x: x.creation_timestamp, reverse=True)
            
            current_prod = production_versions[0]
            previous_prod = production_versions[1]
            
            # Archive current and activate previous
            self.transition_model_stage(current_prod.version, "Archived")
            self.transition_model_stage(previous_prod.version, "Production")
            
            logging.info(f"✅ Rolled back from v{current_prod.version} to v{previous_prod.version}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Rollback failed: {str(e)}")
            return False
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of model file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def validate_model_integrity(self, version: str) -> bool:
        """Validate model file integrity using checksum"""
        try:
            model_info = self.client.get_model_version(self.model_name, version)
            stored_checksum = model_info.tags.get("checksum")
            
            if not stored_checksum:
                logging.warning(f"⚠️ No checksum found for model v{version}")
                return True  # Allow if no checksum stored
            
            # Get model file path
            model_path = f"model_registry/{self.model_name}/v{version}/model.pth"
            
            if not Path(model_path).exists():
                logging.error(f"❌ Model file not found: {model_path}")
                return False
            
            current_checksum = self._calculate_checksum(model_path)
            
            if current_checksum == stored_checksum:
                logging.info(f"✅ Model v{version} integrity verified")
                return True
            else:
                logging.error(f"❌ Model v{version} integrity check failed!")
                return False
                
        except Exception as e:
            logging.error(f"❌ Integrity validation failed: {str(e)}")
            return False
#'''

# ========================================================================
# mlops/performance_monitor.py
# ========================================================================

#PERFORMANCE_MONITOR = '''
"""
Comprehensive Performance Monitoring with Model Drift Detection
"""

import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
import psutil
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import threading
import time

class PerformanceMonitor:
    """
    Advanced performance monitoring with drift detection and alerting
    """
    
    def __init__(self, metrics_db: str = "performance_metrics.db", window_size: int = 1000):
        self.metrics_db = metrics_db
        self.window_size = window_size
        self.baseline_metrics = {}
        self.alert_thresholds = {
            'accuracy_drop': 0.05,
            'drift_score': 0.1,
            'latency_increase': 2.0,
            'memory_usage': 0.85
        }
        
        self._initialize_database()
        self._load_baseline_metrics()
        
        logging.info("✅ Performance monitor initialized")
    
    def _initialize_database(self):
        """Initialize performance metrics database"""
        try:
            with sqlite3.connect(self.metrics_db) as conn:
                cursor = conn.cursor()
                
                # Prediction metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS prediction_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        request_id TEXT,
                        probability REAL,
                        confidence REAL,
                        uncertainty_std REAL,
                        uncertainty_level TEXT,
                        processing_time REAL,
                        model_version TEXT,
                        ground_truth INTEGER,
                        is_correct INTEGER
                    )
                ''')
                
                # System metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        cpu_usage REAL,
                        memory_usage REAL,
                        gpu_usage REAL,
                        active_requests INTEGER,
                        avg_response_time REAL
                    )
                ''')
                
                # Performance alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        alert_type TEXT,
                        severity TEXT,
                        message TEXT,
                        metrics TEXT,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                # Model drift table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_drift (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        drift_type TEXT,
                        drift_score REAL,
                        reference_window TEXT,
                        current_window TEXT,
                        p_value REAL,
                        significant_drift BOOLEAN
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Database initialization failed: {str(e)}")
            raise
    
    def log_inference(
        self, 
        request_id: str,
        processing_time: float,
        prediction_confidence: float,
        uncertainty_level: str,
        probability: float = None,
        uncertainty_std: float = None,
        model_version: str = "2.0.0",
        ground_truth: Optional[int] = None
    ):
        """Log individual inference metrics"""
        try:
            timestamp = datetime.now().isoformat()
            is_correct = None
            
            if ground_truth is not None and probability is not None:
                predicted = 1 if probability > 0.5 else 0
                is_correct = 1 if predicted == ground_truth else 0
            
            with sqlite3.connect(self.metrics_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO prediction_metrics 
                    (timestamp, request_id, probability, confidence, uncertainty_std,
                     uncertainty_level, processing_time, model_version, ground_truth, is_correct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp, request_id, probability, prediction_confidence,
                    uncertainty_std, uncertainty_level, processing_time,
                    model_version, ground_truth, is_correct
                ))
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Failed to log inference metrics: {str(e)}")
    
    def log_system_metrics(self):
        """Log current system performance metrics"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            # GPU usage (if available)
            gpu_usage = None
            if torch.cuda.is_available():
                try:
                    gpu_usage = torch.cuda.utilization() / 100.0
                except:
                    gpu_usage = None
            
            # Calculate average response time from recent predictions
            avg_response_time = self._get_recent_avg_response_time()
            active_requests = self._get_active_requests_count()
            
            with sqlite3.connect(self.metrics_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_metrics 
                    (timestamp, cpu_usage, memory_usage, gpu_usage, 
                     active_requests, avg_response_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp, cpu_usage, memory_usage, gpu_usage,
                    active_requests, avg_response_time
                ))
                conn.commit()
            
            # Check for performance alerts
            self._check_system_alerts(cpu_usage, memory_usage, gpu_usage, avg_response_time)
                
        except Exception as e:
            logging.error(f"❌ Failed to log system metrics: {str(e)}")
    
    def detect_model_drift(self, window_days: int = 7) -> Dict[str, Any]:
        """
        Detect model drift using statistical tests
        """
        try:
            # Get recent predictions
            end_date = datetime.now()
            start_date = end_date - timedelta(days=window_days)
            
            with sqlite3.connect(self.metrics_db) as conn:
                # Current window data
                current_df = pd.read_sql_query('''
                    SELECT probability, confidence, uncertainty_std, processing_time
                    FROM prediction_metrics 
                    WHERE timestamp >= ? AND timestamp <= ?
                ''', conn, params=(start_date.isoformat(), end_date.isoformat()))
                
                # Reference window data (previous period)
                ref_start = start_date - timedelta(days=window_days)
                ref_end = start_date
                
                reference_df = pd.read_sql_query('''
                    SELECT probability, confidence, uncertainty_std, processing_time
                    FROM prediction_metrics 
                    WHERE timestamp >= ? AND timestamp <= ?
                ''', conn, params=(ref_start.isoformat(), ref_end.isoformat()))
            
            if len(current_df) < 50 or len(reference_df) < 50:
                logging.warning("⚠️ Insufficient data for drift detection")
                return {"drift_detected": False, "reason": "insufficient_data"}
            
            # Perform drift tests
            drift_results = {}
            
            for column in ['probability', 'confidence', 'uncertainty_std']:
                if column in current_df.columns and column in reference_df.columns:
                    # Kolmogorov-Smirnov test
                    ks_stat, p_value = stats.ks_2samp(
                        reference_df[column].dropna(),
                        current_df[column].dropna()
                    )
                    
                    # Population Stability Index (PSI)
                    psi_score = self._calculate_psi(
                        reference_df[column].dropna(),
                        current_df[column].dropna()
                    )
                    
                    drift_results[column] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'psi_score': psi_score,
                        'drift_detected': p_value < 0.05 or psi_score > 0.1
                    }
            
            # Overall drift assessment
            overall_drift = any(result['drift_detected'] for result in drift_results.values())
            
            # Log drift results
            self._log_drift_results(drift_results, overall_drift)
            
            return {
                'drift_detected': overall_drift,
                'drift_results': drift_results,
                'window_days': window_days,
                'current_samples': len(current_df),
                'reference_samples': len(reference_df)
            }
            
        except Exception as e:
            logging.error(f"❌ Drift detection failed: {str(e)}")
            return {"drift_detected": False, "error": str(e)}
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics summary"""
        try:
            with sqlite3.connect(self.metrics_db) as conn:
                # Get recent performance data
                df = pd.read_sql_query('''
                    SELECT * FROM prediction_metrics 
                    WHERE timestamp >= datetime('now', '-1 day')
                ''', conn)
                
                if len(df) == 0:
                    return {"error": "No recent data available"}
                
                # Calculate metrics
                metrics = {
                    "accuracy": float(df['is_correct'].mean()) if 'is_correct' in df and df['is_correct'].notna().any() else None,
                    "avg_confidence": float(df['confidence'].mean()),
                    "avg_processing_time": float(df['processing_time'].mean()),
                    "total_predictions": len(df),
                    "high_uncertainty_rate": float((df['uncertainty_level'] == 'High').mean()),
                    "last_updated": datetime.now().isoformat()
                }
                
                # Add system metrics
                system_df = pd.read_sql_query('''
                    SELECT * FROM system_metrics 
                    WHERE timestamp >= datetime('now', '-1 hour')
                    ORDER BY timestamp DESC LIMIT 1
                ''', conn)
                
                if len(system_df) > 0:
                    latest_system = system_df.iloc[0]
                    metrics.update({
                        "cpu_usage": float(latest_system['cpu_usage']),
                        "memory_usage": float(latest_system['memory_usage']),
                        "gpu_usage": float(latest_system['gpu_usage']) if latest_system['gpu_usage'] else None
                    })
                
                return metrics
                
        except Exception as e:
            logging.error(f"❌ Failed to get current metrics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins based on reference data
            bin_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
            
            # Ensure unique bin edges
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                return 0.0
            
            # Calculate distributions
            ref_dist = pd.cut(reference, bins=bin_edges, include_lowest=True, duplicates='drop').value_counts(normalize=True)
            cur_dist = pd.cut(current, bins=bin_edges, include_lowest=True, duplicates='drop').value_counts(normalize=True)
            
            # Align distributions
            ref_dist = ref_dist.reindex(cur_dist.index, fill_value=0.001)  # Small value to avoid log(0)
            cur_dist = cur_dist.reindex(ref_dist.index, fill_value=0.001)
            
            # Calculate PSI
            psi = ((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum()
            
            return float(psi)
            
        except Exception as e:
            logging.error(f"❌ PSI calculation failed: {str(e)}")
            return 0.0
    
    def _get_recent_avg_response_time(self, minutes: int = 5) -> float:
        """Get average response time from recent predictions"""
        try:
            with sqlite3.connect(self.metrics_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT AVG(processing_time) as avg_time
                    FROM prediction_metrics 
                    WHERE timestamp >= datetime('now', '-{} minutes')
                '''.format(minutes))
                
                result = cursor.fetchone()
                return float(result[0]) if result[0] else 0.0
                
        except Exception as e:
            logging.error(f"❌ Failed to get avg response time: {str(e)}")
            return 0.0
    
    def _get_active_requests_count(self) -> int:
        """Get count of active/recent requests"""
        try:
            with sqlite3.connect(self.metrics_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM prediction_metrics 
                    WHERE timestamp >= datetime('now', '-1 minute')
                ''')
                
                result = cursor.fetchone()
                return int(result[0]) if result[0] else 0
                
        except Exception as e:
            logging.error(f"❌ Failed to get active requests: {str(e)}")
            return 0
    
    def _check_system_alerts(self, cpu: float, memory: float, gpu: Optional[float], response_time: float):
        """Check for system performance alerts"""
        alerts = []
        
        # Memory usage alert
        if memory > self.alert_thresholds['memory_usage']:
            alerts.append({
                'type': 'high_memory_usage',
                'severity': 'high',
                'message': f"Memory usage critical: {memory:.1%}",
                'value': memory
            })
        
        # Response time alert
        baseline_response_time = self.baseline_metrics.get('avg_processing_time', 5.0)
        if response_time > baseline_response_time * self.alert_thresholds['latency_increase']:
            alerts.append({
                'type': 'high_latency',
                'severity': 'medium',
                'message': f"Response time increased: {response_time:.2f}s vs {baseline_response_time:.2f}s baseline",
                'value': response_time
            })
        
        # Log alerts
        for alert in alerts:
            self._log_alert(alert)
    
    def _log_alert(self, alert: Dict[str, Any]):
        """Log performance alert"""
        try:
            with sqlite3.connect(self.metrics_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_alerts 
                    (timestamp, alert_type, severity, message, metrics)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    alert['type'],
                    alert['severity'],
                    alert['message'],
                    json.dumps(alert)
                ))
                conn.commit()
                
            logging.warning(f"⚠️ Performance Alert: {alert['message']}")
            
        except Exception as e:
            logging.error(f"❌ Failed to log alert: {str(e)}")
    
    def _log_drift_results(self, drift_results: Dict[str, Any], overall_drift: bool):
        """Log drift detection results"""
        try:
            with sqlite3.connect(self.metrics_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO model_drift 
                    (timestamp, drift_type, drift_score, significant_drift)
                    VALUES (?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    'overall',
                    max([r.get('psi_score', 0) for r in drift_results.values()]),
                    overall_drift
                ))
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Failed to log drift results: {str(e)}")
    
    def _load_baseline_metrics(self):
        """Load baseline metrics for comparison"""
        try:
            with sqlite3.connect(self.metrics_db) as conn:
                df = pd.read_sql_query('''
                    SELECT AVG(processing_time) as avg_processing_time,
                           AVG(confidence) as avg_confidence
                    FROM prediction_metrics 
                    WHERE timestamp >= datetime('now', '-30 days')
                ''', conn)
                
                if len(df) > 0 and df.iloc[0]['avg_processing_time']:
                    self.baseline_metrics = {
                        'avg_processing_time': float(df.iloc[0]['avg_processing_time']),
                        'avg_confidence': float(df.iloc[0]['avg_confidence'])
                    }
                else:
                    # Default baseline metrics
                    self.baseline_metrics = {
                        'avg_processing_time': 5.0,
                        'avg_confidence': 0.8
                    }
                    
        except Exception as e:
            logging.warning(f"⚠️ Failed to load baseline metrics: {str(e)}")
            self.baseline_metrics = {
                'avg_processing_time': 5.0,
                'avg_confidence': 0.8
            }
    
    def get_uptime(self) -> str:
        """Get system uptime"""
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            return str(uptime).split('.')[0]  # Remove microseconds
        except:
            return "Unknown"
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    def get_gpu_usage(self) -> Optional[float]:
        """Get current GPU usage percentage"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization()
            return None
        except:
            return None
#'''

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
#'''

print("✅ Created comprehensive MLOps components:")
print("   - model_registry.py: Advanced model versioning and deployment management") 
print("   - performance_monitor.py: Model drift detection and performance tracking")
print("   - data_validator.py: Comprehensive medical image quality validation")
print("\nNext, I'll create the Active Learning components...")