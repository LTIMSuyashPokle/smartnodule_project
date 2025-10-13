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