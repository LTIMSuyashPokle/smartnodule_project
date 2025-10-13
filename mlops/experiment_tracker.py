# ========================================================================
# 3. MLOPS INTEGRATION WITH MLFLOW
# ========================================================================

# mlops/experiment_tracker.py
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging
from typing import Dict, Any, Optional, List
import json
import pandas as pd
import numpy as np
from datetime import datetime
import torch

class MLflowTracker:
    """
    Comprehensive MLflow integration for experiment tracking
    """
    
    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.client = MlflowClient(tracking_uri)
        
        # Setup MLflow
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            
            self.experiment_id = experiment_id
            mlflow.set_experiment(experiment_name)
            
            logging.info(f"✅ MLflow tracking initialized (experiment: {experiment_name})")
            
        except Exception as e:
            logging.error(f"❌ MLflow initialization failed: {str(e)}")
            raise
    
    def log_training_run(
        self,
        model: torch.nn.Module,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        model_path: str,
        run_name: Optional[str] = None
    ) -> str:
        """
        Log complete training run
        """
        with mlflow.start_run(run_name=run_name) as run:
            # Log hyperparameters
            for key, value in hyperparameters.items():
                mlflow.log_param(key, value)
            
            # Log training metrics
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train_{key}", value)
            
            # Log validation metrics
            for key, value in val_metrics.items():
                mlflow.log_metric(f"val_{key}", value)
            
            # Log test metrics
            for key, value in test_metrics.items():
                mlflow.log_metric(f"test_{key}", value)
            
            # Log model
            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name="SmartNodule"
            )
            
            # Log model file
            mlflow.log_artifact(model_path, "checkpoints")
            
            # Log system info
            mlflow.log_param("pytorch_version", torch.__version__)
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
            run_id = run.info.run_id
            logging.info(f"✅ Training run logged (run_id: {run_id})")
            
            return run_id
    
    def log_prediction(
        self,
        request_id: str,
        prediction: Dict[str, Any],
        processing_time: float
    ):
        """
        Log individual prediction for monitoring
        """
        with mlflow.start_run(run_name=f"prediction_{request_id}") as run:
            # Log prediction metrics
            mlflow.log_metric("probability", prediction['probability'])
            mlflow.log_metric("confidence", prediction['confidence'])
            mlflow.log_metric("uncertainty_std", prediction['uncertainty_std'])
            mlflow.log_metric("processing_time", processing_time)
            
            # Log prediction details
            mlflow.log_param("predicted_class", prediction['predicted_class'])
            mlflow.log_param("uncertainty_level", prediction['uncertainty_level'])
            mlflow.log_param("model_version", prediction['model_version'])
            mlflow.log_param("request_id", request_id)
            mlflow.log_param("timestamp", datetime.now().isoformat())
    
    def log_performance_metrics(self, metrics: Dict[str, float], date: str):
        """
        Log daily performance aggregations
        """
        with mlflow.start_run(run_name=f"daily_performance_{date}") as run:
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            mlflow.log_param("date", date)
            mlflow.log_param("metric_type", "daily_aggregation")
    
    def get_best_model(self, metric_name: str = "val_accuracy") -> Dict[str, Any]:
        """
        Retrieve best performing model
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric_name} DESC"],
                max_results=1
            )
            
            if len(runs) > 0:
                best_run = runs.iloc[0]
                return {
                    'run_id': best_run['run_id'],
                    'metrics': {col.replace('metrics.', ''): best_run[col] 
                              for col in best_run.index if col.startswith('metrics.')},
                    'params': {col.replace('params.', ''): best_run[col] 
                             for col in best_run.index if col.startswith('params.')},
                    'model_uri': f"runs:/{best_run['run_id']}/model"
                }
            else:
                return None
                
        except Exception as e:
            logging.error(f"❌ Failed to get best model: {str(e)}")
            return None
    
    def compare_models(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple model runs
        """
        try:
            runs_data = []
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                run_data = {
                    'run_id': run_id,
                    'start_time': run.info.start_time,
                    'status': run.info.status,
                    **run.data.metrics,
                    **run.data.params
                }
                runs_data.append(run_data)
            
            return pd.DataFrame(runs_data)
            
        except Exception as e:
            logging.error(f"❌ Failed to compare models: {str(e)}")
            return pd.DataFrame()
    
    def is_connected(self) -> bool:
        """Check MLflow connection"""
        try:
            self.client.list_experiments()
            return True
        except:
            return False
