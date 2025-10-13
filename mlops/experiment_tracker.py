# ========================================================================
# 3. MLOPS INTEGRATION WITH MLFLOW
# ========================================================================

# mlops/experiment_tracker.py
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import logging
import os
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
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = "default"):
        # Ensure project-local mlflow artifacts and sqlite database are used to avoid readonly paths
        project_root = os.getcwd()
        local_db = os.path.join(project_root, 'mlflow.db')
        local_mlruns = os.path.join(project_root, 'mlruns')
        os.makedirs(local_mlruns, exist_ok=True)

        # If no tracking_uri provided, use project-local sqlite
        if not tracking_uri:
            tracking_uri = f"sqlite:///{local_db}"

        # Normalize to absolute sqlite tracking URI if sqlite provided
        if tracking_uri.startswith('sqlite:') and '///' in tracking_uri:
            # keep as is (absolute or relative path already included)
            pass

        self.tracking_uri = tracking_uri
        # Ensure MLflow uses the local tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        os.environ['MLFLOW_TRACKING_URI'] = self.tracking_uri

        # Initialize client
        self.client = MlflowClient(self.tracking_uri)
        self.experiment_name = experiment_name

        # Create or get experiment, prefer local mlruns artifact location
        try:
            # If experiment exists, check artifact location writability
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                # Create with explicit artifact location in project
                artifact_loc = f"file:{local_mlruns}"
                experiment_id = mlflow.create_experiment(self.experiment_name, artifact_location=artifact_loc)
                self.experiment_id = experiment_id
                mlflow.set_experiment(self.experiment_name)
                logging.info(f"✅ MLflow experiment created locally (experiment: {self.experiment_name})")
            else:
                # Verify artifact_location is writable; if not, create a local experiment fallback
                exp_info = self.client.get_experiment(experiment.experiment_id)
                artifact_location = exp_info.artifact_location or ''
                writable = True
                try:
                    # artifact_location may be 'file:C:/path' - strip 'file:'
                    check_path = artifact_location
                    if check_path.startswith('file:'):
                        check_path = check_path[5:]
                    if not check_path:
                        check_path = local_mlruns
                    # Ensure directory exists
                    os.makedirs(check_path, exist_ok=True)
                    # Test write access by creating a temp file
                    test_file = os.path.join(check_path, '.mlflow_write_test')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except Exception:
                    writable = False

                if not writable:
                    # Create a new experiment with suffix to ensure local artifact location
                    new_name = f"{self.experiment_name}_local"
                    artifact_loc = f"file:{local_mlruns}"
                    experiment_id = mlflow.create_experiment(new_name, artifact_location=artifact_loc)
                    self.experiment_id = experiment_id
                    self.experiment_name = new_name
                    mlflow.set_experiment(self.experiment_name)
                    logging.warning(f"⚠️ Existing MLflow experiment artifact location not writable; created local experiment: {self.experiment_name}")
                else:
                    self.experiment_id = exp_info.experiment_id
                    mlflow.set_experiment(self.experiment_name)
                    logging.info(f"✅ MLflow tracking initialized (experiment: {self.experiment_name})")

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
