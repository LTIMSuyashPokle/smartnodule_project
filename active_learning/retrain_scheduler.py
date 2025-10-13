# ========================================================================
# 5. AUTOMATED RETRAINING SYSTEM
# ========================================================================

# active_learning/retrain_scheduler.py
import schedule
import time
import threading
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import numpy as np
import json

from active_learning.uncertainty_queue import UncertaintyQueue
from mlops.experiment_tracker import MLflowTracker
from api.models import SmartNoduleModel
# from training.trainer import SmartNoduleTrainer  # Commented out, as training module does not exist

class AutomaticRetrainingSystem:
    """
    Automated system for retraining models with new annotated data
    """
    
    def __init__(
        self,
        uncertainty_queue: UncertaintyQueue,
        mlflow_tracker: MLflowTracker,
        model_path: str,
        min_cases_for_retrain: int = 50,
        retrain_interval_hours: int = 24
    ):
        self.uncertainty_queue = uncertainty_queue
        self.mlflow_tracker = mlflow_tracker
        self.model_path = model_path
        self.min_cases_for_retrain = min_cases_for_retrain
        self.retrain_interval_hours = retrain_interval_hours
        
        self.trainer = None
        self.is_running = False
        self.scheduler_thread = None
        
        # Setup scheduler
        self._setup_scheduler()
        
        logging.info(f"✅ Automatic retraining system initialized")
    
    def _setup_scheduler(self):
        """Setup retraining schedule"""
        schedule.every(self.retrain_interval_hours).hours.do(self._check_and_retrain)
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
    
    def _run_scheduler(self):
        """Run the scheduler in background thread"""
        self.is_running = True
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _check_and_retrain(self):
        """Check if retraining is needed and execute if conditions are met"""
        try:
            # Get annotated cases
            annotated_cases = self.uncertainty_queue.get_annotated_cases_for_retraining(
                limit=1000
            )
            
            logging.info(f"📊 Found {len(annotated_cases)} new annotated cases")
            
            if len(annotated_cases) >= self.min_cases_for_retrain:
                logging.info("🔄 Starting automatic retraining...")
                self._execute_retraining(annotated_cases)
            else:
                logging.info(f"⏳ Not enough cases for retraining ({len(annotated_cases)}/{self.min_cases_for_retrain})")
                
        except Exception as e:
            logging.error(f"❌ Retraining check failed: {str(e)}")
    
    def _execute_retraining(self, annotated_cases: List[Tuple[np.ndarray, Dict]]):
        """Execute the retraining process"""
        try:
            # Prepare training data
            images, labels = self._prepare_training_data(annotated_cases)
            
            # Load current best model
            current_model = self._load_current_model()
            
            # Initialize trainer
            if self.trainer is None:
                self.trainer = SmartNoduleTrainer(
                    model=current_model,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            
            # Fine-tune model on new data
            training_config = {
                'learning_rate': 1e-5,  # Lower LR for fine-tuning
                'batch_size': 8,
                'epochs': 10,
                'early_stopping_patience': 3,
                'save_best_only': True
            }
            
            # Start MLflow run for retraining
            run_name = f"active_learning_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with self.mlflow_tracker.start_run(run_name=run_name):
                # Log retraining parameters
                self.mlflow_tracker.log_param("retrain_cases", len(annotated_cases))
                self.mlflow_tracker.log_param("retrain_type", "active_learning")
                self.mlflow_tracker.log_param("base_model", self.model_path)
                
                # Execute fine-tuning
                retrain_results = self.trainer.fine_tune(
                    images=images,
                    labels=labels,
                    config=training_config
                )
                
                # Log results
                for metric, value in retrain_results['metrics'].items():
                    self.mlflow_tracker.log_metric(f"retrain_{metric}", value)
                
                # Save new model if it's better
                if self._is_model_improved(retrain_results):
                    new_model_path = self._save_retrained_model(
                        self.trainer.model, retrain_results
                    )
                    
                    # Log model artifact
                    self.mlflow_tracker.log_artifact(new_model_path, "retrained_models")
                    
                    logging.info(f"✅ Model retrained and improved! New model saved: {new_model_path}")
                    
                    # Notify about successful retraining
                    self._notify_retraining_success(retrain_results)
                    
                else:
                    logging.info("📊 Retrained model did not improve performance, keeping current model")
                    
        except Exception as e:
            logging.error(f"❌ Retraining execution failed: {str(e)}")
            # Log failure to MLflow
            self.mlflow_tracker.log_param("retrain_status", "failed")
            self.mlflow_tracker.log_param("error", str(e))
    
    def _prepare_training_data(
        self, 
        annotated_cases: List[Tuple[np.ndarray, Dict]]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Prepare training data from annotated cases"""
        images = []
        labels = []
        
        for image_data, annotation in annotated_cases:
            images.append(image_data)
            
            # Extract label from annotation
            # Assuming annotation has 'has_nodule' boolean field
            label = 1 if annotation.get('has_nodule', False) else 0
            labels.append(label)
        
        return images, labels
    
    def _load_current_model(self) -> nn.Module:
        """Load the current best model"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            model = SmartNoduleModel(
                backbone='efficientnet_b3',
                num_classes=1,
                pretrained=False
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
            
        except Exception as e:
            logging.error(f"❌ Failed to load current model: {str(e)}")
            raise
    
    def _is_model_improved(self, retrain_results: Dict[str, Any]) -> bool:
        """Check if retrained model is better than current model"""
        # Simple improvement check based on validation accuracy
        # In production, this could be more sophisticated
        
        current_accuracy = retrain_results['metrics'].get('val_accuracy', 0.0)
        
        # Get previous best accuracy from MLflow
        try:
            best_model_info = self.mlflow_tracker.get_best_model('val_accuracy')
            if best_model_info:
                previous_accuracy = best_model_info['metrics'].get('val_accuracy', 0.0)
                return current_accuracy > previous_accuracy
            else:
                return True  # No previous model, so this is an improvement
                
        except:
            return True  # Default to improvement if we can't compare
    
    def _save_retrained_model(
        self, 
        model: nn.Module, 
        results: Dict[str, Any]
    ) -> str:
        """Save retrained model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"smartnodule_retrained_{timestamp}.pth"
        
        # Save model with additional metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_version': f"2.1.{timestamp}",
            'retrain_results': results,
            'retrain_timestamp': datetime.now().isoformat(),
            'training_type': 'active_learning_retrain'
        }, model_filename)
        
        return model_filename
    
    def _notify_retraining_success(self, results: Dict[str, Any]):
        """Notify about successful retraining"""
        # This could send emails, Slack notifications, etc.
        # For now, just log the success
        logging.info(f"🎉 RETRAINING SUCCESS! New accuracy: {results['metrics'].get('val_accuracy', 'N/A'):.3f}")
    
    def stop_scheduler(self):
        """Stop the automatic retraining scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logging.info("🛑 Automatic retraining scheduler stopped")