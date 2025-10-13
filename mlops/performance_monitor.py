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