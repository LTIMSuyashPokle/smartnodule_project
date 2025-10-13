# SMARTNODULE MODULE 4: MONITORING & DEPLOYMENT COMPONENTS
# Complete Implementation of System Monitoring and Deployment Infrastructure
# ========================================================================
# MONITORING COMPONENTS
# ========================================================================

# monitoring/performance_metrics.py
#PERFORMANCE_METRICS = '''
"""
Real-time Performance Metrics Collection and Analysis
"""

import time
import threading
import queue
import sqlite3
import json
import psutil
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class MetricPoint:
    name: str
    value: float
    metric_type: MetricType
    tags: Dict[str, str]
    timestamp: datetime

class RealTimeMetricsCollector:
    """
    High-performance real-time metrics collection system
    """
    
    def __init__(self, metrics_db: str = "realtime_metrics.db", buffer_size: int = 1000):
        self.metrics_db = metrics_db
        self.buffer_size = buffer_size
        
        # In-memory metric storage for real-time access
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.metric_aggregates = defaultdict(list)
        self.custom_metrics = {}
        
        # Thread-safe queue for metric collection
        self.metrics_queue = queue.Queue()
        
        # Control flags
        self.collection_active = False
        self.collection_thread = None
        self.persistence_thread = None
        
        # Performance tracking
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.response_times = deque(maxlen=1000)
        
        self._initialize_database()
        self._start_collection_threads()
        
        logging.info("✅ Real-time metrics collector initialized")
    
    def _initialize_database(self):
        """Initialize metrics database with optimized schema"""
        try:
            with sqlite3.connect(self.metrics_db) as conn:
                cursor = conn.cursor()
                
                # Real-time metrics table with indexes for performance
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS realtime_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        metric_name TEXT,
                        metric_value REAL,
                        metric_type TEXT,
                        tags TEXT,
                        hour_bucket INTEGER,
                        minute_bucket INTEGER
                    )
                ''')
                
                # Create indexes for fast querying
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON realtime_metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_name ON realtime_metrics(metric_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_hour_bucket ON realtime_metrics(hour_bucket)')
                
                # Aggregated metrics table for dashboard performance
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metric_aggregates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        time_bucket TEXT,
                        metric_name TEXT,
                        count INTEGER,
                        sum_value REAL,
                        avg_value REAL,
                        min_value REAL,
                        max_value REAL,
                        std_value REAL,
                        percentile_50 REAL,
                        percentile_95 REAL,
                        percentile_99 REAL,
                        created_at REAL
                    )
                ''')
                
                # System health snapshots
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_health_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        cpu_percent REAL,
                        memory_percent REAL,
                        disk_usage_percent REAL,
                        gpu_utilization REAL,
                        gpu_memory_percent REAL,
                        active_connections INTEGER,
                        queue_size INTEGER,
                        error_rate REAL
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Metrics database initialization failed: {str(e)}")
            raise
    
    def _start_collection_threads(self):
        """Start background threads for metric collection"""
        self.collection_active = True
        
        # Metrics processing thread
        self.collection_thread = threading.Thread(target=self._metrics_processor, daemon=True)
        self.collection_thread.start()
        
        # Database persistence thread
        self.persistence_thread = threading.Thread(target=self._persistence_worker, daemon=True)
        self.persistence_thread.start()
        
        # System health monitoring thread
        self.health_thread = threading.Thread(target=self._system_health_monitor, daemon=True)
        self.health_thread.start()
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE, 
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric point (thread-safe)"""
        try:
            metric_point = MetricPoint(
                name=name,
                value=value,
                metric_type=metric_type,
                tags=tags or {},
                timestamp=datetime.now()
            )
            
            self.metrics_queue.put(metric_point)
            
        except Exception as e:
            logging.error(f"❌ Failed to record metric {name}: {str(e)}")
    
    def record_request_metric(self, endpoint: str, method: str, status_code: int, 
                            response_time: float, error: Optional[str] = None):
        """Record HTTP request metrics"""
        self.request_count += 1
        self.response_times.append(response_time)
        
        if status_code >= 400:
            self.error_count += 1
        
        tags = {
            'endpoint': endpoint,
            'method': method,
            'status_code': str(status_code)
        }
        
        if error:
            tags['error'] = error
        
        # Record multiple related metrics
        self.record_metric('http_requests_total', 1, MetricType.COUNTER, tags)
        self.record_metric('http_response_time', response_time, MetricType.TIMER, tags)
        
        if status_code >= 400:
            self.record_metric('http_errors_total', 1, MetricType.COUNTER, tags)
    
    def record_ai_inference_metric(self, model_version: str, processing_time: float,
                                  prediction_confidence: float, uncertainty_level: str):
        """Record AI inference specific metrics"""
        tags = {
            'model_version': model_version,
            'uncertainty_level': uncertainty_level
        }
        
        self.record_metric('ai_inference_time', processing_time, MetricType.TIMER, tags)
        self.record_metric('ai_prediction_confidence', prediction_confidence, MetricType.GAUGE, tags)
        self.record_metric('ai_inference_total', 1, MetricType.COUNTER, tags)
    
    def get_real_time_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Get real-time metrics for the last N minutes"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            # Filter recent metrics from buffer
            recent_metrics = [m for m in self.metrics_buffer if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return self._get_default_metrics()
            
            # Aggregate metrics by name
            aggregated = defaultdict(list)
            for metric in recent_metrics:
                aggregated[metric.name].append(metric.value)
            
            # Calculate statistics
            result = {}
            for name, values in aggregated.items():
                if values:
                    result[name] = {
                        'count': len(values),
                        'avg': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'latest': values[-1] if values else 0,
                        'p95': np.percentile(values, 95) if len(values) > 5 else values[-1]
                    }
            
            # Add derived metrics
            result['requests_per_minute'] = result.get('http_requests_total', {}).get('count', 0) / minutes
            result['error_rate'] = (self.error_count / max(self.request_count, 1)) * 100
            result['avg_response_time'] = np.mean(list(self.response_times)) if self.response_times else 0
            result['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()
            
            return result
            
        except Exception as e:
            logging.error(f"❌ Failed to get real-time metrics: {str(e)}")
            return self._get_default_metrics()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU metrics (if available)
            gpu_metrics = self._get_gpu_metrics()
            
            # Network connections
            connections = len(psutil.net_connections())
            
            health = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage_percent': (disk.used / disk.total) * 100,
                'network_connections': connections,
                'queue_size': self.metrics_queue.qsize(),
                **gpu_metrics
            }
            
            # Health status
            health['status'] = self._calculate_health_status(health)
            
            return health
            
        except Exception as e:
            logging.error(f"❌ Failed to get system health: {str(e)}")
            return {'status': 'unknown', 'error': str(e)}
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for performance dashboard"""
        try:
            # Real-time metrics
            realtime = self.get_real_time_metrics(minutes=15)
            
            # System health
            health = self.get_system_health()
            
            # Historical trends (last 24 hours)
            trends = self._get_performance_trends(hours=24)
            
            # Top endpoints by volume
            top_endpoints = self._get_top_endpoints(limit=10)
            
            # Error analysis
            error_analysis = self._get_error_analysis()
            
            return {
                'realtime_metrics': realtime,
                'system_health': health,
                'performance_trends': trends,
                'top_endpoints': top_endpoints,
                'error_analysis': error_analysis,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"❌ Failed to get dashboard data: {str(e)}")
            return {'error': str(e)}
    
    def _metrics_processor(self):
        """Background thread to process incoming metrics"""
        batch_size = 100
        batch_timeout = 5.0  # seconds
        batch = []
        last_flush = time.time()
        
        while self.collection_active:
            try:
                # Get metric with timeout
                try:
                    metric = self.metrics_queue.get(timeout=1.0)
                    batch.append(metric)
                    
                    # Add to in-memory buffer for real-time access
                    self.metrics_buffer.append(metric)
                    
                except queue.Empty:
                    pass
                
                # Flush batch when full or timeout reached
                current_time = time.time()
                if len(batch) >= batch_size or (batch and current_time - last_flush > batch_timeout):
                    self._persist_metric_batch(batch)
                    batch = []
                    last_flush = current_time
                
            except Exception as e:
                logging.error(f"❌ Metrics processor error: {str(e)}")
                time.sleep(1)
        
        # Final flush
        if batch:
            self._persist_metric_batch(batch)
    
    def _persistence_worker(self):
        """Background thread for database operations"""
        while self.collection_active:
            try:
                # Aggregate old metrics every minute
                self._create_metric_aggregates()
                
                # Clean up old raw metrics (keep only recent for real-time queries)
                self._cleanup_old_metrics()
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logging.error(f"❌ Persistence worker error: {str(e)}")
                time.sleep(10)
    
    def _system_health_monitor(self):
        """Background thread for system health monitoring"""
        while self.collection_active:
            try:
                health = self.get_system_health()
                self._persist_health_snapshot(health)
                
                # Check for alerts
                self._check_health_alerts(health)
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logging.error(f"❌ Health monitor error: {str(e)}")
                time.sleep(10)
    
    def _persist_metric_batch(self, batch: List[MetricPoint]):
        """Persist a batch of metrics to database"""
        try:
            if not batch:
                return
            
            with sqlite3.connect(self.metrics_db) as conn:
                cursor = conn.cursor()
                
                for metric in batch:
                    timestamp_float = metric.timestamp.timestamp()
                    hour_bucket = int(timestamp_float // 3600)
                    minute_bucket = int(timestamp_float // 60)
                    
                    cursor.execute('''
                        INSERT INTO realtime_metrics 
                        (timestamp, metric_name, metric_value, metric_type, tags, 
                         hour_bucket, minute_bucket)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp_float,
                        metric.name,
                        metric.value,
                        metric.metric_type.value,
                        json.dumps(metric.tags),
                        hour_bucket,
                        minute_bucket
                    ))
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Failed to persist metric batch: {str(e)}")
    
    def _get_gpu_metrics(self) -> Dict[str, Optional[float]]:
        """Get GPU utilization metrics"""
        try:
            if torch.cuda.is_available():
                return {
                    'gpu_utilization': torch.cuda.utilization(),
                    'gpu_memory_percent': (torch.cuda.memory_used() / torch.cuda.max_memory_cached()) * 100
                }
            else:
                return {'gpu_utilization': None, 'gpu_memory_percent': None}
        except:
            return {'gpu_utilization': None, 'gpu_memory_percent': None}
    
    def _calculate_health_status(self, health: Dict[str, Any]) -> str:
        """Calculate overall system health status"""
        cpu = health.get('cpu_percent', 0)
        memory = health.get('memory_percent', 0)
        disk = health.get('disk_usage_percent', 0)
        
        if cpu > 90 or memory > 90 or disk > 95:
            return 'critical'
        elif cpu > 80 or memory > 80 or disk > 90:
            return 'warning'
        elif cpu > 70 or memory > 70 or disk > 85:
            return 'degraded'
        else:
            return 'healthy'
    
    def _get_performance_trends(self, hours: int = 24) -> Dict[str, List]:
        """Get performance trends for dashboard charts"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            with sqlite3.connect(self.metrics_db) as conn:
                # Get hourly aggregates
                df = pd.read_sql_query('''
                    SELECT 
                        datetime(hour_bucket * 3600, 'unixepoch') as hour,
                        metric_name,
                        AVG(metric_value) as avg_value,
                        COUNT(*) as count
                    FROM realtime_metrics 
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY hour_bucket, metric_name
                    ORDER BY hour_bucket
                ''', conn, params=(start_time.timestamp(), end_time.timestamp()))
                
                # Format for frontend charts
                trends = {}
                for name in df['metric_name'].unique():
                    metric_data = df[df['metric_name'] == name]
                    trends[name] = {
                        'timestamps': metric_data['hour'].tolist(),
                        'values': metric_data['avg_value'].tolist()
                    }
                
                return trends
                
        except Exception as e:
            logging.error(f"❌ Failed to get performance trends: {str(e)}")
            return {}
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default/empty metrics structure"""
        return {
            'requests_per_minute': 0,
            'error_rate': 0,
            'avg_response_time': 0,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }
    
    def _get_top_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top endpoints by request volume"""
        # This would analyze the metrics to find top endpoints
        # Simplified implementation
        return []
    
    def _get_error_analysis(self) -> Dict[str, Any]:
        """Get error analysis data"""
        # This would analyze error patterns and frequencies
        return {
            'total_errors': self.error_count,
            'error_rate': (self.error_count / max(self.request_count, 1)) * 100,
            'common_errors': []
        }
    
    def _create_metric_aggregates(self):
        """Create aggregated metrics for efficient dashboard queries"""
        # Implementation for creating hourly/daily aggregates
        pass
    
    def _cleanup_old_metrics(self):
        """Clean up old raw metrics to manage database size"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            with sqlite3.connect(self.metrics_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM realtime_metrics 
                    WHERE timestamp < ?
                ''', (cutoff_time.timestamp(),))
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Failed to cleanup old metrics: {str(e)}")
    
    def _persist_health_snapshot(self, health: Dict[str, Any]):
        """Persist system health snapshot"""
        try:
            with sqlite3.connect(self.metrics_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_health_snapshots 
                    (timestamp, cpu_percent, memory_percent, disk_usage_percent,
                     gpu_utilization, gpu_memory_percent, active_connections, queue_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().timestamp(),
                    health.get('cpu_percent'),
                    health.get('memory_percent'),
                    health.get('disk_usage_percent'),
                    health.get('gpu_utilization'),
                    health.get('gpu_memory_percent'),
                    health.get('network_connections'),
                    health.get('queue_size')
                ))
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Failed to persist health snapshot: {str(e)}")
    
    def _check_health_alerts(self, health: Dict[str, Any]):
        """Check for health-based alerts"""
        # Implementation for health-based alerting
        pass
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        if self.persistence_thread:
            self.persistence_thread.join(timeout=5)
        if hasattr(self, 'health_thread'):
            self.health_thread.join(timeout=5)
        
        logging.info("✅ Metrics collection stopped")

# Global metrics collector instance
metrics_collector = None

def get_metrics_collector() -> RealTimeMetricsCollector:
    """Get global metrics collector instance"""
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = RealTimeMetricsCollector()
    return metrics_collector

# Decorator for automatic metric collection
def monitor_performance(metric_name: str = None):
    """Decorator to automatically monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            name = metric_name or f"function_{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                get_metrics_collector().record_metric(
                    f"{name}_execution_time", 
                    execution_time, 
                    MetricType.TIMER
                )
                get_metrics_collector().record_metric(
                    f"{name}_calls_total", 
                    1, 
                    MetricType.COUNTER
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                get_metrics_collector().record_metric(
                    f"{name}_errors_total", 
                    1, 
                    MetricType.COUNTER,
                    {'error_type': type(e).__name__}
                )
                raise
        
        return wrapper
    return decorator
#'''

# ========================================================================
# monitoring/alert_system.py
# ========================================================================

#ALERT_SYSTEM = '''
"""
Intelligent Alert System with Multiple Notification Channels
"""

import smtplib
import json
import sqlite3
import threading
import time
import requests
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
from enum import Enum
import logging

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    metric_name: str
    current_value: float
    threshold_value: float
    tags: Dict[str, str]
    created_at: datetime
    updated_at: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None

class AlertRule:
    def __init__(self, rule_id: str, name: str, metric_name: str, 
                 condition: str, threshold: float, severity: AlertSeverity,
                 evaluation_window: int = 300):
        self.rule_id = rule_id
        self.name = name
        self.metric_name = metric_name
        self.condition = condition  # 'gt', 'lt', 'eq'
        self.threshold = threshold
        self.severity = severity
        self.evaluation_window = evaluation_window
        self.enabled = True
        self.last_evaluation = None
        self.consecutive_violations = 0

class SmartAlertSystem:
    """
    Intelligent alerting system with smart notifications and escalation
    """
    
    def __init__(self, alerts_db: str = "alerts.db"):
        self.alerts_db = alerts_db
        self.alert_rules = {}
        self.active_alerts = {}
        self.notification_channels = {}
        self.alert_history = []
        
        # Alert processing
        self.processing_active = False
        self.processing_thread = None
        
        # Rate limiting and suppression
        self.rate_limits = {}
        self.suppression_rules = []
        
        self._initialize_database()
        self._load_default_rules()
        self._start_alert_processor()
        
        logging.info("✅ Smart alert system initialized")
    
    def _initialize_database(self):
        """Initialize alerts database"""
        try:
            with sqlite3.connect(self.alerts_db) as conn:
                cursor = conn.cursor()
                
                # Active alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS active_alerts (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        description TEXT,
                        severity TEXT,
                        source TEXT,
                        metric_name TEXT,
                        current_value REAL,
                        threshold_value REAL,
                        tags TEXT,
                        status TEXT,
                        created_at TEXT,
                        updated_at TEXT,
                        acknowledged_by TEXT,
                        resolved_at TEXT
                    )
                ''')
                
                # Alert history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alert_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT,
                        title TEXT,
                        severity TEXT,
                        source TEXT,
                        duration_seconds INTEGER,
                        status TEXT,
                        created_at TEXT,
                        resolved_at TEXT,
                        resolution_method TEXT
                    )
                ''')
                
                # Notification log table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS notification_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT,
                        channel TEXT,
                        recipient TEXT,
                        sent_at TEXT,
                        success BOOLEAN,
                        error_message TEXT
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Alert database initialization failed: {str(e)}")
            raise
    
    def _load_default_rules(self):
        """Load default alert rules for medical AI system"""
        default_rules = [
            AlertRule("cpu_high", "High CPU Usage", "cpu_percent", "gt", 85, AlertSeverity.WARNING),
            AlertRule("memory_high", "High Memory Usage", "memory_percent", "gt", 90, AlertSeverity.ERROR),
            AlertRule("gpu_high", "High GPU Usage", "gpu_utilization", "gt", 95, AlertSeverity.WARNING),
            AlertRule("response_time_high", "High Response Time", "avg_response_time", "gt", 10.0, AlertSeverity.WARNING),
            AlertRule("error_rate_high", "High Error Rate", "error_rate", "gt", 5.0, AlertSeverity.ERROR),
            AlertRule("model_accuracy_low", "Model Accuracy Drop", "accuracy", "lt", 0.90, AlertSeverity.CRITICAL),
            AlertRule("disk_space_low", "Low Disk Space", "disk_usage_percent", "gt", 95, AlertSeverity.CRITICAL),
            AlertRule("queue_size_high", "High Queue Size", "queue_size", "gt", 100, AlertSeverity.WARNING),
            AlertRule("drift_detected", "Model Drift Detected", "drift_score", "gt", 0.1, AlertSeverity.ERROR)
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    def add_notification_channel(self, channel_type: str, config: Dict[str, Any]):
        """Add notification channel (email, slack, webhook, etc.)"""
        self.notification_channels[channel_type] = config
        logging.info(f"✅ Added notification channel: {channel_type}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add new alert rule"""
        self.alert_rules[rule.rule_id] = rule
        logging.info(f"✅ Added alert rule: {rule.name}")
    
    def evaluate_metrics(self, metrics: Dict[str, Any]):
        """Evaluate current metrics against alert rules"""
        try:
            current_time = datetime.now()
            
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                metric_value = self._extract_metric_value(metrics, rule.metric_name)
                if metric_value is None:
                    continue
                
                # Evaluate condition
                violation = self._evaluate_condition(metric_value, rule.condition, rule.threshold)
                
                if violation:
                    rule.consecutive_violations += 1
                    
                    # Only fire alert after consecutive violations (reduces noise)
                    if rule.consecutive_violations >= 2:
                        self._fire_alert(rule, metric_value, current_time)
                else:
                    # Reset violation counter and potentially resolve alert
                    if rule.consecutive_violations > 0:
                        rule.consecutive_violations = 0
                        self._potentially_resolve_alert(rule_id)
                
                rule.last_evaluation = current_time
                
        except Exception as e:
            logging.error(f"❌ Metrics evaluation failed: {str(e)}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == 'gt':
            return value > threshold
        elif condition == 'lt':
            return value < threshold
        elif condition == 'eq':
            return abs(value - threshold) < 0.001
        elif condition == 'gte':
            return value >= threshold
        elif condition == 'lte':
            return value <= threshold
        else:
            return False
    
    def _fire_alert(self, rule: AlertRule, current_value: float, timestamp: datetime):
        """Fire an alert"""
        try:
            alert_id = f"{rule.rule_id}_{int(timestamp.timestamp())}"
            
            # Check if similar alert already exists
            existing_alert_id = f"{rule.rule_id}_active"
            if existing_alert_id in self.active_alerts:
                # Update existing alert
                alert = self.active_alerts[existing_alert_id]
                alert.current_value = current_value
                alert.updated_at = timestamp
                self._update_alert_in_db(alert)
                return
            
            # Create new alert
            alert = Alert(
                id=existing_alert_id,
                title=rule.name,
                description=self._generate_alert_description(rule, current_value),
                severity=rule.severity,
                source="smartnodule_monitor",
                metric_name=rule.metric_name,
                current_value=current_value,
                threshold_value=rule.threshold,
                tags={'rule_id': rule.rule_id},
                created_at=timestamp,
                updated_at=timestamp
            )
            
            # Store alert
            self.active_alerts[alert.id] = alert
            self._save_alert_to_db(alert)
            
            # Send notifications
            self._send_notifications(alert)
            
            logging.warning(f"🚨 ALERT FIRED: {alert.title} (Value: {current_value}, Threshold: {rule.threshold})")
            
        except Exception as e:
            logging.error(f"❌ Failed to fire alert: {str(e)}")
    
    def _potentially_resolve_alert(self, rule_id: str):
        """Potentially resolve an active alert"""
        alert_id = f"{rule_id}_active"
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            # Update database
            self._update_alert_in_db(alert)
            
            # Move to history
            self._move_to_history(alert)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logging.info(f"✅ Alert resolved: {alert.title}")
    
    def _generate_alert_description(self, rule: AlertRule, current_value: float) -> str:
        """Generate human-readable alert description"""
        return f"{rule.metric_name} is {current_value:.2f}, which is {rule.condition} threshold of {rule.threshold}"
    
    def _extract_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from metrics dict, supporting nested paths"""
        try:
            # Support nested paths like 'system_health.cpu_percent'
            keys = metric_name.split('.')
            value = metrics
            
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    return None
            
            # Handle different value formats
            if isinstance(value, dict) and 'avg' in value:
                return float(value['avg'])
            elif isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, dict) and 'latest' in value:
                return float(value['latest'])
            else:
                return None
                
        except (KeyError, TypeError, ValueError):
            return None
    
    def _send_notifications(self, alert: Alert):
        """Send notifications through configured channels"""
        for channel_type, config in self.notification_channels.items():
            try:
                if self._should_notify(alert, channel_type):
                    if channel_type == 'email':
                        self._send_email_notification(alert, config)
                    elif channel_type == 'slack':
                        self._send_slack_notification(alert, config)
                    elif channel_type == 'webhook':
                        self._send_webhook_notification(alert, config)
                    
            except Exception as e:
                logging.error(f"❌ Failed to send {channel_type} notification: {str(e)}")
                self._log_notification_failure(alert.id, channel_type, str(e))
    
    def _should_notify(self, alert: Alert, channel_type: str) -> bool:
        """Check if notification should be sent (rate limiting, severity filtering)"""
        # Rate limiting check
        rate_limit_key = f"{alert.metric_name}_{channel_type}"
        last_notification = self.rate_limits.get(rate_limit_key)
        
        if last_notification:
            time_since_last = (datetime.now() - last_notification).total_seconds()
            min_interval = 300  # 5 minutes minimum between notifications
            
            if time_since_last < min_interval:
                return False
        
        # Severity filtering
        channel_config = self.notification_channels[channel_type]
        min_severity = channel_config.get('min_severity', AlertSeverity.INFO)
        
        severity_levels = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.ERROR: 2,
            AlertSeverity.CRITICAL: 3
        }
        
        if severity_levels[alert.severity] < severity_levels[min_severity]:
            return False
        
        return True
    
    def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] SmartNodule Alert: {alert.title}"
            
            body = f"""
SmartNodule Alert Notification

Alert: {alert.title}
Severity: {alert.severity.value.upper()}
Description: {alert.description}
Source: {alert.source}
Metric: {alert.metric_name}
Current Value: {alert.current_value}
Threshold: {alert.threshold_value}
Time: {alert.created_at.isoformat()}

Please investigate this issue promptly.

SmartNodule Monitoring System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            if config.get('use_tls'):
                server.starttls()
            
            if config.get('username') and config.get('password'):
                server.login(config['username'], config['password'])
            
            server.send_message(msg)
            server.quit()
            
            self._log_notification_success(alert.id, 'email', config['to_emails'])
            self.rate_limits[f"{alert.metric_name}_email"] = datetime.now()
            
        except Exception as e:
            raise Exception(f"Email notification failed: {str(e)}")
    
    def _send_slack_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send Slack notification"""
        try:
            severity_colors = {
                AlertSeverity.INFO: '#36a64f',      # Green
                AlertSeverity.WARNING: '#ff9500',   # Orange
                AlertSeverity.ERROR: '#ff0000',     # Red
                AlertSeverity.CRITICAL: '#8b0000'   # Dark Red
            }
            
            payload = {
                "channel": config['channel'],
                "username": "SmartNodule Monitor",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": severity_colors[alert.severity],
                        "title": f"[{alert.severity.value.upper()}] {alert.title}",
                        "text": alert.description,
                        "fields": [
                            {"title": "Metric", "value": alert.metric_name, "short": True},
                            {"title": "Current Value", "value": str(alert.current_value), "short": True},
                            {"title": "Threshold", "value": str(alert.threshold_value), "short": True},
                            {"title": "Time", "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                        ],
                        "footer": "SmartNodule Monitoring",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            response = requests.post(config['webhook_url'], json=payload)
            response.raise_for_status()
            
            self._log_notification_success(alert.id, 'slack', config['channel'])
            self.rate_limits[f"{alert.metric_name}_slack"] = datetime.now()
            
        except Exception as e:
            raise Exception(f"Slack notification failed: {str(e)}")
    
    def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook notification"""
        try:
            payload = {
                "alert_id": alert.id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "source": alert.source,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "created_at": alert.created_at.isoformat(),
                "tags": alert.tags
            }
            
            headers = {'Content-Type': 'application/json'}
            if 'headers' in config:
                headers.update(config['headers'])
            
            response = requests.post(config['url'], json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            self._log_notification_success(alert.id, 'webhook', config['url'])
            self.rate_limits[f"{alert.metric_name}_webhook"] = datetime.now()
            
        except Exception as e:
            raise Exception(f"Webhook notification failed: {str(e)}")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get list of active alerts"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        # Sort by severity and creation time
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.ERROR: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3
        }
        
        alerts.sort(key=lambda a: (severity_order[a.severity], a.created_at), reverse=True)
        return alerts
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.updated_at = datetime.now()
                
                self._update_alert_in_db(alert)
                logging.info(f"✅ Alert acknowledged: {alert.title} by {acknowledged_by}")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"❌ Failed to acknowledge alert: {str(e)}")
            return False
    
    def _save_alert_to_db(self, alert: Alert):
        """Save alert to database"""
        try:
            with sqlite3.connect(self.alerts_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO active_alerts 
                    (id, title, description, severity, source, metric_name,
                     current_value, threshold_value, tags, status, created_at, updated_at,
                     acknowledged_by, resolved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id, alert.title, alert.description, alert.severity.value,
                    alert.source, alert.metric_name, alert.current_value, alert.threshold_value,
                    json.dumps(alert.tags), alert.status.value, alert.created_at.isoformat(),
                    alert.updated_at.isoformat(), alert.acknowledged_by,
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"❌ Failed to save alert to DB: {str(e)}")
    
    def _update_alert_in_db(self, alert: Alert):
        """Update existing alert in database"""
        self._save_alert_to_db(alert)  # Same operation for SQLite
    
    def _move_to_history(self, alert: Alert):
        """Move resolved alert to history"""
        try:
            duration = None
            if alert.resolved_at and alert.created_at:
                duration = int((alert.resolved_at - alert.created_at).total_seconds())
            
            with sqlite3.connect(self.alerts_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alert_history 
                    (alert_id, title, severity, source, duration_seconds, status,
                     created_at, resolved_at, resolution_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id, alert.title, alert.severity.value, alert.source,
                    duration, alert.status.value, alert.created_at.isoformat(),
                    alert.resolved_at.isoformat() if alert.resolved_at else None,
                    'automatic'
                ))
                
                # Remove from active alerts table
                cursor.execute('DELETE FROM active_alerts WHERE id = ?', (alert.id,))
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Failed to move alert to history: {str(e)}")
    
    def _log_notification_success(self, alert_id: str, channel: str, recipient: str):
        """Log successful notification"""
        try:
            with sqlite3.connect(self.alerts_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO notification_log 
                    (alert_id, channel, recipient, sent_at, success)
                    VALUES (?, ?, ?, ?, ?)
                ''', (alert_id, channel, recipient, datetime.now().isoformat(), True))
                conn.commit()
        except Exception as e:
            logging.error(f"❌ Failed to log notification success: {str(e)}")
    
    def _log_notification_failure(self, alert_id: str, channel: str, error_message: str):
        """Log failed notification"""
        try:
            with sqlite3.connect(self.alerts_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO notification_log 
                    (alert_id, channel, sent_at, success, error_message)
                    VALUES (?, ?, ?, ?, ?)
                ''', (alert_id, channel, datetime.now().isoformat(), False, error_message))
                conn.commit()
        except Exception as e:
            logging.error(f"❌ Failed to log notification failure: {str(e)}")
    
    def _start_alert_processor(self):
        """Start background alert processing thread"""
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._alert_processor, daemon=True)
        self.processing_thread.start()
    
    def _alert_processor(self):
        """Background thread for alert processing and cleanup"""
        while self.processing_active:
            try:
                # Process any pending alert escalations
                self._process_escalations()
                
                # Clean up old resolved alerts from database
                self._cleanup_old_alerts()
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logging.error(f"❌ Alert processor error: {str(e)}")
                time.sleep(10)
    
    def _process_escalations(self):
        """Process alert escalations for unacknowledged critical alerts"""
        # Implementation for escalation logic
        pass
    
    def _cleanup_old_alerts(self):
        """Clean up old alert history"""
        try:
            cutoff_date = datetime.now() - timedelta(days=30)
            
            with sqlite3.connect(self.alerts_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM alert_history 
                    WHERE created_at < ?
                ''', (cutoff_date.isoformat(),))
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Failed to cleanup old alerts: {str(e)}")
    
    def stop_processing(self):
        """Stop alert processing"""
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logging.info("✅ Alert processing stopped")
#'''

print("✅ Created Monitoring components:")
print("   - performance_metrics.py: Real-time metrics collection and analysis")
print("   - alert_system.py: Intelligent alerting with multiple notification channels")
print("\nNext, I'll create the remaining components and __init__.py files...")