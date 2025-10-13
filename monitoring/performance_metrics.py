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


# Alias for compatibility with app.py and other modules

class PerformanceMonitor:

# Alias for compatibility with main.py
    #PerformanceMonitor = RealTimeMetricsCollector
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
        logging.info("✅ Metrics collection stopped")

# Alias for compatibility with app.py and other modules
RealTimeMetricsCollector = PerformanceMonitor

# Global metrics collector instance
metrics_collector = None
def get_metrics_collector() -> PerformanceMonitor:
    """Get global metrics collector instance"""
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = PerformanceMonitor()
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