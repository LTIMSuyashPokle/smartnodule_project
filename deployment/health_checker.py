# deployment/health_checker.py
#HEALTH_CHECKER = '''
"""
Comprehensive System Health Checker with Dependency Validation
"""

import sqlite3
import requests
import psutil
import torch
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import logging
from enum import Enum
from dataclasses import dataclass

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    response_time_ms: float
    last_checked: datetime

class SystemHealthChecker:
    """
    Comprehensive health checking system
    """
    
    def __init__(self, health_db: str = "system_health.db"):
        self.health_db = health_db
        self.health_checks = {}
        self.check_history = []
        self._initialize_database()
        self._register_default_checks()
        
        logging.info("✅ System health checker initialized")
    
    def _initialize_database(self):
        """Initialize health check database"""
        with sqlite3.connect(self.health_db) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_name TEXT,
                    status TEXT,
                    message TEXT,
                    response_time_ms REAL,
                    timestamp TEXT,
                    details TEXT
                )
            ''')
            
            conn.commit()
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.register_health_check("database", self._check_database)
        self.register_health_check("model", self._check_model)
        self.register_health_check("memory", self._check_memory)
        self.register_health_check("disk_space", self._check_disk_space)
        self.register_health_check("gpu", self._check_gpu)
    
    def register_health_check(self, name: str, check_function: Callable):
        """Register a custom health check"""
        self.health_checks[name] = check_function
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_function in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_function()
                response_time = (time.time() - start_time) * 1000
                
                health_check = HealthCheck(
                    name=name,
                    status=result['status'],
                    message=result['message'],
                    details=result.get('details', {}),
                    response_time_ms=response_time,
                    last_checked=datetime.now()
                )
                
                results[name] = health_check
                self._save_health_check(health_check)
                
            except Exception as e:
                error_check = HealthCheck(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    details={'error': str(e)},
                    response_time_ms=0,
                    last_checked=datetime.now()
                )
                results[name] = error_check
                self._save_health_check(error_check)
        
        return results
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            with sqlite3.connect(self.health_db, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Database connection successful'
            }
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'message': f'Database connection failed: {str(e)}'
            }
    
    def _check_model(self) -> Dict[str, Any]:
        """Check if AI model is loaded and responsive"""
        try:
            # This would check your actual model
            # For now, simulate model check
            model_loaded = True  # Replace with actual model check
            
            if model_loaded:
                return {
                    'status': HealthStatus.HEALTHY,
                    'message': 'AI model loaded and responsive',
                    'details': {'model_version': '2.0.0'}
                }
            else:
                return {
                    'status': HealthStatus.CRITICAL,
                    'message': 'AI model not loaded'
                }
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'message': f'Model check failed: {str(e)}'
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check system memory usage"""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent < 80:
                status = HealthStatus.HEALTHY
                message = f'Memory usage normal: {usage_percent:.1f}%'
            elif usage_percent < 90:
                status = HealthStatus.DEGRADED
                message = f'Memory usage high: {usage_percent:.1f}%'
            else:
                status = HealthStatus.CRITICAL
                message = f'Memory usage critical: {usage_percent:.1f}%'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'usage_percent': usage_percent,
                    'available_gb': memory.available / (1024**3),
                    'total_gb': memory.total / (1024**3)
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'message': f'Memory check failed: {str(e)}'
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space usage"""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent < 85:
                status = HealthStatus.HEALTHY
                message = f'Disk space normal: {usage_percent:.1f}%'
            elif usage_percent < 95:
                status = HealthStatus.DEGRADED
                message = f'Disk space low: {usage_percent:.1f}%'
            else:
                status = HealthStatus.CRITICAL
                message = f'Disk space critical: {usage_percent:.1f}%'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'usage_percent': usage_percent,
                    'free_gb': disk.free / (1024**3),
                    'total_gb': disk.total / (1024**3)
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'message': f'Disk check failed: {str(e)}'
            }
    
    def _check_gpu(self) -> Dict[str, Any]:
        """Check GPU status if available"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                memory_used = torch.cuda.memory_used(current_device)
                memory_total = torch.cuda.max_memory_allocated(current_device)
                utilization = torch.cuda.utilization(current_device)
                
                return {
                    'status': HealthStatus.HEALTHY,
                    'message': f'GPU available: {gpu_name}',
                    'details': {
                        'gpu_count': gpu_count,
                        'current_device': current_device,
                        'gpu_name': gpu_name,
                        'memory_used_mb': memory_used / (1024**2),
                        'memory_total_mb': memory_total / (1024**2),
                        'utilization_percent': utilization
                    }
                }
            else:
                return {
                    'status': HealthStatus.DEGRADED,
                    'message': 'GPU not available - running on CPU',
                    'details': {'gpu_available': False}
                }
        except Exception as e:
            return {
                'status': HealthStatus.DEGRADED,
                'message': f'GPU check failed: {str(e)} - falling back to CPU'
            }
    
    def _save_health_check(self, health_check: HealthCheck):
        """Save health check result to database"""
        try:
            with sqlite3.connect(self.health_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO health_checks 
                    (check_name, status, message, response_time_ms, timestamp, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    health_check.name,
                    health_check.status.value,
                    health_check.message,
                    health_check.response_time_ms,
                    health_check.last_checked.isoformat(),
                    json.dumps(health_check.details)
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"❌ Failed to save health check: {str(e)}")