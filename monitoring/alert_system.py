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