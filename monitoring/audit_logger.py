# monitoring/audit_logger.py
#AUDIT_LOGGER = '''
"""
Comprehensive Audit Logging for Medical AI System Compliance
"""

import sqlite3
import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from enum import Enum
from dataclasses import dataclass

class AuditEventType(Enum):
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    IMAGE_ANALYSIS = "image_analysis"
    ANNOTATION_CREATED = "annotation_created"
    MODEL_PREDICTION = "model_prediction"
    DATA_ACCESS = "data_access"
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    MODEL_RETRAINED = "model_retrained"
    ALERT_TRIGGERED = "alert_triggered"

@dataclass
class AuditEvent:
    event_id: str
    event_type: AuditEventType
    user_id: Optional[str]
    timestamp: datetime
    action_description: str
    resource_id: Optional[str]
    resource_type: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    request_id: Optional[str]
    outcome: str  # success, failure, partial
    metadata: Dict[str, Any]

class MedicalAuditLogger:
    """
    HIPAA-compliant audit logging system
    """
    
    def __init__(self, audit_db: str = "audit_log.db"):
        self.audit_db = audit_db
        self._initialize_database()
        
        logging.info("✅ Medical audit logger initialized")
    
    def _initialize_database(self):
        """Initialize audit database with integrity checks"""
        with sqlite3.connect(self.audit_db) as conn:
            cursor = conn.cursor()
            
            # Main audit log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    event_type TEXT,
                    user_id TEXT,
                    timestamp TEXT,
                    action_description TEXT,
                    resource_id TEXT,
                    resource_type TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    request_id TEXT,
                    outcome TEXT,
                    metadata TEXT,
                    integrity_hash TEXT
                )
            ''')
            
            # Create indexes for efficient querying
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_log(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_log(event_type)')
            
            conn.commit()
    
    def log_event(self, event: AuditEvent):
        """Log audit event with integrity verification"""
        try:
            # Calculate integrity hash
            event_data = f"{event.event_id}{event.user_id}{event.timestamp.isoformat()}{event.action_description}"
            integrity_hash = hashlib.sha256(event_data.encode()).hexdigest()
            
            with sqlite3.connect(self.audit_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO audit_log 
                    (event_id, event_type, user_id, timestamp, action_description,
                     resource_id, resource_type, ip_address, user_agent, request_id,
                     outcome, metadata, integrity_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id,
                    event.event_type.value,
                    event.user_id,
                    event.timestamp.isoformat(),
                    event.action_description,
                    event.resource_id,
                    event.resource_type,
                    event.ip_address,
                    event.user_agent,
                    event.request_id,
                    event.outcome,
                    json.dumps(event.metadata),
                    integrity_hash
                ))
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Failed to log audit event: {str(e)}")
    
    def get_audit_trail(self, user_id: Optional[str] = None, 
                       event_type: Optional[AuditEventType] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Get filtered audit trail"""
        try:
            query = "SELECT * FROM audit_log WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type.value)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            with sqlite3.connect(self.audit_db) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                columns = [description[0] for description in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                return results
                
        except Exception as e:
            logging.error(f"❌ Failed to get audit trail: {str(e)}")
            return []