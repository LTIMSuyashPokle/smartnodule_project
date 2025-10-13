# ========================================================================
# 4. ACTIVE LEARNING SYSTEM
# =======================================================================
# active_learning/uncertainty_queue.py
import sqlite3
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import pickle
import base64
from dataclasses import dataclass
from enum import Enum

class CasePriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@dataclass
class UncertainCase:
    case_id: str
    image_data: np.ndarray
    prediction: Dict[str, Any]
    priority: CasePriority
    timestamp: datetime
    patient_id: Optional[str] = None
    clinical_history: Optional[str] = None
    annotation: Optional[Dict[str, Any]] = None
    annotator_id: Optional[str] = None
    annotation_timestamp: Optional[datetime] = None

class UncertaintyQueue:
    def get_uncertain_cases(self, limit: int = 20) -> list:
        """Return pending uncertain cases for review (for Streamlit app compatibility)"""
        try:
            return self.get_pending_cases(limit=limit)
        except Exception as e:
            logging.error(f"Failed to get uncertain cases: {e}")
            return []
    """
    Intelligent queue for uncertain predictions requiring expert review
    """
    
    def __init__(self, db_path: str, threshold: float = 0.15):
        self.db_path = db_path
        self.threshold = threshold
        self._initialize_database()
        
        logging.info(f"✅ Uncertainty queue initialized (threshold: {threshold})")
    
    def _initialize_database(self):
        """Initialize SQLite database for uncertain cases"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create uncertain cases table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS uncertain_cases (
                        case_id TEXT PRIMARY KEY,
                        image_data BLOB,
                        prediction TEXT,
                        priority INTEGER,
                        timestamp TEXT,
                        patient_id TEXT,
                        clinical_history TEXT,
                        status TEXT DEFAULT 'pending',
                        annotation TEXT,
                        annotator_id TEXT,
                        annotation_timestamp TEXT,
                        retrain_used BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                # Create annotation quality table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS annotation_quality (
                        annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        case_id TEXT,
                        annotator_id TEXT,
                        annotation_time_seconds REAL,
                        confidence_score REAL,
                        quality_score REAL,
                        timestamp TEXT,
                        FOREIGN KEY (case_id) REFERENCES uncertain_cases (case_id)
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Database initialization failed: {str(e)}")
            raise
    
    def add_case(
        self,
        request_id: str,
        image_data: np.ndarray,
        prediction: Dict[str, Any],
        priority: int = 1,
        patient_id: Optional[str] = None,
        clinical_history: Optional[str] = None
    ) -> bool:
        """
        Add uncertain case to queue - FIXED VERSION
        """
        try:
            # REMOVED: Threshold check - let the app decide what's uncertain
            # The app's _check_uncertain_case already does the filtering
            
            # Serialize image data
            image_blob = pickle.dumps(image_data)
            prediction_json = json.dumps(prediction)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO uncertain_cases 
                    (case_id, image_data, prediction, priority, timestamp, 
                    patient_id, clinical_history, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                ''', (
                    request_id,
                    image_blob,
                    prediction_json,
                    priority,
                    datetime.now().isoformat(),
                    patient_id,
                    clinical_history
                ))
                
                conn.commit()
            
            logging.info(f"✅ Uncertain case added to queue: {request_id}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to add uncertain case: {str(e)}")
            return False
    
    def get_pending_cases(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get pending cases for annotation (prioritized)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT case_id, prediction, priority, timestamp, 
                           patient_id, clinical_history
                    FROM uncertain_cases 
                    WHERE status = 'pending'
                    ORDER BY priority DESC, timestamp ASC
                    LIMIT ?
                ''', (limit,))
                
                results = cursor.fetchall()
                
                cases = []
                for row in results:
                    case = {
                        'case_id': row[0],
                        'prediction': json.loads(row[1]),
                        'priority': row[2],
                        'timestamp': row[3],
                        'patient_id': row[4],
                        'clinical_history': row[5]
                    }
                    cases.append(case)
                
                return cases
                
        except Exception as e:
            logging.error(f"❌ Failed to get pending cases: {str(e)}")
            return []
    
    def submit_annotation(
        self,
        case_id: str,
        annotation: Dict[str, Any],
        annotator_id: str,
        annotation_time: float,
        confidence_score: float
    ) -> bool:
        """
        Submit expert annotation for uncertain case
        """
        try:
            annotation_json = json.dumps(annotation)
            timestamp = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update case with annotation
                cursor.execute('''
                    UPDATE uncertain_cases 
                    SET annotation = ?, annotator_id = ?, annotation_timestamp = ?, 
                        status = 'annotated'
                    WHERE case_id = ?
                ''', (annotation_json, annotator_id, timestamp, case_id))
                
                # Calculate quality score (simple heuristic)
                quality_score = self._calculate_annotation_quality(
                    annotation_time, confidence_score
                )
                
                # Log annotation quality
                cursor.execute('''
                    INSERT INTO annotation_quality 
                    (case_id, annotator_id, annotation_time_seconds, 
                     confidence_score, quality_score, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    case_id, annotator_id, annotation_time,
                    confidence_score, quality_score, timestamp
                ))
                
                conn.commit()
            
            logging.info(f"✅ Annotation submitted for case: {case_id}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to submit annotation: {str(e)}")
            return False
    
    def get_annotated_cases_for_retraining(self, limit: int = 100) -> List[Tuple[np.ndarray, Dict]]:
        """
        Get annotated cases for model retraining
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT case_id, image_data, annotation
                    FROM uncertain_cases 
                    WHERE status = 'annotated' AND retrain_used = FALSE
                    ORDER BY annotation_timestamp DESC
                    LIMIT ?
                ''', (limit,))
                
                results = cursor.fetchall()
                
                training_data = []
                case_ids = []
                
                for row in results:
                    case_id = row[0]
                    image_data = pickle.loads(row[1])
                    annotation = json.loads(row[2])
                    
                    training_data.append((image_data, annotation))
                    case_ids.append(case_id)
                
                # Mark as used for retraining
                if case_ids:
                    placeholders = ','.join(['?' for _ in case_ids])
                    cursor.execute(f'''
                        UPDATE uncertain_cases 
                        SET retrain_used = TRUE 
                        WHERE case_id IN ({placeholders})
                    ''', case_ids)
                    
                    conn.commit()
                
                return training_data
                
        except Exception as e:
            logging.error(f"❌ Failed to get annotated cases: {str(e)}")
            return []
    
    def _calculate_annotation_quality(
        self, 
        annotation_time: float, 
        confidence_score: float
    ) -> float:
        """
        Calculate annotation quality score
        """
        # Simple quality heuristic
        # Optimal annotation time: 30-120 seconds
        # High confidence is good
        
        time_score = 1.0
        if annotation_time < 10:  # Too fast, likely careless
            time_score = 0.5
        elif annotation_time > 300:  # Too slow, might be distracted
            time_score = 0.8
        
        quality = (time_score * 0.4) + (confidence_score * 0.6)
        return min(1.0, max(0.0, quality))
    
    def get_pending_count(self) -> int:
        """Get number of pending cases"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM uncertain_cases WHERE status = "pending"')
                return cursor.fetchone()[0]
        except:
            return 0
    
    def is_connected(self) -> bool:
        """Check database connection"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                return True
        except:
            return False