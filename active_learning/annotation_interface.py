# active_learning/annotation_interface.py
#ANNOTATION_INTERFACE = '''
"""
Expert Annotation Interface for Medical Images with Quality Control
"""

import streamlit as st
import sqlite3
import json
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import cv2
import pickle
from dataclasses import dataclass
from enum import Enum

class AnnotationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVIEWED = "reviewed"
    REJECTED = "rejected"

@dataclass
class AnnotationTask:
    case_id: str
    image_data: np.ndarray
    ai_prediction: Dict[str, Any]
    priority: int
    patient_id: Optional[str] = None
    clinical_history: Optional[str] = None
    status: AnnotationStatus = AnnotationStatus.PENDING
    assigned_annotator: Optional[str] = None
    created_at: datetime = None
    deadline: datetime = None

class MedicalAnnotationInterface:
    """
    Professional annotation interface for medical experts
    """
    
    def __init__(self, db_path: str = "annotation_interfaces.db"):
        self.db_path = db_path
        self._initialize_database()
        self.annotation_guidelines = self._load_annotation_guidelines()
        
        logging.info("✅ Medical annotation interface initialized")
    
    def _initialize_database(self):
        """Initialize annotation interface database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Annotation tasks table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS annotation_tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        case_id TEXT UNIQUE,
                        image_data BLOB,
                        ai_prediction TEXT,
                        priority INTEGER,
                        patient_id TEXT,
                        clinical_history TEXT,
                        status TEXT,
                        assigned_annotator TEXT,
                        created_at TEXT,
                        deadline TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Expert annotations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS expert_annotations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        case_id TEXT,
                        annotator_id TEXT,
                        annotation_data TEXT,
                        confidence_score REAL,
                        annotation_time_seconds REAL,
                        comments TEXT,
                        review_notes TEXT,
                        quality_score REAL,
                        status TEXT,
                        created_at TEXT,
                        reviewed_at TEXT,
                        reviewed_by TEXT,
                        FOREIGN KEY (case_id) REFERENCES annotation_tasks (case_id)
                    )
                ''')
                
                # Annotator performance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS annotator_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        annotator_id TEXT,
                        total_annotations INTEGER,
                        avg_quality_score REAL,
                        avg_annotation_time REAL,
                        accuracy_score REAL,
                        consistency_score REAL,
                        last_updated TEXT
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Annotation interface DB initialization failed: {str(e)}")
            raise
    
    def _load_annotation_guidelines(self) -> Dict[str, Any]:
        """Load medical annotation guidelines"""
        return {
            "nodule_definition": {
                "min_size_mm": 3,
                "max_size_mm": 30,
                "description": "Round or oval opacity, well or poorly defined"
            },
            "classification_criteria": {
                "definite_nodule": "Clear, well-defined opacity consistent with pulmonary nodule",
                "probable_nodule": "Opacity likely representing nodule, but with some uncertainty",
                "possible_nodule": "Questionable opacity that might represent a nodule",
                "not_nodule": "Opacity clearly not representing a nodule (vessel, artifact, etc.)"
            },
            "quality_requirements": {
                "annotation_time_range": (30, 300),  # seconds
                "required_confidence": 0.7,
                "mandatory_fields": ["has_nodule", "confidence", "location"]
            }
        }
    
    def create_annotation_task(
        self,
        case_id: str,
        image_data: np.ndarray,
        ai_prediction: Dict[str, Any],
        priority: int = 1,
        patient_id: Optional[str] = None,
        clinical_history: Optional[str] = None,
        assigned_annotator: Optional[str] = None
    ) -> bool:
        """Create new annotation task"""
        try:
            # Serialize image data
            image_blob = pickle.dumps(image_data)
            prediction_json = json.dumps(ai_prediction)
            
            # Set deadline based on priority
            deadline = datetime.now()
            if priority == 1:  # Low priority
                deadline = deadline.replace(hour=23, minute=59, second=59)  # End of day
            elif priority == 2:  # Medium priority
                deadline = deadline.replace(hour=deadline.hour + 4)  # 4 hours
            else:  # High priority
                deadline = deadline.replace(hour=deadline.hour + 1)  # 1 hour
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO annotation_tasks 
                    (case_id, image_data, ai_prediction, priority, patient_id,
                     clinical_history, status, assigned_annotator, created_at, deadline)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    case_id, image_blob, prediction_json, priority, patient_id,
                    clinical_history, AnnotationStatus.PENDING.value,
                    assigned_annotator, datetime.now().isoformat(),
                    deadline.isoformat()
                ))
                conn.commit()
            
            logging.info(f"✅ Created annotation task: {case_id}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to create annotation task: {str(e)}")
            return False
    
    def get_pending_tasks(
        self, 
        annotator_id: Optional[str] = None,
        limit: int = 20
    ) -> List[AnnotationTask]:
        """Get pending annotation tasks"""
        try:
            # First, try to get from annotation_tasks table
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT case_id, image_data, ai_prediction, priority,
                        patient_id, clinical_history, status, assigned_annotator,
                        created_at, deadline
                    FROM annotation_tasks 
                    WHERE status = ?
                '''
                params = [AnnotationStatus.PENDING.value]
                
                if annotator_id:
                    query += ' AND (assigned_annotator = ? OR assigned_annotator IS NULL)'
                    params.append(annotator_id)
                
                query += ' ORDER BY priority DESC, created_at ASC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                results = cursor.fetchall()
            
            # If no tasks found, try uncertain_cases.db directly
            if not results:
                try:
                    with sqlite3.connect("uncertain_cases.db") as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            SELECT case_id, image_data, prediction, priority, 
                                patient_id, clinical_history
                            FROM uncertain_cases 
                            WHERE status = 'pending'
                            ORDER BY priority DESC, timestamp ASC 
                            LIMIT ?
                        ''', (limit,))
                        
                        uncertain_results = cursor.fetchall()
                        
                        tasks = []
                        for row in uncertain_results:
                            image_data = pickle.loads(row[1])
                            ai_prediction = json.loads(row[2])
                            
                            task = AnnotationTask(
                                case_id=row[0],
                                image_data=image_data,
                                ai_prediction=ai_prediction,
                                priority=row[3],
                                patient_id=row[4],
                                clinical_history=row[5],
                                status=AnnotationStatus.PENDING
                            )
                            tasks.append(task)
                        
                        return tasks
                except Exception as e:
                    logging.error(f"Failed to get from uncertain_cases.db: {e}")
                    return []
            
            # Process annotation_tasks results
            tasks = []
            for row in results:
                image_data = pickle.loads(row[1])
                ai_prediction = json.loads(row[2])
                
                task = AnnotationTask(
                    case_id=row[0],
                    image_data=image_data,
                    ai_prediction=ai_prediction,
                    priority=row[3],
                    patient_id=row[4],
                    clinical_history=row[5],
                    status=AnnotationStatus(row[6]),
                    assigned_annotator=row[7],
                    created_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    deadline=datetime.fromisoformat(row[9]) if row[9] else None
                )
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logging.error(f"❌ Failed to get pending tasks: {str(e)}")
            return []



    # def get_pending_tasks(
    #     self, 
    #     annotator_id: Optional[str] = None,
    #     limit: int = 20
    # ) -> List[AnnotationTask]:
    #     """Get pending annotation tasks"""
    #     try:
    #         with sqlite3.connect(self.db_path) as conn:
    #             cursor = conn.cursor()
                
    #             query = '''
    #                 SELECT case_id, image_data, ai_prediction, priority,
    #                        patient_id, clinical_history, status, assigned_annotator,
    #                        created_at, deadline
    #                 FROM annotation_tasks 
    #                 WHERE status = ?
    #             '''
    #             params = [AnnotationStatus.PENDING.value]
                
    #             if annotator_id:
    #                 query += ' AND (assigned_annotator = ? OR assigned_annotator IS NULL)'
    #                 params.append(annotator_id)
                
    #             query += ' ORDER BY priority DESC, created_at ASC LIMIT ?'
    #             params.append(limit)
                
    #             cursor.execute(query, params)
    #             results = cursor.fetchall()
                
    #             tasks = []
    #             for row in results:
    #                 image_data = pickle.loads(row[1])
    #                 ai_prediction = json.loads(row[2])
                    
    #                 task = AnnotationTask(
    #                     case_id=row[0],
    #                     image_data=image_data,
    #                     ai_prediction=ai_prediction,
    #                     priority=row[3],
    #                     patient_id=row[4],
    #                     clinical_history=row[5],
    #                     status=AnnotationStatus(row[6]),
    #                     assigned_annotator=row[7],
    #                     created_at=datetime.fromisoformat(row[8]) if row[8] else None,
    #                     deadline=datetime.fromisoformat(row[9]) if row[9] else None
    #                 )
    #                 tasks.append(task)
                
    #             return tasks
                
    #     except Exception as e:
    #         logging.error(f"❌ Failed to get pending tasks: {str(e)}")
    #         return []
    
    def render_annotation_interface(self, annotator_id: str) -> Optional[Dict[str, Any]]:
        """Render Streamlit annotation interface"""
        st.title("🩺 Medical Image Annotation Interface")
        st.markdown("**Expert Review of Uncertain AI Predictions**")
        
        # Get pending tasks
        pending_tasks = self.get_pending_tasks(annotator_id)
        
        if not pending_tasks:
            st.info("✅ No pending annotation tasks!")
            return None
        
        # Task selection
        st.subheader("📋 Pending Tasks")
        task_options = [f"Case {task.case_id[:8]}... (Priority: {task.priority})" 
                       for task in pending_tasks]
        
        selected_idx = st.selectbox("Select task to annotate:", range(len(task_options)), 
                                   format_func=lambda x: task_options[x])
        
        if selected_idx is None:
            return None
        
        selected_task = pending_tasks[selected_idx]
        
        # Mark task as in progress
        self._update_task_status(selected_task.case_id, AnnotationStatus.IN_PROGRESS, annotator_id)
        
        # Display task details
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🖼️ Medical Image")
            
            # # DEBUG INFO 
            # st.write(f"**Debug Info:**")
            # st.write(f"- Image data type: {type(selected_task.image_data)}")
            # st.write(f"- Image shape: {selected_task.image_data.shape if hasattr(selected_task.image_data, 'shape') else 'No shape'}")
            # st.write(f"- Image dtype: {selected_task.image_data.dtype if hasattr(selected_task.image_data, 'dtype') else 'No dtype'}")
            # if hasattr(selected_task.image_data, 'min'):
            #     st.write(f"- Image range: {selected_task.image_data.min():.3f} to {selected_task.image_data.max():.3f}")
            
            # FIXED IMAGE PROCESSING
            display_image = selected_task.image_data
            
            # Fix tensor format: (1, 3, H, W) -> (H, W, 3) or (H, W)
            if len(display_image.shape) == 4 and display_image.shape[0] == 1:
                # Remove batch dimension: (1, C, H, W) -> (C, H, W)
                display_image = display_image[0]
            
            if len(display_image.shape) == 3 and display_image.shape[0] in [1, 3]:
                # Convert from (C, H, W) to (H, W, C)
                display_image = np.transpose(display_image, (1, 2, 0))
            
            # Handle grayscale: (H, W, 1) -> (H, W)
            if len(display_image.shape) == 3 and display_image.shape[2] == 1:
                display_image = display_image.squeeze(axis=2)
            
            # Convert to numpy array if needed
            display_image = np.array(display_image)
            
            # Denormalize if needed (common ImageNet normalization)
            if display_image.min() < 0 and display_image.max() > 1:
                # Reverse ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                # If grayscale, use single channel
                if len(display_image.shape) == 2:
                    mean = mean[0]  # Use first channel mean
                    std = std[0]    # Use first channel std
                elif display_image.shape[2] == 3:
                    mean = mean.reshape(1, 1, 3)
                    std = std.reshape(1, 1, 3)
                
                display_image = display_image * std + mean
                display_image = np.clip(display_image, 0, 1)
            
            # Ensure values are in [0, 1] range
            if display_image.max() > 1.0:
                display_image = display_image / 255.0
            
            # Clip to valid range
            display_image = np.clip(display_image, 0.0, 1.0)
            
            # Convert grayscale to RGB for display
            if len(display_image.shape) == 2:
                display_image = np.stack([display_image] * 3, axis=-1)
            
            # st.write(f"**After processing:**")
            # st.write(f"- Shape: {display_image.shape}")
            # st.write(f"- Range: {display_image.min():.3f} to {display_image.max():.3f}")
            
            st.image(display_image, caption=f"Case {selected_task.case_id}", use_container_width=True)

        # with col1:
        #     st.subheader("🖼️ Medical Image")
        #     # DEBUG INFO - Add this to see what we're working with
        #     st.write(f"**Debug Info:**")
        #     st.write(f"- Image data type: {type(selected_task.image_data)}")
        #     st.write(f"- Image shape: {selected_task.image_data.shape if hasattr(selected_task.image_data, 'shape') else 'No shape'}")
        #     st.write(f"- Image dtype: {selected_task.image_data.dtype if hasattr(selected_task.image_data, 'dtype') else 'No dtype'}")
        #     if hasattr(selected_task.image_data, 'min'):
        #         st.write(f"- Image range: {selected_task.image_data.min():.3f} to {selected_task.image_data.max():.3f}")
        #     # Display image
        #     if len(selected_task.image_data.shape) == 3:
        #         display_image = selected_task.image_data
        #     else:
        #         display_image = cv2.cvtColor(selected_task.image_data, cv2.COLOR_GRAY2RGB)

        #     # Convert to float32 numpy array if needed
        #     display_image = np.array(display_image).astype(np.float32)

        #     # Normalize pixel values to [0, 1] if outside this range
        #     if display_image.max() > 1.0:
        #         display_image = display_image / 255.0

        #     # Clip to [0,1] just in case
        #     display_image = np.clip(display_image, 0.0, 1.0)

        #     st.image(display_image, caption=f"Case {selected_task.case_id}", use_container_width=True)


        #     # if len(selected_task.image_data.shape) == 3:
        #     #     display_image = selected_task.image_data
        #     # else:
        #     #     display_image = cv2.cvtColor(selected_task.image_data, cv2.COLOR_GRAY2RGB)
            
        #     # st.image(display_image, caption=f"Case {selected_task.case_id}", 
        #     #         use_container_width=True)
            
        #     # AI Prediction overlay
        #     if st.checkbox("Show AI Prediction Overlay"):
        #         self._render_ai_overlay(display_image, selected_task.ai_prediction)
        
        with col2:
            st.subheader("🤖 AI Analysis")
            ai_pred = selected_task.ai_prediction
            
            # AI prediction summary
            prob = ai_pred.get('probability', 0)
            confidence = ai_pred.get('confidence', 0)
            uncertainty = ai_pred.get('uncertainty_level', 'Unknown')
            
            st.metric("AI Probability", f"{prob:.3f}")
            st.metric("AI Confidence", f"{confidence:.3f}")
            st.metric("Uncertainty Level", uncertainty)
            
            if prob > 0.5:
                st.error(f"🚨 AI: Nodule Detected")
            else:
                st.success(f"✅ AI: No Nodule")
            
            # Patient information
            if selected_task.patient_id:
                st.write(f"**Patient ID:** {selected_task.patient_id}")
            
            if selected_task.clinical_history:
                st.write(f"**Clinical History:** {selected_task.clinical_history}")
        
        # Annotation form
        st.subheader("👨‍⚕️ Expert Annotation")
        
        annotation_start_time = datetime.now()
        
        with st.form("annotation_form"):
            # Primary annotation
            has_nodule = st.radio(
                "Expert Assessment:",
                options=[True, False, None],
                format_func=lambda x: {
                    True: "✅ Nodule Present",
                    False: "❌ No Nodule",
                    None: "❓ Uncertain/Unclear"
                }[x],
                help="Your expert assessment of nodule presence"
            )
            
            # Confidence in annotation
            confidence_score = st.slider(
                "Confidence in Assessment:",
                min_value=0.1, max_value=1.0, value=0.8, step=0.1,
                help="How confident are you in your assessment?"
            )
            
            # Nodule characteristics (if present)
            nodule_details = {}
            if has_nodule:
                st.write("**Nodule Characteristics:**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    nodule_size = st.number_input("Approximate Size (mm):", 
                                                 min_value=0, max_value=50, value=8)
                    nodule_details['size_mm'] = nodule_size
                
                with col_b:
                    nodule_type = st.selectbox("Nodule Type:", [
                        "Solid", "Ground-glass", "Part-solid", "Calcified", "Not Present"
                    ])
                    nodule_details['type'] = nodule_type
                
                # Location marking (simplified)
                location_description = st.text_area(
                    "Location Description:",
                    placeholder="e.g., Right upper lobe, peripheral",
                    help="Describe the location of the nodule"
                )
                nodule_details['location'] = location_description
            
            # AI agreement assessment
            ai_agreement = st.radio(
                "Agreement with AI:",
                options=["Agree", "Partially Agree", "Disagree"],
                help="Do you agree with the AI's assessment?"
            )
            
            # Comments and notes
            comments = st.text_area(
                "Comments/Notes:",
                placeholder="Any additional observations, concerns, or explanations...",
                help="Optional comments about this case"
            )
            
            # Quality indicators
            image_quality = st.select_slider(
                "Image Quality:",
                options=["Poor", "Fair", "Good", "Excellent"],
                value="Good"
            )
            
            # Submit annotation
            submitted = st.form_submit_button("💾 Submit Annotation")
            
            if submitted:
                annotation_end_time = datetime.now()
                annotation_time = (annotation_end_time - annotation_start_time).total_seconds()
                
                # Compile annotation data
                annotation_data = {
                    "has_nodule": has_nodule,
                    "confidence": confidence_score,
                    "nodule_details": nodule_details if has_nodule else {},
                    "ai_agreement": ai_agreement,
                    "image_quality": image_quality,
                    "annotation_method": "expert_review"
                }
                
                # Submit annotation
                success = self.submit_annotation(
                    case_id=selected_task.case_id,
                    annotator_id=annotator_id,
                    annotation_data=annotation_data,
                    confidence_score=confidence_score,
                    annotation_time=annotation_time,
                    comments=comments
                )
                
                if success:
                    st.success("✅ Annotation submitted successfully!")
                    st.balloons()
                    return annotation_data
                else:
                    st.error("❌ Failed to submit annotation. Please try again.")
        
        return None
    
    def submit_annotation(
        self,
        case_id: str,
        annotator_id: str,
        annotation_data: Dict[str, Any],
        confidence_score: float,
        annotation_time: float,
        comments: Optional[str] = None
    ) -> bool:
        """Submit expert annotation"""
        try:
            # Calculate quality score
            quality_score = self._calculate_annotation_quality(
                annotation_time, confidence_score, annotation_data
            )
            
            annotation_json = json.dumps(annotation_data)
            timestamp = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert annotation
                cursor.execute('''
                    INSERT INTO expert_annotations 
                    (case_id, annotator_id, annotation_data, confidence_score,
                     annotation_time_seconds, comments, quality_score, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    case_id, annotator_id, annotation_json, confidence_score,
                    annotation_time, comments, quality_score, "completed", timestamp
                ))
                
                # Update task status
                cursor.execute('''
                    UPDATE annotation_tasks 
                    SET status = ? 
                    WHERE case_id = ?
                ''', (AnnotationStatus.COMPLETED.value, case_id))
                
                conn.commit()
            
            # Update annotator performance
            self._update_annotator_performance(annotator_id, quality_score, annotation_time)
            
            logging.info(f"✅ Annotation submitted for case: {case_id}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to submit annotation: {str(e)}")
            return False
    
    def _render_ai_overlay(self, image: np.ndarray, ai_prediction: Dict[str, Any]):
        """Render AI prediction overlay on image"""
        # This would render heatmaps, bounding boxes, etc.
        # Simplified for now
        st.write("🤖 AI Prediction Overlay:")
        st.write(f"- Probability: {ai_prediction.get('probability', 0):.3f}")
        st.write(f"- Uncertainty: {ai_prediction.get('uncertainty_level', 'Unknown')}")
        
        # In a full implementation, this would overlay Grad-CAM heatmaps
        if 'explanation' in ai_prediction and ai_prediction['explanation']:
            st.write("- Grad-CAM explanation available")
    
    def _update_task_status(self, case_id: str, status: AnnotationStatus, annotator_id: str):
        """Update task status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE annotation_tasks 
                    SET status = ?, assigned_annotator = ? 
                    WHERE case_id = ?
                ''', (status.value, annotator_id, case_id))
                conn.commit()
        except Exception as e:
            logging.error(f"❌ Failed to update task status: {str(e)}")
    
    def _calculate_annotation_quality(
        self, 
        annotation_time: float, 
        confidence_score: float,
        annotation_data: Dict[str, Any]
    ) -> float:
        """Calculate quality score for annotation"""
        quality_score = 0.0
        
        # Time-based quality (30%)
        guidelines = self.annotation_guidelines['quality_requirements']
        min_time, max_time = guidelines['annotation_time_range']
        
        if min_time <= annotation_time <= max_time:
            time_score = 1.0
        elif annotation_time < min_time:
            time_score = 0.5  # Too fast, potentially careless
        else:
            time_score = max(0.3, 1.0 - (annotation_time - max_time) / 300)  # Too slow
        
        quality_score += time_score * 0.3
        
        # Confidence-based quality (40%)
        confidence_score_normalized = min(1.0, confidence_score / guidelines['required_confidence'])
        quality_score += confidence_score_normalized * 0.4
        
        # Completeness-based quality (30%)
        mandatory_fields = guidelines['mandatory_fields']
        completed_fields = sum(1 for field in mandatory_fields if field in annotation_data)
        completeness_score = completed_fields / len(mandatory_fields)
        quality_score += completeness_score * 0.3
        
        return min(1.0, quality_score)
    
    def _update_annotator_performance(self, annotator_id: str, quality_score: float, annotation_time: float):
        """Update annotator performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current performance
                cursor.execute('''
                    SELECT total_annotations, avg_quality_score, avg_annotation_time
                    FROM annotator_performance 
                    WHERE annotator_id = ?
                ''', (annotator_id,))
                
                result = cursor.fetchone()
                
                if result:
                    # Update existing record
                    total, avg_quality, avg_time = result
                    new_total = total + 1
                    new_avg_quality = (avg_quality * total + quality_score) / new_total
                    new_avg_time = (avg_time * total + annotation_time) / new_total
                    
                    cursor.execute('''
                        UPDATE annotator_performance 
                        SET total_annotations = ?, avg_quality_score = ?, 
                            avg_annotation_time = ?, last_updated = ?
                        WHERE annotator_id = ?
                    ''', (new_total, new_avg_quality, new_avg_time, 
                          datetime.now().isoformat(), annotator_id))
                else:
                    # Create new record
                    cursor.execute('''
                        INSERT INTO annotator_performance 
                        (annotator_id, total_annotations, avg_quality_score, 
                         avg_annotation_time, last_updated)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (annotator_id, 1, quality_score, annotation_time,
                          datetime.now().isoformat()))
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Failed to update annotator performance: {str(e)}")
    
    def get_annotator_performance(self, annotator_id: str) -> Dict[str, Any]:
        """Get annotator performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT total_annotations, avg_quality_score, avg_annotation_time,
                           accuracy_score, consistency_score, last_updated
                    FROM annotator_performance 
                    WHERE annotator_id = ?
                ''', (annotator_id,))
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        "annotator_id": annotator_id,
                        "total_annotations": result[0],
                        "avg_quality_score": result[1],
                        "avg_annotation_time": result[2],
                        "accuracy_score": result[3],
                        "consistency_score": result[4],
                        "last_updated": result[5]
                    }
                else:
                    return {"annotator_id": annotator_id, "total_annotations": 0}
                    
        except Exception as e:
            logging.error(f"❌ Failed to get annotator performance: {str(e)}")
            return {"error": str(e)}
    
    def get_completed_annotations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get completed annotations for training"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT t.case_id, t.image_data, a.annotation_data, a.quality_score
                    FROM annotation_tasks t
                    JOIN expert_annotations a ON t.case_id = a.case_id
                    WHERE a.status = 'completed' AND a.quality_score >= 0.7
                    ORDER BY a.created_at DESC
                    LIMIT ?
                ''', (limit,))
                
                results = cursor.fetchall()
                
                annotations = []
                for row in results:
                    try:
                        image_data = pickle.loads(row[1])
                        annotation_data = json.loads(row[2])
                        
                        annotations.append({
                            "case_id": row[0],
                            "image_data": image_data,
                            "annotation": annotation_data,
                            "quality_score": row[3]
                        })
                    except Exception as e:
                        logging.warning(f"⚠️ Failed to load annotation {row[0]}: {str(e)}")
                        continue
                
                return annotations
                
        except Exception as e:
            logging.error(f"❌ Failed to get completed annotations: {str(e)}")
            return []