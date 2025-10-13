# ========================================================================
# active_learning/quality_controller.py
# ========================================================================

#QUALITY_CONTROLLER = '''
"""
Annotation Quality Controller with Multi-level Review Process
"""

import sqlite3
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from enum import Enum
from scipy import stats
import cv2

class QualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    REJECTED = "rejected"

class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"

class AnnotationQualityController:
    """
    Comprehensive quality control system for medical annotations
    """
    
    def __init__(self, quality_db: str = "annotation_quality.db"):
        self.quality_db = quality_db
        self.quality_thresholds = {
            'min_annotation_time': 15,  # seconds
            'max_annotation_time': 600,  # seconds
            'min_confidence': 0.6,
            'consistency_threshold': 0.8,
            'agreement_threshold': 0.7
        }
        
        self._initialize_quality_db()
        logging.info("✅ Annotation quality controller initialized")
    
    def _initialize_quality_db(self):
        """Initialize quality control database"""
        try:
            with sqlite3.connect(self.quality_db) as conn:
                cursor = conn.cursor()
                
                # Quality reviews table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS quality_reviews (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        case_id TEXT,
                        annotation_id INTEGER,
                        reviewer_id TEXT,
                        quality_level TEXT,
                        review_status TEXT,
                        quality_score REAL,
                        review_comments TEXT,
                        issues_found TEXT,
                        recommendations TEXT,
                        created_at TEXT,
                        reviewed_at TEXT
                    )
                ''')
                
                # Inter-annotator agreement table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS inter_annotator_agreement (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        case_id TEXT,
                        annotator1_id TEXT,
                        annotator2_id TEXT,
                        annotation1_data TEXT,
                        annotation2_data TEXT,
                        agreement_score REAL,
                        disagreement_areas TEXT,
                        consensus_reached BOOLEAN,
                        final_annotation TEXT,
                        created_at TEXT
                    )
                ''')
                
                # Quality metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS quality_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        period_start TEXT,
                        period_end TEXT,
                        total_annotations INTEGER,
                        quality_distribution TEXT,
                        avg_quality_score REAL,
                        annotator_performance TEXT,
                        improvement_suggestions TEXT,
                        created_at TEXT
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"❌ Quality DB initialization failed: {str(e)}")
            raise
    
    def evaluate_annotation_quality(
        self, 
        case_id: str,
        annotation_data: Dict[str, Any],
        annotation_time: float,
        annotator_id: str,
        ai_prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive quality evaluation of annotation
        """
        try:
            quality_issues = []
            quality_score = 0.0
            max_score = 100.0
            
            # 1. Time-based quality assessment (20 points)
            time_score = self._evaluate_annotation_time(annotation_time)
            quality_score += time_score
            
            if time_score < 15:
                if annotation_time < self.quality_thresholds['min_annotation_time']:
                    quality_issues.append("Annotation completed too quickly - possible lack of thorough review")
                else:
                    quality_issues.append("Annotation took excessive time - possible distraction or uncertainty")
            
            # 2. Completeness assessment (25 points)
            completeness_score = self._evaluate_completeness(annotation_data)
            quality_score += completeness_score
            
            if completeness_score < 20:
                quality_issues.append("Missing required annotation fields")
            
            # 3. Consistency assessment (25 points)
            consistency_score = self._evaluate_consistency(annotation_data, annotator_id)
            quality_score += consistency_score
            
            if consistency_score < 20:
                quality_issues.append("Annotation inconsistent with annotator's historical patterns")
            
            # 4. Clinical plausibility (20 points)
            clinical_score = self._evaluate_clinical_plausibility(annotation_data)
            quality_score += clinical_score
            
            if clinical_score < 15:
                quality_issues.append("Annotation contains clinically implausible assessments")
            
            # 5. AI-Human agreement analysis (10 points)
            if ai_prediction:
                agreement_score = self._evaluate_ai_agreement(annotation_data, ai_prediction)
                quality_score += agreement_score
                
                if agreement_score < 5:
                    quality_issues.append("Significant disagreement with AI without clear justification")
            else:
                quality_score += 5  # Neutral score if no AI prediction available
            
            # Determine quality level
            quality_level = self._determine_quality_level(quality_score)
            
            # Determine if review is needed
            needs_review = self._determine_review_requirement(
                quality_score, quality_level, quality_issues
            )
            
            evaluation_result = {
                "case_id": case_id,
                "annotator_id": annotator_id,
                "quality_score": quality_score,
                "max_score": max_score,
                "quality_percentage": (quality_score / max_score) * 100,
                "quality_level": quality_level.value,
                "quality_issues": quality_issues,
                "needs_review": needs_review,
                "review_priority": self._calculate_review_priority(quality_score, quality_issues),
                "evaluation_timestamp": datetime.now().isoformat(),
                "component_scores": {
                    "time_score": time_score,
                    "completeness_score": completeness_score,
                    "consistency_score": consistency_score,
                    "clinical_score": clinical_score,
                    "agreement_score": agreement_score if ai_prediction else 5
                }
            }
            
            # Log quality evaluation
            self._log_quality_evaluation(evaluation_result)
            
            return evaluation_result
            
        except Exception as e:
            logging.error(f"❌ Quality evaluation failed: {str(e)}")
            return {
                "case_id": case_id,
                "quality_score": 0.0,
                "quality_level": QualityLevel.REJECTED.value,
                "quality_issues": [f"Quality evaluation error: {str(e)}"],
                "needs_review": True
            }
    
    def conduct_peer_review(
        self,
        case_id: str,
        original_annotation: Dict[str, Any],
        reviewer_id: str,
        review_annotation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Conduct peer review between two annotations
        """
        try:
            # Calculate agreement metrics
            agreement_metrics = self._calculate_inter_annotator_agreement(
                original_annotation, review_annotation
            )
            
            # Determine consensus
            consensus_result = self._determine_consensus(
                original_annotation, review_annotation, agreement_metrics
            )
            
            # Generate review report
            review_result = {
                "case_id": case_id,
                "agreement_score": agreement_metrics['overall_agreement'],
                "specific_agreements": agreement_metrics['specific_agreements'],
                "disagreement_areas": agreement_metrics['disagreements'],
                "consensus_reached": consensus_result['consensus_reached'],
                "final_annotation": consensus_result['final_annotation'],
                "review_notes": consensus_result['notes'],
                "reviewer_id": reviewer_id,
                "review_timestamp": datetime.now().isoformat()
            }
            
            # Log peer review
            self._log_peer_review(review_result, original_annotation, review_annotation)
            
            return review_result
            
        except Exception as e:
            logging.error(f"❌ Peer review failed: {str(e)}")
            return {
                "case_id": case_id,
                "error": str(e),
                "consensus_reached": False
            }
    
    def _evaluate_annotation_time(self, annotation_time: float) -> float:
        """Evaluate quality based on annotation time"""
        min_time = self.quality_thresholds['min_annotation_time']
        max_time = self.quality_thresholds['max_annotation_time']
        
        if min_time <= annotation_time <= max_time:
            return 20.0
        elif annotation_time < min_time:
            # Too fast - penalize more severely
            return max(0.0, 20.0 * (annotation_time / min_time) * 0.5)
        else:
            # Too slow - gradual penalty
            excess_time = annotation_time - max_time
            penalty = min(15.0, excess_time / 60.0 * 5.0)  # 5 points per minute over
            return max(5.0, 20.0 - penalty)
    
    def _evaluate_completeness(self, annotation_data: Dict[str, Any]) -> float:
        """Evaluate annotation completeness"""
        required_fields = ['has_nodule', 'confidence']
        optional_important_fields = ['nodule_details', 'image_quality']
        
        score = 0.0
        
        # Required fields (20 points)
        completed_required = sum(1 for field in required_fields if field in annotation_data)
        score += (completed_required / len(required_fields)) * 20.0
        
        # Optional important fields (5 points)
        completed_optional = sum(1 for field in optional_important_fields if field in annotation_data)
        score += (completed_optional / len(optional_important_fields)) * 5.0
        
        # Bonus for detailed nodule characteristics if nodule present
        if annotation_data.get('has_nodule') and 'nodule_details' in annotation_data:
            nodule_details = annotation_data['nodule_details']
            detail_fields = ['size_mm', 'type', 'location']
            completed_details = sum(1 for field in detail_fields if field in nodule_details)
            if completed_details >= 2:
                score += 2.0  # Bonus points
        
        return min(25.0, score)
    
    def _evaluate_consistency(self, annotation_data: Dict[str, Any], annotator_id: str) -> float:
        """Evaluate consistency with annotator's historical patterns"""
        try:
            # Get historical annotations for this annotator
            historical_annotations = self._get_annotator_history(annotator_id, limit=50)
            
            if len(historical_annotations) < 5:
                return 20.0  # Neutral score for new annotators
            
            # Analyze consistency patterns
            current_confidence = annotation_data.get('confidence', 0.5)
            historical_confidences = [ann.get('confidence', 0.5) for ann in historical_annotations]
            
            # Confidence consistency
            confidence_std = np.std(historical_confidences + [current_confidence])
            confidence_score = max(0, 15 - confidence_std * 20)  # Lower std = higher score
            
            # Decision pattern consistency
            current_decision = annotation_data.get('has_nodule')
            if current_decision is not None:
                historical_decisions = [ann.get('has_nodule') for ann in historical_annotations 
                                      if ann.get('has_nodule') is not None]
                
                if len(historical_decisions) > 0:
                    positive_rate = sum(historical_decisions) / len(historical_decisions)
                    # Check if current decision fits typical pattern
                    if (positive_rate > 0.7 and current_decision) or \
                       (positive_rate < 0.3 and not current_decision) or \
                       (0.3 <= positive_rate <= 0.7):
                        decision_score = 10.0
                    else:
                        decision_score = 5.0
                else:
                    decision_score = 8.0
            else:
                decision_score = 0.0
            
            return confidence_score + decision_score
            
        except Exception as e:
            logging.warning(f"⚠️ Consistency evaluation failed: {str(e)}")
            return 15.0  # Neutral score on error
    
    def _evaluate_clinical_plausibility(self, annotation_data: Dict[str, Any]) -> float:
        """Evaluate clinical plausibility of annotation"""
        score = 20.0
        
        # Check confidence vs decision consistency
        confidence = annotation_data.get('confidence', 0.5)
        has_nodule = annotation_data.get('has_nodule')
        
        if has_nodule is not None:
            # High confidence should match clear decisions
            if confidence > 0.8:
                score += 2.0  # Bonus for high confidence
            elif confidence < 0.6 and has_nodule is not None:
                score -= 5.0  # Penalty for low confidence with definitive decision
        
        # Check nodule characteristics plausibility
        if has_nodule and 'nodule_details' in annotation_data:
            nodule_details = annotation_data['nodule_details']
            
            # Size plausibility
            if 'size_mm' in nodule_details:
                size = nodule_details['size_mm']
                if not (1 <= size <= 50):  # Reasonable size range
                    score -= 8.0
                elif not (3 <= size <= 30):  # Standard nodule range
                    score -= 3.0
            
            # Type consistency
            if 'type' in nodule_details:
                valid_types = ['Solid', 'Ground-glass', 'Part-solid', 'Calcified']
                if nodule_details['type'] not in valid_types:
                    score -= 5.0
        
        return max(0.0, score)
    
    def _evaluate_ai_agreement(
        self, 
        annotation_data: Dict[str, Any], 
        ai_prediction: Dict[str, Any]
    ) -> float:
        """Evaluate agreement between human annotation and AI prediction"""
        try:
            score = 5.0  # Base score
            
            human_decision = annotation_data.get('has_nodule')
            ai_probability = ai_prediction.get('probability', 0.5)
            ai_decision = ai_probability > 0.5
            
            if human_decision is not None:
                # Perfect agreement
                if human_decision == ai_decision:
                    score = 10.0
                else:
                    # Disagreement - check if it's justified
                    human_confidence = annotation_data.get('confidence', 0.5)
                    ai_confidence = ai_prediction.get('confidence', 0.5)
                    
                    # If both are confident but disagree, it's more concerning
                    if human_confidence > 0.8 and ai_confidence > 0.8:
                        score = 2.0  # Significant disagreement
                    elif ai_prediction.get('uncertainty_level') == 'High':
                        score = 7.0  # Disagreement with uncertain AI is more acceptable
                    else:
                        score = 4.0  # Moderate disagreement
            
            return score
            
        except Exception as e:
            logging.warning(f"⚠️ AI agreement evaluation failed: {str(e)}")
            return 5.0
    
    def _determine_quality_level(self, quality_score: float) -> QualityLevel:
        """Determine quality level based on score"""
        percentage = quality_score
        
        if percentage >= 90:
            return QualityLevel.EXCELLENT
        elif percentage >= 80:
            return QualityLevel.GOOD
        elif percentage >= 70:
            return QualityLevel.FAIR
        elif percentage >= 60:
            return QualityLevel.POOR
        else:
            return QualityLevel.REJECTED
    
    def _determine_review_requirement(
        self, 
        quality_score: float, 
        quality_level: QualityLevel,
        issues: List[str]
    ) -> bool:
        """Determine if annotation needs expert review"""
        if quality_level in [QualityLevel.POOR, QualityLevel.REJECTED]:
            return True
        
        if quality_score < 75:
            return True
        
        # Check for specific concerning issues
        concerning_keywords = ['inconsistent', 'disagreement', 'implausible', 'too quickly']
        for issue in issues:
            if any(keyword in issue.lower() for keyword in concerning_keywords):
                return True
        
        return False
    
    def _calculate_review_priority(self, quality_score: float, issues: List[str]) -> int:
        """Calculate review priority (1=low, 2=medium, 3=high)"""
        if quality_score < 50:
            return 3  # High priority
        elif quality_score < 70:
            return 2  # Medium priority
        else:
            return 1  # Low priority
    
    def _calculate_inter_annotator_agreement(
        self, 
        annotation1: Dict[str, Any],
        annotation2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate agreement metrics between two annotations"""
        agreements = {}
        disagreements = []
        
        # Primary decision agreement
        decision1 = annotation1.get('has_nodule')
        decision2 = annotation2.get('has_nodule')
        
        if decision1 is not None and decision2 is not None:
            agreements['primary_decision'] = decision1 == decision2
            if not agreements['primary_decision']:
                disagreements.append('primary_decision')
        
        # Confidence agreement
        conf1 = annotation1.get('confidence', 0.5)
        conf2 = annotation2.get('confidence', 0.5)
        confidence_diff = abs(conf1 - conf2)
        agreements['confidence'] = confidence_diff < 0.2
        
        if not agreements['confidence']:
            disagreements.append(f'confidence_difference_{confidence_diff:.2f}')
        
        # Overall agreement score
        agreement_scores = [score for score in agreements.values() if isinstance(score, bool)]
        overall_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.5
        
        return {
            'overall_agreement': overall_agreement,
            'specific_agreements': agreements,
            'disagreements': disagreements,
            'confidence_difference': confidence_diff
        }
    
    def _determine_consensus(
        self,
        annotation1: Dict[str, Any],
        annotation2: Dict[str, Any],
        agreement_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine consensus between two annotations"""
        overall_agreement = agreement_metrics['overall_agreement']
        
        if overall_agreement >= self.quality_thresholds['agreement_threshold']:
            # High agreement - use higher confidence annotation
            conf1 = annotation1.get('confidence', 0.5)
            conf2 = annotation2.get('confidence', 0.5)
            
            final_annotation = annotation1 if conf1 >= conf2 else annotation2
            
            return {
                'consensus_reached': True,
                'final_annotation': final_annotation,
                'notes': f'Consensus reached with {overall_agreement:.2f} agreement',
                'resolution_method': 'higher_confidence'
            }
        else:
            # Low agreement - requires expert review
            return {
                'consensus_reached': False,
                'final_annotation': None,
                'notes': f'Consensus not reached - agreement only {overall_agreement:.2f}',
                'resolution_method': 'expert_review_required'
            }
    
    def _get_annotator_history(self, annotator_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get historical annotations for consistency analysis"""
        try:
            # This would typically query your annotation database
            # For now, return empty list - implement based on your annotation storage
            return []
        except Exception as e:
            logging.error(f"❌ Failed to get annotator history: {str(e)}")
            return []
    
    def _log_quality_evaluation(self, evaluation_result: Dict[str, Any]):
        """Log quality evaluation to database"""
        try:
            with sqlite3.connect(self.quality_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO quality_reviews 
                    (case_id, quality_level, quality_score, issues_found, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    evaluation_result['case_id'],
                    evaluation_result['quality_level'],
                    evaluation_result['quality_score'],
                    json.dumps(evaluation_result['quality_issues']),
                    evaluation_result['evaluation_timestamp']
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"❌ Failed to log quality evaluation: {str(e)}")
    
    def _log_peer_review(
        self, 
        review_result: Dict[str, Any],
        original_annotation: Dict[str, Any],
        review_annotation: Dict[str, Any]
    ):
        """Log peer review results"""
        try:
            with sqlite3.connect(self.quality_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO inter_annotator_agreement 
                    (case_id, annotation1_data, annotation2_data, agreement_score,
                     disagreement_areas, consensus_reached, final_annotation, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    review_result['case_id'],
                    json.dumps(original_annotation),
                    json.dumps(review_annotation),
                    review_result['agreement_score'],
                    json.dumps(review_result['disagreement_areas']),
                    review_result['consensus_reached'],
                    json.dumps(review_result.get('final_annotation')),
                    review_result['review_timestamp']
                ))
                conn.commit()
        except Exception as e:
            logging.error(f"❌ Failed to log peer review: {str(e)}")
    
    def generate_quality_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate quality report for specified period"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with sqlite3.connect(self.quality_db) as conn:
                # Get quality reviews in period
                df = pd.read_sql_query('''
                    SELECT quality_level, quality_score, issues_found, created_at
                    FROM quality_reviews 
                    WHERE created_at >= ? AND created_at <= ?
                ''', conn, params=(start_date.isoformat(), end_date.isoformat()))
                
                if len(df) == 0:
                    return {"error": "No quality data available for the specified period"}
                
                # Quality distribution
                quality_distribution = df['quality_level'].value_counts().to_dict()
                
                # Average quality score
                avg_quality = df['quality_score'].mean()
                
                # Common issues
                all_issues = []
                for issues_json in df['issues_found']:
                    try:
                        issues = json.loads(issues_json)
                        all_issues.extend(issues)
                    except:
                        continue
                
                issue_counts = pd.Series(all_issues).value_counts().head(5).to_dict()
                
                return {
                    "period": f"{start_date.date()} to {end_date.date()}",
                    "total_reviews": len(df),
                    "quality_distribution": quality_distribution,
                    "average_quality_score": avg_quality,
                    "common_issues": issue_counts,
                    "quality_trend": "stable",  # Would calculate actual trend
                    "recommendations": self._generate_recommendations(df, issue_counts)
                }
                
        except Exception as e:
            logging.error(f"❌ Failed to generate quality report: {str(e)}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, df: pd.DataFrame, common_issues: Dict[str, int]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        # Low quality score recommendations
        if df['quality_score'].mean() < 75:
            recommendations.append("Overall annotation quality is below target - consider additional training")
        
        # Time-based recommendations
        if "too quickly" in ' '.join(common_issues.keys()):
            recommendations.append("Multiple annotations completed too quickly - emphasize thorough review")
        
        # Consistency recommendations
        if "inconsistent" in ' '.join(common_issues.keys()):
            recommendations.append("Consistency issues detected - provide clearer annotation guidelines")
        
        if len(recommendations) == 0:
            recommendations.append("Quality metrics are good - maintain current standards")
        
        return recommendations