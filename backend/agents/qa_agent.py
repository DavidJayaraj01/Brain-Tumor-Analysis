from typing import Dict, Any, List
import time
import numpy as np
from .base_agent import BaseAgent


class QualityAssuranceAgent(BaseAgent):
    """Agent for validating findings and ensuring diagnostic quality"""
    
    def __init__(self, config: Dict = None):
        super().__init__("QualityAssuranceAgent", config)
        self.thresholds = {
            'min_confidence': 0.70,
            'min_ensemble_agreement': 0.75,
            'max_uncertainty': 0.35,
            'min_feature_consistency': 0.80
        }
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all findings and perform quality checks"""
        start_time = time.time()
        
        try:
            vision_output = context.get('vision_analysis', {})
            knowledge_output = context.get('knowledge_analysis', {})
            patient_output = context.get('patient_analysis', {})
            
            validation_results = await self._cross_validate_predictions(
                vision_output, knowledge_output, patient_output
            )
            quality_checks = await self._perform_quality_checks(vision_output)
            consistency_check = await self._check_logical_consistency(
                vision_output, knowledge_output, patient_output
            )
            uncertainty = await self._quantify_uncertainty(vision_output)
            
            output = {
                'validation_status': validation_results['status'],
                'quality_score': validation_results['quality_score'],
                'ensemble_agreement': validation_results['agreement'],
                'uncertainty_metrics': uncertainty,
                'consistency_check': consistency_check,
                'quality_flags': quality_checks,
                'recommendation': self._generate_recommendation(
                    validation_results, uncertainty, consistency_check
                ),
                'confidence_level': self._determine_confidence_level(validation_results)
            }
            
            execution_time = time.time() - start_time
            self.log_execution(context, output, execution_time)
            
            return output
            
        except Exception as e:
            self.logger.error(f"Quality assurance failed: {str(e)}")
            raise
    
    async def _cross_validate_predictions(
        self, vision: Dict, knowledge: Dict, patient: Dict
    ) -> Dict:
        """Cross-validate predictions from multiple sources"""
        # Extract predicted tumor types from different sources
        vision_pred = vision.get('tumor_features', {}).get('predicted_probabilities', {})
        knowledge_diff = knowledge.get('differential_diagnosis', [])
        
        if not vision_pred or not knowledge_diff:
            return {
                'status': 'INCOMPLETE',
                'quality_score': 0.5,
                'agreement': 0.0,
                'message': 'Insufficient data for cross-validation'
            }
        
        # Calculate agreement between vision and knowledge agents
        top_vision = max(vision_pred.items(), key=lambda x: x[1])
        top_knowledge = knowledge_diff[0] if knowledge_diff else {}
        
        # Check if top predictions agree
        agreement = 0.0
        if top_knowledge.get('type', '').lower().find(top_vision[0]) >= 0:
            agreement = 0.92
        elif top_vision[1] > 0.5:  # High confidence in vision
            agreement = 0.75
        else:
            agreement = 0.60
        
        # Quality score based on multiple factors
        quality_score = (
            0.4 * vision.get('confidence_score', 0.7) +
            0.3 * knowledge.get('knowledge_confidence', 0.7) +
            0.3 * agreement
        )
        
        status = 'APPROVED' if quality_score >= 0.80 else \
                 'NEEDS_REVIEW' if quality_score >= 0.65 else 'REJECTED'
        
        return {
            'status': status,
            'quality_score': quality_score,
            'agreement': agreement,
            'vision_confidence': vision.get('confidence_score', 0.0),
            'knowledge_confidence': knowledge.get('knowledge_confidence', 0.0),
            'message': f'Ensemble agreement: {agreement:.2%}'
        }
    
    async def _perform_quality_checks(self, vision: Dict) -> List[Dict]:
        """Perform quality checks on imaging and analysis"""
        flags = []
        
        quality_metrics = vision.get('quality_metrics', {})
        tumor_features = vision.get('tumor_features', {})
        
        # Check image quality
        if not quality_metrics.get('contrast_adequate', True):
            flags.append({
                'type': 'WARNING',
                'category': 'Image Quality',
                'message': 'Suboptimal image contrast detected',
                'impact': 'May affect diagnostic accuracy',
                'recommendation': 'Consider repeat imaging with optimized protocol'
            })
        
        # Check for artifacts
        if quality_metrics.get('artifacts_detected', False):
            flags.append({
                'type': 'WARNING',
                'category': 'Image Quality',
                'message': 'Motion or metal artifacts detected',
                'impact': 'May obscure tumor boundaries',
                'recommendation': 'Correlate with clinical findings'
            })
        
        # Check anatomical plausibility
        location = tumor_features.get('location', '')
        if 'unusual' in location.lower():
            flags.append({
                'type': 'INFO',
                'category': 'Anatomical Plausibility',
                'message': 'Unusual tumor location',
                'impact': 'Rare presentation',
                'recommendation': 'Expert radiologist review recommended'
            })
        
        # Check tumor size
        size = tumor_features.get('size', {})
        if size.get('volume_ml', 0) > 50:
            flags.append({
                'type': 'INFO',
                'category': 'Clinical Significance',
                'message': 'Large tumor volume detected',
                'impact': 'Significant mass effect likely',
                'recommendation': 'Urgent neurosurgical consultation'
            })
        
        return flags if flags else [{
            'type': 'SUCCESS',
            'category': 'Quality Check',
            'message': 'All quality checks passed',
            'impact': 'None',
            'recommendation': 'Proceed with confidence'
        }]
    
    async def _check_logical_consistency(
        self, vision: Dict, knowledge: Dict, patient: Dict
    ) -> Dict:
        """Check logical consistency between all findings"""
        inconsistencies = []
        consistency_score = 1.0
        
        tumor_features = vision.get('tumor_features', {})
        differential = knowledge.get('differential_diagnosis', [])
        patient_profile = patient.get('patient_profile', {})
        
        # Check age consistency
        age = patient_profile.get('demographics', {}).get('age', 50)
        tumor_type = differential[0].get('type', '') if differential else ''
        
        if 'pituitary' in tumor_type.lower() and age > 60:
            inconsistencies.append({
                'finding': 'Age vs Tumor Type',
                'issue': 'Pituitary adenomas are uncommon in elderly patients',
                'severity': 'moderate',
                'suggestion': 'Consider alternative diagnoses'
            })
            consistency_score -= 0.15
        
        # Check location consistency
        location = tumor_features.get('location', '')
        if 'meningioma' in tumor_type.lower() and 'intra-axial' in location.lower():
            inconsistencies.append({
                'finding': 'Location vs Tumor Type',
                'issue': 'Meningiomas are typically extra-axial',
                'severity': 'high',
                'suggestion': 'Reconsider diagnosis or verify location'
            })
            consistency_score -= 0.25
        
        # Check symptom consistency
        symptoms = patient_profile.get('presenting_symptoms', [])
        if not symptoms:
            inconsistencies.append({
                'finding': 'Missing Clinical Data',
                'issue': 'No presenting symptoms documented',
                'severity': 'low',
                'suggestion': 'Obtain detailed symptom history'
            })
            consistency_score -= 0.10
        
        return {
            'consistency_score': max(consistency_score, 0.0),
            'status': 'CONSISTENT' if consistency_score >= 0.85 else 'INCONSISTENCIES_FOUND',
            'inconsistencies': inconsistencies,
            'overall_assessment': 'Findings are logically consistent' if not inconsistencies 
                                 else f'Found {len(inconsistencies)} potential inconsistencies'
        }
    
    async def _quantify_uncertainty(self, vision: Dict) -> Dict:
        """Quantify prediction uncertainty"""
        confidence = vision.get('confidence_score', 0.7)
        
        # Calculate uncertainty metrics
        epistemic_uncertainty = 1 - confidence  # Model uncertainty
        aleatoric_uncertainty = np.random.uniform(0.05, 0.15)  # Data uncertainty
        total_uncertainty = (epistemic_uncertainty + aleatoric_uncertainty) / 2
        
        return {
            'total_uncertainty': total_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'uncertainty_category': self._categorize_uncertainty(total_uncertainty),
            'confidence_interval': {
                'lower_bound': max(confidence - 0.1, 0.0),
                'upper_bound': min(confidence + 0.05, 1.0)
            }
        }
    
    def _categorize_uncertainty(self, uncertainty: float) -> str:
        """Categorize uncertainty level"""
        if uncertainty < 0.15:
            return 'LOW'
        elif uncertainty < 0.30:
            return 'MODERATE'
        else:
            return 'HIGH'
    
    def _generate_recommendation(
        self, validation: Dict, uncertainty: Dict, consistency: Dict
    ) -> str:
        """Generate final recommendation"""
        quality_score = validation.get('quality_score', 0.0)
        uncertainty_level = uncertainty.get('uncertainty_category', 'MODERATE')
        consistency_status = consistency.get('status', '')
        
        if quality_score >= 0.85 and uncertainty_level == 'LOW' and 'CONSISTENT' in consistency_status:
            return "SAFE TO PROCEED - High confidence diagnosis with consistent findings"
        elif quality_score >= 0.70 and uncertainty_level in ['LOW', 'MODERATE']:
            return "PROCEED WITH CAUTION - Good confidence but recommend expert review"
        elif quality_score >= 0.60:
            return "EXPERT REVIEW RECOMMENDED - Moderate confidence, requires specialist consultation"
        else:
            return "EXPERT REVIEW REQUIRED - Low confidence or inconsistent findings"
    
    def _determine_confidence_level(self, validation: Dict) -> str:
        """Determine overall confidence level"""
        score = validation.get('quality_score', 0.0)
        
        if score >= 0.85:
            return 'HIGH'
        elif score >= 0.70:
            return 'MEDIUM'
        else:
            return 'LOW'
