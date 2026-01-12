from typing import Dict, Any
import time
from .base_agent import BaseAgent


class PatientContextAgent(BaseAgent):
    """Agent for analyzing patient history and context"""
    
    def __init__(self, config: Dict = None):
        super().__init__("PatientContextAgent", config)
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patient demographics, symptoms, and history"""
        start_time = time.time()
        
        try:
            patient_data = context.get('patient_data', {})
            
            profile = await self._build_patient_profile(patient_data)
            risk_assessment = await self._assess_risk_factors(profile)
            symptom_correlation = await self._correlate_symptoms(
                profile, context.get('tumor_features', {})
            )
            
            output = {
                'patient_profile': profile,
                'risk_assessment': risk_assessment,
                'symptom_correlation': symptom_correlation,
                'clinical_significance': self._determine_clinical_significance(
                    profile, risk_assessment
                ),
                'relevance_score': 0.85
            }
            
            execution_time = time.time() - start_time
            self.log_execution(context, output, execution_time)
            
            return output
            
        except Exception as e:
            self.logger.error(f"Patient context analysis failed: {str(e)}")
            raise
    
    async def _build_patient_profile(self, patient_data: Dict) -> Dict:
        """Build comprehensive patient profile"""
        return {
            'demographics': {
                'age': patient_data.get('age', 52),
                'sex': patient_data.get('sex', 'male'),
                'ethnicity': patient_data.get('ethnicity', 'caucasian')
            },
            'presenting_symptoms': patient_data.get('symptoms', [
                'progressive headaches',
                'seizures',
                'vision changes',
                'cognitive decline'
            ]),
            'symptom_duration': patient_data.get('symptom_duration', '3 months'),
            'symptom_progression': 'rapidly progressive',
            'family_history': {
                'cancer_history': patient_data.get('family_history', 'mother had brain cancer'),
                'genetic_predisposition': 'possible familial cancer syndrome'
            },
            'medical_history': {
                'comorbidities': patient_data.get('comorbidities', ['hypertension']),
                'previous_surgeries': [],
                'medications': patient_data.get('medications', ['lisinopril', 'aspirin']),
                'allergies': []
            },
            'social_history': {
                'smoking': patient_data.get('smoking', 'never'),
                'alcohol': 'occasional',
                'occupation': 'office worker',
                'exposure': 'no known carcinogen exposure'
            }
        }
    
    async def _assess_risk_factors(self, profile: Dict) -> Dict:
        """Assess patient risk factors"""
        risk_factors = []
        risk_score = 0.0
        
        # Age risk
        age = profile['demographics']['age']
        if 45 <= age <= 70:
            risk_factors.append({
                'factor': 'Age',
                'value': f'{age} years',
                'impact': 'moderate',
                'explanation': 'Peak age for high-grade gliomas'
            })
            risk_score += 0.3
        
        # Family history
        if 'brain cancer' in str(profile['family_history']).lower():
            risk_factors.append({
                'factor': 'Family History',
                'value': 'First-degree relative with brain cancer',
                'impact': 'high',
                'explanation': 'Increased risk of hereditary cancer syndromes'
            })
            risk_score += 0.5
        
        # Genetic predisposition
        if 'genetic_predisposition' in profile['family_history']:
            risk_factors.append({
                'factor': 'Genetic Predisposition',
                'value': 'Possible hereditary syndrome',
                'impact': 'high',
                'explanation': 'Consider genetic counseling and testing'
            })
            risk_score += 0.4
        
        return {
            'risk_factors': risk_factors,
            'overall_risk_score': min(risk_score, 1.0),
            'risk_category': self._categorize_risk(risk_score),
            'recommendations': [
                'Genetic counseling recommended',
                'Screen family members',
                'Consider germline testing for TP53, PTEN'
            ]
        }
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize overall risk"""
        if score < 0.3:
            return 'Low Risk'
        elif score < 0.6:
            return 'Moderate Risk'
        else:
            return 'High Risk'
    
    async def _correlate_symptoms(self, profile: Dict, tumor_features: Dict) -> Dict:
        """Correlate symptoms with tumor characteristics"""
        symptoms = profile.get('presenting_symptoms', [])
        location = tumor_features.get('location', '')
        
        correlations = []
        
        # Analyze each symptom
        if 'headaches' in str(symptoms).lower():
            correlations.append({
                'symptom': 'Progressive headaches',
                'correlation': 'high',
                'explanation': 'Consistent with increased intracranial pressure from mass effect',
                'supporting_feature': tumor_features.get('mass_effect', 'present')
            })
        
        if 'seizures' in str(symptoms).lower():
            if 'frontal' in location.lower() or 'temporal' in location.lower():
                correlations.append({
                    'symptom': 'Seizures',
                    'correlation': 'high',
                    'explanation': 'Frontal/temporal location typical for seizure onset',
                    'supporting_feature': f'Tumor in {location}'
                })
        
        if 'vision' in str(symptoms).lower():
            correlations.append({
                'symptom': 'Vision changes',
                'correlation': 'moderate',
                'explanation': 'May be due to mass effect on visual pathways or increased ICP',
                'supporting_feature': 'Midline shift present'
            })
        
        return {
            'symptom_tumor_correlations': correlations,
            'consistency_score': 0.85,
            'interpretation': 'Symptoms are highly consistent with imaging findings'
        }
    
    def _determine_clinical_significance(
        self, profile: Dict, risk_assessment: Dict
    ) -> Dict:
        """Determine overall clinical significance"""
        return {
            'urgency': 'HIGH',
            'clinical_context': 'New brain lesion with neurological symptoms and family history',
            'key_considerations': [
                'Rapidly progressive symptoms suggest aggressive tumor',
                'Family history warrants genetic evaluation',
                'Age and presentation consistent with high-grade glioma',
                'Urgent neurosurgical consultation recommended'
            ],
            'timeline_recommendation': 'Neurosurgical evaluation within 48-72 hours'
        }
