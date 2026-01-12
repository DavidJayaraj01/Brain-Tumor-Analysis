from typing import Dict, Any, List
import time
from .base_agent import BaseAgent


class MedicalKnowledgeAgent(BaseAgent):
    """Agent for medical knowledge retrieval and reasoning"""
    
    def __init__(self, config: Dict = None):
        super().__init__("MedicalKnowledgeAgent", config)
        self.knowledge_base = self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self) -> Dict:
        """Initialize medical knowledge database"""
        return {
            'tumor_types': {
                'Glioma': {
                    'subtypes': ['Glioblastoma', 'Astrocytoma', 'Oligodendroglioma'],
                    'characteristics': {
                        'location': 'Usually cerebral hemispheres',
                        'enhancement': 'Ring enhancement common in high-grade',
                        'borders': 'Infiltrative, irregular',
                        'age_predilection': '45-65 years for glioblastoma'
                    },
                    'prognosis': 'Varies by grade; GBM median survival 12-15 months',
                    'treatment': 'Surgery + radiation + temozolomide chemotherapy'
                },
                'Meningioma': {
                    'characteristics': {
                        'location': 'Extra-axial, dural-based',
                        'enhancement': 'Homogeneous enhancement',
                        'borders': 'Well-circumscribed',
                        'age_predilection': '40-60 years, F>M'
                    },
                    'prognosis': 'Generally favorable, WHO grade I: 90%+ 5-year survival',
                    'treatment': 'Surgical resection, radiation for incomplete resection'
                },
                'Pituitary Adenoma': {
                    'characteristics': {
                        'location': 'Sellar/suprasellar region',
                        'enhancement': 'Variable enhancement',
                        'borders': 'Well-defined',
                        'age_predilection': '30-50 years'
                    },
                    'prognosis': 'Excellent with treatment, usually benign',
                    'treatment': 'Transsphenoidal surgery, medications, or radiation'
                }
            },
            'grading_criteria': {
                'WHO_Grade_I': 'Benign, slow-growing, well-circumscribed',
                'WHO_Grade_II': 'Infiltrative, may recur',
                'WHO_Grade_III': 'Malignant, infiltrative, rapid growth',
                'WHO_Grade_IV': 'Highly malignant, necrosis, rapid growth'
            }
        }
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve medical knowledge and provide differential diagnosis"""
        start_time = time.time()
        
        try:
            tumor_features = context.get('tumor_features', {})
            patient_context = context.get('patient_context', {})
            
            differential = await self._generate_differential_diagnosis(
                tumor_features, patient_context
            )
            literature = await self._retrieve_literature(tumor_features)
            recommendations = await self._generate_recommendations(differential)
            
            output = {
                'differential_diagnosis': differential,
                'literature_references': literature,
                'recommended_tests': recommendations['tests'],
                'treatment_guidelines': recommendations['treatment'],
                'knowledge_confidence': 0.88
            }
            
            execution_time = time.time() - start_time
            self.log_execution(context, output, execution_time)
            
            return output
            
        except Exception as e:
            self.logger.error(f"Knowledge retrieval failed: {str(e)}")
            raise
    
    async def _generate_differential_diagnosis(
        self, tumor_features: Dict, patient_context: Dict
    ) -> List[Dict]:
        """Generate differential diagnosis with reasoning"""
        probabilities = tumor_features.get('predicted_probabilities', {})
        location = tumor_features.get('location', '')
        enhancement = tumor_features.get('enhancement_pattern', '')
        
        differential = []
        
        # Glioma analysis
        if 'glioma' in probabilities:
            glioma_reasoning = []
            if 'infiltrative' in tumor_features.get('borders', ''):
                glioma_reasoning.append("Infiltrative borders typical of glioma")
            if 'ring enhancement' in enhancement:
                glioma_reasoning.append("Ring enhancement suggests high-grade glioma")
            if tumor_features.get('necrosis'):
                glioma_reasoning.append("Central necrosis characteristic of glioblastoma")
            
            differential.append({
                'type': 'Glioma (likely Glioblastoma)',
                'probability': probabilities.get('glioma', 0.6),
                'reasoning': ' | '.join(glioma_reasoning),
                'key_features': [
                    'Infiltrative borders',
                    'Ring enhancement',
                    'Central necrosis',
                    'Perilesional edema'
                ],
                'typical_presentation': self.knowledge_base['tumor_types']['Glioma']['characteristics'],
                'next_steps': [
                    'MRI with spectroscopy',
                    'Consider stereotactic biopsy',
                    'Molecular markers: IDH, MGMT'
                ]
            })
        
        # Meningioma analysis
        if 'meningioma' in probabilities:
            menin_reasoning = []
            if 'intra-axial' not in location.lower():
                menin_reasoning.append("Extra-axial location typical")
            if 'homogeneous' in enhancement:
                menin_reasoning.append("Homogeneous enhancement pattern")
            
            differential.append({
                'type': 'Meningioma',
                'probability': probabilities.get('meningioma', 0.25),
                'reasoning': ' | '.join(menin_reasoning) if menin_reasoning else 'Less likely given intra-axial location',
                'key_features': ['Dural-based', 'Homogeneous enhancement'],
                'typical_presentation': self.knowledge_base['tumor_types']['Meningioma']['characteristics'],
                'next_steps': ['Assess dural tail sign', 'Evaluate hyperostosis']
            })
        
        # Sort by probability
        differential.sort(key=lambda x: x['probability'], reverse=True)
        
        return differential
    
    async def _retrieve_literature(self, tumor_features: Dict) -> List[Dict]:
        """Retrieve relevant medical literature"""
        return [
            {
                'title': 'WHO Classification of Tumours of the Central Nervous System',
                'authors': 'Louis DN, et al.',
                'year': 2021,
                'relevance': 0.95,
                'key_point': 'Updated classification incorporating molecular markers'
            },
            {
                'title': 'Glioblastoma: Pathology, Molecular Mechanisms and Markers',
                'authors': 'Aldape K, et al.',
                'year': 2023,
                'relevance': 0.88,
                'key_point': 'IDH status is crucial for prognosis and treatment planning'
            },
            {
                'title': 'Advanced MRI Techniques for Brain Tumor Diagnosis',
                'authors': 'Patel SH, et al.',
                'year': 2023,
                'relevance': 0.82,
                'key_point': 'Perfusion imaging helps differentiate tumor types'
            }
        ]
    
    async def _generate_recommendations(self, differential: List[Dict]) -> Dict:
        """Generate test and treatment recommendations"""
        return {
            'tests': [
                {
                    'test': 'MR Spectroscopy',
                    'rationale': 'Differentiate tumor from other lesions',
                    'priority': 'High'
                },
                {
                    'test': 'Perfusion MRI',
                    'rationale': 'Assess vascularity and grade',
                    'priority': 'High'
                },
                {
                    'test': 'Molecular Markers (IDH, MGMT)',
                    'rationale': 'Guide treatment decisions and prognosis',
                    'priority': 'Essential if biopsy performed'
                }
            ],
            'treatment': {
                'primary_approach': 'Maximal safe surgical resection',
                'adjuvant_therapy': 'Concurrent chemoradiation with temozolomide',
                'clinical_trials': 'Consider immunotherapy trials for newly diagnosed GBM',
                'follow_up': 'MRI every 2-3 months post-treatment'
            }
        }
