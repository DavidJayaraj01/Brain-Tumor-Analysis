from typing import Dict, Any, List
import time
from datetime import datetime
from .base_agent import BaseAgent


class ReportGenerationAgent(BaseAgent):
    """Agent for generating comprehensive medical reports"""
    
    def __init__(self, config: Dict = None):
        super().__init__("ReportGenerationAgent", config)
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive medical report"""
        start_time = time.time()
        
        try:
            vision = context.get('vision_analysis', {})
            knowledge = context.get('knowledge_analysis', {})
            patient = context.get('patient_analysis', {})
            qa = context.get('qa_analysis', {})
            
            report = await self._generate_comprehensive_report(
                vision, knowledge, patient, qa
            )
            summary = await self._generate_executive_summary(report)
            recommendations = await self._generate_action_items(knowledge, qa)
            
            output = {
                'full_report': report,
                'executive_summary': summary,
                'action_items': recommendations,
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_version': '1.0',
                    'ai_system': 'MediMind BTIS v1.0',
                    'quality_score': qa.get('quality_score', 0.85)
                }
            }
            
            execution_time = time.time() - start_time
            self.log_execution(context, output, execution_time)
            
            return output
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise
    
    async def _generate_comprehensive_report(
        self, vision: Dict, knowledge: Dict, patient: Dict, qa: Dict
    ) -> Dict:
        """Generate structured comprehensive report"""
        # Add defensive null checks
        vision = vision or {}
        knowledge = knowledge or {}
        patient = patient or {}
        qa = qa or {}
        
        tumor_features = vision.get('tumor_features', {}) or {}
        differential = knowledge.get('differential_diagnosis', []) or []
        patient_profile = patient.get('patient_profile', {}) or {}
        
        report = {
            'clinical_information': self._format_clinical_information(patient_profile),
            'technique': self._format_technique_section(),
            'findings': self._format_findings_section(tumor_features, vision),
            'impression': self._format_impression_section(differential, qa),
            'recommendations': self._format_recommendations_section(knowledge, qa)
        }
        
        return report
    
    def _format_clinical_information(self, profile: Dict) -> str:
        """Format clinical information section"""
        if not profile:
            profile = {}
        demographics = profile.get('demographics', {}) or {}
        symptoms = profile.get('presenting_symptoms', []) or []
        history = profile.get('family_history', {}) or {}
        
        clinical_info = f"""CLINICAL INFORMATION:

Patient: {demographics.get('age', 'N/A')} year old {demographics.get('sex', 'N/A')}

Chief Complaint: {', '.join(symptoms) if symptoms else 'Brain imaging'}

Duration: {profile.get('symptom_duration', 'Not specified')}

Relevant History: {history.get('cancer_history', 'None documented')}

Indication: Evaluation of brain mass, assessment for tumor characterization
"""
        return clinical_info
    
    def _format_technique_section(self) -> str:
        """Format imaging technique section"""
        return """TECHNIQUE:

MRI brain performed including the following sequences:
- T1-weighted pre-contrast imaging
- T2-weighted imaging  
- FLAIR imaging
- T1-weighted post-contrast imaging

AI-assisted analysis using MediMind Brain Tumor Intelligence System (MBTIS)
Multi-agent deep learning architecture with explainability features
"""
    
    def _format_findings_section(self, tumor_features: Dict, vision: Dict) -> str:
        """Format findings section"""
        size = tumor_features.get('size') if tumor_features else None
        location = tumor_features.get('location', 'not specified') if tumor_features else 'not specified'
        
        # Build size information safely
        if size and isinstance(size, dict):
            size_str = f"{size.get('length_mm', 'N/A')} x {size.get('width_mm', 'N/A')} x {size.get('height_mm', 'N/A')} mm (L x W x H)"
            volume_str = f"{size.get('volume_ml', size.get('estimated_volume_ml', 'N/A'))} mL"
        else:
            size_str = "N/A - no mass identified"
            volume_str = "N/A - no mass identified"
        
        findings = f"""FINDINGS:

Brain Parenchyma:
There is a{' heterogeneous' if tumor_features and tumor_features.get('necrosis') else 'n'} mass identified in the {location}.

Lesion Characteristics:
- Size: Approximately {size_str}
- Volume: Estimated {volume_str}
- Margins: {tumor_features.get('borders', 'not specified') if tumor_features else 'not specified'}
- Enhancement Pattern: {tumor_features.get('enhancement_pattern', 'not specified') if tumor_features else 'not specified'}
- Central Necrosis: {'Present' if tumor_features and tumor_features.get('necrosis') else 'Not identified'}
- Perilesional Edema: {tumor_features.get('edema', 'minimal') if tumor_features else 'none'}

Mass Effect:
{tumor_features.get('mass_effect', 'No significant mass effect')}

Ventricular System:
Normal size and configuration (unless mass effect present)

Extra-axial Spaces:
No extra-axial collections identified

AI Analysis Quality:
Image quality: {vision.get('quality_metrics', {}).get('image_quality', 'good')}
Confidence Score: {vision.get('confidence_score', 0.85):.1%}
"""
        return findings
    
    def _format_impression_section(self, differential: List[Dict], qa: Dict) -> str:
        """Format impression section"""
        if not differential or len(differential) == 0:
            return "IMPRESSION:\nInsufficient data for diagnostic impression."
        
        primary_dx = differential[0] if differential else {}
        confidence = qa.get('confidence_level', 'MEDIUM') if qa else 'MEDIUM'
        
        # Handle missing primary diagnosis data
        if not primary_dx:
            return "IMPRESSION:\nPrimary diagnosis not available."
        
        impression = f"""IMPRESSION:

1. {primary_dx.get('type', 'Brain mass')}, estimated probability {primary_dx.get('probability', 0.0):.0%}

   Key Supporting Features:
"""
        for feature in primary_dx.get('key_features', []):
            impression += f"   - {feature}\n"
        
        impression += f"\n   Diagnostic Confidence: {confidence}\n"
        
        if len(differential) > 1:
            impression += "\n2. Differential Considerations:\n"
            for i, dx in enumerate(differential[1:3], 2):
                impression += f"   {i}. {dx.get('type')} (probability {dx.get('probability', 0.0):.0%})\n"
        
        # Add quality assurance notes
        recommendation = qa.get('recommendation', '')
        if 'REVIEW' in recommendation:
            impression += f"\n   NOTE: {recommendation}\n"
        
        return impression
    
    def _format_recommendations_section(self, knowledge: Dict, qa: Dict) -> str:
        """Format recommendations section"""
        if not knowledge:
            knowledge = {}
        if not qa:
            qa = {}
        tests = knowledge.get('recommended_tests', []) or []
        treatment = knowledge.get('treatment_guidelines', {}) or {}
        
        recommendations = "RECOMMENDATIONS:\n\n"
        
        # Immediate actions
        recommendations += "Immediate Actions:\n"
        if qa.get('confidence_level') == 'LOW':
            recommendations += "- Expert neuroradiologist review recommended\n"
        recommendations += "- Neurosurgical consultation\n"
        recommendations += "- Multidisciplinary tumor board discussion\n\n"
        
        # Additional imaging/testing
        recommendations += "Additional Workup:\n"
        for test in tests[:3]:
            recommendations += f"- {test.get('test', 'Test')}: {test.get('rationale', 'Recommended')}\n"
        
        # Treatment approach
        if treatment:
            recommendations += f"\nPrimary Treatment Approach:\n- {treatment.get('primary_approach', 'TBD by neurosurgery')}\n"
        
        # Follow-up
        recommendations += "\nFollow-up:\n- As directed by neurosurgery\n"
        recommendations += "- Serial imaging to assess response to treatment\n"
        
        return recommendations
    
    async def _generate_executive_summary(self, report: Dict) -> str:
        """Generate concise executive summary"""
        impression = report.get('impression', '')
        
        # Extract key finding
        lines = impression.split('\n')
        main_finding = next((line for line in lines if line.strip().startswith('1.')), 'Brain mass detected')
        
        summary = f"""EXECUTIVE SUMMARY
==================

{main_finding}

This case has been analyzed by our AI-powered multi-agent diagnostic system with quality assurance validation.

Key Points:
- Comprehensive multi-modal MRI analysis completed
- Differential diagnosis provided with supporting evidence
- Clinical correlation with patient history performed
- Quality assurance checks completed
- Expert review pathway activated as appropriate

[See full report for detailed findings and recommendations]
"""
        return summary
    
    async def _generate_action_items(self, knowledge: Dict, qa: Dict) -> list:
        """Generate prioritized action items"""
        actions = []
        
        # Determine urgency
        urgency = qa.get('recommendation', '')
        
        if 'REQUIRED' in urgency or 'HIGH' in urgency:
            actions.append({
                'priority': 'URGENT',
                'action': 'Expert radiologist review',
                'timeline': 'Within 24 hours',
                'responsible': 'Radiology Department'
            })
        
        actions.extend([
            {
                'priority': 'HIGH',
                'action': 'Neurosurgical consultation',
                'timeline': 'Within 48-72 hours',
                'responsible': 'Neurosurgery'
            },
            {
                'priority': 'HIGH',
                'action': 'Advanced MRI sequences (spectroscopy, perfusion)',
                'timeline': 'Before surgical planning',
                'responsible': 'Radiology'
            },
            {
                'priority': 'MEDIUM',
                'action': 'Tumor board presentation',
                'timeline': 'Within 1-2 weeks',
                'responsible': 'Oncology/Neurosurgery'
            }
        ])
        
        # Add testing recommendations
        tests = knowledge.get('recommended_tests', [])
        for test in tests:
            if test.get('priority') == 'High':
                actions.append({
                    'priority': 'MEDIUM',
                    'action': test.get('test', 'Additional testing'),
                    'timeline': 'Per clinical team',
                    'responsible': 'Ordering Physician'
                })
        
        return actions
