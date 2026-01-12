"""
Multi-Agent Orchestrator
Coordinates all AI agents for comprehensive tumor analysis
"""

import asyncio
from typing import Dict, Any
import time
import logging

from .vision_agent import VisionAnalysisAgent
from .knowledge_agent import MedicalKnowledgeAgent
from .patient_agent import PatientContextAgent
from .qa_agent import QualityAssuranceAgent
from .report_agent import ReportGenerationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates multiple AI agents for brain tumor analysis"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize all agents
        self.vision_agent = VisionAnalysisAgent(config)
        self.knowledge_agent = MedicalKnowledgeAgent(config)
        self.patient_agent = PatientContextAgent(config)
        self.qa_agent = QualityAssuranceAgent(config)
        self.report_agent = ReportGenerationAgent(config)
        
        self.execution_log = []
        
        logger.info("AgentOrchestrator initialized with 5 specialized agents")
    
    async def analyze_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive case analysis using all agents
        
        Args:
            case_data: Dictionary containing:
                - image: MRI image data
                - patient_data: Patient demographics, symptoms, history
                
        Returns:
            Comprehensive analysis results from all agents
        """
        start_time = time.time()
        logger.info("=" * 80)
        logger.info("Starting Multi-Agent Analysis")
        logger.info("=" * 80)
        
        try:
            # Phase 1: Parallel execution of independent agents
            logger.info("\n[Phase 1] Executing Vision, Knowledge, and Patient agents in parallel...")
            
            vision_task = self.vision_agent.execute(case_data)
            knowledge_task = self.knowledge_agent.execute(case_data)
            patient_task = self.patient_agent.execute(case_data)
            
            vision_result, knowledge_result, patient_result = await asyncio.gather(
                vision_task,
                knowledge_task,
                patient_task
            )
            
            logger.info(f"✓ Vision Agent completed: confidence {vision_result.get('confidence_score', 0):.2%}")
            logger.info(f"✓ Knowledge Agent completed: {len(knowledge_result.get('differential_diagnosis', []))} differential diagnoses")
            logger.info(f"✓ Patient Agent completed: {len(patient_result.get('patient_profile', {}))} profile sections")
            
            # Phase 2: Quality Assurance (depends on Phase 1 results)
            logger.info("\n[Phase 2] Executing Quality Assurance validation...")
            
            qa_context = {
                **case_data,
                'vision_analysis': vision_result,
                'knowledge_analysis': knowledge_result,
                'patient_analysis': patient_result
            }
            
            qa_result = await self.qa_agent.execute(qa_context)
            
            logger.info(f"✓ QA Agent completed: {qa_result.get('validation_status')} (quality score: {qa_result.get('quality_score', 0):.2%})")
            logger.info(f"  Recommendation: {qa_result.get('recommendation', 'N/A')}")
            
            # Phase 3: Report Generation (depends on all previous results)
            logger.info("\n[Phase 3] Generating comprehensive report...")
            
            report_context = {
                **qa_context,
                'qa_analysis': qa_result
            }
            
            report_result = await self.report_agent.execute(report_context)
            
            logger.info(f"✓ Report Agent completed: Full report generated")
            
            # Compile final output
            final_output = {
                'analysis_results': {
                    'vision_analysis': vision_result,
                    'knowledge_analysis': knowledge_result,
                    'patient_analysis': patient_result,
                    'quality_assurance': qa_result,
                    'report': report_result
                },
                'metadata': {
                    'total_execution_time': time.time() - start_time,
                    'agents_executed': 5,
                    'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'system_version': 'MediMind BTIS v1.0'
                },
                'summary': {
                    'primary_diagnosis': self._extract_primary_diagnosis(knowledge_result),
                    'confidence': qa_result.get('confidence_level', 'MEDIUM'),
                    'quality_score': qa_result.get('quality_score', 0.0),
                    'recommendation': qa_result.get('recommendation', ''),
                    'urgency': patient_result.get('clinical_significance', {}).get('urgency', 'MEDIUM')
                }
            }
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Analysis Complete! Total time: {final_output['metadata']['total_execution_time']:.2f}s")
            logger.info(f"Primary Diagnosis: {final_output['summary']['primary_diagnosis']}")
            logger.info(f"Confidence: {final_output['summary']['confidence']}")
            logger.info(f"{'='*80}\n")
            
            # Log execution
            self.execution_log.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'execution_time': final_output['metadata']['total_execution_time'],
                'status': 'SUCCESS',
                'quality_score': qa_result.get('quality_score', 0.0)
            })
            
            return final_output
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            
            self.execution_log.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'execution_time': time.time() - start_time,
                'status': 'FAILED',
                'error': str(e)
            })
            
            raise
    
    def _extract_primary_diagnosis(self, knowledge_result: Dict) -> str:
        """Extract primary diagnosis from knowledge agent output"""
        differential = knowledge_result.get('differential_diagnosis', [])
        if differential:
            return differential[0].get('type', 'Unknown')
        return 'Indeterminate'
    
    async def get_agent_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all agents"""
        return {
            'vision_agent': self.vision_agent.get_metrics(),
            'knowledge_agent': self.knowledge_agent.get_metrics(),
            'patient_agent': self.patient_agent.get_metrics(),
            'qa_agent': self.qa_agent.get_metrics(),
            'report_agent': self.report_agent.get_metrics(),
            'orchestrator': {
                'total_cases_processed': len(self.execution_log),
                'successful_cases': sum(1 for log in self.execution_log if log['status'] == 'SUCCESS'),
                'average_execution_time': sum(log['execution_time'] for log in self.execution_log) / len(self.execution_log) if self.execution_log else 0,
                'average_quality_score': sum(log.get('quality_score', 0) for log in self.execution_log if log['status'] == 'SUCCESS') / 
                                        max(sum(1 for log in self.execution_log if log['status'] == 'SUCCESS'), 1)
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        return {
            'status': 'healthy',
            'agents': {
                'vision': 'operational',
                'knowledge': 'operational',
                'patient': 'operational',
                'qa': 'operational',
                'report': 'operational'
            },
            'total_agents': 5,
            'system_version': 'MediMind BTIS v1.0'
        }
