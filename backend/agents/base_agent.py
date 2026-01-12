from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all AI agents in the system"""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.execution_history = []
        self.logger = logging.getLogger(f"Agent.{name}")
        
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent's main task"""
        pass
    
    def log_execution(self, input_data: Dict, output_data: Dict, execution_time: float):
        """Log agent execution for monitoring"""
        self.execution_history.append({
            'timestamp': datetime.now().isoformat(),
            'input_size': len(str(input_data)),
            'output_size': len(str(output_data)),
            'execution_time': execution_time,
            'success': True
        })
        self.logger.info(f"{self.name} executed in {execution_time:.2f}s")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        if not self.execution_history:
            return {}
        
        execution_times = [h['execution_time'] for h in self.execution_history]
        return {
            'agent_name': self.name,
            'total_executions': len(self.execution_history),
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times)
        }
