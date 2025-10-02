from typing import Dict, List
from src.tools.models import ModelFactory
from src.utils.logger import get_logger
from src.prompts.templates.reporter_prompt import REPORT_GENERATION_PROMPT
import json

class ReportGeneratorNode:
    """
    Generates the final comprehensive report using Claude Sonnet.
    """
    def __init__(self):
        self.client = ModelFactory.get_optimal_model_for_task("report_generation") # Claude Sonnet
        self.logger = get_logger(__name__)

    def execute(self, state: Dict) -> Dict:
        """
        Generates the final report and adds it to the state.
        
        Args:
            state: The final ResearchState with all data.
        
        Returns:
            Updated state with 'final_report' populated.
        """
        self.logger.info("Executing Report Generator Node")
        
        prompt = REPORT_GENERATION_PROMPT.format(state=json.dumps(state, indent=2, default=str))
        
        try:
            report = self.client.generate(prompt)
            state['final_report'] = report
            self.logger.info("Successfully generated final report.")
            
        except Exception as e:
            self.logger.error(f"An error occurred during report generation: {e}")
            state['final_report'] = "Error: Could not generate report."

        return state
