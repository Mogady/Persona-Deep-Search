from typing import Dict, List
from src.tools.models import ModelFactory
from src.utils.logger import get_logger
from src.utils.config import Config
from src.database.repository import ResearchRepository
from src.prompts.templates.reporter_prompt import REPORT_GENERATION_PROMPT
import json

class ReportGeneratorNode:
    """
    Generates the final comprehensive report using Claude Sonnet.
    """
    def __init__(self, config: Config, repository: ResearchRepository):
        """
        Initialize the report generator with Claude Sonnet.

        Args:
            config: Configuration object with all settings
            repository: Database repository for persistence
        """
        self.config = config
        self.repository = repository
        self.client = ModelFactory.get_optimal_model_for_task("report_generation") # Claude Sonnet
        self.logger = get_logger(__name__)

        # Load config values
        self.temperature = config.performance.report_generation_temperature

        self.logger.info(f"Initialized ReportGeneratorNode with Claude Sonnet (temp: {self.temperature})")

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

            # Save final report to database
            session_id = state.get("session_id")
            if session_id and report:
                try:
                    connection_graph = state.get("connection_graph")
                    self.repository.save_final_report(
                        session_id=session_id,
                        report=report,
                        connection_graph=connection_graph
                    )
                    self.logger.info("Saved final report to database")

                except Exception as e:
                    self.logger.error(f"Failed to save final report to database: {e}", exc_info=True)
                    # Don't fail the workflow if DB save fails

        except Exception as e:
            self.logger.error(f"An error occurred during report generation: {e}")
            state['final_report'] = "Error: Could not generate report."

        return state
