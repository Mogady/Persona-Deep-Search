import pytest

from src.agents.nodes.risk_analyzer import RiskAnalyzerNode
from src.agents.nodes.connection_mapper import ConnectionMapperNode
from src.agents.nodes.reporter import ReportGeneratorNode

@pytest.fixture
def sample_validated_facts():
    """Provides a list of validated facts for testing the analysis and reporting flow."""
    return [
        {
            "content": "Satya Nadella became CEO of Microsoft in February 2014.",
            "source_url": "https://www.microsoft.com/en-us/about/leadership/satya-nadella",
            "confidence": 0.95
        },
        {
            "content": "Satya Nadella was born in Hyderabad, India.",
            "source_url": "https://www.nytimes.com/tech/microsoft-ceo-nadella",
            "confidence": 0.9
        },
        {
            "content": "Microsoft acquired GitHub for $7.5 billion in 2018 under Nadella's leadership.",
            "source_url": "https://news.microsoft.com/2018/06/04/microsoft-to-acquire-github-for-7-5-billion/",
            "confidence": 0.98
        }
    ]

class TestAnalysisReportingIntegration:

    @pytest.mark.integration
    def test_full_analysis_and_reporting_flow(self, sample_validated_facts):
        """Tests the full flow from validated facts to a final report using real AI models."""
        # Initial state with validated facts
        state = {
            'target_name': 'Satya Nadella',
            'collected_facts': sample_validated_facts,
            'risk_flags': [],
            'connections': []
        }

        # Instantiate the nodes
        risk_analyzer = RiskAnalyzerNode()
        connection_mapper = ConnectionMapperNode()
        report_generator = ReportGeneratorNode()

        # Execute the analysis and reporting flow
        state = risk_analyzer.execute(state)
        state = connection_mapper.execute(state)
        state = report_generator.execute(state)

        # Verify the final state
        assert 'risk_flags' in state
        assert 'connections' in state
        assert 'final_report' in state

        # Check that the report is a non-empty string
        assert isinstance(state['final_report'], str)
        assert len(state['final_report']) > 0

        print(f"\nGenerated Report:\n{state['final_report']}")
