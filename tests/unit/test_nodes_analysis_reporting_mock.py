import unittest
from unittest.mock import patch, MagicMock
import json

from src.agents.nodes.risk_analyzer import RiskAnalyzerNode
from src.agents.nodes.connection_mapper import ConnectionMapperNode
from src.agents.nodes.reporter import ReportGeneratorNode

class TestAnalysisReportingNodes(unittest.TestCase):

    @patch('src.tools.models.ModelFactory.get_optimal_model_for_task')
    def test_risk_analyzer_node(self, mock_get_model):
        mock_client = MagicMock()
        mock_get_model.return_value = mock_client

        risk_analyzer = RiskAnalyzerNode()

        # Mock risks with all required fields
        mock_risks = [
            {
                "severity": "High",
                "category": "Financial",
                "description": "Test risk description",
                "evidence": ["A fact about finances"],
                "confidence": 0.75
            }
        ]
        mock_client.generate.return_value = json.dumps(mock_risks)

        state = {
            'collected_facts': [
                {
                    'content': 'A fact about finances',
                    'confidence': 0.8,
                    'source_domain': 'example.com'
                }
            ]
        }

        result_state = risk_analyzer.execute(state)

        self.assertIn('risk_flags', result_state)
        self.assertEqual(len(result_state['risk_flags']), 1)
        self.assertEqual(result_state['risk_flags'][0]['severity'], 'High')
        self.assertEqual(result_state['risk_flags'][0]['category'], 'Financial')
        self.assertIn('confidence', result_state['risk_flags'][0])

    @patch('src.tools.models.ModelFactory.get_optimal_model_for_task')
    def test_connection_mapper_node(self, mock_get_model):
        mock_client = MagicMock()
        mock_get_model.return_value = mock_client

        connection_mapper = ConnectionMapperNode()

        # Mock connections with all required fields
        mock_connections = [
            {
                "entity_a": "Person A",
                "entity_b": "Company B",
                "relationship_type": "Employment",
                "evidence": ["Person A worked at Company B"],
                "confidence": 0.85,
                "time_period": "2020-2023"
            }
        ]
        mock_client.generate.return_value = json.dumps(mock_connections)

        state = {
            'collected_facts': [
                {
                    'content': 'Person A worked at Company B',
                    'confidence': 0.9,
                    'source_domain': 'linkedin.com',
                    'entities': {
                        'people': ['Person A'],
                        'companies': ['Company B']
                    }
                }
            ]
        }

        result_state = connection_mapper.execute(state)

        self.assertIn('connections', result_state)
        self.assertEqual(len(result_state['connections']), 1)
        self.assertEqual(result_state['connections'][0]['entity_a'], 'Person A')
        self.assertEqual(result_state['connections'][0]['relationship_type'], 'Employment')
        self.assertIn('confidence', result_state['connections'][0])

    @patch('src.tools.models.ModelFactory.get_optimal_model_for_task')
    def test_report_generator_node(self, mock_get_model):
        mock_client = MagicMock()
        mock_get_model.return_value = mock_client

        report_generator = ReportGeneratorNode()

        mock_report = "# Final Report\n\nThis is a test report."
        mock_client.generate.return_value = mock_report

        state = {
            'collected_facts': [],
            'risk_flags': [],
            'connections': []
        }

        result_state = report_generator.execute(state)

        self.assertIn('final_report', result_state)
        self.assertEqual(result_state['final_report'], mock_report)

if __name__ == '__main__':
    unittest.main()
