"""
Concise prompts for the Report Generator Node
"""

REPORT_GENERATION_PROMPT = """
Generate markdown report from research data.

Data:
{state}

Structure:
1. Executive Summary (3-4 sentences key findings)
2. Subject Overview (bio/professional)
3. Key Facts (by category, 🟢🟡🔴 confidence)
4. Risk Assessment (severity + evidence)
5. Network Analysis (connections)
6. Timeline (chronological events)
7. Source Summary (domains, diversity)
8. Confidence Assessment (reliability)
9. Recommendations (follow-up)

Format: Markdown. Bold names. Tables for data. Professional tone. Markdown only.
"""
