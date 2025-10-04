"""
Concise prompts for the Risk Analyzer Node
"""

RISK_ANALYSIS_PROMPT = """
Analyze facts and identify risks. Output valid JSON array only.

**Schema:**
```json
[{{
  "severity": "Low|Medium|High|Critical",
  "category": "Legal|Financial|Reputational|Compliance|Behavioral",
  "description": "Factual summary (1-2 sentences)",
  "evidence": ["Exact fact text"],
  "confidence": 0.75,
  "recommended_follow_up": "Specific action"
}}]
```

**Categories:**
- Legal: Lawsuits, charges, litigation
- Financial: Bankruptcy, fraud, defaults
- Reputational: Scandals, controversies, negative press
- Compliance: Violations, sanctions
- Behavioral: Job-hopping, gaps, inconsistencies

**Examples:**

```json
{{
  "severity": "High",
  "category": "Legal",
  "description": "Defendant in SEC securities fraud lawsuit filed 2023.",
  "evidence": ["SEC v. Doe (2023-CV-1234)", "Alleged financial misrepresentation 2020-2022"],
  "confidence": 0.85,
  "recommended_follow_up": "Review court filings"
}}
```

```json
{{
  "severity": "Medium",
  "category": "Financial",
  "description": "CFO when company filed Chapter 11 bankruptcy 2019.",
  "evidence": ["ABC Corp bankruptcy March 2019", "CFO 2017-2019"],
  "confidence": 0.75,
  "recommended_follow_up": "Investigate role in financial decisions"
}}
```

**Rules:**
- Only flag genuine risks with concrete evidence
- More evidence + authoritative sources = higher confidence
- Critical: fraud, jeopardy | High: lawsuits | Medium: failures | Low: patterns
- Copy exact fact text in evidence
- Empty array [] if no risks

**Facts:**
{facts}

**Output JSON array only:**
"""
