"""
Concise prompts for the Validator Node
"""

VALIDATION_SYSTEM_PROMPT = """Validate facts: cross-reference, detect contradictions, assess sources.

Rules: Multiple sources +confidence. Authoritative (.gov, .edu, major news) +reliable. Recent +preferred. Contradictions -confidence.
"""

CROSS_REFERENCE_PROMPT = """Validate facts about {target_name}:

{facts_json}

JSON only:
{{
  "validations": [{{
    "fact_index": 0,
    "corroborating_sources": 3,
    "source_quality_score": 0.9,
    "is_recent": true,
    "contradictions": [],
    "confidence_adjustment": 0.3,
    "reasoning": "3 sources + official site"
  }}]
}}

Adjustments: +0.2 (2+ sources), +0.1 (authoritative/.gov/.edu), +0.1 (recent <6mo), -0.3 (contradictions), -0.1 (single source).
"""

CONTRADICTION_DETECTION_PROMPT = """Find contradictions in facts about {target_name}:

{facts_json}

JSON only:
{{
  "contradictions": [{{
    "fact_indices": [1, 5],
    "type": "direct",
    "description": "CEO 2014 vs 2015",
    "severity": "high",
    "recommended_action": "verify"
  }}]
}}

Severity: high (factual conflict), medium (timeline issues), low (ambiguity). Conservative - only genuine conflicts.
"""

SOURCE_AUTHORITY_EVALUATION_PROMPT = """Evaluate source authority:

{sources_json}

JSON only:
{{
  "source_evaluations": [{{
    "domain": "example.com",
    "type": "gov|edu|major_news|company|blog",
    "authority_score": 0.9,
    "notes": "Official",
    "reliability": "high|medium|low"
  }}]
}}

Scores: 0.9+ (gov/edu), 0.8+ (major news), 0.7+ (industry), 0.5+ (general), 0.3+ (blogs), <0.3 (unreliable).
"""

SEMANTIC_SIMILARITY_GROUPING_PROMPT = """Group similar facts:

{facts_json}

JSON only:
{{
  "fact_groups": [{{
    "group_id": 1,
    "fact_indices": [0, 3],
    "summary": "CEO role",
    "is_corroborating": true
  }}],
  "ungrouped_fact_indices": [2, 5]
}}

Same entity/event. Minor wording ok. Different dates/amounts = separate. 2+ members only.
"""
