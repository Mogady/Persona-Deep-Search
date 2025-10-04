"""
Concise prompts for the Content Extractor Node
"""

EXTRACTION_SYSTEM_PROMPT = """Extract atomic facts about {target_name}. One fact = one claim.

Categories: biographical|professional|financial|behavioral

Good: "CEO of Microsoft Feb 2014"
Bad: "Well-known" (vague), "CEO and leads AI" (split to 2)

Normalize entities. Confidence: 0.9 (direct quotes/official) to 0.3 (vague).
"""

EXTRACTION_USER_PROMPT = """Extract facts about {target_name}:

Title: {title}
URL: {url}
Domain: {domain}
Content: {content}

JSON only:
{{
  "facts": [{{
    "content": "Atomic fact",
    "category": "biographical|professional|financial|behavioral",
    "confidence": 0.75,
    "entities": {{"people": [], "companies": [], "locations": [], "dates": []}}
  }}]
}}

Atomic facts only. Direct statements. Normalize names. No inferences. Empty [] if none.
"""

BATCH_EXTRACTION_PROMPT = """Extract facts from {num_results} results about {target_name}:

{results_text}

JSON only:
{{
  "facts": [{{
    "content": "Atomic fact",
    "category": "biographical|professional|financial|behavioral",
    "confidence": 0.75,
    "source_url": "url",
    "source_domain": "domain",
    "entities": {{"people": [], "companies": [], "locations": [], "dates": []}}
  }}]
}}

Atomic. Direct statements. Normalize. Deduplicate. No vague facts.
"""
