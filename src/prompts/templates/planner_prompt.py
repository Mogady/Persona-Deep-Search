"""
Concise prompts for the Query Planner Node
"""

BROAD_DISCOVERY_PROMPT = """Generate 3-5 search queries for: {target_name}

Focus: Professional role, companies, news, education, social media.

Unique specific queries. Different angles (LinkedIn, news, sites).

IMPORTANT: Return ONLY a JSON array of strings. No explanations, no markdown code blocks, no additional text.
Format: ["query 1", "query 2", "query 3"]

Example: ["John Doe LinkedIn profile", "John Doe CEO company", "John Doe news articles"]
"""

TARGETED_INVESTIGATION_PROMPT = """Deep investigation on: {target_name}

Facts: {facts_summary}

Entities - People: {people} | Companies: {companies} | Locations: {locations}

Generate 3-5 NEW queries: Dig into companies/roles, verify timeline, find connections, investigate locations.

Avoid: {explored_topics}

IMPORTANT: Return ONLY a JSON array of strings. No explanations, no markdown code blocks, no additional text.
Format: ["query 1", "query 2", "query 3"]
"""

CONNECTION_MINING_PROMPT = """Map networks for: {target_name}

People: {people} | Companies: {companies}

Generate 3-5 queries: Boards, partnerships, investments, family, shared affiliations.

IMPORTANT: Return ONLY a JSON array of strings. No explanations, no markdown code blocks, no additional text.
Format: ["query 1", "query 2", "query 3"]
"""

VALIDATION_PROMPT = """Verify info for: {target_name}

Low-confidence facts: {low_confidence_facts}

Generate 3-5 queries: Find more sources, resolve contradictions, authoritative (.gov/.edu/major news).

IMPORTANT: Return ONLY a JSON array of strings. No explanations, no markdown code blocks, no additional text.
Format: ["query 1", "query 2", "query 3"]
"""

# Helper function to select the right prompt based on iteration
def get_prompt_for_iteration(iteration: int) -> str:
    """
    Select the appropriate prompt template based on iteration number.

    Args:
        iteration: Current research iteration (1-7)

    Returns:
        Appropriate prompt template string
    """
    if iteration == 1:
        return BROAD_DISCOVERY_PROMPT
    elif iteration in [2, 3]:
        return TARGETED_INVESTIGATION_PROMPT
    elif iteration in [4, 5]:
        return CONNECTION_MINING_PROMPT
    else:  # 6, 7
        return VALIDATION_PROMPT
