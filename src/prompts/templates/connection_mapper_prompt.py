"""
Concise prompts for the Connection Mapper Node
"""

CONNECTION_MAPPER_PROMPT = """
Map relationships between entities from facts. Output valid JSON array only.

**Schema:**
```json
[{{
  "entity_a": "Entity A Name",
  "entity_b": "Entity B Name",
  "relationship_type": "Employment|Investment|Board/Advisory|Family/Personal|Co-founder/Business partner|Educational|Other",
  "evidence": ["Exact fact text"],
  "confidence": 0.80,
  "time_period": "2018-2020|Ongoing|Unknown"
}}]
```

**Relationship Types:**
- Employment: Person→Company (directional)
- Investment: Investor→Investee (directional)
- Board/Advisory: Person→Board (directional)
- Family/Personal: Person↔Person (symmetric)
- Co-founder/Business partner: ↔ (symmetric)
- Educational: Person↔School (symmetric)
- Other: Any other type

**Examples:**

```json
{{
  "entity_a": "Satya Nadella",
  "entity_b": "Microsoft Corporation",
  "relationship_type": "Employment",
  "evidence": ["CEO of Microsoft Feb 2014", "Led Cloud group"],
  "confidence": 0.95,
  "time_period": "2014-Present"
}}
```

```json
{{
  "entity_a": "Larry Page",
  "entity_b": "Sergey Brin",
  "relationship_type": "Co-founder/Business partner",
  "evidence": ["Co-founded Google 1998"],
  "confidence": 0.95,
  "time_period": "1998-Present"
}}
```

**Rules:**
- Directional: Employment, Investment, Board/Advisory (order matters)
- Symmetric: Family, Educational, Co-founder (order doesn't matter)
- Use proper names, title case, consistent naming
- Copy exact fact text in evidence
- More evidence = higher confidence
- Extract time periods when available
- No self-connections (entity_a ≠ entity_b)
- Empty array [] if no connections

**Facts:**
{facts}

**Output JSON array only:**
"""
