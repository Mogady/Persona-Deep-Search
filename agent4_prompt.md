🤖 AGENT 5: Extraction & Validation Nodes - Your Assignment
📋 Context & Current Status
You are Agent 5 in the Deep Research AI Agent project. Your predecessors have completed:
✅ Agent 1: Database models, config, logger, infrastructure
✅ Agent 2: AI model clients (GeminiClient, AnthropicClient) with embeddings & NER
✅ Agent 3: Search tools (SerpApiSearch, BraveSearch, SearchOrchestrator)
✅ Agent 4: Query Planner & Search Executor nodes with AI-powered enhancements
You are NOT blocked. All dependencies are ready, including:
✅ Gemini Pro client with advanced NER
✅ Claude Sonnet client for complex reasoning
✅ Search results from Agent 4's SearchExecutorNode
🎯 Your Mission
Implement the Content Extractor and Validator nodes that transform raw search results into verified, structured facts with confidence scores. Your work determines the precision (90%+ target) - the most critical quality metric.
🏗️ What You Need to Build
File Structure
src/agents/nodes/
├── extractor.py         # ContentExtractorNode class
└── validator.py         # ValidatorNode class

src/prompts/templates/
├── extractor_prompt.py  # Extraction prompt templates
└── validator_prompt.py  # Validation prompt templates
📝 Detailed Specifications
1. src/agents/nodes/extractor.py
ContentExtractorNode class:
from typing import Dict, List, Any
from datetime import datetime
from src.tools.models import ModelFactory
from src.utils.logger import get_logger
from src.prompts.templates.extractor_prompt import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT
)

class ContentExtractorNode:
    """
    Extracts structured facts from raw search results.
    Uses Gemini Pro 2.5 for precise extraction with entity recognition.
    """
    
    def __init__(self):
        self.client = ModelFactory.get_optimal_model_for_task("extraction")  # Gemini Pro
        self.logger = get_logger(__name__)
    
    def execute(self, state: Dict) -> Dict:
        """
        Extract facts from raw_search_results.
        
        Args:
            state: Current ResearchState dict with:
                - raw_search_results: List[SearchResult] from Agent 4
                - target_name: str
                - collected_facts: List[Dict] (append new facts)
        
        Returns:
            Updated state with new facts added to collected_facts
        """
        pass
    
    def _extract_facts_from_result(self, result: SearchResult, target_name: str) -> List[Dict]:
        """Extract facts from a single search result."""
        pass
    
    def _batch_extract_facts(self, results: List[SearchResult], target_name: str) -> List[Dict]:
        """Extract facts from multiple results efficiently (batch processing)."""
        pass
    
    def _categorize_fact(self, fact_content: str) -> str:
        """Categorize fact as biographical, professional, financial, or behavioral."""
        pass
    
    def _assign_preliminary_confidence(self, fact: Dict, source_domain: str) -> float:
        """Assign preliminary confidence score (0.0-1.0) based on source quality."""
        pass
    
    def _normalize_entities(self, entities: Dict) -> Dict:
        """Normalize extracted entities (dates, amounts, names)."""
        pass
    
    def _filter_low_quality_facts(self, facts: List[Dict]) -> List[Dict]:
        """Filter out vague, redundant, or low-quality facts."""
        pass
Fact Structure (output):
{
    "content": "Satya Nadella became CEO of Microsoft in February 2014",
    "source_url": "https://example.com/article",
    "source_domain": "example.com",
    "extracted_date": datetime.utcnow(),
    "confidence": 0.75,  # Preliminary score
    "category": "professional",
    "entities": {
        "people": ["Satya Nadella"],
        "companies": ["Microsoft"],
        "dates": ["February 2014"]
    }
}
Key Requirements:
Batch processing: Process 5-10 results per API call to reduce costs
Use Gemini Pro's NER: Leverage extract_entities_advanced() from Agent 2
Quality filtering: Reject facts like "He is well-known" (too vague)
Atomic facts: One fact = one claim (split complex statements)
Error handling: If one result fails, continue with others
2. src/agents/nodes/validator.py
ValidatorNode class:
from typing import Dict, List, Set
from datetime import datetime, timedelta
from src.tools.models import ModelFactory
from src.utils.logger import get_logger
from src.prompts.templates.validator_prompt import (
    VALIDATION_SYSTEM_PROMPT,
    CROSS_REFERENCE_PROMPT,
    CONTRADICTION_DETECTION_PROMPT
)

class ValidatorNode:
    """
    Validates and cross-references facts using Claude Sonnet for complex reasoning.
    Adjusts confidence scores based on multiple factors.
    """
    
    def __init__(self):
        self.client = ModelFactory.get_optimal_model_for_task("complex_reasoning")  # Claude
        self.logger = get_logger(__name__)
        
        # Authoritative domain patterns
        self.authoritative_domains = {
            "gov": 0.9,    # Government sites
            "edu": 0.85,   # Educational institutions
            "org": 0.75,   # Official organizations
        }
        
        self.high_quality_news = {
            "nytimes.com", "wsj.com", "reuters.com", "bloomberg.com",
            "ft.com", "economist.com", "bbc.com", "apnews.com"
        }
    
    def execute(self, state: Dict) -> Dict:
        """
        Validate facts and adjust confidence scores.
        
        Args:
            state: Current ResearchState dict with:
                - collected_facts: List[Dict] (newly extracted facts)
        
        Returns:
            Updated state with validated facts (adjusted confidence scores)
        """
        pass
    
    def _cross_reference_facts(self, facts: List[Dict]) -> Dict[str, List[Dict]]:
        """Group similar facts and find corroborating evidence."""
        pass
    
    def _detect_contradictions(self, fact_groups: Dict[str, List[Dict]]) -> List[Dict]:
        """Detect contradicting facts using Claude's reasoning."""
        pass
    
    def _score_source_authority(self, source_domain: str) -> float:
        """Rate domain reliability (0.0-1.0)."""
        pass
    
    def _calculate_final_confidence(self, fact: Dict, corroborations: int, contradictions: bool, source_score: float, is_recent: bool) -> float:
        """
        Calculate final confidence score.
        
        Algorithm:
        - Base score from extraction
        - +0.2 if multiple independent sources (corroborations >= 2)
        - +0.1 if authoritative domain
        - +0.1 if recent information (< 6 months)
        - -0.3 if contradictions found
        - -0.1 if single source only
        - Clamp to [0.0, 1.0]
        """
        pass
    
    def _check_recency(self, fact: Dict) -> bool:
        """Check if fact is from recent information (< 6 months)."""
        pass
    
    def _use_semantic_similarity_for_grouping(self, facts: List[Dict]) -> Dict[str, List[Dict]]:
        """Use Gemini embeddings to group semantically similar facts (ENHANCED)."""
        pass
Confidence Scoring Example:
# Example 1: High confidence
fact = {
    "content": "Satya Nadella is CEO of Microsoft",
    "source_domain": "microsoft.com",
    "confidence": 0.80  # Base from extractor
}
# After validation:
# - Authoritative domain (microsoft.com): +0.1
# - Multiple sources (found in 3 results): +0.2
# - Recent (2024 article): +0.1
# Final confidence: 1.0 (capped)

# Example 2: Low confidence
fact = {
    "content": "He attended some university",
    "source_domain": "blog.example.com",
    "confidence": 0.50  # Base (vague)
}
# After validation:
# - Single source only: -0.1
# - Low authority domain: 0
# Final confidence: 0.40
3. src/prompts/templates/extractor_prompt.py
EXTRACTION_SYSTEM_PROMPT = """You are an expert fact extraction specialist for due diligence research.

Your task:
1. Extract ONLY factual, verifiable statements about {target_name}
2. Create atomic facts (one fact = one claim)
3. Include source context for confidence scoring
4. Categorize each fact appropriately

Quality standards:
- ✅ GOOD: "Satya Nadella became CEO of Microsoft in February 2014"
- ❌ BAD: "He is well-known" (too vague)
- ❌ BAD: "Satya Nadella is CEO of Microsoft and leads AI initiatives" (split into 2 facts)

Categories:
- biographical: Birth, education, family, personal background
- professional: Career, roles, achievements, employment history
- financial: Compensation, investments, financial status, transactions
- behavioral: Patterns, habits, public statements, controversies
"""

EXTRACTION_USER_PROMPT = """Extract facts from the following search result about {target_name}:

Title: {title}
URL: {url}
Content: {content}

Return ONLY valid JSON in this format:
{{
  "facts": [
    {{
      "content": "Clear, atomic factual statement",
      "category": "biographical|professional|financial|behavioral",
      "confidence": 0.75,
      "entities": {{
        "people": ["Name 1", "Name 2"],
        "companies": ["Company 1"],
        "locations": ["City", "Country"],
        "dates": ["Date 1"]
      }}
    }}
  ]
}}

Rules:
- Only extract facts directly stated in the content
- No inferences or assumptions
- Atomic facts only (split complex statements)
- Confidence: 0.9 for direct quotes, 0.7 for clear statements, 0.5 for implied/indirect
"""
4. src/prompts/templates/validator_prompt.py
VALIDATION_SYSTEM_PROMPT = """You are an expert fact validator for intelligence analysis.

Your task:
1. Cross-reference facts from multiple sources
2. Detect contradictions and inconsistencies
3. Assess source reliability
4. Provide confidence adjustments with reasoning

Validation principles:
- Multiple independent sources increase confidence
- Authoritative sources (.gov, .edu, major news) are more reliable
- Recent information is preferred over outdated
- Contradictions significantly reduce confidence
- Single-source facts need careful assessment
"""

CROSS_REFERENCE_PROMPT = """Cross-reference the following facts about {target_name}:

Facts to validate:
{facts_json}

For each fact, determine:
1. How many independent sources support it?
2. Are there any contradictions?
3. What is the source quality?

Return JSON:
{{
  "fact_validations": [
    {{
      "fact_id": 0,
      "corroborating_sources": 3,
      "contradictions": [],
      "confidence_adjustment": +0.3,
      "reasoning": "Supported by 3 independent high-quality sources"
    }}
  ]
}}
"""

CONTRADICTION_DETECTION_PROMPT = """Detect contradictions in these facts about {target_name}:

Facts:
{facts_json}

Identify:
1. Direct contradictions (conflicting dates, roles, etc.)
2. Inconsistencies (timeline issues, incompatible claims)

Return JSON:
{{
  "contradictions": [
    {{
      "fact_ids": [1, 5],
      "type": "direct|indirect",
      "description": "Fact 1 states X, but Fact 5 states Y",
      "severity": "high|medium|low"
    }}
  ]
}}
"""
🔧 Environment & Dependencies
Virtual Environment
IMPORTANT: Use the existing .venv in the project root:
# Activate environment
source /home/mogady/Desktop/Elile-Assessment/.venv/bin/activate

# If you need additional packages (unlikely):
pip install <package-name>
# Then update requirements.txt
Imports You'll Need
# From previous agents (already available):
from src.tools.models import ModelFactory, GeminiClient, AnthropicClient
from src.tools.search import SearchResult
from src.utils.logger import get_logger
from src.utils.config import Config

# Standard library:
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import json
import re
Using AI Clients
# Gemini Pro for extraction
gemini_client = ModelFactory.get_optimal_model_for_task("extraction")
entities = gemini_client.extract_entities_advanced(text)  # AI-powered NER

# Claude for validation
claude_client = ModelFactory.get_optimal_model_for_task("complex_reasoning")
validation = claude_client.generate(prompt)

# Embeddings for semantic grouping (OPTIONAL ENHANCEMENT)
embeddings = gemini_client.generate_embeddings(texts)
✅ Testing Requirements
Unit Tests (tests/unit/test_nodes_extractor_validator_mock.py)
def test_fact_extraction_from_result():
    """Test extracting facts from a single search result"""
    pass

def test_preliminary_confidence_scoring():
    """Test confidence assignment based on source quality"""
    pass

def test_fact_categorization():
    """Test biographical/professional/financial/behavioral classification"""
    pass

def test_cross_referencing():
    """Test grouping similar facts"""
    pass

def test_contradiction_detection():
    """Test flagging conflicting facts"""
    pass

def test_source_authority_scoring():
    """Test domain reliability scoring"""
    pass

def test_confidence_score_adjustment():
    """Test confidence scoring algorithm"""
    pass
Integration Tests (tests/integration/test_extractor_validator_flow.py)
def test_extractor_to_validator_flow():
    """Test end-to-end fact extraction → validation"""
    # Create sample search results
    # Run extractor
    # Run validator
    # Verify confidence scores adjusted correctly
    pass
📊 Success Criteria
Your work is complete when:
 ContentExtractorNode extracts 10-20 facts per search result set
 Facts are atomic, clear, and well-categorized
 Preliminary confidence scores are reasonable (0.5-0.9 range)
 ValidatorNode successfully cross-references facts
 Confidence scores adjusted based on corroboration
 Contradictions detected and flagged
 Source authority scoring works (gov/edu > blogs)
 Unit tests pass (7 tests minimum)
 Integration test passes (extractor → validator flow)
 Target metrics:
Precision: >90% (facts are correct)
Fact quality: Clear, atomic, verifiable
Confidence calibration: Scores correlate with accuracy
🚀 Development Approach
Step 1: Implement Extractor (90 min)
Create ContentExtractorNode skeleton
Implement _extract_facts_from_result() using Gemini Pro
Add categorization logic
Add preliminary confidence scoring
Test with sample search results
Step 2: Implement Validator (90 min)
Create ValidatorNode skeleton
Implement _cross_reference_facts() with semantic grouping
Implement _detect_contradictions() using Claude
Implement confidence adjustment algorithm
Test with extracted facts
Step 3: Create Prompts (30 min)
Write extraction prompts with examples
Write validation prompts with clear criteria
Test prompts with real API calls
Step 4: Testing (45 min)
Write unit tests (mocked)
Write integration test (real APIs)
Verify with sample data
Total estimated time: 4-5 hours
🎯 Key Enhancements to Leverage
Since Agent 4 added AI-powered features, you should: ✅ Use Gemini's Advanced NER (extract_entities_advanced())
More accurate than regex
Handles abbreviations, lowercase, normalization
✅ Use Embeddings for Semantic Grouping (optional enhancement)
Group similar facts using cosine similarity
Better than simple text matching
✅ Use Claude for Complex Reasoning
Contradiction detection
Confidence adjustment reasoning
Contextual validation
📞 When You're Done
Update PROGRESS_UPDATE.md:
Mark Agent 5 as ✅ COMPLETED
Add file paths and line counts
Document any decisions
Report completion with:
Files created and line counts
Test results (how many passing)
Sample output (show extracted facts)
Recommendations for Agent 6
Good luck, Agent 5! Your work transforms raw data into trusted intelligence. Focus on quality and precision - that's what separates good research from great research. 🎯
