# Deep Research AI Agent - Project Overview

## 🎯 Project Mission

Build an autonomous research agent capable of conducting comprehensive investigations on individuals or entities to uncover hidden connections, potential risks, and strategic insights for due diligence operations.

## 📊 Project Context

**Timeline:** 4 days development
**Deployment:** GCP (minimal resources)
**UI Framework:** Chainlit
**Core Framework:** LangGraph for agent orchestration
**Development Approach:** Multi-agent development using Claude Code agents

## 🏗️ System Architecture

### High-Level Flow
```
User Input (Chainlit UI)
    ↓
LangGraph Orchestrator (State Machine)
    ↓
7 Specialized Agent Nodes
    ↓
Tools Layer (Search APIs, AI Models, Database)
    ↓
Structured Report Output
```

### Agent Nodes Overview

1. **Query Planner Node** - Generates intelligent search queries
2. **Search Executor Node** - Executes searches via SerpApi
3. **Content Extractor Node** - Extracts structured facts from results
4. **Validator Node** - Cross-references and scores confidence
5. **Risk Analyzer Node** - Identifies red flags and concerns
6. **Connection Mapper Node** - Maps entity relationships
7. **Report Generator Node** - Synthesizes comprehensive reports

## 🔧 Technology Stack

### Core Frameworks
- **LangGraph**: Agent workflow orchestration
- **LangChain**: Tool integration and prompt management
- **Chainlit**: User interface

### AI Models (Multi-Model Strategy)
- **Anthropic Claude Sonnet 4.5**: Complex reasoning, risk analysis, validation, report generation
- **Gemini Pro 2.5**: Fact extraction, entity recognition, connection mapping
- **Gemini Flash 2.5**: Fast query generation, preliminary filtering, quick analysis

### Search & Data Collection
- **Primary**: SerpApi (Google Search API, comprehensive results)
- **Secondary**: Firecrawl API (clean content extraction)
- **Fallback**: Playwright (minimal, only for critical targets)

### Infrastructure
- **Database**: PostgreSQL (Cloud SQL on GCP)
- **Storage**: GCP Cloud Storage (logs, reports)
- **Secrets**: GCP Secret Manager
- **Deployment**: GCP Cloud Run

## 📁 Project Structure

```
deep-research-agent/
├── CLAUDE.md                      # This file
├── PROGRESS_UPDATE.md             # Development tracking
├── AGENT_PROMPTS.md               # Individual agent prompts
├── README.md                      # User-facing documentation
├── requirements.txt
├── .env.example
│
├── src/
│   ├── agents/
│   │   ├── graph.py              # LangGraph workflow definition
│   │   ├── state.py              # State schema (TypedDict)
│   │   └── nodes/
│   │       ├── __init__.py
│   │       ├── planner.py
│   │       ├── searcher.py
│   │       ├── extractor.py
│   │       ├── validator.py
│   │       ├── risk_analyzer.py
│   │       ├── connection_mapper.py
│   │       └── reporter.py
│   │
│   ├── tools/
│   │   ├── search/
│   │   │   ├── __init__.py
│   │   │   ├── serp_api_search.py
│   │   │   ├── firecrawl.py      # Optional backup
│   │   │   └── web_scraper.py    # Minimal Playwright
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── claude_client.py
│   │       ├── gemini_pro_client.py
│   │       └── gemini_flash_client.py
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── templates/
│   │       ├── planner_prompt.py
│   │       ├── extractor_prompt.py
│   │       ├── validator_prompt.py
│   │       ├── risk_analyzer_prompt.py
│   │       ├── connection_mapper_prompt.py
│   │       └── reporter_prompt.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py             # SQLAlchemy models
│   │   └── repository.py         # Data access layer
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── rate_limiter.py
│   │
│   └── ui/
│       └── chainlit_app.py
│
├── tests/
│   ├── evaluation/
│   │   ├── personas.json         # 3 test personas with ground truth
│   │   ├── test_agent.py
│   │   └── metrics.py
│   └── unit/
│       ├── test_nodes.py
│       └── test_tools.py
│
├── infra/
│   ├── terraform/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── docker/
│       └── Dockerfile
│
├── logs/
├── reports/
└── docs/
    ├── ARCHITECTURE.md
    ├── API_REFERENCE.md
    └── DEPLOYMENT.md
```

## 🔄 LangGraph State Schema

```python
from typing import TypedDict, List, Dict, Set, Optional
from datetime import datetime

class Fact(TypedDict):
    content: str
    source_url: str
    extracted_date: datetime
    confidence: float
    category: str  # biographical, professional, financial, behavioral

class Connection(TypedDict):
    entity_a: str
    entity_b: str
    relationship_type: str
    evidence: List[str]
    confidence: float

class RiskFlag(TypedDict):
    severity: str  # Low, Medium, High, Critical
    category: str  # legal, financial, reputational, compliance
    description: str
    evidence: List[str]
    confidence: float

class SearchQuery(TypedDict):
    query: str
    timestamp: datetime
    results_count: int
    relevance_score: float

class ResearchState(TypedDict):
    # Input
    target_name: str
    research_depth: int  # Max iterations (5-7)
    
    # Progressive Knowledge Building
    collected_facts: List[Fact]
    connections: List[Connection]
    risk_flags: List[RiskFlag]
    search_history: List[SearchQuery]
    
    # Planning & Control
    current_iteration: int
    next_queries: List[str]
    explored_topics: Set[str]
    confidence_scores: Dict[str, float]
    
    # Metadata
    session_id: str
    start_time: datetime
    
    # Output
    final_report: Optional[str]
    connection_graph: Optional[Dict]
```

## 🔍 Search Strategy

### Progressive Investigation Levels

**Level 1: Broad Discovery (Iteration 1)**
- Basic biographical search
- Professional background
- News mentions
- Social media presence

**Level 2: Targeted Investigation (Iterations 2-3)**
- Company affiliations + roles
- Geographic locations + timeline
- Educational background verification
- Public statements/interviews

**Level 3: Deep Connection Mining (Iterations 4-5)**
- Board memberships
- Business partnerships
- Family/associate connections
- Investment portfolio

**Level 4: Risk Assessment (Iteration 6)**
- Legal issues validation
- Financial irregularities
- Reputational concerns
- Regulatory actions

**Level 5: Gap Filling (Iteration 7)**
- Verify low-confidence facts
- Find additional sources
- Resolve contradictions

### Query Generation Rules

1. **Start Broad**: Initial queries are general
2. **Follow Leads**: Each iteration builds on previous findings
3. **Diversify Sources**: Ensure multiple domains per topic
4. **Avoid Repetition**: Track explored topics
5. **Strategic Depth**: Focus on anomalies and gaps

## 🎯 Evaluation Framework

### Test Personas (Ground Truth Dataset)

**Persona 1: "The Clean Executive"**
- Public figure, minimal controversy
- 15-20 hidden facts (easy to medium difficulty)
- Tests: Comprehensive fact gathering, accuracy

**Persona 2: "The Controversial Entrepreneur"**
- Multiple ventures, some failed
- 15-20 hidden facts (medium to hard difficulty)
- Tests: Risk detection, connection mapping

**Persona 3: "The Low-Profile Investor"**
- Minimal online presence
- 15-20 hidden facts (hard difficulty)
- Tests: Creative search strategies, fragment synthesis

### Success Metrics

- **Fact Discovery Rate**: % of hidden facts found (target: 70%+)
- **Precision**: % of facts correctly identified (target: 90%+)
- **Source Diversity**: Unique domains (target: 10+ per session)
- **Confidence Calibration**: Score accuracy correlation
- **Query Efficiency**: Facts per search (target: 2-3)
- **Execution Time**: Total research duration (target: 5-7 minutes)

## 🚀 Development Phases

### Day 1: Foundation
- Project structure setup
- Database schema and models
- AI model clients (Claude, GPT-4, Gemini)
- Tavily search integration
- Basic LangGraph workflow

### Day 2: Core Nodes
- Implement all 7 agent nodes
- Prompt engineering for each model
- State management logic
- Search progression algorithm
- Fact extraction and validation

### Day 3: Integration & Testing
- End-to-end workflow testing
- Evaluation with 3 test personas
- Refinement based on metrics
- Error handling and logging
- Chainlit UI development

### Day 4: Polish & Deploy
- Performance optimization
- Comprehensive documentation
- GCP deployment setup
- Demo preparation
- Edge case handling

## 🔐 Security & Best Practices

- Store API keys in GCP Secret Manager
- Rate limiting on all API calls
- Input validation and sanitization
- Error handling with graceful degradation
- Audit logging for all searches
- GDPR/privacy considerations for personal data

## 📊 Key Performance Indicators

- **Latency**: < 10 minutes per full research
- **API Costs**: < $1 per research session
- **Accuracy**: 90%+ precision on facts
- **Recall**: 70%+ of hidden facts discovered
- **Source Quality**: 80%+ from authoritative domains

## 🎓 References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [SerpApi Documentation](https://serpapi.com/search-api)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)
- [Chainlit Documentation](https://docs.chainlit.io/)

## 💡 Key Differentiators

1. **Multi-Model Orchestration**: Each model used for its strengths
2. **Intelligent Query Evolution**: Sophisticated search progression
3. **Confidence Calibration**: Accurate scoring with source tracking
4. **Audit Trail**: Complete transparency
5. **Speed**: Optimized for 5-7 minute comprehensive research
6. **Visual Outputs**: Connection graphs and timelines

---