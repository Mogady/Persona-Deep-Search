# Deep Research AI Agent

> An autonomous multi-agent research system for conducting comprehensive due diligence investigations on individuals and entities.

## 🎯 Project Overview

This AI-powered research agent autonomously investigates targets by intelligently searching the web, extracting facts, validating information across sources, identifying risks, mapping connections, and generating comprehensive reports. Built with LangGraph orchestration and a multi-model AI strategy for optimal performance.

### Key Capabilities

- **Progressive Search Strategy**: Iteratively deepens investigation across 7 iterations (broad → targeted → connections → validation)
- **Multi-Source Validation**: Cross-references facts across multiple sources with confidence scoring
- **Risk Detection**: Automatically identifies legal, financial, reputational, compliance, and behavioral red flags
- **Network Analysis**: Maps relationships between entities, organizations, and events
- **Comprehensive Reporting**: Generates structured markdown reports with evidence citations

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | LangGraph | Multi-agent workflow coordination |
| **AI Models** | Claude Sonnet 4.5 | Complex reasoning, risk analysis, reporting |
| | Gemini Pro 2.5 | Fact extraction, entity recognition |
| | Gemini Flash 2.5 | Fast query generation, filtering |
| **Search** | SerpApi | Google Search API integration |
| **Database** | PostgreSQL | State persistence, audit trail |
| **UI** | Chainlit | Interactive web interface |
| **Deployment** | Docker + GCP | Containerized cloud deployment |

---

## 🔄 System Architecture & Flow

### High-Level Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         USER INPUT (Chainlit UI)                          │
│                   "Research: [Target Name]" + Depth (1-7)                 │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH ORCHESTRATOR                             │
│                     (ResearchWorkflow State Machine)                      │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │    CREATE SESSION IN DB       │
                    │  session_id, target, status   │
                    └───────────────┬───────────────┘
                                    │
        ╔═══════════════════════════╩═══════════════════════════╗
        ║              ITERATION LOOP (1-7 times)                ║
        ║                                                        ║
        ║  ┌─────────────────────────────────────────────────┐  ║
        ║  │  NODE 1: QUERY PLANNER (Gemini Flash)          │  ║
        ║  │  • Iteration 1: Broad discovery queries        │  ║
        ║  │  • Iteration 2-3: Targeted investigation       │  ║
        ║  │  • Iteration 4-5: Deep connection mining       │  ║
        ║  │  • Iteration 6-7: Validation queries           │  ║
        ║  │  OUTPUT: 3-5 unique search queries             │  ║
        ║  └──────────────────────┬──────────────────────────┘  ║
        ║                         │                              ║
        ║                         ▼                              ║
        ║  ┌─────────────────────────────────────────────────┐  ║
        ║  │  NODE 2: SEARCH EXECUTOR (SerpApi)             │  ║
        ║  │  • Parallel query execution (5 workers)        │  ║
        ║  │  • Result deduplication (URL + content)        │  ║
        ║  │  • Source diversity tracking                   │  ║
        ║  │  OUTPUT: 10-50 search results                  │  ║
        ║  └──────────────────────┬──────────────────────────┘  ║
        ║                         │                              ║
        ║                         ▼                              ║
        ║  ┌─────────────────────────────────────────────────┐  ║
        ║  │  NODE 3: CONTENT EXTRACTOR (Gemini Pro)        │  ║
        ║  │  • Batch extraction (5 results/call)           │  ║
        ║  │  • Atomic fact generation                      │  ║
        ║  │  • Category classification (4 types)           │  ║
        ║  │  • Entity extraction (NER)                     │  ║
        ║  │  OUTPUT: 5-20 structured facts                 │  ║
        ║  └──────────────────────┬──────────────────────────┘  ║
        ║                         │                              ║
        ║                         ▼                              ║
        ║  ┌─────────────────────────────────────────────────┐  ║
        ║  │  NODE 4: VALIDATOR (Claude Sonnet)             │  ║
        ║  │  • Semantic similarity grouping (embeddings)   │  ║
        ║  │  • Cross-reference facts across sources        │  ║
        ║  │  • Contradiction detection                     │  ║
        ║  │  • Confidence adjustment (+/- scoring)         │  ║
        ║  │  OUTPUT: Validated facts with confidence       │  ║
        ║  └──────────────────────┬──────────────────────────┘  ║
        ║                         │                              ║
        ║                         ▼                              ║
        ║  ┌─────────────────────────────────────────────────┐  ║
        ║  │  NODE 5: RISK ANALYZER (Claude Sonnet)         │  ║
        ║  │  • Pattern detection across facts              │  ║
        ║  │  • 5 categories: Legal, Financial, etc.        │  ║
        ║  │  • Severity classification (Low → Critical)    │  ║
        ║  │  OUTPUT: Risk flags with evidence              │  ║
        ║  └──────────────────────┬──────────────────────────┘  ║
        ║                         │                              ║
        ║                         ▼                              ║
        ║  ┌─────────────────────────────────────────────────┐  ║
        ║  │  NODE 6: CONNECTION MAPPER (Gemini Pro)        │  ║
        ║  │  • Entity relationship extraction              │  ║
        ║  │  • 6 types: Employment, Investment, etc.       │  ║
        ║  │  • Graph construction (nodes + edges)          │  ║
        ║  │  OUTPUT: Connection graph                      │  ║
        ║  └──────────────────────┬──────────────────────────┘  ║
        ║                         │                              ║
        ║                         ▼                              ║
        ║  ┌─────────────────────────────────────────────────┐  ║
        ║  │         CHECKPOINT: Save to Database            │  ║
        ║  │  • Facts, Risks, Connections, Search History   │  ║
        ║  └──────────────────────┬──────────────────────────┘  ║
        ║                         │                              ║
        ║                         ▼                              ║
        ║  ┌─────────────────────────────────────────────────┐  ║
        ║  │          CONDITIONAL LOGIC                      │  ║
        ║  │  ┌─────────────────────────────────────────┐   │  ║
        ║  │  │ Continue if:                            │   │  ║
        ║  │  │  • current_iteration < research_depth   │   │  ║
        ║  │  │  • new facts discovered                 │   │  ║
        ║  │  │  • not manually stopped                 │   │  ║
        ║  │  └─────────────────────────────────────────┘   │  ║
        ║  └─────────────────────┬─────────────────────────────┘  ║
        ║            YES ◄───────┘                              ║
        ║             │                                         ║
        ║             └──────────── LOOP BACK TO NODE 1        ║
        ║                                                       ║
        ║            NO ──────────────────┐                    ║
        ╚═══════════════════════════════╪═══════════════════════╝
                                        │
                                        ▼
                    ┌───────────────────────────────────┐
                    │  NODE 7: REPORT GENERATOR         │
                    │         (Claude Sonnet)            │
                    │  • Executive summary               │
                    │  • Subject overview                │
                    │  • Facts by category               │
                    │  • Risk assessment                 │
                    │  • Network analysis                │
                    │  • Timeline                        │
                    │  • Recommendations                 │
                    │  OUTPUT: Markdown report           │
                    └───────────────┬───────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │    SAVE FINAL STATE TO DB         │
                    │  • Complete session                │
                    │  • Final report text               │
                    │  • Connection graph JSON           │
                    └───────────────┬───────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │     RETURN TO USER (Chainlit)     │
                    │  • Display formatted report       │
                    │  • Show facts table               │
                    │  • Visualize connections          │
                    │  • Download options (MD/JSON)     │
                    └───────────────────────────────────┘
```

### State Flow Example

**Input**: Research "Elon Musk" with depth=7

```
Iteration 1: Broad Discovery
  ├─ Queries: ["Elon Musk biography", "Elon Musk companies", "Elon Musk news"]
  ├─ Search: 30 results from diverse sources
  ├─ Extract: 15 facts (Tesla CEO, SpaceX founder, etc.)
  └─ State: 15 facts collected → CONTINUE

Iteration 2: Targeted Investigation
  ├─ Queries: ["Elon Musk Tesla compensation", "SpaceX Starship development"]
  ├─ Extract: 12 new facts (compensation details, SpaceX milestones)
  └─ State: 27 facts collected → CONTINUE

Iteration 3: Targeted Investigation
  ├─ Queries: ["Elon Musk Twitter acquisition", "Neuralink FDA approval"]
  ├─ Extract: 10 new facts
  └─ State: 37 facts collected → CONTINUE

Iteration 4: Deep Connection Mining
  ├─ Queries: ["Elon Musk board positions", "Musk business partnerships"]
  ├─ Connections: Tesla-SolarCity, PayPal cofounders, etc.
  └─ State: 45 facts, 8 connections → CONTINUE

Iteration 5: Deep Connection Mining
  ├─ Queries: ["Elon Musk investors", "Musk family background"]
  ├─ Connections: Peter Thiel, venture capital relationships
  └─ State: 52 facts, 15 connections → CONTINUE

Iteration 6: Validation & Gap Filling
  ├─ Queries: ["Elon Musk SEC lawsuit details", verify low-confidence facts]
  ├─ Risks: SEC violations (Medium severity)
  └─ State: 58 facts, 15 connections, 3 risks → CONTINUE

Iteration 7: Final Validation
  ├─ Queries: Verify remaining low-confidence facts
  ├─ Final validation and cross-referencing
  └─ State: 61 facts, 18 connections, 4 risks → REPORT

Final Report: Comprehensive markdown with all findings
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **PostgreSQL 14+** (or Docker)
- **API Keys**:
  - Anthropic API key (Claude Sonnet)
  - Google AI API key (Gemini Pro & Flash)
  - SerpApi key (Google Search)

### 1. Clone Repository

```bash
git clone <repository-url>
cd Elile-Assessment
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys and settings
nano .env  # or use your preferred editor
```

**Required environment variables:**

```bash
# API Keys (REQUIRED)
ANTHROPIC_API_KEY=sk-ant-xxxxx
GOOGLE_API_KEY=xxxxx
SERPAPI_KEY=xxxxx

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/research_agent

# Application Settings
MAX_SEARCH_ITERATIONS=7
LOG_LEVEL=INFO
ENVIRONMENT=development

# Model Configuration
CLAUDE_MODEL=claude-sonnet-4-5-20250929
GEMINI_PRO_MODEL=gemini-2.5-pro
GEMINI_FLASH_MODEL=gemini-2.5-flash

# Performance Settings
MAX_CONCURRENT_SEARCH_CALLS=5
MAX_CONCURRENT_LLM_CALLS=10
EXTRACTION_BATCH_SIZE=5
```

### 4. Set Up Database

**Option A: Using Docker (Recommended)**

```bash
# Start PostgreSQL container
docker run -d \
  --name research-db \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=research_agent \
  -e POSTGRES_USER=research_user \
  -p 5432:5432 \
  postgres:14

# Wait for database to be ready (10 seconds)
sleep 10
```

**Option B: Use Existing PostgreSQL**

Ensure PostgreSQL is running and create a database:

```sql
CREATE DATABASE research_agent;
CREATE USER research_user WITH PASSWORD 'yourpassword';
GRANT ALL PRIVILEGES ON DATABASE research_agent TO research_user;
```

### 5. Run Database Migrations

```bash
# Create tables using Alembic
alembic upgrade head
```

### 6. Run the Application

```bash
# Start Chainlit UI
chainlit run src/ui/chainlit_app.py

# The UI will be available at http://localhost:8000
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
# Build the image
docker build -t deep-research-agent -f Dockerfile .
```

### Run with Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: research_agent
      POSTGRES_USER: research_user
      POSTGRES_PASSWORD: yourpassword
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U research_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  app:
    image: deep-research-agent
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://research_user:yourpassword@postgres:5432/research_agent
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      GOOGLE_API_KEY: ${GOOGLE_API_KEY}
      SERPAPI_KEY: ${SERPAPI_KEY}
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./reports:/app/reports

volumes:
  postgres_data:
```

Run with:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Run Docker Manually

```bash
# Run application container (after PostgreSQL is running)
docker run -d \
  --name research-agent \
  -p 8000:8000 \
  --env-file .env \
  -e DATABASE_URL=postgresql://research_user:yourpassword@host.docker.internal:5432/research_agent \
  deep-research-agent

# View logs
docker logs -f research-agent

# Stop container
docker stop research-agent
docker rm research-agent
```

---

## 📊 Performance Metrics

**Typical Performance (7 iterations):**

| Metric | Value |
|--------|-------|
| Execution Time | 5-7 minutes |
| Facts Discovered | 40-60 |
| API Calls | ~210 LLM calls |
| API Cost | $0.50-$1.00 |
| Source Diversity | 15-20 unique domains |
| Precision | >90% |
| Fact Discovery Rate | >70% |

**Optimizations Applied:**

- ✅ 66% reduction in API calls (new_facts strategy)
- ✅ 10x faster extraction (concurrent batch processing)
- ✅ Semantic deduplication (85% accuracy)
- ✅ AI-powered entity extraction (90-95% accuracy)
- ✅ Rate limiting for controlled concurrency

---

## 📁 Project Structure

```
Elile-Assessment/
├── src/
│   ├── agents/
│   │   ├── graph.py              # LangGraph workflow orchestration
│   │   ├── state.py              # State schema (TypedDict)
│   │   └── nodes/                # 7 specialized agent nodes
│   │       ├── planner.py        # Query generation (Gemini Flash)
│   │       ├── searcher.py       # Search execution (SerpApi)
│   │       ├── extractor.py      # Fact extraction (Gemini Pro)
│   │       ├── validator.py      # Validation (Claude Sonnet)
│   │       ├── risk_analyzer.py  # Risk detection (Claude Sonnet)
│   │       ├── connection_mapper.py  # Network mapping (Gemini Pro)
│   │       └── reporter.py       # Report generation (Claude Sonnet)
│   ├── tools/
│   │   ├── models/               # AI model clients
│   │   │   ├── anthropic_client.py
│   │   │   └── gemini_client.py
│   │   └── search/               # Search integrations
│   │       ├── serp_api_search.py
│   │       └── brave_search.py
│   ├── database/
│   │   ├── models.py             # SQLAlchemy models
│   │   ├── repository.py         # Data access layer
│   │   └── migrations/           # Alembic migrations
│   ├── prompts/
│   │   └── templates/            # Prompt templates for each node
│   ├── utils/
│   │   ├── config.py             # Configuration management
│   │   ├── logger.py             # Structured logging (Loguru)
│   │   └── rate_limiter.py       # Concurrent execution control
│   └── ui/
│       └── chainlit_app.py       # Chainlit web interface
├── tests/
│   ├── unit/                     # Unit tests (mocked)
│   ├── integration/              # Integration tests (real APIs)
│   └── evaluation/               # Test personas & metrics
├── Dockerfile                    # Multi-stage Docker build
├── docker-compose.yml            # Docker Compose configuration
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── alembic.ini                   # Database migration config
└── README.md                     # This file
```

---

## 🧪 Testing

```bash
# Run unit tests (no API calls)
pytest tests/unit/ -v

# Run integration tests (requires API keys)
pytest tests/integration/ -v

# Run specific test file
pytest tests/unit/test_nodes_planner_mock.py -v

# Run with coverage
pytest --cov=src tests/
```

---

## 🔒 Security & Privacy

- ✅ API keys stored in environment variables (never committed)
- ✅ Rate limiting on all external API calls
- ✅ Input validation and sanitization
- ✅ Database connection via SQLAlchemy (SQL injection protection)

---

## 📝 Example Usage

### Via Chainlit UI

1. Open browser to `http://localhost:8000`
2. Enter target name: `"Elon Musk"`
3. Select research depth: `7 iterations`
4. Click "Start Research"
5. Watch real-time progress updates
6. Download comprehensive report (Markdown/JSON)

### Via Python API

```python
from src.agents.graph import ResearchWorkflow
from src.database.repository import ResearchRepository
from src.utils.config import Config

# Initialize
config = Config.from_env()
repository = ResearchRepository(config.database.url)
workflow = ResearchWorkflow(repository, config)

# Run research
result = workflow.run_research(
    target_name="Elon Musk",
    research_depth=7
)

# Access results
print(result["final_report"])
print(f"Facts: {len(result['collected_facts'])}")
print(f"Risks: {len(result['risk_flags'])}")
print(f"Connections: {len(result['connections'])}")
```
