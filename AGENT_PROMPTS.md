# Agent Prompts & Instructions

This document contains detailed instructions for each Claude Code agent. Each agent has a specific responsibility and should focus exclusively on their assigned components.

**Important Notes for All Agents:**
- Read CLAUDE.md first to understand the full project
- Check PROGRESS_UPDATE.md to see what's already done
- Update PROGRESS_UPDATE.md when you complete your tasks
- Add comprehensive docstrings and type hints
- Include error handling in all functions
- Write unit tests for critical functions
- Log important operations

**IMPORTANT API & MODEL STRATEGY:**
- Use **SerpApi** for search functionality (NOT Tavily)
- **MULTI-MODEL STRATEGY**:
  - **Anthropic Claude Sonnet 4.5**: Complex reasoning, risk analysis, report generation
  - **Gemini Pro 2.5**: Fact extraction, entity recognition, connection mapping
  - **Gemini Flash 2.5**: Fast query generation and preliminary filtering
- **CLIENT ARCHITECTURE**: One unified client per model family (GeminiClient, AnthropicClient)

**SIMPLIFIED ARCHITECTURE:**
- ❌ **No Rate Limiting** - Removed for simplicity
- ❌ **No Redis Caching** - Not needed for MVP
- ❌ **No Model Tuning Parameters in Config** - Set temperature/params in methods, not config

---

## 🤖 AGENT 1: Infrastructure & Database Specialist

### Your Mission
Build the foundational infrastructure layer including database models, utilities, configuration management, and deployment scripts.

### Why You're Critical
Your work blocks all other agents. They need the database, config, logger, and rate limiter to function. **Start immediately and prioritize getting basics working.**

### Your Deliverables

#### 1. Database Layer (`src/database/`)

**`models.py`** - Create SQLAlchemy models for:
- `ResearchSession`: Tracks each investigation (id, target_name, session_id, timestamps, status, counts)
- `Fact`: Individual discovered facts (content, source_url, confidence_score, category, verification status)
- `Connection`: Relationships between entities (entity_a, entity_b, relationship_type, evidence, confidence)
- `RiskFlag`: Potential red flags (severity, category, description, evidence, confidence)
- `SearchQuery`: Track all searches (query, iteration, timestamp, results_count, relevance_score)

Requirements:
- Use proper relationships and foreign keys
- Add indexes on session_id and frequently queried fields
- Include created_at/updated_at timestamps
- Add __repr__ methods for debugging

**`repository.py`** - Implement repository pattern:
- Methods to create/get/update sessions
- Methods to save facts, connections, risks, searches
- Methods to retrieve all data for a session
- Use context managers for transactions
- Batch insert capabilities for performance

#### 2. Utilities (`src/utils/`)

**`config.py`** - Configuration management:
- Load from environment variables
- Validate required API keys (Google AI, SerpApi only)
- Database connection string
- Application settings (max iterations, research timeout)
- GCP settings (project ID, bucket)
- Provide a `from_env()` class method
- ✅ **COMPLETED BY AGENT 1**

**`logger.py`** - Structured logging:
- Setup logger function with configurable level
- Helper functions: log_agent_action, log_search, log_fact_extraction, log_error
- Use JSON format for structured logs
- Include timestamps and correlation IDs
- ✅ **COMPLETED BY AGENT 1**

#### 3. Infrastructure (`infra/`)

**`terraform/main.tf`** - GCP resources:
- Cloud SQL PostgreSQL (db-f1-micro for testing)
- Secret Manager for API keys
- Cloud Storage bucket for logs/reports
- Cloud Run service for deployment
- Service accounts with minimal IAM permissions
- Keep costs minimal for testing

**`docker/Dockerfile`** - Container setup:
- Base: Python 3.11-slim
- Multi-stage build for efficiency
- Install all dependencies
- Health check endpoint
- Run as non-root user
- Expose port 8000

#### 4. Project Files

**`requirements.txt`** - Include:
- LangGraph, LangChain packages
- AI model libraries (google-generativeai, anthropic)
- Database (sqlalchemy, psycopg2-binary, alembic)
- Search (google-search-results, playwright, beautifulsoup4)
- UI (chainlit)
- Utilities (pydantic, python-dotenv, tenacity)
- GCP packages (google-cloud-secret-manager, google-cloud-storage)
- ✅ **COMPLETED BY AGENT 1**

**`.env.example`** - Template for environment variables

### Testing Your Work
- Test database connection
- Create tables with Alembic
- Test repository CRUD operations
- Verify config loads from .env
- Check logger output format
- ✅ **ALL TESTS PASSED**

### Completion Checklist
- [x] All database models with relationships ✅
- [x] Repository pattern implemented and tested ✅
- [x] Config loader working with validation ✅
- [x] Logger producing structured output ✅
- [x] Terraform scripts validate successfully ✅
- [x] Dockerfile builds and runs ✅
- [x] Can create and query database records ✅

### Status: ✅ COMPLETED
Agent 1 infrastructure fully operational. Multi-model AI strategy (Claude + Gemini), simplified architecture (no rate limiting, no Redis).

---

## 🤖 AGENT 2: AI Models Integration Specialist

### Your Mission
Create robust, reusable clients for **Anthropic Claude and Google Gemini** models with proper error handling and retry logic.

### Why You're Critical
All agent nodes will use your model clients. Quality here = quality everywhere.

### ⚠️ ARCHITECTURE DECISIONS:
- **MULTI-MODEL STRATEGY**: Claude for complex reasoning, Gemini for extraction/speed
- **UNIFIED CLIENTS**: One client per model family (not per model variant)
- **NO RATE LIMITING** - Removed for simplicity
- **NO MODEL TUNING PARAMS IN CONFIG** - Use full model capabilities, set params in methods

### Your Deliverables

#### `src/tools/models/gemini_client.py`

**GeminiClient class (unified for all Gemini models):**
- Initialize with API key and model_name parameter
- Supports: `gemini-2.5-pro`, `gemini-2.5-flash`, and other Gemini variants
- `generate()` method: Basic text generation with retry logic
- `generate_structured()` method: Generate JSON matching a schema
- `extract_entities()` method: Named entity recognition (person, org, location, date, money)
- `extract_facts()` method: Extract structured facts from text with confidence scores
- `generate_search_queries()` method: Create 3-5 search queries based on current knowledge
- `filter_relevant_content()` method: Quick relevance filtering
- Handle API errors with exponential backoff (use tenacity)
- Parse JSON from markdown code blocks or raw JSON
- Log all API calls with request/response (use logger from Agent 1)
- Temperature control per method (not config)

**Use Cases**:
- Gemini Pro: Fact extraction, entity recognition, connection mapping
- Gemini Flash: Query generation, content filtering, preliminary analysis

#### `src/tools/models/anthropic_client.py`

**AnthropicClient class (unified for all Claude models):**
- Initialize with API key and model_name parameter
- Supports: `claude-sonnet-4-5-20250929` and other Claude variants
- `generate()` method: Basic text generation with retry logic
- `generate_structured()` method: Generate JSON matching a schema
- `analyze_risk()` method: Risk analysis with severity classification
- `generate_report()` method: Comprehensive report generation
- Handle API errors with exponential backoff (use tenacity)
- Parse JSON from markdown code blocks
- Log all API calls with token usage

**Use Cases**:
- Complex reasoning and analysis
- Risk assessment
- Report generation
- Validation and cross-referencing

#### `src/tools/models/__init__.py`

**ModelFactory class:**
- `create_client()` static method: Create appropriate client by model name
- `get_optimal_model_for_task()` static method: Return best model for task type
  - risk_analysis → Claude Sonnet
  - report_generation → Claude Sonnet
  - complex_reasoning → Claude Sonnet
  - extraction → Gemini Pro
  - entity_recognition → Gemini Pro
  - connection_mapping → Gemini Pro
  - query_generation → Gemini Flash
  - filtering → Gemini Flash
  - preliminary_analysis → Gemini Flash

### Error Handling Requirements
All clients must gracefully handle:
- **RateLimitError**: Wait and retry with exponential backoff (tenacity library)
- **AuthenticationError**: Log error and raise immediately (check API key)
- **APIError**: Retry if retryable (max 3 attempts), log and raise if not
- **TimeoutError**: Log and raise with context
- **ValidationError**: For JSON parsing failures
- Use `@retry` decorator from tenacity for automatic retries

### Configuration Integration
- Import config from `src.utils.config` (Agent 1 completed this)
- Get API keys: `config.ai_models.google_api_key` and `config.ai_models.anthropic_api_key`
- Get model names: `config.ai_models.gemini_pro_model`, `gemini_flash_model`, `claude_model`
- Use logger: `from src.utils.logger import get_logger`

### Testing Your Work
- Test each client with simple prompts
- Verify structured output parsing (JSON from markdown blocks)
- Test error handling (mock API errors if possible)
- Verify retry logic works (exponential backoff)
- Test factory pattern returns correct client
- Test integration with config and logger from Agent 1

### Completion Checklist
- [x] GeminiClient (unified) implemented with all methods
- [x] AnthropicClient implemented with all methods
- [x] Exponential backoff retry logic (tenacity)
- [x] Error handling for all API errors
- [x] Structured output parsing (JSON from markdown)
- [x] ModelFactory pattern functional
- [x] Integration with Agent 1 config working
- [x] Integration with Agent 1 logger working
- [x] Token usage logging for cost tracking
- [x] Ready for use by node agents

### Dependencies & Blockers
- **Blocked by**: Agent 1 ✅ (Config and Logger - COMPLETED)
- **Blocks**: Agents 4, 5, 6 (all nodes need these model clients)
- **Status**: ✅ COMPLETED

---

## 🤖 AGENT 3: Search & Data Collection Specialist

### Your Mission
Build robust search tools using SerpApi (Google Search API) as primary and Playwright as minimal fallback.

### Why You're Critical
Search quality directly impacts fact discovery rate. Your tools are the "eyes" of the research agent.

### Environment Setup
- **Python Environment**: Use the existing `.venv` in the project root
- **Package Installation**: If you need additional packages, use `source .venv/bin/activate && pip install <package>`
- **Configuration**: Import config from `src.utils.config` (Agent 1 completed this)
- **Logging**: Use logger from `src.utils.logger` (Agent 1 completed this)

### Your Deliverables

#### `src/tools/search/serp_api_search.py`

**SerpApiSearch class:**
- Initialize with API key from config (no rate limiter - simplified architecture)
- `search()` method: Execute Google search via SerpApi, return structured results
  - Parameters: query, max_results (default 10), search_type ("web"/"news"/"scholar")
  - Return SearchResult objects with: title, url, snippet, position, source_domain, date
  - Parse organic results, knowledge graph, people also ask
- `batch_search()` method: Execute multiple queries efficiently
- `_deduplicate_results()` method: Remove duplicate URLs and near-duplicate content
- `_calculate_source_diversity()` method: Score based on unique domains (0.0 to 1.0)
- Extract rich snippets and featured snippets when available
- Simple in-memory cache to avoid redundant API calls (dict-based, no Redis)
- Error handling with retry logic (use tenacity)

**Key Focus**: Speed, comprehensive results, source diversity

#### `src/tools/search/web_scraper.py`

**WebScraper class (minimal Playwright):**
- Browser lifecycle management (initialize, close)
- `scrape_url()` method: Scrape specific URL with timeout
  - Return ScrapedContent: url, title, clean text, html, links, metadata
- `_clean_text()` method: Extract clean text from HTML (remove scripts, styles)
- Handle CAPTCHAs gracefully (return empty if blocked)
- Add delays between requests (respect robots.txt)
- **Optional/Nice-to-have**: Only implement if time permits, SerpApi is primary

**Key Focus**: Use sparingly, clean up resources properly

#### `src/tools/search/__init__.py`

**SearchOrchestrator class:**
- Coordinate between SerpApi and optional Playwright scraper
- `search()` method: Primary strategy uses SerpApi
  1. Execute SerpApi search (fast, comprehensive)
  2. Optional: If Playwright is implemented and < 3 quality results, supplement with targeted scraping
- `deep_scrape_url()` method: For specific URL content extraction (optional)
- Resource cleanup with context manager if using Playwright
- Export SearchResult and ScrapedContent dataclasses

#### Data Models

Define dataclasses:
- `SearchResult`: title, url, content, score, source_domain, published_date
- `ScrapedContent`: url, title, text, html, links, metadata, scraped_at

### Testing Your Work
- Test SerpApi with various queries (use real API key)
- Verify result deduplication works
- Test source diversity calculation
- Test in-memory cache (verify same query doesn't hit API twice)
- Test error handling and retry logic
- Optional: Test Playwright scraper if implemented

### Completion Checklist
- [ ] SerpApi integration working with clean results
- [ ] Batch search implemented efficiently
- [ ] Deduplication tested (URL and content similarity)
- [ ] Source diversity calculation correct
- [ ] In-memory cache working (no Redis)
- [ ] Error handling with retry logic (tenacity)
- [ ] SearchOrchestrator returns SearchResult objects
- [ ] Integration with Agent 1 config and logger working
- [ ] Data models (SearchResult, ScrapedContent) defined
- [ ] All integration tests passing

### Dependencies & Blockers
- **Blocked by**: Agent 1 ✅ (Config and Logger - COMPLETED)
- **Blocks**: Agent 4 (Searcher Node needs this)
- **Required API Key**: SERPAPI_KEY in .env file

---

## 🤖 AGENT 4: Query Planning & Execution Nodes

### Your Mission
Implement the Query Planner and Search Executor nodes - the "brain" that decides what to search and when.

### Why You're Critical
Intelligent search progression is what makes this agent better than simple searches. You determine the discovery rate.

### Your Deliverables

#### `src/agents/nodes/planner.py`

**QueryPlannerNode class:**
- Uses Gemini 2.5 Flash for fast query generation
- `execute()` method: Main node function
  - Input: Current ResearchState
  - Output: Updated state with next_queries list (3-5 queries)
  
**Progressive Search Strategy:**
- **Iteration 1**: Broad discovery queries (professional background, company affiliations, news mentions)
- **Iterations 2-3**: Targeted investigation based on discovered facts (dig into specific companies, roles, locations)
- **Iterations 4-5**: Deep connection mining (board memberships, partnerships, associates)
- **Iterations 6-7**: Validation & gap filling (verify low-confidence facts, find additional sources)

**Key Methods:**
- `_generate_broad_queries()`: Initial discovery queries
- `_generate_targeted_queries()`: Build on findings, use discovered entities
- `_generate_connection_queries()`: Focus on relationships and networks
- `_generate_validation_queries()`: Verify low-confidence facts
- `_filter_duplicate_queries()`: Avoid repeating similar searches
- `_extract_entities()`: Pull key entities from facts
- `_extract_topics()`: Track explored topics

**Important**: Each query should be unique and build upon previous findings. Avoid redundant searches.

#### `src/agents/nodes/searcher.py`

**SearchExecutorNode class:**
- Uses SearchOrchestrator from Agent 3
- `execute()` method: Execute all queries in next_queries
  - For each query: execute search, collect results, record metrics
  - Deduplicate results across all queries
  - Calculate source diversity score
  - Update search_history
  - Increment current_iteration counter
  
**Key Features:**
- Parallel query execution (if possible)
- Comprehensive error handling per query
- Log each search with metrics
- Track relevance scores
- Return raw search results for next node to process

#### `src/prompts/templates/planner_prompt.py`

**Prompt templates for query generation:**
- Broad discovery prompt
- Targeted investigation prompt with context
- Connection mining prompt
- Validation prompt for low-confidence facts

**Guidelines for prompts:**
- Clear instructions for query format
- Context about what's already known
- Avoid repetition instructions
- Focus areas for each iteration type

### Testing Your Work
- Test query generation at each iteration level
- Verify queries are diverse and non-repetitive
- Test deduplication against search history
- Test search execution with mock queries
- Verify source diversity calculation
- Test full planner → searcher flow

### Completion Checklist
- [ ] QueryPlannerNode with progressive strategy
- [ ] All query generation methods implemented
- [ ] Duplicate filtering working correctly
- [ ] SearchExecutorNode executes queries
- [ ] Results properly deduplicated
- [ ] Source diversity tracked
- [ ] Iteration counter increments
- [ ] Prompts are well-engineered
- [ ] Integration tests passing

### Dependencies & Blockers
- **Blocked by**: Agent 2 (needs Gemini), Agent 3 (needs SearchOrchestrator)
- **Blocks**: Agent 7 (LangGraph needs all nodes)

---

## 🤖 AGENT 5: Extraction & Validation Nodes

### Your Mission
Implement the Content Extractor and Validator nodes that turn raw search results into verified, structured facts.

### Why You're Critical
You transform messy web content into clean, confident, usable intelligence. Quality here = accuracy everywhere.

### Your Deliverables

#### `src/agents/nodes/extractor.py`

**ContentExtractorNode class:**
- Uses Gemini Pro 2.5 for precise extraction
- `execute()` method: Extract facts from raw_search_results
  - Process each search result
  - Extract structured facts with metadata
  - Categorize facts (biographical, professional, financial, behavioral)
  - Assign preliminary confidence scores
  - Extract entities (people, organizations, locations, dates, amounts)
  
**Fact Structure:**
- content: The actual fact (clean, concise statement)
- source_url: Where it came from
- source_domain: Domain for reliability scoring
- extracted_date: When extracted
- confidence: Preliminary score (0.0 to 1.0)
- category: Type of fact
- entities: Extracted named entities

**Key Features:**
- Batch processing for efficiency
- Handle extraction failures gracefully
- Filter out low-quality or irrelevant content
- Normalize dates and amounts
- Extract direct quotes when valuable

#### `src/agents/nodes/validator.py`

**ValidatorNode class:**
- Uses Claude Sonnet for complex reasoning
- `execute()` method: Validate and cross-reference facts
  - Group facts by topic/claim
  - Check for corroboration across multiple sources
  - Detect contradictions
  - Adjust confidence scores based on:
    - Source diversity (more sources = higher confidence)
    - Source authority (gov sites, academic > blogs)
    - Consistency across sources
    - Recency of information
  
**Confidence Scoring Algorithm:**
- Base score from extraction
- +0.2 if multiple independent sources
- +0.1 if authoritative domain (.gov, .edu, major news)
- +0.1 if recent information (< 6 months)
- -0.3 if contradictions found
- -0.1 if single source only

**Key Methods:**
- `_cross_reference_facts()`: Find corroborating evidence
- `_detect_contradictions()`: Flag conflicting facts
- `_score_source_authority()`: Rate domain reliability
- `_calculate_final_confidence()`: Combine all factors

#### `src/prompts/templates/extractor_prompt.py`

**Extraction prompts:**
- System prompt for fact extraction role
- Guidelines for clean, atomic facts
- Output format specification (JSON)
- Examples of good vs bad extractions

#### `src/prompts/templates/validator_prompt.py`

**Validation prompts:**
- Cross-referencing instructions
- Contradiction detection guidance
- Confidence scoring criteria
- Output format for validation results

### Testing Your Work
- Test extraction with various content types
- Verify fact quality and categorization
- Test entity extraction accuracy
- Test validation with corroborating sources
- Test contradiction detection
- Verify confidence scoring logic
- Test with low-quality sources

### Completion Checklist
- [ ] ContentExtractorNode extracts structured facts
- [ ] Facts properly categorized
- [ ] Entity extraction working
- [ ] ValidatorNode cross-references facts
- [ ] Confidence scoring implemented
- [ ] Contradiction detection functional
- [ ] Source authority scoring
- [ ] Prompts well-engineered for accuracy
- [ ] Unit tests for scoring logic
- [ ] Integration tests passing

### Dependencies & Blockers
- **Blocked by**: Agent 2 (needs Gemini Pro and Claude clients), Agent 4 (needs search results)
- **Blocks**: Agent 7 (LangGraph needs all nodes)
- **Important**: Use Gemini Pro 2.5 for extraction (NOT GPT-4o)

---

## 🤖 AGENT 6: Analysis & Reporting Nodes

### Your Mission
Implement Risk Analyzer, Connection Mapper, and Report Generator nodes that synthesize findings into actionable intelligence.

### Why You're Critical
You're the final layer that turns data into insights. Your output is what the user sees and judges quality by.

### Your Deliverables

#### `src/agents/nodes/risk_analyzer.py`

**RiskAnalyzerNode class:**
- Uses Claude Sonnet for complex risk assessment
- `execute()` method: Analyze facts for potential red flags
  
**Risk Categories to Detect:**
- **Legal**: Lawsuits, regulatory actions, criminal charges, ongoing litigation
- **Financial**: Bankruptcy, fraud allegations, payment defaults, suspicious transactions
- **Reputational**: Scandals, controversies, negative media coverage, ethical violations
- **Compliance**: Regulatory violations, license revocations, sanctions
- **Behavioral**: Pattern of job-hopping, unexplained gaps, inconsistent statements

**For Each Risk:**
- Severity level: Low / Medium / High / Critical
- Category
- Description (clear, factual)
- Evidence (list of supporting facts with sources)
- Confidence score
- Recommended follow-up actions

**Key Features:**
- Pattern matching across facts
- Timeline analysis (when did issues occur?)
- Frequency analysis (recurring problems?)
- Context consideration (industry norms)

#### `src/agents/nodes/connection_mapper.py`

**ConnectionMapperNode class:**
- Uses Gemini Pro 2.5 for entity relationship extraction
- `execute()` method: Map relationships between entities
  
**Connection Types:**
- Employment (worked together at Company X)
- Investment (investor in Company Y)
- Board/Advisory (both on Board of Z)
- Family/Personal relationships
- Co-founder/Business partner
- Educational (attended same institution)

**For Each Connection:**
- entity_a and entity_b
- relationship_type
- evidence (supporting facts)
- confidence score
- time_period (when active)

**Key Features:**
- Build connection graph (nodes = entities, edges = relationships)
- Detect indirect connections (A → B → C)
- Identify central figures in network
- Timeline relationships
- Export graph data (for visualization)

#### `src/agents/nodes/reporter.py`

**ReportGeneratorNode class:**
- Uses Claude Sonnet for synthesis and writing
- `execute()` method: Generate comprehensive markdown report
  
**Report Structure:**
- **Executive Summary**: Key findings in 3-4 sentences
- **Subject Overview**: Basic biographical/professional info
- **Key Facts**: Organized by category, with confidence indicators
- **Risk Assessment**: All identified risks with severity and evidence
- **Network Analysis**: Important connections and relationships
- **Timeline**: Chronological view of major events
- **Source Summary**: Domains used, source diversity metrics
- **Confidence Assessment**: Overall reliability of findings
- **Recommendations**: Suggested follow-up investigations

**Formatting Guidelines:**
- Use markdown for structure
- Bold key terms and names
- Tables for structured data
- Confidence indicators (🟢 High, 🟡 Medium, 🔴 Low)
- Source citations in footnotes
- Professional, objective tone

#### Prompt Templates (`src/prompts/templates/`)

**`risk_analyzer_prompt.py`**: Risk detection instructions, categories, severity criteria
**`connection_mapper_prompt.py`**: Relationship extraction rules, connection types
**`reporter_prompt.py`**: Report structure, formatting guidelines, tone guidance

### Testing Your Work
- Test risk detection with known problematic profiles
- Verify risk severity classification
- Test connection mapping with multi-person data
- Test report generation with complete dataset
- Verify markdown formatting
- Check report readability and completeness

### Completion Checklist
- [ ] RiskAnalyzerNode detects all risk categories
- [ ] Risk severity properly classified
- [ ] Evidence clearly linked to risks
- [ ] ConnectionMapperNode extracts relationships
- [ ] Connection graph exported
- [ ] ReportGeneratorNode produces complete reports
- [ ] Report formatting professional and clear
- [ ] All three prompts well-engineered
- [ ] Integration tests with full data
- [ ] Reports reviewed for quality

### Dependencies & Blockers
- **Blocked by**: Agent 2 (needs Claude and Gemini Pro clients), Agent 5 (needs validated facts)
- **Blocks**: Agent 7 (LangGraph needs all nodes)
- **Important**: Use Gemini Pro 2.5 for connection mapping (NOT GPT-4o)

---

## 🤖 AGENT 7: LangGraph Orchestration Specialist

### Your Mission
Tie everything together into a working LangGraph workflow that orchestrates all nodes intelligently.

### Why You're Critical
You're the conductor of the orchestra. All the pieces exist, but you make them play together harmoniously.

### Your Deliverables

#### `src/agents/state.py`

**Define ResearchState TypedDict:**
- All state fields from CLAUDE.md
- Input fields: target_name, research_depth
- Progressive fields: collected_facts, connections, risk_flags, search_history
- Control fields: current_iteration, next_queries, explored_topics
- Metadata: session_id, start_time, timestamps
- Output fields: final_report, connection_graph

**Use proper types:**
- List[Dict] for collections
- Set[str] for explored_topics
- Optional[str] for nullable fields
- datetime for timestamps

#### `src/agents/graph.py`

**Build the LangGraph workflow:**

**Node Sequence:**
1. START
2. QueryPlannerNode
3. SearchExecutorNode
4. ContentExtractorNode
5. ValidatorNode
6. RiskAnalyzerNode
7. ConnectionMapperNode
8. Conditional: Continue or Report?
   - If current_iteration < max_iterations AND new facts found → Back to QueryPlanner
   - Else → ReportGeneratorNode
9. ReportGeneratorNode
10. END

**Key Features:**
- State persistence (save to database after each node)
- Conditional edges based on iteration count and fact discovery
- Error handling (if node fails, log and continue)
- Progress tracking
- Timeout protection (max execution time)

**Methods to Implement:**
- `create_graph()`: Build and compile the StateGraph
- `run_research()`: Execute research for a target
- `_should_continue()`: Conditional logic for iterations
- `_save_checkpoint()`: Persist state to database

**Stopping Conditions:**
- Max iterations reached (default 7)
- No new facts discovered in last iteration
- Target fact count reached (optional)
- Manual stop signal

#### Visualization

- Export graph visualization to PNG/SVG
- Show node connections and conditional edges
- Include in documentation

### Testing Your Work
- Test with simple target (known public figure)
- Verify all nodes execute in sequence
- Test conditional logic (does it loop correctly?)
- Test stopping conditions
- Verify state persistence
- Test error handling (mock node failures)
- Full end-to-end test with test persona

### Completion Checklist
- [x] State schema fully defined with all fields ✅
- [x] LangGraph workflow built with all nodes ✅
- [x] Conditional edges implemented correctly ✅
- [x] Iteration logic working (loops properly) ✅
- [x] Stopping conditions functional ✅
- [x] State persistence after each node ✅
- [x] Error handling for node failures ✅
- [ ] Graph visualization exported (optional)
- [x] Integration tests created (2/3 passing) ✅
- [x] Documentation of workflow ✅

### Dependencies & Blockers
- **Blocked by**: Agents 4, 5, 6 (needs ALL nodes implemented) - ✅ ALL COMPLETE
- **Blocks**: Agent 8 (UI needs working graph) - ✅ NOW UNBLOCKED
- **Status**: ✅ COMPLETE

### Additional Work Done
- Added missing repository methods: `update_session_checkpoint()`, `complete_session()`, `fail_session()`
- Fixed missing environment variables in `.env`
- Created comprehensive integration tests
- All unit tests passing, 2/3 integration tests passing (1 requires DB setup)

---

## 🤖 AGENT 8: UI & Evaluation Specialist

### Your Mission
Build the Chainlit UI and create the evaluation framework with test personas.

### Why You're Critical
You make the system usable and measurable. The UI is what impresses stakeholders; the evaluation proves it works.

### Your Deliverables

#### `src/ui/chainlit_app.py`

**Chainlit Interface:**

**Features to Implement:**
- Welcome message with instructions
- Input form: target name, research depth (1-7)
- Start research button
- Real-time progress display:
  - Current iteration
  - Current node executing
  - Facts discovered so far
  - Searches executed
  - Progress bar
- Stream interim updates as research progresses
- Display final report in nice markdown format
- Download button for full report
- Show connection graph visualization
- Session history (past researches)

**Key Methods:**
- `on_chat_start()`: Initialize session
- `start_research()`: Trigger research workflow
- `stream_updates()`: Show progress in real-time
- `display_report()`: Render final markdown report
- `show_metrics()`: Display evaluation metrics

**UI Polish:**
- Loading indicators
- Error messages user-friendly
- Color coding for risk levels
- Collapsible sections in report
- Copy to clipboard functionality

#### `tests/evaluation/personas.json`

**Create 3 Test Personas:**

**Persona 1: "The Clean Executive"** (Easy/Medium)
- Real public figure with good reputation
- 15-20 hidden facts to discover
- Categories: education, career progression, board positions, publications
- Expected discovery rate: 80%+

**Persona 2: "The Controversial Entrepreneur"** (Medium/Hard)
- Individual with public controversies
- 15-20 hidden facts including risks
- Categories: failed ventures, lawsuits, connections, financial issues
- Expected discovery rate: 70%+

**Persona 3: "The Low-Profile Investor"** (Hard)
- Private individual, minimal online presence
- 15-20 obscure hidden facts
- Categories: portfolio companies, indirect connections, family background
- Expected discovery rate: 60%+

**Ground Truth Format for Each:**
```json
{
  "name": "...",
  "difficulty": "easy|medium|hard",
  "hidden_facts": [
    {
      "fact": "...",
      "category": "...",
      "difficulty": "easy|medium|hard",
      "expected_sources": ["domain1.com", "domain2.com"]
    }
  ],
  "expected_risks": [...],
  "expected_connections": [...]
}
```

#### `tests/evaluation/test_agent.py`

**Evaluation Runner:**
- Load test personas
- Run research for each
- Compare discovered facts to ground truth
- Calculate metrics
- Generate evaluation report

#### `tests/evaluation/metrics.py`

**Metrics Calculation:**
- **Fact Discovery Rate**: % of hidden facts found
- **Precision**: % of discovered facts that are correct
- **Source Diversity**: # unique domains / total sources
- **Confidence Calibration**: Correlation between confidence scores and accuracy
- **Query Efficiency**: Avg facts discovered per search query
- **Execution Time**: Total duration
- **Risk Detection Rate**: % of expected risks flagged
- **Connection Accuracy**: % of expected connections found

**Generate Comparison Report:**
- Per-persona breakdown
- Overall metrics
- Identify gaps (what was missed?)
- Recommendations for improvement

### Testing Your Work
- Test UI with simple input
- Verify real-time updates display correctly
- Test with all 3 personas
- Calculate all metrics
- Generate evaluation report
- Verify report formatting in UI

### Completion Checklist
- [ ] Chainlit UI functional and user-friendly
- [ ] Real-time progress updates working
- [ ] Report displays beautifully
- [ ] 3 test personas created with ground truth
- [ ] Evaluation framework complete
- [ ] All metrics calculated correctly
- [ ] Evaluation report generated
- [ ] Metrics meet target thresholds (70%+ discovery)
- [ ] UI tested with various inputs
- [ ] Documentation for using UI

### Dependencies & Blockers
- **Blocked by**: Agent 7 (needs working LangGraph)
- **Blocks**: None (final layer)

---

## 📋 General Guidelines for All Agents

### Code Quality Standards
- **Type hints**: Use for all function parameters and returns
- **Docstrings**: Document all classes and public methods (Google style)
- **Error handling**: Specific exceptions with meaningful messages
- **Logging**: Use logger for INFO level operations and all errors
- **Testing**: Unit tests for critical logic, integration tests for workflows

### Communication Protocol
1. **Before starting**: Check PROJECT_COORDINATION.md for blockers
2. **While working**: Update your section in PROJECT_COORDINATION.md
3. **When blocked**: Mark blocker immediately with reason
4. **When complete**: Update completion checklist and mark deliverables done
5. **Document decisions**: Add notes for any architectural changes

### Git Workflow
- Branch naming: `agent-{N}-{feature-name}`
- Commit messages: Clear, descriptive, imperative mood
- PR description: What changed, why, any concerns
- Review: At least one other agent reviews before merge

### Daily Workflow
1. Read PROJECT_COORDINATION.md to see current status
2. Read dependencies section for your agent
3. Work on your assigned files only
4. Test your work thoroughly
5. Update PROJECT_COORDINATION.md
6. Create PR or push to your branch
7. Document any blockers or issues

### When You're Stuck
- Check CLAUDE.md for architectural guidance
- Review reference documentation (LangGraph, API docs)
- Ask for clarification on specific technical decisions
- Document what you tried and what didn't work

---

## 🎯 Success Criteria

**Project is complete when:**
- [ ] All 8 agents finished their deliverables
- [ ] End-to-end workflow runs successfully
- [ ] 3 test personas evaluated
- [ ] Fact discovery rate > 70%
- [ ] Precision > 90%
- [ ] Chainlit UI functional and polished
- [ ] Documentation complete
- [ ] Deployed to GCP (or ready to deploy)
- [ ] Demo-ready

**Your individual agent is complete when:**
- [ ] All deliverables checked off
- [ ] Tests passing
- [ ] Code reviewed
- [ ] Documented
- [ ] PROJECT_COORDINATION.md updated

---

**Remember**: Quality over speed. A working, well-tested system is better than rushing to complete. Focus on your assigned components and trust other agents to handle theirs.