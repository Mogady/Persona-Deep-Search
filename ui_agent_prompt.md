# 🎨 AGENT 8: UI Specialist - Chainlit Interface

## 📋 Your Mission

Build a **high-quality, real-time Chainlit UI** that provides complete visibility into the research workflow. Users should see **every internal step** of the 7-node sequence in a beautiful, intuitive interface.

---

## 🎯 Critical Success Criteria

1. **Real-time Progress Visibility**: Users see what's happening at each node as it happens
2. **Complete Transparency**: All internal outputs (queries, facts, risks, connections) displayed
3. **User Control**: Key quality configuration exposed (NOT performance configs)
4. **Beautiful Design**: Professional, clean, easy to navigate
5. **Responsive**: Fast updates, smooth transitions, no lag

---

## 🏗️ System Architecture You're Building On

### LangGraph Workflow (7 Nodes - Sequential with Loop)
```
START → Planner → Searcher → Extractor → Validator → Risk Analyzer → Connection Mapper → [Decision: Continue/Report] → Reporter → END
                    ↑                                                                              |
                    └──────────────────────────────────────────────────────────────────────────────┘
                                            (Loop back if more iterations needed)
```

### State Schema (ResearchState TypedDict)
```python
{
    # Input
    "target_name": str,
    "research_depth": int,  # Max iterations (1-7)
    "session_id": str,

    # Progressive Knowledge Building
    "collected_facts": List[Fact],  # All facts discovered
    "new_facts": List[Fact],        # Facts from current iteration only
    "connections": List[Connection],
    "risk_flags": List[RiskFlag],
    "search_history": List[SearchQuery],

    # Control Flow
    "current_iteration": int,
    "next_queries": List[str],
    "explored_topics": Set[str],
    "raw_search_results": List[Dict],
    "facts_before_iteration": int,

    # Metadata
    "start_time": datetime,
    "last_update": datetime,

    # Output
    "final_report": str,  # Markdown formatted
    "connection_graph": Dict  # JSON graph structure
}
```

### Data Types (TypedDict)

**Fact:**
```python
{
    "content": str,
    "source_url": str,
    "source_domain": str,
    "extracted_date": datetime,
    "confidence": float,  # 0.0 to 1.0
    "category": str,  # biographical, professional, financial, behavioral
    "entities": Dict[str, List[str]]
}
```

**Connection:**
```python
{
    "entity_a": str,
    "entity_b": str,
    "relationship_type": str,  # "employment", "investment", "board_member", etc.
    "evidence": List[str],
    "confidence": float,
    "time_period": str
}
```

**RiskFlag:**
```python
{
    "severity": str,  # "low", "medium", "high", "critical"
    "category": str,  # "legal", "financial", "reputational", "compliance", "operational"
    "description": str,
    "evidence": List[str],
    "confidence": float,
    "recommended_follow_up": str
}
```

**SearchQuery:**
```python
{
    "query": str,
    "timestamp": datetime,
    "results_count": int,
    "relevance_score": float
}
```

### Database Models (PostgreSQL)

**All data is persisted to database:**
- `ResearchSession`: Main session metadata
- `Fact`: Individual facts with source attribution
- `Connection`: Entity relationships
- `RiskFlag`: Identified risks
- `SearchQuery`: Query audit trail

**Repository Methods Available:**
```python
repository.create_session(session_id, target_name, research_depth)
repository.update_session_checkpoint(session_id, node_name, state)
repository.complete_session(session_id)
repository.get_session(session_id)
repository.get_all_facts(session_id)
repository.get_all_connections(session_id)
repository.get_all_risk_flags(session_id)
repository.get_all_search_queries(session_id)
```

---

## 🎨 UI Requirements

### 1. Welcome Screen

**Elements:**
- Welcoming header: "Deep Research AI Agent"
- Subtitle: "Autonomous investigation powered by LangGraph + Multi-Model AI"
- Brief description of what it does

**Input Form:**
```
Target Name: [________________]  (text input, required)

Research Depth: [Slider: 1 ─────●─── 7]  (default: 7)
  ℹ️ "Number of research iterations (more = deeper but slower)"

──── Quality Settings ────

Fact Extraction Temperature: [Slider: 0.0 ─●───── 1.0]  (default: 0.3)
  ℹ️ "Lower = more factual, Higher = more creative"

Validation Temperature: [Slider: 0.0 ●───── 1.0]  (default: 0.2)
  ℹ️ "Confidence scoring sensitivity"

Risk Analysis Temperature: [Slider: 0.0 ──●──── 1.0]  (default: 0.3)
  ℹ️ "Risk detection sensitivity"

Report Generation Temperature: [Slider: 0.0 ───●─── 1.0]  (default: 0.4)
  ℹ️ "Report creativity vs. precision"

[Start Research] button
```

**Notes:**
- Only expose **quality-affecting configs** (temperatures), NOT performance configs
- Use tooltips/help text to explain each parameter
- Validate input (target name required, valid ranges)
- Default values should be production-ready

---

### 2. Research Progress View (Real-time Updates)

**Layout:**
```
┌─────────────────────────────────────────────────────────────────┐
│  Researching: [Target Name]                      Session: [UUID] │
│  Progress: Iteration 3/7                              [Stop] btn │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [███████████████████░░░░░░░░] 65% Complete                     │
│                                                                   │
│  Current Step: ✅ Planner → ✅ Searcher → 🔄 Extractor          │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  📊 LIVE STATS                                                   │
│  ├─ Facts Discovered: 24                                         │
│  ├─ Risk Flags: 2 (1 Medium, 1 Low)                            │
│  ├─ Connections: 5                                               │
│  ├─ Searches Executed: 15                                        │
│  └─ Elapsed Time: 2m 34s                                        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Node-by-Node Updates (Accordion/Collapsible Sections):**

#### **Iteration 1 ▼**

**📝 1. Query Planner**
```
Status: ✅ Complete (1.2s)

Generated Queries:
  1. "John Doe professional background career history"
  2. "John Doe company affiliations board positions"
  3. "John Doe news mentions media coverage"
  4. "John Doe education university degree"
  5. "John Doe LinkedIn profile"
```

**🔍 2. Search Executor**
```
Status: ✅ Complete (3.4s)

Executed 5 queries:
  ✅ Query 1: 10 results (relevance: 0.85)
  ✅ Query 2: 10 results (relevance: 0.78)
  ✅ Query 3: 8 results (relevance: 0.92)
  ✅ Query 4: 7 results (relevance: 0.71)
  ✅ Query 5: 10 results (relevance: 0.88)

Total Results: 45
Unique Domains: 12 (forbes.com, linkedin.com, bloomberg.com, ...)
Source Diversity: 0.83
```

**📦 3. Content Extractor**
```
Status: ✅ Complete (5.1s)

Extracted Facts: 8 new facts

Examples:
  📌 "John Doe is CEO of Acme Corp since 2019" (Confidence: 0.95)
     Source: forbes.com | Category: Professional

  📌 "Graduated from MIT with Computer Science degree in 2005" (Confidence: 0.92)
     Source: linkedin.com | Category: Educational

  📌 "Serves on board of directors at Tech Foundation" (Confidence: 0.88)
     Source: bloomberg.com | Category: Professional

[Show All 8 Facts ▼]
```

**✓ 4. Validator**
```
Status: ✅ Complete (4.2s)

Validated 8 facts:
  ✅ 6 facts confirmed (multiple sources)
  ⚠️  2 facts flagged (single source, confidence adjusted)

Confidence Adjustments:
  ↗ +0.2 for "CEO of Acme Corp" (3 sources corroborate)
  ↗ +0.1 for "MIT graduate" (authoritative source: mit.edu)
  ↘ -0.1 for "Tech Foundation board" (single source)

Cross-references Found: 4
Contradictions Detected: 0
```

**🚨 5. Risk Analyzer**
```
Status: ✅ Complete (3.8s)

Risks Identified: 0 new risks

No significant risks detected in current facts.
```

**🔗 6. Connection Mapper**
```
Status: ✅ Complete (2.9s)

Connections Found: 2 new connections

  🔗 John Doe → Acme Corp
     Type: Employment (CEO)
     Confidence: 0.95
     Period: 2019-Present

  🔗 John Doe → Tech Foundation
     Type: Board Member
     Confidence: 0.88
     Period: 2020-Present

[View Connection Graph ▼]
```

**🔄 Decision: Continue to Iteration 2** (35 iterations remaining, 8 new facts found)

---

#### **Iteration 2 ▼**
... (Repeat same structure)

---

**Real-time Update Strategy:**
- Use Chainlit's `cl.Message` streaming to show node progress
- Update stats live as each node completes
- Highlight current active node with spinning icon (🔄)
- Mark completed nodes with checkmark (✅)
- Use `cl.Step` context manager for each node

---

### 3. Final Report Display

**Layout:**
```
┌─────────────────────────────────────────────────────────────────┐
│  ✅ Research Complete!                                           │
│  Target: [Name] | Duration: 5m 23s | Session: [UUID]            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  📊 SUMMARY STATS                                                │
│  ├─ Facts Discovered: 42 (Avg Confidence: 0.87)                │
│  ├─ Risk Flags: 3 (1 High, 2 Medium)                           │
│  ├─ Connections: 12                                              │
│  ├─ Iterations: 5/7 (stopped early - no new facts)             │
│  └─ Source Diversity: 18 unique domains                         │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Download Report (MD)] [Download Data (JSON)] [Share]          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

──────────────── FINAL REPORT ────────────────

[Display the markdown report generated by Reporter node]
  - Executive Summary
  - Subject Overview
  - Key Facts (organized by category)
  - Risk Assessment
  - Network Analysis
  - Timeline
  - Source Summary
  - Recommendations

──────────────── DETAILED DATA ────────────────

📋 All Facts (42) ▼
  [Filterable table with columns: Content, Category, Confidence, Source, Date]
  [Sortable by confidence, category, date]

🚨 Risk Flags (3) ▼
  [Severity badges: 🔴 High, 🟡 Medium]
  [Expandable details with evidence]

🔗 Connections (12) ▼
  [Interactive connection graph visualization]
  [List view with evidence]

🔍 Search History (32 queries) ▼
  [Timeline of all queries with results count]
```

**Report Rendering:**
- Use `cl.Message` with markdown formatting
- Color-code risk severity (🔴🟡🟢)
- Make facts expandable (show source on click)
- Interactive connection graph using a simple visualization library
- Download buttons export full data

---

### 4. Session History (Optional but Nice)

```
Previous Research Sessions:

┌─────────────────────────────────────────────────────┐
│  Jane Smith                              2025-10-02  │
│  Facts: 38 | Risks: 1 | Duration: 4m 12s            │
│  [View Results]                                      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Tech Corp Inc                           2025-10-01  │
│  Facts: 52 | Risks: 4 | Duration: 6m 45s            │
│  [View Results]                                      │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Technical Implementation

### File Structure
```
src/ui/
├── chainlit_app.py           # Main Chainlit app
├── components/
│   ├── progress_display.py   # Real-time progress components
│   ├── report_renderer.py    # Final report rendering
│   └── config_form.py         # Input form components
└── utils/
    └── formatters.py          # Helper functions for formatting
```

### Key Chainlit APIs to Use

```python
import chainlit as cl
from src.agents.graph import ResearchWorkflow
from src.database.repository import ResearchRepository
from src.utils.config import Config

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    await cl.Message(content="Welcome to Deep Research AI Agent!").send()
    # Show input form

@cl.on_message
async def main(message: cl.Message):
    """Handle user input and start research"""
    # Parse input
    # Start research workflow
    # Stream updates

async def run_research_with_updates(target_name, research_depth, config_overrides):
    """
    Execute research workflow with real-time UI updates.

    Strategy:
    1. Create ResearchWorkflow instance
    2. Hook into each node execution
    3. Stream updates using cl.Step and cl.Message
    4. Update progress indicators
    """

    # Create progress message
    progress_msg = cl.Message(content="Starting research...")
    await progress_msg.send()

    # Initialize workflow
    config = Config.from_env()

    # Override quality settings from user input
    config.performance.fact_extraction_temperature = config_overrides.get("fact_extraction_temp", 0.3)
    config.performance.validation_temperature = config_overrides.get("validation_temp", 0.2)
    config.performance.risk_analysis_temperature = config_overrides.get("risk_temp", 0.3)
    config.performance.report_generation_temperature = config_overrides.get("report_temp", 0.4)

    repository = ResearchRepository(config.database.database_url)
    workflow = ResearchWorkflow(repository, config)

    # Run workflow and stream updates
    # (You'll need to modify the workflow to support callbacks or use a wrapper)

    # Option 1: Polling approach (simpler)
    session_id = "..."
    state = workflow.run_research(target_name, research_depth, session_id)

    # Periodically check database for updates
    while not_complete:
        checkpoint = repository.get_session_checkpoint(session_id)
        await update_ui(checkpoint)
        await asyncio.sleep(1)

    # Option 2: Callback approach (better)
    # Modify ResearchWorkflow to accept callback function
    # Call callback after each node with updated state

    # Display final results
    await display_final_report(state)
```

### Streaming Updates with Chainlit Steps

```python
async def display_node_execution(node_name, state):
    """Display real-time node execution"""

    async with cl.Step(name=node_name) as step:
        if node_name == "Planner":
            queries = state.get("next_queries", [])
            step.output = f"Generated {len(queries)} queries:\n" + "\n".join(f"  {i+1}. {q}" for i, q in enumerate(queries))

        elif node_name == "Searcher":
            search_history = state.get("search_history", [])
            results_count = sum(q.get("results_count", 0) for q in search_history[-5:])
            step.output = f"Executed queries, found {results_count} results"

        elif node_name == "Extractor":
            new_facts = state.get("new_facts", [])
            step.output = f"Extracted {len(new_facts)} new facts"

            # Show examples
            if new_facts[:3]:
                examples = "\n".join(
                    f"  📌 {fact['content'][:80]}... (Confidence: {fact['confidence']:.2f})"
                    for fact in new_facts[:3]
                )
                step.output += f"\n\nExamples:\n{examples}"

        elif node_name == "Validator":
            facts = state.get("collected_facts", [])
            avg_confidence = sum(f['confidence'] for f in facts) / len(facts) if facts else 0
            step.output = f"Validated {len(facts)} facts (Avg confidence: {avg_confidence:.2f})"

        elif node_name == "Risk Analyzer":
            risks = state.get("risk_flags", [])
            step.output = f"Identified {len(risks)} risk flags"

            # Show severity breakdown
            severity_counts = {}
            for risk in risks:
                sev = risk.get("severity", "unknown")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            if severity_counts:
                breakdown = ", ".join(f"{count} {sev}" for sev, count in severity_counts.items())
                step.output += f" ({breakdown})"

        elif node_name == "Connection Mapper":
            connections = state.get("connections", [])
            step.output = f"Mapped {len(connections)} connections"

        elif node_name == "Reporter":
            step.output = "Generated comprehensive report"
```

### Configuration Mapping

**User Input → Config Override:**
```python
config_overrides = {
    "research_depth": user_input["research_depth"],  # 1-7
    "fact_extraction_temp": user_input["fact_extraction_temperature"],  # 0.0-1.0
    "validation_temp": user_input["validation_temperature"],  # 0.0-1.0
    "risk_temp": user_input["risk_analysis_temperature"],  # 0.0-1.0
    "report_temp": user_input["report_generation_temperature"],  # 0.0-1.0
}

# Apply to config object
config.application.max_search_iterations = config_overrides["research_depth"]
config.performance.fact_extraction_temperature = config_overrides["fact_extraction_temp"]
config.performance.validation_temperature = config_overrides["validation_temp"]
config.performance.risk_analysis_temperature = config_overrides["risk_temp"]
config.performance.report_generation_temperature = config_overrides["report_temp"]
```

**DO NOT expose these (performance-only):**
- `max_concurrent_llm_calls`
- `max_concurrent_search_calls`
- `extraction_batch_size`
- `api_request_timeout`
- `api_retry_attempts`
- Query generation temperature (always use default)
- Categorization temperature (always use default)
- Connection mapping temperature (always use default)

---

## 🎯 Key Implementation Steps

1. **Setup Chainlit App Structure**
   - Create `src/ui/chainlit_app.py`
   - Configure Chainlit settings (`.chainlit` folder)
   - Set up imports and dependencies

2. **Build Welcome Screen**
   - Input form with validation
   - Quality settings sliders
   - Start button handler

3. **Implement Research Execution Wrapper**
   - Wrapper around `ResearchWorkflow.run_research()`
   - Real-time progress updates
   - Node-by-node streaming

4. **Create Progress Display Components**
   - Overall progress bar
   - Live stats counter
   - Node execution steps
   - Iteration accordion

5. **Build Final Report Display**
   - Markdown rendering
   - Interactive data tables
   - Connection graph visualization
   - Download buttons

6. **Add Error Handling**
   - Graceful failures
   - User-friendly error messages
   - Retry options

7. **Polish & UX Improvements**
   - Loading indicators
   - Smooth transitions
   - Responsive design
   - Accessibility

---

## 📊 Data Flow

```
User Input (Form)
    ↓
Config Override
    ↓
ResearchWorkflow.run_research()
    ↓
[Node 1: Planner] → Stream Update → UI displays queries
    ↓
[Node 2: Searcher] → Stream Update → UI displays search results
    ↓
[Node 3: Extractor] → Stream Update → UI displays new facts
    ↓
[Node 4: Validator] → Stream Update → UI displays validation
    ↓
[Node 5: Risk Analyzer] → Stream Update → UI displays risks
    ↓
[Node 6: Connection Mapper] → Stream Update → UI displays connections
    ↓
[Decision: Continue?] → UI shows iteration complete, starting next
    ↓ (loop or continue)
[Node 7: Reporter] → Stream Update → UI displays final report
    ↓
Final Display (Report + Data + Downloads)
```

---

## ✅ Completion Checklist

- [ ] Chainlit app structure created
- [ ] Welcome screen with input form implemented
- [ ] Quality configuration sliders working
- [ ] Research workflow integration complete
- [ ] Real-time progress updates streaming
- [ ] All 7 nodes display their outputs
- [ ] Iteration looping visible in UI
- [ ] Final report rendering beautifully
- [ ] Facts table filterable and sortable
- [ ] Risk flags color-coded by severity
- [ ] Connection graph visualized
- [ ] Download buttons functional (MD + JSON)
- [ ] Error handling graceful
- [ ] Loading states smooth
- [ ] UI tested with real research session
- [ ] Documentation added

---

## 🎨 Design Principles

1. **Transparency**: Every internal step visible
2. **Clarity**: Complex data presented simply
3. **Responsiveness**: Fast updates, no blocking
4. **Beauty**: Professional, modern design
5. **Usability**: Intuitive, minimal learning curve
6. **Reliability**: Graceful error handling

---

## 🚀 Getting Started

1. **Read this entire prompt carefully**
2. **Review the existing code:**
   - `src/agents/graph.py` - Workflow orchestration
   - `src/agents/state.py` - State schema
   - `src/database/models.py` - Database models
   - `src/utils/config.py` - Configuration
3. **Create `src/ui/chainlit_app.py`**
4. **Start with welcome screen**
5. **Implement research execution with streaming**
6. **Build progress display components**
7. **Add final report rendering**
8. **Test end-to-end**
9. **Polish and refine**

---

## 📚 References

- **Chainlit Documentation**: https://docs.chainlit.io/
- **LangGraph State Management**: https://langchain-ai.github.io/langgraph/
- **Existing Codebase**: `src/agents/`, `src/database/`, `src/utils/`

---

## 💡 Pro Tips

1. **Use Chainlit Steps for each node** - `async with cl.Step(name="Planner") as step:`
2. **Stream updates frequently** - Users love seeing progress
3. **Make data interactive** - Expandable sections, filterable tables
4. **Test with real API calls** - Don't use mocked data
5. **Handle long-running tasks** - Show time elapsed, allow cancellation
6. **Beautiful > Feature-rich** - Polish what you build
7. **Mobile-responsive** - Chainlit supports it, use it

---

## 🎯 Success = User Can:

✅ Input a target name and configure quality settings
✅ See exactly what each node is doing in real-time
✅ Watch queries being generated and executed
✅ See facts being extracted and validated
✅ Watch risk analysis and connection mapping live
✅ Read a beautiful final report
✅ Download all data (markdown + JSON)
✅ Understand the entire research process without confusion

---

**You are building the face of this project. Make it shine! 🌟**
