"""
Chainlit UI for Deep Research AI Agent - Complete Redesign
Rich, real-time interface showing all background operations with full transparency
"""

import chainlit as cl
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List
import json
from collections import defaultdict

from src.agents.graph import ResearchWorkflow
from src.database.repository import ResearchRepository
from src.utils.config import Config
from src.utils.logger import get_logger

config = Config.from_env()
logger = get_logger(__name__)

class UIContentFormatter:
    """Formats content for rich display in Chainlit UI"""
    
    @staticmethod
    def format_search_queries(queries: List[str], iteration: int) -> str:
        """Format search queries with rich markdown"""
        content = f"# 🔍 Search Queries - Iteration {iteration}\n\n"
        content += "## Strategy Analysis\n\n"
        
        # Categorize queries
        categories = {
            "biographical": [],
            "professional": [],
            "connections": [],
            "validation": [],
            "other": []
        }
        
        for query in queries:
            query_lower = query.lower()
            if any(term in query_lower for term in ["birth", "education", "family", "personal"]):
                categories["biographical"].append(query)
            elif any(term in query_lower for term in ["ceo", "company", "role", "employment"]):
                categories["professional"].append(query)
            elif any(term in query_lower for term in ["partner", "connection", "relationship"]):
                categories["connections"].append(query)
            elif any(term in query_lower for term in ["verify", "confirm", "official"]):
                categories["validation"].append(query)
            else:
                categories["other"].append(query)
        
        for category, cat_queries in categories.items():
            if cat_queries:
                content += f"### {category.title()} Queries\n"
                for q in cat_queries:
                    content += f"- `{q}`\n"
                content += "\n"
        
        content += f"\n**Total Queries:** {len(queries)}"
        return content
    
    @staticmethod
    def format_search_results(results: List[Dict], queries: List[str]) -> str:
        """Format search results with detailed information"""
        content = "# 📊 Search Results Analysis\n\n"
        
        if queries:
            content += "## Executed Queries\n"
            for q in queries:
                content += f"- `{q}`\n"
            content += "\n---\n\n"
        
        content += f"## Results Summary\n"
        content += f"- **Total Results:** {len(results)}\n"
        
        # Group by domain
        domains = defaultdict(list)
        for result in results:
            domain = result.get('source_domain', 'unknown')
            domains[domain].append(result)
        
        content += f"- **Unique Sources:** {len(domains)}\n"
        content += f"- **Average Relevance:** {sum(r.get('score', 0) for r in results) / len(results) if results else 0:.2f}\n\n"
        
        content += "## Top Results by Source\n\n"
        for domain, domain_results in sorted(domains.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            content += f"### {domain} ({len(domain_results)} results)\n"
            for result in domain_results[:3]:  # Show top 3 per domain
                content += f"- **{result.get('title', 'No title')}**\n"
                content += f"  - URL: `{result.get('url', '')}`\n"
                content += f"  - Snippet: {result.get('content', '')[:200]}...\n\n"
        
        return content
    
    @staticmethod
    def format_extracted_facts(facts: List[Dict], new_count: int = 0) -> str:
        """Format extracted facts with categories and confidence"""
        content = "# 📋 Extracted Facts Analysis\n\n"
        
        if new_count > 0:
            content += f"## 🆕 New Facts This Iteration: {new_count}\n\n"
        
        # Group by category
        by_category = defaultdict(list)
        for fact in facts:
            by_category[fact.get('category', 'unknown')].append(fact)
        
        # Statistics
        content += "## Statistics\n"
        content += f"- **Total Facts:** {len(facts)}\n"
        content += f"- **Average Confidence:** {sum(f.get('confidence', 0) for f in facts) / len(facts) if facts else 0:.2f}\n"
        content += f"- **High Confidence (>0.8):** {sum(1 for f in facts if f.get('confidence', 0) > 0.8)}\n\n"
        
        # Facts by category
        for category in ['biographical', 'professional', 'financial', 'behavioral', 'unknown']:
            cat_facts = by_category.get(category, [])
            if cat_facts:
                content += f"## {category.title()} Facts ({len(cat_facts)})\n\n"
                
                # Sort by confidence
                cat_facts.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                
                for fact in cat_facts:
                    conf = fact.get('confidence', 0)
                    emoji = "🟢" if conf > 0.8 else "🟡" if conf > 0.6 else "🔴"
                    
                    content += f"### {emoji} {fact.get('content', '')}\n"
                    content += f"- **Confidence:** {conf:.2f}\n"
                    content += f"- **Source:** [{fact.get('source_domain', 'unknown')}]({fact.get('source_url', '#')})\n"
                    content += f"- **Extracted:** {fact.get('extracted_date', 'unknown')}\n"
                    
                    # Show entities if present
                    entities = fact.get('entities', {})
                    if entities:
                        content += "- **Entities:**\n"
                        for entity_type, entity_list in entities.items():
                            if entity_list:
                                content += f"  - {entity_type}: {', '.join(entity_list)}\n"
                    content += "\n"
        
        return content
    
    @staticmethod
    def format_validation_report(facts: List[Dict], contradictions: List[Dict]) -> str:
        """Format validation analysis with contradictions and adjustments"""
        content = "# ✅ Fact Validation Report\n\n"
        
        # Validation statistics
        verified_count = sum(1 for f in facts if f.get('confidence', 0) > 0.8)
        corroborated = sum(1 for f in facts if f.get('corroborations', 0) > 0)
        
        content += "## Validation Metrics\n"
        content += f"- **Facts Validated:** {len(facts)}\n"
        content += f"- **High Confidence:** {verified_count} ({verified_count/len(facts)*100 if facts else 0:.1f}%)\n"
        content += f"- **Corroborated:** {corroborated}\n"
        content += f"- **Contradictions Found:** {len(contradictions)}\n\n"
        
        # Show contradictions
        if contradictions:
            content += "## ⚠️ Contradictions Detected\n\n"
            for contradiction in contradictions:
                content += f"### Contradiction {contradictions.index(contradiction) + 1}\n"
                content += f"- **Facts in conflict:** {contradiction.get('fact_indices', [])}\n"
                content += f"- **Description:** {contradiction.get('description', 'No description')}\n\n"
        
        # Show confidence distribution
        content += "## Confidence Distribution\n\n"
        high_conf = [f for f in facts if f.get('confidence', 0) > 0.8]
        medium_conf = [f for f in facts if 0.6 < f.get('confidence', 0) <= 0.8]
        low_conf = [f for f in facts if f.get('confidence', 0) <= 0.6]

        content += f"- 🟢 **High (>0.8):** {len(high_conf)} facts\n"
        content += f"- 🟡 **Medium (0.6-0.8):** {len(medium_conf)} facts\n"
        content += f"- 🔴 **Low (≤0.6):** {len(low_conf)} facts\n\n"

        if high_conf:
            content += "### Top High-Confidence Facts\n\n"
            for fact in high_conf[:5]:  # Show top 5
                content += f"- {fact.get('content', '')[:100]}... (Confidence: {fact.get('confidence', 0):.2f})\n"
            content += "\n"
        return content
    
    @staticmethod
    def format_risk_analysis(risks: List[Dict]) -> str:
        """Format risk analysis with detailed breakdowns"""
        content = "# 🚨 Risk Analysis Report\n\n"
        
        if not risks:
            content += "✅ **No significant risks identified**\n"
            return content
        
        # Group by severity
        by_severity = defaultdict(list)
        for risk in risks:
            by_severity[risk.get('severity', 'unknown').lower()].append(risk)
        
        # Statistics
        content += "## Risk Summary\n"
        content += f"- **Total Risks:** {len(risks)}\n"
        content += f"- **Critical:** {len(by_severity.get('critical', []))}\n"
        content += f"- **High:** {len(by_severity.get('high', []))}\n"
        content += f"- **Medium:** {len(by_severity.get('medium', []))}\n"
        content += f"- **Low:** {len(by_severity.get('low', []))}\n\n"
        
        # Detailed risks by severity
        for severity in ['critical', 'high', 'medium', 'low']:
            severity_risks = by_severity.get(severity, [])
            if severity_risks:
                emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
                content += f"## {emoji} {severity.upper()} Severity Risks\n\n"
                
                for risk in severity_risks:
                    content += f"### {risk.get('category', 'Unknown')} Risk\n"
                    content += f"**{risk.get('description', 'No description')}**\n\n"
                    content += f"- **Confidence:** {risk.get('confidence', 0):.2f}\n"
                    content += f"- **Category:** {risk.get('category', 'unknown')}\n"
                    
                    # Evidence
                    evidence = risk.get('evidence', [])
                    if evidence:
                        content += "- **Evidence:**\n"
                        for ev in evidence:
                            content += f"  - {ev}\n"
                
                    content += "\n---\n\n"
        
        return content
    
    @staticmethod
    def format_connections_graph(connections: List[Dict]) -> str:
        """Format connection mapping with relationship details"""
        content = "# 🔗 Entity Connection Map\n\n"
        
        if not connections:
            content += "No connections identified yet.\n"
            return content
        
        # Group by relationship type
        by_type = defaultdict(list)
        for conn in connections:
            by_type[conn.get('relationship_type', 'Other')].append(conn)
        
        # Statistics
        unique_entities = set()
        for conn in connections:
            unique_entities.add(conn.get('entity_a', ''))
            unique_entities.add(conn.get('entity_b', ''))
        
        content += "## Network Statistics\n"
        content += f"- **Total Connections:** {len(connections)}\n"
        content += f"- **Unique Entities:** {len(unique_entities)}\n"
        content += f"- **Relationship Types:** {len(by_type)}\n\n"
        
        # Connections by type
        for rel_type, type_conns in by_type.items():
            content += f"## {rel_type} Relationships ({len(type_conns)})\n\n"
            
            for conn in type_conns:
                content += f"### {conn.get('entity_a', '')} ↔️ {conn.get('entity_b', '')}\n"
                content += f"- **Confidence:** {conn.get('confidence', 0):.2f}\n"
                
                time_period = conn.get('time_period', '')
                if time_period and time_period != 'Unknown':
                    content += f"- **Time Period:** {time_period}\n"
                
                evidence = conn.get('evidence', [])
                if evidence:
                    content += "- **Evidence:**\n"
                    for ev in evidence[:3]:  # Show first 3
                        content += f"  - {ev[:150]}...\n"
                
                content += "\n"
        
        return content
    
    @staticmethod
    def format_iteration_summary(iteration: int, state: Dict) -> str:
        """Format a complete iteration summary"""
        content = f"# 📊 Iteration {iteration} Complete Summary\n\n"
        
        facts = state.get('collected_facts', [])
        new_facts = state.get('new_facts', [])
        risks = state.get('risk_flags', [])
        connections = state.get('connections', [])
        queries = state.get('next_queries', [])
        
        content += "## Metrics This Iteration\n"
        content += f"- **New Facts Discovered:** {len(new_facts)}\n"
        content += f"- **Total Facts:** {len(facts)}\n"
        content += f"- **New Risks:** {len([r for r in risks if r.get('identified_at', '').startswith(datetime.now().strftime('%Y-%m-%d'))])}\n"
        content += f"- **New Connections:** {len([c for c in connections if c.get('identified_at', '').startswith(datetime.now().strftime('%Y-%m-%d'))])}\n"
        content += f"- **Queries Executed:** {len(queries)}\n\n"
        
        # Key findings
        content += "## Key Findings\n\n"
        
        # Top facts
        top_facts = sorted(new_facts, key=lambda x: x.get('confidence', 0), reverse=True)[:3]
        if top_facts:
            content += "### Top New Facts\n"
            for fact in top_facts:
                content += f"- {fact.get('content', '')} (Confidence: {fact.get('confidence', 0):.2f})\n"
            content += "\n"
        
        # Critical risks
        critical_risks = [r for r in risks if r.get('severity', '').lower() == 'critical']
        if critical_risks:
            content += "### ⚠️ Critical Risks\n"
            for risk in critical_risks:
                content += f"- {risk.get('description', '')}\n"
            content += "\n"
        
        return content


# ==================== Real-time State Monitor ====================

class ResearchStateMonitor:
    """Monitors research state and triggers UI updates"""
    
    def __init__(self, session_id: str, repository: ResearchRepository):
        self.session_id = session_id
        self.repository = repository
        self.last_state = {}
        self.formatter = UIContentFormatter()
        self.facts_per_iteration = {}  # Track fact count at start of each iteration
        self.risks_per_iteration = {}
        self.connections_per_iteration = {}

    
    def get_state_changes(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Detects changes AND returns the full current state. Always returns two values."""
        changes = {}
        full_state = {
            'facts': [], 'risks': [], 'connections': [], 
            'searches': [], 'iteration': self.last_state.get('iteration', 0),
            'current_node': self.last_state.get('last_node')
        }

        session = self.repository.get_research_session(self.session_id)
        if not session:
            # This guard is crucial: always return a 2-tuple to prevent unpacking errors.
            return changes, full_state

        # --- Populate full_state with all current data ---
        full_state['facts'] = [self._fact_to_dict(f) for f in self.repository.get_facts_by_session(self.session_id)]
        full_state['risks'] = [self._risk_to_dict(r) for r in self.repository.get_risks_by_session(self.session_id)]
        full_state['connections'] = [self._connection_to_dict(c) for c in self.repository.get_connections_by_session(self.session_id)]
        full_state['searches'] = [self._search_to_dict(s) for s in self.repository.get_searches_by_session(self.session_id)]
        
        # --- Detect and populate the 'changes' dictionary for transition logic ---
        fact_count = len(full_state['facts'])
        if fact_count != self.last_state.get('fact_count', 0):
            changes['new_facts'] = fact_count - self.last_state.get('fact_count', 0)
            changes['facts'] = full_state['facts']
            self.last_state['fact_count'] = fact_count

        search_count = len(full_state['searches'])
        if search_count != self.last_state.get('search_count', 0):
            changes['new_searches'] = search_count - self.last_state.get('search_count', 0)
            changes['searches'] = full_state['searches']
            self.last_state['search_count'] = search_count

        risk_count = len(full_state['risks'])
        if risk_count != self.last_state.get('risk_count', 0):
            changes['new_risks'] = risk_count - self.last_state.get('risk_count', 0)
            changes['risks'] = full_state['risks']
            self.last_state['risk_count'] = risk_count
        
        conn_count = len(full_state['connections'])
        if conn_count != self.last_state.get('conn_count', 0):
            changes['new_connections'] = conn_count - self.last_state.get('conn_count', 0)
            changes['connections'] = full_state['connections']
            self.last_state['conn_count'] = conn_count

        valid_iterations = [
            s['iteration'] for s in full_state['searches'] if s.get('iteration') is not None
        ]
        if valid_iterations:
            current_iteration = max(valid_iterations)
            full_state['iteration'] = current_iteration
            if current_iteration != self.last_state.get('iteration', 0):
                changes['iteration'] = current_iteration
                self.last_state['iteration'] = current_iteration
                # Store fact/risk/connection counts at start of new iteration
                prev_iter = current_iteration - 1
                if prev_iter > 0 and prev_iter not in self.facts_per_iteration:
                    self.facts_per_iteration[prev_iter] = len(full_state["facts"])
                    self.risks_per_iteration[prev_iter] = len(full_state["risks"])
                    self.connections_per_iteration[prev_iter] = len(full_state["connections"])
        
        if session.last_checkpoint_node_name != self.last_state.get('last_node'):
            changes['current_node'] = session.last_checkpoint_node_name
            full_state['current_node'] = session.last_checkpoint_node_name
            self.last_state['last_node'] = session.last_checkpoint_node_name
        
        # Add iteration tracking to state
        full_state["facts_per_iteration"] = self.facts_per_iteration
        full_state["risks_per_iteration"] = self.risks_per_iteration
        full_state["connections_per_iteration"] = self.connections_per_iteration
        return changes, full_state
    
    def get_search_results_for_iteration(self, iteration: int) -> List[Dict]:
        """Retrieve search results for an iteration from the database."""
        return self.repository.get_search_results(self.session_id, iteration)

    
    def _fact_to_dict(self, fact) -> Dict:
        return {
            "content": fact.content,
            "source_url": fact.source_url,
            "source_domain": fact.source_url.split('/')[2] if '/' in fact.source_url else fact.source_url,
            "confidence": fact.confidence_score,
            "category": fact.category.value,
            "extracted_date": fact.extracted_date.isoformat() if fact.extracted_date else None,
            "entities": {},  # Would need to parse from raw_context
        }
    
    def _risk_to_dict(self, risk) -> Dict:
        return {
            "severity": risk.severity.value,
            "category": risk.category.value,
            "description": risk.description,
            "evidence": risk.evidence,
            "confidence": risk.confidence_score,
        }
    
    def _connection_to_dict(self, conn) -> Dict:
        return {
            "entity_a": conn.entity_a,
            "entity_b": conn.entity_b,
            "relationship_type": conn.relationship_type,
            "evidence": conn.evidence,
            "confidence": conn.confidence_score,
        }
    
    def _search_to_dict(self, search) -> Dict:
        return {
            "query": search.query,
            "iteration": search.iteration,
            "results_count": search.results_count,
            "relevance_score": search.relevance_score or 0.0,
            "executed_at": search.executed_at.isoformat() if search.executed_at else None,
        }


# ==================== Main UI Application ====================

@cl.on_chat_start
async def start():
    """Initialize the research session"""
    
    # Welcome message with capabilities
    await cl.Message(
        content="""# 🔬 Deep Research AI Agent

Welcome to the autonomous research system powered by LangGraph and Multi-Model AI.

## What This Agent Does
- 🔍 **Progressive Search**: Starts broad, then focuses on specific areas
- 📊 **Multi-Source Analysis**: Searches across multiple domains and sources  
- ✅ **Fact Validation**: Cross-references and validates information
- 🚨 **Risk Detection**: Identifies potential red flags and concerns
- 🔗 **Connection Mapping**: Discovers relationships between entities
- 📝 **Comprehensive Reporting**: Generates detailed research reports

## How to Use
1. Enter the name of the person or entity to research
2. Configure the research settings
3. Watch the real-time analysis unfold
4. Review detailed findings in the side panels

Let's begin!"""
    ).send()
    
    # Get target name
    target_response = await cl.AskUserMessage(
        content="**Who would you like to research?** Enter a name (person or organization):",
        timeout=300
    ).send()
    
    if not target_response or not target_response['output'].strip():
        await cl.Message(content="❌ No target provided. Please refresh to start again.").send()
        return
    
    target_name = target_response['output'].strip()
    cl.user_session.set("target_name", target_name)
    
    # Show settings
    settings = cl.ChatSettings(
        [
            cl.input_widget.Slider(
                id="research_depth",
                label="Research Depth (iterations)",
                initial=7,
                min=1,
                max=7,
                step=1,
                description="Number of research iterations (more = deeper but slower)"
            ),
            cl.input_widget.Slider(
                id="max_results_per_query",
                label="Max Search Results per Query",
                initial=10,
                min=3,
                max=20,
                step=1,
                description="Maximum number of results to retrieve for a query"
            ),
            cl.input_widget.Slider(
                id="fact_extraction_temp",
                label="Fact Extraction Temperature",
                initial=0.3,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Lower = more factual, Higher = more creative"
            ),
            cl.input_widget.Slider(
                id="validation_temp",
                label="Validation Temperature",
                initial=0.2,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Confidence scoring sensitivity"
            ),
            cl.input_widget.Slider(
                id="risk_analysis_temp",
                label="Risk Analysis Temperature",
                initial=0.3,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Risk detection sensitivity"
            ),
            cl.input_widget.Slider(
                id="report_generation_temp",
                label="Report Generation Temperature",
                initial=0.4,
                min=0.0,
                max=1.0,
                step=0.1,
                description="Report creativity vs. precision"
            ),
        ]
    )
    await settings.send()
    
    await cl.Message(
        content=f"""## Ready to Research: **{target_name}**

Type **'start'** to begin the investigation. 

During the research, you'll see:
- Each iteration with its own set of steps
- Real-time updates as each agent works
- Detailed findings in expandable side panels
- Progress indicators and statistics

The research will adapt based on findings, focusing on the most promising leads."""
    ).send()
    
    cl.user_session.set("awaiting_start", True)


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages and start research"""
    
    if cl.user_session.get("awaiting_start"):
        if message.content.strip().lower() in ['start', 'begin', 'go']:
            await start_research()
        else:
            # User is providing a new target name
            new_target = message.content.strip()
            cl.user_session.set("target_name", new_target)
            await cl.Message(
                content=f"""## Ready to Research: **{new_target}**

Type **'start'** to begin the investigation."""
            ).send()
    else:
        # Research in progress
        await cl.Message(
            content="Research is in progress. You can view the details in the side panels by clicking on any step."
        ).send()


async def start_research():
    """Initialize and run the research workflow"""
    research_successful = False
    target_name = cl.user_session.get("target_name")
    settings = cl.user_session.get("settings", {})
    research_depth = settings.get("research_depth", 7)
    config_overrides = {
        "fact_extraction_temp": settings.get("fact_extraction_temp", 0.3),
        "validation_temp": settings.get("validation_temp", 0.2),
        "risk_temp": settings.get("risk_analysis_temp", 0.3),
        "report_temp": settings.get("report_generation_temp", 0.4),
        "max_results_per_query": settings.get("max_results_per_query", 10),
    }

    config.performance.fact_extraction_temperature = config_overrides["fact_extraction_temp"]
    config.performance.validation_temperature = config_overrides["validation_temp"]
    config.performance.risk_analysis_temperature = config_overrides["risk_temp"]
    config.performance.report_generation_temperature = config_overrides["report_temp"]
    config.application.max_search_iterations = research_depth
    config.search.max_results_per_query = config_overrides["max_results_per_query"]

    
    cl.user_session.set("awaiting_start", False)
    
    # Initialize components
    repository = ResearchRepository(config.database.database_url)
    workflow = ResearchWorkflow(repository, config)
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create main research message
    main_msg = await cl.Message(
        content=f"""## 🚀 Research Started

**Target:** {target_name}
**Session:** `{session_id}`
**Depth:** {research_depth} iterations
Research is now running. Updates will appear below."""
    ).send()
    
    # Initialize state monitor
    monitor = ResearchStateMonitor(session_id, repository)
    formatter = UIContentFormatter()
    
    # Track UI components
    iteration_steps = {}
    node_steps = {}
    
    # Start research in background
    stop_event = asyncio.Event()
    research_task = asyncio.create_task(
        asyncio.to_thread(
            workflow.run_research,
            target_name,
            research_depth,
            session_id,
            stop_event
        )
    )
    
    try:
        # Monitor and update UI
        await monitor_research_progress(
            research_task,
            monitor,
            formatter,
            iteration_steps,
            node_steps,
            research_depth,
            workflow  # Pass workflow reference
        )
        
        # Get final state
        final_state = await research_task


        # Display final report
        try:
            await display_final_report(
                final_state,
                session_id,
                repository,
                formatter
            )
        
        
        except Exception as e:
            logger.error(f"Failed to display final report: {e}", exc_info=True)
            await cl.Message(
                content=f"❌ Failed to generate final report: {str(e)}"
            ).send()
        
        # Always send completion message
        try:
            await cl.Message(
                content="""## ✨ Research Complete!

Your research session has finished. """ + (f"Check the final report above for details." if final_state.get('final_report') else "Some parts of the report may have failed.")
            ).send()
        except Exception as e2:
            logger.error(f"Failed to send completion message: {e2}")

        research_successful = True

    except Exception as e:
        logger.error(f"Research failed: {e}")
        await cl.Message(
            content=f"❌ Research encountered an error:\n```\n{str(e)}\n```"
        ).send()
        stop_event.set()

    finally:
        # Reset for next research
        if research_successful:
            cl.user_session.set("awaiting_start", True)

async def monitor_research_progress(
    research_task: asyncio.Task,
    monitor: ResearchStateMonitor,
    formatter: UIContentFormatter,
    iteration_steps: Dict,
    node_steps: Dict,
    research_depth: int,
    workflow: ResearchWorkflow
):
    """Monitors research progress and updates UI in real-time with resilient logic."""

    last_processed_node_key = None
    last_seen_node = None

    while not research_task.done():
        await asyncio.sleep(0.5)

        changes, full_state = monitor.get_state_changes()

        current_iteration = full_state.get('iteration', 0)
        current_node_name = full_state.get('current_node')

        # Skip if no node name
        if not current_node_name:
            continue


        # Handle Reporter node specially (it runs AFTER iterations complete, so iteration might be 0 or final)
        if current_node_name == "Reporter":
            if "Reporter" not in node_steps:
                # Close last iteration node if exists
                if last_processed_node_key and last_processed_node_key in node_steps:
                    last_node_info = last_processed_node_key.split('_', 1)
                    if len(last_node_info) == 2:
                        last_iteration = int(last_node_info[0])
                        last_node_name = last_node_info[1]
                        await update_node_step(
                            node_steps[last_processed_node_key],
                            last_node_name, full_state, formatter, monitor, last_iteration, "completed"
                        )

                # Create Reporter step
                report_step = cl.Step(
                    name=f"📝 Generating Final Report",
                    type="run", show_input=False
                )
                report_step.output = "⏳ Synthesizing all findings..."
                await report_step.send()
                node_steps["Reporter"] = report_step
            continue

        # Skip if iteration is 0 (hasn't started yet)
        if current_iteration == 0:
            continue

        # Create iteration step if needed
        if current_iteration not in iteration_steps:
            iter_step = cl.Step(
                name=f"🔄 Iteration {current_iteration}/{research_depth}",
                type="run", show_input=False
            )
            await iter_step.send()
            iteration_steps[current_iteration] = iter_step

        node_key = f"{current_iteration}_{current_node_name}"

        # Close previous node if we moved to a different node
        if last_seen_node and last_seen_node != current_node_name and last_processed_node_key and last_processed_node_key in node_steps:
            last_node_info = last_processed_node_key.split('_', 1)  # Split only on first underscore
            if len(last_node_info) == 2:
                last_iteration = int(last_node_info[0])
                last_node_name = last_node_info[1]
                await update_node_step(
                    node_steps[last_processed_node_key],
                    last_node_name, full_state, formatter, monitor, last_iteration, "completed"
                )

        # Create new node step if needed
        if node_key not in node_steps:
            parent_step = iteration_steps.get(current_iteration)
            node_step = cl.Step(
                name=f"🔧 {current_node_name}",
                type="run", parent_id=parent_step.id if parent_step else None,
                show_input=False
            )
            node_step.output = "⏳ Processing..."
            await node_step.send()
            node_steps[node_key] = node_step

        # Update current node
        await update_node_step(
            node_steps[node_key],
            current_node_name, full_state, formatter, monitor, current_iteration, "running"
        )

        last_processed_node_key = node_key
        last_seen_node = current_node_name


    # Mark final node as completed when research is done
    if last_processed_node_key and last_processed_node_key in node_steps:
        last_node_info = last_processed_node_key.split('_', 1)
        if len(last_node_info) == 2:
            last_iteration = int(last_node_info[0])
            last_node_name = last_node_info[1]
            changes_final, full_state_final = monitor.get_state_changes()
            await update_node_step(
                node_steps[last_processed_node_key],
                last_node_name, full_state_final, formatter, monitor, last_iteration, "completed"
            )

    # Mark Reporter as completed if it exists
    if "Reporter" in node_steps:
        reporter_step = node_steps["Reporter"]
        reporter_step.output = "✅ Final report generated"
        await reporter_step.update()


async def update_node_step(
    step: cl.Step,
    node_name: str,
    full_state: Dict,
    formatter: UIContentFormatter,
    monitor: ResearchStateMonitor,
    iteration: int,
    status: str
):
    """Update a node step with current data"""

    elements = []
    output_text = None


    if node_name == "Planner":
        queries = list(set([s['query'] for s in full_state['searches'] if s['iteration'] == iteration]))
        if queries:
            output_text = f"✅ Generated {len(queries)} search **Strategy**. Click to view details."
            elements.append(cl.Text(
                name="Strategy",
                content=formatter.format_search_queries(queries, iteration),
                display="side"
            ))
        else:
            output_text = "⏳ Generating search queries..." if status == "running" else "✅ Queries generated"

    elif node_name == "Searcher":
        searches_this_iter = [s for s in full_state['searches'] if s['iteration'] == iteration]
        if searches_this_iter:
            results = monitor.get_search_results_for_iteration(iteration)
            if results:
                output_text = f"✅ Executed {len(searches_this_iter)} searches, found {len(results)} **Results**. Click to view."
                elements.append(cl.Text(
                    name="Results",
                    content=formatter.format_search_results(results, [s['query'] for s in searches_this_iter]),
                    display="side"
                ))
            else:
                output_text = f"✅ Executed {len(searches_this_iter)} searches"
        else:
            output_text = "⏳ Executing searches..." if status == "running" else "✅ Searches completed"

    elif node_name == "Extractor":
        all_facts = full_state["facts"]
        # Get only NEW facts from this iteration
        facts_before = full_state.get("facts_per_iteration", {}).get(iteration - 1, 0)
        new_facts = all_facts[facts_before:] if len(all_facts) > facts_before else all_facts
        
        if new_facts:
            total_count = len(all_facts)
            new_count = len(new_facts)
            output_text = f"✅ Extracted {new_count} new facts ({total_count} total). View **Facts**. Click to view details."
            elements.append(cl.Text(
                name="Facts",
                content=formatter.format_extracted_facts(new_facts),
                display="side"
            ))
        else:
            output_text = "⏳ Extracting facts..." if status == "running" else "✅ Extraction complete"
    elif node_name == "Validator":
        facts = full_state['facts']
        if facts:
            output_text = f"✅ Validated {len(facts)} facts. View **Validation** report."
            elements.append(cl.Text(
                name="Validation",
                content=formatter.format_validation_report(facts, []),
                display="side"
            ))
        else:
            output_text = "⏳ Validating facts..." if status == "running" else "✅ Validation complete"
    elif node_name == "Risk Analyzer":
        all_risks = full_state['risks']
        # Get only NEW risks from this iteration
        risks_before = full_state.get("risks_per_iteration", {}).get(iteration - 1, 0)
        new_risks = all_risks[risks_before:] if len(all_risks) > risks_before else all_risks
        
        if new_risks:
            total_count = len(all_risks)
            new_count = len(new_risks)
            output_text = f"🚨 Found {new_count} new risks ({total_count} total). View **Risks**."
            elements.append(cl.Text(
                name="Risks",
                content=formatter.format_risk_analysis(new_risks),
                display="side"
            ))
        else:
            output_text = "⏳ Analyzing risks..." if status == "running" else "✅ No new risks identified"



    elif node_name == "Connection Mapper":
        all_connections = full_state['connections']
        # Get only NEW connections from this iteration
        connections_before = full_state.get("connections_per_iteration", {}).get(iteration - 1, 0)
        new_connections = all_connections[connections_before:] if len(all_connections) > connections_before else all_connections
        
        if new_connections:
            total_count = len(all_connections)
            new_count = len(new_connections)
            output_text = f"🔗 Mapped {new_count} new connections ({total_count} total). View **Connections**."
            elements.append(cl.Text(
                name="Connections",
                content=formatter.format_connections_graph(new_connections),
                display="side"
            ))
        else:
            output_text = "⏳ Mapping connections..." if status == "running" else "✅ No new connections found"
    elif node_name == "Reporter":


        # Reporter node handled separately in monitor_research_progress
        output_text = "⏳ Generating final report..." if status == "running" else "✅ Final report generated"

    # Set the output
    if output_text:
        step.output = output_text

    # Mark as completed
    if status == "completed" and "⏳" in step.output:
        step.output = step.output.replace("⏳", "✅")

    # Attach elements
    if elements:
        step.elements = elements

    await step.update()


async def display_final_report(
    final_state: Dict,
    session_id: str,
    repository: ResearchRepository,
    formatter: UIContentFormatter
):
    """Display comprehensive final report"""


    # Create final report step
    report_step = cl.Step(
        name="📊 Final Research Report",
        type="run",
        show_input=False
    )
    await report_step.send()
    
    # Gather all data
    facts = repository.get_facts_by_session(session_id)
    risks = repository.get_risks_by_session(session_id)
    connections = repository.get_connections_by_session(session_id)
    searches = repository.get_searches_by_session(session_id)
    
    # Create a temporary monitor instance for conversion
    temp_monitor = ResearchStateMonitor(session_id, repository)
    
    # Convert to dicts
    facts_dict = [temp_monitor._fact_to_dict(f) for f in facts]
    risks_dict = [temp_monitor._risk_to_dict(r) for r in risks]
    connections_dict = [temp_monitor._connection_to_dict(c) for c in connections]
    searches_dict = [temp_monitor._search_to_dict(s) for s in searches]
    
    # Create detailed report sections FIRST (to know what's available)
    elements = []
    available_sections = []

    # Main report
    if final_state.get('final_report'):
        elements.append(
            cl.Text(
                name="Summary",
                content=final_state['final_report'],
                display="side"
            )
        )
        available_sections.append("**Summary**")

    # All facts with analysis
    elements.append(
        cl.Text(
            name="Facts",
            content=formatter.format_extracted_facts(facts_dict),
            display="side"
        )
    )
    available_sections.append("**Facts**")

    # Risk assessment
    if risks_dict:
        elements.append(
            cl.Text(
                name="Risks",
                content=formatter.format_risk_analysis(risks_dict),
                display="side"
            )
        )
        available_sections.append("**Risks**")

    # Connection network
    if connections_dict:
        elements.append(
            cl.Text(
                name="Network",
                content=formatter.format_connections_graph(connections_dict),
                display="side"
            )
        )
        available_sections.append("**Network**")

    # Search history
    search_content = "# 🔍 Search History\n\n"
    by_iteration = defaultdict(list)
    for s in searches_dict:
        by_iteration[s['iteration']].append(s)

    for iteration in sorted(by_iteration.keys()):
        search_content += f"## Iteration {iteration}\n"
        for search in by_iteration[iteration]:
            search_content += f"- `{search['query']}` ({search['results_count']} results, relevance: {search['relevance_score']:.2f})\n"
        search_content += "\n"

    elements.append(
        cl.Text(
            name="History",
            content=search_content,
            display="side"
        )
    )
    available_sections.append("**History**")

    # Raw data export
    raw_data = {
        "session_id": session_id,
        "target": final_state.get('target_name'),
        "facts": facts_dict,
        "risks": risks_dict,
        "connections": connections_dict,
        "searches": searches_dict,
        "iterations": final_state.get('current_iteration', 0)
    }

    elements.append(
        cl.Text(
            name="Data",
            content=f"```json\n{json.dumps(raw_data, indent=2, default=str)}\n```",
            display="side"
        )
    )
    available_sections.append("**Data**")

    # Create summary with clickable sections
    sections_list = " | ".join(available_sections)
    summary = f"""## ✨ Research Complete!

### 📈 Final Statistics
- **Total Facts Discovered:** {len(facts)}
- **Risk Flags Identified:** {len(risks)}
- **Connections Mapped:** {len(connections)}
- **Search Queries Executed:** {len(searches)}
- **Iterations Completed:** {final_state.get('current_iteration', 0)}

### 🎯 Key Insights
- **High Confidence Facts:** {sum(1 for f in facts_dict if f['confidence'] > 0.8)}
- **Critical Risks:** {sum(1 for r in risks_dict if r['severity'].lower() == 'critical')}
- **Strong Connections:** {sum(1 for c in connections_dict if c['confidence'] > 0.8)}

### 📊 Available Reports
Click to explore: {sections_list}"""

    # Update the report step with elements and final output
    report_step.elements = elements
    report_step.output = summary
    await report_step.update()



@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates"""
    cl.user_session.set("settings", settings)
