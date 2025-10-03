import asyncio
from src.agents.graph import ResearchWorkflow
from src.utils.config import Config
from src.database.repository import ResearchRepository
from src.utils.logger import setup_logging
from loguru import logger
import uuid
from datetime import datetime

def run_full_workflow():
    """
    Runs the entire research workflow from start to finish.
    """
    config = Config.from_env()
    setup_logging(config)
    logger.info("Configuration loaded.")

    try:
        repository = ResearchRepository(config.database.database_url)
        logger.info("Database repository initialized.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return

    workflow = ResearchWorkflow(repository, config)
    logger.info("Research workflow initialized.")

    target_name = "Satya Nadella"
    research_depth = 7

    logger.info(f"Starting FULL research for: {target_name} with depth: {research_depth}")
    print("="*70)

    try:
        final_state = workflow.run_research(
            target_name=target_name,
            research_depth=research_depth
        )
        print("="*70)
        logger.success("Research finished successfully.")
        print("\n\n" + "="*30 + " FINAL REPORT " + "="*30)
        print(final_state.get("final_report", "No report was generated."))

    except Exception as e:
        logger.exception("The research workflow failed unexpectedly.")

def run_step_by_step():
    """
    Runs each node of the workflow individually for one iteration to debug.
    """
    config = Config.from_env()
    setup_logging(config)
    logger.info("Configuration loaded for step-by-step debug.")

    try:
        repository = ResearchRepository(config.database.database_url)
        logger.info("Database repository initialized.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return

    workflow = ResearchWorkflow(repository, config)
    logger.info("Research workflow initialized.")

    # Manually create the initial state
    target_name = "Satya Nadella"
    research_depth = 7
    session_id = str(uuid.uuid4())
    state = {
        "target_name": target_name,
        "research_depth": research_depth,
        "session_id": session_id,
        "collected_facts": [],
        "connections": [],
        "risk_flags": [],
        "search_history": [],
        "facts_before_iteration": 0,
        "current_iteration": 0,
        "next_queries": [],
        "explored_topics": set(),
        "raw_search_results": [],
        "start_time": datetime.now(),
        "last_update": None,
        "final_report": None,
        "connection_graph": None
    }
    repository.create_session(session_id=session_id, target_name=target_name, research_depth=research_depth)

    logger.info("--- STARTING STEP-BY-STEP EXECUTION (1st Iteration) ---")

    try:
        # --- 1. Planner ---
        logger.info("[STEP 1] Running Planner Node...")
        state['current_iteration'] = state.get('current_iteration', 0) + 1
        state['facts_before_iteration'] = len(state.get('collected_facts', []))
        state = workflow.planner.execute(state)
        logger.info(f"  -> Planner generated {len(state.get('next_queries', []))} queries.")
        logger.info(f"  -> Queries: {state.get('next_queries')}")

        # --- 2. Searcher ---
        logger.info("[STEP 2] Running Searcher Node...")
        state = workflow.searcher.execute(state)
        logger.info(state)
        logger.info(f"  -> Searcher found {len(state.get('raw_search_results', []))} raw results.")

        # --- 3. Extractor ---
        logger.info("[STEP 3] Running Extractor Node...")
        state = workflow.extractor.execute(state)
        logger.info(f"  -> Extractor found {len(state.get('collected_facts', []))} new facts.")
        logger.info(state.get('collected_facts', []))

        # --- 4. Validator ---
        logger.info("[STEP 4] Running Validator Node...")
        state = workflow.validator.execute(state)
        avg_confidence = sum(f.get('confidence', 0) for f in state.get('collected_facts', [])) / len(state.get('collected_facts', [])) if state.get('collected_facts') else 0
        logger.info(f"  -> Validator processed facts. Average confidence is now {avg_confidence:.2f}.")
        logger.info(state.get('collected_facts', []))

        # --- 5. Risk Analyzer ---
        logger.info("[STEP 5] Running Risk Analyzer Node...")
        state = workflow.risk_analyzer.execute(state)
        logger.info(f"  -> Risk Analyzer identified {len(state.get('risk_flags', []))} risks.")
        logger.info(state.get('risk_flags', []))
        # --- 6. Connection Mapper ---
        logger.info("[STEP 6] Running Connection Mapper Node...")
        state = workflow.connection_mapper.execute(state)
        logger.info(f"  -> Connection Mapper identified {len(state.get('connections', []))} connections.")
        logger.info(state.get('connections', []))

        # --- 7. Reporter ---
        logger.success("[FINAL STEP] Running Reporter Node...")
        state = workflow.reporter.execute(state)
        logger.info("  -> Report generation complete.")

        print("\n\n" + "="*30 + " FINAL REPORT " + "="*30)
        print(state.get("final_report", "No report was generated."))

    except Exception as e:
        logger.exception("A step-by-step execution failed.")

    logger.info("--- END OF STEP-BY-STEP TEST ---")

if __name__ == "__main__":
    # By default, run the step-by-step test for debugging.
    # To run the full workflow, comment out the line below and uncomment the other.
    run_step_by_step()
    # run_full_workflow()