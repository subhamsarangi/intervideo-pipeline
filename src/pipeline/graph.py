"""
Phase 11: LangGraph Pipeline Assembly

Main graph orchestration with conditional edges, parallel fan-out, and SqliteSaver checkpointing.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Union
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite

from src.models.pipeline import PipelineState
from src.pipeline.nodes.security import security_node
from src.pipeline.nodes.triage import triage_node
from src.pipeline.nodes.enrichment import enrichment_node
from src.pipeline.nodes.project_questions import project_questions_node
from src.pipeline.nodes.skill_questions import skill_questions_node
from src.pipeline.nodes.merge_dedupe import merge_dedupe_node
from src.pipeline.nodes.judge_evaluation import judge_evaluation_node

logger = logging.getLogger(__name__)


def should_enrich(state: Union[PipelineState, Dict[str, Any]]) -> bool:
    """Conditional: route to enrichment if fast-moving tech flagged."""
    if isinstance(state, dict):
        triage_result = state.get("triage_result")
    else:
        triage_result = state.triage_result
    
    if triage_result is None:
        return False
    
    # If any items flagged for enrichment, enrich
    flagged = getattr(triage_result, "flagged_for_enrichment", [])
    return len(flagged) > 0





async def build_graph(db_path: str):
    """
    Assemble the LangGraph StateGraph with all nodes, conditional edges, and checkpointing.
    
    Args:
        db_path: Path to SQLite database for checkpointing (or ":memory:" for in-memory)
    
    Returns:
        Compiled graph ready for execution
    
    Node Flow:
        1. security_node (sync) → wraps/validates CV/JD
        2. triage_node (async) → ranks projects/skills, flags tech for enrichment
        3. [conditional] enrichment_node (async) → web search for flagged tech
        4. [parallel fan-out]
           - project_questions_node (async) → generates questions from projects
           - skill_questions_node (async) → generates questions from skills
        5. merge_dedupe_node (async) → combines, deduplicates, checks coverage
        6. judge_evaluation_node (async) → scores questions, saves to DB
        
        NOTE: gap_fill_node deferred to Phase 8b, fitness_node deferred to Phase 10
    """
    
    # Initialize state graph
    graph = StateGraph(PipelineState)
    
    # Add all nodes
    graph.add_node("security", security_node)
    graph.add_node("triage", triage_node)
    graph.add_node("enrichment", enrichment_node)
    graph.add_node("project_questions", project_questions_node)
    graph.add_node("skill_questions", skill_questions_node)
    graph.add_node("merge_dedupe", merge_dedupe_node)
    graph.add_node("judge_evaluation", judge_evaluation_node)
    
    # Define main execution flow
    graph.add_edge("security", "triage")
    graph.add_edge("triage", "enrichment")
    graph.add_edge("enrichment", "project_questions")
    graph.add_edge("project_questions", "skill_questions")
    graph.add_edge("skill_questions", "merge_dedupe")
    graph.add_edge("merge_dedupe", "judge_evaluation")
    graph.add_edge("judge_evaluation", END)
    
    # Set entry point
    graph.set_entry_point("security")
    
    # Compile graph WITHOUT checkpointing to avoid schema issues
    compiled_graph = graph.compile()
    
    logger.info(f"Built LangGraph pipeline (checkpointing disabled)")
    
    return compiled_graph


async def run_pipeline(
    state: PipelineState,
    db_path: str,
    thread_id: Optional[str] = None,
    resume_from_checkpoint: bool = False,
) -> Dict[str, Any]:
    """
    Execute the compiled pipeline graph.
    
    Args:
        state: Initial PipelineState with run_id, cv_id, jd_id
        db_path: Path to SQLite database
        thread_id: Checkpoint thread ID (defaults to run_id)
        resume_from_checkpoint: If True, resume from last checkpoint instead of starting fresh
    
    Returns:
        Final state dict after full pipeline execution
    
    Raises:
        RuntimeError: If pipeline execution fails
    """
    
    if thread_id is None:
        thread_id = str(state.run_id)
    
    graph = await build_graph(db_path)
    
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    
    try:
        # Convert state to dict for invoke
        state_dict = state.model_dump() if hasattr(state, 'model_dump') else state
        
        # Run graph async
        final_state = await graph.ainvoke(state_dict, config)
        
        logger.info(f"Pipeline {thread_id} completed successfully")
        return final_state
        
    except Exception as e:
        logger.error(f"Pipeline {thread_id} failed: {str(e)}")
        raise RuntimeError(f"Pipeline execution failed: {str(e)}") from e


async def get_pipeline_checkpoint(
    db_path: str,
    thread_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve checkpoint for a pipeline run (for resumption).
    
    Args:
        db_path: Path to SQLite database (or ":memory:")
        thread_id: Thread ID of the run to resume
    
    Returns:
        Last saved state dict, or None if no checkpoint exists
    """
    
    conn = await aiosqlite.connect(db_path)
    checkpointer = AsyncSqliteSaver(conn)
    
    try:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await checkpointer.aget_tuple(config)
        
        if checkpoint:
            logger.info(f"Found checkpoint for thread {thread_id}")
            return checkpoint.checkpoint.get("channel_values") if hasattr(checkpoint.checkpoint, "get") else None
        
        logger.info(f"No checkpoint found for thread {thread_id}")
        return None
    finally:
        await conn.close()


# Export for use in API/main runners
__all__ = ["build_graph", "run_pipeline", "get_pipeline_checkpoint"]
