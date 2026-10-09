"""Pipeline API routes"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from ...database.db import (
    get_pipeline_run,
    list_questions_by_run,
    get_pipeline_steps,
)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/{run_id}")
async def get_run(run_id: int) -> Dict[str, Any]:
    """Get pipeline run details"""
    run = get_pipeline_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return run


@router.get("/{run_id}/questions")
async def get_run_questions(
    run_id: int,
    status: str = "candidate"
) -> List[Dict[str, Any]]:
    """Get questions for a pipeline run (minimal fields for client)"""
    run = get_pipeline_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    questions = list_questions_by_run(run_id, status=status)
    # Return only essential fields for the client (question_text, id, level)
    return [
        {
            "id": q.get("id"),
            "question_text": q.get("question_text"),
            "level": q.get("level"),
        }
        for q in questions
    ]


@router.get("/{run_id}/steps")
async def get_run_steps(run_id: int) -> List[Dict[str, Any]]:
    """Get pipeline execution steps"""
    run = get_pipeline_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    steps = get_pipeline_steps(run_id)
    return steps
