"""Pipeline API routes"""
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from ...database.db import (
    create_pipeline_run,
    get_cv,
    get_db_path,
    get_jd,
    get_pipeline_run,
    list_llm_operations_by_run,
    list_questions_by_run,
    get_pipeline_steps,
    update_pipeline_run_status,
)
from ...models.pipeline import PipelineState
from ...pipeline.graph import run_pipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
logger = logging.getLogger(__name__)


class PipelineStartRequest(BaseModel):
    cv_id: int = Field(gt=0)
    jd_id: int = Field(gt=0)
    prompt_version: str | None = None


async def _run_pipeline_in_background(
    run_id: int,
    cv_id: int,
    jd_id: int,
    prompt_version: str | None,
    db_path: str,
) -> None:
    """Execute pipeline and persist an error status if graph execution fails."""
    try:
        update_pipeline_run_status(run_id, "running", db_path=db_path)
        state = PipelineState(
            run_id=run_id,
            cv_id=cv_id,
            jd_id=jd_id,
            prompt_version=prompt_version,
            db_path=db_path,
            status="running",
        )
        await run_pipeline(state, db_path)
    except Exception as err:
        logger.exception("Pipeline run %s failed", run_id)
        update_pipeline_run_status(
            run_id,
            status="failed",
            error_message=f"{type(err).__name__}: {str(err).strip() or repr(err)}",
            db_path=db_path,
        )


@router.post("/start", status_code=202)
async def start_pipeline(
    request: PipelineStartRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, int | str]:
    """Create a pipeline run and execute it in the background."""
    db_path = get_db_path()
    cv = get_cv(request.cv_id, db_path=db_path)
    if cv is None:
        raise HTTPException(status_code=404, detail="CV not found")
    if cv["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"CV is not ready; current status: {cv['status']}",
        )

    jd = get_jd(request.jd_id, db_path=db_path)
    if jd is None:
        raise HTTPException(status_code=404, detail="JD not found")
    if jd["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"JD is not ready; current status: {jd['status']}",
        )

    run_id = create_pipeline_run(
        request.cv_id,
        request.jd_id,
        request.prompt_version,
        db_path=db_path,
    )
    background_tasks.add_task(
        _run_pipeline_in_background,
        run_id,
        request.cv_id,
        request.jd_id,
        request.prompt_version,
        db_path,
    )
    return {"run_id": run_id, "status": "pending"}


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


@router.get("/{run_id}/operations")
async def get_run_operations(run_id: int) -> Dict[str, Any]:
    """Get per-operation LLM usage and aggregated cost for a run."""
    run = get_pipeline_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    operations = list_llm_operations_by_run(run_id)
    return {
        "run_id": run_id,
        "total_cost_usd": round(
            sum(operation["cost_usd"] for operation in operations), 10
        ),
        "operations": operations,
    }
