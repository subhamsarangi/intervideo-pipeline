"""LLM Judge evaluation node: scores and persists questions to database"""

from typing import Any, Dict, List, Union

import instructor
from pydantic import BaseModel, Field

from src.database.db import (
    create_pipeline_step,
    update_pipeline_step,
    update_pipeline_run_status,
    save_question,
)
from src.models.jd import ParsedJD
from src.models.judge_evaluation import QuestionScore
from src.models.pipeline import PipelineState
from src.utils.embeddings import get_async_openai_client
from src.utils.judge_prompts import JUDGE_SYSTEM_PROMPT, get_judge_user_prompt
from src.utils.llm_tracking import increment_llm_calls

JUDGE_MODEL = "gpt-5.4-nano"
BATCH_SIZE = 10
MAX_ATTEMPTS = 2  # initial call + 1 retry if scores are missing


class JudgeResponseModel(BaseModel):
    """Response model for judge evaluation (compatible with Instructor)"""
    question_scores: List[QuestionScore] = Field(..., description="Scores for each question")


async def _score_batch(client, batch: List[Dict], jd_dict: Dict, cv_dict: Dict) -> Dict[int, QuestionScore]:
    """Call the judge once. Returns {1-based question index: QuestionScore}."""
    user_prompt = get_judge_user_prompt(batch, jd_dict, cv_dict)

    # create_with_completion returns (parsed_model, raw_completion) so usage is available
    response, completion = await client.chat.completions.create_with_completion(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_model=JudgeResponseModel,
        temperature=0.3,  # Lower temp for consistent grading
    )

    usage = getattr(completion, "usage", None)
    if usage:
        increment_llm_calls(
            operation="judge_evaluation",
            model=JUDGE_MODEL,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )

    by_index: Dict[int, QuestionScore] = {}
    for s in response.question_scores:
        # Ignore out-of-range or duplicate indices
        if 1 <= s.question_index <= len(batch) and s.question_index not in by_index:
            by_index[s.question_index] = s
    return by_index


async def judge_evaluation_node(
    state: Union[PipelineState, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    LLM Judge: Score all merged questions on clarity, relevance, difficulty_match, groundedness.

    overall_score / pass_threshold are computed in code (see models.judge_evaluation).
    Saves scored questions to database with status (rejected_auto or candidate).

    Args:
        state: PipelineState or dict with run_id, merged_questions, jd, cv

    Returns:
        Dict with status, evaluated_questions, passed_count, rejected_count, error_message
    """
    if isinstance(state, PipelineState):
        state_dict = state.model_dump()
    else:
        state_dict = dict(state)

    run_id = state_dict.get("run_id")
    db_path = state_dict.get("db_path")

    if run_id is None:
        raise ValueError("run_id is required in pipeline state")

    # Create pipeline step record
    step_id = create_pipeline_step(run_id, "judge_evaluation", db_path=db_path)
    update_pipeline_step(step_id, status="running", db_path=db_path)

    try:
        # Get merged questions
        merged_questions: List[Dict] = state_dict.get("merged_questions", [])

        if not merged_questions:
            update_pipeline_step(
                step_id,
                status="completed",
                output_json={"evaluated": 0, "passed": 0, "rejected": 0},
                db_path=db_path,
            )
            return {
                "status": "running",
                "evaluated_questions": [],
                "passed_count": 0,
                "rejected_count": 0,
                "error_message": None,
            }

        # Get JD and CV context
        jd_obj = state_dict.get("jd")
        if jd_obj is None:
            raise ValueError("jd required in state")
        if isinstance(jd_obj, dict):
            jd_obj = ParsedJD.model_validate(jd_obj)
        jd_dict = jd_obj.model_dump()

        cv_context = state_dict.get("cv", {})
        if isinstance(cv_context, dict):
            cv_dict = cv_context
        else:
            cv_dict = cv_context.model_dump() if hasattr(cv_context, "model_dump") else {}

        client = instructor.from_openai(get_async_openai_client())

        all_evaluated: List[Dict] = []
        passed_ids: List[Any] = []
        rejected_ids: List[Any] = []
        failed_questions: List[Dict] = []

        for batch_start in range(0, len(merged_questions), BATCH_SIZE):
            batch_no = batch_start // BATCH_SIZE
            batch = merged_questions[batch_start : batch_start + BATCH_SIZE]

            # Score the batch, retrying once if any question is missing a score
            scores: Dict[int, QuestionScore] = {}
            last_error = None
            for _ in range(MAX_ATTEMPTS):
                try:
                    got = await _score_batch(client, batch, jd_dict, cv_dict)
                    for idx, s in got.items():
                        scores.setdefault(idx, s)
                except Exception as e:
                    last_error = str(e)
                    print(f"Error evaluating batch {batch_no}: {e}")
                if len(scores) == len(batch):
                    break

            # Save scored questions; record any that never got a score
            for idx, q in enumerate(batch, start=1):
                score_obj = scores.get(idx)
                if score_obj is None:
                    failed_questions.append({
                        "batch": batch_no,
                        "question_text": q.get("question_text"),
                        "reason": last_error or "judge returned no score for this question",
                    })
                    continue

                status = "candidate" if score_obj.pass_threshold else "rejected_auto"

                judge_scores = {
                    "clarity": score_obj.scores.clarity,
                    "relevance": score_obj.scores.relevance,
                    "difficulty_match": score_obj.scores.difficulty_match,
                    "groundedness": score_obj.scores.groundedness,
                    "redundancy": score_obj.scores.redundancy,
                    "overall": score_obj.overall_score,
                }

                question_id = save_question(
                    run_id=run_id,
                    question_data=q,
                    judge_scores=judge_scores,
                    judge_feedback=score_obj.feedback,
                    judge_pass=score_obj.pass_threshold,
                    status=status,
                    db_path=db_path,
                )

                if score_obj.pass_threshold:
                    passed_ids.append(question_id)
                else:
                    rejected_ids.append(question_id)

                all_evaluated.append({
                    "question_id": question_id,
                    "question_text": q.get("question_text"),
                    "scores": judge_scores,
                    "feedback": score_obj.feedback,
                    "status": status,
                    "rejection_reason": score_obj.rejection_reason,
                })

        # Calculate statistics
        passed_count = len(passed_ids)
        rejected_count = len(rejected_ids)
        total_count = len(all_evaluated)
        pass_rate = passed_count / total_count if total_count > 0 else 0

        if failed_questions:
            print(f"Judge: {len(failed_questions)} question(s) could not be scored")

        # Log completion
        update_pipeline_step(
            step_id,
            status="completed",
            output_json={
                "evaluated": total_count,
                "passed": passed_count,
                "rejected": rejected_count,
                "pass_rate": pass_rate,
                "failed_count": len(failed_questions),
                "failed_questions": failed_questions,
            },
            db_path=db_path,
        )

        return {
            "status": "running",
            "evaluated_questions": all_evaluated,
            "passed_count": passed_count,
            "rejected_count": rejected_count,
            "pass_rate": pass_rate,
            "error_message": None,
        }

    except Exception as e:
        error_msg = str(e)
        update_pipeline_step(
            step_id,
            status="failed",
            error_message=error_msg,
            db_path=db_path,
        )
        update_pipeline_run_status(
            run_id,
            status="failed",
            error_message=error_msg,
            db_path=db_path,
        )
        return {
            "status": "failed",
            "error_message": error_msg,
        }