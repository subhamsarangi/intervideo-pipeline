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

    print("   → judge_evaluation: scoring & filtering questions...", flush=True)

    run_id = state_dict.get("run_id")
    db_path = state_dict.get("db_path")

    if run_id is None:
        raise ValueError("run_id is required in pipeline state")

    # Create pipeline step record
    step_id = create_pipeline_step(run_id, "judge_evaluation", db_path=db_path)
    update_pipeline_step(step_id, status="running", db_path=db_path)

    try:
        # Get merged questions
        merged_qs = state_dict.get("merged_questions")
        if merged_qs is None or (isinstance(merged_qs, dict) and not merged_qs.get("questions")):
            merged_questions = []
        elif isinstance(merged_qs, dict):
            merged_questions = merged_qs.get("questions", [])
        elif hasattr(merged_qs, "questions"):
            merged_questions = merged_qs.questions
        else:
            merged_questions = []

        if not merged_questions:
            print(f"      → no questions to evaluate", flush=True)
            update_pipeline_step(
                step_id,
                status="completed",
                output_json={"evaluated": 0, "passed": 0, "rejected": 0},
                db_path=db_path,
            )
            return {
                "evaluated_questions": [],
                "passed_count": 0,
                "rejected_count": 0,
            }

        print(f"      → evaluating {len(merged_questions)} questions in batches of {BATCH_SIZE}...", flush=True)

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
            
            # Convert objects to dicts for prompt building
            batch_dicts = [q.model_dump() if hasattr(q, 'model_dump') else q for q in batch]

            print(f"         → batch {batch_no+1}/{(len(merged_questions) + BATCH_SIZE - 1) // BATCH_SIZE}: {len(batch)} questions...", flush=True)

            # Score the batch, retrying once if any question is missing a score
            scores: Dict[int, QuestionScore] = {}
            last_error = None
            for _ in range(MAX_ATTEMPTS):
                try:
                    got = await _score_batch(client, batch_dicts, jd_dict, cv_dict)
                    print(f"            scored {len(got)}/{len(batch)}", flush=True)
                    for idx, s in got.items():
                        scores.setdefault(idx, s)
                except Exception as e:
                    last_error = str(e)
                    print(f"            error: {e}", flush=True)
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

        print(f"      → final: {total_count} evaluated | {passed_count} passed ({pass_rate*100:.1f}%) | {rejected_count} rejected", flush=True)

        if failed_questions:
            print(f"         ⚠️  {len(failed_questions)} questions could not be scored", flush=True)

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
            "evaluated_questions": all_evaluated,
            "passed_count": passed_count,
            "rejected_count": rejected_count,
            "pass_rate": pass_rate,
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
            "error_message": error_msg,
        }