"""Merge, deduplicate, and rerank questions from project and skill nodes"""

import json
from typing import Any, Dict, List, Optional, Union

from src.database.db import (
    create_pipeline_step,
    update_pipeline_step,
    update_pipeline_run_status,
)
from src.models.jd import ParsedJD, JDRequirement
from src.models.pipeline import PipelineState, GeneratedQuestion, MergedQuestionSet
from src.utils.embeddings import get_async_openai_client, cosine_similarity_matrix
from src.utils.llm_tracking import increment_llm_calls
import numpy as np


DEDUP_SYSTEM_PROMPT = """You are an expert at identifying semantically similar questions.

Your task: Given a list of interview questions, identify which questions are semantically similar (asking roughly the same thing).

Return JSON array with groups of similar questions:
[
  {
    "group_id": 0,
    "questions": [0, 1, 3],  // indices of similar questions
    "representative": "Best question text to keep",
    "reason": "Why these are similar"
  }
]

Keep only one question per group. Choose the most specific/valuable one.
Return ONLY JSON, no markdown.
"""


async def merge_dedupe_node(
    state: Union[PipelineState, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge, deduplicate, and rerank questions.

    Combines project_questions + skill_questions, removes duplicates via LLM clustering,
    checks coverage of JD requirements, reranks by relevance.

    Args:
        state: PipelineState or dict with project_questions, skill_questions, jd, etc.

    Returns:
        Dict with status, merged_questions, coverage_gaps, and error_message
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
    step_id = create_pipeline_step(run_id, "merge_dedupe", db_path=db_path)
    update_pipeline_step(step_id, status="running", db_path=db_path)

    try:
        # Get questions from both nodes
        project_questions: List[Dict] = state_dict.get("project_questions", [])
        skill_questions: List[Dict] = state_dict.get("skill_questions", [])

        # Combine
        all_questions = project_questions + skill_questions

        if not all_questions:
            update_pipeline_step(
                step_id,
                status="completed",
                output_json={"merged_count": 0, "coverage_gaps": []},
                db_path=db_path,
            )
            return {
                "status": "running",
                "merged_questions": [],
                "coverage_gaps": [],
                "error_message": None,
            }

        # Deduplicate via LLM semantic clustering
        deduplicated = await _deduplicate_questions(all_questions)

        # Get JD for coverage check
        jd_obj = state_dict.get("jd")
        if jd_obj is None:
            raise ValueError("jd required in state")
        if isinstance(jd_obj, dict):
            jd_obj = ParsedJD.model_validate(jd_obj)

        # Check coverage (which requirements have questions)
        coverage_gaps = _check_coverage(deduplicated, jd_obj)

        # Rerank by relevance to JD
        reranked = await _rerank_questions(deduplicated, jd_obj)

        # Log completion
        update_pipeline_step(
            step_id,
            status="completed",
            output_json={
                "combined": len(all_questions),
                "deduplicated": len(deduplicated),
                "coverage_gaps": len(coverage_gaps),
            },
            db_path=db_path,
        )

        return {
            "status": "running",
            "merged_questions": [q.model_dump() if isinstance(q, GeneratedQuestion) else q for q in reranked],
            "coverage_gaps": coverage_gaps,
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


async def _deduplicate_questions(questions: List[Dict]) -> List[GeneratedQuestion]:
    """
    Deduplicate questions using LLM semantic clustering.

    Groups similar questions, keeps best from each group.
    """
    if len(questions) <= 1:
        return [GeneratedQuestion.model_validate(q) if isinstance(q, dict) else q for q in questions]

    # Build question texts for clustering
    question_texts = [q.get("question_text", "") for q in questions]

    # Use embeddings for initial clustering
    client = get_async_openai_client()
    embeddings_response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=question_texts,
    )

    embeddings = [item.embedding for item in embeddings_response.data]
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Track LLM call
    if hasattr(embeddings_response, "usage") and embeddings_response.usage:
        increment_llm_calls(
            operation="merge_dedupe_embeddings",
            model="text-embedding-3-small",
            input_tokens=embeddings_response.usage.prompt_tokens,
            output_tokens=0,
        )

    # Compute similarity matrix
    sim_matrix = cosine_similarity_matrix(embeddings_array, embeddings_array)

    # Simple clustering: questions with similarity > 0.85 are similar
    kept_indices = set(range(len(questions)))
    groups = []

    for i in range(len(questions)):
        if i not in kept_indices:
            continue

        similar = [i]
        for j in range(i + 1, len(questions)):
            if j in kept_indices and sim_matrix[i][j] > 0.85:
                similar.append(j)

        if len(similar) > 1:
            # Mark duplicates as removed, keep first
            for j in similar[1:]:
                kept_indices.discard(j)
            groups.append(similar)

    # Return kept questions
    deduplicated = [
        GeneratedQuestion.model_validate(questions[i]) if isinstance(questions[i], dict) else questions[i]
        for i in sorted(kept_indices)
    ]

    return deduplicated


def _check_coverage(questions: List[GeneratedQuestion], jd: ParsedJD) -> List[Dict]:
    """
    Check which JD requirements have corresponding questions.

    Returns list of uncovered requirements.
    """
    if not questions or not jd.requirements:
        return []

    # Simple check: requirement is covered if any question mentions key terms from requirement
    covered_requirements = set()
    uncovered = []

    for req in jd.requirements:
        req_text = req.text.lower()
        req_keywords = set(req_text.split())

        # Check if any question addresses this requirement
        for q in questions:
            q_text = q.question_text.lower()
            # Simple keyword overlap check
            if sum(1 for kw in req_keywords if kw in q_text) >= 2:  # At least 2 keywords match
                covered_requirements.add(req.text)
                break

        if req.text not in covered_requirements:
            uncovered.append({
                "requirement": req.text,
                "type": req.requirement_type,
                "reason": "No questions address this requirement"
            })

    return uncovered


async def _rerank_questions(questions: List[GeneratedQuestion], jd: ParsedJD) -> List[GeneratedQuestion]:
    """
    Rerank questions by relevance to JD.

    Uses cosine similarity between question and JD summary.
    """
    if len(questions) <= 1:
        return questions

    # Get embedding for JD summary
    client = get_async_openai_client()

    jd_text = f"{jd.job_title}. {jd.summary}. Requirements: {' '.join(r.text for r in jd.requirements)}"

    embeddings_response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=[jd_text] + [q.question_text for q in questions],
    )

    embeddings = [item.embedding for item in embeddings_response.data]

    # Track LLM call
    if hasattr(embeddings_response, "usage") and embeddings_response.usage:
        increment_llm_calls(
            operation="merge_dedupe_rerank",
            model="text-embedding-3-small",
            input_tokens=embeddings_response.usage.prompt_tokens,
            output_tokens=0,
        )

    jd_embedding = np.array(embeddings[0], dtype=np.float32)
    question_embeddings = np.array(embeddings[1:], dtype=np.float32)

    # Compute similarity to JD
    similarities = cosine_similarity_matrix(question_embeddings, jd_embedding.reshape(1, -1)).flatten()

    # Sort questions by relevance (descending)
    ranked_indices = np.argsort(-similarities)
    reranked = [questions[i] for i in ranked_indices]

    return reranked
