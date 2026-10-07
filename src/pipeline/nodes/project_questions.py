"""Project questions generation node for LangGraph pipeline"""

import json
from typing import Any, Dict, List, Optional, Union

from src.database.db import (
    create_pipeline_step,
    update_pipeline_step,
    update_pipeline_run_status,
)
from src.models.cv import ParsedCV
from src.models.jd import ParsedJD
from src.models.pipeline import PipelineState, GeneratedQuestion, TriageResult
from src.utils.embeddings import get_async_openai_client
from src.utils.llm_tracking import increment_llm_calls


SYSTEM_PROMPT = """You are an expert technical interviewer. Your task is to generate insightful technical questions based on a candidate's projects and a job description.

Requirements:
1. Generate 3-5 targeted questions per project
2. Questions should probe technical depth, decision-making, and relevance to the JD
3. Each question should be answerable from the project description
4. Format output as valid JSON array with this structure:
   [
     {
       "question_text": "...",
       "relevance_note": "..."
     }
   ]
5. Return ONLY the JSON array, no markdown formatting or extra text
"""


async def project_questions_node(
    state: Union[PipelineState, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate questions from ranked projects.

    Takes ranked projects from triage_result and JD, generates targeted interview questions.
    Logs execution and results to database.

    Args:
        state: PipelineState or dict with run_id, cv_id, jd_id, triage_result, cv, jd

    Returns:
        Dict with status, project_questions list, and error_message
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
    step_id = create_pipeline_step(run_id, "project_questions", db_path=db_path)
    update_pipeline_step(step_id, status="running", db_path=db_path)

    try:
        # Get triage result (ranked projects)
        triage_result: Optional[TriageResult] = state_dict.get("triage_result")
        if triage_result is None:
            raise ValueError("triage_result required in state")

        if isinstance(triage_result, dict):
            triage_result = TriageResult.model_validate(triage_result)

        ranked_projects = triage_result.ranked_projects or []

        # Get JD
        jd_obj = state_dict.get("jd")
        if jd_obj is None:
            raise ValueError("jd required in state")
        if isinstance(jd_obj, dict):
            jd_obj = ParsedJD.model_validate(jd_obj)

        # If no ranked projects, return empty
        if not ranked_projects:
            update_pipeline_step(
                step_id,
                status="completed",
                output_json={"questions": []},
                db_path=db_path,
            )
            return {
                "status": "running",
                "project_questions": [],
                "error_message": None,
            }

        # Generate questions for each project
        questions: List[GeneratedQuestion] = []
        client = get_async_openai_client()

        for proj_idx, proj_data in enumerate(ranked_projects):
            proj_obj = proj_data.get("project")
            if not proj_obj:
                continue

            # Build user prompt for this project
            project_prompt = f"""
Project Title: {proj_obj.get('title', 'N/A')}
Description: {proj_obj.get('description', 'N/A')}
Technologies: {', '.join(proj_obj.get('technologies', []))}

Job Role: {jd_obj.job_title}
Company: {jd_obj.company or 'N/A'}
Job Summary: {jd_obj.summary or 'N/A'}

Generate 3-5 targeted interview questions about this project that would help assess the candidate's fit for the role.
"""

            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": project_prompt},
                    ],
                    temperature=0.7,
                    top_p=0.9,
                )

                # Track LLM call
                if hasattr(response, "usage") and response.usage:
                    increment_llm_calls(
                        operation="project_questions_generation",
                        model="gpt-4o-mini",
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                    )

                # Parse response
                result_text = response.choices[0].message.content
                questions_data = json.loads(result_text)

                # Convert to GeneratedQuestion objects
                for q_data in questions_data:
                    gen_q = GeneratedQuestion(
                        question_text=q_data.get("question_text", ""),
                        source_type="project",
                        source_id=proj_data.get("id"),
                        evidence_snippet=proj_obj.get("description", ""),
                    )
                    questions.append(gen_q)

            except json.JSONDecodeError:
                # LLM didn't return valid JSON, log but continue
                print(f"Failed to parse JSON for project {proj_idx}: {result_text[:200]}")
            except Exception as e:
                print(f"Error generating questions for project {proj_idx}: {e}")

        # Log completion
        update_pipeline_step(
            step_id,
            status="completed",
            output_json={"question_count": len(questions)},
            db_path=db_path,
        )

        return {
            "status": "running",
            "project_questions": [q.model_dump() for q in questions],
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
