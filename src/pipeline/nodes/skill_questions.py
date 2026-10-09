"""Skill questions generation node for LangGraph pipeline with Instructor"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel
import instructor

from src.database.db import (
    create_pipeline_step,
    update_pipeline_step,
    update_pipeline_run_status,
)
from src.models.jd import ParsedJD
from src.models.pipeline import PipelineState, TriageResult
from src.models.structured_question import StructuredQuestion, ProficiencyLevel
from src.utils.embeddings import get_async_openai_client
from src.utils.llm_tracking import increment_llm_calls
from src.utils.question_prompts import SKILL_QUESTIONS_SYSTEM_PROMPT, get_skill_user_prompt

# Hard limit: only generate questions for top N skills
MAX_SKILLS_TO_PROCESS = 5


class SkillQuestionsResponse(BaseModel):
    """Response model for skill questions"""
    questions: List[StructuredQuestion]


async def skill_questions_node(
    state: Union[PipelineState, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate structured questions from ranked skills using Instructor.

    Takes ranked skills from triage_result and JD, generates targeted interview questions
    with proficiency levels, key points, and follow-ups.
    
    Args:
        state: PipelineState or dict with run_id, cv_id, jd_id, triage_result, jd

    Returns:
        Dict with status, skill_questions list, error_message, level_distribution
    """
    if isinstance(state, PipelineState):
        state_dict = state.model_dump()
    else:
        state_dict = dict(state)

    print("   → skill_questions: generating skill-based questions...", flush=True)

    run_id = state_dict.get("run_id")
    db_path = state_dict.get("db_path")

    if run_id is None:
        raise ValueError("run_id is required in pipeline state")

    # Create pipeline step record
    step_id = create_pipeline_step(run_id, "skill_questions", db_path=db_path)
    update_pipeline_step(step_id, status="running", db_path=db_path)

    try:
        # Get triage result (ranked skills)
        triage_result: Optional[TriageResult] = state_dict.get("triage_result")
        if triage_result is None:
            raise ValueError("triage_result required in state")

        if isinstance(triage_result, dict):
            triage_result = TriageResult.model_validate(triage_result)

        ranked_skills = triage_result.ranked_skills or []

        # Get JD
        jd_obj = state_dict.get("jd")
        if jd_obj is None:
            raise ValueError("jd required in state")
        if isinstance(jd_obj, dict):
            jd_obj = ParsedJD.model_validate(jd_obj)

        # If no ranked skills, return empty
        if not ranked_skills:
            update_pipeline_step(
                step_id,
                status="completed",
                output_json={"questions": [], "level_distribution": {}},
                db_path=db_path,
            )
            return {
                "status": "running",
                "skill_questions": [],
                "level_distribution": {},
                "error_message": None,
            }

        # Hard limit: only process top N skills
        skills_to_process = ranked_skills[:MAX_SKILLS_TO_PROCESS]
        if len(ranked_skills) > MAX_SKILLS_TO_PROCESS:
            print(f"      → processing only top {MAX_SKILLS_TO_PROCESS}/{len(ranked_skills)} skills (cost optimization)", flush=True)

        # Generate questions for each skill using Instructor
        all_questions: List[StructuredQuestion] = []
        client = get_async_openai_client()
        
        # Wrap client with Instructor
        client = instructor.from_openai(client)

        print(f"      → calling gpt-5.4-nano for {len(skills_to_process)} skills...", flush=True)

        for skill_idx, skill_data in enumerate(skills_to_process):
            skill_obj = skill_data.get("skill")
            if not skill_obj:
                continue

            # Get user prompt
            user_prompt = get_skill_user_prompt(skill_obj, jd_obj.model_dump())

            try:
                # Call with Instructor for structured output
                print(f"         → [{skill_idx+1}/{len(skills_to_process)}] {skill_obj.get('skill_name', 'Skill')}...", flush=True)
                response = await client.chat.completions.create(
                    model="gpt-5.4-nano",
                    messages=[
                        {"role": "system", "content": SKILL_QUESTIONS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_model=SkillQuestionsResponse,
                    temperature=0.7,
                    top_p=0.9,
                )

                # Track LLM call
                if hasattr(response, "usage") and response.usage:
                    print(f"            {response.usage.prompt_tokens}+{response.usage.completion_tokens} tokens", flush=True)
                    increment_llm_calls(
                        operation="skill_questions_generation",
                        model="gpt-5.4-nano",
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                    )

                # Add source info to each question
                for q in response.questions:
                    q.source_type = "skill"
                    q.source_id = skill_data.get("id")
                    all_questions.append(q)
                
                print(f"            generated {len(response.questions)} questions", flush=True)

            except Exception as e:
                print(f"Error generating questions for skill {skill_idx} ({skill_obj.get('skill_name')}): {e}")

        # Calculate level distribution
        level_dist = {
            "junior": sum(1 for q in all_questions if q.level == ProficiencyLevel.JUNIOR),
            "mid": sum(1 for q in all_questions if q.level == ProficiencyLevel.MID),
            "senior": sum(1 for q in all_questions if q.level == ProficiencyLevel.SENIOR),
        }

        print(f"      → total: {len(all_questions)} questions ({level_dist['junior']}j / {level_dist['mid']}m / {level_dist['senior']}s)", flush=True)

        # Log completion
        update_pipeline_step(
            step_id,
            status="completed",
            output_json={
                "question_count": len(all_questions),
                "level_distribution": level_dist
            },
            db_path=db_path,
        )

        return {
            "skill_questions": [q.model_dump() for q in all_questions],
            "level_distribution": level_dist,
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
            "skill_questions": [q.model_dump() for q in all_questions],
            "level_distribution": level_dist,
        }
