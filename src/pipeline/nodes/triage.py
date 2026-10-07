"""Relevance triage node for LangGraph pipeline using OpenAI embeddings"""

import json
import re
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np

from src.database.db import (
    get_cv,
    get_jd,
    create_pipeline_step,
    update_pipeline_step,
    update_pipeline_run_status,
)
from src.models.cv import ParsedCV, CVProject, CVSkill
from src.models.jd import ParsedJD, JDRequirement
from src.models.pipeline import PipelineState, TriageResult
from src.utils.embeddings import (
    get_embeddings_batch,
    cosine_similarity_matrix,
)

# Known fast-moving or niche technologies that warrant live web search enrichment
FAST_MOVING_PATTERNS = [
    r"\blangchain\b",
    r"\bllamaindex\b",
    r"\bllamaparse\b",
    r"\bcrewai\b",
    r"\blanggraph\b",
    r"\bautogen\b",
    r"\bvllm\b",
    r"\bollama\b",
    r"\bdspy\b",
    r"\binstructor\b",
    r"\bdeepseek\b",
    r"\bqdrant\b",
    r"\bchromadb\b",
    r"\bpinecone\b",
    r"\bweaviate\b",
    r"\bmilvus\b",
    r"\bnext\.?js\s*(?:1[4-9]|[2-9]\d)\b",
    r"\bsvelte\s*5\b",
    r"\bastro\b",
    r"\bhtmx\b",
    r"\bturborepo\b",
    r"\bbun\b",
    r"\bdeno\b",
    r"\bmojo\b",
    r"\brag\b",
    r"\bllm-as-a-judge\b",
    r"\bv\d+(\.\d+)+\b",
]


def detect_fast_moving_tech(text: str) -> List[str]:
    """
    Detect technologies in text that are fast-moving or recent and warrant web enrichment.

    Args:
        text: Text to scan

    Returns:
        List of detected technology keywords
    """
    if not text:
        return []

    found: Set[str] = set()
    for pattern in FAST_MOVING_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            found.add(match.group(0).strip())

    return sorted(list(found))


async def triage_node(state: Union[PipelineState, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Relevance Triage LangGraph node.

    1. Embeds all CV projects and skills.
    2. Embeds all JD requirements.
    3. Computes cosine similarity to rank projects and skills against JD.
    4. Identifies fast-moving/unfamiliar tech for web enrichment.
    5. Logs execution and results to database.
    6. Returns updated state dictionary.
    """
    if isinstance(state, PipelineState):
        state_dict = state.model_dump()
    else:
        state_dict = dict(state)

    run_id = state_dict.get("run_id")
    cv_id = state_dict.get("cv_id")
    jd_id = state_dict.get("jd_id")
    db_path = state_dict.get("db_path")

    if run_id is None:
        raise ValueError("run_id is required in pipeline state")
    if cv_id is None:
        raise ValueError("cv_id is required in pipeline state")
    if jd_id is None:
        raise ValueError("jd_id is required in pipeline state")

    # 1. Create pipeline step record
    step_id = create_pipeline_step(run_id, "relevance_triage", db_path=db_path)
    update_pipeline_step(step_id, status="running", db_path=db_path)

    try:
        # 2. Retrieve or parse CV and JD
        cv_obj = state_dict.get("cv")
        if cv_obj is None or not isinstance(cv_obj, (ParsedCV, dict)):
            cv_doc = get_cv(cv_id, db_path=db_path)
            if not cv_doc:
                raise ValueError(f"CV {cv_id} not found")
            parsed_data = cv_doc.get("parsed_data")
            if isinstance(parsed_data, str):
                parsed_data = json.loads(parsed_data)
            cv_obj = ParsedCV.model_validate(parsed_data)
        elif isinstance(cv_obj, dict):
            cv_obj = ParsedCV.model_validate(cv_obj)

        jd_obj = state_dict.get("jd")
        if jd_obj is None or not isinstance(jd_obj, (ParsedJD, dict)):
            jd_doc = get_jd(jd_id, db_path=db_path)
            if not jd_doc:
                raise ValueError(f"JD {jd_id} not found")
            parsed_data = jd_doc.get("parsed_data")
            if isinstance(parsed_data, str):
                parsed_data = json.loads(parsed_data)
            jd_obj = ParsedJD.model_validate(parsed_data)
        elif isinstance(jd_obj, dict):
            jd_obj = ParsedJD.model_validate(jd_obj)

        projects: List[CVProject] = cv_obj.projects or []
        skills: List[CVSkill] = cv_obj.skills or []
        requirements: List[JDRequirement] = jd_obj.requirements or []

        flagged_items: Set[str] = set()

        # 3. Construct text representations for embedding
        # JD texts (each requirement + overall JD context)
        jd_texts = []
        for req in requirements:
            req_str = f"{req.text} ({req.requirement_type})"
            if req.category:
                req_str += f" [{req.category}]"
            jd_texts.append(req_str)
            # Check JD requirements for fast-moving tech
            for tech in detect_fast_moving_tech(req.text):
                flagged_items.add(tech)

        if not jd_texts:
            # Fallback if no specific requirements extracted
            jd_texts = [f"{jd_obj.job_title} at {jd_obj.company or ''}: {jd_obj.summary or 'Job requirements'}"]

        # CV project texts
        proj_texts = []
        for i, p in enumerate(projects):
            tech_str = ", ".join(p.technologies) if p.technologies else ""
            desc = p.description or ""
            p_text = f"{p.title}: {desc}"
            if tech_str:
                p_text += f". Technologies: {tech_str}"
            proj_texts.append(p_text)

            # Check project for fast-moving tech
            for tech in detect_fast_moving_tech(p_text):
                flagged_items.add(tech)

        # CV skill texts
        skill_texts = []
        for i, s in enumerate(skills):
            s_text = s.skill_name
            if s.proficiency_level:
                s_text += f" ({s.proficiency_level})"
            if s.years_of_experience:
                s_text += f" {s.years_of_experience} years"
            skill_texts.append(s_text)

            # Check skill for fast-moving tech
            for tech in detect_fast_moving_tech(s_text):
                flagged_items.add(tech)

        # 4. Generate embeddings in batches
        all_texts_to_embed = jd_texts + proj_texts + skill_texts
        all_embeddings = await get_embeddings_batch(all_texts_to_embed)

        jd_len = len(jd_texts)
        proj_len = len(proj_texts)

        jd_embeddings = np.array(all_embeddings[:jd_len], dtype=np.float32)
        proj_embeddings = np.array(all_embeddings[jd_len : jd_len + proj_len], dtype=np.float32)
        skill_embeddings = np.array(all_embeddings[jd_len + proj_len :], dtype=np.float32)

        # 5. Calculate similarity scores and rank
        # Rank Projects
        ranked_projects = []
        if proj_len > 0 and jd_len > 0:
            sim_matrix_proj = cosine_similarity_matrix(proj_embeddings, jd_embeddings)
            # For each project, compute max relevance across JD requirements and top-2 mean
            for idx, proj in enumerate(projects):
                proj_sims = sim_matrix_proj[idx]
                max_score = float(np.max(proj_sims)) if proj_sims.size > 0 else 0.0
                ranked_projects.append(
                    {
                        "id": f"proj_{idx + 1}",
                        "title": proj.title,
                        "score": round(max_score, 4),
                        "project": proj.model_dump(),
                    }
                )
            # Sort by score descending
            ranked_projects.sort(key=lambda x: x["score"], reverse=True)

        # Rank Skills
        ranked_skills = []
        if len(skill_texts) > 0 and jd_len > 0:
            sim_matrix_skills = cosine_similarity_matrix(skill_embeddings, jd_embeddings)
            for idx, skill in enumerate(skills):
                skill_sims = sim_matrix_skills[idx]
                max_score = float(np.max(skill_sims)) if skill_sims.size > 0 else 0.0
                ranked_skills.append(
                    {
                        "id": f"skill_{idx + 1}",
                        "skill_name": skill.skill_name,
                        "score": round(max_score, 4),
                        "skill": skill.model_dump(),
                    }
                )
            # Sort by score descending
            ranked_skills.sort(key=lambda x: x["score"], reverse=True)

        # 6. Build TriageResult
        triage_result = TriageResult(
            ranked_projects=ranked_projects,
            ranked_skills=ranked_skills,
            flagged_for_enrichment=sorted(list(flagged_items)),
        )

        # 7. Log completion to pipeline_steps
        update_pipeline_step(
            step_id,
            status="completed",
            output_json=triage_result.model_dump(),
            db_path=db_path,
        )

        return {
            "status": "running",
            "cv": cv_obj,
            "jd": jd_obj,
            "triage_result": triage_result,
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


# Alias for explicit naming
relevance_triage_node = triage_node
