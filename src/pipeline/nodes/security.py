"""Security validation node for LangGraph pipeline"""

import json
from typing import Any, Dict, Optional, Union
from src.database.db import (
    get_cv,
    get_jd,
    create_pipeline_step,
    update_pipeline_step,
    update_pipeline_run_status,
)
from src.models.cv import ParsedCV
from src.models.jd import ParsedJD
from src.models.pipeline import PipelineState
from src.utils.security import (
    validate_cv_text,
    validate_jd_text,
    validate_jd_url,
    wrap_content_delimiters,
)


def security_node(state: Union[PipelineState, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Security validation node for pipeline.

    1. Loads CV and JD from database using run state.
    2. Runs prompt injection and SSRF validation.
    3. Wraps content with delimiters.
    4. Logs execution to pipeline_steps table.
    5. Returns updated state dictionary.
    """
    if isinstance(state, PipelineState):
        state_dict = state.model_dump()
    else:
        state_dict = dict(state)

    print("   → security: validating CV/JD...", flush=True)

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
    step_id = create_pipeline_step(run_id, "security_validation", db_path=db_path)
    update_pipeline_step(step_id, status="running", db_path=db_path)

    try:
        # 2. Load CV and JD from database
        cv_doc = get_cv(cv_id, db_path=db_path)
        if not cv_doc:
            raise ValueError(f"CV with id {cv_id} not found in database")

        jd_doc = get_jd(jd_id, db_path=db_path)
        if not jd_doc:
            raise ValueError(f"JD with id {jd_id} not found in database")

        cv_raw_text = cv_doc.get("raw_text") or ""
        jd_raw_text = jd_doc.get("raw_text") or ""
        jd_source_type = jd_doc.get("source_type")
        jd_source_url = jd_doc.get("source_url")

        # 3. Run validation checks
        print(f"      → checking {len(cv_raw_text)} chars CV for injection patterns...", flush=True)
        validate_cv_text(cv_raw_text)
        print(f"      → checking {len(jd_raw_text)} chars JD for injection patterns...", flush=True)
        validate_jd_text(jd_raw_text)

        if jd_source_type == "url" and jd_source_url:
            print(f"      → validating JD URL: {jd_source_url}", flush=True)
            validate_jd_url(jd_source_url)

        # 4. Wrap content in delimiters
        print(f"      → wrapping CV & JD in security delimiters...", flush=True)
        wrapped_cv = wrap_content_delimiters(cv_raw_text, "cv")
        wrapped_jd = wrap_content_delimiters(jd_raw_text, "jd")

        # 5. Parse structured data if available in DB
        parsed_cv_data = cv_doc.get("parsed_data")
        parsed_jd_data = jd_doc.get("parsed_data")

        if isinstance(parsed_cv_data, str):
            parsed_cv_data = json.loads(parsed_cv_data)
        if isinstance(parsed_jd_data, str):
            parsed_jd_data = json.loads(parsed_jd_data)

        parsed_cv = ParsedCV.model_validate(parsed_cv_data) if parsed_cv_data else None
        parsed_jd = ParsedJD.model_validate(parsed_jd_data) if parsed_jd_data else None

        # 6. Log success to pipeline_steps
        step_output = {
            "cv_valid": True,
            "jd_valid": True,
            "cv_id": cv_id,
            "jd_id": jd_id,
            "wrapped_cv_len": len(wrapped_cv),
            "wrapped_jd_len": len(wrapped_jd),
        }
        update_pipeline_step(
            step_id,
            status="completed",
            output_json=step_output,
            db_path=db_path,
        )

        return {
            "cv": parsed_cv,
            "jd": parsed_jd,
            "wrapped_cv": wrapped_cv,
            "wrapped_jd": wrapped_jd,
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


# Alias for explicit naming
security_validation_node = security_node
