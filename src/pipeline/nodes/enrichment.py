"""Web enrichment node for LangGraph pipeline using Tavily search API"""

import os
from typing import Any, Dict, List, Optional, Union

from src.database.db import (
    create_pipeline_step,
    update_pipeline_step,
    update_pipeline_run_status,
)
from src.models.pipeline import PipelineState, EnrichmentResult, TriageResult
from src.utils.llm_tracking import increment_llm_calls
from src.utils.security import wrap_content_delimiters, validate_cv_text


# Max characters to keep per source snippet to avoid bloating context
MAX_SNIPPET_CHARS = 800

# Max Tavily search results to consume per flagged item
MAX_RESULTS_PER_QUERY = 3


def _build_search_query(item: str, jd_title: str = "") -> str:
    """
    Construct a focused Tavily search query for a flagged tech/keyword.

    Args:
        item: The flagged technology or keyword.
        jd_title: Job title for additional context.

    Returns:
        A targeted search query string.
    """
    base = f"{item} latest features best practices"
    if jd_title:
        base += f" for {jd_title}"
    return base


def _sanitize_and_wrap(text: str, source_label: str) -> str:
    """
    Truncate, strip, and wrap enrichment text in content delimiters.
    Treats search results as untrusted — same treatment as CV/JD text.

    Args:
        text: Raw snippet text from Tavily.
        source_label: Label identifying the source (URL or title).

    Returns:
        Delimiter-wrapped, truncated text.
    """
    truncated = text[:MAX_SNIPPET_CHARS].strip()
    labelled = f"[Source: {source_label}]\n{truncated}"
    return wrap_content_delimiters(labelled, source_name="web_enrichment")


async def _call_tavily(query: str) -> List[Dict[str, Any]]:
    """
    Execute a Tavily search and return result list.

    Args:
        query: Search query string.

    Returns:
        List of result dicts with keys: title, url, content.

    Raises:
        ImportError: If tavily-python is not installed.
        RuntimeError: If TAVILY_API_KEY is missing.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY environment variable is not set")

    try:
        from tavily import AsyncTavilyClient
    except ImportError:
        raise ImportError(
            "tavily-python package is required for web enrichment. "
            "Install with: uv add tavily-python"
        )

    client = AsyncTavilyClient(api_key=api_key)
    response = await client.search(
        query=query,
        max_results=MAX_RESULTS_PER_QUERY,
        search_depth="basic",
    )
    return response.get("results", [])


async def enrichment_node(
    state: Union[PipelineState, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Conditional Web Enrichment LangGraph node.

    1. Reads flagged_for_enrichment from triage result.
    2. Skips entirely if no items were flagged (zero cost).
    3. For each flagged item, fires a targeted Tavily search.
    4. Treats all results as untrusted: truncates, validates, wraps in delimiters.
    5. Tracks per-call Tavily cost ($0.005/call) via llm_tracking.
    6. Logs execution to pipeline_steps table.
    7. Returns enrichment context keyed by flagged item.
    """
    if isinstance(state, PipelineState):
        state_dict = state.model_dump()
    else:
        state_dict = dict(state)

    run_id = state_dict.get("run_id")
    db_path = state_dict.get("db_path")

    if run_id is None:
        raise ValueError("run_id is required in pipeline state")

    step_id = create_pipeline_step(run_id, "web_enrichment", db_path=db_path)
    update_pipeline_step(step_id, status="running", db_path=db_path)

    try:
        # 1. Read triage result to get flagged items
        triage_raw = state_dict.get("triage_result")
        if triage_raw is None:
            # No triage result — skip enrichment gracefully
            enrichment_result = EnrichmentResult(
                enrichment_context={},
                sources_used=[],
            )
            update_pipeline_step(
                step_id,
                status="completed",
                output_json={"skipped": True, "reason": "no_triage_result"},
                db_path=db_path,
            )
            return {
                "status": state_dict.get("status", "running"),
                "enrichment_result": enrichment_result,
                "error_message": None,
            }

        if isinstance(triage_raw, TriageResult):
            flagged_items = triage_raw.flagged_for_enrichment
        elif isinstance(triage_raw, dict):
            flagged_items = triage_raw.get("flagged_for_enrichment", [])
        else:
            flagged_items = []

        # 2. Skip if no items flagged — zero API calls, zero cost
        if not flagged_items:
            enrichment_result = EnrichmentResult(
                enrichment_context={},
                sources_used=[],
            )
            update_pipeline_step(
                step_id,
                status="completed",
                output_json={"skipped": True, "reason": "no_flagged_items"},
                db_path=db_path,
            )
            return {
                "status": state_dict.get("status", "running"),
                "enrichment_result": enrichment_result,
                "error_message": None,
            }

        # 3. Retrieve job title for better query context
        jd_obj = state_dict.get("jd")
        jd_title = ""
        if isinstance(jd_obj, dict):
            jd_title = jd_obj.get("job_title", "")
        elif hasattr(jd_obj, "job_title"):
            jd_title = jd_obj.job_title or ""

        # 4. Search each flagged item
        enrichment_context: Dict[str, str] = {}
        sources_used: List[str] = []
        search_errors: List[str] = []

        for item in flagged_items:
            query = _build_search_query(item, jd_title)
            try:
                results = await _call_tavily(query)

                # Track Tavily cost — $0.005 per call
                increment_llm_calls(
                    operation=f"tavily_search:{item}",
                    model="tavily-search",
                )

                if not results:
                    enrichment_context[item] = ""
                    continue

                # 5. Build sanitized context from top results
                snippets = []
                for r in results[:MAX_RESULTS_PER_QUERY]:
                    url = r.get("url", "unknown")
                    content = r.get("content", r.get("snippet", ""))
                    if content:
                        # Validate snippet for injection before storing
                        if validate_cv_text(content):
                            snippets.append(_sanitize_and_wrap(content, url))
                            sources_used.append(url)

                enrichment_context[item] = "\n\n".join(snippets)

            except RuntimeError as e:
                # Missing API key — abort enrichment node entirely
                raise
            except Exception as e:
                # Per-item search failure — log and continue
                search_errors.append(f"{item}: {e}")
                enrichment_context[item] = ""

        enrichment_result = EnrichmentResult(
            enrichment_context=enrichment_context,
            sources_used=sources_used,
        )

        step_output = enrichment_result.model_dump()
        if search_errors:
            step_output["search_errors"] = search_errors

        update_pipeline_step(
            step_id,
            status="completed",
            output_json=step_output,
            db_path=db_path,
        )

        return {
            "status": state_dict.get("status", "running"),
            "enrichment_result": enrichment_result,
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
web_enrichment_node = enrichment_node
