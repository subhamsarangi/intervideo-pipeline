"""Tests for the web enrichment node (Phase 6)"""

import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.database.db import (
    init_db,
    create_cv,
    create_jd,
    create_pipeline_run,
    get_pipeline_steps,
    get_pipeline_run,
    update_cv_status,
    update_jd_status,
)
from src.models.cv import ParsedCV, CVProject, CVSkill
from src.models.jd import ParsedJD, JDRequirement
from src.models.pipeline import PipelineState, TriageResult, EnrichmentResult
from src.pipeline.nodes.enrichment import (
    enrichment_node,
    web_enrichment_node,
    _build_search_query,
    _sanitize_and_wrap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_db(tmp_path):
    """Temporary SQLite database for each test."""
    db_path = str(tmp_path / "test_enrichment.db")
    init_db(db_path)
    return db_path


@pytest.fixture
def sample_triage_result():
    """Triage result with two flagged items."""
    return TriageResult(
        ranked_projects=[],
        ranked_skills=[],
        flagged_for_enrichment=["LangGraph", "DeepSeek"],
    )


@pytest.fixture
def empty_triage_result():
    """Triage result with no flagged items."""
    return TriageResult(
        ranked_projects=[],
        ranked_skills=[],
        flagged_for_enrichment=[],
    )


@pytest.fixture
def mock_tavily_results():
    """Fake Tavily API response for two results."""
    return [
        {
            "url": "https://docs.langgraph.dev/intro",
            "title": "LangGraph Introduction",
            "content": "LangGraph is a library for building stateful multi-agent applications.",
        },
        {
            "url": "https://github.com/langgraph",
            "title": "LangGraph GitHub",
            "content": "Source code for the LangGraph framework.",
        },
    ]


def _make_run(test_db):
    """Helper: create CV, JD, and pipeline run, return run_id."""
    cv_id = create_cv("Test CV", "raw cv", test_db)
    update_cv_status(cv_id=cv_id, status="completed", parsed_data={}, db_path=test_db)
    jd_id = create_jd("Test JD", "raw jd", "pasted", None, test_db)
    update_jd_status(jd_id=jd_id, status="completed", parsed_data={}, db_path=test_db)
    run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)
    return run_id


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_build_search_query_with_title(self):
        q = _build_search_query("LangGraph", "Senior ML Engineer")
        assert "LangGraph" in q
        assert "Senior ML Engineer" in q

    def test_build_search_query_no_title(self):
        q = _build_search_query("vLLM")
        assert "vLLM" in q

    def test_sanitize_and_wrap_truncates(self):
        long_text = "x" * 2000
        result = _sanitize_and_wrap(long_text, "http://example.com")
        # Must be shorter than input and contain delimiter markers
        assert len(result) < len(long_text)
        assert "web_enrichment" in result.lower() or "<<<" in result or ">>>" in result

    def test_sanitize_and_wrap_labels_source(self):
        result = _sanitize_and_wrap("some content", "http://example.com/article")
        assert "example.com" in result


# ---------------------------------------------------------------------------
# Integration tests — enrichment node
# ---------------------------------------------------------------------------

class TestEnrichmentNode:
    @pytest.mark.asyncio
    async def test_skip_when_no_flagged_items(self, test_db, empty_triage_result):
        """Node should skip all API calls when nothing is flagged."""
        run_id = _make_run(test_db)

        result = await enrichment_node({
            "run_id": run_id,
            "db_path": test_db,
            "triage_result": empty_triage_result,
        })

        assert result["status"] == "running"
        enrichment = result["enrichment_result"]
        assert isinstance(enrichment, EnrichmentResult)
        assert enrichment.enrichment_context == {}
        assert enrichment.sources_used == []

        steps = get_pipeline_steps(run_id, db_path=test_db)
        assert len(steps) == 1
        assert steps[0]["step_name"] == "web_enrichment"
        assert steps[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_skip_when_no_triage_result(self, test_db):
        """Node should skip gracefully when triage_result is absent."""
        run_id = _make_run(test_db)

        result = await enrichment_node({
            "run_id": run_id,
            "db_path": test_db,
            # no triage_result key
        })

        assert result["status"] == "running"
        assert result["enrichment_result"].enrichment_context == {}

    @pytest.mark.asyncio
    async def test_calls_tavily_per_flagged_item(self, test_db, sample_triage_result, mock_tavily_results):
        """Node calls Tavily once per flagged item and collects context."""
        run_id = _make_run(test_db)

        with patch(
            "src.pipeline.nodes.enrichment._call_tavily",
            new_callable=AsyncMock,
            return_value=mock_tavily_results,
        ):
            result = await enrichment_node({
                "run_id": run_id,
                "db_path": test_db,
                "triage_result": sample_triage_result,
                "jd": {"job_title": "ML Engineer"},
            })

        assert result["status"] == "running"
        enrichment = result["enrichment_result"]
        assert "LangGraph" in enrichment.enrichment_context
        assert "DeepSeek" in enrichment.enrichment_context
        assert len(enrichment.sources_used) > 0

        steps = get_pipeline_steps(run_id, db_path=test_db)
        assert steps[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_dict_triage_input(self, test_db, mock_tavily_results):
        """Node accepts raw dict triage_result (deserialized from DB state)."""
        run_id = _make_run(test_db)

        with patch(
            "src.pipeline.nodes.enrichment._call_tavily",
            new_callable=AsyncMock,
            return_value=mock_tavily_results,
        ):
            result = await enrichment_node({
                "run_id": run_id,
                "db_path": test_db,
                "triage_result": {
                    "ranked_projects": [],
                    "ranked_skills": [],
                    "flagged_for_enrichment": ["Qdrant"],
                },
            })

        assert result["status"] == "running"
        assert "Qdrant" in result["enrichment_result"].enrichment_context

    @pytest.mark.asyncio
    async def test_per_item_tavily_failure_continues(self, test_db, sample_triage_result):
        """A failure on one flagged item should not abort the rest."""
        run_id = _make_run(test_db)

        call_count = 0

        async def fake_tavily(query):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("simulated network timeout")
            return [{"url": "http://ok.com", "content": "All good here."}]

        with patch("src.pipeline.nodes.enrichment._call_tavily", side_effect=fake_tavily):
            result = await enrichment_node({
                "run_id": run_id,
                "db_path": test_db,
                "triage_result": sample_triage_result,  # has LangGraph + DeepSeek
            })

        assert result["status"] == "running"
        # Should still complete — not fail
        steps = get_pipeline_steps(run_id, db_path=test_db)
        assert steps[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_missing_api_key_fails_run(self, test_db, sample_triage_result, monkeypatch):
        """Missing TAVILY_API_KEY should set run status to failed."""
        run_id = _make_run(test_db)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        result = await enrichment_node({
            "run_id": run_id,
            "db_path": test_db,
            "triage_result": sample_triage_result,
        })

        assert result["status"] == "failed"
        assert "TAVILY_API_KEY" in result["error_message"]

        run = get_pipeline_run(run_id, db_path=test_db)
        assert run["status"] == "failed"

    @pytest.mark.asyncio
    async def test_enrichment_results_wrapped_as_untrusted(self, test_db, mock_tavily_results):
        """Verify all enrichment snippets are wrapped in content delimiters."""
        run_id = _make_run(test_db)

        with patch(
            "src.pipeline.nodes.enrichment._call_tavily",
            new_callable=AsyncMock,
            return_value=mock_tavily_results,
        ):
            result = await enrichment_node({
                "run_id": run_id,
                "db_path": test_db,
                "triage_result": {
                    "ranked_projects": [],
                    "ranked_skills": [],
                    "flagged_for_enrichment": ["LangGraph"],
                },
            })

        context = result["enrichment_result"].enrichment_context.get("LangGraph", "")
        assert context != ""
        # Content must be wrapped — delimiters injected by _sanitize_and_wrap
        assert "Source:" in context or "<<<" in context or "web_enrichment" in context.lower()
