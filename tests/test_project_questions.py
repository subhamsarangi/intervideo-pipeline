"""Tests for project questions generation node"""

import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock

from src.database.db import (
    init_db,
    create_cv,
    create_jd,
    create_pipeline_run,
    get_pipeline_steps,
    update_cv_status,
    update_jd_status,
)
from src.models.cv import ParsedCV, CVProject, CVSkill
from src.models.jd import ParsedJD, JDRequirement
from src.models.pipeline import PipelineState, TriageResult
from src.pipeline.nodes.project_questions import project_questions_node


@pytest.fixture
def test_db():
    """Create temporary test database"""
    db_path = "test_project_questions.db"
    init_db(db_path)
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


class TestProjectQuestionsNode:
    @pytest.mark.asyncio
    async def test_basic_question_generation(self, test_db):
        """Test basic project question generation"""
        # Setup CV
        parsed_cv = ParsedCV(
            name="Alice Engineer",
            summary="Full stack dev",
            projects=[
                CVProject(
                    title="E-Commerce Platform",
                    description="Built scalable backend with microservices handling 10k RPM",
                    technologies=["Python", "FastAPI", "PostgreSQL", "Redis"],
                ),
                CVProject(
                    title="Data Pipeline",
                    description="ETL pipeline processing 1GB daily data with Airflow",
                    technologies=["Python", "Airflow", "Spark"],
                ),
            ],
            skills=[
                CVSkill(skill_name="FastAPI", proficiency_level="expert", years_of_experience=4.0),
            ],
        )

        parsed_jd = ParsedJD(
            job_title="Senior Backend Engineer",
            company="TechCorp",
            summary="Looking for backend engineer with microservices experience",
            requirements=[
                JDRequirement(text="5+ years Python", requirement_type="must_have"),
                JDRequirement(text="FastAPI/REST APIs", requirement_type="must_have"),
            ],
        )

        cv_id = create_cv("CV", "raw", test_db)
        update_cv_status(cv_id=cv_id, status="completed", parsed_data=parsed_cv.model_dump(), db_path=test_db)

        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        update_jd_status(jd_id=jd_id, status="completed", parsed_data=parsed_jd.model_dump(), db_path=test_db)

        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''[
            {"question_text": "How did you handle concurrency in the platform?", "relevance_note": "Probes technical depth"},
            {"question_text": "What was the scaling strategy?", "relevance_note": "Tests architecture knowledge"}
        ]'''
        mock_response.usage.prompt_tokens = 150
        mock_response.usage.completion_tokens = 50

        triage_result = TriageResult(
            ranked_projects=[
                {
                    "id": "proj_1",
                    "title": "E-Commerce Platform",
                    "score": 0.95,
                    "project": parsed_cv.projects[0].model_dump(),
                },
            ],
            ranked_skills=[],
            flagged_for_enrichment=[],
        )

        with patch(
            "src.pipeline.nodes.project_questions.get_async_openai_client"
        ) as mock_client:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            with patch(
                "src.pipeline.nodes.project_questions.increment_llm_calls"
            ) as mock_track:
                result = await project_questions_node({
                    "run_id": run_id,
                    "cv_id": cv_id,
                    "jd_id": jd_id,
                    "triage_result": triage_result,
                    "cv": parsed_cv,
                    "jd": parsed_jd,
                    "db_path": test_db,
                })

        assert result["status"] == "running"
        assert len(result["project_questions"]) == 2
        assert result["project_questions"][0]["question_text"] == "How did you handle concurrency in the platform?"
        assert result["project_questions"][0]["source_type"] == "project"
        assert result["project_questions"][0]["source_id"] == "proj_1"

        # Verify LLM tracking called
        mock_track.assert_called()

        # Verify DB step logged
        steps = get_pipeline_steps(run_id, db_path=test_db)
        assert any(s["step_name"] == "project_questions" for s in steps)

    @pytest.mark.asyncio
    async def test_empty_projects(self, test_db):
        """Test handling of empty project list"""
        parsed_cv = ParsedCV(name="Bob", summary="Dev", projects=[], skills=[])
        parsed_jd = ParsedJD(
            job_title="Backend Dev",
            requirements=[JDRequirement(text="Python", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        update_cv_status(cv_id=cv_id, status="completed", parsed_data=parsed_cv.model_dump(), db_path=test_db)

        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        update_jd_status(jd_id=jd_id, status="completed", parsed_data=parsed_jd.model_dump(), db_path=test_db)

        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        triage_result = TriageResult(ranked_projects=[], ranked_skills=[], flagged_for_enrichment=[])

        result = await project_questions_node({
            "run_id": run_id,
            "cv_id": cv_id,
            "jd_id": jd_id,
            "triage_result": triage_result,
            "cv": parsed_cv,
            "jd": parsed_jd,
            "db_path": test_db,
        })

        assert result["status"] == "running"
        assert len(result["project_questions"]) == 0
        assert result["error_message"] is None

    @pytest.mark.asyncio
    async def test_cost_tracking(self, test_db):
        """Test that LLM costs are tracked for each project"""
        parsed_cv = ParsedCV(
            name="Charlie",
            summary="Dev",
            projects=[
                CVProject(title="Project A", description="Desc A", technologies=["Python"]),
                CVProject(title="Project B", description="Desc B", technologies=["Go"]),
            ],
            skills=[],
        )

        parsed_jd = ParsedJD(
            job_title="Engineer",
            requirements=[JDRequirement(text="Backend", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        update_cv_status(cv_id=cv_id, status="completed", parsed_data=parsed_cv.model_dump(), db_path=test_db)

        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        update_jd_status(jd_id=jd_id, status="completed", parsed_data=parsed_jd.model_dump(), db_path=test_db)

        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[{"question_text": "Test?", "relevance_note": "test"}]'
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 100

        triage_result = TriageResult(
            ranked_projects=[
                {
                    "id": "proj_1",
                    "title": "Project A",
                    "score": 0.9,
                    "project": parsed_cv.projects[0].model_dump(),
                },
                {
                    "id": "proj_2",
                    "title": "Project B",
                    "score": 0.8,
                    "project": parsed_cv.projects[1].model_dump(),
                },
            ],
            ranked_skills=[],
            flagged_for_enrichment=[],
        )

        with patch("src.pipeline.nodes.project_questions.get_async_openai_client") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            with patch("src.pipeline.nodes.project_questions.increment_llm_calls") as mock_track:
                result = await project_questions_node({
                    "run_id": run_id,
                    "cv_id": cv_id,
                    "jd_id": jd_id,
                    "triage_result": triage_result,
                    "cv": parsed_cv,
                    "jd": parsed_jd,
                    "db_path": test_db,
                })

        # Should be called once per project (2 projects)
        assert mock_track.call_count == 2
        for call in mock_track.call_args_list:
            assert call.kwargs["operation"] == "project_questions_generation"
            assert call.kwargs["model"] == "gpt-5.6-luna"
            assert call.kwargs["input_tokens"] == 200
            assert call.kwargs["output_tokens"] == 100

    @pytest.mark.asyncio
    async def test_missing_triage_result(self, test_db):
        """Test error handling when triage_result missing"""
        parsed_cv = ParsedCV(name="Dave", summary="Dev", projects=[], skills=[])
        parsed_jd = ParsedJD(
            job_title="Engineer",
            requirements=[JDRequirement(text="Python", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        result = await project_questions_node({
            "run_id": run_id,
            "cv_id": cv_id,
            "jd_id": jd_id,
            "cv": parsed_cv,
            "jd": parsed_jd,
            "db_path": test_db,
        })

        assert result["status"] == "failed"
        assert "triage_result" in result["error_message"].lower()

    @pytest.mark.asyncio
    async def test_malformed_json_response(self, test_db):
        """Test graceful handling of LLM returning invalid JSON"""
        parsed_cv = ParsedCV(
            name="Eve",
            summary="Dev",
            projects=[
                CVProject(title="Project", description="Desc", technologies=["Python"]),
            ],
            skills=[],
        )

        parsed_jd = ParsedJD(
            job_title="Engineer",
            requirements=[JDRequirement(text="Python", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        update_cv_status(cv_id=cv_id, status="completed", parsed_data=parsed_cv.model_dump(), db_path=test_db)

        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        update_jd_status(jd_id=jd_id, status="completed", parsed_data=parsed_jd.model_dump(), db_path=test_db)

        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Not valid JSON at all!"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        triage_result = TriageResult(
            ranked_projects=[
                {
                    "id": "proj_1",
                    "title": "Project",
                    "score": 0.9,
                    "project": parsed_cv.projects[0].model_dump(),
                },
            ],
            ranked_skills=[],
            flagged_for_enrichment=[],
        )

        with patch("src.pipeline.nodes.project_questions.get_async_openai_client") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await project_questions_node({
                "run_id": run_id,
                "cv_id": cv_id,
                "jd_id": jd_id,
                "triage_result": triage_result,
                "cv": parsed_cv,
                "jd": parsed_jd,
                "db_path": test_db,
            })

        # Should return completed status with empty questions (graceful failure)
        assert result["status"] == "running"
        assert len(result["project_questions"]) == 0
        assert result["error_message"] is None
