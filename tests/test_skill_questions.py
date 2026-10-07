"""Tests for skill questions generation node"""

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
from src.models.cv import ParsedCV, CVSkill
from src.models.jd import ParsedJD, JDRequirement
from src.models.pipeline import PipelineState, TriageResult
from src.pipeline.nodes.skill_questions import skill_questions_node


@pytest.fixture
def test_db():
    """Create temporary test database"""
    db_path = "test_skill_questions.db"
    init_db(db_path)
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


class TestSkillQuestionsNode:
    @pytest.mark.asyncio
    async def test_basic_question_generation(self, test_db):
        """Test basic skill question generation"""
        parsed_cv = ParsedCV(
            name="Bob Developer",
            summary="Backend engineer",
            projects=[],
            skills=[
                CVSkill(skill_name="Python", proficiency_level="expert", years_of_experience=6.0),
                CVSkill(skill_name="PostgreSQL", proficiency_level="advanced", years_of_experience=5.0),
                CVSkill(skill_name="FastAPI", proficiency_level="expert", years_of_experience=4.0),
            ],
        )

        parsed_jd = ParsedJD(
            job_title="Senior Backend Engineer",
            company="TechCorp",
            summary="Looking for Python expert",
            requirements=[
                JDRequirement(text="5+ years Python", requirement_type="must_have"),
                JDRequirement(text="PostgreSQL expertise", requirement_type="must_have"),
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
            {"question_text": "How have you optimized Python performance in high-throughput systems?", "relevance_note": "Probes deep experience"},
            {"question_text": "Can you describe your experience with async/await patterns?", "relevance_note": "Tests async knowledge"}
        ]'''
        mock_response.usage.prompt_tokens = 120
        mock_response.usage.completion_tokens = 40

        triage_result = TriageResult(
            ranked_projects=[],
            ranked_skills=[
                {
                    "id": "skill_1",
                    "skill_name": "Python",
                    "score": 0.98,
                    "skill": parsed_cv.skills[0].model_dump(),
                },
                {
                    "id": "skill_2",
                    "skill_name": "PostgreSQL",
                    "score": 0.96,
                    "skill": parsed_cv.skills[1].model_dump(),
                },
            ],
            flagged_for_enrichment=[],
        )

        with patch("src.pipeline.nodes.skill_questions.get_async_openai_client") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            with patch("src.pipeline.nodes.skill_questions.increment_llm_calls") as mock_track:
                result = await skill_questions_node({
                    "run_id": run_id,
                    "cv_id": cv_id,
                    "jd_id": jd_id,
                    "triage_result": triage_result,
                    "jd": parsed_jd,
                    "db_path": test_db,
                })

        assert result["status"] == "running"
        assert len(result["skill_questions"]) == 4  # 2 questions per skill (Python, PostgreSQL)
        assert result["skill_questions"][0]["question_text"] == "How have you optimized Python performance in high-throughput systems?"
        assert result["skill_questions"][0]["source_type"] == "skill"
        assert result["skill_questions"][0]["source_id"] == "skill_1"

        # Verify LLM tracking called
        mock_track.assert_called()

        # Verify DB step logged
        steps = get_pipeline_steps(run_id, db_path=test_db)
        assert any(s["step_name"] == "skill_questions" for s in steps)

    @pytest.mark.asyncio
    async def test_empty_skills(self, test_db):
        """Test handling of empty skill list"""
        parsed_cv = ParsedCV(name="Charlie", summary="Dev", projects=[], skills=[])
        parsed_jd = ParsedJD(
            job_title="Backend Dev",
            requirements=[JDRequirement(text="Python", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        triage_result = TriageResult(ranked_projects=[], ranked_skills=[], flagged_for_enrichment=[])

        result = await skill_questions_node({
            "run_id": run_id,
            "cv_id": cv_id,
            "jd_id": jd_id,
            "triage_result": triage_result,
            "jd": parsed_jd,
            "db_path": test_db,
        })

        assert result["status"] == "running"
        assert len(result["skill_questions"]) == 0
        assert result["error_message"] is None

    @pytest.mark.asyncio
    async def test_cost_tracking(self, test_db):
        """Test that LLM costs are tracked for each skill"""
        parsed_cv = ParsedCV(
            name="Diana",
            summary="Dev",
            projects=[],
            skills=[
                CVSkill(skill_name="Kubernetes", proficiency_level="intermediate", years_of_experience=2.0),
                CVSkill(skill_name="Docker", proficiency_level="expert", years_of_experience=4.0),
                CVSkill(skill_name="AWS", proficiency_level="advanced", years_of_experience=3.0),
            ],
        )

        parsed_jd = ParsedJD(
            job_title="DevOps Engineer",
            requirements=[JDRequirement(text="Container orchestration", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        update_cv_status(cv_id=cv_id, status="completed", parsed_data=parsed_cv.model_dump(), db_path=test_db)

        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        update_jd_status(jd_id=jd_id, status="completed", parsed_data=parsed_jd.model_dump(), db_path=test_db)

        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[{"question_text": "K8s experience?", "relevance_note": "test"}]'
        mock_response.usage.prompt_tokens = 150
        mock_response.usage.completion_tokens = 60

        triage_result = TriageResult(
            ranked_projects=[],
            ranked_skills=[
                {
                    "id": "skill_1",
                    "skill_name": "Kubernetes",
                    "score": 0.9,
                    "skill": parsed_cv.skills[0].model_dump(),
                },
                {
                    "id": "skill_2",
                    "skill_name": "Docker",
                    "score": 0.95,
                    "skill": parsed_cv.skills[1].model_dump(),
                },
                {
                    "id": "skill_3",
                    "skill_name": "AWS",
                    "score": 0.87,
                    "skill": parsed_cv.skills[2].model_dump(),
                },
            ],
            flagged_for_enrichment=[],
        )

        with patch("src.pipeline.nodes.skill_questions.get_async_openai_client") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            with patch("src.pipeline.nodes.skill_questions.increment_llm_calls") as mock_track:
                result = await skill_questions_node({
                    "run_id": run_id,
                    "cv_id": cv_id,
                    "jd_id": jd_id,
                    "triage_result": triage_result,
                    "jd": parsed_jd,
                    "db_path": test_db,
                })

        # Should be called once per skill (3 skills)
        assert mock_track.call_count == 3
        for call in mock_track.call_args_list:
            assert call.kwargs["operation"] == "skill_questions_generation"
            assert call.kwargs["model"] == "gpt-4o-mini"
            assert call.kwargs["input_tokens"] == 150
            assert call.kwargs["output_tokens"] == 60

    @pytest.mark.asyncio
    async def test_missing_triage_result(self, test_db):
        """Test error handling when triage_result missing"""
        parsed_cv = ParsedCV(name="Eve", summary="Dev", projects=[], skills=[])
        parsed_jd = ParsedJD(
            job_title="Engineer",
            requirements=[JDRequirement(text="Python", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        result = await skill_questions_node({
            "run_id": run_id,
            "cv_id": cv_id,
            "jd_id": jd_id,
            "jd": parsed_jd,
            "db_path": test_db,
        })

        assert result["status"] == "failed"
        assert "triage_result" in result["error_message"].lower()

    @pytest.mark.asyncio
    async def test_proficiency_aware_questions(self, test_db):
        """Test that questions account for proficiency level"""
        parsed_cv = ParsedCV(
            name="Frank",
            summary="Dev",
            projects=[],
            skills=[
                CVSkill(skill_name="Rust", proficiency_level="beginner", years_of_experience=0.5),
            ],
        )

        parsed_jd = ParsedJD(
            job_title="Rust Developer",
            requirements=[JDRequirement(text="Rust experience", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        update_cv_status(cv_id=cv_id, status="completed", parsed_data=parsed_cv.model_dump(), db_path=test_db)

        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        update_jd_status(jd_id=jd_id, status="completed", parsed_data=parsed_jd.model_dump(), db_path=test_db)

        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '[{"question_text": "Beginner-friendly question", "relevance_note": "test"}]'
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 30

        triage_result = TriageResult(
            ranked_projects=[],
            ranked_skills=[
                {
                    "id": "skill_1",
                    "skill_name": "Rust",
                    "score": 0.5,
                    "skill": parsed_cv.skills[0].model_dump(),
                },
            ],
            flagged_for_enrichment=[],
        )

        with patch("src.pipeline.nodes.skill_questions.get_async_openai_client") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await skill_questions_node({
                "run_id": run_id,
                "cv_id": cv_id,
                "jd_id": jd_id,
                "triage_result": triage_result,
                "jd": parsed_jd,
                "db_path": test_db,
            })

        assert result["status"] == "running"
        assert len(result["skill_questions"]) == 1
        # Evidence should include proficiency level
        assert "beginner" in result["skill_questions"][0]["evidence_snippet"].lower()

    @pytest.mark.asyncio
    async def test_malformed_json_response(self, test_db):
        """Test graceful handling of LLM returning invalid JSON"""
        parsed_cv = ParsedCV(
            name="Grace",
            summary="Dev",
            projects=[],
            skills=[
                CVSkill(skill_name="Go", proficiency_level="intermediate", years_of_experience=2.0),
            ],
        )

        parsed_jd = ParsedJD(
            job_title="Go Developer",
            requirements=[JDRequirement(text="Go", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        update_cv_status(cv_id=cv_id, status="completed", parsed_data=parsed_cv.model_dump(), db_path=test_db)

        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        update_jd_status(jd_id=jd_id, status="completed", parsed_data=parsed_jd.model_dump(), db_path=test_db)

        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Not JSON at all"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 30

        triage_result = TriageResult(
            ranked_projects=[],
            ranked_skills=[
                {
                    "id": "skill_1",
                    "skill_name": "Go",
                    "score": 0.8,
                    "skill": parsed_cv.skills[0].model_dump(),
                },
            ],
            flagged_for_enrichment=[],
        )

        with patch("src.pipeline.nodes.skill_questions.get_async_openai_client") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await skill_questions_node({
                "run_id": run_id,
                "cv_id": cv_id,
                "jd_id": jd_id,
                "triage_result": triage_result,
                "jd": parsed_jd,
                "db_path": test_db,
            })

        # Should complete gracefully with empty questions
        assert result["status"] == "running"
        assert len(result["skill_questions"]) == 0
        assert result["error_message"] is None
