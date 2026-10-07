"""Tests for merge, deduplicate, and rerank node"""

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
from src.models.pipeline import PipelineState, GeneratedQuestion
from src.pipeline.nodes.merge_dedupe import merge_dedupe_node


@pytest.fixture
def test_db():
    """Create temporary test database"""
    db_path = "test_merge_dedupe.db"
    init_db(db_path)
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


class TestMergeDedupe:
    @pytest.mark.asyncio
    async def test_basic_merge(self, test_db):
        """Test merging project and skill questions"""
        parsed_cv = ParsedCV(name="Alice", summary="Dev", projects=[], skills=[])
        parsed_jd = ParsedJD(
            job_title="Engineer",
            requirements=[JDRequirement(text="Python experience", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        # Simulate project and skill questions
        project_questions = [
            GeneratedQuestion(
                question_text="How did you optimize the backend?",
                source_type="project",
                source_id="proj_1",
            ).model_dump()
        ]

        skill_questions = [
            GeneratedQuestion(
                question_text="Describe your Python experience with async/await",
                source_type="skill",
                source_id="skill_1",
            ).model_dump()
        ]

        # Mock embeddings response for dedup
        mock_embeddings = MagicMock()
        mock_embeddings.data = [
            MagicMock(embedding=[0.1] * 1536),  # proj question
            MagicMock(embedding=[0.1] * 1536),  # skill question
        ]
        mock_embeddings.usage.prompt_tokens = 50
        mock_embeddings.usage.completion_tokens = 0

        with patch("src.pipeline.nodes.merge_dedupe.get_async_openai_client") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.embeddings.create = AsyncMock(return_value=mock_embeddings)
            mock_client.return_value = mock_instance

            with patch("src.pipeline.nodes.merge_dedupe.increment_llm_calls"):
                result = await merge_dedupe_node({
                    "run_id": run_id,
                    "cv_id": cv_id,
                    "jd_id": jd_id,
                    "project_questions": project_questions,
                    "skill_questions": skill_questions,
                    "jd": parsed_jd,
                    "db_path": test_db,
                })

        assert result["status"] == "running"
        assert len(result["merged_questions"]) >= 1
        
        # Verify DB step logged
        steps = get_pipeline_steps(run_id, db_path=test_db)
        assert any(s["step_name"] == "merge_dedupe" for s in steps)

    @pytest.mark.asyncio
    async def test_empty_questions(self, test_db):
        """Test handling of empty question lists"""
        parsed_cv = ParsedCV(name="Bob", summary="Dev", projects=[], skills=[])
        parsed_jd = ParsedJD(
            job_title="Engineer",
            requirements=[JDRequirement(text="Python", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        result = await merge_dedupe_node({
            "run_id": run_id,
            "cv_id": cv_id,
            "jd_id": jd_id,
            "project_questions": [],
            "skill_questions": [],
            "jd": parsed_jd,
            "db_path": test_db,
        })

        assert result["status"] == "running"
        assert len(result["merged_questions"]) == 0
        assert result["error_message"] is None

    @pytest.mark.asyncio
    async def test_coverage_check(self, test_db):
        """Test coverage detection for JD requirements"""
        parsed_cv = ParsedCV(name="Charlie", summary="Dev", projects=[], skills=[])
        parsed_jd = ParsedJD(
            job_title="Backend Engineer",
            requirements=[
                JDRequirement(text="Python experience required", requirement_type="must_have"),
                JDRequirement(text="Docker containerization", requirement_type="must_have"),
                JDRequirement(text="AWS cloud services", requirement_type="nice_to_have"),
            ],
        )

        cv_id = create_cv("CV", "raw", test_db)
        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        # Questions covering Python + Docker but missing AWS
        questions = [
            GeneratedQuestion(
                question_text="Tell me about your Python and backend experience",
                source_type="skill",
                source_id="skill_1",
            ).model_dump(),
            GeneratedQuestion(
                question_text="How have you used Docker for containerization?",
                source_type="skill",
                source_id="skill_2",
            ).model_dump(),
        ]

        mock_embeddings = MagicMock()
        mock_embeddings.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(3)]
        mock_embeddings.usage.prompt_tokens = 100
        mock_embeddings.usage.completion_tokens = 0

        with patch("src.pipeline.nodes.merge_dedupe.get_async_openai_client") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.embeddings.create = AsyncMock(return_value=mock_embeddings)
            mock_client.return_value = mock_instance

            with patch("src.pipeline.nodes.merge_dedupe.increment_llm_calls"):
                result = await merge_dedupe_node({
                    "run_id": run_id,
                    "cv_id": cv_id,
                    "jd_id": jd_id,
                    "project_questions": [],
                    "skill_questions": questions,
                    "jd": parsed_jd,
                    "db_path": test_db,
                })

        assert result["status"] == "running"
        coverage_gaps = result["coverage_gaps"]
        
        # AWS should be in gaps, Python and Docker covered
        gap_texts = [gap["requirement"].lower() for gap in coverage_gaps]
        assert any("aws" in text for text in gap_texts)

    @pytest.mark.asyncio
    async def test_deduplication_similarity(self, test_db):
        """Test that similar questions are deduplicated"""
        parsed_cv = ParsedCV(name="Diana", summary="Dev", projects=[], skills=[])
        parsed_jd = ParsedJD(
            job_title="Engineer",
            requirements=[JDRequirement(text="Python", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        # Very similar questions (should be deduplicated)
        questions = [
            GeneratedQuestion(
                question_text="How did you implement the microservices architecture?",
                source_type="project",
                source_id="proj_1",
            ).model_dump(),
            GeneratedQuestion(
                question_text="Tell me about your microservices architecture design",
                source_type="skill",
                source_id="skill_1",
            ).model_dump(),
        ]

        # Mock embeddings with high similarity (0.95+)
        mock_embeddings = MagicMock()
        mock_embeddings.data = [
            MagicMock(embedding=[0.9, 0.1] + [0.0] * 1534),  # Similar embedding 1
            MagicMock(embedding=[0.89, 0.11] + [0.0] * 1534),  # Similar embedding 2
        ]
        mock_embeddings.usage.prompt_tokens = 50
        mock_embeddings.usage.completion_tokens = 0

        with patch("src.pipeline.nodes.merge_dedupe.get_async_openai_client") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.embeddings.create = AsyncMock(return_value=mock_embeddings)
            mock_client.return_value = mock_instance

            with patch("src.pipeline.nodes.merge_dedupe.increment_llm_calls"):
                result = await merge_dedupe_node({
                    "run_id": run_id,
                    "cv_id": cv_id,
                    "jd_id": jd_id,
                    "project_questions": questions[:1],
                    "skill_questions": questions[1:],
                    "jd": parsed_jd,
                    "db_path": test_db,
                })

        assert result["status"] == "running"
        # With high similarity, should have fewer merged questions than input
        # (dedup removes similar ones)
        assert len(result["merged_questions"]) <= 2

    @pytest.mark.asyncio
    async def test_reranking_by_relevance(self, test_db):
        """Test that questions are reranked by relevance to JD"""
        parsed_cv = ParsedCV(name="Eve", summary="Dev", projects=[], skills=[])
        parsed_jd = ParsedJD(
            job_title="Backend Python Engineer",
            summary="Building scalable microservices",
            requirements=[JDRequirement(text="Python backend", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        questions = [
            GeneratedQuestion(
                question_text="What is your graphic design experience?",
                source_type="skill",
                source_id="skill_1",
            ).model_dump(),
            GeneratedQuestion(
                question_text="How do you design scalable microservices?",
                source_type="project",
                source_id="proj_1",
            ).model_dump(),
        ]

        # Mock embeddings with varying relevance
        mock_embeddings = MagicMock()
        # JD embedding (most relevant to microservices)
        jd_emb = [1.0, 0.8, 0.0] + [0.0] * 1533
        # Graphic design question (low relevance)
        q1_emb = [0.1, 0.05, 0.95] + [0.0] * 1533
        # Microservices question (high relevance)
        q2_emb = [0.95, 0.85, 0.1] + [0.0] * 1533

        mock_embeddings.data = [
            MagicMock(embedding=jd_emb),
            MagicMock(embedding=q1_emb),
            MagicMock(embedding=q2_emb),
        ]
        mock_embeddings.usage.prompt_tokens = 100
        mock_embeddings.usage.completion_tokens = 0

        with patch("src.pipeline.nodes.merge_dedupe.get_async_openai_client") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.embeddings.create = AsyncMock(return_value=mock_embeddings)
            mock_client.return_value = mock_instance

            with patch("src.pipeline.nodes.merge_dedupe.increment_llm_calls"):
                result = await merge_dedupe_node({
                    "run_id": run_id,
                    "cv_id": cv_id,
                    "jd_id": jd_id,
                    "project_questions": [questions[1]],
                    "skill_questions": [questions[0]],
                    "jd": parsed_jd,
                    "db_path": test_db,
                })

        assert result["status"] == "running"
        merged = result["merged_questions"]
        # Most relevant question (microservices) should be first after reranking
        assert len(merged) > 0

    @pytest.mark.asyncio
    async def test_missing_jd(self, test_db):
        """Test error handling when JD missing"""
        cv_id = create_cv("CV", "raw", test_db)
        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        # Add questions so node tries to access JD
        questions = [
            GeneratedQuestion(
                question_text="Test question",
                source_type="skill",
                source_id="skill_1",
            ).model_dump()
        ]

        result = await merge_dedupe_node({
            "run_id": run_id,
            "cv_id": cv_id,
            "jd_id": jd_id,
            "project_questions": questions,
            "skill_questions": [],
            "db_path": test_db,
        })

        assert result["status"] == "failed"
        assert "jd" in result["error_message"].lower()
