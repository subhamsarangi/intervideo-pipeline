"""Tests for relevance triage node and embedding ranking"""

import pytest
import os
from src.database.db import (
    init_db,
    create_cv,
    create_jd,
    create_pipeline_run,
    get_pipeline_run,
    get_pipeline_steps,
    update_cv_status,
    update_jd_status,
)
from src.models.cv import ParsedCV, CVProject, CVSkill
from src.models.jd import ParsedJD, JDRequirement
from src.models.pipeline import PipelineState
from src.pipeline.nodes.triage import (
    triage_node,
    relevance_triage_node,
    detect_fast_moving_tech,
)


@pytest.fixture
def test_db():
    """Create a temporary test database"""
    db_path = "test_triage.db"
    init_db(db_path)
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


class TestTechDetection:
    def test_detect_fast_moving_tech(self):
        """Test detection of modern fast-moving frameworks and keywords"""
        text = "Built an agentic RAG workflow with LangGraph, vLLM, and Qdrant database."
        flagged = detect_fast_moving_tech(text)
        assert any("langgraph" in f.lower() for f in flagged)
        assert any("vllm" in f.lower() for f in flagged)
        assert any("qdrant" in f.lower() for f in flagged)
        assert any("rag" in f.lower() for f in flagged)

    def test_detect_versioned_tech(self):
        """Test detection of versioned libraries"""
        text = "Upgraded to Next.js 15 and Svelte 5 with v2.1.0 release."
        flagged = detect_fast_moving_tech(text)
        assert any("next" in f.lower() for f in flagged)
        assert any("svelte" in f.lower() for f in flagged)
        assert any("v2.1.0" in f.lower() for f in flagged)

    def test_clean_text_no_flags(self):
        """Standard mature tech should not be flagged"""
        text = "Developed SQL queries in PostgreSQL and wrote Python unit tests."
        flagged = detect_fast_moving_tech(text)
        assert len(flagged) == 0


class TestTriageNode:
    @pytest.mark.asyncio
    async def test_triage_node_ranking_and_enrichment(self, test_db, tracker):
        """Test that triage ranks matching projects higher and flags modern tech"""
        parsed_cv = ParsedCV(
            name="Bob Smith",
            summary="Full stack engineer",
            projects=[
                CVProject(
                    title="Python Backend API",
                    description="Built high-performance microservices with FastAPI and PostgreSQL.",
                    technologies=["Python", "FastAPI", "PostgreSQL"],
                ),
                CVProject(
                    title="Legacy WordPress Theme",
                    description="Designed custom PHP themes for marketing blogs.",
                    technologies=["PHP", "WordPress", "CSS"],
                ),
                CVProject(
                    title="Autonomous Agent Pipeline",
                    description="Built AI agents using LangGraph and DeepSeek models.",
                    technologies=["LangGraph", "DeepSeek", "Python"],
                ),
            ],
            skills=[
                CVSkill(skill_name="FastAPI", proficiency_level="expert", years_of_experience=4.0),
                CVSkill(skill_name="Photoshop", proficiency_level="beginner", years_of_experience=1.0),
                CVSkill(skill_name="Python", proficiency_level="expert", years_of_experience=6.0),
            ],
        )

        parsed_jd = ParsedJD(
            job_title="Senior Python Backend Engineer",
            company="Tech Corp",
            summary="Looking for a Python specialist to build scalable FastAPI services.",
            requirements=[
                JDRequirement(text="5+ years of Python experience", requirement_type="must_have"),
                JDRequirement(text="Strong knowledge of FastAPI and REST APIs", requirement_type="must_have"),
                JDRequirement(text="Experience with PostgreSQL databases", requirement_type="must_have"),
            ],
        )

        cv_id = create_cv("Bob CV", "raw text", test_db)
        update_cv_status(cv_id=cv_id, status="completed", parsed_data=parsed_cv.model_dump(), db_path=test_db)

        jd_id = create_jd("Python JD", "raw text", "pasted", None, test_db)
        update_jd_status(jd_id=jd_id, status="completed", parsed_data=parsed_jd.model_dump(), db_path=test_db)

        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        state = PipelineState(
            run_id=run_id,
            cv_id=cv_id,
            jd_id=jd_id,
            cv=parsed_cv,
            jd=parsed_jd,
            db_path=test_db,
        )

        with tracker("Triage Execution") as bar:
            bar.update(30, "embedding & ranking")
            result = await triage_node(state)
            bar.update(70, "done")

        assert result["status"] == "running"
        triage_res = result["triage_result"]
        assert triage_res is not None

        # Verify projects ranking
        ranked_projs = triage_res.ranked_projects
        assert len(ranked_projs) == 3
        # Python Backend API should be ranked higher than WordPress theme
        titles = [p["title"] for p in ranked_projs]
        assert titles.index("Python Backend API") < titles.index("Legacy WordPress Theme")

        # Verify skills ranking
        ranked_skills = triage_res.ranked_skills
        assert len(ranked_skills) == 3
        skill_names = [s["skill_name"] for s in ranked_skills]
        assert skill_names.index("FastAPI") < skill_names.index("Photoshop")
        assert skill_names.index("Python") < skill_names.index("Photoshop")

        # Verify flagged for enrichment (LangGraph, DeepSeek)
        flagged = triage_res.flagged_for_enrichment
        assert any("langgraph" in f.lower() for f in flagged)
        assert any("deepseek" in f.lower() for f in flagged)

        # Verify DB step recorded
        steps = get_pipeline_steps(run_id, db_path=test_db)
        assert len(steps) == 1
        assert steps[0]["step_name"] == "relevance_triage"
        assert steps[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_triage_node_dict_input(self, test_db):
        """Test triage node accepts raw dictionary state and loads from DB"""
        parsed_cv = ParsedCV(
            name="Jane",
            projects=[CVProject(title="Project A", description="Python dev")],
            skills=[CVSkill(skill_name="Python")],
        )
        parsed_jd = ParsedJD(
            job_title="Developer",
            requirements=[JDRequirement(text="Python required", requirement_type="must_have")],
        )

        cv_id = create_cv("Jane CV", "raw text", test_db)
        update_cv_status(cv_id=cv_id, status="completed", parsed_data=parsed_cv.model_dump(), db_path=test_db)

        jd_id = create_jd("Job JD", "raw text", "pasted", None, test_db)
        update_jd_status(jd_id=jd_id, status="completed", parsed_data=parsed_jd.model_dump(), db_path=test_db)

        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        result = await relevance_triage_node({
            "run_id": run_id,
            "cv_id": cv_id,
            "jd_id": jd_id,
            "db_path": test_db,
        })

        assert result["status"] == "running"
        assert len(result["triage_result"].ranked_projects) == 1

    @pytest.mark.asyncio
    async def test_triage_node_missing_cv_fails(self, test_db):
        """Test triage node handles missing documents gracefully"""
        jd_id = create_jd("Job JD", "raw text", "pasted", None, test_db)
        run_id = create_pipeline_run(99999, jd_id, db_path=test_db)

        result = await triage_node({
            "run_id": run_id,
            "cv_id": 99999,
            "jd_id": jd_id,
            "db_path": test_db,
        })

        assert result["status"] == "failed"
        assert "not found" in result["error_message"].lower()

        run = get_pipeline_run(run_id, db_path=test_db)
        assert run["status"] == "failed"
