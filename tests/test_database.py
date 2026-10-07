"""Tests for database layer"""

import pytest
import sqlite3
import json
import os
from src.database.db import (
    init_db,
    get_connection,
    create_cv,
    get_cv,
    list_cvs,
    update_cv_status,
    create_jd,
    get_jd,
    list_jds,
    update_jd_status,
    create_pipeline_run,
    get_pipeline_run,
    update_pipeline_run_status,
    list_pipeline_runs,
    create_pipeline_step,
    update_pipeline_step,
    get_pipeline_steps,
    create_question,
    update_question_scores,
    get_questions,
    create_fitness_score,
    get_fitness_scores,
    create_run_summary,
    get_run_summary,
)
from src.database.migrate import migrate_up


@pytest.fixture
def test_db():
    """Create temporary test database"""
    db_path = "test_recroot.db"
    init_db(db_path)
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


class TestCVDocuments:
    def test_create_cv(self, test_db):
        cv_id = create_cv("Senior Engineer CV", "Raw CV text", test_db)
        assert cv_id > 0

        cv = get_cv(cv_id, test_db)
        assert cv["title"] == "Senior Engineer CV"
        assert cv["raw_text"] == "Raw CV text"
        assert cv["status"] == "pending"

    def test_list_cvs(self, test_db):
        create_cv("CV 1", "text 1", test_db)
        create_cv("CV 2", "text 2", test_db)

        cvs = list_cvs(test_db)
        assert len(cvs) == 2

    def test_update_cv_status(self, test_db):
        cv_id = create_cv("CV", "text", test_db)
        parsed_data = {"projects": [], "skills": []}

        update_cv_status(cv_id, "completed", parsed_data, None, db_path=test_db)
        cv = get_cv(cv_id, test_db)

        assert cv["status"] == "completed"
        assert json.loads(cv["parsed_data"]) == parsed_data

    def test_update_cv_error(self, test_db):
        cv_id = create_cv("CV", "text", test_db)
        update_cv_status(cv_id, "failed", None, "Parse error", db_path=test_db)

        cv = get_cv(cv_id, test_db)
        assert cv["status"] == "failed"
        assert cv["error_message"] == "Parse error"


class TestJDDocuments:
    def test_create_jd_pasted(self, test_db):
        jd_id = create_jd("Backend Role", "Looking for...", "pasted", None, test_db)
        assert jd_id > 0

        jd = get_jd(jd_id, test_db)
        assert jd["title"] == "Backend Role"
        assert jd["source_type"] == "pasted"

    def test_create_jd_url(self, test_db):
        jd_id = create_jd(
            "Frontend Role", "content", "url", "https://example.com", test_db
        )
        jd = get_jd(jd_id, test_db)

        assert jd["source_type"] == "url"
        assert jd["source_url"] == "https://example.com"

    def test_list_jds(self, test_db):
        create_jd("JD 1", "text", "pasted", None, test_db)
        create_jd("JD 2", "text", "url", "https://example.com", test_db)

        jds = list_jds(test_db)
        assert len(jds) == 2

    def test_update_jd_status(self, test_db):
        jd_id = create_jd("JD", "text", "pasted", None, test_db)
        parsed_data = {"requirements": []}

        update_jd_status(jd_id, "completed", parsed_data, None, db_path=test_db)
        jd = get_jd(jd_id, test_db)

        assert jd["status"] == "completed"
        assert json.loads(jd["parsed_data"]) == parsed_data


class TestPipelineRuns:
    def test_create_pipeline_run(self, test_db):
        cv_id = create_cv("CV", "text", test_db)
        jd_id = create_jd("JD", "text", "pasted", None, test_db)

        run_id = create_pipeline_run(cv_id, jd_id, "v1.0", test_db)
        assert run_id > 0

        run = get_pipeline_run(run_id, test_db)
        assert run["cv_id"] == cv_id
        assert run["jd_id"] == jd_id
        assert run["prompt_version"] == "v1.0"
        assert run["status"] == "pending"

    def test_update_pipeline_run_status(self, test_db):
        cv_id = create_cv("CV", "text", test_db)
        jd_id = create_jd("JD", "text", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, "v1.0", test_db)

        output = {"questions": []}
        update_pipeline_run_status(run_id, "completed", output, None, test_db)

        run = get_pipeline_run(run_id, test_db)
        assert run["status"] == "completed"
        assert json.loads(run["output_json"]) == output

    def test_list_pipeline_runs(self, test_db):
        cv_id = create_cv("CV", "text", test_db)
        jd_id = create_jd("JD", "text", "pasted", None, test_db)

        create_pipeline_run(cv_id, jd_id, "v1.0", test_db)
        create_pipeline_run(cv_id, jd_id, "v1.0", test_db)

        runs = list_pipeline_runs(limit=10, db_path=test_db)
        assert len(runs) == 2


class TestPipelineSteps:
    def test_create_pipeline_step(self, test_db):
        cv_id = create_cv("CV", "text", test_db)
        jd_id = create_jd("JD", "text", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, "v1.0", test_db)

        step_id = create_pipeline_step(run_id, "security_validation", test_db)
        assert step_id > 0

    def test_update_pipeline_step(self, test_db):
        cv_id = create_cv("CV", "text", test_db)
        jd_id = create_jd("JD", "text", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, "v1.0", test_db)
        step_id = create_pipeline_step(run_id, "security_validation", test_db)

        output = {"validated": True}
        update_pipeline_step(step_id, "completed", output, None, test_db)

        steps = get_pipeline_steps(run_id, test_db)
        assert len(steps) == 1
        assert steps[0]["status"] == "completed"
        assert json.loads(steps[0]["output_json"]) == output

    def test_get_pipeline_steps(self, test_db):
        cv_id = create_cv("CV", "text", test_db)
        jd_id = create_jd("JD", "text", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, "v1.0", test_db)

        create_pipeline_step(run_id, "step_1", test_db)
        create_pipeline_step(run_id, "step_2", test_db)

        steps = get_pipeline_steps(run_id, test_db)
        assert len(steps) == 2


class TestQuestions:
    def test_create_question(self, test_db):
        cv_id = create_cv("CV", "text", test_db)
        jd_id = create_jd("JD", "text", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, "v1.0", test_db)

        q_id = create_question(
            run_id,
            "What is your experience?",
            "project",
            "proj_1",
            "backend experience",
            {"start": 10, "end": 25},
            test_db,
        )
        assert q_id > 0

        questions = get_questions(run_id, test_db)
        assert len(questions) == 1
        assert questions[0]["question_text"] == "What is your experience?"

    def test_update_question_scores(self, test_db):
        cv_id = create_cv("CV", "text", test_db)
        jd_id = create_jd("JD", "text", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, "v1.0", test_db)
        q_id = create_question(
            run_id, "Question?", "skill", "skill_1", None, None, test_db
        )

        update_question_scores(q_id, 8.5, 9.0, 7.0, test_db)

        questions = get_questions(run_id, test_db)
        assert questions[0]["judge_relevance_score"] == 8.5
        assert questions[0]["judge_groundedness_score"] == 9.0
        assert questions[0]["judge_redundancy_score"] == 7.0


class TestFitnessScores:
    def test_create_fitness_score(self, test_db):
        cv_id = create_cv("CV", "text", test_db)
        jd_id = create_jd("JD", "text", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, "v1.0", test_db)

        score_id = create_fitness_score(
            run_id,
            "5+ years Python",
            "must_have",
            8.5,
            "CV mentions 7 years Python",
            test_db,
        )
        assert score_id > 0

        scores = get_fitness_scores(run_id, test_db)
        assert len(scores) == 1
        assert scores[0]["match_score"] == 8.5


class TestRunSummaries:
    def test_create_run_summary(self, test_db):
        cv_id = create_cv("CV", "text", test_db)
        jd_id = create_jd("JD", "text", "pasted", None, test_db)
        run_id = create_pipeline_run(cv_id, jd_id, "v1.0", test_db)

        summary_id = create_run_summary(run_id, 12, 8.3, 0.82, 0.85, test_db)
        assert summary_id > 0

        summary = get_run_summary(run_id, test_db)
        assert summary["total_questions"] == 12
        assert summary["avg_judge_score"] == 8.3
        assert summary["fitness_score"] == 0.82


class TestMigrations:
    def test_migrate_up(self):
        """Test migration system"""
        db_path = "test_migration.db"

        # Initialize fresh DB
        init_db(db_path)

        # Run migrations
        from src.database.migrate import migrate_up

        migrate_up(db_path)

        # Verify migrations table exists
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM migrations")
            applied = [row[0] for row in cursor.fetchall()]
            assert "001_init_schema" in applied

        # Cleanup
        os.remove(db_path)
