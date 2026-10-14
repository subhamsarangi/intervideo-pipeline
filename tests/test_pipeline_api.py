"""Tests for pipeline API routes."""

import asyncio
import sqlite3
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api import app as api_module
from src.api.routes import pipeline as pipeline_routes
from src.api.routes.pipeline import _run_pipeline_in_background
from src.database.db import (
    create_cv,
    create_jd,
    create_pipeline_run,
    create_pipeline_step,
    get_pipeline_run,
    get_connection,
    init_db,
    update_cv_status,
    update_jd_status,
)
from src.utils.llm_tracking import MODEL_COSTS, increment_llm_calls


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "pipeline_api.db")
    init_db(db_path)
    monkeypatch.setenv("DATABASE_PATH", db_path)
    with TestClient(api_module.app) as test_client:
        yield test_client, db_path


def _create_ready_inputs(db_path):
    cv_id = create_cv("candidate.pdf", "Candidate CV", db_path=db_path)
    update_cv_status(
        cv_id, "completed", parsed_data={"name": "Candidate"}, db_path=db_path
    )
    jd_id = create_jd("Engineer role", "Build services", db_path=db_path)
    update_jd_status(
        jd_id, "completed", parsed_data={"job_title": "Engineer"}, db_path=db_path
    )
    return cv_id, jd_id


def test_start_pipeline_creates_run_and_schedules_execution(client, monkeypatch):
    test_client, db_path = client
    cv_id, jd_id = _create_ready_inputs(db_path)
    execute = AsyncMock(return_value={"status": "completed"})
    monkeypatch.setattr(pipeline_routes, "run_pipeline", execute)

    response = test_client.post(
        "/pipeline/start",
        json={"cv_id": cv_id, "jd_id": jd_id, "prompt_version": "v2"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    run = get_pipeline_run(body["run_id"], db_path)
    assert run["cv_id"] == cv_id
    assert run["jd_id"] == jd_id
    assert run["prompt_version"] == "v2"
    assert run["status"] == "running"
    state, actual_db_path = execute.await_args.args
    assert state.run_id == body["run_id"]
    assert state.cv_id == cv_id
    assert state.jd_id == jd_id
    assert actual_db_path == db_path


def test_start_pipeline_records_execution_failure(client, monkeypatch):
    test_client, db_path = client
    cv_id, jd_id = _create_ready_inputs(db_path)
    execute = AsyncMock(side_effect=RuntimeError("graph unavailable"))
    monkeypatch.setattr(pipeline_routes, "run_pipeline", execute)

    response = test_client.post(
        "/pipeline/start", json={"cv_id": cv_id, "jd_id": jd_id}
    )

    run = get_pipeline_run(response.json()["run_id"], db_path)
    assert response.status_code == 202
    assert run["status"] == "failed"
    assert run["error_message"] == "RuntimeError: graph unavailable"


def test_background_startup_failure_is_recorded_not_left_pending(monkeypatch):
    statuses = []

    def update_status(run_id, status, error_message=None, db_path=None):
        statuses.append((run_id, status, error_message, db_path))
        if status == "running":
            raise RuntimeError("could not mark run running")

    execute = AsyncMock()
    monkeypatch.setattr(pipeline_routes, "update_pipeline_run_status", update_status)
    monkeypatch.setattr(pipeline_routes, "run_pipeline", execute)

    asyncio.run(_run_pipeline_in_background(12, 3, 3, None, "db.sqlite"))

    assert statuses == [
        (12, "running", None, "db.sqlite"),
        (
            12,
            "failed",
            "RuntimeError: could not mark run running",
            "db.sqlite",
        ),
    ]
    execute.assert_not_awaited()


def test_api_startup_migrates_existing_database_missing_pipeline_cost(
    tmp_path, monkeypatch
):
    db_path = str(tmp_path / "legacy_api.db")
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("ALTER TABLE pipeline_runs DROP COLUMN total_cost_usd")
    monkeypatch.setenv("DATABASE_PATH", db_path)

    with TestClient(api_module.app) as test_client:
        assert test_client.get("/health").json() == {"status": "ok"}

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(pipeline_runs)")
        }
    assert "total_cost_usd" in columns


def test_get_pipeline_operations_returns_usage_and_cost(client):
    test_client, db_path = client
    cv_id, jd_id = _create_ready_inputs(db_path)
    run_id = create_pipeline_run(cv_id, jd_id, db_path=db_path)
    step_id = create_pipeline_step(run_id, "question_generation", db_path=db_path)
    increment_llm_calls(
        operation="generate_questions",
        model="gpt-5.4-nano",
        input_tokens=2344,
        output_tokens=1895,
        run_id=run_id,
        step_id=step_id,
        db_path=db_path,
    )

    response = test_client.get(f"/pipeline/{run_id}/operations")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == run_id
    assert len(body["operations"]) == 1
    operation = body["operations"][0]
    assert operation["run_id"] == run_id
    assert operation["step_id"] == step_id
    assert operation["operation"] == "generate_questions"
    assert operation["model"] == "gpt-5.4-nano"
    assert operation["input_tokens"] == 2344
    assert operation["output_tokens"] == 1895
    expected_cost = (
        2344 * MODEL_COSTS["gpt-5.4-nano"]["input"]
        + 1895 * MODEL_COSTS["gpt-5.4-nano"]["output"]
    ) / 1_000_000
    assert expected_cost == pytest.approx(0.0014886)
    assert operation["cost_usd"] == 0.0014886
    with get_connection(db_path) as conn:
        total_cost = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM llm_operations WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
    assert body["total_cost_usd"] == pytest.approx(total_cost)


def test_get_pipeline_operations_returns_empty_list_for_run_without_calls(client):
    test_client, db_path = client
    cv_id, jd_id = _create_ready_inputs(db_path)
    run_id = create_pipeline_run(cv_id, jd_id, db_path=db_path)

    response = test_client.get(f"/pipeline/{run_id}/operations")

    assert response.status_code == 200
    assert response.json()["operations"] == []
    assert response.json()["total_cost_usd"] == 0


def test_get_pipeline_operations_returns_404_for_missing_run(client):
    test_client, _ = client

    response = test_client.get("/pipeline/999/operations")

    assert response.status_code == 404
    assert response.json()["detail"] == "Pipeline run not found"


def test_get_pipeline_history_returns_recent_runs_newest_first(client):
    test_client, db_path = client
    cv_id, jd_id = _create_ready_inputs(db_path)
    earlier_run = create_pipeline_run(cv_id, jd_id, "v1", db_path=db_path)
    later_run = create_pipeline_run(cv_id, jd_id, "v2", db_path=db_path)

    response = test_client.get("/pipeline/history")

    assert response.status_code == 200
    runs = response.json()
    assert [run["id"] for run in runs[:2]] == [later_run, earlier_run]
    assert runs[0]["prompt_version"] == "v2"


def test_get_pipeline_history_applies_limit(client):
    test_client, db_path = client
    cv_id, jd_id = _create_ready_inputs(db_path)
    run_ids = [
        create_pipeline_run(cv_id, jd_id, db_path=db_path) for _ in range(3)
    ]

    response = test_client.get("/pipeline/history?limit=2")

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert [run["id"] for run in response.json()] == list(reversed(run_ids[-2:]))


@pytest.mark.parametrize("limit", ["0", "101", "not-a-number"])
def test_get_pipeline_history_rejects_invalid_limit(client, limit):
    test_client, _ = client

    response = test_client.get(f"/pipeline/history?limit={limit}")

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("cv_status", "jd_status", "expected_detail"),
    [
        ("pending", "completed", "CV is not ready; current status: pending"),
        ("completed", "processing", "JD is not ready; current status: processing"),
    ],
)
def test_start_pipeline_rejects_unfinished_inputs(
    client, cv_status, jd_status, expected_detail
):
    test_client, db_path = client
    cv_id = create_cv("candidate.pdf", "Candidate CV", db_path=db_path)
    jd_id = create_jd("Engineer role", "Build services", db_path=db_path)
    if cv_status == "completed":
        update_cv_status(cv_id, "completed", parsed_data={}, db_path=db_path)
    if jd_status == "completed":
        update_jd_status(jd_id, "completed", parsed_data={}, db_path=db_path)
    else:
        update_jd_status(jd_id, "processing", db_path=db_path)

    response = test_client.post(
        "/pipeline/start", json={"cv_id": cv_id, "jd_id": jd_id}
    )

    assert response.status_code == 409
    assert response.json()["detail"] == expected_detail


@pytest.mark.parametrize(
    ("missing", "expected_detail"),
    [
        ("cv", "CV not found"),
        ("jd", "JD not found"),
    ],
)
def test_start_pipeline_returns_404_for_missing_input(
    client, missing, expected_detail
):
    test_client, db_path = client
    cv_id, jd_id = _create_ready_inputs(db_path)
    if missing == "cv":
        cv_id = 999
    else:
        jd_id = 999

    response = test_client.post(
        "/pipeline/start", json={"cv_id": cv_id, "jd_id": jd_id}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == expected_detail
