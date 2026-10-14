"""Tests for pipeline API routes."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api import app as api_module
from src.api.routes import pipeline as pipeline_routes
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
    assert run["error_message"] == "graph unavailable"


def test_get_pipeline_operations_returns_usage_and_cost(client):
    test_client, db_path = client
    cv_id, jd_id = _create_ready_inputs(db_path)
    run_id = create_pipeline_run(cv_id, jd_id, db_path=db_path)
    step_id = create_pipeline_step(run_id, "question_generation", db_path=db_path)
    increment_llm_calls(
        operation="generate_questions",
        model="gpt-5.4-nano",
        input_tokens=1000,
        output_tokens=500,
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
    assert operation["input_tokens"] == 1000
    assert operation["output_tokens"] == 500
    expected_cost = (
        1000 * MODEL_COSTS["gpt-5.4-nano"]["input"]
        + 500 * MODEL_COSTS["gpt-5.4-nano"]["output"]
    ) / 1_000_000
    assert operation["cost_usd"] == pytest.approx(expected_cost)
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
