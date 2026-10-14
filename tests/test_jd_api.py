"""Tests for JD API endpoints."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app as api_module
from src.api.routes import jd as jd_routes
from src.database.db import get_jd, init_db, update_jd_status


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "jd_api.db")
    upload_dir = tmp_path / "uploads"
    init_db(db_path)
    monkeypatch.setenv("DATABASE_PATH", db_path)
    monkeypatch.setattr(jd_routes, "UPLOAD_DIR", upload_dir)
    with TestClient(api_module.app) as test_client:
        yield test_client, db_path, upload_dir


def test_create_pasted_jd_and_read_endpoints(client, monkeypatch):
    test_client, db_path, _ = client
    parsed = {"job_title": "Engineer", "requirements": []}

    async def fake_parse(source_type, source_data, source_url=None, jd_id=None, db_path=None):
        assert source_type == "pasted"
        assert source_data == "Build reliable systems"
        update_jd_status(jd_id, "completed", parsed_data=parsed, db_path=db_path)

    monkeypatch.setattr(jd_routes, "parse_jd", fake_parse)
    response = test_client.post(
        "/jd",
        data={"source_type": "pasted", "text": "Build reliable systems"},
    )

    assert response.status_code == 202
    body = response.json()
    jd = get_jd(body["jd_id"], db_path)
    assert body["status"] == "pending"
    assert jd["status"] == "completed"
    assert json.loads(jd["parsed_data"]) == parsed

    assert test_client.get("/jd").json()[0]["parsed_data"] == parsed
    assert test_client.get(f"/jd/{body['jd_id']}").json()["raw_text"] == (
        "Build reliable systems"
    )
    status = test_client.get(f"/jd/{body['jd_id']}/status").json()
    assert status["jd_id"] == body["jd_id"]
    assert status["status"] == "completed"
    assert status["error_message"] is None


@pytest.mark.parametrize(
    ("form_data", "expected_detail"),
    [
        ({"source_type": "pasted", "text": "  "}, "Pasted JD text is required"),
        ({"source_type": "url"}, "JD URL is required"),
    ],
)
def test_create_jd_requires_source_data(client, form_data, expected_detail):
    test_client, _, _ = client

    response = test_client.post("/jd", data=form_data)

    assert response.status_code == 400
    assert response.json()["detail"] == expected_detail


def test_create_url_jd_records_parse_failure(client, monkeypatch):
    test_client, db_path, _ = client

    async def fail_parse(*args, **kwargs):
        raise ValueError("URL could not be fetched")

    monkeypatch.setattr(jd_routes, "parse_jd", fail_parse)
    response = test_client.post(
        "/jd",
        data={"source_type": "url", "url": "https://example.com/job"},
    )

    jd = get_jd(response.json()["jd_id"], db_path)
    assert response.status_code == 202
    assert jd["source_url"] == "https://example.com/job"
    assert jd["status"] == "failed"
    assert jd["error_message"] == "URL could not be fetched"


def test_create_url_jd_parses_in_background(client, monkeypatch):
    test_client, db_path, _ = client
    parsed = {"job_title": "Engineer", "requirements": []}

    async def fake_parse(source_type, source_data, source_url=None, jd_id=None, db_path=None):
        assert source_type == "url"
        assert source_data == ""
        assert source_url == "https://example.com/job"
        update_jd_status(jd_id, "completed", parsed_data=parsed, db_path=db_path)

    monkeypatch.setattr(jd_routes, "parse_jd", fake_parse)
    response = test_client.post(
        "/jd",
        data={"source_type": "url", "url": "https://example.com/job"},
    )

    assert response.status_code == 202
    jd = get_jd(response.json()["jd_id"], db_path)
    assert jd["source_url"] == "https://example.com/job"
    assert jd["status"] == "completed"
    assert json.loads(jd["parsed_data"]) == parsed


def test_create_uploaded_jd_file_parses_and_cleans_up(client, monkeypatch):
    test_client, db_path, upload_dir = client

    async def fake_parse(source_type, source_data, source_url=None, jd_id=None, db_path=None):
        assert source_type == "file"
        assert source_data == "Role: Engineer"
        assert source_url == "role.txt"
        update_jd_status(jd_id, "completed", parsed_data={"requirements": []}, db_path=db_path)

    monkeypatch.setattr(jd_routes, "parse_jd", fake_parse)
    response = test_client.post(
        "/jd",
        data={"source_type": "file"},
        files={"file": ("role.txt", b"Role: Engineer", "text/plain")},
    )

    assert response.status_code == 202
    jd = get_jd(response.json()["jd_id"], db_path)
    assert jd["title"] == "role.txt"
    assert jd["status"] == "completed"
    assert list(upload_dir.iterdir()) == []


def test_create_uploaded_pdf_uses_pdf_parser(client, monkeypatch):
    test_client, db_path, upload_dir = client

    async def fake_parse(source_type, source_data, source_url=None, jd_id=None, db_path=None):
        assert source_type == "pdf"
        assert source_url == "role.pdf"
        assert Path(source_data).read_bytes() == b"%PDF-1.7\nmock"
        update_jd_status(jd_id, "completed", parsed_data={"requirements": []}, db_path=db_path)

    monkeypatch.setattr(jd_routes, "parse_jd", fake_parse)
    response = test_client.post(
        "/jd",
        data={"source_type": "file"},
        files={"file": ("role.pdf", b"%PDF-1.7\nmock", "application/pdf")},
    )

    assert response.status_code == 202
    jd = get_jd(response.json()["jd_id"], db_path)
    assert jd["status"] == "completed"
    assert list(upload_dir.iterdir()) == []


@pytest.mark.parametrize("endpoint", ["/jd/999", "/jd/999/status"])
def test_get_jd_returns_404_for_missing_id(client, endpoint):
    test_client, _, _ = client

    response = test_client.get(endpoint)

    assert response.status_code == 404
    assert response.json()["detail"] == "JD not found"
