"""Tests for CV upload API."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api import app as api_module
from src.api.routes import cv as cv_routes
from src.database.db import create_cv, get_cv, init_db, update_cv_status


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "cv_api.db")
    upload_dir = tmp_path / "uploads"
    init_db(db_path)
    monkeypatch.setenv("DATABASE_PATH", db_path)
    monkeypatch.setattr(cv_routes, "UPLOAD_DIR", upload_dir)
    with TestClient(api_module.app) as test_client:
        yield test_client, db_path, upload_dir


def test_upload_cv_parses_pdf_in_background(client, monkeypatch):
    test_client, db_path, upload_dir = client
    parsed = {"name": "Test Candidate", "projects": [], "skills": []}

    async def fake_parse(file_path, cv_id=None, db_path=None):
        assert file_path.endswith(".pdf")
        from src.database.db import update_cv_status

        update_cv_status(
            cv_id,
            "completed",
            parsed_data=parsed,
            db_path=db_path,
        )

    monkeypatch.setattr(cv_routes, "parse_cv", fake_parse)
    response = test_client.post(
        "/cv/upload",
        files={"file": ("candidate.pdf", b"%PDF-1.7\nmock", "application/pdf")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    cv = get_cv(body["cv_id"], db_path)
    assert cv["title"] == "candidate.pdf"
    assert cv["status"] == "completed"
    assert json.loads(cv["parsed_data"]) == parsed
    assert list(upload_dir.iterdir()) == []


def test_upload_cv_records_background_parse_failure(client, monkeypatch):
    test_client, db_path, upload_dir = client

    async def fail_parse(*args, **kwargs):
        raise RuntimeError("parser unavailable")

    monkeypatch.setattr(cv_routes, "parse_cv", fail_parse)
    response = test_client.post(
        "/cv/upload",
        files={"file": ("candidate.pdf", b"%PDF-1.7\nmock", "application/pdf")},
    )

    cv = get_cv(response.json()["cv_id"], db_path)
    assert response.status_code == 202
    assert cv["status"] == "failed"
    assert cv["error_message"] == "RuntimeError: parser unavailable"
    assert list(upload_dir.iterdir()) == []


def test_get_cv_list_detail_and_status(client):
    test_client, db_path, _ = client
    cv_id = create_cv("candidate.pdf", "Raw CV", db_path=db_path)
    update_cv_status(
        cv_id,
        "completed",
        parsed_data={"name": "Test Candidate", "skills": []},
        db_path=db_path,
    )

    list_response = test_client.get("/cv")
    detail_response = test_client.get(f"/cv/{cv_id}")
    status_response = test_client.get(f"/cv/{cv_id}/status")

    assert list_response.status_code == 200
    assert list_response.json()[0]["id"] == cv_id
    assert list_response.json()[0]["parsed_data"] == {
        "name": "Test Candidate",
        "skills": [],
    }
    assert detail_response.status_code == 200
    assert detail_response.json()["raw_text"] == "Raw CV"
    assert detail_response.json()["parsed_data"]["name"] == "Test Candidate"
    assert status_response.status_code == 200
    assert status_response.json()["cv_id"] == cv_id
    assert status_response.json()["status"] == "completed"
    assert status_response.json()["error_message"] is None


@pytest.mark.parametrize("endpoint", ["/cv/999", "/cv/999/status"])
def test_get_cv_returns_404_for_missing_id(client, endpoint):
    test_client, _, _ = client

    response = test_client.get(endpoint)

    assert response.status_code == 404
    assert response.json()["detail"] == "CV not found"


@pytest.mark.parametrize(
    ("filename", "content_kind", "expected_status"),
    [
        ("candidate.txt", "pdf", 400),
        ("candidate.pdf", "not_pdf", 400),
        ("candidate.pdf", "oversized", 413),
    ],
    ids=["wrong-extension", "invalid-pdf-signature", "oversized-pdf"],
)
def test_upload_cv_rejects_invalid_files(
    client, filename, content_kind, expected_status
):
    test_client, db_path, upload_dir = client
    content = {
        "pdf": b"%PDF-1.7\nmock",
        "not_pdf": b"not a PDF",
        "oversized": b"%PDF-" + b"x" * (10 * 1024 * 1024),
    }[content_kind]
    response = test_client.post(
        "/cv/upload",
        files={"file": (filename, content, "application/pdf")},
    )

    assert response.status_code == expected_status
    assert response.json()["detail"]
    assert not upload_dir.exists() or list(upload_dir.iterdir()) == []
