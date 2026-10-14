"""Tests for API-only end-to-end pipeline runner."""

import asyncio
import io
import json
from pathlib import Path
import sys

import httpx
import pytest

from run_full_pipeline_with_api import (
    _request,
    _wait_for_status,
    run_full_pipeline,
)


def test_runner_uses_api_for_upload_parse_run_and_results(tmp_path, capsys):
    cv_path = tmp_path / "candidate.pdf"
    jd_path = tmp_path / "role.pdf"
    cv_path.write_bytes(b"%PDF-1.7 cv")
    jd_path.write_bytes(b"%PDF-1.7 jd")
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/cv/upload":
            return httpx.Response(202, json={"cv_id": 10, "status": "pending"})
        if request.method == "POST" and request.url.path == "/jd":
            assert b'name="source_type"\r\n\r\nfile' in request.content
            return httpx.Response(202, json={"jd_id": 20, "status": "pending"})
        if request.method == "POST" and request.url.path == "/pipeline/start":
            assert json.loads(request.content) == {"cv_id": 10, "jd_id": 20}
            return httpx.Response(202, json={"run_id": 30, "status": "pending"})
        if request.method == "GET" and request.url.path in (
            "/cv/10/status",
            "/jd/20/status",
        ):
            return httpx.Response(200, json={"status": "completed"})
        if request.method == "GET" and request.url.path == "/pipeline/30":
            return httpx.Response(200, json={"status": "completed"})
        if request.method == "GET" and request.url.path == "/pipeline/30/questions":
            return httpx.Response(
                200,
                json=[{"question_text": "Describe your experience."}],
            )
        if request.method == "GET" and request.url.path == "/pipeline/30/operations":
            return httpx.Response(200, json={"total_cost_usd": 0.012})
        return httpx.Response(404)

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            return await run_full_pipeline(
                client, cv_path, jd_path, poll_interval=0.001, timeout=1
            )

    assert asyncio.run(run()) == 0
    output = capsys.readouterr().out
    assert "Questions: 1" in output
    assert "Pipeline operation cost: $0.012000" in output
    assert [path for method, path in requests if method == "POST"] == [
        "/cv/upload",
        "/jd",
        "/pipeline/start",
    ]


def test_runner_reports_failed_background_parse(tmp_path):
    cv_path = tmp_path / "candidate.pdf"
    jd_path = tmp_path / "role.pdf"
    cv_path.write_bytes(b"%PDF-1.7 cv")
    jd_path.write_bytes(b"%PDF-1.7 jd")

    def handler(request):
        if request.method == "POST" and request.url.path == "/cv/upload":
            return httpx.Response(202, json={"cv_id": 10, "status": "pending"})
        if request.method == "GET" and request.url.path == "/cv/10/status":
            return httpx.Response(
                200,
                json={"status": "failed", "error_message": "parser failed"},
            )
        return httpx.Response(404)

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            return await run_full_pipeline(client, cv_path, jd_path, timeout=1)

    with pytest.raises(RuntimeError, match="CV parsing failed: parser failed"):
        asyncio.run(run())


def test_runner_explains_failure_when_api_has_no_error_message(tmp_path):
    cv_path = tmp_path / "candidate.pdf"
    jd_path = tmp_path / "role.pdf"
    cv_path.write_bytes(b"%PDF-1.7 cv")
    jd_path.write_bytes(b"%PDF-1.7 jd")

    def handler(request):
        if request.method == "POST" and request.url.path == "/cv/upload":
            return httpx.Response(202, json={"cv_id": 10, "status": "pending"})
        if request.method == "GET" and request.url.path == "/cv/10/status":
            return httpx.Response(200, json={"status": "failed", "error_message": ""})
        return httpx.Response(404)

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            return await run_full_pipeline(client, cv_path, jd_path, timeout=1)

    with pytest.raises(
        RuntimeError,
        match="API returned no error details; inspect server logs",
    ):
        asyncio.run(run())


def test_runner_identifies_blank_http_timeout_errors():
    def handler(request):
        raise httpx.ReadTimeout("")

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            return await _request(client, "GET", "/pipeline/30")

    with pytest.raises(
        RuntimeError,
        match=r"GET /pipeline/30 timed out \(ReadTimeout\)",
    ):
        asyncio.run(run())


def test_runner_retries_transient_read_error_on_get():
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadError("")
        return httpx.Response(200, json={"status": "completed"})

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            return await _request(client, "GET", "/pipeline/30")

    assert asyncio.run(run()) == {"status": "completed"}
    assert attempts == 2


def test_runner_explains_exhausted_read_error():
    def handler(request):
        raise httpx.ReadError("")

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            return await _request(client, "GET", "/pipeline/30")

    with pytest.raises(
        RuntimeError,
        match=r"GET /pipeline/30: API closed the connection.*Check API server logs",
    ):
        asyncio.run(run())


def test_runner_does_not_repeat_progress_logs_when_not_interactive(
    tmp_path, capsys
):
    statuses = iter(
        [
            {"status": "processing"},
            {"status": "processing"},
            {"status": "completed"},
        ]
    )

    def handler(request):
        if request.method == "GET" and request.url.path == "/cv/10/status":
            return httpx.Response(200, json=next(statuses))
        return httpx.Response(404)

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            return await _wait_for_status(
                client,
                "/cv/10/status",
                "CV parsing",
                poll_interval=0.001,
                timeout=1,
            )

    result = asyncio.run(run())
    assert result["status"] == "completed"
    output = capsys.readouterr().out
    assert output.count("CV parsing: waiting...") == 1
    assert output.count("CV parsing: completed") == 1
    assert "CV parsing: processing" not in output


def test_runner_updates_tty_while_api_request_is_pending(monkeypatch):
    class TTYCapture(io.StringIO):
        def isatty(self):
            return True

    output = TTYCapture()
    monkeypatch.setattr(sys, "stdout", output)
    attempts = 0

    async def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.sleep(1.1)
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(200, json={"status": "completed"})

    async def run():
        async with httpx.AsyncClient(
            base_url="http://test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await _wait_for_status(
                client,
                "/cv/10/status",
                "CV parsing",
                poll_interval=0.001,
                timeout=2,
            )

    assert asyncio.run(run())["status"] == "completed"
    assert "checking API" in output.getvalue()
    assert "completed" in output.getvalue()


def test_runner_rejects_non_pdf_input(tmp_path):
    cv_path = tmp_path / "candidate.txt"
    jd_path = tmp_path / "role.pdf"
    cv_path.write_text("CV")
    jd_path.write_bytes(b"%PDF-1.7 jd")

    async def run():
        async with httpx.AsyncClient(base_url="http://test") as client:
            return await run_full_pipeline(client, cv_path, jd_path)

    with pytest.raises(ValueError, match="CV file must be a PDF"):
        asyncio.run(run())
