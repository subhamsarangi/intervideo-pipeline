#!/usr/bin/env python3
"""Run CV/JD parsing and question generation exclusively through the HTTP API."""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import httpx


async def _request(
    client: httpx.AsyncClient, method: str, path: str, **kwargs: Any
) -> Any:
    response = await client.request(method, path, **kwargs)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as err:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"{method} {path} failed ({response.status_code}): {detail}") from err
    return response.json()


async def _wait_for_status(
    client: httpx.AsyncClient,
    path: str,
    label: str,
    poll_interval: float,
    timeout: float,
) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        result = await _request(client, "GET", path)
        status = result["status"]
        print(f"   {label}: {status}", flush=True)
        if status == "completed":
            return result
        if status == "failed":
            raise RuntimeError(
                f"{label} failed: {result.get('error_message') or 'unknown error'}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {label} after {timeout:g}s")
        await asyncio.sleep(poll_interval)


async def run_full_pipeline(
    client: httpx.AsyncClient,
    cv_path: Path,
    jd_path: Path,
    poll_interval: float = 2.0,
    timeout: float = 600.0,
) -> int:
    """Upload input PDFs, run pipeline, and print generated questions."""
    for label, path in (("CV", cv_path), ("JD", jd_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"{label} file must be a PDF: {path}")

    print(f"CV: {cv_path}")
    with cv_path.open("rb") as cv_file:
        cv_upload = await _request(
            client,
            "POST",
            "/cv/upload",
            files={"file": (cv_path.name, cv_file, "application/pdf")},
        )
    cv_id = cv_upload["cv_id"]
    await _wait_for_status(
        client, f"/cv/{cv_id}/status", "CV parsing", poll_interval, timeout
    )

    print(f"\nJD: {jd_path}")
    with jd_path.open("rb") as jd_file:
        jd_upload = await _request(
            client,
            "POST",
            "/jd",
            data={"source_type": "file"},
            files={"file": (jd_path.name, jd_file, "application/pdf")},
        )
    jd_id = jd_upload["jd_id"]
    await _wait_for_status(
        client, f"/jd/{jd_id}/status", "JD parsing", poll_interval, timeout
    )

    run_response = await _request(
        client, "POST", "/pipeline/start", json={"cv_id": cv_id, "jd_id": jd_id}
    )
    run_id = run_response["run_id"]
    await _wait_for_status(
        client, f"/pipeline/{run_id}", "Pipeline", poll_interval, timeout
    )

    questions = await _request(
        client, "GET", f"/pipeline/{run_id}/questions", params={"status": "candidate"}
    )
    operations = await _request(client, "GET", f"/pipeline/{run_id}/operations")
    print(f"\nCompleted | Run: {run_id} | CV: {cv_id} | JD: {jd_id}")
    print(f"Questions: {len(questions)}")
    for index, question in enumerate(questions[:3], start=1):
        print(f"  {index}. {question.get('question_text', '')}")
    if len(questions) > 3:
        print(f"  ... and {len(questions) - 3} more")
    print(f"Pipeline operation cost: ${operations['total_cost_usd']:.6f}")
    return 0


async def _async_main(args: argparse.Namespace) -> int:
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(base_url=args.api_url.rstrip("/"), timeout=timeout) as client:
        return await run_full_pipeline(
            client,
            args.cv,
            args.jd,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument(
        "--cv", type=Path, default=Path("tests/fixtures/cv_jd_testing_doc1.pdf")
    )
    parser.add_argument(
        "--jd", type=Path, default=Path("tests/fixtures/jd_testing_doc1.pdf")
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.poll_interval <= 0 or args.timeout <= 0:
        parser.error("--poll-interval and --timeout must be greater than zero")
    try:
        return asyncio.run(_async_main(args))
    except (httpx.HTTPError, RuntimeError, TimeoutError, FileNotFoundError, ValueError) as err:
        print(f"\nERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
