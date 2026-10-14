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
    read_retries = 2 if method.upper() == "GET" else 0
    for attempt in range(read_retries + 1):
        try:
            response = await client.request(method, path, **kwargs)
            break
        except httpx.TimeoutException as err:
            raise RuntimeError(
                f"{method} {path} timed out ({type(err).__name__})"
            ) from err
        except httpx.ReadError as err:
            if attempt < read_retries:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(
                f"{method} {path}: API closed the connection while sending its "
                f"response after {attempt + 1} attempts. Check API server logs "
                "for the request traceback and verify GET /health."
            ) from err
        except httpx.RequestError as err:
            detail = str(err).strip() or repr(err)
            raise RuntimeError(
                f"{method} {path} request failed ({type(err).__name__}): {detail}"
            ) from err
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as err:
        try:
            payload = response.json()
            detail = payload.get("detail") if isinstance(payload, dict) else None
        except ValueError:
            detail = None
        detail = detail or response.text.strip() or response.reason_phrase
        raise RuntimeError(
            f"{method} {path} failed ({response.status_code}): {detail}"
        ) from err
    return response.json()


async def _wait_for_status(
    client: httpx.AsyncClient,
    path: str,
    label: str,
    poll_interval: float,
    timeout: float,
) -> dict:
    deadline = time.monotonic() + timeout
    started_at = time.monotonic()
    spinner = ("|", "/", "-", "\\")
    frame = 0
    line_width = 0
    is_tty = sys.stdout.isatty()
    last_log = started_at
    if not is_tty:
        print(f"   {label}: waiting...", flush=True)

    while True:
        request_task = asyncio.create_task(_request(client, "GET", path))
        while not request_task.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)
                raise TimeoutError(
                    f"Timed out waiting for {label} after {timeout:g}s "
                    f"(last API request: GET {path})"
                )
            await asyncio.wait({request_task}, timeout=min(1.0, remaining))
            if request_task.done():
                break
            elapsed = int(time.monotonic() - started_at)
            if is_tty:
                line = (
                    f"   {label}: {spinner[frame % len(spinner)]} "
                    f"checking API ({elapsed}s)"
                )
                sys.stdout.write(
                    f"\r{line}{' ' * max(0, line_width - len(line))}"
                )
                sys.stdout.flush()
                line_width = max(line_width, len(line))
                frame += 1
            elif time.monotonic() - last_log >= 10:
                print(f"   {label}: still waiting ({elapsed}s)", flush=True)
                last_log = time.monotonic()

        result = request_task.result()
        status = result["status"]
        if is_tty:
            elapsed = int(time.monotonic() - started_at)
            line = f"   {label}: {status} ({elapsed}s)"
            sys.stdout.write(
                f"\r{line}{' ' * max(0, line_width - len(line))}"
            )
            sys.stdout.flush()
            line_width = max(line_width, len(line))
        if status == "completed":
            if is_tty:
                line = f"   {label}: completed"
                print(f"\r{line}{' ' * max(0, line_width - len(line))}")
            else:
                print(f"   {label}: completed", flush=True)
            return result
        if status == "failed":
            if is_tty:
                line = f"   {label}: failed"
                print(f"\r{line}{' ' * max(0, line_width - len(line))}")
            else:
                print(f"   {label}: failed", flush=True)
            detail = result.get("error_message")
            raise RuntimeError(
                f"{label} failed: {detail.strip() if isinstance(detail, str) and detail.strip() else 'API returned no error details; inspect server logs'}"
            )
        if time.monotonic() >= deadline:
            if is_tty:
                line = f"   {label}: timed out"
                print(f"\r{line}{' ' * max(0, line_width - len(line))}")
            raise TimeoutError(
                f"Timed out waiting for {label} after {timeout:g}s"
            )
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
        detail = str(err).strip() or repr(err) or type(err).__name__
        print(f"\nERROR [{type(err).__name__}]: {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
