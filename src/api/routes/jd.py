"""Job description API endpoints."""

import json
import logging
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from ...database.db import create_jd, get_db_path, get_jd, list_jds, update_jd_status
from ...parsers.jd_parser import parse_jd

router = APIRouter(prefix="/jd", tags=["jd"])
logger = logging.getLogger(__name__)

MAX_JD_BYTES = 10 * 1024 * 1024
UPLOAD_DIR = Path("uploads")
ALLOWED_FILE_EXTENSIONS = {".pdf", ".txt", ".html"}


def _serialize_jd(jd: dict) -> dict:
    """Return parsed JD JSON as an object rather than its SQLite string form."""
    serialized = dict(jd)
    if serialized["parsed_data"] is not None:
        serialized["parsed_data"] = json.loads(serialized["parsed_data"])
    return serialized


def _get_jd_or_404(jd_id: int) -> dict:
    jd = get_jd(jd_id)
    if jd is None:
        raise HTTPException(status_code=404, detail="JD not found")
    return jd


async def _parse_uploaded_jd(
    jd_id: int,
    source_type: str,
    source_data: str,
    source_url: str | None,
    db_path: str,
    file_path: Path | None = None,
) -> None:
    """Run JD parsing in background and persist failures to its DB record."""
    try:
        update_jd_status(jd_id, "processing", db_path=db_path)
        parser_source_type = source_type
        if source_type == "file" and file_path is not None:
            if file_path.suffix.lower() == ".pdf":
                parser_source_type = "pdf"
                source_data = str(file_path)
            else:
                source_data = file_path.read_text(encoding="utf-8")
        await parse_jd(
            parser_source_type,
            source_data,
            source_url=source_url,
            jd_id=jd_id,
            db_path=db_path,
        )
    except Exception as err:
        logger.exception("JD parsing failed for JD %s", jd_id)
        update_jd_status(
            jd_id,
            "failed",
            error_message=f"{type(err).__name__}: {str(err).strip() or repr(err)}",
            db_path=db_path,
        )
    finally:
        if file_path is not None:
            file_path.unlink(missing_ok=True)


@router.post("", status_code=202)
async def create_and_parse_jd(
    background_tasks: BackgroundTasks,
    source_type: Literal["pasted", "url", "file"] = Form(...),
    text: str | None = Form(None),
    url: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> dict[str, int | str]:
    """Create a JD from pasted text, URL, or uploaded PDF/text/HTML file."""
    source_data = ""
    source_url = None
    title = "Job Description"
    file_path = None

    if source_type == "pasted":
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Pasted JD text is required")
        source_data = text
        title = "Pasted JD"
    elif source_type == "url":
        if not url or not url.strip():
            raise HTTPException(status_code=400, detail="JD URL is required")
        source_url = url.strip()
    else:
        if file is None or not file.filename:
            raise HTTPException(status_code=400, detail="JD file is required")
        filename = Path(file.filename).name
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_FILE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="JD file must be PDF, TXT, or HTML",
            )
        content = await file.read(MAX_JD_BYTES + 1)
        await file.close()
        if len(content) > MAX_JD_BYTES:
            raise HTTPException(status_code=413, detail="JD file exceeds the 10 MB limit")
        if extension == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF")
        else:
            try:
                content.decode("utf-8")
            except UnicodeDecodeError as err:
                raise HTTPException(
                    status_code=400, detail="JD text file must use UTF-8 encoding"
                ) from err
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        file_path = UPLOAD_DIR / f"jd_{uuid4().hex}{extension}"
        file_path.write_bytes(content)
        title = filename[:255]
        source_url = filename

    db_path = get_db_path()
    try:
        jd_id = create_jd(
            title,
            source_data,
            source_type=source_type,
            source_url=source_url,
            db_path=db_path,
        )
    except Exception:
        if file_path is not None:
            file_path.unlink(missing_ok=True)
        raise

    background_tasks.add_task(
        _parse_uploaded_jd,
        jd_id,
        source_type,
        source_data,
        source_url,
        db_path,
        file_path,
    )
    return {"jd_id": jd_id, "status": "pending"}


@router.get("")
def get_jds() -> list[dict]:
    """List job descriptions, newest first."""
    return [_serialize_jd(jd) for jd in list_jds()]


@router.get("/{jd_id}/status")
def get_jd_status(jd_id: int) -> dict:
    """Get parsing status for a job description."""
    jd = _get_jd_or_404(jd_id)
    return {
        "jd_id": jd["id"],
        "status": jd["status"],
        "error_message": jd["error_message"],
        "updated_at": jd["updated_at"],
    }


@router.get("/{jd_id}")
def get_jd_details(jd_id: int) -> dict:
    """Get full job description details."""
    return _serialize_jd(_get_jd_or_404(jd_id))
