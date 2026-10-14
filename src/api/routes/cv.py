"""CV upload endpoints."""

import json
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from ...database.db import create_cv, get_cv, get_db_path, list_cvs, update_cv_status
from ...parsers.cv_parser import parse_cv

router = APIRouter(prefix="/cv", tags=["cv"])
logger = logging.getLogger(__name__)

MAX_CV_BYTES = 10 * 1024 * 1024
UPLOAD_DIR = Path("uploads")


def _serialize_cv(cv: dict) -> dict:
    """Return parsed CV JSON as an object rather than its SQLite string form."""
    serialized = dict(cv)
    if serialized["parsed_data"] is not None:
        serialized["parsed_data"] = json.loads(serialized["parsed_data"])
    return serialized


def _get_cv_or_404(cv_id: int) -> dict:
    cv = get_cv(cv_id)
    if cv is None:
        raise HTTPException(status_code=404, detail="CV not found")
    return cv


async def _parse_uploaded_cv(cv_id: int, file_path: Path, db_path: str) -> None:
    """Parse uploaded CV and persist any background failure to its DB record."""
    try:
        update_cv_status(cv_id, "processing", db_path=db_path)
        await parse_cv(str(file_path), cv_id=cv_id, db_path=db_path)
    except Exception as err:
        logger.exception("CV parsing failed for CV %s", cv_id)
        update_cv_status(
            cv_id,
            "failed",
            error_message=f"{type(err).__name__}: {str(err).strip() or repr(err)}",
            db_path=db_path,
        )
    finally:
        file_path.unlink(missing_ok=True)


@router.post("/upload", status_code=202)
async def upload_cv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, int | str]:
    """Save a PDF CV and parse it asynchronously."""
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Upload must be a PDF file")

    content = await file.read(MAX_CV_BYTES + 1)
    await file.close()
    if len(content) > MAX_CV_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 10 MB upload limit")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / f"cv_{uuid4().hex}.pdf"
    file_path.write_bytes(content)

    title = filename[:255]
    db_path = get_db_path()
    try:
        cv_id = create_cv(title, "", db_path=db_path)
    except Exception:
        file_path.unlink(missing_ok=True)
        raise

    background_tasks.add_task(_parse_uploaded_cv, cv_id, file_path, db_path)
    return {"cv_id": cv_id, "status": "pending"}


@router.get("")
def get_cvs() -> list[dict]:
    """List CV documents, newest first."""
    return [_serialize_cv(cv) for cv in list_cvs()]


@router.get("/{cv_id}/status")
def get_cv_status(cv_id: int) -> dict:
    """Get parsing status for a CV document."""
    cv = _get_cv_or_404(cv_id)
    return {
        "cv_id": cv["id"],
        "status": cv["status"],
        "error_message": cv["error_message"],
        "updated_at": cv["updated_at"],
    }


@router.get("/{cv_id}")
def get_cv_details(cv_id: int) -> dict:
    """Get full CV document details."""
    return _serialize_cv(_get_cv_or_404(cv_id))
