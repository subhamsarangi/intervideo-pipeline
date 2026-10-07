"""Database layer for RecRoot"""

import sqlite3
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List
import os


def get_db_path() -> str:
    """Get database path from environment or use default"""
    return os.getenv("DATABASE_PATH", "recroot.db")


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database from schema.sql"""
    if db_path is None:
        db_path = get_db_path()

    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path) as f:
        schema = f.read()

    conn = sqlite3.connect(db_path)
    conn.executescript(schema)
    conn.commit()
    conn.close()


@contextmanager
def get_connection(db_path: Optional[str] = None):
    """Context manager for database connections"""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# CV Documents
def create_cv(title: str, raw_text: str, db_path: Optional[str] = None) -> int:
    """Create a new CV document"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO cv_documents (title, raw_text, status) VALUES (?, ?, 'pending')",
            (title, raw_text),
        )
        return cursor.lastrowid


def get_cv(cv_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get CV by ID"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cv_documents WHERE id = ?", (cv_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_cvs(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all CVs"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cv_documents ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]


def update_cv_status(
    cv_id: int,
    status: str,
    parsed_data: Optional[Dict] = None,
    error_message: Optional[str] = None,
    llm_calls_count: Optional[int] = None,
    llm_calls_log: Optional[list] = None,
    total_input_tokens: Optional[int] = None,
    total_output_tokens: Optional[int] = None,
    total_cost_usd: Optional[float] = None,
    db_path: Optional[str] = None,
) -> None:
    """Update CV parsing status with optional LLM tracking data"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Build dynamic UPDATE query based on what's provided
        updates = ["status = ?"]
        params = [status]

        if parsed_data is not None:
            updates.append("parsed_data = ?")
            params.append(json.dumps(parsed_data))

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        if llm_calls_count is not None:
            updates.append("llm_calls_count = ?")
            params.append(llm_calls_count)

        if llm_calls_log is not None:
            updates.append("llm_calls_log = ?")
            params.append(json.dumps(llm_calls_log))

        if total_input_tokens is not None:
            updates.append("total_input_tokens = ?")
            params.append(total_input_tokens)

        if total_output_tokens is not None:
            updates.append("total_output_tokens = ?")
            params.append(total_output_tokens)

        if total_cost_usd is not None:
            updates.append("total_cost_usd = ?")
            params.append(total_cost_usd)

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(cv_id)

        query = f"UPDATE cv_documents SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)


# JD Documents
def create_jd(
    title: str,
    raw_text: str,
    source_type: str = "pasted",
    source_url: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Create a new JD document"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO jd_documents (title, raw_text, source_type, source_url, status) VALUES (?, ?, ?, ?, 'pending')",
            (title, raw_text, source_type, source_url),
        )
        return cursor.lastrowid


def get_jd(jd_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get JD by ID"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jd_documents WHERE id = ?", (jd_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_jds(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all JDs"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jd_documents ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]


def update_jd_status(
    jd_id: int,
    status: str,
    parsed_data: Optional[Dict] = None,
    error_message: Optional[str] = None,
    llm_calls_count: Optional[int] = None,
    llm_calls_log: Optional[list] = None,
    total_input_tokens: Optional[int] = None,
    total_output_tokens: Optional[int] = None,
    total_cost_usd: Optional[float] = None,
    db_path: Optional[str] = None,
) -> None:
    """Update JD parsing status with optional LLM tracking data"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Build dynamic UPDATE query based on what's provided
        updates = ["status = ?"]
        params = [status]

        if parsed_data is not None:
            updates.append("parsed_data = ?")
            params.append(json.dumps(parsed_data))

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        if llm_calls_count is not None:
            updates.append("llm_calls_count = ?")
            params.append(llm_calls_count)

        if llm_calls_log is not None:
            updates.append("llm_calls_log = ?")
            params.append(json.dumps(llm_calls_log))

        if total_input_tokens is not None:
            updates.append("total_input_tokens = ?")
            params.append(total_input_tokens)

        if total_output_tokens is not None:
            updates.append("total_output_tokens = ?")
            params.append(total_output_tokens)

        if total_cost_usd is not None:
            updates.append("total_cost_usd = ?")
            params.append(total_cost_usd)

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(jd_id)

        query = f"UPDATE jd_documents SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)


# Pipeline Runs
def create_pipeline_run(
    cv_id: int,
    jd_id: int,
    prompt_version: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Create a new pipeline run"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO pipeline_runs (cv_id, jd_id, prompt_version, status) VALUES (?, ?, ?, 'pending')",
            (cv_id, jd_id, prompt_version),
        )
        return cursor.lastrowid


def get_pipeline_run(
    run_id: int, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get pipeline run by ID"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pipeline_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_pipeline_run_status(
    run_id: int,
    status: str,
    output_json: Optional[Dict] = None,
    error_message: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Update pipeline run status"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if output_json:
            cursor.execute(
                "UPDATE pipeline_runs SET status = ?, output_json = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, json.dumps(output_json), error_message, run_id),
            )
        else:
            cursor.execute(
                "UPDATE pipeline_runs SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, error_message, run_id),
            )


def list_pipeline_runs(
    limit: int = 50, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List recent pipeline runs"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM pipeline_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]


# Pipeline Steps
def create_pipeline_step(
    run_id: int, step_name: str, db_path: Optional[str] = None
) -> int:
    """Create a new pipeline step"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO pipeline_steps (run_id, step_name, status) VALUES (?, ?, 'pending')",
            (run_id, step_name),
        )
        return cursor.lastrowid


def update_pipeline_step(
    step_id: int,
    status: str,
    output_json: Optional[Dict] = None,
    error_message: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Update pipeline step"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if output_json:
            cursor.execute(
                "UPDATE pipeline_steps SET status = ?, output_json = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, json.dumps(output_json), error_message, step_id),
            )
        else:
            cursor.execute(
                "UPDATE pipeline_steps SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, error_message, step_id),
            )


def get_pipeline_steps(
    run_id: int, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get all steps for a pipeline run"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM pipeline_steps WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


# Questions
def create_question(
    run_id: int,
    question_text: str,
    source_type: Optional[str] = None,
    source_id: Optional[str] = None,
    evidence_snippet: Optional[str] = None,
    evidence_span: Optional[Dict] = None,
    db_path: Optional[str] = None,
) -> int:
    """Create a new question"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO questions (run_id, question_text, source_type, source_id, evidence_snippet, evidence_span) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                question_text,
                source_type,
                source_id,
                evidence_snippet,
                json.dumps(evidence_span) if evidence_span else None,
            ),
        )
        return cursor.lastrowid


def update_question_scores(
    question_id: int,
    relevance_score: Optional[float] = None,
    groundedness_score: Optional[float] = None,
    redundancy_score: Optional[float] = None,
    db_path: Optional[str] = None,
) -> None:
    """Update question judge scores"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE questions SET judge_relevance_score = ?, judge_groundedness_score = ?, judge_redundancy_score = ? WHERE id = ?",
            (relevance_score, groundedness_score, redundancy_score, question_id),
        )


def get_questions(run_id: int, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all questions for a pipeline run"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM questions WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


# Fitness Scores
def create_fitness_score(
    run_id: int,
    requirement_text: str,
    requirement_type: str,
    match_score: float,
    evidence: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Create a fitness score entry"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO fitness_scores (run_id, requirement_text, requirement_type, match_score, evidence) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, requirement_text, requirement_type, match_score, evidence),
        )
        return cursor.lastrowid


def get_fitness_scores(
    run_id: int, db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get all fitness scores for a run"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM fitness_scores WHERE run_id = ?", (run_id,))
        return [dict(row) for row in cursor.fetchall()]


# Run Summaries
def create_run_summary(
    run_id: int,
    total_questions: int,
    avg_judge_score: float,
    fitness_score: float,
    fitness_weighted_score: float,
    db_path: Optional[str] = None,
) -> int:
    """Create a run summary"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO run_summaries (run_id, total_questions, avg_judge_score, fitness_score, fitness_weighted_score) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                run_id,
                total_questions,
                avg_judge_score,
                fitness_score,
                fitness_weighted_score,
            ),
        )
        return cursor.lastrowid


def get_run_summary(
    run_id: int, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get run summary"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM run_summaries WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


# LLM Cost Analysis Functions
def get_cv_cost(cv_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get CV parsing cost and LLM call details"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT llm_calls_count, llm_calls_log, total_input_tokens, total_output_tokens, total_cost_usd FROM cv_documents WHERE id = ?",
            (cv_id,),
        )
        row = cursor.fetchone()
        if not row:
            return {}

        return {
            "cv_id": cv_id,
            "llm_calls_count": row[0] or 0,
            "llm_calls_log": json.loads(row[1]) if row[1] else [],
            "total_input_tokens": row[2] or 0,
            "total_output_tokens": row[3] or 0,
            "total_cost_usd": row[4] or 0.0,
        }


def get_jd_cost(jd_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get JD parsing cost and LLM call details"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT llm_calls_count, llm_calls_log, total_input_tokens, total_output_tokens, total_cost_usd FROM jd_documents WHERE id = ?",
            (jd_id,),
        )
        row = cursor.fetchone()
        if not row:
            return {}

        return {
            "jd_id": jd_id,
            "llm_calls_count": row[0] or 0,
            "llm_calls_log": json.loads(row[1]) if row[1] else [],
            "total_input_tokens": row[2] or 0,
            "total_output_tokens": row[3] or 0,
            "total_cost_usd": row[4] or 0.0,
        }


def get_average_costs(
    document_type: str = "cv", db_path: Optional[str] = None
) -> Dict[str, float]:
    """Calculate average parsing costs by document type"""
    table = "cv_documents" if document_type == "cv" else "jd_documents"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT AVG(llm_calls_count), AVG(total_input_tokens), AVG(total_output_tokens), AVG(total_cost_usd) FROM {table} WHERE total_cost_usd > 0"
        )
        row = cursor.fetchone()

        if not row or not any(row):
            return {
                "avg_calls": 0,
                "avg_input_tokens": 0,
                "avg_output_tokens": 0,
                "avg_cost_usd": 0.0,
                "document_count": 0,
            }

        # Get document count
        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE total_cost_usd > 0")
        count = cursor.fetchone()[0]

        return {
            "avg_calls": round(row[0] or 0, 1),
            "avg_input_tokens": round(row[1] or 0),
            "avg_output_tokens": round(row[2] or 0),
            "avg_cost_usd": round(row[3] or 0, 6),
            "document_count": count,
        }


def display_cost_report(db_path: Optional[str] = None):
    """Display formatted cost analysis report"""
    cv_avg = get_average_costs("cv", db_path)
    jd_avg = get_average_costs("jd", db_path)

    print("📊 LLM Cost Analysis Report")
    print("=" * 40)

    if cv_avg["document_count"] > 0:
        print(f"CV Parsing Average ({cv_avg['document_count']} documents):")
        print(f"  Calls: {cv_avg['avg_calls']}")
        print(
            f"  Tokens: {cv_avg['avg_input_tokens']} input + {cv_avg['avg_output_tokens']} output"
        )
        print(f"  Cost: ${cv_avg['avg_cost_usd']:.6f} per CV")
        print()

    if jd_avg["document_count"] > 0:
        print(f"JD Parsing Average ({jd_avg['document_count']} documents):")
        print(f"  Calls: {jd_avg['avg_calls']}")
        print(
            f"  Tokens: {jd_avg['avg_input_tokens']} input + {jd_avg['avg_output_tokens']} output"
        )
        print(f"  Cost: ${jd_avg['avg_cost_usd']:.6f} per JD")
        print()

    if cv_avg["document_count"] > 0 and jd_avg["document_count"] > 0:
        combined_cost = cv_avg["avg_cost_usd"] + jd_avg["avg_cost_usd"]
        print(f"Combined Average: ${combined_cost:.6f} per CV+JD pair")
        print(f"1000 evaluations: ${combined_cost * 1000:.2f}")

    if cv_avg["document_count"] == 0 and jd_avg["document_count"] == 0:
        print("No parsing cost data available yet.")
