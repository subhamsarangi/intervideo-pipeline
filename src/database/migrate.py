"""Database migration utilities"""

import sqlite3
from pathlib import Path
from typing import Optional
from src.database.db import get_db_path


def migrate_up(db_path: Optional[str] = None) -> None:
    """Apply all pending migrations"""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create migrations table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()

    # Get applied migrations
    cursor.execute("SELECT name FROM migrations")
    applied = {row[0] for row in cursor.fetchall()}

    # Define migrations
    migrations = {
        "001_init_schema": init_schema,
        "002_add_llm_tracking": add_llm_tracking_columns,
    }

    # Apply pending migrations
    for name, migration_func in migrations.items():
        if name not in applied:
            print(f"Applying migration: {name}")
            migration_func(conn)
            cursor.execute("INSERT INTO migrations (name) VALUES (?)", (name,))
            conn.commit()

    conn.close()
    print("Migrations complete")


def init_schema(conn: sqlite3.Connection) -> None:
    """001: Initialize database schema"""
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path) as f:
        schema = f.read()
    conn.executescript(schema)


def add_llm_tracking_columns(conn: sqlite3.Connection) -> None:
    """002: Add LLM tracking columns to cv_documents and jd_documents"""
    cursor = conn.cursor()

    # Add llm_calls_count to cv_documents
    try:
        cursor.execute(
            "ALTER TABLE cv_documents ADD COLUMN llm_calls_count INTEGER DEFAULT 0"
        )
    except sqlite3.OperationalError:
        pass  # Column might already exist

    # Add other tracking columns to cv_documents
    try:
        cursor.execute("ALTER TABLE cv_documents ADD COLUMN llm_calls_log JSON")
        cursor.execute(
            "ALTER TABLE cv_documents ADD COLUMN total_input_tokens INTEGER DEFAULT 0"
        )
        cursor.execute(
            "ALTER TABLE cv_documents ADD COLUMN total_output_tokens INTEGER DEFAULT 0"
        )
        cursor.execute(
            "ALTER TABLE cv_documents ADD COLUMN total_cost_usd REAL DEFAULT 0.0"
        )
    except sqlite3.OperationalError:
        pass

    # Add llm_calls_count to jd_documents
    try:
        cursor.execute(
            "ALTER TABLE jd_documents ADD COLUMN llm_calls_count INTEGER DEFAULT 0"
        )
    except sqlite3.OperationalError:
        pass  # Column might already exist

    # Add other tracking columns to jd_documents
    try:
        cursor.execute("ALTER TABLE jd_documents ADD COLUMN llm_calls_log JSON")
        cursor.execute(
            "ALTER TABLE jd_documents ADD COLUMN total_input_tokens INTEGER DEFAULT 0"
        )
        cursor.execute(
            "ALTER TABLE jd_documents ADD COLUMN total_output_tokens INTEGER DEFAULT 0"
        )
        cursor.execute(
            "ALTER TABLE jd_documents ADD COLUMN total_cost_usd REAL DEFAULT 0.0"
        )
    except sqlite3.OperationalError:
        pass


def migrate_down(db_path: Optional[str] = None) -> None:
    """Rollback last migration"""
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, name FROM migrations ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()

    if row:
        migration_id, name = row
        print(f"Rolling back migration: {name}")
        cursor.execute("DELETE FROM migrations WHERE id = ?", (migration_id,))
        conn.commit()
        print("Rollback complete")
    else:
        print("No migrations to rollback")

    conn.close()


def add_llm_tracking_columns(conn: sqlite3.Connection) -> None:
    """002: Add LLM tracking columns to cv_documents and jd_documents"""
    cursor = conn.cursor()

    # Add columns to cv_documents
    try:
        cursor.execute("ALTER TABLE cv_documents ADD COLUMN llm_calls_log JSON")
        cursor.execute(
            "ALTER TABLE cv_documents ADD COLUMN total_input_tokens INTEGER DEFAULT 0"
        )
        cursor.execute(
            "ALTER TABLE cv_documents ADD COLUMN total_output_tokens INTEGER DEFAULT 0"
        )
        cursor.execute(
            "ALTER TABLE cv_documents ADD COLUMN total_cost_usd REAL DEFAULT 0.0"
        )
    except sqlite3.OperationalError:
        pass  # Columns might already exist

    # Add columns to jd_documents
    try:
        cursor.execute("ALTER TABLE jd_documents ADD COLUMN llm_calls_log JSON")
        cursor.execute(
            "ALTER TABLE jd_documents ADD COLUMN total_input_tokens INTEGER DEFAULT 0"
        )
        cursor.execute(
            "ALTER TABLE jd_documents ADD COLUMN total_output_tokens INTEGER DEFAULT 0"
        )
        cursor.execute(
            "ALTER TABLE jd_documents ADD COLUMN total_cost_usd REAL DEFAULT 0.0"
        )
    except sqlite3.OperationalError:
        pass  # Columns might already exist


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "down":
        migrate_down()
    else:
        migrate_up()
