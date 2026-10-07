"""Clean up dummy data from database"""

import sqlite3
from src.database.db import get_db_path


DUMMY_MARKER = "DUMMY_TEST_DATA"


def cleanup():
    """Delete all dummy data marked rows"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    deleted_counts = {}

    # Delete from cv_documents
    cursor.execute(
        "DELETE FROM cv_documents WHERE title LIKE ?", (f"%{DUMMY_MARKER}%",)
    )
    deleted_counts["cv_documents"] = cursor.rowcount

    # Delete from jd_documents
    cursor.execute(
        "DELETE FROM jd_documents WHERE title LIKE ?", (f"%{DUMMY_MARKER}%",)
    )
    deleted_counts["jd_documents"] = cursor.rowcount

    # Get IDs of deleted runs to cascade delete
    cursor.execute(
        "SELECT id FROM pipeline_runs WHERE cv_id NOT IN (SELECT id FROM cv_documents) OR jd_id NOT IN (SELECT id FROM jd_documents)"
    )
    orphaned_runs = [row[0] for row in cursor.fetchall()]

    # Delete related data
    for run_id in orphaned_runs:
        cursor.execute("DELETE FROM pipeline_steps WHERE run_id = ?", (run_id,))
        deleted_counts["pipeline_steps"] = (
            deleted_counts.get("pipeline_steps", 0) + cursor.rowcount
        )

        cursor.execute("DELETE FROM questions WHERE run_id = ?", (run_id,))
        deleted_counts["questions"] = (
            deleted_counts.get("questions", 0) + cursor.rowcount
        )

        cursor.execute("DELETE FROM fitness_scores WHERE run_id = ?", (run_id,))
        deleted_counts["fitness_scores"] = (
            deleted_counts.get("fitness_scores", 0) + cursor.rowcount
        )

        cursor.execute("DELETE FROM run_summaries WHERE run_id = ?", (run_id,))
        deleted_counts["run_summaries"] = (
            deleted_counts.get("run_summaries", 0) + cursor.rowcount
        )

    # Delete orphaned runs
    cursor.execute(
        "DELETE FROM pipeline_runs WHERE id IN ({})".format(
            ",".join("?" * len(orphaned_runs))
        ),
        orphaned_runs,
    )
    deleted_counts["pipeline_runs"] = cursor.rowcount

    conn.commit()
    conn.close()

    print(f"✅ Cleaned up dummy data marked with: {DUMMY_MARKER}")
    for table, count in deleted_counts.items():
        if count > 0:
            print(f"   {table}: {count} rows deleted")

    if sum(deleted_counts.values()) == 0:
        print("   (No dummy data found)")


if __name__ == "__main__":
    cleanup()
