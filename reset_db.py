#!/usr/bin/env python3
"""
Reset database - deletes all data and recreates schema
"""

import os
import sys
import sqlite3
from pathlib import Path

from src.database.db import get_db_path, init_db


def show_table_contents():
    """Show counts and preview of each table"""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    tables = [
        "cv_documents",
        "jd_documents",
        "pipeline_runs",
        "pipeline_steps",
        "questions",
        "fitness_scores",
        "interview_sessions",
        "interview_turns",
    ]
    
    print("\n" + "=" * 80)
    print("CURRENT DATABASE CONTENTS")
    print("=" * 80)
    
    total_records = 0
    
    for table in tables:
        try:
            # Get count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            total_records += count
            
            print(f"\n{table}: {count} records")
            
            if count > 0:
                # Show first 3 records
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                rows = cursor.fetchall()
                
                if rows:
                    # Show column names
                    cols = [desc[0] for desc in cursor.description]
                    key_cols = [c for c in cols if c in ['id', 'title', 'candidate_name', 'job_title', 'question_text', 'status']][:3]
                    
                    for row in rows:
                        row_dict = dict(row)
                        preview = " | ".join(f"{col}={row_dict.get(col, 'N/A')}" for col in key_cols)
                        print(f"  - {preview}")
                    
                    if count > 3:
                        print(f"  ... and {count - 3} more")
        
        except sqlite3.OperationalError:
            # Table doesn't exist
            pass
    
    conn.close()
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {total_records} records across all tables")
    print("=" * 80)
    
    return total_records


def main():
    db_path = get_db_path()
    db_file = Path(db_path)
    
    print("=" * 80)
    print("DATABASE RESET")
    print("=" * 80)
    print(f"\nDatabase: {db_file.absolute()}\n")
    
    if not db_file.exists():
        print("⚠️  Database does not exist yet.")
        print("   Creating fresh database...\n")
        init_db()
        print("✓ Database created successfully.\n")
        return 0
    
    # Show current contents
    try:
        total = show_table_contents()
    except Exception as e:
        print(f"\n⚠️  Could not read database: {e}")
        total = 0
    
    if total == 0:
        print("\nDatabase is already empty. Nothing to delete.")
        return 0
    
    print("\n⚠️  WARNING: This will DELETE ALL DATA above!")
    print()
    
    response = input("Type 'Y' to confirm deletion: ").strip()
    
    if response != "Y":
        print("\n❌ Aborted. No changes made.\n")
        return 1
    
    print("\n🗑️  Deleting database file...")
    try:
        os.remove(db_file)
        print("   ✓ Deleted")
    except Exception as e:
        print(f"   ❌ Failed to delete: {e}")
        return 1
    
    print("\n🔨 Recreating database schema...")
    try:
        init_db()
        print("   ✓ Schema created")
    except Exception as e:
        print(f"   ❌ Failed to create schema: {e}")
        return 1
    
    print("\n" + "=" * 80)
    print("✓ DATABASE RESET COMPLETE")
    print("=" * 80)
    print("Database is now empty and ready for new data.\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
