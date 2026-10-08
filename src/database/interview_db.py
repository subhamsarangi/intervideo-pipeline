"""Database helpers for interview sessions, answers, and decisions (Phase 11b)"""

import json
from typing import Optional, Dict, Any, List
from datetime import datetime

from src.database.db import get_connection


def create_interview_session(
    session_id: str,
    run_id: int,
    total_questions: int,
    planned_duration_seconds: int,
    db_path: Optional[str] = None,
) -> int:
    """Create new interview session"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO interview_sessions 
               (session_id, run_id, total_questions, planned_duration_seconds, start_time, is_active)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 1)""",
            (session_id, run_id, total_questions, planned_duration_seconds),
        )
        return cursor.lastrowid


def get_interview_session(session_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get interview session by ID"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM interview_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_interview_session(
    session_id: str,
    end_time: Optional[datetime] = None,
    total_duration_seconds: Optional[int] = None,
    avg_answer_quality: Optional[float] = None,
    fatigue_detected: Optional[bool] = None,
    exit_reason: Optional[str] = None,
    is_active: Optional[bool] = None,
    db_path: Optional[str] = None,
) -> None:
    """Update interview session"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if end_time is not None:
            updates.append("end_time = ?")
            params.append(end_time)
        
        if total_duration_seconds is not None:
            updates.append("total_duration_seconds = ?")
            params.append(total_duration_seconds)
        
        if avg_answer_quality is not None:
            updates.append("avg_answer_quality = ?")
            params.append(avg_answer_quality)
        
        if fatigue_detected is not None:
            updates.append("fatigue_detected = ?")
            params.append(1 if fatigue_detected else 0)
        
        if exit_reason is not None:
            updates.append("exit_reason = ?")
            params.append(exit_reason)
        
        if is_active is not None:
            updates.append("is_active = ?")
            params.append(1 if is_active else 0)
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(session_id)
        
        query = f"UPDATE interview_sessions SET {', '.join(updates)} WHERE session_id = ?"
        cursor.execute(query, params)


def save_interview_answer(
    session_id: str,
    question_id: int,
    question_text: str,
    question_level: str,
    question_type: str,
    asked_at: datetime,
    answer_text: str,
    answered_at: datetime,
    duration_seconds: int,
    quality: str,
    feedback: str,
    follow_ups_asked: int = 0,
    follow_up_texts: Optional[List[str]] = None,
    db_path: Optional[str] = None,
) -> int:
    """Save an answer record"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO interview_answers
               (session_id, question_id, question_text, question_level, question_type,
                asked_at, answer_text, answered_at, duration_seconds, quality, feedback,
                follow_ups_asked, follow_up_texts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                question_id,
                question_text,
                question_level,
                question_type,
                asked_at,
                answer_text,
                answered_at,
                duration_seconds,
                quality,
                feedback,
                follow_ups_asked,
                json.dumps(follow_up_texts) if follow_up_texts else None,
            ),
        )
        return cursor.lastrowid


def get_interview_answer(answer_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get answer by ID"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM interview_answers WHERE id = ?", (answer_id,))
        row = cursor.fetchone()
        if row:
            row_dict = dict(row)
            if row_dict.get("follow_up_texts"):
                row_dict["follow_up_texts"] = json.loads(row_dict["follow_up_texts"])
            return row_dict
        return None


def list_session_answers(session_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all answers for a session"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM interview_answers WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        results = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            if row_dict.get("follow_up_texts"):
                row_dict["follow_up_texts"] = json.loads(row_dict["follow_up_texts"])
            results.append(row_dict)
        return results


def save_interview_decision(
    session_id: str,
    answer_id: int,
    action: str,
    follow_up: Optional[str] = None,
    reasoning: Optional[str] = None,
    confidence: Optional[float] = None,
    db_path: Optional[str] = None,
) -> int:
    """Save agent decision"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO interview_decisions
               (session_id, answer_id, action, follow_up, reasoning, confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, answer_id, action, follow_up, reasoning, confidence),
        )
        return cursor.lastrowid


def list_session_decisions(session_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all decisions for a session"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM interview_decisions WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_session_metrics(session_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Calculate metrics for a session"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Get session
        cursor.execute("SELECT * FROM interview_sessions WHERE session_id = ?", (session_id,))
        session_row = cursor.fetchone()
        if not session_row:
            return {}
        
        # Get all answers
        cursor.execute(
            "SELECT * FROM interview_answers WHERE session_id = ?",
            (session_id,),
        )
        answers = [dict(row) for row in cursor.fetchall()]
        
        if not answers:
            return dict(session_row)
        
        # Quality distribution
        quality_dist = {}
        for quality in ["EXCELLENT", "GOOD", "FAIR", "POOR", "OFF_TOPIC"]:
            count = len([a for a in answers if a["quality"] == quality])
            quality_dist[quality] = count
        
        # By level
        questions_by_level = {}
        performance_by_level = {}
        for level in ["junior", "mid", "senior"]:
            level_answers = [a for a in answers if a["question_level"] == level]
            questions_by_level[level] = len(level_answers)
            
            if level_answers:
                quality_scores = {
                    "EXCELLENT": 1.0,
                    "GOOD": 0.8,
                    "FAIR": 0.6,
                    "POOR": 0.3,
                    "OFF_TOPIC": 0.0,
                }
                scores = [quality_scores.get(a["quality"], 0.5) for a in level_answers]
                performance_by_level[level] = sum(scores) / len(scores)
            else:
                performance_by_level[level] = 0.0
        
        # Fatigue detection
        last_3 = answers[-3:] if len(answers) >= 3 else answers
        poor_answers = len([a for a in last_3 if a["quality"] in ["POOR", "OFF_TOPIC"]])
        fatigue = poor_answers >= 2
        
        return {
            "session_id": session_id,
            "total_questions": session_row["total_questions"],
            "questions_asked": len(answers),
            "quality_distribution": quality_dist,
            "avg_answer_quality": session_row["avg_answer_quality"],
            "total_duration_seconds": session_row["total_duration_seconds"],
            "fatigue_detected": bool(session_row["fatigue_detected"]),
            "questions_by_level": questions_by_level,
            "performance_by_level": performance_by_level,
            "exit_reason": session_row["exit_reason"],
        }


def list_run_sessions(run_id: int, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all interview sessions for a pipeline run"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM interview_sessions WHERE run_id = ? ORDER BY created_at DESC",
            (run_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


# Analytics queries

def get_session_quality_distribution(session_id: str, db_path: Optional[str] = None) -> Dict[str, int]:
    """Get count of each quality level for a session"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT quality, COUNT(*) as count FROM interview_answers 
               WHERE session_id = ? GROUP BY quality""",
            (session_id,),
        )
        return {row["quality"]: row["count"] for row in cursor.fetchall()}


def get_session_by_level(session_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Questions and performance breakdown by level"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        result = {}
        for level in ["junior", "mid", "senior"]:
            cursor.execute(
                """SELECT COUNT(*) as total, 
                          SUM(CASE WHEN quality IN ('excellent', 'good') THEN 1 ELSE 0 END) as strong
                   FROM interview_answers
                   WHERE session_id = ? AND question_level = ?""",
                (session_id, level),
            )
            row = cursor.fetchone()
            if row and row["total"] > 0:
                result[level] = {
                    "questions": row["total"],
                    "strong_answers": row["strong"] or 0,
                    "strength_pct": (row["strong"] or 0) / row["total"] * 100,
                }
        
        return result


def get_session_decision_breakdown(session_id: str, db_path: Optional[str] = None) -> Dict[str, int]:
    """Count of each decision type"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT action, COUNT(*) as count FROM interview_decisions
               WHERE session_id = ? GROUP BY action""",
            (session_id,),
        )
        return {row["action"]: row["count"] for row in cursor.fetchall()}


def get_average_answer_time(session_id: str, db_path: Optional[str] = None) -> Optional[float]:
    """Average time spent per answer"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT AVG(duration_seconds) as avg_time FROM interview_answers
               WHERE session_id = ?""",
            (session_id,),
        )
        row = cursor.fetchone()
        return row["avg_time"] if row and row["avg_time"] else None


def get_probe_effectiveness(session_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """How many probes led to better answers"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Get decisions where action=PROBE_DEEPER
        cursor.execute(
            """SELECT id, answer_id FROM interview_decisions
               WHERE session_id = ? AND action = 'PROBE_DEEPER'""",
            (session_id,),
        )
        probe_decisions = [dict(row) for row in cursor.fetchall()]
        
        if not probe_decisions:
            return {"probes": 0, "led_to_follow_ups": 0}
        
        # Check which had follow-ups
        with_followups = 0
        for decision in probe_decisions:
            cursor.execute(
                """SELECT follow_ups_asked FROM interview_answers WHERE id = ?""",
                (decision["answer_id"],),
            )
            row = cursor.fetchone()
            if row and row["follow_ups_asked"] > 0:
                with_followups += 1
        
        return {
            "probes": len(probe_decisions),
            "led_to_follow_ups": with_followups,
            "followup_rate": with_followups / len(probe_decisions) * 100 if probe_decisions else 0,
        }


def get_session_fatigue_signals(session_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Detect fatigue patterns"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT quality, duration_seconds FROM interview_answers
               WHERE session_id = ? ORDER BY created_at""",
            (session_id,),
        )
        answers = [dict(row) for row in cursor.fetchall()]
        
        if len(answers) < 3:
            return {"detected": False, "reason": "too_few_answers"}
        
        # Check last 3 answers for poor quality
        last_3 = answers[-3:]
        poor_count = len([a for a in last_3 if a["quality"] in ["poor", "off_topic"]])
        
        # Check for declining time (talking less)
        if len(answers) >= 5:
            early_avg = sum(a["duration_seconds"] or 0 for a in answers[:3]) / 3
            late_avg = sum(a["duration_seconds"] or 0 for a in answers[-3:]) / 3
            time_decline = (early_avg - late_avg) / early_avg if early_avg > 0 else 0
        else:
            time_decline = 0
        
        detected = poor_count >= 2 or time_decline > 0.3
        
        return {
            "detected": detected,
            "poor_answers_in_last_3": poor_count,
            "time_decline_pct": round(time_decline * 100, 1),
            "signals": [
                "declining_quality" if poor_count >= 2 else None,
                "talking_less" if time_decline > 0.3 else None,
            ],
        }


def get_run_interview_stats(run_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate stats across all sessions for a run"""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Get all sessions
        cursor.execute(
            """SELECT COUNT(*) as total_sessions, 
                      AVG(avg_answer_quality) as avg_quality,
                      AVG(total_duration_seconds) as avg_duration,
                      SUM(CASE WHEN fatigue_detected = 1 THEN 1 ELSE 0 END) as fatigue_count
               FROM interview_sessions
               WHERE run_id = ?""",
            (run_id,),
        )
        row = cursor.fetchone()
        
        return {
            "total_sessions": row["total_sessions"] or 0,
            "avg_quality": round(row["avg_quality"], 2) if row["avg_quality"] else 0,
            "avg_duration_minutes": round((row["avg_duration"] or 0) / 60, 1),
            "sessions_with_fatigue": row["fatigue_count"] or 0,
        }
