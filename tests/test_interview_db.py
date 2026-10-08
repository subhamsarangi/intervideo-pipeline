"""Tests for interview database operations (Phase 11b)"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from src.database.db import init_db
from src.database.interview_db import (
    create_interview_session,
    get_interview_session,
    update_interview_session,
    save_interview_answer,
    get_interview_answer,
    list_session_answers,
    save_interview_decision,
    list_session_decisions,
    get_session_metrics,
    get_session_quality_distribution,
    get_session_by_level,
    get_session_decision_breakdown,
    get_average_answer_time,
    get_session_fatigue_signals,
    get_run_interview_stats,
    list_run_sessions,
)


@pytest.fixture
def db_path():
    """Create temporary database for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Path(tmpdir) / "test_interview.db"
        init_db(str(db))
        yield str(db)


class TestSessionManagement:
    """Test interview session CRUD operations"""
    
    def test_create_interview_session(self, db_path):
        """Create and retrieve session"""
        session_id = "test_session_001"
        db_id = create_interview_session(
            session_id=session_id,
            run_id=1,
            total_questions=5,
            planned_duration_seconds=1800,
            db_path=db_path,
        )
        
        assert db_id > 0
        
        # Retrieve
        session = get_interview_session(session_id, db_path=db_path)
        assert session is not None
        assert session["session_id"] == session_id
        assert session["run_id"] == 1
        assert session["total_questions"] == 5
        assert session["is_active"] == 1
    
    def test_update_interview_session(self, db_path):
        """Update session with end time and metrics"""
        session_id = "test_session_002"
        create_interview_session(
            session_id=session_id,
            run_id=1,
            total_questions=3,
            planned_duration_seconds=1800,
            db_path=db_path,
        )
        
        # Update
        end_time = datetime.now()
        update_interview_session(
            session_id=session_id,
            end_time=end_time,
            total_duration_seconds=900,
            avg_answer_quality=0.85,
            fatigue_detected=False,
            exit_reason="all_asked",
            is_active=False,
            db_path=db_path,
        )
        
        # Verify
        session = get_interview_session(session_id, db_path=db_path)
        assert session["total_duration_seconds"] == 900
        assert session["avg_answer_quality"] == 0.85
        assert session["fatigue_detected"] == 0
        assert session["exit_reason"] == "all_asked"
        assert session["is_active"] == 0
    
    def test_list_run_sessions(self, db_path):
        """List all sessions for a run"""
        run_id = 1
        
        # Create 3 sessions
        for i in range(3):
            create_interview_session(
                session_id=f"session_{i}",
                run_id=run_id,
                total_questions=5,
                planned_duration_seconds=1800,
                db_path=db_path,
            )
        
        # List
        sessions = list_run_sessions(run_id, db_path=db_path)
        assert len(sessions) == 3


class TestAnswerManagement:
    """Test answer recording and retrieval"""
    
    def test_save_interview_answer(self, db_path):
        """Save and retrieve answer"""
        session_id = "test_session_003"
        create_interview_session(
            session_id=session_id,
            run_id=1,
            total_questions=5,
            planned_duration_seconds=1800,
            db_path=db_path,
        )
        
        now = datetime.now()
        answer_id = save_interview_answer(
            session_id=session_id,
            question_id=1,
            question_text="Describe distributed systems",
            question_level="senior",
            question_type="design",
            asked_at=now,
            answer_text="Redis cluster with replication...",
            answered_at=now,
            duration_seconds=45,
            quality="excellent",
            feedback="Comprehensive answer",
            follow_ups_asked=0,
            db_path=db_path,
        )
        
        assert answer_id > 0
        
        # Retrieve
        answer = get_interview_answer(answer_id, db_path=db_path)
        assert answer is not None
        assert answer["quality"] == "excellent"
        assert answer["duration_seconds"] == 45
        assert answer["question_level"] == "senior"
    
    def test_list_session_answers(self, db_path):
        """List all answers for session"""
        session_id = "test_session_004"
        create_interview_session(
            session_id=session_id,
            run_id=1,
            total_questions=5,
            planned_duration_seconds=1800,
            db_path=db_path,
        )
        
        now = datetime.now()
        
        # Save 3 answers
        for i in range(3):
            save_interview_answer(
                session_id=session_id,
                question_id=i+1,
                question_text=f"Question {i+1}",
                question_level="mid",
                question_type="technical",
                asked_at=now,
                answer_text=f"Answer {i+1}",
                answered_at=now,
                duration_seconds=30,
                quality="good",
                feedback=f"Good answer {i+1}",
                db_path=db_path,
            )
        
        # List
        answers = list_session_answers(session_id, db_path=db_path)
        assert len(answers) == 3
        assert all(a["session_id"] == session_id for a in answers)


class TestDecisionManagement:
    """Test decision recording and retrieval"""
    
    def test_save_interview_decision(self, db_path):
        """Save and retrieve decision"""
        session_id = "test_session_005"
        create_interview_session(
            session_id=session_id,
            run_id=1,
            total_questions=5,
            planned_duration_seconds=1800,
            db_path=db_path,
        )
        
        now = datetime.now()
        answer_id = save_interview_answer(
            session_id=session_id,
            question_id=1,
            question_text="Test Q",
            question_level="junior",
            question_type="technical",
            asked_at=now,
            answer_text="Test A",
            answered_at=now,
            duration_seconds=15,
            quality="fair",
            feedback="Needs work",
            db_path=db_path,
        )
        
        decision_id = save_interview_decision(
            session_id=session_id,
            answer_id=answer_id,
            action="probe_deeper",
            follow_up="Can you explain more?",
            reasoning="Incomplete answer",
            confidence=0.8,
            db_path=db_path,
        )
        
        assert decision_id > 0
    
    def test_list_session_decisions(self, db_path):
        """List all decisions for session"""
        session_id = "test_session_006"
        create_interview_session(
            session_id=session_id,
            run_id=1,
            total_questions=5,
            planned_duration_seconds=1800,
            db_path=db_path,
        )
        
        now = datetime.now()
        
        # Save 2 answers + decisions
        for i in range(2):
            answer_id = save_interview_answer(
                session_id=session_id,
                question_id=i+1,
                question_text=f"Q{i+1}",
                question_level="mid",
                question_type="behavioral",
                asked_at=now,
                answer_text=f"A{i+1}",
                answered_at=now,
                duration_seconds=20,
                quality="good",
                feedback=f"Good",
                db_path=db_path,
            )
            
            save_interview_decision(
                session_id=session_id,
                answer_id=answer_id,
                action="continue_normal",
                reasoning="Good answer",
                confidence=0.9,
                db_path=db_path,
            )
        
        # List
        decisions = list_session_decisions(session_id, db_path=db_path)
        assert len(decisions) == 2


class TestAnalytics:
    """Test analytics and aggregation queries"""
    
    def test_get_session_quality_distribution(self, db_path):
        """Get quality counts"""
        session_id = "test_session_007"
        create_interview_session(
            session_id=session_id,
            run_id=1,
            total_questions=5,
            planned_duration_seconds=1800,
            db_path=db_path,
        )
        
        now = datetime.now()
        qualities = ["excellent", "good", "fair", "poor"]
        
        for quality in qualities:
            save_interview_answer(
                session_id=session_id,
                question_id=len(qualities),
                question_text="Q",
                question_level="mid",
                question_type="technical",
                asked_at=now,
                answer_text="A",
                answered_at=now,
                duration_seconds=30,
                quality=quality,
                feedback="Test",
                db_path=db_path,
            )
        
        dist = get_session_quality_distribution(session_id, db_path=db_path)
        assert len(dist) == 4
        assert all(dist[q] == 1 for q in qualities)
    
    def test_get_session_by_level(self, db_path):
        """Get performance by level"""
        session_id = "test_session_008"
        create_interview_session(
            session_id=session_id,
            run_id=1,
            total_questions=10,
            planned_duration_seconds=1800,
            db_path=db_path,
        )
        
        now = datetime.now()
        
        # Senior: 2 strong, 1 weak
        for i in range(3):
            quality = "excellent" if i < 2 else "poor"
            save_interview_answer(
                session_id=session_id,
                question_id=i+1,
                question_text="Q",
                question_level="senior",
                question_type="design",
                asked_at=now,
                answer_text="A",
                answered_at=now,
                duration_seconds=45,
                quality=quality,
                feedback="Test",
                db_path=db_path,
            )
        
        # Junior: 1 strong
        save_interview_answer(
            session_id=session_id,
            question_id=4,
            question_text="Q",
            question_level="junior",
            question_type="technical",
            asked_at=now,
            answer_text="A",
            answered_at=now,
            duration_seconds=20,
            quality="good",
            feedback="Test",
            db_path=db_path,
        )
        
        by_level = get_session_by_level(session_id, db_path=db_path)
        
        assert by_level["senior"]["questions"] == 3
        assert by_level["senior"]["strong_answers"] == 2
        assert by_level["junior"]["questions"] == 1
        assert by_level["junior"]["strong_answers"] == 1
    
    def test_get_session_decision_breakdown(self, db_path):
        """Get decision counts"""
        session_id = "test_session_009"
        create_interview_session(
            session_id=session_id,
            run_id=1,
            total_questions=5,
            planned_duration_seconds=1800,
            db_path=db_path,
        )
        
        now = datetime.now()
        
        # Create 2 continue, 1 probe, 1 skip
        actions = ["continue_normal", "continue_normal", "probe_deeper", "skip_question"]
        
        for i, action in enumerate(actions):
            answer_id = save_interview_answer(
                session_id=session_id,
                question_id=i+1,
                question_text="Q",
                question_level="mid",
                question_type="technical",
                asked_at=now,
                answer_text="A",
                answered_at=now,
                duration_seconds=30,
                quality="good",
                feedback="Test",
                db_path=db_path,
            )
            
            save_interview_decision(
                session_id=session_id,
                answer_id=answer_id,
                action=action,
                reasoning="Test",
                confidence=0.8,
                db_path=db_path,
            )
        
        breakdown = get_session_decision_breakdown(session_id, db_path=db_path)
        assert breakdown["continue_normal"] == 2
        assert breakdown["probe_deeper"] == 1
        assert breakdown["skip_question"] == 1
    
    def test_get_average_answer_time(self, db_path):
        """Get average time per answer"""
        session_id = "test_session_010"
        create_interview_session(
            session_id=session_id,
            run_id=1,
            total_questions=5,
            planned_duration_seconds=1800,
            db_path=db_path,
        )
        
        now = datetime.now()
        durations = [30, 45, 15]
        
        for i, duration in enumerate(durations):
            save_interview_answer(
                session_id=session_id,
                question_id=i+1,
                question_text="Q",
                question_level="mid",
                question_type="technical",
                asked_at=now,
                answer_text="A",
                answered_at=now,
                duration_seconds=duration,
                quality="good",
                feedback="Test",
                db_path=db_path,
            )
        
        avg = get_average_answer_time(session_id, db_path=db_path)
        assert avg == 30.0  # (30+45+15)/3
    
    def test_get_session_fatigue_signals(self, db_path):
        """Detect fatigue patterns"""
        session_id = "test_session_011"
        create_interview_session(
            session_id=session_id,
            run_id=1,
            total_questions=5,
            planned_duration_seconds=1800,
            db_path=db_path,
        )
        
        now = datetime.now()
        
        # Good answers early, poor last 3
        qualities = ["excellent", "good", "poor", "poor", "off_topic"]
        
        for i, quality in enumerate(qualities):
            save_interview_answer(
                session_id=session_id,
                question_id=i+1,
                question_text="Q",
                question_level="mid",
                question_type="technical",
                asked_at=now,
                answer_text="A",
                answered_at=now,
                duration_seconds=30,
                quality=quality,
                feedback="Test",
                db_path=db_path,
            )
        
        fatigue = get_session_fatigue_signals(session_id, db_path=db_path)
        assert fatigue["detected"] is True
        assert fatigue["poor_answers_in_last_3"] >= 2
    
    def test_get_run_interview_stats(self, db_path):
        """Get aggregate stats for run"""
        run_id = 1
        
        # Create 2 sessions
        for sess_idx in range(2):
            session_id = f"session_{sess_idx}"
            create_interview_session(
                session_id=session_id,
                run_id=run_id,
                total_questions=3,
                planned_duration_seconds=1800,
                db_path=db_path,
            )
            
            now = datetime.now()
            
            # Save 2 answers per session
            for i in range(2):
                save_interview_answer(
                    session_id=session_id,
                    question_id=i+1,
                    question_text="Q",
                    question_level="mid",
                    question_type="technical",
                    asked_at=now,
                    answer_text="A",
                    answered_at=now,
                    duration_seconds=30,
                    quality="good",
                    feedback="Test",
                    db_path=db_path,
                )
            
            # Update session with metrics
            update_interview_session(
                session_id=session_id,
                total_duration_seconds=600,
                avg_answer_quality=0.8,
                fatigue_detected=False,
                db_path=db_path,
            )
        
        stats = get_run_interview_stats(run_id, db_path=db_path)
        assert stats["total_sessions"] == 2
        assert stats["avg_quality"] == 0.8
        assert stats["avg_duration_minutes"] == 10.0  # 600s / 60
        assert stats["sessions_with_fatigue"] == 0
