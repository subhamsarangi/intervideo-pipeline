"""Tests for LLM Judge evaluation node and persistence"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.database.db import init_db, save_question, get_question, list_questions_by_run, update_question_status
from src.models.judge_evaluation import JudgeScores, QuestionScore
from src.pipeline.nodes.judge_evaluation import judge_evaluation_node


@pytest.fixture
def db_path(tmp_path):
    """Create temporary database for testing"""
    db = tmp_path / "test_judge.db"
    init_db(str(db))
    return str(db)


def test_save_question(db_path):
    """Test saving a question with judge scores"""
    run_id = 1
    question_data = {
        "question_text": "What is your experience with distributed systems?",
        "level": "senior",
        "skill_required": "Distributed Systems",
        "question_type": "technical",
        "key_points": ["Consistency models", "Fault tolerance", "Scalability"],
        "follow_ups": ["How do you handle network partitions?"],
        "source_type": "project",
        "source_id": "proj_123",
        "difficulty": 4,
        "gen_model": "gpt-5.4-nano",
    }
    
    judge_scores = {
        "clarity": 0.9,
        "relevance": 0.85,
        "difficulty_match": 0.88,
        "groundedness": 0.92,
        "redundancy": 0.95,
        "overall": 0.90,
    }
    
    question_id = save_question(
        run_id=run_id,
        question_data=question_data,
        judge_scores=judge_scores,
        judge_feedback="Excellent question, well-grounded in CV",
        judge_pass=True,
        status="candidate",
        db_path=db_path,
    )
    
    assert question_id > 0
    
    # Retrieve and verify
    q = get_question(question_id, db_path=db_path)
    assert q is not None
    assert q["question_text"] == question_data["question_text"]
    assert q["level"] == "senior"
    assert q["judge_overall_score"] == 0.90
    assert q["judge_pass"] == True
    assert q["status"] == "candidate"


def test_save_rejected_question(db_path):
    """Test saving a rejected question"""
    question_data = {
        "question_text": "What is your favorite color?",
        "level": "junior",
        "skill_required": "Communication",
        "question_type": "behavioral",
        "key_points": ["Response", "Personality"],
        "follow_ups": [],
        "source_type": "skill",
        "source_id": "skill_comm",
    }
    
    judge_scores = {
        "clarity": 0.7,
        "relevance": 0.2,  # Very low relevance
        "difficulty_match": 0.6,
        "groundedness": 0.3,
        "redundancy": 0.4,
        "overall": 0.35,  # Below 0.65 threshold
    }
    
    question_id = save_question(
        run_id=1,
        question_data=question_data,
        judge_scores=judge_scores,
        judge_feedback="Lacks relevance to technical role",
        judge_pass=False,
        status="rejected_auto",
        db_path=db_path,
    )
    
    q = get_question(question_id, db_path=db_path)
    assert q["status"] == "rejected_auto"
    assert q["judge_overall_score"] == 0.35
    assert q["judge_pass"] == False


def test_list_questions_by_run(db_path):
    """Test retrieving all questions for a run"""
    run_id = 1
    
    # Save 3 passed questions
    for i in range(3):
        save_question(
            run_id=run_id,
            question_data={
                "question_text": f"Question {i}",
                "level": "mid",
                "skill_required": "Python",
                "question_type": "technical",
                "key_points": ["A", "B"],
                "follow_ups": [],
            },
            judge_scores={
                "clarity": 0.85,
                "relevance": 0.90,
                "difficulty_match": 0.80,
                "groundedness": 0.88,
                "redundancy": 0.85,
                "overall": 0.86,
            },
            judge_pass=True,
            status="candidate",
            db_path=db_path,
        )
    
    # Save 2 rejected questions
    for i in range(2):
        save_question(
            run_id=run_id,
            question_data={
                "question_text": f"Bad question {i}",
                "level": "junior",
                "skill_required": "Generic",
                "question_type": "behavioral",
                "key_points": ["A"],
                "follow_ups": [],
            },
            judge_scores={
                "clarity": 0.6,
                "relevance": 0.3,
                "difficulty_match": 0.5,
                "groundedness": 0.4,
                "redundancy": 0.4,
                "overall": 0.42,
            },
            judge_pass=False,
            status="rejected_auto",
            db_path=db_path,
        )
    
    # List all
    all_q = list_questions_by_run(run_id, db_path=db_path)
    assert len(all_q) == 5
    
    # List only passed
    passed = list_questions_by_run(run_id, status="candidate", db_path=db_path)
    assert len(passed) == 3
    assert all(q["status"] == "candidate" for q in passed)
    
    # List only rejected
    rejected = list_questions_by_run(run_id, status="rejected_auto", db_path=db_path)
    assert len(rejected) == 2
    assert all(q["status"] == "rejected_auto" for q in rejected)


def test_update_question_status(db_path):
    """Test updating question status (e.g., candidate → verified)"""
    question_id = save_question(
        run_id=1,
        question_data={
            "question_text": "Test",
            "level": "mid",
            "skill_required": "Test",
            "question_type": "technical",
            "key_points": [],
            "follow_ups": [],
        },
        judge_pass=True,
        status="candidate",
        db_path=db_path,
    )
    
    # Update to verified
    update_question_status(
        question_id,
        status="verified",
        review_note="Approved by human reviewer",
        db_path=db_path,
    )
    
    q = get_question(question_id, db_path=db_path)
    assert q["status"] == "verified"
    assert q["review_note"] == "Approved by human reviewer"


def test_judge_scores_pydantic():
    """Test JudgeScores validation"""
    scores = JudgeScores(
        clarity=0.9,
        relevance=0.85,
        difficulty_match=0.88,
        groundedness=0.92,
        redundancy=0.95,
    )
    
    assert scores.clarity == 0.9
    assert scores.redundancy == 0.95
    
    # Test bounds
    with pytest.raises(ValueError):
        JudgeScores(clarity=1.5, relevance=0.8, difficulty_match=0.8, groundedness=0.8, redundancy=0.8)


def test_question_score_pydantic():
    """Test QuestionScore validation"""
    score = QuestionScore(
        question_text="What is your experience?",
        scores=JudgeScores(
            clarity=0.85,
            relevance=0.90,
            difficulty_match=0.80,
            groundedness=0.88,
            redundancy=0.85,
        ),
        overall_score=0.86,
        pass_threshold=True,
        feedback="Good question",
        rejection_reason=None,
    )
    
    assert score.pass_threshold == True
    assert score.overall_score == 0.86
    
    # Test below threshold
    score_rejected = QuestionScore(
        question_text="Bad question",
        scores=JudgeScores(
            clarity=0.5,
            relevance=0.3,
            difficulty_match=0.4,
            groundedness=0.4,
            redundancy=0.5,
        ),
        overall_score=0.42,
        pass_threshold=False,
        feedback="Too generic",
        rejection_reason="Low relevance",
    )
    
    assert score_rejected.pass_threshold == False


@pytest.mark.asyncio
async def test_judge_evaluation_node_mock(db_path):
    """Test judge evaluation node with mocked LLM"""
    state = {
        "run_id": 1,
        "merged_questions": [
            {
                "question_text": "Describe your experience with REST APIs",
                "level": "mid",
                "skill_required": "Backend Development",
                "question_type": "technical",
                "key_points": ["HTTP methods", "Status codes", "RESTful design"],
                "follow_ups": ["How do you handle errors?"],
                "source_type": "project",
                "source_id": "proj_api",
            },
            {
                "question_text": "What is your favorite book?",
                "level": "junior",
                "skill_required": "General",
                "question_type": "behavioral",
                "key_points": ["Reading habits"],
                "follow_ups": [],
                "source_type": "skill",
                "source_id": "skill_comm",
            },
        ],
        "jd": {
            "job_title": "Senior Backend Engineer",
            "summary": "Build scalable backend systems",
            "requirements": [
                {"text": "REST API design", "requirement_type": "must_have"},
                {"text": "Python", "requirement_type": "must_have"},
            ],
        },
        "cv": {"name": "John Doe", "total_years": 5},
        "db_path": db_path,
    }
    
    # Mock the async OpenAI client
    mock_response = MagicMock()
    mock_response.question_scores = [
        QuestionScore(
            question_text="Describe your experience with REST APIs",
            scores=JudgeScores(
                clarity=0.9, relevance=0.95, difficulty_match=0.88,
                groundedness=0.90, redundancy=0.85
            ),
            overall_score=0.90,
            pass_threshold=True,
            feedback="Excellent technical question",
            rejection_reason=None,
        ),
        QuestionScore(
            question_text="What is your favorite book?",
            scores=JudgeScores(
                clarity=0.7, relevance=0.2, difficulty_match=0.5,
                groundedness=0.3, redundancy=0.6
            ),
            overall_score=0.35,
            pass_threshold=False,
            feedback="Not relevant to role",
            rejection_reason="Low relevance",
        ),
    ]
    mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    
    with patch("src.pipeline.nodes.judge_evaluation.get_async_openai_client") as mock_client:
        mock_instructor_client = AsyncMock()
        mock_instructor_client.chat.completions.create.return_value = mock_response
        
        with patch("src.pipeline.nodes.judge_evaluation.instructor.from_openai", return_value=mock_instructor_client):
            result = await judge_evaluation_node(state)
    
    assert result["status"] == "running"
    assert result["passed_count"] == 1
    assert result["rejected_count"] == 1
    assert len(result["evaluated_questions"]) == 2
    assert result["evaluated_questions"][0]["status"] == "candidate"
    assert result["evaluated_questions"][1]["status"] == "rejected_auto"


def test_question_uniqueness_by_hash(db_path):
    """Test that question_hash prevents duplicates"""
    question_data = {
        "question_text": "What is your experience with Python?",
        "level": "mid",
        "skill_required": "Python",
        "question_type": "technical",
        "key_points": ["A"],
        "follow_ups": [],
    }
    
    # Insert first time
    id1 = save_question(
        run_id=1,
        question_data=question_data,
        judge_pass=True,
        status="candidate",
        db_path=db_path,
    )
    
    # Try to insert same question again (should fail due to UNIQUE constraint)
    with pytest.raises(Exception):  # SQLite IntegrityError
        save_question(
            run_id=1,
            question_data=question_data,
            judge_pass=True,
            status="candidate",
            db_path=db_path,
        )
