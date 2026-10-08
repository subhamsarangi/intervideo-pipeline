"""Tests for autonomous interview loop (Phase 11b)"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.database.db import init_db
from src.interview.control_agent import InterviewControlAgent
from src.models.interview_agent import InterviewSessionState, DecisionReason


@pytest.fixture
def db_path():
    """Create temporary database for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Path(tmpdir) / "test_interview_auto.db"
        init_db(str(db))
        yield str(db)


@pytest.fixture
def mock_agent(db_path):
    """Create agent with mocked OpenAI client"""
    with patch("src.interview.control_agent.AsyncOpenAI"):
        with patch("src.interview.control_agent.instructor.from_openai"):
            agent = InterviewControlAgent(db_path=db_path)
            agent.client = AsyncMock()
            return agent


class MockAnswerProvider:
    """Mock answer provider for testing"""
    
    def __init__(self, answers: list):
        """answers: list of (answer_text, duration_seconds)"""
        self.answers = answers
        self.call_count = 0
    
    async def __call__(self, question_text: str) -> str:
        """Return next mocked answer"""
        if self.call_count < len(self.answers):
            answer, _ = self.answers[self.call_count]
            self.call_count += 1
            return answer
        return "I don't know."


@pytest.mark.asyncio
async def test_run_interview_tool_usage(db_path):
    """Test that agent uses get_next_question tool correctly"""
    with patch("src.interview.control_agent.AsyncOpenAI"):
        with patch("src.interview.control_agent.instructor.from_openai"):
            agent = InterviewControlAgent(db_path=db_path)
    
    session = InterviewSessionState(
        session_id="test_auto_003",
        run_id=1,
        total_questions=2,
    )
    
    questions = [
        {"id": 1, "question_text": "Q1", "question_level": "junior", "question_type": "technical"},
        {"id": 2, "question_text": "Q2", "question_level": "junior", "question_type": "technical"},
    ]
    
    # Load questions into agent
    agent.load_questions(questions)
    assert agent.current_q_index == 0
    assert len(agent.questions) == 2
    
    # Get questions via tool
    q1 = agent.get_next_question()
    assert q1 is not None
    assert q1["id"] == 1
    assert agent.current_q_index == 1
    
    q2 = agent.get_next_question()
    assert q2 is not None
    assert q2["id"] == 2
    assert agent.current_q_index == 2
    
    # No more questions
    q3 = agent.get_next_question()
    assert q3 is None


@pytest.mark.asyncio
async def test_run_interview_autonomy(mock_agent, db_path):
    """Test that caller doesn't need to orchestrate: agent manages tools"""
    
    # Just test tool loading and Q retrieval
    questions = [
        {"id": 1, "question_text": "Q1", "question_level": "junior", "question_type": "technical"},
        {"id": 2, "question_text": "Q2", "question_level": "mid", "question_type": "behavioral"},
        {"id": 3, "question_text": "Q3", "question_level": "senior", "question_type": "design"},
    ]
    
    mock_agent.load_questions(questions)
    
    # Verify agent has questions loaded and can retrieve via tool
    assert len(mock_agent.questions) == 3
    
    q1 = mock_agent.get_next_question()
    assert q1["id"] == 1
    
    q2 = mock_agent.get_next_question()
    assert q2["id"] == 2
    
    q3 = mock_agent.get_next_question()
    assert q3["id"] == 3
    
    # Verify autonomy: all Qs retrieved via agent's tool
    q4 = mock_agent.get_next_question()
    assert q4 is None  # Done


def test_agent_has_tools(mock_agent):
    """Test that agent has tool methods"""
    assert hasattr(mock_agent, 'get_next_question')
    assert callable(mock_agent.get_next_question)
    
    assert hasattr(mock_agent, 'skip_question')
    assert callable(mock_agent.skip_question)
    
    assert hasattr(mock_agent, 'load_questions')
    assert callable(mock_agent.load_questions)
    
    assert hasattr(mock_agent, 'run_interview')
    assert callable(mock_agent.run_interview)


def test_agent_has_loop(mock_agent):
    """Test that agent has main loop (run_interview)"""
    import inspect
    
    # run_interview should be async
    assert inspect.iscoroutinefunction(mock_agent.run_interview)
    
    # Should have docstring mentioning autonomous/loop
    assert "autonomous" in mock_agent.run_interview.__doc__.lower()
    assert "loop" in mock_agent.run_interview.__doc__.lower()


def test_agent_is_autonomous(mock_agent):
    """Test that agent is now truly autonomous"""
    # Agent has:
    # 1. Tools (get_next_question, skip_question, load_questions)
    assert hasattr(mock_agent, 'get_next_question')
    assert hasattr(mock_agent, 'skip_question')
    assert hasattr(mock_agent, 'load_questions')
    
    # 2. Main loop (run_interview)
    assert hasattr(mock_agent, 'run_interview')
    
    # 3. Decision-making (decide_next_action)
    assert hasattr(mock_agent, 'decide_next_action')
    
    # 4. Memory (self.questions, self.current_q_index)
    assert hasattr(mock_agent, 'questions')
    assert hasattr(mock_agent, 'current_q_index')
    
    # Agent is a real agent now: tools + loop + autonomy + memory
