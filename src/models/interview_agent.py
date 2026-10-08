"""
Interview Control Agent Models

Manages interview flow, question progression, and exit conditions.
Tracks answered questions state and makes decisions on interview pacing.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class AnswerQuality(str, Enum):
    """Quality rating of candidate answer"""
    EXCELLENT = "excellent"  # Directly answers, detailed, relevant
    GOOD = "good"  # Answers with minor gaps
    FAIR = "fair"  # Partially answers, needs probing
    POOR = "poor"  # Doesn't address question
    OFF_TOPIC = "off_topic"  # Not related to question


class DecisionReason(str, Enum):
    """Why agent made a decision"""
    CONTINUE_NORMAL = "continue_normal"  # Good answer, move to next
    PROBE_DEEPER = "probe_deeper"  # Incomplete answer, ask follow-up
    SKIP_QUESTION = "skip_question"  # Candidate struggled too much
    TIME_LIMIT = "time_limit"  # Running out of time
    FATIGUE = "fatigue"  # Candidate appears tired/disengaged
    ALL_ASKED = "all_asked"  # All questions completed
    CANDIDATE_EXIT = "candidate_exit"  # User requested to stop


class AnswerRecord(BaseModel):
    """Record of a single question-answer pair"""
    question_id: int
    question_text: str
    question_level: str  # junior/mid/senior
    question_type: str  # technical/behavioral/design
    
    asked_at: datetime
    answer_text: Optional[str] = None
    answered_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None  # Time spent on this question
    
    quality: Optional[AnswerQuality] = None
    feedback: Optional[str] = None  # Why agent rated it this way
    
    follow_ups_asked: int = 0  # How many follow-ups for this question
    follow_up_texts: List[str] = Field(default_factory=list)


class InterviewSessionState(BaseModel):
    """Tracks entire interview session state"""
    session_id: str
    run_id: int
    
    # Question tracking
    total_questions: int  # How many questions planned
    questions_asked: int = 0
    questions_skipped: int = 0
    questions_remaining: int = Field(init=False)
    
    # Answers tracking
    answers: List[AnswerRecord] = Field(default_factory=list)
    avg_answer_quality: float = 0.0  # Running average
    
    # Timing
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[int] = None
    planned_duration_seconds: int = 1800  # 30 minutes default
    
    # Exit condition
    is_active: bool = True
    exit_reason: Optional[DecisionReason] = None
    exit_message: Optional[str] = None
    
    @property
    def questions_remaining(self) -> int:
        """Calculate remaining questions"""
        return self.total_questions - self.questions_asked


class AgentDecision(BaseModel):
    """Decision made by interview control agent"""
    action: DecisionReason
    
    # What to do
    next_question_id: Optional[int] = None  # If continue or probe
    follow_up: Optional[str] = None  # If probe deeper
    
    # Why
    reasoning: str
    confidence: float = Field(ge=0, le=1)  # 0-1 confidence in decision
    
    # Metadata
    made_at: datetime = Field(default_factory=datetime.now)
    based_on_answer: Optional[str] = None  # Last answer text (for context)


class InterviewMetrics(BaseModel):
    """Summary metrics for interview"""
    total_questions: int
    questions_asked: int
    questions_skipped: int
    
    quality_distribution: Dict[AnswerQuality, int]  # Count per quality level
    avg_answer_quality: float
    best_answer: Optional[AnswerQuality] = None
    worst_answer: Optional[AnswerQuality] = None
    
    total_duration_minutes: float
    avg_question_duration_seconds: float
    
    fatigue_detected: bool
    reason_for_exit: DecisionReason
    
    questions_by_level: Dict[str, int]  # Count per level (junior/mid/senior)
    performance_by_level: Dict[str, float]  # Avg quality per level
