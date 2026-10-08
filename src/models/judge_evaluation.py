"""LLM Judge evaluation schema for question scoring"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, computed_field
from enum import Enum


# Single source of truth for scoring. `redundancy` is reported but not weighted.
WEIGHTS: Dict[str, float] = {
    "clarity": 0.30,
    "relevance": 0.35,
    "difficulty_match": 0.20,
    "groundedness": 0.15,
}
PASS_THRESHOLD = 0.65


class QuestionStatus(str, Enum):
    """Question status in review pipeline"""
    REJECTED_AUTO = "rejected_auto"
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    REJECTED_HUMAN = "rejected_human"


class JudgeScores(BaseModel):
    """Normalized judge scores (0-1 scale)"""
    clarity: float = Field(..., ge=0, le=1, description="How clear is the question (0-1)")
    relevance: float = Field(..., ge=0, le=1, description="How relevant to role (0-1)")
    difficulty_match: float = Field(..., ge=0, le=1, description="Matches the question's level tag (0-1)")
    groundedness: float = Field(..., ge=0, le=1, description="Grounded in CV/JD (0-1)")
    redundancy: float = Field(..., ge=0, le=1, description="Uniqueness vs other questions in batch (0-1)")

    def weighted_overall(self) -> float:
        """Weighted overall score (redundancy excluded), rounded to 2 decimals."""
        return round(sum(getattr(self, k) * w for k, w in WEIGHTS.items()), 2)


class QuestionScore(BaseModel):
    """Score for a single question.

    Field order matters: the LLM generates fields in this order, so it writes
    feedback (reasoning) before committing to numeric scores.

    overall_score and pass_threshold are computed in code, NOT by the LLM.
    """

    question_index: int = Field(
        ..., ge=1, description="1-based number of the question as shown in the prompt"
    )

    question_text: str = Field(..., description="The question being scored (verbatim)")

    feedback: str = Field(..., description="1-2 sentences of reasoning, naming the weakest dimension")

    scores: JudgeScores = Field(..., description="Normalized scores (0-1)")

    rejection_reason: Optional[str] = Field(
        default=None,
        description="If not passing, short label (e.g., 'Low relevance', 'Ambiguous wording', 'Level mismatch')",
    )

    @computed_field  # type: ignore[misc]
    @property
    def overall_score(self) -> float:
        return self.scores.weighted_overall()

    @computed_field  # type: ignore[misc]
    @property
    def pass_threshold(self) -> bool:
        return self.overall_score >= PASS_THRESHOLD


class QuestionSetEvaluation(BaseModel):
    """Evaluation results for a batch of questions"""

    question_scores: List[QuestionScore] = Field(..., description="Scores for each question")

    average_clarity: float
    average_relevance: float
    average_difficulty_match: float
    average_groundedness: float
    average_overall: float

    passed_count: int = Field(..., description=f"Questions passing threshold (>= {PASS_THRESHOLD})")
    rejected_count: int = Field(..., description="Questions below threshold")
    total_count: int

    pass_rate: float = Field(..., description="Fraction passing (0-1)")