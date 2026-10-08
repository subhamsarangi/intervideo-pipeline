"""LLM Judge evaluation schema for question scoring"""

from typing import List, Optional
from pydantic import BaseModel, Field
from src.models.structured_question import StructuredQuestion


class QuestionScore(BaseModel):
    """Score for a single question"""
    
    question_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the question (if available)"
    )
    
    question_text: str = Field(
        ...,
        description="The question being scored"
    )
    
    clarity_score: int = Field(
        ...,
        description="Clarity of the question (0-10)",
        ge=0,
        le=10
    )
    
    relevance_score: int = Field(
        ...,
        description="Relevance to the role/JD (0-10)",
        ge=0,
        le=10
    )
    
    difficulty_match_score: int = Field(
        ...,
        description="How well difficulty matches proficiency level (0-10)",
        ge=0,
        le=10
    )
    
    redundancy_score: int = Field(
        ...,
        description="Uniqueness (10 = not redundant, 0 = highly redundant) (0-10)",
        ge=0,
        le=10
    )
    
    overall_score: int = Field(
        ...,
        description="Weighted average: clarity 25% + relevance 30% + difficulty_match 25% + redundancy 20% (0-10)",
        ge=0,
        le=10
    )
    
    feedback: str = Field(
        ...,
        description="Brief feedback explaining the scores"
    )
    
    keep: bool = Field(
        default=True,
        description="Recommendation: keep if overall_score >= 6"
    )


class QuestionSetEvaluation(BaseModel):
    """Evaluation results for a set of questions"""
    
    question_scores: List[QuestionScore] = Field(
        ...,
        description="Scores for each question"
    )
    
    average_clarity: float = Field(
        ...,
        description="Average clarity across all questions"
    )
    
    average_relevance: float = Field(
        ...,
        description="Average relevance across all questions"
    )
    
    average_difficulty_match: float = Field(
        ...,
        description="Average difficulty match across all questions"
    )
    
    average_redundancy: float = Field(
        ...,
        description="Average redundancy (uniqueness) across all questions"
    )
    
    average_overall: float = Field(
        ...,
        description="Average overall score across all questions"
    )
    
    questions_to_keep: int = Field(
        ...,
        description="Count of questions with overall_score >= 6"
    )
    
    total_questions: int = Field(
        ...,
        description="Total questions evaluated"
    )
    
    summary: str = Field(
        ...,
        description="Summary of evaluation results"
    )
