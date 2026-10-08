"""Structured question schema with proficiency levels and follow-ups"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ProficiencyLevel(str, Enum):
    """Proficiency level for questions"""
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"


class QuestionType(str, Enum):
    """Type of interview question"""
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    DESIGN = "design"
    SITUATIONAL = "situational"
    FOLLOW_UP = "follow_up"


class StructuredQuestion(BaseModel):
    """Structured question schema with metadata for interview"""
    
    question_text: str = Field(
        ...,
        description="The actual interview question to ask"
    )
    
    level: ProficiencyLevel = Field(
        ...,
        description="Target proficiency level (junior/mid/senior)"
    )
    
    skill_required: str = Field(
        ...,
        description="Primary skill being assessed (e.g., 'Python', 'System Design', 'Communication')"
    )
    
    question_type: QuestionType = Field(
        ...,
        description="Type of question (behavioral/technical/design/situational)"
    )
    
    key_points: List[str] = Field(
        ...,
        description="3-5 key points the ideal answer should cover",
        min_length=3,
        max_length=5
    )
    
    follow_ups: List[str] = Field(
        default_factory=list,
        description="2-3 follow-up questions to probe deeper if needed",
        max_length=3
    )
    
    relevance_note: Optional[str] = Field(
        default=None,
        description="Why this question is relevant to the role"
    )
    
    source_type: Optional[str] = Field(
        default=None,
        description="Source: 'project' or 'skill'"
    )
    
    source_id: Optional[str] = Field(
        default=None,
        description="ID of source project or skill"
    )


class StructuredQuestionSet(BaseModel):
    """Set of structured questions from a single source (project or skill)"""
    
    questions: List[StructuredQuestion] = Field(
        ...,
        description="List of structured questions"
    )
    
    source_name: str = Field(
        ...,
        description="Name of project or skill generating these questions"
    )
    
    source_type: str = Field(
        ...,
        description="'project' or 'skill'"
    )
    
    level_distribution: dict = Field(
        default_factory=dict,
        description="Count of questions per level: {'junior': N, 'mid': N, 'senior': N}"
    )


# Proficiency level examples (for prompt engineering)
LEVEL_DEFINITIONS = {
    "junior": {
        "description": "Entry-level / 0-2 years experience",
        "example_question": "Can you explain what this technology does and when you'd use it?",
        "focus": "Fundamentals, terminology, basic problem-solving"
    },
    "mid": {
        "description": "Intermediate / 2-5 years experience",
        "example_question": "Tell me about a project where you used this. What challenges did you face and how did you solve them?",
        "focus": "Real-world application, trade-offs, debugging, architecture decisions"
    },
    "senior": {
        "description": "Expert / 5+ years experience",
        "example_question": "How would you design a system using this technology at scale? What edge cases would you consider?",
        "focus": "System design, scalability, mentoring, strategic decisions, optimization"
    }
}
