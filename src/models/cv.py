"""Pydantic models for CV data"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any


class EvidenceSpan(BaseModel):
    """Evidence location in source text"""

    start: int = Field(..., description="Start character position")
    end: int = Field(..., description="End character position")


class CVProject(BaseModel):
    """Project from CV"""

    title: str = Field(..., description="Project title")
    description: str = Field(..., description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Tech stack used")
    duration: Optional[str] = Field(None, description="Project duration/timeframe")
    evidence_span: Optional[EvidenceSpan] = Field(
        None, description="Location in CV text"
    )


class CVSkill(BaseModel):
    """Skill from CV"""

    skill_name: str = Field(..., description="Name of skill")
    proficiency_level: Optional[str] = Field(
        None, description="e.g., 'expert', 'intermediate'"
    )
    years_of_experience: Optional[float] = Field(
        None, description="Years of experience"
    )
    evidence_span: Optional[EvidenceSpan] = Field(
        None, description="Location in CV text"
    )


class ParsedCV(BaseModel):
    """Structured CV data after parsing"""

    name: Optional[str] = Field(None, description="Candidate name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    summary: Optional[str] = Field(None, description="Professional summary")
    projects: List[CVProject] = Field(
        default_factory=list, description="List of projects"
    )
    skills: List[CVSkill] = Field(default_factory=list, description="List of skills")
    experience_years: Optional[float] = Field(
        None, description="Total years of experience"
    )


class CVDocument(BaseModel):
    """CV document in database"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    raw_text: str
    parsed_data: Optional[ParsedCV] = None
    status: str = Field(..., description="pending|processing|completed|failed")
    error_message: Optional[str] = None
    created_at: str
    updated_at: str
