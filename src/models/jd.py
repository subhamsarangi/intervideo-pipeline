"""Pydantic models for Job Description data"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List


class JDRequirement(BaseModel):
    """Single requirement from JD"""

    text: str = Field(..., description="Requirement text")
    requirement_type: str = Field(..., description="'must_have' or 'nice_to_have'")
    category: Optional[str] = Field(
        None, description="e.g., 'technical', 'soft_skills', 'experience'"
    )

    @field_validator("requirement_type")
    @classmethod
    def validate_requirement_type(cls, v):
        if v not in ("must_have", "nice_to_have"):
            raise ValueError("requirement_type must be 'must_have' or 'nice_to_have'")
        return v


class ParsedJD(BaseModel):
    """Structured JD data after parsing"""

    job_title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    summary: Optional[str] = Field(None, description="Job summary/overview")
    requirements: List[JDRequirement] = Field(
        default_factory=list, description="List of requirements"
    )


class JDDocument(BaseModel):
    """JD document in database"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    raw_text: str
    parsed_data: Optional[ParsedJD] = None
    source_type: str = Field(..., description="'pasted'|'url'|'file'")
    source_url: Optional[str] = None
    status: str = Field(..., description="pending|processing|completed|failed")
    error_message: Optional[str] = None
    created_at: str
    updated_at: str
