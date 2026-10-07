"""Pydantic models for pipeline state and results"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from src.models.cv import ParsedCV
from src.models.jd import ParsedJD


class TriageResult(BaseModel):
    """Result from relevance triage step"""

    ranked_projects: List[Dict[str, Any]] = Field(
        default_factory=list, description="Projects ranked by relevance with scores"
    )
    ranked_skills: List[Dict[str, Any]] = Field(
        default_factory=list, description="Skills ranked by relevance with scores"
    )
    flagged_for_enrichment: List[str] = Field(
        default_factory=list,
        description="Project/skill IDs flagged as recent/unfamiliar",
    )


class EnrichmentResult(BaseModel):
    """Result from web enrichment step"""

    enrichment_context: Dict[str, str] = Field(
        default_factory=dict, description="Enrichment data keyed by project/skill ID"
    )
    sources_used: List[str] = Field(
        default_factory=list, description="URLs/sources used"
    )


class GeneratedQuestion(BaseModel):
    """Single generated question"""

    question_text: str = Field(..., description="The question")
    source_type: str = Field(..., description="'project' or 'skill'")
    source_id: Optional[str] = Field(None, description="Project/skill ID")
    evidence_snippet: Optional[str] = Field(None, description="Evidence from CV")
    evidence_span: Optional[Dict[str, int]] = Field(
        None, description="{'start': int, 'end': int}"
    )


class MergedQuestionSet(BaseModel):
    """Merged and deduplicated questions"""

    total_questions: int = Field(..., description="Total question count")
    questions: List[GeneratedQuestion] = Field(default_factory=list)
    coverage: Dict[str, bool] = Field(
        default_factory=dict, description="Which JD requirements are covered"
    )


class JudgeScore(BaseModel):
    """Judge evaluation for a single question"""

    question_id: int = Field(..., description="ID of question being scored")
    relevance_score: float = Field(..., ge=0, le=10, description="0-10 relevance")
    groundedness_score: float = Field(..., ge=0, le=10, description="0-10 groundedness")
    redundancy_score: float = Field(..., ge=0, le=10, description="0-10 redundancy")
    reasoning: Optional[str] = Field(None, description="Explanation of scores")


class FitnessScore(BaseModel):
    """Fitness score for a single requirement"""

    requirement_id: int
    requirement_text: str = Field(..., description="JD requirement")
    requirement_type: str = Field(..., description="'must_have' or 'nice_to_have'")
    match_score: float = Field(..., ge=0, le=10, description="0-10 match strength")
    evidence: Optional[str] = Field(None, description="Evidence from CV")
    weighted_score: Optional[float] = Field(None, description="Score × weight")


class PipelineState(BaseModel):
    """Complete pipeline execution state"""

    run_id: int
    cv_id: int
    jd_id: int

    # Input data
    cv: Optional[ParsedCV] = None
    jd: Optional[ParsedJD] = None
    wrapped_cv: Optional[str] = None
    wrapped_jd: Optional[str] = None
    db_path: Optional[str] = None

    # Step outputs
    triage_result: Optional[TriageResult] = None
    enrichment_result: Optional[EnrichmentResult] = None
    project_questions: List[GeneratedQuestion] = Field(default_factory=list)
    skill_questions: List[GeneratedQuestion] = Field(default_factory=list)
    merged_questions: Optional[MergedQuestionSet] = None
    judge_scores: List[JudgeScore] = Field(default_factory=list)
    fitness_scores: List[FitnessScore] = Field(default_factory=list)

    # Metadata
    prompt_version: Optional[str] = None
    status: str = Field(
        default="pending", description="pending|running|completed|failed"
    )
    error_message: Optional[str] = None


class PipelineRunResponse(BaseModel):
    """API response for pipeline run"""

    run_id: int
    status: str
    cv_id: int
    jd_id: int
    created_at: str
    updated_at: str
    progress_percent: int = Field(default=0)
