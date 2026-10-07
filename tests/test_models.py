"""Tests for Pydantic models"""

import pytest
from src.models.cv import CVProject, CVSkill, ParsedCV, EvidenceSpan
from src.models.jd import JDRequirement, ParsedJD
from src.models.pipeline import (
    TriageResult,
    EnrichmentResult,
    GeneratedQuestion,
    MergedQuestionSet,
    JudgeScore,
    FitnessScore,
    PipelineState,
)


class TestCVModels:
    def test_cv_project(self):
        project = CVProject(
            title="Backend API",
            description="Built REST API with FastAPI",
            technologies=["Python", "FastAPI", "PostgreSQL"],
            duration="6 months",
            evidence_span=EvidenceSpan(start=100, end=150),
        )
        assert project.title == "Backend API"
        assert len(project.technologies) == 3
        assert project.evidence_span.start == 100

    def test_cv_skill(self):
        skill = CVSkill(
            skill_name="Python",
            proficiency_level="expert",
            years_of_experience=7.5,
            evidence_span=EvidenceSpan(start=50, end=60),
        )
        assert skill.skill_name == "Python"
        assert skill.years_of_experience == 7.5

    def test_parsed_cv(self):
        cv = ParsedCV(
            name="John Doe",
            email="john@example.com",
            phone="+1-555-0000",
            summary="Senior backend engineer",
            projects=[
                CVProject(title="Project 1", description="Desc 1"),
                CVProject(title="Project 2", description="Desc 2"),
            ],
            skills=[
                CVSkill(skill_name="Python", proficiency_level="expert"),
                CVSkill(skill_name="Go", proficiency_level="intermediate"),
            ],
            experience_years=8,
        )
        assert cv.name == "John Doe"
        assert len(cv.projects) == 2
        assert len(cv.skills) == 2


class TestJDModels:
    def test_jd_requirement(self):
        req = JDRequirement(
            text="5+ years of Python experience",
            requirement_type="must_have",
            category="experience",
        )
        assert req.text == "5+ years of Python experience"
        assert req.requirement_type == "must_have"

    def test_requirement_type_validation(self):
        """Ensure invalid type raises error"""
        with pytest.raises(ValueError):
            JDRequirement(text="Some requirement", requirement_type="invalid")

    def test_parsed_jd(self):
        jd = ParsedJD(
            job_title="Senior Backend Engineer",
            company="TechCorp",
            summary="We're looking for...",
            requirements=[
                JDRequirement(text="Python", requirement_type="must_have"),
                JDRequirement(text="Docker", requirement_type="nice_to_have"),
            ],
        )
        assert jd.job_title == "Senior Backend Engineer"
        assert len(jd.requirements) == 2


class TestPipelineModels:
    def test_triage_result(self):
        triage = TriageResult(
            ranked_projects=[{"id": "p1", "score": 0.95}],
            ranked_skills=[{"name": "Python", "score": 0.88}],
            flagged_for_enrichment=["p2", "s1"],
        )
        assert len(triage.ranked_projects) == 1
        assert len(triage.flagged_for_enrichment) == 2

    def test_enrichment_result(self):
        enrichment = EnrichmentResult(
            enrichment_context={"p1": "Recent context about project"},
            sources_used=["https://example.com"],
        )
        assert "p1" in enrichment.enrichment_context
        assert len(enrichment.sources_used) == 1

    def test_generated_question(self):
        question = GeneratedQuestion(
            question_text="Tell us about your Python experience",
            source_type="skill",
            source_id="skill_1",
            evidence_snippet="Expert in Python for 7 years",
            evidence_span={"start": 10, "end": 40},
        )
        assert question.source_type == "skill"
        assert question.evidence_span["start"] == 10

    def test_merged_question_set(self):
        qs = MergedQuestionSet(
            total_questions=5,
            questions=[
                GeneratedQuestion(
                    question_text="Q1", source_type="project", source_id="p1"
                ),
                GeneratedQuestion(
                    question_text="Q2", source_type="skill", source_id="s1"
                ),
            ],
            coverage={"req_1": True, "req_2": False},
        )
        assert qs.total_questions == 5
        assert len(qs.questions) == 2

    def test_judge_score(self):
        score = JudgeScore(
            question_id=1,
            relevance_score=8.5,
            groundedness_score=9.0,
            redundancy_score=7.0,
            reasoning="Highly relevant to the role",
        )
        assert score.relevance_score == 8.5
        assert 0 <= score.groundedness_score <= 10

    def test_judge_score_validation(self):
        """Ensure scores are 0-10"""
        with pytest.raises(ValueError):
            JudgeScore(
                question_id=1,
                relevance_score=11.0,  # Invalid
                groundedness_score=9.0,
                redundancy_score=7.0,
            )

    def test_fitness_score(self):
        score = FitnessScore(
            requirement_id=1,
            requirement_text="5+ years Python",
            requirement_type="must_have",
            match_score=8.0,
            evidence="CV shows 7 years Python",
            weighted_score=16.0,  # 8.0 × 2 (must_have weight)
        )
        assert score.match_score == 8.0
        assert score.weighted_score == 16.0

    def test_pipeline_state(self):
        cv = ParsedCV(name="John", projects=[], skills=[])
        jd = ParsedJD(job_title="Backend Engineer", requirements=[])

        state = PipelineState(
            run_id=1,
            cv_id=1,
            jd_id=1,
            cv=cv,
            jd=jd,
            prompt_version="v1.0",
            status="running",
        )
        assert state.run_id == 1
        assert state.status == "running"
        assert state.cv.name == "John"
        assert state.jd.job_title == "Backend Engineer"


class TestModelSerialization:
    def test_cv_to_dict(self):
        cv = ParsedCV(name="John", projects=[], skills=[])
        data = cv.model_dump()
        assert data["name"] == "John"
        assert isinstance(data, dict)

    def test_cv_to_json(self):
        cv = ParsedCV(name="John", projects=[], skills=[])
        json_str = cv.model_dump_json()
        assert isinstance(json_str, str)
        assert "John" in json_str

    def test_pipeline_state_serialization(self):
        state = PipelineState(run_id=1, cv_id=1, jd_id=1, status="pending")
        data = state.model_dump()
        assert data["run_id"] == 1
        assert data["status"] == "pending"
