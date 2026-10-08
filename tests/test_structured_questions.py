"""Tests for structured questions with proficiency levels and Instructor"""

import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock

from src.database.db import (
    init_db,
    create_cv,
    create_jd,
    create_pipeline_run,
    get_pipeline_steps,
    update_cv_status,
    update_jd_status,
)
from src.models.cv import ParsedCV, CVProject, CVSkill
from src.models.jd import ParsedJD, JDRequirement
from src.models.structured_question import (
    StructuredQuestion,
    ProficiencyLevel,
    QuestionType,
)
from src.pipeline.nodes.project_questions import project_questions_node
from src.pipeline.nodes.skill_questions import skill_questions_node


@pytest.fixture
def test_db():
    """Create temporary test database"""
    db_path = "test_structured_questions.db"
    init_db(db_path)
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


class TestStructuredQuestionSchema:
    """Test StructuredQuestion schema validation"""

    def test_structured_question_valid(self):
        """Test creating valid structured question"""
        q = StructuredQuestion(
            question_text="How did you optimize performance?",
            level=ProficiencyLevel.MID,
            skill_required="System Design",
            question_type=QuestionType.TECHNICAL,
            key_points=["Identified bottleneck", "Implemented caching", "Measured improvement"],
            follow_ups=["Why that specific approach?", "What alternatives did you consider?"],
            relevance_note="Tests architecture thinking",
            source_type="project",
            source_id="proj_1",
        )
        
        assert q.question_text == "How did you optimize performance?"
        assert q.level == ProficiencyLevel.MID
        assert q.skill_required == "System Design"
        assert len(q.key_points) == 3
        assert len(q.follow_ups) == 2

    def test_structured_question_min_key_points(self):
        """Test schema enforces minimum 3 key points"""
        with pytest.raises(ValueError):
            StructuredQuestion(
                question_text="Test question",
                level=ProficiencyLevel.JUNIOR,
                skill_required="Python",
                question_type=QuestionType.BEHAVIORAL,
                key_points=["Only", "Two"],  # Too few
            )

    def test_structured_question_max_follow_ups(self):
        """Test schema enforces maximum 3 follow-ups"""
        with pytest.raises(ValueError):
            StructuredQuestion(
                question_text="Test question",
                level=ProficiencyLevel.SENIOR,
                skill_required="AWS",
                question_type=QuestionType.DESIGN,
                key_points=["A", "B", "C"],
                follow_ups=["Q1", "Q2", "Q3", "Q4"],  # Too many
            )

    def test_proficiency_levels(self):
        """Test all proficiency levels are valid"""
        for level in [ProficiencyLevel.JUNIOR, ProficiencyLevel.MID, ProficiencyLevel.SENIOR]:
            q = StructuredQuestion(
                question_text=f"Test {level}",
                level=level,
                skill_required="Test Skill",
                question_type=QuestionType.TECHNICAL,
                key_points=["A", "B", "C"],
            )
            assert q.level == level

    def test_question_types(self):
        """Test all question types are valid"""
        for qtype in [
            QuestionType.BEHAVIORAL,
            QuestionType.TECHNICAL,
            QuestionType.DESIGN,
            QuestionType.SITUATIONAL,
        ]:
            q = StructuredQuestion(
                question_text=f"Test {qtype}",
                level=ProficiencyLevel.MID,
                skill_required="Skill",
                question_type=qtype,
                key_points=["A", "B", "C"],
            )
            assert q.question_type == qtype


class TestProjectQuestionsStructured:
    """Test project questions with structured output"""

    @pytest.mark.asyncio
    async def test_project_questions_structured_output(self, test_db):
        """Test project questions returns structured questions with levels"""
        from src.models.pipeline import TriageResult

        parsed_cv = ParsedCV(
            name="Alice",
            summary="Dev",
            projects=[
                CVProject(
                    title="Distributed Cache",
                    description="Built Redis cluster handling 100K ops/sec",
                    technologies=["Redis", "Go", "Distributed Systems"],
                )
            ],
            skills=[],
        )

        parsed_jd = ParsedJD(
            job_title="Backend Engineer",
            requirements=[JDRequirement(text="System design", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        update_cv_status(cv_id=cv_id, status="completed", parsed_data=parsed_cv.model_dump(), db_path=test_db)

        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        update_jd_status(jd_id=jd_id, status="completed", parsed_data=parsed_jd.model_dump(), db_path=test_db)

        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        # Mock Instructor response
        mock_questions = [
            StructuredQuestion(
                question_text="How did you handle cache consistency?",
                level=ProficiencyLevel.MID,
                skill_required="Distributed Systems",
                question_type=QuestionType.TECHNICAL,
                key_points=["Consistency model", "Trade-offs", "Implementation"],
                follow_ups=["Alternatives?", "Under 10x load?"],
            ),
            StructuredQuestion(
                question_text="Describe the replication strategy",
                level=ProficiencyLevel.SENIOR,
                skill_required="System Design",
                question_type=QuestionType.DESIGN,
                key_points=["Replication approach", "Failure handling", "Monitoring"],
                follow_ups=["How to recover?"],
            ),
        ]

        triage_result = TriageResult(
            ranked_projects=[
                {
                    "id": "proj_1",
                    "title": "Distributed Cache",
                    "score": 0.95,
                    "project": parsed_cv.projects[0].model_dump(),
                }
            ],
            ranked_skills=[],
            flagged_for_enrichment=[],
        )

        with patch("src.pipeline.nodes.project_questions.instructor.from_openai") as mock_instructor:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.questions = mock_questions
            mock_response.usage.prompt_tokens = 150
            mock_response.usage.completion_tokens = 100
            
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_instructor.return_value = mock_client

            with patch("src.pipeline.nodes.project_questions.increment_llm_calls"):
                result = await project_questions_node({
                    "run_id": run_id,
                    "cv_id": cv_id,
                    "jd_id": jd_id,
                    "triage_result": triage_result,
                    "cv": parsed_cv,
                    "jd": parsed_jd,
                    "db_path": test_db,
                })

        assert result["status"] == "running"
        assert len(result["project_questions"]) == 2
        
        # Verify structured fields
        q1 = result["project_questions"][0]
        assert q1["level"] == "mid"
        assert len(q1["key_points"]) == 3
        assert len(q1["follow_ups"]) == 2
        
        q2 = result["project_questions"][1]
        assert q2["level"] == "senior"
        
        # Verify level distribution
        assert result["level_distribution"]["mid"] == 1
        assert result["level_distribution"]["senior"] == 1

    @pytest.mark.asyncio
    async def test_project_questions_mixed_levels(self, test_db):
        """Test that questions are assigned appropriate levels"""
        from src.models.pipeline import TriageResult

        parsed_cv = ParsedCV(
            name="Bob",
            summary="Dev",
            projects=[
                CVProject(
                    title="Web App",
                    description="Built with React and Node",
                    technologies=["React", "Node.js"],
                )
            ],
            skills=[],
        )

        parsed_jd = ParsedJD(
            job_title="Frontend Dev",
            requirements=[JDRequirement(text="React", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        update_cv_status(cv_id=cv_id, status="completed", parsed_data=parsed_cv.model_dump(), db_path=test_db)

        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        update_jd_status(jd_id=jd_id, status="completed", parsed_data=parsed_jd.model_dump(), db_path=test_db)

        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        # Create questions with mixed levels
        mock_questions = [
            StructuredQuestion(
                question_text="What is JSX?",
                level=ProficiencyLevel.JUNIOR,
                skill_required="React",
                question_type=QuestionType.TECHNICAL,
                key_points=["JSX basics", "Syntax", "Benefits"],
            ),
            StructuredQuestion(
                question_text="Describe lifecycle hooks",
                level=ProficiencyLevel.MID,
                skill_required="React",
                question_type=QuestionType.TECHNICAL,
                key_points=["Hooks", "Use cases", "Performance"],
            ),
            StructuredQuestion(
                question_text="Design a scalable state management",
                level=ProficiencyLevel.SENIOR,
                skill_required="React Architecture",
                question_type=QuestionType.DESIGN,
                key_points=["Architecture", "Scalability", "Testing"],
            ),
        ]

        triage_result = TriageResult(
            ranked_projects=[
                {
                    "id": "proj_1",
                    "title": "Web App",
                    "score": 0.9,
                    "project": parsed_cv.projects[0].model_dump(),
                }
            ],
            ranked_skills=[],
            flagged_for_enrichment=[],
        )

        with patch("src.pipeline.nodes.project_questions.instructor.from_openai") as mock_instructor:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.questions = mock_questions
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 80
            
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_instructor.return_value = mock_client

            with patch("src.pipeline.nodes.project_questions.increment_llm_calls"):
                result = await project_questions_node({
                    "run_id": run_id,
                    "cv_id": cv_id,
                    "jd_id": jd_id,
                    "triage_result": triage_result,
                    "cv": parsed_cv,
                    "jd": parsed_jd,
                    "db_path": test_db,
                })

        assert result["level_distribution"]["junior"] == 1
        assert result["level_distribution"]["mid"] == 1
        assert result["level_distribution"]["senior"] == 1


class TestSkillQuestionsStructured:
    """Test skill questions with structured output"""

    @pytest.mark.asyncio
    async def test_skill_questions_structured_output(self, test_db):
        """Test skill questions returns structured questions with levels"""
        from src.models.pipeline import TriageResult

        parsed_cv = ParsedCV(
            name="Charlie",
            summary="Dev",
            projects=[],
            skills=[
                CVSkill(skill_name="Python", proficiency_level="expert", years_of_experience=6.0),
            ],
        )

        parsed_jd = ParsedJD(
            job_title="Python Backend",
            requirements=[JDRequirement(text="Python", requirement_type="must_have")],
        )

        cv_id = create_cv("CV", "raw", test_db)
        update_cv_status(cv_id=cv_id, status="completed", parsed_data=parsed_cv.model_dump(), db_path=test_db)

        jd_id = create_jd("JD", "raw", "pasted", None, test_db)
        update_jd_status(jd_id=jd_id, status="completed", parsed_data=parsed_jd.model_dump(), db_path=test_db)

        run_id = create_pipeline_run(cv_id, jd_id, db_path=test_db)

        # Expert with 6 years should get senior level questions
        mock_questions = [
            StructuredQuestion(
                question_text="Design a high-performance async framework",
                level=ProficiencyLevel.SENIOR,
                skill_required="Python",
                question_type=QuestionType.DESIGN,
                key_points=["Async patterns", "Scalability", "Edge cases"],
                follow_ups=["How to handle exceptions?", "Monitoring?"],
            ),
        ]

        triage_result = TriageResult(
            ranked_projects=[],
            ranked_skills=[
                {
                    "id": "skill_1",
                    "skill_name": "Python",
                    "score": 0.98,
                    "skill": parsed_cv.skills[0].model_dump(),
                }
            ],
            flagged_for_enrichment=[],
        )

        with patch("src.pipeline.nodes.skill_questions.instructor.from_openai") as mock_instructor:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.questions = mock_questions
            mock_response.usage.prompt_tokens = 120
            mock_response.usage.completion_tokens = 90
            
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_instructor.return_value = mock_client

            with patch("src.pipeline.nodes.skill_questions.increment_llm_calls"):
                result = await skill_questions_node({
                    "run_id": run_id,
                    "cv_id": cv_id,
                    "jd_id": jd_id,
                    "triage_result": triage_result,
                    "jd": parsed_jd,
                    "db_path": test_db,
                })

        assert result["status"] == "running"
        assert len(result["skill_questions"]) == 1
        assert result["skill_questions"][0]["level"] == "senior"
        assert result["level_distribution"]["senior"] == 1
