"""Live demo: Generate real structured questions using OpenAI + Instructor"""

import asyncio
import os
import sys
import json
from typing import List
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import instructor
from openai import AsyncOpenAI

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.structured_question import StructuredQuestion, ProficiencyLevel, QuestionType
from src.utils.question_prompts import (
    PROJECT_QUESTIONS_SYSTEM_PROMPT,
    SKILL_QUESTIONS_SYSTEM_PROMPT,
    get_project_user_prompt,
    get_skill_user_prompt,
)

load_dotenv()


class ProjectQuestionsResponse(BaseModel):
    """Response model for project questions"""
    questions: List[StructuredQuestion]


class SkillQuestionsResponse(BaseModel):
    """Response model for skill questions"""
    questions: List[StructuredQuestion]


class LiveQuestionGenerator:
    """Generate real structured questions via OpenAI + Instructor"""

    def __init__(self):
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.client = instructor.from_openai(client)
        self.total_tokens = 0
        self.total_cost = 0.0

    async def generate_project_questions(
        self, project: dict, jd: dict, candidate_level: str = None
    ) -> List[StructuredQuestion]:
        """
        Generate structured questions from a project using OpenAI + Instructor.

        Args:
            project: Project dict with title, description, technologies
            jd: Job description dict with job_title, company, summary
            candidate_level: Optional level hint (junior/mid/senior)

        Returns:
            List of StructuredQuestion objects
        """
        user_prompt = get_project_user_prompt(project, jd, candidate_level)

        print(f"\n🔄 Generating project questions for '{project.get('title', 'Project')}'...")

        try:
            response = await self.client.chat.completions.create(
                model="gpt-5.4-nano",
                temperature=0.7,
                max_retries=2,
                messages=[
                    {"role": "system", "content": PROJECT_QUESTIONS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=ProjectQuestionsResponse,
            )

            # Track tokens
            if hasattr(response, "usage"):
                self.total_tokens += response.usage.prompt_tokens + response.usage.completion_tokens
                # Approximate cost (gpt-5.4-nano: ~$0.00015/1K input, $0.0006/1K output)
                self.total_cost += (
                    response.usage.prompt_tokens * 0.00015 / 1000
                    + response.usage.completion_tokens * 0.0006 / 1000
                )

            questions = response.questions

            # Enrich with project metadata
            for q in questions:
                q.source_type = "project"
                q.source_id = project.get("id", "project_unknown")

            print(f"   ✅ Generated {len(questions)} questions")
            return questions

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def generate_skill_questions(self, skill: dict, jd: dict) -> List[StructuredQuestion]:
        """
        Generate structured questions from a skill using OpenAI + Instructor.

        Args:
            skill: Skill dict with skill_name, proficiency_level, years_of_experience
            jd: Job description dict with job_title, company, summary

        Returns:
            List of StructuredQuestion objects
        """
        user_prompt = get_skill_user_prompt(skill, jd)

        print(f"\n🔄 Generating skill questions for '{skill.get('skill_name', 'Skill')}'...")

        try:
            response = await self.client.chat.completions.create(
                model="gpt-5.4-nano",
                temperature=0.7,
                max_retries=2,
                messages=[
                    {"role": "system", "content": SKILL_QUESTIONS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=SkillQuestionsResponse,
            )

            # Track tokens
            if hasattr(response, "usage"):
                self.total_tokens += response.usage.prompt_tokens + response.usage.completion_tokens
                self.total_cost += (
                    response.usage.prompt_tokens * 0.00015 / 1000
                    + response.usage.completion_tokens * 0.0006 / 1000
                )

            questions = response.questions

            # Enrich with skill metadata
            for q in questions:
                q.source_type = "skill"
                q.source_id = skill.get("id", "skill_unknown")

            print(f"   ✅ Generated {len(questions)} questions")
            return questions

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return []


def print_question(q: StructuredQuestion, idx: int):
    """Pretty print a structured question"""
    level_emoji = {"junior": "🟢", "mid": "🟡", "senior": "🔴"}
    type_icon = {
        "behavioral": "🎯",
        "technical": "⚙️",
        "design": "🏗️",
        "situational": "🤔",
    }

    print(f"\n   Q{idx}. [{level_emoji.get(q.level, '❓')} {q.level.upper()}] {type_icon.get(q.question_type, '❓')} {q.question_type.upper()}")
    print(f"       {q.question_text}")
    print(f"       Skill: {q.skill_required}")

    if q.relevance_note:
        print(f"       💡 {q.relevance_note}")

    print(f"       Key Points:")
    for point in q.key_points:
        print(f"         • {point}")

    if q.follow_ups:
        print(f"       Follow-ups:")
        for follow_up in q.follow_ups:
            print(f"         ➜ {follow_up}")


async def main():
    print("=" * 80)
    print("LIVE DEMO: Real Structured Questions via OpenAI + Instructor")
    print("=" * 80)

    # Sample data
    project = {
        "id": "proj_distributed_cache",
        "title": "Distributed Cache System",
        "description": "Built a distributed in-memory cache layer supporting multi-region replication with consistent hashing and automatic failover. Implemented using Redis Cluster with Lua scripting for atomic operations.",
        "technologies": ["Redis", "Lua", "Python", "Go", "gRPC"],
    }

    skill = {
        "id": "skill_system_design",
        "skill_name": "System Design",
        "proficiency_level": "intermediate",
        "years_of_experience": 4,
    }

    jd = {
        "job_title": "Senior Backend Engineer",
        "company": "TechCorp",
        "summary": "We're looking for a backend engineer experienced with distributed systems, microservices, and high-scale infrastructure.",
    }

    generator = LiveQuestionGenerator()

    print(f"\n📝 INPUTS")
    print("-" * 80)
    print(f"Project: {project['title']}")
    print(f"Skill: {skill['skill_name']}")
    print(f"JD: {jd['job_title']} @ {jd['company']}")

    # Generate project questions
    project_qs = await generator.generate_project_questions(project, jd)

    # Generate skill questions
    skill_qs = await generator.generate_skill_questions(skill, jd)

    all_questions = project_qs + skill_qs

    if not all_questions:
        print("\n❌ No questions generated. Check API key and network.")
        return

    # Calculate level distribution
    level_dist = {
        "junior": sum(1 for q in all_questions if q.level == ProficiencyLevel.JUNIOR),
        "mid": sum(1 for q in all_questions if q.level == ProficiencyLevel.MID),
        "senior": sum(1 for q in all_questions if q.level == ProficiencyLevel.SENIOR),
    }

    print(f"\n📊 RESULTS")
    print("-" * 80)
    print(f"Project Questions: {len(project_qs)}")
    print(f"Skill Questions: {len(skill_qs)}")
    print(f"Total: {len(all_questions)}")

    print(f"\n📈 PROFICIENCY LEVEL DISTRIBUTION")
    print("-" * 80)
    for level, count in level_dist.items():
        bar = "█" * count + "░" * (6 - count)
        pct = (count / len(all_questions) * 100) if all_questions else 0
        print(f"   {level.upper():6s}: {bar} ({count}, {pct:.0f}%)")

    print(f"\n🎯 PROJECT QUESTIONS")
    print("-" * 80)
    for idx, q in enumerate(project_qs, 1):
        print_question(q, idx)

    print(f"\n💾 SKILL QUESTIONS")
    print("-" * 80)
    for idx, q in enumerate(skill_qs, 1):
        print_question(q, idx)

    # Show structured data
    print(f"\n📋 STRUCTURED DATA (JSON-ready)")
    print("-" * 80)
    if all_questions:
        print("First question:")
        q_dict = all_questions[0].model_dump()
        print(json.dumps(q_dict, indent=2, default=str))

    # Statistics
    print(f"\n📊 STATISTICS")
    print("-" * 80)
    avg_key_points = sum(len(q.key_points) for q in all_questions) / len(all_questions)
    avg_follow_ups = sum(len(q.follow_ups) for q in all_questions) / len(all_questions)
    question_types = {}
    for q in all_questions:
        question_types[q.question_type] = question_types.get(q.question_type, 0) + 1

    print(f"   Average key points per question: {avg_key_points:.1f}")
    print(f"   Average follow-ups per question: {avg_follow_ups:.1f}")
    print(f"   Question types:")
    for qtype, count in question_types.items():
        print(f"     • {qtype}: {count}")

    print(f"\n💰 API USAGE")
    print("-" * 80)
    print(f"   Total tokens: {generator.total_tokens:,}")
    print(f"   Estimated cost: ${generator.total_cost:.4f}")

    print("\n" + "=" * 80)
    print("Live Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
