"""Live demo: Generate skill questions using real OpenAI API + fixture data"""

import asyncio
import os
from dotenv import load_dotenv

from src.database.db import init_db
from src.models.cv import ParsedCV, CVSkill
from src.models.jd import ParsedJD, JDRequirement
from src.models.pipeline import TriageResult
from src.pipeline.nodes.skill_questions import skill_questions_node

load_dotenv()

# Initialize demo database
DB_PATH = "demo_skill_questions.db"
init_db(DB_PATH)


async def main():
    print("=" * 80)
    print("LIVE DEMO: Skill Questions Generation (Fixture Data)")
    print("=" * 80)

    # Real CV data with diverse skills
    parsed_cv = ParsedCV(
        name="Priya Sharma",
        summary="AI/ML Engineer with generative AI experience",
        projects=[],
        skills=[
            CVSkill(skill_name="Python", proficiency_level="expert", years_of_experience=6.0),
            CVSkill(skill_name="LLMs & Generative AI", proficiency_level="expert", years_of_experience=2.5),
            CVSkill(skill_name="LangChain", proficiency_level="advanced", years_of_experience=1.5),
            CVSkill(skill_name="AWS Cloud", proficiency_level="advanced", years_of_experience=4.0),
            CVSkill(skill_name="REST APIs", proficiency_level="expert", years_of_experience=5.0),
            CVSkill(skill_name="Docker & Containerization", proficiency_level="intermediate", years_of_experience=2.0),
        ],
    )

    # Parse JD from fixture
    with open("tests/fixtures/jobdesc1.txt", "r") as f:
        jd_text = f.read()

    parsed_jd = ParsedJD(
        job_title="Generative AI Developer",
        company="Impetus",
        summary="Seeking GenAI developer with LLM and cloud experience",
        requirements=[
            JDRequirement(text="3+ years professional software development", requirement_type="must_have"),
            JDRequirement(text="Strong Python proficiency", requirement_type="must_have"),
            JDRequirement(text="Hands-on cloud platform experience", requirement_type="must_have"),
            JDRequirement(text="LLM and generative AI concepts exposure", requirement_type="must_have"),
            JDRequirement(text="RESTful APIs and microservices experience", requirement_type="nice_to_have"),
            JDRequirement(text="RAG frameworks and implementations", requirement_type="nice_to_have"),
            JDRequirement(text="LangChain or orchestration frameworks", requirement_type="nice_to_have"),
            JDRequirement(text="AWS Cloud experience", requirement_type="nice_to_have"),
        ],
    )

    # Triage result (simulated - ranked skills)
    triage_result = TriageResult(
        ranked_projects=[],
        ranked_skills=[
            {
                "id": "skill_1",
                "skill_name": "Python",
                "score": 0.99,
                "skill": parsed_cv.skills[0].model_dump(),
            },
            {
                "id": "skill_2",
                "skill_name": "LLMs & Generative AI",
                "score": 0.97,
                "skill": parsed_cv.skills[1].model_dump(),
            },
            {
                "id": "skill_3",
                "skill_name": "LangChain",
                "score": 0.94,
                "skill": parsed_cv.skills[2].model_dump(),
            },
            {
                "id": "skill_4",
                "skill_name": "AWS Cloud",
                "score": 0.92,
                "skill": parsed_cv.skills[3].model_dump(),
            },
            {
                "id": "skill_5",
                "skill_name": "REST APIs",
                "score": 0.88,
                "skill": parsed_cv.skills[4].model_dump(),
            },
            {
                "id": "skill_6",
                "skill_name": "Docker & Containerization",
                "score": 0.75,
                "skill": parsed_cv.skills[5].model_dump(),
            },
        ],
        flagged_for_enrichment=["LangChain", "RAG"],
    )

    print(f"\n📋 CV: {parsed_cv.name}")
    print(f"   Skills: {len(parsed_cv.skills)}")
    for s in parsed_cv.skills:
        print(f"   - {s.skill_name} ({s.proficiency_level}, {s.years_of_experience}y)")

    print(f"\n🎯 JD: {parsed_jd.job_title} @ {parsed_jd.company}")
    print(f"   Requirements: {len(parsed_jd.requirements)}")

    print(f"\n⚙️  Running skill_questions_node...")
    print("   Generating questions for ranked skills...\n")

    # Run node
    result = await skill_questions_node({
        "run_id": 999,
        "cv_id": 999,
        "jd_id": 999,
        "triage_result": triage_result,
        "jd": parsed_jd,
        "db_path": DB_PATH,
    })

    if result["status"] == "failed":
        print(f"❌ Error: {result['error_message']}")
        return

    questions = result["skill_questions"]
    print(f"✅ Generated {len(questions)} questions!\n")

    # Display questions grouped by skill
    current_skill = None
    question_count = 0
    for q in questions:
        skill_id = q.get("source_id")
        if skill_id != current_skill:
            current_skill = skill_id
            skill_name = next(
                (s["skill_name"] for s in triage_result.ranked_skills if s["id"] == skill_id),
                "Unknown"
            )
            # Find proficiency level
            skill_obj = next(
                (s["skill"] for s in triage_result.ranked_skills if s["id"] == skill_id),
                {}
            )
            prof = skill_obj.get("proficiency_level", "unknown")
            print(f"\n📌 Skill: {skill_name} ({prof})")
            print("-" * 76)
            question_count = 0

        question_count += 1
        print(f"   Q{question_count}: {q['question_text']}")

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Cleanup
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
