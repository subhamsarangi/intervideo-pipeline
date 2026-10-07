"""Live demo: Generate project questions using real OpenAI API + fixture data"""

import asyncio
import os
from dotenv import load_dotenv

from src.database.db import init_db
from src.models.cv import ParsedCV, CVProject
from src.models.jd import ParsedJD, JDRequirement
from src.models.pipeline import TriageResult
from src.pipeline.nodes.project_questions import project_questions_node

load_dotenv()

# Initialize demo database
DB_PATH = "demo_project_questions.db"
init_db(DB_PATH)


async def main():
    print("=" * 80)
    print("LIVE DEMO: Project Questions Generation (Fixture Data)")
    print("=" * 80)

    # Real CV data
    parsed_cv = ParsedCV(
        name="Priya Sharma",
        summary="AI/ML Engineer with generative AI experience",
        projects=[
            CVProject(
                title="LangChain-based RAG Pipeline",
                description=(
                    "Built a Retrieval-Augmented Generation (RAG) system using LangChain and OpenAI embeddings. "
                    "Integrated with custom knowledge base, achieved 92% accuracy on domain-specific queries. "
                    "Deployed on AWS with Lambda for serverless execution, handles 1000s of requests daily."
                ),
                technologies=["Python", "LangChain", "OpenAI", "AWS", "Lambda", "RAG"],
            ),
            CVProject(
                title="Generative AI Chatbot for Customer Support",
                description=(
                    "Developed production chatbot using GPT-4 fine-tuning and prompt engineering. "
                    "Integrated with Slack and web interfaces, reduced support tickets by 40%. "
                    "Implemented context management with Redis, handled multi-turn conversations."
                ),
                technologies=["Python", "GPT-4", "LLMs", "Slack API", "Redis", "FastAPI"],
            ),
            CVProject(
                title="Cloud-based ML Model Deployment",
                description=(
                    "Deployed machine learning models on AWS SageMaker with auto-scaling. "
                    "Built CI/CD pipeline with GitHub Actions, achieved sub-second inference latency. "
                    "Monitored performance metrics, maintained 99.95% uptime."
                ),
                technologies=["Python", "AWS SageMaker", "Docker", "GitHub Actions", "ML"],
            ),
        ],
        skills=[],
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
            JDRequirement(text="Problem-solving and analytical mindset", requirement_type="must_have"),
            JDRequirement(text="RESTful APIs and microservices experience", requirement_type="nice_to_have"),
            JDRequirement(text="RAG frameworks and implementations", requirement_type="nice_to_have"),
            JDRequirement(text="LangChain or orchestration frameworks", requirement_type="nice_to_have"),
            JDRequirement(text="AWS Cloud experience", requirement_type="nice_to_have"),
        ],
    )

    # Triage result (simulated - ranked projects)
    triage_result = TriageResult(
        ranked_projects=[
            {
                "id": "proj_1",
                "title": "LangChain-based RAG Pipeline",
                "score": 0.98,
                "project": parsed_cv.projects[0].model_dump(),
            },
            {
                "id": "proj_2",
                "title": "Generative AI Chatbot for Customer Support",
                "score": 0.96,
                "project": parsed_cv.projects[1].model_dump(),
            },
            {
                "id": "proj_3",
                "title": "Cloud-based ML Model Deployment",
                "score": 0.87,
                "project": parsed_cv.projects[2].model_dump(),
            },
        ],
        ranked_skills=[],
        flagged_for_enrichment=["LangChain", "RAG"],
    )

    print(f"\n📋 CV: {parsed_cv.name}")
    print(f"   Projects: {len(parsed_cv.projects)}")
    for p in parsed_cv.projects:
        print(f"   - {p.title} ({', '.join(p.technologies)})")

    print(f"\n🎯 JD: {parsed_jd.job_title} @ {parsed_jd.company}")
    print(f"   Requirements: {len(parsed_jd.requirements)}")

    print(f"\n⚙️  Running project_questions_node...")
    print("   Generating questions for ranked projects...\n")

    # Run node
    result = await project_questions_node({
        "run_id": 999,  # Dummy run_id (no DB access)
        "cv_id": 999,
        "jd_id": 999,
        "triage_result": triage_result,
        "cv": parsed_cv,
        "jd": parsed_jd,
        "db_path": DB_PATH,
    })

    if result["status"] == "failed":
        print(f"❌ Error: {result['error_message']}")
        return

    questions = result["project_questions"]
    print(f"✅ Generated {len(questions)} questions!\n")

    # Display questions grouped by project
    current_project = None
    for q in questions:
        proj_id = q.get("source_id")
        if proj_id != current_project:
            current_project = proj_id
            proj_title = next(
                (p["title"] for p in triage_result.ranked_projects if p["id"] == proj_id),
                "Unknown"
            )
            print(f"\n📌 Project: {proj_title}")
            print("-" * 76)

        print(f"   Q: {q['question_text']}")
        print()

    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Cleanup
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
