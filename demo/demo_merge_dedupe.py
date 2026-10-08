"""Live demo: Merge, deduplicate, and rerank questions"""

import asyncio
import os
from dotenv import load_dotenv

from src.database.db import init_db
from src.models.jd import ParsedJD, JDRequirement
from src.models.pipeline import GeneratedQuestion
from src.pipeline.nodes.merge_dedupe import merge_dedupe_node

load_dotenv()

# Initialize demo database
DB_PATH = "demo_merge_dedupe.db"
init_db(DB_PATH)


async def main():
    print("=" * 80)
    print("LIVE DEMO: Merge, Deduplicate, and Rerank Questions")
    print("=" * 80)

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

    # Simulated project questions (from earlier demo)
    project_questions = [
        GeneratedQuestion(
            question_text="Can you explain the architecture of your LangChain-based RAG pipeline and how each component interacts?",
            source_type="project",
            source_id="proj_1",
            evidence_snippet="LangChain-based RAG Pipeline",
        ).model_dump(),
        GeneratedQuestion(
            question_text="What were the key factors that contributed to achieving 92% accuracy in your system?",
            source_type="project",
            source_id="proj_1",
            evidence_snippet="LangChain-based RAG Pipeline",
        ).model_dump(),
        GeneratedQuestion(
            question_text="How did you handle the integration of the custom knowledge base with LangChain?",
            source_type="project",
            source_id="proj_1",
            evidence_snippet="LangChain-based RAG Pipeline",
        ).model_dump(),
        GeneratedQuestion(
            question_text="Can you explain the architecture of your CI/CD pipeline using GitHub Actions?",
            source_type="project",
            source_id="proj_3",
            evidence_snippet="Cloud-based ML Model Deployment",
        ).model_dump(),
    ]

    # Simulated skill questions (from earlier demo)
    skill_questions = [
        GeneratedQuestion(
            question_text="Can you describe a project where you implemented a generative AI model using Python?",
            source_type="skill",
            source_id="skill_1",
            evidence_snippet="Python (expert, 6.0y)",
        ).model_dump(),
        GeneratedQuestion(
            question_text="How do you optimize the performance of Python code when working with large language models?",
            source_type="skill",
            source_id="skill_1",
            evidence_snippet="Python (expert, 6.0y)",
        ).model_dump(),
        GeneratedQuestion(
            question_text="What are best practices for managing dependencies in Python when developing LLM applications?",
            source_type="skill",
            source_id="skill_1",
            evidence_snippet="Python (expert, 6.0y)",
        ).model_dump(),
        GeneratedQuestion(
            question_text="Can you describe a project where you implemented a large language model in a cloud environment?",
            source_type="skill",
            source_id="skill_2",
            evidence_snippet="LLMs & Generative AI (expert, 2.5y)",
        ).model_dump(),
        GeneratedQuestion(
            question_text="How do you approach fine-tuning a pre-trained LLM for a specific application?",
            source_type="skill",
            source_id="skill_2",
            evidence_snippet="LLMs & Generative AI (expert, 2.5y)",
        ).model_dump(),
        GeneratedQuestion(
            question_text="Can you explain how you have used LangChain to manage interactions between different LLMs?",
            source_type="skill",
            source_id="skill_3",
            evidence_snippet="LangChain (advanced, 1.5y)",
        ).model_dump(),
    ]

    print(f"\n📊 Input Summary:")
    print(f"   Project questions: {len(project_questions)}")
    print(f"   Skill questions: {len(skill_questions)}")
    print(f"   Total questions: {len(project_questions) + len(skill_questions)}")

    print(f"\n🎯 JD: {parsed_jd.job_title} @ {parsed_jd.company}")
    print(f"   Requirements: {len(parsed_jd.requirements)}")
    for req in parsed_jd.requirements:
        print(f"   - {req.text} ({req.requirement_type})")

    print(f"\n⚙️  Running merge_dedupe_node...")
    print("   Combining, deduplicating, checking coverage, reranking...\n")

    # Run node
    result = await merge_dedupe_node({
        "run_id": 999,
        "cv_id": 999,
        "jd_id": 999,
        "project_questions": project_questions,
        "skill_questions": skill_questions,
        "jd": parsed_jd,
        "db_path": DB_PATH,
    })

    if result["status"] == "failed":
        print(f"❌ Error: {result['error_message']}")
        return

    merged = result["merged_questions"]
    gaps = result["coverage_gaps"]

    print(f"✅ Processing complete!\n")
    print(f"📈 Output Summary:")
    print(f"   Original: {len(project_questions) + len(skill_questions)} questions")
    print(f"   After dedup + rerank: {len(merged)} questions")
    print(f"   Reduction: {len(project_questions) + len(skill_questions) - len(merged)} removed (duplicates)")
    print(f"   Coverage gaps: {len(gaps)} uncovered requirements")

    print(f"\n📌 Final Question List (Ranked by Relevance):")
    print("-" * 76)
    for i, q in enumerate(merged, 1):
        source_type = q.get("source_type", "unknown")
        source_id = q.get("source_id", "?")
        print(f"\n   {i}. [{source_type.upper()}] {q['question_text']}")

    if gaps:
        print(f"\n⚠️  Coverage Gaps ({len(gaps)} uncovered requirements):")
        print("-" * 76)
        for gap in gaps:
            print(f"   • {gap['requirement']} ({gap['type']})")
            print(f"     Reason: {gap['reason']}")
    else:
        print(f"\n✅ All JD requirements covered!")

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
