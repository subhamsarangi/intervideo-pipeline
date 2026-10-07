"""Seed dummy data for UI testing"""

import json
from src.database.db import (
    create_cv,
    create_jd,
    create_pipeline_run,
    create_pipeline_step,
    create_question,
    create_fitness_score,
    update_cv_status,
    update_jd_status,
    update_pipeline_run_status,
    update_pipeline_step,
    get_db_path,
)


DUMMY_MARKER = "DUMMY_TEST_DATA"


def seed_data():
    """Add dummy data to database"""
    db_path = get_db_path()

    # Create CV
    cv_id = create_cv(
        f"Test CV - {DUMMY_MARKER}",
        "Senior Backend Engineer with 8 years experience in Python, Go, and Rust. "
        "Expertise in microservices, Kubernetes, and cloud infrastructure. "
        "Led teams at startup and FAANG. Projects: payment system, analytics platform, ML pipeline.",
        db_path,
    )

    parsed_cv = {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "projects": [
            {
                "title": "Payment Processing System",
                "description": "Built distributed payment processor handling $1M+ daily",
                "technologies": ["Python", "PostgreSQL", "Kafka"],
            },
            {
                "title": "Analytics Platform",
                "description": "Real-time analytics with 100M+ events/day",
                "technologies": ["Go", "Cassandra", "ClickHouse"],
            },
        ],
        "skills": [
            {
                "skill_name": "Python",
                "proficiency_level": "expert",
                "years_of_experience": 8,
            },
            {
                "skill_name": "Go",
                "proficiency_level": "advanced",
                "years_of_experience": 5,
            },
            {
                "skill_name": "Kubernetes",
                "proficiency_level": "expert",
                "years_of_experience": 4,
            },
        ],
        "experience_years": 8,
    }
    update_cv_status(cv_id, "completed", parsed_cv, None, db_path)
    print(f"✓ Created CV (id={cv_id})")

    # Create JD
    jd_id = create_jd(
        f"Senior Backend Engineer - {DUMMY_MARKER}",
        "Looking for senior backend engineer to join our platform team. "
        "Must have: 5+ years backend, Python or Go. Nice to have: Kubernetes, distributed systems.",
        "pasted",
        None,
        db_path,
    )

    parsed_jd = {
        "job_title": "Senior Backend Engineer",
        "company": "TechCorp",
        "requirements": [
            {"text": "5+ years backend development", "requirement_type": "must_have"},
            {"text": "Python or Go expertise", "requirement_type": "must_have"},
            {"text": "Kubernetes experience", "requirement_type": "nice_to_have"},
            {"text": "Leadership experience", "requirement_type": "nice_to_have"},
        ],
    }
    update_jd_status(jd_id, "completed", parsed_jd, None, db_path)
    print(f"✓ Created JD (id={jd_id})")

    # Create pipeline run
    run_id = create_pipeline_run(cv_id, jd_id, "v1.0", db_path)

    # Create pipeline steps
    step1_id = create_pipeline_step(run_id, "security_validation", db_path)
    update_pipeline_step(
        step1_id, "completed", {"validated": True, "safe": True}, None, db_path
    )

    step2_id = create_pipeline_step(run_id, "relevance_triage", db_path)
    update_pipeline_step(
        step2_id,
        "completed",
        {
            "ranked_projects": [
                {"id": "p1", "title": "Payment System", "relevance_score": 0.95},
                {"id": "p2", "title": "Analytics Platform", "relevance_score": 0.88},
            ]
        },
        None,
        db_path,
    )

    step3_id = create_pipeline_step(run_id, "batched_generation_projects", db_path)
    update_pipeline_step(
        step3_id,
        "completed",
        {
            "p1": [
                "Describe your experience building distributed payment systems",
                "How did you handle failure scenarios in high-stakes transactions?",
            ]
        },
        None,
        db_path,
    )

    step4_id = create_pipeline_step(run_id, "merge_dedupe", db_path)
    update_pipeline_step(
        step4_id, "completed", {"total_questions": 5, "deduped": 1}, None, db_path
    )

    step5_id = create_pipeline_step(run_id, "llm_judge_eval", db_path)
    update_pipeline_step(step5_id, "completed", {"avg_score": 8.3}, None, db_path)

    update_pipeline_run_status(
        run_id,
        "completed",
        {"total_questions": 5, "avg_judge_score": 8.3, "fitness_score": 0.82},
        None,
        db_path,
    )
    print(f"✓ Created pipeline run (id={run_id}) with 5 steps")

    # Create questions
    q1_id = create_question(
        run_id,
        "Describe your experience building distributed payment systems at scale",
        "project",
        "p1",
        "Built distributed payment processor handling $1M+ daily",
        None,
        db_path,
    )

    q2_id = create_question(
        run_id,
        "How did you handle failure scenarios in high-stakes transactions?",
        "project",
        "p1",
        "Led teams at startup and FAANG",
        None,
        db_path,
    )

    q3_id = create_question(
        run_id,
        "Tell us about your Kubernetes expertise and how you've used it in production",
        "skill",
        "k8s",
        "Expertise in microservices, Kubernetes, and cloud infrastructure",
        None,
        db_path,
    )

    print(f"✓ Created 3 questions")

    # Create fitness scores
    create_fitness_score(
        run_id,
        "5+ years backend development",
        "must_have",
        9.0,
        "Jane has 8 years",
        db_path,
    )
    create_fitness_score(
        run_id, "Python or Go expertise", "must_have", 8.5, "Expert in both", db_path
    )
    create_fitness_score(
        run_id,
        "Kubernetes experience",
        "nice_to_have",
        8.0,
        "4 years production K8s",
        db_path,
    )
    print(f"✓ Created 3 fitness scores")

    print(f"\n✅ Dummy data seeded! Marked with: {DUMMY_MARKER}")
    print(f"   CV ID: {cv_id}")
    print(f"   JD ID: {jd_id}")
    print(f"   Run ID: {run_id}")


if __name__ == "__main__":
    seed_data()
