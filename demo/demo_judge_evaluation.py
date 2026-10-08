"""Live demo: LLM Judge evaluation with scoring and persistence"""

import asyncio
import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db import init_db, get_question, list_questions_by_run, save_question
from src.models.judge_evaluation import JudgeScores, QuestionScore
from src.utils.judge_prompts import JUDGE_SYSTEM_PROMPT, get_judge_user_prompt

load_dotenv()


def print_score_bar(score: float, label: str = "") -> str:
    """Pretty print a score as a bar"""
    filled = int(score * 20)  # 20 chars = 1.0
    bar = "█" * filled + "░" * (20 - filled)
    pct = score * 100
    return f"{label:20s} {bar} {pct:5.1f}%"


async def evaluate_questions_live():
    """Live evaluation of questions via OpenAI + Judge"""

    print("=" * 100)
    print("LIVE DEMO: LLM Judge Evaluation & Persistence")
    print("=" * 100)

    # Sample questions to evaluate
    questions = [
        {
            "question_text": "Describe how you would design a distributed cache system for a high-traffic e-commerce platform.",
            "level": "senior",
            "skill_required": "Distributed Systems",
            "question_type": "design",
            "key_points": ["Consistency models", "Fault tolerance", "Scalability"],
            "follow_ups": ["How would you handle network partitions?"],
            "source_type": "project",
            "source_id": "proj_cache",
        },
        {
            "question_text": "Tell me about a time you debugged a performance issue in production. What tools did you use?",
            "level": "mid",
            "skill_required": "Backend Performance",
            "question_type": "behavioral",
            "key_points": ["Problem identification", "Debugging methodology", "Solution implementation"],
            "follow_ups": ["How did you prevent it from happening again?"],
            "source_type": "project",
            "source_id": "proj_perf",
        },
        {
            "question_text": "What is your favorite color and why?",
            "level": "junior",
            "skill_required": "General",
            "question_type": "behavioral",
            "key_points": ["Personal preference"],
            "follow_ups": [],
            "source_type": "skill",
            "source_id": "skill_comm",
        },
        {
            "question_text": "Explain the difference between const and let in JavaScript.",
            "level": "junior",
            "skill_required": "JavaScript",
            "question_type": "technical",
            "key_points": ["Scope differences", "Reassignment rules", "Hoisting"],
            "follow_ups": ["When would you use one over the other?"],
            "source_type": "skill",
            "source_id": "skill_js",
        },
        {
            "question_text": "How would you migrate a monolith to microservices without downtime?",
            "level": "senior",
            "skill_required": "System Architecture",
            "question_type": "design",
            "key_points": ["Strangler pattern", "Data migration", "Backward compatibility"],
            "follow_ups": ["How do you manage data consistency?", "What's your rollback strategy?"],
            "source_type": "project",
            "source_id": "proj_migration",
        },
    ]

    # JD context
    jd = {
        "job_title": "Senior Backend Engineer",
        "summary": "Build scalable, reliable backend systems for high-traffic platforms",
        "requirements": [
            "Distributed systems design",
            "Performance optimization",
            "System architecture",
            "Python & JavaScript proficiency",
        ],
    }

    # CV context
    cv = {"name": "Alice Johnson", "total_years": 6}

    print(f"\n📝 INPUT")
    print("-" * 100)
    print(f"Questions to evaluate: {len(questions)}")
    print(f"Role: {jd['job_title']}")
    print(f"Candidate: {cv['name']}, {cv['total_years']} years")

    # Call judge
    print(f"\n🔄 Calling LLM Judge...")
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    client = instructor.from_openai(client)

    user_prompt = get_judge_user_prompt(questions, jd, cv)

    class JudgeResponseModel(BaseModel):
        """Response model for judge"""
        question_scores: list[QuestionScore]

    try:
        response = await client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_model=JudgeResponseModel,
            temperature=0.3,
            reasoning_effort= 'none'
        )

        print(f"   ✅ Judge completed")

        # Match scores to questions by question_index (not by position)
        by_index = {s.question_index: s for s in response.question_scores}
        missing = [i for i in range(1, len(questions) + 1) if i not in by_index]
        if missing:
            print(f"   ⚠️  Judge returned no score for question(s): {missing}")
        scored = [
            (questions[i - 1], by_index[i])
            for i in sorted(by_index)
            if 1 <= i <= len(questions)
        ]

        # Initialize demo database
        db_path = "demo_judge.db"
        init_db(db_path)

        # Save questions
        run_id = 1
        passed = []
        rejected = []

        print(f"\n💾 Saving to database...")
        for i, (q, score_obj) in enumerate(scored):
            judge_scores = {
                "clarity": score_obj.scores.clarity,
                "relevance": score_obj.scores.relevance,
                "difficulty_match": score_obj.scores.difficulty_match,
                "groundedness": score_obj.scores.groundedness,
                "redundancy": score_obj.scores.redundancy,
                "overall": score_obj.overall_score,
            }

            status = "candidate" if score_obj.pass_threshold else "rejected_auto"
            question_id = save_question(
                run_id=run_id,
                question_data=q,
                judge_scores=judge_scores,
                judge_feedback=score_obj.feedback,
                judge_pass=score_obj.pass_threshold,
                status=status,
                db_path=db_path,
            )

            if score_obj.pass_threshold:
                passed.append(question_id)
            else:
                rejected.append(question_id)

        print(f"   ✅ Saved {len(scored)} questions")

        # Display results
        print(f"\n📊 EVALUATION RESULTS")
        print("-" * 100)
        print(f"Total evaluated: {len(scored)}")
        print(f"Passed (candidate):   {len(passed)} ✅")
        print(f"Rejected (auto):      {len(rejected)} ❌")
        print(f"Pass rate:            {len(passed) / len(scored) * 100 if scored else 0:.1f}%")

        print(f"\n🎯 DETAILED SCORES")
        print("-" * 100)

        for i, (q, score_obj) in enumerate(scored):
            status_emoji = "✅" if score_obj.pass_threshold else "❌"

            print(f"\n{i + 1}. [{score_obj.scores.clarity * 100:3.0f}% clear] {status_emoji}")
            print(f"   Q: {q['question_text'][:80]}...")
            print(f"   Level: {q['level'].upper()}, Type: {q['question_type']}")

            print(f"   " + print_score_bar(score_obj.scores.clarity, "Clarity"))
            print(f"   " + print_score_bar(score_obj.scores.relevance, "Relevance"))
            print(f"   " + print_score_bar(score_obj.scores.difficulty_match, "Difficulty Match"))
            print(f"   " + print_score_bar(score_obj.scores.groundedness, "Groundedness"))
            print(f"   " + print_score_bar(score_obj.scores.redundancy, "Redundancy"))
            print(f"   " + "=" * 76)
            print(f"   " + print_score_bar(score_obj.overall_score, "OVERALL"))

            print(f"   Feedback: {score_obj.feedback}")
            if score_obj.rejection_reason:
                print(f"   ⚠️  Rejection: {score_obj.rejection_reason}")

        # Show database queries
        print(f"\n🗄️  DATABASE QUERIES")
        print("-" * 100)

        # Get all questions
        all_q = list_questions_by_run(run_id, db_path=db_path)
        print(f"All questions in run: {len(all_q)}")

        # Get only passed
        candidates = list_questions_by_run(run_id, status="candidate", db_path=db_path)
        print(f"Candidate questions: {len(candidates)}")
        for q in candidates:
            print(f"  • [{q['level'].upper()}] {q['question_text'][:70]}... (score: {q['judge_overall_score']:.2f})")

        # Get only rejected
        auto_rejected = list_questions_by_run(run_id, status="rejected_auto", db_path=db_path)
        print(f"Auto-rejected questions: {len(auto_rejected)}")
        for q in auto_rejected:
            print(f"  • [{q['level'].upper()}] {q['question_text'][:70]}... (score: {q['judge_overall_score']:.2f})")

        # Show one full record
        print(f"\n📋 SAMPLE QUESTION RECORD (JSON)")
        print("-" * 100)
        if all_q:
            q_sample = all_q[0]
            # Convert to JSON-friendly format
            q_json = {
                "id": q_sample["id"],
                "question_text": q_sample["question_text"],
                "level": q_sample["level"],
                "skill": q_sample["skill"],
                "question_type": q_sample["question_type"],
                "judge_scores": {
                    "clarity": q_sample["judge_clarity_score"],
                    "relevance": q_sample["judge_relevance_score"],
                    "difficulty_match": q_sample["judge_difficulty_match_score"],
                    "groundedness": q_sample["judge_groundedness_score"],
                    "redundancy": q_sample["judge_redundancy_score"],
                    "overall": q_sample["judge_overall_score"],
                },
                "status": q_sample["status"],
                "created_at": q_sample["created_at"],
            }
            print(json.dumps(q_json, indent=2))

        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)

        print("\n" + "=" * 100)
        print("Live Demo Complete!")
        print("=" * 100)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(evaluate_questions_live())
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()