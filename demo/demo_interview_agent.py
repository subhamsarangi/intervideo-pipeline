"""
Demo: Interview Control Agent with Database Persistence

Shows agent making decisions on:
- Answer quality assessment  
- Follow-up generation
- Interview pacing (continue/probe/skip/exit)
- Session summary metrics
- Persistent storage to database
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interview.control_agent import InterviewControlAgent
from src.models.interview_agent import (
    InterviewSessionState,
    AnswerQuality,
    DecisionReason,
)
from src.database.db import init_db
from src.database.interview_db import (
    create_interview_session,
    list_session_answers,
    list_session_decisions,
    get_session_quality_distribution,
    get_session_by_level,
    get_session_decision_breakdown,
    get_session_fatigue_signals,
)


def print_banner(text: str):
    print("\n" + "=" * 100)
    print(text.center(100))
    print("=" * 100)


def print_section(text: str):
    print(f"\n{'─' * 100}")
    print(f"▶ {text}")
    print(f"{'─' * 100}")


async def demo():
    """Run interview agent demo with DB persistence"""
    
    print_banner("Interview Control Agent Demo (with DB Persistence)")
    
    # Setup database
    db_path = "demo_interview.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db(db_path)
    print(f"✅ Database initialized: {db_path}")
    
    # Initialize agent with DB path
    print_section("Initializing Agent")
    agent = InterviewControlAgent(db_path=db_path)
    print("   ✅ Agent initialized with database persistence")
    
    # Create session
    print_section("Creating Interview Session")
    session = InterviewSessionState(
        session_id="demo_session_001",
        run_id=1,
        total_questions=5,
    )
    
    # Save session to DB
    db_session_id = create_interview_session(
        session_id=session.session_id,
        run_id=session.run_id,
        total_questions=session.total_questions,
        planned_duration_seconds=session.planned_duration_seconds,
        db_path=db_path,
    )
    
    print(f"   ✅ Session created: {session.session_id}")
    print(f"   💾 Saved to DB (ID: {db_session_id})")
    print(f"   📋 Questions planned: {session.total_questions}")
    print(f"   ⏱️  Duration: {session.planned_duration_seconds // 60} minutes")
    
    # Demo Q&A scenarios
    test_cases = [
        {
            "q_id": 1,
            "q": "Describe how you would design a distributed cache system.",
            "level": "senior",
            "type": "design",
            "answer": "I would use Redis with a cluster mode for high availability. Key decisions: "
                     "1) Consistency model - eventual consistency for reads, 2) Eviction policy - LRU, "
                     "3) Replication - 2x replicas for fault tolerance. For network partitions, I'd route "
                     "writes to primary only. Monitoring via Prometheus for hit rates and latency.",
            "duration": 45,
        },
        {
            "q_id": 2,
            "q": "Tell me about a time you debugged a production issue.",
            "level": "mid",
            "type": "behavioral",
            "answer": "Um, there was this one time with the database. It was slow. We fixed it.",
            "duration": 15,
        },
        {
            "q_id": 3,
            "q": "Explain the difference between const and let in JavaScript.",
            "level": "junior",
            "type": "technical",
            "answer": "const is immutable but let is mutable. Const has block scope and let also has block scope. "
                     "You can reassign let but not const. Hoisting is different too.",
            "duration": 30,
        },
    ]
    
    # Process Q&A
    print_section("Interview Simulation")
    
    answer_ids = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📌 Question {i}/{len(test_cases)}")
        print(f"   Q ({test['level'].upper()}): {test['q'][:70]}...")
        
        # Assess answer
        print(f"\n   🔍 Assessing answer quality...")
        record = await agent.record_answer(
            session,
            test["q_id"],
            test["q"],
            test["level"],
            test["type"],
            test["answer"],
            test["duration"],
        )
        
        # Get the answer ID from DB (last inserted)
        answers = list_session_answers(session.session_id, db_path=db_path)
        last_answer = answers[-1] if answers else None
        answer_id = last_answer["id"] if last_answer else None
        answer_ids.append(answer_id)
        
        print(f"   ✓ Quality: {record.quality.value.upper()}")
        print(f"   💭 Reasoning: {record.feedback[:80]}...")
        print(f"   ⏱️  Duration: {record.duration_seconds}s")
        print(f"   📊 Session avg quality: {session.avg_answer_quality:.2f}")
        print(f"   💾 Saved to DB (Answer ID: {answer_id})")
        
        # Decide next action
        print(f"\n   🤔 Deciding next action...")
        decision = await agent.decide_next_action(
            session,
            record.quality,
            test["answer"],
            test["q"],
            answer_id,
        )
        
        print(f"   ✓ Action: {decision.action.value}")
        print(f"   📝 Reasoning: {decision.reasoning[:80]}...")
        print(f"   📈 Confidence: {decision.confidence:.1%}")
        print(f"   💾 Decision saved to DB")
        
        # Generate follow-up if needed
        if decision.action == DecisionReason.PROBE_DEEPER:
            print(f"\n   💬 Generating follow-up...")
            follow_up = await agent.generate_follow_up(test["q"], test["answer"])
            print(f"   ✓ Follow-up: {follow_up}")
            record.follow_ups_asked = 1
            record.follow_up_texts.append(follow_up)
    
    # Generate summary
    print_section("Interview Summary")
    
    metrics = await agent.generate_summary(session)
    
    print(f"\n📊 Results:")
    print(f"   Questions asked: {metrics.questions_asked}/{metrics.total_questions}")
    print(f"   Average quality: {metrics.avg_answer_quality:.2f}/1.0")
    print(f"   Total time: {metrics.total_duration_minutes:.1f} minutes")
    print(f"   Avg per question: {metrics.avg_question_duration_seconds}s")
    print(f"   💾 Summary saved to DB")
    
    print(f"\n📈 Quality Distribution:")
    for quality, count in metrics.quality_distribution.items():
        if count > 0:
            pct = (count / metrics.questions_asked) * 100 if metrics.questions_asked > 0 else 0
            bar = "█" * int(count * 4)
            print(f"   {quality.value:15s}: {bar} ({count}, {pct:.0f}%)")
    
    print(f"\n📚 Performance by Level:")
    for level, performance in metrics.performance_by_level.items():
        asked = metrics.questions_by_level.get(level, 0)
        if asked > 0:
            bar = "█" * int(performance * 20)
            print(f"   {level.upper():10s}: {bar} {performance:.2f} ({asked} questions)")
    
    print(f"\n🚪 Exit Info:")
    print(f"   Reason: {metrics.reason_for_exit.value}")
    print(f"   Fatigue detected: {'Yes ⚠️' if metrics.fatigue_detected else 'No ✅'}")
    
    # Retrieve and display data from database
    print_section("Database Verification")
    
    # Answers stored
    stored_answers = list_session_answers(session.session_id, db_path=db_path)
    print(f"\n💾 Stored Answers: {len(stored_answers)}")
    for i, ans in enumerate(stored_answers, 1):
        print(f"   {i}. [{ans['quality']}] {ans['question_text'][:60]}... ({ans['duration_seconds']}s)")
    
    # Decisions stored
    stored_decisions = list_session_decisions(session.session_id, db_path=db_path)
    print(f"\n💾 Stored Decisions: {len(stored_decisions)}")
    for i, dec in enumerate(stored_decisions, 1):
        print(f"   {i}. {dec['action']} (confidence: {dec['confidence']:.1%})")
    
    # Quality distribution
    quality_dist = get_session_quality_distribution(session.session_id, db_path=db_path)
    print(f"\n💾 Quality Distribution (from DB):")
    for quality, count in quality_dist.items():
        print(f"   {quality}: {count}")
    
    # By level
    by_level = get_session_by_level(session.session_id, db_path=db_path)
    print(f"\n💾 Performance by Level (from DB):")
    for level, stats in by_level.items():
        print(f"   {level.upper()}: {stats['questions']} questions, "
              f"{stats['strong_answers']} strong ({stats['strength_pct']:.0f}%)")
    
    # Decision breakdown
    decision_dist = get_session_decision_breakdown(session.session_id, db_path=db_path)
    print(f"\n💾 Decision Breakdown (from DB):")
    for action, count in decision_dist.items():
        print(f"   {action}: {count}")
    
    # Fatigue signals
    fatigue = get_session_fatigue_signals(session.session_id, db_path=db_path)
    print(f"\n💾 Fatigue Analysis (from DB):")
    print(f"   Detected: {fatigue['detected']}")
    print(f"   Poor answers (last 3): {fatigue['poor_answers_in_last_3']}")
    print(f"   Time decline: {fatigue['time_decline_pct']}%")
    if fatigue['signals']:
        print(f"   Signals: {', '.join([s for s in fatigue['signals'] if s])}")
    
    print_banner("Demo Complete!")
    print("\n✨ Agent successfully:")
    print("   ✓ Assessed answer quality")
    print("   ✓ Generated follow-ups")
    print("   ✓ Made pacing decisions")
    print("   ✓ Produced session summary")
    print("   ✓ Persisted all data to database")
    print(f"\n💾 Database file: {db_path}")
    print("\n")


if __name__ == "__main__":
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
