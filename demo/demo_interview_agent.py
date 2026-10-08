"""
Demo: Interview Control Agent

Shows agent making decisions on:
- Answer quality assessment  
- Follow-up generation
- Interview pacing (continue/probe/skip/exit)
- Session summary metrics
"""

import asyncio
import sys
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


def print_banner(text: str):
    print("\n" + "=" * 100)
    print(text.center(100))
    print("=" * 100)


def print_section(text: str):
    print(f"\n{'─' * 100}")
    print(f"▶ {text}")
    print(f"{'─' * 100}")


async def demo():
    """Run interview agent demo"""
    
    print_banner("Interview Control Agent Demo")
    
    # Initialize agent
    print_section("Initializing Agent")
    agent = InterviewControlAgent()
    print("   ✅ Agent initialized")
    
    # Create session
    print_section("Creating Interview Session")
    session = InterviewSessionState(
        session_id="demo_session_001",
        run_id=1,
        total_questions=5,
    )
    print(f"   ✅ Session created: {session.session_id}")
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
            "expect": AnswerQuality.EXCELLENT,
        },
        {
            "q_id": 2,
            "q": "Tell me about a time you debugged a production issue.",
            "level": "mid",
            "type": "behavioral",
            "answer": "Um, there was this one time with the database. It was slow. We fixed it.",
            "duration": 15,
            "expect": AnswerQuality.POOR,
        },
        {
            "q_id": 3,
            "q": "Explain the difference between const and let in JavaScript.",
            "level": "junior",
            "type": "technical",
            "answer": "const is immutable but let is mutable. Const has block scope and let also has block scope. "
                     "You can reassign let but not const. Hoisting is different too.",
            "duration": 30,
            "expect": AnswerQuality.GOOD,
        },
    ]
    
    # Process Q&A
    print_section("Interview Simulation")
    
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
        
        print(f"   ✓ Quality: {record.quality.value.upper()}")
        print(f"   💭 Reasoning: {record.feedback[:80]}...")
        print(f"   ⏱️  Duration: {record.duration_seconds}s")
        print(f"   📊 Session avg quality: {session.avg_answer_quality:.2f}")
        
        # Decide next action
        print(f"\n   🤔 Deciding next action...")
        decision = await agent.decide_next_action(
            session,
            record.quality,
            test["answer"],
            test["q"],
        )
        
        print(f"   ✓ Action: {decision.action.value}")
        print(f"   📝 Reasoning: {decision.reasoning[:80]}...")
        print(f"   📈 Confidence: {decision.confidence:.1%}")
        
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
    
    print_banner("Demo Complete!")
    print("\n✨ Agent successfully:")
    print("   ✓ Assessed answer quality")
    print("   ✓ Generated follow-ups")
    print("   ✓ Made pacing decisions")
    print("   ✓ Produced session summary\n")


if __name__ == "__main__":
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
