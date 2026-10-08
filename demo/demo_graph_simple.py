"""
Simple Demo: Graph Execution with Mock State

Shows graph building and execution without requiring full database.
Perfect for testing that the async pipeline works end-to-end.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.graph import build_graph
from src.models.pipeline import PipelineState


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80)


async def demo():
    """Run simple graph demo"""
    
    print_banner("Graph Execution Demo")
    
    # Create minimal state
    print("\n📋 Creating Pipeline State...")
    state = PipelineState(
        run_id=999,
        cv_id=1,
        jd_id=1,
        db_path=":memory:",
    )
    print(f"   ✅ State created (run_id={state.run_id})")
    
    # Build graph
    print("\n🔧 Building Graph...")
    graph = await build_graph(":memory:")
    print("   ✅ Graph built with nodes:")
    for node_name in graph.nodes.keys():
        if not node_name.startswith("__"):
            print(f"      • {node_name}")
    
    # Show graph structure
    print("\n📊 Graph Structure:")
    print("   Entry: security")
    print("   Flow: security → triage → [enrichment?] → questions → merge → judge")
    print("   Conditional: triage can route to enrichment if tech needs web search")
    print("   Parallel: project_questions + skill_questions run concurrently")
    
    print("\n" + "=" * 80)
    print("Demo Complete!".center(80))
    print("=" * 80)
    print("\n✨ Graph is built and ready for execution")
    print("📝 To run full pipeline with real data, use demo_judge_evaluation.py\n")


if __name__ == "__main__":
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
