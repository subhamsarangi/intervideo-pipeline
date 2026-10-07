"""
Quick debug script: run a Tavily search and show exactly what the pipeline sees.
Usage: uv run python scripts/tavily_debug.py
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Reuse the pipeline helpers so output is identical to what enrichment_node produces
from src.pipeline.nodes.enrichment import _call_tavily, _build_search_query, _sanitize_and_wrap

TOPIC = "LLM-agent"


async def main():
    print("=" * 60)
    print(f"  TAVILY DEBUG — topic: '{TOPIC}'")
    print("=" * 60)

    query = _build_search_query(TOPIC, jd_title="AI Engineer")
    print(f"\n📡 Query sent to Tavily:\n  {query}\n")

    print("-" * 60)
    results = await _call_tavily(query)
    print(f"🔎 Raw results returned: {len(results)}\n")

    for i, r in enumerate(results, 1):
        print(f"  ── Result {i} ──────────────────────────────────")
        print(f"  Title  : {r.get('title', '(none)')}")
        print(f"  URL    : {r.get('url', '(none)')}")
        raw_content = r.get("content", r.get("snippet", ""))
        print(f"  Content: {raw_content[:300]}{'...' if len(raw_content) > 300 else ''}")
        print()

    print("=" * 60)
    print("  WHAT THE PIPELINE SEES (after sanitize + delimiter wrap)")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        url = r.get("url", "unknown")
        content = r.get("content", r.get("snippet", ""))
        if content:
            wrapped = _sanitize_and_wrap(content, url)
            print(f"\n  ── Snippet {i} ─────────────────────────────────")
            print(wrapped)

    print("\n" + "=" * 60)
    print("  MERGED CONTEXT BLOCK (all snippets joined)")
    print("=" * 60)

    snippets = []
    sources = []
    for r in results:
        url = r.get("url", "unknown")
        content = r.get("content", r.get("snippet", ""))
        if content:
            snippets.append(_sanitize_and_wrap(content, url))
            sources.append(url)

    merged = "\n\n".join(snippets)
    print(merged)

    print(f"\n✅ Sources used ({len(sources)}):")
    for s in sources:
        print(f"  • {s}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
