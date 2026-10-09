#!/usr/bin/env python3
"""
End-to-end pipeline runner with checkpoint system:
1. Parse uploads/cv.pdf (cached)
2. Parse uploads/jd.pdf (cached)
3. Run full question generation pipeline
4. Display results + cost tracking

Checkpoints use file hashes to detect changes automatically.
"""

import asyncio
import sys
import json
import hashlib
from pathlib import Path
from typing import Optional

from src.database.db import (
    create_cv,
    create_jd,
    create_pipeline_run,
    list_questions_by_run,
    get_cv_cost,
    get_jd_cost,
    get_cv,
    get_jd,
)
from src.parsers.cv_parser import parse_cv
from src.parsers.jd_parser import parse_jd, auto_generate_jd_title
from src.pipeline.graph import build_graph, run_pipeline
from src.utils.llm_tracking import get_tracker, reset_tracker


CACHE_DIR = Path(".pipeline_cache")


def file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_checkpoint(name: str, file_path: Path, record_id: int):
    """Save checkpoint with file hash"""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{name}.json"
    data = {
        "id": record_id,
        "file_hash": file_hash(file_path),
        "file_name": file_path.name
    }
    cache_file.write_text(json.dumps(data, indent=2))
    print(f"   💾 Checkpoint saved: {name}")


def load_checkpoint(name: str, file_path: Path) -> Optional[int]:
    """Load checkpoint if file hash matches. If DB record missing, clear cache to force reparse."""
    cache_file = CACHE_DIR / f"{name}.json"
    if not cache_file.exists():
        return None
    
    cached = json.loads(cache_file.read_text())
    current_hash = file_hash(file_path)
    
    if cached.get("file_hash") != current_hash:
        print(f"   ⚠️  File changed, cache invalid")
        return None
    
    # Verify ID exists in DB
    record_id = cached.get("id")
    if name == "cv_parsed":
        record = get_cv(record_id)
    else:
        record = get_jd(record_id)
    
    if not record:
        print(f"   ⚠️  Database record missing, clearing cache (will reparse)")
        cache_file.unlink()  # Delete cache file to force reparse
        return None
    
    return record_id


def print_step(step: int, title: str):
    """Print step header"""
    print(f"\n{'─' * 80}")
    print(f"STEP {step}: {title}")
    print(f"{'─' * 80}")


def print_error(error_msg: str, detail: Optional[str] = None):
    """Print error with optional detail"""
    print(f"\n❌ ERROR: {error_msg}")
    if detail:
        print(f"   {detail}")
    print()


def print_progress(msg: str):
    """Print progress indicator"""
    print(f"   ⏳ {msg}...", flush=True)


def print_cost_summary(cv_id: int, jd_id: int):
    """Print LLM usage and cost breakdown"""
    print(f"\n{'─' * 80}")
    print("COST TRACKING & LLM USAGE")
    print(f"{'─' * 80}")
    
    cv_cost = get_cv_cost(cv_id)
    jd_cost = get_jd_cost(jd_id)
    tracker = get_tracker()
    pipeline_cost = tracker.calculate_cost()
    
    print(f"\n📄 CV Parsing:")
    print(f"   Calls: {cv_cost['llm_calls_count']} | Tokens: {cv_cost['total_input_tokens']}+{cv_cost['total_output_tokens']} | Cost: ${cv_cost['total_cost_usd']:.6f}")
    
    print(f"\n📄 JD Parsing:")
    print(f"   Calls: {jd_cost['llm_calls_count']} | Tokens: {jd_cost['total_input_tokens']}+{jd_cost['total_output_tokens']} | Cost: ${jd_cost['total_cost_usd']:.6f}")
    
    print(f"\n⚙️  Pipeline:")
    # Handle pipeline_cost which might have different key names
    pipeline_calls = pipeline_cost.get('llm_calls_count') or pipeline_cost.get('call_count') or 0
    pipeline_input = pipeline_cost.get('total_input_tokens') or pipeline_cost.get('input_tokens') or 0
    pipeline_output = pipeline_cost.get('total_output_tokens') or pipeline_cost.get('output_tokens') or 0
    pipeline_usd = pipeline_cost.get('total_cost_usd') or pipeline_cost.get('cost_usd') or 0.0
    
    print(f"   Calls: {pipeline_calls} | Tokens: {pipeline_input}+{pipeline_output} | Cost: ${pipeline_usd:.6f}")
    
    total_cost = cv_cost['total_cost_usd'] + jd_cost['total_cost_usd'] + pipeline_usd
    total_calls = cv_cost['llm_calls_count'] + jd_cost['llm_calls_count'] + pipeline_calls
    total_input = cv_cost['total_input_tokens'] + jd_cost['total_input_tokens'] + pipeline_input
    total_output = cv_cost['total_output_tokens'] + jd_cost['total_output_tokens'] + pipeline_output
    
    print(f"\n💰 TOTAL: {total_calls} calls | {total_input}+{total_output} tokens | ${total_cost:.6f}")
    print(f"   1000 sessions: ~${total_cost * 1000:.2f}")


async def main():
    print("=" * 80)
    print("FULL PIPELINE EXECUTION: CV + JD → Questions")
    print("=" * 80)
    print("\n💾 Smart checkpoint system: file hashes + DB validation")
    print("   Delete .pipeline_cache/ to force re-parse\n")
    
    reset_tracker()
    
    cv_path = Path("uploads/cv.pdf")
    jd_path = Path("uploads/jd.pdf")
    
    if not cv_path.exists():
        print_error(f"CV not found: {cv_path.absolute()}")
        return 1
    
    if not jd_path.exists():
        print_error(f"JD not found: {jd_path.absolute()}")
        return 1
    
    print(f"📄 CV: {cv_path}")
    print(f"📄 JD: {jd_path}")
    
    cv_id = None
    jd_id = None
    run_id = None
    
    # Step 1: Parse CV
    print_step(1, "Parsing CV PDF")
    
    cv_id = load_checkpoint("cv_parsed", cv_path)
    if cv_id:
        print(f"   ✓ Using cached CV (ID: {cv_id})")
        cv_data = get_cv(cv_id)
        if cv_data and cv_data.get('parsed_data'):
            parsed_data = cv_data['parsed_data']
            if isinstance(parsed_data, str):
                parsed_data = json.loads(parsed_data)
            print(f"      {parsed_data.get('name', 'N/A')} | {parsed_data.get('experience_years', 'N/A')}y | {len(parsed_data.get('projects', []))}p | {len(parsed_data.get('skills', []))}s")
    else:
        try:
            cv_id = create_cv(title=cv_path.name, raw_text="")
            print(f"\n   Created CV record: {cv_id}")
            
            print_progress("LlamaParse + extract (cost: $0.001-0.003)")
            parsed_cv = await parse_cv(str(cv_path), cv_id=cv_id)
            
            print(f"\n   ✓ CV parsed: {parsed_cv.name or 'N/A'} | {parsed_cv.experience_years or 'N/A'}y | {len(parsed_cv.projects)}p | {len(parsed_cv.skills)}s")
            
            save_checkpoint("cv_parsed", cv_path, cv_id)
            
        except Exception as e:
            print_error("CV parsing failed", str(e))
            return 1
    
    # Step 2: Parse JD
    print_step(2, "Parsing JD PDF")
    
    jd_id = load_checkpoint("jd_parsed", jd_path)
    if jd_id:
        print(f"   ✓ Using cached JD (ID: {jd_id})")
        jd_data = get_jd(jd_id)
        if jd_data and jd_data.get('parsed_data'):
            parsed_data = jd_data['parsed_data']
            if isinstance(parsed_data, str):
                parsed_data = json.loads(parsed_data)
            title = parsed_data.get('job_title', 'N/A')
            reqs = parsed_data.get('requirements', [])
            must_have = [r for r in reqs if r.get('requirement_type') == 'must_have']
            print(f"      {title} | {len(must_have)} must-have | {len(reqs) - len(must_have)} nice-to-have")
    else:
        try:
            jd_id = create_jd(title=jd_path.name, raw_text="", source_type="file", source_url=str(jd_path))
            print(f"   Created JD record: {jd_id}")
            
            print_progress("LlamaParse + extract (cost: $0.0001-0.002)")
            parsed_jd = await parse_jd(source_type="pdf", source_data=str(jd_path), jd_id=jd_id)
            
            must_have = [r for r in parsed_jd.requirements if r.requirement_type == 'must_have']
            nice_to_have = [r for r in parsed_jd.requirements if r.requirement_type == 'nice_to_have']
            
            print(f"\n   ✓ JD parsed: {parsed_jd.job_title or 'N/A'} | {len(must_have)} must-have | {len(nice_to_have)} nice-to-have")
            
            save_checkpoint("jd_parsed", jd_path, jd_id)
            
        except Exception as e:
            print_error("JD parsing failed", str(e))
            return 1
    
    # Step 3: Run pipeline
    try:
        print_step(3, "Running Pipeline (7 nodes: security→triage→enrichment→project_questions+skill_questions→merge_dedupe→judge)")
        run_id = create_pipeline_run(cv_id, jd_id)
        print(f"   Run ID: {run_id}\n   Processing:")
        
        from src.database.db import get_db_path
        from src.models.pipeline import PipelineState
        
        db_path = get_db_path()
        state = PipelineState(run_id=run_id, cv_id=cv_id, jd_id=jd_id)
        
        result = await run_pipeline(state=state, db_path=db_path)
        
        print(f"\n   ✓ Pipeline complete")
        
    except Exception as e:
        print_error("Pipeline failed", str(e))
        return 1
    
    # Step 4: Results
    try:
        print_step(4, "Results")
        
        questions = list_questions_by_run(run_id, status="candidate")
        
        if not questions:
            print("\n   ⚠️  No questions generated")
            return 1
        
        print(f"\n   ✓ {len(questions)} questions\n")
        
        by_level = {"junior": [], "mid": [], "senior": []}
        for q in questions:
            level = q.get("level") or "mid"  # Use "mid" if level is None
            by_level[level].append(q)
        
        print(f"   Level distribution: {len(by_level['junior'])} junior | {len(by_level['mid'])} mid | {len(by_level['senior'])} senior\n")
        
        for i, q in enumerate(questions[:3], 1):
            level = (q.get('level') or 'mid').upper()
            print(f"   {i}. [{level}] {q['question_text']}")
        
        if len(questions) > 3:
            print(f"   ... +{len(questions) - 3} more\n")
        
        print_cost_summary(cv_id, jd_id)
        
        print(f"\n{'='*80}")
        print(f"✓ COMPLETE | Run: {run_id} | CV: {cv_id} | JD: {jd_id}")
        print(f"GET http://localhost:8000/pipeline/{run_id}/questions")
        print(f"{'='*80}\n")
        
        return 0
        
    except Exception as e:
        import traceback
        print_error("Results failed", f"{type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
