"""
Live Demo: Full Pipeline Execution with Real Data

Demonstrates end-to-end pipeline flow:
1. Security validation
2. Triage & ranking
3. Question generation (project + skill)
4. Merge & deduplication
5. Judge evaluation
6. Save to database

Shows visual progress and outputs results to terminal and file.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.graph import build_graph, run_pipeline
from src.models.pipeline import PipelineState
from src.database.db import init_db, get_cv, get_jd, list_questions_by_run

load_dotenv()


def print_banner(text: str, char: str = "="):
    """Print a banner with text"""
    width = 100
    print("\n" + char * width)
    print(text.center(width))
    print(char * width)


def print_section(text: str):
    """Print a section header"""
    print(f"\n{'─' * 100}")
    print(f"▶ {text}")
    print(f"{'─' * 100}")


async def demo_full_pipeline():
    """Execute full pipeline with real CV/JD data"""
    
    print_banner("LIVE DEMO: Full Pipeline Execution", "=")
    
    # Setup database - use mock data inline
    db_path = ":memory:"  # Use in-memory DB for demo
    
    print_section("Database Setup")
    print(f"📁 Using in-memory database with mock data")
    init_db(db_path)
    
    # Create mock CV data inline
    import sqlite3
    import json
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Insert mock CV
    cursor.execute("""
        INSERT INTO cv_documents (id, title, candidate_name, raw_text, total_years_experience, projects, skills, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        1,
        "Senior Backend Engineer CV",
        "Priya Sharma",
        "Mock CV content",
        6,
        json.dumps([
            {"name": "E-commerce Platform", "tech_stack": ["Python", "Django", "PostgreSQL"]},
            {"name": "Payment Gateway", "tech_stack": ["Go", "Redis", "Kubernetes"]}
        ]),
        json.dumps([
            {"name": "Python", "years": 6},
            {"name": "Distributed Systems", "years": 4},
            {"name": "PostgreSQL", "years": 5}
        ]),
        "completed"
    ))
    
    # Insert mock JD
    cursor.execute("""
        INSERT INTO jd_documents (id, title, job_title, company, required_skills, raw_text, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        1,
        "Senior Backend Engineer Role",
        "Senior Backend Engineer",
        "TechCorp",
        json.dumps([
            {"name": "Python", "required": True},
            {"name": "Distributed Systems", "required": True},
            {"name": "System Design", "required": True}
        ]),
        "Mock JD content",
        "completed"
    ))
    
    conn.commit()
    cursor.execute("SELECT * FROM cv_documents LIMIT 1")
    cv_row = cursor.fetchone()
    
    if not cv_row:
        print("   ❌ No CVs found in database. Run: python scripts/seed_dummy_data.py")
        conn.close()
        return
    
    cv = dict(cv_row)
    cv_id = cv['id']
    
    # Parse JSON fields
    import json
    if cv.get('projects'):
        cv['projects'] = json.loads(cv['projects']) if isinstance(cv['projects'], str) else cv['projects']
    if cv.get('skills'):
        cv['skills'] = json.loads(cv['skills']) if isinstance(cv['skills'], str) else cv['skills']
    
    # Get first JD
    cursor.execute("SELECT * FROM jd_documents WHERE id = 1")
    jd_row = cursor.fetchone()
    
    if not jd_row:
        print("   ❌ No JDs found in database. Run: python scripts/seed_dummy_data.py")
        conn.close()
        return
    
    jd = dict(jd_row)
    jd_id = jd['id']
    
    # Parse JSON fields
    if jd.get('required_skills'):
        jd['required_skills'] = json.loads(jd['required_skills']) if isinstance(jd['required_skills'], str) else jd['required_skills']
    if jd.get('nice_to_have_skills'):
        jd['nice_to_have_skills'] = json.loads(jd['nice_to_have_skills']) if isinstance(jd['nice_to_have_skills'], str) else jd['nice_to_have_skills']
    
    conn.close()
    
    print(f"   📄 CV: {cv['candidate_name']}")
    print(f"      Years of Experience: {cv['total_years_experience']}")
    print(f"      Projects: {len(cv.get('projects', []))}")
    print(f"      Skills: {len(cv.get('skills', []))}")
    
    print(f"\n   📋 JD: {jd['job_title']}")
    print(f"      Company: {jd.get('company', 'N/A')}")
    print(f"      Required Skills: {len(jd.get('required_skills', []))}")
    
    # Create pipeline state
    run_id = 1
    state = PipelineState(
        run_id=run_id,
        cv_id=cv_id,
        jd_id=jd_id,
        db_path=db_path,
    )
    
    print_section("Pipeline Execution")
    print(f"   🚀 Starting pipeline for run_id={run_id}")
    print(f"   📊 Thread ID: run_{run_id}")
    print(f"   ⏱️  Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        # Build and execute pipeline
        print("\n   Building graph with checkpointing...")
        graph = await build_graph(db_path)
        print("   ✅ Graph built successfully")
        
        print("\n   Executing pipeline nodes:")
        print("      1️⃣  Security validation")
        print("      2️⃣  Triage & ranking")
        print("      3️⃣  Question generation (project + skill)")
        print("      4️⃣  Merge & deduplication")
        print("      5️⃣  Judge evaluation")
        
        # Run pipeline
        final_state = await run_pipeline(state, db_path)
        
        print(f"\n   ✅ Pipeline completed at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Display results
        print_section("Pipeline Results")
        
        # Get questions from database
        questions = list_questions_by_run(run_id, db_path=db_path)
        
        print(f"📊 Total Questions Generated: {len(questions)}")
        
        # Group by status
        candidate = [q for q in questions if q['status'] == 'candidate']
        rejected = [q for q in questions if q['status'] == 'rejected_auto']
        
        print(f"\n✅ Candidate Questions: {len(candidate)}")
        print(f"❌ Auto-Rejected: {len(rejected)}")
        
        if len(questions) > 0:
            pass_rate = (len(candidate) / len(questions)) * 100
            print(f"📈 Pass Rate: {pass_rate:.1f}%")
        
        # Show candidate questions by level
        print_section("Candidate Questions by Level")
        
        levels = {}
        for q in candidate:
            level = q['level'].upper()
            if level not in levels:
                levels[level] = []
            levels[level].append(q)
        
        for level in ['JUNIOR', 'MID', 'SENIOR']:
            if level in levels:
                qs = levels[level]
                print(f"\n{level} ({len(qs)} questions):")
                for i, q in enumerate(qs[:3], 1):  # Show first 3
                    score = q['judge_overall_score']
                    q_type = q['question_type'].upper()
                    print(f"   {i}. [{score:.2f}] [{q_type}]")
                    print(f"      {q['question_text'][:80]}...")
                
                if len(qs) > 3:
                    print(f"   ... and {len(qs) - 3} more")
        
        # Show rejected questions
        if rejected:
            print_section("Auto-Rejected Questions")
            print(f"\nTop rejection reasons:\n")
            for i, q in enumerate(rejected[:3], 1):
                score = q['judge_overall_score']
                reason = q.get('judge_feedback', 'No feedback')[:60]
                print(f"   {i}. [{score:.2f}] {q['question_text'][:60]}...")
                print(f"      Reason: {reason}...")
        
        # Show score distribution
        print_section("Score Distribution")
        
        if candidate:
            scores = [q['judge_overall_score'] for q in candidate]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            
            print(f"\n   Average Score: {avg_score:.2f}")
            print(f"   Min Score:     {min_score:.2f}")
            print(f"   Max Score:     {max_score:.2f}")
            
            # Show histogram
            print("\n   Score Distribution:")
            bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            bin_labels = ["0.0-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
            
            for i in range(len(bins) - 1):
                count = len([s for s in scores if bins[i] <= s < bins[i+1]])
                bar = "█" * count
                print(f"   {bin_labels[i]}: {bar} ({count})")
        
        # Save output to file
        output_file = f"demo_pipeline_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        print_section("Saving Output")
        
        with open(output_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("FULL PIPELINE EXECUTION RESULTS\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"Run ID: {run_id}\n")
            f.write(f"CV: {cv['candidate_name']} ({cv['total_years_experience']} years)\n")
            f.write(f"JD: {jd['job_title']}\n")
            f.write(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Total Questions: {len(questions)}\n")
            f.write(f"Candidate: {len(candidate)}\n")
            f.write(f"Rejected: {len(rejected)}\n\n")
            
            f.write("CANDIDATE QUESTIONS:\n")
            f.write("-" * 100 + "\n\n")
            
            for i, q in enumerate(candidate, 1):
                f.write(f"{i}. [{q['level'].upper()}] [{q['question_type'].upper()}] Score: {q['judge_overall_score']:.2f}\n")
                f.write(f"   Q: {q['question_text']}\n")
                if q.get('follow_up_questions'):
                    f.write(f"   Follow-ups: {', '.join(q['follow_up_questions'])}\n")
                f.write(f"   Key Points: {', '.join(q.get('key_points', []))}\n")
                f.write(f"   Source: {q['source_type']} - {q.get('source_id', 'N/A')}\n")
                f.write(f"   Judge Scores: Clarity={q['judge_clarity_score']:.2f}, "
                       f"Relevance={q['judge_relevance_score']:.2f}, "
                       f"Groundedness={q['judge_groundedness_score']:.2f}\n")
                f.write(f"   Feedback: {q.get('judge_feedback', 'N/A')}\n\n")
        
        print(f"   💾 Output saved to: {output_file}")
        print(f"   📊 Database: {db_path}")
        
        print_banner("Demo Complete!", "=")
        print(f"\n✨ Pipeline executed successfully!")
        print(f"📂 Check '{output_file}' for detailed results")
        print(f"🗄️  Database '{db_path}' contains all question data\n")
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(demo_full_pipeline())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
