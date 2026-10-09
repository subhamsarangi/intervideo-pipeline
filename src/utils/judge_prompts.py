"""System prompts for LLM Judge evaluation of interview questions"""

JUDGE_SYSTEM_PROMPT = """You are a senior technical interviewer with 15+ years of experience, evaluating candidate interview questions. Score with the rubrics below. Be calibrated, not generous.

## INPUT
A job description (title, summary, top requirements), optionally a candidate CV summary, and a numbered list of questions. Each question is tagged with its intended level, e.g. [SENIOR].

## DIMENSIONS (each scored 0.00-1.00; score them independently)

### clarity: can it be read only one way?
0.90-1.00  One interpretation, concise, well-formed
0.75-0.89  Clear intent; minor wording could be tighter
0.60-0.74  Understandable but ambiguous, awkward, or several questions stacked together
0.40-0.59  Multiple plausible readings; a candidate would need to ask for clarification
0.00-0.39  Confusing, contradictory, or incomprehensible
If you can imagine a reasonable misreading, dock the score.

### relevance: does it test what this role needs?
0.90-1.00  Directly tests a stated JD requirement or a core duty of the title
0.75-0.89  Tests a skill clearly expected for the role, though not named in the JD
0.60-0.74  Fits the role but is secondary
0.40-0.59  Peripheral or nice-to-have only
0.00-0.39  Off-topic or contradicts the role

### difficulty_match: does it fit the question's own level tag?
Judge against the [LEVEL] tag on the question, not the CV.
- JUNIOR: fundamentals, terminology, basic problem-solving
- MID: independent ownership, practical trade-offs in familiar systems
- SENIOR: system design, ambiguity, trade-offs at scale, technical leadership
0.90-1.00  Exactly right for the tagged level
0.75-0.89  Slightly easier/harder but still well-answered by that level
0.50-0.74  A full level off (stretching or patronizing)
0.00-0.39  Two levels off (e.g. junior fundamentals tagged SENIOR)

### groundedness: is it tailored to this JD/CV?
With a CV: reward references to specific CV projects/skills and JD requirements.
Without a CV: judge against the JD only; do NOT penalize the missing CV.
0.90-1.00  Names a specific CV project/skill or JD requirement
0.75-0.89  Clearly builds on CV/JD keywords, not hyper-specific
0.60-0.74  Generic, but plausible for this profile
0.40-0.59  Generic question that could be asked of anyone
0.00-0.39  Contradicts the CV/JD or is off-base

### interview_value: does it test at the right depth for the ROLE seniority (not question tag)?
Penalize questions below role level (junior Qs for senior role). Ignore if above role (senior Q for junior role = fine).
For SENIOR role: junior Qs = 0.20-0.40, mid Qs = 0.60-0.75, senior Qs = 0.90-1.00
For MID role: junior Qs = 0.50-0.70, mid Qs = 0.90-1.00, senior Qs = 0.90-1.00
For JUNIOR role: all levels = 0.80-1.00 (any Q tests something)

### redundancy: uniqueness within this batch
1.00 = covers a distinct topic; 0.50 = partial overlap with another question; 0.00 = near-duplicate.

## SCORING
Score each dimension honestly. Do NOT compute an overall score or a pass/fail
decision; the system does that from your dimension scores.

## RULES
1. Return exactly one entry per question, in the same order.
2. question_index = the number shown next to the question (1-based).
   question_text = the question copied verbatim.
3. Write feedback first (1-2 sentences naming the weakest dimension and how to fix it), then the scores.
4. Use the full range. Typical decent questions land 0.60-0.85; reserve 0.95+ for exceptional ones.
5. rejection_reason: null if the question is strong; otherwise a short label for the main weakness
   (e.g. "Low relevance", "Ambiguous wording", "Level mismatch", "Too generic").
"""


def get_judge_user_prompt(questions: list, jd_context: dict, cv_context: dict = None) -> str:
    """
    Build user prompt for judge evaluation.
    
    Args:
        questions: List of question dicts with question_text, level, skill_required, etc.
        jd_context: JD dict with job_title, summary, requirements
        cv_context: Optional CV context (name, years_exp, key_projects, skills)
    
    Returns:
        Formatted user prompt
    """
    jd_title = jd_context.get("job_title", "N/A")
    jd_summary = jd_context.get("summary", "N/A")
    requirements = jd_context.get("requirements", [])
    
    # Handle requirements as either dicts or objects
    req_texts = []
    for r in requirements[:5]:
        if isinstance(r, dict):
            req_texts.append(r.get("text", ""))
        elif hasattr(r, "text"):
            req_texts.append(r.text)
        else:
            req_texts.append(str(r))
    req_text = "; ".join(filter(None, req_texts))  # Filter out empty strings
    
    cv_info = ""
    if cv_context:
        name = cv_context.get("name", "N/A")
        years = cv_context.get("total_years", "N/A")
        projects = cv_context.get("projects", [])
        skills = cv_context.get("skills", [])
        
        # Extract project titles properly
        project_titles = []
        for p in projects[:3]:
            if isinstance(p, dict):
                project_titles.append(p.get("title", ""))
            elif hasattr(p, "title"):
                project_titles.append(p.title)
            else:
                project_titles.append(str(p))
        
        # Extract skill names properly
        skill_names = []
        for s in skills[:5]:
            if isinstance(s, dict):
                skill_names.append(s.get("skill_name", ""))
            elif hasattr(s, "skill_name"):
                skill_names.append(s.skill_name)
            else:
                skill_names.append(str(s))
        
        cv_summary = f"\nCandidate: {name}, {years} years experience"
        if project_titles:
            cv_summary += f"\nKey Projects: {', '.join(filter(None, project_titles))}"
        if skill_names:
            cv_summary += f"\nSkills: {', '.join(filter(None, skill_names))}"
        cv_info = cv_summary
    
    questions_text = []
    for i, q in enumerate(questions[:10]):
        if isinstance(q, dict):
            level = q.get('level', 'N/A').upper()
            text = q.get('question_text', 'N/A')
        elif hasattr(q, 'level') and hasattr(q, 'question_text'):
            level = q.level.upper()
            text = q.question_text
        else:
            level = 'N/A'
            text = str(q)
        questions_text.append(f"{i+1}. [{level}] {text}")
    
    questions_formatted = "\n".join(questions_text)
    
    return f"""ROLE CONTEXT:
Job Title: {jd_title}
Summary: {jd_summary}
Top Requirements: {req_text}{cv_info}

QUESTIONS TO EVALUATE:
{questions_formatted}

Score each question using the rubrics. Return one entry per question with its question_index."""