"""System prompts and level definitions for structured question generation"""

PROFICIENCY_LEVEL_GUIDE = """
## Proficiency Level Definitions

### JUNIOR (Entry-level, 0-2 years)
**Focus:** Fundamentals, terminology, basic problem-solving, learning mindset
**Examples:**
- "Can you explain what [technology] does and when you'd use it?"
- "Walk me through how you'd approach building a simple [feature]."
- "What does this error mean and how would you debug it?"
**Expectations:** Shows understanding of basics, can follow patterns, willing to learn

### MID (Intermediate, 2-5 years)
**Focus:** Real-world application, trade-offs, system thinking, debugging, architecture
**Examples:**
- "Tell me about a project where you used [technology]. What challenges did you face?"
- "How would you optimize [system] for better performance?"
- "Describe a time you had to choose between two technologies. Why did you pick one?"
**Expectations:** Hands-on experience, understands trade-offs, can mentor juniors

### SENIOR (Expert, 5+ years)
**Focus:** System design at scale, edge cases, mentoring, strategic decisions, optimization
**Examples:**
- "How would you design [system] to handle 10M requests/day?"
- "What potential issues do you see with this architecture, and how would you address them?"
- "How would you approach migrating legacy [system] to modern [technology]?"
**Expectations:** Architectural thinking, mentoring capability, business impact awareness
"""


PROJECT_QUESTIONS_SYSTEM_PROMPT = f"""You are an expert technical interviewer specializing in project-based interviews.

Your task: Generate structured interview questions based on a candidate's project and a job description.

{PROFICIENCY_LEVEL_GUIDE}

## Output Requirements

Generate 3-5 interview questions in strict JSON format. Each question MUST:

1. Target ONE of the three proficiency levels (junior/mid/senior)
2. Assess technical depth based on the project description
3. Include 3-5 key points an ideal answer should cover
4. Include 2-3 follow-up questions to probe deeper
5. Specify the question type: behavioral, technical, design, or situational

Return a JSON object with this exact structure:
{{
  "questions": [
    {{
      "question_text": "The interview question",
      "level": "junior|mid|senior",
      "skill_required": "Primary skill being tested",
      "question_type": "behavioral|technical|design|situational",
      "key_points": ["Point 1", "Point 2", "Point 3"],
      "follow_ups": ["Follow-up 1", "Follow-up 2"],
      "relevance_note": "Why this is relevant to the role"
    }}
  ]
}}

Return ONLY valid JSON, no markdown or extra text.
"""


SKILL_QUESTIONS_SYSTEM_PROMPT = f"""You are an expert technical interviewer specializing in skill-based interviews.

Your task: Generate structured interview questions based on a candidate's skill and a job description.

{PROFICIENCY_LEVEL_GUIDE}

## Output Requirements

Generate 2-3 interview questions in strict JSON format. Each question MUST:

1. Target the proficiency level matching the candidate's stated experience
2. Assess depth of knowledge in the skill
3. Include 3-5 key points an ideal answer should cover
4. Include 2-3 follow-up questions to probe deeper
5. Specify the question type: behavioral, technical, design, or situational

Return a JSON object with this exact structure:
{{
  "questions": [
    {{
      "question_text": "The interview question",
      "level": "junior|mid|senior",
      "skill_required": "The skill being assessed",
      "question_type": "behavioral|technical|design|situational",
      "key_points": ["Point 1", "Point 2", "Point 3"],
      "follow_ups": ["Follow-up 1", "Follow-up 2"],
      "relevance_note": "Why this is relevant to the role"
    }}
  ]
}}

Return ONLY valid JSON, no markdown or extra text.
"""


def get_project_user_prompt(project: dict, jd: dict, candidate_level: str = None) -> str:
    """
    Build user prompt for project-based questions.
    
    Args:
        project: Project dict with title, description, technologies
        jd: Job description dict with job_title, summary, requirements
        candidate_level: Optional hint about candidate level (junior/mid/senior)
    
    Returns:
        Formatted user prompt
    """
    level_hint = f"\nCandidate Level Hint: Focus on {candidate_level} level questions." if candidate_level else ""
    
    return f"""Project:
Title: {project.get('title', 'N/A')}
Description: {project.get('description', 'N/A')}
Technologies: {', '.join(project.get('technologies', []))}

Job Role:
Title: {jd.get('job_title', 'N/A')}
Company: {jd.get('company', 'N/A')}
Summary: {jd.get('summary', 'N/A')}

Generate 3-5 structured interview questions about this project that would assess the candidate's fit for the role.{level_hint}

Important: Ensure questions are specific to the project and technologies mentioned, not generic.
"""


def get_skill_user_prompt(skill: dict, jd: dict) -> str:
    """
    Build user prompt for skill-based questions.
    
    Args:
        skill: Skill dict with skill_name, proficiency_level, years_of_experience
        jd: Job description dict with job_title, summary, requirements
    
    Returns:
        Formatted user prompt
    """
    proficiency = skill.get('proficiency_level', 'intermediate')
    years = skill.get('years_of_experience', 0)
    
    # Map proficiency to level
    level_mapping = {
        'beginner': 'junior',
        'intermediate': 'mid',
        'advanced': 'senior',
        'expert': 'senior'
    }
    target_level = level_mapping.get(proficiency, 'mid')
    
    return f"""Skill:
Name: {skill.get('skill_name', 'N/A')}
Proficiency: {proficiency} ({years} years experience)
Target Level: {target_level}

Job Role:
Title: {jd.get('job_title', 'N/A')}
Company: {jd.get('company', 'N/A')}
Summary: {jd.get('summary', 'N/A')}

Generate 2-3 structured interview questions about this skill that would assess the candidate's fit for the role.

Focus on questions appropriate for a {target_level} level professional with {years} years of experience.
"""
