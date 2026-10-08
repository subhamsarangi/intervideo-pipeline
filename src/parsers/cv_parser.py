"""CV Parser using LlamaParse + OpenAI structured extraction with Instructor"""

import asyncio
import json
import os
import re
from typing import Optional, List
from dotenv import load_dotenv
from llama_cloud import AsyncLlamaCloud
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field

from src.models.cv import ParsedCV, CVProject, CVSkill
from src.utils.llm_tracking import increment_llm_calls, get_tracker, reset_tracker

# Load environment variables
load_dotenv()


def get_llama_api_key():
    """Get LlamaCloud API key"""
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY not set in .env")
    return api_key


def get_openai_client():
    """Get OpenAI client with Instructor"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    client = OpenAI(api_key=api_key)
    # Patch the client with Instructor for structured output
    return instructor.from_openai(client)


async def parse_cv_file(file_path: str) -> str:
    """Upload CV PDF to LlamaParse and get markdown text"""
    api_key = get_llama_api_key()
    client = AsyncLlamaCloud(api_key=api_key)

    try:
        file_obj = await client.files.create(file=file_path, purpose="parse")

        result = await client.parsing.parse(
            file_id=file_obj.id, tier="agentic", version="latest", expand=["markdown"]
        )

        if result.markdown and result.markdown.pages:
            md_text = "\n\n".join(p.markdown for p in result.markdown.pages)
        else:
            md_text = ""

        increment_llm_calls(
            operation="llamaparse_upload_and_parse",
            model="llamaparse",
            input_tokens=0,
            output_tokens=0,
        )

        return md_text

    except Exception as e:
        raise Exception(f"LlamaParse failed: {str(e)}")


def extract_cv_structure(raw_text: str) -> ParsedCV:
    """Use OpenAI with Instructor to extract CV data with per-experience project extraction"""
    client = get_openai_client()

    # Phase 1: Extract basic info and work experiences (without projects)
    class WorkExperience(BaseModel):
        """Work experience entry"""

        company: str
        role: str
        duration: Optional[str] = None
        description: str
        technologies: List[str] = Field(
            default_factory=list, description="Technologies used"
        )

    class Phase1Data(BaseModel):
        """Phase 1: Basic info and work experience"""

        name: Optional[str] = None
        email: Optional[str] = None
        phone: Optional[str] = None
        summary: Optional[str] = None
        experience_years: Optional[float] = None
        work_experiences: List[WorkExperience] = Field(default_factory=list)

    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        response_model=Phase1Data,
        messages=[
            {
                "role": "system",
                "content": """Extract basic CV information and work experiences.

For each work experience:
- Extract company, role, duration, description
- Extract technologies used
- Keep the full description for later project extraction""",
            },
            {"role": "user", "content": f"Extract CV information:\n\n{raw_text}"},
        ],
        temperature=0.1,
    )
    phase1 = response
    # Get usage from the raw response
    raw_response = response._raw_response
    increment_llm_calls(
        "extract_cv_phase1",
        input_tokens=raw_response.usage.prompt_tokens,
        output_tokens=raw_response.usage.completion_tokens,
    )

    # Phase 2: For each work experience, extract projects from its description
    all_projects = []
    for exp in phase1.work_experiences:
        # Call LLM for each work experience to extract projects
        class ProjectsFromExp(BaseModel):
            """Projects extracted from a single work experience"""

            projects: List[CVProject] = Field(default_factory=list)

        response = client.chat.completions.create(
            model="gpt-5.6-luna",
            response_model=ProjectsFromExp,
            messages=[
                {
                    "role": "system",
                    "content": """Extract all PROJECTS mentioned in this work experience description.

CRITICAL: Avoid splitting a single project into multiple entries.

A project is ONE complete achievement:
- A feature/system built (even if it mentions multiple parts, it's ONE project)
- A product shipped (single product = one entry)
- A significant accomplishment (keep related work together)
- An initiative led (one initiative = one entry)

RULES:
1. NEVER split "X and Y as part of the same system" into separate projects
2. If description is about ONE main achievement, create ONE project entry
3. Only create separate projects if they are CLEARLY distinct efforts
4. When uncertain, keep related work as a SINGLE project
5. Each project should have a clear, complete title describing the whole achievement
6. Include all related accomplishments in one description

Example: "Built user auth and payment for e-commerce platform" = ONE project (related features)
Example: "Led rewrite of legacy system and migrated data" = ONE project (single effort)
Example: "Built auth system" AND separately "Led data migration project" = TWO projects (distinct)

If no explicit projects are mentioned, create ONE project from the overall role description.""",
                },
                {
                    "role": "user",
                    "content": f"Extract projects from this work experience:\n\nCompany: {exp.company}\nRole: {exp.role}\nDuration: {exp.duration}\n\nDescription:\n{exp.description}",
                },
            ],
            temperature=0.1,
        )
        exp_projects = response
        # Get usage from the raw response
        raw_response = response._raw_response
        increment_llm_calls(
            "extract_cv_experience_projects",
            input_tokens=raw_response.usage.prompt_tokens,
            output_tokens=raw_response.usage.completion_tokens,
        )

        # Add extracted projects, inheriting tech stack if not specified
        for proj in exp_projects.projects:
            if not proj.technologies and exp.technologies:
                proj.technologies = exp.technologies
            all_projects.append(proj)

    # Phase 3: Extract standalone projects and skills
    class Phase3Data(BaseModel):
        """Phase 3: Standalone projects and skills"""

        standalone_projects: List[CVProject] = Field(
            default_factory=list, description="Projects NOT in work experience"
        )
        skills: List[CVSkill] = Field(default_factory=list)

    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        response_model=Phase3Data,
        messages=[
            {
                "role": "system",
                "content": """Extract standalone projects and skills from the CV.

Standalone projects: Projects listed separately from work experience (e.g., in a "Projects" section, side projects, personal projects)
Skills: Extract all skills with proficiency and years if mentioned.""",
            },
            {
                "role": "user",
                "content": f"Extract standalone projects and skills:\n\n{raw_text}",
            },
        ],
        temperature=0.1,
    )
    phase3 = response
    # Get usage from the raw response
    raw_response = response._raw_response
    increment_llm_calls(
        "extract_cv_phase3_standalone",
        input_tokens=raw_response.usage.prompt_tokens,
        output_tokens=raw_response.usage.completion_tokens,
    )

    # Merge all projects and skills
    all_projects.extend(phase3.standalone_projects)

    # Deduplicate projects to reduce splitting
    all_projects = deduplicate_projects(all_projects, client)

    # Build final ParsedCV
    return ParsedCV(
        name=phase1.name,
        email=phase1.email,
        phone=phase1.phone,
        summary=phase1.summary,
        experience_years=phase1.experience_years,
        projects=all_projects,
        skills=phase3.skills,
    )


def deduplicate_projects(projects: List[CVProject], client) -> List[CVProject]:
    """Deduplicate and merge similar projects using LLM"""
    if len(projects) <= 1:
        return projects

    # Ask LLM to identify duplicates and merge them
    class MergedProject(BaseModel):
        """Merged project entry"""

        original_indices: List[int] = Field(
            ..., description="Indices of merged projects (0-based)"
        )
        merged_title: str
        merged_description: str
        merged_technologies: List[str] = Field(default_factory=list)

    class DeduplicationResult(BaseModel):
        """Result of deduplication"""

        projects_to_keep: List[int] = Field(
            ..., description="Indices of unique projects to keep"
        )
        merged_projects: List[MergedProject] = Field(
            default_factory=list, description="Projects that should be merged"
        )

    # Format projects for the LLM
    projects_text = "\n".join(
        [
            f"{i}. Title: {p.title}\n   Description: {p.description[:200]}...\n   Tech: {', '.join(p.technologies)}"
            for i, p in enumerate(projects)
        ]
    )

    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        response_model=DeduplicationResult,
        messages=[
            {
                "role": "system",
                "content": """Analyze these projects and identify duplicates or near-duplicates that should be merged.

Return:
1. projects_to_keep: Indices of projects that are UNIQUE
2. merged_projects: Groups of projects that are duplicates/should be merged

A duplicate is: same project described differently, or one project split into multiple entries.
Merge them into a single entry with combined info.""",
            },
            {
                "role": "user",
                "content": f"Deduplicate these projects:\n\n{projects_text}",
            },
        ],
        temperature=0.1,
    )
    dedup = response
    # Get usage from the raw response
    raw_response = response._raw_response
    increment_llm_calls(
        "deduplicate_cv_projects",
        input_tokens=raw_response.usage.prompt_tokens,
        output_tokens=raw_response.usage.completion_tokens,
    )

    # Build deduplicated list
    result = []

    # Add unique projects
    for idx in dedup.projects_to_keep:
        if 0 <= idx < len(projects):
            result.append(projects[idx])

    # Add merged projects
    for merged in dedup.merged_projects:
        if merged.original_indices and all(
            0 <= i < len(projects) for i in merged.original_indices
        ):
            # Create merged project from first original
            original = projects[merged.original_indices[0]]
            merged_proj = CVProject(
                title=merged.merged_title or original.title,
                description=merged.merged_description or original.description,
                technologies=merged.merged_technologies or original.technologies,
                duration=original.duration,
            )
            result.append(merged_proj)

    return result if result else projects


async def parse_cv(
    file_path: str, cv_id: Optional[int] = None, db_path: Optional[str] = None
) -> ParsedCV:
    """Complete CV parsing pipeline with tracking persistence"""
    reset_tracker()

    raw_text = await parse_cv_file(file_path)

    if not raw_text or len(raw_text.strip()) == 0:
        raise ValueError("No text extracted from CV")

    parsed_cv = extract_cv_structure(raw_text)

    # Save parsed content to outputs folder
    save_cv_to_markdown(parsed_cv, file_path)

    # Save tracking data to database if cv_id provided
    if cv_id is not None:
        tracker = get_tracker()
        cost_info = tracker.calculate_cost()

        from src.database.db import update_cv_status

        update_cv_status(
            cv_id=cv_id,
            status="completed",
            parsed_data=parsed_cv.model_dump(),
            llm_calls_count=tracker.get_count(),
            llm_calls_log=tracker.get_calls(),
            total_input_tokens=cost_info["total_input_tokens"],
            total_output_tokens=cost_info["total_output_tokens"],
            total_cost_usd=cost_info["total_cost_usd"],
            db_path=db_path,
        )

    return parsed_cv


def save_cv_to_markdown(parsed_cv: ParsedCV, original_file_path: str):
    """Save parsed CV content as markdown file in outputs folder"""
    from pathlib import Path
    from datetime import datetime

    # Create outputs folder if it doesn't exist
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    # Generate filename from original file or CV name
    original_name = Path(original_file_path).stem
    safe_name = parsed_cv.name.replace(" ", "_") if parsed_cv.name else original_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cv_{safe_name}_{timestamp}.md"
    output_path = outputs_dir / filename

    # Build markdown content
    md_lines = [
        f"# CV: {parsed_cv.name or 'Unnamed'}",
        "",
        "## Contact Information",
        f"- **Name:** {parsed_cv.name or 'N/A'}",
        f"- **Email:** {parsed_cv.email or 'N/A'}",
        f"- **Phone:** {parsed_cv.phone or 'N/A'}",
        f"- **Experience:** {parsed_cv.experience_years or 'N/A'} years",
        "",
    ]

    if parsed_cv.summary:
        md_lines.extend(
            [
                "## Summary",
                parsed_cv.summary,
                "",
            ]
        )

    if parsed_cv.projects:
        md_lines.extend(
            [
                f"## Projects ({len(parsed_cv.projects)})",
                "",
            ]
        )
        for i, proj in enumerate(parsed_cv.projects, 1):
            md_lines.extend(
                [
                    f"### {i}. {proj.title}",
                    f"**Duration:** {proj.duration or 'N/A'}",
                    "",
                    proj.description,
                    "",
                ]
            )
            if proj.technologies:
                md_lines.extend(
                    [
                        f"**Technologies:** {', '.join(proj.technologies)}",
                        "",
                    ]
                )

    if parsed_cv.skills:
        md_lines.extend(
            [
                f"## Skills ({len(parsed_cv.skills)})",
                "",
            ]
        )
        # Group skills by proficiency if available
        skills_by_prof = {}
        for skill in parsed_cv.skills:
            prof = skill.proficiency_level or "General"
            if prof not in skills_by_prof:
                skills_by_prof[prof] = []
            skills_by_prof[prof].append(skill)

        for prof, skills in skills_by_prof.items():
            if prof != "General":
                md_lines.append(f"### {prof.title()}")
            skill_list = ", ".join(
                [
                    f"{s.skill_name}"
                    + (f" ({s.years_of_experience}y)" if s.years_of_experience else "")
                    for s in skills
                ]
            )
            md_lines.extend([skill_list, ""])

    md_lines.append(
        f"\n---\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )

    # Write to file
    output_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"✓ CV saved to: {output_path}")


def auto_generate_cv_title(parsed_cv: ParsedCV) -> str:
    """Auto-generate CV title from parsed data"""
    if parsed_cv.name:
        return f"{parsed_cv.name}'s CV"

    if parsed_cv.experience_years:
        return f"Senior Engineer CV ({int(parsed_cv.experience_years)} yrs)"

    return "Unnamed CV"
