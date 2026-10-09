"""JD Parser - handles pasted text, URLs, PDF files, and file uploads with Instructor"""

import json
import os
import re
from typing import Optional
from dotenv import load_dotenv
import httpx
from openai import OpenAI
import instructor
from llama_cloud import AsyncLlamaCloud

from src.models.jd import ParsedJD, JDRequirement
from src.utils.security import validate_jd_url
from src.utils.llm_tracking import increment_llm_calls, get_tracker, reset_tracker

# Load environment variables
load_dotenv()


def get_openai_client():
    """Get OpenAI client with Instructor"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    client = OpenAI(api_key=api_key)
    # Patch the client with Instructor for structured output
    return instructor.from_openai(client)


def get_plain_openai_client():
    """Get plain OpenAI client (for non-structured tasks like title cleaning)"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=api_key)


def get_llama_api_key() -> str:
    """Get LlamaParse API key from environment"""
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY not set in .env")
    return api_key


async def parse_jd_pdf(file_path: str) -> str:
    """Parse JD PDF using LlamaParse (same as CV parsing)"""
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
            operation="llamaparse_jd_pdf",
            model="llamaparse",
            input_tokens=0,
            output_tokens=0,
        )

        return md_text

    except Exception as e:
        raise Exception(f"LlamaParse JD PDF failed: {str(e)}")


async def fetch_url_content(url: str) -> str:
    """Fetch job description from URL"""
    validate_jd_url(url)

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        content = response.text
        content = re.sub(r"<[^>]+>", "", content)
        return content


def extract_jd_structure(raw_text: str) -> ParsedJD:
    """Use OpenAI with Instructor to extract JD data with structured validation"""
    client = get_openai_client()

    response = client.chat.completions.create(
        model="gpt-5.4-nano",
        response_model=ParsedJD,
        messages=[
            {
                "role": "system",
                "content": """Extract job requirements from the text and return valid structured data.

For each requirement:
- Classify as 'must_have' (essential) or 'nice_to_have' (preferred)
- Extract exact text
- Categorize: 'technical', 'soft_skills', 'experience', or 'other'""",
            },
            {"role": "user", "content": f"Extract requirements from:\n\n{raw_text}"},
        ],
        temperature=0.1,
    )
    parsed_jd = response
    # Get usage from the raw response
    raw_response = response._raw_response
    increment_llm_calls(
        "extract_jd_structure",
        input_tokens=raw_response.usage.prompt_tokens,
        output_tokens=raw_response.usage.completion_tokens,
    )

    return parsed_jd


async def parse_jd_pasted(text: str) -> ParsedJD:
    """Parse pasted JD text"""
    if not text or len(text.strip()) == 0:
        raise ValueError("Empty JD text")
    return extract_jd_structure(text)


async def parse_jd_url(url: str) -> ParsedJD:
    """Parse JD from URL"""
    raw_text = await fetch_url_content(url)
    if not raw_text or len(raw_text.strip()) == 0:
        raise ValueError("No content fetched from URL")

    parsed = extract_jd_structure(raw_text)

    # Use OpenAI to clean up job title if it contains noise
    if parsed.job_title:
        client = get_plain_openai_client()
        try:
            response = client.chat.completions.create(
                model="gpt-5.4-nano",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a job title cleaner. Given a job title that may contain location, 
urgency modifiers, or other metadata, extract just the core job title.

Return ONLY the clean job title as a single line, nothing else.
Examples:
- "Urgent Hiring - Python Backend with AIML (West Bengal)" → "Python Backend with AIML"
- "Sr. Engineer, Bangalore | Full-Stack (Immediate)" → "Sr. Engineer, Full-Stack"
- "DevOps Engineer - Remote/Bangalore" → "DevOps Engineer"
""",
                    },
                    {
                        "role": "user",
                        "content": f"Clean this job title: {parsed.job_title}",
                    },
                ],
                temperature=0.1,
            )
            increment_llm_calls(
                "extract_jd_url_title",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            clean_title = response.choices[0].message.content.strip()
            if clean_title and len(clean_title) > 3:  # Only use if reasonable
                parsed.job_title = clean_title
        except Exception:
            pass  # Keep original title if cleaning fails

    return parsed


async def parse_jd_file(file_content: str, file_name: str) -> ParsedJD:
    """Parse JD from uploaded file"""
    if not file_content or len(file_content.strip()) == 0:
        raise ValueError("Empty file content")

    if file_name.endswith(".html"):
        file_content = re.sub(r"<[^>]+>", "", file_content)

    return extract_jd_structure(file_content)


async def parse_jd(
    source_type: str,
    source_data: str,
    source_url: Optional[str] = None,
    jd_id: Optional[int] = None,
    db_path: Optional[str] = None,
) -> ParsedJD:
    """Main JD parsing entry point with tracking persistence"""
    reset_tracker()

    if source_type == "pasted":
        parsed_jd = await parse_jd_pasted(source_data)
    elif source_type == "url":
        if not source_url:
            raise ValueError("source_url required for URL source type")
        parsed_jd = await parse_jd_url(source_url)
    elif source_type == "pdf":
        # source_data is file path for PDF
        raw_text = await parse_jd_pdf(source_data)
        if not raw_text or len(raw_text.strip()) == 0:
            raise ValueError("No text extracted from JD PDF")
        parsed_jd = extract_jd_structure(raw_text)
    elif source_type == "file":
        if not source_url:
            raise ValueError("file_name required for file source type")
        parsed_jd = await parse_jd_file(source_data, source_url)
    else:
        raise ValueError(f"Invalid source_type: {source_type}")

    # Save parsed content to outputs folder
    save_jd_to_markdown(parsed_jd, source_data)

    # Save tracking data to database if jd_id provided
    if jd_id is not None:
        tracker = get_tracker()
        cost_info = tracker.calculate_cost()

        from src.database.db import update_jd_status

        update_jd_status(
            jd_id=jd_id,
            status="completed",
            parsed_data=parsed_jd.model_dump(),
            llm_calls_count=tracker.get_count(),
            llm_calls_log=tracker.get_calls(),
            total_input_tokens=cost_info["total_input_tokens"],
            total_output_tokens=cost_info["total_output_tokens"],
            total_cost_usd=cost_info["total_cost_usd"],
            db_path=db_path,
        )

    return parsed_jd


def auto_generate_jd_title(parsed_jd: ParsedJD) -> str:
    """Auto-generate JD title from parsed data"""
    if parsed_jd.job_title and parsed_jd.company:
        return f"{parsed_jd.job_title} at {parsed_jd.company}"
    if parsed_jd.job_title:
        return parsed_jd.job_title
    return "Unnamed Job Description"


def save_jd_to_markdown(parsed_jd: ParsedJD, file_path: str) -> None:
    """Save parsed JD to outputs folder as markdown"""
    from pathlib import Path
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r"[^\w\s-]", "", parsed_jd.job_title or "JD").replace(" ", "_")
    output_file = Path(f"outputs/jd_{safe_title}_{timestamp}.md")
    output_file.parent.mkdir(exist_ok=True)
    
    # Separate must-have and nice-to-have
    must_have = [r for r in (parsed_jd.requirements or []) if r.requirement_type == "must_have"]
    nice_to_have = [r for r in (parsed_jd.requirements or []) if r.requirement_type == "nice_to_have"]
    
    content = f"""# Job Description: {parsed_jd.job_title or 'N/A'}

**Company:** {parsed_jd.company or 'N/A'}

## Summary
{parsed_jd.summary or 'N/A'}

## Must-Have Requirements ({len(must_have)})
"""
    
    for i, req in enumerate(must_have, 1):
        content += f"\n{i}. {req.text}"
        if req.category:
            content += f" _{req.category}_"
        content += "\n"
    
    content += f"\n## Nice-to-Have Requirements ({len(nice_to_have)})\n"
    for i, req in enumerate(nice_to_have, 1):
        content += f"\n{i}. {req.text}"
        if req.category:
            content += f" _{req.category}_"
        content += "\n"
    
    output_file.write_text(content)
