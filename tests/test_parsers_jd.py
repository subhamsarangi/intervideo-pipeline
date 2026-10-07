"""Tests for JD parser"""

import pytest
from pathlib import Path
from src.parsers.jd_parser import (
    parse_jd_pasted,
    parse_jd_url,
    extract_jd_structure,
    auto_generate_jd_title,
)
from src.models.jd import ParsedJD, JDRequirement
from src.utils.llm_tracking import reset_tracker, get_llm_summary


class TestJDParser:
    @pytest.mark.asyncio
    async def test_parse_jd_real_file(self, tracker):
        """Test parsing real JD file with progress tracking"""
        reset_tracker()
        jd_path = Path("tests/fixtures/jobdesc1.txt")

        with tracker("JD Parsing") as bar:
            bar.update(10, "loading")
            with open(jd_path, "r") as f:
                jd_text = f.read()

            bar.update(20, "extracting")
            assert len(jd_text) > 0
            parsed = await parse_jd_pasted(jd_text)
            bar.update(60, "validating")

            assert isinstance(parsed, ParsedJD)
            assert len(parsed.requirements) > 0
            bar.update(10, "done")
            print(f"\n✓ JD parsed: {len(parsed.requirements)} requirements found")

    def test_extract_jd_structure_basic(self):
        """Test basic JD extraction"""
        reset_tracker()
        raw_text = """
        Senior Backend Engineer
        
        We're looking for a Senior Backend Engineer to join our platform team.
        
        Requirements:
        - 5+ years of backend development (must have)
        - Python or Go expertise (must have)
        - Kubernetes experience (nice to have)
        - Leadership experience (nice to have)
        """

        parsed = extract_jd_structure(raw_text)
        assert isinstance(parsed, ParsedJD)
        assert len(parsed.requirements) > 0
        must_haves = [
            r for r in parsed.requirements if r.requirement_type == "must_have"
        ]
        assert len(must_haves) > 0

    def test_auto_generate_jd_title(self):
        """Test JD title generation"""
        jd = ParsedJD(job_title="Senior Backend Engineer", company="TechCorp")
        title = auto_generate_jd_title(jd)
        assert "Senior Backend Engineer" in title
        assert "TechCorp" in title

    def test_auto_generate_jd_title_no_company(self):
        """Test JD title generation without company"""
        jd = ParsedJD(job_title="Data Scientist")
        title = auto_generate_jd_title(jd)
        assert "Data Scientist" in title

    def test_auto_generate_jd_title_empty(self):
        """Test JD title generation with empty data"""
        jd = ParsedJD()
        title = auto_generate_jd_title(jd)
        assert isinstance(title, str)
        assert len(title) > 0

    @pytest.mark.asyncio
    async def test_jd_parser_with_real_url(self):
        """Test JD parsing with real URL - should clean title intelligently"""
        reset_tracker()
        url = "https://www.shine.com/jobs/urgent-hiring-python-backend-with-aiml-west-bengal/calsoft/19164800?utm_campaign=google_jobs_apply&utm_source=google_jobs_apply&utm_medium=organic"

        parsed = await parse_jd_url(url)

        print(f"\n✓ URL parsed successfully!")
        print(f"  Job Title: {parsed.job_title}")
        print(f"  Company: {parsed.company}")
        print(f"  Requirements found: {len(parsed.requirements)}")

        # Display LLM tracking summary
        tracking_summary = get_llm_summary()
        print(f"\n  {tracking_summary.replace(chr(10), chr(10) + '  ')}")

        assert parsed.job_title is not None, "Job title should be extracted"
        assert len(parsed.requirements) > 0, "Requirements should be extracted"

        # Verify requirement types are valid
        for req in parsed.requirements:
            assert req.requirement_type in (
                "must_have",
                "nice_to_have",
            ), f"Invalid requirement_type: {req.requirement_type}"

        # Title should be cleaned (no location or urgency modifiers)
        assert (
            "urgent" not in parsed.job_title.lower()
        ), "Title should not contain 'urgent'"
        assert (
            "west bengal" not in parsed.job_title.lower()
        ), "Title should not contain location"

        print(f"\nFirst 3 requirements:")
        for i, req in enumerate(parsed.requirements[:3], 1):
            print(f"  {i}. [{req.requirement_type}] {req.text[:70]}")

    @pytest.mark.asyncio
    async def test_jd_parser_with_pasted_text(self, tracker):
        """Test JD parsing with pasted text"""
        reset_tracker()
        jd_text = """
        Senior Python Backend Engineer
        
        Company: TechCorp
        
        We're hiring a Senior Backend Engineer to lead our platform development.
        
        Requirements:
        - 5+ years of Python backend development (must have)
        - Experience with FastAPI or Django (must have)
        - PostgreSQL expertise (must have)
        - AWS/GCP experience (nice to have)
        - Leadership experience (nice to have)
        - GraphQL knowledge (nice to have)
        """

        with tracker("JD Paste Parsing", 100) as bar:
            bar.update(20, "validating")
            parsed = await parse_jd_pasted(jd_text)
            bar.update(50, "parsing")

            print(f"\n✓ Pasted text parsed successfully!")
            print(f"  Job Title: {parsed.job_title}")
            print(f"  Company: {parsed.company}")
            print(f"  Requirements: {len(parsed.requirements)}")

            bar.update(30, "done")

            assert len(parsed.requirements) > 0, "Requirements should be extracted"

            # Verify requirement types
            for req in parsed.requirements:
                assert req.requirement_type in (
                    "must_have",
                    "nice_to_have",
                ), f"Invalid type: {req.requirement_type}"

    def test_extract_jd_empty_text(self):
        """Test handling empty JD text"""
        reset_tracker()
        parsed = extract_jd_structure("")
        assert isinstance(parsed, ParsedJD)

    def test_extract_jd_minimal_text(self):
        """Test minimal JD text"""
        reset_tracker()
        parsed = extract_jd_structure("Python developer needed")
        assert isinstance(parsed, ParsedJD)

    def test_jd_requirement_validation(self):
        """Test JDRequirement type validation"""
        req = JDRequirement(text="5+ years experience", requirement_type="must_have")
        assert req.requirement_type == "must_have"

        with pytest.raises(ValueError):
            JDRequirement(text="5+ years experience", requirement_type="invalid_type")
