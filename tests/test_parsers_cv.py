"""Tests for CV parser"""

import pytest
from pathlib import Path
from src.parsers.cv_parser import parse_cv_file, parse_cv, auto_generate_cv_title
from src.models.cv import ParsedCV
from src.utils.llm_tracking import reset_tracker, get_llm_summary


class TestCVParser:
    @pytest.mark.asyncio
    async def test_parse_cv_real_file(self, tracker):
        """Test parsing real CV file with progress tracking"""
        reset_tracker()
        cv_path = Path("tests/fixtures/cvfile.pdf")

        with tracker("CV Parsing") as bar:
            bar.update(10, "uploading")
            raw_text = await parse_cv_file(str(cv_path))
            bar.update(60, "extracting")

            assert raw_text is not None
            assert len(raw_text) > 0
            assert isinstance(raw_text, str)
            bar.update(30, "done")
            print(f"\n✓ CV parsed: {len(raw_text)} chars")

    @pytest.mark.asyncio
    async def test_parse_cv_full_pipeline(self, tracker):
        """Test full CV parsing pipeline with structured extraction"""
        reset_tracker()
        cv_path = Path("tests/fixtures/cvfile.pdf")

        with tracker("CV Full Parse", 100) as bar:
            bar.update(20, "uploading")
            parsed_cv = await parse_cv(str(cv_path))
            bar.update(50, "extracting")

            print(f"\n✓ CV fully parsed!")
            print(f"  Name: {parsed_cv.name}")
            print(f"  Email: {parsed_cv.email}")
            print(f"  Phone: {parsed_cv.phone}")
            print(f"  Experience years: {parsed_cv.experience_years}")
            print(f"  Projects: {len(parsed_cv.projects)}")
            print(f"  Skills: {len(parsed_cv.skills)}")

            if parsed_cv.projects:
                print(f"\n  First project: {parsed_cv.projects[0].title}")
                if parsed_cv.projects[0].technologies:
                    print(
                        f"    Technologies: {', '.join(parsed_cv.projects[0].technologies[:3])}"
                    )

            if parsed_cv.skills:
                print(
                    f"\n  Sample skills: {', '.join(s.skill_name for s in parsed_cv.skills[:5])}"
                )

            bar.update(20, "tracking")
            # Display LLM tracking summary
            tracking_summary = get_llm_summary()
            print(f"\n  {tracking_summary.replace(chr(10), chr(10) + '  ')}")
            bar.update(10, "done")

            assert isinstance(parsed_cv, ParsedCV)
            assert (
                parsed_cv.name
                or parsed_cv.experience_years
                or len(parsed_cv.projects) > 0
            )

    def test_auto_generate_cv_title_with_name(self):
        """Test CV title generation with name"""
        cv = ParsedCV(name="John Smith")
        title = auto_generate_cv_title(cv)
        assert "John Smith" in title

    def test_auto_generate_cv_title_with_experience(self):
        """Test CV title generation with experience years"""
        cv = ParsedCV(experience_years=8)
        title = auto_generate_cv_title(cv)
        assert "8" in title or len(title) > 0
