"""Tests for LLM tracking persistence to database"""

import pytest
import asyncio
from pathlib import Path
from src.database.db import (
    init_db,
    create_cv,
    create_jd,
    get_cv_cost,
    get_jd_cost,
    get_average_costs,
    display_cost_report,
)
from src.parsers.cv_parser import parse_cv
from src.parsers.jd_parser import parse_jd
import os


@pytest.fixture
def test_db():
    """Create temporary test database"""
    db_path = "test_tracking.db"
    init_db(db_path)  # Use current schema.sql which has all tracking columns
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


class TestTrackingPersistence:
    @pytest.mark.asyncio
    async def test_cv_tracking_persistence(self, test_db, tracker):
        """Test CV parsing saves tracking data to database"""
        # Create CV in database
        cv_id = create_cv("Test CV", "raw text", test_db)

        # Parse CV with tracking
        cv_path = Path("tests/fixtures/cvfile.pdf")

        with tracker("CV Tracking Test") as bar:
            bar.update(20, "parsing")
            parsed_cv = await parse_cv(str(cv_path), cv_id=cv_id, db_path=test_db)
            bar.update(50, "saving")

            # Get cost data from database
            cost_data = get_cv_cost(cv_id, test_db)
            bar.update(30, "done")

            print(f"\n✓ CV tracking saved to database")
            print(f"  CV ID: {cv_id}")
            print(f"  LLM Calls: {cost_data['llm_calls_count']}")
            print(f"  Input Tokens: {cost_data['total_input_tokens']}")
            print(f"  Output Tokens: {cost_data['total_output_tokens']}")
            print(f"  Total Cost: ${cost_data['total_cost_usd']:.6f}")
            print(f"  Call Log Length: {len(cost_data['llm_calls_log'])}")

            # Verify data was saved
            assert cost_data["llm_calls_count"] > 0, "Should have LLM calls"
            assert cost_data["total_input_tokens"] > 0, "Should have input tokens"
            assert cost_data["total_output_tokens"] > 0, "Should have output tokens"
            assert cost_data["total_cost_usd"] > 0, "Should have cost > 0"
            assert len(cost_data["llm_calls_log"]) > 0, "Should have call log"

            # Verify call log structure
            first_call = cost_data["llm_calls_log"][0]
            assert "operation" in first_call, "Call should have operation"
            assert "model" in first_call, "Call should have model"
            assert "input_tokens" in first_call, "Call should have input tokens"
            assert "output_tokens" in first_call, "Call should have output tokens"

    @pytest.mark.asyncio
    async def test_jd_tracking_persistence(self, test_db, tracker):
        """Test JD parsing saves tracking data to database"""
        # Create JD in database
        jd_id = create_jd("Test JD", "raw text", "pasted", None, test_db)

        jd_text = """
        Senior Python Backend Engineer
        
        Requirements:
        - 5+ years Python (must have)
        - FastAPI experience (must have)  
        - AWS knowledge (nice to have)
        """

        with tracker("JD Tracking Test") as bar:
            bar.update(30, "parsing")
            parsed_jd = await parse_jd("pasted", jd_text, jd_id=jd_id, db_path=test_db)
            bar.update(40, "saving")

            # Get cost data from database
            cost_data = get_jd_cost(jd_id, test_db)
            bar.update(30, "done")

            print(f"\n✓ JD tracking saved to database")
            print(f"  JD ID: {jd_id}")
            print(f"  LLM Calls: {cost_data['llm_calls_count']}")
            print(f"  Input Tokens: {cost_data['total_input_tokens']}")
            print(f"  Output Tokens: {cost_data['total_output_tokens']}")
            print(f"  Total Cost: ${cost_data['total_cost_usd']:.6f}")
            print(f"  Requirements Found: {len(parsed_jd.requirements)}")

            # Verify data was saved
            assert cost_data["llm_calls_count"] > 0, "Should have LLM calls"
            assert cost_data["total_input_tokens"] > 0, "Should have input tokens"
            assert cost_data["total_output_tokens"] > 0, "Should have output tokens"
            assert cost_data["total_cost_usd"] > 0, "Should have cost > 0"
            assert len(parsed_jd.requirements) > 0, "Should extract requirements"

    def test_cost_accuracy_validation(self, test_db):
        """Test cost calculation accuracy against known token pricing"""
        # Known: gpt-5.6-luna = $0.075 per 1M input, $0.30 per 1M output
        from src.utils.llm_tracking import MODEL_COSTS

        # Test calculation
        input_tokens = 1000
        output_tokens = 500
        model = "gpt-5.6-luna"

        expected_cost = (input_tokens * MODEL_COSTS[model]["input"] / 1_000_000) + (
            output_tokens * MODEL_COSTS[model]["output"] / 1_000_000
        )

        # Expected: (1000 * 0.075 / 1M) + (500 * 0.30 / 1M) = 0.000225
        assert (
            abs(expected_cost - 0.000225) < 0.000001
        ), f"Cost calculation incorrect: {expected_cost}"

        print(f"\n✓ Cost calculation validation")
        print(f"  1000 input + 500 output tokens = ${expected_cost:.6f}")
        print(
            f"  Model pricing: ${MODEL_COSTS[model]['input']}/1M input, ${MODEL_COSTS[model]['output']}/1M output"
        )

    def test_cost_query_functions(self, test_db):
        """Test cost query functions with empty database"""
        # Test with no data
        cv_avg = get_average_costs("cv", test_db)
        jd_avg = get_average_costs("jd", test_db)

        assert cv_avg["document_count"] == 0, "Should have 0 CVs"
        assert jd_avg["document_count"] == 0, "Should have 0 JDs"

        # Test display function
        print(f"\n--- Empty Database Cost Report ---")
        display_cost_report(test_db)

        print(f"\n✓ Cost query functions work with empty database")
