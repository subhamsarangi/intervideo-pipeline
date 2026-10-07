"""Gap-fill node for retrying empty/thin question generation results"""

from typing import Any, Dict, Optional, Union

from src.models.pipeline import PipelineState


async def gap_fill_node(
    state: Union[PipelineState, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Gap-fill LangGraph node.

    Retries question generation when results are empty or thin.

    Triggers:
    - project_questions returned 0 questions
    - skill_questions returned <2 total questions
    - After merge/dedupe, <50% JD requirements have ≥1 question

    Implements adjusted prompts to generate fallback questions.

    Args:
        state: PipelineState or dict

    Returns:
        Dict with status, additional_questions, and error_message
    """
    # TODO: Implement gap-fill retry logic
    # 1. Check if gap-fill needed (coverage threshold)
    # 2. Identify uncovered JD requirements
    # 3. Generate fallback questions with adjusted prompts
    # 4. Return additional questions to merge with existing
    
    return {
        "status": "running",
        "additional_questions": [],
        "error_message": None,
    }
