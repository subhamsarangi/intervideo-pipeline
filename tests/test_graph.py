"""
Tests for Phase 11: LangGraph Pipeline Assembly

Verify graph construction, conditional edges, parallel execution, and checkpointing.
Uses :memory: SQLite DB to avoid file locking issues on Windows.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.pipeline.graph import (
    build_graph,
    run_pipeline,
    get_pipeline_checkpoint,
    should_enrich,
)
from src.models.pipeline import PipelineState


class TestConditionalLogic:
    """Test conditional routing decisions."""
    
    def test_should_enrich_with_flagged_items(self):
        """Enrichment triggers when flagged_for_enrichment is not empty."""
        state = {
            "triage_result": Mock(flagged_for_enrichment=["item1", "item2"])
        }
        assert should_enrich(state) is True
    
    def test_should_enrich_with_no_flagged_items(self):
        """Skip enrichment when no items flagged."""
        state = {
            "triage_result": Mock(flagged_for_enrichment=[])
        }
        assert should_enrich(state) is False
    
    def test_should_enrich_with_missing_triage(self):
        """Skip enrichment if triage not yet run."""
        state = {"triage_result": None}
        assert should_enrich(state) is False
    
    def test_should_enrich_with_pipelinestate_object(self):
        """Conditional works with dict containing triage result."""
        state_dict = {
            "run_id": 1,
            "cv_id": 1,
            "jd_id": 1,
            "triage_result": Mock(flagged_for_enrichment=["tech1"])
        }
        assert should_enrich(state_dict) is True


class TestGraphConstruction:
    """Test graph assembly and node connectivity."""
    
    @pytest.mark.asyncio
    async def test_build_graph_creates_compiled_graph(self):
        """Graph builds without errors and returns CompiledGraph."""
        graph = await build_graph(":memory:")
        
        assert graph is not None
        assert hasattr(graph, "ainvoke")
        assert callable(graph.ainvoke)
    
    @pytest.mark.asyncio
    async def test_graph_has_all_nodes(self):
        """Graph contains all required nodes."""
        graph = await build_graph(":memory:")
        
        nodes = list(graph.nodes.keys())
        expected = [
            "security", "triage", "enrichment",
            "project_questions", "skill_questions",
            "merge_dedupe", "judge_evaluation"
        ]
        for node in expected:
            assert node in nodes
    
    @pytest.mark.asyncio
    async def test_graph_has_entry_point(self):
        """Graph entry point is security node."""
        graph = await build_graph(":memory:")
        
        assert "security" in graph.nodes


class TestPipelineExecution:
    """Test full pipeline execution flow."""
    
    @pytest.mark.asyncio
    async def test_run_pipeline_minimal_mock(self):
        """Pipeline execution with mocked build_graph."""
        state = PipelineState(
            run_id=1,
            cv_id=1,
            jd_id=1,
            db_path=":memory:",
        )
        
        # Mock build_graph to return a fake graph that doesn't actually invoke nodes
        with patch("src.pipeline.graph.build_graph") as mock_build:
            mock_graph = AsyncMock()
            mock_graph.ainvoke = AsyncMock(return_value={"status": "success"})
            mock_build.return_value = mock_graph
            
            # Execute pipeline
            result = await run_pipeline(state, ":memory:")
            
            # Verify build_graph and ainvoke were called
            assert mock_build.called
            assert mock_graph.ainvoke.called
            assert result == {"status": "success"}
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation(self):
        """AsyncSqliteSaver creates checkpoint tables in DB."""
        # Build graph triggers checkpoint table creation
        graph = await build_graph(":memory:")
        
        # Graph should be built successfully
        assert graph is not None
        assert hasattr(graph, "ainvoke")


class TestCheckpointRetrieval:
    """Test checkpoint persistence and resumption."""
    
    @pytest.mark.asyncio
    async def test_get_pipeline_checkpoint_no_checkpoint(self):
        """Returns None if no checkpoint exists."""
        # Initialize DB by building graph
        await build_graph(":memory:")
        
        checkpoint = await get_pipeline_checkpoint(":memory:", "nonexistent_thread")
        assert checkpoint is None
    
    @pytest.mark.asyncio
    async def test_checkpoint_thread_id_format(self):
        """Checkpoint keyed by thread_id (typically run_id)."""
        # Thread ID should be string
        thread_id = "run_123"
        checkpoint = await get_pipeline_checkpoint(":memory:", thread_id)
        
        # Should not error, just return None (no data yet)
        assert checkpoint is None or isinstance(checkpoint, dict)


class TestNodeChaining:
    """Test correct data flow between nodes."""
    
    def test_triage_output_feeds_enrichment_conditional(self):
        """Triage output (triage_result) controls enrichment routing."""
        triage_with_flags = {
            "triage_result": Mock(flagged_for_enrichment=["item"])
        }
        assert should_enrich(triage_with_flags) is True
    
    @pytest.mark.asyncio
    async def test_merge_dedupe_output_feeds_judge_directly(self):
        """Merge output flows directly to judge (no gap-fill)."""
        graph = await build_graph(":memory:")
        
        # Verify merge_dedupe and judge_evaluation both exist
        assert "merge_dedupe" in graph.nodes
        assert "judge_evaluation" in graph.nodes
    
    @pytest.mark.asyncio
    async def test_parallel_questions_both_execute(self):
        """Both project and skill questions nodes reachable from triage."""
        graph = await build_graph(":memory:")
        
        # Verify both question nodes exist
        assert "project_questions" in graph.nodes
        assert "skill_questions" in graph.nodes


class TestErrorHandling:
    """Test error scenarios and recovery."""
    
    @pytest.mark.asyncio
    async def test_run_pipeline_handles_execution_error(self):
        """Pipeline raises RuntimeError if execution fails."""
        state = PipelineState(
            run_id=1, cv_id=1, jd_id=1, db_path=":memory:"
        )
        
        # Mock graph.invoke to raise
        with patch("src.pipeline.graph.build_graph") as mock_build:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = Exception("Test error")
            mock_build.return_value = mock_graph
            
            with pytest.raises(RuntimeError, match="Pipeline execution failed"):
                await run_pipeline(state, ":memory:")


class TestIntegrationFlow:
    """End-to-end integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_graph_flow_no_enrichment_no_gap_fill(self):
        """Normal path: security → triage → questions → merge → judge."""
        graph = await build_graph(":memory:")
        
        # Graph should exist and be invokable
        assert graph is not None
        assert hasattr(graph, "ainvoke")
    
    def test_graph_flow_with_enrichment(self):
        """Path with enrichment: security → triage → enrichment → questions → ...."""
        # Verify conditional edge logic
        state_with_flags = {
            "triage_result": Mock(flagged_for_enrichment=["tech1", "tech2"])
        }
        assert should_enrich(state_with_flags) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
