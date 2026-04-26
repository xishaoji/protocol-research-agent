"""
Unit tests for core/nodes.py
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import importlib

from core.state import AgentState
from langchain_core.messages import AIMessage, SystemMessage


@patch.dict('os.environ', {
    'OPENAI_API_KEY': 'test-key',
    'OPENAI_BASE_URL': 'https://api.test.com/v1',
    'TAVILY_API_KEY': 'test-tavily-key'
})
@pytest.mark.asyncio
async def test_researcher_node_execution():
    """Test researcher_node calls the ResearcherAgent."""
    from core.nodes import GraphNodes, AVAILABLE_TOOLS

    with patch('core.nodes.ResearcherAgent') as MockResearcher:
        mock_researcher = Mock()
        mock_researcher.ainvoke = AsyncMock(
            return_value=AIMessage(content="Research result")
        )
        MockResearcher.return_value = mock_researcher

        with patch('core.nodes.ReportAgent') as MockReport:
            mock_report = Mock()
            mock_report.ainvoke = AsyncMock(
                return_value=AIMessage(content="Report result")
            )
            MockReport.return_value = mock_report

            nodes = GraphNodes(model_name="test-model")
            state = AgentState(
                messages=[SystemMessage(content="Test")],
                search_count=0, query="test", research_targets=[],
                search_results=[], report_draft="",
                is_satisfactory=False, revision_count=0
            )

            result = await nodes.researcher_node(state)

            assert "messages" in result
            assert "search_count" in result
            assert result["search_count"] == 1


@patch.dict('os.environ', {
    'OPENAI_API_KEY': 'test-key',
    'OPENAI_BASE_URL': 'https://api.test.com/v1',
    'TAVILY_API_KEY': 'test-tavily-key'
})
@pytest.mark.asyncio
async def test_researcher_node_search_limit():
    """Test researcher_node forces report when search_count >= 10."""
    from core.nodes import GraphNodes

    with patch('core.nodes.ResearcherAgent') as MockResearcher:
        mock_researcher = Mock()
        MockResearcher.return_value = mock_researcher

        with patch('core.nodes.ReportAgent') as MockReport:
            MockReport.return_value = Mock()

            nodes = GraphNodes(model_name="test-model")
            state = AgentState(
                messages=[SystemMessage(content="Test")],
                search_count=10, query="test", research_targets=[],
                search_results=[], report_draft="",
                is_satisfactory=False, revision_count=0
            )

            result = await nodes.researcher_node(state)

            # Should return system message forcing report
            assert "messages" in result
            assert result["search_count"] == 0  # Reset after triggering limit
            assert isinstance(result["messages"][0], SystemMessage)
            assert "已达上限" in result["messages"][0].content


@patch.dict('os.environ', {
    'OPENAI_API_KEY': 'test-key',
    'OPENAI_BASE_URL': 'https://api.test.com/v1',
    'TAVILY_API_KEY': 'test-tavily-key'
})
@pytest.mark.asyncio
async def test_writer_node():
    """Test writer_node generates report."""
    from core.nodes import GraphNodes

    with patch('core.nodes.ResearcherAgent') as MockResearcher:
        MockResearcher.return_value = Mock()

        with patch('core.nodes.ReportAgent') as MockReport:
            mock_report = Mock()
            mock_report.ainvoke = AsyncMock(
                return_value=AIMessage(content="Final report content")
            )
            MockReport.return_value = mock_report

            nodes = GraphNodes(model_name="test-model")
            state = AgentState(
                messages=[SystemMessage(content="Test")],
                search_count=0, query="test", research_targets=[],
                search_results=[], report_draft="",
                is_satisfactory=False, revision_count=0
            )

            result = await nodes.writer_node(state)

            assert "report_draft" in result
            assert result["report_draft"] == "Final report content"
            assert "messages" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])