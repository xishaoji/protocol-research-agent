"""
Integration tests for the agent workflow.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from core.state import AgentState
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


@patch.dict('os.environ', {
    'OPENAI_API_KEY': 'test-key',
    'OPENAI_BASE_URL': 'https://api.test.com/v1',
    'TAVILY_API_KEY': 'test-tavily-key'
})
@patch('core.nodes.ResearcherAgent')
@patch('core.nodes.ReportAgent')
@pytest.mark.asyncio
async def test_workflow_researcher_to_writer(mock_report_cls, mock_researcher_cls):
    """Test researcher searches then writer produces report."""
    mock_researcher = Mock()
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "tavily_web_search", "args": {"query": "test"}, "id": "call_1"}]
    )
    complete_msg = AIMessage(content="Research complete")

    mock_researcher.ainvoke = AsyncMock(side_effect=[tool_call_msg, complete_msg])
    mock_researcher_cls.return_value = mock_researcher

    mock_report = Mock()
    mock_report.ainvoke = AsyncMock(return_value=AIMessage(content="Final report"))
    mock_report_cls.return_value = mock_report

    from core.nodes import GraphNodes

    nodes = GraphNodes(model_name="test-model")
    state = AgentState(
        messages=[HumanMessage(content="Test query")],
        search_count=0, query="test", research_targets=[],
        search_results=[], report_draft="",
        is_satisfactory=False, revision_count=0
    )

    # First call returns tool call message
    result1 = await nodes.researcher_node(state)
    assert result1["search_count"] == 1
    assert result1["messages"][0].tool_calls is not None

    # Writer generates final report
    result2 = await nodes.writer_node(state)
    assert "report_draft" in result2


@patch.dict('os.environ', {
    'OPENAI_API_KEY': 'test-key',
    'OPENAI_BASE_URL': 'https://api.test.com/v1',
    'TAVILY_API_KEY': 'test-tavily-key'
})
def test_should_continue_routing():
    """Test the should_continue routing logic (now module-level function)."""
    from core.graph import should_continue

    # State with tool call -> should route to "tools"
    state_with_tool = {
        "messages": [AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"query": "test"}, "id": "call_1"}]
        )],
        "search_count": 1
    }
    assert should_continue(state_with_tool) == "tools"

    # State without tool call -> should route to "report"
    state_without_tool = {
        "messages": [AIMessage(content="Research complete")],
        "search_count": 2
    }
    assert should_continue(state_without_tool) == "report"


@patch.dict('os.environ', {
    'OPENAI_API_KEY': 'test-key',
    'OPENAI_BASE_URL': 'https://api.test.com/v1',
    'TAVILY_API_KEY': 'test-tavily-key'
})
def test_state_persistence_in_workflow():
    """Test state persists search_count across iterations."""
    state = AgentState(
        messages=[HumanMessage(content="Test query")],
        search_count=0, query="test", research_targets=[],
        search_results=[], report_draft="",
        is_satisfactory=False, revision_count=0
    )

    for _ in range(3):
        state["search_count"] += 1

    assert state["search_count"] == 3