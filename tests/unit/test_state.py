"""
Unit tests for AgentState in core/state.py
"""
import pytest
from core.state import AgentState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def test_agent_state_defaults():
    """Test AgentState TypedDict with empty initialization."""
    state = AgentState(
        messages=[],
        search_count=0,
        query="",
        research_targets=[],
        search_results=[],
        report_draft="",
        is_satisfactory=False,
        revision_count=0
    )
    assert state["messages"] == []
    assert state["search_count"] == 0
    assert state["report_draft"] == ""


def test_agent_state_with_values():
    """Test AgentState with initial values (TypedDict)."""
    initial_messages = [
        SystemMessage(content="System prompt"),
        HumanMessage(content="User query")
    ]
    state = AgentState(
        messages=initial_messages,
        search_count=0,
        query="test query",
        research_targets=[],
        search_results=[],
        report_draft="",
        is_satisfactory=False,
        revision_count=0
    )
    assert len(state["messages"]) == 2
    assert state["query"] == "test query"
    assert state["is_satisfactory"] is False


def test_agent_state_fields():
    """Test AgentState has all required fields."""
    state = AgentState(
        messages=[],
        search_count=0,
        query="",
        research_targets=[],
        search_results=[],
        report_draft="",
        is_satisfactory=False,
        revision_count=0
    )
    # All TypedDict keys should be present
    expected_keys = {
        "messages", "search_count", "query", "research_targets",
        "search_results", "report_draft", "is_satisfactory", "revision_count"
    }
    assert set(state.keys()) == expected_keys


def test_state_search_count_increment():
    """Test that search_count can be properly tracked."""
    state = AgentState(search_count=0)
    state["search_count"] += 1
    assert state["search_count"] == 1

    state["search_count"] += 1
    assert state["search_count"] == 2


def test_state_messages_append():
    """Test appending messages to state."""
    state = AgentState(messages=[])
    state["messages"].append(SystemMessage(content="Test"))
    assert len(state["messages"]) == 1
    assert state["messages"][0].content == "Test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])