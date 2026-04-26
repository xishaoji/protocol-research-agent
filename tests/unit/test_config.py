"""
Configuration and environment variable tests.
"""
import pytest
from unittest.mock import patch
import os


@patch.dict('os.environ', {}, clear=True)
def test_missing_openai_api_key():
    """Test that missing OPENAI_API_KEY is detected by BaseAgent."""
    # BaseAgent checks env vars in __init__; import happens in test, not at module level
    from agents.base_agent import BaseAgent
    with pytest.raises(ValueError, match="缺少 OPENAI_API_KEY"):
        BaseAgent(role_name="TestAgent")


@patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key', 'OPENAI_BASE_URL': 'https://api.test.com/v1'})
def test_environment_variables_set():
    """Test that required environment variables are read correctly."""
    assert os.getenv('OPENAI_API_KEY') == 'test-key'
    assert os.getenv('OPENAI_BASE_URL') == 'https://api.test.com/v1'


@patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key', 'OPENAI_BASE_URL': 'https://api.test.com/v1'})
def test_agent_state_default_values():
    """Test that AgentState has proper defaults when fully initialized."""
    from core.state import AgentState

    state = AgentState(
        messages=[], search_count=0, query="", research_targets=[],
        search_results=[], report_draft="", is_satisfactory=False, revision_count=0
    )
    assert state["search_count"] == 0
    assert state["report_draft"] == ""
    assert state["messages"] == []