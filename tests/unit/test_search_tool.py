"""
Unit tests for search_tool.py

search_tool uses lazy initialization for the Tavily client.
We patch _get_search_client to return a mock client.
"""
import pytest
import json
from unittest.mock import Mock, patch


@pytest.fixture(autouse=True)
def setup_env():
    """Set required env vars before each test."""
    with patch.dict('os.environ', {'TAVILY_API_KEY': 'test-tavily-key'}):
        yield


def test_search_web_success():
    """Test successful web search."""
    from tools.search_tool import search_web

    mock_results = [
        {"title": "Test A", "url": "https://ex.com/a", "content": "Content A"},
        {"title": "Test B", "url": "https://ex.com/b", "content": "Content B"}
    ]

    mock_client = Mock()
    mock_client.invoke.return_value = mock_results

    with patch('tools.search_tool._get_search_client', return_value=mock_client):
        result = search_web.invoke({"query": "test query"})
        result_data = json.loads(result)

        assert isinstance(result_data, list)
        assert len(result_data) == 2
        assert result_data[0]["title"] == "Test A"


def test_search_web_api_failure():
    """Test error handling on API failure."""
    from tools.search_tool import search_web

    mock_client = Mock()
    mock_client.invoke.side_effect = Exception("API Error")

    with patch('tools.search_tool._get_search_client', return_value=mock_client):
        result = search_web.invoke({"query": "test"})
        result_data = json.loads(result)

        assert "error" in result_data


def test_search_web_empty_results():
    """Test search with empty results."""
    from tools.search_tool import search_web

    mock_client = Mock()
    mock_client.invoke.return_value = []

    with patch('tools.search_tool._get_search_client', return_value=mock_client):
        result = search_web.invoke({"query": "empty"})
        result_data = json.loads(result)

        assert result_data == []


def test_search_web_content_truncation():
    """Test long content is truncated to 800 chars."""
    from tools.search_tool import search_web

    long_content = "a" * 1000
    mock_results = [{
        "title": "Long", "url": "https://ex.com/l", "content": long_content
    }]

    mock_client = Mock()
    mock_client.invoke.return_value = mock_results

    with patch('tools.search_tool._get_search_client', return_value=mock_client):
        result = search_web.invoke({"query": "long"})
        result_data = json.loads(result)

        assert len(result_data[0]["content"]) <= 800
