"""
Unit tests for search_tool.py

search_tool checks TAVILY_API_KEY at import time.
The @tool decorator returns a StructuredTool - use .invoke().
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

    # Patch the search_client *instance* directly, since it's created at module level
    with patch('tools.search_tool.search_client') as mock_client:
        mock_client.invoke.return_value = mock_results

        result = search_web.invoke({"query": "test query"})
        result_data = json.loads(result)

        assert isinstance(result_data, list)
        assert len(result_data) == 2
        assert result_data[0]["title"] == "Test A"


def test_search_web_api_failure():
    """Test error handling on API failure."""
    from tools.search_tool import search_web

    with patch('tools.search_tool.search_client') as mock_client:
        mock_client.invoke.side_effect = Exception("API Error")

        result = search_web.invoke({"query": "test"})
        result_data = json.loads(result)

        assert "error" in result_data


def test_search_web_empty_results():
    """Test search with empty results."""
    from tools.search_tool import search_web

    with patch('tools.search_tool.search_client') as mock_client:
        mock_client.invoke.return_value = []

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

    with patch('tools.search_tool.search_client') as mock_client:
        mock_client.invoke.return_value = mock_results

        result = search_web.invoke({"query": "long"})
        result_data = json.loads(result)

        assert len(result_data[0]["content"]) <= 800