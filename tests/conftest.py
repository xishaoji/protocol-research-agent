"""
Pytest configuration and shared fixtures for the research agent project.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


@pytest.fixture
def mock_environment():
    """Mock all required environment variables."""
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_BASE_URL': 'https://api.test.com/v1',
        'TAVILY_API_KEY': 'test-tavily-key'
    }):
        yield


@pytest.fixture
def sample_messages():
    """Sample message list for testing."""
    return [
        SystemMessage(content="Test system message"),
        HumanMessage(content="Test human message"),
        AIMessage(content="Test AI message")
    ]


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock_llm = MagicMock()
    mock_llm.bind_tools = Mock(return_value=mock_llm)
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
    return mock_llm


@pytest.fixture
def mock_tool():
    """Mock tool for testing."""
    mock_tool = Mock()
    mock_tool.name = "mock_tool"
    return mock_tool