"""
Unit tests for base_agent.py
"""
import pytest
from unittest.mock import Mock, patch


@patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key', 'OPENAI_BASE_URL': 'https://api.test.com/v1'})
def test_base_agent_initialization():
    """Test BaseAgent initialization."""
    from agents.base_agent import BaseAgent

    with patch('agents.base_agent.ChatOpenAI') as mock_chat:
        mock_llm = Mock()
        mock_chat.return_value = mock_llm

        agent = BaseAgent(role_name="TestAgent", model_name="gpt-test", temperature=0.2)

        assert agent.role_name == "TestAgent"
        mock_chat.assert_called_once_with(
            model="gpt-test",
            temperature=0.2,
            max_retries=3,
            timeout=60
        )


def test_base_agent_missing_api_key():
    """Test BaseAgent raises error when OPENAI_API_KEY is missing."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="缺少 OPENAI_API_KEY 环境变量"):
            from agents.base_agent import BaseAgent
            BaseAgent(role_name="TestAgent")


@patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key', 'OPENAI_BASE_URL': 'https://api.test.com/v1'})
def test_get_prompt_template():
    """Test prompt template generation."""
    from agents.base_agent import BaseAgent

    with patch('agents.base_agent.ChatOpenAI') as mock_chat:
        mock_llm = Mock()
        mock_chat.return_value = mock_llm

        agent = BaseAgent(role_name="TestAgent")
        prompt_template = agent.get_prompt_template("Test system message")

        assert prompt_template is not None
        assert "system" in str(prompt_template.messages[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])