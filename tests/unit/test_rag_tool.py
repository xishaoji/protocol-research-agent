"""
Unit tests for rag_tool.py

rag_tool.py has module-level code that loads FAISS if ./faiss_db exists.
We mock os.path.exists to ensure clean import, then set vector_db as needed.
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture(autouse=True)
def setup_env():
    """Set required env vars and prevent FAISS loading at import."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}, clear=False):
        with patch('os.path.exists', return_value=False):
            yield


def test_search_internal_docs_no_database():
    """Test search when vector database is not initialized."""
    from tools.rag_tool import search_internal_docs

    # Ensure vector_db is None for this test
    import tools.rag_tool
    tools.rag_tool.vector_db = None

    result = search_internal_docs.invoke({"query": "test query"})
    result_data = json.loads(result)

    assert "error" in result_data
    assert "未初始化" in result_data["error"]


def test_search_internal_docs_success():
    """Test successful internal document search."""
    from tools.rag_tool import search_internal_docs

    # Create mock DB
    mock_db = MagicMock()
    mock_doc1 = MagicMock()
    mock_doc1.metadata = {"source": "doc1.pdf", "page": 1}
    mock_doc1.page_content = "Charging protocol OCPP 1.6 content"
    mock_doc2 = MagicMock()
    mock_doc2.metadata = {"source": "doc2.pdf", "page": 5}
    mock_doc2.page_content = "More protocol details"

    mock_db.similarity_search_with_score.return_value = [
        (mock_doc1, 0.1), (mock_doc2, 0.2)
    ]

    import tools.rag_tool
    old_db = tools.rag_tool.vector_db
    tools.rag_tool.vector_db = mock_db

    try:
        result = search_internal_docs.invoke({"query": "charging protocols"})
        result_data = json.loads(result)

        assert len(result_data) == 2
        assert result_data[0]["source"] == "doc1.pdf"
        assert result_data[0]["page"] == 1
        assert "OCPP" in result_data[0]["content"]
    finally:
        tools.rag_tool.vector_db = old_db


def test_search_internal_docs_error_handling():
    """Test error handling during search."""
    from tools.rag_tool import search_internal_docs

    mock_db = MagicMock()
    mock_db.similarity_search_with_score.side_effect = Exception("DB Error")

    import tools.rag_tool
    old_db = tools.rag_tool.vector_db
    tools.rag_tool.vector_db = mock_db

    try:
        result = search_internal_docs.invoke({"query": "test"})
        result_data = json.loads(result)
        assert "error" in result_data
    finally:
        tools.rag_tool.vector_db = old_db


def test_search_internal_docs_empty_results():
    """Test search when no similar documents found."""
    from tools.rag_tool import search_internal_docs

    mock_db = MagicMock()
    mock_db.similarity_search_with_score.return_value = []

    import tools.rag_tool
    old_db = tools.rag_tool.vector_db
    tools.rag_tool.vector_db = mock_db

    try:
        result = search_internal_docs.invoke({"query": "nothing"})
        result_data = json.loads(result)
        assert result_data == []
    finally:
        tools.rag_tool.vector_db = old_db