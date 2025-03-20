import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from processor.rag_system import RAGSystem
import os
import chromadb
from chromadb.errors import InvalidCollectionException

@pytest.fixture
def mock_embedder():
    """Fixture for mocking SentenceTransformer"""
    mock = Mock()
    mock.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    return mock

@pytest.fixture
def mock_anthropic():
    """Fixture for mocking Anthropic client"""
    mock = Mock()
    mock.messages.create.return_value = Mock(
        content=[Mock(text="Test recommendation")]
    )
    return mock

@pytest.fixture
def mock_chroma_client():
    """Fixture for mocking ChromaDB client"""
    mock = Mock()
    mock.get_collection.return_value = Mock()
    mock.create_collection.return_value = Mock()
    return mock

@pytest.fixture
def sample_data():
    """Fixture for sample knowledge base data"""
    return pd.DataFrame({
        'instruction': ['How do I reset password?'],
        'category': ['account'],
        'intent': ['password_reset'],
        'response': ['To reset your password, click on forgot password link']
    })

@pytest.fixture
def rag_system(mock_embedder, mock_anthropic, mock_chroma_client):
    """Fixture for RAG system with mocked dependencies"""
    with patch('processor.rag_system.SentenceTransformer', return_value=mock_embedder), \
         patch('processor.rag_system.Anthropic', return_value=mock_anthropic), \
         patch('processor.rag_system.chromadb.PersistentClient', return_value=mock_chroma_client), \
         patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
        return RAGSystem()

def test_init_without_api_key():
    """Test initialization without API key"""
    with patch.dict(os.environ, clear=True), \
         pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable not set"):
        RAGSystem()

def test_load_knowledge_base(rag_system, sample_data, tmp_path):
    """Test loading knowledge base from CSV"""
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    
    rag_system.load_knowledge_base(csv_path)
    
    assert len(rag_system.knowledge_base) == 1
    assert rag_system.knowledge_base[0]['id'] == '1'
    assert 'response' in rag_system.knowledge_base[0]

def test_populate_knowledge_base(rag_system):
    """Test populating ChromaDB with knowledge base"""
    rag_system.knowledge_base = [
        {'id': '1', 'response': 'Test response 1'},
        {'id': '2', 'response': 'Test response 2'}
    ]
    
    rag_system.collection.get.return_value = {'ids': []}
    rag_system.populate_knowledge_base()
    
    rag_system.collection.add.assert_called_once()

def test_retrieve_documents(rag_system):
    """Test document retrieval from ChromaDB"""
    rag_system.collection.query.return_value = {
        'documents': [['doc1', 'doc2']]
    }
    
    result = rag_system.retrieve_documents("test query")
    
    assert isinstance(result, list)
    assert len(result) == 2
    rag_system.collection.query.assert_called_once()

def test_generate_recommendation(rag_system):
    """Test recommendation generation using Anthropic API"""
    retrieved_docs = ['doc1', 'doc2']
    result = rag_system.generate_recommendation("test ticket", retrieved_docs)
    
    assert isinstance(result, str)
    assert result == "Test recommendation"
    rag_system.anthropic_client.messages.create.assert_called_once()

def test_process_ticket(rag_system):
    """Test end-to-end ticket processing"""
    rag_system.retrieve_documents = Mock(return_value=['doc1'])
    rag_system.generate_recommendation = Mock(return_value="Test recommendation")
    
    result = rag_system.process_ticket("test ticket")
    
    assert isinstance(result, str)
    assert result == "Test recommendation"
    rag_system.retrieve_documents.assert_called_once_with("test ticket")
    rag_system.generate_recommendation.assert_called_once()

def test_error_handling_retrieve_documents(rag_system):
    """Test error handling in document retrieval"""
    rag_system.collection.query.side_effect = Exception("Test error")
    
    result = rag_system.retrieve_documents("test query")
    
    assert result == []

def test_error_handling_generate_recommendation(rag_system):
    """Test error handling in recommendation generation"""
    rag_system.anthropic_client.messages.create.side_effect = Exception("Test error")
    
    result = rag_system.generate_recommendation("test ticket", ["doc1"])
    
    assert result == "Unable to generate recommendation due to an error."