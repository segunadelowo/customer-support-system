import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from models.intent_classification_model import IntentClassificationModel

@pytest.fixture
def model():
    """Fixture to create a model instance for testing"""
    return IntentClassificationModel(
        vocab_size=100,
        embedding_dim=8,
        max_length=5,
        filepath=Path("tests/test_artifacts")
    )

@pytest.fixture
def sample_data():
    """Fixture to create sample training data"""
    data = {
        'instruction': [
            "How do I reset my password?",
            "Where is my order?",
            "Cancel my subscription",
            "Update billing info"
        ],
        'category': ['account', 'order', 'subscription', 'billing'],
        'intent': ['password_reset', 'order_status', 'cancel_subscription', 'update_billing'],
        'response': ['Here is how to reset...', 'Your order is...', 'To cancel...', 'To update billing...']
    }
    return pd.DataFrame(data)

def test_init(model):
    """Test model initialization"""
    assert model.vocab_size == 100
    assert model.embedding_dim == 8
    assert model.max_length == 5
    assert model.model is None
    assert model.tokenizer is not None
    assert model.label_encoder is not None

def test_load_and_prepare_data(model, sample_data, tmp_path):
    """Test data loading and preparation"""
    # Save sample data to temporary CSV
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    
    # Test loading data
    texts, labels = model.load_and_prepare_data(csv_path)
    assert len(texts) == 4
    assert len(labels) == 4
    assert all(isinstance(text, str) for text in texts)

def test_encode_labels(model):
    """Test label encoding"""
    labels = pd.Series(['intent1', 'intent2', 'intent1', 'intent3'])
    encoded_labels, num_classes = model.encode_labels(labels)
    
    assert len(encoded_labels) == 4
    assert num_classes == 3
    assert isinstance(encoded_labels, np.ndarray)

def test_tokenize_texts(model):
    """Test text tokenization"""
    texts = pd.Series([
        "test text one",
        "test text two"
    ])
    padded_sequences = model.tokenize_texts(texts)
    
    assert isinstance(padded_sequences, np.ndarray)
    assert padded_sequences.shape[1] == model.max_length
    assert padded_sequences.shape[0] == 2

def test_split_data(model):
    """Test data splitting"""
    # Create dummy data
    sequences = np.random.rand(100, 5)
    labels = np.random.randint(0, 3, 100)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(sequences, labels)
    
    # Check shapes
    assert len(X_train) == 60  # 60% of data
    assert len(X_val) == 20    # 20% of data
    assert len(X_test) == 20   # 20% of data

def test_one_hot_encode_labels(model):
    """Test one-hot encoding of labels"""
    y_train = np.array([0, 1, 2])
    y_val = np.array([1, 0])
    y_test = np.array([2, 1])
    num_classes = 3
    
    y_train_encoded, y_val_encoded, y_test_encoded = model.one_hot_encode_labels(
        y_train, y_val, y_test, num_classes
    )
    
    assert y_train_encoded.shape == (3, 3)
    assert y_val_encoded.shape == (2, 3)
    assert y_test_encoded.shape == (2, 3)

def test_create_text_classifier(model):
    """Test model creation"""
    num_classes = 3
    classifier = model.create_text_classifier(num_classes)
    
    assert isinstance(classifier, tf.keras.Model)
    assert classifier.output_shape == (None, num_classes)

def test_predict_sample(model, sample_data, tmp_path):
    """Test prediction on a sample text"""
    # Setup model with sample data
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    
    # Train model with minimal data
    texts, labels = model.load_and_prepare_data(csv_path)
    encoded_labels, num_classes = model.encode_labels(labels)
    padded_sequences = model.tokenize_texts(texts)
    X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(padded_sequences, encoded_labels)
    y_train, y_val, y_test = model.one_hot_encode_labels(y_train, y_val, y_test, num_classes)
    
    model.model = model.create_text_classifier(num_classes)
    model.compile_and_train_model(X_train, y_train, X_val, y_val, epochs=1)
    
    # Test prediction
    sample_text = ["How do I reset my password?"]
    prediction = model.predict_sample(sample_text)
    assert isinstance(prediction, str)