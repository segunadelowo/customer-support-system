import pandas as pd
from pathlib import Path
from models.intent_classification_model import IntentClassificationModel


def main():

    
    filepath_data = '/data/bitext.csv'
    

    texts = [
    "I need help with my account",
    "How do I reset my password",
    "My order is delayed",
    "Can I get a refund",
    "Payment failed again",
    "Where is my delivery",
    "Account login issue",
    "I want to cancel my order",
    "Refund process is slow",
    "Check my payment status"
    ]
    labels = [
    "account_issue", "password_reset", "order_delay", "refund_request",
    "payment_issue", "delivery_status", "account_issue", "order_cancel",
    "refund_request", "payment_issue"
    ]

    artifact_path = Path.cwd() / 'model_artifacts'
    classifier = IntentClassificationModel(filepath=artifact_path)
    texts, labels = classifier.load_and_prepare_data(filepath_data)
    encoded_labels, num_classes = classifier.encode_labels(labels)
    classifier.model = classifier.create_text_classifier(num_classes)
    padded_sequences = classifier.tokenize_texts(texts)
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.split_data(padded_sequences, encoded_labels)
    y_train, y_val, y_test = classifier.one_hot_encode_labels(y_train, y_val, y_test, num_classes)
    
    history = classifier.compile_and_train_model(X_train, y_train, X_val, y_val)
    
    metrics_dict = classifier.evaluate_model(X_test, y_test)
    print("\nTest Metrics:")
    for metric, value in metrics_dict.items():
        print(f"Test {metric.capitalize()}: {value:.4f}")
    
    sample_text = ["I want to reset my password"]
    predicted_label = classifier.predict_sample(sample_text)
    print(f"\nSample Text: '{sample_text[0]}' -> Predicted Label: {predicted_label}")

if __name__ == "__main__":
    main()
