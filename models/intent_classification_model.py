import os
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import mlflow
from mlflow.tensorflow import MlflowCallback

#mlflow.tensorflow.autolog(disable=True)
mlflow.set_tracking_uri('http://127.0.0.1:8881')
mlflow.set_experiment('intent_classification')
mlflow.tensorflow.autolog()

class IntentClassificationModel:
    def __init__(self, vocab_size=1000, embedding_dim=16, max_length=10,filepath=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        self.label_encoder = LabelEncoder()
        self.model = None
        self.filepath = filepath
       

    def load_and_prepare_data(self, filepath):
        train_data = pd.read_csv(filepath)
        train_data = train_data[['instruction', 'category', 'intent', 'response']]
        labels = train_data['intent']
        texts = train_data['instruction']
        return texts, labels

    def encode_labels(self, labels):
        encoded_labels = np.array(self.label_encoder.fit_transform(labels))
        num_classes = len(self.label_encoder.classes_)
        return encoded_labels, num_classes

    def tokenize_texts(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_length, padding='post')
        return padded_sequences

    def split_data(self, padded_sequences, encoded_labels):
        X_temp, X_test, y_temp, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def one_hot_encode_labels(self, y_train, y_val, y_test, num_classes):
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)
        y_test = to_categorical(y_test, num_classes)
        return y_train, y_val, y_test

    def create_text_classifier(self, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_length),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def compile_and_train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=64):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc', multi_label=False)
            ]
        )
        self.model.summary()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),

        ]

        with mlflow.start_run() as run:
            # Train the model on GPU
            with tf.device('/GPU:0'):
                history = self.model.fit(
                    X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    verbose=1,
                    callbacks=callbacks.append(MlflowCallback(run))
                )

            self._save_artifacts(self.filepath)
            mlflow.log_artifact(self.filepath / "label_encoder.joblib")
            mlflow.log_artifact(self.filepath / "tokenizer.joblib")
        return history

    def evaluate_model(self, X_test, y_test):
        metrics = self.model.evaluate(X_test, y_test, verbose=0)
        metric_names = ["loss", "accuracy", "precision", "recall", "auc"]
        metrics_dict = {name: value for name, value in zip(metric_names, metrics)}
        return metrics_dict

    def _save_artifacts(self, filepath):
        
        # Create directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)
        joblib.dump(self.label_encoder, os.path.join(filepath, 'label_encoder.joblib'))
        joblib.dump(self.tokenizer, os.path.join(filepath, 'tokenizer.joblib'))

    
    def predict_sample(self, sample_text):
        sample_sequence = self.tokenizer.texts_to_sequences(sample_text)
        sample_padded = tf.keras.preprocessing.sequence.pad_sequences(sample_sequence, maxlen=self.max_length, padding='post')
        prediction = self.model.predict(sample_padded, verbose=0)
        predicted_label_idx = np.argmax(prediction, axis=1)[0]
        predicted_label = self.label_encoder.inverse_transform([predicted_label_idx])[0]
        return predicted_label