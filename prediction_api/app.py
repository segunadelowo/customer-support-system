# app.py
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = FastAPI()

# Define request body model
class PredictionRequest(BaseModel):
    ticket: str

# Load the TensorFlow model and artifacts
try:
    # Assuming the model is saved in SavedModel format
    model = tf.keras.models.load_model('/app/model')
    label_encoder = joblib.load('/app/label_encoder.joblib')
    tokenizer = joblib.load('/app/tokenizer.joblib')
    vader = SentimentIntensityAnalyzer()

except Exception as e:
    raise Exception(f"Failed to load model: {str(e)}")

def preprocess_input(ticket: str):
    # Tokenize the input
    sequence = tokenizer.texts_to_sequences([ticket])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=10, padding='post')
    return padded_sequence

def get_sentiment(ticket):

    scores = vader.polarity_scores(ticket)
    return scores['compound']


@app.post("/predict")
async def predict(request: PredictionRequest):

    try:
        # Preprocess the input
        processed_input = preprocess_input(
            request.ticket,
        )
        
        # Make prediction
        prediction = model.predict(processed_input)
        predicted_label_idx = np.argmax(prediction, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]
        sentiment_score = get_sentiment(request.ticket)

        return {
            "status": "success",
            "predicted_label": predicted_label,
            "prediction_confidence": float(prediction[0][predicted_label_idx]),
            "sentiment_score": sentiment_score
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)