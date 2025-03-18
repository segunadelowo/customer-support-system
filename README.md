# Customer Support System

The Customer Support System is a multi-component application designed to automate and enhance customer support operations using machine learning, sentiment analysis, and retrieval augmented generation (RAG). The system processes incoming support tickets, classifies their intent, analyzes sentiment, and either provides automated responses or routes the tickets to the appropriate support queue.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
  - [Intent Classification Module](#intent-classification-module)
  - [Prediction API](#prediction-api)
  - [Ticket Processor](#ticket-processor)
  - [Ticket API](#ticket-api)
  - [RAG System](#rag-system)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The project leverages both Python and Go to deliver a robust customer support system that includes:
- **Machine Learning Models**: For intent classification, using TensorFlow and scikit-learn.
- **APIs**: A FastAPI-based prediction API to classify tickets and obtain sentiment scores, and a Go-based ticket submission API that interacts with RabbitMQ.
- **Ticket Processing**: A ticket processor that consumes queue messages, makes decisions on routing or automation, and integrates with a retrieval augmented generation (RAG) system for knowledge-based recommendations.
- **RAG System**: Uses a pre-built knowledge base and Anthropic API to generate concise resolution recommendations using context from stored documents.

---

## Architecture

The system is composed of the following main parts:

1. **Intent Classification Module** (`models` directory)
   - Contains modules to load data, train a text classification model, and manage tokenization and label encoding.
   - Files: `intent_classification_model.py`, `train_model.py`.

2. **Prediction API** (`prediction_api` directory)
   - A FastAPI application that loads the trained model, tokenizer, and label encoder.
   - Provides endpoints to predict the intent of a ticket and compute a sentiment score.
   - File: `app.py`.

3. **RAG System and Ticket Processing** (`processor` directory)
   - The RAG system (`rag_system.py`) retrieves context from a knowledge base stored in a CSV and enriched in a ChromaDB instance.
   - The Ticket Processor (`ticket_processor.py`) consumes messages from a RabbitMQ queue, classifies tickets, and either automates responses or routes them for manual intervention.

4. **Ticket API** (`ticket_api` directory)
   - A Go-based REST API that accepts ticket submissions, serializes them, and publishes messages to a RabbitMQ queue for asynchronous processing.
   - File: `main.go`.

---

## Components

### Intent Classification Module

- **Purpose**: Train and evaluate a text classification model to determine the intent behind customer support instructions.
- **Key Files**:
  - `models/intent_classification_model.py`: Defines the `IntentClassificationModel` class which handles data loading, preprocessing, model definition, training, artifact saving, and prediction.
  - `models/train_model.py`: Script to execute training, evaluation, and sample prediction on support tickets.
  
### Prediction API

- **Purpose**: Provide an HTTP API using FastAPI to predict the intent of a given support ticket and analyze sentiment using NLTK’s VADER.
- **Key File**:
  - `prediction_api/app.py`: Loads the model and required artifacts and exposes endpoints:
    - POST `/predict`: Predicts intent and returns corresponding confidence and sentiment score.
    - GET `/health`: Health check endpoint.

### Ticket Processor

- **Purpose**: Process tickets from a message queue, classify them, decide on routing, and trigger automated responses using the RAG system.
- **Key Files**:
  - `processor/rag_system.py`: Implements a RAG system that encodes documents, manages a knowledge base stored in CSV, and generates recommendations.
  - `processor/ticket_processor.py`: Consumes support ticket messages from RabbitMQ, classifies them using the Prediction API, and routes or automates responses based on predefined logic.

### Ticket API

- **Purpose**: Accept incoming support tickets via a RESTful API and publish them to a RabbitMQ queue.
- **Key File**:
  - `ticket_api/main.go`: A Go application built with Gin that listens for ticket submissions on port 8080 and enqueues them for processing.

---

## Setup and Installation

1. **Clone the Repository**

   ```sh
   git clone <repository-url>
   cd customer-support-system
   ```

2. **Install Python Dependencies**

   Ensure you have Python 3.10 or higher. Install dependencies using pip:

   ```sh
   pip install -r requirements.txt
   ```

   Alternatively, if using [pyproject.toml](pyproject.toml), configure your environment accordingly.

3. **Set Up RabbitMQ**

   Install and run RabbitMQ on your local machine or configure the environment variable `RABBITMQ_URL` to point to your RabbitMQ instance.

4. **Configure Environment Variables**

   For the RAG System, set the environment variable for Anthropic API key:

   ```sh
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

5. **Running the Ticket API (Go Service)**

   Navigate to the `ticket_api` directory and run the service:

   ```sh
   cd ticket_api
   go mod tidy
   go run main.go
   ```

6. **Running the Prediction API**

   Run the FastAPI application:

   ```sh
   uvicorn prediction_api.app:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Train the Model**

   Execute the training script to build and evaluate the intent classification model:

   ```sh
   python models/train_model.py
   ```

8. **Start the Ticket Processor**

   Finally, start the ticket processor to begin consuming messages from RabbitMQ:

   ```sh
   python processor/ticket_processor.py
   ```

---

## Usage

- **Submit a Support Ticket**:  
  Use the Go-based Ticket API to submit a support ticket via HTTP POST to `http://localhost:8080/tickets`. The ticket details will be enqueued for processing.

- **Prediction API**:  
  For immediate classification and sentiment analysis, use the FastAPI endpoint at `http://localhost:8000/predict` with a JSON payload:
  ```json
  {
    "ticket": "I need help with my account."
  }
  ```

- **Ticket Processing**:  
  The ticket processor automatically consumes messages from RabbitMQ, uses the Prediction API and RAG System to determine automatic responses or routing, and republishes them to specific queues.

---

## Dependencies

The main dependencies include:
- Python packages:
  - numpy, pandas, scikit-learn, joblib
  - tensorflow, keras, tensorflow-metal
  - fastAPI, uvicorn
  - pika (for RabbitMQ interactions)
  - nltk (with vader lexicon for sentiment analysis)
  - sentence-transformers
  - chromadb
- Go modules:
  - Gin framework for HTTP routing
  - amqp for RabbitMQ integration

See [pyproject.toml](pyproject.toml) for a complete list of Python dependencies.

---

## Project Structure

```
customer-support-system/
├── chroma_db/               # ChromaDB artifacts
├── data/                    # CSV data files for knowledge base and training
├── downloaded_artifacts/    # Pre-trained artifacts (models, encoders, tokenizers)
├── model_artifacts/         # Output artifacts for trained models
├── models/                  # ML model definitions and training scripts
│   ├── intent_classification_model.py
│   └── train_model.py
├── prediction_api/          # FastAPI application for predictions
│   └── app.py
├── processor/               # RAG system and ticket processing modules
│   ├── rag_system.py
│   └── ticket_processor.py
├── setup/                   # Setup scripts for artifacts (e.g., MLflow integration)
├── ticket_api/              # Go-based ticket submission API
│   └── main.go
├── pyproject.toml           # Python project configuration and dependencies
└── README.md                # This file
```

---

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your improvements. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is open source and available under the MIT License.

---

Enjoy building efficient customer support workflows!
