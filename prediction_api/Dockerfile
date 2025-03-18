# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project folders
COPY ./model ./model
COPY ./app.py .
COPY ./label_encoder.joblib .
COPY ./tokenizer.joblib .
COPY ./sentiment ./sentiment  



# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]   
