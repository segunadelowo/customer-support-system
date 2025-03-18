import joblib
import mlflow
import os
import pickle
from mlflow.tracking import MlflowClient

def download_mlflow_artifacts(run_id, model_name="model", encoder_name="label_encoder", tokenizer_name="tokenizer", output_dir="downloaded_artifacts"):
    """
    Download MLflow model and label encoder artifacts from a specific run.
    
    Parameters:
    - run_id (str): The MLflow run ID containing the artifacts
    - model_name (str): Name of the registered model artifact (default: "model")
    - encoder_name (str): Name of the label encoder artifact (default: "label_encoder")
    - output_dir (str): Directory to save the downloaded artifacts (default: "downloaded_artifacts")
    """
    
    # Set MLflow tracking URI (update this to your MLflow server address)
    mlflow.set_tracking_uri("http://localhost:8881")  # Replace with your MLflow server URI
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize MLflow client
        client = MlflowClient()
        
        # Download and load the model
        model_path = os.path.join(output_dir, f"{model_name}")
        model_artifact_path = client.download_artifacts(run_id, model_name)
        
        # If model is a directory (like for some ML frameworks), copy it directly
        if os.path.isdir(model_artifact_path):
            import shutil
            shutil.copytree(model_artifact_path, model_path, dirs_exist_ok=True)
        
        # Download and load the label encoder
        encoder_path = os.path.join(output_dir, f"{encoder_name}")
        encoder_artifact_path = client.download_artifacts(run_id, encoder_name)
        
        # Load and save the label encoder
        with open(encoder_artifact_path, 'rb') as f:
            #label_encoder = pickle.load(f)
            label_encoder = joblib.load(f)
        with open(encoder_path, 'wb') as f:
            joblib.dump(label_encoder, f)


        tokenizer_path = os.path.join(output_dir, f"{tokenizer_name}")
        tokenizer_artifact_path = client.download_artifacts(run_id, tokenizer_name)
        
        # Load and save the label encoder
        with open(tokenizer_artifact_path, 'rb') as f:
            #label_encoder = pickle.load(f)
            tokenizer = joblib.load(f)
        with open(tokenizer_path, 'wb') as f:
            joblib.dump(tokenizer, f)

        
        print(f"Successfully downloaded artifacts to {output_dir}")
        print(f"Model saved at: {model_path}")
        print(f"Label encoder saved at: {encoder_path}")
        
        return tokenizer, label_encoder
    
    except Exception as e:
        print(f"Error downloading artifacts: {str(e)}")
        return None, None

def main():
    # Example usage
    # Replace with your actual run ID from MLflow
    run_id = "cbec51b4d18042758088ae2718a03912"  # Get this from MLflow UI or tracking server
    
    # Customize these if your artifact names or output directory are different
    model_name = "model"
    encoder_name = "label_encoder.joblib"
    token_name = "tokenizer.joblib"
    output_dir = "downloaded_artifacts"
    
    # Download the artifacts
    tokenizer, label_encoder = download_mlflow_artifacts(
        run_id=run_id,
        model_name=model_name,
        encoder_name=encoder_name,
        tokenizer_name=token_name,
        output_dir=output_dir
    )
    
    # Verify the downloads (optional)
    if tokenizer is not None and label_encoder is not None:
        print("Artifacts downloaded successfully!")
        # You can add additional verification steps here if needed

if __name__ == "__main__":
    main()