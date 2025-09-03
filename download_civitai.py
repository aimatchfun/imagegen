import requests
import os
from pathlib import Path


def download_civitai_model(model_url, token, output_dir="models"):
    """
    Downloads a model from CivitAI using the provided token.
    
    Args:
        model_url (str): The CivitAI model download URL
        token (str): The CivitAI API token
        output_dir (str): Directory to save the downloaded model
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract model name from URL for filename
    model_id = model_url.split('/models/')[1].split('?')[0]
    filename = f"civitai_model_{model_id}.safetensors"
    output_path = os.path.join(output_dir, filename)
    
    # Headers with token for authentication
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print(f"Downloading model from CivitAI...")
    print(f"URL: {model_url}")
    print(f"Output: {output_path}")
    
    try:
        # Make the request with authentication
        response = requests.get(model_url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Get file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        
        # Download the file with progress
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print(f"\nDownload completed successfully!")
        print(f"Model saved to: {output_path}")
        return output_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")
        return None


if __name__ == "__main__":
    # CivitAI model download configuration
    CIVITAI_TOKEN = os.getenv('CIVITAI_TOKEN')
    MODEL_URL = "https://civitai.com/api/download/models/1199750?type=Model&format=SafeTensor&size=pruned&fp=fp16"
    
    if CIVITAI_TOKEN:
        print("Downloading CivitAI model...")
        download_civitai_model(MODEL_URL, CIVITAI_TOKEN)
    else:
        print("CIVITAI_TOKEN environment variable not set.")
        print("Please set CIVITAI_TOKEN environment variable to download from CivitAI.")
        print("Example: export CIVITAI_TOKEN='your_token_here'")
