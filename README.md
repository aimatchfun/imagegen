# Image Generation with Stable Diffusion and CivitAI Models

This project provides a pipeline to download models from CivitAI and generate images using Stable Diffusion.

## Features
- Download models from CivitAI using API tokens
- Generate high-quality images from text prompts
- Optimized for GPU performance with xformers and CUDA
- Ready for deployment with Docker and RunPod

## Prerequisites
- Python 3.11
- NVIDIA GPU with CUDA 12.1 support (for local development)
- CivitAI API token (set as `CIVITAI_TOKEN` environment variable)

## Installation

### Local Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/imagegen.git
cd imagegen
```

2. Create and activate a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set your CivitAI token:
```bash
export CIVITAI_TOKEN='your_token_here'
```

### Docker Setup
```bash
docker build -t imagegen .
docker run --gpus all -e CIVITAI_TOKEN='your_token_here' imagegen
```

## Usage

### Downloading Models
Run the download script to get the pony model from CivitAI:
```bash
python download_civitai.py
```

### Generating Images
Use the handler to generate images:
```bash
python handler.py
```

This will create a sample image `generated_pony.png` with a default prompt.

### Custom Generation
Modify `handler.py` to use your own prompts:
```python
image_bytes = generate_image(
    prompt="your custom prompt here",
    negative_prompt="elements to avoid"
)
```

## Configuration

### Environment Variables
- `CIVITAI_TOKEN`: Your CivitAI API token (required for downloads)

### Model Settings
- Model files are saved to `models/` directory
- Default model: Pony Diffusion (CivitAI Model ID: 439889)

## Deployment

The project includes GitHub Actions workflows for:
- CI: Testing and dependency updates
- CD: Docker image building and deployment

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) for details.