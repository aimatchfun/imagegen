import os
import torch
from diffusers import StableDiffusionPipeline
from typing import Optional
from download_civitai import download_civitai_model

# Configuration
MODEL_DIR = "models"
MODEL_FILENAME = "civitai_model_439889.safetensors"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Check if model exists, download if not
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}. Downloading...")
    CIVITAI_TOKEN = os.getenv('CIVITAI_TOKEN')
    MODEL_URL = "https://civitai.com/api/download/models/439889?type=Model&format=SafeTensor&size=pruned&fp=fp16"
    if CIVITAI_TOKEN:
        download_civitai_model(MODEL_URL, CIVITAI_TOKEN, MODEL_DIR)
    else:
        raise ValueError("CIVITAI_TOKEN environment variable not set. Please set it to download the model.")

# Load the model pipeline
pipe = StableDiffusionPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=torch.float16,
    use_safetensors=True
).to(DEVICE)

# Optimize for performance
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()


def generate_image(prompt: str, negative_prompt: Optional[str] = None, 
                  num_inference_steps: int = 30, guidance_scale: float = 7.5,
                  width: int = 512, height: int = 512) -> bytes:
    """
    Generate an image from a text prompt using the loaded model.
    
    Args:
        prompt: The text prompt to generate the image
        negative_prompt: Text to avoid in the generated image
        num_inference_steps: Number of denoising steps
        guidance_scale: Scale for classifier-free guidance
        width: Width of the output image
        height: Height of the output image
        
    Returns:
        bytes: The generated image in bytes format
    """
    try:
        # Generate the image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height
        ).images[0]
        
        # Convert to bytes
        from io import BytesIO
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
        
    except Exception as e:
        print(f"Error generating image: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    try:
        print("Generating test image...")
        image_bytes = generate_image(
            prompt="a cute pony with rainbow mane, digital art",
            negative_prompt="blurry, low quality, deformed"
        )
        
        # Save to file for testing
        with open("generated_pony.png", "wb") as f:
            f.write(image_bytes)
        print("Image generated successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
