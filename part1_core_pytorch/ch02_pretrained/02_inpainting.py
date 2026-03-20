# part1_core_pytorch/ch02_pretrained/02_inpainting.py
"""
Chapter 2 Section 2.2: Generating and Editing Images
As per the text: "The mask we pass to the pretrained diffusion model is a digital stencil."
"""
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

def main():
    # 1. Setup Device 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("Warning: Inpainting on CPU will be extremely slow.")

    # 2. Load Pipeline 
    # As per the text: "A pipeline bundles all the parts needed for generation"
    print("Loading Stable Diffusion Inpainting Pipeline...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # 3. Load Inputs 
    # Note: Ensure horse.jpg and horse_mask.jpg exist in data/p1ch2/
    try:
        img = Image.open("data/p1ch2/horse.jpg").convert("RGB")
        mask_img = Image.open("data/p1ch2/horse_mask.jpg").convert("RGB")
    except FileNotFoundError:
        print("Image/Mask not found. Please ensure horse.jpg and horse_mask.jpg exist.")
        return

    # 4. Define Prompt 
    prompt = "a zebra replacing the original horse, same pose, same lighting, background unchanged"
    negative = "distorted background, blurry, text, watermark"

    # 5. Generate 
    print("Generating image...")
    out = pipe(
        prompt=prompt, 
        image=img, 
        mask_image=mask_img, 
        negative_prompt=negative,
        guidance_scale=7.5, 
        strength=0.8,
        generator=torch.Generator(device).manual_seed(42) # For reproducibility
    )

    # 6. Save Output
    out.images[0].save("data/p1ch2/zebra_result.jpg")
    print("Result saved to data/p1ch2/zebra_result.jpg")

if __name__ == "__main__":
    main()