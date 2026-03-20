# part1_core_pytorch/ch02_pretrained/03_model_zoos.py
"""
Chapter 2 Section 2.3 & 2.4: Model Zoo and Scene Description
As per the text: "Hugging Face has become increasingly popular as a high-level wrapper."
"""
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def text_generation():
    # Section 2.3: Text Generation (GPT-2)
    print("\n--- Text Generation (GPT-2) ---")
    generator = pipeline('text-generation', model='gpt2')
    result = generator("AI models are so smart they can replace my", max_length=10)
    print(f"Output: {result[0]['generated_text']}")

def image_captioning():
    # Section 2.4: Image Captioning (BLIP)
    print("\n--- Image Captioning (BLIP) ---")
    # As per the text: "BLIP is a multimodal model... designed to handle different types of data"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    try:
        # Reuse horse image from previous example
        image = Image.open("../../data/p1ch2/horse.jpg").convert("RGB")
    except FileNotFoundError:
        print("Image not found for captioning.")
        return

    # Inference
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"Caption: {caption}")

if __name__ == "__main__":
    text_generation()
    image_captioning()