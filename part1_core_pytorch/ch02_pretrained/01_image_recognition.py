"""
Chapter 2 Section 2.1: Pretrained Network for Image Recognition
As per the text: "Using an off-the-shelf model can be a quick way to jump-start a deep learning project."
"""
import torch
from torchvision import models, transforms
from PIL import Image

def main():
    # 1. As per the text: "vit_b_16 sports 88.6 million parameters"
    print("Loading Vision Transformer...")
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    
    # 2. As per the text: "These need to match what was presented to the network during training"
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 3. Note: Ensure 'data/p1ch2/bobby.jpg' exists or replace with a valid path
    try:
        img = Image.open("data/p1ch2/bobby.jpg").convert("RGB")
    except FileNotFoundError:
        print("Image not found. Please place bobby.jpg in data/p1ch2/")
        return

    # 4. Prepare Batch (Section 2.1.4)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    
    # 5. As per the text: "To do inference, we need to put the network in eval mode"
    vit.eval()
    
    with torch.no_grad():
        out = vit(batch_t)
    
    # 6. As per the text: "Use torch.nn.functional.softmax... to normalize our outputs"
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    
    # 7. Note: Ensure 'imagenet_classes.txt' exists or download from book repo
    try:
        with open('data/p1ch2/imagenet_classes.txt') as f:
            labels = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Labels file not found. Skipping label lookup.")
        labels = [f"Class_{i}" for i in range(1000)]

    # Get Top Prediction
    _, index = torch.max(out, 1)
    print(f"Prediction: {labels[index.item()]} ({percentage[index.item()].item():.2f}%)")

    # Top 5 Predictions
    _, indices = torch.sort(out, descending=True)
    print("\nTop 5 Predictions:")
    for idx in indices[0][:5]:
        print(f"- {labels[idx].title()}: {percentage[idx].item():.2f}%")

if __name__ == "__main__":
    main()