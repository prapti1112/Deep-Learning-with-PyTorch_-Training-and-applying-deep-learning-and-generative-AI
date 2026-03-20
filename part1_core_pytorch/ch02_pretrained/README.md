# Chapter 2: Pretrained Networks - Learning Summary

## 1. Value Proposition
*   **Immediate Application:** Using off-the-shelf models allows practitioners to "jump-start a deep learning project" without training from scratch (Section 2 Introduction).
*   **Inference Mastery:** Learn the mechanics of running trained models on real-world data, including preprocessing, eval mode, and output interpretation (Section 2.1.5).
*   **Generative & Multimodal Exposure:** Gain hands-on experience with diffusion models (inpainting) and multimodal models (image captioning) before understanding their internal architecture (Sections 2.2 & 2.4).

## 2. Key Concepts & Definitions

### 2.1 Image Recognition (TorchVision)
*   **Inference:** "In the field of deep learning, the term used for running a trained model on new data is called inference" (Section 2.1.5).
*   **Preprocessing:** Inputs must match training conditions (resize, crop, normalize). "These need to match what was presented to the network during training if we want the network to produce meaningful answers" (Section 2.1.4).
*   **Eval Mode:** "To do inference, we need to put the network in eval mode" to ensure layers like dropout and batch normalization behave correctly (Section 2.1.5).
*   **Models Covered:**
    *   **AlexNet:** A historical milestone convolutional network (Section 2.1.2).
    *   **Vision Transformer (ViT):** Uses self-attention mechanisms; demonstrates significantly lower error rates than AlexNet (Section 2.1.3).

### 2.2 Image Generation & Editing (Diffusion)
*   **Inpainting:** "Text-guided, mask-aware, iterative refinement that can make localized, realistic edits without retraining a model" (Section 2.2).
*   **The Mask:** "Black pixels mark protected areas that are copied back unchanged... White pixels mark the editable, 'primed' region where the model may repaint" (Section 2.2).
*   **Pipeline:** Bundles trained components (tokenizer, UNet, VAE, scheduler) into a simple function (Section 2.2.2).

### 2.3 Model Zoo (Hugging Face)
*   **Definition:** "A model zoo refers to a collection or repository of pretrained models... Hugging Face has become increasingly popular as a high-level wrapper" (Section 2.3).
*   **Transformers:** The `transformers` package holds a host of transformer models (e.g., GPT-2 for text generation) (Section 2.3).

### 2.4 Multimodal Models (BLIP)
*   **Definition:** "BLIP is a multimodal model; that is, it is designed to handle different types of data (or modalities)—text and images, in this case—as its input" (Section 2.4).
*   **Architecture:** Uses an **image encoder** to generate numerical representations and a **text decoder** to generate coherent sentences (Section 2.4).

## 3. Sample Programs & Code Snippets

### 3.1 Image Recognition (AlexNet/ViT)
```python
# Loading pretrained weights
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# Preprocessing Pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference Workflow
vit.eval()
with torch.no_grad():
    out = vit(batch_t)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
```

## 4. Exercises (Section 2.6)

| Exercise | Task | Status |
|---|---|---|
2.1 | Feed golden retriever image into horse-to-zebra model. Prepare image & observe output. | [x]
2.2 | Search Hugging Face hub. Count models, find an interesting one, understand documentation. | [x]
2.2c | Bookmark the project to revisit after finishing the book. | [x]

## 5. Visual Flows
Inference Process
```
[IMAGE] → [RESIZE/CROP/NORMALIZE] → [FORWARD PASS] → [SCORES] → [SOFTMAX/ARGMAX] → [LABEL]
```

Inpainting Inputs
```
[IMAGE] + [MASK] + [PROMPT] → [DIFFUSION MODEL] → [EDITED IMAGE]
```

BLIP Captioning
```
[IMAGE] → [IMAGE ENCODER] → [EMBEDDING] → [TEXT DECODER] → [CAPTION]
```