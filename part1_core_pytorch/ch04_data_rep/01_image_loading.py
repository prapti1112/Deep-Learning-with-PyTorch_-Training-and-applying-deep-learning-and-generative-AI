# part1_core_pytorch/ch04_data_rep/01_image_loading.py
import torch
import imageio.v2 as imageio
import numpy as np

def main():
    # 4.1.2 Loading an image file
    # Note: Ensure you have a sample image at this path
    img_arr = imageio.imread('data/p1ch2/bobby.jpg') 
    print(f"NumPy Shape (HWC): {img_arr.shape}")
    
    # 4.1.3 Changing the layout (HWC -> CHW)
    img = torch.from_numpy(img_arr)
    img_chw = img.permute(2, 0, 1)
    print(f"Tensor Shape (CHW): {img_chw.shape}")
    
    # 4.1.4 Normalizing the data
    # Option A: Scale to [0, 1]
    img_float = img_chw.float() / 255.0
    
    # Option B: Standardization (Zero Mean, Unit Std)
    for c in range(3):
        mean = img_float[c].mean()
        std = img_float[c].std()
        img_float[c] = (img_float[c] - mean) / std
        
    print(f"Normalized Range: [{img_float.min():.2f}, {img_float.max():.2f}]")

if __name__ == "__main__":
    main()