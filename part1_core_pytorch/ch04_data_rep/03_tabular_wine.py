# part1_core_pytorch/ch04_data_rep/03_tabular_wine.py
import torch
import numpy as np

def main():
    # 4.3.2 Loading a wine data tensor
    # Note: Requires winequality-white.csv
    try:
        wine_path = "data/p1ch4/tabular-wine/winequality-white.csv"
        wine_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
        wine = torch.from_numpy(wine_numpy)
        
        # 4.3.3 Representing scores (Splitting data vs target)
        data = wine[:, :-1]   # All columns except last
        target = wine[:, -1]  # Last column (quality)
        
        # 4.3.4 One-hot encoding
        # Convert quality (3-9) to 0-6 index for one-hot
        target_int = (target - 3).long() 
        target_onehot = torch.zeros(target_int.shape[0], 7)
        target_onehot.scatter_(1, target_int.unsqueeze(1), 1.0)
        
        print(f"Data Shape: {data.shape}")
        print(f"One-Hot Shape: {target_onehot.shape}")
        print(f"Sample One-Hot: {target_onehot[0]}")
    except Exception as e:
        print(f"Skipping tabular load (data missing): {e}")

if __name__ == "__main__":
    main()