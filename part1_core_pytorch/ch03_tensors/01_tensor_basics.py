# part1_core_pytorch/ch03_tensors/01_tensor_basics.py
"""
Chapter 3 Section 3.2-3.6: Tensor Basics
As per the text: "Tensors are the fundamental data structures in deep learning frameworks."
"""
import torch

def main():
    # 3.2.2 Constructing tensors
    a = torch.ones(3)
    print(f"Ones: {a}")
    
    points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
    print(f"Points shape: {points.shape}")  # torch.Size([3, 2])

    # 3.3 Indexing
    print(f"First row: {points[0]}")
    print(f"First column: {points[:, 0]}")
    print(f"Subtensor (skip first row): {points[1:]}")

    # 3.4 Broadcasting
    # As per the text: "Broadcasting is a mechanism that simplifies complex tensor operations"
    x = torch.ones(())      # scalar
    y = torch.ones(3, 1)    # column vector
    z = torch.ones(1, 3)    # row vector
    print(f"Broadcasted shape (y*z): {(y * z).shape}")  # torch.Size([3, 3])

    # 3.5 Named tensors (Experimental)
    # As per the text: "PyTorch 1.3 added named tensors as a prototype feature"
    weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
    print(f"Named tensor: {weights_named.names}")

    # 3.6 Dtype management
    # As per the text: "The default data type for tensors is 32-bit floating-point."
    double_points = torch.ones(10, 2, dtype=torch.double)
    print(f"Dtype: {double_points.dtype}")  # torch.float64
    
    # Casting
    float_points = double_points.to(torch.float)
    print(f"Casted Dtype: {float_points.dtype}")  # torch.float32

if __name__ == "__main__":
    main()