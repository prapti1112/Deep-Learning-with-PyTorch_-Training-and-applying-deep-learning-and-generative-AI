# part1_core_pytorch/ch03_tensors/03_interop_serialization.py
"""
Chapter 3 Section 3.10-3.13: GPU, NumPy, Serialization
As per the text: "PyTorch features seamless interoperability with NumPy... 
and an extensive library of operations on them."
"""
import torch
import numpy as np
import h5py
import os

def main():
    # 3.10 Moving tensors to GPU
    if torch.cuda.is_available():
        points_gpu = torch.tensor([[4.0, 1.0]], device='cuda')
        print(f"GPU Tensor: {points_gpu.device}")
        points_cpu = points_gpu.to(device='cpu')
        print(f"Back to CPU: {points_cpu.device}")
    else:
        print("CUDA not available. Skipping GPU tests.")

    # 3.11 NumPy interoperability
    # As per the text: "The returned array shares the same underlying buffer with the tensor storage."
    points = torch.ones(3, 4)
    points_np = points.numpy()
    points_np[0, 0] = 99.0
    print(f"Tensor after NumPy change: {points[0, 0]}")  # 99.0 (Zero-copy share)

    # 3.13 Serializing tensors (Pickle)
    torch.save(points, 'ourpoints.t')
    points_loaded = torch.load('ourpoints.t')
    print(f"Loaded via torch.load: {points_loaded[0, 0]}")

    # 3.13.1 Serializing to HDF5 (Interoperable)
    # As per the text: "HDF5 is a portable, widely supported format for representing 
    # serialized multidimensional arrays... Python supports HDF5 through the h5py library"
    with h5py.File('ourpoints.hdf5', 'w') as f:
        f.create_dataset('coords', data=points.numpy())
    
    with h5py.File('ourpoints.hdf5', 'r') as f:
        dset = f['coords']
        print(f"Loaded via HDF5: {dset[0, 0]}")
    
    # Cleanup
    os.remove('ourpoints.t')
    os.remove('ourpoints.hdf5')

if __name__ == "__main__":
    main()