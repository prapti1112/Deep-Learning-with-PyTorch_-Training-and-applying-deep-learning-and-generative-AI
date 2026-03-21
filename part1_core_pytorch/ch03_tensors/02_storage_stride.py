# part1_core_pytorch/ch03_tensors/02_storage_stride.py
"""
Chapter 3 Section 3.8-3.9: Storage & Stride
As per the text: "A PyTorch Tensor instance is a view of such a Storage instance 
that is capable of indexing into that storage using an offset and per-dimension strides."
"""
import torch

def main():
    # 3.8.1 Indexing into storage
    points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
    print(f"Storage: {points.storage()}")
    
    # Modifying storage modifies the tensor
    points_storage = points.storage()
    points_storage[0] = 2.0
    print(f"Modified points: {points}")  # First element changed

    # 3.9.1 Views (No copy)
    second_point = points[1]
    print(f"Shared storage: {points.storage().data_ptr() == second_point.storage().data_ptr()}")  # True

    # 3.9.2 Transposing without copying
    points_t = points.t()
    print(f"Original stride: {points.stride()}")   # (2, 1)
    print(f"Transposed stride: {points_t.stride()}") # (1, 2)
    print(f"Transposed shares storage: {points.storage().data_ptr() == points_t.storage().data_ptr()}") # True

    # 3.9.4 Contiguous tensors
    # As per the text: "A tensor whose values are laid out in the storage starting 
    # from the rightmost dimension onward... is defined as contiguous."
    print(f"Is points contiguous: {points.is_contiguous()}")      # True
    print(f"Is points_t contiguous: {points_t.is_contiguous()}")  # False
    
    # Fixing contiguity (allocates new storage)
    points_t_cont = points_t.contiguous()
    print(f"Is points_t_cont contiguous: {points_t_cont.is_contiguous()}") # True

if __name__ == "__main__":
    main()