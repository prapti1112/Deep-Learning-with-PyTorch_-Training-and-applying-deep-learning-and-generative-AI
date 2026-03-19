# part1_core_pytorch/ch01_intro/check_environment.py
"""
Chapter 1 Exercise 1.7: Environment Verification
As per the text: "We hope it is at least 3.10!"
"""
import sys
import torch

def check_environment():
    print("--- Chapter 1 Environment Check ---")
    
    # 1a. Python Version
    print(f"Python Version: {sys.version}")
    if sys.version_info < (3, 10):
        print("⚠️  WARNING: Text recommends Python 3.10 or later.")
    else:
        print("✅ Python version is compatible.")
    
    # 1b. PyTorch Import & Version
    try:
        print(f"PyTorch Version: {torch.__version__}")
        print("✅ Torch imported successfully.")
    except ImportError:
        print("❌ FAILED: Could not import torch.")
        return

    # 1c. CUDA Availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.get_device_name(0)}")
        print("✅ GPU acceleration is ready.")
    else:
        print("⚠️  No GPU detected. Training will run on CPU (slower).")
        print("   As per text Section 1.6: 'Running on a GPU cuts training time by at least an order of magnitude'.")

if __name__ == "__main__":
    check_environment()