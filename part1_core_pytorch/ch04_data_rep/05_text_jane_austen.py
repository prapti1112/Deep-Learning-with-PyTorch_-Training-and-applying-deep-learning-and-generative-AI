# part1_core_pytorch/ch04_data_rep/05_text_jane_austen.py
import torch

def main():
    # 4.5.2 One-hot-encoding characters
    sample_text = "Hello PyTorch"
    
    # Create tensor (Length of string x Vocabulary Size (ASCII 128))
    letter_t = torch.zeros(len(sample_text), 128)
    
    for i, letter in enumerate(sample_text.lower()):
        letter_index = ord(letter) if ord(letter) < 128 else 0
        letter_t[i][letter_index] = 1
    
    print(f"Text: '{sample_text}'")
    print(f"Encoding Shape: {letter_t.shape}")
    print(f"Non-zero elements (should equal text length): {letter_t.sum().item()}")

if __name__ == "__main__":
    main()