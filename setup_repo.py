#!/usr/bin/env python3
"""
Script to set up the directory structure for the Deep Learning with PyTorch 48-Hour Sprint.
Usage: python setup_sprint_repo.py --base-dir ./deep-learning-pytorch-sprint
"""

import argparse
import os
from pathlib import Path

# Define the directory structure relative to the base directory
DIRECTORIES = [
    "data/raw",
    "data/processed",
    "data/external",
    "part1_core_pytorch/ch01_intro",
    "part1_core_pytorch/ch02_pretrained",
    "part1_core_pytorch/ch03_tensors",
    "part1_core_pytorch/ch04_data_rep",
    "part1_core_pytorch/ch05_mechanics",
    "part1_core_pytorch/ch06_nn_fit",
    "part1_core_pytorch/ch07_images",
    "part1_core_pytorch/ch08_convolutions",
    "part2_applications/ch09_transformers",
    "part2_applications/ch10_diffusion",
    "part2_applications/ch11_cancer_intro",
    "part2_applications/ch12_data_loading",
    "part2_applications/ch13_classification",
    "part2_applications/ch14_metrics_aug",
    "part2_applications/ch15_segmentation",
    "part2_applications/ch16_distributed",
    "part2_applications/ch17_deployment",
    "utils",
    "docs",
]

# Define files to create with optional initial content
FILES = {
    "README.md": """# Deep Learning with PyTorch: 48-Hour Sprint

## Goal
Complete the Master Index and TODO list from "Deep Learning with PyTorch, Second Edition".

## Structure
- `part1_core_pytorch/`: Foundations (Chapters 1-8)
- `part2_applications/`: Real-world Projects (Chapters 9-17)
- `data/`: Raw and processed datasets
- `utils/`: Shared helper functions
""",
    "requirements.txt": """torch
torchvision
torchaudio
matplotlib
numpy
pandas
jupyter
tensorboard
huggingface_hub
transformers
diffusers
""",
    ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.env

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints
""",
    "TODO.md": """# Sprint Progress Tracker

| Chapter | Type | Item Name / Description | Location / File Reference | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Ch 1** | Code | Python Interactive Checks | `torch.cuda.is_available()`, imports | [ ] |
| **Ch 1** | Code | Jupyter Notebook Boilerplate | `%matplotlib inline`, `torch.manual_seed` | [ ] |
| **Ch 3** | Code | Tensor Basics Notebook | `part1_core_pytorch/ch03_tensors/` | [ ] |
| **Ch 5** | Code | Autograd Implementation | `part1_core_pytorch/ch05_mechanics/` | [ ] |
| **Ch 8** | Code | CNN Training Script | `part1_core_pytorch/ch08_convolutions/` | [ ] |
| **Ch 13** | Code | Training Application | `part2_applications/ch13_classification/` | [ ] |
| **Ch 17** | Code | Deployment Example | `part2_applications/ch17_deployment/` | [ ] |

*Note: Populate this table with the full Master Index list.*
""",
}

def create_structure(base_path: Path):
    """Creates directories and files based on the configuration."""
    print(f"🚀 Setting up sprint repository at: {base_path.resolve()}")
    
    # Create Directories
    for dir_path in DIRECTORIES:
        full_path = base_path / dir_path
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
            
            # Add .gitkeep to data folders to ensure they are tracked by git
            if "data" in dir_path and not list(full_path.iterdir()):
                (full_path / ".gitkeep").touch()
        except Exception as e:
            print(f"❌ Failed to create {dir_path}: {e}")

    # Create Files
    for file_name, content in FILES.items():
        full_path = base_path / file_name
        try:
            if not full_path.exists():
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Created file: {file_name}")
            else:
                print(f"⚠️  Skipped existing file: {file_name}")
        except Exception as e:
            print(f"❌ Failed to create {file_name}: {e}")

    print("\n🎉 Setup complete! Ready to start your 48-hour sprint.")
    print(f"👉 Next step: Open {base_path / 'TODO.md'} and start coding.")

def main():
    parser = argparse.ArgumentParser(description="Setup PyTorch Sprint Repository")
    parser.add_argument(
        "--base-dir", 
        type=str, 
        default=".",
        help="Base directory path for the project (default: ./deep-learning-pytorch-sprint)"
    )
    args = parser.parse_args()

    base_path = Path(args.base_dir)
    
    if base_path.exists() and any(base_path.iterdir()):
        response = input(f"⚠️  Directory '{base_path}' is not empty. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    create_structure(base_path)

if __name__ == "__main__":
    main()