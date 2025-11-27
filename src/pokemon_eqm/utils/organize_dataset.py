#!/usr/bin/env python3
"""
Organize Pokemon dataset for training with class labels.
Creates a training directory with only base, mega, regional, and gigantamax forms.
"""
import os
import shutil
from pathlib import Path

def organize_for_training(
    raw_dir="data/raw",
    train_dir="data/train",
    categories=["base", "mega", "regional", "gigantamax"]
):
    """
    Organize Pokemon images for training.
    
    Args:
        raw_dir: Directory containing downloaded images organized by category
        train_dir: Output directory for training (will have class subdirectories)
        categories: List of categories to include in training
    """
    print(f"Organizing dataset from {raw_dir} to {train_dir}...")
    
    # Create train directory
    os.makedirs(train_dir, exist_ok=True)
    
    total_count = 0
    for category in categories:
        source_dir = os.path.join(raw_dir, category)
        if not os.path.exists(source_dir):
            print(f"Warning: {source_dir} does not exist, skipping...")
            continue
            
        target_dir = os.path.join(train_dir, category)
        
        # Remove existing symlinks/files if they exist
        if os.path.exists(target_dir):
            if os.path.islink(target_dir):
                os.unlink(target_dir)
            else:
                shutil.rmtree(target_dir)
        
        # Create symlink to avoid duplicating data
        os.symlink(os.path.abspath(source_dir), target_dir)
        
        count = len(os.listdir(source_dir))
        total_count += count
        print(f"  {category}: {count} images")
    
    print(f"\nTotal training images: {total_count}")
    print(f"Training directory created at: {train_dir}")
    print(f"Classes: {', '.join(categories)}")

if __name__ == "__main__":
    organize_for_training()
