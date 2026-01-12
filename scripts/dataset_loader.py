"""
Dataset Loader for Brain Tumor Classification
==============================================

Loads images from the organized dataset folders.

Author: Brain Tumor Detection Team
Date: January 2026
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np


class BrainTumorDataset(Dataset):
    """
    Brain Tumor Classification Dataset
    
    Supports 4 classes:
    - 0: glioma
    - 1: meningioma
    - 2: notumor
    - 3: pituitary
    """
    
    def __init__(self, root_dir, split='Training', transform=None):
        """
        Args:
            root_dir: Path to dataset root (e.g., 'C:/Users/david/Downloads/1512427/archive')
            split: 'Training' or 'Testing'
            transform: Data augmentation transforms
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Class mapping
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} {split} samples")
        self._print_class_distribution()
    
    def _load_samples(self):
        """Load all image paths and their labels"""
        split_dir = self.root_dir / self.split
        
        for class_name in self.classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            # Get all image files
            for img_path in class_dir.glob('*.jpg'):
                label = self.class_to_idx[class_name]
                self.samples.append((str(img_path), label))
            
            # Also check for png files
            for img_path in class_dir.glob('*.png'):
                label = self.class_to_idx[class_name]
                self.samples.append((str(img_path), label))
    
    def _print_class_distribution(self):
        """Print distribution of classes"""
        class_counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] += 1
        
        print(f"\nClass distribution ({self.split}):")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'classification_label': label
        }


def get_transforms(split='train', img_size=224):
    """
    Get data augmentation transforms
    
    Args:
        split: 'train' or 'val/test'
        img_size: Target image size
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(data_root, batch_size=16, num_workers=4, img_size=224):
    """
    Create train and test dataloaders
    
    Args:
        data_root: Root directory of dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes
        img_size: Image size
    
    Returns:
        train_loader, test_loader
    """
    # Create datasets
    train_dataset = BrainTumorDataset(
        root_dir=data_root,
        split='Training',
        transform=get_transforms('train', img_size)
    )
    
    test_dataset = BrainTumorDataset(
        root_dir=data_root,
        split='Testing',
        transform=get_transforms('test', img_size)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Test the dataset loader
    data_root = r'C:\Users\david\Downloads\1512427\archive'
    
    print("Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(data_root, batch_size=8)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading a batch
    print("\nLoading a sample batch...")
    for batch in train_loader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Labels shape: {batch['classification_label'].shape}")
        print(f"Labels: {batch['classification_label']}")
        break
