"""
Combined Dataset Loader - Archive Images + .mat Files
======================================================

Loads both preprocessed images and MATLAB .mat files for comprehensive training.

Author: MediMind BTIS Team
Date: January 2026
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm


class MatBrainTumorDataset(Dataset):
    """
    Brain Tumor Dataset from .mat files
    
    Loads MRI scans from MATLAB .mat format files.
    Each .mat file contains:
    - cjdata.image: MRI scan image
    - cjdata.label: Tumor type label
    - cjdata.tumorBorder: Tumor boundary (optional)
    """
    
    def __init__(self, mat_folders, transform=None, cache_images=False):
        """
        Args:
            mat_folders: List of folder paths containing .mat files
            transform: Data augmentation transforms
            cache_images: Whether to cache images in memory (faster but uses RAM)
        """
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None
        
        # Collect all .mat files
        self.mat_files = []
        print(f"\n📂 Loading .mat files...")
        
        for folder in mat_folders:
            folder_path = Path(folder)
            if folder_path.exists():
                mat_files = sorted(folder_path.glob('*.mat'))
                self.mat_files.extend(mat_files)
                print(f"   Found {len(mat_files)} files in {folder_path.name}")
        
        print(f"   Total .mat files: {len(self.mat_files)}")
        
        # Load labels and verify files
        self.valid_files = []
        self.labels = []
        
        print(f"\n🔍 Verifying .mat files...")
        for mat_file in tqdm(self.mat_files, desc="   Checking files"):
            try:
                with h5py.File(str(mat_file), 'r') as f:
                    # Extract label from cjdata/label
                    if 'cjdata' in f and 'label' in f['cjdata']:
                        label = int(f['cjdata']['label'][0, 0])
                        self.valid_files.append(mat_file)
                        self.labels.append(label)
                        
                        # Cache image if requested
                        if cache_images:
                            image = f['cjdata']['image'][:]
                            self.image_cache[str(mat_file)] = image
                        
            except Exception as e:
                # Skip corrupted files
                continue
        
        print(f"   ✅ Valid files: {len(self.valid_files)}")
        self._print_label_distribution()
    
    def _print_label_distribution(self):
        """Print distribution of labels"""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"\n   Label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"      Label {label}: {count} images")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        mat_file = self.valid_files[idx]
        label = self.labels[idx]
        
        # Load image
        if self.cache_images and str(mat_file) in self.image_cache:
            image = self.image_cache[str(mat_file)]
        else:
            with h5py.File(str(mat_file), 'r') as f:
                image = f['cjdata']['image'][:]
        
        # Convert to PIL Image
        # Normalize to 0-255 range
        if image.max() > 0:
            image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)
        
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Map labels to 0-3 (glioma, meningioma, notumor, pituitary)
        # .mat files use labels 1-3 (no class 4 - pituitary in this dataset)
        # Map: 1->0 (glioma), 2->1 (meningioma), 3->2 (notumor)
        label_mapped = int(label) - 1
        if label_mapped < 0 or label_mapped > 3:
            label_mapped = 0  # Default to glioma if unknown
        
        return {
            'image': image,
            'classification_label': label_mapped
        }


class ImageBrainTumorDataset(Dataset):
    """
    Brain Tumor Dataset from image files (jpg/png)
    
    Uses the archive dataset structure.
    """
    
    def __init__(self, root_dir, split='Training', transform=None):
        """
        Args:
            root_dir: Path to dataset root (e.g., 'datasets/archive')
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
        
        print(f"\n📁 Loaded {len(self.samples)} {split} images from archive")
        self._print_class_distribution()
    
    def _load_samples(self):
        """Load all image paths and their labels"""
        split_dir = self.root_dir / self.split
        
        for class_name in self.classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            
            # Get all image files
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                for img_path in class_dir.glob(ext):
                    label = self.class_to_idx[class_name]
                    self.samples.append((str(img_path), label))
    
    def _print_class_distribution(self):
        """Print distribution of classes"""
        class_counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] += 1
        
        print(f"   Class distribution:")
        for cls, count in class_counts.items():
            print(f"      {cls:12s}: {count} images")
    
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


def create_combined_dataloaders(archive_root='datasets/archive',
                                mat_folders=None,
                                batch_size=32,
                                num_workers=8,
                                img_size=224,
                                cache_mat_images=False):
    """
    Create train and test dataloaders combining archive images and .mat files
    
    Args:
        archive_root: Root directory of archive dataset
        mat_folders: List of folders containing .mat files
        batch_size: Batch size for training
        num_workers: Number of worker processes
        img_size: Image size
        cache_mat_images: Cache .mat images in memory (faster, uses ~3-4GB RAM)
    
    Returns:
        train_loader, test_loader, dataset_info
    """
    print("=" * 70)
    print("📊 Creating Combined Dataset Loaders")
    print("=" * 70)
    
    # Default .mat folders if not specified
    if mat_folders is None:
        mat_folders = [
            'datasets/brainTumorDataPublic_1-766',
            'datasets/brainTumorDataPublic_767-1532',
            'datasets/brainTumorDataPublic_1533-2298',
            'datasets/brainTumorDataPublic_2299-3064'
        ]
    
    # Transforms
    train_transform = get_transforms('train', img_size)
    test_transform = get_transforms('test', img_size)
    
    # Load archive datasets
    archive_train = ImageBrainTumorDataset(
        root_dir=archive_root,
        split='Training',
        transform=train_transform
    )
    
    archive_test = ImageBrainTumorDataset(
        root_dir=archive_root,
        split='Testing',
        transform=test_transform
    )
    
    # Load .mat datasets
    mat_train = MatBrainTumorDataset(
        mat_folders=mat_folders,
        transform=train_transform,
        cache_images=cache_mat_images
    )
    
    # Combine datasets
    # Use 90% of .mat files for training, 10% for testing
    mat_train_size = int(0.9 * len(mat_train))
    mat_test_size = len(mat_train) - mat_train_size
    
    mat_train_subset, mat_test_subset = torch.utils.data.random_split(
        mat_train,
        [mat_train_size, mat_test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply test transform to test subset
    mat_test_dataset = MatBrainTumorDataset(
        mat_folders=mat_folders,
        transform=test_transform,
        cache_images=False  # Don't cache test set
    )
    mat_test_final = torch.utils.data.Subset(mat_test_dataset, mat_test_subset.indices)
    
    # Combine all datasets
    train_dataset = ConcatDataset([archive_train, mat_train_subset])
    test_dataset = ConcatDataset([archive_test, mat_test_final])
    
    print("\n" + "=" * 70)
    print("📈 Dataset Summary")
    print("=" * 70)
    print(f"\n🎓 Training Set:")
    print(f"   Archive images: {len(archive_train):,}")
    print(f"   .mat files:     {len(mat_train_subset):,}")
    print(f"   Total:          {len(train_dataset):,}")
    
    print(f"\n📝 Testing Set:")
    print(f"   Archive images: {len(archive_test):,}")
    print(f"   .mat files:     {len(mat_test_final):,}")
    print(f"   Total:          {len(test_dataset):,}")
    
    total_images = len(train_dataset) + len(test_dataset)
    print(f"\n🎯 Grand Total:    {total_images:,} images")
    print("=" * 70 + "\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dataset_info = {
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'total_size': total_images,
        'archive_train': len(archive_train),
        'archive_test': len(archive_test),
        'mat_train': len(mat_train_subset),
        'mat_test': len(mat_test_final)
    }
    
    return train_loader, test_loader, dataset_info
