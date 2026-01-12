"""
Brain Tumor Detection - Simple Training Script
===============================================

Quick start training script with GPU support and automatic best model saving.

Usage:
    # Basic training (50 epochs)
    python train_model.py
    
    # Custom epochs and batch size
    python train_model.py --epochs 100 --batch_size 32
    
    # Resume from checkpoint
    python train_model.py --resume checkpoints/best_model.pth

Author: MediMind BTIS Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.dataset_loader import BrainTumorDataset, get_transforms, create_dataloaders
from models.hybrid_architecture import H2A_MTN


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Brain Tumor Detection Model')
    
    # Data
    parser.add_argument('--data_root', type=str, 
                       default='datasets/archive',
                       help='Root directory of dataset')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay (L2 regularization)')
    
    # Model
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Number of tumor classes')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    
    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def get_device(device_arg='auto'):
    """Get the best available device"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device('cpu')
            print("⚠️  No GPU found, using CPU (training will be slower)")
    else:
        device = torch.device(device_arg)
        print(f"Using device: {device}")
    
    return device


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['classification_label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Get classification logits
        if isinstance(outputs, dict):
            cls_logits = outputs['classification']
        else:
            cls_logits = outputs
        
        # Compute loss
        loss = criterion(cls_logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(cls_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        current_acc = 100 * correct / total
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy tracking
    class_correct = [0] * 4
    class_total = [0] * 4
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Valid]')
        
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['classification_label'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get classification logits
            if isinstance(outputs, dict):
                cls_logits = outputs['classification']
            else:
                cls_logits = outputs
            
            # Compute loss
            loss = criterion(cls_logits, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(cls_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
            
            # Update progress bar
            current_acc = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    # Print per-class accuracy
    print("\n  Per-class Accuracy:")
    for i in range(4):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"    {class_names[i]:12s}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_acc, 
                   checkpoint_dir, is_best=False, filename=None):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'timestamp': datetime.now().isoformat()
    }
    
    if filename is None:
        filename = f'checkpoint_epoch_{epoch+1}.pth'
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"  💾 Saved BEST model: {best_path}")
    
    return filepath


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Previous val_acc: {checkpoint.get('val_acc', 0):.2f}%")
    
    return start_epoch


def main():
    """Main training function"""
    args = parse_args()
    
    print("=" * 80)
    print("🧠 Brain Tumor Detection - Model Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.data_root}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Image Size: {args.img_size}x{args.img_size}")
    print(f"  Checkpoint Dir: {args.checkpoint_dir}")
    print()
    
    # Device
    device = get_device(args.device)
    
    # Create dataloaders
    print("\n📂 Loading Dataset...")
    train_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    # Create model
    print("\n🔨 Building Model...")
    model = H2A_MTN(
        num_classes=args.num_classes,
        img_size=args.img_size
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print("\n" + "=" * 80)
    print("🚀 Starting Training...")
    print("=" * 80 + "\n")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"  🎉 New best validation accuracy: {best_val_acc:.2f}%")
        
        if (epoch + 1) % args.save_every == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, val_acc,
                args.checkpoint_dir, is_best=is_best
            )
        
        print()
    
    # Training complete
    print("=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    print("\n🎊 Happy analyzing! 🧠")


if __name__ == '__main__':
    main()
