"""
Full Dataset Training - Archive + All .mat Files (1-3064)
==========================================================

Trains using ALL available data:
- 5,712 archive training images
- 1,311 archive testing images  
- 3,064 .mat files (90/10 train/test split)

Total: ~9,087 images!

Usage:
    python train_full_dataset.py

Author: MediMind BTIS Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.mat_dataset_loader import create_combined_dataloaders
from models.hybrid_architecture import H2A_MTN


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train on Full Dataset (Archive + .mat)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (32 optimal for RTX 4060)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay')
    
    # Model
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num_classes', type=int, default=4,
                       help='Number of classes')
    
    # System
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--cache_mat', action='store_true',
                       help='Cache .mat images in RAM (faster, uses ~4GB)')
    
    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_full',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save every N epochs')
    
    return parser.parse_args()


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
    
    # Per-class accuracy
    class_correct = [0] * 4
    class_total = [0] * 4
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Valid]')
        
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['classification_label'].to(device)
            
            outputs = model(images)
            
            if isinstance(outputs, dict):
                cls_logits = outputs['classification']
            else:
                cls_logits = outputs
            
            loss = criterion(cls_logits, labels)
            
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
    """Save checkpoint"""
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
        best_path = os.path.join(checkpoint_dir, 'best_model_full.pth')
        torch.save(checkpoint, best_path)
        print(f"  💾 Saved BEST model: {best_path}")


def main():
    """Main training function"""
    args = parse_args()
    
    print("=" * 80)
    print("🧠 Brain Tumor Detection - FULL DATASET Training")
    print("   Archive Images + All 3,064 .mat Files")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Image Size: {args.img_size}x{args.img_size}")
    print(f"  Cache .mat images: {args.cache_mat}")
    print(f"  Checkpoint Dir: {args.checkpoint_dir}")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("⚠️  Using CPU (training will be slow)\n")
    
    # Create dataloaders
    print("📂 Loading FULL Dataset (this may take a few minutes)...")
    train_loader, test_loader, dataset_info = create_combined_dataloaders(
        archive_root='datasets/archive',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        cache_mat_images=args.cache_mat
    )
    
    # Create model
    print("🔨 Building Model...")
    model = H2A_MTN(
        num_classes=args.num_classes,
        img_size=args.img_size
    )
    model = model.to(device)
    
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
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'dataset_info': dataset_info
    }
    
    best_val_acc = 0.0
    
    print("\n" + "=" * 80)
    print("🚀 Starting Training on FULL Dataset...")
    print("=" * 80 + "\n")
    
    # Training loop
    for epoch in range(args.epochs):
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
        
        # Print summary
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
    print(f"\n🎯 Dataset Summary:")
    print(f"   Total images trained on: {dataset_info['total_size']:,}")
    print(f"   Archive images: {dataset_info['archive_train'] + dataset_info['archive_test']:,}")
    print(f"   .mat files: {dataset_info['mat_train'] + dataset_info['mat_test']:,}")
    print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"\n💾 Best model saved to: {os.path.join(args.checkpoint_dir, 'best_model_full.pth')}")
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, 'training_history_full.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📊 Training history saved to: {history_path}")
    
    print("\n🎊 Congratulations! Your model is trained on the FULL dataset! 🧠")


if __name__ == '__main__':
    main()
