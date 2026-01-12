"""
Simple Training Script for Brain Tumor Classification
=====================================================

Trains the H2A-MTN model on GPU using the actual dataset.

Usage:
    python train_simple.py --epochs 50 --batch_size 16

Author: Brain Tumor Detection Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_architecture import H2A_MTN
from scripts.dataset_loader import create_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['classification_label'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            cls_logits = outputs['classification']
            loss = criterion(cls_logits, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(cls_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['classification_label'].to(device)
            
            outputs = model(images)
            cls_logits = outputs['classification']
            loss = criterion(cls_logits, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(cls_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def main(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training Brain Tumor Classification Model")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = H2A_MTN(
        num_classes=4,  # glioma, meningioma, notumor, pituitary
        img_size=args.img_size,
        patch_size=16,
        in_channels=3
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
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
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Create checkpoint directory
    checkpoint_dir = Path(__file__).parent.parent / 'models' / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting Training for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc
            history['best_epoch'] = epoch + 1
            
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Save latest checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {history['best_epoch']})")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")
    
    # Save training history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to: {history_path}")
    print(f"{'='*60}\n")
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Brain Tumor Classification Model')
    
    # Data
    parser.add_argument('--data_root', type=str, 
                       default=r'C:\Users\david\Downloads\1512427\archive',
                       help='Root directory of dataset')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
    args = parser.parse_args()
    
    # Train model
    model, history = main(args)
