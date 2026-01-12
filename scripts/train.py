"""
Training Script for H2A-MTN
===========================

Multi-task training with:
- Mixed precision training (FP16)
- Learning rate scheduling
- Model checkpointing
- TensorBoard logging
- Early stopping
- Gradient clipping

Author: Brain Tumor Detection Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import os
import time
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_architecture import H2A_MTN
from models.losses import MultiTaskLoss
from scripts.metrics import compute_metrics


class Trainer:
    """
    Multi-Task Training Manager
    """
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 scheduler,
                 device,
                 config):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        self.patience_counter = 0
        self.epoch = 0
        
        # Checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
    def train_epoch(self):
        """
        Train for one epoch
        """
        self.model.train()
        
        epoch_losses = {
            'total': 0,
            'classification': 0,
            'segmentation': 0,
            'grading': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1}/{self.config.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            targets = {
                'classification': batch['classification_label'].to(self.device),
                'segmentation': batch['segmentation_mask'].to(self.device) if batch['segmentation_mask'] is not None else None,
                'grading': batch['grading_label'].to(self.device) if batch['grading_label'] is not None else None
            }
            task_mask = batch['task_mask']
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    losses = self.criterion(outputs, targets, task_mask)
                
                # Backward pass
                self.scaler.scale(losses['total']).backward()
                
                # Gradient clipping
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.grad_clip
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(images)
                losses = self.criterion(outputs, targets, task_mask)
                
                losses['total'].backward()
                
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.grad_clip
                    )
                
                self.optimizer.step()
            
            # Update epoch losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'cls': losses['classification'].item() if 'classification' in losses else 0,
                'seg': losses.get('segmentation', torch.tensor(0)).item(),
                'grad': losses.get('grading', torch.tensor(0)).item()
            })
            
            # Log to TensorBoard (every 100 iterations)
            if batch_idx % 100 == 0:
                global_step = self.epoch * len(self.train_loader) + batch_idx
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        self.writer.add_scalar(f'train/{key}', value.item(), global_step)
        
        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    def validate(self):
        """
        Validate on validation set
        """
        self.model.eval()
        
        val_losses = {
            'total': 0,
            'classification': 0,
            'segmentation': 0,
            'grading': 0
        }
        
        # Metrics storage
        all_preds_cls = []
        all_targets_cls = []
        all_preds_seg = []
        all_targets_seg = []
        all_preds_grade = []
        all_targets_grade = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                images = batch['image'].to(self.device)
                targets = {
                    'classification': batch['classification_label'].to(self.device),
                    'segmentation': batch['segmentation_mask'].to(self.device) if batch['segmentation_mask'] is not None else None,
                    'grading': batch['grading_label'].to(self.device) if batch['grading_label'] is not None else None
                }
                task_mask = batch['task_mask']
                
                # Forward pass
                outputs = self.model(images)
                losses = self.criterion(outputs, targets, task_mask)
                
                # Update losses
                for key in val_losses:
                    if key in losses:
                        val_losses[key] += losses[key].item()
                
                # Store predictions and targets for metrics
                all_preds_cls.append(outputs['classification'].cpu())
                all_targets_cls.append(targets['classification'].cpu())
                
                if targets['segmentation'] is not None:
                    all_preds_seg.append(torch.sigmoid(outputs['segmentation']).cpu())
                    all_targets_seg.append(targets['segmentation'].cpu())
                
                if targets['grading'] is not None:
                    all_preds_grade.append(outputs['grading'].cpu())
                    all_targets_grade.append(targets['grading'].cpu())
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)
        
        # Compute metrics
        metrics = compute_metrics(
            cls_preds=torch.cat(all_preds_cls),
            cls_targets=torch.cat(all_targets_cls),
            seg_preds=torch.cat(all_preds_seg) if all_preds_seg else None,
            seg_targets=torch.cat(all_targets_seg) if all_targets_seg else None,
            grade_preds=torch.cat(all_preds_grade) if all_preds_grade else None,
            grade_targets=torch.cat(all_targets_grade) if all_targets_grade else None
        )
        
        return val_losses, metrics
    
    def train(self):
        """
        Full training loop
        """
        print(f"\nStarting training for {self.config.epochs} epochs")
        print("=" * 80)
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            val_losses, val_metrics = self.validate()
            
            # Learning rate scheduling
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_losses['total'])
            else:
                self.scheduler.step()
            
            # Log to TensorBoard
            self.writer.add_scalar('epoch/train_loss', train_losses['total'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_losses['total'], epoch)
            self.writer.add_scalar('epoch/lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'metrics/{key}', value, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print(f"Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f}")
            print(f"Classification Acc: {val_metrics['classification_accuracy']:.4f}")
            if 'dice_score' in val_metrics:
                print(f"Segmentation Dice: {val_metrics['dice_score']:.4f}")
            if 'grading_accuracy' in val_metrics:
                print(f"Grading Acc: {val_metrics['grading_accuracy']:.4f}")
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_loss.pth', metrics=val_metrics)
                print("✓ Saved best model (loss)")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if 'dice_score' in val_metrics and val_metrics['dice_score'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice_score']
                self.save_checkpoint('best_dice.pth', metrics=val_metrics)
                print("✓ Saved best model (dice)")
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint every N epochs
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', metrics=val_metrics)
        
        print("\nTraining completed!")
        print("=" * 80)
        
        self.writer.close()
    
    def save_checkpoint(self, filename, metrics=None):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice,
            'config': vars(self.config),
            'metrics': metrics
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_dice = checkpoint['best_val_dice']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✓ Loaded checkpoint from epoch {self.epoch}")


def get_config():
    """
    Training configuration
    """
    parser = argparse.ArgumentParser(description='Train H2A-MTN')
    
    # Model
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--num_grades', type=int, default=2)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'plateau', 'step'])
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=15)
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='../datasets')
    parser.add_argument('--checkpoint_dir', type=str, default='../results/checkpoints')
    parser.add_argument('--log_dir', type=str, default='../results/logs')
    
    # Misc
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    Main training function
    """
    # Configuration
    config = get_config()
    set_seed(config.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    print("\nInitializing model...")
    model = H2A_MTN(
        num_classes=config.num_classes,
        num_grades=config.num_grades
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = MultiTaskLoss(
        num_classes=config.num_classes,
        num_grades=config.num_grades,
        use_uncertainty=True
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    if config.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.epochs,
            eta_min=1e-6
        )
    elif config.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:  # step
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.1
        )
    
    # Data loaders (placeholder - implement actual dataloaders)
    print("\nLoading datasets...")
    # TODO: Implement actual dataloaders
    # train_loader = get_train_loader(config)
    # val_loader = get_val_loader(config)
    
    # For now, create dummy loaders for testing
    print("Note: Using dummy dataloaders for demonstration")
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=None,  # train_loader,
        val_loader=None,  # val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    # Resume from checkpoint
    if config.resume:
        trainer.load_checkpoint(config.resume)
    
    # Train
    # trainer.train()


if __name__ == "__main__":
    main()
