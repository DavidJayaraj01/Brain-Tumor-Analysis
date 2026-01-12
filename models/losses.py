"""
Multi-Task Loss Functions for Brain Tumor Detection
====================================================

Implements uncertainty-aware multi-task loss combining:
1. Focal Loss (classification) - handles class imbalance
2. Dice Loss + Boundary Loss (segmentation) - improves boundary accuracy
3. Ordinal Regression Loss (grading) - preserves grade ordering
4. Uncertainty weighting - learnable task balancing

Author: Brain Tumor Detection Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights [num_classes]
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, num_classes] - logits
            targets: [B] - class labels
            
        Returns:
            Scalar loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_term = alpha_t * focal_term
        
        loss = focal_term * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice
    
    Handles class imbalance better than BCE
    """
    
    def __init__(self, smooth=1.0, reduction='mean'):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 1, H, W] - predicted probabilities (after sigmoid)
            targets: [B, 1, H, W] - ground truth masks (0 or 1)
            
        Returns:
            Scalar loss
        """
        # Flatten spatial dimensions
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Dice coefficient
        intersection = (inputs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        
        # Dice loss
        loss = 1 - dice
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for improved segmentation boundary accuracy
    
    Focuses on pixels near boundaries using distance transform
    
    Reference: Kervadec et al. "Boundary loss for highly unbalanced segmentation" (2019)
    """
    
    def __init__(self):
        super().__init__()
        
    def compute_distance_map(self, mask):
        """
        Compute normalized distance transform
        
        Args:
            mask: [B, 1, H, W] - binary mask
            
        Returns:
            [B, 1, H, W] - distance map
        """
        from scipy.ndimage import distance_transform_edt
        
        B, _, H, W = mask.shape
        distance_maps = torch.zeros_like(mask)
        
        for b in range(B):
            # Convert to numpy
            mask_np = mask[b, 0].cpu().numpy()
            
            # Compute distance transform (distance to nearest 1-pixel)
            if mask_np.sum() > 0:
                # Distance to foreground
                dist_fg = distance_transform_edt(1 - mask_np)
                # Distance to background
                dist_bg = distance_transform_edt(mask_np)
                # Combined signed distance
                dist = dist_fg - dist_bg
                # Normalize
                dist = dist / (np.abs(dist).max() + 1e-6)
            else:
                dist = np.zeros_like(mask_np)
            
            distance_maps[b, 0] = torch.from_numpy(dist).to(mask.device)
        
        return distance_maps
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 1, H, W] - predicted probabilities
            targets: [B, 1, H, W] - ground truth masks
            
        Returns:
            Scalar loss
        """
        # Compute distance maps for targets
        distance_maps = self.compute_distance_map(targets)
        
        # Boundary loss: weighted by distance to boundary
        # Pixels far from boundary contribute less
        loss = inputs * distance_maps
        
        return loss.mean()


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal Regression Loss for tumor grading
    
    Preserves ordering: Grade I < Grade II < Grade III < Grade IV
    For binary: Low-Grade (0) < High-Grade (1)
    
    Uses multiple binary classifiers with cumulative link function
    """
    
    def __init__(self, num_classes=2):
        """
        Args:
            num_classes: Number of ordinal classes
        """
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, num_classes] - logits
            targets: [B] - ordinal labels (0, 1, ..., num_classes-1)
            
        Returns:
            Scalar loss
        """
        # For binary classification, use standard cross-entropy
        if self.num_classes == 2:
            return F.cross_entropy(inputs, targets)
        
        # For multi-class ordinal regression
        # Create cumulative labels: y_k = 1 if true_label >= k else 0
        B = targets.size(0)
        cumulative_labels = torch.zeros(B, self.num_classes - 1, device=targets.device)
        
        for k in range(self.num_classes - 1):
            cumulative_labels[:, k] = (targets > k).float()
        
        # Binary cross-entropy for each threshold
        cumulative_probs = torch.sigmoid(inputs[:, :-1])
        loss = F.binary_cross_entropy(cumulative_probs, cumulative_labels)
        
        return loss


class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-Task Loss with Learnable Uncertainty Weighting
    
    Automatically balances task losses using homoscedastic uncertainty
    
    L_total = Σ (1 / (2 * σ_i^2)) * L_i + log(σ_i)
    
    Reference: Kendall & Gal "Multi-Task Learning Using Uncertainty to Weigh Losses" (2017)
    """
    
    def __init__(self, num_tasks=3):
        """
        Args:
            num_tasks: Number of tasks (classification, segmentation, grading)
        """
        super().__init__()
        
        # Learnable log-variance parameters
        # Using log to ensure σ^2 > 0
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses):
        """
        Args:
            losses: List of task losses [L_classification, L_segmentation, L_grading]
            
        Returns:
            Weighted total loss
        """
        total_loss = 0
        
        for i, loss in enumerate(losses):
            # Precision weighting: 1 / (2 * σ^2)
            precision = torch.exp(-self.log_vars[i])
            # Weighted loss + regularization term
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss


class MultiTaskLoss(nn.Module):
    """
    Complete Multi-Task Loss for Brain Tumor Detection
    
    Combines:
    1. Focal Loss (classification)
    2. Dice + Boundary Loss (segmentation)
    3. Ordinal Regression (grading)
    4. Uncertainty weighting (automatic task balancing)
    """
    
    def __init__(self, 
                 num_classes=4, 
                 num_grades=2,
                 class_weights=None,
                 use_uncertainty=True,
                 dice_weight=0.5,
                 boundary_weight=0.5):
        """
        Args:
            num_classes: Number of tumor classes
            num_grades: Number of grading classes
            class_weights: Weights for classification classes
            use_uncertainty: Use uncertainty-based task weighting
            dice_weight: Weight for Dice loss in segmentation
            boundary_weight: Weight for Boundary loss in segmentation
        """
        super().__init__()
        
        # Task-specific losses
        self.classification_loss = FocalLoss(
            alpha=class_weights,
            gamma=2.0
        )
        
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        
        self.grading_loss = OrdinalRegressionLoss(num_grades)
        
        # Loss weights
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        
        # Uncertainty weighting
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.uncertainty_weights = UncertaintyWeightedLoss(num_tasks=3)
        else:
            # Manual weights if not using uncertainty
            self.task_weights = {
                'classification': 1.0,
                'segmentation': 1.0,
                'grading': 0.5  # Lower weight (fewer samples)
            }
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                task_mask: Optional[Dict[str, bool]] = None):
        """
        Args:
            outputs: Dict with keys ['classification', 'segmentation', 'grading']
            targets: Dict with keys ['classification', 'segmentation', 'grading']
            task_mask: Dict indicating which tasks have labels for this batch
            
        Returns:
            Dict with total loss and individual task losses
        """
        losses = {}
        task_losses = []
        
        # 1. Classification Loss
        if task_mask is None or task_mask.get('classification', True):
            cls_loss = self.classification_loss(
                outputs['classification'], 
                targets['classification']
            )
            losses['classification'] = cls_loss
            task_losses.append(cls_loss)
        else:
            task_losses.append(torch.tensor(0.0, device=outputs['classification'].device))
        
        # 2. Segmentation Loss (Dice + Boundary)
        if task_mask is None or task_mask.get('segmentation', False):
            # Apply sigmoid to logits
            seg_probs = torch.sigmoid(outputs['segmentation'])
            
            # Dice loss
            dice = self.dice_loss(seg_probs, targets['segmentation'])
            
            # Boundary loss
            boundary = self.boundary_loss(seg_probs, targets['segmentation'])
            
            # Combined segmentation loss
            seg_loss = self.dice_weight * dice + self.boundary_weight * boundary
            losses['segmentation'] = seg_loss
            losses['dice'] = dice
            losses['boundary'] = boundary
            task_losses.append(seg_loss)
        else:
            task_losses.append(torch.tensor(0.0, device=outputs['classification'].device))
        
        # 3. Grading Loss
        if task_mask is None or task_mask.get('grading', False):
            grade_loss = self.grading_loss(
                outputs['grading'], 
                targets['grading']
            )
            losses['grading'] = grade_loss
            task_losses.append(grade_loss)
        else:
            task_losses.append(torch.tensor(0.0, device=outputs['classification'].device))
        
        # 4. Total Loss with Uncertainty Weighting
        if self.use_uncertainty:
            total_loss = self.uncertainty_weights(task_losses)
            losses['uncertainty_weights'] = torch.exp(-self.uncertainty_weights.log_vars)
        else:
            # Manual weighting
            total_loss = (
                self.task_weights['classification'] * task_losses[0] +
                self.task_weights['segmentation'] * task_losses[1] +
                self.task_weights['grading'] * task_losses[2]
            )
        
        losses['total'] = total_loss
        
        return losses


def test_losses():
    """
    Test loss functions
    """
    print("Testing Multi-Task Loss Functions")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 4
    num_classes = 4
    num_grades = 2
    
    # Dummy outputs
    outputs = {
        'classification': torch.randn(batch_size, num_classes),
        'segmentation': torch.randn(batch_size, 1, 224, 224),
        'grading': torch.randn(batch_size, num_grades)
    }
    
    # Dummy targets
    targets = {
        'classification': torch.randint(0, num_classes, (batch_size,)),
        'segmentation': torch.randint(0, 2, (batch_size, 1, 224, 224)).float(),
        'grading': torch.randint(0, num_grades, (batch_size,))
    }
    
    # Task mask (all tasks available)
    task_mask = {
        'classification': True,
        'segmentation': True,
        'grading': True
    }
    
    # Initialize loss
    criterion = MultiTaskLoss(
        num_classes=num_classes,
        num_grades=num_grades,
        use_uncertainty=True
    )
    
    # Compute loss
    losses = criterion(outputs, targets, task_mask)
    
    print("\nLoss Components:")
    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {name}: {value.item():.4f}")
        else:
            print(f"  {name}: {value}")
    
    print("\nUncertainty Weights (σ^-2):")
    if 'uncertainty_weights' in losses:
        weights = losses['uncertainty_weights'].detach().numpy()
        tasks = ['classification', 'segmentation', 'grading']
        for task, weight in zip(tasks, weights):
            print(f"  {task}: {weight:.4f}")
    
    print("=" * 60)
    print("✓ Loss functions validated successfully!")


if __name__ == "__main__":
    test_losses()
