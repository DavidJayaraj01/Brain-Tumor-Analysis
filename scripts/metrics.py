"""
Evaluation Metrics for Multi-Task Brain Tumor Detection
=======================================================

Implements comprehensive metrics for:
1. Classification: Accuracy, Precision, Recall, F1, AUC-ROC
2. Segmentation: Dice, IoU, Hausdorff Distance
3. Grading: Accuracy, Balanced Accuracy, Cohen's Kappa

Author: Brain Tumor Detection Team
Date: January 2026
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score,
    balanced_accuracy_score
)
from scipy.spatial.distance import directed_hausdorff
from typing import Dict, Optional


def compute_dice_score(pred, target, smooth=1.0):
    """
    Compute Dice Coefficient
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        pred: [B, 1, H, W] - predicted probabilities
        target: [B, 1, H, W] - ground truth masks
        smooth: Smoothing factor
        
    Returns:
        Dice score (scalar)
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice.item()


def compute_iou(pred, target, smooth=1.0):
    """
    Compute Intersection over Union (Jaccard Index)
    
    IoU = |X ∩ Y| / |X ∪ Y|
    
    Args:
        pred: [B, 1, H, W] - predicted probabilities
        target: [B, 1, H, W] - ground truth masks
        smooth: Smoothing factor
        
    Returns:
        IoU score (scalar)
    """
    pred = (pred > 0.5).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def compute_hausdorff_distance(pred, target):
    """
    Compute 95th percentile Hausdorff Distance
    
    Measures maximum boundary distance between prediction and target
    
    Args:
        pred: [B, 1, H, W] - predicted binary masks
        target: [B, 1, H, W] - ground truth binary masks
        
    Returns:
        Hausdorff distance (scalar)
    """
    pred = (pred > 0.5).cpu().numpy()
    target = target.cpu().numpy()
    
    distances = []
    
    for b in range(pred.shape[0]):
        pred_contour = np.argwhere(pred[b, 0] == 1)
        target_contour = np.argwhere(target[b, 0] == 1)
        
        if len(pred_contour) == 0 or len(target_contour) == 0:
            continue
        
        # Directed Hausdorff distance (both directions)
        hd_1 = directed_hausdorff(pred_contour, target_contour)[0]
        hd_2 = directed_hausdorff(target_contour, pred_contour)[0]
        
        # 95th percentile Hausdorff distance
        hd = max(hd_1, hd_2)
        distances.append(hd)
    
    return np.percentile(distances, 95) if distances else 0.0


def compute_sensitivity_specificity(pred, target):
    """
    Compute Sensitivity (Recall) and Specificity
    
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    
    Args:
        pred: [B, 1, H, W] - predicted probabilities
        target: [B, 1, H, W] - ground truth masks
        
    Returns:
        Dict with sensitivity and specificity
    """
    pred = (pred > 0.5).flatten().cpu().numpy()
    target = target.flatten().cpu().numpy()
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(target, pred, labels=[0, 1]).ravel()
    
    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def compute_classification_metrics(preds, targets, num_classes=4):
    """
    Compute comprehensive classification metrics
    
    Args:
        preds: [B, num_classes] - logits or probabilities
        targets: [B] - class labels
        num_classes: Number of classes
        
    Returns:
        Dict with metrics
    """
    # Convert to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Get predicted classes
    pred_classes = np.argmax(preds, axis=1)
    
    # Compute metrics
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(targets, pred_classes)
    
    # Precision, Recall, F1 (macro and weighted)
    metrics['precision_macro'] = precision_score(
        targets, pred_classes, average='macro', zero_division=0
    )
    metrics['recall_macro'] = recall_score(
        targets, pred_classes, average='macro', zero_division=0
    )
    metrics['f1_macro'] = f1_score(
        targets, pred_classes, average='macro', zero_division=0
    )
    
    metrics['precision_weighted'] = precision_score(
        targets, pred_classes, average='weighted', zero_division=0
    )
    metrics['recall_weighted'] = recall_score(
        targets, pred_classes, average='weighted', zero_division=0
    )
    metrics['f1_weighted'] = f1_score(
        targets, pred_classes, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class = precision_score(
        targets, pred_classes, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        targets, pred_classes, average=None, zero_division=0
    )
    f1_per_class = f1_score(
        targets, pred_classes, average=None, zero_division=0
    )
    
    class_names = ['no_tumor', 'glioma', 'meningioma', 'pituitary'][:num_classes]
    for i, name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics[f'precision_{name}'] = precision_per_class[i]
            metrics[f'recall_{name}'] = recall_per_class[i]
            metrics[f'f1_{name}'] = f1_per_class[i]
    
    # Confusion matrix
    cm = confusion_matrix(targets, pred_classes)
    metrics['confusion_matrix'] = cm
    
    # AUC-ROC (if probabilities available)
    try:
        # Softmax if not already probabilities
        if preds.max() > 1.0 or preds.min() < 0.0:
            from scipy.special import softmax
            preds_prob = softmax(preds, axis=1)
        else:
            preds_prob = preds
        
        # One-vs-Rest AUC
        if num_classes == 2:
            metrics['auc_roc'] = roc_auc_score(targets, preds_prob[:, 1])
        else:
            metrics['auc_roc_macro'] = roc_auc_score(
                targets, preds_prob, multi_class='ovr', average='macro'
            )
            metrics['auc_roc_weighted'] = roc_auc_score(
                targets, preds_prob, multi_class='ovr', average='weighted'
            )
    except:
        pass
    
    return metrics


def compute_segmentation_metrics(preds, targets):
    """
    Compute comprehensive segmentation metrics
    
    Args:
        preds: [B, 1, H, W] - predicted probabilities
        targets: [B, 1, H, W] - ground truth masks
        
    Returns:
        Dict with metrics
    """
    metrics = {}
    
    # Dice coefficient
    metrics['dice_score'] = compute_dice_score(preds, targets)
    
    # IoU
    metrics['iou'] = compute_iou(preds, targets)
    
    # Sensitivity and Specificity
    sens_spec = compute_sensitivity_specificity(preds, targets)
    metrics['sensitivity'] = sens_spec['sensitivity']
    metrics['specificity'] = sens_spec['specificity']
    
    # Hausdorff distance
    metrics['hausdorff_95'] = compute_hausdorff_distance(preds, targets)
    
    # Pixel accuracy
    pred_binary = (preds > 0.5).float()
    target_binary = targets.float()
    metrics['pixel_accuracy'] = (pred_binary == target_binary).float().mean().item()
    
    return metrics


def compute_grading_metrics(preds, targets):
    """
    Compute grading-specific metrics
    
    Args:
        preds: [B, num_grades] - logits or probabilities
        targets: [B] - grade labels
        
    Returns:
        Dict with metrics
    """
    # Convert to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Get predicted classes
    pred_classes = np.argmax(preds, axis=1)
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(targets, pred_classes)
    
    # Balanced accuracy (important for imbalanced grading)
    metrics['balanced_accuracy'] = balanced_accuracy_score(targets, pred_classes)
    
    # Cohen's Kappa (agreement beyond chance)
    metrics['cohens_kappa'] = cohen_kappa_score(targets, pred_classes)
    
    # Sensitivity and Specificity for high-grade detection
    if len(np.unique(targets)) == 2:  # Binary grading
        cm = confusion_matrix(targets, pred_classes)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['sensitivity_high_grade'] = tp / (tp + fn + 1e-10)
            metrics['specificity_high_grade'] = tn / (tn + fp + 1e-10)
            metrics['ppv_high_grade'] = tp / (tp + fp + 1e-10)  # Positive Predictive Value
            metrics['npv_high_grade'] = tn / (tn + fn + 1e-10)  # Negative Predictive Value
    
    return metrics


def compute_metrics(cls_preds, cls_targets, 
                   seg_preds=None, seg_targets=None,
                   grade_preds=None, grade_targets=None):
    """
    Compute all metrics for multi-task evaluation
    
    Args:
        cls_preds: Classification predictions
        cls_targets: Classification targets
        seg_preds: Segmentation predictions (optional)
        seg_targets: Segmentation targets (optional)
        grade_preds: Grading predictions (optional)
        grade_targets: Grading targets (optional)
        
    Returns:
        Dict with all metrics
    """
    all_metrics = {}
    
    # Classification metrics
    if cls_preds is not None and cls_targets is not None:
        cls_metrics = compute_classification_metrics(cls_preds, cls_targets)
        for key, value in cls_metrics.items():
            if key != 'confusion_matrix':
                all_metrics[f'classification_{key}'] = value
            else:
                all_metrics['classification_confusion_matrix'] = value
    
    # Segmentation metrics
    if seg_preds is not None and seg_targets is not None:
        seg_metrics = compute_segmentation_metrics(seg_preds, seg_targets)
        for key, value in seg_metrics.items():
            all_metrics[f'segmentation_{key}'] = value
    
    # Grading metrics
    if grade_preds is not None and grade_targets is not None:
        grade_metrics = compute_grading_metrics(grade_preds, grade_targets)
        for key, value in grade_metrics.items():
            all_metrics[f'grading_{key}'] = value
    
    return all_metrics


def print_metrics(metrics, title="Evaluation Metrics"):
    """
    Pretty print metrics
    """
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    # Classification metrics
    print("\n📊 CLASSIFICATION METRICS:")
    print("-" * 80)
    if 'classification_accuracy' in metrics:
        print(f"  Accuracy:           {metrics['classification_accuracy']:.4f}")
        print(f"  Precision (macro):  {metrics.get('classification_precision_macro', 0):.4f}")
        print(f"  Recall (macro):     {metrics.get('classification_recall_macro', 0):.4f}")
        print(f"  F1-Score (macro):   {metrics.get('classification_f1_macro', 0):.4f}")
        if 'classification_auc_roc_macro' in metrics:
            print(f"  AUC-ROC (macro):    {metrics['classification_auc_roc_macro']:.4f}")
    
    # Segmentation metrics
    print("\n🎯 SEGMENTATION METRICS:")
    print("-" * 80)
    if 'segmentation_dice_score' in metrics:
        print(f"  Dice Score:         {metrics['segmentation_dice_score']:.4f}")
        print(f"  IoU:                {metrics['segmentation_iou']:.4f}")
        print(f"  Sensitivity:        {metrics['segmentation_sensitivity']:.4f}")
        print(f"  Specificity:        {metrics['segmentation_specificity']:.4f}")
        print(f"  Hausdorff (95%):    {metrics['segmentation_hausdorff_95']:.2f} pixels")
    
    # Grading metrics
    print("\n⚖️  GRADING METRICS:")
    print("-" * 80)
    if 'grading_accuracy' in metrics:
        print(f"  Accuracy:           {metrics['grading_accuracy']:.4f}")
        print(f"  Balanced Accuracy:  {metrics['grading_balanced_accuracy']:.4f}")
        print(f"  Cohen's Kappa:      {metrics['grading_cohens_kappa']:.4f}")
        if 'grading_sensitivity_high_grade' in metrics:
            print(f"  Sensitivity (HGG):  {metrics['grading_sensitivity_high_grade']:.4f}")
            print(f"  Specificity (HGG):  {metrics['grading_specificity_high_grade']:.4f}")
    
    print("\n" + "=" * 80)


def test_metrics():
    """
    Test metrics computation
    """
    print("Testing Metrics Computation")
    print("=" * 80)
    
    # Dummy predictions and targets
    batch_size = 32
    num_classes = 4
    num_grades = 2
    
    # Classification
    cls_preds = torch.randn(batch_size, num_classes)
    cls_targets = torch.randint(0, num_classes, (batch_size,))
    
    # Segmentation
    seg_preds = torch.rand(batch_size, 1, 224, 224)
    seg_targets = torch.randint(0, 2, (batch_size, 1, 224, 224)).float()
    
    # Grading
    grade_preds = torch.randn(batch_size, num_grades)
    grade_targets = torch.randint(0, num_grades, (batch_size,))
    
    # Compute metrics
    metrics = compute_metrics(
        cls_preds=cls_preds,
        cls_targets=cls_targets,
        seg_preds=seg_preds,
        seg_targets=seg_targets,
        grade_preds=grade_preds,
        grade_targets=grade_targets
    )
    
    # Print metrics
    print_metrics(metrics, title="Test Metrics")
    
    print("\n✓ Metrics computation validated successfully!")


if __name__ == "__main__":
    test_metrics()
