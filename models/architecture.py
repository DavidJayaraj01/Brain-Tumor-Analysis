"""
Hybrid Transformer-CNN Model Architecture
Combines EfficientNetV2, Vision Transformer, and Swin Transformer
for Multi-Task Learning: Classification, Segmentation, Grading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Dict, Tuple, List
import numpy as np


class FeatureFusionModule(nn.Module):
    """
    Fuses features from multiple sources with adaptive weighting
    """
    def __init__(self, num_features: int, num_sources: int = 3):
        super().__init__()
        self.num_sources = num_sources
        self.weight_fc = nn.Sequential(
            nn.Linear(num_features * num_sources, 512),
            nn.ReLU(),
            nn.Linear(512, num_sources),
            nn.Softmax(dim=1)
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [B, C, H, W] tensors
        Returns:
            Fused feature tensor [B, C, H, W]
        """
        batch_size = features[0].shape[0]
        
        # Adaptive weighting
        concatenated = torch.cat([f.view(batch_size, -1) for f in features], dim=1)
        weights = self.weight_fc(concatenated)
        weights = weights.view(batch_size, -1, 1, 1)
        
        # Normalize features to same scale
        normalized_features = [f / (f.norm() + 1e-8) for f in features]
        
        # Weighted fusion
        fused = sum(w[:, i:i+1] * normalized_features[i] for i in range(self.num_sources))
        return fused


class LocalFeatureExtractor(nn.Module):
    """
    EfficientNetV2 for local feature extraction
    Captures fine-grained tumor characteristics
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # EfficientNetV2-S pre-trained on ImageNet
        self.backbone = timm.create_model('efficientnetv2_s', pretrained=pretrained, 
                                         in_chans=4, num_classes=0)  # No classification head
        
        # Feature pyramid levels
        self.feature_dim = 1280
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, 4, 224, 224]
        Returns:
            Multi-scale features
        """
        features = {}
        
        # Forward pass captures intermediate features
        for layer in self.backbone.blocks:
            x = layer(x)
            
        features['conv'] = x  # [B, 1280, 7, 7]
        return features


class GlobalContextExtractor(nn.Module):
    """
    Vision Transformer for global context
    Captures long-range dependencies
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model('vit_b_16_384', pretrained=pretrained,
                                     in_chans=4, num_classes=0)
        self.patch_embed = self.vit.patch_embed
        self.feature_dim = 768
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, 4, 384, 384]
        Returns:
            Global context features
        """
        features = {}
        
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        x = self.vit.blocks(x)
        features['vit'] = x  # [B, 145, 768]
        
        return features


class HierarchicalAttentionExtractor(nn.Module):
    """
    Swin Transformer for hierarchical attention
    Multi-scale processing with shifted windows
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.swin = timm.create_model('swin_base_patch4_window7_224', 
                                      pretrained=pretrained,
                                      in_chans=4, num_classes=0)
        self.feature_dim = 1024
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, 4, 224, 224]
        Returns:
            Hierarchical features
        """
        features = {}
        
        x = self.swin.patch_embed(x)
        x = self.swin.pos_drop(x)
        
        # Store intermediate hierarchical features
        for i, layer in enumerate(self.swin.layers):
            x = layer(x)
            
        features['swin'] = x  # Hierarchical representation
        return features


class SegmentationHead(nn.Module):
    """
    Decoder for tumor segmentation
    Outputs: [B, 4, H, W] - 4 tumor classes
    """
    def __init__(self, in_channels: int = 1280, num_classes: int = 4):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class ClassificationHead(nn.Module):
    """
    Classification head for tumor detection
    Outputs: Binary logits for presence/absence
    """
    def __init__(self, in_features: int = 768, hidden_dim: int = 512):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 2),  # Binary classification
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class GradingHead(nn.Module):
    """
    Grading head for WHO classification
    Outputs: Binary logits for LGG/HGG classification
    """
    def __init__(self, in_features: int = 768, hidden_dim: int = 512):
        super().__init__()
        
        self.grader = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 2),  # LGG vs HGG
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.grader(x)


class UnifiedMultiTaskModel(nn.Module):
    """
    Complete unified model combining all components
    Multi-task learning: Classification + Segmentation + Grading
    """
    
    def __init__(self, 
                 num_classes: int = 4,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5):
        super().__init__()
        
        # Feature extractors
        self.local_extractor = LocalFeatureExtractor(pretrained=pretrained)
        self.global_extractor = GlobalContextExtractor(pretrained=pretrained)
        self.hierarchical_extractor = HierarchicalAttentionExtractor(pretrained=pretrained)
        
        # Feature fusion
        self.fusion = FeatureFusionModule(num_features=512, num_sources=3)
        
        # Task-specific heads
        self.segmentation_head = SegmentationHead(in_channels=1280, num_classes=num_classes)
        self.classification_head = ClassificationHead(in_features=768, hidden_dim=512)
        self.grading_head = GradingHead(in_features=768, hidden_dim=512)
        
        # Uncertainty estimation (Monte Carlo Dropout)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Shared projection layer
        self.shared_projection = nn.Linear(768 + 1280, 512)
        
    def extract_features(self, x_224: torch.Tensor, x_384: torch.Tensor) -> torch.Tensor:
        """
        Extract and fuse features from multiple sources
        
        Args:
            x_224: Input at 224x224 resolution [B, 4, 224, 224]
            x_384: Input at 384x384 resolution [B, 4, 384, 384]
            
        Returns:
            Fused features [B, 512]
        """
        # Local features from EfficientNetV2
        local_feats = self.local_extractor(x_224)['conv']  # [B, 1280, 7, 7]
        
        # Global features from ViT
        global_feats = self.global_extractor(x_384)['vit']  # [B, 145, 768]
        
        # Hierarchical features from Swin
        hier_feats = self.hierarchical_extractor(x_224)['swin']  # [B, N, 1024]
        
        # Global pooling for classifier input
        global_pooled = global_feats[:, 0, :]  # CLS token [B, 768]
        
        # Local spatial feature pooling
        local_pooled = F.adaptive_avg_pool2d(local_feats, 1).squeeze(-1).squeeze(-1)  # [B, 1280]
        
        # Concatenate for shared representation
        shared_features = torch.cat([global_pooled, local_pooled], dim=1)  # [B, 2048]
        shared_features = self.shared_projection(shared_features)  # [B, 512]
        
        return shared_features, local_feats
    
    def forward(self, 
                x_224: torch.Tensor, 
                x_384: torch.Tensor,
                mc_dropout: int = 1) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task outputs
        
        Args:
            x_224: Input at 224x224 [B, 4, 224, 224]
            x_384: Input at 384x384 [B, 4, 384, 384]
            mc_dropout: Number of MC-Dropout samples for uncertainty
            
        Returns:
            Dictionary with:
                - 'classification': [B, 2]
                - 'segmentation': [B, 4, 224, 224]
                - 'grading': [B, 2]
                - 'uncertainties': uncertainty estimates
        """
        outputs = {}
        uncertainties = {
            'classification': [],
            'grading': [],
        }
        
        # Feature extraction
        shared_features, local_feats = self.extract_features(x_224, x_384)
        
        # Task 1: Segmentation (deterministic)
        seg_output = self.segmentation_head(local_feats)
        outputs['segmentation'] = seg_output  # [B, 4, 224, 224]
        
        # Tasks 2 & 3: Classification and Grading with MC-Dropout
        for _ in range(mc_dropout):
            feat_dropped = self.dropout(shared_features)
            
            clf_out = self.classification_head(feat_dropped)
            outputs['classification'] = clf_out if _ == 0 else outputs['classification']
            uncertainties['classification'].append(clf_out)
            
            grade_out = self.grading_head(feat_dropped)
            outputs['grading'] = grade_out if _ == 0 else outputs['grading']
            uncertainties['grading'].append(grade_out)
        
        # Compute uncertainty estimates
        if mc_dropout > 1:
            clf_stack = torch.stack(uncertainties['classification'])  # [MC, B, 2]
            grade_stack = torch.stack(uncertainties['grading'])  # [MC, B, 2]
            
            # Bayesian variance as uncertainty
            outputs['clf_uncertainty'] = clf_stack.var(dim=0).mean(dim=1)  # [B]
            outputs['grade_uncertainty'] = grade_stack.var(dim=0).mean(dim=1)  # [B]
        else:
            outputs['clf_uncertainty'] = torch.zeros(shared_features.shape[0])
            outputs['grade_uncertainty'] = torch.zeros(shared_features.shape[0])
        
        return outputs


class MultiTaskLoss(nn.Module):
    """
    Combined loss function for multi-task learning
    Balances classification, segmentation, and grading losses
    """
    
    def __init__(self, 
                 lambda_clf: float = 1.0,
                 lambda_seg: float = 2.0,
                 lambda_grade: float = 1.5,
                 dice_weight: float = 0.5):
        super().__init__()
        
        self.lambda_clf = lambda_clf
        self.lambda_seg = lambda_seg
        self.lambda_grade = lambda_grade
        self.dice_weight = dice_weight
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        
    def dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """
        Dice loss for segmentation
        """
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth dictionary
                - 'classification': [B] binary labels
                - 'segmentation': [B, H, W] integer labels
                - 'grading': [B] binary labels
                
        Returns:
            Loss dictionary with total loss and individual losses
        """
        losses = {}
        
        # Classification loss
        clf_loss = self.ce_loss(outputs['classification'], targets['classification'].long())
        losses['clf_loss'] = clf_loss * self.lambda_clf
        
        # Segmentation loss (Dice + CE)
        seg_logits = outputs['segmentation']
        seg_targets = targets['segmentation'].long()
        
        ce_seg_loss = self.ce_loss(seg_logits, seg_targets)
        dice_loss = self.dice_coefficient(
            F.softmax(seg_logits, dim=1)[:, 1:].max(dim=1)[0],  # Tumor classes
            (seg_targets > 0).float()
        )
        seg_loss = ce_seg_loss * (1 - self.dice_weight) + dice_loss * self.dice_weight
        losses['seg_loss'] = seg_loss * self.lambda_seg
        
        # Grading loss
        grade_loss = self.ce_loss(outputs['grading'], targets['grading'].long())
        losses['grade_loss'] = grade_loss * self.lambda_grade
        
        # Total loss
        losses['total'] = losses['clf_loss'] + losses['seg_loss'] + losses['grade_loss']
        
        return losses


# Model instantiation helper
def build_unified_model(pretrained: bool = True, num_classes: int = 4) -> UnifiedMultiTaskModel:
    """
    Factory function to create unified model
    """
    model = UnifiedMultiTaskModel(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=0.5
    )
    return model


if __name__ == "__main__":
    # Test model
    model = build_unified_model(pretrained=False)
    
    # Dummy inputs
    x_224 = torch.randn(2, 4, 224, 224)
    x_384 = torch.randn(2, 4, 384, 384)
    
    # Forward pass
    outputs = model(x_224, x_384, mc_dropout=3)
    
    print("Model Architecture Test")
    print("=" * 50)
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")
    
    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
