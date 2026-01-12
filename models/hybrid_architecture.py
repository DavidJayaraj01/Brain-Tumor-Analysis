"""
Hybrid Hierarchical Attention Multi-Task Network (H2A-MTN)
===========================================================

Architecture combining:
1. EfficientNetV2 - Local feature extraction
2. Vision Transformer (ViT) - Global context modeling
3. Swin Transformer - Hierarchical multi-scale decoding

Multi-Task Outputs:
- Classification (4-class): glioma, meningioma, pituitary, no tumor
- Segmentation: Pixel-wise tumor mask
- Grading (binary): low-grade vs high-grade

Author: Brain Tumor Detection Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Dict, Tuple, Optional
import math


class EfficientNetV2Encoder(nn.Module):
    """
    EfficientNetV2 backbone for local feature extraction
    
    Features:
    - Parameter-efficient MBConv and Fused-MBConv blocks
    - Progressive learning capability
    - Multi-scale feature extraction
    """
    
    def __init__(self, model_name='tf_efficientnetv2_s', pretrained=True):
        """
        Args:
            model_name: EfficientNetV2 variant ('s', 'm', 'l')
            pretrained: Use ImageNet pretrained weights
        """
        super().__init__()
        
        # Load pretrained EfficientNetV2
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,  # Return intermediate feature maps
            out_indices=(1, 2, 3, 4)  # Multi-scale features
        )
        
        # Feature dimensions for EfficientNetV2-S
        # Scale 1: 24 channels, 56x56
        # Scale 2: 48 channels, 28x28
        # Scale 3: 64 channels, 14x14
        # Scale 4: 160 channels, 7x7
        self.feature_dims = self.backbone.feature_info.channels()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of feature maps at different scales
        """
        features = self.backbone(x)  # List of 4 feature maps
        return features


class PatchEmbedding(nn.Module):
    """
    Convert image into sequence of patches for Vision Transformer
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional patch embedding
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, num_patches, embed_dim]
        """
        x = self.projection(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism
    """
    
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, embed_dim]
        Returns:
            [B, N, embed_dim]
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with attention and MLP
    """
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for global context modeling
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            cls_token: [B, embed_dim] - for classification
            patch_tokens: [B, num_patches, embed_dim] - for segmentation
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Separate class token and patch tokens
        cls_token = x[:, 0]  # [B, embed_dim]
        patch_tokens = x[:, 1:]  # [B, num_patches, embed_dim]
        
        return cls_token, patch_tokens


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention for Swin Transformer
    """
    
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Initialize
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, x):
        """
        Args:
            x: [B*num_windows, window_size*window_size, C]
        Returns:
            [B*num_windows, window_size*window_size, C]
        """
        B_, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block with shifted window attention
    """
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x, H, W):
        """
        Args:
            x: [B, H*W, C]
            H, W: Spatial dimensions
        """
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Window partition
        x_windows = self.window_partition(shifted_x, self.window_size)  # [B*num_windows, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [B*num_windows, window_size*window_size, C]
        
        # Attention
        attn_windows = self.attn(x_windows)
        
        # Window reverse
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def window_partition(self, x, window_size):
        """
        Args:
            x: [B, H, W, C]
            window_size: int
        Returns:
            windows: [B*num_windows, window_size, window_size, C]
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    def window_reverse(self, windows, window_size, H, W):
        """
        Args:
            windows: [B*num_windows, window_size, window_size, C]
            window_size: int
            H, W: int
        Returns:
            x: [B, H, W, C]
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class SwinTransformerDecoder(nn.Module):
    """
    Hierarchical decoder using Swin Transformer blocks
    """
    
    def __init__(self, in_channels=768, num_classes=1, depths=[2, 2, 2, 2]):
        super().__init__()
        
        # Decoder stages with upsampling
        self.decoder_stages = nn.ModuleList()
        channels = [512, 256, 128, 64]
        
        for i, (depth, out_ch) in enumerate(zip(depths, channels)):
            stage = nn.ModuleList([
                SwinTransformerBlock(
                    dim=out_ch,
                    num_heads=out_ch // 32,
                    window_size=7,
                    shift_size=0 if (j % 2 == 0) else 3
                )
                for j in range(depth)
            ])
            self.decoder_stages.append(stage)
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels if i == 0 else channels[i-1], channels[i], 
                              kernel_size=2, stride=2)
            for i in range(len(channels))
        ])
        
        # Final segmentation head
        self.seg_head = nn.Conv2d(channels[-1], num_classes, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - features from encoder
        Returns:
            [B, num_classes, H*16, W*16] - segmentation mask
        """
        for i, (upsample, stage) in enumerate(zip(self.upsample_layers, self.decoder_stages)):
            # Upsample
            x = upsample(x)  # [B, C, H, W]
            
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            # Swin blocks
            for block in stage:
                x = block(x, H, W)
            
            x = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        
        # Final segmentation
        seg_map = self.seg_head(x)
        return seg_map


class CrossAttentionFusion(nn.Module):
    """
    Fuse CNN and Transformer features using cross-attention
    """
    
    def __init__(self, cnn_dim, vit_dim, out_dim):
        super().__init__()
        
        self.out_dim = out_dim
        
        # Project CNN features to common dimension
        self.cnn_proj = nn.Conv2d(cnn_dim, out_dim, kernel_size=1)
        
        # Project ViT features to common dimension
        self.vit_proj = nn.Linear(vit_dim, out_dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(out_dim, num_heads=8)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, cnn_features, vit_features):
        """
        Args:
            cnn_features: [B, C_cnn, H, W]
            vit_features: [B, N, C_vit] where N=196 for 14x14 patches
        Returns:
            [B, out_dim, H, W]
        """
        B, C_cnn, H, W = cnn_features.shape  # H=W=7
        _, N, C_vit = vit_features.shape  # N=196
        
        # Project CNN features
        cnn_proj = self.cnn_proj(cnn_features)  # [B, out_dim, H, W] = [B, 512, 7, 7]
        
        # Project ViT features  
        vit_proj = self.vit_proj(vit_features)  # [B, N, out_dim] = [B, 196, 512]
        
        # Reshape ViT features to match CNN spatial dimensions
        # ViT has 14x14=196 patches, CNN has 7x7=49 positions
        # Use adaptive pooling to downsample ViT patches
        vit_spatial = vit_proj.permute(0, 2, 1).view(B, self.out_dim, 14, 14)  # [B, 512, 14, 14]
        vit_downsampled = torch.nn.functional.adaptive_avg_pool2d(vit_spatial, (H, W))  # [B, 512, 7, 7]
        
        # Fusion
        fused = torch.cat([cnn_proj, vit_downsampled], dim=1)  # [B, out_dim*2, H, W] = [B, 1024, 7, 7]
        fused = self.fusion(fused)  # [B, out_dim, H, W] = [B, 512, 7, 7]
        
        return fused


class H2A_MTN(nn.Module):
    """
    Hybrid Hierarchical Attention Multi-Task Network
    
    Architecture:
    1. EfficientNetV2 Encoder - Local features
    2. Vision Transformer - Global context
    3. Cross-Attention Fusion
    4. Swin Transformer Decoder - Segmentation
    5. Multi-Task Heads - Classification, Segmentation, Grading
    """
    
    def __init__(self, 
                 num_classes=4,
                 num_grades=2,
                 img_size=224,
                 dropout=0.1):
        super().__init__()
        
        # 1. EfficientNetV2 Encoder
        self.efficientnet = EfficientNetV2Encoder('tf_efficientnetv2_s', pretrained=True)
        
        # 2. Vision Transformer
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=dropout
        )
        
        # 3. Feature Fusion
        # Fuse Effic256,  # EfficientNetV2-S final layer has 256 channels
        self.fusion = CrossAttentionFusion(
            cnn_dim=self.efficientnet.feature_dims[-1],  # 160 for EfficientNetV2-S
            vit_dim=768,
            out_dim=512
        )
        
        # 4. Swin Transformer Decoder for Segmentation
        self.swin_decoder = SwinTransformerDecoder(
            in_channels=512,
            num_classes=1,  # Binary segmentation
            depths=[2, 2, 2, 2]
        )
        
        # 5. Task-Specific Heads
        
        # Classification Head (uses ViT cls token + EfficientNet features)
        self.classification_head = nn.Sequential(
            nn.Linear(768 + 256, 512),  # ViT cls (768) + EfficientNet final (256)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Grading Head (similar to classification)
        self.grading_head = nn.Sequential(
            nn.Linear(768 + 256, 512),  # ViT cls (768) + EfficientNet final (256)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_grades)
        )
        
        # Global Average Pooling for CNN features
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224]
            
        Returns:
            Dict with keys:
                - 'classification': [B, num_classes]
                - 'segmentation': [B, 1, 224, 224]
                - 'grading': [B, num_grades]
        """
        B = x.shape[0]
        
        # 1. EfficientNet Encoder
        cnn_features = self.efficientnet(x)  # List of 4 feature maps
        cnn_final = cnn_features[-1]  # [B, 256, 7, 7]
        
        # 2. Vision Transformer
        vit_cls, vit_patches = self.vit(x)  # [B, 768], [B, 196, 768]
        
        # 3. Feature Fusion for Segmentation
        fused_features = self.fusion(cnn_final, vit_patches)  # [B, 512, 7, 7]
        
        # 4. Segmentation Decoder
        segmentation_logits = self.swin_decoder(fused_features)  # [B, 1, 224, 224]
        
        # 5. Classification and Grading
        # Combine ViT cls token with global pooled CNN features
        cnn_pooled = self.gap(cnn_final).view(B, -1)  # [B, 256]
        combined_features = torch.cat([vit_cls, cnn_pooled], dim=1)  # [B, 768+256=1024]
        
        classification_logits = self.classification_head(combined_features)
        grading_logits = self.grading_head(combined_features)
        
        return {
            'classification': classification_logits,
            'segmentation': segmentation_logits,
            'grading': grading_logits,
            'features': combined_features,  # For explainability
            'cnn_features': cnn_final,  # For Grad-CAM
            'vit_attention': vit_patches  # For attention visualization
        }


def test_model():
    """
    Test model instantiation and forward pass
    """
    model = H2A_MTN(num_classes=4, num_grades=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    outputs = model(x)
    
    print("\nOutput shapes:")
    print(f"Classification: {outputs['classification'].shape}")  # [2, 4]
    print(f"Segmentation: {outputs['segmentation'].shape}")  # [2, 1, 224, 224]
    print(f"Grading: {outputs['grading'].shape}")  # [2, 2]
    
    return model


if __name__ == "__main__":
    print("Testing H2A-MTN Architecture...")
    print("=" * 60)
    model = test_model()
    print("=" * 60)
    print("✓ Model architecture validated successfully!")
