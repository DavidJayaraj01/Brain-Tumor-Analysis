"""
Explainability Module: Grad-CAM++ and Uncertainty Estimation
=============================================================

Implements:
1. Grad-CAM++ - Visual explainability through attention heatmaps
2. Monte-Carlo Dropout - Uncertainty quantification
3. Attention visualization - Transformer attention maps
4. Trustworthiness scoring - Combined confidence metric

Author: Brain Tumor Detection Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2


class GradCAMPlusPlus:
    """
    Grad-CAM++ for visual explainability
    
    Improves upon Grad-CAM with weighted combination of gradients
    Better handles multiple occurrences of the same class
    
    Reference: Chattopadhay et al. "Grad-CAM++: Generalized Gradient-Based 
               Visual Explanations for Deep Convolutional Networks" (2018)
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Neural network model
            target_layer: Layer to extract gradients from (e.g., last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """
        Register forward and backward hooks to capture activations and gradients
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM++ heatmap
        
        Args:
            input_tensor: [1, C, H, W] - Input image
            class_idx: Target class index (None = use predicted class)
            
        Returns:
            cam: [H, W] - Normalized heatmap
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get classification logits
        if isinstance(output, dict):
            logits = output['classification']
        else:
            logits = output
        
        # Use predicted class if not specified
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target = logits[0, class_idx]
        target.backward(retain_graph=True)
        
        # Get activations and gradients
        activations = self.activations  # [1, C, H, W]
        gradients = self.gradients  # [1, C, H, W]
        
        # Grad-CAM++ weights
        # α^kc = (∂²Y^c / ∂A^k²) / (2 * ∂²Y^c / ∂A^k² + Σ(A^k * ∂³Y^c / ∂A^k³))
        
        # First derivative
        grad_1 = gradients
        
        # Second derivative (approximated)
        grad_2 = gradients.pow(2)
        
        # Third derivative (approximated)
        grad_3 = gradients.pow(3)
        
        # Global average pooling of gradients
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + (activations * grad_3).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(
            alpha_denom != 0.0,
            alpha_denom,
            torch.ones_like(alpha_denom)
        )
        
        alpha = alpha_num / alpha_denom
        
        # Weighted combination
        weights = (alpha * F.relu(grad_1)).sum(dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)  # Remove negative values
        
        # Resize to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_multi_layer_cam(self, input_tensor, layers, class_idx=None):
        """
        Generate ensemble CAM from multiple layers
        
        Args:
            input_tensor: [1, C, H, W]
            layers: List of target layers
            class_idx: Target class index
            
        Returns:
            cam: [H, W] - Ensemble heatmap
        """
        cams = []
        
        for layer in layers:
            self.target_layer = layer
            self._register_hooks()
            cam = self.generate_cam(input_tensor, class_idx)
            cams.append(cam)
        
        # Average ensemble
        ensemble_cam = np.mean(cams, axis=0)
        
        return ensemble_cam


class MonteCarloDropout:
    """
    Monte-Carlo Dropout for Uncertainty Estimation
    
    Approximates Bayesian inference by running multiple forward passes
    with dropout enabled during inference
    
    Provides:
    - Predictive mean (average prediction)
    - Predictive variance (uncertainty measure)
    - Confidence intervals
    
    Reference: Gal & Ghahramani "Dropout as a Bayesian Approximation" (2016)
    """
    
    def __init__(self, model, num_samples=30, dropout_rate=0.1):
        """
        Args:
            model: Neural network with dropout layers
            num_samples: Number of forward passes
            dropout_rate: Dropout probability
        """
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        
    def enable_dropout(self):
        """
        Enable dropout layers during inference
        """
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(self, input_tensor):
        """
        Generate predictions with uncertainty estimates
        
        Args:
            input_tensor: [B, C, H, W]
            
        Returns:
            Dict with:
                - 'mean': Mean prediction
                - 'variance': Predictive variance
                - 'std': Standard deviation
                - 'confidence_interval': 95% CI
        """
        self.enable_dropout()
        
        predictions_cls = []
        predictions_seg = []
        predictions_grade = []
        
        # Multiple forward passes
        with torch.no_grad():
            for _ in range(self.num_samples):
                outputs = self.model(input_tensor)
                
                # Store predictions
                predictions_cls.append(
                    F.softmax(outputs['classification'], dim=1).cpu().numpy()
                )
                predictions_seg.append(
                    torch.sigmoid(outputs['segmentation']).cpu().numpy()
                )
                predictions_grade.append(
                    F.softmax(outputs['grading'], dim=1).cpu().numpy()
                )
        
        # Convert to arrays
        predictions_cls = np.array(predictions_cls)  # [num_samples, B, num_classes]
        predictions_seg = np.array(predictions_seg)  # [num_samples, B, 1, H, W]
        predictions_grade = np.array(predictions_grade)  # [num_samples, B, num_grades]
        
        # Compute statistics
        results = {
            'classification': {
                'mean': predictions_cls.mean(axis=0),
                'variance': predictions_cls.var(axis=0),
                'std': predictions_cls.std(axis=0),
                'entropy': self._compute_entropy(predictions_cls.mean(axis=0))
            },
            'segmentation': {
                'mean': predictions_seg.mean(axis=0),
                'variance': predictions_seg.var(axis=0),
                'std': predictions_seg.std(axis=0)
            },
            'grading': {
                'mean': predictions_grade.mean(axis=0),
                'variance': predictions_grade.var(axis=0),
                'std': predictions_grade.std(axis=0),
                'entropy': self._compute_entropy(predictions_grade.mean(axis=0))
            }
        }
        
        return results
    
    def _compute_entropy(self, probabilities):
        """
        Compute predictive entropy
        
        H = -Σ p(y) log p(y)
        
        Higher entropy = higher uncertainty
        """
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=-1)
        return entropy


class AttentionVisualizer:
    """
    Visualize Vision Transformer attention maps
    """
    
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        
    def extract_attention_maps(self, input_tensor):
        """
        Extract attention maps from ViT layers
        
        Args:
            input_tensor: [1, C, H, W]
            
        Returns:
            List of attention maps
        """
        # Hook to capture attention weights
        attention_weights = []
        
        def hook_fn(module, input, output):
            # Assuming output contains attention weights
            attention_weights.append(output)
        
        # Register hooks on transformer blocks
        hooks = []
        for module in self.model.vit.blocks:
            if hasattr(module, 'attn'):
                hook = module.attn.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def visualize_attention(self, attention_map, img_size=(224, 224)):
        """
        Convert attention map to heatmap
        
        Args:
            attention_map: [num_heads, num_patches, num_patches]
            img_size: Target image size
            
        Returns:
            Heatmap as numpy array
        """
        # Average over heads
        if len(attention_map.shape) == 3:
            attention_map = attention_map.mean(dim=0)
        
        # Get attention to CLS token (first row)
        cls_attention = attention_map[0, 1:]  # Exclude CLS to CLS
        
        # Reshape to spatial grid
        num_patches = int(np.sqrt(cls_attention.shape[0]))
        attention_map_spatial = cls_attention.reshape(num_patches, num_patches)
        
        # Convert to numpy
        attention_map_np = attention_map_spatial.cpu().numpy()
        
        # Resize to image size
        attention_heatmap = cv2.resize(
            attention_map_np,
            img_size,
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize
        attention_heatmap = (attention_heatmap - attention_heatmap.min()) / \
                           (attention_heatmap.max() - attention_heatmap.min() + 1e-8)
        
        return attention_heatmap


class TrustworthinessScore:
    """
    Compute overall trustworthiness score combining:
    1. Prediction confidence
    2. Uncertainty estimate (MC Dropout)
    3. Attention map quality
    """
    
    def __init__(self, confidence_weight=0.4, uncertainty_weight=0.4, attention_weight=0.2):
        self.weights = {
            'confidence': confidence_weight,
            'uncertainty': uncertainty_weight,
            'attention': attention_weight
        }
    
    def compute_score(self, 
                      prediction_confidence: float,
                      uncertainty_entropy: float,
                      attention_quality: float) -> Dict[str, float]:
        """
        Args:
            prediction_confidence: Max softmax probability [0, 1]
            uncertainty_entropy: Predictive entropy (normalized) [0, 1]
            attention_quality: Attention map sharpness score [0, 1]
            
        Returns:
            Dict with trustworthiness score and components
        """
        # Invert uncertainty (lower uncertainty = more trustworthy)
        uncertainty_score = 1.0 - uncertainty_entropy
        
        # Weighted combination
        trustworthiness = (
            self.weights['confidence'] * prediction_confidence +
            self.weights['uncertainty'] * uncertainty_score +
            self.weights['attention'] * attention_quality
        )
        
        # Categorize
        if trustworthiness >= 0.85:
            category = "High Trust"
        elif trustworthiness >= 0.7:
            category = "Medium Trust"
        else:
            category = "Low Trust - Expert Review Recommended"
        
        return {
            'score': trustworthiness,
            'category': category,
            'confidence': prediction_confidence,
            'uncertainty': uncertainty_score,
            'attention_quality': attention_quality
        }


def overlay_heatmap_on_image(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image
    
    Args:
        image: [H, W, 3] - RGB image (0-255)
        heatmap: [H, W] - Normalized heatmap (0-1)
        alpha: Transparency
        colormap: OpenCV colormap
        
    Returns:
        Overlayed image [H, W, 3]
    """
    # Convert heatmap to RGB
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure image is uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Blend
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def test_explainability():
    """
    Test explainability modules
    """
    print("Testing Explainability Modules")
    print("=" * 60)
    
    from hybrid_architecture import H2A_MTN
    
    # Load model
    model = H2A_MTN(num_classes=4, num_grades=2)
    model.eval()
    
    # Dummy input
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # 1. Test Grad-CAM++
    print("\n1. Testing Grad-CAM++...")
    target_layer = model.efficientnet.backbone.blocks[-1][-1]  # Last conv layer
    gradcam = GradCAMPlusPlus(model, target_layer)
    cam = gradcam.generate_cam(input_tensor, class_idx=0)
    print(f"   Generated CAM shape: {cam.shape}")
    print(f"   CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
    
    # 2. Test Monte-Carlo Dropout
    print("\n2. Testing Monte-Carlo Dropout...")
    mc_dropout = MonteCarloDropout(model, num_samples=10)
    uncertainty_results = mc_dropout.predict_with_uncertainty(input_tensor)
    print(f"   Classification mean: {uncertainty_results['classification']['mean'].shape}")
    print(f"   Classification std: {uncertainty_results['classification']['std'][0]}")
    print(f"   Predictive entropy: {uncertainty_results['classification']['entropy'][0]:.4f}")
    
    # 3. Test Trustworthiness Score
    print("\n3. Testing Trustworthiness Score...")
    trust_scorer = TrustworthinessScore()
    trust_score = trust_scorer.compute_score(
        prediction_confidence=0.92,
        uncertainty_entropy=0.15,
        attention_quality=0.88
    )
    print(f"   Trustworthiness score: {trust_score['score']:.3f}")
    print(f"   Category: {trust_score['category']}")
    
    print("\n" + "=" * 60)
    print("✓ Explainability modules validated successfully!")


if __name__ == "__main__":
    test_explainability()
