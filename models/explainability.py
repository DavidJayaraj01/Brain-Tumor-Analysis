"""
Explainability and Uncertainty Estimation Modules
- Grad-CAM++ for visual explanations
- Monte-Carlo Dropout for uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np
import cv2


class GradCAMPlusPlus:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM++)
    Provides visual explanations for model predictions
    
    Reference: Chattopadhyay et al., "Grad-CAM++: Generalized Gradient-based 
               Visual Explanations for Deep Convolutional Networks"
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Args:
            model: PyTorch model
            target_layer: Name of layer to visualize (e.g., 'backbone.features.16')
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        target = self._get_target_layer()
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)
    
    def _get_target_layer(self) -> nn.Module:
        """Get the target layer from model"""
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                return module
        raise ValueError(f"Layer {self.target_layer} not found in model")
    
    def generate_cam(self, 
                     input_tensor: torch.Tensor,
                     target_class: int,
                     eigen_smooth: bool = True) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap
        
        Args:
            input_tensor: Input image [1, C, H, W]
            target_class: Target class index for gradient computation
            eigen_smooth: Whether to use eigenvalues for smoothing
            
        Returns:
            CAM heatmap [H, W] normalized to [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        with torch.enable_grad():
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            
            # Get target class output
            if isinstance(output, dict):
                # Multi-task model
                task_output = output['classification']
            else:
                task_output = output
            
            target_score = task_output[0, target_class]
            
            # Backward pass
            self.model.zero_grad()
            target_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        b, c, h, w = activations.shape
        
        # Compute weights using Grad-CAM++ formula
        numerator = gradients ** 2
        denominator = 2 * (gradients ** 2) + (activations * (gradients ** 3)).sum(dim=(2, 3), keepdim=True)
        denominator = torch.clamp(denominator, min=1e-8)
        
        weights = numerator / denominator
        weights = weights.sum(dim=(2, 3), keepdim=True)  # Global average pooling
        
        # Compute CAM
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)
        
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        # Upsample to input size
        if input_tensor.shape[-2:] != cam.shape[-2:]:
            cam = F.interpolate(cam, size=input_tensor.shape[-2:], 
                              mode='bilinear', align_corners=False)
        
        return cam[0, 0].cpu().numpy()
    
    def overlay_on_image(self, 
                        image: np.ndarray, 
                        cam: np.ndarray,
                        colormap: str = 'jet',
                        alpha: float = 0.4) -> np.ndarray:
        """
        Overlay CAM on original image
        
        Args:
            image: Original image [H, W] or [H, W, 3]
            cam: CAM heatmap [H, W]
            colormap: OpenCV colormap name
            alpha: Transparency of overlay
            
        Returns:
            Overlaid image [H, W, 3]
        """
        # Ensure image is 3-channel
        if len(image.shape) == 2:
            image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Normalize CAM to 0-255
        cam_normalized = (cam * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
        
        # Overlay
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlaid


class MonteCarloDropout:
    """
    Monte Carlo Dropout for Uncertainty Estimation
    Performs multiple forward passes with different dropout masks
    
    Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation"
    """
    
    def __init__(self, 
                 model: nn.Module,
                 num_samples: int = 30,
                 dropout_rate: float = 0.5):
        """
        Args:
            model: PyTorch model with dropout layers
            num_samples: Number of MC samples for uncertainty estimation
            dropout_rate: Dropout rate during inference
        """
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
    
    def enable_dropout(self):
        """Enable dropout during inference"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep in train mode to apply dropout
    
    def disable_dropout(self):
        """Disable dropout during inference"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
    
    def estimate_uncertainty(self, 
                            input_tensor: torch.Tensor,
                            task: str = 'classification') -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty through MC-Dropout sampling
        
        Args:
            input_tensor: Input tensor [B, C, H, W]
            task: Task type ('classification', 'segmentation', 'grading')
            
        Returns:
            Dictionary with:
                - 'predictions': Mean predictions
                - 'uncertainties': Prediction variance (aleatoric)
                - 'std': Standard deviation
                - 'confidence': Max probability with uncertainty
        """
        self.model.eval()
        self.enable_dropout()
        
        predictions_list = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(input_tensor)
                
                if isinstance(output, dict):
                    task_output = output[task]
                else:
                    task_output = output
                
                predictions_list.append(task_output)
        
        self.disable_dropout()
        
        # Stack predictions: [num_samples, B, ...]
        stacked_predictions = torch.stack(predictions_list)
        
        # Compute statistics
        mean_predictions = stacked_predictions.mean(dim=0)
        std_predictions = stacked_predictions.std(dim=0)
        variance_predictions = (stacked_predictions.var(dim=0))
        
        # Convert to numpy
        if task in ['classification', 'grading']:
            probs = F.softmax(mean_predictions, dim=1)
            confidence = probs.max(dim=1).values
            
            results = {
                'predictions': mean_predictions.cpu().numpy(),
                'probabilities': probs.cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'uncertainties': variance_predictions.mean(dim=1).cpu().numpy(),
                'std': std_predictions.mean(dim=1).cpu().numpy(),
                'all_samples': stacked_predictions.cpu().numpy()
            }
        else:  # Segmentation
            results = {
                'predictions': mean_predictions.cpu().numpy(),
                'uncertainties': variance_predictions.mean(dim=(2, 3)).cpu().numpy(),
                'std': std_predictions.mean(dim=(2, 3)).cpu().numpy(),
                'all_samples': stacked_predictions.cpu().numpy()
            }
        
        return results
    
    def epistemic_aleatoric_decomposition(self,
                                         input_tensor: torch.Tensor,
                                         task: str = 'classification') -> Dict[str, np.ndarray]:
        """
        Decompose uncertainty into epistemic (model) and aleatoric (data) components
        
        Returns:
            Dictionary with epistemic and aleatoric uncertainties
        """
        self.model.eval()
        self.enable_dropout()
        
        predictions_list = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(input_tensor)
                
                if isinstance(output, dict):
                    task_output = output[task]
                else:
                    task_output = output
                
                if task in ['classification', 'grading']:
                    probs = F.softmax(task_output, dim=1)
                else:
                    probs = F.softmax(task_output, dim=1)
                
                predictions_list.append(probs)
        
        self.disable_dropout()
        
        # Stack: [num_samples, B, num_classes, ...]
        stacked_probs = torch.stack(predictions_list)
        
        # Aleatoric uncertainty: E[Var(y|x)]
        variance_per_sample = stacked_probs.var(dim=0)
        aleatoric = variance_per_sample.mean(dim=1)
        
        # Epistemic uncertainty: Var[E(y|x)]
        mean_probs = stacked_probs.mean(dim=0)
        epistemic = mean_probs.var(dim=1)
        
        return {
            'aleatoric': aleatoric.cpu().numpy(),
            'epistemic': epistemic.cpu().numpy(),
            'total': (aleatoric + epistemic).cpu().numpy()
        }


class UncertaintyCalibration:
    """
    Calibrate uncertainty estimates for clinical reliability
    """
    
    @staticmethod
    def expected_calibration_error(predictions: np.ndarray,
                                   labels: np.ndarray,
                                   n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE)
        Measures difference between predicted confidence and actual accuracy
        
        Lower ECE indicates better calibration
        """
        confidences = predictions.max(axis=1)
        predicted_labels = predictions.argmax(axis=1)
        accuracies = (predicted_labels == labels).astype(float)
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accs = []
        bin_confs = []
        
        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_accs.append(accuracies[mask].mean())
                bin_confs.append(confidences[mask].mean())
            else:
                bin_accs.append(0)
                bin_confs.append(0)
        
        ece = np.mean(np.abs(np.array(bin_accs) - np.array(bin_confs)))
        return ece
    
    @staticmethod
    def temperature_scaling(logits: np.ndarray,
                          labels: np.ndarray) -> float:
        """
        Learn temperature parameter T for calibration
        Logits are divided by T before softmax for calibration
        """
        from scipy.optimize import minimize
        
        def nll_loss(T):
            scaled_logits = logits / T
            probs = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
            nll = -np.log(probs[np.arange(len(labels)), labels]).mean()
            return nll
        
        result = minimize(nll_loss, x0=1.0, method='Nelder-Mead')
        return result.x[0]
    
    @staticmethod
    def confidence_thresholding(predictions: np.ndarray,
                               uncertainties: np.ndarray,
                               threshold: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply confidence thresholding for clinical safety
        Only return predictions with sufficient confidence
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            threshold: Confidence threshold
            
        Returns:
            Tuple of (filtered_predictions, rejection_mask)
        """
        confidences = predictions.max(axis=1)
        mask = confidences >= threshold
        
        predictions_filtered = predictions[mask]
        rejection_mask = ~mask
        
        return predictions_filtered, rejection_mask


class ExplainabilityReport:
    """
    Generate comprehensive explainability report for clinical use
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradcam = None
        self.mc_dropout = None
    
    def generate_report(self,
                       image: torch.Tensor,
                       target_class: int,
                       save_dir: str = None) -> Dict:
        """
        Generate complete explainability report
        
        Returns:
            Dictionary with:
                - grad_cam_heatmap
                - uncertainty_estimates
                - confidence_scores
                - clinical_recommendation
        """
        report = {}
        
        # 1. Grad-CAM++ visualization
        if self.gradcam is None:
            self.gradcam = GradCAMPlusPlus(self.model, target_layer='local_extractor.backbone.blocks.11')
        
        cam_heatmap = self.gradcam.generate_cam(image, target_class)
        report['grad_cam_heatmap'] = cam_heatmap
        
        # 2. MC-Dropout uncertainty
        if self.mc_dropout is None:
            self.mc_dropout = MonteCarloDropout(self.model, num_samples=30)
        
        uncertainty_data = self.mc_dropout.estimate_uncertainty(image, task='classification')
        report['uncertainties'] = uncertainty_data
        
        # 3. Epistemic vs Aleatoric
        uncertainty_decomp = self.mc_dropout.epistemic_aleatoric_decomposition(image)
        report['epistemic_uncertainty'] = uncertainty_decomp['epistemic']
        report['aleatoric_uncertainty'] = uncertainty_decomp['aleatoric']
        
        # 4. Clinical recommendation
        confidence = uncertainty_data['confidence'][0]
        total_uncertainty = uncertainty_decomp['total'][0]
        
        if confidence > 0.9 and total_uncertainty < 0.1:
            report['clinical_recommendation'] = 'HIGH CONFIDENCE - Proceed with diagnosis'
        elif confidence > 0.75 and total_uncertainty < 0.2:
            report['clinical_recommendation'] = 'MODERATE CONFIDENCE - Consider specialist review'
        else:
            report['clinical_recommendation'] = 'LOW CONFIDENCE - Additional imaging recommended'
        
        return report


if __name__ == "__main__":
    from models.architecture import build_unified_model
    
    # Create model
    model = build_unified_model(pretrained=False)
    
    # Test Grad-CAM++
    print("Testing Grad-CAM++...")
    gradcam = GradCAMPlusPlus(model, target_layer='local_extractor.backbone.blocks.11')
    
    # Dummy input
    x = torch.randn(1, 4, 224, 224)
    cam = gradcam.generate_cam(x, target_class=1)
    print(f"CAM shape: {cam.shape}")
    print(f"CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
    
    # Test MC-Dropout
    print("\nTesting MC-Dropout...")
    mc_dropout = MonteCarloDropout(model, num_samples=5)
    uncertainty = mc_dropout.estimate_uncertainty(x, task='classification')
    print(f"Predictions shape: {uncertainty['predictions'].shape}")
    print(f"Confidence: {uncertainty['confidence']}")
    print(f"Uncertainty: {uncertainty['uncertainties']}")
