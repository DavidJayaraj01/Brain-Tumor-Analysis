"""
Model Loader for Trained Brain Tumor Detection Model
Loads the trained H2A_MTN model from checkpoint
"""

import torch
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.hybrid_architecture import H2A_MTN

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage the trained model"""
    
    def __init__(self, checkpoint_path: str = None, device: str = None):
        """
        Initialize model loader
        
        Args:
            checkpoint_path: Path to model checkpoint (.pkl file)
            device: 'cuda' or 'cpu'
        """
        if checkpoint_path is None:
            checkpoint_path = str(project_root / 'checkpoints_full' / 'final_model_50epochs.pkl')
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.model = None
        self.model_info = None
        
    def load_model(self, num_classes: int = 4, img_size: int = 224):
        """
        Load trained model from checkpoint
        
        Args:
            num_classes: Number of tumor classes (default: 4)
            img_size: Input image size (default: 224)
            
        Returns:
            Loaded model in eval mode
        """
        try:
            logger.info(f"Loading model from: {self.checkpoint_path}")
            
            # Check if checkpoint exists
            if not Path(self.checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Create model architecture
            self.model = H2A_MTN(
                num_classes=num_classes,
                img_size=img_size
            )
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Store model info
            self.model_info = {
                'best_val_acc': checkpoint.get('best_val_acc', 'N/A'),
                'total_epochs': checkpoint.get('total_epochs_completed', checkpoint.get('epoch', 50)),
                'timestamp': checkpoint.get('timestamp', 'N/A')
            }
            
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"   Best Validation Accuracy: {self.model_info['best_val_acc']:.2f}%" 
                       if isinstance(self.model_info['best_val_acc'], (int, float)) 
                       else f"   Model Info: {self.model_info}")
            logger.info(f"   Device: {self.device}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor):
        """
        Run inference on input tensor
        
        Args:
            input_tensor: Preprocessed image tensor [B, C, H, W]
            
        Returns:
            Model predictions dict
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Ensure tensor is on correct device
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.to(self.device)
        else:
            input_tensor = torch.from_numpy(input_tensor).float().to(self.device)
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        return outputs
    
    def get_model_info(self):
        """Get model information"""
        return self.model_info


# Singleton instance
_model_loader_instance = None


def get_model_loader(checkpoint_path: str = None):
    """Get or create singleton model loader instance"""
    global _model_loader_instance
    
    if _model_loader_instance is None:
        _model_loader_instance = ModelLoader(checkpoint_path)
        _model_loader_instance.load_model()
    
    return _model_loader_instance
