"""
GPU-Optimized Training Configuration
====================================

Quick training with RTX 4060 optimized settings.

Usage:
    python train_gpu.py
    
This uses optimal batch size and settings for RTX 4060 (8GB VRAM).
"""

import subprocess
import sys

# Optimal settings for RTX 4060
CONFIG = {
    'epochs': 50,
    'batch_size': 32,      # Optimal for 8GB VRAM
    'lr': 0.001,
    'num_workers': 8,      # More workers for faster data loading
    'data_root': 'datasets/archive',
    'checkpoint_dir': 'checkpoints',
    'save_every': 5
}

def main():
    """Run training with GPU-optimized settings"""
    
    print("=" * 70)
    print("🚀 GPU-Optimized Training for RTX 4060")
    print("=" * 70)
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key:15s}: {value}")
    print()
    
    # Build command
    cmd = [
        sys.executable, 
        'train_model.py',
        '--epochs', str(CONFIG['epochs']),
        '--batch_size', str(CONFIG['batch_size']),
        '--lr', str(CONFIG['lr']),
        '--num_workers', str(CONFIG['num_workers']),
        '--data_root', CONFIG['data_root'],
        '--checkpoint_dir', CONFIG['checkpoint_dir'],
        '--save_every', str(CONFIG['save_every'])
    ]
    
    print("Starting training...")
    print("Press Ctrl+C to stop anytime (progress will be saved)\n")
    print("=" * 70)
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Your progress has been saved in checkpoints/")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
