"""
Pre-Training Check Script
=========================

Verifies that everything is ready for training before you start.

Usage:
    python check_training_setup.py
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} (need 3.8+)")
        return False

def check_pytorch():
    """Check PyTorch installation"""
    print("\n🔥 Checking PyTorch...")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ GPU: {gpu_name}")
            print(f"   ✅ GPU Memory: {gpu_memory:.2f} GB")
            print(f"   ⚡ CUDA Version: {torch.version.cuda}")
        else:
            print(f"   ⚠️  No GPU detected (training will be slow)")
            print(f"   💡 CPU training is functional but ~10x slower")
        
        return True
    except ImportError:
        print("   ❌ PyTorch not installed")
        print("   💡 Run: pip install -r requirements_training.txt")
        return False

def check_dependencies():
    """Check other dependencies"""
    print("\n📦 Checking dependencies...")
    
    required = {
        'torchvision': 'torchvision',
        'timm': 'timm',
        'tqdm': 'tqdm',
        'PIL': 'Pillow',
        'numpy': 'numpy'
    }
    
    all_ok = True
    for module, package in required.items():
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} not installed")
            all_ok = False
    
    if not all_ok:
        print("\n   💡 Install missing packages:")
        print("      pip install -r requirements_training.txt")
    
    return all_ok

def check_dataset():
    """Check dataset structure"""
    print("\n📂 Checking dataset...")
    
    data_root = Path('datasets/archive')
    
    if not data_root.exists():
        print(f"   ❌ Dataset root not found: {data_root}")
        print("   💡 Ensure dataset is in: datasets/archive/")
        return False
    
    print(f"   ✅ Dataset root found: {data_root}")
    
    # Check splits
    splits = ['Training', 'Testing']
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"   ❌ {split} folder not found")
            continue
        
        print(f"\n   📁 {split}:")
        total_images = 0
        
        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"      ❌ {cls} folder missing")
                continue
            
            # Count images
            jpg_count = len(list(cls_dir.glob('*.jpg')))
            png_count = len(list(cls_dir.glob('*.png')))
            total = jpg_count + png_count
            total_images += total
            
            if total > 0:
                print(f"      ✅ {cls:12s}: {total:4d} images")
            else:
                print(f"      ⚠️  {cls:12s}: 0 images")
        
        print(f"      Total: {total_images} images")
    
    return True

def check_model():
    """Check model can be imported"""
    print("\n🔨 Checking model...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from models.hybrid_architecture import H2A_MTN
        
        print("   ✅ Model imports successfully")
        
        # Try to create model
        try:
            import torch
            model = H2A_MTN(num_classes=4, img_size=224)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   ✅ Model created: {total_params:,} parameters")
            return True
        except Exception as e:
            print(f"   ❌ Model creation failed: {e}")
            return False
            
    except ImportError as e:
        print(f"   ❌ Model import failed: {e}")
        return False

def check_training_script():
    """Check training script exists"""
    print("\n📝 Checking training script...")
    
    script = Path('train_model.py')
    if script.exists():
        print(f"   ✅ Training script found: {script}")
        return True
    else:
        print(f"   ❌ Training script not found: {script}")
        return False

def check_disk_space():
    """Check available disk space"""
    print("\n💾 Checking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(Path.cwd())
        
        free_gb = free / (1024**3)
        print(f"   Available: {free_gb:.2f} GB")
        
        if free_gb < 5:
            print(f"   ⚠️  Low disk space (need ~10GB for checkpoints)")
        else:
            print(f"   ✅ Sufficient disk space")
        
        return True
    except:
        print("   ⚠️  Could not check disk space")
        return True

def main():
    """Run all checks"""
    print("=" * 70)
    print("🔍 Pre-Training Setup Check")
    print("=" * 70)
    
    checks = [
        check_python_version(),
        check_pytorch(),
        check_dependencies(),
        check_dataset(),
        check_model(),
        check_training_script(),
        check_disk_space()
    ]
    
    print("\n" + "=" * 70)
    
    if all(checks):
        print("✅ ALL CHECKS PASSED!")
        print("=" * 70)
        print("\n🚀 You're ready to start training!")
        print("\nRun:")
        print("  python train_model.py --epochs 50 --batch_size 16")
        print("\nOr for a quick test:")
        print("  python train_model.py --epochs 5 --batch_size 8")
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 70)
        print("\n⚠️  Please fix the issues above before training")
        print("\nCommon fixes:")
        print("  1. Install dependencies:")
        print("     pip install -r requirements_training.txt")
        print("  2. Verify dataset location:")
        print("     datasets/archive/Training/")
        print("     datasets/archive/Testing/")
    
    print()

if __name__ == '__main__':
    main()
