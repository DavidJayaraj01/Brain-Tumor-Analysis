# 🎯 Complete Training Guide - RTX 4060 GPU

## ✅ Setup Complete!

Your system has:
- ✅ RTX 4060 GPU (8GB VRAM)
- ✅ 5,712 training images + 1,311 test images (archive dataset)
- ✅ 3,064 additional .mat files (brainTumorDataPublic datasets)
- ✅ PyTorch with CUDA 11.8 (installing...)

---

## 🚀 Start Training (After GPU Setup)

### Step 1: Verify GPU is Workinge1

```bash
python check_training_setup.py
```

You should see:
```
✅ Using GPU: NVIDIA GeForce RTX 4060
   GPU Memory: 8.00 GB
```

### Step 2: Start Training with Optimal Settings

**Option A: Use GPU-optimized script (Recommended)**
```bash
python train_gpu.py
```


This automatically uses:
- Batch size: 32 (optimal for 8GB VRAM)
- Epochs: 50
- Workers: 8 (fast data loading)
- Learning rate: 0.001

**Option B: Custom training**
```bash
# Standard training (50 epochs, ~30 minutes)
python train_model.py --epochs 50 --batch_size 32 --num_workers 8

# Quick test (10 epochs, ~6 minutes)
python train_model.py --epochs 10 --batch_size 32 --num_workers 8

# High accuracy (100 epochs, ~1 hour)
python train_model.py --epochs 100 --batch_size 32 --num_workers 8

# Maximum batch size (if you have 8GB free)
python train_model.py --epochs 50 --batch_size 64 --num_workers 8
```

---

## ⚡ Performance Expectations

### RTX 4060 Training Speed

| Epochs | Batch Size | Time | Expected Accuracy |
|--------|-----------|------|------------------|
| 10 | 32 | ~6 min | 75-82% |
| 25 | 32 | ~15 min | 85-90% |
| 50 | 32 | ~30 min | 90-95% |
| 100 | 32 | ~1 hour | 93-97% |

**Your GPU is ~15x faster than CPU!** 🔥

### GPU Utilization

Monitor your GPU:
```bash
# In another terminal
nvidia-smi -l 1
```

You should see:
- GPU Utilization: 95-100%
- Memory Usage: ~6-7 GB (with batch_size=32)
- Temperature: 60-75°C (normal)

---

## 📊 What You'll See During Training

```
Epoch 1 [Train]: 100%|████████| 178/178 [00:35<00:00, loss: 0.8234, acc: 65.23%]
Epoch 1 [Valid]: 100%|████████| 41/41 [00:05<00:00, loss: 0.7654, acc: 68.75%]

  Per-class Accuracy:
    glioma      : 72.50% (217/300)
    meningioma  : 68.75% (210/306)
    notumor     : 75.00% (304/405)
    pituitary   : 65.25% (195/300)

📊 Epoch 1/50 Summary:
  Train Loss: 0.8234 | Train Acc: 65.23%
  Val Loss:   0.7654 | Val Acc:   68.75%
  LR: 0.001000
  
  🎉 New best validation accuracy: 68.75%
  💾 Saved BEST model: checkpoints/best_model.pth
```

With GPU, each epoch takes ~30-40 seconds (vs 8-10 minutes on CPU!)

---

## 💾 Saved Models

After training:

```
checkpoints/
├── best_model.pth              ← Use this! (highest accuracy)
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
├── ...
├── checkpoint_epoch_50.pth
└── training_history.json       ← Training metrics
```

---

## 🎯 Recommended Training Strategy

### For Best Results:

1. **Quick test (verify GPU works):**
   ```bash
   python train_model.py --epochs 5 --batch_size 32
   ```
   Time: ~3 minutes
   Expected: 70-75% accuracy

2. **Standard training (recommended):**
   ```bash
   python train_model.py --epochs 50 --batch_size 32
   ```
   Time: ~30 minutes
   Expected: 90-95% accuracy

3. **High accuracy (if needed):**
   ```bash
   python train_model.py --epochs 100 --batch_size 32
   ```
   Time: ~1 hour
   Expected: 93-97% accuracy

---

## ⚙️ Optimizing for Your GPU

### Batch Size Guide (RTX 4060 - 8GB VRAM)

| Batch Size | VRAM Usage | Speed | Recommendation |
|-----------|-----------|-------|----------------|
| 16 | ~4 GB | Slower | If other apps running |
| 32 | ~6-7 GB | **Optimal** | **Recommended** |
| 48 | ~7-8 GB | Faster | If nothing else running |
| 64 | ~8+ GB | Fastest | May cause OOM |

**Start with 32, increase if you have memory!**

### If You Get "CUDA out of memory":

```bash
# Reduce batch size
python train_model.py --batch_size 24

# Or even smaller
python train_model.py --batch_size 16
```

---

## 🔄 Using the .mat Dataset Files

You have 3,064 additional .mat files. To use them:

1. They need preprocessing (MATLAB format to images)
2. The archive dataset (5,712 images) is already preprocessed
3. Recommended: Use archive dataset for now (it's ready!)

If you want to use .mat files later, we can create a converter.

---

## 📈 Monitoring Training

### Watch Progress in Real-Time

**Terminal 1: Training**
```bash
python train_gpu.py
```

**Terminal 2: GPU Monitor**
```bash
nvidia-smi -l 1
```

### Check Training Progress

Open `checkpoints/training_history.json` to see:
- Train/Val loss over time
- Train/Val accuracy trends
- Identify overfitting

---

## 🎊 After Training

### 1. Find Your Best Model

```bash
# Located at:
checkpoints/best_model.pth
```

### 2. Check Final Accuracy

Look at the last epoch output:
```
Best Validation Accuracy: 94.25%
```

### 3. Use Model in Production

Update `backend/main.py` to load your trained model:

```python
import torch
from models.hybrid_architecture import H2A_MTN

# Load trained model
model = H2A_MTN(num_classes=4)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Move to GPU for faster inference
device = torch.device('cuda')
model = model.to(device)
```

### 4. Test the Model

```python
# Test on a new image
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('test_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    prediction = output['classification'].argmax(1).item()

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(f"Prediction: {classes[prediction]}")
```

---

## 🆘 Troubleshooting

### GPU Not Detected

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Should print: True
```

If False:
1. Ensure NVIDIA drivers are installed
2. Run `nvidia-smi` to verify GPU
3. Reinstall PyTorch with CUDA

### Training Too Slow

- Check GPU utilization: `nvidia-smi`
- Should be 90-100% during training
- Increase `--num_workers` (try 8 or 12)
- Increase batch size if VRAM available

### Out of Memory

```bash
# Reduce batch size
python train_model.py --batch_size 24

# Or close other applications using GPU
```

### Model Accuracy Not Improving

- Train for more epochs (50-100)
- Check if overfitting (train acc >> val acc)
- Try different learning rates: `--lr 0.0005` or `--lr 0.002`

---

## ✅ Quick Start Checklist

Before training:
- [ ] PyTorch CUDA installed (wait for pip to finish)
- [ ] Run `python check_training_setup.py`
- [ ] Verify GPU detected
- [ ] Close unnecessary applications (free up VRAM)

During training:
- [ ] Monitor GPU usage with `nvidia-smi`
- [ ] Watch validation accuracy increasing
- [ ] Check VRAM doesn't exceed 8GB
- [ ] Note: Can stop anytime with Ctrl+C

After training:
- [ ] Find `best_model.pth` in checkpoints/
- [ ] Val accuracy > 90% for production use
- [ ] Review per-class accuracy
- [ ] Test model on new images

---

## 🚀 Ready to Train!

**Once PyTorch finishes installing, run:**

```bash
# Verify GPU
python check_training_setup.py

# Start training
python train_gpu.py
```

**Expected timeline:**
- Setup verification: 30 seconds
- Training 50 epochs: ~30 minutes
- Total: ~31 minutes to trained model! 🎉

**With RTX 4060, you can iterate quickly:**
- Try different hyperparameters
- Experiment with architectures
- Train multiple models

Enjoy your GPU-powered training! ⚡🧠
