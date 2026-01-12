# 🎓 Model Training Guide
## Brain Tumor Detection System

### Quick Start (3 Steps)

#### 1. Install Training Dependencies

```bash
# Install PyTorch training requirements
pip install -r requirements_training.txt
```

**Note:** This will install:
- PyTorch 2.1.0 (with CUDA support if GPU available)
- timm (PyTorch Image Models)
- Training utilities (tensorboard, tqdm, etc.)

#### 2. Verify Dataset Structure

Ensure your dataset is organized like this:
```
datasets/archive/
├── Training/
│   ├── glioma/      (images: *.jpg, *.png)
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

#### 3. Start Training

**Basic Training (50 epochs):**
```bash
python train_model.py
```

**Custom Configuration:**
```bash
python train_model.py --epochs 100 --batch_size 32 --lr 0.001
```

---

## 📋 Training Options

### Common Configurations

#### Quick Test (CPU or Small GPU)
```bash
python train_model.py --epochs 10 --batch_size 8 --num_workers 2
```

#### Standard Training (GPU Recommended)
```bash
python train_model.py --epochs 50 --batch_size 16 --lr 0.001
```

#### High-Performance Training (Powerful GPU)
```bash
python train_model.py --epochs 100 --batch_size 32 --lr 0.001 --weight_decay 0.0001
```

### All Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `datasets/archive` | Path to dataset directory |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `16` | Batch size (reduce if out of memory) |
| `--lr` | `0.001` | Learning rate |
| `--weight_decay` | `0.0001` | L2 regularization |
| `--img_size` | `224` | Input image size |
| `--num_classes` | `4` | Number of tumor classes |
| `--num_workers` | `4` | Data loading workers |
| `--device` | `auto` | Device (auto/cuda/cpu) |
| `--checkpoint_dir` | `checkpoints` | Where to save models |
| `--resume` | `None` | Resume from checkpoint |
| `--save_every` | `5` | Save checkpoint every N epochs |

---

## 🔄 Resume Training

If training is interrupted, resume from the last checkpoint:

```bash
python train_model.py --resume checkpoints/best_model.pth
```

Or from a specific epoch:

```bash
python train_model.py --resume checkpoints/checkpoint_epoch_25.pth
```

---

## 📊 Monitoring Training

### During Training

Watch the progress bars for:
- **Train Loss**: Should decrease over time
- **Train Acc**: Should increase over time
- **Val Loss**: Should decrease (if increasing, model is overfitting)
- **Val Acc**: Should increase (this is your main metric)

### Per-Class Accuracy

After each validation epoch, you'll see:
```
Per-class Accuracy:
  glioma      : 92.50% (148/160)
  meningioma  : 88.75% (142/160)
  notumor     : 95.00% (152/160)
  pituitary   : 91.25% (146/160)
```

This shows which tumor types the model performs best/worst on.

### Training History

After training, check `checkpoints/training_history.json`:
```json
{
  "train_loss": [0.8234, 0.5123, ...],
  "train_acc": [65.23, 78.45, ...],
  "val_loss": [0.7654, 0.4897, ...],
  "val_acc": [68.75, 82.34, ...]
}
```

---

## 💾 Saved Checkpoints

After training, you'll find in `checkpoints/`:

- `best_model.pth` - **Best performing model** (highest val accuracy)
- `checkpoint_epoch_5.pth` - Checkpoint after epoch 5
- `checkpoint_epoch_10.pth` - Checkpoint after epoch 10
- ... (every 5 epochs by default)
- `training_history.json` - Training metrics history

**Use `best_model.pth` for inference!**

---

## 🎯 Expected Results

### Training Time Estimates

| Hardware | Batch Size | Time per Epoch | 50 Epochs |
|----------|------------|----------------|-----------|
| CPU | 8 | ~20-30 min | ~16-25 hours |
| GPU (GTX 1660) | 16 | ~3-5 min | ~2.5-4 hours |
| GPU (RTX 3060) | 32 | ~1-2 min | ~1-2 hours |
| GPU (RTX 4090) | 64 | ~30-60 sec | ~25-50 min |

### Accuracy Benchmarks

| Epochs | Expected Val Accuracy |
|--------|---------------------|
| 10 | 70-80% |
| 25 | 85-92% |
| 50 | 90-95% |
| 100 | 93-97% |

**Note:** Actual results depend on dataset quality and size.

---

## 🐛 Troubleshooting

### Out of Memory (CUDA)

**Error:** `RuntimeError: CUDA out of memory`

**Solution:** Reduce batch size
```bash
python train_model.py --batch_size 8
# or even smaller
python train_model.py --batch_size 4
```

### Slow Training (CPU)

If no GPU is detected, training will be very slow.

**Check GPU availability:**
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

**Solutions:**
- Install CUDA-enabled PyTorch
- Use Google Colab (free GPU)
- Reduce epochs for testing

### Dataset Not Found

**Error:** `Warning: .../glioma does not exist`

**Solution:** Update data path
```bash
python train_model.py --data_root "path/to/your/dataset"
```

### Model Not Loading

**Error:** When resuming from checkpoint

**Solution:** Ensure checkpoint is compatible
```bash
# Start fresh training if incompatible
python train_model.py
# Don't use --resume
```

---

## 📈 Advanced: Hyperparameter Tuning

### Learning Rate

Too high (>0.01): Training unstable, loss fluctuates
```bash
python train_model.py --lr 0.0001  # Try lower
```

Too low (<0.0001): Training too slow
```bash
python train_model.py --lr 0.001   # Try higher
```

### Batch Size

Larger batch = faster but needs more memory
```bash
# Try doubling if you have memory
python train_model.py --batch_size 32
```

Smaller batch = slower but more stable
```bash
# Use if out of memory
python train_model.py --batch_size 8
```

### Data Augmentation

Edit `scripts/dataset_loader.py` in `get_transforms()` function to adjust:
- Rotation degrees
- Flip probabilities
- Color jitter
- Affine transformations

---

## 🚀 Next Steps After Training

### 1. Evaluate Model
```python
# Test on new images
python evaluate_model.py --checkpoint checkpoints/best_model.pth
```

### 2. Use in Production

Update `backend/main.py` to load your trained model:
```python
model = H2A_MTN(num_classes=4)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 3. Export for Deployment

```python
# Convert to ONNX for faster inference
import torch.onnx
torch.onnx.export(model, dummy_input, "model.onnx")
```

---

## 📞 Support

### Common Issues

1. **Low accuracy after training**
   - Train for more epochs
   - Try different learning rates
   - Check dataset quality
   - Ensure dataset is balanced

2. **Overfitting (train acc >> val acc)**
   - Add more data augmentation
   - Increase weight decay
   - Reduce model size
   - Use dropout

3. **Underfitting (both acc low)**
   - Train longer
   - Increase learning rate
   - Reduce regularization
   - Use larger model

### Getting Help

1. Check training logs in console
2. Review `training_history.json`
3. Verify dataset is correct
4. Test with smaller subset first

---

## ✅ Training Checklist

Before you start:
- [ ] Installed `requirements_training.txt`
- [ ] Dataset in `datasets/archive/` folder
- [ ] GPU detected (optional but recommended)
- [ ] At least 8GB free disk space for checkpoints
- [ ] Checked dataset has all 4 classes

During training:
- [ ] Monitor train/val accuracy
- [ ] Watch for overfitting
- [ ] Check GPU utilization (if available)
- [ ] Ensure checkpoints are being saved

After training:
- [ ] Found `best_model.pth` in checkpoints/
- [ ] Val accuracy > 85% (for production use)
- [ ] Reviewed per-class accuracy
- [ ] Saved training_history.json

---

**Ready to train? Run:**
```bash
python train_model.py --epochs 50 --batch_size 16
```

**Good luck! 🎉🧠**
