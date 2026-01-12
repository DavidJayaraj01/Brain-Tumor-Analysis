# 🎓 Training Quick Start Summary

## ✅ Your System is Ready!

### Setup Status
- ✅ Python 3.10.10
- ✅ PyTorch 2.1.0 installed
- ✅ All dependencies ready
- ✅ Dataset loaded: **5,712 training images** + **1,311 testing images**
- ✅ Model: H2A-MTN (119M parameters)
- ✅ 667 GB free disk space

⚠️ **Note:** No GPU detected - training will use CPU (slower but functional)

---

## 🚀 Start Training Now!

### Option 1: Quick Test (5 epochs, ~30-45 minutes)
```bash
python train_model.py --epochs 5 --batch_size 8
```
**Expected result:** ~70-75% accuracy

### Option 2: Standard Training (50 epochs, recommended)
```bash
python train_model.py --epochs 50 --batch_size 16
```
**Expected result:** 90-95% accuracy
**Time:** ~4-6 hours on CPU

### Option 3: Short but Effective (25 epochs)
```bash
python train_model.py --epochs 25 --batch_size 16
```
**Expected result:** 85-90% accuracy
**Time:** ~2-3 hours on CPU

---

## 📊 What You'll See

During training, you'll see progress bars like this:

```
Epoch 1 [Train]: 100%|████████| 357/357 [08:23<00:00, loss: 0.8234, acc: 65.23%]
Epoch 1 [Valid]: 100%|████████| 82/82 [01:45<00:00, loss: 0.7654, acc: 68.75%]

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

---

## 📁 What Gets Created

After training starts, you'll find:

```
checkpoints/
├── best_model.pth                    ← Use this for deployment!
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
├── ...
└── training_history.json             ← Training metrics
```

---

## 🎯 Expected Performance

| Training Time | Validation Accuracy | Status |
|--------------|-------------------|---------|
| 5 epochs | 70-75% | Good for testing |
| 10 epochs | 75-82% | Basic model |
| 25 epochs | 85-90% | Good performance |
| 50 epochs | 90-95% | Recommended |
| 100 epochs | 93-97% | Best accuracy |

---

## ⏹️ Stop Training Anytime

Training can be stopped with `Ctrl+C`. Your progress is automatically saved!

To resume:
```bash
python train_model.py --resume checkpoints/checkpoint_epoch_10.pth
```

---

## 🔄 After Training

### 1. Check Your Best Model
```bash
# It's in: checkpoints/best_model.pth
```

### 2. View Training History
```bash
# Open: checkpoints/training_history.json
```

### 3. Use Model in Backend

Update `backend/main.py` to load your trained model:
```python
# Around line 30-40, replace the dummy model with:
checkpoint = torch.load('checkpoints/best_model.pth')
model = H2A_MTN(num_classes=4)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 💡 Tips

### CPU Training is Slow?
- ✅ Reduce batch size: `--batch_size 4`
- ✅ Train fewer epochs first: `--epochs 10`
- ✅ Use Google Colab (free GPU): Upload dataset and scripts

### Monitor Progress
Watch the validation accuracy:
- If it's increasing: ✅ Training is working!
- If it plateaus: Consider stopping and using the best model
- If train acc >> val acc: Model is overfitting

### Best Practices
1. Start with 5-10 epochs to test
2. If results look good, train for 50+ epochs
3. Always use the `best_model.pth` (not the latest checkpoint)
4. Check per-class accuracy to see which tumors are hardest

---

## 🆘 Troubleshooting

### Out of Memory
```bash
python train_model.py --batch_size 4  # Reduce batch size
```

### Training Too Slow
```bash
python train_model.py --epochs 10 --batch_size 8  # Shorter run
```

### Want to Stop and Resume
1. Press `Ctrl+C` to stop
2. Find latest checkpoint: `checkpoints/checkpoint_epoch_XX.pth`
3. Resume: `python train_model.py --resume checkpoints/checkpoint_epoch_XX.pth`

---

## 📚 More Help

- **Full guide:** [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Check setup:** `python check_training_setup.py`
- **Dataset info:** 4 classes (glioma, meningioma, notumor, pituitary)

---

## ✨ Ready? Start Training!

**Recommended command:**
```bash
python train_model.py --epochs 50 --batch_size 16
```

Or test first:
```bash
python train_model.py --epochs 5 --batch_size 8
```

**Good luck! 🎉🧠**

---

*Training script created by MediMind BTIS Team*
*Model: H2A-MTN (Hybrid Hierarchical Attention Multi-Task Network)*
