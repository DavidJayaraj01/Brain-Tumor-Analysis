# Quick Start Guide
## Brain Tumor Detection System - H2A-MTN

**Get the system running in 10 minutes!**

---

## Prerequisites

✅ **Python 3.8+** installed  
✅ **Node.js 16+** installed  
✅ **8GB+ RAM**  
✅ **CUDA 11.8+** (optional, for GPU acceleration)

---

## Step 1: Clone Repository

```bash
git clone https://github.com/[username]/brain-tumor-detection.git
cd brain-tumor-detection
```

---

## Step 2: Setup Backend (FastAPI)

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model (94MB)
# Option 1: wget (Linux/Mac)
wget https://[model-hosting-url]/h2a_mtn_weights.onnx -P models/

# Option 2: Manual download
# Visit: https://[model-hosting-url]/h2a_mtn_weights.onnx
# Save to: backend/models/h2a_mtn_weights.onnx

# Start backend server
python main.py
```

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Model loaded successfully: h2a_mtn_weights.onnx
```

**✅ Backend is now running!** Leave this terminal open.

---

## Step 3: Setup Frontend (React)

Open a **NEW terminal** window:

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Expected output:**
```
  VITE v5.0.8  ready in 324 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h to show help
```

**✅ Frontend is now running!**

---

## Step 4: Use the Application

1. **Open your browser:** Navigate to `http://localhost:5173`

2. **Check API status:** You should see a **green badge** in the header saying "API Online"

3. **Upload an MRI scan:**
   - Click the upload area or drag-and-drop an image
   - Supported formats: JPEG, PNG
   - Max size: 10MB

4. **Analyze:** Click "Analyze MRI Scan" button

5. **View Results:**
   - **Classification:** Tumor type with confidence scores
   - **Segmentation:** Tumor mask overlay
   - **Grading:** Low-Grade vs High-Grade prediction
   - **Explainability:** Grad-CAM++ heatmap showing attention regions
   - **Uncertainty:** Trustworthiness score (High/Medium/Low trust)

---

## Step 5: Test with Sample Data

Use provided test samples:

```bash
# Download sample MRI scans
cd backend
mkdir test_samples
cd test_samples

# Download samples (replace with actual URLs)
wget https://[sample-data-url]/glioma_sample.jpg
wget https://[sample-data-url]/meningioma_sample.jpg
wget https://[sample-data-url]/pituitary_sample.jpg
wget https://[sample-data-url]/notumor_sample.jpg
```

Upload these samples to see different tumor types!

---

## Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError: No module named 'fastapi'`  
**Solution:** Make sure virtual environment is activated and dependencies installed:
```bash
pip install -r requirements.txt
```

**Problem:** `FileNotFoundError: model weights not found`  
**Solution:** Download model weights to `backend/models/` directory

**Problem:** Port 8000 already in use  
**Solution:** Change port in `backend/main.py`:
```python
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Change to 8001
```
And update frontend `src/services/api.ts` baseURL to match.

### Frontend Issues

**Problem:** `npm install` fails  
**Solution:** Clear npm cache and retry:
```bash
npm cache clean --force
npm install
```

**Problem:** "API Offline" badge shown  
**Solution:** 
1. Check backend is running on port 8000
2. Check firewall isn't blocking localhost:8000
3. Open `http://localhost:8000/health` in browser to verify API

**Problem:** CORS errors in browser console  
**Solution:** Backend already has CORS enabled. If issue persists, check backend logs.

---

## API Testing (Optional)

Test API directly with curl:

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model_info

# Get supported classes
curl http://localhost:8000/classes

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/mri_scan.jpg"
```

---

## Next Steps

### For Users:
- Read [README.md](README.md) for detailed usage guide
- Try different MRI scans to see model predictions
- Explore trustworthiness scores (high trust = automated, low trust = needs review)

### For Developers:
- See [documentation/](documentation/) for technical details
- Review [models/hybrid_architecture.py](models/hybrid_architecture.py) for architecture
- Check [scripts/train.py](scripts/train.py) to retrain model
- Read [IEEE_RESEARCH_PAPER.md](documentation/IEEE_RESEARCH_PAPER.md) for methodology

### For Researchers:
- Study [VIVA_QA.md](documentation/VIVA_QA.md) for detailed Q&A
- Review ablation studies in research paper
- Cite this work if you use it (see README for citation)

---

## Performance Expectations

| Metric | Value |
|--------|-------|
| **Inference Time** | 0.74s (with TensorRT) / 2.1s (ONNX only) |
| **Accuracy** | 97.8% (classification) |
| **Dice Score** | 0.913 (segmentation) |
| **Model Size** | 94MB |
| **Memory Usage** | ~2GB (backend) + ~500MB (frontend) |

---

## System Requirements

### Minimum:
- CPU: Intel Core i5 / AMD Ryzen 5
- RAM: 8GB
- Inference: ~2-3 seconds (CPU mode)

### Recommended:
- CPU: Intel Core i7 / AMD Ryzen 7
- RAM: 16GB
- GPU: NVIDIA RTX 3060 (8GB VRAM)
- Inference: ~0.7 seconds (GPU mode with TensorRT)

---

## Support

- **Issues:** [GitHub Issues](https://github.com/[username]/brain-tumor-detection/issues)
- **Email:** [your.email@example.com]
- **Documentation:** [Full README](README.md)

---

## Quick Command Reference

```bash
# Start backend
cd backend && python main.py

# Start frontend
cd frontend && npm run dev

# Run tests (if available)
cd backend && pytest tests/

# Build frontend for production
cd frontend && npm run build

# Benchmark model
cd backend && python scripts/deployment_optimization.py --benchmark
```

---

**🎉 You're all set! The system is now running.**

**Access the web application at:** `http://localhost:5173`

**Happy diagnosing! 🧠**
