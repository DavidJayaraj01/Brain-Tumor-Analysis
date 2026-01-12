# 🚀 Quick Start Guide - MediMind BТIS

## Get Up and Running in 2 Minutes

### Step 1: Start Backend Server
```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Step 2: Start Frontend (New Terminal)
```bash
cd frontend
npm run dev
```

You should see:
```
  VITE v7.3.1  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

### Step 3: Open Browser
Navigate to: **http://localhost:5173**

### Step 4: Upload & Analyze
1. Fill patient information (optional):
   - Age: e.g., 45
   - Sex: Male/Female/Other
   - Symptoms: e.g., "Headaches, vision changes"

2. Upload MRI scan:
   - Click upload area or drag-and-drop
   - Supported formats: PNG, JPG, JPEG
   - Test images in `datasets/` folder

3. View Results:
   - **📊 Overview**: Quick diagnosis summary
   - **🔬 Detailed Analysis**: Full differential diagnosis with reasoning
   - **📋 Medical Report**: Professional documentation (downloadable)
   - **💬 Ask AI**: Interactive Q&A about the case

## Test with Sample Data

### Using Existing Datasets
```bash
# Test images are in:
datasets/archive/Testing/glioma/
datasets/archive/Testing/meningioma/
datasets/archive/Testing/pituitary/
datasets/archive/Testing/notumor/
```

### Quick Test Command
```bash
# From project root
# 1. Start backend in terminal 1:
cd backend && uvicorn main:app --reload

# 2. Start frontend in terminal 2:
cd frontend && npm run dev

# 3. Open http://localhost:5173 and upload any image from datasets/archive/Testing/
```

## What You'll See

### Multi-Agent Analysis Output
```json
{
  "prediction": "glioblastoma",
  "confidence": 0.94,
  "multi_agent_analysis": {
    "vision_analysis": {
      "tumor_size": "3.2cm",
      "location": "left temporal lobe",
      "enhancement_pattern": "ring-enhancing"
    },
    "differential_diagnosis": [
      {
        "diagnosis": "Glioblastoma (Grade IV)",
        "probability": 0.94,
        "reasoning": "Ring-enhancing mass with central necrosis..."
      }
    ],
    "patient_context": {
      "age_relevance": "Peak incidence age for glioblastoma",
      "risk_factors": ["Age 45", "Male gender"]
    },
    "quality_assurance": {
      "confidence": 0.94,
      "validation_passed": true
    },
    "medical_report": {
      "executive_summary": "...",
      "clinical_information": "...",
      "findings": "...",
      "impression": "...",
      "recommendations": "..."
    }
  },
  "analysis_metadata": {
    "total_execution_time": 2.3,
    "agents_executed": 5
  }
}
```

## Troubleshooting

### Backend Issues

**Port already in use:**
```bash
# Use different port
uvicorn main:app --reload --port 8001
```

**Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend Issues

**Build errors:**
```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Port 5173 in use:**
```bash
# Vite will auto-increment to 5174, 5175, etc.
# Or specify port:
npm run dev -- --port 3000
```

## API Testing

### Test Backend Directly
```bash
# Health check
curl http://localhost:8000/

# Agent health
curl http://localhost:8000/agent/health

# Upload test (replace with actual image)
curl -X POST http://localhost:8000/predict \
  -F "file=@datasets/archive/Testing/glioma/image1.jpg" \
  -F "patient_age=45" \
  -F "patient_sex=male" \
  -F "symptoms=headaches"
```

## Next Steps

1. **Explore Documentation**: See [MULTI_AGENT_SYSTEM.md](MULTI_AGENT_SYSTEM.md)
2. **Customize Agents**: Edit files in `backend/agents/`
3. **Enhance UI**: Modify components in `frontend/src/components/`
4. **Add Features**: Follow the Future Enhancements roadmap

## Need Help?

- Check [README.md](README.md) for project overview
- See [documentation/](documentation/) for detailed guides
- Review agent code in [backend/agents/](backend/agents/)
- Examine frontend components in [frontend/src/components/](frontend/src/components/)

---

**🎉 You're ready to analyze brain tumors with multi-agent AI!**
