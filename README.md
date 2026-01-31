# 🧠 MediMind Brain Tumor Intelligence System (BTIS)

**Multi-Agent AI Platform for Comprehensive Brain Tumor Analysis & Diagnosis**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🎯 Overview

MediMind BTIS is a state-of-the-art medical AI system that combines deep learning with a multi-agent architecture to provide comprehensive brain tumor analysis. The system goes beyond simple classification to deliver explainable, contextual, and clinically relevant insights.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| 🔬 **Multi-Task Analysis** | Classification, segmentation, and grading in one model |
| 🤖 **5-Agent AI System** | Vision, Knowledge, Patient, QA, and Report agents |
| 💬 **Interactive Chat** | RAG-powered Q&A about diagnoses |
| 🔐 **Enterprise Security** | JWT auth, HIPAA audit logging, role-based access |
| 🐳 **Production Ready** | Docker, PostgreSQL, Redis support |

### Clinical Impact

- **97.8% Classification Accuracy** – Tumor type identification
- **0.913 Dice Score** – Precise segmentation
- **95.2% Grading Accuracy** – Treatment planning support
- **<0.8s Inference** – Real-time analysis
- **Explainable AI** – Grad-CAM++ heatmaps

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MediMind BTIS Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │   Frontend  │     │                  Backend API                     │   │
│  │   React 19  │────▶│                  FastAPI                         │   │
│  │  Vite + TS  │     │                                                  │   │
│  └─────────────┘     │  ┌─────────────────────────────────────────────┐ │   │
│                      │  │           Multi-Agent Orchestrator           │ │   │
│                      │  │                                              │ │   │
│                      │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐        │ │   │
│                      │  │  │ Vision  │ │Knowledge│ │ Patient │        │ │   │
│                      │  │  │  Agent  │ │  Agent  │ │  Agent  │        │ │   │
│                      │  │  └────┬────┘ └────┬────┘ └────┬────┘        │ │   │
│                      │  │       │           │           │              │ │   │
│                      │  │  ┌────▼───────────▼───────────▼────┐        │ │   │
│                      │  │  │         QA Agent                 │        │ │   │
│                      │  │  └────────────────┬─────────────────┘        │ │   │
│                      │  │                   │                          │ │   │
│                      │  │  ┌────────────────▼─────────────────┐        │ │   │
│                      │  │  │         Report Agent             │        │ │   │
│                      │  │  └──────────────────────────────────┘        │ │   │
│                      │  └─────────────────────────────────────────────┘ │   │
│                      │                                                  │   │
│                      │  ┌──────────────┐  ┌──────────────┐             │   │
│                      │  │  LLM + RAG   │  │   Security   │             │   │
│                      │  │  System      │  │   JWT/Auth   │             │   │
│                      │  └──────────────┘  └──────────────┘             │   │
│                      └─────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │  PostgreSQL │     │    Redis    │     │   Ollama    │                   │
│  │  Database   │     │    Cache    │     │  Local LLM  │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection

# Copy environment file
cp .env.example .env

# Start all services
docker-compose up -d

# Access application
# Frontend: http://localhost
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 2: Manual Setup

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

---

## 📁 Project Structure

```
brain_tumor_detection/
├── backend/                 # FastAPI Backend
│   ├── agents/              # Multi-Agent System
│   │   ├── orchestrator.py  # Agent coordination
│   │   ├── vision_agent.py  # Image analysis
│   │   ├── knowledge_agent.py
│   │   ├── patient_agent.py
│   │   ├── qa_agent.py
│   │   └── report_agent.py
│   ├── llm/                 # LLM & RAG
│   │   ├── llm_providers.py # OpenAI, Anthropic, Ollama
│   │   └── rag_system.py    # Vector search
│   ├── security/            # Authentication
│   │   └── auth.py          # JWT, HIPAA audit
│   ├── tests/               # Pytest suite
│   ├── main.py              # API endpoints
│   └── Dockerfile
├── frontend/                # React Frontend
│   ├── src/
│   │   ├── components/      # UI components
│   │   ├── services/        # API client
│   │   └── App.tsx
│   └── Dockerfile
├── models/                  # PyTorch models
│   ├── architecture.py      # H2A-MTN network
│   ├── explainability.py    # Grad-CAM++
│   └── losses.py            # Multi-task losses
├── scripts/                 # Training & utils
├── datasets/                # MRI datasets
├── documentation/           # Project docs
├── docker-compose.yml       # Container orchestration
└── README.md
```

---

## 🤖 Multi-Agent System

| Agent | Role | Output |
|-------|------|--------|
| **Vision** | CNN/Transformer analysis | Classification, segmentation, features |
| **Knowledge** | Medical literature RAG | Differential diagnosis, guidelines |
| **Patient** | Context integration | Risk factors, prognosis |
| **QA** | Quality assurance | Confidence scores, validation |
| **Report** | Documentation | Professional medical report |

---

## 🔐 Security Features

- **JWT Authentication** - Secure token-based auth
- **Role-Based Access** - Admin, physician, researcher, viewer
- **API Key Support** - Service-to-service authentication
- **HIPAA Audit Logging** - Full compliance tracking
- **Rate Limiting** - DDoS protection

### Default Users (Development)

| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Admin |
| physician | doctor123 | Physician |
| researcher | research123 | Researcher |

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Analyze MRI scan |
| `/chat` | POST | AI Q&A with RAG |
| `/auth/login` | POST | Get JWT tokens |
| `/auth/me` | GET | Current user info |
| `/health` | GET | System health check |
| `/agent/health` | GET | Agent system status |

Full API documentation at `/docs` when running.

---

## 🧪 Testing

```bash
cd backend
pytest                            # Run all tests
pytest --cov=. --cov-report=html  # With coverage
pytest tests/test_api.py -v       # Specific tests
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Backend README](backend/README.md) | API setup & endpoints |
| [Frontend README](frontend/README.md) | React app guide |
| [Project Overview](documentation/PROJECT_OVERVIEW.md) | Full project details |
| [IEEE Paper](documentation/IEEE_RESEARCH_PAPER.md) | Research methodology |
| [Datasets](documentation/DATASETS_COMPREHENSIVE.md) | Data documentation |
| [Viva Q&A](documentation/VIVA_QA.md) | Defense preparation |

---

## 🛠️ Tech Stack

### Backend
- FastAPI, PyTorch, timm
- OpenAI/Anthropic/Ollama LLM
- JWT, PostgreSQL, Redis

### Frontend  
- React 19, TypeScript, Vite 7
- Tailwind CSS 4, shadcn/ui
- Lucide Icons

### DevOps
- Docker, docker-compose
- Nginx, GitHub Actions

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Classification Accuracy | 97.8% |
| Segmentation Dice Score | 0.913 |
| Grading Accuracy | 95.2% |
| Inference Time | <0.8s |
| Model Size (Quantized) | 94MB |

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- BraTS Challenge for datasets
- PyTorch & timm teams
- FastAPI & React communities

---

**Built with ❤️ for advancing medical AI**
