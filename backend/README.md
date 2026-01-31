# MediMind BTIS - Backend API

FastAPI-powered multi-agent AI system for brain tumor analysis.

## 🚀 Tech Stack

- **FastAPI** - High-performance async API framework
- **PyTorch** - Deep learning inference
- **Multi-Agent System** - 5 specialized AI agents
- **LLM Integration** - OpenAI, Anthropic, Ollama support
- **RAG System** - Medical knowledge retrieval
- **JWT Auth** - Secure authentication

## 📁 Project Structure

```
backend/
├── main.py               # FastAPI application & endpoints
├── model_loader.py       # PyTorch model loading
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration
├── agents/               # Multi-Agent System
│   ├── orchestrator.py   # Agent coordination
│   ├── vision_agent.py   # Image analysis
│   ├── knowledge_agent.py# Medical knowledge
│   ├── patient_agent.py  # Patient context
│   ├── qa_agent.py       # Quality assurance
│   └── report_agent.py   # Report generation
├── llm/                  # LLM & RAG System
│   ├── llm_providers.py  # OpenAI, Anthropic, Ollama
│   └── rag_system.py     # Vector search & retrieval
├── security/             # Authentication
│   └── auth.py           # JWT, API keys, HIPAA audit
├── tests/                # Test suite
│   ├── conftest.py       # Pytest fixtures
│   └── test_api.py       # API tests
├── config/               # Configuration
└── utils/                # Utilities
```

## 🛠️ Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
# OR
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🔧 Configuration

### Environment Variables

```env
# Security
JWT_SECRET_KEY=your-secret-key-change-in-production

# LLM API Keys (optional)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost:5432/medimind
REDIS_URL=redis://localhost:6379/0
```

## 📡 API Endpoints

### Public Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| GET | `/health` | Health check |
| GET | `/classes` | Tumor classifications |

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/login` | Get JWT tokens |
| POST | `/auth/refresh` | Refresh access token |
| POST | `/auth/logout` | Invalidate session |
| GET | `/auth/me` | Current user info |

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Analyze MRI scan |
| POST | `/chat` | AI Q&A about diagnosis |
| GET | `/agent/health` | Agent system status |
| GET | `/agent/metrics` | Performance metrics |

## 🤖 Multi-Agent System

### Agent Pipeline

```
MRI Upload → Vision Agent → Knowledge Agent → Patient Agent → QA Agent → Report Agent
                ↓               ↓                ↓              ↓            ↓
           Classification   Literature      Risk Factors   Validation   Medical
           Segmentation     Guidelines      Prognosis      Quality      Report
           Features         Differential    Context        Score
```

### Agents

1. **Vision Agent** - CNN/Transformer image analysis
2. **Knowledge Agent** - Medical literature & guidelines
3. **Patient Agent** - Patient context integration
4. **QA Agent** - Quality assurance & validation
5. **Report Agent** - Professional report generation

## 🔐 Security Features

- **JWT Authentication** - Access & refresh tokens
- **API Key Support** - Service-to-service auth
- **Role-Based Access** - Admin, physician, researcher, viewer
- **Rate Limiting** - Request throttling
- **HIPAA Audit Logging** - Compliance tracking

### Default Users

| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Admin |
| physician | doctor123 | Physician |
| researcher | research123 | Researcher |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/test_api.py::TestAPIEndpoints -v
```

## 🐳 Docker

```bash
# Build image
docker build -t medimind-backend .

# Run container
docker run -p 8000:8000 \
  -e JWT_SECRET_KEY=your-secret \
  medimind-backend
```

## 📊 API Documentation

When running, access interactive docs at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔗 Related

- [Frontend App](../frontend/README.md)
- [Main Documentation](../README.md)
