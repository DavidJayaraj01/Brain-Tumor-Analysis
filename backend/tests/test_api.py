"""
Pytest Test Suite for MediMind Brain Tumor Intelligence System
Comprehensive tests for API, agents, models, and security
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import io
import json

# FastAPI testing
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# Import app and modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Fixtures
@pytest.fixture
def test_image():
    """Create a test MRI image"""
    img = Image.new('RGB', (224, 224), color='gray')
    # Add some variation to simulate MRI
    pixels = img.load()
    for i in range(224):
        for j in range(224):
            # Create a circular pattern to simulate tumor
            dist = np.sqrt((i - 112)**2 + (j - 112)**2)
            if dist < 40:
                pixels[i, j] = (200, 200, 200)
            else:
                pixels[i, j] = (100, 100, 100)
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


@pytest.fixture
def test_image_bytes(test_image):
    """Get test image as bytes"""
    return test_image.read()


@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing"""
    return {
        "age": 52,
        "sex": "male",
        "symptoms": ["headaches", "seizures", "vision changes"]
    }


@pytest.fixture
def sample_prediction_response():
    """Sample prediction API response"""
    return {
        "classification": {
            "No Tumor": 0.05,
            "Glioma": 0.75,
            "Meningioma": 0.15,
            "Pituitary Adenoma": 0.05
        },
        "predicted_class": "Glioma",
        "confidence": 0.75,
        "grading": {"Low-Grade": 0.3, "High-Grade": 0.7},
        "predicted_grade": "High-Grade",
        "trustworthiness_score": 0.82
    }


class TestAPIEndpoints:
    """Test suite for FastAPI endpoints"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client"""
        from backend.main import app
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint returns status"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "version" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_classes_endpoint(self):
        """Test get classes endpoint"""
        response = self.client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert "classification_classes" in data
        assert "grading_classes" in data
    
    def test_predict_endpoint_requires_image(self):
        """Test predict endpoint requires image file"""
        response = self.client.post("/predict")
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_with_image(self, test_image):
        """Test predict endpoint with valid image"""
        test_image.seek(0)
        response = self.client.post(
            "/predict",
            files={"file": ("test.png", test_image, "image/png")},
            data={"patient_age": 52, "patient_sex": "male"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "classification" in data
    
    def test_predict_endpoint_rejects_non_image(self):
        """Test predict endpoint rejects non-image files"""
        response = self.client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400


class TestAuthentication:
    """Test suite for authentication system"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client"""
        from backend.main import app
        self.client = TestClient(app)
    
    def test_login_success(self):
        """Test successful login"""
        response = self.client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_failure(self):
        """Test login with wrong credentials"""
        response = self.client.post(
            "/auth/login",
            json={"username": "admin", "password": "wrongpassword"}
        )
        assert response.status_code == 401
    
    def test_protected_endpoint_without_token(self):
        """Test protected endpoint requires authentication"""
        response = self.client.get("/auth/me")
        assert response.status_code == 401
    
    def test_protected_endpoint_with_token(self):
        """Test protected endpoint with valid token"""
        # Login first
        login_response = self.client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123"}
        )
        token = login_response.json()["access_token"]
        
        # Access protected endpoint
        response = self.client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"


class TestRAGSystem:
    """Test suite for RAG (Retrieval-Augmented Generation) system"""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup RAG system"""
        from backend.llm.rag_system import rag_system
        await rag_system.initialize()
        self.rag = rag_system
    
    @pytest.mark.asyncio
    async def test_knowledge_base_initialization(self):
        """Test knowledge base initializes with documents"""
        from backend.llm.rag_system import MedicalKnowledgeBase
        kb = MedicalKnowledgeBase()
        await kb.initialize()
        assert kb._is_initialized
        assert len(kb.vector_store.documents) > 0
    
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """Test search returns relevant documents"""
        from backend.llm.rag_system import rag_system
        await rag_system.initialize()
        
        results = await rag_system.knowledge_base.search("glioblastoma treatment", k=3)
        assert len(results) > 0
        assert all("content" in r for r in results)
        assert all("source" in r for r in results)
    
    @pytest.mark.asyncio
    async def test_rag_query(self):
        """Test RAG query returns structured response"""
        from backend.llm.rag_system import get_rag_response
        
        response = await get_rag_response(
            "What is the treatment for glioblastoma?",
            {"diagnosis": "Glioblastoma", "tumor_type": "High-grade glioma"}
        )
        
        assert "answer" in response
        assert "confidence" in response
        assert "sources" in response
        assert len(response["answer"]) > 50


class TestVisionAgent:
    """Test suite for Vision Agent"""
    
    @pytest.mark.asyncio
    async def test_vision_agent_analyze(self, test_image_bytes):
        """Test vision agent can analyze images"""
        from backend.agents.vision_agent import VisionAgent
        
        agent = VisionAgent()
        
        # Create numpy array from image
        img = Image.open(io.BytesIO(test_image_bytes))
        img_array = np.array(img.resize((224, 224)))
        
        # Preprocess
        img_normalized = img_array.astype(np.float32) / 255.0
        if len(img_normalized.shape) == 3:
            img_normalized = np.transpose(img_normalized, (2, 0, 1))
        img_tensor = np.expand_dims(img_normalized, axis=0)
        
        result = await agent.analyze({"image": img_tensor})
        
        assert "predictions" in result
        assert "tumor_features" in result
        assert "confidence_score" in result


class TestKnowledgeAgent:
    """Test suite for Knowledge Agent"""
    
    @pytest.mark.asyncio
    async def test_knowledge_agent_analyze(self):
        """Test knowledge agent provides medical context"""
        from backend.agents.knowledge_agent import KnowledgeAgent
        
        agent = KnowledgeAgent()
        
        vision_result = {
            "predictions": {"classification": {"Glioma": 0.85}},
            "tumor_features": {"classification": "Glioma"}
        }
        
        result = await agent.analyze({
            "vision_analysis": vision_result,
            "patient_data": {"age": 52, "sex": "male"}
        })
        
        assert "differential_diagnosis" in result
        assert "literature_references" in result
        assert "treatment_guidelines" in result


class TestOrchestrator:
    """Test suite for Agent Orchestrator"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_full_analysis(self, test_image_bytes):
        """Test orchestrator coordinates all agents"""
        from backend.agents.orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator()
        
        # Create test input
        img = Image.open(io.BytesIO(test_image_bytes))
        img_array = np.array(img.resize((224, 224)))
        img_normalized = img_array.astype(np.float32) / 255.0
        if len(img_normalized.shape) == 3:
            img_normalized = np.transpose(img_normalized, (2, 0, 1))
        img_tensor = np.expand_dims(img_normalized, axis=0)
        
        case_data = {
            "image": img_tensor,
            "patient_age": 52,
            "patient_data": {"age": 52, "sex": "male", "symptoms": ["headaches"]}
        }
        
        result = await orchestrator.analyze_case(case_data)
        
        assert "summary" in result
        assert "analysis_results" in result
        assert "metadata" in result
        assert "vision_analysis" in result["analysis_results"]


class TestLLMProviders:
    """Test suite for LLM providers"""
    
    @pytest.mark.asyncio
    async def test_mock_response_generation(self):
        """Test mock response when API keys not available"""
        from backend.llm.llm_providers import OpenAIProvider
        
        provider = OpenAIProvider(api_key=None)  # No API key
        response = await provider.generate("What is glioblastoma diagnosis?")
        
        assert len(response) > 50
        assert "diagnosis" in response.lower() or "glioma" in response.lower()
    
    @pytest.mark.asyncio
    async def test_llm_manager_fallback(self):
        """Test LLM manager fallback mechanism"""
        from backend.llm.llm_providers import LLMManager
        
        manager = LLMManager()
        response = await manager.generate_with_fallback(
            "Explain brain tumor treatment options"
        )
        
        assert len(response) > 20


class TestSecurityModule:
    """Test suite for security module"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        from backend.security.auth import hash_password, verify_password
        
        password = "test_password_123"
        hashed = hash_password(password)
        
        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrong_password", hashed)
    
    def test_token_creation(self):
        """Test JWT token creation"""
        from backend.security.auth import create_access_token, decode_token, User
        
        user = User(
            id="test_user",
            username="testuser",
            email="test@example.com",
            role="physician",
            permissions=["read", "write"]
        )
        
        token = create_access_token(user)
        payload = decode_token(token)
        
        assert payload["sub"] == "test_user"
        assert payload["username"] == "testuser"
        assert payload["type"] == "access"
    
    def test_audit_logging(self):
        """Test HIPAA audit logging"""
        from backend.security.auth import audit_logger, User
        
        user = User(
            id="test_user",
            username="testuser",
            email="test@example.com",
            role="physician",
            permissions=["read"]
        )
        
        entry = audit_logger.log_access(
            user=user,
            action="VIEW_SCAN",
            resource="/predict",
            patient_id="patient_001"
        )
        
        assert entry["user_id"] == "test_user"
        assert entry["action"] == "VIEW_SCAN"
        assert entry["success"] is True


class TestDataPipeline:
    """Test suite for data pipeline components"""
    
    def test_image_preprocessing(self):
        """Test image preprocessing function"""
        from backend.main import preprocess_image
        
        img = Image.new('RGB', (512, 512), color='gray')
        result = preprocess_image(img)
        
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32
    
    def test_heatmap_generation(self):
        """Test heatmap generation"""
        from backend.main import generate_heatmap
        
        heatmap = generate_heatmap(224, 224)
        
        assert heatmap.shape == (224, 224)
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1
    
    def test_segmentation_mask_generation(self):
        """Test segmentation mask generation"""
        from backend.main import generate_segmentation_mask
        
        mask = generate_segmentation_mask(224, 224)
        
        assert mask.shape == (224, 224)
        assert set(np.unique(mask)).issubset({0, 255})
    
    def test_colormap_application(self):
        """Test colormap application to heatmap"""
        from backend.main import apply_colormap, generate_heatmap
        
        heatmap = generate_heatmap()
        colored = apply_colormap(heatmap)
        
        assert colored.shape == (224, 224, 3)
        assert colored.dtype == np.uint8


# Async test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
