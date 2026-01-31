"""
LLM Package
Provides LLM providers and RAG system for medical AI assistance
"""

from .llm_providers import (
    BaseLLM,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    LLMManager,
    llm_manager,
    get_llm_response
)

from .rag_system import (
    RAGSystem,
    MedicalKnowledgeBase,
    rag_system,
    get_rag_response
)

__all__ = [
    # LLM Providers
    "BaseLLM",
    "OpenAIProvider", 
    "AnthropicProvider",
    "OllamaProvider",
    "LLMManager",
    "llm_manager",
    "get_llm_response",
    
    # RAG System
    "RAGSystem",
    "MedicalKnowledgeBase",
    "rag_system",
    "get_rag_response"
]
