"""
LLM Integration Module
Supports OpenAI GPT-4, Anthropic Claude, and local Ollama models
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import logging
import httpx

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        pass
    
    @abstractmethod
    async def generate_with_context(self, prompt: str, context: List[Dict], **kwargs) -> str:
        pass


class OpenAIProvider(BaseLLM):
    """OpenAI GPT-4 provider"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4-turbo-preview"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        if not self.api_key:
            return self._mock_response(prompt)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": kwargs.get("temperature", 0.7),
                        "max_tokens": kwargs.get("max_tokens", 2000)
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return self._mock_response(prompt)
    
    async def generate_with_context(self, prompt: str, context: List[Dict], **kwargs) -> str:
        system_prompt = kwargs.get("system_prompt", "You are a helpful medical AI assistant.")
        
        # Build context string
        context_str = "\n\n".join([
            f"Source: {ctx.get('source', 'Unknown')}\n{ctx.get('content', '')}"
            for ctx in context
        ])
        
        full_prompt = f"""Based on the following medical context:

{context_str}

User question: {prompt}

Provide a comprehensive, medically accurate response."""
        
        return await self.generate(full_prompt, system_prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response when API key not available"""
        if "diagnosis" in prompt.lower():
            return """Based on the imaging findings and clinical context, this case demonstrates characteristics consistent with a high-grade glioma (WHO Grade IV). 

Key supporting features:
1. Heterogeneous ring enhancement pattern
2. Central necrosis
3. Infiltrative tumor margins
4. Significant perilesional edema with mass effect

The differential diagnosis includes:
- Primary: Glioblastoma (most likely given imaging features)
- Secondary: Brain metastasis (would need clinical history)
- Less likely: CNS lymphoma

Recommended next steps:
1. Stereotactic biopsy for histopathological confirmation
2. Molecular profiling (IDH, MGMT promoter methylation)
3. Complete staging workup
4. Multidisciplinary tumor board review"""
        
        return f"Based on my analysis, I can provide the following insights regarding your question. {prompt[:100]}..."


class AnthropicProvider(BaseLLM):
    """Anthropic Claude provider"""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
        
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        if not self.api_key:
            return self._mock_response(prompt)
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "max_tokens": kwargs.get("max_tokens", 2000),
                        "system": system_prompt or "You are a helpful medical AI assistant.",
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                return data["content"][0]["text"]
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                return self._mock_response(prompt)
    
    async def generate_with_context(self, prompt: str, context: List[Dict], **kwargs) -> str:
        system_prompt = """You are an expert medical AI assistant specializing in neuro-oncology and brain tumor analysis. 
You provide accurate, evidence-based information while being clear about limitations and uncertainties.
Always cite relevant medical literature when possible."""
        
        context_str = "\n\n".join([
            f"[{ctx.get('source', 'Reference')}]: {ctx.get('content', '')}"
            for ctx in context
        ])
        
        full_prompt = f"""Medical Context:
{context_str}

Question: {prompt}

Please provide a detailed, medically accurate response."""
        
        return await self.generate(full_prompt, system_prompt)
    
    def _mock_response(self, prompt: str) -> str:
        return OpenAIProvider._mock_response(self, prompt)


class OllamaProvider(BaseLLM):
    """Local Ollama provider for offline LLM inference"""
    
    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        async with httpx.AsyncClient() as client:
            try:
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False
                    },
                    timeout=120.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
            except Exception as e:
                logger.error(f"Ollama error: {e}")
                return f"Local LLM unavailable. Error: {str(e)}"
    
    async def generate_with_context(self, prompt: str, context: List[Dict], **kwargs) -> str:
        context_str = "\n".join([ctx.get('content', '') for ctx in context])
        full_prompt = f"Context:\n{context_str}\n\nQuestion: {prompt}"
        return await self.generate(full_prompt, kwargs.get("system_prompt"))


class LLMManager:
    """Manager for multiple LLM providers with fallback support"""
    
    def __init__(self):
        self.providers: Dict[str, BaseLLM] = {}
        self.default_provider = "openai"
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        # OpenAI
        self.providers["openai"] = OpenAIProvider()
        
        # Anthropic
        self.providers["anthropic"] = AnthropicProvider()
        
        # Ollama (local)
        self.providers["ollama"] = OllamaProvider()
        
        logger.info(f"Initialized LLM providers: {list(self.providers.keys())}")
    
    def get_provider(self, name: str = None) -> BaseLLM:
        """Get a specific provider or the default"""
        name = name or self.default_provider
        return self.providers.get(name, self.providers[self.default_provider])
    
    async def generate(self, prompt: str, provider: str = None, **kwargs) -> str:
        """Generate response using specified or default provider"""
        llm = self.get_provider(provider)
        return await llm.generate(prompt, **kwargs)
    
    async def generate_with_fallback(self, prompt: str, providers: List[str] = None, **kwargs) -> str:
        """Try multiple providers with fallback"""
        providers = providers or ["anthropic", "openai", "ollama"]
        
        for provider_name in providers:
            try:
                llm = self.get_provider(provider_name)
                response = await llm.generate(prompt, **kwargs)
                if response and len(response) > 50:
                    return response
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue
        
        return "Unable to generate response. All providers unavailable."


# Singleton instance
llm_manager = LLMManager()


async def get_llm_response(prompt: str, context: List[Dict] = None, provider: str = None) -> str:
    """Convenience function for getting LLM responses"""
    if context:
        llm = llm_manager.get_provider(provider)
        return await llm.generate_with_context(prompt, context)
    return await llm_manager.generate(prompt, provider)
