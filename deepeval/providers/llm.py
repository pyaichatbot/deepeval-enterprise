"""
LLM provider implementations for the DeepEval framework.

This module contains implementations for various Large Language Model providers
including OpenAI, Anthropic, and others.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass

# Optional integrations
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    api_key: str
    model: str
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    timeout: int = 30
    max_retries: int = 3


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM."""
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM."""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "provider": self.__class__.__name__,
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, config: LLMConfig):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        super().__init__(config)
        self.client = openai.AsyncOpenAI(api_key=config.api_key)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        try:
            # Merge config with kwargs
            request_params = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
                "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
            }
            
            # Remove None values
            request_params = {k: v for k, v in request_params.items() if v is not None}
            
            response = await self.client.chat.completions.create(**request_params)
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response using OpenAI API."""
        try:
            request_params = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "stream": True
            }
            
            # Remove None values
            request_params = {k: v for k, v in request_params.items() if v is not None}
            
            stream = await self.client.chat.completions.create(**request_params)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"OpenAI streaming error: {e}")
            raise

    def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        return (
            self.config.api_key is not None and
            self.config.model is not None and
            len(self.config.api_key) > 0
        )


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider."""

    def __init__(self, config: LLMConfig):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not available. Install with: pip install anthropic")
        
        super().__init__(config)
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic API."""
        try:
            request_params = {
                "model": self.config.model,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 1000),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
            }
            
            # Remove None values
            request_params = {k: v for k, v in request_params.items() if v is not None}
            
            response = await self.client.messages.create(**request_params)
            return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response using Anthropic API."""
        try:
            request_params = {
                "model": self.config.model,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 1000),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", self.config.temperature),
                "stream": True
            }
            
            # Remove None values
            request_params = {k: v for k, v in request_params.items() if v is not None}
            
            stream = await self.client.messages.stream(**request_params)
            
            async for chunk in stream:
                if chunk.type == "content_block_delta":
                    yield chunk.delta.text
                    
        except Exception as e:
            self.logger.error(f"Anthropic streaming error: {e}")
            raise

    def validate_config(self) -> bool:
        """Validate Anthropic configuration."""
        return (
            self.config.api_key is not None and
            self.config.model is not None and
            len(self.config.api_key) > 0
        )


class GoogleProvider(LLMProvider):
    """Google Gemini LLM provider."""

    def __init__(self, config: LLMConfig):
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google Generative AI library not available. Install with: pip install google-generativeai")
        
        super().__init__(config)
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Google Gemini API."""
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", self.config.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
            )
            
            response = await self.model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Google Gemini API error: {e}")
            raise

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response using Google Gemini API."""
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", self.config.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
            )
            
            response = await self.model.generate_content_async(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            self.logger.error(f"Google Gemini streaming error: {e}")
            raise

    def validate_config(self) -> bool:
        """Validate Google configuration."""
        return (
            self.config.api_key is not None and
            self.config.model is not None and
            len(self.config.api_key) > 0
        )


class MockProvider(LLMProvider):
    """Mock LLM provider for testing and development."""

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = LLMConfig(
                api_key="mock_key",
                model="mock-model",
                temperature=0.7
            )
        super().__init__(config)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response."""
        # Simple mock response based on prompt content
        if "relevance" in prompt.lower():
            return "0.8"
        elif "fluency" in prompt.lower():
            return "0.9"
        elif "bias" in prompt.lower():
            return "0.1"
        elif "toxicity" in prompt.lower():
            return "0.0"
        else:
            return "This is a mock response for testing purposes."

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate mock streaming response."""
        response = await self.generate(prompt, **kwargs)
        words = response.split()
        for word in words:
            yield word + " "
            # Simulate streaming delay
            import asyncio
            await asyncio.sleep(0.1)

    def validate_config(self) -> bool:
        """Mock provider always validates."""
        return True


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_provider(
        provider_type: str,
        api_key: str,
        model: str,
        **kwargs
    ) -> LLMProvider:
        """Create an LLM provider instance."""
        config = LLMConfig(
            api_key=api_key,
            model=model,
            **kwargs
        )
        
        if provider_type.lower() == "openai":
            return OpenAIProvider(config)
        elif provider_type.lower() == "anthropic":
            return AnthropicProvider(config)
        elif provider_type.lower() == "google":
            return GoogleProvider(config)
        elif provider_type.lower() == "mock":
            return MockProvider(config)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    @staticmethod
    def create_from_env(provider_type: str, model: str, **kwargs) -> LLMProvider:
        """Create provider using environment variables for API key."""
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY"
        }
        
        if provider_type.lower() not in env_var_map:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        api_key = os.getenv(env_var_map[provider_type.lower()])
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {env_var_map[provider_type.lower()]}")
        
        return LLMProviderFactory.create_provider(provider_type, api_key, model, **kwargs)

    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available providers based on installed libraries."""
        providers = ["mock"]  # Mock is always available
        
        if OPENAI_AVAILABLE:
            providers.append("openai")
        if ANTHROPIC_AVAILABLE:
            providers.append("anthropic")
        if GOOGLE_AVAILABLE:
            providers.append("google")
        
        return providers


class LLMProviderManager:
    """Manager for multiple LLM providers."""

    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.default_provider: Optional[str] = None

    def add_provider(self, name: str, provider: LLMProvider):
        """Add a provider to the manager."""
        self.providers[name] = provider
        if self.default_provider is None:
            self.default_provider = name

    def get_provider(self, name: Optional[str] = None) -> LLMProvider:
        """Get a provider by name, or default if not specified."""
        if name is None:
            name = self.default_provider
        
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not found")
        
        return self.providers[name]

    def list_providers(self) -> List[str]:
        """List all available provider names."""
        return list(self.providers.keys())

    def remove_provider(self, name: str):
        """Remove a provider from the manager."""
        if name in self.providers:
            del self.providers[name]
            if self.default_provider == name:
                self.default_provider = list(self.providers.keys())[0] if self.providers else None
