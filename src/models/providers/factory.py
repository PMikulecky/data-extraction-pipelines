#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul implementující tovární třídu pro vytváření instancí poskytovatelů modelů.
"""

from typing import Dict, Optional, Type, Any, Union
import os

from ..base.provider import TextModelProvider, VisionModelProvider, EmbeddingModelProvider
from .openai_provider import OpenAITextModelProvider, OpenAIVisionModelProvider, OpenAIEmbeddingModelProvider
from .anthropic_provider import AnthropicTextModelProvider, AnthropicVisionModelProvider
from .ollama_provider import OllamaTextModelProvider, OllamaEmbeddingModelProvider, OllamaVisionModelProvider
from .gemini_provider import GeminiTextModelProvider, GeminiVisionModelProvider, GeminiEmbeddingModelProvider


class ModelProviderFactory:
    """
    Tovární třída pro vytváření instancí poskytovatelů modelů.
    """
    
    # Mapování názvů poskytovatelů na implementace TextModelProvider
    TEXT_PROVIDERS = {
        "openai": OpenAITextModelProvider,
        "anthropic": AnthropicTextModelProvider,
        "ollama": OllamaTextModelProvider,
        "gemini": GeminiTextModelProvider
    }
    
    # Mapování názvů poskytovatelů na implementace VisionModelProvider
    VISION_PROVIDERS = {
        "openai": OpenAIVisionModelProvider,
        "anthropic": AnthropicVisionModelProvider,
        "ollama": OllamaVisionModelProvider,
        "gemini": GeminiVisionModelProvider
    }
    
    # Mapování názvů poskytovatelů na implementace EmbeddingModelProvider
    EMBEDDING_PROVIDERS = {
        "openai": OpenAIEmbeddingModelProvider,
        "ollama": OllamaEmbeddingModelProvider,
        "gemini": GeminiEmbeddingModelProvider
    }
    
    @classmethod
    def create_text_provider(cls, provider_name: str, model_name: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> TextModelProvider:
        """
        Vytvoří instanci TextModelProvider.
        
        Args:
            provider_name: Název poskytovatele (např. "openai", "anthropic", "ollama", "gemini")
            model_name: Název modelu
            api_key: API klíč
            **kwargs: Další parametry pro inicializaci poskytovatele
            
        Returns:
            Instance TextModelProvider
            
        Raises:
            ValueError: Pokud poskytovatel není podporován
        """
        if provider_name not in cls.TEXT_PROVIDERS:
            raise ValueError(f"Poskytovatel '{provider_name}' není podporován pro textové modely. "
                           f"Podporovaní poskytovatelé: {list(cls.TEXT_PROVIDERS.keys())}")
        
        provider_class = cls.TEXT_PROVIDERS[provider_name]
        
        # Vytvoření instance poskytovatele
        if model_name:
            provider = provider_class(model_name=model_name)
        else:
            provider = provider_class()
        
        # Inicializace poskytovatele
        provider.initialize(api_key=api_key, **kwargs)
        
        return provider
    
    @classmethod
    def create_vision_provider(cls, provider_name: str, model_name: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> VisionModelProvider:
        """
        Vytvoří instanci VisionModelProvider.
        
        Args:
            provider_name: Název poskytovatele (např. "openai", "anthropic", "ollama", "gemini")
            model_name: Název modelu
            api_key: API klíč
            **kwargs: Další parametry pro inicializaci poskytovatele
            
        Returns:
            Instance VisionModelProvider
            
        Raises:
            ValueError: Pokud poskytovatel není podporován
        """
        if provider_name not in cls.VISION_PROVIDERS:
            raise ValueError(f"Poskytovatel '{provider_name}' není podporován pro vizuální modely. "
                           f"Podporovaní poskytovatelé: {list(cls.VISION_PROVIDERS.keys())}")
        
        provider_class = cls.VISION_PROVIDERS[provider_name]
        
        # Vytvoření instance poskytovatele
        if model_name:
            provider = provider_class(model_name=model_name)
        else:
            provider = provider_class()
        
        # Inicializace poskytovatele
        provider.initialize(api_key=api_key, **kwargs)
        
        return provider
    
    @classmethod
    def create_embedding_provider(cls, provider_name: str, model_name: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> EmbeddingModelProvider:
        """
        Vytvoří instanci EmbeddingModelProvider.
        
        Args:
            provider_name: Název poskytovatele (např. "openai", "ollama", "gemini")
            model_name: Název modelu
            api_key: API klíč
            **kwargs: Další parametry pro inicializaci poskytovatele
            
        Returns:
            Instance EmbeddingModelProvider
            
        Raises:
            ValueError: Pokud poskytovatel není podporován
        """
        if provider_name not in cls.EMBEDDING_PROVIDERS:
            raise ValueError(f"Poskytovatel '{provider_name}' není podporován pro embedding modely. "
                           f"Podporovaní poskytovatelé: {list(cls.EMBEDDING_PROVIDERS.keys())}")
        
        provider_class = cls.EMBEDDING_PROVIDERS[provider_name]
        
        # Vytvoření instance poskytovatele
        if model_name:
            provider = provider_class(model_name=model_name)
        else:
            provider = provider_class()
        
        # Inicializace poskytovatele
        provider.initialize(api_key=api_key, **kwargs)
        
        return provider 