#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul obsahující základní abstraktní třídy pro poskytovatele AI modelů.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from PIL import Image


class TextModelProvider(ABC):
    """
    Abstraktní třída pro poskytovatele textových modelů.
    """
    
    @abstractmethod
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s API klíčem a dalšími parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu
            **kwargs: Další parametry specifické pro konkrétního poskytovatele
        """
        pass
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generuje text na základě promptu.
        
        Args:
            prompt: Prompt pro generování textu
            **kwargs: Další parametry pro generování textu
            
        Returns:
            Vygenerovaný text
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od tohoto poskytovatele.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        pass


class VisionModelProvider(ABC):
    """
    Abstraktní třída pro poskytovatele vizuálních modelů.
    """
    
    @abstractmethod
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s API klíčem a dalšími parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu
            **kwargs: Další parametry specifické pro konkrétního poskytovatele
        """
        pass
    
    @abstractmethod
    def generate_text_from_image(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """
        Generuje text na základě obrázku a promptu.
        
        Args:
            image: Obrázek (PIL.Image)
            prompt: Prompt pro generování textu
            **kwargs: Další parametry pro generování textu
            
        Returns:
            Vygenerovaný text
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od tohoto poskytovatele.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        pass


class EmbeddingModelProvider(ABC):
    """
    Abstraktní třída pro poskytovatele embedding modelů.
    """
    
    @abstractmethod
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s API klíčem a dalšími parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu
            **kwargs: Další parametry specifické pro konkrétního poskytovatele
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Získá embedding vektory pro zadané texty.
        
        Args:
            texts: Seznam textů pro získání embeddingů
            **kwargs: Další parametry pro generování embeddingů
            
        Returns:
            Seznam embedding vektorů
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od tohoto poskytovatele.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        pass


class MultimodalModelProvider(ABC):
    """
    Základní abstraktní třída pro poskytovatele multimodálních modelů.
    Implementuje rozhraní pro komunikaci s multimodálními modely.
    """
    
    # Seznam dostupných modelů
    AVAILABLE_MODELS = []
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Inicializace poskytovatele multimodálního modelu.
        
        Args:
            model_name: Název modelu
        """
        self.api_key = None
        self.model_name = model_name or self.get_default_model()
        self.client = None
    
    @classmethod
    def get_default_model(cls) -> str:
        """
        Získá výchozí model.
        
        Returns:
            Název výchozího modelu
        """
        return cls.AVAILABLE_MODELS[0] if cls.AVAILABLE_MODELS else "default"
    
    @abstractmethod
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele.
        
        Args:
            api_key: API klíč
            **kwargs: Další parametry pro inicializaci
        """
        pass
    
    @abstractmethod
    def generate_text_from_image_and_text(self, image: Image.Image, text: str, prompt: str) -> tuple[str, Dict[str, int]]:
        """
        Generuje text na základě obrázku, textu a promptu.
        
        Args:
            image: Obrázek
            text: Doplňující text
            prompt: Dotaz/prompt pro model
            
        Returns:
            tuple: Vygenerovaný text a slovník s použitými tokeny
        """
        pass 