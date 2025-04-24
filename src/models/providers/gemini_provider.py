#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro implementaci poskytovatele Google Gemini API.
"""

import os
import base64
from io import BytesIO
import json
import requests
from typing import Dict, List, Any, Optional
from PIL import Image
import re

from ..base.provider import TextModelProvider, VisionModelProvider, EmbeddingModelProvider


class GeminiTextModelProvider(TextModelProvider):
    """
    Třída pro implementaci poskytovatele textových modelů Google Gemini.
    """
    
    # Seznam dostupných textových modelů
    AVAILABLE_MODELS = [
        "gemini-1.0-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.5-pro-preview-03-25"
    ]
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Inicializace poskytovatele Google Gemini.
        
        Args:
            model_name: Název modelu pro použití
        """
        self.model_name = model_name
        self.api_key = None
        self.api_base = "https://generativelanguage.googleapis.com/v1"
    
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s API klíčem a dalšími parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu
            **kwargs: Další parametry specifické pro Gemini
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API klíč Google Gemini není k dispozici. Nastavte proměnnou prostředí GEMINI_API_KEY nebo předejte api_key parametr.")
        
        # Vyčištění API klíče od mezer a neviditelných znaků
        self.api_key = re.sub(r'\s+', '', self.api_key)
        self.api_key = ''.join(char for char in self.api_key if char.isprintable())
        
        # Zpracování dalších parametrů
        if "api_base" in kwargs:
            self.api_base = kwargs["api_base"]
        
        if "model_name" in kwargs:
            self.model_name = kwargs["model_name"]
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generuje text na základě promptu pomocí Google Gemini API.
        
        Args:
            prompt: Prompt pro generování textu
            **kwargs: Další parametry pro generování textu
            
        Returns:
            Vygenerovaný text
        """
        # Příprava dotazu pro API
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        url = f"{self.api_base}/models/{self.model_name}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Odeslání dotazu
        try:
            print(f"Odesílám požadavek na Google Gemini API (model: {self.model_name})")
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30  # Přidání timeoutu pro prevenci nekonečného čekání
            )
            
            # Zpracování odpovědi
            if response.status_code == 200:
                response_data = response.json()
                if 'candidates' in response_data and len(response_data['candidates']) > 0:
                    # Extrahujeme text z odpovědi
                    candidate = response_data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        return parts[0].get('text', '')
                    
                return ""
            else:
                error_msg = f"Chyba při dotazu na Google Gemini API: {response.status_code} - {response.text}"
                print(error_msg)
                raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            print(f"Síťová chyba při dotazu na Google Gemini API: {e}")
            raise Exception(f"Síťová chyba při dotazu na Google Gemini API: {e}")
    
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od Google Gemini.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        return self.AVAILABLE_MODELS


class GeminiVisionModelProvider(VisionModelProvider):
    """
    Třída pro implementaci poskytovatele vizuálních modelů Google Gemini.
    """
    
    # Seznam dostupných vizuálních modelů
    AVAILABLE_MODELS = [
        "gemini-1.0-pro-vision",
        "gemini-1.5-pro-vision",
        "gemini-1.5-flash-vision",
        "gemini-2.5-pro-preview-03-25"
    ]
    
    def __init__(self, model_name: str = "gemini-1.5-flash-vision"):
        """
        Inicializace poskytovatele Google Gemini Vision.
        
        Args:
            model_name: Název modelu pro použití
        """
        self.model_name = model_name
        self.api_key = None
        self.api_base = "https://generativelanguage.googleapis.com/v1"
    
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s API klíčem a dalšími parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu
            **kwargs: Další parametry specifické pro Gemini
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API klíč Google Gemini není k dispozici. Nastavte proměnnou prostředí GEMINI_API_KEY nebo předejte api_key parametr.")
        
        # Vyčištění API klíče od mezer a neviditelných znaků
        self.api_key = re.sub(r'\s+', '', self.api_key)
        self.api_key = ''.join(char for char in self.api_key if char.isprintable())
        
        # Zpracování dalších parametrů
        if "api_base" in kwargs:
            self.api_base = kwargs["api_base"]
        
        if "model_name" in kwargs:
            self.model_name = kwargs["model_name"]
    
    @staticmethod
    def encode_image(image: Image.Image) -> str:
        """
        Zakóduje obrázek do base64.
        
        Args:
            image: PIL.Image objekt
            
        Returns:
            Zakódovaný obrázek v base64
        """
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def generate_text_from_image(self, image: Image.Image, prompt: str, **kwargs) -> str:
        """
        Generuje text na základě obrázku a promptu pomocí Google Gemini API.
        
        Args:
            image: PIL.Image objekt
            prompt: Prompt pro generování textu
            **kwargs: Další parametry pro generování textu
            
        Returns:
            Vygenerovaný text
        """
        # Zakódování obrázku do base64
        image_base64 = self.encode_image(image)
        
        # Příprava dotazu pro API
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        url = f"{self.api_base}/models/{self.model_name}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Odeslání dotazu
        try:
            print(f"Odesílám požadavek na Google Gemini Vision API (model: {self.model_name})")
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60  # Delší timeout pro zpracování obrázků
            )
            
            # Zpracování odpovědi
            if response.status_code == 200:
                response_data = response.json()
                if 'candidates' in response_data and len(response_data['candidates']) > 0:
                    # Extrahujeme text z odpovědi
                    candidate = response_data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        return parts[0].get('text', '')
                    
                return ""
            else:
                error_msg = f"Chyba při dotazu na Google Gemini Vision API: {response.status_code} - {response.text}"
                print(error_msg)
                raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            print(f"Síťová chyba při dotazu na Google Gemini Vision API: {e}")
            raise Exception(f"Síťová chyba při dotazu na Google Gemini Vision API: {e}")
    
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od Google Gemini Vision.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        return self.AVAILABLE_MODELS


class GeminiEmbeddingModelProvider(EmbeddingModelProvider):
    """
    Třída pro implementaci poskytovatele embedding modelů Google Gemini.
    """
    
    # Seznam dostupných embedding modelů
    AVAILABLE_MODELS = [
        "embedding-001",
        "text-embedding-004"
    ]
    
    def __init__(self, model_name: str = "text-embedding-004"):
        """
        Inicializace poskytovatele Google Gemini Embeddings.
        
        Args:
            model_name: Název modelu pro použití
        """
        self.model_name = model_name
        self.api_key = None
        self.api_base = "https://generativelanguage.googleapis.com/v1"
    
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s API klíčem a dalšími parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu
            **kwargs: Další parametry specifické pro Gemini
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API klíč Google Gemini není k dispozici. Nastavte proměnnou prostředí GEMINI_API_KEY nebo předejte api_key parametr.")
        
        # Vyčištění API klíče od mezer a neviditelných znaků
        self.api_key = re.sub(r'\s+', '', self.api_key)
        self.api_key = ''.join(char for char in self.api_key if char.isprintable())
        
        # Zpracování dalších parametrů
        if "api_base" in kwargs:
            self.api_base = kwargs["api_base"]
        
        if "model_name" in kwargs:
            self.model_name = kwargs["model_name"]
    
    def get_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Získá embedding vektory pro zadané texty pomocí Google Gemini API.
        
        Args:
            texts: Seznam textů pro získání embeddingů
            **kwargs: Další parametry pro generování embeddingů
            
        Returns:
            Seznam embedding vektorů
        """
        results = []
        
        for text in texts:
            url = f"{self.api_base}/models/{self.model_name}:embedContent?key={self.api_key}"
            
            payload = {
                "model": f"models/{self.model_name}",
                "content": {
                    "parts": [
                        {
                            "text": text
                        }
                    ]
                }
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    # Extrahujeme vektor embeddingů
                    if 'embedding' in response_data and 'values' in response_data['embedding']:
                        embedding_values = response_data['embedding']['values']
                        results.append(embedding_values)
                    else:
                        results.append([])
                else:
                    error_msg = f"Chyba při získávání embeddingů z Google Gemini API: {response.status_code} - {response.text}"
                    print(error_msg)
                    raise Exception(error_msg)
            except requests.exceptions.RequestException as e:
                print(f"Síťová chyba při dotazu na Google Gemini API: {e}")
                raise Exception(f"Síťová chyba při dotazu na Google Gemini API: {e}")
        
        return results
    
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od Google Gemini Embeddings.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        return self.AVAILABLE_MODELS 