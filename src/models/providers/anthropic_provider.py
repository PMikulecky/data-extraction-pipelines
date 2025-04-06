#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro implementaci poskytovatele Anthropic API.
"""

import os
import base64
from io import BytesIO
import json
import requests
from typing import Dict, List, Any, Optional
from PIL import Image
import re

from ..base.provider import TextModelProvider, VisionModelProvider


class AnthropicTextModelProvider(TextModelProvider):
    """
    Třída pro implementaci poskytovatele textových modelů Anthropic.
    """
    
    # Seznam dostupných textových modelů
    AVAILABLE_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1"
    ]
    
    def __init__(self, model_name: str = "claude-3-haiku-20240307"):
        """
        Inicializace poskytovatele Anthropic.
        
        Args:
            model_name: Název modelu pro použití
        """
        self.model_name = model_name
        self.api_key = None
        self.api_base = "https://api.anthropic.com/v1"
    
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s API klíčem a dalšími parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu
            **kwargs: Další parametry specifické pro Anthropic
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("API klíč Anthropic není k dispozici. Nastavte proměnnou prostředí ANTHROPIC_API_KEY nebo předejte api_key parametr.")
        
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
        Generuje text na základě promptu pomocí Anthropic API.
        
        Args:
            prompt: Prompt pro generování textu
            **kwargs: Další parametry pro generování textu
            
        Returns:
            Vygenerovaný text
        """
        # Příprava dotazu pro API
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Odeslání dotazu
        try:
            print(f"Odesílám požadavek na Anthropic API (model: {self.model_name})")
            print(f"API klíč začíná: {self.api_key[:10]}..., délka: {len(self.api_key)}")
            
            response = requests.post(
                f"{self.api_base}/messages",
                headers=headers,
                json=payload,
                timeout=30  # Přidání timeoutu pro prevenci nekonečného čekání
            )
            
            # Zpracování odpovědi
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            else:
                try:
                    error_data = response.json()
                    error_type = error_data.get('error', {}).get('type', 'unknown')
                    error_message = error_data.get('error', {}).get('message', 'No message')
                    error_msg = f"Chyba při dotazu na Anthropic API: {response.status_code} - {response.text}"
                    print(f"Detaily chyby: typ={error_type}, zpráva={error_message}")
                    print(f"Použitá URL: {self.api_base}/messages")
                    print(f"Použité hlavičky: {headers}")
                    raise Exception(error_msg)
                except Exception as e:
                    error_msg = f"Chyba při dotazu na Anthropic API: {response.status_code} - {response.text}"
                    print(error_msg)
                    raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            print(f"Síťová chyba při dotazu na Anthropic API: {e}")
            raise Exception(f"Síťová chyba při dotazu na Anthropic API: {e}")
    
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od Anthropic.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        return self.AVAILABLE_MODELS


class AnthropicVisionModelProvider(VisionModelProvider):
    """
    Třída pro implementaci poskytovatele vizuálních modelů Anthropic.
    """
    
    # Seznam dostupných vizuálních modelů
    AVAILABLE_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]
    
    def __init__(self, model_name: str = "claude-3-haiku-20240307"):
        """
        Inicializace poskytovatele Anthropic Vision.
        
        Args:
            model_name: Název modelu pro použití
        """
        self.model_name = model_name
        self.api_key = None
        self.api_base = "https://api.anthropic.com/v1"
    
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s API klíčem a dalšími parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu
            **kwargs: Další parametry specifické pro Anthropic
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("API klíč Anthropic není k dispozici. Nastavte proměnnou prostředí ANTHROPIC_API_KEY nebo předejte api_key parametr.")
        
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
        Generuje text na základě obrázku a promptu pomocí Anthropic API.
        
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
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        # Vytvoření zprávy s obrázkem
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Odeslání dotazu
        try:
            print(f"Odesílám požadavek na Anthropic API s obrázkem (model: {self.model_name})")
            print(f"API klíč začíná: {self.api_key[:10]}..., délka: {len(self.api_key)}")
            
            response = requests.post(
                f"{self.api_base}/messages",
                headers=headers,
                json=payload,
                timeout=60  # Delší timeout pro obrázky
            )
            
            # Zpracování odpovědi
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            else:
                try:
                    error_data = response.json()
                    error_type = error_data.get('error', {}).get('type', 'unknown')
                    error_message = error_data.get('error', {}).get('message', 'No message')
                    error_msg = f"Chyba při dotazu na Anthropic API s obrázkem: {response.status_code} - {response.text}"
                    print(f"Detaily chyby: typ={error_type}, zpráva={error_message}")
                    print(f"Použitá URL: {self.api_base}/messages")
                    print(f"Použité hlavičky: {headers}")
                    raise Exception(error_msg)
                except Exception as e:
                    error_msg = f"Chyba při dotazu na Anthropic API s obrázkem: {response.status_code} - {response.text}"
                    print(error_msg)
                    raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            print(f"Síťová chyba při dotazu na Anthropic API s obrázkem: {e}")
            raise Exception(f"Síťová chyba při dotazu na Anthropic API s obrázkem: {e}")
    
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od Anthropic.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        return self.AVAILABLE_MODELS 