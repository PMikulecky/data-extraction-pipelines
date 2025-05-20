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
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import re

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("VAROVÁNÍ: Knihovna google-generativeai není k dispozici. Gemini provider nebude dostupný.")

from ..base.provider import TextModelProvider, VisionModelProvider, EmbeddingModelProvider, MultimodalModelProvider


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
    
    def generate_text(self, prompt: str, **kwargs) -> tuple[str, dict]:
        """
        Generuje text na základě promptu pomocí Google Gemini API.
        
        Args:
            prompt: Prompt pro generování textu
            **kwargs: Další parametry pro generování textu
            
        Returns:
            tuple: Vygenerovaný text a slovník s počty tokenů ({'input_tokens': N, 'output_tokens': M})
        """
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1000)
        url = f"{self.api_base}/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
        }
        headers = {"Content-Type": "application/json"}
        token_usage = {"input_tokens": 0, "output_tokens": 0} # Default
        
        try:
            print(f"Odesílám požadavek na Google Gemini API (model: {self.model_name})")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                response_data = response.json()
                text_content = ""
                # Extrakce textu
                if 'candidates' in response_data and len(response_data['candidates']) > 0:
                    candidate = response_data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        text_content = candidate['content']['parts'][0].get('text', '')
                # Extrakce tokenů
                if 'usageMetadata' in response_data:
                    usage_meta = response_data['usageMetadata']
                    token_usage['input_tokens'] = usage_meta.get('promptTokenCount', 0)
                    token_usage['output_tokens'] = usage_meta.get('candidatesTokenCount', 0)
                return text_content, token_usage
            else:
                error_msg = f"Chyba při dotazu na Google Gemini API: {response.status_code} - {response.text}"
                print(error_msg)
                return "", token_usage # Vrátit default při chybě
        except requests.exceptions.RequestException as e:
            print(f"Síťová chyba při dotazu na Google Gemini API: {e}")
            return "", token_usage # Vrátit default při chybě
    
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
    
    def generate_text_from_image(self, image: Image.Image, prompt: str, **kwargs) -> tuple[str, dict]:
        """
        Generuje text na základě obrázku a promptu pomocí Google Gemini API.
        
        Args:
            image: PIL.Image objekt
            prompt: Prompt pro generování textu
            **kwargs: Další parametry pro generování textu
            
        Returns:
            tuple: Vygenerovaný text a slovník s počty tokenů ({'input_tokens': N, 'output_tokens': M})
        """
        image_base64 = self.encode_image(image)
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1000)
        url = f"{self.api_base}/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                    ]
                }
            ],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
        }
        headers = {"Content-Type": "application/json"}
        token_usage = {"input_tokens": 0, "output_tokens": 0} # Default
        
        try:
            print(f"Odesílám požadavek na Google Gemini Vision API (model: {self.model_name})")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                response_data = response.json()
                text_content = ""
                # Extrakce textu
                if 'candidates' in response_data and len(response_data['candidates']) > 0:
                    candidate = response_data['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                         # Může být více částí, spojíme je
                         text_content = " ".join([p.get('text', '') for p in candidate['content']['parts']]).strip()
                # Extrakce tokenů
                if 'usageMetadata' in response_data:
                    usage_meta = response_data['usageMetadata']
                    token_usage['input_tokens'] = usage_meta.get('promptTokenCount', 0)
                    token_usage['output_tokens'] = usage_meta.get('candidatesTokenCount', 0)
                return text_content, token_usage
            else:
                error_msg = f"Chyba při dotazu na Google Gemini Vision API: {response.status_code} - {response.text}"
                print(error_msg)
                return "", token_usage # Vrátit default při chybě
        except requests.exceptions.RequestException as e:
            print(f"Síťová chyba při dotazu na Google Gemini Vision API: {e}")
            return "", token_usage # Vrátit default při chybě
    
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
        # Příprava dotazu pro API
        url = f"{self.api_base}/models/{self.model_name}:batchEmbedContents?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        # Gemini API očekává obsah ve formátu "contents"
        requests_payload = [
            {"model": f"models/{self.model_name}", "content": {"parts": [{"text": text}]}}
            for text in texts
        ]
        
        payload = {"requests": requests_payload}
        
        # Odeslání dotazu
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            # Zpracování odpovědi
            if response.status_code == 200:
                data = response.json()
                if "embeddings" in data:
                    return [item["values"] for item in data["embeddings"]]
                else:
                    print("Chyba: Odpověď Gemini Embedding API neobsahuje klíč 'embeddings'.")
                    # Vrátit prázdný seznam správné délky?
                    return [[] for _ in texts]
            else:
                error_msg = f"Chyba při získávání embeddingů z Google Gemini API: {response.status_code} - {response.text}"
                print(error_msg)
                raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            print(f"Síťová chyba při získávání embeddingů z Google Gemini API: {e}")
            raise Exception(f"Síťová chyba při získávání embeddingů z Google Gemini API: {e}")
    
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od Google Gemini Embeddings.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        return self.AVAILABLE_MODELS 


class GeminiMultimodalModelProvider(MultimodalModelProvider):
    """
    Poskytovatel multimodálních modelů Gemini.
    """
    
    # Seznam dostupných modelů
    AVAILABLE_MODELS = [
        "gemini-1.5-pro",
        "gemini-1.5-flash"
    ]
    
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele.
        
        Args:
            api_key: API klíč pro Gemini
            **kwargs: Další parametry pro inicializaci
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Knihovna google-generativeai není nainstalována. Nainstalujte ji pomocí příkazu: pip install google-generativeai")
        
        # Použití API klíče z parametru nebo prostředí
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API klíč pro Gemini není zadán. Zadejte ho jako parametr nebo nastavte proměnnou prostředí GEMINI_API_KEY.")
        
        # Inicializace klienta
        genai.configure(api_key=self.api_key)
        
        # Konfigurace modelu
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
        
        # Generativní konfigurace
        self.generation_config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 32,
            "max_output_tokens": 500,
        }
        
        try:
            # Vytvoření modelu
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=safety_settings
            )
        except Exception as e:
            print(f"Chyba při inicializaci Gemini modelu: {e}")
            self.model = None
    
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
        if not self.model:
            self.initialize()
        
        # Sestavení dotazu s obrázkem a textem
        full_prompt = f"{prompt}\n\nExtrahovaný text z dokumentu: {text}"
        
        # Zpracování vstupu
        try:
            # Vytvoření požadavku s obrázkem a textem
            response = self.model.generate_content([full_prompt, image])
            
            # Extrakce odpovědi
            if hasattr(response, 'text'):
                text_result = response.text
            else:
                text_result = response.candidates[0].content.parts[0].text
            
            # Aproximace tokenů (Gemini API nevrací počty tokenů)
            # Hrubý odhad: 1 token ≈ 4 znaky
            prompt_tokens = len(full_prompt) + 500  # 500 je odhad pro obrázek
            completion_tokens = len(text_result)
            
            token_usage = {
                "input_tokens": prompt_tokens // 4,
                "output_tokens": completion_tokens // 4
            }
            
            return text_result, token_usage
            
        except Exception as e:
            print(f"Chyba při generování textu z obrázku a textu pomocí Gemini: {e}")
            return "", {"input_tokens": 0, "output_tokens": 0} 