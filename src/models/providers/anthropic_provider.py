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
import time

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
    
    # Limity tokenů pro čekání (mírně pod skutečnými limity)
    INPUT_TOKEN_THRESHOLD = 40000
    OUTPUT_TOKEN_THRESHOLD = 8000
    
    def __init__(self, model_name: str = "claude-3-haiku-20240307"):
        """
        Inicializace poskytovatele Anthropic.
        
        Args:
            model_name: Název modelu pro použití
        """
        self.model_name = model_name
        self.api_key = None
        self.api_base = "https://api.anthropic.com/v1"
        # Kumulativní počty tokenů pro sledování limitu
        self.cumulative_input_tokens = 0
        self.cumulative_output_tokens = 0
        self.last_reset_time = time.time()
    
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
    
    def _check_and_wait_for_rate_limit(self):
        """Zkontroluje kumulativní tokeny a počká, pokud je dosaženo prahu."""
        # Zjednodušený reset - pokud uplynula více než minuta od posledního resetu, vynulujeme
        # To nemusí být přesné, ale je to jednoduchá aproximace minutového okna Anthropicu
        if time.time() - self.last_reset_time > 60:
             print("Uplynula minuta od posledního resetu/čekání, resetuji kumulativní tokeny.")
             self.cumulative_input_tokens = 0
             self.cumulative_output_tokens = 0
             self.last_reset_time = time.time()

        if self.cumulative_input_tokens >= self.INPUT_TOKEN_THRESHOLD or \
           self.cumulative_output_tokens >= self.OUTPUT_TOKEN_THRESHOLD:
            print(f"WARN: Dosažen práh tokenů (In: {self.cumulative_input_tokens}/{self.INPUT_TOKEN_THRESHOLD}, Out: {self.cumulative_output_tokens}/{self.OUTPUT_TOKEN_THRESHOLD}). Čekám 60 sekund...")
            time.sleep(60)
            print("Pokračuji po čekání...")
            # Po čekání resetujeme čítače a čas
            self.cumulative_input_tokens = 0
            self.cumulative_output_tokens = 0
            self.last_reset_time = time.time()

    def generate_text(self, prompt: str, **kwargs) -> tuple[str, dict]:
        """
        Generuje text na základě promptu pomocí Anthropic API.
        
        Args:
            prompt: Prompt pro generování textu
            **kwargs: Další parametry pro generování textu
            
        Returns:
            tuple: Vygenerovaný text a slovník s počty tokenů ({'input_tokens': N, 'output_tokens': M})
        """
        # Kontrola limitu PŘED voláním API
        self._check_and_wait_for_rate_limit()

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 1000)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        token_usage = {"input_tokens": 0, "output_tokens": 0} # Default
        try:
            print(f"Odesílám požadavek na Anthropic API (model: {self.model_name})")
            # print(f"API klíč začíná: {self.api_key[:10]}..., délka: {len(self.api_key)}")
            response = requests.post(f"{self.api_base}/messages", headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                response_data = response.json()
                text_content = response_data["content"][0]["text"]
                # Extrahovat token usage, pokud existuje
                if "usage" in response_data:
                    usage_data = response_data["usage"]
                    token_usage["input_tokens"] = usage_data.get("input_tokens", 0)
                    token_usage["output_tokens"] = usage_data.get("output_tokens", 0)
                    # Aktualizace kumulativních tokenů PO úspěšném volání
                    self.cumulative_input_tokens += token_usage["input_tokens"]
                    self.cumulative_output_tokens += token_usage["output_tokens"]
                    print(f"  Anthropic Tokens Used (request): In={token_usage['input_tokens']}, Out={token_usage['output_tokens']}")
                    print(f"  Anthropic Tokens Cumulative (since last wait/reset): In={self.cumulative_input_tokens}, Out={self.cumulative_output_tokens}")
                return text_content, token_usage
            else:
                try:
                    error_data = response.json()
                    error_type = error_data.get('error', {}).get('type', 'unknown')
                    error_message = error_data.get('error', {}).get('message', 'No message')
                    error_msg = f"Chyba při dotazu na Anthropic API: {response.status_code} - {response.text}"
                    print(f"Detaily chyby: typ={error_type}, zpráva={error_message}")
                    # print(f"Použitá URL: {self.api_base}/messages")
                    # print(f"Použité hlavičky: {headers}")
                except Exception as e:
                    error_msg = f"Chyba při dotazu na Anthropic API: {response.status_code} - {response.text}"
                    print(error_msg)
                return "", token_usage # Vrátit prázdný string a nulové tokeny při chybě
        except requests.exceptions.RequestException as e:
            print(f"Síťová chyba při dotazu na Anthropic API: {e}")
            return "", token_usage # Vrátit prázdný string a nulové tokeny při chybě
    
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
    
    # Limity tokenů pro čekání (sdílené s textovým providerem, pokud jsou stejná instance?)
    # Pro jistotu je definujeme i zde, i když by instance měla být sdílená v rámci logiky
    INPUT_TOKEN_THRESHOLD = 40000
    OUTPUT_TOKEN_THRESHOLD = 8000
    
    def __init__(self, model_name: str = "claude-3-haiku-20240307"):
        """
        Inicializace poskytovatele Anthropic Vision.
        
        Args:
            model_name: Název modelu pro použití
        """
        self.model_name = model_name
        self.api_key = None
        self.api_base = "https://api.anthropic.com/v1"
        # Kumulativní počty tokenů pro sledování limitu
        self.cumulative_input_tokens = 0
        self.cumulative_output_tokens = 0
        self.last_reset_time = time.time()
    
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
    
    def _check_and_wait_for_rate_limit(self):
        """Zkontroluje kumulativní tokeny a počká, pokud je dosaženo prahu."""
        # Zjednodušený reset - pokud uplynula více než minuta od posledního resetu, vynulujeme
        if time.time() - self.last_reset_time > 60:
             print("Uplynula minuta od posledního resetu/čekání, resetuji kumulativní tokeny.")
             self.cumulative_input_tokens = 0
             self.cumulative_output_tokens = 0
             self.last_reset_time = time.time()

        if self.cumulative_input_tokens >= self.INPUT_TOKEN_THRESHOLD or \
           self.cumulative_output_tokens >= self.OUTPUT_TOKEN_THRESHOLD:
            print(f"WARN: Dosažen práh tokenů (In: {self.cumulative_input_tokens}/{self.INPUT_TOKEN_THRESHOLD}, Out: {self.cumulative_output_tokens}/{self.OUTPUT_TOKEN_THRESHOLD}). Čekám 60 sekund...")
            time.sleep(60)
            print("Pokračuji po čekání...")
            # Po čekání resetujeme čítače a čas
            self.cumulative_input_tokens = 0
            self.cumulative_output_tokens = 0
            self.last_reset_time = time.time()

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
        Generuje text na základě obrázku a promptu pomocí Anthropic API.
        
        Args:
            image: PIL.Image objekt
            prompt: Prompt pro generování textu
            **kwargs: Další parametry pro generování textu
            
        Returns:
            tuple: Vygenerovaný text a slovník s počty tokenů ({'input_tokens': N, 'output_tokens': M})
        """
        # Kontrola limitu PŘED voláním API
        self._check_and_wait_for_rate_limit()

        image_base64 = self.encode_image(image)
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
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}
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
        token_usage = {"input_tokens": 0, "output_tokens": 0} # Default
        try:
            print(f"Odesílám požadavek na Anthropic Vision API (model: {self.model_name})")
            # print(f"API klíč začíná: {self.api_key[:10]}..., délka: {len(self.api_key)}")
            response = requests.post(f"{self.api_base}/messages", headers=headers, json=payload, timeout=60) # Zvýšený timeout pro VLM
            
            if response.status_code == 200:
                response_data = response.json()
                text_content = response_data["content"][0]["text"]
                # Extrahovat token usage, pokud existuje
                if "usage" in response_data:
                    usage_data = response_data["usage"]
                    token_usage["input_tokens"] = usage_data.get("input_tokens", 0)
                    token_usage["output_tokens"] = usage_data.get("output_tokens", 0)
                    # Aktualizace kumulativních tokenů PO úspěšném volání
                    self.cumulative_input_tokens += token_usage["input_tokens"]
                    self.cumulative_output_tokens += token_usage["output_tokens"]
                    print(f"  Anthropic Tokens Used (request): In={token_usage['input_tokens']}, Out={token_usage['output_tokens']}")
                    print(f"  Anthropic Tokens Cumulative (since last wait/reset): In={self.cumulative_input_tokens}, Out={self.cumulative_output_tokens}")
                return text_content, token_usage
            else:
                try:
                    error_data = response.json()
                    error_type = error_data.get('error', {}).get('type', 'unknown')
                    error_message = error_data.get('error', {}).get('message', 'No message')
                    error_msg = f"Chyba při dotazu na Anthropic Vision API: {response.status_code} - {response.text}"
                    print(f"Detaily chyby: typ={error_type}, zpráva={error_message}")
                except Exception as e:
                    error_msg = f"Chyba při dotazu na Anthropic Vision API: {response.status_code} - {response.text}"
                    print(error_msg)
                return "", token_usage # Vrátit prázdný string a nulové tokeny při chybě
        except requests.exceptions.RequestException as e:
            print(f"Síťová chyba při dotazu na Anthropic Vision API: {e}")
            return "", token_usage # Vrátit prázdný string a nulové tokeny při chybě
    
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od Anthropic.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        return self.AVAILABLE_MODELS 