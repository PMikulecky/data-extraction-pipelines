#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro implementaci poskytovatele OpenAI API.
"""

import os
import base64
from io import BytesIO
import requests
from typing import Dict, List, Any, Optional
from PIL import Image

from ..base.provider import TextModelProvider, VisionModelProvider, EmbeddingModelProvider


class OpenAITextModelProvider(TextModelProvider):
    """
    Třída pro implementaci poskytovatele textových modelů OpenAI.
    """
    
    # Seznam dostupných textových modelů
    AVAILABLE_MODELS = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o"
    ]
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Inicializace poskytovatele OpenAI.
        
        Args:
            model_name: Název modelu pro použití
        """
        self.model_name = model_name
        self.api_key = None
        self.api_base = "https://api.openai.com/v1"
    
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s API klíčem a dalšími parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu
            **kwargs: Další parametry specifické pro OpenAI
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API klíč OpenAI není k dispozici. Nastavte proměnnou prostředí OPENAI_API_KEY nebo předejte api_key parametr.")
        
        # Zpracování dalších parametrů
        if "api_base" in kwargs:
            self.api_base = kwargs["api_base"]
        
        if "model_name" in kwargs:
            self.model_name = kwargs["model_name"]
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generuje text na základě promptu pomocí OpenAI API.
        
        Args:
            prompt: Prompt pro generování textu
            **kwargs: Další parametry pro generování textu
            
        Returns:
            Vygenerovaný text
        """
        # Příprava dotazu pro API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
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
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload
        )
        
        # Zpracování odpovědi
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            error_msg = f"Chyba při dotazu na OpenAI API: {response.status_code} - {response.text}"
            print(error_msg)
            raise Exception(error_msg)
    
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od OpenAI.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        return self.AVAILABLE_MODELS


class OpenAIVisionModelProvider(VisionModelProvider):
    """
    Třída pro implementaci poskytovatele vizuálních modelů OpenAI.
    """
    
    # Seznam dostupných vizuálních modelů
    AVAILABLE_MODELS = [
        "gpt-4-vision-preview",
        "gpt-4o"
    ]
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Inicializace poskytovatele OpenAI Vision.
        
        Args:
            model_name: Název modelu pro použití
        """
        self.model_name = model_name
        self.api_key = None
        self.api_base = "https://api.openai.com/v1"
    
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s API klíčem a dalšími parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu
            **kwargs: Další parametry specifické pro OpenAI
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API klíč OpenAI není k dispozici. Nastavte proměnnou prostředí OPENAI_API_KEY nebo předejte api_key parametr.")
        
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
        Generuje text na základě obrázku a promptu pomocí OpenAI API.
        
        Args:
            image: Obrázek (PIL.Image)
            prompt: Prompt pro generování textu
            **kwargs: Další parametry pro generování textu
            
        Returns:
            Vygenerovaný text
        """
        # Zakódování obrázku
        base64_image = self.encode_image(image)
        
        # Příprava dotazu pro API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        max_tokens = kwargs.get("max_tokens", 1000)
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens
        }
        
        # Odeslání dotazu
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload
        )
        
        # Zpracování odpovědi
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            error_msg = f"Chyba při dotazu na OpenAI Vision API: {response.status_code} - {response.text}"
            print(error_msg)
            raise Exception(error_msg)
    
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od OpenAI.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        return self.AVAILABLE_MODELS


class OpenAIEmbeddingModelProvider(EmbeddingModelProvider):
    """
    Třída pro implementaci poskytovatele embedding modelů OpenAI.
    """
    
    # Seznam dostupných embedding modelů
    AVAILABLE_MODELS = [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large"
    ]
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Inicializace poskytovatele OpenAI Embeddings.
        
        Args:
            model_name: Název modelu pro použití
        """
        self.model_name = model_name
        self.api_key = None
        self.api_base = "https://api.openai.com/v1"
    
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s API klíčem a dalšími parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu
            **kwargs: Další parametry specifické pro OpenAI
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API klíč OpenAI není k dispozici. Nastavte proměnnou prostředí OPENAI_API_KEY nebo předejte api_key parametr.")
        
        # Zpracování dalších parametrů
        if "api_base" in kwargs:
            self.api_base = kwargs["api_base"]
        
        if "model_name" in kwargs:
            self.model_name = kwargs["model_name"]
    
    def get_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Získá embedding vektory pro zadané texty pomocí OpenAI API.
        
        Args:
            texts: Seznam textů pro získání embeddingů
            **kwargs: Další parametry pro generování embeddingů
            
        Returns:
            Seznam embedding vektorů
        """
        # Příprava dotazu pro API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "input": texts,
            "model": self.model_name
        }
        
        # Odeslání dotazu
        response = requests.post(
            f"{self.api_base}/embeddings",
            headers=headers,
            json=payload
        )
        
        # Zpracování odpovědi
        if response.status_code == 200:
            data = response.json()["data"]
            return [item["embedding"] for item in data]
        else:
            error_msg = f"Chyba při získávání embeddingů z OpenAI API: {response.status_code} - {response.text}"
            print(error_msg)
            raise Exception(error_msg)
    
    def get_embedding_function(self):
        """
        Vrací embedding funkci kompatibilní s Chroma a dalšími vektorovými databázemi.
        
        Returns:
            Callable: Funkce, která převádí texty na vektory
        """
        # Nejprve zkusíme novější langchain-openai balíček (preferovaný)
        try:
            from langchain_openai import OpenAIEmbeddings
            
            # Vytvoření instance OpenAIEmbeddings s našimi parametry
            embedding_function = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base
            )
            
            print(f"Vytvořena embedding funkce pomocí langchain_openai.OpenAIEmbeddings pro model {self.model_name}")
            return embedding_function
        except ImportError:
            # Pokud není k dispozici, zkusíme starší langchain_community
            try:
                from langchain_community.embeddings import OpenAIEmbeddings
                
                # Vytvoření instance OpenAIEmbeddings s našimi parametry
                embedding_function = OpenAIEmbeddings(
                    model=self.model_name,
                    openai_api_key=self.api_key,
                    openai_api_base=self.api_base
                )
                
                print(f"Vytvořena embedding funkce pomocí langchain_community.embeddings.OpenAIEmbeddings pro model {self.model_name}")
                return embedding_function
            except ImportError:
                # Pokud ani jedna knihovna není k dispozici, vytvoříme vlastní funkci
                print("Knihovna langchain-openai ani langchain_community.embeddings není k dispozici, používám vlastní implementaci")
                
                def embedding_function(texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    return self.get_embeddings(texts)
                
                return embedding_function
    
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od OpenAI.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        return self.AVAILABLE_MODELS 