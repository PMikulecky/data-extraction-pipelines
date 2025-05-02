#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro implementaci poskytovatele Ollama API.
"""

import os
import base64
from io import BytesIO
import requests
from typing import Dict, List, Any, Optional
from PIL import Image

from ..base.provider import TextModelProvider, VisionModelProvider, EmbeddingModelProvider


class OllamaTextModelProvider(TextModelProvider):
    """
    Třída pro implementaci poskytovatele textových modelů Ollama.
    """
    
    # Seznam dostupných textových modelů - bude doplněn dynamicky
    AVAILABLE_MODELS = []
    
    def __init__(self, model_name: str = "llama2"):
        """
        Inicializace poskytovatele Ollama.
        
        Args:
            model_name: Název modelu pro použití
        """
        self.model_name = model_name
        self.api_base = "http://127.0.0.1:11434/api"
    
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu (pro Ollama není vyžadován)
            **kwargs: Další parametry specifické pro Ollama
        """
        # Zpracování dalších parametrů
        if "api_base" in kwargs:
            self.api_base = kwargs["api_base"]
        
        if "model_name" in kwargs:
            self.model_name = kwargs["model_name"]
        
        # Zkusíme získat seznam dostupných modelů
        try:
            self._update_available_models()
        except Exception as e:
            print(f"Varování: Nepodařilo se získat seznam modelů z Ollama API: {e}")
    
    def _update_available_models(self) -> None:
        """
        Aktualizuje seznam dostupných modelů z Ollama API.
        """
        try:
            response = requests.get(f"{self.api_base}/tags", timeout=5)
            
            if response.status_code == 200:
                models_data = response.json()
                if "models" in models_data:
                    OllamaTextModelProvider.AVAILABLE_MODELS = [model["name"] for model in models_data["models"]]
                else:
                    # Starší verze API
                    OllamaTextModelProvider.AVAILABLE_MODELS = [model["name"] for model in models_data]
            else:
                print(f"Chyba při získávání seznamu modelů z Ollama API: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Výjimka při získávání seznamu modelů z Ollama API: {e}")
    
    def generate_text(self, prompt: str, **kwargs) -> tuple[str, dict]:
        """
        Generuje text na základě promptu pomocí Ollama API.
        
        Args:
            prompt: Prompt pro generování textu
            **kwargs: Další parametry pro generování textu
            
        Returns:
            tuple: Vygenerovaný text a slovník s počty tokenů ({'input_tokens': 0, 'output_tokens': 0} - Ollama API je nevrací standardně)
        """
        headers = {"Content-Type": "application/json"}
        temperature = kwargs.get("temperature", 0.0)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
        for param in ['num_predict', 'top_k', 'top_p', 'repeat_penalty', 'presence_penalty', 'frequency_penalty', 'stop']:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        # Default token usage for Ollama
        token_usage = {"input_tokens": 0, "output_tokens": 0}
        
        try:
            response = requests.post(f"{self.api_base}/generate", headers=headers, json=payload, timeout=500)
            
            if response.status_code == 200:
                # Zkusíme získat tokeny, pokud jsou v odpovědi (některé verze/modely je mohou vracet)
                response_data = response.json()
                text_content = response_data.get("response", "")
                token_usage["input_tokens"] = response_data.get("prompt_eval_count", 0)
                token_usage["output_tokens"] = response_data.get("eval_count", 0)
                return text_content, token_usage
            else:
                error_msg = f"Chyba při dotazu na Ollama API: {response.status_code} - {response.text}"
                print(error_msg)
                return "", token_usage # Vrátit default při chybě
        except requests.exceptions.RequestException as e:
            print(f"Síťová chyba při dotazu na Ollama API: {e}")
            return "", token_usage # Vrátit default při chybě
    
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od Ollama.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        # Pokud jsme ještě nezískali modely, zkusíme to nyní
        if not OllamaTextModelProvider.AVAILABLE_MODELS:
            self._update_available_models()
        
        return OllamaTextModelProvider.AVAILABLE_MODELS


class OllamaVisionModelProvider(VisionModelProvider):
    """
    Třída pro implementaci poskytovatele vizuálních modelů Ollama.
    """
    
    # Seznam dostupných vizuálních modelů - bude doplněn dynamicky
    AVAILABLE_MODELS = []
    
    def __init__(self, model_name: str = "llava"):
        """
        Inicializace poskytovatele Ollama pro vizuální modely.
        
        Args:
            model_name: Název modelu pro použití (např. llava, bakllava)
        """
        self.model_name = model_name
        self.api_base = "http://127.0.0.1:11434/api"
    
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu (pro Ollama není vyžadován)
            **kwargs: Další parametry specifické pro Ollama
        """
        # Zpracování dalších parametrů
        if "api_base" in kwargs:
            self.api_base = kwargs["api_base"]
        
        if "model_name" in kwargs:
            self.model_name = kwargs["model_name"]
        
        # Zkusíme získat seznam dostupných modelů
        try:
            self._update_available_models()
        except Exception as e:
            print(f"Varování: Nepodařilo se získat seznam modelů z Ollama API: {e}")
    
    def _update_available_models(self) -> None:
        """
        Aktualizuje seznam dostupných modelů z Ollama API.
        Filtruje pouze multimodální modely, které podporují obrázky.
        """
        try:
            response = requests.get(f"{self.api_base}/tags", timeout=5)
            
            if response.status_code == 200:
                models_data = response.json()
                if "models" in models_data:
                    # Příklady multimodálních modelů: llava, bakllava, moondream, co:here (všechny modely podporující vision)
                    multimodal_model_prefixes = ["llava", "bakllava", "clip", "vision", "cogvlm", "moondream", "co:here"]
                    
                    all_models = [model["name"] for model in models_data["models"]]
                    OllamaVisionModelProvider.AVAILABLE_MODELS = [
                        model for model in all_models
                        if any(model.startswith(prefix) or prefix in model.lower() for prefix in multimodal_model_prefixes)
                    ]
                else:
                    # Starší verze API
                    all_models = [model["name"] for model in models_data]
                    multimodal_model_prefixes = ["llava", "bakllava", "clip", "vision", "cogvlm", "moondream", "co:here"]
                    OllamaVisionModelProvider.AVAILABLE_MODELS = [
                        model for model in all_models
                        if any(model.startswith(prefix) or prefix in model.lower() for prefix in multimodal_model_prefixes)
                    ]
            else:
                print(f"Chyba při získávání seznamu modelů z Ollama API: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Výjimka při získávání seznamu modelů z Ollama API: {e}")
            
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
        Generuje text na základě obrázku a promptu pomocí Ollama API.
        
        Args:
            image: PIL.Image objekt
            prompt: Prompt pro generování textu
            **kwargs: Další parametry pro generování textu
            
        Returns:
            tuple: Vygenerovaný text a slovník s počty tokenů ({'input_tokens': 0, 'output_tokens': 0} - Ollama API je nevrací standardně)
        """
        base64_image = self.encode_image(image)
        headers = {"Content-Type": "application/json"}
        temperature = kwargs.get("temperature", 0.0)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "images": [base64_image],
            "stream": False
        }
        for param in ['num_predict', 'top_k', 'top_p', 'repeat_penalty', 'presence_penalty', 'frequency_penalty', 'stop']:
            if param in kwargs:
                payload[param] = kwargs[param]
                
        # Default token usage for Ollama
        token_usage = {"input_tokens": 0, "output_tokens": 0}
        
        try:
            response = requests.post(f"{self.api_base}/generate", headers=headers, json=payload, timeout=500)
            
            if response.status_code == 200:
                 # Zkusíme získat tokeny, pokud jsou v odpovědi
                response_data = response.json()
                text_content = response_data.get("response", "")
                token_usage["input_tokens"] = response_data.get("prompt_eval_count", 0)
                token_usage["output_tokens"] = response_data.get("eval_count", 0)
                return text_content, token_usage
            else:
                error_msg = f"Chyba při dotazu na Ollama Vision API: {response.status_code} - {response.text}"
                print(error_msg)
                return "", token_usage # Vrátit default při chybě
        except requests.exceptions.RequestException as e:
            print(f"Síťová chyba při dotazu na Ollama Vision API: {e}")
            return "", token_usage # Vrátit default při chybě
    
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných vizuálních modelů od Ollama.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        # Pokud jsme ještě nezískali modely, zkusíme to nyní
        if not OllamaVisionModelProvider.AVAILABLE_MODELS:
            self._update_available_models()
        
        # Pokud stále nemáme modely, vrátíme alespoň základní seznam
        if not OllamaVisionModelProvider.AVAILABLE_MODELS:
            return ["llava", "bakllava", "moondream"]
        
        return OllamaVisionModelProvider.AVAILABLE_MODELS


class OllamaEmbeddingModelProvider(EmbeddingModelProvider):
    """
    Třída pro implementaci poskytovatele embedding modelů Ollama.
    """
    
    def __init__(self, model_name: str = "llama2"):
        """
        Inicializace poskytovatele Ollama pro embeddings.
        
        Args:
            model_name: Název modelu pro použití
        """
        self.model_name = model_name
        self.api_base = "http://127.0.0.1:11434/api"
    
    def initialize(self, api_key: Optional[str] = None, **kwargs) -> None:
        """
        Inicializuje poskytovatele modelu s parametry.
        
        Args:
            api_key: API klíč pro přístup k modelu (pro Ollama není vyžadován)
            **kwargs: Další parametry specifické pro Ollama
        """
        # Zpracování dalších parametrů
        if "api_base" in kwargs:
            self.api_base = kwargs["api_base"]
        
        if "model_name" in kwargs:
            self.model_name = kwargs["model_name"]
    
    def get_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Získá embedding vektory pro zadané texty pomocí Ollama API.
        
        Args:
            texts: Seznam textů pro získání embeddingů
            **kwargs: Další parametry pro generování embeddingů
            
        Returns:
            Seznam embedding vektorů
        """
        result = []
        
        # Zpracujeme každý text samostatně
        for text in texts:
            # Příprava dotazu pro API
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            # Odeslání dotazu
            try:
                response = requests.post(
                    f"{self.api_base}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=180  # Timeout nastaven na 180 sekund
                )
                
                # Zpracování odpovědi
                if response.status_code == 200:
                    result.append(response.json()["embedding"])
                else:
                    error_msg = f"Chyba při získávání embeddings z Ollama API: {response.status_code} - {response.text}"
                    print(error_msg)
                    raise Exception(error_msg)
            except requests.exceptions.RequestException as e:
                print(f"Síťová chyba při dotazu na Ollama API: {e}")
                raise Exception(f"Síťová chyba při dotazu na Ollama API embeddings: {e}")
        
        return result
    
    def get_embedding_function(self):
        """
        Vrací embedding funkci kompatibilní s Chroma a dalšími vektorovými databázemi.
        
        Returns:
            Callable: Funkce, která převádí texty na vektory
        """
        try:
            # Pokus o import funkcí z langchain pro vytvoření adaptéru
            from langchain_core.embeddings import Embeddings
            
            # Vytvoření vlastní třídy embeddings kompatibilní s langchain
            class OllamaEmbeddings(Embeddings):
                def __init__(self, provider):
                    self.provider = provider
                
                def embed_documents(self, texts):
                    """Embed a list of documents using Ollama."""
                    return self.provider.get_embeddings(texts)
                
                def embed_query(self, text):
                    """Embed a query using Ollama."""
                    return self.provider.get_embeddings([text])[0]
            
            # Vytvoření instance embeddings
            embedding_function = OllamaEmbeddings(self)
            
            print(f"Vytvořena embedding funkce pro Ollama s modelem {self.model_name}")
            return embedding_function
            
        except ImportError:
            # Pokud langchain není k dispozici, vytvoříme vlastní funkci
            print("Knihovna langchain_core.embeddings není k dispozici, používám vlastní implementaci")
            
            def embedding_function(texts):
                if isinstance(texts, str):
                    texts = [texts]
                return self.get_embeddings(texts)
            
            return embedding_function
    
    def get_available_models(self) -> List[str]:
        """
        Vrací seznam dostupných modelů od Ollama použitelných pro embedding.
        
        Returns:
            Seznam názvů dostupných modelů
        """
        # Pro embedding můžeme použít jakýkoliv textový model z Ollama
        return OllamaTextModelProvider().get_available_models() 