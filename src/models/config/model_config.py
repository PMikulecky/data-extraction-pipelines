#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro konfiguraci modelů a poskytovatelů AI.
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path


class ModelConfig:
    """
    Třída pro konfiguraci modelů a poskytovatelů AI.
    """
    
    # Výchozí konfigurace
    DEFAULT_CONFIG = {
        "text": {
            "provider": "openai",
            "model": "gpt-3.5-turbo"
        },
        "vision": {
            "provider": "openai",
            "model": "gpt-4o"
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small"
        },
        "text_pipeline": {
            "enabled": True,
            "max_text_length": 6000,
            "extract_references": True,
            "use_direct_pattern_extraction": True
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializace konfigurace modelů.
        
        Args:
            config_path: Cesta ke konfiguračnímu souboru (volitelné)
        """
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        Načte konfiguraci ze souboru.
        
        Args:
            config_path: Cesta ke konfiguračnímu souboru
        """
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                print(f"Konfigurační soubor {config_path} nebyl nalezen, používám výchozí konfiguraci.")
                return
            
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # Aktualizace konfigurace
            for key in self.config:
                if key in loaded_config:
                    self.config[key].update(loaded_config[key])
            
            print(f"Konfigurace načtena z {config_path}")
        
        except Exception as e:
            print(f"Chyba při načítání konfigurace: {e}")
            print("Používám výchozí konfiguraci.")
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Uloží aktuální konfiguraci do souboru.
        
        Args:
            config_path: Cesta ke konfiguračnímu souboru (volitelné)
        """
        save_path = config_path or self.config_path
        
        if not save_path:
            print("Není zadána cesta pro uložení konfigurace.")
            return
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            
            print(f"Konfigurace uložena do {save_path}")
        
        except Exception as e:
            print(f"Chyba při ukládání konfigurace: {e}")
    
    def get_text_config(self) -> Dict[str, Any]:
        """
        Získá konfiguraci pro textové modely.
        
        Returns:
            Konfigurace pro textové modely
        """
        return self.config.get("text", self.DEFAULT_CONFIG["text"])
    
    def get_vision_config(self) -> Dict[str, Any]:
        """
        Získá konfiguraci pro vizuální modely.
        
        Returns:
            Konfigurace pro vizuální modely
        """
        return self.config.get("vision", self.DEFAULT_CONFIG["vision"])
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Získá konfiguraci pro embedding modely.
        
        Returns:
            Konfigurace pro embedding modely
        """
        return self.config.get("embedding", self.DEFAULT_CONFIG["embedding"])
    
    def set_text_config(self, provider: str, model: str) -> None:
        """
        Nastaví konfiguraci pro textové modely.
        
        Args:
            provider: Název poskytovatele
            model: Název modelu
        """
        self.config["text"] = {
            "provider": provider,
            "model": model
        }
    
    def set_vision_config(self, provider: str, model: str) -> None:
        """
        Nastaví konfiguraci pro vizuální modely.
        
        Args:
            provider: Název poskytovatele
            model: Název modelu
        """
        self.config["vision"] = {
            "provider": provider,
            "model": model
        }
    
    def set_embedding_config(self, provider: str, model: str) -> None:
        """
        Nastaví konfiguraci pro embedding modely.
        
        Args:
            provider: Název poskytovatele
            model: Název modelu
        """
        self.config["embedding"] = {
            "provider": provider,
            "model": model
        }
    
    def get_text_pipeline_config(self) -> Dict[str, Any]:
        """
        Získá konfiguraci pro textovou pipeline.
        
        Returns:
            Konfigurace pro textovou pipeline
        """
        return self.config.get("text_pipeline", self.DEFAULT_CONFIG["text_pipeline"])
    
    def set_text_pipeline_config(self, enabled=True, max_text_length=6000, 
                                extract_references=True, use_direct_pattern_extraction=True) -> None:
        """
        Nastaví konfiguraci pro textovou pipeline.
        
        Args:
            enabled (bool): Zda je pipeline povolena
            max_text_length (int): Maximální délka textu pro dotazy
            extract_references (bool): Zda extrahovat reference
            use_direct_pattern_extraction (bool): Zda používat přímou extrakci pomocí vzorů
        """
        self.config["text_pipeline"] = {
            "enabled": enabled,
            "max_text_length": max_text_length,
            "extract_references": extract_references,
            "use_direct_pattern_extraction": use_direct_pattern_extraction
        }
    
    def get_available_providers(self) -> Dict[str, List[str]]:
        """
        Získá seznam dostupných poskytovatelů pro jednotlivé typy modelů.
        
        Returns:
            Slovník s dostupnými poskytovateli pro každý typ modelu
        """
        from ..providers.factory import ModelProviderFactory
        
        return {
            "text": list(ModelProviderFactory.TEXT_PROVIDERS.keys()),
            "vision": list(ModelProviderFactory.VISION_PROVIDERS.keys()),
            "embedding": list(ModelProviderFactory.EMBEDDING_PROVIDERS.keys())
        }
    
    def get_available_models(self, provider_name: str, model_type: str) -> List[str]:
        """
        Získá seznam dostupných modelů pro daného poskytovatele a typ modelu.
        
        Args:
            provider_name: Název poskytovatele
            model_type: Typ modelu (text, vision, embedding)
            
        Returns:
            Seznam dostupných modelů
            
        Raises:
            ValueError: Pokud poskytovatel nebo typ modelu není podporován
        """
        from ..providers.factory import ModelProviderFactory
        
        if model_type == "text":
            if provider_name not in ModelProviderFactory.TEXT_PROVIDERS:
                raise ValueError(f"Poskytovatel '{provider_name}' není podporován pro textové modely.")
            
            provider_class = ModelProviderFactory.TEXT_PROVIDERS[provider_name]
            return provider_class.AVAILABLE_MODELS
        
        elif model_type == "vision":
            if provider_name not in ModelProviderFactory.VISION_PROVIDERS:
                raise ValueError(f"Poskytovatel '{provider_name}' není podporován pro vizuální modely.")
            
            provider_class = ModelProviderFactory.VISION_PROVIDERS[provider_name]
            return provider_class.AVAILABLE_MODELS
        
        elif model_type == "embedding":
            if provider_name not in ModelProviderFactory.EMBEDDING_PROVIDERS:
                raise ValueError(f"Poskytovatel '{provider_name}' není podporován pro embedding modely.")
            
            provider_class = ModelProviderFactory.EMBEDDING_PROVIDERS[provider_name]
            return provider_class.AVAILABLE_MODELS
        
        else:
            raise ValueError(f"Neplatný typ modelu: {model_type}. Podporované typy: text, vision, embedding.")


# Globální instance konfigurace
config = ModelConfig()


def load_config(config_path: str) -> ModelConfig:
    """
    Načte konfiguraci z externího souboru a vytvoří globální instanci.
    
    Args:
        config_path: Cesta ke konfiguračnímu souboru
        
    Returns:
        Instance ModelConfig
    """
    global config
    config = ModelConfig(config_path)
    return config


def get_config() -> ModelConfig:
    """
    Získá globální instanci konfigurace.
    
    Returns:
        Instance ModelConfig
    """
    return config 