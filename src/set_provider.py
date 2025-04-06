#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skript pro nastavení poskytovatele API před spuštěním hlavního skriptu.
"""

import os
import json
import shutil
from pathlib import Path
import argparse

# Definice cest
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
MODELS_JSON = CONFIG_DIR / "models.json"
MODELS_ANTHROPIC_JSON = CONFIG_DIR / "models_anthropic.json"
MODELS_OPENAI_JSON = CONFIG_DIR / "models_openai.json"

def set_provider(provider_name):
    """
    Nastaví poskytovatele API před spuštěním hlavního skriptu.
    
    Args:
        provider_name (str): Název poskytovatele (anthropic nebo openai)
        
    Returns:
        bool: True pokud byl poskytovatel úspěšně nastaven, jinak False
    """
    if provider_name.lower() == "anthropic":
        # Kontrola, zda existuje soubor models_anthropic.json
        if not MODELS_ANTHROPIC_JSON.exists():
            print(f"Chyba: Soubor {MODELS_ANTHROPIC_JSON} neexistuje.")
            return False
        
        # Záloha aktuální konfigurace
        if MODELS_JSON.exists() and not MODELS_OPENAI_JSON.exists():
            print(f"Zálohuji aktuální konfiguraci do {MODELS_OPENAI_JSON}...")
            shutil.copy(MODELS_JSON, MODELS_OPENAI_JSON)
        
        # Kopírování konfigurace pro Anthropic
        print(f"Nastavuji poskytovatele na Anthropic...")
        shutil.copy(MODELS_ANTHROPIC_JSON, MODELS_JSON)
        
        return True
    
    elif provider_name.lower() == "openai":
        # Kontrola, zda existuje soubor models_openai.json
        if MODELS_OPENAI_JSON.exists():
            # Kopírování konfigurace pro OpenAI
            print(f"Nastavuji poskytovatele na OpenAI...")
            shutil.copy(MODELS_OPENAI_JSON, MODELS_JSON)
        else:
            # Vytvoření výchozí konfigurace pro OpenAI
            openai_config = {
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
                }
            }
            
            # Uložení konfigurace
            print(f"Vytvářím výchozí konfiguraci pro OpenAI...")
            with open(MODELS_JSON, 'w', encoding='utf-8') as f:
                json.dump(openai_config, f, ensure_ascii=False, indent=2)
            
            # Záloha konfigurace
            with open(MODELS_OPENAI_JSON, 'w', encoding='utf-8') as f:
                json.dump(openai_config, f, ensure_ascii=False, indent=2)
        
        return True
    
    else:
        print(f"Chyba: Neznámý poskytovatel '{provider_name}'. Podporováni jsou 'anthropic' a 'openai'.")
        return False

def main():
    """
    Hlavní funkce skriptu.
    """
    parser = argparse.ArgumentParser(description="Nastavení poskytovatele API před spuštěním hlavního skriptu")
    parser.add_argument("provider", choices=["anthropic", "openai"], help="Název poskytovatele (anthropic nebo openai)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Podrobnější výstup")
    
    args = parser.parse_args()
    
    # Nastavení poskytovatele
    success = set_provider(args.provider)
    
    if success:
        print(f"Poskytovatel byl úspěšně nastaven na '{args.provider}'.")
        
        # Zobrazení aktuální konfigurace
        if args.verbose and MODELS_JSON.exists():
            print("\nAktuální konfigurace:")
            with open(MODELS_JSON, 'r', encoding='utf-8') as f:
                config = json.load(f)
                for section, section_config in config.items():
                    provider = section_config.get("provider", "N/A")
                    model = section_config.get("model", "N/A")
                    print(f"  {section}: provider='{provider}', model='{model}'")
    else:
        print("Chyba při nastavování poskytovatele.")
        exit(1)

if __name__ == "__main__":
    main() 