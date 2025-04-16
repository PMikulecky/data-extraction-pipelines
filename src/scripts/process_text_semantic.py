#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skript pro sémantickou analýzu a aktualizaci výsledků porovnání text pipeline.
"""

import os
import sys
import json
from pathlib import Path

# Přidání cesty k projektu
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.semantic_comparison import semantic_compare_and_update

def main():
    """
    Hlavní funkce skriptu.
    """
    # Cesty k souborům
    base_dir = Path(__file__).resolve().parent.parent.parent
    input_path = base_dir / "results/openai-gpt_20250416_001451/text_comparison.json"
    output_path = base_dir / "results/openai-gpt_20250416_001451/text_comparison_semantic.json"
    
    # Kontrola existence vstupního souboru
    if not input_path.exists():
        print(f"Chyba: Vstupní soubor {input_path} neexistuje.")
        return 1
    
    # Načtení dat
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Chyba při načítání vstupního souboru: {e}")
        return 1
    
    # Zpracování dat
    print(f"Zpracovávám soubor {input_path}...")
    try:
        updated_data = semantic_compare_and_update(data)
    except Exception as e:
        print(f"Chyba při zpracování dat: {e}")
        return 1
    
    # Uložení výsledků
    try:
        # Vytvoření adresáře, pokud neexistuje
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Chyba při ukládání výstupního souboru: {e}")
        return 1
    
    print(f"Sémanticky vylepšené porovnání uloženo do {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 