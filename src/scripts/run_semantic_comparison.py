#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skript pro spuštění sémantického porovnání mezi extrahovanými a referenčními daty.
Používá vylepšený algoritmus s podporou pro LLM porovnávání autorů.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Přidání cesty k projektu
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.semantic_comparison import process_comparison_files

# Načtení proměnných prostředí z .env souboru
load_dotenv()

def main():
    """
    Hlavní funkce skriptu pro spuštění sémantického porovnání.
    """
    parser = argparse.ArgumentParser(
        description='Spustí sémantické porovnání mezi extrahovanými a referenčními daty.'
    )
    
    parser.add_argument(
        '--vlm',
        type=str,
        default='results/vlm_comparison.json',
        help='Cesta k VLM porovnávacímu souboru (výchozí: results/vlm_comparison.json)'
    )
    
    parser.add_argument(
        '--embedded',
        type=str,
        default='results/embedded_comparison.json',
        help='Cesta k Embedded porovnávacímu souboru (výchozí: results/embedded_comparison.json)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/semantic_comparison_results.json',
        help='Cesta k výstupnímu souboru (výchozí: results/semantic_comparison_results.json)'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Vypne použití LLM pro porovnávání autorů (úspora nákladů)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Specifikuje model LLM (např. gpt-3.5-turbo, gpt-4-turbo)'
    )
    
    args = parser.parse_args()
    
    # Nastavení proměnných prostředí z příkazové řádky
    if args.no_llm:
        os.environ["USE_LLM_FOR_AUTHORS"] = "0"
    
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    
    # Absolutní cesty k souborům
    base_dir = Path(__file__).resolve().parent.parent.parent
    vlm_path = base_dir / args.vlm
    embedded_path = base_dir / args.embedded
    output_path = base_dir / args.output
    
    # Kontrola existence souborů
    if not vlm_path.exists():
        print(f"Chyba: Soubor {vlm_path} neexistuje")
        return 1
    
    if not embedded_path.exists():
        print(f"Chyba: Soubor {embedded_path} neexistuje")
        return 1
    
    # Zpracování souborů a spuštění sémantického porovnání
    try:
        print(f"Spouštím sémantické porovnání...")
        print(f"VLM soubor: {vlm_path}")
        print(f"Embedded soubor: {embedded_path}")
        print(f"Výstupní soubor: {output_path}")
        print(f"Použití LLM: {'Vypnuto' if args.no_llm else 'Zapnuto'}")
        if args.model:
            print(f"LLM model: {args.model}")
        
        vlm_updated, embedded_updated = process_comparison_files(
            str(vlm_path),
            str(embedded_path),
            str(output_path)
        )
        
        print(f"\nSémantické porovnání dokončeno.")
        return 0
    
    except Exception as e:
        print(f"Chyba při sémantickém porovnání: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 