#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skript pro spuštění extrakce metadat s různými konfiguracemi modelů Ollama.
Skript projde definované konfigurace a postupně spustí hlavní program pro každou z nich.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Cesty k důležitým adresářům a souborům
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
RESULTS_DIR = BASE_DIR / "results"
TEMP_CONFIG_FILE = CONFIG_DIR / "temp_model_config.json"
MAIN_SCRIPT = BASE_DIR / "src" / "main.py"


def parse_args():
    """Parsování argumentů příkazové řádky."""
    parser = argparse.ArgumentParser(description="Spustí extrakci metadat s různými konfiguracemi modelů Ollama.")
    parser.add_argument('--config', type=str, default=None, help="Cesta ke konfiguračnímu souboru s definicemi modelů")
    parser.add_argument('--limit', type=int, default=None, help="Omezení počtu zpracovaných souborů")
    parser.add_argument('--year-filter', type=int, nargs='+', default=None, help="Filtrování článků podle roku vydání")
    parser.add_argument('--skip-download', action='store_true', help="Přeskočí stahování PDF souborů")
    parser.add_argument('--skip-semantic', action='store_true', help="Přeskočí sémantické porovnání výsledků")
    parser.add_argument('--force-extraction', action='store_true', help="Vynutí novou extrakci i když výsledky již existují")
    parser.add_argument('--verbose', action='store_true', help="Podrobnější výstup")
    return parser.parse_args()


def get_ollama_configurations() -> List[Dict[str, Any]]:
    """
    Vrátí seznam předefinovaných konfigurací modelů Ollama.
    Každá konfigurace obsahuje kompletní nastavení pro text, vision a embedding modely.
    
    Returns:
        List[Dict[str, Any]]: Seznam konfigurací modelů
    """
    # Výchozí konfigurace, pokud nejsou definovány externě
    return [
        # Konfigurace 1: Llama 3.1 + Llama 3.2 Vision
        {
            "name": "llama3-latest",
            "text": {
                "provider": "ollama",
                "model": "llama3.1:8b"
            },
            "vision": {
                "provider": "ollama",
                "model": "llama3.2-vision:11b"
            },
            "embedding": {
                "provider": "ollama",
                "model": "mxbai-embed-large:335m"
            }
        },
        # Konfigurace 2: Mistral + LLaVA
        {
            "name": "mistral-llava",
            "text": {
                "provider": "ollama",
                "model": "mistral:7b-instruct-v0.2-q4_K_M"
            },
            "vision": {
                "provider": "ollama",
                "model": "llava:13b"
            },
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text"
            }
        },
        # Konfigurace 3: Llama 2 + CogVLM
        {
            "name": "llama2-cogvlm",
            "text": {
                "provider": "ollama",
                "model": "llama2:7b"
            },
            "vision": {
                "provider": "ollama",
                "model": "cogvlm"
            },
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text"
            }
        }
    ]


def load_configurations_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Načte konfigurace modelů z externího JSON souboru.
    
    Args:
        file_path (str): Cesta k JSON souboru s konfiguracemi
        
    Returns:
        List[Dict[str, Any]]: Seznam konfigurací modelů
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        
        # Kontrola, že načtený objekt je seznam konfigurací
        if not isinstance(configs, list):
            print(f"Chyba: Soubor {file_path} neobsahuje seznam konfigurací.")
            return get_ollama_configurations()
            
        return configs
    except Exception as e:
        print(f"Chyba při načítání konfigurací ze souboru {file_path}: {e}")
        return get_ollama_configurations()


def save_temp_config(config: Dict[str, Any]) -> None:
    """
    Uloží dočasnou konfiguraci modelu do souboru.
    
    Args:
        config (Dict[str, Any]): Konfigurace modelu k uložení
    """
    # Odebereme klíč "name", který není součástí standardní konfigurace
    config_to_save = {k: v for k, v in config.items() if k != "name"}
    
    try:
        with open(TEMP_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Chyba při ukládání dočasné konfigurace: {e}")
        sys.exit(1)


def prepare_result_directory(config_name: str) -> str:
    """
    Připraví adresář pro výsledky dané konfigurace.
    
    Args:
        config_name (str): Název konfigurace
        
    Returns:
        str: Cesta k adresáři s výsledky
    """
    # Vytvoříme adresář pro výsledky s časovým razítkem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_DIR / f"{config_name}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    return str(result_dir)


def run_extraction(config: Dict[str, Any], args) -> None:
    """
    Spustí extrakci metadat s danou konfigurací.
    
    Args:
        config (Dict[str, Any]): Konfigurace modelu
        args: Argumenty příkazové řádky
    """
    config_name = config.get("name", "unknown_config")
    print(f"\n\n=== Spouštím extrakci s konfigurací: {config_name} ===")
    print(f"Text model: {config['text']['provider']} - {config['text']['model']}")
    print(f"Vision model: {config['vision']['provider']} - {config['vision']['model']}")
    print(f"Embedding model: {config['embedding']['provider']} - {config['embedding']['model']}")
    
    # Uložíme dočasnou konfiguraci
    save_temp_config(config)
    
    # Připravíme výstupní adresář
    result_dir = prepare_result_directory(config_name)
    result_dir_path = Path(result_dir)
    
    # Připravíme argumenty pro main.py
    cmd_args = [sys.executable, str(MAIN_SCRIPT)]
    
    # Přidáme cestu ke konfiguraci
    cmd_args.extend(["--config", str(TEMP_CONFIG_FILE)])
    
    # Přidáme modely
    cmd_args.extend(["--models", "embedded", "vlm"])
    
    # Přidáme další argumenty
    if args.limit:
        cmd_args.extend(["--limit", str(args.limit)])
    if args.year_filter:
        cmd_args.extend(["--year-filter"] + [str(year) for year in args.year_filter])
    if args.skip_download:
        cmd_args.append("--skip-download")
    if args.skip_semantic:
        cmd_args.append("--skip-semantic")
    if args.force_extraction:
        cmd_args.append("--force-extraction")
    if args.verbose:
        cmd_args.append("--verbose")
    
    # Spustíme proces
    print(f"Spouštím příkaz: {' '.join(cmd_args)}")
    try:
        result = subprocess.run(cmd_args, check=True)
        print(f"Extrakce dokončena s návratovým kódem: {result.returncode}")
        
        # Kopírujeme konfigurační soubor do adresáře s výsledky pro pozdější referenci
        shutil.copy(TEMP_CONFIG_FILE, result_dir_path / "used_config.json")
        
        # Kopírujeme výsledky do specifického adresáře
        result_files = [
            "embedded_results.json",
            "vlm_results.json",
            "embedded_comparison.json",
            "vlm_comparison.json",
            "semantic_comparison_results.json",
            "embedded_comparison_semantic.json",
            "vlm_comparison_semantic.json",
            "comparison_results.png",
            "overall_results.png",
            "overall_results.csv",
            "detailed_results.csv",
            "summary_results.csv"
        ]
        
        # Kopírujeme všechny výsledky, které existují
        for filename in result_files:
            source_file = RESULTS_DIR / filename
            if source_file.exists():
                print(f"Kopíruji {filename} do adresáře výsledků...")
                shutil.copy(source_file, result_dir_path / filename)
            
    except subprocess.CalledProcessError as e:
        print(f"Chyba při spuštění extrakce: {e}")
    except KeyboardInterrupt:
        print("\nExtrakce přerušena uživatelem.")
        sys.exit(1)


def main():
    """Hlavní funkce skriptu."""
    args = parse_args()
    
    # Načteme konfigurace
    if args.config:
        configurations = load_configurations_from_file(args.config)
    else:
        configurations = get_ollama_configurations()
    
    print(f"Načteno {len(configurations)} konfigurací modelů.")
    
    # Projdeme všechny konfigurace a spustíme extrakci
    for i, config in enumerate(configurations, 1):
        print(f"\nKonfigurace {i}/{len(configurations)}")
        run_extraction(config, args)
    
    # Odstraníme dočasnou konfiguraci
    if TEMP_CONFIG_FILE.exists():
        TEMP_CONFIG_FILE.unlink()
    
    print("\nVšechny extrakce dokončeny.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram přerušen uživatelem.")
        sys.exit(1) 