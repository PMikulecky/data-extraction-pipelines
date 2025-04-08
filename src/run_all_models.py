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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
        # Konfigurace 1: Gemma 3 + Llama 3.2 Vision
        {
            "name": "gemma3-llama3.2-vision",
            "text": {
                "provider": "ollama",
                "model": "gemma3:4b"
            },
            "vision": {
                "provider": "ollama",
                "model": "llama3.2-vision:11b"
            },
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text:latest"
            }
        },
        # Konfigurace 2: Qwen 2.5 + MiniCPM
        {
            "name": "qwen2.5-minicpm",
            "text": {
                "provider": "ollama",
                "model": "qwen2.5:7b"
            },
            "vision": {
                "provider": "ollama",
                "model": "minicpm-v:8b"
            },
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text:latest"
            }
        },
        # Konfigurace 3: Llama 3.2 + Granite
        {
            "name": "llama3.2-granite",
            "text": {
                "provider": "ollama",
                "model": "llama3.2:3b"
            },
            "vision": {
                "provider": "ollama",
                "model": "granite3.2-vision:2b"
            },
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text:latest"
            }
        },
        # Konfigurace 4: Llama 3.1 + LLaVA
        {
            "name": "llama3.1-llava",
            "text": {
                "provider": "ollama",
                "model": "llama3.1:8b"
            },
            "vision": {
                "provider": "ollama",
                "model": "llava-llama3:8b"
            },
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text:latest"
            }
        },
        # Konfigurace 5: Phi 4 + Llama 3.2 Vision
        {
            "name": "phi4-llama3.2-vision",
            "text": {
                "provider": "ollama",
                "model": "phi4:14b"
            },
            "vision": {
                "provider": "ollama",
                "model": "llama3.2-vision:11b"
            },
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text:latest"
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


def run_extraction(config: Dict[str, Any], args, processed_models: Dict[str, str]) -> None:
    """
    Spustí extrakci metadat s danou konfigurací.
    
    Args:
        config (Dict[str, Any]): Konfigurace modelu
        args: Argumenty příkazové řádky
        processed_models (Dict[str, str]): Slovník již zpracovaných modelů a cest k jejich výsledkům
    """
    config_name = config.get("name", "unknown_config")
    
    # Získání identifikátorů jednotlivých modelů
    text_model_id = f"{config['text']['provider']}_{config['text']['model']}"
    vision_model_id = f"{config['vision']['provider']}_{config['vision']['model']}"
    embedding_model_id = f"{config['embedding']['provider']}_{config['embedding']['model']}"
    
    # Kontrola, zda byly všechny modely v konfiguraci již zpracovány
    text_already_processed = text_model_id in processed_models
    vision_already_processed = vision_model_id in processed_models
    embedding_already_processed = embedding_model_id in processed_models
    
    # Pokud byly všechny modely již zpracovány, můžeme přeskočit extrakci
    if text_already_processed and vision_already_processed and embedding_already_processed:
        print(f"\n\n=== Přeskakuji extrakci pro konfiguraci: {config_name} ===")
        print("Všechny modely již byly zpracovány v předchozích bězích.")
        print(f"Text model: {config['text']['provider']} - {config['text']['model']} (již zpracováno)")
        print(f"Vision model: {config['vision']['provider']} - {config['vision']['model']} (již zpracováno)")
        print(f"Embedding model: {config['embedding']['provider']} - {config['embedding']['model']} (již zpracováno)")
        
        # Připravíme výstupní adresář
        result_dir = prepare_result_directory(config_name)
        result_dir_path = Path(result_dir)
        
        # Zkopírujeme existující výsledky do nového adresáře
        print(f"Kopíruji existující výsledky...")
        
        # Uložíme konfiguraci do adresáře s výsledky
        with open(result_dir_path / "used_config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # Spojujeme výsledky modelů do jednoho výstupu
        combine_existing_results(
            processed_models[text_model_id],
            processed_models[vision_model_id],
            processed_models[embedding_model_id],
            result_dir_path
        )
        
        return
    
    print(f"\n\n=== Spouštím extrakci s konfigurací: {config_name} ===")
    print(f"Text model: {config['text']['provider']} - {config['text']['model']}" + 
          (" (již zpracováno)" if text_already_processed else ""))
    print(f"Vision model: {config['vision']['provider']} - {config['vision']['model']}" + 
          (" (již zpracováno)" if vision_already_processed else ""))
    print(f"Embedding model: {config['embedding']['provider']} - {config['embedding']['model']}" + 
          (" (již zpracováno)" if embedding_already_processed else ""))
    
    # Uložíme dočasnou konfiguraci
    save_temp_config(config)
    
    # Načteme konfiguraci do globální instance
    from models.config.model_config import load_config
    load_config(str(TEMP_CONFIG_FILE))
    
    # Připravíme výstupní adresář
    result_dir = prepare_result_directory(config_name)
    result_dir_path = Path(result_dir)
    
    # Vymažeme předchozí výsledky, abychom vynutili novou extrakci
    print(f"Mažu předchozí výsledky pro vynucení nové extrakce...")
    result_files_to_delete = [
        RESULTS_DIR / "embedded_results.json",
        RESULTS_DIR / "vlm_results.json",
        RESULTS_DIR / "embedded_comparison.json",
        RESULTS_DIR / "vlm_comparison.json",
        RESULTS_DIR / "semantic_comparison_results.json",
        RESULTS_DIR / "embedded_comparison_semantic.json",
        RESULTS_DIR / "vlm_comparison_semantic.json"
    ]
    
    for file_path in result_files_to_delete:
        if file_path.exists():
            print(f"  Mažu {file_path.name}")
            file_path.unlink()
    
    # Připravíme argumenty pro main.py
    cmd_args = [sys.executable, str(MAIN_SCRIPT)]
    
    # Přidáme cestu ke konfiguraci
    cmd_args.extend(["--config", str(TEMP_CONFIG_FILE)])
    
    # Přidáme modely
    cmd_args.extend(["--models", "embedded", "vlm"])
    
    # Vždy vynucujeme novou extrakci
    cmd_args.append("--force-extraction")
    
    # Přidáme další argumenty
    if args.limit:
        cmd_args.extend(["--limit", str(args.limit)])
    if args.year_filter:
        cmd_args.extend(["--year-filter"] + [str(year) for year in args.year_filter])
    if args.skip_download:
        cmd_args.append("--skip-download")
    if args.skip_semantic:
        cmd_args.append("--skip-semantic")
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
        
        # Aktualizujeme slovník zpracovaných modelů
        processed_models[text_model_id] = str(result_dir_path)
        processed_models[vision_model_id] = str(result_dir_path)
        processed_models[embedding_model_id] = str(result_dir_path)
            
    except subprocess.CalledProcessError as e:
        print(f"Chyba při spuštění extrakce: {e}")
    except KeyboardInterrupt:
        print("\nExtrakce přerušena uživatelem.")
        sys.exit(1)


def combine_existing_results(text_model_dir: str, vision_model_dir: str, embedding_model_dir: str, output_dir: Path) -> None:
    """
    Kombinuje existující výsledky z jednotlivých modelů.
    
    Args:
        text_model_dir (str): Cesta k adresáři s výsledky text modelu
        vision_model_dir (str): Cesta k adresáři s výsledky vision modelu
        embedding_model_dir (str): Cesta k adresáři s výsledky embedding modelu
        output_dir (Path): Cesta k výstupnímu adresáři
    """
    # Seznam souborů, které chceme zkopírovat
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
    
    # Zjistíme, který adresář obsahuje které soubory
    # Preferujeme nejnovější výsledky, pokud existují
    for filename in result_files:
        copied = False
        
        # Zkontrolujeme nejprve text model
        source_file = Path(text_model_dir) / filename
        if source_file.exists():
            print(f"Kopíruji {filename} z výsledků text modelu...")
            shutil.copy(source_file, output_dir / filename)
            copied = True
        
        # Pokud ne, zkusíme vision model
        if not copied:
            source_file = Path(vision_model_dir) / filename
            if source_file.exists():
                print(f"Kopíruji {filename} z výsledků vision modelu...")
                shutil.copy(source_file, output_dir / filename)
                copied = True
        
        # Pokud ne, zkusíme embedding model
        if not copied:
            source_file = Path(embedding_model_dir) / filename
            if source_file.exists():
                print(f"Kopíruji {filename} z výsledků embedding modelu...")
                shutil.copy(source_file, output_dir / filename)


def create_final_comparison(result_dirs: List[str], is_final: bool = False) -> None:
    """
    Vytvoří souhrnné porovnání všech modelů.
    
    Args:
        result_dirs (List[str]): Seznam cest k adresářům s výsledky jednotlivých běhů
        is_final (bool): Zda jde o finální porovnání (pro účely výpisu zpráv)
    """
    if not is_final:
        print("\n=== Aktualizuji průběžné porovnání modelů ===")
    else:
        print("\n=== Vytvářím finální porovnání všech modelů ===")
    
    # Vytvoření výstupního adresáře pro souhrnné výsledky
    summary_dir = RESULTS_DIR / "final_comparison"
    summary_dir.mkdir(exist_ok=True)
    
    # Načtení dat ze všech běhů
    all_results = {}
    print(f"Prohledávám {len(result_dirs)} adresářů s výsledky...")
    
    for result_dir in result_dirs:
        # Načtení konfigurace
        config_path = Path(result_dir) / "used_config.json"
        if not config_path.exists():
            print(f"  Přeskakuji adresář {result_dir} - chybí konfigurační soubor")
            continue
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # Načtení výsledků - kontrolujeme více možných míst
            comparison_files = [
                Path(result_dir) / "semantic_comparison_results.json",
                Path(result_dir) / "embedded_comparison_semantic.json",
                Path(result_dir) / "vlm_comparison_semantic.json",
                Path(result_dir) / "embedded_comparison.json",
                Path(result_dir) / "vlm_comparison.json"
            ]
            
            found_results = False
            for comparison_path in comparison_files:
                if comparison_path.exists():
                    print(f"  Načítám výsledky z {comparison_path}")
                    with open(comparison_path, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    
                    # Identifikace modelu z názvu adresáře a názvu souboru
                    dir_name = Path(result_dir).name
                    file_name = comparison_path.name
                    
                    # Zjištění názvu modelu
                    if "vlm" in file_name.lower():
                        model_prefix = "VLM"
                    elif "embedded" in file_name.lower():
                        model_prefix = "EMBEDDED"
                    else:
                        # Pokud nejde identifikovat z názvu souboru, použijeme název adresáře
                        model_prefix = dir_name.split('_')[0].upper()
                    
                    # Kompletní název modelu - kombinace názvu adresáře a typu modelu
                    model_name = f"{model_prefix}-{dir_name.split('_')[0]}"
                    
                    # Ověříme strukturu dat
                    if 'metrics' in results:
                        all_results[model_name] = results
                        found_results = True
                        print(f"  Úspěšně načten model: {model_name}")
                    else:
                        print(f"  Neplatná struktura dat v souboru {comparison_path}")
            
            if not found_results:
                print(f"  V adresáři {result_dir} nebyly nalezeny žádné validní výsledky")
        
        except Exception as e:
            print(f"  Chyba při zpracování adresáře {result_dir}: {e}")
    
    if not all_results:
        print("Nenalezeny žádné výsledky pro porovnání.")
        return
        
    print(f"Nalezeno {len(all_results)} modelů pro porovnání.")
    
    # Vytvoření DataFrame pro každý typ metadat
    metadata_fields = ['title', 'authors', 'abstract', 'keywords', 'doi', 'year', 
                      'journal', 'volume', 'issue', 'pages', 'publisher', 'references']
    
    # Příprava dat pro tabulky
    field_results = {field: [] for field in metadata_fields}
    overall_results = []
    
    for model_name, results in all_results.items():
        # Pro každé pole metadat
        for field in metadata_fields:
            if field in results.get('metrics', {}):
                field_results[field].append({
                    'Model': model_name,
                    'Úspěšnost': results['metrics'][field]['mean'],
                    'Medián': results['metrics'][field]['median'],
                    'Min': results['metrics'][field]['min'],
                    'Max': results['metrics'][field]['max'],
                    'Počet': results['metrics'][field]['count']
                })
        
        # Celkové výsledky
        if 'metrics' in results and 'overall' in results['metrics']:
            overall_results.append({
                'Model': model_name,
                'Celková úspěšnost': results['metrics']['overall']['mean'],
                'Medián': results['metrics']['overall']['median'],
                'Min': results['metrics']['overall']['min'],
                'Max': results['metrics']['overall']['max'],
                'Počet': results['metrics']['overall']['count']
            })
    
    # Počítadlo vytvořených souborů
    created_files = []
    
    # Uložení tabulek pro jednotlivá pole
    for field, data in field_results.items():
        if data:
            df = pd.DataFrame(data)
            output_file = summary_dir / f"{field}_comparison.csv"
            df.to_csv(output_file, index=False)
            created_files.append(output_file.name)
            
            # Vytvoření grafu pro pole
            plt.figure(figsize=(10, 6))
            bars = plt.bar(df['Model'], df['Úspěšnost'])
            
            # Přidání hodnot nad sloupce
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2%}',
                        ha='center', va='bottom')
            
            plt.title(f'Porovnání úspěšnosti modelů - {field}')
            plt.xlabel('Model')
            plt.ylabel('Úspěšnost')
            plt.ylim(0, 1.05)
            plt.xticks(rotation=45)
            plt.tight_layout()
            output_file = summary_dir / f"{field}_comparison.png"
            plt.savefig(output_file)
            created_files.append(output_file.name)
            plt.close()
    
    # Uložení celkových výsledků
    df_overall = None
    if overall_results:
        df_overall = pd.DataFrame(overall_results)
        output_file = summary_dir / "overall_comparison.csv"
        df_overall.to_csv(output_file, index=False)
        created_files.append(output_file.name)
        
        # Vytvoření grafu celkových výsledků
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_overall['Model'], df_overall['Celková úspěšnost'])
        
        # Přidání hodnot nad sloupce
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        
        plt.title('Celková úspěšnost modelů')
        plt.xlabel('Model')
        plt.ylabel('Úspěšnost')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_file = summary_dir / "overall_comparison.png"
        plt.savefig(output_file)
        created_files.append(output_file.name)
        plt.close()
    
    if is_final:
        print(f"\nFinální souhrnné výsledky byly uloženy do adresáře: {summary_dir}")
        print(f"Vytvořeno {len(created_files)} souborů.")
        print("\nVytvořené soubory:")
        print("1. CSV soubory s detailními výsledky:")
        for field in metadata_fields:
            if f"{field}_comparison.csv" in created_files:
                print(f"   - {field}_comparison.csv")
        if "overall_comparison.csv" in created_files:
            print("   - overall_comparison.csv")
        print("\n2. Grafy porovnání:")
        for field in metadata_fields:
            if f"{field}_comparison.png" in created_files:
                print(f"   - {field}_comparison.png")
        if "overall_comparison.png" in created_files:
            print("   - overall_comparison.png")
    else:
        print(f"\nPrůběžné výsledky byly aktualizovány v adresáři: {summary_dir}")
        print(f"Vytvořeno/aktualizováno {len(created_files)} souborů.")
        if df_overall is not None and len(all_results) > 0:
            # Výpis aktuálního pořadí modelů
            print("\nAktuální pořadí modelů:")
            df_overall_sorted = df_overall.sort_values('Celková úspěšnost', ascending=False)
            for i, (_, row) in enumerate(df_overall_sorted.iterrows(), 1):
                print(f"{i}. {row['Model']}: {row['Celková úspěšnost']:.2%}")


def main():
    """Hlavní funkce skriptu."""
    args = parse_args()
    
    # Načteme konfigurace
    if args.config:
        # Použijeme explicitně zadaný konfigurační soubor
        configurations = load_configurations_from_file(args.config)
    else:
        # Použijeme výchozí model_configs.json
        default_config = CONFIG_DIR / "model_configs.json"
        if default_config.exists():
            print(f"Používám výchozí konfigurační soubor: {default_config}")
            configurations = load_configurations_from_file(str(default_config))
        else:
            print(f"Výchozí konfigurační soubor {default_config} nenalezen, používám zabudované konfigurace.")
            configurations = get_ollama_configurations()
    
    print(f"Načteno {len(configurations)} konfigurací modelů.")
    
    # Seznam adresářů s výsledky
    result_dirs = []
    
    # Slovník pro sledování již zpracovaných modelů
    # Klíč: identifikátor modelu (provider_model), Hodnota: cesta k adresáři s výsledky
    processed_models = {}
    
    # Projdeme všechny konfigurace a spustíme extrakci
    for i, config in enumerate(configurations, 1):
        print(f"\nKonfigurace {i}/{len(configurations)}")
        result_dir = prepare_result_directory(config.get("name", "unknown_config"))
        result_dirs.append(result_dir)
        run_extraction(config, args, processed_models)
        
        # Aktualizace průběžných statistik po každém běhu
        create_final_comparison(result_dirs, is_final=False)
    
    # Odstraníme dočasnou konfiguraci
    if TEMP_CONFIG_FILE.exists():
        TEMP_CONFIG_FILE.unlink()
    
    print("\nVšechny extrakce dokončeny.")
    
    # Vytvoření finálního souhrnného porovnání
    create_final_comparison(result_dirs, is_final=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram přerušen uživatelem.")
        sys.exit(1) 