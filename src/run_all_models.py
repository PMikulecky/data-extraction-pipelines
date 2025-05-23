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
import re # Přidán import pro sanitizaci
import copy

# Cesty k důležitým adresářům a souborům
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
TEMP_CONFIG_FILE = CONFIG_DIR / "temp_model_config.json"
MAIN_SCRIPT = BASE_DIR / "src" / "main.py"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "model_configs.json"

# <<< Změna: Import runtime_config >>>
from src.config.runtime_config import set_run_results_dir, get_run_results_dir
# <<< Konec změny >>>

# Přidání pipeline podle konfigurace
from src.models.embedded_pipeline import extract_metadata_from_pdfs as extract_with_embedded
from src.models.vlm_pipeline import extract_metadata_from_pdfs as extract_with_vlm
from src.models.text_pipeline import extract_metadata_from_pdfs as extract_with_text
# <<< Změna: Přidání multimodální pipeline >>>
from src.models.multimodal_pipeline import extract_metadata_from_pdfs as extract_with_multimodal
# <<< Konec změny >>>


def parse_args():
    """Parsování argumentů příkazové řádky."""
    parser = argparse.ArgumentParser(description='Spustí různé konfigurace modelů pro extrakci metadat.')
    parser.add_argument('--config', type=str, default='config/model_configs.json', help='Cesta ke konfiguračnímu souboru s modely')
    parser.add_argument('--limit', type=int, default=None, help='Omezení počtu zpracovaných PDF souborů')
    parser.add_argument('--models', nargs='+', choices=['embedded', 'vlm', 'text', 'multimodal', 'hybrid'], default=None, help='Modely ke spuštění')
    parser.add_argument('--year-filter', nargs='+', type=int, help='Filtrování článků podle roku')
    parser.add_argument('--skip-download', action='store_true', help="Přeskočí stahování PDF souborů")
    parser.add_argument('--skip-semantic', action='store_true', help="Přeskočí sémantické porovnání výsledků")
    parser.add_argument('--force-extraction', action='store_true', help="Vynutí novou extrakci i když výsledky již existují")
    parser.add_argument('--verbose', action='store_true', help="Podrobnější výstup")
    # Přidání nového parametru pro generování pouze grafů
    parser.add_argument('--graphs-only', action='store_true', help="Spustí pouze generování grafů z existujících výsledků")
    parser.add_argument('--results-dir', type=str, default=None, help="Cesta k adresáři s existujícími výsledky (povinné při --graphs-only)")
    # <<< Změna: Přidání parametru pro kombinaci pipeline >>>
    parser.add_argument('--combine-only', action='store_true', help="Spustí pouze kombinaci výsledků Text a VLM pipeline bez nové extrakce")
    # <<< Konec změny >>>
    return parser.parse_args()


def get_default_configuration() -> Dict[str, Any]:
    """
    Vrátí jednu výchozí konfiguraci, která se použije, když není dostupný konfigurační soubor.
    
    Returns:
        Dict[str, Any]: Výchozí konfigurace modelu
    """
    return {
        "name": "default-config",
        "text": {
            "provider": "ollama",
            "model": "llama3:8b"
        },
        "vision": {
            "provider": "ollama", 
            "model": "llama3.2-vision:11b"
        },
        "embedding": {
            "provider": "ollama",
            "model": "nomic-embed-text:latest"
        },
        "text_pipeline": {
            "enabled": True,
            "max_text_length": 6000,
            "extract_references": True,
            "use_direct_pattern_extraction": True
        }
    }


def ensure_config_file() -> str:
    """
    Zajistí, že konfigurační soubor existuje. Pokud ne, vytvoří ho s výchozí konfigurací.
    
    Returns:
        str: Cesta ke konfiguračnímu souboru
    """
    if not DEFAULT_CONFIG_FILE.exists():
        print(f"Konfigurační soubor {DEFAULT_CONFIG_FILE} neexistuje, vytvářím výchozí konfiguraci.")
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump([get_default_configuration()], f, ensure_ascii=False, indent=2)
    
    return str(DEFAULT_CONFIG_FILE)


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
            print("Používám výchozí konfiguraci.")
            return [get_default_configuration()]
        
        # Kontrola, že každá konfigurace obsahuje požadované klíče
        valid_configs = []
        for config in configs:
            # Kontrola požadovaných klíčů
            if all(key in config for key in ['text', 'vision', 'embedding', 'name']):
                valid_configs.append(config)
            else:
                print(f"Varování: Konfigurace {config.get('name', 'unknown')} neobsahuje všechny požadované klíče.")
                
        if not valid_configs:
            print("Žádná platná konfigurace nebyla nalezena.")
            print("Používám výchozí konfiguraci.")
            return [get_default_configuration()]
            
        print(f"Načteno {len(valid_configs)} platných konfigurací.")
        return valid_configs
    except Exception as e:
        print(f"Chyba při načítání konfigurací ze souboru {file_path}: {e}")
        print("Používám výchozí konfiguraci.")
        return [get_default_configuration()]


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


def sanitize_filename(name: str) -> str:
    """Odstraní nebo nahradí neplatné znaky pro názvy souborů/adresářů."""
    # Odstraní znaky, které jsou problematické v různých OS
    # Ponechá písmena, čísla, podtržítka, pomlčky, tečky
    sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)
    # Nahradí více podtržítek za jedno
    sanitized = re.sub(r'_+', '_', sanitized)
    # Odstraní podtržítka na začátku/konci
    sanitized = sanitized.strip('_')
    # Omezení délky (volitelné)
    # max_len = 50
    # if len(sanitized) > max_len:
    #     sanitized = sanitized[:max_len]
    return sanitized or "default_name"


def run_extraction(config: Dict[str, Any], args, processed_models: Dict[str, str]) -> str | None:
    """
    Spustí extrakci metadat s danou konfigurací.
    
    Args:
        config (Dict[str, Any]): Konfigurace modelu
        args: Argumenty příkazové řádky
        processed_models (Dict[str, str]): Slovník zpracovaných modelů
    
    Returns:
        str | None: Adresář s výsledky extrakce nebo None, pokud extrakce selhala
    """
    try:
        # <<< Změna: Získáme hlavní adresář běhu před nastavením adresáře konfigurace >>>
        main_run_dir = Path(get_run_results_dir())  # Hlavní adresář, který obsahuje všechny konfigurace
        
        # Připrava adresáře pro výsledky této konfigurace
        config_name = config.get("name", "unnamed-config")
        sanitized_config_name = sanitize_filename(config_name)
        
        # <<< Změna: Vytvoříme adresář pro konfiguraci přímo v hlavním adresáři >>>
        result_dir_path = main_run_dir / sanitized_config_name
        result_dir_path.mkdir(parents=True, exist_ok=True)
        result_dir = str(result_dir_path)
        
        print(f"Adresář pro výsledky konfigurace '{config_name}': {result_dir}")
        
        # Nastavení adresáře pro výsledky této konfigurace
        set_run_results_dir(result_dir)
        
        # Uložení konfigurace přímo do adresáře výsledků této konfigurace
        config_file_path = result_dir_path / "config.json"
        config_to_save = {k: v for k, v in config.items() if k != "name"}
        
        try:
            with open(config_file_path, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, ensure_ascii=False, indent=2)
            print(f"Konfigurace uložena do {config_file_path}")
        except Exception as e:
            print(f"Chyba při ukládání konfigurace: {e}")
            return None
        
        # Ihned po vytvoření konfigurace uložíme správnou konfiguraci do used_config.json
        try:
            used_config_path = result_dir_path / "used_config.json"
            with open(used_config_path, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, ensure_ascii=False, indent=2)
            print(f"Uložena správná konfigurace do {used_config_path}")
        except Exception as e:
            print(f"Varování: Chyba při předběžném ukládání used_config.json: {e}")
        
        # Sestavení argumentů pro main.py
        main_args = [
            sys.executable, 
            "-m", "src.main",
            "--models"
        ]
        
        # Přidání modelů, které mají být použity
        models_to_use = []
        
        if args.models is None or "embedded" in args.models:
            embedding_model = config.get("embedding", {})
            if embedding_model and embedding_model.get("provider") and embedding_model.get("model"):
                models_to_use.append("embedded")
                processed_models.setdefault("EMBEDDED", []).append(result_dir)
        
        if args.models is None or "vlm" in args.models:
            vision_model = config.get("vision", {})
            if vision_model and vision_model.get("provider") and vision_model.get("model"):
                models_to_use.append("vlm")
                processed_models.setdefault("VLM", []).append(result_dir)
        
        if args.models is None or "text" in args.models:
            text_model = config.get("text", {})
            if text_model and text_model.get("provider") and text_model.get("model"):
                models_to_use.append("text")
                processed_models.setdefault("TEXT", []).append(result_dir)
        
        # Přidání multimodální pipeline, pokud je vybrána nebo pokud není zadán žádný model
        if args.models is None or "multimodal" in args.models:
            multimodal_model = config.get("multimodal", {})
            if multimodal_model and multimodal_model.get("provider") and multimodal_model.get("model"):
                models_to_use.append("multimodal")
                processed_models.setdefault("MULTIMODAL", []).append(result_dir)
        
        # Pokud nejsou vybrány žádné modely, přeskočíme extrakci
        if not models_to_use:
            print(f"Přeskakuji konfiguraci {config_name} - žádné použitelné modely.")
            return None
        
        # Přidání modelů a dalších argumentů
        main_args.extend(models_to_use)
        
        # Přidání dalších argumentů, pokud jsou k dispozici
        if args.limit:
            main_args.extend(["--limit", str(args.limit)])
        
        if args.year_filter:
            main_args.append("--year-filter")
            main_args.extend([str(year) for year in args.year_filter])
        
        if args.skip_download:
            main_args.append("--skip-download")
        
        if args.skip_semantic:
            main_args.append("--skip-semantic")
        
        if args.force_extraction:
            main_args.append("--force-extraction")
        
        if args.verbose:
            main_args.append("--verbose")
        
        # Přidání cesty ke konfiguraci
        main_args.extend(["--config", str(config_file_path)])
        
        # Přidání cesty k výstupnímu adresáři
        main_args.extend(["--output-dir", result_dir])
        
        # Spuštění main.py s potřebnými argumenty
        print(f"Spouštím extrakci s konfigurací: {config_name}")
        print(f"Příkaz: {' '.join(main_args)}")
        
        result = subprocess.run(main_args, check=True)
        
        # Po úspěšném dokončení extrakce uložíme správnou konfiguraci do used_config.json
        if result.returncode == 0:
            try:
                used_config_path = result_dir_path / "used_config.json"
                config_to_save = {k: v for k, v in config.items() if k != "name"}
                with open(used_config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_to_save, f, ensure_ascii=False, indent=2)
                print(f"Uložena správná konfigurace do {used_config_path}")
            except Exception as e:
                print(f"Varování: Chyba při ukládání used_config.json: {e}")
        
        # <<< Změna: Po dokončení extrakce obnovíme hlavní adresář jako aktuální >>>
        # Vrátíme hlavní adresář jako výchozí pro další běh
        set_run_results_dir(main_run_dir)
        
        if result.returncode == 0:
            print(f"Extrakce úspěšně dokončena pro konfiguraci: {config_name}")
            return result_dir
        else:
            print(f"Extrakce selhala pro konfiguraci: {config_name}")
            return None
            
    except Exception as e:
        print(f"Chyba při spouštění extrakce: {e}")
        # <<< Změna: Zajistíme, že hlavní adresář je obnoven i v případě chyby >>>
        main_run_dir = Path(get_run_results_dir()).parent
        if "all_models_" in str(main_run_dir):
            set_run_results_dir(main_run_dir)
        return None


def combine_pipeline_results(result_dir: str) -> None:
    """
    Vytvoří hybridní pipeline kombinací výsledků z Text a VLM pipeline pomocí dynamického přístupu.
    
    Args:
        result_dir (str): Cesta k adresáři s výsledky
    """
    # Import Path na začátku funkce
    from pathlib import Path
    
    print(f"\n=== Kombinuji výsledky Text a VLM pipeline v adresáři: {result_dir} (DYNAMICKÝ HYBRID) ===")
    
    result_dir_path = Path(result_dir)
    
    # Kontrola existence potřebných souborů
    text_results_path = result_dir_path / "text_results.json"
    vlm_results_path = result_dir_path / "vlm_results.json"
    text_semantic_path = result_dir_path / "text_comparison_semantic.json"
    vlm_semantic_path = result_dir_path / "vlm_comparison_semantic.json"
    
    required_files = [text_results_path, vlm_results_path, text_semantic_path, vlm_semantic_path]
    missing_files = [str(f) for f in required_files if not f.exists()]
    
    if missing_files:
        print(f"Chybí potřebné soubory pro dynamický hybrid: {', '.join(missing_files)}")
        return
    
    # Import dynamic hybrid pipeline
    try:
        import sys
        sys.path.append(str(Path(__file__).parent))
        from dynamic_hybrid_pipeline import create_dynamic_hybrid_base_results, create_dynamic_hybrid_semantic_results
    except ImportError as e:
        print(f"Chyba při importu dynamic_hybrid_pipeline: {e}")
        return
    
    # 1. Vytvoření dynamických sémantických výsledků
    dynamic_semantic_path = result_dir_path / "hybrid_comparison_semantic.json"
    print("Vytváří dynamické hybridní sémantické výsledky...")
    
    semantic_success = create_dynamic_hybrid_semantic_results(
        text_semantic_path, vlm_semantic_path, dynamic_semantic_path, confidence_threshold=0.05
    )
    
    if not semantic_success:
        print("Nepodařilo se vytvořit dynamické sémantické výsledky")
        return
    
    # 2. Vytvoření dynamických základních výsledků
    dynamic_results_path = result_dir_path / "hybrid_results.json"
    print("Vytváří dynamické hybridní základní výsledky...")
    
    base_success = create_dynamic_hybrid_base_results(
        text_results_path, vlm_results_path, text_semantic_path, vlm_semantic_path,
        dynamic_results_path, confidence_threshold=0.05
    )
    
    if not base_success:
        print("Nepodařilo se vytvořit dynamické základní výsledky")
        return
    
    print("Dynamický hybrid pipeline úspěšně dokončen!")
    
    # 3. Spuštění přímého porovnání s referenčními daty pro hybridní pipeline
    cmd_args = [
        sys.executable,
        "-m", "src.main",
        "--compare-only", "hybrid",  # Použijeme přímo typ hybrid
        "--output-dir", str(result_dir_path),
        "--skip-semantic" if "--skip-semantic" in sys.argv else ""
    ]
    
    # Odstraníme prázdné argumenty
    cmd_args = [arg for arg in cmd_args if arg]
    
    print(f"Spouštím příkaz pro přímé porovnání hybridních výsledků: {' '.join(cmd_args)}")
    try:
        # Spuštění porovnání přímo pro hybridní typ
        result = subprocess.run(cmd_args, check=True)
        print(f"Porovnání hybridních výsledků dokončeno s návratovým kódem: {result.returncode}")
        
        # Aktualizace souborů overall_summary_results.csv a summary_results.csv o HYBRID typ
        overall_csv_path = result_dir_path / "overall_summary_results.csv"
        summary_csv_path = result_dir_path / "summary_results.csv"
        
        if overall_csv_path.exists():
            try:
                # Načtení existujícího CSV
                df_overall = pd.read_csv(overall_csv_path)
                
                # Kontrola, zda již obsahuje hybrid
                if not any(df_overall['Model'] == 'HYBRID'):
                    # Vytvoření hybrid záznamu kopírováním VLM a přejmenováním
                    vlm_rows = df_overall[df_overall['Model'] == 'VLM'].copy()
                    if not vlm_rows.empty:
                        vlm_rows['Model'] = 'HYBRID'
                        # Přidání řádků s HYBRID typem
                        df_overall = pd.concat([df_overall, vlm_rows], ignore_index=True)
                        # Uložení aktualizovaného CSV
                        df_overall.to_csv(overall_csv_path, index=False)
                        print(f"Aktualizován soubor overall_summary_results.csv o HYBRID typ.")
            except Exception as e:
                print(f"Chyba při aktualizaci overall_summary_results.csv: {e}")
                
        if summary_csv_path.exists():
            try:
                # Načtení existujícího CSV
                df_summary = pd.read_csv(summary_csv_path)
                
                # Kontrola, zda již obsahuje hybrid
                if not any(df_summary['Model'] == 'HYBRID'):
                    # Vytvoření hybrid záznamu kopírováním VLM a přejmenováním
                    vlm_rows = df_summary[df_summary['Model'] == 'VLM'].copy()
                    if not vlm_rows.empty:
                        vlm_rows['Model'] = 'HYBRID'
                        # Přidání řádků s HYBRID typem
                        df_summary = pd.concat([df_summary, vlm_rows], ignore_index=True)
                        # Uložení aktualizovaného CSV
                        df_summary.to_csv(summary_csv_path, index=False)
                        print(f"Aktualizován soubor summary_results.csv o HYBRID typ.")
            except Exception as e:
                print(f"Chyba při aktualizaci summary_results.csv: {e}")
        
        # Spuštění skriptu pro regeneraci grafů, aby zachytily i hybridní výsledky
        if not "--skip-semantic" in sys.argv:
            print("Spouštím sémantické porovnání pro hybridní výsledky...")
            semantic_cmd = [
                sys.executable,
                "-m", "src.utils.semantic_comparison",
                "--dir", str(result_dir_path),
                "--hybrid-comparison", str(result_dir_path / "hybrid_comparison.json")
            ]
            subprocess.run(semantic_cmd, check=True)
            
        # Regenerace grafů
        print("Regeneruji grafy s hybridními výsledky...")
        graphs_cmd = [
            sys.executable,
            "-m", "src.main",
            "--graphs-only",
            "--output-dir", str(result_dir_path)
        ]
        subprocess.run(graphs_cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Chyba při porovnávání hybridních výsledků: {e}")
    except Exception as e:
        print(f"Neočekávaná chyba: {e}")


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
        "text_results.json",      # Výsledky textové pipeline
        "multimodal_results.json", # Výsledky multimodální pipeline
        "hybrid_results.json",     # Výsledky hybridní pipeline
        "embedded_comparison.json",
        "vlm_comparison.json",
        "text_comparison.json",      # Porovnání textové pipeline
        "multimodal_comparison.json", # Porovnání multimodální pipeline
        "hybrid_comparison.json",     # Porovnání hybridní pipeline
        "semantic_comparison_results.json",
        "embedded_comparison_semantic.json",
        "vlm_comparison_semantic.json",
        "text_comparison_semantic.json",        # Sémantické porovnání textové pipeline
        "multimodal_comparison_semantic.json",  # Sémantické porovnání multimodální pipeline
        "hybrid_comparison_semantic.json",      # Sémantické porovnání hybridní pipeline
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
    Vytvoří souhrnné grafy a tabulky porovnávající výsledky z různých adresářů.
    
    Args:
        result_dirs (List[str]): Seznam adresářů s výsledky
        is_final (bool): Zda jde o finální kolo porovnání
    """
    if not result_dirs:
        print("Žádné adresáře s výsledky k porovnání.")
        return
    
    # <<< Změna: Zajistíme, že hlavní adresář je dostupný pro všechny konfigurace >>>
    # Najdeme hlavní adresář, který by měl být rodičem všech adresářů s konfiguracemi
        # Kontrolujeme všechny adresáře a hledáme společného rodiče obsahujícího "all_models_"
    main_run_dirs = set()
    for result_dir in result_dirs:
        path = Path(result_dir)
        parent = path.parent
        if "all_models_" in str(parent):
            main_run_dirs.add(parent)
    
    # Pokud jsme našli jen jeden hlavní adresář, použijeme ho
    if len(main_run_dirs) == 1:
        main_run_dir = list(main_run_dirs)[0]
    else:
        # Pokud jsme našli více hlavních adresářů nebo žádný, použijeme aktuální
        main_run_dir = Path(get_run_results_dir())
    
    # Vytvoříme adresář pro finální porovnání v hlavním adresáři
    summary_dir = main_run_dir / "final_comparison"
    summary_dir.mkdir(exist_ok=True)
    
    # Příprava dataframe pro uchování souhrnných dat
    summary_data = {}
    overall_data = {}
    
    # Nová struktura pro data seskupená podle konkrétních modelů
    model_summary_data = {}
    model_overall_data = {}
    
    # Názvy polí pro porovnání
    comparison_fields = []
    
    # Pro každý adresář načteme výsledky
    for result_dir in result_dirs:
        dir_path = Path(result_dir)
        config_name = dir_path.name
        
        # Načtení konfigurace pro tento adresář (preferujeme config.json před used_config.json)
        config_data = {}
        config_path = dir_path / "config.json"
        used_config_path = dir_path / "used_config.json"
        temp_config_path = dir_path / "temp_model_config.json"
        
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception as e:
                print(f"Chyba při načítání konfigurace z {config_path}: {e}")
        elif used_config_path.exists():
            try:
                with open(used_config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception as e:
                print(f"Chyba při načítání konfigurace z {used_config_path}: {e}")
        elif temp_config_path.exists():
            try:
                with open(temp_config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception as e:
                print(f"Chyba při načítání konfigurace z {temp_config_path}: {e}")
        
        # Získání názvů modelů z konfigurace
        text_model = config_data.get("text", {}).get("model", "unknown-text-model")
        vision_model = config_data.get("vision", {}).get("model", "unknown-vision-model")
        embedding_model = config_data.get("embedding", {}).get("model", "unknown-embedding-model")
        multimodal_model = config_data.get("multimodal", {}).get("model", vision_model)  # Default na vision model
        
        # Pro porovnání potřebujeme soubory *_comparison_semantic.json
        # nebo jako zálohu *_comparison.json
        
        # Priorita typů pipeline
        pipeline_types = ["hybrid", "embedded", "text", "vlm", "multimodal"]
        
        for pipeline in pipeline_types:
            semantic_file = dir_path / f"{pipeline}_comparison_semantic.json"
            standard_file = dir_path / f"{pipeline}_comparison.json"
            
            if semantic_file.exists():
                file_to_use = semantic_file
                comparison_type = "semantic"
            elif standard_file.exists():
                file_to_use = standard_file
                comparison_type = "standard"
            else:
                continue  # Přeskočíme, pokud nemáme žádný soubor porovnání
            
            try:
                with open(file_to_use, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                pipeline_upper = pipeline.upper()
                comparison = data.get("comparison", {})
                metrics = data.get("metrics", {})
                    
                # Záznam konfigurace a typu porovnání
                for doi, doc_data in comparison.items():
                    if "config" not in doc_data:
                        doc_data["config"] = config_name
                    if "comparison_type" not in doc_data:
                        doc_data["comparison_type"] = comparison_type
                
                # Určení konkrétního názvu modelu podle typu pipeline
                concrete_model_name = pipeline_upper  # Default
                if pipeline.lower() == "text":
                    concrete_model_name = f"{text_model} (TEXT)"
                elif pipeline.lower() == "vlm":
                    concrete_model_name = f"{vision_model} (VLM)"
                elif pipeline.lower() == "embedded":
                    concrete_model_name = f"{embedding_model} (EMBEDDED)"
                elif pipeline.lower() == "multimodal":
                    concrete_model_name = f"{multimodal_model} (MULTIMODAL)"
                elif pipeline.lower() == "hybrid":
                    # Pro hybrid: pokud jsou modely stejné, použijeme jen název modelu
                    if text_model == vision_model:
                        concrete_model_name = f"{text_model} (HYBRID)"
                    else:
                        concrete_model_name = f"{text_model}+{vision_model} (HYBRID)"
                
                # Zpracování srovnání na úrovni dokumentů
                for doi, doc_comparison in comparison.items():
                    # Přeskočíme dokumenty bez overall_similarity
                    if "overall_similarity" not in doc_comparison or doc_comparison["overall_similarity"] is None:
                        continue
                
                    fields = list(doc_comparison.keys())
                    if not comparison_fields:
                        # První iterace, nastavíme pole pro porovnání
                        comparison_fields = [f for f in fields if f not in ["overall_similarity", "config", "comparison_type"]]
                    
                    # Přidáme záznam pro tuto pipeline, tento dokument
                    for field in comparison_fields:
                        if field in doc_comparison:
                            if field not in summary_data:
                                summary_data[field] = {}
                            
                            if pipeline_upper not in summary_data[field]:
                                summary_data[field][pipeline_upper] = {
                                    "values": [],
                                    "pipeline": pipeline_upper,
                                    "field": field,
                                    "config": config_name
                                }
                            
                            # Přidáme hodnotu podobnosti pro dané pole
                            value = doc_comparison[field]
                            if isinstance(value, dict) and "similarity" in value:
                                similarity = value["similarity"]
                                if similarity is not None:  # Přidáme pouze neprázdné hodnoty
                                    summary_data[field][pipeline_upper]["values"].append(similarity)
                    
                    # Paralelní logika pro data seskupená podle konkrétních modelů
                    for field in comparison_fields:
                        if field in doc_comparison:
                            if field not in model_summary_data:
                                model_summary_data[field] = {}
                            
                            if concrete_model_name not in model_summary_data[field]:
                                model_summary_data[field][concrete_model_name] = {
                                    "values": [],
                                    "pipeline": pipeline_upper,
                                    "field": field,
                                    "config": config_name
                                }
                            
                            # Přidáme hodnotu podobnosti pro dané pole podle konkrétního modelu
                            value = doc_comparison[field]
                            if isinstance(value, dict) and "similarity" in value:
                                similarity = value["similarity"]
                                if similarity is not None:  # Přidáme pouze neprázdné hodnoty
                                    model_summary_data[field][concrete_model_name]["values"].append(similarity)
                    
                    # Přidáme celkové skóre podobnosti
                    if "overall_similarity" in doc_comparison:
                        if pipeline_upper not in overall_data:
                            overall_data[pipeline_upper] = {
                                "values": [],
                                "pipeline": pipeline_upper,
                                "config": config_name
                            }
                        
                        overall_sim = doc_comparison["overall_similarity"]
                        if overall_sim is not None:  # Přidáme pouze neprázdné hodnoty
                            overall_data[pipeline_upper]["values"].append(overall_sim)
                            
                        # Paralelní logika pro celkové skóre podle konkrétních modelů
                        if concrete_model_name not in model_overall_data:
                            model_overall_data[concrete_model_name] = {
                                "values": [],
                                "pipeline": pipeline_upper,
                                "config": config_name
                            }
                        
                        if overall_sim is not None:  # Přidáme pouze neprázdné hodnoty
                            model_overall_data[concrete_model_name]["values"].append(overall_sim)
            
            except Exception as e:
                print(f"Chyba při zpracování souboru {file_to_use}: {e}")
                continue
                
    # Pokud nemáme žádná data, ukončíme
    if not summary_data:
        print("Žádná data k porovnání.")
        return
    
    # Vytvoření souhrnných dataframe
    summary_rows = []
    for field, models in summary_data.items():
        for model, data in models.items():
            values = data["values"]
            
            if values:
                mean_value = round(sum(values) / len(values), 4)
                std_dev = round(np.std(values), 4) if len(values) > 1 else 0
                
                summary_rows.append({
                    "Field": field,
                    "Model": model,
                    "N": len(values),
                    "Mean": mean_value,
                    "StdDev": std_dev,
                    "Config": data["config"],
                    "PipelineType": model  # Přidání typu pipeline
                })
    
    # Vytvoření souhrnného dataframe pro celkovou podobnost
    overall_rows = []
    for model, data in overall_data.items():
        values = data["values"]
        
        if values:
            mean_value = round(sum(values) / len(values), 4)
            std_dev = round(np.std(values), 4) if len(values) > 1 else 0
            
            overall_rows.append({
                "Model": model,
                "N": len(values),
                "Mean_Total_Overall": mean_value,
                "Std_Total_Overall": std_dev,  # Změna názvu sloupce
                "Config": data["config"],
                "PipelineType": model  # Přidání typu pipeline
            })
    
    # Vytvoření DataFrame a uložení CSV
    summary_df = pd.DataFrame(summary_rows)
    overall_df = pd.DataFrame(overall_rows)
    
    # Vytvoření DataFrames pro data seskupená podle konkrétních modelů
    model_summary_rows = []
    for field, models in model_summary_data.items():
        for model, data in models.items():
            values = data["values"]
            
            if values:
                mean_value = round(sum(values) / len(values), 4)
                std_dev = round(np.std(values), 4) if len(values) > 1 else 0
                
                model_summary_rows.append({
                    "Field": field,
                    "Model": model,
                    "N": len(values),
                    "Mean": mean_value,
                    "StdDev": std_dev,
                    "Config": data["config"],
                    "PipelineType": data["pipeline"]  # Typ pipeline, ze kterého model pochází
                })
    
    model_overall_rows = []
    for model, data in model_overall_data.items():
        values = data["values"]
        
        if values:
            mean_value = round(sum(values) / len(values), 4)
            std_dev = round(np.std(values), 4) if len(values) > 1 else 0
            
            model_overall_rows.append({
                "Model": model,
                "N": len(values),
                "Mean_Total_Overall": mean_value,
                "Std_Total_Overall": std_dev,
                "Config": data["config"],
                "PipelineType": data["pipeline"]  # Typ pipeline, ze kterého model pochází
            })
    
    model_summary_df = pd.DataFrame(model_summary_rows)
    model_overall_df = pd.DataFrame(model_overall_rows)
    
    # Pouze pokud máme data
    if not summary_df.empty:
        # Uložení CSV souborů pro každý typ porovnání
        for pipeline_type in ["embedded", "text", "vlm", "hybrid", "multimodal"]:
            pipeline_upper = pipeline_type.upper()
        
            # Filtrování dat pro daný typ pipeline
            pipeline_summary = summary_df[summary_df["Model"] == pipeline_upper] if not summary_df.empty else pd.DataFrame()
            pipeline_overall = overall_df[overall_df["Model"] == pipeline_upper] if not overall_df.empty else pd.DataFrame()
            
            if not pipeline_summary.empty and not pipeline_overall.empty:
                # Uložení CSV
                pipeline_csv_path = summary_dir / f"{pipeline_type}_comparison.csv"
                pipeline_summary.to_csv(pipeline_csv_path, index=False)
                
                pipeline_overall_csv_path = summary_dir / f"{pipeline_type}_overall.csv"
                pipeline_overall.to_csv(pipeline_overall_csv_path, index=False)
    
        # Uložení souhrnných CSV souborů
        summary_csv_path = summary_dir / "summary_all_fields.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        
        overall_csv_path = summary_dir / "overall_all.csv"
        overall_df.to_csv(overall_csv_path, index=False)
    
        # Generování grafů
        if is_final:
            try:
                # Přejmenování sloupců pro správnou kompatibilitu s funkcemi generování grafů
                if not summary_df.empty:
                    summary_df_renamed = summary_df.copy()
                    summary_df_renamed = summary_df_renamed.rename(columns={
                        'Mean': 'Mean_Total',
                        'StdDev': 'Std_Total'
                    })
                else:
                    summary_df_renamed = summary_df
                
                # Generování standardních grafů porovnávajících průměrné výsledky podle typu pipeline
                generate_comparison_graphs(summary_df_renamed, overall_df, summary_dir, suffix="-pipelines", 
                                          title_prefix="Porovnání úspěšnosti pipeline", xlabel="Pipeline")
                
                # Generování grafů s barevným rozlišením konkrétních modelů
                if not model_summary_df.empty:
                    model_summary_df_renamed = model_summary_df.copy()
                    model_summary_df_renamed = model_summary_df_renamed.rename(columns={
                        'Mean': 'Mean_Total',
                        'StdDev': 'Std_Total'
                    })
                else:
                    model_summary_df_renamed = model_summary_df
                
                generate_comparison_graphs_with_colors(model_summary_df_renamed, model_overall_df, summary_dir, suffix="-models", 
                                                      title_prefix="Porovnání úspěšnosti modelů", xlabel="Model")
            except Exception as e:
                print(f"Chyba při generování grafů: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Průběžné porovnání dokončeno. Finální grafy budou vygenerovány na konci běhu.")
    else:
        print("Žádná data k vykreslení grafů.")


def generate_comparison_graphs(summary_df, overall_df, summary_dir, suffix="", title_prefix="Porovnání úspěšnosti", xlabel="Model"):
    """
    Generuje grafy porovnání na základě dodaných dataframů.
    
    Args:
        summary_df: DataFrame s výsledky pro jednotlivá pole
        overall_df: DataFrame s celkovými výsledky
        summary_dir: Adresář pro uložení výsledků
        suffix: Přípona pro názvy souborů (např. "-pipelines" nebo "-models")
        title_prefix: Prefix pro název grafů
        xlabel: Popisek osy X
        
    Returns:
        List[str]: Seznam názvů vytvořených souborů
    """
    created_files = []
    
    # Kontrola základních požadavků na dataframy
    required_summary_cols = ['Field', 'Model', 'Mean_Total', 'Std_Total']
    if 'Mean_Base' in summary_df.columns and 'Mean_Improved' in summary_df.columns:
        has_improvement_data = True
    else:
        has_improvement_data = False
    
    if summary_df.empty and overall_df.empty:
        print(f"Nebylo možné spojit žádná data pro grafy{suffix}.")
        return created_files

    # Barva pro vylepšenou část (sémantické zlepšení)
    improved_color = "#f8c471"  # světle oranžová

    # --- Vytvoření grafů pro jednotlivá pole ---
    if not summary_df.empty and all(col in summary_df.columns for col in required_summary_cols):
        metadata_fields = summary_df['Field'].unique()
        
        for field in metadata_fields:
            df_field = summary_df[summary_df['Field'] == field].sort_values(by='Mean_Total', ascending=False)
            
            if df_field.empty:
                continue

            plt.figure(figsize=(max(10, len(df_field['Model'])*0.8), 6))
            
            means = df_field['Mean_Total']
            errors = df_field['Std_Total'].fillna(0)
            models = df_field['Model']
            
            x_pos = np.arange(len(models))
            
            if has_improvement_data:
                # Vykreslení základní části a vylepšené části odděleně
                base_values = df_field['Mean_Base']
                improved_values = df_field['Mean_Improved']
                
                # Základní část
                plt.bar(x_pos, base_values, color='skyblue', alpha=0.8)
                
                # Vylepšená část (pokud existuje)
                plt.bar(x_pos, improved_values, bottom=base_values, color=improved_color, alpha=0.8)
                
                # Chybové úsečky na celkové hodnotě
                plt.errorbar(x_pos, means, yerr=errors, fmt='none', ecolor='black', capsize=5)
            else:
                # Vykreslení standardních sloupců
                bars = plt.bar(x_pos, means, color='skyblue')
                plt.errorbar(x_pos, means, yerr=errors, fmt='none', ecolor='black', capsize=5)

            plt.title(f'{title_prefix} - {field} (průměr ±1σ)')
            plt.xlabel(xlabel)
            plt.ylabel('Průměrná podobnost')
            plt.xticks(x_pos, models, rotation=45, ha="right")
            plt.ylim(0, max(1.05, (means + errors).max() * 1.1))
            
            # Přidání legendy pro vylepšení
            if has_improvement_data:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='skyblue', alpha=0.8, label='Základní hodnota'),
                    Patch(facecolor=improved_color, alpha=0.8, label='Vylepšení sémantickou kontrolou')
                ]
                plt.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            output_file = summary_dir / f"{field}_comparison{suffix}.png"
            try:
                plt.savefig(output_file)
                created_files.append(output_file.name)
            except Exception as e:
                print(f"Chyba při ukládání grafu {output_file.name}: {e}")
            plt.close()
            
            output_csv = summary_dir / f"{field}_comparison{suffix}.csv"
            try:
                field_columns = ['Model', 'Mean_Total', 'Std_Total']
                # Přidáme další sloupce, pokud existují
                for col in ['Mean_Base', 'Mean_Improved', 'OriginalModels', 'PipelineTypes', 'PipelineType']:
                    if col in df_field.columns:
                        field_columns.append(col)
                df_field[field_columns].to_csv(output_csv, index=False, float_format='%.4f')
                created_files.append(output_csv.name)
            except Exception as e:
                 print(f"Chyba při ukládání CSV {output_csv.name}: {e}")

    else:
        print(f"Přeskakuji generování grafů a CSV pro jednotlivá pole{suffix} - chybí data nebo sloupce.")

    # --- Vytvoření grafu celkových výsledků ---
    if not overall_df.empty and 'Mean_Total_Overall' in overall_df.columns and 'Std_Total_Overall' in overall_df.columns:
        
        df_overall_sorted = overall_df.sort_values(by='Mean_Total_Overall', ascending=False)
        
        means_overall = df_overall_sorted['Mean_Total_Overall']
        errors_overall = df_overall_sorted['Std_Total_Overall'].fillna(0)
        models_overall = df_overall_sorted['Model']
        
        x_pos_overall = np.arange(len(models_overall))

        plt.figure(figsize=(max(10, len(models_overall)*0.8), 6))
        
        if 'Mean_Base_Overall' in df_overall_sorted.columns and 'Mean_Improved_Overall' in df_overall_sorted.columns:
            # Vykreslení základní části a vylepšené části odděleně
            base_values = df_overall_sorted['Mean_Base_Overall']
            improved_values = df_overall_sorted['Mean_Improved_Overall']
            
            # Základní část
            plt.bar(x_pos_overall, base_values, color='lightcoral', alpha=0.8)
            
            # Vylepšená část (pokud existuje)
            plt.bar(x_pos_overall, improved_values, bottom=base_values, color=improved_color, alpha=0.8)
            
            # Chybové úsečky na celkové hodnotě
            plt.errorbar(x_pos_overall, means_overall, yerr=errors_overall, fmt='none', ecolor='black', capsize=5)
            
            # Legenda
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightcoral', alpha=0.8, label='Základní hodnota'),
                Patch(facecolor=improved_color, alpha=0.8, label='Vylepšení sémantickou kontrolou')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
        else:
            # Vykreslení standardních sloupců
            bars_overall = plt.bar(x_pos_overall, means_overall, color='lightcoral')
            plt.errorbar(x_pos_overall, means_overall, yerr=errors_overall, fmt='none', ecolor='black', capsize=5)

        plt.title(f'Celková úspěšnost {xlabel.lower()}ů (průměr ±1σ)')
        plt.xlabel(xlabel)
        plt.ylabel('Průměrná celková podobnost')
        plt.xticks(x_pos_overall, models_overall, rotation=45, ha="right")
        plt.ylim(0, max(1.05, (means_overall + errors_overall).max() * 1.1))
        plt.tight_layout()
        
        output_file = summary_dir / f"overall_comparison{suffix}.png"
        try:
            plt.savefig(output_file)
            created_files.append(output_file.name)
        except Exception as e:
            print(f"Chyba při ukládání grafu {output_file.name}: {e}")
        plt.close()

        output_csv_overall = summary_dir / f"overall_comparison{suffix}.csv"
        try:
            overall_columns = ['Model', 'Mean_Total_Overall', 'Std_Total_Overall']
            # Přidáme další sloupce, pokud existují
            for col in ['Mean_Base_Overall', 'Mean_Improved_Overall', 'OriginalModels', 'PipelineTypes', 'PipelineType']:
                if col in df_overall_sorted.columns:
                    overall_columns.append(col)
            df_overall_sorted[overall_columns].to_csv(output_csv_overall, index=False, float_format='%.4f')
            created_files.append(output_csv_overall.name)
        except Exception as e:
             print(f"Chyba při ukládání CSV {output_csv_overall.name}: {e}")
             
        # Uložení spojených finálních dat
        try:
            summary_df.to_csv(summary_dir / f"final_summary_all_fields{suffix}.csv", index=False, float_format='%.4f')
            overall_df.to_csv(summary_dir / f"final_overall_all{suffix}.csv", index=False, float_format='%.4f')
            created_files.append(f"final_summary_all_fields{suffix}.csv")
            created_files.append(f"final_overall_all{suffix}.csv")
        except Exception as e:
            print(f"Chyba při ukládání finálních spojených CSV{suffix}: {e}")

    else:
         print(f"Přeskakuji generování celkového grafu a CSV{suffix} - chybí data nebo sloupce.")
         
    return created_files


def generate_comparison_graphs_with_colors(summary_df, overall_df, summary_dir, suffix="", title_prefix="Porovnání úspěšnosti", xlabel="Model"):
    """
    Generuje grafy porovnání na základě dodaných dataframů s barevným rozlišením podle typu pipeline.
    
    Args:
        summary_df: DataFrame s výsledky pro jednotlivá pole
        overall_df: DataFrame s celkovými výsledky
        summary_dir: Adresář pro uložení výsledků
        suffix: Přípona pro názvy souborů
        title_prefix: Prefix pro název grafů
        xlabel: Popisek osy X
        
    Returns:
        List[str]: Seznam názvů vytvořených souborů
    """
    created_files = []
    
    # Kontrola základních požadavků na dataframy
    required_summary_cols = ['Field', 'Model', 'Mean_Total', 'Std_Total']
    if 'Mean_Base' in summary_df.columns and 'Mean_Improved' in summary_df.columns:
        has_improvement_data = True
    else:
        has_improvement_data = False
    
    if summary_df.empty and overall_df.empty:
        print(f"Nebylo možné spojit žádná data pro grafy{suffix}.")
        return created_files

    # Definice barev pro typy pipeline
    pipeline_colors = {
        'TEXT': 'royalblue',
        'EMBEDDED': 'green',
        'VLM': 'firebrick',
        'HYBRID': 'purple',  # Fialová barva pro hybridní pipeline
        'MULTIMODAL': 'orange'  # Oranžová barva pro multimodální pipeline
    }
    
    # Barva pro vylepšenou část (sémantické zlepšení)
    improved_color = "#f8c471"  # světle oranžová
    
    # --- Vytvoření grafů pro jednotlivá pole ---
    if not summary_df.empty and all(col in summary_df.columns for col in required_summary_cols):
        metadata_fields = summary_df['Field'].unique()
        
        for field in metadata_fields:
            df_field = summary_df[summary_df['Field'] == field].sort_values(by='Mean_Total', ascending=False)
            
            if df_field.empty:
                continue

            plt.figure(figsize=(max(10, len(df_field['Model'])*0.8), 6))
            
            means = df_field['Mean_Total']
            errors = df_field['Std_Total'].fillna(0)
            models = df_field['Model']
            
            x_pos = np.arange(len(models))
            
            if has_improvement_data:
                # Vykreslení základní části a vylepšené části odděleně, s barvami podle pipeline
                for i, (_, row) in enumerate(df_field.iterrows()):
                    pipeline_type = row.get('PipelineType', 'unknown')
                    base_color = pipeline_colors.get(pipeline_type, 'gray')
                    
                    # Základní část
                    base_value = row['Mean_Base']
                    plt.bar(x_pos[i], base_value, color=base_color, alpha=0.8)
                    
                    # Vylepšená část (pokud existuje)
                    improved_value = row['Mean_Improved']
                    if improved_value > 0:
                        plt.bar(x_pos[i], improved_value, bottom=base_value, color=improved_color, alpha=0.8)
                
                # Chybové úsečky na celkové hodnotě
                plt.errorbar(x_pos, means, yerr=errors, fmt='none', ecolor='black', capsize=5)
            else:
                # Použití barev podle typu pipeline
                bar_colors = []
                for _, row in df_field.iterrows():
                    pipeline_type = row.get('PipelineType', 'unknown')
                    bar_colors.append(pipeline_colors.get(pipeline_type, 'gray'))
                
                bars = plt.bar(x_pos, means, color=bar_colors)
                plt.errorbar(x_pos, means, yerr=errors, fmt='none', ecolor='black', capsize=5)

            plt.title(f'{title_prefix} - {field} (průměr ±1σ)')
            plt.xlabel(xlabel)
            plt.ylabel('Průměrná podobnost')
            plt.xticks(x_pos, models, rotation=45, ha="right")
            plt.ylim(0, max(1.05, (means + errors).max() * 1.1))
            
            # Přidání legendy
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=pipeline_colors['TEXT'], label='Text model'),
                Patch(facecolor=pipeline_colors['EMBEDDED'], label='Embedding model'),
                Patch(facecolor=pipeline_colors['VLM'], label='Vision model'),
                Patch(facecolor=pipeline_colors['HYBRID'], label='Hybrid model (LLM+VLM)'),
                Patch(facecolor=pipeline_colors['MULTIMODAL'], label='Multimodal model (Text+Vision)')
            ]
            
            # Přidání legendy pro vylepšení
            if has_improvement_data:
                legend_elements.append(
                    Patch(facecolor=improved_color, alpha=0.8, label='Vylepšení sémantickou kontrolou')
                )
            
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            output_file = summary_dir / f"{field}_comparison{suffix}.png"
            try:
                plt.savefig(output_file)
                created_files.append(output_file.name)
            except Exception as e:
                print(f"Chyba při ukládání grafu {output_file.name}: {e}")
            plt.close()
            
            output_csv = summary_dir / f"{field}_comparison{suffix}.csv"
            try:
                field_columns = ['Model', 'Mean_Total', 'Std_Total', 'PipelineType']
                # Přidáme další sloupce, pokud existují
                for col in ['Mean_Base', 'Mean_Improved', 'OriginalModels']:
                    if col in df_field.columns:
                        field_columns.append(col)
                df_field[field_columns].to_csv(output_csv, index=False, float_format='%.4f')
                created_files.append(output_csv.name)
            except Exception as e:
                 print(f"Chyba při ukládání CSV {output_csv.name}: {e}")

    else:
        print(f"Přeskakuji generování grafů a CSV pro jednotlivá pole{suffix} - chybí data nebo sloupce.")

    # --- Vytvoření grafu celkových výsledků ---
    if not overall_df.empty and 'Mean_Total_Overall' in overall_df.columns and 'Std_Total_Overall' in overall_df.columns:
        
        df_overall_sorted = overall_df.sort_values(by='Mean_Total_Overall', ascending=False)
        
        means_overall = df_overall_sorted['Mean_Total_Overall']
        errors_overall = df_overall_sorted['Std_Total_Overall'].fillna(0)
        models_overall = df_overall_sorted['Model']
        
        x_pos_overall = np.arange(len(models_overall))

        plt.figure(figsize=(max(10, len(models_overall)*0.8), 6))
        
        if 'Mean_Base_Overall' in df_overall_sorted.columns and 'Mean_Improved_Overall' in df_overall_sorted.columns:
            # Vykreslení základní části a vylepšené části odděleně, s barvami podle pipeline
            for i, (_, row) in enumerate(df_overall_sorted.iterrows()):
                pipeline_type = row.get('PipelineType', 'unknown')
                base_color = pipeline_colors.get(pipeline_type, 'gray')
                
                # Základní část
                base_value = row['Mean_Base_Overall']
                plt.bar(x_pos_overall[i], base_value, color=base_color, alpha=0.8)
                
                # Vylepšená část (pokud existuje)
                improved_value = row['Mean_Improved_Overall']
                if improved_value > 0:
                    plt.bar(x_pos_overall[i], improved_value, bottom=base_value, color=improved_color, alpha=0.8)
            
            # Chybové úsečky na celkové hodnotě
            plt.errorbar(x_pos_overall, means_overall, yerr=errors_overall, fmt='none', ecolor='black', capsize=5)
        else:
            # Použití barev podle typu pipeline
            bar_colors = []
            for _, row in df_overall_sorted.iterrows():
                pipeline_type = row.get('PipelineType', 'unknown')
                bar_colors.append(pipeline_colors.get(pipeline_type, 'gray'))
            
            bars_overall = plt.bar(x_pos_overall, means_overall, color=bar_colors)
            plt.errorbar(x_pos_overall, means_overall, yerr=errors_overall, fmt='none', ecolor='black', capsize=5)

        plt.title(f'Celková úspěšnost {xlabel.lower()}ů (průměr ±1σ)')
        plt.xlabel(xlabel)
        plt.ylabel('Průměrná celková podobnost')
        plt.xticks(x_pos_overall, models_overall, rotation=45, ha="right")
        plt.ylim(0, max(1.05, (means_overall + errors_overall).max() * 1.1))
        
        # Přidání legendy
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=pipeline_colors['TEXT'], label='Text model'),
            Patch(facecolor=pipeline_colors['EMBEDDED'], label='Embedding model'),
            Patch(facecolor=pipeline_colors['VLM'], label='Vision model'),
            Patch(facecolor=pipeline_colors['HYBRID'], label='Hybrid model (Text+VLM)'),
            Patch(facecolor=pipeline_colors['MULTIMODAL'], label='Multimodal model (Text+Vision)')
        ]
        
        # Přidání legendy pro vylepšení
        if 'Mean_Improved_Overall' in df_overall_sorted.columns:
            legend_elements.append(
                Patch(facecolor=improved_color, alpha=0.8, label='Vylepšení sémantickou kontrolou')
            )
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        output_file = summary_dir / f"overall_comparison{suffix}.png"
        try:
            plt.savefig(output_file)
            created_files.append(output_file.name)
        except Exception as e:
            print(f"Chyba při ukládání grafu {output_file.name}: {e}")
        plt.close()

        output_csv_overall = summary_dir / f"overall_comparison{suffix}.csv"
        try:
            overall_columns = ['Model', 'Mean_Total_Overall', 'Std_Total_Overall', 'PipelineType']
            # Přidáme další sloupce, pokud existují
            for col in ['Mean_Base_Overall', 'Mean_Improved_Overall', 'OriginalModels']:
                if col in df_overall_sorted.columns:
                    overall_columns.append(col)
            df_overall_sorted[overall_columns].to_csv(output_csv_overall, index=False, float_format='%.4f')
            created_files.append(output_csv_overall.name)
        except Exception as e:
             print(f"Chyba při ukládání CSV {output_csv_overall.name}: {e}")
             
        # Uložení spojených finálních dat
        try:
            summary_df.to_csv(summary_dir / f"final_summary_all_fields{suffix}.csv", index=False, float_format='%.4f')
            overall_df.to_csv(summary_dir / f"final_overall_all{suffix}.csv", index=False, float_format='%.4f')
            created_files.append(f"final_summary_all_fields{suffix}.csv")
            created_files.append(f"final_overall_all{suffix}.csv")
        except Exception as e:
            print(f"Chyba při ukládání finálních spojených CSV{suffix}: {e}")

    else:
         print(f"Přeskakuji generování celkového grafu a CSV{suffix} - chybí data nebo sloupce.")
         
    return created_files


def generate_graphs_only(results_dir: str):
    """
    Generuje pouze grafy z existujících výsledků bez spouštění extrakce.
    
    Args:
        results_dir (str): Cesta k adresáři s výsledky
    """
    if not results_dir:
        print("CHYBA: Při použití --graphs-only musíte zadat cestu k adresáři s výsledky (--results-dir)")
        sys.exit(1)
    
    results_path = Path(results_dir)
    if not results_path.exists() or not results_path.is_dir():
        print(f"CHYBA: Zadaný adresář s výsledky '{results_dir}' neexistuje nebo není adresář")
        sys.exit(1)
    
    print(f"\n=== Generování grafů z existujících výsledků v adresáři: {results_dir} ===")
    
    # Nastavení adresáře s výsledky jako hlavní adresář běhu
    set_run_results_dir(results_path)
    
    # Najdeme všechny podadresáře, které odpovídají konfiguracím
    # (přeskočíme adresář final_comparison, který obsahuje souhrnné grafy)
    config_dirs = [d for d in results_path.iterdir() 
                  if d.is_dir() and d.name != "final_comparison"]
    
    if not config_dirs:
        print("Nebyly nalezeny žádné adresáře s konfiguracemi.")
        return
    
    print(f"Nalezeno {len(config_dirs)} adresářů s konfiguracemi.")
    
    # Seznam adresářů pro vytvoření souhrnných grafů
    result_dirs = []
    
    # Pro každou konfiguraci spustíme skript na generování grafů
    for config_dir in config_dirs:
        print(f"\nZpracovávám konfiguraci: {config_dir.name}")
        
        # Kontrola, zda existují potřebné soubory pro generování grafů
        csv_files = list(config_dir.glob("*.csv"))
        json_files = list(config_dir.glob("*comparison*.json"))
        
        if not csv_files or not json_files:
            print(f"Adresář {config_dir.name} neobsahuje potřebné soubory (CSV nebo JSON). Přeskakuji.")
            continue
        
        # Nastavení adresáře konfigurace jako aktuální adresář pro výsledky
        set_run_results_dir(config_dir)
        
        # Generování grafů pomocí příkazu pro main.py
        cmd_args = [
            sys.executable,
            "-m", "src.main",
            "--graphs-only",
            "--output-dir", str(config_dir),
            "--config", str(config_dir / "config.json")
        ]
        
        print(f"Spouštím příkaz: {' '.join(cmd_args)}")
        try:
            result = subprocess.run(cmd_args, check=True)
            print(f"Generování grafů dokončeno s návratovým kódem: {result.returncode}")
            result_dirs.append(str(config_dir))
        except subprocess.CalledProcessError as e:
            print(f"Chyba při generování grafů: {e}")
        except Exception as e:
            print(f"Neočekávaná chyba: {e}")
    
    # Nastavení hlavního adresáře běhu zpět na původní hodnotu
    set_run_results_dir(results_path)
    
    # Vytvoření souhrnných grafů
    if result_dirs:
        print("\n=== Generování souhrnných grafů ===")
        create_final_comparison(result_dirs, is_final=True)
    else:
        print("\nNebyly nalezeny žádné adresáře s platným formátem dat pro vytvoření souhrnných grafů.")


def combine_results_only(results_dir: str):
    """
    Kombinuje výsledky Text a VLM pipeline v existujících adresářích bez provádění nové extrakce.
    
    Args:
        results_dir (str): Cesta k adresáři s výsledky
    """
    if not results_dir:
        print("CHYBA: Při použití --combine-only musíte zadat cestu k adresáři s výsledky (--results-dir)")
        sys.exit(1)
    
    results_path = Path(results_dir)
    print(f"Kontroluji existenci zadaného adresáře: {results_path}")
    
    if not results_path.exists():
        print(f"CHYBA: Zadaný adresář s výsledky '{results_dir}' neexistuje")
        sys.exit(1)
    
    if not results_path.is_dir():
        print(f"CHYBA: Zadaná cesta '{results_dir}' není adresář")
        sys.exit(1)
    
    print(f"\n=== Kombinace výsledků Text a VLM pipeline v adresáři: {results_dir} ===")
    
    # Kontrola, zda adresář obsahuje přímo soubory text_results.json a vlm_results.json
    text_results_path = results_path / "text_results.json"
    vlm_results_path = results_path / "vlm_results.json"
    
    if text_results_path.exists() and vlm_results_path.exists():
        print(f"Nalezeny soubory text_results.json a vlm_results.json přímo v zadaném adresáři.")
        # Nastavení adresáře s výsledky jako hlavní adresář běhu
        set_run_results_dir(results_path)
        # Kombinace výsledků
        combine_pipeline_results(str(results_path))
        return
    
    # Pokud soubory nejsou přímo v adresáři, hledáme podadresáře
    print(f"Hledám soubory v podadresářích...")
    
    # Nastavení adresáře s výsledky jako hlavní adresář běhu
    set_run_results_dir(results_path)
    
    # Najdeme všechny podadresáře, které odpovídají konfiguracím
    # (přeskočíme adresář final_comparison, který obsahuje souhrnné grafy)
    config_dirs = [d for d in results_path.iterdir() 
                  if d.is_dir() and d.name != "final_comparison"]
    
    if not config_dirs:
        print("Nebyly nalezeny žádné adresáře s konfiguracemi.")
        return
    
    print(f"Nalezeno {len(config_dirs)} adresářů s konfiguracemi.")
    
    # Seznam adresářů pro vytvoření souhrnných grafů
    result_dirs = []
    
    # Pro každou konfiguraci vytvoříme hybridní výsledky
    for config_dir in config_dirs:
        print(f"\nZpracovávám konfiguraci: {config_dir.name}")
        
        # Kontrola, zda existují potřebné soubory pro kombinaci
        text_file = config_dir / "text_results.json"
        vlm_file = config_dir / "vlm_results.json"
        
        if not text_file.exists():
            print(f"Soubor text_results.json neexistuje v adresáři {config_dir.name}.")
        else:
            print(f"Nalezen text_results.json: {text_file}")
            
        if not vlm_file.exists():
            print(f"Soubor vlm_results.json neexistuje v adresáři {config_dir.name}.")
        else:
            print(f"Nalezen vlm_results.json: {vlm_file}")
        
        if not text_file.exists() or not vlm_file.exists():
            print(f"Adresář {config_dir.name} neobsahuje potřebné soubory (text_results.json nebo vlm_results.json). Přeskakuji.")
            continue
        
        # Kombinace výsledků
        combine_pipeline_results(str(config_dir))
        result_dirs.append(str(config_dir))
    
    # Vytvoření souhrnných grafů
    if result_dirs:
        print("\n=== Generování souhrnných grafů s hybridními výsledky ===")
        
        # Připravíme slovník zpracovaných modelů pro správné zobrazení hybridů v grafech
        processed_models = {}
        for result_dir in result_dirs:
            # Pro každý adresář s výsledky přidáme záznam do zpracovaných modelů
            processed_models.setdefault("HYBRID", []).append(result_dir)
            
        # Vytvoření finálního porovnání
        create_final_comparison(result_dirs, is_final=True)
    else:
        print("\nNebyly nalezeny žádné adresáře, kde by bylo možné vytvořit hybridní výsledky.")


def main():
    """Hlavní funkce skriptu."""
    args = parse_args()
    
    # Zpracování parametru --graphs-only
    if args.graphs_only:
        generate_graphs_only(args.results_dir)
        return
    
    # Zpracování parametru --combine-only
    if args.combine_only:
        combine_results_only(args.results_dir)
        return
    
    # <<< Změna: Nastavení hlavního adresáře pro výsledky tohoto běhu run_all_models >>>
    timestamp_all = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_run_dir = BASE_DIR / "results" / f"all_models_{timestamp_all}"
    set_run_results_dir(overall_run_dir)
    # <<< Konec změny >>>
    
    # Zajistíme vytvoření konfiguračního souboru, pokud neexistuje
    default_config_path = ensure_config_file()
    
    # Načteme konfigurace
    if args.config:
        # Použijeme explicitně zadaný konfigurační soubor
        configurations = load_configurations_from_file(args.config)
    else:
        # Použijeme výchozí model_configs.json
        print(f"Používám výchozí konfigurační soubor: {default_config_path}")
        configurations = load_configurations_from_file(default_config_path)
    
    print(f"Načteno {len(configurations)} konfigurací modelů.")
    
    # Seznam adresářů s výsledky
    result_dirs = []
    
    # Slovník pro sledování již zpracovaných modelů
    # Klíč: identifikátor modelu (provider_model), Hodnota: cesta k adresáři s výsledky
    processed_models = {}
    
    # Projdeme všechny konfigurace a spustíme extrakci
    for i, config in enumerate(configurations, 1):
        print(f"\nKonfigurace {i}/{len(configurations)}")
        
        # Zajistíme, že konfigurace obsahuje povinné sekce
        if 'text_pipeline' not in config:
            config['text_pipeline'] = {
                "enabled": True,
                "max_text_length": 6000,
                "extract_references": True,
                "use_direct_pattern_extraction": True
            }
            print("Přidána chybějící konfigurace pro textovou pipeline")
        
        result_dir = run_extraction(config, args, processed_models)
        if result_dir:
            result_dirs.append(result_dir)
            # Vytvoření hybridních výsledků pro tuto konfiguraci
            combine_pipeline_results(result_dir)
            # Přidání hybridní pipeline do seznamu zpracovaných modelů
            processed_models.setdefault("HYBRID", []).append(result_dir)
        
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