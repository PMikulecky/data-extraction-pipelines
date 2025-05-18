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


def prepare_result_directory(config_name: str) -> str:
    """
    Připraví adresář pro výsledky dané konfigurace v rámci hlavního adresáře běhu.
    
    Args:
        config_name (str): Název konfigurace
        
    Returns:
        str: Cesta k adresáři s výsledky pro danou konfiguraci
    """
    main_run_dir = get_run_results_dir() # Získáme hlavní adresář běhu run_all_models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # <<< Změna: Sanitizace názvu konfigurace >>>
    sanitized_config_name = sanitize_filename(config_name)
    # Název adresáře bude obsahovat sanitizovaný název konfigurace a časové razítko
    result_dir_path = main_run_dir / f"{sanitized_config_name}_{timestamp}"
    # <<< Konec změny >>>
    result_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Adresář pro výsledky konfigurace '{config_name}': {result_dir_path}")
    return str(result_dir_path)


def run_extraction(config: Dict[str, Any], args, processed_models: Dict[str, str]) -> str | None:
    """
    Spustí extrakci metadat s danou konfigurací a vrátí cestu k adresáři s výsledky.
    
    Args:
        config (Dict[str, Any]): Konfigurace modelu
        args: Argumenty příkazové řádky
        processed_models (Dict[str, str]): Seznam již zpracovaných modelů a cest k jejich výsledkům
    """
    config_name = config.get("name", "unknown")
    print(f"\n=== Spouštím extrakci metadat s konfigurací: {config_name} ===")
    
    # Příprava adresáře pro výsledky
    result_dir = prepare_result_directory(config_name)
    
    # Uložení dočasné konfigurace
    save_temp_config(config)
    
    # Nastavení argumentů pro spuštění hlavního programu
    cmd_args = [
        sys.executable,
        "-m", "src.main", # Spustit jako modul
        "--models", "embedded", "vlm", "text", "hybrid",
        "--output-dir", result_dir,
        "--limit", str(args.limit) if args.limit else "100",
    ]
    
    # Přidáme cestu ke konfiguraci
    cmd_args.extend(["--config", str(TEMP_CONFIG_FILE)])
    
    # Přidáme další argumenty
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
        shutil.copy(TEMP_CONFIG_FILE, Path(result_dir) / "used_config.json")
        
        # Aktualizujeme slovník zpracovaných modelů (pro budoucí použití, např. přeskakování)
        # Klíč může být název konfigurace nebo kombinace modelů
        processed_models[config_name] = result_dir
            
        return result_dir
            
    except subprocess.CalledProcessError as e:
        print(f"Chyba při spuštění extrakce: {e}")
        return None
    except KeyboardInterrupt:
        print("\nExtrakce přerušena uživatelem.")
        sys.exit(1)


def combine_pipeline_results(result_dir: str) -> None:
    """
    Vytvoří hybridní pipeline kombinací výsledků z Text a VLM pipeline.
    
    Args:
        result_dir (str): Cesta k adresáři s výsledky
    """
    print(f"\n=== Kombinuji výsledky Text a VLM pipeline v adresáři: {result_dir} ===")
    
    result_dir_path = Path(result_dir)
    
    # Načtení výsledků Text pipeline
    text_results_path = result_dir_path / "text_results.json"
    if not text_results_path.exists():
        print(f"Soubor s výsledky Text pipeline ({text_results_path}) neexistuje.")
        return
    
    try:
        with open(text_results_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
    except Exception as e:
        print(f"Chyba při načítání výsledků Text pipeline: {e}")
        return
    
    # Načtení výsledků VLM pipeline
    vlm_results_path = result_dir_path / "vlm_results.json"
    if not vlm_results_path.exists():
        print(f"Soubor s výsledky VLM pipeline ({vlm_results_path}) neexistuje.")
        return
    
    try:
        with open(vlm_results_path, 'r', encoding='utf-8') as f:
            vlm_data = json.load(f)
    except Exception as e:
        print(f"Chyba při načítání výsledků VLM pipeline: {e}")
        return
    
    # Inicializace hybridních výsledků
    hybrid_results = {
        "results": [],
        "timings": {}
    }
    
    # Příprava výsledků Text pipeline
    text_results_by_doi = {}
    if isinstance(text_data, dict):
        # Formát: text_data je slovník, kde klíče jsou DOI a hodnoty jsou metadata
        for doi, item in text_data.items():
            if isinstance(item, dict) and "doi" in item:
                text_results_by_doi[item["doi"]] = item
            else:
                # Zkusit použít DOI z klíče
                item_copy = dict(item) if isinstance(item, dict) else {"content": item}
                item_copy["doi"] = doi
                text_results_by_doi[doi] = item_copy
    elif "results" in text_data and isinstance(text_data["results"], list):
        # Formát: text_data.results je seznam metadat
        for item in text_data["results"]:
            if "doi" in item:
                text_results_by_doi[item["doi"]] = item
    else:
        print("Neznámý formát text_results.json, přeskakuji.")
        return
    
    # Příprava výsledků VLM pipeline
    vlm_results_by_doi = {}
    if "results" in vlm_data and isinstance(vlm_data["results"], dict):
        # Formát: vlm_data.results je slovník, kde klíče jsou DOI a hodnoty jsou metadata
        for doi, item in vlm_data["results"].items():
            if isinstance(item, dict) and "doi" in item:
                vlm_results_by_doi[item["doi"]] = item
            else:
                # Zkusit použít DOI z klíče
                item_copy = dict(item) if isinstance(item, dict) else {"content": item}
                item_copy["doi"] = doi
                vlm_results_by_doi[doi] = item_copy
    elif "results" in vlm_data and isinstance(vlm_data["results"], list):
        # Formát: vlm_data.results je seznam metadat
        for item in vlm_data["results"]:
            if "doi" in item:
                vlm_results_by_doi[item["doi"]] = item
    else:
        print("Neznámý formát vlm_results.json, přeskakuji.")
        return
    
    # Sloučení časů zpracování (součet časů obou pipeline)
    text_timings = text_data.get("timings", {})
    vlm_timings = vlm_data.get("timings", {})
    
    if isinstance(text_timings, dict) and isinstance(vlm_timings, dict):
        # Pokud timings jsou slovníky, počítáme total a average
        text_total = sum(text_timings.values()) if text_timings else 0
        vlm_total = sum(vlm_timings.values()) if vlm_timings else 0
        total_time = text_total + vlm_total
        avg_time = total_time / 2  # nebo (len(text_timings) + len(vlm_timings)) pokud není 0
        
        hybrid_results["timings"] = {
            "total": total_time,
            "average": avg_time
        }
    else:
        # Pokud timings mají jiný formát, použijeme text_timings
        hybrid_results["timings"] = text_timings
    
    # Přidáme token_usages, pokud jsou k dispozici
    if "token_usages" in text_data or "token_usages" in vlm_data:
        hybrid_results["token_usages"] = {}
        if "token_usages" in text_data:
            hybrid_results["token_usages"].update(text_data["token_usages"])
        if "token_usages" in vlm_data:
            hybrid_results["token_usages"].update(vlm_data["token_usages"])
    
    # Definice metadat, která preferujeme z VLM pipeline
    vlm_preferred_fields = ["title", "authors", "doi", "issue", "volume", "journal", "publisher", "year"]
    
    # Kombinace výsledků
    processed_dois = set(text_results_by_doi.keys()) | set(vlm_results_by_doi.keys())
    print(f"Celkem nalezeno {len(processed_dois)} unikátních DOI ke zpracování.")
    
    for doi in processed_dois:
        hybrid_item = {}
        
        # Začneme s daty z Text pipeline (pokud existují)
        if doi in text_results_by_doi:
            hybrid_item = copy.deepcopy(text_results_by_doi[doi])
        
        # Přidáme nebo nahradíme preferovaná pole z VLM pipeline
        if doi in vlm_results_by_doi:
            vlm_item = vlm_results_by_doi[doi]
            
            for field in vlm_preferred_fields:
                # Přidáme pole z VLM pouze pokud existuje a není prázdné
                if field in vlm_item and vlm_item[field] and (field not in hybrid_item or not hybrid_item[field]):
                    hybrid_item[field] = vlm_item[field]
            
            # Pro ostatní pole v VLM, která nejsou v seznamu preferovaných, ale chybí v Text výsledcích
            for field, value in vlm_item.items():
                if field not in vlm_preferred_fields and (field not in hybrid_item or not hybrid_item[field]) and value:
                    hybrid_item[field] = value
        
        # Přidání záznamu do hybridních výsledků, pokud má alespoň některá metadata
        if any(field != "doi" and value for field, value in hybrid_item.items()):
            hybrid_results["results"].append(hybrid_item)
    
    print(f"Vytvořeno {len(hybrid_results['results'])} hybridních záznamů.")
    
    # Uložení hybridních výsledků
    hybrid_results_path = result_dir_path / "hybrid_results.json"
    try:
        with open(hybrid_results_path, 'w', encoding='utf-8') as f:
            json.dump(hybrid_results, f, ensure_ascii=False, indent=2)
        print(f"Hybridní výsledky byly uloženy do souboru: {hybrid_results_path}")
    except Exception as e:
        print(f"Chyba při ukládání hybridních výsledků: {e}")
        return

    # Spuštění porovnání s referenčními daty pro hybridní pipeline
    cmd_args = [
        sys.executable,
        "-m", "src.main",
        "--compare-only", "hybrid",
        "--output-dir", str(result_dir_path),
        "--skip-semantic" if "--skip-semantic" in sys.argv else ""
    ]
    
    # Odstraníme prázdné argumenty
    cmd_args = [arg for arg in cmd_args if arg]
    
    print(f"Spouštím příkaz pro porovnání hybridních výsledků: {' '.join(cmd_args)}")
    try:
        result = subprocess.run(cmd_args, check=True)
        print(f"Porovnání hybridních výsledků dokončeno s návratovým kódem: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Chyba při porovnání hybridních výsledků: {e}")


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
        "text_results.json",  # Přidány výsledky textové pipeline
        "hybrid_results.json",  # Přidány výsledky hybridní pipeline
        "embedded_comparison.json",
        "vlm_comparison.json",
        "text_comparison.json",  # Přidáno porovnání textové pipeline
        "hybrid_comparison.json",  # Přidáno porovnání hybridní pipeline
        "semantic_comparison_results.json",
        "embedded_comparison_semantic.json",
        "vlm_comparison_semantic.json",  # Přidáno sémantické porovnání textové pipeline
        "hybrid_comparison_semantic.json",  # Přidáno sémantické porovnání hybridní pipeline
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
    Vytvoří souhrnné porovnání výsledků ze všech spuštěných konfigurací.
    
    Args:
        result_dirs (List[str]): Seznam cest k adresářům s výsledky jednotlivých konfigurací.
        is_final (bool): Zda se jedná o finální porovnání po všech bězích.
    """
    if not result_dirs:
        print("Žádné výsledky k porovnání.")
        return
    
    # Adresář pro souhrnné výsledky v hlavním adresáři běhu
    summary_dir = get_run_results_dir() / "final_comparison"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Prohledávám {len(result_dirs)} adresářů s výsledky pro summary a overall CSV...")

    # Slovníky pro ukládání výsledků podle typu pipeline (embedded, text, vlm)
    all_summary_by_pipeline = {}  # Klíč: (pipeline_typ, field), Hodnota: seznam hodnot
    all_overall_by_pipeline = {}  # Klíč: pipeline_typ, Hodnota: seznam hodnot
    
    # Slovníky pro ukládání výsledků podle konkrétního modelu
    all_summary_by_specific_model = {}  # Klíč: (model_name, field), Hodnota: seznam hodnot
    all_overall_by_specific_model = {}  # Klíč: model_name, Hodnota: seznam hodnot
    
    # Slovník pro mapování modelů na typy pipeline
    model_to_pipeline_type = {}  # Klíč: model_name, Hodnota: pipeline_type (text, vision, embedding, hybrid)

    for result_dir_path_str in result_dirs:
        result_dir = Path(result_dir_path_str)
        
        # Získání jména konfigurace z názvu adresáře
        config_name_from_dir = result_dir.name 
        
        # Načtení used_config.json pro identifikaci modelů
        models_by_type = {
            "text": "",
            "vision": "",
            "embedding": ""
        }
        
        config_path = result_dir / "used_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    used_config = json.load(f)
                
                # Získáme názvy konkrétních modelů
                for pipeline_type in ["text", "vision", "embedding"]:
                    if pipeline_type in used_config and "model" in used_config[pipeline_type]:
                        # Zde ponecháme celý název modelu včetně verze
                        models_by_type[pipeline_type] = used_config[pipeline_type]["model"]
                        # Zapamatujeme si, že tento model patří do dané pipeline
                        model_to_pipeline_type[models_by_type[pipeline_type]] = pipeline_type
            except Exception as e:
                print(f"  Chyba při načítání konfigurace {config_path}: {e}")

        summary_csv_path = result_dir / "summary_results.csv"
        overall_csv_path = result_dir / "overall_summary_results.csv"

        if summary_csv_path.exists():
            try:
                df_summary = pd.read_csv(summary_csv_path)
                
                if 'Model' in df_summary.columns:
                    # Extrahujeme typ pipeline (EMBEDDED, TEXT, VLM)
                    df_summary['OriginalModel'] = df_summary['Model']  # Zachováme původní název
                    df_summary['PipelineType'] = df_summary['Model'].apply(
                        lambda m: m.split("-")[-1] if "-" in m else m
                    )
                    
                    # Mapování typů pipeline na konkrétní modely
                    pipeline_to_model_map = {
                        "EMBEDDED": models_by_type["embedding"],
                        "TEXT": models_by_type["text"],
                        "VLM": models_by_type["vision"],
                        "HYBRID": "hybrid-text-vlm"  # Standardní název pro hybridní pipeline
                    }
                    
                    # Přidáme názvy konkrétních modelů
                    df_summary['SpecificModel'] = df_summary['PipelineType'].apply(
                        lambda p: pipeline_to_model_map.get(p, "unknown")
                    )
                else:
                    print(f"Varování: Chybí sloupec 'Model' v {summary_csv_path}.")
                    continue
                
                # Ukládáme data podle typu pipeline
                for _, row in df_summary.iterrows():
                    # Klíč pro pipeline
                    key_pipeline = (row['PipelineType'], row['Field'])
                    if key_pipeline not in all_summary_by_pipeline:
                        all_summary_by_pipeline[key_pipeline] = []
                    all_summary_by_pipeline[key_pipeline].append({
                        'Mean_Total': row['Mean_Total'],
                        'Std_Total': row['Std_Total'] if 'Std_Total' in row else 0,
                        'OriginalModel': row.get('OriginalModel', row['PipelineType'])
                    })
                    
                    # Klíč pro konkrétní model
                    specific_model = row['SpecificModel']
                    if specific_model != "unknown" and specific_model:
                        key_specific_model = (specific_model, row['Field'])
                        if key_specific_model not in all_summary_by_specific_model:
                            all_summary_by_specific_model[key_specific_model] = []
                        all_summary_by_specific_model[key_specific_model].append({
                            'Mean_Total': row['Mean_Total'],
                            'Std_Total': row['Std_Total'] if 'Std_Total' in row else 0,
                            'PipelineType': row['PipelineType'],
                            'OriginalModel': row.get('OriginalModel', row['PipelineType'])
                        })
                
                print(f"  Načten soubor: {summary_csv_path}")
            except Exception as e:
                print(f"  Chyba při načítání {summary_csv_path}: {e}")
        else:
            print(f"  Soubor nenalezen: {summary_csv_path}")

        if overall_csv_path.exists():
            try:
                df_overall = pd.read_csv(overall_csv_path)
                
                if 'Model' in df_overall.columns:
                    # Extrahujeme typ pipeline (EMBEDDED, TEXT, VLM)
                    df_overall['OriginalModel'] = df_overall['Model']
                    df_overall['PipelineType'] = df_overall['Model'].apply(
                        lambda m: m.split("-")[-1] if "-" in m else m
                    )
                    
                    # Mapování typů pipeline na konkrétní modely
                    pipeline_to_model_map = {
                        "EMBEDDED": models_by_type["embedding"],
                        "TEXT": models_by_type["text"],
                        "VLM": models_by_type["vision"],
                        "HYBRID": "hybrid-text-vlm"  # Standardní název pro hybridní pipeline
                    }
                    
                    # Přidáme názvy konkrétních modelů
                    df_overall['SpecificModel'] = df_overall['PipelineType'].apply(
                        lambda p: pipeline_to_model_map.get(p, "unknown")
                    )
                else:
                    print(f"Varování: Chybí sloupec 'Model' v {overall_csv_path}.")
                    continue
                
                # Ukládáme data do slovníku podle typu pipeline
                for _, row in df_overall.iterrows():
                    # Pro pipeline
                    key_pipeline = row['PipelineType']
                    if key_pipeline not in all_overall_by_pipeline:
                        all_overall_by_pipeline[key_pipeline] = []
                    all_overall_by_pipeline[key_pipeline].append({
                        'Mean_Total_Overall': row['Mean_Total_Overall'],
                        'Std_Total_Overall': row['Std_Total_Overall'] if 'Std_Total_Overall' in row else 0,
                        'OriginalModel': row.get('OriginalModel', row['PipelineType'])
                    })
                    
                    # Pro konkrétní model
                    specific_model = row['SpecificModel']
                    if specific_model != "unknown" and specific_model:
                        if specific_model not in all_overall_by_specific_model:
                            all_overall_by_specific_model[specific_model] = []
                        all_overall_by_specific_model[specific_model].append({
                            'Mean_Total_Overall': row['Mean_Total_Overall'],
                            'Std_Total_Overall': row['Std_Total_Overall'] if 'Std_Total_Overall' in row else 0,
                            'PipelineType': row['PipelineType'],
                            'OriginalModel': row.get('OriginalModel', specific_model)
                        })
                
                print(f"  Načten soubor: {overall_csv_path}")
            except Exception as e:
                print(f"  Chyba při načítání {overall_csv_path}: {e}")
        else:
            print(f"  Soubor nenalezen: {overall_csv_path}")
    
    # --- 1. Generování grafů podle typu pipeline ---
    print("\nGenerování grafů podle typu pipeline (EMBEDDED, TEXT, VLM)...")
    
    # Vytvoření agregovaných dataframů pro jednotlivá pole podle pipeline
    rows_summary_pipeline = []
    for (pipeline_type, field), values in all_summary_by_pipeline.items():
        # Výpočet průměru a směrodatné odchylky
        mean_values = [v['Mean_Total'] for v in values]
        avg_mean = sum(mean_values) / len(mean_values)
        std_values = [v['Std_Total'] for v in values]
        avg_std = (sum([s**2 for s in std_values]) / len(std_values))**0.5
        
        rows_summary_pipeline.append({
            'Model': pipeline_type,
            'Field': field,
            'Mean_Total': avg_mean,
            'Std_Total': avg_std,
            'OriginalModels': ", ".join(set([v['OriginalModel'] for v in values]))
        })
    
    # Vytvoření agregovaného dataframu pro celkové výsledky podle pipeline
    rows_overall_pipeline = []
    for pipeline_type, values in all_overall_by_pipeline.items():
        mean_values = [v['Mean_Total_Overall'] for v in values]
        avg_mean = sum(mean_values) / len(mean_values)
        std_values = [v['Std_Total_Overall'] for v in values]
        avg_std = (sum([s**2 for s in std_values]) / len(std_values))**0.5
        
        rows_overall_pipeline.append({
            'Model': pipeline_type,
            'Mean_Total_Overall': avg_mean,
            'Std_Total_Overall': avg_std,
            'OriginalModels': ", ".join(set([v['OriginalModel'] for v in values]))
        })
    
    # Vytvoření dataframů z agregovaných dat podle pipeline
    pipeline_summary_df = pd.DataFrame(rows_summary_pipeline) if rows_summary_pipeline else pd.DataFrame()
    pipeline_overall_df = pd.DataFrame(rows_overall_pipeline) if rows_overall_pipeline else pd.DataFrame()
    
    # Generování grafů podle typu pipeline
    created_files_pipeline = generate_comparison_graphs(
        summary_df=pipeline_summary_df,
        overall_df=pipeline_overall_df,
        summary_dir=summary_dir,
        suffix="-pipelines",
        title_prefix="Porovnání úspěšnosti typů pipeline",
        xlabel="Typ pipeline"
    )
    
    # --- 2. Generování grafů podle konkrétních modelů ---
    print("\nGenerování grafů podle konkrétních modelů...")
    
    # Vytvoření agregovaných dataframů pro jednotlivá pole podle konkrétních modelů
    rows_summary_specific_model = []
    for (specific_model, field), values in all_summary_by_specific_model.items():
        # Výpočet průměru a směrodatné odchylky
        mean_values = [v['Mean_Total'] for v in values]
        avg_mean = sum(mean_values) / len(mean_values)
        std_values = [v['Std_Total'] for v in values]
        avg_std = (sum([s**2 for s in std_values]) / len(std_values))**0.5
        
        # Zjistíme typ pipeline pro tento model
        pipeline_type = next((v['PipelineType'] for v in values if 'PipelineType' in v), 'unknown')
        
        rows_summary_specific_model.append({
            'Model': specific_model,
            'Field': field,
            'Mean_Total': avg_mean,
            'Std_Total': avg_std,
            'PipelineType': pipeline_type,  # Přidáme typ pipeline pro barevné rozlišení
            'OriginalModels': ", ".join(set([v['OriginalModel'] for v in values]))
        })
    
    # Vytvoření agregovaného dataframu pro celkové výsledky podle konkrétních modelů
    rows_overall_specific_model = []
    for specific_model, values in all_overall_by_specific_model.items():
        mean_values = [v['Mean_Total_Overall'] for v in values]
        avg_mean = sum(mean_values) / len(mean_values)
        std_values = [v['Std_Total_Overall'] for v in values]
        avg_std = (sum([s**2 for s in std_values]) / len(std_values))**0.5
        
        # Zjistíme typ pipeline pro tento model
        pipeline_type = next((v['PipelineType'] for v in values if 'PipelineType' in v), 'unknown')
        
        rows_overall_specific_model.append({
            'Model': specific_model,
            'Mean_Total_Overall': avg_mean,
            'Std_Total_Overall': avg_std,
            'PipelineType': pipeline_type,  # Přidáme typ pipeline pro barevné rozlišení
            'OriginalModels': ", ".join(set([v['OriginalModel'] for v in values]))
        })
    
    # Vytvoření dataframů z agregovaných dat podle konkrétních modelů
    specific_model_summary_df = pd.DataFrame(rows_summary_specific_model) if rows_summary_specific_model else pd.DataFrame()
    specific_model_overall_df = pd.DataFrame(rows_overall_specific_model) if rows_overall_specific_model else pd.DataFrame()
    
    # Generování grafů podle konkrétních modelů s barevným rozlišením podle typu pipeline
    created_files_specific_model = generate_comparison_graphs_with_colors(
        summary_df=specific_model_summary_df,
        overall_df=specific_model_overall_df,
        summary_dir=summary_dir,
        suffix="-models",
        title_prefix="Porovnání úspěšnosti konkrétních modelů",
        xlabel="Model"
    )
    
    # Sloučení seznamů vytvořených souborů
    created_files = created_files_pipeline + created_files_specific_model
    unique_created_files = sorted(list(set(created_files)))
    
    # Závěrečný výpis
    if is_final:
        print(f"\nFinální souhrnné výsledky byly uloženy do adresáře: {summary_dir}")
        print(f"Vytvořeno/aktualizováno {len(unique_created_files)} souborů.")
        if unique_created_files:
             print("\nVytvořené soubory:")
             for fname in unique_created_files:
                 print(f"  - {fname}")
    else:
        print(f"\nPrůběžné výsledky byly aktualizovány v adresáři: {summary_dir}")
        print(f"Vytvořeno/aktualizováno {len(unique_created_files)} souborů.")
        
    # Výpis aktuálního pořadí typů pipeline
    if not pipeline_overall_df.empty and 'Mean_Total_Overall' in pipeline_overall_df.columns:
        print("\nAktuální pořadí typů pipeline (podle průměrné celkové úspěšnosti):")
        df_overall_sorted_pipeline = pipeline_overall_df.sort_values('Mean_Total_Overall', ascending=False)
        for i, (_, row) in enumerate(df_overall_sorted_pipeline.iterrows(), 1):
            pipeline_disp = row['Model']
            mean_disp = row['Mean_Total_Overall']
            std_disp = row['Std_Total_Overall'] if 'Std_Total_Overall' in row and pd.notna(row['Std_Total_Overall']) else 0
            print(f"{i}. {pipeline_disp}: {mean_disp:.4f} ± {std_disp:.4f}")
            
    # Výpis aktuálního pořadí konkrétních modelů
    if not specific_model_overall_df.empty and 'Mean_Total_Overall' in specific_model_overall_df.columns:
        print("\nAktuální pořadí konkrétních modelů (podle průměrné celkové úspěšnosti):")
        df_overall_sorted_model = specific_model_overall_df.sort_values('Mean_Total_Overall', ascending=False)
        for i, (_, row) in enumerate(df_overall_sorted_model.iterrows(), 1):
            model_disp = row['Model']
            mean_disp = row['Mean_Total_Overall']
            std_disp = row['Std_Total_Overall'] if 'Std_Total_Overall' in row and pd.notna(row['Std_Total_Overall']) else 0
            pipeline_type = row.get('PipelineType', 'neznámá pipeline')
            print(f"{i}. {model_disp}: {mean_disp:.4f} ± {std_disp:.4f} (typ: {pipeline_type})")


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
    required_overall_cols = ['Model', 'Mean_Total_Overall', 'Std_Total_Overall']
    
    if summary_df.empty and overall_df.empty:
        print(f"Nebylo možné spojit žádná data pro grafy{suffix}.")
        return created_files

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
            
            bars = plt.bar(x_pos, means, color='skyblue')
            plt.errorbar(x_pos, means, yerr=errors, fmt='none', ecolor='black', capsize=5)

            plt.title(f'{title_prefix} - {field} (průměr ±1σ)')
            plt.xlabel(xlabel)
            plt.ylabel('Průměrná podobnost')
            plt.xticks(x_pos, models, rotation=45, ha="right")
            plt.ylim(0, max(1.05, (means + errors).max() * 1.1))
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
                for col in ['OriginalModels', 'PipelineTypes', 'PipelineType']:
                    if col in df_field.columns:
                        field_columns.append(col)
                df_field[field_columns].to_csv(output_csv, index=False, float_format='%.4f')
                created_files.append(output_csv.name)
            except Exception as e:
                 print(f"Chyba při ukládání CSV {output_csv.name}: {e}")

    else:
        print(f"Přeskakuji generování grafů a CSV pro jednotlivá pole{suffix} - chybí data nebo sloupce.")

    # --- Vytvoření grafu celkových výsledků ---
    if not overall_df.empty and all(col in overall_df.columns for col in required_overall_cols):
        
        df_overall_sorted = overall_df.sort_values(by='Mean_Total_Overall', ascending=False)
        
        means_overall = df_overall_sorted['Mean_Total_Overall']
        errors_overall = df_overall_sorted['Std_Total_Overall'].fillna(0)
        models_overall = df_overall_sorted['Model']
        
        x_pos_overall = np.arange(len(models_overall))

        plt.figure(figsize=(max(10, len(models_overall)*0.8), 6))
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
            for col in ['OriginalModels', 'PipelineTypes', 'PipelineType']:
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
    required_summary_cols = ['Field', 'Model', 'Mean_Total', 'Std_Total', 'PipelineType']
    required_overall_cols = ['Model', 'Mean_Total_Overall', 'Std_Total_Overall', 'PipelineType']
    
    if summary_df.empty and overall_df.empty:
        print(f"Nebylo možné spojit žádná data pro grafy{suffix}.")
        return created_files

    # Definice barev pro typy pipeline
    pipeline_colors = {
        'TEXT': 'royalblue',
        'EMBEDDED': 'green',
        'VLM': 'firebrick',
        'HYBRID': 'purple'  # Fialová barva pro hybridní pipeline
    }
    
    # --- Vytvoření grafů pro jednotlivá pole ---
    if not summary_df.empty and all(col in summary_df.columns for col in required_summary_cols[:4]):
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
                Patch(facecolor=pipeline_colors['HYBRID'], label='Hybrid model (Text+VLM)')
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
                field_columns = ['Model', 'Mean_Total', 'Std_Total', 'PipelineType']
                # Přidáme další sloupce, pokud existují
                for col in ['OriginalModels']:
                    if col in df_field.columns:
                        field_columns.append(col)
                df_field[field_columns].to_csv(output_csv, index=False, float_format='%.4f')
                created_files.append(output_csv.name)
            except Exception as e:
                 print(f"Chyba při ukládání CSV {output_csv.name}: {e}")

    else:
        print(f"Přeskakuji generování grafů a CSV pro jednotlivá pole{suffix} - chybí data nebo sloupce.")

    # --- Vytvoření grafu celkových výsledků ---
    if not overall_df.empty and all(col in overall_df.columns for col in required_overall_cols[:3]):
        
        df_overall_sorted = overall_df.sort_values(by='Mean_Total_Overall', ascending=False)
        
        means_overall = df_overall_sorted['Mean_Total_Overall']
        errors_overall = df_overall_sorted['Std_Total_Overall'].fillna(0)
        models_overall = df_overall_sorted['Model']
        
        x_pos_overall = np.arange(len(models_overall))

        plt.figure(figsize=(max(10, len(models_overall)*0.8), 6))
        
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
            Patch(facecolor=pipeline_colors['HYBRID'], label='Hybrid model (Text+VLM)')
        ]
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
            for col in ['OriginalModels']:
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
            "--output-dir", str(config_dir)
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