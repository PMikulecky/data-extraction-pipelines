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
        "--models", "embedded", "vlm", "text",
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
        "embedded_comparison.json",
        "vlm_comparison.json",
        "text_comparison.json",  # Přidáno porovnání textové pipeline
        "semantic_comparison_results.json",
        "embedded_comparison_semantic.json",
        "vlm_comparison_semantic.json",
        "text_comparison_semantic.json",  # Přidáno sémantické porovnání textové pipeline
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
    
    # <<< Změna: Adresář pro souhrnné výsledky v hlavním adresáři běhu >>>
    summary_dir = get_run_results_dir() / "final_comparison"
    summary_dir.mkdir(parents=True, exist_ok=True)
    # <<< Konec změny >>>
    
    all_summaries = []
    all_overalls = []
    model_names = []

    print(f"Prohledávám {len(result_dirs)} adresářů s výsledky pro summary a overall CSV...")

    for result_dir_path_str in result_dirs:
        result_dir = Path(result_dir_path_str)
        
        # Získání jména modelu z názvu adresáře nebo konfigurace
        # Jednoduchý přístup: použít název adresáře
        model_name = result_dir.name 
        # Lze vylepšit načtením z used_config.json, pokud je potřeba specifičtější název
        config_name_from_dir = result_dir.name # Použijeme název adresáře jako základ

        summary_csv_path = result_dir / "summary_results.csv"
        overall_csv_path = result_dir / "overall_summary_results.csv"

        found_summary = False
        if summary_csv_path.exists():
            try:
                df_summary = pd.read_csv(summary_csv_path)
                # <<< Změna: Kombinace názvu konfigurace a modelu z CSV >>>
                # Očekáváme sloupec 'Model' v CSV (např. EMBEDDED, VLM, TEXT)
                if 'Model' in df_summary.columns:
                    df_summary['Model'] = df_summary['Model'].apply(lambda m: f"{config_name_from_dir}-{m}")
                else:
                    print(f"Varování: Chybí sloupec 'Model' v {summary_csv_path}. Používám pouze název adresáře.")
                    df_summary['Model'] = config_name_from_dir
                # <<< Konec změny >>>
                all_summaries.append(df_summary)
                found_summary = True
                print(f"  Načten soubor: {summary_csv_path}")
            except Exception as e:
                print(f"  Chyba při načítání {summary_csv_path}: {e}")
        else:
            print(f"  Soubor nenalezen: {summary_csv_path}")

        found_overall = False
        if overall_csv_path.exists():
            try:
                df_overall = pd.read_csv(overall_csv_path)
                 # <<< Změna: Kombinace názvu konfigurace a modelu z CSV >>>
                 # Očekáváme sloupec 'Model' v CSV (např. EMBEDDED, VLM, TEXT)
                if 'Model' in df_overall.columns:
                    df_overall['Model'] = df_overall['Model'].apply(lambda m: f"{config_name_from_dir}-{m}")
                else:
                    print(f"Varování: Chybí sloupec 'Model' v {overall_csv_path}. Používám pouze název adresáře.")
                    df_overall['Model'] = config_name_from_dir
                # <<< Konec změny >>>
                all_overalls.append(df_overall)
                found_overall = True
                print(f"  Načten soubor: {overall_csv_path}")
            except Exception as e:
                print(f"  Chyba při načítání {overall_csv_path}: {e}")
        else:
             print(f"  Soubor nenalezen: {overall_csv_path}")

        if found_summary or found_overall:
             # Přidáváme pouze název konfigurace, unikátní modely budou v DataFrame
             if config_name_from_dir not in model_names: 
                model_names.append(config_name_from_dir)


    if not all_summaries and not all_overalls:
        print("Nenalezeny žádné výsledky (summary_results.csv nebo overall_summary_results.csv) pro porovnání.")
        return

    # Spojení DataFrames
    final_summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    final_overall_df = pd.concat(all_overalls, ignore_index=True) if all_overalls else pd.DataFrame()
    
    if final_summary_df.empty and final_overall_df.empty:
        print("Nebylo možné spojit žádná data z CSV souborů.")
        return
        
    print(f"Nalezeno {len(model_names)} unikátních modelů/konfigurací pro porovnání.")
    
    # Ujistíme se, že máme potřebné sloupce (názvy podle src/main.py::visualize_results)
    required_summary_cols = ['Field', 'Model', 'Mean_Total', 'Std_Total']
    required_overall_cols = ['Model', 'Mean_Total_Overall', 'Std_Total_Overall']

    # Kontrola sloupců
    if not final_summary_df.empty and not all(col in final_summary_df.columns for col in required_summary_cols):
         print(f"Varování: Chybí některé požadované sloupce v summary_results.csv ({required_summary_cols}). Grafy polí nemusí být kompletní.")
         # Můžeme zkusit pokračovat s dostupnými sloupci, nebo zde skončit
         # Prozatím budeme pokračovat a matplotlib/pandas si poradí s chybějícími daty (NaN)

    if not final_overall_df.empty and not all(col in final_overall_df.columns for col in required_overall_cols):
         print(f"Varování: Chybí některé požadované sloupce v overall_summary_results.csv ({required_overall_cols}). Celkový graf nemusí být vytvořen.")
         # Prozatím budeme pokračovat


    # Počítadlo vytvořených souborů
    created_files = []

    # --- Vytvoření grafů pro jednotlivá pole ---
    if not final_summary_df.empty and all(col in final_summary_df.columns for col in required_summary_cols):
        metadata_fields = final_summary_df['Field'].unique()
        
        for field in metadata_fields:
            df_field = final_summary_df[final_summary_df['Field'] == field].sort_values(by='Model')
            
            if df_field.empty:
                continue

            plt.figure(figsize=(max(10, len(df_field['Model'])*0.8), 6)) # Dynamická šířka
            
            means = df_field['Mean_Total']
            errors = df_field['Std_Total'].fillna(0) # Nahradit NaN nulou pro errorbar
            models = df_field['Model']
            
            x_pos = np.arange(len(models))
            
            bars = plt.bar(x_pos, means, color='skyblue') 
            # Přidání chybových úseček
            plt.errorbar(x_pos, means, yerr=errors, fmt='none', ecolor='black', capsize=5)

            # Přidání hodnot nad sloupce (volitelné, může být nepřehledné s error bary)
            # for i, bar in enumerate(bars):
            #     height = bar.get_height()
            #     plt.text(bar.get_x() + bar.get_width()/2., height + errors.iloc[i] * 1.1 , # Pozice nad error bar
            #             f'{height:.3f}', # Zobrazit s desetinnými místy
            #             ha='center', va='bottom', fontsize=9)

            plt.title(f'Porovnání úspěšnosti modelů - {field} (průměr ±1σ)')
            plt.xlabel('Model / Konfigurace')
            plt.ylabel('Průměrná podobnost')
            plt.xticks(x_pos, models, rotation=45, ha="right") # Použít názvy modelů jako popisky osy x
            plt.ylim(0, max(1.05, (means + errors).max() * 1.1)) # Dynamický horní limit Y osy
            plt.tight_layout()
            
            # <<< Změna: Uložení do summary_dir >>>
            output_file = summary_dir / f"{field}_comparison.png"
            # <<< Konec změny >>>
            try:
                plt.savefig(output_file)
                created_files.append(output_file.name)
            except Exception as e:
                print(f"Chyba při ukládání grafu {output_file.name}: {e}")
            plt.close()
            
            # Uložení CSV pro dané pole (volitelné, lze nahradit jedním souhrnným CSV)
            # <<< Změna: Uložení do summary_dir >>>
            output_csv = summary_dir / f"{field}_comparison.csv"
            # <<< Konec změny >>>
            try:
                # Ukládáme jen relevantní sloupce
                df_field[['Model', 'Mean_Total', 'Std_Total']].to_csv(output_csv, index=False, float_format='%.4f')
                created_files.append(output_csv.name)
            except Exception as e:
                 print(f"Chyba při ukládání CSV {output_csv.name}: {e}")

    else:
        print("Přeskakuji generování grafů a CSV pro jednotlivá pole - chybí data nebo sloupce v summary_results.")


    # --- Vytvoření grafu celkových výsledků ---
    if not final_overall_df.empty and all(col in final_overall_df.columns for col in required_overall_cols):
        
        df_overall_sorted = final_overall_df.sort_values(by='Model')
        
        means_overall = df_overall_sorted['Mean_Total_Overall']
        errors_overall = df_overall_sorted['Std_Total_Overall'].fillna(0)
        models_overall = df_overall_sorted['Model']
        
        x_pos_overall = np.arange(len(models_overall))

        plt.figure(figsize=(max(10, len(models_overall)*0.8), 6)) # Dynamická šířka
        bars_overall = plt.bar(x_pos_overall, means_overall, color='lightcoral')
        # Přidání chybových úseček
        plt.errorbar(x_pos_overall, means_overall, yerr=errors_overall, fmt='none', ecolor='black', capsize=5)

        # Přidání hodnot nad sloupce (volitelné)
        # for i, bar in enumerate(bars_overall):
        #     height = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width()/2., height + errors_overall.iloc[i] * 1.1,
        #             f'{height:.3f}',
        #             ha='center', va='bottom', fontsize=9)

        plt.title('Celková úspěšnost modelů / konfigurací (průměr ±1σ)')
        plt.xlabel('Model / Konfigurace')
        plt.ylabel('Průměrná celková podobnost')
        plt.xticks(x_pos_overall, models_overall, rotation=45, ha="right")
        plt.ylim(0, max(1.05, (means_overall + errors_overall).max() * 1.1))
        plt.tight_layout()
        
        # <<< Změna: Uložení do summary_dir >>>
        output_file = summary_dir / "overall_comparison.png"
        # <<< Konec změny >>>
        try:
            plt.savefig(output_file)
            created_files.append(output_file.name)
        except Exception as e:
            print(f"Chyba při ukládání grafu {output_file.name}: {e}")
        plt.close()

        # Uložení celkového CSV (volitelné, lze nahradit jedním souhrnným)
        # <<< Změna: Uložení do summary_dir >>>
        output_csv_overall = summary_dir / "overall_comparison.csv"
        # <<< Konec změny >>>
        try:
            # Ukládáme jen relevantní sloupce
            df_overall_sorted[['Model', 'Mean_Total_Overall', 'Std_Total_Overall']].to_csv(output_csv_overall, index=False, float_format='%.4f')
            created_files.append(output_csv_overall.name)
        except Exception as e:
             print(f"Chyba při ukládání CSV {output_csv_overall.name}: {e}")
             
        # Uložení spojených finálních dat (volitelné, ale užitečné)
        try:
            # <<< Změna: Uložení do summary_dir >>>
            final_summary_df.to_csv(summary_dir / "final_summary_all_fields.csv", index=False, float_format='%.4f')
            final_overall_df.to_csv(summary_dir / "final_overall_all_models.csv", index=False, float_format='%.4f')
            # <<< Konec změny >>>
            created_files.append("final_summary_all_fields.csv")
            created_files.append("final_overall_all_models.csv")
        except Exception as e:
            print(f"Chyba při ukládání finálních spojených CSV: {e}")

    else:
         print("Přeskakuji generování celkového grafu a CSV - chybí data nebo sloupce v overall_summary_results.")


    # Závěrečný výpis
    unique_created_files = sorted(list(set(created_files))) # Odstranění duplicit a seřazení
    
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
        
    # Výpis aktuálního pořadí modelů z finálních dat, pokud jsou dostupná
    if not final_overall_df.empty and 'Mean_Total_Overall' in final_overall_df.columns:
        print("\nAktuální pořadí modelů (podle průměrné celkové úspěšnosti):")
        df_overall_sorted_final = final_overall_df.sort_values('Mean_Total_Overall', ascending=False)
        for i, (_, row) in enumerate(df_overall_sorted_final.iterrows(), 1):
            model_disp = row['Model']
            mean_disp = row['Mean_Total_Overall']
            std_disp = row['Std_Total_Overall'] if 'Std_Total_Overall' in row and pd.notna(row['Std_Total_Overall']) else 0
            print(f"{i}. {model_disp}: {mean_disp:.4f} ± {std_disp:.4f}")
    elif not final_overall_df.empty:
         print("\nNebylo možné určit pořadí modelů - chybí sloupec 'Mean_Total_Overall'.")


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


def main():
    """Hlavní funkce skriptu."""
    args = parse_args()
    
    # Zpracování parametru --graphs-only
    if args.graphs_only:
        generate_graphs_only(args.results_dir)
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