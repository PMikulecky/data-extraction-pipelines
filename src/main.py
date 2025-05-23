#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hlavní skript pro spuštění procesu extrakce metadat z PDF souborů a porovnání výsledků.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
import argparse
import time
import re  # Přidáno pro práci s regulárními výrazy
import numpy as np # Přidáno pro výpočty
import logging # Přidáno pro logování v pomocných funkcích
from datetime import datetime # Přidán import
import subprocess

# Import lokálních modulů
from src.data_preparation import filter_papers_with_valid_doi_and_references as filter_papers_with_valid_doi
from src.pdf_downloader import download_pdfs_for_filtered_papers
from src.models.embedded_pipeline import extract_metadata_from_pdfs as extract_with_embedded
from src.models.vlm_pipeline import extract_metadata_from_pdfs as extract_with_vlm
from src.models.text_pipeline import extract_metadata_from_pdfs as extract_with_text
from src.models.multimodal_pipeline import extract_metadata_from_pdfs as extract_with_multimodal
from src.utils.metadata_comparator import compare_all_metadata, calculate_overall_metrics, MetadataComparator
from src.utils.semantic_comparison import process_comparison_files
# Import konfiguračního modulu
from src.models.config.model_config import load_config, get_config
from src.config.runtime_config import set_run_results_dir, get_run_results_dir
# <<< Změna: Import runtime_config >>>
# <<< Konec změny >>>

# Načtení proměnných prostředí
load_dotenv()

# Vyčištění API klíčů - přidáno pro odstranění problémů s neviditelnými znaky
def clean_api_keys():
    """Vyčistí API klíče od mezer a neviditelných znaků"""
    # Anthropic API klíč
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        # Odstranění mezer, tabulátorů a nových řádků
        anthropic_key = re.sub(r'\s+', '', anthropic_key)
        # Odstranění jiných netisknutelných znaků
        anthropic_key = ''.join(char for char in anthropic_key if char.isprintable())
        # Nastavení vyčištěného klíče zpět do prostředí
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        print(f"Anthropic API klíč vyčištěn, začíná: {anthropic_key[:10]}..., délka: {len(anthropic_key)} znaků")
    
    # OpenAI API klíč
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        # Odstranění mezer, tabulátorů a nových řádků
        openai_key = re.sub(r'\s+', '', openai_key)
        # Odstranění jiných netisknutelných znaků
        openai_key = ''.join(char for char in openai_key if char.isprintable())
        # Nastavení vyčištěného klíče zpět do prostředí
        os.environ["OPENAI_API_KEY"] = openai_key
        print(f"OpenAI API klíč vyčištěn, začíná: {openai_key[:10]}..., délka: {len(openai_key)} znaků")

# Vyčištění API klíčů hned po načtení
clean_api_keys()

# Definice cest
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_CSV = DATA_DIR / "papers.csv"
FILTERED_CSV = DATA_DIR / "papers-filtered.csv"
PDF_DIR = DATA_DIR / "pdfs"
# RESULTS_DIR = BASE_DIR / "results" # Odstraněno - použijeme get_run_results_dir()
CONFIG_DIR = BASE_DIR / "config"
MODELS_JSON = CONFIG_DIR / "models.json"

# Vytvoření adresářů, pokud neexistují
PDF_DIR.mkdir(parents=True, exist_ok=True)
# RESULTS_DIR.mkdir(parents=True, exist_ok=True) # Odstraněno

# Načtení konfigurace
# Budeme kontrolovat, zda byla již konfigurace načtena nebo zda má být načtena z argumentu v main()
if MODELS_JSON.exists():
    print(f"Výchozí konfigurační soubor {MODELS_JSON} nalezen (může být přepsán parametrem --config)")
else:
    print(f"Výchozí konfigurační soubor {MODELS_JSON} neexistuje (může být zadán parametrem --config)")

# Vlastní načtení konfigurace bude provedeno v main() na základě argumentu --config

def prepare_reference_data(csv_path):
    """
    Připraví referenční data z CSV souboru.
    
    Args:
        csv_path (str): Cesta k CSV souboru
        
    Returns:
        dict: Referenční data pro každou práci
    """
    print(f"Načítám referenční data z {csv_path}...")
    df = pd.read_csv(csv_path)
    
    reference_data = {}
    
    for _, row in df.iterrows():
        paper_id = str(row['id'])
        
        # Extrakce metadat z CSV
        metadata = {
            'title': row.get('dc.title[en]', row.get('dc.title[cs]', '')),
            'authors': row.get('dc.contributor.author', ''),
            'abstract': row.get('dc.description.abstract[en]', row.get('dc.description.abstract[cs]', '')),
            'keywords': row.get('dc.subject[en]', row.get('dc.subject[cs]', '')),
            'doi': row.get('dc.identifier.doi', ''),
            'year': row.get('dc.date.issued', ''),
            'journal': row.get('dc.relation.ispartof', ''),
            'volume': row.get('utb.relation.volume', ''),
            'issue': row.get('utb.relation.issue', ''),
            'pages': f"{row.get('dc.citation.spage', '')}-{row.get('dc.citation.epage', '')}",
            'publisher': row.get('dc.publisher', '')
            # 'references' pole vyřazeno z extrakce
        }
        
        reference_data[paper_id] = metadata
    
    return reference_data


def run_extraction_pipeline(limit=None, models=None, year_filter=None, skip_download=False, skip_semantic=False, force_extraction=False, include_references=False):
    """
    Spustí celý proces extrakce metadat a porovnání výsledků.
    
    Args:
        limit (int, optional): Omezení počtu zpracovaných souborů
        models (list, optional): Seznam modelů k použití
        year_filter (list, optional): Seznam let pro filtrování článků
        skip_download (bool, optional): Přeskočí stahování PDF souborů
        skip_semantic (bool, optional): Přeskočí sémantické porovnání
        force_extraction (bool): Vynutí novou extrakci metadat
        include_references (bool): Zahrne reference do textu pro embedded pipeline
        
    Returns:
        tuple: Výsledky porovnání, časy, tokeny
    """
    # Zobrazení API klíčů před spuštěním extrakce
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    print("\n=== Kontrola API klíčů před spuštěním extrakce ===")
    if anthropic_key:
        print(f"Anthropic API klíč je nastaven, začíná: {anthropic_key[:10]}..., délka: {len(anthropic_key)} znaků")
        # Kontrola, zda klíč obsahuje netisknutelné znaky
        if any(not c.isprintable() for c in anthropic_key):
            print("VAROVÁNÍ: Anthropic API klíč obsahuje netisknutelné znaky!")
        # Kontrola, zda klíč obsahuje bílé znaky
        if any(c.isspace() for c in anthropic_key):
            print("VAROVÁNÍ: Anthropic API klíč obsahuje bílé znaky!")
    else:
        print("Anthropic API klíč není nastaven!")
    
    if openai_key:
        print(f"OpenAI API klíč je nastaven, začíná: {openai_key[:10]}..., délka: {len(openai_key)} znaků")
    else:
        print("OpenAI API klíč není nastaven!")
    
    # Výchozí modely
    if models is None:
        models = ['embedded']
    
    # 1. Příprava dat
    if not os.path.exists(FILTERED_CSV):
        print("Filtruji akademické práce s validním DOI...")
        filter_papers_with_valid_doi(INPUT_CSV, FILTERED_CSV)
    
    # 2. Stažení PDF souborů
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    if not pdf_files and not skip_download:
        print("Stahuji PDF soubory...")
        # Pokud je zadán filtr podle roku, použijeme ho
        if year_filter:
            print(f"Filtrování článků podle let: {year_filter}")
            # Načtení dat
            df = pd.read_csv(FILTERED_CSV)
            # Filtrování podle roku
            df = df[df['dc.date.issued'].astype(str).isin([str(year) for year in year_filter])]
            # Uložení filtrovaných dat do dočasného souboru
            temp_csv = DATA_DIR / "papers-filtered-by-year.csv"
            df.to_csv(temp_csv, index=False)
            # Stažení PDF souborů pro filtrované články
            download_pdfs_for_filtered_papers(temp_csv, PDF_DIR, limit=limit)
        else:
            download_pdfs_for_filtered_papers(FILTERED_CSV, PDF_DIR, limit=limit)
    elif skip_download:
        print("Stahování PDF souborů přeskočeno (--skip-download).")
    
    # 3. Příprava referenčních dat
    reference_data = prepare_reference_data(FILTERED_CSV)
    
    # 4. Extrakce metadat pomocí různých modelů
    results = {}
    all_timings = {} # Slovník pro sběr časů
    all_token_usages = {} # Nový slovník pro sběr tokenů

    for model in models:
        output_file = get_run_results_dir() / f"{model}_results.json"
        api_key = None # Resetovat pro každý model
        provider_name = None
        model_name_extracted = None
        current_results = None
        current_timings = None
        current_token_usages = None # Inicializace pro tokeny

        if model == 'embedded':
            print("\n=== Extrakce metadat pomocí Embedded pipeline ===")
            try:
                config = get_config()
                text_config = config.get_text_config()
                provider_name = text_config.get("provider")
                model_name_extracted = text_config.get("model")
                # <<< Přidáno: Načtení konfigurace embeddingů >>>
                embedding_config = config.get_embedding_config()
                embedding_provider_name = embedding_config.get("provider")
                embedding_model_name = embedding_config.get("model")
                # <<< Konec přidání >>>
                print(f"Použití providera {provider_name} a modelu {model_name_extracted} z konfigurace")

                # <<< Změna: Určení správných API klíčů pro text a embedding >>>
                text_api_key = None
                if provider_name == "anthropic": text_api_key = anthropic_key
                elif provider_name == "openai": text_api_key = openai_key
                if text_api_key: print(f"Předávám explicitně {provider_name} API klíč pro text...")

                embedding_api_key = None
                if embedding_provider_name == "anthropic": embedding_api_key = anthropic_key
                elif embedding_provider_name == "openai": embedding_api_key = openai_key
                if embedding_api_key: print(f"Předávám explicitně {embedding_provider_name} API klíč pro embedding...")
                # <<< Konec změny >>>

                # <<< Změna: Zachytit results, timings a token_usages >>>
                current_results, current_timings, current_token_usages = extract_with_embedded(
                    pdf_dir=PDF_DIR,
                    limit=limit,
                    force_extraction=force_extraction,
                    provider_name=provider_name,
                    model_name=model_name_extracted,
                    text_api_key=text_api_key,
                    embedding_api_key=embedding_api_key,
                    embedding_provider_name=embedding_provider_name,
                    embedding_model_name=embedding_model_name,
                    exclude_references=not include_references
                )
                # <<< Konec změny >>>
            except Exception as e:
                import traceback
                print(f"Chyba při extrakci metadat pomocí Embedded pipeline: {e}")
                print(f"Podrobnosti chyby: {traceback.format_exc()}")
                current_results = {}; current_timings = {}; current_token_usages = {}

        elif model == 'vlm':
            print("\n=== Extrakce metadat pomocí VLM pipeline ===")
            try:
                config = get_config()
                vision_config = config.get_vision_config()
                provider_name = vision_config.get("provider")
                model_name_extracted = vision_config.get("model")
                print(f"Použití providera {provider_name} a modelu {model_name_extracted} z konfigurace")

                # <<< Změna: Určení správného API klíče pro VLM >>>
                vision_api_key = None
                if provider_name == "anthropic": vision_api_key = anthropic_key
                elif provider_name == "openai": vision_api_key = openai_key
                if vision_api_key: print(f"Předávám explicitně {provider_name} API klíč...")
                # <<< Konec změny >>>

                # <<< Změna: Zachytit results, timings a token_usages >>>
                current_results, current_timings, current_token_usages = extract_with_vlm(
                    pdf_dir=PDF_DIR,
                    limit=limit,
                    force_extraction=force_extraction,
                    provider_name=provider_name,
                    model_name=model_name_extracted,
                    api_key=vision_api_key
                )
                # <<< Konec změny >>>
            except Exception as e:
                import traceback
                print(f"Chyba při extrakci metadat pomocí VLM pipeline: {e}")
                print(f"Podrobnosti chyby: {traceback.format_exc()}")
                current_results = {}; current_timings = {}; current_token_usages = {}

        elif model == 'text':
            print("\n=== Extrakce metadat pomocí textové pipeline ===")
            try:
                config = get_config()
                text_config = config.get_text_config()
                provider_name = text_config.get("provider")
                model_name_extracted = text_config.get("model")
                print(f"Použití providera {provider_name} a modelu {model_name_extracted} z konfigurace")

                # <<< Změna: Určení správného API klíče pro text >>>
                text_api_key = None
                if provider_name == "anthropic": text_api_key = anthropic_key
                elif provider_name == "openai": text_api_key = openai_key
                if text_api_key: print(f"Předávám explicitně {provider_name} API klíč...")
                # <<< Konec změny >>>

                # <<< Změna: Zachytit results, timings a token_usages >>>
                current_results, current_timings, current_token_usages = extract_with_text(
                    pdf_dir=PDF_DIR,
                    limit=limit,
                    force_extraction=force_extraction,
                    provider_name=provider_name,
                    model_name=model_name_extracted,
                    api_key=text_api_key
                )
                # <<< Konec změny >>>
            except Exception as e:
                import traceback
                print(f"Chyba při extrakci metadat pomocí textové pipeline: {e}")
                print(f"Podrobnosti chyby: {traceback.format_exc()}")
                current_results = {}; current_timings = {}; current_token_usages = {}
                
        elif model == 'multimodal':
            print("\n=== Extrakce metadat pomocí Multimodální pipeline ===")
            try:
                config = get_config()
                multimodal_config = config.get_multimodal_config()
                provider_name = multimodal_config.get("provider")
                model_name_extracted = multimodal_config.get("model")
                print(f"Použití providera {provider_name} a modelu {model_name_extracted} z konfigurace")

                # Určení správného API klíče pro multimodální model
                multimodal_api_key = None
                if provider_name == "anthropic": multimodal_api_key = anthropic_key
                elif provider_name == "openai": multimodal_api_key = openai_key
                elif provider_name == "gemini": multimodal_api_key = os.getenv("GEMINI_API_KEY", "")
                if multimodal_api_key: print(f"Předávám explicitně {provider_name} API klíč...")

                # Volání extrakce pomocí multimodální pipeline
                current_results, current_timings, current_token_usages = extract_with_multimodal(
                    pdf_dir=PDF_DIR,
                    limit=limit,
                    force_extraction=force_extraction,
                    provider_name=provider_name,
                    model_name=model_name_extracted,
                    api_key=multimodal_api_key
                )
            except Exception as e:
                import traceback
                print(f"Chyba při extrakci metadat pomocí Multimodální pipeline: {e}")
                print(f"Podrobnosti chyby: {traceback.format_exc()}")
                current_results = {}; current_timings = {}; current_token_usages = {}
        
        # Uložit výsledky, časy a tokeny
        results[model] = current_results if current_results is not None else {}
        all_timings[model] = current_timings if current_timings is not None else {}
        all_token_usages[model] = current_token_usages if current_token_usages is not None else {} # Uložit tokeny

    # 5. Porovnání výsledků s referenčními daty
    comparison_results = {}
    for model_name, model_results in results.items():
        # Přeskočit modely bez výsledků
        if not model_results:
            print(f"Model {model_name} nemá žádné výsledky, přeskakuji porovnání.")
            continue
            
        print(f"\n=== Porovnávání výsledků modelu {model_name} ===")
        comparison = compare_all_metadata(model_results, reference_data)
        metrics = calculate_overall_metrics(comparison)

        comparison_results[model_name] = {
            'comparison': comparison,
            'metrics': metrics
        }

        # Uložení výsledků porovnání
        comparison_output = get_run_results_dir() / f"{model_name}_comparison.json"
        try:
            with open(comparison_output, 'w', encoding='utf-8') as f:
                json.dump(comparison_results[model_name], f, ensure_ascii=False, indent=2)
            print(f"Výsledky porovnání pro {model_name} uloženy do {comparison_output}")
        except Exception as e:
             print(f"Chyba při ukládání porovnání pro {model_name}: {e}")

    # 6. Sémantické porovnání (volitelné, pokud není přeskočeno a existují porovnávací soubory)
    semantic_comparison_performed = False
    comparison_files_for_plotting = {} # Slovník s finálními cestami pro vizualizaci

    if not skip_semantic and comparison_results:
        print("\n=== Sémantické porovnání výsledků ===")
        try:
            # Připravíme cesty k původním souborům porovnání
            vlm_comp_path = get_run_results_dir() / "vlm_comparison.json" if "vlm" in comparison_results else None
            emb_comp_path = get_run_results_dir() / "embedded_comparison.json" if "embedded" in comparison_results else None
            txt_comp_path = get_run_results_dir() / "text_comparison.json" if "text" in comparison_results else None
            mul_comp_path = get_run_results_dir() / "multimodal_comparison.json" if "multimodal" in comparison_results else None

            # Odstraníme cesty k neexistujícím souborům
            vlm_comp_path_str = str(vlm_comp_path) if vlm_comp_path and vlm_comp_path.exists() else None
            emb_comp_path_str = str(emb_comp_path) if emb_comp_path and emb_comp_path.exists() else None
            txt_comp_path_str = str(txt_comp_path) if txt_comp_path and txt_comp_path.exists() else None
            mul_comp_path_str = str(mul_comp_path) if mul_comp_path and mul_comp_path.exists() else None

            if not emb_comp_path_str and not vlm_comp_path_str and not txt_comp_path_str and not mul_comp_path_str:
                 print("Nebyly nalezeny žádné soubory *_comparison.json pro sémantické zpracování.")
            else:
                # Volání funkce pro sémantické porovnání
                # Ta uloží *_comparison_semantic.json soubory do get_run_results_dir()
                updated_semantic_data = process_comparison_files(
                    output_dir=get_run_results_dir(), # Výstupní adresář je adresář tohoto běhu
                    vlm_comparison_path=vlm_comp_path_str,
                    embedded_comparison_path=emb_comp_path_str,
                    text_comparison_path=txt_comp_path_str,
                    multimodal_comparison_path=mul_comp_path_str
                )
                
                # Pokud porovnání proběhlo, nastavíme flag a aktualizujeme cesty
                if updated_semantic_data:
                    semantic_comparison_performed = True
                    print("Sémantické porovnání dokončeno.")
                    # Aktualizujeme cesty pro vizualizaci na sémantické verze
                    for model_key in comparison_results.keys():
                        semantic_file = get_run_results_dir() / f"{model_key}_comparison_semantic.json"
                        if semantic_file.exists():
                            comparison_files_for_plotting[model_key] = semantic_file
                        else:
                             # Pokud sémantický soubor nevznikl (např. chyba), použijeme původní
                             original_file = get_run_results_dir() / f"{model_key}_comparison.json"
                             if original_file.exists():
                                 comparison_files_for_plotting[model_key] = original_file
                else:
                     print("Sémantické porovnání neproběhlo nebo nevrátilo žádná data.")

        except Exception as e:
            import traceback
            print(f"Chyba během sémantického porovnání: {e}")
            print(traceback.format_exc())
    
    # Pokud sémantické porovnání neproběhlo nebo bylo přeskočeno, použijeme původní soubory
    if not semantic_comparison_performed:
        print("Používám základní výsledky porovnání pro vizualizaci.")
        for model_key in comparison_results.keys():
             original_file = get_run_results_dir() / f"{model_key}_comparison.json"
             if original_file.exists():
                 comparison_files_for_plotting[model_key] = original_file
    
    # 7. Vizualizace výsledků
    visualize_results(comparison_files_for_plotting, all_timings, all_token_usages, include_semantic=semantic_comparison_performed)
    
    return comparison_results, all_timings, all_token_usages


# <<< Přidáno: Pomocná funkce pro načítání JSON dat >>>
# Přesunuto sem, aby byla definována před prepare_plotting_data
from src.utils.metadata_comparator import MetadataComparator # Import třídy pro METADATA_FIELDS
import logging # Zajistit, že logging je dostupný

def load_comparison_data_from_path(file_path: Path):
    """Načte data porovnání z dané cesty."""
    if not file_path or not file_path.exists():
        logging.error(f"Soubor {file_path} neexistuje.")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Zde můžeme přidat robustnější načítání s podporou NaN, pokud je potřeba
            # Prozatím standardní json.load
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Chyba při dekódování JSON v souboru {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Obecná chyba při načítání souboru {file_path}: {e}")
        return None
# <<< Konec přidání >>>


def prepare_plotting_data(comparison_files, all_timings, all_token_usages, include_semantic):
    """
    Připraví data pro vykreslování ze souborů s výsledky porovnání.
    
    Args:
        comparison_files (dict): Slovník, kde klíč je název modelu a hodnota je Path k souboru
                                 s výsledky porovnání (buď základní, nebo sémantický).
        all_timings (dict): Slovník s časy extrakce pro každý model a dokument.
        all_token_usages (dict): Slovník s počty tokenů pro každý model a dokument.
        include_semantic (bool): Zda má vizualizace zahrnovat sémantické zlepšení.
        
    Returns:
        dict: Slovník obsahující dva DataFrame - overall_df a detailed_df pro vizualizaci
    """
    print(f"Připravuji data pro grafy...")
    
    # Definice barev pro jednotlivé pipeline v grafech
    pipeline_colors = {
        "text": "#1f77b4",  # modrá
        "vlm": "#ff7f0e",  # oranžová
        "embedded": "#2ca02c",  # zelená
        "multimodal": "#d62728",  # červená
        "hybrid": "#17becf"  # tyrkysová
    }
    
    # Inicializace prázdných seznamů pro data
    detailed_data = []
    overall_data = []
    
    # Slovníky pro ukládání základních a sémantických dat pro každý model
    basic_data = {}
    semantic_data = {}
    
    # Zpracování každého model a souboru s jeho výsledky
    for model, comparison_file in comparison_files.items():
        # Přeskočit soubory s '_semantic' v názvu - zpracujeme je později jinak
        if "_semantic" in model:
            continue
        
        model_upper = model.upper()
        
        # Načtení dat ze základního porovnávacího souboru
        basic_comparison_data = load_comparison_data_from_path(comparison_file)
        if not basic_comparison_data:
            print(f"VAROVÁNÍ: Nepodařilo se načíst data pro {model} z {comparison_file}.")
            continue
            
        # Načtení dat ze sémantického porovnávacího souboru, pokud existuje
        semantic_comparison_file = comparison_file.parent / f"{model}_comparison_semantic.json"
        semantic_comparison_data = None
        if include_semantic and semantic_comparison_file.exists():
            semantic_comparison_data = load_comparison_data_from_path(semantic_comparison_file)
            if not semantic_comparison_data:
                print(f"VAROVÁNÍ: Nepodařilo se načíst sémantická data pro {model} z {semantic_comparison_file}.")
        
        # Uložíme reference na data pro pozdější zpracování
        basic_data[model] = basic_comparison_data
        if semantic_comparison_data:
            semantic_data[model] = semantic_comparison_data
            
        # Zpracování časů a tokenů pro tento model
        model_timings = all_timings.get(model, {})
        model_tokens = all_token_usages.get(model, {})
        
        # Výpočet průměrných časů a tokenů
        if model_timings:
            avg_duration = sum(model_timings.values()) / len(model_timings)
            total_duration = sum(model_timings.values())
        else:
            avg_duration = None
            total_duration = None
        
        # Výpočet celkového počtu tokenů
        total_input_tokens = 0
        total_output_tokens = 0
        if model_tokens:
            for _, tokens in model_tokens.items():
                if isinstance(tokens, dict):
                    total_input_tokens += tokens.get('input_tokens', 0)
                    total_output_tokens += tokens.get('output_tokens', 0)
        
        # Přidání celkových metrik pro tento model
        # Použijeme sémantická data, pokud jsou k dispozici
        metrics_data = semantic_comparison_data.get('metrics') if semantic_comparison_data else basic_comparison_data.get('metrics')
        
        if metrics_data:
            # Přidáme celkovou podobnost
            if 'overall_similarity' in metrics_data:
                overall_value = metrics_data['overall_similarity']
                # Pokud máme sémantická data, zjistíme základní a vylepšenou část
                base_value = basic_comparison_data.get('metrics', {}).get('overall_similarity', 0)
                improved_value = max(0, overall_value - base_value) if semantic_comparison_data else 0
                
                overall_data.append({
                    'Model': model_upper,
                    'Metric': 'Overall Similarity',
                    'Value': overall_value, 
                    'BaseValue': base_value,
                    'ImprovedValue': improved_value,
                    'PipelineType': model_upper  # Přidáno pro rozlišení v grafech
                })
            
            # Přidáme metriky pro jednotlivé typy metadat (pokud existují)
            for field in ['title', 'authors', 'abstract', 'keywords', 'doi', 'year', 'journal', 'volume', 'issue', 'pages', 'publisher']:
                # 'references' vyřazeno z extrakce
                field_key = f"{field}_similarity"
                if field_key in metrics_data:
                    field_value = metrics_data[field_key]
                    # Pokud máme sémantická data, zjistíme základní a vylepšenou část
                    base_field_value = basic_comparison_data.get('metrics', {}).get(field_key, 0)
                    improved_field_value = max(0, field_value - base_field_value) if semantic_comparison_data else 0
                    
                    overall_data.append({
                        'Model': model_upper,
                        'Metric': field.capitalize(),
                        'Value': field_value,
                        'BaseValue': base_field_value,
                        'ImprovedValue': improved_field_value,
                        'PipelineType': model_upper
                    })
        
        # Přidání metrik pro čas a tokeny
        if avg_duration is not None:
            overall_data.append({
                'Model': model_upper,
                'Metric': 'Avg Duration (s)',
                'Value': avg_duration,
                'BaseValue': avg_duration,
                'ImprovedValue': 0,
                'Type': 'Performance',
                'PipelineType': model_upper
            })
        
        overall_data.append({
            'Model': model_upper,
            'Metric': 'Input Tokens',
            'Value': total_input_tokens,
            'BaseValue': total_input_tokens,
            'ImprovedValue': 0,
            'Type': 'Usage',
            'PipelineType': model_upper
        })
        
        overall_data.append({
            'Model': model_upper,
            'Metric': 'Output Tokens',
            'Value': total_output_tokens,
            'BaseValue': total_output_tokens,
            'ImprovedValue': 0,
            'Type': 'Usage',
            'PipelineType': model_upper
        })
        
        # Zpracování detailních dat pro každý dokument a typ metadat
        # Použijeme sémantická data, pokud jsou k dispozici
        comparison_dict = semantic_comparison_data.get('comparison') if semantic_comparison_data else basic_comparison_data.get('comparison')
        
        if not comparison_dict:
            print(f"VAROVÁNÍ: Žádná data pro porovnání v modelu {model}.")
            continue
            
        for paper_id, paper_data in comparison_dict.items():
            # Přidána kontrola prázdných dat
            if paper_data is None:
                print(f"VAROVÁNÍ: Prázdná data pro dokument {paper_id} v modelu {model}, přeskakuji.")
                continue
                
            for metadata_field, field_data in paper_data.items():
                if metadata_field == 'overall_similarity':
                    continue  # Přeskočíme, to je souhrnná metrika
                
                # Získání hodnoty podobnosti
                similarity = None
                if isinstance(field_data, dict) and 'similarity' in field_data:
                    similarity = field_data['similarity']
                elif isinstance(field_data, (int, float)):
                    similarity = field_data
                
                if similarity is not None:
                    # Získání referenčních a extrahovaných hodnot
                    reference = None
                    extracted = None
                    if isinstance(field_data, dict):
                        reference = field_data.get('reference')
                        extracted = field_data.get('extracted')
                    
                    # Získání času a tokenů pro dokument
                    duration = model_timings.get(str(paper_id), None)
                    
                    doc_tokens = model_tokens.get(str(paper_id), {})
                    input_tokens = 0
                    output_tokens = 0
                    if isinstance(doc_tokens, dict):
                        input_tokens = doc_tokens.get('input_tokens', 0)
                        output_tokens = doc_tokens.get('output_tokens', 0)
                    
                    # Získání základní hodnoty podobnosti (před sémantickou kontrolou)
                    basic_similarity = None
                    if basic_comparison_data and 'comparison' in basic_comparison_data:
                        basic_paper_data = basic_comparison_data['comparison'].get(paper_id, {})
                        if basic_paper_data and metadata_field in basic_paper_data:
                            basic_field_data = basic_paper_data[metadata_field]
                            if isinstance(basic_field_data, dict) and 'similarity' in basic_field_data:
                                basic_similarity = basic_field_data['similarity']
                            elif isinstance(basic_field_data, (int, float)):
                                basic_similarity = basic_field_data
                    
                    # Pokud nemáme základní podobnost, použijeme současnou hodnotu
                    if basic_similarity is None:
                        basic_similarity = similarity
                    
                    # Výpočet vylepšené části (po sémantické kontrole)
                    improved_similarity = max(0, similarity - basic_similarity)
                    
                    # Přidání detailních dat
                    detailed_data.append({
                        'DOI': paper_id,
                        'Model': model_upper,
                        'Metadata': metadata_field.capitalize(),
                        'Value': similarity,  # Celková hodnota (po sémantické kontrole)
                        'BaseValue': basic_similarity,  # Základní hodnota před sémantickou kontrolou
                        'ImprovedValue': improved_similarity,  # Hodnota přidaná sémantickou kontrolou
                        'Reference': reference,
                        'Extracted': extracted,
                        'Duration': duration,
                        'Input_Tokens': input_tokens,
                        'Output_Tokens': output_tokens,
                        'PipelineType': model_upper
                    })
    
    # Vytvoření DataFrame
    detailed_df = pd.DataFrame(detailed_data)
    overall_df = pd.DataFrame(overall_data)
    
    if detailed_df.empty or overall_df.empty:
        print("Varování: Některý z výsledných DataFrame je prázdný!")
        return None
    
    return {
        "detailed_df": detailed_df,
        "overall_df": overall_df
    }


# Přidáno pro vykreslení box plotu porovnání polí
def plot_comparison_boxplot(detailed_df, filename="comparison_results_boxplot.png"):
    """
    Vytvoří krabicový graf (boxplot) porovnání podle typů metadat.
    
    Args:
        detailed_df (DataFrame): DataFrame s detailními skóre
        filename (str, optional): Název souboru pro uložení grafu. Defaults to "comparison_results_boxplot.png".
    """
    if detailed_df.empty:
        print("Nedostatek dat pro vytvoření boxplotu porovnání.")
        return

    if 'Metadata' not in detailed_df.columns:
        print("V DataFrame chybí sloupec 'Metadata'. Boxplot porovnání nebude vytvořen.")
        return
        
    try:
        plt.figure(figsize=(14, 8))
        
        # Použití Seaborn pro boxplot - nyní použijeme data po sémantické kontrole (Value)
        sns.boxplot(x='Metadata', y='Value', hue='Model', data=detailed_df)
        
        plt.title('Porovnání podobnosti podle typu metadat')
        plt.xlabel('Typ metadat')
        plt.ylabel('Podobnost')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Model', loc='upper right')
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Box plot porovnání polí uložen do {filename}")
        plt.close()
    except Exception as e:
        print(f"Chyba při vytváření boxplotu porovnání: {e}")
        import traceback
        traceback.print_exc()


# Přidáno pro vykreslení celkového box plotu
def plot_overall_boxplot(detailed_df, filename="overall_results_boxplot.png"):
    """
    Vytvoří krabicový graf (boxplot) celkového porovnání modelů.
    
    Args:
        detailed_df (DataFrame): DataFrame s detailními skóre
        filename (str, optional): Název souboru pro uložení grafu. Defaults to "overall_results_boxplot.png".
    """
    if detailed_df.empty:
        print("Nedostatek dat pro vytvoření celkového boxplotu.")
        return

    if 'DOI' not in detailed_df.columns or 'Model' not in detailed_df.columns:
        print("V DataFrame chybí potřebné sloupce. Celkový boxplot nebude vytvořen.")
        return
        
    try:
        # Agregace dat na úrovni dokumentu a modelu - nyní používáme pouze hodnoty po sémantické kontrole
        overall_scores = detailed_df.groupby(['DOI', 'Model'])['Value'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        
        # Použití Seaborn pro boxplot
        sns.boxplot(x='Model', y='Value', data=overall_scores)
        
        plt.title('Celkové porovnání modelů')
        plt.xlabel('Model')
        plt.ylabel('Průměrná podobnost')
        plt.tight_layout()
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Celkový box plot uložen do {filename}")
        plt.close()
    except Exception as e:
        print(f"Chyba při vytváření celkového boxplotu: {e}")
        import traceback
        traceback.print_exc()

def visualize_results(comparison_files, all_timings, all_token_usages, include_semantic=False):
    """
    Vizualizuje výsledky porovnání a vytváří grafy.
    
    Args:
        comparison_files (dict): Slovník s cestami k souborům s výsledky porovnání
        all_timings (dict): Slovník s časy pro každý model
        all_token_usages (dict): Slovník s počty tokenů pro každý model
        include_semantic (bool): Zda má vizualizace zahrnovat sémantické zlepšení
    """
    print("\n=== Vizualizace výsledků porovnání ===")
    
    # Příprava dat pro grafy
    plot_data = prepare_plotting_data(comparison_files, all_timings, all_token_usages, include_semantic)
    
    if plot_data is None:
        print("Nepodařilo se připravit data pro grafy. Vizualizace nebude vytvořena.")
        return

    detailed_df = plot_data["detailed_df"]
    overall_df = plot_data["overall_df"]
    
    # Definice barev pro typy modelů v grafech
    pipeline_colors = {
        "TEXT": "#1f77b4",  # modrá
        "VLM": "#ff7f0e",   # oranžová
        "EMBEDDED": "#2ca02c", # zelená
        "MULTIMODAL": "#d62728", # červená
        "HYBRID": "#17becf"  # tyrkysová
    }
    
    # Barva pro vylepšenou část (sémantické zlepšení)
    improved_color = "#f8c471"  # světle oranžová
    
    # 1. Graf celkového porovnání (Overall Similarity)
    overall_similarity_df = overall_df[overall_df['Metric'] == 'Overall Similarity']
    
    # Kontrola, zda máme data pro vytvoření grafu
    if len(overall_similarity_df) > 0:
        plt.figure(figsize=(10, 6))
        
        # Použijeme různé barvy pro různé typy modelů
        for model in overall_similarity_df['Model'].unique():
            model_data = overall_similarity_df[overall_similarity_df['Model'] == model]
            if model in pipeline_colors:
                # Vykreslit základní část sloupce
                base_value = model_data['BaseValue'].values[0]
                improved_value = model_data['ImprovedValue'].values[0]
                total_value = model_data['Value'].values[0]
                
                # Základní část
                plt.bar(model, base_value, color=pipeline_colors.get(model, 'gray'), alpha=0.8)
                
                # Vylepšená část (pokud existuje)
                if improved_value > 0:
                    plt.bar(model, improved_value, bottom=base_value, color=improved_color, alpha=0.8)
        
        plt.title('Celkové porovnání extrakce metadat')
        plt.xlabel('Model')
        plt.ylabel('Průměrná podobnost')
        plt.ylim(0, 1.1)  # Rozsah od 0 do 1.1 (pro viditelnost hodnot blížících se 1)
        
        # Popisky nad sloupci - celková hodnota po sémantické kontrole
        for i, model in enumerate(overall_similarity_df['Model']):
            total_height = overall_similarity_df[overall_similarity_df['Model'] == model]['Value'].values[0]
            plt.text(i, total_height + 0.01, f'{total_height:.2f}', 
                    ha='center', va='bottom', color='black', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(get_run_results_dir() / 'overall_results.png', dpi=300, bbox_inches='tight')
        print(f"Graf celkového porovnání uložen: {get_run_results_dir() / 'overall_results.png'}")

    plt.close()

    # Vytvoření souhrnného boxplotu
    plot_overall_boxplot(detailed_df, str(get_run_results_dir() / 'overall_results_boxplot.png'))
    
    # 2. Graf porovnání podle metadat
    # Kontrola, zda máme dostatek dat pro vytvoření grafu porovnání
    if len(detailed_df.get('Metadata', pd.Series()).unique()) == 0:
        print("Varování: Nedostatek dat pro vytvoření grafu porovnání podle metadat.")
    else:
        # Oddělíme data podle metadat
        plt.figure(figsize=(12, 8))
        
        metadata_fields = detailed_df['Metadata'].unique()
        x = np.arange(len(metadata_fields))
        
        model_count = len(detailed_df['Model'].unique())
        width = 0.8 / model_count  # šířka sloupce
        
        # Pro každý model vytvoříme skupinu sloupců
        i = 0
        for model in sorted(detailed_df['Model'].unique()):
            model_data = detailed_df[detailed_df['Model'] == model]
            
            for field_idx, field in enumerate(metadata_fields):
                field_data = model_data[model_data['Metadata'] == field]
                
                if len(field_data) > 0:
                    # Výpočet průměrných hodnot základní a vylepšené části
                    avg_base = field_data['BaseValue'].mean()
                    avg_improved = field_data['ImprovedValue'].mean()
                    
                    # Pozice sloupce
                    pos = field_idx + i * width - width * (model_count - 1) / 2
                    
                    # Vykreslení základní části
                    plt.bar(pos, avg_base, width * 0.9, 
                           color=pipeline_colors.get(model, 'gray'), alpha=0.8)
                    
                    # Vykreslení vylepšené části (pokud existuje)
                    if avg_improved > 0:
                        plt.bar(pos, avg_improved, width * 0.9, 
                               bottom=avg_base, color=improved_color, alpha=0.8)
                    
                    # Přidání popisku s celkovou hodnotou
                    total_height = avg_base + avg_improved
                    if field_idx == 0:  # Přidat legendu pouze pro první sloupec v každé skupině
                        plt.text(pos, total_height + 0.02, f'{total_height:.2f}',
                                ha='center', va='bottom', fontsize=8)
            
            i += 1
        
        plt.xlabel('Typ metadat')
        plt.ylabel('Průměrná podobnost')
        plt.title('Porovnání úspěšnosti extrakce podle typu metadat')
        plt.xticks(x, metadata_fields, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        
        # Vlastní legenda
        from matplotlib.patches import Patch
        legend_elements = []
        
        # Přidání všech modelů do legendy
        for model in sorted(detailed_df['Model'].unique()):
            legend_elements.append(Patch(facecolor=pipeline_colors.get(model, 'gray'), alpha=0.8, 
                                        label=model))
        
        # Přidání vylepšené části do legendy
        if include_semantic:
            legend_elements.append(Patch(facecolor=improved_color, alpha=0.8, 
                                         label='Vylepšení sémantickou kontrolou'))
        
        plt.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        
        plt.savefig(get_run_results_dir() / 'comparison_results.png', dpi=300, bbox_inches='tight')
        print(f"Graf porovnání podle typu metadat uložen: {get_run_results_dir() / 'comparison_results.png'}")
    
    plt.close()

    # Vytvoření boxplotu pro porovnání podle typů metadat
    plot_comparison_boxplot(detailed_df, str(get_run_results_dir() / 'comparison_results_boxplot.png'))
    
    # Uložení detailních výsledků do CSV
    try:
        # Upravíme sloupce před uložením
        detailed_csv_columns = ['DOI', 'Model', 'Metadata', 'Value', 'BaseValue', 'ImprovedValue']
        detailed_df.to_csv(get_run_results_dir() / 'detailed_scores_all.csv', 
                         columns=detailed_csv_columns, index=False)
        print(f"Detailní výsledky uloženy do CSV: {get_run_results_dir() / 'detailed_scores_all.csv'}")
    except Exception as e:
        print(f"Chyba při ukládání detailních výsledků do CSV: {e}")
    
    # Vytvoření souhrnného CSV podle typů metadat
    try:
        # Agregace dat podle modelu a typu metadat
        summary_data = []
        for model in detailed_df['Model'].unique():
            model_data = detailed_df[detailed_df['Model'] == model]
            
            for field in detailed_df['Metadata'].unique():
                field_data = model_data[model_data['Metadata'] == field]
                
                if len(field_data) > 0:
                    summary_data.append({
                        'Model': model,
                        'Field': field,
                        'Mean_Total': field_data['Value'].mean(),
                        'Std_Total': field_data['Value'].std() if len(field_data) > 1 else 0,
                        'Mean_Base': field_data['BaseValue'].mean(),
                        'Mean_Improved': field_data['ImprovedValue'].mean(),
                        'Count': len(field_data)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(get_run_results_dir() / 'summary_results.csv', index=False)
        print(f"Souhrnné výsledky uloženy do CSV: {get_run_results_dir() / 'summary_results.csv'}")
    except Exception as e:
        print(f"Chyba při ukládání souhrnných výsledků do CSV: {e}")
    
    # Vytvoření souhrnného CSV s celkovými metrikami
    try:
        # Agregace celkových metrik podle modelu
        overall_data = []
        for model in detailed_df['Model'].unique():
            model_data = detailed_df[detailed_df['Model'] == model]
            
            if len(model_data) > 0:
                overall_data.append({
                    'Model': model,
                    'Mean_Total_Overall': model_data['Value'].mean(),
                    'Std_Total_Overall': model_data['Value'].std() if len(model_data) > 1 else 0,
                    'Mean_Base_Overall': model_data['BaseValue'].mean(),
                    'Mean_Improved_Overall': model_data['ImprovedValue'].mean(),
                    'Count_Overall': len(model_data)
                })
        
        overall_summary_df = pd.DataFrame(overall_data)
        overall_summary_df.to_csv(get_run_results_dir() / 'overall_summary_results.csv', index=False)
        print(f"Celkové metriky uloženy do CSV: {get_run_results_dir() / 'overall_summary_results.csv'}")
    except Exception as e:
        print(f"Chyba při ukládání celkových metrik do CSV: {e}")
    
    # Vytvoření souhrnného JSON souboru se všemi sémantickými porovnáními
    if include_semantic:
        try:
            semantic_summary = {}
            for model, comparison_file in comparison_files.items():
                if "_semantic" not in model:  # Bereme pouze základní modely
                    semantic_file = comparison_file.parent / f"{model}_comparison_semantic.json"
                    if semantic_file.exists():
                        with open(semantic_file, 'r', encoding='utf-8') as f:
                            semantic_summary[model] = json.load(f)
            
            # Vytvoření adresáře, pokud neexistuje
            final_dir = get_run_results_dir() / "final_comparison"
            final_dir.mkdir(exist_ok=True)
            
            with open(final_dir / "semantic_comparison_summary.json", 'w', encoding='utf-8') as f:
                json.dump(semantic_summary, f, ensure_ascii=False, indent=2)
            print(f"Souhrnné porovnání uloženo: {final_dir / 'semantic_comparison_summary.json'}")
        except Exception as e:
            print(f"Chyba při vytváření souhrnného porovnání: {e}")
    
    print("Vizualizace výsledků dokončena.")

def generate_graphs_only():
    """
    Generuje pouze grafy z existujících výsledků bez spouštění extrakce.
    """
    print("\n=== Generování grafů z existujících výsledků ===")
    
    # Načtení výsledků z externích souborů
    output_dir = get_run_results_dir()
    
    # Načtení souborů s výsledky porovnání
    comparison_files = {}
    all_timings = {}
    all_token_usages = {}
    
    # Hledání všech souborů s porovnáním pro základní typy pipeline
    for pipeline_type in ["text", "embedded", "vlm", "multimodal", "hybrid"]:
        comparison_path = output_dir / f"{pipeline_type}_comparison.json"
        if comparison_path.exists():
            comparison_files[pipeline_type] = comparison_path
        
        # Načtení času a tokenů
        results_path = output_dir / f"{pipeline_type}_results.json"
        if results_path.exists():
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    results_data = json.load(f)
                
                # Extract timing information
                if "timings" in results_data:
                    all_timings[pipeline_type] = results_data["timings"]
                    print(f"Načteny časy a tokeny z {results_path}")
                else:
                    print(f"Soubor {results_path} neobsahuje informace o časech.")
                
                # Extract token usage if available
                token_info = {}
                if "results" in results_data:
                    for item in results_data["results"]:
                        if "doi" in item and "token_usage" in item:
                            token_info[item["doi"]] = item["token_usage"]
                    
                    if token_info:
                        all_token_usages[pipeline_type] = token_info
            except Exception as e:
                print(f"Chyba při načítání souboru {results_path}: {e}")
    
    # Sémantické porovnání
    for sem_file in output_dir.glob("*_comparison_semantic.json"):
        pipeline_type = sem_file.name.replace("_comparison_semantic.json", "")
        comparison_files[f"{pipeline_type}_semantic"] = sem_file
        print(f"Nalezen soubor sémantického porovnání pro {pipeline_type}: {sem_file}")
    
    if not comparison_files:
        print("Nebyly nalezeny žádné soubory s výsledky porovnání. Nelze vygenerovat grafy.")
        return
    
    # Zjistíme, zda by mělo být zahrnuto sémantické porovnání
    include_semantic = any("_semantic.json" in str(path) for path in comparison_files.values())
    
    # Generování grafů a tabulek
    print(f"Generování grafů z {len(comparison_files)} modelů, sémantické porovnání: {include_semantic}")
    visualize_results(comparison_files, all_timings, all_token_usages, include_semantic)
    
    print("Generování grafů dokončeno.")

def main():
    """
    Hlavní funkce pro spuštění procesu.
    """
    parser = argparse.ArgumentParser(description='Extrakce metadat z PDF souborů a porovnání výsledků.')
    parser.add_argument('--limit', type=int, default=None, help='Omezení počtu zpracovaných souborů')
    parser.add_argument('--models', nargs='+', choices=['embedded', 'vlm', 'text', 'multimodal'], default=['embedded'],
                        help='Modely k použití (embedded, vlm, text, multimodal)')
    parser.add_argument('--year-filter', nargs='+', type=int, help='Filtrování článků podle roku')
    parser.add_argument('--verbose', '-v', action='store_true', help='Podrobnější výstup')
    parser.add_argument('--skip-download', action='store_true', help='Přeskočí stahování PDF souborů')
    parser.add_argument('--skip-semantic', action='store_true', help='Přeskočí sémantické porovnání výsledků')
    parser.add_argument('--force-extraction', action='store_true', help='Vynutí novou extrakci metadat i když výsledky již existují')
    parser.add_argument('--config', type=str, default=None, help='Cesta ke konfiguračnímu souboru modelů')
    # <<< Změna: Přidání argumentu pro výstupní adresář >>>
    parser.add_argument('--output-dir', type=str, default=None, help='Adresář pro uložení výsledků tohoto běhu')
    # <<< Konec změny >>>
    parser.add_argument('--graphs-only', action='store_true', help="Spustí pouze generování grafů z existujících výsledků")
    # <<< Změna: Přidání podpory pro hybrid v parametru compare-only >>>
    parser.add_argument('--compare-only', type=str, choices=['embedded', 'vlm', 'text', 'multimodal', 'hybrid'], 
                        help="Pouze porovná výsledky daného modelu s referenčními daty bez nové extrakce")
    # <<< Konec změny >>>
    # <<< Přidáno: Parametr pro zahrnutí referencí do extrakce >>>
    parser.add_argument('--include-references', action='store_true', 
                        help="Zahrne reference do textu pro embedded pipeline (výchozí je vyloučit reference)")
    # <<< Konec přidání >>>
    
    args = parser.parse_args()
    
    # <<< Změna: Nastavení adresáře pro výsledky běhu >>>
    if args.output_dir:
        # Použijeme adresář z argumentu (pro volání z run_all_models.py)
        run_dir = Path(args.output_dir).resolve()
    else:
        # Vygenerujeme nový adresář s časovým razítkem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = BASE_DIR / "results" / f"main_{timestamp}"
    set_run_results_dir(run_dir)
    # <<< Konec změny >>>
    
    # Nastavení úrovně logování
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        print("Zapnuto podrobné logování")
    
    # Načtení konfigurace
    config_path = args.config if args.config else str(MODELS_JSON)
    
    # Při --graphs-only se pokusíme načíst konfiguraci z used_config.json v adresáři výsledků
    # POUZE pokud není explicitně zadán --config parametr
    if args.graphs_only and not args.config:
        used_config_path = get_run_results_dir() / "used_config.json"
        if used_config_path.exists():
            print(f"Načítám konfiguraci z {used_config_path} (režim --graphs-only)...")
            config_path = str(used_config_path)
        else:
            print(f"Soubor used_config.json nebyl nalezen v {get_run_results_dir()}, používám globální konfiguraci.")
    elif args.graphs_only and args.config:
        print(f"Používám explicitně zadanou konfiguraci {config_path} (režim --graphs-only)...")
    
    if os.path.exists(config_path):
        print(f"Načítám konfiguraci z {config_path}...")
        load_config(config_path)
        config = get_config()
        text_config = config.get_text_config()
        vision_config = config.get_vision_config()
        embedding_config = config.get_embedding_config()
        multimodal_config = config.get_multimodal_config() if 'multimodal' in args.models else None
        
        print(f"Konfigurace načtena z {config_path}")
        print(f"  Text provider: {text_config['provider']}, model: {text_config['model']}")
        print(f"  Vision provider: {vision_config['provider']}, model: {vision_config['model']}")
        print(f"  Embedding provider: {embedding_config['provider']}, model: {embedding_config['model']}")
        if multimodal_config:
            print(f"  Multimodal provider: {multimodal_config['provider']}, model: {multimodal_config['model']}")
        
        # Uložíme použitou konfiguraci pro pozdější analýzu (pouze pokud není --graphs-only)
        if not args.graphs_only:
            with open(get_run_results_dir() / "used_config.json", 'w', encoding='utf-8') as f:
                json.dump(config.config, f, ensure_ascii=False, indent=2)
    else:
        print(f"VAROVÁNÍ: Konfigurační soubor {config_path} neexistuje, používám výchozí konfiguraci.")
        # Uložíme výchozí konfiguraci
        with open(get_run_results_dir() / "used_config.json", 'w', encoding='utf-8') as f:
            json.dump(get_config().config, f, ensure_ascii=False, indent=2)
    
    # Nastavení pro vynucenou extrakci
    if args.force_extraction:
        # Odstranění existujících souborů s výsledky
        for model in args.models:
             # <<< Změna: Použití get_run_results_dir() >>>
            result_file = get_run_results_dir() / f"{model}_results.json"
             # <<< Konec změny >>>
            if result_file.exists():
                print(f"Odstraňuji existující výsledky: {result_file}")
                result_file.unlink()
    
    # Zpracování parametru --graphs-only
    if args.graphs_only:
        generate_graphs_only()
        return
    
    # <<< Změna: Zpracování parametru --compare-only >>>
    # Zpracování parametru --compare-only
    if args.compare_only:
        try:
            print(f"\n=== Porovnávání existujících výsledků modelu {args.compare_only} ===")
            
            # Načtení referenčních dat
            reference_data = prepare_reference_data(FILTERED_CSV)
            
            # Cesta k souboru s výsledky modelu
            model_results_path = get_run_results_dir() / f"{args.compare_only}_results.json"
            
            if not model_results_path.exists():
                print(f"CHYBA: Soubor s výsledky {model_results_path} neexistuje.")
                return
            
            # Načtení výsledků modelu
            try:
                with open(model_results_path, 'r', encoding='utf-8') as f:
                    model_data = json.load(f)
                    
                    # Příprava výsledků pro porovnání, podle struktury souboru
                    results = []
                    
                    # Pro multimodal a hybrid bereme výsledky ze "results" pole
                    if args.compare_only in ["multimodal", "hybrid"]:
                        if "results" in model_data and isinstance(model_data["results"], list):
                            results = model_data["results"]
                        else:
                            print(f"CHYBA: Soubor {model_results_path} nemá očekávanou strukturu s polem 'results'.")
                            return
                    else:
                        # Pro ostatní modely bereme výsledky přímo, nebo z "results" pole
                        results = model_data.get("results", [])
                    
                    if not results:
                        print(f"CHYBA: Soubor {model_results_path} neobsahuje žádné výsledky.")
                        return
                    
                    # Konverze seznamu výsledků na slovník podle DOI
                    results_dict = {}
                    for item in results:
                        if "doi" in item:
                            results_dict[item["doi"]] = item
                    
                    print(f"Načteno {len(results_dict)} výsledků z {model_results_path}")
                    
                    # Porovnání výsledků s referenčními daty
                    
                    # Konverze referenčních dat na slovník s DOI jako klíčem
                    reference_by_doi = {}
                    for paper_id, ref_data in reference_data.items():
                        if "doi" in ref_data and ref_data["doi"]:
                            reference_by_doi[ref_data["doi"]] = ref_data
                    
                    print(f"Referenční data: {len(reference_data)} položek, z toho {len(reference_by_doi)} s DOI")
                    
                    # Pokud jde o hybrid nebo multimodal, mapujeme pomocí DOI
                    if args.compare_only in ["hybrid", "multimodal"]:
                        # Vytvoříme nový slovník, kde klíčem bude ID z referenčních dat
                        mapped_results = {}
                        for doi, result_item in results_dict.items():
                            # Najdeme odpovídající ID v referenčních datech
                            for paper_id, ref_item in reference_data.items():
                                if "doi" in ref_item and ref_item["doi"] == doi:
                                    # Nalezena shoda, uložíme s ID jako klíčem
                                    mapped_results[paper_id] = result_item
                                    break
                        
                        if mapped_results:
                            print(f"Úspěšně namapováno {len(mapped_results)} výsledků podle DOI")
                            comparison = compare_all_metadata(mapped_results, reference_data)
                        else:
                            print(f"VAROVÁNÍ: Žádný výsledek nemohl být namapován na referenční data podle DOI")
                            comparison = {}
                    else:
                        # Pro ostatní typy pipeline používáme původní metodu
                        comparison = compare_all_metadata(results_dict, reference_data)
                    
                    metrics = calculate_overall_metrics(comparison)
                    
                    comparison_results = {
                        'comparison': comparison,
                        'metrics': metrics
                    }
                    
                    # Uložení výsledků porovnání
                    comparison_output = get_run_results_dir() / f"{args.compare_only}_comparison.json"
                    
                    with open(comparison_output, "w", encoding="utf-8") as f:
                        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
                    
                    print(f"Výsledky porovnání uloženy do: {comparison_output}")
                    
                    # Volitelné sémantické porovnání
                    if not args.skip_semantic:
                        print(f"Spouštím sémantické porovnání pro model {args.compare_only}...")
                        try:
                            semantic_cmd = [
                                sys.executable,
                                "-m", "src.utils.semantic_comparison",
                                "--dir", str(get_run_results_dir())
                            ]
                            result = subprocess.run(semantic_cmd, check=True)
                            print(f"Sémantické porovnání dokončeno s návratovým kódem: {result.returncode}")
                        except subprocess.CalledProcessError as e:
                            print(f"Chyba při sémantickém porovnání: {e}")
                        except Exception as e:
                            print(f"Neočekávaná chyba při sémantickém porovnání: {e}")
                            import traceback
                            traceback.print_exc()
            except Exception as e:
                print(f"Chyba při porovnávání: {e}")
                import traceback
                traceback.print_exc()
        
            return
        except Exception as e:
            print(f"Chyba při zpracování parametru --compare-only: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Spuštění extrakce a porovnání
    try:
        run_extraction_pipeline(
            limit=args.limit, 
            models=args.models, 
            year_filter=args.year_filter, 
            skip_download=args.skip_download,
            skip_semantic=args.skip_semantic,
            force_extraction=args.force_extraction,
            include_references=args.include_references
        )
    except Exception as e:
        print(f"Chyba při spuštění pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Přidat nastavení logování na začátek main, pokud není globální
    log_level = logging.INFO if any('-v' in arg or '--verbose' in arg for arg in sys.argv) else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    main() 