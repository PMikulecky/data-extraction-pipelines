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

# Import lokálních modulů
from src.data_preparation import filter_papers_with_valid_doi_and_references as filter_papers_with_valid_doi
from src.pdf_downloader import download_pdfs_for_filtered_papers
from src.models.embedded_pipeline import extract_metadata_from_pdfs as extract_with_embedded
from src.models.vlm_pipeline import extract_metadata_from_pdfs as extract_with_vlm
from src.models.text_pipeline import extract_metadata_from_pdfs as extract_with_text
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
if MODELS_JSON.exists():
    print(f"Načítám konfiguraci z {MODELS_JSON}...")
    load_config(MODELS_JSON)
    config = get_config()
    text_config = config.get_text_config()
    vision_config = config.get_vision_config()
    embedding_config = config.get_embedding_config()
    print(f"Konfigurace načtena:")
    print(f"  Text provider: {text_config['provider']}, model: {text_config['model']}")
    print(f"  Vision provider: {vision_config['provider']}, model: {vision_config['model']}")
    print(f"  Embedding provider: {embedding_config['provider']}, model: {embedding_config['model']}")
else:
    print(f"Konfigurační soubor {MODELS_JSON} neexistuje, používám výchozí konfiguraci.")


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
            'publisher': row.get('dc.publisher', ''),
            'references': row.get('utb.fulltext.references', '')
        }
        
        reference_data[paper_id] = metadata
    
    return reference_data


def run_extraction_pipeline(limit=None, models=None, year_filter=None, skip_download=False, skip_semantic=False, force_extraction=False):
    """
    Spustí celý proces extrakce metadat a porovnání výsledků.
    
    Args:
        limit (int, optional): Omezení počtu zpracovaných souborů
        models (list, optional): Seznam modelů k použití
        year_filter (list, optional): Seznam let pro filtrování článků
        skip_download (bool, optional): Přeskočí stahování PDF souborů
        skip_semantic (bool, optional): Přeskočí sémantické porovnání
        force_extraction (bool): Vynutí novou extrakci metadat
        
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
                    embedding_model_name=embedding_model_name
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

            # Odstraníme cesty k neexistujícím souborům
            vlm_comp_path_str = str(vlm_comp_path) if vlm_comp_path and vlm_comp_path.exists() else None
            emb_comp_path_str = str(emb_comp_path) if emb_comp_path and emb_comp_path.exists() else None
            txt_comp_path_str = str(txt_comp_path) if txt_comp_path and txt_comp_path.exists() else None

            if not emb_comp_path_str and not vlm_comp_path_str and not txt_comp_path_str:
                 print("Nebyly nalezeny žádné soubory *_comparison.json pro sémantické zpracování.")
            else:
                # Volání funkce pro sémantické porovnání
                # Ta uloží *_comparison_semantic.json soubory do get_run_results_dir()
                updated_semantic_data = process_comparison_files(
                    output_dir=get_run_results_dir(), # Výstupní adresář je adresář tohoto běhu
                    vlm_comparison_path=vlm_comp_path_str,
                    embedded_comparison_path=emb_comp_path_str,
                    text_comparison_path=txt_comp_path_str
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
    Připraví data pro vykreslování ze souborů s výsledky porovnání a PŘEDANÝCH časů extrakce.
    
    Args:
        comparison_files (dict): Slovník, kde klíč je název modelu a hodnota je Path k souboru
                                 s výsledky porovnání (buď základní, nebo sémantický).
        all_timings (dict): Slovník s časy extrakce pro každý model a dokument.
        all_token_usages (dict): Slovník s počty tokenů pro každý model a dokument.
        include_semantic (bool): Zda má vizualizace zahrnovat sémantické zlepšení.
        
    Vrací DataFrame pro detailní výsledky a DataFrame pro souhrnné výsledky.
    """
    logging.info(f"Spouštím prepare_plotting_data s include_semantic: {include_semantic}")
    logging.info(f"Soubory k načtení: {comparison_files}")
    all_data = []
    detailed_scores = []
    available_models = [] # Seznam modelů, pro které máme data

    # Získáme seznam názvů modelů ze slovníku souborů
    model_names_from_files = list(comparison_files.keys())

    for model in sorted(model_names_from_files):
        logging.info(f"-- Zpracovávám model: {model} --")
        
        # Načteme finální data porovnání (mohou být základní nebo sémantická)
        final_comparison_path = comparison_files.get(model)
        if not final_comparison_path or not final_comparison_path.exists():
             logging.warning(f"Soubor s výsledky porovnání pro model {model} nebyl nalezen v '{final_comparison_path}'. Model bude přeskočen.")
             continue
        
        logging.info(f"Načítám finální data pro {model} z {final_comparison_path}...")
        final_data = load_comparison_data_from_path(final_comparison_path)
        if not final_data or "comparison" not in final_data:
            logging.warning(f"Data porovnání pro model {model} jsou neúplná nebo nevalidní v souboru {final_comparison_path}. Model bude přeskočen.")
            continue

        # Načteme základní data porovnání VŽDY, pokud chceme zobrazit sémantické zlepšení
        base_data = None
        base_comparison_path = get_run_results_dir() / f"{model}_comparison.json"
        if include_semantic and base_comparison_path.exists():
            logging.info(f"Načítám základní data pro {model} z {base_comparison_path} pro výpočet zlepšení...")
            base_data = load_comparison_data_from_path(base_comparison_path)
            if not base_data or "comparison" not in base_data:
                logging.warning(f"Základní data porovnání pro model {model} v {base_comparison_path} jsou neúplná nebo nevalidní. Sémantické zlepšení nebude možné spočítat.")
                base_data = None # Resetujeme, aby se nepoužila neúplná data
        elif include_semantic:
             logging.warning(f"Základní soubor porovnání {base_comparison_path} nebyl nalezen. Sémantické zlepšení nebude možné spočítat.")

        # Pokud sémantické porovnání neproběhlo (include_semantic=False) nebo se nepodařilo načíst base_data,
        # pak base_data budou stejná jako final_data pro účely výpočtu (zlepšení bude 0)
        if not base_data:
            base_data = final_data
            logging.debug(f"Používám finální data jako základní pro model {model}.")

        model_timings = all_timings.get(model, {})
        model_token_usages = all_token_usages.get(model, {}) # Získat tokeny
        if not model_timings: logging.warning(f"Data o časech pro {model} nebyla nalezena.")
        if not model_token_usages: logging.warning(f"Data o tokenech pro {model} nebyla nalezena.")

        # Přidáváme model do available_models, pouze pokud má data
        if final_data.get("comparison"): # Ověříme, že máme co porovnávat
            available_models.append(model.upper())
        else:
            logging.warning(f"Finální data pro model {model} jsou prázdná ({final_comparison_path}), nepřidávám do available_models.")
            continue # Přeskočit model bez dat

        comparison_source = "semantic" if include_semantic and final_comparison_path.name.endswith("_semantic.json") else "base"
        logging.info(f"Zdroj porovnání pro {model}: {comparison_source}")

        final_comparison_dict = final_data.get("comparison", {})
        base_comparison_dict = base_data.get("comparison", {})
        logging.info(f"Počet dokumentů ve final_comparison_dict pro {model}: {len(final_comparison_dict)}")
        
        docs_processed = 0
        fields_processed = 0
        doc_ids_to_process = list(final_comparison_dict.keys())
        logging.info(f"Nalezeno {len(doc_ids_to_process)} ID dokumentů ke zpracování pro {model}.")
        
        for doc_id in doc_ids_to_process:
            final_doc_results = final_comparison_dict.get(doc_id)
            base_doc_results = base_comparison_dict.get(doc_id) # Může být None
            
            if not final_doc_results:
                logging.warning(f"Přeskakuji doc_id {doc_id} pro model {model}, nenalezeny finální výsledky.")
                continue

            docs_processed += 1
            
            # Získat čas a tokeny pro tento dokument
            duration = model_timings.get(str(doc_id))
            if duration is None or duration < 0: duration = np.nan
            
            doc_token_usage = model_token_usages.get(str(doc_id), {"input_tokens": 0, "output_tokens": 0}) # Default
            input_tokens = doc_token_usage.get("input_tokens", 0)
            output_tokens = doc_token_usage.get("output_tokens", 0)

            defined_fields = [f for f in MetadataComparator.METADATA_FIELDS]
            
            for field in defined_fields:
                final_scores_or_value = final_doc_results.get(field)
                final_similarity = 0
                if isinstance(final_scores_or_value, dict):
                    final_similarity = final_scores_or_value.get("similarity", 0)
                elif isinstance(final_scores_or_value, (float, int)):
                    final_similarity = float(final_scores_or_value)
                elif final_scores_or_value is None:
                     pass # final_similarity zůstane 0
                else:
                    logging.warning(f"Neočekávaný typ finální hodnoty pro pole {field} u {doc_id}/{model}: {type(final_scores_or_value)}. Similarity bude 0.")

                base_similarity_score = 0
                if base_doc_results:
                    base_scores_or_value = base_doc_results.get(field)
                    if isinstance(base_scores_or_value, dict):
                        base_similarity_score = base_scores_or_value.get("similarity", 0)
                    elif isinstance(base_scores_or_value, (float, int)):
                        base_similarity_score = float(base_scores_or_value)
                    elif base_scores_or_value is None:
                         pass # base_similarity_score zůstane 0
                    else:
                         logging.warning(f"Neočekávaný typ základní hodnoty pro pole {field} u {doc_id}/{model}: {type(base_scores_or_value)}. Similarity bude 0.")
                else:
                    # Pokud nemáme base_doc_results (např. chyba načítání), použijeme finální jako základní
                    base_similarity_score = final_similarity

                # <<< OPRAVA: Zajistit, že hodnoty jsou čísla před odečtením >>>
                # Převedeme None nebo NaN na 0.0
                final_similarity_num = float(final_similarity) if pd.notna(final_similarity) else 0.0
                base_similarity_score_num = float(base_similarity_score) if pd.notna(base_similarity_score) else 0.0
                
                # Výpočet sémantického zlepšení
                semantic_improvement = 0
                if include_semantic and final_comparison_path.name.endswith("_semantic.json"):
                     # Odečítáme až poté, co jsme zajistili, že jde o čísla
                     semantic_improvement = max(0.0, final_similarity_num - base_similarity_score_num)
                else:
                    # Pokud nezahrnujeme sémantiku nebo finální data nejsou sémantická,
                    # pak base = final a zlepšení je 0. Upravíme i base_similarity_score_num pro konzistenci.
                    base_similarity_score_num = final_similarity_num
                # <<< KONEC OPRAVY >>>
                
                fields_processed += 1

                detailed_scores.append({
                    "doc_id": str(doc_id),
                    "model": model.upper(),
                    "field": field,
                    "similarity": final_similarity_num,
                    "source": comparison_source,
                    "duration": duration,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                })

                all_data.append({
                    "doc_id": str(doc_id),
                    "Model": model.upper(),
                    "Field": field,
                    "Base_Similarity": base_similarity_score_num,
                    "Semantic_Improvement": semantic_improvement,
                    "Total_Similarity": final_similarity_num,
                    "Duration": duration,
                    "Input_Tokens": input_tokens,
                    "Output_Tokens": output_tokens
                })
        logging.info(f"Pro model {model} zpracováno {docs_processed} dokumentů a {fields_processed} záznamů polí.")

    if not all_data:
        logging.error("Chyba: Nebyla nalezena žádná data k vizualizaci po zpracování všech modelů.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    logging.info(f"Celkem záznamů pro agregaci: {len(all_data)}")
    logging.info(f"Celkem záznamů pro detailní skóre: {len(detailed_scores)}")

    plot_df_agg = pd.DataFrame(all_data)
    detailed_scores_df = pd.DataFrame(detailed_scores)

    if plot_df_agg.empty or detailed_scores_df.empty:
         logging.error("Vytvořené DataFrames jsou prázdné.")
         return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- Výpočet statistik pro graf porovnání polí ---
    try:
        # Vyloučit řádky s NaN duration pro výpočet statistik času, pokud je to žádoucí
        # summary_stats_timed = plot_df_agg.dropna(subset=['Duration'])
        # Nebo ponechat a NaN se budou ignorovat při mean/std
        summary_stats = plot_df_agg.groupby(['Model', 'Field']).agg(
            Mean_Base=('Base_Similarity', 'mean'),
            Mean_Improvement=('Semantic_Improvement', 'mean'),
            Std_Total=('Total_Similarity', 'std'),
            Mean_Total=('Total_Similarity', 'mean')
            # Mean_Duration=('Duration', 'mean') # Průměrný čas na pole nedává moc smysl
        ).reset_index()
        summary_stats.fillna({'Std_Total': 0}, inplace=True)
        logging.info(f"Vytvořen summary_stats DataFrame s {len(summary_stats)} řádky.")
        # logging.debug(f"Náhled summary_stats:\n{summary_stats.head().to_string()}")
    except Exception as e:
        logging.error(f"Chyba při agregaci summary_stats: {e}")
        summary_stats = pd.DataFrame()


    # --- Výpočet statistik pro celkový graf (včetně času a tokenů) ---
    try:
        # Agregace na úrovni dokumentu
        overall_per_doc = plot_df_agg.groupby(['Model', 'doc_id']).agg(
            Doc_Base_Overall=('Base_Similarity', 'mean'),
            Doc_Total_Overall=('Total_Similarity', 'mean'),
            Duration=('Duration', 'first'),
            Input_Tokens=('Input_Tokens', 'first'), # Tokeny jsou stejné pro všechna pole dokumentu
            Output_Tokens=('Output_Tokens', 'first')
        ).reset_index()

        # Agregace přes všechny dokumenty pro každý model
        overall_summary = overall_per_doc.groupby('Model').agg(
            Mean_Base_Overall=('Doc_Base_Overall', 'mean'),
            Std_Base_Overall=('Doc_Base_Overall', 'std'),
            Mean_Total_Overall=('Doc_Total_Overall', 'mean'),
            Std_Total_Overall=('Doc_Total_Overall', 'std'),
            Mean_Duration=('Duration', 'mean'),
            Std_Duration=('Duration', 'std'),
            Total_Input_Tokens=('Input_Tokens', 'sum'), # Celkový součet tokenů
            Total_Output_Tokens=('Output_Tokens', 'sum')
        ).reset_index()

        # Výpočet zlepšení
        overall_summary['Mean_Improvement'] = overall_summary['Mean_Total_Overall'].subtract(overall_summary['Mean_Base_Overall'], fill_value=0)
        
        # Doplnění chybějících hodnot
        overall_summary.fillna({
            'Std_Base_Overall': 0,
            'Std_Total_Overall': 0,
            'Std_Duration': 0,
            'Mean_Improvement': 0,
            'Total_Input_Tokens': 0, # Pokud by byly všechny NaN
            'Total_Output_Tokens': 0
        }, inplace=True)
        
        # Převedení tokenů na integer
        overall_summary['Total_Input_Tokens'] = overall_summary['Total_Input_Tokens'].astype(int)
        overall_summary['Total_Output_Tokens'] = overall_summary['Total_Output_Tokens'].astype(int)

        logging.info(f"Vytvořen overall_summary DataFrame s {len(overall_summary)} řádky.")
    except Exception as e:
        logging.error(f"Chyba při agregaci overall_summary: {e}")
        overall_summary = pd.DataFrame()


    # logging.debug(f"Náhled detailed_scores_df:\n{detailed_scores_df.head().to_string()}")

    # <<< PŘIDÁNO LOGOVÁNÍ VÝSTUPNÍCH DATAFRAMES >>>
    logging.info("--- Výstupní DataFrames pro vizualizaci ---")
    try:
        logging.info(f"summary_stats HEAD:\n{summary_stats.head().to_string()}")
    except Exception as log_e:
        logging.error(f"Chyba při logování summary_stats: {log_e}")
    try:
        logging.info(f"overall_summary:\n{overall_summary.to_string()}")
    except Exception as log_e:
        logging.error(f"Chyba při logování overall_summary: {log_e}")
    # Logování detailed_scores může být příliš velké, vynecháme
    # try:
    #     logging.info(f"detailed_scores_df HEAD:\n{detailed_scores_df.head().to_string()}")
    # except Exception as log_e:
    #      logging.error(f"Chyba při logování detailed_scores_df: {log_e}")
    logging.info("-------------------------------------------")
    # <<< KONEC LOGOVÁNÍ >>>

    return summary_stats, overall_summary, detailed_scores_df


# Přidáno pro vykreslení box plotu porovnání polí
def plot_comparison_boxplot(detailed_df, filename="comparison_results_boxplot.png"):
    """Vykreslí box plot pro porovnání výsledků podle polí."""
    if detailed_df.empty:
        print("Přeskakuji vykreslení box plotu pro porovnání polí - žádná data.")
        return

    plt.figure(figsize=(18, 10)) # Větší graf
    sns.boxplot(data=detailed_df, x='field', y='similarity', hue='model', palette='viridis')
    plt.title('Distribuce skóre podobnosti podle polí a modelů')
    plt.xlabel('Pole metadat')
    plt.ylabel('Skóre podobnosti')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1.05)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Upravit layout pro legendu mimo
    # <<< Změna: Použití get_run_results_dir() >>>
    filepath = get_run_results_dir() / filename
    # <<< Konec změny >>>
    try:
        plt.savefig(filepath)
        print(f"Box plot porovnání polí uložen do {filepath}")
    except Exception as e:
        print(f"Chyba při ukládání grafu {filepath}: {e}")
    plt.close()

# Přidáno pro vykreslení celkového box plotu
def plot_overall_boxplot(detailed_df, filename="overall_results_boxplot.png"):
    """Vykreslí box plot pro celkové porovnání modelů."""
    if detailed_df.empty:
        print("Přeskakuji vykreslení celkového box plotu - žádná data.")
        return

    # Spočítat celkové skóre pro každý dokument a model (průměr přes pole)
    overall_scores = detailed_df.groupby(['doc_id', 'model'])['similarity'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    # Oprava FutureWarning - explicitně nastavit hue a vypnout legendu
    sns.boxplot(data=overall_scores, x='model', y='similarity', hue='model', palette='viridis', legend=False)
    plt.title('Distribuce celkového skóre podobnosti podle modelů')
    plt.xlabel('Model')
    plt.ylabel('Průměrné skóre podobnosti dokumentu')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    # <<< Změna: Použití get_run_results_dir() >>>
    filepath = get_run_results_dir() / filename
    # <<< Konec změny >>>
    try:
        plt.savefig(filepath)
        print(f"Celkový box plot uložen do {filepath}")
    except Exception as e:
        print(f"Chyba při ukládání grafu {filepath}: {e}")
    plt.close()

def visualize_results(comparison_files, all_timings, all_token_usages, include_semantic=False):
    """
    Vykreslí porovnání výsledků modelů s error bary a box ploty.
    Přidána all_token_usages.
    """
    summary_stats, overall_summary, detailed_scores_df = prepare_plotting_data(
        comparison_files, 
        all_timings, 
        all_token_usages,
        include_semantic
    )

    if summary_stats.empty or overall_summary.empty:
        print("Nelze vykreslit grafy - chybí data.")
        return

    model_names = overall_summary['Model'].unique()
    if len(model_names) == 0:
        print("Žádné modely s daty pro vizualizaci.")
        return

    # Definice barev - AKTUALIZOVÁNO
    base_colors = {
        'EMBEDDED': '#3F5FDE', # Tmavší modrá
        'VLM': '#FF4747',      # Tmavší červeno-oranžová
        'TEXT': '#292F36',    # Tmavší zelená
    }
    lighter_semantic_colors = {
        'EMBEDDED': '#B7C3F3',      # Světlejší modrá
        'VLM': '#FF6B6B',    # Světlejší červeno-oranžová
        'TEXT': '#586574'       # Světlejší zelená
    }
    # Použít jen barvy pro modely, které máme
    colors = {m: base_colors.get(m, 'grey') for m in model_names}
    # Nepotřebujeme colors['semantic_improvement'] v této podobě


    # --- Graf porovnání polí (s error bary) ---
    plt.figure(figsize=(18, 10))

    fields = sorted(summary_stats['Field'].unique())
    n_models = len(model_names)
    x = np.arange(len(fields))
    width = 0.8 / n_models

    for i, model_name in enumerate(model_names):
        model_data = summary_stats[(summary_stats['Model'] == model_name)].set_index('Field').reindex(fields).reset_index()
        positions = x - (width * n_models / 2) + (i * width) + width / 2

        # <<< ZMĚNA: Vykreslení sloupců a error barů zvlášť >>>
        # 1. Základní sloupec
        plt.bar(positions, model_data['Mean_Base'], width,
                label=f'{model_name} - základní' if i == 0 else "",
                color=colors[model_name]) # Základní barva

        # 2. Sémantické zlepšení (pokud je)
        if include_semantic:
            plt.bar(positions, model_data['Mean_Improvement'], width,
                    bottom=model_data['Mean_Base'], # Navazuje na základní
                    label=f'{model_name} - sém. zlepšení' if i == 0 else "",
                    color=lighter_semantic_colors.get(model_name, 'lightgrey')) # Světlejší barva

        # 3. Chybové úsečky pro celkovou výšku
        total_heights = model_data['Mean_Total'] # Průměrná celková výška
        errors = model_data['Std_Total'].fillna(0) # Směrodatná odchylka celkové výšky
        plt.errorbar(positions, total_heights, yerr=errors, fmt='none',
                     ecolor='black', capsize=4) # Černé error bary
        # <<< KONEC ZMĚNY vykreslení sloupců >>>

    plt.xlabel('Pole metadat')
    plt.ylabel('Průměrná podobnost')
    plt.title('Porovnání úspěšnosti modelů v extrakci metadat (s chybovými úsečkami ±1σ celk. skóre)')
    plt.xticks(x, fields, rotation=45, ha='right')
    plt.ylim(0, 1.1)

    # <<< ZMĚNA: Aktualizace legendy >>>
    from matplotlib.patches import Patch
    legend_elements = []
    for model_name in model_names:
         legend_elements.append(Patch(facecolor=colors[model_name], label=f'{model_name} - základní'))
         if include_semantic:
             legend_elements.append(Patch(facecolor=lighter_semantic_colors.get(model_name, 'lightgrey'), label=f'{model_name} - sém. zlepšení'))
    plt.legend(handles=legend_elements, title="Model & Typ skóre", bbox_to_anchor=(1.05, 1), loc='upper left')
    # <<< KONEC ZMĚNY legendy >>>

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    # <<< Změna: Použití get_run_results_dir() >>>
    filepath = get_run_results_dir() / "comparison_results.png"
    # <<< Konec změny >>>
    try:
        plt.savefig(filepath)
        print(f"Graf porovnání polí uložen do {filepath}")
    except Exception as e:
        print(f"Chyba při ukládání grafu {filepath}: {e}")
    plt.close()

    # --- Graf celkových výsledků (s error bary) ---
    plt.figure(figsize=(10, 6))
    x_overall = np.arange(len(model_names))

    # <<< ZMĚNA: Vykreslení celkových sloupců a error barů zvlášť (pro konzistenci a jasnější barvu) >>>
    # 1. Základní sloupec
    plt.bar(x_overall, overall_summary['Mean_Base_Overall'],
             color=[colors.get(m, 'grey') for m in overall_summary['Model']])

    # 2. Sémantické zlepšení (pokud je)
    if include_semantic:
        plt.bar(x_overall, overall_summary['Mean_Improvement'],
                bottom=overall_summary['Mean_Base_Overall'], # Navazuje na základní
                color=[lighter_semantic_colors.get(m, 'lightgrey') for m in overall_summary['Model']])

    # 3. Chybové úsečky pro celkovou výšku
    total_heights_overall = overall_summary['Mean_Total_Overall']
    errors_overall = overall_summary['Std_Total_Overall'].fillna(0)
    plt.errorbar(x_overall, total_heights_overall, yerr=errors_overall, fmt='none',
                 ecolor='black', capsize=5) # Černé error bary
    # <<< KONEC ZMĚNY vykreslení celkových sloupců >>>

    plt.xlabel('Model')
    plt.ylabel('Průměrná celková podobnost')
    plt.title('Celková úspěšnost modelů (průměr ±1σ celk. skóre)')
    plt.xticks(x_overall, overall_summary['Model'])
    plt.ylim(0, 1.1)
    # Legenda zde není nutná, protože ji máme v detailním grafu
    plt.legend().set_visible(False)
    plt.tight_layout()
    # <<< Změna: Použití get_run_results_dir() >>>
    filepath = get_run_results_dir() / "overall_results.png"
    # <<< Konec změny >>>
    try:
        plt.savefig(filepath)
        print(f"Graf celkových výsledků uložen do {filepath}")
    except Exception as e:
        print(f"Chyba při ukládání grafu {filepath}: {e}")
    plt.close()

    # --- Generování Box plotů ---
    plot_comparison_boxplot(detailed_scores_df)
    plot_overall_boxplot(detailed_scores_df)

    # Uložení nových souhrnných tabulek
    try:
        summary_stats_path = get_run_results_dir() / "summary_results.csv"
        summary_stats.to_csv(summary_stats_path, index=False, float_format='%.4f')
        print(f"Souhrnné statistiky (průměr, std dev) uloženy do {summary_stats_path}")

        overall_summary_path = get_run_results_dir() / "overall_summary_results.csv"
        # Zajistíme správné pořadí sloupců včetně tokenů
        cols_order = [
            'Model', 'Mean_Total_Overall', 'Std_Total_Overall',
            'Mean_Duration', 'Std_Duration',
            'Total_Input_Tokens', 'Total_Output_Tokens',
            'Mean_Base_Overall', 'Std_Base_Overall', 'Mean_Improvement'
        ]
        final_cols_order = [col for col in cols_order if col in overall_summary.columns]
        overall_summary[final_cols_order].to_csv(overall_summary_path, index=False, float_format='%.4f')
        print(f"Celkové souhrnné statistiky (včetně času a tokenů) uloženy do {overall_summary_path}")

        detailed_scores_path = get_run_results_dir() / "detailed_scores_all.csv"
        # Přidat tokeny i sem?
        detailed_cols = [
            'doc_id', 'model', 'field', 'similarity', 'duration', 
            'input_tokens', 'output_tokens', 'source'
        ]
        final_detailed_cols = [col for col in detailed_cols if col in detailed_scores_df.columns]
        detailed_scores_df[final_detailed_cols].to_csv(detailed_scores_path, index=False, float_format='%.4f')
        print(f"Detailní skóre (včetně času a tokenů) uloženy do {detailed_scores_path}")

    except Exception as e:
        print(f"Chyba při ukládání CSV souborů: {e}")


    # Výpis celkových výsledků (včetně času a tokenů)
    print("\nCelkové výsledky (průměr ± std dev, celkové tokeny):")
    for _, row in overall_summary.iterrows():
        similarity_str = f"{row['Mean_Total_Overall']:.4f} ± {row['Std_Total_Overall']:.4f}"
        duration_str = f"{row['Mean_Duration']:.2f}s ± {row['Std_Duration']:.2f}s" if pd.notna(row['Mean_Duration']) else "N/A"
        token_str = f"Tokens(In={row['Total_Input_Tokens']}, Out={row['Total_Output_Tokens']})"
        print(f"Model {row['Model']}: Podobnost={similarity_str}, Čas={duration_str}, {token_str} ", end="")
        if include_semantic and 'Mean_Base_Overall' in row and 'Mean_Improvement' in row and pd.notna(row['Mean_Base_Overall']):
             print(f"(základní: {row['Mean_Base_Overall']:.4f}, zlepšení: +{row['Mean_Improvement']:.4f})")
        else:
             print()

def generate_graphs_only():
    """
    Generuje grafy z existujících souborů s výsledky bez spouštění extrakce.
    """
    print("\n=== Generování grafů z existujících výsledků ===")
    
    # Kontrola existence potřebných souborů
    comparison_files = {}
    all_timings = {}
    all_token_usages = {}
    
    # Kontrola existence souborů porovnání pro různé modely
    for model in ['embedded', 'vlm', 'text', 'hybrid']:  # <<< Změna: Přidání hybridní pipeline >>>
        # Nejprve zkusíme sémantické porovnání (priorita)
        semantic_comparison_file = get_run_results_dir() / f"{model}_comparison_semantic.json"
        basic_comparison_file = get_run_results_dir() / f"{model}_comparison.json"
        
        if semantic_comparison_file.exists():
            comparison_files[model] = semantic_comparison_file
            print(f"Nalezen soubor sémantického porovnání pro {model}: {semantic_comparison_file}")
        elif basic_comparison_file.exists():
            comparison_files[model] = basic_comparison_file
            print(f"Nalezen soubor základního porovnání pro {model}: {basic_comparison_file}")
    
    # Kontrola existence souborů s výsledky pro získání časů
    for model in ['embedded', 'vlm', 'text', 'hybrid']:  # <<< Změna: Přidání hybridní pipeline >>>
        results_file = get_run_results_dir() / f"{model}_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'timings' in data:
                        all_timings[model] = data['timings']
                    if 'token_usages' in data:
                        all_token_usages[model] = data['token_usages']
                print(f"Načteny časy a tokeny z {results_file}")
            except Exception as e:
                print(f"Chyba při načítání {results_file}: {e}")
    
    if not comparison_files:
        print("Nebyly nalezeny žádné soubory s výsledky porovnání. Nelze vygenerovat grafy.")
        return
    
    # Zjistíme, zda by mělo být zahrnuto sémantické porovnání
    include_semantic = any(str(path).endswith("_semantic.json") for path in comparison_files.values())
    
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
    parser.add_argument('--models', nargs='+', choices=['embedded', 'vlm', 'text', 'hybrid'], default=['embedded'],
                        help='Modely k použití (embedded, vlm, text, hybrid)')
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
    # <<< Změna: Přidání parametru compare-only >>>
    parser.add_argument('--compare-only', type=str, choices=['embedded', 'vlm', 'text', 'hybrid'], 
                        help="Pouze porovná výsledky daného modelu s referenčními daty bez nové extrakce")
    # <<< Konec změny >>>
    
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
    
    # Načtení konfigurace, pokud je zadána
    if args.config:
        if os.path.exists(args.config):
            print(f"Načítám konfiguraci z {args.config}...")
            load_config(args.config)
            config = get_config()
            text_config = config.get_text_config()
            vision_config = config.get_vision_config()
            embedding_config = config.get_embedding_config()
            print(f"Konfigurace načtena z {args.config}")
            print(f"  Text provider: {text_config['provider']}, model: {text_config['model']}")
            print(f"  Vision provider: {vision_config['provider']}, model: {vision_config['model']}")
            print(f"  Embedding provider: {embedding_config['provider']}, model: {embedding_config['model']}")
        else:
            print(f"VAROVÁNÍ: Konfigurační soubor {args.config} neexistuje, používám výchozí konfiguraci.")
    
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
                    comparison = compare_all_metadata(results_dict, reference_data)
                    metrics = calculate_overall_metrics(comparison)
                    
                    comparison_results = {
                        'comparison': comparison,
                        'metrics': metrics
                    }
                    
                    # Uložení výsledků porovnání
                    comparison_output = get_run_results_dir() / f"{args.compare_only}_comparison.json"
                    with open(comparison_output, 'w', encoding='utf-8') as f:
                        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
                    print(f"Výsledky porovnání pro {args.compare_only} uloženy do {comparison_output}")
                    
                    # Pokud nemáme přeskočit sémantické porovnání, spustíme i to
                    if not args.skip_semantic:
                        from src.utils.semantic_comparison import process_comparison_files
                        
                        print(f"\n=== Sémantické porovnání výsledků {args.compare_only} ===")
                        process_comparison_files(
                            output_dir=get_run_results_dir(),
                            vlm_comparison_path=str(comparison_output) if args.compare_only == "vlm" else None,
                            embedded_comparison_path=str(comparison_output) if args.compare_only == "embedded" else None,
                            text_comparison_path=str(comparison_output) if args.compare_only == "text" else None,
                            hybrid_comparison_path=str(comparison_output) if args.compare_only == "hybrid" else None
                        )
                    
                    # Vygenerujeme grafy na základě existujících souborů
                    generate_graphs_only()
                    
            except Exception as e:
                print(f"Chyba při načítání a porovnávání výsledků: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"Chyba při porovnávání existujících výsledků: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        
        return
    # <<< Konec změny >>>

    start_time = time.time()
    
    try:
        # Výpis aktuální konfigurace z globální konfigurace modelů
        config = get_config()
        text_config = config.get_text_config()
        vision_config = config.get_vision_config()
        print(f"Použité nastavení modelů:")
        print(f"  Text provider: {text_config['provider']}, model: {text_config['model']}")
        print(f"  Vision provider: {vision_config['provider']}, model: {vision_config['model']}")
        
        run_extraction_pipeline(
            limit=args.limit, 
            models=args.models, 
            year_filter=args.year_filter, 
            skip_download=args.skip_download,
            skip_semantic=args.skip_semantic,
            force_extraction=args.force_extraction
        )
        
        elapsed_time = time.time() - start_time
        print(f"\nCelý proces dokončen za {elapsed_time:.2f} sekund.")
    except Exception as e:
        print(f"Chyba při spuštění procesu: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Přidat nastavení logování na začátek main, pokud není globální
    log_level = logging.INFO if any('-v' in arg or '--verbose' in arg for arg in sys.argv) else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    main() 