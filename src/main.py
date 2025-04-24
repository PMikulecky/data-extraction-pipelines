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

# Import lokálních modulů
from data_preparation import filter_papers_with_valid_doi_and_references as filter_papers_with_valid_doi
from pdf_downloader import download_pdfs_for_filtered_papers
from models.embedded_pipeline import extract_metadata_from_pdfs as extract_with_embedded
# Dočasně zakomentováno kvůli problémům s importem
from models.vlm_pipeline import extract_metadata_from_pdfs as extract_with_vlm
from models.text_pipeline import extract_metadata_from_pdfs as extract_with_text
from utils.metadata_comparator import compare_all_metadata, calculate_overall_metrics
from utils.semantic_comparison import process_comparison_files
# Import konfiguračního modulu
from models.config.model_config import load_config, get_config

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
RESULTS_DIR = BASE_DIR / "results"
CONFIG_DIR = BASE_DIR / "config"
MODELS_JSON = CONFIG_DIR / "models.json"

# Vytvoření adresářů, pokud neexistují
PDF_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
        dict: Výsledky porovnání
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
    
    for model in models:
        if model == 'embedded':
            print("\n=== Extrakce metadat pomocí Embedded pipeline ===")
            print(f"Anthropic API klíč před extrakcí, začíná: {anthropic_key[:10]}..., délka: {len(anthropic_key)} znaků")
            output_file = RESULTS_DIR / "embedded_results.json"
            
            try:
                # Explicitní předání API klíče
                from models.config.model_config import get_config
                config = get_config()
                text_config = config.get_text_config()
                provider_name = text_config.get("provider")
                model_name = text_config.get("model")
                
                print(f"Použití providera {provider_name} a modelu {model_name} z konfigurace")
                
                # Explicitní předání API klíče podle poskytovatele
                api_key = None
                if provider_name == "anthropic":
                    api_key = anthropic_key
                    print(f"Předávám explicitně Anthropic API klíč, začátek: {api_key[:10]}...")
                elif provider_name == "openai":
                    api_key = openai_key
                    print(f"Předávám explicitně OpenAI API klíč, začátek: {api_key[:10]}...")
                
                results['embedded'] = extract_with_embedded(
                    PDF_DIR, 
                    output_file, 
                    limit=limit, 
                    force_extraction=force_extraction,
                    provider_name=provider_name,
                    model_name=model_name,
                    api_key=api_key
                )
            except Exception as e:
                import traceback
                print(f"Chyba při extrakci metadat pomocí Embedded pipeline: {e}")
                print(f"Podrobnosti chyby: {traceback.format_exc()}")
                results['embedded'] = {}
                
        elif model == 'vlm':
            print("\n=== Extrakce metadat pomocí VLM pipeline ===")
            print(f"Anthropic API klíč před extrakcí, začíná: {anthropic_key[:10]}..., délka: {len(anthropic_key)} znaků")
            output_file = RESULTS_DIR / "vlm_results.json"
            
            try:
                # Explicitní předání API klíče
                from models.config.model_config import get_config
                config = get_config()
                vision_config = config.get_vision_config()
                provider_name = vision_config.get("provider")
                model_name = vision_config.get("model")
                
                print(f"Použití providera {provider_name} a modelu {model_name} z konfigurace")
                
                # Explicitní předání API klíče podle poskytovatele
                api_key = None
                if provider_name == "anthropic":
                    api_key = anthropic_key
                    print(f"Předávám explicitně Anthropic API klíč, začátek: {api_key[:10]}...")
                elif provider_name == "openai":
                    api_key = openai_key
                    print(f"Předávám explicitně OpenAI API klíč, začátek: {api_key[:10]}...")
                
                results['vlm'] = extract_with_vlm(
                    PDF_DIR, 
                    output_file, 
                    limit=limit,
                    force_extraction=force_extraction,
                    provider_name=provider_name,
                    model_name=model_name,
                    api_key=api_key
                )
            except Exception as e:
                import traceback
                print(f"Chyba při extrakci metadat pomocí VLM pipeline: {e}")
                print(f"Podrobnosti chyby: {traceback.format_exc()}")
                results['vlm'] = {}
        
        elif model == 'text':
            print("\n=== Extrakce metadat pomocí textové pipeline ===")
            print(f"Anthropic API klíč před extrakcí, začíná: {anthropic_key[:10]}..., délka: {len(anthropic_key)} znaků")
            output_file = RESULTS_DIR / "text_results.json"
            
            try:
                # Explicitní předání API klíče
                from models.config.model_config import get_config
                config = get_config()
                text_config = config.get_text_config()
                provider_name = text_config.get("provider")
                model_name = text_config.get("model")
                
                print(f"Použití providera {provider_name} a modelu {model_name} z konfigurace")
                
                # Explicitní předání API klíče podle poskytovatele
                api_key = None
                if provider_name == "anthropic":
                    api_key = anthropic_key
                    print(f"Předávám explicitně Anthropic API klíč, začátek: {api_key[:10]}...")
                elif provider_name == "openai":
                    api_key = openai_key
                    print(f"Předávám explicitně OpenAI API klíč, začátek: {api_key[:10]}...")
                
                results['text'] = extract_with_text(
                    PDF_DIR, 
                    output_file, 
                    limit=limit, 
                    force_extraction=force_extraction,
                    provider_name=provider_name,
                    model_name=model_name,
                    api_key=api_key
                )
            except Exception as e:
                import traceback
                print(f"Chyba při extrakci metadat pomocí textové pipeline: {e}")
                print(f"Podrobnosti chyby: {traceback.format_exc()}")
                results['text'] = {}
    
    # 5. Porovnání výsledků s referenčními daty
    comparison_results = {}
    
    for model_name, model_results in results.items():
        print(f"\n=== Porovnávání výsledků modelu {model_name} ===")
        comparison = compare_all_metadata(model_results, reference_data)
        metrics = calculate_overall_metrics(comparison)
        
        comparison_results[model_name] = {
            'comparison': comparison,
            'metrics': metrics
        }
        
        # Uložení výsledků porovnání
        comparison_output = RESULTS_DIR / f"{model_name}_comparison.json"
        with open(comparison_output, 'w', encoding='utf-8') as f:
            json.dump(comparison_results[model_name], f, ensure_ascii=False, indent=2)
    
    # 6. Sémantické porovnání (nový krok)
    semantic_comparison_results = None
    if not skip_semantic and len(comparison_results) >= 1:
        print("\n=== Sémantické porovnání výsledků ===")
        
        # Cesty k souborům porovnání
        comparison_files = {}
        for model_name in comparison_results.keys():
            comparison_files[model_name] = RESULTS_DIR / f"{model_name}_comparison.json"
        
        # Kontrola, zda existují potřebné soubory
        if 'embedded' in comparison_files and os.path.exists(comparison_files['embedded']):
            embedded_comparison_path = comparison_files['embedded']
            
            vlm_comparison_path = None
            if 'vlm' in comparison_files and os.path.exists(comparison_files['vlm']):
                vlm_comparison_path = comparison_files['vlm']
            
            text_comparison_path = None
            if 'text' in comparison_files and os.path.exists(comparison_files['text']):
                text_comparison_path = comparison_files['text']
                
            # Pokračovat pouze pokud máme alespoň VLM nebo text pipeline
            if vlm_comparison_path or text_comparison_path:
                # Výstupní soubor pro sémantické porovnání
                semantic_output = RESULTS_DIR / "semantic_comparison_results.json"
                
                # Spuštění sémantického porovnání
                print("Provádím sémantické porovnání výsledků...")
                try:
                    # Spuštění sémantického porovnání
                    if vlm_comparison_path and text_comparison_path:
                        # Pokud máme všechny tři, použijeme všechny
                        vlm_updated, embedded_updated, text_updated = process_comparison_files(
                            vlm_comparison_path, 
                            embedded_comparison_path,
                            semantic_output,
                            text_comparison_path
                        )
                        
                        # Uložení výsledků sémantického porovnání
                        semantic_comparison_results = {
                            'vlm': vlm_updated,
                            'embedded': embedded_updated,
                            'text': text_updated
                        }
                        
                        # Nahrazení původních výsledků sémanticky vylepšenými
                        comparison_results['vlm_semantic'] = vlm_updated
                        comparison_results['embedded_semantic'] = embedded_updated
                        comparison_results['text_semantic'] = text_updated
                        
                        print(f"Sémanticky vylepšené porovnání uloženo do {semantic_output}")
                        print(f"Samostatné soubory uloženy jako:")
                        print(f"- vlm_comparison_semantic.json")
                        print(f"- embedded_comparison_semantic.json")
                        print(f"- text_comparison_semantic.json")
                        
                    elif vlm_comparison_path:
                        # Pokud máme jen VLM a embedded
                        vlm_updated, embedded_updated = process_comparison_files(
                            vlm_comparison_path, 
                            embedded_comparison_path,
                            semantic_output
                        )
                        
                        # Uložení výsledků sémantického porovnání
                        semantic_comparison_results = {
                            'vlm': vlm_updated,
                            'embedded': embedded_updated
                        }
                        
                        # Nahrazení původních výsledků sémanticky vylepšenými
                        comparison_results['vlm_semantic'] = vlm_updated
                        comparison_results['embedded_semantic'] = embedded_updated
                        
                        print(f"Sémanticky vylepšené porovnání uloženo do {semantic_output}")
                        print(f"Samostatné soubory uloženy jako:")
                        print(f"- vlm_comparison_semantic.json")
                        print(f"- embedded_comparison_semantic.json")
                        
                    elif text_comparison_path:
                        # Pokud máme jen text a embedded
                        embedded_updated, text_updated = process_comparison_files(
                            embedded_comparison_path,
                            embedded_comparison_path,
                            semantic_output,
                            text_comparison_path
                        )
                        
                        # Uložení výsledků sémantického porovnání
                        semantic_comparison_results = {
                            'embedded': embedded_updated,
                            'text': text_updated
                        }
                        
                        # Nahrazení původních výsledků sémanticky vylepšenými
                        comparison_results['embedded_semantic'] = embedded_updated
                        comparison_results['text_semantic'] = text_updated
                        
                        print(f"Sémanticky vylepšené porovnání uloženo do {semantic_output}")
                        print(f"Samostatné soubory uloženy jako:")
                        print(f"- embedded_comparison_semantic.json")
                        print(f"- text_comparison_semantic.json")
                        
                except Exception as e:
                    print(f"Chyba při sémantickém porovnání: {e}")
            else:
                print("Sémantické porovnání přeskočeno - chybí výsledky VLM nebo text modelu.")
        else:
            print("Sémantické porovnání přeskočeno - chybí výsledky Embedded modelu.")
    elif skip_semantic:
        print("Sémantické porovnání přeskočeno (--skip-semantic).")
    
    # 7. Vizualizace výsledků
    visualize_results(comparison_results, include_semantic=(semantic_comparison_results is not None))
    
    return comparison_results


# Přidáno pro načítání dat z JSON
def load_comparison_data(model_name, use_semantic=False):
    """Načte data porovnání pro daný model."""
    suffix = "_comparison_semantic.json" if use_semantic else "_comparison.json"
    file_path = RESULTS_DIR / f"{model_name}{suffix}"
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Chyba při načítání souboru {file_path}: {e}")
            return None
    else:
        # Hledáme i v podadresářích, pokud je model vnořený (např. z run_all_models)
        # Jednoduchý příklad - hledá v RESULTS_DIR/[model_name]/[model_name]_comparison...
        nested_file_path = RESULTS_DIR / model_name / f"{model_name}{suffix}"
        if nested_file_path.exists():
            logging.info(f"Nalezen vnořený soubor výsledků: {nested_file_path}")
            try:
                with open(nested_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Chyba při načítání vnořeného souboru {nested_file_path}: {e}")
                return None
        logging.warning(f"Soubor s výsledky {file_path} (ani vnořený) nebyl nalezen.")
        return None


def prepare_plotting_data(models, include_semantic):
    """
    Připraví data pro vykreslování ze souborů s výsledky porovnání a časů extrakce.
    Vrací DataFrame pro detailní výsledky a DataFrame pro souhrnné výsledky.
    """
    logging.info(f"Spouštím prepare_plotting_data s modely: {models}, include_semantic: {include_semantic}")
    all_data = []
    detailed_scores = []
    available_models = []

    base_models = [m.replace('_semantic', '') for m in models]
    base_models = sorted(list(set(base_models)))
    logging.info(f"Zpracovávám základní modely: {base_models}")

    for model in base_models:
        logging.info(f"-- Zpracovávám model: {model} --")
        logging.info(f"Načítám základní data pro {model}...")
        base_data = load_comparison_data(model, use_semantic=False)
        if not base_data or ("results" not in base_data and "comparison" not in base_data):
            logging.warning(f"Nebyly nalezeny nebo jsou neúplné/nevalidní základní výsledky pro model {model}, model bude přeskočen.")
            continue

        # <<< Načtení časů extrakce >>>
        timing_file_path = RESULTS_DIR / f"{model}_timing.json"
        model_timings = {}
        try:
            if timing_file_path.exists():
                with open(timing_file_path, 'r') as f:
                    model_timings = json.load(f)
                logging.info(f"Načten soubor s časy: {timing_file_path}")
            else:
                 # Hledání ve vnořeném adresáři
                 nested_timing_path = RESULTS_DIR / model / f"{model}_timing.json"
                 if nested_timing_path.exists():
                     with open(nested_timing_path, 'r') as f:
                         model_timings = json.load(f)
                     logging.info(f"Nalezen vnořený soubor s časy: {nested_timing_path}")
                 else:
                     logging.warning(f"Soubor s časy {timing_file_path} (ani vnořený) nebyl nalezen pro model {model}.")
        except Exception as e:
             logging.error(f"Chyba při načítání souboru s časy {timing_file_path} (nebo vnořeného): {e}")
        # <<< Konec načtení časů >>>

        base_results_key = "results" if "results" in base_data else "comparison"
        logging.debug(f"Klíč pro výsledky v base_data: {base_results_key}")
        available_models.append(model.upper())

        semantic_data = None
        comparison_source = "base"
        final_data_source = base_data
        final_results_key = base_results_key

        if include_semantic:
            logging.info(f"Načítám sémantická data pro {model}...")
            semantic_data = load_comparison_data(model, use_semantic=True)
            if not semantic_data or ("results" not in semantic_data and "comparison" not in semantic_data):
                logging.warning(f"Nebyly nalezeny nebo jsou neúplné/nevalidní sémantické výsledky pro model {model}, použijí se základní.")
            else:
                 semantic_results_key = "results" if "results" in semantic_data else "comparison"
                 logging.debug(f"Klíč pro výsledky v semantic_data: {semantic_results_key}")
                 comparison_source = "semantic"
                 final_data_source = semantic_data
                 final_results_key = semantic_results_key
                 logging.info(f"Použiji sémantická data (klíč: {final_results_key}) pro {model}.")

        logging.info(f"Počet dokumentů ve final_data_source ({comparison_source}, klíč: {final_results_key}) pro {model}: {len(final_data_source.get(final_results_key, {}))}")

        docs_processed = 0
        fields_processed = 0
        # Procházíme klíče dokumentů z finálního zdroje dat (může být base nebo semantic)
        doc_ids_to_process = list(final_data_source.get(final_results_key, {}).keys())
        logging.info(f"Nalezeno {len(doc_ids_to_process)} ID dokumentů ke zpracování.")
        
        for doc_id in doc_ids_to_process:
            doc_results = final_data_source.get(final_results_key, {}).get(doc_id)
            
            # Kontrola, zda máme výsledky pro tento dokument
            if not doc_results:
                logging.warning(f"Přeskakuji doc_id {doc_id}, nenalezeny výsledky ve final_data_source.")
                continue

            docs_processed += 1
            doc_comparison = {}
            if isinstance(doc_results, dict) and "comparison" in doc_results:
                doc_comparison = doc_results.get("comparison", {})
            elif isinstance(doc_results, dict):
                 doc_comparison = doc_results
                 logging.debug(f"Dokument {doc_id} nemá klíč 'comparison', používám přímo obsah.")
            else:
                logging.warning(f"Neočekávaný formát pro doc_results u {doc_id}: {type(doc_results)}. Přeskakuji dokument.")
                continue

            if not doc_comparison:
                logging.debug(f"Dokument {doc_id} neobsahuje data pro porovnání. Přeskakuji.")
                continue

            # <<< Získání času pro dokument >>>
            # Používáme str(doc_id) pro konzistenci s JSON klíči
            duration = model_timings.get(str(doc_id))
            if duration is None or duration < 0: # Zahrnuje i náš indikátor chyby -1.0
                logging.debug(f"Čas pro dokument {doc_id} modelu {model} nebyl nalezen nebo je neplatný ({duration}). Nastavuji na NaN.")
                duration = np.nan # Použít NaN pro chybějící/neplatné časy
            # <<< Konec získání času >>>

            # Iterujeme přes pole definovaná ve třídě, ne přes výsledky (kvůli konzistenci)
            # Předpokládáme, že METADATA_FIELDS jsou definována někde globálně nebo importována
            # Pokud ne, museli bychom je získat jinak (např. z prvního dokumentu)
            defined_fields = [
                'title', 'authors', 'abstract', 'keywords', 'doi', 'year',
                'journal', 'volume', 'issue', 'pages', 'publisher', 'references'
            ] # TODO: Možná lépe načíst dynamicky?
            
            for field in defined_fields:
                scores_or_value = doc_comparison.get(field)
                similarity = 0
                
                # Zpracování hodnoty - může být dict nebo float
                if isinstance(scores_or_value, dict):
                    similarity = scores_or_value.get("similarity", 0)
                elif isinstance(scores_or_value, (float, int)):
                    similarity = float(scores_or_value)
                # Pokud pole chybí v porovnání, similarity zůstane 0
                elif scores_or_value is None:
                    logging.debug(f"Pole '{field}' chybí v porovnání pro doc_id {doc_id}. Similarity bude 0.")
                else:
                    logging.warning(f"Neočekávaný typ hodnoty pro pole {field} u {doc_id}: {type(scores_or_value)}. Similarity bude 0.")
                
                fields_processed += 1 # Počítáme i pole s nulovou podobností

                detailed_scores.append({
                    "doc_id": str(doc_id), # Ukládat jako string
                    "model": model.upper(),
                    "field": field,
                    "similarity": similarity,
                    "source": comparison_source,
                    "duration": duration # Přidáno (bude NaN pokud čas chybí)
                })

                # Získání základního skóre (musí také zvládnout oba formáty a chybějící pole)
                base_similarity_score = 0
                try:
                    base_doc_data = base_data.get(base_results_key, {}).get(str(doc_id), {})
                    base_comparison_data = {}
                    if isinstance(base_doc_data, dict) and "comparison" in base_doc_data:
                         base_comparison_data = base_doc_data.get("comparison", {})
                    elif isinstance(base_doc_data, dict): # Přímý přístup
                         base_comparison_data = base_doc_data

                    base_scores_or_value = base_comparison_data.get(field)
                    if isinstance(base_scores_or_value, dict):
                        base_similarity_score = base_scores_or_value.get("similarity", 0)
                    elif isinstance(base_scores_or_value, (float, int)):
                        base_similarity_score = float(base_scores_or_value)
                except Exception as e:
                    logging.debug(f"Chyba při hledání základního skóre pro {doc_id}/{field}: {e}")

                semantic_improvement = max(0, similarity - base_similarity_score) if comparison_source == "semantic" else 0

                all_data.append({
                    "doc_id": str(doc_id), # Ukládat jako string
                    "Model": model.upper(),
                    "Field": field,
                    "Base_Similarity": base_similarity_score,
                    "Semantic_Improvement": semantic_improvement,
                    "Total_Similarity": similarity,
                    "Duration": duration # Přidáno (bude NaN pokud čas chybí)
                })
        logging.info(f"Pro model {model} zpracováno {docs_processed} dokumentů a {fields_processed} záznamů polí (včetně chybějících).")

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


    # --- Výpočet statistik pro celkový graf (včetně času) ---
    try:
        # <<< Změna: Přidat agregaci času na úrovni dokumentu >>>
        # Agregujeme podobnost a vezmeme PRVNÍ platnou hodnotu času pro daný dokument/model
        # Protože čas by měl být stejný pro všechna pole daného dokumentu
        overall_per_doc = plot_df_agg.groupby(['Model', 'doc_id']).agg(
            Doc_Base_Overall=('Base_Similarity', 'mean'),
            Doc_Total_Overall=('Total_Similarity', 'mean'),
            Duration=('Duration', 'first') # Vezmeme první hodnotu (měla by být stejná, nebo NaN)
        ).reset_index()

        # Nyní agregujeme průměry a směrodatné odchylky přes všechny dokumenty pro každý model
        overall_summary = overall_per_doc.groupby('Model').agg(
            Mean_Base_Overall=('Doc_Base_Overall', 'mean'),
            Std_Base_Overall=('Doc_Base_Overall', 'std'),
            Mean_Total_Overall=('Doc_Total_Overall', 'mean'),
            Std_Total_Overall=('Doc_Total_Overall', 'std'),
            Mean_Duration=('Duration', 'mean'),  # Průměrný čas na dokument (ignoruje NaN)
            Std_Duration=('Duration', 'std')     # Směrodatná odchylka času (ignoruje NaN)
        ).reset_index()

        overall_summary['Mean_Improvement'] = overall_summary['Mean_Total_Overall'] - overall_summary['Mean_Base_Overall']
        # Doplnění chybějících std dev (pokud byl jen jeden dokument nebo všechny časy byly NaN)
        overall_summary.fillna({
            'Std_Base_Overall': 0,
            'Std_Total_Overall': 0,
            'Std_Duration': 0
        }, inplace=True)
        # <<< Konec změny >>>
        logging.info(f"Vytvořen overall_summary DataFrame s {len(overall_summary)} řádky.")
        # logging.debug(f"Náhled overall_summary:\n{overall_summary.to_string()}")
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
    filepath = RESULTS_DIR / filename
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
    filepath = RESULTS_DIR / filename
    try:
        plt.savefig(filepath)
        print(f"Celkový box plot uložen do {filepath}")
    except Exception as e:
        print(f"Chyba při ukládání grafu {filepath}: {e}")
    plt.close()

def visualize_results(comparison_results, include_semantic=False):
    """
    Vykreslí porovnání výsledků modelů s error bary a box ploty.
    """
    summary_stats, overall_summary, detailed_scores_df = prepare_plotting_data(list(comparison_results.keys()), include_semantic)

    if summary_stats.empty or overall_summary.empty:
        print("Nelze vykreslit grafy - chybí data.")
        return

    model_names = overall_summary['Model'].unique()
    if len(model_names) == 0:
        print("Žádné modely s daty pro vizualizaci.")
        return

    # Definice barev
    base_colors = {
        'EMBEDDED': 'skyblue',
        'VLM': 'lightcoral',
        'TEXT': 'lightgreen',
    }
    # <<< ZMĚNA: Definice světlejších barev pro sémantické zlepšení >>>
    lighter_semantic_colors = {
        'EMBEDDED': 'lightsteelblue', # Světlejší než steelblue
        'VLM': 'salmon',           # Světlejší než indianred
        'TEXT': 'palegreen'        # Světlejší než seagreen
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
    filepath = RESULTS_DIR / "comparison_results.png"
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
    filepath = RESULTS_DIR / "overall_results.png"
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
        summary_stats_path = RESULTS_DIR / "summary_results.csv"
        # Přidáme sloupec s průměrnou dobou trvání i sem?
        # Možná lepší nechat summary_stats jen pro podobnost a overall pro vše
        summary_stats.to_csv(summary_stats_path, index=False, float_format='%.4f')
        print(f"Souhrnné statistiky (průměr, std dev) uloženy do {summary_stats_path}")

        overall_summary_path = RESULTS_DIR / "overall_summary_results.csv"
        # Zajistíme správné pořadí sloupců pro lepší čitelnost
        cols_order = [
            'Model', 'Mean_Total_Overall', 'Std_Total_Overall',
            'Mean_Duration', 'Std_Duration', 'Mean_Base_Overall',
            'Std_Base_Overall', 'Mean_Improvement'
        ]
        # Zahrnout pouze sloupce, které skutečně existují v DataFrame
        final_cols_order = [col for col in cols_order if col in overall_summary.columns]
        overall_summary[final_cols_order].to_csv(overall_summary_path, index=False, float_format='%.4f')
        print(f"Celkové souhrnné statistiky (včetně času) uloženy do {overall_summary_path}")

        detailed_scores_path = RESULTS_DIR / "detailed_scores_all.csv"
        # Zajistíme pořadí sloupců
        detailed_cols = [
            'doc_id', 'model', 'field', 'similarity', 'duration', 'source'
        ]
        final_detailed_cols = [col for col in detailed_cols if col in detailed_scores_df.columns]
        detailed_scores_df[final_detailed_cols].to_csv(detailed_scores_path, index=False, float_format='%.4f')
        print(f"Detailní skóre (včetně času) pro box ploty uloženy do {detailed_scores_path}")

    except Exception as e:
        print(f"Chyba při ukládání CSV souborů: {e}")


    # Výpis celkových výsledků (průměr ± std dev) - včetně času
    print("\nCelkové výsledky (průměr ± std dev):")
    # Použijeme overall_summary pro výpis, který již obsahuje NaN ošetření
    for _, row in overall_summary.iterrows():
        # <<< Změna: Přidat výpis času a ošetřit NaN >>>
        similarity_str = f"{row['Mean_Total_Overall']:.4f} ± {row['Std_Total_Overall']:.4f}"
        duration_str = f"{row['Mean_Duration']:.2f}s ± {row['Std_Duration']:.2f}s" if pd.notna(row['Mean_Duration']) else "N/A"
        print(f"Model {row['Model']}: Podobnost={similarity_str}, Čas={duration_str} ", end="")
        # <<< Konec změny >>>
        if include_semantic and 'Mean_Base_Overall' in row and 'Mean_Improvement' in row and pd.notna(row['Mean_Base_Overall']):
             # Kontrola existence sloupců pro případ chyby při agregaci
             print(f"(základní: {row['Mean_Base_Overall']:.4f}, zlepšení: +{row['Mean_Improvement']:.4f})")
        else:
             print() # Jen nový řádek


def main():
    """
    Hlavní funkce pro spuštění procesu.
    """
    parser = argparse.ArgumentParser(description='Extrakce metadat z PDF souborů a porovnání výsledků.')
    parser.add_argument('--limit', type=int, default=None, help='Omezení počtu zpracovaných souborů')
    parser.add_argument('--models', nargs='+', choices=['embedded', 'vlm', 'text'], default=['embedded'],
                        help='Modely k použití (embedded, vlm, text)')
    parser.add_argument('--year-filter', nargs='+', type=int, help='Filtrování článků podle roku')
    parser.add_argument('--verbose', '-v', action='store_true', help='Podrobnější výstup')
    parser.add_argument('--skip-download', action='store_true', help='Přeskočí stahování PDF souborů')
    parser.add_argument('--skip-semantic', action='store_true', help='Přeskočí sémantické porovnání výsledků')
    parser.add_argument('--force-extraction', action='store_true', help='Vynutí novou extrakci metadat i když výsledky již existují')
    parser.add_argument('--config', type=str, default=None, help='Cesta ke konfiguračnímu souboru modelů')
    
    args = parser.parse_args()
    
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
            result_file = RESULTS_DIR / f"{model}_results.json"
            if result_file.exists():
                print(f"Odstraňuji existující výsledky: {result_file}")
                result_file.unlink()
    
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