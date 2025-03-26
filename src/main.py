#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hlavní skript pro spuštění procesu extrakce metadat z PDF souborů a porovnání výsledků.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
import argparse
import time

# Import lokálních modulů
from data_preparation import filter_papers_with_valid_doi
from pdf_downloader import download_pdfs_for_filtered_papers
from models.embedded_pipeline import extract_metadata_from_pdfs as extract_with_embedded
# Dočasně zakomentováno kvůli problémům s importem
# from models.vlm_pipeline import extract_metadata_from_pdfs as extract_with_vlm
from utils.metadata_comparator import compare_all_metadata, calculate_overall_metrics

# Načtení proměnných prostředí
load_dotenv()

# Definice cest
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_CSV = DATA_DIR / "papers.csv"
FILTERED_CSV = DATA_DIR / "papers-filtered.csv"
PDF_DIR = DATA_DIR / "pdfs"
RESULTS_DIR = BASE_DIR / "results"

# Vytvoření adresářů, pokud neexistují
PDF_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def run_extraction_pipeline(limit=None, models=None, year_filter=None, skip_download=False):
    """
    Spustí celý proces extrakce metadat a porovnání výsledků.
    
    Args:
        limit (int, optional): Omezení počtu zpracovaných souborů
        models (list, optional): Seznam modelů k použití
        year_filter (list, optional): Seznam let pro filtrování článků
        skip_download (bool, optional): Přeskočí stahování PDF souborů
        
    Returns:
        dict: Výsledky porovnání
    """
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
    
    if 'embedded' in models:
        print("\n=== Extrakce metadat pomocí Embedded pipeline ===")
        embedded_output = RESULTS_DIR / "embedded_results.json"
        
        if os.path.exists(embedded_output):
            print(f"Načítám existující výsledky z {embedded_output}...")
            with open(embedded_output, 'r', encoding='utf-8') as f:
                embedded_results = json.load(f)
        else:
            print("Spouštím extrakci metadat pomocí Embedded pipeline...")
            embedded_results = extract_with_embedded(PDF_DIR, embedded_output, limit=limit)
        
        results['embedded'] = embedded_results
    
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
    
    # 6. Vizualizace výsledků
    visualize_results(comparison_results)
    
    return comparison_results


def visualize_results(comparison_results):
    """
    Vizualizuje výsledky porovnání.
    
    Args:
        comparison_results (dict): Výsledky porovnání
    """
    print("\n=== Vizualizace výsledků ===")
    
    # Příprava dat pro vizualizaci
    models = list(comparison_results.keys())
    fields = comparison_results[models[0]]['metrics'].keys()
    
    # Vytvoření DataFrame pro vizualizaci
    data = []
    
    for model in models:
        for field in fields:
            if field != 'overall':
                mean_similarity = comparison_results[model]['metrics'][field]['mean']
                data.append({
                    'Model': model,
                    'Field': field,
                    'Similarity': mean_similarity
                })
    
    df = pd.DataFrame(data)
    
    # Vytvoření grafu
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Field', y='Similarity', hue='Model', data=df)
    plt.title('Porovnání úspěšnosti modelů v extrakci metadat')
    plt.xlabel('Pole metadat')
    plt.ylabel('Průměrná podobnost')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Uložení grafu
    plt.savefig(RESULTS_DIR / "comparison_results.png")
    print(f"Graf uložen do {RESULTS_DIR / 'comparison_results.png'}")
    
    # Vytvoření tabulky s celkovými výsledky
    overall_data = []
    
    for model in models:
        overall_similarity = comparison_results[model]['metrics']['overall']['mean']
        overall_data.append({
            'Model': model,
            'Overall Similarity': overall_similarity
        })
    
    overall_df = pd.DataFrame(overall_data)
    overall_df.to_csv(RESULTS_DIR / "overall_results.csv", index=False)
    print(f"Celkové výsledky uloženy do {RESULTS_DIR / 'overall_results.csv'}")
    
    # Vytvoření detailní CSV tabulky s výsledky pro každý dokument a pole
    detailed_data = []
    
    for model in models:
        for paper_id, paper_comparison in comparison_results[model]['comparison'].items():
            for field, field_data in paper_comparison.items():
                # Kontrola, zda field_data je slovník nebo přímo hodnota
                if isinstance(field_data, dict):
                    similarity = field_data.get('similarity', 0.0)
                    extracted = field_data.get('extracted', '')
                    reference = field_data.get('reference', '')
                else:
                    # Pokud field_data je přímo hodnota (např. numpy.float64)
                    similarity = float(field_data) if field_data is not None else 0.0
                    extracted = ''
                    reference = ''
                
                detailed_data.append({
                    'Model': model,
                    'Paper ID': paper_id,
                    'Field': field,
                    'Similarity': similarity,
                    'Extracted Value': extracted,
                    'Reference Value': reference
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(RESULTS_DIR / "detailed_results.csv", index=False)
    print(f"Detailní výsledky uloženy do {RESULTS_DIR / 'detailed_results.csv'}")
    
    # Vytvoření souhrnné tabulky s průměrnými výsledky pro každé pole
    summary_data = []
    
    for model in models:
        for field in fields:
            if field != 'overall':
                metrics = comparison_results[model]['metrics'][field]
                summary_data.append({
                    'Model': model,
                    'Field': field,
                    'Mean Similarity': metrics['mean'],
                    'Median Similarity': metrics['median'],
                    'Min Similarity': metrics['min'],
                    'Max Similarity': metrics['max'],
                    'Count': metrics['count']
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(RESULTS_DIR / "summary_results.csv", index=False)
    print(f"Souhrnné výsledky uloženy do {RESULTS_DIR / 'summary_results.csv'}")
    
    # Výpis celkových výsledků
    print("\nCelkové výsledky:")
    for model in models:
        overall_similarity = comparison_results[model]['metrics']['overall']['mean']
        print(f"Model {model}: {overall_similarity:.4f}")


def main():
    """
    Hlavní funkce pro spuštění procesu.
    """
    parser = argparse.ArgumentParser(description='Extrakce metadat z PDF souborů a porovnání výsledků.')
    parser.add_argument('--limit', type=int, default=None, help='Omezení počtu zpracovaných souborů')
    parser.add_argument('--models', nargs='+', choices=['embedded'], default=['embedded'],
                        help='Modely k použití (embedded)')
    parser.add_argument('--year-filter', nargs='+', type=int, help='Filtrování článků podle roku')
    parser.add_argument('--verbose', '-v', action='store_true', help='Podrobnější výstup')
    parser.add_argument('--skip-download', action='store_true', help='Přeskočí stahování PDF souborů')
    
    args = parser.parse_args()
    
    # Nastavení úrovně logování
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        print("Zapnuto podrobné logování")
    
    start_time = time.time()
    
    try:
        run_extraction_pipeline(limit=args.limit, models=args.models, year_filter=args.year_filter, skip_download=args.skip_download)
        
        elapsed_time = time.time() - start_time
        print(f"\nCelý proces dokončen za {elapsed_time:.2f} sekund.")
    except Exception as e:
        print(f"Chyba při spuštění procesu: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 