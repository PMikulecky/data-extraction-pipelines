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
import re  # Přidáno pro práci s regulárními výrazy

# Import lokálních modulů
from data_preparation import filter_papers_with_valid_doi
from pdf_downloader import download_pdfs_for_filtered_papers
from models.embedded_pipeline import extract_metadata_from_pdfs as extract_with_embedded
# Dočasně zakomentováno kvůli problémům s importem
from models.vlm_pipeline import extract_metadata_from_pdfs as extract_with_vlm
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
            
            if 'vlm' in comparison_files and os.path.exists(comparison_files['vlm']):
                vlm_comparison_path = comparison_files['vlm']
                
                # Výstupní soubor pro sémantické porovnání
                semantic_output = RESULTS_DIR / "semantic_comparison_results.json"
                
                # Spuštění sémantického porovnání
                print("Provádím sémantické porovnání výsledků...")
                try:
                    # Spuštění sémantického porovnání
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
                except Exception as e:
                    print(f"Chyba při sémantickém porovnání: {e}")
            else:
                print("Sémantické porovnání přeskočeno - chybí výsledky VLM modelu.")
        else:
            print("Sémantické porovnání přeskočeno - chybí výsledky Embedded modelu.")
    elif skip_semantic:
        print("Sémantické porovnání přeskočeno (--skip-semantic).")
    
    # 7. Vizualizace výsledků
    visualize_results(comparison_results, include_semantic=(semantic_comparison_results is not None))
    
    return comparison_results


def visualize_results(comparison_results, include_semantic=False):
    """
    Vizualizuje výsledky porovnání.
    
    Args:
        comparison_results (dict): Výsledky porovnání
        include_semantic (bool): Zda zahrnout i sémanticky vylepšené výsledky
    """
    print("\n=== Vizualizace výsledků ===")
    
    # Příprava dat pro vizualizaci
    models = [model for model in comparison_results.keys() if not model.endswith('_semantic')]
    
    if len(models) == 0:
        print("Nejsou k dispozici žádné modely pro vizualizaci.")
        return
    
    fields = comparison_results[models[0]]['metrics'].keys()
    
    # Vytvoření DataFrame pro vizualizaci
    data = []
    
    for field in fields:
        if field != 'overall':
            for model in models:
                base_similarity = comparison_results[model]['metrics'][field]['mean']
                semantic_similarity = comparison_results.get(f"{model}_semantic", {}).get('metrics', {}).get(field, {}).get('mean', base_similarity)
                
                data.append({
                    'Model': model.upper(),
                    'Field': field,
                    'Base_Similarity': base_similarity,
                    'Semantic_Improvement': max(0, semantic_similarity - base_similarity)  # Zajistí, že improvement nebude záporný
                })
    
    df = pd.DataFrame(data)
    
    # Definice barev
    colors = {
        'EMBEDDED': '#2B7AB8',  # Sytá modrá jako základ
        'VLM': '#E85D45',  # Sytá červená jako základ
        'semantic_improvement': {
            'EMBEDDED': '#8ECAE6',  # Světlá modrá pro sémantické zlepšení
            'VLM': '#FFB5A7'  # Světlá červená pro sémantické zlepšení
        }
    }

    # Vytvoření grafu
    plt.figure(figsize=(14, 10))
    
    # Vykreslení základních sloupců
    base_bars = plt.bar(df.index, df['Base_Similarity'], 
                       color=[colors[model] for model in df['Model']])
    
    # Vykreslení sémantického zlepšení nad základními sloupci
    if include_semantic:
        semantic_bars = plt.bar(df.index, df['Semantic_Improvement'],
                              bottom=df['Base_Similarity'],
                              color=[colors['semantic_improvement'][model] for model in df['Model']])
    
    # Nastavení popisků osy x
    plt.xticks(range(len(df)), 
               [f"{row['Field']}\n({row['Model']})" for _, row in df.iterrows()],
               rotation=45, ha='right')
    
    plt.title('Porovnání úspěšnosti modelů v extrakci metadat')
    plt.xlabel('Pole metadat a model')
    plt.ylabel('Průměrná podobnost')
    plt.ylim(0, 1.05)
    
    # Vytvoření vlastní legendy s informacemi o obou modelech
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['EMBEDDED'], label='EMBEDDED - základní'),
        Patch(facecolor=colors['semantic_improvement']['EMBEDDED'], label='EMBEDDED - sémantické zlepšení'),
        Patch(facecolor=colors['VLM'], label='VLM - základní'),
        Patch(facecolor=colors['semantic_improvement']['VLM'], label='VLM - sémantické zlepšení')
    ]
    
    if include_semantic:
        plt.legend(handles=legend_elements, loc='upper right')
    else:
        # Pokud není zahrnuto sémantické porovnání, zobrazit jen základní modely
        plt.legend(handles=[legend_elements[0], legend_elements[2]], loc='upper right')
    
    plt.tight_layout()
    
    # Uložení grafu
    plt.savefig(RESULTS_DIR / "comparison_results.png")
    print(f"Graf uložen do {RESULTS_DIR / 'comparison_results.png'}")
    
    # Vytvoření grafu pro celkové výsledky
    plt.figure(figsize=(10, 6))
    
    # Příprava dat pro celkové výsledky
    overall_data = []
    for model in models:
        base_overall = comparison_results[model]['metrics']['overall']['mean']
        semantic_overall = comparison_results.get(f"{model}_semantic", {}).get('metrics', {}).get('overall', {}).get('mean', base_overall)
        
        overall_data.append({
            'Model': model.upper(),
            'Base_Overall': base_overall,
            'Semantic_Improvement': max(0, semantic_overall - base_overall)
        })
    
    overall_df = pd.DataFrame(overall_data)
    
    # Vykreslení základních sloupců pro celkové výsledky
    base_bars = plt.bar(overall_df.index, overall_df['Base_Overall'],
                       color=[colors[model] for model in overall_df['Model']])
    
    # Vykreslení sémantického zlepšení pro celkové výsledky
    if include_semantic:
        semantic_bars = plt.bar(overall_df.index, overall_df['Semantic_Improvement'],
                              bottom=overall_df['Base_Overall'],
                              color=[colors['semantic_improvement'][model] for model in overall_df['Model']])
    
    plt.xticks(range(len(overall_df)), overall_df['Model'])
    plt.title('Celková úspěšnost modelů v extrakci metadat')
    plt.xlabel('Model')
    plt.ylabel('Průměrná celková podobnost')
    plt.ylim(0, 1.05)
    
    # Využití stejné legendy i pro celkové výsledky
    if include_semantic:
        plt.legend(handles=legend_elements, loc='upper right')
    else:
        plt.legend(handles=[legend_elements[0], legend_elements[2]], loc='upper right')
    
    plt.tight_layout()
    
    # Uložení grafu celkových výsledků
    plt.savefig(RESULTS_DIR / "overall_results.png")
    print(f"Graf celkových výsledků uložen do {RESULTS_DIR / 'overall_results.png'}")
    
    # Uložení výsledků do CSV souborů
    df.to_csv(RESULTS_DIR / "detailed_results.csv", index=False)
    print(f"Detailní výsledky uloženy do {RESULTS_DIR / 'detailed_results.csv'}")
    
    overall_df.to_csv(RESULTS_DIR / "overall_results.csv", index=False)
    print(f"Celkové výsledky uloženy do {RESULTS_DIR / 'overall_results.csv'}")
    
    # Výpis celkových výsledků
    print("\nCelkové výsledky:")
    for _, row in overall_df.iterrows():
        total_similarity = row['Base_Overall'] + row['Semantic_Improvement']
        improvement_percent = (row['Semantic_Improvement'] / row['Base_Overall'] * 100) if row['Base_Overall'] > 0 else 0
        print(f"Model {row['Model']}: {total_similarity:.4f} (základní: {row['Base_Overall']:.4f}, zlepšení: +{row['Semantic_Improvement']:.4f}, {improvement_percent:.2f}%)")


def main():
    """
    Hlavní funkce pro spuštění procesu.
    """
    parser = argparse.ArgumentParser(description='Extrakce metadat z PDF souborů a porovnání výsledků.')
    parser.add_argument('--limit', type=int, default=None, help='Omezení počtu zpracovaných souborů')
    parser.add_argument('--models', nargs='+', choices=['embedded', 'vlm'], default=['embedded'],
                        help='Modely k použití (embedded, vlm)')
    parser.add_argument('--year-filter', nargs='+', type=int, help='Filtrování článků podle roku')
    parser.add_argument('--verbose', '-v', action='store_true', help='Podrobnější výstup')
    parser.add_argument('--skip-download', action='store_true', help='Přeskočí stahování PDF souborů')
    parser.add_argument('--skip-semantic', action='store_true', help='Přeskočí sémantické porovnání výsledků')
    parser.add_argument('--force-extraction', action='store_true', help='Vynutí novou extrakci metadat i když výsledky již existují')
    parser.add_argument('--config', type=str, default=None, help='Cesta ke konfiguračnímu souboru modelů')
    
    args = parser.parse_args()
    
    # Nastavení úrovně logování
    if args.verbose:
        import logging
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
    main() 