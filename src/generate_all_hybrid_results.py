#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skript pro generování hybridních výsledků pro všechny konfigurace v běhu
a vytvoření souhrnného porovnání všech konfigurací.
"""

import os
import subprocess
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_command(cmd, verbose=True):
    """Spustí příkaz a vrátí jeho návratový kód."""
    if verbose:
        print(f"Spouštím příkaz: {cmd}")
    process = subprocess.run(cmd, shell=True, capture_output=not verbose)
    if verbose and process.returncode != 0:
        print(f"Příkaz selhal s kódem {process.returncode}")
        print(f"Chybový výstup: {process.stderr.decode('utf-8', errors='replace') if hasattr(process, 'stderr') else 'N/A'}")
    return process.returncode

def combine_results_for_config(config_dir, verbose=True):
    """
    Vytvoří hybridní výsledky pro danou konfiguraci pomocí dynamického hybrid pipeline.
    
    Args:
        config_dir (str): Cesta k adresáři s konfigurací
    
    Returns:
        bool: True pokud kombinace proběhla úspěšně
    """
    # Nejprve zkontrolujeme, zda existují potřebné soubory
    config_path = Path(config_dir)
    text_results_path = config_path / "text_results.json"
    vlm_results_path = config_path / "vlm_results.json"
    hybrid_results_path = config_path / "hybrid_results.json"
    
    if not text_results_path.exists() or not vlm_results_path.exists():
        if verbose:
            print(f"Přeskakuji {config_dir} - chybí text_results.json nebo vlm_results.json")
        return False
    
    # Pokud již hybrid_results.json existuje, přeskočíme
    if hybrid_results_path.exists() and os.path.getsize(hybrid_results_path) > 0:
        if verbose:
            print(f"Přeskakuji {config_dir} - hybrid_results.json již existuje")
        return True
    
    # Vytvoříme hybridní výsledky pomocí DYNAMICKÉHO hybrid pipeline pro základní výsledky
    # Nejprve potřebujeme sémantické výsledky pro rozhodování
    semantic_success = combine_semantic_results_for_config(config_dir, verbose)
    if not semantic_success:
        if verbose:
            print(f"Nepodařilo se vytvořit sémantické výsledky pro {config_dir}")
        return False
    
    cmd = f'python -m src.dynamic_hybrid_pipeline --dir "{config_dir}" --base-only --confidence-threshold 0.05'
    base_success = run_command(cmd, verbose) == 0
    
    if base_success:
        # Spuštění přímého porovnání s referenčními daty pro hybridní pipeline
        comparison_cmd = f'python -m src.run_all_models --combine-only --results-dir "{config_dir}"'
        return run_command(comparison_cmd, verbose) == 0
    
    return False

def combine_semantic_results_for_config(config_dir, verbose=True):
    """
    Vytvoří hybridní sémantické výsledky pro danou konfiguraci pomocí dynamického hybrid pipeline.
    
    Args:
        config_dir (str): Cesta k adresáři s konfigurací
    
    Returns:
        bool: True pokud kombinace proběhla úspěšně
    """
    # Nejprve zkontrolujeme, zda existují potřebné soubory
    config_path = Path(config_dir)
    text_semantic_path = config_path / "text_comparison_semantic.json"
    vlm_semantic_path = config_path / "vlm_comparison_semantic.json"
    hybrid_semantic_path = config_path / "hybrid_comparison_semantic.json"
    
    if not text_semantic_path.exists() or not vlm_semantic_path.exists():
        if verbose:
            print(f"Přeskakuji sémantickou kombinaci {config_dir} - chybí potřebné soubory")
        return False
    
    # Pokud již hybrid_comparison_semantic.json existuje, přeskočíme
    if hybrid_semantic_path.exists() and os.path.getsize(hybrid_semantic_path) > 0:
        if verbose:
            print(f"Přeskakuji sémantickou kombinaci {config_dir} - hybrid_comparison_semantic.json již existuje")
        return True
    
    # Vytvoříme hybridní sémantické výsledky pomocí DYNAMICKÉHO hybrid pipeline
    cmd = f'python -m src.dynamic_hybrid_pipeline --dir "{config_dir}" --semantic-only --confidence-threshold 0.05'
    return run_command(cmd, verbose) == 0

def generate_graphs_for_config(config_dir, verbose=True):
    """
    Vygeneruje grafy pro danou konfiguraci.
    
    Args:
        config_dir (str): Cesta k adresáři s konfigurací
    
    Returns:
        bool: True pokud generování grafů proběhlo úspěšně
    """
    cmd = f'python -m src.main --graphs-only --output-dir "{config_dir}"'
    return run_command(cmd, verbose) == 0

def create_final_comparison(base_dir, output_dir, verbose=True):
    """
    Vytvoří finální porovnání všech konfigurací.
    
    Args:
        base_dir (str): Cesta k základnímu adresáři obsahujícímu všechny konfigurace
        output_dir (str): Cesta k výstupnímu adresáři pro finální porovnání
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Najít všechny adresáře konfigurací
    config_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name != "final_comparison"]
    
    if verbose:
        print(f"Nalezeno {len(config_dirs)} adresářů konfigurací")
    
    # Shromáždit data ze všech konfigurací
    all_models_data = []
    best_models = {
        'embedded': {'score': 0, 'config': None},
        'text': {'score': 0, 'config': None},
        'vlm': {'score': 0, 'config': None},
        'hybrid': {'score': 0, 'config': None}
    }
    
    for config_dir in config_dirs:
        config_name = config_dir.name
        
        if verbose:
            print(f"Zpracovávám konfiguraci: {config_name}")
        
        # Najít soubory s výsledky
        summary_file = config_dir / "overall_summary_results.csv"
        
        if not summary_file.exists():
            if verbose:
                print(f"Přeskakuji {config_name} - chybí overall_summary_results.csv")
            continue
        
        # Načíst souhrnné výsledky
        try:
            summary_df = pd.read_csv(summary_file)
            
            # Formát názvu modelu: MODEL_CONFIG
            summary_df['Model_Ext'] = summary_df['Model'] + '_' + config_name
            
            # Přidat do souhrnného DataFrame
            all_models_data.append(summary_df)
            
            # Aktualizovat nejlepší modely
            for _, row in summary_df.iterrows():
                model_type = row['Model'].lower()
                if model_type in best_models and row['Mean_Total_Overall'] > best_models[model_type]['score']:
                    best_models[model_type]['score'] = row['Mean_Total_Overall']
                    best_models[model_type]['config'] = config_name
            
        except Exception as e:
            if verbose:
                print(f"Chyba při zpracování {config_name}: {e}")
    
    if not all_models_data:
        print("Nebyly nalezeny žádné výsledky k porovnání!")
        return
    
    # Sloučit všechny DataFrame do jednoho
    all_models_df = pd.concat(all_models_data)
    
    # Uložit sloučený DataFrame
    combined_csv_path = output_path / "all_models_comparison.csv"
    all_models_df.to_csv(combined_csv_path, index=False)
    
    # Vytvořit soubor s nejlepšími modely
    best_models_path = output_path / "best_models.json"
    with open(best_models_path, 'w', encoding='utf-8') as f:
        json.dump(best_models, f, indent=2)
    
    # Vygenerovat graf porovnání všech modelů
    generate_comparison_graph(all_models_df, output_path)
    
    # Kopírovat nejlepší konfigurace do finálního adresáře
    for model_type, data in best_models.items():
        if data['config']:
            if verbose:
                print(f"Nejlepší {model_type}: {data['config']} (skóre: {data['score']:.4f})")
            
            # Kopírovat grafy nejlepších modelů do finálního adresáře
            source_dir = base_path / data['config']
            try:
                for file_name in ["comparison_results.png", "overall_results.png", 
                                  "comparison_results_boxplot.png", "overall_results_boxplot.png"]:
                    source_file = source_dir / file_name
                    if source_file.exists():
                        target_file = output_path / f"best_{model_type}_{file_name}"
                        shutil.copy2(source_file, target_file)
                        if verbose:
                            print(f"Kopíruji {source_file} -> {target_file}")
            except Exception as e:
                if verbose:
                    print(f"Chyba při kopírování souborů: {e}")
    
    if verbose:
        print(f"Finální porovnání uloženo do {output_path}")

def generate_comparison_graph(df, output_dir):
    """
    Vygeneruje graf porovnání všech modelů.
    
    Args:
        df (DataFrame): DataFrame s daty všech modelů
        output_dir (Path): Cesta k výstupnímu adresáři
    """
    plt.figure(figsize=(15, 8))
    
    # Získat unikátní typy modelů (např. EMBEDDED, TEXT, VLM, HYBRID)
    model_types = df['Model'].unique()
    
    # Definice barev pro typy modelů
    colors = {
        'EMBEDDED': '#3F5FDE',  # Modrá
        'TEXT': '#292F36',     # Černá
        'VLM': '#FF4747',      # Červená
        'HYBRID': '#FFA500'    # Oranžová
    }
    
    bar_width = 0.8 / len(df['Model_Ext'].unique()) * len(model_types)
    
    # Seskupit podle typu modelu a konfigurace
    grouped = df.sort_values(['Model', 'Model_Ext'])
    
    x = np.arange(len(model_types))
    
    # Pro každý typ modelu vykreslit všechny konfigurace
    for i, model_type in enumerate(model_types):
        model_data = grouped[grouped['Model'] == model_type]
        
        # Vykreslit sloupce pro každou konfiguraci
        for j, (_, row) in enumerate(model_data.iterrows()):
            config_name = row['Model_Ext'].split('_', 1)[1] if '_' in row['Model_Ext'] else 'unknown'
            x_pos = i + (j * bar_width / len(model_data))
            
            plt.bar(x_pos, row['Mean_Total_Overall'], 
                   width=bar_width / len(model_data),
                   color=colors.get(model_type, 'grey'),
                   alpha=0.7,
                   label=f"{model_type}_{config_name}")
            
            # Přidat error bar
            plt.errorbar(x_pos, row['Mean_Total_Overall'], 
                        yerr=row['Std_Total_Overall'] if 'Std_Total_Overall' in row else 0,
                        fmt='none', ecolor='black', capsize=5)
    
    plt.xlabel('Typ modelu')
    plt.ylabel('Průměrná podobnost')
    plt.title('Porovnání všech konfigurací modelů')
    plt.xticks(x, model_types)
    plt.ylim(0, 1.1)
    
    plt.legend(title="Model & Konfigurace", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Uložit graf
    output_file = output_dir / "all_models_comparison.png"
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    # Vytvořit tabulku výsledků
    generate_results_table(df, output_dir)

def generate_results_table(df, output_dir):
    """
    Vygeneruje tabulku s výsledky všech modelů.
    
    Args:
        df (DataFrame): DataFrame s daty všech modelů
        output_dir (Path): Cesta k výstupnímu adresáři
    """
    # Připravit pivot tabulku
    pivot_df = df.pivot_table(
        index='Model', 
        columns=['Model_Ext'], 
        values=['Mean_Total_Overall', 'Mean_Duration', 'Total_Input_Tokens', 'Total_Output_Tokens'],
        aggfunc='mean'
    )
    
    # Uložit jako HTML
    html_path = output_dir / "results_table.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("<html><head><style>")
        f.write("table { border-collapse: collapse; width: 100%; }")
        f.write("th, td { text-align: left; padding: 8px; }")
        f.write("tr:nth-child(even) { background-color: #f2f2f2; }")
        f.write("th { background-color: #4CAF50; color: white; }")
        f.write("</style></head><body>")
        f.write("<h2>Výsledky všech modelů</h2>")
        f.write(pivot_df.to_html())
        f.write("</body></html>")
    
    # Uložit jako CSV
    csv_path = output_dir / "results_table.csv"
    pivot_df.to_csv(csv_path)

def main():
    parser = argparse.ArgumentParser(description="Generování hybridních výsledků a porovnání všech konfigurací")
    parser.add_argument("--base-dir", type=str, required=True, help="Základní adresář s konfiguracemi")
    parser.add_argument("--hybrid-only", action="store_true", help="Pouze vygenerovat hybridní výsledky, bez grafů")
    parser.add_argument("--graphs-only", action="store_true", help="Pouze vygenerovat grafy, bez hybridních výsledků")
    parser.add_argument("--final-only", action="store_true", help="Pouze vygenerovat finální porovnání")
    parser.add_argument("--verbose", "-v", action="store_true", help="Zobrazit podrobné výstupy")
    parser.add_argument("--output-dir", type=str, default=None, help="Výstupní adresář pro finální porovnání (výchozí: <base_dir>/final_comparison)")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "final_comparison"
    
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"Chyba: Adresář {base_dir} neexistuje nebo není adresář")
        return
    
    # Najít všechny adresáře konfigurací
    config_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name != "final_comparison"]
    
    if args.verbose:
        print(f"Nalezeno {len(config_dirs)} adresářů konfigurací")
    
    # Pro každou konfiguraci
    if not args.final_only:
        for config_dir in config_dirs:
            config_name = config_dir.name
            
            if args.verbose:
                print(f"\n=== Zpracovávám konfiguraci: {config_name} ===")
            
            # 1. Vygenerovat hybridní výsledky
            if not args.graphs_only:
                success = combine_results_for_config(config_dir, args.verbose)
                if success and args.verbose:
                    print(f"Hybridní výsledky pro {config_name} úspěšně vygenerovány")
                
                # Vygenerovat hybridní sémantické výsledky
                semantic_success = combine_semantic_results_for_config(config_dir, args.verbose)
                if semantic_success and args.verbose:
                    print(f"Hybridní sémantické výsledky pro {config_name} úspěšně vygenerovány")
            
            # 2. Vygenerovat grafy
            if not args.hybrid_only:
                graph_success = generate_graphs_for_config(config_dir, args.verbose)
                if graph_success and args.verbose:
                    print(f"Grafy pro {config_name} úspěšně vygenerovány")
    
    # 3. Vytvořit finální porovnání
    create_final_comparison(base_dir, output_dir, args.verbose)

if __name__ == "__main__":
    main() 