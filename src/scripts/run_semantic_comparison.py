#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skript pro spuštění sémantického porovnání mezi extrahovanými a referenčními daty.
Používá vylepšený algoritmus s podporou pro LLM porovnávání autorů.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional

# Přidání cesty k projektu
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config.runtime_config import set_run_results_dir, get_run_results_dir
from src.utils.semantic_comparison import process_comparison_files

# Načtení proměnných prostředí z .env souboru
load_dotenv()

# Základní adresář
BASE_DIR = Path(__file__).resolve().parent.parent.parent

def find_latest_run_dir(base_results_dir: Path, prefix: str = "main_") -> Optional[Path]:
    """Najde poslední adresář běhu main.py."""
    run_dirs = [d for d in base_results_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not run_dirs:
        return None
    # Seřadí adresáře podle času v názvu (nejnovější první)
    run_dirs.sort(key=lambda x: x.name.split('_')[-1], reverse=True)
    return run_dirs[0]

def main():
    """
    Hlavní funkce skriptu pro spuštění sémantického porovnání.
    """
    parser = argparse.ArgumentParser(
        description='Spustí sémantické porovnání mezi extrahovanými a referenčními daty.'
    )
    
    parser.add_argument(
        '--source-run-dir', 
        type=str, 
        default=None,
        help='Cesta k adresáři běhu main.py, ze kterého se načtou výsledky porovnání. Pokud není zadáno, pokusí se najít poslední běh.'
    )
    parser.add_argument(
        '--vlm-comp-file', 
        type=str, 
        default='vlm_comparison.json',
        help='Název souboru s VLM porovnáním uvnitř zdrojového adresáře.'
    )
    parser.add_argument(
        '--embedded-comp-file', 
        type=str, 
        default='embedded_comparison.json',
        help='Název souboru s Embedded porovnáním uvnitř zdrojového adresáře.'
    )
    parser.add_argument(
        '--text-comp-file', 
        type=str, 
        default='text_comparison.json',
        help='Název souboru s Text porovnáním uvnitř zdrojového adresáře (pokud existuje).'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='semantic_comparison_summary.json',
        help='Název výstupního souboru pro souhrnné výsledky v adresáři tohoto běhu.'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Vypne použití LLM pro porovnávání autorů (úspora nákladů)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Specifikuje model LLM (např. gpt-3.5-turbo, gpt-4-turbo)'
    )
    
    args = parser.parse_args()
    
    # Nastavení proměnných prostředí z příkazové řádky
    if args.no_llm:
        os.environ["USE_LLM_FOR_AUTHORS"] = "0"
    
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    
    # <<< Změna: Nastavení adresáře pro výsledky tohoto běhu >>>
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_DIR / "results" / f"semantic_{timestamp}"
    set_run_results_dir(run_dir)
    # <<< Konec změny >>>
    
    # <<< Změna: Určení zdrojového adresáře a cest k souborům >>>
    source_run_dir = None
    if args.source_run_dir:
        source_run_dir = Path(args.source_run_dir).resolve()
        if not source_run_dir.is_dir():
            print(f"Chyba: Zadaný zdrojový adresář {source_run_dir} neexistuje nebo není adresář.")
            return 1
    else:
        print("Zdrojový adresář nebyl zadán, hledám poslední běh 'main'...")
        source_run_dir = find_latest_run_dir(BASE_DIR / "results", prefix="main_")
        if not source_run_dir:
            print("Chyba: Nebyl nalezen žádný adresář běhu 'main' v adresáři 'results/'.")
            return 1
        print(f"Nalezen zdrojový adresář: {source_run_dir}")

    vlm_path_str = str(source_run_dir / args.vlm_comp_file) if (source_run_dir / args.vlm_comp_file).exists() else None
    embedded_path_str = str(source_run_dir / args.embedded_comp_file) if (source_run_dir / args.embedded_comp_file).exists() else None
    text_path_str = str(source_run_dir / args.text_comp_file) if (source_run_dir / args.text_comp_file).exists() else None
    
    # Výstupní cesta pro souhrn (není nutně vytvářena funkcí process_comparison_files)
    output_summary_path = get_run_results_dir() / args.output 
    # <<< Konec změny >>>
    
    # Kontrola existence alespoň jednoho souboru
    if not vlm_path_str and not embedded_path_str and not text_path_str:
        print(f"Chyba: V adresáři {source_run_dir} nebyly nalezeny žádné zdrojové soubory pro porovnání ({args.vlm_comp_file}, {args.embedded_comp_file}, {args.text_comp_file}).")
        return 1
    
    # Zpracování souborů a spuštění sémantického porovnání
    try:
        print(f"\nSpouštím sémantické porovnání...")
        print(f"Zdrojový adresář: {source_run_dir}")
        if vlm_path_str: print(f"VLM soubor: {vlm_path_str}")
        if embedded_path_str: print(f"Embedded soubor: {embedded_path_str}")
        if text_path_str: print(f"Text soubor: {text_path_str}")
        print(f"Výstupní adresář: {get_run_results_dir()}")
        print(f"Použití LLM pro autory: {'Vypnuto' if args.no_llm else 'Zapnuto'}")
        if args.model:
            print(f"LLM model: {args.model}")
        
        # <<< Změna: Volání process_comparison_files s output_dir >>>
        updated_results_dict = process_comparison_files(
            output_dir=get_run_results_dir(),
            vlm_comparison_path=vlm_path_str,
            embedded_comparison_path=embedded_path_str,
            text_comparison_path=text_path_str
        )
        # <<< Konec změny >>>
        
        # Uložení souhrnného souboru (volitelné)
        if updated_results_dict:
            try:
                with open(output_summary_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_results_dict, f, ensure_ascii=False, indent=2)
                print(f"Souhrnné výsledky sémantického porovnání uloženy do: {output_summary_path}")
            except Exception as e:
                print(f"Chyba při ukládání souhrnných výsledků do {output_summary_path}: {e}")
        else:
            print("Nebyly vygenerovány žádné výsledky pro uložení do souhrnného souboru.")
            
        return 0
    
    except Exception as e:
        import traceback
        print(f"Chyba při sémantickém porovnání: {e}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 