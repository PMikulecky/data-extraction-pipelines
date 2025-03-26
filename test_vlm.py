#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Jednoduchý testovací skript pro VLM pipeline.
"""

import os
import sys
from pathlib import Path

# Přidání cesty k modulům
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

try:
    print("Importuji VLM pipeline...")
    from src.models.vlm_pipeline import extract_metadata_from_pdfs
    print("Import VLM pipeline úspěšný.")
    
    # Nastavení cest
    pdf_dir = current_dir / "data" / "pdfs"
    results_file = current_dir / "results" / "vlm_test_results.json"
    
    print(f"PDF adresář: {pdf_dir}")
    print(f"Soubor s výsledky: {results_file}")
    
    # Kontrola existence adresáře a PDF souborů
    if not pdf_dir.exists():
        print(f"Adresář {pdf_dir} neexistuje.")
        sys.exit(1)
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"V adresáři {pdf_dir} nejsou žádné PDF soubory.")
        sys.exit(1)
    
    print(f"Nalezeno {len(pdf_files)} PDF souborů.")
    print(f"První soubor: {pdf_files[0]}")
    
    # Spuštění extrakce
    print("Spouštím extrakci metadat...")
    results = extract_metadata_from_pdfs(str(pdf_dir), str(results_file), limit=1)
    
    print("Extrakce dokončena.")
    print(f"Výsledky uloženy do {results_file}")
    
except ImportError as e:
    print(f"Chyba při importu: {e}")
except Exception as e:
    import traceback
    print(f"Chyba: {e}")
    print(traceback.format_exc()) 