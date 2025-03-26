#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro přípravu dat pro srovnání AI modelů v extrakci metadat z akademických PDF.
"""

import os
import pandas as pd
import re
from pathlib import Path

# Definice cest
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_CSV = DATA_DIR / "papers.csv"
OUTPUT_CSV = DATA_DIR / "papers-filtered.csv"


def is_valid_doi(doi):
    """
    Kontroluje, zda je poskytnutý DOI validní.
    
    Args:
        doi (str): DOI k ověření
        
    Returns:
        bool: True pokud je DOI validní, jinak False
    """
    if not doi or pd.isna(doi):
        return False
        
    # Základní regex pro DOI
    doi_pattern = r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$'
    return bool(re.match(doi_pattern, str(doi), re.IGNORECASE))


def filter_papers_with_valid_doi(input_csv, output_csv):
    """
    Filtruje záznamy akademických prací, které mají validní DOI.
    
    Args:
        input_csv (str): Cesta ke vstupnímu CSV souboru
        output_csv (str): Cesta k výstupnímu CSV souboru
    
    Returns:
        int: Počet filtrovaných záznamů
    """
    print(f"Načítám data z {input_csv}...")
    df = pd.read_csv(input_csv, low_memory=False)
    
    print(f"Celkový počet záznamů: {len(df)}")
    
    # Extrakce sloupce s DOI
    doi_column = 'dc.identifier.doi'
    
    if doi_column not in df.columns:
        raise ValueError(f"Sloupec '{doi_column}' nebyl nalezen v CSV souboru.")
    
    # Filtrování záznamů s validním DOI
    df['valid_doi'] = df[doi_column].apply(is_valid_doi)
    filtered_df = df[df['valid_doi'] == True].copy()
    filtered_df.drop('valid_doi', axis=1, inplace=True)
    
    # Uložení filtrovaných dat
    filtered_count = len(filtered_df)
    print(f"Počet záznamů s validním DOI: {filtered_count}")
    
    # Vytvoření adresáře, pokud neexistuje
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Uložení výstupu
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtrovaná data uložena do {output_csv}")
    
    return filtered_count


def main():
    """
    Hlavní funkce pro přípravu dat.
    """
    try:
        filtered_count = filter_papers_with_valid_doi(INPUT_CSV, OUTPUT_CSV)
        print(f"Úspěšně filtrováno {filtered_count} záznamů s validním DOI.")
    except Exception as e:
        print(f"Chyba při filtrování dat: {e}")


if __name__ == "__main__":
    main() 