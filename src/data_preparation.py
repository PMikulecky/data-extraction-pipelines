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

# Definice sloupců
DOI_COLUMN = 'dc.identifier.doi'
REFERENCES_COLUMN = 'utb.fulltext.references'


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


def has_valid_references(references):
    """
    Kontroluje, zda jsou reference validní a neprázdné.
    
    Args:
        references (str): Reference k ověření
        
    Returns:
        bool: True pokud jsou reference validní, jinak False
    """
    if not references or pd.isna(references):
        return False
        
    # Odstranění bílých znaků a kontrola délky
    cleaned_refs = str(references).strip()
    return len(cleaned_refs) > 0


def filter_papers_with_valid_doi_and_references(input_csv, output_csv):
    """
    Filtruje záznamy akademických prací, které mají validní DOI a reference.
    
    Args:
        input_csv (str): Cesta ke vstupnímu CSV souboru
        output_csv (str): Cesta k výstupnímu CSV souboru
    
    Returns:
        dict: Statistiky filtrování
    """
    print(f"Načítám data z {input_csv}...")
    df = pd.read_csv(input_csv, low_memory=False)
    
    print(f"Celkový počet záznamů: {len(df)}")
    
    # Kontrola přítomnosti sloupců
    required_columns = [DOI_COLUMN, REFERENCES_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Chybějící sloupce v CSV souboru: {', '.join(missing_columns)}")
    
    # Filtrování záznamů s validním DOI
    df['valid_doi'] = df[DOI_COLUMN].apply(is_valid_doi)
    doi_filtered = df[df['valid_doi'] == True].copy()
    print(f"Počet záznamů s validním DOI: {len(doi_filtered)}")
    
    # Filtrování záznamů s validními referencemi
    doi_filtered['valid_references'] = doi_filtered[REFERENCES_COLUMN].apply(has_valid_references)
    filtered_df = doi_filtered[doi_filtered['valid_references'] == True].copy()
    
    # Odstranění pomocných sloupců
    filtered_df.drop(['valid_doi', 'valid_references'], axis=1, inplace=True)
    
    # Uložení filtrovaných dat
    filtered_count = len(filtered_df)
    print(f"Počet záznamů s validním DOI a referencemi: {filtered_count}")
    
    # Vytvoření adresáře, pokud neexistuje
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Uložení výstupu
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtrovaná data uložena do {output_csv}")
    
    # Statistiky
    stats = {
        'total_records': len(df),
        'with_valid_doi': len(doi_filtered),
        'with_valid_references': filtered_count,
        'percentage_with_doi': (len(doi_filtered) / len(df)) * 100,
        'percentage_with_references': (filtered_count / len(doi_filtered)) * 100
    }
    
    print("\nStatistiky filtrování:")
    print(f"- Celkový počet záznamů: {stats['total_records']}")
    print(f"- Záznamů s validním DOI: {stats['with_valid_doi']} ({stats['percentage_with_doi']:.1f}%)")
    print(f"- Záznamů s validním DOI a referencemi: {stats['with_valid_references']} ({stats['percentage_with_references']:.1f}%)")
    
    return stats


def main():
    """
    Hlavní funkce pro přípravu dat.
    """
    try:
        stats = filter_papers_with_valid_doi_and_references(INPUT_CSV, OUTPUT_CSV)
        print(f"\nÚspěšně filtrováno {stats['with_valid_references']} záznamů s validním DOI a referencemi.")
    except Exception as e:
        print(f"Chyba při filtrování dat: {e}")


if __name__ == "__main__":
    main() 