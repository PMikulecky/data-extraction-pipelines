#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro správu běhové konfigurace, zejména cesty k adresáři s výsledky.
"""

from pathlib import Path
from typing import Optional
import os

# Globální proměnná pro uložení cesty k adresáři s výsledky aktuálního běhu
_run_results_dir: Optional[Path] = None

# Základní adresář pro výsledky, pokud není nastaven specifický adresář běhu
BASE_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

def set_run_results_dir(path: Path) -> None:
    """
    Nastaví cestu k adresáři s výsledky pro aktuální běh.
    Také zajistí vytvoření tohoto adresáře.
    """
    global _run_results_dir
    _run_results_dir = path
    try:
        os.makedirs(_run_results_dir, exist_ok=True)
        print(f"Adresář pro výsledky běhu nastaven na: {_run_results_dir}")
    except OSError as e:
        print(f"Chyba při vytváření adresáře {_run_results_dir}: {e}")
        # Případně zde můžeme vyvolat výjimku nebo nastavit _run_results_dir na None

def get_run_results_dir() -> Path:
    """
    Vrátí cestu k adresáři s výsledky pro aktuální běh.
    Pokud není nastavena, vrátí základní adresář 'results/'.
    """
    if _run_results_dir is None:
        # Zajistíme vytvoření základního adresáře, pokud ještě neexistuje
        BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        # print("VAROVÁNÍ: Adresář výsledků běhu nebyl explicitně nastaven, používá se základní adresář 'results/'.")
        return BASE_RESULTS_DIR
    return _run_results_dir 