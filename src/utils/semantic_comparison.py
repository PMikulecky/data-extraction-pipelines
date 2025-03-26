#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro sémantické porovnání extrahovaných a referenčních hodnot 
a vylepšení metriky similarity na základě faktické shody.
"""

import os
import json
import re
import unicodedata
from pathlib import Path
import numpy as np
from difflib import SequenceMatcher
import pandas as pd
import requests
from typing import Dict, List, Any, Union, Optional
import time

# Načtení proměnných prostředí z .env souboru (pokud existuje)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("dotenv není nainstalován, používám stávající proměnné prostředí")

# Definice cest
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Získání OpenAI API klíče z prostředí
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Konfigurace LLM
USE_LLM_FOR_AUTHORS = True  # Nastavit na False pro vypnutí LLM porovnávání (úspora nákladů)
LLM_MODEL = "gpt-3.5-turbo"  # Nebo "gpt-4" pro lepší výsledky, ale dražší
LLM_TEMPERATURE = 0.0  # Nízká teplota pro deterministické výsledky
LLM_MAX_TOKENS = 50  # Očekáváme jen krátkou odpověď (true/false)
LLM_REQUEST_TIMEOUT = 30  # Timeout pro API požadavek v sekundách
LLM_RETRY_COUNT = 2  # Počet opakování při selhání


def remove_diacritics(text):
    """
    Odstraní diakritiku z textu.
    
    Args:
        text (str): Vstupní text s diakritikou
        
    Returns:
        str: Text bez diakritiky
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    
    text = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in text if not unicodedata.combining(c)])


def call_openai_api(prompt: str) -> str:
    """
    Volá OpenAI API s daným promptem.
    
    Args:
        prompt (str): Text promptu pro LLM
        
    Returns:
        str: Odpověď modelu
    """
    if not OPENAI_API_KEY:
        print("VAROVÁNÍ: OPENAI_API_KEY není nastaven. Nelze použít LLM porovnávání.")
        return "false"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Jsi precizní nástroj pro porovnávání seznamů autorů. Vracíš POUZE 'true' pokud seznamy obsahují stejné autory bez ohledu na formát zápisu, jinak 'false'."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS
    }
    
    for attempt in range(LLM_RETRY_COUNT + 1):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=LLM_REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip().lower()
            else:
                print(f"Chyba API: {response.status_code} - {response.text}")
                if attempt < LLM_RETRY_COUNT:
                    time.sleep(2 * (attempt + 1))  # Exponenciální backoff
                    continue
                return "error"
                
        except Exception as e:
            print(f"Chyba při volání OpenAI API: {e}")
            if attempt < LLM_RETRY_COUNT:
                time.sleep(2 * (attempt + 1))
                continue
            return "error"
    
    return "error"


def compare_authors_with_llm(extracted: str, reference: str) -> bool:
    """
    Porovná dva seznamy autorů pomocí LLM.
    
    Args:
        extracted (str): Extrahovaný seznam autorů
        reference (str): Referenční seznam autorů
        
    Returns:
        bool: True pokud jsou seznamy fakticky shodné, jinak False
    """
    # Příprava promptu
    prompt = f"""
Porovnej tyto dva seznamy autorů a urči, zda jde o stejné autory (stejné osoby):

Seznam 1: {extracted}
Seznam 2: {reference}

Ignoruj rozdíly ve formátu zápisu (např. "Jméno Příjmení" vs "Příjmení, Jméno"), 
pořadí autorů, diakritiku a jiné drobné rozdíly.

Vrať pouze jedno slovo - "true" pokud jde o fakticky stejné autory, 
nebo "false" pokud nejde o stejné autory.
    """
    
    # Volání API a zpracování odpovědi
    response = call_openai_api(prompt)
    
    # Logování pro diagnostiku
    # print(f"LLM odpověď na porovnání autorů: {response}")
    
    # Vrácení výsledku
    if response.lower() in ["true", "true.", "ano", "yes"]:
        return True
    elif response == "error":
        print("VAROVÁNÍ: Chyba při volání LLM API, používám fallback na algoritmické porovnání")
        return False  # Při chybě defaultujeme na False a použijeme algoritmické porovnání
    else:
        return False


def normalize_authors(authors_text):
    """
    Normalizuje text autorů pro porovnání.
    
    Args:
        authors_text (str): Text s autory
        
    Returns:
        list: Seznam normalizovaných jmen autorů
    """
    if not authors_text or pd.isna(authors_text):
        return []
    
    # Nahrazení oddělovačů
    authors_text = authors_text.replace("||", ",")
    
    # Rozdělení na jednotlivé autory
    authors = [a.strip() for a in re.split(r',|\n', authors_text) if a.strip()]
    
    # Normalizace každého autora
    normalized_authors = []
    for author in authors:
        # Odstranění diakritiky
        author = remove_diacritics(author)
        
        # Odstranění titulů a dalších částí
        author = re.sub(r'\s*\(.*?\)\s*', ' ', author)
        author = re.sub(r'\s*-\s*.*?$', '', author)
        
        # Rozdělení na jméno a příjmení
        parts = author.split(',')
        if len(parts) == 2:
            # Formát "Příjmení, Jméno"
            surname, firstname = parts
            normalized = f"{firstname.strip()} {surname.strip()}".lower()
        else:
            # Formát "Jméno Příjmení" nebo jiný
            normalized = author.lower()
        
        # Odstranění nadbytečných mezer
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        if normalized:
            normalized_authors.append(normalized)
    
    return normalized_authors


def compare_author_sets(extracted, reference):
    """
    Porovná dvě sady autorů a určí, zda jsou fakticky shodné.
    
    Args:
        extracted (list): Seznam extrahovaných autorů
        reference (list): Seznam referenčních autorů
        
    Returns:
        bool: True pokud jsou sady fakticky shodné, jinak False
    """
    if not extracted or not reference:
        return False
    
    # Pokud se počet autorů významně liší, nejsou shodné
    if abs(len(extracted) - len(reference)) > max(1, min(len(extracted), len(reference)) // 3):
        return False
    
    # Příprava pro fuzzy matching
    matches = []
    reference_copy = reference.copy()
    
    for ext_author in extracted:
        best_match = None
        best_score = 0
        
        for i, ref_author in enumerate(reference_copy):
            # Použijeme různé metriky a bereme tu nejlepší
            score1 = SequenceMatcher(None, ext_author, ref_author).ratio()
            
            # Porovnáme části jména (křestní, příjmení)
            ext_parts = set(p for p in ext_author.split() if len(p) > 1)
            ref_parts = set(p for p in ref_author.split() if len(p) > 1)
            
            if ext_parts and ref_parts:
                # Kolik částí jména se shoduje
                matching_parts = ext_parts.intersection(ref_parts)
                parts_score = len(matching_parts) / max(len(ext_parts), len(ref_parts))
                score2 = parts_score
            else:
                score2 = 0
            
            # Vybereme lepší skóre
            score = max(score1, score2)
            
            if score > best_score and score > 0.5:  # Threshold
                best_score = score
                best_match = (i, ref_author)
        
        if best_match:
            matches.append((ext_author, best_match[1], best_score))
            # Odstraníme použitý referenční prvek
            reference_copy.pop(best_match[0])
    
    # Pokud máme dostatek shod, považujeme sady za shodné
    match_threshold = min(len(extracted), len(reference)) * 0.7
    return len(matches) >= match_threshold


def normalize_doi(doi):
    """
    Normalizuje DOI pro porovnání.
    
    Args:
        doi (str): DOI k normalizaci
        
    Returns:
        str: Normalizovaný DOI
    """
    if not isinstance(doi, str) or pd.isna(doi):
        return ""
    
    # Odstranění prefixů jako https://doi.org/
    doi = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', doi)
    
    # Odstranění nadbytečných mezer a převod na malá písmena
    doi = doi.lower().strip()
    
    return doi


def compare_dois(extracted, reference):
    """
    Porovná dva DOI a určí, zda jsou fakticky shodné.
    
    Args:
        extracted (str): Extrahovaný DOI
        reference (str): Referenční DOI
        
    Returns:
        bool: True pokud jsou DOI fakticky shodné, jinak False
    """
    if not extracted or not reference or pd.isna(extracted) or pd.isna(reference):
        return False
    
    norm_extracted = normalize_doi(extracted)
    norm_reference = normalize_doi(reference)
    
    # Přímá shoda
    if norm_extracted == norm_reference:
        return True
    
    # Fuzzy matching pro případ malých rozdílů
    similarity = SequenceMatcher(None, norm_extracted, norm_reference).ratio()
    return similarity > 0.9


def normalize_journal(journal):
    """
    Normalizuje název časopisu pro porovnání.
    
    Args:
        journal (str): Název časopisu
        
    Returns:
        str: Normalizovaný název časopisu
    """
    if not isinstance(journal, str) or pd.isna(journal):
        return ""
    
    # Odstranění diakritiky
    journal = remove_diacritics(journal)
    
    # Odstranění nadbytečných částí
    journal = re.sub(r'\s*\(.*?\)\s*', ' ', journal)
    
    # Odstranění běžných přípon/předpon časopisů
    journal = re.sub(r'-Switzerland$', '', journal)
    journal = re.sub(r'-Journal$', '', journal)
    
    # Převod na malá písmena a odstranění mezer
    journal = journal.lower().strip()
    
    return journal


def compare_journals(extracted, reference):
    """
    Porovná dva názvy časopisů a určí, zda jsou fakticky shodné.
    
    Args:
        extracted (str): Extrahovaný název časopisu
        reference (str): Referenční název časopisu
        
    Returns:
        bool: True pokud jsou názvy fakticky shodné, jinak False
    """
    if not extracted or not reference or pd.isna(extracted) or pd.isna(reference):
        return False
    
    norm_extracted = normalize_journal(extracted)
    norm_reference = normalize_journal(reference)
    
    # Přímá shoda
    if norm_extracted == norm_reference:
        return True
    
    # Součástí jeden druhého
    if norm_extracted in norm_reference or norm_reference in norm_extracted:
        return True
    
    # Fuzzy matching pro případ malých rozdílů
    similarity = SequenceMatcher(None, norm_extracted, norm_reference).ratio()
    return similarity > 0.8


def normalize_publisher(publisher):
    """
    Normalizuje název vydavatele pro porovnání.
    
    Args:
        publisher (str): Název vydavatele
        
    Returns:
        str: Normalizovaný název vydavatele
    """
    if not isinstance(publisher, str) or pd.isna(publisher):
        return ""
    
    # Odstranění diakritiky
    publisher = remove_diacritics(publisher)
    
    # Zkrácené a plné názvy
    publisher = publisher.replace("MDPI", "Multidisciplinary Digital Publishing Institute")
    
    # Převod na malá písmena a odstranění mezer
    publisher = publisher.lower().strip()
    
    return publisher


def compare_publishers(extracted, reference):
    """
    Porovná dva názvy vydavatelů a určí, zda jsou fakticky shodné.
    
    Args:
        extracted (str): Extrahovaný název vydavatele
        reference (str): Referenční název vydavatele
        
    Returns:
        bool: True pokud jsou názvy fakticky shodné, jinak False
    """
    if not extracted or not reference or pd.isna(extracted) or pd.isna(reference):
        return False
    
    norm_extracted = normalize_publisher(extracted)
    norm_reference = normalize_publisher(reference)
    
    # Přímá shoda
    if norm_extracted == norm_reference:
        return True
    
    # Součástí jeden druhého
    if norm_extracted in norm_reference or norm_reference in norm_extracted:
        return True
    
    # Fuzzy matching pro případ malých rozdílů
    similarity = SequenceMatcher(None, norm_extracted, norm_reference).ratio()
    return similarity > 0.7


def normalize_keywords(keywords_text):
    """
    Normalizuje klíčová slova pro porovnání.
    
    Args:
        keywords_text (str): Text s klíčovými slovy
        
    Returns:
        set: Množina normalizovaných klíčových slov
    """
    if not isinstance(keywords_text, str) or pd.isna(keywords_text):
        return set()
    
    # Nahrazení oddělovačů
    keywords_text = keywords_text.replace("||", ",")
    
    # Rozdělení na jednotlivá klíčová slova
    keywords = [k.strip() for k in re.split(r',|\n', keywords_text) if k.strip()]
    
    # Normalizace každého klíčového slova
    normalized_keywords = set()
    for keyword in keywords:
        # Odstranění diakritiky
        keyword = remove_diacritics(keyword)
        
        # Převod na malá písmena
        keyword = keyword.lower().strip()
        
        if keyword:
            normalized_keywords.add(keyword)
    
    return normalized_keywords


def compare_keywords(extracted, reference):
    """
    Porovná dvě sady klíčových slov a určí faktickou shodu.
    
    Args:
        extracted (str): Extrahovaná klíčová slova
        reference (str): Referenční klíčová slova
        
    Returns:
        float: Skóre shody v rozsahu 0-1
    """
    if not extracted or not reference or pd.isna(extracted) or pd.isna(reference):
        return 0.0
    
    norm_extracted = normalize_keywords(extracted)
    norm_reference = normalize_keywords(reference)
    
    if not norm_extracted or not norm_reference:
        return 0.0
    
    # Počet shodných klíčových slov
    common_keywords = norm_extracted.intersection(norm_reference)
    
    # Jaccard koeficient
    jaccard = len(common_keywords) / len(norm_extracted.union(norm_reference))
    
    # Procento shodných klíčových slov z menší sady
    recall = len(common_keywords) / min(len(norm_extracted), len(norm_reference))
    
    # Kombinované skóre
    score = (jaccard + recall) / 2
    
    # Pokud je dostatečně vysoké, považujeme za shodné
    return 1.0 if score > 0.7 else score


def semantic_compare_and_update(comparison_data):
    """
    Provede sémantické porovnání extrahovaných a referenčních hodnot 
    a aktualizuje similarity skóre.
    
    Args:
        comparison_data (dict): Data porovnání z JSON souborů
        
    Returns:
        dict: Aktualizovaná data porovnání
    """
    # Počítadla úprav pro statistiky
    updates_count = {
        "authors": 0,
        "authors_llm": 0,  # Počet změn pomocí LLM
        "authors_algo": 0, # Počet změn pomocí algoritmu
        "doi": 0,
        "journal": 0,
        "publisher": 0,
        "keywords": 0,
        "total": 0
    }
    
    # Projdeme všechny články
    for paper_id, paper_data in comparison_data["comparison"].items():
        # Autoři
        if "authors" in paper_data and paper_data["authors"].get("similarity", 0) < 0.7:
            extracted = paper_data["authors"].get("extracted", "")
            reference = paper_data["authors"].get("reference", "")
            
            authors_match = False
            llm_used = False
            
            # Nejprve zkusíme LLM porovnání, pokud je povoleno
            if USE_LLM_FOR_AUTHORS and OPENAI_API_KEY and extracted and reference:
                try:
                    print(f"Používám LLM pro porovnání autorů dokumentu {paper_id}...")
                    authors_match = compare_authors_with_llm(extracted, reference)
                    llm_used = True
                    
                    if authors_match:
                        paper_data["authors"]["similarity"] = 1.0
                        paper_data["authors"]["note"] = "Sémanticky shodné (LLM)"
                        updates_count["authors"] += 1
                        updates_count["authors_llm"] += 1
                        updates_count["total"] += 1
                        print(f"✓ LLM identifikoval shodné autory pro dokument {paper_id}")
                    else:
                        print(f"✗ LLM nenašel shodu autorů pro dokument {paper_id}")
                
                except Exception as e:
                    print(f"Chyba při LLM porovnání autorů pro paper_id={paper_id}: {e}")
                    llm_used = False
            
            # Pokud LLM selhal nebo nenašel shodu, použijeme algoritmické porovnání
            if not llm_used or not authors_match:
                try:
                    norm_extracted = normalize_authors(extracted)
                    norm_reference = normalize_authors(reference)
                    
                    authors_match = compare_author_sets(norm_extracted, norm_reference)
                    
                    if authors_match:
                        paper_data["authors"]["similarity"] = 1.0
                        paper_data["authors"]["note"] = "Sémanticky shodné (algoritmus)"
                        updates_count["authors"] += 1
                        updates_count["authors_algo"] += 1
                        updates_count["total"] += 1
                except Exception as e:
                    print(f"Chyba při algoritmickém porovnání autorů pro paper_id={paper_id}: {e}")
        
        # DOI
        if "doi" in paper_data and paper_data["doi"].get("similarity", 0) < 0.9:
            extracted = paper_data["doi"].get("extracted", "")
            reference = paper_data["doi"].get("reference", "")
            
            if compare_dois(extracted, reference):
                paper_data["doi"]["similarity"] = 1.0
                paper_data["doi"]["note"] = "Sémanticky shodné (opraveno)"
                updates_count["doi"] += 1
                updates_count["total"] += 1
        
        # Časopis
        if "journal" in paper_data and paper_data["journal"].get("similarity", 0) < 0.9:
            extracted = paper_data["journal"].get("extracted", "")
            reference = paper_data["journal"].get("reference", "")
            
            if compare_journals(extracted, reference):
                paper_data["journal"]["similarity"] = 1.0
                paper_data["journal"]["note"] = "Sémanticky shodné (opraveno)"
                updates_count["journal"] += 1
                updates_count["total"] += 1
                
        # Vydavatel
        if "publisher" in paper_data and paper_data["publisher"].get("similarity", 0) < 0.9:
            extracted = paper_data["publisher"].get("extracted", "")
            reference = paper_data["publisher"].get("reference", "")
            
            if compare_publishers(extracted, reference):
                paper_data["publisher"]["similarity"] = 1.0
                paper_data["publisher"]["note"] = "Sémanticky shodné (opraveno)"
                updates_count["publisher"] += 1
                updates_count["total"] += 1
        
        # Klíčová slova
        if "keywords" in paper_data and paper_data["keywords"].get("similarity", 0) < 0.9:
            extracted = paper_data["keywords"].get("extracted", "")
            reference = paper_data["keywords"].get("reference", "")
            
            score = compare_keywords(extracted, reference)
            if score == 1.0:
                paper_data["keywords"]["similarity"] = 1.0
                paper_data["keywords"]["note"] = "Sémanticky shodné (opraveno)"
                updates_count["keywords"] += 1
                updates_count["total"] += 1
            elif score > paper_data["keywords"].get("similarity", 0):
                paper_data["keywords"]["similarity"] = score
                paper_data["keywords"]["note"] = "Sémanticky opraveno"
                updates_count["keywords"] += 1
                updates_count["total"] += 1
        
        # Aktualizace celkového similarity skóre
        if "overall_similarity" in paper_data:
            # Vypočítáme průměr ze všech polí
            similarities = [
                data["similarity"] 
                for field, data in paper_data.items() 
                if field != "overall_similarity" and isinstance(data, dict) and "similarity" in data
            ]
            
            if similarities:
                paper_data["overall_similarity"] = np.mean(similarities)
    
    # Výpis statistik aktualizací
    print(f"\nProvedené sémantické aktualizace:")
    for field, count in updates_count.items():
        if field not in ["total", "authors_llm", "authors_algo"]:
            print(f"- {field}: {count}x")
    
    # Detailnější výpis pro autory
    if updates_count["authors"] > 0:
        print(f"  • autoři pomocí LLM: {updates_count['authors_llm']}x")
        print(f"  • autoři pomocí algoritmu: {updates_count['authors_algo']}x")
    
    print(f"Celkem provedeno {updates_count['total']} aktualizací")
    
    # Aktualizace celkových metrik
    update_overall_metrics(comparison_data)
    
    return comparison_data


def update_overall_metrics(comparison_data):
    """
    Aktualizuje celkové metriky po sémantickém porovnání.
    
    Args:
        comparison_data (dict): Data porovnání
        
    Returns:
        dict: Data porovnání s aktualizovanými metrikami
    """
    if "metrics" not in comparison_data:
        comparison_data["metrics"] = {}
    
    # Vytvoříme strukturu pro všechny pole
    all_fields = set()
    for paper_data in comparison_data["comparison"].values():
        all_fields.update([
            field for field in paper_data.keys() 
            if field != "overall_similarity" and isinstance(paper_data[field], dict)
        ])
    
    # Inicializace metrik pro každé pole
    for field in all_fields:
        comparison_data["metrics"][field] = {
            "values": [],
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0
        }
    
    # Přidání položky pro overall
    comparison_data["metrics"]["overall"] = {
        "values": [],
        "mean": 0.0,
        "median": 0.0,
        "min": 0.0,
        "max": 0.0,
        "count": 0
    }
    
    # Shromáždíme hodnoty
    for paper_data in comparison_data["comparison"].values():
        # Pole
        for field in all_fields:
            if field in paper_data and "similarity" in paper_data[field]:
                comparison_data["metrics"][field]["values"].append(paper_data[field]["similarity"])
        
        # Overall
        if "overall_similarity" in paper_data:
            comparison_data["metrics"]["overall"]["values"].append(paper_data["overall_similarity"])
    
    # Vypočítáme statistiky
    for field, metrics in comparison_data["metrics"].items():
        values = metrics["values"]
        if values:
            metrics["mean"] = float(np.mean(values))
            metrics["median"] = float(np.median(values))
            metrics["min"] = float(min(values))
            metrics["max"] = float(max(values))
            metrics["count"] = len(values)
            
            # Odstraníme seznam hodnot pro úsporu místa
            del metrics["values"]
    
    return comparison_data


def load_json_with_nan_handling(file_path):
    """
    Načte JSON soubor s podporou pro NaN hodnoty.
    
    Args:
        file_path (str): Cesta k JSON souboru
        
    Returns:
        dict: Načtená data
    """
    # Vlastní parser pro nan
    class NanHandler(json.JSONDecoder):
        def __init__(self, *args, **kwargs):
            json.JSONDecoder.__init__(self, *args, **kwargs)
            self.parse_string = self._parse_string
            self.scan_once = json.scanner.py_make_scanner(self)
            
        def _parse_string(self, s, end):
            if s[end-3:end] == 'NaN':
                return float('nan'), end
            return json.JSONDecoder.parse_string(self, s, end)
    
    try:
        # Nejprve zkusíme standardní načtení
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Pokud selže, zkusíme upravit soubor a načíst znovu
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Nahrazení NaN za null v JSON
            content = re.sub(r':\s*NaN', ': null', content)
            
            # Načtení upraveného JSON
            return json.loads(content)
        except Exception as e:
            print(f"Chyba při načítání JSON souboru {file_path}: {e}")
            # Vytvoříme minimální strukturu
            return {"comparison": {}, "metrics": {}}


def process_comparison_files(vlm_comparison_path, embedded_comparison_path, output_path=None):
    """
    Zpracuje soubory s porovnáním a vytvoří sémanticky vylepšené porovnání.
    
    Args:
        vlm_comparison_path (str): Cesta k souboru vlm_comparison.json
        embedded_comparison_path (str): Cesta k souboru embedded_comparison.json
        output_path (str, optional): Cesta pro výstupní soubor
        
    Returns:
        dict, dict: Dvojice aktualizovaných dat porovnání (vlm, embedded)
    """
    # Načtení souborů s podporou pro NaN hodnoty
    vlm_data = load_json_with_nan_handling(vlm_comparison_path)
    embedded_data = load_json_with_nan_handling(embedded_comparison_path)
    
    # Zpracování dat
    print("\nZpracování VLM dat...")
    vlm_updated = semantic_compare_and_update(vlm_data)
    
    print("\nZpracování Embedded dat...")
    embedded_updated = semantic_compare_and_update(embedded_data)
    
    # Uložení výsledků
    if output_path:
        # Vytvoříme adresáře, pokud neexistují
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Uložíme výsledky do zadaného souboru
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "vlm": vlm_updated,
                "embedded": embedded_updated
            }, f, ensure_ascii=False, indent=2)
    
    # Vytvoříme samostatné aktualizované soubory
    vlm_output = str(vlm_comparison_path).replace('.json', '_semantic.json')
    with open(vlm_output, 'w', encoding='utf-8') as f:
        json.dump(vlm_updated, f, ensure_ascii=False, indent=2)
    
    embedded_output = str(embedded_comparison_path).replace('.json', '_semantic.json')
    with open(embedded_output, 'w', encoding='utf-8') as f:
        json.dump(embedded_updated, f, ensure_ascii=False, indent=2)
    
    return vlm_updated, embedded_updated


if __name__ == "__main__":
    # Cesty k souborům
    vlm_comparison_path = RESULTS_DIR / "vlm_comparison.json"
    embedded_comparison_path = RESULTS_DIR / "embedded_comparison.json"
    output_path = RESULTS_DIR / "semantic_comparison_results.json"
    
    # Zpracování
    vlm_updated, embedded_updated = process_comparison_files(
        vlm_comparison_path, 
        embedded_comparison_path,
        output_path
    )
    
    print(f"\nSémanticky vylepšené porovnání uloženo do {output_path}")
    print(f"Samostatné soubory uloženy jako:")
    print(f"- {vlm_comparison_path.name.replace('.json', '_semantic.json')}")
    print(f"- {embedded_comparison_path.name.replace('.json', '_semantic.json')}") 