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


def normalize_single_reference(ref):
    """
    Normalizuje jednu referenční položku pro porovnání.
    Odstraní čísla na začátku, závorky a další formátovací znaky.
    """
    import re
    
    # Odstranění čísel na začátku a závorek (např. [1], (1), 1.)
    ref = re.sub(r'^\s*[\[\(]?\d+[\]\)]?\.?\s*', '', ref.strip())
    
    # Odstranění DOI prefixů a URL
    ref = re.sub(r'https?://(?:dx\.)?doi\.org/|doi:\s*', '', ref)
    
    # Odstranění dalších URL
    ref = re.sub(r'https?://\S+', '', ref)
    
    # Odstranění diakritiky
    ref = remove_diacritics(ref)
    
    # Převod na malá písmena a odstranění nadbytečných mezer
    ref = ' '.join(ref.lower().split())
    
    return ref


def split_references(references_text):
    """
    Rozdělí text s referencemi na jednotlivé citace.
    """
    import re
    
    # Zkusíme najít vzory pro rozdělení referencí
    # Běžné vzory: [1] Autor..., 1. Autor..., (1) Autor...
    if isinstance(references_text, list):
        return references_text
    
    # Rozdělení podle běžných vzorů
    patterns = [
        r'(?:\[\d+\]|\(\d+\)|\d+\.)\s+',  # [1], (1), 1.
        r'\n\s*(?=\[\d+\]|\(\d+\)|\d+\.)', # Nový řádek následovaný číslem
        r'\r\n\s*(?=\[\d+\]|\(\d+\)|\d+\.)', # Windows nový řádek 
        r'\n{2,}',  # Dvojité nové řádky
    ]
    
    # Zkusíme každý vzor na rozdělení, dokud nedostaneme více než jednu položku
    references = [references_text]
    for pattern in patterns:
        refs = re.split(pattern, references_text)
        refs = [r.strip() for r in refs if r.strip()]
        if len(refs) > 1:
            references = refs
            break
    
    return references


def compare_references_with_llm(extracted_refs, reference_refs, max_refs=5):
    """
    Porovná reference pomocí LLM pro určení sémantické podobnosti.
    Pro efektivitu omezíme počet porovnávaných referencí.
    """
    if not extracted_refs or not reference_refs:
        return 0.0
    
    # Pro efektivitu porovnáme jen omezený počet referencí
    extracted_sample = extracted_refs[:max_refs]
    reference_sample = reference_refs[:max_refs]
    
    matched_count = 0
    total_comparisons = min(len(extracted_sample), len(reference_sample))
    
    if total_comparisons == 0:
        return 0.0
    
    for i, ext_ref in enumerate(extracted_sample):
        if i >= len(reference_sample):
            break
            
        ref_ref = reference_sample[i]
        
        prompt = f"""
        Porovnej tyto dvě vědecké citace a rozhodni, zda se jedná o stejný zdroj, i když mohou být v jiném formátu:
        
        Citace 1: {ext_ref}
        
        Citace 2: {ref_ref}
        
        Vrať pouze 'true', pokud se jedná o stejný zdroj (stejní autoři, název, rok), nebo 'false', pokud jde o různé zdroje.
        """
        
        try:
            result = call_openai_api(prompt)
            if "true" in result.lower():
                matched_count += 1
        except Exception as e:
            print(f"Chyba při porovnávání referencí pomocí LLM: {e}")
            # Fallback na textové porovnání
            if compare_references_algorithmically(ext_ref, ref_ref) > 0.5:
                matched_count += 1
    
    return matched_count / total_comparisons


def compare_references_algorithmically(extracted, reference):
    """
    Algoritmické porovnání referencí bez použití LLM.
    """
    from difflib import SequenceMatcher
    
    # Normalizace textů
    extracted_norm = normalize_single_reference(extracted)
    reference_norm = normalize_single_reference(reference)
    
    # Základní podobnost textu
    base_similarity = SequenceMatcher(None, extracted_norm, reference_norm).ratio()
    
    # Extrahujeme důležité části (autoři, rok, název)
    import re
    
    # Hledání roku (4 číslice v závorce nebo za závorkou)
    extracted_year = re.search(r'\(?(\d{4})\)?', extracted_norm)
    reference_year = re.search(r'\(?(\d{4})\)?', reference_norm)
    
    year_match = False
    if extracted_year and reference_year:
        year_match = extracted_year.group(1) == reference_year.group(1)
    
    # Pokud se rok shoduje, zvýšíme váhu podobnosti
    if year_match:
        base_similarity = (base_similarity + 0.3) / 1.3
    
    return base_similarity


def compare_references(extracted, reference, use_llm=True):
    """
    Sémantické porovnání referencí.
    """
    # Pokud jsou vstupy prázdné
    if not extracted or not reference:
        return 0.0
    
    # Rozdělení na jednotlivé reference
    extracted_refs = split_references(extracted)
    reference_refs = split_references(reference)
    
    # Pokud nemáme reference k porovnání
    if not extracted_refs or not reference_refs:
        return 0.0
    
    # Porovnání délky seznamů referencí (míra pokrytí)
    len_similarity = min(len(extracted_refs), len(reference_refs)) / max(len(extracted_refs), len(reference_refs))
    
    # Porovnání obsahu bez ohledu na pořadí
    if use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            # Omezení počtu porovnávaných referencí pro LLM (kvůli efektivitě)
            max_refs = 10
            extracted_sample = extracted_refs[:max_refs]
            reference_sample = reference_refs[:max_refs]
            
            # Matice podobností pro každý pár referencí
            similarity_matrix = []
            for ext_ref in extracted_sample:
                row = []
                for ref_ref in reference_sample:
                    prompt = f"""
                    Porovnej tyto dvě vědecké citace a rozhodni, zda se jedná o stejný zdroj, i když mohou být v jiném formátu:
                    
                    Citace 1: {ext_ref}
                    
                    Citace 2: {ref_ref}
                    
                    Vrať pouze 'true', pokud se jedná o stejný zdroj (stejní autoři, název, rok), nebo 'false', pokud jde o různé zdroje.
                    """
                    
                    try:
                        result = call_openai_api(prompt)
                        score = 1.0 if "true" in result.lower() else 0.0
                    except Exception as e:
                        print(f"Chyba při porovnávání referencí pomocí LLM: {e}")
                        # Fallback na algoritmické porovnání
                        score = compare_references_algorithmically(ext_ref, ref_ref)
                    
                    row.append(score)
                similarity_matrix.append(row)
            
            # Nalezení nejlepších shod pro každou extrahovanou referenci
            # Použijeme "greedy" přístup - vždy vybereme nejlepší dostupnou shodu
            matches = []
            used_indices = set()
            
            for i, row in enumerate(similarity_matrix):
                best_match_idx = -1
                best_match_score = 0.0
                
                for j, score in enumerate(row):
                    if j not in used_indices and score > best_match_score:
                        best_match_score = score
                        best_match_idx = j
                
                if best_match_idx != -1:
                    matches.append(best_match_score)
                    used_indices.add(best_match_idx)
            
            # Výpočet průměrné podobnosti nalezených shod
            content_similarity = sum(matches) / len(extracted_sample) if extracted_sample else 0.0
            
        except Exception as e:
            print(f"Chyba při použití LLM pro porovnání referencí: {e}")
            use_llm = False
    
    if not use_llm:
        # Algoritmické porovnání s hledáním nejlepších shod
        similarity_matrix = []
        for ext_ref in extracted_refs:
            similarities = [compare_references_algorithmically(ext_ref, ref_ref) for ref_ref in reference_refs]
            similarity_matrix.append(similarities)
        
        # Nalezení nejlepších shod pro každou extrahovanou referenci
        matches = []
        used_indices = set()
        
        for i, row in enumerate(similarity_matrix):
            best_match_idx = -1
            best_match_score = 0.5  # Práh pro považování za shodu
            
            for j, score in enumerate(row):
                if j not in used_indices and score > best_match_score:
                    best_match_score = score
                    best_match_idx = j
            
            if best_match_idx != -1:
                matches.append(best_match_score)
                used_indices.add(best_match_idx)
        
        # Výpočet průměrné podobnosti nalezených shod
        content_similarity = sum(matches) / len(extracted_refs) if extracted_refs else 0.0
    
    # Kombinujeme podobnost délky seznamu a obsahu s větší váhou na obsah
    final_similarity = 0.3 * len_similarity + 0.7 * content_similarity
    
    return final_similarity


def semantic_compare_and_update(comparison_data):
    """
    Provede sémantické porovnání a aktualizuje hodnoty podobnosti.
    """
    # Počítadla pro statistiky
    authors_llm_count = 0
    authors_algo_count = 0
    authors_update_count = 0
    doi_update_count = 0
    journal_update_count = 0
    publisher_update_count = 0
    references_update_count = 0  # Nové počítadlo
    total_updates = 0
    
    # Kontrola, zda jsou k dispozici data pro porovnání
    if "comparison" not in comparison_data:
        return comparison_data
    
    # Procházení dokumentů
    for doc_id, doc_data in comparison_data["comparison"].items():
        # Porovnání a aktualizace autorů
        if "authors" in doc_data:
            orig_similarity = doc_data["authors"]["similarity"]
            extracted = doc_data["authors"]["extracted"]
            reference = doc_data["authors"]["reference"]
            
            # Pokus o sémantické porovnání pomocí LLM
            llm_match = False
            if USE_LLM_FOR_AUTHORS and OPENAI_API_KEY:
                try:
                    llm_match = compare_authors_with_llm(extracted, reference)
                    if llm_match:
                        authors_llm_count += 1
                        print(f"Dokument {doc_id}: LLM identifikoval shodné autory.")
                except Exception as e:
                    print(f"Chyba při porovnávání autorů pomocí LLM: {e}")
            
            # Pokud LLM neidentifikoval shodu nebo není k dispozici, použijeme algoritmické porovnání
            if not llm_match:
                new_similarity = compare_author_sets(extracted, reference)
                if new_similarity > orig_similarity:
                    authors_algo_count += 1
            else:
                new_similarity = 1.0  # Pokud LLM identifikoval shodu, považujeme podobnost za 100%
            
            # Aktualizace podobnosti, pokud je vyšší než původní
            if new_similarity > orig_similarity:
                doc_data["authors"]["similarity"] = new_similarity
                doc_data["authors"]["semantically_identical"] = True
                authors_update_count += 1
                total_updates += 1
                
        # Porovnání a aktualizace DOI
        if "doi" in doc_data:
            orig_similarity = doc_data["doi"]["similarity"]
            extracted = doc_data["doi"]["extracted"]
            reference = doc_data["doi"]["reference"]
            
            new_similarity = compare_dois(extracted, reference)
            if new_similarity > orig_similarity:
                doc_data["doi"]["similarity"] = new_similarity
                doc_data["doi"]["semantically_identical"] = True
                doi_update_count += 1
                total_updates += 1
        
        # Porovnání a aktualizace časopisu
        if "journal" in doc_data:
            orig_similarity = doc_data["journal"]["similarity"]
            extracted = doc_data["journal"]["extracted"]
            reference = doc_data["journal"]["reference"]
            
            new_similarity = compare_journals(extracted, reference)
            if new_similarity > orig_similarity:
                doc_data["journal"]["similarity"] = new_similarity
                doc_data["journal"]["semantically_identical"] = True
                journal_update_count += 1
                total_updates += 1
        
        # Porovnání a aktualizace vydavatele
        if "publisher" in doc_data:
            orig_similarity = doc_data["publisher"]["similarity"]
            extracted = doc_data["publisher"]["extracted"]
            reference = doc_data["publisher"]["reference"]
            
            new_similarity = compare_publishers(extracted, reference)
            if new_similarity > orig_similarity:
                doc_data["publisher"]["similarity"] = new_similarity
                doc_data["publisher"]["semantically_identical"] = True
                publisher_update_count += 1
                total_updates += 1
        
        # Porovnání a aktualizace klíčových slov
        if "keywords" in doc_data:
            orig_similarity = doc_data["keywords"]["similarity"]
            extracted = doc_data["keywords"]["extracted"]
            reference = doc_data["keywords"]["reference"]
            
            new_similarity = compare_keywords(extracted, reference)
            if new_similarity > orig_similarity:
                doc_data["keywords"]["similarity"] = new_similarity
                doc_data["keywords"]["semantically_identical"] = True
                total_updates += 1
        
        # Nové porovnání pro reference
        if "references" in doc_data:
            orig_similarity = doc_data["references"]["similarity"]
            extracted = doc_data["references"]["extracted"]
            reference = doc_data["references"]["reference"]
            
            # Sémantické porovnání referencí
            new_similarity = compare_references(extracted, reference)
    
            # Aktualizace podobnosti, pokud je vyšší než původní
            if new_similarity > orig_similarity:
                doc_data["references"]["similarity"] = new_similarity
                doc_data["references"]["semantically_improved"] = True
                references_update_count += 1
                total_updates += 1
                print(f"Dokument {doc_id}: Sémanticky vylepšené reference.")
    
    # Aktualizace metrik
    update_overall_metrics(comparison_data)
    
    # Výpis statistik
    print(f"\nStatistiky sémantických aktualizací:")
    print(f"Autoři aktualizováni: {authors_update_count}x")
    print(f"DOI aktualizováno: {doi_update_count}x")
    print(f"Časopis aktualizován: {journal_update_count}x")
    print(f"Vydavatel aktualizován: {publisher_update_count}x")
    print(f"Reference aktualizovány: {references_update_count}x")  # Nový výpis
    print(f"Autoři identifikováni pomocí LLM: {authors_llm_count}x")
    print(f"Autoři identifikováni pomocí algoritmu: {authors_algo_count}x")
    print(f"Celkem provedeno aktualizací: {total_updates}")
    
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


def process_comparison_files(vlm_comparison_path, embedded_comparison_path, output_path=None, text_comparison_path=None):
    """
    Zpracuje soubory s porovnáním a vytvoří sémanticky vylepšené porovnání.
    
    Args:
        vlm_comparison_path (str): Cesta k souboru vlm_comparison.json
        embedded_comparison_path (str): Cesta k souboru embedded_comparison.json
        output_path (str, optional): Cesta pro výstupní soubor
        text_comparison_path (str, optional): Cesta k souboru text_comparison.json
        
    Returns:
        tuple: Aktualizovaná data porovnání (vlm, embedded, text) nebo (vlm, embedded) pokud text_comparison_path není zadán
    """
    # Načtení souborů s podporou pro NaN hodnoty
    vlm_data = load_json_with_nan_handling(vlm_comparison_path)
    embedded_data = load_json_with_nan_handling(embedded_comparison_path)
    text_data = None
    
    if text_comparison_path and os.path.exists(text_comparison_path):
        text_data = load_json_with_nan_handling(text_comparison_path)
    
    # Zpracování dat
    print("\nZpracování VLM dat...")
    vlm_updated = semantic_compare_and_update(vlm_data)
    
    print("\nZpracování Embedded dat...")
    embedded_updated = semantic_compare_and_update(embedded_data)
    
    text_updated = None
    if text_data:
        print("\nZpracování Text dat...")
        text_updated = semantic_compare_and_update(text_data)
    
    # Uložení výsledků
    if output_path:
        # Vytvoříme adresáře, pokud neexistují
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Výsledky pro ukládání
        result_data = {
                "vlm": vlm_updated,
                "embedded": embedded_updated
        }
        
        # Přidáme text data, pokud existují
        if text_updated:
            result_data["text"] = text_updated
        
        # Uložíme výsledky do zadaného souboru
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    # Vytvoříme samostatné aktualizované soubory
    vlm_output = str(vlm_comparison_path).replace('.json', '_semantic.json')
    with open(vlm_output, 'w', encoding='utf-8') as f:
        json.dump(vlm_updated, f, ensure_ascii=False, indent=2)
    
    embedded_output = str(embedded_comparison_path).replace('.json', '_semantic.json')
    with open(embedded_output, 'w', encoding='utf-8') as f:
        json.dump(embedded_updated, f, ensure_ascii=False, indent=2)
    
    # Vytvoříme soubor text_semantic, pokud existují text data
    if text_updated:
        text_output = str(text_comparison_path).replace('.json', '_semantic.json')
        with open(text_output, 'w', encoding='utf-8') as f:
            json.dump(text_updated, f, ensure_ascii=False, indent=2)
    
    # Vracíme aktualizovaná data
    if text_updated:
        return vlm_updated, embedded_updated, text_updated
    else:
        return vlm_updated, embedded_updated


if __name__ == "__main__":
    # Cesty k souborům
    vlm_comparison_path = RESULTS_DIR / "vlm_comparison.json"
    embedded_comparison_path = RESULTS_DIR / "embedded_comparison.json"
    text_comparison_path = RESULTS_DIR / "text_comparison.json"
    output_path = RESULTS_DIR / "semantic_comparison_results.json"
    
    # Kontrola, zda existuje text_comparison_path
    text_param = text_comparison_path if os.path.exists(text_comparison_path) else None
    
    # Zpracování
    if text_param:
        vlm_updated, embedded_updated, text_updated = process_comparison_files(
            vlm_comparison_path, 
            embedded_comparison_path,
            output_path,
            text_param
        )
        print(f"- {text_comparison_path.name.replace('.json', '_semantic.json')}")
    else:
        vlm_updated, embedded_updated = process_comparison_files(
            vlm_comparison_path, 
            embedded_comparison_path,
            output_path
        )
    
    print(f"\nSémanticky vylepšené porovnání uloženo do {output_path}")
    print(f"Samostatné soubory uloženy jako:")
    print(f"- {vlm_comparison_path.name.replace('.json', '_semantic.json')}")
    print(f"- {embedded_comparison_path.name.replace('.json', '_semantic.json')}") 