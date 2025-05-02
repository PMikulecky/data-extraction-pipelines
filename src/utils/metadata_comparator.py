#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro porovnání metadat extrahovaných z PDF souborů s referenčními daty.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import difflib
from datetime import datetime


class MetadataComparator:
    """
    Třída pro porovnání metadat extrahovaných z PDF souborů s referenčními daty.
    """
    
    # Definice metadat, která budou porovnávána
    METADATA_FIELDS = [
        'title',           # Název práce
        'authors',         # Seznam autorů
        'abstract',        # Abstrakt
        'keywords',        # Klíčová slova
        'doi',             # DOI
        'year',            # Rok vydání
        'journal',         # Název časopisu/konference
        'volume',          # Ročník
        'issue',           # Číslo
        'pages',           # Stránky
        'publisher',       # Vydavatel
        'references'       # Seznam referencí
    ]
    
    def __init__(self, reference_data):
        """
        Inicializace komparátoru metadat.
        
        Args:
            reference_data (dict): Referenční data pro porovnání
        """
        self.reference_data = reference_data
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def compare_metadata(self, extracted_data, paper_id):
        """
        Porovná extrahovaná metadata s referenčními daty.
        
        Args:
            extracted_data (dict): Extrahovaná metadata
            paper_id (str): ID akademické práce
            
        Returns:
            dict: Výsledky porovnání
        """
        if paper_id not in self.reference_data:
            raise ValueError(f"Referenční data pro ID {paper_id} nebyla nalezena.")
        
        reference = self.reference_data[paper_id]
        results = {}
        
        # Porovnání jednotlivých metadat
        for field in self.METADATA_FIELDS:
            if field in extracted_data and field in reference:
                similarity = self._calculate_similarity(extracted_data[field], reference[field], field)
                results[field] = {
                    'similarity': similarity,
                    'extracted': extracted_data[field],
                    'reference': reference[field]
                }
            else:
                results[field] = {
                    'similarity': 0.0,
                    'extracted': extracted_data.get(field, None),
                    'reference': reference.get(field, None)
                }
        
        # Výpočet celkové podobnosti
        valid_similarities = [results[field]['similarity'] for field in self.METADATA_FIELDS 
                             if field in results and results[field]['similarity'] is not None]
        
        results['overall_similarity'] = np.mean(valid_similarities) if valid_similarities else None
        
        return results
    
    def _calculate_similarity(self, extracted, reference, field_type):
        """
        Vypočítá podobnost mezi extrahovanými a referenčními daty.
        
        Args:
            extracted: Extrahovaná data
            reference: Referenční data
            field_type (str): Typ porovnávaného pole
            
        Returns:
            float: Hodnota podobnosti (0.0 - 1.0)
        """
        # Kontrola, zda jsou obě hodnoty k dispozici a reference není NaN
        if extracted is None or reference is None or pd.isna(reference):
            return None # Vracíme None, pokud nelze porovnat
        
        # Převod na řetězce
        extracted_str = str(extracted).lower()
        reference_str = str(reference).lower()
        
        # Prázdné řetězce
        if not extracted_str or not reference_str:
            return 0.0
        
        # Různé metody porovnání podle typu pole
        if field_type in ['title', 'abstract']:
            # Pro delší texty použijeme TF-IDF a kosinovou podobnost
            try:
                tfidf_matrix = self.vectorizer.fit_transform([extracted_str, reference_str])
                return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                # Fallback na porovnání sekvencí
                return difflib.SequenceMatcher(None, extracted_str, reference_str).ratio()
        
        elif field_type == 'authors':
            # Pro autory porovnáme jednotlivá jména
            return self._compare_authors(extracted_str, reference_str)
        
        elif field_type == 'year':
            # Pro rok porovnáme přesnou shodu
            extracted_year = self._extract_year(extracted_str)
            reference_year = self._extract_year(reference_str)
            return 1.0 if extracted_year == reference_year else 0.0
        
        elif field_type == 'doi':
            # Pro DOI porovnáme přesnou shodu (bez ohledu na velikost písmen)
            return 1.0 if extracted_str.strip() == reference_str.strip() else 0.0
        
        elif field_type in ['volume', 'issue', 'pages']:
            # Pro číselné údaje porovnáme přesnou shodu
            extracted_clean = re.sub(r'[^\d-]', '', extracted_str)
            reference_clean = re.sub(r'[^\d-]', '', reference_str)
            return 1.0 if extracted_clean == reference_clean else 0.0
        
        elif field_type == 'references':
            # Pro reference porovnáme počet a podobnost
            return self._compare_references(extracted_str, reference_str)
        
        else:
            # Pro ostatní pole použijeme porovnání sekvencí
            return difflib.SequenceMatcher(None, extracted_str, reference_str).ratio()
    
    def _compare_authors(self, extracted_authors, reference_authors):
        """
        Porovná seznamy autorů.
        
        Args:
            extracted_authors (str): Extrahovaní autoři
            reference_authors (str): Referenční autoři
            
        Returns:
            float: Hodnota podobnosti (0.0 - 1.0)
        """
        # Rozdělení řetězců na jednotlivé autory
        extracted_list = self._split_authors(extracted_authors)
        reference_list = self._split_authors(reference_authors)
        
        if not extracted_list or not reference_list:
            return 0.0
        
        # Výpočet podobnosti pro každého autora
        matches = 0
        for ext_author in extracted_list:
            best_match = max([difflib.SequenceMatcher(None, ext_author, ref_author).ratio() 
                             for ref_author in reference_list])
            if best_match > 0.8:  # Práh pro shodu
                matches += 1
        
        # Výpočet F1 skóre (kombinace přesnosti a úplnosti)
        precision = matches / len(extracted_list) if extracted_list else 0
        recall = matches / len(reference_list) if reference_list else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _split_authors(self, authors_str):
        """
        Rozdělí řetězec autorů na seznam jednotlivých autorů.
        
        Args:
            authors_str (str): Řetězec autorů
            
        Returns:
            list: Seznam jednotlivých autorů
        """
        # Odstranění běžných oddělovačů
        authors_str = authors_str.replace(' and ', ', ')
        
        # Rozdělení podle čárky nebo středníku
        authors = re.split(r'[,;]\s*', authors_str)
        
        # Odstranění prázdných řetězců a normalizace
        return [author.strip().lower() for author in authors if author.strip()]
    
    def _extract_year(self, year_str):
        """
        Extrahuje rok z řetězce.
        
        Args:
            year_str (str): Řetězec obsahující rok
            
        Returns:
            str: Extrahovaný rok
        """
        # Hledání 4místného čísla, které může být rokem
        match = re.search(r'(19|20)\d{2}', year_str)
        if match:
            year = match.group(0)
            # Kontrola, zda je rok platný (ne v budoucnosti)
            current_year = datetime.now().year
            if 1900 <= int(year) <= current_year:
                return year
        
        return year_str.strip()
    
    def _compare_references(self, extracted_refs, reference_refs):
        """
        Porovná seznamy referencí.
        
        Args:
            extracted_refs (str): Extrahované reference
            reference_refs (str): Referenční reference
            
        Returns:
            float: Hodnota podobnosti (0.0 - 1.0)
        """
        # Rozdělení na jednotlivé reference
        extracted_list = self._split_references(extracted_refs)
        reference_list = self._split_references(reference_refs)
        
        if not extracted_list or not reference_list:
            return 0.0
        
        # Porovnání počtu referencí
        count_similarity = min(len(extracted_list), len(reference_list)) / max(len(extracted_list), len(reference_list))
        
        # Porovnání obsahu referencí (náhodný vzorek pro efektivitu)
        content_similarities = []
        sample_size = min(10, min(len(extracted_list), len(reference_list)))
        
        for i in range(sample_size):
            idx = int(i * len(reference_list) / sample_size)
            ref = reference_list[idx]
            
            # Hledání nejlepší shody v extrahovaných referencích
            best_match = max([difflib.SequenceMatcher(None, ref, ext_ref).ratio() 
                             for ext_ref in extracted_list])
            content_similarities.append(best_match)
        
        # Kombinace podobnosti počtu a obsahu
        content_similarity = np.mean(content_similarities) if content_similarities else 0.0
        return 0.6 * count_similarity + 0.4 * content_similarity
    
    def _split_references(self, refs_str):
        """
        Rozdělí řetězec referencí na seznam jednotlivých referencí.
        
        Args:
            refs_str (str): Řetězec referencí
            
        Returns:
            list: Seznam jednotlivých referencí
        """
        # Rozdělení podle čísel nebo prázdných řádků
        lines = refs_str.split('\n')
        references = []
        current_ref = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_ref:
                    references.append(current_ref)
                    current_ref = ""
            elif re.match(r'^\[\d+\]|\d+\.', line):
                if current_ref:
                    references.append(current_ref)
                current_ref = line
            else:
                current_ref += " " + line if current_ref else line
        
        if current_ref:
            references.append(current_ref)
        
        return [ref.strip().lower() for ref in references if ref.strip()]


def compare_all_metadata(extracted_data_dict, reference_data_dict):
    """
    Porovná všechna extrahovaná metadata s referenčními daty.
    
    Args:
        extracted_data_dict (dict): Slovník s extrahovanými metadaty pro každou práci
        reference_data_dict (dict): Slovník s referenčními daty pro každou práci
        
    Returns:
        dict: Výsledky porovnání pro každou práci
    """
    comparator = MetadataComparator(reference_data_dict)
    results = {}
    
    for paper_id, extracted_data in extracted_data_dict.items():
        if paper_id in reference_data_dict:
            try:
                results[paper_id] = comparator.compare_metadata(extracted_data, paper_id)
            except Exception as e:
                print(f"Chyba při porovnávání metadat pro ID {paper_id}: {e}")
                results[paper_id] = {'error': str(e)}
        else:
            print(f"Referenční data pro ID {paper_id} nebyla nalezena.")
    
    return results


def calculate_overall_metrics(comparison_results):
    """
    Vypočítá celkové metriky pro všechny porovnané práce.
    
    Args:
        comparison_results (dict): Výsledky porovnání pro každou práci
        
    Returns:
        dict: Celkové metriky
    """
    metrics = {field: [] for field in MetadataComparator.METADATA_FIELDS}
    metrics['overall'] = []
    
    for paper_id, results in comparison_results.items():
        if 'error' in results:
            continue
        
        # Přidáme overall_similarity pouze pokud není None
        overall_sim = results.get('overall_similarity')
        if overall_sim is not None:
            metrics['overall'].append(overall_sim)
        
        for field in MetadataComparator.METADATA_FIELDS:
            if field in results and 'similarity' in results[field]:
                # Přidáme similarity pouze pokud není None
                field_sim = results[field]['similarity']
                if field_sim is not None:
                    metrics[field].append(field_sim)
    
    # Výpočet průměrných hodnot
    averages = {}
    for field, values in metrics.items():
        if values:
            averages[field] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        else:
            # Pokud nejsou žádné validní hodnoty, nastavíme metriky na None nebo NaN
            averages[field] = {
                'mean': np.nan,
                'median': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0
            }
    
    return averages


if __name__ == "__main__":
    # Příklad použití
    import json
    
    # Testovací data
    reference_data = {
        '1': {
            'title': 'Example Paper Title',
            'authors': 'John Smith, Jane Doe',
            'year': '2020'
        }
    }
    
    extracted_data = {
        '1': {
            'title': 'Example Paper Title with a Small Difference',
            'authors': 'Smith J., Doe J.',
            'year': '2020'
        }
    }
    
    # Porovnání
    results = compare_all_metadata(extracted_data, reference_data)
    metrics = calculate_overall_metrics(results)
    
    print(json.dumps(results, indent=2))
    print("\nCelkové metriky:")
    print(json.dumps(metrics, indent=2)) 