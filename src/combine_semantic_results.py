#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skript pro kombinaci výsledků ze souborů text_comparison_semantic.json a vlm_comparison_semantic.json
a vytvoření nového souboru hybrid_comparison_semantic.json.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import os

def load_json_file(file_path):
    """
    Načte JSON soubor a ošetří NaN hodnoty.
    """
    # Načíst obsah souboru jako text
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Nahradit "NaN" řetězce hodnotou null
    content = content.replace('NaN', 'null')
    
    # Parsovat JSON
    return json.loads(content)

def combine_semantic_results(text_file_path, vlm_file_path, output_file_path):
    """
    Kombinuje výsledky z text a vlm semantic souborů podle pravidel:
    1. Z VLM bereme: title, authors, doi, issue, volume, journal, publisher, year
    2. Z Text bereme: abstract, keywords a cokoliv co chybí ve VLM
    """
    print(f"Načítám text_comparison_semantic.json z: {text_file_path}")
    print(f"Načítám vlm_comparison_semantic.json z: {vlm_file_path}")
    
    try:
        text_data = load_json_file(text_file_path)
        vlm_data = load_json_file(vlm_file_path)
    except Exception as e:
        print(f"Chyba při načítání souborů: {e}")
        return False
    
    # Inicializace hybridního výsledku
    hybrid_data = {
        "comparison": {},
        "metrics": {}
    }
    
    # Vytvoření hybridního porovnání
    print("Kombinuji výsledky...")
    
    # Seznam dokumentů v obou souborech
    text_doc_ids = set(text_data.get("comparison", {}).keys())
    vlm_doc_ids = set(vlm_data.get("comparison", {}).keys())
    all_doc_ids = text_doc_ids.union(vlm_doc_ids)
    
    print(f"Nalezeno {len(text_doc_ids)} dokumentů v text souboru")
    print(f"Nalezeno {len(vlm_doc_ids)} dokumentů v vlm souboru")
    print(f"Celkem {len(all_doc_ids)} unikátních dokumentů")
    
    # Pole, která preferujeme z VLM
    vlm_preferred_fields = ['title', 'authors', 'doi', 'issue', 'volume', 'journal', 'publisher', 'year']
    # Pole, která preferujeme z Text
    text_preferred_fields = ['abstract', 'keywords']
    
    # Pro každý dokument
    for doc_id in all_doc_ids:
        hybrid_data["comparison"][doc_id] = {}
        
        # Získat data dokumentu z obou zdrojů (pokud existují)
        text_doc = text_data.get("comparison", {}).get(doc_id, {})
        vlm_doc = vlm_data.get("comparison", {}).get(doc_id, {})
        
        # Seznam všech polí v obou dokumentech
        all_fields = set()
        if text_doc:
            all_fields.update(text_doc.keys())
        if vlm_doc:
            all_fields.update(vlm_doc.keys())
        
        # Pro každé pole v dokumentu
        for field in all_fields:
            # Výchozí hodnoty
            field_data = None
            
            # Pomocná funkce pro získání hodnoty podobnosti
            def get_similarity(doc, field_name):
                if field_name not in doc:
                    return None
                    
                field_value = doc[field_name]
                
                # Pokud je hodnota slovník s klíčem 'similarity'
                if isinstance(field_value, dict) and 'similarity' in field_value:
                    return field_value['similarity']
                # Pokud je hodnota přímo číslo (float/int)
                elif isinstance(field_value, (float, int)):
                    return field_value
                # Pro ostatní případy vrátíme None
                return None
            
            # Pomocná funkce pro získání celého pole
            def get_field_data(doc, field_name):
                if field_name not in doc:
                    return None
                return doc[field_name]
            
            # 1. Preferovaná pole z VLM
            vlm_similarity = get_similarity(vlm_doc, field)
            text_similarity = get_similarity(text_doc, field)
            
            if field in vlm_preferred_fields and vlm_similarity is not None:
                field_data = get_field_data(vlm_doc, field)
            # 2. Preferovaná pole z Text
            elif field in text_preferred_fields and text_similarity is not None:
                field_data = get_field_data(text_doc, field)
            # 3. Pole, které chybí ve VLM, bereme z Text
            elif field not in vlm_doc or vlm_similarity is None:
                if field in text_doc and text_similarity is not None:
                    field_data = get_field_data(text_doc, field)
            # 4. Jinak bereme lepší skóre z obou zdrojů
            else:
                # Pokud některá podobnost je None, použijeme druhou
                if text_similarity is None and vlm_similarity is not None:
                    field_data = get_field_data(vlm_doc, field)
                elif vlm_similarity is None and text_similarity is not None:
                    field_data = get_field_data(text_doc, field)
                # Jinak bereme vyšší podobnost
                elif text_similarity is not None and vlm_similarity is not None:
                    if text_similarity >= vlm_similarity:
                        field_data = get_field_data(text_doc, field)
                    else:
                        field_data = get_field_data(vlm_doc, field)
            
            # Pokud jsme našli data pro pole, přidáme je do hybridního výsledku
            if field_data is not None:
                hybrid_data["comparison"][doc_id][field] = field_data
    
    # Výpočet celkových metrik
    total_similarity = 0
    count = 0
    
    for doc_id, doc_data in hybrid_data["comparison"].items():
        for field, field_data in doc_data.items():
            # Získání hodnoty podobnosti (buď ze slovníku, nebo přímo hodnota)
            similarity = None
            if isinstance(field_data, dict) and 'similarity' in field_data:
                similarity = field_data['similarity']
            elif isinstance(field_data, (float, int)):
                similarity = field_data
                
            if similarity is not None:
                total_similarity += similarity
                count += 1
    
    # Průměrná podobnost
    avg_similarity = total_similarity / count if count > 0 else 0
    
    hybrid_data["metrics"] = {
        "avg_similarity": avg_similarity,
        "fields_count": count,
        "docs_count": len(hybrid_data["comparison"])
    }
    
    # Uložení výsledku
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(hybrid_data, f, indent=2, ensure_ascii=False)
        print(f"Hybridní výsledky uloženy do: {output_file_path}")
        print(f"Průměrné skóre podobnosti: {avg_similarity:.4f}")
        print(f"Počet polí: {count}")
        print(f"Počet dokumentů: {len(hybrid_data['comparison'])}")
        return True
    except Exception as e:
        print(f"Chyba při ukládání výsledku: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Kombinace výsledků z text_comparison_semantic.json a vlm_comparison_semantic.json")
    parser.add_argument("--dir", type=str, required=True, help="Adresář s výsledky")
    
    args = parser.parse_args()
    
    results_dir = Path(args.dir)
    
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"Chyba: Adresář {results_dir} neexistuje nebo není adresář")
        return
    
    text_file = results_dir / "text_comparison_semantic.json"
    vlm_file = results_dir / "vlm_comparison_semantic.json"
    output_file = results_dir / "hybrid_comparison_semantic.json"
    
    if not text_file.exists():
        print(f"Chyba: Soubor {text_file} neexistuje")
        return
    
    if not vlm_file.exists():
        print(f"Chyba: Soubor {vlm_file} neexistuje")
        return
    
    success = combine_semantic_results(text_file, vlm_file, output_file)
    
    if success:
        print("Kombinace dokončena úspěšně")
    else:
        print("Kombinace selhala")

if __name__ == "__main__":
    main() 