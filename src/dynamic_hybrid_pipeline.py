#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dynamický hybrid pipeline, který vybírá nejlepší výsledky z TEXT a VLM pipeline
na základě sémantických skóre místo statických pravidel.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import copy
import sys
import subprocess
from typing import Dict, Any, Optional, Tuple

def load_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Načte JSON soubor a ošetří NaN hodnoty.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Nahradit "NaN" řetězce hodnotou null
        content = content.replace('NaN', 'null')
        
        return json.loads(content)
    except Exception as e:
        print(f"Chyba při načítání souboru {file_path}: {e}")
        return {}

def get_similarity_score(field_data: Any) -> Optional[float]:
    """
    Extrahuje skóre podobnosti z dat pole.
    
    Args:
        field_data: Data pole (může být dict s 'similarity' nebo přímo float/int)
    
    Returns:
        Skóre podobnosti nebo None pokud není k dispozici
    """
    if isinstance(field_data, dict) and 'similarity' in field_data:
        return field_data['similarity']
    elif isinstance(field_data, (float, int)) and not np.isnan(field_data):
        return field_data
    return None

def get_field_data_with_score(doc_data: Dict[str, Any], field: str) -> Tuple[Any, Optional[float]]:
    """
    Získá data pole a jeho skóre podobnosti.
    
    Args:
        doc_data: Data dokumentu
        field: Název pole
    
    Returns:
        Tuple (data_pole, skóre_podobnosti)
    """
    if field not in doc_data:
        return None, None
    
    field_data = doc_data[field]
    similarity_score = get_similarity_score(field_data)
    
    return field_data, similarity_score

def create_dynamic_hybrid_semantic_results(text_semantic_path: Path, vlm_semantic_path: Path, 
                                         output_path: Path, confidence_threshold: float = 0.1) -> bool:
    """
    Vytvoří dynamické hybridní sémantické výsledky vybíráním nejlepších skóre z TEXT a VLM.
    
    Args:
        text_semantic_path: Cesta k text_comparison_semantic.json
        vlm_semantic_path: Cesta k vlm_comparison_semantic.json
        output_path: Cesta k výstupnímu souboru
        confidence_threshold: Minimální rozdíl v skóre pro výběr lepšího výsledku
    
    Returns:
        True pokud bylo úspěšné
    """
    print(f"=== Vytváří dynamický hybrid z sémantických výsledků ===")
    print(f"TEXT semantic: {text_semantic_path}")
    print(f"VLM semantic: {vlm_semantic_path}")
    print(f"Výstup: {output_path}")
    print(f"Práh důvěry: {confidence_threshold}")
    
    # Načtení dat
    text_data = load_json_file(text_semantic_path)
    vlm_data = load_json_file(vlm_semantic_path)
    
    if not text_data or not vlm_data:
        print("Chyba: Nepodařilo se načíst vstupní data")
        return False
    
    # Inicializace hybridního výsledku
    hybrid_data = {
        "comparison": {},
        "metrics": {},
        "selection_stats": {
            "text_preferred": 0,
            "vlm_preferred": 0,
            "equal_scores": 0,
            "missing_comparisons": 0
        }
    }
    
    # Získání všech dokumentů
    text_doc_ids = set(text_data.get("comparison", {}).keys())
    vlm_doc_ids = set(vlm_data.get("comparison", {}).keys())
    all_doc_ids = text_doc_ids.union(vlm_doc_ids)
    
    print(f"Nalezeno {len(text_doc_ids)} dokumentů v TEXT")
    print(f"Nalezeno {len(vlm_doc_ids)} dokumentů v VLM")
    print(f"Celkem {len(all_doc_ids)} unikátních dokumentů")
    
    # Statistiky výběru
    selection_stats = {
        "text_wins": {},
        "vlm_wins": {},
        "equal_scores": {},
        "text_only": {},
        "vlm_only": {}
    }
    
    # Zpracování každého dokumentu
    for doc_id in all_doc_ids:
        hybrid_data["comparison"][doc_id] = {}
        
        text_doc = text_data.get("comparison", {}).get(doc_id, {})
        vlm_doc = vlm_data.get("comparison", {}).get(doc_id, {})
        
        # Získání všech polí
        all_fields = set()
        if text_doc:
            all_fields.update(text_doc.keys())
        if vlm_doc:
            all_fields.update(vlm_doc.keys())
        
        # Zpracování každého pole
        for field in all_fields:
            text_field_data, text_score = get_field_data_with_score(text_doc, field)
            vlm_field_data, vlm_score = get_field_data_with_score(vlm_doc, field)
            
            selected_data = None
            selection_reason = "missing"
            
            # Logika výběru na základě skóre
            if text_score is not None and vlm_score is not None:
                # Oba zdroje mají data
                score_diff = text_score - vlm_score
                
                if abs(score_diff) <= confidence_threshold:
                    # Skóre jsou podobná, preferujeme VLM (původní logika)
                    selected_data = vlm_field_data
                    selection_reason = "equal_scores"
                    selection_stats["equal_scores"][field] = selection_stats["equal_scores"].get(field, 0) + 1
                elif text_score > vlm_score:
                    # TEXT má výrazně lepší skóre
                    selected_data = text_field_data
                    selection_reason = "text_better"
                    selection_stats["text_wins"][field] = selection_stats["text_wins"].get(field, 0) + 1
                else:
                    # VLM má výrazně lepší skóre
                    selected_data = vlm_field_data
                    selection_reason = "vlm_better"
                    selection_stats["vlm_wins"][field] = selection_stats["vlm_wins"].get(field, 0) + 1
                    
            elif text_score is not None:
                # Pouze TEXT má data
                selected_data = text_field_data
                selection_reason = "text_only"
                selection_stats["text_only"][field] = selection_stats["text_only"].get(field, 0) + 1
                
            elif vlm_score is not None:
                # Pouze VLM má data
                selected_data = vlm_field_data
                selection_reason = "vlm_only"
                selection_stats["vlm_only"][field] = selection_stats["vlm_only"].get(field, 0) + 1
            
            # Uložení vybrané hodnoty
            if selected_data is not None:
                hybrid_data["comparison"][doc_id][field] = selected_data
                
                # Aktualizace celkových statistik
                if selection_reason == "text_better" or selection_reason == "text_only":
                    hybrid_data["selection_stats"]["text_preferred"] += 1
                elif selection_reason == "vlm_better" or selection_reason == "vlm_only":
                    hybrid_data["selection_stats"]["vlm_preferred"] += 1
                elif selection_reason == "equal_scores":
                    hybrid_data["selection_stats"]["equal_scores"] += 1
            else:
                hybrid_data["selection_stats"]["missing_comparisons"] += 1
    
    # Výpočet celkových metrik
    total_similarity = 0
    count = 0
    
    for doc_id, doc_data in hybrid_data["comparison"].items():
        for field, field_data in doc_data.items():
            similarity = get_similarity_score(field_data)
            if similarity is not None:
                total_similarity += similarity
                count += 1
    
    avg_similarity = total_similarity / count if count > 0 else 0
    
    hybrid_data["metrics"] = {
        "avg_similarity": avg_similarity,
        "fields_count": count,
        "docs_count": len(hybrid_data["comparison"]),
        "confidence_threshold": confidence_threshold
    }
    
    # Detailní statistiky výběru
    hybrid_data["detailed_selection_stats"] = selection_stats
    
    # Uložení výsledku
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(hybrid_data, f, indent=2, ensure_ascii=False)
        
        # Výpis statistik
        print(f"\n=== Výsledky dynamického hybrid ===")
        print(f"Průměrné skóre podobnosti: {avg_similarity:.4f}")
        print(f"Počet polí: {count}")
        print(f"Počet dokumentů: {len(hybrid_data['comparison'])}")
        print(f"\n=== Statistiky výběru ===")
        print(f"TEXT preferováno: {hybrid_data['selection_stats']['text_preferred']}")
        print(f"VLM preferováno: {hybrid_data['selection_stats']['vlm_preferred']}")
        print(f"Stejné skóre: {hybrid_data['selection_stats']['equal_scores']}")
        print(f"Chybějící porovnání: {hybrid_data['selection_stats']['missing_comparisons']}")
        
        print(f"\n=== Detailní statistiky podle polí ===")
        for category, stats in selection_stats.items():
            if stats:
                print(f"{category.upper()}:")
                for field, count in sorted(stats.items()):
                    print(f"  {field}: {count}")
        
        return True
        
    except Exception as e:
        print(f"Chyba při ukládání výsledku: {e}")
        return False

def create_dynamic_hybrid_base_results(text_results_path: Path, vlm_results_path: Path, 
                                     text_semantic_path: Path, vlm_semantic_path: Path,
                                     output_path: Path, confidence_threshold: float = 0.1) -> bool:
    """
    Vytvoří dynamické hybridní základní výsledky vybíráním nejlepších dat na základě sémantických skóre.
    
    Args:
        text_results_path: Cesta k text_results.json
        vlm_results_path: Cesta k vlm_results.json
        text_semantic_path: Cesta k text_comparison_semantic.json
        vlm_semantic_path: Cesta k vlm_comparison_semantic.json
        output_path: Cesta k výstupnímu souboru
        confidence_threshold: Minimální rozdíl v skóre pro výběr lepšího výsledku
    
    Returns:
        True pokud bylo úspěšné
    """
    print(f"\n=== Vytváří dynamické hybridní základní výsledky ===")
    
    # Načtení všech potřebných souborů
    text_results = load_json_file(text_results_path)
    vlm_results = load_json_file(vlm_results_path)
    text_semantic = load_json_file(text_semantic_path)
    vlm_semantic = load_json_file(vlm_semantic_path)
    
    if not all([text_results, vlm_results, text_semantic, vlm_semantic]):
        print("Chyba: Nepodařilo se načíst všechny potřebné soubory")
        return False
    
    # Inicializace hybridních výsledků
    hybrid_results = {
        "results": [],
        "timings": {},
        "token_usages": {},
        "selection_metadata": {
            "confidence_threshold": confidence_threshold,
            "selection_stats": {
                "text_preferred": 0,
                "vlm_preferred": 0,
                "equal_scores": 0
            }
        }
    }
    
    # Zpracování timings (součet obou pipeline)
    text_timings = text_results.get("timings", {})
    vlm_timings = vlm_results.get("timings", {})
    
    if isinstance(text_timings, dict) and isinstance(vlm_timings, dict):
        text_total = sum(text_timings.values()) if text_timings else 0
        vlm_total = sum(vlm_timings.values()) if vlm_timings else 0
        total_time = text_total + vlm_total
        avg_time = total_time / 2
        
        hybrid_results["timings"] = {
            "total": total_time,
            "average": avg_time,
            "text_time": text_total,
            "vlm_time": vlm_total
        }
    
    # Zpracování token_usages
    if "token_usages" in text_results:
        hybrid_results["token_usages"].update(text_results["token_usages"])
    if "token_usages" in vlm_results:
        hybrid_results["token_usages"].update(vlm_results["token_usages"])
    
    # Příprava dat podle DOI
    text_results_by_doi = {}
    vlm_results_by_doi = {}
    
    # Zpracování TEXT výsledků
    if isinstance(text_results, dict):
        if "results" in text_results and isinstance(text_results["results"], list):
            for item in text_results["results"]:
                if "doi" in item:
                    text_results_by_doi[item["doi"]] = item
        else:
            # Formát: text_results je slovník s DOI jako klíči
            for doi, item in text_results.items():
                if doi not in ["timings", "token_usages"] and isinstance(item, dict):
                    if "doi" not in item:
                        item = copy.deepcopy(item)
                        item["doi"] = doi
                    text_results_by_doi[doi] = item
    
    # Zpracování VLM výsledků
    if "results" in vlm_results:
        vlm_data = vlm_results["results"]
        if isinstance(vlm_data, dict):
            for doi, item in vlm_data.items():
                if isinstance(item, dict):
                    if "doi" not in item:
                        item = copy.deepcopy(item)
                        item["doi"] = doi
                    vlm_results_by_doi[doi] = item
        elif isinstance(vlm_data, list):
            for item in vlm_data:
                if "doi" in item:
                    vlm_results_by_doi[item["doi"]] = item
    
    # Získání sémantických skóre pro každý dokument a pole
    text_semantic_scores = {}
    vlm_semantic_scores = {}
    
    text_comparison = text_semantic.get("comparison", {})
    vlm_comparison = vlm_semantic.get("comparison", {})
    
    for doc_id in text_comparison:
        text_semantic_scores[doc_id] = {}
        for field, field_data in text_comparison[doc_id].items():
            score = get_similarity_score(field_data)
            if score is not None:
                text_semantic_scores[doc_id][field] = score
    
    for doc_id in vlm_comparison:
        vlm_semantic_scores[doc_id] = {}
        for field, field_data in vlm_comparison[doc_id].items():
            score = get_similarity_score(field_data)
            if score is not None:
                vlm_semantic_scores[doc_id][field] = score
    
    # Kombinace výsledků na základě sémantických skóre
    all_dois = set(text_results_by_doi.keys()) | set(vlm_results_by_doi.keys())
    
    print(f"Zpracovává {len(all_dois)} unikátních DOI")
    
    for doi in all_dois:
        hybrid_item = {}
        
        # Začneme s prázdným záznamem
        if doi in text_results_by_doi:
            hybrid_item = copy.deepcopy(text_results_by_doi[doi])
        elif doi in vlm_results_by_doi:
            hybrid_item = copy.deepcopy(vlm_results_by_doi[doi])
        
        # Pro každé pole rozhodneme, kterou hodnotu použít
        text_item = text_results_by_doi.get(doi, {})
        vlm_item = vlm_results_by_doi.get(doi, {})
        
        # Získání všech polí
        all_fields = set()
        if text_item:
            all_fields.update(text_item.keys())
        if vlm_item:
            all_fields.update(vlm_item.keys())
        
        for field in all_fields:
            if field in ["timings", "token_usages"]:
                continue  # Přeskočíme metadata
            
            text_has_field = field in text_item and text_item[field]
            vlm_has_field = field in vlm_item and vlm_item[field]
            
            if not text_has_field and not vlm_has_field:
                continue
            
            # Získání sémantických skóre
            text_score = text_semantic_scores.get(doi, {}).get(field)
            vlm_score = vlm_semantic_scores.get(doi, {}).get(field)
            
            selected_from = "none"
            
            if text_score is not None and vlm_score is not None:
                # Oba mají sémantické skóre
                score_diff = text_score - vlm_score
                
                if abs(score_diff) <= confidence_threshold:
                    # Podobné skóre, preferujeme VLM (původní logika)
                    if vlm_has_field:
                        hybrid_item[field] = vlm_item[field]
                        selected_from = "vlm"
                    elif text_has_field:
                        hybrid_item[field] = text_item[field]
                        selected_from = "text"
                    hybrid_results["selection_metadata"]["selection_stats"]["equal_scores"] += 1
                elif text_score > vlm_score:
                    # TEXT má lepší skóre
                    if text_has_field:
                        hybrid_item[field] = text_item[field]
                        selected_from = "text"
                    hybrid_results["selection_metadata"]["selection_stats"]["text_preferred"] += 1
                else:
                    # VLM má lepší skóre
                    if vlm_has_field:
                        hybrid_item[field] = vlm_item[field]
                        selected_from = "vlm"
                    hybrid_results["selection_metadata"]["selection_stats"]["vlm_preferred"] += 1
                    
            elif text_score is not None and text_has_field:
                # Pouze TEXT má sémantické skóre
                hybrid_item[field] = text_item[field]
                selected_from = "text"
                hybrid_results["selection_metadata"]["selection_stats"]["text_preferred"] += 1
                
            elif vlm_score is not None and vlm_has_field:
                # Pouze VLM má sémantické skóre
                hybrid_item[field] = vlm_item[field]
                selected_from = "vlm"
                hybrid_results["selection_metadata"]["selection_stats"]["vlm_preferred"] += 1
                
            elif vlm_has_field:
                # Žádné sémantické skóre, preferujeme VLM
                hybrid_item[field] = vlm_item[field]
                selected_from = "vlm"
                
            elif text_has_field:
                # Žádné sémantické skóre, fallback na TEXT
                hybrid_item[field] = text_item[field]
                selected_from = "text"
        
        # Přidání záznamu do výsledků pokud má nějaká data
        if any(field != "doi" and value for field, value in hybrid_item.items() if field not in ["timings", "token_usages"]):
            hybrid_results["results"].append(hybrid_item)
    
    print(f"Vytvořeno {len(hybrid_results['results'])} dynamických hybridních záznamů")
    
    # Uložení výsledků
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(hybrid_results, f, ensure_ascii=False, indent=2)
        
        stats = hybrid_results["selection_metadata"]["selection_stats"]
        print(f"\n=== Statistiky výběru pro základní výsledky ===")
        print(f"TEXT preferováno: {stats['text_preferred']}")
        print(f"VLM preferováno: {stats['vlm_preferred']}")
        print(f"Stejné skóre: {stats['equal_scores']}")
        
        return True
        
    except Exception as e:
        print(f"Chyba při ukládání základních výsledků: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Dynamický hybrid pipeline vybírající nejlepší výsledky na základě sémantických skóre")
    parser.add_argument("--dir", type=str, required=True, help="Adresář s výsledky")
    parser.add_argument("--confidence-threshold", type=float, default=0.1, help="Práh důvěry pro výběr lepšího výsledku (default: 0.1)")
    parser.add_argument("--semantic-only", action="store_true", help="Pouze vytvoří dynamické sémantické výsledky")
    parser.add_argument("--base-only", action="store_true", help="Pouze vytvoří dynamické základní výsledky")
    
    args = parser.parse_args()
    
    results_dir = Path(args.dir)
    
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"Chyba: Adresář {results_dir} neexistuje nebo není adresář")
        return
    
    # Kontrola požadovaných souborů
    text_semantic_path = results_dir / "text_comparison_semantic.json"
    vlm_semantic_path = results_dir / "vlm_comparison_semantic.json"
    text_results_path = results_dir / "text_results.json"
    vlm_results_path = results_dir / "vlm_results.json"
    
    success = True
    
    # Vytvoření dynamických sémantických výsledků
    if not args.base_only:
        if text_semantic_path.exists() and vlm_semantic_path.exists():
            output_semantic_path = results_dir / "dynamic_hybrid_comparison_semantic.json"
            success &= create_dynamic_hybrid_semantic_results(
                text_semantic_path, vlm_semantic_path, output_semantic_path, args.confidence_threshold
            )
        else:
            print(f"Chybí sémantické soubory: {text_semantic_path}, {vlm_semantic_path}")
            success = False
    
    # Vytvoření dynamických základních výsledků
    if not args.semantic_only:
        if all(p.exists() for p in [text_results_path, vlm_results_path, text_semantic_path, vlm_semantic_path]):
            output_base_path = results_dir / "dynamic_hybrid_results.json"
            success &= create_dynamic_hybrid_base_results(
                text_results_path, vlm_results_path, text_semantic_path, vlm_semantic_path,
                output_base_path, args.confidence_threshold
            )
        else:
            missing = [str(p) for p in [text_results_path, vlm_results_path, text_semantic_path, vlm_semantic_path] if not p.exists()]
            print(f"Chybí soubory: {', '.join(missing)}")
            success = False
    
    if success:
        print(f"\n=== Dynamický hybrid dokončen úspěšně ===")
        print(f"Výsledky uloženy v {results_dir}")
    else:
        print(f"\n=== Dynamický hybrid selhal ===")

if __name__ == "__main__":
    main() 