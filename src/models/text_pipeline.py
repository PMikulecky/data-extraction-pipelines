#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro implementaci textové pipeline pro extrakci metadat z PDF souborů.
"""

import os
import re
import json
import PyPDF2
import time
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.models.providers.factory import ModelProviderFactory
from src.models.config.model_config import get_config
from src.config.runtime_config import get_run_results_dir

from dotenv import load_dotenv


class TextPipeline:
    """
    Třída pro implementaci textové pipeline pro extrakci metadat z PDF souborů.
    Využívá pouze textovou reprezentaci PDF dokumentu bez embeddings nebo vizuálních modelů.
    """
    
    # Definice metadat, která budou extrahována
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
        'publisher'        # Vydavatel
        # 'references'     # Seznam referencí - vyřazeno z extrakce
    ]
    
    # Šablony dotazů pro extrakci metadat
    QUERY_TEMPLATES = {
        'title': "Jaký je název této akademické práce? Vrať pouze název bez jakéhokoliv dalšího textu.",
        'authors': "Kdo jsou autoři této akademické práce? Uveď všechny autory ve formátu 'Jméno Příjmení'. Vrať pouze seznam autorů bez jakéhokoliv dalšího textu.",
        'abstract': "Jaký je abstrakt této akademické práce? Vrať pouze abstrakt bez jakéhokoliv dalšího textu.",
        'keywords': "Jaká jsou klíčová slova této akademické práce? Vrať pouze klíčová slova jako seznam oddělený čárkami bez jakéhokoliv dalšího textu.",
        'doi': "Jaké je DOI této akademické práce? Vrať pouze DOI bez jakéhokoliv dalšího textu.",
        'year': "V jakém roce byla tato akademická práce publikována? Vrať pouze rok bez jakéhokoliv dalšího textu.",
        'journal': "V jakém časopise nebo sborníku konference byla tato akademická práce publikována? Vrať pouze název časopisu nebo konference bez jakéhokoliv dalšího textu.",
        'volume': "Jaké je číslo ročníku časopisu, ve kterém byla tato akademická práce publikována? Vrať pouze číslo ročníku bez jakéhokoliv dalšího textu.",
        'issue': "Jaké je číslo vydání časopisu, ve kterém byla tato akademická práce publikována? Vrať pouze číslo vydání bez jakéhokoliv dalšího textu.",
        'pages': "Jaké jsou čísla stránek této akademické práce v časopise nebo sborníku? Vrať pouze čísla stránek (např. '123-145') bez jakéhokoliv dalšího textu.",
        'publisher': "Kdo je vydavatelem této akademické práce? Vrať pouze název vydavatele bez jakéhokoliv dalšího textu.",
        'references': "Uveď seznam referencí citovaných v této akademické práci. Vrať pouze seznam referencí bez jakéhokoliv dalšího textu."
    }
    
    # Mapování mezi metadaty a částmi dokumentu - již není používáno, ale ponecháno pro kompatibilitu
    METADATA_TO_DOCUMENT_PART = {
        'title': 'title_page',
        'authors': 'title_page',
        'abstract': 'abstract',
        'keywords': 'abstract',
        'doi': 'title_page',
        'year': 'title_page',
        'journal': 'title_page',
        'volume': 'title_page',
        'issue': 'title_page',
        'pages': 'title_page',
        'publisher': 'title_page',
        'references': 'references'
    }
    
    def __init__(self, model_name=None, provider_name=None, api_key=None):
        """
        Inicializace textové pipeline.
        
        Args:
            model_name (str, optional): Název modelu
            provider_name (str, optional): Název poskytovatele API
            api_key (str, optional): API klíč pro přístup k modelu
        """
        # Načtení konfigurace
        config = get_config()
        text_config = config.get_text_config()
        
        # Použití parametrů nebo konfigurace
        self.provider_name = provider_name or text_config["provider"]
        self.model_name = model_name or text_config["model"]
        self.api_key = api_key
        
        # Inicializace poskytovatele modelu
        self.text_provider = ModelProviderFactory.create_text_provider(
            provider_name=self.provider_name,
            model_name=self.model_name,
            api_key=self.api_key
        )
        
        print(f"Inicializován textový model: {self.model_name} od poskytovatele: {self.provider_name}")
    
    def query_text_model(self, text, query) -> tuple[str, dict]:
        """
        Dotaz na textový model.
        
        Args:
            text (str): Kontext (část textu dokumentu)
            query (str): Dotaz pro model
            
        Returns:
            tuple: Odpověď modelu (str) a slovník s tokeny (dict)
        """
        prompt = f"""Kontext (část akademické práce):
{text}

Dotaz: {query}

Odpověď:"""
        
        # Použití poskytovatele pro generování odpovědi
        # Předpokládáme, že provider nyní vrací (text, token_usage)
        text_result, token_usage = self.text_provider.generate_text(prompt)
        return text_result, token_usage
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extrahuje text z PDF souboru použitím PyPDF2.
        
        Args:
            pdf_path (str): Cesta k PDF souboru
            
        Returns:
            str: Extrahovaný text
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                
                return text
        except Exception as e:
            print(f"Chyba při extrakci textu z PDF souboru {pdf_path}: {e}")
            return ""
    
    def extract_metadata_from_text_part(self, text, field) -> tuple[str, dict]:
        """
        Extrahuje metadata z určité části textu.
        
        Args:
            text (str): Text dokumentu
            field (str): Pole metadat k extrakci
            
        Returns:
            tuple: Extrahovaná hodnota (str) a slovník s tokeny (dict)
        """
        if not text:
            return "", {"input_tokens": 0, "output_tokens": 0}
        
        query = self.QUERY_TEMPLATES.get(field, f"Co je {field} této akademické práce?")
        
        # Limitování délky textu (může ovlivnit počet input tokenů)
        max_text_length = 6000
        original_len = len(text)
        if original_len > max_text_length:
            if field in ['references']: text = text[-max_text_length:]
            elif field in ['abstract', 'keywords']: text = text[:max_text_length]
            else: text = text[:max_text_length]
            print(f"  (Text pro pole '{field}' zkrácen z {original_len} na {len(text)} znaků)")
        
        token_usage = {"input_tokens": 0, "output_tokens": 0} # Default
        try:
            # Dotaz na model, získáme i tokeny
            result, token_usage = self.query_text_model(text, query)
            return result.strip() if result else "", token_usage
        except Exception as e:
            print(f"Chyba při extrakci pole {field} z textu: {e}")
            return "", token_usage # Vrátit default i při chybě
    
    def extract_metadata(self, pdf_path) -> tuple[dict, float | None, dict]:
        """
        Extrahuje metadata z PDF souboru.
        
        Args:
            pdf_path (str): Cesta k PDF souboru
            
        Returns:
            tuple(dict, float | None, dict): Extrahovaná metadata, doba trvání a celkové token usage
        """
        start_time = time.perf_counter()
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Zpracovávám PDF soubor {pdf_path} (Text)...")
        total_token_usage_doc = {"input_tokens": 0, "output_tokens": 0}

        try:
            full_text = self.extract_text_from_pdf(pdf_path)
            if not full_text:
                print(f"Nepodařilo se extrahovat text z PDF souboru {pdf_path}")
                duration = time.perf_counter() - start_time
                return {}, duration, total_token_usage_doc

            metadata = {}
            for field in self.METADATA_FIELDS:
                print(f"Extrahuji pole {field}...")
                try:
                    # Extrakce hodnoty i tokenů
                    field_value, field_token_usage = self.extract_metadata_from_text_part(full_text, field)
                    metadata[field] = field_value
                    # Agregace tokenů
                    total_token_usage_doc["input_tokens"] += field_token_usage.get("input_tokens", 0)
                    total_token_usage_doc["output_tokens"] += field_token_usage.get("output_tokens", 0)
                except Exception as e:
                    print(f"Chyba při extrakci pole {field} pro {pdf_path}: {e}")
                    metadata[field] = ""

            enhanced_metadata = self.enhance_metadata_with_direct_matches(metadata, full_text)
            duration = time.perf_counter() - start_time
            print(f"Extrakce pro {paper_id} (Text) trvala {duration:.2f} sekund.")
            print(f"Tokeny pro {paper_id} (Text): Vstup={total_token_usage_doc['input_tokens']}, Výstup={total_token_usage_doc['output_tokens']}")
            return enhanced_metadata, duration, total_token_usage_doc # Vrátit i tokeny
        except Exception as e:
             print(f"Obecná chyba při zpracování PDF {pdf_path} v extract_metadata (Text): {e}")
             duration = time.perf_counter() - start_time
             return {}, duration, total_token_usage_doc # Vrátit i tokeny
    
    def extract_metadata_batch(self, pdf_paths, output_file=None) -> tuple[dict, dict, dict]:
        """
        Extrahuje metadata z více PDF souborů.
        
        Args:
            pdf_paths (list): Seznam cest k PDF souborům
            output_file (str, optional): Cesta k výstupnímu souboru pro průběžné ukládání
            
        Returns:
            tuple(dict, dict, dict): Slovník metadat, slovník časů, slovník tokenů
        """
        results = {}
        extraction_times = {} 
        token_usages = {} # Nový slovník pro tokeny
        
        pdf_files_to_process = pdf_paths # Předejeme celý seznam, pokud není output_file
        
        # Pokud máme output_file, můžeme načíst již zpracované
        if output_file and Path(output_file).exists():
            try:
                 with open(output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                 print(f"Nalezeno {len(existing_results)} existujících výsledků v {output_file}")
                 # Načteme i časy a tokeny, pokud existují
                 timing_file = Path(output_file).parent / f"{Path(output_file).stem.replace('_results', '')}_timing.json"
                 token_file = Path(output_file).parent / f"{Path(output_file).stem.replace('_results', '')}_tokens.json"
                 if timing_file.exists():
                     with open(timing_file, 'r', encoding='utf-8') as f:
                         extraction_times = json.load(f)
                 if token_file.exists():
                     with open(token_file, 'r', encoding='utf-8') as f:
                         token_usages = json.load(f)
                 
                 # Odfiltrujeme již zpracované soubory
                 processed_ids = set(existing_results.keys())
                 pdf_files_to_process = [p for p in pdf_paths if Path(p).stem not in processed_ids]
                 results.update(existing_results) # Přidáme existující
                 print(f"Zbývá zpracovat {len(pdf_files_to_process)} souborů.")
            except Exception as e:
                 print(f"Chyba při načítání existujících výsledků z {output_file}: {e}. Zpracuji vše znovu.")
                 pdf_files_to_process = pdf_paths
                 results = {}
                 extraction_times = {}
                 token_usages = {}

        if not pdf_files_to_process:
             print("Žádné nové soubory ke zpracování.")
             return results, extraction_times, token_usages

        # Zpracování zbývajících souborů
        for pdf_path in tqdm(pdf_files_to_process, desc="Extrakce metadat (Text)"):
            pdf_id = Path(pdf_path).stem
            duration = None
            token_usage = {"input_tokens": 0, "output_tokens": 0}
            try:
                metadata_dict, duration, token_usage = self.extract_metadata(pdf_path)
                results[pdf_id] = metadata_dict
            except Exception as e:
                print(f"Chyba při zpracování souboru {pdf_path}: {e}")
                results[pdf_id] = None
            finally:
                extraction_times[pdf_id] = duration
                token_usages[pdf_id] = token_usage # Uložit tokeny
                
                # Průběžné ukládání (pokud je zadán output_file)
                if output_file:
                    # Uložíme metadata
                    try:
                        save_results = {k: (v if v is not None else {"error": "Extraction failed"}) for k, v in results.items()}
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(save_results, f, ensure_ascii=False, indent=2)
                    except Exception as save_e:
                         print(f"Chyba při průběžném ukládání výsledků: {save_e}")
                    # Uložíme časy
                    try:
                        timing_file = Path(output_file).parent / f"{Path(output_file).stem.replace('_results', '')}_timing.json"
                        save_timings = {k: (v if v is not None else -1.0) for k, v in extraction_times.items()}
                        with open(timing_file, 'w', encoding='utf-8') as f:
                             json.dump(save_timings, f, ensure_ascii=False, indent=2)
                    except Exception as save_t_e:
                         print(f"Chyba při průběžném ukládání časů: {save_t_e}")
                    # Uložíme tokeny
                    try:
                        token_file = Path(output_file).parent / f"{Path(output_file).stem.replace('_results', '')}_tokens.json"
                        save_tokens = {k: (v if v is not None else {"input_tokens": 0, "output_tokens": 0}) for k, v in token_usages.items()}
                        with open(token_file, 'w', encoding='utf-8') as f:
                             json.dump(save_tokens, f, ensure_ascii=False, indent=2)
                    except Exception as save_tok_e:
                         print(f"Chyba při průběžném ukládání tokenů: {save_tok_e}")
                        
        return results, extraction_times, token_usages # Vrátit i tokeny
    
    def extract_direct_matches(self, text):
        """
        Extrahuje metadata pomocí regulárních výrazů přímo z textu.
        
        Args:
            text (str): Text dokumentu
            
        Returns:
            dict: Extrahovaná metadata pomocí přímého vyhledávání
        """
        metadata = {}
        
        # DOI pattern
        doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Z0-9]+'
        doi_match = re.search(doi_pattern, text, re.IGNORECASE)
        if doi_match:
            metadata['doi'] = doi_match.group(0)
        
        # Rok publikace
        year_pattern = r'(?:published|copyright|©|\(c\))(?:\s+in)?\s+(\d{4})'
        year_match = re.search(year_pattern, text, re.IGNORECASE)
        if year_match:
            metadata['year'] = year_match.group(1)
        
        # Stránky
        pages_pattern = r'pages\s+(\d+[-–]\d+)'
        pages_match = re.search(pages_pattern, text, re.IGNORECASE)
        if pages_match:
            metadata['pages'] = pages_match.group(1)
        
        return metadata
    
    def enhance_metadata_with_direct_matches(self, metadata, text):
        """
        Vylepší extrahovaná metadata přímými nálezy z textu.
        
        Args:
            metadata (dict): Původní metadata
            text (str): Text dokumentu
            
        Returns:
            dict: Vylepšená metadata
        """
        # Extrakce přímých nálezů
        direct_matches = self.extract_direct_matches(text)
        
        # Kombinace metadat (přímé nálezy mají přednost)
        enhanced_metadata = metadata.copy()
        for key, value in direct_matches.items():
            if value and (key not in enhanced_metadata or not enhanced_metadata[key]):
                enhanced_metadata[key] = value
        
        return enhanced_metadata


def extract_metadata_from_pdfs(pdf_dir, output_file=None, limit=None, force_extraction=False, provider_name=None, model_name=None, api_key=None) -> tuple[dict, dict, dict]:
    """
    Hlavní funkce pro spuštění textové pipeline pro extrakci metadat.
    Vrací (results, timings, token_usages).
    """
    results = None
    timings = {}
    token_usages = {} # Inicializace
    output_file_path = None
    timing_output_file = None
    token_output_file = None # Cesta pro tokeny

    # Nastavení cest, pokud je zadán output_file
    if output_file:
        output_file_path = Path(output_file)
        try:
            timing_output_file = output_file_path.parent / f"{output_file_path.stem.replace('_results', '')}_timing.json"
            token_output_file = output_file_path.parent / f"{output_file_path.stem.replace('_results', '')}_tokens.json"
        except Exception as e:
            print(f"Chyba při vytváření cest k souborům z {output_file}: {e}")
            # Nemůžeme pokračovat bez platných cest, pokud byl zadán output_file
            return {}, {}, {}
    else: 
        # Pokud není zadán output_file, použijeme runtime config
        run_dir = get_run_results_dir()
        output_file_path = run_dir / "text_results.json"
        timing_output_file = run_dir / "text_timing.json"
        token_output_file = run_dir / "text_tokens.json"


    # Zkontroluje, zda výsledky už existují
    if output_file_path.exists() and not force_extraction:
        print(f"Výsledky již existují v {output_file_path}. Přeskakuji extrakci.")
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            # Pokusíme se načíst i časy a tokeny
            if timing_output_file and timing_output_file.exists():
                 with open(timing_output_file, 'r', encoding='utf-8') as f: timings = json.load(f)
            if token_output_file and token_output_file.exists():
                 with open(token_output_file, 'r', encoding='utf-8') as f: token_usages = json.load(f)
        except Exception as e:
             print(f"Chyba při načítání existujících souborů: {e}. Vynucuji novou extrakci.")
             results = None; timings = {}; token_usages = {}

        if results is not None:
            return results, timings, token_usages

    # --- Sekce pro spuštění extrakce ---
    print("Spouštím extrakci (Text)...")
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if limit: pdf_files = pdf_files[:limit]
    if not pdf_files: print("Nebyly nalezeny žádné PDF soubory."); return {}, {}, {}

    pipeline = TextPipeline(provider_name=provider_name, model_name=model_name, api_key=api_key)
    try:
        # Zavoláme batch s předáním output_file_path pro průběžné ukládání
        results, timings, token_usages = pipeline.extract_metadata_batch(pdf_files, output_file=output_file_path)
    except Exception as batch_e:
        print(f"Chyba během dávkové extrakce (Text): {batch_e}")
        return {}, {}, {}

    # Finální uložení není potřeba, protože batch ukládá průběžně
    print(f"Extrakce (Text) dokončena. Výsledky jsou v {output_file_path}")
    
    return results, timings, token_usages # Vrátí metadata, časy i tokeny

# ... (if __name__ == '__main__': block) ... 