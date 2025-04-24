#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro implementaci textové pipeline pro extrakci metadat z PDF souborů.
"""

import os
import re
import json
import PyPDF2  # Přidán import PyPDF2
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Optional
import time # Přidat na začátek souboru

from models.config.model_config import get_config
from models.providers.factory import ModelProviderFactory


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
        'publisher',       # Vydavatel
        'references'       # Seznam referencí
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
    
    def query_text_model(self, text, query):
        """
        Dotaz na textový model.
        
        Args:
            text (str): Kontext (část textu dokumentu)
            query (str): Dotaz pro model
            
        Returns:
            str: Odpověď modelu
        """
        # Sestavení promptu pro model
        prompt = f"""Kontext (část akademické práce):
{text}

Dotaz: {query}

Odpověď:"""
        
        # Použití poskytovatele pro generování odpovědi
        return self.text_provider.generate_text(prompt)
    
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
    
    def extract_metadata_from_text_part(self, text, field):
        """
        Extrahuje metadata z určité části textu.
        
        Args:
            text (str): Text dokumentu
            field (str): Pole metadat k extrakci
            
        Returns:
            str: Extrahovaná hodnota
        """
        if not text:
            return ""
        
        # Získání šablony dotazu pro dané pole
        query = self.QUERY_TEMPLATES.get(field, f"Co je {field} této akademické práce?")
        
        # Limitování délky textu (pro omezení počtu tokenů)
        max_text_length = 6000  # přibližně 1500 tokenů
        if len(text) > max_text_length:
            # Pro různá pole použijeme různé části textu
            if field in ['references']:
                # Pro reference použijeme konec textu
                text = text[-max_text_length:]
            elif field in ['abstract', 'keywords']:
                # Pro abstrakt a klíčová slova použijeme začátek textu
                text = text[:max_text_length]
            else:
                # Pro ostatní metadata použijeme začátek textu
                text = text[:max_text_length]
        
        try:
            # Dotaz na model
            result = self.query_text_model(text, query)
            return result.strip() if result else ""
        except Exception as e:
            print(f"Chyba při extrakci pole {field} z textu: {e}")
            return ""
    
    def extract_metadata(self, pdf_path):
        """
        Extrahuje metadata z PDF souboru.
        
        Args:
            pdf_path (str): Cesta k PDF souboru
            
        Returns:
            tuple(dict, float | None): Extrahovaná metadata a doba trvání extrakce v sekundách (nebo None při chybě)
        """
        start_time = time.perf_counter() # Měření času - START
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Zpracovávám PDF soubor {pdf_path} (Text)...")

        try:
            # Extrahujeme text z PDF pomocí naší nové metody
            full_text = self.extract_text_from_pdf(pdf_path)

            if not full_text:
                print(f"Nepodařilo se extrahovat text z PDF souboru {pdf_path}")
                duration = time.perf_counter() - start_time # Měření času - END (i při chybě)
                return {}, duration # Vrátit délku trvání i při chybě

            # Extrakce metadat
            metadata = {}
            for field in self.METADATA_FIELDS:
                print(f"Extrahuji pole {field}...")
                try:
                    # Extrakce hodnoty z celého textu
                    metadata[field] = self.extract_metadata_from_text_part(full_text, field)
                except Exception as e:
                    print(f"Chyba při extrakci pole {field} pro {pdf_path}: {e}")
                    metadata[field] = "" # Nebo jiná chybová hodnota

            # Zkusíme vylepšit metadata přímými nálezy
            enhanced_metadata = self.enhance_metadata_with_direct_matches(metadata, full_text)

            duration = time.perf_counter() - start_time # Měření času - END
            print(f"Extrakce pro {paper_id} (Text) trvala {duration:.2f} sekund.")
            return enhanced_metadata, duration
        except Exception as e:
             # Zachycení obecné chyby během zpracování
             print(f"Obecná chyba při zpracování PDF {pdf_path} v extract_metadata (Text): {e}")
             duration = time.perf_counter() - start_time
             return {}, duration # Vrátit délku trvání i při chybě
    
    def extract_metadata_batch(self, pdf_paths, output_file=None):
        """
        Extrahuje metadata z více PDF souborů.
        
        Args:
            pdf_paths (list): Seznam cest k PDF souborům
            output_file (str, optional): Cesta k výstupnímu souboru pro průběžné ukládání
            
        Returns:
            tuple(dict, dict): Slovník s extrahovanými metadaty a slovník s časy extrakce
        """
        results = {}
        extraction_times = {} # Nový slovník pro časy

        for pdf_path in tqdm(pdf_paths, desc="Extrakce metadat (Text)"):
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
            # print(f"\nZpracovávám PDF soubor {pdf_path} (ID: {paper_id})...") # Odstraněno, tqdm stačí

            try:
                 # Volání upravené metody extract_metadata
                metadata, duration = self.extract_metadata(pdf_path)
                results[paper_id] = metadata
                # Uložení času (i pokud je None nebo při chybě)
                extraction_times[paper_id] = duration

                # Průběžné ukládání výsledků (pouze metadata)
                if output_file:
                    # Ošetření None hodnot před uložením do JSON
                    save_data = {k: (v if v is not None else {"error": "Extraction failed"}) for k, v in results.items()}
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                 # Tato chyba by neměla nastat, pokud extract_metadata správně vrací duration
                print(f"Neočekávaná chyba při zpracování PDF souboru {pdf_path} v extract_metadata_batch (Text): {e}")
                results[paper_id] = {"error": str(e)}
                extraction_times[paper_id] = None # Explicitně None pro neočekávanou chybu

        return results, extraction_times
    
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


def extract_metadata_from_pdfs(pdf_dir, output_file=None, limit=None, force_extraction=False, provider_name=None, model_name=None, api_key=None):
    """
    Hlavní funkce pro spuštění textové pipeline pro extrakci metadat.
    """
    # Zkontroluje, zda výsledky už existují
    if output_file and os.path.exists(output_file) and not force_extraction:
        print(f"Výsledky již existují v {output_file}. Přeskakuji extrakci.")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except json.JSONDecodeError:
             print(f"VAROVÁNÍ: Soubor {output_file} je poškozený. Vynucuji novou extrakci.")
             results = None
             force_extraction = True # Vynutit novou extrakci

        if not force_extraction:
            # Pokusíme se načíst i časy (pokud existují)
            timing_output_file = Path(output_file).parent / f"{Path(output_file).stem.replace('_results', '')}_timing.json"
            if timing_output_file.exists():
                print(f"Načítám existující časy z {timing_output_file}")
            return results # Vrátí pouze metadata

    # Získání seznamu PDF souborů
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if limit:
        pdf_files = pdf_files[:limit]

    if not pdf_files:
        print("Nebyly nalezeny žádné PDF soubory pro zpracování.")
        return {}

    # Inicializace pipeline
    pipeline = TextPipeline(provider_name=provider_name, model_name=model_name, api_key=api_key)

    # Extrakce metadat
    results, timings = pipeline.extract_metadata_batch(pdf_files, output_file) # Získání metadat i časů

    # Uložení výsledků (metadata)
    if output_file:
        # Ošetření None hodnot před uložením do JSON
        save_results = {k: (v if v is not None else {"error": "Extraction failed"}) for k, v in results.items()}
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        print(f"Výsledky extrakce (metadata) uloženy do {output_file}")

    # Uložení časů do samostatného souboru
    if output_file: # Uložíme časy jen pokud ukládáme i výsledky
        # Ošetření None hodnot pro časy
        save_timings = {k: (v if v is not None else -1.0) for k, v in timings.items()} # -1 jako indikátor chyby
        timing_output_file = Path(output_file).parent / f"{Path(output_file).stem.replace('_results', '')}_timing.json"
        with open(timing_output_file, 'w', encoding='utf-8') as f:
            json.dump(save_timings, f, ensure_ascii=False, indent=2)
        print(f"Výsledky extrakce (časy) uloženy do {timing_output_file}")

    return results # Vrátí pouze metadata 