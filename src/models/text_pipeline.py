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
            dict: Extrahovaná metadata
        """
        # Extrahujeme text z PDF pomocí naší nové metody
        full_text = self.extract_text_from_pdf(pdf_path)
        
        if not full_text:
            print(f"Nepodařilo se extrahovat text z PDF souboru {pdf_path}")
            return {}
        
        # Extrakce metadat
        metadata = {}
        for field in self.METADATA_FIELDS:
            print(f"Extrahuji pole {field}...")
            
            # Extrakce hodnoty z celého textu
            metadata[field] = self.extract_metadata_from_text_part(full_text, field)
        
        # Zkusíme vylepšit metadata přímými nálezy
        enhanced_metadata = self.enhance_metadata_with_direct_matches(metadata, full_text)
        
        return enhanced_metadata
    
    def extract_metadata_batch(self, pdf_paths, output_file=None):
        """
        Extrahuje metadata z více PDF souborů.
        
        Args:
            pdf_paths (list): Seznam cest k PDF souborům
            output_file (str, optional): Cesta k výstupnímu souboru
            
        Returns:
            dict: Extrahovaná metadata pro každý soubor
        """
        results = {}
        
        for pdf_path in tqdm(pdf_paths, desc="Extrakce metadat"):
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
            print(f"\nZpracovávám PDF soubor {pdf_path} (ID: {paper_id})...")
            
            try:
                metadata = self.extract_metadata(pdf_path)
                results[paper_id] = metadata
                
                # Průběžné ukládání výsledků
                if output_file:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Chyba při zpracování PDF souboru {pdf_path}: {e}")
                results[paper_id] = {"error": str(e)}
        
        return results
    
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
    Extrahuje metadata z PDF souborů pomocí textové pipeline.
    
    Args:
        pdf_dir (str): Cesta k adresáři s PDF soubory
        output_file (str, optional): Cesta k výstupnímu souboru
        limit (int, optional): Omezení počtu zpracovaných souborů
        force_extraction (bool): Vynutí novou extrakci i když výsledky již existují
        provider_name (str, optional): Název poskytovatele API
        model_name (str, optional): Název modelu
        api_key (str, optional): API klíč pro přístup k modelu
        
    Returns:
        dict: Extrahovaná metadata pro každý soubor
    """
    # Vždy vytváříme nové výsledky bez ohledu na existenci souborů
    print(f"Provádím novou extrakci metadat...")
    
    # Získání seznamu PDF souborů
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    # Omezení počtu souborů
    if limit is not None and limit > 0:
        pdf_files = pdf_files[:limit]
    
    print(f"Počet PDF souborů ke zpracování: {len(pdf_files)}")
    
    # Inicializace pipeline
    pipeline = TextPipeline(
        model_name=model_name,
        provider_name=provider_name,
        api_key=api_key
    )
    
    # Cesty k PDF souborům
    pdf_paths = [os.path.join(pdf_dir, f) for f in pdf_files]
    
    # Extrakce metadat
    results = pipeline.extract_metadata_batch(pdf_paths, output_file)
    
    return results 