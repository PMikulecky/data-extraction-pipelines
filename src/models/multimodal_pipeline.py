#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro implementaci Multimodální pipeline pro extrakci metadat z PDF souborů.
Kombinuje textový i vizuální vstup pro zlepšení přesnosti extrakce.
"""

import os
import json
import base64
from io import BytesIO
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import time
import PyPDF2
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("VAROVÁNÍ: Knihovna pdf2image není k dispozici. Pokusím se použít alternativní metodu.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    if not PDF2IMAGE_AVAILABLE:
        print("VAROVÁNÍ: Ani pdf2image ani PyMuPDF nejsou k dispozici.")

from dotenv import load_dotenv

# Import pro poskytovatele modelů
from src.models.providers.factory import ModelProviderFactory
from src.models.config.model_config import get_config
from src.config.runtime_config import get_run_results_dir

# Import lokálního modulu pro analýzu PDF
try:
    from src.utils.pdf_analyzer import PDFAnalyzer
except ImportError as e:
    print(f"Nepodařilo se importovat PDFAnalyzer: {e}")
    raise

# Načtení proměnných prostředí
load_dotenv()

# Definice cest
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
RESULTS_DIR = BASE_DIR / "results"

# Vytvoření adresáře pro výsledky, pokud neexistuje
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class MultimodalPipeline:
    """
    Třída pro implementaci Multimodální pipeline pro extrakci metadat z PDF souborů.
    Kombinuje textový a vizuální vstup pro zlepšení přesnosti extrakce.
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
        'title': "What is the title of this academic paper? Extract it from both the image and the provided text. Return only the title without any additional text.",
        'authors': "Who are the authors of this academic paper? Extract them from both the image and the provided text. List all authors in the format 'First Name Last Name'. Return only the list of authors without any additional text.",
        'abstract': "What is the abstract of this academic paper? Extract it from both the image and the provided text. Return only the abstract without any additional text.",
        'keywords': "What are the keywords of this academic paper? Extract them from both the image and the provided text. Return only the keywords as a comma-separated list without any additional text.",
        'doi': "What is the DOI of this academic paper? Extract it from both the image and the provided text. Return only the DOI without any additional text.",
        'year': "In what year was this academic paper published? Extract it from both the image and the provided text. Return only the year without any additional text.",
        'journal': "In which journal or conference proceedings was this academic paper published? Extract it from both the image and the provided text. Return only the name of the journal or conference without any additional text.",
        'volume': "What is the volume number of the journal in which this academic paper was published? Extract it from both the image and the provided text. Return only the volume number without any additional text.",
        'issue': "What is the issue number of the journal in which this academic paper was published? Extract it from both the image and the provided text. Return only the issue number without any additional text.",
        'pages': "What are the page numbers of this academic paper in the journal or proceedings? Extract them from both the image and the provided text. Return only the page numbers (e.g., '123-145') without any additional text.",
        'publisher': "Who is the publisher of this academic paper? Extract it from both the image and the provided text. Return only the name of the publisher without any additional text.",
        'references': "List the first 5 references cited in this academic paper. Extract them from both the image and the provided text. Return only the list of references without any additional text."
    }
    
    def __init__(self, model_name=None, provider_name=None, api_key=None):
        """
        Inicializace Multimodální pipeline.
        
        Args:
            model_name (str, optional): Název modelu
            provider_name (str, optional): Název poskytovatele API
            api_key (str, optional): API klíč pro přístup k modelu
        """
        # Načtení konfigurace
        config = get_config()
        multimodal_config = config.get_multimodal_config()
        
        # Použití parametrů nebo konfigurace
        self.provider_name = provider_name or multimodal_config["provider"]
        self.model_name = model_name or multimodal_config["model"]
        self.api_key = api_key
        
        # Inicializace poskytovatele modelu
        self.multimodal_provider = ModelProviderFactory.create_multimodal_provider(
            provider_name=self.provider_name,
            model_name=self.model_name,
            api_key=self.api_key
        )
        
        print(f"Inicializován multimodální model: {self.model_name} od poskytovatele: {self.provider_name}")
    
    def query_multimodal(self, image, text, query) -> tuple[str, dict]:
        """
        Dotaz na multimodální model s obrázkem a textem.
        
        Args:
            image: PIL.Image objekt
            text (str): Extrahovaný text z oblasti
            query (str): Dotaz pro model
            
        Returns:
            tuple: Odpověď modelu (str) a slovník s tokeny (dict)
        """
        # Použití poskytovatele pro generování textu z obrázku a textu
        text_result, token_usage = self.multimodal_provider.generate_text_from_image_and_text(image, text, query)
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
                all_text = []
                
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        all_text.append((i, page_text))
                
                return all_text
        except Exception as e:
            print(f"Chyba při extrakci textu z PDF souboru {pdf_path}: {e}")
            return []
    
    def extract_metadata_from_combined_sources(self, images, page_texts, field) -> tuple[str, dict]:
        """
        Extrahuje metadata z kombinace obrázků a textu.
        
        Args:
            images (list): Seznam PIL.Image objektů
            page_texts (list): Seznam textů stránek ve formátu [(stránka, text), ...]
            field (str): Pole metadat k extrakci
            
        Returns:
            tuple: Extrahovaná hodnota (str) a agregovaný slovník tokenů (dict)
        """
        # Výběr relevantních obrázků podle typu pole
        relevant_images = []
        
        if field in ['title', 'authors', 'doi', 'year', 'journal', 'publisher']:
            # Pro tyto pole použijeme pouze první stránku (titulní)
            if images and len(images) > 0:
                relevant_images = [(0, images[0])]
        
        elif field == 'abstract':
            # Pro abstrakt použijeme první 2 stránky
            relevant_images = [(i, img) for i, img in enumerate(images[:min(2, len(images))])]
        
        elif field in ['volume', 'issue', 'pages']:
            # Pro tyto pole použijeme první stránku
            if images and len(images) > 0:
                relevant_images = [(0, images[0])]
        
        elif field == 'keywords':
            # Pro klíčová slova použijeme první 2 stránky
            relevant_images = [(i, img) for i, img in enumerate(images[:min(2, len(images))])]
        
        elif field == 'references':
            # Pro reference použijeme poslední stránky
            relevant_images = [(len(images) - i - 1, img) for i, img in enumerate(reversed(images[:min(3, len(images))]))]
        
        # Pokud nemáme relevantní obrázky, vrátíme prázdný řetězec a nulové tokeny
        if not relevant_images:
            return "", {"input_tokens": 0, "output_tokens": 0}
        
        # Dotaz pro každý relevantní obrázek
        query = self.QUERY_TEMPLATES.get(field, f"What is the {field} of this academic paper?")
        
        # Extrakce metadat z každého obrázku s textem
        results = []
        total_token_usage = {"input_tokens": 0, "output_tokens": 0}
        
        for page_idx, image in relevant_images:
            # Najít odpovídající text stránky
            page_text = ""
            for p_idx, text in page_texts:
                if p_idx == page_idx:
                    page_text = text
                    break
            
            try:
                result, token_usage = self.query_multimodal(image, page_text, query)
                if result:
                    results.append(result)
                    # Agregovat tokeny
                    total_token_usage["input_tokens"] += token_usage.get("input_tokens", 0)
                    total_token_usage["output_tokens"] += token_usage.get("output_tokens", 0)
            except Exception as e:
                print(f"Chyba při dotazu na multimodální model pro pole {field}: {e}")
        
        # Sloučení výsledků
        # Pro většinu polí vezmeme první výsledek
        if results:
            if field in ['abstract', 'references']:
                # Pro abstrakt a reference sloučíme všechny výsledky
                return "\n\n".join(results), total_token_usage
            else:
                # Pro ostatní pole vezmeme první výsledek
                return results[0], total_token_usage
        
        return "", total_token_usage
    
    def convert_pdf_to_images(self, pdf_path, dpi=200):
        """
        Konvertuje PDF na seznam PIL.Image objektů.
        
        Args:
            pdf_path (str): Cesta k PDF souboru
            dpi (int): DPI pro převod
            
        Returns:
            list: Seznam PIL.Image objektů
        """
        if PDF2IMAGE_AVAILABLE:
            try:
                print(f"Konvertuji PDF na obrázky (pdf2image): {pdf_path}")
                pages = convert_from_path(pdf_path, dpi=dpi)
                return pages
            except Exception as e:
                print(f"Chyba při konverzi PDF na obrázky pomocí pdf2image: {e}")
        
        if PYMUPDF_AVAILABLE:
            try:
                print(f"Konvertuji PDF na obrázky (PyMuPDF): {pdf_path}")
                doc = fitz.open(pdf_path)
                pages = []
                
                for page in doc:
                    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    pages.append(img)
                
                return pages
            except Exception as e:
                print(f"Chyba při konverzi PDF na obrázky pomocí PyMuPDF: {e}")
        
        print("Nepodařilo se konvertovat PDF na obrázky, žádná z metod není k dispozici.")
        return []
    
    def extract_metadata(self, pdf_path) -> tuple[dict, float | None, dict]:
        """
        Extrahuje metadata z PDF souboru pomocí multimodální metody.
        
        Args:
            pdf_path (str): Cesta k PDF souboru
            
        Returns:
            tuple: Extrahovaná metadata, doba trvání a slovník s použitými tokeny
        """
        start_time = time.perf_counter()
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Zpracovávám PDF soubor {pdf_path} (Multimodal)...")
        
        # Inicializace slovníku pro výsledky
        metadata_results = {field: "" for field in self.METADATA_FIELDS}
        total_token_usage = {"input_tokens": 0, "output_tokens": 0}

        try:
            # Konverze PDF na obrázky
            pages = self.convert_pdf_to_images(pdf_path)
            if not pages:
                raise ValueError("Nepodařilo se získat obrázky stránek z PDF")
            
            # Extrakce textu z PDF
            page_texts = self.extract_text_from_pdf(pdf_path)
            
            # Extrakce metadat z obrázků a textu
            for field in self.METADATA_FIELDS:
                print(f"  Extrahuji {field}...")
                field_value, token_usage = self.extract_metadata_from_combined_sources(pages, page_texts, field)
                metadata_results[field] = field_value
                
                # Přidání tokenů k celkovému počtu
                total_token_usage["input_tokens"] += token_usage.get("input_tokens", 0)
                total_token_usage["output_tokens"] += token_usage.get("output_tokens", 0)
            
            # Výpočet celkové doby trvání
            duration = time.perf_counter() - start_time
            print(f"Dokončeno zpracování {pdf_path} za {duration:.2f} sekund")
            
            return metadata_results, duration, total_token_usage
            
        except Exception as e:
            print(f"Chyba při extrakci metadat z {pdf_path}: {e}")
            duration = time.perf_counter() - start_time
            return metadata_results, duration, total_token_usage
    
    def extract_metadata_batch(self, pdf_paths, limit=None) -> tuple[dict, dict, dict]:
        """
        Extrahuje metadata ze seznamu PDF souborů.
        
        Args:
            pdf_paths (list): Seznam cest k PDF souborům
            limit (int, optional): Omezení počtu zpracovaných souborů
            
        Returns:
            tuple: Slovník s výsledky, slovník s časy zpracování a slovník s použitými tokeny
        """
        # Omezení počtu souborů, pokud je limit zadán
        if limit and limit > 0:
            pdf_paths = pdf_paths[:limit]
        
        batch_results = {}
        timings = {}
        token_usages = {}
        
        for pdf_path in tqdm(pdf_paths, desc="Extrahuji metadata (Multimodal)"):
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Extrakce metadat z PDF
            metadata_results, duration, token_usage = self.extract_metadata(pdf_path)
            
            # Uložení výsledků
            batch_results[paper_id] = metadata_results
            timings[paper_id] = duration
            token_usages[paper_id] = token_usage
        
        return batch_results, timings, token_usages


def extract_metadata_from_pdfs(pdf_dir, model_name=None, limit=None, provider_name=None, api_key=None, force_extraction=False) -> tuple[dict, dict, dict]:
    """
    Extrahuje metadata z PDF souborů použitím multimodální metody.
    
    Args:
        pdf_dir (str): Cesta k adresáři s PDF soubory
        model_name (str, optional): Název modelu
        limit (int, optional): Omezení počtu zpracovaných souborů
        provider_name (str, optional): Název poskytovatele API
        api_key (str, optional): API klíč
        force_extraction (bool): Vynutí novou extrakci metadat
        
    Returns:
        tuple: Výsledky extrakce, časy zpracování a použité tokeny
    """
    # Převod cesty na Path
    pdf_dir = Path(pdf_dir)
    
    # Získání seznamu PDF souborů
    pdf_files = sorted([os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')])
    
    # Kontrola, zda jsme už provedli extrakci
    results_path = get_run_results_dir() / "multimodal_results.json"
    
    if results_path.exists() and not force_extraction:
        print(f"Načítám existující výsledky z {results_path}...")
        with open(results_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            
        results = loaded_data.get("results", {})
        timings = loaded_data.get("timings", {})
        token_usages = loaded_data.get("token_usages", {})
        
        if results and not force_extraction:
            print(f"Načteno {len(results)} existujících výsledků. Pro novou extrakci použijte parametr --force-extraction.")
            return results, timings, token_usages
    
    # Inicializace multimodální pipeline
    pipeline = MultimodalPipeline(model_name=model_name, provider_name=provider_name, api_key=api_key)
    
    # Extrakce metadat
    results, timings, token_usages = pipeline.extract_metadata_batch(pdf_files, limit=limit)
    
    # Uložení výsledků
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "timings": timings,
            "token_usages": token_usages
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Výsledky uloženy do {results_path}")
    
    return results, timings, token_usages 