#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro implementaci VLM (Vision Language Model) pipeline pro extrakci metadat z PDF souborů.
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
    # Místo relativního importu použijeme absolutní import
    import sys
    import os
    from pathlib import Path
    # Přidání nadřazeného adresáře do sys.path
    # sys.path.append(str(Path(__file__).resolve().parent.parent)) # Odstraněno - řešeno absolutním importem
    # <<< Změna: Oprava importu na absolutní >>>
    from src.utils.pdf_analyzer import PDFAnalyzer
    # <<< Konec změny >>>
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


class VLMPipeline:
    """
    Třída pro implementaci VLM pipeline pro extrakci metadat z PDF souborů.
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
        'title': "What is the title of this academic paper? Return only the title without any additional text.",
        'authors': "Who are the authors of this academic paper? List all authors in the format 'First Name Last Name'. Return only the list of authors without any additional text.",
        'abstract': "What is the abstract of this academic paper? Return only the abstract without any additional text.",
        'keywords': "What are the keywords of this academic paper? Return only the keywords as a comma-separated list without any additional text.",
        'doi': "What is the DOI of this academic paper? Return only the DOI without any additional text.",
        'year': "In what year was this academic paper published? Return only the year without any additional text.",
        'journal': "In which journal or conference proceedings was this academic paper published? Return only the name of the journal or conference without any additional text.",
        'volume': "What is the volume number of the journal in which this academic paper was published? Return only the volume number without any additional text.",
        'issue': "What is the issue number of the journal in which this academic paper was published? Return only the issue number without any additional text.",
        'pages': "What are the page numbers of this academic paper in the journal or proceedings? Return only the page numbers (e.g., '123-145') without any additional text.",
        'publisher': "Who is the publisher of this academic paper? Return only the name of the publisher without any additional text.",
        'references': "List the first 5 references cited in this academic paper. Return only the list of references without any additional text."
    }
    
    def __init__(self, model_name=None, provider_name=None, api_key=None):
        """
        Inicializace VLM pipeline.
        
        Args:
            model_name (str, optional): Název modelu
            provider_name (str, optional): Název poskytovatele API
            api_key (str, optional): API klíč pro přístup k modelu
        """
        # Načtení konfigurace
        config = get_config()
        vision_config = config.get_vision_config()
        
        # Použití parametrů nebo konfigurace
        self.provider_name = provider_name or vision_config["provider"]
        self.model_name = model_name or vision_config["model"]
        self.api_key = api_key
        
        # Inicializace poskytovatele modelu
        self.vision_provider = ModelProviderFactory.create_vision_provider(
            provider_name=self.provider_name,
            model_name=self.model_name,
            api_key=self.api_key
        )
        
        print(f"Inicializován vizuální model: {self.model_name} od poskytovatele: {self.provider_name}")
    
    def query_vlm(self, image, query) -> tuple[str, dict]:
        """
        Dotaz na VLM model s obrázkem.
        
        Args:
            image: PIL.Image objekt
            query (str): Dotaz pro model
            
        Returns:
            tuple: Odpověď modelu (str) a slovník s tokeny (dict)
        """
        # Použití poskytovatele pro generování textu z obrázku
        # Předpokládáme, že provider nyní vrací (text, token_usage)
        text_result, token_usage = self.vision_provider.generate_text_from_image(image, query)
        return text_result, token_usage
    
    def extract_metadata_from_images(self, images, field) -> tuple[str, dict]:
        """
        Extrahuje metadata z obrázků.
        
        Args:
            images (list): Seznam PIL.Image objektů
            field (str): Pole metadat k extrakci
            
        Returns:
            tuple: Extrahovaná hodnota (str) a agregovaný slovník tokenů (dict)
        """
        # Výběr relevantních obrázků podle typu pole
        relevant_images = []
        
        if field in ['title', 'authors', 'doi', 'year', 'journal', 'publisher']:
            # Pro tyto pole použijeme pouze první stránku (titulní)
            if images and len(images) > 0:
                relevant_images = [images[0]]
        
        elif field == 'abstract':
            # Pro abstrakt použijeme první 2 stránky
            relevant_images = images[:min(2, len(images))]
        
        elif field in ['volume', 'issue', 'pages']:
            # Pro tyto pole použijeme první stránku
            if images and len(images) > 0:
                relevant_images = [images[0]]
        
        elif field == 'keywords':
            # Pro klíčová slova použijeme první 2 stránky
            relevant_images = images[:min(2, len(images))]
        
        elif field == 'references':
            # Pro reference použijeme poslední stránky
            relevant_images = images[-min(3, len(images)):]
        
        # Pokud nemáme relevantní obrázky, vrátíme prázdný řetězec a nulové tokeny
        if not relevant_images:
            return "", {"input_tokens": 0, "output_tokens": 0}
        
        # Dotaz pro každý relevantní obrázek
        query = self.QUERY_TEMPLATES.get(field, f"What is the {field} of this academic paper?")
        
        # Extrakce metadat z každého obrázku
        results = []
        total_token_usage = {"input_tokens": 0, "output_tokens": 0}
        for image in relevant_images:
            try:
                result, token_usage = self.query_vlm(image, query)
                if result:
                    results.append(result)
                    # Agregovat tokeny
                    total_token_usage["input_tokens"] += token_usage.get("input_tokens", 0)
                    total_token_usage["output_tokens"] += token_usage.get("output_tokens", 0)
            except Exception as e:
                print(f"Chyba při extrakci pole {field} z obrázku: {e}")
        
        # Kombinace výsledků
        if not results:
            return "", total_token_usage
        
        # Pro většinu polí použijeme pouze první výsledek
        final_result = ""
        if field not in ['abstract', 'references']:
            final_result = results[0]
        else:
            # Pro abstrakt a reference spojíme výsledky
            final_result = "\n".join(results)
            
        return final_result, total_token_usage
    
    def extract_metadata(self, pdf_path) -> tuple[dict, float | None, dict]:
        """
        Extrahuje metadata z PDF souboru.
        
        Args:
            pdf_path (str): Cesta k PDF souboru
            
        Returns:
            tuple(dict, float | None, dict): Extrahovaná metadata, doba trvání a celkové token usage
        """
        start_time = time.perf_counter() # Měření času - START
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Zpracovávám PDF soubor {pdf_path} (VLM)...")
        total_token_usage_doc = {"input_tokens": 0, "output_tokens": 0} # Inicializace pro dokument

        try:
            # Převod PDF na obrázky
            images = self.convert_pdf_to_images(pdf_path)

            if not images:
                print(f"Nepodařilo se převést PDF soubor {pdf_path} na obrázky")
                duration = time.perf_counter() - start_time
                return {}, duration, total_token_usage_doc # Vrátit i tokeny

            # Extrakce metadat
            metadata = {}
            for field in self.METADATA_FIELDS:
                print(f"Extrahuji pole {field}...")
                try:
                    # Získat hodnotu i tokeny pro toto pole
                    field_value, field_token_usage = self.extract_metadata_from_images(images, field)
                    metadata[field] = field_value
                    # Agregovat tokeny pro celý dokument
                    total_token_usage_doc["input_tokens"] += field_token_usage.get("input_tokens", 0)
                    total_token_usage_doc["output_tokens"] += field_token_usage.get("output_tokens", 0)
                except Exception as e:
                     print(f"Chyba při extrakci pole {field} pro {pdf_path}: {e}")
                     metadata[field] = ""

            duration = time.perf_counter() - start_time # Měření času - END
            print(f"Extrakce pro {paper_id} (VLM) trvala {duration:.2f} sekund.")
            print(f"Tokeny pro {paper_id} (VLM): Vstup={total_token_usage_doc['input_tokens']}, Výstup={total_token_usage_doc['output_tokens']}")
            return metadata, duration, total_token_usage_doc # Vrátit i tokeny
        except Exception as e:
             print(f"Obecná chyba při zpracování PDF {pdf_path} v extract_metadata (VLM): {e}")
             duration = time.perf_counter() - start_time
             return {}, duration, total_token_usage_doc # Vrátit i tokeny
    
    def convert_pdf_to_images(self, pdf_path, dpi=200):
        """
        Převede PDF soubor na seznam obrázků.
        
        Args:
            pdf_path (str): Cesta k PDF souboru
            dpi (int): Rozlišení obrázků
            
        Returns:
            list: Seznam PIL.Image objektů
        """
        images = []
        
        # Zkusíme nejprve PyMuPDF, protože nevyžaduje externí závislosti
        if PYMUPDF_AVAILABLE:
            try:
                print(f"Převádím PDF na obrázky pomocí PyMuPDF...")
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
                doc.close()
                
                if images:
                    print(f"Úspěšně převedeno {len(images)} stránek PDF na obrázky pomocí PyMuPDF.")
                    return images
                else:
                    print("PyMuPDF nevytvořil žádné obrázky, zkusím alternativní metodu.")
            except Exception as e:
                print(f"Chyba při převodu PDF na obrázky pomocí PyMuPDF: {e}")
                print("Zkusím alternativní metodu...")
        
        # Pokud PyMuPDF selhal nebo není k dispozici, zkusíme pdf2image
        if PDF2IMAGE_AVAILABLE:
            try:
                print(f"Převádím PDF na obrázky pomocí pdf2image (dpi={dpi})...")
                images = convert_from_path(pdf_path, dpi=dpi)
                
                if images:
                    print(f"Úspěšně převedeno {len(images)} stránek PDF na obrázky pomocí pdf2image.")
                    return images
                else:
                    print("pdf2image nevytvořil žádné obrázky.")
            except Exception as e:
                print(f"Chyba při převodu PDF na obrázky pomocí pdf2image: {e}")
        
        # Obě metody selhaly nebo nejsou k dispozici
        if not images:
            print("Žádná knihovna pro převod PDF na obrázky není k dispozici nebo všechny metody selhaly.")
            print("Pro správnou funkci VLM pipeline nainstalujte PyMuPDF (pip install pymupdf)")
            print("nebo pdf2image s Poppler (pip install pdf2image a nainstalujte Poppler do systému).")
        
        return images
    
    def extract_metadata_batch(self, pdf_paths, limit=None) -> tuple[dict, dict, dict]:
        """
        Extrahuje metadata z dávky PDF souborů.
        
        Args:
            pdf_paths (list): Seznam cest k PDF souborům
            limit (int, optional): Omezení počtu zpracovaných souborů
            
        Returns:
            tuple: Slovník s metadaty, slovník s časy, slovník s token usage
        """
        results = {}
        timings = {}
        token_usages = {} # Nový slovník pro tokeny
        
        # Omezení počtu souborů
        if limit:
            pdf_paths = pdf_paths[:limit]
            
        # Použití tqdm pro zobrazení průběhu
        for pdf_path in tqdm(pdf_paths, desc="Extrakce metadat (VLM)"):
            # start_time = time.time() # Už nepotřebujeme, měří se v extract_metadata
            pdf_id = Path(pdf_path).stem
            duration = None # Default
            token_usage = {"input_tokens": 0, "output_tokens": 0} # Default
            try:
                # <<< Změna: Ukládat metadata, duration a token usage >>>
                metadata_dict, duration, token_usage = self.extract_metadata(pdf_path)
                results[pdf_id] = metadata_dict
                # <<< Konec změny >>>
            except Exception as e:
                print(f"Chyba při zpracování souboru {pdf_path}: {e}")
                results[pdf_id] = None # Zaznamenat chybu
                # duration a token_usage zůstanou defaultní
            finally:
                # <<< Změna: Ukládat duration a token_usage získaný z extract_metadata >>>
                timings[pdf_id] = duration
                token_usages[pdf_id] = token_usage # Uložit tokeny
                if duration is not None:
                    print(f"Soubor {pdf_id} zpracován za {duration:.2f} sekund. Tokeny: V={token_usage.get('input_tokens',0)}, V={token_usage.get('output_tokens',0)}")
                else:
                    print(f"Soubor {pdf_id} - chyba při měření času extrakce.")
                # <<< Konec změny >>>
                
        return results, timings, token_usages # Vrátit i tokeny


def extract_metadata_from_pdfs(pdf_dir, model_name=None, limit=None, provider_name=None, api_key=None, force_extraction=False) -> tuple[dict, dict, dict]:
    """
    Extrahuje metadata z PDF souborů v daném adresáři pomocí VLM pipeline.
    
    Args:
        pdf_dir (str): Cesta k adresáři s PDF soubory
        model_name (str, optional): Název modelu
        limit (int, optional): Omezení počtu zpracovaných souborů
        provider_name (str, optional): Název poskytovatele API
        api_key (str, optional): API klíč
        force_extraction (bool): Vynutí novou extrakci metadat
        
    Returns:
        tuple: Slovník s metadaty, slovník s časy, slovník s token usage
    """
    output_file_path = get_run_results_dir() / "vlm_results.json"
    
    # Kontrola, zda výsledky již existují a není vynucena extrakce
    if not force_extraction and output_file_path.exists():
        print(f"Načítám existující výsledky z {output_file_path}...")
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Očekáváme formát {"results": {...}, "timings": {...}, "token_usages": {...}}
            if isinstance(data, dict) and "results" in data and "timings" in data and "token_usages" in data:
                 print("Nalezena i data o tokenech.")
                 return data["results"], data["timings"], data["token_usages"]
            # Fallback pro starší formát bez tokenů
            elif isinstance(data, dict) and "results" in data and "timings" in data:
                 print("VAROVÁNÍ: Data o tokenech nebyla nalezena v existujícím souboru. Vracím prázdný slovník tokenů.")
                 return data["results"], data["timings"], {}
            else:
                 print("VAROVÁNÍ: Formát existujícího souboru s výsledky neodpovídá očekávání. Spouštím novou extrakci.")
        except json.JSONDecodeError:
            print(f"Chyba při čtení JSON souboru {output_file_path}. Spouštím novou extrakci.")
        except Exception as e:
            print(f"Neočekávaná chyba při načítání výsledků z {output_file_path}: {e}. Spouštím novou extrakci.")

    # Inicializace pipeline
    pipeline = VLMPipeline(
        model_name=model_name,
        provider_name=provider_name,
        api_key=api_key
    )
    
    # Získání seznamu PDF souborů
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    # Extrakce metadat
    results, timings, token_usages = pipeline.extract_metadata_batch(pdf_files, limit=limit)
    
    # Uložení výsledků ve strukturovaném formátu
    output_data = {
        "results": results,
        "timings": timings,
        "token_usages": token_usages # Přidat tokeny
    }
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Výsledky extrakce (VLM) včetně tokenů uloženy do {output_file_path}")
    except Exception as e:
        print(f"Chyba při ukládání výsledků do {output_file_path}: {e}")
        
    return results, timings, token_usages # Vrátit i tokeny


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Spustí VLM pipeline pro extrakci metadat z PDF souborů.')
    parser.add_argument('--pdf_dir', type=str, default=str(PDF_DIR), help='Cesta k adresáři s PDF soubory')
    # <<< Změna: Odstranění argumentu --output_file >>>
    # parser.add_argument('--output_file', type=str, default=str(RESULTS_DIR / "vlm_results.json"), help='Cesta k výstupnímu souboru')
    # <<< Konec změny >>>
    parser.add_argument('--model_name', type=str, default=None, help='Název modelu')
    parser.add_argument('--limit', type=int, default=None, help='Omezení počtu zpracovaných souborů')
    parser.add_argument('--provider_name', type=str, default=None, help='Název poskytovatele API (openai, anthropic)')
    parser.add_argument('--force', action='store_true', help='Vynutit novou extrakci i když výsledky již existují')
    
    args = parser.parse_args()
    
    # <<< Změna: Nastavení výchozího adresáře pro běh, pokud je skript spuštěn samostatně >>>
    from src.config.runtime_config import set_run_results_dir
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run_dir = Path(__file__).resolve().parent.parent.parent / "results" / f"vlm_standalone_{timestamp}"
    set_run_results_dir(default_run_dir)
    # <<< Konec změny >>>
    
    # Načtení API klíče (pokud je potřeba)
    api_key = os.getenv("ANTHROPIC_API_KEY") if args.provider_name == "anthropic" else os.getenv("OPENAI_API_KEY") if args.provider_name == "openai" else None

    extract_metadata_from_pdfs(
        pdf_dir=args.pdf_dir, 
        # output_file=args.output_file, # Odstraněno
        model_name=args.model_name, 
        limit=args.limit,
        provider_name=args.provider_name,
        api_key=api_key,
        force_extraction=args.force
    ) 