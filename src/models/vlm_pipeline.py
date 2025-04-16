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
from .providers.factory import ModelProviderFactory
from .config.model_config import get_config

# Import lokálního modulu pro analýzu PDF
try:
    # Místo relativního importu použijeme absolutní import
    import sys
    import os
    from pathlib import Path
    # Přidání nadřazeného adresáře do sys.path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.pdf_analyzer import PDFAnalyzer
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
        'publisher',       # Vydavatel
        'references'       # Seznam referencí
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
    
    def query_vlm(self, image, query):
        """
        Dotaz na VLM model s obrázkem.
        
        Args:
            image: PIL.Image objekt
            query (str): Dotaz pro model
            
        Returns:
            str: Odpověď modelu
        """
        # Použití poskytovatele pro generování textu z obrázku
        return self.vision_provider.generate_text_from_image(image, query)
    
    def extract_metadata_from_images(self, images, field):
        """
        Extrahuje metadata z obrázků.
        
        Args:
            images (list): Seznam PIL.Image objektů
            field (str): Pole metadat k extrakci
            
        Returns:
            str: Extrahovaná hodnota
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
        
        # Pokud nemáme relevantní obrázky, vrátíme prázdný řetězec
        if not relevant_images:
            return ""
        
        # Dotaz pro každý relevantní obrázek
        query = self.QUERY_TEMPLATES.get(field, f"What is the {field} of this academic paper?")
        
        # Extrakce metadat z každého obrázku
        results = []
        for image in relevant_images:
            try:
                result = self.query_vlm(image, query)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Chyba při extrakci pole {field} z obrázku: {e}")
        
        # Kombinace výsledků
        if not results:
            return ""
        
        # Pro většinu polí použijeme pouze první výsledek
        if field not in ['abstract', 'references']:
            return results[0]
        
        # Pro abstrakt a reference spojíme výsledky
        return "\n".join(results)
    
    def extract_metadata(self, pdf_path):
        """
        Extrahuje metadata z PDF souboru.
        
        Args:
            pdf_path (str): Cesta k PDF souboru
            
        Returns:
            dict: Extrahovaná metadata
        """
        # Převod PDF na obrázky
        images = self.convert_pdf_to_images(pdf_path)
        
        if not images:
            print(f"Nepodařilo se převést PDF soubor {pdf_path} na obrázky")
            return {}
        
        # Extrakce metadat
        metadata = {}
        for field in self.METADATA_FIELDS:
            print(f"Extrahuji pole {field}...")
            metadata[field] = self.extract_metadata_from_images(images, field)
        
        return metadata
    
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


def extract_metadata_from_pdfs(pdf_dir, output_file, model_name=None, limit=None, provider_name=None, api_key=None, force_extraction=False):
    """
    Hlavní funkce pro extrakci metadat z PDF souborů.
    
    Args:
        pdf_dir (str): Cesta k adresáři s PDF soubory
        output_file (str): Cesta k výstupnímu souboru
        model_name (str, optional): Název modelu
        limit (int, optional): Maximální počet PDF souborů ke zpracování
        provider_name (str, optional): Název poskytovatele API
        api_key (str, optional): API klíč
        force_extraction (bool): Vynutí novou extrakci i když výsledky již existují
    """
    # Inicializace pipeline
    pipeline = VLMPipeline(
        model_name=model_name,
        provider_name=provider_name,
        api_key=api_key
    )
    
    # Získání seznamu PDF souborů
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    # Omezení počtu souborů ke zpracování
    if limit:
        pdf_files = pdf_files[:limit]
    
    # Vždy vytváříme nové výsledky bez ohledu na existenci souborů
    print(f"Provádím novou extrakci metadat...")
    results = {}
    
    # Extrakce metadat z PDF souborů
    with tqdm(total=len(pdf_files), desc="Extrakce metadat") as pbar:
        for pdf_file in pdf_files:
            # Získání ID PDF z názvu souboru
            pdf_id = pdf_file.stem
            
            print(f"\nZpracovávám PDF soubor {pdf_file} (ID: {pdf_id})...")
            
            try:
                # Extrakce metadat z PDF souboru
                metadata = pipeline.extract_metadata(pdf_file)
                
                # Uložení metadat do výsledků
                results[pdf_id] = metadata
                
                # Uložení aktuálních výsledků (průběžné ukládání)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Chyba při zpracování PDF {pdf_file}: {e}")
            
            pbar.update(1)
    
    # Uložení konečných výsledků
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results


if __name__ == "__main__":
    # Příklad použití
    import sys
    import argparse
    
    # Vytvoření parseru argumentů
    parser = argparse.ArgumentParser(description='Extrakce metadat z PDF souborů pomocí VLM pipeline.')
    parser.add_argument('--pdf_dir', type=str, default=str(PDF_DIR), help='Cesta k adresáři s PDF soubory')
    parser.add_argument('--output_file', type=str, default=str(RESULTS_DIR / "vlm_results.json"), help='Cesta k výstupnímu souboru')
    parser.add_argument('--model_name', type=str, default=None, help='Název modelu')
    parser.add_argument('--limit', type=int, default=None, help='Omezení počtu zpracovaných souborů')
    parser.add_argument('--provider', type=str, default=None, help='Název poskytovatele API (openai, anthropic)')
    parser.add_argument('--config', type=str, default=None, help='Cesta ke konfiguračnímu souboru')
    
    # Parsování argumentů
    args = parser.parse_args()
    
    # Načtení konfigurace, pokud je zadána
    if args.config:
        from .config.model_config import load_config
        load_config(args.config)
    
    # Spuštění extrakce metadat
    results = extract_metadata_from_pdfs(
        args.pdf_dir, 
        args.output_file, 
        model_name=args.model_name,
        limit=args.limit,
        provider_name=args.provider
    )
    
    print(f"Extrakce metadat dokončena. Výsledky uloženy do {args.output_file}") 