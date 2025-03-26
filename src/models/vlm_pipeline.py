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
from pdf2image import convert_from_path
from dotenv import load_dotenv

# Import lokálního modulu pro analýzu PDF
try:
    from ..utils.pdf_analyzer import PDFAnalyzer
except ImportError:
    # Pokud relativní import selže, zkusíme absolutní import
    from src.utils.pdf_analyzer import PDFAnalyzer

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
    
    def __init__(self, model_name="gpt-4-vision-preview", api_key=None):
        """
        Inicializace VLM pipeline.
        
        Args:
            model_name (str): Název modelu pro VLM
            api_key (str, optional): API klíč pro přístup k modelu
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API klíč není k dispozici. Nastavte proměnnou prostředí OPENAI_API_KEY nebo předejte api_key parametr.")
    
    def encode_image(self, image):
        """
        Zakóduje obrázek do base64.
        
        Args:
            image: PIL.Image objekt
            
        Returns:
            str: Zakódovaný obrázek v base64
        """
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def query_vlm(self, image, query):
        """
        Dotaz na VLM model s obrázkem.
        
        Args:
            image: PIL.Image objekt
            query (str): Dotaz pro model
            
        Returns:
            str: Odpověď modelu
        """
        # Zakódování obrázku
        base64_image = self.encode_image(image)
        
        # Příprava dotazu pro API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }
        
        # Odeslání dotazu
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        # Zpracování odpovědi
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Chyba při dotazu na VLM: {response.status_code} - {response.text}")
            return ""
    
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
        # Analýza PDF pro identifikaci klíčových částí
        analyzer = PDFAnalyzer(pdf_path)
        analyzer.analyze()
        
        # Získání obrázků stránek
        title_image = analyzer.get_title_page_image()
        abstract_image = analyzer.get_abstract_page_image()
        reference_images = analyzer.get_reference_page_images()
        
        # Kombinace všech obrázků
        all_images = []
        if title_image:
            all_images.append(title_image)
        if abstract_image and abstract_image != title_image:
            all_images.append(abstract_image)
        all_images.extend([img for img in reference_images if img not in all_images])
        
        # Extrakce metadat
        metadata = {}
        for field in self.METADATA_FIELDS:
            print(f"Extrahuji pole {field}...")
            metadata[field] = self.extract_metadata_from_images(all_images, field)
        
        return metadata
    
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


def extract_metadata_from_pdfs(pdf_dir, output_file, model_name="gpt-4-vision-preview", limit=None):
    """
    Extrahuje metadata z PDF souborů v adresáři pomocí VLM.
    
    Args:
        pdf_dir (str): Cesta k adresáři s PDF soubory
        output_file (str): Cesta k výstupnímu souboru
        model_name (str, optional): Název modelu pro VLM
        limit (int, optional): Omezení počtu zpracovaných souborů
        
    Returns:
        dict: Extrahovaná metadata pro každý soubor
    """
    # Inicializace pipeline
    pipeline = VLMPipeline(model_name=model_name)
    
    # Získání seznamu PDF souborů
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    # Omezení počtu souborů, pokud je zadáno
    if limit:
        pdf_files = pdf_files[:limit]
    
    # Extrakce metadat
    results = pipeline.extract_metadata_batch(pdf_files, output_file)
    
    return results


if __name__ == "__main__":
    # Příklad použití
    import sys
    
    # Výchozí hodnoty
    pdf_directory = PDF_DIR
    output_file = RESULTS_DIR / "vlm_results.json"
    model_name = "gpt-4-vision-preview"
    limit = 5  # Omezení počtu souborů pro testování
    
    # Zpracování argumentů příkazové řádky
    if len(sys.argv) > 1:
        pdf_directory = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        model_name = sys.argv[3]
    if len(sys.argv) > 4:
        limit = int(sys.argv[4])
    
    # Extrakce metadat
    results = extract_metadata_from_pdfs(pdf_directory, output_file, model_name, limit)
    
    print(f"\nExtrakce metadat dokončena. Výsledky uloženy do {output_file}")
    print(f"Zpracováno {len(results)} PDF souborů.") 