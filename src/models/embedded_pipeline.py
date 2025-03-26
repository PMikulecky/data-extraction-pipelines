#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro implementaci Embedded pipeline pro extrakci metadat z PDF souborů.
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import pro OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

# Načtení proměnných prostředí
load_dotenv()

# Definice cest
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
RESULTS_DIR = BASE_DIR / "results"

# Vytvoření adresáře pro výsledky, pokud neexistuje
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class EmbeddedPipeline:
    """
    Třída pro implementaci Embedded pipeline pro extrakci metadat z PDF souborů.
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
        'references': "List the references cited in this academic paper. Return only the list of references without any additional text."
    }
    
    def __init__(self, model_name="gpt-3.5-turbo", chunk_size=1000, chunk_overlap=200):
        """
        Inicializace Embedded pipeline.
        
        Args:
            model_name (str): Název modelu pro ChatOpenAI
            chunk_size (int): Velikost chunků pro rozdělení textu
            chunk_overlap (int): Překryv chunků
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Inicializace komponent
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=0)
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extrahuje text z PDF souboru.
        
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
    
    def extract_metadata_field(self, text, field):
        """
        Extrahuje metadata z textu.
        
        Args:
            text (str): Text pro extrakci metadat
            field (str): Pole metadat k extrakci
            
        Returns:
            str: Extrahovaná hodnota
        """
        # Dotaz pro extrakci metadat
        query = self.QUERY_TEMPLATES.get(field, f"What is the {field} of this academic paper?")
        
        # Použití pouze prvních 4000 znaků textu pro každý dotaz
        context = text[:4000]
        prompt = f"Na základě následujícího textu z akademické práce odpověz na otázku: {query}\n\nText: {context}"
        
        # Extrakce metadat
        try:
            print(f"  Spouštím dotaz pro pole {field}...")
            result = self.llm.invoke(prompt).content
            return result.strip()
        except Exception as e:
            import traceback
            print(f"Chyba při extrakci pole {field}: {e}")
            print(f"Podrobnosti chyby: {traceback.format_exc()}")
            return ""
    
    def extract_metadata(self, pdf_path):
        """
        Extrahuje metadata z PDF souboru.
        
        Args:
            pdf_path (str): Cesta k PDF souboru
            
        Returns:
            dict: Extrahovaná metadata
        """
        # Extrakce textu z PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            print(f"Nepodařilo se extrahovat text z PDF souboru {pdf_path}")
            return {}
        
        # Rozdělení textu na chunky
        chunks = self.text_splitter.split_text(text)
        
        # Použití pouze prvních 3 chunků pro extrakci metadat
        text_for_extraction = "\n\n".join(chunks[:3])
        
        # Extrakce metadat
        metadata = {}
        for field in self.METADATA_FIELDS:
            print(f"Extrahuji pole {field}...")
            metadata[field] = self.extract_metadata_field(text_for_extraction, field)
        
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


def extract_metadata_from_pdfs(pdf_dir, output_file, model_name="gpt-3.5-turbo", limit=None):
    """
    Extrahuje metadata z PDF souborů v adresáři.
    
    Args:
        pdf_dir (str): Cesta k adresáři s PDF soubory
        output_file (str): Cesta k výstupnímu souboru
        model_name (str, optional): Název modelu pro ChatOpenAI
        limit (int, optional): Omezení počtu zpracovaných souborů
        
    Returns:
        dict: Extrahovaná metadata pro každý soubor
    """
    # Inicializace pipeline
    pipeline = EmbeddedPipeline(model_name=model_name)
    
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
    import argparse
    
    # Vytvoření parseru argumentů
    parser = argparse.ArgumentParser(description='Extrakce metadat z PDF souborů pomocí Embedded pipeline.')
    parser.add_argument('--pdf_dir', type=str, default=str(PDF_DIR), help='Cesta k adresáři s PDF soubory')
    parser.add_argument('--output_file', type=str, default=str(RESULTS_DIR / "embedded_results.json"), help='Cesta k výstupnímu souboru')
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo", help='Název modelu pro ChatOpenAI')
    parser.add_argument('--limit', type=int, default=None, help='Omezení počtu zpracovaných souborů')
    
    # Zpracování argumentů
    args = parser.parse_args()
    
    # Extrakce metadat
    results = extract_metadata_from_pdfs(args.pdf_dir, args.output_file, args.model_name, args.limit)
    
    print(f"\nExtrakce metadat dokončena. Výsledky uloženy do {args.output_file}")
    print(f"Zpracováno {len(results)} PDF souborů.") 