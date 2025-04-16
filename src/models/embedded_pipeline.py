#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro implementaci Embedded pipeline pro extrakci metadat z PDF souborů.
"""

import os
import json
import numpy as np
import uuid
from pathlib import Path
from tqdm import tqdm
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Import pro poskytovatele modelů
from .providers.factory import ModelProviderFactory
from .config.model_config import get_config

from dotenv import load_dotenv

# Načtení proměnných prostředí
load_dotenv()

# Definice cest
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
RESULTS_DIR = BASE_DIR / "results"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Vytvoření adresářů pro výsledky a vectorstore, pokud neexistují
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


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
    
    def __init__(self, model_name=None, chunk_size=1000, chunk_overlap=200, provider_name=None, api_key=None, embedding_model_name=None, embedding_provider_name=None, vectorstore_path=None):
        """
        Inicializace Embedded pipeline.
        
        Args:
            model_name (str, optional): Název modelu pro použití
            chunk_size (int): Velikost chunků pro rozdělení textu
            chunk_overlap (int): Překryv chunků
            provider_name (str, optional): Název poskytovatele API
            api_key (str, optional): API klíč
            embedding_model_name (str, optional): Název modelu pro embeddings
            embedding_provider_name (str, optional): Název poskytovatele API pro embeddings
            vectorstore_path (str, optional): Cesta k adresáři pro vectorstore
        """
        # Načtení konfigurace
        config = get_config()
        text_config = config.get_text_config()
        embedding_config = config.get_embedding_config()
        
        # Použití parametrů nebo konfigurace
        self.provider_name = provider_name or text_config["provider"]
        self.model_name = model_name or text_config["model"]
        self.embedding_provider_name = embedding_provider_name or embedding_config["provider"]
        self.embedding_model_name = embedding_model_name or embedding_config["model"]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.api_key = api_key
        self.vectorstore_path = vectorstore_path or str(VECTORSTORE_DIR / "embedded_pipeline")
        
        # Inicializace komponent
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        # Inicializace poskytovatele modelu pro text
        self.text_provider = ModelProviderFactory.create_text_provider(
            provider_name=self.provider_name,
            model_name=self.model_name,
            api_key=self.api_key
        )
        
        # Inicializace poskytovatele modelu pro embeddings
        self.embedding_provider = ModelProviderFactory.create_embedding_provider(
            provider_name=self.embedding_provider_name,
            model_name=self.embedding_model_name,
            api_key=self.api_key
        )
        
        # Inicializace vectorstore
        self.vectorstore = None
        
        print(f"Inicializován textový model: {self.model_name} od poskytovatele: {self.provider_name}")
        print(f"Inicializován embedding model: {self.embedding_model_name} od poskytovatele: {self.embedding_provider_name}")
    
    def initialize_vectorstore(self):
        """
        Inicializuje vectorstore. Pokud již existuje, načte ho, jinak vytvoří nový.
        """
        os.makedirs(self.vectorstore_path, exist_ok=True)
        
        # Kontrola, zda už existuje vectorstore
        if os.path.exists(os.path.join(self.vectorstore_path, "chroma.sqlite3")):
            print(f"Načítám existující vectorstore z {self.vectorstore_path}")
            
            # Použití adapteru pro embedding provider
            embedding_function = self.embedding_provider.get_embedding_function()
            
            # Načtení existujícího vectorstore
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=embedding_function
            )
        else:
            print(f"Vytvářím nový vectorstore v {self.vectorstore_path}")
            
            # Vectorstore bude vytvořen později při vkládání dokumentů
            self.vectorstore = None
    
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
    
    def create_document_chunks(self, text, metadata=None):
        """
        Vytvoří chunky dokumentu z textu s metadaty.
        
        Args:
            text (str): Text dokumentu
            metadata (dict, optional): Metadata dokumentu
            
        Returns:
            list: Seznam dokumentů
        """
        # Import modulu Document z langchain_core
        from langchain_core.documents import Document
        
        # Rozdělení textu na chunky
        chunks = self.text_splitter.split_text(text)
        print(f"Text rozdělen na {len(chunks)} chunků")
        
        # Vytvoření dokumentů s metadaty
        documents = []
        for i, chunk in enumerate(chunks):
            # Zkopírování a zpracování metadat
            doc_metadata = {}
            if metadata:
                # Převedení všech metadat na jednoduché typy (str, int, float, bool)
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        doc_metadata[key] = value
                    else:
                        # Převod složitých typů na řetězce
                        doc_metadata[key] = str(value)
            
            # Přidání indexu chunku
            doc_metadata["chunk"] = i
            
            # Vytvoření dokumentu s langchain Document formátem
            doc = Document(
                page_content=chunk,
                metadata=doc_metadata
            )
            documents.append(doc)
        
        print(f"Vytvořeno {len(documents)} Document objektů")
        if documents:
            print(f"Ukázka prvního dokumentu - type: {type(documents[0])}, page_content: {documents[0].page_content[:30]}...")
        
        return documents
    
    def add_document_to_vectorstore(self, documents):
        """
        Přidá dokument do vectorstore.
        
        Args:
            documents (list): Seznam dokumentů s formátem Document
        """
        # Kontrola, že documents jsou správného typu
        if not documents:
            print("VAROVÁNÍ: Seznam dokumentů je prázdný")
            return
        
        print(f"Vytvářím nový vectorstore v {self.vectorstore_path}")
        
        # Pokud stále neexistuje, vytvoříme nový
        if self.vectorstore is None:
            print("Vytvářím nový vectorstore s dokumenty aktuálního PDF")
            
            # Generování unikátních ID pro dokumenty
            ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents]
            print(f"Vygenerováno {len(ids)} unikátních ID")
            
            # Získání embedding funkce
            embedding_function = self.embedding_provider.get_embedding_function()
            print(f"Embedding funkce vytvořena: {embedding_function}")
            
            try:
                # Vytvoření jedinečné složky pro tento vectorstore
                import random
                unique_path = os.path.join(self.vectorstore_path, f"doc_{random.randint(10000, 99999)}")
                os.makedirs(unique_path, exist_ok=True)
                print(f"Vytvářím Chroma vectorstore s {len(documents)} dokumenty v {unique_path}")
                
                # Vytvoření vectorstore
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embedding_function,
                    persist_directory=unique_path,
                    ids=ids
                )
                print(f"Chroma vectorstore vytvořen úspěšně")
                # Metoda persist() již není v nejnovější verzi Chroma podporována
            except Exception as e:
                print(f"CHYBA při vytváření vectorstore: {e}")
                import traceback
                print(traceback.format_exc())
        else:
            print("Přidávám dokumenty do existujícího vectorstore")
            
            # Generování unikátních ID pro dokumenty
            ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents]
            
            # Přidání dokumentů do existujícího vectorstore
            try:
                self.vectorstore.add_documents(
                    documents=documents,
                    ids=ids
                )
                print(f"Úspěšně přidáno {len(documents)} dokumentů do existujícího vectorstore")
            except Exception as e:
                print(f"CHYBA při přidávání dokumentů do vectorstore: {e}")
                import traceback
                print(traceback.format_exc())
    
    def search_similar_chunks(self, query, limit=5):
        """
        Vyhledá podobné chunky v vectorstore.
        
        Args:
            query (str): Dotaz pro vyhledávání
            limit (int): Počet výsledků
            
        Returns:
            list: Seznam podobných dokumentů
        """
        if self.vectorstore is None:
            print("Vectorstore není inicializován")
            return []
        
        # Vyhledání podobných dokumentů
        docs = self.vectorstore.similarity_search(query, k=limit)
        return docs
    
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
        query = self.QUERY_TEMPLATES.get(field, f"Co je {field} této akademické práce?")
        
        # Pokud je vectorstore inicializován, použijeme podobnostní vyhledávání
        if hasattr(self, 'vectorstore') and self.vectorstore is not None:
            similar_chunks = self.search_similar_chunks(query, limit=5)  # Zvýšení limitu z 2 na 5
            if similar_chunks:
                context = "\n\n".join([doc.page_content for doc in similar_chunks])
            else:
                # Pokud nejsou žádné podobné chunky, použijeme prvních 4000 znaků textu
                context = text[:4000]
        else:
            # Použití pouze prvních 4000 znaků textu pro každý dotaz
            context = text[:4000]
        
        prompt = f"Na základě následujícího textu z akademické práce odpověz na otázku: {query}\n\nText: {context}\n\nOdpověz pouze požadovanou informací bez dalšího vysvětlování."
        
        # Extrakce metadat pomocí poskytovatele
        try:
            print(f"  Spouštím dotaz pro pole {field}...")
            result = self.text_provider.generate_text(prompt)
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
        
        # Reset a vytvoření nového vectorstore pouze pro tento dokument
        self.vectorstore = None
        print(f"Vytvářím nový vectorstore pro dokument {pdf_path}")
        
        # Vytvoření chunků z dokumentu
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
        metadata = {"source": pdf_path, "paper_id": paper_id}
        documents = self.create_document_chunks(text, metadata)
        
        # Přidání dokumentu do vectorstore
        print(f"Přidávám {len(documents)} chunků do nového vectorstore")
        self.add_document_to_vectorstore(documents)
        
        # Extrakce metadat
        metadata_result = {}
        for field in self.METADATA_FIELDS:
            print(f"Extrahuji pole {field}...")
            value = self.extract_metadata_field(text, field)
            metadata_result[field] = value
            print(f"Extrahovaná hodnota pro pole {field}: {value[:100]}..." if len(value) > 100 else f"Extrahovaná hodnota pro pole {field}: {value}")
        
        return metadata_result
    
    def extract_metadata_batch(self, pdf_paths, output_file=None):
        """
        Extrahuje metadata z více PDF souborů.
        
        Args:
            pdf_paths (list): Seznam cest k PDF souborům
            output_file (str, optional): Cesta k výstupnímu souboru
            
        Returns:
            dict: Extrahovaná metadata pro každý soubor
        """
        # Nebudeme inicializovat vectorstore zde - vytvoříme ho pro každý dokument zvlášť
        
        results = {}
        
        for pdf_path in tqdm(pdf_paths, desc="Extrakce metadat"):
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
            print(f"\nZpracovávám PDF soubor {pdf_path} (ID: {paper_id})...")
            
            try:
                # Reset vectorstore pro každý dokument
                self.vectorstore = None
                
                metadata = self.extract_metadata(pdf_path)
                results[paper_id] = metadata
                
                # Průběžné ukládání výsledků
                if output_file:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                
                # Po extrakci metadat zrušíme vectorstore, aby se nepoužíval pro další dokumenty
                self.vectorstore = None
                
            except Exception as e:
                print(f"Chyba při zpracování PDF souboru {pdf_path}: {e}")
                results[paper_id] = {"error": str(e)}
        
        return results


def extract_metadata_from_pdfs(pdf_dir, output_file, model_name=None, limit=None, provider_name=None, api_key=None, force_extraction=False, embedding_model_name=None, embedding_provider_name=None, vectorstore_path=None):
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
        embedding_model_name (str, optional): Název modelu pro embeddings
        embedding_provider_name (str, optional): Název poskytovatele API pro embeddings
        vectorstore_path (str, optional): Cesta k adresáři pro vectorstore
    """
    # Načtení konfigurace
    config = get_config()
    text_config = config.get_text_config()
    embedding_config = config.get_embedding_config()
    
    # Pokud není explicitně zadán provider, načteme ho z konfigurace
    if provider_name is None:
        provider_name = text_config["provider"]
        print(f"Používám provider z konfigurace: {provider_name}")
    else:
        print(f"Používám explicitně zadaný provider: {provider_name}")
        
    # Pokud není explicitně zadán model, načteme ho z konfigurace
    if model_name is None:
        model_name = text_config["model"]
        print(f"Používám model z konfigurace: {model_name}")
    else:
        print(f"Používám explicitně zadaný model: {model_name}")
    
    # Pokud není explicitně zadán embedding provider, načteme ho z konfigurace
    if embedding_provider_name is None:
        embedding_provider_name = embedding_config["provider"]
        print(f"Používám embedding provider z konfigurace: {embedding_provider_name}")
    else:
        print(f"Používám explicitně zadaný embedding provider: {embedding_provider_name}")
    
    # Pokud není explicitně zadán embedding model, načteme ho z konfigurace
    if embedding_model_name is None:
        embedding_model_name = embedding_config["model"]
        print(f"Používám embedding model z konfigurace: {embedding_model_name}")
    else:
        print(f"Používám explicitně zadaný embedding model: {embedding_model_name}")
    
    print(f"Inicializuji EmbeddedPipeline s provider={provider_name}, model={model_name}, embedding_provider={embedding_provider_name}, embedding_model={embedding_model_name}")
    
    # Vytvoření instance EmbeddedPipeline
    pipeline = EmbeddedPipeline(
        model_name=model_name,
        provider_name=provider_name,
        api_key=api_key,
        embedding_model_name=embedding_model_name,
        embedding_provider_name=embedding_provider_name,
        vectorstore_path=vectorstore_path
    )
    
    print(f"EmbeddedPipeline inicializován s provider={pipeline.provider_name}, model={pipeline.model_name}, embedding_provider={pipeline.embedding_provider_name}, embedding_model={pipeline.embedding_model_name}")
    
    # Získání seznamu PDF souborů
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    # Omezení počtu souborů ke zpracování
    if limit:
        pdf_files = pdf_files[:limit]
    
    # Extrakce metadat
    print(f"\n=== Extrakce metadat pomocí Embedded pipeline ===")
    
    # Vždy vytváříme nové výsledky bez ohledu na existenci souborů
    print(f"Provádím novou extrakci metadat...")
    results = {}
    
    # Extrakce metadat z PDF souborů
    metadata = pipeline.extract_metadata_batch(pdf_files, output_file)
    
    return metadata


if __name__ == "__main__":
    # Příklad použití
    import sys
    import argparse
    
    # Vytvoření parseru argumentů
    parser = argparse.ArgumentParser(description='Extrakce metadat z PDF souborů pomocí Embedded pipeline.')
    parser.add_argument('--pdf_dir', type=str, default=str(PDF_DIR), help='Cesta k adresáři s PDF soubory')
    parser.add_argument('--output_file', type=str, default=str(RESULTS_DIR / "embedded_results.json"), help='Cesta k výstupnímu souboru')
    parser.add_argument('--model_name', type=str, default=None, help='Název modelu')
    parser.add_argument('--limit', type=int, default=None, help='Omezení počtu zpracovaných souborů')
    parser.add_argument('--provider', type=str, default=None, help='Název poskytovatele API (openai, anthropic)')
    parser.add_argument('--config', type=str, default=None, help='Cesta ke konfiguračnímu souboru')
    parser.add_argument('--embedding_model', type=str, default=None, help='Název embedding modelu')
    parser.add_argument('--embedding_provider', type=str, default=None, help='Název poskytovatele embedding API')
    parser.add_argument('--vectorstore_path', type=str, default=None, help='Cesta k adresáři pro vectorstore')
    parser.add_argument('--force', action='store_true', help='Vynutit novou extrakci i když výsledky již existují')
    
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
        provider_name=args.provider,
        force_extraction=args.force,
        embedding_model_name=args.embedding_model,
        embedding_provider_name=args.embedding_provider,
        vectorstore_path=args.vectorstore_path
    )
    
    print(f"Extrakce metadat dokončena. Výsledky uloženy do {args.output_file}") 