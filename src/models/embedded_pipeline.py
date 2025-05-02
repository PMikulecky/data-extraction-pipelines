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
import time
import shutil

# Import pro poskytovatele modelů
from src.models.providers.factory import ModelProviderFactory
from src.models.config.model_config import get_config
from src.config.runtime_config import get_run_results_dir

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
    
    def __init__(self, model_name=None, chunk_size=1000, chunk_overlap=200, provider_name=None, api_key=None, embedding_model_name=None, embedding_provider_name=None, vectorstore_path=None, text_api_key=None, embedding_api_key=None):
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
            text_api_key (str, optional): API klíč pro textový model
            embedding_api_key (str, optional): API klíč pro embedding model
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
        self.text_api_key = text_api_key
        self.embedding_api_key = embedding_api_key
        
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
            api_key=self.text_api_key
        )
        
        # Inicializace poskytovatele modelu pro embeddings
        self.embedding_provider = ModelProviderFactory.create_embedding_provider(
            provider_name=self.embedding_provider_name,
            model_name=self.embedding_model_name,
            api_key=self.embedding_api_key
        )
        
        # Inicializace vectorstore
        self.vectorstore = None
        
        print(f"Inicializován textový model: {self.model_name} od poskytovatele: {self.provider_name}")
        print(f"Inicializován embedding model: {self.embedding_model_name} od poskytovatele: {self.embedding_provider_name}")
    
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
        try:
            docs = self.vectorstore.similarity_search(query, k=limit)
            return docs
        except Exception as e:
             print(f"Chyba při similarity_search: {e}")
             return []

    # PŘEJMENOVÁNO A UPRAVENO pro tokeny
    def extract_field_value_with_context(self, context: str, field: str) -> tuple[str, dict]:
        """
        Extrahuje hodnotu pole z daného kontextu pomocí LLM.
        
        Args:
            context (str): Kontext získaný z relevantních chunků.
            field (str): Název pole metadat.
            
        Returns:
            tuple: Extrahovaná hodnota (str) a slovník s tokeny (dict).
        """
        query = self.QUERY_TEMPLATES.get(field, f"Co je {field} této akademické práce?")
        prompt = f"Na základě následujícího textu z akademické práce odpověz na otázku: {query}\n\nText: {context}\n\nOdpověz pouze požadovanou informací bez dalšího vysvětlování."
        
        token_usage = {"input_tokens": 0, "output_tokens": 0} # Default
        try:
            print(f"  Spouštím dotaz pro pole '{field}'...")
            # Získáme text i tokeny
            value, token_usage = self.text_provider.generate_text(prompt)
            print(f"  Extrahovaná hodnota pro '{field}': {value[:50]}... Tokeny: {token_usage}")
            return value.strip(), token_usage
        except Exception as e:
            import traceback
            print(f"Chyba při extrakci pole '{field}' s LLM: {e}")
            # print(f"Podrobnosti chyby: {traceback.format_exc()}")
            return "", token_usage # Vrátit default i při chybě

    def extract_metadata(self, pdf_path) -> tuple[dict, float | None, dict]: # Změna návratového typu
        """
        Extrahuje metadata z PDF souboru.
        
        Args:
            pdf_path (str): Cesta k PDF souboru
            
        Returns:
            tuple(dict, float | None, dict): Extrahovaná metadata, doba trvání a celkové token usage
        """
        start_time = time.perf_counter()
        paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Zpracovávám PDF soubor {pdf_path} (Embedded)...")
        total_token_usage_doc = {"input_tokens": 0, "output_tokens": 0}
        doc_vectorstore_path = None
        doc_vectorstore = None
        metadata_result = {}

        try:
            # Extrakce textu
            full_text = self.extract_text_from_pdf(pdf_path)
            if not full_text:
                print(f"Nepodařilo se extrahovat text z {pdf_path}")
                duration = time.perf_counter() - start_time
                return {}, duration, total_token_usage_doc

            # Vytvoření chunků a vectorstore pro tento dokument
            doc_metadata = {"source": str(pdf_path)} 
            documents = self.create_document_chunks(full_text, metadata=doc_metadata)
            
            if not documents:
                 print(f"Nepodařilo se vytvořit chunky pro {pdf_path}")
                 duration = time.perf_counter() - start_time
                 return {}, duration, total_token_usage_doc

            # Použijeme unikátní cestu pro každý dokument ve složce běhu
            run_vectorstore_dir = Path(get_run_results_dir()) / "vectorstore_cache" / f"doc_{paper_id}_{uuid.uuid4().hex[:8]}"
            doc_vectorstore_path = str(run_vectorstore_dir)
            run_vectorstore_dir.mkdir(parents=True, exist_ok=True)

            embedding_function = self.embedding_provider.get_embedding_function()
            print(f"Vytvářím vectorstore pro {paper_id} v {doc_vectorstore_path}...")
            doc_vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embedding_function,
                persist_directory=doc_vectorstore_path
            )
            print("Vectorstore pro dokument vytvořen.")

            # Extrakce jednotlivých polí s využitím vectorstore
            for field in self.METADATA_FIELDS:
                print(f"Extrahuji pole {field}...")
                query = self.QUERY_TEMPLATES.get(field, f"Jaká je hodnota pole {field}?")
                try:
                    # Vyhledání relevantních chunků
                    similar_chunks = doc_vectorstore.similarity_search(query, k=5)
                    if not similar_chunks:
                         print(f"  Nenalezeny žádné podobné chunky pro pole '{field}'.")
                         metadata_result[field] = ""
                         continue 
                         
                    context = "\n\n".join([chunk.page_content for chunk in similar_chunks])
                    
                    # Extrakce hodnoty a tokenů z kontextu
                    value, field_token_usage = self.extract_field_value_with_context(context, field)
                    metadata_result[field] = value
                    
                    # Agregace tokenů
                    total_token_usage_doc["input_tokens"] += field_token_usage.get("input_tokens", 0)
                    total_token_usage_doc["output_tokens"] += field_token_usage.get("output_tokens", 0)
                    
                except Exception as e:
                    print(f"Chyba při hledání nebo extrakci pole {field} pro {pdf_path}: {e}")
                    metadata_result[field] = ""

            duration = time.perf_counter() - start_time # Měření času - END
            print(f"Extrakce pro {paper_id} (Embedded) trvala {duration:.2f} sekund.")
            print(f"Tokeny pro {paper_id} (Embedded): Vstup={total_token_usage_doc['input_tokens']}, Výstup={total_token_usage_doc['output_tokens']}")
            return metadata_result, duration, total_token_usage_doc # Vrátit i tokeny
            
        except Exception as e:
            import traceback
            print(f"Obecná chyba při zpracování {pdf_path} v extract_metadata (Embedded): {e}")
            # print(traceback.format_exc())
            duration = time.perf_counter() - start_time
            return {}, duration, total_token_usage_doc # Vrátit prázdná metadata a dobu trvání do chyby
        finally:
             # Vyčištění vectorstore, pokud byl vytvořen
             if doc_vectorstore_path and os.path.exists(doc_vectorstore_path):
                 try:
                     print(f"Mažu dočasný vectorstore: {doc_vectorstore_path}")
                     shutil.rmtree(doc_vectorstore_path)
                 except Exception as cleanup_e:
                      print(f"Chyba při mazání vectorstore {doc_vectorstore_path}: {cleanup_e}")

    def extract_metadata_batch(self, pdf_paths, limit=None) -> tuple[dict, dict, dict]: # Změna návratového typu
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
        
        if limit: pdf_paths = pdf_paths[:limit]
            
        for pdf_path in tqdm(pdf_paths, desc="Extrakce metadat (Embedded)"):
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
                timings[pdf_id] = duration
                token_usages[pdf_id] = token_usage # Uložit tokeny
                
        return results, timings, token_usages # Vrátit i tokeny


def extract_metadata_from_pdfs(pdf_dir, limit=None, provider_name=None, force_extraction=False, embedding_model_name=None, embedding_provider_name=None, model_name=None, vectorstore_path=None, text_api_key=None, embedding_api_key=None) -> tuple[dict, dict, dict]: # Změna návratového typu
    """
    Extrahuje metadata z PDF souborů v daném adresáři pomocí Embedded pipeline.
    
    Args:
        pdf_dir (str): Cesta k adresáři s PDF soubory
        limit (int, optional): Omezení počtu zpracovaných souborů
        provider_name (str, optional): Název poskytovatele API pro textový model
        force_extraction (bool): Vynutí novou extrakci metadat
        embedding_model_name (str, optional): Název modelu pro embeddings
        embedding_provider_name (str, optional): Název poskytovatele API pro embeddings
        model_name (str, optional): Název textového modelu
        vectorstore_path (str, optional): Cesta k adresáři pro vectorstore
        text_api_key (str, optional): API klíč pro textový model
        embedding_api_key (str, optional): API klíč pro embedding model
        
    Returns:
        tuple: Slovník s metadaty, slovník s časy, slovník s token usage
    """
    output_file_path = get_run_results_dir() / "embedded_results.json"
    
    # Kontrola, zda výsledky již existují
    if not force_extraction and output_file_path.exists():
        print(f"Načítám existující výsledky z {output_file_path}...")
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and "results" in data and "timings" in data and "token_usages" in data:
                 print("Nalezena i data o tokenech.")
                 return data["results"], data["timings"], data["token_usages"]
            elif isinstance(data, dict) and "results" in data and "timings" in data:
                 print("VAROVÁNÍ: Data o tokenech nebyla nalezena. Vracím prázdný slovník tokenů.")
                 return data["results"], data["timings"], {}
            else:
                 print("VAROVÁNÍ: Neplatný formát souboru. Spouštím novou extrakci.")
        except Exception as e:
            print(f"Chyba při načítání výsledků z {output_file_path}: {e}. Spouštím novou extrakci.")

    # Inicializace pipeline
    pipeline = EmbeddedPipeline(
        model_name=model_name,
        provider_name=provider_name,
        embedding_model_name=embedding_model_name,
        embedding_provider_name=embedding_provider_name,
        vectorstore_path=vectorstore_path, # Předání cesty
        text_api_key=text_api_key,
        embedding_api_key=embedding_api_key
    )
    
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    results, timings, token_usages = pipeline.extract_metadata_batch(pdf_files, limit=limit)
    
    # Uložení výsledků
    output_data = {"results": results, "timings": timings, "token_usages": token_usages}
    try:
        output_file_path.parent.mkdir(parents=True, exist_ok=True) # Zajistit existenci adresáře
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Výsledky extrakce (Embedded) včetně tokenů uloženy do {output_file_path}")
    except Exception as e:
        print(f"Chyba při ukládání výsledků do {output_file_path}: {e}")
        
    return results, timings, token_usages


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Spustí Embedded pipeline pro extrakci metadat z PDF souborů.')
    parser.add_argument('--pdf_dir', type=str, default=str(PDF_DIR), help='Cesta k adresáři s PDF soubory')
    parser.add_argument('--limit', type=int, default=None, help='Omezení počtu zpracovaných souborů')
    parser.add_argument('--provider_name', type=str, default=None, help='Název poskytovatele API pro textový model')
    parser.add_argument('--model_name', type=str, default=None, help='Název textového modelu')
    parser.add_argument('--embedding_model', type=str, default=None, help='Název embedding modelu')
    parser.add_argument('--embedding_provider', type=str, default=None, help='Název poskytovatele embedding API')
    parser.add_argument('--vectorstore_path', type=str, default=None, help='Cesta k adresáři pro vectorstore')
    parser.add_argument('--force', action='store_true', help='Vynutit novou extrakci i když výsledky již existují')
    
    args = parser.parse_args()
    
    # <<< Změna: Nastavení výchozího adresáře pro běh, pokud je skript spuštěn samostatně >>>
    from src.config.runtime_config import set_run_results_dir
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run_dir = Path(__file__).resolve().parent.parent.parent / "results" / f"embedded_standalone_{timestamp}"
    set_run_results_dir(default_run_dir)
    # <<< Konec změny >>>
    
    # <<< Změna: Načtení obou API klíčů >>>
    text_api_key_env = os.getenv("ANTHROPIC_API_KEY") if args.provider_name == "anthropic" else os.getenv("OPENAI_API_KEY") if args.provider_name == "openai" else None
    embedding_api_key_env = os.getenv("ANTHROPIC_API_KEY") if args.embedding_provider == "anthropic" else os.getenv("OPENAI_API_KEY") if args.embedding_provider == "openai" else None
    # <<< Konec změny >>>

    extract_metadata_from_pdfs(
        pdf_dir=args.pdf_dir, 
        limit=args.limit, 
        provider_name=args.provider_name,
        model_name=args.model_name,
        embedding_provider_name=args.embedding_provider,
        embedding_model_name=args.embedding_model,
        vectorstore_path=args.vectorstore_path,
        force_extraction=args.force,
        text_api_key=text_api_key_env,
        embedding_api_key=embedding_api_key_env
    ) 