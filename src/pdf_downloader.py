#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro stahování PDF souborů akademických prací pomocí DOI.
"""

import os
import time
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
from urllib.parse import quote_plus
from dotenv import load_dotenv
import random

# Načtení proměnných prostředí
load_dotenv()

# Definice cest
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FILTERED_CSV = DATA_DIR / "papers-filtered.csv"
PDF_DIR = DATA_DIR / "pdfs"

# Vytvoření adresáře pro PDF soubory, pokud neexistuje
PDF_DIR.mkdir(parents=True, exist_ok=True)

# API endpointy pro získání PDF
UNPAYWALL_API_URL = "https://api.unpaywall.org/v2/{doi}?email={email}"
CROSSREF_API_URL = "https://api.crossref.org/works/{doi}"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/v1/paper/{doi}"


def get_pdf_url_from_unpaywall(doi, email):
    """
    Získá URL PDF souboru z Unpaywall API.
    
    Args:
        doi (str): DOI akademické práce
        email (str): Email pro API
        
    Returns:
        str: URL PDF souboru nebo None, pokud není k dispozici
    """
    try:
        url = UNPAYWALL_API_URL.format(doi=quote_plus(doi), email=email)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9,cs;q=0.8',
            'Connection': 'keep-alive',
            'Referer': 'https://scholar.google.com/',
            'DNT': '1',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'cross-site',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Kontrola, zda je k dispozici open access PDF
            if data.get('is_oa') and data.get('best_oa_location'):
                pdf_url = data['best_oa_location'].get('url_for_pdf')
                if pdf_url:
                    return pdf_url
                
                # Pokud není k dispozici přímý odkaz na PDF, zkusíme použít URL článku
                html_url = data['best_oa_location'].get('url')
                if html_url and html_url.endswith('.pdf'):
                    return html_url
        
        return None
    except Exception as e:
        print(f"Chyba při získávání PDF URL z Unpaywall pro DOI {doi}: {e}")
        return None


def get_pdf_url_from_crossref(doi):
    """
    Získá URL PDF souboru z Crossref API.
    
    Args:
        doi (str): DOI akademické práce
        
    Returns:
        str: URL PDF souboru nebo None, pokud není k dispozici
    """
    try:
        url = CROSSREF_API_URL.format(doi=quote_plus(doi))
        email = os.getenv('UNPAYWALL_EMAIL', 'user@example.com')
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9,cs;q=0.8',
            'Connection': 'keep-alive',
            'Referer': 'https://scholar.google.com/',
            'DNT': '1',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'cross-site',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'mailto': email
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Kontrola, zda je k dispozici odkaz na PDF
            if 'message' in data and 'link' in data['message']:
                for link in data['message']['link']:
                    if link.get('content-type') == 'application/pdf':
                        return link.get('URL')
        
        return None
    except Exception as e:
        print(f"Chyba při získávání PDF URL z Crossref pro DOI {doi}: {e}")
        return None


def get_pdf_url_from_semantic_scholar(doi):
    """
    Získá URL PDF souboru z Semantic Scholar API.
    
    Args:
        doi (str): DOI akademické práce
        
    Returns:
        str: URL PDF souboru nebo None, pokud není k dispozici
    """
    try:
        url = SEMANTIC_SCHOLAR_API_URL.format(doi=quote_plus(doi))
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9,cs;q=0.8',
            'Connection': 'keep-alive',
            'Referer': 'https://scholar.google.com/',
            'DNT': '1',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'cross-site',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Kontrola, zda je k dispozici odkaz na PDF
            if 'openAccessPdf' in data and data['openAccessPdf']:
                return data['openAccessPdf'].get('url')
        
        return None
    except Exception as e:
        print(f"Chyba při získávání PDF URL z Semantic Scholar pro DOI {doi}: {e}")
        return None


def get_pdf_url_from_direct_doi(doi):
    """
    Pokusí se získat URL PDF souboru přímo z DOI URL.
    
    Args:
        doi (str): DOI akademické práce
        
    Returns:
        str: URL PDF souboru nebo None, pokud není k dispozici
    """
    try:
        # Vytvoření DOI URL
        doi_url = f"https://doi.org/{doi}"
        
        # Nastavení hlaviček pro simulaci prohlížeče
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9,cs;q=0.8',
            'Connection': 'keep-alive',
            'Referer': 'https://scholar.google.com/',
            'DNT': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
        }
        
        # Získání přesměrování z DOI URL
        session = requests.Session()
        response = session.head(doi_url, headers=headers, allow_redirects=True)
        
        # Kontrola, zda URL končí na .pdf
        final_url = response.url
        if final_url.endswith('.pdf'):
            return final_url
        
        # Pokud URL nekončí na .pdf, zkusíme přidat .pdf na konec
        pdf_url = final_url
        if not pdf_url.endswith('/'):
            pdf_url += '/'
        pdf_url += 'pdf'
        
        # Zkusíme, zda existuje URL s /pdf na konci
        pdf_response = session.head(pdf_url, headers=headers, allow_redirects=True)
        if pdf_response.status_code == 200:
            return pdf_url
        
        # Zkusíme ještě alternativní cestu k PDF
        if not final_url.endswith('/'):
            alt_pdf_url = final_url + '/download'
            alt_pdf_response = session.head(alt_pdf_url, headers=headers, allow_redirects=True)
            if alt_pdf_response.status_code == 200:
                return alt_pdf_url
        
        return None
    except Exception as e:
        print(f"Chyba při získávání PDF URL přímo z DOI {doi}: {e}")
        return None


def download_pdf(url, output_path):
    """
    Stáhne PDF soubor z URL a uloží ho do zadané cesty.
    
    Args:
        url (str): URL PDF souboru
        output_path (str): Cesta pro uložení PDF souboru
        
    Returns:
        bool: True, pokud bylo stahování úspěšné, jinak False
    """
    try:
        # Vytvoření session pro zachování cookies
        session = requests.Session()
        
        # Nastavení hlaviček pro simulaci prohlížeče
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9,cs;q=0.8',
            'Connection': 'keep-alive',
            'Referer': 'https://scholar.google.com/',
            'DNT': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
        }
        
        # Nejprve navštívíme stránku, abychom získali cookies
        session.get(url.split('/pdf')[0] if '/pdf' in url else url.rsplit('/', 1)[0], headers=headers, timeout=30)
        
        # Poté stáhneme PDF
        pdf_headers = headers.copy()
        pdf_headers['Accept'] = 'application/pdf,application/x-pdf,application/octet-stream,*/*'
        
        # Přidání náhodného zpoždění před stažením (simulace lidského chování)
        time.sleep(1 + random.random() * 2)
        
        response = session.get(url, stream=True, headers=pdf_headers, timeout=30)
        
        # Pokud dostaneme 403, zkusíme jiný přístup
        if response.status_code == 403:
            print(f"Chyba 403 při stahování PDF z {url}, zkouším alternativní přístup...")
            
            # Zkusíme přidat další hlavičky, které mohou pomoci
            enhanced_headers = pdf_headers.copy()
            enhanced_headers['Accept-Encoding'] = 'gzip, deflate, br'
            enhanced_headers['Origin'] = url.split('/')[0] + '//' + url.split('/')[2]
            enhanced_headers['Host'] = url.split('/')[2]
            
            # Zkusíme znovu s vylepšenými hlavičkami
            time.sleep(2 + random.random() * 3)  # Delší zpoždění před opakováním
            response = session.get(url, stream=True, headers=enhanced_headers, timeout=30)
        
        if response.status_code == 200:
            # Kontrola, zda je obsah skutečně PDF
            content_type = response.headers.get('Content-Type', '')
            
            # Kontrola, zda je obsah PDF podle Content-Type nebo podle prvních bytů
            is_pdf = 'application/pdf' in content_type
            
            # Kontrola prvních bytů souboru (PDF začíná sekvencí %PDF-)
            first_bytes = response.content[:5]
            is_pdf_by_content = first_bytes == b'%PDF-'
            
            if not is_pdf and not is_pdf_by_content:
                if 'text/html' in content_type:
                    print(f"Chyba: URL {url} vrátila HTML stránku místo PDF")
                    # Kontrola, zda HTML obsahuje chybové hlášení o přístupu
                    html_content = response.text.lower()
                    if 'access denied' in html_content or 'not authorized' in html_content or 'subscription required' in html_content or 'paywall' in html_content:
                        print(f"Chyba: Přístup k PDF je omezen - vyžaduje předplatné nebo institucionální přístup")
                    return False
                else:
                    print(f"Varování: Obsah na URL {url} nemusí být PDF (Content-Type: {content_type})")
            
            # Stažení a uložení souboru pouze pokud je to PDF
            if is_pdf or is_pdf_by_content or url.endswith('.pdf'):
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Ověření, zda je stažený soubor skutečně PDF
                try:
                    with open(output_path, 'rb') as f:
                        header = f.read(5)
                        if header != b'%PDF-':
                            print(f"Chyba: Stažený soubor není validní PDF (chybí hlavička PDF)")
                            os.remove(output_path)  # Odstranění nevalidního souboru
                            return False
                except Exception as e:
                    print(f"Chyba při ověřování PDF souboru: {e}")
                    os.remove(output_path)  # Odstranění nevalidního souboru
                    return False
                
                return True
            else:
                print(f"Chyba: Obsah na URL {url} není PDF")
                return False
        else:
            print(f"Chyba při stahování PDF: HTTP status {response.status_code}")
            return False
    except Exception as e:
        print(f"Chyba při stahování PDF z {url}: {e}")
        return False


def download_pdfs_for_filtered_papers(csv_path, pdf_dir, limit=None):
    """
    Stáhne PDF soubory pro filtrované akademické práce.
    
    Args:
        csv_path (str): Cesta k CSV souboru s filtrovanými daty
        pdf_dir (str): Adresář pro uložení PDF souborů
        limit (int, optional): Omezení počtu stahovaných souborů
        
    Returns:
        tuple: (počet úspěšně stažených souborů, celkový počet zpracovaných záznamů)
    """
    # Načtení emailu pro Unpaywall API
    email = os.getenv('UNPAYWALL_EMAIL', 'user@example.com')
    
    # Načtení dat
    print(f"Načítám filtrovaná data z {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Kontrola, zda existuje sloupec s rokem vydání
    year_column = None
    for col in df.columns:
        if 'year' in col.lower() or 'date' in col.lower():
            year_column = col
            break
    
    # Pokud existuje sloupec s rokem, seřadíme podle něj od nejnovějších po nejstarší
    if year_column:
        print(f"Řadím práce podle sloupce '{year_column}' od nejnovějších po nejstarší...")
        
        # Převod roku na číselný formát, pokud je to možné
        try:
            # Pokud sloupec obsahuje celé datum, extrahujeme rok
            if df[year_column].dtype == 'object':
                df['year_numeric'] = df[year_column].str.extract(r'(\d{4})', expand=False).astype(float)
            else:
                df['year_numeric'] = df[year_column].astype(float)
            
            # Seřazení podle roku sestupně (od nejnovějších)
            df = df.sort_values(by='year_numeric', ascending=False)
            
            # Odstranění pomocného sloupce
            df = df.drop('year_numeric', axis=1)
        except Exception as e:
            print(f"Varování: Nepodařilo se seřadit podle roku: {e}")
    else:
        print("Varování: Nenalezen sloupec s rokem vydání, práce nebudou seřazeny podle roku.")
    
    # Omezení počtu záznamů, pokud je zadáno
    if limit:
        df = df.head(limit)
    
    # Statistiky
    total_papers = len(df)
    successful_downloads = 0
    
    print(f"Začínám stahování PDF souborů pro {total_papers} akademických prací...")
    
    # Iterace přes záznamy
    for index, row in tqdm(df.iterrows(), total=total_papers, desc="Stahování PDF"):
        doi = row['dc.identifier.doi']
        paper_id = row['id']
        
        # Výpis informací o aktuální práci včetně roku, pokud je k dispozici
        if year_column:
            year_info = row[year_column]
            print(f"Zpracovávám práci ID {paper_id} (DOI: {doi}, Rok: {year_info})")
        else:
            print(f"Zpracovávám práci ID {paper_id} (DOI: {doi})")
        
        # Cesta pro uložení PDF
        pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
        
        # Přeskočení, pokud soubor již existuje
        if os.path.exists(pdf_path):
            print(f"PDF pro ID {paper_id} (DOI: {doi}) již existuje, přeskakuji.")
            successful_downloads += 1
            continue
        
        # Získání URL PDF z různých zdrojů
        pdf_url = None
        
        # 1. Zkusíme Unpaywall
        pdf_url = get_pdf_url_from_unpaywall(doi, email)
        
        # 2. Pokud Unpaywall selže, zkusíme Crossref
        if not pdf_url:
            pdf_url = get_pdf_url_from_crossref(doi)
        
        # 3. Pokud Crossref selže, zkusíme Semantic Scholar
        if not pdf_url:
            pdf_url = get_pdf_url_from_semantic_scholar(doi)
        
        # 4. Pokud Semantic Scholar selže, zkusíme přímo z DOI URL
        if not pdf_url:
            pdf_url = get_pdf_url_from_direct_doi(doi)
        
        # Stažení PDF, pokud je URL k dispozici
        if pdf_url:
            print(f"Stahuji PDF pro ID {paper_id} (DOI: {doi}) z URL: {pdf_url}")
            if download_pdf(pdf_url, pdf_path):
                successful_downloads += 1
                print(f"PDF úspěšně staženo a uloženo do {pdf_path}")
            else:
                print(f"Nepodařilo se stáhnout PDF pro ID {paper_id} (DOI: {doi})")
        else:
            print(f"Nepodařilo se najít URL PDF pro ID {paper_id} (DOI: {doi})")
        
        # Krátká pauza mezi požadavky, abychom nepřetížili API
        time.sleep(1)
    
    return successful_downloads, total_papers


def main():
    """
    Hlavní funkce pro stahování PDF souborů.
    """
    try:
        # Kontrola, zda existuje filtrovaný CSV soubor
        if not os.path.exists(FILTERED_CSV):
            print(f"Filtrovaný CSV soubor {FILTERED_CSV} nebyl nalezen. Nejprve spusťte data_preparation.py.")
            return
        
        # Stažení PDF souborů
        successful, total = download_pdfs_for_filtered_papers(FILTERED_CSV, PDF_DIR)
        
        print(f"\nStahování dokončeno. Úspěšně staženo {successful} z {total} PDF souborů.")
    except Exception as e:
        print(f"Chyba při stahování PDF souborů: {e}")


if __name__ == "__main__":
    main() 