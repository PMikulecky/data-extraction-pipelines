#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modul pro analýzu PDF souborů akademických prací.
Identifikuje titulní stranu, hlavní text a strany s referencemi.
"""

import os
import re
import PyPDF2
from pathlib import Path
from pdf2image import convert_from_path
import numpy as np
from collections import Counter


class PDFAnalyzer:
    """
    Třída pro analýzu PDF souborů akademických prací.
    """
    
    # Klíčová slova pro identifikaci referencí
    REFERENCE_KEYWORDS = [
        'references', 'bibliography', 'literature cited', 'works cited',
        'reference list', 'cited literature', 'literatura', 'reference',
        'bibliografie', 'citovaná literatura', 'seznam literatury'
    ]
    
    # Klíčová slova pro identifikaci abstraktu
    ABSTRACT_KEYWORDS = [
        'abstract', 'summary', 'abstrakt', 'souhrn', 'shrnutí'
    ]
    
    def __init__(self, pdf_path):
        """
        Inicializace analyzátoru PDF.
        
        Args:
            pdf_path (str): Cesta k PDF souboru
        """
        self.pdf_path = pdf_path
        self.num_pages = 0
        self.text_by_page = []
        self.title_page = 0
        self.abstract_page = 0
        self.reference_start_page = 0
        self.main_text_pages = []
        self.images = []
        
        # Načtení PDF
        self._load_pdf()
    
    def _load_pdf(self):
        """
        Načte PDF soubor a extrahuje text z každé stránky.
        """
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                self.num_pages = len(reader.pages)
                
                # Extrakce textu z každé stránky
                for i in range(self.num_pages):
                    page = reader.pages[i]
                    text = page.extract_text()
                    self.text_by_page.append(text if text else "")
                
                # Konverze stránek na obrázky
                self.images = convert_from_path(self.pdf_path)
                
                print(f"PDF soubor {self.pdf_path} úspěšně načten. Počet stránek: {self.num_pages}")
        except Exception as e:
            print(f"Chyba při načítání PDF souboru {self.pdf_path}: {e}")
            raise
    
    def analyze(self):
        """
        Analyzuje PDF soubor a identifikuje klíčové části.
        
        Returns:
            dict: Slovník s informacemi o struktuře dokumentu
        """
        # Identifikace titulní strany
        self.identify_title_page()
        
        # Identifikace abstraktu
        self.identify_abstract_page()
        
        # Identifikace stran s referencemi
        self.identify_reference_pages()
        
        # Identifikace hlavního textu
        self.identify_main_text_pages()
        
        return {
            'title_page': self.title_page,
            'abstract_page': self.abstract_page,
            'reference_start_page': self.reference_start_page,
            'main_text_pages': self.main_text_pages,
            'num_pages': self.num_pages
        }
    
    def identify_title_page(self):
        """
        Identifikuje titulní stranu dokumentu.
        Obvykle je to první strana s nejmenším množstvím textu.
        """
        # Předpokládáme, že titulní strana je první strana
        self.title_page = 0
        
        # Pokud je více stránek, zkontrolujeme, zda první strana obsahuje málo textu
        if self.num_pages > 1:
            text_lengths = [len(text) for text in self.text_by_page[:min(3, self.num_pages)]]
            if text_lengths[0] > 0 and (text_lengths[0] < text_lengths[1] / 2 or text_lengths[0] < 200):
                self.title_page = 0
            else:
                # Hledáme stranu s nejmenším množstvím textu z prvních 3 stran
                min_text_page = np.argmin(text_lengths)
                if text_lengths[min_text_page] > 0:
                    self.title_page = min_text_page
    
    def identify_abstract_page(self):
        """
        Identifikuje stranu s abstraktem.
        """
        # Hledáme klíčová slova pro abstrakt v prvních několika stránkách
        for i in range(min(5, self.num_pages)):
            text = self.text_by_page[i].lower()
            for keyword in self.ABSTRACT_KEYWORDS:
                if keyword in text:
                    self.abstract_page = i
                    return
        
        # Pokud nebyl nalezen abstrakt, předpokládáme, že je na stejné straně jako titul
        # nebo na následující straně
        self.abstract_page = min(self.title_page + 1, self.num_pages - 1)
    
    def identify_reference_pages(self):
        """
        Identifikuje strany s referencemi.
        """
        # Hledáme klíčová slova pro reference
        for i in range(self.num_pages):
            text = self.text_by_page[i].lower()
            
            # Hledáme nadpis referencí
            for keyword in self.REFERENCE_KEYWORDS:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text):
                    # Kontrola, zda je to skutečně nadpis referencí (na začátku řádku nebo samostatně)
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and keyword in line.lower() and len(line) < 50:
                            self.reference_start_page = i
                            return
        
        # Pokud nebyly nalezeny reference, předpokládáme, že jsou na poslední straně
        self.reference_start_page = max(0, self.num_pages - 1)
    
    def identify_main_text_pages(self):
        """
        Identifikuje strany s hlavním textem.
        """
        # Hlavní text je mezi abstraktem a referencemi
        start_page = self.abstract_page + 1
        end_page = self.reference_start_page - 1 if self.reference_start_page > 0 else self.num_pages - 1
        
        # Kontrola, zda je rozsah platný
        if start_page <= end_page:
            self.main_text_pages = list(range(start_page, end_page + 1))
        else:
            # Pokud není platný rozsah, použijeme všechny stránky kromě titulní a referencí
            self.main_text_pages = [i for i in range(self.num_pages) 
                                   if i != self.title_page and i < self.reference_start_page]
    
    def get_title_page_image(self):
        """
        Vrátí obrázek titulní strany.
        
        Returns:
            PIL.Image: Obrázek titulní strany
        """
        if self.images and self.title_page < len(self.images):
            return self.images[self.title_page]
        return None
    
    def get_abstract_page_image(self):
        """
        Vrátí obrázek strany s abstraktem.
        
        Returns:
            PIL.Image: Obrázek strany s abstraktem
        """
        if self.images and self.abstract_page < len(self.images):
            return self.images[self.abstract_page]
        return None
    
    def get_reference_page_images(self):
        """
        Vrátí obrázky stran s referencemi.
        
        Returns:
            list: Seznam obrázků stran s referencemi
        """
        if not self.images:
            return []
        
        # Vrátíme všechny stránky od začátku referencí do konce
        return self.images[self.reference_start_page:] if self.reference_start_page < len(self.images) else []
    
    def get_main_text_page_images(self):
        """
        Vrátí obrázky stran s hlavním textem.
        
        Returns:
            list: Seznam obrázků stran s hlavním textem
        """
        if not self.images:
            return []
        
        return [self.images[i] for i in self.main_text_pages if i < len(self.images)]
    
    def get_title_page_text(self):
        """
        Vrátí text titulní strany.
        
        Returns:
            str: Text titulní strany
        """
        if self.title_page < len(self.text_by_page):
            return self.text_by_page[self.title_page]
        return ""
    
    def get_abstract_text(self):
        """
        Vrátí text abstraktu.
        
        Returns:
            str: Text abstraktu
        """
        if self.abstract_page < len(self.text_by_page):
            return self.text_by_page[self.abstract_page]
        return ""
    
    def get_reference_text(self):
        """
        Vrátí text referencí.
        
        Returns:
            str: Text referencí
        """
        if self.reference_start_page < len(self.text_by_page):
            # Spojíme text všech stran s referencemi
            return "\n".join(self.text_by_page[self.reference_start_page:])
        return ""
    
    def get_main_text(self):
        """
        Vrátí hlavní text dokumentu.
        
        Returns:
            str: Hlavní text dokumentu
        """
        return "\n".join([self.text_by_page[i] for i in self.main_text_pages 
                         if i < len(self.text_by_page)])
    
    def get_full_text(self):
        """
        Vrátí celý text dokumentu.
        
        Returns:
            str: Celý text dokumentu
        """
        return "\n".join(self.text_by_page)


def analyze_pdf(pdf_path):
    """
    Analyzuje PDF soubor a vrátí informace o jeho struktuře.
    
    Args:
        pdf_path (str): Cesta k PDF souboru
        
    Returns:
        dict: Slovník s informacemi o struktuře dokumentu
    """
    analyzer = PDFAnalyzer(pdf_path)
    return analyzer.analyze()


if __name__ == "__main__":
    # Příklad použití
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if os.path.exists(pdf_path):
            try:
                result = analyze_pdf(pdf_path)
                print(f"Analýza PDF souboru {pdf_path}:")
                print(f"Počet stránek: {result['num_pages']}")
                print(f"Titulní strana: {result['title_page'] + 1}")
                print(f"Strana s abstraktem: {result['abstract_page'] + 1}")
                print(f"Začátek referencí: {result['reference_start_page'] + 1}")
                print(f"Hlavní text: strany {[p + 1 for p in result['main_text_pages']]}")
            except Exception as e:
                print(f"Chyba při analýze PDF souboru: {e}")
        else:
            print(f"Soubor {pdf_path} neexistuje.")
    else:
        print("Použití: python pdf_analyzer.py <cesta_k_pdf>") 