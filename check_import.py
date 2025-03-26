#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skript pro kontrolu importu balíčku langchain_community
"""

try:
    import langchain_community
    print("langchain_community je úspěšně nainstalován")
    print(f"Cesta k balíčku: {langchain_community.__file__}")
except ImportError as e:
    print(f"Chyba při importu: {e}")

try:
    from langchain.vectorstores import FAISS
    print("Import FAISS z langchain.vectorstores je úspěšný")
    print(f"Cesta k modulu: {FAISS.__module__}")
except ImportError as e:
    print(f"Chyba při importu FAISS z langchain.vectorstores: {e}")
    
try:
    from langchain_community.vectorstores import FAISS
    print("Import FAISS z langchain_community.vectorstores je úspěšný")
except ImportError as e:
    print(f"Chyba při importu FAISS z langchain_community.vectorstores: {e}") 