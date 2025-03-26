#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Skript pro kontrolu nainstalovaných balíčků z requirements.txt
"""

import importlib
import subprocess
import sys

def check_requirements(requirements_file):
    """
    Kontroluje, zda jsou všechny balíčky z requirements.txt nainstalovány.
    
    Args:
        requirements_file (str): Cesta k souboru requirements.txt
    """
    # Načtení požadavků z requirements.txt
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Získání seznamu nainstalovaných balíčků
    try:
        installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'list']).decode('utf-8')
    except subprocess.CalledProcessError:
        print("Nepodařilo se získat seznam nainstalovaných balíčků.")
        return
    
    missing = []
    installed = []
    
    for requirement in requirements:
        # Extrakce názvu balíčku (bez verze)
        package_name = requirement.split('>=')[0].split('==')[0].strip()
        
        # Kontrola, zda je balíček nainstalován
        if package_name.lower() in installed_packages.lower():
            installed.append(package_name)
        else:
            # Zkusíme importovat balíček jako alternativní kontrolu
            try:
                importlib.import_module(package_name)
                installed.append(package_name)
            except ImportError:
                missing.append(requirement)
    
    if missing:
        print(f"Chybějící balíčky: {missing}")
    else:
        print("Všechny balíčky jsou nainstalovány.")
    
    print(f"\nNainstalované balíčky ({len(installed)}/{len(requirements)}):")
    for package in installed:
        print(f"- {package}")

if __name__ == "__main__":
    check_requirements("requirements.txt") 