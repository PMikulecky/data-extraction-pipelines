# Extrakce metadat z akademických PDF pomocí AI

Tento projekt porovnává úspěšnost různých AI modelů (LLM a VLM) v extrakci metadat z PDF souborů akademických prací.

## Popis projektu

Projekt zahrnuje následující komponenty:

1. **Příprava dat**:
   - Načtení seznamu akademických prací z `papers.csv`
   - Filtrace prací s validním DOI
   - Vytvoření `papers-filtered.csv` s použitelnými záznamy
   - Stažení PDF souborů pomocí API
   - Sestavení referenčních dat pro porovnání

2. **Inicializace**:
   - Nastavení API modelů
   - Definice metadat pro porovnání
   - Analýza dokumentů pro identifikaci titulní strany, hlavního textu a referencí

3. **Extrakce dat pomocí AI**:
   - Embedded pipeline: Text → Chunking → Vektorová databáze → Dotazy
   - VLM metoda: Zpracování obrázků z relevantních stran dokumentu

4. **Finalizace**:
   - Porovnání výsledků s referenčními daty
   - Vytvoření tabulky s výsledky úspěšnosti modelů

## Instalace a požadavky

```bash
# Klonování repozitáře
git clone <repository-url>
cd metadata-extraction-ai

# Instalace závislostí
pip install -r requirements.txt
```

## Použití

```bash
# Příprava dat
python src/data_preparation.py

# Extrakce metadat a vyhodnocení
python src/main.py
```

## Struktura projektu

```
metadata-extraction-ai/
├── data/
│   ├── papers.csv                # Vstupní data
│   ├── papers-filtered.csv       # Filtrovaná data
│   └── pdfs/                     # Adresář pro stažené PDF soubory
├── src/
│   ├── data_preparation.py       # Skript pro přípravu dat
│   ├── pdf_downloader.py         # Skript pro stahování PDF
│   ├── models/
│   │   ├── embedded_pipeline.py  # Implementace Embedded pipeline
│   │   └── vlm_pipeline.py       # Implementace VLM pipeline
│   ├── utils/
│   │   ├── pdf_analyzer.py       # Nástroje pro analýzu PDF
│   │   └── metadata_comparator.py # Nástroje pro porovnání metadat
│   └── main.py                   # Hlavní skript
├── results/                      # Výsledky porovnání
├── requirements.txt              # Seznam závislostí
└── README.md                     # Dokumentace projektu
``` 