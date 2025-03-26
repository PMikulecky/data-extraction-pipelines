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

4. **Porovnání výsledků**:
   - Textové porovnání s referenčními daty
   - Výpočet skóre podobnosti pro každý typ metadat

5. **Sémantické porovnání** (nová komponenta):
   - Normalizace extrahovaných a referenčních dat
   - Sémantická analýza faktické shody (i při rozdílném formátu)
   - Korekce hodnocení na základě faktické správnosti

6. **Finalizace**:
   - Vykreslení grafů s porovnáním modelů
   - Vytvoření tabulek s výsledky úspěšnosti

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

# Volitelné parametry
python src/main.py --models embedded vlm --limit 10 --skip-download
```

### Parametry příkazové řádky

- `--models`: Seznam modelů k použití (embedded, vlm)
- `--limit`: Omezení počtu zpracovaných souborů
- `--year-filter`: Filtrování článků podle roku vydání
- `--skip-download`: Přeskočí stahování PDF souborů
- `--skip-semantic`: Přeskočí sémantické porovnání výsledků
- `--verbose`: Podrobnější výstup

## Sémantické porovnání

Komponenta sémantického porovnání vylepšuje hodnocení podobnosti mezi extrahovanými a referenčními daty. Standardní porovnání často selhává při odlišném formátování textu, pořadí slov nebo přítomnosti diakritiky.

### Co řeší sémantické porovnání

1. **Různé formáty zápisu autorů** - "Příjmení, Jméno" vs "Jméno Příjmení"
2. **Různé formáty DOI** - s prefixem "https://doi.org/" nebo bez něj
3. **Variace v názvech časopisů** - zkrácené či plné názvy, s/bez pomlček
4. **Odlišné zápisy vydavatelů** - zkratky vs plné názvy
5. **Odlišná klíčová slova** - různé oddělovače, pořadí, diakritika

### Porovnávání autorů pomocí LLM

Nová verze sémantického porovnání umožňuje využít jazykový model (LLM) pro inteligentní porovnávání autorů:

- **Flexibilní porovnání** - umí rozpoznat jména autorů bez ohledu na formát
- **Jazyková inteligence** - správně vyhodnotí diakritiku a jazykové variace
- **Jednoduché použití** - vyžaduje pouze API klíč OpenAI v souboru `.env`

### Použití

```bash
# Spuštění s výchozím nastavením
python src/scripts/run_semantic_comparison.py

# Vypnutí LLM porovnávání (úspora nákladů)
python src/scripts/run_semantic_comparison.py --no-llm

# Použití výkonnějšího modelu
python src/scripts/run_semantic_comparison.py --model gpt-4-turbo

# Specifikace vlastních souborů
python src/scripts/run_semantic_comparison.py --vlm cesta/k/vlm.json --embedded cesta/k/embedded.json
```

### Konfigurace

Pro použití LLM je nutné mít platný API klíč OpenAI v souboru `.env`:

```
OPENAI_API_KEY=váš_api_klíč_zde
```

Další konfigurační možnosti:

- `USE_LLM_FOR_AUTHORS` - (true/false) zapne nebo vypne LLM porovnávání
- `LLM_MODEL` - model ke použití (gpt-3.5-turbo, gpt-4-turbo, atd.)
- `LLM_TEMPERATURE` - teplota při generování (doporučeno 0.0 pro přesné výsledky)

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
│   │   ├── metadata_comparator.py # Nástroje pro porovnání metadat
│   │   └── semantic_comparison.py # Sémantické porovnání extrakcí
│   ├── scripts/
│   │   └── run_semantic_comparison.py # Samostatný skript pro sémantické porovnání
│   └── main.py                   # Hlavní skript
├── results/                      # Výsledky porovnání
├── requirements.txt              # Seznam závislostí
└── README.md                     # Dokumentace projektu
```

## Výstupy

Aplikace generuje následující výstupy:

1. **JSON soubory s výsledky**:
   - `embedded_results.json`, `vlm_results.json`: Extrahovaná metadata
   - `embedded_comparison.json`, `vlm_comparison.json`: Výsledky porovnání s referencemi
   - `embedded_comparison_semantic.json`, `vlm_comparison_semantic.json`: Výsledky po sémantickém porovnání
   - `semantic_comparison_results.json`: Souhrnné výsledky sémantického porovnání

2. **Vizualizace**:
   - `comparison_results.png`: Graf porovnání úspěšnosti modelů pro jednotlivá pole metadat
   - `overall_results.png`: Graf celkové úspěšnosti modelů

3. **CSV tabulky**:
   - `overall_results.csv`: Tabulka s celkovými výsledky
   - `detailed_results.csv`: Detailní výsledky pro každý dokument a pole
   - `summary_results.csv`: Souhrnné statistiky pro každé pole 