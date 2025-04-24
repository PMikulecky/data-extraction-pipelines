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

# Instalace závislostí pro vektorovou databázi (pro Embedded pipeline)
pip install langchain-chroma chromadb
```

### Požadavky

Projekt vyžaduje následující hlavní knihovny:

- langchain - pro zpracování textu a vektorovou databázi
- langchain-chroma, chromadb - pro vektorovou databázi
- openai, anthropic - pro připojení k API modelů
- PyPDF2 - pro extrakci textu z PDF souborů
- matplotlib, pandas - pro analýzu a vizualizaci výsledků

Kompletní seznam závislostí je v souboru `requirements.txt`.

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
- `--force-extraction`: Vynutí novou extrakci i když výsledky již existují

### Hromadné spuštění s různými konfiguracemi modelů

Pro snadné porovnání různých modelů Ollama lze použít skript pro hromadné spuštění:

```bash
# Spuštění s výchozími konfiguracemi modelů
python src/run_all_models.py

# Použití vlastního konfiguračního souboru
python src/run_all_models.py --config config/model_configs.json

# Omezení počtu zpracovaných souborů
python src/run_all_models.py --limit 5 --skip-download
```

Konfigurační soubor obsahuje seznam různých konfigurací modelů ve formátu:

```json
[
  {
    "name": "konfigurace-1",
    "text": {
      "provider": "ollama",
      "model": "llama3.1:8b"
    },
    "vision": {
      "provider": "ollama",
      "model": "llama3.2-vision:11b"
    },
    "embedding": {
      "provider": "ollama",
      "model": "mxbai-embed-large:335m"
    }
  },
  {
    "name": "konfigurace-2",
    ...
  }
]
```

Výsledky pro každou konfiguraci budou uloženy v samostatném adresáři v rámci složky `results/`.

## Embedded Pipeline

Embedded Pipeline zpracovává PDF dokumenty následujícím způsobem:

1. **Extrakce textu**: Text je extrahován z PDF souboru.
2. **Chunking**: Text je rozdělen na menší části (chunky) s překryvem pro zachování kontextu.
3. **Vektorizace**: Pro každý PDF dokument je vytvořen samostatný vectorstore.
   - Každý chunk je převeden na vektorovou reprezentaci pomocí embedding modelu.
   - Chunky jsou uloženy ve vectorstore s metadaty pro snadné vyhledávání.
4. **Dotazování**: Pro extrakci metadat se provádí sémantické vyhledávání:
   - Pro každý typ metadat (titul, autoři, atd.) je vytvořen specifický dotaz.
   - Vyhledání 5 nejvíce relevantních chunků v dokumentu.
   - Spojení těchto chunků do kontextu pro jazykový model.
5. **Extrakce**: LLM model extrahuje požadovaná metadata z poskytnutého kontextu.

Vectorstore pro každý dokument je vytvářen v adresáři `vectorstore/embedded_pipeline` s náhodně generovaným ID, což zajišťuje izolaci dat a nezávislé zpracování dokumentů.

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
│   ├── run_all_models.py         # Skript pro hromadné spuštění s různými konfiguracemi
│   └── main.py                   # Hlavní skript
├── config/
│   ├── models.json               # Aktuální konfigurace modelů
│   └── model_configs.json        # Konfigurace pro hromadné spuštění
├── results/                      # Výsledky porovnání
├── vectorstore/                  # Adresář pro vektorové databáze
│   └── embedded_pipeline/        # Vektorové databáze pro Embedded pipeline
├── requirements.txt              # Seznam závislostí
└── README.md                     # Dokumentace projektu
```

## Výstupy

Aplikace generuje následující výstupy v adresáři `results/`:

1. **JSON soubory s výsledky**:
   - `[model]_results.json`: Extrahovaná metadata pro daný model (např. `embedded_results.json`).
   - `[model]_comparison.json`: Výsledky porovnání s referencemi pro daný model.
   - `[model]_comparison_semantic.json`: Výsledky po sémantickém porovnání (pokud bylo spuštěno).
   - `semantic_comparison_results.json`: Souhrnné výsledky sémantického porovnání (pokud bylo spuštěno samostatně).

2. **Vizualizace**:
   - `comparison_results.png`: Sloupcový graf porovnání úspěšnosti modelů pro jednotlivá pole metadat. **Nově obsahuje chybové úsečky (±1σ)**.
   - `overall_results.png`: Sloupcový graf celkové úspěšnosti modelů. **Nově obsahuje chybové úsečky (±1σ)**.
   - **`comparison_results_boxplot.png`**: **Nový box plot** zobrazující distribuci skóre podobnosti pro jednotlivá pole a modely.
   - **`overall_results_boxplot.png`**: **Nový box plot** zobrazující distribuci celkového skóre podobnosti pro jednotlivé modely.

3. **CSV tabulky**:
   - `summary_results.csv`: Souhrnné statistiky (průměr, směrodatná odchylka) pro každé pole a model.
   - `overall_summary_results.csv`: Celkové souhrnné statistiky (průměr, směrodatná odchylka) pro každý model.
   - `detailed_scores_all.csv`: Detailní skóre podobnosti pro každý dokument, pole a model, použité pro generování box plotů a výpočet statistik.
   - `detailed_results.csv`: *Původní* detailní výsledky (může mít jinou strukturu).
   - `overall_results.csv`: *Původní* celkové výsledky (může mít jinou strukturu). 