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
   - Text pipeline: Přímé zpracování textu LLM modelem
   - VLM metoda: Zpracování obrázků z relevantních stran dokumentu
   - Hybridní pipeline: Kombinace výsledků Text a VLM pipeline

4. **Porovnání výsledků**:
   - Textové porovnání s referenčními daty
   - Výpočet skóre podobnosti pro každý typ metadat
   - **Poznámka:** Pole, u kterých chybí referenční hodnota (např. `NaN`), jsou nyní ignorována při výpočtu celkové podobnosti (`overall_similarity`) pro daný dokument. Tím se zabrání zkreslení skóre nulovými hodnotami. Pokud dokument nemá žádná porovnatelná pole, jeho `overall_similarity` bude `None`.

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
python src/main.py --models embedded vlm text hybrid --limit 10 --skip-download
```

### Parametry příkazové řádky

- `--models`: Seznam modelů k použití (embedded, vlm, text, hybrid)
- `--limit`: Omezení počtu zpracovaných souborů
- `--year-filter`: Filtrování článků podle roku vydání
- `--skip-download`: Přeskočí stahování PDF souborů
- `--skip-semantic`: Přeskočí sémantické porovnání výsledků
- `--verbose`: Podrobnější výstup
- `--force-extraction`: Vynutí novou extrakci i když výsledky již existují
- `--compare-only`: Porovná pouze výsledky bez provádění nové extrakce (lze specifikovat typ modelu)

### Hromadné spuštění s různými konfiguracemi modelů

Pro snadné porovnání různých modelů lze použít skript pro hromadné spuštění:

```bash
# Spuštění s výchozími konfiguracemi modelů
python -m src.run_all_models

# Použití vlastního konfiguračního souboru
python -m src.run_all_models --config config/model_configs.json

# Omezení počtu zpracovaných souborů
python -m src.run_all_models --limit 5 --skip-download

# Generování pouze hybridních výsledků pro existující výsledky
python -m src.run_all_models --combine-only --results-dir "cesta/k/adresáři/s/výsledky"
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

### Generování a regenerace grafů

Po dokončení hromadného spuštění jsou automaticky generovány dva druhy grafů v adresáři `final_comparison/`:

1. **Grafy typů pipeline** (s příponou `-pipelines`):
   - Porovnávají výsledky podle typu pipeline (EMBEDDED, TEXT, VLM, HYBRID)
   - Průměrují výsledky ze stejných typů pipeline napříč všemi konfiguracemi

2. **Grafy konkrétních modelů** (s příponou `-models`):
   - Porovnávají výsledky jednotlivých modelů (např. llama3:8b, gpt-4o, mxbai-embed-large:335m)
   - Barevně rozlišují typy modelů (modrá = text, zelená = embedding, červená = vision, oranžová = hybrid)
   - Umožňují přímé porovnání úspěšnosti konkrétních modelů bez ohledu na konfiguraci

Pro regeneraci grafů z existujících výsledků (např. po aktualizaci kódu) lze použít:

```bash
# Regenerace grafů z existujících výsledků
python -m src.run_all_models --graphs-only --results-dir "cesta/k/adresáři/s/výsledky"

# Alternativně použít nový skript pro generování hybridních výsledků a grafů
python -m src.generate_all_hybrid_results --base-dir "cesta/k/adresáři/s/konfiguracemi" --verbose
```

Tímto způsobem můžete vytvořit aktualizované grafy bez nutnosti znovu spouštět celou extrakci dat.

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

## Hybridní Pipeline

Hybridní pipeline kombinuje výsledky z Text a VLM pipeline pro dosažení lepších výsledků extrakce dat. Princip je následující:

1. **Zpracování pomocí Text pipeline**: Extrakce metadat pomocí textové pipeline.
2. **Zpracování pomocí VLM pipeline**: Extrakce metadat pomocí VLM (Vision Language Model) pipeline.
3. **Kombinace výsledků**:
   - Z VLM pipeline se preferenčně načítají následující metadata: title, authors, doi, issue, volume, journal, publisher, year
   - Z Text pipeline se načítají: abstract, keywords
   - Z Text pipeline se také doplňují jakákoliv pole, která se nepodařilo načíst z VLM pipeline

Tento přístup využívá silných stránek obou metod - VLM pipeline je lepší v rozpoznávání strukturovaných dat na titulní straně dokumentu, zatímco textová pipeline lépe zpracovává souvislý text abstraktu a klíčových slov.

### Použití Hybridní Pipeline

Hybridní pipeline lze spustit několika způsoby:

```bash
# Jako součást běžného spuštění
python -m src.main --models text vlm hybrid

# Pouze kombinace existujících výsledků
python -m src.run_all_models --combine-only --results-dir "cesta/k/adresáři/s/výsledky"

# V rámci hromadného vyhodnocení
python -m src.generate_all_hybrid_results --base-dir "cesta/k/adresáři/se/všemi/konfiguracemi"
```

Výsledky hybridní pipeline jsou ukládány jako `hybrid_results.json` a po sémantickém porovnání jako `hybrid_comparison_semantic.json`.

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

## Hromadné generování hybridních výsledků a porovnání

Pro snadné generování hybridních výsledků a porovnání všech konfigurací byl implementován skript `generate_all_hybrid_results.py`. Tento skript umožňuje:

1. Zpracovat adresář s výsledky z různých konfigurací modelů
2. Pro každou konfiguraci vytvořit hybridní výsledky (kombinace Text a VLM)
3. Generovat grafy pro všechny konfigurace
4. Vytvořit finální porovnání všech konfigurací

### Použití

```bash
# Základní použití
python -m src.generate_all_hybrid_results --base-dir "cesta/k/adresáři/s/konfiguracemi"

# Pouze hybridní výsledky, bez generování grafů
python -m src.generate_all_hybrid_results --base-dir "cesta/k/adresáři/s/konfiguracemi" --hybrid-only

# Pouze generování grafů, bez hybridních výsledků
python -m src.generate_all_hybrid_results --base-dir "cesta/k/adresáři/s/konfiguracemi" --graphs-only

# Pouze finální porovnání
python -m src.generate_all_hybrid_results --base-dir "cesta/k/adresáři/s/konfiguracemi" --final-only

# Podrobný výstup
python -m src.generate_all_hybrid_results --base-dir "cesta/k/adresáři/s/konfiguracemi" --verbose
```

### Výstupy

Skript vytváří následující výstupy:

1. **Hybridní výsledky** pro každou konfiguraci:
   - `hybrid_results.json` - výsledky hybridní kombinace
   - `hybrid_comparison_semantic.json` - výsledky po sémantickém porovnání

2. **Grafy** pro každou konfiguraci:
   - Standardní grafy (comparison_results.png, overall_results.png, atd.)
   - Box ploty pro detailnější pohled na distribuci výsledků

3. **Finální porovnání** v adresáři `final_comparison/`:
   - `all_models_comparison.csv` - tabulka s porovnáním všech modelů
   - `all_models_comparison.png` - graf porovnávající všechny konfigurace
   - `best_models.json` - informace o nejlepších modelech pro každý typ pipeline
   - `best_[pipeline]_*.png` - grafy zobrazující výsledky nejlepších modelů

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
│   ├── generate_all_hybrid_results.py # Skript pro hromadné generování hybridních výsledků
│   ├── combine_semantic_results.py # Skript pro kombinaci sémantických výsledků
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

Aplikace generuje výstupy v adresáři `results/`. Každý běh hlavního skriptu (`main.py`, `run_all_models.py`, `run_semantic_comparison.py`) vytvoří vlastní podadresář pojmenovaný podle typu běhu a časového razítka (např. `results/main_20231027_103000/` nebo `results/all_models_20231027_110000/`).

**Struktura pro `main.py` nebo `run_semantic_comparison.py` (např. `results/main_YYYYMMDD_HHMMSS/`):**

1.  **JSON soubory s výsledky**:
    *   `[model]_results.json`: Extrahovaná metadata a časy zpracování pro daný model (např. `embedded_results.json`). Obsahuje klíče `"results"` a `"timings"`.
    *   `[model]_comparison.json`: Výsledky základního porovnání s referencemi pro daný model.
    *   `[model]_comparison_semantic.json`: Výsledky po sémantickém porovnání (pokud bylo spuštěno).
    *   `semantic_comparison_summary.json`: Souhrnné výsledky sémantického porovnání (pokud bylo spuštěno pomocí `run_semantic_comparison.py`, obsahuje aktualizovaná data pro všechny zpracované modely).
    *   `hybrid_results.json`: Výsledky hybridní kombinace Text a VLM pipeline.
    *   `hybrid_comparison_semantic.json`: Výsledky hybridní kombinace po sémantickém porovnání.

2.  **Vizualizace**:
    *   `comparison_results.png`: Sloupcový graf porovnání úspěšnosti modelů pro jednotlivá pole metadat (s chybovými úsečkami ±1σ).
    *   `overall_results.png`: Sloupcový graf celkové úspěšnosti modelů (s chybovými úsečkami ±1σ).
    *   `comparison_results_boxplot.png`: Box plot zobrazující distribuci skóre podobnosti pro jednotlivá pole a modely.
    *   `overall_results_boxplot.png`: Box plot zobrazující distribuci celkového skóre podobnosti pro jednotlivé modely.

3.  **CSV tabulky**:
    *   `summary_results.csv`: Souhrnné statistiky (průměr, směrodatná odchylka) pro každé pole a model.
    *   `overall_summary_results.csv`: Celkové souhrnné statistiky (průměr, směrodatná odchylka, průměrný čas) pro každý model.
    *   `detailed_scores_all.csv`: Detailní skóre podobnosti a časy pro každý dokument, pole a model.

**Struktura pro `run_all_models.py` (např. `results/all_models_YYYYMMDD_HHMMSS/`):**

*   **Podadresáře pro jednotlivé konfigurace**: Každá testovaná konfigurace modelu bude mít vlastní podadresář (např. `konfigurace-1_20231027_110010/`) se strukturou výstupů stejnou jako pro `main.py` (viz výše).
*   **Adresář `final_comparison/`**: Obsahuje souhrnné výsledky porovnávající výsledky spuštěné v rámci tohoto běhu `run_all_models.py`:
    *   **Grafy podle typu pipeline** (s příponou `-pipelines`):
        *   `[field]_comparison-pipelines.png`: Graf porovnání průměrné úspěšnosti podle typu pipeline (EMBEDDED, TEXT, VLM, HYBRID) pro konkrétní pole.
        *   `overall_comparison-pipelines.png`: Graf celkové úspěšnosti jednotlivých typů pipeline.
        *   `[field]_comparison-pipelines.csv`: CSV data k příslušnému grafu.

    *   **Grafy podle konkrétních modelů** (s příponou `-models`):
        *   `[field]_comparison-models.png`: Graf porovnání průměrné úspěšnosti konkrétních modelů pro dané pole, s barevným rozlišením typu pipeline.
        *   `overall_comparison-models.png`: Graf celkové úspěšnosti konkrétních modelů s barevným rozlišením typu pipeline.
        *   `[field]_comparison-models.csv`: CSV data k příslušnému grafu.

    *   **Souhrnné CSV soubory**:
        *   `final_summary_all_fields-pipelines.csv` a `final_summary_all_fields-models.csv`: Kompletní data pro všechna pole a všechny modely.
        *   `final_overall_all-pipelines.csv` a `final_overall_all-models.csv`: Souhrnná celková data pro všechny modely.
        *   `all_models_comparison.csv`: Tabulka s porovnáním všech modelů ze všech konfigurací.
        *   `best_models.json`: Informace o nejlepších modelech pro každý typ pipeline. 