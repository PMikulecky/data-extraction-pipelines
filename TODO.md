# TODO seznam

## Vysoká priorita

### Implementace textové pipeline
- [x] Implementovat čistě textovou pipeline
  - Využít rozdělení dokumentu na části (úvod, tělo, reference)
  - Extrahovat metadata pouze z textu
  - Optimalizovat výběr relevantních částí dokumentu pro extrakci

### Vylepšení porovnávání
- [x] Implementovat validaci všech metadat (již implementováno v `metadata_comparator.py`)
- [x] Vytvořit report o neshodách (již implementováno v `run_all_models.py`)
- [ ] Přidat možnost manuálního přezkoumání neshod
- [ ] Vylepšit sémantické porovnání pro specifické typy metadat (např. reference)

### Filtrace dokumentů
- [x] Přidat kontrolu na přítomnost DOI (již implementováno v `data_preparation.py`)
- [x] Přidat kontrolu na vyplněný sloupec referencí (již implementováno v `data_preparation.py`)
- [x] Vytvořit `papers-filtered.csv` pouze s kompletními záznamy o DOI a reference (již implementováno)
- [ ] Vylepšit stahování PDF souborů
  - Přidat podporu pro více zdrojů PDF
  - Implementovat retry mechanismus pro neúspěšné pokusy
  - Přidat logování neúspěšných pokusů pro pozdější analýzu

### Optimalizace Embedded pipeline
- [ ] Maximalizovat úspěšnost Embedded pipeline
  - Ověřit použití maximálního kontextového okna
  - Optimalizovat chunking strategii
  - Vylepšit dotazovací prompty pro extrakci metadat (již implementovány v `embedded_pipeline.py`)

## Střední priorita

### Dokumentace
- [ ] Aktualizovat dokumentaci
  - Přidat popis nové textové pipeline
  - Dokumentovat vylepšené porovnání referencí
  - Přidat příklady použití
  - Dokumentovat proces stahování PDF a řešení běžných problémů

### Testování
- [ ] Rozšířit testovací sadu
  - Přidat testy pro novou textovou pipeline
  - Vytvořit testy pro porovnávání referencí
  - Implementovat testy pro filtrování dokumentů
  - Přidat testy pro stahování PDF z různých zdrojů

## Nízká priorita

### Vylepšení uživatelského rozhraní
- [ ] Přidat více konfiguračních možností
  - Možnost nastavení velikosti kontextového okna
  - Výběr částí dokumentu pro extrakci
  - Konfigurace filtrovacích pravidel
  - Nastavení parametrů pro stahování PDF
- [x] Vyčistit a centralizovat konfiguraci modelů
  - Definovat konfigurace pouze v jednom souboru (`model_configs.json`)
  - Odstranit duplicitní definice z kódu
  - Přidat validaci konfiguračního souboru

### Monitoring a logování
- [ ] Vylepšit systém logování
  - Přidat detailní logy pro extrakci
  - Implementovat monitoring úspěšnosti
  - Vytvořit reporty o chybách
  - Přidat statistiky o úspěšnosti stahování PDF 