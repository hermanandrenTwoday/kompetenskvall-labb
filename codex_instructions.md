# CODEX_INSTRUCTIONS.md

## Syfte

Skapa ett litet, pedagogiskt och körbart labbrepo för BI-konsulter som ska få en praktisk introduktion till AI/ML i notebook-format.

Repo:t ska vara enkelt att förstå, enkelt att köra och tillräckligt realistiskt för att kännas relevant för BI-konsulter. Fokus är inte forskning eller avancerad ML, utan ett tydligt end-to-end-flöde med:

- inläsning av data
- datatvätt
- join / feature engineering
- träning av modell
- sparad modell / pipeline
- inference på ny rad
- utvärdering / validering
- snygga visualiseringar

Detta ska vara en **riktig ML-pipeline**, inte bara analys i notebook.  
Vi kör **inte low-code**, **inte AutoML**, och **inte drag-and-drop**.  
Vi kör **notebooks**, helst plattformsneutralt men gärna kompatibelt med Databricks.

---

## Målgrupp

Målgruppen är BI-konsulter med varierande teknisk nivå.

Detta innebär att lösningen ska:

- vara enkel att följa
- ha tydlig struktur
- undvika onödigt avancerade ML-koncept
- använda välkända Python-bibliotek
- visa tydligt affärsnyttiga steg som BI-personer känner igen:
  - joins
  - saknade värden
  - normalisering
  - feature engineering
  - modellträning
  - inference
  - visuals

---

## Övergripande labb-case

Bygg ett **clustering-case på kundliknande data**.

Exempel på case:
- kundsegmentering
- gruppering av kunder baserat på beteende, storlek, aktivitet, köpvolym, geografiska attribut, produktmix etc.

Det ska kännas som ett BI-/analytikercase, inte som ett rent akademiskt iris-exempel.

---

## Viktig princip

Det här repo:t måste innehålla en **faktisk ML-pipeline**.

Det betyder att lösningen ska visa hela kedjan:

1. Läs in rådata
2. Förbered data
3. Bygg preprocessing-steg
4. Träna modell
5. Spara fitted pipeline / modell
6. Läsa in ny rad eller nya rader
7. Köra inference med samma fitted preprocessing + modell
8. Returnera cluster assignment / prediction

Det får **inte** lösas som:
- man manuellt gör om preprocessing separat för ny data
- man tränar en modell utan att spara preprocessing-logiken
- man bara visar analys utan riktig inference

Använd därför en tydlig pipeline-struktur, helst med `scikit-learn Pipeline` och `ColumnTransformer` där det passar.

---

## Leverabler som ska skapas

### 1. Datafiler
Minst 2 CSV-filer:

- en huvudtabell med kundliknande data
- en lookup-/dimensionstabell som används i en join

Exempel:
- `customers.csv`
- `regions.csv`

Dessutom ska det finnas en liten fil för inference, t.ex.:
- `new_customers.csv`

Datat ska innehålla:
- numeriska features
- minst någon kategorisk feature
- minst någon kolumn med missing values
- en join-nyckel
- data som lämpar sig för clustering

Datat får gärna vara syntetiskt men ska se realistiskt ut.

---

### 2. Notebooks

Skapa 2 notebooks:

#### Notebook 1: `01_prepare_and_explore.ipynb`

Syfte:
- läsa in data
- göra join
- göra enkel datakvalitetskontroll
- inspektera null-värden
- definiera features
- förbereda data för modellering
- göra explorativ analys
- skapa tydlig grund för pipeline-träning

Denna notebook ska vara pedagogisk och tydligt visa:
- vad som är rådata
- vad som är features
- vilka kolumner som är numeriska respektive kategoriska
- vilka problem som måste lösas före träning

#### Notebook 2: `02_train_and_infer.ipynb`

Syfte:
- bygga en riktig ML-pipeline
- träna clusteringmodell
- utvärdera modellresultat
- spara pipeline/modell
- läsa in nya observationer
- köra inference på nya rader
- skriva ut resultat

Denna notebook ska innehålla:

- preprocessing med `ColumnTransformer`
- imputering av missing values
- encoding av kategoriska features om det behövs
- scaling av numeriska features
- modellträning med `KMeans`
- val av antal kluster på enkel pedagogisk nivå
- utvärdering med t.ex. silhouette score
- tilldelning av cluster labels till träningsdata
- inference på `new_customers.csv`
- sparande av pipeline/modell till fil, t.ex. med `joblib`

Inference-delen ska tydligt visa att **samma pipeline används** för nya rader som för träningsdata.

---

### 3. README

Skapa en tydlig `README.md` som beskriver:

- syftet med labben
- målgruppen
- repo-struktur
- hur man kör labben
- förkunskaper
- vad deltagarna förväntas lära sig
- att repo:t visar en riktig ML-pipeline från träning till inference

README:n ska vara kort men tydlig.

---

### 4. Requirements-fil

Skapa `requirements.txt` med endast nödvändiga beroenden.

Använd:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter
- joblib

Undvik tunga eller onödiga beroenden.

---

## Repo-struktur

Använd ungefär denna struktur:

```text
ai-bi-lab/
├─ data/
│  ├─ raw/
│  │  ├─ customers.csv
│  │  ├─ regions.csv
│  │  └─ new_customers.csv
│  └─ processed/
│     └─ customer_enriched.csv
├─ notebooks/
│  ├─ 01_prepare_and_explore.ipynb
│  └─ 02_train_and_infer.ipynb
├─ models/
│  └─ clustering_pipeline.joblib
├─ outputs/
│  ├─ clustered_customers.csv
│  ├─ inference_results.csv
│  └─ figures/
├─ requirements.txt
├─ README.md
└─ .gitignore