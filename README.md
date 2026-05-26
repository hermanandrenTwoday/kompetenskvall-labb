# Kompetenskväll labb

Ett enkelt Databricks-labb för BI-konsulter som visar en komplett ML-pipeline för kundsegmentering.

## Mål

Ladda upp projektet som en zip i Databricks Free och kör notebooksen i ordning:

1. `notebooks/00_csv_to_delta_setup.py`
2. `notebooks/01_prepare_and_explore.py`
3. `notebooks/02_train_and_infer.py`

## Vad labbet visar

- CSV till Delta tables
- SQL-first förberedelse av data
- join mot lookup-tabell
- kundfeatures på kundnivå
- KMeans-klustring med `scikit-learn Pipeline`
- val av antal kluster med silhouette score och elbow-metoden
- inference på nya kunder med samma pipeline

## Övningsmoment i notebook 02

Notebook 02 startar med en enkel RFM-baseline:

- `recency_days`
- `frequency`
- `monetary`

Deltagarna kan sedan slå på fler föreslagna features i notebooken och jämföra silhouette score, elbow-kurva, klusterprofil och inference-resultat.

## Repo-struktur

- `data/raw/transactions.csv` - råa transaktioner
- `data/raw/regions.csv` - lookup-tabell för land och region
- `data/raw/new_customers.csv` - nya kunder för inference
- `notebooks/00_csv_to_delta_setup.py` - konverterar CSV till Delta
- `notebooks/01_prepare_and_explore.py` - bygger customer features
- `notebooks/02_train_and_infer.py` - tränar modell och gör inference

Notebooksen skapar dessa foldrar när de körs:

- `data/processed/`
- `outputs/`
- `outputs/figures/`
- `models/`

## Databricks Free workflow

1. Zippa projektmappen.
2. Ladda upp zippen i Databricks workspace.
3. Öppna `00_csv_to_delta_setup.py` och kör hela notebooken.
4. Öppna `01_prepare_and_explore.py` och kör hela notebooken.
5. Öppna `02_train_and_infer.py` och kör hela notebooken.

Notebooksen försöker använda catalog `workspace` och schema `kompetenskvall_labb`. Om `workspace` inte finns faller notebook 00 tillbaka till `hive_metastore`.

## Datakälla

Labbet använder UCI-datasetet Online Retail:
https://archive.ics.uci.edu/dataset/352/online-retail

## Lokalt Python-labb

Om du vill testa delar lokalt kan du installera beroenden:

```bash
pip install -r requirements.txt
```

Databricks-notebooksen kräver dock Spark/Databricks för Delta-steget.
