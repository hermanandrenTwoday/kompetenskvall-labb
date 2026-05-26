# Kompetenskvall labb

Ett enkelt Databricks-labb for BI-konsulter som visar en komplett ML-pipeline for kundsegmentering.

## Mal

Ladda upp projektet som en zip i Databricks Free och kor notebooksen i ordning:

1. `notebooks/00_csv_to_delta_setup.py`
2. `notebooks/01_prepare_and_explore.py`
3. `notebooks/02_train_and_infer.py`

## Vad labbet visar

- CSV till Delta tables
- SQL-first forberedelse av data
- join mot lookup-tabell
- kundfeatures pa kundniva
- KMeans-klustring med `scikit-learn Pipeline`
- val av antal kluster med silhouette score
- inference pa nya kunder med samma pipeline

## Repo-struktur

- `data/raw/transactions.csv` - raa transaktioner
- `data/raw/regions.csv` - lookup-tabell for land och region
- `data/raw/new_customers.csv` - nya kunder for inference
- `notebooks/00_csv_to_delta_setup.py` - konverterar CSV till Delta
- `notebooks/01_prepare_and_explore.py` - bygger customer features
- `notebooks/02_train_and_infer.py` - tranar modell och gor inference

Notebooksen skapar dessa foldrar nar de kors:

- `data/processed/`
- `outputs/`
- `outputs/figures/`
- `models/`

## Databricks Free workflow

1. Zippa projektmappen.
2. Ladda upp zippen i Databricks workspace.
3. Oppna `00_csv_to_delta_setup.py` och kor hela notebooken.
4. Oppna `01_prepare_and_explore.py` och kor hela notebooken.
5. Oppna `02_train_and_infer.py` och kor hela notebooken.

Notebooksen forsoker anvanda catalog `workspace` och schema `kompetenskvall_labb`. Om `workspace` inte finns faller notebook 00 tillbaka till `hive_metastore`.

## Datakalla

Labbet anvander UCI-datasetet Online Retail:
https://archive.ics.uci.edu/dataset/352/online-retail

## Lokalt Python-labb

Om du vill testa delar lokalt kan du installera beroenden:

```bash
pip install -r requirements.txt
```

Databricks-notebooksen kraver dock Spark/Databricks for Delta-steget.
