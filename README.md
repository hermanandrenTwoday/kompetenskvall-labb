# AI / BI Lab

Ett enkelt notebook-labb för BI-konsulter som visar en riktig ML-pipeline för kundsegmentering.

## Vad labbet visar

- SQL-first förberedelse av data (join + feature engineering)
- KMeans-klustring med `scikit-learn Pipeline`
- val av antal kluster med silhouette score
- inference på nya rader med samma pipeline
- tydliga 2D-plottar av träning och inference

## Datakälla

Labbet använder UCI-datasetet **Online Retail**:
https://archive.ics.uci.edu/dataset/352/online-retail

## Repo-struktur

- `data/raw/transactions.csv` - råa transaktioner
- `data/raw/regions.csv` - lookup-tabell för länder
- `data/raw/new_customers.csv` - rader för inference
- `data/processed/customer_enriched.csv` - träningsdata på kundnivå
- `notebooks/01_prepare_and_explore.ipynb` - SQL-first förberedelse
- `notebooks/02_train_and_infer.ipynb` - träning, silhouette, inference

## Körning

1. Installera beroenden: `pip install -r requirements.txt`
2. Kör `notebooks/01_prepare_and_explore.ipynb`
3. Kör `notebooks/02_train_and_infer.ipynb`

## Viktigt om prediction-kolumnen

`PredictedCluster` är modellens numeriska kluster-ID från `KMeans`.
Ingen manuell namngivning av kluster används.

## Förkunskaper

- grundläggande Python och notebook-kunskap
- ingen avancerad ML krävs

## Referens

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
