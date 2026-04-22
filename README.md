# AI / BI Lab

Ett litet notebook-baserat labb för BI-konsulter som visar en riktig pipeline för kundsegmentering.

## Det här visar labbet

- inläsning av verklig transaktionsdata
- join mot en lookup-tabell
- aggregering till kundnivå med RFM-liknande features
- datakontroll och saknade värden
- preprocessing med `ColumnTransformer`
- clustering med KMeans
- modellval med silhouette score
- sparad pipeline med `joblib`
- inference på nya kunder med samma fitted preprocessing

## Datakälla

Labbet använder det verkliga UCI-datasetet **Online Retail**:
https://archive.ics.uci.edu/dataset/352/online-retail

Notebook-flödet bygger kundnivåfeatures från transaktionsdata och lägger in ett litet antal saknade värden för att demonstrera imputering på ett realistiskt sätt.

## Repo-struktur

- `data/raw/transactions.csv` - råa transaktioner
- `data/raw/regions.csv` - lookup-tabell för länder
- `data/raw/customers.csv` - aggregerad kundtabell
- `data/raw/new_customers.csv` - rader för inference
- `data/processed/customer_enriched.csv` - träningsdata med missing values
- `notebooks/` - förberedelse och träning/inference
- `models/` - sparad pipeline
- `outputs/` - klustrad data, inference-resultat och figurer

## Körning

1. Installera beroenden: `pip install -r requirements.txt`
2. Öppna `notebooks/01_prepare_and_explore.ipynb`
3. Kör `notebooks/02_train_and_infer.ipynb`

## Förkunskaper

- grundläggande Python och notebook-kunskap
- ingen avancerad ML krävs

## Vad deltagarna lär sig

- hur en BI-join blir ett ML-feature set
- varför preprocessing måste sparas tillsammans med modellen
- hur clustering utvärderas med silhouette score
- hur samma pipeline används för inference på nya rader

## Referens

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
