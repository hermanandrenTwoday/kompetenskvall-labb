# AI / BI Lab

Ett litet notebook-baserat labb f?r BI-konsulter som visar en riktig pipeline f?r kundsegmentering.

## Det h?r visar labbet

- inl?sning av verklig transaktionsdata
- join mot en lookup-tabell
- aggregering till kundniv? med RFM-liknande features
- datakontroll och saknade v?rden
- preprocessing med `ColumnTransformer`
- clustering med KMeans
- modellval med silhouette score
- sparad pipeline med `joblib`
- inference p? nya kunder med samma fitted preprocessing

## Datak?lla

Labbet anv?nder det verkliga UCI-datasetet **Online Retail**:
https://archive.ics.uci.edu/dataset/352/online-retail

Notebook-fl?det bygger kundniv?features fr?n transaktionsdata och l?gger in ett litet antal saknade v?rden f?r att demonstrera imputering p? ett realistiskt s?tt.

## Repo-struktur

- `data/raw/transactions.csv` - r?a transaktioner
- `data/raw/regions.csv` - lookup-tabell f?r l?nder
- `data/raw/customers.csv` - aggregerad kundtabell
- `data/raw/new_customers.csv` - rader f?r inference
- `data/processed/customer_enriched.csv` - tr?ningsdata med missing values
- `notebooks/` - f?rberedelse och tr?ning/inference
- `models/` - sparad pipeline
- `outputs/` - klustrad data, inference-resultat och figurer

## K?rning

1. Installera beroenden: `pip install -r requirements.txt`
2. ?ppna `notebooks/01_prepare_and_explore.ipynb`
3. K?r `notebooks/02_train_and_infer.ipynb`

## F?rkunskaper

- grundl?ggande Python och notebook-kunskap
- ingen avancerad ML kr?vs

## Vad deltagarna l?r sig

- hur en BI-join blir ett ML-feature set
- varf?r preprocessing m?ste sparas tillsammans med modellen
- hur clustering utv?rderas med silhouette score
- hur samma pipeline anv?nds f?r inference p? nya rader

## Referens

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
