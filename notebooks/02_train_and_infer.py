# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Träna och inferera
# MAGIC
# MAGIC Den här notebooken förväntar sig att notebook 01 har skapat träningsdata på kundnivå.
# MAGIC Den jämför flera värden på `k`, tränar KMeans-pipelines och predikterar kluster för nya kunder.

# COMMAND ----------

# MAGIC %pip install -q "threadpoolctl>=3.5.0"

# COMMAND ----------

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# COMMAND ----------

def get_lab_root() -> Path:
    if "dbutils" in globals():
        notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        workspace_path = Path(notebook_path)
        if not str(workspace_path).startswith("/Workspace/"):
            workspace_path = Path("/Workspace") / str(workspace_path).lstrip("/")
        return workspace_path.parent.parent

    return Path(__file__).resolve().parent.parent


def resolve_catalog(schema_name: str) -> str | None:
    for catalog_name in ["workspace", "hive_metastore"]:
        try:
            spark.sql(f"DESCRIBE SCHEMA `{catalog_name}`.`{schema_name}`")
            return catalog_name
        except Exception:
            pass
    return None


def table_id(table_name: str) -> str:
    return f"`{target_catalog}`.`{target_schema}`.`{table_name}`"


lab_root = get_lab_root()
raw_path = lab_root / "data" / "raw"
processed_path = lab_root / "data" / "processed"
outputs = lab_root / "outputs"
outputs_figures = outputs / "figures"
models = lab_root / "models"

processed_path.mkdir(parents=True, exist_ok=True)
outputs_figures.mkdir(parents=True, exist_ok=True)
models.mkdir(parents=True, exist_ok=True)

target_schema = "kompetenskvall_labb"
target_catalog = resolve_catalog(target_schema)

print(f"Labbrot: {lab_root}")
print(f"Använder schema: {target_catalog}.{target_schema}" if target_catalog else "Inget Delta-schema hittades; använder CSV-filer.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Läs träningsdata på kundnivå
# MAGIC
# MAGIC Modellen tränas på en rad per kund från notebook 01. Varje rad beskriver beteende:
# MAGIC hur nylig kunden är, hur ofta kunden köper, hur mycket kunden spenderar och relaterade korgmått.

# COMMAND ----------

# DBTITLE 1,Läs träningsdata
if target_catalog:
    train_df = spark.table(table_id("customer_enriched")).toPandas()
else:
    train_csv = processed_path / "customer_enriched.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Saknar {train_csv}. Kör notebook 01 först.")
    train_df = pd.read_csv(train_csv)

train_df["last_purchase_date"] = pd.to_datetime(train_df["last_purchase_date"])
if "display" in globals():
    display(train_df.head())
else:
    print(train_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Kort förklaring av silhouette score och elbow
# MAGIC
# MAGIC Silhouette score mäter hur tydligt en punkt hör till sitt eget kluster jämfört med närmaste andra kluster.
# MAGIC
# MAGIC - nära `1.0`: tydligare separation
# MAGIC - nära `0.0`: kluster överlappar
# MAGIC - under `0.0`: många punkter kan vara svagt placerade
# MAGIC
# MAGIC Se det som en teknisk signal, inte som ett slutligt affärssvar. Ett lite lägre värde kan fortfarande vara användbart
# MAGIC om segmenten är enklare att förstå och agera på.
# MAGIC
# MAGIC Elbow-metoden tittar på inertia: hur tätt punkterna ligger runt sina klustercentrum. Inertia blir nästan alltid
# MAGIC bättre när `k` ökar, så vi letar efter punkten där förbättringen börjar plana ut. Den böjen är "elbow".
# MAGIC Även detta är en vägledning, inte en regel.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Labbövning: välj features
# MAGIC
# MAGIC Modellen börjar med en enkel RFM-baseline:
# MAGIC
# MAGIC - `recency_days`
# MAGIC - `frequency`
# MAGIC - `monetary`
# MAGIC
# MAGIC Föreslagna features att testa nästa:
# MAGIC
# MAGIC - `avg_order_value` - skiljer många små köp från färre stora köp
# MAGIC - `basket_size` - fångar total volym
# MAGIC - `avg_items_per_invoice` - skiljer bulk-köp från små varukorgar
# MAGIC - `unique_products` - fångar bredd i sortiment
# MAGIC - `avg_unit_price` - fångar premiumbeteende jämfört med lågpris
# MAGIC - `RegionGroup` - testar om geografi förbättrar eller stör segmenteringen
# MAGIC
# MAGIC Ändra `use_extended_features` nedan och kör om notebooken. Jämför silhouette score, klusterprofiler och inference-punkter.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Förbehandla före klustring
# MAGIC
# MAGIC KMeans är avståndsbaserad. Om vi använde råa värden skulle stora kolumner som `monetary` dominera mindre kolumner.
# MAGIC Pipelinen gör därför detta:
# MAGIC
# MAGIC - fyller saknade numeriska värden med median
# MAGIC - använder `log1p` för att minska extrem retail-skevhet
# MAGIC - standardiserar numeriska kolumner till jämförbar skala
# MAGIC - one-hot-encodar kategorikolumner när extended features är aktiverade

# COMMAND ----------

# DBTITLE 1,Definiera preprocessing-pipeline
use_extended_features = False

baseline_numeric_features = [
    "recency_days",
    "frequency",
    "monetary",
]
suggested_numeric_features = [
    "avg_order_value",
    "basket_size",
    "avg_items_per_invoice",
    "unique_products",
    "avg_unit_price",
]

numeric_features = baseline_numeric_features.copy()
categorical_features = []

if use_extended_features:
    numeric_features = baseline_numeric_features + suggested_numeric_features
    categorical_features = ["RegionGroup"]

feature_columns = numeric_features + categorical_features
X_train = train_df[feature_columns].copy()

print("Featureläge:", "extended" if use_extended_features else "baseline")
print("Numeriska features:", numeric_features)
print("Kategoriska features:", categorical_features)

numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Träna flera kandidater för klustring
# MAGIC
# MAGIC Här finns ingen label-kolumn. KMeans grupperar kunder efter likhet i den förberedda feature-rymden.
# MAGIC Vi jämför flera värden på `k` för att visa att antalet segment är ett modellval.

# COMMAND ----------

# DBTITLE 1,Träna modeller och jämför k
k_values = [2, 3, 4, 5, 6, 7]
model_selection_rows = []
trained_pipelines = {}
train_cluster_by_k = {}

for k in k_values:
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=20)),
        ]
    )

    train_clusters = pipeline.fit_predict(X_train)
    X_prepared = pipeline.named_steps["preprocessor"].transform(X_train)
    score = silhouette_score(X_prepared, train_clusters)
    inertia = pipeline.named_steps["kmeans"].inertia_

    trained_pipelines[k] = pipeline
    train_cluster_by_k[k] = train_clusters
    model_selection_rows.append({"k": k, "silhouette_score": score, "inertia": inertia})

model_selection_df = pd.DataFrame(model_selection_rows).sort_values("k")
best_k = int(model_selection_df.loc[model_selection_df["silhouette_score"].idxmax(), "k"])
if "display" in globals():
    display(model_selection_df)
else:
    print(model_selection_df)

# COMMAND ----------

# DBTITLE 1,Skapa output-dataframe med klustertilldelningar
train_output_df = train_df.copy()
for k in k_values:
    train_output_df[f"Cluster_k{k}"] = train_cluster_by_k[k]

if "display" in globals():
    display(train_output_df.head())
else:
    print(train_output_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jämför silhouette och elbow
# MAGIC
# MAGIC Silhouette-plotten markerar högsta värdet. Elbow-plotten visar om fler kluster fortfarande ger en
# MAGIC meningsfull minskning av inertia.
# MAGIC
# MAGIC I ett verkligt CRM- eller BI-case skulle man också kontrollera om segmenten är begripliga, tillräckligt stora
# MAGIC och användbara för åtgärder.

# COMMAND ----------

# DBTITLE 1,Plotta silhouette och elbow
sns.set_theme(style="whitegrid", context="talk")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.lineplot(data=model_selection_df, x="k", y="silhouette_score", marker="o", ax=axes[0], color="#0b5cad")
axes[0].axvline(best_k, color="#d62728", linestyle="--", linewidth=1)
axes[0].set_title("Silhouette score")
axes[0].set_xlabel("Antal kluster (k)")
axes[0].set_ylabel("Silhouette score")
axes[0].set_xticks(k_values)
for _, row in model_selection_df.iterrows():
    axes[0].annotate(
        f"{row['silhouette_score']:.3f}",
        (row["k"], row["silhouette_score"]),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=10,
    )

sns.lineplot(data=model_selection_df, x="k", y="inertia", marker="o", ax=axes[1], color="#2ca02c")
axes[1].set_title("Elbow-kurva")
axes[1].set_xlabel("Antal kluster (k)")
axes[1].set_ylabel("Inertia")
axes[1].set_xticks(k_values)

plt.tight_layout()
plt.savefig(outputs_figures / "model_selection_silhouette_elbow.png", dpi=160, bbox_inches="tight")
plt.show()

print(f"Valt k enligt högst silhouette score: {best_k}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Läs in kunder för inference
# MAGIC
# MAGIC Inference betyder att vi använder den tränade pipelinen på nya kunder. Den viktiga regeln är att nya kunder måste ha
# MAGIC samma feature-kolumner som träningsdatan. Pipelinen hanterar saknade värden och skalning på samma sätt som vid träning.

# COMMAND ----------

# DBTITLE 1,Läs in nya kunder och regioner
if target_catalog:
    new_customers_df = spark.table(table_id("new_customers")).toPandas()
    regions_df = spark.table(table_id("regions")).toPandas()
else:
    new_customers_df = pd.read_csv(raw_path / "new_customers.csv")
    regions_df = pd.read_csv(raw_path / "regions.csv")

inference_df = new_customers_df.merge(
    regions_df[["Country", "RegionGroup"]],
    on="Country",
    how="left",
)
inference_df["RegionGroup"] = inference_df["RegionGroup"].fillna("Other")
if "display" in globals():
    display(inference_df)
else:
    print(inference_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Snabb jämförelse: baseline vs extended features
# MAGIC
# MAGIC Den här jämförelsen tränar samma `k=3`-modell två gånger:
# MAGIC
# MAGIC - baseline med RFM-features
# MAGIC - utökad feature-uppsättning med korg-, produkt-, pris- och regionfeatures
# MAGIC
# MAGIC Poängen är att snabbt visa att feature-val kan ändra segmenteringen.
# MAGIC När features ändras ändrar vi också vad "liknande kund" betyder.

# COMMAND ----------

# DBTITLE 1,Jämför baseline och utökad feature-uppsättning för k=3
feature_sets = {
    "baseline_rfm": {
        "numeric": baseline_numeric_features,
        "categorical": [],
    },
    "extended": {
        "numeric": baseline_numeric_features + suggested_numeric_features,
        "categorical": ["RegionGroup"],
    },
}

comparison_rows = []
comparison_predictions = {}

for feature_set_name, feature_set in feature_sets.items():
    comparison_numeric_features = feature_set["numeric"]
    comparison_categorical_features = feature_set["categorical"]
    comparison_feature_columns = comparison_numeric_features + comparison_categorical_features

    comparison_transformers = [
        ("num", numeric_pipeline, comparison_numeric_features),
    ]
    if comparison_categorical_features:
        comparison_transformers.append(("cat", categorical_pipeline, comparison_categorical_features))

    comparison_preprocessor = ColumnTransformer(comparison_transformers)
    comparison_pipeline = Pipeline(
        [
            ("preprocessor", comparison_preprocessor),
            ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=20)),
        ]
    )

    comparison_X_train = train_df[comparison_feature_columns].copy()
    comparison_clusters = comparison_pipeline.fit_predict(comparison_X_train)
    comparison_X_prepared = comparison_pipeline.named_steps["preprocessor"].transform(comparison_X_train)
    comparison_score = silhouette_score(comparison_X_prepared, comparison_clusters)

    comparison_rows.append(
        {
            "feature_set": feature_set_name,
            "feature_count": len(comparison_feature_columns),
            "silhouette_score_k3": comparison_score,
            "cluster_counts": ", ".join(
                f"{cluster_id}: {count}"
                for cluster_id, count in pd.Series(comparison_clusters).value_counts().sort_index().items()
            ),
        }
    )
    comparison_predictions[feature_set_name] = comparison_pipeline.predict(
        inference_df[comparison_feature_columns]
    )

feature_comparison_df = pd.DataFrame(comparison_rows)
if "display" in globals():
    display(feature_comparison_df)
else:
    print(feature_comparison_df)

feature_comparison_inference_df = inference_df[["CustomerID", "Country"]].copy()
feature_comparison_inference_df["baseline_rfm_k3"] = comparison_predictions["baseline_rfm"]
feature_comparison_inference_df["extended_k3"] = comparison_predictions["extended"]
feature_comparison_inference_df["changed"] = (
    feature_comparison_inference_df["baseline_rfm_k3"]
    != feature_comparison_inference_df["extended_k3"]
)

if "display" in globals():
    display(feature_comparison_inference_df)
else:
    print(feature_comparison_inference_df)

changed_count = int(feature_comparison_inference_df["changed"].sum())
print(f"Feature-valet flyttade {changed_count} av {len(feature_comparison_inference_df)} inference-kunder för k=3.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediktera kluster för nya kunder
# MAGIC
# MAGIC Det numeriska kluster-id:t är inte en affärsetikett. Det säger bara vilken inlärd grupp kunden ligger närmast.
# MAGIC Vi tolkar grupperna genom att titta på feature-medelvärden och plottar nedan.

# COMMAND ----------

# DBTITLE 1,Prediktera kluster för nya kunder
for k in k_values:
    inference_df[f"PredictedCluster_k{k}"] = trained_pipelines[k].predict(inference_df[feature_columns])

if "display" in globals():
    display(inference_df)
else:
    print(inference_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tolka kluster med profiler
# MAGIC
# MAGIC Kluster-id:n är godtyckliga. För att förstå dem jämför vi genomsnittligt beteende per kluster:
# MAGIC recency, frequency, monetary value och order value. Det här är steget där BI-domänkunskap är viktigast.

# COMMAND ----------

# DBTITLE 1,Klusterprofil för k=3
k3_profile_df = (
    train_output_df.groupby("Cluster_k3")
    .agg(
        customers=("CustomerID", "count"),
        recency_mean=("recency_days", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean"),
        avg_order_value_mean=("avg_order_value", "mean"),
    )
    .round(2)
)
k3_profile_df["share_of_customers"] = (k3_profile_df["customers"] / len(train_output_df)).round(3)
k3_profile_df = k3_profile_df[
    ["customers", "share_of_customers", "recency_mean", "frequency_mean", "monetary_mean", "avg_order_value_mean"]
]
if "display" in globals():
    display(k3_profile_df)
else:
    print(k3_profile_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jämför klusterbeteende över fler features
# MAGIC
# MAGIC Staplarna är indexerade mot hela datasetet. Värdet `1.0` betyder att klustret ligger på totalgenomsnittet.
# MAGIC Värden över eller under `1.0` gör det enklare att beskriva klustret på affärsspråk.

# COMMAND ----------

# DBTITLE 1,Utökad visualisering av klusterprofil
k3_extended_profile_df = (
    train_output_df.groupby("Cluster_k3")
    .agg(
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        avg_order_value=("avg_order_value", "mean"),
        basket_size=("basket_size", "mean"),
        unique_products=("unique_products", "mean"),
        avg_items_per_invoice=("avg_items_per_invoice", "mean"),
    )
    .round(2)
)

plot_features = [
    "frequency",
    "monetary",
    "avg_order_value",
    "basket_size",
    "unique_products",
    "avg_items_per_invoice",
]
feature_label_map = {
    "frequency": "Frequency",
    "monetary": "Monetary",
    "avg_order_value": "Snittvärde per order",
    "basket_size": "Korgstorlek",
    "unique_products": "Unika produkter",
    "avg_items_per_invoice": "Snittartiklar per faktura",
}

k3_relative_df = k3_extended_profile_df[plot_features].div(train_output_df[plot_features].mean(), axis=1)
k3_relative_long_df = (
    k3_relative_df.reset_index().melt(id_vars="Cluster_k3", var_name="feature", value_name="relative_index")
)
k3_relative_long_df["feature_label"] = k3_relative_long_df["feature"].map(feature_label_map)

plt.figure(figsize=(12, 5))
sns.barplot(
    data=k3_relative_long_df,
    x="feature_label",
    y="relative_index",
    hue="Cluster_k3",
    palette="tab10",
)
plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
plt.title("k=3-klusterprofil över flera features (index mot total)")
plt.xlabel("Feature")
plt.ylabel("Relativt index (1.0 = totalgenomsnitt)")
plt.xticks(rotation=20, ha="right")
plt.legend(title="Kluster", loc="upper right")
plt.tight_layout()
plt.savefig(outputs_figures / "k3_cluster_feature_bars.png", dpi=160, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plotta kluster i affärsnära axlar
# MAGIC
# MAGIC `recency_days` och `monetary` är enkla att förklara: hur nyligen kunden köpte och hur mycket kunden spenderade.
# MAGIC X-markeringarna är nya inference-kunder. Eftersom detta bara är en 2D-vy kan överlappande punkter fortfarande
# MAGIC separeras av andra features som modellen använder.

# COMMAND ----------

# DBTITLE 1,Scatterplots med kluster och inference-punkter
plot_x = "recency_days"
plot_y = "monetary"

plot_k_values = [2, 3, 4]

fig, axes = plt.subplots(1, len(plot_k_values), figsize=(18, 5), sharey=True)
for ax, k in zip(axes, plot_k_values):
    cluster_order = list(range(k))
    cluster_colors = dict(zip(cluster_order, sns.color_palette("tab10", k)))

    plot_train_df = train_output_df[[plot_x, plot_y, f"Cluster_k{k}"]].rename(columns={f"Cluster_k{k}": "Cluster"})
    plot_inference_df = inference_df[[plot_x, plot_y, f"PredictedCluster_k{k}"]].rename(
        columns={f"PredictedCluster_k{k}": "Cluster"}
    )

    sns.scatterplot(
        data=plot_train_df,
        x=plot_x,
        y=plot_y,
        hue="Cluster",
        hue_order=cluster_order,
        palette=cluster_colors,
        alpha=0.5,
        s=30,
        legend=False,
        ax=ax,
    )

    sns.scatterplot(
        data=plot_inference_df,
        x=plot_x,
        y=plot_y,
        hue="Cluster",
        hue_order=cluster_order,
        palette=cluster_colors,
        marker="X",
        s=170,
        edgecolor="black",
        linewidth=0.8,
        legend=False,
        ax=ax,
    )

    ax.set_yscale("log")
    ax.set_title(f"k = {k}")
    ax.set_xlabel("Recency (dagar)")

axes[0].set_ylabel("Monetary (log-skala)")
fig.suptitle("Kluster med inference-punkter (X) för valda k-värden", fontsize=16)
plt.tight_layout()
plt.savefig(outputs_figures / "cluster_scatter_k234_with_inference.png", dpi=160, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## PCA-vy av modellens feature-rymd
# MAGIC
# MAGIC PCA komprimerar den förberedda feature-rymden till två dimensioner. Det är användbart som en andra vy,
# MAGIC men det är fortfarande en förenkling. Använd den för att diskutera struktur, inte som en exakt karta över modellen.

# COMMAND ----------

# DBTITLE 1,PCA-visualisering med inference-punkter
best_preprocessor = trained_pipelines[best_k].named_steps["preprocessor"]
X_train_prepared = best_preprocessor.transform(X_train)
X_inference_prepared = best_preprocessor.transform(inference_df[feature_columns])

pca_model = PCA(n_components=2, random_state=42)
train_pca_values = pca_model.fit_transform(X_train_prepared)
inference_pca_values = pca_model.transform(X_inference_prepared)

train_pca_df = pd.DataFrame(train_pca_values, columns=["PC1", "PC2"])
inference_pca_df = pd.DataFrame(inference_pca_values, columns=["PC1", "PC2"])

fig, axes = plt.subplots(1, len(plot_k_values), figsize=(18, 5), sharex=True, sharey=True)
for ax, k in zip(axes, plot_k_values):
    cluster_order = list(range(k))
    cluster_colors = dict(zip(cluster_order, sns.color_palette("tab10", k)))

    plot_train_pca_df = train_pca_df.copy()
    plot_train_pca_df["Cluster"] = train_output_df[f"Cluster_k{k}"].values

    plot_inference_pca_df = inference_pca_df.copy()
    plot_inference_pca_df["Cluster"] = inference_df[f"PredictedCluster_k{k}"].values

    sns.scatterplot(
        data=plot_train_pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        hue_order=cluster_order,
        palette=cluster_colors,
        alpha=0.5,
        s=30,
        legend=False,
        ax=ax,
    )

    sns.scatterplot(
        data=plot_inference_pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        hue_order=cluster_order,
        palette=cluster_colors,
        marker="X",
        s=170,
        edgecolor="black",
        linewidth=0.8,
        legend=False,
        ax=ax,
    )

    ax.set_title(f"k = {k}")
    ax.set_xlabel("PCA-komponent 1")

axes[0].set_ylabel("PCA-komponent 2")
fig.suptitle("Kluster med inference-punkter (X) i PCA-rymd för valda k-värden", fontsize=16)
plt.tight_layout()
plt.savefig(outputs_figures / "cluster_scatter_pca_k234_with_inference.png", dpi=160, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spara resultat
# MAGIC
# MAGIC Utdata skrivs tillbaka till labbmappen så att det kan inspekteras eller laddas ner:
# MAGIC klustrade träningskunder, inference-resultat, modellvalsvärden, figurer och den tränade pipelinen.

# COMMAND ----------

# DBTITLE 1,Spara utdata
train_output_df.to_csv(outputs / "clustered_customers.csv", index=False)
inference_df.to_csv(outputs / "inference_results.csv", index=False)
model_selection_df.to_csv(outputs / "model_selection.csv", index=False)
joblib.dump(trained_pipelines[best_k], models / "clustering_pipeline.joblib")

print(f"Sparade output till: {outputs}")
print(f"Sparade bästa pipeline, k={best_k}, till: {models / 'clustering_pipeline.joblib'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Diskussionsfrågor
# MAGIC
# MAGIC - Vilket `k` skulle du välja om detta var ett CRM-segmenteringscase?
# MAGIC - Vilka kunder i inference-tabellen borde sälj eller marknad agera på först?
# MAGIC - Vilka extra features gjorde segmenten bättre eller sämre?
# MAGIC - Vilken data saknas för en mer användbar kundsegmentering?
# MAGIC - Är alla kluster tillräckligt stora och tydliga för att vara operationellt användbara?
