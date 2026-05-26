# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Train and infer
# MAGIC
# MAGIC This notebook expects notebook 01 to have created customer training data.
# MAGIC It compares k=2,3,4, trains KMeans pipelines, and predicts clusters for new customers.

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
    try:
        notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        workspace_path = Path(notebook_path)
        if not str(workspace_path).startswith("/Workspace/"):
            workspace_path = Path("/Workspace") / str(workspace_path).lstrip("/")
        return workspace_path.parent.parent
    except Exception:
        return Path.cwd().parent


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

print(f"Lab root: {lab_root}")
print(f"Using schema: {target_catalog}.{target_schema}" if target_catalog else "No Delta schema found; using CSV files.")

# COMMAND ----------

# DBTITLE 1,Read training data
if target_catalog:
    train_df = spark.table(table_id("customer_enriched")).toPandas()
else:
    train_csv = processed_path / "customer_enriched.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}. Run notebook 01 first.")
    train_df = pd.read_csv(train_csv)

train_df["last_purchase_date"] = pd.to_datetime(train_df["last_purchase_date"])
display(train_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silhouette score short explanation
# MAGIC
# MAGIC Silhouette score measures how clearly one point belongs to its own cluster compared with the nearest other cluster.
# MAGIC
# MAGIC - close to `1.0`: clearer separation
# MAGIC - close to `0.0`: overlapping clusters
# MAGIC - below `0.0`: many points may be assigned poorly

# COMMAND ----------

# DBTITLE 1,Define preprocessing pipeline
numeric_features = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "basket_size",
    "avg_items_per_invoice",
    "unique_products",
    "avg_unit_price",
]
categorical_features = ["RegionGroup"]
feature_columns = numeric_features + categorical_features
X_train = train_df[feature_columns].copy()

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

# DBTITLE 1,Train models and compare k
k_values = [2, 3, 4]
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

    trained_pipelines[k] = pipeline
    train_cluster_by_k[k] = train_clusters
    model_selection_rows.append({"k": k, "silhouette_score": score})

model_selection_df = pd.DataFrame(model_selection_rows).sort_values("k")
best_k = int(model_selection_df.loc[model_selection_df["silhouette_score"].idxmax(), "k"])
display(model_selection_df)

# COMMAND ----------

# DBTITLE 1,Create output dataframe with cluster assignments
train_output_df = train_df.copy()
for k in k_values:
    train_output_df[f"Cluster_k{k}"] = train_cluster_by_k[k]

display(train_output_df.head())

# COMMAND ----------

# DBTITLE 1,Plot silhouette scores
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(9, 5))
sns.lineplot(data=model_selection_df, x="k", y="silhouette_score", marker="o", ax=ax, color="#0b5cad")
ax.axvline(best_k, color="#d62728", linestyle="--", linewidth=1)
ax.set_title("Silhouette score by number of clusters")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Silhouette score")
ax.set_xticks(k_values)
for _, row in model_selection_df.iterrows():
    ax.annotate(
        f"{row['silhouette_score']:.3f}",
        (row["k"], row["silhouette_score"]),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=10,
    )
plt.tight_layout()
plt.savefig(outputs_figures / "silhouette_scores.png", dpi=160, bbox_inches="tight")
plt.show()

print(f"Selected k by highest silhouette score: {best_k}")

# COMMAND ----------

# DBTITLE 1,Read new customers and regions
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
display(inference_df)

# COMMAND ----------

# DBTITLE 1,Predict clusters for new customers
for k in k_values:
    inference_df[f"PredictedCluster_k{k}"] = trained_pipelines[k].predict(inference_df[feature_columns])

display(inference_df)

# COMMAND ----------

# DBTITLE 1,Cluster profile for k=3
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
display(k3_profile_df)

# COMMAND ----------

# DBTITLE 1,Extended cluster profile visualization
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
    "avg_order_value": "Avg order value",
    "basket_size": "Basket size",
    "unique_products": "Unique products",
    "avg_items_per_invoice": "Avg items/invoice",
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
plt.title("k=3 cluster profile across multiple features (index vs total)")
plt.xlabel("Feature")
plt.ylabel("Relative index (1.0 = overall mean)")
plt.xticks(rotation=20, ha="right")
plt.legend(title="Cluster", loc="upper right")
plt.tight_layout()
plt.savefig(outputs_figures / "k3_cluster_feature_bars.png", dpi=160, bbox_inches="tight")
plt.show()

# COMMAND ----------

# DBTITLE 1,Cluster scatter plots with inference points
plot_x = "recency_days"
plot_y = "monetary"

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, k in zip(axes, k_values):
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
    ax.set_xlabel("Recency (days)")

axes[0].set_ylabel("Monetary (log scale)")
fig.suptitle("Clusters with inference points (X) for k = 2, 3, 4", fontsize=16)
plt.tight_layout()
plt.savefig(outputs_figures / "cluster_scatter_k234_with_inference.png", dpi=160, bbox_inches="tight")
plt.show()

# COMMAND ----------

# DBTITLE 1,PCA visualization with inference points
best_preprocessor = trained_pipelines[best_k].named_steps["preprocessor"]
X_train_prepared = best_preprocessor.transform(X_train)
X_inference_prepared = best_preprocessor.transform(inference_df[feature_columns])

pca_model = PCA(n_components=2, random_state=42)
train_pca_values = pca_model.fit_transform(X_train_prepared)
inference_pca_values = pca_model.transform(X_inference_prepared)

train_pca_df = pd.DataFrame(train_pca_values, columns=["PC1", "PC2"])
inference_pca_df = pd.DataFrame(inference_pca_values, columns=["PC1", "PC2"])

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
for ax, k in zip(axes, k_values):
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
    ax.set_xlabel("PCA component 1")

axes[0].set_ylabel("PCA component 2")
fig.suptitle("Clusters with inference points (X) in PCA space for k = 2, 3, 4", fontsize=16)
plt.tight_layout()
plt.savefig(outputs_figures / "cluster_scatter_pca_k234_with_inference.png", dpi=160, bbox_inches="tight")
plt.show()

# COMMAND ----------

# DBTITLE 1,Save outputs
train_output_df.to_csv(outputs / "clustered_customers.csv", index=False)
inference_df.to_csv(outputs / "inference_results.csv", index=False)
model_selection_df.to_csv(outputs / "model_selection.csv", index=False)
joblib.dump(trained_pipelines[best_k], models / "clustering_pipeline.joblib")

print(f"Saved outputs to: {outputs}")
print(f"Saved best pipeline, k={best_k}, to: {models / 'clustering_pipeline.joblib'}")
