# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Prepare and explore
# MAGIC
# MAGIC This notebook expects notebook 00 to have created raw Delta tables.
# MAGIC It builds customer-level training data, saves it as Delta, and writes a local CSV for notebook 02.

# COMMAND ----------

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

def get_lab_root() -> Path:
    if "dbutils" in globals():
        notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        workspace_path = Path(notebook_path)
        if not str(workspace_path).startswith("/Workspace/"):
            workspace_path = Path("/Workspace") / str(workspace_path).lstrip("/")
        return workspace_path.parent.parent

    return Path(__file__).resolve().parent.parent


def resolve_catalog(schema_name: str) -> str:
    for catalog_name in ["workspace", "hive_metastore"]:
        try:
            spark.sql(f"DESCRIBE SCHEMA `{catalog_name}`.`{schema_name}`")
            return catalog_name
        except Exception:
            pass
    raise RuntimeError("Could not find schema. Run notebook 00 first.")


def table_id(table_name: str) -> str:
    return f"`{target_catalog}`.`{target_schema}`.`{table_name}`"


lab_root = get_lab_root()
raw_path = lab_root / "data" / "raw"
outputs = lab_root / "outputs"
outputs_figures = outputs / "figures"
processed_path = lab_root / "data" / "processed"

outputs_figures.mkdir(parents=True, exist_ok=True)
processed_path.mkdir(parents=True, exist_ok=True)

target_schema = "kompetenskvall_labb"
spark_available = "spark" in globals()
target_catalog = resolve_catalog(target_schema) if spark_available else None

print(f"Lab root: {lab_root}")
if spark_available:
    print(f"Using schema: {target_catalog}.{target_schema}")
else:
    print("Spark is not available. Using local CSV fallback.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check input quality
# MAGIC
# MAGIC Before building any model input, we check that the transaction table has the basics we need:
# MAGIC customer id, country, and non-negative revenue. This is the same kind of sanity check you would do before
# MAGIC building a BI dataset or semantic model.

# COMMAND ----------

# DBTITLE 1,Data quality check
if spark_available:
    quality_df = spark.sql(f"""
        SELECT
            'transactions' AS table_name,
            COUNT(*) AS rows,
            SUM(CASE WHEN CustomerID IS NULL THEN 1 ELSE 0 END) AS missing_customer_id,
            SUM(CASE WHEN Country IS NULL THEN 1 ELSE 0 END) AS missing_country,
            SUM(CASE WHEN Revenue IS NULL OR Revenue < 0 THEN 1 ELSE 0 END) AS invalid_revenue
        FROM {table_id("transactions")}
    """).toPandas()
else:
    transactions_pdf = pd.read_csv(raw_path / "transactions.csv")
    quality_df = pd.DataFrame(
        [
            {
                "table_name": "transactions",
                "rows": len(transactions_pdf),
                "missing_customer_id": transactions_pdf["CustomerID"].isna().sum(),
                "missing_country": transactions_pdf["Country"].isna().sum(),
                "invalid_revenue": (transactions_pdf["Revenue"].isna() | (transactions_pdf["Revenue"] < 0)).sum(),
            }
        ]
    )

if "display" in globals():
    display(quality_df)
else:
    print(quality_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build one row per customer
# MAGIC
# MAGIC The raw table has one row per transaction line. For segmentation we need one row per customer,
# MAGIC because the model should compare customer behavior, not individual purchases.
# MAGIC
# MAGIC The features below are similar to a BI customer mart:
# MAGIC
# MAGIC - `recency_days`: days since the customer's last purchase
# MAGIC - `frequency`: number of invoices
# MAGIC - `monetary`: total revenue
# MAGIC - basket and product features that describe how the customer buys

# COMMAND ----------

# DBTITLE 1,Build customer features
if spark_available:
    customer_features_df = spark.sql(f"""
    WITH cleaned AS (
        SELECT
            t.CustomerID,
            t.Country,
            COALESCE(r.RegionGroup, 'Other') AS RegionGroup,
                t.InvoiceDate,
                t.InvoiceNo,
                t.Quantity,
                t.UnitPrice,
                t.Revenue
            FROM {table_id("transactions")} t
            LEFT JOIN {table_id("regions")} r
                ON t.Country = r.Country
        WHERE t.CustomerID IS NOT NULL
          AND t.Revenue IS NOT NULL
          AND t.Revenue >= 0
    ),
    snapshot AS (
        SELECT DATE_ADD(MAX(InvoiceDate), 1) AS snapshot_date
        FROM cleaned
    ),
    aggregated AS (
        SELECT
            CustomerID,
            Country,
            RegionGroup,
            MAX(InvoiceDate) AS last_purchase_date,
            DATEDIFF((SELECT snapshot_date FROM snapshot), MAX(InvoiceDate)) AS recency_days,
            COUNT(DISTINCT InvoiceNo) AS frequency,
            ROUND(SUM(Revenue), 2) AS monetary,
            ROUND(AVG(Revenue), 2) AS avg_order_value,
                SUM(Quantity) AS basket_size,
                ROUND(AVG(Quantity), 2) AS avg_items_per_invoice,
                COUNT(DISTINCT CONCAT(InvoiceNo, '-', COALESCE(CAST(Quantity AS STRING), 'NA'))) AS unique_products,
                ROUND(AVG(UnitPrice), 2) AS avg_unit_price
            FROM cleaned
            GROUP BY CustomerID, Country, RegionGroup
        )
        SELECT * FROM aggregated
    """)

    (
        customer_features_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_id("customer_enriched"))
    )

    customer_count = customer_features_df.count()
    print(f"Saved {customer_count:,} customers to {target_catalog}.{target_schema}.customer_enriched")
    customer_features_pdf = customer_features_df.toPandas()
else:
    regions_pdf = pd.read_csv(raw_path / "regions.csv")
    cleaned_pdf = (
        transactions_pdf.merge(regions_pdf[["Country", "RegionGroup"]], on="Country", how="left")
        .assign(RegionGroup=lambda df: df["RegionGroup"].fillna("Other"))
    )
    cleaned_pdf = cleaned_pdf[
        cleaned_pdf["CustomerID"].notna()
        & cleaned_pdf["Revenue"].notna()
        & (cleaned_pdf["Revenue"] >= 0)
    ].copy()
    cleaned_pdf["InvoiceDate"] = pd.to_datetime(cleaned_pdf["InvoiceDate"])
    snapshot_date = cleaned_pdf["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    customer_features_pdf = (
        cleaned_pdf.groupby(["CustomerID", "Country", "RegionGroup"], as_index=False)
        .agg(
            last_purchase_date=("InvoiceDate", "max"),
            frequency=("InvoiceNo", "nunique"),
            monetary=("Revenue", "sum"),
            avg_order_value=("Revenue", "mean"),
            basket_size=("Quantity", "sum"),
            avg_items_per_invoice=("Quantity", "mean"),
            avg_unit_price=("UnitPrice", "mean"),
        )
    )
    customer_features_pdf["recency_days"] = (snapshot_date - customer_features_pdf["last_purchase_date"]).dt.days
    unique_products_pdf = (
        cleaned_pdf.assign(invoice_quantity=cleaned_pdf["InvoiceNo"].astype(str) + "-" + cleaned_pdf["Quantity"].astype(str))
        .groupby(["CustomerID", "Country", "RegionGroup"])["invoice_quantity"]
        .nunique()
        .rename("unique_products")
        .reset_index()
    )
    customer_features_pdf = customer_features_pdf.merge(
        unique_products_pdf,
        on=["CustomerID", "Country", "RegionGroup"],
        how="left",
    )
    customer_features_pdf = customer_features_pdf[
        [
            "CustomerID",
            "Country",
            "RegionGroup",
            "last_purchase_date",
            "recency_days",
            "frequency",
            "monetary",
            "avg_order_value",
            "basket_size",
            "avg_items_per_invoice",
            "unique_products",
            "avg_unit_price",
        ]
    ]
    customer_features_pdf = customer_features_pdf.round(
        {
            "monetary": 2,
            "avg_order_value": 2,
            "avg_items_per_invoice": 2,
            "avg_unit_price": 2,
        }
    )
    print(f"Built {len(customer_features_pdf):,} customers from local CSV files")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the prepared training data
# MAGIC
# MAGIC Notebook 02 uses this customer-level dataset for training. We save it both as a Delta table and as a CSV copy.
# MAGIC The CSV is practical for pandas/scikit-learn, while Delta keeps the Databricks workflow visible.

# COMMAND ----------

# DBTITLE 1,Save CSV copy for pandas workflow
customer_features_csv = processed_path / "customer_enriched.csv"
customer_features_pdf.to_csv(customer_features_csv, index=False)

print(f"Saved CSV copy to: {customer_features_csv}")
if "display" in globals():
    display(customer_features_pdf.head())
else:
    print(customer_features_pdf.head())

# COMMAND ----------

missing_values_df = (
    customer_features_pdf.isna()
    .sum()
    .sort_values(ascending=False)
    .rename("missing_values")
    .to_frame()
)
if "display" in globals():
    display(missing_values_df)
else:
    print(missing_values_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore feature relationships
# MAGIC
# MAGIC The heatmap shows which numeric features tend to move together. Strong correlation is not automatically bad,
# MAGIC but it tells us that several columns may describe similar behavior. In the next notebook, changing the feature
# MAGIC list changes what the clustering model considers "similar customers".

# COMMAND ----------

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

plt.figure(figsize=(10, 8))
corr_matrix = customer_features_pdf[numeric_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(outputs_figures / "correlation_heatmap.png", dpi=100, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Output from notebook 01:
# MAGIC - Delta table: `kompetenskvall_labb.customer_enriched`
# MAGIC - CSV file: `data/processed/customer_enriched.csv`
# MAGIC - Figure: `outputs/figures/correlation_heatmap.png`
