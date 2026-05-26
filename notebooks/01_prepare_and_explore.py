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
    try:
        notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        workspace_path = Path(notebook_path)
        if not str(workspace_path).startswith("/Workspace/"):
            workspace_path = Path("/Workspace") / str(workspace_path).lstrip("/")
        return workspace_path.parent.parent
    except Exception:
        return Path.cwd().parent


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
outputs = lab_root / "outputs"
outputs_figures = outputs / "figures"
processed_path = lab_root / "data" / "processed"

outputs_figures.mkdir(parents=True, exist_ok=True)
processed_path.mkdir(parents=True, exist_ok=True)

target_schema = "kompetenskvall_labb"
target_catalog = resolve_catalog(target_schema)

print(f"Lab root: {lab_root}")
print(f"Using schema: {target_catalog}.{target_schema}")

# COMMAND ----------

# DBTITLE 1,Data quality check
quality_df = spark.sql(f"""
    SELECT
        'transactions' AS table_name,
        COUNT(*) AS rows,
        SUM(CASE WHEN CustomerID IS NULL THEN 1 ELSE 0 END) AS missing_customer_id,
        SUM(CASE WHEN Country IS NULL THEN 1 ELSE 0 END) AS missing_country,
        SUM(CASE WHEN Revenue IS NULL OR Revenue < 0 THEN 1 ELSE 0 END) AS invalid_revenue
    FROM {table_id("transactions")}
""").toPandas()

display(quality_df)

# COMMAND ----------

# DBTITLE 1,Build customer features
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
    aggregated AS (
        SELECT
            CustomerID,
            Country,
            RegionGroup,
            MAX(InvoiceDate) AS last_purchase_date,
            DATEDIFF(CURRENT_DATE(), MAX(InvoiceDate)) AS recency_days,
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

# COMMAND ----------

# DBTITLE 1,Save CSV copy for pandas workflow
customer_features_pdf = customer_features_df.toPandas()
customer_features_csv = processed_path / "customer_enriched.csv"
customer_features_pdf.to_csv(customer_features_csv, index=False)

print(f"Saved CSV copy to: {customer_features_csv}")
display(customer_features_pdf.head())

# COMMAND ----------

missing_values_df = (
    customer_features_pdf.isna()
    .sum()
    .sort_values(ascending=False)
    .rename("missing_values")
    .to_frame()
)
display(missing_values_df)

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
