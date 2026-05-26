# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Förbered och utforska
# MAGIC
# MAGIC Den här notebooken förväntar sig att notebook 00 har skapat råa Delta-tabeller.
# MAGIC Den bygger träningsdata på kundnivå, sparar resultatet som Delta och skriver en lokal CSV-kopia för notebook 02.

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
    raise RuntimeError("Kunde inte hitta schema. Kör notebook 00 först.")


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

print(f"Labbrot: {lab_root}")
if spark_available:
    print(f"Använder schema: {target_catalog}.{target_schema}")
else:
    print("Spark är inte tillgängligt. Använder lokal CSV-fallback.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Kontrollera datakvalitet
# MAGIC
# MAGIC Innan vi bygger modellinput kontrollerar vi att transaktionstabellen har grunderna vi behöver:
# MAGIC kund-id, land och icke-negativ intäkt. Det är samma typ av rimlighetskontroll som man gör innan
# MAGIC man bygger en BI-datamängd eller semantisk modell.

# COMMAND ----------

# DBTITLE 1,Datakvalitetskontroll
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
# MAGIC ## Bygg en rad per kund
# MAGIC
# MAGIC Råtabellen har en rad per transaktionsrad. För segmentering behöver vi en rad per kund,
# MAGIC eftersom modellen ska jämföra kundbeteende och inte enskilda köp.
# MAGIC
# MAGIC Features nedan liknar en kundmart i BI:
# MAGIC
# MAGIC - `recency_days`: antal dagar sedan kundens senaste köp
# MAGIC - `frequency`: antal fakturor
# MAGIC - `monetary`: total intäkt
# MAGIC - korg- och produktfeatures som beskriver hur kunden köper

# COMMAND ----------

# DBTITLE 1,Bygg kundfeatures
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
    print(f"Sparade {customer_count:,} kunder till {target_catalog}.{target_schema}.customer_enriched")
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
    print(f"Byggde {len(customer_features_pdf):,} kunder från lokala CSV-filer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spara färdig träningsdata
# MAGIC
# MAGIC Notebook 02 använder den här kundnivåtabellen för träning. Vi sparar den både som Delta-tabell och som CSV-kopia.
# MAGIC CSV är praktiskt för pandas/scikit-learn, medan Delta gör Databricks-flödet tydligt.

# COMMAND ----------

# DBTITLE 1,Spara CSV-kopia för pandas-flödet
customer_features_csv = processed_path / "customer_enriched.csv"
customer_features_pdf.to_csv(customer_features_csv, index=False)

print(f"Sparade CSV-kopia till: {customer_features_csv}")
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
# MAGIC ## Utforska relationer mellan features
# MAGIC
# MAGIC Heatmapen visar vilka numeriska features som tenderar att röra sig tillsammans. Stark korrelation är inte automatiskt dåligt,
# MAGIC men det säger att flera kolumner kan beskriva liknande beteende. I nästa notebook ändrar feature-listan vad
# MAGIC klustringsmodellen betraktar som "liknande kunder".

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
plt.title("Korrelationsmatris för features")
plt.tight_layout()
plt.savefig(outputs_figures / "correlation_heatmap.png", dpi=100, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Utdata från notebook 01:
# MAGIC - Delta-tabell: `kompetenskvall_labb.customer_enriched`
# MAGIC - CSV-fil: `data/processed/customer_enriched.csv`
# MAGIC - Figur: `outputs/figures/correlation_heatmap.png`
