# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - CSV to Delta setup
# MAGIC
# MAGIC Run this first after uploading the project folder/zip to Databricks.
# MAGIC It reads the CSV files in `data/raw` and creates Delta tables used by notebooks 01 and 02.

# COMMAND ----------

from pathlib import Path
import pandas as pd

# COMMAND ----------

def get_lab_root() -> Path:
    """Resolve the uploaded lab root from the current notebook path."""
    try:
        notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        workspace_path = Path(notebook_path)
        if not str(workspace_path).startswith("/Workspace/"):
            workspace_path = Path("/Workspace") / str(workspace_path).lstrip("/")
        return workspace_path.parent.parent
    except Exception:
        return Path.cwd().parent


def pick_catalog(schema_name: str) -> str:
    """Prefer the Databricks Free workspace catalog, fall back to hive_metastore."""
    for catalog_name in ["workspace", "hive_metastore"]:
        try:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog_name}`.`{schema_name}`")
            return catalog_name
        except Exception as exc:
            print(f"Could not use catalog {catalog_name}: {exc}")
    raise RuntimeError("No usable catalog found. Try creating a schema manually in Databricks first.")


def table_id(catalog_name: str, schema_name: str, table_name: str) -> str:
    return f"`{catalog_name}`.`{schema_name}`.`{table_name}`"


lab_root = get_lab_root()
raw_path = lab_root / "data" / "raw"
target_schema = "kompetenskvall_labb"
target_catalog = pick_catalog(target_schema)

print(f"Lab root: {lab_root}")
print(f"Raw CSV path: {raw_path}")
print(f"Target schema: {target_catalog}.{target_schema}")

# COMMAND ----------

expected_files = ["transactions.csv", "regions.csv", "new_customers.csv"]
missing_files = [name for name in expected_files if not (raw_path / name).exists()]

if missing_files:
    raise FileNotFoundError(
        "Missing CSV files in data/raw: "
        + ", ".join(missing_files)
        + f"\nExpected folder: {raw_path}"
    )

csv_files = sorted(raw_path.glob("*.csv"))
print(f"Found {len(csv_files)} CSV file(s):")
for csv_file in csv_files:
    print(f"- {csv_file.name} ({csv_file.stat().st_size / (1024 * 1024):.2f} MB)")

# COMMAND ----------

results = []

for csv_file in csv_files:
    table_name = csv_file.stem.lower()
    full_table_name = table_id(target_catalog, target_schema, table_name)

    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(str(csv_file))
    )

    row_count = df.count()
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(full_table_name)
    )

    results.append(
        {
            "file": csv_file.name,
            "table": f"{target_catalog}.{target_schema}.{table_name}",
            "rows": row_count,
            "status": "created_or_replaced",
        }
    )
    print(f"Created/replaced {target_catalog}.{target_schema}.{table_name} ({row_count:,} rows)")

results_df = pd.DataFrame(results)
display(results_df)

# COMMAND ----------

spark.sql(f"SHOW TABLES IN `{target_catalog}`.`{target_schema}`")
