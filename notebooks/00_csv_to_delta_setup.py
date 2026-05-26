# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - CSV till Delta
# MAGIC
# MAGIC Kör detta först efter att projektmappen/zippen har laddats upp till Databricks.
# MAGIC Notebooken läser CSV-filerna i `data/raw` och skapar Delta-tabeller som används av notebook 01 och 02.

# COMMAND ----------

from pathlib import Path
import pandas as pd

# COMMAND ----------

def get_lab_root() -> Path:
    """Hitta den uppladdade labbroten från aktuell notebook-sökväg."""
    if "dbutils" in globals():
        notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        workspace_path = Path(notebook_path)
        if not str(workspace_path).startswith("/Workspace/"):
            workspace_path = Path("/Workspace") / str(workspace_path).lstrip("/")
        return workspace_path.parent.parent

    return Path(__file__).resolve().parent.parent


def pick_catalog(schema_name: str) -> str:
    """Använd helst Databricks Free-katalogen workspace, annars hive_metastore."""
    for catalog_name in ["workspace", "hive_metastore"]:
        try:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog_name}`.`{schema_name}`")
            return catalog_name
        except Exception as exc:
            print(f"Kunde inte använda catalog {catalog_name}: {exc}")
    raise RuntimeError("Ingen användbar catalog hittades. Testa att skapa schema manuellt i Databricks först.")


def table_id(catalog_name: str, schema_name: str, table_name: str) -> str:
    return f"`{catalog_name}`.`{schema_name}`.`{table_name}`"


lab_root = get_lab_root()
raw_path = lab_root / "data" / "raw"
target_schema = "kompetenskvall_labb"
spark_available = "spark" in globals()
target_catalog = pick_catalog(target_schema) if spark_available else None

print(f"Labbrot: {lab_root}")
print(f"Sökväg till rå CSV-data: {raw_path}")
if spark_available:
    print(f"Målschema: {target_catalog}.{target_schema}")
else:
    print("Spark är inte tillgängligt. Lokal körning validerar bara CSV-filerna.")

# COMMAND ----------

expected_files = ["transactions.csv", "regions.csv", "new_customers.csv"]
missing_files = [name for name in expected_files if not (raw_path / name).exists()]

if missing_files:
    raise FileNotFoundError(
        "Saknar CSV-filer i data/raw: "
        + ", ".join(missing_files)
        + f"\nFörväntad mapp: {raw_path}"
    )

csv_files = sorted(raw_path.glob("*.csv"))
print(f"Hittade {len(csv_files)} CSV-fil(er):")
for csv_file in csv_files:
    print(f"- {csv_file.name} ({csv_file.stat().st_size / (1024 * 1024):.2f} MB)")

# COMMAND ----------

results = []

for csv_file in csv_files:
    table_name = csv_file.stem.lower()
    if not spark_available:
        row_count = sum(1 for _ in csv_file.open("r", encoding="utf-8")) - 1
        results.append(
            {
                "file": csv_file.name,
                "table": None,
                "rows": row_count,
                "status": "validated_local_csv",
            }
        )
        print(f"Validerade {csv_file.name} ({row_count:,} rader)")
        continue

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
    print(f"Skapade/ersatte {target_catalog}.{target_schema}.{table_name} ({row_count:,} rader)")

results_df = pd.DataFrame(results)
if "display" in globals():
    display(results_df)
else:
    print(results_df)

# COMMAND ----------

if spark_available:
    spark.sql(f"SHOW TABLES IN `{target_catalog}`.`{target_schema}`")
