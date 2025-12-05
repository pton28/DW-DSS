import duckdb
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# -----------------------------------------
# DIRECTORY SETUP - Đường dẫn tương đối từ thư mục ETL
# -----------------------------------------
BASE_DIR = Path(__file__).parent.parent
CLEANED_DIR = BASE_DIR / "Dataset" / "Cleaned"
GOLD_DIR = BASE_DIR / "Dataset" / "Gold"
DIM_DIR = GOLD_DIR / "Dims"
FACT_DIR = GOLD_DIR / "Facts"

DATA_WAREHOUSE_DIR = BASE_DIR / "Data_warehouse"
DATA_WAREHOUSE_DIR.mkdir(exist_ok=True)

# Tạo các thư mục Gold, Dims, Facts
for directory in [GOLD_DIR, DIM_DIR, FACT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_WAREHOUSE_DIR / "dw.duckdb"

# -----------------------------------------
# CONNECT TO DUCKDB
# -----------------------------------------
con = duckdb.connect(str(DB_PATH))


# -----------------------------------------
# CREATE STAR SCHEMA
# -----------------------------------------
def create_star_schema():
    """Tạo schema cho Data Warehouse từ file SQL_script.sql"""
    schema_file = BASE_DIR / "SQL_script.sql"
    
    if not schema_file.exists():
        logger.error(f"SQL_script.sql not found at {schema_file}")
        raise FileNotFoundError(f"SQL_script.sql not found at {schema_file}")

    logger.info("🎯 Creating Data Warehouse Schema...")
    logger.info(f"Schema file: {schema_file}")
    
    with open(schema_file, "r", encoding="utf-8") as f:
        sql_script = f.read()
        con.execute(sql_script)
    
    logger.info("✅ Star Schema created successfully.")


# -----------------------------------------
# LOAD DIMENSION TABLES
# -----------------------------------------
def load_dim_tables():
    """Load các bảng dimension từ Cleaned -> Gold/Dims -> DW"""
    logger.info("📦 Loading DIMENSION tables...")
    logger.info(f"Source: {CLEANED_DIR}")
    logger.info(f"Gold Dims: {DIM_DIR}")

    dim_files = {
        "dim_date": ["date_key", "date", "day", "month", "month_name", "quarter", "year"],
        "dim_company": ["company_key", "ticker", "company_name", "industry", "exchange"],
        "dim_fin_metric": ["metric_key", "metric_name", "metric_group", "unit"]
    }

    for dim_name, columns in dim_files.items():
        parquet_path = CLEANED_DIR / f"{dim_name}.parquet"
        
        if not parquet_path.exists():
            logger.warning(f"⚠ Skipping {dim_name}, file not found at {parquet_path}")
            continue

        # Đọc từ Cleaned
        df = pd.read_parquet(parquet_path)
        
        # Kiểm tra và lọc các cột cần thiết
        available_cols = [col for col in columns if col in df.columns]
        if len(available_cols) != len(columns):
            missing = set(columns) - set(available_cols)
            logger.warning(f"[{dim_name}] Missing columns: {missing}")
        
        df_filtered = df[available_cols]

        # Lưu vào Gold/Dims
        gold_path = DIM_DIR / f"{dim_name}.parquet"
        df_filtered.to_parquet(gold_path, index=False)
        logger.info(f"   💾 Saved to Gold: {gold_path.relative_to(BASE_DIR)}")

        # Insert vào DW
        con.execute(f"DELETE FROM {dim_name}")
        con.execute(
            f"INSERT INTO {dim_name} SELECT * FROM df_filtered"
        )

        row_count = con.execute(f"SELECT COUNT(*) FROM {dim_name}").fetchone()[0]
        logger.info(f"   ✓ Loaded {dim_name}: {row_count} rows")

    logger.info("✅ DIMENSION tables loaded.")


# -----------------------------------------
# LOAD FACT TABLES
# -----------------------------------------
def load_fact_tables():
    """Load các bảng fact từ Cleaned -> Gold/Facts -> DW"""
    logger.info("📦 Loading FACT tables...")
    logger.info(f"Source: {CLEANED_DIR}")
    logger.info(f"Gold Facts: {FACT_DIR}")

    fact_files = {
        "fact_finance": [
            "date_key", "company_key", "fin_type_key",
            "metric_key", "value"
        ],
        "fact_stock_prices": [
            "date_key", "company_key", "stock_metric_key", "value"
        ]
    }

    for fact_name, columns in fact_files.items():
        parquet_path = CLEANED_DIR / f"{fact_name}.parquet"
        
        if not parquet_path.exists():
            logger.warning(f"⚠ Skipping {fact_name}, file not found at {parquet_path}")
            continue

        # Đọc từ Cleaned
        df = pd.read_parquet(parquet_path)
        
        # Kiểm tra và lọc các cột cần thiết
        available_cols = [col for col in columns if col in df.columns]
        if len(available_cols) != len(columns):
            missing = set(columns) - set(available_cols)
            logger.warning(f"[{fact_name}] Missing columns: {missing}")
        
        df_filtered = df[available_cols]

        # Lưu vào Gold/Facts
        gold_path = FACT_DIR / f"{fact_name}.parquet"
        df_filtered.to_parquet(gold_path, index=False)
        logger.info(f"   💾 Saved to Gold: {gold_path.relative_to(BASE_DIR)}")

        # Insert vào DW
        con.execute(f"DELETE FROM {fact_name}")
        con.execute(
            f"INSERT INTO {fact_name} SELECT * FROM df_filtered"
        )

        row_count = con.execute(f"SELECT COUNT(*) FROM {fact_name}").fetchone()[0]
        logger.info(f"   ✓ Loaded {fact_name}: {row_count} rows")

    logger.info("✅ FACT tables loaded.")


# -----------------------------------------
# VERIFY DATA WAREHOUSE
# -----------------------------------------
def verify_data_warehouse():
    """Kiểm tra dữ liệu trong Data Warehouse"""
    logger.info("🔍 Verifying Data Warehouse...")
    logger.info(f"Database: {DB_PATH.relative_to(BASE_DIR)}")
    
    tables = [
        "dim_date", "dim_company", "dim_fin_metric",
        "fact_finance", "fact_stock_prices"
    ]
    
    for table in tables:
        try:
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"   {table}: {count:,} rows")
        except Exception as e:
            logger.error(f"   {table}: ERROR - {e}")
    
    logger.info("✅ Verification completed.")


# -----------------------------------------
# VERIFY GOLD LAYER
# -----------------------------------------
def verify_gold_layer():
    """Kiểm tra các file trong Gold layer"""
    logger.info("🔍 Verifying Gold Layer...")
    
    logger.info(f"   📁 Dims folder: {DIM_DIR.relative_to(BASE_DIR)}")
    dim_files = list(DIM_DIR.glob("*.parquet"))
    if dim_files:
        for file in dim_files:
            df = pd.read_parquet(file)
            logger.info(f"      - {file.name}: {len(df):,} rows")
    else:
        logger.warning("      No dimension files found")
    
    logger.info(f"   📁 Facts folder: {FACT_DIR.relative_to(BASE_DIR)}")
    fact_files = list(FACT_DIR.glob("*.parquet"))
    if fact_files:
        for file in fact_files:
            df = pd.read_parquet(file)
            logger.info(f"      - {file.name}: {len(df):,} rows")
    else:
        logger.warning("      No fact files found")
    
    logger.info("✅ Gold layer verification completed.")


# -----------------------------------------
# PIPELINE RUNNER
# -----------------------------------------
def run_loading_pipeline():
    """Chạy toàn bộ quá trình Loading"""
    logger.info("\n======================================")
    logger.info("🚀 STARTING LOADING PIPELINE")
    logger.info("======================================\n")

    try:
        create_star_schema()
        load_dim_tables()
        load_fact_tables()
        verify_gold_layer()
        verify_data_warehouse()
        
        logger.info("\n======================================")
        logger.info("✅ LOADING PIPELINE COMPLETED")
        logger.info("======================================\n")
        
    except Exception as e:
        logger.error(f"❌ Loading pipeline failed: {e}")
        raise
    finally:
        con.close()
        logger.info("Database connection closed.")


# -----------------------------------------
# Run standalone
# -----------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    run_loading_pipeline()