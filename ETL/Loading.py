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
# CONSTANTS
# -----------------------------------------
FIN_TYPE_MAPPING = {
    "balance_sheet": 1,
    "income_statement": 2,
    "cash_flow": 3
}

STOCK_METRIC_MAPPING = {
    'open': 1, 'high': 2, 'low': 3, 'close': 4, 'volume': 5,
    'pct_change': 6, 'daily_return': 7, 'volatility_5d': 8, 'volatility_20d': 9
}

# -----------------------------------------
# DATABASE CONNECTION MANAGER
# -----------------------------------------
class DatabaseConnection:
    """Context manager for DuckDB connection"""
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
    
    def __enter__(self):
        self.conn = duckdb.connect(str(self.db_path))
        logger.info(f"Connected to database: {self.db_path}")
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
        if exc_type:
            logger.error(f"Error in database operation: {exc_val}")
        return False

# -----------------------------------------
# VALIDATION FUNCTIONS
# -----------------------------------------
def validate_cleaned_data(df, df_name):
    """Validate cleaned data has required columns"""
    if 'date' not in df.columns:
        raise ValueError(f"{df_name} missing 'date' column")
    
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        logger.warning(f"   Converting 'date' to datetime in {df_name}")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    null_dates = df['date'].isnull().sum()
    if null_dates > 0:
        logger.warning(f"   {df_name}: {null_dates} null dates found")
    
    if 'symbol' not in df.columns:
        df['symbol'] = 'GOOGL'
        logger.info(f"   {df_name}: Added default symbol 'GOOGL'")
    
    logger.info(f"   ✓ Validated {df_name}: {len(df):,} rows")
    return df

# -----------------------------------------
# CREATE STAR SCHEMA
# -----------------------------------------
def create_star_schema(conn):
    """Tạo schema cho Data Warehouse từ file SQL_script.sql"""
    schema_file = BASE_DIR / "SQL_script.sql"
    
    if not schema_file.exists():
        logger.error(f"SQL_script.sql not found at {schema_file}")
        raise FileNotFoundError(f"SQL_script.sql not found at {schema_file}")

    logger.info("🎯 Creating Data Warehouse Schema...")
    logger.info(f"Schema file: {schema_file}")
    
    try:
        with open(schema_file, "r", encoding="utf-8") as f:
            sql_script = f.read()
            conn.execute(sql_script)
        logger.info("✅ Star Schema created successfully.")
    except Exception as e:
        logger.error(f"Failed to create schema: {e}")
        raise

# ---------------------------------------------------------
# BUILD DIMENSION TABLES
# ---------------------------------------------------------
def build_dim_date(prices, bs, inc, cf):
    """Build date dimension from prices"""
    logger.info("   Building dim_date...")
    
    all_dates = pd.concat([
        prices[['date']],
        bs[['date']],
        inc[['date']],
        cf[['date']]
    ]).drop_duplicates().dropna()
    
    if all_dates.empty:
        raise ValueError("No valid dates found in prices")
    
    df = all_dates.copy()
    df['date_key'] = df['date'].dt.strftime("%Y%m%d").astype(int)
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    df['quarter'] = "Q" + df['date'].dt.quarter.astype(str)
    df['year'] = df['date'].dt.year
    
    result = df[['date_key', 'date', 'day', 'month', 'month_name', 'quarter', 'year']]
    logger.info(f"   ✓ Built dim_date: {len(result):,} unique dates "
                f"(from {result['date'].min().date()} to {result['date'].max().date()})")
    return result

def build_dim_company(bs, inc, cf):
    """Build company dimension from financial statements"""
    logger.info("   Building dim_company...")
    
    # Validate symbol column exists
    for name, df in [("balance_sheet", bs), ("income_statement", inc), ("cash_flow", cf)]:
        if 'symbol' not in df.columns:
            raise ValueError(f"{name} missing 'symbol' column")
    
    df = pd.concat([
        bs[['symbol']], 
        inc[['symbol']], 
        cf[['symbol']]
    ]).drop_duplicates().dropna()
    
    if df.empty:
        raise ValueError("No valid company symbols found")
    
    df = df.reset_index(drop=True)
    df['company_key'] = df.index + 1
    df['ticker'] = df['symbol']
    df['company_name'] = df['symbol']
    df['industry'] = "Technology"
    df['exchange'] = "NASDAQ"
    
    result = df[['company_key', 'ticker', 'company_name', 'industry', 'exchange']]
    logger.info(f"   ✓ Built dim_company: {len(result)} companies")
    return result

def extract_metrics_from_financial(df, statement_type):
    """Extract financial metrics as dimension"""
    metric_cols = [c for c in df.columns if c not in ['symbol', 'date', 'reportedcurrency', 'fiscaldateending']]
    
    if not metric_cols:
        logger.warning(f"No metric columns found in {statement_type}")
        return pd.DataFrame(columns=['metric_key', 'metric_name', 'metric_group', 'unit'])
    
    out = pd.DataFrame({
        "metric_name": metric_cols,
        "metric_group": statement_type,
        "unit": "USD"
    })
    out['metric_key'] = out.index + 1

    return out[['metric_key', 'metric_name', 'metric_group', 'unit']]

def build_dim_fin_metric(bs, inc, cf):
    """Build financial metric dimension"""
    logger.info("   Building dim_fin_metric...")
    
    bs_m = extract_metrics_from_financial(bs, "balance_sheet")
    inc_m = extract_metrics_from_financial(inc, "income_statement")
    cf_m = extract_metrics_from_financial(cf, "cash_flow")

    result = pd.concat([bs_m, inc_m, cf_m]).drop_duplicates('metric_name').reset_index(drop=True)
    result['metric_key'] = result.index + 1  # Re-index after concat
    
    logger.info(f"   ✓ Built dim_fin_metric: {len(result)} metrics")
    return result

# ---------------------------------------------------------
# BUILD FACT TABLES
# ---------------------------------------------------------
def build_fact_finance(bs, inc, cf, dim_company, dim_fin_metric, dim_date):
    """Build fact table for financial data"""
    logger.info("   Building fact_finance...")
    
    def melt_financial(df, statement_name, fin_type_key):
        metric_cols = [c for c in df.columns if c not in ['symbol', 'date', 'reportedcurrency', 'fiscaldateending']]
        
        if not metric_cols:
            logger.warning(f"No metric columns in {statement_name}")
            return pd.DataFrame(columns=['date_key', 'company_key', 'fin_type_key', 'metric_key', 'value'])
        
        melted = df.melt(
            id_vars=['symbol', 'date'], 
            value_vars=metric_cols,
            var_name='metric_name', 
            value_name='value'
        )
        
        # Remove null values
        melted = melted.dropna(subset=['value'])
        
        merged = (
            melted
            .merge(dim_company[['company_key', 'ticker']], left_on='symbol', right_on='ticker', how='inner')
            .merge(dim_fin_metric[['metric_key', 'metric_name']], on='metric_name', how='inner')
            .merge(dim_date[['date_key', 'date']], on='date', how='inner')
        )
        merged['fin_type_key'] = fin_type_key
        
        logger.info(f"      - {statement_name}: {len(merged)} records")
        return merged[['date_key', 'company_key', 'fin_type_key', 'metric_key', 'value']]

    bs_fact = melt_financial(bs, "balance_sheet", FIN_TYPE_MAPPING["balance_sheet"])
    inc_fact = melt_financial(inc, "income_statement", FIN_TYPE_MAPPING["income_statement"])
    cf_fact = melt_financial(cf, "cash_flow", FIN_TYPE_MAPPING["cash_flow"])

    result = pd.concat([bs_fact, inc_fact, cf_fact], ignore_index=True)
    result['fact_finance_id'] = result.index + 1
    result = result[[
        'fact_finance_id', 
        'date_key', 
        'company_key', 
        'fin_type_key', 
        'metric_key', 
        'value'
    ]]
    logger.info(f"   ✓ Built fact_finance: {len(result)} total records")
    return result

def build_fact_stock_prices(prices, dim_company, dim_date):
    """Build fact table for stock prices"""
    logger.info("   Building fact_stock_prices...")
    
    metric_cols = [col for col in STOCK_METRIC_MAPPING.keys() if col in prices.columns]
    
    if not metric_cols:
        logger.warning("No stock metric columns found")
        return pd.DataFrame(columns=['date_key', 'company_key', 'stock_metric_key', 'value'])
    
    melted = prices.melt(
        id_vars=['symbol', 'date'], 
        value_vars=metric_cols,
        var_name='metric', 
        value_name='value'
    )
    
    # Remove null values
    melted = melted.dropna(subset=['value'])
    
    # Map metric names to keys
    melted['stock_metric_key'] = melted['metric'].map(STOCK_METRIC_MAPPING)
    
    merged = (
        melted
        .merge(dim_company[['company_key', 'ticker']], left_on='symbol', right_on='ticker', how='inner')
        .merge(dim_date[['date_key', 'date']], on='date', how='inner')
    )
    merged['fact_stock_prices_id'] = merged.index + 1
    result = merged[[
        'fact_stock_prices_id',
        'date_key', 
        'company_key', 
        'stock_metric_key', 
        'value'
    ]]
    logger.info(f"   ✓ Built fact_stock_prices: {len(result)} records")
    return result

# ---------------------------------------------------------
# LOAD GOLD + INSERT INTO DW
# ---------------------------------------------------------
def load_gold_and_dw(conn, df, name):
    """Save to Gold layer and insert into Data Warehouse"""
    try:
        # Determine output directory
        gold_path = DIM_DIR / f"{name}.parquet" if name.startswith("dim") else FACT_DIR / f"{name}.parquet"
        
        # Save to Gold
        df.to_parquet(gold_path, index=False)
        logger.info(f"   💾 Saved to Gold: {gold_path.relative_to(BASE_DIR)}")
        
        # Insert to DW
        conn.execute(f"DELETE FROM {name}")
        conn.execute(f"INSERT INTO {name} SELECT * FROM df")
        
        # Verify
        count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        logger.info(f"   ✓ Loaded {name}: {count:,} rows to DW")
        
    except Exception as e:
        logger.error(f"Failed to load {name}: {e}")
        raise

# -----------------------------------------
# VERIFY DATA WAREHOUSE
# -----------------------------------------
def verify_gold_layer():
    """Verify Gold layer files"""
    logger.info("\n🔍 Verifying Gold Layer...")
    
    logger.info("   📁 Dims folder:")
    dim_files = list(DIM_DIR.glob("*.parquet"))
    if dim_files:
        for f in sorted(dim_files):
            df = pd.read_parquet(f)
            logger.info(f"      - {f.name:30s}: {len(df):6,} rows")
    else:
        logger.warning("      No dimension files found")
    
    logger.info("   📁 Facts folder:")
    fact_files = list(FACT_DIR.glob("*.parquet"))
    if fact_files:
        for f in sorted(fact_files):
            df = pd.read_parquet(f)
            logger.info(f"      - {f.name:30s}: {len(df):6,} rows")
    else:
        logger.warning("      No fact files found")
    
    logger.info("✅ Gold layer verified")

# -----------------------------------------
# VERIFY DATA WAREHOUSE
# -----------------------------------------
def verify_data_warehouse(conn):
    """Verify Data Warehouse content"""
    logger.info("\n🔍 Verifying Data Warehouse...")
    logger.info(f"Database: {DB_PATH.relative_to(BASE_DIR)}")
    
    tables = [
        "dim_date", "dim_company", "dim_fin_metric",
        "fact_finance", "fact_stock_prices"
    ]
    
    total_rows = 0
    for table in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info(f"   {table:20s}: {count:8,} rows")
            total_rows += count
        except Exception as e:
            logger.error(f"   {table:20s}: ERROR - {e}")
    
    logger.info(f"\n   Total rows in DW: {total_rows:,}")
    logger.info("✅ Data Warehouse verified")

# -----------------------------------------
# PIPELINE RUNNER
# -----------------------------------------
def run_loading_pipeline():
    """Run complete Loading pipeline"""
    logger.info("\n" + "=" * 70)
    logger.info("🚀 STARTING LOADING PIPELINE")
    logger.info("=" * 70 + "\n")

    try:
        logger.info("📂 Loading cleaned data...")

        required_files = {
            "balance_sheet": CLEANED_DIR / "balance_sheet.parquet",
            "income_statement": CLEANED_DIR / "income_statement.parquet",
            "cash_flow": CLEANED_DIR / "cash_flow.parquet",
            "prices": CLEANED_DIR / "prices.parquet"
        }

        for name, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")
        
        bs = pd.read_parquet(required_files["balance_sheet"])
        inc = pd.read_parquet(required_files["income_statement"])
        cf = pd.read_parquet(required_files["cash_flow"])
        prices = pd.read_parquet(required_files["prices"])
        
        logger.info("✅ All cleaned data loaded successfully\n")

        logger.info("🔍 Step 2: Validating data...")
        
        bs = validate_cleaned_data(bs, "balance_sheet")
        inc = validate_cleaned_data(inc, "income_statement")
        cf = validate_cleaned_data(cf, "cash_flow")
        prices = validate_cleaned_data(prices, "prices")
        
        logger.info("✅ Data validation completed\n")

        logger.info("📦 Step 3: Building dimension tables...")
        
        dim_date = build_dim_date(prices, bs, inc, cf)
        dim_company = build_dim_company(bs, inc, cf)
        dim_fin_metric = build_dim_fin_metric(bs, inc, cf)
        
        logger.info("✅ All dimensions built\n")

        logger.info("📦 Step 4: Building fact tables...")
        
        fact_fin = build_fact_finance(bs, inc, cf, dim_company, dim_fin_metric, dim_date)
        fact_price = build_fact_stock_prices(prices, dim_company, dim_date)
        
        logger.info("✅ All facts built\n")
        
        with DatabaseConnection(DB_PATH) as conn:
            logger.info("📤 Creating schema and loading data...\n")
            
            create_star_schema(conn)
            
            logger.info("Loading dimension tables...")
            load_gold_and_dw(conn, dim_date, "dim_date")
            load_gold_and_dw(conn, dim_company, "dim_company")
            load_gold_and_dw(conn, dim_fin_metric, "dim_fin_metric")
            
            logger.info("\nLoading fact tables...")
            load_gold_and_dw(conn, fact_fin, "fact_finance")
            load_gold_and_dw(conn, fact_price, "fact_stock_prices")
            
            logger.info("\n✅ All data loaded to Gold & DW\n")
            
            verify_gold_layer()
            verify_data_warehouse(conn)

        logger.info("\n" + "=" * 70)
        logger.info("🎉 LOADING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70 + "\n")

    except FileNotFoundError as e:
        logger.error(f"\n❌ File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"\n❌ Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"\n❌ Loading pipeline failed: {e}", exc_info=True)
        raise

# -----------------------------------------
# Run standalone
# -----------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    run_loading_pipeline()