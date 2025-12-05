import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Config - Đường dẫn tương đối từ thư mục ETL
# -------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
CLEANED_DIR = BASE_DIR / "Dataset" / "Cleaned"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def convert_date(df, column_name):
    if column_name in df.columns:
        df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
        missing = df[column_name].isnull().sum()
        if missing > 0:
            logger.warning(f"[{column_name}] {missing} invalid date values → coerced to NaT")
    return df


def add_price_features(df):
    if {"close"}.issubset(df.columns):
        df = df.sort_values("date")
        df["pct_change"] = df["close"].pct_change()
        df["daily_return"] = df["close"].diff()
        df["volatility_5d"] = df["pct_change"].rolling(5).std()
        df["volatility_20d"] = df["pct_change"].rolling(20).std()
    else:
        logger.warning("[prices] Column 'close' not found → cannot compute returns.")
    return df


def fill_missing_numeric(df):
    num_cols = df.select_dtypes(include=["float", "int"]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled missing values in {col} using median = {median_val}")
    return df

# -------------------------------------------------
# Main TRANSFORM function
# -------------------------------------------------
def transform_data(raw):
    logger.info("Starting TRANSFORM step...")
    logger.info(f"Output directory: {CLEANED_DIR}")

    transformed = {}

    # ---------------------------------
    # 1. Balance Sheet
    # ---------------------------------
    bs = raw.get("balance_sheet").copy()
    bs = normalize_columns(bs)
    bs = fill_missing_numeric(bs)
    transformed["balance_sheet"] = bs
    logger.info("Transformed balance_sheet → OK")

    # ---------------------------------
    # 2. Income Statement
    # ---------------------------------
    inc = raw.get("income_statement").copy()
    inc = normalize_columns(inc)
    inc = fill_missing_numeric(inc)
    transformed["income_statement"] = inc
    logger.info("Transformed income_statement → OK")

    # ---------------------------------
    # 3. Cash Flow
    # ---------------------------------
    cf = raw.get("cash_flow").copy()
    cf = normalize_columns(cf)
    cf = fill_missing_numeric(cf)
    transformed["cash_flow"] = cf
    logger.info("Transformed cash_flow → OK")

    # ---------------------------------
    # 4. Daily Prices
    # ---------------------------------
    prices = raw.get("prices").copy()
    prices = normalize_columns(prices)
    prices = convert_date(prices, "date")
    prices = add_price_features(prices)
    transformed["prices"] = prices
    logger.info("Transformed prices → OK")

    logger.info("TRANSFORM step completed.")
    
    # ---------------------------------
    # Save to Cleaned directory
    # ---------------------------------
    save_cleaned_data(transformed)
    
    return transformed


def save_cleaned_data(transformed):
    """Lưu dữ liệu đã transform vào thư mục Cleaned"""
    logger.info("Saving cleaned data to parquet files...")
    
    for name, df in transformed.items():
        output_path = CLEANED_DIR / f"{name}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {name} → {output_path}")
    
    logger.info("All cleaned data saved successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("This file defines transform functions.")