import logging
import pandas as pd
from pathlib import Path

# -----------------------------------------
# Logging
# -----------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------
# Config - Đường dẫn tương đối từ thư mục ETL
# -----------------------------------------
# Lấy thư mục gốc của project (DW-DSS)
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "Dataset" / "Raw"

FILES = {
    "balance_sheet": "googl_balance_sheet.csv",
    "income_statement": "googl_income_statement.csv",
    "cash_flow": "googl_cash_flow_statement.csv",
    "prices": "googl_daily_prices.csv",
}

# -----------------------------------------
# Helpers
# -----------------------------------------
def check_null_columns(df, name):
    """In thông tin các cột bị thiếu dữ liệu"""
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]

    if null_cols.empty:
        logger.info(f"[{name}] No missing values.")
        return

    logger.warning(f"[{name}] Columns with missing values:")

    missing_stats = (
        pd.DataFrame({
            "Missing Count": null_cols,
            "Percentage": (null_cols / len(df) * 100).round(1)
        })
        .sort_values("Missing Count", ascending=False)
    )

    logger.warning("\n" + missing_stats.head(5).to_string())


def preprocess_raw(df):
    """Tiền xử lý cơ bản trước khi Transform"""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.dropna(axis=1, how="all")
    df = df.drop_duplicates()
    return df

# -----------------------------------------
# Main EXTRACT function
# -----------------------------------------
def extract_data():
    logger.info("Starting EXTRACT step...")
    logger.info(f"Reading from: {RAW_DIR}")

    data_frames = {}

    for name, filename in FILES.items():
        file_path = RAW_DIR / filename

        try:
            df = pd.read_csv(file_path, na_values=["None"])

            logger.info(f"[{name}] Loaded successfully → shape = {df.shape}")

            # Preprocess nhẹ
            df = preprocess_raw(df)

            # Check nulls
            total_nulls = df.isnull().sum().sum()
            if total_nulls == 0:
                logger.info(f"[{name}] Clean dataset (no missing values).")
            else:
                logger.warning(f"[{name}] Missing values detected: {total_nulls} cells.")
                check_null_columns(df, name)

            data_frames[name] = df

        except FileNotFoundError:
            logger.error(f"[{name}] File NOT FOUND → {file_path}")
        except Exception as e:
            logger.error(f"[{name}] Error loading file: {e}")

    logger.info("EXTRACT step completed.")
    return data_frames


# -----------------------------------------
# Run standalone for debugging
# -----------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    extract_data()