import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Thêm thư mục ETL vào Python path để import được các module
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from Extracting import extract_data
from Trainsforming import transform_data
from Loading import run_loading_pipeline

# -----------------------------------------
# Logging Configuration
# -----------------------------------------
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Tạo tên file log với timestamp
log_filename = LOG_DIR / f"etl_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------
# Preprocessing Helpers
# -----------------------------------------
def preprocess_dataframe(df):
    """
    Clean raw dataframe before Transform step.
    Applied to all raw files.
    """
    original_shape = df.shape
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Log changes
    if df.shape != original_shape:
        logger.info(f"   Preprocessing: {original_shape} → {df.shape} "
                   f"(removed {original_shape[0] - df.shape[0]} duplicates, "
                   f"{original_shape[1] - df.shape[1]} empty columns)")

    return df


# -----------------------------------------
# ETL Pipeline Validation
# -----------------------------------------
def validate_environment():
    """Kiểm tra môi trường trước khi chạy ETL"""
    logger.info("🔍 Validating environment...")
    
    required_dirs = [
        BASE_DIR / "Dataset" / "Raw",
        BASE_DIR / "Dataset" / "Cleaned",
        BASE_DIR / "Dataset" / "Gold" / "Dims",
        BASE_DIR / "Dataset" / "Gold" / "Facts",
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not directory.exists():
            missing_dirs.append(directory)
            directory.mkdir(parents=True, exist_ok=True)
            logger.warning(f"   Created missing directory: {directory.relative_to(BASE_DIR)}")
    
    # Check SQL_script.sql
    schema_path = BASE_DIR / "SQL_script.sql"
    if not schema_path.exists():
        logger.error(f"   ❌ SQL_script.sql not found at {schema_path}")
        return False
    
    # Check raw data files
    raw_dir = BASE_DIR / "Dataset" / "Raw"
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"   ❌ No CSV files found in {raw_dir}")
        return False
    
    logger.info(f"   ✓ Found {len(csv_files)} raw data files")
    logger.info(f"   ✓ Schema file: {schema_path.relative_to(BASE_DIR)}")
    logger.info("✅ Environment validation passed")
    return True


def print_pipeline_summary(raw_data, transformed_data):
    """In tóm tắt kết quả ETL"""
    logger.info("\n" + "=" * 70)
    logger.info("📊 ETL PIPELINE SUMMARY")
    logger.info("=" * 70)
    
    # Extract summary
    logger.info("\n📥 EXTRACT Phase:")
    for name, df in raw_data.items():
        logger.info(f"   • {name:20s}: {df.shape[0]:6,} rows × {df.shape[1]:3} columns")
    
    # Transform summary
    logger.info("\n⚙️  TRANSFORM Phase:")
    for name, df in transformed_data.items():
        logger.info(f"   • {name:20s}: {df.shape[0]:6,} rows × {df.shape[1]:3} columns")
    
    # File locations
    logger.info("\n📁 Output Locations:")
    logger.info(f"   • Cleaned data    : Dataset/Cleaned/")
    logger.info(f"   • Gold Dimensions : Dataset/Gold/Dims/")
    logger.info(f"   • Gold Facts      : Dataset/Gold/Facts/")
    logger.info(f"   • Data Warehouse  : Data_warehouse/dw.duckdb")
    logger.info(f"   • Log file        : {log_filename.relative_to(BASE_DIR)}")
    
    logger.info("\n" + "=" * 70)


# -----------------------------------------
# ETL Main Pipeline
# -----------------------------------------
def run_etl():
    """Main ETL orchestrator"""
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("🚀 STARTING ETL PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Project root: {BASE_DIR}")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # =====================================
        # Step 0: Validation
        # =====================================
        if not validate_environment():
            logger.error("❌ Environment validation failed. Aborting ETL.")
            sys.exit(1)

        # =====================================
        # Step 1: Extract
        # =====================================
        logger.info("\n" + "─" * 70)
        logger.info("📥 STEP 1/3: EXTRACTING RAW DATA")
        logger.info("─" * 70)
        
        raw_data = extract_data()
        
        if not raw_data:
            logger.error("❌ No data extracted. Aborting ETL.")
            sys.exit(1)
        
        logger.info(f"✅ Extracted {len(raw_data)} datasets successfully")

        # =====================================
        # Step 1.5: Preprocessing
        # =====================================
        logger.info("\n🔧 Preprocessing raw datasets...")
        for name in raw_data:
            logger.info(f"   Processing {name}...")
            raw_data[name] = preprocess_dataframe(raw_data[name])
        
        logger.info("✅ Preprocessing completed")

        # =====================================
        # Step 2: Transform
        # =====================================
        logger.info("\n" + "─" * 70)
        logger.info("⚙️  STEP 2/3: TRANSFORMING DATA")
        logger.info("─" * 70)
        
        transformed_data = transform_data(raw_data)
        
        if not transformed_data:
            logger.error("❌ No data transformed. Aborting ETL.")
            sys.exit(1)
        
        logger.info(f"✅ Transformed {len(transformed_data)} datasets successfully")

        # =====================================
        # Step 3: Load
        # =====================================
        logger.info("\n" + "─" * 70)
        logger.info("📤 STEP 3/3: LOADING TO DATA WAREHOUSE")
        logger.info("─" * 70)
        
        run_loading_pipeline()
        
        logger.info("✅ Data loaded to warehouse successfully")

        # =====================================
        # Final Summary
        # =====================================
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_pipeline_summary(raw_data, transformed_data)
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 ETL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"End time  : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Duration  : {duration:.2f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Log saved : {log_filename.relative_to(BASE_DIR)}")
        logger.info("=" * 70 + "\n")
        
        logger.info("✨ Your data warehouse is ready for analysis!")
        
        return True

    except Exception as e:
        logger.error("\n" + "=" * 70)
        logger.error("❌ ETL PIPELINE FAILED")
        logger.error("=" * 70)
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error(f"Check log file for details: {log_filename.relative_to(BASE_DIR)}")
        logger.error("=" * 70)
        
        return False


# -----------------------------------------
# Entry Point
# -----------------------------------------
if __name__ == "__main__":
    success = run_etl()
    sys.exit(0 if success else 1)