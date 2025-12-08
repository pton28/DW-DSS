import duckdb
import pandas as pd
from pathlib import Path

# Cấu hình đường dẫn
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "Data_warehouse" / "dw.duckdb"
GOLD_DIR = BASE_DIR / "Dataset" / "Gold"

def check_duckdb():
    print("\n" + "="*50)
    print("🦈 KIỂM TRA DATA WAREHOUSE (DuckDB)")
    print("="*50)

    if not DB_PATH.exists():
        print(f"❌ Không tìm thấy file database tại: {DB_PATH}")
        return

    try:
        con = duckdb.connect(str(DB_PATH))
        
        # 1. Liệt kê các bảng
        print("\n--- 📂 DANH SÁCH BẢNG ---")
        tables = con.execute("SHOW TABLES").fetchdf()
        if tables.empty:
            print("⚠️ Database chưa có bảng nào!")
        else:
            print(tables)

        # 2. Kiểm tra chi tiết từng bảng quan trọng
        target_tables = [
            "dim_company", 
            "dim_date", 
            "dim_fin_metric", 
            "fact_finance", 
            "fact_stock_prices"
        ]

        for table in target_tables:
            print(f"\n" + "-"*40)
            print(f"🔎 BẢNG: {table}")
            print("-"*40)
            try:
                schema_df = con.execute(f"DESCRIBE {table}").fetchdf()
                
                # Chỉ lấy 2 cột quan trọng là column_name và column_type để hiển thị
                print("1️⃣  CẤU TRÚC CỘT (SCHEMA):")
                print(schema_df[['column_name', 'column_type']].to_string(index=False))
                print("."*40)
                
                # Đếm số dòng
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"   -> Tổng số dòng: {count:,}")
                
                # In 3 dòng đầu
                if count > 0:
                    print("   -> Dữ liệu mẫu (Top 3):")
                    df_sample = con.execute(f"SELECT * FROM {table} LIMIT 3").fetchdf()
                    print(df_sample.to_string(index=False))
                else:
                    print("   ⚠️ Bảng rỗng!")
            except Exception as e:
                print(f"   ❌ Lỗi truy vấn bảng này (có thể chưa được tạo): {e}")
                
        con.close()

    except Exception as e:
        print(f"❌ Lỗi kết nối DuckDB: {e}")

def check_parquet_files():
    print("\n" + "="*50)
    print("✨ KIỂM TRA GOLD LAYER (Parquet Files)")
    print("="*50)
    
    # Kiểm tra Facts và Dims
    dirs_to_check = [GOLD_DIR / "Dims", GOLD_DIR / "Facts"]
    
    for folder in dirs_to_check:
        if not folder.exists():
            continue
            
        print(f"\n📂 Folder: {folder.name}")
        files = list(folder.glob("*.parquet"))
        
        if not files:
            print("   (Trống)")
            continue
            
        for f in files:
            try:
                df = pd.read_parquet(f)
                print(f"   📄 {f.name:<25} | {len(df):,}")
            except Exception:
                print(f"   ❌ {f.name} (Lỗi đọc file)")

if __name__ == "__main__":
    check_parquet_files()
    check_duckdb()