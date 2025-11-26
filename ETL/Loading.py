import os
import glob
import duckdb
import pandas as pd

# --- 1️⃣ Đường dẫn và đầu ra ---
data_path = "../Dataset/Cleaned"
output_path = "../Data_warehouse/all_stocks.csv"
db_output_path = "../Data_warehouse/staging.db"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# --- 2️⃣ Tìm tất cả file CSV trong các thư mục con ---
csv_files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)

all_data = []

# --- 3️⃣ Đọc và chuẩn hoá từng file ---
for file in csv_files:
    try:
        if "indicators" in file.lower():
            continue  # bỏ qua file tính chỉ báo kỹ thuật nếu có
        try:
            df = pd.read_csv(file)
            print(f"🔄 Đang xử lý: {file} ({len(df)} dòng)")
        except Exception as e:
            print(f"Lỗi: {e}")

        # Chuẩn hóa tên cột (tránh lỗi chữ hoa/thường)
        df.columns = [c.capitalize() for c in df.columns]

        # Chuẩn hoá cột ngày
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
            
        # Thêm vào danh sách tổng
        all_data.append(df)
    except Exception as e:
        print(f"Bỏ qua file {file}: {e}")

# --- 4️⃣ Gộp và Xuất ---
if all_data:
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.sort_values(["Symbol", "Date"], inplace=True)
    
    merged_df.to_csv(output_path, index=False)

    print(f"📦 Đang ghi vào Database: {db_output_path}...")
    if os.path.exists(db_output_path):
        try:
            os.remove(db_output_path)  # Xoá file cũ nếu tồn tại
        except Exception as e:
            print(f"Lỗi khi xoá file cũ: {e}")
            exit()
            
    conn = duckdb.connect(db_output_path)
    conn.execute("CREATE OR REPLACE TABLE historical_stock_price AS SELECT * FROM merged_df")
    row_count = conn.execute("SELECT COUNT(*) FROM historical_stock_price").fetchone()[0]
    conn.close()

    print(f"\n✅  TẢI VÀO KHO DỮ LIỆU THÀNH CÔNG!")
    print(f"📊  Tổng số dòng: {len(merged_df)}")
    print(f"📄  File Warehouse: {output_path}")
    print(f"🗄️  Database Table: 'historical_stock_price' trong {db_output_path} ({row_count} dòng)")
else:
    print("Không có dữ liệu để gộp.")
