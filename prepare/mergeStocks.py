import pandas as pd
import glob
import os

# --- 1️⃣ Đường dẫn và đầu ra ---
data_path = "./Dataset/Raw"
output_path = "./Data_warehouse/all_stocks.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# --- 2️⃣ Tìm tất cả file CSV trong các thư mục con ---
csv_files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)

all_data = []

# --- 3️⃣ Đọc và chuẩn hoá từng file ---
for file in csv_files:
    if "indicators" in file.lower():
        continue  # bỏ qua file tính chỉ báo kỹ thuật nếu có

    df = pd.read_csv(file)
    print(f"🔄 Đang xử lý: {file} ({len(df)} dòng)")

    # Chuẩn hóa tên cột (tránh lỗi chữ hoa/thường)
    df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "symbol": "Symbol"
    }, inplace=True)

    # Chuẩn hoá cột ngày
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")

    # Nếu thiếu cột Symbol → tự tạo từ tên thư mục hoặc file
    if "Symbol" not in df.columns:
        symbol = os.path.basename(file).split(".")[0]
        symbol = symbol.upper().replace("_CLEAN", "")
        df["Symbol"] = symbol

    # Giữ lại các cột cần thiết
    keep_cols = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Thêm vào danh sách tổng
    all_data.append(df)

# --- 4️⃣ Gộp tất cả lại ---
merged_df = pd.concat(all_data, ignore_index=True)
merged_df.dropna(subset=["Date", "Close"], inplace=True)
merged_df.sort_values(["Symbol", "Date"], inplace=True)
merged_df.reset_index(drop=True, inplace=True)

# --- 5️⃣ Xuất file tổng hợp ---
merged_df.to_csv(output_path, index=False)

print(f"\n✅ Đã gộp thành công {len(csv_files)} file cổ phiếu!")
print(f"📁 File tổng hợp lưu tại: {output_path}")
print(f"📊 Tổng số dòng: {len(merged_df)}")
print("📋 Cấu trúc:")
print(merged_df.head())
