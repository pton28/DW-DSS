import pandas as pd

# --- 1. Đọc dữ liệu từ 2 file ---
goog_1 = pd.read_csv("./Dataset/Raw/GOOG_2021.csv")
goog_2 = pd.read_csv("./Dataset/Raw/GOOG_2025.csv")

# --- 2. Chuẩn hoá tên cột ---
# Đảm bảo cả hai có cùng tên cột cơ bản
goog_1.rename(columns={
    ""
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume"
}, inplace=True)

goog_2.rename(columns={
    "Date": "Date",
    "Open": "Open",
    "High": "High",
    "Low": "Low",
    "Close": "Close",
    "Volume": "Volume"
}, inplace=True)

# --- 3. Chuẩn hoá kiểu dữ liệu ngày ---
# Chuyển về datetime và bỏ timezone
goog_1["Date"] = pd.to_datetime(goog_1["Date"]).dt.tz_localize(None)
goog_2["Date"] = pd.to_datetime(goog_2["Date"], utc=True, errors="coerce").dt.tz_convert(None)

# --- 4. Giữ các cột quan trọng ---
cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
goog_1 = goog_1[cols]
goog_2 = goog_2[cols]

# --- 5. Gộp dữ liệu và loại bỏ trùng ---
goog_all = pd.concat([goog_1, goog_2], ignore_index=True)
goog_all.drop_duplicates(subset="Date", inplace=True)
goog_all.sort_values("Date", inplace=True)
goog_all.reset_index(drop=True, inplace=True)

# --- 6. Thêm cột Symbol ---
goog_all["Symbol"] = "GOOG"

# --- 7. Chuẩn hoá định dạng ngày sang YYYY-MM-DD ---
goog_all["Date"] = goog_all["Date"].dt.strftime("%Y-%m-%d")

# --- 8. Xuất file đã chuẩn hoá ---
output_path = "./Dataset/Raw/GOOG.csv"
goog_all.to_csv(output_path, index=False)

print(f"✅ Đã tích hợp và chuẩn hoá dữ liệu GOOG thành công! File lưu tại: {output_path}")
print(f"Số dòng sau khi gộp: {len(goog_all)}")
print(goog_all.head())
