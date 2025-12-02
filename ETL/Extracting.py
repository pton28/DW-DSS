import os
import pandas as pd
import yfinance as yf
from datetime import date

# --- Cấu hình ---
save_path = "../Dataset/Raw/"
os.makedirs(save_path, exist_ok=True)

ticker = 'GOOG'
start_date = "2016-06-1"
end_date = "2025-11-07"


print("Bắt đầu tải dữ liệu từ Yahoo Finance...")
print(f"📈 Đang tải dữ liệu: {ticker}")

try:
    # Gọi API từ Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)

    # Chuẩn hóa tên cột
    df = df.rename(columns={
        "Date": "Date", "Open": "Open", "High": "High", 
        "Low": "Low", "Close": "Close", "Volume": "Volume"
    })
    
    req_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in req_cols if c in df.columns]]
    
    df["Symbol"] = ticker
    print(f"   -> Tải xong {ticker}: {len(df)} dòng")

    try:
        df_path = os.path.join(save_path, "GOOG_API.csv")
        df.to_csv(df_path, index=False)
        print(f"\nHoàn tất! Dữ liệu đã được tải và lưu vào thư mục {df_path}")
    except Exception as e:
        print(f"Lỗi tải {ticker}: {e}")

except Exception as e:
    print(f"Lỗi tải {ticker}: {e}")