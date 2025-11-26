import yfinance as yf
from datetime import date
import pandas as pd
import os

# --- Cấu hình ---
save_path = "../Dataset/Raw/"
os.makedirs(save_path, exist_ok=True)

tickers = ["AAPL", "MSFT", "AMZN", "NVDA"]
start_date = "2016-01-01"
end_date = "2025-11-07"

all_df = []

print("Bắt đầu tải dữ liệu từ Yahoo Finance...")
for ticker in tickers:
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
        all_df.append(df)
        print(f"   -> Tải xong {ticker}: {len(df)} dòng")
    except Exception as e:
        print(f"Lỗi tải {ticker}: {e}")

if all_df:
    print("\nĐang gộp dữ liệu 4 mã...")    
    final_df = pd.concat(all_df, ignore_index=True)

    final_file_path = os.path.join(save_path, "bigTech.csv")
    final_df.to_csv(final_file_path, index=False)

    print(f"\nHoàn tất! Dữ liệu 4 mã đã được tải và lưu vào thư mục {final_file_path}")
else:
    print("Không tải được dữ liệu nào")
