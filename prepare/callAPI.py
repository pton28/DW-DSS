import yfinance as yf
from datetime import date
import pandas as pd
import os

# Lấy ticker của Google
gg_stock = yf.Ticker("GOOGL")

print("\n--- Dữ liệu 2021 đến 2025 ---")
hist = gg_stock.history(start="2021-11-06", end="2025-11-07")
hist.to_csv("GOOG_2025.csv")

# ============================
# ETL - Tải dữ liệu cổ phiếu công nghệ Mỹ
# ============================

# Danh sách 4 mã cổ phiếu cần tải
tickers = ["AAPL", "MSFT", "AMZN", "NVDA"]

# Thời gian tải dữ liệu
start_date = "2016-01-01"
end_date = "2025-11-07"

# Thư mục lưu dữ liệu (có thể đổi tuỳ bạn)
save_path = "./Dataset/Raw/"

# Tạo thư mục nếu chưa có
os.makedirs(save_path, exist_ok=True)

# Tải và lưu từng mã cổ phiếu
for ticker in tickers:
    print(f"📈 Đang tải dữ liệu: {ticker}")
    
    # Gọi API từ Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    
    # Chuẩn hóa tên cột
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    
    # Thêm cột tên mã cổ phiếu
    df["Symbol"] = ticker
    
    # Lưu ra file CSV
    file_path = os.path.join(save_path, f"{ticker}.csv")
    df.to_csv(file_path, index=False)
    
    print(f"✅ Đã lưu {ticker} vào {file_path}")

print("\n🎯 Hoàn tất! Dữ liệu 4 mã đã được tải và lưu vào thư mục data_lake/")
