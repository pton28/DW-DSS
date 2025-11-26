import os
import glob
import pandas as pd
import numpy as np

SOURCE_DIR = '../Dataset/Raw'
OUTPUT_DIR = '../Dataset/Cleaned'

os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)

csv_files = glob.glob(os.path.join(SOURCE_DIR, "**", "*.csv"), recursive=True)

def basic_cleaning(df):
    """Làm sạch cơ bản để đưa lên Silver Zone."""
    # Xóa duplicates
    df = df.drop_duplicates()

    # Xử lý NaN bằng forward-fill và back-fill
    df = df.ffill().bfill()

    # Chuẩn hóa tên cột
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Giữ lại các cột cần thiết (tên cột đã được chuyển về chữ thường)
    keep_cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep_cols if c in df.columns]]
    
    # Chuyển cột ngày về datetime nếu có
    date_cols = [col for col in df.columns if "date" in col]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], utc=True)
        except:
            pass

    return df


def inject_symbol_if_missing(df, file_path):
    """
    LOGIC QUAN TRỌNG: Xử lý trường hợp file lẻ (GOOG.csv) không có cột Symbol.
    """
    if 'symbol' not in df.columns:
        file_name = os.path.basename(file_path)
        symbol_name = os.path.splitext(file_name)[0].upper()
        
        print(f"Không thấy cột Symbol. Tự động gán: {symbol_name}")
        df['symbol'] = symbol_name
        
    return df

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))

def feature_engineering(df): 
    if 'symbol' in df:
        df = df.sort_values(['symbol', 'date'])
        
        # RETURN
        df['return'] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # MA (Moving Average)
        df['ma20'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())
        df['ma50'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(50).mean())

        # RSI
        df['rsi'] = df.groupby('symbol')['close'].transform(lambda x: calc_rsi(x))
        
        # Fill NaN sinh ra do rolling
        df = df.fillna(0)
    return df

if not csv_files:
    print("Không tìm thấy file CSV nào trong thư mục Raw!")
else:
    for file_path in csv_files:
        try:
            file_name = os.path.basename(file_path)
            print(f"Đang xử lý: {file_name}")
            
            # 1. Đọc dữ liệu
            df = pd.read_csv(file_path)
            
            # 2. Làm sạch cơ bản
            clean_df = basic_cleaning(df)
            
            # 3. XỬ LÝ RIÊNG: Fill cột Symbol nếu thiếu
            clean_df = inject_symbol_if_missing(clean_df, file_path)

            # 4. Tạo Feature (Lúc này chắc chắn đã có symbol)
            clean_df = feature_engineering(clean_df)
            
            # 5. Lưu file (Thêm hậu tố _cleaned để dễ phân biệt)
            output_name = file_name.replace(".csv", "_cleaned.csv")
            save_path = os.path.join(OUTPUT_DIR, output_name)
            
            clean_df.to_csv(save_path, index=False)
            print(f"   -> Đã lưu tại: {save_path}")
            
        except Exception as e:
            print(f"Lỗi khi xử lý file {file_path}: {e}")

print("Hoàn tất quá trình Transform.")
    