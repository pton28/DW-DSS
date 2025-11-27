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
def safe_div(a, b):
    return np.where(b == 0, np.nan, a / b)

def compute_technical_indicators(df_raw):
    """
    Tạo các chỉ báo kỹ thuật nâng cao.
    QUAN TRỌNG: Tất cả các chỉ báo đều dùng .shift(1) để đảm bảo KHÔNG NHÌN THẤY TƯƠNG LAI (No Data Leakage).
    Dữ liệu tại dòng T sẽ chỉ chứa thông tin của T-1, T-2...
    """
    df = df_raw.copy()

    # Tạo các chuỗi giá trị đã dịch chuyển (Shifted Series)
    # Tại index t, chúng ta chỉ biết giá đóng cửa của t-1
    close_t = df['close'].shift(1)
    open_t = df['open'].shift(1)
    high_t = df['high'].shift(1)
    low_t = df['low'].shift(1)
    vol_t = df['volume'].shift(1)

    # 1. Price Features (Cơ bản)
    df['hl_pct'] = safe_div(high_t - low_t, close_t)
    df['co_pct'] = safe_div(close_t - open_t, open_t)

    # 2. Moving Averages & Slopes (Xu hướng)
    for w in [3, 5, 10, 20, 50]:
        df[f'close_ma_{w}'] = close_t.rolling(w).mean()
    
    df['ma20'] = df['close_ma_20']
    df['ma50'] = df['close_ma_50']
    
    # Độ dốc của đường MA (cho biết xu hướng đang mạnh hay yếu)
    df['ma20_slope'] = df['ma20'] - df['ma20'].shift(1)
    df['ma50_slope'] = df['ma50'] - df['ma50'].shift(1)
    # Khoảng cách giữa MA20 và MA50
    df['ma_cross_dist'] = safe_div(df['ma20'] - df['ma50'], df['ma50'])

    # 3. RSI (Momentum)
    delta = close_t.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = safe_div(roll_up, roll_down)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 4. ATR (Volatility - Độ biến động)
    tr1 = high_t - low_t
    tr2 = (high_t - df['close'].shift(2)).abs()
    tr3 = (low_t - df['close'].shift(2)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()

    # 5. Bollinger Bands
    bb_mid = close_t.rolling(20).mean()
    bb_std = close_t.rolling(20).std()
    df['bb_width'] = safe_div(2 * bb_std, bb_mid)

    # 6. Stochastic Oscillator
    low_min = low_t.rolling(14).min()
    high_max = high_t.rolling(14).max()
    df['stoch_k'] = safe_div(close_t - low_min, high_max - low_min) * 100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # 7. Lag Features (Dữ liệu quá khứ)
    # Lag N nghĩa là lấy giá trị của N ngày trước đó
    for lag in [1, 2, 3, 4, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'return_lag_{lag}'] = df['close'].pct_change().shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

    # 8. Mô hình Nến (Candlestick Patterns)
    body = (close_t - open_t).abs()
    candle_range = (high_t - low_t)
    upper_wick = high_t - pd.concat([close_t, open_t], axis=1).max(axis=1)
    lower_wick = pd.concat([close_t, open_t], axis=1).min(axis=1) - low_t

    df['candle_body_pct'] = safe_div(body, candle_range)
    df['upper_wick_pct'] = safe_div(upper_wick, candle_range)
    df['lower_wick_pct'] = safe_div(lower_wick, candle_range)

    df['is_doji'] = (df['candle_body_pct'] < 0.1).astype(int)
    # Hammer: Bóng nến dưới dài, thân nhỏ
    df['is_hammer'] = ((df['lower_wick_pct'] > 0.5) & (df['candle_body_pct'] < 0.4)).astype(int)
    # Inverted Hammer: Bóng nến trên dài, thân nhỏ
    df['is_inverted_hammer'] = ((df['upper_wick_pct'] > 0.5) & (df['candle_body_pct'] < 0.4)).astype(int)

    # 9. Volume Features
    df['volume_ma5'] = vol_t.rolling(5).mean()
    df['volume_ma20'] = vol_t.rolling(20).mean()
    # Xếp hạng volume trong 20 ngày qua
    df['volume_rank_20'] = vol_t.rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan)

    # 10. Volatility Regime (Chế độ biến động)
    # So sánh biến động ngắn hạn (20 ngày) vs dài hạn (100 ngày)
    # shift(2) để tính return của t-1
    ret_shift1 = df['close'].pct_change().shift(1)
    short_vol = ret_shift1.rolling(20).std()
    long_vol = ret_shift1.rolling(100).std()
    df['vol_regime'] = (short_vol > long_vol).astype(int)

    return df

def create_targets(df):
    """
    Tạo biến mục tiêu (Target) cho bài toán Machine Learning.
    Mục tiêu phải là tương lai (shift -1).
    """
    # 1. Next_Return: Giá ngày mai tăng/giảm bao nhiêu %? (Dùng cho Hồi quy)
    df['Next_Return'] = df['close'].pct_change().shift(-1)
    
    # 2. Next_Direction: Ngày mai Tăng (1) hay Giảm (0)? (Dùng cho Phân loại)
    df['Next_Direction'] = (df['Next_Return'] > 0).astype(int)
    
    # 3. (Tuỳ chọn) Signal: Logic thủ công để đối chiếu (giống file cũ)
    # Bạn có thể giữ hoặc bỏ, nhưng model XGBoost sẽ dùng Next_Direction
    def heuristic_signal(row):
        if row['rsi'] < 30: return 1
        if row['rsi'] > 70: return -1
        return 0
    df['Signal_Heuristic'] = df.apply(heuristic_signal, axis=1)

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
            clean_df = compute_technical_indicators(clean_df)
            
            # 5. Tạo Tín hiệu (Mua hay bán cổ phiếu)
            clean_df = create_targets(clean_df)
            clean_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

            # 6. Lưu file (Thêm hậu tố _cleaned để dễ phân biệt)
            output_name = file_name.replace(".csv", "_cleaned.csv")
            save_path = os.path.join(OUTPUT_DIR, output_name)
            
            clean_df.to_csv(save_path, index=False)
            print(f"   -> Đã lưu tại: {save_path}")
            
        except Exception as e:
            print(f"Lỗi khi xử lý file {file_path}: {e}")

print("Hoàn tất quá trình Transform.")
    