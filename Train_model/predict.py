import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from sklearn.preprocessing import RobustScaler

# ==========================================
# CẤU HÌNH
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(CURRENT_DIR, '../Dataset/Cleaned/GOOG_cleaned.csv')
MODEL_CLF_PATH = os.path.join(CURRENT_DIR, 'xgb_classifier.joblib')
MODEL_REG_PATH = os.path.join(CURRENT_DIR, 'xgb_regressor.joblib')
SCALER_PATH = os.path.join(CURRENT_DIR, 'scaler.joblib')


# =========================================================
# HÀM CHÍNH DÙNG ĐỂ IMPORT VÀ GỌI TỪ STREAMLIT
# =========================================================
def predict_days(n):
    """
    Trả về dữ liệu mô phỏng để dùng trong Streamlit.
    
    Output:
        {
            "future_dates": DatetimeIndex,
            "simulations": DataFrame,
            "last_close": float,
            "pred_return": float,
            "pred_direction": int,
            "volatility": float
        }
    """
    # 1. LOAD dữ liệu
    df = pd.read_csv(DATA_PATH)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

    # 2. LOAD model
    clf = joblib.load(MODEL_CLF_PATH)
    reg = joblib.load(MODEL_REG_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    excluded = ['Next_Return', 'Next_Direction', 'symbol']
    feature_cols = [c for c in df.columns if c not in excluded]
    
    # 3. Lấy dữ liệu gần nhất
    last_row = df.iloc[[-1]][feature_cols]
    last_close = df.iloc[-1]['close']
    last_date = df.index[-1]
    
    # 4. Dự đoán giá ngày tiếp theo
    last_row_scaled = scaler.transform(last_row)
    pred_return = reg.predict(last_row_scaled)[0] # AI dự báo % tăng giảm ngày mai
    pred_dir = clf.predict(last_row_scaled)[0]    # AI dự báo hướng (1: Tăng, 0: Giảm)

    daily_volatility = df['return'].tail(30).std()
    
    # 5. Mô phỏng Monte Carlo
    simulations = 1000
    simulation_data = {}
    for i in range(simulations):
        price_list = []
        price = last_close + (1 + pred_return)
        price_list.append(price)

        historical_drift = df['return'].mean()
        for _ in range(n - 1):
            shock = np.random.normal(0, daily_volatility)
            price = price * (1 + historical_drift + shock)
            price_list.append(price)

        simulation_data[i] = price_list

    simualation_df = pd.DataFrame(simulation_data)

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n)
    return {
        "future_dates": future_dates,
        "simulations": simualation_df,
        "last_close": last_close,
        "pred_return": pred_return,
        "pred_direction": pred_dir,
        "volatility": daily_volatility
    }

def run_simulation():
    print("\n" + "="*50)
    print("🔮 DỰ BÁO GIÁ CỔ PHIẾU THEO SỐ NGÀY TÙY CHỌN")
    print("="*50)

    try:
        # 1. NHẬP INPUT TỪ NGƯỜI DÙNG
        try:
            days_input = input(">> Nhập số ngày bạn muốn dự đoán (VD: 5, 10, 30): ")
            n_days = int(days_input)
            if n_days <= 0: raise ValueError
        except ValueError:
            print("Lỗi: Vui lòng nhập một số nguyên dương.")
            return

        res = predict_days(n_days)

        future_dates = res['future_dates']   
        simulation_df = res['simulations']     
        last_close = res['last_close']
        pred_return = res['pred_return']
        pred_dir = res['pred_direction']
        daily_volatility = res['volatility']

        print(f"\n[Dữ liệu cơ sở]")
        print(f"Giá hiện tại: ${last_close:.2f}")
        print(f"Biến động (30 ngày): {daily_volatility*100:.2f}%")
        print(f"AI Dự báo xu hướng: {'TĂNG' if pred_dir==1 else 'GIẢM'} ({pred_return*100:+.2f}%)")
        
        # Lấy giá trị cuối cùng của tất cả kịch bản
        ending_values = simulation_df.iloc[-1]
        avg_price = ending_values.mean()
        max_price = ending_values.quantile(0.95) # Kịch bản lạc quan (Top 5%)
        min_price = ending_values.quantile(0.05) # Kịch bản bi quan (Bottom 5%)
        
        roi_avg = (avg_price - last_close) / last_close * 100

        print("\n" + "="*50)
        print(f"📊 KẾT QUẢ DỰ BÁO SAU {n_days} NGÀY")
        print("="*50)
        print(f"Giá trung bình dự kiến: ${avg_price:.2f} ({roi_avg:+.2f}%)")
        print(f"Kịch bản lạc quan (95%): ${max_price:.2f}")
        print(f"Kịch bản bi quan (5%):   ${min_price:.2f}")
        print("-" * 50)
        
        # Khuyến nghị
        if roi_avg > 1.5:
            print("💡 KHUYẾN NGHỊ: MUA (Lợi nhuận kỳ vọng cao)")
        elif roi_avg < -1.5:
            print("💡 KHUYẾN NGHỊ: BÁN (Rủi ro giảm giá lớn)")
        else:
            print("💡 KHUYẾN NGHỊ: NẮM GIỮ / QUAN SÁT (Biên độ nhỏ)")

        # 7. VẼ BIỂU ĐỒ
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(future_dates, simulation_df.iloc[:, :50], color='gray', alpha=0.1, linewidth=1)
        plt.plot(future_dates, simulation_df.mean(axis=1), color='blue', linewidth=3, label='Trung bình dự kiến')
        plt.axhline(y=last_close, color='red', linestyle='--', label='Giá hiện tại')
        
        if n_days <= 10:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        elif n_days < 30:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        else:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

        plt.title(f'Mô phỏng giá cổ phiếu trong {n_days} ngày tới (Monte Carlo)')
        plt.xlabel('Ngày')
        plt.ylabel('Giá ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file model. Hãy chạy 'xgboost_model.py --save_models' trước.")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    run_simulation()