import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

warnings.filterwarnings("ignore")

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "Train_model", "best_stock_price_model.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "Dataset", "Cleaned", "prices_cleaned.csv")

def load_resources():
    """Load model và dữ liệu lịch sử mới nhất"""
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy model tại {MODEL_PATH}. Hãy chạy file train trước.")
    
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise Exception(f"Lỗi khi load model: {e}")
    
    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy dữ liệu tại {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Lấy dòng dữ liệu cuối cùng để làm điểm bắt đầu dự đoán
    last_row = df.iloc[-1]
    
    # Tính toán các chỉ số trung bình
    avg_volatility_5d = df['volatility_5d'].tail(5).mean()
    avg_volatility_20d = df['volatility_20d'].tail(20).mean()
    avg_volume = df['volume'].tail(5).mean()
    
    # Tính daily volatility từ lịch sử
    daily_returns = df['close'].pct_change().dropna()
    daily_volatility = daily_returns.std()
    
    context = {
        'last_row': last_row,
        'avg_volatility_5d': avg_volatility_5d,
        'avg_volatility_20d': avg_volatility_20d,
        'avg_volume': avg_volume,
        'daily_volatility': daily_volatility
    }
    
    return model, context

def predict_future(num_days, model, context):
    """
    Dự báo giá trong n ngày tiếp theo.
    Cơ chế: Dùng giá dự đoán của ngày T để làm đầu vào cho ngày T+1 (Recursive).
    """
    future_predictions = []
    
    # Khởi tạo trạng thái hiện tại từ dữ liệu thực tế cuối cùng
    current_state = context['last_row'].copy()
    current_date = current_state['date']
    
    # Danh sách các features model cần (phải đúng thứ tự lúc train)
    features = ['open', 'high', 'low', 'close', 'volume', 
                'pct_change', 'daily_return', 'volatility_5d', 'volatility_20d']
    
    for i in range(num_days):
        # 1. Chuẩn bị dữ liệu đầu vào (X)
        input_data = current_state[features].values.reshape(1, -1)
        
        # 2. Dự đoán giá Close ngày tiếp theo
        pred_close = model.predict(input_data)[0]
        
        # 3. Tính ngày tiếp theo
        next_date = current_date + timedelta(days=1)
        # Nếu rơi vào T7, CN thì nhảy sang T2 (giả lập đơn giản)
        if next_date.weekday() == 5: # Saturday
            next_date += timedelta(days=2)
        elif next_date.weekday() == 6: # Sunday
            next_date += timedelta(days=1)
            
        # 4. Lưu kết quả
        future_predictions.append({
            'date': next_date,
            'predicted_price': pred_close
        })
        
        # 5. Cập nhật trạng thái (current_state) để dự đoán ngày kế tiếp
        prev_close = current_state['close']
        
        # Giả định: Open ngày sau = Close ngày trước
        current_state['open'] = prev_close 
        current_state['close'] = pred_close
        
        # Giả định: High/Low dao động theo độ biến động trung bình
        volatility = context['avg_volatility_5d']
        current_state['high'] = pred_close * (1 + volatility)
        current_state['low'] = pred_close * (1 - volatility)
        
        # Giả định: Volume bằng trung bình 5 ngày gần nhất
        current_state['volume'] = context['avg_volume']
        
        # Cập nhật các chỉ số biến động
        current_state['pct_change'] = (pred_close - prev_close) / prev_close
        current_state['daily_return'] = pred_close - prev_close
        
        # Giữ nguyên volatility
        current_state['volatility_5d'] = context['avg_volatility_5d']
        current_state['volatility_20d'] = context['avg_volatility_20d']
        
        current_date = next_date

    return pd.DataFrame(future_predictions)

def simulate_monte_carlo(num_days, num_simulations, model, context):
    """
    Chạy n mô phỏng. Mỗi mô phỏng sẽ thêm yếu tố ngẫu nhiên (Noise) 
    dựa trên độ biến động lịch sử vào kết quả dự đoán của Model.
    """
    simulation_results = []

    start_price = context['last_row']['close']
    start_date = context['last_row']['date']
    daily_vol = context['daily_volatility']

    base_forecast = predict_future(num_days, model, context)
    trend_prices = base_forecast['predicted_price'].values
    dates = base_forecast['date'].values

    for sim in range(num_simulations):
        sim_prices = []
        current_sim_price = start_price
        
        for day in range(num_days):
            # Lấy giá dự báo từ mô hình gốc (Trend)
            model_price = trend_prices[day]
            
            # Tính % thay đổi dự kiến của mô hình
            prev_price_trend = start_price if day == 0 else trend_prices[day-1]
            expected_return = (model_price - prev_price_trend) / prev_price_trend
            
            # Thêm yếu tố ngẫu nhiên (Random Shock)
            shock = np.random.normal(0, daily_vol)
            
            # Giá mô phỏng
            sim_return = expected_return + shock
            current_sim_price = current_sim_price * (1 + sim_return)
            
            sim_prices.append(current_sim_price)
            
        simulation_results.append(sim_prices)
    
    # Chuyển thành numpy array để tính toán thống kê: (num_sims, num_days)
    sim_matrix = np.array(simulation_results)
    
    # Tính các đường phân vị (Percentiles)
    summary_data = []
    for i in range(num_days):
        day_prices = sim_matrix[:, i]
        summary_data.append({
            'date': dates[i],
            'mean_price': np.mean(day_prices),
            'median_price': np.percentile(day_prices, 50),
            'p95_price': np.percentile(day_prices, 95),  # Kịch bản Tốt (Top 5%)
            'p75_price': np.percentile(day_prices, 75),  # Khá tốt
            'p25_price': np.percentile(day_prices, 25),  # Khá xấu
            'p5_price': np.percentile(day_prices, 5),    # Kịch bản Xấu (Bottom 5%)
            'max_price': np.max(day_prices),
            'min_price': np.min(day_prices),
            'std_dev': np.std(day_prices)
        })
        
    return pd.DataFrame(summary_data), sim_matrix

def plot_monte_carlo_results(summary_df, sim_matrix, current_price, num_days, save_path='./Image/monte_carlo_forecast.png'):
    """
    Vẽ biểu đồ Monte Carlo simulation với các kịch bản
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    dates = pd.to_datetime(summary_df['date'])
    
    # --- Biểu đồ 1: Các đường phân vị và vùng tin cậy ---
    ax1.set_title(f'Monte Carlo Simulation - {num_days} Ngày Tới', fontsize=16, fontweight='bold')
    
    # Vẽ vùng tin cậy (confidence interval)
    ax1.fill_between(dates, summary_df['p5_price'], summary_df['p95_price'], 
                     alpha=0.2, color='blue', label='90% Confidence Interval (P5-P95)')
    ax1.fill_between(dates, summary_df['p25_price'], summary_df['p75_price'], 
                     alpha=0.3, color='green', label='50% Confidence Interval (P25-P75)')
    
    # Vẽ các đường chính
    ax1.plot(dates, summary_df['mean_price'], 'b-', linewidth=2.5, label='Giá Trung Bình (Mean)', marker='o')
    ax1.plot(dates, summary_df['median_price'], 'g--', linewidth=2, label='Giá Trung Vị (Median)', marker='s')
    ax1.plot(dates, summary_df['p95_price'], 'lime', linewidth=1.5, label='Kịch bản Tốt (P95)', linestyle='-.', alpha=0.8)
    ax1.plot(dates, summary_df['p5_price'], 'red', linewidth=1.5, label='Kịch bản Xấu (P5)', linestyle='-.', alpha=0.8)
    
    # Đường giá hiện tại
    ax1.axhline(y=current_price, color='black', linestyle='--', linewidth=2, label=f'Giá Hiện Tại: ${current_price:.2f}')
    
    ax1.set_xlabel('Ngày', fontsize=12)
    ax1.set_ylabel('Giá ($)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Biểu đồ 2: Tất cả các đường mô phỏng (spaghetti plot) ---
    ax2.set_title('Tất Cả Các Kịch Bản Mô Phỏng', fontsize=14, fontweight='bold')
    
    # Vẽ một số đường mô phỏng (không vẽ hết vì quá nhiều)
    num_to_plot = min(100, sim_matrix.shape[0])
    for i in range(num_to_plot):
        ax2.plot(dates, sim_matrix[i, :], alpha=0.1, color='gray', linewidth=0.5)
    
    # Vẽ đường trung bình lên trên
    ax2.plot(dates, summary_df['mean_price'], 'b-', linewidth=3, label='Giá Trung Bình')
    ax2.axhline(y=current_price, color='black', linestyle='--', linewidth=2, label=f'Giá Hiện Tại: ${current_price:.2f}')
    
    ax2.set_xlabel('Ngày', fontsize=12)
    ax2.set_ylabel('Giá ($)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Biểu đồ đã được lưu tại: {save_path}")
    plt.show()

def recommend_action_monte_carlo(current_price, summary_df):
    """Đưa ra khuyến nghị dựa trên Monte Carlo simulation"""
    
    final_mean = summary_df['mean_price'].iloc[-1]
    final_median = summary_df['median_price'].iloc[-1]
    final_p95 = summary_df['p95_price'].iloc[-1]
    final_p5 = summary_df['p5_price'].iloc[-1]
    
    max_price = summary_df['mean_price'].max()
    min_price = summary_df['mean_price'].min()
    
    # Tính % thay đổi dựa trên giá trung bình
    change_percent = ((final_mean - current_price) / current_price) * 100
    
    # Tính lợi nhuận kỳ vọng
    expected_profit = final_mean - current_price
    expected_profit_percent = change_percent
    
    # Tính rủi ro (downside risk) - xác suất giá giảm
    all_final_prices = summary_df['mean_price'].iloc[-1]
    
    # Đánh giá xu hướng
    advice = ""
    status = ""
    risk_level = ""
    
    if change_percent > 5.0:
        status = "🟢 TĂNG MẠNH"
        advice = f"KHUYẾN NGHỊ MUA. Giá dự kiến tăng {change_percent:.2f}%."
        risk_level = "Trung bình"
    elif change_percent > 2.0:
        status = "🟢 TĂNG VỪA"
        advice = f"NÊN CÂN NHẮC MUA. Xu hướng tích cực với lợi nhuận kỳ vọng {change_percent:.2f}%."
        risk_level = "Trung bình"
    elif change_percent > 0.5:
        status = "🟡 TĂNG NHẸ"
        advice = "CÓ THỂ MUA. Xu hướng tăng nhẹ nhưng không rõ ràng."
        risk_level = "Cao"
    elif change_percent < -5.0:
        status = "🔴 GIẢM MẠNH"
        advice = f"KHUYẾN NGHỊ BÁN/KHÔNG MUA. Giá dự kiến giảm {abs(change_percent):.2f}%."
        risk_level = "Cao"
    elif change_percent < -2.0:
        status = "🔴 GIẢM VỪA"
        advice = f"NÊN CẨN TRỌNG. Xu hướng giảm với rủi ro mất {abs(change_percent):.2f}%."
        risk_level = "Cao"
    elif change_percent < -0.5:
        status = "🟡 GIẢM NHẸ"
        advice = "GIỮ NGUYÊN hoặc CẨN TRỌNG. Xu hướng giảm nhẹ."
        risk_level = "Trung bình"
    else:
        status = "🟡 ĐI NGANG"
        advice = "GIỮ NGUYÊN. Giá biến động không đáng kể."
        risk_level = "Thấp"
    
    return {
        'status': status,
        'advice': advice,
        'change_percent': change_percent,
        'expected_profit': expected_profit,
        'expected_profit_percent': expected_profit_percent,
        'current_price': current_price,
        'mean_target_price': final_mean,
        'median_target_price': final_median,
        'best_case_price': final_p95,
        'worst_case_price': final_p5,
        'highest_mean_price': max_price,
        'lowest_mean_price': min_price,
        'risk_level': risk_level
    }

def print_monte_carlo_summary(recommendation, num_days, num_sims):
    """In ra tóm tắt kết quả Monte Carlo"""
    print("\n" + "="*70)
    print(f"🎲 KẾT QUẢ MONTE CARLO SIMULATION ({num_sims} lần mô phỏng, {num_days} ngày)")
    print("="*70)
    
    print(f"\n📍 GIÁ HIỆN TẠI: ${recommendation['current_price']:.2f}")
    print(f"\n📊 DỰ BÁO CUỐI KỲ (Sau {num_days} ngày):")
    print(f"   • Giá Trung Bình (Mean):    ${recommendation['mean_target_price']:.2f}")
    print(f"   • Giá Trung Vị (Median):    ${recommendation['median_target_price']:.2f}")
    print(f"   • Kịch bản Tốt (P95):       ${recommendation['best_case_price']:.2f}")
    print(f"   • Kịch bản Xấu (P5):        ${recommendation['worst_case_price']:.2f}")
    
    print(f"\n💰 LỢI NHUẬN KỲ VỌNG (theo giá trung bình):")
    profit_sign = "+" if recommendation['expected_profit'] > 0 else ""
    print(f"   • Lợi nhuận: {profit_sign}${recommendation['expected_profit']:.2f}")
    print(f"   • Tỷ suất:   {profit_sign}{recommendation['expected_profit_percent']:.2f}%")
    
    print(f"\n📈 TRONG QUÁ TRÌNH:")
    print(f"   • Giá cao nhất có thể:  ${recommendation['highest_mean_price']:.2f}")
    print(f"   • Giá thấp nhất có thể: ${recommendation['lowest_mean_price']:.2f}")
    
    print(f"\n⚠️  MỨC ĐỘ RỦI RO: {recommendation['risk_level']}")
    
    print(f"\n{recommendation['status']}")
    print(f"💡 {recommendation['advice']}")
    print("="*70 + "\n")

def run_prediction(num_days: int, use_monte_carlo=False, num_sims=2000):
    """
    Hàm gọi từ Streamlit hoặc chạy độc lập.
    """
    try:
        model, context = load_resources()
        current_price = context['last_row']['close']
        
        if use_monte_carlo:
            # Chạy Monte Carlo
            summary_df, sim_matrix = simulate_monte_carlo(num_days, num_sims, model, context)
            
            # Khuyến nghị dựa trên Monte Carlo
            rec = recommend_action_monte_carlo(current_price, summary_df)
            
            # Vẽ biểu đồ
            plot_monte_carlo_results(summary_df, sim_matrix, current_price, num_days)
            
            # In tóm tắt
            print_monte_carlo_summary(rec, num_days, num_sims)
            
            return {
                'type': 'monte_carlo',
                'data': summary_df,
                'raw_matrix': sim_matrix,
                'ruecommendation': rec,
                'crrent_price': current_price
            }, None
            
        else:
            # Chạy cơ bản
            forecast_df = predict_future(num_days, model, context)
            rec = recommend_action_basic(current_price, forecast_df)
            
            return {
                'type': 'basic',
                'data': forecast_df,
                'recommendation': rec,
                'current_price': current_price
            }, None

    except Exception as e:
        return None, str(e)

def recommend_action_basic(current_price, future_df):
    """Đưa ra khuyến nghị Mua/Bán dựa trên xu hướng (cho dự đoán cơ bản)"""
    max_price = future_df['predicted_price'].max()
    min_price = future_df['predicted_price'].min()
    final_price = future_df['predicted_price'].iloc[-1]
    
    change_percent = ((final_price - current_price) / current_price) * 100
    
    advice = ""
    status = ""
    
    if change_percent > 2.0:
        status = "🟢 TĂNG MẠNH"
        advice = f"NÊN MUA. Giá dự kiến tăng {change_percent:.2f}%."
    elif change_percent > 0.5:
        status = "🟢 TĂNG NHẸ"
        advice = "CÂN NHẮC MUA. Xu hướng tăng nhẹ."
    elif change_percent < -2.0:
        status = "🔴 GIẢM MẠNH"
        advice = f"NÊN BÁN/KHÔNG MUA. Giá dự kiến giảm {abs(change_percent):.2f}%."
    elif change_percent < -0.5:
        status = "🔴 GIẢM NHẸ"
        advice = "CẨN TRỌNG. Xu hướng giảm nhẹ."
    else:
        status = "🟡 ĐI NGANG"
        advice = "GIỮ NGUYÊN. Giá biến động không đáng kể."
        
    return {
        'status': status,
        'advice': advice,
        'change_percent': change_percent,
        'target_price': final_price,
        'highest_price': max_price,
        'lowest_price': min_price
    }

# --- Chạy thử nghiệm độc lập ---
if __name__ == "__main__":
    print("🎲 Monte Carlo Simulation - Stock Price Forecasting")
    print("-" * 70)
    
    try:
        n = int(input("Nhập số ngày muốn dự đoán (vd: 7, 14, 30): "))
        
        print(f"\n⏳ Đang chạy 2000 mô phỏng cho {n} ngày...\n")
        
        res, err = run_prediction(n, use_monte_carlo=True, num_sims=2000)
        
        if err:
            print(f"❌ Lỗi: {err}")
        else:
            print("\n✅ Hoàn thành!")
            
    except ValueError:
        print("❌ Vui lòng nhập số nguyên hợp lệ!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng.")
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")