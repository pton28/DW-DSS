import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import hàm chạy Dự đoán
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, root_dir)

from Train_model.predict import predict_days
from Train_model.xgboost_model import train_and_save_models, backtest_realistic

# Setup cho streamlit
st.markdown("""
<style>
    /* Tìm tất cả các nút bấm và chỉnh style */
    div.stButton > button:first-child {
        text-align: left;
        padding-left: 20px; /* Thêm khoảng cách lề trái cho đẹp */
        width: 100%; /* Kéo dài nút ra hết khung */
    }
</style>
""", unsafe_allow_html=True)

SOURCE_PATH = '../Dataset/Cleaned/GOOG_cleaned.csv'
MODEL_DIR = '../Train_model'

data = pd.read_csv(SOURCE_PATH)
df = pd.DataFrame(data)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

overview = df[['open', 'high', 'low', 'close', 'volume']]
table = overview.describe()

if 'show_eda' not in st.session_state:
        st.session_state.show_eda = False
if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False

def eda_activate():
    st.session_state.show_eda=True
    st.session_state.show_prediction=False

def predict_activate():
    st.session_state.show_prediction=True
    st.session_state.show_eda=False


st.write("""
# Báo cáo Bài tập lớn Kho dữ Liệu và Hệ hỗ trợ Quyết định
## Phân tích xu hướng và hỗ trợ đầu tư cổ phiếu Google (GOOG)
""")

st.sidebar.header("Phân tích xu hướng và hỗ trợ đưa ra quyết định đầu tư")
st.sidebar.markdown("---")

eda = st.sidebar.button("Trực quan hoá xu hướng giá của mẫu cổ phiếu", on_click=eda_activate)
predict = st.sidebar.button("Dự đoán hỗ trợ đưa ra quyết định đầu tư", on_click=predict_activate)           

if st.session_state.show_eda:
    st.header("Xu hướng giá của mẫu cổ phiếu tính đến thời điểm khảo sát")
    show_trend = st.checkbox("Hiển thị đường xu hướng", value=True)

    window_size = st.slider("Khoảng thời gian (ngày) để tính xu hướng:", min_value=5, max_value=100, value=30)

    df_chart = df.resample('ME')[['open','high','low','close','volume']].mean()
    features = st.multiselect("What do you want to explore ?", ["open", "high", "low", "close", "volume"])
    for feature in features:
        col_name = feature.capitalize()

        st.write(f"## {col_name} Chart")
        fig, ax = plt.subplots()
        ax.plot(df_chart.index, df_chart[feature], label='Dữ liệu thực tế', alpha=0.6)

        if show_trend:
            trend_data = df_chart[feature].rolling(window=window_size).mean()
            ax.plot(df_chart.index, trend_data, color='red', linewidth=2, label=f"Xu hướng (MA {window_size})")

        ax.set_xlabel("Date")
        ax.set_ylabel(col_name)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        st.pyplot(fig)

elif st.session_state.show_prediction:
    st.header("Mô hình giả định đầu tư để chứng minh độ tin cậy")
    st.write("### Giả định đầu tư bằng phương pháp Backtesting")

    model_files = ['xgb_classifier.joblib', 'xgb_regressor.joblib', 'scaler.joblib']
    models_exist = all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in model_files)
    tab1, tab2, tab3 = st.tabs(["Huấn luyện mô hình", "Giả định", "Hỗ trợ đầu tư"])
    with tab1:
        st.write("## Huấn luyện và đánh giá mô hình")
        train = st.button("Bắt đầu huấn luyện", use_container_width=True)
        if train:
            with st.spinner("Đang huấn luyện mô hình XGBoost..."):
                try:
                    metrics = train_and_save_models(SOURCE_PATH, MODEL_DIR, return_metrics=True)
                    st.session_state.models_trained = True
                    st.session_state.training_metrics = metrics
                    st.success("Mô hình đã được huấn luyện và lưu thành công!")
                    models_exist = True
                except Exception as e:
                    st.error(f"Lỗi : {e}")
        if models_exist or st.session_state.get('models_trained', False):
            st.markdown("---")
            st.write("### Thông tin mô hình")

            if 'training_metrics' in st.session_state:
                metrics = st.session_state.training_metrics
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tập huấn luyện", f"{metrics['train_size']} ngày")
                with col2:
                    st.metric("Tập kiểm tra", f"{metrics['test_size']} ngày")
                with col3:
                    ratio = metrics['train_size'] / (metrics['train_size'] + metrics['test_size'])
                    st.metric("Tỷ lệ Train/Test", f"{ratio:.0%}")
                
                st.markdown("---")

                st.write("### Hiệu suất phân loại (Classification)")
                clf_metrics = metrics['clf_metrics']

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    acc = np.mean(clf_metrics['accuracy'])
                    st.metric("Accuracy", f"{acc:.2%}")
                with col2:
                    prec = np.mean(clf_metrics['precision'])
                    st.metric("Precision", f"{prec:.2%}")
                with col3:
                    rec = np.mean(clf_metrics['recall'])
                    st.metric("Recall", f"{rec:.2%}")
                with col4:
                    f1 = np.mean(clf_metrics['f1'])
                    st.metric("F1-Score", f"{f1:.2%}")

                st.write("### Hiệu suất hồi quy (Regression)")
                reg_metrics = metrics['reg_metrics']

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    mae = np.mean(reg_metrics['mae'])
                    st.metric("MAE", f"{mae:.4f}")
                with col2:
                    rmse = np.mean(reg_metrics['rmse'])
                    st.metric("RMSE", f"{rmse:.4f}")
                with col3:
                    r2 = np.mean(reg_metrics['r2'])
                    st.metric("R-squared Score", f"{r2:.4f}")
                with col4:
                    dir_acc = np.mean(reg_metrics['direction_acc'])
                    st.metric("Direction Accuracy", f"{dir_acc:.2%}")

                st.write("### Top 10 Features quan trọng nhất")
                fi = metrics['feature_importance'].head(10)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(fi['feature'], fi['importance'], color='steelblue')
                ax.set_xlabel('Importance Score', fontsize=12)
                ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.write("### Kết quả Backtest")
                bt = metrics['backtest_results']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Strategy Return", f"{bt['total_return']*100:+.2f}%")
                with col2:
                    st.metric("Buy & Hold Return", f"{bt['buy_hold_return']*100:+.2f}%")
                with col3:
                    st.metric("Win Rate", f"{bt['win_rate']*100:.1f}%")
                with col4:
                    st.metric("Sharpe Ratio", f"{bt['sharpe_ratio']:.2f}")

                st.metric("Số lượng giao dịch", bt['num_trades'])
            else:
                st.info("Huấn luyện mô hình để xem chi tiết metrics.")
    
    with tab2:
        st.write("## Mô phỏng đầu tư")
        
        if not models_exist:
            st.warning("Vui lòng huấn luyện mô hình trước (Tab 'Training & Metrics')")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                investment_amount = st.number_input(
                    "Số tiền muốn đầu tư (USD):",
                    min_value=100,
                    max_value=1000000,
                    value=10000,
                    step=1000
                )
            with col2:
                st.write("")
                st.write("")
                simulate_btn = st.button("Bắt đầu chạy mô phỏng", use_container_width=True)
            
            if simulate_btn:
                with st.spinner("Đang chạy mô phỏng đầu tư..."):
                    try:
                        # Load models and make predictions
                        clf = joblib.load(os.path.join(MODEL_DIR, 'xgb_classifier.joblib'))
                        reg = joblib.load(os.path.join(MODEL_DIR, 'xgb_regressor.joblib'))
                        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
                        
                        # Prepare test data
                        split_idx = int(len(df) * 0.8)
                        test = df.iloc[split_idx:].dropna()
                        
                        excluded = ['Next_Return','Next_Direction','symbol']
                        feature_cols = [c for c in df.columns if c not in excluded]
                        
                        X_test = test[feature_cols]
                        X_test_s = scaler.transform(X_test)
                        
                        # Make predictions
                        test_proba = clf.predict_proba(X_test_s)[:,1]
                        threshold = 0.5
                        test_signals = np.where(test_proba >= threshold, 1, 0)
                        
                        # Run backtest
                        backtest_results = backtest_realistic(test, test_signals, transaction_cost=0.001)
                        
                        # Calculate investment returns
                        strategy_return = backtest_results['total_return']
                        buyhold_return = backtest_results['buy_hold_return']
                        
                        final_value_strategy = investment_amount * (1 + strategy_return)
                        final_value_buyhold = investment_amount * (1 + buyhold_return)
                        
                        profit_strategy = final_value_strategy - investment_amount
                        profit_buyhold = final_value_buyhold - investment_amount
                        
                        # Display results
                        st.success(" Mô phỏng hoàn tất!")
                        
                        st.write("### Kết quả đầu tư")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Chiến lược AI")
                            st.metric("Vốn ban đầu", f"${investment_amount:,.2f}")
                            st.metric("Giá trị cuối", f"${final_value_strategy:,.2f}", 
                                     f"{profit_strategy:+,.2f}")
                            st.metric("Lợi nhuận %", f"{strategy_return*100:+.2f}%")
                            st.metric("Win Rate", f"{backtest_results['win_rate']*100:.1f}%")
                            st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
                            st.metric("Số giao dịch", backtest_results['num_trades'])
                        
                        with col2:
                            st.markdown("#### Buy & Hold")
                            st.metric("Vốn ban đầu", f"${investment_amount:,.2f}")
                            st.metric("Giá trị cuối", f"${final_value_buyhold:,.2f}", 
                                     f"{profit_buyhold:+,.2f}")
                            st.metric("Lợi nhuận %", f"{buyhold_return*100:+.2f}%")
                            st.write("")
                            st.write("")
                            st.write("")
                        
                        # Comparison
                        diff = profit_strategy - profit_buyhold
                        if diff > 0:
                            st.success(f"Chiến lược AI tốt hơn Buy & Hold: +${diff:,.2f}")
                        else:
                            st.warning(f"Chiến lược AI kém hơn Buy & Hold: ${diff:,.2f}")
                        
                        # Visualize equity curves
                        st.write("### Biểu đồ hiệu suất")
                        
                        equity_curve = backtest_results['daily_equity']
                        equity_strategy = np.array(equity_curve) * investment_amount
                        n = len(equity_strategy)
                        equity_buyhold = (test['close'].iloc[:n] / test['close'].iloc[0]).values * investment_amount
                        time_index = test.index[:n]

                        fig, ax = plt.subplots(figsize=(14, 7))
                        ax.plot(test.index[:len(equity_strategy)], equity_strategy, 
                               label=f'AI Strategy (${final_value_strategy:,.0f})', 
                               color='blue', linewidth=2.5)
                        ax.plot(test.index[:len(equity_buyhold)], equity_buyhold[:len(equity_strategy)], 
                               label=f'Buy & Hold (${final_value_buyhold:,.0f})', 
                               color='gray', linestyle='--', linewidth=2, alpha=0.7)
                        ax.axhline(investment_amount, color='red', linestyle=':', 
                                  label='Vốn ban đầu', alpha=0.5)
                        ax.set_xlabel("Thời gian", fontsize=12)
                        ax.set_ylabel("Giá trị tài sản ($)", fontsize=12)
                        ax.set_title("So sánh hiệu suất: AI Strategy vs Buy & Hold", 
                                    fontsize=14, fontweight='bold')
                        ax.legend(fontsize=11)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Trade signals visualization
                        st.write("### Điểm vào/ra lệnh")
                        
                        trades = backtest_results['trades']
                        buy_dates = [t['date'] for t in trades if t['type'] == 'BUY']
                        buy_prices = [t['price'] for t in trades if t['type'] == 'BUY']
                        sell_dates = [t['date'] for t in trades if t['type'] == 'SELL']
                        sell_prices = [t['price'] for t in trades if t['type'] == 'SELL']
                        
                        fig, ax = plt.subplots(figsize=(14, 7))
                        ax.plot(test.index, test['close'], label='Giá cổ phiếu', 
                               color='black', alpha=0.5, linewidth=1.5)
                        ax.scatter(buy_dates, buy_prices, marker='^', color='green', 
                                  s=150, label='Tín hiệu MUA', zorder=5, edgecolors='darkgreen', linewidths=2)
                        ax.scatter(sell_dates, sell_prices, marker='v', color='red', 
                                  s=150, label='Tín hiệu BÁN', zorder=5, edgecolors='darkred', linewidths=2)
                        ax.set_xlabel("Thời gian", fontsize=12)
                        ax.set_ylabel("Giá ($)", fontsize=12)
                        ax.set_title("Điểm vào/ra lệnh trên biểu đồ giá", 
                                    fontsize=14, fontweight='bold')
                        ax.legend(fontsize=11)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                    except Exception as e:
                        st.error(f"Lỗi khi chạy mô phỏng: {str(e)}")
    with tab3:
        st.header("Mô hình dự đoán - hỗ trợ quyết định đầu tư")
        st.write("### Dự đoán giá cổ phiếu bằng Monte Carlo")
        days = st.number_input("Bạn muốn xem giá cổ phiếu trong vòng bao nhiêu ngày ?", min_value=1, max_value=91, value=10)
        check = st.button("Bắt đầu xem")
        if check:
            with st.spinner('Đang chạy mô phỏng...'): # Thêm hiệu ứng quay
                st.session_state.prediction_result = predict_days(days)
                st.session_state.has_run = True
                st.session_state['support_check'] = False
        
        if st.session_state.get('has_run', False):
            res = st.session_state.prediction_result
            
            future_dates = res["future_dates"]
            simulation_df = res["simulations"]
            last_close = res["last_close"]

            ending_values = simulation_df.iloc[-1]
            avg_price = ending_values.mean()
            max_price = ending_values.quantile(0.95)
            min_price = ending_values.quantile(0.05)

            roi_avg = (avg_price - last_close) / last_close * 100

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(future_dates, simulation_df.iloc[:, :50], color='gray', alpha=0.1)
            ax.plot(future_dates, simulation_df.mean(axis=1), linewidth=3, label="Trung bình dự kiến")
            ax.axhline(last_close, linestyle="--", color="red", label="Giá hiện tại")
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            st.write(f"### Kết quả dự báo sau {days} ngày:")
            st.write(f"Giá trung bình dự kiến: ${avg_price:.2f}")
            st.write(f"Kịch bản lạc quan (95%): ${max_price:.2f}")
            st.write(f"Kịch bản bi quan (5%): ${min_price:.2f}")

            st.markdown("---")
            pred = st.checkbox("Bạn có muốn chúng tôi hỗ trợ bạn ra quyết định ?", key='support_check')
            if pred:
                if roi_avg > 1.5: 
                    st.success(f"KHUYẾN NGHỊ: MUA (Lợi nhuận kỳ vọng cao: {roi_avg:.2f}%)")
                elif roi_avg < -2:
                    st.error(f"KHUYẾN NGHỊ: BÁN (Rủi ro giảm giá lớn: {roi_avg:.2f}%)")
                else:
                    st.info(f"KHUYẾN NGHỊ: NẮM GIỮ / QUAN SÁT (Biên độ nhỏ: {roi_avg:.2f}%)")
                
            st.success("Quá trình dự đoán hoàn tất !")