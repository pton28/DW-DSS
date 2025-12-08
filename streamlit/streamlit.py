import os
import sys
import warnings
import pandas as pd
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
# Thêm đường dẫn cha vào sys.path để import Train_model từ cấp cha
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Train_model.predict import run_prediction # Import file predict.py nằm cùng thư mục
# Cấu hình trang
st.set_page_config(page_title="Financial Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
GOLD_DIR = os.path.join(PROJECT_ROOT, "Dataset", "Gold")
DIM_DIR = os.path.join(GOLD_DIR, "Dims")
FACT_DIR = os.path.join(GOLD_DIR, "Facts")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    """Load dữ liệu từ các file CSV Star Schema"""
    try:
        # Load Dimensions
        dim_company = pd.read_csv(os.path.join(DIM_DIR, "dim_company.csv"))
        dim_date = pd.read_csv(os.path.join(DIM_DIR, "dim_date.csv"))
        dim_fin_metric = pd.read_csv(os.path.join(DIM_DIR, "dim_fin_metric.csv"))
        dim_fin_statement_type = pd.read_csv(os.path.join(DIM_DIR, "dim_fin_statement_type.csv"))
        dim_stock_metric = pd.read_csv(os.path.join(DIM_DIR, "dim_stock_metric.csv"))
        
        # Load Facts
        fact_finance = pd.read_csv(os.path.join(FACT_DIR, 'fact_finance.csv'))
        fact_stock_prices = pd.read_csv(os.path.join(FACT_DIR, 'fact_stock_prices.csv'))
        
        # Xử lý kiểu dữ liệu ngày tháng
        dim_date['date'] = pd.to_datetime(dim_date['date'])
        
        return {
            'dim_company': dim_company,
            'dim_date': dim_date,
            'dim_fin_metric': dim_fin_metric,
            'dim_fin_statement_type': dim_fin_statement_type,
            'dim_stock_metric': dim_stock_metric,
            'fact_finance': fact_finance,
            'fact_stock_prices': fact_stock_prices
        }
    except FileNotFoundError as e:
        st.error(f"Thiếu file dữ liệu: {e}")
        return None

data = load_data()

if data:
    # --- 2. DATA PROCESSING (RECONSTRUCT) ---
    
    # A. Xử lý dữ liệu Giá cổ phiếu (Fact -> Wide Table)
    def get_stock_df(company_ticker):
        # 1. Lấy company_key
        comp_info = data['dim_company'][data['dim_company']['ticker'] == company_ticker]
        if comp_info.empty: return pd.DataFrame()
        comp_key = comp_info.iloc[0]['company_key']
        
        # 2. Filter Fact table
        df_fact = data['fact_stock_prices'][data['fact_stock_prices']['company_key'] == comp_key]
        
        # 3. Join với Dim Date và Dim Stock Metric
        df_merged = df_fact.merge(data['dim_date'], on='date_key')
        df_merged = df_merged.merge(data['dim_stock_metric'], on='stock_metric_key')
        
        # 4. Pivot: Chuyển metric_name thành cột (open, high, low, close...)
        df_pivot = df_merged.pivot_table(
            index='date', 
            columns='metric_name', 
            values='value'
        ).reset_index()
        
        # Thêm cột symbol để tương thích với các module khác
        df_pivot['symbol'] = company_ticker
        df_pivot = df_pivot.sort_values('date')
        
        return df_pivot

    # B. Xử lý dữ liệu Tài chính
    def get_financial_df(company_ticker, statement_type_key):
        # 1. Lấy company_key
        comp_info = data['dim_company'][data['dim_company']['ticker'] == company_ticker]
        if comp_info.empty: return pd.DataFrame()
        comp_key = comp_info.iloc[0]['company_key']
        
        # 2. Filter Fact table theo Company và Loại báo cáo
        df_fact = data['fact_finance'][
            (data['fact_finance']['company_key'] == comp_key) & 
            (data['fact_finance']['fin_type_key'] == statement_type_key)
        ]
        
        # 3. Join với Dim Date và Dim Fin Metric
        df_merged = df_fact.merge(data['dim_date'], on='date_key')
        df_merged = df_merged.merge(data['dim_fin_metric'], on='metric_key')
        
        # 4. Pivot để hiển thị đẹp hơn (Mỗi chỉ số là 1 dòng, Cột là Quý/Năm)
        # Tuy nhiên để vẽ biểu đồ, ta giữ dạng Long hoặc Pivot theo Metric
        return df_merged.sort_values('date')

    # --- 3. SIDEBAR ---
    st.sidebar.header("Cấu hình Dashboard")
    
    # Chọn công ty
    company_list = data['dim_company']['ticker'].unique()
    selected_ticker = st.sidebar.selectbox("Chọn Mã Cổ Phiếu", company_list)
    
    # Lấy dữ liệu đã xử lý cho công ty được chọn
    stock_df = get_stock_df(selected_ticker)
    
    # Chọn khoảng thời gian
    min_date = stock_df['date'].min().date()
    max_date = stock_df['date'].max().date()
    
    start_date, end_date = st.sidebar.date_input(
        "Khoảng thời gian", 
        value=[min_date, max_date],
        min_value=min_date, 
        max_value=max_date
    )
    
    # Filter theo ngày
    mask = (stock_df['date'].dt.date >= start_date) & (stock_df['date'].dt.date <= end_date)
    filtered_stock_df = stock_df.loc[mask]

    # --- 4. MAIN UI ---
    st.title(f"📊 Dashboard Tài Chính: {selected_ticker}")
    
    # TABS
    tab1, tab2, tab3 = st.tabs(["📈 Biến động Giá", "💰 Báo cáo Tài chính", "🤖 Dự báo AI"])
    
    # === TAB 1: STOCK PRICES ===
    with tab1:
        # Metrics hàng đầu
        latest_data = stock_df.iloc[-1]
        prev_data = stock_df.iloc[-2]
        change = latest_data['close'] - prev_data['close']
        pct_change = (change / prev_data['close']) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Giá Đóng Cửa", f"${latest_data['close']:.2f}", f"{pct_change:.2f}%")
        col2.metric("Giá Mở Cửa", f"${latest_data['open']:.2f}")
        col3.metric("Cao Nhất", f"${latest_data['high']:.2f}")
        col4.metric("Thấp Nhất", f"${latest_data['low']:.2f}")
        
        # Biểu đồ Nến (Candlestick)
        fig = go.Figure(data=[go.Candlestick(
            x=filtered_stock_df['date'],
            open=filtered_stock_df['open'],
            high=filtered_stock_df['high'],
            low=filtered_stock_df['low'],
            close=filtered_stock_df['close']
        )])
        fig.update_layout(title="Biểu đồ Giá Cổ Phiếu", xaxis_title="Ngày", yaxis_title="Giá (USD)")
        st.plotly_chart(fig, width='stretch')
        
        # Biểu đồ Volume
        st.subheader("Khối lượng giao dịch")
        fig_vol = px.bar(filtered_stock_df, x='date', y='volume')
        st.plotly_chart(fig_vol, width='stretch')

    # === TAB 2: FINANCIALS ===
    with tab2:
        st.subheader("Dữ liệu Báo cáo Tài chính")
        
        # Chọn loại báo cáo
        type_map = dict(zip(data['dim_fin_statement_type']['description'], data['dim_fin_statement_type']['fin_type_key']))
        selected_type_name = st.selectbox("Loại báo cáo", list(type_map.keys()))
        selected_type_key = type_map[selected_type_name]
        
        fin_df = get_financial_df(selected_ticker, selected_type_key)
        
        if not fin_df.empty:
            # Pivot để hiển thị dạng bảng: Index=Metric, Columns=Date
            display_df = fin_df.pivot(index='metric_name', columns='date', values='value')
            st.dataframe(display_df)
            
            # Vẽ biểu đồ một số chỉ số quan trọng
            st.subheader("Xu hướng chỉ số tài chính")
            metrics = fin_df['metric_name'].unique()
            selected_metrics = st.multiselect("Chọn chỉ số để vẽ biểu đồ", metrics, default=metrics[:2])
            
            if selected_metrics:
                chart_data = fin_df[fin_df['metric_name'].isin(selected_metrics)]
                fig_fin = px.line(chart_data, x='date', y='value', color='metric_name', markers=True)
                st.plotly_chart(fig_fin, width='stretch')
        else:
            st.info("Không có dữ liệu cho loại báo cáo này.")

    # === TAB 3: PREDICTION ===
    with tab3:
        st.header("Dự báo Giá Cổ Phiếu (AI)")
        
        # Kiểm tra và import module predict
        try:               
            days = st.number_input("Số ngày dự báo:", min_value=10, max_value=200, value=30)
            
            if st.button("Chạy Dự Báo"):
                with st.spinner("Đang tính toán..."):
                    result, err = run_prediction(days, use_monte_carlo=True, num_sims=2000)
                    
                    if err:
                        st.error(f"Lỗi: {err}")
                    else:
                        rec = result.get("ruecommendation")
                        forecast_df = result.get("data")
                        curr_price = result.get("crrent_price")

                        # Hiển thị kết quả
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Giá hiện tại", f"${rec.get('current_price', 0):.2f}")
                        c2.metric("Giá mục tiêu", f"${rec['mean_target_price']:.2f}", f"{rec['expected_profit_percent']:.2f}%")
                        c3.metric("Tình trạng", rec.get('status', 'N/A'))
                        
                        st.info(rec.get('advice', ''))
                        
                        # Biểu đồ dự báo
                        fig = go.Figure()

                        title_text = f'Monte Carlo Simulation - {days} Ngày Tới'
                        chart_df = forecast_df.copy()

                        fig.add_trace(
                            go.Scatter(
                                x=chart_df["date"],
                                y=chart_df["p95_price"],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=chart_df["date"],
                                y=chart_df["p5_price"],
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(173, 216, 230, 0.3)',
                                name='90% Confidence Interval (P5-P95)',
                                hoverinfo='skip'
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=chart_df["date"],
                                y=chart_df["p75_price"],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            )
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=chart_df["date"],
                                y=chart_df["p25_price"],
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor='rgba(144, 238, 144, 0.4)',
                                name='50% Confidence Interval (P25-P75)',
                                hoverinfo='skip'
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=chart_df["date"],
                                y=chart_df["mean_price"],
                                mode='lines',
                                name='Giá Trung Bình (Mean)',
                                line=dict(color='blue', width=2)
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=chart_df["date"],
                                y=chart_df["median_price"],
                                mode='lines',
                                name='Giá Trung Vị (Median)',
                                line=dict(color='green', width=2, dash='dash')
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=chart_df["date"],
                                y=chart_df["p95_price"],
                                mode='lines',
                                name='Kịch bản Tốt (P95)',
                                line=dict(color='lightgreen', width=1, dash='dot')
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=chart_df["date"],
                                y=chart_df["p5_price"],
                                mode='lines',
                                name='Kịch bản Xấu (Pt5)',
                                line=dict(color='red', width=1, dash='dot')
                            )
                        )

                        fig.add_hline(
                            y=curr_price,
                            line_dash="dash",
                            line_color="black",
                            annotation_text=f"Giá Hiện Tại: ${curr_price:.2f}",
                            annotation_position="right"
                        )

                        fig.update_xaxes(title_text="Ngày")
                        fig.update_yaxes(title_text="Giá ($)")
                        
                        fig.update_layout(
                            title=title_text,
                            height=600,
                            hovermode='x unified',
                            legend=dict(
                                orientation="v",
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.subheader("Chi tiết dự báo")
                        display_df = chart_df.copy()
                        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
                        st.dataframe(display_df, use_container_width=True)
                        
        except ImportError:
            st.warning("⚠️ Không tìm thấy module 'predict.py'. Hãy đảm bảo bạn đã upload file này.")
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")

else:
    st.warning("Chưa có dữ liệu. Hãy đảm bảo các file CSV (fact_*, dim_*) nằm cùng thư mục với file này.")