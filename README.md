# 📊 DW-DSS: Data Warehouse & Decision Support System for Stock Price Forecasting

Hệ thống **kho dữ liệu tích hợp** cho dự báo giá cổ phiếu sử dụng **ETL Pipeline**, **Machine Learning** (Linear Regression), và **Monte Carlo Simulation**.

---

## 📁 Cấu Trúc Dự Án

```
DW-DSS/
│
├── 📂 Dataset/                          # Dữ liệu theo các tầng (Bronze → Silver → Gold)
│   ├── Raw/                             # Bronze: Dữ liệu thô từ API (chưa xử lý)
│   │   ├── googl_balance_sheet.csv
│   │   ├── googl_cash_flow_statement.csv
│   │   ├── googl_daily_prices.csv
│   │   └── googl_income_statement.csv
│   ├── Cleaned/                         # Silver: Dữ liệu làm sạch + Feature Engineering
│   │   ├── balance_sheet_cleaned.csv
│   │   ├── cash_flow_cleaned.csv
│   │   ├── income_statement_cleaned.csv
│   │   ├── prices_cleaned.csv
│   │   └── GOOG_cleaned.csv            
│   └── Gold/                            # Gold: Dữ liệu chuẩn hóa (Star Schema)
│       ├── Dims/                        # Bảng Dimension
│       │   ├── dim_company.csv
│       │   ├── dim_date.csv
│       │   ├── dim_fin_metric.csv
│       │   ├── dim_fin_statement_type.csv
│       │   └── dim_stock_metric.csv
│       └── Facts/                       # Bảng Fact
│           ├── fact_finance.csv
│           └── fact_stock_prices.csv
│
├── 📂 Data_warehouse/                   # Kho dữ liệu cuối cùng
│   └── dw.duckdb                        # DuckDB database (nếu dùng)
│
├── 📂 ETL/                              # Pipeline Trích xuất → Chuyển đổi → Nạp
│   ├── Extracting.py                    # 🔹 Tải dữ liệu từ Yahoo Finance API
│   ├── Transforming.py                  # 🔹 Làm sạch + Feature Engineering
│   ├── Loading.py                       # 🔹 Hợp nhất + Lưu vào warehouse
│   └── Run_ETL.py                       # 🔹 Điều phối chạy toàn bộ pipeline
│
├── 📂 Train_model/                      # Machine Learning & Prediction
│   ├── predict_model.py                 # 🔹 Huấn luyện Linear Regression
│   ├── predict.py                       # 🔹 Dự báo giá (Monte Carlo 2000 simulations)
│   │                                    #    - Tương tác nhập n ngày
│   │                                    #    - Vẽ biểu đồ 2000 paths + trung bình
│   │                                    #    - Thống kê (trung bình, percentile 5%-95%)
│   ├── best_stock_price_model.pkl       # Model hồi quy giá
│   ├── __init__.py                      # RobustScaler cho feature normalization
│   └── Image/                           # Hình ảnh mô hình/biểu đồ
│
├── 📂 streamlit/                        # Dashboard & Visualization
│   ├── streamlit.py                     # 🔹 Web app Streamlit
│   │                                    #    - Vẽ trend + technical indicators
│   │                                    #    - Hiển thị dự báo
│   └── Image/                           # Assets (logo, icon)
│
├── 📂 Visualization/                    # Các script visualization thêm
│   └── visualization.py                 # Vẽ biểu đồ chi tiết
│
├── 📂 logs/                             # Logs & output
│
├── 📄 requirements.txt                  # 📦 Danh sách thư viện cần thiết
├── 📄 SQL_script.sql                    # SQL queries (nếu dùng DuckDB/SQL)
├── 📄 README.md                         # 📖 Tài liệu này
└── 📄 .gitignore                        # Git ignore rules

```

---

## 🎯 Tính Năng Chính

### 1️⃣ **ETL Pipeline** (Extract → Transform → Load)
- **Extracting:** Tải dữ liệu từ Yahoo Finance (giá cổ phiếu, báo cáo tài chính)
- **Transforming:** 
  - Làm sạch, chuẩn hóa cột
  - Tạo 60+ technical indicators (MA, RSI, ATR, Bollinger Bands, Stochastic, Candle patterns, lags, volume metrics)
  - Kiểm soát data leakage (tất cả features dùng shift(1) - chỉ dữ liệu quá khứ)
- **Loading:** Hợp nhất dữ liệu → kho warehouse

### 2️⃣ **Machine Learning** (XGBoost)
- **Huấn luyện:**
  - XGBoost Classifier: Dự báo hướng giá (UP/DOWN)
  - XGBoost Regressor: Dự báo giá cụ thể
  - Walk-forward expanding validation (4 folds)
  - Thực tế backtest (entry/exit at next open)
  
- **Hiệu suất:**
  - Sharpé ratio, Win rate, Profit factor

### 3️⃣ **Dự Báo & Mô Phỏng** (Prediction)
- **Interactive Prediction:**
  - Người dùng nhập n ngày muốn dự báo
  - Chạy Monte Carlo 1000 simulations
  - Hiển thị:
    - 1000 đường giá (semi-transparent)
    - Đường trung bình (bold)
    - Vùng tin cậy 90% (5%-95% percentile)
    - Phân phối xác suất giá cuối
    - Khuyến nghị MUA/BÁN/QUAN SÁT

### 4️⃣ **Dashboard & Visualization** (Streamlit)
- Trend charts + Technical indicators
- Performance metrics
- Interactive forecasting

---

## 🚀 Cài Đặt & Sử Dụng

### 1. **Cài Đặt Môi Trường**

**Python 3.8+** (khuyến nghị 3.10+)

```bash
# Clone repo
git clone https://github.com/pton28/DW-DSS.git
cd DW-DSS

# Tạo virtual environment
python -m venv .venv

# Kích hoạt venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Cài thư viện
pip install -r requirements.txt
```

### 2. **Chạy ETL Pipeline**

```bash
# Trích xuất → Chuyển đổi → Nạp
python ETL/Run_ETL.py

# Kết quả: GOOG_cleaned.csv được tạo trong Dataset/Cleaned/
```

### 3. **Huấn Luyện Mô Hình**

```bash
cd Train_model
python xgboost_model.py

# Output: 
#   - xgb_classifier.joblib
#   - xgb_regressor.joblib
#   - scaler.joblib
#   - Backtest results + metrics
```

### 4. **Dự Báo Giá (Interactive)**

```bash
cd Train_model
python predict.py

# Nhập: số ngày (VD: 5, 10, 30)
# Output: 
#   - Biểu đồ Monte Carlo (1000 simulations)
#   - Thống kê giá
#   - Khuyến nghị
```

### 5. **Chạy Dashboard Streamlit**

```bash
cd streamlit
streamlit run streamlit.py

# Mở browser → http://localhost:8501
```

---

## 📦 Thư Viện Chính

| Thư viện | Mục đích |
|---------|---------|
| `pandas` | Xử lý dữ liệu |
| `numpy` | Tính toán số học |
| `yfinance` | Tải dữ liệu từ Yahoo Finance |
| `xgboost` | Machine Learning (gradient boosting) |
| `scikit-learn` | Feature scaling, validation |
| `joblib` | Lưu/load model |
| `matplotlib`, `plotly` | Visualization |
| `streamlit` | Web dashboard |
| `duckdb` | Kho dữ liệu (tùy chọn) |

---

## 📊 Dữ Liệu

### **Nguồn Dữ Liệu**
- **Yahoo Finance API** (yfinance)
- **Cổ phiếu:** Google (GOOGL)
- **Loại dữ liệu:** 
  - Daily stock prices (OHLCV)
  - Financial statements (Balance Sheet, Income, Cash Flow)

### **Chu Kỳ Dữ Liệu**
- **Giá cổ phiếu:** 2016-06-14 → 2021-06-11 (1259 ngày giao dịch)
- **Báo cáo tài chính:** Hàng quý

### **Cột Chính (GOOG_cleaned.csv)**
| Cột | Kiểu | Mô Tả |
|-----|------|-------|
| `date` | datetime | Ngày giao dịch |
| `symbol` | object | Mã cổ phiếu |
| `close` | float64 | Giá đóng cửa |
| `volume` | int64 | Khối lượng giao dịch |
| `return` | float64 | Lợi suất ngày |
| `MA_5`, `MA_20` | float64 | Moving Average 5/20 ngày |
| `RSI`, `ATR`, `MACD` | float64 | Technical Indicators |
| ...và 50+ features khác | | Lags, Bollinger Bands, Stochastic, v.v. |

---

## 🔧 Hướng Dẫn Chi Tiết

### **A. Xử Lý Lỗi ETL**

**Lỗi:** `Missing column 'date' in parse_dates`
- **Nguyên nhân:** File không có cột `date` hoặc tên cột khác
- **Khắc phục:** Transforming.py tự động phát hiện cột ngày (date, datetime, timestamp, time)

**Lỗi:** `name 'df' is not defined`
- **Nguyên nhân:** File read thất bại nhưng không bỏ qua
- **Khắc phục:** Loading.py thêm try-except, bỏ qua file lỗi

### **B. Feature Engineering**

Tất cả features được tạo từ dữ liệu **quá khứ** (shift(1)) để tránh **data leakage**:

```python
# ❌ Lỗi (data leakage):
df['future_return'] = df['return'].shift(-1)  # Biết tương lai!

# ✅ Đúng (dùng dữ liệu quá khứ):
df['past_return'] = df['return'].shift(1)     # Chỉ dùng quá khứ
```

### **C. Dự Báo & Monte Carlo**

**Cơ chế:**
1. Load dữ liệu + models (XGBoost)
2. Dự báo hướng giá + return ngày kế tiếp
3. Tính toán historical drift & volatility (30 ngày)
4. Chạy 1000 independent MC simulations:
   - `price_{t+1} = price_t × (1 + historical_drift + normal(0, volatility))`
5. Tính thống kê (trung bình, percentile, vùng tin cậy)
6. Vẽ biểu đồ + khuyến nghị

---

## 📈 Kết Quả Ví Dụ

```
==============================================================
🔮 KẾT QUẢ DỰ ĐOÁN GIÁ CỔ PHIẾU GOOGLE (GOOG)
==============================================================

📊 DỮ LIỆU CƠ SỞ
  • Giá đóng cửa hiện tại: $1234.56
  • Độ biến động (30 ngày): 2.1234%
  • Drift trung bình: +0.0512%
  • Dự báo hướng AI: 📈 TĂNG (1)

📈 KẾT QUẢ DỰ ĐOÁN SAU 5 NGÀY (Monte Carlo 1000 simulations)
  • Giá trung bình dự kiến: $1267.89 (+2.70%)
  • Kịch bản lạc quan (95%): $1312.45
  • Kịch bản bi quan (5%):  $1218.34
  • Khoảng dao động: $1218.34 - $1312.45

💡 KHUYẾN NGHỊ
  ✅ NÊN MUA - Lợi nhuận kỳ vọng cao (+2.70%)
==============================================================
```

---

## 🛠️ Troubleshooting

### **Lỗi import Train_model từ Streamlit**
```
ModuleNotFoundError: No module named 'Train_model'
```

**Giải pháp:** Streamlit.py đã được sửa (import sys + sys.path.insert)

```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Train_model.predict import run_prediction
```

### **Mô hình không tìm thấy**
- Chạy `Train_model/xgboost_model.py` trước

### **Dữ liệu không tìm thấy**
- Chạy `ETL/Run_ETL.py` để tạo GOOG_cleaned.csv

### **Các gói bị thiếu**
```bash
pip install -r requirements.txt --upgrade
```

---

## 📝 Ghi Chú Phát Triển

- ✅ ETL Pipeline: Hoàn tất
- ✅ Feature Engineering: 60+ indicators
- ✅ Monte Carlo Prediction: 2000 simulations + visualization
- ✅ Streamlit Dashboard: Web UI
- 🔲 API REST: (Tương lai)
- 🔲 Real-time prediction: (Tương lai)

---

## 👤 Tác Giả & Liên Hệ

**Repository:** [DW-DSS](https://github.com/pton28/DW-DSS)

---

## 📜 License

MIT License - Tự do sử dụng & sửa đổi

---

**Cập nhật lần cuối:** December 2025