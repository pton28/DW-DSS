DW-DSS/
├── Data_warehouse/      # Nơi chứa dữ liệu cuối cùng (Gold Zone)
│   ├── all_stocks.csv   # File tổng hợp tất cả mã cổ phiếu
│   └── staging.db       # Cơ sở dữ liệu lưu trữ (nếu có dùng DuckDB/SQLite)
├── Dataset/             # Nơi chứa dữ liệu thô và đã xử lý
│   ├── Raw/             # Dữ liệu gốc tải từ API (Bronze Zone)
│   └── Cleaned/         # Dữ liệu đã làm sạch cơ bản (Silver Zone)
├── ETL/                 # Mã nguồn quy trình ETL
│   ├── Extracting.py    # Tải dữ liệu từ Yahoo Finance
│   ├── Transforming.py  # Làm sạch, chuẩn hóa cột
│   ├── Loading.py       # Gộp và lưu vào Data Warehouse
│   └── Run_ETL.py       # Script điều phối chạy toàn bộ quy trình
├── Train model/         # (Đang phát triển) Nơi chứa code mô hình AI/ML
├── check.py             # Script kiểm tra nhanh dữ liệu/môi trường
├── requirement.txt      # Danh sách các thư viện cần thiết
└── README.md            # Tài liệu hướng dẫn

🛠️ Quy trình ETL (Extract - Transform - Load)

Extract (Trích xuất):
    - Sử dụng yfinance để tải dữ liệu lịch sử (OHLCV).
    - Phạm vi: Các mã cổ phiếu lớn (AAPL, MSFT, AMZN, NVDA...).
    - Đầu ra: File .csv trong thư mục Dataset/Raw.

Transform (Chuyển đổi):
    - Chuẩn hóa tên cột (về dạng chữ thường: date, open, close...).
    - Loại bỏ các cột thừa (Dividends, Stock Splits).
    - Xử lý giá trị NaN (Forward/Backward Fill).
    - Đầu ra: File .csv trong thư mục Dataset/Cleaned.

Load (Tải):
    - Hợp nhất tất cả các file đã làm sạch.
    - Lọc dữ liệu theo thời gian và cấu trúc chuẩn.
    - Lưu vào Kho dữ liệu chính: Data_warehouse/all_stocks.csv.

⚙️ Cài đặt & Sử dụng
1. Yêu cầu hệ thống
    - Python 3.8+
    - Các thư viện: pandas, yfinance, duckdb (nếu dùng), scikit-learn (cho phần model).

2. Cài đặt thư viện
Chạy lệnh sau để cài các gói cần thiết:
    pip install -r requirement.txt

3. Chạy quy trình ETL
Để cập nhật dữ liệu mới nhất, chạy file điều phối:
    python ETL/Run_ETL.py

Hệ thống sẽ tự động thực hiện 3 bước và báo cáo kết quả trên terminal.

📊 Dữ liệu mục tiêu
Dữ liệu bao gồm các trường thông tin chính:
    - Date: Ngày giao dịch.
    - Symbol: Mã cổ phiếu (VD: AAPL).
    - Open/High/Low/Close: Các mức giá trong ngày.
    - Volume: Khối lượng giao dịch.