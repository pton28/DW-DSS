import os
import duckdb
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "Data_warehouse", "staging.db")

def inspect_database():
    print(f"🔍 Đang kết nối tới: {DB_PATH}")
    
    if not os.path.exists(DB_PATH):
        print("❌ Lỗi: File database không tồn tại! Hãy chạy Loading.py trước.")
        return

    # Mở kết nối ở chế độ READ_ONLY=True để an toàn, tránh lỗi database lock
    conn = duckdb.connect(DB_PATH, read_only=True)

    try:
        # 1. Kiểm tra danh sách các bảng
        print("\n" + "="*40)
        print("📂 DANH SÁCH BẢNG (TABLES)")
        print("="*40)
        tables = conn.execute("SHOW TABLES").df()
        if tables.empty:
            print("⚠️ Database rỗng, chưa có bảng nào.")
            return
        print(tables)

        # Lấy tên bảng đầu tiên tìm thấy (thường là fact_stocks)
        table_name = tables.iloc[0, 0]

        # 2. Xem cấu trúc bảng (Schema)
        print("\n" + "="*40)
        print(f"🏗️  CẤU TRÚC BẢNG: {table_name}")
        print("="*40)
        # DESCRIBE giúp xem tên cột và kiểu dữ liệu (DOUBLE, VARCHAR, DATE...)
        schema = conn.execute(f"DESCRIBE {table_name}").df()
        print(schema[['column_name', 'column_type']])

        # 3. Xem mẫu 5 dòng dữ liệu đầu tiên
        print("\n" + "="*40)
        print(f"👀 5 DÒNG DỮ LIỆU ĐẦU TIÊN")
        print("="*40)
        # query().df() trả về Pandas DataFrame nhìn rất đẹp
        print(conn.execute(f"SELECT * FROM {table_name} LIMIT 5").df())

        # 4. Thống kê số lượng theo mã chứng khoán
        print("\n" + "="*40)
        print(f"📊 THỐNG KÊ DỮ LIỆU")
        print("="*40)
        query_stats = f"""
            SELECT 
                Symbol, 
                COUNT(*) as Total_Rows, 
                MIN(Date) as Start_Date, 
                MAX(Date) as End_Date,
                ROUND(AVG(Close), 2) as Avg_Close
            FROM {table_name}
            GROUP BY Symbol
            ORDER BY Symbol
        """
        print(conn.execute(query_stats).df())

    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")
    finally:
        conn.close()
        print("\n✅ Đã đóng kết nối.")

if __name__ == "__main__":
    inspect_database()
