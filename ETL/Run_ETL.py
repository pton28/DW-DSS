import subprocess
import sys
import os
import time

def run_script(script_name):
    """Hàm chạy một file python con và kiểm tra lỗi."""
    print(f"\n{'='*40}")
    print(f"🚀 Đang chạy: {script_name}...")
    print(f"{'='*40}")
    
    start_time = time.time()
    
    # Kiểm tra file có tồn tại không
    if not os.path.exists(script_name):
        print(f"❌ Lỗi: Không tìm thấy file {script_name}")
        return False

    try:
        # Chạy script và chờ nó kết thúc
        result = subprocess.run([sys.executable, script_name], check=True)
        
        duration = time.time() - start_time
        print(f"✅ Hoàn thành {script_name} trong {duration:.2f} giây.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi chạy {script_name}. Mã lỗi: {e.returncode}")
        return False

def main():
    print("🏁 BẮT ĐẦU QUY TRÌNH ETL CHỨNG KHOÁN 🏁")
    
    # Danh sách các bước theo thứ tự
    scripts = ["Extracting.py", "Transforming.py", "Loading.py"]
    
    for script in scripts:
        success = run_script(script)
        if not success:
            print("\n🛑 Quy trình ETL bị dừng do lỗi.")
            break
    else:
        # Chỉ chạy khi vòng lặp không bị break (tức là tất cả đều thành công)
        print("\n🎉 TOÀN BỘ QUY TRÌNH ETL ĐÃ HOÀN TẤT THÀNH CÔNG.")
        print(f"📂 Kiểm tra dữ liệu cuối cùng tại thư mục: ../Data_warehouse/")

if __name__ == "__main__":
    main()