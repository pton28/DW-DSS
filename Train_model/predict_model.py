import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. LOAD DỮ LIỆU ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "Dataset", "Cleaned", "prices_cleaned.csv")

if not os.path.exists(DATA_PATH):
    print(f"❌ Không tìm thấy file {DATA_PATH}")
    exit()

df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# --- 2. CHUẨN BỊ DỮ LIỆU ---
# Dự đoán % thay đổi của ngày hôm sau
df['target'] = df['pct_change'].shift(-1)
df = df.dropna()

features = ['open', 'high', 'low', 
            'close', 'volume', 'pct_change', 
            'daily_return', 'volatility_5d', 'volatility_20d']
X = df[features]
y = df['target']

train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
test_dates = df['date'].iloc[train_size:]

# Lấy giá đóng cửa gốc của tập test để khôi phục giá sau này
test_close_prices = df['close'].iloc[train_size:]

print(f"🔹 Train set: {len(X_train)} | Test set: {len(X_test)}")

# --- CẬP NHẬT HÀM DA ---
def directional_accuracy(y_true, y_pred):
    # Vì y là pct_change, chỉ cần so sánh dấu: Cùng dương (tăng) hoặc cùng âm (giảm)
    matches = (np.sign(y_true) == np.sign(y_pred))
    return np.mean(matches)

# --- 3. HUẤN LUYỆN & SO SÁNH ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,       
        max_depth=10,           
        min_samples_split=20,   
        min_samples_leaf=5,    
        random_state=42,        
        n_jobs=-1               
    )
}

best_model_name = None
best_score = float('inf')
results = {} 
reconstructed_prices = {} 
results_list = []

print("\n📊 Bắt đầu huấn luyện...")

for name, model in models.items():
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test) 
    
    r2_train = r2_score(y_train, train_preds) 
    r2_test = r2_score(y_test, test_preds)
    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    da = directional_accuracy(y_test.values, test_preds) * 100
    score = rmse*100*0.5 + (100-da)*0.5 
    
    results[name] = test_preds

    # Giá dự đoán = Giá hôm nay * (1 + % Tăng trưởng dự đoán)
    price_pred = test_close_prices * (1 + test_preds)
    reconstructed_prices[name] = price_pred

    results_list.append({
        "Model": name,
        "RMSE": rmse,
        "R² train": r2_train,
        "R² test": r2_test,
        "DA (%)": da,
        "Score": score,
        "Gap": r2_train - r2_test
    })
    
    if score < best_score:
        best_score = score
        best_model_name = name
        best_model_obj = model
        
df_results = pd.DataFrame(results_list)
df_results = df_results.sort_values(by="Score")
print("\nBẢNG SO SÁNH MÔ HÌNH (Dựa trên dự đoán % thay đổi):")
print(df_results)

# --- 4. LƯU MODEL ---
print(f"\n🏆 Model tốt nhất: {best_model_name}")
joblib.dump(best_model_obj, 'best_stock_return_model.pkl')

# --- 5. VẼ BIỂU ĐỒ GIÁ (ĐÃ KHÔI PHỤC) ---
# Chú ý: Chúng ta vẽ GIÁ THỰC TẾ của ngày hôm sau (Target Price)
# Giá thực tế hôm sau = Giá đóng cửa hôm nay * (1 + pct_change thực tế)
actual_next_day_price = test_close_prices * (1 + y_test)

plt.figure(figsize=(14, 7))
plt.plot(test_dates, actual_next_day_price, label='Thực tế (Actual Price)', color='black', alpha=0.5)

colors = {'Linear Regression': 'blue', 'Random Forest': 'orange'}
for name, price_pred in reconstructed_prices.items():
    linestyle = '--' if name == best_model_name else ':'
    plt.plot(test_dates, price_pred, label=f'{name} (Predicted)', linestyle=linestyle, color=colors[name])

plt.title(f'Dự đoán Giá Cổ Phiếu (Model dự đoán Return -> Quy đổi ra Price)')
plt.xlabel('Thời gian')
plt.ylabel('Giá (VNĐ)')
plt.legend()
plt.grid(True)
plt.savefig('./Image/model_comparison_chart_reconstructed.png')
plt.show()