import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

# --- 2. CHUẨN BỊ DỮ LIỆU (Pre-processing) ---
# Target: Giá Close của ngày hôm sau
df['target'] = df['close'].shift(-1)
df = df.dropna()

features = ['open', 'high', 'low', 'close', 'volume', 'pct_change', 'daily_return', 'volatility_5d', 'volatility_20d']
X = df[features]
y = df['target']

# Chia tập Train/Test theo thời gian (80/20)
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
test_dates = df['date'].iloc[train_size:]

print(f"🔹 Train set: {len(X_train)} | Test set: {len(X_test)}")

def directional_accuracy(y_true, y_pred):
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return np.mean(true_dir == pred_dir)

# --- 3. HUẤN LUYỆN & SO SÁNH ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

best_model_name = None
best_score = float('inf')
results = {}

print("\n📊 Bắt đầu huấn luyện...")

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    r2 = r2_score(y_test, preds)
    da = directional_accuracy(y_test.values, preds) * 100
    score = rmse*0.5 + (100-da*100)*0.3 + mape*0.2
    
    results[name] = preds
    print(f"\n📌 {name}")
    print(f"   MAE     = {mae:.4f}")
    print(f"   RMSE    = {rmse:.4f}")
    print(f"   MAPE    = {mape:.2f}%")
    print(f"   R-Squared     = {r2:.4f}")
    print(f"   DA      = {da:.2f}%")
    print(f"   SCORE      = {score:.2f}%")
    
    # Tìm model tốt nhất
    if score < best_score:
        best_score = score
        best_model_name = name
        best_model_obj = model

# --- 4. LƯU MODEL TỐT NHẤT ---
print(f"\n🏆 Model tốt nhất: {best_model_name} (RMSE={best_score:.2f})")
save_filename = 'best_stock_price_model.pkl'
joblib.dump(best_model_obj, save_filename)
print(f"💾 Đã lưu model vào file: {save_filename}")

# --- 5. VẼ BIỂU ĐỒ SO SÁNH ---
plt.figure(figsize=(14, 7))
plt.plot(test_dates, y_test, label='Thực tế (Actual)', color='black', alpha=0.5)

colors = {'Linear Regression': 'blue', 'Random Forest': 'orange'}
for name, preds in results.items():
    linestyle = '--' if name == best_model_name else ':'
    plt.plot(test_dates, preds, label=f'{name}', linestyle=linestyle, color=colors[name])

plt.title(f'So sánh Dự đoán Giá Cổ Phiếu (Best: {best_model_name})')
plt.xlabel('Thời gian')
plt.ylabel('Giá (Close)')
plt.legend()
plt.grid(True)
plt.savefig('./Image/model_comparison_chart.png') # Lưu biểu đồ
plt.show()