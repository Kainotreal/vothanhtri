import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === BƯỚC 1: TẠO DỮ LIỆU MẪU ===
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 điểm, 1 đặc trưng
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + nhiễu

# === BƯỚC 2: CHIA TÁCH DỮ LIỆU ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === BƯỚC 3: KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH ===
model = LinearRegression()
model.fit(X_train, y_train)

# Hiển thị các hệ số đã học
print(f"Hệ số chặn (Beta 0): {model.intercept_[0]:.4f}")
print(f"Hệ số góc (Beta 1): {model.coef_[0][0]:.4f}")

# === BƯỚC 4: DỰ ĐOÁN ===
y_pred = model.predict(X_test)

# === BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH ===
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n--- KẾT QUẢ ĐÁNH GIÁ ---")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# === BƯỚC 6: TRỰC QUAN HÓA ===
plt.figure(figsize=(12, 5))

# Subplot 1: Dữ liệu và đường hồi quy
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='lightblue', label='Tập huấn luyện', alpha=0.6)
plt.scatter(X_test, y_test, color='blue', label='Tập kiểm tra', alpha=0.6)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Đường hồi quy')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Hồi quy tuyến tính')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Actual vs Predicted
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Dự đoán hoàn hảo')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === BƯỚC 7: DỰ ĐOÁN CHO DỮ LIỆU MỚI ===
new_X = np.array([[0.5], [1.0], [1.5]])
predictions = model.predict(new_X)
print(f"\n--- DỰ ĐOÁN CHO DỮ LIỆU MỚI ---")
for x_val, pred in zip(new_X, predictions):
    print(f"Với x = {x_val[0]:.1f} => y_dự đoán = {pred[0]:.4f}")