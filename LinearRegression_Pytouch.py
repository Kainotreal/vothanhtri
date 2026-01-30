import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# === BƯỚC 1: ĐỊNH NGHĨA MÔ HÌNH ===
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

# === BƯỚC 2: TẠO DỮ LIỆU ===
torch.manual_seed(42)
X_train = torch.randn(100, 1)
y_train = 4 + 3 * X_train + torch.randn(100, 1) * 0.5

X_test = torch.randn(20, 1)
y_test = 4 + 3 * X_test + torch.randn(20, 1) * 0.5

# === BƯỚC 3: KHỞI TẠO MÔ HÌNH VÀ CÔNG CỤ ===
model = LinearRegressionModel(input_dim=1, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# === BƯỚC 4: HUẤN LUYỆN MÔ HÌNH ===
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())
    
    # Backward pass và optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # In thông tin mỗi 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# === BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH ===
model.eval()  # Chuyển sang chế độ evaluation
with torch.no_grad(): # Tắt gradient computation
    y_pred_test = model(X_test)
    test_loss = criterion(y_pred_test, y_test)
    
    # Tính R2 score
    ss_res = torch.sum((y_test - y_pred_test) ** 2)
    ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
    r2_score = 1 - ss_res / ss_tot

print(f"\n--- KẾT QUẢ ---")
print(f"Test Loss (MSE): {test_loss.item():.4f}")
print(f"R2 Score: {r2_score.item():.4f}")
print(f"Learned W: {model.linear.weight.item():.4f}")
print(f"Learned b: {model.linear.bias.item():.4f}")

# === BƯỚC 6: TRỰC QUAN HÓA ===
plt.figure(figsize=(15, 5))

# Subplot 1: Loss curve
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss qua các Epochs')
plt.grid(True, alpha=0.3)

# Subplot 2: Training data và đường hồi quy
plt.subplot(1, 3, 2)
plt.scatter(X_train.numpy(), y_train.numpy(), alpha=0.5, label='Dữ liệu huấn luyện')
X_range = torch.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
with torch.no_grad():
    y_range = model(X_range)
plt.plot(X_range.numpy(), y_range.numpy(), 'r-', linewidth=2, label='Đường hồi quy')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Mô hình hồi quy tuyến tính')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Actual vs Predicted
plt.subplot(1, 3, 3)
with torch.no_grad():
    plt.scatter(y_test.numpy(), y_pred_test.numpy(), alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.title('Actual vs Predicted (Test Set)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()