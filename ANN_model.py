# ================================================================
# BƯỚC 4: HUẤN LUYỆN MÔ HÌNH ANN (Artificial Neural Network)
# ================================================================
# - Input: file CSV đã có cột "set" (train / val)
# - Output: Biểu đồ loss & accuracy theo epoch cho các mô hình
# ================================================================

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 🔧 TỰ CHỌN BACKEND HIỂN THỊ ĐỒ THỊ
try:
    matplotlib.use("TkAgg")  # mở cửa sổ riêng nếu có GUI
except Exception:
    matplotlib.use("Agg")    # fallback nếu môi trường không hỗ trợ

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Ẩn log TensorFlow cho gọn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================
# 1️⃣ ĐỌC DỮ LIỆU
# ==============================
file_path = r"D:\Hiep\GK_AI\thuyet_trinh+Code\code\data\hu_features_encoded_train.csv"  # file đã có cột set
df = pd.read_csv(file_path)

# Cột đầu vào (Hu Moments)
X_cols = ['Hu1','Hu2','Hu3','Hu4','Hu5','Hu6','Hu7']
# Cột đầu ra (one-hot)
y_cols = ['class_class_0','class_class_1','class_class_2','class_class_3','class_class_4']

# Chia tập train / val
X_train = df[df['set'] == 'train'][X_cols].values
y_train = df[df['set'] == 'train'][y_cols].values
X_val = df[df['set'] == 'val'][X_cols].values
y_val = df[df['set'] == 'val'][y_cols].values

print("✅ Dữ liệu đã sẵn sàng:")
print(f"   - Số mẫu train: {X_train.shape[0]}")
print(f"   - Số mẫu val:   {X_val.shape[0]}")

# ==============================
# 2️⃣ XÂY DỰNG & HUẤN LUYỆN MÔ HÌNH
# ==============================
def train_ann(hidden_neurons):
    """Huấn luyện mô hình ANN với số nơ-ron ẩn tùy chọn"""
    model = Sequential([
        Dense(hidden_neurons, input_dim=7, activation='sigmoid'),
        Dense(5, activation='softmax')
    ])
    
    optimizer = SGD(learning_rate=0.1) # Tốc độ học
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200, #số lần lặp (epochs)
        batch_size=len(X_train),  # Batch Gradient Descent
        verbose=0
    )
    return model, history

# ==============================
# 3️⃣ HUẤN LUYỆN VỚI 4 CẤU HÌNH KHÁC NHAU
# ==============================
neurons_list = [7,8,9,10] # Số nơ-ron ẩn khác nhau để thử
histories = {}

for n in neurons_list:
    print(f"\n🔹 Huấn luyện mô hình với {n} nơ-ron ẩn ...")
    _, hist = train_ann(n)
    histories[n] = hist

# ==============================
# 4️⃣ VẼ ĐỒ THỊ HÀM LỖI (LOSS)
# ==============================
plt.figure(figsize=(10,6))
for n, hist in histories.items():
    plt.plot(hist.history['loss'], label=f"Train loss (hidden={n})")
    plt.plot(hist.history['val_loss'], '--', label=f"Val loss (hidden={n})")

plt.title("Biểu đồ hàm lỗi Train/Validation theo Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss (Categorical Cross-Entropy)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=True)  # 🔥 đảm bảo đồ thị không bị tắt

# ==============================
# 5️⃣ VẼ ĐỒ THỊ ĐỘ CHÍNH XÁC (ACCURACY)
# ==============================
plt.figure(figsize=(10,6))
for n, hist in histories.items():
    plt.plot(hist.history['accuracy'], label=f"Train acc (hidden={n})")
    plt.plot(hist.history['val_accuracy'], '--', label=f"Val acc (hidden={n})")

plt.title("Độ chính xác Train/Validation theo Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=True)  # 🔥 giữ cửa sổ hiển thị đến khi bạn đóng

# ==============================
# 6️⃣ GỢI Ý NHẬN XÉT (cho báo cáo)
# ==============================
print("\n📊 GỢI Ý NHẬN XÉT:")
print("- Nếu train_loss và val_loss cùng giảm dần → mô hình học tốt, không bị quá khớp.")
print("- Nếu train_loss giảm nhưng val_loss tăng → mô hình có thể bị overfitting.")
print("- Nếu loss dao động mạnh → thử giảm learning_rate (ví dụ 0.05 hoặc 0.01).")
print("- So sánh 4 mô hình, chọn số nơ-ron ẩn cho val_loss nhỏ và val_accuracy cao nhất.")
