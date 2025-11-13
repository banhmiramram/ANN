# ================================================================
# BƯỚC 5: KIỂM TRA & ĐÁNH GIÁ MÔ HÌNH ANN
# ================================================================
# - Input:
#     + Mô hình đã lưu: best_model.h5
#     + File test CSV: hu_features_encoded_test.csv
# - Output:
#     + Accuracy trên tập test
#     + File CSV ghi kết quả dự đoán
# ================================================================

import os
import pandas as pd
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score

# Ẩn log TensorFlow cho gọn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================
# 1️⃣ ĐỌC DỮ LIỆU TEST
# ==============================
test_path = r"D:\Hiep\GK_AI\thuyet_trinh+Code\code\data\hu_features_encoded_test.csv"  # ⚠️ đường dẫn file test
df_test = pd.read_csv(test_path)

# Xác định cột đặc trưng và nhãn
X_cols = ['Hu1','Hu2','Hu3','Hu4','Hu5','Hu6','Hu7']
y_cols = ['class_class_0','class_class_1','class_class_2','class_class_3','class_class_4']

X_test = df_test[X_cols].values
y_test = df_test[y_cols].values

print(f"✅ Đã đọc {X_test.shape[0]} mẫu test.")

# ==============================
# 2️⃣ NẠP LẠI MÔ HÌNH
# ==============================
model_path = "best_model.h5"
model = load_model(model_path)
print(f"✅ Đã nạp mô hình từ: {model_path}")

# ==============================
# 3️⃣ DỰ ĐOÁN TRÊN TẬP TEST
# ==============================
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)  # chuyển xác suất thành nhãn
y_true = np.argmax(y_test, axis=1)

# ==============================
# 4️⃣ TÍNH ĐỘ CHÍNH XÁC (ACCURACY)
# ==============================
acc = accuracy_score(y_true, y_pred)
print(f"\n🎯 Độ chính xác (Accuracy) trên tập test: {acc:.4f}")

# ==============================
# 5️⃣ GHI KẾT QUẢ DỰ ĐOÁN RA FILE CSV
# ==============================
df_test['true_label'] = y_true
df_test['pred_label'] = y_pred

output_path = r"D:\Hiep\GK_AI\thuyet_trinh+Code\code\results\test_predictions.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_test.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"📂 Đã lưu kết quả dự đoán tại: {output_path}")

# ==============================
# 6️⃣ NHẬN XÉT (cho báo cáo)
# ==============================
print("\n📊 GỢI Ý NHẬN XÉT:")
print("- Nếu accuracy > 0.8 → mô hình học tốt.")
print("- Nếu thấp (<0.5), có thể do dữ liệu test khác biệt hoặc thiếu dữ liệu train.")
print("- Bạn có thể thử tăng epoch, thêm nơ-ron ẩn, hoặc giảm learning rate để cải thiện.")
