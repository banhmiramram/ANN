import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Thư mục chứa ảnh nhị phân (đầu ra của bước 2) ===
input_root = r"D:\Hiep\GK_AI\thuyet_trinh+Code\code\data\output_bin"

# === Danh sách lưu đặc trưng ===
features = []

# === Duyệt qua từng class ===
for class_name in os.listdir(input_root):
    class_path = os.path.join(input_root, class_name)
    if not os.path.isdir(class_path):
        continue

    # Duyệt ảnh trong từng class
    for filename in os.listdir(class_path):
        if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
            continue

        img_path = os.path.join(class_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # === Tính Moments & Hu Moments ===
        moments = cv2.moments(img)
        huMoments = cv2.HuMoments(moments)

        # === Lấy log giữ nguyên dấu ===
        hu_log = -np.sign(huMoments) * np.log10(np.abs(huMoments) + 1e-10)
        hu_log = hu_log.flatten()  # Chuyển thành 1D

        # Thêm vào danh sách
        features.append([class_name, *hu_log])

# === Chuyển sang DataFrame và lưu ===
columns = ["class", "Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7"]
df = pd.DataFrame(features, columns=columns)

# Chuẩn hóa các giá trị Hu về [0, 1] để dễ vẽ
df_norm = df.copy()
for col in columns[1:]:
    df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Lưu ra file CSV
output_csv = os.path.join(os.path.dirname(input_root), "hu_features.csv")
df_norm.to_csv(output_csv, index=False)
print(f"✅ Đã lưu đặc trưng Hu moments vào: {output_csv}")

# === Vẽ biểu đồ phân tán Hu1 - Hu2 ===
plt.figure(figsize=(8, 6))
for class_name, group in df_norm.groupby("class"):
    plt.scatter(group["Hu1"], group["Hu2"], label=class_name, alpha=0.7)

plt.xlabel("Hu1 (normalized)")
plt.ylabel("Hu2 (normalized)")
plt.title("Phân bố đặc trưng Hu’s Moments (Hu1 vs Hu2)")
plt.legend()
plt.grid(True)
plt.show()
