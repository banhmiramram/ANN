import os
import cv2
import numpy as np

# Đường dẫn gốc chứa folder OUTPUT
input_root = r"D:\Hiep\GK_AI\thuyet_trinh+Code\code\data\OUTPUT"
output_root = r"D:\Hiep\GK_AI\thuyet_trinh+Code\code\data\out_put_bina"

# Duyệt toàn bộ thư mục con trong OUTPUT
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            input_path = os.path.join(root, file)
            
            # Tạo đường dẫn output tương ứng
            relative_path = os.path.relpath(root, input_root)
            output_dir = os.path.join(output_root, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file)

            # Đọc ảnh gốc
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Chuyển sang ảnh nhị phân bằng ngưỡng Otsu
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Lưu ảnh nhị phân
            cv2.imwrite(output_path, binary)

print("✅ Đã chuyển tất cả ảnh sang ảnh nhị phân và lưu vào thư mục out_put_bina.")
