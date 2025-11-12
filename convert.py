import cv2
import os

# === Thư mục gốc chứa data ===
input_root = r"D:\Hiep\GK_AI\thuyet_trinh+Code\code\data"
output_root = os.path.join(os.path.dirname(input_root), "output_bin")

# Tạo thư mục output_bin nếu chưa có
os.makedirs(output_root, exist_ok=True)

# === Duyệt qua toàn bộ thư mục con ===
for class_name in os.listdir(input_root):
    class_path = os.path.join(input_root, class_name)
    if not os.path.isdir(class_path):
        continue

    # Tạo thư mục tương ứng trong output_bin
    output_class_path = os.path.join(output_root, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    # Duyệt ảnh trong từng class
    for filename in os.listdir(class_path):
        if not (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")):
            continue

        # Đường dẫn ảnh gốc
        input_path = os.path.join(class_path, filename)

        # Đọc ảnh màu
        img = cv2.imread(input_path)

        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Nhị phân hóa ảnh: vùng sáng → trắng (255), vùng tối → đen (0)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Lưu vào thư mục output tương ứng
        output_path = os.path.join(output_class_path, filename)
        cv2.imwrite(output_path, binary)

        print(f"Đã xử lý: {input_path} -> {output_path}")

print("✅ Hoàn thành! Ảnh nhị phân được lưu trong thư mục:", output_root)
