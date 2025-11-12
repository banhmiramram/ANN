import os
import cv2
import random
import numpy as np
from pathlib import Path
import shutil

# === CONFIG ===
DATA_DIR = r"D:\HIEP\GK_AI\THUYET_TRINH+CODE\CODE\data"
OUTPUT_DIR = os.path.join(DATA_DIR, "OUTPUT")
ANGLES = [7, -7, 15, -15]
NUM_CLASSES = 5
CLASS_SIZE = 50  # mỗi class có 50 ảnh gốc

os.makedirs(OUTPUT_DIR, exist_ok=True)

def rotate_image(image, angle):
    """Xoay ảnh quanh tâm (centroid)."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def augment_class(class_path, output_class_path, start_id):
    """Thực hiện augmentation và gán ID cho ảnh."""
    os.makedirs(output_class_path, exist_ok=True)
    img_files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    current_id = start_id
    id_list = []

    for img_name in img_files:
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Không đọc được ảnh: {img_path}")
            continue

        # Lưu ảnh gốc
        save_name = f"{current_id:03d}.jpg"
        cv2.imwrite(os.path.join(output_class_path, save_name), img)
        id_list.append(current_id)
        current_id += 1

        # Xoay ảnh và lưu
        for ang in ANGLES:
            rotated = rotate_image(img, ang)
            save_name = f"{current_id:03d}_rot{ang}.jpg"
            cv2.imwrite(os.path.join(output_class_path, save_name), rotated)
            current_id += 1

    return id_list

def split_data(class_path):
    """Chia dữ liệu class thành 3 folder: 35 - 8 - 7 ảnh."""
    images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
    random.shuffle(images)

    folder_sizes = [35, 8, 7]
    folder_names = ["Folder1", "Folder2", "Folder3"]

    for size, fname in zip(folder_sizes, folder_names):
        folder_path = os.path.join(class_path, fname)
        os.makedirs(folder_path, exist_ok=True)

        selected = images[:size]
        images = images[size:]

        for img in selected:
            src = os.path.join(class_path, img)
            dst = os.path.join(folder_path, img)
            shutil.move(src, dst)

        # Ghi danh sách ID
        ids = [img.split("_")[0].split(".")[0] for img in selected]
        with open(os.path.join(class_path, f"{fname}_ID.txt"), "w") as f:
            f.write("\n".join(ids))

def main():
    start_id = 1
    for class_idx in range(NUM_CLASSES):
        class_input = os.path.join(DATA_DIR, f"class_{class_idx}")
        class_output = os.path.join(OUTPUT_DIR, f"class_{class_idx}")
        print(f"🔄 Xử lý {class_input} ...")

        ids = augment_class(class_input, class_output, start_id)
        split_data(class_output)
        start_id += CLASS_SIZE

    print("✅ Hoàn thành chia dữ liệu và augmentation!")

if __name__ == "__main__":
    main()
