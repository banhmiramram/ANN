import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# === 1. Đọc dữ liệu gốc từ file CSV ===
file_path = r"D:\Hiep\GK_AI\thuyet_trinh+Code\code\data\hu_features.csv"
df = pd.read_csv(file_path)

print("✅ Đã đọc dữ liệu thành công:", df.shape)
print("Các cột:", list(df.columns))

# === 2. Mã hoá nhãn one-hot ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(df['class'])
y_onehot = to_categorical(y_encoded)

# Đặt tên cột one-hot tương ứng
y_columns = [f"class_{label}" for label in encoder.classes_]
y_df = pd.DataFrame(y_onehot, columns=y_columns)

# Ghép X (7 đặc trưng) + y_onehot
X = df.drop(columns=['class'])
data_encoded = pd.concat([X, y_df], axis=1)

# === 3. Chia train/validation ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y_df, test_size=0.2, random_state=42, stratify=y_encoded
)

train_data = pd.concat([X_train, y_train], axis=1)
train_data["set"] = "train"

val_data = pd.concat([X_val, y_val], axis=1)
val_data["set"] = "validation"

final_df = pd.concat([train_data, val_data])

# === 4. Ghi ra file Excel mới để không ảnh hưởng file gốc ===
output_path = r"D:\Hiep\GK_AI\thuyet_trinh+Code\code\data\hu_features_encoded.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="hu_features_original", index=False)
    final_df.to_excel(writer, sheet_name="encoded_split", index=False)

print(f"✅ Đã tạo file Excel mới: {output_path}")
print("   • Sheet 1: hu_features_original (bản gốc)")
print("   • Sheet 2: encoded_split (đã mã hoá + chia train/val)")
