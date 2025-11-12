import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ---- 1. Đọc file gốc ----
file_path = r"D:\Hiep\GK_AI\thuyet_trinh+Code\code\data\hu_features_train.csv"
df = pd.read_csv(file_path)

# ---- 2. Mã hóa nhãn ----
df['class'] = df['class'].astype('category')
y = df['class'].cat.codes  # Chuyển class_0 → 0, class_1 → 1, ...
y_encoded = to_categorical(y)

# ---- 3. Gộp lại thành DataFrame mới ----
y_cols = [f'class_class_{i}' for i in range(y_encoded.shape[1])]
y_df = pd.DataFrame(y_encoded, columns=y_cols)

df_new = pd.concat([df.drop(columns=['class']), y_df], axis=1)

# ---- 4. Chia dữ liệu train/validation ----
train_idx, val_idx = train_test_split(df_new.index, test_size=0.2, random_state=42, shuffle=True)
df_new['set'] = 'train'
df_new.loc[val_idx, 'set'] = 'val'

# ---- 5. Lưu ra file mới ----
new_file = r"D:\Hiep\GK_AI\thuyet_trinh+Code\code\data\hu_features_encoded_train.csv"
df_new.to_csv(new_file, index=False)

print("✅ Đã mã hóa và chia dữ liệu. File mới lưu tại:")
print(new_file)
print(df_new.head())
