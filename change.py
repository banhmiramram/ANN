import os
import unicodedata

# Thư mục chứa ảnh cần đổi tên
folder_path = r"D:\Hiep\GK_AI\thuyet_trinh+Code\code\data\class_..."

def remove_accents(s):
    """Loại bỏ dấu tiếng Việt"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

for filename in os.listdir(folder_path):
    # Bỏ dấu
    no_accents = remove_accents(filename)
    
    # Xóa chữ 'Bansaocua' (không phân biệt hoa thường)
    no_bansaocua = no_accents.replace('Bansaocua', '').replace('bansaocua', '')
    
    # Giữ lại chữ, số, ., _
    new_name = ''.join(c for c in no_bansaocua if c.isalnum() or c in ['.', '_'])
    
    if filename != new_name:
        print(f"Đổi: {filename} -> {new_name}")
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
    else:
        print(f"Giữ nguyên: {filename}")
