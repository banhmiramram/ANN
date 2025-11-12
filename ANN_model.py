import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# =====================================================
# BƯỚC 4: HUẤN LUYỆN MÔ HÌNH ANN
# =====================================================

class ANNClassifier:
    """
    Mạng Neural Network với:
    - 1 lớp ẩn (hidden layer)
    - Hàm kích hoạt: sigmoid (hidden), softmax (output)
    - Hàm lỗi: Categorical Cross-Entropy (CCE)
    - Training mode: Batch Gradient Descent (BGD)
    """
    
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.1):
        """
        Khởi tạo mô hình ANN
        
        Parameters:
        -----------
        n_input : int - Số nơ-ron lớp input (7 đặc trưng Hu's moments)
        n_hidden : int - Số nơ-ron lớp ẩn (7, 8, 9, hoặc 10)
        n_output : int - Số nơ-ron lớp output (số class)
        learning_rate : float - Tốc độ học η = 0.1
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        
        # Khởi tạo trọng số ngẫu nhiên
        np.random.seed(42)  # Để kết quả có thể tái lập
        
        # Trọng số giữa input và hidden layer
        # Shape: (n_input, n_hidden)
        self.W1 = np.random.randn(n_input, n_hidden) * 0.5
        self.b1 = np.zeros((1, n_hidden))
        
        # Trọng số giữa hidden và output layer
        # Shape: (n_hidden, n_output)
        self.W2 = np.random.randn(n_hidden, n_output) * 0.5
        self.b2 = np.zeros((1, n_output))
        
        # Lưu lại lịch sử loss
        self.train_loss_history = []
        self.val_loss_history = []
        
    def sigmoid(self, z):
        """Hàm kích hoạt sigmoid cho lớp ẩn"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, a):
        """Đạo hàm của sigmoid"""
        return a * (1 - a)
    
    def softmax(self, z):
        """Hàm kích hoạt softmax cho lớp output"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def categorical_crossentropy(self, y_true, y_pred):
        """
        Hàm lỗi Categorical Cross-Entropy (CCE)
        
        Parameters:
        -----------
        y_true : array - One-hot encoded labels
        y_pred : array - Predicted probabilities
        
        Returns:
        --------
        loss : float - Giá trị loss trung bình
        """
        # Tránh log(0)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        # Tính CCE
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    def forward(self, X):
        """
        Lan truyền tiến (Forward Propagation)
        
        Parameters:
        -----------
        X : array - Input features, shape (n_samples, n_input)
        
        Returns:
        --------
        output : array - Output predictions, shape (n_samples, n_output)
        """
        # Lớp ẩn: z1 = X @ W1 + b1
        self.z1 = np.dot(X, self.W1) + self.b1
        # Kích hoạt sigmoid
        self.a1 = self.sigmoid(self.z1)
        
        # Lớp output: z2 = a1 @ W2 + b2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Kích hoạt softmax
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def backward(self, X, y_true, y_pred):
        """
        Lan truyền ngược (Backward Propagation)
        
        Parameters:
        -----------
        X : array - Input features
        y_true : array - One-hot encoded true labels
        y_pred : array - Predicted probabilities
        """
        m = X.shape[0]  # Số lượng mẫu
        
        # Tính gradient lớp output
        # dL/dz2 = y_pred - y_true (với softmax + CCE)
        dz2 = y_pred - y_true
        
        # Gradient cho W2 và b2
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Tính gradient lớp ẩn
        # dL/da1 = dz2 @ W2.T
        da1 = np.dot(dz2, self.W2.T)
        # dL/dz1 = da1 * sigmoid'(a1)
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        
        # Gradient cho W1 và b1
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Cập nhật trọng số (Batch Gradient Descent)
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X_train, y_train, X_val, y_val, epochs=200):
        """
        Huấn luyện mô hình
        
        Parameters:
        -----------
        X_train : array - Training features
        y_train : array - Training labels (one-hot encoded)
        X_val : array - Validation features
        y_val : array - Validation labels (one-hot encoded)
        epochs : int - Số epoch (200)
        """
        print(f"Bắt đầu huấn luyện mô hình với {epochs} epochs...")
        print(f"Số mẫu training: {X_train.shape[0]}")
        print(f"Số mẫu validation: {X_val.shape[0]}")
        print(f"Cấu trúc mạng: {self.n_input} -> {self.n_hidden} -> {self.n_output}")
        print(f"Tốc độ học: {self.learning_rate}")
        print("-" * 60)
        
        for epoch in range(epochs):
            # Forward propagation trên tập training
            y_train_pred = self.forward(X_train)
            
            # Tính loss training
            train_loss = self.categorical_crossentropy(y_train, y_train_pred)
            self.train_loss_history.append(train_loss)
            
            # Backward propagation và cập nhật trọng số
            self.backward(X_train, y_train, y_train_pred)
            
            # Validation (không cập nhật trọng số)
            y_val_pred = self.forward(X_val)
            val_loss = self.categorical_crossentropy(y_val, y_val_pred)
            self.val_loss_history.append(val_loss)
            
            # In kết quả mỗi 20 epochs
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f}")
        
        print("-" * 60)
        print("Hoàn thành huấn luyện!")
    
    def predict(self, X):
        """
        Dự đoán class cho dữ liệu mới
        
        Parameters:
        -----------
        X : array - Input features
        
        Returns:
        --------
        predictions : array - Predicted class labels
        """
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def plot_loss_curves(self, save_path=None):
        """
        Vẽ đồ thị hàm lỗi training và validation theo epoch
        
        Parameters:
        -----------
        save_path : str - Đường dẫn lưu ảnh (nếu có)
        """
        plt.figure(figsize=(12, 5))
        
        # Đồ thị Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.train_loss_history) + 1), 
                self.train_loss_history, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (CCE)', fontsize=12)
        plt.title('Training Loss theo Epoch', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Đồ thị Validation Loss
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(self.val_loss_history) + 1), 
                self.val_loss_history, 'r-', linewidth=2, label='Validation Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (CCE)', fontsize=12)
        plt.title('Validation Loss theo Epoch', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu đồ thị tại: {save_path}")
        
        plt.show()
        
        # Vẽ cả hai loss trên cùng một đồ thị để so sánh
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_loss_history) + 1), 
                self.train_loss_history, 'b-', linewidth=2, label='Training Loss')
        plt.plot(range(1, len(self.val_loss_history) + 1), 
                self.val_loss_history, 'r-', linewidth=2, label='Validation Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (CCE)', fontsize=12)
        plt.title('So sánh Training Loss và Validation Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        if save_path:
            comparison_path = save_path.replace('.png', '_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu đồ thị so sánh tại: {comparison_path}")
        
        plt.show()


# =====================================================
# HÀM HỖ TRỢ: CHUẨN BỊ DỮ LIỆU
# =====================================================

def prepare_data_for_training(features_folder1, features_folder2, features_folder3, 
                              labels_folder1, labels_folder2, labels_folder3):
    """
    Chuẩn bị dữ liệu từ 3 folder và mã hóa one-hot
    
    Parameters:
    -----------
    features_folder1/2/3 : array - Đặc trưng từ 3 folder
    labels_folder1/2/3 : array - Nhãn từ 3 folder
    
    Returns:
    --------
    X_train, y_train, X_val, y_val, X_test, y_test, n_classes
    """
    # Folder 1: Training (35 mẫu mỗi class)
    X_train = features_folder1
    y_train_raw = labels_folder1
    
    # Folder 2: Validation (8 mẫu mỗi class)
    X_val = features_folder2
    y_val_raw = labels_folder2
    
    # Folder 3: Test (7 mẫu mỗi class)
    X_test = features_folder3
    y_test_raw = labels_folder3
    
    # Mã hóa one-hot cho labels
    n_classes = len(np.unique(y_train_raw))
    
    # Chuyển labels sang one-hot encoding
    def one_hot_encode(labels, n_classes):
        one_hot = np.zeros((len(labels), n_classes))
        for i, label in enumerate(labels):
            one_hot[i, label] = 1
        return one_hot
    
    y_train = one_hot_encode(y_train_raw, n_classes)
    y_val = one_hot_encode(y_val_raw, n_classes)
    y_test = one_hot_encode(y_test_raw, n_classes)
    
    print("=" * 60)
    print("CHUẨN BỊ DỮ LIỆU HOÀN TẤT")
    print("=" * 60)
    print(f"Training set:   {X_train.shape[0]} mẫu x {X_train.shape[1]} đặc trưng")
    print(f"Validation set: {X_val.shape[0]} mẫu x {X_val.shape[1]} đặc trưng")
    print(f"Test set:       {X_test.shape[0]} mẫu x {X_test.shape[1]} đặc trưng")
    print(f"Số lượng class: {n_classes}")
    print("=" * 60)
    print()
    
    return X_train, y_train, X_val, y_val, X_test, y_test, n_classes


# =====================================================
# CHƯƠNG TRÌNH CHÍNH: THỰC HIỆN BƯỚC 4
# =====================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BƯỚC 4: HUẤN LUYỆN MÔ HÌNH ANN")
    print("=" * 60)
    print()
    
    # -----------------------------------------------
    # CHUẨN BỊ DỮ LIỆU
    # -----------------------------------------------
    # Giả sử bạn đã có dữ liệu từ Bước 3 được lưu trong file hoặc biến
    # Ví dụ: Load từ file numpy
    
    # THAY THẾ PHẦN NÀY BẰNG CÁCH LOAD DỮ LIỆU THỰC TẾ CỦA BẠN:
    # features_folder1 = np.load('features_folder1.npy')  # 35 mẫu/class
    # labels_folder1 = np.load('labels_folder1.npy')
    # features_folder2 = np.load('features_folder2.npy')  # 8 mẫu/class
    # labels_folder2 = np.load('labels_folder2.npy')
    # features_folder3 = np.load('features_folder3.npy')  # 7 mẫu/class
    # labels_folder3 = np.load('labels_folder3.npy')
    
    # ĐÂY LÀ DỮ LIỆU MẪU (CHỈ ĐỂ DEMO):
    # Thay thế bằng dữ liệu thực của bạn
    np.random.seed(42)
    n_samples_train = 35 * 5  # 35 mẫu x 5 class
    n_samples_val = 8 * 5     # 8 mẫu x 5 class
    n_samples_test = 7 * 5    # 7 mẫu x 5 class
    n_features = 7            # 7 đặc trưng Hu's moments
    n_classes = 5             # 5 class
    
    features_folder1 = np.random.rand(n_samples_train, n_features)
    labels_folder1 = np.repeat(range(n_classes), 35)
    
    features_folder2 = np.random.rand(n_samples_val, n_features)
    labels_folder2 = np.repeat(range(n_classes), 8)
    
    features_folder3 = np.random.rand(n_samples_test, n_features)
    labels_folder3 = np.repeat(range(n_classes), 7)
    
    # Chuẩn bị dữ liệu
    X_train, y_train, X_val, y_val, X_test, y_test, n_classes = prepare_data_for_training(
        features_folder1, features_folder2, features_folder3,
        labels_folder1, labels_folder2, labels_folder3
    )
    
    # -----------------------------------------------
    # HUẤN LUYỆN MÔ HÌNH VỚI CÁC CẤU HÌNH KHÁC NHAU
    # -----------------------------------------------
    # Theo yêu cầu: thử với số nơ-ron ẩn = 7, 8, 9, 10
    
    hidden_neurons_list = [7, 8, 9, 10]
    models = {}
    
    for n_hidden in hidden_neurons_list:
        print(f"\n{'='*60}")
        print(f"HUẤN LUYỆN MÔ HÌNH VỚI {n_hidden} NƠ-RON ẨN")
        print(f"{'='*60}\n")
        
        # Khởi tạo mô hình
        model = ANNClassifier(
            n_input=7,           # 7 đặc trưng Hu's moments
            n_hidden=n_hidden,   # Số nơ-ron ẩn
            n_output=n_classes,  # Số class
            learning_rate=0.1    # η = 0.1
        )
        
        # Huấn luyện mô hình
        model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=200
        )
        
        # Lưu mô hình
        models[n_hidden] = model
        
        # Vẽ đồ thị loss
        model.plot_loss_curves(save_path=f'loss_curves_{n_hidden}_neurons.png')
        
        print()
    
    # -----------------------------------------------
    # NHẬN XÉT VỀ CÁC ĐỒ THỊ HÀM LỖI
    # -----------------------------------------------
    print("\n" + "=" * 60)
    print("NHẬN XÉT VỀ HÀM LỖI:")
    print("=" * 60)
    print("""
    1. Training Loss:
       - Giảm dần theo epoch, cho thấy mô hình đang học tốt từ dữ liệu training
       - Độ dốc giảm dần, cho thấy mô hình hội tụ
    
    2. Validation Loss:
       - Nếu validation loss giảm cùng với training loss: 
         → Mô hình tổng quát hóa tốt, không bị overfitting
       - Nếu validation loss tăng trong khi training loss giảm:
         → Có dấu hiệu overfitting, cần điều chỉnh (early stopping, regularization)
    
    3. So sánh các mô hình với số nơ-ron ẩn khác nhau:
       - Mô hình với loss thấp hơn và ổn định hơn là lựa chọn tốt hơn
       - Cần xem xét cả training loss và validation loss để chọn mô hình
    
    4. Lưu ý:
       - Nếu cả hai loss đều cao: mô hình underfitting (cần tăng độ phức tạp)
       - Nếu training loss thấp nhưng validation loss cao: overfitting
    """)
    
    # So sánh loss cuối cùng của các mô hình
    print("\nSO SÁNH LOSS CUỐI CÙNG CỦA CÁC MÔ HÌNH:")
    print("-" * 60)
    for n_hidden, model in models.items():
        final_train_loss = model.train_loss_history[-1]
        final_val_loss = model.val_loss_history[-1]
        print(f"{n_hidden} nơ-ron ẩn: Train Loss = {final_train_loss:.6f}, "
              f"Val Loss = {final_val_loss:.6f}")
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH BƯỚC 4!")
    print("=" * 60)