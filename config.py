import os
import torch

class Config:
    # Thư mục dữ liệu
    # Tự động nhận diện nếu đang chạy trên Kaggle
    if os.path.exists("/kaggle"):
        # Lưu ý: Thay "ten-dataset-cua-ban" bằng đúng thư mục bạn đã add vào Kaggle
        DATA_DIR = "/kaggle/input/datasets/lngkhoanguyns/datasetcapcha/dataset" 
    else:
        DATA_DIR = r"C:\Users\nguye\Documents\CaNhan\DeepLearning\giaiCapCha\dataset"
        
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    # Kích thước ảnh đầu vào (Chiều cao, Chiều rộng)
    IMAGE_WIDTH = 120
    IMAGE_HEIGHT = 40
    
    # Cấu hình huấn luyện
    BATCH_SIZE = 128  # Tăng lên 128 vì có 2 GPU (mỗi GPU sẽ xử lý 64 ảnh)
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 7 # Dừng sớm nếu sau 7 epochs loss không giảm
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
    # Cấu hình kiến trúc CRNN
    RNN_HIDDEN_SIZE = 256
    
    # Cấu hình thiết bị
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_MULTI_GPU = torch.cuda.device_count() > 1
    
    # Tiền xử lý số luồng
    NUM_WORKERS = 4 if os.name != 'nt' else 0  # 4 là mức tối ưu an toàn nhất cho đa số Kaggle kernels

    # Bảng ký tự mặc định
    VOCAB = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
