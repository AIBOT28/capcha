# Hướng dẫn Training Mô hình Giải CAPTCHA trên Kaggle

Để tận dụng sức mạnh của 2 GPU T4 hoàn toàn miễn phí trên Kaggle, bạn hãy làm theo các bước sau:

## Bước 1: Chuẩn bị Notebook & Data trên Kaggle
1. Đăng nhập Kaggle -> Bấm **Create** -> **New Notebook**.
2. Ở cột bên phải (Settings), bật tùy chọn **Accelerator** sang **GPU T4 x2**. Bật **Internet** thành **On**.
3. Chọn **Add Data** -> Upload file `datasetCAPCHA.zip` của bạn lên Kaggle và đặt tên là `ten-dataset-cua-ban` (Hoặc sửa tên thư mục ở phần `config.py` cho khớp). Kaggle sẽ tự động giải nén thư mục này vào đường dẫn `/kaggle/input/...`.

## Bước 2: Chạy mã nguồn

Tạo một Cell (ô code) mới trong Notebook và copy nguyên đoạn code sau dán vào:

```python
# 1. Tải toàn bộ source code từ Github của bạn về Kaggle
!git clone https://github.com/AIBOT28/capcha.git

# 2. Di chuyển vào thư mục code vừa tải
%cd capcha

# 3. Cài đặt các thư viện cần thiết (Kaggle đã có sẵn pytorch nên sẽ chạy rất nhanh)
!pip install -r requirements.txt

# 4. (TÙY CHỌN) Kiểm tra xem DATA_DIR trong config.py đã đúng đường dẫn dataset trên Kaggle chưa
# Nếu chưa đúng tên, bạn có thể chạy lệnh sau để thay đổi trực tiếp bằng python:
'''
import config
config.Config.DATA_DIR = "/kaggle/input/ten-dataset-cua-ban/dataset"
config.Config.TRAIN_DIR = "/kaggle/input/ten-dataset-cua-ban/dataset/train"
config.Config.VAL_DIR = "/kaggle/input/ten-dataset-cua-ban/dataset/val"
config.Config.TEST_DIR = "/kaggle/input/ten-dataset-cua-ban/dataset/test"
'''

# 5. Kích hoạt huấn luyện với 2 GPU
!python train.py
```

## Bước 3: Lấy kết quả về máy
Sau khi quá trình Training kết thúc (hoặc Early Stopping dừng lại), mô hình sẽ tự động tạo ra 2 file lưu ở `/kaggle/working/capcha`:
- `best_model.pth` (Trọng số mạng neural tốt nhất)
- `training_history.png` (Biểu đồ huấn luyện)

Kaggle cho phép bạn bấm vào bảng `Output` ở góc phải màn hình để tải 2 file này về máy tính cá nhân.
