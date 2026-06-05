import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from config import Config
from model import CRNN

def decode_predictions(preds, idx_to_char):
    # preds: [seq_len, batch_size, num_classes]
    # Lấy class có xác suất cao nhất tại mỗi time step
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    
    # CTC Decoding: Loại bỏ blank (0) và các ký tự trùng lặp liên tiếp
    char_list = []
    for i in range(len(preds)):
        if preds[i] != 0 and (not (i > 0 and preds[i - 1] == preds[i])):
            char_list.append(idx_to_char[preds[i].item()])
            
    return ''.join(char_list)

def predict(image_path, model_path='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    num_chars = len(Config.VOCAB)
    model = CRNN(num_chars, Config.RNN_HIDDEN_SIZE).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"Không tìm thấy {model_path}. Vui lòng train mô hình trước.")
        return

    # Transform
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device) # Thêm batch dimension
    
    idx_to_char = {idx + 1: char for idx, char in enumerate(Config.VOCAB)}
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        prediction = decode_predictions(output, idx_to_char)
        
    print(f"Ảnh: {os.path.basename(image_path)} | Dự đoán: {prediction}")
    return prediction

if __name__ == '__main__':
    # Test thử 1 ảnh trong thư mục test
    test_dir = Config.TEST_DIR
    if os.path.exists(test_dir):
        test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
        if test_images:
            sample_img = os.path.join(test_dir, test_images[0])
            predict(sample_img)
        else:
            print("Không có ảnh nào trong thư mục test.")
    else:
        print(f"Thư mục {test_dir} không tồn tại.")
