import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import Config

class CaptchaDataset(Dataset):
    def __init__(self, data_dir, is_train=False):
        self.data_dir = data_dir
        self.is_train = is_train
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Build vocabulary mapping
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(Config.VOCAB)} # 0 is reserved for CTC blank
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(Config.VOCAB)}
        
        # Data augmentation for training
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(degrees=2), # Rất nhỏ để không làm mất chữ
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Nhãn là tên file (bỏ phần mở rộng)
        label_str = os.path.splitext(os.path.basename(img_path))[0]
        
        image = self.transform(image)
        
        # Encode label
        target = [self.char_to_idx.get(char, 1) for char in label_str] # Fallback to 1 if not in vocab, usually shouldn't happen
        target_length = len(target)
        
        return image, torch.tensor(target, dtype=torch.long), torch.tensor(target_length, dtype=torch.long), label_str
