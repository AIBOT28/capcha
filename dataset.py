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
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                # Giảm độ xoay và bóp méo (vì chữ đã bị dính chùm rồi)
                transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.9, 1.1), shear=3),
                transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
                # Giảm độ mờ
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
                # Tắt/Giảm mạnh ElasticTransform vì rất dễ làm rách nét chữ
                transforms.RandomApply([transforms.ElasticTransform(alpha=10.0, sigma=3.0)], p=0.1) if hasattr(transforms, 'ElasticTransform') else transforms.RandomApply([], p=0),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                # Giảm tỉ lệ RandomErasing để tránh làm mất nét chữ (vì đã có đường kẻ chéo đè lên)
                transforms.RandomErasing(p=0.1, scale=(0.01, 0.05), ratio=(0.3, 3.3), value=0)
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
