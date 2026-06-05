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
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
                transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.85, 1.15), shear=8),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
                # transforms.ElasticTransform có thể uốn éo chữ cực kỳ hiệu quả cho CAPTCHA (yêu cầu torchvision >= 0.12)
                transforms.RandomApply([transforms.ElasticTransform(alpha=20.0, sigma=5.0)], p=0.5) if hasattr(transforms, 'ElasticTransform') else transforms.RandomApply([], p=0),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                # RandomErasing (xóa các mảng ngẫu nhiên) mô phỏng đường nhiễu gạch chéo che khuất chữ
                transforms.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
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
