import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import CaptchaDataset
from model import CRNN
from predict import decode_predictions

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Exact Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_acc'], 'g-', label='Val Exact Accuracy')
    plt.title('Exact Match Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Plot CER
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_cer'], 'm-', label='Val CER')
    plt.title('Character Error Rate (CER)')
    plt.xlabel('Epochs')
    plt.ylabel('Error Rate (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nĐã lưu biểu đồ huấn luyện vào file 'training_history.png'")
    plt.close()

def train():
    print(f"Sử dụng thiết bị: {Config.DEVICE}")
    if Config.USE_MULTI_GPU:
        print(f"Phát hiện {torch.cuda.device_count()} GPU. Bật Multi-GPU training!")

    # 1. Dataset & DataLoader
    train_dataset = CaptchaDataset(Config.TRAIN_DIR, is_train=True)
    val_dataset = CaptchaDataset(Config.VAL_DIR, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # 2. Model, Loss, Optimizer
    num_chars = len(Config.VOCAB)
    model = CRNN(num_chars, Config.RNN_HIDDEN_SIZE).to(Config.DEVICE)
    
    if Config.USE_MULTI_GPU:
        model = nn.DataParallel(model) # Bật Multi-GPU

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 3. Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_cer': []}

    for epoch in range(Config.EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.EPOCHS} [Train]')
        for images, targets, target_lengths, _ in loop:
            images = images.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images) # [Batch, Seq_len, Num_classes]
            outputs = outputs.permute(1, 0, 2) # [Seq_len, Batch, Num_classes]
            
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
            
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss.backward()
            
            # Gradient clipping để tránh nổ gradient trong RNN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        total_cer = 0.0
        
        idx_to_char = {idx + 1: char for idx, char in enumerate(Config.VOCAB)}
        
        with torch.no_grad():
            for images, targets, target_lengths, labels in val_loader:
                images = images.to(Config.DEVICE)
                targets = targets.to(Config.DEVICE)
                
                outputs = model(images)
                outputs = outputs.permute(1, 0, 2) # [Seq_len, Batch, Num_classes]
                input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
                loss = criterion(outputs, targets, input_lengths, target_lengths)
                val_loss += loss.item()
                
                # Tính toán Accuracy (Khớp chính xác hoàn toàn)
                _, preds = outputs.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                
                # Decode từng mẫu trong batch
                batch_size = images.size(0)
                preds_size = outputs.size(0)
                
                for i in range(batch_size):
                    # Tách prediction cho từng ảnh
                    pred_tensor = preds[i * preds_size : (i + 1) * preds_size]
                    
                    char_list = []
                    for j in range(len(pred_tensor)):
                        if pred_tensor[j] != 0 and (not (j > 0 and pred_tensor[j - 1] == pred_tensor[j])):
                            char_list.append(idx_to_char[pred_tensor[j].item()])
                    
                    pred_str = ''.join(char_list)
                    true_str = labels[i]
                    
                    if pred_str == true_str:
                        correct_preds += 1
                    total_preds += 1
                    
                    # Tính CER (Character Error Rate) dựa trên khoảng cách Levenshtein
                    dist = levenshtein_distance(pred_str, true_str)
                    total_cer += dist / max(len(true_str), 1)
                
        val_loss /= len(val_loader)
        val_acc = (correct_preds / total_preds) * 100 if total_preds > 0 else 0
        val_cer = (total_cer / total_preds) * 100 if total_preds > 0 else 0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_cer'].append(val_cer)
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val CER: {val_cer:.2f}%")
        
        # Learning Rate Scheduler
        scheduler.step(val_loss)

        # Early Stopping & Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            state_dict = model.module.state_dict() if Config.USE_MULTI_GPU else model.state_dict()
            torch.save(state_dict, 'best_model.pth')
            print("Đã lưu best_model.pth mới!")
        else:
            epochs_no_improve += 1
            print(f"Không cải thiện {epochs_no_improve}/{Config.EARLY_STOPPING_PATIENCE} epochs.")
            if epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
                print("Early Stopping kích hoạt! Đã dừng huấn luyện.")
                break

    # Vẽ biểu đồ sau khi kết thúc huấn luyện
    plot_training_history(history)

    # 4. Đánh giá trên tập TEST sau khi huấn luyện xong
    print("\n" + "="*50)
    print("BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP TEST")
    print("="*50)
    
    test_dataset = CaptchaDataset(Config.TEST_DIR, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # Nạp best_model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    test_correct = 0
    test_total = 0
    test_total_cer = 0.0
    
    with torch.no_grad():
        for images, targets, target_lengths, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(Config.DEVICE)
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2) # [Seq_len, Batch, Num_classes]
            
            _, preds = outputs.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            
            batch_size = images.size(0)
            preds_size = outputs.size(0)
            
            for i in range(batch_size):
                pred_tensor = preds[i * preds_size : (i + 1) * preds_size]
                char_list = []
                for j in range(len(pred_tensor)):
                    if pred_tensor[j] != 0 and (not (j > 0 and pred_tensor[j - 1] == pred_tensor[j])):
                        char_list.append(idx_to_char[pred_tensor[j].item()])
                
                pred_str = ''.join(char_list)
                if pred_str == labels[i]:
                    test_correct += 1
                test_total += 1
                
                dist = levenshtein_distance(pred_str, labels[i])
                test_total_cer += dist / max(len(labels[i]), 1)
                
    test_acc = (test_correct / test_total) * 100 if test_total > 0 else 0
    test_cer = (test_total_cer / test_total) * 100 if test_total > 0 else 0
    
    print(f"\n=> KẾT QUẢ TẬP TEST: {test_correct}/{test_total} ảnh đúng. ĐỘ CHÍNH XÁC: {test_acc:.2f}% | CER: {test_cer:.2f}%\n")

if __name__ == '__main__':
    train()
