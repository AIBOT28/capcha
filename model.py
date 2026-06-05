import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=256):
        super(CRNN, self).__init__()
        
        # CNN Backbone (VGG-style sâu và chuẩn cho OCR)
        # Input: [Batch, 3, 40, 120]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 20x60
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 10x30
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), padding=(0, 1)), # 5x31
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), padding=(0, 1)), # 2x32
            nn.Conv2d(512, 512, kernel_size=2, padding=0), nn.BatchNorm2d(512), nn.ReLU(True) # 1x31
        )
        
        # RNN
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )
        
        # Linear + CTC Blank token
        self.fc = nn.Linear(hidden_size, num_chars + 1)

    def forward(self, x):
        # x: [Batch, 3, 40, 120]
        conv = self.cnn(x)
        # conv shape: [Batch, 512, 1, Width]
        
        b, c, h, w = conv.size()
        assert h == 1, "Chiều cao feature map phải bằng 1"
        
        # Chuyển đổi để phù hợp với RNN: [Seq_len, Batch, Features]
        conv = conv.squeeze(2) # [Batch, 512, Width]
        conv = conv.permute(2, 0, 1) # [Width, Batch, 512]
        
        rnn_out = self.rnn(conv) # [Width, Batch, hidden_size]
        output = self.fc(rnn_out) # [Width, Batch, num_chars + 1]
        
        # Log_softmax cho CTC Loss
        return torch.nn.functional.log_softmax(output, dim=2)


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output
