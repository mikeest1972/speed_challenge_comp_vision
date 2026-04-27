import torch.nn as nn
import torch



class SpeedPredictorModel(nn.Module):

    def __init__(self):
        super().__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32,kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(256,1)
    

    def forward(self,x):
        # x shape: (batch, window_size, 2, H, W)
        feature = []
        for i in range(x.shape[1]):
            frame = x[:,i, :, :, :]
            frame = frame.permute(0,3,1,2)
            feat = self.pool(self.relu(self.conv1(frame)))
            feat = self.pool(self.relu(self.conv2(feat)))
            feat = self.pool(self.relu(self.conv3(feat)))
            feat = self.gap(feat)
            feat = self.flatten(feat)
            feature.append(feat)
        x = torch.stack(feature,dim=1)
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)
        
        x = self.fc(x)
        return x