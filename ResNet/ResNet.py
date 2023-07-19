import torch
from torch import nn 

class ResNet(nn.Module):
    def __init__(self, in_channels=75, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1, stride=1, bias=False),
            nn.BatchNorm2d(64),
        )
        
        self.fc1 = nn.Linear(43264, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, num_classes)
            
    def forward(self, x):
        identity = x
        x = self.conv1(x)  
        x = self.conv2(x)      
        x = self.bn1(x)
        x += self.downsample(identity)
        x = self.relu(x)
        x = self.maxpool(x)         
        
        identity = x
        x = self.conv3(x)  
        x = self.conv4(x)      
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x