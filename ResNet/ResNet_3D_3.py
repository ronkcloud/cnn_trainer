import torch
from torch import nn 

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.Conv3d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm3d(out_channels))
        self.maxpool = nn.MaxPool3d(kernel_size = 2, stride = [3, 2, 2] , padding=1)
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        return out

class ResNet_3D(nn.Module):
    def __init__(self, block, num_classes = 3):
        super(ResNet_3D, self).__init__()
        self.inplanes = 1 
        self.layer0 = self._make_layer(block, 64)
        self.layer1 = self._make_layer(block, 128)
        self.layer2 = self._make_layer(block, 256)
        self.layer3 = self._make_layer(block, 512)
        self.fc = nn.Sequential(
            nn.Linear(9216, 512), 
                # 4608: kernel_size = [3, 2, 2], stride = [3, 2, 2] 
                # 9216: kernel = 2 stride = [3, 2, 2]
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes))
        
    def _make_layer(self, block, planes, stride=1):

        downsample = nn.Sequential(
            nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride),
            nn.BatchNorm3d(planes))
            
        layers = block(self.inplanes, planes, downsample)
        self.inplanes = planes

        return nn.Sequential(layers)
    

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x