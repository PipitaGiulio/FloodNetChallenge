import torch
import torch.nn as nn
import torch.nn.functional as F
class ClassificationNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxPool = nn.MaxPool2d(2, stride=2)
        #224x224
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(32)
        #112x112
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(64)
        #56x56
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(128)
        #28x28
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batch4 = nn.BatchNorm2d(256)
        #14x14
        #self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch1(x)
        x = self.maxPool(x)
        x = F.relu(self.conv2(x))
        x = self.batch2(x)
        x = self.maxPool(x)
        x = F.relu(self.conv3(x))
        x = self.batch3(x)
        x = self.maxPool(x)
        x = F.relu(self.conv4(x))
        x = self.batch4(x)
        x = self.maxPool(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)
        return x