import torch
import torch.nn as nn
import torch.nn.functional as F


class VQA_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.embdedding = nn.Embedding(50, 20)
        self.LSTM = nn.LSTM(20, hidden_size=512, num_layers=2, 
                    dropout=0.2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 51)
        )

    def forward(self, x_img, x_q):
        CNN_map = self.CNN(x_img)
        CNN_map = torch.flatten(CNN_map, start_dim = 1)
        x_q = self.embdedding(x_q)
        _, (h_n, _) = self.LSTM(x_q)
        LSTM_map = torch.cat((h_n[-2], h_n[-1]), dim=1)
        fc_inp = torch.cat([CNN_map, LSTM_map], dim=1)
        y_hat = self.fc(fc_inp)
        return y_hat


class transfer_VQA(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        self.resnet = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )
               
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.embdedding = nn.Embedding(50, 20)
        self.LSTM = nn.LSTM(20, hidden_size=512, num_layers=2, 
                    dropout=0.2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 51)
        )

    def forward(self, x_img, x_q):
        CNN_map = self.resnet(x_img)
        CNN_map = torch.flatten(CNN_map, start_dim = 1)
        x_q = self.embdedding(x_q)
        _, (h_n, _) = self.LSTM(x_q)
        LSTM_map = torch.cat((h_n[-2], h_n[-1]), dim=1)
        fc_inp = torch.cat([CNN_map, LSTM_map], dim=1)
        y_hat = self.fc(fc_inp)
        return y_hat
    
