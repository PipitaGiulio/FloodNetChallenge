import torch
import torch.nn as nn
import torch.nn.functional as F
   
class PSP_SegmentationNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2, 2)
        #encoders blocks
        self.e1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.e2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.e3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        #psp pooling
        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(128, 32, 1, 1),
            nn.BatchNorm2d(32),
        )

        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Conv2d(128, 32, 1, 1),
            nn.BatchNorm2d(32),
        )

        self.pool3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(128, 32, 1, 1),
            nn.BatchNorm2d(32),
        )

        self.pool6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(128, 32, 1, 1),
            nn.BatchNorm2d(32),
        )
        #skip connections
        self.skip1  = nn.Sequential(
            nn.Conv2d(in_channels= 32 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.skip2  = nn.Sequential(
            nn.Conv2d(in_channels= 64 + 64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.skip3  = nn.Sequential(
            nn.Conv2d(in_channels= 128 + 256, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.f_conv = nn.Conv2d(32, 10, kernel_size=1)



    def forward(self, x):
        x1 = self.e1(x)
        p1 = self.maxpool(x1)
        x2 = self.e2(p1)
        p2 = self.maxpool(x2)
        x3 = self.e3(p2)
        p3 = self.maxpool(x3)
        m1 = F.interpolate(self.pool1(p3), p3.shape[2:], mode='bilinear')
        m2 = F.interpolate(self.pool2(p3), p3.shape[2:], mode='bilinear')
        m3 = F.interpolate(self.pool3(p3), p3.shape[2:], mode='bilinear')
        m6 = F.interpolate(self.pool6(p3), p3.shape[2:], mode='bilinear')
        x = torch.cat([p3, m1, m2, m3, m6], dim=1)
        #now back at x3 dimension
        #skip connection
        x = F.interpolate(x, x3.shape[2:], mode='bilinear')
        x = torch.cat([x, x3], dim=1)
        x = self.skip3(x)

        x = F.interpolate(x, x2.shape[2:], mode='bilinear')
        x = torch.cat([x, x2], dim=1)
        x = self.skip2(x)

        x = F.interpolate(x, x1.shape[2:], mode='bilinear')
        x = torch.cat([x, x1], dim=1)
        x = self.skip1(x)

        x = self.f_conv(x)
        

        return x

class Resnet_Transfer_SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        self.e1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.e2 = resnet.layer1
        self.e3 = resnet.layer2
        self.e4 = resnet.layer3
        self.e5 = resnet.layer4
        self.resnet = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        #psp pooling
        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2048, 64, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Conv2d(2048, 64, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.pool3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(2048, 64, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.pool6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(2048, 64, 1, 1),
            nn.BatchNorm2d(64),
        )

        #skip connections
        self.skip1  = nn.Sequential(
            nn.Conv2d(in_channels= 64 + 256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.skip2  = nn.Sequential(
            nn.Conv2d(in_channels= 256 + 256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.skip3  = nn.Sequential(
            nn.Conv2d(in_channels= 512 + 256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.skip4  = nn.Sequential(
            nn.Conv2d(in_channels= 1024 + 256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.skip5  = nn.Sequential(
            nn.Conv2d(in_channels = 2048 + 256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.f_conv = nn.Conv2d(256, 10, kernel_size=1)

    def forward(self, x):
        x1 = self.e1(x)
        p1 = self.maxpool(x1)
        x2 = self.e2(p1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        pool1 = self.pool1(x5)
        pool1 = F.interpolate(pool1, x5.shape[2:], mode='bilinear')
        pool2 = self.pool2(x5)
        pool2 = F.interpolate(pool2, x5.shape[2:], mode='bilinear')
        pool3 = self.pool3(x5)
        pool3 = F.interpolate(pool3, x5.shape[2:], mode='bilinear')
        pool6 = self.pool6(x5)
        pool6 = F.interpolate(pool6, x5.shape[2:], mode='bilinear')

        x = torch.cat([x5, pool1, pool2, pool3, pool6], dim = 1)
        x = self.skip5(x)
        x = F.interpolate(x, x4.shape[2:], mode='bilinear')
        x = torch.cat([x, x4], dim=1)
        x = self.skip4(x)
        x = F.interpolate(x, x3.shape[2:], mode='bilinear')
        x = torch.cat([x, x3], dim=1)
        x = self.skip3(x)
        x = F.interpolate(x, x2.shape[2:], mode='bilinear')
        x = torch.cat([x, x2], dim=1)
        x = self.skip2(x)
        x = F.interpolate(x, x1.shape[2:], mode='bilinear')
        x = torch.cat([x, x1], dim=1)
        x = self.skip1(x)

        x = self.f_conv(x)
        x = F.interpolate(x, size=(713, 713), mode='bilinear', align_corners=False)
        return x









































































































class SegmentationNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2)

        self.e1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        
        
        self.e2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.e3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        #bottleneck with 45x45 sized imgs
        self.bn = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.u1 = nn.ConvTranspose2d(256, 128, kernel_size=2,stride=2)
        self.d1 = nn.Sequential(
            #plus is the result of concatenation 128 feature map from obtained from upsampling with 128 from skipped connections
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.u2 = nn.ConvTranspose2d(128, 64, kernel_size=2,stride=2)
        self.d2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.u3 = nn.ConvTranspose2d(64, 32, kernel_size=2,stride=2)
        self.d3 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.f_conv = nn.Conv2d(32, 10, kernel_size=1)
     
        

    def forward(self, x):

        #encoding
        conv1 = self.e1(x)
        x = self.maxpool(conv1)
        conv2 = self.e2(x)
        x = self.maxpool(conv2)
        conv3 = self.e3(x)
        x = self.maxpool(conv3)
        #bottleneck
        x = self.bn(x)
        #decoding
        x = self.u1(x)
        x = torch.cat([x, conv3], dim = 1)
        x = self.d1(x)

        x = self.u2(x)
        x = torch.cat([x, conv2], dim = 1)
        x = self.d2(x)

        x = self.u3(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.d3(x)

        #compute segmentation masks
        x = self.f_conv(x)
        return x
 
