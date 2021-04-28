import torch
import torch.nn as nn
from torchvision import models

class resnext(nn.Module):

    def __init__(self, num_classes):
        super(resnext, self).__init__()

        self.num_classes = num_classes

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.backbone = models.resnext101_32x8d(pretrained=True)
        #self.backbone.load_state_dict(torch.load("./models/resnet18.pth"))

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x