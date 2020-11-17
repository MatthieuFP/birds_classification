import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.classifier = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.classifier(x)


class ResNet_Net(nn.Module):
    def __init__(self, drop):
        super(ResNet_Net, self).__init__()
        self.resnet = torchvision.models.resnet101(pretrained=True)
        self.dropout = nn.Dropout(drop)
        self.leaky_relu = nn.LeakyReLU()
        self.linear = nn.Linear(1000, 128)
        self.classifier = nn.Linear(128, 20)

        self.init_weights(self.linear)
        self.init_weights(self.classifier)

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.linear(x)
        output = self.classifier(self.dropout(x))
        return output


class ViT(nn.Module):
    def __init__(self, cfg, drop, pretrained=True):
        super(ViT, self).__init__()
        self.vit = timm.create_model(cfg, pretrained=pretrained, drop_rate=drop)
        self.dropout = nn.Dropout(drop)
        # self.leaky_relu = nn.LeakyReLU()
        # self.linear = nn.Linear(1000, 128)
        # self.classifier = nn.Linear(128, 20)
        self.classifier = nn.Linear(1000, 20)

        # self.init_weights(self.linear)
        self.init_weights(self.classifier)

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.vit(x)
        # x = self.leaky_relu(x)
        # x = self.linear(x)
        output = self.classifier(self.dropout(x))
        return output


class SSL_ViT(nn.Module):
    def __init__(self, model):
        super(SSL_ViT, self).__init__()
        self.base = model
        self._classifier = nn.Linear(20, 21, bias=False)
        self.init_noiseclass()

    def init_noiseclass(self):
        self._classifier.weight[:20] = torch.eye(20)

