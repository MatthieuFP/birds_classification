import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import timm
import pdb

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
        self.vit = timm.create_model(cfg, pretrained=pretrained, drop_rate=0.0)
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
        torch.nn.init.xavier_uniform_(self._classifier.weight)
        #with torch.no_grad():
        #    self._classifier.weight[:20] = torch.nn.Parameter(torch.eye(20), requires_grad=False)   # Freeze

    def forward(self, x):
        x = self.base(x)
        return self._classifier(x)


class stacked_models(nn.Module):
    def __init__(self, cfg, drop, pretrained=True):
        super(stacked_models, self).__init__()
        self.vit = timm.create_model(cfg, pretrained=pretrained, drop_rate=0.2)
        self.inceptionv3 = models.inception_v3(pretrained=True, aux_logits=False)
        self.layer1 = nn.Linear(2000, 512)
        self.layer2 = nn.Linear(512, 20)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        self.dropout = nn.Dropout(p=drop)
        self.resize = transforms.Resize((299, 299))

    def forward(self, x):
        x1 = self.vit(x)
        x2 = self.inceptionv3(self.resize(x))
        x_cat = torch.cat((x1, x2), dim=-1)
        x_out = self.layer1(self.dropout(x_cat))
        x_out = self.layer2(self.dropout(x_out))
        return x_out

