# code from https://github.com/PeiqinZhuang/API-Net/blob/master/models.py
import torch
from torch import nn
import torchvision
from torchvision import models
import numpy as np
from model import ViT
from utils.utils import *
import timm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

class API_Net(nn.Module):
    def __init__(self, drop):
        super(API_Net, self).__init__()

        #densenet161 = models.densenet161(pretrained=True)
        #layers = list(densenet161.children())[:-1]

        self.conv = backbone_ViT(drop=drop)
        self.avg = nn.AvgPool2d(kernel_size=14, stride=1)
        self.map1 = nn.Linear(1024 * 2, 256)
        self.map2 = nn.Linear(256, 1024)
        self.fc = nn.Linear(1024, 20)
        self.drop = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()


    def forward(self, images, targets=None, flag='train'):
        conv_out = self.conv(images)
        conv_out = conv_out.permute(0, 2, 1).view(-1, 1024, 14, 14)
        pool_out = self.avg(conv_out).squeeze()

        if flag == 'train':
            intra_pairs, inter_pairs, \
                    intra_labels, inter_labels = self.get_pairs(pool_out, targets)

            features1 = torch.cat([pool_out[intra_pairs[:, 0]], pool_out[inter_pairs[:, 0]]], dim=0)
            features2 = torch.cat([pool_out[intra_pairs[:, 1]], pool_out[inter_pairs[:, 1]]], dim=0)
            labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)


            mutual_features = torch.cat([features1, features2], dim=1)
            map1_out = self.map1(mutual_features)
            map2_out = self.drop(map1_out)
            map2_out = self.map2(map2_out)


            gate1 = torch.mul(map2_out, features1)
            gate1 = self.sigmoid(gate1)

            gate2 = torch.mul(map2_out, features2)
            gate2 = self.sigmoid(gate2)

            features1_self = torch.mul(gate1, features1) + features1
            features1_other = torch.mul(gate2, features1) + features1

            features2_self = torch.mul(gate2, features2) + features2
            features2_other = torch.mul(gate1, features2) + features2

            logit1_self = self.fc(self.drop(features1_self))
            logit1_other = self.fc(self.drop(features1_other))
            logit2_self = self.fc(self.drop(features2_self))
            logit2_other = self.fc(self.drop(features2_other))

            return logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2

        elif flag == 'val':
            return self.fc(pool_out)


    def get_pairs(self, embeddings, labels):
        distance_matrix = pdist(embeddings).detach().cpu().numpy()

        labels = labels.detach().cpu().numpy().reshape(-1,1)
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = (labels == labels.T)
        lb_eqs[dia_inds] = False
        dist_same = distance_matrix.copy()
        dist_same[lb_eqs == False] = np.inf
        intra_idxs = np.argmin(dist_same, axis=1)

        dist_diff = distance_matrix.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        inter_idxs = np.argmin(dist_diff, axis=1)

        intra_pairs = np.zeros([embeddings.shape[0], 2])
        inter_pairs = np.zeros([embeddings.shape[0], 2])
        intra_labels = np.zeros([embeddings.shape[0], 2])
        inter_labels = np.zeros([embeddings.shape[0], 2])
        for i in range(embeddings.shape[0]):
            intra_labels[i, 0] = labels[i]
            intra_labels[i, 1] = labels[intra_idxs[i]]
            intra_pairs[i, 0] = i
            intra_pairs[i, 1] = intra_idxs[i]

            inter_labels[i, 0] = labels[i]
            inter_labels[i, 1] = labels[inter_idxs[i]]
            inter_pairs[i, 0] = i
            inter_pairs[i, 1] = inter_idxs[i]

        intra_labels = torch.from_numpy(intra_labels).long().to(device)
        intra_pairs = torch.from_numpy(intra_pairs).long().to(device)
        inter_labels = torch.from_numpy(inter_labels).long().to(device)
        inter_pairs = torch.from_numpy(inter_pairs).long().to(device)

        return intra_pairs, inter_pairs, intra_labels, inter_labels



class backbone_ViT(nn.Module):
    def __init__(self, drop=0.2):
        super(backbone_ViT, self).__init__()

        self.ViT = load_model(path_model='/experiment/31d84/model.pt', model_type='vit', dropout=drop,
                              cfg='vit_large_patch16_224', use_cuda=False, load_weights=1)
        self.backbone = self.ViT.vit

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone.patch_embed(x)

        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        for blk in self.backbone.blocks:
            x = blk(x)

        # x = self.backbone.norm(x)
        return x[:, 1:, :]

