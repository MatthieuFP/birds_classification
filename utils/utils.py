# -*- coding: utf-8 -*-

"""
Created on Sat Nov 7 12:47:39 2020

@author: matthieufuteral-peter
"""

from logger import logger
from model import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import datasets


def logging(message, stdout):
    '''
    Print with logger the message and add it the output file stdout

    Parameters:
    message (str): message to print
    stdout (list): output file with messages printed

    Return:
    stdout (list): list of printed messages
    '''
    assert type(message) == str, "Message should be a string"
    logger.info(message)
    stdout.append(message)
    return stdout


def load_model(path_model, model_type, dropout, cfg, use_cuda, load_weights=1):
    '''
    Load the pytorch model required by the user.

    Parameters:
        path_model (str): path to the model's weights if load_weights is 1
        model_type (str): REQUIRED - model type to load. Either 'resnet101', 'vit', 'default' or 'stacked'.
        dropout (float): between 0. and 1., dropout parameters of the models
        use_cuda (int): 0 or 1 => Using cuda or not
        load_weights (int): 0 or 1 => load the weights from path_model or not

    Return:
        pytorch model.to(device), trained or not
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'resnet101':
        model = ResNet_Net(drop=dropout)

    elif model_type == 'vit':
        model = ViT(cfg=cfg, drop=dropout, pretrained=bool(1 - load_weights))

    elif model_type == 'default':
        model = Net()

    elif model_type == 'stacked':
        model = stacked_models(cfg=cfg, drop=dropout, pretrained=bool(1 - load_weights))

    else:
        raise NameError('model type not found')
    logger.info("{} model loaded".format(model_type))

    if load_weights and use_cuda:
        state_dict = torch.load(path_model, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("Weights loaded")

    if use_cuda:
        print('Using GPU')
        model.to(device)
    else:
        print('Using CPU')

    return model


class Resize(object):
    '''
    Perform resizing while keeping aspect ratio
    '''
    def __init__(self, size, interpolation=Image.BILINEAR):
        '''
        size (int): new size
        interpolation: interpolation method to resize the image
        '''
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        '''
        Parameter:
            img (PIL.Image): Image to resize

        Return:
            resized img (PIL.Image)
        '''
        old_size = img.size  # old_size is in (width, height) format
        ratio = float(self.size)/max(old_size)
        new_size = [int(x * ratio) for x in old_size]
        new_size[int(np.argmax(new_size))] = self.size  # Assert that max size is 224 and not 223 due to approx error
        return img.resize(tuple(new_size), resample=self.interpolation)


def padding_size(image):
    '''
    Extract the required padding dimension to get squared image

    Parameter:
        image (PIL.Image): image resized by Resize class

    Return:
        padding dimension (tuple)
    '''
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if not h_padding % 1 else h_padding+0.5
    t_pad = v_padding if not v_padding % 1 else v_padding+0.5
    r_pad = h_padding if not h_padding % 1 else h_padding-0.5
    b_pad = v_padding if not v_padding % 1 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class NewPad(object):
    '''
    Perform padding on image to get squarred image
    '''
    def __init__(self, fill=0, padding_mode='constant'):
        '''
        Parameters:
            fill (int or tuple): Pixel fill value for constant fill. Default is 0 (black color)
            padding_mode (str): padding mode to perform padding
        '''
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        '''
        Return the Image padded

        Parameters:
            img (PIL.Image): image to pad

        Return:
            pad img (PIL.Image)
        '''
        return transforms.Pad(padding_size(img), fill=self.fill, padding_mode=self.padding_mode)(img)


def data_transformation(horizontal_flip=1, vertical_flip=1, random_rotation=0, erasing=0, model='vit', train=1,
                        size=224):
    '''
    Create the desired data transformation pipeline

    Parameters:
        horizontal_flip (int): 0 or 1. Perform horizontal flip with prob=0.5 or not
        vertical_flip (int): 0 or 1. Perform vertical flip with prob=0.5 or not
        random_rotation (int): 0 or 1. Perform random rotation from -45 to 45 degrees or not.
        erasing (int): 0 or 1. Perform CutOut of 1% of the input image with probability 0.5 or not
        model (str): REQUIRED model type
        train (int): 0 or 1. Training mode or not (horizontal & vertical flip, random rotation etc...
                             are only performed in training mode)
        size (int): Size of the resized Image.

    Return:
        data transform pipeline (transform.Compose class)
    '''

    data_transforms = []

    if horizontal_flip and train:
        data_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    if vertical_flip and train:
        data_transforms.append(transforms.RandomVerticalFlip(p=0.5))
    if random_rotation and train:
        data_transforms.append(transforms.RandomRotation(degrees=(-45, 45)))

    if model == 'vit' or 'stacked':
        data_transforms.append(transforms.Resize((224, 224)))
    elif model == 'resnet':
        data_transforms += [transforms.Resize(size), transforms.CenterCrop(224)]
    elif model == 'api_net':
        data_transforms += [Resize(size=size), NewPad(fill=0, padding_mode='constant')]
    else:
        raise NameError('Model type not found')

    data_transforms += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])]
    if erasing and train:
        data_transforms.append(transforms.RandomErasing(p=0.5, scale=(0.01, 0.01), ratio=(1., 1.)))

    return transforms.Compose(data_transforms)


def alpha(t, T2, factor):
    '''
    Weight of the unlabeled loss in pseudo labelling algorithm

    :param t: epoch t (or time t)
    :param T2: Last epoch before constant weight
    :param factor: constant

    :return: alpha weight of the unlabeled loss
    '''
    if t < T2:
        return (t * factor)/T2
    else:
        return factor


def load_nabirds(batch_size, data_transforms, data='cropped_NAbirds'):
    '''
    Load nabirds dataset

    Paramaters:
        batch_size (int): size of the batch
        data_transforms (transforms.Compose): data transformation pipeline
        data (str): dataset to load, default : 'cropped-NAbirds'

    Return:
        dataloader (torch.utils.data.DataLoader class)
    '''
    loader = torch.utils.data.DataLoader(datasets.ImageFolder(data,
                                                              transform=data_transforms),
                                                              batch_size=batch_size,
                                                              shuffle=True,
                                                              num_workers=1)
    return loader


def imshow(inp, title=None):
    """Imshow for Tensor. From pytorch.org"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def add_weight_decay(model, value):
    '''
    Add weight decay to model's parameters except biases

    Parameters:
        model (nn.Module pytorch): pytorch model
        value (float): value of the weight decay

    Return:
         list of params with no decay, params with decay
    '''
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': value}]


def smooth_loss(target, output, n_classes, smoothing):
    '''
    compute smooth loss according to smoothing parameter

    Parameters:
        target (torch.Tensor): format (batch_size, ) ground truth labels
        output (torch.Tensor): format (batch_size, n_classes) => model predictions
        n_classes (int): number of classes
        smoothing (float): between 0 and 1. How much to smooth the labels.

    Return:
        Computed smoothed loss
    '''
    assert 0 <= smoothing < 1
    output = output.log_softmax(dim=-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(output, device=target.device)  # Create one hot vector
        true_dist.fill_(smoothing / (n_classes - 1))  # Fill with constant value
        true_dist.scatter_(1, target.unsqueeze(1), 1 - smoothing)  # Replace label index with smooth value
    return torch.mean(torch.sum(-true_dist * output, dim=-1))  # Return computed loss
