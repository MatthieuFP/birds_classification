from logger import logger
from model import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision import datasets


def logging(message, stdout):
    assert type(message) == str, "Message should be a string"
    logger.info(message)
    stdout.append(message)
    return stdout


def load_model(path_model, model_type, dropout, cfg, use_cuda, load_weights=1):
    # Load model
    device = torch.device('cuda' if use_cuda else 'cpu')
    if model_type == 'resnet101':
        model = ResNet_Net(drop=dropout)
    elif model_type == 'vit':
        model = ViT(cfg=cfg, drop=dropout, pretrained=bool(1 - load_weights))
    elif model_type == 'ssl-vit':
        vit_model = ViT(cfg=cfg, drop=dropout, pretrained=0)
        state_dict = torch.load(path_model, map_location=device)
        vit_model.load_state_dict(state_dict)
        model = SSL_ViT(model=vit_model)
        model.eval()
    elif model_type == 'default':
        model = Net()
    else:
        raise NameError('model not found')
    logger.info("{} model loaded".format(model_type))

    if load_weights and use_cuda and model_type != 'ssl-vit':
        state_dict = torch.load(path_model, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')

    return model


def data_transformation(horizontal_flip=1, vertical_flip=1, random_rotation=1, erasing=0, model='vit', train=1, size=224):
    data_transforms = []
    if horizontal_flip and train:
        data_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    if vertical_flip and train:
        data_transforms.append(transforms.RandomVerticalFlip(p=0.5))
    if random_rotation and train:
        data_transforms.append(transforms.RandomRotation(degrees=(-45, 45)))

    if model == 'vit':
        data_transforms.append(transforms.Resize((size, size)))
    elif model == 'resnet':
        data_transforms += [transforms.Resize(size), transforms.CenterCrop(224)]
    elif model =='api_net':
        data_transforms += [transforms.Resize((224, 224))]

    data_transforms += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])]
    if erasing and train:
        data_transforms.append(transforms.RandomErasing(p=0.5, scale=(0.05, 0.15)))

    return transforms.Compose(data_transforms)


def alpha(t, T2, factor):
    '''
    Weight of the unlabeled loss

    :param t: epoch t / time t
    :param T2: Last epoch before constant weight
    :param factor: constant
    :return: alpha weight of the unlabeled loss
    '''

    if t < T2:
        return (t * factor)/T2
    else:
        return factor


def load_nabirds(batch_size, data_transforms):
    loader = torch.utils.data.DataLoader(datasets.ImageFolder('confident_filtered_cropped_NAbirds',
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