import argparse
import json
import os
import glob
import time
from PIL import Image
from utils.utils import *
import numpy as np
import torch.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm_notebook
from model import *
import argparse
from early_stopping.pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import pdb
from uuid import uuid4

path = os.getcwd()
path_id = os.path.join("experiment", '31d84')
path_load_model = os.path.join(path_id, 'model.pt')
path_data = os.path.join(path, 'cropped_NAbirds')
path_save = os.path.join(path, 'confident_filtered_cropped_NAbirds')

data_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
model = load_model(path_load_model, 'vit', 'vit_large_patch16_224', use_cuda=True, load_weights=1)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


for n in range(1, 14):
    model.eval()
    os.makedirs(path_save + '/subsample{}'.format(n), exist_ok=False)
    path_imgs = glob.glob(path_data + '/subsample{}/*.jpg'.format(n))

    saved = 0
    for file in tqdm_notebook(path_imgs):

        img = pil_loader(file)
        data = data_transforms(img)
        data = data.unsqueeze(0)
        if torch.cuda.is_available():
            data = data.cuda()
        output = model(data)
        out = F.softmax(output, dim=-1)

        if torch.max(out, dim=-1)[0].item() > 0.99:
            saved += 1
            print('\n' + str(saved))

            basename = os.path.basename(file)
            img.save(path_save + '/subsample{}/'.format(n) + basename, "JPEG")