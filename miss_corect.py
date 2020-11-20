import argparse
from tqdm import tqdm
import os
import numpy as np
import PIL.Image as Image
import pdb
import torch.nn.functional as F

from model import *
from utils.utils import *

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='cropped_birds', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="model architecture, e.g. resnet"),
parser.add_argument('--cfg', type=str, default='vit_large_patch16_224'),
parser.add_argument('--experiment', type=str, default='semi_supervized_experiment')
parser.add_argument('--RUN_ID', type=str, help='id of the model', required=True),
parser.add_argument('--size', type=int, default=224, help='size of the images'),
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")


args = parser.parse_args()
use_cuda = torch.cuda.is_available()
print('Using GPU : {}'.format(use_cuda))

path = os.getcwd()
path_experiments = os.path.join(path, args.experiment)
path_id = os.path.join(path_experiments, args.RUN_ID)
path_model = os.path.join(path_id, 'model.pt')

# Load model
model = load_model(path_model=path_model,
                   model_type=args.model,
                   cfg=args.cfg,
                   use_cuda=use_cuda,
                   dropout=0.0,
                   load_weights=1)

