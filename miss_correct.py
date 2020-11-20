import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import PIL.Image as Image
import pdb
import torch.nn.functional as F

from model import *
from utils.utils import *


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def missing_pred(model, val_images, val_labels):
    y_true = []
    y_pred = []
    miss = []
    n_miss = 0
    for cat, imgs in val_images.items():
        for img in imgs:
            path_img = os.path.join(val_dir, cat, img)
            data = data_transforms_dev(pil_loader(path_img)).unsqueeze(0).to(device)

            output = F.softmax(model(data), dim=-1)
            pred = torch.argmax(output, dim=-1).item()
            if pred != val_labels[cat]:
                n_miss += 1
                miss.append(pil_loader(path_img))

            y_pred.append(pred)
            y_true.append(val_labels[cat])

    return y_true, y_pred, miss, n_miss


if __name__ == '__main__':

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using GPU : {}'.format(use_cuda))

    path = os.getcwd()
    path_experiments = os.path.join(path, args.experiment)
    path_id = os.path.join(path_experiments, args.RUN_ID)
    path_model = os.path.join(path_id, 'model.pt')
    val_dir = os.path.join(args.data, 'val_images')
    data_transforms_dev = data_transformation(model=args.model, size=args.size, train=0)

    # Load model
    model = load_model(path_model=path_model,
                       model_type=args.model,
                       cfg=args.cfg,
                       use_cuda=use_cuda,
                       dropout=0.0,
                       load_weights=1)
    model.eval()

    val_images = {cat: [img for img in os.listdir(os.path.join(val_dir, cat)) if 'jpg' in img] for cat in os.listdir(val_dir)}
    val_labels = {cat: i for i, cat in enumerate(val_images.keys())}

    y_true, y_pred, miss_pred, n_miss = missing_pred(models, val_images, val_labels)

    classification_report(y_true, y_pred, target_names=list(val_labels.keys()))

    fig = plt.figure(figsize=(8, 8))
    columns = n_miss // 2
    rows = n_miss - columns
    for idx in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, idx)
        plt.imshow(miss_pred[idx])

    plt.axis('off')
    plt.show()




