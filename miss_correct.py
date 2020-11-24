# -*- coding: utf-8 -*-

"""
Created on Sat Nov 7 12:47:39 2020

@author: matthieufuteral-peter
"""

import argparse
import os
from sklearn.metrics import classification_report

from utils.utils import *


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def missing_pred(model, val_images, val_labels):
    '''
    Evaluate the model performances on the dev set

    Parameters:
        model (nn.Module pytorch model): model to evaluate performance
        val_images (dict): keys are categories, values are lists of img path by category
        val_labels (list): labels of images

    Return:
        y_true (list): list of ground truth labels
        y_pred (list): list of predicted labels
        miss (list): list of images with prediction error
        n_miss (int): number of prediction errors
        miss_prob (list): list of probability of the prediction if mistake
    '''
    y_true = []
    y_pred = []
    miss_prob = []
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
                miss_prob.append(torch.max(output).item())

            y_pred.append(pred)
            y_true.append(val_labels[cat])

    return y_true, y_pred, miss, n_miss, miss_prob


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
                       use_cuda=1,
                       dropout=0.0,
                       load_weights=1)
    # Testing mode
    model.eval()

    # Class to index
    val_index = datasets.ImageFolder(args.data + '/val_images', transform=data_transforms_dev).class_to_idx
    # Index to class
    index_to_class = {v: k for k, v in val_index.items()}
    val_images = {cat: [img for img in os.listdir(os.path.join(val_dir, cat)) if 'jpg' in img] for cat in
                  os.listdir(val_dir)}

    y_true, y_pred, miss_pred, n_miss, miss_prob = missing_pred(model, val_images, val_index)

    print(1 - n_miss/176)
    print(miss_prob)

    print(classification_report(y_true, y_pred, target_names=list(val_index.keys())))

    # Display the images where the model made a mistake
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(8, 1)
    for idx in range(n_miss):
        axs = fig.add_subplot(gs[idx])
        axs.set_title('Pred : {} - Ground Truth : {}'.format(index_to_class[y_pred[idx]], index_to_class[y_true[idx]]))
        axs.imshow(miss_pred[idx])
        axs.set_axis_off()

    plt.show()




