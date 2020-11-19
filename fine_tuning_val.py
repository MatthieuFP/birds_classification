import argparse
import json
import os
import time
from utils.utils import *
import numpy as np
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm
from model import *
from early_stopping.pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import pdb
from uuid import uuid4


def validation_train(model, epochs, val_loader, stdout, optimizer, path_save):
    model.train()
    val_loss = []

    for epoch in range(epochs):

        validation_loss = 0
        correct = 0
        for data, target in tqdm(val_loader):

            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            del data, target  # leave space on GPU

        validation_loss /= len(val_loader.dataset)
        score = 100. * correct / len(val_loader.dataset)
        stdout = logging('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                         validation_loss, correct, len(val_loader.dataset), score), stdout)

        val_loss.append(validation_loss)

        if len(val_loss) == 1:
            torch.save(model.state_dict(), path_save)
        elif val_loss[-1] < val_loss[-2]:
            torch.save(model.state_dict(), path_save)

    return val_loss, score.data.item(), stdout


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RecVis A3 training script')
    parser.add_argument('--data', type=str, default='cropped_birds', metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
    parser.add_argument('--batch_size', type=int, default=64, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay ADAM optimizer')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--experiment', type=str, default='semi_supervized_experiment', metavar='E',
                        help='folder where experiment outputs are located.')
    parser.add_argument('--model', type=str, default='vit', help='model to run')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability')
    parser.add_argument('--cfg', type=str, default='vit_large_patch16_224', help='Config ViT model')
    parser.add_argument('--horizontal_flip', type=int, default=1, help="perform hor. flip data augmentation")
    parser.add_argument('--vertical_flip', type=int, default=1, help="perform vert. flip data augmentation")
    parser.add_argument('--RUN_ID', type=str, required=True, help="RUN_ID of the pre trained model")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    stdout = []  # Save stdout file
    RUN_ID = str(uuid4())[0:5]
    print('RUN_ID : {}'.format(RUN_ID))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = os.getcwd()
    path_experiments = os.path.join(path, 'experiment')
    path_id = os.path.join("experiment", args.RUN_ID)
    path_load_model = os.path.join(path_id, 'model.pt')
    path_save = os.path.join(path_experiments, RUN_ID, 'model.pt')
    os.makedirs(os.path.dirname(path_save), exist_ok=False)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Load model
    model = load_model(path_model=path_load_model,
                       model_type=args.model,
                       dropout=args.dropout,
                       cfg=args.cfg,
                       use_cuda=True,
                       load_weights=1)
    # Transform data
    data_transforms_train = data_transformation(horizontal_flip=args.horizontal_flip,
                                                vertical_flip=args.vertical_flip,
                                                model=args.model,
                                                train=1)
    val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/val_images',
                                                                  transform=data_transforms_train),
                                             batch_size=args.batch_size, shuffle=True, num_workers=1)

    # Create folder results
    path_result = os.path.join(args.experiment, RUN_ID)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    val_loss, score, stoudt = validation_train(model, args.epochs, val_loader, stdout, optimizer, path_save)
