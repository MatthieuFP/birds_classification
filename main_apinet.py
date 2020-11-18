import argparse
import os
import json
import time
from tqdm import tqdm, tqdm_notebook
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from early_stopping.pytorchtools import EarlyStopping
import numpy as np
from API_net.models import API_Net
from utils.utils import *
from uuid import uuid4
import pdb


def train(train_loader, model, criterion, optimizer_conv, scheduler_conv, optimizer_fc, scheduler_fc, epoch, device,
          log_interval, train_loss, stdout):

    rank_criterion = nn.MarginRankingLoss(margin=0.05)
    softmax_layer = nn.Softmax(dim=1).to(device)
    train_batch_loss = []

    for batch_idx, (input, target) in tqdm(enumerate(train_loader, 1)):
        model.train()

        input_var = input.to(device)
        target_var = target.to(device).squeeze()

        pdb.set_trace()
        # compute output
        logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2 = model(input_var, target_var,
                                                                                       flag='train')

        batch_size = logit1_self.shape[0]
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)

        self_logits = torch.zeros(2 * batch_size, 20).to(device)
        other_logits = torch.zeros(2 * batch_size, 20).to(device)
        self_logits[:batch_size] = logit1_self
        self_logits[batch_size:] = logit2_self
        other_logits[:batch_size] = logit1_other
        other_logits[batch_size:] = logit2_other

        # compute loss
        logits = torch.cat([self_logits, other_logits], dim=0)
        targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)
        softmax_loss = criterion(logits, targets)

        self_scores = softmax_layer(self_logits)[torch.arange(2 * batch_size).to(device).long(),
                                                 torch.cat([labels1, labels2], dim=0)]
        other_scores = softmax_layer(other_logits)[torch.arange(2 * batch_size).to(device).long(),
                                                   torch.cat([labels1, labels2], dim=0)]
        flag = torch.ones([2 * batch_size, ]).to(device)
        rank_loss = rank_criterion(self_scores, other_scores, flag)

        loss = softmax_loss + rank_loss
        train_batch_loss.append(loss.data.item() / batch_size)

        # compute gradient and do SGD step
        optimizer_conv.zero_grad()
        optimizer_fc.zero_grad()
        loss.backward()

        if epoch >= 8:
            optimizer_conv.step()
        optimizer_fc.step()
        scheduler_conv.step()
        scheduler_fc.step()

        if not batch_idx % log_interval:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(input), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data.item()))

        del input_var, target_var  # leave space on gpu

    train_loss.append(np.mean(train_batch_loss))
    stdout = logging('Train loss Epoch {} : {}'.format(epoch, train_loss[-1]), stdout)

    return model, train_loss, stdout


def validation(model, val_loader, val_loss, stdout, device):
    model.eval()
    validation_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss().to(device)
    for data, target in tqdm(val_loader):
        data = data.to(device)
        target = target.to(device)

        output = model(data, targets=None, flag='val')
        # sum up batch loss

        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        del data, target  # leave space on GPU

    validation_loss /= len(val_loader.dataset)
    score = 100. * correct / len(val_loader.dataset)
    stdout = logging('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        score), stdout)

    val_loss.append(validation_loss)

    return val_loss, score.data.item(), stdout


def main(train_loader, val_loader, model, batch_size, criterion, optimizer_conv, scheduler_conv, optimizer_fc,
         scheduler_fc, epochs, device, log_interval, stdout, early_stopping):

    train_loss, val_loss, val_accuracy, epoch_time = [], [], [], []
    n_batches = len(train_loader.dataset) // batch_size

    for epoch in range(1, epochs + 1):
        stdout = logging("Epoch {} - Start TRAINING".format(epoch), stdout)
        t0 = time.time()
        model, train_loss, stdout = train(train_loader, model, criterion, optimizer_conv, scheduler_conv, optimizer_fc,
                                          scheduler_fc, epoch, device, log_interval, train_loss, stdout)
        val_loss, accuracy, stdout = validation(model, val_loader, val_loss, stdout, device)

        val_accuracy.append(accuracy)

        early_stopping(val_loss[-1], model, stdout)
        if early_stopping.early_stop:
            message = "Early stopping"
            print(message)
            stdout.append(message)
            break

        time_elapsed = time.time() - t0
        stdout = logging("Epoch {} - Time elapsed : {}".format(epoch, time_elapsed), stdout)
        epoch_time.append(time_elapsed)
        stdout.append(' ')

    stdout = logging("Average time per epoch : {}".format(np.mean(epoch_time)), stdout)
    stdout.append(' ')

    return model, train_loss, val_loss, val_accuracy, epoch_time, stdout


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='RecVis A3 training script')
    parser.add_argument('--data', type=str, default='cropped_birds', metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
    parser.add_argument('--batch_size', type=int, default=64, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay ADAM optimizer')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                        help='folder where experiment outputs are located.')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--debug', type=int, default=0, help="debug")
    parser.add_argument('--horizontal_flip', type=int, default=1, help="perform hor. flip data augmentation")
    parser.add_argument('--vertical_flip', type=int, default=1, help="perform vert. flip data augmentation")
    parser.add_argument('--random_rotation', type=int, default=1, help="perform random rotation from -45° to 45°")
    parser.add_argument('--erasing', type=int, default=1, help="perform random erasing or not")
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.manual_seed(args.seed)

    if not args.debug:
        pdb.set_trace = lambda: None

    stdout = []  # Save stdout file
    RUN_ID = str(uuid4())[0:5]
    print('RUN_ID : {}'.format(RUN_ID))
    device = torch.device("cuda" if use_cuda else "cpu")


    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Load model
    model = API_Net(drop=args.dropout)
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer_conv = torch.optim.SGD(model.conv.parameters(), args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay)

    fc_parameters = [value for name, value in model.named_parameters() if 'conv' not in name]
    optimizer_fc = torch.optim.SGD(fc_parameters, args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)

    # Transform data
    data_transforms_train = data_transformation(horizontal_flip=args.horizontal_flip,
                                                random_rotation=args.random_rotation,
                                                erasing=args.erasing,
                                                model='api_net',
                                                size=224,
                                                train=1)
    data_transforms_dev = data_transformation(model='api_net', size=224, train=0)

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/train_images',
                             transform=data_transforms_train),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                             transform=data_transforms_dev),
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    scheduler_conv = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_conv, 100 * len(train_loader))
    scheduler_fc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fc, 100 * len(train_loader))

    if use_cuda:
        print(torch.cuda.get_device_name(0))
        stdout.append(torch.cuda.get_device_name(0))
    else:
        print('Using CPU')

    # Create folder results
    path_result = os.path.join(args.experiment, RUN_ID)
    if not os.path.isdir(path_result):
        os.makedirs(path_result)
        results = {'RUN_ID': RUN_ID,
                   'model': 'API_net / Densenet161 backbone',
                   'batch_size': args.batch_size,
                   'epochs': args.epochs,
                   'lr': args.lr,
                   'optimizer': 'sgd',
                   "patience": args.patience,
                   "weight_decay (adam/sgd)": args.weight_decay,
                   "momentum (sgd)": args.momentum,
                   "scheduler": "CosineAnnealing",
                   "horizontal_flip": args.horizontal_flip,
                   "vertical_flip": args.vertical_flip,
                   "random_rotation": args.random_rotation,
                   "erasing": args.erasing}

        stdout += ['{} : {}'.format(str(k), str(v)) for k, v in results.items()]
        stdout.append(' ')

    # Set up early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(path_result, 'model.pt'))

    # Summary Writer - Tensorboard
    path_report = os.path.join(path_result, 'report')
    model, train_loss, val_loss, val_accuracy, epoch_time, stdout = main(train_loader, val_loader, model, args.batch_size,
                                                                         criterion, optimizer_conv, scheduler_conv,
                                                                         optimizer_fc, scheduler_fc, args.epochs, device,
                                                                         args.log_interval, stdout, early_stopping)

    pdb.set_trace()
    results['train_loss'] = train_loss
    results['val_loss'] = val_loss
    results['val_accuracy'] = val_accuracy
    results['epoch_time'] = epoch_time

    with open(os.path.join(path_result, 'results.json'), 'w') as fj:
        json.dump(results, fj)

    stdout = logging("End of training - save stdout file", stdout)
    with open(os.path.join(path_result, 'stdout.txt'), 'w') as f:
        f.write('\n'.join(stdout))
