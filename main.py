# -*- coding: utf-8 -*-

"""
Created on Sat Nov 7 12:47:39 2020

@author: matthieufuteral-peter
"""

import argparse
import json
import os
import random
import time
from utils.utils import *
import numpy as np
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm, tqdm_notebook
from model import *
from early_stopping.pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import pdb
from uuid import uuid4


def train(epoch, model, train_loader, use_cuda, log_interval, train_loss, stdout, writer, n_batches, batch_size,
          accumulation_steps, device, blurring=0):
    '''
    Training step

    Parameters:
        model (nn.Module pytorch model): pytorch model
        epoch (int): n-th epoch
        train_loader (DataLoader): loader of the training set
        use_cuda (int): computing with GPU or not
        log_interval (int): batch interval between saving weights value and printing loss
        train_loss (list): list of the train loss per epoch
        stdout (list): output file with printing messages - save in list format
        writer (SummaryWriter object): write the weights, gradients and loss values on Tensorboard to be visualized later
        n_batches (int): number of batches to perform 1 epoch
        batch_size (int): number of examples by batch
        device (torch.device object): 'cpu' device or 'cuda' device
        accumulation_steps (int): perform backpropagation every accumulation_steps batches
        blurring (int): 0 or 1. Gaussian blur the input or not

    Return:
        model (nn.Module pytorch model): pytorch model trained for 1 epoch
        train_loss (list): list of loss values
        stdout (list): list of string with messages printed in order to save output file.
    '''

    optimizer.zero_grad()
    model.train()
    train_batch_loss = []

    pdb.set_trace()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        n_iter = batch_idx + (epoch - 1) * n_batches

        if blurring and random.random() < 0.25:  # Blur the Image with probability 1/4
            g_blur = transforms.GaussianBlur(11, sigma=(2, 10))
            data = g_blur(data)

        if use_cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        loss = criterion(output, target)
        loss.backward()

        if not (batch_idx + 1) % accumulation_steps:  # Gradient accumulation to handle limit of GPU RAM
            optimizer.step()
            optimizer.zero_grad()

        # optimizer.step()
        if not (batch_idx + 1) % log_interval:
            writer.add_histogram('layer1', model.layer1.weight, n_iter)  # Check layer1 weights
            writer.add_histogram('layer1', model.layer1.weight.grad, n_iter)  # Check layer1 gradient
            writer.add_histogram('layer2', model.layer2.weight, n_iter)  # Check layer1 weights
            writer.add_histogram('layer2', model.layer2.weight.grad, n_iter)  # Check layer1 gradient

            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(train_batch_loss)))

        # pdb.set_trace()
        train_batch_loss.append(loss.data.item() / batch_size)
        writer.add_scalar('train_loss', loss.data.item(), n_iter)

        del data, target  # leave space on gpu

    # pdb.set_trace()
    train_loss.append(np.mean(train_batch_loss))
    stdout = logging('Train loss Epoch {} : {}'.format(epoch, train_loss[-1]), stdout)

    return model, train_loss, stdout


def validation(model, epoch, val_loader, use_cuda, val_loss, stdout, writer, blurring=0):
    '''
    Perform one epoch on the validation set. Testing mode.

    Parameters:
        model (nn.Module pytorch model): pytorch model
        epoch (int): n-th epoch
        val_loader (DataLoader): loader of the validation set
        use_cuda (int): computing with GPU or not
        val_loss (list): list of the validation loss per epoch
        stdout (list): output file with printing messages - save in list format
        writer (SummaryWriter object): write the weights, gradients and loss values on Tensorboard to be visualized later
        blurring (int): 0 or 1. Gaussian blur the input or not

    Return:
        val_loss (list): list of the validation loss per epoch (current one added)
        score accuracy (float): mean accuracy on the validation set
        stdout (list): output file with printing messages - save in list format
        writer (SummaryWriter object): write the weights, gradients and loss values on Tensorboard to be visualized later
    '''
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(val_loader):

            if blurring and random.random() < 0.25:  # Blur the Image with probability 1/4
                g_blur = transforms.GaussianBlur(11, sigma=(2, 10))
                data = g_blur(data)

            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
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
    writer.add_scalar('validation_loss', validation_loss, epoch)

    return val_loss, score.data.item(), stdout, writer


def main(model, epochs, batch_size, train_loader, val_loader, use_cuda, log_interval, scheduler,
         early_stopping, writer, stdout, accumulation_steps, device, blurring=0):
    '''
    main function to train a model

    Parameters:
        model (nn.Module pytorch model): pytorch model
        epochs (int): number of epochs
        batch_size (int): number of examples by batch
        train_loader (DataLoader): loader of the training set
        val_loader (DataLoader): loader of the validation Images
        use_cuda (int): computing with GPU or not
        log_interval (int): batch interval between saving weights value and printing loss
        scheduler (scheduler object): scheduler to update the learning rate at each epoch
        early_stopping (EarlyStopping object): Stop training if val loss has been not decreasing for patience epochs
        stdout (list): output file with printing messages - save in list format
        writer (SummaryWriter object): write the weights, gradients and loss values on Tensorboard to be visualized later
        accumulation_steps (int): perform backpropagation every accumulation_steps batches
        device (torch.device object): 'cpu' device or 'cuda' device


    Return:
        model (nn.Module pytorch model): pytorch model trained
        train_loss (list): list of training loss values by epoch
        val_loss (list): list of validation loss values by epoch
        val_accuracy (list): list of validation accuracy by epoch
        epoch_time (list): time per epoch
        stdout (list): list of string with messages printed in order to save output file.
    '''

    train_loss, val_loss, val_accuracy, epoch_time = [], [], [], []
    n_batches = len(train_loader.dataset) // batch_size

    for epoch in range(1, epochs + 1):
        stdout = logging("Epoch {} - Start TRAINING".format(epoch), stdout)
        t0 = time.time()
        # Training for one epoch
        model, train_loss, stdout = train(epoch, model, train_loader, use_cuda, log_interval, train_loss,
                                          stdout, writer, n_batches, batch_size, accumulation_steps, device, blurring)

        # Testing mode - test on the validation set
        val_loss, accuracy, stdout, writer = validation(model, epoch, val_loader, use_cuda, val_loss, stdout, writer,
                                                        blurring)

        val_accuracy.append(accuracy)

        # Update learning rate
        scheduler.step()

        # Early stopping
        early_stopping(val_loss[-1], model, stdout)
        if early_stopping.early_stop:
            message = "Early stopping"
            print(message)
            stdout.append(message)
            break

        # Save time elapsed
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
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--optimizer', type=str, default='adam', help='Adam or SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay ADAM optimizer')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                        help='folder where experiment outputs are located.'),
    parser.add_argument('--model', type=str, default='resnet101', help='model to run'),
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability'),
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience"),
    parser.add_argument('--debug', type=int, default=0, help="debug"),
    parser.add_argument('--cfg', type=str, default='vit_large_patch16_224', help='Config ViT model'),
    parser.add_argument('--horizontal_flip', type=int, default=1, help="perform hor. flip data augmentation"),
    parser.add_argument('--vertical_flip', type=int, default=1, help="perform vert. flip data augmentation"),
    parser.add_argument('--random_rotation', type=int, default=1, help="perform random rotation from -45° to 45°"),
    parser.add_argument('--erasing', type=int, default=1, help="perform random erasing or not"),
    parser.add_argument('--accumulation_steps', type=int, default=1, help="Gradient accumulation for GPU RAM"),
    parser.add_argument('--size', type=int, default=224, help='size of the input images')
    parser.add_argument('--blurring', type=int, default=0, help="Perform Gaussian blur or not")
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
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

    # Load model - pretrained from ImageNet
    model = load_model(path_model='',
                       model_type=args.model,
                       dropout=args.dropout,
                       cfg=args.cfg,
                       use_cuda=use_cuda,
                       load_weights=0)

    # Transform data
    data_transforms_train = data_transformation(horizontal_flip=args.horizontal_flip,
                                                random_rotation=args.random_rotation,
                                                erasing=args.erasing,
                                                model=args.model,
                                                size=args.size,
                                                train=1)
    data_transforms_dev = data_transformation(model=args.model, size=args.size, train=0)

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/train_images',
                             transform=data_transforms_train),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                             transform=data_transforms_dev),
        batch_size=args.batch_size, shuffle=False, num_workers=1)

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
                   'model': args.model,
                   'batch_size': args.batch_size,
                   'epochs': args.epochs,
                   'lr': args.lr,
                   'optimizer': args.optimizer,
                   'dropout': args.dropout,
                   "patience": args.patience,
                   "weight_decay (adam)": args.weight_decay,
                   "momentum (sgd)": args.momentum,
                   "scheduler": "CosineAnnealing",
                   "horizontal_flip": args.horizontal_flip,
                   "vertical_flip": args.vertical_flip,
                   "random_rotation": args.random_rotation,
                   "Gaussian blur": args.blurring,
                   "erasing": args.erasing}
        if args.model == 'vit':
            results['cfg'] = args.cfg

        stdout += ['{} : {}'.format(str(k), str(v)) for k, v in results.items()]
        stdout.append(' ')

    # Optimizer
    params = add_weight_decay(model, args.weight_decay)
    if args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise NameError("Non-recognized optimizer - please use --optimizer adam or --optimizer sgd (default: adam)")

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    # Set up early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(path_result, 'model.pt'))

    # Summary Writer - Tensorboard
    path_report = os.path.join(path_result, 'report')
    writer = SummaryWriter(log_dir=path_report)
    model, train_loss, val_loss, val_accuracy, epoch_time, stdout = main(model, args.epochs, args.batch_size,
                                                                         train_loader, val_loader, use_cuda,
                                                                         args.log_interval, scheduler,
                                                                         early_stopping, writer, stdout,
                                                                         args.accumulation_steps, device, args.blurring)

    results['train_loss'] = train_loss
    results['val_loss'] = val_loss
    results['val_accuracy'] = val_accuracy
    results['epoch_time'] = epoch_time

    # Saving results
    with open(os.path.join(path_result, 'results.json'), 'w') as fj:
        json.dump(results, fj)

    stdout = logging("End of training - save stdout file", stdout)
    with open(os.path.join(path_result, 'stdout.txt'), 'w') as f:
        f.write('\n'.join(stdout))


