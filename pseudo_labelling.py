# -*- coding: utf-8 -*-

"""
Created on Sat Nov 7 12:47:39 2020

@author: matthieufuteral-peter
"""


import argparse
import json
import os
import time
from utils.utils import *
from utils.randaugment import RandAugmentMC
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm, tqdm_notebook
from model import *
from early_stopping.pytorchtools import EarlyStopping
from main import validation
from torch.utils.tensorboard import SummaryWriter
import pdb
from uuid import uuid4


def pseudo_labelling(model, epoch, train_loader, unlabeled_loader, use_cuda, log_interval, train_unlabeled_loss,
                     train_labeled_loss, stdout, writer, optimizer, n_batches, batch_size, accumulation_steps,
                     threshold, n_split, strong_augmentation):

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    train_batch_unlabeled_loss = []
    train_batch_labeled_loss = []
    n_sample = 0

    optimizer.zero_grad()

    model.train()
    for batch_idx, (unlabeled_data, _) in tqdm_notebook(enumerate(unlabeled_loader)):
        del _  # memory usage

        n_iter = batch_idx + (epoch - 1) * n_batches
        if use_cuda:
            unlabeled_data = unlabeled_data.cuda()

        model.eval()
        output_unlabeled = model(unlabeled_data)
        _, pseudo_labels = torch.max(output_unlabeled, 1)
        del _  # memory usage

        probs = F.softmax(output_unlabeled, dim=-1)
        index = torch.where(probs > threshold)[0]  # Only use images that are likely to be in our 20 classes

        pdb.set_trace()
        if len(index):  # index.size()
            n_sample += len(index)
            unlabeled_data = unlabeled_data[index]
            pseudo_labels = pseudo_labels[index]

            model.train()
            if strong_augmentation:
                strong_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                       RandAugmentMC(n=2, m=10),
                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])])
                unlabeled_data = strong_transform(unlabeled_data)

            # Unlabeled loss with pseudo labels
            output = model(unlabeled_data)

            unlabeled_loss = alpha(epoch, T2=10, factor=2) * criterion(output, pseudo_labels)
            unlabeled_loss.backward()

            train_batch_unlabeled_loss.append(unlabeled_loss.data.item() / len(index))  # Save unlabeled loss

            if (batch_idx + 1) % accumulation_steps == 0:  # Gradient accumulation to handle limit of GPU RAM
                optimizer.step()
                optimizer.zero_grad()

            logger.info('Unlabeled Train Epoch: {} [{}/{} ({:.0f}%)]\t Average Loss: {:.6f}'.format(
                        epoch, batch_idx * batch_size, len(unlabeled_loader.dataset),
                        100. * batch_idx / len(unlabeled_loader), unlabeled_loss.data.item() / len(index)))

            del unlabeled_data, pseudo_labels, output_unlabeled, probs  # Free space from GPU memory

        # For every n_split batches train one epoch on labeled data
        if not (batch_idx + 1) % n_split:
            print("Split reach - Start training 1 epoch on labeled data")

            # Normal training procedure
            for batch_idx, (data, target) in tqdm_notebook(enumerate(train_loader)):

                if use_cuda:
                    data = data.cuda()
                    target = target.cuda()

                output = model(data)

                labeled_loss = criterion(output, target)

                # Backpropagation
                optimizer.zero_grad()
                labeled_loss.backward()
                optimizer.step()

                train_batch_labeled_loss.append(labeled_loss.item() / batch_size)

                # Remove space from GPU memory
                del data, target, output

            train_labeled_loss.append(np.mean(train_batch_labeled_loss))

            stdout = logging('Labeled Train Epoch: {} \tLoss: {:.6f}'.format(epoch, train_labeled_loss[-1]), stdout)

        # if batch_idx % log_interval == 0:
        #    writer.add_histogram('classifier', model.classifier.weight, n_iter)  # Check classifier weights
        #    writer.add_histogram('grad_classifier', model.classifier.weight.grad, n_iter)  # Check classifier gradient
        #    try:
        #        writer.add_histogram('linear_layer', model.linear.weight, n_iter)  # Check linear layer weights
        #        writer.add_histogram('grad_linear', model.linear.weight.grad, n_iter)  # Check linear layer gradient
        #    except:
        #        continue

        # writer.add_scalar('unlabeled_loss', unlabeled_loss.data.item(), n_iter)
        # if not (batch_idx + 1) % n_split == 0:
        #    writer.add_scalar('labeled_loss', labeled_loss.data.item(), n_iter)

    # pdb.set_trace()
    train_unlabeled_loss.append(np.mean(train_batch_unlabeled_loss))
    stdout = logging('Unlabeled Sample = {} out of {}'.format(n_sample, len(unlabeled_loader.dataset)), stdout)
    stdout = logging('Unlabeled Train loss Epoch {} : {}'.format(epoch, train_unlabeled_loss[-1]), stdout)

    return model, train_unlabeled_loss, train_labeled_loss, stdout


def main(model, epochs, batch_size, train_loader, unlabeled_loader, val_loader, use_cuda, log_interval, scheduler,
         early_stopping, writer, stdout, accumulation_steps, threshold, n_split, strong_augmentation):

    train_unlabeled_loss, train_labeled_loss, val_loss, val_accuracy, epoch_time = [], [], [], [], []
    n_batches = len(train_loader.dataset) // batch_size

    for epoch in range(1, epochs + 1):

        stdout = logging("Epoch {} - Start TRAINING".format(epoch), stdout)
        t0 = time.time()

        model, train_unlabeled_loss, train_labeled_loss, stdout = pseudo_labelling(model, epoch, train_loader,
                                                                                   unlabeled_loader, use_cuda,
                                                                                   log_interval, train_unlabeled_loss,
                                                                                   train_labeled_loss, stdout,
                                                                                   writer, optimizer, n_batches,
                                                                                   batch_size, accumulation_steps,
                                                                                   threshold, n_split, strong_augmentation)

        val_loss, accuracy, stdout = validation(model, epoch, val_loader, use_cuda, val_loss, stdout)

        scheduler.step()

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

    return model, train_unlabeled_loss, train_labeled_loss, val_loss, val_accuracy, epoch_time, stdout


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='RecVis A3 training script')
    parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
    parser.add_argument('--batch_size', type=int, default=64, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay ADAM optimizer')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default='semi_supervized_experiment', metavar='E',
                        help='folder where experiment outputs are located.')
    parser.add_argument('--model', type=str, default='resnet101', help='model to run')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability')
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--debug', type=int, default=0, help="debug")
    parser.add_argument('--cfg', type=str, default='vit_large_patch16_224', help='Config ViT model')
    parser.add_argument('--horizontal_flip', type=int, default=1, help="perform hor. flip data augmentation")
    parser.add_argument('--vertical_flip', type=int, default=1, help="perform vert. flip data augmentation")
    parser.add_argument('--random_rotation', type=int, default=1, help="perform random rotation from -45° to 45°")
    parser.add_argument('--erasing', type=int, default=1, help="perform random erasing or not")
    parser.add_argument('--accumulation_steps', type=int, default=1, help="Gradient accumulation for GPU RAM")
    parser.add_argument('--size', type=int, default=224, help='size of the input images')
    parser.add_argument('--n_split', type=int, default=3000, help='number of unlabeled batch before train labeled batch')
    parser.add_argument('--threshold', type=float, default=0.8, help='if max(prob) < threshold, unlabeled example is not'
                                                                     ' considered')
    parser.add_argument('--RUN_ID', type=str, required=True, help="RUN_ID of the pre trained model")
    parser.add_argument('--strong_augmentation', type=int, default=1, help="perform strong augmentation or not")
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if not args.debug:
        pdb.set_trace = lambda: None

    print('Pseudo Labelling')
    stdout = ['Pseudo Labelling']  # Save stdout file
    RUN_ID = str(uuid4())[0:5]
    print('RUN_ID : {}'.format(RUN_ID))
    device = torch.device("cuda" if use_cuda else "cpu")

    path = os.getcwd()
    path_experiments = os.path.join(path, 'semi_supervized_experiment')
    path_id = os.path.join("experiment", args.RUN_ID)
    path_load_model = os.path.join(path_id, 'model.pt')
    path_model = os.path.join(path_experiments, RUN_ID, 'model.pt')

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Load model
    model = load_model(path_model=path_load_model,
                       model_type=args.model,
                       cfg=args.cfg,
                       use_cuda=use_cuda,
                       load_weights=1)

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

    unlabeled_loader = load_nabirds(args.batch_size, data_transforms_train)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                             transform=data_transforms_dev),
        batch_size=args.batch_size, shuffle=False, num_workers=1)


    # Create folder results
    path_result = os.path.join(args.experiment, RUN_ID)
    if not os.path.isdir(path_result):
        os.makedirs(path_result)
        results = {'RUN_ID': RUN_ID,
                   'model': args.model,
                   'batch_size': args.batch_size,
                   'epochs': args.epochs,
                   'lr': args.lr,
                   'dropout': args.dropout,
                   "patience": args.patience,
                   "weight_decay": args.weight_decay,
                   "scheduler": "CosineAnnealing",
                   "horizontal_flip": args.horizontal_flip,
                   "vertical_flip": args.vertical_flip,
                   "random_rotation": args.random_rotation,
                   "erasing": args.erasing,
                   "pseudo_labelling": 1,
                   "strong_augmentation": args.strong_augmentation}
        if args.model == 'vit':
            results['cfg'] = args.cfg

        stdout += ['{} : {}'.format(str(k), str(v)) for k, v in results.items()]
        stdout.append(' ')

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, verbose=True)

    # Set up early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(path_result, 'model.pt'))

    # Summary Writer - Tensorboard
    path_report = os.path.join(path_result, 'report')
    writer = SummaryWriter(log_dir=path_report)
    model, train_unlabeled_loss, train_labeled_loss, val_loss, val_accuracy, epoch_time, stdout = \
        main(model, args.epochs, args.batch_size, train_loader, unlabeled_loader, val_loader, use_cuda,
             args.log_interval, scheduler, early_stopping, writer, stdout, args.accumulation_steps, args.threshold,
             args.n_split, args.strong_augmentation)

    pdb.set_trace()
    results['train_unlabeled_loss'] = train_unlabeled_loss
    results['train_labeled_loss'] = train_labeled_loss
    results['val_loss'] = val_loss
    results['val_accuracy'] = val_accuracy
    results['epoch_time'] = epoch_time

    with open(os.path.join(path_result, 'results.json'), 'w') as fj:
        json.dump(results, fj)

    stdout = logging("End of training - save stdout file", stdout)
    with open(os.path.join(path_result, 'stdout.txt'), 'w') as f:
        f.write('\n'.join(stdout))





