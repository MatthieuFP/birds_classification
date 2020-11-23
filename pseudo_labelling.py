# -*- coding: utf-8 -*-

"""
Created on Sat Nov 7 12:47:39 2020

@author: matthieufuteral-peter
"""

import random
import argparse
import json
import os
import time
from utils.utils import *
from utils.augmented_dataset import TransformFixMatch
import numpy as np
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
                     train_labeled_loss, stdout, writer, optimizer, n_batches, batch_size, threshold,
                     strong_augmentation, T2, factor, proba, device, loss_smoothing=0, smooth_prob=0.2):
    '''
    Training step of the pseudo labelling algorithm with strong augmentation method

    Parameters:
        model (nn.Module pytorch model): pytorch model
        epoch (int): n-th epoch
        train_loader (DataLoader): loader of the training set
        unlabeled_loader (DataLoader): loader of the unlabeled NABirds Images
        use_cuda (int): computing with GPU or not
        log_interval (int): batch interval between saving weights value and printing loss
        train_unlabeled_loss (list): list of the unlabeled loss per epoch
        train_labeled_loss (list): list of the labeled loss per epoch
        stdout (list): output file with printing messages - save in list format
        writer (SummaryWriter object): write the weights, gradients and loss values on Tensorboard to be visualized later
        optimizer (torch.optim object): optimizer
        n_batches (int): number of batches to perform 1 epoch
        batch_size (int): number of examples by batch
        threshold (float): proba value to assign a label to an unlabeled example
        strong_augmentation (int): Strongly transform the image before training (see FixMatch)
        T2 (int): see utils/utils alpha function
        factor (int): see utils/utils alpha function
        proba (float): probability to choose an unlabeled batch over a labeled batch during training
        device (torch.device object): 'cpu' device or 'cuda' device
        loss_smoothing (int): 0 or 1. Smooth unlabeled loss or not (RECOMMENDED)
        smooth_prob (float): if loss_smoothing else ignore => how much loss is smoothed

    Return:
        model (nn.Module pytorch model): pytorch model trained for 1 epoch
        train_unlabeled_loss (list): list of unlabeled loss values
        train_labeled_loss (list): list of labeled loss values
        stdout (list): list of string with messages printed in order to save output file.
    '''

    train_batch_unlabeled_loss = []
    train_batch_labeled_loss = []
    n_unlabeled_chosen = 0  # Count the number of unlabeled examples chosen during 1 epoch
    n_sample = 0  # Count the number of unlabeled examples used for training (i.e. output > threshold)
    n_labeled = 0  # Count the number of labeled examples chosen during 1 epoch

    optimizer.zero_grad()

    # Training mode
    model.train()

    for batch_idx in tqdm_notebook(range(n_batches)):
        n_iter = batch_idx + (epoch - 1) * n_batches

        if not (batch_idx + 1) % log_interval:
            print('{} batch : {} test unlabeled examples - {} sample examples'.format(batch_idx + 1, n_sample,
                                                                                      n_labeled))

        p = random.random()
        if p < proba:   # select unlabeled example with probability 'proba'
            n_unlabeled_chosen += batch_size

            pdb.set_trace()
            (weak_unlabeled_data, strong_unlabeled_data), _ = next(iter(unlabeled_loader))

            del _  # free memory usage

            if use_cuda:
                weak_unlabeled_data = weak_unlabeled_data.cuda()

            # Testing mode
            model.eval()
            output_unlabeled = model(weak_unlabeled_data)
            _, pseudo_labels = torch.max(output_unlabeled, 1)
            del _  # free memory usage

            probs = F.softmax(output_unlabeled, dim=-1)
            # Choose only index where max proba > threshold in order to be likely to choose an actual label.
            index = torch.where(probs.max(dim=-1).values > threshold)[0].cuda()

            if len(index):  # if the model was confident enough to choose at least one example with proba > threshold
                n_sample += len(index)
                weak_unlabeled_data = weak_unlabeled_data[index]
                pseudo_labels = pseudo_labels[index]

                # Training mode
                model.train()
                if strong_augmentation:  # if consistency regularization (see FixMatch paper)
                    pdb.set_trace()

                    strong_unlabeled_data = strong_unlabeled_data[index]
                    if use_cuda:
                        strong_unlabeled_data = strong_unlabeled_data.cuda()

                    output = model(strong_unlabeled_data)

                else:  # If not consistency regularization only perform pseudo labelling with loss smoothing
                    output = model(weak_unlabeled_data)

                if loss_smoothing:
                    # Unlabeled loss with pseudo labels + loss smoothing (pseudo labels are turned into a vector of
                    # probability with max proba = 1 - smooth_prob
                    unlabeled_loss = alpha(epoch, T2=T2, factor=factor) * smooth_loss(target=pseudo_labels,
                                                                                      output=output,
                                                                                      n_classes=20,
                                                                                      smoothing=smooth_prob)
                else:
                    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
                    # Unlabeled loss with pseudo labels
                    unlabeled_loss = alpha(epoch, T2=T2, factor=factor) * criterion(output, pseudo_labels)

                # Backward propagation
                unlabeled_loss.backward()

                train_batch_unlabeled_loss.append(unlabeled_loss.data.item() / len(index))  # Save unlabeled loss
                optimizer.step()
                optimizer.zero_grad()

                del output, output_unlabeled, probs, pseudo_labels

            del weak_unlabeled_data, strong_unlabeled_data  # Free space from GPU memory

        else:  # Else perform one labeled batch loop
            n_labeled += 1
            (data, target) = next(iter(train_loader))

            if use_cuda:
                data = data.cuda()
                target = target.cuda()

            output = model(data)

            # Compute labeled loss
            # To balance the fact there is no groove bill Ani in the unlabeled set, I empirically added weights to the
            # loss
            loss_weights = torch.ones(20)
            loss_weights[0] = 4
            loss_weights = loss_weights.to(device)
            criterion = torch.nn.CrossEntropyLoss(loss_weights)
            labeled_loss = criterion(output, target)

            # Backpropagation
            optimizer.zero_grad()
            labeled_loss.backward()
            optimizer.step()

            train_batch_labeled_loss.append(labeled_loss.item() / batch_size)

            # Remove space from GPU memory
            del data, target, output

        if not (batch_idx + 1) % log_interval:  # Write weights and gradients on tensorboard every log_interval batch
            writer.add_histogram('layer1', model.layer1.weight, n_iter)  # Check layer1 weights
            writer.add_histogram('layer1', model.layer1.weight.grad, n_iter)  # Check layer1 gradient
            writer.add_histogram('layer2', model.layer2.weight, n_iter)  # Check layer1 weights
            writer.add_histogram('layer2', model.layer2.weight.grad, n_iter)  # Check layer1 gradient

    train_unlabeled_loss.append(np.mean(train_batch_unlabeled_loss))
    train_labeled_loss.append(np.mean(train_batch_labeled_loss))
    print('\n')
    stdout = logging('Unlabeled Sample = {} out of {}'.format(n_sample, len(unlabeled_loader.dataset)), stdout)
    stdout = logging('Unlabeled Train loss Epoch {} : {}'.format(epoch, train_unlabeled_loss[-1]), stdout)

    return model, train_unlabeled_loss, train_labeled_loss, stdout


def main(model, epochs, batch_size, train_loader, unlabeled_loader, val_loader, use_cuda, log_interval, scheduler,
         early_stopping, writer, stdout, threshold, strong_augmentation, T2, factor, proba, device, optimizer,
         loss_smoothing=0, smooth_prob=0.2):
    '''
    main function to train a model with pseudo labelling algorithm

    Parameters:
        model (nn.Module pytorch model): pytorch model
        epochs (int): number of epochs
        batch_size (int): number of examples by batch
        train_loader (DataLoader): loader of the training set
        unlabeled_loader (DataLoader): loader of the unlabeled NABirds Images
        val_loader (DataLoader): loader of the validation Images
        use_cuda (int): computing with GPU or not
        log_interval (int): batch interval between saving weights value and printing loss
        scheduler (scheduler object): scheduler to update the learning rate at each epoch
        early_stopping (EarlyStopping object): Stop training if val loss has been not decreasing for patience epochs
        stdout (list): output file with printing messages - save in list format
        writer (SummaryWriter object): write the weights, gradients and loss values on Tensorboard to be visualized later
        optimizer (torch.optim object): optimizer
        n_batches (int): number of batches to perform 1 epoch
        threshold (float): proba value to assign a label to an unlabeled example
        strong_augmentation (int): Strongly transform the image before training (see FixMatch)
        T2 (int): see utils/utils alpha function
        factor (int): see utils/utils alpha function
        proba (float): probability to choose an unlabeled batch over a labeled batch during training
        device (torch.device object): 'cpu' device or 'cuda' device
        loss_smoothing (int): 0 or 1. Smooth unlabeled loss or not (RECOMMENDED)
        smooth_prob (float): if loss_smoothing else ignore => how much loss is smoothed

    Return:
        model (nn.Module pytorch model): pytorch model trained
        train_unlabeled_loss (list): list of unlabeled loss values by epoch
        train_labeled_loss (list): list of labeled loss values by epoch
        val_loss (list): list of validation loss values by epoch
        val_accuracy (list): list of validation accuracy by epoch
        epoch_time (list): time per epoch
        stdout (list): list of string with messages printed in order to save output file.
    '''

    train_unlabeled_loss, train_labeled_loss, val_loss, val_accuracy, epoch_time = [], [], [], [], []
    n_batches = (len(train_loader.dataset) + len(unlabeled_loader.dataset)) // batch_size

    for epoch in range(1, epochs + 1):

        stdout = logging("Epoch {} - Start TRAINING".format(epoch), stdout)
        t0 = time.time()

        # Training step
        model, train_unlabeled_loss, train_labeled_loss, stdout = pseudo_labelling(model, epoch, train_loader,
                                                                                   unlabeled_loader, use_cuda,
                                                                                   log_interval, train_unlabeled_loss,
                                                                                   train_labeled_loss, stdout, writer,
                                                                                   optimizer, n_batches, batch_size,
                                                                                   threshold, strong_augmentation, T2,
                                                                                   factor, proba, device, loss_smoothing,
                                                                                   smooth_prob)

        # Validation step - check that loss value decreases
        val_loss, accuracy, stdout, writer = validation(model, epoch, val_loader, use_cuda, val_loss, stdout, writer)

        # Update learning rate
        scheduler.step()

        val_accuracy.append(accuracy)

        # Check whether loss decrease - Save model if so otherwise perform early stopping after
        # 5 epochs of non decreasing loss
        early_stopping(val_loss[-1], model, stdout)
        if early_stopping.early_stop:
            message = "Early stopping"
            print(message)
            stdout.append(message)
            break

        # Save time elapsed per epoch
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
    parser.add_argument('--data', type=str, default='cropped_birds', metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
    parser.add_argument('--batch_size', type=int, default=8, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--experiment', type=str, default='experiment')
    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate (default: 2e-5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay ADAM optimizer')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model', type=str, default='vit', help='model to run')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability')
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--debug', type=int, default=0, help="debug")
    parser.add_argument('--cfg', type=str, default='vit_large_patch16_224', help='Config ViT model')
    parser.add_argument('--horizontal_flip', type=int, default=1, help="perform hor. flip data augmentation")
    parser.add_argument('--vertical_flip', type=int, default=1, help="perform vert. flip data augmentation")
    parser.add_argument('--random_rotation', type=int, default=0, help="perform random rotation from -45° to 45°")
    parser.add_argument('--erasing', type=int, default=0, help="perform random erasing or not")
    parser.add_argument('--size', type=int, default=224, help='size of the input images')
    parser.add_argument('--threshold', type=float, default=0.8, help='if max(prob) < threshold, unlabeled example is not'
                                                                     ' considered')
    parser.add_argument('--RUN_ID', type=str, required=True, help="RUN_ID of the pre trained model")
    parser.add_argument('--strong_augmentation', type=int, default=1, help="perform strong augmentation or not")
    parser.add_argument('--T2', type=int, default=5, help="T2 value")
    parser.add_argument('--factor', type=int, default=2, help="factor value")
    parser.add_argument('--proba', type=float, default=0.66)
    parser.add_argument('--loss_smoothing', type=int, default=0)
    parser.add_argument('--smooth_prob', type=float, default=0.2)
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # Debogger
    if not args.debug:
        pdb.set_trace = lambda: None

    print('Pseudo Labelling')
    stdout = ['Pseudo Labelling']  # Save stdout file
    RUN_ID = str(uuid4())[0:5]
    print('RUN_ID : {}'.format(RUN_ID))
    device = torch.device("cuda" if use_cuda else "cpu")

    path = os.getcwd()
    path_experiments = os.path.join(path, 'semi_supervized_experiment')
    path_id = os.path.join(args.experiment, args.RUN_ID)
    path_load_model = os.path.join(path_id, 'model.pt')
    path_model = os.path.join(path_experiments, RUN_ID, 'model.pt')

    # Create experiment folder
    if not os.path.isdir("semi_supervized_experiment"):
        os.makedirs("semi_supervized_experiment")

    # Load model
    model = load_model(path_model=path_load_model,
                       model_type=args.model,
                       dropout=args.dropout,
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

    unlabeled_loader = load_nabirds(args.batch_size, TransformFixMatch())

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                             transform=data_transforms_dev),
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    # Create folder results
    path_result = os.path.join("semi_supervized_experiment", RUN_ID)
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
                   "strong_augmentation": args.strong_augmentation,
                   "proba": args.proba,
                   "loss_smoothing": args.loss_smoothing}
        if args.model == 'vit':
            results['cfg'] = args.cfg

        stdout += ['{} : {}'.format(str(k), str(v)) for k, v in results.items()]
        stdout.append(' ')

    # Optimizer
    params = add_weight_decay(model, args.weight_decay)
    # Optimizer Unlabeled & Labeled
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    # Set up early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(path_result, 'model.pt'))

    # Summary Writer - Tensorboard
    path_report = os.path.join(path_result, 'report')
    writer = SummaryWriter(log_dir=path_report)

    model, train_unlabeled_loss, train_labeled_loss, val_loss, val_accuracy, \
    epoch_time, stdout = main(model, args.epochs, args.batch_size, train_loader, unlabeled_loader, val_loader, use_cuda,
                              args.log_interval, scheduler, early_stopping, writer, stdout, args.threshold,
                              args.strong_augmentation, args.T2, args.factor, args.proba, device, optimizer,
                              args.loss_smoothing, args.smooth_prob)

    results['train_unlabeled_loss'] = train_unlabeled_loss
    results['train_labeled_loss'] = train_labeled_loss
    results['val_loss'] = val_loss
    results['val_accuracy'] = val_accuracy
    results['epoch_time'] = epoch_time

    # Saving results...
    with open(os.path.join(path_result, 'results.json'), 'w') as fj:
        json.dump(results, fj)

    stdout = logging("End of training - save stdout file", stdout)
    with open(os.path.join(path_result, 'stdout.txt'), 'w') as f:
        f.write('\n'.join(stdout))
