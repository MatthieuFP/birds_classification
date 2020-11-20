import argparse
import json
import os
import random
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


def train(epoch, model, train_loader, use_cuda, log_interval, train_loss, stdout, writer, n_batches, batch_size,
          accumulation_steps, device, blurring=0):

    optimizer.zero_grad()
    model.train()
    train_batch_loss = []
    #loss_weights = torch.ones(20)
    #loss_weights[16] = 3
    #loss_weights = loss_weights.to(device)

    pdb.set_trace()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        n_iter = batch_idx + (epoch - 1) * n_batches

        if random.random() < 0.25:  # Blur the Image with probability 1/4
            g_blur = transforms.GaussianBlur(11, sigma=(2, 10))
            data = g_blur(data)
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        # optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')   # loss_weights
        loss = criterion(output, target)
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:  # Gradient accumulation to handle limit of GPU RAM
            optimizer.step()
            optimizer.zero_grad()

        # optimizer.step()
        if batch_idx % log_interval == 0:
          #  writer.add_histogram('classifier', model.classifier.weight, n_iter)  # Check classifier weights
          #  writer.add_histogram('grad_classifier', model.classifier.weight.grad, n_iter)  # Check classifier gradient
          #  try:
          #      writer.add_histogram('linear_layer', model.linear.weight, n_iter)  # Check linear layer weights
          #      writer.add_histogram('grad_linear', model.linear.weight.grad, n_iter)  # Check linear layer gradient
          #  except:
          #      continue

            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

        # pdb.set_trace()
        train_batch_loss.append(loss.data.item() / batch_size)
        writer.add_scalar('train_loss', loss.data.item(), n_iter)

    del data, target  # leave space on gpu

    # pdb.set_trace()
    train_loss.append(np.mean(train_batch_loss))
    stdout = logging('Train loss Epoch {} : {}'.format(epoch, train_loss[-1]), stdout)

    return model, train_loss, stdout


def validation(model, epoch, val_loader, use_cuda, val_loss, stdout, writer, blurring=0):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in tqdm(val_loader):

        if random.random() < 0.25:  # Blur the Image with probability 1/4
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

    train_loss, val_loss, val_accuracy, epoch_time = [], [], [], []
    n_batches = len(train_loader.dataset) // batch_size

    for epoch in range(1, epochs + 1):
        stdout = logging("Epoch {} - Start TRAINING".format(epoch), stdout)
        t0 = time.time()
        model, train_loss, stdout = train(epoch, model, train_loader, use_cuda, log_interval, train_loss,
                                          stdout, writer, n_batches, batch_size, accumulation_steps, device, blurring)
        val_loss, accuracy, stdout, writer = validation(model, epoch, val_loader, use_cuda, val_loss, stdout, writer,
                                                        blurring)

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
    parser.add_argument('--lr_vit', type=float, default=2e-5)
    parser.add_argument('--lr_inc', type=float, default=2e-3)
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

    # Load model
    model = load_model(path_model=None,
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
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer_vit = torch.optim.Adam(model.vit.parameters(), lr=args.lr_vit, weight_decay=args.weight_decay)

    #fc_parameters = [value for name, value in model.named_parameters() if 'vit' not in name]
    #pdb.set_trace()
    #optimizer_inc = torch.optim.SGD(fc_parameters, lr=args.lr_inc,
    #                                momentum=args.momentum,
    #                                weight_decay=args.weight_decay)


    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)
    #scheduler_vit = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_vit, T_max=args.epochs)
    #scheduler_inc = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_inc, T_max=args.epochs)


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

    with open(os.path.join(path_result, 'results.json'), 'w') as fj:
        json.dump(results, fj)

    stdout = logging("End of training - save stdout file", stdout)
    with open(os.path.join(path_result, 'stdout.txt'), 'w') as f:
        f.write('\n'.join(stdout))


