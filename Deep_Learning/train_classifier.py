# Import libraries
import pandas as pd
import numpy as np

import torch
from torch import nn, optim

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from collections import OrderedDict
import time
import copy
import argparse

# Initializing default values
arch = 'vgg16'
hidden_units = 4096
learning_rate = 0.001
epochs = 8
device = 'cpu'

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
                    help='Location of directory with data for image classifier to train and test')
parser.add_argument('-a', '--arch', action='store', type=str,
                    help='Choose among 3 pretrained networks - vgg16, alexnet, and densenet121')
parser.add_argument('-H', '--hidden_units', action='store',
                    type=int, help='Select number of hidden units for 1st layer')
parser.add_argument('-l', '--learning_rate', action='store', type=float,
                    help='Choose a float number as the learning rate for the model')
parser.add_argument('-e', '--epochs', action='store', type=int,
                    help='Choose the number of epochs you want to perform gradient descent')
parser.add_argument('-s', '--save_dir', action='store', type=str,
                    help='Select name of file to save the trained model')
parser.add_argument('-g', '--gpu', action='store_true',
                    help='Use GPU if available')

args, _ = parser.parse_known_args()

# Select parameters for command line
if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model(arch='vgg16', hidden_units=4096, learning_rate=0.001):
    "Load and tune a pre-trained model"
    if arch == 'vgg16':
        # Load a pre-trained model
        model = models.vgg16(pretrained=True)
        # Load a pre-trained model
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError('Unexpected network architecture', arch)

    # Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False

    # Build classifier for model
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units, bias=True)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer


model, criterion, optimizer = build_model(arch, hidden_units, learning_rate)


# Data folders
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Default transforms for the training, validation, and testing sets
if args.data_dir:
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(root=args.data_dir +
                                '/' + x, transform=data_transforms[x])
        for x in list(data_transforms.keys())
    }

# Using the image datasets and the transforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
               for x in ['train', 'valid', 'test']}

# Calculate dataset sizes.
dataset_sizes = {
    x: len(dataloaders[x].dataset)
    for x in list(image_datasets.keys())
}


def train_model(model, criterion, optimizer, num_epochs=8):
    "This function trains a model"
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            if phase == 'valid':
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


trained_model = train_model(model, criterion, optimizer, epochs)


def save_checkpoint(trained_model, filename='checkpoint.pth'):
    "Save network architecture and parameters to file"
    trained_model.class_to_idx = image_datasets['train'].class_to_idx
    trained_model.cpu()
    save_dir = ''
    checkpoint = {
        'state_dict': trained_model.state_dict(),
        'class_to_idx': trained_model.class_to_idx,
    }

    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = 'checkpoint.pth'

    torch.save(checkpoint, save_dir, filename)


save_checkpoint(trained_model)
print(trained_model)
