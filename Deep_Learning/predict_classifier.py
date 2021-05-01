# Import libraries
import numpy as np
import torch
from torch import nn, optim

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
import PIL
from PIL import Image
import os
import json
import argparse


# Initializing default values
checkpoint = 'checkpoint.pth'
filepath = 'cat_to_name.json'
arch = ''
image_path = 'flowers/test/15/image_06360.jpg'
topk = 5

with open(filepath, 'r') as f:
    cat_to_name = json.load(f)


def load_checkpoint(filepath):
    "Load saved model from checkpoint"
    checkpoint = torch.load(filepath)

    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print('Base architecture is not recognized')
    # Instantiate parameterize model
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096, bias=True)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(4096, 102, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model

    # Building image transform
    img_loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    np_image = np.array(pil_image)

    # Normalize values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Set model to evaluate
    model.eval()

    # Extract and convert to numpy array to tensor for PyTorch
    np_array = process_image(image_path)
    tensor_image = torch.from_numpy(np_array)
    tensor_image = tensor_image.float()

    tensor_image = tensor_image.unsqueeze(0)

    with torch.no_grad():
        output = model.forward(tensor_image.cuda())

    # Reverse log conversion to cancel out the LogSoftMax
    output = torch.exp(output)

    # Predict topk probabilities, categories and labels
    topk_probs, topk_indexes = torch.topk(output, topk)

    # Converting ouputs to lists
    topk_probs = topk_probs.tolist()[0]
    topk_indexes = topk_indexes.tolist()[0]

    idx_to_cat = {val: key for key, val in model.class_to_idx.items()}

    top_cats = [idx_to_cat[index] for index in topk_indexes]

    top_labels = [cat_to_name[cat] for cat in top_cats]

    return topk_probs, top_labels,  # top_cats


# Set up parameters for command line
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', action='store', type=str,
                    help='Name of trained model to be loaded and used for predictions.')
parser.add_argument('-i', '--image_path', action='store', type=str,
                    help='Location of image to predict e.g. flowers/test/class/image')
parser.add_argument('-k', '--topk', action='store', type=int,
                    help='Select number of classes you wish to see in descending order.')
parser.add_argument('-j', '--json', action='store', type=str,
                    help='Define name of json file holding class names.')
parser.add_argument('-g', '--gpu', action='store_true',
                    help='Use GPU if available')

args, _ = parser.parse_known_args()

# Select parameters entered in command line
if args.checkpoint:
    checkpoint = args.checkpoint
if args.image_path:
    image_path = args.image_path
if args.topk:
    topk = args.topk
if args.json:
    filepath = args.json
if args.gpu:
    device = torch.device("cuda:0")

# Load saved model from checkpoint
model = load_checkpoint(filepath)

# Carry out prediction
probas, labels = predict(image_path, model, topk)

# Print probabilities and predicted classes
print('Probabilites of top categories:')
print(probas)
print('Labels of top categories:')
print(labels)
