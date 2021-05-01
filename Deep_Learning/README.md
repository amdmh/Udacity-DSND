# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line applications : train_classifier.py, predict_classifier.py

The image classifier to recognize different species of flowers. Dataset contains 102 flower categories.

In Image Classifier Project.ipynb VGG16 from torchvision.models pretrained models was used. It was loaded as a pre-trained network, based on which defined a new, untrained feed-forward network as a classifier, using ReLU activations and dropout. Trained the classifier layers using backpropagation using the pre-trained network to get the features. The loss and accuracy on the validation set were tracked to determine the best hyperparameters. 

### Command line applications train.py and predict.py

For command line applications, there is an option to select either Alexnet or VGG16 models. 

Following arguments mandatory or optional for *train_classifier.py* 

1. `data_dir`. 'Provide data directory. Mandatory argument', type = str
2. `--arch`. 'Alexnet can be used if this argument specified, otherwise VGG16 will be used', type = str
3. `--hidden_units`. 'Hidden units in Classifier. Default value is 4096', type = int
4. `--learning_rate`. 'Learning rate, default value 0.001', type = float
5. `--epochs`. 'Number of epochs', Default value is 8,  type = int
6. `--save_dir`. 'Provide saving directory. Optional argument', type = str
7. `--GPU`. "Option to use GPU", type = str

Following arguments mandatory or optional for predict_classifier.py

1.	'--image_path'. 'Provide path to image. Mandatory argument', type = str
2.	'--checkpoint'. 'Provide path to pre-trained model . Mandatory argument', type = str
3.	'--top_k'. 'Top K most likely classes. Optional', type = int
4.	'--json'. 'Use a mapping of categories to real names from a json file. Optional', type = str
7.	'--GPU'. "Option to use GPU. Optional", type = str
