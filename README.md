# FGVC8-Competition

This [competition](https://www.kaggle.com/c/imet-2021-fgvc8/overview) is part of the 8th Fine-Grained Visual Categorization Workshop at the 2021 Computer Vision and Pattern Recognition (CVPR) Conference. The goal is to recognize artwork attributes using digitized art objects provided by The Metropolitan Museum of Arts and Art Institute of Chicago. 

## Data Description
The provided dataset contains a large number of artwork images annotated with their corresponding attributes.  The photographs are often centered for objects, and in the case where the museum artifact is an entire room, the images are scenic in nature. Each object is annotated by a single annotator without a verification step. Annotators were advised to add multiple labels from an ontology provided by The Met, and additionally are allowed to add free-form text when they see fit. Therefore, the problem is a *multi-label classification task*. There are 5 high-level labels: {'country': 100 values, 'culture': 681 values, 'dimension': 5 values, 'medium': 1920 values, 'tags': 768 values}.

The competition metric, F2 score, was intentionally chosen to provide some robustness against noisy labels, favoring recall over precision.

## Proposed Solution
### Data Preprocessing: 
* Random crop if needed
* Horizontal flip
* Random brightness/contrast modification
* Gaussian noise
* Data standardization using the mean and standard deviation of ImageNet data

### Train
The solution is based on Convolutional neural Network (CNN) models. The code enables to run 4 CNNs: SE-ResNext-50, SE-ResNext-101, VGG-16 with batch normalization, and ResNet-18.
The focal loss is used for its ability to handle imbalanced datasets and yield good performance when the objects to be classified are small.
A learning rate decay was applied, using the cosine annealing scheduler.
All training experiments run with 5-fold cross validation.

To launch the training, simply run:
 `python train.py --Model_Name "se_resnext50"`
### Test
Ensemble models gave the best results. Models were combined by computing the average of the output dense layer (after the sigmoid).

To test a model, simply run:
 `python test.py --Weights_Path "Path To the weights of the models .pth" --Model_Name "se_resnext50"`
