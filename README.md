# Skin-Lesion-Classifier-CNN
A Convolutional Neural Network to classify different types of Malignant/Benign Skin Lesions based on 7 different categories: 
- Melanocytic nevi 
- Melanoma 
- Basal cell carcinoma 
- Actinic Keratoses 
- Benign keratosis 
- Vascular Skin Lesions 
- Dermatofibroma

Inspired by this study: https://arxiv.org/pdf/1810.10348.pdf

## How do I use it?
Firstly, have Python 3.6 installed. Next, install the required dependencies (Setting up a virtual environment would be helpful in this!): 
os, tensorflow, pandas, numpy, matplotlib, itertools, sklearn, shutil, and PIL.

Next, acquire a copy of the HAM10000 dataset with my personal modifications (alternatively can be obtained online):
https://mega.nz/file/UN1kxQTb#KgZ5Nbjp6h-MMPuzMW1uBnGgE7Tgo2xd4lTkJplUrWI

Run 'generate_model.py' in the same directory as the dataset, which will produce a 'saved_model/' directory to run
the model later, as well as a 'history.npy' file which keeps all attributes of the model.

Afterwards, run 'load_model.py' to evaluate the model, and then produce a confusion matrix and a classification report based on our results.

## Results
After training this model on 25 epochs, we are given the following results:

Confusion matrix, without normalization:

![graph_3](https://i.imgur.com/jjTRpEN.png)

Classification Report:

![graph_1](https://i.imgur.com/VaTBsCM.png)

As well as the produced graphs:

![graph_2](https://i.imgur.com/1vzjR5Q.png)
