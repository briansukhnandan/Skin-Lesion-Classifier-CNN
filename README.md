## Skin-Lesion-Classifier-CNN
Convolutional Neural Network to classify different types of Skin Lesions based on 7 different categories: 
- Melanocytic nevi 
- Melanoma 
- Basal cell carcinoma 
- Actinic Keratoses 
- Benign keratosis 
- Vascular Skin Lesions 
- Dermatofibroma

Inspired by this study: https://arxiv.org/pdf/1810.10348.pdf

# How do I use it?
Firstly, the dependencies needed are: 
os, tensorflow, pandas, numpy, matplotlib, itertools, sklearn, shutil, and PIL.

Next, acquire a copy of the HAM10000 dataset with my modifications (alternatively can be obtained online):
https://mega.nz/file/UN1kxQTb#KgZ5Nbjp6h-MMPuzMW1uBnGgE7Tgo2xd4lTkJplUrWI

Run 'generate_model.py' in the same directory as the dataset, which will produce a 'saved_model/' directory to run
the model later, as well as a 'history.npy' file which keeps all attributes of the model.

Afterwards, run 'load_model.py' to produce a confusion matrix and a classification report pertaining to the model.

# Results
After training this model on 25 epochs, we are given the following results:

Confusion matrix, without normalization
[[ 13   1   3   0   3   6   0]
 [  0  26   2   2   0   0   0]
 [  1   1  42   2  11  18   0]
 [  0   0   0   2   1   3   0]
 [  0   2   6   0  19  12   0]
 [  3   3   7   0  10 726   2]
 [  1   0   0   0   0   3   7]]
 

Classification Report:
              precision    recall  f1-score   support

       akiec       0.72      0.50      0.59        26
         bcc       0.79      0.87      0.83        30
         bkl       0.70      0.56      0.62        75
          df       0.33      0.33      0.33         6
         mel       0.43      0.49      0.46        39
          nv       0.95      0.97      0.96       751
        vasc       0.78      0.64      0.70        11

    accuracy                           0.89       938
   macro avg       0.67      0.62      0.64       938
weighted avg       0.89      0.89      0.89       938


As well as the produced graphs:
![graph_1](https://i.imgur.com/1vzjR5Q.png)
