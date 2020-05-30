import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

df_data = pd.read_csv('./HAM10000/HAM10000_metadata.csv')

# Return first rows of df_data, quickly check to see if all info was parsed.
df_data.head()

# This will tell us how many images are associated with each lesion_id.
df = df_data.groupby('lesion_id').count()

# Now we filter out lesion_id's that have only one image associated with it
df = df[df['image_id'] == 1]

df.reset_index(inplace=True)

df.head()

def identify_duplicates(x):
    unique_list = list(df['lesion_id'])

    if x in unique_list:
        return 'no_duplicates'
    else:
        return 'has_duplicates'


# create a new column that is a copy of the lesion_id column
df_data['duplicates'] = df_data['lesion_id']
# apply the function to this new column
df_data['duplicates'] = df_data['duplicates'].apply(identify_duplicates)

df_data.head()

df = df_data[df_data['duplicates'] == 'no_duplicates']

# Now we create a val set using df because we are sure that none of these images
# have augmented duplicates in the train set
y = df['dx']

_, df_val = train_test_split(df, test_size=0.17, random_state=101, stratify=y)

# This set will be df_data excluding all rows that are in the val set

# This function identifies if an image is part of the train
# or val set.
def identify_val_rows(x):
    # create a list of all the lesion_id's in the val set
    val_list = list(df_val['image_id'])

    if str(x) in val_list:
        return 'val'
    else:
        return 'train'


# create a new colum that is a copy of the image_id column
df_data['train_or_val'] = df_data['image_id']
# apply the function to this new column
df_data['train_or_val'] = df_data['train_or_val'].apply(identify_val_rows)

# filter out train rows
df_train = df_data[df_data['train_or_val'] == 'train']

# Set the image_id as the index in df_data
df_data.set_index('image_id', inplace=True)

# Get a list of train and val images
train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

train_path = './HAM10000/train_dir'
validation_path = './HAM10000/val_dir'

train_samples = len(df_train)
validation_samples = len(df_val)

train_batch_size = 10
validation_batch_size = 10
image_size = 299

# Create an Image Data Generator to input later into our model.
data_generation = ImageDataGenerator(
    # Use Inception ResNet v2
    preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input
)

### GETTING BATCHES FROM DIRECTORY PATHS ###
print("\nTrain Batches: ")
train_batches = data_generation.flow_from_directory(directory=train_path,
                                                    target_size=(299,299),
                                                    batch_size=train_batch_size,
                                                    shuffle=True
                                                    )

print("\nValidation Batches: ")
validation_batches = data_generation.flow_from_directory(directory=validation_path,
                                                         target_size=(299,299),
                                                         batch_size=validation_batch_size
                                                         )

print("\nTest Batches: ")
test_batches = data_generation.flow_from_directory(validation_path,
                                           target_size=(image_size,image_size),
                                           batch_size=1,
                                           shuffle=False
                                           )

# Load model from 'saved_model' folder which stores trained model
# from generate_model.py.
loaded_model = tf.keras.models.load_model('saved_model')

# history = loaded_model.fit(train_batches)

print("\nEvaluating model. . .")
val_loss, val_cat_acc = loaded_model.evaluate(test_batches, steps=len(df_val))
print("Evaluating completed!")

history = np.load('my_history.npy', allow_pickle='TRUE').item()

# Get dictionary values which are stored as numpy arrays.
acc = history['categorical_accuracy']
val_acc = history['val_categorical_accuracy']
loss = history['loss']
valid_loss = history['val_loss']

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)

# Epochs will be a number ranging from 1 to the length of epochs we used for training.
# Take the history of one of the attributes, which will contain points on the graph
# for the amount of epochs we used for training.
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, valid_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'r', label='Training Categorical Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Categorical Accuracy')
plt.title('Training and Validation Categorical Accuracy')
plt.legend()
plt.figure()

plt.show()


test_labels = test_batches.classes

matrix_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Make another prediction based on testing batches for report:
predictions = loaded_model.predict_generator(test_batches, steps=len(df_val), verbose=1)

# Plots confusion matrix, set Normalization set to false.
# Taken from SciPy example and Tensorflow docs.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


confusion_matrix = confusion_matrix(test_labels, predictions.argmax(axis=1))

plot_confusion_matrix(confusion_matrix, matrix_labels, title='The Confusion Matrix')

# Get the index of the class with the highest probability score
y_pred = np.argmax(predictions, axis=1)

# Get the labels of the test images.
y_true = test_batches.classes

# Create classification report with matrix labels and target names.
report = classification_report(y_true, y_pred, target_names=matrix_labels)

print("\nClassification Report:")
print(report)