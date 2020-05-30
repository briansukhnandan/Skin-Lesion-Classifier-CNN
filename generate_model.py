#!/usr/bin/env python3

# Disables AVX2 and FMA warnings on Unix-based systems.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#
#   How did I improve the algorithm?
#       - Included a pre-processing step to make it easier for our model to identify.
#       - Augmented data into multiple categories for more precise training.
#       - Added class weights to make model more sensitive to Melanoma.
#       - Added parameters to ImageDataGenerator to apply random modifications to images.
#

# Print available hardware/software for NVIDIA GPUs.
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('XLA_GPU')))
print("CUDA Version: "+str(tf_build_info.cuda_version_number))
# 9.0 in v1.10.0
print("cuDNN Version: "+str(tf_build_info.cudnn_version_number))

# Declare base_dir which images will be stored in.
base_dir = 'HAM10000'

# create train_dir inside HAM10000/ which will contain images for training.
train_dir = os.path.join(base_dir, 'train_dir')

# val_dir which will hold validation images.
val_dir = os.path.join(base_dir, 'val_dir')

# We will be using .jpg images so no other conversion is needed for other file formats.
image_ext = '*.jpg'

# Read .csv file which contains data and labels for each picture.
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

    # Generate a list of items with unique lesion ids from df array.
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


# Copy image_id column to train_or_val column.
df_data['train_or_val'] = df_data['image_id']


# Apply identify_val_rows to this copied column.
df_data['train_or_val'] = df_data['train_or_val'].apply(identify_val_rows)


# Add data that is part of the training dataset by using applied function and taking result.
df_train = df_data[df_data['train_or_val'] == 'train']


# Set the image_id as the index in df_data
df_data.set_index('image_id', inplace=True)

# Get a list of train and val images
train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

for image in train_list:

    file_name = image+'.jpg'
    label = df_data.loc[image, 'dx']

    # destination path to image
    source = os.path.join('./HAM10000', file_name)

    # copy the image from the source to the destination
    dest = os.path.join(train_dir, label, file_name)

    # shutil.copyfile(source, dest)

for image in val_list:

    # Set all file names as name + jpeg extension.
    file_name = image+'.jpg'
    label = df_data.loc[image, 'dx']

    # destination path to image
    source = os.path.join('./HAM10000', file_name)

    # copy the image from the source to the destination
    dest = os.path.join(val_dir, label, file_name)

    # shutil.copyfile(source, dest)


# Now we must add other images except for nv since there is an abundance of nv.
class_list = ['mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# Create an augmented dataset which will expand the dataset of all image
# classes.
for item in class_list:

    # We are creating temporary directories here because we delete these directories later
    # create a base dir
    aug_dir = 'aug_dir'
    os.mkdir(aug_dir)
    # create a dir within the base dir to store images of the same class
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)

    # Get class from class_list array. Represents respective file in HAM10000 dir.
    img_class = item

    # list all images in that directory
    img_list = os.listdir('HAM10000/train_dir/' + img_class)

    # Copy images from the class train dir to the img_dir e.g. class 'mel'
    for file_name in img_list:

        # Create source directory which is in training directory.
        src = os.path.join('HAM10000/train_dir/' + img_class, file_name)

        # Create destination folder which is in image directory.
        dst = os.path.join(img_dir, file_name)

        # Copy all images from src -> dst folder.
        shutil.copyfile(src, dst)

    # Temporary augumented dataset directory, will be deleted upon training.
    path = aug_dir

    # Set our save path to training directory for all augmented imgs produced.
    save_path = 'HAM10000/train_dir/' + img_class

    # Create Datagen generator.
    # We will be using an ImageDataGenerator because we are processing
    # images instead of other primitive data.
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(

        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'

    )

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(path,
                                              save_to_dir=save_path,
                                              save_format='jpg',
                                              target_size=(299, 299),
                                              batch_size=batch_size
                                              )

    # Generate the augmented images and add them to the training folders
    num_aug_images_wanted = 6000  # total number of images we want to have in each class

    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((num_aug_images_wanted - num_files) / batch_size))

    # run the generator and create about 6000 augmented images
    for i in range(0, num_batches):
        images, labels = next(aug_datagen)

    # delete temporary directory with the raw image files
    shutil.rmtree('aug_dir')

    # End for loop

'''
BUILD MODEL
'''

train_path = './HAM10000/train_dir'
validation_path = './HAM10000/val_dir'

train_samples = len(df_train)
validation_samples = len(df_val)

train_batch_size = 10
validation_batch_size = 10
image_size = 299

# Divide total # of samples by batch size to section off training samples into steps.
# Then round up with np.ceil function.
train_steps = np.ceil(train_samples / train_batch_size)
val_steps = np.ceil(validation_samples / validation_batch_size)

# Create an Image Data Generator to input later into our model.
data_generation = ImageDataGenerator(
    # Use Inception ResNet v2
    preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input
)


### GETTING BATCHES FROM DIRECTORY PATHS ###
print("\nTrain Batches: ")
train_batches = data_generation.flow_from_directory(directory=train_path,
                                                    target_size=(image_size,image_size),
                                                    batch_size=train_batch_size,
                                                    shuffle=True
                                                    )

print("\nValidation Batches: ")
validation_batches = data_generation.flow_from_directory(directory=validation_path,
                                                         target_size=(image_size,image_size),
                                                         batch_size=validation_batch_size
                                                         )

print("\nTest Batches: ")
test_batches = data_generation.flow_from_directory(validation_path,
                                           target_size=(image_size,image_size),
                                           batch_size=1,
                                           shuffle=False
                                           )

### TESTING TRAIN_DATA ###

sample_training_images, _ = next(train_batches)


### ENDING TESTING TRAIN_DATA


# Create Inception Res Net model as used in paper
resnet = tf.keras.applications.inception_resnet_v2.InceptionResNetV2()

# print("\nLayers of ResNet: "+str(len(resnet.layers)))


# Exclude the last 1/10 layers of the model.
x = resnet.layers[-28].output

x = tf.keras.layers.Dropout(0.25)(x)

x = tf.keras.layers.Flatten()(x)

# Make a prediction layer with 7 nodes for the 7 dir in our train_dir.
predictions_layer = tf.keras.layers.Dense(7, activation='softmax')(x)


model = tf.keras.Model(inputs=resnet.input, outputs=predictions_layer)


# Train the last 50 layers of the model only.
for layer in model.layers[:-50]:
    layer.trainable = False


print("\nCompiling model:")

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.categorical_accuracy])

print("Model compilation completed!")


# Add weights to try to make the model more sensitive to melanoma
class_weights = {
    0: 1.0,  # akiec
    1: 1.0,  # bcc
    2: 1.0,  # bkl
    3: 1.0,  # df
    4: 3.0,  # mel - Keep this value higher than the rest.
    5: 1.0,  # nv
    6: 1.0,  # vasc
}


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=2,
                                   verbose=1, mode='max', min_lr=0.00001)


callbacks_list = [reduce_lr]


print("Training model. . .")

history = model.fit(train_batches,
                    steps_per_epoch=train_steps,
                    class_weight=class_weights,
                    validation_data=validation_batches,
                    validation_steps=val_steps,
                    epochs=25,
                    callbacks=callbacks_list
                    )

print("Training completed!\n")

print("Saving model. . .")
model.save('saved_model')
print("Saving completed!\n")

# Save history to use when we load model in .py file.
# Save as .npy file which stores numpy arrays.
print("Saving history to 'my_history.npy'")
np.save('my_history.npy', history.history)
print("History saving completed!\n")

print("Model saved to 'my_history.npy', run load_model.py to see results!")
