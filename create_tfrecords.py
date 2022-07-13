#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:55:20 2021

@author: John S. Hyatt
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

###############################################################################

"""
CHOOSE DATASET AND FILE PATH
"""

"""
The available datasets:

###############################################################################

MNIST:
    28x28x1 grayscale images of numbers on a uniform black background.
    10 classes: the digits 0-9.
    Images are heavily curated. All digits have very similar same size, intensity range, orientation, stroke thickness, and location in the image.
    There are different numbers of examples per class in both the training and validation datasets:

        Class    Training    Validation
          0        5923         980
          1        6742         1135
          2        5958         1032
          3        6131         1010
          4        5842         982
          5        5421         892
          6        5918         958
          7        6265         1028
          8        5851         974
          9        5949         1009
        total      60000        10000

###############################################################################

Fashion-MNIST (fMNIST):
    28x28x1 grayscale images of clothes on a uniform black background.
    10 classes: T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot.
    Images are still quite curated, although less so than MNIST. All clothing items (especially within one classs) have very similar size, intensity range, orientation, and location in the image. In particular, the orientation of all shoes is to the left, and almost all trousers have the right leg straight and the left slightly bent, for example.
    Some classes have much more variation than others (trousers in particular are almost all identical) and some classes have significant apparent overlap when compared visually (e.g., T-shirt/top and shirt). Even in the very uniform classes, there are numerous examples that are completely outside the norm for that class, such that they probably won't be identified properly by a classifier (e.g., a pair of spotted briefs in the trousers class).
    Fashion-MNIST has equal numbers of examples per class. There are 6000 examples per class in the training set (60000 total) and 1000 examples per class in the validation set (10000 total).

###############################################################################

IMPORTANT: If this dataset will be used to train a super-resolution model, set COMBINED_DATASET = True. This will allow a batch to contain examples from more than one class. If it will be used to train a class-conditioned generative moel, set COMBINED_DATASET = False. This will keep the classes segregated by batch during training.
"""

# One of 'MNIST' or 'fMNIST'
which_dataset = 'fMNIST'

# File path for saving the tfrecords files
path = './'

# Which classes to include in the created TFRecords file(s)?
which_classes = [0,1,2,3,4,5,6,7,8,9]

# Do you want to generate one TFRecords dataset containing all classes in which_classes, or one TFRecords dataset PER CLASS, containing only examples from that class?
COMBINED_DATASET = True

###############################################################################

"""
First, download the dataset, preprocess, and convert to numpy.
"""

# Load training and validation (test) data using built-in TF function.
if which_dataset == 'MNIST':

    ((x_train,
      y_train),
     (x_val,
      y_val)) = tf.keras.datasets.mnist.load_data()

elif which_dataset == 'fMNIST':

    ((x_train,
      y_train),
     (x_val,
      y_val)) = tf.keras.datasets.fashion_mnist.load_data()

# MNIST and fashion-MNIST do not have a depth dimension in their original format, just spatial dimensions.
if which_dataset in ['MNIST', 'fMNIST']:
    x_train = np.expand_dims(x_train,
                             -1)
    x_val = np.expand_dims(x_val,
                           -1)

# Convert from uint8 to float32.
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)

# By default, the intensities are in [0,255]. Rescale to [0,1].
x_train = x_train / 255
x_val = x_val / 255

# Sort the x data by class. This will make it easier to create custsom datasets (e.g., with only classes 4 and 7, for example).
train_sorting_indices = np.argsort(y_train.flatten())
val_sorting_indices = np.argsort(y_val.flatten())
x_train = x_train[train_sorting_indices]
x_val = x_val[val_sorting_indices]
y_train = y_train[train_sorting_indices]
y_val = y_val[val_sorting_indices]

# For MNIST, there are different numbers of examples per class. These need to be explicitly accounted for.
if which_dataset == 'MNIST':
    train_indices = \
        np.array([5923,6742,5958,6131,5842,5421,5918,6265,5851,5949])
    val_indices = np.array([980,1135,1032,1010,982,892,958,1028,974,1009])
# Fashion-MNIST has 6000 training examples and 1000 validation examples per class.
elif which_dataset == 'fMNIST':
    train_indices = np.array([6000] * 10)
    val_indices = np.array([1000] * 10)

# The x data have now been preprocessed for conversion to tfrecords files.
# We are not doing anything else with the class labels (class-conditional datasets will be generated from the sorted tfrecords files' associated label), but for the sake of completeness, here is the conversion:
y_train = tf.keras.utils.to_categorical(y_train,
                                        num_classes=10,
                                        dtype='float32')
y_val = tf.keras.utils.to_categorical(y_val,
                                      num_classes=10,
                                      dtype='float32')

###############################################################################

"""
FUNCTION DEFINITIONS
"""

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float (32 or 64 precision)."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """ Returns an int64_list from bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _convert_img_to_str(img):
    """
    Convert an image in numpy array format, shape (1, height, width, depth), to a string. Depth is 1 for grayscale, 3 for RGB.
    Returns:
        img_string
        height
        width
        depth
    """

    # height, width, and depth are all going to be features in the serialized example (just the same as the class label will be).
    height = img.shape[1]
    width = img.shape[2]
    depth = img.shape[3]

    img_string = img.tostring()

    return img_string, height, width, depth

def _convert_1hot_to_str(one_hot):
    """
    Convert a one-hot class label vector, shape (1, num_classes), to a string.
    Returns:
        one_hot_string
    """
    one_hot_string = one_hot.tostring()
    return one_hot_string

def _convert_to_example(img_string,
                        height,
                        width,
                        depth,
                        one_hot_string):
    """
    Serialize an example with a single image and corresponding one-hot class label.
    This creates a tf.Example message ready to be written to a file.
    Args:
        img_string:  string, image in array of shape (1, height, width, depth).
        height:  integer, image height in pixels.
        width:  integer, image width in pixels.
        depth:  integer, image depth in pixels.
        one_hot_string:  string, one-hot vector that identifies the ground truth label.
    Returns:
        Example proto
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    feature = {'img': _bytes_feature(img_string),
               'height': _int64_feature(height),
               'width': _int64_feature(width),
               'depth': _int64_feature(depth),
               'label': _bytes_feature(one_hot_string)}

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto

def _make_TFRecord(x,
                   y,
                   output_file):
    """
    Processes and saves an array of image data and array of label data as a TFRecord.
    Args:
        x:  numpy array of images.
            Has shape (num_examples, height, width, depth).
        y:  numpy array of one-hot class labels.
            Has shape (num_examples, num_classes)
        output_file:  string, filename for the saved TFRecord file.
    """

    # Open the file writer.  All of the examples will be written here.
    writer = tf.io.TFRecordWriter(output_file)

    for i in range(x.shape[0]):

        # Get the image and corresponding label for one example.
        img = x[i:i+1]
        label = y[i:i+1]

        # Convert the image to a string and obtain the image dimensions.
        (img_string,
         height,
         width,
         depth) = _convert_img_to_str(img)

        # Convert the one-hot class label to a string.
        one_hot_string = _convert_1hot_to_str(label)

        # Put all the features into the example.
        example = _convert_to_example(img_string,
                                      height,
                                      width,
                                      depth,
                                      one_hot_string)

        # Write the example into the TFRecords file.
        writer.write(example.SerializeToString())
        print(f'Created example {i} of {x.shape[0]}.')

    writer.close()
    print(f'Saved TFRecord to {output_file}.')

def _parse_example(serialized_example):
    """
    Takes a single serialized example and converts it back into an image (AS A TENSOR OBJECT) and the corresponding label.
    """

    features = {'img': tf.io.FixedLenFeature([],
                                             tf.string),
                'height': tf.io.FixedLenFeature([],
                                                tf.int64),
                'width': tf.io.FixedLenFeature([],
                                               tf.int64),
                'depth': tf.io.FixedLenFeature([],
                                               tf.int64),
                'label': tf.io.FixedLenFeature([],
                                               tf.string)}

    parsed_example = tf.io.parse_single_example(serialized=serialized_example,
                                                features=features)

    # Get the class label.
    label = parsed_example['label']
    label = tf.io.decode_raw(label,
                             tf.float32)

    # Get the image dimensions.
    height = parsed_example['height']
    width = parsed_example['width']
    depth = parsed_example['depth']

    # Get the raw byte string and convert it to a tensor object.
    img = parsed_example['img']
    img = tf.io.decode_raw(img,
                           tf.float32)
    # The tensor object is 1-dimensional, so reshape it using the image dimensions we obtained earlier.
    img = tf.reshape(img,
                     shape=(height,
                            width,
                            depth))

    return img, label

###############################################################################

def make_TFRecords(which_classes):

    """
    Function that creates training and validation tfrecords files using the above functions.

    Args:
        which_classes: a list of classes to be compiled into a tfrecords file after preprocessing.

    Generates and saves training and validation tfrecords files as well as an x_demo numpy file.
    """

    # Get a string of the classes.
    class_string = ''
    for i in which_classes:
        class_string += f'{i}'

    # Get training and validation examples to be added to the dataset for each class, as well as their total.
    train_indices_2 = train_indices[which_classes]
    val_indices_2 = val_indices[which_classes]
    num_train_examples = np.sum(train_indices_2)
    num_val_examples = np.sum(val_indices_2)

    # Make arrays containing only data for the classes listed in which_classes.
    x_train_2 = np.zeros((num_train_examples,
                          x_train.shape[1],
                          x_train.shape[2],
                          x_train.shape[3]),
                         dtype=np.float32)
    x_val_2 = np.zeros((num_val_examples,
                        x_train.shape[1],
                        x_train.shape[2],
                        x_train.shape[3]),
                       dtype=np.float32)
    y_train_2 = np.zeros((num_train_examples,
                          10),
                         dtype=np.float32)
    y_val_2 = np.zeros((num_val_examples,
                        10),
                       dtype=np.float32)

    for i in range(len(which_classes)):
        j = which_classes[i]

        x_train_2[np.sum(train_indices_2[:i]) : \
                  np.sum(train_indices_2[:i+1])] = \
            x_train[np.sum(train_indices[:j]) : \
                    np.sum(train_indices[:j+1])]

        x_val_2[np.sum(val_indices_2[:i]) : \
                np.sum(val_indices_2[:i+1])] = \
            x_val[np.sum(val_indices[:j]) : \
                  np.sum(val_indices[:j+1])]

        y_train_2[np.sum(train_indices_2[:i]) : \
                  np.sum(train_indices_2[:i+1])] = \
            y_train[np.sum(train_indices[:j]) : \
                    np.sum(train_indices[:j+1])]

        y_val_2[np.sum(val_indices_2[:i]) : \
                np.sum(val_indices_2[:i+1])] = \
            y_val[np.sum(val_indices[:j]) : \
                  np.sum(val_indices[:j+1])]

    _make_TFRecord(x_train_2,
                   y_train_2,
                   path + f'x_train_{which_dataset}_c{class_string}.tfrecords')

    _make_TFRecord(x_val_2,
                   y_val_2,
                   path + f'x_val_{which_dataset}_c{class_string}.tfrecords')

def verify_TFRecords(path,
                     which_classes):

    """
    Function that verifies that the tfrecords files were created properly.

    Args:
        path: location of xy_train.tfrecords and xy_val.tfrecords
        which_classes: a list of classes to be compiled into a tfrecords file after preprocessing.

    Plots several images from the validation set and prints the corresponding labels.
    """

    # Get a string of the classes.
    class_string = ''
    for i in which_classes:
        class_string += f'{i}'

    xy_val = path + f'x_val_{which_dataset}_c{class_string}.tfrecords'

    xy_val = tf.data.TFRecordDataset(xy_val)

    # Check the first 5 examples from the dataset.
    xy_val = xy_val.take(5).map(_parse_example)

    plt.close('all')

    # Each example looks like (img, label)
    for ex in xy_val:
        img = ex[0].numpy()
        label = ex[1].numpy()

        plt.figure()
        plt.imshow(img)
        print(label)

###############################################################################

if COMBINED_DATASET:
    make_TFRecords(which_classes)

else:
    for which_class in which_classes:
        make_TFRecords([which_class])