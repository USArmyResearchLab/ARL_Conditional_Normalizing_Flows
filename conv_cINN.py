#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:17:00 2021

@author: John S Hyatt
"""

import tensorflow as tf

from conv_cINN_base_functions import (_parse_example,
                                      instance_noise,
                                      preprocess_dataset_class,
                                      preprocess_dataset_SR)

from conv_cINN_make_model import cFlow

import numpy as np

###############################################################################

"""
USER-SPECIFIED HYPERPARAMETERS
"""

# What is the model supposed to do?
# If it is for class generation, `model_type` is 'class'.
# If it is for super-resolution (SR), `model_type` is one of 'SR4,2' or 'SR2,1', where the first number is the amount of downsampling from the original high-resolution image to y (either 4x4 or 2x2), and the second number is the amount of downsampling from the original to x (2x2 or 1x1=no downsampling).
# For multi-level SR, each model is trained independently on the ground truth. It is not desirable to train one model to go from 4x4 to 2x2 reconstructions, and then to feed those reconstructions into a second model, since the reconstructions are not ground truth and may not be consistent with the higher-resolution original, which we have access to during training and so might as well use.
model_type = 'SR2,1' # 'class', 'SR4,2', or 'SR2,1'

# Which image dataset: 'MNIST' or 'fMNIST'?
which_dataset = 'fMNIST'

# Which of the image classes 0 through 9 to train on?
# Note: because the class conditions are NOT one-hot vectors or an orthogonal image, it's not a great idea to do more than 2 classes for class-conditioned image generation with this code.
data_classes = [0,1,2,3,4,5,6,7,8,9]

# Remove the commas in resolutions for labeling saved data later.
type_string = model_type.replace(',',
                                 '')

# Should x be a residual? Or the original image?
# Having a residual is nice for several reasons, one of which is that it gives us a way to sanity check the rsults for the super-resolution case: for 'SR4,2' and 'SR2,1', the 2x2 pixel blocks in the residual corresponding to pixels in the low-res image should sum to 0.
RESIDUAL = True

# Are you using ~logit(x) instead of x for the image?
# ONLY IMPLEMENTED FOR THE CLASS (DISCRETE) CASE.
DISCRETE_LOGITS = True

# If the spatial dimensions will NOT be squeeze/factored AFTER a block, the corresponding entry in the list is 0.
# If they WILL be, the corresponding entry is 1.
# For example, [0,1,0,0] corresponds to two blocks at the original input size, and two that have been halved spatially in both dimensions, while doubled in channel depth.
# For 28x28x2 inputs, that means two 28x28x2 blocks, and two 14x14x4 blocks.
# For 'SR4,2', due to the small dimensions of the inputs, you can only use [0,0,0,0].
squeeze_factor_block_list = [0,1,0,0]

# Number of ResNeXt blocks in A and b in each coupling block.
ResNeXt_block_list = [3,3,3,3]

# The number of kernels decreases roughly as O(N), where N is the number of dimensions in the input.
num_kernels_list = [64,64,32,32]

# The ratio of number of kernels / cardinality remains constant throughout the model.
cardinality_list = [8,8,4,4]

# Size of the square convolutional kernels.
kernel_size = 3

squeeze_factor_string = ''
ResNeXt_block_string = ''
num_kernels_string = ''
cardinality_string = ''

for i in range(len(squeeze_factor_block_list)):
    squeeze_factor_string += f'{squeeze_factor_block_list[i]}'
    ResNeXt_block_string += f'{ResNeXt_block_list[i]}'
    num_kernels_string += f'{num_kernels_list[i]}'
    cardinality_string += f'{cardinality_list[i]}'

# Dilation factors are determined automatically by the code based on the size of your compressed input. A 14x14 input will have dilations [1,2,4], and a 28x28 input will have dilations [1,2,4,8], with 3x3 kernels. See the documentation for cFlow() for details.
# One parallel branch of the model is built for each dilation, and each one's convolutional layers are dilated by this factor. Each square defined by adjacent pixels in the Nth dilation kernel is the same spatial size as the entire kernel in the (N-1)the dilation. See the documentation for dilated_residual_block() for details.
# NOTE: each parallel branch has its number of (non-1x1) kernels divided by the dilation factor. This reduces the number of parameters by ~30% for [1,2,4]. Basicallly, the model spends progressively less of its parameter budget on long-range correlations.
# `num_kernels` / dilation must still be evenly divisible by cardinality.
which_dilations = [1,2,4]

# Layer normalization is better than batch normalization. Implementing it requires some tricks but those are all in the attached code.
LAYER_NORM = True # Whether or not to have layer normalization in the res blocks.

# The usual initializers will give a randomly initialized model log_prob values WAY outside the max numerical value and will set everything to NaN from the first training step. This initializer avoids that problem so the model can actually get meaningful gradients for the first few batches of training data.
init = tf.keras.initializers.Orthogonal(gain=0.1)

# Training hyperparameters.
batch_size = 32
learning_rate = 0.0003
patience = 20 # stop training if val loss doesn't improve after this many epochs.

# If you do not want to save model checkpoints, leave this as None.
model_CHECKPOINT_path = None
# =============================================================================
# model_CHECKPOINT_path = './'
# =============================================================================

# If you do not want to save model history until the end of training (not recommended!), leave this as None.
# Note that the history callback WILL append new training/validation results to the old one if you start training again! That means if you finish training a model, then start training another from scratch, if you don't change the checkpoint path or filename, it will just append to the old history.
hist_CHECKPOINT_path = None
hist_CHECKPOINT_path = './'

# How many epochs to train for between checkpoints?
checkpoint_epochs = 10

# Do you want to do instance noise for the first N epochs?
# If yes, set N below
# If no, leave it as None
num_annealing_epochs = None
num_annealing_epochs = 100

# How many epochs to train on clean data?
num_epochs = 500

# Do you want to train the model or just build it?
# Set this as False if you are loading a pre-trained model.
TRAIN = True

# Where to save the weights and history during training?
# If you don't want to save, comment the second line so this remains None.
# If you do want to save, write the path to save everything on the second line.
SAVE_path = None
# =============================================================================
# SAVE_path = './'
# =============================================================================

# Where to look for the training data TFRecords file.
dataset_path = './'

# Callbacks for model.fit()
callbacks = []

# Callback for early stopping during training.
callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=patience))

###############################################################################

"""
DERIVED HYPERPARAMETERS
"""

# Class conditional generation is a discrete condition problem.
# Super-resolution is a continuous condition problem.
if model_type == 'class':
    discrete_or_continuous = 'DISCRETE'
else:
    discrete_or_continuous = 'CONTINUOUS'

# Need the data classes as a string to identify the right dataset.
data_classes_string = ''
for data_class in data_classes:
    data_classes_string += f'{data_class}'

# Need to convert these to EVENLY SPACED data labels.
data_labels = []
for i in range(len(data_classes)):
    data_labels.append(i)

# Need to standardize those labels.
data_labels = (data_labels - np.mean(data_labels)) / np.std(data_labels)

# Hyperparameters of the model architecture.
# For model_type 'SR_4,2', the actual size is 1/4 of the original (2-fold downsampling in each dimension).
if model_type=='SR4,2':
    if which_dataset in ['MNIST', 'fMNIST']:
        xy_h = 14 # Height of the input.
        xy_w = 14 # Width of the input.
else:
    if which_dataset in ['MNIST', 'fMNIST']:
        xy_h = 28
        xy_w = 28

if which_dataset in ['MNIST', 'fMNIST']:
    x_d = 1 # Depth of the x input (number of channels).

if model_type in ['SR4,2', 'SR2,1']:
    y_d = x_d # Depth of the y input.
elif model_type in ['class']:
    y_d = 1 # just one class label

xy_d = x_d + y_d # Depth of the total xy input.

# String version of dilations for model identification
if not which_dilations:
    dil_string = '1'
else:
    dil_string = ''
    for i in range(len(which_dilations)):
        dil_string += f'{which_dilations[i]}'

# From where to load previous model weights?
# Usually these will either be just preconditioned on noise using conv_pre_training_cINN_on_noiose.py, or have been partially trained using this same script.
# If you are starting from scratch, comment the second line so this remains None.
# If you want to continue training a previously-trained model, write the path to the saved model weights on the second line.
LOAD_path = None
# =============================================================================
# LOAD_path = './weights.h5'
# =============================================================================

###############################################################################

"""
LOAD AND PREPARE THE DATA
"""

# Dataset generation for discrete conditional information (e.g., class conditional) is handled differently than continuous conditional information (e.g., super-resolution), because batches must be segregated by label in the discrete case.
if discrete_or_continuous == 'DISCRETE':

    ###########################################################################
    # Prepare the training data. ##############################################
    ###########################################################################

    # Convert the classes to EVENLY SPACED labels (for example, if the chosen classes are [1,2,5], the labels need to be converted to [0,1,2]).
    data_labels = []
    for i in range(len(data_classes)):
        data_labels.append(i)
    data_labels = np.array(data_labels)

    # And then scale them to the same order as x.
    # Scale to the interval [0,1].
    data_labels = data_labels / data_labels[-1]

    # We will load one class at a time to keep them batch-segregated. Start with the first class.
    data_class = data_classes[0]
    data_label = data_labels[0]

    # Load and parse the data.
    # You need a sub-dataset for each class individually (i.e., the dataset of only class 1, plus the dataset of only class 2, etc.) rather than a single dataset for all classes. This means you will have to create a tfrecords for each individual class.
    x_train = dataset_path + f'x_train_{which_dataset}_c{data_class}.tfrecords'
    x_train = tf.data.TFRecordDataset(x_train)
    x_train = x_train.map(_parse_example,
                          num_parallel_calls=tf.data.AUTOTUNE)

    # For class-conditional generation, we DO need the class label, but we also have to segregate by batch. There's no good way to do that if all the classes are in one big dataset from the beginning, so it's easier to just separate them and add in the class label independently.
    # As a result, we discard the class label that comes with the data.
    x_train = x_train.map(lambda image, label : image,
                          num_parallel_calls=tf.data.AUTOTUNE)

    x_train = preprocess_dataset_class(x_train,
                                       DISCRETE_LOGITS,
                                       a=0.01)

    # Create a y' tensor with the same SPATIAL shape as an x example, but only one depth dimension. All elements in the y' tensor will have the same value, namely that of the label.
    for x in x_train.take(1):
        x_shape = tf.shape(x)

    # Need to set the depth dimension to 1.
    y_shape = [x_shape[0],
               x_shape[1],
               1]

    y_element = tf.ones((y_shape),
                        dtype=tf.float32)
    y_element *= data_label

    # Combine the x and y to get xy.
    xy_train = x_train.map(lambda x_element : \
                           tf.concat([x_element,
                                      y_element],
                                     axis=-1),
                           num_parallel_calls=tf.data.AUTOTUNE)

    # This is CONDITIONAL, so we want to segregate the classes.
    # We have to batch AFTER concatenating, because .concatenate() adds an extra dimension otherwise. However, this also means that, for datasets like MNIST that have a different number of examples in each class there might be up to [number of classes] - 1 batches that mix examples from two classes.
    # In order to avoid this, cut the extra examples by batching (dropping the small last batch) and then unbatching prior to concatenating, then batching again at the end.
    # Even for datasets with the same number of examples in each class, there might be mixed batches if the number of examples per class is not an even multiple of the batch size.
    xy_train = xy_train.batch(batch_size,
                              drop_remainder=True,
                              num_parallel_calls=tf.data.AUTOTUNE)
    xy_train = xy_train.unbatch()

    # Repeat for the remaining classes and concatenate them into the training dataset one at a time. This is inefficient, but we only have to do it once.
    for i in range(len(data_classes) - 1):
        data_class = data_classes[i + 1]
        data_label = data_labels[i + 1]
        x_i = dataset_path + f'x_train_{which_dataset}_c{data_class}.tfrecords'
        x_i = tf.data.TFRecordDataset(x_i)
        x_i = x_i.map(_parse_example,
                      num_parallel_calls=tf.data.AUTOTUNE)
        x_i = x_i.map(lambda image, label : image,
                      num_parallel_calls=tf.data.AUTOTUNE)
        x_i = preprocess_dataset_class(x_i,
                             DISCRETE_LOGITS,
                             a=0.01)
        y_element = tf.ones((y_shape),
                            dtype=tf.float32)
        y_element *= data_label
        xy_i = x_i.map(lambda x_element : \
                       tf.concat([x_element,
                                  y_element],
                                 axis=-1),
                       num_parallel_calls=tf.data.AUTOTUNE)
        xy_i = xy_i.batch(batch_size,
                          drop_remainder=True,
                          num_parallel_calls=tf.data.AUTOTUNE)
        xy_i = xy_i.unbatch()
        xy_train = xy_train.concatenate(xy_i)

    # Cache, since everything after this has stochasticity associated to it.
    xy_train = xy_train.cache()

    # Add a SMALL amount of noise to xy. For MNIST, for example, something like 80% or more of pixels are always black, making a bijective map theoretically impossible as soon as the last instance noise annealing epoch finishes. By adding in a very small baseline of noise after caching, we can avoid this problem without introducing new information into the dataset.
    # This also dequantizes the intensity.
    # This is not necessary for datasets like cifar10 without fixed-value pixels!
    xy_train = xy_train.map(lambda xy_element : \
                            instance_noise(xy_element,
                                           0.98), # 2% noise
                            num_parallel_calls=tf.data.AUTOTUNE)

    # Re-batch everything. This is a little inefficient as ideally the dataset would be cached after batching but before shuffling, but then the 2% noise would be the same every time and since it doesn't represent any actual information, it makes more sense for it to be random every time an element is called. The overhead from batching after the cache should be fairly small.
    # Because each class the dataset has already been truncated to a multiple of batch_size, we don't need to set drop_remainder=True this time.
    xy_train = xy_train.batch(batch_size,
                              num_parallel_calls=tf.data.AUTOTUNE)

    # The number of segregated batches in the dataset (for shuffling and checkpointing purposes) is:
    num_train_batches = 0
    for i in xy_train:
        num_train_batches += 1

    # Shuffle and prefetch.
    xy_train = xy_train.shuffle(num_train_batches)
    xy_train = xy_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ###########################################################################
    # Repeat the above process for the validation data. #######################
    ###########################################################################
    data_class = data_classes[0]
    data_label = data_labels[0]

    # Load and parse the data.
    x_val = dataset_path + f'x_val_{which_dataset}_c{data_class}.tfrecords'
    x_val = tf.data.TFRecordDataset(x_val)
    x_val = x_val.map(_parse_example,
                      num_parallel_calls=tf.data.AUTOTUNE)

    # For class-conditional generation, we DO need the class label, but we also have to segregate by batch. There's no good way to do that if all the classes are in one big dataset from the beginning, so it's easier to just separate them and add in the class label independently.
    # As a result, we discard the class label that comes with the data.
    x_val = x_val.map(lambda image, label : image,
                      num_parallel_calls=tf.data.AUTOTUNE)
    x_val = preprocess_dataset_class(x_val,
                                     DISCRETE_LOGITS,
                                     a=0.01)

    # Create a y' tensor with the same shape as an x example. All elements in the y' tensor will have the same value, namely that of the label.
    # We already know y_shape, since training and validation examples have the same shape.

    y_element = tf.ones((y_shape),
                        dtype=tf.float32)
    y_element *= data_label

    # Combine the x and y to get xy.
    xy_val = x_val.map(lambda x_element : \
                       tf.concat([x_element,
                                  y_element],
                                 axis=-1),
                       num_parallel_calls=tf.data.AUTOTUNE)

    # For xy_val, we can skip segregating the classes. This data isn't used to train anyway, so it can't influence the model to split up Z space.
    # This means we don't have to drop the small last batch in each class.

    # Repeat for the remaining classes and concatenate them into the training dataset one at a time.
    for i in range(len(data_classes) - 1):
        data_class = data_classes[i + 1]
        data_label = data_labels[i + 1]
        x_i = dataset_path + f'x_val_{which_dataset}_c{data_class}.tfrecords'
        x_i = tf.data.TFRecordDataset(x_i)
        x_i = x_i.map(_parse_example,
                      num_parallel_calls=tf.data.AUTOTUNE)
        x_i = x_i.map(lambda image, label : image,
                      num_parallel_calls=tf.data.AUTOTUNE)
        x_i = preprocess_dataset_class(x_i,
                                       DISCRETE_LOGITS,
                                       a=0.01)
        y_element = tf.ones((y_shape),
                            dtype=tf.float32)
        y_element *= data_label
        xy_i = x_i.map(lambda x_element : \
                       tf.concat([x_element,
                                  y_element],
                                 axis=-1),
                       num_parallel_calls=tf.data.AUTOTUNE)
        xy_val = xy_val.concatenate(xy_i)

    # We can add the noise before caching this time, since, again, this data is not used to train the model and so the fact that the noise is the same every time doesn't matter. For the same reason, we don't need to worry about shuffling differently every epoch. This may give us a small efficiency gain for some datasets.
    # This also dequantizes the intensity.
    xy_val = xy_val.map(lambda xy_element : \
                        instance_noise(xy_element,
                                       0.98), # 2% noise
                        num_parallel_calls=tf.data.AUTOTUNE)

    # However, it still makes sense to shuffle once BEFORE batching. This desegregates the classes and may give us more meaningful validation results. The shuffle will be the same every time xy_val is called, because we are caching afterwards.
    num_val_batches = 0
    for i in xy_val:
        num_val_batches += 1

    # Shuffle, batch, cache, prefetch.
    xy_val = xy_val.shuffle(num_val_batches)
    xy_val = xy_val.batch(batch_size,
                          num_parallel_calls=tf.data.AUTOTUNE)

    xy_val = xy_val.cache()
    xy_val = xy_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# if discrete_or_continuous = 'CONTINUOUS':
else:

    # Load the training dataset and prepare it for training.
    # Load all the included classes together. This means you will need to create a single tfrecords that contains all classes in the desired dataset.
    x_train = dataset_path + \
              f'x_train_{which_dataset}_c{data_classes_string}.tfrecords'

    # Load and parse the x dataset.
    x_train = tf.data.TFRecordDataset(x_train)
    x_train = x_train.map(_parse_example,
                          num_parallel_calls=tf.data.AUTOTUNE)

    # The dataset has elements in a tuple (image, class label). For SR, we do not need the class label.
    x_train = x_train.map(lambda image, label : image,
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Convert to the appropriate xy dataset.
    xy_train = preprocess_dataset_SR(x_train,
                                     model_type,
                                     RESIDUAL)

    # Because this is a continuous problem, we don't worry about segregating anything even if we mix classes. (That's probably actually a benefit in this case.)
    # We do this before adding noise as otherwise the 2% noise would be the same every time. Since it doesn't represent any actual information, it makes more sense for it to be random every time an element is called.
    xy_train = xy_train.cache()

    # Add a SMALL amount of noise to xy. For MNIST, for example, something like 80% or more of pixels are always black, making a bijective map theoretically impossible as soon as the last instance noise annealing epoch finishes. By adding in a very small baseline of noise after caching, we can avoid this problem without introducing new information into the dataset.
    # This is not necessary for datasets like cifar10 without fixed-value pixels!
    xy_train = xy_train.map(lambda xy_element : \
                            instance_noise(xy_element,
                                           0.98),
                            num_parallel_calls=tf.data.AUTOTUNE) # 2% noise

    # The number of examples in the dataset (for shuffling purposes) is:
    num_train_examples = 0
    for ex in xy_train:
        num_train_examples += 1

    # Shuffle.
    xy_train = xy_train.shuffle(num_train_examples)

    # Batch. The overhead from batching after the cache should be fairly small.
    xy_train = xy_train.batch(batch_size,
                              num_parallel_calls=tf.data.AUTOTUNE)

    # The number of batches in the dataset (for checkpointing purposes) is:
    num_train_batches = 0
    for i in xy_train:
        num_train_batches += 1

    xy_train = xy_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ######################################################
    # REPEAT THE ABOVE PROCESS FOR THE VALIDATION DATASET.
    ######################################################

    # Load and prepare the validation dataset.
    # For now, we load all the included classes together.
    x_val = dataset_path + \
              f'x_val_{which_dataset}_c{data_classes_string}.tfrecords'

    # Load and parse the x dataset.
    x_val = tf.data.TFRecordDataset(x_val)
    x_val = x_val.map(_parse_example,
                      num_parallel_calls=tf.data.AUTOTUNE)

    # The dataset has elements in a tuple (image, class label). For SR, we do not need the class label.
    x_val = x_val.map(lambda image, label : image,
                      num_parallel_calls=tf.data.AUTOTUNE)

    # Convert to the appropriate xy dataset.
    xy_val = preprocess_dataset_SR(x_val,
                                   model_type,
                                   RESIDUAL)

    # Because this is a continuous problem we don't worry about segregating anything even if we mix classes. (That's probably actually a benefit in this case.)
    # We do this before adding noise as otherwise the 2% noise would be the same every time. Since it doesn't represent any actual information, it makes more sense for it to be random every time an element is called.
    xy_val = xy_val.cache()

    # Add a SMALL amount of noise to xy. For MNIST, for example, something like 80% or more of pixels are always black, making a bijective map theoretically impossible as soon as the last instance noise annealing epoch finishes. By adding in a very small baseline of noise after caching, we can avoid this problem without introducing new information into the dataset.
    # This is not necessary for datasets like cifar10 without fixed-value pixels!
    xy_val = xy_val.map(lambda xy_element : \
                        instance_noise(xy_element,
                                       0.98), # 2% noise
                        num_parallel_calls=tf.data.AUTOTUNE)

    # The number of examples in the dataset (for shuffling purposes) is:
    num_val_examples = 0
    for ex in xy_val:
        num_val_examples += 1

    # Shuffle.
    xy_val = xy_val.shuffle(num_val_examples)

    # Batch. The overhead from batching after the cache should be fairly small.
    xy_val = xy_val.batch(batch_size,
                          num_parallel_calls=tf.data.AUTOTUNE)
    xy_val = xy_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

###############################################################################

"""
BUILD/RECONSTRUCT THE MODEL
"""

# Now that we have the number of training batches, we can write the model checkpoint callback:
if model_CHECKPOINT_path:

    model_CHECKPOINT_name = f'checkpoint_{type_string}_{which_dataset}_{xy_h}x{xy_w}x{xy_d}_SqFa{squeeze_factor_string}_NRB{ResNeXt_block_string}_C{cardinality_string}_NK{num_kernels_string}_KS{kernel_size}_D{dil_string}_LN{LAYER_NORM}'
    model_CHECKPOINT_name += '.e{epoch:02d}.hdf5'

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                        model_CHECKPOINT_path + \
                        model_CHECKPOINT_name,
                        save_weights_only=True,
                        save_freq=checkpoint_epochs*num_train_batches))

# Write a history checkpoint callback:
if hist_CHECKPOINT_path:

    hist_CHECKPOINT_name = f'tra_val_hist_{type_string}_{which_dataset}_{xy_h}x{xy_w}x{xy_d}_SqFa{squeeze_factor_string}_NRB{ResNeXt_block_string}_C{cardinality_string}_NK{num_kernels_string}_KS{kernel_size}_D{dil_string}_LN{LAYER_NORM}'

    callbacks.append(tf.keras.callbacks.CSVLogger(hist_CHECKPOINT_path + \
                                                  hist_CHECKPOINT_name,
                                                  separator=',',
                                                  append=True))

# This will create a .csv file containing the training/validation history, appending the most recent epoch's results at the end of the epoch.
# Note: the loss is not tracked in the way you would expect; the columns don't correspond to the order defined in cFlow(). Check the headers in the .csv file to see what the true order is before printing or plotting.
# You can load the csv file with something like the following code:
# =============================================================================
# import csv
#
# a = []
#
# with open('filename') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         a.append(row)
# headers = a[0] # This tells you the true order
# a = a[1:] # Everything besides the headers
# a = np.array(a).astype(np.float32)
# =============================================================================

# Build a randomly-initialized model.
model = cFlow(io_shape=[xy_h,
                        xy_w,
                        xy_d],
              x_d=x_d,
              squeeze_factor_block_list=squeeze_factor_block_list,
              ResNeXt_block_list=ResNeXt_block_list,
              num_kernels_list=num_kernels_list,
              cardinality_list=cardinality_list)

# Set the optimizer and learning rate.
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=opt)

# CODE FOR LOADING MODEL WEIGHTS
if LOAD_path:

    # Can't load the weights til the model has built. Call the model once first.
    for ex in xy_val.take(1):
        model(ex)

    # Load the weights from the previously trained model.
    model.load_weights(LOAD_path)

# CODE FOR TRAINING THE MODEL
# The model will keep track of epochs in both the annealing and clean training stages, and save checkpoints and history for both together. You will have to keep track of the number of annealing epochs if you want to differentiate between the two regimes.
if TRAIN:

    # If callbacks is an empty list, replace it with None.
    if callbacks == []:
        callbacks = None

    # Initial annealing of the training data (from noise to clean) runs on a for loop, updating alpha each epoch and then fitting one epoch. This is not very efficient, but the overhead should be a relatively small contributor to the total training time.
    # Since we are calling fit in a loop we need to tell it what epoch we are on in order for tensorboard to log data correctly.
    completed_epochs = 0

    if num_annealing_epochs:

        for i in range(num_annealing_epochs):

            alpha = i / num_annealing_epochs

            print(f'Annealing instance noise, alpha={alpha}, annealing epoch {i} of {num_annealing_epochs}.')

            xy_train_noisy = xy_train.unbatch()
            xy_train_noisy = xy_train_noisy.map(lambda xy_element : \
                                                instance_noise(xy_element,
                                                               alpha),
                                            num_parallel_calls=tf.data.AUTOTUNE)
            xy_train_noisy = xy_train_noisy.batch(batch_size,
                                            num_parallel_calls=tf.data.AUTOTUNE)

            xy_val_noisy = xy_val.unbatch()
            xy_val_noisy = xy_val_noisy.map(lambda xy_element : \
                                            instance_noise(xy_element,
                                                           alpha),
                                            num_parallel_calls=tf.data.AUTOTUNE)
            xy_val_noisy = xy_val_noisy.batch(batch_size,
                                            num_parallel_calls=tf.data.AUTOTUNE)

            history = model.fit(xy_train_noisy,
                                epochs=completed_epochs+1,
                                initial_epoch=completed_epochs,
                                verbose=2,
                                validation_data=xy_val_noisy,
                                callbacks=callbacks)

            completed_epochs += 1

        # Remove noisy data to save memory.
        del xy_train_noisy
        del xy_val_noisy

    # Then continue with training on the clean data.
    history = model.fit(xy_train,
                        epochs=num_epochs,
                        initial_epoch=completed_epochs,
                        verbose=2,
                        validation_data=xy_val,
                        callbacks=callbacks)

    # CODE FOR SAVING MODEL HISTORY AND WEIGHTS AFTER ALL TRAINING IS DONE
    if SAVE_path:

        model.save_weights(SAVE_path + f'weights_{type_string}_{which_dataset}_{xy_h}x{xy_w}x{xy_d}_SqFa{squeeze_factor_string}_NRB{ResNeXt_block_string}_C{cardinality_string}_NK{num_kernels_string}_KS{kernel_size}_D{dil_string}_LN{LAYER_NORM}.h5')