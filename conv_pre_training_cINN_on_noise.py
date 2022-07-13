#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:17:57 2021

@author: John S Hyatt
"""

"""
The purpose of this code is to pre-conditioned a model with a given architecture on the prior (standard normal Gaussian noise) only (no signal). The idea is that the pre-conditioned model can then be trained more easily (with instance noise annealing) on actual data, and this pre-conditioning will only have to be done once per architecture.
"""

import tensorflow as tf

import numpy as np

from conv_cINN_make_model import cFlow

from conv_cINN_base_functions import renew_noise

###############################################################################

# How many epochs?
num_epochs = 100

# Hyperparameters of training.
batch_size = 512 # Batch size.
learning_rate = 0.0003 # Adam optimizer learning rate (recommend 0.0001-0.001).
patience = 10 # Number of epochs for early stopping.

# From where to load previous model weights? (If you are continuing a previous pre-condition, either because it had not converged yet or because you wanted to change training hyperparameters).
# If you are starting from scratch, comment the second line so this remains None.
# If you want to continue training a previously-trained model, write the path to the saved model weights on the second line.
LOAD_path = None
# =============================================================================
# LOAD_path = './weights.h5'
# =============================================================================

# Where to save the weights and history during training?
# The entire point of this code is to generate a generically pre-conditioned model, so you need a save path.
SAVE_path = './'

# Saving preconditioning training history.
hist_CHECKPOINT_path = None
hist_CHECKPOINT_path = './'

# Hyperparameters of the model architecture.
# These must be the same as the ones used during training on real data.
squeeze_factor_block_list = [0,1,0,0] # Whether or not a given block squeeze/factors.
ResNeXt_block_list = [3,3,3,3] # Number of ResNeXt blocks per NN.
num_kernels_list = [64,64,32,32]#[64,64,32,32] # Number of kernels per NN layer.
cardinality_list = [8,8,4,4]#[8,8,4,4] # Cardinality of ResNeXt blocks.
kernel_size = 3 # Size of the kernels.
which_dilations = [1,2,4] # Parallel dilations.
LAYER_NORM = True # Whether or not to have layer normalization in the res blocks.
xy_h = 28 # Height of the input.
xy_w = 28 # Width of the input.
xy_d = 2 # TOTAL depth of the input (number of channels).
# Need the depth of the future x-component specifically, as it plays a limited role even in pre-conditioning. It is a required argument when building the model, determining the dimensions used to calculate the log probability.
x_d = 1

# Strings for model architecture labeling/identification.
squeeze_factor_string = ''
ResNeXt_block_string = ''
num_kernels_string = ''
cardinality_string = ''
dil_string = ''

for i in range(len(squeeze_factor_block_list)):
    squeeze_factor_string += f'{squeeze_factor_block_list[i]}'
    ResNeXt_block_string += f'{ResNeXt_block_list[i]}'
    num_kernels_string += f'{num_kernels_list[i]}'
    cardinality_string += f'{cardinality_list[i]}'

for i in range(len(which_dilations)):
    dil_string += f'{which_dilations[i]}'

# The usual initializers will give a randomly initialized model log_prob values WAY outside the max numerical value and will set everything to NaN from the first training step. This initializer avoids that problem so the model can actually get meaningful gradients.
init = tf.keras.initializers.Orthogonal(gain=0.1)

# Callbacks for model.fit()
callbacks = []

# Callback for early stopping during training.
callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                  patience=patience))

# Write a history checkpoint callback:
if hist_CHECKPOINT_path:

    hist_CHECKPOINT_name = f'preconditioning_hist_{xy_h}x{xy_w}x{xy_d}_SqFa{squeeze_factor_string}_NRB{ResNeXt_block_string}_C{cardinality_string}_NK{num_kernels_string}_KS{kernel_size}_D{dil_string}_LN{LAYER_NORM}'

    callbacks.append(tf.keras.callbacks.CSVLogger(hist_CHECKPOINT_path + \
                                                  hist_CHECKPOINT_name,
                                                  separator=',',
                                                  append=True))

###############################################################################

# Generate a dataset of ~10,000 examples of Gaussian noise, N(0,1).
# This noise is NOT set. Every call to the dataset will sample N(0,1) separately.
xy_noise = tf.random.normal(shape=(20*batch_size,
                                   xy_h,
                                   xy_w,
                                   xy_d),
                            mean=0,
                            stddev=1)

xy_noise = tf.data.Dataset.from_tensor_slices(xy_noise)

# Prepare the dataset to be used for training.
xy_noise = xy_noise.cache()
xy_noise = xy_noise.map(lambda xy_element : renew_noise(xy_element))
xy_noise = xy_noise.batch(batch_size)
xy_noise = xy_noise.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

###############################################################################

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

# Compile the model.
model.compile(optimizer=opt)

# CODE FOR LOADING MODEL WEIGHTS
if LOAD_path:
    # Load the weights from the previously trained model.
    model.load_weights(LOAD_path)

# Condition the model on num_epochs worth of noise.
# history is saved by the history checkpointing callback, if desired.
history = model.fit(xy_noise,
                    epochs=num_epochs,
                    verbose=2,
                    callbacks=callbacks)

model.save_weights(SAVE_path + \
                   f'conditioned_weights_{xy_h}x{xy_w}x{xy_d}_SqFa{squeeze_factor_string}_NRB{ResNeXt_block_string}_C{cardinality_string}_NK{num_kernels_string}_KS{kernel_size}_D{dil_string}_LN{LAYER_NORM}.h5')