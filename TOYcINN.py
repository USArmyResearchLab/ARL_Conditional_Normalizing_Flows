#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:13:14 2020

@author: John S Hyatt
"""

import tensorflow as tf

import numpy as np

from TOYcINN_make_datasets import (make_moons_dataset,
                                   make_mixed_dataset,
                                   make_continuous_sectors,
                                   make_discrete_sectors,
                                   instance_noise)

from TOYcINN_make_model import cINN_affine

###############################################################################

"""
Some parts of this code were taken from https://keras.io/examples/generative/real_nvp/, and others from the similar (unofficial) implementation at https://github.com/taesungp/real-nvp/blob/master/real_nvp/nn.py.

Note: these toy models do not incorporate most of the efficiency changes and improvements in the convolutional code used to do super-resolution and class-conditioned image generation. However, they are MUCH more computationally cheap, so this doesn't make much difference, especially as they are intended to demonstrate concepts more than serve any application-based purpose.
    In a similar vein, the script contains code to plot outputs.
"""

###############################################################################

"""
USER-SPECIFIED HYPERPARAMETERS
"""

# Which toy dataset: 'crescents', 'continuous_sectors', or 'mixed'?
which_dataset = 'crescents'
# =============================================================================
# which_dataset = 'continuous_sectors'
# =============================================================================
# =============================================================================
# which_dataset = 'mixed'
# =============================================================================

# For the 'continuous_sectors' dataset, need to set the width (in radians) of the sectors centered on y=theta.
if which_dataset == 'continuous_sectors':
    sector_width = 1

    # Need a single string to include all relevant data info.
    dataset_string = f'contsec_width{sector_width}'

# For the 'mixed' dataset, need to choose which classes to incorporate.
if which_dataset == 'mixed':
    which_classes = [
                     0, # a circle
                     1, # a slash
                     2, # two distinct Gaussian blobs
                     3, # a curve shaped like a sideways 3
                     4, # a uniform square
                     5, # a 3x3 grid of points
                     6  # two fuzzy (but non-overlapping) concentric circles
                     ]

    # Need a single string to include all relevant data info.
    dataset_string = 'mixed_c'

    for which_class in which_classes:
        dataset_string += f'{which_class}'

# For the 'crescents' dataset, should the two crescents be overlapping or separated?
# If True, the crescents will be shifted to create a small region of overlap.
# Also need to set the noisy width of the crescents (higher noise = fatter crescents).
if which_dataset == 'crescents':
    overlapping = False
    noise = 0.05

    # Need a single string to include all relevant data info.
    dataset_string = f'crescent_{overlapping}_{noise}'

    dataset_string = dataset_string.replace('.',
                                            'p')

# All toy problems have 3 coordinates: (x1, x2, y) and (z1, z2, y). This makes these toy problems as simple as possible.
io_shape = 3

# The two Cartesian coordinates are x1 and x2; the class or continuous angle is y. Therefore the "depth" of the x elements is 2.
x_d = 2

# IMPORTANT! num_coupling_blocks MUST be divisible by 6 for 3-dimensional XY.
# In convolutional cINNs, we use 4 masks: 2 checkerboard and 2 channel-wise.
# In densenet cINNs, we just stack all 6 possible combinations of masks.
# For non-3-dimensional XY, this must be changed manually (and so must the mask code).
num_coupling_blocks = 4 # number of complete mask sets in the model.
intermediate_dims = 32 # Number of nodes per intermediate layer in each A and b.
num_layers = 6 # Number of layers in each A and b.

# There are 6 unique arrangements of masks for 3D inputs.
num_coupling_layers = 6 * num_coupling_blocks # Number of coupling layers in the model.

# This initializer keeps log_prob on a randomly initialized model from blowing up so the model can actually get meaningful gradients.
init = tf.keras.initializers.Orthogonal(gain=0.1)

# Training hyperparameters.
batch_size = 1000
learning_rate = 0.0001

# Total number of examples in the dataset.
if which_dataset == 'mixed' \
or which_dataset == 'crescents':
    num_batches_per_class = 20 # 2 classes for crescents, up to 6 for mixed.
elif which_dataset == 'continuous_sectors':
    num_points = batch_size * 20

# Callbacks for model.fit()
# During annealing, we don't use early stopping (and omit it to keep from slowing training down and using up extra memory due to restore_best_weights=True).
callbacks = []
ann_callbacks = []
patience = 10 # stop training if loss (NOT val loss - there is no validation data for these toy models since they can be sampled directly) doesn't improve after this many epochs.
callback_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                           patience=patience,
                                                           restore_best_weights=True)
callbacks.append(callback_early_stopping)

# Do you want to do instance noise for the first N epochs?
# If yes, set the number below
# If no, leave it as None
# My experience is that the best procedure is to anneal out instance noise, then train on clean data.
num_annealing_epochs = None
num_annealing_epochs = 10

# How many epochs to train for (after annealing is complete)?
num_epochs = 1000

# Do you want to train the model or just build it?
TRAIN = True

# Where to save the weights and history during training?
# If you don't want to save, comment the second line so this remains None.
# If you do want to save, write the path to save everything on the second line.
SAVE_path = None
# =============================================================================
# SAVE_path = './'
# =============================================================================

# From where to load previous model weights?
# If you are starting from scratch, comment the second line so this remains None.
# If you want to continue training a previously-trained model, write the path to the saved model weights on the second line.
LOAD_path = None
# =============================================================================
# LOAD_path = './'
# =============================================================================

if LOAD_path:
    LOAD_weights = LOAD_path + f'weights_{dataset_string}_NCL{num_coupling_layers}_ID{intermediate_dims}_NL{num_layers}.npy'
    LOAD_mask = LOAD_path + f'mask_indices_{dataset_string}_NCL{num_coupling_layers}_ID{intermediate_dims}_NL{num_layers}.npy'

# Do you want to plot the results?
PLOT = True

# For continuous sectors, choose the specific discrete sectors to plot after training.
if which_dataset == 'continuous_sectors':
    which_discrete_sectors = [0,
                              np.pi * (1/2),
                              np.pi,
                              np.pi * (3/2)]

# Number of points in the plots.
# For the mixed and crescents datasets, this is the number of points per class.
# We don't want to plot way too few or too many points.
num_plot_points = 10**4

# This is None if you want to let the masks be ordered (semi-)randomly, and has a defined value if you want to reproduce a particular model.
# NOTE: this must have length equal to num_coupling_layers.
# NOTE/INSTRUCTIONS: If you determine the mask order randomly during training, you will have to fix the mask order here when loading the model, because otherwise it will generate random (probably different) masks and the model won't be valid. The code generates an array of indices when saving, so you can just load that here instead of specifying them manually.
mask_indices = None
# Example:
# =============================================================================
# mask_indices = [2, 4, 0, 3, 1, 5]
# =============================================================================
if LOAD_path:
    mask_indices = np.load(LOAD_mask)
    mask_indices = list(mask_indices)

###############################################################################

"""
MAKE THE DATASET
"""

if which_dataset == 'mixed':
    xy = make_mixed_dataset(which_classes,
                            num_batches_per_class,
                            batch_size)

elif which_dataset == 'continuous_sectors':
    xy = make_continuous_sectors(num_points,
                                 batch_size,
                                 sector_width)

elif which_dataset == 'crescents':
    xy = make_moons_dataset(num_batches_per_class,
                            batch_size,
                            noise=noise,
                            overlapping=overlapping)

###############################################################################

"""
BUILD/RECONSTRUCT THE MODEL
"""

# Build a randomly-initialized model.
model = cINN_affine(io_shape,
                    x_d,
                    num_coupling_layers,
                    intermediate_dims,
                    num_layers,
                    init,
                    mask_indices)

# Set the optimizer and learning rate.
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

# CODE FOR LOADING MODEL WEIGHTS
if LOAD_path:
    # Load the weights from the previously trained model.
    array_for_loading_weights = np.load(LOAD_weights,
                                        allow_pickle=True)

    # Copy those weights into the compiled (but randomly initialized) model.
    for i in range(array_for_loading_weights.shape[0]):
        model.coupling_layers_list[i].set_weights(array_for_loading_weights[i])

# CODE FOR TRAINING THE MODEL
if TRAIN:

    # If callbacks is an empty list, replace it with None.
    if callbacks == []:
        callbacks = None
    if ann_callbacks == []:
        ann_callbacks = None

    # Since we are calling fit in a loop, it won't know what epoch it is in unless we tell it.
    completed_epochs = 0

    if num_annealing_epochs:

        # Need to save the training history during annealing.
        # (There is no validation data here; the training data is synthetic and regenerated every epoch.)
        train_history_annealing = []

        for i in range(num_annealing_epochs):

            alpha = i / num_annealing_epochs

            print(f'Annealing instance noise, alpha={alpha}, annealing epoch {i} of {num_annealing_epochs}.')

            xy_noisy = \
                xy.unbatch().map(lambda xy_element : \
                                 instance_noise(xy_element,
                                                alpha)).batch(batch_size)

            history = model.fit(xy_noisy,
                                epochs=completed_epochs+1,
                                initial_epoch=completed_epochs,
                                verbose=2,
                                callbacks=ann_callbacks)

            completed_epochs += 1

            # Tracking the total loss AND all individual components.
            THA_list = [history.history['loss'][0],
                        history.history['z_loss'][0],
                        history.history['y_loss'][0],
                        history.history['detJ_loss'][0]]
            train_history_annealing.append(THA_list)

        train_history_annealing = np.array(train_history_annealing)

        # Save the annealing history for future use.
        if SAVE_path:
            np.save(SAVE_path + f'tra_hist_ann_{dataset_string}_NCL{num_coupling_layers}_ID{intermediate_dims}_NL{num_layers}.npy',
                    train_history_annealing)

    # Then go on with the un-annealed stuff.
    history = model.fit(xy,
                        epochs=completed_epochs+num_epochs,
                        initial_epoch=completed_epochs,
                        verbose=2,
                        callbacks=callbacks)

    # Tracking the total loss AND all individual components.
    THA_list = [history.history['loss'],
                history.history['z_loss'],
                history.history['y_loss'],
                history.history['detJ_loss']]
    train_history = np.array(THA_list)

    if SAVE_path:
        np.save(SAVE_path + f'training_history_{dataset_string}_NCL{num_coupling_layers}_ID{intermediate_dims}_NL{num_layers}.npy',
                    train_history)

if SAVE_path:
    array_for_saving_model_weights = np.array(
                                    [model.coupling_layers_list[i].get_weights()
                                     for i in range(len(model.mask_indices))],
                                    dtype=object)

    np.save(SAVE_path + \
            f'weights_{dataset_string}_NCL{num_coupling_layers}_ID{intermediate_dims}_NL{num_layers}.npy',
            array_for_saving_model_weights)
    np.save(SAVE_path + \
            f'mask_indices_{dataset_string}_NCL{num_coupling_layers}_ID{intermediate_dims}_NL{num_layers}.npy',
            model.mask_indices)

###############################################################################

"""
PLOT THE RESULTS

Note that because all of the data is purely synthetic (and 2-D) in these datasets, we don't bother with validation data (instead just generating new data points every epoch during training). Similarly, we generate new datapoints here.

Also note that everything below this line only runs if PLOT==True.

Some plot code is left here, but commented out. The uncommented ones are the main demonstrations; the commented ones are sanity checks or supporting plots.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as pltick

"""
CONTINUOUS SECTORS
"""

num_plot_points = 10**4

if which_dataset == 'continuous_sectors' \
and PLOT:

    # We're going to make two sets of plots.
    # First is the plot of the continuous data on which the model was trained.
    # This is exactly the same as the training data.
    xy_continuous = make_continuous_sectors(num_plot_points,
                                            num_plot_points,
                                            sector_width)

    # Next, to verify that training on continuous data has not somehow distorted the map so that it ONLY works on the ENTIRE set of possible y, we will plot a few specific sectors with fixed y.
    # These should look like wedges and should map to the Gaussian regardless of angle y.
    xy_discrete = make_discrete_sectors(num_plot_points //
                                            len(which_discrete_sectors),
                                        which_discrete_sectors,
                                        sector_width)

    # Make numpy instances of the datasets because you can't plot a dataset object (it doesn't even have defined values in this case).
    xy_continuous_np = np.zeros((num_plot_points,
                                 3))
    xy_discrete_np = np.zeros((num_plot_points,
                               3))

    i = 0
    for batch in xy_continuous:
        xy_continuous_np[  i   * num_plot_points :
                         (i+1) * num_plot_points] = batch.numpy()
        i += 1

    i = 0
    for batch in xy_discrete:
        xy_discrete_np[  i   * num_plot_points //
                       len(which_discrete_sectors):
                       (i+1) * num_plot_points //
                       len(which_discrete_sectors)] = batch.numpy()
        i += 1

    xy_continuous = xy_continuous_np
    xy_discrete = xy_discrete_np
    y_discrete = xy_discrete_np[:,
                                2:]

    # Sort the continuous ones by y. It makes the plot look a little more orderly.
    xy_continuous = xy_continuous[np.argsort(xy_continuous[:,2])]

    plt.close('all')

    # Plot the history from training
    if TRAIN:
        plt.figure(figsize=(15, 10))
        plt.plot(history.history["loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")

    # From data to latent space (continuous).
    y_continuous = xy_continuous[:,
                                 2:]
    zy_continuous_mapped, _ = model(xy_continuous,
                                    direction=-1)
    z_continuous_mapped = zy_continuous_mapped[:,
                                               0:2]
    y_continuous_mapped = zy_continuous_mapped[:,
                                               2:]

    # Convert the mapped latent space from tensor to array.
    # Shuffle the result so we don't only see the higher-angle plots.
    zy_continuous_mapped = zy_continuous_mapped.numpy()
    np.random.shuffle(zy_continuous_mapped)

    # From data to latent space (discrete).
    zy_discrete_mapped, _ = model(xy_discrete,
                                  direction=-1)
    z_discrete_mapped = zy_discrete_mapped[:,
                                           0:2]
    y_discrete_mapped = zy_discrete_mapped[:,
                                           2:]

    # Get the CONTINUOUS conditioner for mapping from z -> x
    y2_continuous = np.random.uniform(low=0,
                                      high=2*np.pi,
                                      size=num_plot_points)
    y2_continuous = np.expand_dims(y2_continuous,
                                   axis=-1)

    # Get the DISCRETE conditioner for mapping from z -> x
    y2_discrete = np.random.randint(low=0,
                                    high=len(which_discrete_sectors),
                                    size=num_plot_points,
                                    dtype=np.int32)
    y2_discrete = y2_discrete.astype(np.float32)
    y2_discrete = np.expand_dims(y2_discrete,
                                 axis=-1)
    for i in range(y2_discrete.shape[0]):
        y2_discrete[i,0] = which_discrete_sectors[int(y2_discrete[i,0])]
    #y2_discrete *= np.pi

    # Sample from the model prior
    z2 = model.distribution.sample(xy_continuous.shape[0]).numpy()

    # Get the inputs for discrete and continuus z->x mappings
    zy2_discrete = np.concatenate((z2,
                                   y2_discrete),
                                  axis=-1)
    zy2_continuous = np.concatenate((z2,
                                     y2_continuous),
                                    axis=-1)

    xy_continuous_mapped, _ = model(zy2_continuous,
                                    direction=1)
    x_continuous_mapped = xy_continuous_mapped[:,
                                               0:2]
    y2_continuous_mapped = xy_continuous_mapped[:,
                                                2:]

    xy_discrete_mapped, _ = model(zy2_discrete,
                                  direction=1)
    x_discrete_mapped = xy_discrete_mapped[:,
                                           0:2]
    y2_discrete_mapped = xy_discrete_mapped[:,
                                            2:]

    # Plot f_y(...) vs. y' (continuous). This should look like a line between (0,0) and (2*pi,2*pi).
# =============================================================================
#     plt.figure()
#     plt.scatter(y_continuous,
#                 y_continuous_mapped)
#     plt.title('y_mapped vs. y (x -> z)')
# =============================================================================

    # Plot f_y'(...) vs. y (continuous). This should look like a line between (0,0) and (2*pi,2*pi).
# =============================================================================
#     plt.figure()
#     plt.scatter(y2_continuous,
#                 y2_continuous_mapped)
#     plt.title('y2_mapped vs. y2 (z -> x, continuous)')
# =============================================================================

     # Plot f_y(...) vs. y' (discrete). This should look like two points, one at (0,0) and the other at (2*pi,2*pi).
# =============================================================================
#     plt.figure()
#     plt.scatter(y_discrete,
#                 y_discrete_mapped)
#     plt.title('y_mapped vs. y (z -> x, discrete)')
# =============================================================================

     # Plot f_y'(...) vs. y (discrete). This should look like two points, one at (0,0) and the other at (2*pi,2*pi).
# =============================================================================
#     plt.figure()
#     plt.scatter(y2_discrete,
#                 y2_discrete_mapped)
#     plt.title('y2_mapped vs. y2 (z -> x, discrete)')
# =============================================================================

    # Plot the 2x2 (xy'->zy / zy -> xy') plot showing the forward and backward map for the continuous case.
    loc = pltick.MultipleLocator(base=1.0)
    loc2 = pltick.MultipleLocator(base=2.0)

    title_x = 0.5
    title_y = 0.875
    f, axes = plt.subplots(2, 2)
    f.set_size_inches(20, 15)

    # The real data space.
    axes[0, 0].scatter(xy_continuous[:, 0],
                       xy_continuous[:, 1],
                       c=xy_continuous[:,2],
                       s=3)
    axes[0, 0].set_title(r'$\mathbf{x},y$',
                         x=title_x,
                         y=title_y)#"Inference data space X")
    axes[0, 0].set_xlim([-2,
                         2])
    axes[0, 0].set_ylim([-2,
                         2])
    axes[0, 0].xaxis.set_major_locator(loc)
    axes[0, 0].yaxis.set_major_locator(loc)

    # Plot the mapped-to latent space.
    axes[0, 1].scatter(zy_continuous_mapped[:, 0],
                       zy_continuous_mapped[:, 1],
                       c=zy_continuous_mapped[:,2],
                       s=3)
    axes[0, 1].set_title(r"$f_Z(\mathbf{x},y),f_{Y'}(\mathbf{x},y)$",
                         x=title_x,
                         y=title_y)#"Inference latent space Z")
    axes[0, 1].set_xlim([-5,
                         5])
    axes[0, 1].set_ylim([-5,
                         5])
    axes[0, 1].xaxis.set_major_locator(loc2)
    axes[0, 1].yaxis.set_major_locator(loc2)

    # The sampled latent space.
    axes[1, 0].scatter(zy2_continuous[:, 0],
                       zy2_continuous[:, 1],
                       c=zy2_continuous[:,2],
                       s=3)
    axes[1, 0].set_title(r"$\mathbf{z},y'$",
                         x=title_x,
                         y=title_y)
    axes[1, 0].set_xlim([-5,
                         5])
    axes[1, 0].set_ylim([-5,
                         5])
    axes[1, 0].xaxis.set_major_locator(loc2)
    axes[1, 0].yaxis.set_major_locator(loc2)

    # The mapped-to data space.
    cart_x = [x for y,x in sorted(zip(y2_continuous_mapped[:,0],
                                      x_continuous_mapped[:,0]))]
    cart_y = [x for y,x in sorted(zip(y2_continuous_mapped[:,0],
                                      x_continuous_mapped[:,1]))]
    color_y = sorted(y2_continuous_mapped[:,0])
    axes[1, 1].scatter(cart_x,
                       cart_y,
                       c=color_y,
                       s=3)


# =============================================================================
#     axes[1, 1].scatter(xy_continuous_mapped[:, 0],
#                        xy_continuous_mapped[:, 1],
#                        c=xy_continuous_mapped[:, 2],
#                        s=3)
# =============================================================================
    axes[1, 1].set_title(r"$f^{-1}_X(\mathbf{z},y'),f^{-1}_Y(\mathbf{z},y')$",
                         x=title_x,
                         y=title_y)
    axes[1, 1].set_xlim([-2,
                         2])
    axes[1, 1].set_ylim([-2,
                         2])
    axes[1, 1].xaxis.set_major_locator(loc)
    axes[1, 1].yaxis.set_major_locator(loc)

    # Set all the axes to equal aspect ratio
    for i in range(2):
        for j in range(2):
            plt.sca(axes[i,j])
            plt.gca().set_aspect('equal')

    plt.suptitle('Joint Distribution')

    # Plot the 2x2 (xy'->zy / zy -> xy') plot showing the forward and backward map for the discrete case.
    f, axes = plt.subplots(2, 2)
    f.set_size_inches(20, 15)

    # The real data space.
    axes[0, 0].scatter(xy_discrete[:, 0],
                       xy_discrete[:, 1],
                       c=xy_discrete[:,2],
                       s=3)
    axes[0, 0].set_title(r'$\mathbf{x},y$',
                         x=title_x,
                         y=title_y)
    axes[0, 0].set_xlim([-2,
                         2])
    axes[0, 0].set_ylim([-2,
                         2])
    axes[0, 0].xaxis.set_major_locator(loc)
    axes[0, 0].yaxis.set_major_locator(loc)

    # Plot the mapped-to latent space.
    zy_discrete_mapped = tf.random.shuffle(zy_discrete_mapped)

    axes[0, 1].scatter(zy_discrete_mapped[:, 0],
                       zy_discrete_mapped[:, 1],
                       c=zy_discrete_mapped[:,2],
                       s=3)
    axes[0, 1].set_title(r"$f_Z(\mathbf{x},y),f_{Y'}(\mathbf{x},y)$",
                         x=title_x,
                         y=title_y)
    axes[0, 1].set_xlim([-5,
                         5])
    axes[0, 1].set_ylim([-5,
                         5])
    axes[0, 1].xaxis.set_major_locator(loc2)
    axes[0, 1].yaxis.set_major_locator(loc2)

    # The sampled latent space.
    axes[1, 0].scatter(zy2_discrete[:, 0],
                       zy2_discrete[:, 1],
                       c=zy2_discrete[:,2],
                       s=3)
    axes[1, 0].set_title(r"$\mathbf{z},y'$",
                         x=title_x,
                         y=title_y)
    axes[1, 0].set_xlim([-5,
                         5])
    axes[1, 0].set_ylim([-5,
                         5])
    axes[1, 0].xaxis.set_major_locator(loc2)
    axes[1, 0].yaxis.set_major_locator(loc2)

    # The mapped-to data space.
    axes[1, 1].scatter(xy_discrete_mapped[:, 0],
                       xy_discrete_mapped[:, 1],
                       c=xy_discrete_mapped[:, 2],
                       s=3)
    axes[1, 1].set_title(r"$f^{-1}_X(\mathbf{z},y'),f^{-1}_Y(\mathbf{z},y')$",
                         x=title_x,
                         y=title_y)
    axes[1, 1].set_xlim([-2,
                         2])
    axes[1, 1].set_ylim([-2,
                         2])
    axes[1, 1].xaxis.set_major_locator(loc)
    axes[1, 1].yaxis.set_major_locator(loc)

    # Set all the axes to equal aspect ratio
    for i in range(2):
        for j in range(2):
            plt.sca(axes[i,j])
            plt.gca().set_aspect('equal')

    plt.suptitle('Conditional Distributions')

    #############################################
    # Show the spatial order of points in the map
    #############################################
    # SECOND FIGURE FOR PAPER
    # CUT OFF THE BOTTOM PART
    z2 = model.distribution.sample(10000).numpy()
    z2 = np.concatenate((z2,z2,z2,z2))

    # Get the DISCRETE conditioner for mapping from z -> x
    y2_discrete = []
    for i in range(4):
        for j in range(10000):
            y2_discrete.append([which_discrete_sectors[i]])
    y2_discrete = np.array(y2_discrete)

    # Get the inputs for discrete and continuus z->x mappings
    zy2_discrete = np.concatenate((z2,
                                   y2_discrete),
                                  axis=-1)

    xy_discrete_mapped, _ = model(zy2_discrete,
                                  direction=1)
    x_discrete_mapped = xy_discrete_mapped[:,
                                           0:2]
    y2_discrete_mapped = xy_discrete_mapped[:,
                                            2:]

    f, axes = plt.subplots(2, 2)
    f.set_size_inches(20, 15)

    title_x = 0.5
    title_y = 0.875

    # The sampled latent space.
    axes[0,0].scatter(zy2_discrete[:, 0],
                      zy2_discrete[:, 1],
                      c=zy2_discrete[:,0],
                      s=3)
    axes[0,0].set_title(r"$\mathbf{z},y'$",
                        x=title_x,
                        y=title_y)
    axes[0,0].set_xlim([-5,
                        5])
    axes[0,0].set_ylim([-5,
                        5])
    axes[0,0].xaxis.set_major_locator(loc2)
    axes[0,0].yaxis.set_major_locator(loc2)

    axes[1,0].scatter(zy2_discrete[:, 0],
                      zy2_discrete[:, 1],
                      c=zy2_discrete[:,0],
                      s=3)
    axes[1,0].set_title(r"$\mathbf{z},y'$",
                        x=title_x,
                        y=title_y)
    axes[1,0].set_xlim([-5,
                        5])
    axes[1,0].set_ylim([-5,
                        5])
    axes[1,0].xaxis.set_major_locator(loc2)
    axes[1,0].yaxis.set_major_locator(loc2)

    # The mapped-to data space.
    axes[0,1].scatter(xy_discrete_mapped[:, 0],
                      xy_discrete_mapped[:, 1],
                      c=zy2_discrete[:, 0],
                      s=3)
    axes[0,1].set_title(r"$f^{-1}_X(\mathbf{z},y'),f^{-1}_Y(\mathbf{z},y')$",
                        x=title_x,
                        y=title_y)
    axes[0,1].set_xlim([-2,
                        2])
    axes[0,1].set_ylim([-2,
                        2])
    axes[0,1].xaxis.set_major_locator(loc)
    axes[0,1].yaxis.set_major_locator(loc)

    axes[1,1].scatter(xy_discrete_mapped[:, 0],
                      xy_discrete_mapped[:, 1],
                      c=zy2_discrete[:, 0],
                      s=3)
    axes[1,1].set_title(r"$f^{-1}_X(\mathbf{z},y'),f^{-1}_Y(\mathbf{z},y')$",
                        x=title_x,
                        y=title_y)
    axes[1,1].set_xlim([-2,
                        2])
    axes[1,1].set_ylim([-2,
                        2])
    axes[1,1].xaxis.set_major_locator(loc)
    axes[1,1].yaxis.set_major_locator(loc)

    # Set all the axes to equal aspect ratio
    for i in range(2):
        for j in range(2):
            plt.sca(axes[i,j])
            plt.gca().set_aspect('equal')

    plt.suptitle('')

    ###########################################################################

# MIXED DATASET

elif which_dataset == 'mixed' \
and PLOT:

    plt.close('all')

    if TRAIN:
        plt.figure(figsize=(15, 10))
        plt.plot(history.history["loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")

    xy = make_mixed_dataset(which_classes,
                                   num_batches_per_class=1,
                                   batch_size=num_plot_points)

    xy_np = np.zeros((num_plot_points * len(which_classes),
                      3))

    i = 0
    for batch in xy:
        xy_np[  i   * num_plot_points :
              (i+1) * num_plot_points] = batch.numpy()
        i += 1

    xy = xy_np

    xy = xy[np.argsort(xy[:,2])]

    y = xy[:,
           2:]
    zy_mapped, _ = model(xy,
                         direction=-1)
    z_mapped = zy_mapped[:,
                         0:2]
    y_mapped = zy_mapped[:,
                         2:]

    y2 = np.zeros_like(y)
    for i in range(len(which_classes)):
        y2[i*num_plot_points:
           (i+1)*num_plot_points,
           0] = i

    y2 = (y2 - np.mean(y2)) / np.std(y2)

    z = model.distribution.sample(xy.shape[0]).numpy()
    zy = np.concatenate((z,
                         y2),
                        axis=-1)

# =============================================================================
#     zy = zy[np.argsort(zy[:,2])]
# =============================================================================

    xy_mapped, _ = model(zy,
                 direction=1)
    x_mapped = xy_mapped[:,
                         0:2]
    y2_mapped = xy_mapped[:,
                          2:]

    plt.figure()
    plt.scatter(xy[:,2],
                zy_mapped[:,2])
    plt.title('y_mapped vs. y (x -> z)')

    plt.figure()
    plt.scatter(zy[:,2],
                xy_mapped[:,2])
    plt.title('y2_mapped vs. y2 (z -> x)')

    zy_mapped = zy_mapped.numpy()

    for i in range(len(which_classes)):

        #plt.figure()

        f, axes = plt.subplots(2, 2)
        f.set_size_inches(20, 15)

        axes[0, 0].scatter(xy[i*num_plot_points:
                              (i+1)*num_plot_points,
                              0],
                           xy[i*num_plot_points:
                              (i+1)*num_plot_points,
                              1],
                           c=xy[i*num_plot_points:
                                (i+1)*num_plot_points,
                                2],
                           s=3,
                           vmin=-y.max(),
                           vmax=y.max())
        axes[0, 0].set(title="Inference data space X",
                       xlabel="x",
                       ylabel="y")
        axes[0, 0].set_xlim([-2.5,
                             2.5])
        axes[0, 0].set_ylim([-2.5,
                             2.5])

        axes[0, 1].scatter(zy_mapped[i*num_plot_points:
                                     (i+1)*num_plot_points,
                                     0],
                           zy_mapped[i*num_plot_points:
                                     (i+1)*num_plot_points,
                                     1],
                           c=zy_mapped[i*num_plot_points:
                                       (i+1)*num_plot_points,
                                       2],
                           s=3,
                           vmin=-y.max(),
                           vmax=y.max())
        axes[0, 1].set(title="Inference latent space Z",
                       xlabel="x",
                       ylabel="y")
        axes[0, 1].set_xlim([-5,
                             5])
        axes[0, 1].set_ylim([-5,
                             5])
        axes[1, 0].scatter(zy[i*num_plot_points:
                              (i+1)*num_plot_points,
                              0],
                           zy[i*num_plot_points:
                              (i+1)*num_plot_points,
                              1],
                           c=zy[i*num_plot_points:
                                (i+1)*num_plot_points,
                                2],
                           s=3,
                           vmin=-y.max(),
                           vmax=y.max())
        axes[1, 0].set(title="Generated latent space Z",
                       xlabel="x",
                       ylabel="y")
        axes[1, 0].set_xlim([-5,
                             5])
        axes[1, 0].set_ylim([-5,
                             5])
        axes[1, 1].scatter(xy_mapped[i*num_plot_points:
                                     (i+1)*num_plot_points,
                                     0],
                           xy_mapped[i*num_plot_points:
                                     (i+1)*num_plot_points,
                                     1],
                           c=xy_mapped[i*num_plot_points:
                                       (i+1)*num_plot_points,
                                       2],
                           s=3,
                           vmin=-y.max(),
                           vmax=y.max())
        axes[1, 1].set(title="Generated data space X",
                       label="x",
                       ylabel="y")
        axes[1, 1].set_xlim([-2.5,
                             2.5])
        axes[1, 1].set_ylim([-2.5,
                             2.5])

    ###########################################################################

# CRESCENTS

# Paper figure for toy continuous modeling
elif which_dataset == 'crescents' \
and PLOT:

    plt.close('all')

    if TRAIN:
        plt.figure(figsize=(15, 10))
        plt.plot(history.history["loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")

    xy = make_moons_dataset(1,
                            num_plot_points,
                            noise,
                            overlapping)

    xy_np = np.zeros((num_plot_points * 2,
                      3))

    i = 0
    for batch in xy:
        xy_np[  i   * num_plot_points :
              (i+1) * num_plot_points] = batch.numpy()
        i += 1

    xy = xy_np
    np.random.shuffle(xy)

# =============================================================================
#     xy = xy[np.argsort(xy[:,2])]
# =============================================================================

    y = xy[:,
           2:]
    zy_mapped, _ = model(xy,
                         direction=-1)
    z_mapped = zy_mapped[:,
                         0:2]
    y_mapped = zy_mapped[:,
                         2:]

    y2 = np.zeros_like(y)
    y2[num_plot_points:,
       0] = np.ones((num_plot_points,))

    y2 = (y2 - np.mean(y2)) / np.std(y2)

    z = model.distribution.sample(xy.shape[0]).numpy()
    zy = np.concatenate((z,
                         y2),
                        axis=-1)
    np.random.shuffle(zy)

# =============================================================================
#     zy = zy[np.argsort(zy[:,2])]
# =============================================================================

    xy_mapped, _ = model(zy,
                 direction=1)
    x_mapped = xy_mapped[:,
                         0:2]
    y2_mapped = xy_mapped[:,
                          2:]

# =============================================================================
#     plt.figure()
#     plt.scatter(xy[:,2],
#                 zy_mapped[:,2])
#     plt.title('y_mapped vs. y (x -> z)')
# =============================================================================

# =============================================================================
#     plt.figure()
#     plt.scatter(zy[:,2],
#                 xy_mapped[:,2])
#     plt.title('y2_mapped vs. y2 (z -> x)')
# =============================================================================

    zy_mapped = zy_mapped.numpy()

    loc = pltick.MultipleLocator(base=1.0)
    loc2 = pltick.MultipleLocator(base=2.0)

    title_x = 0.5
    title_y = 0.875

    plt.close('all')
    f, axes = plt.subplots(2, 2)
    f.set_size_inches(20, 15)

    axes[0, 0].scatter(xy[:,
                          0],
                       xy[:,
                          1],
                       c=xy[:,
                            2],
                       s=3,
                       vmin=-y.max(),
                       vmax=y.max())
    axes[0, 0].set_title(r'$\mathbf{x},y$',
                         x=title_x,
                         y=title_y)#"Inference data space X")
    axes[0, 0].set_xlim([-3,
                         3])
    axes[0, 0].set_ylim([-3,
                         3])
    axes[0, 0].xaxis.set_major_locator(loc)
    axes[0, 0].yaxis.set_major_locator(loc)

    axes[0, 1].scatter(zy_mapped[:,
                                 0],
                       zy_mapped[:,
                                 1],
                       c=zy_mapped[:,
                                   2],
                       s=3,
                       vmin=-y.max(),
                       vmax=y.max())
    axes[0, 1].set_title(r"$f_Z(\mathbf{x},y),f_{Y'}(\mathbf{x},y)$",
                         x=title_x,
                         y=title_y)#"Inference latent space Z")
    axes[0, 1].set_xlim([-5,
                         5])
    axes[0, 1].set_ylim([-5,
                         5])
    axes[0, 1].xaxis.set_major_locator(loc2)
    axes[0, 1].yaxis.set_major_locator(loc2)

    axes[1, 0].scatter(zy[:,
                          0],
                       zy[:,
                          1],
                       c=zy[:,
                            2],
                       s=3,
                       vmin=-y.max(),
                       vmax=y.max())
    axes[1, 0].set_title(r"$\mathbf{z},y'$",
                         x=title_x,
                         y=title_y)
    axes[1, 0].set_xlim([-5,
                         5])
    axes[1, 0].set_ylim([-5,
                         5])
    axes[1, 0].xaxis.set_major_locator(loc2)
    axes[1, 0].yaxis.set_major_locator(loc2)

    axes[1, 1].scatter(xy_mapped[:,
                                 0],
                       xy_mapped[:,
                                 1],
                       c=xy_mapped[:,
                                   2],
                       s=3,
                       vmin=-y.max(),
                       vmax=y.max())
    axes[1, 1].set_title(r"$f^{-1}_X(\mathbf{z},y'),f^{-1}_Y(\mathbf{z},y')$",
                         x=title_x,
                         y=title_y)
    axes[1, 1].set_xlim([-3,
                         3])
    axes[1, 1].set_ylim([-3,
                         3])
    axes[1, 1].xaxis.set_major_locator(loc)
    axes[1, 1].yaxis.set_major_locator(loc)

    # Set all the axes to equal aspect ratio
    for i in range(2):
        for j in range(2):
            plt.sca(axes[i,j])
            plt.gca().set_aspect('equal')

    plt.suptitle('Joint Distribution')

# =============================================================================
#     zy_hist_0 = np.histogram(zy[:,0],bins=100,density=True)
#     zy2_hist_0 = np.histogram(zy_mapped[:,0],bins=100,density=True)
#     zy_hist_1 = np.histogram(zy[:,1],bins=100,density=True)
#     zy2_hist_1 = np.histogram(zy_mapped[:,1],bins=100,density=True)
#     plt.figure(figsize=(15,10))
#     plt.plot(zy_hist_0[1][:-1],zy_hist_0[0])
#     plt.plot(zy2_hist_0[1][:-1],zy2_hist_0[0])
#     plt.figure(figsize=(15,10))
#     plt.plot(zy_hist_1[1][:-1],zy_hist_1[0])
#     plt.plot(zy2_hist_1[1][:-1],zy2_hist_1[0])
# =============================================================================

# Class interpolation for crescents

    num_interps = 5
    num_extras = 2 # how many above and below the actual limits to go
    num_extras *= 2 # combine above+below


    y2 = np.zeros(((num_interps+num_extras)*num_plot_points,1))
    for i in range(num_interps+num_extras):
        y2[num_plot_points*i:num_plot_points*(i+1),0] = \
            (i/(num_interps - 1))*np.ones((num_plot_points,))

    y2 -= (1/(num_interps-1))*(num_extras/2)

    y2 = (y2 - 0.5) / 0.5 # use mean/stddev values for non-interpolation case

    z = model.distribution.sample((num_interps+num_extras)*num_plot_points).numpy()
    zy = np.concatenate((z,
                         y2),
                        axis=-1)
    # =============================================================================
    # np.random.shuffle(zy)
    # =============================================================================

    # =============================================================================
    #     zy = zy[np.argsort(zy[:,2])]
    # =============================================================================

    xy_mapped, _ = model(zy,
                 direction=1)
    x_mapped = xy_mapped[:,
                         0:2]
    y2_mapped = xy_mapped[:,
                          2:]

    loc = pltick.MultipleLocator(base=1.0)
    loc2 = pltick.MultipleLocator(base=2.0)

    title_x = 0.5
    title_y = 1

    f, axes = plt.subplots(3, 3)
    f.set_size_inches(20, 15)

    ax1 = [0,0,0,1,1,1,2,2,2]
    ax2 = [0,1,2,0,1,2,0,1,2]
    titles = [r'$y=-2$',
              r'$y=-1.5$',
              r'$y=-1$ (true label)',
              r'$y=-0.5$',
              r'$y=0$',
              r'$y=0.5$',
              r'$y=1$ (true label)',
              r'$y=1.5$',
              r'$y=2$']

    for i in range(9):

        axes[ax1[i],ax2[i]].scatter(
            xy_mapped[i*num_plot_points:(i+1)*num_plot_points,
                                        0],
                              xy_mapped[i*num_plot_points:(i+1)*num_plot_points,
                                        1],
                              c=xy_mapped[i*num_plot_points:(i+1)*num_plot_points,
                                          2],
                              s=3,
                              vmin=-y2.max(),
                              vmax=y2.max())
        axes[ax1[i],ax2[i]].set_title(titles[i],
                             x=title_x,
                             y=title_y)
        axes[ax1[i],ax2[i]].set_xlim([-3,
                                      3])
        axes[ax1[i],ax2[i]].set_ylim([-3,
                                      3])
        axes[ax1[i],ax2[i]].xaxis.set_major_locator(loc)
        axes[ax1[i],ax2[i]].yaxis.set_major_locator(loc)

    # Remove ticks
    for i in range(9):
        plt.sca(axes[ax1[i],ax2[i]])
        plt.gca().tick_params(axis='both',
                              which='both',
                              bottom=False,
                              top=False,
                              left=False,
                              right=False,
                              labelbottom=False,
                              labelleft=False)

    # Set all the axes to equal aspect ratio
    for i in range(9):
        plt.sca(axes[ax1[i],ax2[i]])
        plt.gca().set_aspect('equal')