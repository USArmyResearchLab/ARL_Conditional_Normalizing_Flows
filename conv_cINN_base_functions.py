#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:25:58 2021

@author: John S Hyatt
"""

import tensorflow as tf

from tensorflow.keras.layers import (LeakyReLU,
                                     LayerNormalization,
                                     Lambda,
                                     Convolution2D,
                                     Dropout,
                                     Concatenate)

import numpy as np

###############################################################################

"""
FUNCTION FOR PARSING TFRECORDS
"""

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
    # The tensor object is 1-dimensional, so reshape it using the image dimensions obtained earlier.
    img = tf.reshape(img,
                     shape=(height,
                            width,
                            depth))

    return img, label

###############################################################################

"""
FUNCTION FOR 2x2 AVERAGE POOL DOWNSAMPLING
FUNCTION FOR 2x2 UPSAMPLING
"""

@tf.function
def down(img):

    """
    Args:
        img:  an HxWxD image OR a batch of such images.  In the latter case the inputs have an extra (batch) dimension.

    Returns:
        down(img):  a 2x2 average-pooled downsampling of img -> an (H/2)x(W/2)xD image OR a batch of such images.  In the latter case the inputs have an extra (batch) dimension.

    Intended to be used with map().
    """

    # if performing on individual examples with no batch dimension, need to expand dimensions to give a "batch size" of 1.
    batch = True

    if len(tf.shape(img)) == 3:
        img = tf.expand_dims(img,
                             axis=0)

        batch = False

    batch_size = tf.shape(img)[0]
    (M, N) = (tf.shape(img)[1], # img height
              tf.shape(img)[2]) # img width
    depth = tf.shape(img)[3]

    (K, L) = (2, 2)

    MK = M // K
    NL = N // L

    downsampled = img[:,
                      :MK*K,
                      :NL*L,
                      :]

    downsampled = tf.reshape(downsampled,
                             shape=(batch_size,
                                    MK, K,
                                    NL, L,
                                    depth))

    downsampled = tf.reduce_mean(downsampled,
                                 axis=(2,
                                       4))

    # if performing on individual examples, need to squeeze out the "batch" dimension.
    if not batch:
        downsampled = tf.squeeze(downsampled,
                                 axis=0)

    return downsampled

@tf.function
def up(img):

    """
    Args:
        img:  an HxWxD image OR a batch of such images.  In the latter case the inputs have an extra (batch) dimension.

    Returns:
        up(img):  a 2x2-repeated upscaling of img -> a (2*H)x(2*W)xD image OR a batch of such images.  In the latter case the inputs have an extra (batch) dimension.

    Intended to be used with map().
    """

    # if performing on individual examples with no batch dimension, need to expand dimensions to give a "batch size" of 1.
    batch = True

    if len(tf.shape(img)) == 3:
        img = tf.expand_dims(img,
                             axis=0)

        batch = False

    # Upsample in H.
    upscaled = tf.repeat(img,
                         repeats=2,
                         axis=1)
    # Upsample in W.
    upscaled = tf.repeat(upscaled,
                         repeats=2,
                         axis=2)

    # if performing on individual examples, need to squeeze out the "batch" dimension.
    if not batch:
        upscaled = tf.squeeze(upscaled,
                              axis=0)

    return upscaled

###############################################################################

"""
FUNCTIONS FOR DATASET PREPROCESSING
"""

# Preprocessing for a class-conditional model is much simpler than for the other model types if we are working with raw pixels. If we are working with logits instead, this is where that happens.
# The rest of the preprocessing (concatenating y labels to the x elements) is more easily done in-line. See the main script.
def preprocess_dataset_class(x_dataset,
                             LOGITS=False,
                             a=0.01):

    """
    Args:
        x_dataset: the loaded dataset. x has been scaled to the domain [0,1].
        LOGITS: Boolean. if True, x is replaced with logit(x) (where a small fudge factor is included in the definition of "logit" to keep the min and max extreme values finite).
        a: the fudge factor for 0. (The fudge factor for 1 is derived from this; see below.)

    Returns:
        x_dataset_logitized_scaled:
            x -> logit(a + (1-a) * b * x)
               ~ logit(x)
            where b is another (derived) fudge factor. This is then scaled to be on the interval [0,1].

    """

    def _preprocess_for_logit(x_element):

        x_element = a + (1-a) * b * x_element

        return x_element

    def _logit(x_element):

        logit_x_element = tf.math.log(x_element / (1 - x_element))

        return logit_x_element

    def _scale_logit(x_element):

        scaled_x_element = (x_element - min_value) / (max_value - min_value)

        return scaled_x_element

    # Do nothing if the data is not being mapped to logits.
    if not LOGITS:

        return x_dataset

    else:

        # This will give a=0.01 min and 0.99=1-a max.
        b = (1 - 2 * a) / (1 - a)

        # The min and max values for the fudged logit are [logit(a),logit(1-a)]. (For a=0.01, these are [-4.6,4.6]). Shift these to [0,1].
        min_value = _logit(a)
        max_value = _logit(1-a)

        x_dataset = x_dataset.map(_preprocess_for_logit,
                                  num_parallel_calls=tf.data.AUTOTUNE)
        x_dataset = x_dataset.map(_logit,
                                  num_parallel_calls=tf.data.AUTOTUNE)
        x_dataset = x_dataset.map(_scale_logit,
                                  num_parallel_calls=tf.data.AUTOTUNE)

        return x_dataset

def preprocess_dataset_SR(x_dataset,
                          model_type,
                          RESIDUAL=True):

    """
    Args:
        x_dataset:  the loaded dataset.
        model_type:  one of 'SR4,2', 'SR4,1', or 'SR2,1'. The number on the left represents the resolution of y in terms of 4x4 downsampled or 2x2 downsampled. The number on the right represents the resolution of x in terms of 2x2 downsampled or original resolution.
        RESIDUAL: Boolean. if True, x is a residual (i.e. x+y = the true high-res image). if False, x is just the high-res image.

    Returns:
        xy_dataset:  the cache-able dataset including all transformations that are NOT different across epochs. It has elements (x,y).
    """

    def _preprocess_42(x_element_hires):

        x_element = down(x_element_hires)
        y_element = up(down(down(x_element_hires)))

        if RESIDUAL:
            x_element -= y_element

        return tf.concat((x_element,y_element),
                         axis=-1)

    def _preprocess_21(x_element_hires):

        x_element = x_element_hires
        y_element = up(down(x_element_hires))

        if RESIDUAL:
            x_element -= y_element

        return tf.concat((x_element,y_element),
                         axis=-1)

    # '4,2' means the size should be 14x14.
    if model_type == 'SR4,2':
        xy_dataset = x_dataset.map(_preprocess_42,
                                   num_parallel_calls=tf.data.AUTOTUNE)

    # '2,1' means the size should be 28x28.
    elif model_type == 'SR2,1':
        xy_dataset = x_dataset.map(_preprocess_21,
                                   num_parallel_calls=tf.data.AUTOTUNE)

    return xy_dataset

###############################################################################

"""
FUNCTION FOR OBTAINING PIXEL VALUE X FROM X' GENERATED FROM LOGIT-TRAINED MODEL
"""

def de_logitify(x,
                a=0.01):

    """
    Args:
        x: data-space prediction from a model trained on logit-ified inputs.
        a: the same fudge factor used to generate the training data.

    Returns:
        x: the de-logit-ified version of that output.

    This is written to accept numpy inputs, not tensors.
    """

    def _logit(x):

        logit_x = np.log(x / (1 - x))

        return logit_x

    min_value = _logit(a)
    max_value = _logit(1-a)

    def _logistic(x):

        return 1 / (1 + np.exp(-x))

    b = (1 - 2 * a) / (1 - a)

    x = x * (max_value - min_value) + min_value

    return (_logistic(x) - a) / (b * (1 - a))

###############################################################################

"""
DEFINING RESIDUAL BLOCKS
"""

"""
Part of this code is adapted from https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce, which is an implementation of the design described in https://arxiv.org/pdf/1512.03385.pdf and https://arxiv.org/pdf/1611.05431.pdf. It includes the proposed ResNeXt block change from https://arxiv.org/pdf/1603.05027.pdf and changes the (Batch Norm -> ReLU) components to (LeakyReLU -> Layer Norm) (changing the activation function, changing the type of normalization, and swapping the order).
"""

def add_common_layers(y,
                      ln=False,
                      do=False,
                      ln_axis=-1):

    """
    Args:
        y: an input tensor.
        ln: Boolean. Whether or not to include layer normalization across the grouped convolution. (Due to the way the residual block is constructed, in practice, all layers will have the same ln value.)
        do: Boolean. Whether or not to include dropout across the grouped convolution. (Due to the way the residual block is constructed, in practice, all layers will have the same do value.)

    Returns:
        y: an output tensor that has passed through the activation function layer as well as (if applicable) the dropout and/or layer normalization layers.
    """

    y = LeakyReLU()(y)

    if do:
        y = Dropout(do)(y)
    if ln:
        y_h = y.shape[-3]
        y_w = y.shape[-2]
        y_d = y.shape[-1]

        # LayerNormalization has to have everything in one channel or it will normalize along "unexpected" axis (that is, the "expected" behavior is not the desired behavior for this type of input). The following code implements layer normalization using the correct normalization axis.
        # Flatten everything -> LayerNorm -> put back into original shape.
        y = tf.keras.layers.Reshape((y_h * y_w * y_d,))(y)
        y = LayerNormalization(axis=ln_axis)(y)
        y = tf.keras.layers.Reshape((y_h,
                                     y_w,
                                     y_d))(y)

    return y

def grouped_convolution(y,
                        nb_channels,
                        _strides,
                        ksize,
                        dilation,
                        cardinality,
                        init='glorot_uniform'):

    """
    Args:
        y: an input tensor.
        nb_channels: the total number of channels in all branches of the grouped convolution.
        _strides: the `strides` argument in Convolution2D.
        ksize: the `kernel_size` argument in Convolution2D.
        dilation: the `dilation_rate` argument in Convolution2D.
        cardinality: how many branches in the grouped convolution.
        init: the `kernel_initializer` argument in Convolution2D.

    Returns:
        y: an output tensor that has passed through grouped convolution.

    """

    # This is just a standard 2D convolution when `cardinality` == 1.
    if cardinality == 1:
        return Convolution2D(nb_channels,
                             kernel_size=ksize,
                             strides=_strides,
                             padding='same',
                             dilation_rate=dilation,
                             kernel_initializer=init)(y)

    assert not nb_channels % cardinality
    _d = int(nb_channels // cardinality)

    # In a grouped convolution layer, input and output channels are divided into `cardinality` groups, and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
        groups.append(Convolution2D(_d,
                                    kernel_size=ksize,
                                    strides=_strides,
                                    padding='same',
                                    dilation_rate=dilation,
                                    kernel_initializer=init)(group))

    # The output of the layer is a concatenation of all the groups.
    y = tf.keras.layers.concatenate(groups)

    return y

def residual_block(y,
                   nb_channels_in,
                   nb_channels_out,
                   _strides=(1, 1),
                   _project_shortcut=False,
                   ksize=(4, 4),
                   cardinality=4,
                   ln=False,
                   do=False,
                   ln_axis=-1,
                   init='glorot_uniform'):

    """
    Args:
        y: an input tensor.
        nb_channels_in: the reduced number of channels in the bottleneck/dimension reduction step.
        nb_channels_out: the number of channels in the output.
        _strides: the `strides` argument in Convolution2D.
        _project_shortcut: Boolean. The ResNeXt block contains a shortcut/skip connection in order to preserve gradients in an arbitrarily deep network (see references above). If the number of channels in y is the same as nb_channels_out, then the shortcut can combine directly with the output of the convolutions. Otherwise (if this argument is True) the shortcut must be projected into the same space, which it does by passing through a 1x1 convolution layer with `nb_channels_out` channels.
        ksize: the `kernel_size` argument in Convolution2D.
        cardinality: how many branches in the grouped convolution.
        ln: Boolean. Whether or not to include layer normalization across the grouped convolution.
        do: Boolean. Whether or not to include dropout across the grouped convolution.
        ln_axis: the axis across which layer normalization will be computed. See add_common_layers() - this is not quite the same as the `axis` argument of LayerNormalization!
        init: the `kernel_initializer` argument in Convolution2D.

    Returns:
        y: an output tensor that has passed through the residual block.
    """

    # This is the identity / shortcut / skip connection path.
    shortcut = y

    # Modify the residual building block as a bottleneck design to make the network more economical.
    y = add_common_layers(y,
                          ln,
                          do,
                          ln_axis)

    y = Convolution2D(nb_channels_in,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer=init)(y)

    # ResNeXt (identical to ResNet when `cardinality` == 1).
    y = add_common_layers(y,
                          ln,
                          do,
                          ln_axis)

    y = grouped_convolution(y,
                            nb_channels_in,
                            _strides=_strides,
                            ksize=ksize,
                            dilation=(1,1),
                            cardinality=cardinality,
                            init=init)

    y = add_common_layers(y,
                          ln,
                          do,
                          ln_axis)

    # Map the aggregated branches to the desired number of output channels.
    y = Convolution2D(nb_channels_out,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer=init)(y)

    # The identity shortcut is used directly when the input and output are of the same dimensions. Otherwise, it must be projected into the same number of channels (1x1 convolutions) / same feature map size (strided convolutions).
    if _project_shortcut \
    or _strides != (1, 1):
        shortcut = Convolution2D(nb_channels_out,
                                 kernel_size=(1, 1),
                                 strides=_strides,
                                 padding='same',
                                 kernel_initializer=init)(shortcut)

    # Add the shortcut and the transformation block.
    y = tf.keras.layers.add([shortcut, y])

    return y

# Same as residual_block(), but uses parallel dilated convolutions.
def dilated_residual_block(y,
                           nb_channels_in,
                           nb_channels_out,
                           _strides=(1, 1),
                           _project_shortcut=False,
                           _which_dilations=[1,2,4],
                           ksize=(4, 4),
                           cardinality=4,
                           ln=False,
                           do=False,
                           ln_axis=-1,
                           init='glorot_uniform'):

    """
        y: an input tensor.
        nb_channels_in: the reduced number of channels in the bottleneck/dimension reduction step. Must be divisible by cardinality times each individual dilation factor.
        nb_channels_out: the number of channels in the output.
        _strides: the `strides` argument in Convolution2D.
        _project_shortcut: Boolean. The ResNeXt block contains a shortcut/skip connection in order to preserve gradients in an arbitrarily deep network (see references above). If the number of channels in y is the same as nb_channels_out, then the shortcut can combine directly with the output of the convolutions. Otherwise (if this argument is True) the shortcut must be projected into the same space, which it does by passing through a 1x1 convolution layer with `nb_channels_out` channels.

        _which_dilations: a list of dilation factors. dilated_residual_block() essentially replicates residual_block() one time for each element in this list; each of those blocks contains convolutions with the corresponding dilation factor. To save memory, and assuming that fewer kernels are needed to represent longer-range correlations, the number of kernels in each block is also divided by the associated dilation factor. For example, with 3x3 kernels and dilations [1,2,4,8], the kernels in each residual_block() look like:

    1 1 1   1 0 1 0 1   1 0 0 0 1 0 0 0 1   1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1
    1 1 1   0 0 0 0 0   0 0 0 0 0 0 0 0 0   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    1 1 1   1 0 1 0 1   0 0 0 0 0 0 0 0 0   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0   0 0 0 0 0 0 0 0 0   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            1 0 1 0 1   1 0 0 0 1 0 0 0 1   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                        0 0 0 0 0 0 0 0 0   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                        0 0 0 0 0 0 0 0 0   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                        0 0 0 0 0 0 0 0 0   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                        1 0 0 0 1 0 0 0 1   1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1
                                            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                                            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                                            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                                            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                                            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                                            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                                            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                                            1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1

        ksize: the `kernel_size` argument in Convolution2D.
        cardinality: how many branches in the grouped convolution.
        ln: Boolean. Whether or not to include layer normalization across the grouped convolution.
        do: Boolean. Whether or not to include dropout across the grouped convolution.
        ln_axis: the axis across which layer normalization will be computed. See add_common_layers() - this is not quite the same as the `axis` argument of LayerNormalization!
        init: the `kernel_initializer` argument in Convolution2D.

    Returns:
        y: an output tensor that has passed through the residual block. Because the dilations represent longer-range correlations, these transformations will be better able to capter larger-scale pixel correlations (somewhat closer to true semantic relationships than just short-range pixel correlations).
    """

    # This is the identity / shortcut / skip connection path.
    shortcut = y

    # Modify the residual building block as a bottleneck design to make the network more economical.
    y = add_common_layers(y,
                          ln,
                          do,
                          ln_axis)

    y = Convolution2D(nb_channels_in,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer=init)(y)

    # ResNeXt (identical to ResNet when `cardinality` == 1).
    y = add_common_layers(y,
                          ln,
                          do,
                          ln_axis)

    # Branch into parallel convolutions, each with a different dilation value.
    # Reduce the overall number of params by using fewer channels for longer-range features.
    if len(_which_dilations) > 1:
        grouped_convolution_list = []
        for dilation_factor in _which_dilations:

            y_p = grouped_convolution(y,
                                      nb_channels_in // dilation_factor,
                                      _strides=_strides,
                                      ksize=ksize,
                                      dilation=(dilation_factor,
                                                dilation_factor),
                                      cardinality=cardinality,
                                      init=init)

            grouped_convolution_list.append(y_p)

        y = Concatenate(axis=-1)(grouped_convolution_list)

    else:
        dilation_factor = _which_dilations[0]
        y = grouped_convolution(y,
                                nb_channels_in // dilation_factor,
                                _strides=_strides,
                                ksize=ksize,
                                dilation=(dilation_factor,
                                          dilation_factor),
                                cardinality=cardinality,
                                init=init)

    y = add_common_layers(y,
                          ln,
                          do,
                          ln_axis)

    # Map the aggregated branches to the desired number of output channels.
    y = Convolution2D(nb_channels_out,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer=init)(y)

    # The identity shortcut is used directly when the input and output are of the same dimensions. Otherwise, it must be projected into the same number of channels (1x1 convolutions) / same feature map size (strided convolutions).
    if _project_shortcut \
    or _strides != (1, 1):
        shortcut = Convolution2D(nb_channels_out,
                                 kernel_size=(1, 1),
                                 strides=_strides,
                                 padding='same',
                                 kernel_initializer=init)(shortcut)

    # Add the shortcut and the transformation block.
    y = tf.keras.layers.add([shortcut, y])

    return y

###############################################################################

"""
ADDING INSTANCE NOISE
"""

@tf.function
def instance_noise(x_element,
                   alpha):

    """
    Args:
        x_element: a dataset element with shape (height, width, depth).
        alpha: desired relative amount of signal.

    Returns:
        x_element_noisy: mapped from x -> alpha * x + (1 - alpha) * noise, where noise is drawn from N(0,1).
    """

    noise = tf.random.normal(shape=tf.shape(x_element),
                             mean=0,
                             stddev=1)

    x_element_noisy = alpha * x_element + (1 - alpha) * noise

    return x_element_noisy

"""
RENEWING NOISE IN NOISE-ONLY ELEMENTS
"""

# Define a function to sample new Gaussian noise every time the dataset is called.
@tf.function
def renew_noise(element):

    """
    Args:
        element: a dataset element with shape (height, width, depth).

    Returns:
        element_noisy: a new element drawn from N(0,1).
    """

    element_new = tf.random.normal(shape=tf.shape(element),
                                   mean=0,
                                   stddev=1)

    return element_new