#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:31:58 2021

@author: John S Hyatt
"""

import tensorflow as tf

import numpy as np

from math import pi

###############################################################################

def make_moons_dataset(num_batches_per_class,
                       batch_size,
                       noise,
                       overlapping):

    """
    Args:
        num_batches_per_class: the number of batches per class in the dataset (there are only 2 classes). We use this rather than the number of points per class to guarantee that each batch will contain ONLY examples from one class.
        batch_size: the batch size of the generated dataset (i.e. number of points, all from the same class, in a batch)
        noise: noise parameter from sklearn.datasets.make_moons. The larger this value, the "fatter" the crescents.
        overlapping: Boolean. If False, provides the same interleaved noisy crescents as sklearn.datasets.make_moons. If True, vertically shifts them (pre-normalization) so that they overlap.

    Returns:
        xy: a TF dataset object. Note that the data is batched BEFORE being shuffled, since each class should map independently to N(0,1), rather than all classes together mapping to N(0,1) (which could lead to each class mapping to a sub-distribution).
    """

    def my_make_moons(n_samples_per,
                      noise,
                      overlapping):
        """
        Adaptation of sklearn.datasets.make_moons, to allow the generation of overlapping datasets. This is necessary for proper normalization of the tensorflow dataset object's elements. Changes from sklearn.datasets.make_moons are not documented and it doesn't have some functionality I didn't need.

        Arguments:
            n_samples_per : number of points PER CRESCENT.
            noise : standard deviation of Gaussian noise added to the crescents.
            overlapping: Boolean. If False, provides the same interleaved noisy crescents as sklearn.datasets.make_moons. If True, vertically shifts them (pre-normalization) so that they overlap.

        Returns:
            X : ndarray of shape (2*n_samples_per, 2) with the Cartesian coordinates of the points.
            Y : ndarray of shape (2*n_samples_per,) with the class label.
        """

        # x-coordinates of the concave-down circle on the left
        semicircle_1_x = np.cos(np.linspace(0,
                                            pi,
                                            n_samples_per))

        # y-coordinates of the concave-down circle on the left
        semicircle_1_y = np.sin(np.linspace(0,
                                            pi,
                                            n_samples_per))

        # x-coordinates of the concave-up circle on the right
        semicircle_2_x = 1 - np.cos(np.linspace(0,
                                                pi,
                                                n_samples_per))

        # y-coordinates of the concave-up circle on the right
        if not overlapping:
            semicircle_2_y = 1 - np.sin(np.linspace(0,
                                                    pi,
                                                    n_samples_per)) - 0.5
        else:
            semicircle_2_y = 1 - np.sin(np.linspace(0,
                                                    pi,
                                                    n_samples_per)) + 0.25

        # Combine these to get the coordinates.
        X = np.vstack([np.append(semicircle_1_x,
                                 semicircle_2_x),
                       np.append(semicircle_1_y,
                                 semicircle_2_y)]).T

        # The labels are:
        #   0 for the concave-down circle on the left
        #   1 for the concave-up circle on the right (NON-OVERLAPPING)
        #   2 for the concave-up circle on the right (OVERLAPPING)
        # Note that only one of (1 or 2) will be used.
        if not overlapping:
            Y = np.hstack([np.zeros(n_samples_per),
                           np.ones(n_samples_per)])
        else:
            Y = np.hstack([np.zeros(n_samples_per),
                           2*np.ones(n_samples_per)])

        # Add noise to the coordinates
        X += np.random.normal(loc=0,
                              scale=noise,
                              size=X.shape)

        return X, Y

    # Calculate:
    # the number of points per class
    # the total number of points in the dataset (per epoch)
    # the total number of batches
    num_points_per_class = num_batches_per_class * batch_size
    num_classes = 2
    num_points = num_classes * num_points_per_class
    num_batches = num_batches_per_class * num_classes

    # Make an initial numpy array like the dataset to calculate the correct mean and std. dev. for the given value of noise. Mean is always the same (for a given value of overlapping) but the x-components of std. dev. DEPEND ON THE VALUE OF noise.
    # This specific way of doing it assumes that there are enough points per epoch that the mean and std. dev. are constant to within acceptable precision over multiple epochs.
    # Generate a numpy array that will look like the final dataset.
    x,y = my_make_moons(n_samples_per=10**4,
                        noise=noise,
                        overlapping=overlapping)
    y = np.expand_dims(y,
                       axis=1)
    xy = np.concatenate((x,y),
                        axis=1)

    # Find the mean and std. dev.
    xy_mean = np.mean(xy,
                      axis=0)
    xy_std = np.std(xy,
                    axis=0)

    xy_mean = xy_mean.astype(np.float32)
    xy_std = xy_std.astype(np.float32)

    # Make a dataset containing the class labels.
    # Half the class labels will be 0.
    y0 = np.zeros((num_points_per_class,),
                  dtype=np.int32)

    # The other half will be either 1 or 2, depending on overlapping.
    if not overlapping:
        y1 = np.ones((num_points_per_class,),
                     dtype=np.int32)
    else:
        y1 = 2 * np.ones((num_points_per_class,),
                         dtype=np.int32)

    # Combine and convert to a dataset object, then cache it, because y will not change.
    y = np.concatenate((y0,y1))
    y = tf.data.Dataset.from_tensor_slices(y)
    y = y.cache()

    # Unfortunately, due to the way RNG is called by numpy functions vs. TensorFlow functions, I can't just use sklearn.datasets.make_moons directly, or it will generate the same point cloud every epoch.
    # Fortunately, it's not THAT hard to recreate the code for sklearn.datasets.make_moons using tf functions only.
    @tf.function
    def tf_make_moons(y,
                      noise):

        """
        Args:
            y: an element denoting class label, with value 0 or 1.
            noise: std. dev. of Guassian noise added to the data.

        Returns:
            x0, x1, y: an element containing Cartesian coordinates and the class label of a generated point corresponding to the input class label.
        """

        # Get the angle of each point on its respective semicircle.
        angle = tf.random.uniform(shape=[],
                                  minval=0,
                                  maxval=pi)

        # This is for the concave-down semiircle on the left.
        @tf.function
        def make_moon_1():

            x0 = tf.math.cos(angle)
            x1 = tf.math.sin(angle)

            return x0, x1

        # This is for the concave-up semicircle on the right.
        # NON-OVERLAPPING CASE
        @tf.function
        def make_moon_2():

            x0 = 1 - tf.math.cos(angle)
            x1 = 1 - tf.math.sin(angle) - 0.5

            return x0, x1

        # This is for the concave-up semicircle on the right.
        # OVERLAPPING CASE
        @tf.function
        def make_moon_2_overlap():

            x0 = 1 - tf.math.cos(angle)
            x1 = 1 - tf.math.sin(angle) + 0.25

            return x0, x1

        # Pick which of the above functions to evaluate, depending on which crescent we want to generate a point for.
        x0, x1 = tf.switch_case(y,
                        branch_fns={0 : make_moon_1,
                                    1 : make_moon_2,
                                    2 : make_moon_2_overlap})

        # Add Gaussian noise to the generated point.
        x0 += tf.random.normal(shape=[],
                               mean=0.0,
                               stddev=noise,
                               dtype=tf.dtypes.float32)
        x1 += tf.random.normal(shape=[],
                               mean=0.0,
                               stddev=noise,
                               dtype=tf.dtypes.float32)

        # Convert y from integer to float.
        y = tf.cast(y,
                    dtype=tf.dtypes.float32)

        return x0, x1, y

    @tf.function
    def standardize(x0,
                    x1,
                    y,
                    xy_mean,
                    xy_std):

        """
        Standardizes (mean=0, std. dev.=1) the data in the generated dataset. The mean and std. dev. values used to perform this standardization are obtained from a single generated numpy dataset with many (10,000) points per crescent.

        Args:
            x0, x1, y: a single (x,y) element (x0, x1,y)
        Returns:
            x0, x1, y: the same element, but with zero mean, 1 std. dev.
        """

        x0 = (x0 - xy_mean[0]) / xy_std[0]
        x1 = (x1 - xy_mean[1]) / xy_std[1]
        y = (y - xy_mean[2]) / xy_std[2]

        return x0, x1, y

    # Generate the dataset.
    xy = y.map(lambda y: tf_make_moons(y, noise))

    # Standardize the dataset.
    xy = xy.map(lambda x0, x1 ,y: standardize(x0, x1, y,
                                              xy_mean, xy_std))

    # We need all these elements combined, not as a set of three tuples.
    @tf.function
    def combine(x0,
                x1,
                y):

        """
        Args:
            x0, x1, y: elements in the dataset

        Returns:
            [x0, x1, y]: a combined element
        """

        return tf.stack([x0, x1, y],
                        axis=0)

    xy = xy.map(combine)

    # Batch THEN shuffle the dataset, and prefetch as appropriate.
    xy = xy.batch(batch_size)
    xy = xy.shuffle(buffer_size=num_batches)
    xy = xy.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return xy

###############################################################################

def make_mixed_dataset(which_classes,
                       num_batches_per_class,
                       batch_size):

    """
    Makes a dataset of mixed shapes.

    Classes are:
        0: a circle
        1: a slash
        2: two Gaussian blobs
        3: a curve shaped like a sideways 3
        4: a uniform square
        5: a 3x3 grid of points
        6: two concentric circles
    They are all on the same order of size, but not exactly the same.
    Currently, they ARE all centered on the origin.

    The standard deviations for the Gaussian noise in each set are:
        circle: 0.05
        slash: 0.05
        blobs: 0.15
        three: 0.05
        square: N/A (its' a uniform distribution)
        grid: 0.05
        concentric circles: 0.05
    Additionally, the ratio of diameters between the inner and outer concentric circles is 0.6.

    Args:
        which_classes: a list of the desired classes 0-5, for example [0, 1, 4].
        num_batches_per_class: the number of batches per class in the dataset. We use this rather than the number of points per class to guarantee that each batch will contain ONLY examples from one class.
        batch_size: the batch size of the generated dataset (i.e. number of points, all from the same class, in a batch)

    Returns:
        xy: a TF dataset object with elements (x0, x1, y).
        x0, x1: the Cartesian coordinates of the point.
        y: the class label of the point.

        NOTE: the data is batched BEFORE being shuffled, since each class should map independently to N(0,1), rather than all classes together mapping to N(0,1) (which could lead to each class mapping to a sub-distribution).
    """

    # The noise std. devs. and the concentric circle factor.
    # NOTE: these are defined here for the numpy functions, but they are defined SEPARATELY INSIDE THE TF FUNCTIONS. Changing these values here will not change them in the final dataset! All it will do is screw up the standardization. If you want to change them in the dataset, you have to manually change them in the TF functions as well.
    circle_noise = 0.05
    slash_noise = 0.05
    blobs_noise = 0.15
    three_noise = 0.05
    ccirc_noise = 0.05
    ccirc_factor = 0.6

    # Calculate:
    # the number of points per class
    # the number of classes
    # the total number of points in the dataset (per epoch)
    # the total number of batches
    num_points_per_class = num_batches_per_class * batch_size
    num_classes = len(which_classes)
    num_points = num_classes * num_points_per_class
    num_batches = num_batches_per_class * num_classes

    # Make zero vectors of the correct shape
    y = np.zeros((num_points,))

    # For each class, put class labels in y'
    # NOTE: THESE ARE NOT THE SAME AS THE GENERAL CLASS LABELS! The reason for this is that we want our class labels to be evenly spaced in the traning data (e.g., [-1,0,1]) and an arbitrary set of classes (e.g. [0,1,4]) does NOT meet this criterion!
    for i in range(num_classes):
        y[i * num_points_per_class:
          (i+1) * num_points_per_class] = i

    # Convert to a NON-STANDARDIZED dataset object OF INDICES in int32
    y = y.astype('int32')
    y = tf.data.Dataset.from_tensor_slices(y)

    @tf.function
    def get_class_index(y):

        """
        Args:
            y: the (not-yet-standardized) pointMNIST class label on the X side.

        Returns:
            y_index = the actual classes to which the labels apply.

        For example, if
        which_classes = [0, 1, 4], then
        y = [0,0,...,0,
             1,1,...,1,
             2,2,...,2] and
        y_index = [0,0,...,0,
                   1,1,...,1,
                   4,4,...,4]
        """

        y_index = tf.gather(which_classes,
                            y)

        return y_index

    # Now we can get a dataset object containing the classes, separate from the class labels we will use in training. This will be useful for obtaining points in X from digit_images.
    y_ = y.map(get_class_index)

    # Zip these together for ease of processing later.
    yy_ = tf.data.Dataset.zip((y,y_))

    # Cache here, since the proportion of each class is fixed and equal
    yy_ = yy_.cache()

    ###########################################################################

    """
    Need to know the mean and standard deviation of the total dataset. As before, that means numpy arrays.
    """

    def np_circle(n_samples,
                  index):

        """
        Adaptation of sklearn.datasets.make_circles, but with only one circle. Changes from sklearn.datasets.make_circles are not documented.

        Arguments:
            n_samples : number of points in the circle.
            index: the INDEX of the class label, NOT the class label itself. For example, if which_classes=[0,2,5], index=0.

        Returns:
            XY, the concatenation of:
            X = (x0, x1) : ndarray of shape (n_samples_per, 2) with the Cartesian coordinates of the points.
            Y : ndarray of shape (n_samples_per, 1) with the class label.
        """

        # Make evenly spaced points around the circle
        linspace_circ = np.linspace(0,
                                    2*pi,
                                    n_samples,
                                    endpoint=False)

        # Convert to Cartesian coordinates
        circ_x0 = np.cos(linspace_circ)
        circ_x1 = np.sin(linspace_circ)

        X = np.vstack([circ_x0,
                       circ_x1]).T
        Y = index * np.ones(n_samples,
                            dtype=np.intp)

        # Add noise to the coordinates
        X += np.random.normal(loc=0,
                              scale=circle_noise,
                              size=X.shape)

        Y = np.expand_dims(Y,
                           axis=1)
        XY = np.concatenate((X,Y),
                            axis=1)

        return XY

    def np_slash(n_samples,
                 index):

        """
        Makes a noisy "slash"-shaped object.

        Arguments:
            n_samples : number of points in the slash.
            index: the INDEX of the class label, NOT the class label itself. For example, if which_classes=[1,3,5], index=0.

        Returns:
            XY, the concatenation of:
            X = (x0, x1) : ndarray of shape (n_samples_per, 2) with the Cartesian coordinates of the points.
            Y : ndarray of shape (n_samples_per, 1) with the class label.
        """

        # Make evenly spaced points from -1 to 1
        linspace_slash = np.linspace(-1,
                                     1,
                                     n_samples,
                                     endpoint=False)

        # Convert to Cartesian coordinates
        slash_x0 = linspace_slash
        slash_x1 = linspace_slash

        X = np.vstack([slash_x0,
                       slash_x1]).T
        Y = index * np.ones(n_samples,
                            dtype=np.intp)

        # Add noise to the coordinates
        X += np.random.normal(loc=0,
                              scale=slash_noise,
                              size=X.shape)

        Y = np.expand_dims(Y,
                           axis=1)
        XY = np.concatenate((X,Y),
                            axis=1)

        return XY

    def np_blobs(n_samples,
                 index):

        """
        Makes a pair of identical, non-overlapping Gaussian blobs.

        Arguments:
            n_samples : number of points in the blobs (total, not per blob).
            index: the INDEX of the class label, NOT the class label itself. For example, if which_classes=[0,2,5], index=1.

        Returns:
            XY, the concatenation of:
            X = (x0, x1) : ndarray of shape (n_samples_per, 2) with the Cartesian coordinates of the points.
            Y : ndarray of shape (n_samples_per, 1) with the class label.

        NOTE: if n_samples is not divisible by 2, the left blob will have one more point than the right.
        """

        n_samples_right = n_samples // 2
        n_samples_left = n_samples - n_samples_right

        x_left = np.random.normal(loc=[-0.5,
                                       0.5],
                                  scale=blobs_noise,
                                  size=(n_samples_left,
                                        2))
        x_right = np.random.normal(loc=[0.5,
                                        -0.5],
                                   scale=blobs_noise,
                                   size=(n_samples_right,
                                         2))

        X = np.concatenate((x_left,
                            x_right))


        Y = index * np.ones(n_samples,
                            dtype=np.intp)
        Y = np.expand_dims(Y,
                           axis=1)
        XY = np.concatenate((X,Y),
                            axis=1)

        return XY

    def np_three(n_samples,
                 index):

        """
        Makes a vaguely "3"-shaped object.

        Arguments:
            n_samples : number of points in the 3.
            index: the INDEX of the class label, NOT the class label itself. For example, if which_classes=[0,2,3], index=2.

        Returns:
            XY, the concatenation of:
            X = (x0, x1) : ndarray of shape (n_samples_per, 2) with the Cartesian coordinates of the points.
            Y : ndarray of shape (n_samples_per, 1) with the class label.
        """

        n_samples_right = n_samples // 2
        n_samples_left = n_samples - n_samples_right

        # Make evenly spaced points around the semicircle
        linspace_left = np.linspace(0,
                                    pi,
                                    n_samples_left,
                                    endpoint=False)
        linspace_right = np.linspace(0,
                                     pi,
                                     n_samples_right,
                                     endpoint=False)

        # Convert to Cartesian coordinates
        left_x0 = np.cos(linspace_left) + 1
        left_x1 = np.sin(linspace_left)
        right_x0 = np.cos(linspace_right) - 1
        right_x1 = np.sin(linspace_right)

        # Concatenate
        three_x0 = np.concatenate((left_x0,
                                   right_x0))
        three_x0 /= 2
        three_x1 = np.concatenate((left_x1,
                                   right_x1))
        three_x1 = three_x1 * 2 - 1
        X = np.vstack([three_x0,
                       three_x1]).T

        Y = index * np.ones(n_samples,
                            dtype=np.intp)

        # Add noise to the coordinates
        X += np.random.normal(loc=0,
                              scale=three_noise,
                              size=X.shape)

        Y = np.expand_dims(Y,
                           axis=1)
        XY = np.concatenate((X,Y),
                            axis=1)

        return XY

    def np_square(n_samples,
                  index):

        """
        Makes a uniform square.

        Arguments:
            n_samples : number of points in the square.
            index: the INDEX of the class label, NOT the class label itself. For example, if which_classes=[0,2,4], index=2.

        Returns:
            XY, the concatenation of:
            X = (x0, x1) : ndarray of shape (n_samples_per, 2) with the Cartesian coordinates of the points.
            Y : ndarray of shape (n_samples_per, 1) with the class label.
        """

        X = np.random.uniform(low=-1,
                              high=1,
                              size=(n_samples,
                                    2))
        Y = index * np.ones(n_samples,
                            dtype=np.intp)
        Y = np.expand_dims(Y,
                           axis=1)
        XY = np.concatenate((X,Y),
                            axis=1)

        return XY

    def np_grid(n_samples,
                index):

        """
        Makes a 3x3 square grid of small blobs.

        Arguments:
            n_samples : number of points in the blobs (total, not per blob).
            index: the INDEX of the class label, NOT the class label itself. For example, if which_classes=[0,2,5], index=2.

        Returns:
            XY, the concatenation of:
            X = (x0, x1) : ndarray of shape (n_samples_per, 2) with the Cartesian coordinates of the points.
            Y : ndarray of shape (n_samples_per, 1) with the class label.

        NOTE: if n_samples is not divisible by 9, using the following grid,
            123
            456
            789
            the missing points will be taken from 9 first, then 8, etc.
        """

        n_9 = n_samples // 9
        n_8 = (n_samples - n_9) // 8
        n_7 = (n_samples - n_9 - n_8) // 7
        n_6 = (n_samples - n_9 - n_8 - n_7) // 6
        n_5 = (n_samples - n_9 - n_8 - n_7 - n_6) // 5
        n_4 = (n_samples - n_9 - n_8 - n_7 - n_6 - n_5) // 4
        n_3 = (n_samples - n_9 - n_8 - n_7 - n_6 - n_5 - n_4) // 3
        n_2 = (n_samples - n_9 - n_8 - n_7 - n_6 - n_5 - n_4 - n_3) // 2
        n_1 = n_samples - n_9 - n_8 - n_7 - n_6 - n_5 - n_4 - n_3 - n_2

        x_11 = np.random.normal(loc=[-0.8,
                                     0.8],
                                scale=0.05,
                                size=(n_1,
                                      2))
        x_12 = np.random.normal(loc=[0,
                                     0.8],
                                scale=0.05,
                                size=(n_2,
                                      2))
        x_13 = np.random.normal(loc=[0.8,
                                     0.8],
                                scale=0.05,
                                size=(n_3,
                                      2))

        x_21 = np.random.normal(loc=[-0.8,
                                     0],
                                scale=0.05,
                                size=(n_4,
                                      2))
        x_22 = np.random.normal(loc=[0,
                                     0],
                                scale=0.05,
                                size=(n_5,
                                      2))
        x_23 = np.random.normal(loc=[0.8,
                                     0],
                                scale=0.05,
                                size=(n_6,
                                      2))

        x_31 = np.random.normal(loc=[-0.8,
                                     -0.8],
                                scale=0.05,
                                size=(n_7,
                                      2))
        x_32 = np.random.normal(loc=[0,
                                     -0.8],
                                scale=0.05,
                                size=(n_8,
                                      2))
        x_33 = np.random.normal(loc=[0.8,
                                     -0.8],
                                scale=0.05,
                                size=(n_9,
                                      2))

        X = np.concatenate((x_11, x_12, x_13,
                            x_21, x_22, x_23,
                            x_31, x_32, x_33))


        Y = index * np.ones(n_samples,
                            dtype=np.intp)
        Y = np.expand_dims(Y,
                           axis=1)
        XY = np.concatenate((X,Y),
                            axis=1)

        return XY

    def np_concentric_circles(n_samples,
                              index):

        """
        Adaptation of sklearn.datasets.make_circles, now with two circles. Changes from sklearn.datasets.make_circles are not documented.

        Arguments:
            n_samples : number of points in the circles (total, not per circle).
            index: the INDEX of the class label, NOT the class label itself. For example, if which_classes=[0,2,6], index=2.

        Returns:
            XY, the concatenation of:
            X = (x0, x1) : ndarray of shape (n_samples_per, 2) with the Cartesian coordinates of the points.
            Y : ndarray of shape (n_samples_per, 1) with the class label.

        NOTE: if n_samples is not divisible by 2, the inner circle will have one more point than the outer circle.
        """

        if ccirc_factor >= 1 or ccirc_factor < 0:
            raise ValueError("'factor' has to be between 0 and 1.")

        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out

        # Make evenly spaced points around the circle
        linspace_out = np.linspace(0,
                                   2*pi,
                                   n_samples_out,
                                   endpoint=False)
        linspace_in = np.linspace(0,
                                  2*pi,
                                  n_samples_in,
                                  endpoint=False)

        # Convert to Cartesian coordinates
        out_x0 = np.cos(linspace_out)
        out_x1 = np.sin(linspace_out)

        in_x0 = np.cos(linspace_in) * ccirc_factor
        in_x1 = np.sin(linspace_in) * ccirc_factor

        X = np.vstack([np.append(out_x0, in_x0),
                       np.append(out_x1, in_x1)]).T

        # Add noise to the coordinates
        X += np.random.normal(loc=0,
                              scale=ccirc_noise,
                              size=X.shape)

        Y = index * np.ones(n_samples,
                            dtype=np.intp)
        Y = np.expand_dims(Y,
                           axis=1)
        XY = np.concatenate((X,Y),
                            axis=1)

        return XY

    ##########################################################################
    # Remember, all of that was just the numpy part, for getting the mean and standard deviation!
    # Now, we have to actually calculate those.
    ##########################################################################

    # Make the appropriate-shape zero vector (with 10000 points per class)...
    np_xy = np.zeros((10000 * num_classes,
                      3))

    # ...fill it in...
    for i in range(num_classes):

        if which_classes[i] == 0:
            np_xy[10000 * i : 10000 * (i+1)] \
                = np_circle(10000, i)
        elif which_classes[i] == 1:
            np_xy[10000 * i : 10000 * (i+1)] \
                = np_slash(10000, i)
        elif which_classes[i] == 2:
            np_xy[10000 * i : 10000 * (i+1)] \
                = np_blobs(10000, i)
        elif which_classes[i] == 3:
            np_xy[10000 * i : 10000 * (i+1)] \
                = np_three(10000, i)
        elif which_classes[i] == 4:
            np_xy[10000 * i : 10000 * (i+1)] \
                = np_square(10000, i)
        elif which_classes[i] == 5:
            np_xy[10000 * i : 10000 * (i+1)] \
                = np_grid(10000, i)
        elif which_classes[i] == 6:
            np_xy[10000 * i : 10000 * (i+1)] \
                = np_concentric_circles(10000, i)

    # ...and finally, get the mean and standard deviation.
    xy_mean = np.mean(np_xy,
                      axis=0,
                      dtype=np.float32)
    xy_std = np.std(np_xy,
                    axis=0,
                    dtype=np.float32)

    ###########################################################################
    # Define the TensorFlow functions that will be used to generate the data itself.
    ###########################################################################

    @tf.function
    def tf_make_mixed_point(y,
                            y_):

        """
        Args:
            y: an element denoting class label.
            y_: an element denoting the shape label.

            For example, if which_classes is [0, 1, 4], y would be one of [0, 1, 2] and y_ would be the corresponding one of [0, 1, 4].

        Returns:
            x0, x1, y: an element containing Cartesian coordinates and the class label of a generated point corresponding to the input class label.
        """

        # This is for a single circle.
        @tf.function
        def tf_make_circle_point():

            circle_noise = 0.05

            angle = tf.random.uniform(shape=[],
                                      minval=0,
                                      maxval=2*pi)

            x0 = tf.math.cos(angle)
            x1 = tf.math.sin(angle)

            x0 += tf.random.normal(shape=[],
                                   mean=0.0,
                                   stddev=circle_noise,
                                   dtype=tf.dtypes.float32)
            x1 += tf.random.normal(shape=[],
                                   mean=0.0,
                                   stddev=circle_noise,
                                   dtype=tf.dtypes.float32)

            return x0, x1

        # This is for a slash.
        @tf.function
        def tf_make_slash_point():

            slash_noise = 0.05

            line = tf.random.uniform(shape=[],
                                     minval=-1,
                                     maxval=1)

            x0 = line
            x1 = line

            x0 += tf.random.normal(shape=[],
                                   mean=0.0,
                                   stddev=slash_noise,
                                   dtype=tf.dtypes.float32)
            x1 += tf.random.normal(shape=[],
                                   mean=0.0,
                                   stddev=slash_noise,
                                   dtype=tf.dtypes.float32)

            return x0, x1

        # This is for two blobs.
        @tf.function
        def tf_make_blobs_point():

            blobs_noise = 0.15

            # Need to randomly choose which blob the point is in.
            sign = tf.random.uniform(shape=[],
                                     minval=0,
                                     maxval=2,
                                     dtype=tf.dtypes.int32)
            sign = 2 * sign - 1
            sign = tf.cast(sign,
                           tf.float32)

            x0 = -0.5 * sign
            x1 = 0.5 * sign

            x0 += tf.random.normal(shape=[],
                                   mean=0.0,
                                   stddev=blobs_noise,
                                   dtype=tf.dtypes.float32)
            x1 += tf.random.normal(shape=[],
                                   mean=0.0,
                                   stddev=blobs_noise,
                                   dtype=tf.dtypes.float32)

            return x0, x1

        # This is for the "three."
        @tf.function
        def tf_make_three_point():

            three_noise = 0.05

            # Need to randomly choose which arch the point is in.
            which = tf.random.uniform(shape=[],
                                      minval=0,
                                      maxval=2,
                                      dtype=tf.dtypes.int32)
            which = 2 * which - 1
            which = tf.cast(which,
                            dtype=tf.float32)

            angle = tf.random.uniform(shape=[],
                                      minval=0,
                                      maxval=pi)

            x0 = tf.math.cos(angle) + which
            x0 = x0 / 2
            x1 = tf.math.sin(angle)
            x1 = x1 * 2 - 1

            x0 += tf.random.normal(shape=[],
                                   mean=0.0,
                                   stddev=three_noise,
                                   dtype=tf.dtypes.float32)
            x1 += tf.random.normal(shape=[],
                                   mean=0.0,
                                   stddev=three_noise,
                                   dtype=tf.dtypes.float32)

            return x0, x1

        # This is for the uniform square.
        @tf.function
        def tf_make_square_point():

            x0 = tf.random.uniform(shape=[],
                                   minval=-1,
                                   maxval=1,
                                   dtype=tf.dtypes.float32)
            x1 = tf.random.uniform(shape=[],
                                   minval=-1,
                                   maxval=1,
                                   dtype=tf.dtypes.float32)

            return x0, x1

        # This is for the grid.
        @tf.function
        def tf_make_grid_point():

            grid_noise = 0.05

            # Choose which of the 9 grid clusters to put the point in.
            which_0 = tf.random.uniform(shape=[],
                                        minval=-1,
                                        maxval=2,
                                        dtype=tf.dtypes.int32)
            which_1 = tf.random.uniform(shape=[],
                                        minval=-1,
                                        maxval=2,
                                        dtype=tf.dtypes.int32)

            which_0 = tf.cast(which_0,
                              dtype=tf.float32)
            which_1 = tf.cast(which_1,
                              dtype=tf.float32)

            x0 = 0.8 * which_0
            x1 = 0.8 * which_1

            x0 += tf.random.normal(shape=[],
                                   mean=0.0,
                                   stddev=grid_noise,
                                   dtype=tf.dtypes.float32)
            x1 += tf.random.normal(shape=[],
                                   mean=0.0,
                                   stddev=grid_noise,
                                   dtype=tf.dtypes.float32)

            return x0, x1

        # This is for the concentric circles.
        @tf.function
        def tf_make_ccirc_point():

            ccirc_noise = 0.05
            ccirc_factor = 0.6

            # Get the angle.
            angle = tf.random.uniform(shape=[],
                                      minval=0,
                                      maxval=2*pi)

            # Decide whether or not to multiply by the factor.
            which = tf.random.uniform(shape=[],
                                      minval=0,
                                      maxval=2,
                                      dtype=tf.dtypes.int32)

            # Need to define CALLABLE functions for the sin/cos of the angles.
            @tf.function
            def cos_0():
                return tf.math.cos(angle)

            @tf.function
            def sin_0():
                return tf.math.sin(angle)

            @tf.function
            def cos_1():
                return tf.math.cos(angle) * ccirc_factor

            @tf.function
            def sin_1():
                return tf.math.sin(angle) * ccirc_factor

            x0 = tf.switch_case(which,
                        branch_fns={0 : cos_0,
                                    1 : cos_1})
            x1 = tf.switch_case(which,
                        branch_fns={0 : sin_0,
                                    1 : sin_1})

            x0 += tf.random.normal(shape=[],
                                   mean=0.0,
                                   stddev=ccirc_noise,
                                   dtype=tf.dtypes.float32)
            x1 += tf.random.normal(shape=[],
                                   mean=0.0,
                                   stddev=ccirc_noise,
                                   dtype=tf.dtypes.float32)

            return x0, x1

        # Remember, yy_ looks like (y, y_), where y is the evenly-spaced class label that will be used for training, and y_ is the index denoting which shape the point is generated from. (These will only be the same if no shapes are skipped.)
        # Pick which of the above functions to evaluate, depending on which shape we want to generate a point for.
        x0, x1 = tf.switch_case(y_,
                                branch_fns={0 : tf_make_circle_point,
                                            1 : tf_make_slash_point,
                                            2 : tf_make_blobs_point,
                                            3 : tf_make_three_point,
                                            4 : tf_make_square_point,
                                            5 : tf_make_grid_point,
                                            6 : tf_make_ccirc_point})

        # Finally, return the desired final element.
        return x0, x1, y

    # Now we generate points from yy_:
    xy = yy_.map(tf_make_mixed_point)

    # Define the function that will actually do the dataset standardization.
    @tf.function
    def standardize(x0,
                    x1,
                    y,
                    xy_mean,
                    xy_std):

        """
        Standardizes (mean=0, std. dev.=1) the data in the generated dataset. The mean and std. dev. values used to perform this standardization are obtained from a single generated numpy dataset with many (10,000) points per class.

        Args:
            x0, x1, y: a single (x,y) element (x0, x1,y)
            NOTE: y is the class label, not the index of the chosen shape. If some shapes are skipped, the class label and the shape label will not match. This is so the class labels are evenly spaced.
        Returns:
            x0, x1, y: the same element, but with zero mean, 1 std. dev.
        """

        # Convert y from int
        y = tf.cast(y,
                    dtype=tf.float32)

        x0 = (x0 - xy_mean[0]) / xy_std[0]
        x1 = (x1 - xy_mean[1]) / xy_std[1]
        y = (y - xy_mean[2]) / xy_std[2]

        return x0, x1, y

    # Standardize the data.
    xy = xy.map(lambda x0, x1 ,y: standardize(x0, x1, y,
                                              xy_mean, xy_std))

    # We need all these elements combined, not as a set of three tuples.
    @tf.function
    def combine(x0,
                x1,
                y):

        """
        Args:
            x0, x1, y: elements in the dataset

        Returns:
            [x0, x1, y]: a combined element
        """

        return tf.stack([x0, x1, y],
                        axis=0)

    xy = xy.map(combine)

    # Batch BEFORE shuffling.
    xy = xy.batch(batch_size)

    # Shuffle and prefetch.
    xy = xy.shuffle(buffer_size = num_batches)
    xy = xy.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return xy

###############################################################################

def make_continuous_sectors(num_points,
                            batch_size,
                            sector_width):

    """
    Args:
        num_points: the number of points generated in the dataset per call.
        batch_size: the batch size of the generated dataset.
        sector_angle: the angular size of the continuously generated sectors.

    Returns:
        xy: a TF dataset object. Note that UNLIKE THE DISCRETE CLASS DATASETS DEFINED ABOVE, the data is batched AFTER being shuffled, since each there are no longer discrete classes!
        y is generated randomly for each point from values between [0,2*pi).
        x=(x1,x2) describes the Cartesian coordinates of a point drawn with uniform probability from a sector of the unit circle centered on angle y, with angular width sector_width.
    """

    y = np.zeros((num_points,),
                 dtype=np.float32)
    y = tf.data.Dataset.from_tensor_slices(y)
    y = y.cache()

    @tf.function
    def make_csec(sector_width):

        """
        Args:
            sector_width: the width of the sector.

        Returns:
            x0, x1, y: an element containing Cartesian coordinates and the angle between [0, 2*pi) of a point randomly selected (with uniform probability) from a sector with randomly chosen central angle.
        """

        # Get the central angle.
        y = tf.random.uniform(shape=[],
                              minval=0,
                              maxval=2*pi,
                              dtype=tf.dtypes.float32)

        # Get the angular region covered by the sector.
        min_angle = y - sector_width/2
        max_angle = y + sector_width/2

        angle = tf.random.uniform(shape=[],
                                  minval=min_angle,
                                  maxval=max_angle)

        # Get the radius.
        radius = tf.random.uniform(shape=[],
                                   minval=0,
                                   maxval=1)
        radius = radius ** 0.5

        # Get the coordinates.
        x0 = radius * tf.math.cos(angle)
        x1 = radius * tf.math.sin(angle)

        x0 = tf.cast(x0,
                     dtype=tf.dtypes.float32)
        x1 = tf.cast(x1,
                     dtype=tf.dtypes.float32)

        return x0, x1, y

    # Don't need to standardize because the points will always be evenly distributed through the unit circle.
    # No need to shuffle it, either, because it is already completely randomly generated as it is.

    xy = y.map(lambda k : make_csec(sector_width))

    # We need all these elements combined, not as a set of three tuples.
    @tf.function
    def combine(x0,
                x1,
                y):

        """
        Args:
            x0, x1, y: elements in the dataset

        Returns:
            [x0, x1, y]: a combined element
        """

        return tf.stack([x0, x1, y],
                        axis=0)

    xy = xy.map(combine)

    # Batch THEN shuffle the dataset, and prefetch as appropriate.
    xy = xy.batch(batch_size)
    xy = xy.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return xy

def make_discrete_sectors(num_points_per_sector,
                          which_sectors,
                          sector_width):

    """
    Args:
        num_points_per_sector: the number of points generated in the dataset per sector per call. Also the batch size.
        which_sectors: a list of y values to generate sectors from.
        sector_angle: the angular size of the generated sectors.

    Returns:
        xy: a TF dataset object. Note that here, the batches are per sector!
        y is defined by which_sectors.
        x=(x1,x2) describes the Cartesian coordinates of a point drawn with uniform probability from a sector of the unit circle centered on angle y, with angular width sector_width.
    """

    y = np.zeros((num_points_per_sector * len(which_sectors),),
                 dtype=np.float32)

    for i in range(len(which_sectors)):
        y[i*num_points_per_sector:
          (i+1)*num_points_per_sector] = which_sectors[i]

    y = tf.data.Dataset.from_tensor_slices(y)
    y = y.cache()

    @tf.function
    def make_dsec(y,
                  sector_width):

        """
        Args:
            y: the angle of the sector center.
            sector_width: the width of the sector.

        Returns:
            x0, x1, y: an element containing Cartesian coordinates and the angle, selected with uniform probability from between [y-sector_width, y+sector_width).
        """

        # Get the angular region covered by the sector.
        min_angle = y - sector_width/2
        max_angle = y + sector_width/2

        angle = tf.random.uniform(shape=[],
                                  minval=min_angle,
                                  maxval=max_angle)

        # Get the radius.
        radius = tf.random.uniform(shape=[],
                                   minval=0,
                                   maxval=1)
        radius = radius ** 0.5

        # Get the coordinates.
        x0 = radius * tf.math.cos(angle)
        x1 = radius * tf.math.sin(angle)

        x0 = tf.cast(x0,
                     dtype=tf.dtypes.float32)
        x1 = tf.cast(x1,
                     dtype=tf.dtypes.float32)

        return x0, x1, y

    # Don't need to standardize because the points will always be evenly distributed through the unit circle.
    # No need to shuffle it, either, because it is already completely randomly generated as it is.

    xy = y.map(lambda y : make_dsec(y,
                                    sector_width))

    # We need all these elements combined, not as a set of three tuples.
    @tf.function
    def combine(x0,
                x1,
                y):

        """
        Args:
            x0, x1, y: elements in the dataset

        Returns:
            [x0, x1, y]: a combined element
        """

        return tf.stack([x0, x1, y],
                        axis=0)

    xy = xy.map(combine)

    # Batch and prefetch as appropriate.
    xy = xy.batch(num_points_per_sector)
    xy = xy.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return xy

###############################################################################

"""
ADDING INSTANCE NOISE
"""

@tf.function
def instance_noise(xy_element,
                   alpha):

    """
    Args:
        xy_element: [x0, x1, y] (elements in the dataset)
        alpha: relative amount of signal in the output

    Returns:
        xy_element_noisy: mapped from xy -> alpha * xy + (1 - alpha) * noise
        where noise is drawn from N(0,1).

        Remember that since the data have already been standardized (across the dataset, NOT within batches) we already know that we want mean=0, std.dev.=1, and N(0,1) is our p_Z(z), AND we have standardized y as well.
    """

    x1,x2,y = tf.unstack(xy_element,axis=0)
    noise_1 = tf.random.normal(shape=tf.shape(x1),mean=0,stddev=1)
    noise_2 = tf.random.normal(shape=tf.shape(x2),mean=0,stddev=1)
    x1_noisy = alpha * x1 + (1 - alpha) * noise_1
    x2_noisy = alpha * x2 + (1 - alpha) * noise_2
    xy_element_noisy = tf.stack([x1_noisy,x2_noisy,y])

    return xy_element_noisy