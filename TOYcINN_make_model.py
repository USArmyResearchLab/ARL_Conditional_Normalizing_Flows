#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:18:17 2021

@author: John S Hyatt
"""

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import (Model,
                              metrics,
                              regularizers)

from tensorflow.keras.layers import (Input,
                                     Activation,
                                     LeakyReLU,
                                     Dense)

import numpy as np

###############################################################################

"""
DEFINE THE AFFINE COUPLING LAYER
"""

def coupling_layer(u1_size,
                   u2_size,
                   intermediate_dims,
                   num_layers):

    """
    Args:
        u1_size: number of dimensions of points in u1 and v1 (they have the same dimensionality).
        u2_size: number of dimensions of points in u2 and v2 (they have the same dimensionality).
        NOTE: dim(u) = dim(u1) + dim(u2) = dim(v), with dim(v1) = dim(u1) and dim(v2) = dim(u2).
        intermediate_dims: number of nodes per layer in the dense neural networks representing the functions A(.) and b(.), which themselves represent the affine transform v2 = A(u1)u2 + b(u1).
        num_layers: number of dense layers per block in A(.) and b(.).
        NOTE: I use lower triangular rather than upper because tf.linalg has only lower triangular linear operators.

    Returns:
        A TensorFlow model with inputs of shape u1_size and outputs [A(.), b(.)], where A(.) defines the trasnformation matrix and b(.) the translation vector, both with `u2_size` elements.
    """

    # Input to the coupling layer
    layer_input = Input(shape=u1_size,
                        dtype=tf.float32)

    # TRANSLATION
    b_block = Dense(intermediate_dims,
                    kernel_regularizer=regularizers.L1L2()
                    )(layer_input)
    b_block = LeakyReLU()(b_block)

    for i in range(num_layers):
        b_block = Dense(intermediate_dims,
                        kernel_regularizer=regularizers.L1L2()
                        )(b_block)
        b_block = LeakyReLU()(b_block)

    # The final layer in the block has a /linear/ activation, since we don't want to restrict it to positive outputs only.
    b_block = Dense(u2_size,
                    kernel_regularizer=regularizers.L1L2()
                    )(b_block)

    # SCALING
    A_block = Dense(intermediate_dims,
                    kernel_regularizer=regularizers.L1L2()
                    )(layer_input)
    A_block = LeakyReLU()(A_block)

    for i in range(num_layers):
        A_block = Dense(intermediate_dims,
                        kernel_regularizer=regularizers.L1L2()
                        )(A_block)
        A_block = LeakyReLU()(A_block)

    # We can no longer use the RealNVP trick with log/exp to calculate det J. I still use tanh activation because I'm more worried about A having exploding/vanishing gradient problems than I am b.

    # This version for a diagonal scale matrix
    A_block = Dense(u2_size,
                    kernel_regularizer=regularizers.L1L2()
                    )(A_block)

    # tanh activation keeps |det A| from being larger than 1.
# =============================================================================
#   # IMPORTANT NOTE:
# =============================================================================
    # NOTE: in the original paper they multiplied this by a learned scale factor to allow volume increases!
    # ***I have not found this to be necessary in my own toy experiments, so I've left it out. However, other use cases may require it!***
    A_block = Activation('tanh')(A_block)

    return Model(inputs=layer_input,
                 outputs=[A_block,
                          b_block])

###############################################################################

"""
DEFINE A CLASS FOR REAL NVP THAT INHERITS FROM keras.Model
"""

class cINN_affine(Model):

    """
    Args:
        io_shape: the number of input/output dimensions for each coupling layer (always the same; equal to the dimensionality of xy).
        x_d: the dimensionality of the x component. Necessary for splitting x and y'.
        num_coupling_layers: the number of coupling layers. Coupling layers are grouped in sixes, one for each of the masks [1,0,0], [0,1,0], [0,0,1], [0,1,1], [1,0,1], and [1,1,0]. The coupling layers are either ordered randomly or specified by mask_indices.
        intermediate_dims: the number of nodes per dense layer.
        num_layers: the number of dense layers per coupling layer.
        init: which initializer to use when randomly initializing the model.
        mask_indices: The order in which the masks will be applied.

    Returns:
        The dense cINN model specified.
    """

    def __init__(self,
                 io_shape,
                 x_d,
                 num_coupling_layers,
                 intermediate_dims,
                 num_layers,
                 init,
                 mask_indices=None):

        super(cINN_affine,
              self).__init__()

        self.io_shape = io_shape
        self.x_d = x_d
        self.num_coupling_layers = num_coupling_layers
        self.intermediate_dims = intermediate_dims
        self.num_layers = num_layers
        self.init = init
        # NOTE: self.mask_indices is defined below.

        # Weight for the loss term ||y-y'||_1
        # In my experience, with discrete y, the model learns almost before the end of the first epoch. With continuous y, it takes considerably longer (but does gradually improve until it correspondence is very good).
        # As a result, I think lambda_y = 100 is a good value and don't think (at least right now) it needs to be an adjustable parameter.
        self.lambda_y = 100

        # Distribution of the latent space: p_Z(z) = N(0,1), with dim(z)=2
        self.distribution = tfp.distributions.MultivariateNormalDiag(
                                loc=[0]*self.x_d,
                                scale_diag=[1]*self.x_d)

        # Masks: all 6 possible combinations of 3-element masks that aren't [0,0,0] or [1,1,1]. If for some reason you want to make a dense cINN with higher dimensionality, you will want to alter the code to generate the masks procedurally.

        # These masks remove elements not in u_1.
        self.mask_dict_1 = {0 : np.array([0]),
                            1 : np.array([1]),
                            2 : np.array([2]),
                            3 : np.array([0,1]),
                            4 : np.array([0,2]),
                            5 : np.array([1,2])}
        # These masks remove elements not in u_2.
        self.mask_dict_2 = {0 : np.array([1,2]),
                            1 : np.array([0,2]),
                            2 : np.array([0,1]),
                            3 : np.array([2]),
                            4 : np.array([1]),
                            5 : np.array([0])}

        # To actually generate the masks (which are matrices) we need to transform the square identity matrix into a pair of rectangular matrices, whose nonzero elements correspond to the unmasked variables in the two sets of masks.
        identity = np.identity(io_shape,
                               dtype=np.float32)

        self.masks_1 = {}
        self.masks_2 = {}

        # Each mask corresponds to a different coupling layer.
        if mask_indices:
            self.mask_indices = mask_indices
        else:
            self.mask_indices = np.arange(num_coupling_layers,
                                          dtype=np.int32)

        # Note that only 6 masks are possible with 3 dims. If we have >6 coupling layers, we need to repeat some masks.
        for i in self.mask_indices:
            self.masks_1[i] = identity[self.mask_dict_1[i % 6]]
            self.masks_2[i] = identity[self.mask_dict_2[i % 6]]

        # Need the number of dimensions in u1 and u2.
        self.dims_u1 = np.zeros_like(self.mask_indices,
                                     dtype=np.int32)
        self.dims_u2 = np.zeros_like(self.mask_indices,
                                     dtype=np.int32)

        for i in self.mask_indices:
            self.dims_u1[i] = np.sum(self.masks_1[i],
                                     dtype=np.int32)
            self.dims_u2[i] = np.sum(self.masks_2[i],
                                     dtype=np.int32)

        # Need a list of coupling layers
        self.coupling_layers_list = [coupling_layer(int(self.dims_u1[i]),
                                                    int(self.dims_u2[i]),
                                                    intermediate_dims,
                                                    num_layers)
                                     for i in range(num_coupling_layers)]

        # Shuffle the masks.
        if not mask_indices:
            mask_indices_shuffler = \
                np.array([self.mask_indices[6*i:6*(i+1)] \
                          for i in range(num_coupling_layers // 6)])

            for i in range(num_coupling_layers // 6):
                np.random.shuffle(mask_indices_shuffler[i])

            mask_indices_shuffler = mask_indices_shuffler.flatten()

            self.mask_indices = mask_indices_shuffler

        # We're customizing the behavior of `model.fit()`. That means creating a Metric instance to track loss (in this case, Mean).
        self.loss_tracker = metrics.Mean(name='loss')
        self.z_loss_tracker = metrics.Mean(name='z_loss')
        self.y_loss_tracker = metrics.Mean(name='y_loss')
        self.detJ_loss_tracker = metrics.Mean(name='detJ_loss')

    @property
    def metrics(self):
        """
        A list of the model's metrics.
            This model is trained by maximum likelihood using the change of variables from XY'-space to ZY-space. The joint probability distribution in data space can be written as:
            p_{XY'}(x,y') = p_{ZY}(z,y)|det J_f(x,y')|
            where the change of variables is given by the function
            f : XY' -> ZY
            with z = f_Z(x,y') and y = f_Y(x,y').
            Skipping over a bunch of steps (see the pdf) we can rewrite this as
            -log p_{XY'}(x,y') = -log p_Z(f_Z(x,y'))
                                 +lambda_Y ||f_Y(x,y') - y'||_1
                                 -log |det J_f(x,y')|
            Maximum likelihood = "minimizing the distance between the data distribution and the model distribution." The easiest way to think about this is that we are trying to maximize the probability that (f_Z(x,y'), f_Y(x,y')) "could have plausibly been drawn from" the joint distribution p_{ZY}.
            Setting this as part of `model.metrics` allows `fit()` to `reset()` the loss tracker at the start of each epoch of training, and `evaluate()` to `reset()` the loss tracker at the start of each `evaluate()` call. `fit()`, `evaluate()`, and `reset()` are all TF functions.
        """

        # The actual loss being tracked is defined below.
        return [self.loss_tracker,
                self.z_loss_tracker,
                self.y_loss_tracker,
                self.detJ_loss_tracker]

    def call(self,
             u,
             direction=-1):

        """
        Args:
            u: either xy' (direction=-1) or zy (direction=1).
            direction: -1 for mapping from xy'->zy, +1 for mapping from zy->xy'.

        Returns:
            v: either zy (direction=-1) or xy' = (direction=1).
            Note: (z,y) = f^-1(x,y') (direction -1) or
                  (x,y') = f(z,y) (direction 1)
            log_detJ_inv: the log of the determinant of the inverse of the Jacobian associated with the map. Because f is bijective, the determinant of J^-1_f is equal to the determinant of J_{f^-1}. Furthermore, because det J is a product of exponential functions, which are always positive, |det J| = det J.
        """

        """
        The training objective of this conditional xy'<->zy bijective map is similar to that of the unconditional x<->z bijective map.

        For the unconditional version, you use the negative log likelihood of x, -log p_X(x), given by the change of variables formula:
            p_X(x) = p_Z(z) * |det J_f(x)|, with z=f(x)
            -> -log p_X(x) = -log p_Z(f(x)) - log |det J_f(x)|
            By construction, in RealNVP det J is easy to calculate. We know p_Z(f(x)) exactly, because p_Z(z)=N(0,1). Therefore, obtaining the total loss is basically trivial.

        For the conditional version, make the following changes:
            p_X(x) -> p_{XY'}(x,y')
            p_Z(z) -> p_{ZY}(z,y) with (z,y) = f(x,y') = (f_Z(x,y'),f_Y(x,y'))
            J_f(x) -> J_f(x,y')

        See the paper for more details.

        NOTE: y' is NOT independent of x. Specifically, x is dependent on y, and if y'=y, it cannot be the case that x is not dependent on y' (and vice versa). This makes sense, if you think about it: suppose your classes are very different from one another. Then there will be areas in x-space that class A might occupy, but class B never will, and vice versa. Thus, the class (y') partially determines x.

        At the end of the day, we have our final training objective as:
            -log p_{XY'}(x,y') = -log p_Z(f_Z(x,y'))
                                 + lambda_Y ||f_Y(x,y') - y'||_1
                                 - log |det J_f(x,y')|
        (Again, see the paper for details. What matters is that this is not really much more complicated then the log likelihood for the unconditional case.)
        """

        # Since we're summing these terms, set it to 0 and add terms as we stack the coupling layers.
        log_detJ = 0

        # xy' (zy) is only the input at the top (bottom) coupling block.
        # zy (xy') is only the output at the bottom (top) coupling block.
        # At intermediate coupling layers, the input of a coupling layer is the output of the previous coupling layer. However, they will all have the same dimensionality.
        # Pass the input through the coupling layers (from bottom to top if direction=-1, from top to bottom if direction=+1)
        for i in range(self.num_coupling_layers)[::direction]:

            j = self.mask_indices[i]

            # u_1 = mask_1 * u
            # u_2 = mask_2 * u
            # u = transpose(mask_1) * u_1 + transpose(mask_2) * u_2

            mask_1 = self.masks_1[j]
            mask_2 = self.masks_2[j]

            # For whatever reason you can't transpose linear operators in tf.linalg, so we have to define the transpose masks separately before converting them.
            mask_1_T = tf.transpose(mask_1)
            mask_2_T = tf.transpose(mask_2)

            # Convert the masks to linear operators.
            mask_1 = tf.linalg.LinearOperatorFullMatrix(mask_1)
            mask_2 = tf.linalg.LinearOperatorFullMatrix(mask_2)

            mask_1_T = tf.linalg.LinearOperatorFullMatrix(mask_1_T)
            mask_2_T = tf.linalg.LinearOperatorFullMatrix(mask_2_T)

            # Remove the u_2 elements from u to get u_1 and vice versa.
            u_1 = mask_1.matvec(u)
            u_2 = mask_2.matvec(u)

            """
            Note on u and v and masks:
                v = (M1^T)*u1 + (M2^T)*(A*u2 + b)
                u = (M1^T)*v1 + (M2^T)*(A^-1)*(v2 - b)

                with

                u1 = M1*u, u2 = M2*u
                v1 = M1*v, v2 = M2*v
                u = (M1^T)*u1 + (M2^T)*u2
                v = (M1^T)*v1 + (M2^T)*v2

            Further note that M1 and M2 are rectangular matrices made by splitting the identity matrix with size equal to the total dimensionality of the INN's input/output:
                M1*(M1^T) = Identity(dim(u1) x dim(u1))
                M2*(M2^T) = Identity(dim(u2) x dim(u2))
                (M1^T)*M1 + (M2^T)*M2 = Identity(dim(u) x dim(u))

            For example, for a 3-dimensional input/output, for one particular coupling layer where only the second input element is transformed, this might work out to:
                u = [q1
                     q2
                     q3]
                M1 = [1 0 0
                      0 0 1]
                M2 = [0 1 0]

                                 [1 0 0     [q1   [q1
                -> u1 = M1 * u =  0 0 1]  *  q2 =  q3]
                                             q3]
                and similarly, u2 = [q2].
            """

            # A(u1) and b(u1) are defined by the neural networks in the coupling layers.
            (A_u1,
             b_u1) = self.coupling_layers_list[j](u_1)

            # Take the exponent of A. A has a tanh activation at the end, so -1<A(u_1)<1. That means (1/e)<exp(A)<e.
            # Note that this is the element-wise exponent, not the matrix exponent.
            exp_A_u1 = tf.exp(A_u1)

            # dim(A) = N, with N = dim(u_2), and we can express it as a diagonal matrix.
            A_u1 = tf.linalg.LinearOperatorDiag(A_u1)
            exp_A_u1 = tf.linalg.LinearOperatorDiag(exp_A_u1)

            # Obtain u (or v).
            # First, transpose(M1)*u_1 is the same in both directions.
            M1T_u1 = mask_1_T.matvec(u_1)

            # Forward pass (zy->xy')
            if direction == 1:
                # Note that unlike RealNVP we cannot "cheat" to get the inverse. However, it doesn't really matter much, since we don't use this direction during training, only inference.
                # For higher-dimensional problems it would probably be a good idea to figure out how to calculate inverse(exp(A)) *outside* the model and insert it in, but that's a problem for someone else to deal with.
                # It helps that exp_A_u1 is explicitly defined to be a triangular matrix, which I *think* tf.linalg is smart enough to take advantage of when calculating the inverse.
                inv_exp_A_u1 = tf.linalg.inv(exp_A_u1)

                M2T_A_u2_b = inv_exp_A_u1.matvec(u_2 - b_u1)

            # Backward pass (xy'->zy)
            # This is the direction used to train the model!
            elif direction == -1:
                M2T_A_u2_b = exp_A_u1.matvec(u_2) + b_u1

                # TAKING LOG OF EXP IS NECESSARY!
                # If you replace the below line of code with:
                # log_detJ += tf.linalg.det(A_u1)
                # It DOES NOT WORK. It runs, but it creates nonsense results. exp_A_u1 is used to train, so it might be that there's a problem with using A_u1, which is defined before exp_A_u1, to get gradient information from the Jacobian.
                log_detJ += tf.math.log(
                                tf.linalg.det(exp_A_u1))
                # Above, det(...) is always positive, but if that were not the case, you would need an explicit absolute value:
# =============================================================================
#                 log_detJ += tf.math.log(
#                             tf.math.abs(
#                             tf.linalg.det(exp_A_u1)))
# =============================================================================

            M2T_A_u2_b = mask_2_T.matvec(M2T_A_u2_b)

            u = M1T_u1 + M2T_A_u2_b

        # Remember, v = zy (direction=-1) or v = xy' (direction=1).
        v = u

        return v, log_detJ

    def log_loss(self,
                 xy):

        """
        Args:
            xy: the concatenation of x, a point in the X domain, and y, a discrete class label or other conditioning information (including continuous y).

        Returns:
            log_loss: the negative log likelihood training objective
            log_loss = - log p_{X,Y'}(x,y')
                     = - log p_Z(f_Z(x,y'))
                       + lambda_Y * ||f_Y(x,y') - y'||_1
                       - log |det J_f(x,y')|
        """

        x_d = self.x_d

        # Need to know the original class labels.
        y_prime = xy[:,
                     x_d:]

        (zy,
         log_detJ) = self(xy,
                          -1)

        # Separate z and y.
        z = zy[:,
               :x_d]
        y = zy[:,
               x_d:]

        # distribution.log_prob(z) has shape (batch,)
        log_likelihood_z = self.distribution.log_prob(z)

        # y and y' have shapes (batch, y_depth)
        log_likelihood_y = -self.lambda_y * \
                            tf.math.reduce_sum(tf.abs(y-y_prime),
                                               axis=[1])

        log_likelihood = tf.math.reduce_mean(log_likelihood_z + \
                                             log_likelihood_y + \
                                             log_detJ)

        # Want the negative of this
        log_loss = -log_likelihood

        # Return the total loss as well as each component of it
        return log_loss, -tf.reduce_mean(log_likelihood_z), -tf.reduce_mean(log_likelihood_y), -tf.reduce_mean(log_detJ)

    def train_step(self,
                   xy):

        """
        Args:
            xy: (batch of) input data in the XY' domain

        Returns:
            The negative log likelihood, -log p_{X,Y'}(x,y')

        Updates the model's weights.
        """

        with tf.GradientTape() as tape:

            loss, loss_z, loss_y, loss_detJ = self.log_loss(xy)

        grads = tape.gradient(loss,
                              self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,
                                           self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.z_loss_tracker.update_state(loss_z)
        self.y_loss_tracker.update_state(loss_y)
        self.detJ_loss_tracker.update_state(loss_detJ)

        return {'loss': self.loss_tracker.result(),
                'z_loss' : self.z_loss_tracker.result(),
                'y_loss' : self.y_loss_tracker.result(),
                'detJ_loss' : self.detJ_loss_tracker.result()}

    def test_step(self,
                  xy):

        """
        Args:
            xy: (batch of) input data in the XY' domain

        Returns:
            the negative log_likelihood, -log p_{X,Y'}(x,y')

        Does NOT update the model's weights.
        """

        loss, loss_z, loss_y, loss_detJ = self.log_loss(xy)
        self.loss_tracker.update_state(loss)
        self.z_loss_tracker.update_state(loss_z)
        self.y_loss_tracker.update_state(loss_y)
        self.detJ_loss_tracker.update_state(loss_detJ)

        return {'loss': self.loss_tracker.result(),
                'z_loss' : self.z_loss_tracker.result(),
                'y_loss' : self.y_loss_tracker.result(),
                'detJ_loss' : self.detJ_loss_tracker.result()}