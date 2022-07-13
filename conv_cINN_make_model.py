#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:34:11 2021

@author: John S. Hyatt
"""

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import (Model,
                              metrics)

from tensorflow.keras.layers import (Input,
                                     Activation,
                                     LeakyReLU,
                                     LayerNormalization,
                                     Convolution2D,
                                     Concatenate)

from tensorflow.keras.backend import int_shape

from conv_cINN_base_functions import dilated_residual_block

import numpy as np

###############################################################################

"""
With a naive implementation, training starts at NaN loss for too-large networks (really, anything that isn't unrealistically small) due to log_prob(z) = log_prob(f(x)) blowing up when the number of dimensions is not very small. This can be fixed by using Orthogonal kernel initializer (using layer normalization provides some benefit as well).

For datasets such as MNIST, where many elements are always the same (e.g. the black pixels at the edges), dim(X)<<dim(Z) (even without taking into account the fact that a particular real dataset is a low-dimensional embedding in R^N), and the map can't be bijective. Training completely destabilizes once annealing finishes. However, including a small amount of noise in X resolves this problem, since even though most of the dimensions are still meaningless, they are no longer unchanging (effectively, it replaces "delta distributions" of unchanging black background pixels with narrow, but non-delta, distributions of pixel values. The gradients the model receives as a result are meaningless and the noise can be removed during inference. This step is not needed when the dataset does not have pixels of fixed value (like a background).

In the original RealNVP paper, the authors replace x -> logit(a + (1-a)* x/x_max), where a is a small floor (say 0.01), x_max is the highest allowed value of x, and logit(p) = log(p / (1-p)). This apparently serves two purposes:
    1) Probability is defined over an unbounded domain, but many x (e.g., images) have a hard boundary, for example pixel values in the range [0,255]. Logit maps a function on the domain [0,1] to (-inf, inf), making the domain unbounded.
    2) An unrealistically high prediction of logit-intensity will still yield the a realistic intensity (either the minimum or maximum) when de-logit-ized. That is, the generated distribution of intensity should more closely match the true intensity distribution than the generated distribution of logit-intensity matches the true logit-intensity distribution.

I have experimented with modeling x directly, as there's "not much" difference between a hard bounded domain and an unbounded one whose probability density outside the boundary is very low. Because the majority of values logit(p) takes are closely associated with either p~0 or p~1, this may overemphasize high/low pixel intensities.

On the other hand, experimentally, this seems to allow unnaturally high/low pixel values in latent space and, correspondingly, pixel values in x space outside of the allowed boundaries, due to the nonzero density that in practice is modeled outside of the bounded regions. (This can be ambiguous depending on the image preprocessing: generally, it is impossible to assign bounds to the true data distribution with perfect certainty.) Logit-izing the intensity hides this problem; an equivalent solution is just to set intensities below or above the allowed values to the min or max.

They also did L2 weight normalization in addition to (modified) batch normalization. As pointed out in [T. van Laarhoven, "L2 Regularization vs. Batch and Weight Normalization," NeurIPS 2017], it doesn't make sense to do both (or use weight normalization with layer norm, for that matter).

As a result I have NOT used the L2 weight normalization. Instead I have just replaced their modified batch norm with regular layer norm.

Some of this code is based on the https://github.com/taesungp/real-nvp/blob/master/real_nvp/nn.py implementation of the architecture described in https://arxiv.org/pdf/1605.08803.pdf.
"""

"""
NOTE: On some hardware I have observed a speedup when @tf.function-decorating the forward_and_Jacobian() and backward() methods on the new layers defined below. On other hardware I have observed no speedup or even a slight slowdown. I've had the same experience when decorating the call() method in cFlow() instead. I've left both sets of decorators in this code, but commented; uncomment them if you want to experiment.
"""

###############################################################################

"""
GENERIC LAYER CLASS
"""

# Adding some default methods to the Layer class. These will be called by the Layers used to build the model.
# This does NOT remove anything from the Layer class, only adds to it.
class Layer(tf.keras.layers.Layer):

    """
    Used to define other Layers later. This is an abstract class that can propagate in both directions.

    In the forward direction, it passes along u (the input), the log determinant of the Jacobian, and z (specifically, the already-factored-out z components).

    In the backward direction, it passes along v (the input) and z (specifically, the factored-out z components that haven't yet been factored back in).
    """

# =============================================================================
#     @tf.function
# =============================================================================
    def forward_and_Jacobian(self,
                             u,             # the input in a chain x -> z
                             sum_log_det_J, # the total Jacobian so far
                             z):            # the already factored-out z so far
        # It doesn't do anything (this is just an abstract class)
        raise NotImplementedError(str(type(self)))

# =============================================================================
#     @tf.function
# =============================================================================
    def backward(self,
                 v,  # the input in a chain z -> x
                 z): # the factored-out z that hasn't yet been reintegrated
        # It doesn't do anything (this is just an abstract class)
        raise NotImplementedError(str(type(self)))

###############################################################################

"""
A LAYER FOR PERFORMING SCALAR MULTIPLICATION OF THE MODEL INPUTS
"""

class tanh_scaling_layer(Layer):

    """
    Multiplies the inputs by a trainable scalar variable.
    """

    # Layer class takes kwargs (name and dtype). Best practice is to pass these to the parent class.
    def __init__(self, **kwargs):
        super(tanh_scaling_layer, self).__init__(**kwargs)

    # Add the tanh scale, initialized at 1.
    # Not defining a shape defaults it to a scalar.
    def build(self, input_shape):
        initializer = tf.keras.initializers.Ones()
        self.w = self.add_weight(initializer=initializer,
                                 trainable=True)

    def call(self, inputs):
        return tf.math.scalar_mul(self.w,
                                     inputs)

    # Because this Layer defines __init__(), get_config() needs to be redefined to serialize the model.
    # This will include the kwargs passed to the parent class in __init__().
    def get_config(self):
        config = super(tanh_scaling_layer, self).get_config()
        return config

###############################################################################

"""
THE SPATIAL SQUEEZING AND z FACTORING LAYERS
"""

class squeeze_layer(Layer):

    """
    In the forward direction, this Layer spatially halves (in both dimensions, quadrupling the channel depth) both the input and any already-factored-out zy, while passing along log_det_J.

    In the backward direction, this Layer spatially doubles (in both dimensions, quartering the channel depth) both the input and any not-yet-refactored-in zy.

    Note that, unlike the original RealNVP, the "output" side for is zy, not z! This is a conditional model, so y has to be passed through as well.
    """

    # Layer class takes kwargs (name and dtype). Best practice is to pass these to the parent class.
    def __init__(self, **kwargs):
        super(squeeze_layer, self).__init__(**kwargs)

    # Because this Layer defines __init__(), get_config() needs to be redefined to serialize the model.
    # This will include the kwargs passed to the parent class in __init__().
    def get_config(self):
        config = super(squeeze_layer, self).get_config()

        return config

    # From XY -> ZY (U -> V)
# =============================================================================
#     @tf.function
# =============================================================================
    def forward_and_Jacobian(self,
                             u,
                             sum_log_det_J,
                             zy):

        """
        Args:
            u: the input, going in the xy -> zy direction.
            sum_log_det_J: the total sum of the Jacobian contributions to the loss so far.
            zy: the already-factored-out zy so far.

        Returns:
            v: the output, going in the xy -> zy direction, reshaped but not otherwise changed from u by this Layer.
            sum_log_det_J: the total sum of the Jacobian contributions to the loss so far (not changed by this Layer).
            zy: the already-factored-out zy so far, reshaped but not otherwise changed by this Layer.
        """

        u_shape = int_shape(u)

        # In order to be halved spatially, the spatial dimensions must be divisible by 2.
        assert u_shape[1] % 2 == 0 \
           and u_shape[2] % 2 == 0, \
           'u must have spatial dimensions divisible by 2.'

        v = tf.nn.space_to_depth(u, 2) # halving the spatial dimensions

        # Need to keep zy spatial dims the same so new zy factors can be concatenated on.
        if zy is not None:
            zy = tf.nn.space_to_depth(zy, 2) # halving each spatial dimension

        return v, sum_log_det_J, zy

    # From ZY -> XY (V -> U)
# =============================================================================
#     @tf.function
# =============================================================================
    def backward(self,
                 v,  # the input in a chain zy -> xy
                 zy): # the factored-out zy that hasn't yet been reintegrated

        """
        Args:
            v: the input, going in the zy -> xy direction.
            zy: the factored-out zy that hasn't yet been reintegrated.

        Returns:
            u: the output, going in the zy -> xy direction, reshaped but not otherwise changed from v by this Layer.
            zy: the factored-out zy so far, reshaped but not otherwise changed by this Layer.
        """

        v_shape = int_shape(v)

        # Doubling both spatial dimensions means quartering the channel depth.
        assert v_shape[3] % 4 == 0, \
        'v must have channel dimensions divisible by 4.'

        u = tf.nn.depth_to_space(v, 2) # doubling the spatial dimensions

        # Need to "undo" the compression of zy as it propagates back through
        if zy is not None:
            zy = tf.nn.depth_to_space(zy, 2)

        return u, zy

class factor_out_zy_layer(Layer):

    """
    In the forward direction, this Layer factors out half of the non-zy channels into zy.

    In the backward direction, this Layer factors as many channels back in from zy.

    Note that the backward direction DOES perform the first unfactor, while the forward direction DOES NOT perform the final concatenation to zy.
    """

    # Layer class takes kwargs (name and dtype). Best practice is to pass these to the parent class.
    def __init__(self,
                 num_prev_factors,
                 **kwargs):

        """
        Args:
            num_prev_factors: how many times the model has already had part of zy factored out, going in the XY -> ZY direction.
        """

        super(factor_out_zy_layer, self).__init__(**kwargs)
        self.num_prev_factors = num_prev_factors

    # Because this Layer defines __init__(), get_config() needs to be redefined to serialize the model.
    # This will include the kwargs passed to the parent class in __init__().
    def get_config(self):
        config = super(factor_out_zy_layer, self).get_config()

        # Add the new arguments into config.
        config.update({'num_prev_factors' : self.num_prev_factors})

        return config

    # From XY -> ZY (U -> V)
# =============================================================================
#     @tf.function
# =============================================================================
    def forward_and_Jacobian(self,
                             u,
                             sum_log_det_J,
                             zy):

        """
        Args:
            u: the input, going in the xy -> zy direction.
            sum_log_det_J: the total sum of the Jacobian contributions to the loss so far.
            zy: the already-factored-out zy so far.

        Returns:
            v: the output, going in the xy -> zy direction, now with additional zy factored out from u.
            sum_log_det_J: the total sum of the Jacobian contributions to the loss so far (not changed by this Layer).
            zy: the already-factored-out zy so far concatenated with the zy factored out by this Layer.
        """

        u_shape = int_shape(u)

        # Half of the channels will be factored into zy.
        split = u_shape[3] // 2

        factored_zy = u[..., :split]
        v = u[..., split:]

        # If this is the first time, there's no prior zy to concatenate to. Otherwise, concatenate onto the already-factored-out zy.
        if zy is not None:
            zy = tf.concat([zy, factored_zy],
                           axis=3)
        else:
            zy = factored_zy

        return v, sum_log_det_J, zy

    # From ZY -> XY (V -> U)
# =============================================================================
#     @tf.function
# =============================================================================
    def backward(self,
                 v,
                 zy):

        """
        Args:
            v: the input, going in the zy -> xy direction.
            zy: the factored-out zy that hasn't yet been reintegrated.

        Returns:
            u: the output, going in the zy -> xy direction, now with additional components reintegrated from zy.
            zy: the factored-out zy minus the part that was reintegrated by this Layer.
        """

        # At scale s, (1/2)^(s+1) of the original dimensions have been factored out.
        # For a backwards pass at scale s, (1/2)^s of zy should be factored back in.

        zy_shape = int_shape(zy)

        if v is None: # the last layer is all zy
            split = zy_shape[3] // (2**self.num_prev_factors)
        else: # the input has some v and some not-yet-refactored zy
            split = int_shape(v)[3]

        reintegrated_v = zy[..., -split:]
        zy = zy[..., :-split]

        # Redundant assertion.
        assert int_shape(reintegrated_v)[3] == split

        if v is not None:
            u = tf.concat([reintegrated_v, v], 3)
        else:
            u = reintegrated_v

        return u, zy

###############################################################################

"""
DEFINE A SCALING COUPLING LAYER
"""

class coupling_layer(Layer):

    """
    RealNVP coupling layer, with modifications:
        The most important one is that instead of x and z, the end states of the total model are xy and zy. As a result, additional dimensions propagate through each layer!
        Almost as important, because xy and zy have more than one dimension, channelwise masking and coupling is performed BEFORE the squeeze/factor operation in each block. I also only include one of each mask (in order 0, 1, 2, 3) rather than multiples (in order 0, 1, 0, 2, 3, 2). This does not directly affect anything INSIDE each coupling layer, but it does affect the way they are structured in the stack.
        The masks are constructed differently, although conceptually they are the same. In the original paper, the mask was applied to the input (zeroing half of its elements), which was then fed into the coupling function. The mask was then re-applied to the output of the coupling function to "manually" re-zero those elements, before combining it with the other half of the input (obtained via complementary mask). This results in many useless computations (they are just zeroed out by the second, post-coupling mask) AND makes inefficient use of the convolutional kernels, since either 50% of spatial dimensions or 50% of channel dimensions are zero. It might also lead to the kernels learning weird behavior (which would interfere with LayerNorm, since that would be applied before re-masking), complicate analysis of intermediate representations, and for the checkerboard masks in particular, would lead to kernels covering a smaller spatial dimension than necessary. As a result, I compress uv1 (the inputs to A and b) and uv2 so that the calculations v2 = A(u1)*u2 + b(u1) and u2 = A^-1(v1)*(v2 - b(v1)) can be performed without all those extraneous zeros. Note that this does make the MASKING process more computationally intensive, AND memory intensive: only the compressed version of uv2 is required, but both the compressed AND uncompressed versions of uv1 are required (and must be stored simultaneously). vu2 must then be decompressed and added to the uncompressed vu1 = uv1.
        A side benefit from this is that the checkerboard and channelwise masks have different spatial scales, meaning that this implementation probably has somewhat better inductive biases than the original.
        Due to the fact that the model maps XY -> ZY and the log-likelihood-of-Y-equivalent loss term requires access to Y, I don't want to just apply squeezing and factoring to generate a 1D vector Z: the original spatial relationships between pixels are important. Although that loss term does not REQUIRE Y to preserve the spatial relationships for the choice of metric I use (L norm), it might for other choices of metric. As a result, after fully squeezing and factoring ZY, I reshape it to match the original shape of XZ. (This might be a target for efficiency improvements in an actual applied implementation of this model.)
        I use LayerNorm instead of the original paper's modified BatchNorm.
        I do not use L2 weight normalization due to already using LayerNorm. L2 weight normalization and normalization schemes like batch or layer normalization are redundant; see van Laarhoven, "L2 Regularization versus Batch and Weight Normalization" (2017).
        I do not use tanh plus a learnable scaling parameter.
        Logits on f_X are included by default, but they can be replaced by just using f_X by commenting lines in the code, labeled below.
        I include dilated convolutions in the ResNeXt blocks that define A and b.
        I use LeakyReLU instead of ReLU.
    """

    # Layer class takes kwargs (name and dtype). Best practice is to pass these to the parent class.
    def __init__(self,
                 in_shape,
                 which_mask,
                 num_res_blocks,
                 cardinality,
                 num_kernels,
                 kernel_size,
                 init,
                 LAYER_NORM=False,
                 which_dilations=[1,2,4],
                 **kwargs):

        """
        Args:
            in_shape: a list with the shape of the input tensor, [height, width, channel_depth]. Height and width are specified so that the model can be built before calling the coupling function. Depth is specified because if the number of channels in the input is odd (only the case before the first squeeze/factoring layer, if at all), then masks 2 and 3 split the extra channel differently. (The depth doesn't affect anything if the number of channels in the input is even.)
            which_mask: either 0, 1, 2, or 3
                if 0:   mask = [1,1,...,1] [0,0,...,0] [1,1,...,1]...
                               [0,0,...,0] [1,1,...,1] [0,0,...,0]...
                               [1,1,...,1] [0,0,...,0] [1,1,...,1]...
                                   ...         ...         ...

                elif 1: mask = [0,0,...,0] [1,1,...,1] [0,0,...,0]...
                               [1,1,...,1] [0,0,...,0] [1,1,...,1]...
                               [0,0,...,0] [1,1,...,1] [0,0,...,0]...
                                   ...         ...         ...

                elif 2: mask = [1,0,1,0,...] [1,0,1,0,...] [1,0,1,0,...]...
                               [1,0,1,0,...] [1,0,1,0,...] [1,0,1,0,...]...
                               [1,0,1,0,...] [1,0,1,0,...] [1,0,1,0,...]...
                                    ...           ...           ...

                elif 3: mask = [0,1,0,1,...] [0,1,0,1,...] [0,1,0,1,...]...
                               [0,1,0,1,...] [0,1,0,1,...] [0,1,0,1,...]...
                               [0,1,0,1,...] [0,1,0,1,...] [0,1,0,1,...]...
                                    ...           ...           ...
            num_res_blocks: number of residual blocks in each of the neural networks defining A and b.
            cardinality: cardinality of the residual blocks.
            num_kernels: number of kernels per convolutional layer. Note that this number is halved for masks 0 and 1, whose compressed forms have twice as many channels/half as much spatial extent. I therefore assign half as many kernels to these layers as the corresponding channelwise-masked inputs.
            kernel_size: size of the convolutional kernels.
            init: the kernel initializer.
            LAYER_NORM: Boolean. Whether or not the residual blocks will have layer normalization or not.
            which_dilations: A list of the dilation factors to be used in parallel in the residual block. For example, [1,2,4] has, in parallel: no dilation, dilation with 1 zero between nonzero elements, and dilation with 3 zeros. With 3x3 kernels, [1,2,4] gives overlapping receptive fields up to 9x9 and [1,2,4,8] gives receptive fields up to 17x17. With 4x4 kernels, [1,3] gives overlapping receptive fields up to 10x10.

        Note that A and b have the same hyperparameters within a given coupling layer (for a given model, profiling might show that this is not necessary/efficient!) and larger dilations within a layer are given progressively fewer kernels (the assumption being that the larger-length-scale correlations between successive layers can be well incorporated with relatively few kernels; moreover, a model's receptive field typically grows as it gets deeper, and because the intermediate representations' spatial scales decrease every time the model is squeeze/factored).
        """

        super(coupling_layer, self).__init__(**kwargs)
        # Hyperparameters for defining A and b.
        # In this code, these are the same for both A and b, and for all coupling blocks, except that fewer factors listed in which_dilations will be used for smaller-spatial-dimension inputs. A specific implementation would probably benefit from more customized tuning.
        self.input_height = in_shape[0]
        self.input_width = in_shape[1]
        self.input_depth = in_shape[2]
        self.which_mask = which_mask
        self.num_res_blocks = num_res_blocks
        self.cardinality = cardinality
        self.kernel_size = kernel_size
        self.init = init
        self.LAYER_NORM = LAYER_NORM
        self.which_dilations = which_dilations

        assert self.input_height % 2 == 0 \
           and self.input_width % 2 == 0, \
           'u/v must have spatial dimensions divisible by 2.'

        # Masks 0 and 1 correspond to checkerboard masks, whose compressed forms have twice as many channels/half as much spatial extent, and therefore half as many kernels, as the channelwise-masked inputs.
        if self.which_mask in [0,1]:
            self.num_kernels = int(num_kernels / 2)
        elif self.which_mask in [2,3]:
            self.num_kernels = num_kernels

        # self.which_mask defines the mask used to obtain uv1. The mask used to obtain uv2 is its complement.
        if self.which_mask==0:
            self.which_mask_complement = 1
        elif self.which_mask==1:
            self.which_mask_complement = 0
        elif self.which_mask==2:
            self.which_mask_complement = 3
        elif self.which_mask==3:
            self.which_mask_complement = 2

        # Get the shape of the MASKED, COMPRESSED input to the coupling function.
        self.get_masked_compressed_shape()

        # Build the neural networks A and b used in the coupling function.
        self.model_A, self.model_b = self.coupling_function()

    # Because this Layer defines __init__(), get_config() needs to be redefined to serialize the model.
    # This will include the kwargs passed to the parent class in __init__().
    def get_config(self):
        config = super(coupling_layer, self).get_config()

        # Add the new arguments into config.
        config.update({'input_height' : self.input_height,
                       'input_width' : self.input_width,
                       'input_depth' : self.input_depth,
                       'which_mask' : self.which_mask,
                       'num_res_blocks' : self.num_res_blocks,
                       'cardinality' : self.cardinality,
                       'kernel_size' : self.kernel_size,
                       'init' : self.init,
                       'LAYER_NORM' : self.LAYER_NORM,
                       'which_dilations' : self.which_dilations})

        return config

    def A_wrapper(self,
                  A_input):
        """
        A wrapper for model A to allow it to be used inside of a TF graph.
        """
        return self.model_A(A_input)

    def b_wrapper(self,
                  b_input):
        """
        A wrapper for model b to allow it to be used inside of a TF graph.
        """
        return self.model_b(b_input)

    def get_masked_compressed_shape(self):

        """
        Function for obtaining the shape of the (masked, compressed) u1/v1 input to the coupling function, given the full input shape and the mask type.
        """

        if self.which_mask in [0,1]:

            self.compressed_height = int(self.input_height / 2)
            self.compressed_width = int(self.input_width / 2)
            self.compressed_depth = 2 * self.input_depth

        elif self.which_mask in [2,3]:

            self.compressed_height = self.input_height
            self.compressed_width = self.input_width
            # Masks 2 and 3 have different depths if the total number of channels is odd.

            if self.which_mask == 2:

                self.compressed_depth = int(np.ceil(self.input_depth/2))

            elif self.which_mask == 3:

                self.compressed_depth = int(np.floor(self.input_depth/2))

    def mask(self,
             uv,
             which_mask_index,
             compress):

        """
        Args:
            uv = input of shape (batch_size, uv_h, uv_w, uv_d). Going in the forward direction, uv = u; going in the backward direction, uv = v.
                batch_size: number of elements in the batch
                uv_h: the height of the input tensor (currently, must be even).
                uv_w: the width of the input tensor (currently, must be even).
                uv_d: the depth of the input tensor (can be even or odd).
            which_mask_index: one of 0,1,2,3

                if 0: mask = [1,1,...,1] [0,0,...,0] [1,1,...,1]...
                             [0,0,...,0] [1,1,...,1] [0,0,...,0]...
                             [1,1,...,1] [0,0,...,0] [1,1,...,1]...
                                 ...         ...         ...

                elif 1: mask = [0,0,...,0] [1,1,...,1] [0,0,...,0]...
                               [1,1,...,1] [0,0,...,0] [1,1,...,1]...
                               [0,0,...,0] [1,1,...,1] [0,0,...,0]...
                                   ...         ...         ...

                elif 2: mask = [1,0,1,0,...] [1,0,1,0,...] [1,0,1,0,...]...
                               [1,0,1,0,...] [1,0,1,0,...] [1,0,1,0,...]...
                               [1,0,1,0,...] [1,0,1,0,...] [1,0,1,0,...]...
                                    ...           ...           ...

                elif 3: mask = [0,1,0,1,...] [0,1,0,1,...] [0,1,0,1,...]...
                               [0,1,0,1,...] [0,1,0,1,...] [0,1,0,1,...]...
                               [0,1,0,1,...] [0,1,0,1,...] [0,1,0,1,...]...
                                    ...           ...           ...

            compress: Boolean. Removes the masked (zeroed) entries if True.

        Returns:
            uv_masked: Denoting specific elements in uv by

            A1 B1 A1 B1...    A2 B2 A2 B2...    A3 B3 A3 B3...    ...
            C1 D1 C1 D1...    C2 D2 C2 D2...    C3 D3 C3 D3...    ...
            A1 B1 A1 B1...    A2 B2 A2 B2...    A3 B3 A3 B3...    ...
            C1 D1 C1 D1...    C2 D2 C2 D2...    C3 D3 C3 D3...    ...
            .. .. .. ..       .. .. .. ..       .. .. .. ..       ...

            with 1, 2, 3, ... denoting channels:

            if compress==True:

                if which_mask==0:
                this produces an output tensor with shape
                (batch, uv_h/2, uv_w/2, uv_d*2):

                    A1 A1...    A2 A2...    ...    D1 D1...    D2 D2...    ...
                    A1 A1...    A2 A2...    ...    D1 D1...    D2 D2...    ...
                    .. ..       .. ..       ...    .. ..       .. ..       ...

                elif which_mask==1:
                this produces an output tensor with shape
                (batch, uv_h/2, uv_w/2, uv_d*2):

                    B1 B1...    B2 B2...    ...    C1 C1...    C2 C2...    ...
                    B1 B1...    B2 B2...    ...    C1 C1...    C2 C2...    ...
                    .. ..       .. ..       ...    .. ..       .. ..       ...

                elif which_mask==2:
                this produces an output tensor with shape
                (batch, uv_h, uv_w, ceil(uv_d/2)):

                    A1 B1 A1 B1...    A3 B3 A3 B3...    A5 B5 A5 B5...    ...
                    C1 D1 C1 D1...    C3 D3 C3 D3...    C5 D5 C5 D5...    ...
                    A1 B1 A1 B1...    A3 B3 A3 B3...    A5 B5 A5 B5...    ...
                    C1 D1 C1 D1...    C3 D3 C3 D3...    C5 D5 C5 D5...    ...
                    .. .. .. ..       .. .. .. ..       .. .. .. ..       ...

                elif which_mask==3:
                this produces an output tensor with shape
                (batch, uv_h, uv_w, floor(uv_d/2)):

                    A2 B2 A2 B2...    A4 B4 A4 B4...    ...
                    C2 D2 C2 D2...    C4 D4 C4 D4...    ...
                    A2 B2 A2 B2...    A4 B4 A4 B4...    ...
                    C2 D2 C2 D2...    C4 D4 C4 D4...    ...
                    .. .. .. ..       .. .. .. ..       ...

                (The smallest possible channel depth is 2, with mask==2 corresponding to channel 1, and mask==3 corresponding to channel 2.)

            elif compress==False:
                this produces an output tensor with shape (batch, uv_h, uv_w, uv_d):

                if which_mask==0:

                    A1 00 A1 00...    A2 00 A2 00...    A3 00 A3 00...    ...
                    00 D1 00 D1...    00 D2 00 D2...    00 D3 00 D3...    ...
                    A1 00 A1 00...    A2 00 A2 00...    A3 00 A3 00...    ...
                    00 D1 00 D1...    00 D2 00 D2...    00 D3 00 D3...    ...
                    .. .. .. ..       .. .. .. ..       .. .. .. ..       ...

                elif which_mask==1:
                    00 B1 00 B1...    00 B2 00 B2...    00 B3 00 B3...    ...
                    C1 00 C1 00...    C2 00 C2 00...    C3 00 C3 00...    ...
                    00 B1 00 B1...    00 B2 00 B2...    00 B3 00 B3...    ...
                    C1 00 C1 00...    C2 00 C2 00...    C3 00 C3 00...    ...
                    .. .. .. ..       .. .. .. ..       .. .. .. ..       ...

                elif which_mask==2:
                    A1 B1 A1 B1...    00 00 00 00...    A3 B3 A3 B3...    ...
                    C1 D1 C1 D1...    00 00 00 00...    C3 D3 C3 D3...    ...
                    A1 B1 A1 B1...    00 00 00 00...    A3 B3 A3 B3...    ...
                    C1 D1 C1 D1...    00 00 00 00...    C3 D3 C3 D3...    ...
                    .. .. .. ..       .. .. .. ..       .. .. .. ..       ...

                elif which_mask==3:
                    00 00 00 00...    A2 B2 A2 B2...    00 00 00 00...    ...
                    00 00 00 00...    C2 D2 C2 D2...    00 00 00 00...    ...
                    00 00 00 00...    A2 B2 A2 B2...    00 00 00 00...    ...
                    00 00 00 00...    C2 D2 C2 D2...    00 00 00 00...    ...
                    .. .. .. ..       .. .. .. ..       .. .. .. ..       ...
        """

        # The input tensor must have the correct spatial and channel dimensions.
        uv = tf.ensure_shape(uv,shape=[None,
                                       self.input_height,
                                       self.input_width,
                                       self.input_depth])

        (batch_size,
         uv_h,
         uv_w,
         uv_d) = int_shape(uv)

        # Only need to explicitly include the checkerboard mask zeros if the output is not compressed, since if it is compressed it just skips over the zeroed elements.
        if not compress:

            if which_mask_index==0:
                ones = tf.ones(uv_d,)
                zeros = tf.zeros(uv_d,)

                mask = tf.stack([[
                                [ones,
                                 zeros],
                                [zeros,
                                 ones]
                                ]])

            elif which_mask_index==1:
                ones = tf.ones(uv_d,)
                zeros = tf.zeros(uv_d,)

                mask = tf.stack([[
                                [zeros,
                                 ones],
                                [ones,
                                 zeros]
                                ]])

            elif which_mask_index==2:
                one_indices = tf.range(0,
                                       uv_d,
                                       2)
                one_indices = tf.expand_dims(one_indices,
                                             axis=-1)
                # mask==2 has the extra channel if there are an odd number, so use the ceiling function.
                ones = tf.ones(
                           tf.cast(
                               tf.math.ceil(uv_d/2),
                               dtype=tf.int32))
                mask_shape = tf.constant([uv_d])

                mask = tf.scatter_nd(one_indices,
                                     ones,
                                     mask_shape)

                mask = tf.stack([[
                                [mask,
                                 mask],
                                [mask,
                                 mask]
                                ]])

            elif which_mask_index==3:
                one_indices = tf.range(1,
                                       uv_d,
                                       2)
                one_indices = tf.expand_dims(one_indices,
                                             axis=-1)
                # mask==3 doesn't have the extra channel if there are an odd number, so use the floor function.
                ones = tf.ones(
                           tf.cast(
                               tf.math.floor(uv_d/2),
                               dtype=tf.int32))
                mask_shape = tf.constant([uv_d])

                mask = tf.scatter_nd(one_indices,
                                     ones,
                                     mask_shape)

                mask = tf.stack([[
                                [mask,
                                 mask],
                                [mask,
                                 mask]
                                ]])

            # Tile to the correct height/width
            mask = tf.tile(mask,
                           [1,
                            int(uv_h/2),
                            int(uv_w/2),
                            1])

            # The mask must go over the entire batch.
            # Using the einsum method is much more efficient than explicitly repeating the mask `batch_size` times.
            mask = mask[0]

            uv_masked = tf.einsum('jkl,ijkl->ijkl',
                                  mask,
                                  uv)

        # If we are compressing it, just skip over the unwanted components.
        elif compress:

            # For checkerboard masking, we need to skip every other element AND stack the offset remaining elements in the checkerboard.
            if which_mask_index in [0,1]:

                # Note the repeating indices.
                if which_mask_index==0:
                    uv_c0 = uv[:,
                               0::2,
                               0::2,
                               :]
                    uv_c1 = uv[:,
                               1::2,
                               1::2,
                               :]

                # Note the alternating indices.
                elif which_mask_index==1:
                    uv_c0 = uv[:,
                               0::2,
                               1::2,
                               :]
                    uv_c1 = uv[:,
                               1::2,
                               0::2,
                               :]

                uv_masked = Concatenate(axis=-1)([uv_c0,
                                                  uv_c1])

            # For channel-wise masking, we just drop the unwanted channels.
            elif which_mask_index in [2,3]:

                if which_mask_index==2:
                    uv_masked = uv[...,
                                   0::2]

                elif which_mask_index==3:
                    uv_masked = uv[...,
                                   1::2]

        return uv_masked

    def decompress_mask(self,
                        uv_masked_compressed,
                        which_mask_index,
                        uv_shape_OUTPUT):

        """
        Args:
            uv_masked_compressed = compressed tensor input of shape (batch_size, uv_h_c, uv_w_c, uv_d_c). Going in the forward direction, uv = u; going in the backward direction, uv = v.
                batch_size: number of elements in the batch.
                uv_h_c: the height of the COMPRESSED input tensor.
                uv_w_c: the width of the COMPRESSED input tensor.
                uv_d_c: the depth of the COMPRESSED input tensor.
            which_mask_index: one of 0,1,2,3

                if 0: mask = [1,1,...,1] [0,0,...,0] [1,1,...,1]...
                             [0,0,...,0] [1,1,...,1] [0,0,...,0]...
                             [1,1,...,1] [0,0,...,0] [1,1,...,1]...
                                 ...         ...         ...

                elif 1: mask = [0,0,...,0] [1,1,...,1] [0,0,...,0]...
                               [1,1,...,1] [0,0,...,0] [1,1,...,1]...
                               [0,0,...,0] [1,1,...,1] [0,0,...,0]...
                                   ...         ...         ...

                elif 2: mask = [1,0,1,0,...] [1,0,1,0,...] [1,0,1,0,...]...
                               [1,0,1,0,...] [1,0,1,0,...] [1,0,1,0,...]...
                               [1,0,1,0,...] [1,0,1,0,...] [1,0,1,0,...]...
                                    ...           ...           ...

                elif 3: mask = [0,1,0,1,...] [0,1,0,1,...] [0,1,0,1,...]...
                               [0,1,0,1,...] [0,1,0,1,...] [0,1,0,1,...]...
                               [0,1,0,1,...] [0,1,0,1,...] [0,1,0,1,...]...
                                    ...           ...           ...
            uv_shape_OUTPUT = tuple of shape (batch_size, uv_h, uv_w, uv_d) defining the UNCOMPRESSED output shape. This is necessary because the depth of the uncompressed tensor (although not the height or width) can be odd. If this is the case, masks 2 and 3 will have different output depths; otherwise they will have the same output depth.

        Returns:
            uv_masked_uncompressed: the uncompressed version of uv_masked_compressed.
                if mask in [0,1]: uv_masked_compressed has shape
                                  (batch, uv_h/2, uv_w/2, uv_d*2)
                elif mask==2: uv_masked_compressed has shape
                                  (batch, uv_h, uv_w, ceil(uv_d/2))
                elif mask==3: uv_masked_compressed has shape
                                  (batch, uv_h, uv_w, floor(uv_d/2))
                uv_masked_uncompressed has shape
                                  (batch, uv_h, uv_w, uv_d).
            NOTE: uv_h and uv_w are both currently required to be even. uv_d may be odd, in which case uv_masked_compressed has 1 more channel for mask==2, or 1 less channel for mask==3.

            Denoting specific elements in uv by

            A1 B1 A1 B1...    A2 B2 A2 B2...    A3 B3 A3 B3...    ...
            C1 D1 C1 D1...    C2 D2 C2 D2...    C3 D3 C3 D3...    ...
            A1 B1 A1 B1...    A2 B2 A2 B2...    A3 B3 A3 B3...    ...
            C1 D1 C1 D1...    C2 D2 C2 D2...    C3 D3 C3 D3...    ...
            .. .. .. ..       .. .. .. ..       .. .. .. ..       ...

            with 1, 2, 3, ... denoting channels:

            if which_mask==0:
            this converts an input tensor

                A1 A1...    A2 A2...    ...    D1 D1...    D2 D2...    ...
                A1 A1...    A2 A2...    ...    D1 D1...    D2 D2...    ...
                .. ..       .. ..       ...    .. ..       .. ..       ...

            to output tensor

                A1 00 A1 00...    A2 00 A2 00...    ...
                00 D1 00 D1...    00 D2 00 D2...    ...
                A1 00 A1 00...    A2 00 A2 00...    ...
                00 D1 00 D1...    00 D2 00 D2...    ...
                .. .. .. ..       .. .. .. ..       ...

            elif which_mask==1:
            this converts an input tensor

                B1 B1...    B2 B2...    ...    C1 C1...    C2 C2...    ...
                B1 B1...    B2 B2...    ...    C1 C1...    C2 C2...    ...
                .. ..       .. ..       ...    .. ..       .. ..       ...

            to output tensor

                00 B1 00 B1...    00 B2 00 B2...    ...
                C1 00 C1 00...    C2 00 C2 00...    ...
                00 B1 00 B1...    00 B2 00 B2...    ...
                C1 00 C1 00...    C2 00 C2 00...    ...
                .. .. .. ..       .. .. .. ..       ...

            elif which_mask==2:
            this converts an input tensor

                A1 B1 A1 B1...    A3 B3 A3 B3...    ...
                C1 D1 C1 D1...    C3 D3 C3 D3...    ...
                A1 B1 A1 B1...    A3 B3 A3 B3...    ...
                C1 D1 C1 D1...    C3 D3 C3 D3...    ...
                .. .. .. ..       .. .. .. ..       ...

            to output tensor

                A1 B1 A1 B1...    00 00 00 00...    A3 B3 A3 B3...    ...
                C1 D1 C1 D1...    00 00 00 00...    C3 D3 C3 D3...    ...
                A1 B1 A1 B1...    00 00 00 00...    A3 B3 A3 B3...    ...
                C1 D1 C1 D1...    00 00 00 00...    C3 D3 C3 D3...    ...
                .. .. .. ..       .. .. .. ..       .. .. .. ..       ...

            elif which_mask==3:
            this converts an input tensor

                A2 B2 A2 B2...    ...
                C2 D2 C2 D2...    ...
                A2 B2 A2 B2...    ...
                C2 D2 C2 D2...    ...
                .. .. .. ..       ...

            to output tensor

                00 00 00 00...    A2 B2 A2 B2...    00 00 00 00...    ...
                00 00 00 00...    C2 D2 C2 D2...    00 00 00 00...    ...
                00 00 00 00...    A2 B2 A2 B2...    00 00 00 00...    ...
                00 00 00 00...    C2 D2 C2 D2...    00 00 00 00...    ...
                .. .. .. ..       .. .. .. ..       .. .. .. ..       ...
        """

        (batch_size,
         uv_h_c,
         uv_w_c,
         uv_d_c) = int_shape(uv_masked_compressed)

        (batch_size,
         uv_h,
         uv_w,
         uv_d) = uv_shape_OUTPUT

        # Expanding the checkerboard mask.
        if which_mask_index in [0,1]:

            # Input shape is (batch, u_h_c, u_w_c, u_d_c) = (batch, u_h/2, u_w/2, u_d*2).
            assert uv_d_c % 2 == 0, \
            'The compressed, checkerboard-masked u/v should always have an even number of channels.'

            # Each of the two offset checkerboards takes up half of uv_d_c.
            uv_c0 = uv_masked_compressed[...,
                                         :uv_d]
            uv_c1 = uv_masked_compressed[...,
                                         uv_d:]

            # These indices determine the location of the compressed elements and the inserted zeros during expansion. Note that for expanding mask_0-compressed tensors, both expansions per channel are performed with the same index applied twice in sequence, while for the mask_1-compressed tensors, they are applied alternately.
            # For non-square tensors, each dimension needs its own set of 0 and 1 checkerboard indices.
            # NOTE: the "0" and "1" in h0, h1, w0, and w1 do NOT correspond to masks 0 and 1! They are indices for the two possible checkerboard offsets:
            # 1 0 1 0         0 1 0 1
            # 0 1 0 1   and   1 0 1 0
            # 1 0 1 0         0 1 0 1
            # 0 1 0 1         1 0 1 0

            # The indices 0 and 1 go like:
            #   0 1 0 1
            # 0 X X X X
            # 1 X X X X
            # 0 X X X X
            # 1 X X X X
            # with h going across and w going down.
            # In the compressed masked representation, zero entries in the masked input have been removed in favor of splitting alternating checkerboards into channels:
            # mask 0         h0, w0        h1, w1
            # 1 0 1 0        1 0 1 0       0 0 0 0
            # 0 1 0 1   ->   0 0 0 0   +   0 1 0 1
            # 1 0 1 0        1 0 1 0       0 0 0 0
            # 0 1 0 1        0 0 0 0       0 1 0 1

            # mask 1         h0, w1        h1, w0
            # 0 1 0 1        0 1 0 1       0 0 0 0
            # 1 0 1 0   ->   0 0 0 0   +   1 0 1 0
            # 0 1 0 1        0 1 0 1       0 0 0 0
            # 1 0 1 0        0 0 0 0       1 0 1 0

            indices_h0 = tf.range(start=0,
                                  limit=2*uv_h_c,
                                  delta=2)
            indices_h1 = tf.range(start=1,
                                  limit=2*uv_h_c+1,
                                  delta=2)
            indices_w0 = tf.range(start=0,
                                  limit=2*uv_w_c,
                                  delta=2)
            indices_w1 = tf.range(start=1,
                                  limit=2*uv_w_c+1,
                                  delta=2)

            indices_h0 = tf.expand_dims(indices_h0,
                                        axis=-1)    # shape (uv_h/2,1)
            indices_w0 = tf.expand_dims(indices_w0,
                                        axis=-1)    # shape (uv_w/2,1)
            indices_h1 = tf.expand_dims(indices_h1,
                                        axis=-1)    # shape (uv_h/2,1)
            indices_w1 = tf.expand_dims(indices_w1,
                                        axis=-1)    # shape (uv_w/2,1)

            ####################################
            # Expand compressed "A"/"B" channel.
            ####################################
            # In order to use scatter_nd, transpose the batch dimension to the end.
            updates = tf.transpose(uv_c0,
                                   [1,2,3,0])       # shape (uv_h/2,uv_w/2,uv_d,batch)

            # Expand by a factor of 2 in one of the two to-be-expanded dimensions.
            shape = [2,1,1,1] * tf.shape(updates)   # shape (uv_h,uv_w/2,uv_d,batch)

            # `indices` tells scatter_nd where to put the zeros it is inserting.
            # `updates` is the input that scatter_nd is expanding.
            # `shape` tells scatter_nd what the output shape should be.
            # `which_mask` determines whether the indices are identical or complementary. For this particular transform, it is the same for both masks.
            scatter = tf.scatter_nd(indices_h0,
                                    updates,
                                    shape)          # shape (uv_h,uv_w/2,uv_d,batch)

            # Next, we need to expand the other dimension, so transpose those two and repeat the above steps.
            updates = tf.transpose(scatter,
                                   [1,0,2,3])       # shape (uv_w/2,uv_h,uv_d,batch)
            shape = [2,1,1,1] * tf.shape(updates)   # shape (uv_w,uv_h,uv_d,batch)
            # This one does differ depending on the mask.
            if which_mask_index==0:
                scatter = tf.scatter_nd(indices_w0,
                                        updates,
                                        shape)      # shape (uv_w,uv_h,uv_d,batch)
            elif which_mask_index==1:
                scatter = tf.scatter_nd(indices_w1,
                                        updates,
                                        shape)      # shape (uv_w,uv_h,uv_d,batch)

            # Now we have to transpose back into (batch_dim, height, width, channels)
            uv_c0 = tf.transpose(scatter,
                                 [3,1,0,2])          # shape (batch,uv_h,uv_w,uv_d)

            ####################################
            # Expand compressed "C"/"D" channel.
            ####################################
            # Exactly the same as before, except that the indices are different.
            updates = tf.transpose(uv_c1,
                                   [1,2,3,0])       # shape (uv_h/2,uv_w/2,uv_d,batch)
            shape = [2,1,1,1] * tf.shape(updates)   # shape (uv_h,uv_w/2,uv_d,batch)
            # This one is the same for both masks.
            scatter = tf.scatter_nd(indices_h1,
                                    updates,
                                    shape)          # shape (uv_h,uv_w/2,uv_d,batch)
            updates = tf.transpose(scatter,
                                   [1,0,2,3])       # shape (uv_w/2,uv_h,uv_d,batch)
            shape = [2,1,1,1] * tf.shape(updates)   # shape (uv_w,uv_h,uv_d,batch)
            # This one does differ depending on the mask.
            if which_mask_index==0:
                scatter = tf.scatter_nd(indices_w1,
                                        updates,
                                        shape)      # shape (uv_w,uv_h,uv_d,batch)
            elif which_mask_index==1:
                scatter = tf.scatter_nd(indices_w0,
                                        updates,
                                        shape)      # shape (uv_w,uv_h,uv_d,batch)
            uv_c1 = tf.transpose(scatter,
                                 [3,1,0,2])          # shape (batch,uv_h,uv_w,uv_d)

            uv_masked_uncompressed = uv_c0 + uv_c1

        # Expanding the channel-wise mask.
        elif which_mask_index in [2,3]:

            if which_mask_index==2:
                indices = tf.range(start=0,
                                   limit=uv_d,
                                   delta=2)
                indices = tf.expand_dims(indices,
                                         axis=-1)   # shape (ceil(uv_d/2),1)

            elif which_mask_index==3:
                indices = tf.range(start=1,
                                   limit=uv_d,
                                   delta=2)
                indices = tf.expand_dims(indices,
                                         axis=-1)   # shape (floor(uv_d/2),1)

            ####################################
            # Expand from uv_d/2 channel to uv_d channels.
            ####################################
            # In order to use scatter_nd, transpose the channel dimension to the front.
            updates = tf.transpose(uv_masked_compressed,
                                   [3,1,2,0])       # shape ("uv_d/2",uv_h,uv_w,batch)

            # Expand by a factor of 2 in the channel dimension.
            shape = [2,1,1,1] * tf.shape(updates)   # shape ("uv_d",uv_h,uv_w,batch)

            # if uv_d is even, "uv_d/2" = uv_d/2 and nothing more need be done.
            if uv_d % 2:

                # if uv_d is odd and mask==2, "uv_d/2" = ceil(uv_d/2)
                # -> "uv_d/2" * 2 - 1 = uv_d
                if which_mask_index==2:
                    shape -= [1,0,0,0]

                # if uv_d is odd and mask==3, "u_d/2" = floor(u_d/2)
                # -> "uv_d/2" * 2 + 1 = uv_d
                if which_mask_index==3:
                    shape += [1,0,0,0]

            # `indices` tells scatter_nd where to put the zeros it is inserting.
            # `updates` is the input that scatter_nd is expanding.
            # `shape` tell scatter_nd what the output shape should be.
            scatter = tf.scatter_nd(indices,
                                    updates,
                                    shape)          # shape (uv_d,uv_h,uv_w,batch)

            # Now we have to transpose back into (batch_dim, height, width, channels)
            uv_masked_uncompressed = tf.transpose(scatter,
                                                  [3,1,2,0])

        return uv_masked_uncompressed

    # This creates the models A and b. It is fairly time-consuming to run, but only has to be run once (during initialization).
    def coupling_function(self):

        """
        Note that, together, self.input_depth and self.which_mask determine whether the output channel depth of A(uv1) and b(uv1) is the same as the input channel depth of uv1.

        Returns:
            A TensorFlow model with inputs uv1 and outputs [A, b].
            A: the ResNeXt neural network defining the scaling matrix A(uv1). This is an MxM matrix where M = dim(uv2). (The neural network outputs the diagonal entries of A; since A is a diagonal matrix, this is all that is necessary to construct A.)
            b: the ResNeXt neural network defining the translation vector b(uv1), which has the same shape as uv2.
        """

        # Input to the coupling layer.
        uv1_h = self.compressed_height
        uv1_w = self.compressed_width
        uv1_d = self.compressed_depth

        # If the channel depth of uv=(uv1,uv2) is odd, the input depth of A and b is different from the output depth. Whether uv1 or uv2 has one more dimension than the other is determined by the mask.
        if self.input_depth % 2 \
        and self.which_mask==2:
            uv2_d = uv1_d - 1

        elif self.input_depth % 2 \
        and self.which_mask==3:
            uv2_d = uv1_d + 1

        # If the total depth is even, there is no difference.
        # For masks 0 and 1, there is no difference even if the total depth is odd.
        else:
            uv2_d = uv1_d

        # The input layer to [A,b].
        layer_input = Input(shape=(uv1_h,
                                   uv1_w,
                                   uv1_d),
                            dtype=tf.float32)

        # Translation vector component of the coupling function.
        # Remember, activation happens at the BEGINNING of a residual block, so do NOT activate after the convolutional layer.
        b_block = Convolution2D(filters=self.num_kernels,
                                kernel_size=self.kernel_size,
                                strides=(1,1),
                                padding='same',
                                kernel_initializer=self.init)(
                                    layer_input)

        for i in tf.range(self.num_res_blocks):

            b_block = dilated_residual_block(b_block,
                                             nb_channels_in=self.num_kernels,
                                             nb_channels_out=self.num_kernels,
                                             _which_dilations=self.which_dilations,
                                             ksize=self.kernel_size,
                                             cardinality=self.cardinality,
                                             ln=self.LAYER_NORM,
                                             init=self.init)

        # Remember, activation happens at the BEGINNING of a residual block, so DO activate after the last residual block.
        b_block = LeakyReLU()(b_block)

        if self.LAYER_NORM:

            # Layer normalization acts over spatial and channel dimensions simultaneously. It is also possible to normalize over the spatial dimensions independently for each channel by reshaping into (u_h*u_w, num_kernels) and normalizing over axis=-2, but experimentally I haven't found any reason not to use standard layer normalization.
            b_block = tf.keras.layers.Reshape((uv1_h * \
                                               uv1_w * \
                                               self.num_kernels,))(b_block)

            b_block = LayerNormalization(axis=-1)(b_block)

            b_block = tf.keras.layers.Reshape((uv1_h,
                                               uv1_w,
                                               self.num_kernels))(b_block)

        # Return to the output space. This is only the same as the input space for masks 0 and 1, or for masks 2 and 3 IF u_d is even.
        # The final layer has a LINEAR activation!
        b_block = Convolution2D(filters=uv2_d,
                                kernel_size=self.kernel_size,
                                strides=(1,1),
                                padding='same',
                                kernel_initializer=self.init)(
                                    b_block)

        # Scaling matrix component of the coupling function.
        # Remember, activation happens at the BEGINNING of a residual block, so do NOT activate after the convolutional layer.
        A_block = Convolution2D(filters=self.num_kernels,
                                kernel_size=self.kernel_size,
                                strides=(1,1),
                                padding='same',
                                kernel_initializer=self.init)(
                                    layer_input)

        for i in tf.range(self.num_res_blocks):

            A_block = dilated_residual_block(A_block,
                                             nb_channels_in=self.num_kernels,
                                             nb_channels_out=self.num_kernels,
                                             _which_dilations=self.which_dilations,
                                             ksize=self.kernel_size,
                                             cardinality=self.cardinality,
                                             ln=self.LAYER_NORM,
                                             init=self.init)

        # Remember, activation happens at the BEGINNING of a residual block, so DO activate after the last residual block.
        A_block = LeakyReLU()(A_block)

        if self.LAYER_NORM:
            A_block = tf.keras.layers.Reshape((uv1_h * \
                                               uv1_w * \
                                               self.num_kernels,))(A_block)
            A_block = LayerNormalization(axis=-1)(A_block)
            A_block = tf.keras.layers.Reshape((uv1_h,
                                               uv1_w,
                                               self.num_kernels))(A_block)

        # Return to the output space. This is only the same as the input space for masks 0 and 1, or for masks 2 and 3 IF u_d is even.
        A_block = Convolution2D(filters=uv2_d,
                                kernel_size=self.kernel_size,
                                strides=(1,1),
                                padding='same',
                                kernel_initializer=self.init)(
                                    A_block)

        # The final layer has a tanh activation!
        A_block = Activation('tanh')(A_block)

        # A learned scale factor for multiplying the tanh output.
        # To get the values of the scaling weights, inspect the model with model.layers, identify which layers are coupling_layers, and the last layer in each one will only have one weight (the scaling value). That looks like:
        # for i in [list of coupling layers]:
        #     print(model.layers[i].model_A.layers[-1].get_weights()[0])
        # My experience has been that the values will largely be close to 1.
        A_block = tanh_scaling_layer()(A_block)

        # A(uv1), b(uv1)
        model_A = Model(inputs=layer_input,
                        outputs=A_block)
        model_b = Model(inputs=layer_input,
                        outputs=b_block)

        return model_A, model_b

    def forward_coupling_law(self,
                             exp_A_u1,
                             b_u1,
                             u2_compressed):

        """
        Args:
            exp_A_u1: exp(A(u1)), an affine scaling (diagonal) matrix (actually just its diagonal components) obtained by feeding u1 into the coupling function.
            b_u1: b(u1), a translation vector obtained by feeding u1 into the coupling function.
            u2_compressed: the compressed non-u1 components of u.

        Returns:
            v2_compressed: exp(A(u1))*u2_compressed+b(u1)
        """

        v2_compressed = tf.math.multiply(exp_A_u1,
                                         u2_compressed) + b_u1

        return v2_compressed

    def inverse_coupling_law(self,
                             inv_exp_A_v1,
                             b_v1,
                             v2_compressed):

        """
        Args:
            inv_exp_Av1: [exp(A(v1))]^-1, an affine scaling (diagonal) matrix (actually just its diagonal components) obtained by feeding v1 into the coupling function, then inverting (because this is a diagonal matrix, this is equivalent to taking the element-wise inverse).
            bv1: b(v1), a translation vector obtained by feeding v1 into the coupling function.
            v2_compressed: the compressed non-v1 components of v.

        Returns:
            u2_compressed: [exp(A(v1))]^-1*(v2_compressed-b(v1))
        """

        u2_compressed = tf.math.multiply(inv_exp_A_v1,
                                         v2_compressed - b_v1)

        return u2_compressed

# =============================================================================
#     @tf.function
# =============================================================================
    def forward_and_Jacobian(self,
                             u,
                             sum_log_detJ,
                             zy):

        """
        Args:
            u: the input, u = (u1,u2), going in the xy -> zy direction.
            sum_log_detJ: the total sum of the Jacobian contributions to the loss so far.
            zy: the already-factored-out zy so far.

        Returns:
            v: the output, v = (v1,v2), going in the xy -> zy direction. v1 = u1 and v2 = A(u1)*u2 + b(u1).
            sum_log_detJ: the total sum of the Jacobian contributions to the loss, including the new contributions from this Layer.
            zy: the already-factored-out zy so far (not changed by this Layer).
        """

        # The input tensor must have the correct spatial and channel dimensions.
        u = tf.ensure_shape(u,
                            shape=[None,
                                   self.input_height,
                                   self.input_width,
                                   self.input_depth])

        # The shape of the input and output (after compression+decompression) is the same.
        u_shape = int_shape(u)

        # Need the uncompressed u1 to add back to v2 to get v after.
        u1_uncompressed = self.mask(u,
                                    self.which_mask,
                                    compress=False)
        v1 = u1_uncompressed

        # Need the compressed u1 to feed into the sub-models defining the coupling function.
        u1_compressed = self.mask(u,
                                  self.which_mask,
                                  compress=True)

        # Need the compressed u2 to feed into the coupling law.
        u2_compressed = self.mask(u,
                                  self.which_mask_complement,
                                  compress=True)

        # Feed u1 into the coupling function.
        # Note that A has a tanh activation at the end, followed by a scale factor, a:
        # -a<A(u1)<a. That means (1/(e^a))<exp(A)<(e^a).
        A_u1 = self.A_wrapper(u1_compressed)
        b_u1 = self.b_wrapper(u1_compressed)

        exp_A_u1 = tf.exp(A_u1)

        # Obtain the compressed output of the coupling law.
        v2_compressed = self.forward_coupling_law(exp_A_u1,
                                                  b_u1,
                                                  u2_compressed)

        # Decompress v2.
        v2 = self.decompress_mask(v2_compressed,
                                  self.which_mask_complement,
                                  u_shape)

        # Obtain the output, v.
        v = v1 + v2

        # What is the new Jacobian contribution from this layer?
        delta_log_detJ = tf.math.reduce_sum(A_u1,
                                            axis=[1,2,3])
        delta_log_detJ = tf.math.reduce_mean(delta_log_detJ)
        sum_log_detJ += delta_log_detJ

        return v, sum_log_detJ, zy

# =============================================================================
#     @tf.function
# =============================================================================
    def backward(self,
                 v,
                 zy):

        """
        Args:
            v: the input, v = (v1,v2), going in the zy -> xy direction.
            zy: the already-factored-out zy so far.

        Returns:
            u: the output, u = (u1,u2), going in the zy -> xy direction. u1 = v1 and u2 = A^-1(v1)*v2 - b(v1). Note that A^-1 is easy to calculate because A is a diagonal scaling matrix.
            zy: the already-factored-out zy so far (not changed by this Layer).
        """

        # The input tensor must have the correct spatial and channel dimensions.
        v = tf.ensure_shape(v,
                            shape=[None,
                                   self.input_height,
                                   self.input_width,
                                   self.input_depth])

        # The shape of the input and output (after compression/decompression) is the same.
        v_shape = int_shape(v)

        # Need the uncompressed v1 to add back to u2 to get u after.
        v1_uncompressed = self.mask(v,
                                    self.which_mask,
                                    compress=False)
        u1 = v1_uncompressed

        # Need the compressed v1 to feed into the sub-models defining the coupling function.
        v1_compressed = self.mask(v,
                                  self.which_mask,
                                  compress=True)

        # Need the compressed v2 to feed into the coupling law.
        v2_compressed = self.mask(v,
                                  self.which_mask_complement,
                                  compress=True)

        # Feed v1 into the coupling function.
        # Note that A has a tanh activation at the end, followed by a scale factor, a:
        # -a<A(u1)<a. That means (1/(e^a))<exp(A)<(e^a).
        A_v1 = self.A_wrapper(v1_compressed)
        b_v1 = self.b_wrapper(v1_compressed)

        inv_exp_A_v1 = tf.math.reciprocal(tf.exp(A_v1))

        # Obtain the compressed output of the coupling law.
        u2_compressed = self.inverse_coupling_law(inv_exp_A_v1,
                                                  b_v1,
                                                  v2_compressed)

        # Decompress u2.
        u2 = self.decompress_mask(u2_compressed,
                                  self.which_mask_complement,
                                  v_shape)

        # Obtain the output, u.
        u = u1 + u2

        return u, zy

###############################################################################

"""
DEFINE A CNN CLASS FOR REAL NVP THAT INHERITS FROM keras.Model

The TFP bijectors implementation of RealNVP is at https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/bijectors/real_nvp.py

The squeezing code was adapted from the RealNVP implementation at https://github.com/kmkolasinski/deep-learning-notes/blob/master/seminars/2018-10-Normalizing-Flows-NICE-RealNVP-GLOW/notebooks/flow_layers.py

Parts of the coupling layer code were adapted from the RealNVP implementation at https://github.com/taesungp/real-nvp/blob/master/real_nvp/nn.py although the relevant parts appear functionally similar to the RealNVP bijector linked above.
"""

class cFlow(Model):

    """
    Args:
        io_shape: [uv_h, uv_w, uv_d] for THE FIRST coupling layer (i.e. the shape of XY in the forward direction / ZY in the backward direction). Note that factoring zy in/out will CHANGE THE SHAPE of subsequent coupling layers! (The final output is reconstructed to have the same shape as the input.)
        x_d: The depth of the x component. Necessary for splitting x and y' to correctly calculate the log loss in the forward direction.
        squeeze_factor_block_list: A list whose entries are 0 and 1. There is one entry for each coupling block; 1 indicates the input is squeezed and factored AFTER the corresponding block, and 0 indicates that it is not. For example, [0,0,1,1] means there are four coupling blocks, and the input is squeezed once after each of the third and fourth blocks. Each coupling block consists of four successive coupling layers, with masks 0,1,2,3 (two checkerboard and two channel-wise masks) in that order.
        ResNeXt_block_list: A list whose entries are integers. There is one entry for each coupling block; the value of an entry indicates the number of ResNeXt blocks in each of the four A and b within that coupling block.
        num_kernels_list: A list whose entries are integers. There is one entry for each coupling block; the value of an entry indicates the number of convolutional kernels in each convolutional layer CORRESPONDING TO THE CHANNEL-WISE MASKS in the A and b within that coupling block. The checkerboard masks have half spatial dimensions and double channel dimensions compared to the channel-wise masks, so they have half as many kernels. Thus, each entry in num_kernels_list must be divisible by 2.
        cardinality_list: A list whose entries are integers. There is one entry for each coupling block; the value of an entry indicates the number of branches in residual block CORRESPONDING TO THE CHANNEL-WISE MASKS in the A and b within that coupling block. The number of kernels must be evenly divisible by the cardinality. Additionally, because there are half as many kernels in the checkerboard-masked layers, the cardinality for those layers is half as large; thus, each entry in this list must also be divisible by 2.
        lambda_y: Weight for the loss term ||y-y'||_1, a surrogate for the log likelihood under p(Y). It's important that this term dominate at the start of training; in my experience, in most cases, it takes around one epoch for this term to stabilize. This is the sole manually-set hyperparameter in the loss, but (unlike most such hyperparameters) it should have a wide range of acceptable values. lambda_y=100 has worked well for all my experiments.
        ksize: the size of the convolutional kernels in the coupling blocks. Currently, this is the same for all kernels in the model, and only square kernels of shape (ksize,ksize) are supported.
        LAYER_NORM: Boolean. Whether or not the ResNeXt blocks will have layer normalization or not.
        DILATIONS: Boolean. Whether or not the residual blocks have parallel dilated branches or not. If so, each block has parallel branches with dilations [d1,d2,d3,...] under certain circumstances. (In this code, only square dilations are supported, i.e. the kernel is dilated by the same factor in both spatial dimensions.)
            1) Dilations are chosen such that a given dilation is tiled exactly (ksize-1) times (with overlapping edge pixels) by the next-smaller dilation. That is, dksize_(i+1) = (ksize-1)*(dksize_i-1)+1 For 3x3 kernels, that means dilation factors [1,2,4,...]; for 4x4 kernels, that means dilation factors [1,3,9,...].
            2) There is a minimum spatial size for an input, below which dilated kernels of a certain size won't provide any benefit. The threshold is somewhat subjective; I chose to include dilations in the block only when the kernel size in a given spatial dimension follows: spatial_size >= 2*dilated_kernel_size - 1. The size of a dilated kernel in a particular spatial dimension is given by dksize = ksize + (ksize-1)(d-1), where d is the dilation factor in that dimension (a dilation factor of 1 means an undilated kernel). So, if DILATIONS==True, start with d1=1 (always the same), so dksize_i=ksize. Iteratively calculate dksize_(i+1) = (ksize-1) * (dksize_i-1) + 1 until dksize_(i+1) > (uv_hw + 1)/2. Do NOT include that d_(i+1); include all lower dilations in the residual block.
            3) I assume that fewer large dilations are required, in part because a deep network will have a large receptive field towards the bottom, in part because the model is spatially squeezed as zy is factored out, and in part because I expect the pixel correlations at larger length scales will be simpler in some sense, requiring fewer kernels to capture adequately. As a result, I divide the number of kernels in each layer - for BOTH sets of masks - by the dilation factor. Thus, for each layer, the number of kernels divided by the cardinality (this ratio is constant for all masks in a coupling block, because both the number of kernels and the cardinality scale by the same amount between the two mask types) must itself be evenly divisible by the dilation factor.
        init: which initializer to use when randomly initializing the model.

    Returns:
        The conditional flow model specified.
    """

    def __init__(self,
                 io_shape,
                 x_d,
                 squeeze_factor_block_list,
                 ResNeXt_block_list,
                 num_kernels_list,
                 cardinality_list,
                 lambda_y=100,
                 ksize=3,
                 LAYER_NORM=True,
                 DILATIONS=True,
                 init=tf.keras.initializers.Orthogonal(gain=0.1)):

        super(cFlow, self).__init__()

        self.io_shape = io_shape
        self.x_d = x_d
        self.squeeze_factor_block_list = squeeze_factor_block_list
        self.ResNeXt_block_list = ResNeXt_block_list
        self.num_kernels_list = num_kernels_list
        self.cardinality_list = cardinality_list
        self.lambda_y = lambda_y
        self.ksize = ksize
        self.LAYER_NORM = LAYER_NORM
        self.DILATIONS = DILATIONS
        self.init = init

        # The lengths of the various lists should match.
        assert len(self.squeeze_factor_block_list)== \
               len(self.ResNeXt_block_list) == \
               len(self.num_kernels_list)   == \
               len(self.cardinality_list), \
               'squeeze_factor_block_list, ResNeXt_block_list, num_kernels_list, and cardinality_list must all have the same length.'

        # No odd spatial dimensions allowed.
        # io_shape[0]=height, io_shape[1]=width.
        assert not self.io_shape[0] % 2 \
           and not self.io_shape[1] % 2, \
           'The model input and output must have spatial dimensions divisible by 2.'

        # Each entry in num_kernels_list must be divisible by 2, because checkerboard-masked inputs have half as many kernels per layer.
        for num_kernels in self.num_kernels_list:
            assert not num_kernels % 2, \
            'The number of kernels in each layer must be divisible by 2.'

        # For the same reason, each entry in cardinality_list must be divisible by 2.
        for cardinality in self.cardinality_list:
            assert not cardinality % 2, \
            'The cardinality in each layer must be divisible by 2.'

        # The only allowed entries in `squeeze_factor_block_list` are 0 and 1.
        for squeeze_or_not in self.squeeze_factor_block_list:
            assert squeeze_or_not in [0,1], \
            'The only allowed entries in squeeze_factor_block_list are 0 and 1.'

        # The total number of coupling blocks is:
        self.num_coupling_blocks = len(self.squeeze_factor_block_list)

        # Get the spatial dimensions of each coupling block.
        # The spatial dims of the coupling block will be (io_shape[0],io_shape[1]) * 1/(2**s), where s is the number of times u has been squeeze/factored.
        # After squeezing/factoring, the CHANNEL DEPTH remains the same, but the SPATIAL DIMENSIONS have each been halved.
        # Also keep track of the number of previous factor-outs for use in setting up the next factor layer.
        scale_list = []
        num_prev_factors_list = []
        scale_flag = 0
        num_prev_factors = 0

        for i in tf.range(len(self.squeeze_factor_block_list)):

            # Squeeze/factor happens AFTER the coupling layer in a coupling block, so the scale doesn't change until the next block.
            if i == 0:
                squeeze_or_not = 0
            else:
                squeeze_or_not = self.squeeze_factor_block_list[i-1]

            # Get the list of scales for obtaining each coupling block's i/o shape.
            if not scale_flag: # The first block MUST mix all variables.
                scale_list.append(1)
                scale_flag = 1
            else:
                scale_list.append(2**squeeze_or_not * scale_list[-1])

            # Get the list of the number of previous factorings.
            num_prev_factors += squeeze_or_not
            num_prev_factors_list.append(num_prev_factors)

        self.scale_list = np.array(scale_list)
        self.num_prev_factors_list = np.array(num_prev_factors_list)

        # input/output shapes for each coupling block:
        io_shape_list = []
        for i in tf.range(self.num_coupling_blocks):
        #for scale in self.scale_list:
            scale = self.scale_list[i]

            # Ensure that the cumulative scale (MULTIPLIED BY 2 BECAUSE THE CHECKERBOARD-MASKED INPUTS ARE HALVED IN SPATIAL DIMENSIONS) divides evenly into the original spatial dimensions.
            # io_shape[0]=height, io_shape[1]=width.
            assert not io_shape[0] % (scale * 2) \
               and not io_shape[1] % (scale * 2), \
               f'The cumulative scale (multiplied by 2 because the checkerboard-masked u/v are halved in spatial dimensions) must divide evenly into the original i/o spatial dimensions. This failed at block {i}, with i/o shape = {(io_shape[0],io_shape[1])} and scale*2 = {scale*2}.'

            io_shape_list.append([np.int(io_shape[0] / scale), # height
                                  np.int(io_shape[1] / scale), # width
                                  io_shape[2] * scale])                # depth

        self.io_shape_list = np.array(io_shape_list)

        # Create a list of masks for each coupling block. Here, these all have the same order, [0,1,2,3], but could as well be shuffled or manually set.
        # Masks can be of the following types:
        # 0: checkerboard 1 (nonzero top left entry)
        # 1: checkerboard 2 (zero top left entry)
        # 2: channel 1 (nonzero first channel, zero second channel)
        # 3: channel 2 (zero first channel, nonzero second channel)
        # Each coupling block consists of four coupling layers, one for each of the above, ordered (0,1,2,3). That is, for index 0, u1 has a nonzero top left entry and u2 has a zero top left entry, etc.
        self.u1_mask_indices = []
        for i in tf.range(self.num_coupling_blocks):
            u1_mask_indices_this_block = [0,1,  # checkerboard
                                          2,3] # channelwise

            self.u1_mask_indices.append(u1_mask_indices_this_block)

        # Get the dilation factors.
        if self.DILATIONS:

            # The checkerboard- and channel-wise-masked coupling blocks may have different allowed dilations, due to the smaller spatial dimensions in the checkerboard-masked compressed representations. (Note that because adjacent spatial dimensions are stacked as channels, each dilation in the checkerboard-masked compressed representation effectively encompasses a spatial extent twice as large as the actual spatial dimensions of the kernels.)
            self.dilations_list = []

            for block_io_shape in self.io_shape_list:

                dilations_dict = {'checkerboard' : [],
                                  'channelwise'  : []}

                # Only square dilations are supported in this implementation.
                # Find the SMALLEST spatial dimension in the input (the input may not be square). The largest allowed dilation for each block will be just over half of that dimension. Because the spatial dimensions of each block change with each squeeze/factor, successive blocks may allow fewer dilations.
                block_smallest_spatial_dim_channelwise = min(block_io_shape[0],
                                                             block_io_shape[1])
                block_smallest_spatial_dim_checkerboard = \
                    block_smallest_spatial_dim_channelwise / 2

                # Non-dilated kernels will ALWAYS be used.
                d_iplus1 = 1

                sanity_check = 0
                dksize_iplus1 = self.ksize
                # while loop using the (larger) channelwise spatial dimensions. This continues until the next larger dilated convolution would be too large to tile the space twice (with overlapping edge pixels) in the smallest dimension.

                # Regardless of the size of the input, always implement at least the undilated convolutions.
                if dksize_iplus1 > (block_smallest_spatial_dim_channelwise + 1) / 2:

                    dilations_dict['channelwise'].append(d_iplus1)
                    dilations_dict['checkerboard'].append(d_iplus1)

                else:

                    while \
                    dksize_iplus1 < (block_smallest_spatial_dim_channelwise + 1) / 2:

                        # Sanity check for the while loop. There shouldn't be anywhere near 10 dilations for smaller images!
                        assert sanity_check < 10, \
                        'The dilation while loop ran unexpectedly many iterations. Either you are using a dataset of large images (in which case, manually change the parameter that triggers this warning), or something has gone wrong.'

                        # Append the current dilation factor to the appropriate list in the dictionary.
                        dilations_dict['channelwise'].append(d_iplus1)

                        # Extra if statement for the (smaller) checkerboard spatial dimensions.
                        if \
                        d_iplus1 < (block_smallest_spatial_dim_checkerboard + 1) / 2:
                            dilations_dict['checkerboard'].append(d_iplus1)

                        # The next larger dilated kernel.
                        dksize_iplus1 = (self.ksize - 1) * (dksize_iplus1 - 1) + 1

                        # The corresponding dilation factor.
                        d_iplus1 = \
                        ((dksize_iplus1 - self.ksize) / (self.ksize - 1)) + 1

                        # Increment the sanity check.
                        sanity_check += 1

                self.dilations_list.append(dilations_dict)

            # Ensure that the ratio num_kernels / cardinality is evenly divisible by all of the channelwise dilation factors in this block. (Channelwise, because it may have one more dilation than checkerboard.)
            for i in tf.range(self.num_coupling_blocks):
                NKC = self.num_kernels_list[i] / self.cardinality_list[i]
                for dilation in self.dilations_list[i]['channelwise']:
                    assert not NKC % dilation, \
                    f'The ratio (number of kernels / cardinality) must be evenly divisible by each dilation factor used in that coupling block. This failed in coupling block {i}.'

        # Distribution of the latent space for one element (e.g., pixel) of X: p_Z(z) = N(0,1).
        # Has the same dimensionality as one element of X (it will be extended across the spatial dimensions later).
        self.distribution = tfp.distributions.MultivariateNormalDiag(
                                loc=[0]*self.x_d,
                                scale_diag=[1]*self.x_d)

        """
        BUILD THE MODEL
        """

        # This list will contain all layers in the model from end to end EXCEPT the input layer.
        self.layers_list = []

        # This list will contain ONLY squeeze/factor layers, for reshaping zy from a 1x1xN tensor to have the same shape as xy (forward direction), or for reshaping zy back into a 1x1xN tensor (backward direction).
        # The squeeze/factor layers should take up very little memory, so the cost for having a separate list for reshaping should be small.
        self.squeeze_factor_layers_list = []

        for i in tf.range(self.num_coupling_blocks): # Which coupling block.

            # Coupling layers come first.
            for mask in self.u1_mask_indices[i]:  # Which coupling layer.

                # The largest dilation MAY differ between masks [0,1] and [2,3].
                if mask in [0,1]:
                    which_dilations = self.dilations_list[i]['checkerboard']
                elif mask in [2,3]:
                    which_dilations = self.dilations_list[i]['channelwise']

                layer = coupling_layer(in_shape=self.io_shape_list[i],
                                       which_mask=mask,
                                       num_res_blocks=self.ResNeXt_block_list[i],
                                       cardinality=self.cardinality_list[i],
                                       num_kernels=self.num_kernels_list[i],
                                       kernel_size=self.ksize,
                                       init=self.init,
                                       LAYER_NORM=self.LAYER_NORM,
                                       which_dilations=which_dilations)

                # Build the layer.
                layer.build(input_shape=[None,
                                         self.io_shape_list[i][0],
                                         self.io_shape_list[i][1],
                                         self.io_shape_list[i][2]])

                self.layers_list.append(layer)

            # Squeeze and factor layers follow the coupling layers.
            if self.squeeze_factor_block_list[i] == 1:

                layer = squeeze_layer()

                # Build the layer.
                layer.build(input_shape=[None,
                                         self.io_shape_list[i][0],
                                         self.io_shape_list[i][1],
                                         self.io_shape_list[i][2]])

                self.layers_list.append(layer)
                self.squeeze_factor_layers_list.append(layer)

                num_prev_factors = self.num_prev_factors_list[i]
                layer = factor_out_zy_layer(num_prev_factors)

                # Build the layer.
                layer.build(input_shape=[None,
                                         self.io_shape_list[i][0],
                                         self.io_shape_list[i][1],
                                         self.io_shape_list[i][2]])

                self.layers_list.append(layer)
                self.squeeze_factor_layers_list.append(layer)

        # We're customizing the behavior of `model.fit()`. That means creating a Metric instance to track loss (in this case, Mean).
        self.loss_tracker = metrics.Mean(name='loss')
        self.z_loss_tracker = metrics.Mean(name='z_loss')
        self.y_loss_tracker = metrics.Mean(name='y_loss')
        self.detJ_loss_tracker = metrics.Mean(name='detJ_loss')

    @property
    def metrics(self):
        """
        A list of the model's metrics.
            This model is trained by maximum likelihood using the change of variables from X-space to Z-space. The probability distribution in data space can be written as:
            p_{XY}(x,y') = p_{ZY}(z,y)|det J_f(x,y')|
            where the change of variables is given by the function
            f : XY' -> ZY
            with z,y = f(x,y').
            We can rewrite this as
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

# =============================================================================
#     @tf.function
# =============================================================================
    def call(self,
             uv,
             direction=-1):

        """
        Args:
            uv: either xy' (direction=1) or zy (direction=-1).
            direction: +1 for mapping from xy'->zy, -1 for mapping from zy->xy'.

        Returns:
            vu: either zy (direction=1) or xy' = (direction=-1).
                Note: (z,y) = f(x,y') (direction 1) or
                      (x,y') = f^-1(z,y) (direction -1)
            log_detJ (only if direction==1): the log of the determinant of the Jacobian associated with the map XY' -> ZY. Because det J is a product of exponential functions, which are always positive, |det J| = det J.

        By construction, in RealNVP det J is easy to calculate. We know p_Z(f(x)) exactly, because p_Z(z)=N(0,1). Therefore, obtaining the total loss is basically trivial.

        NOTE: y' is NOT independent of x. Specifically, this model enforces that x is dependent on y, and if y'=y, it cannot be the case that x is not dependent on y' (and vice versa). Think about it like this: suppose your classes are very different from one another. Then there will be areas in x-space that class A might occupy, but class B never will, and vice versa. Thus, the class (y') partially determines x.
        """

        if direction == 1:

            log_detJ = 0
            zy = None

            for layer in self.layers_list:

                uv, log_detJ, zy = layer.forward_and_Jacobian(uv,
                                                              log_detJ,
                                                              zy)

            # If there are no squeeze/factor layers, just pass uv through.
            if len(self.squeeze_factor_layers_list)==0:

                vu = uv

            # Otherwise, feed uv and zy backwards through only the squeeze/factor layers.
            else:

                zy = tf.concat([zy,uv], 3)

                # zy has all the correct elements, but they've been squeezed into a tensor of shape 1x1xN. To properly compute the loss, zy needs to be returned to the original shape of xy. This can't be done simply by reshaping, because the elements are moved around in a complicated way by all the squeeze/factoring.
                # The only way to reshape zy, while ensuring each element has the same spatial/channel position as the corresponding element in xy, is to reverse the squeeze/factoring operations.
                vu = None
                for layer in reversed(self.squeeze_factor_layers_list):

                    vu, zy = layer.backward(vu,
                                            zy)

            return vu, log_detJ

        elif direction == -1:

            # Here, the input tensor has to be reshaped to 1x1xN, again ensuring each element goes to the right spatial/channel position.
            # This requires a dummy log_detJ argument because the Jacobian is included in the forward operation.
            zy = None
            log_detJ = None

            # Don't try to call layers from an empty list.
            if not len(self.squeeze_factor_layers_list)==0:

                for layer in self.squeeze_factor_layers_list:

                    uv, log_detJ, zy = layer.forward_and_Jacobian(uv,
                                                                  log_detJ,
                                                                  zy)

            # Now, pass everything through in the reverse direction.
            vu = uv

            for layer in reversed(self.layers_list):

                vu, zy = layer.backward(vu,
                                        zy)

            return vu

    def log_loss(self,
                 xy):

        """
        Args:
            xy: the concatenation of x, a point in the X domain, and y, the conditioning information.

        Returns:
            log_loss: the negative log likelihood training objective (using L1 distance as a surrogate metric for likelihood under p_Y).
            log_loss = - log p_{X,Y'}(x,y')
                     = - log p_Z(f_Z(x,y'))
                       + lambda_Y * ||f_Y(x,y') - y'||_1
                       - log |det J_f(x,y')|
        """

        x_d = self.x_d

        # Need to know the original conditioning information.
        y_prime = xy[...,
                     x_d:]

        (zy,
         log_detJ) = self(xy,
                          1) # direction = +1

        # Separate z and y.
        z = zy[...,
               :x_d]
        y = zy[...,
               x_d:]

        # distribution.log_prob(z) has shape (batch, h, w).
        log_likelihood_z = tf.math.reduce_sum(self.distribution.log_prob(z),
                                            axis=[1,2])

        # y and y' have shapes (batch, h, w, y_depth)
        log_likelihood_y = -self.lambda_y * \
                            tf.math.reduce_sum(tf.abs(y - y_prime),
                                               axis=[1,2,3])

        log_likelihood = tf.math.reduce_mean(log_likelihood_z + \
                                             log_likelihood_y) + \
                                             log_detJ

        # Want the negative of this.
        log_loss = -log_likelihood

        # Return the total loss as well as each component of it.
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

        return {'loss' : self.loss_tracker.result(),
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