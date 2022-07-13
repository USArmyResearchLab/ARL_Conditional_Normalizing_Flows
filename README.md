# MAXIMUM LIKELIHOOD ON THE JOINT (DATA, CONDITION) DISTRIBUTION FOR SOLVING ILL-POSED PROBLEMS WITH CONDITIONAL FLOW MODELS

This project started with the goal of solving ill-posed problems in a statistically rigorous way. For any ill-posed problem, you may have *partial* information about the solution, but not enough to fully determine it. In that case, the best you can do is sample from the distribution of possible solutions and draw conclusions from their statistics. As an example, if you input a low-resolution image, it can produce the distribution of high-resolution images consistent with the input.

This is an implementation of the code used in [1]; please see that paper for a full discussion of the motivation and method. In summary, the code allows you to build and train a conditional flow model on the joint distribution of data (e.g., high-resolution images) and some conditioning information (e.g., class label, low-resolution images, etc.). When a condition is specified, the model will sample from the conditional distribution of data-space points consistent with the condition - for example, it might sample from high-resolution images consistent with a low-resolution condition. Flow models are trained only on the likelihood of reproducing the target distribution, unlike GANs or VAEs, and model the density of that distribution directly. GANs do not optimize on likelihood at all, and VAEs optimize on the lower bound of the likelihood, rather than the likelihood itself.

This method is a generalization of flow models to the conditional case based on the non-conditional RealNVP model [2]. There are numerous other attempts to so generalize RealNVP that non-invertibly embed the condition into the map, the prior, or both, or add additional terms to the likelihood loss function [3-11]. My method is simple, fully invertible, and only optimizes the likelihood. I'm aware of one other method that meets those criteria [12], and it is considerably more computationally and architecturally complex than mine.

There are two major components to this code: the "toy" problems and the "real" problems. The toy problems illustrate the concepts with very low-dimensional point clouds: 2 dimensions for the data, and 1 for the condition. The real problems extend this to class-conditioned image generation and super-resolution of highly degraded images.

Note that this method, by itself, is *not* enough to obtain statistically rigorous results - flow models require careful engineering of their inductive bias to achieve this [13]. However, my method *does* have the same properties as non-conditional flow models, and in principle can therefore be engineered in the same way (this is an important part of future work).

Figures and more details are available in [1].

Code was adapted from implementations of [2] and [14-16].

[1] John Hyatt. Maximum likelihood on the joint (data, condition) distribution for solving ill-posed problems with conditional flow models. Pre-publication (2022). Manuscript included as Manuscript.pdf in this directory.

[2] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using Real NVP. *CoRR* abs/1605.08803 (2016).

[3] Ryan Prenger, Rafael Valle, and Bryan Catanzaro. Waveglow: A flow-based generative network for speech synthesis. *CoRR*, abs/1811.00002 (2018).

[4] Rui Liu, Yu Liu, Xinyu Gong, Xiaogang Wang, and Hongsheng Li. Conditional adversarial generative flow for controllable image synthesis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2019.

[5] Christina Winkler, Daniel E. Worrall, Emiel Hoogeboom, and Max Welling. Learning likelihoods with conditional normalizing flows. *CoRR*, abs/1912.00042 (2019).

[6] You Lu and Pert Huang. Structured output learning with conditional generative flows. *Proceedings of the AAAI Conference on Artificial Intelligence* **34**(04):5005-5012 (2020).

[7] Robin Rombach, Patrick Esser, and Bjorn Ommer. Network-to-network translation with conditional invertible neural networks. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, *Advances in Neural Information Processing Systems*, volume 33, pages 2784-2797. Curran Associates Inc, 2020.

[8] Lynton Ardizzone, Radek Mackowiak, Ullrich Kothe, and Carsten Rother. Exact information bottleneck with invertible neural networks: Getting the best of discriminative and generative modeling. *CoRR*, abs/2001.06448 (2020).

[9] Peter Sorrenson, Carsten Rother, and Ullrich Kothe. Disentanglement by nonlinear ICA with general incompressible-flow networks (GIN). *CoRR*, abs/2001.04872 (2020).

[10] Lynton Ardizzone, Jakob Kruse, Carsten Luth, Niels Bracher, Carsten Rother, and Ullrich Kothe. Conditional invertible neural networks for diverse image-to-image translation. In Z. Akata, A. Geiger, and T. Sattler, editors, *Pattern Recognition*, pages 373-387. Cham, 2021. Springer International Publishing.

[11] Govinda Anantha Padmanabha and Nicholas Zabaras. Solving inverse problems using conditional invertible neural networks. *Journal of Computational Physics* **433**:110194 (2021).

[12] Albert Pumarola, Stefan Popov, Francesc Moreno-Noguer, and Vittorio Ferrari. C-Flow: Conditional generative flow models for images and 3D point clouds. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2020.

[13] Polina Kirichenko, Pavel Izmailov, and Andrew Gordon Wilson. Why normalizing flows fail to detect out-of-distribution data. *Advances in neural information processing systems*, **33**:20578-20589 (2020).

[14] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. *CoRR* abs/1512.03385 (2015).

[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. *CoRR* abs/1603.05027 (2016).

[16] Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. *CoRR* abs/1611.05431 (2016).

## Instructions for training a toy model

The code itself is extensively documented with in-line comments, including explanations of the underlying math. It is written in Python using TensorFlow. To run it, you need TensorFlow 2.7 (or higher), TensorFlow Probability 0.15 (or higher), and their dependencies. Note: Slightly earlier versions will probably work, but certain functions were changed from earlier versions of Tensorflow 2, and the code may break or not function as intended if you use older versions.

The main script for the toy problems is TOYcINN.py, which calls TOYcINN_make_model.py and TOYcINN_make_datasets.py. It contains the code for defining model parameters, filepaths, options, etc., as well as the model training script. Unless you want to build off of my code, you will probably only change parameters in this file, when working with toy problems. The first part of the code, "USER-SPECIFIED HYPERPARAMETERS", is where you will make these changes.

At minimum, you will want to specify (or confirm that the defaults are OK):
	which_dataset: This code supports two discrete datasets (crescents and mixed) and one continuous dataset.
	num_coupling_blocks
	intermediate_dims: How many nodes per densenet layer.
	num_layers: How many densenet layers per coupling block.
	patience: Used in the early stopping callback.
	num_annealing_epochs: Starting with pure noise, how many epochs to anneal out the noise and begin training on clean data.
	TRAIN: If False, the code will just build the model, but not train it.
	SAVE_path: Where to save the trained model.
	LOAD_path: If specified, the model's weights will be loaded from this filepath. NOTE: You must build the model using the same architecture the loaded weights were taken from.
	PLOT: Whether or not to plot results after training.
	mask_indices: If not specified, the model will order masks randomly within coupling blocks. If it is specified, the model will be built with the specified order.
These are described with additional in-line documentation in the scripts. There are additional hyperparameters that you may wish to change, that allow you to vary the dataset properties (for example, whether the two crescents in the 'crescents' dataset overlap or not).

Based on your choices, the script will build your chosen dataset as a TensorFlow dataset object, build a model with your chosen hyperparameters, train it according to your chosen training scheme, and save it at your chosen filepath. The script calls two auxiliary scripts, TOYcINN_make_datasets.py and TOYcINN_make_model.py, to do this.

The toy model is not coded very efficiently, as it is intended for illustrative purposes only. All the neural networks are simple dense nets, the flow model has a very simple design, and it doesn't include many of the features present in the real problem model. However, using the default hyperparameters, you can train a model that performs well on the 'crescents' dataset in only a few minutes with a laptop CPU - it should take only a few seconds per epoch, and only 50 or so epochs. It takes a deeper network to perform well on the 'continuous' dataset, because it is trying to "learn rotations" using only affine scaling and translation transformations. The 'mixed' dataset can be easy or hard, depending on how many (and which) classes you specify.

## Instructions for training a real model

The process here is similar to that for the toy models, but there are some extra steps. The default hyperparameter values should work for both class-conditioned image generation and the 2nd step of super-resolution. You will have to manually remove the squeeze/factor layer in squeeze_factor_block_list and adjust the other hyperparameters to have constant values across all coupling blocks for the first step of super-resolution.

The code itself is extensively documented with in-line comments, including explanations of the underlying math. It is written in Python using TensorFlow. To run it, you need TensorFlow 2.7 (or higher), TensorFlow Probability 0.15 (or higher), and their dependencies. Note: Slightly earlier versions will probably work, but certain functions were changed from earlier versions of Tensorflow 2, and the code may break or not function as intended if you use older versions.

### Instructions for creating the dataset

Unlike the toy model, the data used to train the real models is loaded from a preprocessed TFRecords file. You must create that TFRecords file before starting training. This is done with create_tfrecords.py.

You must specify:
	which_dataset: This code supports MNIST and fashion-MNIST.
	path: where to save the .tfrecords file.
	which_classes: MNIST and fMNIST have 10 classes. You may include anywhere between 2 and 10 of these.
	COMBINED_DATASET: If True, all classes will be combined in one .tfrecords file; if False, there will be one .tfrecords file per class.
In particular, COMBINED_DATASET should be True if you are using it to train a super-resolution model, and False if you are using it to train a class-conditioned generative model.

This will save one or more .tfrecords files at the path you specify, which will be loaded during training.

### Instructions for pre-training the model on noise

Experimentally, I have not found a huge difference if you skip this step, although the early training appears to be more stable, and in principle it should yield better results. If you are testing different architectures, I recommend skipping pre-training to save time until you are satisfied with your architecture, and including this step on the final training.

This step forces the flow model to learn simple properties of the input first. The x-component of the input is just random Gaussian noise; for a Gaussian prior, that is precisely what maximizes the likelihood of the p_Z term in the loss. The y-component is supposed to remain unchanged after passing through the model, so it also forces the model to learn a real identity function on that component. Learning these properties early means that, during the annealing step, the model will have fewer things to learn, and it can gradually incorporate the data properties until annealing is over.

You will use the script conv_pre_training_cINN_on_noise.py for this. Specify your hyperparameters *exactly* as you will during training; the output of this script is a saved weights file that you will then load during the main training script. This script will call conv_cINN_make_model.py and conv_cINN_base_functions.py to build the model exactly the same as during regular training; and it will constantly regenerate new Gaussian noise every epoch, so the model doesn't learn to overfit.

### Instructions for training the model on data

The main script for the real problems is conv_cINN.py, which calls conv_cINN_make_model.py and conv_cINN_base_functions.py. It contains the code for defining model parameters, filepaths, options, etc., as well as the model training script. Unless you want to build off of my code, you will probably only change parameters in this file, when working with real problems. The first part of the code, "USER-SPECIFIED HYPERPARAMETERS", is where you will make these changes.

At minimum, you will want to specify (or confirm that the defaults are OK):
	model_type: super-resolution, either 4x4 -> 2x2 or 2x2 -> 1x1; or class-conditional generation.
	which_dataset: fashion-MNIST or MNIST
	data_classes: A list of the classes to train on. This must correspond to the classes in the dataset you made using create_tfrecords.py.
	RESIDUAL: Only for the super-resolution case. Whether or not the output should be a residual image or the full prediction.
	DISCRETE_LOGITS: If True, use a form of logit(intensity) rather than raw intensity values for pixels in x.
	squeeze_factor_block_list: A list of entries 0 or 1. Each one represents a coupling block in the model. For 0 entries, there is no squeeze/factoring after that block; for 1 entries, half of the internal representation is squeeze/factored out after that block.
	ResNeXt_block_list: A list of the number of ResNeXt blocks for neural networks in a given coupling block.
	num_kernels_list: A list of the number of kernels for neural networks in a given coupling block. These should reduce by a factor of 2 after every '1' entry in squeeze_factor_block_list.
	cardinality_list: A list of the ResNeXt cardinality for neural networks in a given coupling block. These should reduce by a factor of 2 after every '1' entry in squeeze_factor_block_list.
	kernel_size
	which_dilations: List of the dilations to use in each neural network. Note that if a given network's inputs are too small, larger dilations may be ignored.
	LAYER_NORM: Whether or not to include LayerNormalization layers in the neural networks.
	batch_size
	patience: Used in the early stopping callback.
	model_CHECKPOINT_path: the path where the model will save a checkpoint, with a period defined by checkpoint_epochs.
	hist_CHECKPOINT_path: the path where the model will update the training history, with a period defined by checkpoint_epochs.
	checkpoint_epochs: how often to save checkpoints.
	num_annealing_epochs: either None (for no annealing) or the number of epochs to anneal out noise at the start of training.
	num_epochs: number of epochs to train on clean data (unless cut short by the early stopping callback)
	TRAIN: If False, the code will just build the model, but not train it.
	SAVE_path: Where to save the trained model.
	dataset_path: directory where the dataset created in the first step is saved.
	LOAD_path: If specified, the model's weights will be loaded from this filepath. NOTE: You must build the model using the same architecture the loaded weights were taken from.
	
You may also specify:
	init
	learning_rate

These are described with in-line documentation in the scripts.

Note that there are restrictions on which values are allowed for some hyperparameters; for example, if you have too many squeeze layers ('1' entries in squeeze_factor_block_list) and you reach an odd number, the code will not work; each entry in cardinality_list must factor evenly into the corresponding entries in num_kernels_list; and so on. Details are provided in the in-line documentation.

Once the model is fully trained, both the model and the training history will be saved.
