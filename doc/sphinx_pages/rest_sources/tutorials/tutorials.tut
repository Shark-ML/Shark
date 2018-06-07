Tutorials
=========

This page gives a gentle introduction into Shark. The quick tutorial section gives an introduction
into the most important core components. For neural network training, following the neural entwork tutorials is very helpful for a 
first step. If you are already familiar with the Shark architecture, the documentation of the key concepts
and list of classes can be found below:

======================================================	========================================================
Concept 						Class List
======================================================	========================================================
:doc:`concepts/library_design/models`			:doxy:`List<models>`
:doc:`concepts/library_design/losses` 			:doxy:`List<lossfunctions>`
:doc:`concepts/library_design/optimizers`		:doxy:`Gradient-Based Optimization<gradientopt>`,
							:doxy:`Direct-Search Optimizers<singledirect>`, 
							:doxy:`Multi-Objective Optimizers<multidirect>`
:doc:`concepts/library_design/objective_functions` 	:doxy:`List<objfunctions>`
:doc:`concepts/library_design/trainers`			:doxy:`Supervised Trainers<supervised_trainer>`,
							:doxy:`Unsupervised Trainers<unsupervised_trainer>`
:doc:`concepts/library_design/kernels` 			:doxy:`List<kernels>`
======================================================	========================================================

..
	* :doc:`concepts/library_design/stopping_criteria`

Quick tutorial
++++++++++++++++

In case ou are new to Shark, we give you a quick tour over the core components.
We first show how to set up either a traditional Makefile or a CMake file
for your application program. Then we move on to a simple Hello-World example
of what linear binary classification can look like in Shark. The third tutorial
illustrates the model-error-optimizer trias often encountered in Shark through
a simple regression task.

* :doc:`first_steps/your_programs`
* :doc:`first_steps/hello_shark`
* :doc:`first_steps/general_optimization_tasks`

..
	* :doc:`first_steps/when_to_stop`



Neural Networks
++++++++++++++++++++++
A very important class of machine-learning models are Neural Networks. This section
discusses the creation and training of multi-layer neural networks

* :doc:`algorithms/ffnet`
* :doc:`algorithms/DeepMNIST`
* :doc:`algorithms/autoencoders`
* :doc:`algorithms/variational_autoencoders`

Data Handling
+++++++++++++

.. _label_for_data_tutorials:

Since many machine learning algorithms work on real-world datasets, we extensively
cover Shark's :doxy:`Data` class as well as common operations on them:

* :doc:`concepts/data/datasets`
* :doc:`concepts/data/labels`
* :doc:`concepts/data/import_data`
* :doc:`concepts/data/dataset_subsets`
* :doc:`concepts/data/normalization`



Specific Machine-Learning Algorithms
++++++++++++++++++++++++++++++++++++

Here come tutorials for some selected algorithms implemented in Shark.
It must be said that this is only the tip of the iceberg, *many* more
machine learning algorithms and tools are provided by the library.

Let's start with some classical methods:

* :doc:`algorithms/pca`
* :doc:`algorithms/nearestNeighbor`
* :doc:`algorithms/lda`
* :doc:`algorithms/linearRegression`
* :doc:`algorithms/LASSO`
* :doc:`algorithms/kmeans`

Tree-based algorithms:

* :doc:`algorithms/rf`

Kernel methods -- support vector machine training and model selection:

* :doc:`algorithms/svm`
* :doc:`algorithms/svmModelSelection`
* :doc:`algorithms/svmLikelihoodModelSelection`
* :doc:`algorithms/lkc-mkl`
* :doc:`algorithms/linear-svm`
* :doc:`algorithms/kta`
* :doc:`algorithms/kernelBudgetedSGD`

Optimization:Direct-Search
++++++++++++++++++++++++++++++++++++

Shark offers many direct-search algorithms. The most important one is the CMA-ES in the single and multi-objective variants

* :doc:`concepts/optimization/directsearch`
* :doc:`algorithms/cma`
* :doc:`algorithms/mocma`
* :doc:`algorithms/MOOExperiment`

Restricted Boltzman Machines
++++++++++++++++++++++++++++++++++++
* :doc:`algorithms/rbm_module`
* :doc:`algorithms/binary_rbm`

Tools
+++++


Finally, we present functionality which are not machine learning facilities
themselves, but necessary or helpful tools.

For convenience, Shark provides a statistics class wrapper, as well as generic
support for serialization:

* :doc:`concepts/misc/statistics`
* :doc:`concepts/misc/serialization`


For Shark developers
++++++++++++++++++++


Note that Shark follows a

* :doc:`for_developers/codingconvention`.

If you contribute to Shark, you might also find these documents helpful:

* :doc:`for_developers/adding_docs`
* :doc:`concepts/optimization/conventions_derivatives`
* :doc:`concepts/library_design/batches`


