
The Versatility of Learning
===========================

The purpose of this tutorial is to demonstrate the versatility of
Shark for various learning tasks. Drawing on a simple binary
classification task similar to the one in the
:doc:`../../first_steps/hello_shark` tutorial, we will cover five
different learning methods in a single, consistent framework.  The
present tutorial assumes that the reader is already familiar with the
concepts of models and trainers that have been treated in more detail,
e.g., in the :doc:`../../first_steps/general_optimization_tasks`
tutorial.

We will start out with the (hopefully already familiar) structure of a
supervised learning experiment: ::

	#include <shark/Data/Dataset.h>
	#include <shark/Data/Csv.h>
	#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
	using namespace shark;

	int main()
	{
		ClassificationDataset traindata, testdata;
		import_csv(traindata, "data/train.csv", FIRST_COLUMN);
		import_csv(testdata, "data/test.csv", FIRST_COLUMN);

		// TODO: define a model and a trainer

		trainer.train(model, traindata);

		Data<unsigned int> prediction = model(testdata.inputs());

		ZeroOneLoss<unsigned int> loss;
		double error_rate = loss(testdata.labels(), prediction);

		std::cout << "model: " << model.name() << std::endl
			<< "trainer: " << trainer.name() << std::endl
			<< "test error rate: " << error_rate << std::endl;
	}

The program assumes two comma-separated-value (csv) files with
training and test data located in the sub-folder ``/data``. We assume
that the file content describes a two-class (binary) problem, with
labels 0 and 1. The program itself is still a stub, since the actual
model and trainer declarations are missing. The ``ZeroOneLoss``
computed the classification error replacing the loop at the end of the
:doc:`../../first_steps/hello_shark` tutorial.


Many Ways of Classifying Data
=============================

In the following we will demonstrate the versatility of Shark by
inserting five different learning methods into the above program
structure. In passing we will see how to circumvent typical pitfalls.

Linear Discriminant Analysis
----------------------------

Let us start with a classical linear method, namely linear discriminant
analysis (LDA). We need two more includes ::

	#include <shark/Models/LinearClassifier.h>
	#include <shark/Algorithms/Trainers/LDA.h>

and in the place of the "TODO" comment we insert ::

	LinearClassifier model;
	LDA trainer;

That's it! The program is ready to go. For build instructions refer to
the :doc:`../../first_steps/your_programs` tutorial.  You can learn
more on LDA in the :doc:`../../algorithms/lda` tutorial.


Nearest Neighbor Classifier
---------------------------

Let's move from the linear parametric LDA approach to a non-linear,
non-parametric approach.
The arguably simplest non-linear classifier is the nearest neighbor classifier.
This classifier is special in that it does not require a trainer. Let's
remove the LDA code and insert the following code in the appropriate
places: ::

	#include <shark/Models/NearestNeighborClassifier.h>

	unsigned int k = 3;   // number of neighbors
	KDTree<RealVector> kdtree(traindata.inputs());
	NearestNeighborClassifier<RealVector> model(traindata, &kdtree, k);

For the time being ignore the KDTree class, unless you already know what
it does (it represents a data structure that facilitates efficient nearest
neighbor search, in particular in low-dimensional spaces). We also remove
the line ::

	<< "trainer: " << trainer.name() << std::endl

since in this case we do not have a trainer object. Everything should
work right away. For more information on nearest neighbor
classification see the :doc:`../../algorithms/nearestNeighbor` tutorial.


You see, changing the learning method is really easy.
So let's try more.


Support Vector Machine
----------------------

Our next candidate is a non-linear support vector machine (SVM). We will
use a Gaussian radial basis function kernel: ::

	#include <shark/Models/Kernels/GaussianRbfKernel.h>
	#include <shark/Models/Kernels/KernelExpansion.h>
	#include <shark/Algorithms/Trainers/SvmTrainer.h>

	double gamma = 1.0;         // kernel bandwidth parameter
	double C = 10.0;            // regularization parameter
	GaussianRbfKernel<RealVector> kernel(gamma);
	KernelExpansion<RealVector> model(&kernel, true);    // true: decision function with bias parameter
	CSvmTrainer<RealVector> trainer(&kernel, C);

Quite simple, again. However, the attempt to compile this program
results in an error message (or, depending on your compiler, a pile of
hard-to-decrypt messages involving template issues). What went wrong?
The problem is that in Shark there exist (for good reasons) two
different conventions for representing classification labels and
predictions (also refer to the :doc:`../data/labels` tutorial). While
the LinearClassifier and NearestNeighborClassifier models output their
prediction as unsigned integers, the KernelExpansion outputs a RealVector
holding the value(s) of the SVM decision function. For binary classification
it contains a single entry whose sign indicates the prediction. Thus, we
have to turn the line ::

	Data<unsigned int> prediction = model(testdata.inputs());

into ::

	Data<RealVector> prediction = model(testdata.inputs());

Now predictions are stored as RealVectors. The next thing is that these
predictions are fed into the ZeroOneLoss. We change its definition into ::

	ZeroOneLoss<unsigned int, RealVector> loss;

where the first template parameter identifies the ground truth label
type (the type of test.label(n)) and the second template parameter is
the data type of model predictions (it can be dropped if the types
coincide). That's it; you are ready to enjoy the power of non-linear
SVM classification. Much more on SVMs cane be found in the special
SVM tutorials, starting with :doc:`../../algorithms/svm`.



Random Forest
-------------

There is more to explore in Shark. Let's try a random forest instead: ::

	#include <shark/Models/Trees/RFClassifier.h>
	#include <shark/Algorithms/Trainers/RFTrainer.h>

	RFClassifier model;
	RFTrainer trainer;

This one is really straightforward. For an introduction to random forests see the
:doc:`../../algorithms/rf` tutorial.


Neural Network
--------------

As a final example let's look at a more complex case, namely that of
feed forward neural network training. The most basic way of training
these models is by gradient-based minimization of the training error
(empirical risk), measured by some differentiable loss function such
as the squared error or the cross entropy. The computation of the
gradient is built into the neural network class (back-propagation
algorithm), but of course there are various options for solving the
underlying optimization problem. The :doc:`../../first_steps/general_optimization_tasks`
tutorial touches this topic. Here - for consistency with the previous
examples - we will encapsulate the optimization process into the
familiar model and trainer classes. ::

	#include <shark/Models/FFNet.h>
	#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>
	#include <shark/ObjectiveFunctions/ErrorFunction.h>
	#include <shark/Algorithms/GradientDescent/Rprop.h>
	#include <shark/Algorithms/StoppingCriteria/MaxIterations.h>
	#include <shark/Algorithms/Trainers/OptimizationTrainer.h>

	FFNet<LogisticNeuron, LogisticNeuron> model;           // sigmoid transfer function for hidden and output neurons
	model.setStructure(N, M, 2);         // N inputs (depends on the data),
	                                     // M hidden neurons (depends on problem difficulty),
	                                     // and two output neurons (two classes).
	initRandomUniform(model, -0.1, 0.1); // initialize with small random weights
	CrossEntropy trainloss;              // differentiable loss for neural network training
	ErrorFunction<RealVector, unsigned int> error(&model, &trainloss, traindata);
	IRpropPlus optimizer;                // gradient-based optimization algorithm
	MaxIterations<> stop(iterations);    // stop optimization after fixed number of steps
	OptimizationTrainer<RealVector, RealVector, unsigned int> trainer(&error, &optimizer, &stop);

The important classes here are ErrorFunction and OptimizationTrainer.
An ErrorFunction allows us to build an objective function that can be
interfaced by arbitrary optimization strategies from a model, a loss,
and data. The argument of the ErrorFunction is the parameter vector of
the model, and its evaluation computes the empirical risk of the model
measured by the provided loss function on the given data. This allows
for the definition of a general optimization procedure with an iterative
optimizer and a stopping condition. The OptimizationTrainer is a simple
wrapper class that keeps references to the objective function (usually
an ErrorFunction), the optimizer and the stopping condition and
implements a straightforward iterative optimization loop in its train
method. Feel free to use other (differentiable) loss functions for
training, other (usually gradient-based) optimizers, and different
stopping criteria. All this can be done without changing the program
structure. In particular, after all definitions have been made there
will always be a model and trainer, and that's all we need to care for
in the end.


What you learned
================

You should have learned the following aspects in this tutorial:

* Shark is a versatile tool for machine learning. Changing the learning method requires only exchanging a few classes. All objects still conform to the same top level interfaces, such as AbstractModel and AbstractTrainer.
* Nearly everything in Shark is templated. It is not always easy to get all template parameters right in the first attempt. The probably best way of dealing with errors is to check the documentation of the template classes. The meaning of all template parameters should be documented. Often it will also become clear from the template parameter's name.

You may not have understood all details, in particular those hidden in
the various helper classes. If you are particularly interested in one
of the methods then please feel encouraged go ahead and explore the
documentation.

In any case you should have understood how all the different learning
methods are expressed by means of adaptive models and corresponding
trainers. Changing the learning method may involve changing the
particular sub-class, but all relevant objects will still conform to
the same top-level interfaces. Thus, only minimal changes to the
surrounding code will be necessary, if any at all. This design offers
a lot of flexibility, since changing the learning algorithm even late
in a project is usually not a big deal.
