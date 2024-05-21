===============================================
Kernelized Budgeted Stochastic Gradient Descent
===============================================


Support vector machines and other kernel-based learning algorithms 
are widely used and have many benefits. They can be considered as
state-of-the-art algorithms in machine learning. Despite being easy
to use, for larger data sets the kernelization, which was central to
the development of SVM, becomes a bottleneck, as the computation
time grows quadratically in the number of support vector-- but the
latter have been shown to grow linearly in the dataset size.
Therefore the whole training process becomes quadratically,
and impractical for even remotely large datasets.
This problem was called the curse of kernelization in [WangCrammerVucetic2012]_.

There are different ways to solve this problem.
One intuitive method was presented in 
[WangCrammerVucetic2012]. The idea is to put a constraint
on the complexity of the model, i.e. the sparsity of the weight vector.
As the weight vector in features space is a sum of basis functions,
this means that it has to have the form :math:`w = \sum_{i=1}^B k(x_i, \cdot)`,
where B is the chosen budget size and :math:`x_i` are some
data points.

[WangCrammerVucetic2012] employ a well-known 
stochastic gradient descent method, Pegasos, to train the model in a 
perceptron-like fashion:
In each round the algorithms is given a data point.
If it violates the margin with respect to the
current model (so the example is either 
classified incorrectly or with a too low confidence), it will be added
to the weight vector, also called budget, just like in Pegasos. 

Obviously, at some point the budget becomes full. 
In this case, adding a new vector will violate the  size-constraint.
Therefore one needs a way to reduce the size of the budget.
These, often heuristic, methods are called budget maintenence strategies.
Many such strategies exist. One of the easiest is just to remove 
randomly a vector from the budget. Another strategy is remove the
'oldest' vector (this method is called Forgetron).  
Both strategies maintain the budget size, but are not optimal
in a certain sense, as they do not really try to minimize the
degradation of the model that occurs when one removes a 
support vector. A better way was proposed in [WangCrammerVucetic2012]:
The idea is to find a pair of vectors that, when merged into one vector,
does degrade the quality of the solution as low as possible. 
This can be formulated as
an optimization problem, and it can be shown that with a heuristic
search for such a pair, training is much better than with a
random maintenence strategy.

In Shark both strategies,  the remove and the merge strategy, can be applied.
Tthis tutorial shows how to use the Kernelized Budgeted SGD Trainer
in Shark with the merge strategy.



KernelBudgetedSGD in  Shark
--------------------------------

For this tutorial the following include files are needed::


	#include <shark/Algorithms/Trainers/Budgeted/KernelBudgetedSGDTrainer.h> // the KernelBudgetedSGD trainer
	#include <shark/Algorithms/Trainers/Budgeted/MergeBudgetMaintenanceStrategy.h> // the strategy the trainer will use 
	#include <shark/Data/DataDistribution.h> //includes small toy distributions
	#include <shark/Models/Kernels/GaussianRbfKernel.h> //the used kernel for the SVM
	#include <shark/ObjectiveFunctions/Loss/HingeLoss.h> // the loss we want to use for the SGD machine
	#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> //used for evaluation of the classifier
	

Toy problem
^^^^^^^^^^^

In this tutorial, we consider the chessboard problem, which is a well-known
artificial binary benchmark classification problem::


		unsigned int ell = 500;     // number of training data point
		unsigned int tests = 10000; // number of test data points
		
		Chessboard problem; // artificial benchmark data
		ClassificationDataset trainingData = problem.generateDataset(ell);
		ClassificationDataset testData = problem.generateDataset(tests);
		



Model and learning algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The steps to use the KernelBudgetedSGD trainer are the very same
one uses to build a CSvmTrainer :doxy:`CSvmTrainer`. Thus,
to build our trainer, we need a :doxy:`KernelClassifier`  and an
:doxy:`KernelBudgetedSGDTrainer`. 

Our model is given by the two components: A 
standard Gaussian/RBF kernel, which we create as usual::


		double gamma = 0.5;         // kernel bandwidth parameter
		
		GaussianRbfKernel<> kernel(gamma); // Gaussian kernel
		

and a kernel classifier::


		KernelClassifier<RealVector> kernelClassifier; // (affine) linear function in kernel-induced feature space
		

Then, training the machine is simply performed by calling::


		double C = 1.0;          // regularization parameter
		bool bias = false;           // use bias/offset parameter
		size_t budgetSize = 16;     // our model shall contain at most 16 vectors
		size_t epochs = 5;      // we want to run 5 epochs
		
		HingeLoss hingeLoss; // define the loss we want to use while training
		// as the budget maintenance strategy we choose the merge strategy
		MergeBudgetMaintenanceStrategy<RealVector> *strategy = new MergeBudgetMaintenanceStrategy<RealVector>();
		KernelBudgetedSGDTrainer<RealVector> kernelBudgetedSGDtrainer(&kernel, &hingeLoss, C, bias, false, budgetSize, strategy);        // create the trainer
		kernelBudgetedSGDtrainer.setEpochs(epochs);      // set the epochs number
		

As in the :doxy:`CSvmTrainer`, the parameter  ``C`` denotes the 
regularization parameter (the SVM uses the 1-norm
penalty for target margin violations by default) and `bias` the inclusion of a bias term in the solver..



Evaluating the model
^^^^^^^^^^^^^^^^^^^^

To evaluate the model, we simply create a test dataset by generating
another chessboard problem. We can evaluate our trained model 
on the test data set as well as the train dataset (the latter one just to
get a feeling how good the training went and to see overfitting problems).
We consider the standard 0-1 loss as performance measure. The code
then reads::


		ZeroOneLoss<unsigned int> loss; // 0-1 loss
		Data<unsigned int> output = kernelClassifier(trainingData.inputs()); // evaluate on training set
		double train_error = loss.eval(trainingData.labels(), output);
		cout << "training error:\t" <<  train_error << endl;
		output = kernelClassifier(testData.inputs()); // evaluate on test set
		double test_error = loss.eval(testData.labels(), output);
		cout << "test error:\t" << test_error << endl;
		


Full example program
--------------------

The full example program considered in this tutorial is :doxy:`KernelBudgetedSGDTutorial.cpp`.

References
----------

.. [WangCrammerVucetic2012] Z. Wang, K. Crammer and S. Vucetic: Breaking the curse of kernelization: Budgeted stochastic gradient descent for large-scale SVM training. The Journal of Machine Learning Research 13.1 (2012): 3103-3131.
