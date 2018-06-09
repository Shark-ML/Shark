Training Binary Restricted Boltzmann Machines
=============================================



Introduction
++++++++++++


Shark has a module for training restricted Boltzmann machines (RBMs) [Hinton2007]_
[Welling2007]_. All corresponding header files are located in the subdirectory
``<SHARK_SRC_DIR>/include/shark/Unsupervised/RBM/``. We will assume that you
already read the introduction to the RBM module :doc:`rbm_module`.

In the following, we will train and evaluate a Binary RBM using Contrastive Divergence
(CD-1) learning on a toy example. We choose this example as a starting point because
its setup is quite common, and we provide a set of predefined types for it for convenience.

The example file for this tutorial can be found in :doxy:`BinaryRBM.cpp`


Contrastive Divergence Learning -- Theory
+++++++++++++++++++++++++++++++++++++++++


.. todo: this tutorial is a stub. Add further information and formulas about CD-k



Contrastive Divergence Learning -- Code
+++++++++++++++++++++++++++++++++++++++


First, we need to include the following files ::


	//used for training the RBM
	#include <shark/Unsupervised/RBM/BinaryRBM.h>
	#include <shark/Algorithms/GradientDescent/SteepestDescent.h>
	
	//the problem
	#include <shark/Unsupervised/RBM/Problems/BarsAndStripes.h>
	
	//for evaluation
	#include <shark/Unsupervised/RBM/analytics.h>
	#include <iostream>
	

As an example problem, we consider one of the predefined benchmark problems in ``RBM/Problems/``,
namely, the Bars-and-Stripes data set [MacKay2002]_ ::


		BarsAndStripes problem;
		UnlabeledData<RealVector> data = problem.data();
		

Now we can create the RBM. We have to define how many input variables (visible units/observable
variables) our RBM shall have. This depends on the data set from which we want to learn, since
the number of visible neurons has to correspond to the dimensionality of the training data.
Further, we have to choose how many hidden neurons (latent variables) we want. Also, to construct
the RBM, we need to choose a random number generator. Since RBM training is time consuming, we
might later want to start several trials in separate instances. In this setup, being able to
choose a random number generator is crucial. But now, let's construct the beast::


		size_t numberOfHidden = 32;//hidden units of the rbm
		size_t numberOfVisible = problem.inputDimension();//visible units of the inputs
	
		//create rbm with simple binary units
		BinaryRBM rbm(random::globalRng);
		rbm.setStructure(numberOfVisible,numberOfHidden);
		

Using the RBM, we can now construct the k-step Contrastive Divergence error function. Since we
want to model Hinton's famous algorithm we will set k to 1. Throughout the library we use the
convention that all kinds of initialization of the structure must be set before calling `setData`.
This allows the gradients to adjust their internal structures. For CD-k this is not crucial, but
you should get used to it before trying more elaborate gradient approximators::


		BinaryCD cd(&rbm);
		cd.setK(1);
		cd.setData(data);
		

The RBM optimization problem is special in the sense that the error function can not be
evaluated exactly for more complex problems than trivial toy problems, and the gradient can
only be estimated. This is reflected by the fact that all RBM derivatives have the Flag
``HAS_VALUE`` deactivated. Thus, most optimizers will not be able to optimize it. One which
is capable of optimizing it is the ``GradientDescent`` algorithm, which we will use in the
following ::


		SteepestDescent<> optimizer;
		optimizer.setMomentum(0);
		optimizer.setLearningRate(0.1);
		

Since our problem is small, we can actually evaluate the negative log-likelihood. So we use
it at the end to evaluate our training success after training several trials ::


		unsigned int numIterations = 1000;//iterations for training
		unsigned int numTrials = 10;//number of trials for training
		double meanResult = 0;
		for(unsigned int trial = 0; trial != numTrials; ++trial) {
			initRandomUniform(rbm, -0.1,0.1);
			cd.init();
			optimizer.init(cd);
	
			for(unsigned int iteration = 0; iteration != numIterations; ++iteration) {
				optimizer.step(cd);
			}
			//evaluate exact likelihood after training. this is only possible for small problems!
			double likelihood = negativeLogLikelihood(rbm,data);
			std::cout<<trial<<" "<<likelihood<<std::endl;
			meanResult +=likelihood;
		}
		meanResult /= numTrials;
		

Now we can print the results as usual with ::


		cout << "RESULTS: " << std::endl;
		cout << "======== " << std::endl;
		cout << "mean negative log likelihood: " << meanResult << std::endl;
		

and the result will read something like

.. code-block:: none

    RESULTS:
    ========
    mean log likelihood: 192.544
    