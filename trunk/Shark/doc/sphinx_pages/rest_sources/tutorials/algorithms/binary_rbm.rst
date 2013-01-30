Training Binary Restricted Boltzmann Machines
==================================================================
Shark has a module for training restricted Boltzmann machines
(RBMs) [Hinton2007]_ [Welling2007]_. You can find all files in the subdirectory ``shark/Unsupervised/RBM/``. We will assume that
you allready read the introduction to the RBM module :doc:`rbm_module`.
In the following, we will train  and evaluate a Binary RBM using Contrastive Divergence (CD-1) learning on a toy example.
We choose this example as a starting point of the Module because this setup is quite common and we provide a set of predefined
types which make this as easy as possible!

.. todo: this tutorial is a stub. Add further information and formulas about CD-k

First, we need to include the following files ::

  //used for training the RBM
  #include <shark/Unsupervised/RBM/BinaryRBM.h>
  #include <shark/Algorithms/GradientDescent/SteepestDescent.h>
  #include <shark/Rng/GlobalRng.h>

  //the problem
  #include <shark/Unsupervised/RBM/Problems/BarsAndStripes.h>

  //for evaluation
  #include <shark/Unsupervised/RBM/analytics.h>
  #include <iostream>

As an example problem, we consider one of the  predefined benchmark problems in ``RBM/Problems/``,
the Bars and Stripes[MacKay2002]_ ::

  BarsAndStripes problem;
  Set<RealVector> data = problem.data();

Now we can create the RBM. We have to define how many inputs (visible
units/observable variables) our RBM shall have. This information is given by the problem .
Further, we have to choose how many hidden neurons (latent variables) we want. Also to construct the RBM we need to choose
a random number generator. Since RBM training is time consuming we might later want to start several trials in separate
instances. In this setup, being able to choose a random number generator is crucial. But now, let's construct the beast::

  size_t numberOfHidden = 32;
  size_t numberOfVisible = problem.inputDimension();

  //create rbm with simple binary units and use the global random number generator of shark for simplicity
  BinaryRBM rbm(Rng::globalRng);
  rbm.setStructure(numberOfVisible,numberOfHidden);

Using the RBM, now we can construct the k-step Contrastive Divergence error
function. Since we want to model Hintons famous algorithm we will set k to 1. Throughout the Library we use the
convention that all kinds of initialization of the structure must be set before calling `setData`. This allows the gradients
to adjust their internal structures. For CD-k this is not crucial, but you should get used to it before trying more elaborate
gradient approximators::

  BinaryCD cd(&rbm);
  cd.setK(1); // Gibbs sampling steps: k steps = cd-k
  cd.setData(data);

The RBM optimization problem is very special in the sense that the
error function can not be evaluated exactly for more complex problems
than trivial toy problems and the gradient can only be estimated. This
is reflected by the fact that all RBM derivatives have the Flag ``HAS_VALUE`` deactivated. So
most optimizers won't be able to optimize it. One which is capable of
optimizing it is the ``GradientDescent`` algorithm, which we will use
in the following ::

  SteepestDescent optimizer;
  optimizer.setMomentum(0);
  optimizer.setLearningRate(0.1);

Since our Problem is small, we actually can evaluate the negative log-likelihood. So we use it at the end to
evaluate our training success after training several trials ::

  double meanResult = 0;
  for(unsigned int trial = 0; trial != numTrials; ++trial) {
    initializeWeights(rbm);//some routine to initialize the weights. e.g. randomly in (-0.1,0.1)
    optimizer.init(cd);//init must be inside the trial loop to remove old results

    //train
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

the example file for this tutorial can be found in :doxy:`BinaryRBM.cpp`




