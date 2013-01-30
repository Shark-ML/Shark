Deciding when to stop
=====================

.. todo::

    has this ever been tested and the corresp. code added as an example?
    there were some indications that it might not compile due to missing
    includes, so testing is required before removing this todo


Neural-network training example
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


In the previous tutorial, we set up a general optimization task which was
trained iteratively. This approach has two notable usage downsides compared
to the convenience of the one-step LDA-trainer in the first example:

#. We have to decide when the accuracy achieved by the optimization steps
   is high enough.

#. We need to write more code to set up all parts.

While the second point is just a nuisance, the first point is a "real"
structural problem. In the LDA example, we did not have to bother
whether the solution was sufficiently exact, as the LDA problem can be
solved analytically.



Motivation
++++++++++


In general, choosing a good number of iterations
for an iterative optimizer touches on two issues:

* First, simply a computational point of view: we do not want to perform
  more iterations than necessary to reach a "good" solution, but also not
  perform less than required to reach a "good" solution. While in general,
  the optimizers in Shark will stop when no more progress is being made,
  there may still be a useful earlier point to stop -- for example when
  only neglectable progress is being made.

  .. todo::

     is this correct? do all our optimizers stop when there is no more progress?

* Second, stopping early also constitutes a way of regularizing the
  adaptation of a model to a training set. Hence, stopping even earlier
  than would be indicated solely by the training dataset might be desired
  in machine learning usage anyways.

One means of early-stopping that goes beyong picking an arbitrary
number of iterations is monitoring the performance on a validation
split, which needs to be created from the dataset in addition to
training and test split.



Overview
++++++++


This tutorial will introduce different stopping criteria. For sake of example,
it also tackles an again slightly more complex learning task than those in
the previous two tutorials, namely classification with a simple feed-forward
neural network. We show how to create a trainer for this task which generalizes
important concepts and saves us manual work. Then, we construct and compare
three different stopping criteria for that trainer. To this end, we introduce
the ``AbstractStoppingCriterion``, another interface of Shark. In addition to
this "first-steps" tutorial, a concept tutorial on
:doc:`../concepts/library_design/stopping_criteria` exists, complementing
this one.


Building blocks & includes
++++++++++++++++++++++++++

We first list all includes for this tutorial and then motivate their
usage for each one::

   #include <shark/Data/Csv.h>
   #include <shark/Models/FFNet.h> //Feed forward neural network class
   #include <shark/Algorithms/GradientDescent/Rprop.h> //Optimization algorithm
   #include <shark/ObjectiveFunctions/Loss/CrossEntropy.h> //Loss used for training
   #include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> //The real loss for testing.
   #include <shark/ObjectiveFunctions/ErrorFunction.h> //The usual error function
   #include <shark/Algorithms/StoppingCriteria/MaxIterations.h> //A simple stopping criterion that stops after a fixed number of iterations
   #include <shark/Algorithms/StoppingCriteria/TrainingError.h> //Stops when the algorithm seems to converge
   #include <shark/Algorithms/StoppingCriteria/GeneralizationQuotient.h> //Uses the validation error to track the progress
   #include <shark/Algorithms/StoppingCriteria/ValidatedStoppingCriterion.h> //Adds the validation error to the value of the point

As before, ``Csv.h`` is included for data read-in. ``FFNet.h`` is needed
because we want to train a neural network to distinguish between two classes.
``Rprop`` is a fast and stable algorithm for gradient-based optimization of
a differentiable objective function. Since the 0-1-loss is not differentiable,
and would thus not be compatible with any gradient descent method including
Rprop, we instead use the ``CrossEntropy`` as surrogate loss. But for testing,
we still want to use and hence include the ``ZeroOneLoss``. As in the last
tutorial, the ``ErrorFunction`` evaluates a model under a certain loss and
on a certain data set when provided with the model parameters for a current
such evaluation. The remaining includes are needed for the different stopping
criteria we will examine.



Using an AbstractStoppingCriterion
++++++++++++++++++++++++++++++++++

We want to use a feed-forward neural network with one hidden layer and 2 output
neurons for classification, and train it under three different stopping criteria:
a fixed number of iterations, progress on the training error, and progress on a
validation set. To facilitate our experiments, we create one single, local, auxiliary
function that takes an ``AbstractStoppingCriterion`` -- the base class of all
stopping criteria -- as an argument and creates as well
as trains such a neural network under that abstract stopping criterion. In
addition, instead of manually and explicitly coding an optimization loop as in
the previous examples, we use a so-called ``OptimizationTrainer`` that encapsulates
the entire training process given an ObjectiveFunction, Optimizer, and StoppingCriterion.
Overall, we use the following function to create, train and evaluate our neural
network under a given stopping criterion::


   template<class T>
   double experiment(AbstractStoppingCriterion<T> & stoppingCriterion, ClassificationDataset const& trainingset, ClassificationDataset const& testset){
    //create a feed forward neural network with one layer of 10 hidden neurons and one output for every class
    FFNet<LogisticNeuron,LinearNeuron> network;
    network.setStructure(inputDimension(trainingset),10,numberOfClasses(trainingset));
    initRandomUniform(network,-0.1,0.1);

    //define loss and error function
    CrossEntropy loss;
    ErrorFunction<RealVector,unsigned int> errorFunction(&network,&loss);

    //we use IRpropPlus for network optimization
    IRpropPlus optimizer;

    //create an optimization trainer and train the model
    OptimizationTrainer<FFNet<LogisticNeuron,LinearNeuron>,unsigned int > trainer(&errorFunction, &optimizer, &stoppingCriterion);
    trainer.train(network, trainingset);

    // Evaluate the performance on the test set using the classification loss. We set the threshold to 0.5 for Logistic neurons.
    ZeroOneLoss<unsigned int, RealVector> loss01(0.5);
    Data<RealVector> predictions = network(testset.inputs());
    return loss01(testset.labels(),predictions);
   }



Evaluation
++++++++++


Now it is time to load some data and try out different stopping criteria.


Fixed number of iterations
&&&&&&&&&&&&&&&&&&&&&&&&&&


The simplest stopping heuristic is halting after a fixed number of iterations.
``MaxIterations`` then is the subclass of choice, which simply provides this
trivial functionality for within the framework of an AbstractStoppingCriterion.
We try out several different numbers of steps::

   ///load the dataset and split into training, validation and test set.
   ClassificationDataset data;
   import_csv(data, "data/diabetes.csv",LAST_COLUMN, ",");
   data.shuffle();
   ClassificationDataset test = splitAfterElement( data, static_cast<std::size_t>( 0.75*data.numberOfElements() ) );
   ClassificationDataset validation = splitAfterElement( data, static_cast<std::size_t>( 0.66*data.numberOfElements() ) );

   MaxIterations<> maxIterations(10);
   double resultMaxIterations1 = experiment( maxIterations, data,test );
   maxIterations.setMaxIterations(100);
   double resultMaxIterations2 = experiment( maxIterations, data,test );
   maxIterations.setMaxIterations(500);
   double resultMaxIterations3 = experiment( maxIterations, data,test );



Progress on training error
&&&&&&&&&&&&&&&&&&&&&&&&&&

Next we employ a stopping criterion that monitors progress on the
training error. The stopping criterion ``TrainingError`` takes in its
constructor a window size (or number of time steps) :math:`T`  together
with a threshold value :math:`\epsilon`. If the improvement over the
last :math:`T` timesteps does not exceed :math:`\epsilon`, that is,
:math:`E(t-T)-E(t) < \epsilon`, the stopping criterion becomes active
and tells the optimizer to stop (because it assumes that progress over
subsequent optimization steps will be negligible as well). Note that a
danger when using this stopping criterion is that it may stop optimization
even when the algorithm only traverses a locally isolated plateau or saddle
point. However, the optimizer used here, ``IRpropPlus``, dynamically
adapts it step size and and hence is somewhat less vulnerable to these
problems. After all the groundwork has been done, we can test this
stopping criterion with only two lines of code::

  TrainingError<> trainingError( 10, 1.e-5 );
  double resultTrainingError = experiment( trainingError, data, test );



Progress on a validation set
&&&&&&&&&&&&&&&&&&&&&&&&&&&&


To use validation error information, we need to define an additional validation error
function. In the simplest case, this is just an error function using the same objects
as that on the training set, but a different dataset. For simplicity of the tutorial,
we will instead just create it from scratch. The class that takes the current point
of the search space from the optimizer and passes it on the the evaluation error function
is the so-called ``ValidatedStoppingCriterion``. It constructor takes as argument not
only the validation error function, but also another stopping criterion, to which the
result of the validation run is passed and which is prepared to make its decision based
on both training and validation information. In this example, we will use the
``GeneralizationQuotient`` as such a stopping criterion. In detail, it calculates the
ratio of two other criteria to reach its decision, and hence we refer to the class
documentation for an exact description, as well as the scientific publication
mentioned therein.

.. todo::

    the class documentations for most stopping criteria need serious cleanup,
    and also a thourough check if they indeed implement their counterparts from
    the Prechelt paper correctly (i have some serious doubts about the validation-based
    criteria!)! If there are bugs in the code, this tutorial should be re-run and the
    results code and description updated.

In summary, this code uses the progress on a validation set to decide when to stop::

   //create the validation error function
   FFNet<LogisticNeuron,LogisticNeuron> network;
   network.setStructure(inputDimension(data),10,numberOfClasses(data));
   CrossEntropy loss;
   ErrorFunction<RealVector,unsigned int> validationFunction(&network,&loss);
   validationFunction.setDataset(validation);

   //create the generalization quotient and use the vValdiatedStoppingCriterion to add validation information using the validation function
   GeneralizationQuotient<> generalizationQuotient(10,0.1);
   ValidatedStoppingCriterion validatedLoss(&validationFunction,&generalizationQuotient);
   double resultGeneralizationQuotient = experiment(validatedLoss,data,test);



Printing the results
++++++++++++++++++++

Printing all variables of type ``double`` defined in the snippets above, we get

.. code-block:: none

   RESULTS:
   ========

   10 iterations   : 0.5
   100 iterations : 0.375
   500 iterations : 0.40625
   training Error : 0.442708
   generalization Quotient : 0.416667


So stopping after around 100 iterations yielded the lowest error on the test
set. The TrainingError criterion will, as predicted, wait a lot longer. The
GeneralizationQuotient does in fact stop too early in this case, which is very
likely due to the small size of the data set used in the example code.



What you learned
++++++++++++++++


You should have learned the following aspects in this Tutorial:

* How to train a feed forward neural network
* How to create a trainer from a general optimization task
* That the choice of stopping criterion matters.



What next?
++++++++++


Now you should be ready to leave the "first steps" section of the tutorials
and read through its other sections, which will tell you about various
aspects of the library in more detail.
