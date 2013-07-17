

General Optimization Tasks
==========================


Introduction
------------


In the previous tutorial we employed the LDA trainer to solve the simple LDA
optimization task in one shot. In Shark, a trainer implements a solution
strategy for a standard problems, often providing an analytic solution.
In this sense, trainers are one-step optimizers. Given both data and a
model, a trainer finds the solution for its particular task. However,
many problems cannot be solved analytically, and in addition, it may not be
an easy decision when the optimization should stop. For this reason, Shark
provides a framework to set up task-specific optimization problems and solve them
iteratively. We have already seen one part of this general framework, the
model. A model is parameterized by a set of real values which define its
behavior. In many simple cases, a model represents a function, mapping inputs to
outputs, chosen from a parametric family (e.g., the family of linear functions).


Regression
----------

In this tutorial, we will focus on a simple regression task. In regression,
the parameters of a model are chosen such that it fits the training data as well
as possible, usually in a least-squares sense. After its parameters have been
found, the model can be used to predict the output values of new input points.
The Shark model also provides a way to calculate various derivatives. For more
information on Shark models, optimization, and trainers, see the concept
tutorial :doc:`../concepts/optimization/optimizationtrainer`. Other components
relevant to task-specific optimization tasks are described in the following. The
code for this example can be found in
:download:`regressionTutorial.cpp <../../../../../examples/Supervised/regressionTutorial.cpp>`.



Data preparation
%%%%%%%%%%%%%%%%


For this tutorial, we need the following include files::

   #include <shark/Data/Csv.h>
   #include <shark/Algorithms/GradientDescent/CG.h>
   #include <shark/ObjectiveFunctions/ErrorFunction.h>
   #include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
   #include <shark/Models/LinearModel.h>

We first write a short function which automates the data loading procedure.
In contrast to the last tutorial, this time we load a supervised learning data
set from two files. The first one stores input points, the other the corresponding
outputs. In the next step we bind the two loaded data items together to create
our ``RegressionDataset`` (which again is a simple typedef for ``LabeledData<RealVector, RealVector>``)::

  RegressionDataset loadData( const std::string& dataFile, const std::string& labelFile ) {
    Data<RealVector> inputs;
    Data<RealVector> labels;
    import_csv( inputs, dataFile, " " );
    import_csv( labels, labelFile, " " );
    RegressionDataset data( inputs, labels );
    return data;
  }


Now we can load the data using this function, and then create a training and test set
from it. We will again use 80% for training and 20% for testing. As in the previous
tutorial, we call  ``splitAtElement``, which splits the last part from our dataset
into the test set. Our original data set from then on only contains the training data::

  RegressionDataset data = loadData( "data/regressionInputs.csv", "data/regressionLabels.csv" );
  RegressionDataset test = splitAtElement( data, 0.8*data.numberOfElements() );



Setting up the model
%%%%%%%%%%%%%%%%%%%%



In this example, we want to do linear regression. So what we first need
is a linear model. Since we are not using a trainer which nicely sets up the model
for us, we have to configure it for the task. This mainly includes initializing
it with the proper input and output sizes. In our example data set, both
dimensions have size 1. But since this changes with every data set, we just ask
the dataset for the correct values: ::

   LinearModel<> model( inputDimension(data), labelDimension(data) );

The first argument is the dimensionality of the data points and the second parameter
that of the outputs. An affine linear model is a model of the form

.. math::
   f(x) = Ax+b

where :math:`x` is the input, and the matrix :math:`A` and vector
:math:`b` are constant. Thus, the parameters of the model are the
entries of :math:`A` and :math:`b`. Since we only have one output
dimension and one input dimension, this means that our model has two
parameters in total. In the general case, it has :math:`n*m+m`
parameters where :math:`n` is the input dimension and :math:`m` the
output dimension of the model.



Setting up the objective function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

In order to optimize the model defined above, we need to set up some criterion.
For this, we choose the squared loss of the model when applied to the training
data set. Note that the code below could with minor modifications also be used
to circumvent the slightly longer manual for-loop we wrote in the previous
tutorial to evaluate our function on the test set after training. ::

   SquaredLoss<> loss;
   ErrorFunction<RealVector, RealVector> errorFct( &model, &loss );
   errorFct.setDataset( data );

The ``ErrorFunction`` is an instance of ``AbstractObjectiveFunction`` -- see the
concept tutorial on :doc:`../concepts/library_design/objective_functions` for details.
The template parameters indicate that the training data and the labels are vectors
as is the usual case for regression. An objective function is a concept of a
function that evaluates a point in the search space of solutions and returns
its performance for the given problem. In the case of the ``ErrorFunction``,
the search space is formed by the parameters of the model. For evaluation, the
error function retrieves the performance of the given model with the specified
parameters on a data set which is specified using ``setDataset``. To assess the
performance, it applies a loss function to each data point. For linear regression,
we want to use the mean squared error, so we use the ``SquaredLoss`` class.
This loss function computes the squared difference

.. math ::
   L(f(x), y) = (f(x) - y)^2

between the model output :math:`f(x)` and the training label :math:`y`. The ErrorFunction itself
computes the mean

.. math ::
   \frac 1 n \sum_{i=1}^n L(f(x_i),y_i)

of the loss :math:`L` over the data set. There are many more objective functions available in
Shark, see the concept tutorial on :doc:`../concepts/library_design/objective_functions`.
In addition, also see the concept tutorial on :doc:`../concepts/optimization/optimizationtrainer`.



Optimization
%%%%%%%%%%%%


Linear regression can be solved analytically. This is done by the
trainer class ``LinearRegression`` and demonstrated in the tutorial on
:doc:`../algorithms/linearRegression`.  However, the purpose
of this tutorial is to introduce the general optimization framework
for learning, which applies to more complex losses and models, where
no analytic solution is available. So, let us optimize the model by a
general-purpose gradient-based method.


To optimize the above instantiated model under the above defined objective function
``ErrorFct``, we need an optimizer. For our regression task, a conjugate gradient
method is just fine. Also, training for 100 iterations should be more than sufficient,
even for more complex data::

   CG optimizer;
   optimizer.init( errorFct );
   for(int i = 0; i != 100; ++i)
   {
      optimizer.step( errorFct );
   }
   double trainingError = optimizer.solution().value;



Evaluation
%%%%%%%%%%


Again, we want to evaluate the model on a test set and print all results. We could
re-use ``errorFct`` for this by changing the dataset to the test set, but often
it is more convenient to use the loss directly. We let the model evaluate the whole
test set at once and ask the loss how big the error for this set of predictions is::

   model.setParameterVector( optimizer.solution().point );
   Data<RealVector> predictions = model( test.inputs() );
   double testError = loss( test.labels(), predictions );

or, the aforementioned alternative::

  errorFct.setDataset(test);
  double testError = errorFct.eval(optimizer.solution().point);

Let us see the results (do not forget to include the ``iostream`` header
for this and ``using namespace std;``) ::

   cout << "RESULTS: " << endl;
   cout << "======== \n" << endl;
   cout << "training error " << trainingError << endl;
   cout << "test error: " << testError << endl;

The result should read

.. code-block:: none

    RESULTS:
    ========

    training error: 0.0525739
    test error: 0.151367





What you learned
----------------


You should have learned the following aspects in this Tutorial:

* What the main building blocks of a general optimization task are: Data, Error Function, Model, Optimizer
* How to load regression data from two files and split them into training and test set.
* Different ways of error evaluation.



What next?
----------


Now you know the basic architecture of Shark. We will continue with one more introductory
tutorial on stopping criteria, which is most relevant for those working with task-specific
optimization problems. Afterwards, the "first steps" tutorials are completed, and all
other tutorials do not build on each other any longer.
