

Models
======


Models in Shark can be seen as an abstract concept of a function,
transforming an input into an output (or: producing an input given an
output).  In a machine learning context, models often correspond to
hypotheses. Models represent the solutions to machine learning
problems. For example, in classification we want to learn a model
assigning classes to input points. The models are often parameterized,
and then the process of learning corresponds to optimizing model
parameters. After learning, the model with the optimized parameters
represents the solution.

However, models can be more general than parameterized families of
functions, since they may have an internal state. In models with a
non-trivial state, the computation of the output depends on the input
and the state, and the state may change based on the input. Stateful
models are attractive for processing sequence information, in contrast
to independent data instances. Refer to the :doxy:`RNNet` class
implementing recurrent multi-layer perceptrons (which are rather
functionals than functions) for an example.

But back to simpler models for now. A simple model is
the threshold classifier, which receives a real value as input. If the
value is higher than the internal threshold (the model parameter),
then the model assigns a class label of 1, and of 0 otherwise. A
second example is a linear model, which can for example map vectorial
input to a lower dimensional subspace:

.. math::
  f(x) = Ax+b

In this case, we can say that all entries of the matrix *A* and of the
vector *b* form the parameters of the model *f*. Clearly, the linear
model has more parameters than the threshold converter.

The way a model's parameters should be optimized of course depends on
the criterion, or objective function, according to which the model
should be tuned. Whether or not the optimal parameters can be found
analytically will thus depend on both the model as well as of the
objective function. Many algorithms in Shark are gradient-based
optimization methods, which require the model to be differentiable
with respect to its own parameters.



The base class 'AbstractModel'
------------------------------


The base class for models in Shark is the templated class
``AbstractModel<InputTypeT,OutputTypeT>``. For an in-depth description
of its methods, check the doxygen documentation of
:doxy:`shark::AbstractModel`.  Here, we describe how the concepts
introduced above are represented by the interface, and how models can
be used in Shark.

In general, most routines are optimized for batch computation (see the
tutorial on :doc:`batches`), that is, for precessing many
elements at one time. For example, models support to be evaluated on a
batch of inputs and to compute their weighted derivatives for a batch
of inputs at once (also see
:doc:`../optimization/conventions_derivatives`).

The AbstractModel class is templatized on the input type as well as
the output type. For a classification model, the input type is likely
to be a vector type like ``RealVector``, and the output type to be an
``unsigned int`` for a class label.  From these types, the model
infers the rest of the types needed for the interface and made public by
the model:



===================   =========================================================
Types                 Description
===================   =========================================================
``InputType``         Shortcut for the input type
``OutputType``        Shortcut for the output type
``BatchInputType``    A Batch of inputs as returned by Batch<InputType>::type
``BatchOutputType``   A Batch of outputs as returned by Batch<OutputType>::type
===================   =========================================================



The basic capabilities of a model are managed through a set of flags. If a model
can for example calculate the first input derivative, the flag
``HAS_FIRST_INPUT_DERIVATIVE`` is set. If the flag is not set and a function relying on
it is called, an exception is thrown. Flags can be queried using the somewhat
lengthy expression
``model.features().flag()&AbstractModel<InputTypeT,OutputTypeT>::FLAG`` or via
convenience functions summarized in the table below:



=======================================================================   ========================================================
Flag and accessor function name                                           Description
=======================================================================   ========================================================
``HAS_FIRST_PARAMETER_DERIVATIVE``, ``hasFirstParameterDerivative()``     First derivative w.r.t. the parameters is available
``HAS_SECOND_PARAMETER_DERIVATIVE``, ``hasSecondParameterDerivative()``   Second derivative w.r.t. the parameters is available
``HAS_FIRST_INPUT_DERIVATIVE``, ``hasFirstInputDerivative()``             First derivative w.r.t. the inputs is available
``HAS_SECOND_INPUT_DERIVATIVE``, ``hasSecondInputDerivative()``           Second derivative w.r.t. the inputs is available
``IS_SEQUENTIAL``, ``isSequential()``                                     Model is sequential (see below)
=======================================================================   ========================================================



A sequential model can only process a single input at a time and will throw an
exception if multiple inputs are fed in. For these models, the next output depends
on the sequence of previous inputs and thus a batch computation does not make sense.


.. caution::

  Support for the second derivatives is purely experimental and not well
  supported throughout Shark. Changes of the interface are likely.



To evaluate a model, there exist several variants of ``eval`` and
``operator()``. The most notable exception is the statefull valuated version of ``eval``. 
The state allows the model to store computation results during ``eval`` which then can be reused
in the computation of the derivative to save computation time. 
In general, if the state is not required, it is a matter of taste which functions
are called. We recommend using ``operator()`` for convenience.
The list of evaluation functions is:



====================================================================   ===============================================================================
Method                                                                 Description
====================================================================   ===============================================================================
``eval(InputType const&,OutputType&)``                                 Evaluates the model's response to a single input and stores it in the output
``eval(BatchInputType const&, BatchOutputType&)``                      Evaluates the model's response to a batch of inputs and stores them, in
								       corresponding order, in the output batch type
``eval(BatchInputType const&, BatchOutputType&, State& state)``        Same as the batch version of eval, but also stores intermediate results which
                                                                       can be reused in computing the derivative
``OutputType operator()(InputType)``                                   Calls eval(InputType, OutputType) and returns the result
``BatchOutputType operator()(BatchInputType)``                         Calls eval(BatchInputType, BatchOutputType) and returns the result
``Data<OutputType> operator()(Data<InputType>)``                       Evaluates the model's response for a whole dataset and returns the result
====================================================================   ===============================================================================



The only method required to be implemented in a model is the stateful
batch input version of eval. All other evaluation methods are inferred
from this routine. It can also make sense to implement the
single-input version of eval, because the default implementation would
otherwise copy the input into a batch of size 1 and then call the
batch variant. However, the single-input variant will usually not be
called when performance is important, so not implementing it should
not have critical drawbacks from the point of view of the standard
Shark code base. If a model indicates by its flags that it offers
first or second derivatives, then the following methods also need to
be implemented (which are overloaded once for the first derivative,
and once for the first and second derivatives at the same time):



===============================  ==============================================================================
Method                           Description
===============================  ==============================================================================
``weightedParameterDerivative``  Computes first or second drivative w.r.t the parameters for every output value
                                 and input and weights these results together
``weightedInputDerivative``      Computes first or second drivative w.r.t the inputs for every output value
                                 and input and weights these results together
``weightedDerivatives``          Computes first input and parameter derivative at the same time, making it
                                 possible to share calculations of both derivatives
===============================  ==============================================================================

The parameter list of these methods is somewhat lengthy, and thus we
recommend looking up their exact signature in the doxygen
documentation. However, all versions require the state computed during
eval. Example code to evaluate the first derivative of a model with
respect to its parameters thus looks like this::

  BatchInputType inputs; //batch of inputs
  BatchOutputType outputs; //batch of model evaluations
  MyModel model;  //the differentiable model

  // evaluate the model for the inputs and store the intermediate values in the state
  boost::shared_ptr<State> state = model.createState();
  model.eval(inputs,outputs,*state);

  // somehow compute some weights and calculate the parameter derivative
  RealMatrix weights = someFunction(inputs,outputs);
  RealVector derivative;
  modl.weightedParameterDerivative(inputs,weights,*state,derivative);


There are a few more methods which result from the fact that AbstractModel
implements several higher-level interfaces, namely :doxy:`IParameterizable`,
:doxy:`IConfigurable`, :doxy:`INameable`, and :doxy:`ISerializable`. For
example, models are parameterizable, can be configured from a file and
serialized to store results:


======================   ==============================================================================
Method                   Description
======================   ==============================================================================
``numberOfParameters``   Number of parameters which can be optimized
``parameterVector``      Returns the current parameter vector of the model
``setParameterVector``   Sets the parameter vector to new values
``configure``            Configures the model. Options depend on the specific model
``read``, ``write``      Loads and saves a serializable object
``createState``          Returns a newly created State object holding the state to be stored in eval
======================   ==============================================================================





List of Models
--------------


We end this tutorial with a list of some  models currently implemented in Shark,
together with a brief description.


We start with general purpose models:


========================   ==================================================================================
Model                      Description
========================   ==================================================================================
:doxy:`LinearModel`        A simple linear model mapping an n-dimensional input to an m-dimensional output
:doxy:`FFNet`              The well-known feed-forward multilayer perceptron
                           It allows the usage of different types of neurons in the hidden and output layers
:doxy:`RBFNet`             Implements a radial basis function network using gaussian distributions
                           The output is a possibly multidimensional linear combination of inputs
:doxy:`CMACMap`            Discretizes the space using several randomized tile maps and calculates a
                           weighted sum of the discretized activation
:doxy:`RNNet`              Recurrent neural network for sequences
:doxy:`OnlineRNNet`        Recurrent neural network for online learning
:doxy:`KernelExpansion`    linear combination of outputs of :doxy:`AbstractKernelFunction <Kernel>`, given
                           points of a dataset and the point to be evaluated (input point)
========================   ==================================================================================



Some models for Classification or Regression:



=====================================    ========================================================================
Model                                    Description
=====================================    ========================================================================
:doxy:`LinearClassifier`                 Given a metric represented by a scatter matrix and the class means,
                                         assigns a new point to the class with the nearest mean
:doxy:`NBClassifier`                     Standard, but flexible, naive Bayes classifier
:doxy:`OneVersusOneClassifier`           Multi-class classifier which does majority voting using binary
                                         classifiers for every class combination
:doxy:`NearestNeighborClassifier`        Nearest neighbor search for classification using a majority vote system.
:doxy:`NearestNeighborRegression`        Nearest neighbor search for regression; the result is the mean of the
                                         labels of the k nearest neighbors
:doxy:`SoftNearestNeighborClassifier`    Nearest neighbor search for classification; returns the fraction
                                         of votes for a class instead of the majority vote
:doxy:`CARTClassifier`                   Classification and regression tree
:doxy:`RFClassifier`                     Random Forest based on a collection of CART classifiers
=====================================    ========================================================================




Models for Clustering:



========================================== =====================================================================================
Model                                      Description
========================================== =====================================================================================
:doxy:`ClusteringModel`                    Base class for all clustering models, requires an :doxy:`AbstractClustering` to work.
:doxy:`SoftClusteringModel`                Returns for a given point :math:`x` a vector of propabilities :math:`p(c_i|x)`
                                           indicating the propability of the point to be in the cluster :math:`c_i`
:doxy:`HardClusteringModel`                Returns the index of the cluster with highest probability for a given point,
                                           :math:`\arg \max_i p(c_i|x)`.
========================================== =====================================================================================



Special purpose models:



======================================  ======================================================================
Model                                   Description
======================================  ======================================================================
:doxy:`MissingFeaturesKernelExpansion`  KernelExpansion with support for missing input values.
:doxy:`ConcatenatedModel`               Chains two models together by using the output of one model as the
                                        input to the second. It is even possible to calculate the derivative
                                        of such a combination if all models implement it.
:doxy:`LinearNorm`                      For positive inputs, normalize them to unit L_1-norm
:doxy:`Softmax`                         Standard softmax activation/weighting function.
:doxy:`SigmoidModel`                    Maps a real valued input to the unit interval via a sigmoid function.
:doxy:`ThresholdConverter`              If the input is higher than a threshold, assign 1, otherwise 0.
:doxy:`ThresholdVectorConverter`        For every value of the input vector apply a ThresholdConverter.
:doxy:`ArgMaxConverter`                 Assigns the index (e.g., a class label) of the largest component in
                                        the input vector.
======================================  ======================================================================



