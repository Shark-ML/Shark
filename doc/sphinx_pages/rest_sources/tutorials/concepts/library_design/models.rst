

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

A simple model is a linear model, which can for example map vectorial
input to a lower dimensional subspace:

.. math::
  f(x) = Ax+b

In this case, we can say that all entries of the matrix *A* and of the
vector *b* form the parameters of the model *f*. 
Optimizing parameters often requires derivatives
which requires the model to be differentiable
with respect to its own parameters.


List of Classes
---------------------------------
The list of models is available in the :doxy:`class documentation <models>`


The base class 'AbstractModel'
------------------------------


The base class for models in Shark is the templated class
``AbstractModel<InputTypeT,OutputTypeT>``. For an in-depth description
of its methods, check the doxygen documentation of
:doxy:`shark::AbstractModel`.  Here, we describe how the concepts
introduced above are represented by the interface, and how models can
be used in Shark.

In general, most routines are optimized for batch computation (see the
tutorial on :doc:`batches`), that is, for processing many
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
it is called, an exception is thrown. Flags can be queried via
convenience functions summarized in the table below:

=======================================================================   ========================================================
Flag and accessor function name                                           Description
=======================================================================   ========================================================
``HAS_FIRST_PARAMETER_DERIVATIVE``, ``hasFirstParameterDerivative()``     First derivative w.r.t. the parameters is available
``HAS_FIRST_INPUT_DERIVATIVE``, ``hasFirstInputDerivative()``             First derivative w.r.t. the inputs is available
=======================================================================   ========================================================

To evaluate a model, there exist several variants of ``eval`` and
``operator()``. The most notable exception is the stateful valuated version of ``eval``. 
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
first derivatives, then the following methods also need to
be implemented:



===============================  ==============================================================================
Method                           Description
===============================  ==============================================================================
``weightedParameterDerivative``  Computes first or second drivative w.r.t the parameters for every output value
                                 and input and weights these results together
``weightedInputDerivative``      Computes first or second drivative w.r.t the inputs for every output value
                                 and input and weights these results together
``weightedDerivatives``          Computes first input and parameter derivative at the same time, making it
                                 possible to share calculations of both derivatives. Can be omitted.
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
  modl.weightedParameterDerivative(inputs, outputs, weights,*state,derivative);


There are a few more methods which result from the fact that AbstractModel
implements several higher-level interfaces, namely :doxy:`IParameterizable`,
:doxy:`INameable`, and :doxy:`ISerializable`. For
example, models are parameterizable and serialized to store results:


======================   ==============================================================================
Method                   Description
======================   ==============================================================================
``numberOfParameters``   Number of parameters which can be optimized
``parameterVector``      Returns the current parameter vector of the model
``setParameterVector``   Sets the parameter vector to new values
``inputShape``		 Defines the shape that the model expects as input
``outputShape``		 Defines the shape the model will output
``read``, ``write``      Loads and saves a serializable object
``createState``          Returns a newly created State object holding the state to be stored in eval
======================   ==============================================================================
