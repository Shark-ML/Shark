

Objective Functions
===================



An objective function represents an optimization problem for which a (somehow)
optimal solution is to be found, for example using :doc:`optimizers`. Given an
input point within its admissible search space, an objective function returns
a (possibly noisy) objective value (and, optionally, its derivative). The input
is typically a vector of real numbers, but the interface allows for more general
cases as well. For single objective optimization the return value is a real number
and the goal is to minimize it. For multi-objective optimization, the output is
a vector as well, holding the objective function value for each of the multiple
goals.

Besides returning a corresponding value for a search point, objective functions
in Shark also manage their own search space. In detail, they provide a method
indicating if a given point is feasible, and possibly also a method which can
provide the feasible point closest to an infeasible one.



The base class 'AbstractObjectiveFunction <SearchSpaceType, ResultT>'
---------------------------------------------------------------------


Template arguments and public types
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

The base class :doxy:`AbstractObjectiveFunction` has two template arguments.
The first is the SearchSpaceType, which defines the admissible type of inputs.
Currently only one type of search space is implemented and used in Shark, the
typical :doxy:`VectorSpace`. ``ResultT`` defines the return type of the objective
function, which for single-objective functions is ``double``, and ``RealVector``
for multi-objective functions. Thus, a typical single objective function has as
type signature ``AbstractObjectiveFunction< VectorSpace<double>, double >`` and
a multi objective function ``AbstractObjectiveFunction< VectorSpace<double>, RealVector >``.
Based on the search space type, the following types are inferred and made public
as typedefs:


=====================  ================================================================
Types                  Description
=====================  ================================================================
SearchPointType        Type of an input point
FirstOrderDerivative   Type of the derivative with respect to the point
SecondOrderDerivative  Type of the object storing the first and second order derivative
=====================  ================================================================


Flags
&&&&&

Objective functions have a set of flags which indicate their capabilities
and constraints. Accessor functions ease querying these flags:


===============================================================  ==========================================================
Flag, accessor function                                          Description
===============================================================  ==========================================================
``HAS_VALUE``, ``hasValue``                                      The objective function can calculate its own function
                                                                 value. Since this attribute is common to most objective
                                                                 functions, this flag is set by default. See comment below.
``HAS_FIRST_DERIVATIVE``, ``hasFirstDerivative``                 The first derivative can be computed.
``HAS_SECOND_DERIVATIVE``, ``hasSecondDerivative``               The second derivative can be computed.
``IS_CONSTRAINED_FEATURE``, ``isConstrained``                    The input space is constrained. To query whether a point
                                                                 is feasible, the function must then offer a method
                                                                 ``isFeasible``.
``CAN_PROPOSE_STARTING_POINT``, ``canProposeStartingPoint``      The objective function can propose a feasible starting
                                                                 point from which the optimizer can start the optimization.
``CAN_PROVIDE_CLOSEST_FEASIBLE``, ``canProvideClosestFeasible``  A constrained function can provide a function
                                                                 ``closestFeasible`` which returns the closest feasible
                                                                 point given an infeasible one.
``IS_THREAD_SAFE``, ``isThreadSafe``                             This flag indicates that eval and evalDerivative can be
                                                                 called in parallel by the optimizer for different points.
===============================================================  ==========================================================


It might seem strange at first, that we have a flag ``HAS_VALUE``. However, there
exist problems where calculating the function value is hard or infeasible, while
calculating the gradient is still possible. For some simple optimizers, the gradient
information is enough to find a better point. If the flag is not set, calling
``eval`` is disallowed, and the other functions return meaningless values like
``qnan``.


Interface
&&&&&&&&&


Using an objective function is easy, as can be seen in the following
short list of functions:


======================================================================  ==========================================================
Method                                                                  Description
======================================================================  ==========================================================
``init()``                                                              Must be called before starting optimization. Allows the
                                                                        function to generate internal data after configuration.
``bool isFeasible(SearchPointType)``                                    Returns true if a search point is feasible.
``closestFeasible( SearchPointType&)``                                  Selects the feasible point closest to an infeasible one.
``proposeStartingPoint( SearchPointType &)``                            Returns an initial (possibly random) guess for a solution.
``ResultType eval( SearchPointType)``                                   Evaluates the function on a given point.
``ResultType operator()( SearchPointType)``                             Convenience operator. Calls ``eval``.
``ResultType evalDerivative( SearchPointType, FirstOrderDerivative)``   Evaluates the function as well as the first derivative.
``ResultType evalDerivative( SearchPointType, SecondOrderDerivative)``  Evaluates the function as well as the first and second
                                                                        derivative.
======================================================================  ==========================================================


``init`` allows functions to have random components in their setup. For example,
certain benchmark functions can feature random rotation matrices or optimal points.
It is also useful because it allows for easy, centralized configuration and allows
the objective function to update its internal data structures before optimization.
This function is automatically called by the optimizer in its ``init`` function.

If the search space is a vector space, an additional function is added which
returns the dimensionality of the function:


==============================================================================   ===============================================================================
Method                                                                           Description
==============================================================================   ===============================================================================
``std::size_t numberOfVariables()``                                              The dimensionality of the input point.
==============================================================================   ===============================================================================


Aside from this interface, objective functions also have a name which can be
used for automatic generation of output messages; can be configureed from file;
and store the number of times eval was called. The last point is needed when
benchmarking optimizers:


==============================================================================   ===============================================================================
Method                                                                           Description
==============================================================================   ===============================================================================
``std::string name()``                                                           Returns the name of the function.
``std::size_t evaluationCounter()``                                              Returns the number of function evaluations since the last call to init.
``configure(PropertyTree)``                                                      Configures the objective function. Must be called before ``init``.
==============================================================================   ===============================================================================



In summary,
an objective functions has a very simple life cycle: first it is created and
configured, and after that, init is called. Now the function can be evaluated
using the different forms of ``eval`` or ``evalDerivative``.



The hierarchy of objective functions
------------------------------------


All objective functions are derived from AbstractObjectiveFunction. However,
aside from the benchmark functions used for standard testing of optimizers,
most objective functions are derived from more refined interfaces which add
additional methods.

In machine learning, single-objective functions typically need to handle data:
supervised learning tasks use pairs of inputs and labels, and unsupervised
learning tasks only take inputs. To facilitate handling of data for objective
functions, the two interfaces :doxy:`SupervisedObjectiveFunction` and
:doxy:`UnsupervisedObjectiveFunction` are defined:


====================================================   ==============================================
Interfaces                                             Description
====================================================   ==============================================
``SupervisedObjectiveFunction<InputType,LabelType>``   Offers a function ``setDataset`` which takes
                                                       a ``LabeledData<InputType,LabelType>`` object
``UnsupervisedObjectiveFunction<InputType>``           Offers a function ``setData`` which takes an
                                                       ``UnlabeledData<InputType,LabelType>`` object
====================================================   ==============================================


For the multi-objective case, a multi-objective specialization is defined.
This only adds a method that will return the number of objectives:

===============================================  ======================================
Interfaces                                       Description
===============================================  ======================================
``MultiObjectiveFunction<InputType,LabelType>``  Adds a method numberOfObjectives()
===============================================  ======================================




List of Objective functions
----------------------------------------------------------------


Currently there are no multi-objective functions implemented in Shark aside from some
benchmark fucntions, which can be found in ``include/shark/ObjectiveFunctions/Benchmarks``.
However, Shark offers a variety of single-objective functions:

============================================  ===================================================================================
Model                                         Description
============================================  ===================================================================================
:doxy:`CombinedObjectiveFunction`             Weighted sum of several other objective functions.
:doxy:`ErrorFunction`                         Uses a Model, some data and one of the :doc:`losses` to define a supervised problem
:doxy:`NoisyErrorFunction`                    Same as ErrorFunction, but it only uses a subset of the data at every call.
                                              Thus the return value for a given point is noisy.
:doxy:`CrossValidationError`                  Using a partitioning of the data set, trains and evaluates on all possible sets
                                              of training and validation splits. The mean error on all valdiation sets
                                              is returned. Training time is proportional to the number of partitions.
:doxy:`LooError`                              Most extreme form of cross validation: all but one point are part of the training
                                              set. Usually extremely slow.
:doxy:`SparseFFNetError`                      Same as ErrorFunction, but imposes a sparseness constraint on the activation of the
                                              hidden neurons of a neural network using the Kullback-Leibler divergence.
:doxy:`DenoisingAutoencoderError`             Trains a neural network to be an autoencoder which adds noise on the input
                                              by setting an input to 0.
:doxy:`SvmLogisticInterpretation`             Model selection for SVMs using a maximum-likelihood criterion
:doxy:`RadiusMarginQuotient`                  Model selection for SVMs by optimizing the radius-margin quotient.
:doxy:`NegativeGaussianProcessEvidence`       Model selection for a regularization network/Gaussian process.
:doxy:`LooErrorCSvm`                          Special case of the LooError for SVMs using the structure of the solution
                                              to enhance evaluation.
============================================  ===================================================================================


