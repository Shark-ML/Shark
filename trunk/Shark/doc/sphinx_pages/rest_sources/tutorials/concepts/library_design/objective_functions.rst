Objective Functions
===================

An objective function formalizes an optimization problem for which a
(somehow) optimal solution is to be found, for example using
:doc:`optimizers`. Given an input point within its admissible search
space, an objective function returns a (possibly noisy) objective
value and, optionally, its derivative. The input is typically a
vector of real numbers, but the interface allows for more general
cases as well. For single objective optimization the return value is a
real number and the goal is to minimize it. For multi-objective
optimization, the output is a vector holding the objective function
value for each of the multiple goals.

Besides returning a corresponding value for a search point, objective functions
in Shark also manage their own search space. In detail, they provide a method
indicating if a given point is feasible, and possibly also a method which can
provide the feasible point closest to an infeasible one.



The base class 'AbstractObjectiveFunction <SearchPointType, ResultT>'
---------------------------------------------------------------------


Template arguments and public types
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

The base class :doxy:`AbstractObjectiveFunction` has two template arguments.
The first is the SarchPointType, which defines the admissible type of inputs.
``ResultT`` defines the return type of the objective
function, which for single-objective functions is ``double``, and ``RealVector``
for multi-objective functions. Thus, a typical single objective function has as
type signature ``AbstractObjectiveFunction< RealVector, double >`` and
a multi objective function ``AbstractObjectiveFunction< RealVector, RealVector >``.
The following two typedfs are used throughout hark to make the distinction clear:

=================================   ===============================================================================
Typedef                             Description
=================================   ===============================================================================
``SingleObjectiveFunction``         ``SearchPointType`` is ``RealVector``, ``ResultType`` is ``double``
``MultiObjectiveFunction``          ``SearchPointType`` is ``RealVector``, ``ResultType`` is ``RealVector``
=================================   ===============================================================================

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
Flag, accessor function                                          If set to true ...
===============================================================  ==========================================================
``HAS_VALUE``, ``hasValue``                                      The objective function can calculate its own function
                                                                 value. Since this attribute is common to most objective
                                                                 functions, this flag is set by default. See comment below.
``HAS_FIRST_DERIVATIVE``, ``hasFirstDerivative``                 The first derivative can be computed.
``HAS_SECOND_DERIVATIVE``, ``hasSecondDerivative``               The second derivative can be computed.
``IS_CONSTRAINED_FEATURE``, ``isConstrained``                    The input space is constrained, and the function must offer a method
                                                                 ``isFeasible`` for checking whether a point is feasible.
``HAS_CONSTRAINT_HANDLER``, ``hasConstraintHandler``		 Indicates that the constraints are governed by a handler.						 
``CAN_PROPOSE_STARTING_POINT``, ``canProposeStartingPoint``      The objective function can propose a feasible starting
                                                                 point from which the optimizer can start the optimization.
``CAN_PROVIDE_CLOSEST_FEASIBLE``, ``canProvideClosestFeasible``  A constrained function can provide a function
                                                                 ``closestFeasible`` which returns the closest feasible
                                                                 point given an infeasible one.
``IS_THREAD_SAFE``, ``isThreadSafe``                             This flag indicates that eval and evalDerivative can be
                                                                 called in parallel by the optimizer for different points.
===============================================================  ==========================================================


The flag ``HAS_VALUE`` might seem strange at first. However, there
exist scenarios where we want to implement the gradient of an
objective function without a corresponding function itself.  For some
optimizers, the gradient information is enough to find a better
point. If the flag is not set, calling ``eval`` is not allowed, and
other functions return meaningless values like ``qnan``.
The flag ``HAS_CONSTRAINT_HANDLER`` indicates that 
constraints are represented by a secondary object. This object can be quried
and might offer more spcial information about the constraints. for example 
it might indicate that it reprsents box constraints - 
in this case the exact shape of the box can be queried and an algorithm 
might choose a specific strategy based on this information.


Interface
&&&&&&&&&


Using an objective function is easy, as can be seen in the following
short list of functions:


======================================================================  ===================================================================
Method                                                                  Description
======================================================================  ===================================================================
``init()``                                                              Must be called before starting optimization and allows the
                                                                        function to generate internal data after configuration
``getConstraintHandler()``                                              Returns the constraint handler of the function, if it has one.
``announceConstraintHandler(ConstraintHandler*)``                       Protected function which is called from a derived class to indicate 
									the presence of the handler. Sets up all flags of the objective 
									function automatically.
``bool isFeasible(SearchPointType)``                                    Returns true if a search point is feasible
``closestFeasible(SearchPointType&)``                                   Selects the feasible point closest to an infeasible one
``proposeStartingPoint(SearchPointType &)``                             Returns an initial (possibly random) guess for a solution.
``ResultType eval(SearchPointType)``                                    Evaluates the function on a given point
``ResultType operator()(SearchPointType)``                              Convenience operator calling ``eval``
``ResultType evalDerivative(SearchPointType, FirstOrderDerivative)``    Evaluates the function as well as the first derivative
``ResultType evalDerivative(SearchPointType, SecondOrderDerivative)``   Evaluates the function as well as the first and second
                                                                        derivative
======================================================================  ===================================================================

The function ``init`` allows objective functions to have random
components in their setup. For example, certain benchmark functions
can feature random rotation matrices or optimal points.  It is also
useful because it allows for easy, centralized configuration and
allows the objective function to update its internal data structures
before optimization.  This function is automatically called by the
optimizer in its ``init`` function.

If the search space is a vector space, additional functions are added which
return or set the dimensionality of the objective function:


==============================================================================   ===============================================================================
Method                                                                           Description
==============================================================================   ===============================================================================
``std::size_t numberOfVariables()``                                              Returns the required dimensionality of the input point
``bool hasScalableDimensionality()``                                             Returns true when the input space of the function can be scaled. 
										 This is useful for Benchmarking
``setNumberOfVariables( std::size_t )``						 Sets the dimensionality of the input points if the function is scaleable.
==============================================================================   ===============================================================================

MultiObjectiveFunctions offer the same mechanism for the number of objectives

==============================================================================   ===============================================================================
Method                                                                           Description
==============================================================================   ===============================================================================
``std::size_t numberOfObjectivees()``                                            Returns the dimensionality of a result vector
``bool hasScalableObjectives()``          					 Returns true if the number of objectives can be changed, 
										 for example for Benchmarking.                               
``setNumberOfVariables( std::size_t )``						 Sets the number of objectives if it is scalable.
==============================================================================   ===============================================================================


Besides from this interface, objective functions also have a name
which can be used for automatic generation of output messages; can be
configured from a file; and store the number of times ``eval`` was
called. The last feature is needed when benchmarking optimizers:


==============================================================================   ===============================================================================
Method                                                                           Description
==============================================================================   ===============================================================================
``std::string name()``                                                           Returns the name of the function.
``std::size_t evaluationCounter()``                                              Returns the number of function evaluations since the last call to init.
``configure(PropertyTree)``                                                      Configures the objective function. Must be called before ``init``.
==============================================================================   ===============================================================================



In summary, an objective functions has a very simple life
cycle. First, it is created and configured. After that, ``init`` is
called. Then the function can be evaluated using the different forms
of ``eval`` or ``evalDerivative``.

List of Objective functions
----------------------------------------------------------------

There are various single- and multi-objective benchmark functions
implemented in Shark, which can be found in
``shark/ObjectiveFunctions/Benchmarks``.

Furthermore, Shark offers a variety of single-objective functions:

============================================  ===================================================================================
Model                                         Description
============================================  ===================================================================================
:doxy:`CombinedObjectiveFunction`             Weighted sum of several other objective functions.
:doxy:`ErrorFunction`                         Uses a Model, some data and one of the :doc:`losses` to define a supervised problem.
:doxy:`NoisyErrorFunction`                    Same as ErrorFunction, but it only uses a subset of the data at every call.
                                              Thus the return value for a given point is noisy
:doxy:`CrossValidationError`                  *k*-fold cross validation. The mean error over all *k* validation sets
                                              is returned. Training time is proportional to the number of partitions.
:doxy:`LooError`                              Leave-one-out error, the most extreme form of cross validation in which all but one 
                                              point are part of the training sets.
:doxy:`LooErrorCSvm`                          Special case of the ``LooError`` for SVMs using the structure of the SVM solution
                                              to speed-up evaluation.
:doxy:`SparseFFNetError`                      Same as ``ErrorFunction``, but imposes a sparseness constraint on the activation of the
                                              hidden neurons of a neural network using the Kullback-Leibler divergence.
:doxy:`SvmLogisticInterpretation`             Model selection for SVMs using a maximum-likelihood criterion
:doxy:`RadiusMarginQuotient`                  Model selection for SVMs by optimizing the radius-margin quotient.
:doxy:`NegativeGaussianProcessEvidence`       Model selection for a regularization network/Gaussian process.
:doxy:`KernelBasisDistance`                   Measures the distance between two points in a kernel space to
					      approximate one point with a much smaller basis.
:doxy:`KernelTargetAlignment`                 Model selection algorithm which measures for a given kernel,
					      how similar points of the same class are and how dissimilar points of
					      different classes.
:doxy:`NegativeLogLikelihood`                 Measures how probable a dataset is under a given model parameterized
					      by the vector of parameters given in the function argument.
============================================  ===================================================================================


