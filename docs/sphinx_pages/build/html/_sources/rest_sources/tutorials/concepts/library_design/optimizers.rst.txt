Optimizers
==========

Each optimizer in Shark is an iterative algorithm which tries to find a local
minimum of an objective function. For single objective optimization, we would like
to find the global optimum of the objective function :math:`f` :

.. math::
  x^* = \arg \min_x f(x)

However, if the function has more than one local optimum, we can usually only
find one of them. That is, re-starts may be required.

In multi-objective optimization (also known as multi-criteria or
vector optimization), the goal is to optimize with respect to multiple
objective functions at once.  This usually does not lead to a single
point solution, since there exist trade-offs between the different
objectives.  Therefore, the typical goal of vector optimization is to
approximate the set of Pareto optimal solutions as good as possible.
A solution is Pareto optimal if it cannot be improved in one objective
without getting worse in another one.

Optimizers try to find this solution in a stepwise fashion.  Let us
consider single-objective optimization for now.  Given a solution
:math:`x(t)` at time step :math:`t` with objective value
:math:`f(x(t))`, the optimizer looks for a new point :math:`x(t+1)`
such that :math:`f(x(t+1))<f(x(t))`.  Two important types of optimizers can
be distinguished.  The first, gradient-based algorithms use the
gradient of the objective function. The other approach is
direct/derivative-free search, which traverses the search space
without calculating derivative information explicitly. Direct search
algorithms can be as simple as grid search, which simply probes a
predefined grid of points, or more elaborate, like evolutionary
algorithms, which try to infer or substitute for gradient information
by comparing sets of function values. Shark supports both these kinds
of optimizers, and the following tutorials introduces their basics. An
overview over notable implemented optimizers can be found at the
bottom of this tutorial :ref:`here <label_for_list_of_so_optimizers>`.

.. note::

    Shark always assumes minimization tasks. This is no restriction as
    :math:`\arg \max_x f(x) = \arg \min_x -f(x)`.
    
List of Classes
---------------------------------
* :doxy:`Overview and Base-Classes<optimizers>`
* :doxy:`Gradient-Based Optimizers<gradientopt>`
* :doxy:`Direct-Search Optimizers<singledirect>`
* :doxy:`Multi-Objective Optimizers <multidirect>`

The base class 'AbstractOptimizer<SearchPointType, ResultType, SolutionSetType>'
-------------------------------------------------------------------------------------


:doxy:`AbstractOptimizer` is a general and flexible interface for single- as well as
multi-objective optimization on different search sapces. 
We first describe the general interface before we take
a look at the special cases of both single- and multi-objective optimization.

The three template parameters are used to infer the objective function type and
all types are made public using typedefs:


==========================   =================================================================
Types                        Description
==========================   =================================================================
``SearchPointType``          Single point in the search space, representing an input of the
                             objective function. Most likely RealVector.
``ResultType``               Return type of the objective function. For single objective
                             functions, this is double.
``SolutionSetType``          Represents the current best solution of the optimizer. For single
                             objective functions, this is a point-value pair.
``ObjectiveFunctionType``    Type of objective function the algorithm can optimize. Alias for
                             ``AbstractObjectiveFunction<SearchPointType,ResultType>``.
==========================   =================================================================


Every optimizer imposes a set of requirements on the objective functions. These are
organized as a set of flags which can be queried using convenience functions. Note
that there is a big difference to the flag system of ObjectiveFunctions, Models or
Kernels described in some of the other tutorials: the flags of the latter describe
capabilities and not requirements, as is the case here. Via these flags, it can be
easily checked whether an objective function is compatible with an optimizer or not.


============================================================   ====================================================================
Flag, Accessor function                                         Description
============================================================   ====================================================================
``REQUIRES_VALUE``, ``requiresValue``                          The algorithm needs the value of the objective function for a given
                                                               point to decide which step to take next. This means
                                                               :doxy:`AbstractObjectiveFunction::eval` is allowed to be called and
                                                               needs to return a meaningful value.
``REQUIRES_FIRST_DERIVATIVE``, ``requiresFirstDerivative``     The algorithm needs the first derivative of the function.
``REQUIRES_SECOND_DERIVATIVE``, ``requiresSecondDerivative``   The algorithm needs the second derivative of the function.
``CAN_SOLVE_CONSTRAINED``, ``canSolveConstrained``             The algorithm can solve constrained functions. For this it is
                                                               necessary that the objective function implements
                                                               :doxy:`AbstractObjectiveFunction::isFeasible`.
``REQUIRES_CLOSEST_FEASIBLE``, ``requiresClosestFeasible``     Some algorithms need the ability to receive the nearest feasible
                                                               point given an infeasible one.
============================================================   ====================================================================



An optimizer is allowed to check the presence of the correct flags in the
objective function. Moreover, it is allowed to check the flags without actually
requiring them. This allows for different solving strategies given the special
traits of the objective functions. If an objective function abides by the
requirements of the optimizer, the following functions can be called to obtain
the local optimum:



==================================================================   =========================================================================
Method                                                               Description
==================================================================   =========================================================================
``init(ObjectiveFunctionType)``                                      Initializes the optimizer with numInitPoints() random starting point
                                                                     proposed by the objective function. 
                                                                     The function must set the flag ``CAN_PROPOSE_STARTING_POINT`` and
                                                                     implement the function :doxy:`AbstractObjectiveFunction::proposeStartingPoint`.
``init(ObjectiveFunctionType,std::vector<SearchPointType>)``         Initialize the algorithm using a prespecified set of starting points.
                                                                     Number of points should be ``numInitPoints()`` but the algorithm can try
                                                                     to generate additional points if required.
``numInitPoints()``						     Returns the number of initialisation points required by the algorithm.
``step(ObjectiveFunctionType)``                                      Performs one step of the learning algorithm on the objective function.
``SolutionSetType solution()``                                       Returns the current best solution found.
==================================================================   =========================================================================



Also, optimizers offer several other helper functions
(and, in addition to the below, are serializable):

============================================   =========================================================================
Method                                         Description
============================================   =========================================================================
``name()``                                     Returns the name of the optimizer. Useful for text output of results.
============================================   =========================================================================



Here is a short example on how this interface can be used::

  MyObjectiveFunction f;
  MyOptimizer opt;
  f.init();
  opt.init(f);

  while( !someStoppingCriteronMet(opt,f) ) {
      opt.step(f);
  }
  // get the optimal solution
  MyOptimizer::SolutionSetType solution = opt.solution();




The base class 'AbstractSingleObjectiveOptimizer<SearchPointType>'
--------------------------------------------------------------------------

To this point, we have not clarified how the result of ``solution()`` looks
like. For Single objective optimizers,
the solution type is an instance of :doxy:`SingleObjectiveResultSet`.
It stores the best point found so far as well as its function value.
Printing out the result of the last example would look like::

  std::cout << "value:" << opt.solution().value << " point:" << opt.solution().point;

For initialization, usually only a single starting point is needed. This can either be
generated by the function if it can propose a random starting point, or it
can be provided as second argument to ``init``:



==================================================   =================================================================================
Method                                               Description
==================================================   =================================================================================
``init(ObjectiveFunctionType, SearchPointType)``     Initializes the optimizer with a given starting point.
==================================================   =================================================================================



For a new optimizer, only the new version of ``init`` and ``step``
need to be implemented. The optimizer is allowed to evaluate the given
starting point during initialization.


The base class 'AbstractMultiObjectiveOptimizer<SearchPointType>'
---------------------------------------------------------------------------

.. todo::

    ADD TUTORIAL



.. _label_for_list_of_so_optimizers:

