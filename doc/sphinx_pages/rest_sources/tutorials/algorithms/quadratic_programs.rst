Quadratic Program Solver
===============================================

Shark comes with its own solver for Quadratic Programs of the form:

.. math::
  \max_{\alpha} & v^T\alpha - \frac 1 2 \alpha^T Q \alpha \\
  \text{s.t.}  & l_i < \alpha_i < u_i\\
  & \[ \sum_i \alpha_i = 0 \]

Where :math:`v` is a vector and :math:`Q` a matrix. The sum to zero constraint is
optional and the solver is designed for medium to large scale problems. Most importantly it
does not assume that :math:`Q` fits into main memory.
This type of problem most often arises with Support Vector Machines, but especially
the version without equality constraint can be found in a lot of different problem
settings. We will in the following describe the structure of the solver, how
to use it and how to extend it to possibly different types of problems.

The Algorithm used is the Sequential Minimal Optimization(SMO) method which is an decomposition
algorithm. It decomposes the algorithm in a sequence of small problems including only 1 or 2 variables
and solves this sequence iteratively until the algorithm converges to the optimal solution.

More precisely, at every iteration a working set of 1 or 2 variables are chosen of the
problem and this sub problem solved optimally. For more information on SMO read...

.. todo::

    describe basics of algorithm: working set selection, shrinking and update step.

Components of the Solver
---------------------------
The Solver consists of 3 parts which are combined in the QpSolver class:

* problem description
* Problem Definition/shrinking strategy
* working set selection strategy

For example an C-SVM problem with equality constraint and enabled shrinking is set-up like this::

    //problem description
    typedef CSVMProblem<MatrixType> SVMProblemType;
    MatrixType matrix(...);
    SVMProblemType svmProblem(matrix,dataset.labels(),base_type::m_C);

    //constraint specification /shrinking strategy
    typedef SvmShrinkingProblem<SVMProblemType> ConstrainedProblemType;
    ConstrainedProblemType problem(svmProblem,base_type::m_shrinking);

    //combine as a solver
    QpSolver< ConstrainedProblemType > solver(problem);
    //and solve the problem
    QpStoppingCondition stop;
    solver.solve(stop);

The problem description parameterizes the problem, that is the linear and quadratic terms
:math:`v` and :math:`Q`  as well as lower and upper bounds :math:`l` and :math:`u` as well as a
starting point :math:`\alpha` which in this case is just the zero vector. We encapsulate this
description inside a convenient class for easier reuse. The MatrixType is a special
matrix type designed for very big matrices and matrices induced by datapoints and a kernel.

The problem definition transforms the problem description into a real instance of the problem.
This means that additional data structures are allocated, 
for example the gradient of the problem at a given starting point alpha. 
Especially the ConstrainedProblem knows how to perform an optimization step with a given working set.

The working set selection strategy is implicit in this case, as the constraint specification readily comes with a
suggestion for it. We could make this more explicit and tell the Solver to use a specific working set selection strategy::

    QpSolver< ConstrainedProblemType, MVPSelectionCriterion > solver(problem);

In the end a simple call to solve solves the complete problem with a stopping criterion
defined by QpStoppingCriterium which comes with reasonable default values.

In the following, we will describe the Design, starting with the QpSolver class.



QpSolver<ProblemType, SelectionStrategy>
----------------------------------------


This class is a very simple wrapper for the other components which just solves
a given problem. In pseudo code a slightly simplified version of the
SMO algorithm is::

    while(!someStoppingCriterionIsMet){
        (i,j) <- selectWorkingSet(problem);
        problem.updateSMO(i,j)
        problem.shrink()
    }

The QpSolver class thus only takes a description of the problem and a description of the
selection strategy and iteratively calls this methods until the given stopping criterion is met.



The Problem definition
----------------------


This part is the core of the solver and does most of the calculations. The problem interface
can be described by a high-level and a low level interface where the high level interface dscribes
the more abstract parts of the solving process and the low level interface can be used by the working set selection
strategy. The high level interface consists of the following methods:


============================================   ============================================================================
Method                                         Description
============================================   ============================================================================
``smoStep(i,j)``                               Perform a SMO-Step on the problem with working set
                                               given by indices (i,j). The sub-problem is solved optimally
                                               taking the constraints into account and a numerically
                                               stable update of the alpha values and the internal variables
                                               is performed.
``double checkKKT()``			       Returns the current accuracy of the problem derived from the KKT conditions.
					       This is usually of the inf norm of the gradients of variables which can move.
``bool shrink(accuracy)``                      Shrink the problem. At a given level of accuracy of the current solution
                                               unshrinking may be performed previously, to find shrinking errors.
                                               The return value indicates whether a variable was shrunk.
``unshrink()``                                 Unshrink the problem. All Variables become active again and
                                               the gradient are updated. This operation can be very slow!
``functionValue()``                            Returns the function value of the current alpha.
============================================   ============================================================================

Only the high level interface is used by the QpSolver class, thus a given problem type might offer a different
low level interface. This for example happens for the Multiclass Svm problems which can include more complex constraints.
However as box constrained problems are the most important ones we give it as an example:

============================================   ========================================================================
Method                                         Description
============================================   ========================================================================
``dimensions()``                               Number of variables of the problem.
``active()``                                   Number of active variables of the problem. Be aware that
                                               only the data structures of active variables is updated and a
                                               working set can not contain indices of variables which are not active.
                                               variables with indices ``[0,...,active()-1]`` are allowed.
``alpha(i)``                                   Value of the i-th variable
``gradient(i)``                                Gradient of the i-th variable at the current point
``boxMin(i)``                                  Lower bound of the i-th variable.
``boxMax(i)``                                  Upper bound of the i-th variable.
``isLowerBound(i)``                            Returns ``alpha(i) == boxMin(i)`` in an optimized way.
``isUpperBound(i)``                            Returns ``alpha(i) == boxMax(i)`` in an optimized way.
``linear(i)``                                  Returns :math:`v_i`.
``quadratic()``                                Returns a reference to :math:`Q`
``permutation(i)``			       Returns the original index of the i-th variable bfore permutation
``diagonal(i)``                                Returns :math:`Q(i,i)`
``flipCoordinates(i,j)``		       Swaps the i-th and j-th variable
============================================   ========================================================================



Currently there are four types of Problems defined which adhere to this interface:


============================================   =======================================================
Class                                          Description
============================================   =======================================================
BoxConstrainedProblem                          Simple Box constraints and no equality constraints.
                                               Can use working sets of size 1 (i==j). Does not implement
                                               shrinking.
BoxConstrainedShrinkingProblem                 BoxConstrainedProblem with shrinking strategy
SvmProblem                                     Box and Equality constraint. Can only use working sets
                                               of size 2 as it is otherwise impossible to fulfill the
                                               equality constraint. Does not implement shrinking.
SvmShrinkingProblem                            SvmProblem with shrinking strategy
============================================   =======================================================

Even for the Shrinking-Versions shrinking can be turned off in the constructor. The reason for the
two separate versions is that it makes testing easier.


Kernel Matrices
-------------------------------------------

.. todo::

    write this section
