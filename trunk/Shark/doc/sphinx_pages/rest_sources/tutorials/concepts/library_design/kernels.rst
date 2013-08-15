
Kernels
=======


The term *kernel function*, or kernel for short, is overloaded. Here,
we refer to *positive semi-definite kernels*, that is, the functions inducing
reproducing kernel Hilbert spaces [Aronszajn1950]_. They underlie
the "kernel trick" [Schölkopf2002]_, which is used, for instance, in non-linear
support vector machines (SVMs).

This tutorial covers all knowledge required for using existing and
writing you own kernel based predictors and learning algorithms.
That is, this tutorial explains the kernel interface from a user
perspective. Writing a specialized kernel function is explained in
the tutorial :doc:`writing_kernels`.




Background
----------

Given some set :math:`\mathcal X`, a positive semi-definite kernel
:math:`k:\mathcal X\times\mathcal X\to\mathbb R`
is a symmetric function for which

.. math::
  \sum_{i=1}^N\sum_{j=1}^N a_i a_j k(x_i, x_j) \ge 0

for all :math:`N`, all
:math:`x_1,...,x_N\in\mathcal X`, and all
:math:`a_1,...,a_N\in\mathbb R`.

A kernel :math:`k` on :math:`\mathcal X` corresponds to a scalar
product in a dot product space :math:`\mathcal H`, the so called
feature space:

.. math::
  k(x,y) = \langle \phi(x),\phi(y) \rangle_{\mathcal H}

where :math:`x` and :math:`y` are elements of :math:`\mathcal X` ,
:math:`\phi` is a map from :math:`\mathcal X` to :math:`\mathcal H`, and
:math:`\langle \cdot, \cdot \rangle_{\mathcal H}` is the scalar product in
:math:`\mathcal H`.
For details we refer to [Aronszajn1950]_ and [Mercer1909]_.

Many machine learning algorithms can be written in a way that the only
operations involving input elements are scalar products between those
elements.  A common strategy in machine learning is to map the input
data into a feature space :math:`\mathcal H` and to do the learning in
this feature space.  If the only operations in :math:`\mathcal H` are
scalar products, these can be replaced by kernel function evaluations
rendering explicit computations of the mapping :math:`\phi` to feature
space unnecessary. This has some advantages:

- Typically, the kernel can be computed more
  efficiently than the scalar product itself. This allows for working
  in very high-dimensional feature spaces.

- The kernel provides a clean interface between general and
  problem specific aspects of the learning machine.

Thus, the "kernel trick" allows efficient formulation of nonlinear
variants of any algorithm that can be expressed in terms of dot
products.  The choice of the kernel function is crucial for the
performance of the machine learning algorithm.

The generic distance between to points mapped to a kernel-induced
feature space is given by

.. math::
  d(x,y) = \sqrt{\langle \phi(x)-\phi(y), \phi(x)-\phi(y) \rangle_{\mathcal H}}
  =\sqrt{k(x,x) - 2k(x,y) + k(y,y)}

where :math:`d` is the distance between the points :math:`x` and :math:`y`. We call
a kernel normalized, if :math:`k(x,x)=1` for all :math:`x`. In this case calculating
the distance reduces to :math:`d(x,y) =\sqrt{2 - 2k(x,y)}`.


.. _label_for_kernels_in_shark:

Kernels in Shark
&&&&&&&&&&&&&&&&

Shark provides strong support for kernel-based algorithms.  All kernel
functions' base class is the :doxy:`AbstractKernelFunction`. A linear
combination of kernels is represented in Shark as a
:doxy:`KernelExpansion`

.. math::
  \sum_{i=1}^N \alpha_i k(x_i, . ) + b

with :math:`x_1,...,x_N\in\mathcal X`,
:math:`\alpha_1,...,\alpha_N\in\mathbb R`, and optional bias/offset
parameter :math:`b\in\mathbb R`.

Many kernel-based algorithms need to repeatedly evaluate the kernel on
some training data points :math:`x_1,\dots,x_N` or they operate on the
kernel (Gram) matrix :math:`K` with entries :math:`K_{ij}=k(x_i,x_j)`
directly. To save computation time, the matrix :math:`K` would be
stored in memory.  Depending on the hardware, even training sets with
a few hundred thousand can make this prohibitive. Therefore, often only
parts of :math:`K` are calculated at a time, most often matrix rows
or blocks. In Shark, the classes :doxy:`KernelMatrix` and
:doxy:`CachedMatrix` as well as some derived and sibling classes
encapsulate kernel Gram matrices. The :doxy:`CachedMatrix` also
automatically takes care of memory handling.



The base class 'AbstractKernelFunction<InputTypeT>'
----------------------------------------------------


The interface of kernels can be understood as a generalization of the interface
of Models to functions taking two arguments of the same type. All kernels
are derived from the abstract class :doxy:`AbstractKernelFunction`. Due to the
demanding computations involving kernel evaluations, the interface is optimized
for speed, and to allow parallelization of the evaluation of different parts of
the kernel Gram matrix at a time. In the following, the basic design decisions
are outlined and explained. Since kernels and models have much in common,
consider reading the :doc:`models` tutorial first.

Types
&&&&&


First, we introduce the templated types of a Kernel, which are all inferred from
the only template argument ``InputType`` using several metafunctions. As in the Models,
we have the InputType, and the BatchInputType, which is a batch of inputs.
In contrast to Models, we also introduce special reference types:

========================   =========================================================================================================================
Types                      Description
========================   =========================================================================================================================
InputType                  Argument type of the kernel
BatchInputType             Batch of arguments; same as Batch<InputType>::type
ConstInputReference        Constant reference to InputType as returned
                           by ConstProxyReference<InputType>::type; by default this is InputType const&
ConstBatchInputReference   Constant reference to BatchInputType as returned by ConstProxyReference<BatchInputType>::type
========================   =========================================================================================================================

The reason for the ConstBatchInputReference and ConstInputReference types
is that we want to make use of the structure of the arguments to prevent
unnecessary copying: consider a common case when only single elements
of a batch of data are to be computed. If the batch type then is
a matrix, the argument will be a row of this matrix, and not a vector.
Thus, the argument would be automatically copied into a temporary vector,
which is then in turn fed into the kernel. This is of course unnecessary,
and for fast kernels, the copying can exceed the running time of a kernel
evaluation. Thus we use proxy references for vectors, which simply treat
matrix rows and vectors in the same way. This optimization right now only
works for the class of dense vectors and not for example sparse vectors or
even more complex types.

.. todo::

    implications of this? is there a task in the tracker? etc.


Flags
&&&&&

Like a Model, every kernel has a set of flags and convenience access functions
which indicate the traits and capabilities of the kernel:

===================================================================  ======================================================================================
Flag and accessor function name                                      Description
===================================================================  ======================================================================================
``HAS_FIRST_PARAMETER_DERIVATIVE``, ``hasFirstParameterDerivative``  If set, the kernel can evaluate the first derivative w.r.t its parameters
``HAS_FIRST_INPUT_DERIVATIVE``, ``hasFirstInputDerivative``          If set, the kernel can evaluate the first derivative w.r.t its left input parameters;
                                                                     This is no restriction, since kernel functions are symmetric
``IS_NORMALIZED``, ``isNormalized``                                  For all :math:`x` it holds  :math:`k(x,x)=1`
``SUPPORTS_VARIABLE_INPUT_SIZE``, ``supportsVariableInputSize``      Between different calls to :math:`k(x,y)` the number of dimensions of the kernel is
                                                                     allowed to vary; this is needed for kernel evaluation of inputs with missing features
===================================================================  ======================================================================================


Evaluation
&&&&&&&&&&


Next, we introduce the functions evaluating kernels. We have three
types of functions. The first version simply calculates the kernel
value given two inputs. The second computes the kernel evaluation of
two batches of inputs.  Here, the inner product between all points of
the first and second batch is calculated in Hilbert space.  Thus, the
resulting type is a matrix of inner products -- a block of the kernel
Gram matrix. The third version takes two batches as well but also a
state object. The state is a data structure which allows the kernel to
store intermediate results of the evaluation of the kernel
values. These can later be reused in the computation of the
derivatives. Thus, when derivatives are to be computed, this latter
version must be called beforehand to fill the state object with the
correct values. There is no version of the derivative with two single
inputs, because this is a rare use case. If still needed, batches of
size one should be used. The reason for the state object being external
to the kernel class is that this design allows for concurrent evaluation
of the kernel from different threads, with each thread holding its own
state object.

With this in mind, we now present the list of functions for ``eval``, including
the convenience ``operator()``. Let in the following ``I`` be a ``ConstInputReference``
and ``B`` a ``ConstBatchInputReference``.

============================================   =======================================================
Method                                         Description
============================================   =======================================================
double eval(I x, I z)                          Calculates :math:`k(x,z)`
void eval(B X, B Z, RealMatrix& K)             Calculates :math:`K_{ij}=k(x_i,z_j)` for all elements
                                               :math:`x_i` of X and :math:`z_j` of Z
void eval(B X, B Z, RealMatrix& K, State& )    Calls eval(X,Z,K) while storing intermediate results
                                               needed for the derivative functions
double operator()(I x, I z)                    Calls eval(x,z)
RealMatrix operator()(B X, B Z)                Calls eval(X,Z,K) and returns K.
============================================   =======================================================

For a kernel, it is sufficient to implement the batch version of eval that
stores the state, since all other functions can rely on it. However, if speed
is relevant, all three eval functions should be implemented in order to avoid
unnecessary copy operations.


Distances
&&&&&&&&&

As outlined before, kernels can also be used to compute distances between points in :math:`\mathcal H`:

============================================   =======================================================
Method                                         Description
============================================   =======================================================
``double featureDistanceSqr(I x, I z)``        Returns the squared distance between x and z
``double featureDistance(I x, I z)``           Returns the distance between x and z.
``RealMatrix featureDistanceSqr(B X, B Z)``    Returns the squared distances between all points in X to all
                                               points in Z.
============================================   =======================================================



Derivatives
&&&&&&&&&&&

Some Kernels are differentiable with respect to their parameters. This can for example
be exploited in gradient-based optimization of these parameters, which in turn amounts
to a computationally efficient way of finding a suitable space :math:`\mathcal H` in which
to solve a given learning problem. Further, if the input space is differentiable as well,
even the derivative with respect to the inputs can be computed. This is currently
not often used within Shark aside from certain approximation schemes as for
example the :doxy:`SvmApproximation`.

The derivatives are weighted as outlined in :doc:`../optimization/conventions_derivatives`.
The parameter derivative is a weighted sum of the derivatives of all elements of the block
of the kernel matrix. The input derivative has only weights for the inputs of the right
argument.

.. todo::

    math here? mt: yes please! :)

The methods for evaluating the derivatives are:

===================================   ===============================================================================
Method                                Description
===================================   ===============================================================================
``weightedParameterDerivative``       Computes the weighted derivative of the parameters over all elements of a block
                                      of the kernel Gram matrix.
``weightedInputDerivative``           Computes the derivative with respect of the left argument, weighting over all
                                      right arguments.
===================================   ===============================================================================


Putting everything together, we can calculate the derivative of a kernel
like this::

  BatchInputType X; //first batch of inputs
  BatchInputType Y; //second batch of inputs
  RealMatrix K;     //resulting part of the kernel Gram matrix
  MyKernel kernel;  //the differentiable kernel

  // evaluate K for X and Y, store the state
  boost::shared_ptr<State> state = kernel.createState();
  kernel.eval(X, Y, result, *state);

  // somehow compute some weights and calculate the parameter derivative
  RealMatrix weights = someFunction(result, X, Y);
  RealVector derivative;
  kernel.weightedParameterDerivative(X, Y, weights, *state, derivative);


.. todo::

    i think we need some more explanation on the expected size of
    weights, especially since we don't have type checks in the code
    of weightedParameterDerivative (maybe these should be added, too).
    in any case, the workings of weightedParameterDerivative should be
    explained more, or link to some tutorial where this is done.


Other
&&&&&


Kernels support several other concepts. They have parameters, can be
configured, serialized and have an external state object.

===============================   ===============================================================================
Method                            Description
===============================   ===============================================================================
``numberOfParameters``            The number of parameters which can be optimized
``parameterVector``               Returns the current parameters of the kernel object
``setParameterVector``            Sets the new parameter vector
``configure``                     Configures the kernel. Options depend on the specific kernel class
``createState``                   Returns a newly created State object holding the state to be stored in eval
===============================   ===============================================================================



Kernel Helper Functions
------------------------


The file :doxy:`KernelHelpers.h` defines some free functions that help dealing with
common tasks in kernel usage. Currently this file offers the following functions:


=============================================   ===============================================================================
Method                                          Description
=============================================   ===============================================================================
``calculateRegularizedKernelMatrix``            Evaluates the whole kernel Gram matrix given a kernel and a dataset;
                                                optionally, a regularization value is added to the main diagonal
``calculateKernelMatrixParameterDerivative``    Computes the parameter derivative for a kernel Gram matrix defined by a
                                                kernel, dataset, and a weight matrix
=============================================   ===============================================================================


List of Kernels
----------------------------------------------------------------

Shark implements a number of general purpose kernels:

================================  ========================================================================================================================
Model                             Description
================================  ========================================================================================================================
:doxy:`LinearKernel`              Standard Euclidean inner product :math:`k(x,y) = \langle x,y \rangle`
:doxy:`MonomialKernel`            For a given exponent n, computes :math:`k(x,y) = \langle x,y \rangle^n`
:doxy:`PolynomialKernel`          For a given exponent n and offset b, computes :math:`k(x,y) = \left(\langle x,y \rangle+b\right)^n`
:doxy:`DiscreteKernel`            This kernel on a discrete space is explicitly defined by a symmetric, positive semi definite Gram matrix
:doxy:`GaussianRbfKernel`         Gaussian isotropic ("radial basis function") kernel :math:`k(x,y) = e^{-\gamma ||x-y||^2}`
:doxy:`ARDKernelUnconstrained`    Gaussian kernel :math:`k(x,y) = e^{-(x-y)^T C(x-y)}` with diagonal parameter matrix C
================================  ========================================================================================================================


Valid positive semi-definite kernels can be formed, among others, by
adding and multiplying kernels. This leads to a range of what we call
combined kernels listed below:

=============================  ========================================================================================================================
Model                          Description
=============================  ========================================================================================================================
:doxy:`WeightedSumKernel`      For a given set of kernels computes :math:`k(x,y) = k_1(x,y)+\dots + k_n(x,y)`
:doxy:`ProductKernel`          For a given set of kernels computes :math:`k(x,y) = k_1(x,y) \dots k_n(x,y)`
:doxy:`NormalizedKernel`       Normalizes a given kernel; computes: :math:`k(x,y) = k_1(x,y) / \sqrt{k_1(x,x) k_1(y,y)}`
:doxy:`ScaledKernel`           Scales a kernel by a fixed constant
:doxy:`SubrangeKernel`         Weighted sum kernel for vector spaces; every kernel receives only a subrange of the input
:doxy:`MklKernel`              Weighted sum kernel for heterogenous type input tuples;
                               every kernel receives one part of the input tuple
:doxy:`GaussianTaskKernel`     Specialization of the DiscreteKernel for multi task learning
:doxy:`MultiTaskKernel`        Framework kernel for multi task learning with kernels
=============================  ========================================================================================================================


References
----------


.. [Aronszajn1950] Aronszajn, N. Theory of Reproducing Kernels. Transactions of the American Mathematical Society 68 (3): 337–404, 1950.

.. [Mercer1909] Mercer, J. Functions of positive and negative type and their connection with the theory of integral equations.
    In Philosophical Transactions of the Royal Society of London, 1909.

.. [Schölkopf2002] Schölkopf, B. and Smola, A. Learning with Kernels. MIT Press, 2002.
