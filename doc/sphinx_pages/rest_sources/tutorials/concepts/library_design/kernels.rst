

Kernels
=======


The term "kernel function" is much overloaded in mathematics and statistics.
For most machine learning applications, and prominently so support vector machines
(SVMs), the term "kernel function" -- or just "kernel" --  is used as a shorthand
for "Mercer kernel"[Mercer1909]_  or "reproducing kernel". Another important term
is that of the so-called "kernel trick". Below we will first issue a brief reminder
of these terms and concepts. If you are well familiar with Mercer/reproducing kernels,
feel free to jump to the next section on :ref:`label_for_kernels_in_shark`.



Background
----------


Mathematical Background
&&&&&&&&&&&&&&&&&&&&&&&


Given some set :math:`\mathcal X`, a reproducing kernel :math:`k(x,y)` for any
:math:`x,y  \in \mathcal X` can formally be defined in two ways: either as
the equivalent of a scalar product in a Hilbert space
:math:`\mathcal H` of functions on :math:`\mathcal X`, or as a function of two
arguments on :math:`\mathcal H` which is symmetric and positive definite. The
two definitions give rise to the same notion of a kernel function. For more
details, see [Aronszajn1950]_, [Mercer1909]_, and [Scholkopf2002]_, for example.
In the following, we loosen mathematical precision and discuss the concepts
of kernels in view of their implications for efficient computations in kernel-based
learning algorithms.


Algorithmic advantages
&&&&&&&&&&&&&&&&&&&&&&

Consider any Kernel :math:`k` on :math:`\mathcal X` to correspond to a scalar
product in some other space :math:`\mathcal H`:

.. math::
  k(x,y) = \langle \phi(x),\phi(y) \rangle_{\mathcal H}

where :math:`x` and :math:`y` are elements of :math:`\mathcal X` ,
:math:`\phi` is a map from :math:`\mathcal X` to :math:`\mathcal H`, and
:math:`\langle \cdot, \cdot \rangle_{\mathcal H}` is the scalar product in
:math:`\mathcal H`. Calculating the scalar product in the above expression
via the mapping :math:`\phi(x)` can be highly costly computationally, because
:math:`\mathcal H` may be very high- or even infinite-dimensional. Plus, the
cost of calculating the scalar product will still come on top of that.
But because :math:`k` is a kernel, we can use :math:`k` directly to compute
the scalar product, only using :math:`x` and :math:`y` as input without
applying the map :math:`\phi`. Now many prominent algorithms can be rewritten
such that the only computations required in :math:`\mathcal H` are such scalar
products. Once an algorithm has been formulated in this way, it can be sped
up by efficiently calculating these scalar products via a corresponding
kernel. This is called the "kernel trick". It also allows to enhance simple
linear algorithms, like linear support vector machines, to naturally use
nonlinear mappings. This can raise the generality of such an algorithm.

Further, this generalisation capability does not only apply to algorithms
which use scalar products, but also to ones using distances via the formula:

.. math::
  d(x,y) = \sqrt{\langle \phi(x)-\phi(y), \phi(x)-\phi(y) \rangle_{\mathcal H}}
  =\sqrt{k(x,x) - 2k(x,y) + k(y,y)}

where :math:`d` is the distance between the points :math:`x` and :math:`y`. We call
a kernel normalized, if :math:`k(x,x)=1` for all :math:`x`. In this case calculating
the distance reduces to :math:`d(x,y) =\sqrt{2 - 2k(x,y)}`.



Algorithmic disadvantages
&&&&&&&&&&&&&&&&&&&&&&&&&


While a powerful technique, using kernels has a downside as well. Often
one wants to compute a linear combination of kernels, that is, expressions of
the form :math:`\sum_i^N \alpha_i k(x_i,y)` for fixed :math:`\alpha` and
:math:`x_i`. In particular, the :math:`x_i` may be the points of a training
set, :math:`\alpha` the solution vector of an SVM, for example, and :math:`y`
is some arbitrary and varying point for evaluation. If we were operating
in the higher-dimensional feature space :math:`\mathcal H`, we could explicitly
calculate this linear combination, store it, and much more quickly evaluate
the scalar product with an input vector :math:`y`. Thus it can be seen as a
downside of kernel methods that such linear combinations must be calculated
summand by summand for each new evaluation point :math:`y`.


.. _label_for_kernels_in_shark:

Kernels in Shark
&&&&&&&&&&&&&&&&

Coming back to Shark, several classes are relevant for kernels and the
abovementioned upsides and downsides. All kernel functions' base class is
the :doxy:`AbstractKernelFunction`. A linear combination of kernels as
given above is represented in Shark as a :doxy:`KernelExpansion`. Many
kernel-based algorithms also need to repeatedly evaluate kernels between
different points of the training dataset :math:`x_i` and :math:`x_j`,
:math:`k(x_i,x_j)`. Thus ideally, to save computation time, the overall
matrix :math:`K` with entries :math:`K_{ij}` would be stored in memory.
However, even training sets of sizes of a few hundred thousand make this
prohibitive on common PCs. Therefore, only parts of it may be calculated
at a time, most often matrix rows or blocks. In Shark, the classes
:doxy:`KernelMatrix` and :doxy:`CachedMatrix` as well as some derived
and sibling classes encapsulate kernel Gram matrices. The :doxy:`CachedMatrix`
also automatically takes care of memory handling.



The base class 'AbstractKernelFunction<InputTypeT>'
----------------------------------------------------


The interface of kernels can be understood as a generalization of the interface
of Models to functions taking two arguments of the same type. All kernels
are derived from the abstract class :doxy:`AbstractKernelFunction`. Due to the
demanding computations involving kernel evaluations, the interface is optimized
for speed, and to allow parallelization of the evaluation of different parts of
the kernel Gram matrix at a time. In the following, the basic design decisions
are outlined and explained. Since kernels and models have much in common, also
consider reading the :doc:`models` tutorial if you not already did.

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
BatchInputType             Batch of arguments. Same as Batch<InputType>::type.
ConstInputReference        Constant Reference to InputType as returned by ConstProxyReference<InputType>::type. By default this is InputType const&.
ConstBatchInputReference   Constant Reference to BatchInputType as returned by ConstProxyReference<BatchInputType>::type.
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
``HAS_FIRST_PARAMETER_DERIVATIVE``, ``hasFirstParameterDerivative``  If set, the kernel can evaluate the first derivative w.r.t its parameters.
``HAS_FIRST_INPUT_DERIVATIVE``, ``hasFirstInputDerivative``          If set, the kernel can evaluate the first derivative w.r.t its left input parameters.
                                                                     This is no restriction, since kernel functions are symmetric in their two arguments.
``IS_NORMALIZED``, ``isNormalized``                                  For all :math:`x` it holds  :math:`k(x,x)=1`
``SUPPORTS_VARIABLE_INPUT_SIZE``, ``supportsVariableInputSize``      Between different calls to :math:`k(x,y)` the number of dimensions of the kernel is
                                                                     allowed to vary. This is needed for kernel evaluation of inputs with missing features.
===================================================================  ======================================================================================


Evaluation
&&&&&&&&&&


Next, we introduce the functions to evaluate the kernels. Here we have three
types of functions. The first version simply calculates the kernel value given
two inputs. The second computes the kernel evaluation of two batches of inputs.
Here, the inner product between all points of the first and second batch is calculated
in Hilbert space.
Thus, the resulting type is a matrix of inner products -- a block of the kernel Gram
matrix. The third version takes two batches as well but also a state object. The
state is a data structure which allows the kernel to store intermediate results
of the evaluation of the kernel value. These can later be reused in the computation
of the derivatives. Thus, when derivatives are to be computed, this latter version
must be called beforehand to fill the state object with the correct values.
There is no version of the derivative with two single inputs, because this is a rare
use case. If still needed, batches of size one should be used.

With this in mind, we now present the list of functions for ``eval``, including
the convenience ``operator()``. Let in the following ``I`` be a ``ConstInputReference``
and ``B`` a ``ConstBatchInputReference``.

============================================   =======================================================
Method                                         Description
============================================   =======================================================
double eval(I x, I z)                          calculates :math:`k(x,z)`
void eval(B X, B Z, RealMatrix& K)             calculates :math:`K_{ij}=k(x_i,z_j)` for all elements
                                               :math:`x_i` of X and :math:`z_j` of Z
void eval(B X, B Z, RealMatrix& K, State& )    calls eval(X,Z,K) while storing intermediate results
                                               needed for the derivative functions
double operator()(I x, I z)                    calls eval(x,z)
RealMatrix operator()(B X, B Z)                calls eval(X,Z,K) and returns K.
============================================   =======================================================

For a kernel, it is sufficient to implement the Batch version of eval that
stores the state, since all other functions can rely on it. However, if speed
is relevant, all three eval functions should be implemented in order to avoid
unnecessary copy operations.


Distances
&&&&&&&&&

As outlined before, kernels can also be used to compute distances between points in :math:`\mathcal H`:

============================================   =======================================================
Method                                         Description
============================================   =======================================================
``double featureDistanceSqr(I x, I z)``        Returns the squared distance of x and z.
``double featureDistance(I x, I z)``           Returns the distance of x and z.
``RealMatrix featureDistanceSqr(B X, B Z)``    Returns the squared distance of all points in X to all
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
example the :doxy:`SvmApproximation``.

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


Kernels support several other concepts: they have parameters, can be
configured, serialized and have an externalstate object.

===============================   ===============================================================================
Method                            Description
===============================   ===============================================================================
``numberOfParameters``            The number of parameters which can be optimized.
``parameterVector``               Returns the current parameters of the model.
``setParameterVector``            Sets the new parameter vector.
``configure``                     Configures the model. Options depend on the specific model.
``createState``                   Returns a newly created State object holding the state to be stored in eval.
===============================   ===============================================================================



Kernel Helper Functions
------------------------


The file :doxy:`KernelHelpers.h` defines some free functions that help dealing with
common tasks in kernel usage. Currently this file offers the following functions:


=============================================   ===============================================================================
Method                                          Description
=============================================   ===============================================================================
``calculateRegularizedKernelMatrix``            Evaluates the whole kernel Gram matrix given a kernel and a dataset.
                                                Optionally, a regularization value is added to the main diagonal.
``calculateKernelMatrixParameterDerivative``    Computes the parameter derivative for a kernel Gram matrix defined by a
                                                kernel, dataset, and a weight matrix.
=============================================   ===============================================================================


List of Kernels
----------------------------------------------------------------

We end this tutorial with a convenience list of kernels currently implemented
in Shark, together with a small description.

We start with general purpose kernels:

================================  ========================================================================================================================
Model                             Description
================================  ========================================================================================================================
:doxy:`LinearKernel`              Standard Euclidean inner product :math:`k(x,y) = \langle x,y \rangle`
:doxy:`MonomialKernel`            For a given exponent n, computes :math:`k(x,y) = \langle x,y \rangle^n`
:doxy:`PolynomialKernel`          For a given exponent n and offset b, computes :math:`k(x,y) = \left(\langle x,y \rangle+b\right)^n`
:doxy:`DiscreteKernel`            Uses a symmetric weight matrix to compute the kernel value for a finite, discrete space
:doxy:`GaussianRbfKernel`         Gaussian isotropic ("radial basis function") kernel :math:`k(x,y) = e^{-\gamma ||x-y||^2}`
:doxy:`ARDKernelUnconstrained`    Gaussian Kernel :math:`k(x,y) = e^{-(x-y)^T C(x-y)}` with diagonal parameter matrix C
================================  ========================================================================================================================


Due to convenient mathematical properties, valid positive definite kernels can
be formed by adding and multiplying kernels, among others. This leads to a range
of what we call combined kernels listed below:

=============================  ========================================================================================================================
Model                          Description
=============================  ========================================================================================================================
:doxy:`WeightedSumKernel`      For a given set of kernels computes :math:`k(x,y) = k_1(x,y)+\dots + k_n(x,y)`
:doxy:`ProductKernel`          For a given set of kernels computes :math:`k(x,y) = k_1(x,y) \dots k_n(x,y)`
:doxy:`NormalizedKernel`       Normalizes a given Kernel.
:doxy:`ScaledKernel`           Scales a kernel by a fixed constant
:doxy:`SubrangeKernel`         Weighted sum kernel for vector spaces. Every kernel receives only a subrange of the input
:doxy:`MklKernel`              Weighted sum kernel for heterogenous type input tupels.
                               Every kernel recives one part of the input tuple.
:doxy:`GaussianTaskKernel`     Specialization of the DiscreteKernel for multi task learning
:doxy:`MultiTaskKernel`        Framework kernel for multi task learning with kernels
=============================  ========================================================================================================================



References
----------


.. [Aronszajn1950] Aronszajn, N. Theory of Reproducing Kernels. Transactions of the American Mathematical Society 68 (3): 337–404, 1950.

.. [Mercer1909] Mercer, J. Functions of positive and negative type and their connection with the theory of integral equations.
    In Philosophical Transactions of the Royal Society of London, 1909.

.. [Scholkopf2002] Schölkopf, B. and Smola, A. Learning with Kernels. MIT Press, 2002.
