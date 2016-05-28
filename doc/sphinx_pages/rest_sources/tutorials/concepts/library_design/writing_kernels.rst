
Writing Kernel Functions
========================

This tutorial explains in detail how to write your own kernel function.
Be sure to read the general tutorial on :doc:`kernels` first. You should
also be familiar with :doc:`batches`, and for the advanced topics
with :doc:`../optimization/conventions_derivatives`.

Existing kernel functions in Shark are a good starting point for writing
a specialized kernel function. It is always a good idea to look for a
related kernel before reinventing the wheel. However, for the purpose of
this tutorial we will start from scratch.


Example Kernel
--------------

We'll work with two examples in the following. Our main example will be
a kernel on strings. This will allow us to introduce most concepts. For
the rather specific topic of input derivatives we will then switch
to the isotropic Gaussian kernel (as implemented in the class
:doxy:`GaussianRbfKernel`), since strings do not have a differentiable
structure.

For the time being, assume that our input data takes the form of
character strings over a fixed alphabet. Instances could be words, but
they could equally well be gene sequences. An obvious data structure for
representing such objects in C++ is ``std::string``. One could easily
consider a more general container here, such as ``std::basic_string<T>``, so
alphabets can have more than 256 characters. The kernel class could be
templatized either for the character type or even for the whole
container type. Here we stick to a simple ``std::string``, since the
generalization to templates is rather straightforward.

For simplicity, our example kernel will work on strings of fixed length
*N*. In particular we will assume that both input strings have the same
length. Our kernel class will represent a whole family of kernel
functions with an adjustable parameter. For a string :math:`s`, let
:math:`s_i` denote the *i*-th character, and let :math:`\delta(a,b)`
denote the Kronecker symbol, which is one if :math:`a` and :math:`b` match (are equal)
and zero otherwise. Then our kernel reads as follows:

.. math::
  k(s, s') = \sum_{i,j=1}^{N} \delta(s_i, s'_j) \exp \Big( -\gamma (i-j)^2 \Big)

This kernel is a double sum over all pairs of symbols in the two
sequences. Matching entries have a positive contribution that decays
with the distance in position. The parameter :math:`\gamma > 0` controls
how quickly the value decays. In bioinformatics this kernel is known as
the *oligo* *kernel* [Meinicke2004]_ for *k*-mers of length *k*=1. This
kernel is probably not very good for processing real sequences, since it
treats all matches independently and without taking sequence information
into account. Larger values of *k* should be considered in practice, but
we will nevertheless use this kernel for illustration purposes.


A Brand New Kernel Class
------------------------

Now we will cast the above formula into a piece of C++ code. We will
call the new kernel class ``MySequenceKernel``. In Shark, all kernel
functions are derived from the ``AbstractKernelFunction`` interface.
This interface is a template::

  template <class InputTypeT> class AbstractKernelFunction;

The template parameter ``InputTypeT`` describes the data structure holding
inputs to the kernel, in this case ``std::string``. We define a subclass
with a custom constructor taking the kernel parameter :math:`\gamma` as a
parameter, as well as a member variable holding its value::

	#include <shark/Models/Kernels/AbstractKernelFunction.h>
	#include <string>
	
	using namespace shark;
	
	class MySequenceKernel : public AbstractKernelFunction<std::string>
	{
	public:
	    typedef AbstractKernelFunction<std::string> base_type;

	    MySequenceKernel(double gamma)
	    : m_gamma(gamma)
	    {
	        SHARK_ASSERT(m_gamma > 0.0);     // debug mode check
	    }

	protected:
	    double m_gamma;
	};

The super class ``AbstractKernelFunction<std::string>`` introduces an
interface for the evaluations of the kernel. The super class itself
inherits the interfaces ``INameable``, ``IParameterizable``,
``ISerializable``, and ``IConfigurable``, each of which introduces
further parts of the interface. We will go through all of these step by
step.


Giving the Kernel a Name
------------------------

Most things in Shark have a name, i.e., most top level interface classes
inherit ``INameable``. This interface requires that we identify ourselves
by name at runtime as follows::

	std::string name() const
	{
	    return "MySequenceKernel";
	}

The standard convention employed by more than 90% of Shark's classes
is to return the class name. We recommend to stick to this convention
unless there are reasons to deviate.


Evaluating the Kernel
---------------------

The above code compiles, but instantiating an object of type
``MySequenceKernel`` will fail, since a number of interface functions are
pure virtual and need to be overridden. The most important of these is
the ``AbstractKernelFunction::eval`` function. This is the central
location where the actual evaluation of the kernel function takes
place::

	virtual void eval(
	        ConstBatchInputReference batchX1,
	        ConstBatchInputReference batchX2,
	        RealMatrix& result,
	        State& state
	    ) const = 0;

This function takes four arguments: two batches of inputs (refer to the
tutorials on :doc:`batches` if you have not yet done so), a matrix-valued
result parameter, and an intermediate state object for the computation
of derivatives. For the time being we will ignore kernel derivatives and
thus the state object and focus on our core task, namely the computation
of the results from the inputs.

The eval function is supposed to fill the result matrix with the results
of the kernel function applied to all pairs of inputs found in the two
batches. In other words, ``result(i, j)`` has to be filled in with
:math:`k(x_i, y_j)`, where :math:`x_i` is the *i*-th element of the first
batch and :math:`y_j` is the 
*j*-th element of the second batch. In other
words, the ``eval`` function computes the kernel Gram matrix of the two
batches. For the special case of batches of size one it computes a
single kernel value. The reason for computing whole Gram matrices
instead of single kernel values is computation speed: this makes it
possible to profit from optimized linear algebra routines for the
computation of many standard kernels. In our example this is not the
case, therefore we will simply fill the Gram matrix in a double loop::

	void eval(
	        ConstBatchInputReference batchX1,
	        ConstBatchInputReference batchX2,
	        RealMatrix& result,
	        State& state
	    ) const
	{
	    std::size_t s1 = size(batchX1);
	    std::size_t s2 = size(batchX2);
	    result.resize(s1, s2);
	    for (std::size_t i=0; i<s1; i++) {
	        ConstInputReference x_i = get(batchX1, i);
	        for (std::size_t j=0; j<s2; j++) {
	            ConstInputReference y_j = get(batchX2, j);
	            // TODO: evaluate k(x_i, y_j)
	        }
	    }
	}

Inside the double loop we have the two references ``x_i`` and ``y_j`` to
two string instances available, and it remains to compute the kernel
value according to the above formula. The type ``ConstInputReference`` is
defined by the ``AbstractKernelFunction`` (just like the
``ConstBatchInputReference`` type). The following is a brute force
implementation::

	void eval(
	        ConstBatchInputReference batchX1,
	        ConstBatchInputReference batchX2,
	        RealMatrix& result,
	        State& state
	    ) const
	{
	    std::size_t s1 = size(batchX1);
	    std::size_t s2 = size(batchX2);
	    result.resize(s1, s2);
	    for (std::size_t i=0; i<s1; i++) {
	        ConstInputReference x_i = get(batchX1, i);
	        for (std::size_t j=0; j<s2; j++) {
	            ConstInputReference y_j = get(batchX2, j);

	            // evaluate k(x_i, y_j)
	            std::size_t N = y_j.size();       // string length
	            SHARK_ASSERT(x_i.size() == N);    // DEBUG check
	            double sum = 0.0;
	            for (std::size_t p=0; p<N; p++) {
	                for (std::size_t q=0; q<N; q++) {
	                    if (x_i[p] == y_j[q]) {
	                        sum += std::exp(-m_gamma * ((p-q) * (p-q)));
	                    }
	                }
	            }

	            // fill the result matrix
	            result(i, j) = sum;
	        }
	    }
	}

The core algorithmic work is already done!

It is actually possible to speed up the computation quite a bit: the
exponential function is only evaluated at finitely many points, one for
each possible distance between ``p`` and ``q``. These values can be precomputed
(e.g., in the function ``setParameterVector`` below). We will not do
this here since the focus of this tutorial is not on specific
algorithmic improvements.

In the ``AbstractKernelFunction`` interface there is a variants of the
``eval`` function taking two single instances. This is probably closer
to what's naively expected as a kernel function interface. The default
implementation creates two batches of size one and calls the above
function. This means that the data is copied, which is inefficient.
Therefore one may wish to overload this function as follows::

	double eval(ConstInputReference x1, ConstInputReference x2) const {
	    std::size_t N = x1.size();       // string length
	    SHARK_ASSERT(x2.size() == N);    // DEBUG check
	    double sum = 0.0;
	    for (std::size_t p=0; p<N; p++) {
	        for (std::size_t q=0; q<N; q++) {
	            if (x1[p] == x2[q]) sum += std::exp(-m_gamma * ((p-q) * (p-q)));
	        }
	    }
	    return sum;
	}

Overloading this function is not required, but it will speed up algorithms
that need single kernel evaluations. This is rarely the case in Shark,
but it often happens is rapid prototyping code.

Now our first version of the ``MySequenceKernel`` class is operational.
It can be instanciated like this::

	int main(int argc, char** argv)
	{
	    double gamma = strtod(argv[1], NULL);
	    MySequenceKernel kernel(gamma);
	}

Most of Shark's kernel-based learning algorithms are directly ready for
use with the new kernel, such as various flavors of support vector
machines and Gaussian processes. For most tasks we are done at this
point. If this is all you need then you can stop here. Enjoy!

However, in some situations the ability to evaluate the kernel function
alone is not enough. Additional functionality is provided by a number of
interfaces, discussed in the following.


Serialization
-------------

Serialization is a nice-to-have feature. Shark kernels inherit the
ISerializable interface, which demands that two simple functions being
overloaded. We serialize the value of the parameter :math:`\gamma`::

	void read(InArchive& archive) {
	    archive >> m_gamma;
	}

	void write(OutArchive& archive) const {
	    archive << m_gamma;
	}


The Parameter Interface
-----------------------

Recall that the parameter :math:`\gamma` controls how fast the contribution
of a symbol match decays with the distance of the symbols. This parameter
will most probably need problem specific tuning to achieve optimal
performance of any kernel-based learning method. That is, this parameter
should be set by a data driven procedure, which is nothing but machine
learning this parameter from data.

For this purpose its value needs to be accessible by optimization
algorithms in a unified way. This is achieved by the ``IParameterizable``
interface. This is the core learning-related interface of the Shark
library. It allows to query the number of (real-valued) parameters, and
it defines a getter and a setter for the parameter vector::

	std::size_t numberOfParameters() const {
	    return 1;
	}

	RealVector parameterVector() const {
	    return RealVector(1, m_gamma);
	}

	void setParameterVector(RealVector const& newParameters) {
	    SHARK_ASSERT(newParameters.size() == 1);
	    SHARK_ASSERT(newParameters(0) > 0.0);
	    m_gamma = newParameters(0);
	}

Recall the comment above on precomputing the exponential function values
to speed up evaluation. The ``setParameterVector`` function is the best
place for this computation.


Parameter Derivatives
---------------------

We have still left open how to tune the parameter :math:`\gamma` in a problem
specific way. Cross-validation is an obvious, robust, but time consuming
possibility. Other objective functions for kernel selection allow for
more efficient parameter optimization (in particular when there is more
than one parameter), e.g., gradient-based optimization [Igel2007]_ of
the kernel target alignment [Cristianini2002]_. This requires the
kernel function to be differentiable w.r.t. its parameters. Note that we
do not need a differentiable structure on inputs (strings, which there
isn't), but only on parameter values (positive numbers for :math:`\gamma`),
as well as a smooth dependency of the kernel on the parameters.

.. math::
  \frac{\partial k(s, s')}{\partial \gamma} = - \sum_{i,j=1}^{N} \delta(s_i, s'_j) (i-j)^2 \exp \Big( -\gamma (i-j)^2 \Big)

On the software side, we have to make known to the
``AbstractKernelFunction`` interface that our sub-class represents
a differentiable kernel. This is done by setting the flag
``HAS_FIRST_PARAMETER_DERIVATIVE`` in the constructor::

	MySequenceKernel(double gamma)
	: m_gamma(gamma)
	{
	    SHARK_ASSERT(m_gamma > 0.0);
	    this->m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
	}

The derivative values need to be made available to the gradient-based
optimizer through a unified interface. For kernels this is achieved
by overriding the ``weightedParameterDerivative`` function::

	virtual void weightedParameterDerivative(
	        ConstBatchInputReference batchX1,
	        ConstBatchInputReference batchX2,
	        RealMatrix const& coefficients,
	        State const& state,
	        RealVector& gradient
	    );

This function takes five arguments. The first two are the already
familiar data batches, and the fourth is a state object that has been
passed earlier to the ``eval`` function **with the exact same batches**.
Thus, this object can store intermediate values and thus speed up the
computation of the derivative.

If you are completely unfamiliar with the role of a state object in
derivative computations then please read
:doc:`../optimization/conventions_derivatives` before continuing here.

Looking at the above formula, it is easy to see that the derivative is a
cheap by-product of the evaluation of the exponential, at the cost of an
additional multiplication. This hints at the possibility to make
efficient use of the state object. Although this may seem like a very
lucky coincidence it is not; such synergies between computation of the
value and its derivatives are extremely common.

In principle there are different possibilities for implementing this
derivative. The simplest is to ignore possible synergy effects and the
state object completely and to compute the derivative from scratch. This
is very inefficient, since it is obviously possible to reuse some
intermediate values. On the other hand one should avoid using massive
storage for intermediates, since then the runtime could become dominated
by limited memory throughput.

Before deciding what to store in the state object let's look at the
computation the function is required to perform. The gradient vector is
to be filled in with the partial derivatives of the weighted sum of all
kernel values w.r.t. the parameters. In pseudo code the computation reads:

``gradient(p) = \sum_{i,j} coefficient(i, j)``
:math:`\frac{\partial}{\partial \text{parameter}(p)}` ``k(batchX1(i), batchX2(j))``

Precomputing a matrix of entry-wise kernel derivatives (little
computational overhead during evaluation, rather small storage) seems
like a reasonable compromise between computing everything from scratch
(no storage, highly redundant computations for derivatives) and storing
all exponential function evaluations (no additional computation time
during evaluation, but huge storage). A good rule of thumb is that
storing at most a hand full of values per pair of inputs is okay.
Extremely costly to compute kernels may of course prefer to store more
intermediate information. In doubt, there is no way around benchmarking
different versions of the code.

Putting everything together our implementation looks like this::

	struct InternalState : public State {
	    RealMatrix dk_dgamma;   // derivative of kernel k w.r.t. gamma
	};

	std::shared_ptr<State> createState() const {
	    return std::shared_ptr<State>(new InternalState());
	}

	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result, State& state) const {
	    std::size_t s1 = size(batchX1);
	    std::size_t s2 = size(batchX2);
	    result.resize(s1, s2);

	    // prepare state
	    InternalState& s = state.toState<InternalState>();
	    s.dk_dgamma.resize(s1, s2);

	    for (std::size_t i=0; i<s1; i++) {
	        ConstInputReference x_i = get(batchX1, i);
	        for (std::size_t j=0; j<s2; j++) {
	            ConstInputReference y_j = get(batchX2, j);

	            // evaluate k(x_i, y_j)
	            std::size_t N = y_j.size();       // string length
	            SHARK_ASSERT(x_i.size() == N);    // DEBUG check
	            double sum = 0.0;
	            double derivative = 0.0;
	            for (std::size_t p=0; p<N; p++) {
	                for (std::size_t q=0; q<N; q++) {
	                    if (x_i[p] == y_j[q]) {
	                        int d = -((p-q) * (p-q));
	                        double e = std::exp(m_gamma * d);
	                        sum += e;
	                        derivative += d * e;
	                    }
	                }
	            }

	            // fill result matrix and state
	            result(i, j) = sum;
	            s.dk_dgamma(i, j) = derivative;
	        }
	    }
	}

With all derivatives readily computed in the state object the
implementation of the weighted parameter derivative becomes a piece of
cake::

	void weightedParameterDerivative(
	        ConstBatchInputReference batchX1, 
	        ConstBatchInputReference batchX2, 
	        RealMatrix const& coefficients,
	        State const& state, 
	        RealVector& gradient
	    ) const
	{
	    std::size_t s1 = size(batchX1);
	    std::size_t s2 = size(batchX2);
	    InternalState const& s = state.toState<InternalState>();

	    // debug checks
	    SIZE_CHECK(s1 == s.dk_dgamma.size1());
	    SIZE_CHECK(s2 == s.dk_dgamma.size2());

	    // compute weihted sum
	    double sum = 0.0;
	    for (std::size_t i=0; i<s1; i++) {
	        for (std::size_t j=0; j<s2; j++) {
	            sum += coefficients(i, j) * s.dk_dgamma(i, j);
	        }
	    }

	    // return gradient
	    gradient.resize(1);
	    gradient(0) = sum;
	}

Now our evaluation function is a bit more costly than necessary,
provided that we may not always need the derivative. Therefore the
AbstractKernelFunction interface defines one more version of the
``eval`` function, namely without state object::

	void eval(
	        ConstBatchInputReference batchX1,
	        ConstBatchInputReference batchX2,
	        RealMatrix& result
	    ) const

The default implementation creates a state object, calls the pure
virtual evaluation interface, and discards the state. Here we have the
opportunity to reuse our first version of the evaluation code. This
leaves us with an efficient interface for evaluations only and also for
derivative computations.


Input Derivatives
-----------------

Kernels can be defined on arbitrary input spaces, and in the example of
strings we can see that not all of these input spaces are equipped with
a differentiable structure. However, vector spaces are an important
special case. Therefore, the ``AbstractKernelFunction`` interface
provides an optional interface for computing the derivative of the
kernel value with respect to (vector valued) inputs. Therefore we will
now switch to an example with differentiable inputs, for which we pick
``GaussianRbfKernel<RealVector>``. This class computes the kernel

.. math::
	k(x, x') = \exp \Big( -\gamma \|x-x'\|^2 \Big)

with :math:`x` and :math:`x'` represented by ``RealVector`` objects.
Then we can ask how the kernel value varies with :math:`x`:

.. math::
	\frac{\partial k(x, x')}{\partial x} = -2 \|x-x'\|^2 k(x, x') (x-x')

There is no special function for the derivative w.r.t. :math:`x'` because kernels
are symmetric functions and the roles of the arguments can be switched.

The ``AbstractKernelFunction`` super class provides the following
interface::

	void weightedInputDerivative( 
	        ConstBatchInputReference batchX1, 
	        ConstBatchInputReference batchX2, 
	        RealMatrix const& coefficientsX2,
	        State const& state,
	        BatchInputType& gradient
	    );

Again, batches of inputs are evaluated, a matrix of coefficients and a
state object are involved. The gradient is represented by a
``BatchInputType``: technically, the tangent space of the vector space
is identified with the vector space itself (by means of the standard
inner product), and the same data type can be used. If you have no idea
what this math stuff is all about, sit back and simply imagine gradients
as vectors in the input vector space.

Since the function returns a batch of gradient, one for each point in
the first batch, the question is what the coefficients mean. The function
is supposed to compute the following vector:

.. math::
	\begin{pmatrix}
		c_{1,1} \frac{\partial k(x_1, x'_1)}{\partial x_1} + \dots + c_{1,m} \frac{\partial k(x_1, x'_m)}{\partial x_1} \\
		\vdots \\
		c_{n,1} \frac{\partial k(x_n, x'_1)}{\partial x_n} + \dots + c_{n,m} \frac{\partial k(x_n, x'_m)}{\partial x_n} \\
	\end{pmatrix}

The ``InternalState`` structure of the ``GaussianRbfKernel`` class
contains two matrices holding the terms :math:`\|x-x'\|^2` and
:math:`k(x, x')`::

	struct InternalState {
	    RealMatrix norm2;
	    RealMatrix expNorm;
	    ...
	};

With this information we can implement the above formulas into the
weighted input derivative computation::

	void weightedInputDerivative(
	        ConstBatchInputReference batchX1,
	        ConstBatchInputReference batchX2,
	        RealMatrix const& coefficientsX2,
	        State const& state,
	        BatchInputType& gradient
	    ) const
	{
	    std::size_t s1 = size(batchX1);
	    std::size_t s2 = size(batchX2);
	    InternalState const& s = state.toState<InternalState>();

	    gradient.resize(s1, batchX1.size2());   // batch type is a RealMatrix
	    gradient.clear();
	    for (std::size_t i=0; i<s1; i++) {
	        for (std::size_t j=0; j < s2; j++) {
	            noalias(row(gradient, i))
	                    += (coefficientsX2(i, j) * s.expNorm(i, j))
	                    * (row(batchX2, j) - row(batchX1, i));
	        }
	    }
	    gradient *= 2.0 * m_gamma;
	}

Note that this function relies on the same state object that is also
used by the weighted parameter derivative. Thus, the state information
needs to be shared between both functions, which is actually reasonable,
since the terms that can be reused are often very similar. However,
depending on the particular case this may add a new twist to the
consideration which terms to store in the state object.


Normalized Kernels
------------------

Some kernels are *normalized*, meaning that they fulfill
:math:`k(x, x) = 1` for all x. Gaussian kernels are a prominent example.
This property simplifies some computations, such as distances in feature
space:

.. math::
	d \big( \phi_k(x), \phi_k(y) \big) = \sqrt{k(x, x) - 2k(x, y) + k(y, y)} = \sqrt{2 - 2k(x, y)}

Shark profits from such optimized computations if the flag
``IS_NORMALIZED`` is set in the constructor.


References
----------


.. [Meinicke2004] Meinicke, P., Tech, M., Morgenstern, B., Merkl, R.: Oligo kernels for datamining on biological sequences: A case study on prokaryotic translation initiation sites. BMC Bioinformatics 5, 2004.

.. [Cristianini2002] Nello Cristianini, Jaz Kandola, Andre Elisseeff, John Shawe-Taylor: On kernel-target alignment. Advances in Neural Information Processing Systems 14, 2002.

.. [Igel2007] C. Igel, T. Glasmachers, B. Mersch, N. Pfeifer, P. Meinicke. Gradient-Based Optimization of Kernel-Target Alignment for Sequence Kernels Applied to Bacterial Gene Start Detection. IEEE/ACM Transactions on Computational Biology and Bioinformatics (TCBB), 4(2):216-226, 2007.
