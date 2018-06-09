
Linear Kernel Combinations (and a bit of MKL)
=============================================


This tutorial first lists some background information on Multiple Kernel
Learning (MKL) algorithms and Linear Kernel Combinations (LKCs). In the
second part, we start with the actual, hands-on Shark tutorial. This
includes a tour of the different kernel functions which might be handy,
as well as the MKL-typical kernel normalization techniques.

Shark does not currently include a "canonical" MKL algorithm that
optimizes the kernel weights and the parameters of an SVM kernel
expansion jointly. Rather, it offers three kernel classes generally
used in MKL algorithms. The most general of these is the :doxy:`MklKernel`,
which adds up sub-kernels that can operate on completely different input
types. The :doxy:`SubrangeKernel` lets the sub-kernels operate on different
index ranges of the same input vector. Finally, the :doxy:`WeightedSumKernel`
simply passes its inputs to all sub-kernels unchanged.

What you then do with these kernels, i.e., how you choose their weights
and train the resulting SVM is so far up to you.



MKL and LKCs: Background
------------------------


In recent years, so-called Multiple Kernel Learning (MKL) algorithms for
SVMs have become fashionable; see for example [Gonen2011]_ for a review.
That line of research has at its core the idea of using, instead of one
single kernel, a (convex) linear combination of base kernels as a compound
kernel within an SVM.

In more detail: if :math:`k_1` and :math:`k_2` are positive definite kernel
functions, then

.. math::

    k = a \cdot k_1 + b \cdot k_2

is again a positive definite kernel for :math:`a \geq 0` and :math:`b \geq 0`
(and :math:`k \neq 0`). Most of the MKL literature is based on linear kernel
combinations (LKCs) of the general form

.. math::

    k = \sum_i \theta_i k_i \; , \\
    \theta_i \geq 0

Multiple Kernel Learning then refers to learning the positive, real-valued weight vector
:math:`\theta`. Often, an additional constraint is enforced on the kernel weights, for
example :math:`\|\theta\|^2 \leq 1`. It should be noted that the process of learning the
kernel weights (and possibly including training the SVM at the same time) is referred to
as MKL, but merely employing an LKC is not. The most prominent school of MKL algorithms
share the following characteristics:

* The kernel weights :math:`\theta` are optimized together with the SVM weight vector
  :math:`\alpha` (or :math:`w` ) in one single, joint optimization problem. See
  [Kloft2011]_ or [Gonen2011]_.

* The sub-kernels :math:`k_i` are usually regarded as (pseudo-)parameterless. In other
  words, if the sub-kernels do have parameters, these are fixed to one particular value
  and not optimized.

Since learning the kernel weights is integrated into the (modified) main SVM problem
such that it remains convex, proponents of MKL argue that MKL is a convincing way of
"learning a kernel": the weights are guaranteed to reach "the" global optimum to the
optimization problem. On the other hand, just because a problem is convex does not
mean that it is helpful and/or that the base kernels were selected in a helpful way.

A second school of MKL algorithms employ a two-stage process. First, the kernel weights
are optimized using, for example, the kernel-target alignment as a criterion. In the
second step, the full SVM is then trained as usual with fixed kernel weights.

In practice and for many applications, the experimental results of a wide range of MKL
algorithms have proven not very convincing [Gonen2011]_.

Regardless of the kernel weight optimization strategy used and its respective success,
it is instructive to recall the two (different) main motivations for using LKCs in a
learning task:

* In the first scenario, each kernel operates on the exact same set of features.
  That is, the input to each sub-kernel :math:`k_i` is the same as the one to the
  "mother" kernel :math:`k`. The sub-kernels may then either stem from different
  kernel function families, or basically be the same mathematical function but
  with different parameters. The most popular scenario or argument made for such
  a setting is that this way, each sub-kernel can be viewed as a candidate kernel
  for solving the problem at hand. Instead of selecting the kernel family type and/or
  the sub-kernels' parameters in a traditional grid-search (or other hyperparameter
  optimization) setting, MKL algorithms can "choose" their favorite kernel themselves,
  and thus the best sub-kernel parameter, by increasing the weights :math:`\theta_i`
  for all sub-kernels :math:`k_i` which have a meaningful sub-kernel parameter. This
  is sometimes seen by MKL proponents as eliminating or circumventing the SVM model
  selection problem. We will refer to this first scenario as the MKL kernel selection
  scenario.

* In the second scenario, each kernel operates on a different sub-range of the
  input feature vectors. This is for example desired when the feature vector is a
  concatenation of data obtained through different methods, or reflecting different
  properties of the samples. For example, in image processing and computer vision,
  it is common practice to concatenate a color histogram and a histogram of gradients,
  etc. Another typical application domain is biological data, where many different
  ways to characterize or measure the properties of a molecule are conceivable.
  We will refer to this second scenario as the MKL information integration scenario.

Of course, hybrid scenarios, combining both of the above approaches, are conceiveable.



MKL and LKCs in Shark
---------------------

Shark offers three classes which allow for positive linear combinations of
sub-kernels: The :doxy:`WeightedSumKernel`, the :doxy:`SubrangeKernel`, and
the :doxy:`MklKernel`. All three will be described below. But here in short:
in the WeightedSumKernel, the same input gets passed to all sub-kernels. With
the SubrangeKernel, each sub-kernel operates on a certain index range of an
input vector. The MklKernel allows the sub-kernels to be completely heterogenous
(e.g., one operating on a custom data structure and one on a RealVector).

Throughout the tutorial, we will use the following includes and namespaces::

    
	#include <shark/Data/Dataset.h>
	#include <shark/Core/Random.h>
	#include <shark/Algorithms/Trainers/NormalizeKernelUnitVariance.h>
	#include <shark/Models/Kernels/GaussianRbfKernel.h>
	#include <shark/Models/Kernels/WeightedSumKernel.h>
	#include <shark/Models/Kernels/SubrangeKernel.h>
	#include <shark/Models/Kernels/MklKernel.h>
	#include <shark/Models/Kernels/LinearKernel.h>
	#include <shark/Models/Kernels/DiscreteKernel.h>
	#include <shark/Models/Kernels/PolynomialKernel.h>
	#include <boost/fusion/algorithm/iteration/fold.hpp>
	#include <boost/fusion/include/as_vector.hpp>
	
	using namespace shark;
	using namespace std;
	

and (almost everywhere) two two-dimensional test points like so::

    
	    // test points
	    RealVector x1(2);
	    x1(0)=2;
	    x1(1)=1;
	    RealVector x2(2);
	    x2(0)=-2;
	    x2(1)=1;
	    



The :doxy:`WeightedSumKernel`
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


The :doxy:`WeightedSumKernel` class implements a kernel

.. math::

    k(x,z) = \frac{\sum_i \theta_i k_i(x,z)}{\sum_i \theta_i} \;

with the first kernel weight always fixed to one (eliminating one redundant
degree of freedom). The denominator serves to normalize the kernel by the
sum of the sub-kernel weights. Note that internally, **the kernel weights
are computed as exponentials of the externally visible parameters**. To be
clear: in other words, when you set a parameter vector, it will only affect
the :math:`N-1` last kernel weights (the first one being fixed to one [or
zero, in parameter space]), and the weights will be the exponentials
of what you passed as parameter vector. The latter is done to support
unconstrained optimization (no matter what parameters you set, you always
get positive weights). We next set up two base kernels like so::

    
	    // initialize kernels
	    DenseRbfKernel baseKernel1( 0.1 );
	    DenseRbfKernel baseKernel2( 0.01 );
	    std::vector< AbstractKernelFunction<RealVector> * > kernels1;
	    kernels1.push_back( &baseKernel1 );
	    kernels1.push_back( &baseKernel2 );
	    DenseWeightedSumKernel kernel1( kernels1 );
	    

where DenseRbfKernel is a shorthand typedef for an template input type of
RealVector. This is all needed to know to get started -- with maybe one
addition: the :doxy:`WeightedSumKernel` (and in fact, all three LKC kernels
presented in this tutorial) offer three further methods, :doxy:`setAdaptive`,
:doxy:`setAdaptiveAll`, and :doxy:`isAdaptive`. These set or show whether a
sub-kernel's sub-parameters are part of the overall parameter vector::

    void setAdaptive( std::size_t index, bool b = true ){...}
    void setAdaptiveAll( bool b = true ) {...}
    bool isAdaptive( std::size_t index ) const {...}

By default, the sub-kernels contribution to the overall parameter vector is turned
**off**. That is, the only parameters initially visible are the :math:`N-1` last
kernel weights (the first one being fixed to one [or zero, in parameter space]).
Lines of code say it best::

    
	    // examine initial state
	    std::cout << endl << " ======================= WeightedSumKernel: ======================= " << std::endl;
	    cout << endl << "kernel1.isAdaptive(0): " << kernel1.isAdaptive(0) << endl;
	    cout << "kernel1.isAdaptive(1): " << kernel1.isAdaptive(1) << endl;
	    cout << "kernel1.numberOfParameters(): " << kernel1.numberOfParameters() << endl;
	    cout << "kernel1.parameterVector(): " << kernel1.parameterVector() << endl;
	    cout << "kernel1.eval(x1,x2): " << kernel1.eval(x1,x2) << endl << endl;
	    
    
	    // change something
	    RealVector new_params_1( kernel1.numberOfParameters() );
	    new_params_1(0) = 1.0;
	    kernel1.setParameterVector( new_params_1 );
	    
    
	    // examine again
	    cout << "kernel1.parameterVector() with 1st parameter set to 1: " << kernel1.parameterVector() << endl;
	    cout << "kernel1.eval(x1,x2): " << kernel1.eval(x1,x2) << endl << endl;
	    
    
	    // change something else
	    kernel1.setAdaptive(0,true);
	    
    
	    // examine once more
	    cout << "kernel1.isAdaptive(0): " << kernel1.isAdaptive(0) << endl;
	    cout << "kernel1.isAdaptive(1): " << kernel1.isAdaptive(1) << endl;
	    cout << "kernel1.numberOfParameters(): " << kernel1.numberOfParameters() << endl;
	    cout << "kernel1.parameterVector(): " << kernel1.parameterVector() << endl<< endl;
	    
    
	    // another change
	    kernel1.setAdaptive(0,false);
	    kernel1.setAdaptive(1,true);
	    
    
	    // examining again
	    cout << "kernel1.isAdaptive(0): " << kernel1.isAdaptive(0) << endl;
	    cout << "kernel1.isAdaptive(1): " << kernel1.isAdaptive(1) << endl;
	    cout << "kernel1.numberOfParameters(): " << kernel1.numberOfParameters() << endl;
	    cout << "kernel1.parameterVector(): " << kernel1.parameterVector() << endl<< endl;
	    
    
	    // last change
	    kernel1.setAdaptiveAll(true);
	    
    
	    // last examination
	    cout << "kernel1.isAdaptive(0): " << kernel1.isAdaptive(0) << endl;
	    cout << "kernel1.isAdaptive(1): " << kernel1.isAdaptive(1) << endl;
	    cout << "kernel1.numberOfParameters(): " << kernel1.numberOfParameters() << endl;
	    cout << "kernel1.parameterVector(): " << kernel1.parameterVector() << endl;
	    cout << "kernel1.eval(x1,x2): " << kernel1.eval(x1,x2) << endl << endl;
	    


The output of this should be:

.. code-block:: none

    kernel1.isAdaptive(0): 0
    kernel1.isAdaptive(1): 0
    kernel1.numberOfParameters(): 1
    kernel1.parameterVector(): [1](0)
    kernel1.eval(x1,x2): 0.52702

    kernel1.parameterVector() with 1st parameter set to 1: [1](1)
    kernel1.eval(x1,x2): 0.677265

    kernel1.isAdaptive(0): 1
    kernel1.isAdaptive(1): 0
    kernel1.numberOfParameters(): 2
    kernel1.parameterVector(): [2](1,0.1)

    kernel1.isAdaptive(0): 0
    kernel1.isAdaptive(1): 1
    kernel1.numberOfParameters(): 2
    kernel1.parameterVector(): [2](1,0.01)

    kernel1.isAdaptive(0): 1
    kernel1.isAdaptive(1): 1
    kernel1.numberOfParameters(): 3
    kernel1.parameterVector(): [3](1,0.1,0.01)
    kernel1.eval(x1,x2): 0.677265

The kernel evaluations yield exactly what we would expect:

.. math::

    \frac{( 1.0*\exp(-0.1*16) + 1.0*\exp(-0.01*16) ) }{ ( 1.0 + 1.0 )} = 0.527020 \\
    \frac{( 1.0*\exp(-0.1*16) + e*\exp(-0.01*16) ) }{ ( 1.0 + e )} = 0.677265 \; .

The above should also make clear how the sub-kernels' sub-parameters are "seen"
by other Shark algorithms, for example during external parameter optimization.




The :doxy:`SubrangeKernel`
&&&&&&&&&&&&&&&&&&&&&&&&&&

The second LKC class is the :doxy:`SubrangeKernel`. This is similar to the
:doxy:`WeightedSumKernel`, but tailored to the above mentioned "information
integration scenario". Before, in the "kernel selection scenario", each
sub-kernel operated on the entire, full feature vector. In the "information
integration scenario", each sub-kernel only operates on a continuous sub-set
of the feature vector:

.. math::

    k(x,z) = \frac{ \sum_i \theta_i k_i(x_{b_{i}-e_{i}},z_{b_{i}-e_{i}}) } { \sum_i \theta_i } \, .


The index range :math:`b_{i}-e_{i}` denotes the :math:`i`-th continuous
sub-range (inclusive beginning to exclusive end) of the overall feature vector.
Naturally, we need to pass these index pairs to the SubrangeKernel for each
sub-kernel. This is done during construction. First, we set up the sub-kernels
as before::

    
	    DenseRbfKernel baseKernel3(0.1);
	    DenseRbfKernel baseKernel4(0.01);
	    std::vector<AbstractKernelFunction<RealVector>* > kernels2;
	    kernels2.push_back(&baseKernel3);
	    kernels2.push_back(&baseKernel4);
	    

Next, we set up a vector of index pairs for the begin- and end-indices for
each sub-kernel. The SubrangeKernel itself is constructed by passing one
vector of kernels and one of indices::

    
	    std::vector< std::pair< std::size_t, std::size_t > > indcs_1;
	    indcs_1.push_back( std::make_pair( 0,2 ) );
	    indcs_1.push_back( std::make_pair( 0,2 ) );
	    DenseSubrangeKernel kernel2( kernels2, indcs_1 );
	    

In fact, the SubrangeKernel inherits from the WeightedSumKernel. Thus, besides
the constructor, the interfaces are identical. For starters, we let both kernels
treat all features. This is equivalent to the :doxy:`WeightedSumKernel` example
above, as shown by the corresponding commands::

    
	    // examine initial state
	    std::cout << endl << " ======================= SubrangeKernel, full index range: ======================= " << std::endl;
	    cout << endl << "kernel2.isAdaptive(0): " << kernel2.isAdaptive(0) << endl;
	    cout << "kernel2.isAdaptive(1): " << kernel2.isAdaptive(1) << endl;
	    cout << "kernel2.numberOfParameters(): " << kernel2.numberOfParameters() << endl;
	    cout << "kernel2.parameterVector(): " << kernel2.parameterVector() << endl;
	    cout << "kernel2.eval(x1,x2): " << kernel2.eval(x1,x2) << endl << endl;
	    
    
	    // change something
	    RealVector new_params_2( kernel2.numberOfParameters() );
	    new_params_2(0) = 1.0;
	    kernel2.setParameterVector( new_params_2 );
	    
    
	    // examine again
	    cout << "kernel2.parameterVector() with 1st parameter set to 1: " << kernel2.parameterVector() << endl;
	    cout << "kernel2.eval(x1,x2): " << kernel2.eval(x1,x2) << endl << endl;
	    
    
	    // change something else
	    kernel2.setAdaptive(0,true);
	    
    
	    // examine once more
	    cout << "kernel2.isAdaptive(0): " << kernel2.isAdaptive(0) << endl;
	    cout << "kernel2.isAdaptive(1): " << kernel2.isAdaptive(1) << endl;
	    cout << "kernel2.numberOfParameters(): " << kernel2.numberOfParameters() << endl;
	    cout << "kernel2.parameterVector(): " << kernel2.parameterVector() << endl<< endl;
	    
    
	    // another change
	    kernel2.setAdaptive(0,false);
	    kernel2.setAdaptive(1,true);
	    
    
	    // examining again
	    cout << "kernel2.isAdaptive(0): " << kernel2.isAdaptive(0) << endl;
	    cout << "kernel2.isAdaptive(1): " << kernel2.isAdaptive(1) << endl;
	    cout << "kernel2.numberOfParameters(): " << kernel2.numberOfParameters() << endl;
	    cout << "kernel2.parameterVector(): " << kernel2.parameterVector() << endl<< endl;
	    
    
	    // last change
	    kernel2.setAdaptiveAll(true);
	    
    
	    // last examination
	    cout << "kernel2.isAdaptive(0): " << kernel2.isAdaptive(0) << endl;
	    cout << "kernel2.isAdaptive(1): " << kernel2.isAdaptive(1) << endl;
	    cout << "kernel2.numberOfParameters(): " << kernel2.numberOfParameters() << endl;
	    cout << "kernel2.parameterVector(): " << kernel2.parameterVector() << endl;
	    cout << "kernel2.eval(x1,x2): " << kernel2.eval(x1,x2) << endl << endl;
	    

and its resulting output:

.. code-block:: none

    kernel2.isAdaptive(0): 0
    kernel2.isAdaptive(1): 0
    kernel2.numberOfParameters(): 1
    kernel2.parameterVector(): [1](0)
    kernel2.eval(x1,x2): 0.52702

    kernel2.parameterVector() with 1st parameter set to 1: [1](1)
    kernel2.eval(x1,x2): 0.677265

    kernel2.isAdaptive(0): 1
    kernel2.isAdaptive(1): 0
    kernel2.numberOfParameters(): 2
    kernel2.parameterVector(): [2](1,0.1)

    kernel2.isAdaptive(0): 0
    kernel2.isAdaptive(1): 1
    kernel2.numberOfParameters(): 2
    kernel2.parameterVector(): [2](1,0.01)

    kernel2.isAdaptive(0): 1
    kernel2.isAdaptive(1): 1
    kernel2.numberOfParameters(): 3
    kernel2.parameterVector(): [3](1,0.1,0.01)
    kernel2.eval(x1,x2): 0.677265


Now we repeat the above scenario again, however with each sub-kernel operating
on different feature ranges. Setting up the kernels and indices...::

    
	    DenseRbfKernel baseKernel5(0.1);
	    DenseRbfKernel baseKernel6(0.01);
	    std::vector<AbstractKernelFunction<RealVector>* > kernels3;
	    kernels3.push_back(&baseKernel5);
	    kernels3.push_back(&baseKernel6);
	    
    
	    std::vector< std::pair< std::size_t, std::size_t > > indcs_2;
	    indcs_2.push_back( std::make_pair( 0,1 ) );
	    indcs_2.push_back( std::make_pair( 1,2 ) );
	    DenseSubrangeKernel kernel3( kernels3, indcs_2 );
	    

... and again issuing the familiar commands::

    
	    // examine initial state
	    std::cout << endl << " ======================= SubrangeKernel partial index range: ======================= " << std::endl;
	    cout << endl << "kernel3.isAdaptive(0): " << kernel3.isAdaptive(0) << endl;
	    cout << "kernel3.isAdaptive(1): " << kernel3.isAdaptive(1) << endl;
	    cout << "kernel3.numberOfParameters(): " << kernel3.numberOfParameters() << endl;
	    cout << "kernel3.parameterVector(): " << kernel3.parameterVector() << endl;
	    cout << "kernel3.eval(x1,x2): " << kernel3.eval(x1,x2) << endl << endl;
	    
    
	    // change something
	    RealVector new_params_3( kernel3.numberOfParameters() );
	    new_params_3(0) = 1.0;
	    kernel3.setParameterVector( new_params_3 );
	    
    
	    // examine again
	    cout << "kernel3.parameterVector() with 1st parameter set to 1: " << kernel3.parameterVector() << endl;
	    cout << "kernel3.eval(x1,x2): " << kernel3.eval(x1,x2) << endl << endl;
	    
    
	    // change something else
	    kernel3.setAdaptive(0,true);
	    
    
	    // examine once more
	    cout << "kernel3.isAdaptive(0): " << kernel3.isAdaptive(0) << endl;
	    cout << "kernel3.isAdaptive(1): " << kernel3.isAdaptive(1) << endl;
	    cout << "kernel3.numberOfParameters(): " << kernel3.numberOfParameters() << endl;
	    cout << "kernel3.parameterVector(): " << kernel3.parameterVector() << endl<< endl;
	    
    
	    // another change
	    kernel3.setAdaptive(0,false);
	    kernel3.setAdaptive(1,true);
	    
    
	    // examining again
	    cout << "kernel3.isAdaptive(0): " << kernel3.isAdaptive(0) << endl;
	    cout << "kernel3.isAdaptive(1): " << kernel3.isAdaptive(1) << endl;
	    cout << "kernel3.numberOfParameters(): " << kernel3.numberOfParameters() << endl;
	    cout << "kernel3.parameterVector(): " << kernel3.parameterVector() << endl<< endl;
	    
    
	    // last change
	    kernel3.setAdaptiveAll(true);
	    
    
	    // last examination
	    cout << "kernel3.isAdaptive(0): " << kernel3.isAdaptive(0) << endl;
	    cout << "kernel3.isAdaptive(1): " << kernel3.isAdaptive(1) << endl;
	    cout << "kernel3.numberOfParameters(): " << kernel3.numberOfParameters() << endl;
	    cout << "kernel3.parameterVector(): " << kernel3.parameterVector() << endl;
	    cout << "kernel3.eval(x1,x2): " << kernel3.eval(x1,x2) << endl << endl;
	    


We would now expect as outcome of the kernel computations:

.. math::

    \frac{( 1.0*\exp(-0.1*16) + 1.0*\exp(-0.01*0) )}{( 1.0 + 1.0 )} = 0.600948 \\
    \frac{( 1.0*\exp(-0.1*16) + e*\exp(-0.01*0) )} {( 1.0 + e )} = 0.785357

Both values are exactly what we get from the code output:

.. code-block:: none

    kernel3.isAdaptive(0): 0
    kernel3.isAdaptive(1): 0
    kernel3.numberOfParameters(): 1
    kernel3.parameterVector(): [1](0)
    kernel3.eval(x1,x2): 0.600948

    kernel3.parameterVector() with 1st parameter set to 1: [1](1)
    kernel3.eval(x1,x2): 0.785357

    kernel3.isAdaptive(0): 1
    kernel3.isAdaptive(1): 0
    kernel3.numberOfParameters(): 2
    kernel3.parameterVector(): [2](1,0.1)

    kernel3.isAdaptive(0): 0
    kernel3.isAdaptive(1): 1
    kernel3.numberOfParameters(): 2
    kernel3.parameterVector(): [2](1,0.01)

    kernel3.isAdaptive(0): 1
    kernel3.isAdaptive(1): 1
    kernel3.numberOfParameters(): 3
    kernel3.parameterVector(): [3](1,0.1,0.01)
    kernel3.eval(x1,x2): 0.785357



The :doxy:`MklKernel`
&&&&&&&&&&&&&&&&&&&&&


The third class is the :doxy:`MklKernel`. It is similar to the WeightedSumKernel
and the SubrangeKernel, except that it adds up kernels operating on possibly
completely different inputs:

.. math::

    k(x,z) = \frac{ \sum_i \theta_i k_i(x_i,z_i) } { \sum_i \theta_i } \, .


That is, :math:`x_i` and :math:`x_j` (and hence :math:`k_i` and :math:`k_j`)
are allowed to have very different structure (rather than merely being different
subranges of the same input vector). The MklKernel thus allows for the most
diverse information integration settings. This flexibility comes at a small
price of added usage code complexity.

First, there is the question what data type the aggregated tuple of sub-inputs
:math:`x=(x_0,x_1,...)` should have. Shark currently supports binding macros for
arbitrary structures to boost::fusion. For most purposes, it is easiest to declare a 
struct as a composite data type and then adapt it for boost::fusion, like so::

    
	    struct HeterogeneousInputStruct{
	        shark::RealVector rv1;
	        std::size_t st2;
	        shark::RealVector crv3;
	    };
	
	    #ifndef DOXYGEN_SHOULD_SKIP_THIS
	        BOOST_FUSION_ADAPT_STRUCT(
	            HeterogeneousInputStruct,
	            (shark::RealVector, rv1)(std::size_t, st2)(shark::RealVector, crv3)
	        )
	    #endif /* DOXYGEN_SHOULD_SKIP_THIS */
	
	    namespace shark{
	        template<>
	        struct Batch< HeterogeneousInputStruct >{
	            SHARK_CREATE_BATCH_INTERFACE_NO_TPL(
	                HeterogeneousInputStruct,
	                (shark::RealVector, rv1)(std::size_t, st2)(shark::RealVector, crv3)
	            )
	        };
	    }
	    

Here, the first block declares the structure itself. The second block tells
boost::fusion that the struct can be seen or treated as a tuple. The third
block tells Shark to create a suitable batch structure (cf.
:doc:`the Batch tutorial <../concepts/library_design/batches>`) for it. (Side
note: the code is in the beginning of the overall tutorial .cpp file because
the two macros require to be called at global scope.)

Now that we created and announced the data structure, we fill it with data::

    
	    // set dimensions for data
	    std::size_t const num_samples = 2;
	    std::size_t const dim_nonzeros = 2;
	    std::size_t const max_elem_discr_kernel = 3;
	    std::size_t const dim_sparse = 5;
	    // create temporary helper container
	    std::vector<HeterogeneousInputStruct> data( num_samples );
	    // and fill it
	    data[0].rv1.resize( dim_nonzeros ); data[0].crv3.resize( dim_sparse); //size 5
	    data[1].rv1.resize( dim_nonzeros ); data[1].crv3.resize( dim_sparse); //size 5
	    data[0].rv1(0) = 1.0; data[0].rv1(1) = -1.0; data[0].crv3(1) = -0.5; data[0].crv3(4) = 8.0;
	    data[1].rv1(0) = 1.0; data[1].rv1(1) = -2.0; data[1].crv3(1) =  1.0; data[1].crv3(3) = 0.1;
	    data[0].st2 = 1; data[1].st2 = 2;
	    // and use it to create the 'real' dataset
	    Data<HeterogeneousInputStruct> dataset = createDataFromRange( data, 10 );
	    

Next, we create all sub-kernels and the overall MklKernel::

    
	    //create state matrix for the discrete kernel. necessary but not so relevant
	    RealMatrix matK( max_elem_discr_kernel, max_elem_discr_kernel );
	    matK(0,0) = 0.05; matK(1,1) = 1.0;  matK(2,2) = 0.5;
	    matK(0,1) = matK(1,0) = 0.2; matK(0,2) = matK(2,0) = 0.4;  matK(1,2) = matK(2,1) = 0.6;
	    // set up base kernels
	    DenseRbfKernel baseKernelRV1(0.1);
	    DiscreteKernel baseKernelST2(matK);
	    DenseLinearKernel baseKernelCRV3;
	    MklKernel<HeterogeneousInputStruct> mkl_kernel( boost::fusion::make_vector( &baseKernelRV1, &baseKernelST2, &baseKernelCRV3) );
	    

The first three lines provide a state matrix for the discrete kernel (basically
a look-up matrix). The second three set up the three base kernels as usual.
The last line finally creates the MklKernel via yet another boost::fusion
command.

We now again examine the MklKernel's state after creation::

    
	    // examine initial state
	    std::cout << endl << " ======================= MklKernel: ======================= " << std::endl;
	    cout << endl << "mkl_kernel.isAdaptive(0): " << mkl_kernel.isAdaptive(0) << endl;
	    cout << "mkl_kernel.isAdaptive(1): " << mkl_kernel.isAdaptive(1) << endl;
	    cout << "mkl_kernel.isAdaptive(2): " << mkl_kernel.isAdaptive(2) << endl;
	    cout << "mkl_kernel.numberOfParameters(): " << mkl_kernel.numberOfParameters() << endl;
	    cout << "mkl_kernel.parameterVector(): " << mkl_kernel.parameterVector() << endl;
	    cout << "mkl_kernel.eval( dataset.element(0), dataset.element(1) ): " << mkl_kernel.eval( dataset.element(0), dataset.element(1) ) << endl << endl;
	    

It behaves similar to what we saw from the previous kernels. Next we
make all sub-parameters (i.e., the RbfKernel's bandwidth) adaptive and
change two parameters::

    
	    // change something
	    mkl_kernel.setAdaptiveAll(true);
	    RealVector new_params_4( mkl_kernel.numberOfParameters() );
	    new_params_4(0) = 1.0;
	    new_params_4(2) = 0.2;
	    mkl_kernel.setParameterVector( new_params_4 );
	    

Code to examine the outcome::

    
	    // examine effects
	    cout << "mkl_kernel.isAdaptive(0): " << mkl_kernel.isAdaptive(0) << endl;
	    cout << "mkl_kernel.isAdaptive(1): " << mkl_kernel.isAdaptive(1) << endl;
	    cout << "mkl_kernel.isAdaptive(2): " << mkl_kernel.isAdaptive(2) << endl;
	    cout << "mkl_kernel.numberOfParameters(): " << mkl_kernel.numberOfParameters() << endl;
	    cout << "mkl_kernel.parameterVector(): " << mkl_kernel.parameterVector() << endl;
	    cout << "mkl_kernel.eval( dataset.element(0), dataset.element(1) ): " << mkl_kernel.eval( dataset.element(0), dataset.element(1) ) << endl << endl;
	    

We would expect the kernel evaluations to yield:

.. math::

    \frac{( 1.0*\exp(-0.1*1.0) + 1.0*0.6 + 1.0*(-0.5*1.0) )} {( 1.0 + 1.0 + 1.0 )} = 0.334946 \\
    \frac{( 1.0*\exp(-0.2*1.0) + e*0.6 + 1.0*(-0.5*1.0) )} {( 1.0 + e + 1.0   )} = 0.413222

Both values are exactly what we get from the code's output:

.. code-block:: none

    mkl_kernel.isAdaptive(0): 0
    mkl_kernel.isAdaptive(1): 0
    mkl_kernel.isAdaptive(2): 0
    mkl_kernel.numberOfParameters(): 2
    mkl_kernel.parameterVector(): [2](0,0)
    mkl_kernel.eval( dataset.element(0), dataset.element(1) ): 0.334946

    mkl_kernel.isAdaptive(0): 1
    mkl_kernel.isAdaptive(1): 1
    mkl_kernel.isAdaptive(2): 1
    mkl_kernel.numberOfParameters(): 3
    mkl_kernel.parameterVector(): [3](1,0,0.2)
    mkl_kernel.eval( dataset.element(0), dataset.element(1) ): 0.413222



MKL Kernel Normalization
&&&&&&&&&&&&&&&&&&&&&&&&


Since many MKL formulations penalize the (:math:`l_p`-) norm of the kernel weights,
the optimization objective could always be improved by substituting the base kernels
for a common multiple of themselves. For this reason, the (:math:`l_p`-) norm is
usually constrained to a certain value or value range. Similarly, rescaling of
individual kernels (as opposed to changing their associated kernel weight) can
influence the solution found by MKL algorithms. Canonical MKL formulations hence rely
on normalization of the kernel (or data) to unit interval in feature space. Although
Shark does not currently offer a canonical MKL SVM algorithm, we provide a trainer
for "multiplicative normalization" of a :doxy:`MklKernel` function (see [Kloft2011]_).

In detail, the :doxy:`ScaledKernel` wraps an existing kernel to multiply it by a
fixed constant. The :doxy:`NormalizeKernelUnitVariance` class is a Trainer which
initializes this scaling factor of the :doxy:`ScaledKernel`. To normalize a kernel
to unit variance in feature space, we first create and fill an example dataset of
200 9-dimensional samples with random content::

    
	    std::size_t num_dims = 9;
	    std::size_t num_points = 200;
	    std::vector<RealVector> input(num_points);
	    RealVector v(num_dims);
	    for ( std::size_t i=0; i<num_points; i++ ) {
	        for ( std::size_t j=0; j<num_dims; j++ )
	            v(j) = random::uni(random::globalRng, -1,1);
	        input[i] = v;
	    }
	    UnlabeledData<RealVector> rand_data = createDataFromRange( input );
	    

Now let's say we have the following three member kernels and want to build an LKC
from them::

    
	    // declare kernels
	    DenseRbfKernel         unnormalized_kernel1(0.1);
	    DenseLinearKernel      unnormalized_kernel2;
	    DensePolynomialKernel  unnormalized_kernel3(2, 1.0);
	    // declare indices
	    std::vector< std::pair< std::size_t, std::size_t > > indices;
	    indices.push_back( std::make_pair( 0,3 ) );
	    indices.push_back( std::make_pair( 3,6 ) );
	    indices.push_back( std::make_pair( 6,9 ) );
	    

From the first kernel, we declare a :doxy:`ScaledKernel`, which we then
normalize on the given dataset using a :doxy:`NormalizeKernelUnitVariance`
trainer::

    
	    DenseScaledKernel scale( &unnormalized_kernel3 );
	    NormalizeKernelUnitVariance<> normalizer;
	    normalizer.train( scale, rand_data );
	    

Note that the kernel does not know about the dataset, but is influenced by it
indirectly through the trainer. Now we're done. We finally examine the results
from the scaled kernel and trainer, and also re-calculate the kernel's variance
after normalization by hand to verify that it indeed is equal to 1.0::

    
	    std::cout << endl << " ======================= Kernel normalization: ======================= " << std::endl;
	
	    std::cout << endl << "Done training. Factor is " << scale.factor() << std::endl;
	    std::cout << "Mean                   = " << normalizer.mean() << std::endl;
	    std::cout << "Trace                  = " << normalizer.trace() << std::endl << std::endl;
	    //check in feature space
	    double control = 0.0;
	    for ( std::size_t i=0; i<num_points; i++ ) {
	        control += scale.eval(input[i], input[i]);
	        for ( std::size_t j=0; j<num_points; j++ ) {
	            control -= scale.eval(input[i],input[j]) / num_points;
	        }
	    }
	    control /= num_points;
	    std::cout << "Resulting variance of scaled Kernel: " << control << std::endl << std::endl;
	    

This will result in output similar to the following (the first three lines
may vary due to the randomized dataset):

.. code-block:: none

    Done training. Factor is 0.0677846
    Mean                   = 83774.4
    Trace                  = 3369.39

    Resulting variance of scaled Kernel: 1

In the same way, we could also normalize the other two sub-kernels to unit variance
in feature space. Then, we could correct the following code snippet to build a
SubrangeKernel from three properly normalized kernels::

    
	    std::vector<AbstractKernelFunction<RealVector>* > kernels4;
	    kernels4.push_back( &unnormalized_kernel1 );
	    kernels4.push_back( &unnormalized_kernel2 );
	    kernels4.push_back( &scale );
	    DenseSubrangeKernel kernel4( kernels4, indices );
	    


Tutorial source code
&&&&&&&&&&&&&&&&&&&&

You can find the aggregated version of this tutorial's code in
``examples/Supervised/MklKernelTutorial.cpp`` (as generated from
its according .tpp file).



References
----------

.. [Gonen2011] M. Gönen, E. Alpaydin: Multiple Kernel Learning Algorithms. Journal of Machine Learning Research 12, 2011.

.. [Kloft2011] M. Kloft, U. Brefeld, S. Sonnenburg, A. Zien: :math:`l_p`-Norm Multiple Kernel Learning. Journal of Machine Learning Research 12, 2011.




..
    mt: The FullyWeightedSumKernel was apparently removed from the code for some reason.
        Archived below..


..
    The :doxy:`FullyWeightedSumKernel`
    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    The :doxy:`FullyWeightedSumKernel` is almost identical to the :doxy:`WeightedSumKernel`, with the
    exception that in the former, the weight of the first sub-kernel is also a kernel parameter by
    default. Thus, there is one redundant scaling degree of freedom, but this might be desired in rare cases.
    If we execute the same code block as above, this time we get the following result::

        kernel.isAdaptive(0): 0
        kernel.isAdaptive(1): 0
        kernel.numberOfParameters(): 2
        kernel.parameterVector(): [2](0,0)
        kernel.eval(x1,x2): 0.52702

        kernel.parameterVector() with 1st parameter set to 1: [2](1,0)
        kernel.eval(x1,x2): 0.376775
        kernel.parameterVector() with 2nd parameter set to 1: [2](0,1)
        kernel.eval(x1,x2): 0.677265

        kernel.isAdaptive(0): 1
        kernel.isAdaptive(1): 0
        kernel.numberOfParameters(): 3
        kernel.parameterVector(): [3](0,1,0.1)

        kernel.isAdaptive(0): 0
        kernel.isAdaptive(1): 1
        kernel.numberOfParameters(): 3
        kernel.parameterVector(): [3](0,1,0.01)

        kernel.isAdaptive(0): 1
        kernel.isAdaptive(1): 1
        kernel.numberOfParameters(): 4
        kernel.parameterVector(): [4](0,1,0.1,0.01)
        kernel.eval(x1,x2): 0.677265

    And again, this matches our mathematical expectations:

    .. math::

        ( 1.0*\exp(-0.1*16) + 1.0*\exp(-0.01*16) ) / ( 1.0 + 1.0 ) = 0.527020 \\
        ( 1.0*\exp(-0.1*16) + e*\exp(-0.01*16) ) / ( 1.0 + e ) = 0.677265 \; .


