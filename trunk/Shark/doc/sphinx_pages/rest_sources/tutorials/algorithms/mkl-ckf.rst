==========================
Linear Kernel Combinations
==========================

This tutorial first lists some background information on Multiple Kernel
Learning (MKL) algorithms and Linear Kernel Combinations (LKCs). In the second part, we
start with the actual, hands-on Shark-Library tutorial code introductions. This includes
a tour of the different kernel functions which might be handy, as well as the MKL-typical
kernel normalization techniques.

Shark does currently not include a "canonical" MKL algorithm that
optimizes the kernel weights and the parameters of am SVM kernel
expansion jointly. Rather, it offers a kernel function class which is
called :doxy:`MklKernel` and conforms to the general class of kernels
used in MKL algorithms. However, the weights still have to be learned
by some method implemented in Shark for optimizing single-kernel
parameters.

MKL and LKCs: Background
------------------------

In recent years, so-called Multiple Kernel Learning (MKL) algorithms for
SVMs have become fashionable; see for example [Gonen2011]_ for a review.
That line of research has at its core the idea of using, instead of one
single kernel, a convex linear combination of base kernels as a compound
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
kernel weights is referred to as MKL, but merely employing an LKC is not. The most prominent
school of MKL algorithms share the following characteristics:

* The kernel weights :math:`\theta` are optimized together with the SVM weight vector
  :math:`\alpha` (or :math:`w` ) in one single, joint optimization problem. See
  [Kloft2011]_ for details.

* The sub-kernels :math:`k_i` are usually regarded as (pseudo-)parameterless. In other
  words, if the sub-kernels do have parameters, these are fixed to one particular value
  and not optimized over.

Since the problem of learning the kernel weights is integrated into the main SVM problem
such that it remains convex, proponents of MKL have argued that MKL offers a convincing
way of "learning a kernel": the weights are guaranteed to reach the global optimum to the
optimization problem.

A second school of MKL learning algorithms employ a two-stage process. First, the kernel
weights are optimized, for example, using the kernel-target alignment as a criterion. In
the second step, the full SVM is then trained as usual with fixed kernel weights.

In practice and for many applications, the experimental results of a wide range of MKL
algorithms have proven not very convincing [Gonen2011]_. Regardless of the kernel weight
optimization strategy used and its respective success, it is still important to note the
two main motivations for using LKCs in a learning task:

* In the first scenario, each kernel operates on the exact same set of features.
  That is, the input to each sub-kernel :math:`k_i` is the same as the one to the
  "mother" kernel :math:`k`. The sub-kernels may then either stem from different
  kernel function families, or basically be the same mathematical function but
  with different parameters. The most popular scenario or argument made for such
  a setting is that this way, each sub-kernel can be viewed as a candidate kernel
  for solving the problem at hand. Instead of selecting the kernel family type and/or
  the sub-kernels' parameters in a traditional grid-search (or other hyperparameter
  optimization) setting, MKL algorithms can "choose" their favorite kernel, and thus
  the best sub-kernel parameter, themselves by increasing the weights :math:`\theta_i`
  for all sub-kernels :math:`k_i` which have a meaningful sub-kernel parameter. This
  is sometimes seen by MKL proponents as eliminating or circumventing the SVM model
  selection problem. We will refer to this first scenario as the MKL kernel selection
  scenario.

* In the second scenario, each kernel operates on a different sub-range of the
  input feature vectors. This is frequently desired when the feature vector is a
  concatenation of data obtained through different methods or reflecting different
  properties of the examples. For example, in image processing and computer vision
  it is common practice to concatenate a color histogram and a histogram of gradients,
  etc. Another typical application domain is biological data, where many different
  ways to characterize or measure the properties of a molecule are conceivable.
  We will refer to this second scenario as the MKL information integration scenario.

Of course, hybrid scenarios, combining both of the above approaches, are possible as
well.


MKL and LKCs in Shark
---------------------

The :doxy:`WeightedSumKernel`
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

For the MKL kernel selection scenario, Shark features two Kernel function classes
which allow positive linear combinations of arbitrary sub-kernels: the :doxy:`WeightedSumKernel`
class implements a kernel

.. math::

    k(x,z) = \frac{\sum_i \theta_i k_i(x,z)}{\sum_i \theta_i} \; ,

where the first kernel weight is always fixed to one, :math:`\theta_0 = 0`, thus eliminating
one redundant degree of freedom. Also, the kernel result is always normalized by the sum of
the sub-kernel weights. Internally, the kernel weights are represented as exponentials of the
externally visible parameters to ensure positivity. To declare such a kernel, we include the
corresponding header file, as well as that of our future sub-kernel, and declare our usual
namespace directives::

    #include <shark/Models/Kernels/WeightedSumKernel.h>
    #include <shark/Models/Kernels/GaussianRbfKernel.h>
    using namespace shark;
    using namespace std;

We next set up our set of base kernels (here restricting ourselves to two in total)::

    DenseRbfKernel baseKernel1( 0.1 );
    DenseRbfKernel baseKernel2( 0.01 );
    std::vector< DenseKernelFunction* > kernels;
    kernels.push_back(&baseKernel1);
    kernels.push_back(&baseKernel2);
    DenseWeightedSumKernel kernel(kernels);

where DenseRbfKernel and DenseKernelFunction are shorthand typedefs defaulting to ``RealVector``
kernel input types. This is all needed to know to get started using the LKCs -- with maybe one additional
detail: the :doxy:`WeightedSumKernel` offers three additional methods to set or view the sub-kernel's
behavior with respect to the integration of their sub-parameters into the overall parameter vector::

    void setAdaptive( std::size_t index, bool b = true ){...}
    void setAdaptiveAll( bool b = true ) {...}
    bool isAdaptive( std::size_t index ) const {...}

By default, the sub-kernels contribution to the overall parameter vector is turned **off**, that is,
the only parameters initially visible are the :math:`N-1` last kernel weights (the first one being
fixed to one). We can illustrate this and the three above methods' behaviors by continuing with our
above example::

    cout << endl << "kernel.isAdaptive(0): " << kernel.isAdaptive(0) << endl;
    cout << "kernel.isAdaptive(1): " << kernel.isAdaptive(1) << endl;
    cout << "kernel.numberOfParameters(): " << kernel.numberOfParameters() << endl;
    cout << "kernel.parameterVector(): " << kernel.parameterVector() << endl;
    cout << "kernel.eval(x1,x2): " << kernel.eval(x1,x2) << endl << endl;
    RealVector new_params( kernel.numberOfParameters() );
    new_params(0) = 1.0;
    kernel.setParameterVector( new_params );
    cout << "kernel.parameterVector() with 1st parameter set to 1: " << kernel.parameterVector() << endl;
    cout << "kernel.eval(x1,x2): " << kernel.eval(x1,x2) << endl << endl;

    kernel.setAdaptive(0,true);

    cout << "kernel.isAdaptive(0): " << kernel.isAdaptive(0) << endl;
    cout << "kernel.isAdaptive(1): " << kernel.isAdaptive(1) << endl;
    cout << "kernel.numberOfParameters(): " << kernel.numberOfParameters() << endl;
    cout << "kernel.parameterVector(): " << kernel.parameterVector() << endl<< endl;

    kernel.setAdaptive(0,false);
    kernel.setAdaptive(1,true);

    cout << "kernel.isAdaptive(0): " << kernel.isAdaptive(0) << endl;
    cout << "kernel.isAdaptive(1): " << kernel.isAdaptive(1) << endl;
    cout << "kernel.numberOfParameters(): " << kernel.numberOfParameters() << endl;
    cout << "kernel.parameterVector(): " << kernel.parameterVector() << endl<< endl;

    kernel.setAdaptiveAll(true);

    cout << "kernel.isAdaptive(0): " << kernel.isAdaptive(0) << endl;
    cout << "kernel.isAdaptive(1): " << kernel.isAdaptive(1) << endl;
    cout << "kernel.numberOfParameters(): " << kernel.numberOfParameters() << endl;
    cout << "kernel.parameterVector(): " << kernel.parameterVector() << endl;
    cout << "kernel.eval(x1,x2): " << kernel.eval(x1,x2) << endl << endl;

The output of this should be::

    kernel.isAdaptive(0): 0
    kernel.isAdaptive(1): 0
    kernel.numberOfParameters(): 1
    kernel.parameterVector(): [1](0)
    kernel.eval(x1,x2): 0.52702

    kernel.parameterVector() with 1st parameter set to 1: [1](1)
    kernel.eval(x1,x2): 0.677265

    kernel.isAdaptive(0): 1
    kernel.isAdaptive(1): 0
    kernel.numberOfParameters(): 2
    kernel.parameterVector(): [2](1,0.1)

    kernel.isAdaptive(0): 0
    kernel.isAdaptive(1): 1
    kernel.numberOfParameters(): 2
    kernel.parameterVector(): [2](1,0.01)

    kernel.isAdaptive(0): 1
    kernel.isAdaptive(1): 1
    kernel.numberOfParameters(): 3
    kernel.parameterVector(): [3](1,0.1,0.01)
    kernel.eval(x1,x2): 0.677265

The above should make clear how the adaptiveness of the sub-kernels controls the visibility of
their sub-parameters to all other Shark methods, for example external parameter optimization routines.
Also, we can see that the output corresponds exactly to the result of the computations we would
expect:

.. math::

    ( 1.0*\exp(-0.1*16) + 1.0*\exp(-0.01*16) ) / ( 1.0 + 1.0 ) = 0.527020 \\
    ( 1.0*\exp(-0.1*16) + e*\exp(-0.01*16) ) / ( 1.0 + e ) = 0.677265 \; .



The :doxy:`MklKernel`
&&&&&&&&&&&&&&&&&&&&&

The :doxy:`MklKernel` class is again basically identical to the :doxy:`WeightedSumKernel` class,
with however one additional capability, which tailors this kernel class to the aforementioned
"information integration scenario". While in the "kernel selection scenario", each sub-kernel
operates on the entire, full feature vector, in the "information integration scenario", each
sub-kernel only operates on a sub-set of the feature vector:

.. math::

    k(x,z) = \sum_i k_i(x_{b_{i}-e_{i}},z_{b_{i}-e_{i}})

where the index range :math:`b_{i}-e_{i}` denotes the :math:`i` -th sub-range (inclusive beginning to
exclusive end) of the overall feature vector. Naturally, we need to pass these index pairs to
the :doxy:`MklKernel` for each sub-kernel. This is done during construction::

    #include <shark/Models/Kernels/MklKernel.h>

    GaussianRbfKernel<ConstRealVectorRange> baseKernel1(0.1);
    DenseRbfMklKernel baseKernel2(0.01); //two equivalent ways of declaring a DenseRbfMklKernel, see typedefs in MklKernel.h
    std::vector<AbstractKernelFunction<ConstRealVectorRange>* > kernels;
    kernels.push_back(&baseKernel1);
    kernels.push_back(&baseKernel2);

    std::vector< std::pair< std::size_t, std::size_t > > frs;
    frs.push_back( std::make_pair( 0,2 ) );
    frs.push_back( std::make_pair( 0,2 ) );
    DenseMklKernel kernel( kernels, frs );

The last four lines illustrate how to construct the vector of index pairs denoting the beginning
and end indices for each sub-kernel. Here, we have for starters chosen to let both kernels treat
all features. In effect, this is equivalent to the :doxy:`WeightedSumKernel`, and the program
output illustrates this::

    kernel.isAdaptive(0): 0
    kernel.isAdaptive(1): 0
    kernel.numberOfParameters(): 1
    kernel.parameterVector(): [1](0)
    kernel.eval(sub1,sub2): 0.52702

    kernel.parameterVector() with 1st parameter set to 1: [1](1)
    kernel.eval(sub1,sub2): 0.677265

    kernel.isAdaptive(0): 1
    kernel.isAdaptive(1): 0
    kernel.numberOfParameters(): 2
    kernel.parameterVector(): [2](1,0.1)

    kernel.isAdaptive(0): 0
    kernel.isAdaptive(1): 1
    kernel.numberOfParameters(): 2
    kernel.parameterVector(): [2](1,0.01)

    kernel.isAdaptive(0): 1
    kernel.isAdaptive(1): 1
    kernel.numberOfParameters(): 3
    kernel.parameterVector(): [3](1,0.1,0.01)
    kernel.eval(sub1,sub2): 0.677265

Now we repeat the above scenario again, however with each sub-kernel operating on different feature ranges::

    GaussianRbfKernel<ConstRealVectorRange> baseKernel1(0.1);
    DenseRbfMklKernel baseKernel2(0.01); //two equivalent ways of declaring a DenseRbfMklKernel, see typedefs in MklKernel.h
    std::vector<AbstractKernelFunction<ConstRealVectorRange>* > kernels;
    kernels.push_back(&baseKernel1);
    kernels.push_back(&baseKernel2);

    std::vector< std::pair< std::size_t, std::size_t > > frs;
    frs.push_back( std::make_pair( 0,1 ) );
    frs.push_back( std::make_pair( 1,2 ) );
    DenseMklKernel kernel( kernels, frs );

We would now expect as outcome of the first kernel computation:

.. math::

    ( 1.0*\exp(-0.1*16) + 1.0*\exp(-0.01*0) ) / ( 1.0 + 1.0 ) = 0.600948

and, when setting the second weight to 1:

.. math::

    ( 1.0*\exp(-0.1*16) + e*\exp(-0.01*0) ) / ( 1.0 + e ) = 0.785357

Both values are exactly what we get from the code output::

    kernel.isAdaptive(0): 0
    kernel.isAdaptive(1): 0
    kernel.numberOfParameters(): 1
    kernel.parameterVector(): [1](0)
    kernel.eval(sub1,sub2): 0.600948

    kernel.parameterVector() with 1st parameter set to 1: [1](1)
    kernel.eval(sub1,sub2): 0.785357

    kernel.isAdaptive(0): 1
    kernel.isAdaptive(1): 0
    kernel.numberOfParameters(): 2
    kernel.parameterVector(): [2](1,0.1)

    kernel.isAdaptive(0): 0
    kernel.isAdaptive(1): 1
    kernel.numberOfParameters(): 2
    kernel.parameterVector(): [2](1,0.01)

    kernel.isAdaptive(0): 1
    kernel.isAdaptive(1): 1
    kernel.numberOfParameters(): 3
    kernel.parameterVector(): [3](1,0.1,0.01)
    kernel.eval(sub1,sub2): 0.785357



MKL Kernel Normalization
&&&&&&&&&&&&&&&&&&&&&&&&


Since many MKL formulations penalize the (:math:`l_p`-) norm of the kernel weights, the
optimization objective could always be improved by substituting a kernel in question for
a multiple of itself. The canonical MKL formulations hence rely on normalization of the
data to unit interval in feature space. Although Shark does not currently offer a canonical
MKL SVM algorithm, we provide a trainer for "multiplicative normalization" of a :doxy:`MklKernel`
function (see [Kloft2011]_). In detail, we provide a :doxy:`ScaledKernel` which wraps an
existing kernel, multiplying it by a fixed constant. The :doxy:`NormalizeKernelUnitVariance`
class is a trainer which serves to set the scaling factor of the :doxy:`ScaledKernel`. In this
example section, we show how to normalize the kernel to unit variance in feature space. ::

    #include <shark/Algorithms/Trainers/NormalizeKernelUnitVariance.h>

    std::size_t num_dims = 9;
    std::size_t num_points = 200;
    std::vector<RealVector> input(num_points);
    RealVector v(num_dims);
    for ( std::size_t i=0; i<num_points; i++ ) {
        for ( std::size_t j=0; j<num_dims; j++ ) {
            v(j) = Rng::uni(-1,1);
        }
        input[i] = v;
    }
    UnlabeledData<RealVector> data(input);


Here we first included the header file for the kernel normalizer, and then set up
a dataset of 200 9-dimensional samples with random content. As in the examples above,
we next declare an :doxy:`MklKernel`  from several member sub-kernels::

    DenseRbfMklKernel         basekernel1(0.1);
    DenseLinearMklKernel      basekernel2;
    DensePolynomialMklKernel  basekernel3(2, 1.0);

    std::vector< DenseMklKernelFunction * > kernels;
    kernels.push_back(&basekernel1);
    kernels.push_back(&basekernel2);
    kernels.push_back(&basekernel3);

    std::vector< std::pair< std::size_t, std::size_t > > frs;
    frs.push_back( std::make_pair( 0,3 ) );
    frs.push_back( std::make_pair( 3,6 ) );
    frs.push_back( std::make_pair( 6,9 ) );

    DenseMklKernel kernel( kernels, frs );

From the :doxy:`MklKernel` , we declare a :doxy:`ScaledKernel`, which we then
normalize on the given dataset using a :doxy:`NormalizeKernelUnitVariance` trainer::

    DenseScaledKernel scale( &kernel );
    NormalizeKernelUnitVariance<> normalizer;
    normalizer.train( scale, data );

Note that the kernel does not know about the dataset, but is influenced by it
indirectly through the trainer. And, already, we're done. Next we can examine
the specifics of the normalization process, and re-calculate the kernel's
variance after normalization by hand to verify that it indeed is equal to 1.0::

    std::cout << "    Done training. Factor is " << scale.factor() << std::endl;
    std::cout << "    Mean                   = " << normalizer.mean() << std::endl;
    std::cout << "    Trace                  = " << normalizer.trace() << std::endl << std::endl;
    //check in feature space
    double control = 0.0;
    for ( std::size_t i=0; i<num_points; i++ ) {
        control += scale.eval(input[i], input[i]);
        for ( std::size_t j=0; j<num_points; j++ ) {
            control -= scale.eval(input[i],input[j]) / num_points;
        }
    }
    control /= num_points;
    std::cout << "    Variance of scaled MklKernel: " << control << std::endl << std::endl;

This should result in the following output:

    Done training. Factor is 0.71872
    Mean                   = 29476.4
    Trace                  = 425.654

    Variance of scaled MklKernel: 1


Tutorial source code
&&&&&&&&&&&&&&&&&&&&

You can find the aggregated version of this tutorial's code in ``examples/Supervised/MklKernelTutorial.cpp``.



References
----------

.. [Gonen2011] M. Gönen, E. Alpaydin: Multiple Kernel Learning Algorithms. Journal of Machine Learning Research 12, 2011.

.. [Kloft2011] M. Kloft, U. Brefeld, S. Sonnenburg, A. Zien: :math:`l_p`-Norm Multiple Kernel Learning. Journal of Machine Learning Research 12, 2011.




..

    mt: The FullyWeightedSumKernel was apparently removed from the code for some reason.


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
        ( e*\exp(-0.1*16) + 1.0*\exp(-0.01*16) ) / ( e + 1.0 ) = 0.376775 \\
        ( 1.0*\exp(-0.1*16) + e*\exp(-0.01*16) ) / ( 1.0 + e ) = 0.677265 \; .
