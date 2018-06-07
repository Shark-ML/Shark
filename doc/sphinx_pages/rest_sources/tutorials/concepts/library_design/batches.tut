Data Batches
============


Modern computer architectures have grown more complex due to the fact
that the speed of modern CPUs improves much faster than the memory access latency
of the RAM. To achieve the highest performance for numerical data processing, data
points need to be grouped into smaller subsets which we call batches. A batch can be
described as a two dimensional structure. The first dimension, which we call "row",
represents the different data points, and the second dimension, which we refer to as "column",
represents the components of the data space or structure.

Ideally, after creation of the batches, data should be placed dense in the RAM such that
the CPU can evaluate the whole batch at once. A typical example of such a structure is a
matrix when each single data point is a vector. These single vectors are copied into rows
of the matrix to form a batch.

A typical example for an algorithm which hugely benefits
from this new grouping is a linear model without offset. For single vectors x, the
evaluation of such a linear model would simply be written as

.. math::

     f(x)=Ax \enspace .

Here, the computation is a matrix-vector product. Let now X be the matrix holding a
batch of n vectors, for all of which we want to compute the response of the linear
model. Then, when the data are stored in batches and our model can operate directly
on batches, the batch algorithm can be written as

.. math::

     f(X)=AX^T \enspace .

Such a transition to aggregate data structures can yield a speedup of at least
a factor of two when ATLAS is enabled as linear algebra backend, and potentially even
beyond an order of magnitude, depending on the dimensions of A and X. The reason is
that the matrix-matrix multiplication can be computed much more memory friendly than
a series of matrix-vector products. Small blocks of memory can be reused such that the
CPU can hold these parts in its cache. This reduces the number of memory lookups needed
from the relatively slow RAM.




More on batches in Shark
------------------------



Usage of batches in Shark is quite simple in most cases. As long as only standard types are
used as inputs, the :doxy:`Data` class automatically creates efficient batches using a default
size. At the current state of the library, not all algorithms and models make fullest use of
the batch interface -- but in such cases, safe fallbacks to single element methods are used.

The mapping between points and batches is quite simple in most cases. Here is a small list:


===================   ========================
Point type            Batch type
===================   ========================
double                RealVector
int                   IntVector
...                   ...
RealVector            RealMatrix
CompressedIntVector   CompressedIntMatrix
...                   ...
T                     std::vector<T>
===================   ========================


T here stands for an arbitrary type. So the default case for non-standard points is
``std::vector<T>``, which enables Shark to generate batches for all types. In other words,
Shark's batch interface works seamlessly with strings and graphs, and whatever data
type a user may need.

As you can see, the choice of batches is quite convenient for most cases. Single values
are stored in vectors and vectors in matrices. When you write your own programs you can
be assured that the batch types are exactly these and do not need to bother about these types.




Element Access
--------------


When you want to access the i-th element of the matrix, you can write ``row(batch,i)``, or
to query the size use ``batch.size1()``. For vectors you can use ``batch(i)`` and ``batch.size()``.
But what  happens when a more general algorithm, like for example the error function, is to be
implemented? In this case you do not know which functions or methods to use, since the types
shown above have totally different interfaces. Shark circumvents this problem by adapting
and extending the interface of boost.range:


===================   =================================================================
Function              Meaning
===================   =================================================================
boost::begin(batch)   returns an iterator to the beginning of the range of elements
boost::end(batch)     returns an iterator to the end of the range of elements
boost::size(batch)    returns the number of elements in the batch
shark::get(batch,i)   returns a reference to the i-th element of the batch
===================   =================================================================

For typical containers which already support ``batch.begin()``, ``batch.end()`` and ``batch.size()``,
the default implementation provided by Boost is sufficient. For the ublas matrices, Shark provides
reasonably implemented iterators.


.. warning:

    The rest of the tutorial is outdated/wrong/subject to change.

The Batch<T> Traits class
-------------------------


Suppose your data points have an arbitrary type T. There are a few things that you want to know:

- What is the Batch type of T and how can it be created?
- How many elements does the batch have?
- How can single elements be accessed?

Typically T cannot answer this question by itself, since we cannot change its definition.
Even if we could change it for some, it clearly is not possible for basic types like int or
double. Therefore we need an external class that explicitly represents this information
for use at compile-time. For Shark batches, this class is the ``Batch<T>`` class template.
It is a traits class, meaning that it tells you something about a type, in this case T
and its batch type. Let's take a closer look at the basic definition of Batch, for
now in ``include/shark/Data/BatchInterface.h``::

    template<class T>
    struct Batch{
        typedef implementation-specific-type type;            //type of a batch
        typedef implementation-specific-type reference;       //reference to an element of the batch
        typedef implementation-specific-type const_reference; //const_reference to an element of the batch
        typedef implementation-specific-type iterator;        //iterator over all elements of the batch
        typedef implementation-specific-type const_iterator;  //const_iterator over all elements of the batch

        static type createBatch(T const& input, std::size_t size = 1);
    };

We introduce class usage in a step-by-step example -- improving it as we go
along until we do not need to know the type of points any more. We begin with
short example code where the point is a vector and the batch is a matrix::

    RealVector point(10);
    RealMatrix batchOfPoints(5,10);
    row(batchOfPoints, 0) = point;
    std::cout << batchOfPoints.size1();

Let's answer the first question: how to query the type of a batch?
This is easy using ``Batch<T>::type``::

    RealVector point(10);
    Batch<RealVector>::type batchOfPoints(5,10);
    row(batchOfPoints, 0) = point;
    std::cout << batchOfPoints.size1();

Not bad. Still, we explicitly use that batch is a Matrix by calling
its size1 member. That's fine, as long as we know that T can only be a vector. But sometimes
we do not even know that, typically in generic code. Let's begin improving it, by changing the
element access as well as the size query. For the first, we use ``get``, and for the second,
``size``, both from the previous section::

    RealVector point(10);
    Batch<RealVector>::type batchOfPoints(5,10);
    get(batchOfPoints, 0) = point;
    std::cout << size(batchOfPoints);

Now the last thing missing is the creation of the batch. We always need an element to create
a batch from it. It serves as blueprint ensuring that the batch can store the elements. For
example in the case of RealVector, it ensures that the matrix has the same amount of columns
as the vector has dimensions. In this case, we use the point available::

    RealVector point(10);
    Batch<RealVector>::type batchOfPoints = Batch<RealVector>::createBatch(point, 5);
    get(batchOfPoints, 0) = point;
    std::cout << size(batchOfPoints);

While this surely looks more difficult than the first version, it is also completely type
independent. Note that even when creating batches of size 1, it is still necessary to
assign the point after batch initialization (as in line three of the above snippet).
Otherwise, the batches point would be uninitialized.



Adapting Batch<T> for a user defined structure
----------------------------------------------


Let's face it: even though we like to pretend that all our data points are vectors,
we often only make our data look like it when in fact it is a collection of different
types -- for example mixing reals, categorical data and sparse binary features with
strings of varying lengths, or even graphs. We often have routines that produce these
features and then spend a lot of time writing code that creates data vectors out of
the points. Sometimes this is exactly what we want (for example when the model is a
neural net which does not have a sense of data structure at all). But for more
specialized methods we might want to represent this structure explicitly. At the
same time, we want the efficiency of a good data representation in batches. So how do
we solve it? By creating a batch type and specializing ``Batch<T>`` on it!

Let's start with a simple data point::

    struct Point{
        RealVector feature1;
        SparseIntVector feature2;
    };

There is an easy automatic way to create a sufficient specialisation of batch for ``Point`` in Shark using the macro SHARK_CREATE_BATCH_INTERFACE::

    #define PointVars (feature1)(feature2)
    #define PointTypes (RealVector)(SparseIntVector)
    struct Batch< Point >{
        SHARK_CREATE_BATCH_INTERFACE( Point,
            (RealVector, feature1),(SparseIntVector, feature2))
    };

This also works when Point is templatized, for example like this::

    template<class Type1, class Type2>
    struct Point{
        RealVector feature1;
        SparseIntVector feature2;
    };

.. todo::

   TG: should it be Type1 feature1; Type2 feature2; ???
   mt: i second this question

In this case, we have to add the template parameters to the Batch specialisation::

    #define PointVars (feature1)(feature2)
    #define PointTypes (Type1)(Type2)
    #define PointName Point<Type1,Type2>
    template<class Type1,class Type2>
    struct Batch< Point >{
        SHARK_CREATE_BATCH_INTERFACE( PointName, PointVars, PointTypes )
    };
    #undef PointVars
    #undef PointTypes

.. todo::

    mt: undef pointname also?

You see how the convenience macro enables the definition of Shark batch types
for arbitrary data structures without much trouble. Recall that this is only
necessary for user defined non-vectorial data formats. The standard cases, such
as stacking vectors into matrices, are already covered by the Shark library itself.
