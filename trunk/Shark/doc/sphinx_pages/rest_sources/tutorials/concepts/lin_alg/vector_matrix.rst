

LinAlg: Vector and Matrices
===========================



All Linear Algebra related functions and operations are placed in the
LinAlg Module. LinAlg is based on the boost linear algebra system
`uBLAS <http://www.boost.org/doc/libs/release/libs/numeric>`_. This tutorial
will give a short introduction to the library and how it is used in Shark.



uBLAS: Background in a nutshell
-------------------------------


When one looks at the history of BLAS (basic linear algebra libraries), the
C and Fortran versions were the most popular for medium to large scale problems.
They are very fast and offer a wide range of functions. The downside is that
writing code using these libraries is very difficult because the function names
are cryptic such as ''dgemm'' (Double precision GEneral Matrix Matrix multiplication)
and code maintenance needs a lot of experience.  C++ Libraries solve the latter
problem using operator overloading and make it possible to transfer equations
like the following 1:1 into the source code:

.. math::
   z = (A^T+B)(\vec{x}-\vec{y})

This is a lot easier to maintain and change. But these notations have the disadvantage
of typically translating into very slow code, because they tend to store intermediate
results, which could easily be avoided. Boost uBlas tries to solve this problem and
ranges in performance and usability between the previous described approaches.
The above code using Shark typedefs and uBLAS looks like: ::

    RealVector z = inner_prod(trans(A)+B,x-y);

Using proper optimization modes, the compiler will generate the following pseudo-code from the above equations: ::

    for(unsigned i = 0; i != B.size1(); ++i)
    {
        z(i) = 0;
        for(unsigned j = 0; j != x.size(); ++j){
            z(i) += (A(j,i)+B(i,j)) * (x(i) - y(i));
        }
    }

Which is pretty fast. Of course, the performance depends on the
compiler (e.g., the icc may generate faster code than the gcc).  But
still, uBLAS will not outperform standard C-libraries which use extremely
optimized code. And so a matrix-matrix multiplication using other BLAS
libraries such as MKL can be around 50 times faster for big matrices.
Interestingly, however, these issues can be fixed since uBLAS is very
flexible.




uBLAS: Basic usage
------------------


If you like, you can risk a general quick glance over the matrix and
vector types which uBLAS provides
`here <http://www.boost.org/doc/libs/1_52_0/libs/numeric/ublas/doc/types_overview.htm>`_.
The most prominent of these are probably ``boost::numeric::ublas::vector<double>``
and ``boost::numeric::ublas::matrix<double>``. In general, and as you can already
see, uBLAS is heavily template oriented, and using these types directly
leads to very complex code. Therefore, Shark offers a few typedefs for
the most common used types. Also, it provides the nested namespace
``shark::blas``, which is a shortcut for ``boost::numeric::ublas``.
The most important types are XXXVector, XXXMatrix, CompressedXXXVector,
CompressedXXXMatrix, MappedXXXVector, and MappedXXXMatrix. Here, XXX stands
for the element type of the Object: Real(``double``), Float(``float``),
Complex(``std::complex<double>``), Int(``int``) and UInt(``unsigned``). The
Compressed and Mapped versions are sparse, so a ``MappedRealVector`` is a
sparse vector with elements of type double. These Shark typedefs originate
from macros in the file ``include/shark/LinAlg/BLAS/VectorMatrixType.h``,
which you can consult for detailed information.

.. caution::
  There is no default initialization for ublas vectors and matrices.


The uBLAS documentation has a good overview for the
`basic operations <http://www.boost.org/doc/libs/1_44_0/libs/numeric/ublas/doc/operations_overview.htm>`_.
In the following, only a few important points are shown.
Single elements of a vector can be accessed by using operator(): ::

    z(i) = 5;

This also works for matrices, only that you need two parameters. Matrices are formated in the row major format, so ::

    A(i,j)

will access the element in the i-th row and j-th column. Matrices and vectors offer +,- for vector/matrix addition and * for scalar multiplication.
For matrix-matrix multiplication prod() has to be used. But be aware, the following line will not work: ::

    A = prod(prod(B,C),D);

uBLAS prevents this, because nested matrix products are slow when computed without intermediate results. We have to calculate the first matrix
and save the intermediate result before we can compute the second ::

    A1 = prod(B,C);
    A = prod(A1,D);

Another problem is aliasing. Aliasing occurs when the same variable is on both sides of the equation, e.g.: ::

    x = prod(A,x);

Without intermediate results, changing the first element of x would lead to an error in the computation of the second element, since this
would already use the changed vector. uBLAS prevents this automatically by evaluating the right side of the equation, saving the results in
an intermediate vector and copy the result at the end. Unfortunately, this also occurs when no aliasing happens. If you do not want this
overhead, you need to tell uBLAS that everything is okay ::

    noalias(y) = prod(A,x);

If you want to set all elements of a vector or matrix to zero, use
``x.clear()``.

But be aware, for sparse objects ``clear()`` removes all previously set elements, so iterating only over these elements is not possible anymore,
since they do not exist. Conversely setting an element just to 0 will not remove it from the sparse object.

You can also create subsets from vectors and matrices. These can also be used to assign results to them.
The most useful subsets are certainly subranges::

    x = subrange(y,start,end);
    subrange(A,startRow,endRow,startColumn,endColumn) = prod(B,C);





Init: Shark initialization framework for ublas vectors
------------------------------------------------------



Initializing vectors using the bracket notation ``vec(i)`` is cumbersome when you have to initialize bigger vectors.
Often deep nested loops need to be used. This is especially bad since throughout Shark often complex datastructures
are transformed into vectors for the :doxy:`IParameterizable` basis class. For complex structures, this can lead
to errors or incomprehensable code. Therefore Shark offers a smart framework especially designed for this task.
In the following, we will assume the task of storing parameters.

But let's see code. Initializing a vector works like this::

  RealVector parameters(7);
  //some things we want to store in the parameter vector
  RealVector vec (5);
  vec = ...;
  double a = 5;
  double b = 7;

  //and now initialize the parameter vector using
  init(parameters)<<vector,a,b;

After that, parameters is initialized as the vector with elements [0,...,4] being the elements of ``vec``, element 5 being ``a`` and
element 6 being ``b``. The framework also checks whether the length of
parameters and the right side expression are the same. Therefore
it is mandatory to initialize the vector with the correct size. For performance reasons, this check is only done in debug mode. Of course, instead
of simple vectors also subranges or matrix rows are possible.

If on the other hand your model receives a new parameter vector which needs to be split up into components again, the framework can
also handle that by only replacing ``<<`` by ``>>``::

  RealVector parameters = newParameters();
  //components of the parameter vector
  RealVector vector(5);
  double a = 0;
  double b = 0;

  //and now split the parameter vector
  init(parameters) >> vector,a,b;

Of course, most models do not only consist of vectors and numbers. As we force the sizes of both expressions to match, this
framework would not be very useful if we did not support more complex types. So we added some wrappers which can handle single
matrices and containers filled with vectors or matrices::

  RealVector parameters(...);
  //some possible types
  RealMatrix matrix;
  std::vector<RealMatrix> matrices;
  std::vector<RealVector> vectors;

  init(parameters) << toVector(matrix);
  init(parameters) << vectorSet(vectors);
  init(parameters) << matrixSet(matrices);

The entire initialization framework presented here, including the above wrappers, can also be used for sparse vectors and
matrices -- as long as these appear on the right side of the expression. The left hand side always needs to be a dense vector.
In addition, the nonzero elements of a sparse matrix must already be initialized.


The framework can also take ublas expressions, so in principle it is also possible to write::

  init(parameters)<< vec1+vec2 , prod(Mat,vec3);

However, this leads to unreadable code for longer expressions and thus is not very useful. You might want to use ``subrange()`` instead.

In addition, there also exist operators to directly obtain a row or column from a matrix (e.g. ``row()`` or ``RealMatrixRow()``, which
are equivalent when row is applied to a RealMatrix). See `this ublas page <http://www.boost.org/doc/libs/1_40_0/libs/numeric/ublas/doc/operations_overview.htm>_`
for an overview.







.. _label_for_linalg_atlas:


LinAlg: Fast Linear Algebra routines using ATLAS
------------------------------------------------



Since uBLAS is not the fastest linear algebra package, Shark offers a set of functions which are tailored to the most common used interfaces.
At the moment, only ATLAS is supported as their backend, but the routines use a reasonable fast uBLAS backend when Shark is compiled without ATLAS support.
To use them, the header ``shark/LinAlg/fastOperations.h`` must be included.

If you have an expression of the form

.. math::
  \vec{y} \leftarrow \alpha A \vec{x} + \beta \vec{y}\\
  Y \leftarrow \alpha A X + \beta Y

you can use :doxy:`fast_prod`::

  fast_prod(A,x,y,beta,alpha);
  fast_prod(A,X,Y,beta,alpha);

Alpha and beta have the defaults alpha=1.0 and beta = 0.0. This means, that the following expressions are equivalent::

  noalias(y)=prod(A,x);
  fast_prod(A,x,y);

Sometimes, the equation has the form of a symmetric rank-k update

.. math::
  Y \leftarrow \alpha A A^T + \beta Y

in this case, :doxy:`symmRankKUpdate` can be used::

  symmRankKUpdate(A,Y,beta,alpha);

However, Y must also be symmetric before the call, when beta is not 0, else this call will give wrong results.

A word on transposition and subranges: the vector version of fast_prod can be used when A is transposed,
or when x or y are subranges of their original vectors. The fast_prod or symm_prod for matrices only allows for
transposing the arguments, not of the resulting matrix Y. Trying this will fail dramatically during compilation
with long error messages.

For small problems, the performance gain of this functions is minimal, and especially not relevant for matrix-vector
products. In this case, you can just use the uBLAS prod as usual. For big problems, speed-ups of factor 5-10 are possible.




