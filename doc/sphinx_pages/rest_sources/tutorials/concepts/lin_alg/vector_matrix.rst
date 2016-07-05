LinAlg: Vectors and Matrices
============================

All Linear Algebra related functions and operations are placed in the
LinAlg Module. LinAlg is based on an heavily altered subset of the boost linear algebra system
`uBLAS <http://www.boost.org/doc/libs/release/libs/numeric>`_ This tutorial
will give a short introduction to the library and how it is used in Shark.

The linear algebra library has a fundamental distinction between vectors and matrices:
Matrix and vectors are different types. There is no distinction between row and column vectors.
So when multiplying a vector from the right side to a matrix, it is assumed to be a column vector
and when multiplied from the left, it becomes a row-vector. Matrices have an orientation
that is a 1xn matrix has a different structure than a nx1 matrix and both are not the same as a vector.

Furthermore the library differentiates between container, proxies to containers and
vector expressions. The typical vector or matrix objects are a container storing
the values of the matrix. Proxies reference parts of a container, for example a single row
of a matrix or a subrange of a vector. And finally there are expressions which represent
arbitrary computations, for example a matrix-vector product.

Aliasing
------------------------------------------------------

All of then aforementioned entities are objects.
That means that just writing ``A+B`` does not compute the addition of two matrices but returns
an object which can compute the result. the computations occur only when the expression is assigned
to a container or proxy. Thus it is computed only in an expression like ``C=A+B``. In such an expression
the library assumes by default, that the left and right hand side are aliasing, i.e. that the same variable appears on
both sides of the expression. In many cases this means that changing elements of the result during computation,
will change future results. This happens for example in a matrix-matrix multiplication where the same variables are 
read at different points in time. To prevent this, the library generates temporary variables. So that an expression like
``A=prod(A,B)`` will be computed as::

   matrix<double> Temp=prod(A,B); 
   A=Temp; 

In many cases aliasing will not happen and this temporary result is an unneeded overhead. For this case, the 
noalias-proxy can be used::

  noalias(A) = prod(B,C);



We use the following notation for vectors, arguments and scalars:

======================= ====================================
Symbol           	Meaning
======================= ====================================
A,B,C			Matrix expression
x,y			Vector expression
C,z			Matrix or vector container or proxy to a
			subset of the matrix or vector.
t			Scalar value with the same type as
			the elements stored inside the matrix
i,jk,l,m,n		Integer values
b			Boolean value
======================= ====================================


Matrices and Matrix-Proxies and basic operations
------------------------------------------------------

We first want to describe the fundamental operations that are needed to use the library.
That is creating vectors and matrices and access their values. furthermore we describe the basic
operations to create proxies to various parts of a matrix.

=============================================== ==============================================
Operation/Class           			Effect
=============================================== ==============================================
``blas::vector<T>``				Dense Vector storing values of type T.
``blas::matrix<T,Orientation>``			Dense Matrix storing values of type T.
						Values are either stored in row-major or
						column-major format depicted by the tags
						blas::row_major and blas::column_major. 
						row-major is the default format.
``blas::compressed_vector<T>``			Sparse Vector storing values of type T in compressed format.
``blas::compressed_matrix<T>``			Sparse Matrix storing values of type T in compressed format.
						Only row-major format is supported.
``XVector,XMatrix,SparseXVector,SparseXMatrix``	Shorthand for the above types.
						X can be Real, Float, Int, UInt or Bool.
						X can be Real, Float, Int, UInt or Bool.
``XVector x(n,t)``				Creates a XVector x of size i with elements initialized to t.
						By default t is 0.
``XMatrix A(m,n,t)``				Creates a mxn matrix with elements initialized to t. By default t is 0.
``x.size()``					Dimensionality of x.
``A.size1(),A.size2()``		        	Size of the first (number of rows) and second(number of columns) dimension of A.
``A(i,j)``					Returns the ij-th element of the matrix A.
``x(i)``					Returns the i-th element of the vector x.
``row(A,k)``					Returns the k-th row of A as a vector-proxy.
``column(A,k)``					Returns the k-th column of A as a vector-proxy.
``rows(A,k,l)``					Returns the rows k,...,l of A as a matrix-proxy.
``columns(A,k,l)``				Returns the columns k,...,l of A as a matrix-proxy.
``subrange(x,i,j)``				Returns a sub-vector of x with the elements :math:`x_i,\dots,x_{j-1}`.
``subrange(A,i,j,k,l)``				Returns a sub-matrix of A with element indicated by i,j and k,l.
=============================================== ==============================================

Assignment
-----------------------------------------------------

todo: text:

=============================== ==============================================
Operation           		Effect
=============================== ==============================================
``C = E``			Evaluates E and assigns it to C using a temporary matrix.
``noalias(C) = E``		Like above with the difference that E is assigned
				directly to C without a temporary. C is required to have the
				same dimnsions as E.
``C += E, C-=E, C*=E, C/=E``    Equivalent to ``C = C+E, C = C-E, C = C*E`` and ``C= C/E``.
				See the section of Arithmetic operations for further details.
``C *= t, C/=t``        	Equivalent to ``C = C*t`` and ``C= C/t`` without creating a temporary value
				See the section of Arithmetic operations for further details.
``noalias(C) += E, ...``        Equivalent to ``C = C+E,...`` without creating a temporary value.
=============================== ==============================================

Arithmetic Operations and Expressions
--------------------------------------------------
In the following we present a list of arithmetic operations of vectors and matrices.


Elementwise operations transform a matrix or a vector by applying
a function on every element of the matrix: :math:`f(A)_{i,j} =f(A_{i,j})`.
For binary elementwise functions, both arguments are assumed to have
the same dimensionality and the function is applied on every pair
with the same index, that is :math:`f(A,B)_{i,j} = f(A_{i,j},B_{i,j})`.
It is checked in debug mode that both arguments have the same size.
The operations are the same for vectors and matrices and
we only present the matrix version:

=============================== ====================================
Operation           		Effect
=============================== ====================================
``t*B, B*t``      		scalar multiplication: :math:`t \cdot A_{ij}` and :math:`A_{ij}\cdot t`.
``B/t``      			scalar division: :math:`A_{ij}/t`.
``A+B``      			Elementwise Addition: :math:`A_{ij}+B_{ij}`.
``A-B``      			Elementwise Subtraction: :math:`A_{ij}-B_{ij}`.
``A*B, element_prod(A,B)``   	Elementwise Multiplication or Hadamard-Product:
				:math:`A_{ij} \cdot B_{ij}`.
``A/B, element_div(A,B)``	Elementwise division: :math:`A_{ij} \cdot B_{ij}`.
``safe_div(A,B,x)``     	Elementwise division with check for division for zero.
				If :math:`B_{ij} = 0` than the result is x.
``-A``				Negates A: :math:`-A_{ij}`.
``exp(A), log(A),...``  	Math functions applied on every element of the matrix,
				that is for example :math:`exp(A_{ij})`. Supported are:
				exp,log,abs, tanh and sqrt.
``pow(A,t)``			Applies the pow function on every element of A: :math:`pow(A_{ij},t)`
``sqr(A)``			Squares every entry of A, equivalent to A*A.
``sigmoid(A)``			Applies the sigmoid function :math:`f(x)=\frac{1}{1+e^{-x}}`
				to every element of A.
``softPlus(A)``			Applies the softplus function :math:`f(x)=log(1+e^{x})`
				to every element of A.
``trans(A)``			transposes the matrix A.
=============================== ====================================

Be aware that ``A*B`` is not the same as the typical matrix-product. For the typical
matrix-vector operations we use the following syntax:

=============================== ==================================================================
Operation           		Effect
=============================== ==================================================================
``prod(A,B)``			Matrix-Matrix product. Be aware that A is a mxk and B kxn matrix
				so that the resulting matrix is a mxn matrix.
``prod(A,x), prod(x,A)``	Matrix-Vector product :math:`Ax` and :math:`xA`.
``triangular_prod<Type>(A,x)``	Interpretes the matrix A as triangular matrix
				and claculates :math:`Ax`. 
				Type specifies the part of A that 
				is going to be treated as triangular. 
				Type can be lower,upper, unit_lower and unit_upper. The
				uni-variants represent a matrix with unit diagonal.
``triangular_prod<Type>(A,B)``	Interpretes the matrix A as triangular matrix
				and claculates :math:`AB`. 
				Type specifies the part of A that 
				is going to be treated as triangular. 
				Type is the same as above.
``inner_prod(x,y)``		vector product leading a scalar: :math:`\sum_i x_i y_i`.
``outer_prod(x,y)``		outer product leading a matrix C with :math:`C_{ij}=x_i y_j`.
=============================== ==================================================================

The fast variants of the functions above use ATLAS to speed up computation of
big dense matrices. The arguments need to have the right size and need to be at
least matrix or vector proxies. So if the argument is a more complex expression
like A+B or A*B it must be stored in a intermediate matrix first. Always try to
use the fast variants if possible as they can improve the performance of the
computations by an order of magnitude or more.


Examples
-----------------------------------------------------
todo

Initialization framework for vectors
------------------------------------------------------

Initializing vectors using the bracket notation ``vec(i)`` is cumbersome when you have to initialize bigger vectors.
Often deep nested loops need to be used. This is especially bad since throughout Shark often complex datastructures
are transformed into vectors for the :doxy:`IParameterizable` basis class. For complex structures, this can lead
to errors or incomprehensable code. Therefore Shark offers a smart framework especially designed for this task.
In the following, we will assume the task of storing parameters.

But let's see code. Initializing a vector works like this::

  RealVector parameters(7);
  //some things we want to store in the parameter vector
  RealVector vec(5);
  vec = ...;
  double a = 5;
  double b = 7;

  //and now initialize the parameter vector using
  init(parameters)<<vec,a,b;

After that, parameters is initialized as the vector with elements [0,...,4] being the elements of ``vec``, element 5 being ``a`` and
element 6 being ``b``. The framework also checks whether the length of
parameters and the right side expression are the same. Therefore
it is mandatory to initialize the vector with the correct size. For performance reasons, this check is only done in debug mode. Of course, instead
of simple vectors also subranges or matrix rows are possible.

If on the other hand your model receives a new parameter vector which needs to be split up into components again, the framework can
also handle that by only replacing ``<<`` with ``>>``::

  RealVector parameters = newParameters();
  //components of the parameter vector
  RealVector vec(5);
  double a = 0;
  double b = 0;

  //and now split the parameter vector
  init(parameters) >> vec,a,b;

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

The entire initialization framework presented here, including the above wrappers,
can also be used for sparse vectors and matrices -- as long as these appear on
the right side of the expression. The left hand side always needs to be a dense
vector. In addition, the nonzero elements of a sparse matrix must already be
initialized.


The framework can also use more comples expressions, so in principle it is also
possible to write::

  init(parameters)<< vec1+vec2 , prod(Mat,vec3);

However, this leads to unreadable code for longer expressions and thus is not
very useful. You might want to use ``subrange()`` instead.

In addition, there also exist operators to directly obtain a row or column from
a matrix (e.g. ``row()`` or ``RealMatrixRow()``, which are equivalent when row
is applied to a RealMatrix). See `this ublas page
<http://www.boost.org/doc/libs/release/libs/numeric/ublas/doc/operations_overview.htm>`_
for an overview.

