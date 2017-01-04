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

The library further distinguishes between three core concepts:
  * containers, which represent basic vector or matrix types
  * proxies, which represent parts or subsets of a vector or matrix,
    for example a matrix row or the transpose of a matrix.
  * expressions, which represent arithmetic operations on a matrix, for
    example a matrix-vector product or adding two matrices

All operations are lazy. That means that just writing ``A+B`` does not compute the addition of two matrices but returns
an object which can compute the result. the computations occur only when the expression is assigned
to a container or proxy. Thus it is computed only in an expression like ``C=A+B``. 
This lazyness gives the library the possibility to optimize expressions
using build-in knowlege of vector algebra. For example it can turn the multiplication of a vector
with the ivnerse to a matrix into solving a system of equations, which is not only much faster,
but also numerically more stable.

Aliasing
------------------------------------------------------

In an expression such as ``C=A+B`` it i assumed that the left and right hand side are aliasing, i.e. that the same variable appears on
both sides of the expression. In many cases this means that changing elements of the result during computation,
will change future results. This happens for example in a matrix-matrix multiplication where the same variables are 
read at different points in time. To prevent this, the library generates temporary variables. So that an expression like
``A=prod(A,B)`` will be computed as::

   matrix<double> Temp=prod(A,B); 
   A=Temp; 

In many cases aliasing will not happen and this temporary result is an unneeded overhead. For this case, the 
noalias-proxy can be used::

  noalias(A) = prod(B,C);


Operations
--------------------------------------------------------------------

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
*****************************************************************

We first want to describe the fundamental operations that are needed to use the library.
All operations are in the namespace ``blas::``.

=============================================== ==============================================
Operation/Class           			Effect
=============================================== ==============================================
``vector<T>``					Dense Vector storing values of type T.
``matrix<T,Orientation>``			Dense Matrix storing values of type T.
						Values are either stored in row-major or
						column-major format depicted by the tags
						blas::row_major and blas::column_major. 
						row-major is the default format.
``compressed_vector<T>``			Sparse Vector storing values of type T in compressed format.
``compressed_matrix<T>``			Sparse Matrix storing values of type T in compressed format.
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
=============================================== ==============================================

A number of proxy operations are available:

=============================================== ==============================================
Operation/Class           			Effect
=============================================== ==============================================
``subrange(x,i,j)``				Returns a sub-vector of x with the elements :math:`x_i,\dots,x_{j-1}`.
``subrange(A,i,j,k,l)``				Returns a sub-matrix of A with elements indicated by i,j and k,l.
``trans(A)``					Returns a proxy for the transpose of A.
``row(A,k)``					Returns the k-th row of A as a vector-proxy.
``column(A,k)``					Returns the k-th column of A as a vector-proxy.
``rows(A,k,l)``					Returns the rows k,...,l of A as a matrix-proxy.
``columns(A,k,l)``				Returns the columns k,...,l of A as a matrix-proxy.
``diag(A)``					Returns a vector-proxy to the matrix diagonal
=============================================== ==============================================

Be aware that the library is heavily optimizing the generated expressions, therefore it is not straight forward
to assess the type of a returned expression. Thus the auto keyword should be used when generating a proxy::

  auto row_proxy = row(A,3); //3d matrix row
  auto sub_proxy = subrange(row_proxy,3,5); //elements with indices 3 to 5
  sub_proxy(A) *= 2; //multiplies the elements A(3,3) A(3,4) and A(3,5) with 2
  subrange(row(A,3),3,5) *=2; //shorthand notation

Assignment
-----------------------------------------------------

Every Container and Proxy can be assigned values of an expression with the same size.

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
``A+B, A+t``      		Elementwise Addition: :math:`A_{ij}+B_{ij}` or :math:`A_{ij}+t`.
``A-B, A-t``      		Elementwise Subtraction: :math:`A_{ij}-B_{ij}`.
``min(A,B), min(A,t),...``      Elementwise min/max :math:`min(A_{ij},B_{ij})` :math:`min(A_{ij},t)`.
``A*B, element_prod(A,B)``   	Elementwise Multiplication or Hadamard-Product:
				:math:`A_{ij} \cdot B_{ij}`.
``A/B, element_div(A,B), A/t``	Elementwise division: :math:`A_{ij} \cdot B_{ij}`.
``safe_div(A,B,t)``     	Elementwise division with check for division for zero.
				If :math:`B_{ij} = 0` than the result is t.
``-A``				Negates A: :math:`-A_{ij}`.
``exp(A), log(A),abs(A)...``  	Math functions applied on every element of the matrix,
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
				and calculates :math:`Ax`. 
				Type specifies the part of A that 
				is going to be treated as triangular. 
				Type can be lower,upper, unit_lower and unit_upper. The
				unit-variants represent a matrix with unit diagonal.
``triangular_prod<Type>(A,B)``	Interpretes the matrix A as triangular matrix
				and calculates :math:`AB`. 
				Type specifies the part of A that 
				is going to be treated as triangular. 
				Type is the same as above.
``inner_prod(x,y)``		vector product leading a scalar: :math:`\sum_i x_i y_i`.
``outer_prod(x,y)``		outer product leading a matrix C with :math:`C_{ij}=x_i y_j`.
=============================== ==================================================================


Matrix and Vector Reductions
*************************************************************************************
Matrix reductions leaving either a vector or a scalar:

======================================= ==================================================================
Operation           			Effect
======================================= ==================================================================
``sum(A)``				Sum of elements of A :math:`\sum_{ij} A_{ij}`
``max(A), min(A)``			Maximum/Minimum element of A :math:`\max_{ij} A_{ij}`
``sum_rows(A)``				Sum of rows of A :math:`a_i = \sum_{j} A_{ij}`
``sum_columns(A)``			Sum of columns of A :math:`a_j = \sum_{i} A_{ij}`
``trace(A)``				Sum of diagonal elements of A :math:`a_j = \sum_{i} A_{ii}`
``norm_1(A), norm_inf(A)``		p-norm of A 
``norm_sqr(A)``				squared 2-norm of A
``norm_frobenius(A)``			frobenius norm of A :math:`\sum_{ij} A_{ij}A_{ij}`
======================================= ==================================================================

Vetor reductions to a scalar:

======================================= ==================================================================
Operation           			Effect
======================================= ==================================================================
``sum(v)``				Sum of elements of A :math:`\sum_{ij} A_{ij}`
``max(v), min(v)``			Maximum/Minimum element of v :math:`\max_{ij} A_{ij}`
``norm_1(v), norm_2(v), norm_inf(v)``	p-norm of v
``norm_sqr(v)``				squared 2-norm of v
======================================= ==================================================================


Misc
******************************************************

=============================== ==================================================================
Operation           		Effect
=============================== ==================================================================
``repeat(x,m)``			matrix with m rows that are a copy of x :math:`C_{ij}=x_j`.
``repeat(t,m,n)``		Matrix with m rows and n columns with :math:`C_{ij}=t`.
=============================== ==================================================================

Solving Systems of Linear Equations and Matrix Inverses
***********************************************************************************

The library comes with a set of operations to solve linear equations or inverting matrices.
A system of linear equations can have the forms

.. math::
  Ax=b \\
  xA=b \\
  AX=B \\
  XA=B
  
Thus A can either be on the left or right side, or we solve for a single vector or a whole matrix.

There are many different types of system, depending on the shape of A. If A is for example symmetric positive definite,
we can use more efficient and numerically stable algorithms than if A is an arbitrary matrix. Independend of the type of system,
the library offers the following functions:

=============================== ==================================================================
Operation           		Effect
=============================== ==================================================================
``solve(A, b,Type, Side)``	Solves a system of equations Ax=b or xA=b for a shape of A given
				by Type and the side of A given by the Side parameter 
				(``blas::left`` or ``blas::right``)
``solve(A,B,Type, Side)``	Solves a system of equations AX=b or XA=b for a shape of A given
				by Type and the side of A given by the Side parameter 
				(``blas::left`` or ``blas::right``)
``inv(A, type)``		Computes the explicit inverse of A with the shape given by Type.
``prod(inv(A,type),b)``		Computes :math:`A^{-1}b`.
				Equivalent to ``solve(A, b, Type, left)``
``prod(b,inv(A,type))``		Equivalent to ``solve(A, b, Type, right)``
``prod(inv(A,type),B)``		Computes :math:`A^{-1}B`.
				Equivalent to ``solve(A, B, Type, left)``
``prod(B,inv(A,type))``		Equivalent to ``solve(A, B, Type, right)``
=============================== ==================================================================

Note that the ``prod()``-versions are 100% equivalent to the ``solve()`` calls due to the
expression optimizations and it is thus up to preference which version is used.


Shark supports the following choices for Type:

=============================================== ==================================================================
Type	        				Effect
=============================================== ==================================================================
``blas::lower()``				A is a full rank lower triangular matrix.
``blas::upper()``				A is a full rank upper triangular matrix.
``blas::unit_lower()``				A is a lower triangular matrix with unit diagonal.
``blas::unit_upper()``				A is a upper triangular matrix with unit diagonal.
``blas::symm_pos_def()``			A is symmetric positive definite.
						Uses the cholesky decomposition to solve the system
``blas::conjugate_gradient(epsilon,max_iter)``	Uses the iterative conjugate gradient method to solve a
						symmetric positive definite system.
						Stopping criteria are math:``||Ax-b||_{\infty} < \epsilon``
						or the maximum number of iterations is reached. Default
						is math:``\epsilon=10^{-10}`` and unlimited max iterations.
``blas::indefinite_full_rank()``		A is an arbitrary full rank matrix.
						Uses the LU-decomposition to solve the system.
``blas::symm_semi_pos_def()``			A is symmetric positive definite but rank deficient, meaning
						that there might be no solution for Ax=b. Instead
						the solution that minimizes math:``||Ax-b||_2`` is computed.
=============================================== ==================================================================

A small example for the usage is::

  blas::matrix<double> C(100,50);
  //skip: fill C
  //compute a symmetric pos semi-definite matrix A
  blas::matrix<double> A = prod(C,trans(C));
  blas::vector<double> b(100,1.0);//all ones vector
  
  blas::vector<double> solution = prod(inv(A,blas::symm_semi_pos_def()),b);//solves Ax=b


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

