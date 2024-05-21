Remora Linear Algebra
============================

Aliasing
------------------------------------------------------

In an expression such as ``C = A + B`` it is assumed that the left and right hand side are aliasing, i.e. that the same variable can appear on
both sides of the expression. In many cases this means that elements changing early during the computation are read as part of the right hand side and change subsequent computation, which yields wrong results.
This happens for example in a matrix-matrix multiplication. To prevent this, the library generates temporary variables, so that an expression like
``A = A % B`` is computed as::

   matrix<double> Temp = A % B; 
   A = Temp; 

In many cases aliasing does not happen and the temporary variable is an unneeded overhead.
It is avoided with the noalias-proxy::

  noalias(A) = B % C;


Operations
--------------------------------------------------------------------

We use the following notation for vectors, arguments and scalars:

======================= ====================================
Symbol           	Meaning
======================= ====================================
A,B,C			Matrix expression
x,y			Vector expression
t			Scalar value with the same type as
			the elements stored inside the matrix
i,j,k,l,m,n		Integer values
b			Boolean value
======================= ====================================


Matrices, Matrix-Proxies and Basic Operations
*****************************************************************

The following types and operations are at the heart of the library.
All operations are in the namespace ``remora``.

=============================================== ==============================================
Operation/Class           			Effect
=============================================== ==============================================
``vector<T>``					Dense Vector storing values of type T.
``matrix<T,Orientation>``			Dense Matrix storing values of type T.
						Values are either stored in row-major or
						column-major format depicted by the tags
						row_major and column_major. 
						row-major is the default format.
``compressed_vector<T>``			Sparse Vector storing values of type T in compressed format.
``compressed_matrix<T>``			Sparse Matrix storing values of type T in compressed format.
``vector<T> x(n,t)``				Creates a XVector x of size i with elements initialized to t.
						By default t is 0.
``matrix<T,orientation> A(m,n,t)``		Creates a mxn matrix with elements initialized to t. By default t is 0.
						Orientation describes the storage scheme. Can either be 
						``row_major`` or ``column_major``
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
``subrange(A,i,j,k,l)``				Returns a sub-matrix of A with elements indicated by :math:`i,\dots,j-1` and :math:`k,\dots,l-1`.
``trans(A)``					Returns a proxy for the transpose of A.
``row(A,k)``					Returns the k-th row of A as a vector-proxy.
``column(A,k)``					Returns the k-th column of A as a vector-proxy.
``rows(A,k,l)``					Returns the rows :math:`k,\dots,l-1` of A as a matrix-proxy.
``columns(A,k,l)``				Returns the columns :math:`k,\dots,l-1` of A as a matrix-proxy.
``diag(A)``					Returns a vector-proxy to the matrix diagonal
=============================================== ==============================================

The library is heavily optimizing the generated expressions, therefore it is not straightforward
to deduce the type of a returned expression. Thus the auto keyword should be used when generating a proxy::

  auto row_proxy = row(A,2);  // 3rd matrix row
  auto sub_proxy = subrange(row_proxy,3,5);   // elements with indices 3 and 4
  sub_proxy(A) *= 2;   // multiplies the elements A(3,3) and A(3,4) with 2
  subrange(row(A,3),3,5) *=2; //shorthand notation

Assignment
-----------------------------------------------------

Every Container and Proxy can be assigned values of an expression with the same size.

======================================= ==============================================
Operation           		Effect
======================================= ==============================================
``C = E``				Evaluates E and assigns it to C using a temporary matrix.
``noalias(C) = E``			Like above with the difference that E is assigned
					directly to C without a temporary. C is required to have the
					same dimensions as E.
``C += E, C -= E, C *= E, C /= E``	Equivalent to ``C = C+E, C = C-E, C = C*E`` and ``C = C/E``.
					See the section of Arithmetic operations for further details.
``C *= t, C /= t``        		Equivalent to ``C = C*t`` and ``C = C/t`` without creating a temporary value
					See the section of Arithmetic operations for further details.
``noalias(C) += E, ...``        	Equivalent to ``C = C+E,...`` without creating a temporary value.
======================================= ==============================================

Arithmetic Operations and Expressions
--------------------------------------------------
This section lists arithmetic operations on vectors and matrices.

Element-wise operations transform a matrix or a vector by applying
a function to each entry of the matrix: :math:`f(A)_{i,j} =f(A_{i,j})`.
For binary element-wise functions, both arguments are required to have
the same dimensionality and the function is applied to each pair
with corresponding index, i.e., :math:`f(A,B)_{i,j} = f(A_{i,j},B_{i,j})`.
It is checked in debug mode that both arguments have the same size.
The operations are the same for vectors and matrices and
we only present the matrix versions in the following table:

=============================== ====================================
Operation           		Effect
=============================== ====================================
``t*B, B*t``      		scalar multiplication: :math:`t \cdot A_{ij}` and :math:`A_{ij}\cdot t`.
``B/t``      			scalar division: :math:`A_{ij}/t`.
``A+B, A+t, A-B, A-y``      	element-wise addition or subtraction: :math:`A_{ij}+B_{ij}` or :math:`A_{ij}+t`.
``min(A,B), min(A,t),...``      element-wise min/max :math:`min(A_{ij},B_{ij})` :math:`min(A_{ij},t)`.
``A*B, element_prod(A,B)``   	element-wise multiplication or Hadamard product:
				:math:`A_{ij} \cdot B_{ij}`.
``A/B, element_div(A,B), A/t``	element-wise division: :math:`A_{ij} \cdot B_{ij}`.
``safe_div(A,B,t)``     	element-wise division with check for division by zero.
				If :math:`B_{ij} = 0` then the result is t.
``-A``				negation: :math:`-A_{ij}`.
``exp(A), log(A),abs(A)...``  	math functions applied to each entry of the matrix,
				e.g., :math:`exp(A_{ij})`. Supported are:
				exp,log,abs, tanh and sqrt.
``pow(A,t)``			applies the pow function to each entry of A: :math:`pow(A_{ij},t)`
``sqr(A)``			squares each entry of A, equivalent to A*A.
``sigmoid(A)``			Applies the sigmoid function :math:`f(x)=\frac{1}{1+e^{-x}}`
				to each entry of A.
``softPlus(A)``			Applies the softplus function :math:`f(x)=log(1+e^{x})`
				to each entry of A.
=============================== ====================================

Be aware that ``A * B`` is not the same as the usual matrix-product
in linear algebra. For matrix-vector operations we use the following syntax:

======================================= ==================================================================
Operation           			Effect
======================================= ==================================================================
``A%B, prod(A,B)``			Matrix-matrix product. If A is a :math:`m \times k` and B a :math:`k \times n` matrix
					then the result is a :math:`m \times n` matrix.
``x % A, A % x, prod(A,x), prod(x,A)``	Matrix-Vector product :math:`Ax` and :math:`xA`.
``triangular_prod<Type>(A,x)``		Interpretes the matrix A as triangular matrix
					and calculates :math:`Ax`. 
					Type specifies the part of A that 
					is going to be treated as triangular. 
					Type can be lower, upper, unit_lower, and unit_upper. The
					unit-variants represent a matrix with unit diagonal.
``triangular_prod<Type>(A,B)``		Interpretes the matrix A as triangular matrix
					and calculates :math:`AB`. 
					Type specifies the part of A that 
					is treated as triangular. 
					Type is the same as above.
``inner_prod(x,y)``			inner or scalar product, yielding a scalar: :math:`\sum_i x_i y_i`.
``outer_prod(x,y)``			outer product, yielding a matrix C with :math:`C_{ij}=x_i y_j`.
======================================= ==================================================================

Block Matrix Operations
*********************************************
These matrix operation create larger matrices from smaller ones using operators ``&`` and ``|``.
Consider matrices A, B, C and D, from which we'd like to create

.. math::
	C=
		\left[
			\begin{array}{c|c}
				A & B \\
				\hline
				C & D
			\end{array}
		\right]

This can easily be done using ``(A | B) & (C | D)``. The allowed
operations are:

======================================= ==================================================================
Operation           			Effect
======================================= ==================================================================
``x | y``				Creates a vector of the values of x followed by values of y
``A | B``				Block Matrix where B is right of A
``A & B``				Block Matrix where B is below A
``A | x, x | A``			Vector x is interpreted as matrix with one column
``A & x, x & A``			Vector x is interpreted as matrix with one row
``A | t, t | A, A & t, t & A``		Scalar t is interpreted as matrix with a single
					row or column matching A. 
					``(A|1)`` adds a column of all ones to the right
======================================= ==================================================================


Matrix and Vector Reductions
*************************************************************************************

Vector reductions to a scalar:

======================================= ==================================================================
Operation           			Effect
======================================= ==================================================================
``sum(v)``				Sum of entries of A: :math:`\sum_{ij} A_{ij}`
``max(v), min(v)``			Maximum/Minimum entry of v: :math:`\max_{ij} A_{ij}`
``norm_1(v), norm_2(v), norm_inf(v)``	p-norm of v
``norm_sqr(v)``				squared 2-norm of v
======================================= ==================================================================

Vector Sets interpret matrices as sets of vectors where each row or column is one point. This allows
to perform vector operations on all points at the same time:

======================================= ==================================================================
Operation           			Effect
======================================= ==================================================================
``as_rows(A), as_columns(A)``		Interpret rows or columns as independent points.
``sum(as_rows(A)), ...`` 		For each row, compute its sum, maximum element, etc
``norm_1(as_rows(A)), ...`` 		Compute norms of all rows of A
======================================= ==================================================================


Matrix reductions leaving either a vector or a scalar:

======================================= ==================================================================
Operation           			Effect
======================================= ==================================================================
``sum(A)``				Sum of elements of A: :math:`\sum_{ij} A_{ij}`
``max(A), min(A)``			Maximum/Minimum element of A: :math:`\max_{ij} A_{ij}`
``trace(A)``				Sum of diagonal entries of A: :math:`a_j = \sum_{i} A_{ii}`
``norm_1(A), norm_inf(A), norm_2(A)``	p-norm of A
``norm_sqr(A)``				squared 2-norm of A
``norm_frobenius(A)``			Frobenius norm of A :math:`\sum_{ij} A_{ij}A_{ij}`
======================================= ==================================================================




Misc
******************************************************

The repeat function creates matrices by repeating a vector or scalar.

=============================== ==================================================================
Operation           		Effect
=============================== ==================================================================
``repeat(x,m)``			matrix with m rows that are copies of x :math:`C_{ij}=x_j`.
``repeat(t,m,n)``		Matrix with m rows and n columns with :math:`C_{ij}=t`.
=============================== ==================================================================

Solving Systems of Linear Equations and Matrix Inverses
***********************************************************************************

The library comes with a set of operations for solving linear equations and inverting matrices.
A system of linear equations can have the forms

.. math::
  Ax=b \\
  xA=b \\
  AX=B \\
  XA=B
  
Thus A can either be on the left or right side, or we solve for a single vector or a whole matrix.

There are many different types of systems, depending on the shape of A. If A is for example symmetric positive definite,
then we can use more efficient and numerically stable algorithms than if A is an arbitrary matrix. Independent of the type of system,
the library offers the following functions:

=============================== ==================================================================
Operation           		Effect
=============================== ==================================================================
``solve(A, b,Type, Side)``	Solves a system of equations Ax=b or xA=b for a shape of A given
				by Type and the side of A given by the Side parameter 
				(``left`` or ``right``)
``solve(A,B,Type, Side)``	Solves a system of equations AX=b or XA=b for a shape of A given
				by Type and the side of A given by the Side parameter 
				(``left`` or ``right``)
``inv(A, type)``		Computes the explicit inverse of A with the shape given by Type.
``inv(A,type) % b``		Computes :math:`A^{-1}b`.
				Equivalent to ``solve(A, b, Type, left)``
``b % inv(A,type)``		Equivalent to ``solve(A, b, Type, right)``
``inv(A,type) % B``		Computes :math:`A^{-1}B`.
				Equivalent to ``solve(A, B, Type, left)``
``B % inv(A,type)``		Equivalent to ``solve(A, B, Type, right)``
=============================== ==================================================================

Note that the ``prod()``-versions are 100% equivalent to the ``solve()`` calls due to the
expression optimizations and it is thus up to preference which version is used.


Remora supports the following system types:

=============================================== ==================================================================
Type	        				Effect
=============================================== ==================================================================
``lower()``					A is a full rank lower triangular matrix.
``upper()``					A is a full rank upper triangular matrix.
``unit_lower()``				A is a lower triangular matrix with unit diagonal.
``unit_upper()``				A is a upper triangular matrix with unit diagonal.
``symm_pos_def()``				A is symmetric positive definite.
						Uses the cholesky decomposition to solve the system
``conjugate_gradient(epsilon,max_iter)``	Uses the iterative conjugate gradient method to solve a
						symmetric positive definite system.
						Stopping criteria are :math:``||Ax-b||_{\infty} < \epsilon``
						or the maximum number of iterations is reached. Default
						is :math:``\epsilon=10^{-10}`` and unlimited iterations.
``indefinite_full_rank()``			A is an arbitrary full rank matrix.
						Uses the LU-decomposition to solve the system.
``symm_semi_pos_def()``				A is symmetric positive definite but rank deficient, meaning
						that there might be no solution for Ax=b. Instead
						the solution that minimizes :math:``||Ax-b||_2`` is computed.
=============================================== ==================================================================

A small example for the usage is::

  matrix<double> C(100, 50);
  // skip: fill C
  // compute a symmetric pos semi-definite matrix A
  matrix<double> A = C % trans(C);
  vector<double> b(100, 1.0);         // all ones vector
  
  vector<double> solution = inv(A,symm_semi_pos_def()) % b;   // solves Ax=b

