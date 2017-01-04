LinAlg: Eigenvalues, Inverses and Systems of Linear Equations
=============================================================
LinAlg offers a basis set of linear algebra routines especially for
solving systems of linear equations, matrix inversion, and eigenvalue problems.

A lot of the routines presented below use the :doc:`ATLAS <vector_matrix>` backend when available.
It is highly recommended to use this backend. 

Eigenvalues of a Matrix
--------------------------------------------------------------------
Eigenvalue equations are special linear equations of the form:

.. math::
  Ax=\lambda x

The lambdas are called eigenvalues and the :math:`x` are the eigenvectors with :math:`||x||=1`. If A is symmetric, :doxy:`eigensymm` can be used to solve this system::

  eigensymm(A,X,lambda);
  
the :math:`X` stores all eigenvectors of :math:`A` as columns and lambda is a vector of the corresponding eigenvalues. The *i*-th column 
of :math:`X` corresponds to the *i*-th element of lambda. Eigenvalues are sorted in descending order.

Singular Value Decomposition
--------------------------------------------------------------------
The Singular Value Decomposition decomposes :math:`A` into

.. math::
  A=UWV^T
  
Such that the columns of :math:`U` and :math:`V` are orthonormal and
:math:`W` is diagonal. 
The matrix :math:`A` does not need to have full rank. In fact, it does not even need to be quadratic. The
SVD can be used by including ``shark/LinAlg/svd.h``: ::

  svd(A,U,V,W);

The SVD is often used to compute pseudo-inverses, matrices which are only left or right inverses to :math:`A`.
