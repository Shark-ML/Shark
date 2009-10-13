//===========================================================================
/*!
 *  \file VecMat.h
 *
 *  \brief Object oriented encapsulation of vectors and matrices
 *
 *  \par
 *  The aim of the classes Vector and Matrix,
 *  instead of the usage of Array&lt;double&gt;,
 *  is two-fold. First, these classes encapsulate
 *  a large part of the LinAlg functionality
 *  and thus make the library object-orientated.
 *  Second, these classes enable us to write
 *  vector and matrix multiplications in a
 *  clean and natural operator notation.
 *
 *  \author  T. Glasmachers
 *  \date    2008
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 *
 */
//===========================================================================


#ifndef _VecMat_H_
#define _VecMat_H_


#include <Array/Array.h>
#include <Array/ArrayOp.h>


class Matrix;


//!
//! \brief one-dimensional array of double used for vector computations
//!
//! \par
//! The Vector class represents column vectors.
//! There is no notion of a row vector (because it is not needed).
//!
class Vector : public Array<double>
{
public:
	friend class Matrix;

	//! Constructor
	Vector();

	//! Constructor
	Vector(unsigned int len, bool zero = false);

	//! Copy constructor (deep copy)
	Vector(const Array<double>& other);

	//! Copy constructor (deep copy)
	Vector(const Vector& other);

	//! Destructor
	~Vector();


	//! set all entries to the same value
	void operator = (double value)
	{ Array<double>::operator = (value); }

	inline double norm() const
	{
		return sqrt(norm2());
	}

	double norm2() const;
	bool normalize();

	//! \brief Clone the vector (deep copy).
	//!
	//! \par
	//! This method is inherited from ArrayBase.
	ArrayBase* clone() const;

	//! \brief Clone the class of the vector.
	//!
	//! \par
	//! The method returns a new uninitialized
	//! object of the class of the current array.
	//!
	//! \par
	//! This method is inherited from ArrayBase.
	ArrayBase* cloneEmpty() const;

protected:
	//! protected constructor for internal use
	Vector(Array<double>& other, bool reference);

	//! protected constructor for internal use
	Vector(unsigned int* d, double* e, unsigned int nd, unsigned int ne, bool reference);

	//! inherited internal resize interface
	void resize_i(unsigned int* d, unsigned int nd, bool copy);
};


//!
//! \brief two-dimensional array of double used for matrix computations
//!
class Matrix : public Array<double>
{
public:
	friend class Vector;

	//! Constructor
	Matrix();

	//! Constructor
	Matrix(unsigned int rows, unsigned int cols, bool zero = false);

	//! Copy constructor (deep copy)
	Matrix(const Array<double>& other);

	//! Copy constructor (deep copy)
	Matrix(const Matrix& other);

	//! Destructor
	~Matrix();


	//! Returns a matrix object filled with the unit
	//! matrix of the given dimensionality.
	static Matrix unitmatrix(unsigned int dim);

	//! Returns the diagonal matrix built from the given vector.
	static Matrix diagonal(const Array<double>& diag);

	//! set all entries to the same value
	void operator = (double value)
	{ Array<double>::operator = (value); }

	//! Returns the transpose of the matrix.
	Matrix transpose() const;

	//! Compute the generalized Moore-Penrose inverse of the matrix using SVD.
	Matrix inverse(unsigned maxIterations = 200, double tolerance = 1e-10, bool ignoreThreshold = true) const;

	//! Compute the inverse of the matrix.
	//! For non-symmetric matrices the result is undefined.
	Matrix inverseSymm() const;

	//! Compute the inverse of a symmetric positive definite matrix.
	Matrix inverseSymmPositiveDefinite() const;

	//! Returns the generalized inverse matrix of input matrix using Cholesky decomposition
	Matrix inverseCholesky(double thresholdFactor =  1e-9) const; 

	//! Returns the generalized inverse matrix of input matrix assuming that the matrix has full rank
	Matrix inverseMoorePenrose() const;

	//! Compute the Singular Value Decomposition
	void svd(Matrix& U, Matrix& V, Vector& lambda, unsigned int maxIterations = 200, bool ignoreThreshold = false) const;

	//! Compute eigenvectors and eigenvalues.
	//! For non-symmetric matrices the result is undefined.
	void eigensymm(Matrix& eigenvectors, Vector& eigenvalues) const;

	//! Compute real and imaginary parts of the eigenvalues.
	void eigenvalues(Vector& real, Vector* imaginary = NULL) const;

	//! Computes the determinant of the matrix.
	//! For non-symmetric matrices the result is undefined.
	double detSymm() const;

	//! Computes the determinant of the matrix.
	//! For non-symmetric or non-positive definite matrices
	//! the result is undefined.
	double logDetSymm() const;

	//! Computes the trace of the matrix.
	double trace() const;

	//! Checks whether the matrix is a square matrix.
	inline bool isSquare() const
	{
		SIZE_CHECK(ndim() == 2);
		return (dim(0) == dim(1));
	}

	//! Returns a row vector of the matrix (deep copy).
	Vector row(unsigned int r) const;

	//! Returns a column vector of the matrix (deep copy).
	Vector col(unsigned int c) const;

	//! Returns a row vector of the matrix.
	//! This vector is a reference to
	//! (no deep copy of) the matrix row.
	inline Vector operator [] (unsigned int i)
	{
		SIZE_CHECK(ndim() == 2);
		RANGE_CHECK(i < dim(0));
		return Vector(this->d + 1, this->e + dim(1) * i, 2, dim(1), true);
	}

	//! Returns a const row vector of the matrix.
	inline const Vector operator [] (unsigned int i) const
	{
		return row(i);
	}

	//! \brief Clone the matrix (deep copy).
	//!
	//! \par
	//! This method is inherited from ArrayBase.
	ArrayBase* clone() const;

	//! \brief Clone the class of the matrix.
	//!
	//! \par
	//! The method returns a new uninitialized
	//! object of the class of the current array.
	//!
	//! \par
	//! This method is inherited from ArrayBase.
	ArrayBase* cloneEmpty() const;

protected:
	//! inherited internal resize interface
	void resize_i(unsigned int* d, unsigned int nd, bool copy);
};


// comparison
bool operator == (const Vector& v1, const Vector& v2);
bool operator != (const Vector& v1, const Vector& v2);
bool operator == (const Matrix& m1, const Matrix& m2);
bool operator != (const Matrix& m1, const Matrix& m2);

// addition and subtraction
Vector operator + (const Vector& v1, const Vector& v2);
Vector operator - (const Vector& v1, const Vector& v2);
Vector& operator += (Vector& v1, const Vector& v2);
Vector& operator -= (Vector& v1, const Vector& v2);
Matrix operator + (const Matrix& m1, const Matrix& m2);
Matrix operator - (const Matrix& m1, const Matrix& m2);
Matrix& operator += (Matrix& m1, const Matrix& m2);
Matrix& operator -= (Matrix& m1, const Matrix& m2);

// scalar multiplication
Vector operator * (double s, const Vector& v);
Matrix operator * (double s, const Matrix& m);
Vector& operator *= (Vector& v, double s);
Matrix& operator *= (Matrix& m, double s);

// standard multiplication
Vector operator * (const Matrix& m, const Vector& v);
Vector operator * (const Vector& v, const Matrix& m);
Matrix operator * (const Matrix& m1, const Matrix& m2);
Matrix& operator *= (Matrix& m1, const Matrix& m2);

// inner product
double operator * (const Vector& v1, const Vector& v2);

// outer product
Matrix operator % (const Vector& v1, const Vector& v2);


// advanced math
Matrix powerseries(const Matrix& m, const double* coeff, int ncoeff, double acc = 1e-14);
Matrix exp(const Matrix& m);


#endif
