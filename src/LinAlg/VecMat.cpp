//===========================================================================
/*!
 *  \file VecMat.cpp
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



#include <LinAlg/VecMat.h>
#include <LinAlg/LinAlg.h>


Vector::Vector()
: Array<double>(0u)
{
}

Vector::Vector(unsigned int len, bool zero)
: Array<double>(len)
{
	if (zero) (*(Array<double>*)this) = 0.0;
}

Vector::Vector(const Array<double>& other)
: Array<double>(other)
{
	SIZE_CHECK(ndim() == 1);
}

Vector::Vector(const Vector& other)
: Array<double>(other)
{
}

Vector::~Vector()
{
}


double Vector::norm2() const
{
	double ret = 0.0;
	int i, ic = dim(0);
	for (i=0; i<ic; i++)
	{
		double val = operator () (i);
		ret += val * val;
	}
	return ret;
}

bool Vector::normalize()
{
	double n = norm();
	if (n == 0.0) return false;
	int i, ic = dim(0);
	for (i=0; i<ic; i++) operator () (i) /= n;
	return true;
}

ArrayBase* Vector::clone() const
{
	return new Vector(*this);
}

ArrayBase* Vector::cloneEmpty() const
{
	return new Vector(0, false);
}

Vector::Vector(Array<double>& other, bool reference)
: Array<double>(other.dimvec(), other.elemvec(), other.ndim(), other.nelem(), reference)
{
}

Vector::Vector(unsigned int* d, double* e, unsigned int nd, unsigned int ne, bool reference)
: Array<double>(d, e, nd, ne, reference)
{
}

void Vector::resize_i(unsigned int* d, unsigned int nd, bool copy)
{
	SIZE_CHECK(nd == 1);
	Array<double>::resize_i(d, 1, copy);
}


////////////////////////////////////////////////////////////


Matrix::Matrix()
: Array<double>(0u, 0u)
{
}

Matrix::Matrix(unsigned int rows, unsigned int cols, bool zero)
: Array<double>(rows, cols)
{
	if (zero) (*(Array<double>*)this) = 0.0;
}

Matrix::Matrix(const Array<double>& other)
: Array<double>(other)
{
	SIZE_CHECK(ndim() == 2);
}

Matrix::Matrix(const Matrix& other)
: Array<double>(other)
{
	SIZE_CHECK(ndim() == 2);
}

Matrix::~Matrix()
{
}


// static
Matrix Matrix::unitmatrix(unsigned int dim)
{
	Matrix ret(dim, dim, true);
	unsigned int i;
	for (i=0; i<dim; i++) ret(i, i) = 1.0;
	return ret;
}

// static
Matrix Matrix::diagonal(const Array<double>& diag)
{
	SIZE_CHECK(diag.ndim() == 1);
	unsigned int dim = diag.dim(0);
	Matrix ret(dim, dim, true);
	unsigned int i;
	for (i=0; i<dim; i++) ret(i, i) = 1.0;
	return ret;
}

Matrix Matrix::transpose() const
{
	SIZE_CHECK(ndim() == 2);
	Matrix ret(*this);
	ret.Array<double>::transpose();
	return ret;
}

Matrix Matrix::inverse(unsigned maxIterations, double tolerance, bool ignoreThreshold) const
{
	SIZE_CHECK(ndim() == 2);
	Matrix ret(dim(1), dim(0));
	g_inverse(*this, ret, maxIterations, tolerance, ignoreThreshold);
	return ret;
}

Matrix Matrix::inverseCholesky(double thresholdFactor) const
{
	SIZE_CHECK(ndim() == 2);
	Matrix ret(dim(1), dim(0));
	g_inverseCholesky(*this, ret,thresholdFactor);
	return ret;
}

Matrix Matrix::inverseMoorePenrose() const
{
	SIZE_CHECK(ndim() == 2);
	Matrix ret(dim(1), dim(0));
	g_inverseMoorePenrose(*this, ret);
	return ret;
}

Matrix Matrix::inverseSymm() const
{
	SIZE_CHECK(ndim() == 2);
	Matrix ret(dim(1), dim(0));
	invertSymm(ret, *this);
	return ret;
}

Matrix Matrix::inverseSymmPositiveDefinite() const
{
	SIZE_CHECK(ndim() == 2);
	Matrix ret(dim(1), dim(0));
	invertSymmPositiveDefinite(ret, *this);
	return ret;
}

void Matrix::svd(Matrix& U, Matrix& V, Vector& lambda, unsigned int maxIterations, bool ignoreThreshold) const
{
	::svd(*this, U, V, lambda, maxIterations, ignoreThreshold);
}

void Matrix::eigensymm(Matrix& eigenvectors, Vector& eigenvalues) const
{
	SIZE_CHECK(ndim() == 2);
	SIZE_CHECK(dim(0) == dim(1));
	::eigensymm(*this, eigenvectors, eigenvalues);
}

void Matrix::eigenvalues(Vector& real, Vector* imaginary) const
{
	SIZE_CHECK(ndim() == 2);
	SIZE_CHECK(dim(0) == dim(1));
	real.resize(dim(0), false);
	if (imaginary != NULL)
	{
		imaginary->resize(dim(0), false);
		::eigen(*this, real, *imaginary);
	}
	else
	{
		Vector im(dim(0));
		::eigen(*this, real, im);
	}
}

double Matrix::detSymm() const
{
	SIZE_CHECK(ndim() == 2);
	SIZE_CHECK(dim(0) == dim(1));
	Matrix a(*this);
	Matrix v(dim(0), dim(1));
	Vector d(dim(0));
	return ::detsymm(a, v, d);
}

double Matrix::logDetSymm() const
{
	SIZE_CHECK(ndim() == 2);
	SIZE_CHECK(dim(0) == dim(1));
	Matrix a(*this);
	Matrix v(dim(0), dim(1));
	Vector d(dim(0));
	return ::logdetsymm(a, v, d);
}

double Matrix::trace() const
{
	SIZE_CHECK(isSquare());

	double ret = 0.0;
	unsigned int i, ic = dim(0);
	for (i=0; i<ic; i++) ret += operator () (i, i);
	return ret;
}

Vector Matrix::row(unsigned int r) const
{
	SIZE_CHECK(ndim() == 2);
	RANGE_CHECK(r < dim(0));

	unsigned int i, ic = dim(1);
	Vector ret(ic);
	for (i=0; i<ic; i++) ret(i) = operator () (r, i);
	return ret;
}

Vector Matrix::col(unsigned int c) const
{
	SIZE_CHECK(ndim() == 2);
	RANGE_CHECK(c < dim(1));

	unsigned int i, ic = dim(0);
	Vector ret(ic);
	for (i=0; i<ic; i++) ret(i) = operator () (i, c);
	return ret;
}

ArrayBase* Matrix::clone() const
{
	return new Matrix(*this);
}

ArrayBase* Matrix::cloneEmpty() const
{
	return new Matrix(0, 0);
}

void Matrix::resize_i(unsigned int* d, unsigned int nd, bool copy)
{
	SIZE_CHECK(nd == 2);
	Array<double>::resize_i(d, 2, copy);
}


////////////////////////////////////////////////////////////


// comparison
bool operator == (const Vector& v1, const Vector& v2)
{
	return ((Array<double>&)(v1) == (Array<double>&)(v2)); 
}

bool operator != (const Vector& v1, const Vector& v2)
{
	return ((Array<double>&)(v1) != (Array<double>&)(v2)); 
}

bool operator == (const Matrix& m1, const Matrix& m2)
{
	return ((Array<double>&)(m1) == (Array<double>&)(m2)); 
}

bool operator != (const Matrix& m1, const Matrix& m2)
{
	return ((Array<double>&)(m1) != (Array<double>&)(m2)); 
}

// addition and subtraction
Vector operator + (const Vector& v1, const Vector& v2)
{
	Vector ret(v1);
	ret += v2;
	return ret;
}

Vector operator - (const Vector& v1, const Vector& v2)
{
	Vector ret(v1);
	ret -= v2;
	return ret;
}

Vector& operator += (Vector& v1, const Vector& v2)
{
	unsigned int i, ic = v1.dim(0);
	SIZE_CHECK(v2.dim(0) == ic);

	for (i=0; i<ic; i++) v1(i) += v2(i);

	return v1;
}

Vector& operator -= (Vector& v1, const Vector& v2)
{
	unsigned int i, ic = v1.dim(0);
	SIZE_CHECK(v2.dim(0) == ic);

	for (i=0; i<ic; i++) v1(i) -= v2(i);

	return v1;
}

Matrix operator + (const Matrix& m1, const Matrix& m2)
{
	Matrix ret(m1);
	ret += m2;
	return ret;
}

Matrix operator - (const Matrix& m1, const Matrix& m2)
{
	Matrix ret(m1);
	ret -= m2;
	return ret;
}

Matrix& operator += (Matrix& m1, const Matrix& m2)
{
	unsigned int x, xc = m1.dim(1);
	unsigned int y, yc = m1.dim(0);
	SIZE_CHECK(m2.dim(0) == yc && m2.dim(1) == xc);

	for (y=0; y<yc; y++) for (x=0; x<xc; x++) m1(y, x) += m2(y, x);

	return m1;
}

Matrix& operator -= (Matrix& m1, const Matrix& m2)
{
	unsigned int x, xc = m1.dim(1);
	unsigned int y, yc = m1.dim(0);
	SIZE_CHECK(m2.dim(0) == yc && m2.dim(1) == xc);

	for (y=0; y<yc; y++) for (x=0; x<xc; x++) m1(y, x) -= m2(y, x);

	return m1;
}


// scalar multiplication
Vector operator * (double s, const Vector& v)
{
	int y, yc = v.dim(0);
	Vector ret(yc);
	for (y=0; y<yc; y++) ret(y) = s * v(y);
	return ret;
}

Matrix operator * (double s, const Matrix& m)
{
	int x, xc = m.dim(1);
	int y, yc = m.dim(0);
	Matrix ret(yc, xc);
	for (y=0; y<yc; y++) for (x=0; x<xc; x++) ret(y, x) = s * m(y, x);
	return ret;
}

Vector& operator *= (Vector& v, double s)
{
	int y, yc = v.dim(0);
	for (y=0; y<yc; y++) v(y) *= s;
	return v;
}

Matrix& operator *= (Matrix& m, double s)
{
	int x, xc = m.dim(1);
	int y, yc = m.dim(0);
	Matrix ret(yc, xc);
	for (y=0; y<yc; y++) for (x=0; x<xc; x++) m(y, x) *= s;
	return m;
}


// standard multiplication
Vector operator * (const Matrix& m, const Vector& v)
{
	unsigned int x, xc = m.dim(1);
	unsigned int y, yc = m.dim(0);
	SIZE_CHECK(v.dim(0) == xc);
	Vector ret(yc);
	for (y=0; y<yc; y++)
	{
		double a = 0.0;
		for (x=0; x<xc; x++) a += m(y, x) * v(x);
		ret(y) = a;
	}
	return ret;
}

Vector operator * (const Vector& v, const Matrix& m)
{
	unsigned int x, xc = m.dim(1);
	unsigned int y, yc = m.dim(0);
	SIZE_CHECK(v.dim(0) == yc);
	Vector ret(xc);
	for (x=0; x<xc; x++)
	{
		double a = 0.0;
		for (y=0; y<yc; y++) a += m(y, x) * v(y);
		ret(x) = a;
	}
	return ret;
}

Matrix operator * (const Matrix& m1, const Matrix& m2)
{
	unsigned int x, xc = m2.dim(1);
	unsigned int i, ic = m2.dim(0);
	unsigned int y, yc = m1.dim(0);
	SIZE_CHECK(m1.dim(1) == ic);
	Matrix ret(yc, xc);
	for (y=0; y<yc; y++)
	{
		for (x=0; x<xc; x++)
		{
			double a = 0.0;
			for (i=0; i<ic; i++) a += m1(y, i) * m2(i, x);
			ret(y, x) = a;
		}
	}
	return ret;
}

Matrix& operator *= (Matrix& m1, const Matrix& m2)
{
	Matrix tmp(m1.dim(0), m1.dim(1));
	tmp = m1 * m2;
	m1 = tmp;
	return m1;
}

// inner product
double operator * (const Vector& v1, const Vector& v2)
{
	unsigned int i, ic = v1.dim(0);
	SIZE_CHECK(v2.dim(0) == ic);
	double ret = 0.0;
	for (i=0; i<ic; i++) ret += v1(i) * v2(i);
	return ret;
}

// outer product
Matrix operator % (const Vector& v1, const Vector& v2)
{
	int y, yc = v1.dim(0);
	int x, xc = v2.dim(0);
	Matrix ret(yc, xc);
	for (y=0; y<yc; y++) for (x=0; x<xc; x++) ret(y, x) = v1(y) * v2(x);
	return ret;
}


// advanced math
Matrix powerseries(const Matrix& m, const double* coeff, int ncoeff, double acc)
{
	SIZE_CHECK(m.isSquare());

	int dim = m.dim(0);
	Matrix ret(dim, dim);
	Matrix power(Matrix::unitmatrix(dim));

	int i;
	for (i=0; i<ncoeff; i++)
	{
		if (coeff[i] != 0.0)
		{
			if (acc > 0.0)
			{
				double m = 0.0, M = 0.0;
				double mp, mr;
				minmaxElement(power, m, M);
				mp = M; if (m < 0.0 && -m > M) mp = -m;
				minmaxElement(ret, m, M);
				mr = M; if (m < 0.0 && -m > M) mr = -m;
				if (coeff[i] * mp < acc * mr) break;
			}
			ret += coeff[i] * power;
		}
		power *= m;
	}

	return ret;
}

Matrix exp(const Matrix& m)
{
	double coeff[60] = {
		1.0, 1.0, 0.5,
		0.1666666666666666667, 0.04166666666666666667, 0.008333333333333333333,
		0.001388888888888888889, 0.0001984126984126984127, 2.480158730158730159e-05,
		2.755731922398589065e-06, 2.755731922398589065e-07, 2.505210838544171878e-08,
		2.087675698786809898e-09, 1.60590438368216146e-10, 1.147074559772972471e-11,
		7.647163731819816476e-13, 4.779477332387385297e-14, 2.811457254345520763e-15,
		1.561920696858622646e-16, 8.220635246624329717e-18, 4.110317623312164858e-19,
		1.957294106339126123e-20, 8.896791392450573287e-22, 3.868170170630684038e-23,
		1.611737571096118349e-24, 6.446950284384473396e-26, 2.47959626322479746e-27,
		9.183689863795546148e-29, 3.27988923706983791e-30, 1.130996288644771693e-31,
		3.769987628815905644e-33, 1.21612504155351795e-34, 3.800390754854743593e-36,
		1.151633562077195028e-37, 3.387157535521161847e-39, 9.677592958631890992e-41,
		2.688220266286636387e-42, 7.265460179153071315e-44, 1.911963205040281925e-45,
		4.902469756513543398e-47, 1.225617439128385849e-48, 2.989310827142404511e-50,
		7.117406731291439311e-52, 1.655210867742195189e-53, 3.761842881232261792e-55,
		8.359650847182803983e-57, 1.817315401561479127e-58, 3.866628513960593887e-60,
		8.055476070751237264e-62, 1.643974708316579034e-63, 3.287949416633158067e-65,
		6.44695964045717268e-67, 1.239799930857148592e-68, 2.339245152560657722e-70,
		4.331935467704921706e-72, 7.876246304918039466e-74, 1.406472554449649905e-75,
		2.467495709560789306e-77, 4.254302947518602253e-79, 7.210682961895936021e-81,
	};

	return powerseries(m, coeff, 60, 1e-14);
}
