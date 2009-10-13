//===========================================================================
/*!
 *  \file LinAlg.cpp
 *
 *  \brief Some operations for matrices.
 *
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Copyright (c) 1999-2000:
 *      Institut fuer Neuroinformatik<BR>
 *      Ruhr-Universitaet Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de
 *
 *
 *
 *  <BR><HR>
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
 */
//======================================================================


#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <SharkDefs.h>
#include <Array/ArrayOp.h>
#include <LinAlg/LinAlg.h>


//===========================================================================
/*!
 *  \brief Calculates the mean vector of array "x".
 *
 *  Given a \em d -dimensional array \em x with size \em N1 x ... x \em Nd,
 *  this function calculates the mean vector given as:
 *  \f[
 *      mean_j = \frac{1}{N1} \sum_{i=1}^{N1} x_{i,j}
 *  \f]
 *  Example:
 *  \f[
 *      \left(
 *      \begin{array}{*{4}{c}}
 *          1 &  2 &  3 &  4\\
 *          5 &  6 &  7 &  8\\
 *          9 & 10 & 11 & 12\\
 *      \end{array}
 *      \right)
 *      \longrightarrow
 *      \frac{1}{3}
 *      \left(
 *      \begin{array}{*{4}{c}}
 *          1+5+9 & 2+6+10 & 3+7+11 & 4+8+12\\
 *      \end{array}
 *      \right)
 *      \longrightarrow
 *      \left(
 *      \begin{array}{*{4}{c}}
 *          5 &  6 &  7 &  8\\
 *      \end{array}
 *      \right)
 *  \f]
 *
 *      \param  x multidimensional array, from which the
 *                mean value will be calculated
 *      \return the mean vector of \em x
 *      \throw SharkException the type of the exception will
 *             be "size mismatch" and indicates that \em x is
 *             only one-dimensional or has no dimensions
 *
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
Array< double > mean(const Array< double >& x)
{
	SIZE_CHECK(x.ndim() > 1)

	Array< double > m(x[ 0 ]);
	for (unsigned i = 1; i < x.dim(0); ++i)
		m += x[ i ];
	m /= double(x.dim(0));
	return m;
}



//===========================================================================
/*!
 *  \brief Calculates the variance vector of array "x".
 *
 *  Given a \em d -dimensional array \em x with size \em N1 x ... x \em Nd
 *  and mean value vector \em m,
 *  this function calculates the variance vector given as:
 *  \f[
 *      variance = \frac{1}{N1} \sum_{i=1}^{N1} (x_i - m_i)^2
 *  \f]
 *
 *      \param  x multidimensional array, from which the
 *                variance will be calculated
 *      \return the variance vector of \em x
 *      \throw SharkException the type of the exception will
 *             be "size mismatch" and indicates that \em x is
 *             only one-dimensional or has no dimensions or
 *             has no values in its first dimension
 *
 *
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
Array< double > variance(const Array< double >& x)
{
	SIZE_CHECK(x.ndim() > 1 && x.dim(0) > 0)

	unsigned i, j;
	unsigned size = x.nelem() / x.dim(0);
	Array< double > m(size);   // vector of mean values.
	Array< double > v(size);   // vector of variance values

	m = 0.;
	v = 0.;

	for (i = 0; i < x.dim(0); ++i)
		for (j = 0; j < size; ++j) {
			m(j) += x.elem(i * size + j);// sum of elements in j's row
			v(j) += Shark::sqr(x.elem(i * size + j));//sum of squares od elements in j's row
		}

	for (j = 0; j < size; ++j) {
		m(j) /= x.dim(0);//normalize mean
		v(j)  = Shark::max(0., v(j) / x.dim(0) - m(j) * m(j));
	}

	// re-arrange variance array
	if (x.ndim() > 1)
		v.resize(x[ 0 ], true);//ensure v is a column vector

	return v;
}



//===========================================================================
/*!
 *  \brief Calculates the angle between the vectors "x" and "y".
 *
 *  Given the two one-dimensional vectors "x" and "y" with the same
 *  no. \em N of elements, the function calculates:
 *  \f[
 *      angle = \frac{\sum_{i=1}^{N} (x_i * y_i)}{\sum_{i=1}^{N} x_i *
 *              \sum_{i=1}^{N} y_i}
 *  \f]
 *
 *      \param  x one-dimensional vector no. 1
 *      \param  y one-dimensional vector no. 2
 *      \return the angle between \em x and \em y
 *      \throw SharkException the type of the exception will be
 *             "size mismatch" and indicates that \em x is not one-dimensional
 *             or that \em x has not the same size than \em y
 *
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *<
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
double angle(const Array< double >& x, const Array< double >& y)
{
	//return scalar_product( x, y ) / sqrt( sumOfSqr( x ) * sumOfSqr( y ) );

	SIZE_CHECK(x.ndim() == 1 && x.samedim(y))

	double sumxx = 0;
	double sumyy = 0;
	double sumxy = 0;

	for (unsigned i = 0; i < x.nelem(); ++i) {
		sumxx += x(i) * x(i);
		sumyy += y(i) * y(i);
		sumxy += x(i) * y(i);
	}

	return (sumxx == 0 || sumyy == 0) ? 0. : sumxy / sqrt(sumxx * sumyy);
}


//===========================================================================
/*!
 *  \brief Calculates the mean and variance values of matrix "x".
 *
 *  Given the input matrix \em x, the mean and variance values
 *  are calculated as in the functions #mean and #variance.
 *  The mean and variance values are stored in the vectors
 *  \em m and \em v.
 *
 *      \param  x The input matrix.
 *      \param  m Vector of mean values.
 *      \param  v Vector of variances.
 *      \return none.
 *      \throw SharkException the type of the exception will be
 *             "size mismatch" and indicates that \em x is only one-
 *             or non-dimensional
 *
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 *  \sa #mean, #variance
 *
 */
void meanvar
(
	const Array< double >& x,
	Array< double >&       m,
	Array< double >&       v
)
{
	SIZE_CHECK(x.ndim() > 1)

	unsigned i, j;

	// number of elements for each "row":
	unsigned size = x.nelem() / x.dim(0);

	m.resize(size, false);
	v.resize(size, false);
	m = 0.;
	v = 0.;

	for (i = 0; i < x.dim(0); ++i) {
		for (j = 0; j < size; ++j) {
			// Sum of elements of each "row":
			m(j) += x.elem(i * size + j);
			v(j) += Shark::sqr(x.elem(i * size + j));
		}
	}

	for (j = 0; j < size; ++j) {
		// Calculate mean value by dividing
		// by no. of "rows":
		m(j) /= x.dim(0);
		v(j)  = v(j) / x.dim(0) - m(j) * m(j);
	}

	// re-arrange arrays
	m.resize(x[ 0 ], true);
	v.resize(x[ 0 ], true);
}

/*
	/brief Calculates the mean and variance values of 1d-arrays p(x)

	calculates the first two moments of a discrete value array and
	his corresponding x-values
	/param pxA		1d array with functionvalues
	/param xA		1d array with corresponding x values
	/param mA		retunsvalue meanval
	/param vA		returnvalue variance
	/param startA	start of a calculation window
	/param endA		end of a calculation window
*/
void meanvar
(
	const Array< double >& pxA,
	const Array< double >& xA,
	double &mA,
	double &vA,
	const int startA,
	const int endA
)
{
//	SIZE_CHECK( xA.ndim( ) != 1 )
//	SIZE_CHECK( pxA.ndim( ) != xA.ndim( ) )
//	SIZE_CHECK( pxA.nelem( ) != xA.nelem( ) )

	int size = pxA.nelem();
	int startL	= (startA < 0) ? 0 : startA;
	int endL	= (endA == -1) ? size : endA;
	if (startL == endL) {
		mA = startA;
		vA = 0;
	}
	double ew = 0;
	double sum = 0;
	int i;
	// calculate first moment
	for (i = startL; i < endL ;++i) {
		ew	+= pxA.elem(i) * xA.elem(i);
		sum	+= pxA.elem(i);
	}
	mA = (ew == 0) ? 0 : ew / sum;
	// calculate variance
	// variance is the second central moment
	const int ii = 2;
	double vc = 0;
	for (i = startL; i < endL ;++i) {
		vc += pow((xA.elem(i) - mA), (double)ii) * pxA.elem(i);
	}
	vA = sqrt((vc == 0) ? 0 : vc / sum);

}




//===========================================================================
/*!
 *  \brief Calculates the coefficient of correlation of the data
 *         vectors "x" and "y".
 *
 *  Given two data vectors \f$x\f$ and \f$y\f$ of length \f$n\f$,
 *  the function calculates the coefficient of correlation given as
 *
 *  \f$
 *      r := \frac{cov(x, y)}{\Delta x \Delta y}
 *  \f$
 *
 *  where \f$cov(x, y)\f$ is the covariance between the two vectors
 *  (see also #covariance(const Array< double >&, const Array< double >&))
 *  and \f$\Delta x\f$ and \f$\Delta y\f$  are the standard deviations
 *  of \f$x\f$ and \f$y\f$ respectively. <br>
 *  The coefficient of correlation is used to show the dependence
 *  between \f$x\f$ and \f$y\f$. It always holds \f$-1 \leq r \leq 1\f$
 *  and the greater the value of \f$r^2\f$ is, the greater is the
 *  dependence between \f$x\f$ and \f$y\f$.
 *
 *  \param x first data vector.
 *  \param y second data vector.
 *  \return the coefficient of correlation.
 *  \throw SharkException the type of the exception will be
 *         "size mismatch" and indicates that \em x is not one-dimensional
 *         or \em x has not the same size than \em y
 *
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
double corrcoef(const Array< double >& x, const Array< double >& y)
{
	SIZE_CHECK(x.ndim() == 1 && x.samedim(y))

	double sumxx = 0;
	double sumyy = 0;
	double sumxy = 0;
	double sumx  = 0;
	double sumy  = 0;

	if (x.nelem() > 0) {
		for (unsigned i = 0; i < x.nelem(); ++i) {
			sumxx += x(i) * x(i);
			sumyy += y(i) * y(i);
			sumxy += x(i) * y(i);
			sumx  += x(i);
			sumy  += y(i);
		}

		sumxx /= x.nelem();
		sumyy /= x.nelem();
		sumxy /= x.nelem();
		sumx  /= x.nelem();
		sumy  /= x.nelem();

		sumxx -= sumx * sumx;
		sumyy -= sumy * sumy;
		sumxy -= sumx * sumy;
	}

	return (sumxx == 0 || sumyy == 0) ? 0. : sumxy / sqrt(sumxx * sumyy);
}



//===========================================================================
/*!
 *  \brief Calculates the coefficient of correlation matrix of the data
 *         vectors stored in matrix "x".
 *
 *  Given a matrix \f$X = (x_{ij})\f$ of \f$n\f$ vectors with length \f$N\f$,
 *  the function calculates the coefficient of correlation matrix given as
 *
 *  \f$
 *      r := (r_{kl}) \mbox{,\ } r_{kl} =
 *      \frac{c_{kl}}{\Delta x_k \Delta x_l}\mbox{,\ }
 *      k,l = 1, \dots, N
 *  \f$
 *
 *  where \f$c_{kl}\f$ is the entry of the covariance matrix of
 *  \f$x\f$ and \f$y\f$ (see #covariance(const Array<double>& x))
 *  and \f$\Delta x_k\f$ and \f$\Delta x_l\f$ are the standard
 *  deviations of \f$x_k\f$ and \f$x_l\f$ respectively.
 *
 *  \param x The \f$n \times N\f$ input matrix.
 *  \return The \f$N \times N\f$ coefficient of correlation matrix.
 *  \throw SharkException the type of the exception will be
 *         "size mismatch" and indicates that \em x is only one-
 *         or non-dimensional
 *
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
Array< double > corrcoef(const Array< double >& x)
{
	SIZE_CHECK(x.ndim() > 1)

	unsigned i, j;
	Array< double > C(covariance(x));

	for (i = 0; i < C.dim(0); ++i)
		for (j = 0; j < i; ++j)
			if (C(i, i) == 0 || C(j, j) == 0)
				C(i, j) = C(j, i) = 0;
			else
				C(i, j) = C(j , i) = C(i, j) / sqrt(C(i, i) * C(j, j));

	for (i = 0; i < C.dim(0); ++i)
		C(i, i) = 1;

	return C;
}


//===========================================================================
/*!
 *  \brief Calculates the covariance between the data vectors "x" and "y".
 *
 *  Given two data vectors \f$x\f$ and \f$y\f$ with length \f$n\f$,
 *  interpreted as \f$n\f$ points \f$(x_i, y_i)\f$ with
 *  \f$i = 1, \dots, n\f$, the function calculates the covariance given as
 *
 *  \f$
 *      cov = \frac{1}{n - 1} \sum_{i = 1}^n (x_i - \overline{x})
 *      (y_i - \overline{y})
 *  \f$
 *
 *  where \f$\overline{x}\f$ and \f$\overline{y}\f$ are the mean values
 *  of \f$x\f$ and \f$y\f$ respectively.
 *
 *  \param x first data vector.
 *  \param y second data vector.
 *  \return the covariance matrix.
 *  \throw SharkException the type of the exception will be
 *         "type mismatch" and indicates that \em x is not one-dimensional
 *         or \em x has not the same size than \em y
 *
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
double covariance(const Array< double >& x, const Array< double >& y)
{
	SIZE_CHECK(x.ndim() == 1 && x.samedim(y))

	double sumxy = 0;
	double sumx  = 0;
	double sumy  = 0;

	if (x.nelem() > 0) {
		for (unsigned i = 0; i < x.nelem(); ++i) {
			sumxy += x(i) * y(i);
			sumx  += x(i);
			sumy  += y(i);
		}

		sumxy /= x.nelem();
		sumx  /= x.nelem();
		sumy  /= x.nelem();
	}

	return sumxy - sumx * sumy;
}

//===========================================================================
/*!
 *  \brief Calculates the covariance matrix of the data vectors stored in
 *         matrix "x".
 *
 *  Given a matrix \f$X = (x_{ij})\f$ of \f$n\f$ vectors with length \f$N\f$,
 *  the function calculates the covariance matrix given as
 *
 *  \f$
 *      Cov = (c_{kl}) \mbox{,\ } c_{kl} = \frac{1}{n - 1} \sum_{i=1}^n
 *      (x_{ik} - \overline{x_k})(x_{il} - \overline{x_l})\mbox{,\ }
 *      k,l = 1, \dots, N
 *  \f$
 *
 *  where \f$\overline{x_j} = \frac{1}{n} \sum_{i = 1}^n x_{ij}\f$ is the
 *  mean value of \f$x_j \mbox{,\ }j = 1, \dots, N\f$.
 *
 *  \param x The \f$n \times N\f$ input matrix.
 *  \return \f$N \times N\f$ matrix of covariance values.
 *  \throw SharkException the type of the exception will be
 *         "type mismatch" and indicates that \em x is not 2-dimensional
 *
 *
 *  Please follow the link to view the source code of the example.
 *  The example can be executed in the example directory
 *  of package LinAlg.
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
Array< double > covariance(const Array< double >& x)
{
	SIZE_CHECK(x.ndim() == 2)

	unsigned num = x.dim(0);
	unsigned dim = x.dim(1);

	Array< double > mean(dim);
	Array< double > covar(dim, dim);

	mean  = 0.;
	covar = 0.;

	for (unsigned i = num; i--;) {
		mean  += x[ i ];
		covar += outerProduct(x[ i ], x[ i ]);
	}

	mean  /= double(num);
	covar /= double(num);
	covar -= outerProduct(mean, mean);

	return covar;
}

//===========================================================================
/*!
 *  \brief Computes A = BC , where B and C are an \f$n\times l\f$ and a
 *         \f$k\times l\f$ matrix, respectively
 *
 *  
 *  \author  C. Igel
 *  \date    2007
 *
 *  \par Changes
 *      none
 *
 *
 */
void matMat(Array2D<double> &A, const Array2D<double> &B, const Array2D<double> &C) {
  unsigned i, j, k, n = B.dim(0), m = C.dim(1), l = C.dim(0);
  SIZE_CHECK( B.dim(1) == C.dim(0) );  
  A.resize(n, m, false);
  for(i=0; i<n; i++) {
    for(j=0; j<m; j++) {
      A(i,j)=B(i, 0) * C(0, j);
      for(k=1; k<l; k++) {
				A(i,j)+=B(i, k) * C(k, j);
      }
    }
  }
}

//===========================================================================
/*!
 *  \brief Computes A = BC , where B is an \f$n\times n\f$ matrix and
 *         the \f$n\f$-dimensional Array C is viewed as a column vector
 *  
 *  \author  C. Igel
 *  \date    2007
 *
 *
 */
void matColVec(Array<double> &A, const Array2D<double> &B, const Array<double> &C)
{
	SIZE_CHECK(B.ndim() == 2);
	SIZE_CHECK(C.ndim() == 1);

	unsigned int v, vc = B.dim(0);
	unsigned int h, hc = B.dim(1);

	SIZE_CHECK(C.dim(0) == hc);
	A.resize(vc, false);

	for (v=0; v<vc; v++)
	{
		double value = 0.0;
		for (h=0; h<hc; h++) value += B(v, h) * C(h);
		A(v) = value;
	}
}

//===========================================================================
//!
//! \brief Computes A = BC , where B is an \f$n\times n\f$ matrix and
//!        the \f$n\f$-dimensional Array C is viewed as a column vector
//!
//! \author  T. Glasmachers
//! \date    2008
//!
void matColVec(ArrayReference<double> A, const Array2D<double>& B, const ArrayReference<double> C)
{
	SIZE_CHECK(B.ndim() == 2);
	SIZE_CHECK(C.ndim() == 1);

	unsigned int v, vc = B.dim(0);
	unsigned int h, hc = B.dim(1);

	SIZE_CHECK(C.dim(0) == hc);
	A.resize(vc, false);

	for (v=0; v<vc; v++)
	{
		double value = 0.0;
		for (h=0; h<hc; h++) value += B(v, h) * C(h);
		A(v) = value;
	}
}

//===========================================================================
/*!
 *  \brief Computes \f$ A = B C_i \f$ , where \f$ C_i \f$ is a column of the matrix C.
 *  
 *  \author  T. Glasmachers
 *  \date    2007
 *
 *  \par Changes
 *      none
 *
 *
 */
void matColVec(Array<double> &A, const Array2D<double> &B, const Array<double> &C, unsigned int index)
{
	unsigned i, j, n, m;
	SIZE_CHECK( B.ndim() == 2 );
	SIZE_CHECK( C.ndim() == 2 );
	RANGE_CHECK(index < C.dim(1));
	SIZE_CHECK( B.dim(1) == C.dim(0) );
	n = B.dim(0);
	m = B.dim(1);
	A.resize(n, false);
	for(i=0; i<n; i++) {
		A(i)=B(i, 0) * C(0, index);
		for(j=1; j<m; j++) {
			A(i)+=B(i, j) * C(j, index);
		}
	}
}

//===========================================================================
/*!
 *  \brief Computes s = ABC , where A and C are vectors of length \f$n\f$
 *         and B is an \f$n\times n\f$ matrix
 *
 *  
 *  \author  C. Igel
 *  \date    2007
 *
 *  \par Changes
 *      none
 *
 *
 */
double vecMatVec(const Array<double> &A, const Array2D<double> &B, const Array<double> &C) {
  unsigned i, j, n, m;
  double sum =  0, help;
  SIZE_CHECK( A.ndim() == 1 );  
  SIZE_CHECK( B.ndim() == 2 );  
  SIZE_CHECK( C.ndim() == 1 );  
  SIZE_CHECK( B.dim(1) == C.dim(0) );  
  SIZE_CHECK( B.dim(0) == A.dim(0) );  
  n = B.dim(0);
  m = B.dim(1);  

  for(i=0; i<n; i++) {
    help=B(i, 0) * C(0);
    for(j=1; j<m; j++) {
	help+=B(i, j) * C(j);
    }
    sum+=A(i)*help;
  }
  return sum;
}

//===========================================================================
/*!
 *  \brief Computes s = \f$A_iBC_j\f$ , where \f$A_i\f$ and \f$C_j\f$ are vectors of length \f$n\f$ and \f$m\f$, and \f$B\f$ is an \f$n \times m\f$ matrix.
 *
 *  
 *  \author  C. Igel
 *  \date    2007
 *
 *  \par Changes
 *      none
 *
 *
 */
double vecMatVec(const Array<double> &A, unsigned int i, const Array2D<double> &B, const Array<double> &C, unsigned int j) {
	unsigned a, c, n, m;
	double sum =  0, help;
	SIZE_CHECK( A.ndim() == 2 );
	SIZE_CHECK( B.ndim() == 2 );
	SIZE_CHECK( C.ndim() == 2 );
	RANGE_CHECK(i < A.dim(1));
	RANGE_CHECK(j < C.dim(1));
	SIZE_CHECK( B.dim(1) == C.dim(0) );
	SIZE_CHECK( B.dim(0) == A.dim(0) );
	n = B.dim(0);
	m = B.dim(1);

	for(a=0; a<n; a++) {
		help=B(a, 0) * C(0, j);
		for(c=1; c<m; c++) {
			help+=B(a, c) * C(c, j);
		}
		sum+=A(a, i)*help;
	}
	return sum;
}

//===========================================================================
/*!
 *  \brief Computes I = A^T for a square symmetric matrix A
 *  
 *  \author  C. Igel
 *  \date    2007
 *
 *  \par Changes
 *      none
 *
 *
 */
void invertSymm( Array2D<double> &I, const Array2D< double >& A) {
  SIZE_CHECK( A.ndim() == 2 );  
  SIZE_CHECK( A.dim(1) == A.dim(0) );  

  Array<double> lambda;
  Array2D<double> eigenVectors;
  unsigned n = A.dim(0);
  
  I.resize(A, false);
  lambda.resize(n, false);
  eigenVectors.resize(A, false);
  lambda = 0;
  eigenVectors = 0.;
  
  eigensymm(A, eigenVectors, lambda);
  I = 0.;
  
  for(unsigned i = 0; i < n; ++i) I(i,i) = 1./lambda(i);

  I = innerProduct(I , transpose(eigenVectors));
  I = innerProduct(eigenVectors , I);
}


//===========================================================================
/*!
 *  \brief lower triangular Cholesky decomposition
 *
 *  Given an \f$ m \times m \f$ symmetric positive definite matrix
 *  \f$M\f$, compute the lower triangular matrix \f$C\f$ such that \f$
 *  M=CC^T \f$
 *
 *      \param  M \f$ m \times m \f$ matrix, which must be symmetric and positive definite
 *      \param	C \f$ m \times m \f$ matrix, which stores the Cholesky factor
 *      \return none
 *
 *  \author  T. Suttorp and C. Igel
 *  \date    2008
 *
 *  \par Status
 *      stable
 *
 */

void CholeskyDecomposition(const Array2D< double >& M,
													 Array2D< double >& C) 
{
	int i, j, k, m;
	double s;
	SIZE_CHECK(M.dim(0) == M.dim(1));
	
	m = M.dim(0);
	C.resize(m, m, false);
	for(j = 0; j < m; j++) {
		for(i = j; i < m; i++) {
			s = M(i, j);
			for(k = 0; k < j; k++) {
				s -= C(i, k) * C(j, k);
			}
			if (i == j) {
				C(i, j) = sqrt(s);
			}
			else {
				C(i, j) = s/C(j , j);
				C(j, i) = 0;
			}
		}
	}
}

