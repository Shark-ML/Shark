//===========================================================================
/*!
 *  \file ArrayOp.h
 *
 *  \brief Offers some operations for arrays of the template class
 *         Array.
 *
 *  Notice: The five operator types "unary operators", "binary operators",
 *  "compound assignment operators", "unary functions" and "binary functions"
 *  listed above are not given in the function documentation. They
 *  were removed from the reference, because their special definition
 *  causes doxygen to create a corrupted reference. <BR>
 *  But all instances of these operators and a short description
 *  about using them is given in the list below. <BR> <BR>
 *  In this file you will find the following operators and methods:
 *  <ul>
 *      <li>assignment operators for the test of equality, inequality,
 *          \f$<=\f$ and \f$>=\f$ between arrays and array references</li>
 *      <li>general unary operators that can be applied to each
 *          element of one array, with the predefined unary operators:</li>
 *      <ul>
 *          <li>negation ( ! )</li>
 *          <li>complement ( ~ )</li>
 *          <li>positive ( + )</li>
 *          <li>negative ( - )</li>
 *      </ul>
 *      \par Example
 *      \code
 *      Array< double > preArray( 10 ), postArray( 10 );
 *
 *      // Fill Array "preArray" with content:
 *      ...
 *
 *      // Array "postArray" becomes the negation of Array "preArray":
 *      postArray = !preArray;
 *      \endcode
 *
 *      <li>general binary operators that can be applied pairwise to each
 *          element of two arrays or to each element of one array and
 *          a single value, with the predefined binary operators:
 *      <ul>
 *          <li>OR ( || )</li>
 *          <li>AND ( && )</li>
 *          <li>bitwise OR ( | )</li>
 *          <li>bitwise exclusive OR ( ^ )</li>
 *          <li>bitwise AND ( & )</li>
 *          <li>addition ( + )</li>
 *          <li>subtraction ( - )</li>
 *          <li>multiplication ( * )</li>
 *          <li>division ( / )</li>
 *          <li>modulo ( % )</li>
 *      </ul>
 *      \par Example
 *      \code
 *      Array< double > preArray1( 10 ), preArray2( 10 ),
 *                      postArray1( 10 ), postArray2( 10 );
 *      double          operand( -2.0 );
 *
 *      // Fill Arrays "preArray1" and "preArray2" with content:
 *      ...
 *
 *      // Array "postArray1" becomes the sum of the Arrays "preArray1"
 *      // and "preArray2":
 *      postArray1 = preArray1 + preArray2;
 *
 *      // Array "postArray2" becomes the product of the Array "preArray1"
 *      // with the "operand":
 *      postArray2 = preArray1 * operand;
 *
 *      // The order of the operands is not important, so it is
 *      // also possible to write: postArray2 = operand * preArray1;
 *      \endcode
 *
 *      <li>general compound assignment operators that are used as
 *          abbreviations for more complex operations, with the
 *          predefined compound assignment operators:</li>
 *      <ul>
 *          <li>OR into ( |= )</li>
 *          <li>exclusive OR into ( ^= )</li>
 *          <li>AND into ( &= )</li>
 *          <li>increase ( += )</li>
 *          <li>decrease ( -= )</li>
 *          <li>multiply by ( *= )</li>
 *          <li>divide by ( /= )</li>
 *          <li>remainder ( %= )</li>
 *      </ul>
 *      \par Example
 *      \code
 *      Array< double > Array1( 10 ), Array2( 10 );
 *      double          operand( -2.0 );
 *
 *      // Fill Arrays "Array1" and "Array2" with content:
 *      ...
 *
 *      // Array "Array1" becomes the sum of Array "Array2" and itself:
 *      Array1 += Array2;
 *
 *      // And now multiply it with the "operand":
 *      Array1 *= operand;
 *
 *      // Instead of an Array you can also use an Array Reference
 *      // on the left side.
 *      \endcode
 *
 *      <li>general unary functions that are used on all elements
 *          of an array, with the predefined unary functions:</li>
 *      <ul>
 *          <li>arcus cosinus ( acos )</li>
 *          <li>arcus sinus ( asin )</li>
 *          <li>arcus tangens ( atan )</li>
 *          <li>cosinus ( cos )</li>
 *          <li>sinus ( sin )</li>
 *          <li>tangens ( tan )</li>
 *          <li>cosinus hyperbolicus ( cosh )</li>
 *          <li>sinus hyperbolicus ( sinh )</li>
 *          <li>tangens hyperbolicus ( tanh )</li>
 *          <li>arcus cosinus hyperbolicus ( acosh )</li>
 *          <li>arcus sinus hyperbolicus ( asinh )</li>
 *          <li>arcus tangens hyperbolicus ( atanh )</li>
 *          <li>exponential function ( exp )</li>
 *          <li>logarithmus naturalis ( log )</li>
 *          <li>logarithmus decimalis ( log10 )</li>
 *          <li>square root ( sqrt )</li>
 *          <li>cubic root ( cbrt )</li>
 *          <li>round up value ( ceil )</li>
 *          <li>round down value ( floor )</li>
 *          <li>absolute value of floating point ( fabs )</li>
 *      </ul>
 *      \par Example
 *      \code
 *      Array< double > preArray( 10 ), postArray( 10 );
 *
 *      // Fill Array "preArray" with content:
 *      ...
 *
 *      // Array "postArray" contains the cosinus of all values of
 *      // Array "preArray":
 *      postArray = cos( preArray );
 *      \endcode
 *
 *       <li>general binary functions that are used pairwise on each
 *           element of two arrays or on each element of one array and
 *           a single value, with the predefined binary functions:</li>
 *       <ul>
 *           <li>arcus tangens for two parameters ( atan2 )</li>
 *           <li>numeric power ( pow )</li>
 *           <li>remainder of floating point division ( fmod )</li>
 *       </ul>
 *      \par Example
 *      \code
 *      Array< double > preArray1( 10 ), preArray2( 10 ),
 *                      postArray1( 10 ), postArray2( 10 );
 *      double          operand( 2.0 );
 *
 *      // Fill Arrays "preArray1" and "preArray2" with content:
 *      ...
 *
 *      // Array "postArray1" becomes the power of Array "preArray1"
 *      // to "preArray2":
 *      postArray1 = pow( preArray1, preArray2 );
 *
 *      // Array "postArray2" becomes the power of Array "preArray1"
 *      // to the "operand":
 *      postArray2 = pow( preArray1, operand );
 *
 *      // The order of the operands is not important, so it is
 *      // also possible to write: postArray2 = pow( operand, preArray1 );
 *      \endcode
 *
 *       <li>inner product, outer product and scalar product of two
 *           arrays</li>
 *       <li>square and euclidian distance between two arrays</li>
 *       <li>sum, sum of absolute values, sum of square values and
 *           product of array values</li>
 *       <li>overall and interval minimum and maximum values of arrays</li>
 *       <li>input and output methods (input/output stream operator,
 *           prettyprint)</li>
 *       <li>fitting all values of an array to an interval (clip)</li>
 *  </ul>
 *
 *  \author  M. Kreutz
 *  \date    1995-01-01
 *
 *  \par Copyright (c) 1995, 1999:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      Array
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Array. This library is free software;
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
//===========================================================================


#ifndef __ARRAYOP_H
#define __ARRAYOP_H


#include <cmath>
#include <Array/Array.h>


//===================================================================
/*!
 *  \ingroup Compare
 *  \brief Returns "true" if the two array "v" and "w" are equal.
 *
 *  The two arrays are equal, if they have the same dimensions
 *  and all elements are equal.
 *
 *  \param v the first array
 *  \param w the second array
 *  \return "true", when \em v and \em w are equal,
 *          "false" otherwise
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
template < class T >
inline bool operator == (const Array< T >& v, const Array< T >& w)
{

	if (!v.samedim(w)) return false;
	typename Array<T>::iterator iterV, iterW;
	for (iterV = v.begin(), iterW = w.begin(); iterV != v.end();
			iterV++, iterW++) {
		if (*iterV != *iterW) return false;
	}
	return true;
}


//===================================================================
/*!
 *  \ingroup Compare
 *  \brief Returns "true" if array reference "v" and array "w" are equal.
 *
 *  \em v and \em w are equal, if they have the same dimensions
 *  and all elements are equal.
 *
 *  \param v the array reference
 *  \param w the array
 *  \return "true", when \em v and \em w are equal,
 *          "false" otherwise
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
template < class T >
inline bool operator == (const ArrayReference< T > v, const Array< T >& w)
{
	if (!v.samedim(w)) return false;
	typename Array<T>::iterator iterV, iterW;
	for (iterV = v.begin(), iterW = w.begin(); iterV != v.end();
			iterV++, iterW++) {
		if (*iterV != *iterW) return false;
	}
	return true;
}


//===================================================================
/*!
 *  \ingroup Compare
 *  \brief Returns "true" if array "v" and array reference "w" are equal.
 *
 *  \em v and \em w are equal, if they have the same dimensions
 *  and all elements are equal.
 *
 *  \param v the array
 *  \param w the array reference
 *  \return "true", when \em v and \em w are equal,
 *          "false" otherwise
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
template < class T >
inline bool operator == (const Array< T >& v, const ArrayReference< T > w)
{
	if (!v.samedim(w)) return false;
	typename Array<T>::iterator iterV, iterW;
	for (iterV = v.begin(), iterW = w.begin(); iterV != v.end();
			iterV++, iterW++) {
		if (*iterV != *iterW) return false;
	}
	return true;
}


//===================================================================
/*!
 *  \ingroup Compare
 *  \brief Returns "true" if array reference "v" and array reference "w" are equal.
 *
 *  \em v and \em w are equal, if they have the same dimensions
 *  and all elements are equal.
 *
 *  \param v the array reference
 *  \param w the array reference
 *  \return "true", when \em v and \em w are equal,
 *          "false" otherwise
 *
 */
template < class T >
inline bool operator == (const ArrayReference< T > v, const ArrayReference< T > w)
{
	if (!v.samedim(w)) return false;
	typename Array<T>::iterator iterV, iterW;
	for (iterV = v.begin(), iterW = w.begin(); iterV != v.end();
			iterV++, iterW++) {
		if (*iterV != *iterW) return false;
	}
	return true;
}


//===================================================================
/*!
 *  \ingroup Compare
 *  \brief Returns "true" if array "v" is less than array "w".
 *
 *  \em v is less than \em w, if \em v has less elements than \em w
 *  or both have the same number of elements and at least
 *  one element of \em v is less than the corresponding element of
 *  \em w.
 *
 *  \param v the first array
 *  \param w the second array
 *  \return "true", when \em v is less than \em w,
 *          "false" otherwise
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
template < class T >
inline bool operator < (const Array< T >& v, const Array< T >& w)
{
	if (v.nelem() == w.nelem()) {

		for (unsigned i = 0; i < v.nelem(); ++i)
			if (w.elem(i) < v.elem(i))
				return false;

			else if (v.elem(i) < w.elem(i))
				return true;

	}

	return v.nelem() < w.nelem();
}


//===================================================================
/*!
 *  \ingroup Compare
 *  \brief Returns "true" if the two array "v" and "w" are not equal.
 *
 *  The two arrays are not equal, if they have different dimensions
 *  or at least one element of \em v is different to the corresponding
 *  element of \em w.
 *
 *  \param v the first array
 *  \param w the second array
 *  \return "true", when \em v and \em w are not equal,
 *          "false" otherwise
 *
 *  \author  R. Alberts
 *  \date    2002-05-15
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
inline bool operator != (const Array< T >& v, const Array< T >& w)
{
	return !(v == w);
}



//===================================================================
/*!
 *  \ingroup Compare
 *  \brief Returns "true" if array reference "v" and array "w" are not equal.
 *
 *  \em v and \em w are not equal, if they have different dimensions
 *  or at least one element of \em v is not equal to the corresponding
 *  element of \em w.
 *
 *  \param v the array reference
 *  \param w the array
 *  \return "true", when \em v and \em w are not equal,
 *          "false" otherwise
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
template < class T >
inline bool operator != (const ArrayReference< T > v, const Array< T >& w)
{
	return !(v == w);
}

//===================================================================
/*!
 *  \ingroup Compare
 *  \brief Returns "true" if array "v" and array reference "w" are not equal.
 *
 *  \em v and \em w are not equal, if they have different dimensions
 *  or at least one element of \em v is not equal to the corresponding
 *  element of \em w.
 *
 *  \param v the array
 *  \param w the array reference
 *  \return "true", when \em v and \em w are not equal,
 *          "false" otherwise
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
template < class T >
inline bool operator != (const Array< T >& v, const ArrayReference< T > w)
{

	return !(v == w);
}


//===================================================================
/*!
 *  \ingroup Compare
 *  \brief Returns "true" if array reference "v" and array reference "w" are not equal.
 *
 *  \em v and \em w are not equal, if they have different dimensions
 *  or at least one element of \em v is not equal to the corresponding
 *  element of \em w.
 *
 *  \param v the array reference
 *  \param w the array reference
 *  \return "true", when \em v and \em w are not equal,
 *          "false" otherwise
 *
 */
template < class T >
inline bool operator != (const ArrayReference< T > v, const ArrayReference< T > w)
{
	return !(v == w);
}



//===================================================================
/*!
 *  \ingroup Math
 *  \brief Returns the inner product of the two arrays "v" and "w".
 *
 *  The inner product, also known as the scalar product of two
 *  vectors \f$x\f$ and \f$y\f$ with dimension \f$n\f$ is a scalar
 *  value \f$\alpha\f$ given as
 *
 *  \f$
 *  \alpha = x^T \cdot y = \sum_{i=1}^n x_i \cdot y_i
 *  \f$
 *
 *  You can also use this method for vectors as one-dimensional arrays, but
 *  this method is a generalization of the scalar product and is especially
 *  written for N-dimensional arrays with \f$N \geq 2\f$. <br>
 *  Therefore the method will always return an array of scalar values.
 *  If you want to calculate the "original" scalar product for two
 *  vectors, it is more handy to use the method #scalarProduct, because
 *  this method will return a single scalar value. <br>
 *  If you use arrays that represent two matrices \f$A\f$ with dimensions
 *  \f$m_1 \times \dots \times m_k\f$ and \f$B\f$ with dimensions
 *  \f$n_1 \times \dots \times n_l\f$ with \f$l, k \geq 2\f$ and
 *  \f$m_k = n_1\f$ then the result is a matrix \f$C\f$
 *  with dimensions \f$m_1 \times \dots \times m_{k-1} \times n_2 \times
 *  \dots \times n_l\f$. Each scalar value of \f$C\f$ is calculated
 *  by multiplying each "row" of matrix \f$A\f$ with each column
 *  of matrix \f$B\f$, where "row" means always the first dimension
 *  of a matrix and "column" the last dimension. The multiplication
 *  results are then added together.
 *
 *  \param v the first array.
 *  \param w the second array.
 *  \return an array containing the inner product of \em v and \em w
 *  \throw  check_exception the type of the exception will
 *          be "size mismatch" and means that you've called the method
 *          with two one-dimensional arrays, but these vectors
 *          have different sizes
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      2002-03-21, ra: Method crashed, when two one-dimensional
 *      arrays were used. Now the method "scalarProduct"
 *      is called in this case and the result is casted into an array.
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
inline Array< T > innerProduct(const Array< T >& v, const Array< T >& w)
{
	unsigned vd = v.ndim();
	unsigned wd = w.ndim();
	unsigned zd = vd + wd - 2;
	unsigned i, j, k;
	T        t;
	Array< T > z;

	//
	// handle scalar values
	//
	if (vd == 0 || wd == 0) {
		if (vd == 0 && v.nelem() == 1) {
			return w * v.elem(0);
		}
		else if (wd == 0 && w.nelem() == 1) {
			return w.elem(0) * v;
		}
		else {
			return Array< T >();
		}
	}
	// two vectors (the "original" inner product):
	else if (vd == 1 && wd == 1) {
		t = scalarProduct(v, w);
		z.resize(1);
		z(0) = t;
		return Array< T >(z, true);
	}

	SIZE_CHECK(v.dim(vd - 1) == w.dim(0))

	std::vector< unsigned > dim(zd);

	for (i = j = 0; j < vd - 1; dim[ i++ ] = v.dim(j++)) ;
	for (j = 1; j < wd  ; dim[ i++ ] = w.dim(j++)) ;

	z.resize(dim);

	std::vector< unsigned > vdim(vd);
	std::vector< unsigned > wdim(wd);

	std::vector< unsigned > zdim(zd, 0U);

	do {
		for (i = j = 0; j < vd - 1; vdim[ j++ ] = zdim[ i++ ]) ;
		for (j = 1; j < wd  ; wdim[ j++ ] = zdim[ i++ ]) ;
		for (t = T(0), k = 0; k < w.dim(0); k++) {
			vdim[ vd-1 ] = wdim[ 0 ] = k;
			t += v(vdim) * w(wdim);
		}
		z(zdim) = t;

		for (i = 0; i < zd && ++zdim[ i ] >= z.dim(i); zdim[ i++ ] = 0) ;
	}
	while (i < zd);

	return Array< T >(z, true);
}


//===================================================================
/*!
 *  \ingroup Math
 *  \brief Returns the outer product of the two arrays "v" and "w".
 *
 *  The outer product, also known as the dyadic product of a
 *  vector \f$x\f$ with dimension \f$m\f$ and a vector \f$y\f$
 *  with dimension \f$n\f$ is a matrix \f$A\f$ given as
 *
 *  \f$
 *  A = x \cdot y^T
 *  \f$
 *
 *  where \f$A\f$ is a \f$m \times n\f$ matrix with \f$a_{ij} = x_i
 *  \cdot y_j\f$ for \f$i = 1, \dots, m\f$ and \f$j = 1, \dots, n\f$. <br>
 *  This method is a generalization, it works not only for
 *  vectors as one-dimensional arrays, but also for N-dimensional
 *  arrays with \f$N \geq 2\f$. <br>
 *  Given two matrices \f$A\f$ with dimensions
 *  \f$m_1 \times \dots \times m_k\f$
 *  and \f$B\f$ with dimensions \f$n_1 \times \dots \times n_l\f$ the
 *  result is an matrix \f$C\f$ with dimensions
 *  \f$m_1 \times \dots \times m_k \times n_1 \times \dots \times n_l\f$,
 *  where \f$c_{i_1 \dots i_k i_{k+1} \dots i_{k+l}} = a_{i_1 \dots i_k} \cdot b_{i_{k+1} \dots i_{k+l}}\f$ for \f$i_1 = 1, \dots, m_1\mbox{;\ } \dots \mbox{;\ }i_k = 1, \dots, m_k\mbox{;\ }i_{k+1} = 1, \dots, n_1\mbox{;\ } \dots \mbox{;\ }i_{k+l} = 1, \dots, n_l\f$
 *
 *  \param v the first array.
 *  \param w the second array.
 *  \return an array containing the outer product of \em v and \em w
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
template < class T >
inline Array< T > outerProduct(const Array< T >& v, const Array< T >& w)
{
	unsigned vd = v.ndim();
	unsigned wd = w.ndim();
	unsigned zd;

	//
	// handle scalar values
	//
	if (vd == 0 || wd == 0) {
		if (vd == 0 && v.nelem() == 1) {
			return w * v.elem(0);
		}
		else if (wd == 0 && w.nelem() == 1) {
			return w.elem(0) * v;
		}
		else {
			return Array< T >();
		}
	}

	unsigned i, j;

	std::vector< unsigned > dim(zd = vd + wd);

	for (i = j = 0; j < vd; dim[ i++ ] = v.dim(j++));
	for (j = 0; j < wd; dim[ i++ ] = w.dim(j++));

	Array< T > z;
	z.resize(dim);

	std::vector< unsigned > vdim(vd);
	std::vector< unsigned > wdim(wd);
	std::vector< unsigned > zdim(zd, 0U);

	do {
		for (i = j = 0; j < vd; vdim[ j++ ] = zdim[ i++ ]);

		for (j = 0; j < wd; wdim[ j++ ] = zdim[ i++ ]);

		z(zdim) = v(vdim) * w(wdim);

		for (i = 0; i < zd && ++zdim[ i ] >= z.dim(i); zdim[ i++ ] = 0);
	}
	while (i < zd);

	return Array< T >(z, true);
}


//===================================================================
/*!
 *  \ingroup Math
 *  \brief Returns the scalar product of the two arrays "v" and "w".
 *
 *  The scalar product, also known as the inner product of two
 *  vectors \f$x\f$ and \f$y\f$ with dimension \f$n\f$ is a scalar
 *  value \f$\alpha\f$ given as
 *
 *  \f$
 *  \alpha = x^T \cdot y = \sum_{i=1}^n x_i \cdot y_i
 *  \f$
 *
 *  Here, the two arrays \em v and \em w are interpreted as
 *  one-dimensional arrays, by using the element vectors
 *  Array::e of both. So you can use this method for two N-dimensional
 *  arrays with \f$N \geq 2\f$, but will always receive a single
 *  scalar value. <br>
 *  If you want to calculate the inner product for those arrays with
 *  respect to the number of dimensions, use method #innerProduct.
 *
 *  \param v the first array.
 *  \param w the second array.
 *  \return the scalar product \f$\alpha\f$ of the element vectors
 *          of the two arrays
 *  \throw  check_exception the type of the exception will
 *          be "size mismatch" and means that the two vectors
 *          have different sizes
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
template < class T >
inline T scalarProduct(const Array< T >& v, const Array< T >& w)
{
	SIZE_CHECK(v.samedim(w))

	T t(0);
	for (unsigned i = v.nelem(); i--; t += v.elem(i) * w.elem(i));
	return t;
}


//===================================================================
/*!
 *  \ingroup Math
 *  \brief Returns the square distance between the two arrays "v" and "w".
 *
 *  The square distance \f$d\f$ is evaluated as
 *
 *  \f$
 *      d = \sum_{i=0}^{N-1} {(v_i - w_i)}^2
 *  \f$
 *
 *  where \f$N\f$ is the total number of elements in each array
 *  and the single elements are taken in the order as they are stored
 *  in the element vector Array::e. <br>
 *  Notice, that the two arrays must have the same number of elements.
 *
 *  \param v the first array.
 *  \param w the second array.
 *  \return the square distance \f$d\f$
 *  \throw check_exception the type of the exception will be
 *         "size mismatch" and indicates that the two arrays
 *         \em v and \em w have different sizes
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
template < class T >
inline T sqrDistance(const Array< T >& v, const Array< T >& w)
{
	SIZE_CHECK(v.samedim(w))

	T d, t(0);
	for (unsigned i = v.nelem(); i--;) {
		d  = v.elem(i) - w.elem(i);
		t += d * d;
	}

	return t;
}


//===================================================================
/*!
 *  \ingroup Math
 *  \brief Returns the euclidian distance between the two arrays "v" and "w".
 *
 *  The euclidian distance \f$d\f$ is evaluated as
 *
 *  \f$
 *      d = \sqrt{\sum_{i=0}^{N-1} {(v_i - w_i)}^2}
 *  \f$
 *
 *  where \f$N\f$ is the total number of elements in each array
 *  and the single elements are taken in the order as they are stored
 *  in the element vector Array::e. <br>
 *  Notice, that the two arrays must have the same number of elements.
 *
 *  \param v the first array.
 *  \param w the second array.
 *  \return the euclidian distance \f$d\f$
 *  \throw check_exception the type of the exception will be
 *         "size mismatch" and indicates that the two arrays
 *         \em v and \em w have different sizes
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
template < class T >
inline T euclidianDistance(const Array< T >& v, const Array< T >& w)
{
	return sqrt(sqrDistance(v, w));
}


//===================================================================
/*!
 *  \ingroup Math
 *  \brief Returns the sum of all values in array "v".
 *
 *  The sum is evaluated as
 *
 *  \f$
 *      t = \sum_{i=0}^{N-1} v_i
 *  \f$
 *
 *  where \f$N\f$ is the total number of values in the array
 *  and the single values are taken in the order as they are stored
 *  in the element vector Array::e. <br>
 *
 *  \param v the array of which the sum will be calculated
 *  \return the sum \f$t\f$
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
template < class T >
inline T sum(const Array< T >& v)
{
	T t(0);
	for (unsigned i = v.nelem(); i--; t += v.elem(i));
	return t;
}

//===================================================================
/*!
 *  \ingroup Math
 *  \brief Returns the sum of all absolute values in array "v".
 *
 *  The sum is evaluated as
 *
 *  \f$
 *      t = \sum_{i=0}^{N-1} |v_i|
 *  \f$
 *
 *  where \f$N\f$ is the total number of values in the array
 *  and the single values are taken in the order as they are stored
 *  in the element vector Array::e. <br>
 *
 *  \param v the array of which the sum will be calculated
 *  \return the sum \f$t\f$
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
template < class T >
inline T sumOfAbs(const Array< T >& v)
{
	T t(0);
	for (unsigned i = v.nelem(); i--; t += fabs(v.elem(i)));
	return t;
}


//===================================================================
/*!
 *  \ingroup Math
 *  \brief Returns the sum of all square values in array "v".
 *
 *  The sum is evaluated as
 *
 *  \f$
 *      t = \sum_{i=0}^{N-1} v_i^2
 *  \f$
 *
 *  where \f$N\f$ is the total number of values in the array
 *  and the single values are taken in the order as they are stored
 *  in the element vector Array::e. <br>
 *
 *  \param v the array of which the sum will be calculated
 *  \return the sum \f$t\f$
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
template < class T >
inline T sumOfSqr(const Array< T >& v)
{
	T t(0);
	for (unsigned i = v.nelem(); i--; t += v.elem(i) * v.elem(i));
	return t;
}


//===================================================================
/*!
 *  \ingroup Math
 *  \brief Returns the product of all values in array "v".
 *
 *  The product is evaluated as
 *
 *  \f$
 *      t = \prod_{i=0}^{N-1} v_i
 *  \f$
 *
 *  where \f$N\f$ is the total number of values in the array
 *  and the single values are taken in the order as they are stored
 *  in the element vector Array::e. <br>
 *
 *  \param v the array of which the product will be calculated
 *  \return the product \f$t\f$
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
template < class T >
inline T product(const Array< T >& v)
{
	T t(1);
	for (unsigned i = v.nelem(); i--; t *= v.elem(i));
	return t;
}


//===================================================================
/*!
 *  \ingroup Extract
 *  \brief Returns the minimum value stored in array "v".
 *
 *  \param v the array of which the minimum value will be returned
 *  \return the minimum value of \em v
 *  \throw check_exception the type of the exception will be
 *         "range check error" and indicates that \em v contains
 *         no elements
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
template < class T >
inline T minElement(const Array< T >& v)
{
	RANGE_CHECK(v.nelem() > 0)

	T t(v.elem(0));

	for (unsigned i = 1; i < v.nelem(); ++i)
		if (v.elem(i) < t) t = v.elem(i);

	return t;
}


//===================================================================
/*!
 *  \ingroup Extract
 *  \brief Returns the minimum value stored in array "v" between
 *         the positions "start" and "end" - 1.
 *
 *  The positions of the values apply to the positions of the values
 *  in the element vector Array::e. <br>
 *  Besides returning the minimum value, the index (i.e. position
 *  in the element vector) of the minimum value is stored.
 *
 *  \param v     the array of which the minimum value will be returned
 *  \param ind   index of the determined minimum value in the
 *               element vector
 *  \param start the first position in the element vector where
 *               the method will search for the minimum value,
 *               the default value is "0"
 *  \param end   the first position in the element vector where
 *               the method will not search for the minimum value any more.
 *               If \em end is set to "0" (the default value),
 *               then the number of elements of the element vector
 *               is taken for \em end. \em end must be greater than
 *               or equal to \em start, otherwise it will be
 *               set to the value of start
 *  \return the minimum value of \em v in the interval [start, end[
 *  \throw check_exception the type of the exception will be
 *         "range check error" and indicates that \em v contains no
 *         elements or that \em start or \em end exceed the number
 *         of elements in \em v
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      2002-03-13, ra: Last RANGE CHECK set into comments,
 *      because otherwise you can not use the default value "0" for "end"
 *      (and its replacement by the number of elements of "v") if "start"
 *      is set to a value >= 1. To avoid errors,
 *      the value of "end" will be now set to "start", if its less than
 *      the start value
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
inline T minElement(const Array< T >& v, unsigned & ind,
			 unsigned start = 0, unsigned end = 0)
{
	RANGE_CHECK(v.nelem() > 0)
	RANGE_CHECK(v.nelem() > start)
	RANGE_CHECK(v.nelem() >= end)

	RANGE_CHECK(end        >= start)

	T t(v.elem(start));
	ind = start;
	if (end == 0) {
		end = v.nelem();
	}
	for (unsigned i = start; i < end; ++i)
		if (v.elem(i) < t) {
			t   = v.elem(i);
			ind = long(i);
		}

	return t;
}


//===================================================================
/*!
 *  \ingroup Extract
 *  \brief Returns the maximum value stored in array "v".
 *
 *  \param v the array of which the maximum value will be returned
 *  \return the maximum value of \em v
 *  \throw check_exception the type of the exception will be
 *         "range check error" and indicates that \em v contains
 *         no elements
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
template < class T >
inline T maxElement(const Array< T >& v)
{
	RANGE_CHECK(v.nelem() > 0)

	T t(v.elem(0));

	for (unsigned i = 1; i < v.nelem(); ++i)
		if (v.elem(i) > t) t = v.elem(i);

	return t;
}


//===================================================================
/*!
 *  \ingroup Extract
 *  \brief Returns the maximum value stored in array "v" between
 *         the positions "start" and "end" - 1.
 *
 *  The positions of the values apply to the positions of the values
 *  in the element vector Array::e. <br>
 *  Besides returning the maximum value, the index (i.e. position
 *  in the element vector) of the maximum value is stored.
 *
 *  \param v     the array of which the maximum value will be returned
 *  \param ind   index of the determined maximum value in the
 *               element vector
 *  \param start the first position in the element vector where
 *               the method will search for the maximum value,
 *               the default value is "0"
 *  \param end   the first position in the element vector where
 *               the method will not search for the maximum value any more.
 *               If \em end is set to "0" (the default value),
 *               then the number of elements of the element vector
 *               is taken for \em end. \em end must be greater
 *               or equal than \em start, otherwise it will be
 *               set to the value of start
 *  \return the maximum value of \em v in the interval [start, end[
 *  \throw check_exception the type of the exception will be
 *         "range check error" and indicates that \em v contains no
 *         elements or that \em start or \em end exceed the number
 *         of elements in \em v
 *
 *  \author  M. Kreutz
 *  \date    1995
 *
 *  \par Changes
 *      2002-03-13, ra: Last RANGE CHECK set into comments,
 *      because otherwise you can not use the default value "0" for "end"
 *      (and its replacement by the number of elements of "v") if "start"
 *      is set to a value >= 1. To avoid errors,
 *      the value of "end" will be now set to "start", if its less than
 *      the start value
 *
 *  \par Status
 *      stable
 *
 */
template < class T >
inline T maxElement(const Array< T >& v, unsigned & ind,
			 unsigned start = 0, unsigned end = 0)
{
	RANGE_CHECK(v.nelem() > 0)
	RANGE_CHECK(v.nelem() > start)
	RANGE_CHECK(v.nelem() >= end)

	RANGE_CHECK(end        >= start)

	T t(v.elem(start));
	ind = start;
	if (end == 0) {
		end = v.nelem();
	}
	for (unsigned i = start; i < end; ++i)
		if (v.elem(i) > t) {
			t   = v.elem(i);
			ind = long(i);
		}

	return t;
}


//===================================================================
/*!
 *  \ingroup Extract
 *  \brief Determines the minimum and maximum values stored in array "v".
 *
 *  \param v the array of which the minimum and maximum values will be
 *           determined
 *  \param minVal the minimum value of the array
 *  \param maxVal the maximum value of the array
 *  \return none
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
template < class T >
inline void minmaxElement(const Array< T >& v, T& minVal, T& maxVal)
{
	if (v.nelem() > 0) {
		minVal = maxVal = v.elem(0);

		for (unsigned i = 1; i < v.nelem(); ++i) {
			if (v.elem(i) < minVal)
				minVal = v.elem(i);
			else if (v.elem(i) > maxVal)
				maxVal = v.elem(i);
		}
	}
}


//===================================================================
/*!
 *  \ingroup InOut
 *  \brief Output of the content of array "v" to output stream "os"
 *         in a form, where you can better see the array's structure.
 *
 *  This output method works recursively and begins each
 *  new position in the current dimension with a left
 *  parenthesis and ends each single position in the current
 *  dimension with a right parenthesis. <br>
 *  Each single position in the last dimension will be written
 *  in its own line and indented by a number of white spaces.
 *
 *  \par Example
 *  \code
 *  #include "Array/ArrayOp.h"
 *
 *  void main()
 *  {
 *      // test vector:
 *      Array< double > test( 2, 2, 4 );
 *
 *      // Filling the test vector with content:
 *      test( 0, 0, 0 ) = 1.;
 *      test( 0, 0, 1 ) = 2.;
 *      test( 0, 0, 2 ) = 3.;
 *      test( 0, 0, 3 ) = 4.;
 *      test( 0, 1, 0 ) = 5.;
 *      test( 0, 1, 1 ) = 6.;
 *      test( 0, 1, 2 ) = 7.;
 *      test( 0, 1, 3 ) = 8.;
 *      test( 1, 0, 0 ) = 9.;
 *      test( 1, 0, 1 ) = 10.;
 *      test( 1, 0, 2 ) = 11.;
 *      test( 1, 0, 3 ) = 12.;
 *      test( 1, 1, 0 ) = 13.;
 *      test( 1, 1, 1 ) = 14.;
 *      test( 1, 1, 2 ) = 15.;
 *      test( 1, 1, 3 ) = 16.;
 *
 *      // Formatted output of the test vector:
 *      prettyprint( cout, test );
 *  }
 *  \endcode
 *
 *  This example program will produce the output:
 *
 *  \f$
 *  \mbox{\ }\\ \noindent
 *  \mbox{(\ (\ (\ 1 2 3 4\ )}\\
 *  \mbox{\ \ \ \ (\ 5 6 7 8\ )\ )}\\
 *  \mbox{\ \ (\ (\ 9 10 11 12\ )}\\
 *  \mbox{\ \ \ \ (\ 13 14 15 16\ )\ )\ )}\\
 *  \f$
 *
 *  \param os    the output stream to which the array's content will
 *               be written
 *  \param v     the array whose content will be written to \em os
 *  \param depth the number of spaces, that will be used as indent.
 *               This indent will be increased by one for
 *               each subdimension (recursion), the default value
 *               at the first call is "0"
 *  \return "true" if there is another subdimension, that must be
 *          processed recursively, "false" otherwise
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
template < class T >
bool prettyprint(std::ostream& os, const Array< T >& v, unsigned depth = 0)
{
	unsigned i, j;
	bool     nl;

	if (v.ndim() == 0) {
		os << v.elem(0) << " ";
		return false;
	}
	else if (v.ndim() == 1) {
		os << "( ";
		for (i  = 0; i < v.dim(0); i++)
			prettyprint(os, v[ i ], depth + 1);
		os << ") ";
		return true;
	}
	else {
		os << "( ";
		i = 0;
		nl = prettyprint(os, v[ i++ ], depth + 1);
		while (i < v.dim(0)) {
			if (nl) {
				os << "\n  ";
				for (j = 0; j < depth; j++) os << "  ";
			}
			nl = prettyprint(os, v[ i++ ], depth + 1);
		}
		os << ") ";
		return nl;
	}
}

#ifdef __ARRAY_NO_GENERIC_IOSTREAM


//===================================================================
/*!
 *  \ingroup InOut
 *  \brief Reads information about the structure of array "a"
 *         and the array's content from input stream "os".
 *
 *  First the type (represented by a single letter) and the dimension
 *  sizes (in a list) will be read. <br>
 *  Then all array elements will be read and stored in the element vector
 *  Array::d. <br>
 *  For the necessary format of the information, please see the
 *  example program of the output stream operator. <br>
 *  This method can only be used if the preprocessor flag
 *  #__ARRAY_NO_GENERIC_IOSTREAM is unset at the beginning of
 *  file Array.h.
 *
 *  \param is the input stream from which the array's structure and
 *            content are read
 *  \param a  the array, whose structure and content are read from
 *            \em is
 *  \return reference to the input stream
 *  \throw check_exception the type of the exception will be "type mismatch"
 *         and indicates that the first line of data read with the type and
 *         dimension information of the array has the wrong format. This
 *         includes that no whitespaces are allowed in the dimension
 *         sizes list and no whitespaces between the list and the
 *         enclosing parentheses
 *
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
template < class T >
inline std::istream& operator >> (std::istream& is, Array< T >& a)
{
	const unsigned MaxLen = 1024;
	char           s[ MaxLen ];
	char*          p;
	unsigned       i;
	std::vector< unsigned > idx;

	is >> std::ws;
	is.getline(s, MaxLen);

	p = strchr(s, '>');
	if (p && *p) p++;
	if (p && *p) p++;

	TYPE_CHECK(p && *p)
	if (p && *p) {
		do {
			if (*p == ',') p++;
			idx.push_back(unsigned(strtol(p, &p, 10)));
		}
		while (p && *p == ',');

		TYPE_CHECK(p && *p == ')')

		a.resize(idx);

		for (i = 0; i < a.nelem() && is.good(); i++)
			is >> a.elem(i);
	}

	return is >> std::ws;
}


//===================================================================
/*!
 *  \ingroup InOut
 *  \brief Writes information about the structure of array "a"
 *         and the array's content to output stream "os".
 *
 *  First the type (represented by a single letter) and the dimension
 *  sizes (in a list) will be written to one line. <br>
 *  Then all array elements will be written to the next
 *  line, in the order as they are stored in the element vector
 *  Array::d, where the single elements are separated by a tabulator
 *  character. <br>
 *  This method can only be used if the preprocessor flag
 *  #__ARRAY_NO_GENERIC_IOSTREAM is unset at the beginning of
 *  file Array.h.
 *
 *  \par Example
 *  \code
 *  #include "Array/ArrayOp.h"
 *
 *  void main()
 *  {
 *      // test array:
 *      Array< double > test( 2, 2 );
 *
 *      // Filling the array with content:
 *      test( 0, 0 ) = 1.;
 *      test( 0, 1 ) = 2.;
 *      test( 1, 0 ) = 3.;
 *      test( 1, 1 ) = 4.;
 *
 *      // Output of array structure and content:
 *      cout << test << endl;
 *  }
 *  \endcode
 *
 *  This example program will produce the output:
 *
 *  \f$
 *  \mbox{\ }\\ \noindent
 *  \mbox{Array}<\mbox{d}>\mbox{(2,2)}\\
 *  \mbox{1\ \ \ \ 2\ \ \ \ 3\ \ \ \ 4}
 *  \f$
 *
 *  \param os the output stream to which the array's structure and
 *            content are written to
 *  \param a  the array, whose structure and content are written to
 *            \em os
 *  \return reference to the output stream
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
template < class T >
inline std::ostream& operator << (std::ostream& os, const Array< T >& a)
{
	unsigned i;

	os << "Array<" << typeid(T).name() << ">(";
	for (i = 0; i < a.ndim(); ++i) {
		if (i) os << ',';
		os << a.dim(i);
	}
	os << ")\n";

	for (i = 0; i < a.nelem(); ++i) {
		if (i) os << '\t';
		os << a.elem(i);
	}

	return os << std::endl;
}
#endif // __ARRAY_NO_GENERIC_IOSTREAM


//===================================================================
/*!
 *  \brief Each value of the array "valArrA" will be fit into
 *         the interval [lowerBoundA, upperBoundA].
 *
 *  Each value that is less than the lower bound will be set to
 *  the lower bound value and each value that is greater than
 *  the upper bound will be set to the upper bound value. <br>
 *  All other values are left untouched.
 *
 *  \param valArrA     the array that will be processed
 *  \param lowerBoundA the lower bound value
 *  \param upperBoundA the upper bound value
 *  \return none
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
template < class T >
inline void clip
(
	Array< T >& valArrA,
	const T&    lowerBoundA,
	const T&    upperBoundA
)
{
	for (unsigned i = valArrA.nelem(); i--;) {
		if (valArrA.elem(i) < lowerBoundA) {
			valArrA.elem(i) = lowerBoundA;
		}
		else if (valArrA.elem(i) > upperBoundA) {
			valArrA.elem(i) = upperBoundA;
		}
	}
}


#ifndef DOXYGEN
// Please don't remove the line below!
//
#define UnaryOperator( op )                                                \
template < class T >                                                       \
inline Array< T > operator op ( const Array< T >& v )                      \
{                                                                          \
   Array< T > z;                                                           \
                                                                           \
   z.resize( v );                                                          \
   typename Array<T>::iterator iterZ;                                      \
   typename Array<T>::iterator iterV;                                      \
   for(iterV = v.begin(), iterZ = z.begin(); iterV != v.end();             \
		    iterV++, iterZ++)                                      \
   {                                                                       \
		    *iterZ = op *iterV;                                    \
   }                                                                       \
                                                                           \
   return Array< T >( z, true );                                           \
}

// Negation
UnaryOperator(!)

// Complement
UnaryOperator(~)

// Positive
UnaryOperator( +)

// Negative
UnaryOperator(-)

#undef UnaryOperator



#define BinaryOperator( op )                                               \
template < class T >                                                       \
inline Array< T > operator op ( const Array< T >& v, const Array< T >& w ) \
{                                                                          \
   SIZE_CHECK( v.samedim( w ) )                                            \
                                                                           \
   Array< T > z;                                                           \
   z.resize( v );                                                          \
                                                                           \
   typename Array<T>::iterator iterZ;                                      \
   typename Array<T>::iterator iterW;                                      \
   typename Array<T>::iterator iterV;                                      \
   for(iterV = v.begin(), iterW = w.begin(), iterZ = z.begin();            \
		    iterV != v.end(); iterV++, iterW++, iterZ++)           \
   {                                                                       \
		    *iterZ = *iterV op *iterW;                             \
   }                                                                       \
                                                                           \
   return Array< T >( z, true );                                           \
}                                                                          \
                                                                           \
template < class T >                                                       \
inline Array< T > operator op ( const Array< T >& v, T w )                 \
{                                                                          \
   Array< T > z;                                                           \
   z.resize( v );                                                          \
                                                                           \
   typename Array<T>::iterator iterZ;                                      \
   typename Array<T>::iterator iterV;                                      \
   for(iterV = v.begin(), iterZ = z.begin();                               \
		    iterV != v.end(); iterV++, iterZ++)                    \
   {                                                                       \
		    *iterZ = *iterV op w;                                  \
   }                                                                       \
                                                                           \
   return Array< T >( z, true );                                           \
}                                                                          \
                                                                           \
template < class T >                                                       \
inline Array< T > operator op ( T v, const Array< T >& w )                 \
{                                                                          \
   Array< T > z;                                                           \
   z.resize( w );                                                          \
                                                                           \
   typename Array<T>::iterator iterZ;                                      \
   typename Array<T>::iterator iterW;                                      \
   for(iterW = w.begin(), iterZ = z.begin();                               \
		    iterW != w.end(); iterW++, iterZ++)                    \
   {                                                                       \
		    *iterZ = v op *iterW;                                  \
   }                                                                       \
                                                                           \
   return Array< T >( z, true );                                           \
}


// Logical OR
BinaryOperator( ||)


// Logical AND
BinaryOperator( &&)


// Bitwise OR
BinaryOperator( |)


// Bitwise exclusive OR
BinaryOperator( ^)


// Bitwise AND
BinaryOperator(&)


// Addition
BinaryOperator( +)


// Subtraction
BinaryOperator(-)


// Multiplication
BinaryOperator(*)


// Division
BinaryOperator( /)


// Modulo
BinaryOperator( %)

#undef BinaryOperator


#define CompoundAssignmentOperator( op )                                   \
template < class T, class S >                                              \
inline Array< T >& operator op ( Array< T >& v, const Array< S >& w )      \
{                                                                          \
   SIZE_CHECK( v.samedim( w ) )                                            \
                                                                           \
   typename Array<S>::iterator iterW;                                      \
   typename Array<T>::iterator iterV;                                      \
   for(iterV = v.begin(), iterW = w.begin();                               \
		    iterV != v.end(); iterV++, iterW++)                    \
   {                                                                       \
		    *iterV op *iterW;                                      \
   }                                                                       \
                                                                           \
   return v;                                                               \
}                                                                          \
                                                                           \
template < class T >                                                       \
inline Array< T >& operator op ( Array< T >& v, T w )                      \
{                                                                          \
   typename Array<T>::iterator iterV;                                      \
   for(iterV = v.begin(); iterV != v.end(); iterV++)                       \
   {                                                                       \
		    *iterV op w;                                           \
   }                                                                       \
                                                                           \
   return v;                                                               \
}                                                                          \
                                                                           \
template < class T >                                                       \
inline ArrayReference< T > operator op ( ArrayReference< T > v,            \
                                         const Array< T >& w )             \
{                                                                          \
   SIZE_CHECK( v.samedim( w ) )                                            \
                                                                           \
   typename Array<T>::iterator iterW;                                      \
   typename Array<T>::iterator iterV;                                      \
   for(iterV = v.begin(), iterW = w.begin();                               \
		    iterV != v.end(); iterV++, iterW++)                    \
   {                                                                       \
		    *iterV op *iterW;                                      \
   }                                                                       \
                                                                           \
   return v;                                                               \
}                                                                          \
                                                                           \
template < class T >                                                       \
inline ArrayReference< T > operator op ( ArrayReference< T > v, T w )      \
{                                                                          \
   typename Array<T>::iterator iterV;                                      \
   for(iterV = v.begin(); iterV != v.end(); iterV++)                       \
   {                                                                       \
		    *iterV op w;                                           \
   }                                                                       \
                                                                           \
   return v;                                                               \
}


// OR into
CompoundAssignmentOperator( |=)


// Exclusive OR into
CompoundAssignmentOperator( ^=)


// AND into
CompoundAssignmentOperator( &=)


// Increase
CompoundAssignmentOperator( +=)


// Decrease
CompoundAssignmentOperator( -=)


// Multiply by
CompoundAssignmentOperator( *=)


// Divide by
CompoundAssignmentOperator( /=)


// Remainder
CompoundAssignmentOperator( %=)

#undef CompoundAssignmentOperator


#define UnaryFunction( func )                                              \
template < class T >                                                       \
inline Array< T > func ( const Array< T >& v )                             \
{                                                                          \
   Array< T > z;                                                           \
   z.resize( v );                                                          \
                                                                           \
   typename Array<T>::iterator iterZ;                                      \
   typename Array<T>::iterator iterV;                                      \
   for(iterV = v.begin(), iterZ = z.begin(); iterV != v.end();             \
		    iterV++, iterZ++)                                      \
   {                                                                       \
		    *iterZ = func( *iterV );                               \
   }                                                                       \
                                                                           \
   return Array< T >( z, true );                                           \
}


// Trigonometric Functions:


// Arcus Cosinus
UnaryFunction(acos)


// Arcus Sinus
UnaryFunction(asin)


// Arcus Tangens
UnaryFunction(atan)


// Cosinus
UnaryFunction(cos)


// Sinus
UnaryFunction(sin)


// Tangens
UnaryFunction(tan)



// Hyperbolic Functions:


// Cosinus hyperbolicus
UnaryFunction(cosh)


// Sinus hyperbolicus
UnaryFunction(sinh)


// Tangens hyperbolicus
UnaryFunction(tanh)


// Arcus Cosinus hyperbolicus
UnaryFunction(acosh)


// Arcus Sinus hyperbolicus
UnaryFunction(asinh)


// Arcus Tangens hyperbolicus
UnaryFunction(atanh)



// Exponential and Logarithmic Functions:


// Exponential function
UnaryFunction(exp)


// Logarithmus naturalis
UnaryFunction(log)


// Logarithmus decimalis
UnaryFunction(log10)



// Power Functions:


// Square root
UnaryFunction(sqrt)


// Cubic root
UnaryFunction(cbrt)



// Nearest Integer and Absolute Value:


// Round up value
UnaryFunction(ceil)


// Absolute value of floating point
UnaryFunction(fabs)


// Round down value
UnaryFunction(floor)

#undef UnaryFunction


#define BinaryFunction( func )                                             \
template < class T >                                                       \
inline Array< T > func( const Array< T >& v, const Array< T >& w )         \
{                                                                          \
   SIZE_CHECK( v.samedim( w ) )                                            \
                                                                           \
   Array< T > z;                                                           \
   z.resize( v );                                                          \
                                                                           \
   for( unsigned i = z.nelem( ); i--; )                                    \
       z.elem( i ) = func( v.elem( i ), w.elem( i ) );                     \
                                                                           \
   return Array< T >( z, true );                                           \
}                                                                          \
                                                                           \
template < class T >                                                       \
inline Array< T > func( const Array< T >& v, T w )                         \
{                                                                          \
   Array< T > z;                                                           \
   z.resize( v );                                                          \
                                                                           \
   for( unsigned i = z.nelem( ); i--; )                                    \
       z.elem( i ) = func( v.elem( i ), w );                               \
                                                                           \
   return Array< T >( z, true );                                           \
}                                                                          \
                                                                           \
template < class T >                                                       \
inline Array< T > func( T v, const Array< T >& w )                         \
{                                                                          \
   Array< T > z;                                                           \
   z.resize( w );                                                          \
                                                                           \
   for( unsigned i = z.nelem( ); i--; )                                    \
       z.elem( i ) = func( v, w.elem( i ) );                               \
                                                                           \
   return Array< T >( z, true );                                           \
}




// Arcus Tangens for two parameters:
BinaryFunction(atan2)


// Numeric power
BinaryFunction(pow)


// Remainder of floating point division
BinaryFunction(fmod)

#undef BinaryFunction
//
// Please don't remove the line above!
#endif




#endif /* !__ARRAYOP_H */










