//===========================================================================
/*!
 *  \file ArrayTable.h
 *
 *  \brief Contains a class for defining arrays, comparable to Array, but
 *         more efficient by using additional pointers as in class
 *         Array2D, but not only for 2-dimensional arrays, but also for
 *         3-dimensional arrays.
 *
 *  \author  M. Toussaint
 *  \date    2000-09-01
 *
 *  \par Copyright (c) 1999-2001:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-27974<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  \par Project:
 *      Array
 *
 *
 *  <BR>
 * 
 *
 *  <BR><HR>
 *
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


#ifndef __ARRAY_TABLE_H
#define __ARRAY_TABLE_H

#include "Array/Array.h"

//===========================================================================
/*!
 *  \brief Class for defining arrays, comparable to Array, but
 *         more efficient by using additional pointers as in class
 *         Array2D, but not only for 2-dimensional arrays, but also for
 *         3-dimensional arrays.
 *
 *  This class enhances the Array class to allow for a
 *  faster reference on the elements. In case of 1D arrays nothing
 *  changes. In case of 2D arrays it stores additional pointers which
 *  reference to a block of elements belonging to a fixed first index -
 *  thus avoiding the use of multiplication during reference in trade of
 *  consuming more memory (comparable to the Array2D class). <br>
 *  The same principle is recursively applied in
 *  the 3D case: additional pointers reference to a block of 2D pointers
 *  belonging to a fixed first index. It'd be easy to enhance the
 *  principle to 4D, etc. <br>
 *  The speed up is noticeable at least for 3D arrays. <br>
 *  This class is used for the implementation of the recurrent neural networks
 *  "MSERNNet" in package "ReClaM".
 *
 *
 *  \author  M. Toussaint
 *  \date    2000
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 */
template<class T>

class ArrayTable
{

public:

//===========================================================================
	/*!
	 *  \brief Used to store all elements of a 1- to 3-dimensional array
	 *         (as the element vector Array::e).
	 *
	 *  When working with 1-dimensional arrays, there is no difference
	 *  to class \em Array.
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array<T> A;

//===========================================================================
	/*!
	 *  \brief Used for faster access, when working with 2-dimensional arrays.
	 *
	 *  When working with 2-dimensional arrays, this array of pointers
	 *  (comparable to Array2D::ptrArrM) to the 1-dimensional arrays
	 *  in \em A allows a direct access to rows, faster than
	 *  achieved by class \em Array.
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array<T*> P1;

//===========================================================================
	/*!
	 *  \brief Used for faster access, when working with 3-dimensional arrays.
	 *
	 *  When working with 3-dimensional arrays, this array of pointers
	 *  to the 2-dimensional arrays in \em A allows a direct access, faster than
	 *  achieved by class \em Array.
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	Array<T**> P2;


//===========================================================================
	/*!
	 *  \ingroup Resize
	 *  \brief Modifies the size of a 3-dimensional array.
	 *
	 *  This method changes the size of an existing array. The previous
	 *  content is destroyed.
	 *
	 *      \param  d0    New size in the first dimension, i.e. size of first
	 *                    argument of the array.
	 *      \param  d1    New size in the second dimension, i.e. size of second
	 *                    argument of the array.
	 *      \param  d2    New size in the third dimension, i.e. size of third
	 *                    argument of the array.
	 *      \param  dummy Has no effect, is only introduced due to consistency
	 *                    with class #Array.
	 *      \return None.
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void resize(unsigned d0, unsigned d1, unsigned d2, int dummy = 0)
	{
		unsigned i, j;
		A.resize(d0, d1, d2, false);
		P1.resize(d0, d1, false);
		P2.resize(d0, false);
		for (i = 0;i < d0;i++) {
			for (j = 0;j < d1;j++)	P1(i, j) = &A(i, j, 0);
			P2(i) = &P1(i, 0);
		}
	}

//===========================================================================
	/*!
	 *  \ingroup Resize
	 *  \brief Modifies the size of a 2-dimensional array.
	 *
	 *  This method changes the size of an existing array. The previous
	 *  content is destroyed.
	 *
	 *      \param  d0    New size in the first dimension, i.e. size of first
	 *                    argument of the array.
	 *      \param  d1    New size in the second dimension, i.e. size of second
	 *                    argument of the array.
	 *      \param  dummy Has no effect, is only introduced due to consistency
	 *                    with class Array.
	 *      \return none
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void resize(unsigned d0, unsigned d1, int dummy = 0)
	{
		unsigned i;
		A.resize(d0, d1, false);
		P1.resize(d0, false);
		P2.resize(0, false);
		for (i = 0;i < d0;i++) P1(i) = &A(i, 0);
	}

//===========================================================================
	/*!
	 *  \ingroup Resize
	 *  \brief Modifies the size of a 1-dimensional array.
	 *
	 *  This method changes the size of an existing array. The previous
	 *  content is destroyed.
	 *
	 *      \param  d0    New size in the first dimension, i.e. size of first
	 *                    argument of the array
	 *      \param  dummy Has no effect, is only introduced due to consistency
	 *                    with class Array
	 *      \return none
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void resize(unsigned d0, int dummy = 0)
	{
		A.resize(d0, false);
		P1.resize(0, false);
		P2.resize(0, false);
	}


//===========================================================================
	/*!
	 *  \brief Casts current 1-dimensional ArrayTable object to C-style array.
	 *
	 *  When you want to use C-style arrays of type \f$T\f$ to store
	 *  the content of an ArrayTable object, then this operator is
	 *  used for the correct casting.
	 *
	 *  \return C-style array with elements of current ArrayTable object.
	 *
	 *  \par Example
	 *  \code
	 *  // 1-dimensional ArrayTable, allocate memory:
	 *  ArrayTable< double > a; a.resize( 2 );
	 *  // 1-dimensional C-style array, allocate memory:
	 *  double *b; b = (double *) malloc(2 * sizeof(double));
	 *  // Fill ArrayTable "a" with content:
	 *  ...
	 *  b = a;
	 *  \endcode
	 *
	 *  In this example two arrays are used. The first one is of type
	 *  ArrayTable, the second one is an array as used in C. Later in the
	 *  program the ArrayTable object is assigned to the C-style array.
	 *  This assignment needs a casting, where the operator is used.
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	operator T*()
	{
		return A.elemvec();
	}


//===========================================================================
	/*!
	 *  \brief Casts current 2-dimensional ArrayTable object to C-style array.
	 *
	 *  When you want to use C-style arrays of type \f$T \ast\f$ to store
	 *  the content of an ArrayTable object, then this operator is
	 *  used for the correct casting.
	 *
	 *  \return C-style array with elements of current ArrayTable object.
	 *
	 *  \par Example
	 *  \code
	 *  // 2-dimensional ArrayTable, allocate memory:
	 *  ArrayTable< double > a; a.resize( 2, 3 );
	 *  // 2-dimensional C-style array:
	 *  double **b;
	 *  // Allocate memory for first dimension (C-style):
	 *  b = (double **) malloc(2 * sizeof(double *));
	 *  // Auxiliary variable:
	 *  double *c;
	 *  // Allocate memory for second dimension (C-style):
	 *  for (unsigned i = 0; i < 2; i = i + 1) {
	 *      c = (double *) malloc(3 * sizeof(double));
	 *      b[i] = c;
	 *  }
	 *  // Fill ArrayTable "a" with content:
	 *  ...
	 *  b = a;
	 *  \endcode
	 *
	 *  In this example two arrays are used. The first one is of type
	 *  ArrayTable, the second one is an array as used in C. Later in the
	 *  program the ArrayTable object is assigned to the C-style array.
	 *  This assignment needs a casting, where the operator is used.
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	operator T**()
	{
		return P1.elemvec();
	}


//===========================================================================
	/*!
	 *  \brief Casts current 3-dimensional ArrayTable object to C-style array.
	 *
	 *  When you want to use C-style arrays of type \f$T \ast \ast\f$ to store
	 *  the content of an ArrayTable object, then this operator is
	 *  used for the correct casting.
	 *
	 *  \return C-style array with elements of current ArrayTable object.
	 *
	 *  \par Example
	 *  \code
	 *  // 3-dimensional ArrayTable, allocate memory:
	 *  ArrayTable< double > a; a.resize( 2, 3, 4 );
	 *  // 3-dimensional C-style array:
	 *  double ***b;
	 *  // Allocate memory for first dimension (C-style):
	 *  b = (double ***) malloc(2 * sizeof(double **));
	 *  // Auxiliary variables:
	 *  double **c; double *d;
	 *  // Allocate memory for second dimension (C-style):
	 *  for (unsigned i = 0; i < 2; i = i + 1) {
	 *      c = (double **) malloc(3 * sizeof(double *));
	 *      // Allocate memory for third dimension (C-style):
	 *      for (unsigned j = 0; i < 3; j = j + 1) {
	 *          d = (double *) malloc(4 * sizeof(double));
	 *          c[ j ] = d;
	 *      }
	 *      b[i] = c;
	 *  }
	 *  // Fill ArrayTable "a" with content:
	 *  ...
	 *  b = a;
	 *  \endcode
	 *
	 *  In this example two arrays are used. The first one is of type
	 *  ArrayTable, the second one is an array as used in C. Later in the
	 *  program the ArrayTable object is assigned to the C-style array.
	 *  This assignment needs a casting, where the operator is used.
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	operator T***()
	{
		return P2.elemvec();
	}


//===========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Allows access to single entries of a 1-dimensional array.
	 *
	 *  This method provides the functionality to allow access to single elements
	 *  of an 1-dimensional array. The value of the parameter must fit to the size
	 *  of the array chosen by means of #resize
	 *
	 *      \param  i Index of the element.
	 *      \return Entry of \f$i\f$-th element of the array.
	 *      \throw check_exception the type of the exception will be
	 *             "range check error" and indicates that \em i exceeds
	 *             the array's first dimension
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	T& operator()(unsigned i)
	{
		return A(i);
	}


//===========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Allows access to single entries of a 2-dimensional array.
	 *
	 *  This method provides the functionality to allow access to single elements
	 *  of an 2-dimensional array. The values of the parameters must fit to the
	 *  sizes of the array chosen by means of #resize
	 *
	 *      \param  i Index of the first dimension.
	 *      \param  j Index of the second dimension.
	 *      \return Entry of \f$j\f$-th element of the \f$i\f$-th row of the array.
	 *      \throw check_exception the type of the exception will be
	 *             "range check error" and indicates that \em i exceeds
	 *             the array's first dimension or \em j exceeds the
	 *             array's second dimension
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	T& operator()(unsigned i,unsigned j)
	{
		return P1(i)[j];
	}


//===========================================================================
	/*!
	 *  \ingroup Extract
	 *  \brief Allows access to single entries of a 3-dimensional array.
	 *
	 *  This method provides the functionality to allow access to single elements
	 *  of an 3-dimensional array. The values of the parameters must fit to the
	 *  sizes of the array chosen by means of #resize.
	 *
	 *      \param  i Index of the first dimension.
	 *      \param  j Index of the second dimension.
	 *      \param  k Index of the third dimension.
	 *      \return Element with the indices \f$i\f$, \f$j\f$, and \f$k\f$.
	 *      \throw check_exception the type of the exception will be
	 *             "range check error" and indicates that \em i exceeds
	 *             the array's first dimension or \em j exceeds the
	 *             array's second dimension or \em k exceeds the array's
	 *             third dimension
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	T& operator()(unsigned i,unsigned j, unsigned k)
	{
		return P2(i)[j][k];
	}


//===========================================================================
	/*!
	 *  \ingroup Assign
	 *  \brief Assigns the values of one ArrayTable to another.
	 *
	 *  This method assigns the values of one array to another. After this
	 *  operation both arrays have the same content as well as the same size and
	 *  dimensionality.
	 *
	 *      \param a ArrayTable object that is assigned to the current
	 *                         ArrayTable object.
	 *      \return A reference to the current ArrayTable object
	 *              (after assignment).
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      none
	 *
	 */
	ArrayTable<T>& operator =(const ArrayTable<T>& a)
	{
		if (a.A.ndim() == 3) resize(a.A.dim(0), a.A.dim(1), a.A.dim(2), false);
		else {
			if (a.A.ndim() == 2) resize(a.A.dim(0), a.A.dim(1), false);
			else if (a.A.ndim() == 1) resize(a.A.dim(0), false);
		}
		A = a.A;
		return *this;
	}

//===========================================================================
	/*!
	 *  \ingroup Assign
	 *  \brief Assigns one value to all elements of an ArrayTable
	 *
	 *  This method assigns one given value \em d to all elements of an array.
	 *
	 *      \param  d Value which should be assigned.
	 *      \return A reference to the current ArrayTable object
	 *              (after assignment).
	 *
	 *  \author  M. Toussaint
	 *  \date    2000
	 *
	 *  \par Changes
	 *      ra, 2001-12-12:
	 *      Changed type of parameter \em d from "double" to "const T&".
	 *
	 *  \par Status
	 *      none
	 *
	 */
	ArrayTable<T>& operator =(const T& d)
	{
		for (unsigned i = 0;i < A.nelem();i++) A.elem(i) = d;
		return *this;
	}



//===========================================================================
	/*!
	 *  \ingroup Create
	 *  \brief Creates a new empty ArrayTable.
	 *
	 *  \return none
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
	ArrayTable()
{ }

//===========================================================================
	/*!
	 *  \ingroup Create
	 *  \brief Creates a new empty one-dimensional ArrayTable.
	 *
	 *  \param  i the size of the ArrayTable in the first dimension.
	 *  \return none
	 *
	 *  \author  R. Alberts
	 *  \date    2002-05-06
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ArrayTable(unsigned i)
	{
		resize(i);
	}


//===========================================================================
	/*!
	 *  \ingroup Create
	 *  \brief Creates a new empty 2-dimensional ArrayTable.
	 *
	 *  \param  i the size of the ArrayTable in the first dimension.
	 *  \param  j the size of the ArrayTable in the second dimension.
	 *  \return none
	 *
	 *  \author  R. Alberts
	 *  \date    2002-05-06
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ArrayTable(unsigned i, unsigned j)
	{
		resize(i, j);
	}


//===========================================================================
	/*!
	 *  \ingroup Create
	 *  \brief Creates a new empty 3-dimensional ArrayTable.
	 *
	 *  \param  i the size of the ArrayTable in the first dimension.
	 *  \param  j the size of the ArrayTable in the second dimension.
	 *  \param  k the size of the ArrayTable in the third dimension.
	 *  \return none
	 *
	 *  \author  R. Alberts
	 *  \date    2002-05-06
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ArrayTable(unsigned i, unsigned j, unsigned k)
	{
		resize(i, j, k);
	}

//===========================================================================
	/*!
	 *  \ingroup Create
	 *  \brief Creates a new ArrayTable as a copy of the parameter (copy constructor).
	 *
	 *  \param  a the ArrayTable object to copy.
	 *  \return none
	 *
	 *  \author  T. Glasmachers, C. Igel
	 *  \date    2004-11-16, 2009-02-11
	 *
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	ArrayTable(const ArrayTable<T>& a)
	{
		operator = (a);
	}


};


#endif // __ARRAY_TABLE_H

