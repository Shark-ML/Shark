//===========================================================================
/*!
 *  \file ArraySort.h
 *
 *  \brief Iterative implementation of the quicksort algorithm for
 *         one-dimensional arrays with support of index arrays.
 *
 *  \author  M. Kreutz
 *  \date    1995-11-03
 *
 *  \par Copyright (c) 1995-2000:
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

#ifndef __cplusplus
#error Must use C++.
#endif

#ifndef __ARRAYSORT_H
#define __ARRAYSORT_H

#include <climits>
#include <SharkDefs.h>
#include <Array/Array.h>


//! If this flag is defined, then a pivot value is
//! determined by random. Otherwise it is determined
//! by bit operations.
#define PIVOT

#ifdef PIVOT
#    ifndef myRandom
//=============================================================
/*!
 *  \brief Returns a random number between 0 and num - 1.
 *
*  This preprocessor operation is only defined, if
*  the #pivot flag is set.
*
 *  \param  num defines the upper bound for the returned random
*              number
 *  \return a random number between 0 and \em num - 1
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
#        define myRandom( num ) ( rand() % ( num ) )
#    endif /* random */
//=============================================================
/*!
 *  \brief Returns a pivot value as random number between L and R - 1.
 *
 *  The pivot value is only set as random value if the
*  #pivot flag above is set. <br>
*  The pivot value is used for the partitioning step of
*  the quicksort algorithm implementation in function
*  #sort.
*
 *  \param  L defines the lower bound for the returned random
*            number
 *  \param  R defines the upper bound for the returned random
*            number
 *  \return a pivot value between \em L and \em R - 1
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
#    define pivot( L, R ) ( L + myRandom( R - L ) )
#else
//=============================================================
/*!
*  \brief Returns a pivot value between \f$\frac{L + R}{2}\f$
*         and \f$\frac{L + R}{2} + S\f$, where \f$S\f$
*         is the lower one of the two values "L" and "R".
*
*  The pivot value is only set by bit operations, if the
*  #pivot flag above is not set. <br>
*  The pivot value is used for the partitioning step of
*  the quicksort algorithm implementation in function
*  #sort.
*
*  \param  L the first of the two values that determine the
*            pivot value
*  \param  R the first of the two values that determine the
*            pivot value
*  \return a pivot value between \f$\frac{L + R}{2}\f$
*          and \f$\frac{L + R}{2} + S\f$
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
#    define pivot( L, R ) ( ( L >> 1 ) + ( R >> 1 ) + ( L & R & 1 ) )
#endif /* PIVOT */

//=============================================================
/*!
 *  \brief Performs a quicksort on the one-dimensional array
 *         "value" and will also return the index array
 *         "index" of "value".
 *
 *  This method implements the famous quicksort algorithm
 *  not in a recursive but in an iterative way. <br>
 *  Furthermore an index array is returned, corresponding to
 *  the elements of the value array in their original order
 *  and sorted in the same way as the values of the value array.
 *  This index array can be used especially for the functions
 *  "ArrayIndex operator&&(ArrayIndex& a, ArrayIndex& b)" and
 *  "ArrayIndex operator||(ArrayIndex& a, ArrayIndex& b)". <br>
 *  By defining/undefining the flag #pivot, it can be determined,
 *  whether the initial pivot point for the first partition step
 *  is a random value or a value calculated by bit operations.
 *
 *  \param  value the one-dimensional array with the values
 *                that will be sorted. After the method call,
 *                the values in this array are in ascending
 *                order
 *  \param  index will contain the indices of the elements of
 *                array \em value as given before sorting and
 *                in the same order as the corresponding
 *                elements in \em value after the sort
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
void sort(Array< T        >& value,
		  Array< unsigned >& index)
{
	const unsigned StackSize = 100;
	unsigned Lstack[ StackSize ];
	unsigned Rstack[ StackSize ];
	unsigned L, R, i, j, p, q;
	int      stackptr = 1;
	T        val;
	unsigned ind;

	// Initialize index borders:
	L = 0;
	R = value.nelem() - 1;

	// Initialize index border stacks:
	Lstack[ 1 ] = L;
	Rstack[ 1 ] = R;

	// Initialize index vector with the indices of the input vector values:
	index.resize(value.nelem());
	for (i = L; i <= R; i++)
		index.elem(i) = i;

	// Sorting takes place as long as there are elements
	// on the index border stacks (= simulated recursion calls
	// left for processing):
	do
	{
		// Pop the top index border stacks elements, take them as new
		// index borders:
		L = Lstack[ stackptr ];
		R = Rstack[ stackptr ];
		stackptr--;

		// Processing the elements between the current index borders
		// (= borders of the current recursion) as long as they don't meet:
		do
		{
			p = q = pivot(L, R);   // the current pivot values,
			// p used as backup, q can be changed
			i = L;                 // the left pointer
			j = R;                 // the right pointer

			// Process as long as the current left and right pointer
			// don't meet:
			do
			{
				// Skip left pointer elements that are already less than the
				// value at the current pivot position:
				while (value.elem(i) < value.elem(q)) i++;
				// Skip right pointer elements that are already greater than
				// the value at the current pivot position:
				while (value.elem(q) < value.elem(j)) j--;

				if (i <= j)
				{
					if (i != j)
					{
						// One of the pointers meet the pivot?
						// Set the pivot to the opposite pointer
						if (p == i) q = j;
						if (p == j) q = i;

						// Swap the current left pointer and right
						// pointer element
						val             = value.elem(i);
						value.elem(i) = value.elem(j);
						value.elem(j) = val;

						// Swap the indices of the current left pointer and
						// right pointer element
						ind             = index.elem(i);
						index.elem(i) = index.elem(j);
						index.elem(j) = ind;
					}

					// left and right pointer are moving towards each other:
					if (i < UINT_MAX) i++;
					if (j > 0) j--;
				}
			}
			while (i <= j);

			// Simulate sorting recursion, by determining
			// the new index borders and pushing them on
			// the stacks (partitioning).
			if ((j - L) < (R - i))
			{
				if (i < R)
				{
					stackptr++;
					Lstack[ stackptr ] = i;
					Rstack[ stackptr ] = R;
				}
				R = j;
			}
			else
			{
				if (L < j)
				{
					stackptr++;
					Lstack[ stackptr ] = L;
					Rstack[ stackptr ] = j;
				}
				L = i;
			}
		}
		while (L < R);

	}
	while (stackptr > 0);
}

//=============================================================
/*!
 *  \brief Performs a quicksort on the rows of the two-dimensional
 *  array "value" by the values of the first column.
 *
 *  This method implements the famous quicksort algorithm
 *  not in a recursive but in an iterative way. <br>
 *  The sorting is performed by the values of the first column.
 *  All elements of the second column are sorted in the same way
 *  as the corresponding values of the first column belonging to
 *  the same row.<br>
 *  By defining/undefining the flag #pivot, it can be determined,
 *  whether the initial pivot point for the first partition step
 *  is a random value or a value calculated by bit operations.
 *
 *  \param  value the two-dimensional array with the values
 *                that will be sorted. After the method call,
 *                the values of the first column of this array
 *                are in ascending order
 *  \return none
 *
 *  \author  C. Igel
 *  \date    2005
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */

template < class T >
void sort2DBy1st(Array< T >& value)
{
	const unsigned StackSize = 100;
	unsigned Lstack[ StackSize ];
	unsigned Rstack[ StackSize ];
	unsigned L, R, i, j, p, q;
	int      stackptr = 1;
	T  val1, val2;
	unsigned ind;

	Array< unsigned > index;
	// Initialize index borders:
	L = 0;
	R = value.dim(0) - 1;

	// Initialize index border stacks:
	Lstack[ 1 ] = L;
	Rstack[ 1 ] = R;

	// Initialize index vector with the indices of the input vector values:
	index.resize(value.dim(0));
	for (i = L; i <= R; i++)
		index.elem(i) = i;

	// Sorting takes place as long as there are elements
	// on the index border stacks (= simulated recursion calls
	// left for processing):
	do
	{
		// Pop the top index border stacks elements, take them as new
		// index borders:
		L = Lstack[ stackptr ];
		R = Rstack[ stackptr ];
		stackptr--;

		// Processing the elements between the current index borders
		// (= borders of the current recursion) as long as they don't meet:
		do
		{
			p = q = pivot(L, R);   // the current pivot values,
			// p used as backup, q can be changed
			i = L;                 // the left pointer
			j = R;                 // the right pointer

			// Process as long as the current left and right pointer
			// don't meet:
			do
			{
				// Skip left pointer elements that are already less than the
				// value at the current pivot position:
				while (value(i, 0) < value(q, 0)) i++;
				// Skip right pointer elements that are already greater than
				// the value at the current pivot position:
				while (value(q, 0) < value(j, 0)) j--;

				if (i <= j)
				{
					if (i != j)
					{
						// One of the pointers meet the pivot?
						// Set the pivot to the opposite pointer
						if (p == i) q = j;
						if (p == j) q = i;

						// Swap the current left pointer and right
						// pointer element
						val1            = value(i, 0);
						val2            = value(i, 1);
						value(i, 0) = value(j, 0);
						value(i, 1) = value(j, 1);
						value(j, 0) = val1;
						value(j, 1) = val2;

						// Swap the indices of the current left pointer and
						// right pointer element
						ind             = index.elem(i);
						index.elem(i) = index.elem(j);
						index.elem(j) = ind;
					}

					// left and right pointer are moving towards each other:
					if (i < UINT_MAX) i++;
					if (j > 0) j--;
				}
			}
			while (i <= j);

			// Simulate sorting recursion, by determining
			// the new index borders and pushing them on
			// the stacks (partitioning).
			if ((j - L) < (R - i))
			{
				if (i < R)
				{
					stackptr++;
					Lstack[ stackptr ] = i;
					Rstack[ stackptr ] = R;
				}
				R = j;
			}
			else
			{
				if (L < j)
				{
					stackptr++;
					Lstack[ stackptr ] = L;
					Rstack[ stackptr ] = j;
				}
				L = i;
			}
		}
		while (L < R);

	}
	while (stackptr > 0);
}

#endif /* !__ARRAYSORT_H */

