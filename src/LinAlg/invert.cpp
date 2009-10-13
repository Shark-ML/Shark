//===========================================================================
/*!
 *  \file invert.cpp
 *
 *  \brief Determines the generalized inverse matrix of an input matrix
 *         by using singular value decomposition. Used as frontend
 *         for metod #g_inverse when using type "Array" instead of
 *         "Array2D".
 *
 *  \author  M. Kreutz
 *  \date    1998
 *
 *  \par Copyright (c) 1998-2000:
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
//===========================================================================

#include <cmath>
#include <SharkDefs.h>
#include <LinAlg/LinAlg.h>

//===========================================================================
/*!
 *  \brief Returns the generalized inverse matrix of input matrix
 *         "A" by using singular value decomposition. Used as frontend
 *         for method #g_inverse when using type "Array" instead of
 *         "Array2D".
 *
 *  For a more exact description see documentation of method
 *  #g_inverse.
 *  Here not only the usage of variable type "Array< double >"
 *  instead of "Array2D< double >" as storage for matrices
 *  is different, but also the resulting generalized inverse
 *  matrix is returned directly and not given back by assigning
 *  it to a second parameter.
 *
 *  \param  A The input matrix.
 *  \return   The generalized inverse matrix.
 *  \throw SharkException the type of the exception will be
 *         "size mismatch" and indicates that \em A is not a
 *         square matrix
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
 *  \sa g_inverse.cpp, svd.cpp
 *
 */
Array< double > invert(const Array< double >& A)
{
	SIZE_CHECK(A.ndim() == 2)

	Array< double > B(A.dim(1), A.dim(0));
// 	Array2DReference< double > amatL(const_cast< Array< double >& >(A));
// 	Array2DReference< double > bmatL(B);

	g_inverse(A, B);

	return B;
}
