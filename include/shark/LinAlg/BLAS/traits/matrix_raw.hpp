//===========================================================================
/*!
 *  \author O. Krause
 *  \date 2010
 *
 *  \par Copyright (c) 1998-2011:
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
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
//  Based on the boost::numeric bindings
//  Copyright (c) 2002,2003,2004
//  Toon Knapen, Kresimir Fresl, Joerg Walter, Karl Meerbergen
//
// Distributed under the Boost Software License, Version 1.0. 
// (See accompanying file LICENSE_1_0.txt or copy at 
// http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef SHARK_LINALG_BLAS_TRAITS_MATRIX_RAW_HPP
#define SHARK_LINALG_BLAS_TRAITS_MATRIX_RAW_HPP

#include "metafunctions.h"

namespace shark { namespace blas{ namespace traits {



/////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////STORAGE STRIDES////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

///\brief Returns the difference of positions between two rows of the matrix in memory
///
///Works only, if matrix is dense!
template <typename M>
int matrix_stride1(blas::matrix_expression<M> const& m) {
	return ExpressionTraits<M>::stride1(m());
}
///\brief Returns the difference of positions between two columns of the matrix in memory
///
///Works only, if matrix is dense!
template <typename M>
int matrix_stride2(blas::matrix_expression<M> const& m) {
	return ExpressionTraits<M>::stride2(m());
}

////////////////////////////////////////////////////////////////////////////////////////
/////////////////ORIENTATION STUFF/////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
//returns, whether the matrix_expression has row_major format. 
template<class Mat>
bool isRowMajor(blas::matrix_expression<Mat> const&,blas::row_major_tag){
    return true;
}
template<class Mat>
bool isRowMajor(blas::matrix_expression<Mat> const&,blas::column_major_tag){
    return false;
}
template<class Mat>
bool isRowMajor(blas::matrix_expression<Mat> const& mat){
    return isRowMajor(mat,typename Mat::orientation_category());
}
//returns, whether the matrix_expression has column_major format. 
template<class Mat>
bool isColumnMajor(blas::matrix_expression<Mat> const& mat){
    return !isRowMajor(mat);
}
//returns whether matrix expressions A and B have the same orientation of elements
template<class  MatA, class MatB>
bool sameOrientation(blas::matrix_expression<MatA> const&, blas::matrix_expression<MatB> const&){
	return boost::is_same<typename MatA::orientation_category,typename MatB::orientation_category>::value;
}

///returns whether a matrix expression X is in fact a transpose of a matrix T
template<class Mat>
bool isTransposed(blas::matrix_expression<Mat> const& mat){
	return ExpressionTraits<Mat>::transposed;
}
	
//returns the size of the leading dimension of the matrix in memory
template <typename Mat>
int leadingDimension(Mat const& m, blas::row_major_tag) {
	return (int) matrix_stride1(m);
}
template <typename Mat>
int leadingDimension(Mat const& m, blas::column_major_tag) {
	return (int) matrix_stride2(m);
}
template <typename Mat>
int leadingDimension(blas::matrix_expression<Mat> const& m) {
	return leadingDimension(m(), typename Mat::orientation_category());
}

///\brief Returns true, when the matrix has real dense storage and false if it is just a lazy expression or a sparse matrix.
template<class Mat>
bool hasStorage(blas::matrix_expression<Mat> const&){
	return IsDense<Mat>::value;
}

///\brief returns true if the leading dimension is dense.
template<class Mat>
bool hasDenseLeadingDimension(blas::matrix_expression<Mat> const& m){
	if(!isTransposed(m))
		return hasStorage(m)&& leadingDimension(m)==(int)m().size2();
	else
		return hasStorage(m)&& leadingDimension(m)==(int)m().size1();
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////MEMORY ACCESS STUFF///////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
///\brief Returns a pointer to the first element of the matrix
template <typename M>
typename PointerType<M const>::type
matrix_storage(blas::matrix_expression<M> const& m) {
	return ExpressionTraits<M const>::storageBegin(m());
}
///\brief Returns a pointer to the first element of the matrix
template <typename M>
typename PointerType<M>::type
matrix_storage(blas::matrix_expression<M> & m) {
	return ExpressionTraits<M>::storageBegin(m());
}

}
}}

#endif
