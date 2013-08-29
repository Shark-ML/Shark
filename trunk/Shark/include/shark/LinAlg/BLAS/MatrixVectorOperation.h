/**
*
*  \brief Optimized generalized matrix-vector operations for Linear Algebra
*
*  \author O.Krause
*  \date 2011
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


/*
BE AWARE THAT THE METHODS USED HERE ARE WORK IN PROGFRESS AND
ARE NOT GUARANTEED TO WORK PROPERLY IN ALL CASES!
*/
#ifndef SHARK_LINALG_BLAS_MATRIX_VECTOR_OPERATION_H
#define SHARK_LINALG_BLAS_MATRIX_VECTOR_OPERATION_H

#include "Impl/BlockMatrixVectorOperation.inl"
namespace shark { namespace blas{
///\brief implements an operation of the Form \f$ c_i = c_i + \sum_{j=1}^n k(A_{ij},b_j)\f$ for arbitrary kernels k.
template<class MatA,class VecB,class VecC,class ComputeKernel>
void generalMatrixVectorOperation(
	matrix_expression<MatA> const & matA,
	vector_expression<VecB> const & vecB,
	vector_expression<VecC>& vecC,
	ComputeKernel kernel
){
	detail::generalMatrixVectorOperation(
		matA(),vecB(),vecC(),kernel,
		typename MatA::orientation_category(),
		typename MatA::storage_category(),
		typename VecB::storage_category()
	);
}
template<class MatA,class VecB,class VecC,class ComputeKernel>
void generalMatrixVectorOperation(
	matrix_expression<MatA> const & matA,
	vector_expression<VecB> const & vecB,
	matrix_row<VecC> vecC,
	ComputeKernel kernel
){
	typedef vector_expression<matrix_row<VecC> > super;
	generalMatrixVectorOperation(matA(),vecB(),static_cast<super&>(vecC),kernel);
}
template<class MatA,class VecB,class VecC,class ComputeKernel>
void generalMatrixVectorOperation(
	matrix_expression<MatA> const & matA,
	vector_expression<VecB> const & vecB,
	matrix_column<VecC> vecC,
	ComputeKernel kernel
){
	typedef vector_expression<matrix_column<VecC> > super;
	generalMatrixVectorOperation(matA(),vecB(),static_cast<super&>(vecC),kernel);
}
template<class MatA,class VecB,class VecC,class ComputeKernel>
void generalMatrixVectorOperation(
	matrix_expression<MatA> const & matA,
	vector_expression<VecB> const & vecB,
	vector_range<VecC> vecC,
	ComputeKernel kernel
){
	typedef vector_expression<vector_range<VecC> > super;
	generalMatrixVectorOperation(matA(),vecB(),static_cast<super&>(vecC),kernel);
}
}}
#endif
