/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2012
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_LINALG_BLAS_KERNELS_SYEV_HPP
#define SHARK_LINALG_BLAS_KERNELS_SYEV_HPP


#ifdef SHARK_USE_LAPACK
#include "lapack/syev.hpp"
#else
#include "default/syev.hpp"
#endif
	
namespace shark { namespace blas {namespace kernels{
	
///\brief Well known SYmmetric EigenValue function (SYEV).
///
/// A given matrix A is decomposed as 
/// A=QDQ^T
/// where Q is an orthogonal (or unitary) matrix with QQ^T=Q^TQ=I and D are the eigenvalue
/// of A. As A is symmetric, only the lower part of it is accessed for reading.
/// The wholee matrix will in the end contain the eigenvectors of A and thus
/// A is replaced by Q. 
/// Additionally the eigenvalues are stored in the second argument. 
template <typename MatrA, typename VectorB>
void syev(
	matrix_expression<MatrA>& matA,
	vector_expression<VectorB>& eigenValues
) {
	bindings::syev(matA,eigenValues);
}


}}}
#endif