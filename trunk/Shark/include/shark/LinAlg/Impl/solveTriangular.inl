/*!
 *
 *  \author  O.Krause
 *  \date    2012
 *
 *  \par Copyright (c) 1998-2001:
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

#ifndef SHARK_LINALG_IMPL_SOLVE_TRIANGULAR_INL
#define SHARK_LINALG_IMPL_SOLVE_TRIANGULAR_INL

#include <shark/LinAlg/BLAS/kernels/trsm.hpp>
#include <shark/LinAlg/BLAS/kernels/trsv.hpp>


namespace shark{ namespace blas{ namespace detail{

////////////////SOLVE MATRIX_VECTOR///////////////////
template<class MatT,class VecT, class MatrixTag>
void solveTriangularSystemInPlace(
	const matrix_expression<MatT>& A, 
	vector_expression<VecT>& b,
	SolveAXB,
	MatrixTag
){
	kernels::trsv<MatrixTag::is_upper,MatrixTag::is_unit>(A,b);
}
///solving xA=b is equal to transposing A
template<class MatT,class VecT,class MatrixTag>
void solveTriangularSystemInPlace(
	const matrix_expression<MatT>& A, 
	vector_expression<VecT>& b,
	SolveXAB,
	MatrixTag
){
	kernels::trsv<!MatrixTag::is_upper,MatrixTag::is_unit>(trans(A),b);
}

//////////////////SOLVE CHOLESKY////////////////////////////////
template<class MatL,class Arg>
void solveTriangularCholeskyInPlace(
	matrix_expression<MatL> const& L, 
	Arg& b,
	SolveAXB
){
	shark::blas::solveTriangularSystemInPlace<SolveAXB,lower>(L,b);
	shark::blas::solveTriangularSystemInPlace<SolveAXB,upper>(trans(L),b);
}
template<class MatL,class Arg>
void solveTriangularCholeskyInPlace(
	matrix_expression<MatL> const& L, 
	Arg& b,
	SolveXAB
){
	shark::blas::solveTriangularSystemInPlace<SolveXAB,upper>(trans(L),b);
	shark::blas::solveTriangularSystemInPlace<SolveXAB,lower>(L,b);
}

}}}

template<class System,class DiagType,class MatT,class VecT>
void shark::blas::solveTriangularSystemInPlace(
	matrix_expression<MatT> const& A, 
	vector_expression<VecT>& b
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == b().size());

	//dispatcher
	detail::solveTriangularSystemInPlace(A,b,System(),DiagType());

}

template<class System, class DiagType,class MatA,class MatB>
void shark::blas::solveTriangularSystemInPlace(
	matrix_expression<MatA> const& A, 
	matrix_expression<MatB>& B
){
	SIZE_CHECK(A().size1() == A().size2());
	if(System::left){
		SIZE_CHECK(A().size1() == B().size1());
		kernels::trsm<DiagType::is_upper,DiagType::is_unit>(A,B);
	}else{
		SIZE_CHECK(A().size1() == B().size2());
		blas::matrix_transpose<MatB> transB = trans(B);
		kernels::trsm<!DiagType::is_upper,DiagType::is_unit>(trans(A),transB);
	}
	
	
}

template<class System,class MatL,class MatB>
void shark::blas::solveTriangularCholeskyInPlace(
	matrix_expression<MatL> const& L, 
	matrix_expression<MatB>& B
){
	SIZE_CHECK(L().size1() == L().size2());
	if(System::left){
		SIZE_CHECK(L().size1() == B().size1());
	}else{
		SIZE_CHECK(L().size1() == B().size2());
	}
	
	detail::solveTriangularCholeskyInPlace(L,B(),System());
}
template<class System,class MatL,class VecB>
void shark::blas::solveTriangularCholeskyInPlace(
	const matrix_expression<MatL>& L, 
	vector_expression<VecB>& b
){
	SIZE_CHECK(L().size1() == L().size2());
	SIZE_CHECK(L().size2() == b().size());
	
	detail::solveTriangularCholeskyInPlace(L,b(),System());
}

#endif
