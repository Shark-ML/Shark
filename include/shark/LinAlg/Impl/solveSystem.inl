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

#ifndef SHARK_LINALG_IMPL_SOLVE_SYSTEM_INL
#define SHARK_LINALG_IMPL_SOLVE_SYSTEM_INL

#ifdef SHARK_USE_ATLAS
#include <shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/potrs.h>
#include <shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/trsm.h>
#include <shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/trsv.h>
#endif

#include <shark/LinAlg/Inverse.h>

//todo implement this using ATLAS

template<class MatT,class Vec1T,class Vec2T>
void shark::blas::solveSystem(
	const matrix_expression<MatT>& A, 
	vector_expression<Vec1T>& x,
	const vector_expression<Vec2T>& b
){
	SIZE_CHECK(A().size1() == b().size());
	SIZE_CHECK(A().size1() == A().size2());
	std::size_t n = A().size1();
	
	permutation_matrix<std::size_t> permutation(n);
	MatT LUDecomposition= A();
	
	lu_factorize(LUDecomposition,permutation);
	
	ensureSize(x,n);
	noalias(x()) = b();
	
	//lu_substitute(LUDecomposition,permutation,x());
	swap_rows(permutation,x());
	solveTriangularSystemInPlace<SolveAXB,UnitLower>(LUDecomposition,x);
	solveTriangularSystemInPlace<SolveAXB,Upper>(LUDecomposition,x);
}

template<class MatT,class Mat1T,class Mat2T>
void shark::blas::solveSystem(
	matrix_expression<MatT> const& A, 
	matrix_expression<Mat1T>& X,
	matrix_expression<Mat2T> const& B
){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size1() == A().size2());
	std::size_t n = A().size1();
	
	permutation_matrix<std::size_t> permutation(n);
	MatT LUDecomposition = A;
	
	lu_factorize(LUDecomposition,permutation);
	
	ensureSize(X,n,B().size2());
	noalias(X()) = B();
	
	swap_rows(permutation,X());
	solveTriangularSystemInPlace<SolveAXB,UnitLower>(LUDecomposition,X);
	solveTriangularSystemInPlace<SolveAXB,Upper>(LUDecomposition,X);
}

template<class System,class MatT,class Mat1T>
void shark::blas::solveSymmSystemInPlace(
	matrix_expression<MatT> const& A, 
	matrix_expression<Mat1T>& B
){
	if(System::left){
		SIZE_CHECK(A().size1() == B().size1());
	}else{
		SIZE_CHECK(A().size1() == B().size2());
	}
	SIZE_CHECK(A().size1() == A().size2());
	
	matrix<typename MatT::value_type> cholesky;
	choleskyDecomposition(A(),cholesky);

	solveTriangularCholeskyInPlace<System>(cholesky,B);
}

template<class System,class MatT,class VecT>
void shark::blas::solveSymmSystemInPlace(
	matrix_expression<MatT> const& A, 
	vector_expression<VecT>& b
){
	SIZE_CHECK(A().size1() == b().size());
	SIZE_CHECK(A().size1() == A().size2());
	
	matrix<typename MatT::value_type> cholesky;
	choleskyDecomposition(A(),cholesky);
	solveTriangularCholeskyInPlace<System>(cholesky,b);
}

template<class System,class MatT,class Vec1T,class Vec2T>
void shark::blas::solveSymmSystem(
	const matrix_expression<MatT>& A, 
	vector_expression<Vec1T>& x,
	const vector_expression<Vec2T>& b
){
	SIZE_CHECK(A().size1() == b().size());
	SIZE_CHECK(A().size1() == A().size2());
	ensureSize(x,A().size1());
	noalias(x()) = b();
	solveSymmSystemInPlace<System>(A,x);
}
template<class System,class MatT,class Mat1T,class Mat2T>
void shark::blas::solveSymmSystem(
	const matrix_expression<MatT>& A, 
	matrix_expression<Mat1T>& X,
	const matrix_expression<Mat2T>& B
){
	SIZE_CHECK(A().size1() == A().size2());
	if(System::left){
		SIZE_CHECK(A().size1() == B().size1());
	}else{
		SIZE_CHECK(A().size1() == B().size2());
	}
	ensureSize(X,B().size1(),B().size2());
	noalias(X()) = B();
	solveSymmSystemInPlace<System>(A,X);
}

#endif
