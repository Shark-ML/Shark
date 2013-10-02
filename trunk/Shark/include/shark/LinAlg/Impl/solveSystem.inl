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

#include <shark/LinAlg/Cholesky.h>

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

template<class System,class MatT,class VecT>
void shark::blas::solveSymmSemiDefiniteSystemInPlace(
	matrix_expression<MatT> const& A, 
	vector_expression<VecT>& b
){
	//we will ignore the "System" parameter in the vector
	//version as A is symmetric
	
	std::size_t m = A().size1();
	SIZE_CHECK(b().size() == m);
	SIZE_CHECK(A().size2() == m);
	
	// We can't assume that A has full rank, therefore we 
	// compute a (somehow smartly implemented) pseudoinverse
	// This is an implementation suggested by
	// "Fast Computation of Moore-Penrose Inverse Matrices"
	// trading numerical accuracy vs speed. We go for speed.
	//
	// We use the formula for the pseudo-inverse:
	// P^T A' P = L(L^TL)^-1(L^TL)^-1 L^T
	// where L is the lower cholesky factor and P a pivoting matrix.
	//But if A has full rank, we can use the direct inverse.
		
	//do a pivoting cholesky decomposition leading to L and P
	RealMatrix LDecomp(m,m);
	PermutationMatrix permutation(m);
	std::size_t rank = pivotingCholeskyDecomposition(A,permutation,LDecomp);
	
	//only take the nonzero columns as L
	matrix_range<RealMatrix> L = columns(LDecomp,0,rank);
	
	//apply permutation thus remove P from the following equations
	swapRows(permutation,b);
	
	//matrix has full rank, means that we can use the typical cholesky inverse
	if(rank == m){
		solveTriangularCholeskyInPlace<SolveAXB>(L,b);
	}
	else
	{
		//complex case. 
		//A' = L(L^TL)^-1(L^TL)^-1 L^T
		RealMatrix LTL(rank,rank);
		fast_prod(trans(L),L,LTL);
		
		//compute z= L^Tb
		RealVector z(rank);
		fast_prod(trans(L),b,z);
		
		//compute cholesky factor of L^TL
		RealMatrix LTLcholesky(rank,rank);
		choleskyDecomposition(LTL,LTLcholesky);
		
		//A'b =  L(L^TL)^-1(L^TL)^-1z
		solveTriangularCholeskyInPlace<SolveAXB>(LTLcholesky,z);
		solveTriangularCholeskyInPlace<SolveAXB>(LTLcholesky,z);
		fast_prod(L,z,b);
	}
	//finally swap back into the unpermuted coordinate system
	swapRowsInverted(permutation,b);
}

template<class System,class Mat1T,class Mat2T>
void shark::blas::solveSymmSemiDefiniteSystemInPlace(
	matrix_expression<Mat1T> const& A, 
	matrix_expression<Mat2T>& B
){
	SIZE_CHECK(A().size2() == A().size1());
	if(System::left){
		SIZE_CHECK(A().size1() == B().size1());
	}else{
		SIZE_CHECK(A().size1() == B().size2());
	}
	std::size_t m = A().size1();
	
	//see the vector version of this routine for a longer documentation.
		
	//do a pivoting cholesky decomposition leading to L and P
	RealMatrix LDecomp(m,m);
	PermutationMatrix permutation(m);
	std::size_t rank = pivotingCholeskyDecomposition(A,permutation,LDecomp);
	
	//only take the nonzero columns as L
	matrix_range<RealMatrix> L = columns(LDecomp,0,rank);
	
	//apply permutation thus remove P from the following equations
	if(System::left)
		swapRows(permutation,B);
	else
		swapColumns(permutation,B);
	
	//matrix has full rank, means that we can use the typical cholesky inverse
	if(rank == m){
		solveTriangularCholeskyInPlace<System>(L,B);
	}
	else if(System::left)
	{
		//complex case. 
		//X=L(L^TL)^-1(L^TL)^-1 L^TB
		RealMatrix LTL(rank,rank);
		fast_prod(trans(L),L,LTL);
		
		//compute z= L^TB
		RealMatrix Z(rank,B().size2());
		fast_prod(trans(L),B,Z);
		
		//compute cholesky factor of L^TL
		RealMatrix LTLcholesky(rank,rank);
		choleskyDecomposition(LTL,LTLcholesky);
		
		//A'b =  L(L^TL)^-1(L^TL)^-1z
		solveTriangularCholeskyInPlace<System>(LTLcholesky,Z);
		solveTriangularCholeskyInPlace<System>(LTLcholesky,Z);
		fast_prod(L,Z,B);
	}else{
		//complex case. 
		//X=BL(L^TL)^-1(L^TL)^-1 L^T
		RealMatrix LTL(rank,rank);
		fast_prod(trans(L),L,LTL);
		
		//compute z= L^TB
		RealMatrix Z(B().size1(),rank);
		fast_prod(B,L,Z);
		
		//compute cholesky factor of L^TL
		RealMatrix LTLcholesky(rank,rank);
		choleskyDecomposition(LTL,LTLcholesky);
		
		//A'b =  L(L^TL)^-1(L^TL)^-1z
		solveTriangularCholeskyInPlace<System>(LTLcholesky,Z);
		solveTriangularCholeskyInPlace<System>(LTLcholesky,Z);
		fast_prod(Z,trans(L),B);
	}
	//finally swap back into the unpermuted coordinate system
	if(System::left)
		swapRowsInverted(permutation,B);
	else
		swapColumnsInverted(permutation,B);
}

template<class System,class MatT,class VecT>
void shark::blas::generalSolveSystemInPlace(
	matrix_expression<MatT> const& A, 
	vector_expression<VecT>& b
){
	std::size_t m = A().size1();
	std::size_t n = A().size2();
	
	if( System::left){
		SIZE_CHECK(A().size1() == b().size());
		//reduce to the case of quadratic A
		//Ax=b => A^TAx=A'b => x= A'b = (A^TA)' Ab
		// with z = Ab => (A^TA) x= z
		//compute A^TA
		RealMatrix ATA(n,n);
		fast_prod(trans(A),A,ATA);
		
		//compute z=Ab
		RealVector z(n);
		fast_prod(trans(A),b,z);
		
		//call recursively for the quadratic case
		solveSymmSemiDefiniteSystemInPlace<System>(ATA,z);
		b() = z;
	}
	else{
		SIZE_CHECK(A().size2() == b().size());
		//reduce to the case of quadratic A
		//x^TA=b^T => x^TAA'=b^TA' => x^T= b^TA' = b^TA^T(AA^T)'
		// with z = Ab => x^T(AA^T) = z^T
		//compute AAT
		RealMatrix AAT(m,m);
		fast_prod(A,trans(A),AAT);
		
		//compute z=Ab
		RealVector z(m);
		fast_prod(A,b,z);
		
		//call recursively for the quadratic case
		solveSymmSemiDefiniteSystemInPlace<System>(AAT,z);
		b() = z;		
	}
}

template<class System,class MatA,class MatB>
void shark::blas::generalSolveSystemInPlace(
	matrix_expression<MatA> const& A, 
	matrix_expression<MatB>& B
){
	std::size_t m = A().size1();
	std::size_t n = A().size2();
	
	if( System::left){
		SIZE_CHECK(A().size1() == B().size1());
		//reduce to the case of quadratic A
		//Ax=b => A^TAx=A'b => x= A'b = (A^TA)' Ab
		// with z = Ab => (A^TA) x= z
		//compute A^TA
		RealMatrix ATA(n,n);
		fast_prod(trans(A),A,ATA);
		
		//compute Z=AB
		RealVector Z(n,B().size2());
		fast_prod(trans(A),B,Z);
		
		//call recursively for the quadratic case
		solveSymmSemiDefiniteSystemInPlace<System>(ATA,Z);
		B() = Z;
	}
	else{
		SIZE_CHECK(A().size2() == B().size2());
		//reduce to the case of quadratic A
		//x^TA=b^T => x^TAA'=b^TA' => x^T= b^TA' = b^TA^T(AA^T)'
		// with z = Ab => x^T(AA^T) = z^T
		//compute AAT
		RealMatrix AAT(m,m);
		fast_prod(A,trans(A),AAT);
		
		//compute z=Ab
		RealVector Z(m,B().size1());
		fast_prod(A,B,Z);
		
		//call recursively for the quadratic case
		solveSymmSemiDefiniteSystemInPlace<System>(AAT,Z);
		B() = Z;		
	}
}

#endif
