/*!
 *
 *  \author  O.Krause
 *  \date    2012
 *
 * \par Copyright 1995-2015 Shark Development Team
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

#ifndef SHARK_LINALG_IMPL_SOLVE_SYSTEM_INL
#define SHARK_LINALG_IMPL_SOLVE_SYSTEM_INL

//full rank indefinite solvers
#include "../Cholesky.h"

// Symmetric solvers
template<class System,class MatT,class Mat1T>
void shark::blas::solveSymmPosDefSystemInPlace(
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
void shark::blas::solveSymmPosDefSystemInPlace(
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
void shark::blas::solveSymmPosDefSystem(
	const matrix_expression<MatT>& A, 
	vector_expression<Vec1T>& x,
	const vector_expression<Vec2T>& b
){
	SIZE_CHECK(A().size1() == b().size());
	SIZE_CHECK(A().size1() == A().size2());
	ensure_size(x,A().size1());
	noalias(x()) = b();
	solveSymmPosDefSystemInPlace<System>(A,x);
}
template<class System,class MatT,class Mat1T,class Mat2T>
void shark::blas::solveSymmPosDefSystem(
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
	ensure_size(X,B().size1(),B().size2());
	noalias(X()) = B();
	solveSymmPosDefSystemInPlace<System>(A,X);
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
	swap_rows(permutation,b);
	
	//matrix has full rank, means that we can use the typical cholesky inverse
	if(rank == m){
		solveTriangularCholeskyInPlace<SolveAXB>(L,b);
	}
	else if (rank == 0){//A is 0
		b().clear();
	}
	else
	{
		//complex case. 
		//A' = L(L^TL)^-1(L^TL)^-1 L^T
		RealMatrix LTL = prod(trans(L),L);
		
		//compute z= L^Tb
		RealVector z = prod(trans(L),b);
		
		//compute cholesky factor of L^TL
		RealMatrix LTLcholesky(rank,rank);
		choleskyDecomposition(LTL,LTLcholesky);
		
		//A'b =  L(L^TL)^-1(L^TL)^-1z
		solveTriangularCholeskyInPlace<SolveAXB>(LTLcholesky,z);
		solveTriangularCholeskyInPlace<SolveAXB>(LTLcholesky,z);
		noalias(b)=prod(L,z);
	}
	//finally swap back into the unpermuted coordinate system
	swap_rows_inverted(permutation,b);
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
		swap_rows(permutation,B);
	else
		swap_columns(permutation,B);
	
	//matrix has full rank, means that we can use the typical cholesky inverse
	if(rank == m){
		solveTriangularCholeskyInPlace<System>(L,B);
	}
	else if (rank == 0){//A is 0
		B().clear();
	}
	else if(System::left)
	{
		//complex case. 
		//X=L(L^TL)^-1(L^TL)^-1 L^TB
		RealMatrix LTL = prod(trans(L),L);
		
		//compute Z= L^TB
		RealMatrix Z = prod(trans(L),B);
		
		//compute cholesky factor of L^TL
		RealMatrix LTLcholesky(rank,rank);
		choleskyDecomposition(LTL,LTLcholesky);
		
		//A'b =  L(L^TL)^-1(L^TL)^-1z
		solveTriangularCholeskyInPlace<System>(LTLcholesky,Z);
		solveTriangularCholeskyInPlace<System>(LTLcholesky,Z);
		noalias(B) = prod(L,Z);
	}else{
		//complex case. 
		//X=BL(L^TL)^-1(L^TL)^-1 L^T
		RealMatrix LTL = prod(trans(L),L);
		
		//compute z= L^TB
		RealMatrix Z = prod(B,L);
		
		//compute cholesky factor of L^TL
		RealMatrix LTLcholesky(rank,rank);
		choleskyDecomposition(LTL,LTLcholesky);
		
		//A'b =  L(L^TL)^-1(L^TL)^-1z
		solveTriangularCholeskyInPlace<System>(LTLcholesky,Z);
		solveTriangularCholeskyInPlace<System>(LTLcholesky,Z);
		noalias(B) = prod(Z,trans(L));
	}
	//finally swap back into the unpermuted coordinate system
	if(System::left)
		swap_rows_inverted(permutation,B);
	else
		swap_columns_inverted(permutation,B);
}

template<class System,class MatT,class VecT>
void shark::blas::generalSolveSystemInPlace(
	matrix_expression<MatT> const& A, 
	vector_expression<VecT>& b
){
	if( System::left){
		SIZE_CHECK(A().size1() == b().size());
		//reduce to the case of quadratic A
		//Ax=b => A^TAx=A'b => x= A'b = (A^TA)' Ab
		// with z = Ab => (A^TA) x= z
		//compute A^TA
		RealMatrix ATA = prod(trans(A),A);
		
		//compute z=Ab
		RealVector z = prod(trans(A),b);
		
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
		RealMatrix AAT = prod(A,trans(A));
		
		//compute z=Ab
		RealVector z = prod(A,b);
		
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
	if( System::left){
		SIZE_CHECK(A().size1() == B().size1());
		//reduce to the case of quadratic A
		//AX=B => A'AX=A'B => X= A'B = (A^TA)' A^TB
		// with Z = A^TB => (A^TA) X= Z
		//compute A^TA
		RealMatrix ATA = prod(trans(A),A);
		
		RealMatrix Z = prod(trans(A),B);
		
		//call recursively for the quadratic case
		solveSymmSemiDefiniteSystemInPlace<System>(ATA,Z);
		B() = Z;
	}
	else{
		SIZE_CHECK(A().size2() == B().size2());
		//~ //reduce to the case of quadratic A
		//~ //XA=B => XAA'=BA' => X = BA' = BA^T(AA^T)'
		//~ // with Z = BA^T => X(AA^T) = Z
		//~ //compute AAT
		RealMatrix AAT = prod(A,trans(A));
		
		RealMatrix Z= prod(B,trans(A));
		
		//call recursively for the quadratic case
		solveSymmSemiDefiniteSystemInPlace<System>(AAT,Z);
		B() = Z;		
	}
}

#endif
