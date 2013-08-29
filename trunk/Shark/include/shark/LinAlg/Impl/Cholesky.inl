/*!
 *  \brief Cholesky Decompositions for a Matrix A = LL^T
 *
 *
 *  \author  O. Krause
 *  \date    2012
 *
 *  \par Copyright (L) 1999-2001:
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
 *
 *
 */

#ifndef SHARK_IMPL_LINALG_CHOLESKY_INL
#define SHARK_IMPL_LINALG_CHOLESKY_INL

#ifdef SHARK_USE_ATLAS
#include <shark/LinAlg/BLAS/Impl/numeric_bindings/atlas/potrf.h>
#endif

#include <shark/Core/Math.h>

template<class MatrixT,class MatrixL>
void shark::blas::choleskyDecomposition(
	matrix_expression<MatrixT> const& A, 
	matrix_expression<MatrixL>& L
)
{
	size_t m = A().size1();
	ensureSize(L,m, m);
#ifdef SHARK_USE_ATLAS
	zero(L);
	for(std::size_t i = 0; i != m; ++i){
		for(std::size_t j = 0; j <= i; ++j){
			L()(i,j) = A()(i,j);
		}
	}
	if(bindings::potrf(CblasLower,L()) != 0){
		throw SHARKEXCEPTION("[Cholesky Decomposition] The Matrix is not positive definite");
	}
#else
	SIZE_CHECK(A().size1() == A().size2());

	
	for(size_t j = 0; j < m; j++) {
		for(size_t i = j; i < m; i++) {
			double s = A()(i, j);
			for(size_t k = 0; k < j; k++) {
				s -= L()(i, k) * L()(j, k);
			}
			if (i == j) {
				if(s<=0)
					throw SHARKEXCEPTION("[Cholesky Decomposition] The Matrix is not positive definite");
				L()(i, j) = std::sqrt(s);
			}
			else {
				L()(i, j) = s/L()(j , j);
				L()(j, i) = 0;
			}
		}
	}
#endif
}

template<class MatrixT,class MatrixL>
std::size_t shark::blas::pivotingCholeskyDecomposition(
	matrix_expression<MatrixT> const& Aref,
	PermutationMatrix& P,
	matrix_expression<MatrixL>& Lref
){
	typedef typename MatrixT::value_type Value;
	//we don't want to get annoyed by the expressions
	MatrixT const& A = Aref();
	MatrixL& L = Lref();
	
	//ensure sizes are correct
	SIZE_CHECK(A.size1() == A.size2());
	size_t m = A.size1();
	ensureSize(P,m);
	ensureSize(L,m,m);
	noalias(L) = A;
	
	//The Algorithms works as follows
	//we begin with the submatrix L^(0)= A
	//in step k we partion the matrix in the block
	//      |L11 | L12
	//L^(k)=|-----------
	//      |L21 | L^(k+1)
	//where L11 is a lxl submatrix
	//First suspect the case, when the matrix L11 is full rank
	//Then we can calculate:
	//compute matrix C such that L11=CC^T and store the results in L11 
	//solve XC=L21 for X and store in L21
	//L^(k+1)<-L^(k+1)-XX^T
	//L12 <- 0
	//However, if L11 does not have rank(l) we have to use pivoting
	//since L11 can not be decomposed in L11=CC^T
	//This is the case we suspect throughout in the algorithm.
	//In this case, we have to update L11 and L21 incrementally 
	//for every row k and check every time wehther the current pivot is
	//the highest in the remaining diagonal of the matrix.
	//(this also includes the part inside L^(k+1))
	//since we don't know in advance which column will be the next pivot, we
	//also need to delay the L21 update of every column until we know it's pivot
	
	
	//todo: experiment a bit with the sizes
	std::size_t blockSize = 20;
	
	//storage for pivot values
	RealVector pivotValues(m);
	
	//stopping criterion
	double epsilon = shark::sqr(m) * std::numeric_limits<Value>::epsilon() * norm_inf(diag(L));
	//double epsilon = 1.e-15;
	typedef matrix_range<MatrixL> SubL;
	
	for(std::size_t k = 0; k < m; k += blockSize){
		std::size_t currentSize = std::min(m-k,blockSize);//last iteration is smaller
		//partition of the matrix
		SubL Lk = subrange(L,k,m,k,m);
		vector_range<RealVector> pivots = subrange(pivotValues,k,m);
		//we have to dynamically update the pivot values
		//we start every block anew to prevent accumulation of rounding errors
		noalias(pivots) = diag(Lk);
		
		//update current block L11
		for(std::size_t j = 0; j != currentSize; ++j){
			//update pivot values
			if(j > 0){
				subrange(pivots,j,pivots.size()) -= sqr(unitTriangularColumn(Lk,j-1));
			}
			//get current pivot. if it is not equal to the j-th, we swap rows and columns
			//such that j == pivot and Lk(j,j) = pivots(pivot)
			std::size_t pivot = std::max_element(pivots.begin()+j,pivots.end())-pivots.begin();
			if(pivot != j){
				P(k+j) = pivot+k;
				row(L,k+j).swap(row(L,k+pivot));
				column(L,k+j).swap(column(L,k+pivot));
				std::swap(pivots(j),pivots(pivot));
			}
			
			//check whether we are finished
			Value pivotValue = pivots(j);
			if(pivotValue < epsilon){
				//the remaining part is so near 0, we can just ignore it
				Blocking<SubL> LkBlocked(Lk,j,j);
				zero(LkBlocked.lowerRight());
				return k+j;
			}
			
			//update current column
			Lk(j,j) = std::sqrt(pivotValue);
			//the last updates of columns k...k+j-1 did not change
			//this row, so do it now
			if(j > 0){
				Blocking<SubL> LkBlocked(Lk,j+1,j);
				//suppose you are the j-th column
				//than you want to get updated by the last
				//outer products which are induced by your column friends
				//Li...Lj-1
				//so you get the effect of
				//(L1L1T)^(j)+...+(Lj-1Lj-1)^(j)
				//=L1*L1j+L2*L2j+L3*L3j...
				//which is a matrix-vector product
				fast_prod(
					LkBlocked.lowerLeft(),
					subrange(row(Lk,j),0,j),
					unitTriangularColumn(Lk,j),
					1.0,-1.0
				);
			}
			unitTriangularColumn(Lk,j) /= Lk(j,j);
			//set block L12 to 0
			zero(subrange(Lk,j,j+1,j+1,Lk.size2()));
		}
		Blocking<SubL> LkBlocked(Lk,currentSize,currentSize);
		//if we are not finished do the block update
		if(k+currentSize < m){
			symmRankKUpdate(
				LkBlocked.lowerLeft(),
				LkBlocked.lowerRight(),
				1.0,-1.0
			);
		}
	}
	return m;
	
}
#endif
