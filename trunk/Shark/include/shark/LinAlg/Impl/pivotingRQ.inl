/*!
 *  \brief Computes the RQ-Decomposition of a matrix
 *
 *  \author  O.Krause
 *  \date    2012
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

#ifndef SHARK_LINALG_PIVOTINGRQ_INL
#define SHARK_LINALG_PIVOTINGRQ_INL

#include <shark/LinAlg/rotations.h>
#include <shark/LinAlg/solveSystem.h>
#include <algorithm>

template<class MatrixT,class MatrixU>
std::size_t shark::blas::pivotingRQHouseholder
(
	blas::matrix_expression<MatrixT> const& matrixA,
	blas::matrix_container<MatrixU>& matrixR,
	blas::matrix_container<MatrixU>& householderTransform,
	blas::permutation_matrix<std::size_t> & permutation
){
	std::size_t m = matrixA().size1();
	std::size_t n = matrixA().size2();
	std::size_t k = std::min(m,n);
	
	//initialize
	permutation.resize(m);
	matrixR() = matrixA;
	householderTransform().resize(k,n);
	householderTransform().clear();
	
	for(std::size_t i = 0; i != m; ++i)
		permutation(i) = i;
	
	//squared norm of the rows
	//in every step of computation, we choose the row with the highest norm
	RealVector norms(sumColumns(sqr(matrixA)));
	
	//threshold for rank determination. if the squared norm is lower than that
	//the matrix is considered to be 0.
	//shameless ripoff of the Eigen Library.
	//todo: this turned out to reveal false rnaks because the threshold is really
	//small. we use a bigger threshold here, until a rank revealing step is added
	//which is likely to be slow.
//	double threshold = *std::max_element(norms.begin(),norms.end());
//	threshold *= sqr(std::numeric_limits<double>::epsilon());//very small!
//	threshold /= n;
	double threshold = 1.e-15;
	
	//todo: make block oriented
	std::size_t rank = 0;
	for(std::size_t i = 0; i != k; ++i,++rank){
		//we work mainly on a subrange of R aside from pivoting
		blas::matrix_range<MatrixU> subR = subrange(matrixR(),i,m,i,n);
		
		//step 1: pivoting
		std::size_t pivot = std::max_element(norms.begin()+i,norms.end())-norms.begin();
		//the values in the array are not numerically stable, so we have to recompute the pivot
		//before we can tell, whether we are done or not
		double pivotValue = norm_sqr(row(subR,pivot-i));
		//step1.1: rank analysis
		//test, whether we are done. there is no need in swapping if the remainder of the matrix is empty
		//todo: make numerically more sound
		if(pivotValue < threshold ){
			break;
		}
		//if the pivot does not equal the current i, we swap the matrix rows
		if(pivot != i){
			row(matrixR(),i).swap(row(matrixR(),pivot));
			permutation(i) = pivot;
			std::swap(norms(i),norms(pivot));
		}
		
		//step 2: apply householder transformation
		//get the exact part of the current row which is used to store the householder reflection
		blas::matrix_row<MatrixU> r = row(householderTransform(),i);
		blas::vector_range<blas::matrix_row<MatrixU> > reflection = subrange(r,i,n);
		
		//now we are sure that our current pivot is at index i and do a 
		//householder transformation on the first row
		double tau = createHouseholderReflection(row(subR,0),reflection);
		applyHouseholderOnTheRight(subR,reflection,tau);
		noalias(subrange(norms,i,m)) -= sqr(column(subR,0));	
	}
	
	//todo: This RQ algorithm is not really rank revealing, but according to the literature
	//"a good starting point". So there is still something to be done 
	
	//step3: fill the remainder of R with zeros
	subrange(matrixR(),rank,m,rank,n) = RealZeroMatrix(m-rank,n-rank);
	return rank;
}

template<class MatrixT,class Mat>
std::size_t shark::blas::pivotingRQ
(
	blas::matrix_expression<MatrixT> const& matrixA,
	blas::matrix_container<Mat>& matrixR,
	blas::matrix_container<Mat>& matrixQ,
	blas::permutation_matrix<std::size_t> & permutation
){
	std::size_t n = matrixA().size2();
	
	//first we do the householder transformation to get R and the householder transformations
	Mat U;//householder matrix is called U in the paper
	std::size_t rank = pivotingRQHouseholder(matrixA, matrixR(), U,permutation);
	
	//having these, Q is quite efficient to compute using an algorithm from
	//T. Joffrain: On Accumulating Householder Transformations, 
	//ACM Transactions on Mathematical Software
	//The goal is to compute from the single householder transformations a matrix incorporating all of them
	//it turns out, that this is:
	//(I-u_1u_1^T)*...(I-u_lu_rank^T)= I-UT^-1U
	//with T = upperDiag(U^TU) -1/2 diagof(U^T U)
	//this is of course for column major lower triangular matrices, 
	//meaning that we have to transpose our matrices U.
	Mat T(rank,rank);
	T.clear();
	symmRankKUpdate(rows(U,0,rank),T);
	//we now have to explicitely zero the lower half
	//and the diagonal needs to be divided by two.
	for(std::size_t i = 0; i != rank; ++i){
		for(std::size_t j = 0; j != i; ++j){
			T(i,j) = 0;
		}
		T(i,i) /= 2.0;
	}
	
	//Q is now I-UT^-1U T having full rank.
	//again for us it is U^T T^-1 U
	//we first solve for T^-1 U = Temp <=> T Temp = U
	//this saves us from computing the inverse of T
	Mat InvTU(rows(U,0,rank));
	solveTriangularSystemInPlace<SolveAXB,Lower>(trans(T), InvTU);
	matrixQ().resize(n,n);
	//now Compute U^T temp = U^T T^-1 U
	fast_prod(trans(rows(U,0,rank)),InvTU,matrixQ);
	matrixQ()*=-1;
	matrixQ()+=RealIdentityMatrix(n);

	//testing algorithm
//	matrixQ().resize(n,n);
//	matrixQ().clear();
//	for(std::size_t i = 0; i != n; ++i){
//		matrixQ()(i,i) = 1;
//	}
//	//apply transformations one after another.
//	for(std::size_t i = 0; i != std::min(k,rank); ++i){
//		//std::cout<<matrixQ<<std::endl;
//		blas::vector_range<blas::matrix_row<Mat> const > reflection = subrange(row(U,i),i,n);
//		
//		double tau = 0;
//		if(reflection.size() != 1){
//			tau = 2/norm_sqr(reflection);
//		}
//		blas::matrix_range<Mat> subQ = rows(matrixQ(),i,n);
//		applyHouseholderOnTheLeft(subQ,reflection,tau);
//		//std::cout<<matrixQ()<<std::endl;
//		//std::cout<<i<<" "<<reflection<<std::endl;
//	}
	return rank;
}

#endif
