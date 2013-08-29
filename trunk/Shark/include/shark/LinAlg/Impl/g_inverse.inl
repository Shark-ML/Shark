
/*!
 *  \brief Determines the generalized inverse matrix of an input matrix
 *         by using singular value decomposition or QR.
 *
 *  \author  O. Krause
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

#ifndef SHARK_LINALG_IMPL_G_INVERSE_INL
#define SHARK_LINALG_IMPL_G_INVERSE_INL

#include <shark/LinAlg/Cholesky.h>
#include <shark/LinAlg/solveSystem.h>

template<class MatA, class MatU>
void shark::blas::decomposedGeneralInverse(
	matrix_expression<MatA> const& matA,
	matrix_expression<MatU>& matU
){
	SIZE_CHECK(matA().size1() == matA().size2());
	
	std::size_t m = matA().size1();
	ensureSize(matU,m,m);
	
	//do a pivoting cholesky decomposition
	RealMatrix cholesky(m,m);
	PermutationMatrix permutation(m);
	std::size_t rank = pivotingCholeskyDecomposition(matA,permutation,cholesky);

	//full rank, means that we can use the typical cholesky inverse with pivoting
	//so U is P C^-1 P^T
	if(rank == m){
		identity(matU);
		solveTriangularSystemInPlace<SolveXAB,Upper>(trans(cholesky),matU);
		swapFullInverted(permutation,matU);
		return;
	}
	//complex case. 
	//A' = P C(C^TC)^-1(C^TC)^-1 C^T P^T
	//=> P^T U P = C(C^TC)^-1
	//<=> P^T U P (C^TC) = C
	zero(matU);
	RealMatrix CTC(rank,rank);
	symmRankKUpdate(trans(columns(cholesky,0,rank)),CTC);
	
	matrix_range<MatU> submat = columns(matU,0,rank);
	solveSymmSystem<SolveXAB>(CTC,submat,columns(cholesky,0,rank));
	swapFullInverted(permutation,matU);
}


template<class MatrixT>
shark::RealMatrix shark::blas::g_inverse(matrix_expression<MatrixT> const& matrixA){
	std::size_t m = matrixA().size1();
	std::size_t n = matrixA().size2();
	
	//compute AA^T
	RealMatrix AAT(m,m);
	symmRankKUpdate(matrixA,AAT);
	
	//get inverted decomposition of AAT: (AA^T)^-1=UU^T
	RealMatrix U(m,m);
	decomposedGeneralInverse(AAT,U);
	
	//compute result=A^T UU^T
	//compute UU^T and store in AAT, this is the inverse of AAT
	symmRankKUpdate(U,AAT);
	RealMatrix result(n,m);
	fast_prod(trans(matrixA),AAT,result);
	
	return result;
}

#endif
