//===========================================================================
/*!
 * 
 *
 * \brief      Contains the lapack bindings for the symmetric eigenvalue problem syev.
 *
 * \author      O. Krause
 * \date        2010
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
//===========================================================================
#ifndef SHARK_LINALG_BLAS_KERNELS_LAPACK_SYEV_HPP
#define SHARK_LINALG_BLAS_KERNELS_LAPACK_SYEV_HPP

#include "fortran.hpp"
#include "../traits.hpp"

#define SHARK_LAPACK_DSYEV FORTRAN_ID(dsyev)

extern "C"{
void SHARK_LAPACK_DSYEV( 
	const char* jobz, const char* uplo, const int *n,
	double* a, const int * lda, double* w,
	double* work, const int * lwork, int* info
);
}



namespace shark { namespace blas { namespace bindings {

inline void syev(
	int n, bool upper,
	double* A, int lda,
	double* eigenvalues
){
	if(n == 0) return;
	int lwork = std::min<int>(130,n)*n;
	double* work = new double[lwork];
	int info;
	char job = 'V';
	char uplo = upper?'U':'L';
	SHARK_LAPACK_DSYEV(&job, &uplo, &n, A, &lda,eigenvalues,work,&lwork,&info);
	delete work;
	
}


template <typename MatrA, typename VectorB>
void syev(
	matrix_expression<MatrA>& matA,
	vector_expression<VectorB>& eigenValues
) {
	SIZE_CHECK(matA().size1() == matA().size2());
	SIZE_CHECK(matA().size1() == eigenValues().size());
	
	std::size_t n = matA().size1();
	bool upper = false;
	//lapack is column major storage.
	if(boost::is_same<typename MatrA::orientation, blas::row_major>::value){
		upper = !upper;
	}
	syev(
		n, upper,
		traits::storage(matA()),
		traits::leading_dimension(matA()),
		traits::storage(eigenValues())
	);
	
	matA() = trans(matA);
	
	//reverse eigenvectors and eigenvalues
	for (int i = 0; i < n-i-1; i++)
	{
		int l =  n-i-1;
		std::swap(eigenValues()( l ),eigenValues()( i ));
	}
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n-i-1; i++)
		{
			int l =  n-i-1;
			std::swap(matA()( j , l ), matA()( j , i ));
		}
	}
}

}}}

#undef SHARK_LAPACK_DSYEV

#endif
