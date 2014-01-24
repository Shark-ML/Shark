//===========================================================================
/*!
 * 
 * \file        atlas/gemm.hpp
 *
 * \brief       -
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
#ifndef SHARK_LINALG_BLAS_UBLASS_ATLAS_GEMM_HPP
#define SHARK_LINALG_BLAS_UBLASS_ATLAS_GEMM_HPP

#include "cblas_inc.hpp"

namespace shark { namespace blas { namespace bindings {

inline void gemm(CBLAS_ORDER const Order,
	CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	int M, int N, int K,
	float alpha, float const *A, int lda,
	float const *B, int ldb,
	float beta, float *C, int ldc
){
	cblas_sgemm(
		Order, TransA, TransB, M, N, K,
		alpha, A, lda,
		B, ldb,
		beta, C, ldc
	);
}

inline void gemm(CBLAS_ORDER const Order,
	CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	int M, int N, int K,
	double alpha, double const *A, int lda,
	double const *B, int ldb,
	double beta, double *C, int ldc
){
	cblas_dgemm(
		Order, TransA, TransB, M, N, K,
		alpha, 
		A, lda,
		B, ldb,
		beta, 
		C, ldc
	);
}

inline void gemm(CBLAS_ORDER const Order,
	CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	int M, int N, int K,
	float alpha,
	std::complex<float> const *A, int lda,
	std::complex<float> const *B, int ldb,
	float beta,
	std::complex<float>* C, int ldc
) {
	std::complex<float> alphaArg(alpha,0);
	std::complex<float> betaArg(beta,0);
	cblas_cgemm(
		Order, TransA, TransB, M, N, K,
		static_cast<void const *>(&alphaArg),
		static_cast<void const *>(A), lda,
		static_cast<void const *>(B), ldb,
		static_cast<void const *>(&betaArg),
		static_cast<void *>(C), ldc
	);
}

inline void gemm(CBLAS_ORDER const Order,
	CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	int M, int N, int K,
	double alpha,
	std::complex<double> const *A, int lda,
	std::complex<double> const *B, int ldb,
	double beta,
	std::complex<double>* C, int ldc
) {
	std::complex<double> alphaArg(alpha,0);
	std::complex<double> betaArg(beta,0);
	cblas_zgemm(
		Order, TransA, TransB, M, N, K,
		static_cast<void const *>(&alphaArg),
		static_cast<void const *>(A), lda,
		static_cast<void const *>(B), ldb,
		static_cast<void const *>(&betaArg),
		static_cast<void *>(C), ldc
	);
}

// C <- alpha * A * B + beta * C
template <typename MatrA, typename MatrB, typename MatrC>
void gemm(
	matrix_expression<MatrA> const &matA,
	matrix_expression<MatrB> const &matB,
	matrix_expression<MatrC>& matC, 
	typename MatrC::value_type alpha,
	boost::mpl::true_
) {
	SIZE_CHECK(matA().size1() == matC().size1());
	SIZE_CHECK(matB().size2() == matC().size2());
	SIZE_CHECK(matA().size2()== matB().size1());
	
	CBLAS_TRANSPOSE transA = traits::same_orientation(matA,matC)?CblasNoTrans:CblasTrans;
	CBLAS_TRANSPOSE transB = traits::same_orientation(matB,matC)?CblasNoTrans:CblasTrans;
	std::size_t m = matC().size1();
	std::size_t n = matC().size2();
	std::size_t k = matA().size2();
	CBLAS_ORDER stor_ord = (CBLAS_ORDER) storage_order<typename MatrC::orientation >::value;

	gemm(stor_ord, transA, transB, (int)m, (int)n, (int)k, alpha,
		traits::storage(matA()),
		traits::leading_dimension(matA()),
		traits::storage(matB()),
		traits::leading_dimension(matB()),
		typename MatrC::value_type(1),
		traits::storage(matC()),
		traits::leading_dimension(matC())
	);
}


template<class Storage1, class Storage2, class Storage3, class T1, class T2, class T3>
struct optimized_gemm_detail{
	typedef boost::mpl::false_ type;
};
template<>
struct optimized_gemm_detail<
	dense_tag, dense_tag, dense_tag, 
	double, double, double
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_gemm_detail<
	dense_tag, dense_tag, dense_tag, 
	float, float, float
>{
	typedef boost::mpl::true_ type;
};

template<>
struct optimized_gemm_detail<
	dense_tag, dense_tag, dense_tag,
	std::complex<double>, std::complex<double>, std::complex<double>
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_gemm_detail<
	dense_tag, dense_tag, dense_tag,
	std::complex<float>, std::complex<float>, std::complex<float>
>{
	typedef boost::mpl::true_ type;
};

template<class M1, class M2, class M3>
struct  has_optimized_gemm
: public optimized_gemm_detail<
	typename M1::storage_category,
	typename M2::storage_category,
	typename M3::storage_category,
	typename M1::value_type,
	typename M2::value_type,
	typename M3::value_type
>{};

}}}

#endif
