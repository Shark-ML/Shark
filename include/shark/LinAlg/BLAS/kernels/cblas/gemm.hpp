//===========================================================================
/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2010
 *
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
//===========================================================================
#ifndef SHARK_LINALG_BLAS_KERNELS_CBLAS_GEMM_HPP
#define SHARK_LINALG_BLAS_KERNELS_CBLAS_GEMM_HPP

#include "cblas_inc.hpp"

namespace shark { namespace blas { namespace bindings {

inline void gemm(
	CBLAS_ORDER const Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	int M, int N, int K,
	float alpha, float const *A, int lda,
	float const *B, int ldb,
	float beta, float *C, int ldc
){
	cblas_sgemm(
		Order, TransA, TransB,
		M, N, K,
		alpha, A, lda,
		B, ldb,
		beta, C, ldc
	);
}

inline void gemm(
	CBLAS_ORDER const Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	int M, int N, int K,
	double alpha, double const *A, int lda,
	double const *B, int ldb,
	double beta, double *C, int ldc
){
	cblas_dgemm(
		Order, TransA, TransB,
		M, N, K,
		alpha, 
		A, lda,
		B, ldb,
		beta, 
		C, ldc
	);
}

inline void gemm(
	CBLAS_ORDER const Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	int M, int N, int K,
	std::complex<float> alpha,
	std::complex<float> const *A, int lda,
	std::complex<float> const *B, int ldb,
	std::complex<float> beta,
	std::complex<float>* C, int ldc
) {
	cblas_cgemm(
		Order, TransA, TransB,
		M, N, K,
		reinterpret_cast<cblas_float_complex_type const *>(&alpha),
		reinterpret_cast<cblas_float_complex_type const *>(A), lda,
		reinterpret_cast<cblas_float_complex_type const *>(B), ldb,
		reinterpret_cast<cblas_float_complex_type const *>(&beta),
		reinterpret_cast<cblas_float_complex_type *>(C), ldc
	);
}

inline void gemm(
	CBLAS_ORDER const Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
	int M, int N, int K,
	std::complex<double> alpha,
	std::complex<double> const *A, int lda,
	std::complex<double> const *B, int ldb,
	std::complex<double> beta,
	std::complex<double>* C, int ldc
) {
	cblas_zgemm(
		Order, TransA, TransB,
		M, N, K,
		reinterpret_cast<cblas_double_complex_type const *>(&alpha),
		reinterpret_cast<cblas_double_complex_type const *>(A), lda,
		reinterpret_cast<cblas_double_complex_type const *>(B), ldb,
		reinterpret_cast<cblas_double_complex_type const *>(&beta),
		reinterpret_cast<cblas_double_complex_type *>(C), ldc
	);
}

// C <- alpha * A * B + beta * C
template <typename MatA, typename MatB, typename MatC>
void gemm(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag> const& B,
	matrix_expression<MatC, cpu_tag>& C, 
	typename MatC::value_type alpha,
	boost::mpl::true_
) {
	SIZE_CHECK(A().size1() == C().size1());
	SIZE_CHECK(B().size2() == C().size2());
	SIZE_CHECK(A().size2()== B().size1());
	
	CBLAS_TRANSPOSE transA = std::is_same<typename MatA::orientation,typename MatC::orientation>::value?CblasNoTrans:CblasTrans;
	CBLAS_TRANSPOSE transB = std::is_same<typename MatB::orientation,typename MatC::orientation>::value?CblasNoTrans:CblasTrans;
	std::size_t m = C().size1();
	std::size_t n = C().size2();
	std::size_t k = A().size2();
	CBLAS_ORDER stor_ord = (CBLAS_ORDER) storage_order<typename MatC::orientation >::value;

	auto storageA = A().raw_storage();
	auto storageB = B().raw_storage();
	auto storageC = C().raw_storage();
	gemm(stor_ord, transA, transB, (int)m, (int)n, (int)k, alpha,
		storageA.values,
	        storageA.leading_dimension,
		storageB.values,
	        storageB.leading_dimension,
		typename MatC::value_type(1),
		storageC.values,
	        storageC.leading_dimension
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
	typename M1::storage_type::storage_tag,
	typename M2::storage_type::storage_tag,
	typename M3::storage_type::storage_tag,
	typename M1::value_type,
	typename M2::value_type,
	typename M3::value_type
>{};

}}}

#endif
