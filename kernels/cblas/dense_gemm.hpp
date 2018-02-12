//===========================================================================
/*!
 *
 *
 * \brief       cblas binding for dense gemm
 *
 * \author      O. Krause
 * \date        2016
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
#ifndef REMORA_KERNELS_CBLAS_DENSE_GEMM_HPP
#define REMORA_KERNELS_CBLAS_DENSE_GEMM_HPP

#include "cblas_inc.hpp"
#include "../../proxy_expressions.hpp"
#include "../../assignment.hpp"
#include "../../dense.hpp"
#include "../default/simd.hpp"
#include <type_traits>
namespace remora{ namespace bindings {

inline void dense_gemm(
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

inline void dense_gemm(
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

inline void dense_gemm(
	CBLAS_ORDER const Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
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
		Order, TransA, TransB,
		M, N, K,
		reinterpret_cast<cblas_float_complex_type const *>(&alphaArg),
		reinterpret_cast<cblas_float_complex_type const *>(A), lda,
		reinterpret_cast<cblas_float_complex_type const *>(B), ldb,
		reinterpret_cast<cblas_float_complex_type const *>(&betaArg),
		reinterpret_cast<cblas_float_complex_type *>(C), ldc
	);
}

inline void dense_gemm(
	CBLAS_ORDER const Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
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
		Order, TransA, TransB,
		M, N, K,
		reinterpret_cast<cblas_double_complex_type const *>(&alphaArg),
		reinterpret_cast<cblas_double_complex_type const *>(A), lda,
		reinterpret_cast<cblas_double_complex_type const *>(B), ldb,
		reinterpret_cast<cblas_double_complex_type const *>(&betaArg),
		reinterpret_cast<cblas_double_complex_type *>(C), ldc
	);
}

//optimized cblas version
template <typename MatA, typename MatB, typename MatC>
void dense_gemm(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag> const& B,
	matrix_expression<MatC, cpu_tag>& C,
	typename MatC::value_type alpha,
	std::true_type
){
	static_assert(std::is_same<typename MatC::orientation,row_major>::value,"C must be row major");

	CBLAS_TRANSPOSE transA = std::is_same<typename MatA::orientation,typename MatC::orientation>::value?CblasNoTrans:CblasTrans;
	CBLAS_TRANSPOSE transB = std::is_same<typename MatB::orientation,typename MatC::orientation>::value?CblasNoTrans:CblasTrans;
	std::size_t m = C().size1();
	std::size_t n = C().size2();
	std::size_t k = A().size2();
	CBLAS_ORDER stor_ord = (CBLAS_ORDER) storage_order<typename MatC::orientation >::value;

	auto storageA = A().raw_storage();
	auto storageB = B().raw_storage();
	auto storageC = C().raw_storage();
	dense_gemm(stor_ord, transA, transB, (int)m, (int)n, (int)k, alpha,
		storageA.values,
		storageA.leading_dimension,
		storageB.values,
		storageB.leading_dimension,
		typename MatC::value_type(1),
		storageC.values,
		storageC.leading_dimension
	);
}

template <typename MatA, typename MatB, typename MatC>
void dense_gemm(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag> const& B,
	matrix_expression<MatC, cpu_tag>& C,
	typename MatC::value_type alpha,
	std::false_type
){
	typedef typename MatC::value_type value_type;
	std::size_t const tile_size = 512;
	static const std::size_t align = 64;
	std::size_t size1 = C().size1();
	std::size_t size2 = C().size2();
	std::size_t num_blocks = (A().size2()+tile_size-1)/tile_size;
	boost::alignment::aligned_allocator<value_type,align> allocator;
	value_type* A_pointer = allocator.allocate(size1 * tile_size);
	value_type* B_pointer = allocator.allocate(size2 * tile_size);
	for(std::size_t k = 0; k != num_blocks; ++k){
		std::size_t start_k = k * tile_size;
		std::size_t current_size = std::min(tile_size,A().size2() - start_k);
		auto A_block = adapt_matrix(size1, current_size, A_pointer);
		auto B_block = adapt_matrix(current_size, size2, B_pointer);
		noalias(A_block) = subrange(A, 0, size1, start_k, start_k + current_size);
		noalias(B_block) = subrange(B, start_k, start_k + current_size, 0, size2);
		dense_gemm(A_block, B_block, C, alpha, std::true_type());
	}
	allocator.deallocate(A_pointer, size1 * tile_size);
	allocator.deallocate(B_pointer, size1 * tile_size);
}


template<class M1, class M2, class M3>
struct has_optimized_gemm: std::integral_constant<bool,
	allowed_cblas_type<typename M1::value_type>::type::value
	&& std::is_same<typename M1::value_type, typename M2::value_type>::value
	&& std::is_same<typename M1::value_type, typename M3::value_type>::value
	&& std::is_base_of<dense_tag, typename M1::storage_type::storage_tag>::value
	&& std::is_base_of<dense_tag, typename M2::storage_type::storage_tag>::value 
	&& std::is_base_of<dense_tag, typename M3::storage_type::storage_tag>::value 
>{};

template <typename MatA, typename MatB, typename MatC>
void dense_gemm(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag> const& B,
	matrix_expression<MatC, cpu_tag>& C,
	typename MatC::value_type alpha
){
	REMORA_SIZE_CHECK(A().size1() == C().size1());
	REMORA_SIZE_CHECK(B().size2() == C().size2());
	REMORA_SIZE_CHECK(A().size2()== B().size1());
	dense_gemm(
		A,B,C,alpha,
		typename has_optimized_gemm<MatA,MatB,MatC>::type()
	);
}

}}

#endif
