//===========================================================================
/*!
 * 
 *
 * \brief       -
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
#ifndef SHARK_LINALG_BLAS_KERNELS_CLBLAS_GEMM_HPP
#define SHARK_LINALG_BLAS_KERNELS_CLBLAS_GEMM_HPP

#include "clblas_inc.hpp"

namespace shark { namespace blas { namespace bindings {

inline void gemm(
	clblasOrder order, clblasTranspose transA, clblasTranspose transB,
	std::size_t M, std::size_t N, std::size_t K, float alpha,
	boost::compute::vector<float> const& A, std::size_t offA, std::size_t lda,
	boost::compute::vector<float> const& B, std::size_t offB, std::size_t ldb,
	float beta,
	boost::compute::vector<float> const& C, std::size_t offC, std::size_t ldc,
	std::size_t numCommandQueues, cl_command_queue* commandQueues,
	std::size_t numEventsInWaitList, cl_event const*  eventWaitList, cl_event* events
){
	clblasSgemm(
		order, transA, transB,
		M, N, K,
		alpha,
		A.get_buffer().get(), offA, lda,
		B.get_buffer().get(), offB, ldb,
		beta,
		C.get_buffer().get(), offC, ldc,
		numCommandQueues, commandQueues,
		numEventsInWaitList, eventWaitList, events
	);
}

inline void gemm(
	clblasOrder order, clblasTranspose transA, clblasTranspose transB,
	std::size_t M, std::size_t N, std::size_t K, double alpha,
	boost::compute::vector<double> const& A, std::size_t offA, std::size_t lda,
	boost::compute::vector<double> const& B, std::size_t offB, std::size_t ldb,
	double beta,
	boost::compute::vector<double> const& C, std::size_t offC, std::size_t ldc,
	std::size_t numCommandQueues, cl_command_queue* commandQueues,
	std::size_t numEventsInWaitList, cl_event const*  eventWaitList, cl_event* events
){
	clblasDgemm(
		order, transA, transB,
		M, N, K,
		alpha,
		A.get_buffer().get(), offA, lda,
		B.get_buffer().get(), offB, ldb,
		beta,
		C.get_buffer().get(), offC, ldc,
		numCommandQueues, commandQueues,
		numEventsInWaitList, eventWaitList, events
	);
}

}

namespace kernels{

// C <- alpha * A * B + beta * C
template <typename MatA, typename MatB, typename MatC>
void gemm(
	matrix_expression<MatA, gpu_tag> const& A,
	matrix_expression<MatB, gpu_tag> const& B,
	matrix_expression<MatC, gpu_tag>& C, 
	typename MatC::value_type const& alpha
) {
	SIZE_CHECK(A().size1() == C().size1());
	SIZE_CHECK(B().size2() == C().size2());
	SIZE_CHECK(A().size2()== B().size1());
	
	clblasTranspose transA = std::is_same<typename MatA::orientation,typename MatC::orientation>::value?clblasNoTrans:clblasTrans;
	clblasTranspose transB = std::is_same<typename MatB::orientation,typename MatC::orientation>::value?clblasNoTrans:clblasTrans;
	std::size_t m = C().size1();
	std::size_t n = C().size2();
	std::size_t k = A().size2();
	clblasOrder stor_ord = (clblasOrder) clblas::storage_order<typename MatC::orientation >::value;

	auto storageA = A().raw_storage();
	auto storageB = B().raw_storage();
	auto storageC = C().raw_storage();
	bindings::gemm(stor_ord, transA, transB, m, n, k, alpha,
		storageA.buffer, storageA.offset, storageA.leading_dimension,
		storageB.buffer, storageB.offset, storageB.leading_dimension,
		typename MatC::value_type(1),
		storageC.buffer, storageC.offset, storageC.leading_dimension,
		1, &(C().queue().get()),
		0, nullptr, nullptr
	);
}

}}}

#endif
