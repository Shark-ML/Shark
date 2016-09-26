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
#ifndef SHARK_LINALG_BLAS_KERNELS_CLBLAS_TRMM_HPP
#define SHARK_LINALG_BLAS_KERNELS_CLBLAS_TRMM_HPP

#include "clblas_inc.hpp"

namespace shark {namespace blas {namespace bindings {

inline void trmm(
	clblasOrder order, clblasSide side, clblasUplo uplo, clblasTranspose transA, clblasDiag diag,
	std::size_t M, std::size_t N,
	boost::compute::vector<float> const& A, std::size_t offA, std::size_t lda,
	boost::compute::vector<float> const& B, std::size_t offB, std::size_t ldb,
	std::size_t numCommandQueues, cl_command_queue* commandQueues,
	std::size_t numEventsInWaitList, cl_event const*  eventWaitList, cl_event* events
){
	clblasStrmm(
		order, side, uplo, transA, diag, M, N,1.0,
		A.get_buffer().get(), offA, lda,
		B.get_buffer().get(), offB, ldb,
		numCommandQueues, commandQueues,
		numEventsInWaitList, eventWaitList, events
	);
}

inline void trmm(
	clblasOrder order, clblasSide side, clblasUplo uplo, clblasTranspose transA, clblasDiag diag,
	std::size_t M, std::size_t N,
	boost::compute::vector<double> const& A, std::size_t offA, std::size_t lda,
	boost::compute::vector<double> const& B, std::size_t offB, std::size_t ldb,
	std::size_t numCommandQueues, cl_command_queue* commandQueues,
	std::size_t numEventsInWaitList, cl_event const*  eventWaitList, cl_event* events
){
	clblasDtrmm(
		order, side, uplo, transA, diag, M,N,1.0,
		A.get_buffer().get(), offA, lda,
		B.get_buffer().get(), offB, ldb,
		numCommandQueues, commandQueues,
		numEventsInWaitList, eventWaitList, events
	);
}

}

namespace kernels{

template <bool upper, bool unit, typename MatA, typename MatB>
void trmm(
	matrix_expression<MatA, gpu_tag> const& A,
	matrix_expression<MatB, gpu_tag>& B
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == B().size1());
	std::size_t m = A().size1();
	std::size_t n = B().size2();
	
	clblasOrder stor_ordA = (clblasOrder) clblas::storage_order<typename MatA::orientation>::value;
	clblasDiag diag = unit?clblasUnit:clblasNonUnit;
	clblasUplo uplo = upper?clblasUpper:clblasLower;
	clblasTranspose trans = clblasNoTrans;
	
	//special case: MatA and MatB do not have same storage order. in this case compute as
	//AB->B^TA^T where transpose of B is done implicitely by exchanging storage order
	clblasOrder stor_ordB= (clblasOrder) clblas::storage_order<typename MatB::orientation>::value;
	if(stor_ordA != stor_ordB){
		trans = clblasTrans;
		uplo = upper?clblasLower:clblasUpper;
	}
	
	auto storageA = A().raw_storage();
	auto storageB = B().raw_storage();
	bindings::trmm(stor_ordB, clblasLeft, uplo, trans, diag, m, n,
		storageA.buffer, storageA.offset, storageA.leading_dimension,
		storageB.buffer, storageB.offset, storageB.leading_dimension,
		1, &(B().queue().get()),
		0, nullptr, nullptr
	);
}

}}}
#endif
