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

#ifndef SHARK_LINALG_BLAS_KERNELS_CLBLAS_TRSM_HPP
#define SHARK_LINALG_BLAS_KERNELS_CLBLAS_TRSM_HPP

#include "clblas_inc.hpp"

///solves systems of triangular matrices

namespace shark {namespace blas {namespace bindings {
	
inline void trsm(
	clblasOrder order, clblasSide side, clblasUplo uplo, clblasTranspose transA, clblasDiag diag,
	std::size_t N, std::size_t nRHS,
	boost::compute::vector<float> const& A, std::size_t offA, std::size_t lda,
	boost::compute::vector<float>& B, std::size_t offB, std::size_t ldb,
	std::size_t numCommandQueues, cl_command_queue* commandQueues,
	std::size_t numEventsInWaitList, cl_event const*  eventWaitList, cl_event* events
){
	clblasStrsm(
		order, side, uplo, transA, diag, N, nRHS,1.0,
		A.get_buffer().get(), offA, lda,
		B.get_buffer().get(), offB, ldb,
		numCommandQueues, commandQueues,
		numEventsInWaitList, eventWaitList, events
	);
}

inline void trsm(
	clblasOrder order, clblasSide side, clblasUplo uplo, clblasTranspose transA, clblasDiag diag,
	std::size_t N, std::size_t nRHS,
	boost::compute::vector<double> const& A, std::size_t offA, std::size_t lda,
	boost::compute::vector<double>& B, std::size_t offB, std::size_t ldb,
	std::size_t numCommandQueues, cl_command_queue* commandQueues,
	std::size_t numEventsInWaitList, cl_event const*  eventWaitList, cl_event* events
){
	clblasDtrsm(
		order, side, uplo, transA, diag, N, nRHS,1.0,
		A.get_buffer().get(), offA, lda,
		B.get_buffer().get(), offB, ldb,
		numCommandQueues, commandQueues,
		numEventsInWaitList, eventWaitList, events
	);
}

// trsm(): solves A system of linear equations A * X = B
//             when A is A triangular matrix
template <bool Upper, bool unit,typename MatA, typename MatB>
void trsm(
	matrix_expression<MatA, cpu_tag> const& A,
	matrix_expression<MatB, cpu_tag>& B
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size1() == B().size1());
	
	//orientation is defined by the second argument
	clblasOrder stor_ord = (clblasOrder) clblas::storage_order<typename MatB::orientation>::value;
	//if orientations do not match, wecan interpret this as transposing A
	bool transposeA =  !std::is_same<typename MatA::orientation,typename MatB::orientation>::value;
	clblasDiag diag = unit?clblasUnit:clblasNonUnit;
	clblasUplo uplo = (Upper != transposeA)?clblasUpper:clblasLower;
	clblasTranspose transA = transposeA?clblasTrans:clblasNoTrans;
	
	int n = B().size1();
	int nrhs = B().size2();
	auto storageA = A().raw_storage();
	auto storageB = B().raw_storage();
	trsm(stor_ord, clblasLeft, uplo, transA, diag, n, nrhs,
		storageA.buffer, storageA.offset, storageA.leading_dimension,
		storageB.buffer, storageB.offset, storageB.leading_dimension,
		1, &(B().queue().get()),
		0, nullptr, nullptr
	);
}
}}}
#endif
