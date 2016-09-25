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

#ifndef SHARK_LINALG_BLAS_KERNELS_CLBLAS_TRSV_HPP
#define SHARK_LINALG_BLAS_KERNELS_CLBLAS_TRSV_HPP

#include "clblas_inc.hpp"

///solves systems of triangular matrices

namespace shark {namespace blas{ namespace bindings {

inline void trsv(
	clblasOrder order, clblasUplo uplo, clblasTranspose transA, clblasDiag diag,
	std::size_t N,
	boost::compute::vector<float> const& A, std::size_t offA, std::size_t lda,
	boost::compute::vector<float> const& x, std::size_t offx, std::size_t incx,
	std::size_t numCommandQueues, cl_command_queue* commandQueues,
	std::size_t numEventsInWaitList, cl_event const*  eventWaitList, cl_event* events
){
	clblasStrsv (
		order, uplo, transA, diag, N,
		A.get_buffer().get(), offA, lda,
		x.get_buffer().get(), offx, (int)incx,
		numCommandQueues, commandQueues,
		numEventsInWaitList, eventWaitList, events
	);
}

inline void trsv(
	clblasOrder order, clblasUplo uplo, clblasTranspose transA, clblasDiag diag,
	std::size_t N,
	boost::compute::vector<double> const& A, std::size_t offA, std::size_t lda,
	boost::compute::vector<double> const& x, std::size_t offx, std::size_t incx,
	std::size_t numCommandQueues, cl_command_queue* commandQueues,
	std::size_t numEventsInWaitList, cl_event const*  eventWaitList, cl_event* events
){
	clblasDtrsv (
		order, uplo, transA, diag, N,
		A.get_buffer().get(), offA, lda,
		x.get_buffer().get(), offx, (int)incx,
		numCommandQueues, commandQueues,
		numEventsInWaitList, eventWaitList, events
	);
}

}

namespace trsv{

// trsv(): solves A system of linear equations A * x = b
//             when A is A triangular matrix.
template <bool Upper,bool Unit,typename MatA, typename V>
void trsv(
	matrix_expression<MatA, gpu_tag> const& A, 
	vector_expression<V, gpu_tag>& b
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size1()== b().size());
	clblasDiag diag = Unit?clblasUnit:clblasNonUnit;
	clblasOrder stor_ord= (clblasOrder) clblas::storage_order<typename MatA::orientation>::value;
	clblasUplo uplo = Upper?clblasUpper:clblasLower;
	
	auto storageA = A().raw_storage();
	auto storageb = b().raw_storage();
	trsv(stor_ord, uplo, clblasNoTrans,diag, A().size1(),
	        storageA.buffer, storageA.offset, storageA.leading_dimension,
		storageb.buffer, storageb.offset, storageb.stride,
		1, &(b().queue()),
		0, nullptr, nullptr
	);
}

}}}
#endif
