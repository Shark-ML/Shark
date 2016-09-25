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
#ifndef SHARK_LINALG_BLAS_KERNELS_CLBLAS_GEMV_HPP
#define SHARK_LINALG_BLAS_KERNELS_CLBLAS_GEMV_HPP

#include "clblas_inc.hpp"

namespace shark { namespace blas { namespace bindings {
	
inline void gemv(
	clblasOrder order, clblasTranspose transA,
	std::size_t M, std::size_t N, float alpha,
	boost::compute::vector<float> const& A, std::size_t offA, std::size_t lda,
	boost::compute::vector<float> const& x, std::size_t offx, std::size_t incx,
	float beta,
	boost::compute::vector<float> const& y, std::size_t offy, std::size_t incy,
	std::size_t numCommandQueues, cl_command_queue* commandQueues,
	std::size_t numEventsInWaitList, cl_event const*  eventWaitList, cl_event* events
){
	clblasSgemv (
		order, transA,M, N,alpha,
		A.get_buffer().get(), offA, lda,
		x.get_buffer().get(), offx, (int)incx,
		beta,
		y.get_buffer().get(), offy, (int)incy,
		numCommandQueues, commandQueues,
		numEventsInWaitList, eventWaitList, events
	);
}

inline void gemv(
	clblasOrder order, clblasTranspose transA,
	std::size_t M, std::size_t N, double alpha,
	boost::compute::vector<double> const& A, std::size_t offA, std::size_t lda,
	boost::compute::vector<double> const& x, std::size_t offx, std::size_t incx,
	double beta,
	boost::compute::vector<double> const& y, std::size_t offy, std::size_t incy,
	std::size_t numCommandQueues, cl_command_queue* commandQueues,
	std::size_t numEventsInWaitList, cl_event const*  eventWaitList, cl_event* events
){
	clblasDgemv (
		order, transA,M, N,alpha,
		A.get_buffer().get(), offA, lda,
		x.get_buffer().get(), offx, (int)incx,
		beta,
		y.get_buffer().get(), offy, (int)incy,
		numCommandQueues, commandQueues,
		numEventsInWaitList, eventWaitList, events
	);
}

}

namespace kernels{

// y <- alpha * op (A) * x + beta * y
// op (A) == A || A^T || A^H
template <typename MatA, typename VectorX, typename VectorY>
void gemv(
	matrix_expression<MatA, gpu_tag> const& A,
	vector_expression<VectorX, gpu_tag> const& x,
        vector_expression<VectorY, gpu_tag>& y,
	typename VectorY::value_type const& alpha
){
	std::size_t m = A().size1();
	std::size_t n = A().size2();
	
	SIZE_CHECK(x().size() == A().size2());
	SIZE_CHECK(y().size() == A().size1());

	clblasOrder const stor_ord= (clblasOrder)clblas::storage_order<typename MatA::orientation>::value;
	
	auto storageA = A().raw_storage();
	auto storagex = x().raw_storage();
	auto storagey = y().raw_storage();
	bindings::gemv(stor_ord, clblasNoTrans, m, n, alpha.get(),
		storageA.buffer, storageA.offset, storageA.leading_dimension,
		storagex.buffer, storagex.offset, storagex.stride,
		typename VectorY::value_type(1),
		storagey.buffer, storagey.offset, storagey.stride,
		1, &(y().queue()),
		0, nullptr, nullptr
	);
}


}}}
#endif
