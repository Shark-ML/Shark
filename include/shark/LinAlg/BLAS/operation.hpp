/*!
 * \brief       Some special matrix-products
 * 
 * \author      O. Krause
 * \date        2013
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
#ifndef SHARK_LINALG_BLAS_OPERATION_HPP
#define SHARK_LINALG_BLAS_OPERATION_HPP

#include "kernels/gemv.hpp"
#include "kernels/gemm.hpp"
#include "kernels/tpmv.hpp"
#include "kernels/trmv.hpp"
#include "kernels/trmm.hpp"

namespace shark {
namespace blas {
	
namespace detail{
	
///\brief Computes y=alpha*Ax or y += alpha*Ax
template<class ResultV, class M, class V>
void axpy_prod_impl(
	matrix_expression<M> const& matrix,
        vector_expression<V> const& vector,
        vector_expression<ResultV>& result,
	bool init,
	typename ResultV::value_type alpha,
	linear_structure
) {
	
	if (init)
		result().clear();
	
	kernels::gemv(matrix, vector, result,alpha);
}
///\brief Computes y=alpha*Ax or y += alpha*Ax
template<class ResultV, class M, class V>
void axpy_prod_impl(
	matrix_expression<M> const& matrix,
        vector_expression<V> const& vector,
        vector_expression<ResultV>& result,
	bool init,
	typename ResultV::value_type alpha,
	packed_structure
) {
	if(init){
		noalias(result) = vector;
		kernels::tpmv(matrix, result);
		result() *= alpha;
	}else{
		typename vector_temporary<V>::type temp(result);
		noalias(result) = vector;
		kernels::tpmv(matrix, result);
		result() *= alpha;
		noalias(result) += temp;
	}
}

}

	
///\brief Computes y=alpha*Ax or y += alpha*Ax
template<class ResultV, class M, class V>
void axpy_prod(
	matrix_expression<M> const& matrix,
        vector_expression<V> const& vector,
        vector_expression<ResultV>& result, 
	bool init = true,
	typename ResultV::value_type alpha = 1.0
) {
	SIZE_CHECK(matrix().size1()==result().size());
	SIZE_CHECK(matrix().size2()==vector().size());
	

	detail::axpy_prod_impl(matrix, vector, result,init, alpha,typename M::orientation());
}

////\brief Computes C=alpha*Ax or C += alpha*Ax
///
///This the dispatcher for temporary result proxies
template<class ResultV, class M, class V>
void axpy_prod(
	matrix_expression<M> const& matrix,
        vector_expression<V> const& vector,
        temporary_proxy<ResultV> result, 
	bool init = true,
	typename ResultV::value_type alpha = 1.0
) {
	SIZE_CHECK(matrix().size1()==result.size());
	SIZE_CHECK(matrix().size2()==vector().size());
	axpy_prod(matrix,vector,static_cast<ResultV&>(result),init,alpha);
}

///\brief Computes y=alpha*xA or y += alpha*xA
template<class ResultV, class V, class M>
void axpy_prod(
	vector_expression<V> const& vector,
	matrix_expression<M> const& matrix,
        vector_expression<ResultV>& result,
	bool init = true,
	typename ResultV::value_type alpha = 1.0
) {
	SIZE_CHECK(matrix().size2()==result().size());
	SIZE_CHECK(matrix().size1()==vector().size());
	axpy_prod(trans(matrix), vector, result,init,alpha);
}

////\brief Computes C=alpha*xA or C += alpha*xA
///
///This the dispatcher for temporary result proxies
template<class ResultV, class M, class V>
void axpy_prod(
	vector_expression<V> const& vector,
	matrix_expression<M> const& matrix,
        temporary_proxy<ResultV> result, 
	bool init = true,
	typename ResultV::value_type alpha = 1.0
) {
	SIZE_CHECK(matrix().size2()==result.size());
	SIZE_CHECK(matrix().size1()==vector().size());
	axpy_prod(trans(matrix), vector, static_cast<ResultV&>(result),init,alpha);
}

/// \brief Implements the matrix products m+=alpha * e1*e2 or m = alpha*e1*e2.
template<class M, class E1, class E2>
void axpy_prod(
	matrix_expression<E1> const& e1,
        matrix_expression<E2> const& e2,
        matrix_expression<M>& m,
        bool init = true,
	typename M::value_type alpha = 1.0
) {
	SIZE_CHECK(m().size1() == e1().size1());
	SIZE_CHECK(m().size2() == e2().size2());
	SIZE_CHECK(e1().size2() == e2().size1());
	
	if (init)
		m().clear();
	
	kernels::gemm(e1,e2,m,alpha);
}

template<class M, class E1, class E2>
void axpy_prod(
	matrix_expression<E1> const& e1,
        matrix_expression<E2> const& e2,
        temporary_proxy<M> m,
        bool init = true,
	typename M::value_type alpha = 1.0
) {
	axpy_prod(e1,e2,static_cast<M&>(m),init,alpha);
}

/// \brief computes C= alpha*AA^T or C+=alpha* AA^T
template<class M, class E>
void symm_prod(
	matrix_expression<E> const& A,
        matrix_expression<M>& m,
        bool init = true,
	typename M::value_type alpha = 1.0
) {
	SIZE_CHECK(m().size1() == A().size1());
	SIZE_CHECK(m().size2() == m().size1());
	
	axpy_prod(A, trans(A), m,init, alpha);
}

/// \brief computes C= alpha*AA^T or C+=alpha* AA^T
template<class M, class E>
void symm_prod(
	matrix_expression<E> const& A,
	temporary_proxy<M>& m,
        bool init = 1.0,
	typename M::value_type alpha = 1.0
) {
	symm_prod(A, static_cast<M&>(m),init, alpha);
}

/// \brief Computes x=Ax for a triangular matrix A
///
/// The first template argument governs the type
/// of triangular matrix: lower, upper, unit_lower and unit_upper.
///
///Example: triangular_prod<lower>(A,x);
template<class TriangularType, class MatrixA, class V>
void triangular_prod(
	matrix_expression<MatrixA> const& A,
	vector_expression<V>& x
) {
	kernels::trmv<TriangularType::is_upper, TriangularType::is_unit>(A, x);
}

/// \brief Computes B=AB for a triangular matrix A and a dense matrix B in place
///
/// The first template argument governs the type
/// of triangular matrix: lower, upper, unit_lower and unit_upper.
///
///Example: triangular_prod<lower>(A,B);
template<class TriangularType, class MatrixA, class MatB>
void triangular_prod(
	matrix_expression<MatrixA> const& A,
	matrix_expression<MatB>& B
) {
	kernels::trmm<TriangularType::is_upper, TriangularType::is_unit>(A, B);
}

/// \brief triangular prod for temporary left-hand side arguments
///
/// Dispatches to the other versions of triangular_prod, see their documentation
template<class TriangularType, class MatrixA, class E>
void triangular_prod(
	matrix_expression<MatrixA> const& A,
	temporary_proxy<E> e
) {
	triangular_prod<TriangularType>(A, static_cast<E&>(e));
}


}
}

#endif
