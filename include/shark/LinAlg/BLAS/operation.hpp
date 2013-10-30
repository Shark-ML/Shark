#ifndef SHARK_LINALG_BLAS_UBLAS_OPERATION_HPP
#define SHARK_LINALG_BLAS_UBLAS_OPERATION_HPP

#include "kernels/gemv.hpp"
#include "kernels/gemm.hpp"

namespace shark {
namespace blas {

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
	if (init)
		result().clear();

	kernels::gemv(matrix, vector, result,alpha);
}

//dispatch for temporary result proxies
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

//dispatch for vector-matrix product
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


}
}

#endif
