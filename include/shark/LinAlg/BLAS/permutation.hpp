#ifndef SHARK_LINALG_BLAS_UBLAS_PERMUTATION_HPP
#define SHARK_LINALG_BLAS_UBLAS_PERMUTATION_HPP

#include "vector.hpp"

namespace shark { namespace blas{
struct permutation_matrix:public vector<std::size_t> {
	// Construction and destruction
	explicit permutation_matrix(size_type size):vector<std::size_t> (size){
		for (size_type i = 0; i < size; ++ i)
			(*this)(i) = i;
	}

	explicit permutation_matrix(vector<std::size_t> const& init): vector<std::size_t>(init){ }

	// Assignment
	permutation_matrix &operator = (permutation_matrix const& m) {
		vector<std::size_t>::operator = (m);
		return *this;
	}
};

///\brief implements row pivoting at matrix A using permutation P
///
///by convention it is not allowed that P(i) < i. 
template<class M>
void swap_rows(permutation_matrix const& P, matrix_expression<M>& A){
	for (std::size_t i = 0; i != P.size(); ++ i)
		swap_rows(A(),i,P(i));
}

///\brief implements column pivoting of vector A using permutation P
///
///by convention it is not allowed that P(i) < i. 
template<class V>
void swap_rows(permutation_matrix const& P, vector_expression<V>& v){
	for (std::size_t i = 0; i != P.size(); ++ i)
		std::swap(v()(i),v()(P(i)));
}

///\brief implements the inverse row pivoting of vector v using permutation P
///
///This is the inverse operation to swap_rows. 
template<class V, class Permutation>
void swap_rows_inverted(Permutation const& P, vector_expression<V>& v){
	for(std::size_t i = P.size(); i != 0; --i){
		std::size_t k = i-1;
		if(k != P(k)){
			using std::swap;
			swap(v()(k),v()(P(k)));
		}
	}
}

///\brief implements column pivoting at matrix A using permutation P
///
///by convention it is not allowed that P(i) < i. 
template<class M>
void swap_columns(permutation_matrix const& P, matrix_expression<M>& A){
	for(std::size_t i = 0; i != P.size(); ++i)
		swap_columns(A(),i,P(i));
}

///\brief implements the inverse row pivoting at matrix A using permutation P
///
///This is the inverse operation to swapRows. 
template<class M>
void swap_rows_inverted(permutation_matrix const& P, matrix_expression<M>& A){
	for(std::size_t i = P.size(); i != 0; --i){
		swap_rows(A(),i-1,P(i-1));
	}
}

///\brief implements the inverse column pivoting at matrix A using permutation P
///
///This is the inverse operation to swapColumns. 
template<class M>
void swap_columns_inverted(permutation_matrix const& P, matrix_expression<M>& A){
	for(std::size_t i = P.size(); i != 0; --i){
		swap_columns(A(),i-1,P(i-1));
	}
}

///\brief Implements full pivoting at matrix A using permutation P
///
///full pivoting does swap rows and columns such that the diagonal element
///A_ii is then at position A_P(i)P(i)
///by convention it is not allowed that P(i) < i. 
template<class M>
void swap_full(permutation_matrix const& P, matrix_expression<M>& A){
	for(std::size_t i = 0; i != P.size(); ++i){
		swap_rows(A(),i,P(i));
		swap_columns(A(),i,P(i));
	}
}
///\brief implements the inverse full pivoting at matrix A using permutation P
///
///This is the inverse operation to swap_full. 
template<class M>
void swap_full_inverted(permutation_matrix const& P, matrix_expression<M>& A){
	for(std::size_t i = P.size(); i != 0; --i){
		swap_rows(A(),i-1,P(i-1));
		swap_columns(A(),i-1,P(i-1));
	}
}

}}
#endif
