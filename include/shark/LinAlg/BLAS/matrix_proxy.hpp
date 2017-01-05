/*!
 * 
 *
 * \brief       Matrix proxy expressions
 * 
 * 
 *
 * \author      O. Krause
 * \date        2016
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

#ifndef SHARK_LINALG_BLAS_MATRIX_PROXY_HPP
#define SHARK_LINALG_BLAS_MATRIX_PROXY_HPP

#include "detail/expression_optimizers.hpp"

namespace shark {
namespace blas {
	
	
////////////////////////////////////
//// Matrix Transpose
////////////////////////////////////

/// \brief Returns a proxy which transposes the matrix
///
/// given a matrix 
/// A = (1 2 3)
///     (4 5 6)
///     (7 8 9)
///
/// the trans(A) operation results in
/// trans(A) = (1 4 7)
///            (2 5 8)
///            (3 6 9)
template<class M, class Device>
temporary_proxy<typename detail::matrix_transpose_optimizer<M>::type>
trans(matrix_expression<M, Device> & m){
	return detail::matrix_transpose_optimizer<M>::create(m());
}
template<class M, class Device>
typename detail::matrix_transpose_optimizer<typename const_expression<M>::type >::type
trans(matrix_expression<M, Device> const& m){
	typedef typename const_expression<M>::type closure_type;
	return detail::matrix_transpose_optimizer<closure_type>::create(m());
}

template<class M>
temporary_proxy<typename detail::matrix_transpose_optimizer<M>::type>
trans(temporary_proxy<M> m){
	return trans(static_cast<M&>(m));
}

////////////////////////////////////
//// Matrix Row and Column
////////////////////////////////////

/// \brief Returns a vector-proxy representing the i-th row of the Matrix
///
/// given a matrix 
/// A = (1 2 3)
///     (4 5 6)
///     (7 8 9)
///
/// the row(A,1) operation results in
/// row(A,1) = (4,5,6)
template<class M, class Device>
temporary_proxy<typename detail::matrix_row_optimizer<M>::type>
row(matrix_expression<M, Device>& expression, typename M::index_type i){
	return detail::matrix_row_optimizer<M>::create(expression(), i);
}
template<class M, class Device>
typename detail::matrix_row_optimizer<typename const_expression<M>::type>::type
row(matrix_expression<M, Device> const& expression, typename M::index_type i){
	return detail::matrix_row_optimizer<typename const_expression<M>::type>::create(expression(), i);
}

template<class M>
temporary_proxy<typename detail::matrix_row_optimizer<M>::type>
row(temporary_proxy<M> expression, typename M::index_type i){
	return row(static_cast<M&>(expression), i);
}

/// \brief Returns a vector-proxy representing the j-th column of the Matrix
///
/// given a matrix 
/// A = (1 2 3)
///     (4 5 6)
///     (7 8 9)
///
/// the column(A,1) operation results in
/// column(A,1) = (2,5,8)
template<class M, class Device>
auto column(matrix_expression<M, Device>& expression, typename M::index_type j) -> decltype(row(trans(expression),j)){
	return row(trans(expression),j);
}
template<class M, class Device>
auto column(matrix_expression<M, Device> const& expression, typename M::index_type j) -> decltype(row(trans(expression),j)){
	return row(trans(expression),j);
}

template<class M, class Device>
auto column(temporary_proxy<M> expression, typename M::index_type j) -> decltype(row(trans(expression),j)){
	return row(trans(static_cast<M&>(expression)),j);
}

////////////////////////////////////
//// Matrix Diagonal
////////////////////////////////////

///\brief Returns the diagonal of a constant square matrix as vector.
///
/// given a matrix 
/// A = (1 2 3)
///     (4 5 6)
///     (7 8 9)
///
/// the diag operation results in
/// diag(A) = (1,5,9)
template<class M, class Device>
matrix_vector_range<typename const_expression<M>::type > diag(matrix_expression<M, Device> const& mat){
	SIZE_CHECK(mat().size1() == mat().size2());
	matrix_vector_range<typename const_expression<M>::type > diagonal(mat(),0,mat().size1(),0,mat().size1());
	return diagonal;
}

template<class M, class Device>
temporary_proxy< matrix_vector_range<M> > diag(matrix_expression<M, Device>& mat){
	SIZE_CHECK(mat().size1() == mat().size2());
	matrix_vector_range<M> diagonal(mat(),0,mat().size1(),0,mat().size1());
	return diagonal;
}

template<class M>
temporary_proxy< matrix_vector_range<M> > diag(temporary_proxy<M> mat){
	return diag(static_cast<M&>(mat));
}


////////////////////////////////////
//// Matrix Subranges
////////////////////////////////////


///\brief Returns a submatrix of a given matrix.
///
/// given a matrix 
/// A = (1 2 3)
///     (4 5 6)
///     (7 8 9)
///
/// the subrange(A,0,2,1,3) operation results in
/// subrange(A,0,2,1,3) = (4 5)
///                       (7 8)
template<class M, class Device>
temporary_proxy< typename detail::matrix_range_optimizer<M>::type > subrange(
	matrix_expression<M, Device>& expression, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
){
	RANGE_CHECK(start1 <= stop1);
	RANGE_CHECK(start2 <= stop2);
	SIZE_CHECK(stop1 <= expression().size1());
	SIZE_CHECK(stop2 <= expression().size2());
	return detail::matrix_range_optimizer<M>::create(expression(), start1, stop1, start2, stop2);
}
template<class M, class Device>
typename detail::matrix_range_optimizer<typename const_expression<M>::type>::type subrange(
	matrix_expression<M, Device> const& expression, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
){
	RANGE_CHECK(start1 <= stop1);
	RANGE_CHECK(start2 <= stop2);
	SIZE_CHECK(stop1 <= expression().size1());
	SIZE_CHECK(stop2 <= expression().size2());
	return detail::matrix_range_optimizer<typename const_expression<M>::type>::create(expression(), start1, stop1, start2, stop2);
}

template<class M, class Device>
auto subrange(
	temporary_proxy<M> expression, 
	std::size_t start1, std::size_t stop1,
	std::size_t start2, std::size_t stop2
) -> decltype(subrange(static_cast<M&>(expression),start1,stop1,start2,stop2)){
	return subrange(static_cast<M&>(expression),start1,stop1,start2,stop2);
}

template<class M, class Device>
auto rows(
	matrix_expression<M, Device>& expression, 
	std::size_t start, std::size_t stop
) -> decltype(subrange(expression, start, stop, 0,expression().size2())){
	RANGE_CHECK(start <= stop);
	SIZE_CHECK(stop <= expression().size1());
	return subrange(expression, start, stop, 0,expression().size2());
}

template<class M, class Device>
auto rows(
	matrix_expression<M, Device> const& expression, 
	std::size_t start, std::size_t stop
) -> decltype(subrange(expression, start, stop, 0,expression().size2())){
	RANGE_CHECK(start <= stop);
	SIZE_CHECK(stop <= expression().size1());
	return subrange(expression, start, stop, 0,expression().size2());
}

template<class M>
auto rows(
	temporary_proxy<M> expression, 
	std::size_t start, std::size_t stop
) -> decltype( rows(static_cast<M&>(expression),start,stop)){
	return rows(static_cast<M&>(expression),start,stop);
}

template<class M, class Device>
auto columns(
	matrix_expression<M, Device>& expression, 
	typename M::index_type start, typename M::index_type stop
) -> decltype(subrange(expression, 0,expression().size1(), start, stop)){
	RANGE_CHECK(start <= stop);
	SIZE_CHECK(stop <= expression().size2());
	return subrange(expression, 0,expression().size1(), start, stop);
}

template<class M, class Device>
auto columns(
	matrix_expression<M, Device> const& expression, 
	typename M::index_type start, typename M::index_type stop
) -> decltype(subrange(expression, 0,expression().size1(), start, stop)){
	RANGE_CHECK(start <= stop);
	SIZE_CHECK(stop <= expression().size2());
	return subrange(expression, 0,expression().size1(), start, stop);
}

template<class M>
auto columns(
	temporary_proxy<M> expression, 
	std::size_t start, std::size_t stop
) -> decltype(columns(static_cast<M&>(expression),start,stop)){
	return columns(static_cast<M&>(expression),start,stop);
}

////////////////////////////////////
//// Matrix Adaptor
////////////////////////////////////

/// \brief Converts a chunk of memory into a matrix of given size.
template <class T>
temporary_proxy< dense_matrix_adaptor<T> > adapt_matrix(std::size_t size1, std::size_t size2, T* data){
	return dense_matrix_adaptor<T>(data,size1, size2);
}

/// \brief Converts a 2D C-style array into a matrix of given size.
template <class T, std::size_t M, std::size_t N>
temporary_proxy<dense_matrix_adaptor<T> > adapt_matrix(T (&array)[M][N]){
	return dense_matrix_adaptor<T>(&(array[0][0]),M,N);
}

/// \brief Converts a dense vector to a matrix of a given size
template <class V>
typename boost::enable_if<
	boost::is_same<typename V::storage_type::storage_tag,dense_tag>,
	temporary_proxy< dense_matrix_adaptor<
		typename boost::remove_reference<typename V::reference>::type
	> >
>::type
to_matrix(
	vector_expression<V, cpu_tag>& v,
	std::size_t size1, std::size_t size2
){
	typedef typename boost::remove_reference<typename V::reference>::type ElementType;
	return dense_matrix_adaptor<ElementType>(v().raw_storage().values, size1, size2);
}

/// \brief Converts a dense vector to a matrix of a given size
template <class V>
typename boost::enable_if<
	boost::is_same<typename V::storage_type::storage_tag,dense_tag>,
	temporary_proxy< dense_matrix_adaptor<typename V::value_type const> >
>::type 
to_matrix(
	vector_expression<V, cpu_tag> const& v,
	std::size_t size1, std::size_t size2
){
	return dense_matrix_adaptor<typename V::value_type const>(v().raw_storage().values, size1, size2);
}

template <class E>
typename boost::enable_if<
	boost::is_same<typename E::storage_type::storage_tag,dense_tag>,
	temporary_proxy< dense_matrix_adaptor<
		typename boost::remove_reference<typename E::reference>::type
	> >
>::type 
to_matrix(
	temporary_proxy<E> v,
	std::size_t size1, std::size_t size2
){
	return to_matrix(static_cast<E&>(v),size1,size2);
}

}}
#endif
