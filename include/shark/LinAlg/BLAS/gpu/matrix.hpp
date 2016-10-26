/*!
 * \brief       Implements the dense matrix class for the gpu
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
#ifndef SHARK_LINALG_BLAS_GPU_MATRIX_HPP
#define SHARK_LINALG_BLAS_GPU_MATRIX_HPP

#include "traits.hpp"
//~ #include "scalar.hpp"
#include "../assignment.hpp"
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/strided_iterator.hpp>

namespace shark {namespace blas { namespace gpu{
	
namespace detail{
template<class Arg1, class Arg2, class T>
struct induced_matrix_element{
	typedef T result_type;
	Arg1 arg1;
	Arg2 arg2;
	std::size_t leading_dimension;
	boost::compute::buffer const& buffer;
};

template<class Arg1, class Arg2,class T>
boost::compute::detail::meta_kernel& operator<< (
	boost::compute::detail::meta_kernel& k, 
	induced_matrix_element<Arg1, Arg2, T> const& e
){
	return k<< k.get_buffer_identifier<T>(e.buffer, boost::compute::memory_object::global_memory)
		<<'['<<e.arg1 <<'*'<<e.leading_dimension<<'+'<< e.arg2<<']';
}
}

/// \brief A dense matrix of values of type \c T stored on the gpu
///
/// For a \f$(m \times n)\f$-dimensional matrix and \f$ 0 \leq i < m, 0 \leq j < n\f$, every element \f$ m_{i,j} \f$ is mapped to
/// the \f$(i.n + j)\f$-th element of the container for row major orientation or the \f$ (i + j.m) \f$-th element of
/// the container for column major orientation. In a dense matrix all elements are represented in memory in a
/// contiguous chunk of memory by definition.
///
/// Orientation can also be specified, otherwise a \c row_major is used.
///
/// \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
/// \tparam L the storage organization. It can be either \c row_major or \c column_major. Default is \c row_major
template<class T, class L = blas::row_major>
class matrix: public matrix_container<matrix<T,L>, gpu_tag > {
private:
	template<class IndexExpr1, class IndexExpr2>
	detail::induced_matrix_element<IndexExpr1, IndexExpr2, T> get_element(
		IndexExpr1 const& expr1,IndexExpr2 const& expr2,
		row_major
	)const{
		return {expr1, expr2,orientation::index_m(m_size1,m_size2), m_storage.get_buffer()};
	}
	template<class IndexExpr1, class IndexExpr2>
	detail::induced_matrix_element<IndexExpr2, IndexExpr1,T> get_element(
		IndexExpr1 const& expr1,IndexExpr2 const& expr2,
		column_major
	)const{
		return {expr2, expr1,orientation::index_m(m_size1,m_size2), m_storage.get_buffer()};
	}
public:
	typedef T value_type;
	//~ typedef scalar<T> value_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef std::size_t size_type;

	typedef matrix_reference<matrix const> const_closure_type;
	typedef matrix_reference<matrix> closure_type;
	typedef gpu::dense_matrix_storage<T> storage_type;
	typedef gpu::dense_matrix_storage<T> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;
	typedef L orientation;

	// Construction and destruction

	/// \brief Constructor of a matrix with a default queue
	///
	///note that for all operations for which matrix is on the left hand side,
	///the kernels are enqueued on the supplied queue in case of a multi-queue setup.
	matrix(boost::compute::command_queue& queue = boost::compute::system::default_queue())
	: m_storage(queue.get_context())
	, m_queue(&queue),m_size1(0), m_size2(0){}

	/// \brief Constructor of a matrix with a predefined size
	/// By default, its elements are uninitialized
	/// \param size1 number of rows
	/// \param size2 number of columns
	/// \param queue the opencl queue to use by this matrix
	explicit matrix(size_type size1, size_type size2, boost::compute::command_queue& queue = boost::compute::system::default_queue())
	: m_storage(size1 * size2, queue.get_context())
	, m_queue(&queue)
	, m_size1(size1)
	, m_size2(size2){}

	/// \brief Constructor of a matrix with a predefined size initialized to a value
	/// \param size1 number of rows
	/// \param size2 number of columns
	/// \param init value to assign to each element of the matrix
	/// \param queue the opencl queue to use by this matrix
	matrix(size_type size1, size_type size2, value_type const& init, boost::compute::command_queue& queue = boost::compute::system::default_queue())
	: m_storage(size1 * size2, init, queue)
	, m_queue(&queue)
	, m_size1(size1)
	, m_size2(size2){}
	
	/// \brief Move-constructor of a matrix
	/// \param m is the matrix to be moved
	matrix(matrix && m)
	: m_storage(std::move(m.m_storage))
	, m_queue(&m.queue())
	, m_size1(m.size1())
	, m_size2(m.size2()){}

	/// \brief Copy-constructor of a matrix
	/// \param m is the matrix to be duplicated
	matrix(matrix const& m)
	: m_storage(m.m_storage)
	, m_queue(&m.queue())
	, m_size1(m.size1())
	, m_size2(m.size2()){}

	/// \brief Copy-constructor of a matrix from a matrix_expression
	/// \param e the matrix_expression whose values will be duplicated into the matrix
	template<class E>
	matrix(matrix_expression<E, gpu_tag> const& e)
	: m_storage(e().size1() * e().size2(), e().queue().get_context())
	, m_queue(&e().queue())
	, m_size1(e().size1())
	, m_size2(e().size2()){
		assign(*this, e);
	}
	
	// -------------------
	// Assignment operators
	// -------------------
	
	/// \brief Assign a full matrix (\e RHS-matrix) to the current matrix (\e LHS-matrix)
	/// Assign a full matrix (\e RHS-matrix) to the current matrix (\e LHS-matrix). This method does not create any temporary.
	/// \param m is the source matrix container
	/// \return a reference to a matrix (i.e. the destination matrix)
	matrix& operator = (matrix const& m){
		resize(m.size1(),m.size2());
		return assign(*this, m);
	}
	
	/// \brief Move-Assign a full matrix (\e RHS-matrix) to the current matrix (\e LHS-matrix)
	/// \param m is the source matrix container
	/// \return a reference to a matrix (i.e. the destination matrix)
	matrix& operator = (matrix && m){
		m_storage = std::move(m.m_storage);
		m_queue = m.m_queue;
		m_size1 = m.m_size1;
		m_size2 = m.m_size2;
		return *this;
	}
	
	/// \brief Assign a full matrix (\e RHS-matrix) to the current matrix (\e LHS-matrix)
	/// Assign a full matrix (\e RHS-matrix) to the current matrix (\e LHS-matrix). This method does not create any temporary.
	/// \param m is the source matrix container
	/// \return a reference to a matrix (i.e. the destination matrix)
	template<class C>          // Container assignment without temporary
	matrix& operator = (matrix_container<C, gpu_tag> const& m) {
		resize(m().size1(), m().size2());
		return assign(*this, m);
	}

	/// \brief Assign the result of a matrix_expression to the matrix
	///
	/// \param e is a const reference to the matrix_expression
	/// \return a reference to the resulting matrix
	template<class E>
	matrix& operator = (matrix_expression<E, gpu_tag> const& e) {
		matrix temporary(e);
		swap(*this,temporary);
		return *this;
	}

	// ---------
	// Storage interface
	// ---------
	
	///\brief Returns the number of rows of the matrix.
	size_type size1() const {
		return m_size1;
	}
	///\brief Returns the number of columns of the matrix.
	size_type size2() const {
		return m_size2;
	}
	
	boost::compute::command_queue& queue() const{
		return *m_queue;
	}
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return {m_storage,0,orientation::index_m(m_size1,m_size2)};
	}
	
	// Element access
	template <class IndexExpr1, class IndexExpr2>
	detail::induced_matrix_element<IndexExpr1,IndexExpr2,T> operator()(IndexExpr1 const& i, IndexExpr2 const& j) const{
		return this->get_element(i,j,orientation());
	}
	
	
	/// \brief Resize the matrix
	///
	/// This might erase all data stored in the matrix
	///
	/// \param size1 new number of rows
	/// \param size2 new number of columns
	void resize(size_type size1, size_type size2) {
		if(size1 * size2 < m_storage.size())
			m_storage.resize(size1 * size2);
		else
			m_storage = boost::compute::vector<T>(size1 * size2, queue().get_context());
		m_size1 = size1;
		m_size2 = size2;
	}

	
	/// \brief Resize the matrix
	///
	/// This will erase all data stored in the matrix and reinitialize it with the supplied value of init
	///
	/// \param size1 new number of rows
	/// \param size2 new number of columns
	/// \param init the value of all elements
	void resize(size_type size1, size_type size2, value_type init) {
		resize(size1,size2);
		boost::compute::fill(m_storage.begin(),m_storage.end(), init, queue());
	}
	
	void clear(){
		boost::compute::fill(m_storage.begin(),m_storage.end(), value_type/*zero*/(), queue());
	}
	
	// Iterator types
	typedef boost::compute::strided_iterator<typename boost::compute::vector<T>::iterator > row_iterator;
	typedef boost::compute::strided_iterator<typename boost::compute::vector<T>::iterator > column_iterator;
	typedef boost::compute::strided_iterator<typename boost::compute::vector<T>::const_iterator > const_row_iterator;
	typedef boost::compute::strided_iterator<typename boost::compute::vector<T>::const_iterator > const_column_iterator;

	const_row_iterator row_begin(size_type i) const {
		return {m_storage.begin() + i * stride1(), stride2()};
	}
	const_row_iterator row_end(size_type i) const {
		return {m_storage.begin() + i * stride1()+size2()*stride2(), stride2()};
	}
	
	const_row_iterator column_begin(size_type j) const {
		return {m_storage.begin() + j * stride2(), stride1()};
	}
	const_column_iterator column_end(size_type j) const {
		return {m_storage.begin() + j * stride2()+size1()*stride1(), stride1()};
	}
	
	row_iterator row_begin(size_type i){
		return {m_storage.begin() + i * stride1(), stride2()};
	}
	row_iterator row_end(size_type i){
		return {m_storage.begin() + i * stride1()+size2()*stride2(), stride2()};
	}
	
	row_iterator column_begin(size_type j){
		return {m_storage.begin() + j * stride2(), stride1()};
	}
	column_iterator column_end(size_type j){
		return {m_storage.begin() + j * stride2()+size1()*stride1(), stride1()};
	}

	
	/// \brief Swap the content of two matrixs
	/// \param m1 is the first matrix. It takes values from m2
	/// \param m2 is the second matrix It takes values from m1
	friend void swap(matrix& m1, matrix& m2) {
		using std::swap;
		swap(m1.m_storage,m2.m_storage);
		std::swap(m1.m_queue,m2.m_queue);
		std::swap(m1.m_size1,m2.m_size1);
		std::swap(m1.m_size2,m2.m_size2);
	}
	
	
private:
	std::ptrdiff_t stride1() const {
		return (std::ptrdiff_t) orientation::stride1(m_size1, m_size2);
	}
	std::ptrdiff_t stride2() const {
		return (std::ptrdiff_t) orientation::stride2(m_size1, m_size2);
	}
	
	boost::compute::vector<T> m_storage;
	boost::compute::command_queue* m_queue;
	size_type m_size1;
	size_type m_size2;
};
}

template<class T, class L>
struct matrix_temporary_type<T,L,dense_tag, gpu_tag>{
	typedef gpu::matrix<T, L> type;
};

template<class T>
struct matrix_temporary_type<T,unknown_orientation,dense_tag, gpu_tag>{
	typedef gpu::matrix<T, row_major> type;
};
}}

#endif
