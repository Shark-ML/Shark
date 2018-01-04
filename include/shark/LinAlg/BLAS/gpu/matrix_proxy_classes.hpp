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
#ifndef REMORA_GPU_MATRIX_PROXY_CLASSES_HPP
#define REMORA_GPU_MATRIX_PROXY_CLASSES_HPP

#include "../expression_types.hpp"
#include "../detail/traits.hpp"
#include <boost/compute/iterator/strided_iterator.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

namespace remora{
	
namespace detail{
template<class Arg1, class Arg2, class T>
struct induced_matrix_adaptor_element{
	typedef T result_type;
	Arg1 arg1;
	Arg2 arg2;
	gpu::dense_matrix_storage<T, dense_tag> storage;
};

template<class Arg1, class Arg2,class T>
boost::compute::detail::meta_kernel& operator<< (
	boost::compute::detail::meta_kernel& k, 
	induced_matrix_adaptor_element<Arg1, Arg2, T> const& e
){
	return k<< k.get_buffer_identifier<T>(e.storage.buffer, boost::compute::memory_object::global_memory)
		<<" [ "<<e.storage.offset <<"+("<<e.arg1 <<")*"<<e.storage.leading_dimension<<"+"<< "("<<e.arg2<<")"<<']';
}
}

template<class T,class Orientation>
class dense_matrix_adaptor<T, Orientation, gpu_tag>: public matrix_expression<dense_matrix_adaptor<T,Orientation, gpu_tag>, gpu_tag > {
	private:
	template<class IndexExpr1, class IndexExpr2>
	detail::induced_matrix_adaptor_element<IndexExpr1, IndexExpr2, T> get_element(
		IndexExpr1 const& expr1,IndexExpr2 const& expr2,
		row_major
	)const{
		return {expr1, expr2,m_storage};
	}
	template<class IndexExpr1, class IndexExpr2>
	detail::induced_matrix_adaptor_element<IndexExpr2, IndexExpr1,T> get_element(
		IndexExpr1 const& expr1,IndexExpr2 const& expr2,
		column_major
	)const{
		return {expr2, expr1,m_storage};
	}
public:
	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type const& const_reference;
	typedef T& reference;

	typedef matrix_reference<dense_matrix_adaptor> closure_type;
	typedef matrix_reference<dense_matrix_adaptor const> const_closure_type;
	typedef gpu::dense_matrix_storage<T, dense_tag> storage_type;
	typedef gpu::dense_matrix_storage<value_type const, dense_tag> const_storage_type;
        typedef Orientation orientation;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction
	
	dense_matrix_adaptor(dense_matrix_adaptor<value_type, Orientation> const& expression)
	: m_storage(expression.m_storage)
	, m_queue(expression.m_queue)
	, m_size1(expression.size1())
	, m_size2(expression.size2())
	{}
		
	/// \brief Constructor of a vector proxy from a block of memory
	/// \param values the block of memory used
	/// \param size1 size in 1st direction
	/// \param size2 size in 2nd direction
 	/// \param stride1 distance in 1st direction between elements of the dense_matrix_adaptor in memory
 	/// \param stride2 distance in 2nd direction between elements of the dense_matrix_adaptor in memory
	dense_matrix_adaptor(
		storage_type storage, 
		boost::compute::command_queue& queue,
		size_type size1, size_type size2
	)
	: m_storage(storage)
	, m_queue(&queue)
	, m_size1(size1)
	, m_size2(size2){}
	
	template<class E>
	dense_matrix_adaptor(vector_expression<E, gpu_tag> const& expression, std::size_t size1, std::size_t size2)
	: m_queue(&expression().queue())
	, m_size1(size1)
	, m_size2(size2){
		auto storage = expression().raw_storage();
		m_storage.buffer = storage.buffer;
		m_storage.offset = storage.offset;
		m_storage.leading_dimension = orientation::index_m(size1, size2);
		REMORA_RANGE_CHECK(storage.stride == 1);
	}
	
	template<class E>
	dense_matrix_adaptor(vector_expression<E, gpu_tag>& expression, std::size_t size1, std::size_t size2)
	: m_queue(&expression().queue())
	, m_size1(size1)
	, m_size2(size2){
		auto storage = expression().raw_storage();
		m_storage.buffer = storage.buffer;
		m_storage.offset = storage.offset;
		m_storage.leading_dimension = orientation::index_m(size1, size2);
		REMORA_RANGE_CHECK(storage.stride == 1);
	}

	
	// -------------------
	// Assignment operators
	// -------------------
	
	dense_matrix_adaptor& operator = (dense_matrix_adaptor const& e) {
		REMORA_SIZE_CHECK(size1() == e().size1());
		REMORA_SIZE_CHECK(size2() == e().size2());
		return assign(*this, typename matrix_temporary<dense_matrix_adaptor>::type(e));
	}
	template<class E>
	dense_matrix_adaptor& operator = (matrix_expression<E, gpu_tag> const& e) {
		REMORA_SIZE_CHECK(size1() == e().size1());
		REMORA_SIZE_CHECK(size2() == e().size2());
		return assign(*this, typename matrix_temporary<dense_matrix_adaptor>::type(e));
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
	storage_type raw_storage(){
		return {m_storage.buffer, m_storage.offset, m_storage.leading_dimension};
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return {m_storage.buffer, m_storage.offset, m_storage.leading_dimension};
	}
	
	// Element access
	template <class IndexExpr1, class IndexExpr2>
	auto operator()(IndexExpr1 const& i, IndexExpr2 const& j) const -> decltype(std::declval<dense_matrix_adaptor const&>().get_element(i,j,orientation())){
		return this->get_element(i,j,orientation());
	}
	
	// Iterator types
	typedef boost::compute::strided_iterator<boost::compute::buffer_iterator<T> > row_iterator;
	typedef boost::compute::strided_iterator<boost::compute::buffer_iterator<T> > column_iterator;
	typedef boost::compute::strided_iterator<boost::compute::buffer_iterator<T const> > const_row_iterator;
	typedef boost::compute::strided_iterator<boost::compute::buffer_iterator<T const> > const_column_iterator;

	const_row_iterator row_begin(size_type i) const {
		return {buffer_begin() + i * stride1(), stride2()};
	}
	const_row_iterator row_end(size_type i) const {
		return {buffer_begin() + i * stride1()+size2()*stride2(), stride2()};
	}
	
	const_row_iterator column_begin(size_type j) const {
		return {buffer_begin() + j * stride2(), stride1()};
	}
	const_column_iterator column_end(size_type j) const {
		return {buffer_begin() + j * stride2()+size1()*stride1(), stride1()};
	}
	
	row_iterator row_begin(size_type i){
		return {buffer_begin() + i * stride1(), stride2()};
	}
	row_iterator row_end(size_type i){
		return {buffer_begin() + i * stride1()+size2()*stride2(), stride2()};
	}
	
	row_iterator column_begin(size_type j){
		return {buffer_begin() + j * stride2(), stride1()};
	}
	column_iterator column_end(size_type j){
		return {buffer_begin() + j * stride2()+size1()*stride1(), stride1()};
	}

private:
	boost::compute::buffer_iterator<T const> buffer_begin()const{
		return boost::compute::buffer_iterator<T>(m_storage.buffer, m_storage.offset);
	}

	boost::compute::buffer_iterator<T> buffer_begin(){
		return boost::compute::buffer_iterator<T>(m_storage.buffer, m_storage.offset);
	}

	std::ptrdiff_t stride1() const {
		return (std::ptrdiff_t) orientation::stride1(std::size_t(1), m_storage.leading_dimension);
	}
	std::ptrdiff_t stride2() const {
		return (std::ptrdiff_t) orientation::stride2(std::size_t(1), m_storage.leading_dimension);
	}
	
	storage_type m_storage;
	boost::compute::command_queue* m_queue;
	size_type m_size1;
	size_type m_size2;
};

}

#endif
