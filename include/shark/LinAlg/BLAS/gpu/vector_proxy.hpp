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
#ifndef REMORA_GPU_VECTOR_PROXY_HPP
#define REMORA_GPU_VECTOR_PROXY_HPP

#include "../expression_types.hpp"
#include "traits.hpp"
#include <boost/compute/iterator/strided_iterator.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

namespace remora{
	
namespace detail{
template<class Arg, class T>
struct induced_vector_adaptor_element{
	typedef T result_type;
	Arg arg;
	gpu::dense_vector_storage<T> storage;
};

template<class Arg1, class Arg2,class T>
boost::compute::detail::meta_kernel& operator<< (
	boost::compute::detail::meta_kernel& k, 
	induced_vector_adaptor_element<Arg1, Arg2, T> const& e
){
	return k<< k.get_buffer_identifier<T>(e.storage.buffer, boost::compute::memory_object::global_memory)
		<<" [ "<<e.storage.offset <<"+("<<e.arg <<")*"<<e.storage.stride<<']';
}
}
	
template<class T>
class dense_vector_adaptor<T,gpu_tag>: public vector_expression<dense_vector_adaptor<T, gpu_tag>, gpu_tag > {
public:
	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type const& const_reference;
	typedef T&  reference;

	typedef dense_vector_adaptor<T const> const_closure_type;
	typedef dense_vector_adaptor closure_type;
	typedef gpu::dense_vector_storage<T> storage_type;
	typedef gpu::dense_vector_storage<value_type const> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction

	/// \brief Constructor of a proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression The Expression from which to construct the Proxy
 	template<class E>
	dense_vector_adaptor(vector_expression<E, gpu_tag> const& expression)
	: m_storage(expression().raw_storage())
	, m_queue(&expression().queue())
	, m_size(expression().size()){}
	
	/// \brief Constructor of a proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression The Expression from which to construct the Proxy
 	template<class E>
	dense_vector_adaptor(vector_expression<E,gpu_tag>& expression)
	: m_storage(expression().raw_storage())
	, m_queue(&expression().queue())
	, m_size(expression().size()){}
	
	
	template<class E>
	dense_vector_adaptor(matrix_expression<E, gpu_tag> const& expression)
	: m_size(expression().size1() * expression().size2()), m_queue(&expression().queue()){
		gpu::dense_matrix_storage<T> storage = expression().raw_storage();
		m_storage.buffer = storage.buffer;
		m_storage.offset = storage.offset;
		m_storage.stride = storage.leading_dimension;
	}
		
	template<class E>
	dense_vector_adaptor(matrix_expression<E, gpu_tag>& expression)
	: m_size(expression().size1() * expression().size2()), m_queue(&expression().queue()){
		gpu::dense_matrix_storage<T> storage = expression().raw_storage();
		m_storage.buffer = storage.buffer;
		m_storage.offset = storage.offset;
		m_storage.stride = storage.leading_dimension;
	}

	/// \brief Copy-constructor
	/// \param v is the proxy to be copied
	template<class U>
	dense_vector_adaptor(dense_vector_adaptor<U> const& v)
	: m_storage(v.raw_storage())
	, m_queue(&v.queue())
	, m_size(v.size()){}
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_size;
	}
	
	///\brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage() const{
		return m_storage;
	}
	
	boost::compute::command_queue& queue(){
		return *m_queue;
	}
	// --------------
	// Element access
	// --------------

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	template <class IndexExpr>
	induced_vector_adaptor_element<IndexExpr,T> operator()(IndexExpr const& i){
		return {i, m_storage};
	}
	
	// --------------
	// ITERATORS
	// --------------
	

	typedef boost::compute::strided_iterator<boost::compute::buffer_iterator<T> > iterator;
	typedef boost::compute::strided_iterator<boost::compute::buffer_iterator<T> > const_iterator;

	/// \brief return an iterator on the first element of the vector
	const_iterator begin() const {
		return const_iterator(boost::compute::buffer_iterator<T>(m_storage.buffer, m_storage.offset),m_storage.stride);
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator end() const {
		return const_iterator(
			boost::compute::buffer_iterator<T>(m_storage.buffer, m_size * m_storage.stride + m_storage.offset)
			,m_storage.stride
		);
	}

	/// \brief Return an iterator on the first element of the vector
	iterator begin(){
		return iterator(boost::compute::buffer_iterator<T>(m_storage.buffer, m_storage.offset),m_storage.stride);
	}

	/// \brief Return an iterator at the end of the vector
	iterator end(){
		return iterator(
			boost::compute::buffer_iterator<T>(m_storage.buffer, m_size * m_storage.stride + m_storage.offset)
			,m_storage.stride
		);
	}
private:
	storage_type m_storage;
	boost::compute::command_queue* m_queue;
	size_type m_size;
};

}

#endif
