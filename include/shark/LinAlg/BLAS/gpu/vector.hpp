/*!
 * \brief       Implements the dense vector class for the gpu
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
#ifndef REMORA_GPU_VECTOR_HPP
#define REMORA_GPU_VECTOR_HPP

#include "../detail/traits.hpp"
#include "../detail/vector_proxy_classes.hpp"
#include "../assignment.hpp"
#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/fill.hpp>

namespace remora{

namespace detail{
template<class Arg, class T>
struct induced_vector_element{
	typedef T result_type;
	Arg arg;
	std::size_t stride;
	boost::compute::buffer const& buffer;
};

template<class Arg,class T>
boost::compute::detail::meta_kernel& operator<< (
	boost::compute::detail::meta_kernel& k, 
	induced_vector_element<Arg, T> const& e
){
	return k << k.get_buffer_identifier<T>(e.buffer, boost::compute::memory_object::global_memory)
	             <<'['<<e.arg <<'*'<<e.stride<<']';
}
}

/// \brief A dense vector of values of type \c T sored on the GPU
///
/// For a \f$n\f$-dimensional vector \f$v\f$ and \f$0\leq i < n\f$ every element \f$v_i\f$ is mapped
/// to the \f$i\f$-th element of the container.
///
/// \tparam T type of the objects stored in the vector (like int, double, complex,...)
template<class T>
class vector<T, gpu_tag>: public vector_container<vector<T, gpu_tag>, gpu_tag > {
public:
	typedef T value_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef std::size_t size_type;

	typedef vector_reference<vector const> const_closure_type;
	typedef vector_reference<vector> closure_type;
	typedef gpu::dense_vector_storage<T,continuous_dense_tag> storage_type;
	typedef gpu::dense_vector_storage<T,continuous_dense_tag> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction

	/// \brief Constructor of a vector with a default queue
	///
	///note that for all operations for which vector is on the left hand side,
	///the kernels are enqueued on the supplied queue in case of a multi-queue setup.
	vector(boost::compute::command_queue& queue = boost::compute::system::default_queue())
	:m_storage(queue.get_context()), m_queue(&queue){}

	/// \brief Constructor of a vector with a predefined size
	/// By default, its elements are uninitialized.
	/// \param size initial size of the vector
	/// \param queue the opencl queue to use by this vector
	explicit vector(size_type size, boost::compute::command_queue& queue = boost::compute::system::default_queue())
	: m_storage(size,queue.get_context()), m_queue(&queue){}

	/// \brief Constructor of a vector with a predefined size and a unique initial value
	/// \param size of the vector
	/// \param init value to assign to each element of the vector
	/// \param queue the opencl queue to use by this vector
	vector(size_type size, value_type const& init, boost::compute::command_queue& queue = boost::compute::system::default_queue())
	: m_storage(size, init, queue), m_queue(&queue){}
	
	/// \brief Move-constructor of a vector
	/// \param v is the vector to be moved
	vector(vector && v)
	: m_storage(std::move(v.m_storage))
	, m_queue(&v.queue()){}

	/// \brief Copy-constructor of a vector
	/// \param v is the vector to be duplicated
	vector(vector const& v) = default;

	/// \brief Copy-constructor of a vector from a vector_expression
	/// \param e the vector_expression whose values will be duplicated into the vector
	template<class E>
	vector(vector_expression<E, gpu_tag> const& e)
	: m_storage(e().size(), e().queue().get_context())
	, m_queue(&e().queue()){
		assign(*this, e);
	}
	
	/// \brief Copy-constructor of a vector from a vector_expression on a given queue
	/// \param e the vector_expression whose values will be duplicated into the vector
	/// \param queue the queue which should perform the task
	template<class E>
	vector(vector_expression<E, gpu_tag> const& e, boost::compute::command_queue& queue)
	: m_storage(e().size(), queue.get_context())
	, m_queue(&queue){
		assign(*this, e);
	}
	
	// Element access
	template <class IndexExpr>
	detail::induced_vector_element<IndexExpr,T> operator()(IndexExpr const& i) const {
		return {i,1,m_storage.get_buffer()};
	}
	
	// -------------------
	// Assignment operators
	// -------------------
	
	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector). This method does not create any temporary.
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	vector& operator = (vector const& v){
		resize(v.size());
		assign(*this, v);
	}
	
	/// \brief Move-Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	vector& operator = (vector && v){
		m_storage = std::move(v.m_storage);
		m_queue = v.m_queue;
		return *this;
	}
	
	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector). This method does not create any temporary.
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	template<class C>          // Container assignment without temporary
	vector& operator = (vector_container<C, gpu_tag> const& v) {
		resize(v().size());
		return assign(*this, v);
	}

	/// \brief Assign the result of a vector_expression to the vector
	///
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vector& operator = (vector_expression<E, gpu_tag> const& e) {
		vector temporary(e,queue());
		swap(*this,temporary);
		return *this;
	}

	// ---------
	// Storage interface
	// ---------
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_storage.size();
	}
	
	boost::compute::command_queue& queue() const{
		return *m_queue;
	}
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return {m_storage.get_buffer(),0,1};
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return {m_storage.get_buffer(),0,1};
	}
	
	/// \brief Resize the vector
	///
	/// This might erase all data stored in the vector
	///
	/// \param size new size of the vector
	void resize(size_type size) {
		if(size < m_storage.size())
			m_storage.resize(size);
		else
			m_storage = boost::compute::vector<T>(size, queue().get_context());
	}
	
	/// \brief Resize the vector
	///
	/// This will erase all data stored in the vector and reinitialize it with the supplied value of init
	///
	/// \param size new size of the vector
	/// \param init the value of all elements
	void resize(size_type size, value_type init) {
		resize(size);
		boost::compute::fill(m_storage.begin(),m_storage.end(), init);
	}
	
	void clear(){
		boost::compute::fill(m_storage.begin(),m_storage.end(), value_type/*zero*/());
	}
	
	bool empty()const{
		return m_storage.empty();
	}
	
	/// \brief Swap the content of two vectors
	/// \param v1 is the first vector. It takes values from v2
	/// \param v2 is the second vector It takes values from v1
	friend void swap(vector& v1, vector& v2) {
		using std::swap;
		swap(v1.m_storage,v2.m_storage);
		std::swap(v2.m_queue,v2.m_queue);
	}
	
	// Iterator types
	typedef typename boost::compute::vector<T>::iterator iterator;
	typedef typename boost::compute::vector<T>::const_iterator const_iterator;
	
	/// \brief return an iterator on the first element of the vector
	const_iterator cbegin() const {
		return m_storage.begin();
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator cend() const {
		return m_storage.end();
	}

	/// \brief return an iterator on the first element of the vector
	const_iterator begin() const {
		return cbegin();
	}

	/// \brief return an iterator after the last element of the vector
	const_iterator end() const {
		return cend();
	}

	/// \brief Return an iterator on the first element of the vector
	iterator begin(){
		return m_storage.begin();
	}

	/// \brief Return an iterator at the end of the vector
	iterator end(){
		return m_storage.end();
	}
	
private:
	boost::compute::vector<T> m_storage;
	boost::compute::command_queue* m_queue;
};
}

#endif
