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
#ifndef REMORA_GPU_DENSE_HPP
#define REMORA_GPU_DENSE_HPP

#include "../detail/traits.hpp"
#include "../assignment.hpp"
#include <boost/compute/container/vector.hpp>
#include <boost/compute/iterator/strided_iterator.hpp>
#include <boost/compute/algorithm/fill.hpp>
namespace remora{
	
	
template<class T, class Tag>
class dense_vector_adaptor<T, Tag, gpu_tag>: public vector_expression<dense_vector_adaptor<T, Tag, gpu_tag>, gpu_tag > {
public:
	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type const& const_reference;
	typedef T&  reference;

	typedef dense_vector_adaptor<T const, Tag, gpu_tag> const_closure_type;
	typedef dense_vector_adaptor closure_type;
	typedef gpu::dense_vector_storage<T, Tag> storage_type;
	typedef gpu::dense_vector_storage<value_type const, Tag> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction

	/// \brief Copy-constructor
	/// \param v is the proxy to be copied
	template<class U, class Tag2>
	dense_vector_adaptor(dense_vector_adaptor<U, Tag2, gpu_tag> const& v)
	: m_storage(v.raw_storage())
	, m_queue(&v.queue())
	, m_size(v.size()){static_assert(std::is_convertible<Tag2,Tag>::value, "Can not convert storage type of argument to the given Tag");}
	
	/// \brief Constructor of a vector proxy from a block of memory
	/// \param storage the block of memory used
	/// \param size number of elements
	dense_vector_adaptor(
		storage_type storage, 
		boost::compute::command_queue& queue,
		size_type size
	):m_storage(storage)
	, m_queue(&queue)
	, m_size(size){}
	
	dense_vector_adaptor(vector<value_type, gpu_tag> const& v)
	: m_storage(v().raw_storage())
	, m_queue(&v().queue())
	, m_size(v().size()){}
	
	dense_vector_adaptor(vector<value_type, gpu_tag>& v)
	: m_storage(v().raw_storage())
	, m_queue(&v().queue())
	, m_size(v().size()){}
	
	// -------------------
	// Assignment operators
	// -------------------
	
	dense_vector_adaptor& operator = (dense_vector_adaptor const& e) {
		REMORA_SIZE_CHECK(size() == e().size());
		return assign(*this, typename vector_temporary<dense_vector_adaptor>::type(e));
	}
	template<class E>
	dense_vector_adaptor& operator = (vector_expression<E, gpu_tag> const& e) {
		REMORA_SIZE_CHECK(size() == e().size());
		return assign(*this, typename vector_temporary<dense_vector_adaptor>::type(e));
	}
	
	/// \brief Return the size of the vector.
	size_type size() const {
		return m_size;
	}
	
	///\brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage() const{
		return m_storage;
	}
	
	boost::compute::command_queue& queue() const{
		return *m_queue;
	}
	
	void clear(){
		gpu::detail::meta_kernel k("vector_proxy_clear");
		auto v = k.register_args(to_functor(*this));
	
		//create source
		k<<v(k.get_global_id(0))<<" = 0;";
		boost::compute::kernel kernel = k.compile(queue().get_context());
		//enqueue kernel
		std::size_t global_work_size[1] = {size()};
		queue().enqueue_nd_range_kernel(kernel, 1,nullptr, global_work_size, nullptr);
	}
	
	typedef no_iterator iterator;
	typedef no_iterator const_iterator;
	
private:
	template<class,class,class> friend class dense_vector_adaptor;
	dense_vector_adaptor(vector<value_type, gpu_tag> && v);//no construction from temporary vector

	storage_type m_storage;
	boost::compute::command_queue* m_queue;
	size_type m_size;
};
	
template<class T,class Orientation, class Tag>
class dense_matrix_adaptor<T, Orientation, Tag, gpu_tag>: public matrix_expression<dense_matrix_adaptor<T,Orientation, Tag, gpu_tag>, gpu_tag > {
public:
	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type const& const_reference;
	typedef T& reference;

	typedef dense_matrix_adaptor closure_type;
	typedef dense_matrix_adaptor<value_type const, Orientation, Tag, gpu_tag> const_closure_type;
	typedef gpu::dense_matrix_storage<T, Tag> storage_type;
	typedef gpu::dense_matrix_storage<value_type const, Tag> const_storage_type;
        typedef Orientation orientation;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction
	template<class U, class Tag2>
	dense_matrix_adaptor(dense_matrix_adaptor<U, Orientation, Tag2, gpu_tag> const& expression)
	: m_storage(expression.raw_storage())
	, m_queue(&expression.queue())
	, m_size1(expression.size1())
	, m_size2(expression.size2())
	{static_assert(std::is_convertible<Tag2,Tag>::value, "Can not convert storage type of argument to the given Tag");}
		
	/// \brief Constructor of a matrix proxy from a block of memory
	/// \param storage the block of memory used
	/// \param size1 number of rows
	/// \param size2 number of columns
	dense_matrix_adaptor(
		storage_type storage, 
		boost::compute::command_queue& queue,
		size_type size1, size_type size2
	):m_storage(storage)
	, m_queue(&queue)
	, m_size1(size1)
	, m_size2(size2){}
	
	dense_matrix_adaptor(matrix<value_type, Orientation, gpu_tag> const& m )
	: m_storage(m().raw_storage())
	, m_queue(&m().queue())
	, m_size1(m().size1())
	, m_size2(m().size2()){}
	
	dense_matrix_adaptor(matrix<value_type, Orientation, gpu_tag>& m )
	: m_storage(m().raw_storage())
	, m_queue(&m().queue())
	, m_size1(m().size1())
	, m_size2(m().size2()){}
	
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
	template<class E>
	dense_matrix_adaptor& operator = (vector_set_expression<E, gpu_tag> const& e) {
		REMORA_SIZE_CHECK(size1() == typename E::point_orientation::index_M(e().size(), e().point_size()));
		REMORA_SIZE_CHECK(size2() == typename E::point_orientation::index_M(e().size(), e().point_size()));
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
	storage_type raw_storage() const{
		return {m_storage.buffer, m_storage.offset, m_storage.leading_dimension};
	}
	
	void clear(){
		gpu::detail::meta_kernel k("matrix_proxy_clear");
		auto m = k.register_args(to_functor(*this));
	
		//create source
		k<<m(k.get_global_id(0),k.get_global_id(1))<<" = 0;";
		boost::compute::kernel kernel = k.compile(queue().get_context());
		//enqueue kernel
		std::size_t global_work_size[2] = {size1(), size2()};
		queue().enqueue_nd_range_kernel(kernel, 2,nullptr, global_work_size, nullptr);
	}
	
	// Iterator types
	typedef no_iterator major_iterator;
	typedef no_iterator const_major_iterator;


private:
	storage_type m_storage;
	boost::compute::command_queue* m_queue;
	size_type m_size1;
	size_type m_size2;
};	
	
template<class T>
class vector<T, gpu_tag>: public vector_container<vector<T, gpu_tag>, gpu_tag > {
public:
	typedef T value_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef std::size_t size_type;

	typedef dense_vector_adaptor<T const, continuous_dense_tag, gpu_tag> const_closure_type;
	typedef dense_vector_adaptor<T, continuous_dense_tag, gpu_tag> closure_type;
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
	gpu::detail::dense_vector_element<value_type> to_functor() const{
		return  {m_storage.get_buffer()}; 
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
		return assign(*this, v);
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
	
	template<class Archive>
	void serialize(Archive &ar, const unsigned int file_version) {
	}
	
	// Iterator types
	typedef no_iterator iterator;
	typedef no_iterator const_iterator;
private:
	boost::compute::vector<T> m_storage;
	boost::compute::command_queue* m_queue;
};

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
template<class T, class L>
class matrix<T,L, gpu_tag> : public matrix_container<matrix<T,L, gpu_tag>, gpu_tag > {
public:
	typedef T value_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef std::size_t size_type;

	typedef dense_matrix_adaptor<T const,L, continuous_dense_tag, gpu_tag> const_closure_type;
	typedef dense_matrix_adaptor<T,L, continuous_dense_tag, gpu_tag> closure_type;
	typedef gpu::dense_matrix_storage<T, continuous_dense_tag> storage_type;
	typedef gpu::dense_matrix_storage<T const, continuous_dense_tag> const_storage_type;
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
	
	template<class E>
	matrix(vector_set_expression<E, gpu_tag> const& e)
	: m_storage(e().size() * e().point_size(), e().queue().get_context())
	, m_queue(&e().queue())
	, m_size1(E::point_orientation::index_M(e().size(), e().point_size()))
	, m_size2(E::point_orientation::index_m(e().size(), e().point_size())){
		assign(*this, e().expression());
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
	
	
	/// \brief Assign the result of a vector_set to the matrix
	///
	/// \param e is a const reference to the vector_set_expression
	/// \return a reference to the resulting matrix
	template<class E>
	matrix& operator = (vector_set_expression<E, gpu_tag> const& e) {
		matrix temporary(e);
		swap(temporary);
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
		return {m_storage.get_buffer(),0,leading_dimension()};
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return {m_storage.get_buffer(),0,leading_dimension()};
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
	typedef no_iterator major_iterator;
	typedef no_iterator const_major_iterator;

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
	
	template<class Archive>
	void serialize(Archive &ar, const unsigned int file_version) {
	}
private:
	std::size_t leading_dimension() const{
		return orientation::index_m(m_size1, m_size2);
	};
	
	boost::compute::vector<T> m_storage;
	boost::compute::command_queue* m_queue;
	size_type m_size1;
	size_type m_size2;
};


template<class T, class Orientation, bool Upper, bool Unit>
class dense_triangular_proxy<T, Orientation, triangular_tag<Upper, Unit> , gpu_tag>
: public matrix_expression<dense_triangular_proxy<T, Orientation, triangular_tag<Upper, Unit>, gpu_tag>, gpu_tag> {
public:
	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type result_type;
	typedef typename std::conditional<Unit, value_type const&, T&>::type reference;
	typedef value_type const& const_reference;
	typedef dense_triangular_proxy<value_type const, Orientation, triangular_tag<Upper, Unit> , gpu_tag> const_closure_type;
	typedef dense_triangular_proxy<T, Orientation, triangular_tag<Upper, Unit> , gpu_tag> closure_type;

	typedef gpu::dense_matrix_storage<T, dense_tag> storage_type;
	typedef gpu::dense_matrix_storage<value_type const, dense_tag> const_storage_type;

	typedef elementwise<dense_tag> evaluation_category;
	typedef triangular<Orientation,triangular_tag<Upper, Unit> > orientation;


	template<class U>
	dense_triangular_proxy(dense_triangular_proxy<U, Orientation, triangular_tag<Upper, Unit>, gpu_tag> const& expression)
	: m_storage(expression.raw_storage())
	, m_queue(&expression.queue())
	, m_size1(expression.size1())
	, m_size2(expression.size2()){}

	dense_triangular_proxy(storage_type const& storage, boost::compute::command_queue& queue, std::size_t size1, std::size_t size2)
	: m_storage(storage)
	, m_queue(&queue)
	, m_size1(size1)
	, m_size2(size2){}
	
	dense_matrix_adaptor<T, Orientation, dense_tag, gpu_tag> to_dense() const{
		return {m_storage, queue(), m_size1, m_size2};
	}
	
	
	/// \brief Return the number of rows of the matrix
	size_type size1() const {
		return m_size1;
	}
	/// \brief Return the number of columns of the matrix
	size_type size2() const {
		return m_size2;
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage() const{
		return m_storage;
	}
	
	 boost::compute::command_queue& queue()const{
		return *m_queue;
	}

	typedef no_iterator major_iterator;
	typedef no_iterator const_major_iterator;
private:
	storage_type m_storage;
	boost::compute::command_queue* m_queue;
	std::size_t m_size1;
	std::size_t m_size2;
};

//////////////////////////////////
//////Expression Traits
///////////////////////////////////

template<class T>
struct ExpressionToFunctor<vector<T, gpu_tag> >{
	static gpu::detail::dense_vector_element<T> transform(vector<T, gpu_tag> const& e){
		return {e().raw_storage().buffer, 1, 0}; 
	}
};

template<class T, class Orientation>
struct ExpressionToFunctor<matrix<T, Orientation, gpu_tag> >{
	static gpu::detail::dense_matrix_element<T> transform(matrix<T, Orientation, gpu_tag> const& e){
		std::size_t leading = e().raw_storage().leading_dimension;
		return {e().raw_storage().buffer, Orientation::stride1(leading), Orientation::stride2(leading),0}; 
	}
};



template<class T, class Tag>
struct ExpressionToFunctor<dense_vector_adaptor<T, Tag, gpu_tag> >{
	static gpu::detail::dense_vector_element<T> transform(dense_vector_adaptor<T, Tag, gpu_tag> const& e){
		auto const& storage = e().raw_storage(); 
		return {storage.buffer, storage.stride, storage.offset}; 
	}
};

template<class T, class Tag, class Orientation>
struct ExpressionToFunctor<dense_matrix_adaptor<T, Orientation, Tag, gpu_tag> >{
	static gpu::detail::dense_matrix_element<T> transform(dense_matrix_adaptor<T, Orientation, Tag, gpu_tag> const& e){
		auto const& storage = e().raw_storage(); 
		std::size_t stride1 = Orientation::index_m(std::size_t(1), storage.leading_dimension);
		std::size_t stride2 = Orientation::index_M(std::size_t(1), storage.leading_dimension);
		return {storage.buffer, stride1, stride2, storage.offset}; 
	}
};

namespace detail{

template<class T, class Orientation>
struct vector_to_matrix_optimizer<dense_vector_adaptor<T, continuous_dense_tag, gpu_tag>, Orientation >{
	typedef dense_matrix_adaptor<T, Orientation, continuous_dense_tag, gpu_tag> type;
	
	static type create(
		dense_vector_adaptor<T, continuous_dense_tag, gpu_tag> const& v,
		std::size_t size1, std::size_t size2
	){
		gpu::dense_matrix_storage<T, continuous_dense_tag> storage = {v.raw_storage().buffer, v.raw_storage().offset, Orientation::index_m(size1,size2)};
		return type(storage, v.queue(), size1, size2);
	}
};
}

}

#endif
