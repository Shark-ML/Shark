/*!
 * \brief       Implements the dense matrix class for the HIP runtime
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
#ifndef REMORA_HIP_DENSE_HPP
#define REMORA_HIP_DENSE_HPP

#include "../detail/traits.hpp"
#include "../assignment.hpp"
#include "device.hpp"
#include "traits.hpp"
#include "buffer.hpp"
namespace remora{
	
namespace hip{
	template<class M>
	__global__ void clear2d_proxy_kernel(hipLaunchParm lp, M m, size_t size1, size_t size2){
		size_t row_start = (hipBlockIdx_x * 16);
		size_t column_start = (hipBlockIdx_y * 16);
		size_t row_end = min(row_start + 16, size1);
		size_t column_end = min(column_start + 16, size2);
		for(size_t i = row_start+ hipThreadIdx_x; i < row_end; i += hipBlockDim_x){
			for(size_t j = column_start+ hipThreadIdx_y; j < column_end; j += hipBlockDim_y){
				m(i, j) = 0;
			}
		}
	}
}
	
template<class T, class Tag>
class dense_vector_adaptor<T, Tag, hip_tag>: public vector_expression<dense_vector_adaptor<T, Tag, hip_tag>, hip_tag > {
public:
	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type const& const_reference;
	typedef T&  reference;

	typedef dense_vector_adaptor<T const, Tag, hip_tag> const_closure_type;
	typedef dense_vector_adaptor closure_type;
	typedef remora::dense_vector_storage<T, Tag> storage_type;
	typedef remora::dense_vector_storage<value_type const, Tag> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction

	/// \brief Copy-constructor
	/// \param v is the proxy to be copied
	template<class U, class Tag2>
	dense_vector_adaptor(dense_vector_adaptor<U, Tag2, hip_tag> const& v)
	: m_storage(v.raw_storage())
	, m_queue(&v.queue())
	, m_size(v.size()){static_assert(std::is_convertible<Tag2,Tag>::value, "Can not convert storage type of argument to the given Tag");}
	
	/// \brief Constructor of a vector proxy from a block of memory
	/// \param storage the block of memory used
	/// \param size number of elements
	dense_vector_adaptor(
		storage_type storage, 
		hip::device& queue,
		size_type size
	):m_storage(storage)
	, m_queue(&queue)
	, m_size(size){}
	
	dense_vector_adaptor(vector<value_type, hip_tag> const& v)
	: m_storage(v.raw_storage())
	, m_queue(&v.queue())
	, m_size(v.size()){}
	
	dense_vector_adaptor(vector<value_type, hip_tag>& v)
	: m_storage(v.raw_storage())
	, m_queue(&v.queue())
	, m_size(v.size()){}
	
	// -------------------
	// Assignment operators
	// -------------------
	
	dense_vector_adaptor& operator = (dense_vector_adaptor const& e) {
		REMORA_SIZE_CHECK(size() == e().size());
		return assign(*this, typename vector_temporary<dense_vector_adaptor>::type(e));
	}
	template<class E>
	dense_vector_adaptor& operator = (vector_expression<E, hip_tag> const& e) {
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
	
	hip::device& queue() const{
		return *m_queue;
	}
	
	device_traits<hip_tag>::vector_element<T> elements() const{
		return {raw_storage()};
	}
	
	void clear(){
		hip::fill_buffer(m_storage.values, value_type(), size() * m_storage.stride, m_storage.stride, *m_queue);
	}
	
	typedef no_iterator iterator;
	typedef no_iterator const_iterator;
	
private:
	template<class,class,class> friend class dense_vector_adaptor;
	dense_vector_adaptor(vector<value_type, hip_tag> && v);//no construction from temporary vector

	storage_type m_storage;
	hip::device* m_queue;
	size_type m_size;
};
	
template<class T,class Orientation, class Tag>
class dense_matrix_adaptor<T, Orientation, Tag, hip_tag>
: public matrix_expression<dense_matrix_adaptor<T,Orientation, Tag, hip_tag>, hip_tag >{
public:
	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type const& const_reference;
	typedef T& reference;

	typedef dense_matrix_adaptor closure_type;
	typedef dense_matrix_adaptor<value_type const, Orientation, Tag, hip_tag> const_closure_type;
	typedef remora::dense_matrix_storage<T, Tag> storage_type;
	typedef remora::dense_matrix_storage<value_type const, Tag> const_storage_type;
        typedef Orientation orientation;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction
	template<class U, class Tag2>
	dense_matrix_adaptor(dense_matrix_adaptor<U, Orientation, Tag2, hip_tag> const& expression)
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
		hip::device& queue,
		size_type size1, size_type size2
	):m_storage(storage)
	, m_queue(&queue)
	, m_size1(size1)
	, m_size2(size2){}
	
	dense_matrix_adaptor(matrix<value_type, Orientation, hip_tag> const& m )
	: m_storage(m.raw_storage())
	, m_queue(&m.queue())
	, m_size1(m.size1())
	, m_size2(m.size2()){}
	
	dense_matrix_adaptor(matrix<value_type, Orientation, hip_tag>& m )
	: m_storage(m.raw_storage())
	, m_queue(&m.queue())
	, m_size1(m.size1())
	, m_size2(m.size2()){}
	
	// -------------------
	// Assignment operators
	// -------------------
	
	dense_matrix_adaptor& operator = (dense_matrix_adaptor const& e) {
		REMORA_SIZE_CHECK(size1() == e().size1());
		REMORA_SIZE_CHECK(size2() == e().size2());
		return assign(*this, typename matrix_temporary<dense_matrix_adaptor>::type(e));
	}
	template<class E>
	dense_matrix_adaptor& operator = (matrix_expression<E, hip_tag> const& e) {
		REMORA_SIZE_CHECK(size1() == e().size1());
		REMORA_SIZE_CHECK(size2() == e().size2());
		return assign(*this, typename matrix_temporary<dense_matrix_adaptor>::type(e));
	}
	template<class E>
	dense_matrix_adaptor& operator = (vector_set_expression<E, hip_tag> const& e) {
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
	
	hip::device& queue() const{
		return *m_queue;
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage() const{
		return {m_storage.values, m_storage.leading_dimension};
	}
	
	device_traits<hip_tag>::matrix_element<T, orientation> elements() const{
		return {raw_storage()};
	}
	
	void clear(){
		std::size_t blockSize1 = std::min<std::size_t>(16, m_queue->warp_size());
		std::size_t blockSize2 = std::min<std::size_t>(16, m_queue->warp_size() / blockSize1);
		std::size_t numBlocks1 = (m_size1 + 16 - 1) / 16;
		std::size_t numBlocks2 = (m_size2  + 16 - 1) / 16;
		auto stream = get_stream(*m_queue).handle();
		hipLaunchKernel(
			hip::clear2d_proxy_kernel, 
			dim3(numBlocks1, numBlocks2), dim3(blockSize1, blockSize2), 0, stream,
			elements(), m_size1, m_size2
		);
	}
	
	// Iterator types
	typedef no_iterator major_iterator;
	typedef no_iterator const_major_iterator;


private:
	storage_type m_storage;
	hip::device* m_queue;
	size_type m_size1;
	size_type m_size2;
};	
	
template<class T>
class vector<T, hip_tag>: public vector_container<vector<T, hip_tag>, hip_tag > {
public:
	typedef T value_type;
	typedef value_type const& const_reference;
	typedef value_type& reference;
	typedef std::size_t size_type;

	typedef dense_vector_adaptor<T const, continuous_dense_tag, hip_tag> const_closure_type;
	typedef dense_vector_adaptor<T, continuous_dense_tag, hip_tag> closure_type;
	typedef remora::dense_vector_storage<T const,continuous_dense_tag> const_storage_type;
	typedef remora::dense_vector_storage<T,continuous_dense_tag> storage_type;
	
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction

	/// \brief Constructor of a vector with a default queue
	///
	///note that for all operations for which vector is on the left hand side,
	///the kernels are enqueued on the supplied queue in case of a multi-queue setup.
	vector(hip::device& queue = device_traits<hip_tag>::default_queue())
	:m_storage(0, queue){}

	/// \brief Constructor of a vector with a predefined size
	/// By default, its elements are uninitialized.
	/// \param size initial size of the vector
	/// \param queue the hip queue to use by this vector
	explicit vector(size_type size, hip::device& queue = device_traits<hip_tag>::default_queue())
	: m_storage(size, queue){}

	/// \brief Constructor of a vector with a predefined size and a unique initial value
	/// \param size of the vector
	/// \param init value to assign to each element of the vector
	/// \param queue the hip queue to use by this vector
	vector(size_type size, value_type const& init, hip::device& queue = device_traits<hip_tag>::default_queue())
	: m_storage(size, queue){
		hip::fill_buffer(m_storage.get(), init, m_storage.size(), 1, queue);
	}
	
	/// \brief Move-constructor of a vector
	/// \param v is the vector to be moved
	vector(vector && v): m_storage(std::move(v.m_storage)){}

	/// \brief Copy-constructor of a vector
	/// \param v is the vector to be duplicated
	vector(vector const& v) = default;

	/// \brief Copy-constructor of a vector from a vector_expression
	/// \param e the vector_expression whose values will be duplicated into the vector
	template<class E>
	vector(vector_expression<E, hip_tag> const& e)
	: m_storage(e().size(), e().queue()){
		assign(*this, e);
	}
	
	/// \brief Copy-constructor of a vector from a vector_expression on a given queue
	/// \param e the vector_expression whose values will be duplicated into the vector
	/// \param queue the queue which should perform the task
	template<class E>
	vector(vector_expression<E, hip_tag> const& e, hip::device& queue)
	: m_storage(e().size(), queue){
		assign(*this, e);
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
		return *this;
	}
	
	/// \brief Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector)
	/// Assign a full vector (\e RHS-vector) to the current vector (\e LHS-vector). This method does not create any temporary.
	/// \param v is the source vector container
	/// \return a reference to a vector (i.e. the destination vector)
	template<class C>          // Container assignment without temporary
	vector& operator = (vector_container<C, hip_tag> const& v) {
		resize(v().size());
		return assign(*this, v);
	}

	/// \brief Assign the result of a vector_expression to the vector
	///
	/// \param e is a const reference to the vector_expression
	/// \return a reference to the resulting vector
	template<class E>
	vector& operator = (vector_expression<E, hip_tag> const& e) {
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
	
	hip::device& queue() const{
		return m_storage.device();
	}
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return {m_storage.get(), 1};
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return {m_storage.get(), 1};
	}
	
	device_traits<hip_tag>::vector_element<T> elements(){
		return {raw_storage()};
	}
	device_traits<hip_tag>::vector_element<T const> elements() const{
		return {raw_storage()};
	}
	
	/// \brief Resize the vector
	///
	/// This might erase all data stored in the vector
	///
	/// \param size new size of the vector
	void resize(size_type size) {
		m_storage.resize(size);
	}
	
	void clear(){
		hip::fill_buffer(m_storage.get(), value_type(), m_storage.size(), 1, m_storage.device());
	}
	
	bool empty()const{
		return size() == 0;
	}
	
	/// \brief Swap the content of two vectors
	/// \param v1 is the first vector. It takes values from v2
	/// \param v2 is the second vector It takes values from v1
	friend void swap(vector& v1, vector& v2) {
		std::swap(v1.m_storage,v2.m_storage);
	}
	
	template<class Archive>
	void serialize(Archive &ar, const unsigned int file_version) {}
	
	// Iterator types
	typedef no_iterator iterator;
	typedef no_iterator const_iterator;
private:
	hip::buffer<value_type> m_storage;
};

/// \brief A dense matrix of values of type \c T stored on the hip
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
class matrix<T,L, hip_tag> : public matrix_container<matrix<T,L, hip_tag>, hip_tag > {
public:
	typedef T value_type;
	typedef value_type const& const_reference;
	typedef value_type& reference;
	typedef std::size_t size_type;

	typedef dense_matrix_adaptor<T const,L, continuous_dense_tag, hip_tag> const_closure_type;
	typedef dense_matrix_adaptor<T,L, continuous_dense_tag, hip_tag> closure_type;
	typedef remora::dense_matrix_storage<T, continuous_dense_tag> storage_type;
	typedef remora::dense_matrix_storage<T const, continuous_dense_tag> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;
	typedef L orientation;

	// Construction and destruction

	/// \brief Constructor of a matrix with a default queue
	///
	///note that for all operations for which matrix is on the left hand side,
	///the kernels are enqueued on the supplied queue in case of a multi-queue setup.
	matrix(hip::device& queue = device_traits<hip_tag>::default_queue())
	: m_storage(0, queue), m_size1(0), m_size2(0){}

	/// \brief Constructor of a matrix with a predefined size
	/// By default, its elements are uninitialized
	/// \param size1 number of rows
	/// \param size2 number of columns
	/// \param queue the hip queue to use by this matrix
	explicit matrix(size_type size1, size_type size2, hip::device& queue = device_traits<hip_tag>::default_queue())
	: m_storage(size1 * size2, queue)
	, m_size1(size1)
	, m_size2(size2){}

	/// \brief Constructor of a matrix with a predefined size initialized to a value
	/// \param size1 number of rows
	/// \param size2 number of columns
	/// \param init value to assign to each element of the matrix
	/// \param queue the hip queue to use by this matrix
	matrix(size_type size1, size_type size2, value_type const& init, hip::device& queue = device_traits<hip_tag>::default_queue())
	: m_storage(size1 * size2, queue)
	, m_size1(size1)
	, m_size2(size2){
		hip::fill_buffer(m_storage.get(), init, m_storage.size(), 1, m_storage.device());
	}
	
	/// \brief Move-constructor of a matrix
	/// \param m is the matrix to be moved
	matrix(matrix && m)
	: m_storage(std::move(m.m_storage))
	, m_size1(m.size1())
	, m_size2(m.size2()){}

	/// \brief Copy-constructor of a matrix
	/// \param m is the matrix to be duplicated
	matrix(matrix const& m)
	: m_storage(m.m_storage)
	, m_size1(m.size1())
	, m_size2(m.size2()){}

	/// \brief Copy-constructor of a matrix from a matrix_expression
	/// \param e the matrix_expression whose values will be duplicated into the matrix
	template<class E>
	matrix(matrix_expression<E, hip_tag> const& e)
	: m_storage(e().size1() * e().size2(), e().queue())
	, m_size1(e().size1())
	, m_size2(e().size2()){
		assign(*this, e);
	}
	
	template<class E>
	matrix(vector_set_expression<E, hip_tag> const& e)
	: m_storage(e().size() * e().point_size(), e().queue())
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
		if(size1() != m.size1() || size2() != m.size2()){
			m_storage.resize(m.size1() * m.size2());
		}
		return assign(*this, m);
	}
	
	/// \brief Move-Assign a full matrix (\e RHS-matrix) to the current matrix (\e LHS-matrix)
	/// \param m is the source matrix container
	/// \return a reference to a matrix (i.e. the destination matrix)
	matrix& operator = (matrix && m){
		m_storage = std::move(m.m_storage);
		m_size1 = m.m_size1;
		m_size2 = m.m_size2;
		return *this;
	}
	
	/// \brief Assign a full matrix (\e RHS-matrix) to the current matrix (\e LHS-matrix)
	/// Assign a full matrix (\e RHS-matrix) to the current matrix (\e LHS-matrix). This method does not create any temporary.
	/// \param m is the source matrix container
	/// \return a reference to a matrix (i.e. the destination matrix)
	template<class C>          // Container assignment without temporary
	matrix& operator = (matrix_container<C, hip_tag> const& m) {
		if(size1() != m.size1() || size2() != m.size2()){
			m_storage.resize(m.size1() * m.size2());
		}
		return assign(*this, m);
	}

	/// \brief Assign the result of a matrix_expression to the matrix
	///
	/// \param e is a const reference to the matrix_expression
	/// \return a reference to the resulting matrix
	template<class E>
	matrix& operator = (matrix_expression<E, hip_tag> const& e) {
		matrix temporary(e);
		swap(*this,temporary);
		return *this;
	}
	
	
	/// \brief Assign the result of a vector_set to the matrix
	///
	/// \param e is a const reference to the vector_set_expression
	/// \return a reference to the resulting matrix
	template<class E>
	matrix& operator = (vector_set_expression<E, hip_tag> const& e) {
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
	
	hip::device& queue() const{
		return m_storage.device();
	}
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return {m_storage.get(), orientation::index_m(m_size1, m_size2)};
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return {m_storage.get(), orientation::index_m(m_size1, m_size2)};
	}
	
	device_traits<hip_tag>::matrix_element<value_type, orientation> elements(){
		return {raw_storage()};
	}
	device_traits<hip_tag>::matrix_element<value_type const, orientation> elements() const{
		return {raw_storage()};
	}
	
	/// \brief Resize the matrix
	///
	/// This might erase all data stored in the matrix
	///
	/// \param size1 new number of rows
	/// \param size2 new number of columns
	void resize(size_type size1, size_type size2) {
		m_storage.resize(size1 * size2);
		m_size1 = size1;
		m_size2 = size2;
	}
	
	void clear(){
		hip::fill_buffer(m_storage.get(), value_type(), m_storage.size(), 1, m_storage.device());
	}
	
	// Iterator types
	typedef no_iterator major_iterator;
	typedef no_iterator const_major_iterator;

	/// \brief Swap the content of two matrixs
	/// \param m1 is the first matrix. It takes values from m2
	/// \param m2 is the second matrix It takes values from m1
	friend void swap(matrix& m1, matrix& m2) {
		std::swap(m1.m_storage,m2.m_storage);
		std::swap(m1.m_size1,m2.m_size1);
		std::swap(m1.m_size2,m2.m_size2);
	}
	
	template<class Archive>
	void serialize(Archive &ar, const unsigned int file_version) {}
private:
	hip::buffer<value_type> m_storage;
	size_type m_size1;
	size_type m_size2;
};


template<class T, class Orientation, bool Upper, bool Unit>
class dense_triangular_proxy<T, Orientation, triangular_tag<Upper, Unit> , hip_tag>
: public matrix_expression<dense_triangular_proxy<T, Orientation, triangular_tag<Upper, Unit>, hip_tag>, hip_tag> {
public:
	typedef std::size_t size_type;
	typedef typename std::remove_const<T>::type value_type;
	typedef value_type result_type;
	typedef typename std::conditional<Unit, value_type const&, T&>::type reference;
	typedef value_type const& const_reference;
	typedef dense_triangular_proxy<value_type const, Orientation, triangular_tag<Upper, Unit> , hip_tag> const_closure_type;
	typedef dense_triangular_proxy<T, Orientation, triangular_tag<Upper, Unit> , hip_tag> closure_type;

	typedef dense_matrix_storage<T, dense_tag> storage_type;
	typedef dense_matrix_storage<value_type const, dense_tag> const_storage_type;

	typedef elementwise<dense_tag> evaluation_category;
	typedef triangular<Orientation,triangular_tag<Upper, Unit> > orientation;


	template<class U>
	dense_triangular_proxy(dense_triangular_proxy<U, Orientation, triangular_tag<Upper, Unit>, hip_tag> const& expression)
	: m_storage(expression.raw_storage())
	, m_queue(&expression.queue())
	, m_size1(expression.size1())
	, m_size2(expression.size2()){}

	dense_triangular_proxy(storage_type const& storage, hip::device& queue, std::size_t size1, std::size_t size2)
	: m_storage(storage)
	, m_queue(&queue)
	, m_size1(size1)
	, m_size2(size2){}
	
	dense_matrix_adaptor<T, Orientation, dense_tag, hip_tag> to_dense() const{
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
	
	 hip::device& queue()const{
		return *m_queue;
	}

	typedef no_iterator major_iterator;
	typedef no_iterator const_major_iterator;
private:
	storage_type m_storage;
	hip::device* m_queue;
	std::size_t m_size1;
	std::size_t m_size2;
};

//////////////////////////////////
//////Expression Traits
///////////////////////////////////

namespace detail{

template<class T, class Orientation>
struct vector_to_matrix_optimizer<dense_vector_adaptor<T, continuous_dense_tag, hip_tag>, Orientation >{
	typedef dense_matrix_adaptor<T, Orientation, continuous_dense_tag, hip_tag> type;
	
	static type create(
		dense_vector_adaptor<T, continuous_dense_tag, hip_tag> const& v,
		std::size_t size1, std::size_t size2
	){
		dense_matrix_storage<T, continuous_dense_tag> storage = {v.raw_storage().values, Orientation::index_m(size1,size2)};
		return type(storage, v.queue(), size1, size2);
	}
};
}

}

#endif
