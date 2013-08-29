/*!
 *  \brief Proxycontainer for Dense Vectors
 *
 * Sometimes we have to use self_types in the context of virtual context. Since virtual
 * templates are not allowed, a wrapper is needed which hides the type.
 * However type erasure is slow regarding element access, but for self_type types
 * with actual storage, like RealVector, one can emulate a general self_type by directly
 * accessing the memory.
 *
 *  \author O.Krause
 *  \date 2012
 *
 *  \par Copyright(c) 1998-2007:
 *	  Institut f&uuml;r Neuroinformatik<BR>
 *	  Ruhr-Universit&auml;t Bochum<BR>
 *	  D-44780 Bochum, Germany<BR>
 *	  Phone: +49-234-32-25558<BR>
 *	  Fax:   +49-234-32-14209<BR>
 *	  eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *	  www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *	  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or(at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#ifndef SHARK_LINALG_PROXY_H
#define SHARK_LINALG_PROXY_H

#include <shark/LinAlg/BLAS/ublas.h>
#include <shark/LinAlg/BLAS/traits/vector_raw.hpp>
#include <shark/LinAlg/BLAS/traits/matrix_raw.hpp>
#include <shark/Core/utility/Iterators.h>

#include <algorithm>

namespace shark{namespace blas{

template<class ValueType>
class FixedDenseVectorProxy: public vector_expression<FixedDenseVectorProxy<ValueType> > {
	typedef FixedDenseVectorProxy<ValueType> self_type;
public:

	//std::container types
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename boost::remove_const<ValueType>::type value_type;
	typedef value_type const& const_reference;
	typedef ValueType&  reference;
	typedef ValueType* pointer;
	typedef value_type const* const_pointer;
	//ublas types
	typedef vector_reference<self_type const> const const_closure_type;
	typedef vector_reference<self_type> closure_type;
	typedef dense_tag storage_category;

	// Construction and destruction

	/// \brief Constructor of a self_type proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression The Expression from which to construct the Proxy
 	template<class E>
	FixedDenseVectorProxy(vector_expression<E> const& expression):
		m_data(traits::vector_storage(expression())),
		m_size(expression().size()),
		m_stride(traits::vector_stride(expression())){
		
		BOOST_STATIC_ASSERT(traits::IsDense<E>::value);
	}
	
	/// \brief Constructor of a self_type proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression The Expression from which to construct the Proxy
 	template<class E>
	FixedDenseVectorProxy(vector_expression<E>& expression)
	: m_data(traits::vector_storage(expression()))
	, m_size(expression().size())
	, m_stride(traits::vector_stride(expression()))
	{
		
		BOOST_STATIC_ASSERT(traits::IsDense<E>::value);
	}
		
	/// \brief Constructor of a self_type proxy from a block of memory
	/// \param memory the block of memory used
	/// \param size size of the self_type
 	/// \param stride distance between elements of the self_type in memory
	FixedDenseVectorProxy(pointer data, size_type size, difference_type stride = 1 ):
		m_data(data),m_size(size),m_stride(stride){}	

	/// \brief Copy-constructor of a self_type
	/// \param v is the proxy to be copied
	template<class T>
	FixedDenseVectorProxy(FixedDenseVectorProxy<T> const& v)
	:m_data(v.data()),m_size(v.size()),m_stride(v.stride())
	{}
	
	/// \brief Return the size of the self_type
	size_type size() const {
		return m_size;
	}
	
	difference_type stride()const{
		return m_stride;
	}
	
	pointer data()const{
		return m_data;
	}
	
	bool same_closure(self_type const& t) const {
		//same closure if the data segments are overlapping
		return (t.data()+t.size()) > data() && (t.data() < data()+size());
	}
	
	// --------------
	// Element access
	// --------------
	
	/// \brief Return a pointer to the element \f$i\f$
	/// \param i index of the element
	pointer find_element(size_type i) {
		return m_data+(i*m_stride);
	}

	/// \brief Return a const pointer to the element \f$i\f$
	/// \param i index of the element
	const_pointer find_element(size_type i) const {
		return m_data+(i*m_stride);
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator()(size_type i) const {
		return m_data[i*m_stride];
	}
	
	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator()(size_type i) {
		return m_data[i*m_stride];
	}	

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	const_reference operator[](size_type i) const {
		return m_data[i*m_stride];
	}
	
	/// \brief Return a reference to the element \f$i\f$
	/// \param i index of the element
	reference operator[](size_type i) {
		return m_data[i*m_stride];
	}

	// ------------------
	// Element assignment
	// ------------------
	
	/// \brief Set element \f$i\f$ to the value \c t
	/// \param i index of the element
	/// \param t reference to the value to be set
	reference insert_element(size_type i, const_reference t) {
		return(*this)[i] = t;
	}

	/// \brief Set element \f$i\f$ to the \e zero value
	/// \param i index of the element
	void erase_element(size_type i) {
		(*this)[i] = value_type/*zero*/();
	}
		
	// -------
	// ASSIGNING
	// -------

	 // Assignment
	template<class V>
	self_type& assign(vector_expression<V> const& other) {
		SIZE_CHECK(other().size() == size());
		for(std::size_t i = 0; i != m_size; ++i){
			(*this)(i) = other()(i);
		}
		return *this;
	}
	template<class V>
	self_type& plus_assign(vector_expression<V> const& other) {
		SIZE_CHECK(other().size() == size());
		for(std::size_t i = 0; i != m_size; ++i){
			(*this)(i) += other()(i);
		}
		return *this;
	}
	template<class V>
	self_type& minus_assign(vector_expression<V> const& other) {
		SIZE_CHECK(other().size() == size());
		for(std::size_t i = 0; i != m_size; ++i){
			(*this)(i) -= other()(i);
		}
		return *this;
	}
	
	template<class V>
	self_type& operator=(vector_expression<V> const& other) {
		return assign(vector<value_type>(other));
	}
	
	template<class V>
	self_type& operator+=(vector_expression<V> const& other) {
		return plus_assign(vector<value_type>(other));
	}
	template<class V>
	self_type& operator-=(vector_expression<V> const& other) {
		return minus_assign(vector<value_type>(other));
	}
	
	template<class T>
	self_type& operator*=(T const& t) {
		for(std::size_t i = 0; i != m_size; ++i){
			(*this)(i) *= t;
		}
		return *this;
	}
	template<class T>
	self_type& operator/=(T const&  t) {
		for(std::size_t i = 0; i != m_size; ++i){
			(*this)(i) /= t;
		}
		return *this;
	}

	// --------
	// Swapping
	// --------

	/// \brief Swap the content of the self_type with another self_type
	/// \param v is the self_type to be swapped with
	void swap(self_type& v) {
		vector_swap<scalar_swap> (*this, v);
	}

	// --------------
	// ITERATORS
	// --------------
	
	typedef indexed_iterator<self_type, dense_random_access_iterator_tag> iterator;
	typedef indexed_const_iterator<self_type, dense_random_access_iterator_tag> const_iterator;
	
	/// \brief Return a const iterator to the element \e i
	/// \param i index of the element
	const_iterator find(size_type i) const {
		return const_iterator(*this, i);
	}

	/// \brief Return an iterator to the element \e i
	/// \param i index of the element
	iterator find(size_type i) {
		return iterator(*this, i);
	}

	/// \brief return an iterator on the first element of the self_type
	const_iterator begin() const {
		return find(0);
	}

	/// \brief return an iterator after the last element of the self_type
	const_iterator end() const {
		return find(m_size);
	}

	/// \brief Return an iterator on the first element of the self_type
	iterator begin() {
		return find(0);
	}

	/// \brief Return an iterator at the end of the self_type
	iterator end() {
		return find(m_size);
	}

	// Reverse iterator
	typedef reverse_iterator_base<const_iterator> const_reverse_iterator;
	typedef reverse_iterator_base<iterator> reverse_iterator;

	/// \brief Return a const reverse iterator before the first element of the reversed self_type(i.e. end() of normal self_type)
	const_reverse_iterator rbegin() const {
		return const_reverse_iterator(end());
	}

	/// \brief Return a const reverse iterator on the end of the reverse self_type(i.e. first element of the normal self_type) 
	const_reverse_iterator rend() const {
		return const_reverse_iterator(begin());
	}

	/// \brief Return a const reverse iterator before the first element of the reversed self_type(i.e. end() of normal self_type)
	reverse_iterator rbegin() {
		return reverse_iterator(end());
	}

	/// \brief Return a const reverse iterator on the end of the reverse self_type(i.e. first element of the normal self_type) 
	reverse_iterator rend() {
		return reverse_iterator(begin());
	}
private:
	pointer m_data;
	std::size_t m_size;
	std::ptrdiff_t m_stride;
};

template<class ValueType,class Orientation=row_major>
class FixedDenseMatrixProxy: public matrix_expression<FixedDenseMatrixProxy<ValueType,Orientation> > {
	typedef FixedDenseMatrixProxy<ValueType,Orientation> self_type;
public:

	//std::container types
	typedef typename Orientation::orientation_category orientation_category;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename boost::remove_const<ValueType>::type value_type;
	typedef value_type const& const_reference;
	typedef ValueType&  reference;
	typedef ValueType* pointer;
	typedef value_type const* const_pointer;
	//ublas types
	typedef matrix_reference<self_type const> const const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef dense_tag storage_category;
        

	// Construction and destruction

	/// \brief Constructor of a self_type proxy from a Dense MatrixExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
 	template<class E>
	FixedDenseMatrixProxy(matrix_expression<E> const& expression)
	: m_data(traits::matrix_storage(expression()))
	, m_size1(expression().size1())
	, m_size2(expression().size2())
	, m_stride1(traits::matrix_stride1(expression()))
	, m_stride2(traits::matrix_stride2(expression()))
	{
		BOOST_STATIC_ASSERT(traits::IsDense<E>::value);
		BOOST_STATIC_ASSERT((//either same orientation or transposed, not both
			boost::is_same<typename traits::Orientation<E>::type,orientation_category>::value
			^ traits::ExpressionTraits<E>::transposed
		));
	}
	
	/// \brief Constructor of a self_type proxy from a Dense MatrixExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
 	template<class E>
	FixedDenseMatrixProxy(matrix_expression<E>& expression)
	: m_data(traits::matrix_storage(expression()))
	, m_size1(expression().size1())
	, m_size2(expression().size2())
	, m_stride1(traits::matrix_stride1(expression()))
	, m_stride2(traits::matrix_stride2(expression()))
	{
		BOOST_STATIC_ASSERT(traits::IsDense<E>::value);
		BOOST_STATIC_ASSERT((//either same orientation or transposed, not both
			boost::is_same<typename traits::Orientation<E>::type,orientation_category>::value
			^ traits::ExpressionTraits<E>::transposed
		));
	}
		
	/// \brief Constructor of a self_type proxy from a block of memory
	/// \param memory the block of memory used
	/// \param size size of the self_type
 	/// \param stride distance between elements of the self_type in memory
	FixedDenseMatrixProxy(
		pointer data, 
		size_type size1, size_type size2,
		difference_type stride1 = 0,difference_type stride2 = 0 
	)
	: m_data(data)
	, m_size1(size1)
	, m_size2(size2)
	, m_stride1(stride1)
	, m_stride2(stride2)
	{
		if(!m_stride1)
			m_stride1= Orientation::fast_i()?1:m_size2;
		if(!m_stride2)
			m_stride2= Orientation::fast_j()?1:m_size1;
	}
		
	/// \brief Return the number of rows of the matrix
	size_type size1() const {
		return m_size1;
	}
	/// \brief Return the number of columns of the matrix
	size_type size2() const {
		return m_size2;
	}
	
	difference_type stride1()const{
		return m_stride1;
	}
	difference_type stride2()const{
		return m_stride2;
	}
	
	pointer data()const{
		return m_data;
	}
	
	void clear(){
		for(std::size_t i = 0; i != size1(); ++i){
			for(std::size_t j = 0; j != size2(); ++j){
				at_element(i,j) = value_type();
			}
		}
	}


	// ------------------
	// Element assignment
	// ------------------
	
	//~ /// \brief Set element \f$i\f$ to the value \c t
	//~ /// \param i index of the element
	//~ /// \param t reference to the value to be set
	//~ reference insert_element(size_type i, const_reference t) {
		//~ return(*this)[i] = t;
	//~ }

	//~ /// \brief Set element \f$i\f$ to the \e zero value
	//~ /// \param i index of the element
	//~ void erase_element(size_type i) {
		//~ (*this)[i] = value_type/*zero*/();
	//~ }
	// -------
	// ASSIGNING
	// -------

	 // Assignment
	template<class M>
	self_type& assign(matrix_expression<M> const& other) {
		SIZE_CHECK(other().size1() == size1());
		SIZE_CHECK(other().size2() == size2());
		matrix_assign<scalar_assign>(*this,other);
		return *this;
	}
	template<class M>
	self_type& plus_assign(matrix_expression<M> const& other) {
		SIZE_CHECK(other().size1() == size1());
		SIZE_CHECK(other().size2() == size2());
		matrix_assign<scalar_plus_assign>(*this,other);
		return *this;
	}
	template<class M>
	self_type& minus_assign(matrix_expression<M> const& other) {
		SIZE_CHECK(other().size1() == size1());
		SIZE_CHECK(other().size2() == size2());
		matrix_assign<scalar_minus_assign>(*this,other);
		return *this;
	}
	
	template<class M>
	self_type& operator=(matrix_expression<M> const& other) {
		return assign(typename matrix_temporary_traits<M>::type(other));
	}
	 
        self_type &operator = (self_type const&m) {
            return assign(m);
        }
	
	template<class M>
	self_type& operator+=(matrix_expression<M> const& other) {
		return plus_assign(typename matrix_temporary_traits<M>::type(other));
	}
	template<class M>
	self_type& operator-=(matrix_expression<M> const& other) {
		return minus_assign(typename matrix_temporary_traits<M>::type(other));
	}
	
	template<class T>
	self_type& operator*=(T const& t) {
		matrix_assign_scalar<scalar_multiplies_assign> (*this, t);
		return *this;
	}
	template<class T>
	self_type& operator/=(T const&  t) {
		matrix_assign_scalar<scalar_divides_assign> (*this, t);
		return *this;
	}
	
	// --------------
	// Element access
	// --------------
	
	const_reference operator () (size_type i, size_type j) const {
		return m_data[i*m_stride1+j*m_stride2];
        }
	reference at_element (size_type i, size_type j) {
		return m_data[i*m_stride1+j*m_stride2];
        }
        reference operator () (size_type i, size_type j) {
		return at_element(i,j);
        }	
	
        reference insert_element (size_type i, size_type j, value_type t) {
		return (at_element (i, j) = t); 
        }
        void erase_element (size_type i, size_type j) {
		at_element (i, j) = value_type/*zero*/(); 
        }


	// --------
	// Swapping
	// --------

	/// \brief Swap the content of the self_type with another self_type
	/// \param v is the self_type to be swapped with
	void swap(self_type& v) {
		matrix_swap<scalar_swap> (*this, v);
	}

	// --------------
	// ITERATORS
	// --------------

	typedef indexed_iterator1<self_type, dense_random_access_iterator_tag> iterator1;
        typedef indexed_iterator2<self_type, dense_random_access_iterator_tag> iterator2;
        typedef indexed_const_iterator1<self_type, dense_random_access_iterator_tag> const_iterator1;
        typedef indexed_const_iterator2<self_type, dense_random_access_iterator_tag> const_iterator2;
	
        const_iterator1 find1 (int /* rank */, size_type i, size_type j) const {
		return const_iterator1 (*this, i, j);
        }
        iterator1 find1 (int /* rank */, size_type i, size_type j) {
		return iterator1 (*this, i, j);
        }
        const_iterator2 find2 (int /* rank */, size_type i, size_type j) const {
		return const_iterator2 (*this, i, j);
        }
        iterator2 find2 (int /* rank */, size_type i, size_type j) {
		return iterator2 (*this, i, j);
        }
	
	
        const_iterator1 begin1() const{
		return find1(0,0,0);
        }
	const_iterator1 end1() const{
		return find1(0,size1(),0);
        }

        iterator1 begin1(){
		return find1(0,0,0);
        }
	iterator1 end1(){
		return find1(0,size1(),0);
        }

        const_iterator2 begin2() const{
		return find2(0,0,0);
        }
	const_iterator2 end2() const{
		return find2(0,0,size2());
        }

        iterator2 begin2(){
		return find2(0,0,0);
        }
	iterator2 end2(){
		return find2(0,0,size2());
        }

	// Reverse iterators
        typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
        typedef reverse_iterator_base1<iterator1> reverse_iterator1;

        const_reverse_iterator1 rbegin1 () const {
            return const_reverse_iterator1 (end1 ());
        }
        const_reverse_iterator1 rend1 () const {
            return const_reverse_iterator1 (begin1 ());
        }

        reverse_iterator1 rbegin1 () {
            return reverse_iterator1 (end1 ());
        }
        reverse_iterator1 rend1 () {
            return reverse_iterator1 (begin1 ());
        }

        typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;
        typedef reverse_iterator_base2<iterator2> reverse_iterator2;

        const_reverse_iterator2 rbegin2 () const {
            return const_reverse_iterator2 (end2 ());
        }
        const_reverse_iterator2 rend2 () const {
            return const_reverse_iterator2 (begin2 ());
        }

        reverse_iterator2 rbegin2 () {
            return reverse_iterator2 (end2 ());
        }
        reverse_iterator2 rend2 () {
            return reverse_iterator2 (begin2 ());
        }
private:
	pointer m_data;
	size_type m_size1;
	size_type m_size2;
	difference_type m_stride1;
	difference_type m_stride2;
};


template<class ValueType,class IndexType>
class FixedSparseVectorProxy: public vector_expression<FixedSparseVectorProxy<ValueType,IndexType> > {
	typedef FixedSparseVectorProxy<ValueType,IndexType> self_type;
public:

	//std::container types
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename boost::remove_const<ValueType>::type value_type;
	typedef value_type const& const_reference;
	typedef ValueType&  reference;
	typedef ValueType* pointer;
	typedef IndexType const* const_index_pointer;
	typedef value_type const* const_pointer;
	//ublas types
	typedef dense_tag storage_category;
	typedef vector_reference<self_type const> const const_closure_type;
	typedef vector_reference<self_type> closure_type;

	// Construction and destruction

	/// \brief Constructor of a self_type proxy from a Dense VectorExpression
	///
	/// Be aware that the expression must live longer than the proxy!
	/// \param expression Expression from which to construct the Proxy
 	template<class E>
	FixedSparseVectorProxy(vector_expression<E> const& expression):
		m_storage(traits::ExpressionTraits<E const>::compressedStorage(expression())),
		m_size(expression().size())
	{
		BOOST_STATIC_ASSERT(traits::IsSparse<E>::value);
	}
	
	FixedSparseVectorProxy():m_size(0){}
		
	/// \brief Constructor of a vector proxy from a block of memory
	/// \param size the size of the vector represented by the memory
	/// \param data the block of memory used to store the values
	/// \param indizes the block of memory used to store the indizes
	/// \param memoryLength length of the strip of memory
	/// \param startIndex used when the index array does not start from 0, indicates the index of the first element having position 0 in the vector
	FixedSparseVectorProxy(size_type size, const_pointer data, const_index_pointer indizes, size_type memoryLength, size_type startIndex = 0 )
	:m_size(size){
		m_storage.nonZeros=memoryLength;
		m_storage.data = data;
		m_storage.indizes = indizes;
		m_storage.startIndex = startIndex;
	}
	
	FixedSparseVectorProxy& operator=(FixedSparseVectorProxy const& other){
		m_storage = other.m_storage;
		m_size = other.m_size;
		return *this;
	}
	
	/// \brief Return the size of the self_type
	size_type size() const {
		return m_size;
	}
	
	size_type nnz()const{
		return m_storage.nonZeros;
	}
	
	const_pointer data()const{
		return m_storage.data;
	}
	
	const_index_pointer indizes()const{
		return m_storage.indizes;
	}
	
	IndexType startIndex()const{
		return m_storage.startIndex;
	}
	
	traits::CompressedVectorStorage<value_type,IndexType> const& storage()const {
		return m_storage;
	}
	
	// --------------
	// Element access
	// --------------

	/// \brief Return a const pointer to the element \f$i\f$
	/// \param i index of the element
	value_type find_element(size_type i) const {
		SIZE_CHECK(i < m_size);
		i += startIndex();
		const_index_pointer pos = std::lower_bound(indizes(),indizes()+nnz(), i);
		difference_type diff = pos-indizes();
		if(diff == (difference_type) nnz() || *pos != i)
			return value_type();
		return data()[diff];
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	value_type operator()(size_type i) const {
		return find_element(i);
	}

	/// \brief Return a const reference to the element \f$i\f$
	/// \param i index of the element
	value_type operator[](size_type i) const {
		return find_element(i);
	}

	// --------------
	// ITERATORS
	// --------------
	
	 class const_iterator
	: public bidirectional_iterator_base<sparse_bidirectional_iterator_tag,const_iterator, value_type> {
	public:
		typedef typename FixedSparseVectorProxy::value_type value_type;
		typedef typename FixedSparseVectorProxy::difference_type difference_type;
		typedef typename FixedSparseVectorProxy::const_reference reference;
		typedef typename FixedSparseVectorProxy::const_pointer pointer;

		// Construction and destruction

		const_iterator (){}

		const_iterator (const_pointer data, const_index_pointer index, size_type startIndex)
		:m_data(data), m_index(index),m_startIndex(startIndex){}

		// Arithmetic

		const_iterator &operator ++ () {
			++ m_data;
			++ m_index;
			return *this;
		}

		const_iterator &operator -- () {
			-- m_data;
			-- m_index;
			return *this;
		}

		// Dereference

		const_reference operator * () const {
			return *m_data;
		}

		// Index

		size_type index () const {
			return *m_index - m_startIndex;
		}

		// Assignment
		const_iterator &operator = (const const_iterator &it) {
			m_data = it.m_data;
			m_index = it.m_index;
			m_startIndex = it.m_startIndex;
			return *this;
		}

		// Comparison
		bool operator == (const const_iterator &it) const {
			return m_data == it.m_data;
		}

	private:
		const_pointer m_data;
		const_index_pointer m_index;
		size_type m_startIndex;
        };
	
	typedef const_iterator iterator;
	
	/// \brief Return a const iterator to the element \e i
	/// \param i index of the element
	const_iterator find(size_type i) const {
		SIZE_CHECK(i < m_size);
		i += startIndex();
		const_index_pointer pos = std::lower_bound(indizes(),indizes()+nnz(), i);
		difference_type diff = pos-indizes();
		return const_iterator(data()+diff,pos,startIndex());
	}


	/// \brief return an iterator on the first element of the self_type
	const_iterator begin() const {
		return const_iterator(data(),indizes(),startIndex());
	}

	/// \brief return an iterator after the last element of the self_type
	const_iterator end() const {
		return const_iterator(data()+nnz(),indizes()+nnz(),startIndex());
	}

	// Reverse iterator
	typedef reverse_iterator_base<const_iterator> const_reverse_iterator;
	typedef reverse_iterator_base<iterator> reverse_iterator;

	/// \brief Return a const reverse iterator before the first element of the reversed self_type(i.e. end() of normal self_type)
	const_reverse_iterator rbegin() const {
		return const_reverse_iterator(end());
	}

	/// \brief Return a const reverse iterator on the end of the reverse self_type(i.e. first element of the normal self_type) 
	const_reverse_iterator rend() const {
		return const_reverse_iterator(begin());
	}

	/// \brief Return a const reverse iterator before the first element of the reversed self_type(i.e. end() of normal self_type)
	reverse_iterator rbegin() {
		return reverse_iterator(end());
	}

	/// \brief Return a const reverse iterator on the end of the reverse self_type(i.e. first element of the normal self_type) 
	reverse_iterator rend() {
		return reverse_iterator(begin());
	}
private:
	traits::CompressedVectorStorage<value_type,IndexType> m_storage;
	std::size_t m_size;
};

//support for our vector type traits
namespace traits{

template<class T>
struct ExpressionTraitsBase<FixedDenseVectorProxy<T> >{
	typedef FixedDenseVectorProxy<T> type;
	typedef type const const_type;
	typedef typename FixedDenseVectorProxy<T>::pointer value_pointer;
	
	typedef DenseStorage StorageCategory;
	
	static std::size_t stride(const_type& v){
		return v.stride();
	}
};
template<class T>
struct ExpressionTraitsBase<FixedDenseVectorProxy<T> const>{
	typedef FixedDenseVectorProxy<T> const type;
	typedef type const const_type;
	typedef typename FixedDenseVectorProxy<T>::const_pointer value_pointer;
	
	typedef DenseStorage StorageCategory;
	
	static std::size_t stride(const_type& v){
		return v.stride();
	}
};

template<class T,class BaseExpression>
SHARK_DENSETRAITSSPEC(FixedDenseVectorProxy<T>)
	static value_pointer storageBegin(type& v){
		return v.data();
	}
	static value_pointer storageEnd(type& v){
		return v.data();
	}
};

template<class T,class I>
struct ExpressionTraitsBase<FixedSparseVectorProxy<T,I> >{
	typedef FixedSparseVectorProxy<T,I> type;
	typedef type const const_type;
	typedef typename FixedSparseVectorProxy<T,I>::pointer value_pointer;
	
	typedef CompressedStorage StorageCategory;
	
	static std::size_t stride(const_type& v){
		return v.stride();
	}
};
template<class T,class I>
struct ExpressionTraitsBase<FixedSparseVectorProxy<T,I> const >{
	typedef FixedSparseVectorProxy<T,I> const type;
	typedef type const_type;
	typedef typename FixedSparseVectorProxy<T,I>::pointer value_pointer;
	
	typedef CompressedStorage StorageCategory;
	
	static std::size_t stride(const_type& v){
		return 1;
	}
};

template<class T, class I, class BaseExpression>
SHARK_COMPRESSEDTRAITSSPEC(FixedSparseVectorProxy<T BOOST_PP_COMMA() I>)
	typedef CompressedVectorStorage<typename boost::remove_const<T>::type,I > storage;

	static storage compressedStorage(type& v){
		return v.storage();
	}
};

template<class T,class O>
struct ExpressionTraitsBase<FixedDenseMatrixProxy<T,O> const>{
	typedef FixedDenseMatrixProxy<T,O> const type;
	typedef type const_type;
	typedef typename FixedDenseMatrixProxy<T,O>::pointer value_pointer;
	
	typedef typename O::orientation_category orientation;
	typedef DenseStorage StorageCategory;
	static const bool transposed=false;
	
	static std::size_t stride1(const_type& m){
		return m.stride1();
	}
	static std::size_t stride2(const_type& m){
		return m.stride2();
	}
};

template<class T,class O>
struct ExpressionTraitsBase<FixedDenseMatrixProxy<T,O> >{
	typedef FixedDenseMatrixProxy<T,O> type;
	typedef type const const_type;
	typedef typename FixedDenseMatrixProxy<T,O>::pointer value_pointer;
	
	typedef typename O::orientation_category orientation;
	typedef DenseStorage StorageCategory;
	static const bool transposed=false;
	
	static std::size_t stride1(const_type& m){
		return m.stride1();
	}
	static std::size_t stride2(const_type& m){
		return m.stride2();
	}
};

template<class T,class O,class BaseExpression>
SHARK_DENSETRAITSSPEC(FixedDenseMatrixProxy<T BOOST_PP_COMMA() O>)
	static value_pointer storageBegin(type& m){
		return m.data();
	}
	static value_pointer storageEnd(type& m){
		return m.data();
	}
};

}
}
}

#endif
