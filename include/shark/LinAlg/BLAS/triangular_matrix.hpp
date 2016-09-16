/*!
 * \brief       Implements a matrix with triangular storage layout
 * 
 * \author      O. Krause
 * \date        2015
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
#ifndef SHARK_LINALG_BLAS_TRIANGULAR_MATRIX_HPP
#define SHARK_LINALG_BLAS_TRIANGULAR_MATRIX_HPP


#include "assignment.hpp"
#include "detail/matrix_proxy_classes.hpp"

#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>

namespace shark {
namespace blas {

template<class T, class Orientation, class TriangularType>
class triangular_matrix:public matrix_container<triangular_matrix<T,Orientation,TriangularType>, cpu_tag > {
	typedef triangular_matrix<T, Orientation,TriangularType> self_type;
	typedef std::vector<T> array_type;
public:
	typedef typename array_type::value_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef std::size_t index_type;

	typedef const matrix_reference<const self_type> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef packed_matrix_storage<T> storage_type;
	typedef packed_matrix_storage<T const> const_storage_type;
	typedef elementwise_tag evaluation_category;
	typedef triangular<Orientation,TriangularType> orientation;

	// Construction and destruction

	/// Default triangular_matrix constructor. Make a dense matrix of size (0,0)
	triangular_matrix():m_size(0){}

	/** Packed matrix constructor with defined size
	 * \param size number of rows and columns
	 */
	triangular_matrix(index_type size):m_size(size),m_data(size * (size+1)/2) {}

	/** Packed matrix constructor with defined size and an initial value for all triangular matrix elements
	 * \param size number of rows and columns
	 * \param init initial value of the non-zero elements
	 */
	triangular_matrix(index_type size, scalar_type init):m_size(size),m_data(size * (size+1)/2,init) {}

	/** Copy-constructor of a dense matrix
	 * \param m is a dense matrix
	 */
	triangular_matrix(triangular_matrix const& m):m_size(m.m_size), m_data(m.m_data) {}

	/** Copy-constructor of a dense matrix from a matrix expression
	 * \param e is a matrix expression which has to be triangular
	 */
	template<class E>
	triangular_matrix(matrix_expression<E, cpu_tag> const& e)
		:m_size(e().size1()), m_data(m_size * (m_size+1)/2)
	{
		assign(*this, e);
	}
	
	///\brief Returns the number of rows of the matrix.
	index_type size1() const {
		return m_size;
	}
	///\brief Returns the number of columns of the matrix.
	index_type size2() const {
		return m_size;
	}
	
	storage_type raw_storage(){
		return {m_data.data(), m_data.size()};
	}
	
	const_storage_type raw_storage()const{
		return {m_data.data(), m_data.size()};
	}


	// Resizing
	/** Resize a matrix to new dimensions. If resizing is performed, the data is not preserved.
	 * \param size the new number of rows and columns
	 */
	void resize(index_type size) {
		m_data.resize(size*(size+1)/2);
		m_size = size;
	}
	
	void resize(index_type size1, index_type size2) {
		SIZE_CHECK(size1 == size2);
		resize(size1);
		(void)size2;
	}
	
	void clear(){
		std::fill(m_data.begin(), m_data.end(), value_type/*zero*/());
	}

	// Element access read only
	const_reference operator()(index_type i, index_type j) const {
		SIZE_CHECK(i < size1());
		SIZE_CHECK(j < size2());
		if(!orientation::non_zero(i,j)) 
			return value_type();
		SIZE_CHECK(orientation::element(i,j,size1(),packed_tag())<m_data.size());
		return m_data [orientation::element(i,j,size1(),packed_tag())];
	}
	
	// separate write access
	void set_element(index_type i,index_type j, value_type t){
		SIZE_CHECK(i < size1());
		SIZE_CHECK(j < size2());
		SIZE_CHECK(orientation::non_zero(i,j));
		m_data [orientation::element(i,j,size1(),packed_tag())] = t;
	}
	
	bool non_zero(index_type i,index_type j)const{
		SIZE_CHECK(i < size1());
		SIZE_CHECK(j < size2());
		return orientation::non_zero(i,j);
	}
	
	/*! @note "pass by value" the key idea to enable move semantics */
	triangular_matrix& operator = (triangular_matrix m) {
		swap(m);
		return *this;
	}
	template<class C>          // Container assignment without temporary
	triangular_matrix& operator = (matrix_container<C, cpu_tag> const& m) {
		SIZE_CHECK(m().size1()==m().size2());
		resize(m().size1());
		assign(*this, m);
		return *this;
	}
	template<class E>
	triangular_matrix& operator = (matrix_expression<E, cpu_tag> const& e) {
		self_type temporary(e);
		swap(temporary);
		return *this;
	}

	// Swapping
	void swap(triangular_matrix& m) {
		std::swap(m_size, m.m_size);
		m_data.swap(m.m_data);
	}
	friend void swap(triangular_matrix& m1, triangular_matrix& m2) {
		m1.swap(m2);
	}
	
	///////////iterators
	template<class TIter>
	class major1_iterator:
	public random_access_iterator_base<
		major1_iterator<TIter>,
		typename boost::remove_const<T>::type,
		packed_random_access_iterator_tag
	>{
	private:
		index_type offset(index_type n)const{
			index_type k = m_index;
			if(n >= 0){
				return (n*(2*k+n+1))/2;
			}else{
				k+= n;
				n*=-1;
				return -(n*(2*k+n+1))/2;
			}
		}
	public:
		typedef typename boost::remove_const<TIter>::type value_type;
		typedef TIter& reference;
		typedef TIter* pointer;

		// Construction
		major1_iterator() {}
		major1_iterator(pointer arrayBegin, index_type index, index_type /*size*/)
		:m_pos(arrayBegin), m_index(index){}
		
		template<class U>
		major1_iterator(major1_iterator<U> const& iter)
		:m_pos(iter.m_pos), m_index(iter.m_index){}
			
		template<class U>
		major1_iterator& operator=(major1_iterator<U> const& iter){
			m_pos = iter.m_pos;
			m_index = iter.m_index;
			return *this;
		}

		// Arithmetic
		major1_iterator& operator ++ () {
			++m_index;
			m_pos += m_index;
			return *this;
		}
		major1_iterator& operator -- () {
			m_pos -= m_index;
			--m_index;
			return *this;
		}
		major1_iterator& operator += (index_type n) {
			m_pos += offset(n);
			m_index += n;
			return *this;
		}
		major1_iterator& operator -= (index_type n) {
			m_pos += offset(-n);
			m_index -= n;
			return *this;
		}
		template<class U>
		index_type operator - (major1_iterator<U> const& it) const {
			return m_index - it.m_index;
		}

		// Dereference
		reference operator*() const {
			return *m_pos;
		}
		reference operator [](index_type n) const {
			return m_pos[offset(n)];
		}

		// Index
		index_type index() const {
			return m_index;
		}

		// Comparison
		template<class U>
		bool operator == (major1_iterator<U> const& it) const {
			return m_index == it.m_index;
		}
		template<class U>
		bool operator <  (major1_iterator<U> const& it) const {
			return m_index < it.m_index;
		}

	private:
		pointer m_pos;
		index_type m_index;
		template<class> friend class major1_iterator;
	};
	
	template<class TIter>
	class major2_iterator:
	public random_access_iterator_base<
		major2_iterator<TIter>,
		typename boost::remove_const<T>::type,
		packed_random_access_iterator_tag
	>{
	private:
		index_type offset(index_type n)const{
			index_type k = m_size-m_index-1;
			if(n >= 0){
				return (2*k*n-n*n+n)/2;
			}else{
				n*=-1;
				k+= n;
				return -(2*k*n-n*n+n)/2;
			}
		}
	public:
		typedef typename boost::remove_const<TIter>::type value_type;
		typedef TIter& reference;
		typedef TIter* pointer;

		// Construction
		major2_iterator() {}
		major2_iterator(pointer arrayBegin, index_type index, index_type size)
		:m_pos(arrayBegin), m_index(index), m_size(size){}
		
		template<class U>
		major2_iterator(major2_iterator<U> const& iter)
		:m_pos(iter.m_pos), m_index(iter.m_index), m_size(iter.m_size){}
			
		template<class U>
		major2_iterator& operator=(major2_iterator<U> const& iter){
			m_pos = iter.m_pos;
			m_index = iter.m_index;
			m_size = iter.m_size;
			return *this;
		}

		// Arithmetic
		major2_iterator& operator ++ () {
			++m_index;
			m_pos += m_size-m_index;
			return *this;
		}
		major2_iterator& operator -- () {
			m_pos -= m_size-m_index;
			--m_index;
			return *this;
		}
		major2_iterator& operator += (index_type n) {
			m_pos += offset(n);
			m_index += n;
			return *this;
		}
		major2_iterator& operator -= (index_type n) {
			m_pos += offset(-n);
			m_index -= n;
			return *this;
		}
		template<class U>
		std::ptrdiff_t operator - (major2_iterator<U> const& it) const {
			return static_cast<std::ptrdiff_t>(m_index) - static_cast<std::ptrdiff_t>(it.m_index);
		}

		// Dereference
		reference operator*() const {
			return *m_pos;
		}
		reference operator [](index_type n) const {
			return m_pos[offset(n)];
		}

		// Index
		index_type index() const {
			return m_index;
		}

		// Comparison
		template<class U>
		bool operator == (major2_iterator<U> const& it) const {
			return m_index == it.m_index;
		}
		template<class U>
		bool operator <  (major2_iterator<U> const& it) const {
			return m_index < it.m_index;
		}

	private:
		pointer m_pos;
		index_type m_index;
		index_type m_size;
		template<class> friend class major2_iterator;
	};
	
	typedef typename boost::mpl::if_<
		boost::is_same<Orientation,row_major>,
		dense_storage_iterator<value_type,packed_random_access_iterator_tag>,
		typename boost::mpl::if_c<
			TriangularType::is_upper,
			major1_iterator<value_type>,
			major2_iterator<value_type>
		>::type
	>::type row_iterator;
	typedef typename boost::mpl::if_<
		boost::is_same<Orientation,row_major>,
		typename boost::mpl::if_c<
			TriangularType::is_upper,
			major2_iterator<value_type>,
			major1_iterator<value_type>
		>::type,
		dense_storage_iterator<value_type,packed_random_access_iterator_tag>
	>::type column_iterator;
	
	typedef typename boost::mpl::if_<
		boost::is_same<Orientation,row_major>,
		dense_storage_iterator<value_type const,packed_random_access_iterator_tag>,
		typename boost::mpl::if_c<
			TriangularType::is_upper,
			major1_iterator<value_type const>,
			major2_iterator<value_type const>
		>::type
	>::type const_row_iterator;
	typedef typename boost::mpl::if_<
		boost::is_same<Orientation,row_major>,
		typename boost::mpl::if_c<
			TriangularType::is_upper,
			major2_iterator<value_type const>,
			major1_iterator<value_type const>
		>::type,
		dense_storage_iterator<value_type const,packed_random_access_iterator_tag>
	>::type const_column_iterator;
	
public:

	const_row_iterator row_begin(index_type i) const {
		index_type index = TriangularType::is_upper?i:0;
		return const_row_iterator(
			m_data.data()+orientation::element(i,index,size1(),packed_tag())
			,index
			,orientation::orientation::stride2(size1(),size2())//1 if row_major, size2() otherwise
		);
	}
	const_row_iterator row_end(index_type i) const {
		index_type index = TriangularType::is_upper?size2():i+1;
		return const_row_iterator(
			m_data.data() + orientation::element(i, index, size1(),packed_tag())
			,index
			,orientation::orientation::stride2(size1(),size2())//1 if row_major, size2() otherwise
		);
	}
	row_iterator row_begin(index_type i){
		index_type index = TriangularType::is_upper?i:0;
		return row_iterator(
			m_data.data() + orientation::element(i, index, size1(),packed_tag())
			,index
			,orientation::orientation::stride2(size1(),size2())//1 if row_major, size2() otherwise
		);
	}
	row_iterator row_end(index_type i){
		index_type index = TriangularType::is_upper?size2():i+1;
		return row_iterator(
			m_data.data() + orientation::element(i, index, size1(),packed_tag())
			,index
			,orientation::orientation::stride2(size1(),size2())//1 if row_major, size2() otherwise
		);
	}
	
	const_column_iterator column_begin(index_type i) const {
		index_type index = TriangularType::is_upper?0:i;
		return const_column_iterator(
			m_data.data() + orientation::element(index, i, size1(),packed_tag())
			,index
			,orientation::orientation::stride1(size1(),size2())//size1() if row_major, 1 otherwise
		);
	}
	const_column_iterator column_end(index_type i) const {
		index_type index = TriangularType::is_upper?i+1:size2();
		return const_column_iterator(
			m_data.data() + orientation::element(index, i, size1(),packed_tag())
			,index
			,orientation::orientation::stride1(size1(),size2())//size1() if row_major, 1 otherwise
		);
	}
	column_iterator column_begin(index_type i){
		index_type index = TriangularType::is_upper?0:i;
		return column_iterator(
			m_data.data() + orientation::element(index, i, size1(),packed_tag())
			,index
			,orientation::orientation::stride1(size1(),size2())//size1() if row_major, 1 otherwise
		);
	}
	column_iterator column_end(index_type i){
		index_type index = TriangularType::is_upper?i+1:size2();
		return column_iterator(
			m_data.data() + orientation::element(index, i, size1(),packed_tag())
			,index
			,orientation::orientation::stride1(size1(),size2())//size1() if row_major, 1 otherwise
		);
	}

	// Serialization
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */) {
		boost::serialization::collection_size_type s(m_size);

		// serialize the sizes
		ar & boost::serialization::make_nvp("size",s);

		// copy the values back if loading
		if (Archive::is_loading::value) {
			m_size = s;
		}
		ar & boost::serialization::make_nvp("data",m_data);
	}

private:
	index_type m_size;
	array_type m_data;
};

template<class T, class Orientation, class TriangularType>
struct const_expression<triangular_matrix<T,Orientation, TriangularType> >{
	typedef triangular_matrix<T,Orientation, TriangularType> const type;
};
template<class T, class Orientation, class TriangularType>
struct const_expression<triangular_matrix<T,Orientation, TriangularType> const>{
	typedef triangular_matrix<T,Orientation, TriangularType> const type;
};

}
}

#endif
