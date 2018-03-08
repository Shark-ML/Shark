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
#ifndef REMORA_TRIANGULAR_MATRIX_HPP
#define REMORA_TRIANGULAR_MATRIX_HPP


#include "assignment.hpp"
#include "detail/matrix_proxy_classes.hpp"

#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>

namespace remora{

template<class T, class Orientation, class TriangularType>
class triangular_matrix:public matrix_container<triangular_matrix<T,Orientation,TriangularType>, cpu_tag > {
	typedef triangular_matrix<T, Orientation,TriangularType> self_type;
	typedef std::vector<T> array_type;
public:
	typedef typename array_type::value_type value_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef std::size_t size_type;

	typedef const matrix_reference<const self_type> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef packed_matrix_storage<T> storage_type;
	typedef packed_matrix_storage<T const> const_storage_type;
	typedef elementwise<packed_tag> evaluation_category;
	typedef triangular<Orientation,TriangularType> orientation;

	// Construction and destruction

	/// Default triangular_matrix constructor. Make a dense matrix of size (0,0)
	triangular_matrix():m_size(0){}

	/** Packed matrix constructor with defined size
	 * \param size number of rows and columns
	 */
	triangular_matrix(size_type size):m_size(size),m_data(size * (size+1)/2) {}

	/** Packed matrix constructor with defined size and an initial value for all triangular matrix elements
	 * \param size number of rows and columns
	 * \param init initial value of the non-zero elements
	 */
	triangular_matrix(size_type size, value_type init):m_size(size),m_data(size * (size+1)/2,init) {}

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
	size_type size1() const {
		return m_size;
	}
	///\brief Returns the number of columns of the matrix.
	size_type size2() const {
		return m_size;
	}

	storage_type raw_storage(){
		return {m_data.data(), m_data.size()};
	}

	const_storage_type raw_storage()const{
		return {m_data.data(), m_data.size()};
	}
	
	typename device_traits<cpu_tag>::queue_type& queue(){
		return device_traits<cpu_tag>::default_queue();
	}


	// Resizing
	/** Resize a matrix to new dimensions. If resizing is performed, the data is not preserved.
	 * \param size the new number of rows and columns
	 */
	void resize(size_type size) {
		m_data.resize(size*(size+1)/2);
		m_size = size;
	}

	void resize(size_type size1, size_type size2) {
		REMORA_SIZE_CHECK(size1 == size2);
		resize(size1);
		(void)size2;
	}

	void clear(){
		std::fill(m_data.begin(), m_data.end(), value_type/*zero*/());
	}

	// Element access read only
	const_reference operator()(size_type i, size_type j) const {
		REMORA_SIZE_CHECK(i < size1());
		REMORA_SIZE_CHECK(j < size2());
		if(!orientation::non_zero(i,j))
			return value_type();
		REMORA_SIZE_CHECK(orientation::element(i,j,size1(),packed_tag())<m_data.size());
		return m_data [orientation::element(i,j,size1(),packed_tag())];
	}

	// separate write access
	void set_element(size_type i,size_type j, value_type t){
		REMORA_SIZE_CHECK(i < size1());
		REMORA_SIZE_CHECK(j < size2());
		REMORA_SIZE_CHECK(orientation::non_zero(i,j));
		m_data [orientation::element(i,j,size1(),packed_tag())] = t;
	}

	bool non_zero(size_type i,size_type j)const{
		REMORA_SIZE_CHECK(i < size1());
		REMORA_SIZE_CHECK(j < size2());
		return orientation::non_zero(i,j);
	}

	/*! @note "pass by value" the key idea to enable move semantics */
	triangular_matrix& operator = (triangular_matrix m) {
		swap(m);
		return *this;
	}
	template<class C>          // Container assignment without temporary
	triangular_matrix& operator = (matrix_container<C, cpu_tag> const& m) {
		REMORA_SIZE_CHECK(m().size1()==m().size2());
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
	typedef iterators::dense_storage_iterator<value_type,iterators::packed_random_access_iterator_tag> major_iterator;
	typedef iterators::dense_storage_iterator<value_type const,iterators::packed_random_access_iterator_tag> const_major_iterator;

	const_major_iterator major_begin(size_type i) const {
		bool is_row_major = std::is_same<Orientation, row_major>::value;
		size_typ index = (TriangularType::is_upper == is_row_major)?i:0;
		return const_major_iterator(m_data.data() + triangular<row_major,TriangularType>::element(i,index,major_size(*this),packed_tag()),index,1);
	}
	const_major_iterator major_end(size_type i) const {
		bool is_row_major = std::is_same<Orientation, row_major>::value;
		size_type index = (TriangularType::is_upper == is_row_major)? minor_size(): i + 1;
		return const_major_iterator(m_data.data() + triangular<row_major,TriangularType>::element(i, index, major_size(*this),packed_tag()),index,1);
	}
	major_iterator major_begin(size_type i){
		bool is_row_major = std::is_same<Orientation, row_major>::value;
		size_typ index = (TriangularType::is_upper == is_row_major)?i:0;
		return major_iterator(m_data.data() + triangular<row_major,TriangularType>::element(i, index, major_size(*this),packed_tag()),index,1);
	}
	major_iterator major_end(size_type i){
		bool is_row_major = std::is_same<Orientation, row_major>::value;
		size_type index = (TriangularType::is_upper == is_row_major)? minor_size(): i + 1;
		return major_iterator(m_data.data() + triangular<row_major,TriangularType>::element(i, index, major_size(*this),packed_tag()),index,1);
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
	size_type m_size;
	array_type m_data;
};

template<class T, class Orientation, class TriangularType>
struct matrix_temporary_type<T,triangular<Orientation, TriangularType >,packed_tag, cpu_tag> {
	typedef triangular_matrix<T,Orientation, TriangularType> type;
};

}

#endif
