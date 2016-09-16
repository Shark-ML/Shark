/*!
 * \brief       Sparse matrix class
 * 
 * \author      O. Krause
 * \date        2013
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
#ifndef SHARK_LINALG_BLAS_MATRIX_SPARSE_HPP
#define SHARK_LINALG_BLAS_MATRIX_SPARSE_HPP

#include "assignment.hpp"
#include "detail/matrix_proxy_classes.hpp"

namespace shark {
namespace blas {

template<class T, class I=std::size_t>
class compressed_matrix:public matrix_container<compressed_matrix<T, I>, cpu_tag > {
	typedef compressed_matrix<T, I> self_type;
public:
	typedef I index_type;
	typedef T value_type;
	typedef value_type scalar_type;
	

	typedef T const& const_reference;
	class reference {
	private:
		const_reference value()const {
			return const_cast<self_type const&>(m_matrix)(m_i,m_j);
		}
		value_type& ref() const {
			//get array bounds
			index_type const *start = m_matrix.m_indices.data() + m_matrix.m_rowStart[m_i];
			index_type const *end = m_matrix.m_indices.data() + m_matrix.m_rowEnd[m_i];
			//find position of the index in the array
			index_type const *pos = std::lower_bound(start,end,m_j);

			if (pos != end && *pos == m_j)
				return m_matrix.m_values[(pos-start)+m_matrix.m_rowStart[m_i]];
			else {
				//create iterator to the insertion position and insert new element
				row_iterator posIter(
				    m_matrix.m_values.data(),
				    m_matrix.m_indices.data(),
				    pos-start + m_matrix.m_rowStart[m_i]
				    ,m_i
				);
				return *m_matrix.set_element(posIter, m_j, m_matrix.m_zero);
			}
		}

	public:
		// Construction and destruction
		reference(compressed_matrix &m, index_type i, index_type j):
			m_matrix(m), m_i(i), m_j(j) {}

		// Assignment
		value_type& operator = (value_type d)const {
			return ref() = d;
		}
		value_type& operator=(reference const & other){
			return ref() = other.value();
		}
		value_type& operator += (value_type d)const {
			return ref()+=d;
		}
		value_type& operator -= (value_type d)const {
			return ref()-=d;
		}
		value_type& operator *= (value_type d)const {
			return ref()*=d;
		}
		value_type& operator /= (value_type d)const {
			return ref()/=d;
		}
		
		operator const_reference() const {
			return value();
		}
	private:
		compressed_matrix& m_matrix;
		index_type m_i;
		index_type m_j;
	};

	typedef matrix_reference<self_type const> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef sparse_matrix_storage<T,I> storage_type;
	typedef sparse_matrix_storage<value_type const,index_type const> const_storage_type;
	typedef elementwise_tag evaluation_category;
	typedef row_major orientation;

	// Construction and destruction
	compressed_matrix()
		: m_size1(0), m_size2(0), m_nnz(0)
		, m_rowStart(1,0), m_indices(0), m_values(0), m_zero(0) {}

	compressed_matrix(index_type size1, index_type size2, index_type non_zeros = 0)
		: m_size1(size1), m_size2(size2), m_nnz(0)
		, m_rowStart(size1 + 1,0)
		, m_rowEnd(size1,0)
		, m_indices(non_zeros), m_values(non_zeros), m_zero(0) {}

	template<class E>
	compressed_matrix(matrix_expression<E, cpu_tag> const& e, index_type non_zeros = 0)
		: m_size1(e().size1()), m_size2(e().size2()), m_nnz(0)
		, m_rowStart(e().size1() + 1, 0)
		, m_rowEnd(e().size1(), 0)
		, m_indices(non_zeros), m_values(non_zeros), m_zero(0) {
		assign(*this, e);
	}

	// Accessors
	index_type size1() const {
		return m_size1;
	}
	index_type size2() const {
		return m_size2;
	}

	std::size_t nnz_capacity() const {
		return m_values.size();
	}
	std::size_t row_capacity(index_type row)const {
		RANGE_CHECK(row < size1());
		return m_rowStart[row+1] - m_rowStart[row];
	}
	std::size_t nnz() const {
		return m_nnz;
	}
	std::size_t inner_nnz(index_type row) const {
		return m_rowEnd[row] - m_rowStart[row];
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return {m_values.data(),m_indices.data(), m_rowStart.data(), m_rowEnd.data()};
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return {m_values.data(),m_indices.data(), m_rowStart.data(), m_rowEnd.data()};
	}

	void set_filled(std::size_t non_zeros) {
		m_nnz = non_zeros;
	}
	
	void set_row_filled(index_type i,std::size_t non_zeros) {
		SIZE_CHECK(i < size1());
		SIZE_CHECK(non_zeros <=row_capacity(i));
		
		m_rowEnd[i] = m_rowStart[i]+non_zeros;
		//correct end pointers
		if(i == size1()-1)
			m_rowStart[size1()] = m_rowEnd[i];
	}

	void resize(index_type size1, index_type size2) {
		m_size1 = size1;
		m_size2 = size2;
		m_nnz = 0;
		//clear row data
		m_rowStart.resize(m_size1 + 1);
		m_rowEnd.resize(m_size1);
		std::fill(m_rowStart.begin(),m_rowStart.end(),0);
		std::fill(m_rowEnd.begin(),m_rowEnd.end(),0);
	}
	void reserve(std::size_t non_zeros) {
		if (non_zeros < nnz_capacity()) return;
		//non_zeros = std::min(m_size2*m_size1,non_zeros);//this can lead to totally strange errors.
		m_indices.resize(non_zeros);
		m_values.resize(non_zeros);
	}

	void reserve_row(index_type row, std::size_t non_zeros) {
		RANGE_CHECK(row < size1());
		non_zeros = std::min(m_size2,non_zeros);
		if (non_zeros <= row_capacity(row)) return;
		std::size_t spaceDifference = non_zeros - row_capacity(row);

		//check if there is place in the end of the container to store the elements
		if (spaceDifference > nnz_capacity()-m_rowStart.back()) {
			reserve(nnz_capacity()+std::max<std::size_t>(nnz_capacity(),2*spaceDifference));
		}
		//move the elements of the next rows to make room for the reserved space
		for (index_type i = size1()-1; i != row; --i) {
			value_type* values = m_values.data() + m_rowStart[i];
			value_type* valueRowEnd = m_values.data() + m_rowEnd[i];
			index_type* indices = m_indices.data() + m_rowStart[i];
			index_type* indicesEnd = m_indices.data() + m_rowEnd[i];
			std::copy_backward(values,valueRowEnd, valueRowEnd+spaceDifference);
			std::copy_backward(indices,indicesEnd, indicesEnd+spaceDifference);
			m_rowStart[i]+=spaceDifference;
			m_rowEnd[i]+=spaceDifference;
		}
		m_rowStart.back() +=spaceDifference;
		SIZE_CHECK(row_capacity(row) == non_zeros);
	}

	void clear() {
		m_nnz = 0;
		m_rowStart [0] = 0;
	}

	// Element access
	const_reference operator()(index_type i, index_type j) const {
		SIZE_CHECK(i < size1());
		SIZE_CHECK(j < size2());
		//get array bounds
		index_type const *start = m_indices.data() + m_rowStart[i];
		index_type const *end = m_indices.data() + m_rowEnd[i];
		//find position of the index in the array
		index_type const *pos = std::lower_bound(start,end,j);

		if (pos != end && *pos == j)
			return m_values[(pos-start)+m_rowStart[i]];
		else
			return m_zero;
	}

	reference operator()(index_type i, index_type j) {
		SIZE_CHECK(i < size1());
		SIZE_CHECK(j < size2());
		return reference(*this,i,j);
	}

	// Assignment
	template<class C>          // Container assignment without temporary
	compressed_matrix &operator = (matrix_container<C, cpu_tag> const& m) {
		resize(m().size1(), m().size2());
		assign(*this, m);
		return *this;
	}
	template<class E>
	compressed_matrix &operator = (matrix_expression<E, cpu_tag> const& e) {
		self_type temporary(e, nnz_capacity());
		swap(temporary);
		return *this;
	}

	// Swapping
	void swap(compressed_matrix &m) {
		std::swap(m_size1, m.m_size1);
		std::swap(m_size2, m.m_size2);
		std::swap(m_nnz, m.m_nnz);
		m_rowStart.swap(m.m_rowStart);
		m_rowEnd.swap(m.m_rowEnd);
		m_indices.swap(m.m_indices);
		m_values.swap(m.m_values);
	}

	friend void swap(compressed_matrix &m1, compressed_matrix &m2) {
		m1.swap(m2);
	}

	friend void swap_rows(compressed_matrix& a, index_type i, compressed_matrix& b, index_type j) {
		SIZE_CHECK(i < a.size1());
		SIZE_CHECK(j < b.size1());
		SIZE_CHECK(a.size2() == b.size2());
		
		//rearrange (i,j) such that i has equal or more elements than j
		if(a.inner_nnz(i) < b.inner_nnz(j)){
			swap_rows(b,j,a,i);
			return;
		}
		
		std::size_t nnzi = a.inner_nnz(i);
		std::size_t nnzj = b.inner_nnz(j);
		
		//reserve enough space for swapping
		b.reserve_row(j,nnzi);
		SIZE_CHECK(b.row_capacity(j) >= nnzi);
		SIZE_CHECK(a.row_capacity(i) >= nnzj);
		
		index_type* indicesi = a.m_indices.data() + a.m_rowStart[i];
		index_type* indicesj = b.m_indices.data() + b.m_rowStart[j];
		value_type* valuesi = a.m_values.data() + a.m_rowStart[i];
		value_type* valuesj = b.m_values.data() + b.m_rowStart[j];
		
		//swap all elements of j with the elements in i, don't care about unitialized elements in j
		std::swap_ranges(indicesi,indicesi+nnzi,indicesj);
		std::swap_ranges(valuesi, valuesi+nnzi,valuesj);
		
		//if both rows had the same number of elements, we are done.
		if(nnzi == nnzj)
			return;
		
		//otherwise correct end pointers
		a.set_row_filled(i,nnzj);
		b.set_row_filled(j,nnzi);
	}
	
	friend void swap_rows(compressed_matrix& a, index_type i, index_type j) {
		if(i == j) return;
		swap_rows(a,i,a,j);
	}
	
	typedef compressed_storage_iterator<value_type const, index_type const> const_row_iterator;
	typedef compressed_storage_iterator<value_type, index_type const> row_iterator;

	const_row_iterator row_begin(index_type i) const {
		SIZE_CHECK(i < size1());
		RANGE_CHECK(m_rowStart[i] <= m_rowEnd[i]);//internal check
		return const_row_iterator(m_values.data(), m_indices.data(), m_rowStart[i],i);
	}

	const_row_iterator row_end(index_type i) const {
		SIZE_CHECK(i < size1());
		RANGE_CHECK(m_rowStart[i] <= m_rowEnd[i]);//internal check
		return const_row_iterator(m_values.data(), m_indices.data(), m_rowEnd[i],i);
	}

	row_iterator row_begin(index_type i) {
		SIZE_CHECK(i < size1());
		RANGE_CHECK(m_rowStart[i] <= m_rowEnd[i]);//internal check
		return row_iterator(m_values.data(), m_indices.data(), m_rowStart[i],i);
	}

	row_iterator row_end(index_type i) {
		SIZE_CHECK(i < size1());
		RANGE_CHECK(m_rowStart[i] <= m_rowEnd[i]);//internal check
		return row_iterator(m_values.data(), m_indices.data(), m_rowEnd[i],i);
	}
	
	typedef compressed_storage_iterator<value_type const, index_type const> const_column_iterator;
	typedef compressed_storage_iterator<value_type, index_type const> column_iterator;
	
	row_iterator set_element(row_iterator pos, index_type index, value_type value) {
		std::size_t row = pos.row();
		RANGE_CHECK(row < size1());
		RANGE_CHECK(index_type(row_end(row) - pos) <= row_capacity(row));
		//todo: check in debug, that iterator position is valid

		//shortcut: element already exists.
		if (pos != row_end(row) && pos.index() == index) {
			*pos = value;
			return pos;
		}

		//get position of the element in the array.
		std::ptrdiff_t arrayPos = (pos - row_begin(row)) + m_rowStart[row];

		//check that there is enough space in the row. this invalidates pos.
		if (row_capacity(row) ==  inner_nnz(row))
			reserve_row(row,std::max<std::size_t>(2*row_capacity(row),1));

		//copy the remaining elements further to make room for the new element
		std::copy_backward(
		    m_values.begin() + arrayPos, m_values.begin() + m_rowEnd[row],
		    m_values.begin() + m_rowEnd[row] + 1
		);
		std::copy_backward(
		    m_indices.begin()+arrayPos, m_indices.begin() + m_rowEnd[row],
		    m_indices.begin() + m_rowEnd[row] + 1
		);
		//insert new element
		m_values[arrayPos] = value;
		m_indices[arrayPos] = index;
		++m_rowEnd[row];
		++m_nnz;

		//return new iterator to the inserted element.
		return row_iterator(m_values.data(), m_indices.data(), arrayPos,row);

	}

	row_iterator clear_range(row_iterator start, row_iterator end) {
		std::size_t row = start.row();
		RANGE_CHECK(row == end.row());
		//get position of the elements in the array.
		index_type rowEndPos = m_rowEnd[row];
		index_type rowStartPos = m_rowStart[row];
		index_type rangeStartPos = start - row_begin(row)+rowStartPos;
		index_type rangeEndPos = end - row_begin(row)+rowStartPos;
		std::ptrdiff_t rangeSize = end - start;

		//remove the elements in the range
		std::copy(
		    m_values.begin()+rangeEndPos,m_values.begin() + rowEndPos, m_values.begin() + rangeStartPos
		);
		std::copy(
		    m_indices.begin()+rangeEndPos,m_indices.begin() + rowEndPos , m_indices.begin() + rangeStartPos
		);
		m_rowEnd[row] -= rangeSize;
		m_nnz -= rangeSize;
		//return new iterator to the next element
		return row_iterator(m_values.data(), m_indices.data(), rangeStartPos,row);
	}

	row_iterator clear_element(row_iterator elem) {
		RANGE_CHECK(elem != row_end());
		row_iterator next = elem;
		++next;
		clear_range(elem,next);
	}

	// Serialization
	template<class Archive>
	void serialize(Archive &ar, const unsigned int /* file_version */) {
		ar &boost::serialization::make_nvp("outer_indices", m_rowStart);
		ar &boost::serialization::make_nvp("outer_indices_end", m_rowEnd);
		ar &boost::serialization::make_nvp("inner_indices", m_indices);
		ar &boost::serialization::make_nvp("values", m_values);
	}

private:
	index_type m_size1;
	index_type m_size2;
	index_type m_nnz;
	std::vector<index_type> m_rowStart;
	std::vector<index_type> m_rowEnd;
	std::vector<index_type> m_indices;
	std::vector<value_type> m_values;
	value_type m_zero;
};

template<class T>
struct matrix_temporary_type<T,row_major,sparse_bidirectional_iterator_tag> {
	typedef compressed_matrix<T> type;
};

template<class T>
struct matrix_temporary_type<T,unknown_orientation,sparse_bidirectional_iterator_tag> {
	typedef compressed_matrix<T> type;
};

template<class T, class I>
struct const_expression<compressed_matrix<T,I> >{
	typedef compressed_matrix<T,I> const type;
};
template<class T, class I>
struct const_expression<compressed_matrix<T,I> const>{
	typedef compressed_matrix<T,I> const type;
};

}
}

#endif
