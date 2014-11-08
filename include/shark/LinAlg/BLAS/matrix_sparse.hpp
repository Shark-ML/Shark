#ifndef SHARK_LINALG_BLAS_MATRIX_SPARSE_HPP
#define SHARK_LINALG_BLAS_MATRIX_SPARSE_HPP

#include "matrix_proxy.hpp"

namespace shark {
namespace blas {

template<class T, class I=std::size_t>
class compressed_matrix:public matrix_container<compressed_matrix<T, I> > {
	typedef compressed_matrix<T, I> self_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef value_type scalar_type;
	typedef T const* const_pointer;
	typedef T* pointer;

	typedef I index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type* index_pointer;

	typedef T const &const_reference;
	class reference {
	private:
		const_reference value()const {
			return const_cast<self_type const&>(m_matrix)(m_i,m_j);
		}
		value_type& ref() const {
			//get array bounds
			index_type const *start = &m_matrix.m_indices[m_matrix.m_rowStart[m_i]];
			index_type const *end = &m_matrix.m_indices[m_matrix.m_rowEnd[m_i]];
			//find position of the index in the array
			index_type const *pos = std::lower_bound(start,end,m_j);

			if (pos != end && *pos == m_j)
				return m_matrix.m_values[(pos-start)+m_matrix.m_rowStart[m_i]];
			else {
				//create iterator to the insertion position and insert new element
				row_iterator posIter(
				    m_matrix.values(),
				    m_matrix.inner_indices(),
				    pos-start + m_matrix.m_rowStart[m_i]
				    ,m_i
				);
				return *m_matrix.set_element(posIter, m_j, m_matrix.m_zero);
			}
		}

	public:
		// Construction and destruction
		reference(compressed_matrix &m, size_type i, size_type j):
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
		size_type m_i;
		size_type m_j;
	};

	typedef const matrix_reference<self_type const> const_closure_type;
	typedef matrix_reference<self_type> closure_type;

	typedef sparse_tag storage_category;
	typedef row_major orientation;

	// Construction and destruction
	compressed_matrix()
		: m_size1(0), m_size2(0), m_nnz(0)
		, m_rowStart(1,0), m_indices(0), m_values(0), m_zero(0) {}

	compressed_matrix(size_type size1, size_type size2, size_type non_zeros = 0)
		: m_size1(size1), m_size2(size2), m_nnz(0)
		, m_rowStart(size1 + 1,0)
		, m_rowEnd(size1,0)
		, m_indices(non_zeros), m_values(non_zeros), m_zero(0) {}

	template<class E>
	compressed_matrix(const matrix_expression<E> &e, size_type non_zeros = 0)
		: m_size1(e().size1()), m_size2(e().size2()), m_nnz(0)
		, m_rowStart(e().size1() + 1, 0)
		, m_rowEnd(e().size1(), 0)
		, m_indices(non_zeros), m_values(non_zeros), m_zero(0) {
		kernels::assign(*this, e);
	}

	// Accessors
	size_type size1() const {
		return m_size1;
	}
	size_type size2() const {
		return m_size2;
	}

	size_type nnz_capacity() const {
		return m_values.size();
	}
	size_type row_capacity(std::size_t row)const {
		RANGE_CHECK(row < size1());
		return m_rowStart[row+1] - m_rowStart[row];
	}
	size_type nnz() const {
		return m_nnz;
	}
	size_type inner_nnz(std::size_t row) const {
		return m_rowEnd[row] - m_rowStart[row];
	}

	index_type const *outer_indices() const {
		return &m_rowStart[0];
	}
	index_type const *outer_indices_end() const {
		return &m_rowEnd[0];
	}
	index_type const *inner_indices() const {
		return &m_indices[0];
	}
	value_type const *values() const {
		return &m_values[0];
	}

	index_type *outer_indices() {
		return &m_rowStart[0];
	}
	index_type *outer_indices_end() {
		return &m_rowEnd[0];
	}
	index_type *inner_indices() {
		return &m_indices[0];
	}
	value_type *values() {
		return &m_values[0];
	}

	void set_filled(size_type non_zeros) {
		m_nnz = non_zeros;
	}
	
	void set_row_filled(size_type i,size_type non_zeros) {
		SIZE_CHECK(i < size1());
		SIZE_CHECK(non_zeros <=row_capacity(i));
		
		m_rowEnd[i] = m_rowStart[i]+non_zeros;
		//correct end pointers
		if(i == size1()-1)
			m_rowStart[size1()] = m_rowEnd[i];
	}

	void resize(size_type size1, size_type size2) {
		m_size1 = size1;
		m_size2 = size2;
		m_nnz = 0;
		//clear row data
		m_rowStart.resize(m_size1 + 1);
		m_rowEnd.resize(m_size1);
		std::fill(m_rowStart.begin(),m_rowStart.end(),0);
		std::fill(m_rowEnd.begin(),m_rowEnd.end(),0);
	}
	void reserve(size_type non_zeros) {
		if (non_zeros < nnz_capacity()) return;
		//non_zeros = std::min(m_size2*m_size1,non_zeros);//this can lead to totally strange errors.
		m_indices.resize(non_zeros);
		m_values.resize(non_zeros);
	}

	void reserve_row(std::size_t row, std::size_t non_zeros) {
		RANGE_CHECK(row < size1());
		non_zeros = std::min(m_size2,non_zeros);
		if (non_zeros < row_capacity(row)) return;
		std::size_t spaceDifference = non_zeros-row_capacity(row);

		//check if there is place in the end of the container to store the elements
		if (spaceDifference > nnz_capacity()-m_rowStart.back()) {
			reserve(nnz_capacity()+std::max<std::size_t>(nnz_capacity(),2*spaceDifference));
		}
		//move the elements of the next rows to make room for the reserved space
		for (std::size_t i = size1()-1; i != row; --i) {
			value_type *values = &m_values[m_rowStart[i]];
			value_type *valueRowEnd = &m_values[m_rowEnd[i]];
			index_type *indices = &m_indices[m_rowStart[i]];
			index_type *indicesEnd = &m_indices[m_rowEnd[i]];
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
	const_reference operator()(size_type i, size_type j) const {
		SIZE_CHECK(i < size1());
		SIZE_CHECK(j < size2());
		//get array bounds
		index_type const *start = &m_indices[m_rowStart[i]];
		index_type const *end = &m_indices[m_rowEnd[i]];
		//find position of the index in the array
		index_type const *pos = std::lower_bound(start,end,j);

		if (pos != end && *pos == j)
			return m_values[(pos-start)+m_rowStart[i]];
		else
			return m_zero;
	}

	reference operator()(size_type i, size_type j) {
		SIZE_CHECK(i < size1());
		SIZE_CHECK(j < size2());
		return reference(*this,i,j);
	}

	// Assignment
	template<class C>          // Container assignment without temporary
	compressed_matrix &operator = (const matrix_container<C> &m) {
		resize(m().size1(), m().size2());
		assign(m);
		return *this;
	}
	compressed_matrix &assign_temporary(compressed_matrix &m) {
		swap(m);
		return *this;
	}
	template<class E>
	compressed_matrix &operator = (const matrix_expression<E> &e) {
		self_type temporary(e, nnz_capacity());
		return assign_temporary(temporary);
	}
	template<class E>
	compressed_matrix &assign(const matrix_expression<E> &e) {
		kernels::assign(*this, e);
		return *this;
	}
	template<class E>
	compressed_matrix &operator += (const matrix_expression<E> &e) {
		self_type temporary(*this + e, nnz_capacity());
		return assign_temporary(temporary);
	}
	template<class C>          // Container assignment without temporary
	compressed_matrix &operator += (const matrix_container<C> &m) {
		plus_assign(m);
		return *this;
	}
	template<class E>
	compressed_matrix &plus_assign(const matrix_expression<E> &e) {
		kernels::assign<scalar_plus_assign> (*this, e);
		return *this;
	}
	template<class E>
	compressed_matrix &operator -= (const matrix_expression<E> &e) {
		self_type temporary(*this - e, nnz_capacity());
		return assign_temporary(temporary);
	}
	template<class C>          // Container assignment without temporary
	compressed_matrix &operator -= (const matrix_container<C> &m) {
		minus_assign(m);
		return *this;
	}
	template<class E>
	compressed_matrix &minus_assign(const matrix_expression<E> &e) {
		kernels::assign<scalar_minus_assign> (*this, e);
		return *this;
	}
	compressed_matrix &operator *= (value_type t) {
		kernels::assign<scalar_multiply_assign> (*this, t);
		return *this;
	}
	compressed_matrix &operator /= (value_type t) {
		kernels::assign<scalar_divide_assign> (*this, t);
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

	friend void swap_rows(compressed_matrix& a, size_type i, compressed_matrix& b, size_type j) {
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
		
		index_type* indicesi = &a.m_indices[a.m_rowStart[i]];
		index_type* indicesj = &b.m_indices[b.m_rowStart[j]];
		value_type* valuesi = &a.m_values[a.m_rowStart[i]];
		value_type* valuesj = &b.m_values[b.m_rowStart[j]];
		
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
	
	friend void swap_rows(compressed_matrix& a, size_type i, size_type j) {
		if(i == j) return;
		swap_rows(a,i,a,j);
	}
	
	typedef compressed_storage_iterator<value_type const, index_type const> const_row_iterator;
	typedef compressed_storage_iterator<value_type, index_type const> row_iterator;

	const_row_iterator row_begin(std::size_t i) const {
		SIZE_CHECK(i < size1());
		RANGE_CHECK(m_rowStart[i] <= m_rowEnd[i]);//internal check
		return const_row_iterator(values(), inner_indices(), m_rowStart[i],i);
	}

	const_row_iterator row_end(std::size_t i) const {
		SIZE_CHECK(i < size1());
		RANGE_CHECK(m_rowStart[i] <= m_rowEnd[i]);//internal check
		return const_row_iterator(values(), inner_indices(), m_rowEnd[i],i);
	}

	row_iterator row_begin(std::size_t i) {
		SIZE_CHECK(i < size1());
		RANGE_CHECK(m_rowStart[i] <= m_rowEnd[i]);//internal check
		return row_iterator(values(), inner_indices(), m_rowStart[i],i);
	}

	row_iterator row_end(std::size_t i) {
		SIZE_CHECK(i < size1());
		RANGE_CHECK(m_rowStart[i] <= m_rowEnd[i]);//internal check
		return row_iterator(values(), inner_indices(), m_rowEnd[i],i);
	}
	
	typedef compressed_storage_iterator<value_type const, index_type const> const_column_iterator;
	typedef compressed_storage_iterator<value_type, index_type const> column_iterator;
	
	row_iterator set_element(row_iterator pos, size_type index, value_type value) {
		std::size_t row = pos.row();
		RANGE_CHECK(row < size1());
		RANGE_CHECK(row_end(row) - pos <= row_capacity(row));
		//todo: check in debug, that iterator position is valid

		//shortcut: element already exists.
		if (pos != row_end(row) && pos.index() == index) {
			*pos = value;
			return pos;
		}

		//get position of the element in the array.
		difference_type arrayPos = pos - row_begin(row) + m_rowStart[row];

		//check that there is enough space in the row. this invalidates pos.
		if (row_capacity(row) ==  inner_nnz(row))
			reserve_row(row,std::max<std::size_t>(2*row_capacity(row),1));

		//copy the remaining elements further to make room for the new element
		std::copy_backward(
		    m_values.begin()+arrayPos, m_values.begin() + m_rowEnd[row],
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
		return row_iterator(values(), inner_indices(), arrayPos,row);

	}

	row_iterator clear_range(row_iterator start, row_iterator end) {
		std::size_t row = start.row();
		RANGE_CHECK(row == end.row());
		//get position of the elements in the array.
		size_type rowEndPos = m_rowEnd[row];
		size_type rowStartPos = m_rowStart[row];
		difference_type rangeStartPos = start - row_begin(row)+rowStartPos;
		difference_type rangeEndPos = end - row_begin(row)+rowStartPos;
		difference_type rangeSize = end - start;

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
		return row_iterator(values(), inner_indices(), rangeStartPos,row);
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
	size_type m_size1;
	size_type m_size2;
	size_type m_nnz;
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

}
}

#endif
