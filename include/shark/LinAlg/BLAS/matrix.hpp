#ifndef SHARK_LINALG_BLAS_UBLAS_MATRIX_HPP
#define SHARK_LINALG_BLAS_UBLAS_MATRIX_HPP

#include "matrix_proxy.hpp"
#include "vector_proxy.hpp"
#include "kernels/matrix_assign.hpp"

#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>

namespace shark {
namespace blas {

/** \brief A dense matrix of values of type \c T.
 *
 * For a \f$(m \times n)\f$-dimensional matrix and \f$ 0 \leq i < m, 0 \leq j < n\f$, every element \f$ m_{i,j} \f$ is mapped to
 * the \f$(i.n + j)\f$-th element of the container for row major orientation or the \f$ (i + j.m) \f$-th element of
 * the container for column major orientation. In a dense matrix all elements are represented in memory in a
 * contiguous chunk of memory by definition.
 *
 * Orientation can also be specified, otherwise a \c row_major is used.
 *
 * \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
 * \tparam L the storage organization. It can be either \c row_major or \c column_major. Default is \c row_major
 */
template<class T, class L=row_major>
class matrix:public matrix_container<matrix<T, L> > {
	typedef matrix<T, L> self_type;
	typedef std::vector<T> array_type;
public:
	typedef typename array_type::size_type size_type;
	typedef typename array_type::difference_type difference_type;
	typedef typename array_type::value_type value_type;
	typedef typename array_type::const_reference const_reference;
	typedef typename array_type::reference reference;
	typedef const T* const_pointer;
	typedef T* pointer;

	typedef std::size_t index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type index_pointer;

	typedef const matrix_reference<const self_type> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef dense_tag storage_category;
	typedef L orientation;

	// Construction and destruction

	/// Default dense matrix constructor. Make a dense matrix of size (0,0)
	matrix():m_size1(0), m_size2(0){}

	/** Dense matrix constructor with defined size
	 * \param size1 number of rows
	 * \param size2 number of columns
	 */
	matrix(size_type size1, size_type size2)
		:m_size1(size1), m_size2(size2), m_data(size1 * size2) {}

	/** Dense matrix constructor with defined size a initial value for all the matrix elements
	 * \param size1 number of rows
	 * \param size2 number of columns
	 * \param init initial value assigned to all elements
	 */
	matrix(size_type size1, size_type size2, const value_type& init):
		m_size1(size1), m_size2(size2), m_data(size1 * size2, init) {}

	/** Dense matrix constructor with defined size and an initial data array
	 * \param size1 number of rows
	 * \param size2 number of columns
	 * \param data array to copy into the matrix. Must have the same dimension as the matrix
	 */
	matrix(size_type size1, size_type size2, const array_type& data):
		m_size1(size1), m_size2(size2), m_data(data) {}

	/** Copy-constructor of a dense matrix
	 * \param m is a dense matrix
	 */
	matrix(const matrix& m):
		m_size1(m.m_size1), m_size2(m.m_size2), m_data(m.m_data) {}

	/** Copy-constructor of a dense matrix from a matrix expression
	 * \param e is a matrix expression
	 */
	template<class E>
	matrix(matrix_expression<E> const& e):
		m_size1(e().size1()), m_size2(e().size2()), m_data(m_size1 * m_size2) {
		assign(e);
	}

	// ---------
	// Dense low level interface
	// ---------
	
	///\brief Returns the number of rows of the matrix.
	size_type size1() const {
		return m_size1;
	}
	///\brief Returns the number of columns of the matrix.
	size_type size2() const {
		return m_size2;
	}
	
	///\brief Returns the stride in memory between two rows.
	difference_type stride1()const{
		return orientation::stride1(m_size1,m_size2);
	}
	///\brief Returns the stride in memory between two columns.
	difference_type stride2()const{
		return orientation::stride2(m_size1,m_size2);
	}
	
	///\brief Returns the pointer to the beginning of the matrix storage
	///
	/// Grants low-level access to the matrix internals. Element order depends on whether the matrix is row_major or column_major.
	/// to access element (i,j) use storage()[i*stride1()+j*stride2()].
	const_pointer storage()const{
		return &m_data[0];
	}
	
	///\brief Returns the pointer to the beginning of the matrix storage
	///
	/// Grants low-level access to the matrix internals. Element order depends on whether the matrix is row_major or column_major.
	/// to access element (i,j) use storage()[i*stride1()+j*stride2()].
	pointer storage(){
		return &m_data[0];
	}
	
	// ---------
	// High level interface
	// ---------

	// Resizing
	/** Resize a matrix to new dimensions. If resizing is performed, the data is not preserved.
	 * \param size1 the new number of rows
	 * \param size2 the new number of colums
	 */
	void resize(size_type size1, size_type size2) {
		m_data.resize(size1* size2);
		m_size1 = size1;
		m_size2 = size2;
	}
	
	void clear(){
		std::fill(m_data.begin(), m_data.end(), value_type/*zero*/());
	}

	// Element access
	const_reference operator()(index_type i, index_type j) const {
		return m_data [orientation::element(i, m_size1, j, m_size2)];
	}
	reference operator()(index_type i, index_type j) {
		return m_data [orientation::element(i, m_size1, j, m_size2)];
	}
	
	// Assignment
	
	template<class E>
	matrix& assign(matrix_expression<E> const& e) {
		kernels::assign(*this,e);
		return *this;
	}
	template<class E>
	matrix& plus_assign(matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		kernels::assign<scalar_plus_assign> (*this, e);
		return *this;
	}
	template<class E>
	matrix& minus_assign(matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		kernels::assign<scalar_minus_assign> (*this, e);
		return *this;
	}
	
	template<class E>
	matrix& multiply_assign(matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		kernels::assign<scalar_multiply_assign> (*this, e);
		return *this;
	}
	template<class E>
	matrix& divide_assign(matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		kernels::assign<scalar_divide_assign> (*this, e);
		return *this;
	}
	
	/*! @note "pass by value" the key idea to enable move semantics */
	matrix& operator = (matrix m) {
		swap(m);
		return *this;
	}
	template<class C>          // Container assignment without temporary
	matrix& operator = (const matrix_container<C>& m) {
		resize(m().size1(), m().size2());
		assign(m);
		return *this;
	}
	template<class E>
	matrix& operator = (matrix_expression<E> const& e) {
		self_type temporary(e);
		swap(temporary);
		return *this;
	}
	template<class E>
	matrix& operator += (matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		self_type temporary(*this + e);
		swap(temporary);
		return *this;
	}
	template<class C>          // Container assignment without temporary
	matrix& operator += (const matrix_container<C>& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		return plus_assign(e);
	}
	
	template<class E>
	matrix& operator -= (matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		self_type temporary(*this - e);
		swap(temporary);
		return *this;
	}
	template<class C>          // Container assignment without temporary
	matrix& operator -= (matrix_container<C> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		return minus_assign(e);
	}
	
	template<class E>
	matrix& operator *= (matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		self_type temporary(*this * e);
		swap(temporary);
		return *this;
	}
	template<class C>          // Container assignment without temporary
	matrix& operator *= (const matrix_container<C>& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		return multiply_assign(e);
	}
	
	template<class E>
	matrix& operator /= (matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		self_type temporary(*this / e);
		swap(temporary);
		return *this;
	}
	template<class C>          // Container assignment without temporary
	matrix& operator /= (matrix_container<C> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		return divide_assign(e);
	}
	
	matrix& operator *= (value_type t) {
		kernels::assign<scalar_multiply_assign> (*this, t);
		return *this;
	}
	matrix& operator /= (value_type t) {
		kernels::assign<scalar_divide_assign> (*this, t);
		return *this;
	}

	// Swapping
	void swap(matrix& m) {
		std::swap(m_size1, m.m_size1);
		std::swap(m_size2, m.m_size2);
		m_data.swap(m.m_data);
	}
	friend void swap(matrix& m1, matrix& m2) {
		m1.swap(m2);
	}
	
	friend void swap_rows(matrix& a, index_type i, matrix& b, index_type j){
		SIZE_CHECK(i < a.size1());
		SIZE_CHECK(j < b.size1());
		SIZE_CHECK(a.size2() == b.size2());
		for(std::size_t k = 0; k != a.size2(); ++k){
			std::swap(a(i,k),b(j,k));
		}
	}
	
	friend void swap_rows(matrix& a, index_type i, index_type j) {
		if(i == j) return;
		swap_rows(a,i,a,j);
	}
	
	
	friend void swap_columns(matrix& a, index_type i, matrix& b, index_type j){
		SIZE_CHECK(i < a.size2());
		SIZE_CHECK(j < b.size2());
		SIZE_CHECK(a.size1() == b.size1());
		for(std::size_t k = 0; k != a.size1(); ++k){
			std::swap(a(k,i),b(k,j));
		}
	}
	
	friend void swap_columns(matrix& a, index_type i, index_type j) {
		if(i == j) return;
		swap_columns(a,i,a,j);
	}

	//Iterators
	typedef dense_storage_iterator<value_type> row_iterator;
	typedef dense_storage_iterator<value_type> column_iterator;
	typedef dense_storage_iterator<value_type const> const_row_iterator;
	typedef dense_storage_iterator<value_type const> const_column_iterator;

	const_row_iterator row_begin(index_type i) const {
		return const_row_iterator(&m_data[0] + i*stride1(),0,stride2());
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(&m_data[0] + i*stride1(),size2(),stride2());
	}
	row_iterator row_begin(index_type i){
		return row_iterator(&m_data[0] + i*stride1(),0,stride2());
	}
	row_iterator row_end(index_type i){
		return row_iterator(&m_data[0] + i*stride1(),size2(),stride2());
	}
	
	const_row_iterator column_begin(std::size_t j) const {
		return const_column_iterator(&(*this)(0,0)+j*stride2(),0,stride1());
	}
	const_column_iterator column_end(std::size_t j) const {
		return const_column_iterator(&(*this)(0,0)+j*stride2(),size1(),stride1());
	}
	column_iterator column_begin(std::size_t j){
		return column_iterator(&(*this)(0,0)+j*stride2(),0,stride1());
	}
	column_iterator column_end(std::size_t j){
		return column_iterator(&(*this)(0,0)+j*stride2(),size1(),stride1());
	}
	
	typedef typename blas::major_iterator<self_type>::type major_iterator;
	
	//sparse interface
	major_iterator set_element(major_iterator pos, index_type index, value_type value) {
		RANGE_CHECK(pos.index() == index);
		*pos=value;
		return pos;
	}
	
	major_iterator clear_element(major_iterator elem) {
		*elem = value_type();
		return elem+1;
	}
	
	major_iterator clear_range(major_iterator start, major_iterator end) {
		std::fill(start,end,value_type());
		return end;
	}
	
	void reserve(size_type non_zeros) {}
	
	void reserve_row(std::size_t, std::size_t){}
	void reserve_column(std::size_t, std::size_t){}

	// Serialization
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */) {

		// we need to copy to a collection_size_type to get a portable
		// and efficient boost::serialization
		boost::serialization::collection_size_type s1(m_size1);
		boost::serialization::collection_size_type s2(m_size2);

		// serialize the sizes
		ar& boost::serialization::make_nvp("size1",s1)
		& boost::serialization::make_nvp("size2",s2);

		// copy the values back if loading
		if (Archive::is_loading::value) {
			m_size1 = s1;
			m_size2 = s2;
		}
		ar& boost::serialization::make_nvp("data",m_data);
	}

private:
	size_type m_size1;
	size_type m_size2;
	array_type m_data;
};
template<class T, class L>
struct matrix_temporary_type<T,L,dense_random_access_iterator_tag>{
	typedef matrix<T,L> type;
};

/** \brief An diagonal matrix with values stored inside a diagonal vector
 *
 * the matrix stores a Vector representing the diagonal.
 */
template<class VectorType>
class diagonal_matrix: public matrix_container<diagonal_matrix< VectorType > > {
	typedef diagonal_matrix< VectorType > self_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename VectorType::value_type value_type;
	typedef typename VectorType::const_reference const_reference;
	typedef typename VectorType::reference reference;
	typedef typename VectorType::pointer pointer;
	typedef typename VectorType::const_pointer const_pointer;

	typedef std::size_t index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type index_pointer;

	typedef const matrix_reference<const self_type> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef sparse_tag storage_category;
	typedef unknown_orientation orientation;

	// Construction and destruction
	diagonal_matrix():m_zero(){}
	diagonal_matrix(VectorType const& diagonal):m_zero(),m_diagonal(diagonal){}

	// Accessors
	size_type size1() const {
		return m_diagonal.size();
	}
	size_type size2() const {
		return m_diagonal.size();
	}
	
	// Element access
	const_reference operator()(index_type i, index_type j) const {
		if (i == j)
			return m_diagonal(i);
		else
			return m_zero;
	}

	// Assignment
	diagonal_matrix& operator = (const diagonal_matrix& m) {
		m_diagonal = m.m_diagonal;
		return *this;
	}

	// Swapping
	void swap(diagonal_matrix& m) {
		swap(m_diagonal,m.m_diagonal);
	}
	friend void swap(diagonal_matrix& m1, diagonal_matrix& m2) {
		m1.swap(m2);
	}
	
	//Iterators
	
	class const_row_iterator:public bidirectional_iterator_base<const_row_iterator, value_type> {
	public:
		typedef typename diagonal_matrix::value_type value_type;
		typedef typename diagonal_matrix::difference_type difference_type;
		typedef typename diagonal_matrix::const_reference reference;
		typedef value_type const* pointer;

		// Construction and destruction
		const_row_iterator(){}
		const_row_iterator(index_type index, value_type value, bool isEnd)
			:m_index(index),m_value(value),m_isEnd(isEnd){}

		// Arithmetic
		const_row_iterator& operator ++ () {
			m_isEnd = true;
			return *this;
		}
		const_row_iterator& operator -- () {
			m_isEnd = false;
			return *this;
		}

		// Dereference
		const_reference operator*() const {
			return m_value;
		}

		// Indices
		index_type index() const{
			return m_index;
		}

		// Assignment
		const_row_iterator& operator = (const_row_iterator const& it) {
			m_index = it.m_index;
			return *this;
		}

		// Comparison
		bool operator == (const_row_iterator const& it) const {
			RANGE_CHECK(m_index == it.m_index);
			return m_isEnd == it.m_isEnd;
		}

	private:
		index_type m_index;
		value_type m_value;
		bool m_isEnd;
	};
	typedef const_row_iterator const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	
	const_row_iterator row_begin(index_type i) const {
		return row_iterator(i, m_diagonal(i),false);
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(i, m_zero,true);
	}
	const_column_iterator column_begin(index_type i) const {
		return column_iterator(i, m_diagonal(i),false);
	}
	const_column_iterator column_end(index_type i) const {
		return const_column_iterator(i, m_zero,true);
	}

private:
	value_type const m_zero;
	VectorType m_diagonal; 
};

}
}

#endif
