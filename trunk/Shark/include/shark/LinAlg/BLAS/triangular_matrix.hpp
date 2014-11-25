#ifndef SHARK_LINALG_BLAS_TRIANGULAR_MATRIX_HPP
#define SHARK_LINALG_BLAS_TRIANGULAR_MATRIX_HPP

#include "matrix_proxy.hpp"
#include "vector_proxy.hpp"
#include "kernels/matrix_assign.hpp"

#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>

namespace shark {
namespace blas {

template<class T, class Orientation, class TriangularType>
class triangular_matrix:public matrix_container<triangular_matrix<T,Orientation,TriangularType> > {
	typedef triangular_matrix<T, Orientation,TriangularType> self_type;
	typedef std::vector<T> array_type;
public:
	typedef typename array_type::size_type size_type;
	typedef typename array_type::difference_type difference_type;
	typedef typename array_type::value_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef const T* const_pointer;
	typedef T* pointer;

	typedef std::size_t index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type index_pointer;

	typedef const matrix_reference<const self_type> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef packed_tag storage_category;
	typedef packed<Orientation,TriangularType> orientation;

	// Construction and destruction

	/// Default triangular_matrix constructor. Make a dense matrix of size (0,0)
	triangular_matrix():m_size(0){}

	/** Packed matrix constructor with defined size
	 * \param size number of rows and columns
	 */
	triangular_matrix(size_type size):m_size(size),m_data(size * (size+1)/2) {}

	/** Packed matrix constructor with defined size and an initial value for all triangular matrix elements
	 * \param size number of rows and columns
	 */
	triangular_matrix(size_type size, scalar_type init):m_size(size),m_data(size * (size+1)/2,init) {}

	/** Copy-constructor of a dense matrix
	 * \param m is a dense matrix
	 */
	triangular_matrix(const triangular_matrix& m):m_size(m.m_size), m_data(m.m_data) {}

	/** Copy-constructor of a dense matrix from a matrix expression
	 * \param e is a matrix expression which has to be triangular
	 */
	template<class E>
	triangular_matrix(matrix_expression<E> const& e)
		:m_size(e().size1()), m_data(m_size * (m_size+1)/2)
	{
		assign(e);
	}

	// ---------
	// Low level interface
	// ---------
	
	///\brief Returns the number of rows of the matrix.
	size_type size1() const {
		return m_size;
	}
	///\brief Returns the number of columns of the matrix.
	size_type size2() const {
		return m_size;
	}
	
	///\brief Returns the pointer to the beginning of the matrix storage
	///
	/// Grants low-level access to the matrix internals. Element order depends on whether the matrix is
	///  row_major or column_major and upper or lower triangular
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
	
	///\brief Number of nonzero-elements stores in the matrix.
	size_type nnz() const {
		return m_data.size();
	}
	
	// ---------
	// High level interface
	// ---------

	// Resizing
	/** Resize a matrix to new dimensions. If resizing is performed, the data is not preserved.
	 * \param size the new number of rows and columns
	 */
	void resize(size_type size) {
		m_data.resize(size*(size+1)/2);
		m_size = size;
	}
	
	void clear(){
		std::fill(m_data.begin(), m_data.end(), value_type/*zero*/());
	}

	// Element access read only
	const_reference operator()(index_type i, index_type j) const {
		if(!orientation::non_zero(i,j)) 
			return value_type();
		return m_data [orientation::element(i,j,size1())];
	}
	
	// separate write access
	void set_element(std::size_t i,std::size_t j, value_type t){
		SIZE_CHECK(orientation::non_zero(i,j));
		m_data [orientation::element(i,j,size1())] = t;
	}
	
	bool non_zero(std::size_t i,std::size_t j)const{
		return orientation::non_zero(i,j);
	}
	
	// Assignment
	template<class E>
	triangular_matrix& assign(matrix_expression<E> const& e) {
		kernels::assign(*this,e);
		return *this;
	}
	template<class E>
	triangular_matrix& plus_assign(matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		kernels::assign<scalar_plus_assign> (*this, e);
		return *this;
	}
	template<class E>
	triangular_matrix& minus_assign(matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		kernels::assign<scalar_minus_assign> (*this, e);
		return *this;
	}
	
	template<class E>
	triangular_matrix& multiply_assign(matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		kernels::assign<scalar_multiply_assign> (*this, e);
		return *this;
	}
	template<class E>
	triangular_matrix& divide_assign(matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		kernels::assign<scalar_divide_assign> (*this, e);
		return *this;
	}
	
	/*! @note "pass by value" the key idea to enable move semantics */
	triangular_matrix& operator = (triangular_matrix m) {
		swap(m);
		return *this;
	}
	template<class C>          // Container assignment without temporary
	triangular_matrix& operator = (const matrix_container<C>& m) {
		SIZE_CHECK(m().size1()==m().size2());
		resize(m().size1());
		assign(m);
		return *this;
	}
	template<class E>
	triangular_matrix& operator = (matrix_expression<E> const& e) {
		self_type temporary(e);
		swap(temporary);
		return *this;
	}
	template<class E>
	triangular_matrix& operator += (matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		self_type temporary(*this + e);
		swap(temporary);
		return *this;
	}
	template<class C>          // Container assignment without temporary
	triangular_matrix& operator += (const matrix_container<C>& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		return plus_assign(e);
	}
	
	template<class E>
	triangular_matrix& operator -= (matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		self_type temporary(*this - e);
		swap(temporary);
		return *this;
	}
	template<class C>          // Container assignment without temporary
	triangular_matrix& operator -= (matrix_container<C> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		return minus_assign(e);
	}
	
	template<class E>
	triangular_matrix& operator *= (matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		self_type temporary(*this * e);
		swap(temporary);
		return *this;
	}
	template<class C>          // Container assignment without temporary
	triangular_matrix& operator *= (const matrix_container<C>& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		return multiply_assign(e);
	}
	
	template<class E>
	triangular_matrix& operator /= (matrix_expression<E> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		self_type temporary(*this / e);
		swap(temporary);
		return *this;
	}
	template<class C>          // Container assignment without temporary
	triangular_matrix& operator /= (matrix_container<C> const& e) {
		SIZE_CHECK(size1() == e().size1());
		SIZE_CHECK(size2() == e().size2());
		return divide_assign(e);
	}
	
	triangular_matrix& operator *= (scalar_type t) {
		kernels::assign<scalar_multiply_assign> (*this, t);
		return *this;
	}
	triangular_matrix& operator /= (scalar_type t) {
		kernels::assign<scalar_divide_assign> (*this, t);
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
		difference_type offset(difference_type n)const{
			difference_type k = m_index;
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
		major1_iterator(pointer arrayBegin, size_type index, difference_type /*size*/)
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
		major1_iterator& operator += (difference_type n) {
			m_pos += offset(n);
			m_index += n;
			return *this;
		}
		major1_iterator& operator -= (difference_type n) {
			m_pos += offset(-n);
			m_index -= n;
			return *this;
		}
		template<class U>
		difference_type operator - (major1_iterator<U> const& it) const {
			return m_index - it.m_index;
		}

		// Dereference
		reference operator*() const {
			return *m_pos;
		}
		reference operator [](difference_type n) const {
			return m_pos[offset(n)];
		}

		// Index
		size_type index() const {
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
		difference_type m_index;
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
		difference_type offset(difference_type n)const{
			difference_type k = m_size-m_index-1;
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
		major2_iterator(pointer arrayBegin, size_type index, difference_type size)
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
		major2_iterator& operator += (difference_type n) {
			m_pos += offset(n);
			m_index += n;
			return *this;
		}
		major2_iterator& operator -= (difference_type n) {
			m_pos += offset(-n);
			m_index -= n;
			return *this;
		}
		template<class U>
		difference_type operator - (major2_iterator<U> const& it) const {
			return m_index - it.m_index;
		}

		// Dereference
		reference operator*() const {
			return *m_pos;
		}
		reference operator [](difference_type n) const {
			return m_pos[offset(n)];
		}

		// Index
		size_type index() const {
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
		difference_type m_index;
		difference_type m_size;
		template<class> friend class major2_iterator;
	};
	
	typedef typename boost::mpl::if_<
		boost::is_same<Orientation,row_major>,
		dense_storage_iterator<value_type>,
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
		dense_storage_iterator<value_type>
	>::type column_iterator;
	
	typedef typename boost::mpl::if_<
		boost::is_same<Orientation,row_major>,
		dense_storage_iterator<value_type const>,
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
		dense_storage_iterator<value_type const>
	>::type const_column_iterator;
	
public:

	const_row_iterator row_begin(index_type i) const {
		std::size_t index = TriangularType::is_upper?i:0;
		return const_row_iterator(
			&m_data[orientation::element(i,index,size1())]
			,index
			,orientation::stride2(size1(),size2())//1 if row_major, size2() otherwise
		);
	}
	const_row_iterator row_end(index_type i) const {
		std::size_t index = TriangularType::is_upper?size2():i+1;
		return const_row_iterator(
			&m_data[orientation::element(i,index,size1())]
			,index
			,orientation::stride2(size1(),size2())//1 if row_major, size2() otherwise
		);
	}
	row_iterator row_begin(index_type i){
		std::size_t index = TriangularType::is_upper?i:0;
		return row_iterator(
			&m_data[orientation::element(i,index,size1())]
			,index
			,orientation::stride2(size1(),size2())//1 if row_major, size2() otherwise
		);
	}
	row_iterator row_end(index_type i){
		std::size_t index = TriangularType::is_upper?size2():i+1;
		return row_iterator(
			&m_data[orientation::element(i,index,size1())]
			,index
			,orientation::stride2(size1(),size2())//1 if row_major, size2() otherwise
		);
	}
	
	const_column_iterator column_begin(index_type i) const {
		std::size_t index = TriangularType::is_upper?0:i;
		return const_column_iterator(
			&m_data[orientation::element(index,i,size1())]
			,index
			,orientation::stride1(size1(),size2())//size1() if row_major, 1 otherwise
		);
	}
	const_column_iterator column_end(index_type i) const {
		std::size_t index = TriangularType::is_upper?i+1:size2();
		return const_column_iterator(
			&m_data[orientation::element(index,i,size1())]
			,index
			,orientation::stride1(size1(),size2())//size1() if row_major, 1 otherwise
		);
	}
	column_iterator column_begin(index_type i){
		std::size_t index = TriangularType::is_upper?0:i;
		return column_iterator(
			&m_data[orientation::element(index,i,size1())]
			,index
			,orientation::stride1(size1(),size2())//size1() if row_major, 1 otherwise
		);
	}
	column_iterator column_end(index_type i){
		std::size_t index = TriangularType::is_upper?i+1:size2();
		return column_iterator(
			&m_data[orientation::element(index,i,size1())]
			,index
			,orientation::stride1(size1(),size2())//size1() if row_major, 1 otherwise
		);
	}

	// Serialization
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */) {

		// we need to copy to a collection_size_type to get a portable
		// and efficient boost::serialization
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

}
}

#endif
