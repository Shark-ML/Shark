#ifndef SHARK_LINALG_BLAS_UBLAS_VECTOR_SPARSE_
#define SHARK_LINALG_BLAS_UBLAS_VECTOR_SPARSE_

#include "vector_proxy.hpp"

namespace shark {
namespace blas {

/** \brief Compressed array based sparse vector
 *
 * a sparse vector of values of type T of variable size. The non zero values are stored as
 * two seperate arrays: an index array and a value array. The index array is always sorted
 * and there is at most one entry for each index. Inserting an element can be time consuming.
 * If the vector contains a few zero entries, then it is better to have a normal vector.
 * If the vector has a very high dimension with a few non-zero values, then this vector is
 * very memory efficient (at the cost of a few more computations).
 *
 * For a \f$n\f$-dimensional compressed vector and \f$0 \leq i < n\f$ the non-zero elements
 * \f$v_i\f$ are mapped to consecutive elements of the index and value container, i.e. for
 * elements \f$k = v_{i_1}\f$ and \f$k + 1 = v_{i_2}\f$ of these containers holds \f$i_1 < i_2\f$.
 *
 * Supported parameters for the adapted array (indices and values) are \c unbounded_array<> ,
 * \c bounded_array<> and \c std::vector<>.
 *
 * \tparam T the type of object stored in the vector (like double, float, complex, etc...)
 * \tparam I the indices stored in the vector
 */
template<class T, class I>
class compressed_vector:public vector_container<compressed_vector<T, I> > {

	typedef T &true_reference;
	typedef compressed_vector<T, I> self_type;
public:

	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef T *pointer;
	typedef const T *const_pointer;
	typedef const T& const_reference;

	typedef I index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type index_pointer;
	
	class reference {
	private:

		const_reference value()const {
			return const_cast<self_type const&>(m_vector)(m_i);
		}
		value_type& ref() const {
			//find position of the index in the array
			index_type const* start = m_vector.indices();
			index_type const* end = start + m_vector.nnz();
			index_type const *pos = std::lower_bound(start,end,m_i);

			if (pos != end&& *pos == m_i)
				return m_vector.m_values[pos-start];
			else {
				//create iterator to the insertion position and insert new element
				iterator posIter(m_vector.values(),m_vector.indices(),pos-start);
				return *m_vector.set_element(posIter, m_i, m_vector.m_zero);
			}
		}

	public:
		// Construction and destruction
		reference(self_type &m, size_type i):
			m_vector(m), m_i(i) {}

		// Assignment
		value_type& operator = (value_type d)const {
			return ref()=d;
		}
		
		value_type& operator=(reference const &v ){
			return ref() = v.value();
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

		// Comparison
		bool operator == (value_type d) const {
			return value() == d;
		}
		bool operator != (value_type d) const {
			return value() != d;
		}
		
		operator const_reference() const{
			return value();
		}
	private:
		self_type& m_vector;
		size_type m_i;
	};

	typedef const vector_reference<const self_type> const_closure_type;
	typedef vector_reference<self_type> closure_type;
	typedef sparse_tag storage_category;

	// Construction and destruction
	compressed_vector():m_size(0), m_nnz(0),m_indices(1,0),m_zero(0){}
	explicit compressed_vector(size_type size, value_type value = value_type(), size_type non_zeros = 0)
	:m_size(size), m_nnz(0), m_indices(non_zeros,0), m_values(non_zeros),m_zero(0){}
	template<class AE>
	compressed_vector(vector_expression<AE> const& ae, size_type non_zeros = 0)
	:m_size(ae().size()), m_nnz(0), m_indices(non_zeros,0), m_values(non_zeros),m_zero(0)
	{
		kernels::assign(*this, ae);
	}

	// Accessors
	size_type size() const {
		return m_size;
	}
	size_type nnz_capacity() const {
		return m_indices.size();
	}
	size_type nnz() const {
		return m_nnz;
	}

	// Storage accessors
	void set_filled(size_type filled) {
		SIZE_CHECK(filled <= nnz_capacity());
		m_nnz = filled;
	}
	
	index_type const* indices() const{
		if(size() == 0)
			return 0;
		return &m_indices[0];
	}
	index_type* indices(){
		if(size() == 0)
			return 0;
		return &m_indices[0];
	}
	value_type const* values() const {
		if(size() == 0)
			return 0;
		return &m_values[0];
	}
	value_type* values(){
		if(size() == 0)
			return 0;
		return &m_values[0];
	}

	void resize(size_type size) {
		m_size = size;
		m_nnz = 0;
	}
	void reserve(size_type non_zeros) {
		if(non_zeros <= nnz_capacity()) return;
		non_zeros = std::min(size(),non_zeros);
		m_indices.resize(non_zeros);
		m_values.resize(non_zeros);
	}

	// Element access
	const_reference operator()(size_type i) const {
		SIZE_CHECK(i < m_size);
		std::size_t pos = lower_bound(i);
		if (pos == nnz() || m_indices[pos] != i)
			return m_zero;
		return m_values [pos];
	}
	reference operator()(size_type i) {
		return reference(*this,i);
	}


	const_reference operator [](size_type i) const {
		return (*this)(i);
	}
	reference operator [](size_type i) {
		return (*this)(i);
	}

	// Zeroing
	void clear() {
		m_nnz = 0;
	}

	// Assignment
	compressed_vector &operator = (compressed_vector const& v) {
		m_size = v.m_size;
		m_nnz = v.m_nnz;
		m_indices = v.m_indices;
		m_values = v.m_values;
		return *this;
	}
	template<class C>          // Container assignment without temporary
	compressed_vector &operator = (vector_container<C> const& v) {
		resize(v().size(), false);
		assign(v);
		return *this;
	}

	compressed_vector &assign_temporary(compressed_vector& v) {
		swap(v);
		return *this;
	}
	template<class AE>
	compressed_vector &operator = (vector_expression<AE> const& ae) {
		self_type temporary(ae, nnz_capacity());
		return assign_temporary(temporary);
	}
	template<class AE>
	compressed_vector &assign(vector_expression<AE> const& ae) {
		kernels::assign(*this, ae);
		return *this;
	}

	// Computed assignment
	template<class AE>
	compressed_vector &operator += (vector_expression<AE> const& ae) {
		self_type temporary(*this + ae, nnz_capacity());
		return assign_temporary(temporary);
	}
	template<class C>          // Container assignment without temporary
	compressed_vector &operator += (vector_container<C> const& v) {
		plus_assign(v);
		return *this;
	}
	template<class AE>
	compressed_vector &plus_assign(vector_expression<AE> const& ae) {
		kernels::assign<scalar_plus_assign> (*this, ae);
		return *this;
	}
	template<class AE>
	compressed_vector &operator -= (vector_expression<AE> const& ae) {
		self_type temporary(*this - ae, nnz_capacity());
		return assign_temporary(temporary);
	}
	template<class C>          // Container assignment without temporary
	compressed_vector &operator -= (vector_container<C> const& v) {
		minus_assign(v);
		return *this;
	}
	template<class AE>
	compressed_vector &minus_assign(vector_expression<AE> const& ae) {
		kernels::assign<scalar_minus_assign> (*this, ae);
		return *this;
	}
	compressed_vector &operator *= (value_type t) {
		kernels::assign<scalar_multiply_assign> (*this, t);
		return *this;
	}
	compressed_vector &operator /= (value_type t) {
		kernels::assign<scalar_divide_assign> (*this, t);
		return *this;
	}

	// Swapping
	void swap(compressed_vector &v) {
		std::swap(m_size, v.m_size);
		std::swap(m_nnz, v.m_nnz);
		m_indices.swap(v.m_indices);
		m_values.swap(v.m_values);
	}

	friend void swap(compressed_vector& v1, compressed_vector& v2){
		v1.swap(v2);
	}

	// Iterator types
	typedef compressed_storage_iterator<value_type const, index_type const> const_iterator;
	typedef compressed_storage_iterator<value_type, index_type const> iterator;

	const_iterator begin() const {
		return const_iterator(values(),indices(),0);
	}

	const_iterator end() const {
		return const_iterator(values(),indices(),nnz());
	}

	iterator begin() {
		return iterator(values(),indices(),0);
	}

	iterator end() {
		return iterator(values(),indices(),nnz());
	}
	
	// Element assignment
	iterator set_element(iterator pos, size_type index, value_type value) {
		RANGE_CHECK(pos - begin() <=m_size);
		
		if(pos != end() && pos.index() == index){
			*pos = value;
			return pos;
		}
		//get position of the new element in the array.
		difference_type arrayPos = pos - begin();
		if (m_nnz <= nnz_capacity())//reserve more space if needed, this invalidates pos.
			reserve(std::max<std::size_t>(2 * nnz_capacity(),1));
		
		//copy the remaining elements to make space for the new ones
		std::copy_backward(
			m_values.begin()+arrayPos,m_values.begin() + m_nnz , m_values.begin() + m_nnz +1
		);
		std::copy_backward(
			m_indices.begin()+arrayPos,m_indices.begin() + m_nnz , m_indices.begin() + m_nnz +1
		);
		//insert new element
		m_values[arrayPos] = value;
		m_indices[arrayPos] = index;
		++m_nnz;
		
		
		//return new iterator to the inserted element.
		return iterator(values(),indices(),arrayPos);
	}
	
	iterator clear_range(iterator start, iterator end) {
		//get position of the elements in the array.
		difference_type startPos = start - begin();
		difference_type endPos = end - begin();
		
		//remove the elements in the range
		std::copy(
			m_values.begin()+endPos,m_values.begin() + m_nnz, m_values.begin() + startPos
		);
		std::copy(
			m_indices.begin()+endPos,m_indices.begin() + m_nnz , m_indices.begin() + startPos
		);
		m_nnz -= endPos - startPos;
		//return new iterator to the next element
		return iterator(values(),indices(), startPos);
	}

	iterator clear_element(iterator pos){
		//get position of the element in the array.
		difference_type arrayPos = pos - begin();
		if(arrayPos == m_nnz-1){//last element
			--m_nnz;
			return end();
		}
		
		std::copy(
			m_values.begin()+arrayPos+1,m_values.begin() + m_nnz , m_values.begin() + arrayPos
		);
		std::copy(
			m_indices.begin()+arrayPos+1,m_indices.begin() + m_nnz , m_indices.begin() + arrayPos
		);
		//return new iterator to the next element
		return iterator(values(),indices(),arrayPos);
	}

	// Serialization
	template<class Archive>
	void serialize(Archive &ar, const unsigned int /* file_version */) {
		boost::serialization::collection_size_type s(m_size);
		ar &boost::serialization::make_nvp("size",s);
		if (Archive::is_loading::value) {
			m_size = s;
		}
		// ISSUE: m_indices and m_values are undefined between m_nnz and capacity (trouble with 'nan'-values)
		ar &boost::serialization::make_nvp("nnz", m_nnz);
		ar &boost::serialization::make_nvp("indices", m_indices);
		ar &boost::serialization::make_nvp("values", m_values);
	}

private:
	std::size_t lower_bound( index_type t)const{
		index_type const* begin = indices();
		index_type const* end = indices()+nnz();
		return std::lower_bound(begin, end, t)-begin;
	}

	size_type m_size;
	size_type m_nnz;
	std::vector<index_type> m_indices;
	std::vector<value_type> m_values;
	value_type m_zero;
};

template<class T>
struct vector_temporary_type<T,sparse_bidirectional_iterator_tag>{
	typedef compressed_vector<T> type;
};

}}

#endif
