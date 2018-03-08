/*!
 * \brief       Classes used for vector proxies
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
 #ifndef REMORA_SPARSE_HPP
#define REMORA_SPARSE_HPP


#include "detail/storage.hpp"
#include "detail/traits.hpp"
#include "cpu/sparse.hpp"
#include "cpu/sparse_matrix.hpp"
#include "expression_types.hpp"
#include "assignment.hpp"


namespace remora{

template<class T, class I = std::size_t> class compressed_vector;
template<class T, class I = std::size_t> class compressed_vector_adaptor;
template<class T, class I = std::size_t, class Orientation = row_major> class compressed_matrix;


/** \brief Compressed array based sparse vector
 *
 * a sparse vector of values of type T of variable size. The non zero values are stored as
 * two seperate arrays: an index array and a value array. The index array is always sorted
 * and there is at most one entry for each index. Inserting an element can be time consuming.
 * If the vector has a very high dimension with a few non-zero values, then this vector is
 * very time and memory efficient.
 *
 * For a \f$n\f$-dimensional compressed vector and \f$0 \leq i < n\f$ the non-zero elements
 * \f$v_i\f$ are mapped to consecutive elements of the index and value container, i.e. for
 * elements \f$k = v_{i_1}\f$ and \f$k + 1 = v_{i_2}\f$ of these containers holds \f$i_1 < i_2\f$.
 *
 * \tparam T the type of object stored in the vector (like double, float, complex, etc...)
 * \tparam I the indices stored in the vector
 */
template<class T, class I>
class compressed_vector
:public vector_container<compressed_vector<T, I>, cpu_tag >
,public detail::BaseSparseVector<detail::VectorStorage<T,I > >{
public:
	typedef T value_type;
	typedef I size_type;
	typedef T const& const_reference;
	typedef T& reference;
	
	typedef detail::compressed_vector_reference<compressed_vector const> const_closure_type;
	typedef detail::compressed_vector_reference<compressed_vector> closure_type;
	typedef sparse_vector_storage<T,I> storage_type;
	typedef sparse_vector_storage<T const,I const> const_storage_type;
	typedef elementwise<sparse_tag> evaluation_category;

	// Construction and destruction
	compressed_vector()
	: detail::BaseSparseVector<detail::VectorStorage<T,I> >(
		detail::VectorStorage<T, I>(),0,0
	){}
	explicit compressed_vector(size_type size, size_type non_zeros = 0)
	: detail::BaseSparseVector<detail::VectorStorage<T,I> >(
		detail::VectorStorage<T, I>(),size,non_zeros
	){}
	template<class E>
	compressed_vector(vector_expression<E, cpu_tag> const& e, size_type non_zeros = 0)
	: detail::BaseSparseVector<detail::VectorStorage<T,I> >(
		detail::VectorStorage<T, I>(),e().size(),non_zeros
	){
		assign(*this,e);
	}
	
	void resize(size_type size, bool keep = false){
		this->do_resize(size, keep);
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return this->m_manager.m_storage;
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return this->m_manager.m_storage;
	}
	
	typename device_traits<cpu_tag>::queue_type& queue() const{
		return device_traits<cpu_tag>::default_queue();
	}

	friend void swap(compressed_vector& v1, compressed_vector& v2){
		std::swap(v1.m_size, v2.m_size);
		v1.m_indices.swap(v2.m_indices);
		v1.m_values.swap(v2.m_values);
		v1.m_storage.swap(v2.m_storage);
	}

	// Assignment
	compressed_vector& operator = (compressed_vector const& v) = default;
	template<class C>          // Container assignment without temporary
	compressed_vector& operator = (vector_container<C, cpu_tag> const& v) {
		this->resize(v().size(), false);
		assign(*this, v);
		return *this;
	}
	template<class AE>
	compressed_vector& operator = (vector_expression<AE, cpu_tag> const& ae) {
		compressed_vector temporary(ae, this->nnz_capacity());
		swap(temporary);
		return *this;
	}

	// Serialization
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */) {
		ar & static_cast< detail::BaseSparseVector<detail::VectorStorage<T,I > >& >(*this);
	}
};


/** \brief Wraps external memory compatible to the format of a compressed vector
 *
 * For a \f$n\f$-dimensional compressed vector and \f$0 \leq i < n\f$ the non-zero elements
 * \f$v_i\f$ are mapped to consecutive elements of the index and value container, i.e. for
 * elements \f$k = v_{i_1}\f$ and \f$k + 1 = v_{i_2}\f$ of these containers holds \f$i_1 < i_2\f$.
 *
 * There are 4 values needed: the address of the arrays of indices and values, the number of nonzero elements
 * and the size of the arrays (which can be larger than the number of noznero elements to allow for insertion).
 *
 * \tparam T the type of object stored in the vector (like double, float, complex, etc...)
 * \tparam I the indices stored in the vector
 */
template<class T, class I>
class compressed_vector_adaptor
: public vector_expression<compressed_vector_adaptor<T, I>, cpu_tag>
,public detail::BaseSparseVector<detail::VectorStorageReference<T,I > >{
public:
	typedef typename std::remove_const<T>::type value_type;
	typedef typename std::remove_const<I>::type size_type;
	typedef T const& const_reference;
	typedef T& reference;
	
	typedef detail::compressed_vector_reference<compressed_vector_adaptor const> const_closure_type;
	typedef detail::compressed_vector_reference<compressed_vector_adaptor> closure_type;
	typedef sparse_vector_storage<T,I> storage_type;
	typedef sparse_vector_storage<T const,I const> const_storage_type;
	typedef elementwise<sparse_tag> evaluation_category;

	// Construction and destruction

	/// \brief Constructs the adaptor from external storage
	explicit compressed_vector_adaptor(size_type size, storage_type storage)
	: detail::BaseSparseVector<detail::VectorStorageReference<T,I> >(
		detail::VectorStorageReference<T,I>(storage), size, storage.nnz
	){}
	
	/// \brief Covnerts an expression into an adaptor
	template<class E>
	compressed_vector_adaptor(vector_expression<E, cpu_tag> const& e)
	: detail::BaseSparseVector<detail::VectorStorageReference<T,I> >(
		detail::VectorStorageReference<T,I>(e().raw_storage()), e().size(), e().raw_storage().nnz
	){}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return this->m_manager.m_storage;
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return this->m_manager.m_storage;
	}
	
	typename device_traits<cpu_tag>::queue_type& queue() const{
		return device_traits<cpu_tag>::default_queue();
	}

	// Assignment
	compressed_vector_adaptor& operator = (compressed_vector_adaptor const& v) {
		kernels::assign(*this, typename vector_temporary<compressed_vector_adaptor>::type(v));
		return *this;
	}
	template<class AE>
	compressed_vector_adaptor& operator = (vector_expression<AE, cpu_tag> const& ae) {
		kernels::assign(*this, typename vector_temporary<AE>::type(ae));
		return *this;
	}
};

template<class T, class I, class Orientation>
class compressed_matrix:public matrix_container<compressed_matrix<T, I, Orientation>, cpu_tag >{
public:
	typedef I size_type;
	typedef T value_type;
	typedef T const& const_reference;
	typedef T& reference;
	
	typedef detail::compressed_matrix_proxy<compressed_matrix<T, I, Orientation> const, Orientation> const_closure_type;
	typedef detail::compressed_matrix_proxy<compressed_matrix<T, I, Orientation>, Orientation> closure_type;
	typedef sparse_matrix_storage<T, I> storage_type;
	typedef sparse_matrix_storage<T const, I const> const_storage_type;
	typedef elementwise<sparse_tag> evaluation_category;
	typedef Orientation orientation;

	compressed_matrix():m_impl(detail::MatrixStorage<T,I>(0,0)){}
	
	compressed_matrix(size_type rows, size_type cols, size_type non_zeros = 0)
	:m_impl(detail::MatrixStorage<T,I>(orientation::index_M(rows,cols),orientation::index_m(rows,cols)),non_zeros){}
	
	template<class E>
	compressed_matrix(matrix_expression<E, cpu_tag> const& m, size_type non_zeros = 0)
	:m_impl(
		detail::MatrixStorage<T,I>(orientation::index_M(m().size1(),m().size2()),orientation::index_m(m().size1(),m().size2())),
		non_zeros
	){
		assign(*this,m);
	}
	
	template<class E>
	compressed_matrix operator=(matrix_container<E, cpu_tag> const& m){
		resize(m.size1(),m.size2());
		return assign(*this,m);
	}
	template<class E>
	compressed_matrix& operator=(matrix_expression<E, cpu_tag> const& m){
		compressed_matrix temporary(m);
		m_impl = std::move(temporary.m_impl);
		return *this;
	}
	
	///\brief Number of rows of the matrix
	size_type size1() const {
		return orientation::index_M(m_impl.major_size(), m_impl.minor_size());
	}
	
	///\brief Number of columns of the matrix
	size_type size2() const {
		return orientation::index_m(m_impl.major_size(), m_impl.minor_size());
	}

	/// \brief Number of nonzeros this matrix can maximally store before requiring new memory
	std::size_t nnz_capacity() const{
		return m_impl.nnz_capacity();
	}
	/// \brief Number of reserved elements in the matrix (> number of nonzeros stored)
	std::size_t nnz_reserved() const {
		return m_impl.nnz_reserved();
	}
	/// \brief Number of nonzeros the major index (a row or column depending on orientation) can maximally store before a resize
	std::size_t major_capacity(size_type i)const{
		return m_impl.major_capacity(i);
	}
	/// \brief Number of nonzeros the major index (a row or column depending on orientation) currently stores
	std::size_t major_nnz(size_type i) const {
		return m_impl.major_nnz(i);
	}

	/// \brief Set the total number of nonzeros stored by the matrix
	void set_nnz(std::size_t non_zeros) {
		m_impl.set_nnz(non_zeros);
	}
	/// \brief Set the number of nonzeros stored in the major index (a row or column depending on orientation)
	void set_major_nnz(size_type i,std::size_t non_zeros) {
		m_impl.set_major_nnz(i,non_zeros);
	}
	
	const_storage_type raw_storage()const{
		return m_impl.raw_storage();
	}
	storage_type raw_storage(){
		return m_impl.raw_storage();
	}
	
	typename device_traits<cpu_tag>::queue_type& queue() const{
		return device_traits<cpu_tag>::default_queue();
	}
	
	void reserve(std::size_t non_zeros) {
		m_impl.reserve(non_zeros);
	}

	void major_reserve(size_type i, std::size_t non_zeros, bool exact_size = false) {
		m_impl.major_reserve(i, non_zeros, exact_size);
	}

	void resize(size_type rows, size_type columns){
		m_impl.resize(orientation::index_M(rows,columns),orientation::index_m(rows,columns));
	}
	
	typedef typename detail::compressed_matrix_impl<detail::MatrixStorage<T,I> >::const_major_iterator const_major_iterator;
	typedef typename detail::compressed_matrix_impl<detail::MatrixStorage<T,I> >::major_iterator major_iterator;

	const_major_iterator major_begin(size_type i) const {
		return m_impl.cmajor_begin(i);
	}

	const_major_iterator major_end(size_type i) const{
		return m_impl.cmajor_end(i);
	}

	major_iterator major_begin(size_type i) {
		return m_impl.major_begin(i);
	}

	major_iterator major_end(size_type i) {
		return m_impl.major_end(i);
	}
	
	major_iterator set_element(major_iterator pos, size_type index, value_type value){
		return m_impl.set_element(pos, index, value);
	}

	major_iterator clear_range(major_iterator start, major_iterator end) {
		return m_impl.clear_range(start,end);
	}
	
	void clear() {
		for(std::size_t i = 0; i != m_impl.major_size(); ++i){
			clear_range(major_begin(i),major_end(i));
		}
	}
	// Serialization
	template<class Archive>
	void serialize(Archive &ar, const unsigned int /* file_version */) {
		ar & m_impl;
	}

private:
	detail::compressed_matrix_impl<detail::MatrixStorage<T,I> > m_impl;
};



//~ ///\brief Wraps externally provided storage into a sparse matrix interface
//~ ///
//~ /// Note that for this class, storage is limited and if an insertion operation takes more space than available, it will throw an exception
//~ template<class T, class I, class Orientation>
//~ class compressed_matrix_adaptor:public matrix_container<compressed_matrix_adaptor<T, I, Orientation>, cpu_tag >{
//~ public:
	//~ typedef I size_type;
	//~ typedef T value_type;
	//~ typedef T const& const_reference;
	//~ typedef T& reference;
	
	//~ typedef detail::compressed_matrix_proxy<compressed_matrix<T, I, Orientation> const, Orientation> const_closure_type;
	//~ typedef detail::compressed_matrix_proxy<compressed_matrix<T, I, Orientation>, Orientation> closure_type;
	//~ typedef sparse_matrix_storage<T, I> storage_type;
	//~ typedef sparse_matrix_storage<T const, I const> const_storage_type;
	//~ typedef elementwise<sparse_tag> evaluation_category;
	//~ typedef Orientation orientation;
	
	//~ compressed_matrix_adaptor(size_type rows, size_type cols, storage_type storage)
	//~ :m_impl(detail::MatrixStorageAdaptor<T,I>(orientation::index_M(rows,cols),orientation::index_m(rows,cols)),storage){}
	
	//~ template<class E>
	//~ compressed_matrix operator=(matrix_container<E, cpu_tag> const& m){
		//~ return assign(*this,m);
	//~ }
	//~ template<class E>
	//~ compressed_matrix& operator=(matrix_expression<E, cpu_tag> const& m){
		//~ compressed_matrix temporary(m);
		//~ swap(*this,temporary);
		//~ return *this;
	//~ }
	
	//~ ///\brief Number of rows of the matrix
	//~ size_type size1() const {
		//~ return orientation::index_M(m_impl.major_size(), m_impl.minor_size());
	//~ }
	
	//~ ///\brief Number of columns of the matrix
	//~ size_type size2() const {
		//~ return orientation::index_m(m_impl.major_size(), m_impl.minor_size());
	//~ }

	//~ /// \brief Number of nonzeros this matrix can maximally store before memory is exhausted
	//~ std::size_t nnz_capacity() const{
		//~ return m_impl.nnz_capacity();
	//~ }
	//~ /// \brief Number of reserved elements in the matrix (> number of nonzeros stored, < nnz_capacity)
	//~ std::size_t nnz_reserved() const {
		//~ return m_impl.nnz_reserved();
	//~ }
	//~ /// \brief Number of nonzeros the major index (a row or column depending on orientation) can maximally store before a resize
	//~ std::size_t major_capacity(size_type i)const{
		//~ return m_impl.major_capacity(i);
	//~ }
	//~ /// \brief Number of nonzeros the major index (a row or column depending on orientation) currently stores
	//~ std::size_t major_nnz(size_type i) const {
		//~ return m_impl.major_nnz(i);
	//~ }

	//~ /// \brief Set the total number of nonzeros stored by the matrix
	//~ void set_nnz(std::size_t non_zeros) {
		//~ m_impl.set_nnz(non_zeros);
	//~ }
	//~ /// \brief Set the number of nonzeros stored in the major index (a row or column depending on orientation)
	//~ void set_major_nnz(size_type i,std::size_t non_zeros) {
		//~ m_impl.set_major_nnz(i,non_zeros);
	//~ }
	
	//~ const_storage_type raw_storage()const{
		//~ return m_impl.raw_storage();
	//~ }
	//~ storage_type raw_storage(){
		//~ return m_impl.raw_storage();
	//~ }
	
	//~ typename device_traits<cpu_tag>::queue_type& queue() const{
		//~ return device_traits<cpu_tag>::default_queue();
	//~ }
	
	//~ void reserve(std::size_t non_zeros) {
		//~ m_impl.reserve(non_zeros);
	//~ }

	//~ void major_reserve(size_type i, std::size_t non_zeros) {
		//~ m_impl.major_reserve(i, non_zeros, true);
	//~ }
	
	//~ typedef typename detail::compressed_matrix_impl<detail::MatrixStorage<T,I> >::const_major_iterator const_major_iterator;
	//~ typedef typename detail::compressed_matrix_impl<detail::MatrixStorage<T,I> >::major_iterator major_iterator;

	//~ const_major_iterator major_begin(size_type i) const {
		//~ return m_impl.cmajor_begin(i);
	//~ }

	//~ const_major_iterator major_end(size_type i) const{
		//~ return m_impl.cmajor_end(i);
	//~ }

	//~ major_iterator major_begin(size_type i) {
		//~ return m_impl.major_begin(i);
	//~ }

	//~ major_iterator major_end(size_type i) {
		//~ return m_impl.major_end(i);
	//~ }
	
	//~ major_iterator set_element(major_iterator pos, size_type index, value_type value){
		//~ return m_impl.set_element(pos, index, value);
	//~ }

	//~ major_iterator clear_range(major_iterator start, major_iterator end) {
		//~ return m_impl.clear_range(start,end);
	//~ }
	
	//~ void clear() {
		//~ for(std::size_t i = 0; i != m_impl.major_size(); ++i){
			//~ clear_range(major_begin(i),major_end(i));
		//~ }
	//~ }
//~ private:
	//~ detail::compressed_matrix_impl<detail::MatrixStorage<T,I> > m_impl;
//~ };

namespace detail{
////////////////////////MATRIX ROW//////////////////////
template<class M>
struct matrix_row_optimizer<detail::compressed_matrix_proxy<M, row_major> >{
	typedef compressed_matrix_row<detail::compressed_matrix_proxy<M, row_major> > type;
	
	static type create(detail::compressed_matrix_proxy<M, row_major> const& m, std::size_t i){
		//create vector reference
		return type(m,i);
	}
};


////////////////////////MATRIX TRANSPOSE//////////////////////
template<class M, class Orientation>
struct matrix_transpose_optimizer<detail::compressed_matrix_proxy<M, Orientation> >{
	typedef detail::compressed_matrix_proxy<M, typename Orientation::transposed_orientation> type;
	
	static type create(detail::compressed_matrix_proxy<M, Orientation> const& m){
		return type(m.matrix(), m.start(), m.end());
	}
};

////////////////////////MATRIX ROWS//////////////////////
template<class M>
struct matrix_rows_optimizer<detail::compressed_matrix_proxy<M, row_major> >{
	typedef detail::compressed_matrix_proxy<M, row_major> type;
	
	static type create(detail::compressed_matrix_proxy<M, row_major> const& m, 
		std::size_t start, std::size_t end
	){
		return type(m.matrix(), start + m.start(), end+m.start());
	}
};
}

template<class T, class O>
struct matrix_temporary_type<T,O,sparse_tag, cpu_tag> {
	typedef compressed_matrix<T,std::size_t, O> type;
};

template<class T>
struct vector_temporary_type<T,sparse_tag, cpu_tag>{
	typedef compressed_vector<T> type;
};

}
#endif
