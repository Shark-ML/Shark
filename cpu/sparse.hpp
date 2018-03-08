/*!
 * \brief       Implementation of the sparse matrix class
 * 
 * \author      O. Krause
 * \date        2017
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
#ifndef REMORA_CPU_SPARSE_HPP
#define REMORA_CPU_SPARSE_HPP

#include "iterator.hpp"

#include <vector>

#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>
 
namespace remora{ namespace detail{ 

template<class T, class I>
struct VectorStorageReference{
	typedef sparse_vector_storage<T,I> storage_type;
	typedef I size_type;
	typedef T value_type;
	
	std::size_t max_capacity() const{
		return m_storage.capacity;
	}
	
	void reserve(size_type non_zeros){
		assert(non_zeros <= max_capacity());
	}
	
	VectorStorageReference(storage_type const& storage)
	: m_storage(storage){}
	
	storage_type m_storage;
};


template<class T, class I>
struct VectorStorage{
	typedef sparse_vector_storage<T,I> storage_type;
	typedef I size_type;
	typedef T value_type;
	
	std::size_t max_capacity() const{
		return std::numeric_limits<std::size_t>::max()/sizeof(T);
	}
	
	void reserve(size_type non_zeros){
		if(non_zeros > m_storage.capacity){
			m_indices.resize(non_zeros);
			m_values.resize(non_zeros);
		}
		m_storage.values = m_values.data();
		m_storage.indices = m_indices.data();
		m_storage.capacity = non_zeros;
	}
	
	VectorStorage(): m_storage({nullptr,nullptr,0,0}){}
	
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* file_version */) {
		ar & boost::serialization::make_nvp("nnz", m_storage.nnz);
		ar & boost::serialization::make_nvp("capacity", m_storage.capacity);
		ar & boost::serialization::make_nvp("indices", m_indices);
		ar & boost::serialization::make_nvp("values", m_values);
		m_storage.values = m_values.data();
		m_storage.indices = m_indices.data();
	}
	storage_type m_storage;
private:
	std::vector<T> m_values;
	std::vector<I> m_indices;
};

template<class StorageManager>
struct BaseSparseVector{
	typedef typename StorageManager::storage_type storage_type;
	typedef typename StorageManager::size_type size_type;
	typedef typename StorageManager::value_type value_type;
	
	BaseSparseVector(StorageManager const& manager, std::size_t size, std::size_t nnz)
	: m_manager(manager)
	, m_size(size){};
	
	/// \brief Return the size of the vector
	size_type size() const {
		return m_size;
	}
	
	/// \brief number of non-zeros currently stored in the vector
	size_type nnz() const {
		return m_manager.m_storage.nnz;
	}
	
	/// \brief number of nnz that can be stored before more memory needs to be allocated
	size_type nnz_capacity() const{
		return m_manager.m_storage.capacity;
	}
	
	/// \brief upper bound for nnz capacity
	size_type max_nnz_capacity() const{
		return std::min(size(), m_manager.max_capacity());
	}
	
	/// \brief reserve space for more non-zero elements
	void reserve(size_type non_zeros) {
		if(non_zeros <= nnz_capacity()) return;
		m_manager.reserve(non_zeros);
	}
	
	
	// --------------
	// ITERATORS
	// --------------

	typedef iterators::compressed_storage_iterator<value_type, size_type> iterator;
	typedef iterators::compressed_storage_iterator<value_type const, size_type const> const_iterator;

	/// \brief return an iterator behind the last non-zero element of the vector
	const_iterator begin() const {
		return const_iterator(m_manager.m_storage.values, m_manager.m_storage.indices, 0);
	}

	/// \brief return an iterator behind the last non-zero element of the vector
	const_iterator end() const {
		return const_iterator(m_manager.m_storage.values, m_manager.m_storage.indices, nnz());
	}
	
	/// \brief return an iterator behind the last non-zero element of the vector
	iterator begin() {
		return iterator(m_manager.m_storage.values, m_manager.m_storage.indices, 0);
	}

	/// \brief return an iterator behind the last non-zero element of the vector
	iterator end() {
		return iterator(m_manager.m_storage.values, m_manager.m_storage.indices, nnz());
	}
	
	iterator set_element(iterator pos, size_type index, value_type value) {
		REMORA_RANGE_CHECK(size_type(pos - begin()) <=m_size);
		
		if(pos != end() && pos.index() == index){
			*pos = value;
			return pos + 1;
		}
		//get position of the new element in the array.
		std::ptrdiff_t arrayPos = pos - begin();
		if(nnz() == nnz_capacity())//reserve more space if needed, this invalidates pos.
			reserve(std::min(std::max<size_type>(5,2 * nnz_capacity()),max_nnz_capacity()));
		
		//copy the remaining elements to make space for the new ones
		auto values = m_manager.m_storage.values;
		auto indices = m_manager.m_storage.indices;
		std::copy_backward(values + arrayPos, values + nnz() , values + nnz() +1);
		std::copy_backward(indices + arrayPos, indices + nnz(), indices + nnz() +1);
		//insert new element
		values[arrayPos] = value;
		indices[arrayPos] = index;
		m_manager.m_storage.nnz += 1;
		
		//return new iterator to the next element.
		return iterator(values,indices,arrayPos+1);
	}
	
	iterator clear_range(iterator start, iterator end) {
		//get position of the elements in the array.
		std::ptrdiff_t startPos = start - begin();
		std::ptrdiff_t endPos = end - begin();
		
		//remove the elements in the range
		auto values = m_manager.m_storage.values;
		auto indices = m_manager.m_storage.indices;
		std::copy(values + endPos, values + nnz(), values + startPos);
		std::copy(indices + endPos, indices + nnz() , indices + startPos);
		m_manager.m_storage.nnz -= endPos - startPos;
		//return new iterator to the next element
		return iterator(values, indices, startPos);
	}

	iterator clear_element(iterator pos){
		auto end = pos + 1;
		return clear_range(pos, end);
	}
	
	void clear(){
		clear_range(begin(),end());
	}
protected:
	StorageManager m_manager;
	std::size_t m_size;

	void do_resize(size_type size, bool keep){
		//delete all elements which have indices larger than the new size
		if(size < m_size){
			auto pos = keep? end() : begin();
			auto start = begin();
			while(pos != start){
				auto new_pos = pos-1;
				if(pos.index() < size)
					break;
				pos = new_pos;
			}
			clear_range(pos,end());
		}
		m_size = size;
	}
};

template<class V>
class compressed_vector_reference: public vector_expression<compressed_vector_reference<V>, cpu_tag >{
private:
	template<class> friend class compressed_vector_reference;
public:
	typedef typename V::size_type size_type;
	typedef typename V::value_type value_type;
	typedef typename V::const_reference const_reference;
	typedef typename remora::reference<V>::type reference;
	
	typedef compressed_vector_reference<V const> const_closure_type;
	typedef compressed_vector_reference closure_type;
	typedef typename remora::storage<V>::type storage_type;
	typedef typename V::const_storage_type const_storage_type;
	typedef elementwise<sparse_tag> evaluation_category;

	// Construction
	compressed_vector_reference(V& v):m_vector(&v){}
	//copy or conversion ctor non-const ->const proxy
	compressed_vector_reference(compressed_vector_reference<typename std::remove_const<V>::type > const& ref):m_vector(ref.m_vector){}
	
	//assignment from different expression
	template<class E>
	compressed_vector_reference& operator=(matrix_expression<E, cpu_tag> const& e){
		return assign(*this, typename vector_temporary<E>::type(e));
	}
	
	compressed_vector_reference& operator=(compressed_vector_reference const& e){
		return assign(*this, typename vector_temporary<V>::type(e));
	}
	
	///\brief Returns the underlying storage structure for low level access
	storage_type raw_storage(){
		return m_vector->raw_storage();
	}
	
	///\brief Returns the underlying storage structure for low level access
	const_storage_type raw_storage() const{
		return m_vector->raw_storage();
	}
	
	/// \brief Return the size of the vector
	size_type size() const {
		return m_vector->size();
	}
	
	size_type nnz() const {
		return m_vector->nnz();
	}
	
	size_type nnz_capacity() const {
		return m_vector->nnz_capacity();
	}
	
	void reserve(size_type non_zeros) {
		m_vector->reserve(non_zeros);
	}
	
	typename device_traits<cpu_tag>::queue_type& queue(){
		return device_traits<cpu_tag>::default_queue();
	}
	
	// --------------
	// ITERATORS
	// --------------

	typedef typename std::conditional<
		std::is_const<V>::value,
		typename V::const_iterator,
		typename V::iterator
	>::type iterator;
	typedef iterator const_iterator;

	/// \brief return an iterator tp the first non-zero element of the vector
	iterator begin() const {
		return m_vector->begin();
	}

	/// \brief return an iterator behind the last non-zero element of the vector
	iterator end() const {
		return m_vector->end();
	}
	
	iterator set_element(iterator pos, size_type index, value_type value) {
		return m_vector->set_element(pos,index,value);
	}
	
	iterator clear_range(iterator start, iterator end) {
		m_vector->clear_range(start,end);
	}

	iterator clear_element(iterator pos){
		m_vector->clear_element(pos);
	}
	
	void clear(){
		m_vector->clear();
	}
private:
	V* m_vector;
};

}}
#endif