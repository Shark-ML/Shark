/*!
 * 
 *
 * \brief       Defines the Batch Interface for a type, e.g., for every type a container with optimal structure.
 * 
 * 
 *
 * \author      O.Krause
 * \date        2012
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_DATA_BATCHINTERFACE_H
#define SHARK_DATA_BATCHINTERFACE_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Shape.h>
#include <shark/Core/utility/Iterators.h>
#include <shark/Core/Shape.h>

#include <type_traits>

namespace shark{

namespace detail{

/// \brief default implementation of a batch where BatchType is a proper sequence
template<class BatchType>
struct SimpleBatch{
	/// \brief Type of a batch of elements.
	typedef BatchType type;
	/// \brief Reference to a batch
	typedef type& proxy_type;
	/// \brief Const Reference to a batch
	typedef type const& const_proxy_type;
	/// \brief Type of a single element.
	typedef typename type::reference reference;
	/// \brief Type of a single immutable element.
	typedef typename type::const_reference const_reference;
	
	/// \brief The type of the elements stored in the batch 
	typedef typename type::value_type value_type;
	
	/// \brief the type of shape describing the elements
	typedef Shape shape_type;
	
	///\brief creates a batch able to store elements of the structure of input (e.g. same dimensionality)
	static type createBatch(value_type const& input, std::size_t size = 1){
		return type(size,input);
	}

	///\brief creates a batch storing the elements referenced by the provided range
	template<class Iterator>
	static type createBatchFromRange(Iterator const& begin, Iterator const& end){
		type batch(end-begin);
		auto pos = begin;
		for(std::size_t i = 0; i != batch.size(); ++i, ++pos){
			batch[i] = *pos;
		}
		return batch;
	}
	
	static type createBatchFromShape(Shape const&, std::size_t size = 1){
		return type(size);
	}
	
	template<class T>
	static std::size_t size(T const& batch){return batch.size();}
	
	template<class T>
	static typename T::reference get(T& batch, std::size_t i){
		return batch[i];
	}
	template<class T>
	static typename T::const_reference get(T const& batch, std::size_t i){
		return batch[i];
	}
};


///\brief Wrapper for a matrix row, which offers a conversion operator to
/// to the Vector Type.
template<class Matrix>
class MatrixRowReference: public blas::detail::matrix_row_optimizer<
	typename blas::closure<Matrix>::type
>::type{
private:
	typedef typename blas::detail::matrix_row_optimizer<
		typename blas::closure<Matrix>::type
	>::type row_type;
public:
	typedef typename blas::vector_temporary<Matrix>::type Vector;

	MatrixRowReference( Matrix& matrix, std::size_t i)
	:row_type(row(matrix,i)){}
	MatrixRowReference(row_type const& matrixrow)
	:row_type(matrixrow){}
	
	template<class M2>
	MatrixRowReference(MatrixRowReference<M2> const& matrixrow)
	:row_type(matrixrow){}
	
	template<class T> 
	const MatrixRowReference& operator=(const T& argument){
		static_cast<row_type&>(*this)=argument;
		return *this;
	}
	
	operator Vector(){
		return Vector(*this);
	}
};

template<class Matrix>
struct VectorBatch{
	/// \brief Type of a batch of elements.
	typedef typename blas::matrix_temporary<Matrix>::type type;
	
	/// \brief Reference to a batch
	typedef type& proxy_type;
	/// \brief Const Reference to a batch
	typedef type const& const_proxy_type;
	
	/// \brief The type of the elements stored in the batch 
	typedef typename blas::vector_temporary<Matrix>::type value_type;
	
	
	/// \brief Type of a single element.
	typedef detail::MatrixRowReference<Matrix> reference;
	/// \brief Type of a single immutable element.
	typedef detail::MatrixRowReference<const Matrix> const_reference;
	
	/// \brief the type of shape describing the elements
	typedef Shape shape_type;
	
	///\brief creates a batch with input as size blueprint
	template<class Element>
	static type createBatch(Element const& input, std::size_t size = 1){
		return type(size,input.size());
	}
	///\brief creates a batch storing the elements referenced by the provided range
	template<class Iterator>
	static type createBatchFromRange(Iterator const& start, Iterator const& end){
		type batch(end - start,start->size());
		Iterator pos = start;
		for(std::size_t i = 0; i != batch.size1(); ++i, ++pos){
			noalias(row(batch,i)) = *pos;
		}
		return batch;
	}
	
	/// \brief Creates a batch with enough dimensions to store a vector of a specified shape
	static type createBatchFromShape(Shape const& shape, std::size_t size = 1){
		return type(size,shape.numElements());
	}
	
	static std::size_t size(Matrix const& batch){return batch.size1();}
	static reference get( Matrix& batch, std::size_t i){
		return reference(batch,i);
	}
	static const_reference get( Matrix const& batch, std::size_t i){
		return const_reference(batch,i);
	}
};


}

///\brief class which helps using different batch types
///
/// e.g. creating a batch of a single element or returning a single element
/// when the element type is arithmetic, like int,double, std::complex,...
/// the return value will be a linear algebra compatible vector, like RealVector.
/// If it is not, for example a string, the return value will be a std::vector<T>. 
template<class T>
struct Batch
:public std::conditional<
	std::is_arithmetic<T>::value,
	detail::SimpleBatch<blas::vector<T> >,
	detail::SimpleBatch<std::vector<T> >
>::type{};

/// \brief Specialization for vectors which should be matrices in batch mode!
template<class T, class Device>
struct Batch<blas::vector<T, Device> >: public detail::VectorBatch<blas::matrix<T, blas::row_major, Device> >{};


/// \brief specialization for ublas compressed vectors which are compressed matrices in batch mode!
template<class T>
struct Batch<shark::blas::compressed_vector<T> >{
	/// \brief Type of a batch of elements.
	typedef shark::blas::compressed_matrix<T> type;
	
	/// \brief Reference to a batch
	typedef type& proxy_type;
	/// \brief Const Reference to a batch
	typedef type const& const_proxy_type;
	
	/// \brief The type of the elements stored in the batch 
	typedef shark::blas::compressed_vector<T> value_type;
	
	/// \brief Type of a single element.
	typedef detail::MatrixRowReference<type> reference;
	/// \brief Type of a single immutable element.
	typedef detail::MatrixRowReference<const type> const_reference;
	

	/// \brief the type of shape describing the elements
	typedef Shape shape_type;
	
	///\brief creates a batch with input as size blueprint
	template<class Element>
	static type createBatch(Element const& input, std::size_t size = 1){
		return type(size,input.size());
	}

	///\brief creates a batch storing the elements referenced by the provided range
	template<class Iterator>
	static type createBatchFromRange(Iterator const& start, Iterator const& end){
		//before creating the batch, we need the number of nonzero elements
		std::size_t nonzeros = 0;
		for(Iterator pos = start; pos != end; ++pos){
			nonzeros += pos->nnz();
		}
		
		std::size_t size = end - start;
		type batch(size,start->size(),nonzeros);
		
		Iterator pos = start;
		for(std::size_t i = 0; i != size; ++i, ++pos){
			auto row_start = batch.major_end(i);
			for(auto elem_pos = pos->begin(); elem_pos != pos->end(); ++elem_pos){
				row_start = batch.set_element(row_start, elem_pos.index(), *elem_pos);
			}
		}
		return batch;
	}
	
	/// \brief Creates a batch with enough dimensions to store a vector of a specified shape
	static type createBatchFromShape(Shape const& shape, std::size_t size = 1){
		return type(size,shape.numElements());
	}

	static std::size_t size(type const& batch){return batch.size1();}
	static reference get( type& batch, std::size_t i){
		return reference(batch,i);
	}
	static const_reference get( type const& batch, std::size_t i){
		return const_reference(batch,i);
	}
};

template<class M>
struct Batch<detail::MatrixRowReference<M> >
:public Batch<typename detail::MatrixRowReference<M>::Vector>{};


template<class BatchType>
struct BatchTraits{
	typedef Batch<typename std::decay<BatchType>::type::value_type> type;
};

template<class T, class Device>
struct BatchTraits<blas::matrix<T, blas::row_major, Device> >{
	typedef Batch<blas::vector<T, Device> > type;
};
template<class T>
struct BatchTraits<blas::compressed_matrix<T> >{
	typedef Batch<blas::compressed_vector<T> > type;
};
template<class T, class Tag, class Device>
struct BatchTraits<blas::dense_matrix_adaptor<T, blas::row_major, Tag, Device> >{
	typedef detail::VectorBatch<blas::dense_matrix_adaptor<T, blas::row_major, Tag, Device> > type;
};

namespace detail{
template<class T>
struct batch_to_element{
	typedef typename BatchTraits<T>::type::value_type type;
};
template<class T>
struct batch_to_element<T&>{
	typedef typename BatchTraits<T>::type::value_type type;
};
template<class T>
struct batch_to_element<T const&>{
	typedef typename BatchTraits<T>::type::value_type type;
};

template<class T>
struct batch_to_reference{
	typedef typename BatchTraits<T>::type::reference type;
};
template<class T>
struct batch_to_reference<T const>{
	typedef typename BatchTraits<T>::type::const_reference type;
};
template<class T>
struct batch_to_reference<T&>{
	typedef typename BatchTraits<T>::type::reference type;
};
template<class T>
struct batch_to_reference<T const&>{
	typedef typename BatchTraits<T>::type::const_reference type;
};

template<class T>
struct element_to_batch{
	typedef typename Batch<T>::type type;
};
template<class T>
struct element_to_batch<T&>{
	typedef typename Batch<T>::type& type;
};
template<class T>
struct element_to_batch<T const&>{
	typedef typename Batch<T>::type const& type;
};
template<class M>
struct element_to_batch<detail::MatrixRowReference<M> >{
	typedef typename Batch<typename detail::MatrixRowReference<M>::Vector>::type& type;
};
template<class M>
struct element_to_batch<detail::MatrixRowReference<M const> >{
	typedef typename Batch<typename detail::MatrixRowReference<M>::Vector>::type const& type;
};
}

template<class T>
struct shape_type{
	typedef typename Batch<T>::shape_type type;
};

///\brief creates a batch from a range of inputs
template<class T, class Range>
typename Batch<T>::type createBatch(Range const& range){
	return Batch<T>::createBatchFromRange(range.begin(),range.end());
}

///\brief creates a batch from a range of inputs
template<class Range>
typename Batch<typename Range::value_type>::type createBatch(Range const& range){
	return Batch<typename Range::value_type>::createBatchFromRange(range.begin(),range.end());
}

template<class T, class Iterator>
typename Batch<T>::type createBatch(Iterator const& begin, Iterator const& end){
	return Batch<T>::createBatchFromRange(begin,end);
}

template<class BatchT>
auto getBatchElement(BatchT& batch, std::size_t i)->decltype(BatchTraits<BatchT>::type::get(std::declval<BatchT&>(),i)){
	return BatchTraits<BatchT>::type::get(batch,i);
}

template<class BatchT>
auto getBatchElement(BatchT const& batch, std::size_t i)->decltype(BatchTraits<BatchT>::type::get(std::declval<BatchT const&>(),i)){
	return BatchTraits<BatchT>::type::get(batch,i);
}

template<class BatchT>
std::size_t batchSize(BatchT const& batch){
	return BatchTraits<BatchT>::type::size(batch);
}

template<class Sequence>
class BatchIterator: public SHARK_ITERATOR_FACADE<
	BatchIterator<Sequence>,
	typename detail::batch_to_element<Sequence>::type,
	std::random_access_iterator_tag,
	typename detail::batch_to_reference<Sequence>::type
>{
private:
	
	typedef typename detail::batch_to_reference<Sequence>::type Reference;
public:
	BatchIterator() : m_position(0) {}

	BatchIterator(Sequence& seq, std::size_t position)
	: m_sequence(&seq),m_position(position) {}

	template<class S>
	BatchIterator(BatchIterator<S> const& other)
	: m_sequence(other.m_sequence),m_position(other.m_position) {}

private:
	friend class SHARK_ITERATOR_CORE_ACCESS;
	template <class> friend class BatchIterator;

	void increment() {
		++m_position;
	}
	void decrement() {
		--m_position;
	}

	void advance(std::ptrdiff_t n){
		m_position += n;
	}

	template<class Iter>
	std::ptrdiff_t distance_to(const Iter& other) const{
		return (std::ptrdiff_t)other.m_position - (std::ptrdiff_t)m_position;
	}

	template<class Iter>
	bool equal(Iter const& other) const{
		return (m_position == other.m_position);
	}
	Reference dereference() const {
		return getBatchElement(*m_sequence, m_position);
	}

	Sequence* m_sequence;
	std::size_t m_position;
};

}
#endif
