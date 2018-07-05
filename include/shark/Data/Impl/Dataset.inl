/*!
 *
 * \brief Internal functionality and implementation of the Data class
 *
 *  \author O. Krause
 *  \date 2012
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
#ifndef SHARK_DATA_IMPL_DATASET_INL
#define SHARK_DATA_IMPL_DATASET_INL

#include <shark/Data/BatchInterface.h>
#include <shark/Core/ISerializable.h>
#include <shark/Core/Exception.h>
#include <shark/Core/utility/CanBeCalled.h>

#include <boost/mpl/eval_if.hpp>

#include <boost/serialization/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/iterator/indirect_iterator.hpp>
#include <boost/range/adaptor/indirected.hpp>

#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include <algorithm>
#include <memory>

namespace shark {
namespace detail{

/**
 * \ingroup shark_detail
 *
 * @{
 */
	
	
inline std::size_t numberOfBatches(std::size_t numElements, std::size_t maximumBatchSize){
	if(maximumBatchSize == 0)
		maximumBatchSize =numElements;
	std::size_t batches = numElements / maximumBatchSize;
	if(numElements-batches*maximumBatchSize > 0)
		++batches;
	return batches;
}

///\brief Computes a partitioning of a st of elements in batches.
///	
/// Given a number of elements and the maximum size of a batch, 
/// computes the optimal number of batches and returns the size of every batch such that
/// all batches have as equal a size as possible.
///
/// \param numElements number of elements to partition
/// \param maximumBatchSize the maximum size of a batch
/// \return a vector with th size of every batch
inline std::vector<std::size_t> optimalBatchSizes(std::size_t numElements, std::size_t maximumBatchSize){
	std::size_t batches = numberOfBatches(numElements, maximumBatchSize);
	std::vector<std::size_t> batchSizes(batches);
	std::size_t optimalBatchSize=numElements/batches;
	std::size_t remainder = numElements-batches*optimalBatchSize;

	for(std::size_t i = 0; i != batches; ++i){
		std::size_t size = optimalBatchSize + (i<remainder);
		batchSizes[i] = size;
	}
	return batchSizes;
}

///\brief Given the sizes of the partition sets and the maximum batch size, computes a good partitioning.
///
/// \param[in]   partitionSizes    sizes of the partitions (number of elements)
/// \param[out]  partitionStart    indices of the starting batches of the partition
/// \param[out]  batchSizes        sizes of the batches
/// \param[in]   maximumBatchSize  maximal batch size
/// \return                        the total number of batches
inline std::size_t batchPartitioning(
	std::vector<std::size_t> const& partitionSizes,
	std::vector<std::size_t>& partitionStart,
	std::vector<std::size_t>& batchSizes,
	std::size_t maximumBatchSize
){
	std::size_t sumOfBatches = 0;
	std::size_t numberOfPartitions=partitionSizes.size();
	for (std::size_t i = 0; i != numberOfPartitions; i++){
		partitionStart.push_back(sumOfBatches);
		std::vector<std::size_t> batchSizesOfPartition = optimalBatchSizes(partitionSizes[i],maximumBatchSize);
		batchSizes.insert(batchSizes.end(),batchSizesOfPartition.begin(),batchSizesOfPartition.end());
		sumOfBatches+=batchSizesOfPartition.size();
	}
	return sumOfBatches;
}

/// compute the complement of the indices with respect to the set [0,...n[
template<class T,class T2>
void complement(
	T const& set,
	std::size_t n,
	T2& comp)
{
	std::vector<std::size_t> parentSet(n);
	for(std::size_t i = 0; i != n; ++i){
		parentSet[i]=i;
	}
	std::vector<std::size_t> setCopy(set.begin(),set.end());
	std::sort(setCopy.begin(),setCopy.end());

	std::vector<std::size_t> resultSet(parentSet.size());
	std::vector<std::size_t>::iterator pos = std::set_difference(
		parentSet.begin(),parentSet.end(),
		setCopy.begin(),setCopy.end(),
		resultSet.begin()
	);
	comp.resize(std::distance(resultSet.begin(),pos));
	std::copy(resultSet.begin(),pos,comp.begin());
}

/// \brief Shared memory container class with slicing
template <class Type>
class SharedContainer : public ISerializable
{
public:
	typedef typename Batch<Type>::type BatchType;
	typedef typename Batch<Type>::reference reference;
	typedef typename Batch<Type>::const_reference const_reference;

private:
	typedef Batch<Type> BatchTraits;
	typedef std::vector<boost::shared_ptr<BatchType> > Container;
public:

	///\brief Create an empty container.
	SharedContainer(){}

	///\brief creates a shared container with a number of empty batches.
	SharedContainer(std::size_t numBatches){
		m_data.resize(numBatches);
		for(std::size_t i = 0; i != numBatches; ++i){
			m_data[i] = boost::make_shared<BatchType>();
		}
	}

	///\brief Create a shared container as subset of another container
	///
	///@param container the container from which the subset is to be generated
	///@param indizes indizes indicating which batches should be shared
	SharedContainer(SharedContainer const& container, std::vector<std::size_t> const& indizes){
		m_data.resize(indizes.size());
		for(std::size_t i = 0; i != indizes.size(); ++i){
			SIZE_CHECK(indizes[i] < container.size());
			m_data[i] = container.m_data[indizes[i]];
		}
	}

	/// \brief Clear the contents of this container without affecting the others.
	void clear(){
		m_data.clear();
	}

	///\brief check whether the set is empty
	bool empty() const{
		return size() == 0;
	}

	///\brief Return the number of batches.
	std::size_t size() const{
		return m_data.size();
	}

	///\brief Computes the total number of elements
	std::size_t numberOfElements()const{
		std::size_t numElems = 0;
		for(std::size_t i = 0; i != m_data.size(); ++i){
			numElems+=BatchTraits::size(*m_data[i]);
		}
		return numElems;
	}

	boost::shared_ptr<BatchType> const& pointer(std::size_t i)const{
		return m_data[i];
	}

	BatchType const& batch(std::size_t i)const{
		SIZE_CHECK(i < size());
		return *m_data[i];
	}
	BatchType& batch(std::size_t i){
		SIZE_CHECK(i < size());
		return *m_data[i];
	}

	template <class T> bool operator == (const SharedContainer<T>& rhs) {
		return (m_data == rhs.m_data);
	}

	////////////////////////////ITERATOR INTERFACE//////////////////////////////////////

	///////////ITERATORS OVER THE BATCHES//////////////////
	typedef boost::indirect_iterator< typename Container::const_iterator,const BatchType, boost::use_default, BatchType const& > const_iterator;
	typedef boost::indirect_iterator< typename Container::iterator > iterator;

	///\brief Iterator access over the batches.
	const_iterator begin() const{
		return const_iterator(m_data.begin());

	}

	///\brief Iterator access over the batches.
	const_iterator end() const{
		return const_iterator(m_data.end());
	}

	///\brief Iterator access over the batches.
	iterator begin(){
		return iterator(m_data.begin());
	}

	///\brief Iterator access over the batches.
	iterator end(){
		return iterator(m_data.end());
	}

	///////////////////////ADDING NEW BATCHES////////////////////////
	void push_back(BatchType const& batch){
		m_data.push_back(boost::make_shared<BatchType>(batch));
	}
	
	void append(SharedContainer const& other){
		m_data.insert(m_data.end(),other.m_data.begin(),other.m_data.end());
	}

	////////////////////////////SPLITTING//////////////////////////////////////

	///\brief splits the batch indicated by the iterator at elementIndex in two parts.
	///
	///Order of elements remain unchanged. SharedContainer is not allowed to be shared for
	///this to work.
	void splitBatch(iterator position, std::size_t elementIndex){
		SHARK_RUNTIME_CHECK(isIndependent(), "Container is not Independent");
		SIZE_CHECK(elementIndex <= batchSize(*position));

		BatchType& source=*position;
		std::size_t leftElements = elementIndex;
		std::size_t rightElements = batchSize(source)-leftElements;

		if(leftElements == 0 || rightElements == 0)
			return;

		auto leftSplit = boost::make_shared<BatchType>(BatchTraits::createBatch(getBatchElement(source,0),leftElements));
		auto rightSplit = boost::make_shared<BatchType>(BatchTraits::createBatch(getBatchElement(source,0),rightElements));
		for(std::size_t i = 0; i != leftElements; ++i){
			getBatchElement(*leftSplit,i) = getBatchElement(source,i);
		}
		for(std::size_t i = 0; i != rightElements; ++i){
			getBatchElement(*rightSplit,i) = getBatchElement(source,i + leftElements);
		}
		*(position.base())=rightSplit;//override old batch
		m_data.insert(position.base(),leftSplit);

	}

	///\brief Splits the container in two independent parts. The left part remains in the container, the right is stored as return type
	///
	///Order of elements remain unchanged. The SharedContainer is not allowed to be shared for
	///this to work.
	SharedContainer<Type> splice(iterator position){
		SHARK_RUNTIME_CHECK(isIndependent(), "Container is not Independent");
		SharedContainer<Type> right;
		right.m_data.assign(position.base(),m_data.end());
		m_data.erase(position.base(),m_data.end());
		return right;
	}
	
	/// \brief Creates a vector with the batch sizes of every batch.
	///
	/// This method can be used to ensure
	/// that two SharedContainers have the same batch structure.
	std::vector<std::size_t> getPartitioning()const{
		std::vector<std::size_t> batchSizes(size());
		for(std::size_t i = 0; i != size(); ++i){
			batchSizes[i] = BatchTraits::size(*m_data[i]);
		}
		return batchSizes;
	}
	/////////////////////MISC/////////////////////////////////

	///\brief  Is the container independent of all others?
	///
	/// In other words, does it NOT share data?
	/// This method checks for every batch if it is shared. So it should not be called too often.
	bool isIndependent() const{
		for(std::size_t i = 0; i != m_data.size(); ++i){
			if(!m_data[i].unique()){
				return false;
			}
		}
		return true;
	}

	///\brief Ensures that the container is independent.
	///
	/// Makes sure that the data of this instance can be
	/// modified without affecting other containers. If
	/// necessary, a deep copy of the data is made.
	void makeIndependent(){
		if (isIndependent()){
			return;
		}
		Container dataCopy(m_data.size());
		for(std::size_t i = 0; i != m_data.size(); ++i){
			dataCopy[i].reset(new BatchType(*(m_data[i])));
		}

		using std::swap;
		swap(m_data,dataCopy);
	}

	/// from ISerializable
	void read(InArchive& archive){
		archive & m_data;
	}

	/// from ISerializable
	void write(OutArchive& archive) const{
		archive & m_data;
	}
	
	friend void swap(SharedContainer& a, SharedContainer& b){
		a.m_data.swap(b.m_data);
	}
private:
	/// \brief Shared storage for the element batches.
	Container m_data;
};

template<class C>
struct BatchRange{
	typedef IndexingIterator<BatchRange<C> > iterator;
	typedef IndexingIterator<BatchRange<C> const > const_iterator;
	typedef typename C::batch_type value_type;
	typedef typename boost::mpl::if_<
		std::is_const<C>,
		typename C::const_batch_reference,
		typename C::batch_reference
	>::type reference;
	typedef typename C::const_batch_reference const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	
	BatchRange()=default;
	BatchRange(C* container):m_container(container){}
	
	template<class C2>
	BatchRange(BatchRange<C2> const& other):m_container(other.m_container){}
	
	std::size_t size()const{
		return m_container->numberOfBatches();
	}
	
	bool empty()const{return size() == 0;}
	
	iterator begin(){
		return iterator(*this,0);
	}
	const_iterator begin()const{
		return const_iterator(*this,0);
	}

	iterator end(){
		return iterator(*this,size());
	}
	const_iterator end()const{
		return const_iterator(*this,size());
	}

	reference operator[](std::size_t i){
		return m_container->batch(i);
	}
	const_reference operator[](std::size_t i)const{
		return m_container->batch(i);
	}
	
	reference front(){
		return m_container->batch(0);
	}
	const_reference front()const{
		return m_container->batch(0);
	}
private:
	template <class> friend struct BatchRange;
	C* m_container;
};

/// \brief For Data<T> and functor F calculates the result of the resulting elements F(T).
template<class Functor, class T>
struct TransformedDataElement{
private:
	template<class B>
	struct TransformedDataElementTypeFromBatch{
		typedef typename batch_to_element<
			typename std::result_of<Functor&&(B)>::type 
		>::type type;
	};
public:
	typedef typename std::conditional<
		!CanBeCalled<Functor,typename Batch<T>::type>::value,
		std::result_of<Functor&&(T) >,
		TransformedDataElementTypeFromBatch<
			typename Batch<T>::type 
		>
	>::type::type type;
};
/** @*/
}
}

#endif
