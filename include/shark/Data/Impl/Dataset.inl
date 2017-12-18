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
	std::vector<std::size_t> batchSizes;
	std::size_t batches = numElements / maximumBatchSize;
	if(numElements-batches*maximumBatchSize > 0)
		++batches;
	std::size_t optimalBatchSize=numElements/batches;
	std::size_t remainder = numElements-batches*optimalBatchSize;

	for(std::size_t j = 0; j != batches; ++j){
		std::size_t size = (j<remainder)?optimalBatchSize+1:optimalBatchSize;
		batchSizes.push_back(size);
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

/// compute the index set for the range [0, ..., size[
template<class T>
void range(size_t size, T& indices){
	indices.resize(size);
	for (size_t i=0; i<size; i++) indices[i] = i;
}
/// compute the index set for the range [start, ..., size+start[
template<class T>
void range(size_t size,size_t start, T& indices){
	indices.resize(size);
	for (size_t i=0; i<size; i++) indices[i] = start+i;
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

	///\brief Create an empty container of specified size with copies of an element
	///
	///@param size the new size of the container
	///@param element the blueprint element from which to create the Container
	///@param batchSize the size of the batches. if this is 0, the size is unlimited
	SharedContainer(std::size_t size, Type const& element, std::size_t batchSize){
		initializeBatches(size,element,batchSize);
	}

	///\brief Create container from new data.
	///
	///Creates the SharedContainer and splits the incoming data into several batches
	///
	///@param data the data from which to create the Container
	///@param maximumBatchSize The size of the batches. If set to 0, the size is unlimited
	template<class Range>
	SharedContainer(Range const& data, std::size_t maximumBatchSize){
		SIZE_CHECK(data.size() != 0 );
		std::size_t points = data.size();
		if(maximumBatchSize == 0)
			maximumBatchSize = points;
		
		//first determin the optimal number of batches as well as batch size
		std::size_t batches = points / maximumBatchSize;
		if(points > batches*maximumBatchSize)
			++batches;
		std::size_t optimalBatchSize=points/batches;
		std::size_t remainder = points-batches*optimalBatchSize;

		//now create the batches taking the remainder into account
		m_data.reserve(batches);
		std::size_t batchStart = 0;
		for(std::size_t i = 0; i != batches; ++i){
			std::size_t size = (i<remainder)?optimalBatchSize+1:optimalBatchSize;
			std::size_t batchEnd = batchStart+size;//sliced is last element inclusive
			push_back(Batch<Type>::createBatch(
				boost::make_iterator_range(boost::begin(data)+batchStart,boost::begin(data)+batchEnd)
			));
			batchStart+=size;
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
		std::copy(BatchTraits::begin(source),BatchTraits::begin(source)+leftElements,BatchTraits::begin(*leftSplit));
		std::copy(BatchTraits::begin(source)+leftElements,BatchTraits::end(source),BatchTraits::begin(*rightSplit));
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

	///\brief Reorders the batch structure in the container to that indicated by the batchSizes vector
	///
	///After the operation the container will contain batchSizes.size() batchs with the i-th batch having size batchSize[i].
	///However the sum of all batch sizes must be equal to the current number of elements
	template<class Range>
	void repartition(Range const& batchSizes){
		std::size_t sum = 0;
		for(std::size_t i: batchSizes)
			sum += i;
		SIZE_CHECK(sum == numberOfElements());


		SHARK_RUNTIME_CHECK(isIndependent(), "Container is not Independent");
		Container newPartitioning;
		std::size_t currentBatch = 0;
		std::size_t currentBatchIndex = 0;
		for(std::size_t i = 0; i != batchSizes.size(); ++i){
			//create new batch
			std::size_t currentBatchSize = batchSizes[i];
			boost::shared_ptr<BatchType> newBatch = boost::make_shared<BatchType>(BatchTraits::createBatch(getBatchElement(batch(currentBatch),0),currentBatchSize));
			for(std::size_t j = 0; j != currentBatchSize; ++j){
				getBatchElement(*newBatch,j)=getBatchElement(batch(currentBatch),currentBatchIndex);
				++currentBatchIndex;
				if(currentBatchIndex == BatchTraits::size(batch(currentBatch))){
					m_data[currentBatch].reset();//free old memory
					++currentBatch;
					currentBatchIndex = 0;
				}
			}
			newPartitioning.push_back(newBatch);
		}
		//sanity check
		SIZE_CHECK(currentBatch == size());
		//swap old(mpty) with new partitioning
		swap(m_data,newPartitioning);


	}
	
	/// \brief Creates a vector with the batch sizes of every batch.
	///
	/// This method can be used together with repartition to ensure
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

	void initializeBatches(std::size_t numElements, Type const& element,std::size_t batchSize){
		m_data.clear();
		if(batchSize == 0|| batchSize > numElements){
			push_back(BatchTraits::createBatch(element,numElements));
		}
		else
		{
			std::size_t batches = numElements/batchSize+(numElements%batchSize > 0);
			m_data.reserve(batches);
			std::size_t finalBatchSize = numElements;
			for(std::size_t batch = 0; batch != batches-1; ++batch){
				push_back(BatchTraits::createBatch(element,batchSize));
				finalBatchSize-=batchSize;
			}
			push_back(BatchTraits::createBatch(element,finalBatchSize));
		}
	}
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

template<class Dataset>
class DataElementIterator: 
public SHARK_ITERATOR_FACADE< 
	DataElementIterator<Dataset>, 
	typename Dataset::element_type, 
	std::random_access_iterator_tag, 
	typename boost::mpl::if_<
		std::is_const<Dataset>,
		typename Dataset::const_element_reference,
		typename Dataset::element_reference
	>::type
>{
private:
	Dataset* m_container;
	std::size_t m_batchPosition;
	std::size_t m_elementPosition;
	std::size_t m_positionInSequence;
public:
	typedef typename boost::mpl::if_<
		std::is_const<Dataset>,
		typename Dataset::const_element_reference,
		typename Dataset::element_reference
	>::type reference;

	DataElementIterator()
	:m_positionInSequence(0){}

	DataElementIterator(
		Dataset* container,
		std::size_t batchPosition,
		std::size_t elementPosition,
		std::size_t positionInSequence
	):m_container(container),
	m_batchPosition(batchPosition),
	m_elementPosition(elementPosition),
	m_positionInSequence(positionInSequence){}

	template<class D>
	DataElementIterator(DataElementIterator<D> const& other)
	:m_container(other.m_container),
	m_batchPosition(other.m_batchPosition),
	m_elementPosition(other.m_elementPosition),
	m_positionInSequence(other.m_positionInSequence){}
	
	template<class D>
	DataElementIterator operator=(DataElementIterator<D> const& other){
		m_container = other.m_container;
		m_batchPosition = other.m_batchPosition;
		m_elementPosition = other.m_elementPosition;
		m_positionInSequence = other.m_positionInSequence;
	}
		
	std::size_t index()const{
		return m_positionInSequence;
	}
	
	auto getInnerIterator()const ->decltype (batchBegin(m_container->batch(m_batchPosition))){
		return batchBegin(m_container->batch(m_batchPosition)) + m_elementPosition;
	}

private:
	friend class SHARK_ITERATOR_CORE_ACCESS;
	template <class> friend class DataElementIterator;

	void increment() {
		++m_positionInSequence;
		++m_elementPosition;
		if(m_elementPosition == batchSize(m_container->batch(m_batchPosition))){
			++m_batchPosition;
			m_elementPosition = 0;
		}
	}
	void decrement() {
		SIZE_CHECK(m_positionInSequence);//don't call this method when the iterator is on the first element
		--m_positionInSequence;
		if(m_elementPosition == 0){
			--m_batchPosition;
			m_elementPosition = batchSize(m_container->batch(m_batchPosition));
		}
		--m_elementPosition;
	}
	//this is not exactly O(1) as the standard wants. in fact it's O(n) in the number of inner sequences
	//so approximately O(1) if the size of a sequence is big...
	void advance(std::ptrdiff_t n){
		m_positionInSequence += n;
		n += m_elementPosition;//jump from the start of the current inner sequence
		m_elementPosition = 0;
		if( n == 0)
			return;
		if(n < 0){
			std::size_t npos = -n;
			--m_batchPosition;
			--npos;
			//jump over the outer position until we are in the correct range again
			while (npos != 0 && npos >= batchSize(m_container->batch(m_batchPosition)) ){
				npos -= batchSize(m_container->batch(m_batchPosition));
				--m_batchPosition;
			}
			m_elementPosition = batchSize(m_container->batch(m_batchPosition)) - 1 - npos;
		}
		else{
			std::size_t npos = n;
			//jump over the outer position until we are in the correct range again
			while (npos != 0 && npos >= batchSize(m_container->batch(m_batchPosition))){
				npos -= batchSize(m_container->batch(m_batchPosition));
				++m_batchPosition;
				SHARK_RUNTIME_CHECK(m_batchPosition != m_container->numberOfBatches() || (npos == 0), "iterator went past the end");
			}
			m_elementPosition = npos;
		}
	}

	template<class Iter>
	std::ptrdiff_t distance_to(const Iter& other) const{
		return (std::ptrdiff_t)other.m_positionInSequence - (std::ptrdiff_t)m_positionInSequence;
	}

	template<class Iter>
	bool equal(Iter const& other) const{
		return m_positionInSequence == other.m_positionInSequence;
	}
	reference dereference() const {
		auto&& batch = m_container->batch(m_batchPosition);
		return getBatchElement(batch,m_elementPosition);
	}
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
	typedef typename boost::mpl::eval_if<
		CanBeCalled<Functor,T>,
		std::result_of<Functor&&(T) >,
		TransformedDataElementTypeFromBatch<
			typename Batch<T>::type 
		>
	>::type type;
};
/** @*/
}


///\brief Input-Label pair of data
template<class InputType,class LabelType>
struct InputLabelPair{
	InputType input;
	LabelType label;
	
	InputLabelPair(){}

	template<class I, class L>
	InputLabelPair(
		I&& input,
		L&& label
	):input(input),label(label){}
	
	template<class InputT, class LabelT>
	InputLabelPair(
		InputLabelPair<InputT,LabelT> const& pair
	):input(pair.input),label(pair.label){}
	
	InputLabelPair& operator=(
		InputLabelPair const& pair
	){
		input = pair.input;
		label = pair.label;
		return *this;
	}
	
	template<class InputT, class LabelT>
	InputLabelPair& operator=(
		InputLabelPair<InputT,LabelT> const& pair
	){
		input = pair.input;
		label = pair.label;
		return *this;
	}
		
	friend bool operator<(InputLabelPair const& op1, InputLabelPair const& op2){
		return op1.label < op2.label;
	}
};

template<class I1, class L1, class I2, class L2>
void swap(InputLabelPair<I1, L1>&& p1, InputLabelPair<I2, L2>&& p2){
	using std::swap;
	swap(p1.input,p2.input);
	swap(p1.label,p2.label);
}

///\brief Input label pair of batches
template<class Batch1Type,class Batch2Type>
struct InputLabelBatch{
private:
	typedef typename BatchTraits<typename std::decay<Batch1Type>::type >::type Batch1Traits;
	typedef typename BatchTraits<typename std::decay<Batch2Type>::type >::type Batch2Traits;
public:
	Batch1Type input;
	Batch2Type label;

	typedef InputLabelPair<
		typename Batch1Traits::value_type,
		typename Batch2Traits::value_type
	> value_type;
	//the decltype below adds correct const semantic if the template arguments are references.
	//the behaviour is the same as mimiking the pair {getBatchElement(input,i), getBatchElement(label,i)}
	//depending on whether input or label are const or not (which for reference types should not make any difference)
	typedef InputLabelPair<
		decltype(getBatchElement(std::declval<Batch1Type&>(),0)),
		decltype(getBatchElement(std::declval<Batch2Type&>(),0))
	> reference;
	typedef InputLabelPair<
		decltype(getBatchElement(std::declval<typename std::add_const<Batch1Type>::type&>(),0)),
		decltype(getBatchElement(std::declval<typename std::add_const<Batch2Type>::type&>(),0))
	> const_reference;
	typedef IndexingIterator<InputLabelBatch> iterator;
	typedef IndexingIterator<InputLabelBatch const> const_iterator;

	template<class I, class L>
	InputLabelBatch(
		I&& input,
		L&& label
	):input(input),label(label){}
	
	template<class Pair>
	InputLabelBatch(
		std::size_t size,Pair const& p
	):input(Batch1Traits::createBatch(p.input,size)),label(Batch2Traits::createBatch(p.label,size)){}
	
	template<class I, class L>
	InputLabelBatch& operator=(InputLabelBatch<I,L> const& batch){
		input = batch.input;
		label = batch.label;
		return *this;
	}


	std::size_t size()const{
		return Batch1Traits::size(input);
	}
	
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
		return reference(getBatchElement(input,i),getBatchElement(label,i));
	}
	const_reference operator[](std::size_t i)const{
		return const_reference(getBatchElement(input,i),getBatchElement(label,i));
	}
};

template<class I1, class L1, class I2, class L2>
void swap(InputLabelBatch<I1, L1>& p1, InputLabelBatch<I2, L2>& p2){
	using std::swap;
	swap(p1.input,p2.input);
	swap(p1.label,p2.label);
}

template<class InputType, class LabelType>
struct Batch<InputLabelPair<InputType, LabelType> >
: public detail::SimpleBatch<
	InputLabelBatch<typename detail::element_to_batch<InputType>::type, typename detail::element_to_batch<LabelType>::type>
>{};

template<class InputBatchType, class LabelBatchType>
struct BatchTraits<InputLabelBatch<InputBatchType, LabelBatchType> >{
	typedef typename detail::batch_to_element<InputBatchType>::type InputElem;
	typedef typename detail::batch_to_element<LabelBatchType>::type LabelElem;
	typedef Batch<InputLabelPair<InputElem,LabelElem> > type;
};


}

#endif
