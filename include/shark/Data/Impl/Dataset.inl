/*!
 *  \author O. Krause
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_DATA_IMPL_DATASET_INL
#define SHARK_DATA_IMPL_DATASET_INL

#include <shark/Data/BatchInterface.h>
#include <shark/Core/ISerializable.h>
#include <shark/Core/utility/ZipPair.h>
#include <shark/Core/Exception.h>
#include <shark/Core/utility/CanBeCalled.h>

#include <boost/mpl/eval_if.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/iterator/indirect_iterator.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/adaptor/indirected.hpp>

#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <algorithm>

#include <boost/foreach.hpp>
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

/// \brief Shared memory container class with slicing
template <class Type>
class SharedContainer : public ISerializable
{
public:
	typedef typename Batch<Type>::type BatchType;
	typedef typename Batch<Type>::reference reference;
	typedef typename Batch<Type>::const_reference const_reference;
	template <class T> friend bool operator == (const SharedContainer<T>& op1, const SharedContainer<T>& op2);
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
			m_data[i].reset(new BatchType());
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
		SIZE_CHECK(shark::size(data) != 0 );
		std::size_t points = shark::size(data);
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

	///\brief Creates a shared container form another with different batch architecture
	///
	///@param container The old container from whcih to create the one with the new batch sizes
	///@param batchSizes vector with the size of every batch of this container
	///@param dummy to distinguish this call from the subset call
	SharedContainer(SharedContainer const& container, std::vector<std::size_t> batchSizes, bool dummy){
		//create batches
		for(std::size_t i = 0; i != batchSizes.size(); ++i){
			m_data.push_back(
				boost::shared_ptr<BatchType>(
					new BatchType(BatchTraits::createBatch(*container.elemBegin(),batchSizes[i]))
				)
			);
		}

		//copy data into batches
		int pos = 0;
		std::size_t batch=0;
		for(std::size_t i = 0; i != batchSizes.size(); ++i){
			for(std::size_t j = 0; j != batchSizes[i]; ++j,++pos){
				if(pos==(int)shark::size(*(container.m_data[batch]))){
					pos = 0;
					++batch;
				}
				get(*(m_data[i]),j)=get(*(container.m_data[batch]),pos);
			}
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
			numElems+=boost::size(*m_data[i]);
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

	///////////ITERATORS OVER THE ELEMENTS//////////////////
private:
	struct BatchRange:public boost::iterator_range<iterator>{
	BatchRange(iterator const& begin, iterator const& end):boost::iterator_range<iterator>(begin,end){}
	};
	struct ConstBatchRange:public boost::iterator_range<const_iterator>{
	ConstBatchRange(const_iterator const& begin, const_iterator const& end)
	:boost::iterator_range<const_iterator>(begin,end){}
	};
public:

	typedef MultiSequenceIterator<BatchRange > element_iterator;
	typedef MultiSequenceIterator<ConstBatchRange > const_element_iterator;


	///\brief Iterator access over the single elements
	const_element_iterator elemBegin() const{
		if(size() == 0)
			return elemEnd();
		
		return const_element_iterator(begin(),end(),begin(),boost::begin(*m_data[0]),0);
	}

	///\brief Iterator access over the single elements
	const_element_iterator elemEnd() const{
		return const_element_iterator(begin(),end(),end(),typename BatchTraits::const_iterator(),numberOfElements());
	}

	///\brief Iterator access over the single elements
	element_iterator elemBegin(){
		if(size() == 0)
			return elemEnd();
		return element_iterator(begin(),end(),begin(),boost::begin(*m_data[0]),0);
	}

	///\brief Iterator access over the single elements
	element_iterator elemEnd(){
		return element_iterator(begin(),end(),end(),typename BatchTraits::iterator(),numberOfElements());
	}

	///////////////////////ADDING NEW BATCHES////////////////////////
	void push_back(BatchType const& batch){
		m_data.push_back(boost::shared_ptr<BatchType>(
			new BatchType(batch)
		));
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
		SHARK_CHECK(isIndependent(), "[SharedContainer::splitBlock] Container is not Independent");
		SIZE_CHECK(elementIndex <= shark::size(*position));

		BatchType& source=*position;
		std::size_t leftElements = elementIndex;
		std::size_t rightElements = shark::size(source)-leftElements;

		if(leftElements == 0 || rightElements == 0)
			return;

		boost::shared_ptr<BatchType> leftSplit(
			new BatchType(BatchTraits::createBatch(get(source,0),leftElements))
		);
		boost::shared_ptr<BatchType> rightSplit(
			new BatchType(BatchTraits::createBatch(get(source,0),rightElements))
		);
		std::copy(boost::begin(source),boost::begin(source)+leftElements,boost::begin(*leftSplit));
		std::copy(boost::begin(source)+leftElements,boost::end(source),boost::begin(*rightSplit));
		*(position.base())=rightSplit;//override old batch
		m_data.insert(position.base(),leftSplit);

	}

	///\brief Splits the container in two independent parts. The lft part remains in the containr, the right is stored as return type
	///
	///Order of elements remain unchanged. The SharedContainer is not allowed to be shared for
	///this to work.
	SharedContainer<Type> splice(iterator position){
		SHARK_CHECK(isIndependent(), "[SharedContainer::splice] Container is not Independent");
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
		BOOST_FOREACH(std::size_t i, batchSizes)
			sum += i;
		SIZE_CHECK(sum == numberOfElements());


		SHARK_CHECK(isIndependent(), "[SharedContainer::repartition] Container is not Independent");
		Container newPartitioning;
		std::size_t currentBatch = 0;
		std::size_t currentBatchIndex = 0;
		for(std::size_t i = 0; i != batchSizes.size(); ++i){
			//create new batch
			std::size_t currentBatchSize = batchSizes[i];
			boost::shared_ptr<BatchType> newBatch(new BatchType(BatchTraits::createBatch(get(batch(currentBatch),0),currentBatchSize)));
			for(std::size_t j = 0; j != currentBatchSize; ++j){
				get(*newBatch,j)=get(batch(currentBatch),currentBatchIndex);
				++currentBatchIndex;
				if(currentBatchIndex == shark::size(batch(currentBatch))){
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

	/////////////////////MISC/////////////////////////////////

	///\brief  Is the container independent of all others?
	///
	/// In other words, does it NOT share data?
	/// This method checks every BatchType if it is shared. So it should not be called too often.
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

/// compute the complement of the indices with respect to the set [0,...n[
template<class T,class T2>
void complement(
	T const& set,
	std::size_t n,
	T2& comp)
{
	std::vector<unsigned int > parentSet(n);
	for(std::size_t i = 0; i != n; ++i){
		parentSet[i]=i;
	}
	std::vector<unsigned int > setCopy(set.begin(),set.end());
	std::sort(setCopy.begin(),setCopy.end());

	std::vector<unsigned int > resultSet(parentSet.size());
	std::vector<unsigned int >::iterator pos = std::set_difference(
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

struct TransformOneVersusRestLabels
{
	TransformOneVersusRestLabels(unsigned int oneClass)
	: m_oneClass(oneClass)
	{ }

	typedef unsigned int result_type;

	unsigned int operator() (unsigned int label) const
	{
		return ((label == m_oneClass) ? 1 : 0);
	}

private:
	unsigned int m_oneClass;
};

///\brief Selects a subset of features from a given Matrix
///
///TODO: this should be a local class in selectFeatures. However, C++03 does not allow
///such a class to be a template argument. C++11 does.
template<class FeatureSet>
class SelectFeatures{
	public:
		SelectFeatures(FeatureSet const& f):features(f){}
			
		typedef RealMatrix result_type;
		
		RealMatrix operator()(RealMatrix const& input)const{
			RealMatrix output(input.size1(),features.size());
			for(std::size_t i = 0; i != input.size1(); ++i){
				for(std::size_t j = 0; j != features.size(); ++j){
					output(i,j) = input(i,features[j]);
				}
			}
			return output;
		}
	private:
		FeatureSet const& features;
	};

/// \brief For Data<T> and functor F calculates the result of the resulting elements F(T).
template<class Functor, class T>
struct TransformedDataElement{
private:
	template<class B>
	struct TransformedDataElementTypeFromBatch{
		typedef typename boost::range_value<
			typename boost::result_of<Functor(B)>::type 
		>::type type;
	};
public:
	typedef typename boost::mpl::eval_if<
		CanBeCalled<Functor,T>,
		boost::result_of<Functor(T) >,
		TransformedDataElementTypeFromBatch<
			typename Batch<T>::type 
		>
	>::type type;
};
}//end namespace detail

///\brief The type used to mimic a pair of data.
///
/// The template parameters choose which type of data is stored as input or labels
template<class InputType,class LabelType>
struct DataPair{
	typedef InputType& InputReference;
	typedef LabelType& LabelReference;
	typedef InputType const& ConstInputReference;
	typedef LabelType const& ConstLabelReference;
	InputType input;
	LabelType label;

	DataPair(
		InputType const& input,
		LabelType const& label
	):input(input),label(label){}
	template<class InputT, class LabelT>
	DataPair(
		InputT& input,
		LabelT& label
	):input(input),label(label){}

	template<class InputT, class LabelT>
	DataPair(
		DataPair<InputT,LabelT> const& pair
	):input(pair.input),label(pair.label){}

	template<class T>
	DataPair(
		T const& pair
	):input(pair.input),label(pair.label){}
		
	friend bool operator<(DataPair const& op1, DataPair const& op2){
		return op1.label < op2.label;
	}
};

/// \cond

template<class I, class L,class InputIterator, class LabelIterator>
struct PairReference<DataPair<I, L>, InputIterator, LabelIterator >{
	struct type{
		typedef typename boost::iterator_reference<InputIterator>::type InputReference;
		typedef typename boost::iterator_reference<LabelIterator>::type LabelReference;
		typedef InputReference ConstInputReference;
		typedef LabelReference ConstLabelReference;
		InputReference input;
		LabelReference label;

		type(
			InputReference input,
			LabelReference label
		):input(input),label(label){}

		template<class Other>
		type(
			Other const& pair
		):input(pair.input),label(pair.label){}

		template<class Other>
		type& operator=(Other const& pair){
			input = pair.input;
			label = pair.label;
			return *this;
		}
		type& operator=(type const& pair){
			input = pair.input;
			label = pair.label;
			return *this;
		}
		
		friend void swap(type a, type b){
			using std::swap;
			swap(a.input,b.input);
			swap(a.label,b.label);
		}
		friend bool operator<(type const& op1, type const& op2){
			return op1.label < op2.label;
		}
		friend bool operator<(type const& op1, DataPair<I,L> const& op2){
			return op1.label < op2.label;
		}
		friend bool operator<(DataPair<I,L> const&  op1, type const& op2){
			return op1.label < op2.label;
		}
	};
};

/// \endcond

///\brief The type used to mimic a pair of data batches.
///
/// The template parameters choose which type of data is stored as input or labels
template<class InputBatchType,class LabelBatchType>
struct DataBatchPair{
private:
	//A Bunch of typedefs needed to query the types of iterators, references and values of the batches
	typedef typename boost::range_iterator<InputBatchType>::type InputBatchIterator;
	typedef typename boost::range_iterator<InputBatchType const>::type ConstInputBatchIterator;
	typedef typename boost::range_iterator<LabelBatchType>::type LabelBatchIterator;
	typedef typename boost::range_iterator<LabelBatchType const>::type ConstLabelBatchIterator;

	typedef typename boost::iterator_value<InputBatchIterator>::type InputType;
	typedef typename boost::iterator_value<LabelBatchIterator>::type LabelType;

public:
	typedef DataPair<InputType,LabelType> value_type;
	typedef PairIterator<
		value_type,InputBatchIterator,LabelBatchIterator
	> iterator;
	typedef PairIterator<
		value_type,ConstInputBatchIterator,ConstLabelBatchIterator
	> const_iterator;
	typedef typename iterator::reference reference;
	typedef typename const_iterator::reference const_reference;

	InputBatchType input;
	LabelBatchType label;

	DataBatchPair(
		InputBatchType const& input,
		LabelBatchType const& label
	):input(input),label(label){}

	template<class InputBatchT, class LabelBatchT>
	DataBatchPair(
		InputBatchT& input,
		LabelBatchT& label
	):input(input),label(label){}

	template<class Other>
	DataBatchPair(
		Other const& pair
	):input(pair.input),label(pair.label){}


	iterator begin(){
		return iterator(boost::begin(input),boost::begin(label));
	}
	const_iterator begin()const{
		return const_iterator(boost::begin(input),boost::begin(label));
	}

	iterator end(){
		return iterator(boost::end(input),boost::end(label));
	}
	const_iterator end()const{
		return const_iterator(boost::end(input),boost::end(label));
	}

	std::size_t size()const{
		return boost::size(input);
	}

	reference operator[](std::size_t i){
		return get(*this,i);
	}
	const_reference operator[](std::size_t i)const{
		return get(*this,i);
	}
};

/// \cond

template<class InputType, class LabelType>
struct Batch<DataPair<InputType, LabelType> >{
private:
	template<class T>
	struct GetElementInput{
		typedef typename T::ConstInputReference result_type;
		
		result_type operator()(T const& element)const{
			return element.input;
		}
	};
	template<class T>
	struct GetElementLabel{
		typedef typename T::ConstLabelReference result_type;
		
		result_type operator()(T const& element)const{
			return element.label;
		}
	};
public:
	/// \brief Type of a batch of elements.
	typedef DataBatchPair<
		typename Batch<InputType>::type,
		typename Batch<LabelType>::type
	> type;
	
	/// \brief The type of the elements stored in the batch 
	typedef DataPair<InputType, LabelType> value_type;
	
	/// \brief Reference to a single element.
	typedef typename type::reference reference;
	
	/// \brief Const Reference to a single element.
	typedef typename type::const_reference const_reference;
	
	/// \brief Iterator over the elements
	typedef typename type::iterator iterator;
	/// \brief Iterator over the elements
	typedef typename type::const_iterator const_iterator;
	
	///\brief creates a batch with input as size blueprint
	static type createBatch(value_type const& input, std::size_t size = 1){
		return type(
			Batch<InputType>::createBatch(input.input,size),
			Batch<LabelType>::createBatch(input.label,size)
		);
	}
	///\brief creates a batch storing the elements referenced by the provided range
	template<class Range>
	static type createBatch(Range const& range){
		typedef typename 
			boost::remove_const<typename boost::remove_reference<
				typename Range::const_reference
			>::type>::type R;
		using boost::adaptors::transform;
		return type(
			Batch<InputType>::createBatchFromRange(transform(range, GetElementInput<R>())),
			Batch<LabelType>::createBatchFromRange(transform(range, GetElementLabel<R>()))
		);
	}
	
	
	static void resize(type& batch, std::size_t batchSize, std::size_t elements){
		Batch<InputType>::resize(batch.input,batchSize,elements);
		Batch<LabelType>::resize(batch.label,batchSize,elements);
	}
};

template<class InputBatchType, class LabelBatchType,class OuterInputBatchIterator, class OuterLabelBatchIterator>
struct PairReference<DataBatchPair<InputBatchType, LabelBatchType>, OuterInputBatchIterator, OuterLabelBatchIterator >{
private:
	typedef typename boost::iterator_reference<OuterInputBatchIterator>::type InputBatchReference;
	typedef typename boost::iterator_reference<OuterLabelBatchIterator>::type LabelBatchReference;

	typedef typename boost::range_iterator<typename boost::remove_reference<InputBatchReference>::type >::type InputBatchIterator;
	typedef typename boost::range_iterator<typename boost::remove_reference<LabelBatchReference>::type>::type LabelBatchIterator;
	typedef typename boost::iterator_value<InputBatchIterator>::type InputType;
	typedef typename boost::iterator_value<LabelBatchIterator>::type LabelType;
public:
	struct type{
		typedef DataPair<InputType,LabelType> value_type;
		typedef PairIterator<
			value_type,InputBatchIterator,LabelBatchIterator
		> iterator;
		typedef iterator const_iterator;

		typedef typename boost::iterator_reference<iterator>::type reference;
		typedef typename boost::iterator_reference<const_iterator>::type const_reference;


		InputBatchReference input;
		LabelBatchReference label;

		type(
			InputBatchReference input,
			LabelBatchReference label
		):input(input),label(label){}

		template<class Other>
		type(
			Other const& pair
		):input(pair.input),label(pair.label){}

		template<class Other>
		type& operator=(Other const& pair){
			input = pair.input;
			label = pair.label;
			return *this;
		}
		type& operator=(type const& pair){
			input = pair.input;
			label = pair.label;
			return *this;
		}

		iterator begin(){
			return iterator(boost::begin(input),boost::begin(label));
		}
		const_iterator begin()const{
			return const_iterator(
				boost::begin(const_cast<typename boost::remove_reference<InputBatchReference>::type& >(input)),
				boost::begin(const_cast<typename boost::remove_reference<LabelBatchReference>::type&>(label))
			);
		}

		iterator end(){
			return iterator(boost::end(input),boost::end(label));
		}
		const_iterator end()const{
			return const_iterator(
				boost::end(const_cast<typename boost::remove_reference<InputBatchReference>::type& >(input)),
				boost::end(const_cast<typename boost::remove_reference<LabelBatchReference>::type&>(label))
			);
		}

		std::size_t size()const{
			return boost::size(input);
		}

		reference operator[](std::size_t i){
			return get(*this,i);
		}
		const_reference operator[](std::size_t i)const{
			return get(*this,i);
		}
		
		friend void swap(type& a, type& b){
			using std::swap;
			swap(a.input,b.input);
			swap(a.label,b.label);
		}
	};
};
/// \endcond

/** @*/
}
#endif
