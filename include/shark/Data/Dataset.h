//===========================================================================
/*!
 * 
 *
 * \brief       Data for (un-)supervised learning.
 * 
 * 
 * \par
 * This file provides containers for data used by the models, loss
 * functions, and learning algorithms (trainers). The reason for
 * dedicated containers of this type is that data often need to be
 * split into subsets, such as training and test data, or folds in
 * cross-validation. The containers in this file provide memory
 * efficient mechanisms for managing and providing such subsets.
 * 
 * 
 * 
 *
 * \author      O. Krause, T. Glasmachers
 * \date        2010-2014
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
//===========================================================================

#ifndef SHARK_DATA_DATASET_H
#define SHARK_DATA_DATASET_H

#include <boost/range/iterator_range.hpp>
#include <shark/Core/Exception.h>
#include <shark/Core/Threading/Algorithms.h>
#include <shark/Core/utility/functional.h>
#include <shark/Core/Random.h>
#include <shark/Core/Shape.h>
#include "Impl/Dataset.inl"
#include "Impl/InputLabelPair.h"
#include <shark/Data/BatchInterface.h>
#include <shark/Data/DataView.h>
#include <shark/Data/Generator.h>

namespace shark {

///
/// \brief Data container.
///
/// The Data class is Shark's container for machine learning data.
/// This container (and its sub-classes) is used for input data,
/// labels, and model outputs.
///
/// \par
/// The Data container organizes the data it holds in batches.
/// This means, that it tries to find a good data representation for a whole
/// set of, for example 100 data points, at the same time. If the type of data it stores
/// is for example RealVector, the batches of this type are RealMatrices. This is good because most often
/// operations on the whole matrix are faster than operations on the separate vectors.
/// Nearly all operations of the set have to be interpreted in terms of the batch.
///\par
///When you need to explicitely iterate over all elements, you can use:
///\code
/// Data<RealVector> data;
/// for(auto elem: elements(data)){
///     std::cout<<*pos<<" ";
///     auto ref=*pos;
///     ref*=2;
///     std::cout<<*pos<<std::endl;
///}
///\endcode
/// \par
/// Element wise accessing of elements is usually slower than accessing the batches.
/// Of course, when you want to use batches, you need to know the actual batch type. This depends on the actual type of the input.
/// here are the rules:
/// if the input is an arithmetic type like int or double, the result will be a vector of this
/// (i.e. double->RealVector or Int->IntVector).
/// For vectors the results are matrices as mentioned above. If the vector is sparse, so is the matrix.
/// And for everything else the batch type is just a std::vector of the type, so no optimization can be applied.
/// \par
/// When constructing the container the batchSize can be set. If it is not set by the user the default batchSize is chosen. A BatchSize of 0
/// corresponds to putting all data into a single batch. Beware that not only the data needs storage but also
/// the various models during computation. So the actual amount of space to compute a batch can greatly exceed the batch size.
///
/// An additional feature of the Data class is that it can be used to create lazy subsets. So the batches of a dataset
/// can be shared between various instances of the data class without additional memory overhead.
///
///
///\warning Be aware --especially for derived containers like LabeledData-- that the set does not enforce structural consistency.
/// When you change the structure of the data part for example by directly changing the size of the batches, the size of the labels is not
/// enforced to change accordingly. Also when creating subsets of a set changing the parent will change it's siblings and conversely. The programmer
/// needs to ensure structural integrity!
/// \endcode
///\todo expand docu
template <class Type>
class Data{
private:
	typedef Batch<Type> BatchTraits;
public:
	typedef std::size_t size_type;
	typedef ptrdiff_t difference_type;
	typedef Type element_type;
	typedef typename BatchTraits::shape_type shape_type;
	typedef typename BatchTraits::type value_type;
	typedef typename BatchTraits::type& reference;
	typedef typename BatchTraits::type const& const_reference;

	typedef std::vector<size_type> IndexSet;

	///\brief Returns the number of batches of the set.
	size_type size() const{
		return m_data.size();
	}
	///\brief Returns the total number of elements.
	size_type numberOfElements() const{
		std::size_t numElems = 0;
		for(auto const& p: m_data){
			numElems+=batchSize(*p);
		}
		return numElems;
	}
	
	///\brief Returns the shape of the elements in the dataset.
	shape_type const& shape() const{
		return m_shape;
	}
	
	///\brief Sets the shape of the elements in the dataset.
	void setShape(shape_type const& shape){
		m_shape = shape;
	}

	///\brief Check whether the set is empty.
	bool empty() const{
		return m_data.empty();
	}

	// BATCH ACCESS
	reference operator[](size_type i){
		return *m_data[i];
	}
	const_reference operator[](size_type i) const{
		return *m_data[i];
	}
	// ITERATOR ACCESS
	typedef IndexingIterator<Data > iterator;
	typedef IndexingIterator<Data const > const_iterator;
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
	

	// CONSTRUCTORS

	///\brief Constructor which constructs an empty set
	Data(){ }

	///\brief Constructs a set holding a specific number of elements of a given shape.
	///
	/// Optionally the desired batch Size can be set
	///
	///@param numElements number of data points stored in the dataset
	///@param shape The shape of the elements to create
	///@param batchSize the size of the batches. if this is 0, the size is unlimited
	explicit Data(size_type numElements, shape_type const& shape, size_type batchSize = constants::DefaultBatchSize)
	: m_data( detail::numberOfBatches(numElements, batchSize)), m_shape(shape){
		auto partitioning = detail::optimalBatchSizes(numElements, batchSize);
		for(size_type i = 0; i != partitioning.size(); ++i){
			m_data[i] = boost::make_shared<value_type>(BatchTraits::createBatchFromShape(shape, partitioning[i]));
		}
	}
	
	///\brief Constructs a set with a given shape and a chosen partitioning
	///
	/// For each element in partitioning, a batch is created with the given size
	///
	///@param partitioning batch sizes of the dataset
	///@param shape The shape of the elements to create
	explicit Data(std::vector<size_type> const& partitioning, shape_type const& shape)
	: m_data( partitioning.size()), m_shape(shape){
		for(size_type i = 0; i != partitioning.size(); ++i){
			m_data[i] = boost::make_shared<value_type>(BatchTraits::createBatchFromShape(shape, partitioning[i]));
		}
	}

	// MISC
	bool operator==(Data const& other){
		if(size() != other.size()) return false;
		for(std::size_t i = 0; i != size(); ++i){
			if(m_data[i] != other.m_data[i]) return false; 
		}
		return true;
	}
	
	bool operator!=(Data const& other){
		return !(*this == other);
	}
	
	template<class Archive>
	void serialize(Archive & archive, unsigned int const){
		archive & m_data;
		archive & m_shape;
	}
	
	///\brief  Is the container independent of all others?
	///
	/// In other words, does it NOT share data?
	/// This method checks for every batch if it is shared.
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
		Data dataCopy;
		for(std::size_t i = 0; i != size(); ++i){
			m_data[i] = boost::make_shared<value_type>(*m_data[i]);
		}
	}


	// METHODS TO ALTER BATCH STRUCTURE
	///\brief splits the given batch in two parts.
	///
	///Order of elements remain unchanged. Data is not allowed to be shared for
	///this to work.
	void splitBatch(size_type batch, size_type elementIndex){
		SHARK_RUNTIME_CHECK(isIndependent(), "Container is not Independent");

		value_type& source= *m_data[batch];
		SIZE_CHECK(elementIndex <= batchSize(source));
		std::size_t leftElements = elementIndex;
		std::size_t rightElements = batchSize(source)-leftElements;

		if(leftElements == 0 || rightElements == 0)
			return;

		auto leftSplit = boost::make_shared<value_type>(BatchTraits::createBatch(getBatchElement(source,0),leftElements));
		auto rightSplit = boost::make_shared<value_type>(BatchTraits::createBatch(getBatchElement(source,0),rightElements));
		for(std::size_t i = 0; i != leftElements; ++i){
			getBatchElement(*leftSplit,i) = getBatchElement(source,i);
		}
		for(std::size_t i = 0; i != rightElements; ++i){
			getBatchElement(*rightSplit,i) = getBatchElement(source,i + leftElements);
		}
		m_data[batch] = rightSplit;//override old batch
		m_data.insert(m_data.begin() + batch, leftSplit);

	}

	///\brief Splits the container into two independent parts. The front part remains in the container, the back part is returned.
	///
	///Order of elements remain unchanged. 
	Data splice(size_type batch){
		Data right;
		right.m_data.assign(m_data.begin() + batch, m_data.end());
		right.m_shape = m_shape;
		m_data.erase(m_data.begin() + batch,m_data.end());
		return right;
	}

	/// \brief Appends the contents of another data object to the end
	///
	/// The batches are not copied but now referenced from both datasets. Thus changing the appended
	/// dataset might change this one as well.
	void append(Data const& other){
		m_data.insert(m_data.end(),other.m_data.begin(),other.m_data.end());
	}
	void push_back(const_reference batch){
		m_data.push_back(boost::make_shared<value_type>(batch));
	}
	
	/// \brief Creates a vector with the batch sizes of every batch.
	///
	/// This method can be used together with to ensure
	/// that two datasets have the same batch structure.
	std::vector<size_type> getPartitioning()const{
		std::vector<size_type> batchSizes(size());
		for(std::size_t i = 0; i != size(); ++i){
			batchSizes[i] = BatchTraits::size(*m_data[i]);
		}
		return batchSizes;
	}

	// SUBSETS
	Data indexedSubset(IndexSet const& indices) const{
		Data subset;
		subset.m_data.resize(indices.size());
		for(std::size_t i = 0; i != indices.size(); ++i){
			SIZE_CHECK(indices[i] < size());
			subset.m_data[i] = m_data[indices[i]];
		}
		subset.m_shape = m_shape;
		return subset;
	}

	friend void swap(Data& a, Data& b){
		swap(a.m_data,b.m_data);
		std::swap(a.m_shape,b.m_shape);
	}
protected:
	std::vector<boost::shared_ptr<value_type> > m_data; ;///< stores the data
	shape_type m_shape;///< shape of a datapoint
};

template<class T>
struct InputToDataType{
	typedef Data<T> type;
};

/**
 * \ingroup shark_globals
 * @{
 */

/// Outstream of elements.
template<class T>
std::ostream &operator << (std::ostream &stream, const Data<T>& d) {
	for(auto elem:elements(d))
		stream << elem << "\n";
	return stream;
}

/// \brief Returns a shuffled copy of the input data
///
/// The order of points is randomized and a copy of the initial data object returned.
/// The batch sizes are the same as in the original dataset.
/// \param data the dataset to shuffle
template<class T>
Data<T> shuffle(Data<T> const& data){
	return toDataset(randomSubset(elements(data), data.numberOfElements()),data.getPartition());
}
/** @} */


///
/// \brief Data set for supervised learning.
///
/// The LabeledData class extends Data for the
/// representation of inputs. In addition it holds and
/// provides access to the corresponding labels.
///
/// LabeledData tries to mimic the underlying data as pairs of input and label data.
/// this means that when accessing a batch by calling batch(i) or choosing one of the iterators
/// one access the input batch by batch(i).input and the labels by batch(i).label
template <class InputT, class LabelT>
class LabeledData{
public:
	typedef InputT InputType;
	typedef LabelT LabelType;
	typedef Data<InputT> InputContainer;
	typedef Data<LabelT> LabelContainer;
	typedef typename InputContainer::IndexSet IndexSet;

	typedef std::size_t size_type;
	typedef ptrdiff_t difference_type;
	typedef InputLabelPair<InputType,LabelType> element_type;
	// TYPEDEFS FOR PAIRS
	typedef typename Batch<element_type>::shape_type shape_type;
	typedef typename Batch<element_type>::type value_type;
	typedef typename Batch<element_type>::proxy_type reference;
	typedef typename Batch<element_type>::const_proxy_type const_reference;

	///\brief Access to inputs as a separate container.
	InputContainer const& inputs() const{
		return m_data;
	}
	///\brief Access to inputs as a separate container.
	InputContainer& inputs(){
		return m_data;
	}

	///\brief Access to labels as a separate container.
	LabelContainer const& labels() const{
		return m_label;
	}
	///\brief Access to labels as a separate container.
	LabelContainer& labels(){
		return m_label;
	}

	///\brief Returns the number of batches of the set.
	size_type size() const{
		return m_data.size();
	}
	///\brief Returns the total number of elements.
	size_type numberOfElements() const{
		return m_data.numberOfElements();
	}
	///\brief Check whether the set is empty.
	bool empty() const{
		return m_data.empty();
	}
	
	
	///\brief Returns the shape of the elements in the dataset.
	shape_type shape() const{
		return {m_data.shape(),m_label.shape()};
	}
	
	///\brief Sets the shape of the elements in the dataset.
	void setShape(shape_type const& shape){
		m_data.setShape(shape.input);
		m_label.setShape(shape.label);
	}
	

	// BATCH ACCESS
	reference operator[](size_type i){
		return {m_data[i],m_label[i]};
	}
	const_reference operator[](size_type i) const{
		return {m_data[i],m_label[i]};
	}
	// ITERATOR ACCESS
	typedef IndexingIterator<LabeledData> iterator;
	typedef IndexingIterator<LabeledData const> const_iterator;
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

	// CONSTRUCTORS

	///\brief Empty data set.
	LabeledData(){}

	///\brief Constructs a set holding a specific number of elements of a given shape.
	///
	/// to create a dataset with 100 dimensional inputs and 8 classes, write
	/// LabeledData<RealVector, unsigned int> data(numPoints,{{100}.{8}})
	/// Optionally the desired batch size can be set.
	///
	///@param numElements number of data points stored in the dataset
	///@param shape The shape of the datapoints
	///@param batchSize the size of the batches. if this is 0, the size is unlimited
	explicit LabeledData(size_type numElements, shape_type const& shape, size_type batchSize = constants::DefaultBatchSize)
	: m_data(numElements, shape.input, batchSize), m_label(numElements, shape.label, batchSize){}
	
	///\brief Constructs a set with a given shape and a chosen partitioning
	///
	/// For each element in partitioning, a batch is created with the given size
	///
	///@param partitioning batch sizes of the dataset
	///@param shape The shape of the elements to create
	explicit LabeledData(std::vector<size_type> const& partitioning, shape_type const& shape)
	: m_data( partitioning, shape.input), m_label(partitioning,shape.label){}
	
	explicit LabeledData(Data<InputT> const& data, Data<LabelT> const& label)
	: m_data(data), m_label(label){
		SHARK_RUNTIME_CHECK(data.size() == label.size(), "number of input batches and number of label batches must agree");
		for(size_type i  = 0; i != data.size(); ++i){
			SHARK_RUNTIME_CHECK(batchSize(data[i]) == batchSize(label[i]), "batch sizes of inputs and labels must agree");
		}
	}
	
	// MISC
	bool operator==(LabeledData const& other){
		return (m_data == other.m_data) && (m_label == other.m_label);
	}
	
	bool operator!=(LabeledData const& other){
		return !(*this == other);
	}
	
	
	template<class Archive>
	void serialize(Archive & archive, unsigned int const){
		archive & m_data;
		archive & m_label;
	}

	///\brief This method makes the vector independent of all siblings and parents.
	virtual void makeIndependent(){
		m_label.makeIndependent();
		m_data.makeIndependent();
	}

	void splitBatch(size_type batch, size_type elementIndex){
		m_data.splitBatch(batch,elementIndex);
		m_label.splitBatch(batch,elementIndex);
	}

	///\brief Splits the container into two independent parts. The left part remains in the container, the right is stored as return type
	///
	///Order of elements remain unchanged. The SharedVector is not allowed to be shared for
	///this to work.
	LabeledData splice(size_type batch){
		return LabeledData(m_data.splice(batch),m_label.splice(batch));
	}

	/// \brief Appends the contents of another data object to the end
	///
	/// The batches are not copied but now referenced from both datasets. Thus changing the appended
	/// dataset might change this one as well.
	void append(LabeledData const& other){
		m_data.append(other.m_data);
		m_label.append(other.m_label);
	}
	
	void push_back(
		typename Batch<InputType>::type const& inputs, 
		typename Batch<LabelType>::type const& labels
	){
		m_data.push_back(inputs);
		m_label.push_back(labels);
	}
	
	void push_back(
		const_reference batch
	){
		push_back(batch.input,batch.label);
	}

	
	/// \brief Creates a vector with the batch sizes of every batch.
	///
	/// This method can be used to ensure
	/// that two datasets have the same batch structure.
	std::vector<size_type> getPartitioning()const{
		return m_data.getPartitioning();
	}

	friend void swap(LabeledData& a, LabeledData& b){
		swap(a.m_data,b.m_data);
		swap(a.m_label,b.m_label);
	}


	// SUBSETS

	///\brief Fill in the subset defined by the list of indices.
	LabeledData indexedSubset(IndexSet const& indices) const{
		return LabeledData(m_data.indexedSubset(indices), m_label.indexedSubset(indices));
	}
protected:
	InputContainer m_data;               /// point data
	LabelContainer m_label;		/// label data
};

template<class I, class L>
struct InputToDataType<InputLabelPair<I,L> >{
	typedef LabeledData<I,L> type;
};



/// specialized template for classification with unsigned int labels
typedef LabeledData<RealVector, unsigned int> ClassificationDataset;

/// specialized template for regression with RealVector labels
typedef LabeledData<RealVector, RealVector> RegressionDataset;

/// specialized template for classification with unsigned int labels and sparse data
typedef LabeledData<CompressedRealVector, unsigned int> CompressedClassificationDataset;

/// specialized templates for generators returning labeled data batches
template<class I, class L>
using LabeledDataGenerator = Generator<InputLabelPair<I,L> >;

namespace detail{
template<class T>
struct InferShape{
	static Shape infer(T const&){return {};}
};

template<class T>
struct InferShape<Data<T> >{
	static typename Batch<T>::shape_type infer(Data<T> const&){return {};}
};

template<>
struct InferShape<Data<unsigned int > >{
	static Shape infer(Data<unsigned int > const& labels){
		unsigned int classes = 0;
		for(std::size_t i = 0; i != labels.size(); ++i){
			classes = std::max(classes,*std::max_element(labels[i].begin(),labels[i].end()));
		}
		return {classes+1};
	}
};

template<class T>
struct InferShape<Data<blas::vector<T> > >{
	static Shape infer(Data<blas::vector<T> > const& f){
		return {f[0].size2()};
	}
};

template<class T>
struct InferShape<Data<blas::compressed_vector<T> > >{
	static Shape infer(Data<blas::compressed_vector<T> > const& f){
		return {f[0].size2()};
	}
};

}

/**
 * \addtogroup shark_globals
 * @{
 */

///brief  Outstream of elements for labeled data.
template<class T, class U>
std::ostream &operator << (std::ostream &stream, const LabeledData<T, U>& d) {
	for(auto elem: elements(d))
		stream << elem.input << " [" << elem.label <<"]"<< "\n";
	return stream;
}

/// \brief Returns a shuffled copy of the input data
///
/// The order of (input-label)-pairs is randomized and a copy of the initial data object returned.
/// The batch sizes are the same as in the original dataset.
/// \param data the dataset to shuffle
template<class I, class L>
LabeledData<I,L> shuffle(LabeledData<I,L> const& data){
	return toDataset(randomSubset(elements(data), data.numberOfElements()),data.getPartitioning());
}

/// \brief creates a data object from a range of elements with a given partitioning
template<class Range>
Data<typename Range::value_type>
createDataFromRange(Range const& inputs, std::vector<std::size_t> const& partitioning){
	typedef typename Range::value_type value_type;

	typename Batch<value_type>::shape_type shape;//TODO HACK
	Data<value_type> data(partitioning, shape);
	
	SHARK_RUNTIME_CHECK(inputs.size() == data.numberOfElements(), "Partition has not the same number of elements as number of elements in the input set");

	//now create the batches
	auto start= inputs.begin();
	for(std::size_t i = 0; i != data.size(); ++i){
		std::size_t size = batchSize(data[i]);
		data[i] = createBatch<value_type>(start,start+size);
		start = start+size;
	}
	data.setShape(detail::InferShape<Data<value_type> >::infer(data));
	return data;
}
/// \brief creates a data object from a range of elements
template<class Range>
Data<typename Range::value_type>
createDataFromRange(Range const& inputs, std::size_t maximumBatchSize = constants::DefaultBatchSize){
	auto partitioning = detail::optimalBatchSizes(inputs.size(), maximumBatchSize);
	return createDataFromRange(inputs, partitioning);
}

/// \brief Convenience Function. Creates a labeled data object from two ranges, representing inputs and labels
///
/// inputs and labels must have the same numbers of batches and the batch sizes must agree.
template<class I, class L>
LabeledData<I, L> createLabeledData(Data<I> const& inputs, Data<L> const& labels){
	return LabeledData<I, L>(inputs,labels);
}

/// \brief creates a labeled data object from two ranges, representing inputs and labels
template<class RangeI, class RangeL>
LabeledData<
	typename RangeI::value_type,
	typename RangeL::value_type
> createLabeledDataFromRange(RangeI const& inputs, RangeL const& labels, std::size_t maximumBatchSize = constants::DefaultBatchSize){
	SHARK_RUNTIME_CHECK(inputs.size() == labels.size(),"Number of inputs and number of labels must agree");
	return createLabeledData(
		createDataFromRange(inputs, maximumBatchSize),
		createDataFromRange(labels, maximumBatchSize)
	);
}

/// \brief creates a labeled data object from two ranges, representing inputs and labels
template<class RangeI, class RangeL>
LabeledData<
	typename RangeI::value_type,
	typename RangeL::value_type
> createLabeledDataFromRange(RangeI const& inputs, RangeL const& labels, std::vector<std::size_t> const& partitioning){
	SHARK_RUNTIME_CHECK(inputs.size() == labels.size(),"Number of inputs and number of labels must agree");
	return createLabeledData(
		createDataFromRange(inputs, partitioning),
		createDataFromRange(labels, partitioning)
	);
}

//////////////ALTERNATIVE VIEWS OF A DATASET

/// \brief Creates a DataView from a Data object.
///
/// This is just a helper function to omit the actual type of the view
///
/// \param set the dataset from which to create the view
template<class DatasetType>
DataView<typename std::remove_reference<DatasetType>::type >  elements(DatasetType&& set){
	return DataView<typename std::remove_reference<DatasetType>::type>(std::forward<DatasetType>(set));
}

/// \brief Creates a Generator from a dataset.
///
/// The generator generates an infinite sequence of data by picking a batch at random.
/// Note that until Generator is destroyed, the supplied data set is shared.
/// Generators can use caching which allows generating batche sin parallel
/// using the global ThreadPool. This is not useful if only batches from the dataset are returned.
/// However, if another expensive operation is bperformed via a transform -
/// for example moving a batch to GPU, preprocessing or data augmentation, caching is helpful.
///
/// \param set the dataset from which to create the generator
/// \param cacheSize how many elements should be cached. default is 0.
template<class DatasetType>
Generator<typename DatasetType::element_type >  generator(DatasetType const& set, std::size_t cacheSize = 0){
	auto gen = [set]() -> typename DatasetType::value_type{
		std::size_t i = random::discrete(random::globalRng(), std::size_t(0), set.size() -1 );
		return set[i];
	};
	
	return Generator<typename DatasetType::element_type >(gen, set.shape(), cacheSize);
}




// FUNCTIONS FOR DIMENSIONALITY


///\brief Return the number of classes of a set of class labels with unsigned int label encoding
inline unsigned int numberOfClasses(Data<unsigned int> const& labels){
	unsigned int classes = 0;
	for(std::size_t i = 0; i != labels.size(); ++i){
		classes = std::max(classes,*std::max_element(labels[i].begin(),labels[i].end()));
	}
	return classes+1;
}

///\brief Returns the number of members of each class in the dataset.
inline std::vector<std::size_t> classSizes(Data<unsigned int> const& labels){
	std::vector<std::size_t> classCounts(numberOfClasses(labels),0u);
	for(std::size_t i = 0; i != labels.size(); ++i){
		for(unsigned int elem: labels[i]){
			classCounts[elem]++;
		}
	}
	return classCounts;
}

///\brief  Return the dimensionality of a  dataset.
template <class InputType>
std::size_t dataDimension(Data<InputType> const& dataset){
	SHARK_ASSERT(dataset.numberOfElements() > 0);
	return dataset.shape().numElements();
}

///\brief  Return the input dimensionality of a labeled dataset.
template <class InputType, class LabelType>
std::size_t inputDimension(LabeledData<InputType, LabelType> const& dataset){
	return dataDimension(dataset.inputs());
}

///\brief  Return the label/output dimensionality of a labeled dataset.
template <class InputType, class LabelType>
std::size_t labelDimension(LabeledData<InputType, LabelType> const& dataset){
	return dataDimension(dataset.labels());
}
///\brief Return the number of classes (highest label value +1) of a classification dataset with unsigned int label encoding
template <class InputType>
std::size_t numberOfClasses(LabeledData<InputType, unsigned int> const& dataset){
	return numberOfClasses(dataset.labels());
}
///\brief Returns the number of members of each class in the dataset.
template<class InputType, class LabelType>
inline std::vector<std::size_t> classSizes(LabeledData<InputType, LabelType> const& dataset){
	return classSizes(dataset.labels());
}

///\brief Transforms a dataset using a Functor f and returns the transformed result.
///
/// \param data The dataset to transform
/// \param f the function that is applied element by element
/// \param shape the resulting shape of the transformation
template<class T, class Functor>
Data<typename detail::TransformedBatchElement<Functor,typename Batch<T>::type>::element_type >
transform(
	Data<T> const& data, Functor f,
	typename detail::TransformedBatchElement<Functor,typename Batch<T>::type>::shape_type const& shape
){
	typedef typename detail::TransformedBatchElement<Functor,typename Batch<T>::type>::element_type ResultType;
	Data<ResultType> result(data.getPartitioning(), shape);
	threading::transform(data, result,
		[f](typename Data<T>::const_reference input ){
			return transformBatch(input, f);
		}, threading::globalThreadPool());
	return result;
}
///\brief Transforms the inputs of a dataset using a Functor f and returns the transformed result.
///
/// \param data The dataset to transform
/// \param f the function that is applied element by element
/// \param shape the resulting shape of the transformation
template<class I, class L,  class Functor>
auto transformInputs(
	LabeledData<I,L> const& data, Functor const& f, 
	typename  detail::TransformedBatchElement<Functor,typename Batch<I>::type>::shape_type const& shape)
->decltype(createLabeledData(transform(data.inputs(),f, shape),data.labels())){
	return createLabeledData(transform(data.inputs(),f, shape),data.labels());
}
///\brief Transforms the labels of a dataset using a Functor f and returns the transformed result.
///
/// \param data The dataset to transform
/// \param f the function that is applied element by element
/// \param shape the resulting shape of the transformation
template<class I, class L, class Functor>
auto transformLabels(
	LabeledData<I,L> const& data, Functor const& f, 
	typename  detail::TransformedBatchElement<Functor,typename Batch<L>::type>::shape_type const& shape)
->decltype(createLabeledData(data.inputs(), transform(data.labels(), f, shape))){
	return createLabeledData(data.inputs(), transform(data.labels(), f, shape));
}

///\brief Creates a copy of a dataset selecting only a certain set of features.
template<class T, class FeatureSet>
Data<blas::vector<T> > selectFeatures(Data<blas::vector<T> > const& data,FeatureSet const& features){
	auto select = [&](blas::matrix<T> const& input){
		blas::matrix<T> output(input.size1(),features.size());
		for(std::size_t i = 0; i != input.size1(); ++i){
			for(std::size_t j = 0; j != features.size(); ++j){
				output(i,j) = input(i,features[j]);
			}
		}
		return output;
	};
	return transform(data,select, {features.size()});
}

template<class T, class L, class FeatureSet>
LabeledData<blas::vector<T> ,L> selectInputFeatures(LabeledData<blas::vector<T> ,L> const& data,FeatureSet const& features){
	return createLabeledData(selectFeatures(data.inputs(),features), data.labels());
}



/// \brief Removes the last part of a given dataset and returns a new split containing the removed elements
///
/// For this operation, the dataset is not allowed to be shared.
/// \brief data The dataset which should be splited
/// \brief index the first element to be split
/// \returns the  set which contains the splitd element (right part of the given set)
template<class DatasetT>
DatasetT splitAtElement(DatasetT& data, std::size_t elementIndex){
	SIZE_CHECK(elementIndex<=data.numberOfElements());

	std::size_t batchPos = 0;
	std::size_t batchStart = 0;
	while(batchStart + batchSize(data[batchPos]) < elementIndex){
		batchStart += batchSize(data[batchPos]);
		++batchPos;
	};
	std::size_t splitPoint = elementIndex-batchStart;
	if(splitPoint != 0){
		data.splitBatch(batchPos,splitPoint);
		++batchPos;
	}

	return data.splice(batchPos);
}

/// \brief Construct a binary (two-class) one-versus-rest problem from a multi-class problem.
///
/// \par
/// The function returns a new LabeledData object. The input part
/// coincides with the multi-class data, but the label part is replaced
/// with binary labels 0 and 1. All instances of the given class
/// (parameter oneClass) get a label of one, all others are assigned a
/// label of zero.
template<class I>
LabeledData<I,unsigned int> oneVersusRestProblem(
	LabeledData<I,unsigned int>const& data,
	unsigned int oneClass
){
	return transformLabels(data, [=](unsigned int label){return (unsigned int)(label == oneClass);}, {2});
}


///\brief reorders the dataset such, that points are grouped by labels
///
/// The elements are not only reordered but the batches are also resized such, that every batch
/// only contains elements of one class. This method must be used in order to use binarySubproblem.
template<class I>
void repartitionByClass(LabeledData<I,unsigned int>& data,std::size_t batchSize = constants::DefaultBatchSize){
	std::vector<std::size_t > classCounts = classSizes(data);
	std::vector<std::size_t > partitioning;//new, optimal partitioning of the data according to the batch sizes
	std::vector<std::size_t > classStart;//at which batch the elements of the class are starting
	detail::batchPartitioning(classCounts, classStart, partitioning, batchSize);

	std::vector<std::size_t> classIndex(classCounts.size(),0);
	for(std::size_t i = 1; i != classIndex.size();++i){
		classIndex[i] = classIndex[i-1] + classCounts[i-1];
	}
	std::vector<std::size_t> elemIndex(data.numberOfElements(), 0); 
	std::size_t index = 0;
	for (auto const& elem: elements(data)){
		std::size_t c = elem.label;
		elemIndex[classIndex[c] ] = index;
		++index;
		++classIndex[c];
	}
	
	data = toDataset(subset(elements(data), elemIndex),partitioning);
}

/// \brief Extract a binary problem from a class-sorted dataset
///
/// This function is mostly interesting for one-versus-one multiclass approaches.
/// We assume that the dataset is organized such that each batch contains only
/// labels of the same class. It picks up all batches with the desired classes and transforms their labels to 0 and 1.
/// This organization cna be performed via a call to repartitionByClass.
///
///\param data the dataset the binary problem is extracted from
///\param zeroClass the class that is transformed to zero-labels in the binary problems
///\param oneClass the class that is transformed to one-labels in the binary problems
template<class I>
LabeledData<I,unsigned int> binarySubProblem(
	LabeledData<I,unsigned int>const& data,
	unsigned int zeroClass,
	unsigned int oneClass
){
	std::vector<std::size_t> indexSet;
	
	bool foundZero = false;
	bool foundOne = false;
	for(std::size_t b = 0; b != data.size(); ++b){
		unsigned int label = data[b].label(0);
		if(label == zeroClass || label == oneClass){
			indexSet.push_back(b);
			foundZero |= (label == zeroClass);
			foundOne |= (label == oneClass);
		}
	}
	SHARK_RUNTIME_CHECK(foundZero, "First class does not exist");
	SHARK_RUNTIME_CHECK(foundOne, "Second class does not exist");

	return oneVersusRestProblem(data.indexedSubset(indexSet), oneClass);
}



template <typename T>
blas::vector<T> getColumn(Data<blas::vector<T> > const& data, std::size_t columnID) {
	SHARK_ASSERT(dataDimension(data) > columnID);
	blas::vector<T> newColumn(data.numberOfElements());
	std::size_t start = 0;
	for(blas::matrix<T> const& batch: data){
		std::size_t end = start + batch.size1();
		noalias(subrange(newColumn,start, end)) = column(batch, columnID);
		start = end;
	}
	return newColumn;
}

template <typename T>
void setColumn(Data<blas::vector<T>>& data, std::size_t columnID, blas::vector<T> const& newColumn) {
	SHARK_ASSERT(dataDimension(data) > columnID);
	SHARK_ASSERT(data.numberOfElements() == newColumn.size());
	std::size_t start = 0;
	for(blas::matrix<T>& batch: data){
		std::size_t end = start + batch.size1();
		noalias(column(batch, columnID)) = subrange(newColumn,start, end);
		start = end;
	}
}

/** @*/
}

#endif
