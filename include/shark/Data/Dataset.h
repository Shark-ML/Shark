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
#include <shark/Core/OpenMP.h>
#include <shark/Core/utility/functional.h>
#include <boost/iterator/transform_iterator.hpp>
#include <shark/Core/Random.h>
#include <shark/Core/Shape.h>
#include "Impl/Dataset.inl"
#include "Impl/InputLabelPair.h"
#include <shark/Data/DataView.h>

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
class Data : public ISerializable
{
protected:
	typedef detail::SharedContainer<Type> Container;

	Container m_data;///< data
	typename Batch<Type>::shape_type m_shape;///< shape of a datapoint
public:
	typedef Type element_type;
	typedef typename Batch<Type>::shape_type shape_type;

	typedef typename Container::BatchType batch_type;
	typedef batch_type& batch_reference;
	typedef batch_type const& const_batch_reference;

	typedef std::vector<std::size_t> IndexSet;

	/// \brief Two containers compare equal if they share the same data.
	template <class T> bool operator == (const Data<T>& rhs) {
		return (m_data == rhs.m_data);
	}

	/// \brief Two containers compare unequal if they don't share the same data.
	template <class T> bool operator != (const Data<T>& rhs) {
		return (! (*this == rhs));
	}

	template <class InputT, class LabelT> friend class LabeledData;

	// RANGES
	typedef detail::BatchRange<Data<Type> > batch_range;
	typedef detail::BatchRange<Data<Type> const> const_batch_range;

	///\brief Returns the range of batches.
	const_batch_range batches()const{
		return const_batch_range(this);
	}
	///\brief Returns the range of batches.
	batch_range batches(){
		return batch_range(this);
	}

	///\brief Returns the number of batches of the set.
	std::size_t numberOfBatches() const{
		return m_data.size();
	}
	///\brief Returns the total number of elements.
	std::size_t numberOfElements() const{
		return m_data.numberOfElements();
	}
	
	
	///\brief Returns the shape of the elements in the dataset.
	shape_type const& shape() const{
		return m_shape;
	}
	
	///\brief Returns the shape of the elements in the dataset.
	shape_type& shape(){
		return m_shape;
	}

	///\brief Check whether the set is empty.
	bool empty() const{
		return m_data.empty();
	}

	// BATCH ACCESS
	batch_reference batch(std::size_t i){
		return *(m_data.begin()+i);
	}
	const_batch_reference batch(std::size_t i) const{
		return *(m_data.begin()+i);
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
	explicit Data(std::size_t numElements, shape_type const& shape, std::size_t batchSize = constants::DefaultBatchSize)
	: m_data( detail::numberOfBatches(numElements, batchSize)), m_shape(shape){
		auto batches = detail::optimalBatchSizes(numElements, batchSize);
		for(std::size_t i = 0; i != batches.size(); ++i){
			m_data.batch(i) = Batch<Type>::createBatchFromShape(shape, batches[i]);
		}
	}
	
	///\brief Constructs a set with a given shape and a chosen partitioning
	///
	/// For each element in partitioning, a batch is created with the given size
	///
	///@param partitioning batch sizes of the dataset
	///@param shape The shape of the elements to create
	explicit Data(std::vector<std::size_t> const& partitioning, shape_type const& shape)
	: m_data( partitioning.size()), m_shape(shape){
		for(std::size_t i = 0; i != numberOfBatches(); ++i){
			m_data.batch(i) = Batch<Type>::createBatchFromShape(shape, partitioning[i]);
		}
	}

	// MISC

	void read(InArchive& archive){
		archive >> m_data;
		archive >> m_shape;
	}

	void write(OutArchive& archive) const{
		archive << m_data;
		archive << m_shape;
	}
	///\brief This method makes the vector independent of all siblings and parents.
	virtual void makeIndependent(){
		m_data.makeIndependent();
	}


	// METHODS TO ALTER BATCH STRUCTURE
	void splitBatch(std::size_t batch, std::size_t elementIndex){
		m_data.splitBatch(m_data.begin()+batch,elementIndex);
	}

	///\brief Splits the container into two independent parts. The front part remains in the container, the back part is returned.
	///
	///Order of elements remain unchanged. The SharedVector is not allowed to be shared for
	///this to work.
	Data splice(std::size_t batch){
		Data right;
		right.m_data = m_data.splice(m_data.begin()+batch);
		right.m_shape = m_shape;
		return right;
	}

	/// \brief Appends the contents of another data object to the end
	///
	/// The batches are not copied but now referenced from both datasets. Thus changing the appended
	/// dataset might change this one as well.
	void append(Data const& other){
		m_data.append(other.m_data);
	}
	
	void push_back(const_batch_reference batch){
		m_data.push_back(batch);
	}
	
	/// \brief Creates a vector with the batch sizes of every batch.
	///
	/// This method can be used together with to ensure
	/// that two datasets have the same batch structure.
	std::vector<std::size_t> getPartitioning()const{
		return m_data.getPartitioning();
	}

	// SUBSETS
	Data indexedSubset(IndexSet const& indices) const{
		Data subset;
		subset.m_data = Container(m_data,indices);
		subset.m_shape = m_shape;
		return subset;
	}

	friend void swap(Data& a, Data& b){
		swap(a.m_data,b.m_data);
		std::swap(a.m_shape,b.m_shape);
	}
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
class LabeledData : public ISerializable
{
public:
	typedef InputT InputType;
	typedef LabelT LabelType;
	typedef Data<InputT> InputContainer;
	typedef Data<LabelT> LabelContainer;
	typedef typename InputContainer::IndexSet IndexSet;

	// TYPEDEFS FOR PAIRS
	typedef InputLabelBatch<
		typename Batch<InputType>::type,
		typename Batch<LabelType>::type
	> batch_type;

	typedef InputLabelPair<InputType,LabelType> element_type;
	typedef typename Batch<element_type>::shape_type shape_type;

	// TYPEDEFS FOR REFERENCES
	typedef InputLabelBatch<
		typename Batch<InputType>::type&,
		typename Batch<LabelType>::type&
	> batch_reference;
	typedef InputLabelBatch<
		typename Batch<InputType>::type const&,
		typename Batch<LabelType>::type const&
	> const_batch_reference;

	typedef detail::BatchRange<LabeledData<InputType,LabelType> > batch_range;
	typedef detail::BatchRange<LabeledData<InputType,LabelType> const> const_batch_range;
	
	///\brief Returns the range of batches.
	const_batch_range batches()const{
		return const_batch_range(this);
	}
	///\brief Returns the range of batches.
	batch_range batches(){
		return batch_range(this);
	}

	///\brief Returns the number of batches of the set.
	std::size_t numberOfBatches() const{
		return m_data.numberOfBatches();
	}
	///\brief Returns the total number of elements.
	std::size_t numberOfElements() const{
		return m_data.numberOfElements();
	}

	///\brief Check whether the set is empty.
	bool empty() const{
		return m_data.empty();
	}

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

	// CONSTRUCTORS

	///\brief Empty data set.
	LabeledData()
	{}

	///\brief Constructs a set holding a specific number of elements of a given shape.
	///
	/// to create a dataset with 100 dimensional inputs and 8 classes, write
	/// LabeledData<RealVector, unsigned int> data(numPoints,{{100}.{8}})
	/// Optionally the desired batch size can be set.
	///
	///@param numElements number of data points stored in the dataset
	///@param shape The shape of the datapoints
	///@param batchSize the size of the batches. if this is 0, the size is unlimited
	explicit LabeledData(std::size_t numElements, shape_type const& shape, std::size_t batchSize = constants::DefaultBatchSize)
	: m_data(numElements, shape.input, batchSize), m_label(numElements, shape.label, batchSize){}
	
	///\brief Constructs a set with a given shape and a chosen partitioning
	///
	/// For each element in partitioning, a batch is created with the given size
	///
	///@param partitioning batch sizes of the dataset
	///@param shape The shape of the elements to create
	explicit LabeledData(std::vector<std::size_t> const& partitioning, shape_type const& shape)
	: m_data( partitioning, shape.input), m_label(partitioning,shape.label){}

	///\brief Construction from data.
	///
	/// Beware that when calling this constructor the organization of batches must be equal in both
	/// containers. This Constructor will not split the data!
	LabeledData(Data<InputType> const& inputs, Data<LabelType> const& labels)
	: m_data(inputs), m_label(labels){
		SHARK_RUNTIME_CHECK(inputs.numberOfElements() == labels.numberOfElements(), "number of inputs and number of labels must agree");
		for(std::size_t i  = 0; i != inputs.numberOfBatches(); ++i){
			SHARK_RUNTIME_CHECK(batchSize(inputs.batch(i)) == batchSize(labels.batch(i)), "batch sizes of inputs and labels must agree");
		}
	}

	// BATCH ACCESS
	batch_reference batch(std::size_t i){
		return batch_reference(m_data.batch(i),m_label.batch(i));
	}
	const_batch_reference batch(std::size_t i) const{
		return const_batch_reference(m_data.batch(i),m_label.batch(i));
	}
	
	///\brief Returns the Shape of the inputs.
	typename Batch<InputType>::shape_type const& inputShape() const{
		return m_data.shape();
	}
	
	///\brief Returns the Shape of the inputs.
	typename Batch<InputType>::shape_type& inputShape(){
		return m_data.shape();
	}
	
	///\brief Returns the Shape of the labels.
	typename Batch<LabelType>::shape_type const& labelShape() const{
		return m_label.shape();
	}
	
	///\brief Returns the Shape of the labels.
	typename Batch<LabelType>::shape_type& labelShape(){
		return m_label.shape();
	}
	
	///\brief Returns the Shape of the elements
	shape_type shape() const{
		return {inputShape(), labelShape()};
	}

	// MISC

	/// from ISerializable
	void read(InArchive& archive){
		archive & m_data;
		archive & m_label;
	}

	/// from ISerializable
	void write(OutArchive& archive) const{
		archive & m_data;
		archive & m_label;
	}

	///\brief This method makes the vector independent of all siblings and parents.
	virtual void makeIndependent(){
		m_label.makeIndependent();
		m_data.makeIndependent();
	}

	void splitBatch(std::size_t batch, std::size_t elementIndex){
		m_data.splitBatch(batch,elementIndex);
		m_label.splitBatch(batch,elementIndex);
	}

	///\brief Splits the container into two independent parts. The left part remains in the container, the right is stored as return type
	///
	///Order of elements remain unchanged. The SharedVector is not allowed to be shared for
	///this to work.
	LabeledData splice(std::size_t batch){
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
		const_batch_reference batch
	){
		push_back(batch.input,batch.label);
	}

	
	/// \brief Creates a vector with the batch sizes of every batch.
	///
	/// This method can be used to ensure
	/// that two datasets have the same batch structure.
	std::vector<std::size_t> getPartitioning()const{
		return m_data.getPartitioning();
	}

	friend void swap(LabeledData& a, LabeledData& b){
		swap(a.m_data,b.m_data);
		swap(a.m_label,b.m_label);
	}


	// SUBSETS

	///\brief Fill in the subset defined by the list of indices.
	LabeledData indexedSubset(IndexSet const& indices) const{
		return LabeledData(m_data.indexedSubset(indices),m_label.indexedSubset(indices));
	}
protected:
	InputContainer m_data;               /// point data
	LabelContainer m_label;		/// label data
};

/// specialized template for classification with unsigned int labels
typedef LabeledData<RealVector, unsigned int> ClassificationDataset;

/// specialized template for regression with RealVector labels
typedef LabeledData<RealVector, RealVector> RegressionDataset;

/// specialized template for classification with unsigned int labels and sparse data
typedef LabeledData<CompressedRealVector, unsigned int> CompressedClassificationDataset;

template<class Functor, class T>
struct TransformedData{
	typedef Data<typename detail::TransformedDataElement<Functor,T>::type > type;
};

namespace detail{
template<class T>
struct InferShape{
	static Shape infer(T const&){return {};}
};

template<class T>
struct InferShape<Data<T> >{
	static typename Batch<T>::shape_type infer(Data<T> const&){return {};}
};

template<class T>
struct InferShape<Data<blas::vector<T> > >{
	static Shape infer(Data<blas::vector<T> > const& f){
		return {f.batch(0).size2()};
	}
};

template<class T>
struct InferShape<Data<blas::compressed_vector<T> > >{
	static Shape infer(Data<blas::compressed_vector<T> > const& f){
		return {f.batch(0).size2()};
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

/// \brief creates a data object from a range of elements
template<class Range>
Data<typename Range::value_type>
createDataFromRange(Range const& inputs, std::size_t maximumBatchSize = constants::DefaultBatchSize){
	typedef typename Range::value_type value_type;

	auto shape = Batch<value_type>::inferShape(inputs.begin(), inputs.end());
	Data<value_type> data(inputs.size(), shape, maximumBatchSize);

	//now create the batches taking the remainder into account
	auto start= inputs.begin();
	for(std::size_t i = 0; i != data.numberOfBatches(); ++i){
		std::size_t size = batchSize(data.batch(i));
		data.batch(i) = createBatch<value_type>(start,start+size);
		start = start+size;
	}
	return data;
}

/// \brief creates a labeled data object from two ranges, representing inputs and labels
template<class RangeI, class RangeL>
LabeledData<
	typename RangeI::value_type,
	typename RangeL::value_type
> createLabeledDataFromRange(RangeI const& inputs, RangeL const& labels, std::size_t maximumBatchSize = 0){
	SHARK_RUNTIME_CHECK(inputs.size() == labels.size(),"Number of inputs and number of labels must agree");
	typedef typename RangeI::value_type Input;
	typedef typename RangeL::value_type Label;

	return LabeledData<Input,Label>(
		createDataFromRange(inputs, maximumBatchSize),
		createDataFromRange(labels, maximumBatchSize)
	);
}



// FUNCTIONS FOR DIMENSIONALITY


///\brief Return the number of classes of a set of class labels with unsigned int label encoding
inline unsigned int numberOfClasses(Data<unsigned int> const& labels){
	unsigned int classes = 0;
	for(std::size_t i = 0; i != labels.numberOfBatches(); ++i){
		classes = std::max(classes,*std::max_element(labels.batch(i).begin(),labels.batch(i).end()));
	}
	return classes+1;
}

///\brief Returns the number of members of each class in the dataset.
inline std::vector<std::size_t> classSizes(Data<unsigned int> const& labels){
	std::vector<std::size_t> classCounts(numberOfClasses(labels),0u);
	for(std::size_t i = 0; i != labels.numberOfBatches(); ++i){
		for(unsigned int elem: labels.batch(i)){
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

// TRANSFORMATION
///\brief Transforms a dataset using a Functor f and returns the transformed result.
///
/// this version is used, when the Functor supports only element-by-element transformations
template<class T,class Functor>
typename boost::lazy_disable_if<
	CanBeCalled<Functor,typename Data<T>::batch_type>,
	TransformedData<Functor,T>
>::type
transform(Data<T> const& data, Functor f){
	typedef typename detail::TransformedDataElement<Functor,T>::type ResultType;
	int batches = (int) data.numberOfBatches();
	Data<ResultType> result(data.getPartitioning(), typename Data<ResultType>::shape_type());//TODO HACK!!!!
	SHARK_PARALLEL_FOR(int i = 0; i < batches; ++i){
		typedef BatchIterator<typename Batch<T>::type const> Iterator;
		
		result.batch(i) = createBatch<ResultType>(
			boost::make_transform_iterator(Iterator(data.batch(i),0), f),
			boost::make_transform_iterator(Iterator(data.batch(i), batchSize(data.batch(i))), f)
		);
	}
	result.shape() = detail::InferShape<Data<ResultType> >::infer(result);
	return result;
}

///\brief Transforms a dataset using a Functor f and returns the transformed result.
///
/// this version is used, when the Functor supports batch-by-batch transformations
template<class T,class Functor>
typename boost::lazy_enable_if<
	CanBeCalled<Functor,typename Data<T>::batch_type>,
	TransformedData<Functor,T>
>::type
transform(Data<T> const& data, Functor const& f){
	typedef typename detail::TransformedDataElement<Functor,T>::type ResultType;
	int batches = (int) data.numberOfBatches();
	Data<ResultType> result(data.getPartitioning(), f.outputShape());
	SHARK_PARALLEL_FOR(int i = 0; i < batches; ++i)
		result.batch(i)= f(data.batch(i));
	return result;
}

///\brief Transforms the inputs of a dataset and return the transformed result.
template<class I,class L, class Functor>
LabeledData<typename detail::TransformedDataElement<Functor,I >::type, L >
transformInputs(LabeledData<I,L> const& data, Functor const& f){
	typedef LabeledData<typename detail::TransformedDataElement<Functor,I>::type,L > DatasetType;
	return DatasetType(transform(data.inputs(),f),data.labels());
}
///\brief Transforms the labels of a dataset and returns the transformed result.
template<class I,class L, class Functor>
LabeledData<I,typename detail::TransformedDataElement<Functor,L >::type >
transformLabels(LabeledData<I,L> const& data, Functor const& f){
	typedef LabeledData<I,typename detail::TransformedDataElement<Functor,L>::type > DatasetType;
	return DatasetType(data.inputs(),transform(data.labels(),f));
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
	return transform(data,select);
}

template<class T, class FeatureSet>
LabeledData<RealVector,T> selectInputFeatures(LabeledData<RealVector,T> const& data,FeatureSet const& features){
	return LabeledData<RealVector,T>(selectFeatures(data.inputs(),features), data.labels());
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
	while(batchStart + batchSize(data.batch(batchPos)) < elementIndex){
		batchStart += batchSize(data.batch(batchPos));
		++batchPos;
	};
	std::size_t splitPoint = elementIndex-batchStart;
	if(splitPoint != 0){
		data.splitBatch(batchPos,splitPoint);
		++batchPos;
	}

	return data.splice(batchPos);
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

template<class I>
LabeledData<I,unsigned int> binarySubProblem(
	LabeledData<I,unsigned int>const& data,
	unsigned int zeroClass,
	unsigned int oneClass
){
	std::vector<std::size_t> indexSet;
	std::size_t smaller = std::min(zeroClass,oneClass);
	std::size_t bigger = std::max(zeroClass,oneClass);
	std::size_t numBatches = data.numberOfBatches();

	//find first class
	std::size_t start= 0;
	for(;start != numBatches && getBatchElement(data.batch(start),0).label != smaller;++start);
	SHARK_RUNTIME_CHECK(start != numBatches, "First class does not exist");

	//copy batch indices of first class
	for(;start != numBatches && getBatchElement(data.batch(start),0).label == smaller; ++start)
		indexSet.push_back(start);

	//find second class

	for(;start != numBatches && getBatchElement(data.batch(start),0).label != bigger;++start);
	SHARK_RUNTIME_CHECK(start != numBatches, "Second class does not exist");

	//copy batch indices of second class
	for(;start != numBatches && getBatchElement(data.batch(start),0).label == bigger; ++start)
		indexSet.push_back(start);

	return transformLabels(data.indexedSubset(indexSet), [=](unsigned int label){return (unsigned int)(label == oneClass);});
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
	return transformLabels(data, [=](unsigned int label){return (unsigned int)(label == oneClass);});
}

template <typename T>
blas::vector<T> getColumn(Data<blas::vector<T> > const& data, std::size_t columnID) {
	SHARK_ASSERT(dataDimension(data) > columnID);
	blas::vector<T> newColumn(data.numberOfElements());
	std::size_t start = 0;
	for(blas::matrix<T> const& batch: data.batches()){
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
	for(blas::matrix<T>& batch: data.batches()){
		std::size_t end = start + batch.size1();
		noalias(column(batch, columnID)) = subrange(newColumn,start, end);
		start = end;
	}
}

/** @*/
}

#endif
