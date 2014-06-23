//===========================================================================
/*!
 * 
 *
 * \brief       Weighted data sets for (un-)supervised learning.
 * 
 * 
 * \par
 * This file provides containers for data used by the models, loss
 * functions, and learning algorithms (trainers). The reason for
 * dedicated containers of this type is that data often need to be
 * split into subsets, such as training and test data, or folds in
 * cross-validation. The containers in this file provide memory
 * efficient mechanisms for managing and providing such subsets.
 * The speciality of these containers are that they are weighted.
 * 
 * 
 *
 * \author    O. Krause
 * \date       2014
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
//===========================================================================

#ifndef SHARK_DATA_WEIGHTED_DATASET_H
#define SHARK_DATA_WEIGHTED_DATASET_H

#include <shark/Data/Dataset.h>
namespace shark {
	
namespace detail{
template <class DataContainerT>
class BaseWeightedDataset : public ISerializable
{
private:
	typedef BaseWeightedDataset<DataContainerT> self_type;
public:
	typedef typename DataContainerT::element_type DataType;
	typedef double WeightType;
	typedef DataContainerT DataContainer;
	typedef Data<WeightType> WeightContainer;
	typedef typename DataContainer::IndexSet IndexSet;

	// TYPEDEFS fOR PAIRS
	typedef WeightedDataPair<
		DataType,
		WeightType
	> element_type;

	typedef typename Batch<element_type>::type batch_type;

	// TYPEDEFS FOR  RANGES
	typedef typename PairRangeType<
		element_type,
		typename DataContainer::element_range,
		typename WeightContainer::element_range
	>::type element_range;
	typedef typename PairRangeType<
		element_type,
		typename DataContainer::const_element_range,
		typename WeightContainer::const_element_range
	>::type const_element_range;
	typedef typename PairRangeType<
		batch_type,
		typename DataContainer::batch_range,
		typename WeightContainer::batch_range
	>::type batch_range;
	typedef typename PairRangeType<
		batch_type,
		typename DataContainer::const_batch_range,
		typename WeightContainer::const_batch_range
	>::type const_batch_range;

	// TYPEDEFS FOR REFERENCES
	typedef typename boost::range_reference<batch_range>::type batch_reference;
	typedef typename boost::range_reference<const_batch_range>::type const_batch_reference;
	typedef typename boost::range_reference<element_range>::type element_reference;
	typedef typename boost::range_reference<const_element_range>::type const_element_reference;

	///\brief Returns the range of elements.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	const_element_range elements()const{
		return zipPairRange<element_type>(m_data.elements(),m_weights.elements());
	}
	///\brief Returns therange of elements.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	element_range elements(){
		return zipPairRange<element_type>(m_data.elements(),m_weights.elements());
	}

	///\brief Returns the range of batches.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	const_batch_range batches()const{
		return zipPairRange<batch_type>(m_data.batches(),m_weights.batches());
	}
	///\brief Returns the range of batches.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	batch_range batches(){
		return zipPairRange<batch_type>(m_data.batches(),m_weights.batches());
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

	///\brief Access to the stored data points as a separate container.
	DataContainer const& data() const{
		return m_data;
	}
	///\brief Access to the stored data points as a separate container.
	DataContainer& data(){
		return m_data;
	}

	///\brief Access to weights as a separate container.
	WeightContainer const& weights() const{
		return m_weights;
	}
	///\brief Access to weights as a separate container.
	WeightContainer& weights(){
		return m_weights;
	}

	// CONSTRUCTORS

	///\brief Constructs an Empty data set.
	BaseWeightedDataset()
	{}

	///\brief Create an empty set with just the correct number of batches.
	///
	/// The user must initialize the dataset after that by himself.
	BaseWeightedDataset(std::size_t numBatches)
	: m_data(numBatches),m_weights(numBatches)
	{}

	/// \brief Construtor using a single element as blueprint to create a dataset with a specified number of elements.
	///
	/// Optionally the desired batch Size can be set
	///
	///@param size the new size of the container
	///@param element the blueprint element from which to create the Container
	///@param batchSize the size of the batches. if this is 0, the size is unlimited
	BaseWeightedDataset(std::size_t size, element_type const& element, std::size_t batchSize)
	: m_data(size,element.data,batchSize)
	, m_weights(size,element.weight,batchSize)
	{}

	///\brief Construction from data and a dataset rpresnting the weights
	///
	/// Beware that when calling this constructor the organization of batches must be equal in both
	/// containers. This Constructor will not reorganize the data!
	BaseWeightedDataset(DataContainer const& data, Data<WeightType> const& weights)
	: m_data(data), m_weights(weights)
	{
		SHARK_CHECK(data.numberOfElements() == weights.numberOfElements(), "[ BaseWeightedDataset::WeightedUnlabeledData] number of data and number of weights must agree");
#ifndef DNDEBUG
		for(std::size_t i  = 0; i != data.numberOfBatches(); ++i){
			SIZE_CHECK(shark::size(data.batch(i))==shark::size(weights.batch(i)));
		}
#endif
	}
	
	///\brief Construction from data. All points get the same weight assigned
	BaseWeightedDataset(DataContainer const& data, double weight)
	: m_data(data), m_weights(data.numberOfBatches())
	{
		for(std::size_t i = 0; i != numberOfBatches(); ++i){
			std::size_t batchSize = boost::size(m_data.batch(i));
			m_weights.batch(i) = Batch<WeightType>::type(batchSize,weight);
		}
	}
	
	
	// ELEMENT ACCESS
	element_reference element(std::size_t i){
		return element_reference(m_data.element(i),m_weights.element(i));
	}
	const_element_reference element(std::size_t i) const{
		return const_element_reference(m_data.element(i),m_weights.element(i));
	}

	// BATCH ACCESS
	batch_reference batch(std::size_t i){
		return batch_reference(m_data.batch(i),m_weights.batch(i));
	}
	const_batch_reference batch(std::size_t i) const{
		return const_batch_reference(m_data.batch(i),m_weights.batch(i));
	}

	// MISC

	/// from ISerializable
	void read(InArchive& archive){
		archive & m_data;
		archive & m_weights;
	}

	/// from ISerializable
	void write(OutArchive& archive) const{
		archive & m_data;
		archive & m_weights;
	}

	///\brief This method makes the vector independent of all siblings and parents.
	virtual void makeIndependent(){
		m_weights.makeIndependent();
		m_data.makeIndependent();
	}

	///\brief shuffles all elements in the entire dataset (that is, also across the batches)
	virtual void shuffle(){
		DiscreteUniform<Rng::rng_type> uni(Rng::globalRng);
		shark::shuffle(this->elements().begin(),this->elements().end(), uni);
	}

	void splitBatch(std::size_t batch, std::size_t elementIndex){
		m_data.splitBatch(batch,elementIndex);
		m_weights.splitBatch(batch,elementIndex);
	}

	/// \brief Appends the contents of another data object to the end
	///
	/// The batches are not copied but now referenced from both datasets. Thus changing the appended
	/// dataset might change this one as well.
	void append(self_type const& other){
		m_data.append(other.m_data);
		m_weights.append(other.m_weights);
	}


	///\brief Reorders the batch structure in the container to that indicated by the batchSizes vector
	///
	///After the operation the container will contain batchSizes.size() batches with the i-th batch having size batchSize[i].
	///However the sum of all batch sizes must be equal to the current number of elements
	template<class Range>
	void repartition(Range const& batchSizes){
		m_data.repartition(batchSizes);
		m_weights.repartition(batchSizes);
	}
	
	/// \brief Creates a vector with the batch sizes of every batch.
	///
	/// This method can be used together with repartition to ensure
	/// that two datasets have the same batch structure.
	std::vector<std::size_t> getPartitioning()const{
		return m_data.getPartitioning();
	}

	friend void swap( self_type& a, self_type& b){
		swap(a.m_data,b.m_data);
		swap(a.m_weights,b.m_weights);
	}


	// SUBSETS

	///\brief Fill in the subset defined by the list of indices.
	void indexedSubset(IndexSet const& indices, self_type& subset) const{
		m_data.indexedSubset(indices,subset.m_data);
		m_weights.indexedSubset(indices,subset.m_weights);
	}

	///\brief Fill in the subset defined by the list of indices as well as its complement.
	void indexedSubset(IndexSet const& indices, self_type& subset, self_type& complement)const{
		IndexSet comp;
		detail::complement(indices,m_data.numberOfBatches(),comp);
		m_data.indexedSubset(indices,subset.m_data);
		m_weights.indexedSubset(indices,subset.m_weights);
		m_data.indexedSubset(comp,complement.m_data);
		m_weights.indexedSubset(comp,complement.m_weights);
	}
private:
	DataContainer m_data;               /// point data
	WeightContainer m_weights; /// weight data
};

}

///
/// \brief Weighted data set for unsupervised learning
///
/// The WeightedUnlabeledData class extends UnlabeledData for the
/// representation of data. In addition it holds and provides access to the corresponding weights.
///
/// WeightedUnlabeledData tries to mimic the underlying data as pairs of data points and weights.
/// this means that when accessing a batch by calling batch(i) or choosing one of the iterators
/// one access the input batch by batch(i).data and the weights by batch(i).weight
///
///this also holds true for single element access using operator(). Be aware, that direct access to element is
///a linear time operation. So it is not advisable to iterate over the elements, but instead iterate over the batches.
template <class DataT>
class WeightedUnlabeledData : public detail::BaseWeightedDataset <UnlabeledData<DataT> >
{
private:
	typedef WeightedUnlabeledData<DataT> self_type;
	typedef detail::BaseWeightedDataset <UnlabeledData<DataT> > base_type;
public:
	using base_type::data;
	using base_type::weights;
	typedef typename base_type::DataType DataType;
	typedef typename base_type::WeightType WeightType;
	typedef typename base_type::element_type element_type;
	typedef DataT InputType;

	BOOST_STATIC_CONSTANT(std::size_t, DefaultBatchSize = UnlabeledData<DataT>::DefaultBatchSize);

	// CONSTRUCTORS

	///\brief Empty data set.
	WeightedUnlabeledData()
	{}

	///\brief Create an empty set with just the correct number of batches.
	///
	/// The user must initialize the dataset after that by himself.
	WeightedUnlabeledData(std::size_t numBatches)
	: base_type(numBatches)
	{}

	/// \brief Construtor using a single element as blueprint to create a dataset with a specified number of elements.
	///
	/// Optionally the desired batch Size can be set
	///
	///@param size the new size of the container
	///@param element the blueprint element from which to create the Container
	///@param batchSize the size of the batches. if this is 0, the size is unlimited
	WeightedUnlabeledData(std::size_t size, element_type const& element, std::size_t batchSize = DefaultBatchSize)
	: base_type(size,element,batchSize){}

	///\brief Construction from data.
	///
	/// Beware that when calling this constructor the organization of batches must be equal in both
	/// containers. This Constructor will not reorganize the data!
	WeightedUnlabeledData(UnlabeledData<DataType> const& data, Data<WeightType> const& weights)
	: base_type(data,weights)
	{}
		
	///\brief Construction from data and a constant weight for all elements
	WeightedUnlabeledData(UnlabeledData<DataType> const& data, double weight)
	: base_type(data,weight)
	{}
		
	//we additionally add the two below for compatibility with UnlabeledData
		
	///\brief Access to the inputs as a separate container.
	UnlabeledData<DataT> const& inputs() const{
		return data();
	}
	///\brief Access to the inputs as a separate container.
	UnlabeledData<DataT>& inputs(){
		return data();
	}

	///\brief Splits the container into two independent parts. The left part remains in the container, the right is stored as return type
	///
	///Order of elements remain unchanged. The SharedVector is not allowed to be shared for
	///this to work.
	self_type splice(std::size_t batch){
		return self_type(data().splice(batch),weights().splice(batch));
	}

	friend void swap(WeightedUnlabeledData& a, WeightedUnlabeledData& b){
		swap(static_cast<base_type&>(a),static_cast<base_type&>(b));
	}
};

///brief  Outstream of elements for weighted data.
template<class T>
std::ostream &operator << (std::ostream &stream, const WeightedUnlabeledData<T>& d) {
	typedef typename WeightedUnlabeledData<T>::const_element_reference reference;
	typename WeightedUnlabeledData<T>::const_element_range elements = d.elements();
	BOOST_FOREACH(reference elem,elements)
		stream << elem.weight << " [" << elem.data<<"]"<< "\n";
	return stream;
}

/// \brief creates a weighted unweighted data object from two ranges, representing data and weights
template<class DataRange, class WeightRange>
typename boost::disable_if<
	boost::is_arithmetic<WeightRange>,
	WeightedUnlabeledData<
		typename boost::range_value<DataRange>::type
	> 
>::type createUnlabeledDataFromRange(DataRange const& data, WeightRange const& weights, std::size_t batchSize = 0){
	SHARK_CHECK(boost::size(data) == boost::size(weights),
	"[createDataFromRange] number of data points and number of weights must agree");
	typedef typename boost::range_value<DataRange>::type Data;

	if (batchSize == 0)
		batchSize = WeightedUnlabeledData<Data>::DefaultBatchSize;

	return WeightedUnlabeledData<Data>(
		shark::createUnlabeledDataFromRange(data,batchSize),
		createDataFromRange(weights,batchSize)
	);
}


///
/// \brief Weighted data set for supervised learning
///
/// The WeightedLabeledData class extends LabeledData for the
/// representation of data. In addition it holds and provides access to the corresponding weights.
///
/// WeightedLabeledData tries to mimic the underlying data as pairs of data tuples(input,label) and weights.
/// this means that when accessing a batch by calling batch(i) or choosing one of the iterators
/// one access the databatch by batch(i).data and the weights by batch(i).weight. to access the points and labels
/// use batch(i).data.input and batch(i).data.label
///
///this also holds true for single element access using operator(). Be aware, that direct access to element is
///a linear time operation. So it is not advisable to iterate over the elements, but instead iterate over the batches.
///
/// It is possible to gains everal views on the set. one can either get access to inputs, labels and weights separately
/// or gain access to the unweighted dataset of inputs and labels. Additionally the sets support on-the-fly creation
/// of the (inputs,weights) subset for unsupervised weighted learning
template <class InputT, class LabelT>
class WeightedLabeledData : public detail::BaseWeightedDataset <LabeledData<InputT,LabelT> >
{
private:
	typedef WeightedLabeledData<InputT,LabelT> self_type;
	typedef detail::BaseWeightedDataset <LabeledData<InputT,LabelT> > base_type;
public:
	typedef typename base_type::DataType DataType;
	typedef typename base_type::WeightType WeightType;
	typedef InputT InputType;
	typedef LabelT LabelType;
	typedef typename base_type::element_type element_type;

	using base_type::data;
	using base_type::weights;

	BOOST_STATIC_CONSTANT(std::size_t, DefaultBatchSize = (LabeledData<InputT,LabelT>::DefaultBatchSize));

	// CONSTRUCTORS

	///\brief Empty data set.
	WeightedLabeledData()
	{}

	///\brief Create an empty set with just the correct number of batches.
	///
	/// The user must initialize the dataset after that by himself.
	WeightedLabeledData(std::size_t numBatches)
	: base_type(numBatches)
	{}

	/// \brief Construtor using a single element as blueprint to create a dataset with a specified number of elements.
	///
	/// Optionally the desired batch Size can be set
	///
	///@param size the new size of the container
	///@param element the blueprint element from which to create the Container
	///@param batchSize the size of the batches. if this is 0, the size is unlimited
	WeightedLabeledData(std::size_t size, element_type const& element, std::size_t batchSize = DefaultBatchSize)
	: base_type(size,element,batchSize){}

	///\brief Construction from data.
	///
	/// Beware that when calling this constructor the organization of batches must be equal in both
	/// containers. This Constructor will not reorganize the data!
	WeightedLabeledData(LabeledData<InputType,LabelType> const& data, Data<WeightType> const& weights)
	: base_type(data,weights)
	{}
		
	///\brief Construction from data and a constant weight for all elements
	WeightedLabeledData(LabeledData<InputType,LabelType> const& data, double weight)
	: base_type(data,weight)
	{}
		
	///\brief Access to the inputs as a separate container.
	UnlabeledData<InputType> const& inputs() const{
		return data().inputs();
	}
	///\brief Access to the inputs as a separate container.
	UnlabeledData<InputType>& inputs(){
		return data().inputs();
	}
	
	///\brief Access to the labels as a separate container.
	Data<LabelType> const& labels() const{
		return data().labels();
	}
	///\brief Access to the labels as a separate container.
	Data<LabelType>& labels(){
		return data().labels();
	}
	
	/// \brief Constructs an WeightedUnlabeledData object for the inputs.
	WeightedUnlabeledData<InputType> weightedInputs() const{
		return WeightedUnlabeledData<InputType>(data().inputs(),weights());
	}

	///\brief Splits the container into two independent parts. The left part remains in the container, the right is stored as return type
	///
	///Order of elements remain unchanged. The SharedVector is not allowed to be shared for
	///this to work.
	self_type splice(std::size_t batch){
		return self_type(data().splice(batch),weights().splice(batch));
	}

	friend void swap(self_type& a, self_type& b){
		swap(static_cast<base_type&>(a),static_cast<base_type&>(b));
	}
};

///brief  Outstream of elements for weighted labeled data.
template<class T, class U>
std::ostream &operator << (std::ostream &stream, const WeightedLabeledData<T, U>& d) {
	typedef typename WeightedLabeledData<T, U>::const_element_reference reference;
	typename WeightedLabeledData<T, U>::const_element_range elements = d.elements();
	BOOST_FOREACH(reference elem,elements)
		stream << elem.weight <<" ("<< elem.data.label << " [" << elem.data.input<<"] )"<< "\n";
	return stream;
}

//Stuff for Dimensionality and querying of basic information

template<class InputType>
double sumOfWeights(WeightedUnlabeledData<InputType> const& dataset){
	double weightSum = 0;
	for(std::size_t i = 0; i != dataset.numberOfBatches(); ++i){
		weightSum += sum(dataset.batch(i).weight);
	}
	return weightSum;
}
template<class InputType, class LabelType>
double sumOfWeights(WeightedLabeledData<InputType,LabelType> const& dataset){
	double weightSum = 0;
	for(std::size_t i = 0; i != dataset.numberOfBatches(); ++i){
		weightSum += sum(dataset.batch(i).weight);
	}
	return weightSum;
}

inline unsigned int numberOfClasses(WeightedUnlabeledData<unsigned int> const& labels){
	return numberOfClasses(labels.data());
}

///\brief Returns the number of members of each class in the dataset.
inline std::vector<std::size_t> classSizes(WeightedUnlabeledData<unsigned int> const& labels){
	return classSizes(labels.data());
}

///\brief  Return the dimnsionality of points of a weighted dataset
template <class InputType>
unsigned int dataDimension(WeightedUnlabeledData<InputType> const& dataset){
	return dataDimension(dataset.data());
}

///\brief  Return the input dimensionality of a weighted labeled dataset.
template <class InputType, class LabelType>
unsigned int inputDimension(WeightedLabeledData<InputType, LabelType> const& dataset){
	return dataDimension(dataset.inputs());
}

///\brief  Return the label/output dimensionality of a labeled dataset.
template <class InputType, class LabelType>
unsigned int labelDimension(WeightedLabeledData<InputType, LabelType> const& dataset){
	return dataDimension(dataset.labels());
}
///\brief Return the number of classes (highest label value +1) of a classification dataset with unsigned int label encoding
template <class InputType>
unsigned int numberOfClasses(WeightedLabeledData<InputType, unsigned int> const& dataset){
	return numberOfClasses(dataset.labels());
}

///\brief Returns the number of members of each class in the dataset.
template<class InputType, class LabelType>
inline std::vector<std::size_t> classSizes(WeightedLabeledData<InputType, LabelType> const& dataset){
	return classSizes(dataset.labels());
}

//creation of weighted datasets

/// \brief creates a weighted unweighted data object from two ranges, representing data and weights
template<class InputRange,class LabelRange, class WeightRange>
typename boost::disable_if<
	boost::is_arithmetic<WeightRange>,
	WeightedLabeledData<
		typename boost::range_value<InputRange>::type,
		typename boost::range_value<LabelRange>::type
	>
>::type createLabeledDataFromRange(InputRange const& inputs, LabelRange const& labels, WeightRange const& weights, std::size_t batchSize = 0){
	SHARK_CHECK(boost::size(inputs) == boost::size(labels),
	"[createDataFromRange] number of data points and number of weights must agree");
	SHARK_CHECK(boost::size(inputs) == boost::size(weights),
	"[createDataFromRange] number of data points and number of weights must agree");
	typedef typename boost::range_value<InputRange>::type InputType;
	typedef typename boost::range_value<LabelRange>::type LabelType;

	if (batchSize == 0)
		batchSize = WeightedLabeledData<InputRange,LabelRange>::DefaultBatchSize;

	return WeightedLabeledData<InputType,LabelType>(
		createLabeledDataFromRange(inputs,labels,batchSize),
		createDataFromRange(weights,batchSize)
	);
}

/// \brief Creates a bootstrap partition of a labeled dataset and returns it using weighting.
///
/// Bootstrapping resamples the dataset by drawing a set of points with
/// replacement. Thus the sampled set will contain some points multiple times
/// and some points not at all. Bootstrapping is usefull to obtain unbiased
/// measurements of the mean and variance of an estimator.
///
/// Optionally the size of the bootstrap (that is, the number of sampled points)
/// can be set. By default it is 0, which indicates that it is the same size as the original dataset.
template<class InputType, class LabelType>
WeightedLabeledData< InputType, LabelType> bootstrap(
	LabeledData<InputType,LabelType> const& dataset,
	std::size_t bootStrapSize = 0
){
	if(bootStrapSize == 0)
		bootStrapSize = dataset.numberOfElements();
	
	WeightedLabeledData<InputType,LabelType> bootstrapSet(dataset,0.0);

	for(std::size_t i = 0; i != bootStrapSize; ++i){
		std::size_t index = Rng::discrete(0,bootStrapSize-1);
		bootstrapSet.element(index).weight += 1.0;
	}
	return bootstrapSet;
}

/// \brief Creates a bootstrap partition of an unlabeled dataset and returns it using weighting.
///
/// Bootstrapping resamples the dataset by drawing a set of points with
/// replacement. Thus the sampled set will contain some points multiple times
/// and some points not at all. Bootstrapping is usefull to obtain unbiased
/// measurements of the mean and variance of an estimator.
///
/// Optionally the size of the bootstrap (that is, the number of sampled points)
/// can be set. By default it is 0, which indicates that it is the same size as the original dataset.
template<class InputType>
WeightedUnlabeledData<InputType> bootstrap(
	UnlabeledData<InputType> const& dataset,
	std::size_t bootStrapSize = 0
){
	if(bootStrapSize == 0)
		bootStrapSize = dataset.numberOfElements();
	
	WeightedUnlabeledData<InputType> bootstrapSet(dataset,0.0);

	for(std::size_t i = 0; i != bootStrapSize; ++i){
		std::size_t index = Rng::discrete(0,bootStrapSize-1);
		bootstrapSet.element(index).weight += 1.0;
	}
	return bootstrapSet;
}

/** @*/
}

#endif
