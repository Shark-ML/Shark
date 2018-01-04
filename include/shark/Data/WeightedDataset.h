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

#ifndef SHARK_DATA_WEIGHTED_DATASET_H
#define SHARK_DATA_WEIGHTED_DATASET_H

#include <shark/Data/Dataset.h>
namespace shark {
	
///\brief Input-Label pair of data
template<class DataType, class WeightType>
struct WeightedDataPair{
	DataType data;
	WeightType weight;
	
	WeightedDataPair(){}
	
	template<class DataT, class WeightT>
	WeightedDataPair(
		DataT&& data,
		WeightT&& weight
	):data(data),weight(weight){}
	
	template<class DataT, class WeightT>
	WeightedDataPair(
		WeightedDataPair<DataT,WeightT> const& pair
	):data(pair.data),weight(pair.weight){}
	
	template<class DataT, class WeightT>
	WeightedDataPair& operator=(WeightedDataPair<DataT,WeightT> const& batch){
		data = batch.data;
		weight = batch.weight;
		return *this;
	}
	WeightedDataPair& operator=(WeightedDataPair const& batch){
		data = batch.data;
		weight = batch.weight;
		return *this;
	}
};

template<class D1, class W1, class D2, class W2>
void swap(WeightedDataPair<D1, W1>&& p1, WeightedDataPair<D2, W2>&& p2){
	using std::swap;
	swap(std::forward<D1>(p1.data),std::forward<D2>(p2.data));
	swap(std::forward<W1>(p1.weight),std::forward<W2>(p2.weight));
}

template<class DataBatchType,class WeightBatchType>
struct WeightedDataBatch{
private:
	typedef typename BatchTraits<typename std::decay<DataBatchType>::type >::type DataBatchTraits;
	typedef typename BatchTraits<typename std::decay<WeightBatchType>::type >::type WeightBatchTraits;
public:
	DataBatchType data;
	WeightBatchType weight;

	typedef WeightedDataPair<
		typename DataBatchTraits::value_type,
		typename WeightBatchTraits::value_type
	> value_type;
	typedef WeightedDataPair<
		decltype(getBatchElement(std::declval<DataBatchType&>(),0)),
		decltype(getBatchElement(std::declval<WeightBatchType&>(),0))
	> reference;
	typedef WeightedDataPair<
		decltype(getBatchElement(std::declval<typename std::add_const<DataBatchType>::type&>(),0)),
		decltype(getBatchElement(std::declval<typename std::add_const<WeightBatchType>::type&>(),0))
	> const_reference;
	typedef IndexingIterator<WeightedDataBatch> iterator;
	typedef IndexingIterator<WeightedDataBatch const> const_iterator;

	template<class D, class W>
	WeightedDataBatch(
		D&& data,
		W&& weight
	):data(data),weight(weight){}
	
	template<class Pair>
	WeightedDataBatch(
		std::size_t size,Pair const& p
	):data(DataBatchTraits::createBatch(p.data,size)),weight(WeightBatchTraits::createBatch(p.weight,size)){}
	
	template<class I, class L>
	WeightedDataBatch& operator=(WeightedDataBatch<I,L> const& batch){
		data = batch.data;
		weight = batch.weight;
		return *this;
	}

	std::size_t size()const{
		return DataBatchTraits::size(data);
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
		return reference(getBatchElement(data,i),getBatchElement(weight,i));
	}
	const_reference operator[](std::size_t i)const{
		return const_reference(getBatchElement(data,i),getBatchElement(weight,i));
	}
};

template<class D1, class W1, class D2, class W2>
void swap(WeightedDataBatch<D1, W1>& p1, WeightedDataBatch<D2, W2>& p2){
	using std::swap;
	swap(p1.data,p2.data);
	swap(p1.weight,p2.weight);
}

template<class DataType, class WeightType>
struct Batch<WeightedDataPair<DataType, WeightType> >
: public detail::SimpleBatch<
	WeightedDataBatch<typename detail::element_to_batch<DataType>::type, typename detail::element_to_batch<WeightType>::type>
>{};

template<class DataType, class WeightType>
struct BatchTraits<WeightedDataBatch<DataType, WeightType> >{
	typedef typename detail::batch_to_element<DataType>::type DataElem;
	typedef typename detail::batch_to_element<WeightType>::type WeightElem;
	typedef Batch<WeightedDataPair<DataElem,WeightElem> > type;
};


namespace detail{
template <class DataContainerT>
class BaseWeightedDataset : public ISerializable
{
public:
	typedef typename DataContainerT::element_type DataType;
	typedef double WeightType;
	typedef DataContainerT DataContainer;
	typedef Data<WeightType> WeightContainer;
	typedef typename DataContainer::IndexSet IndexSet;

	// TYPEDEFS FOR PAIRS
	typedef WeightedDataPair<
		DataType,
		WeightType
	> element_type;

	typedef WeightedDataBatch<
		typename DataContainer::batch_type,
		typename WeightContainer::batch_type
	> batch_type;

	// TYPEDEFS FOR BATCH REFERENCES
	typedef WeightedDataBatch<
		typename DataContainer::batch_reference,
		typename WeightContainer::batch_reference
	> batch_reference;
	typedef WeightedDataBatch<
		typename DataContainer::const_batch_reference,
		typename WeightContainer::const_batch_reference
	> const_batch_reference;
	
	typedef typename Batch<element_type>::reference element_reference;
	typedef typename Batch<element_type>::const_reference const_element_reference;

	typedef boost::iterator_range< detail::DataElementIterator<BaseWeightedDataset<DataContainer> > > element_range;
	typedef boost::iterator_range< detail::DataElementIterator<BaseWeightedDataset<DataContainer> const> > const_element_range;
	typedef detail::BatchRange<BaseWeightedDataset<DataContainer> > batch_range;
	typedef detail::BatchRange<BaseWeightedDataset<DataContainer> const> const_batch_range;


	///\brief Returns the range of elements.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	const_element_range elements()const{
		return const_element_range(
			detail::DataElementIterator<BaseWeightedDataset<DataContainer> const>(this,0,0,0),
			detail::DataElementIterator<BaseWeightedDataset<DataContainer> const>(this,numberOfBatches(),0,numberOfElements())
		);
	}
	///\brief Returns therange of elements.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	element_range elements(){
		return element_range(
			detail::DataElementIterator<BaseWeightedDataset<DataContainer> >(this,0,0,0),
			detail::DataElementIterator<BaseWeightedDataset<DataContainer> >(this,numberOfBatches(),0,numberOfElements())
		);
	}
	
	///\brief Returns the range of batches.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	const_batch_range batches()const{
		return const_batch_range(this);
	}
	///\brief Returns the range of batches.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
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
		SHARK_RUNTIME_CHECK(data.numberOfElements() == weights.numberOfElements(), "[ BaseWeightedDataset::WeightedUnlabeledData] number of data and number of weights must agree");
#ifndef DNDEBUG
		for(std::size_t i  = 0; i != data.numberOfBatches(); ++i){
			SIZE_CHECK(batchSize(data.batch(i)) == batchSize(weights.batch(i)));
		}
#endif
	}
	
	///\brief Construction from data. All points get the same weight assigned
	BaseWeightedDataset(DataContainer const& data, double weight)
	: m_data(data), m_weights(data.numberOfBatches())
	{
		for(std::size_t i = 0; i != numberOfBatches(); ++i){
			m_weights.batch(i) = Batch<WeightType>::type(batchSize(m_data.batch(i)),weight);
		}
	}
	
	
	// ELEMENT ACCESS
	element_reference element(std::size_t i){
		return *(detail::DataElementIterator<BaseWeightedDataset<DataContainer> >(this,0,0,0)+i);
	}
	const_element_reference element(std::size_t i) const{
		return *(detail::DataElementIterator<BaseWeightedDataset<DataContainer> const>(this,0,0,0)+i);
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
		shark::shuffle(this->elements().begin(),this->elements().end(), random::globalRng);
	}

	void splitBatch(std::size_t batch, std::size_t elementIndex){
		m_data.splitBatch(batch,elementIndex);
		m_weights.splitBatch(batch,elementIndex);
	}

	/// \brief Appends the contents of another data object to the end
	///
	/// The batches are not copied but now referenced from both datasets. Thus changing the appended
	/// dataset might change this one as well.
	void append(BaseWeightedDataset const& other){
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

	friend void swap( BaseWeightedDataset& a, BaseWeightedDataset& b){
		swap(a.m_data,b.m_data);
		swap(a.m_weights,b.m_weights);
	}


	// SUBSETS

	///\brief Fill in the subset defined by the list of indices.
	BaseWeightedDataset indexedSubset(IndexSet const& indices) const{
		BaseWeightedDataset subset;
		subset.m_data = m_data.indexedSubset(indices);
		subset.m_weights = m_weights.indexedSubset(indices);
		return subset;
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
	
	///\brief Returns the Shape of the data.
	Shape const& shape() const{
		return data().shape();
	}
	
	///\brief Returns the Shape of the data.
	Shape& shape(){
		return data().shape();
	}
	///\brief Splits the container into two independent parts. The left part remains in the container, the right is stored as return type
	///
	///Order of elements remain unchanged. The SharedVector is not allowed to be shared for
	///this to work.
	WeightedUnlabeledData splice(std::size_t batch){
		return WeightedUnlabeledData(data().splice(batch),weights().splice(batch));
	}

	friend void swap(WeightedUnlabeledData& a, WeightedUnlabeledData& b){
		swap(static_cast<base_type&>(a),static_cast<base_type&>(b));
	}
};

///brief  Outstream of elements for weighted data.
template<class T>
std::ostream &operator << (std::ostream &stream, const WeightedUnlabeledData<T>& d) {
	for(auto elem: d.elements())
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

	SHARK_RUNTIME_CHECK(batchSize(data) == batchSize(weights),"Number of datapoints and number of weights must agree");

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
	
	///\brief Returns the Shape of the inputs.
	Shape const& inputShape() const{
		return inputs().shape();
	}
	
	///\brief Returns the Shape of the inputs.
	Shape& inputShape(){
		return inputs().shape();
	}
	
	///\brief Returns the Shape of the labels.
	Shape const& labelShape() const{
		return labels().shape();
	}
	
	///\brief Returns the Shape of the labels.
	Shape& labelShape(){
		return labels().shape();
	}
	
	/// \brief Constructs an WeightedUnlabeledData object for the inputs.
	WeightedUnlabeledData<InputType> weightedInputs() const{
		return WeightedUnlabeledData<InputType>(data().inputs(),weights());
	}

	///\brief Splits the container into two independent parts. The left part remains in the container, the right is stored as return type
	///
	///Order of elements remain unchanged. The SharedVector is not allowed to be shared for
	///this to work.
	WeightedLabeledData splice(std::size_t batch){
		return WeightedLabeledData(data().splice(batch),weights().splice(batch));
	}

	friend void swap(WeightedLabeledData& a, WeightedLabeledData& b){
		swap(static_cast<base_type&>(a),static_cast<base_type&>(b));
	}
};

///brief  Outstream of elements for weighted labeled data.
template<class T, class U>
std::ostream &operator << (std::ostream &stream, const WeightedLabeledData<T, U>& d) {
	for(auto elem: d.elements())
		stream << elem.weight <<" ("<< elem.data.label << " [" << elem.data.input<<"] )"<< "\n";
	return stream;
}

//Stuff for Dimensionality and querying of basic information

inline std::size_t numberOfClasses(WeightedUnlabeledData<unsigned int> const& labels){
	return numberOfClasses(labels.data());
}

///\brief Returns the number of members of each class in the dataset.
inline std::vector<std::size_t> classSizes(WeightedUnlabeledData<unsigned int> const& labels){
	return classSizes(labels.data());
}

///\brief  Return the dimnsionality of points of a weighted dataset
template <class InputType>
std::size_t dataDimension(WeightedUnlabeledData<InputType> const& dataset){
	return dataDimension(dataset.data());
}

///\brief  Return the input dimensionality of a weighted labeled dataset.
template <class InputType, class LabelType>
std::size_t inputDimension(WeightedLabeledData<InputType, LabelType> const& dataset){
	return dataDimension(dataset.inputs());
}

///\brief  Return the label/output dimensionality of a labeled dataset.
template <class InputType, class LabelType>
std::size_t labelDimension(WeightedLabeledData<InputType, LabelType> const& dataset){
	return dataDimension(dataset.labels());
}
///\brief Return the number of classes (highest label value +1) of a classification dataset with unsigned int label encoding
template <class InputType>
std::size_t numberOfClasses(WeightedLabeledData<InputType, unsigned int> const& dataset){
	return numberOfClasses(dataset.labels());
}

///\brief Returns the number of members of each class in the dataset.
template<class InputType, class LabelType>
inline std::vector<std::size_t> classSizes(WeightedLabeledData<InputType, LabelType> const& dataset){
	return classSizes(dataset.labels());
}

///\brief Returns the total sum of weights.
template<class InputType>
double sumOfWeights(WeightedUnlabeledData<InputType> const& dataset){
	double weightSum = 0;
	for(std::size_t i = 0; i != dataset.numberOfBatches(); ++i){
		weightSum += sum(dataset.batch(i).weight);
	}
	return weightSum;
}
///\brief Returns the total sum of weights.
template<class InputType, class LabelType>
double sumOfWeights(WeightedLabeledData<InputType,LabelType> const& dataset){
	double weightSum = 0;
	for(std::size_t i = 0; i != dataset.numberOfBatches(); ++i){
		weightSum += sum(dataset.batch(i).weight);
	}
	return weightSum;
}

/// \brief Computes the cumulative weight of every class.
template<class InputType>
RealVector classWeight(WeightedLabeledData<InputType,unsigned int> const& dataset){
	RealVector weights(numberOfClasses(dataset),0.0);
	for(auto const& elem: dataset.elements()){
		weights(elem.data.label) += elem.weight;
	}
	return weights;
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

	SHARK_RUNTIME_CHECK(batchSize(inputs) == batchSize(labels),
	"number of inputs and number of labels must agree");
	SHARK_RUNTIME_CHECK(batchSize(inputs) == batchSize(weights),
	"number of data points and number of weights must agree");
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
		std::size_t index = random::discrete(random::globalRng, std::size_t(0),bootStrapSize-1);
		bootstrapSet.element(index).weight += 1.0;
	}
	bootstrapSet.inputShape() = dataset.inputShape();
	bootstrapSet.labelShape() = dataset.labelShape();
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
		std::size_t index = random::discrete(random::globalRng, std::size_t(0),bootStrapSize-1);
		bootstrapSet.element(index).weight += 1.0;
	}
	bootstrapSet.shape() = dataset.shape();
	return bootstrapSet;
}

}

#endif
