//===========================================================================
/*!
 * 
 *
 * \brief       Data for (un-)base_typevised learning.
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
 * \date        2010-2013
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

#ifndef SHARK_DATA_DATASET_H
#define SHARK_DATA_DATASET_H

#include <boost/foreach.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/range/algorithm/sort.hpp>

#include <shark/Core/Exception.h>
#include <shark/Core/OpenMP.h>
#include <shark/Core/utility/functional.h>
#include <shark/Rng/GlobalRng.h>
#include "Impl/Dataset.inl"

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
/// Nearly all operations of the set have to be interpreted in terms of the batch. So the iterator interface will
/// give access to the batches but not to single elements. For this separate element_iterators and const_element_iterators
/// can be used.
///\par
/// There are a lot of these typedefs. The typical typedefs for containers like batch_type or iterator are chosen
/// as types for the batch interface. For accessing single elements, a different set of typedefs is in place. Thus instead of iterator
/// you must write element_iterator and instead of batch_type write element_type. Usually you should not use element_type except when
/// you want to actually copy the data. Instead use element_reference or const_element_reference. Note that these are proxy objects and not
/// actual references to element_type!
/// A short example for these typedefs:
///\code
///typedef Data<RealVector> Set;
/// Set data;
/// for(Set::element_iterator pos=data.elemBegin();pos!= data.elemEnd();++pos){
///     std::cout<<*pos<<" ";
///     Set::element_reference ref=*pos;
///     ref*=2;
///     std::cout<<*pos<<std::endl;
///}
///\endcode
///When you write C++11 code, this is of course much simpler:
///\code
/// Data<RealVector> data;
/// for(auto pos=data.elemBegin();pos!= data.elemEnd();++pos){
///     std::cout<<*pos<<" ";
///     auto ref=*pos;
///     ref*=2;
///     std::cout<<*pos<<std::endl;
///}
///\endcode
/// \par
/// Element wise accessing of elements is usually slower than accessing the batches. If possible, use direct batch access, or
/// at least use the iterator interface to iterate over all elements. Random access to single elements is linear time, so use it wisely.
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
/// For example this is dangerous:
/// \code
/// void function(Data<unsigned int>& data){
///      Data<unsigned int> newData(...);
///      data=newData;
/// }
/// \endcode
/// When data was originally a labeledData object, and newData has a different batch structure than data, this will lead to structural inconsistencies.
/// When function is rewritten such that newData has the same structure as data, this code is perfectly fine. The best way to get around this problem is
/// by rewriting the code as:
/// \code
/// Data<unsigned int> function(){
///      Data<unsigned int> newData(...);
///      return newData;
/// }
/// \endcode
///\todo expand docu
template <class Type>
class Data : public ISerializable
{
protected:
	typedef detail::SharedContainer<Type> Container;
	typedef Data<Type> self_type;

	Container m_data;		///< data
public:
	/// \brief Defines the default batch size of the Container.
	///
	/// Zero means: unlimited
	BOOST_STATIC_CONSTANT(std::size_t, DefaultBatchSize = 256);

	typedef typename Container::BatchType batch_type;
	typedef batch_type& batch_reference;
	typedef batch_type const& const_batch_reference;

	typedef Type element_type;
	typedef typename Batch<element_type>::reference element_reference;
	typedef typename Batch<element_type>::const_reference const_element_reference;

	typedef std::vector<std::size_t> IndexSet;

	template <class T> friend bool operator == (const Data<T>& op1, const Data<T>& op2);
	template <class InputT, class LabelT> friend class LabeledData;


	// RANGES
	typedef boost::iterator_range<typename Container::element_iterator> element_range;
	typedef boost::iterator_range<typename Container::const_element_iterator> const_element_range;
	typedef boost::iterator_range<typename Container::iterator> batch_range;
	typedef boost::iterator_range<typename Container::const_iterator> const_batch_range;


	///\brief Returns the range of elements.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	const_element_range elements()const{
		return const_element_range(m_data.elemBegin(),m_data.elemEnd());
	}
	///\brief Returns therange of elements.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	element_range elements(){
		return element_range(m_data.elemBegin(),m_data.elemEnd());
	}

	///\brief Returns the range of batches.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	const_batch_range batches()const{
		return const_batch_range(m_data.begin(),m_data.end());
	}
	///\brief Returns the range of batches.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	batch_range batches(){
		return batch_range(m_data.begin(),m_data.end());
	}

	///\brief Returns the number of batches of the set.
	std::size_t numberOfBatches() const{
		return m_data.size();
	}
	///\brief Returns the total number of elements.
	std::size_t numberOfElements() const{
		return m_data.numberOfElements();
	}

	///\brief Check whether the set is empty.
	bool empty() const{
		return m_data.empty();
	}

	// ELEMENT ACCESS
	element_reference element(std::size_t i){
		return *(m_data.elemBegin()+i);
	}
	const_element_reference element(std::size_t i) const{
		return *(m_data.elemBegin()+i);
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

	///\brief Construct a dataset with empty batches.
	explicit Data(std::size_t numBatches) : m_data( numBatches )
	{ }

	///\brief Construct a dataset with different batch sizes as a copy of another dataset
	explicit Data(Data const& container, std::vector<std::size_t> batchSizes)
	: m_data( container.m_data, batchSizes, true )
	{ }

	///\brief Construction with size and a single element
	///
	/// Optionally the desired batch Size can be set
	///
	///@param size the new size of the container
	///@param element the blueprint element from which to create the Container
	///@param batchSize the size of the batches. if this is 0, the size is unlimited
	explicit Data(std::size_t size, element_type const& element, std::size_t batchSize = DefaultBatchSize)
	: m_data(size,element,batchSize)
	{ }

	//~ /// Construction from data
	//~ ///@param points the data from which to create the Container
	//~ ///@param batchSize the size of the batches. if this is 0, the size is unlimited
	//~ Data(std::vector<element_type> const& points, std::size_t batchSize = DefaultBatchSize)
	//~ : m_data(points,batchSize)
	//~ { }

	// MISC

	void read(InArchive& archive){
		archive >> m_data;
	}

	void write(OutArchive& archive) const{
		archive << m_data;
	}
	///\brief This method makes the vector independent of all siblings and parents.
	virtual void makeIndependent(){
		m_data.makeIndependent();
	}


	// METHODS TO ALTER BATCH STRUCTURE

	void splitBatch(std::size_t batch, std::size_t elementIndex){
		m_data.splitBatch(m_data.begin()+batch,elementIndex);
	}

	///\brief Splits the container in two independent parts. The left part remains in the container, the right is stored as return type
	///
	///Order of elements remain unchanged. The SharedVector is not allowed to be shared for
	///this to work.
	self_type splice(std::size_t batch){
		self_type right;
		right.m_data=m_data.splice(m_data.begin()+batch);
		return right;
	}

	/// \brief Appends the contents of another data object to the end
	///
	/// The batches are not copied but now referenced from both datasets. Thus changing the appended
	/// dataset might change this one as well.
	void append(self_type const& other){
		m_data.append(other.m_data);
	}

	///\brief Reorders the batch structure in the container to that indicated by the batchSizes vector
	///
	///After the operation the container will contain batchSizes.size() batchs with the i-th batch having size batchSize[i].
	///However the sum of all batch sizes must be equal to the current number of elements
	template<class Range>
	void repartition(Range const& batchSizes){
		m_data.repartition(batchSizes);
	}

	// SUBSETS
	///\brief Fill in the subset defined by the list of indices.
	void indexedSubset(IndexSet const& indices, self_type& subset) const{
		subset.m_data=Container(m_data,indices);
	}

	///\brief Fill in the subset defined by the list of indices as well as its complement.
	void indexedSubset(IndexSet const& indices, self_type& subset, self_type& complement) const{
		IndexSet comp;
		detail::complement(indices,m_data.size(),comp);
		subset.m_data=Container(m_data,indices);
		complement.m_data=Container(m_data,comp);
	}

	friend void swap(Data& a, Data& b){
		swap(a.m_data,b.m_data);
	}
};

/**
 * \ingroup shark_globals
 * @{
 */

/// Outstream of elements.
template<class T>
std::ostream &operator << (std::ostream &stream, const Data<T>& d) {
	typedef typename Data<T>::const_element_reference reference;
	typename Data<T>::const_element_range elements = d.elements();
	BOOST_FOREACH(reference elem,elements)
		stream << elem << "\n";
	return stream;
}
/** @} */

/// \brief Data set for unsupervised learning.
///
/// The UnlabeledData class is basically a standard Data container
/// with the special interpretation of its data point being
/// "inputs" to a learning algorithm.
///
template <class InputT>
class UnlabeledData : public Data<InputT>
{
public:
	typedef InputT element_type;
	typedef Data<element_type> base_type;
	typedef UnlabeledData<element_type> self_type;
	typedef element_type InputType;
	typedef detail::SharedContainer<InputT> InputContainer;

protected:
	using base_type::m_data;
public:

	///\brief Constructor.
	UnlabeledData()
	{ }

	//~ ///\brief Construction from data.
	//~ UnlabeledData(std::vector<InputT> const& points,std::size_t batchSize = base_type::DefaultBatchSize)
	//~ : base_type(points,batchSize)
	//~ { }

	///\brief Construction from data.
	UnlabeledData(Data<InputT> const& points)
	: base_type(points)
	{ }

	///\brief Construction with size and a single element
	///
	/// Optionally the desired batch Size can be set
	///
	///@param size the new size of the container
	///@param element the blueprint element from which to create the Container
	///@param batchSize the size of the batches. if this is 0, the size is unlimited
	UnlabeledData(std::size_t size, element_type const& element, std::size_t batchSize = base_type::DefaultBatchSize)
	: base_type(size,element,batchSize)
	{ }

	///\brief Create an empty set with just the correct number of batches.
	///
	/// The user must initialize the dataset after that by himself.
	UnlabeledData(std::size_t numBatches)
	: base_type(numBatches)
	{ }

	///\brief Construct a dataset with different batch sizes. it is a copy of the other dataset
	UnlabeledData(UnlabeledData const& container, std::vector<std::size_t> batchSizes)
		:base_type(container,batchSizes){}

	/// \brief we allow assignment from Data.
	self_type operator=(Data<InputT> const& data){
		static_cast<Data<InputT>& >(*this) = data;
		return *this;
	}

	///\brief Access to the base_type class as "inputs".
	///
	/// Added for consistency with the LabeledData::labels() method.
	self_type& inputs(){
		return *this;
	}

	///\brief Access to the base_type class as "inputs".
	///
	/// Added for consistency with the LabeledData::labels() method.
	self_type const& inputs() const{
		return *this;
	}

	///\brief Splits the container in two independent parts. The left part remains in the container, the right is stored as return type
	///
	///Order of elements remain unchanged. The SharedVector is not allowed to be shared for
	///this to work.
	self_type splice(std::size_t batch){
		self_type right;
		right.m_data=m_data.splice(m_data.begin()+batch);
		return right;
	}

	///\brief shuffles all elements in the entire dataset (that is, also across the batches)
	virtual void shuffle(){
		DiscreteUniform<Rng::rng_type> uni(Rng::globalRng);
		shark::shuffle(this->elements().begin(),this->elements().end(), uni);
	}
};

///
/// \brief Data set for supervised learning.
///
/// The LabeledData class extends UnlabeledData for the
/// representation of inputs. In addition it holds and
/// provides access to the corresponding labels.
///
/// LabeledData tries to mimic the underlying data as pairs of input and label data.
///this means that when accessing a batch by calling batch(splitPointber) or choosing one of the iterators
/// one access the input batch by batch(i).input and the labels by batch(i).label
///
///this also holds true for single element access using operator(). Be aware, that direct access to element is
///a linear time operation. So it is not advisable to iterate over the elements, but instead iterate over the batches.
template <class InputT, class LabelT>
class LabeledData : public ISerializable
{
protected:
	typedef LabeledData<InputT, LabelT> self_type;
public:
	typedef InputT InputType;
	typedef LabelT LabelType;
	typedef UnlabeledData<InputT> InputContainer;
	typedef Data<LabelT> LabelContainer;
	typedef typename InputContainer::IndexSet IndexSet;

	BOOST_STATIC_CONSTANT(std::size_t, DefaultBatchSize = InputContainer::DefaultBatchSize);

	// TYPEDEFS fOR PAIRS
	typedef DataBatchPair<
		typename Batch<InputType>::type,
		typename Batch<LabelType>::type
	> batch_type;

	typedef DataPair<
		InputType,
		LabelType
	> element_type;

	// TYPEDEFS FOR  RANGES
	typedef typename PairRangeType<
		element_type,
		typename InputContainer::element_range,
		typename LabelContainer::element_range
	>::type element_range;
	typedef typename PairRangeType<
		element_type,
		typename InputContainer::const_element_range,
		typename LabelContainer::const_element_range
	>::type const_element_range;
	typedef typename PairRangeType<
		batch_type,
		typename InputContainer::batch_range,
		typename LabelContainer::batch_range
	>::type batch_range;
	typedef typename PairRangeType<
		batch_type,
		typename InputContainer::const_batch_range,
		typename LabelContainer::const_batch_range
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
		return zipPairRange<element_type>(m_data.elements(),m_label.elements());
	}
	///\brief Returns therange of elements.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	element_range elements(){
		return zipPairRange<element_type>(m_data.elements(),m_label.elements());
	}

	///\brief Returns the range of batches.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	const_batch_range batches()const{
		return zipPairRange<batch_type>(m_data.batches(),m_label.batches());
	}
	///\brief Returns the range of batches.
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	batch_range batches(){
		return zipPairRange<batch_type>(m_data.batches(),m_label.batches());
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

	///\brief Create an empty set with just the correct number of batches.
	///
	/// The user must initialize the dataset after that by himself.
	LabeledData(std::size_t numBatches)
	: m_data(numBatches),m_label(numBatches)
	{}

	///
	/// Optionally the desired batch Size can be set
	///
	///@param size the new size of the container
	///@param element the blueprint element from which to create the Container
	///@param batchSize the size of the batches. if this is 0, the size is unlimited
	LabeledData(std::size_t size, element_type const& element, std::size_t batchSize = DefaultBatchSize)
	: m_data(size,element.input,batchSize),
	  m_label(size,element.label,batchSize)
	{}

	///\brief Construction from data.
	///
	///Beware that when calling this constructor the organization of batches must be equal in both
        ///containers. This Constructor will not split the data!
	LabeledData(Data<InputType> const& inputs, Data<LabelType> const& labels)
	: m_data(inputs), m_label(labels)
	{
		SHARK_CHECK(inputs.numberOfElements() == labels.numberOfElements(), "[LabeledData::LabeledData] number of inputs and number of labels must agree");
#ifndef DNDEBUG
		for(std::size_t i  = 0; i != inputs.numberOfBatches(); ++i){
			SIZE_CHECK(shark::size(inputs.batch(i))==shark::size(labels.batch(i)));
		}
#endif
	}
	// ELEMENT ACCESS
	element_reference element(std::size_t i){
		return element_reference(m_data.element(i),m_label.element(i));
	}
	const_element_reference element(std::size_t i) const{
		return const_element_reference(m_data.element(i),m_label.element(i));
	}

	// BATCH ACCESS
	batch_reference batch(std::size_t i){
		return batch_reference(m_data.batch(i),m_label.batch(i));
	}
	const_batch_reference batch(std::size_t i) const{
		return const_batch_reference(m_data.batch(i),m_label.batch(i));
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

	///\brief shuffles all elements in the entire dataset (that is, also across the batches)
	virtual void shuffle(){
		DiscreteUniform<Rng::rng_type> uni(Rng::globalRng);
		shark::shuffle(this->elements().begin(),this->elements().end(), uni);
	}

	void splitBatch(std::size_t batch, std::size_t elementIndex){
		m_data.splitBatch(batch,elementIndex);
		m_label.splitBatch(batch,elementIndex);
	}

	///\brief Splits the container into two independent parts. The left part remains in the container, the right is stored as return type
	///
	///Order of elements remain unchanged. The SharedVector is not allowed to be shared for
	///this to work.
	self_type splice(std::size_t batch){
		return self_type(m_data.splice(batch),m_label.splice(batch));
	}

	/// \brief Appends the contents of another data object to the end
	///
	/// The batches are not copied but now referenced from both datasets. Thus changing the appended
	/// dataset might change this one as well.
	void append(self_type const& other){
		m_data.append(other.m_data);
		m_label.append(other.m_label);
	}


	///\brief Reorders the batch structure in the container to that indicated by the batchSizes vector
	///
	///After the operation the container will contain batchSizes.size() batches with the i-th batch having size batchSize[i].
	///However the sum of all batch sizes must be equal to the current number of elements
	template<class Range>
	void repartition(Range const& batchSizes){
		m_data.repartition(batchSizes);
		m_label.repartition(batchSizes);
	}

	friend void swap(LabeledData& a, LabeledData& b){
		swap(a.m_data,b.m_data);
		swap(a.m_label,b.m_label);
	}


	// SUBSETS

	///\brief Fill in the subset defined by the list of indices.
	void indexedSubset(IndexSet const& indices, self_type& subset) const{
		m_data.indexedSubset(indices,subset.m_data);
		m_label.indexedSubset(indices,subset.m_label);
	}

	///\brief Fill in the subset defined by the list of indices as well as its complement.
	void indexedSubset(IndexSet const& indices, self_type& subset, self_type& complement)const{
		IndexSet comp;
		detail::complement(indices,m_data.size(),comp);
		m_data.indexedSubset(indices,subset.m_data);
		m_label.indexedSubset(indices,subset.m_label);
		m_data.indexedSubset(comp,complement.m_data);
		m_label.indexedSubset(comp,complement.m_label);
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


/**
 * \addtogroup shark_globals
 * @{
 */

/// \brief creates a data object from a range of elements
template<class Range>
Data<typename boost::range_value<Range>::type>
createDataFromRange(Range const& inputs, std::size_t maximumBatchSize = 0){
	typedef typename boost::range_value<Range const>::type Input;
	typedef typename boost::range_iterator<Range const>::type Iterator;

	if (maximumBatchSize == 0)
		maximumBatchSize = Data<Input>::DefaultBatchSize;

	std::size_t numPoints = shark::size(inputs);
	//first determine the optimal number of batches as well as batch size
	std::size_t batches = numPoints / maximumBatchSize;
	if(numPoints > batches*maximumBatchSize)
		++batches;
	std::size_t optimalBatchSize=numPoints/batches;
	std::size_t remainder = numPoints-batches*optimalBatchSize;
	Data<Input> data(batches);

	//now create the batches taking the remainder into account
	Iterator start= boost::begin(inputs);
	for(std::size_t i = 0; i != batches; ++i){
		std::size_t size = (i<remainder)?optimalBatchSize+1:optimalBatchSize;
		Iterator end = start+size;
		data.batch(i) = createBatch<Input>(
			boost::make_iterator_range(start,end)
		);
		start = end;
	}

	return data;
}
/// \brief creates a labeled data object from two ranges, representing inputs and labels
template<class Range1, class Range2>
LabeledData<
	typename boost::range_value<Range1>::type,
	typename boost::range_value<Range2>::type
> createLabeledDataFromRange(Range1 const& inputs, Range2 const& labels, std::size_t batchSize = 0){
	SHARK_CHECK(boost::size(inputs) == boost::size(labels),
	"[createDataFromRange] number of inputs and number of labels must agree");
	typedef typename boost::range_value<Range1>::type Input;
	typedef typename boost::range_value<Range2>::type Label;

	if (batchSize == 0)
		batchSize = LabeledData<Input,Label>::DefaultBatchSize;

	return LabeledData<Input,Label>(
		createDataFromRange(inputs,batchSize),
		createDataFromRange(labels,batchSize)
	);
}

///brief  Outstream of elements for labeled data.
template<class T, class U>
std::ostream &operator << (std::ostream &stream, const LabeledData<T, U>& d) {
	typedef typename LabeledData<T, U>::const_element_reference reference;
	typename LabeledData<T, U>::const_element_range elements = d.elements();
	BOOST_FOREACH(reference elem,elements)
		stream << elem.input << " [" << elem.label <<"]"<< "\n";
	return stream;
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
		std::size_t batchSize = size(labels.batch(i));
		for(std::size_t j = 0; j != batchSize; ++j){
			classCounts[labels.batch(i)(j)]++;
		}
	}
	return classCounts;
}

/// Return the dimensionality of a  dataset.
template <class InputType>
unsigned int dataDimension(Data<InputType> const& dataset){
	SHARK_ASSERT(dataset.numberOfElements() > 0);
	return dataset.element(0).size();
}

/// Return the input dimensionality of a labeled dataset.
template <class InputType, class LabelType>
unsigned int inputDimension(LabeledData<InputType, LabelType> const& dataset){
	return dataDimension(dataset.inputs());
}

/// Return the label/output dimensionality of a labeled dataset.
template <class InputType, class LabelType>
unsigned int labelDimension(LabeledData<InputType, LabelType> const& dataset){
	return dataDimension(dataset.labels());
}
///\brief Return the number of classes (highest label value +1) of a classification dataset with unsigned int label encoding
template <class InputType>
unsigned int numberOfClasses(LabeledData<InputType, unsigned int> const& dataset){
	return numberOfClasses(dataset.labels());
}
/// Return the number of classes (size of the label vector)
/// of a classification dataset with RealVector label encoding.
template <class InputType>
unsigned int numberOfClasses(LabeledData<InputType, RealVector> const& dataset){
	SHARK_ASSERT(dataset.numberOfElements() > 0);
	return dataset.element(0).label.size();
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
	Data<ResultType> result(batches);
	SHARK_PARALLEL_FOR(int i = 0; i < batches; ++i)
		result.batch(i)= createBatch<T>(boost::adaptors::transform(data.batch(i), f));
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
	Data<ResultType> result(batches);
	SHARK_PARALLEL_FOR(int i = 0; i < batches; ++i)
		result.batch(i)= f(data.batch(i));
	return result;
}

///\brief Transforms the inputs of a dataset and return the transformed result.
template<class I,class L, class Functor>
LabeledData<typename detail::TransformedDataElement<Functor,I>::type, L >
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

template<class DatasetT>
DatasetT indexedSubset(
	DatasetT const& dataset,
	typename DatasetT::IndexSet const& indices
){
	DatasetT subset;
	dataset.indexedSubset(indices,subset);
	return subset;
}
///\brief  Fill in the subset of batches [start,...,size+start[.
template<class DatasetT>
DatasetT rangeSubset(DatasetT const& dataset, std::size_t start, std::size_t end){
	typename DatasetT::IndexSet indices;
	detail::range(end-start, start, indices);
	return indexedSubset(dataset,indices);
}
///\brief  Fill in the subset of batches [0,...,size[.
template<class DatasetT>
DatasetT rangeSubset(DatasetT const& dataset, std::size_t size){
	return rangeSubset(dataset,size,0);
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
	while(batchStart + boost::size(data.batch(batchPos)) < elementIndex){
		batchStart += boost::size(data.batch(batchPos));
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
void repartitionByClass(LabeledData<I,unsigned int>& data,std::size_t batchSize = LabeledData<I,unsigned int>::DefaultBatchSize){
	std::vector<std::size_t > classCounts = classSizes(data);
	std::vector<std::size_t > partitioning;//new, optimal partitioning of the data according to the batch sizes
	std::vector<std::size_t > classStart;//at which batch the elements of the class are starting
	detail::batchPartitioning(classCounts, classStart, partitioning, batchSize);

	data.repartition(partitioning);

	// Now place examples into the batches reserved for their class...

	// The following line does the job in principle but it crashes with clang on the mac:
//	boost::sort(data.elements());//todo we are lying here, use bidirectional iterator sort.

	// The following fixes the issue. As an aside it is even linear time:
	std::vector<std::size_t> bat = classStart;           // batch index until which the class is already filled in
	std::vector<std::size_t> idx(classStart.size(), 0);  // index within the batch until which the class is already filled in
	unsigned int c = 0;                                  // current class in whose batch space we operate
	typedef typename Batch<I>::type InputBatchType;
	typedef typename Batch<unsigned int>::type LabelBatchType;
	for (std::size_t b=0; b<data.numberOfBatches(); b++)
	{
		// update class range index
		std::size_t e = 0;
		while (c + 1 < classStart.size() && b == classStart[c + 1])
		{
			c++;
			b = bat[c];
			e = idx[c];
		}
		if (b == data.numberOfBatches()) break;

		InputBatchType& bi1 = data.inputs().batch(b);
		LabelBatchType& bl1 = data.labels().batch(b);
		while (true)
		{
			unsigned int l = shark::get(bl1, e);
			if (l == c)   // leave element in place
			{
				e++;
				idx[c] = e;
				if (e == boost::size(bl1))
				{
					e = 0;
					idx[c] = 0;
					bat[c]++;
					break;
				}
			}
			else   // swap elements
			{
				InputBatchType& bi2 = data.inputs().batch(bat[l]);
				LabelBatchType& bl2 = data.labels().batch(bat[l]);
				swap(shark::get(bi1, e), shark::get(bi2, idx[l]));
				shark::get(bl1, e) = shark::get(bl2, idx[l]);
				shark::get(bl2, idx[l]) = l;
				idx[l]++;
				if (idx[l] == boost::size(bl2))
				{
					idx[l] = 0;
					bat[l]++;
				}
			}
		}
	}
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
	for(;start != numBatches && get(data.batch(start),0).label != smaller;++start);
	SHARK_CHECK(start != numBatches, "[shark::binarySubProblem] class does not exist");

	//copy batch indices of first class
	for(;start != numBatches && get(data.batch(start),0).label == smaller; ++start)
		indexSet.push_back(start);

	//find second class
	for(;start != numBatches && get(data.batch(start),0).label != bigger;++start);
	SHARK_CHECK(start != numBatches, "[shark::binarySubProblem] class does not exist");

	//copy batch indices of second class
	for(;start != numBatches && get(data.batch(start),0).label == bigger; ++start)
		indexSet.push_back(start);

	return transformLabels(indexedSubset(data,indexSet), detail::TransformOneVersusRestLabels(oneClass));
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
	unsigned int oneClass)
{
	return transformLabels(data, detail::TransformOneVersusRestLabels(oneClass));
}

///
///\brief Transformation function multiplying the elements in a dataset by a scalar or component-wise by values stores in a vector
///
class Multiply {
public:
	///@param factor All components of all vectors in the dataset are multiplied by this number
	Multiply(double factor) : m_factor(factor), m_scalar(true) {}
	///@param factor For all elements in the dataset, the i-th component is multiplied with the i-th component of this vector
	Multiply(const RealVector factor) : m_factor(0), m_factorv(factor), m_scalar(false) {}

	typedef RealVector result_type;

	RealVector operator()(RealVector input) const {
		if(m_scalar) {
			for(std::size_t i = 0; i != input.size(); ++i) input(i) *= m_factor;
			return input;
		} else {
			SIZE_CHECK(m_factorv.size() == input.size());
			for(std::size_t i = 0; i != input.size(); ++i) input(i) *= m_factorv(i);
			return input;
		}
	}
private:
	double m_factor;
	RealVector m_factorv;
	bool m_scalar;
};

///
///\brief Transformation function dividing the elements in a dataset by a scalar or component-wise by values stores in a vector
///
class Divide {
public:
	///@param factor All components of all vectors in the dataset are divided by this number
	Divide(double factor) : m_factor(factor), m_scalar(true) {}
	///@param factor For all elements in the dataset, the i-th component is divided by the i-th component of this vector
	Divide(const RealVector factor) : m_factor(0), m_factorv(factor), m_scalar(false) {}

	typedef RealVector result_type;

	RealVector operator()(RealVector input) const {
		if(m_scalar) {
			for(std::size_t i = 0; i != input.size(); ++i) input(i) /= m_factor;
			return input;
		} else {
			SIZE_CHECK(m_factorv.size() == input.size());
			for(std::size_t i = 0; i != input.size(); ++i) input(i) /= m_factorv(i);
			return input;
		}
	}
private:
	double m_factor;
	RealVector m_factorv;
	bool m_scalar;
};


///
///\brief Transformation function adding a vector or a scalar to the elements in a dataset
///
class Shift {
public:
	///@param offset Scalar added to all components of all vectors in the dataset
	Shift(double offset) : m_offset(offset), m_scalar(true) {}
	///@param offset Vector added to vectors in the dataset
	Shift(const RealVector offset) : m_offsetv(offset), m_scalar(false) {}

	typedef RealVector result_type;

	RealVector operator()(RealVector input) const {
		if(m_scalar) {
			for(std::size_t i = 0; i != input.size(); ++i)
				input(i) += m_offset;
		} else {
			SIZE_CHECK(m_offsetv.size() == input.size());
			for(std::size_t i = 0; i != input.size(); ++i)
				input(i) += m_offsetv(i);
		}
		return input;

	}
private:
	double m_offset;
	RealVector m_offsetv;
	bool m_scalar;
};

///
///\brief Transformation function truncating elements in a dataset
///
class Truncate {
public:
	///@param minValue All elements below this value are cut to the minimum value
	///@param maxValue All elements above this value are cut to the maximum value
	Truncate(double minValue,double maxValue) : m_min(minValue), m_max(maxValue){}
	///@param minv Lower bound for element-wise truncation
	///@param maxv Upper bound for element-wise truncation
	Truncate(const RealVector minv, const RealVector maxv) : m_min(1), m_max(-1), m_minv(minv), m_maxv(maxv) { SIZE_CHECK(m_minv.size() == m_maxv.size()); }

	typedef RealVector result_type;

	RealVector operator()(RealVector input) const {
		if(m_min < m_max) {
			for(std::size_t i = 0; i != input.size(); ++i){
				input(i) = std::max(m_min, std::min(m_max, input(i)));
			}
		} else {
			SIZE_CHECK(m_minv.size() == input.size());
			for(std::size_t i = 0; i != input.size(); ++i){
				input(i) = std::max(m_minv(i), std::min(m_maxv(i), input(i)));
			}
		}
		return input;
	}
private:
	double m_min;
	double m_max;
	RealVector m_minv;
	RealVector m_maxv;
};

///
///\brief Transformation function first truncating and then rescaling elements in a dataset
///
class TruncateAndRescale {
public:
	///@param minCutValue All elements below this value are cut to the minimum value
	///@param maxCutValue All elements above this value are cut to the maximum value
	///@param minValue The imterval [minCutValue, maxCutValue] is mapped to [minValue, maxValue]
	///@param maxValue The imterval [minCutValue, maxCutValue] is mapped to [minValue, maxValue]
	TruncateAndRescale(double minCutValue, double maxCutValue, double minValue = 0., double maxValue = 1.) : m_minCut(minCutValue), m_maxCut(maxCutValue), m_range(maxValue - minValue), m_min(minValue), m_scalar(true) {}
	///@param minv Lower bound for element-wise truncation
	///@param maxv Upper bound for element-wise truncation
	///@param minValue The imterval [minv, maxv is mapped to [minValue, maxValue]
	///@param maxValue The imterval [minv, maxv] is mapped to [minValue, maxValue]
	TruncateAndRescale(const RealVector minv, const RealVector maxv, double minValue = 0., double maxValue = 1.) : m_minCutv(minv), m_maxCutv(maxv), m_range(maxValue - minValue), m_min(minValue), m_scalar(false) { SIZE_CHECK(m_minCutv.size() == m_maxCutv.size()); }

	typedef RealVector result_type;

	RealVector operator()(RealVector input) const {
		if(m_scalar) {
			for(std::size_t i = 0; i != input.size(); ++i){
				input(i) = (std::max(m_minCut, std::min(m_maxCut, input(i))) - m_minCut)  / (m_maxCut - m_minCut) * m_range + m_min;
			}
		} else {
			SIZE_CHECK(m_minCutv.size() == input.size());
			for(std::size_t i = 0; i != input.size(); ++i){
				input(i) = (std::max(m_minCutv(i), std::min(m_maxCutv(i), input(i))) - m_minCutv(i))  / (m_maxCutv(i) - m_minCutv(i)) * m_range + m_min;
			}
		}
		return input;
	}
private:
	double m_minCut;
	double m_maxCut;
	RealVector m_minCutv;
	RealVector m_maxCutv;
	double m_range; // maximum - minimum
	double m_min;
	bool m_scalar;
};

/** @*/
}

#endif
