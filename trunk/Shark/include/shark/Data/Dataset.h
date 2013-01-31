//===========================================================================
/*!
 *  \brief Data for (un-)base_typevised learning.
 *
 *
 *  \par
 *  This file provides containers for data used by the models, loss
 *  functions, and learning algorithms (trainers). The reason for
 *  dedicated containers of this type is that data often need to be
 *  split into subsets, such as training and test data, or folds in
 *  cross-validation. The containers in this file provide memory
 *  efficient mechanisms for managing and providing such subsets.
 *
 *
 *  \author  O. Krause, T. Glasmachers
 *  \date    2010-2013
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
//===========================================================================


#ifndef SHARK_DATA_DATASET_H
#define SHARK_DATA_DATASET_H

#include <boost/foreach.hpp>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/iterator_range.hpp>

#include <fstream>

#include <shark/Core/Exception.h>
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
/// There are a lot of these typedefs. The typical typedefs for containers like value_type or iterator are chosen
/// as types for the batch interface. For accessing single elements, a different set of typedefs is in place. Thus instead of iterator
/// you must write element_iterator and instead of value_type write element_type. Usually you should not use element_type escept when
/// you want to actually copy the data. Instead use element_reference or const_element_reference. Note, that these are proxy objects and not
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
/// When constructing the container the batchSize can be set. If it is not set by the user he default batchSize is chosen. A BatchSize of 0
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

	typedef typename Container::BatchType value_type;
	typedef value_type& reference;
	typedef value_type const& const_reference;

	typedef Type element_type;
	typedef typename Batch<element_type>::reference element_reference;
	typedef typename Batch<element_type>::const_reference const_element_reference;
	typedef std::vector<std::size_t> IndexSet;

	template <class T> friend bool operator == (const Data<T>& op1, const Data<T>& op2);
	template <class InputT, class LabelT> friend class LabeledData;


	// ITERATORS
	typedef typename Container::const_iterator const_iterator;
	typedef typename Container::iterator iterator;
	typedef typename Container::const_element_iterator const_element_iterator;
	typedef typename Container::element_iterator element_iterator;

	// ITERATORS FOR BATCHES
	///\brief Iterator to the beginning of the container.
	const_iterator begin() const{
		return m_data.begin();
	}

	///\brief Iterator to the end of the container.
	const_iterator end() const{
		return m_data.end();
	}

	///\brief Iterator to the beginning of the container.
	iterator begin(){
		return m_data.begin();
	}

	///\brief Iterator to the end of the container.
	iterator end(){
		return m_data.end();
	}
	// ITERATORS FOR ELEMENTS
	///\brief Iterator to the beginning of the container.
	const_element_iterator elemBegin() const{
		return m_data.elemBegin();
	}

	///\brief Iterator to the end of the container.
	const_element_iterator elemEnd() const{
		return m_data.elemEnd();
	}

	///\brief Iterator to the beginning of the container.
	element_iterator elemBegin(){
		return m_data.elemBegin();
	}

	///\brief Iterator to the end of the container.
	element_iterator elemEnd(){
		return m_data.elemEnd();
	}

	// ELEMENT ACCESS

	element_reference operator()(std::size_t i){
		return *(elemBegin()+i);
	}
	const_element_reference operator()(std::size_t i) const{
		return *(elemBegin()+i);
	}

	// BATCH ACCESS
	reference batch(std::size_t i){
		return *(begin()+i);
	}
	const_reference batch(std::size_t i) const{
		return *(begin()+i);
	}


	// RANGE ACCESS FOR ELEMENTS
	typedef boost::iterator_range<element_iterator> element_range;
	typedef boost::iterator_range<const_element_iterator> const_element_range;

	///\brief Returns the  half open range of elements [elemBegin(),elemEnd()).
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	const_element_range elements()const{
		return const_element_range(elemBegin(),elemEnd());
	}
	///\brief Returns the  half open range of elements [elemBegin(),elemEnd()).
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	element_range elements(){
		return element_range(elemBegin(),elemEnd());
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

	/// Construction from data
	///@param points the data from which to create the Container
	///@param batchSize the size of the batches. if this is 0, the size is unlimited
	Data(std::vector<element_type> const& points, std::size_t batchSize = DefaultBatchSize)
	: m_data(points,batchSize)
	{ }

	//~ ///\brief Construction from a data container and an index container.
	//~ ///
	//~ /// This is used internally by the LabeledData class.
	//~ /// IT IS RECOMMENDED NOT TO USE THIS FUNCTION DIRECTLY,
	//~ /// since this interface may be subject to change.
	//~ Data(Container const& container, const IndexSets& indexSets)
	//~ : m_data(container)
	//~ , m_indexSets(indexSets)
	//~ { }

	// MISC
	///\brief Returns the number of batches of the set.
	std::size_t size() const{
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

	void read(InArchive& archive){
		archive >> m_data;
		//~ archive >> m_indexSets;
	}

	void write(OutArchive& archive) const{
		archive << m_data;
		//~ archive << m_indexSets;
	}
	///\brief This method makes the vector independent of all siblings and parents.
	virtual void makeIndependent(){
		m_data.makeIndependent();
	}


	// METHODS TO ALTER BATCH STRUCTURE

	void splitBatch(iterator position, std::size_t elementIndex){
		m_data.splitBatch(position,elementIndex);
	}

	///\brief Splits the container in two independent parts. The left part remains in the container, the right is stored as return type
	///
	///Order of elements remain unchanged. The SharedVector is not allowed to be shared for
	///this to work.
	self_type splice(iterator position){
		self_type right;
		right.m_data=m_data.splice(position);
		return right;
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

	//~ /// \brief Fill in the named subset.
	//~ ///
	//~ ///If the subset does not exist the method throws an exception.
	//~ void namedSubset(std::string const& name, self_type& subset) const{
		//~ IndexSet const& indices = m_indexSets.findSubset(name);
		//~ subset.m_data=Container(m_data, indices);
	//~ }

	//~ ///\brief Fill in the named subset and its complement.
	//~ ///
	//~ /// If the subset does not exist the method throws an exception.
	//~ void namedSubset(std::string const& name, self_type& subset, self_type& complement) const{
		//~ IndexSet const& indices = m_indexSets.findSubset(name);
		//~ IndexSet comp(indices.size());
		//~ detail::complement(indices,m_data.size(),comp);
		//~ subset.m_data = Container(m_data, indices);
		//~ complement.m_data = Container(m_data, comp);
	//~ }

	//~ ///\brief Create a new name, under which the given list of indices is stored as a named subset.
	//~ void createNamedSubset(std::string const& name, IndexSet const& indices){
		//~ m_indexSets.createNamedSubset(name, indices);
	//~ }

	//~ ///\brief Create a new name, under which a previously created subset is stored.
	//~ ///
	//~ ///of course the subset must be previously created by this instance or one of
	//~ ///it's non-independent siblings.
	//~ void createNamedSubset(std::string const& name, self_type const& subset){
		//~ //unfortunately, we need to convert the batch vector of the set into an indexset...
		//~ std::vector<std::size_t> indizes(subset.m_data.size());
		//~ for(std::size_t i = 0; i != subset.m_data.size(); ++i){
			//~ for(std::size_t j = 0; j != m_data.size(); ++j){
				//~ if(m_data.pointer(j) == subset.m_data.pointer(i)){
					//~ indizes[i]=j;
					//~ break;
				//~ }
			//~ }
		//~ }
		//~ m_indexSets.createNamedSubset(name, indizes);
	//~ }

	//~ ///\brief Check whether a subset with the given name exists.
	//~ bool hasNamedSubset(std::string const& name) const{
		//~ return m_indexSets.hasNamedSubset(name);
	//~ }

	//~ ///\brief Returns the indices of a previously created subset.
	//~ ///
	//~ ///mostly used for saving the Dataset to a file
	//~ void namedSubsetIndices(std::string const& name, IndexSet& indices){
		//~ indices = m_indexSets.findSubset(name);
	//~ }
};

/**
 * \ingroup shark_globals
 *
 * @{
 */

/// Outstream of elements.
template<class T>
std::ostream &operator << (std::ostream &stream, const Data<T>& d) {
	typedef typename Data<T>::const_element_reference reference;
	BOOST_FOREACH(reference elem,d.elements())
		stream << elem << "\n";
	return stream;
}
/** @*/

/// \brief data set for unbase_typevised learning
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

	///\brief Construction from data.
	UnlabeledData(std::vector<InputT> const& points,std::size_t batchSize = base_type::DefaultBatchSize)
	: base_type(points,batchSize)
	{ }

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

	//~ ///\brief Construction from a data container and an index container.
	//~ ///
	//~ /// This is used internally by the LabeledData class.
	//~ /// IT IS RECOMMENDED NOT TO USE THIS FUNCTION DIRECTLY,
	//~ /// since this interface may be subject to change.
	//~ UnlabeledData(InputContainer const& container, typename base_type::IndexSets const & indexSets)
	//~ : base_type(container,indexSets){}

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
	self_type splice(typename base_type::iterator position){
		self_type right;
		right.m_data=m_data.splice(position);
		return right;
	}

	///\brief shuffles all elements in the entire dataset (that is, also across the batches)
	virtual void shuffle(){
		DiscreteUniform<Rng::rng_type> uni(Rng::globalRng);
		std::random_shuffle(m_data.elemBegin(),m_data.elemEnd(),uni);
	}
};

///
/// \brief Data set for base_typevised learning.
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
class LabeledData
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

	// TYPEDEFS
	typedef DataBatchPair<
		typename Batch<InputType>::type,
		typename Batch<LabelType>::type
	> value_type;

	typedef DataPair<
		InputType,
		LabelType
	> element_type;

	// ITERATORS
	typedef PairIterator<
		value_type,
		typename InputContainer::iterator,
		typename LabelContainer::iterator
	> iterator;
	typedef PairIterator<
		value_type,
		typename InputContainer::const_iterator,typename LabelContainer::const_iterator
	> const_iterator;

	typedef PairIterator<
		element_type,
		typename InputContainer::element_iterator,
		typename LabelContainer::element_iterator
	> element_iterator;
	typedef PairIterator<
		element_type,
		typename InputContainer::const_element_iterator,
		typename LabelContainer::const_element_iterator
	> const_element_iterator;

	// TYPEDEFS FOR REFERENCES
	typedef typename boost::iterator_reference<iterator>::type reference;
	typedef typename boost::iterator_reference<const_iterator>::type const_reference;
	typedef typename boost::iterator_reference<element_iterator>::type element_reference;
	typedef typename boost::iterator_reference<const_element_iterator>::type const_element_reference;

	// ITERATORS FOR BATCHES
	///\brief Iterator to the beginning of the container with respect to the batches.
	const_iterator begin() const{
		return const_iterator(m_data.begin(), m_label.begin());
	}

	///\brief Iterator to the end of the container with respect to the batches.
	const_iterator end() const{
		return const_iterator(m_data.end(), m_label.end());
	}

	///\brief Iterator to the beginning of the container with respect to the batches.
	iterator begin(){
		return iterator(m_data.begin(), m_label.begin());
	}

	///\brief Iterator to the end of the container with respect to the batches.
	iterator end(){
		return iterator(m_data.end(), m_label.end());
	}

	// ITERATORS FOR ELEMENTS
	///\brief Iterator to the beginning of the container with respect to the elements.
	const_element_iterator elemBegin() const{
		return const_element_iterator(m_data.elemBegin(), m_label.elemBegin());
	}

	///\brief Iterator to the end of the container with respect to the elements.
	const_element_iterator elemEnd() const{
		return const_element_iterator(m_data.elemEnd(), m_label.elemEnd());
	}

	///\brief Iterator to the beginning of the container with respect to the elements.
	element_iterator elemBegin(){
		return element_iterator(m_data.elemBegin(), m_label.elemBegin());
	}

	///\brief Iterator to the end of the container with respect to the elements
	element_iterator elemEnd(){
		return element_iterator(m_data.elemEnd(), m_label.elemEnd());
	}

	// RANGE ACCESS FOR ELEMENTS
	typedef boost::iterator_range<element_iterator> element_range;
	typedef boost::iterator_range<const_element_iterator> const_element_range;

	///\brief Returns the  half open range of elements [elemBegin(),elemEnd()).
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	const_element_range elements()const{
		return const_element_range(elemBegin(),elemEnd());
	}
	///\brief Returns the  half open range of elements [elemBegin(),elemEnd()).
	///
	///It is compatible to boost::range and STL and can be used whenever an algorithm requires
	///element access via begin()/end() in which case data.elements() provides the correct interface
	element_range elements(){
		return element_range(elemBegin(),elemEnd());
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
	LabeledData(std::vector<InputType> const& inputs, std::vector<LabelType> const& labels, std::size_t batchSize = DefaultBatchSize)
	: m_data(inputs,batchSize),
	  m_label(labels,batchSize)
	{
		SHARK_CHECK(inputs.size() == labels.size(), "[LabeledData::LabeledData] number of inputs and number of labels must agree");
	}

	///\brief Construction from data, subsets are defined by the inputs part.
	///
	///Beware that, when calling this constructors the organization of batches must be equal in both containers. This
	///Constructor won't split the data!
	LabeledData(Data<InputType> const& inputs, Data<LabelType> const& labels)
	: m_data(inputs), m_label(labels)
	{
		SHARK_CHECK(inputs.numberOfElements() == labels.numberOfElements(), "[LabeledData::LabeledData] number of inputs and number of labels must agree");
#ifndef DNDEBUG
		for(std::size_t i  = 0; i != inputs.size(); ++i){
			SIZE_CHECK(shark::size(inputs.batch(i))==shark::size(labels.batch(i)));
		}
#endif
	}

	// SIZE

	///\brief Returns the number of batches of the set.
	std::size_t size() const{
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

	element_reference operator()(std::size_t i){
		return *(elemBegin()+i);
	}
	const_element_reference operator()(std::size_t i) const{
		return *(elemBegin()+i);
	}

	// BATCH ACCESS
	reference batch(std::size_t i){
		SIZE_CHECK(i < size());
		return *(begin()+i);
	}
	const_reference batch(std::size_t i) const{
		SIZE_CHECK(i < size());
		return *(begin()+i);
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
		std::random_shuffle(elemBegin(),elemEnd(),uni);
	}

	void splitBatch(iterator position, std::size_t elementIndex){
		m_data.splitBatch(position.iterators().first,elementIndex);
		m_label.splitBatch(position.iterators().second,elementIndex);
	}

	///\brief Splits the container into two independent parts. The left part remains in the container, the right is stored as return type
	///
	///Order of elements remain unchanged. The SharedVector is not allowed to be shared for
	///this to work.
	self_type splice(iterator position){
		InputContainer dataRight = m_data.splice(position.iterators().first);
		LabelContainer labelsRight = m_label.splice(position.iterators().second);
		return self_type(dataRight,labelsRight);
	}

	///\brief Reorders the batch structure in the container to that indicated by the batchSizes vector
	///
	///After the operation the container will contain batchSizes.size() batchs with the i-th batch having size batchSize[i].
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

	//~ ///\brief  Fill in the named subset. If the subset does not exist the method throws an exception.
	//~ void namedSubset(std::string const& name, self_type& subset) const{
		//~ IndexSet const& indices = m_indexSets.findSubset(name);
		//~ subset.m_data = InputContainer(m_data,indices);
		//~ subset.m_label = LabelContainer(m_label,indices);
	//~ }

	//~ ///\brief Fill in the named subset and its complement. If the subset does not exist the method throws an exception.
	//~ void namedSubset(std::string const& name, self_type& subset, self_type& complement) const{
		//~ IndexSet const& indices = m_indexSets.findSubset(name);
		//~ IndexSet comp;
		//~ detail::complement(indices,this->size(),comp);
		//~ subset.m_data = InputContainer(m_data,indices);
		//~ subset.m_label = LabelContainer(m_label,indices);
		//~ complement.m_data = InputContainer(m_data,comp);
		//~ complement.m_label = LabelContainer(m_label,comp);
	//~ }

	//~ ///\brief  Create a new name, under which the given list of indices is stored as a named subset.
	//~ void createNamedSubset(std::string const& name, IndexSet const& indices){
		//~ m_indexSets.createNamedSubset(name, indices);
	//~ }

	//~ ///\brief Create a new name, under which a previously created subset is stored
	//~ ///
	//~ ///of course the subset must be previously created by this instance or one of
	//~ ///it's non-independent siblings.
	//~ void createNamedSubset(std::string const& name, self_type const& subset){
		//~ //unfortunately, we need to convert the batch vector of the set into an indexset...
		//~ std::vector<std::size_t> indizes(subset.m_data.size());
		//~ for(std::size_t i = 0; i != subset.m_data.size(); ++i){
			//~ for(std::size_t j = 0; j != m_data.size(); ++j){
				//~ if(m_data.pointer(j) == subset.m_data.pointer(i)){
					//~ indizes[i]=j;
					//~ break;
				//~ }
			//~ }
		//~ }
		//~ m_indexSets.createNamedSubset(name, indizes);
	//~ }
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

/**
 * \ingroup shark_globals
 *
 * @{
 */

///brief  Outstream of elements for labeled data.
template<class T, class U>
std::ostream &operator << (std::ostream &stream, const LabeledData<T, U>& d) {
	typedef typename LabeledData<T, U>::const_element_reference reference;
	BOOST_FOREACH(reference elem,d.elements())
		stream << elem.input << " [" << elem.label <<"]"<< "\n";
	return stream;
}


	// FUNCTIONS FOR DIMENSIONALITY


///\brief Return the number of classes of a set of class labels with unsigned int label encoding
inline unsigned int numberOfClasses(Data<unsigned int> const& labels){
	unsigned int classes = 0;
	BOOST_FOREACH(Data<unsigned int>::const_reference labelBatch, labels){
		classes = std::max(classes,*std::max_element(labelBatch.begin(),labelBatch.end()));
	}
	return classes+1;
}

///\brief Returns the number of members of each class in the dataset.
inline std::vector<std::size_t> classSizes(Data<unsigned int> const& labels){
	std::vector<std::size_t> classCounts(numberOfClasses(labels),0u);
	for(std::size_t i = 0; i != labels.size(); ++i){
		for(std::size_t j = 0; j != size(labels.batch(i)); ++j){
			classCounts[labels.batch(i)(j)]++;
		}
	}
	return classCounts;
}

/// Return the dimensionality of a  dataset.
template <class InputType>
unsigned int dataDimension(Data<InputType> const& dataset){
	SHARK_ASSERT(dataset.numberOfElements() > 0);
	return dataset.elemBegin()->size();
}

/// Return the input dimensionality of a labeled dataset.
template <class InputType, class LabelType>
unsigned int inputDimension(LabeledData<InputType, LabelType> const& dataset){
	SHARK_ASSERT(dataset.numberOfElements() > 0);
	return dataset.elemBegin()->input.size();
}

/// Return the label/output dimensionality of a labeled dataset.
template <class InputType, class LabelType>
unsigned int labelDimension(LabeledData<InputType, LabelType> const& dataset){
	SHARK_ASSERT(dataset.numberOfElements() > 0);
	return dataset.elemBegin()->label.size();
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
	return dataset.elemBegin()->label.size();
}
///\brief Returns the number of members of each class in the dataset.
template<class InputType, class LabelType>
inline std::vector<std::size_t> classSizes(LabeledData<InputType, LabelType> const& dataset){
	return classSizes(dataset.labels());
}

	// TRANSFORMATION
///\brief Transforms a dataset using a Function f and returns the transformed result.
//TODO: implement more efficient
template<class T,class Functor>
Data<T> transform(Data<T> const& data, Functor f){
	Data<T> result(data);
	result.makeIndependent();
	for(std::size_t i = 0; i != data.size(); ++i)
		for(std::size_t j = 0; j != shark::size(data.batch(i)); ++j)
			get(result.batch(i),j) = f(get(data.batch(i),j));
	return result;
}

///\brief Transforms the inputs of a dataset and return the transformed result.
template<class I,class L, class Functor>
LabeledData<I,L> transformInputs(LabeledData<I,L> const& data, Functor f){
	return LabeledData<I,L>(transform(data.inputs(),f),data.labels());
}
///\brief Transforms the labels of a dataset and returns the transformed result.
template<class I,class L, class Functor>
LabeledData<I,L> transformLabels(LabeledData<I,L> const& data, Functor f){
	return LabeledData<I,L>(data.inputs(),transform(data.labels(),f));
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
DatasetT splitAtElement(DatasetT& data, std::size_t index){
	SIZE_CHECK(index<data.numberOfElements());
	//create vector with batch sizes and split the batch in which the element trainSize is found
	typename DatasetT::iterator batch = data.begin();
	std::size_t batchEnd = size(*batch);
	while(batchEnd <=index){
		++batch;
		std::size_t batchSize = size(*batch);
		batchEnd+=batchSize;
	}
	std::size_t splitPoint = size(*batch) -(batchEnd-index);
	std::size_t splitBatch = batch-data.begin();
	data.splitBatch(batch,splitPoint);
	++splitBatch;
	return data.splice(data.begin()+splitBatch);
}


///\brief reorders the dataset such, that points are grouped by labels
///
/// The elements ar enot only reordered but the batches are also resized such, that every batch
/// only contains elemnts of one class. This method must be used in order to use binarySubproblem. 
template<class I>
void repartitionByClass(LabeledData<I,unsigned int>& data,std::size_t batchSize = LabeledData<I,unsigned int>::DefaultBatchSize){
	std::vector<std::size_t > classCounts = classSizes(data);
	std::vector<std::size_t > partitioning;//new, optimal partitioning of the data according to the batch sizes
	std::vector<std::size_t > classStart;//at which batch the elements of the class are starting
	detail::batchPartitioning(classCounts, partitioning, classStart, batchSize);

	data.repartition(partitioning);
	std::sort(data.elemBegin(),data.elemEnd(),detail::ElementSort());//todo we are lying here, use bidirectional iterator sort.
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

	//find first class
	std::size_t start= 0;
	for(;get(data.batch(start),0).label != smaller;++start);
	SHARK_CHECK(start != data.size(), "[shark::binarySubProblem] class does not exist");

	//copy batch indices of first class
	for(;start != data.size() && get(data.batch(start),0).label == smaller; ++start)
		indexSet.push_back(start);

	//find second class
	for(;start != data.size() && get(data.batch(start),0).label != bigger;++start);
	SHARK_CHECK(start != data.size(), "[shark::binarySubProblem] class does not exist");

	//copy batch indices of second class
	for(;get(data.batch(start),0).label == bigger; ++start)
		indexSet.push_back(start);

	return transformLabels(indexedSubset(data,indexSet), detail::TransformBinaryLabels(zeroClass,oneClass));
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

}
/** @*/
#endif
