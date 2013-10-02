//===========================================================================
/*!
 *  \brief Fast lookup for elements in constant datasets
 *
 *
 *
 *  \author  O. Krause
 *  \date    2012
 *
 *  \par Copyright (c) 2010-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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


#ifndef SHARK_DATA_DATAVIEW_H
#define SHARK_DATA_DATAVIEW_H

#include <shark/Data/Dataset.h>
#include <shark/Core/utility/functional.h>
#include <boost/type_traits/is_const.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/bind.hpp>
#include <boost/range/algorithm/copy.hpp>
namespace shark {
/// \brief Constant time Element-Lookup for Datasets
///
/// Datasets are only fast for random lookup of batches. Since batch sizes can be arbitrary structured and
/// changed by the user, there is no way for the Data and LabeledData classes to provide a fast random access
/// to single elements.  Still, this property is needed quite often, for example for creating subsets, 
/// randomize data or tree structures. 
/// A View stores the position of every element in a dataset. So it has constant time access to the elements but 
/// it also requires linear memory in the number of elements in the set. This is typically small compared 
/// to the size of the set  itself, but construction imposes an considerable overhead.
///
/// In contrast to (Un)LabeledData , which is centered around batches, the View is centered around single elements,
/// so it's iterators iterate over the elements. batch access is again by using batch(start,end) which on the fly
/// creates the batch from the elements in [start,end]
/// for a better support for bagging an index method is added which returns the position of the element in the
/// underlying dataset. Also the iterators are indexed and return this index. (todo: is this useful?) 
template <class DatasetType>//template parameter can be const!
class DataView
{
public:
	typedef typename boost::remove_const<DatasetType>::type dataset_type;//(non const) type of the underlying dataset
	typedef typename dataset_type::element_type value_type;
	typedef typename dataset_type::const_element_reference const_reference;
	typedef typename dataset_type::batch_type batch_type;
	//we want to support immutable as well as mutable datasets. So we query whether the dataset is mutable and 
	//change the reference type to const if the dataset is immutable
	typedef typename boost::mpl::if_<
		boost::is_const<DatasetType>,
		typename dataset_type::const_element_reference,
		typename dataset_type::element_reference
	>::type reference;
	
private:
	typedef typename boost::mpl::if_<
		boost::is_const<DatasetType>,
		typename dataset_type::const_batch_range,
		typename dataset_type::batch_range
	>::type batch_range;
	template<class Reference, class View>
	class IteratorBase: public boost::iterator_facade_fixed<
		IteratorBase<Reference,View>,
		value_type,
		std::random_access_iterator_tag,
		Reference
	>{
	public:
		IteratorBase(){}
	
		IteratorBase(View& view, std::size_t position)
		: mpe_view(&view),m_position(position) {}
	
		template<class R,class V>
		IteratorBase(IteratorBase<R,V> const& other)
		: mpe_view(other.mpe_view),m_position(other.position){}
		
		/// \brief returns the position of the element referenced by the iterator inside the dataset
		///
		/// This is usefull for bagging, when identical elements between several susbsets are to be identified
		std::size_t index()const{
			return mpe_view->index(m_position);
		}

	private:
		friend class boost::iterator_core_access_fixed;
		template <class, class> friend class IteratorBase;

		void increment() {
			++m_position;
		}
		void decrement() {
			--m_position;
		}

		void advance(std::ptrdiff_t n){
			m_position+=n;
		}
	
		template<class R,class V>
		std::ptrdiff_t distance_to(IteratorBase<R,V> const& other) const{
			return (std::ptrdiff_t)other.m_position - (std::ptrdiff_t)m_position;
		}
	
		template<class R,class V>
		bool equal(IteratorBase<R,V> const& other) const{
			return m_position == other.m_position;
		}
		Reference dereference() const { 
			return (*mpe_view)[m_position];
		}

		View* mpe_view;
		std::size_t m_position;
	};
public:
	typedef IteratorBase<reference,DataView<DatasetType> > iterator;
	typedef IteratorBase<const_reference, DataView<DatasetType> const > const_iterator;
	
	DataView(){}
	DataView(DatasetType& dataset)
	:m_dataset(dataset),m_indices(dataset.numberOfElements())
	{
		std::size_t index = 0;
		for(std::size_t i = 0; i != dataset.numberOfBatches(); ++i){
			std::size_t batchSize = shark::size(dataset.batch(i));
			for(std::size_t j = 0; j != batchSize; ++j,++index){
				m_indices[index].batch = i;
				m_indices[index].positionInBatch = j;
				m_indices[index].datasetIndex = index;
			}
		}
	}
	
	/// create a subset of the dataset type using only the elemnt indexed by indices
	template<class IndexRange>
	DataView(DataView<DatasetType> const& view, IndexRange const& indices)
	:m_dataset(view.m_dataset),m_indices(shark::size(indices))
	{
		for(std::size_t i = 0; i != m_indices.size(); ++i)
			m_indices[i] = view.m_indices[indices[i]];
	}
	
	reference operator[](std::size_t position){
		SIZE_CHECK(position < size());
		Index const& index = m_indices[position];
		return get(m_dataset.batch(index.batch),index.positionInBatch);
	}
	const_reference operator[](std::size_t position) const{
		SIZE_CHECK(position < size());
		Index const& index = m_indices[position];
		return get(m_dataset.batch(index.batch),index.positionInBatch);
	}
	/// \brief returns the position of the element inside the dataset
	///
	/// This is usefull for bagging, when identical elements between several susbsets are to be identified
	std::size_t index(std::size_t position)const{
		return m_indices[position].datasetIndex;
	}
	
	std::size_t size() const{
		return m_indices.size();
	}
	
	iterator begin(){
		return iterator(*this, 0);
	}
	const_iterator begin() const{
		return const_iterator(*this, 0);
	}
	iterator end(){
		return iterator(*this, size());
	}
	const_iterator end() const{
		return const_iterator(*this, size());
	}
	
	dataset_type const& dataset()const{
		return m_dataset;
	}
private:
	dataset_type m_dataset;
	// Stores for an element of the dataset, at which position of which batch it is located
	// as well as the real index of the element inside the dataset
	struct Index{
		std::size_t batch;//the batch in which the element is located
		std::size_t positionInBatch;//at which position in the batch it is
		std::size_t datasetIndex;//index inside the dataset
	};
	std::vector<Index> m_indices;//stores for every element of the set it's position inside the dataset
};

/// \brief creates a subset of a DataView with elements indexed by indices
///
/// \param view the view for which the subset is to be created
/// \param indizes the index of the elements to be stored in the view 
template<class DatasetType,class IndexRange>
DataView<DatasetType> subset(DataView<DatasetType> const& view, IndexRange const& indizes){
	//O.K. todo: Remove constructor later on, this is a quick fix.
	return DataView<DatasetType>(view,indizes);
}

/// \brief creates a random subset of a DataView with given size
///
/// \param view the view for which the subset is to be created
/// \param size the size of the subset
template<class DatasetType>
DataView<DatasetType> randomSubset(DataView<DatasetType> const& view, std::size_t size){
	std::vector<std::size_t> indices(view.size());
	boost::iota(indices,0);
	partial_shuffle(indices.begin(),indices.begin()+size,indices.end());
	return subset(view,boost::make_iterator_range(indices.begin(),indices.begin()+size));
}

/// \brief Creates a batch given a set of indices
///
/// \param view the view from which the batch is to be created
/// \param indizes the set of indizes defining the batch
template<class DatasetType,class IndexRange>
typename DataView<DatasetType>::batch_type subBatch(
	DataView<DatasetType> const& view, 
	IndexRange const& indizes
){
	//create a subset of the view containing the elements of the batch
	DataView<DatasetType> batchElems = subset(view,indizes);
	
	//and now use the batch range construction to create it
	return Batch<typename DatasetType::element_type>::createBatch(batchElems);
}

/// \brief Creates a random batch of a given size
///
/// \param view the view from which the batch is to be created
/// \param size the size of the batch
template<class DatasetType>
typename DataView<DatasetType>::batch_type randomSubBatch(
	DataView<DatasetType> const& view, 
	std::size_t size
){
	std::vector<std::size_t> indices(view.size());
	boost::iota(indices,0);
	partial_shuffle(indices.begin(),indices.begin()+size,indices.end());
	return subBatch(view,boost::make_iterator_range(indices.begin(),indices.begin()+size));
}

/// \brief Creates a View from a dataset.
///
/// This is just a helper function to omit the actual type of the view
///
/// \param set the dataset from which to create the view
template<class DatasetType>
DataView<DatasetType>  toView(DatasetType& set){
	return DataView<DatasetType>(set);
}

/// \brief Creates a new dataset from a View.
///
/// When the elements of a View needs to be processed repeatedly it is often better to use
/// the packed format of the Dataset again, since then the faster batch processing can be used
///
/// \param view the view from which to create the new dataset
/// \param batchSize the size of the batches in the dataset
template<class T>
typename DataView<T>::dataset_type 
toDataset(DataView<T> const& view, std::size_t batchSize = DataView<T>::dataset_type::DefaultBatchSize){
	if(view.size() == 0)
		return typename DataView<T>::dataset_type();
	//O.K. todo: this is slow for sparse elements, use subBatch or something similar.
	std::size_t elements = view.size();
	typename DataView<T>::dataset_type dataset(elements,view[0],batchSize);
	std::size_t batches = dataset.numberOfBatches();
	
	std::size_t element = 0;
	for(std::size_t i = 0; i != batches; ++i){
		std::size_t batchSize = shark::size(dataset.batch(i));
		for(std::size_t j = 0; j != batchSize; ++j, ++element){
			get(dataset.batch(i),j) = view[element];
		}
	}
	return dataset;
}

/// Return the number of classes (size of the label vector)
/// of a classification dataset with RealVector label encoding.
template <class DatasetType>
unsigned int numberOfClasses(DataView<DatasetType> const& view){
	return numberOfClasses(view.dataset());
}

/// Return the input dimensionality of the labeled dataset represented by the view
template <class DatasetType>
std::size_t inputDimension(DataView<DatasetType> const& view){
	return inputDimension(view.dataset());
}
/// Return the label dimensionality of the labeled dataset represented by the view
template <class DatasetType>
std::size_t labelDimension(DataView<DatasetType> const& view){
	return labelDimension(view.dataset());
}

/// Return the dimensionality of the dataset represented by the view
template <class DatasetType>
std::size_t dataDimension(DataView<DatasetType> const& view){
	return dataDimension(view.dataset());
}


/** @*/
}
#endif
