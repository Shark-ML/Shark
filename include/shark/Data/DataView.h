//===========================================================================
/*!
 * 
 *
 * \brief       Fast lookup for elements in constant datasets
 * 
 * 
 * 
 * 
 *
 * \author      O. Krause
 * \date        2012
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


#ifndef SHARK_DATA_DATAVIEW_H
#define SHARK_DATA_DATAVIEW_H

#include <shark/Core/utility/functional.h>
#include <numeric>
#include <shark/Data/BatchInterface.h>
namespace shark {
	
	
namespace constants{
	const std::size_t DefaultBatchSize = 256;
}

/// \brief Constant time Element-Lookup for Datasets
///
/// Datasets are fast for random lookup of batches. Since batch sizes can be arbitrary structured and
/// changed by the user, there is no way for the Data and LabeledData classes to provide fast random access
/// to single elements. Still, this property is needed quite often, for example for creating subsets,
/// randomize data or tree structures. 
/// A View stores the position of every element in a dataset. So it has constant time access to the elements but
/// it also requires linear memory in the number of elements in the set. This is typically small compared
/// to the size of the set itself, but construction imposes an considerable overhead.
///
/// In contrast to (Un)LabeledData, which is centered around batches, the View is centered around single elements,
/// so its iterators iterate over the elements.
/// For a better support for bagging an index method is added which returns the position of the element in the
/// underlying data container. Also the iterators are indexed and return this index.
template <class DatasetType>   //template parameter can be const!
class DataView
{
public:
	typedef typename std::remove_const<DatasetType>::type dataset_type;   //(non const) type of the underlying dataset
	typedef typename dataset_type::element_type value_type;
	typedef typename dataset_type::value_type batch_type;
	typedef typename dataset_type::shape_type shape_type;
	// We want to support immutable as well as mutable datasets. So we query whether the dataset
	// is mutable and change the reference type to const if the dataset is immutable.
	typedef typename detail::batch_to_reference<
		typename std::conditional<
			std::is_const<DatasetType>::value,
			batch_type const,
			batch_type
		>::type
	>::type reference;
	typedef typename detail::batch_to_reference<batch_type const>::type const_reference;

	typedef IndexingIterator<DataView> iterator;
	typedef IndexingIterator<DataView<DatasetType> const > const_iterator;

	DataView(){}
	DataView(DatasetType& dataset)
	:m_dataset(dataset),m_indices(m_dataset.numberOfElements())
	{
		std::size_t index = 0;
		for(std::size_t i = 0; i != m_dataset.size(); ++i){
			std::size_t batchSize = Batch<value_type>::size(m_dataset[i]);
			for(std::size_t j = 0; j != batchSize; ++j,++index){
				m_indices[index].batch = i;
				m_indices[index].positionInBatch = j;
				m_indices[index].datasetIndex = index;
			}
		}
	}
	
	DataView(DatasetType&& dataset)
	:m_dataset(std::move(dataset)),m_indices(m_dataset.numberOfElements())
	{
		std::size_t index = 0;
		for(std::size_t i = 0; i != m_dataset.size(); ++i){
			std::size_t batchSize = Batch<value_type>::size(m_dataset[i]);
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
	:m_dataset(view.m_dataset),m_indices(indices.size())
	{
		for(std::size_t i = 0; i != m_indices.size(); ++i)
			m_indices[i] = view.m_indices[indices[i]];
	}
	
	shape_type shape() const{
		return dataset().shape();
	}

	reference operator[](std::size_t position){
		SIZE_CHECK(position < size());
		Index const& index = m_indices[position];
		return Batch<value_type>::get(static_cast<DatasetType&>(m_dataset)[index.batch],index.positionInBatch);
	}
	const_reference operator[](std::size_t position) const{
		SIZE_CHECK(position < size());
		Index const& index = m_indices[position];
		return getBatchElement(m_dataset[index.batch],index.positionInBatch);
	}
	
	reference front(){
		SIZE_CHECK(size() != 0);
		return (*this)[0];
	}
	const_reference front()const{
		SIZE_CHECK(size() != 0);
		return (*this)[0];
	}
	reference back(){
		SIZE_CHECK(size() != 0);
		return (*this)[size()-1];
	}
	const_reference back()const{
		SIZE_CHECK(size() != 0);
		return (*this)[size()-1];
	}

	/// \brief Position of the element in the dataset.
	///
	/// This is useful for bagging, when identical elements among
	/// several subsets are to be identified.
	std::size_t index(std::size_t position)const{
		return m_indices[position].datasetIndex;
	}

	/// \brief Index of the batch holding the element.
	std::size_t batch(std::size_t position) const {
		return m_indices[position].batch;
	}

	/// \brief Index inside the batch holding the element.
	std::size_t positionInBatch(std::size_t position) const {
		return m_indices[position].positionInBatch;
	}
	
	
	/// \brief exchanges elements i and j in the Dataview.
	///
	/// This does not change the order in the underlying dataset.
	void swapElements(std::size_t i, std::size_t j){
		std::swap(m_indices[i], m_indices[j]);
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
	std::iota(indices.begin(),indices.end(),0);
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
	return createBatch(batchElems);
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
	std::iota(indices.begin(),indices.end(),0);
	partial_shuffle(indices.begin(),indices.begin()+size,indices.end());
	return subBatch(view,boost::make_iterator_range(indices.begin(),indices.begin()+size));
}

/// \brief Creates a View from a dataset.
///
/// This is just a helper function to omit the actual type of the view
///
/// \param set the dataset from which to create the view
template<class DatasetType>
DataView<typename std::remove_reference<DatasetType>::type >  elements(DatasetType&& set){
	return DataView<typename std::remove_reference<DatasetType>::type>(std::forward<DatasetType>(set));
}


/// \brief Creates a new dataset from a View.
///
/// \param view the view from which to create the new dataset
/// \param maximumBatchSize the size of the batches in the dataset
template<class T>
typename DataView<T>::dataset_type 
toDataset(DataView<T> const& view, std::size_t maximumBatchSize = constants::DefaultBatchSize){
	if(view.size() == 0)
		return typename DataView<T>::dataset_type();
	
	typename DataView<T>::dataset_type dataset(view.size(), view.shape(), maximumBatchSize);
	
	std::size_t batchStart = 0;
	for(std::size_t i = 0; i != dataset.size(); ++i){
		std::size_t batchEnd = batchStart + batchSize(dataset[i]);
		dataset[i] = createBatch<typename DataView<T>::value_type>(view.begin()+batchStart, view.begin()+batchEnd);
		batchStart = batchEnd;
	}
	return dataset;
}

/// \brief Creates a new dataset from a View.
///
/// \param view the view from which to create the new dataset
/// \param batchSizes the sizes of each individual batch
template<class T>
typename DataView<T>::dataset_type 
toDataset(DataView<T> const& view, std::vector<std::size_t> const& batchSizes){
	typename DataView<T>::dataset_type dataset(batchSizes, view.shape());
	
	std::size_t batchStart = 0;
	for(std::size_t i = 0; i != dataset.size(); ++i){
		std::size_t batchEnd = batchStart + batchSize(dataset[i]);
		dataset[i] = createBatch<typename DataView<T>::value_type>(view.begin()+batchStart, view.begin()+batchEnd);
		batchStart = batchEnd;
	}
	return dataset;
}

/// Return the number of classes (size of the label vector)
/// of a classification dataset with RealVector label encoding.
template <class DatasetType>
std::size_t numberOfClasses(DataView<DatasetType> const& view){
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
