//===========================================================================
/*!
 *  \brief Tools for cross-validation
 *
 *  \author O.Krause
 *  \date 2010-2012
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

#ifndef SHARK_DATA_CVDATASETTOOLS_H
#define SHARK_DATA_CVDATASETTOOLS_H

#include <shark/Data/Dataset.h>
#include <shark/Rng/DiscreteUniform.h>
#include <algorithm>
//#include <shark/SharkDefs.h>
#include <shark/Data/DataView.h>

namespace shark {

template<class DatasetTypeT>
class CVFolds {
public:
	typedef DatasetTypeT DatasetType;
	typedef typename DatasetType::IndexSet IndexSet;

	/// \brief Creates an empty set of folds.
	CVFolds() {}
	///\brief partitions set in validation folds indicated by the second argument.
	///
	///It is assumed that the batches in set are stored such that the batches partitionStart[i]...partitionStart[i+1] form a validation set.
	///The last batch is partitionStart[n]....set.size() thus there is no need to add the last element.
	CVFolds(
	    DatasetType const &set,
	    std::vector<std::size_t> const &partitionStart
	) : m_dataset(set) {
		std::size_t numberOfPartitions = partitionStart.size();
		std::size_t foldElementStart = 0; //element index of the starting fold
		for (std::size_t partition = 0; partition != numberOfPartitions; partition++) {
			m_foldElementStart.push_back(foldElementStart);
			std::size_t partitionSize = (partition+1 == numberOfPartitions) ? set.numberOfBatches() : partitionStart[partition+1];
			partitionSize -= partitionStart[partition];
			//create the set with the indices of the validation set of the current partition
			//also update the starting element
			IndexSet validationIndizes(partitionSize);
			for (std::size_t batch = 0; batch != partitionSize; ++batch) {
				validationIndizes[batch]=batch+partitionStart[partition];
				foldElementStart += boost::size(set.batch(validationIndizes[batch]));
			}
			//we need the training part of the set for creation of subsets, but this is
			//just the complement of the validation set.
			IndexSet partitionIndizes;
			detail::complement(validationIndizes,set.numberOfBatches(),partitionIndizes);

			//now add the partition to the folds.
			m_folds.push_back(partitionIndizes);
		}
		m_foldElementStart.push_back(foldElementStart);//this is now the total number of elements
		
		//internal sanity check
		SIZE_CHECK(set.numberOfElements() == foldElementStart);
	}

	DatasetType training(std::size_t i) const {
		SIZE_CHECK(i < m_folds.size());

		return indexedSubset(m_dataset, m_folds[i]);
	}
	DatasetType validation(std::size_t i) const {
		SIZE_CHECK(i < m_folds.size());

		//TODO: make use of the range structure of the validation fold.
		IndexSet validationFold;
		detail::complement(m_folds[i], m_dataset.numberOfBatches(), validationFold);

		return indexedSubset(m_dataset, validationFold);
	}

	///\brief returns the indices that make up the i-th fold
	IndexSet const &foldIndices(std::size_t i)const {
		SIZE_CHECK(i < m_folds.size());
		return m_folds[i];
	}

	///\brief Returns the number of folds of the dataset.
	std::size_t size()const {
		return m_folds.size();
	}

	/// \brief Returns the overall number of elements in the partitioned dataset
	std::size_t numberOfElements() const {
		return m_foldElementStart[size()];
	}
	/// \brief Returns the overall number of elements in the fold
	std::size_t numberOfElements(std::size_t i) const {
		SIZE_CHECK(i < size());
		return m_foldElementStart[i+1]-m_foldElementStart[i];
	}
	
	/// \brief rturns the index of the first element of the i-th fold in the dataset.
	std::size_t foldElementStart(std::size_t i) const {
		SIZE_CHECK(i < size());
		return m_foldElementStart[i];
	}
	
	/// \brief Returns the dataset underying the folds
	DatasetType const& dataset()const{
		return m_dataset;
	}

private:
	DatasetType m_dataset;
	std::vector<IndexSet> m_folds;
	std::vector<std::size_t> m_foldElementStart;
};

namespace detail {

///\brief Version of createCVSameSizeBalanced which works regardless of the label type
///
/// Instead of a class label to interpret, this class uses a membership vector for every class which
/// members[k][i] returns the positon of the i-th member of class k in the set.
template<class I, class L>
CVFolds<LabeledData<I,L> > createCVSameSizeBalanced(
    LabeledData<I,L> &set,
    std::size_t numberOfPartitions,
    std::vector< std::vector<std::size_t> > members,
    std::size_t batchSize
) {
	std::size_t numInputs = set.numberOfElements();
	std::size_t numClasses = members.size();

	//shuffle elements in members
	DiscreteUniform< Rng::rng_type > uni(shark::Rng::globalRng) ;
	for (std::size_t c = 0; c != numClasses; c++) {
		std::random_shuffle(members[c].begin(), members[c].end(), uni);
	}

	//calculate number of elements per validation subset in the new to construct container
	std::size_t nn = numInputs / numberOfPartitions;
	std::size_t leftOver = numInputs % nn;
	std::vector<std::size_t> validationSize(numberOfPartitions,nn);
	for (std::size_t partition = 0; partition != leftOver; partition++) {
		validationSize[partition]++;
	}

	//calculate the size of the batches for every validation part
	std::vector<std::size_t> partitionStart;
	std::vector<std::size_t> batchSizes;
	std::size_t numBatches = batchPartitioning(validationSize,partitionStart,batchSizes,batchSize);


	LabeledData<I,L> newSet(numBatches);//set of empty batches
	DataView<LabeledData<I,L> > setView(set);//fast access to single elements of the original set
	std::vector<std::size_t> validationSetStart = partitionStart;//current index for the batch of every fold
	//partition classes into the validation subsets of newSet
	std::size_t fold = 0;//current fold
	std::vector<std::vector<std::size_t> > batchElements(numberOfPartitions);
	for (std::size_t c = 0; c != numClasses; c++) {
		for (std::size_t i = 0; i != members[c].size(); i++) {
			std::size_t oldPos = members[c][i];
			std::size_t batchNumber = validationSetStart[fold];

			batchElements[fold].push_back(oldPos);

			//if all elements for the current batch are found, create it
			if (batchElements[fold].size() == batchSizes[batchNumber]) {
				newSet.batch(validationSetStart[fold]) = subBatch(setView,batchElements[fold]);
				batchElements[fold].clear();
				++validationSetStart[fold];
			}

			fold = (fold+1) % numberOfPartitions;
		}
	}

	//swap old and new set
	swap(set, newSet);

	//create folds
	return CVFolds<LabeledData<I,L> >(set,partitionStart);

}
}//namespace detail

/**
 * \ingroup shark_globals
 *
 * @{
 */

//! \brief Create a partition for cross validation
//!
//! The subset each training examples belongs to
//! is drawn independently and uniformly distributed.
//! For every partition, all but one subset form the
//! training set, while the remaining one is used for
//! validation. The partitions can be accessed using
//! getCVPartitionName
//!
//! \param set the input data for which the new partitions are created
//! \param numberOfPartitions  number of partitions to create
//! \param batchSize  maximum batch size
template<class I,class L>
CVFolds<LabeledData<I,L> > createCVIID(LabeledData<I,L> &set,
        std::size_t numberOfPartitions,
        std::size_t batchSize=Data<I>::DefaultBatchSize) {
	std::vector<std::size_t> indices(set.numberOfElements());
	for (std::size_t i=0; i != set.numberOfElements(); i++)
		indices[i] = Rng::discrete(0, numberOfPartitions - 1);
	return createCVIndexed(set,numberOfPartitions,indices,batchSize);
}

//! \brief Create a partition for cross validation
//!
//! Every subset contains (approximately) the same
//! number of elements. For every partition, all
//! but one subset form the training set, while the
//! remaining one is used for validation. The partitions
//! can be accessed using getCVPartitionName
//!
//! \param numberOfPartitions  number of partitions to create
//! \param set the input data from which to draw the partitions
//! \param batchSize  maximum batch size
template<class I,class L>
CVFolds<LabeledData<I,L> > createCVSameSize(LabeledData<I,L> &set,std::size_t numberOfPartitions,std::size_t batchSize = LabeledData<I,L>::DefaultBatchSize) {
	std::size_t numInputs = set.numberOfElements();

	//calculate the number of validation examples for every partition
	std::vector<std::size_t> validationSize(numberOfPartitions);
	std::size_t inputsForValidation = numInputs / numberOfPartitions;
	std::size_t leftOver = numInputs - inputsForValidation * numberOfPartitions;
	for (std::size_t i = 0; i != numberOfPartitions; i++) {
		std::size_t vs=inputsForValidation+(i<leftOver);
		validationSize[i] =vs;
	}

	//calculate the size of batches for every validation part and their total number
	std::vector<std::size_t> partitionStart;
	std::vector<std::size_t> batchSizes;
	detail::batchPartitioning(validationSize,partitionStart,batchSizes,batchSize);

	set.repartition(batchSizes);
	set.shuffle();

	CVFolds<LabeledData<I,L> > folds(set,partitionStart);
	return folds;//set;
}


//! \brief Create a partition for cross validation
//!
//! Every subset contains (approximately) the same
//! number of elements. For every partition, all
//! but one subset form the training set, while the
//! remaining one is used for validation.
//!
//! \param numberOfPartitions  number of partitions to create
//! \param set the input data from which to draw the partitions
//! \param batchSize  maximum batch size
template<class I>
CVFolds<LabeledData<I,unsigned int> > createCVSameSizeBalanced(
	LabeledData<I,unsigned int> &set,std::size_t numberOfPartitions,
	std::size_t batchSize=Data<I>::DefaultBatchSize
){
	DataView<LabeledData<I,unsigned int> > setView(set);
	std::size_t numInputs = setView.size();
	std::size_t numClasses = numberOfClasses(set);

	//find members of each class
	std::vector< std::vector<std::size_t> > members(numClasses);
	for (std::size_t i = 0; i != numInputs; i++) {
		members[setView[i].label].push_back(i);
	}
	return detail::createCVSameSizeBalanced(set,numberOfPartitions,members,batchSize);

}

//! \brief Create a partition for cross validation
//!
//! Every subset contains (approximately) the same
//! number of elements. For every partition, all
//! but one subset form the training set, while the
//! remaining one is used for validation.
//! This function assumes one-hot encoding for the labels.
//!
//! \param set the input data from which to draw the partitions
//! \param numberOfPartitions  number of partitions to create
//! \param batchSize  maximum batch size
template<class I>
CVFolds<LabeledData<I,RealVector> > createCVSameSizeBalanced(
	LabeledData<I,RealVector> &set,
	std::size_t numberOfPartitions, 
	std::size_t batchSize=Data<I>::DefaultBatchSize
){
	DataView<LabeledData<I,RealVector> > setView(set);
	std::size_t numInputs = setView.size();

	//calculate number of classes
	std::size_t numClasses = numberOfClasses(set);

	//find members of each class
	std::vector< std::vector<std::size_t> > members(numClasses);
	for (std::size_t i = 0; i != numInputs; i++) {
		//find class represented by label
		//we first use max_element to get an iterator to the position of the maximum index
		//then we calculate the distance between class 0 and the position which gives the classID
		unsigned int classID = std::distance(
		        setView[i].label.begin(),
		        std::max_element(setView[i].label.begin(),setView[i].label.end())
		        );
		members[classID].push_back(i);
	}
	return detail::createCVSameSizeBalanced(set,numberOfPartitions,members,batchSize);

}


//! \brief Create a partition for cross validation from indices
//!
//! Create a partition from indices. The indices vector for each sample states of what
//! validation partition that sample should become a member. In other words, the index
//! maps a sample to a validation partition, meaning that it will become a part of the
//! training partition for all other folds.
//!
//! \param set partitions will be subsets of this set
//! \param numberOfPartitions  number of partitions to create
//! \param indices             partition indices of the examples in [0, ..., numberOfPartitions[.
//! \param batchSize  maximum batch size
template<class I,class L>
CVFolds<LabeledData<I,L> > createCVIndexed(
	LabeledData<I,L> &set,std::size_t numberOfPartitions,
	std::vector<std::size_t> indices,
	std::size_t batchSize=Data<I>::DefaultBatchSize
) {
	std::size_t numInputs = set.numberOfElements();
	SIZE_CHECK(indices.size() == numInputs);
	SIZE_CHECK(numberOfPartitions == *std::max_element(indices.begin(),indices.end())+1);

	//calculate the size of validation partitions
	std::vector<std::size_t> validationSize(numberOfPartitions,0);
	for (std::size_t input = 0; input != numInputs; input++) {
		validationSize[indices[input]]++;
	}

	//calculate the size of batches for every validation part and their total number
	std::vector<std::size_t> partitionStart;
	std::vector<std::size_t> batchSizes;
	std::size_t numBatches = detail::batchPartitioning(validationSize,partitionStart,batchSizes,batchSize);

	//construct a new set with the correct batch format from the old set
	LabeledData<I,L> newSet(numBatches);
	DataView<LabeledData<I,L> > setView(set);//fast access to single elements of the original set
	std::vector<std::size_t> validationSetStart = partitionStart;//current index for the batch of every parittion
	std::vector<std::vector<std::size_t> > batchElements(numberOfPartitions);
	for (std::size_t input = 0; input != numInputs; input++) {
		std::size_t partition = indices[input];
		batchElements[partition].push_back(input);

		//if all elements for the current batch are found, create it
		std::size_t batchNumber = validationSetStart[partition];
		if (batchElements[partition].size() == batchSizes[batchNumber]) {
			newSet.batch(validationSetStart[partition]) = subBatch(setView,batchElements[partition]);
			batchElements[partition].clear();
			++validationSetStart[partition];
		}
	}
	swap(set, newSet);
	//now we only need to create the subset itself
	return CVFolds<LabeledData<I,L> >(set,partitionStart);
}

// much more to come...

/** @}*/
}
#include "Impl/CVDatasetTools.inl"
#endif
