//===========================================================================
/*!
 *
 *
 * \brief       Tools for cross-validation
 *
 *
 *
 * \author      O.Krause
 * \date        2010-2012
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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

#ifndef SHARK_DATA_CVDATASETTOOLS_H
#define SHARK_DATA_CVDATASETTOOLS_H

#include <shark/Data/Dataset.h>
#include <shark/Rng/DiscreteUniform.h>
#include <algorithm>
//
#include <shark/Data/DataView.h>

#include <utility> //for std::pair

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
	/// The folds are given as the batch indices of the validation sets
	CVFolds(
	    DatasetType const &set,
	    std::vector<IndexSet> const &validationIndizes
	) : m_dataset(set),m_validationFolds(validationIndizes) {}
	
	CVFolds(
		DatasetType const &set,
		std::vector<std::size_t> const &foldStart
	) : m_dataset(set){
		for (std::size_t partition = 0; partition != foldStart.size(); partition++) {
			std::size_t partitionSize = (partition+1 == foldStart.size()) ? set.numberOfBatches() : foldStart[partition+1];
			partitionSize -= foldStart[partition];
			//create the set with the indices of the validation set of the current partition
			//also update the starting element
			IndexSet validationIndizes(partitionSize);
			for (std::size_t batch = 0; batch != partitionSize; ++batch) {
				validationIndizes[batch]=batch+foldStart[partition];
			}
			m_validationFolds.push_back(validationIndizes);
		}
	}

	DatasetType training(std::size_t i) const {
		SIZE_CHECK(i < size());

		return indexedSubset(m_dataset, trainingFoldIndices(i));
	}
	DatasetType validation(std::size_t i) const {
		SIZE_CHECK(i < size());

		return indexedSubset(m_dataset,validationFoldIndices(i));
	}

	///\brief returns the indices that make up the i-th validation fold
	IndexSet const &validationFoldIndices(std::size_t i)const {
		SIZE_CHECK(i < size());
		return m_validationFolds[i];
	}
	
	IndexSet trainingFoldIndices(std::size_t i)const {
		SIZE_CHECK(i < size());
		IndexSet trainingFold;
		detail::complement(m_validationFolds[i], m_dataset.numberOfBatches(), trainingFold);
		return trainingFold;
	}

	///\brief Returns the number of folds of the dataset.
	std::size_t size()const {
		return m_validationFolds.size();
	}

	//~ /// \brief Returns the overall number of elements in the partitioned dataset
	//~ std::size_t numberOfElements() const {
		//~ return m_foldElementStart[size()];
	//~ }
	//~ /// \brief Returns the overall number of elements in the i-th training fold
	//~ std::size_t numberOfTrainingElements(std::size_t i) const {
		//~ SIZE_CHECK(i < size());
		//~ return m_datasetSize-m_validationFoldSizes[i];
	//~ }
	//~ /// \brief Returns the overall number of elements in the i-th valdiation fold
	//~ std::size_t numberOfValidationElements(std::size_t i) const {
		//~ SIZE_CHECK(i < size());
		//~ return m_validationFoldSizes[i];
	//~ }

	/// \brief Returns the dataset underying the folds
	DatasetType const& dataset()const{
		return m_dataset;
	}
	
	/// \brief Returns the dataset underying the folds
	DatasetType& dataset(){
		return m_dataset;
	}

private:
	DatasetType m_dataset;
	std::vector<IndexSet> m_validationFolds;
	std::size_t m_datasetSize;
	std::vector<std::size_t> m_validationFoldSizes;
};


/// auxiliary typedef for createCVSameSizeBalanced and createCVFullyIndexed, stores location index in the first and partition index in the second
typedef std::pair< std::vector<std::size_t> , std::vector<std::size_t> > RecreationIndices;

namespace detail {

///\brief Version of createCVSameSizeBalanced which works regardless of the label type
///
/// Instead of a class label to interpret, this class uses a membership vector for every
/// class which members[k][i] returns the positon of the i-th member of class k in the set.
template<class I, class L>
CVFolds<LabeledData<I,L> > createCVSameSizeBalanced(
	LabeledData<I,L> &set,
	std::size_t numberOfPartitions,
	std::vector< std::vector<std::size_t> > members,
	std::size_t batchSize,
	RecreationIndices * cv_indices = NULL //if not NULL: the first vector stores location information, and
					  // the second the partition information. The i-th value of the
					  // first vector shows what the original position of the now i-th
					  // sample was. The i-th value of the second vector shows what
					  // partition that sample now belongs to.
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

	//initialize the list of position indices which can later be used to re-create the fold (via createCV(Fully)Indexed)
	if ( cv_indices != NULL ) {
		cv_indices->first.clear();
		cv_indices->first.resize( numInputs );
		cv_indices->second.clear();
		cv_indices->second.resize( numInputs );
	}

	size_t j = 0; //for recreation indices
	for (std::size_t c = 0; c != numClasses; c++) {
		for (std::size_t i = 0; i != members[c].size(); i++) {
			std::size_t oldPos = members[c][i];
			std::size_t batchNumber = validationSetStart[fold];

			batchElements[fold].push_back(oldPos);

			if ( cv_indices != NULL ) {
				cv_indices->first[ j ] = oldPos; //store the position in which the (now) i-th sample previously resided
				cv_indices->second[ j ] = fold; //store the partition to which the (now) i-th sample gets assigned
				// old: //(*cv_indices)[ oldPos ] = fold; //store in vector to recreate partition if desired
			}

			//if all elements for the current batch are found, create it
			if (batchElements[fold].size() == batchSizes[batchNumber]) {
				newSet.batch(validationSetStart[fold]) = subBatch(setView,batchElements[fold]);
				batchElements[fold].clear();
				++validationSetStart[fold];
			}

			fold = (fold+1) % numberOfPartitions;

			j++;
		}
	}
	SHARK_ASSERT( j == numInputs );

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
//! \param cv_indices if not NULL [default]: for each element, store the fold it is assigned to; this can be used to later/externally recreate the fold via createCVIndexed
template<class I>
CVFolds<LabeledData<I,unsigned int> > createCVSameSizeBalanced (
	LabeledData<I,unsigned int> &set,
	std::size_t numberOfPartitions,
	std::size_t batchSize=Data<I>::DefaultBatchSize,
	RecreationIndices * cv_indices = NULL //if not NULL: for each element, store the fold it is assigned to; this can be used to later/externally recreate the fold via createCVIndexed
){
	DataView<LabeledData<I,unsigned int> > setView(set);
	std::size_t numInputs = setView.size();
	std::size_t numClasses = numberOfClasses(set);


	//find members of each class
	std::vector< std::vector<std::size_t> > members(numClasses);
	for (std::size_t i = 0; i != numInputs; i++) {
		members[setView[i].label].push_back(i);
	}
	return detail::createCVSameSizeBalanced(set, numberOfPartitions, members, batchSize, cv_indices);

}

//! \brief Create a partition for cross validation without changing the dataset
//!
//! This method behaves similar to createCVIID
//! with the difference that batches are not reordered. Thus the batches
//! are only rearranged randomly in folds, but the dataset itself is not changed.
//!
//! \param numberOfPartitions  number of partitions to create
//! \param set the input data from which to draw the partitions
template<class I, class L>
CVFolds<LabeledData<I,L> > createCVBatch (
	LabeledData<I,L> const& set,
	std::size_t numberOfPartitions
){
	std::vector<std::size_t> indizes(set.numberOfBatches());
	for(std::size_t i= 0; i != set.numberOfBatches(); ++i)
		indizes[i] = i;
	DiscreteUniform<Rng::rng_type> uni(Rng::globalRng);
	shark::shuffle(indizes.begin(),indizes.end(), uni);
	
	typedef typename LabeledData<I,L>::IndexSet IndexSet;
	
	std::vector<IndexSet> folds;
	std::size_t partitionSize = set.numberOfBatches()/numberOfPartitions;
	std::size_t remainder = set.numberOfBatches() - partitionSize*numberOfPartitions;
	std::vector<std::size_t>::iterator pos = indizes.begin();
	for(std::size_t i = 0; i!= numberOfPartitions; ++i){
		std::size_t size = partitionSize;
		if(remainder> 0){
			++size;
			--remainder;
		}
		folds.push_back(IndexSet(pos,pos+size));
		pos+=size;
	}
	return CVFolds<LabeledData<I,L> >(set,folds);
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
	LabeledData<I,L> &set,
    std::size_t numberOfPartitions,
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
	DataView<LabeledData<I,L> > setView(set); //fast access to single elements of the original set
	std::vector<std::size_t> validationSetStart = partitionStart; //current index for the batch of every partition
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



//! \brief Create a partition for cross validation from indices for both ordering and partitioning.
//!
//! Create a partition from indices. There is one index vector assigning an order
//! to the samples, and another one assigning each sample to a validation partition.
//! That is, given a dataset set, and at the i-th processing step, this function puts
//! the order_indices[i]-th sample into the partition_indices[i]-th partition. The
//! order_indices part of the above procedure matters if both an inner and
//! outer partition are to be recreated: for the inner partition to be recreated, too,
//! the outer partition must be recreated in the same order, not just partitioned into
//! the same splits.
//!
//! \param set                  partitions will be subsets of this set
//! \param numberOfPartitions   number of partitions to create
//! \param indices              stores location index in the first and partition index in the second vector
//! \param batchSize            maximum batch size
template<class I,class L>
CVFolds<LabeledData<I,L> > createCVFullyIndexed(
	LabeledData<I,L> &set,
	std::size_t numberOfPartitions,
	RecreationIndices indices,
	std::size_t batchSize=Data<I>::DefaultBatchSize
) {
	std::size_t numInputs = set.numberOfElements();
	SIZE_CHECK(indices.first.size() == numInputs);
	SIZE_CHECK(indices.second.size() == numInputs);
	SIZE_CHECK(numberOfPartitions == *std::max_element(indices.second.begin(),indices.second.end())+1);

	//calculate the size of validation partitions
	std::vector<std::size_t> validationSize(numberOfPartitions,0);
	for (std::size_t input = 0; input != numInputs; input++) {
		validationSize[indices.second[input]]++;
	}

	//calculate the size of batches for every validation part and their total number
	std::vector<std::size_t> partitionStart;
	std::vector<std::size_t> batchSizes;
	std::size_t numBatches = detail::batchPartitioning(validationSize,partitionStart,batchSizes,batchSize);

	//construct a new set with the correct batch format from the old set
	LabeledData<I,L> newSet(numBatches);
	DataView<LabeledData<I,L> > setView(set); //fast access to single elements of the original set
	std::vector<std::size_t> validationSetStart = partitionStart; //current index for the batch of every partition
	std::vector<std::vector<std::size_t> > batchElements(numberOfPartitions);
	for (std::size_t input = 0; input != numInputs; input++) {
		std::size_t partition = indices.second[input]; //the second vector's contents indicate the partition to assign each sample to.
		batchElements[partition].push_back( indices.first[input] ); //the first vector's contents indicate from what original position to get the next sample.

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
