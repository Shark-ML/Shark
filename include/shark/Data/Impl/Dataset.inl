/*!
 *
 * \brief Internal functionality and implementation of the Data class
 *
 *  \author O. Krause
 *  \date 2012
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
#ifndef SHARK_DATA_IMPL_DATASET_INL
#define SHARK_DATA_IMPL_DATASET_INL

#include <shark/Data/BatchInterface.h>
#include <shark/Core/utility/CanBeCalled.h>

#include <boost/serialization/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#include <algorithm>
#include <memory>

namespace shark {
namespace detail{

/**
 * \ingroup shark_detail
 *
 * @{
 */
	
	
inline std::size_t numberOfBatches(std::size_t numElements, std::size_t maximumBatchSize){
	if(maximumBatchSize == 0)
		maximumBatchSize =numElements;
	std::size_t batches = numElements / maximumBatchSize;
	if(numElements-batches*maximumBatchSize > 0)
		++batches;
	return batches;
}

///\brief Computes a partitioning of a st of elements in batches.
///	
/// Given a number of elements and the maximum size of a batch, 
/// computes the optimal number of batches and returns the size of every batch such that
/// all batches have as equal a size as possible.
///
/// \param numElements number of elements to partition
/// \param maximumBatchSize the maximum size of a batch
/// \return a vector with th size of every batch
inline std::vector<std::size_t> optimalBatchSizes(std::size_t numElements, std::size_t maximumBatchSize){
	std::size_t batches = numberOfBatches(numElements, maximumBatchSize);
	std::vector<std::size_t> batchSizes(batches);
	std::size_t optimalBatchSize=numElements/batches;
	std::size_t remainder = numElements-batches*optimalBatchSize;

	for(std::size_t i = 0; i != batches; ++i){
		std::size_t size = optimalBatchSize + (i<remainder);
		batchSizes[i] = size;
	}
	return batchSizes;
}

///\brief Given the sizes of the partition sets and the maximum batch size, computes a good partitioning.
///
/// \param[in]   partitionSizes    sizes of the partitions (number of elements)
/// \param[out]  partitionStart    indices of the starting batches of the partition
/// \param[out]  batchSizes        sizes of the batches
/// \param[in]   maximumBatchSize  maximal batch size
/// \return                        the total number of batches
inline std::size_t batchPartitioning(
	std::vector<std::size_t> const& partitionSizes,
	std::vector<std::size_t>& partitionStart,
	std::vector<std::size_t>& batchSizes,
	std::size_t maximumBatchSize
){
	std::size_t sumOfBatches = 0;
	std::size_t numberOfPartitions=partitionSizes.size();
	for (std::size_t i = 0; i != numberOfPartitions; i++){
		partitionStart.push_back(sumOfBatches);
		std::vector<std::size_t> batchSizesOfPartition = optimalBatchSizes(partitionSizes[i],maximumBatchSize);
		batchSizes.insert(batchSizes.end(),batchSizesOfPartition.begin(),batchSizesOfPartition.end());
		sumOfBatches+=batchSizesOfPartition.size();
	}
	return sumOfBatches;
}

/// compute the complement of the indices with respect to the set [0,...n[
template<class T,class T2>
void complement(
	T const& set,
	std::size_t n,
	T2& comp)
{
	std::vector<std::size_t> parentSet(n);
	for(std::size_t i = 0; i != n; ++i){
		parentSet[i]=i;
	}
	std::vector<std::size_t> setCopy(set.begin(),set.end());
	std::sort(setCopy.begin(),setCopy.end());

	std::vector<std::size_t> resultSet(parentSet.size());
	std::vector<std::size_t>::iterator pos = std::set_difference(
		parentSet.begin(),parentSet.end(),
		setCopy.begin(),setCopy.end(),
		resultSet.begin()
	);
	comp.resize(std::distance(resultSet.begin(),pos));
	std::copy(resultSet.begin(),pos,comp.begin());
}

/// \brief For Data<T> and functor F calculates the result of the resulting elements F(T).
template<class Functor, class T>
struct TransformedDataElement{
private:
	template<class B>
	struct TransformedDataElementTypeFromBatch{
		typedef typename batch_to_element<
			typename std::result_of<Functor&&(B)>::type 
		>::type type;
	};
public:
	typedef typename std::conditional<
		!CanBeCalled<Functor,typename Batch<T>::type>::value,
		std::result_of<Functor&&(T) >,
		TransformedDataElementTypeFromBatch<
			typename Batch<T>::type 
		>
	>::type::type type;
};
/** @*/
}
}

#endif
