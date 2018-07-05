//===========================================================================
/*!
 * 
 *
 * \brief       Efficient brute force implementation of nearest neighbors.
 * 
 * 
 *
 * \author      O.Krause
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

#ifndef SHARK_ALGORITHMS_NEARESTNEIGHBORS_SIMPLENEARESTNEIGHBORS_H
#define SHARK_ALGORITHMS_NEARESTNEIGHBORS_SIMPLENEARESTNEIGHBORS_H

#include <shark/Algorithms/NearestNeighbors/AbstractNearestNeighbors.h>
#include <shark/Models/Kernels/AbstractMetric.h>
#include <shark/Core/Threading/Algorithms.h>
#include <algorithm>


namespace shark {

///\brief Brute force optimized nearest neighbor implementation
///
///Returns the labels and distances of the k nearest neighbors of a point 
/// The distance is measured using an arbitrary metric
template<class InputType, class LabelType>
class SimpleNearestNeighbors:public AbstractNearestNeighbors<InputType,LabelType>{
private:
	typedef AbstractNearestNeighbors<InputType,LabelType> base_type;
public:
	typedef LabeledData<InputType, LabelType> Dataset;
	typedef AbstractMetric<InputType> Metric;
	typedef typename base_type::DistancePair DistancePair;
	typedef typename Batch<InputType>::type BatchInputType;

	/// \brief Constructor.
	///
	/// \par Construct a "brute force" nearest neighbors search algorithm
	/// from data and a metric. Refer to the AbstractMetric class for details.
	/// The "default" Euclidean metric is realized by providing a pointer to
	/// an object of type LinearKernel<InputType>.
	SimpleNearestNeighbors(Dataset const& dataset, Metric const* metric)
	:m_dataset(dataset), mep_metric(metric){
		this->m_inputShape=dataset.inputShape();
	}

	///\brief Return the k nearest neighbors of the query point.
	std::vector<DistancePair> getNeighbors(BatchInputType const& patterns, std::size_t k)const{
		std::size_t numPatterns = batchSize(patterns);
		std::size_t numBatches = m_dataset.numberOfBatches();
		std::size_t maxThreads = std::min(threading::globalThreadPool().numWorkers(),numBatches);
		//heaps of key value pairs (distance,classlabel). One heap for every pattern and thread.
		//For memory alignment reasons, all heaps are stored in one continuous array
		//the heaps are stored such, that for every pattern the heaps for every thread
		//are forming one memory area. so later we can just merge all heaps of one pattern using make_heap
		//be aware that the values created here allready form a heap since they are all
		//identical maximum distance.
		std::vector<DistancePair> heaps(k*numPatterns*maxThreads,DistancePair(std::numeric_limits<double>::max(),LabelType()));
		//iterate over all batches of the training set in parallel and let
		//every thread do a KNN-Search on it's subset of data
		
		auto updateHeap=[&](std::size_t thread){
			//get the partition of batches for this thread
			std::size_t workSize = (numBatches + maxThreads - 1) / maxThreads;
			std::size_t begin = thread * workSize;
			std::size_t end = begin + std::min(workSize, numBatches - begin);
			//iterate over all batches of this thread
			for(std::size_t b = begin; b != end; ++b){ 
				//evaluate distances between the points of the patterns and the batch
				RealMatrix distances=mep_metric->featureDistanceSqr(patterns,m_dataset.batch(b).input);
				
				//now update the heaps with the distances
				for(std::size_t p = 0; p != numPatterns; ++p){
					std::size_t batchSize = distances.size2();
					
					//get current heap
					std::size_t heap = p*maxThreads + thread;
					auto heapStart=heaps.begin()+heap*k;
					auto heapEnd=heapStart+k;
					auto biggest=heapEnd-1;//position of biggest element
					
					//update heap values using the new distances
					for(std::size_t i = 0; i != batchSize; ++i){
						if(biggest->key >= distances(p,i)){
							//push the smaller neighbor in the heap and replace the biggest one
							biggest->key=distances(p,i);
							biggest->value=getBatchElement(m_dataset.batch(b).label,i);
							std::push_heap(heapStart,heapEnd);
							//pop biggest element, so that 
							//biggest is again the biggest element
							std::pop_heap(heapStart,heapEnd);
						}
					}
				}
			}
		};
		//fire off the threads
		threading::parallelND({maxThreads}, {1}, updateHeap,  threading::globalThreadPool());
		
		
		std::vector<DistancePair> results(k*numPatterns);
		//finally, we merge all threads in one heap which has the inverse ordering
		//and create a class histogram over the smallest k neighbors
		for(std::size_t p = 0; p < numPatterns; ++p){
			//find range of the heaps for all threads
			auto heapStart=heaps.begin()+p*maxThreads*k;
			auto heapEnd=heapStart+maxThreads*k;
			auto neighborEnd=heapEnd-k;
			auto smallest=heapEnd-1;//position of biggest element
			//create one single heap of the range with inverse ordering
			//takes O(maxThreads*k)
			std::make_heap(heapStart,heapEnd,std::greater<DistancePair>());
			
			//create histogram from the neighbors
			for(std::size_t i = 0;heapEnd!=neighborEnd;--heapEnd,--smallest,++i){
				std::pop_heap(heapStart,heapEnd,std::greater<DistancePair>());
				results[i+p*k].key = smallest->key;
				results[i+p*k].value = smallest->value; 
			}
		}
		return results;
	}

	/// \brief Direct access to the underlying data set of nearest neighbor points.
	LabeledData<InputType,LabelType>const& dataset()const {
		return m_dataset;
	}

private:
	Dataset m_dataset;                        ///< data set of nearest neighbor points
	Metric const* mep_metric;                 ///< metric for measuring distances, usually given by a kernel function
};


}
#endif
