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

#ifndef SHARK_ALGORITHMS_NEARESTNEIGHBORS_SIMPLENEARESTNEIGHBORS_H
#define SHARK_ALGORITHMS_NEARESTNEIGHBORS_SIMPLENEARESTNEIGHBORS_H

#include <shark/Algorithms/NearestNeighbors/AbstractNearestNeighbors.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Core/OpenMP.h>
#include <algorithm>
namespace shark {


///\brief Brute force optimized nearest neighbor implementation
///
///Returns the labels and distances of the k nearest neighbors of a point 
template<class InputType, class LabelType>
class SimpleNearestNeighbors:public AbstractNearestNeighbors<InputType,LabelType>{
private:
	typedef AbstractNearestNeighbors<InputType,LabelType> base_type;
public:
	typedef LabeledData<InputType, LabelType> Dataset;
	typedef AbstractKernelFunction<InputType> Kernel;
	typedef typename base_type::DistancePair DistancePair;
	typedef typename Batch<InputType>::type BatchInputType;

	SimpleNearestNeighbors(Dataset const& dataset, Kernel const* kernel)
	:m_dataset(dataset), mep_kernel(kernel){}
		
	///\brief returns the k nearest neighbors of the point
	std::vector<DistancePair> getNeighbors(BatchInputType const& patterns, std::size_t k)const{
		std::size_t numPatterns = size(patterns);
		std::size_t maxThreads = std::min(SHARK_NUM_THREADS,m_dataset.numberOfBatches());
		//heaps of key value pairs (distance,classlabel). One heap for every pattern and thread.
		//For memory alignment reasons, all heaps are stored in one continuous array
		//the heaps are stored such, that for every pattern the heaps for every thread
		//are forming one memory area. so later we can just merge all 4 heaps using make_heap
		//be aware that the values created here allready form a heap since they are all
		//identical maximum distance.
		std::vector<DistancePair> heaps(k*numPatterns*maxThreads,DistancePair(std::numeric_limits<double>::max(),LabelType()));
		typedef typename std::vector<DistancePair>::iterator iterator;
		//iterate over all batches of the training set in parallel and let
		//every thread do a KNN-Search on it's subset of data
		SHARK_PARALLEL_FOR(int b = 0; b < (int)m_dataset.numberOfBatches(); ++b){
			//evaluate distances between the points of the patterns and the batch
			RealMatrix distances=mep_kernel->featureDistanceSqr(patterns,m_dataset.batch(b).input);
			
			//now update the heaps with the distances
			for(std::size_t p = 0; p != numPatterns; ++p){
				std::size_t batchSize = distances.size2();
				
				//get current heap
				std::size_t heap = p*maxThreads+SHARK_THREAD_NUM;
				iterator heapStart=heaps.begin()+heap*k;
				iterator heapEnd=heapStart+k;
				iterator biggest=heapEnd-1;//position of biggest element
				
				//update heap values using the new distances
				for(std::size_t i = 0; i != batchSize; ++i){
					if(biggest->key >= distances(p,i)){
						//push the smaller neighbor in the heap and replace the biggest one
						biggest->key=distances(p,i);
						biggest->value=get(m_dataset.batch(b).label,i);
						std::push_heap(heapStart,heapEnd);
						//pop biggest element, so that 
						//biggest is again the biggest element
						std::pop_heap(heapStart,heapEnd);
					}
				}
			}
		}
		std::vector<DistancePair> results(k*numPatterns);
		//finally, we merge all threads in one heap which has the inverse ordering
		//and create a class histogram over the smallest k neighbors
		//std::cout<<"info "<<numPatterns<<" "<<maxThreads<<" "<<k<<std::endl;
		SHARK_PARALLEL_FOR(int p = 0; p < (int)numPatterns; ++p){
			//find range of the heaps for all threads
			iterator heapStart=heaps.begin()+p*maxThreads*k;
			iterator heapEnd=heapStart+maxThreads*k;
			iterator neighborEnd=heapEnd-k;
			iterator smallest=heapEnd-1;//position of biggest element
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
	LabeledData<InputType,LabelType>const& dataset()const {
		return m_dataset;
	}
	
	
private:
	Dataset m_dataset;
	Kernel const* mep_kernel; 
};


}
#endif
