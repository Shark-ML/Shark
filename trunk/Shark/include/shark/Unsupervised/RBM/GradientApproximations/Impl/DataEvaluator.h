/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_UNSUPERVISED_RBM_IMPL_DATAEVALUATOR_H
#define SHARK_UNSUPERVISED_RBM_IMPL_DATAEVALUATOR_H

#include <shark/Unsupervised/RBM/Sampling/GibbsOperator.h>
#include <algorithm>
namespace shark{
namespace detail{
///\brief function used by gradient approximators to calculate the gradient of the data
template<class VectorType,class RBM>
RealVector evaluateData(Data<VectorType> const& data, RBM& rbm, std::size_t batchesForTraining = 0 ){
	typedef GibbsOperator<RBM> Operator;
	Operator gibbs(&rbm);
	
	if( batchesForTraining == 0){
		batchesForTraining =  data.numberOfBatches();
	}
	
	std::size_t elements = 0;
	//get the batches for this iteration
	std::vector<std::size_t> batchIds(data.numberOfBatches());
	{
		for(std::size_t i = 0; i != data.numberOfBatches(); ++i){
			batchIds[i] = i;
		}
		DiscreteUniform<typename RBM::RngType> uni(rbm.rng(),0,1);
		std::random_shuffle(batchIds.begin(),batchIds.end(),uni);
		for(std::size_t i = 0; i != batchesForTraining; ++i){
			elements += data.batch(batchIds[i]).size1();
		}
	}
	
	RealVector derivative(rbm.numberOfParameters(),0);
	
	std::size_t threads = std::min<std::size_t>(batchesForTraining,SHARK_NUM_THREADS);
	std::size_t numBatches = batchesForTraining/threads;
	
	SHARK_PARALLEL_FOR(int t = 0; t < (int)threads; ++t){
		AverageEnergyGradient<RBM> empiricalAverage(&rbm);
		
		std::size_t threadElements = 0;
		
		std::size_t batchStart = t*numBatches;
		std::size_t batchEnd = (t== (int)threads-1)? batchesForTraining : batchStart+numBatches;
		for(std::size_t i = batchStart; i != batchEnd; ++i){
			RealMatrix const& batch = data.batch(batchIds[i]);
			threadElements += batch.size1();
			
			//create the batches for evaluation
			typename Operator::HiddenSampleBatch hiddenBatch(batch.size1(),rbm.numberOfHN());
			typename Operator::VisibleSampleBatch visibleBatch(batch.size1(),rbm.numberOfVN());
			
			visibleBatch.state = batch;
			gibbs.precomputeHidden(hiddenBatch,visibleBatch,blas::repeat(1.0,batch.size1()));
			empiricalAverage.addVH(hiddenBatch,visibleBatch);
		}
		SHARK_CRITICAL_REGION{
			double weight = threadElements/double(elements);
			noalias(derivative) += weight* empiricalAverage.result();
		}
	}
	return derivative;
}

}
}

#endif
