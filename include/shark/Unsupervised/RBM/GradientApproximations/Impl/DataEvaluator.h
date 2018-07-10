/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_UNSUPERVISED_RBM_IMPL_DATAEVALUATOR_H
#define SHARK_UNSUPERVISED_RBM_IMPL_DATAEVALUATOR_H

#include <shark/Unsupervised/RBM/Sampling/GibbsOperator.h>
#include <algorithm>
#include <shark/Core/Threading/Algorithms.h>
namespace shark{
namespace detail{
///\brief function used by gradient approximators to calculate the gradient of the data
template<class VectorType,class RBM>
RealVector evaluateData(Data<VectorType> const& data, RBM& rbm, std::size_t batchesForTraining = 0 ){
	//get the batches for this iteration
	if( batchesForTraining == 0){
		batchesForTraining =  data.numberOfBatches();
	}
	
	std::size_t elements = 0;
	
	std::vector<std::size_t> batchIds(data.numberOfBatches());
	{
		for(std::size_t i = 0; i != data.numberOfBatches(); ++i){
			batchIds[i] = i;
		}
		std::shuffle(batchIds.begin(),batchIds.end(), random::globalRng());
		batchIds.erase(batchIds.begin() + batchesForTraining, batchIds.end());
		for(std::size_t i = 0; i != batchesForTraining; ++i){
			elements += data.batch(batchIds[i]).size1();
		}
	}
	
	//maps the ith batch to its gradient
	auto map = [&rbm, &data, elements](std::size_t i){
		typedef GibbsOperator<RBM> Operator;
		Operator gibbs(&rbm);
		typename RBM::GradientType empiricalAverage(&rbm);
		RealMatrix const& batch = data.batch(i);
		
		//create the batches for evaluation
		typename Operator::HiddenSample hiddenBatch(batch.size1(),rbm.numberOfHN());
		typename Operator::VisibleSample visibleBatch(batch.size1(),rbm.numberOfVN());
		
		visibleBatch.state = batch;
		gibbs.precomputeHidden(hiddenBatch,visibleBatch,blas::repeat(1.0,batch.size1()));
		empiricalAverage.addVH(hiddenBatch,visibleBatch);
		
		double weight = batch.size1()/double(elements);
		return RealVector(weight * empiricalAverage.result());
	};
	RealVector derivative(rbm.numberOfParameters(),0);
	return threading::mapAccumulate( batchIds, std::move(derivative), map, threading::globalThreadPool());
}

}
}

#endif
