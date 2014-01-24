/*!
 * 
 * \file        RBM/analytics.h
 *
 * \brief       -
 *
 * \author      -
 * \date        -
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
#ifndef SHARK_UNSUPERVISED_RBM_ANALYTICS_H
#define SHARK_UNSUPERVISED_RBM_ANALYTICS_H


#include "Impl/analytics.h"

#include <shark/Unsupervised/RBM/RBM.h>
#include <shark/Data/Dataset.h>

namespace shark {
///\brief Calculates the value of the partition function $Z$.
///
///Only useful for small input and theoretical analysis
///
///@param rbm the RBM for which to calculate the function
///@param beta the inverse temperature of the RBM. default is 1
///@return the value of the partition function $Z*e^(-constant)$
template<class RBMType>
double logPartitionFunction(RBMType const& rbm, double beta = 1.0) {
	//choose correct version based on the enumeration tags
	typedef typename RBMType::HiddenType::StateSpace::EnumerationTag HiddenTag;
	typedef typename RBMType::VisibleType::StateSpace::EnumerationTag VisibleTag;
	
	return detail::logPartitionFunction(rbm,VisibleTag(),HiddenTag(),beta);
}


///\brief Estimates the negative log-likelihood of a set of input vectors under the models distribution using the partition function
///
///Only useful for small input and theoretical analysis
///
///@param rbm the Restricted Boltzmann machine for which the negative log likelihood of the data is to be calculated
///@param inputs the input vectors
///@param logPartition the logarithmic value of the partition function of the RBM.
///@param beta the inverse temperature of the RBM. default is 1
///@return the log-likelihood
template<class RBMType>
double negativeLogLikelihoodFromLogPartition(
	RBMType const&rbm, 
	UnlabeledData<RealVector> const& inputs, 
	double logPartition, 
	double beta = 1.0
) {
	double logP=0;
	BOOST_FOREACH(RealMatrix const& batch,inputs.batches()) {
		logP += sum(rbm.energy().logUnnormalizedPropabilityVisible(batch, blas::repeat(beta,batch.size1())));
		logP -= batch.size1()*logPartition;
	}
	return -logP;
}

///\brief Estimates the negative log-likelihood of a set of input vectors under the models distribution.
///
///Only useful for small input and theoretical analysis
///
///@param rbm the Restricted Boltzmann machine for which the negative log likelihood of the data is to be calculated
///@param inputs the input vectors
///@param beta the inverse temperature of the RBM. default is 1
///@return the log-likelihood
template<class RBMType>
double negativeLogLikelihood(
	RBMType const& rbm, 
	UnlabeledData<RealVector> const& inputs, 
	double beta = 1.0
) {
	double const logPartition = logPartitionFunction(rbm,beta);
	return negativeLogLikelihoodFromLogPartition(rbm,inputs,logPartition,beta);
}

enum PartitionEstimationAlgorithm{
	AIS,
	AISMean,
	TwoSidedAISMean,
	AcceptanceRatio,
	AcceptanceRatioMean
};

inline double estimateLogFreeEnergyFromEnergySamples(
	RealMatrix const& energyDiffUp,
	RealMatrix const& energyDiffDown,
	PartitionEstimationAlgorithm algorithm = AIS
){	
	std::size_t chains = energyDiffUp.size1();
	std::size_t samples = energyDiffUp.size2();
	double deltaF = 0;
	switch(algorithm){
	case AIS:
		deltaF = soft_max(-sum_rows(energyDiffUp))-std::log(double(samples));
	break;
	case AISMean:
		for(std::size_t i = chains-1; i != 0; --i){
			deltaF += soft_max(-row(energyDiffUp,i))-std::log(double(samples));
		}
	break;
	case TwoSidedAISMean:
		for(std::size_t i = chains-1; i != 0; --i){
			deltaF += detail::twoSidedAIS(row(energyDiffUp,i),row(energyDiffDown,i-1));
		}
	break;
	case AcceptanceRatioMean:
		for(std::size_t i = chains-1; i != 0; --i){
			deltaF += detail::acceptanceRatio(row(energyDiffUp,i),row(energyDiffDown,i-1));
		}
	break;
	case AcceptanceRatio:
		deltaF = detail::acceptanceRatio(sum_rows(energyDiffUp),sum_rows(energyDiffDown));
	}
	
	return deltaF;
}

template<class RBMType>
double estimateLogFreeEnergy(
	RBMType& rbm, Data<RealVector> const& initDataset, 
	RealVector const& beta, std::size_t samples,
	PartitionEstimationAlgorithm algorithm = AIS,
	float burninInPercentage =0.1
){
	std::size_t chains = beta.size();
	RealMatrix energyDiffUp(chains,samples);
	RealMatrix energyDiffDown(chains,samples);
	detail::sampleEnergies(rbm,initDataset,beta,energyDiffUp,energyDiffDown,burninInPercentage);
	
	return estimateLogFreeEnergyFromEnergySamples(
		energyDiffUp,energyDiffDown,algorithm
	);
}

template<class RBMType>
double annealedImportanceSampling(
	RBMType& rbm,RealVector const& beta, std::size_t samples
){
	std::size_t chains = beta.size();
	RealMatrix energyDiffTempering(chains,samples,0.0);
	detail::sampleEnergiesWithTempering(rbm,beta,energyDiffTempering);
	
	return soft_max(-sum_rows(energyDiffTempering))-std::log(double(samples));
}


}
#endif
