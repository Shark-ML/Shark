/*!
 * 
 *
 * \brief       TypedIndividual

 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_CMA_INDIVIDUAL_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_CMA_INDIVIDUAL_H

#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/CMA/Chromosome.h>

#include <shark/LinAlg/Base.h>
#include <vector>

namespace shark {

template<class FitnessType>
class CMAIndividual : public Individual<RealVector,FitnessType, CMAChromosome>{
public:
	using Individual<RealVector,FitnessType, CMAChromosome>::chromosome;
	using Individual<RealVector,FitnessType, CMAChromosome>::searchPoint;
	/**
	 * \brief Default constructor that initializes the individual's attributes to default values.
	 */
	CMAIndividual(){}
	CMAIndividual(
		std::size_t searchSpaceDimension,
		double successThreshold = 0.44,
		double initialStepSize = 1.0
	){
		chromosome() = CMAChromosome(searchSpaceDimension, successThreshold, initialStepSize);
		searchPoint().resize(searchSpaceDimension);
	}
	
	void updateAsParent(CMAChromosome::IndividualSuccess offspringSuccess){
		chromosome().updateAsParent(offspringSuccess);
	}
	void updateAsOffspring(){
		chromosome().updateAsOffspring();
	}
	void mutate(){
		MultiVariateNormalDistribution::result_type sample = chromosome().m_mutationDistribution();
		chromosome().m_lastStep = sample.first;
		chromosome().m_lastZ = sample.second;
		searchPoint() += chromosome().m_stepSize * sample.first;
	}
	
	double& noSuccessfulOffspring(){
		return chromosome().m_noSuccessfulOffspring;
	}
	
	double noSuccessfulOffspring()const{
		return chromosome().m_noSuccessfulOffspring;
	}
};

}
#endif
