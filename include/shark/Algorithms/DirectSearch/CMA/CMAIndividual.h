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
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <https://shark-ml.github.io/Shark/>
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
	CMAIndividual():m_parent(0){}
	CMAIndividual(
		std::size_t searchSpaceDimension,
		double successThreshold = 0.44,
		double initialStepSize = 1.0
	):m_parent(0){
		chromosome() = CMAChromosome(searchSpaceDimension, successThreshold, initialStepSize);
		searchPoint().resize(searchSpaceDimension);
	}
	
	void updateAsParent(CMAChromosome::IndividualSuccess offspringSuccess){
		chromosome().updateAsParent(offspringSuccess);
	}
	void updateAsOffspring(){
		chromosome().updateAsOffspring();
	}
	template<class randomType>
	void mutate(randomType& rng){
		chromosome().m_mutationDistribution.generate(
			rng, chromosome().m_lastStep,chromosome().m_lastZ
		);
		noalias(searchPoint()) += chromosome().m_stepSize * chromosome().m_lastStep;
	}
	
	double& noSuccessfulOffspring(){
		return chromosome().m_noSuccessfulOffspring;
	}
	
	double noSuccessfulOffspring()const{
		return chromosome().m_noSuccessfulOffspring;
	}
	
	std::size_t parent()const{
		return m_parent;
	}
	std::size_t& parent(){
		return m_parent;
	}
private:
	std::size_t m_parent;
};

}
#endif
