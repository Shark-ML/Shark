/*!
 * 
 *
 * \brief       Implements the most recent version of the elitist CMA-ES.
 * 
 *
 * \author      O.Krause, T.Voss
 * \date        2014
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
#include <shark/Algorithms/DirectSearch/ElitistCMA.h>
#include <algorithm>

using namespace shark;

ElitistCMA::ElitistCMA(): m_activeUpdate(true) {
	m_features |= REQUIRES_VALUE;
}

void ElitistCMA::read( InArchive & archive ) {
	archive >> m_individual;
	archive >> m_ancestralFitness;
	archive >> m_activeUpdate;
}

void ElitistCMA::write( OutArchive & archive ) const {
	archive << m_individual;
	archive << m_ancestralFitness;
	archive << m_activeUpdate;
}


void ElitistCMA::init( ObjectiveFunctionType const& function, SearchPointType const& p){
	
	//create and evaluate individual
	m_individual = CMAIndividual<double>(p.size());
	m_individual.chromosome().m_stepSize = std::sqrt(1.0/p.size());
	noalias(m_individual.searchPoint()) = p;
	m_evaluator(function,m_individual);
	
	//fill the history with the current individual
	m_ancestralFitness.resize(5);
	std::fill(m_ancestralFitness.begin(),m_ancestralFitness.end(),m_individual.penalizedFitness());
	
	//starting point is best solution so far
	m_best.point = p;
	m_best.value = m_individual.unpenalizedFitness();
}

void ElitistCMA::step(ObjectiveFunctionType const& function) {
	//create and evaluate offspring
	m_individual.mutate();
	m_evaluator( function, m_individual );
	
	//evaluate success status of individual
	CMAChromosome::IndividualSuccess  success = CMAChromosome::Successful;
	if( m_individual.penalizedFitness() >= m_ancestralFitness.back()){
		success = CMAChromosome::Unsuccessful;
	}
	if(m_activeUpdate && m_individual.penalizedFitness() > m_ancestralFitness.front()){
		success = CMAChromosome::Failure;
	}
	
	//if the new individual is better, keep it and adapt its strategy
	if(success == CMAChromosome::Successful){
		m_individual.updateAsOffspring();
		
		//new best solution found
		noalias(m_best.point) = m_individual.searchPoint();
		m_best.value = m_individual.unpenalizedFitness();
		
		//update ancestory
		m_ancestralFitness.erase(m_ancestralFitness.begin());
		m_ancestralFitness.push_back(m_individual.penalizedFitness());
		
	}else{
		//reset to parent individual
		noalias(m_individual.searchPoint()) = m_best.point;
		//update its strategy
		m_individual.updateAsParent(success);
	}
	
	
}
