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
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Individual.h>

using namespace shark;

namespace {
/**
* \brief Calculates the expected length of a vector of length n.
* \param [in] n The length of the vector.
*/ 
double chi( unsigned int n ) {
	return( std::sqrt( static_cast<double>( n ) )*(1. - 1./(4.*n) + 1./(21.*n*n)) );
}
		
}

ElitistCMA::ElitistCMA(): m_lambda( 1 ),m_activeUpdate(false) {
	m_features |= REQUIRES_VALUE;
}

void ElitistCMA::read( InArchive & archive ) {
	archive >> m_lambda;
	archive >> m_activeUpdate;

	archive >> m_fitness;
	archive >> m_lastFitness;
	archive >> m_fitnessUpdateFrequency;
	archive >> m_generationCounter;
	
	archive >> m_targetSuccessProbability;
	archive >> m_successProbability;
	archive >> m_successProbabilityThreshold;
	archive >> m_mean;
	archive >> m_evolutionPathC;
	archive >> m_mutationDistribution;
	archive >> m_sigma;
	
	archive >> m_cSuccessProb;
	archive >> m_cC;
	archive >> m_cCov;
	archive >> m_cCovMinus;
	archive >> m_cSigma;
	archive >> m_dSigma;
}

void ElitistCMA::write( OutArchive & archive ) const {
	archive << m_lambda;
	archive << m_activeUpdate;

	archive << m_fitness;
	archive << m_lastFitness;
	archive << m_fitnessUpdateFrequency;
	archive << m_generationCounter;
	
	archive << m_targetSuccessProbability;
	archive << m_successProbability;
	archive << m_successProbabilityThreshold;
	archive << m_mean;
	archive << m_evolutionPathC;
	archive << m_mutationDistribution;
	archive << m_sigma;
	
	archive << m_cSuccessProb;
	archive << m_cC;
	archive << m_cCov;
	archive << m_cCovMinus;
	archive << m_cSigma;
	archive << m_dSigma;
}


void ElitistCMA::init( ObjectiveFunctionType const& function, SearchPointType const& p){
	std::size_t numberOfVariables = p.size();

	m_fitnessUpdateFrequency = 5;
	m_generationCounter = 0;
	m_evolutionPathC = blas::repeat( 0.0,numberOfVariables );
	m_mean = p;
	m_mutationDistribution.resize( numberOfVariables );
	m_sigma = 1.0;
	
	m_dSigma = 1. + numberOfVariables/(2.*m_lambda);

	m_targetSuccessProbability = 1./(5.+::sqrt( m_lambda/4.0 ));
	m_successProbabilityThreshold = 0.44;
	m_cSuccessProb = (m_targetSuccessProbability*m_lambda)/(2+m_targetSuccessProbability*m_lambda);

	m_cC = 2./(numberOfVariables + 2.);
	m_cCov = 2./(sqr(numberOfVariables) + 6);
	m_cCovMinus = 0.4/( std::pow(numberOfVariables, 1.6 )+1. );

	m_fitness = function( m_mean );
	m_lastFitness = m_fitness;
	
	m_best.point = p;
	m_best.value = m_fitness;

}

/**
* \brief Executes one iteration of the algorithm.
*/
void ElitistCMA::step(ObjectiveFunctionType const& function) {
	//create offspring and select the best
	PenalizingEvaluator evaluator;
	Individual<RealVector,double> bestOffspring;
	bestOffspring.penalizedFitness() = std::numeric_limits<double>::max();
	for( unsigned int i = 0; i != m_lambda; ++i ) {
		Individual<RealVector,double> offspring;
		offspring.searchPoint() = m_mean + m_sigma * m_mutationDistribution().first;
		evaluator( function, offspring );
		if(offspring.penalizedFitness() <= bestOffspring.penalizedFitness()){
			bestOffspring = offspring;
		}
	}

	updateStrategyParameters( bestOffspring.searchPoint(), bestOffspring.penalizedFitness());

	//update the solution if a better was sound
	if(bestOffspring.penalizedFitness() < m_best.value){
		m_best.point = bestOffspring.searchPoint();
		m_best.value = bestOffspring.penalizedFitness();
	}
}

void  ElitistCMA::updateCovarianceMatrix( RealVector const& point ) {
	if( m_successProbability < m_successProbabilityThreshold ) {
		m_evolutionPathC = (1-m_cC)*m_evolutionPathC + 
			::sqrt( m_cC*(2-m_cC) ) * 1./m_sigma * (point - m_mean);
		RealMatrix & C = m_mutationDistribution.covarianceMatrix();
		C = (1. - m_cCov) * C + m_cCov * blas::outer_prod( m_evolutionPathC, m_evolutionPathC );
	} else {
		m_evolutionPathC = (1-m_cC)*m_evolutionPathC;		  
		RealMatrix & C = m_mutationDistribution.covarianceMatrix();
		C = (1. - m_cCov+m_cCov*m_cC*(2.-m_cC)) * C + 
			m_cCov *blas::outer_prod( m_evolutionPathC, m_evolutionPathC );		    
	}
	m_mutationDistribution.update();
}

void ElitistCMA::activeCovarianceMatrixUpdate( RealVector const& point ) {
	RealVector z = (point - m_mean)/m_sigma;
	if( 1 - norm_sqr( z ) * m_cCovMinus/(1+m_cCovMinus) <= 0 )
		return;
	RealMatrix & C = m_mutationDistribution.covarianceMatrix();
	C = (1. - m_cCovMinus) * C - m_cCovMinus * blas::outer_prod( z, z );
	m_mutationDistribution.update();
}

void ElitistCMA::updateStrategyParameters( RealVector const& point, double fitness ) {
	bool successful = fitness < m_fitness;//better than the best known?
	bool unsuccessful = fitness > m_lastFitness;//worse than the the last version a few steps back
	m_successProbability = (1.-m_cSuccessProb)*m_successProbability + m_cSuccessProb * (successful ? 1 : 0);

	// Covariance Matrix update
	if( successful )
		updateCovarianceMatrix( point );
	else if( unsuccessful  && m_activeUpdate )
		activeCovarianceMatrixUpdate( point );

	//update the step size based on the current success probability
	m_sigma *= ::exp( 1./m_dSigma * (m_successProbability - m_targetSuccessProbability)/(1-m_targetSuccessProbability) );

	if( successful ) {
		m_mean = point;
		m_fitness = fitness;
	}

	if( m_generationCounter % m_fitnessUpdateFrequency == 0 )
		m_lastFitness = m_fitness;

	m_generationCounter++;

}
