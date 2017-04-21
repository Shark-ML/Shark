/*!
 *
 * \brief       Implements the Cross Entropy Algorithm.
 * 
 * Christophe Thiery, Bruno Scherrer. Improvements on Learning Tetris with Cross Entropy.
 * International Computer Games Association Journal, ICGA, 2009, 32. <inria-00418930>
 * 
 *
 * \author      Jens Holm, Mathias Petræus and Mark Wulff
 * \date        January 2016
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
 #define SHARK_COMPILE_DLL

#include <shark/Core/Exception.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/ElitistSelection.h>
#include <shark/Algorithms/DirectSearch/CrossEntropyMethod.h>

using namespace shark;

/**
* \brief Suggest a population size of 100.
*/
unsigned CrossEntropyMethod::suggestPopulationSize(  ) {
	/*
	 * Most papers suggests a population size of 100, thus
	 * simply choose 100.
	 */
	return 100;
}

/**
* \brief Calculates Selection Size for the supplied Population Size.
*/
unsigned int CrossEntropyMethod::suggestSelectionSize( unsigned int populationSize ) {
	/* 
	 * Most papers says 10% of the population size is used for
	 * the new generation, thus, just take 10% of the population size.
	 */
	return (unsigned int) (populationSize / 10.0);
}

CrossEntropyMethod::CrossEntropyMethod()
: m_variance( 0 )
, m_counter( 0 )
, m_noise (boost::shared_ptr<INoiseType> (new ConstantNoise(0.0)))
{
	m_features |= REQUIRES_VALUE;
}

void CrossEntropyMethod::read( InArchive & archive ) {
	archive >> m_numberOfVariables;
	archive >> m_selectionSize;
	archive >> m_populationSize;

	archive >> m_variance;

	archive >> m_mean;

	archive >> m_counter;
}

void CrossEntropyMethod::write( OutArchive & archive ) const {
	archive << m_numberOfVariables;
	archive << m_selectionSize;
	archive << m_populationSize;

	archive << m_variance;

	archive << m_mean;

	archive << m_counter;
}


void CrossEntropyMethod::init( ObjectiveFunctionType const& function, SearchPointType const& p) {
	
	unsigned int populationSize = CrossEntropyMethod::suggestPopulationSize( );
	unsigned int selectionSize = CrossEntropyMethod::suggestSelectionSize( populationSize );
	
	// Most papers set the variance to 100 by default.
	RealVector initialVariance(p.size(),100);
	init( function,
		p,
		populationSize,
		selectionSize,
		initialVariance
	);
}

/**
* \brief Initializes the algorithm for the supplied objective function.
*/
void CrossEntropyMethod::init(
	ObjectiveFunctionType const& function, 
	SearchPointType const& initialSearchPoint,
	unsigned int populationSize,
	unsigned int selectionSize,
	RealVector initialVariance
) {
	checkFeatures(function);
	
	m_numberOfVariables = function.numberOfVariables();
	m_populationSize = populationSize;
	m_selectionSize = selectionSize;
	m_variance = initialVariance;

	m_mean.resize( m_numberOfVariables );
	m_mean.clear();

	m_mean = initialSearchPoint;
	m_best.point = initialSearchPoint;
	m_best.value = function(initialSearchPoint);
	m_counter = 0;

}

/**
* \brief Updates the strategy parameters based on the supplied parent population.
*/
void CrossEntropyMethod::updateStrategyParameters( const std::vector<Individual<RealVector, double> > & parents ) {

	/* Calculate the centroid of the parents */
	RealVector m(m_numberOfVariables);
	for (std::size_t i = 0; i < m_numberOfVariables; i++)
	{
		m(i) = 0;
		for (std::size_t j = 0; j < parents.size(); j++)
		{
			m(i) += parents[j].searchPoint()(i);
		}
		m(i) /= double(parents.size());
	}


	// mean update
	m_mean = m;

	// Variance update
	size_t nParents = parents.size();
	double normalizationFactor = 1.0 / double(nParents);

	for (std::size_t j = 0; j < m_numberOfVariables; j++) {
		double innerSum = 0.0;
		for (std::size_t i = 0; i < parents.size(); i++) {
			double diff = parents[i].searchPoint()(j) - m(j);
			innerSum += diff * diff;
		}
		innerSum *= normalizationFactor;

		// Apply noise
		m_variance(j) = innerSum + m_noise->noiseValue(m_counter);
	}

}

/**
* \brief Executes one iteration of the algorithm.
*/
void CrossEntropyMethod::step(ObjectiveFunctionType const& function){
	
	std::vector< IndividualType > offspring( m_populationSize );

	PenalizingEvaluator penalizingEvaluator;
	for( std::size_t i = 0; i < offspring.size(); i++ ) {
		RealVector sample(m_numberOfVariables);
		for (std::size_t j = 0; j < m_numberOfVariables; j++){
			sample(j) = random::gauss(random::globalRng,m_mean(j), m_variance(j)); // N (0, 100)
		}
		offspring[i].searchPoint() = sample;
	}

	penalizingEvaluator( function, offspring.begin(), offspring.end() );

	// Selection
	std::vector< Individual<RealVector, double> > parents( m_selectionSize );
	ElitistSelection< IndividualType::FitnessOrdering > selection;
	selection(offspring.begin(),offspring.end(),parents.begin(), parents.end());
	// Strategy parameter update
	m_counter++; // increase generation counter
	updateStrategyParameters( parents );

	m_best.point= parents[ 0 ].searchPoint();
	m_best.value= parents[ 0 ].unpenalizedFitness();
}

