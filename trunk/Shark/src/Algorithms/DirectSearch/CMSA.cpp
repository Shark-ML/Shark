/*!
 * 
 *
 * \brief       Implements the CMSA.
 * 
 * The algorithm is described in
 * 
 * H. G. Beyer, B. Sendhoff (2008). 
 * Covariance Matrix Adaptation Revisited – The CMSA Evolution Strategy –
 * In Proceedings of the Tenth International Conference on Parallel Problem Solving from Nature
 * (PPSN X), pp. 123-132, LNCS, Springer-Verlag
 * 
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
#include <shark/Algorithms/DirectSearch/CMSA.h>

#include <shark/Core/Exception.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/GlobalIntermediateRecombination.h>

#include <boost/lambda/lambda.hpp>
#include <boost/math/special_functions.hpp>

using namespace shark;


namespace{
	struct FitnessComparator {
		template<typename Individual>
		bool operator()( const Individual & a, const Individual & b ) const {
			return a.fitness( shark::tag::PenalizedFitness() )[0] < b.fitness( shark::tag::PenalizedFitness() )[0];
		}
	};

	struct IdentityExtractor {
		template<typename T>
		const T & operator()( const T & t ) const {
			return t;
		}
	};

	struct PointExtractor {
		template<typename T>
		const RealVector & operator()( const T & t ) const {
			return *t;
		}
	};

}

void CMSA::init( ObjectiveFunctionType const& function, SearchPointType const& p) {
	m_numberOfVariables = p.size();

	m_lambda = 4 * m_numberOfVariables;
	m_mu = m_lambda / 4;

	m_chromosome.m_mean = p;
	m_chromosome.m_mutationDistribution.resize( m_numberOfVariables );

	m_chromosome.m_sigma = 1.0;
	m_chromosome.m_cSigma = 1./::sqrt( 2. * m_numberOfVariables );
	m_chromosome.m_cC = 1. + (m_numberOfVariables*(m_numberOfVariables + 1.))/(2.*m_mu);

	std::vector< RealVector > parents( m_mu, p );
	m_chromosome.m_mean = cog( parents, IdentityExtractor() );
}


void CMSA::step(ObjectiveFunctionType const& function){
	std::vector< CMSA::Individual > offspring( m_lambda );

	shark::soo::PenalizingEvaluator penalizingEvaluator;
	for( unsigned int i = 0; i < offspring.size(); i++ ) {		    
		MultiVariateNormalDistribution::ResultType sample = m_chromosome.m_mutationDistribution();
		offspring[i].get<0>().m_sigma = m_chromosome.m_sigma * ::exp( m_chromosome.m_cSigma * Rng::gauss( 0, 1 ) );
		offspring[i].get<0>().m_step = sample.first;
		*offspring[i] = m_chromosome.m_mean + offspring[i].get<0>().m_sigma * sample.first;
		boost::tuple< ObjectiveFunctionType::ResultType, ObjectiveFunctionType::ResultType > evalResult;
		evalResult = penalizingEvaluator( function, *offspring[i] );

		offspring[i].fitness( shark::tag::UnpenalizedFitness() )[0] = boost::get< shark::soo::PenalizingEvaluator::UNPENALIZED_RESULT >( evalResult );
		offspring[i].fitness( shark::tag::PenalizedFitness() )[0] = boost::get< shark::soo::PenalizingEvaluator::PENALIZED_RESULT >( evalResult );		    

	}

	// Selection
	std::sort( offspring.begin(), offspring.end(), FitnessComparator() );
	std::vector< CMSA::Individual > parentsNew( offspring.begin(), offspring.begin() + m_mu );

	// Strategy parameter update
	updateStrategyParameters( parentsNew );

	m_best.point = *parentsNew.front();
	m_best.value = parentsNew.front().fitness(shark::tag::UnpenalizedFitness())[0];
}

void CMSA::updateStrategyParameters( const std::vector< CMSA::Individual > & offspringNew ) {
	RealVector xPrimeNew = cog( offspringNew, PointExtractor() );
	// Covariance Matrix Update
	RealMatrix Znew( m_numberOfVariables, m_numberOfVariables,0.0 );
	RealMatrix C( m_chromosome.m_mutationDistribution.covarianceMatrix() );
	// Rank-mu-Update
	for( unsigned int i = 0; i < m_mu; i++ ) {
		Znew += 1./m_mu * blas::outer_prod( 
			offspringNew[i].get<0>().m_step,
			offspringNew[i].get<0>().m_step
			/*
			(*offspringNew[i]/offspringNew[i].get<0>().m_sigma - m_chromosome.m_mean),
									(*offspringNew[i]/offspringNew[i].get<0>().m_sigma - m_chromosome.m_mean)*/
			
			);
	}
	C = (1. - 1./m_chromosome.m_cC) * C + 1./m_chromosome.m_cC * Znew;
	m_chromosome.m_mutationDistribution.setCovarianceMatrix( C );

	// Step size update
	double sigmaNew = 0.;
	//double sigma = 0.;
	for( unsigned int i = 0; i < m_mu; i++ ) {
		sigmaNew += 1./m_mu * offspringNew[i].get<0>().m_sigma;
	}
	m_chromosome.m_sigma = sigmaNew;
	m_chromosome.m_mean = xPrimeNew;
}

void CMSA::read( InArchive & archive ) {
	archive >> m_numberOfVariables;
	archive >> m_mu;
	archive >> m_lambda;
	archive >> m_chromosome;
}
void CMSA::write( OutArchive & archive ) const {
	archive << m_numberOfVariables;
	archive << m_mu;
	archive << m_lambda;
	archive << m_chromosome;
}
