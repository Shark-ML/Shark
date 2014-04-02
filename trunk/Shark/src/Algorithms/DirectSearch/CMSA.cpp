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
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>

using namespace shark;


namespace{
	struct FitnessComparator {
		template<typename IndividualType>
		bool operator()( const IndividualType & a, const IndividualType & b ) const {
			return a.penalizedFitness() < b.penalizedFitness();
		}
	};

	struct PointExtractor {
		template<typename T>
		const RealVector & operator()( const T & t ) const {
			return t.searchPoint();
		}
	};

}

void CMSA::init( ObjectiveFunctionType const& function, SearchPointType const& p) {
	m_numberOfVariables = p.size();

	m_lambda = 4 * m_numberOfVariables;
	m_mu = m_lambda / 4;

	m_mean = p;
	m_mutationDistribution.resize( m_numberOfVariables );

	m_sigma = 1.0;
	m_cSigma = 1./::sqrt( 2. * m_numberOfVariables );
	m_cC = 1. + (m_numberOfVariables*(m_numberOfVariables + 1.))/(2.*m_mu);
}


void CMSA::step(ObjectiveFunctionType const& function){
	std::vector< IndividualType > offspring( m_lambda );

	PenalizingEvaluator penalizingEvaluator;
	for( unsigned int i = 0; i < offspring.size(); i++ ) {		    
		MultiVariateNormalDistribution::result_type sample = m_mutationDistribution();
		offspring[i].chromosome().sigma = m_sigma * ::exp( m_cSigma * Rng::gauss( 0, 1 ) );
		offspring[i].chromosome().step = sample.first;
		offspring[i].searchPoint() = m_mean + offspring[i].chromosome().sigma * sample.first;
		penalizingEvaluator( function, offspring[i] );
	}

	// Selection
	std::sort( offspring.begin(), offspring.end(), FitnessComparator() );
	std::vector< IndividualType > parentsNew( offspring.begin(), offspring.begin() + m_mu );

	// Strategy parameter update
	updateStrategyParameters( parentsNew );

	m_best.point = parentsNew.front().searchPoint();
	m_best.value = parentsNew.front().unpenalizedFitness();
}

void CMSA::updateStrategyParameters( const std::vector< CMSA::IndividualType > & offspringNew ) {
	RealVector xPrimeNew = cog( offspringNew, PointExtractor() );
	// Covariance Matrix Update
	RealMatrix Znew( m_numberOfVariables, m_numberOfVariables,0.0 );
	RealMatrix& C = m_mutationDistribution.covarianceMatrix();
	// Rank-mu-Update
	for( unsigned int i = 0; i < m_mu; i++ ) {
		noalias(Znew) += 1./m_mu * blas::outer_prod( 
			offspringNew[i].chromosome().step,
			offspringNew[i].chromosome().step
		);
	}
	noalias(C) = (1. - 1./m_cC) * C + 1./m_cC * Znew;
	m_mutationDistribution.update();

	// Step size update
	double sigmaNew = 0.;
	//double sigma = 0.;
	for( unsigned int i = 0; i < m_mu; i++ ) {
		sigmaNew += 1./m_mu * offspringNew[i].chromosome().sigma;
	}
	m_sigma = sigmaNew;
	m_mean = xPrimeNew;
}

void CMSA::read( InArchive & archive ) {
	archive >> m_numberOfVariables;
	archive >> m_mu;
	archive >> m_lambda;
	
	archive >> m_sigma;
	archive >> m_cC;
	archive >> m_cSigma;

	archive >> m_mean;
	archive >> m_mutationDistribution;
}
void CMSA::write( OutArchive & archive ) const {
	archive << m_numberOfVariables;
	archive << m_mu;
	archive << m_lambda;
	
	archive << m_sigma;
	archive << m_cC;
	archive << m_cSigma;

	archive << m_mean;
	archive << m_mutationDistribution;
}
