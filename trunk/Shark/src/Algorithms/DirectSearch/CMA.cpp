/*!
 *
 * \brief       Implements the most recent version of the non-elitist CMA-ES.
 * 
 * The algorithm is described in
 * Hansen, N. The CMA Evolution Startegy: A Tutorial, June 28, 2011
 * and the eqation numbers refer to this publication (retrieved April 2014).
 * 
 *
 * \author      Thomas Voss and Christian Igel
 * \date        April 2014
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
#include <shark/Algorithms/DirectSearch/CMA.h>

#include <shark/Core/Exception.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/ElitistSelection.h>

using namespace shark;

//Functors used by the CMA-ES

namespace{
	struct PointExtractor {
		template<typename T>
		const RealVector & operator()( const T & t ) const {
			return t.searchPoint();
		}
	};

	struct StepExtractor {
		template<typename T>
		const RealVector & operator()( const T & t ) const {
			return  t.chromosome();
		}
	};
	
	double chi( unsigned int n ) {
		return( std::sqrt( static_cast<double>( n ) )*(1. - 1./(4.*n) + 1./(21.*n*n)) );
	}

}

/**
* \brief Calculates lambda for the supplied dimensionality n.
*/
unsigned CMA::suggestLambda( unsigned int dimension ) {
	unsigned lambda = unsigned( 4. + ::floor( 3. * ::log( static_cast<double>( dimension ) ) ) ); // eq. (44)
	// heuristic for small search spaces
	lambda = std::max<unsigned int>( 5, std::min( lambda, dimension ) );
	return( lambda );
}

/**
* \brief Calculates mu for the supplied lambda and the recombination strategy.
*/
double CMA::suggestMu( unsigned int lambda, RecombinationType recomb) {
	switch( recomb ) {
		case EQUAL:         
			return lambda / 4.;
		case LINEAR:        
			return lambda / 2.; 
		case SUPERLINEAR:   
			return lambda / 2.; // eq. (44)
	}
	return 0;
}

CMA::CMA()
:m_recombinationType( SUPERLINEAR )
, m_sigma( 0 )
, m_cC( 0 )
, m_c1( 0 )
, m_cMu( 0 )
, m_cSigma( 0 )
, m_dSigma( 0 )
, m_muEff( 0 )
, m_lowerBound( 1E-20)
, m_counter( 0 ) {
	m_features |= REQUIRES_VALUE;
}

/**
* \brief Configures the algorithm based on the supplied configuration.
*/
void CMA::configure( const PropertyTree & node ) {
	m_recombinationType = static_cast<RecombinationType>( node.get<unsigned int>( "RecombinationType", SUPERLINEAR ) );
}

void CMA::read( InArchive & archive ) {
	archive >> m_numberOfVariables;
	archive >> m_mu;
	archive >> m_lambda;
	archive >> m_recombinationType;
	archive >> m_lowerBound;

	archive >> m_sigma;

	archive >> m_cC;
	archive >> m_c1;
	archive >> m_cMu;
	archive >> m_cSigma;
	archive >> m_dSigma;

	archive >> m_muEff;

	archive >> m_mean;
	archive >> m_weights;

	archive >> m_evolutionPathC;
	archive >> m_evolutionPathSigma;
	archive >> m_mutationDistribution;

	archive >> m_counter;
}

void CMA::write( OutArchive & archive ) const {
	archive << m_numberOfVariables;
	archive << m_mu;
	archive << m_lambda;
	
	archive << m_recombinationType;
	archive << m_lowerBound;

	archive << m_sigma;

	archive << m_cC;
	archive << m_c1;
	archive << m_cMu;
	archive << m_cSigma;
	archive << m_dSigma;

	archive << m_muEff;

	archive << m_mean;
	archive << m_weights;

	archive << m_evolutionPathC;
	archive << m_evolutionPathSigma;
	archive << m_mutationDistribution;

	archive << m_counter;
}


void CMA::init( ObjectiveFunctionType const& function, SearchPointType const& p) {
	unsigned int lambda = CMA::suggestLambda( p.size() );
	unsigned int mu = CMA::suggestMu(  lambda, m_recombinationType );
	init( function,
		p,
		lambda,
		mu,
		1.0/std::sqrt(double(p.size()))
	);
}

/**
* \brief Initializes the algorithm for the supplied objective function.
*/
void CMA::init( 
	ObjectiveFunctionType const& function, 
	SearchPointType const& initialSearchPoint,
	unsigned int lambda, 
	double mu,
	double initialSigma,				       
	const boost::optional< RealMatrix > & initialCovarianceMatrix
) {

	m_numberOfVariables = function.numberOfVariables();
	m_lambda = lambda;
	m_mu = static_cast<unsigned int>(::floor(mu));
	m_sigma = initialSigma;

	m_mean.resize( m_numberOfVariables );
	m_evolutionPathC.resize( m_numberOfVariables );
	m_evolutionPathSigma.resize( m_numberOfVariables );
	m_mutationDistribution.resize( m_numberOfVariables );
	m_mean.clear();
	m_evolutionPathC.clear();
	m_evolutionPathSigma.clear();
	if(initialCovarianceMatrix){
		m_mutationDistribution.covarianceMatrix() = *initialCovarianceMatrix;
		m_mutationDistribution.update();
	}
		
	//weighting of the k-best individuals
	m_weights.resize(m_mu);
	switch (m_recombinationType) {
	case EQUAL:
		for (unsigned int i = 0; i < m_mu; i++)
			m_weights(i) = 1;
		break;
	case LINEAR:
		for (unsigned int i = 0; i < m_mu; i++)
			m_weights(i) = mu-i;
		break;
	case SUPERLINEAR:
		for (unsigned int i = 0; i < m_mu; i++)
			m_weights(i) = ::log(mu + 0.5) - ::log(1. + i); // eq. (45)
		break;
	}
	m_weights /= sum(m_weights); // eq. (45)
	m_muEff = 1. / sum(sqr(m_weights)); // equal to sum(m_weights)^2 / sum(sqr(m_weights))

	// Step size control
	m_cSigma = (m_muEff + 2.)/(m_numberOfVariables + m_muEff + 5.); // eq. (46)
	m_dSigma = 1. + 2. * std::max(0., ::sqrt((m_muEff-1.)/(m_numberOfVariables+1)) - 1.) + m_cSigma; // eq. (46)

	m_cC = (4. + m_muEff / m_numberOfVariables) / (m_numberOfVariables + 4. +  2 * m_muEff / m_numberOfVariables); // eq. (47)
	m_c1 = 2 / (sqr(m_numberOfVariables + 1.3) + m_muEff); // eq. (48)
	double alphaMu = 2.;
	m_cMu = std::min(1. - m_c1, alphaMu * (m_muEff - 2. + 1./m_muEff) / (sqr(m_numberOfVariables + 2) + alphaMu * m_muEff / 2)); // eq. (49)

	m_mean = initialSearchPoint;
	m_best.point = initialSearchPoint; // CI: you can argue about this, as the point is not evaluated
	m_best.value = function(initialSearchPoint); //OK: evaluating performance of first point :P

	m_lowerBound = 1E-20;
	m_counter = 0;
}

/**
* \brief Updates the strategy parameters based on the supplied offspring population.
*/
void CMA::updateStrategyParameters( const std::vector<Individual<RealVector, double, RealVector> > & offspring ) {
	RealVector z = weightedSum( offspring, m_weights, StepExtractor() ); // eq. (38)
	RealVector m = weightedSum( offspring, m_weights, PointExtractor() ); // eq. (39) 
	RealVector y = (m - m_mean) / m_sigma;

	// Step size update
	RealVector CInvY = blas::prod( m_mutationDistribution.eigenVectors(), z ); // C^(-1/2)y = Bz
	m_evolutionPathSigma = (1. - m_cSigma)*m_evolutionPathSigma + std::sqrt( m_cSigma * (2. - m_cSigma) * m_muEff ) * CInvY; // eq. (40)
	m_sigma *= std::exp( (m_cSigma / m_dSigma) * (norm_2(m_evolutionPathSigma)/ chi( m_numberOfVariables ) - 1.) ); // eq. (39)

	// Covariance matrix update
	RealMatrix& C = m_mutationDistribution.covarianceMatrix();
	RealMatrix Z( m_numberOfVariables, m_numberOfVariables, 0.0); // matric for rank-mu update
	for( unsigned int i = 0; i < m_mu; i++ ) {
		noalias(Z) += m_weights( i ) * blas::outer_prod(
			offspring[i].searchPoint() - m_mean,
			offspring[i].searchPoint() - m_mean
		);
	}
	
	double hSigLHS = norm_2( m_evolutionPathSigma ) / std::sqrt(1. - pow((1 - m_cSigma), 2.*(m_counter+1)));
	double hSigRHS = (1.4 + 2 / (m_numberOfVariables+1.)) * chi( m_numberOfVariables );
	double hSig = 0;
	if(hSigLHS < hSigRHS) hSig = 1.;
	double deltaHSig = (1.-hSig) * m_cC * (2. - m_cC);

	m_evolutionPathC = (1. - m_cC ) * m_evolutionPathC + hSig * std::sqrt( m_cC * (2. - m_cC) * m_muEff ) * y; // eq. (42)
	noalias(C) = (1.-m_c1 - m_cMu) * C + m_c1 * ( blas::outer_prod( m_evolutionPathC, m_evolutionPathC ) + deltaHSig * C) + m_cMu * 1./sqr( m_sigma ) * Z; // eq. (43)

	// update mutation distribution
	m_mutationDistribution.update();
	
	// check for numerical stability
	double ev = m_mutationDistribution.eigenValues()( m_mutationDistribution.eigenValues().size() - 1 );
	if( m_sigma * std::sqrt( std::fabs( ev ) ) < m_lowerBound )
		m_sigma = m_lowerBound / std::sqrt( std::fabs( ev ) );

	m_mean = m;
}

/**
* \brief Executes one iteration of the algorithm.
*/
void CMA::step(ObjectiveFunctionType const& function){

	std::vector< Individual<RealVector, double, RealVector> > offspring( m_lambda );

	PenalizingEvaluator penalizingEvaluator;
	for( unsigned int i = 0; i < offspring.size(); i++ ) {
		MultiVariateNormalDistribution::result_type sample = m_mutationDistribution();
		offspring[i].chromosome() = sample.second;
		offspring[i].searchPoint() = m_mean + m_sigma * sample.first;
	}
	penalizingEvaluator( function, offspring.begin(), offspring.end() );

	// Selection
	std::vector< Individual<RealVector, double, RealVector> > parents( m_mu );
	ElitistSelection<FitnessExtractor> selection;
	selection(offspring.begin(),offspring.end(),parents.begin(), parents.end());
	// Strategy parameter update
	m_counter++; // increase generation counter
	updateStrategyParameters( parents );

	m_best.point= parents[ 0 ].searchPoint();
	m_best.value= parents[ 0 ].unpenalizedFitness();
}
