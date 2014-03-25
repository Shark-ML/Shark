/*!
 * 
 *
 * \brief       Implements the most recent version of the non-elitist CMA-ES.
 * 
 * The algorithm is described in
 * 
 * Hansen, N., S. Kern (2004). Evaluating the CMA Evolution Strategy
 * on Multimodal Test Functions. In Proceedings of the Eighth
 * International Conference on Parallel Problem Solving from Nature
 * (PPSN VIII), pp. 282-291, LNCS, Springer-Verlag
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
#include <shark/Algorithms/DirectSearch/CMA.h>

#include <shark/Core/Exception.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/MuKommaLambdaSelection.h>

using namespace shark;

//Functors used by the CMA-ES

namespace{
	struct FitnessComparator {
		template<typename Individual>
		bool operator()( const Individual & a, const Individual & b ) const {
			return( a.fitness( shark::tag::PenalizedFitness() )[0] < b.fitness( shark::tag::PenalizedFitness() )[0] );
		}
	};

	struct PointExtractor {
		template<typename T>
		const RealVector & operator()( const T & t ) const {
			return( *t );
		}
	};

	struct StepExtractor {
		template<typename T>
		const RealVector & operator()( const T & t ) const {
			return( boost::get<0>( t ) );
		}
	};
	
	double chi( unsigned int n ) {
		return( std::sqrt( static_cast<double>( n ) )*(1. - 1./(4.*n) + 1./(21.*n*n)) );
	}
	
	template<typename VectorType>
	double h( const VectorType & v, double cSigma, unsigned int n ) {
		static double d = norm_2( v ) / std::sqrt( 1 -sqr( 1 - cSigma ) );
		return( d < (1.5 + 1./(n-0.5))* chi(n) ? 1 : 0 );
	}
}

/**
* \brief Calculates lambda for the supplied dimensionality n.
*/
unsigned CMA::suggestLambda( unsigned int dimension ) {
	unsigned lambda = unsigned( 4. + ::floor( 3. * ::log( static_cast<double>( dimension ) ) ) );
	// heuristics for small search spaces
	lambda = std::max<unsigned int>( 5, std::min( lambda, dimension ) );
	return( lambda );
}

/**
* \brief Calculates mu for the supplied lambda and the recombination strategy.
*/
unsigned CMA::suggestMu( unsigned int lambda, RecombinationType recomb) {
	unsigned int mu;
	switch( recomb ) {
		case EQUAL:         
			mu = static_cast<unsigned int>( ::floor( lambda / 4. ) ); 
			break;
		case LINEAR:        
			mu = static_cast<unsigned int>( ::floor( lambda / 2. ) ); 
			break;
		case SUPERLINEAR:   
			mu = static_cast<unsigned int>( ::floor( lambda / 2. ) ); 
			break;
	}
	return( mu );
}

CMA::CMA()
:m_recombinationType( SUPERLINEAR )
, m_updateType( RANK_ONE_AND_MU )
, m_sigma( 0 )
, m_cC( 0 )
, m_cCU( 0 )
, m_cCov( 0 )
, m_cSigma( 0 )
, m_cSigmaU( 0 )
, m_dSigma( 0 )
, m_muEff( 0 )
, m_muCov( 0 ) {
	m_features |= REQUIRES_VALUE;
}

/**
* \brief Configures the algorithm based on the supplied configuration.
*/
void CMA::configure( const PropertyTree & node ) {
	m_recombinationType = static_cast<RecombinationType>( node.get<unsigned int>( "RecombinationType", SUPERLINEAR ) );
	m_updateType = static_cast<UpdateType>( node.get<unsigned int>( "UpdateType", RANK_ONE_AND_MU ) );
}

void CMA::read( InArchive & archive ) {
	archive >> m_numberOfVariables;
	archive >> m_mu;
	archive >> m_lambda;
	archive >> m_recombinationType;
	archive >> m_updateType;

	archive >> m_sigma;

	archive >> m_cC;
	archive >> m_cCU;
	archive >> m_cCov;
	archive >> m_cSigma;
	archive >> m_cSigmaU;
	archive >> m_dSigma;

	archive >> m_muEff;
	archive >> m_muCov;

	archive >> m_mean;
	archive >> m_weights;

	archive >> m_evolutionPathC;
	archive >> m_evolutionPathSigma;
	archive >> m_mutationDistribution;
}

void CMA::write( OutArchive & archive ) const {
	archive << m_numberOfVariables;
	archive << m_mu;
	archive << m_lambda;
	
	archive << m_recombinationType;
	archive << m_updateType;

	archive << m_sigma;

	archive << m_cC;
	archive << m_cCU;
	archive << m_cCov;
	archive << m_cSigma;
	archive << m_cSigmaU;
	archive << m_dSigma;

	archive << m_muEff;
	archive << m_muCov;

	archive << m_mean;
	archive << m_weights;

	archive << m_evolutionPathC;
	archive << m_evolutionPathSigma;
	archive << m_mutationDistribution;
}


void CMA::init( ObjectiveFunctionType const& function, SearchPointType const& p) {
	unsigned int lambda = CMA::suggestLambda( p.size() );
	unsigned int mu = CMA::suggestMu(  lambda, m_recombinationType );
	init( function,
		p,
		lambda,
		mu,
		1.0
	);
}

/**
* \brief Initializes the algorithm for the supplied objective function.
*/
void CMA::init( 
	ObjectiveFunctionType const& function, 
	SearchPointType const& initialSearchPoint,
	unsigned int lambda, 
	unsigned int mu,
	double initialSigma,				       
	const boost::optional< RealMatrix > & initialCovarianceMatrix
) {

	m_numberOfVariables = function.numberOfVariables();
	m_lambda = lambda;
	m_mu = mu;
	m_sigma = initialSigma;

	m_mean.resize( m_numberOfVariables );
	m_evolutionPathC.resize( m_numberOfVariables );
	m_evolutionPathSigma.resize( m_numberOfVariables );
	m_mutationDistribution.resize( m_numberOfVariables );
	m_mean.clear();
	m_evolutionPathC.clear();
	m_evolutionPathSigma.clear();
	if(initialCovarianceMatrix){
		m_mutationDistribution.setCovarianceMatrix(*initialCovarianceMatrix);
	}
	
	
	//weighting of the k-best individuals
	m_weights.resize(mu);
	switch (m_recombinationType) {
	case EQUAL:
		for (unsigned int i = 0; i < mu; i++)
			m_weights(i) = 1;
		break;
	case LINEAR:
		for (unsigned int i = 0; i < mu; i++)
			m_weights(i) = mu-i;
		break;
	case SUPERLINEAR:
		for (unsigned int i = 0; i < mu; i++)
			m_weights(i) = ::log(mu + 1.) - ::log(1. + i);
		break;
	}
	m_weights /= sum(m_weights);
	m_muEff = 1. / sum(sqr(m_weights));

	// Step size control
	m_cSigma = (m_muEff + 2.)/(m_numberOfVariables + m_muEff + 3.);
	m_dSigma = 1. + 2. * std::max(0., ::sqrt((m_muEff-1.)/(m_numberOfVariables+1)) - 1.) + m_cSigma;

	// Covariance matrix adaptation
	switch (m_updateType) {
	case RANK_ONE:
		m_muCov = 1;
		break;
	case RANK_ONE_AND_MU:
		m_muCov = m_muEff;
		break;
	}

	m_cC = 4. / (4. + m_numberOfVariables);
	m_cCov = 1. / m_muCov * 2. / sqr(m_numberOfVariables + ::sqrt(2.)) +
		(1 - 1. / m_muCov) * std::min(1., (2 * m_muEff - 1) / (sqr(m_numberOfVariables + 2) + m_muEff));

	m_cCU = ::sqrt((2 - m_cC) * m_cC);
	m_cSigmaU = ::sqrt((2 - m_cSigma) * m_cSigma);

	m_mean = initialSearchPoint;
	m_best.point = initialSearchPoint; // CI: you can argue about this, as the point is not evaluated
	m_best.value = function(initialSearchPoint);//OK: evaluating performance of first point :P
}

/**
* \brief Updates the strategy parameters based on the supplied offspring population.
*/
void CMA::updateStrategyParameters( const std::vector<TypedIndividual<RealVector, RealVector> > & offspring ) {
	RealVector xPrime = cog( offspring, m_weights, PointExtractor() );
	RealVector cogSteps = cog( offspring, m_weights, StepExtractor() );

	RealVector v = blas::prod( m_mutationDistribution.eigenVectors(), cogSteps );

	// Covariance Matrix Update
	m_evolutionPathC = (1. - m_cC ) * m_evolutionPathC + m_cCU * std::sqrt( m_muEff )  / m_sigma * (xPrime-m_mean);
	RealMatrix Z( m_numberOfVariables, m_numberOfVariables, 0.0);
	RealMatrix C( m_mutationDistribution.covarianceMatrix() );
	// Rank-1-Update
	C = (1.-m_cCov) * C +
		m_cCov / m_muCov * blas::outer_prod( m_evolutionPathC, m_evolutionPathC );
	// Rank-mu-Update
	for( unsigned int i = 0; i < m_mu; i++ ) {
		Z += m_weights( i ) * blas::outer_prod(
			*offspring[i] - m_mean,
			*offspring[i] - m_mean
		);
	}
	C += m_cCov * (1.-1./m_muCov) * 1./sqr( m_sigma ) * Z;
	m_mutationDistribution.setCovarianceMatrix( C );
	
	// Step size update
	m_evolutionPathSigma = (1. - m_cSigma)*m_evolutionPathSigma +m_cSigmaU * std::sqrt( m_muEff ) * v;
	m_sigma *= std::exp( (m_cSigma / m_dSigma) * (norm_2(m_evolutionPathSigma)/ chi( m_numberOfVariables ) - 1.) );
	
	
	// check for numerical stability
	double ev = m_mutationDistribution.eigenValues()( m_mutationDistribution.eigenValues().size() - 1 );
	if( m_sigma * std::sqrt( std::fabs( ev ) ) < 1E-20 )
		m_sigma = 1E-20 / std::sqrt( std::fabs( ev ) );

	m_mean = xPrime;
}

/**
* \brief Executes one iteration of the algorithm.
*/
void CMA::step(ObjectiveFunctionType const& function){

	std::vector< TypedIndividual<RealVector, RealVector> > offspring( m_lambda );

	shark::soo::PenalizingEvaluator penalizingEvaluator;
	for( unsigned int i = 0; i < offspring.size(); i++ ) {
		MultiVariateNormalDistribution::ResultType sample = m_mutationDistribution();
		offspring[ i ].get<0>() = sample.second;
		*offspring[i] = m_mean + m_sigma * sample.first;

		boost::tuple< ObjectiveFunctionType::ResultType, ObjectiveFunctionType::ResultType > evalResult;
		evalResult = penalizingEvaluator( function, *offspring[i] );

		offspring[i].fitness( shark::tag::UnpenalizedFitness() )[0] = boost::get< shark::soo::PenalizingEvaluator::UNPENALIZED_RESULT >( evalResult );
		offspring[i].fitness( shark::tag::PenalizedFitness() )[0] = boost::get< shark::soo::PenalizingEvaluator::PENALIZED_RESULT >( evalResult );
	}

	// Selection
	std::vector< TypedIndividual<RealVector, RealVector> > parents( m_mu );
	select_mu_komma_lambda_p( 
		parents.begin(),
		parents.end(),
		offspring.begin(),
		offspring.end() ,
		FitnessComparator()
	);
	// Strategy parameter update
	updateStrategyParameters( parents );

	m_best.point= *parents[ 0 ];
	m_best.value= parents[ 0 ].fitness( shark::tag::UnpenalizedFitness() )[0];
}
