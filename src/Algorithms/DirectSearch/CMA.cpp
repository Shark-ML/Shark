/*!
 * 
 * \file        CMA.cpp
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
#include <shark/Algorithms/DirectSearch/Operators/Initializers/CovarianceMatrixInitializer.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/GlobalIntermediateRecombination.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/MuKommaLambdaSelection.h>

#include <boost/lambda/lambda.hpp>
#include <boost/math/special_functions.hpp>

using namespace shark;


//Chromosome

cma::Chromosome::Chromosome( unsigned int dimension) : m_recombinationType( SUPERLINEAR ),
	m_updateType( RANK_ONE_AND_MU ),
	m_sigma( 0 ),
	m_cC( 0 ),
	m_cCU( 0 ),
	m_cCov( 0 ),
	m_cSigma( 0 ),
	m_cSigmaU( 0 ),
	m_dSigma( 0 ),
	m_muEff( 0 ),
	m_muCov( 0 ) {
}

/**
* \brief Adjusts the dimension of the chromosome.
*/
void cma::Chromosome::setDimension( unsigned int dimension ) {
	m_mean.resize( dimension );
	m_evolutionPathC.resize( dimension );
	m_evolutionPathSigma.resize( dimension );
	m_mutationDistribution.resize( dimension );
	m_mean.clear();
	m_evolutionPathC.clear();
	m_evolutionPathSigma.clear();
}

//Functors used by the CMA-ES

namespace{
	struct FitnessComparator {
		template<typename Individual>
		bool operator()( const Individual & a, const Individual & b ) const {
			return( a.fitness( shark::tag::PenalizedFitness() )[0] < b.fitness( shark::tag::PenalizedFitness() )[0] );
		}
	};

	struct IdentityExtractor {
		template<typename T>
		const T & operator()( const T & t ) const {
			return( t );
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
unsigned CMA::suggestMu( unsigned int lambda, shark::cma::RecombinationType recomb) {
	unsigned int mu;
	switch( recomb ) {
		case shark::cma::EQUAL:         
			mu = static_cast<unsigned int>( ::floor( lambda / 4. ) ); 
			break;
		case shark::cma::LINEAR:        
			mu = static_cast<unsigned int>( ::floor( lambda / 2. ) ); 
			break;
		case shark::cma::SUPERLINEAR:   
			mu = static_cast<unsigned int>( ::floor( lambda / 2. ) ); 
			break;
	}
	return( mu );
}

CMA::CMA() {
	m_features |= REQUIRES_VALUE;
}

/**
* \brief Configures the algorithm based on the supplied configuration.
*/
void CMA::configure( const PropertyTree & node ) {
	m_chromosome.m_recombinationType = static_cast<shark::cma::RecombinationType>( node.get<unsigned int>( "RecombinationType", shark::cma::SUPERLINEAR ) );
	m_chromosome.m_updateType = static_cast<shark::cma::UpdateType>( node.get<unsigned int>( "UpdateType", shark::cma::RANK_MU ) );
}

void CMA::read( InArchive & archive ) {
	archive >> m_numberOfVariables;
	archive >> m_mu;
	archive >> m_lambda;
	archive >> m_chromosome;
}

void CMA::write( OutArchive & archive ) const {
	archive << m_numberOfVariables;
	archive << m_mu;
	archive << m_lambda;
	archive << m_chromosome;
}


void CMA::init( ObjectiveFunctionType const& function, SearchPointType const& p) {
	unsigned int lambda = CMA::suggestLambda( p.size() );
	init( p.size(),
		lambda,
		CMA::suggestMu(  lambda, m_chromosome.m_recombinationType ),
		p,1.0
	);
}

/**
* \brief Initializes the algorithm for the supplied objective function.
*/
void CMA::init( 
	unsigned int numberOfVariables, 
	unsigned int lambda, 
	unsigned int mu,
	const RealVector & initialSearchPoint,
	double initialSigma,				       
	const boost::optional< RealMatrix > & initialCovarianceMatrix
) {

	m_numberOfVariables = numberOfVariables;

	m_lambda = lambda;
	m_mu = mu;
	m_chromosome.m_sigma = initialSigma;

	shark::cma::Initializer initializer;
	initializer( m_chromosome,
		 m_numberOfVariables,
		 m_lambda,
		 m_mu,
		 m_chromosome.m_sigma
	);

	m_chromosome.m_mean = initialSearchPoint;
	m_best.point = initialSearchPoint; // CI: you can argue about this, as the point is not evaluated
}

/**
* \brief Updates the strategy parameters based on the supplied offspring population.
*/
void CMA::updateStrategyParameters( const std::vector<TypedIndividual<RealVector, RealVector> > & offspring ) {
	RealVector xPrime = cog( offspring, m_chromosome.m_weights, PointExtractor() );
	RealVector cogSteps = cog( offspring, m_chromosome.m_weights, StepExtractor() );

	RealVector v = blas::prod( m_chromosome.m_mutationDistribution.eigenVectors(), cogSteps );

	m_chromosome.m_evolutionPathC = (1. - m_chromosome.m_cC ) * m_chromosome.m_evolutionPathC 
		+ m_chromosome.m_cCU * std::sqrt( m_chromosome.m_muEff )  / m_chromosome.m_sigma * (xPrime-m_chromosome.m_mean);

	// Covariance Matrix Update
	RealMatrix Z( m_numberOfVariables, m_numberOfVariables, 0.0);

	RealMatrix C( m_chromosome.m_mutationDistribution.covarianceMatrix() );
	// Rank-1-Update
	C = (1.-m_chromosome.m_cCov) * C +
		m_chromosome.m_cCov / m_chromosome.m_muCov * blas::outer_prod( m_chromosome.m_evolutionPathC, m_chromosome.m_evolutionPathC );
	// Rank-mu-Update
	for( unsigned int i = 0; i < m_mu; i++ ) {
		Z += m_chromosome.m_weights( i ) * blas::outer_prod(
			*offspring[i] - m_chromosome.m_mean,
			*offspring[i] - m_chromosome.m_mean
		);
	}
	C += m_chromosome.m_cCov * (1.-1./m_chromosome.m_muCov) * 1./sqr( m_chromosome.m_sigma ) * Z;
	m_chromosome.m_mutationDistribution.setCovarianceMatrix( C );

	// Step size update
	m_chromosome.m_evolutionPathSigma = (1. - m_chromosome.m_cSigma)*m_chromosome.m_evolutionPathSigma +
		m_chromosome.m_cSigmaU * std::sqrt( m_chromosome.m_muEff ) * v;

	double sum = norm_2(m_chromosome.m_evolutionPathSigma);
	//~ for( unsigned int i = 0; i < m_chromosome.m_evolutionPathSigma.size(); i++ )
		//~ sum += sqr( m_chromosome.m_evolutionPathSigma( i ) );
	//~ sum = std::sqrt( sum );

	m_chromosome.m_sigma *= std::exp( (m_chromosome.m_cSigma / m_chromosome.m_dSigma) * (sum/ chi( m_numberOfVariables ) - 1.) );
	double ev = m_chromosome.m_mutationDistribution.eigenValues()( m_chromosome.m_mutationDistribution.eigenValues().size() - 1 );

	if( m_chromosome.m_sigma * std::sqrt( std::fabs( ev ) ) < 1E-20 )
		m_chromosome.m_sigma = 1E-20 / std::sqrt( std::fabs( ev ) );

	m_chromosome.m_mean = xPrime;
}

/**
* \brief Executes one iteration of the algorithm.
*/
void CMA::step(ObjectiveFunctionType const& function){

	std::vector< TypedIndividual<RealVector, RealVector> > offspring( m_lambda );

	shark::soo::PenalizingEvaluator penalizingEvaluator;
	for( unsigned int i = 0; i < offspring.size(); i++ ) {
		MultiVariateNormalDistribution::ResultType sample = m_chromosome.m_mutationDistribution();
		offspring[ i ].get<0>() = sample.second;
		*offspring[i] = m_chromosome.m_mean + m_chromosome.m_sigma * sample.first;

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
