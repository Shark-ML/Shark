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
#include <shark/Algorithms/DirectSearch/CMA.h>

#include <shark/Core/Exception.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/ElitistSelection.h>
#include <shark/Core/utility/KeyValuePair.h>
#include <algorithm>
using namespace shark;

namespace{
	//computes percentile({|1-R|,|2-R|,...,|N-R|},q)
	int deltaLim(int rank, std::size_t N, double percentile){
		blas::vector<int> rankDistr(N);
		std::iota(rankDistr.begin(),rankDistr.end(),1-rank);
		noalias(rankDistr) = abs(rankDistr);
		//~ std::sort(rankDistr.begin(),rankDistr.end());
		auto pos = rankDistr.begin() +std::size_t(percentile * (rankDistr.size()-1));
		std::nth_element(rankDistr.begin(), pos, rankDistr.end());
		return *pos;
	}
}

/**
* \brief Calculates lambda for the supplied dimensionality n.
*/
std::size_t CMA::suggestLambda( std::size_t dimension ) {
	std::size_t lambda = std::size_t( 4. + ::floor( 3. * ::log( static_cast<double>( dimension ) ) ) ); // eq. (44)
	// heuristic for small search spaces
	lambda = std::max<std::size_t>( 5, std::min( lambda, dimension ) );
	return lambda;
}

/**
* \brief Calculates mu for the supplied lambda and the recombination strategy.
*/
std::size_t CMA::suggestMu( std::size_t lambda, RecombinationType recomb) {
	switch( recomb ) {
		case EQUAL:         
			return lambda / 4;
		case LINEAR:        
			return lambda / 2; 
		case SUPERLINEAR:   
			return lambda / 2;
	}
	return 0;
}

CMA::CMA(random::rng_type& rng)
: m_userSetMu(false)
, m_userSetLambda(false)
, m_initSigma(-1)
, m_recombinationType( SUPERLINEAR )
, m_sigma( 0 )
, m_cC( 0 )
, m_c1( 0 )
, m_cMu( 0 )
, m_cSigma( 0 )
, m_dSigma( 0 )
, m_muEff( 0 )
, m_lowerBound( 1E-40)
, m_counter( 0 )
, mpe_rng(&rng){
	m_features |= REQUIRES_VALUE;
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
	
	archive >> m_numEvaluations;
	archive >> m_numEvalIncreaseFactor;
	archive >> m_rLambda;
	archive >> m_rankChangeQuantile;
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
	
	archive << m_numEvaluations;
	archive << m_numEvalIncreaseFactor;
	archive << m_rLambda;
	archive << m_rankChangeQuantile;
}


void CMA::init( ObjectiveFunctionType const& function, SearchPointType const& p) {
	SIZE_CHECK(p.size() == function.numberOfVariables());
	checkFeatures(function);
	std::vector<RealVector> points(1,p);
	std::vector<double> functionValues(1,function.eval(p));

	std::size_t lambda = m_userSetLambda? m_lambda:CMA::suggestLambda( p.size() );
	std::size_t mu  = m_userSetMu? m_mu:CMA::suggestMu(lambda, m_recombinationType);
	RANGE_CHECK(mu < lambda);
	double sigma = (m_initSigma > 0)? m_initSigma : 1.0/std::sqrt(double(p.size()));
	doInit(
		points,
		functionValues,
		lambda,
		mu,
		sigma
	);
}

void CMA::init( 
	ObjectiveFunctionType const& function, 
	SearchPointType const& p,
	std::size_t lambda,
	std::size_t mu,
	double initialSigma,				       
	const boost::optional< RealMatrix > & initialCovarianceMatrix
) {
	SIZE_CHECK(p.size() == function.numberOfVariables());
	RANGE_CHECK(mu < lambda);
	setMu(mu);
	setLambda(lambda);
	checkFeatures(function);
	std::vector<RealVector> points(1,p);
	std::vector<double> functionValues(1,function.eval(p));
	doInit(
		points,
		functionValues,
		lambda,
		mu,
		initialSigma
	);
	if(initialCovarianceMatrix){
		m_mutationDistribution.covarianceMatrix() = *initialCovarianceMatrix;
		m_mutationDistribution.update();
	}
}
void CMA::doInit( 
	std::vector<SearchPointType> const& initialSearchPoints,
	std::vector<ResultType> const& initialValues,
	std::size_t lambda,
	std::size_t mu,
	double initialSigma
) {
	SIZE_CHECK(initialSearchPoints.size() > 0);
	
	m_numberOfVariables =initialSearchPoints[0].size();
	m_lambda = lambda;
	m_mu = mu;
	m_sigma =  initialSigma;
	m_numEvalIncreaseFactor = 1.25;
	
	m_numEvaluations = 1;
	m_rLambda = std::max(0.1, 2.0 / m_lambda);
	m_rankChangeQuantile = 0.2 * 0.5; //0.2 *50% quantile
	m_mean.resize( m_numberOfVariables );
	m_evolutionPathC.resize( m_numberOfVariables );
	m_evolutionPathSigma.resize( m_numberOfVariables );
	m_mutationDistribution.resize( m_numberOfVariables );
	m_mean.clear();
	m_evolutionPathC.clear();
	m_evolutionPathSigma.clear();
	
		
	//weighting of the k-best individuals
	m_weights.resize(m_mu);
	switch (m_recombinationType) {
	case EQUAL:
		for (std::size_t i = 0; i < m_mu; i++)
			m_weights(i) = 1;
		break;
	case LINEAR:
		for (std::size_t i = 0; i < m_mu; i++)
			m_weights(i) = (double)(mu-i);
		break;
	case SUPERLINEAR:
		for (std::size_t i = 0; i < m_mu; i++)
			m_weights(i) = ::log(mu + 0.5) - ::log(1. + i); // eq. (45)
		break;
	}
	m_weights /= sum(m_weights); // eq. (45)
	m_muEff = 1. / sum(sqr(m_weights)); // equal to sum(m_weights)^2 / sum(sqr(m_weights))

	// Step size control
	m_cSigma = (m_muEff + 2.)/(m_numberOfVariables + m_muEff + 3.); // eq. (46)
	m_dSigma = 1.0 + 2. * std::max(0., ::sqrt((m_muEff-1.)/(m_numberOfVariables+1)) - 1.) + m_cSigma; // eq. (46)

	m_cC = (4. + m_muEff / m_numberOfVariables) / (m_numberOfVariables + 4. +  2 * m_muEff / m_numberOfVariables); // eq. (47)
	m_c1 = 2 / (sqr(m_numberOfVariables + 1.3) + m_muEff); // eq. (48)
	double alphaMu = 2.;
	double rankMuAlpha = 0.3;//but is it really?
	m_cMu = std::min(1. - m_c1, alphaMu * ( rankMuAlpha + m_muEff - 2. + 1./m_muEff) / (sqr(m_numberOfVariables + 2) + alphaMu * m_muEff / 2)); // eq. (49)
	
	std::size_t pos = std::min_element(initialValues.begin(),initialValues.end())-initialValues.begin();
	m_mean = initialSearchPoints[pos];
	m_best.point = initialSearchPoints[pos];
	m_best.value = initialValues[pos];
	m_lowerBound = 1E-40;
	m_counter = 0;
}

std::vector<CMA::IndividualType> CMA::generateOffspring( ) const{
	std::vector< IndividualType > offspring( m_lambda );
	for( std::size_t i = 0; i < offspring.size(); i++ ) {
		MultiVariateNormalDistribution::result_type sample = m_mutationDistribution(*mpe_rng);
		offspring[i].chromosome() = sample.second;
		offspring[i].searchPoint() = m_mean + m_sigma * sample.first;
	}
	return offspring;
}

void CMA::updatePopulation( std::vector<IndividualType> const& offspring ) {
	std::vector< IndividualType > selectedOffspring( m_mu );
	ElitistSelection<IndividualType::FitnessOrdering > selection;
	selection(offspring.begin(),offspring.end(),selectedOffspring.begin(), selectedOffspring.end());
	m_counter++;
	
	RealVector z( m_numberOfVariables, 0. );
	RealVector m( m_numberOfVariables, 0. );
	for( std::size_t j = 0; j < selectedOffspring.size(); j++ ){
		noalias(z) += m_weights( j ) * selectedOffspring[j].chromosome();// eq. (38)
		noalias(m) += m_weights( j ) * selectedOffspring[j].searchPoint();// eq. (39)
	}
	RealVector y = (m - m_mean) / m_sigma;

	// Covariance matrix update
	RealMatrix& C = m_mutationDistribution.covarianceMatrix();
	RealMatrix Z( m_numberOfVariables, m_numberOfVariables, 0.0); // matric for rank-mu update
	for( std::size_t i = 0; i < m_mu; i++ ) {
		noalias(Z) += m_weights( i ) * blas::outer_prod(
			selectedOffspring[i].searchPoint() - m_mean,
			selectedOffspring[i].searchPoint() - m_mean
		);
	}
	double n = static_cast<double>(m_numberOfVariables);
	double expectedChi = std::sqrt( n )*(1. - 1./(4.*n) + 1./(21.*n*n));
	double hSigLHS = norm_2( m_evolutionPathSigma ) / std::sqrt(1. - pow((1 - m_cSigma), 2.*(m_counter+1)));
	double hSigRHS = (1.4 + 2 / (m_numberOfVariables + 1.)) * expectedChi;
	double hSig = 0;
	if(hSigLHS < hSigRHS) hSig = 1.;
	double deltaHSig = (1.-hSig*hSig) * m_cC * (2. - m_cC);

	m_evolutionPathC = (1. - m_cC ) * m_evolutionPathC + hSig * std::sqrt( m_cC * (2. - m_cC) * m_muEff ) * y; // eq. (42)
	noalias(C) = (1.-m_c1 - m_cMu) * C + m_c1 * ( blas::outer_prod( m_evolutionPathC, m_evolutionPathC ) + deltaHSig * C) + m_cMu * 1./sqr( m_sigma ) * Z; // eq. (43)

	// Step size update
	RealVector CInvY = blas::prod( m_mutationDistribution.eigenVectors(), z ); // C^(-1/2)y = Bz
	m_evolutionPathSigma = (1. - m_cSigma)*m_evolutionPathSigma + std::sqrt( m_cSigma * (2. - m_cSigma) * m_muEff ) * CInvY; // eq. (40)
	m_sigma *= std::exp((m_cSigma / m_dSigma) * (norm_2(m_evolutionPathSigma) / expectedChi - 1.)); // eq. (39)

	// update mutation distribution
	m_mutationDistribution.update();
	
	//mean update
	m_mean = m;
	
	// check for numerical stability
	double ev = m_mutationDistribution.eigenValues()( m_mutationDistribution.eigenValues().size() - 1 );
	if( m_sigma * std::sqrt( std::fabs( ev ) ) < m_lowerBound )
		m_sigma = m_lowerBound / std::sqrt( std::fabs( ev ) );

	//store best point
	m_best.point= selectedOffspring[ 0 ].searchPoint();
	m_best.value= selectedOffspring[ 0 ].unpenalizedFitness();

}
void CMA::step(ObjectiveFunctionType const& function){
	std::vector<IndividualType> offspring = generateOffspring();
	PenalizingEvaluator penalizingEvaluator;
	penalizingEvaluator.m_numEvaluations = m_numEvaluations;
	penalizingEvaluator( function, offspring.begin(), offspring.end() );
	//check if the number of Evaluations must be increased on a noisy function
	if(function.isNoisy()){
		//compute number of points to reevaluate
		double reevalFraction = m_rLambda * m_lambda;
		std::size_t lambdaReeval = std::lround(m_rLambda * m_lambda);
		double rest = reevalFraction - lambdaReeval;
		if(rest > 0){
			lambdaReeval += random::coinToss(*mpe_rng,rest);
		}
		//only continue if we have at elast one point
		if(lambdaReeval > 0){
			//save old function values of the population
			std::vector<shark::KeyValuePair<double,std::size_t> > ranksOld(m_lambda);
			for(std::size_t i = 0; i != m_lambda; ++i){
				ranksOld[i].key = offspring[i].penalizedFitness();
				ranksOld[i].value = i;
			}
			
			//compute and save new function values of the population
			penalizingEvaluator(function,offspring.begin(),offspring.begin() + lambdaReeval);
			std::vector<shark::KeyValuePair<double,std::size_t> > ranksNew(m_lambda);
			for(std::size_t i = 0; i != m_lambda; ++i){
				ranksNew[i].key = offspring[i].penalizedFitness();
				ranksNew[i].value = i;
			}
			
			// update function values of the population to be the mean of old and new values,
			// this gives some more stability
			// This is a difference compared to
			// Hansen, N., et al. "A method for handling uncertainty in evolutionary
			// optimization with an application to feedback control of combustion." 
			// IEEE Transactions on Evolutionary Computation 13.1 (2009): 180-197.
			// While Hansen is considering the ranks to be more stable, the average
			// rank is a biased estimate, while the average value
			// leads to an unbiased estimate of the true rank.
			// This difference can be felt in noise with heavy outliers where rank averaging
			// has the same effect as ignoring the worst quantiles. In a skewed noise distribution the
			// average is however dominated by the outliers -> rank averaging
			// optimizes a different, easier, function and is not guarantueed to converge to
			// the true optimum for non-stationary noise.
			for(std::size_t i = 0; i != lambdaReeval; ++i){
				offspring[i].unpenalizedFitness() += ranksOld[i].key;
				offspring[i].unpenalizedFitness() /= 2;
			}
			
			//compute the noise estimate by computing a statistic over the rank changes
			//of the points that got evaluated twice
			
			//compute old and new ranks
			std::sort(ranksOld.begin(),ranksOld.end());
			std::sort(ranksNew.begin(),ranksNew.end());
			std::vector<std::pair<int, int> > ranks(lambdaReeval);
			for(int i = 0; i != (int)m_lambda; ++i){
				if(ranksOld[i].value < lambdaReeval){
					ranks[ranksOld[i].value].first = i;
				}
				if(ranksNew[i].value < lambdaReeval){
					ranks[ranksNew[i].value].second = i;
				}
			}
			//compute noise estimate based on rank changes
			//s is our current noise estimate
			double s = 0;
			for(std::size_t i = 0; i != lambdaReeval; ++i){
				//measured rank difference(number of ranks between old and new point
				//we use ranks in-between because if both estimates have a very close estimate,
				//they will be placed directly next to each other -> 0 ranks in-between.
				s += std::abs(ranks[i].first-ranks[i].second) - 1;
				//minus a percentile of the expected difference on a truely random function
				//as the first and second rank are interchangeable, the average is computed
				s -= 0.5 * deltaLim(ranks[i].second - (ranks[i].second > ranks[i].first), 2*m_lambda-1,m_rankChangeQuantile);
				s -= 0.5 * deltaLim(ranks[i].first - (ranks[i].first > ranks[i].second), 2*m_lambda-1,m_rankChangeQuantile);
			}		
			s /= lambdaReeval;
			//simple adaptation of the number of reevaluations
			if(s > 0){
				double rawIncrease = (m_numEvalIncreaseFactor - 1.0 )* m_numEvaluations;
				m_numEvaluations += std::max<std::size_t>(1,std::lround(rawIncrease));
			}else if (s < 0 && m_numEvaluations > 1){
				double rawDecrease = (1.0 - 1.0/m_numEvalIncreaseFactor) * m_numEvaluations;
				m_numEvaluations -= std::max<std::size_t>(1,std::lround(rawDecrease));
			}
		}
	}
	updatePopulation(offspring);
}

