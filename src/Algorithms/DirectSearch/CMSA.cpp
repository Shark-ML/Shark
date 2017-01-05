/*!
 * 
 *
 * \brief       Implements the CMSA.
 * 
 * The algorithm is described in
 * 
 * H. G. Beyer, B. Sendhoff (2008). 
 * Covariance Matrix Adaptation Revisited: The CMSA Evolution Strategy
 * In Proceedings of the Tenth International Conference on Parallel Problem Solving from Nature
 * (PPSN X), pp. 123-132, LNCS, Springer-Verlag
 * 
 *
 * \author      -
 * \date        -
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
#include <shark/Algorithms/DirectSearch/CMSA.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/ElitistSelection.h>
using namespace shark;

void CMSA::init( ObjectiveFunctionType & function, SearchPointType const& p) {
	SIZE_CHECK(p.size() == function.numberOfVariables());
	checkFeatures(function);
	std::vector<RealVector> points(1,p);
	std::vector<double> functionValues(1,function.eval(p));
	
	std::size_t lambda = m_userSetLambda? m_lambda:4 * p.size();
	std::size_t mu  = m_userSetMu? m_mu:lambda / 4;
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
void CMSA::init( 
	ObjectiveFunctionType& function, 
	SearchPointType const& p,
	std::size_t lambda,
	std::size_t mu,
	double initialSigma,				       
	const boost::optional< RealMatrix > & initialCovarianceMatrix
) {
	SIZE_CHECK(p.size() == function.numberOfVariables());
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

void CMSA::doInit( 
	std::vector<SearchPointType> const& initialSearchPoints,
	std::vector<ResultType> const& initialValues,
	std::size_t lambda,
	std::size_t mu,
	double sima
) {
	SIZE_CHECK(initialSearchPoints.size() > 0);
	m_numberOfVariables = initialSearchPoints[0].size();

	m_lambda = lambda;
	m_mu = mu;

	m_mutationDistribution.resize( m_numberOfVariables );
	m_sigma =  (m_initSigma == 0)? 1.0/std::sqrt(double(m_numberOfVariables)): m_initSigma;
	m_cSigma = 1./::sqrt( 2. * m_numberOfVariables );
	m_cC = 1. + (m_numberOfVariables*(m_numberOfVariables + 1.))/(2.*m_mu);
	
	std::size_t pos = std::min_element(initialValues.begin(),initialValues.end())-initialValues.begin();
	m_mean = initialSearchPoints[pos];
	m_best.point = initialSearchPoints[pos];
	m_best.value = initialValues[pos];
}


void CMSA::step(ObjectiveFunctionType const& function){
	std::vector<IndividualType> offspring = generateOffspring();
	PenalizingEvaluator penalizingEvaluator;
	penalizingEvaluator( function, offspring.begin(), offspring.end() );
	updatePopulation(offspring);
}

std::vector<CMSA::IndividualType> CMSA::generateOffspring( ) const{
	std::vector< IndividualType > offspring( m_lambda );
	for( std::size_t i = 0; i < offspring.size(); i++ ) {		    
		MultiVariateNormalDistribution::result_type sample = m_mutationDistribution(*mpe_rng);
		offspring[i].chromosome().sigma = m_sigma * ::exp( m_cSigma * gauss(*mpe_rng, 0, 1 ) );
		offspring[i].chromosome().step = sample.first;
		offspring[i].searchPoint() = m_mean + offspring[i].chromosome().sigma * sample.first;
	}
	return offspring;
}
void CMSA::updatePopulation(std::vector< IndividualType > const& offspring ) {
	std::vector< IndividualType > selectedOffspring( m_mu );
	ElitistSelection< IndividualType::FitnessOrdering > selection;
	selection(offspring.begin(),offspring.end(),selectedOffspring.begin(), selectedOffspring.end());
	
	RealVector xPrimeNew ( m_numberOfVariables, 0. );
	for( auto const& ind : selectedOffspring )
		noalias(xPrimeNew) += ind.searchPoint() / m_mu;
	
	// Covariance Matrix Update
	RealMatrix Znew( m_numberOfVariables, m_numberOfVariables,0.0 );
	RealMatrix& C = m_mutationDistribution.covarianceMatrix();
	// Rank-mu-Update
	for( std::size_t i = 0; i < m_mu; i++ ) {
		noalias(Znew) += 1./m_mu * blas::outer_prod( 
			selectedOffspring[i].chromosome().step,
			selectedOffspring[i].chromosome().step
		);
	}
	noalias(C) = (1. - 1./m_cC) * C + 1./m_cC * Znew;
	m_mutationDistribution.update();

	// Step size update
	double sigmaNew = 0.;
	//double sigma = 0.;
	for( std::size_t i = 0; i < m_mu; i++ ) {
		sigmaNew += 1./m_mu * selectedOffspring[i].chromosome().sigma;
	}
	m_sigma = sigmaNew;
	m_mean = xPrimeNew;
	m_best.point= selectedOffspring[ 0 ].searchPoint();
	m_best.value= selectedOffspring[ 0 ].unpenalizedFitness();
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
