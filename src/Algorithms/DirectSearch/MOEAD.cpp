/*!
 *
 *
 * \brief		Implements the MOEA/D algorithm.
 *
 * \author		Bjoern Bugge Grathwohl
 * \date		February 2017
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#define SHARK_COMPILE_DLL
#include <shark/Algorithms/DirectSearch/MOEAD.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/Scalarizers/Tchebycheff.h>
using namespace shark;

MOEAD::MOEAD(random::rng_type & rng) : mpe_rng(&rng){
	mu() = 100;
	crossoverProbability() = 0.9;
	nc() = 20.0; // parameter for crossover operator
	nm() = 20.0; // parameter for mutation operator
	neighbourhoodSize() = 10;
	m_features |= CAN_SOLVE_CONSTRAINED;
}

void MOEAD::init(
	ObjectiveFunctionType const& function,
	std::vector<SearchPointType> const & initialSearchPoints
){
	checkFeatures(function);
	std::vector<RealVector> values(initialSearchPoints.size());
	for(std::size_t i = 0; i < initialSearchPoints.size(); ++i){
		SHARK_RUNTIME_CHECK(function.isFeasible(initialSearchPoints[i]), "Supplied points are not feasible");
		values[i] = function.eval(initialSearchPoints[i]);
	}
	std::size_t dim = function.numberOfVariables();
	RealVector lowerBounds(dim, -1e20);
	RealVector upperBounds(dim, 1e20);
	if(function.hasConstraintHandler()){
		SHARK_RUNTIME_CHECK(
			function.getConstraintHandler().isBoxConstrained(),
			"Algorithm does only allow box constraints"
		);
		typedef BoxConstraintHandler<SearchPointType> ConstraintHandler;
		ConstraintHandler const & handler =
			static_cast<ConstraintHandler const &>(
				function.getConstraintHandler());
		lowerBounds = handler.lower();
		upperBounds = handler.upper();
	}
	doInit(
		initialSearchPoints, values, lowerBounds,
		upperBounds, mu(), nm(), nc(), crossoverProbability(),
		neighbourhoodSize()
	);
}

void MOEAD::step(ObjectiveFunctionType const & function){
	PenalizingEvaluator penalizingEvaluator;
	// y in paper
	std::vector<IndividualType> offspring = generateOffspring();
	// Evaluate the objective function on our new candidate
	penalizingEvaluator(function, offspring[0]);
	updatePopulation(offspring);
}


void MOEAD::doInit(
	std::vector<SearchPointType> const & initialSearchPoints,
	std::vector<ResultType> const & functionValues,
	RealVector const & lowerBounds,
	RealVector const & upperBounds,
	std::size_t const mu,
	double const nm,
	double const nc,
	double const crossover_prob,
	std::size_t const neighbourhoodSize,
	std::vector<Preference> const & weightVectorPreferences
){
	SIZE_CHECK(initialSearchPoints.size() > 0);

	m_curParentIndex = 0;
	const std::size_t numOfObjectives = functionValues[0].size();
	// Decomposition-related initialization
	std::size_t numLatticeTicks = computeOptimalLatticeTicks(numOfObjectives, mu);
	if(weightVectorPreferences.empty())
	{
		m_weights = sampleLatticeUniformly(*mpe_rng,
		                                   weightLattice(numOfObjectives, 
		                                                 numLatticeTicks),
		                                   mu);
	}
	else
	{
		m_weights = preferenceAdjustedWeightVectors(numOfObjectives, 
		                                            numLatticeTicks, 
		                                            weightVectorPreferences);
	}

	// m_weights.size1() will be equal to mu whenever no ROI points are given.
	// If the user supplied regions of interest, the number of weights will be
	// more than the supplied mu, and therefore the actual m_mu value that is
	// used in the algorithm will be set based on the size of the wight matrix.
	m_mu = m_weights.size1();
	m_neighbourhoodSize = neighbourhoodSize;
	m_neighbourhoods = computeClosestNeighbourIndicesOnLattice(
		m_weights, neighbourhoodSize
	);
	SIZE_CHECK(m_neighbourhoods.size1() == m_mu);
	m_mutation.m_nm = nm;
	m_crossover.m_nc = nc;
	m_crossoverProbability = crossover_prob;
	m_parents.resize(m_mu);
	m_best.resize(m_mu);
	// If the number of supplied points is smaller than mu, fill everything in
	std::size_t numPoints = 0;
	if(initialSearchPoints.size() <= m_mu){
		numPoints = initialSearchPoints.size();
		for(std::size_t i = 0; i < numPoints; ++i){
			m_parents[i].searchPoint() = initialSearchPoints[i];
			m_parents[i].penalizedFitness() = functionValues[i];
			m_parents[i].unpenalizedFitness() = functionValues[i];
		}
	}
	// Copy points randomly
	for(std::size_t i = numPoints; i < m_mu; ++i){
		std::size_t index = random::discrete(*mpe_rng, std::size_t(0), initialSearchPoints.size() - 1);
		m_parents[i].searchPoint() = initialSearchPoints[index];
		m_parents[i].penalizedFitness() = functionValues[index];
		m_parents[i].unpenalizedFitness() = functionValues[index];
	}
	m_bestDecomposedValues = RealVector(numOfObjectives, 1e30);
	// Create initial mu best points
	for(std::size_t i = 0; i < m_mu; ++i){
		m_best[i].point = m_parents[i].searchPoint();
		m_best[i].value = m_parents[i].unpenalizedFitness();
	}
	m_crossover.init(lowerBounds, upperBounds);
	m_mutation.init(lowerBounds, upperBounds);
}

// Make me an offspring...
std::vector<MOEAD::IndividualType> MOEAD::generateOffspring() const{
	// Below should be in its own "selector"...
	
	// 1. Randomly select two indices k,l from B(i)
	const std::size_t k = m_neighbourhoods(m_curParentIndex, random::discrete(*mpe_rng, std::size_t(0), m_neighbourhoods.size2() - 1));
	const std::size_t l = m_neighbourhoods(m_curParentIndex, random::discrete(*mpe_rng, std::size_t(0), m_neighbourhoods.size2() - 1));
	//    Then generate a new solution y from x_k and x_l
	IndividualType x_k = m_parents[k];
	IndividualType x_l = m_parents[l];

	if(random::coinToss(*mpe_rng, m_crossoverProbability)){
		m_crossover(*mpe_rng, x_k, x_l);
	}
	m_mutation(*mpe_rng, x_k);
	return {x_k};
}

void MOEAD::updatePopulation(std::vector<IndividualType> const & offspringvec){
	SIZE_CHECK(offspringvec.size() == 1);
	const IndividualType & offspring = offspringvec[0];
	// 2.3. Update the "Z" vector.
	RealVector candidate = offspring.unpenalizedFitness();
	for(std::size_t i = 0; i < candidate.size(); ++i){
		m_bestDecomposedValues[i] = std::min(
			m_bestDecomposedValues[i], candidate[i]
		);
	}
	// 2.4. Update of neighbouring solutions
	for(std::size_t j : row(m_neighbourhoods, m_curParentIndex)){
		auto lambda_j = row(m_weights, j);
		IndividualType & x_j = m_parents[j];
		RealVector const& z = m_bestDecomposedValues;
		// if g^te(y' | lambda^j,z) <= g^te(x_j | lambda^j,z)
		double tnew = tchebycheffScalarizer(
			offspring.unpenalizedFitness(), lambda_j, z
		);
		double told = tchebycheffScalarizer(
			x_j.unpenalizedFitness(), lambda_j, z
		);
		if(tnew <= told){
			// then set x^j <- y'
			// and FV^j <- F(y')
			// This is done below, since the F-value is
			// contained in the offspring itself (the unpenalizedFitness)
			x_j = offspring;
			noalias(m_best[j].point) = x_j.searchPoint();
			m_best[j].value = x_j.unpenalizedFitness();
		}
	}
	// Finally, advance the parent index counter.
	m_curParentIndex = (m_curParentIndex + 1) % m_neighbourhoods.size1();
}
