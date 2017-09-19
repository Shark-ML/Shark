//===========================================================================
/*!
 *
 *
 * \brief		Implements the RVEA algorithm.
 *
 * \author		Bjoern Bugge Grathwohl
 * \date		March 2017
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
//===========================================================================

#define SHARK_COMPILE_DLL
#include <shark/Algorithms/DirectSearch/RVEA.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/TournamentSelection.h>

using namespace shark;

RVEA::RVEA(random::rng_type & rng) : m_rng(&rng){
	approxMu() = 100;
	m_mu = approxMu();
	crossoverProbability() = 0.9;
	nc() = 20.0; // parameter for crossover operator
	nm() = 20.0; // parameter for mutation operator
	alpha() = 2.0; // parameter for reference vector selection
	adaptationFrequency() = 0.1;
	maxIterations() = 0; // must be set by user!
	this->m_features |= CAN_SOLVE_CONSTRAINED;
}

void RVEA::init(
	ObjectiveFunctionType const& function,
	std::vector<SearchPointType> const & initialSearchPoints){

	checkFeatures(function);
	std::vector<RealVector> values(initialSearchPoints.size());
	for(std::size_t i = 0; i < initialSearchPoints.size(); ++i)
	{
		SHARK_RUNTIME_CHECK(function.isFeasible(initialSearchPoints[i]),
		                    "[" + name() + "::init] starting " +
		                    "point(s) not feasible");
		values[i] = function.eval(initialSearchPoints[i]);
	}
	std::size_t dim = function.numberOfVariables();
	RealVector lowerBounds(dim, -1e20);
	RealVector upperBounds(dim, 1e20);
	if(function.hasConstraintHandler() &&
	   function.getConstraintHandler().isBoxConstrained())
	{
		typedef BoxConstraintHandler<SearchPointType> ConstraintHandler;
		ConstraintHandler const & handler =
			static_cast<ConstraintHandler const &>(
				function.getConstraintHandler());
		lowerBounds = handler.lower();
		upperBounds = handler.upper();
	}
	else
	{
		SHARK_RUNTIME_CHECK(
			function.hasConstraintHandler() &&
			!function.getConstraintHandler().isBoxConstrained(),
			"[" + name() + "::init] Algorithm does " +
			"only allow box constraints"
			);
	}
	doInit(initialSearchPoints, values, lowerBounds,
	       upperBounds, approxMu(), nm(), nc(),
	       crossoverProbability(), alpha(), adaptationFrequency(),
	       maxIterations());
}

void RVEA::step(ObjectiveFunctionType const & function){
	PenalizingEvaluator penalizingEvaluator;
	std::vector<IndividualType> offspring = generateOffspring();
	penalizingEvaluator(function, offspring.begin(), offspring.end());
	updatePopulation(offspring);
}

std::size_t RVEA::suggestMu(std::size_t n, std::size_t const approx_mu){
	std::size_t t = computeOptimalLatticeTicks(n, approx_mu);
	return shark::detail::sumlength(n, t);
}


void RVEA::doInit(
	std::vector<SearchPointType> const & initialSearchPoints,
	std::vector<ResultType> const & functionValues,
	RealVector const & lowerBounds,
	RealVector const & upperBounds,
	std::size_t const approx_mu,
	double const nm,
	double const nc,
	double const crossover_prob,
	double const alph,
	double const fr,
	std::size_t const max_iterations,
	std::vector<Preference> const & referenceVectorPreferences){

	SIZE_CHECK(initialSearchPoints.size() > 0);

	const std::size_t numOfObjectives = functionValues[0].size();
	const std::size_t ticks = computeOptimalLatticeTicks(numOfObjectives, 
	                                                     approx_mu);

	if(referenceVectorPreferences.empty())
	{
       // The default reference vectors are sampled on the unit sphere.
		m_referenceVectors = unitVectorsOnLattice(numOfObjectives, ticks);
	}
	else
	{
		m_referenceVectors = preferenceAdjustedUnitVectors(
			numOfObjectives, ticks,
			referenceVectorPreferences);
	}
	// Set the reference vectors
	m_adaptation.m_initVecs = m_referenceVectors;
	m_referenceVectorMinAngles = RealVector(m_referenceVectors.size1());
	m_adaptation.updateAngles(m_referenceVectors, m_referenceVectorMinAngles);

	m_mu = m_referenceVectors.size1();
	SIZE_CHECK(m_mu == suggestMu(numOfObjectives, approx_mu));
	m_curIteration = 0;
	maxIterations() = max_iterations;
	m_mutation.m_nm = nm;
	m_crossover.m_nc = nc;
	m_crossoverProbability = crossover_prob;
	alpha() = alph;
	adaptationFrequency() = fr;
	m_parents.resize(m_mu);
	m_best.resize(m_mu);
	// If the number of supplied points is smaller than mu, fill everything
	// in
	std::size_t numPoints = 0;
	if(initialSearchPoints.size() <= m_mu)
	{
		numPoints = initialSearchPoints.size();
		for(std::size_t i = 0; i < numPoints; ++i)
		{
			m_parents[i].searchPoint() = initialSearchPoints[i];
			m_parents[i].penalizedFitness() = functionValues[i];
			m_parents[i].unpenalizedFitness() = functionValues[i];
		}
	}
	// Copy points randomly
	for(std::size_t i = numPoints; i < m_mu; ++i)
	{
		std::size_t index = random::discrete(*m_rng, std::size_t(0),
		                             initialSearchPoints.size() - 1);
		m_parents[i].searchPoint() = initialSearchPoints[index];
		m_parents[i].penalizedFitness() = functionValues[index];
		m_parents[i].unpenalizedFitness() = functionValues[index];
	}
	// Create initial mu best points
	for(std::size_t i = 0; i < m_mu; ++i)
	{
		m_best[i].point = m_parents[i].searchPoint();
		m_best[i].value = m_parents[i].unpenalizedFitness();
	}
	m_crossover.init(lowerBounds, upperBounds);
	m_mutation.init(lowerBounds, upperBounds);
}


std::vector<RVEA::IndividualType> RVEA::generateOffspring() const{
	SHARK_RUNTIME_CHECK(maxIterations() > 0,
	                    "Maximum number of iterations not set.");
	TournamentSelection<IndividualType::RankOrdering> selection;
	std::vector<IndividualType> offspring(mu());
	selection(*m_rng,
	          m_parents.begin(), m_parents.end(),
	          offspring.begin(), offspring.end());
	for(std::size_t i = 0; i < mu() - 1; i += 2)
	{
		if(random::coinToss(*m_rng, m_crossoverProbability))
		{
			m_crossover(*m_rng, offspring[i], offspring[i + 1]);
		}
	}
	for(std::size_t i = 0; i < mu(); ++i)
	{
		m_mutation(*m_rng, offspring[i]);
	}
	return offspring;
}

void RVEA::updatePopulation(std::vector<IndividualType> const & offspringvec){
	m_parents.insert(m_parents.end(), offspringvec.begin(),
	                 offspringvec.end());
	m_selection(m_parents,
	            m_referenceVectors,
	            m_referenceVectorMinAngles,
	            m_curIteration + 1);

	std::partition(m_parents.begin(),
	               m_parents.end(),
	               [](IndividualType const & ind){
		               return ind.selected();
	               });
	m_parents.erase(m_parents.begin() + mu(), m_parents.end());

	for(std::size_t i = 0; i < mu(); ++i)
	{
		noalias(m_best[i].point) = m_parents[i].searchPoint();
		m_best[i].value = m_parents[i].unpenalizedFitness();
	}

	if(willAdaptReferenceVectors())
	{
		m_adaptation(m_parents,
		             m_referenceVectors,
		             m_referenceVectorMinAngles);
	}
	++m_curIteration;
}
