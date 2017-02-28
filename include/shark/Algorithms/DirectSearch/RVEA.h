// To be the RVEA algorithm.
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_RVEA
#define SHARK_ALGORITHMS_DIRECT_SEARCH_RVEA

#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/Operators/ReferenceVectorAdaptation.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/ReferenceVectorGuidedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/TournamentSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>



namespace shark {

class RVEA : public AbstractMultiObjectiveOptimizer<RealVector>
{
public:
	typedef shark::Individual<RealVector, RealVector> IndividualType;

	RVEA(DefaultRngType & rng = Rng::globalRng) : mpe_rng(&rng)
	{
		mu() = 100;
		crossoverProbability() = 0.9;
		nc() = 20.0; // parameter for crossover operator
		nm() = 20.0; // parameter for mutation operator
		alpha() = 2.0; // parameter for reference vector selection
		fr() = 0.2;
		// Don't do anything...
		this->m_features |= CAN_SOLVE_CONSTRAINED;
	}

	std::string name() const override
	{
		return "RVEA";
	}

	double crossoverProbability() const
	{
		return m_crossoverProbability;
	}

	double & crossoverProbability()
	{
		return m_crossoverProbability;
	}

	double nm() const
	{
		return m_mutation.m_nm;
	}

	double & nm()
	{
		return m_mutation.m_nm;
	}

	double nc() const
	{
		return m_crossover.m_nc;
	}

	double & nc()
	{
		return m_crossover.m_nc;
	}

	double alpha() const
	{
		return m_alpha;
	}

	double & alpha()
	{
		return m_alpha;
	}

	double fr() const
	{
		return m_fr;
	}

	double & fr()
	{
		return m_fr;
	}

	std::size_t mu() const
	{
		return m_mu;
	}

	std::size_t & mu()
	{
		return m_mu;
	}

	RealMatrix referenceVectors() const
	{
		return m_unitReferenceVectors;
	}

	RealMatrix & referenceVectors()
	{
		return m_unitReferenceVectors;
	}

	std::size_t maxIterations() const
	{
		return m_maxIterations;
	}

	std::size_t & maxIterations()
	{
		return m_maxIterations;
	}

	template <typename Archive>
	void serialize(Archive & archive)
	{
		archive & BOOST_SERIALIZATION_NVP(m_crossoverProbability);
		archive & BOOST_SERIALIZATION_NVP(m_mu);
		archive & BOOST_SERIALIZATION_NVP(m_parents);
		archive & BOOST_SERIALIZATION_NVP(m_best);
		archive & BOOST_SERIALIZATION_NVP(m_bestDecomposedValues);
		archive & BOOST_SERIALIZATION_NVP(m_crossover);
		archive & BOOST_SERIALIZATION_NVP(m_mutation);
	}

	void init(ObjectiveFunctionType & function) override
	{
		checkFeatures(function);
		SHARK_RUNTIME_CHECK(function.canProposeStartingPoint(),
		                    "[" + name() + "::init] Objective function " +
		                    "does not propose a starting point");
		std::vector<SearchPointType> points(mu());
		for(std::size_t i = 0; i < mu(); ++i)
		{
			points[i] = function.proposeStartingPoint();
		}
		init(function, points);
	}

	void init(ObjectiveFunctionType & function,
			  std::vector<SearchPointType> const & initialSearchPoints) override
	{
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
		       upperBounds, mu(), nm(), nc(), 
		       crossoverProbability(), alpha(), fr(), 0);
	}

	void step(ObjectiveFunctionType const & function) override
	{
		PenalizingEvaluator penalizingEvaluator;
		std::vector<IndividualType> offspring = generateOffspring();
		penalizingEvaluator(function, offspring[0]);
		updatePopulation(offspring);
	}

protected:

	void doInit(
		std::vector<SearchPointType> const & initialSearchPoints,
		std::vector<ResultType> const & functionValues,
		RealVector const & lowerBounds,
		RealVector const & upperBounds,
		std::size_t const mu,
		double const nm,
		double const nc,
		double const crossover_prob,
		double const alpha,
		double const fr,
		std::size_t const max_iterations)
	{
		SIZE_CHECK(initialSearchPoints.size() > 0);

		const std::size_t numOfObjectives = functionValues[0].size();

		// Set the reference vectors
		std::size_t c = computeOptimalLatticeTicks(numOfObjectives, mu);
		m_referenceVectors = unitVectorsOnLattice(numOfObjectives, c);
		m_initialReferenceVectors = m_referenceVectors;

		m_curIteration = 0;
		m_maxIterations = max_iterations;
		m_mu = mu;
		m_mutation.m_nm = nm;
		m_crossover.m_nc = nc;
		m_crossoverProbability = crossover_prob;
		m_alpha = alpha;
		m_fr = fr;
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
			std::size_t index = discrete(*mpe_rng, 0,
			                             initialSearchPoints.size() - 1);
			m_parents[i].searchPoint() = initialSearchPoints[index];
			m_parents[i].penalizedFitness() = functionValues[index];
			m_parents[i].unpenalizedFitness() = functionValues[index];
		}
		m_bestDecomposedValues = RealVector(numOfObjectives, 1e30);
		// Create initial mu best points
		for(std::size_t i = 0; i < m_mu; ++i)
		{
			m_best[i].point = m_parents[i].searchPoint();
			m_best[i].value = m_parents[i].unpenalizedFitness();
		}
		m_crossover.init(lowerBounds, upperBounds);
		m_mutation.init(lowerBounds, upperBounds);
	}

	// Make me an offspring...
	std::vector<IndividualType> generateOffspring() const
	{
		TournamentSelection<IndividualType::RankOrdering> selection;
		std::vector<IndividualType> offspring(mu());
		selection(*mpe_rng,
		          m_parents.begin(), m_parents.end(),
		          offspring.begin(), offspring.end());

		for(std::size_t i = 0; i < mu() - 1; i += 2)
		{
			if(coinToss(*mpe_rng, m_crossoverProbability))
			{
				m_crossover(*mpe_rng, offspring[i], offspring[i + 1]);
			}
		}
		for(std::size_t i = 0; i < mu(); ++i)
		{
			m_mutation(*mpe_rng, offspring[i]);
		}
		return offspring;
	}

	void updatePopulation(std::vector<IndividualType> const & offspringvec)
	{
		SIZE_CHECK(m_maxIterations > 0);
		m_parents.insert(m_parents.end(), offspringvec.begin(),
		                 offspringvec.end());
		ReferenceVectorGuidedSelection selection;
		selection(alpha(), m_parents, 
		          m_referenceVectors, m_curIteration, m_maxIteration);
		std::partition(m_parents.begin(), 
		               m_parents.end(), 
		               IndividualType::IsSelected);
		m_parents.erase(m_parents.begin() + mu(), m_parents.end());

		for(std::size_t i = 0; i < mu(); ++i)
		{
			noalias(m_best[i].point) = m_parents[i].searchPoint();
			m_best[i].value = m_parents[i].unpenalizedFitness();
		}

		ReferenceVectorAdaptation adapt;
		adapt(m_fr, m_parents, m_referenceVectors, 
		      m_initialReferenceVectors, m_curIteration, m_maxIteration);

		++m_curIteration;
	}

	std::vector<IndividualType> m_parents;

private:
	DefaultRngType * mpe_rng;
	double m_crossoverProbability; ///< Probability of crossover happening.
	std::size_t m_mu; ///< Size of parent population and the "N" from the paper
	SimulatedBinaryCrossover<SearchPointType> m_crossover;
	PolynomialMutator m_mutation;
	double m_alpha;
	double m_fr; // hyperparameter for reference vector adaptation.
	std::size_t m_curIteration;
	std::size_t m_maxIterations;
	
	RealMatrix m_initialReferenceVectors;
	RealMatrix m_referenceVectors;
};

} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_RVEA
