#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_RVEA
#define SHARK_ALGORITHMS_DIRECT_SEARCH_RVEA

#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/Operators/ReferenceVectorAdaptation.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/ReferenceVectorGuidedSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Selection/TournamentSelection.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/Lattice.h>


namespace shark {

class RVEA : public AbstractMultiObjectiveOptimizer<RealVector>
{
public:
	typedef shark::Individual<RealVector, RealVector> IndividualType;

	RVEA(DefaultRngType & rng = Rng::globalRng) : m_rng(&rng)
	{
		approxMu() = 100;
		m_mu = approxMu();
		crossoverProbability() = 0.9;
		nc() = 20.0; // parameter for crossover operator
		nm() = 20.0; // parameter for mutation operator
		alpha() = 2.0; // parameter for reference vector selection
		fr() = 0.1;
		maxIterations() = 0; // must be set by user
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

	// std::size_t & mu()
	// {
	// 	return m_mu;
	// }

	std::size_t approxMu() const
	{
		return m_approxMu;
	}

	std::size_t & approxMu()
	{
		return m_approxMu;
	}

	RealMatrix referenceVectors() const
	{
		return m_referenceVectors;
	}

	RealMatrix & referenceVectors()
	{
		return m_referenceVectors;
	}
	
	RealMatrix initialReferenceVectors() const
	{
		return m_initialReferenceVectors;
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
#define S(var) archive & BOOST_SERIALIZATION_NVP(var)
		S(m_crossoverProbability);
		S(m_mu);
		S(m_parents);
		S(m_best);
		S(m_crossover);
		S(m_mutation);
		S(m_alpha);
		S(m_fr);
		S(m_curIteration);
		S(m_maxIterations);
		S(m_initialReferenceVectors);
		S(m_referenceVectors);
#undef S
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
		       upperBounds, approxMu(), nm(), nc(), 
		       crossoverProbability(), alpha(), fr(), maxIterations());
	}

	void step(ObjectiveFunctionType const & function) override
	{
		PenalizingEvaluator penalizingEvaluator;
		std::vector<IndividualType> offspring = generateOffspring();
		penalizingEvaluator(function, offspring.begin(), offspring.end());
		updatePopulation(offspring);
	}

protected:

	void doInit(
		std::vector<SearchPointType> const & initialSearchPoints,
		std::vector<ResultType> const & functionValues,
		RealVector const & lowerBounds,
		RealVector const & upperBounds,
		std::size_t const approx_mu,
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
		std::size_t ticks = computeOptimalLatticeTicks(numOfObjectives, approx_mu);
		m_referenceVectors = unitVectorsOnLattice(numOfObjectives, ticks);
		m_initialReferenceVectors = m_referenceVectors;
		// m_additionalReferenceVectors = m_referenceVectors;
		m_referenceVectorMinAngles = RealVector(m_referenceVectors.size1());
		updateReferenceVectorMinAngles();

		m_mu = m_referenceVectors.size1();
		m_curIteration = 0;
		m_maxIterations = max_iterations;
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
			std::size_t index = discrete(*m_rng, 0,
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

	std::vector<IndividualType> generateOffspring() const
	{
		SHARK_RUNTIME_CHECK(m_maxIterations > 0, 
		                    "Maximum number of iterations not set.");
		TournamentSelection<IndividualType::RankOrdering> selection;
		std::vector<IndividualType> offspring(mu());
		selection(*m_rng,
		          m_parents.begin(), m_parents.end(),
		          offspring.begin(), offspring.end());

		for(std::size_t i = 0; i < mu() - 1; i += 2)
		{
			if(coinToss(*m_rng, m_crossoverProbability))
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

	void updatePopulation(std::vector<IndividualType> const & offspringvec)
	{
		m_parents.insert(m_parents.end(), offspringvec.begin(),
		                 offspringvec.end());
		ReferenceVectorGuidedSelection selection;
		selection(alpha(), m_parents, 
		          m_referenceVectors,
		          m_referenceVectorMinAngles,
		          m_curIteration + 1, m_maxIterations);

		std::partition(m_parents.begin(), 
		               m_parents.end(), 
		               IndividualType::IsSelected);
		m_parents.erase(m_parents.begin() + mu(), m_parents.end());

		for(std::size_t i = 0; i < mu(); ++i)
		{
			noalias(m_best[i].point) = m_parents[i].searchPoint();
			m_best[i].value = m_parents[i].unpenalizedFitness();
		}

		if(false){
			std::ofstream file("fitness_" + std::to_string(m_curIteration) + ".dat");
			for(auto & p : m_parents)
			{
				for(auto & x : p.unpenalizedFitness())
				{
					file << x << " ";
				}
				file << "\n";
			}
			std::ofstream reffile("refvecs_" + std::to_string(m_curIteration) + ".dat");
			RealMatrix m = m_referenceVectors;
			for(std::size_t i = 0; i < m.size1(); ++i)
			{
				for(std::size_t j = 0; j < m.size2(); ++j)
				{
					reffile << m(i, j) << " ";
				}
				reffile << "\n";
			}
		}
		std::size_t x = std::ceil(m_fr * m_maxIterations);
		if(m_curIteration % x == 0)
		{
			referenceVectorAdaptation(
				m_parents, m_referenceVectors, m_initialReferenceVectors);
			updateReferenceVectorMinAngles();
		}

		++m_curIteration;
	}

	void updateReferenceVectorMinAngles()
	{
		const std::size_t s = m_referenceVectors.size1();
		RealMatrix m = acos(prod(m_referenceVectors, trans(m_referenceVectors))) +
			to_diagonal(RealVector(s, 1e10));
		for(std::size_t i = 0; i < s; ++i)
		{
			m_referenceVectorMinAngles[i] = min(row(m, i));
		}
	}

	std::vector<IndividualType> m_parents;

private:
	DefaultRngType * m_rng;
	double m_crossoverProbability; ///< Probability of crossover happening.
	std::size_t m_mu; ///< Size of parent population and number of reference vectors
	std::size_t m_approxMu; ///< The "approximate" value of mu that the user asked for.
	SimulatedBinaryCrossover<SearchPointType> m_crossover;
	PolynomialMutator m_mutation;
	double m_alpha;
	double m_fr; // hyperparameter for reference vector adaptation.
	std::size_t m_curIteration;
	std::size_t m_maxIterations;
	
	RealMatrix m_initialReferenceVectors;
	RealMatrix m_referenceVectors;
//	RealMatrix m_additionalReferenceVectors;
	RealVector m_referenceVectorMinAngles;
};

} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_RVEA
