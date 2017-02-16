//===========================================================================
/*!
 * 
 *
 * \brief       Implements the MOEA/D algorithm.
 * 
 * Q. Zhang and H. Li, “MOEA/D: a multi-objective evolutionary algorithm based
 * on decomposition,” IEEE Transactions on Evolutionary Computation, vol. 11,
 * no. 6, pp. 712–731, 2007
 * DOI: 10.1109/TEVC.2007.892759
 * 
 *
 * \author      Bjørn Bugge Grathwohl
 * \date        February 2017
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
//===========================================================================

#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_MOEAD
#define SHARK_ALGORITHMS_DIRECT_SEARCH_MOEAD

#include <shark/Algorithms/AbstractMultiObjectiveOptimizer.h>
#include <shark/Algorithms/DirectSearch/Individual.h>
#include <shark/Algorithms/DirectSearch/Operators/Recombination/SimulatedBinaryCrossover.h>
#include <shark/Algorithms/DirectSearch/Operators/Mutation/PolynomialMutation.h>
#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/Algorithms/DirectSearch/Operators/Grid.h>

//#include <shark/LinAlg/BLAS/remora.hpp>


namespace shark {

namespace detail {


template <typename IndividualType>
double tchebycheff_scalarizer(IndividualType const & individual, 
                              RealVector const & weights, 
                              RealVector const & optimalPointFitness)
{
    SIZE_CHECK(individual.unpenalizedFitness().size() == optimalPointFitness.size());

    const RealVector & fitness = individual.unpenalizedFitness();
    const std::size_t num_objectives = fitness.size();
    double max_fun = -1.0e+30;
    for(std::size_t i = 0; i < num_objectives; ++i)
    {
        max_fun = std::max(max_fun, 
                           weights[i] * std::abs(fitness[i] - 
                                                 optimalPointFitness[i]));
    }
    return max_fun;
}



// init:
// 1: uniformly generate N weight vectors (uniform distr.)
// 2: calculate pairwise distances between weight vectors
// 3: for each weight vector, pick the T closest vectors and remember their indices.
// 4: with these indices, make B structure I -> [I] that maps an index of a weight vector to a list of the T closest weight indices.

} // namespace detail


/**
 * \brief Implements the MOEA/D.
 *
 * More doc...
 */
class MOEAD : public AbstractMultiObjectiveOptimizer<RealVector> 
{
public:
    typedef shark::Individual<RealVector, RealVector> IndividualType;

    MOEAD(DefaultRngType & rng = Rng::globalRng) : mpe_rng(&rng)
    {
        mu() = 100;
        crossoverProbability() = 0.9;
        nc() = 20.0; // parameter for crossover operator
        nm() = 20.0; // parameter for mutation operator 
        neighbourhoodSize() = 10;
        // FIXME: Can it?
        this->m_features |= 
            AbstractMultiObjectiveOptimizer<RealVector>::CAN_SOLVE_CONSTRAINED;
    }

    std::string name() const override
    {
        return "MOEA/D";
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
    
    // When asking me what is the mu, answer the actual value...
    std::size_t mu() const
    {
        return m_mu;
    }
    //... but when the user asks to change the mu, he can only request that the
    // actual mu is about that value.  The actual value depends on the mu' and
    // the dimensionality of the problem.
    std::size_t & mu()
    {
        //return m_mu;
        return m_desired_mu;
    }

    std::size_t neighbourhoodSize() const
    {
        return m_neighbourhoodSize;
    }

    std::size_t & neighbourhoodSize()
    {
        return m_neighbourhoodSize;
    }

    template <typename Archive>
    void serialize(Archive & archive) override
    {
        archive & BOOST_SERIALIZATION_NVP(m_crossoverProbability);
        archive & BOOST_SERIALIZATION_NVP(m_mu);
        archive & BOOST_SERIALIZATION_NVP(m_desired_mu);
        archive & BOOST_SERIALIZATION_NVP(m_mu_prime);
        archive & BOOST_SERIALIZATION_NVP(m_parents);
        archive & BOOST_SERIALIZATION_NVP(m_best);
        archive & BOOST_SERIALIZATION_NVP(m_weights);
        archive & BOOST_SERIALIZATION_NVP(m_neighbourhoods);
        archive & BOOST_SERIALIZATION_NVP(m_neighbourhoodSize);
        archive & BOOST_SERIALIZATION_NVP(m_bestDecomposedValues);
        archive & BOOST_SERIALIZATION_NVP(m_crossover);
        archive & BOOST_SERIALIZATION_NVP(m_mutation);
    }

    void init(ObjectiveFunctionType & function) override
    {
        checkFeatures(function);
        if(!function.canProposeStartingPoint())
        {
            throw SHARKEXCEPTION("[" + name() + "::init] Objective function " +
                                 "does not propose a starting point");
        }
        std::vector<SearchPointType> points(mu());
        for(std::size_t i = 0; i < mu(); ++i)
        {
            points[i] = function.proposeStartingPoint();
        }
        init(function, points);
    }
    
    void init(ObjectiveFunctionType & function, 
              std::vector<SearchPointType> const & initialSearchPoints)
    {
        checkFeatures(function);
        std::vector<RealVector> values(initialSearchPoints.size());
        for(std::size_t i = 0; i < initialSearchPoints.size(); ++i)
        {
            if(!function.isFeasible(initialSearchPoints[i]))
            {
				throw SHARKEXCEPTION("[" + name() + "::init] starting " +
                                     "point(s) not feasible");
            }

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
            throw SHARKEXCEPTION("[" + name() + "::init] Algorithm does " +
                                 "only allow box constraints");
        }
        doInit(initialSearchPoints, values, lowerBounds, 
               upperBounds, mu(), nm(), nc(), crossoverProbability(),
               neighbourhoodSize());
    }

    void step(ObjectiveFunctionType const & function) override
    {
        PenalizingEvaluator penalizingEvaluator;

        for(std::size_t i = 0; i < m_neighbourhoods.size1(); ++i)
        {
            // 2.1.
            IndividualType offspring = generateOffspring(i); // y in paper
            // 2.2. Apply a problem-specific repair/improvement heuristic on y
            // to make y'
            m_repairFunction(offspring); // See footnote on p 715
            // 2.3. Update z
            penalizingEvaluator(function, offspring);
            // TODO: Unpenalized or penalized fitness?
            RealVector candidate = offspring.unpenalizedFitness();
            for(std::size_t i = 0; i < candidate.size(); ++i)
            {
                m_bestDecomposedValues[i] = std::min(m_bestDecomposedValues[i], 
                                                     candidate[i]);
            }
            updatePopulation(i, offspring);
        }
    }
    
protected:

    void doInit(std::vector<SearchPointType> const & initialSearchPoints,
                std::vector<ResultType> const & functionValues,
                RealVector const & lowerBounds,
                RealVector const & upperBounds,
                std::size_t const mu,
                double const nm,
                double const nc,
                double const crossover_prob,
                std::size_t const neighbourhoodSize)
    {
        SIZE_CHECK(initialSearchPoints.size() > 0);
        
        const std::size_t numOfObjectives = functionValues[0].size();
        // Decomposition-related initialization
        m_mu_prime = bestPointCountForLattice(numOfObjectives, desired_mu);
        m_weights = weightLattice(numOfObjectives, m_mu_prime);
        
        m_neighbourhoodSize = neighbourhoodSize;
        m_neighbourhoods = closestIndices(m_weights, 
                                          neighbourhoodSize);
        
        m_mu = m_weights.size1();
        m_mutation.m_nm = nm;
        m_crossover.m_nc = nc;
        m_crossoverProbability = crossover_prob;
		m_parents.resize(m_mu);
        m_best.resize(m_mu);
        // If the number of supplied points is smaller than mu, fill everything in
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
    IndividualType generateOffspring(std::size_t const i) const
    {
        // Below should be in its own "selector"...
        DiscreteUniform<> uniform_int_dist(*mpe_rng, 0, m_neighbourhoods.size2() - 1);
        // 1. Randomly select two indices k,l from B(i)
        const std::size_t k = m_neighbourhoods(i, uniform_int_dist());
        const std::size_t l = m_neighbourhoods(i, uniform_int_dist());
        //    Then generate a new solution y from x_k and x_l
        IndividualType x_k = m_parents[k];
        IndividualType x_l = m_parents[l];
        
        if(coinToss(*mpe_rng, m_crossoverProbability))
        {
            m_crossover(*mpe_rng, x_k, x_l);
        }
        m_mutation(*mpe_rng, x_k);
        return {x_k};
    }

    void updatePopulation(std::vector<IndividualType> const & offspringvec)
    {
        SIZE_CHECK(offspringvec.size() == 1);
        const IndividualType & offspring = offspringvec[0];
        // 2.3. Update the "Z" vector.
        RealVector candidate = offspring.unpenalizedFitness();
        for(std::size_t i = 0; i < candidate.size(); ++i)
        {
            m_bestDecomposedValues[i] = std::min(m_bestDecomposedValues[i], 
                                                 candidate[i]);
        }
        // 2.4. Update of neighbouring solutions
        for(auto iter = m_neighbourhoods.row_begin(m_curParentIndex);
            iter != m_neighbourhoods.row_end(m_curParentIndex);
        return x_k;
    }

    void updatePopulation(std::size_t const idx, 
                          IndividualType const & offspring)
    {
        // 2.4. Update of neighbouring solutions
        for(auto iter = m_neighbourhoods.row_begin(idx);
            iter != m_neighbourhoods.row_end(idx);
            ++iter)
        {
            std::size_t j = *iter;
            const RealVector lambda_j(m_weights.row_begin(j), 
                                      m_weights.row_end(j));
            IndividualType & x_j = m_parents[j];
            const RealVector & z = m_bestDecomposedValues;
            // if g^te(y' | lambda^j,z) <= g^te(x_j | lambda^j,z)
            using detail::tchebycheff_scalarizer;
            double tnew = tchebycheff_scalarizer<IndividualType>(offspring, lambda_j, z);
            double told = tchebycheff_scalarizer<IndividualType>(x_j, lambda_j, z);
            if(tnew <= told)
            {
                // then set x^j <- y'
                // and FV^j <- F(y')
                // This is done below, since the F-value is
                // contained in the offspring itself (the unpenalizedFitness)
                x_j = offspring;
                noalias(m_best[j].point) = x_j.searchPoint();
                m_best[j].value = x_j.unpenalizedFitness();
            }
        }
        // 2.5. Update of EP
        // This is not done in the authors' own implementation?
    }

    std::vector<IndividualType> m_parents;    

private:
    DefaultRngType * mpe_rng;
    double m_crossoverProbability; ///< Probability of crossover happening.
    std::vector<IndividualType> m_parents;
    std::size_t m_mu_prime; ///< mu' is the factor used for getting the actual mu.
    std::size_t m_mu; ///< Size of parent population and the "N" from the paper
    std::size_t m_desired_mu; ///< What the user asks for to be mu.


    std::size_t m_neighbourhoodSize; // The "T" from the paper
    RealMatrix m_weights; // All the lambdas from the paper
    UIntMatrix m_neighbourhoods; // size1: The "B" from the paper; size2: the "T"
    RealVector m_bestDecomposedValues; // The "z" from the paper

    SimulatedBinaryCrossover<SearchPointType> m_crossover;
    PolynomialMutator m_mutation;
    std::function<void(IndividualType &)> m_repairFunction;
};


} // namespace shark

#endif // SHARK_ALGORITHMS_DIRECT_SEARCH_MOEAD
