/*!
 * 
 *
 * \brief       TypedIndividual

 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_TYPED_INDIVIDUAL_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_TYPED_INDIVIDUAL_H

#include <shark/Algorithms/DirectSearch/EA.h>
#include <shark/Algorithms/DirectSearch/CMA/Chromosome.h>

#include <shark/LinAlg/Base.h>
#include <vector>

namespace shark {

class CMAIndividual{
public:

	typedef RealVector FitnessType;
	typedef SearchSpaceType search_point_type;
	
	// Functors to use for the stl algorithms
	///\brief returns true if the individual is selected for the next parent set
	static bool IsSelected(this_type const& individual){
		return individual.selected();
	}
	
	///\brief Ordering relation by the ranks of the individuals
	static bool RankOrdering(this_type const& individual1, this_type const& individual2){
		return individual1.rank() < individual2.rank();
	}

	/**
	 * \brief Default constructor that initializes the individual's attributes to default values.
	 */
	CMAIndividual()
	: m_age(0)
	, m_rank(0)
	, m_selected(false){}
		
		
	CMAIndividual(
		std::size_t searchSpaceDimension,
		std::size_t numberOfObjectives,
		double successThreshold,
		double initialStepSize
	)
	: m_searchPoint(searchSpaceDimension)
	, m_chromosome(searchSpaceDimnsion, successThreshold, initialStepSize)
	, m_age(0)
	, m_rank(0)
	, m_selected(false)
	, m_penalizedFitness(numberOfObjectives)
	, m_unpenalizedFitness(numberOfObjectives){}
	
	void update(){
		m_chromosome.update();
	}
	void mutate(){
		MultiVariateNormalDistribution::ResultType sample = m_chromosome.m_mutationDistribution();
		m_chromosome.m_lastStep = sample.first;
		m_searchPoint += m_chromosome.m_stepSize * sample.first;
		m_chromosome.m_nedsCovarianceUpdate = true;
	}

	/**
	 * \brief Returns a non-const reference to the search point that is associated with the individual.
	 */
	SearchSpaceType &searchPoint() {
		return m_searchPoint;
	}

	/**
	 * \brief Returns a const reference to the search point that is associated with the individual.
	 */
	const SearchSpaceType &searchPoint() const {
		return m_searchPoint;
	}

	/**
	 * \brief Returns the number of objectives.
	 */
	unsigned int noObjectives() const {
		return m_unpenalizedFitness.size();
	}

	/**
	 * \brief Adjusts the number of objectives
	 */
	void setNoObjectives(unsigned int noObjectives) {
		m_penalizedFitness.resize(noObjectives);
		m_unpenalizedFitness.resize(noObjectives);
	}

	/**
	 * \brief Returns the age of the individual (in generations).
	 */
	unsigned int age() const {
		return m_age;
	}

	/**
	 * \brief Returns a reference to the age of the individual (in generations).
	 * Allows for lvalue()-semantics.
	 */
	unsigned int &age() {
		return m_age;
	}

	/**
	 * \brief Returns a non-const reference to the unpenalized fitness of the individual. Allows for lvalue()-semantics.
	 *
	 * Please note that the vector of fitness values is of size 1 in the case of single-objective
	 * optimization.
	 */
	FitnessType &fitness(tag::UnpenalizedFitness fitness) {
		return m_unpenalizedFitness);
	}

	/*!
	 * \brief Returns a const reference to the unpenalized fitness of the individual.
	 *
	 * Please note that the vector of fitness values is of size 1 in the case of single-objective
	 * optimization.
	 */
	const FitnessType &fitness(tag::UnpenalizedFitness fitness) const {
		return m_unpenalizedFitness);
	}

	/**
	 * \brief Returns a non-const reference to the penalized fitness of the individual. Allows for lvalue()-semantics.
	 *
	 * Please note that the vector of fitness values is of size 1 in the case of single-objective
	 * optimization. For further information on the difference between penalized and unpenalized fitness, please
	 * refer to the documentation of the respective tags.
	 */
	FitnessType &fitness(tag::PenalizedFitness fitness) {
		return m_penalizedFitness);
	}

	/**
	 * \brief Returns a const reference to the penalized fitness of the individual.
	 *
	 * Please note that the vector of fitness values is of size 1 in the case of single-objective
	 * optimization. For further information on the difference between penalized and unpenalized fitness, please
	 * refer to the documentation of the respective tags.
	 */
	const FitnessType &fitness(tag::PenalizedFitness fitness) const {
		return m_penalizedFitness);
	}

	/**
	 * \brief Returns the level of non-dominance of the individual.
	 */
	unsigned int rank() const {
		return m_rank;
	}

	/**
	 * \brief Returns a reference to the level of non-dominance of the individual. Allows for lvalue()-semantic.
	 */
	unsigned int &rank() {
		return m_rank;
	}

	/**
	 * \brief Returns true if the individual is selected for the next parent generation 
	 */
	bool selected() const {
		return m_selected;
	}

	/**
	 * \brief Returns true if the individual is selected for the next parent generation 
	 */
	bool &selected() {
		return m_selected;
	}

	/**
	 * \brief Stores the individual and all of its chromosomes in an archive.
	 */
	template<typename Archive>
	void serialize(Archive &archive, const unsigned int version) {
		archive &BOOST_SERIALIZATION_NVP(m_searchPoint);
		archive &BOOST_SERIALIZATION_NVP(m_age);
		archive &BOOST_SERIALIZATION_NVP(m_rank);
		archive &BOOST_SERIALIZATION_NVP(m_selected);
		archive &BOOST_SERIALIZATION_NVP(m_penalizedFitness);
		archive &BOOST_SERIALIZATION_NVP(m_unpenalizedFitness);
	}

protected:
	SearchSpaceType m_searchPoint; ///< The search point associated with the individual.
	CMAChromosome m_chromosome; ///< The chromosome of the strategy parameters
	unsigned int m_age;	///< The age of the individual (in generations).
	unsigned int m_rank; ///< The level of non-dominance of the individual. The lower the better.
	bool m_selected; ///< Is the individual selcted for the next parent set?

	FitnessType m_penalizedFitness; ///< Penalized fitness of the individual.
	FitnessType m_unpenalizedFitness; ///< Unpenalized fitness of the individual.

};

}
#endif
