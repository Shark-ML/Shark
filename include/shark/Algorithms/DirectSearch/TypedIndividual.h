/*!
 * 
 * \file        TypedIndividual.h
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
#include <shark/Algorithms/DirectSearch/Traits/FitnessTraits.h>

#include <shark/LinAlg/Base.h>

//#include <boost/flyweight.hpp>
#include <boost/tuple/tuple.hpp>

#include <vector>

namespace shark {

namespace detail {

/**
 * \brief Models an empty chromosome type.
 *
 * The chromosome concept is defined is follows:
 *	- A chromosome needs to be default constructable.
 *	- A chromosome needs to be copy-constructable.
 *	- A chromosome needs to be assignable.
 *	- The following expressions need be valid:
 *		- c.read( a );
 *		- c.write( a ):
 *		- c.serialize( a );
 *
 * Where c is an instance of type C that adheres to the chromosome
 * concept, a is an instance of type A that adheres to the archive concept.
 */
struct unused {

	template<typename Archive>
	void read(Archive &archive) { (void) archive; }

	template<typename Archive>
	void write(Archive &archive) const { (void) archive; }

	template<typename Archive>
	void serialize(Archive &archive, const unsigned int version) {
		(void) archive;
	}

};
}

/**
 * \brief Explicitly wraps up a chromosome index.
 */
template<unsigned int Index>
struct ChromosomeIndex {
	static const unsigned int INDEX = Index;
};

/**
 * \brief TypedIndividual is a templated class modelling
 * an individual that acts as a candidate solution in an evolutionary algorithm.
 *
 * The class TypedIndividual is a templated class modelling an individual that acts
 * as a candidate solution in an evolutionary algorithm. The class allows for modelling
 * arbitrary types of search spaces by means of the template parameter SearchSpaceType.
 * In addition, arbitrary types of chromosomes can be associated with an individual without the
 * need to implement an explicit interface.
 *
 * \code
 * struct Chromosome { double m_someValue; };
 *
 * TypedIndividual<RealVector,Chromosome> individual;
 * \endcode
 *
 * Defines an individual that carries a real vector as "search point" and that
 * uses one chromosome of type "Chromosome".
 *
 * \tparam SearchSpaceType The type of the search space.
 * \tparam ChromosomeType1 The first chromosome type, defaults to an empty type.
 * \tparam ChromosomeType2 The second chromosome type, defaults to an empty type.
 * \tparam ChromosomeType3 The third chromosome type, defaults to an empty type.
 * \tparam ChromosomeType4 The fourth chromosome type, defaults to an empty type.
 * \tparam ChromosomeType5 The fifth chromosome type, defaults to an empty type.
 * \tparam ChromosomeType6 The sixth chromosome type, defaults to an empty type.
 * \tparam ChromosomeType7 The seventh chromosome type, defaults to an empty type.
 * \tparam ChromosomeType8 The eigth chromosome type, defaults to an empty type.
 * \tparam ChromosomeType9 The ninth chromosome type, defaults to an empty type.
 * \tparam ChromosomeType10 The tenth chromosome type, defaults to an empty type.
 */
template<
typename SearchSpaceType,
         typename ChromosomeType1 = detail::unused,
         typename ChromosomeType2 = detail::unused,
         typename ChromosomeType3 = detail::unused,
         typename ChromosomeType4 = detail::unused,
         typename ChromosomeType5 = detail::unused,
         typename ChromosomeType6 = detail::unused,
         typename ChromosomeType7 = detail::unused,
         typename ChromosomeType8 = detail::unused,
         typename ChromosomeType9 = detail::unused,
         typename ChromosomeType10 = detail::unused
         > class TypedIndividual :
/** \cond */
	public boost::tuple<ChromosomeType1,
	ChromosomeType2,
	ChromosomeType3,
	ChromosomeType4,
	ChromosomeType5,
	ChromosomeType6,
	ChromosomeType7,
	ChromosomeType8,
	ChromosomeType9,
	ChromosomeType10>
/** \endcond */    {
public:
	typedef boost::tuple<ChromosomeType1,
	        ChromosomeType2,
	        ChromosomeType3,
	        ChromosomeType4,
	        ChromosomeType5,
	        ChromosomeType6,
	        ChromosomeType7,
	        ChromosomeType8,
	        ChromosomeType9,
	        ChromosomeType10> super;

	typedef TypedIndividual<
	SearchSpaceType,
	ChromosomeType1,
	ChromosomeType2,
	ChromosomeType3,
	ChromosomeType4,
	ChromosomeType5,
	ChromosomeType6,
	ChromosomeType7,
	ChromosomeType8,
	ChromosomeType9,
	ChromosomeType10
	> this_type;

	typedef RealVector FitnessType;

	typedef SearchSpaceType search_point_type;

	/**
	 * \brief Default constructor that initializes the individual's attributes to default values.
	 */
	TypedIndividual() : m_age(0),
		m_matingProbability(0.),
		m_selectionProbability(0.),
		m_rank(0),
		m_share(0) {
		setNoObjectives(1);
	}

	/**
	 * \brief Returns a non-const reference to the search point that is associated with the individual.
	 */
	inline SearchSpaceType &operator*() {
		return(m_searchPoint);
	}

	/**
	 * \brief Returns a const reference to the search point that is associated with the individual.
	 */
	inline const SearchSpaceType &operator*() const {
		return(m_searchPoint);
	}

	/**
	 * \brief Returns a non-const reference to the search point that is associated with the individual.
	 */
	inline SearchSpaceType &searchPoint() {
		return(m_searchPoint);
	}

	/**
	 * \brief Returns a const reference to the search point that is associated with the individual.
	 */
	inline const SearchSpaceType &searchPoint() const {
		return(m_searchPoint);
	}

	/**
	 * \brief Returns the number of objectives.
	 */
	inline unsigned int noObjectives() const {
		return(m_noObjectives);
	}

	/**
	 * \brief Adjusts the number of objectives and resizes the fitness vectors.
	 */
	inline void setNoObjectives(unsigned int noObjectives) {

		m_noObjectives = noObjectives;

		m_penalizedFitness.resize(noObjectives);
		m_unpenalizedFitness.resize(noObjectives);
		m_scaledFitness.resize(noObjectives);
	}

	/**
	 * \brief Returns the age of the individual (in generations).
	 */
	inline unsigned int age() const {
		return(m_age);
	}

	/**
	 * \brief Returns a reference to the age of the individual (in generations).
	 * Allows for lvalue()-semantics.
	 */
	inline unsigned int &age() {
		return(m_age);
	}

	/**
	 * \brief Adjusts the age of the individual (in generations).
	 * Allows for lvalue()-semantics.
	 */
	inline void setAge(unsigned int age) {
		m_age = age;
	}

	/**
	 * \brief Returns a non-const reference to the unpenalized fitness of the individual. Allows for lvalue()-semantics.
	 *
	 * Please note that the vector of fitness values is of size 1 in the case of single-objective
	 * optimization.
	 */
	inline FitnessType &fitness(tag::UnpenalizedFitness fitness) {
		return(m_unpenalizedFitness);
	}

	/*!
	 * \brief Returns a const reference to the unpenalized fitness of the individual.
	 *
	 * Please note that the vector of fitness values is of size 1 in the case of single-objective
	 * optimization.
	 */
	inline const FitnessType &fitness(tag::UnpenalizedFitness fitness) const {
		return(m_unpenalizedFitness);
	}

	/**
	 * \brief Returns a non-const reference to the penalized fitness of the individual. Allows for lvalue()-semantics.
	 *
	 * Please note that the vector of fitness values is of size 1 in the case of single-objective
	 * optimization. For further information on the difference between penalized and unpenalized fitness, please
	 * refer to the documentation of the respective tags.
	 */
	inline FitnessType &fitness(tag::PenalizedFitness fitness) {
		return(m_penalizedFitness);
	}

	/**
	 * \brief Returns a const reference to the penalized fitness of the individual.
	 *
	 * Please note that the vector of fitness values is of size 1 in the case of single-objective
	 * optimization. For further information on the difference between penalized and unpenalized fitness, please
	 * refer to the documentation of the respective tags.
	 */
	inline const FitnessType &fitness(tag::PenalizedFitness fitness) const {
		return(m_penalizedFitness);
	}

	/**
	 * \brief Returns a non-const reference to the scaled fitness of the individual. Allows for lvalue()-semantics.
	 *
	 * Please note that the vector of fitness values is of size 1 in the case of single-objective
	 * optimization.
	 */
	inline FitnessType &fitness(tag::ScaledFitness fitness) {
		return(m_scaledFitness);
	}

	/**
	 * \brief Returns a const reference to the penalized fitness of the individual.
	 *
	 * Please note that the vector of fitness values is of size 1 in the case of single-objective
	 * optimization.
	 */
	inline const FitnessType &fitness(tag::ScaledFitness fitness) const {
		return(m_scaledFitness);
	}


	/**
	 * \brief Returns a non-const reference to the mating probability of the individual.
	 */
	inline double &probability(tag::MatingProbability p) {
		return(m_matingProbability);
	}

	/**
	 * \brief Returns a const reference to the mating probability of the individual.
	 */
	inline const double &probability(tag::MatingProbability p) const {
		return(m_matingProbability);
	}

	/**
	 * \brief Returns a non-const reference to the selection probability of the individual.
	 */
	inline double &probability(tag::SelectionProbability p) {
		return(m_selectionProbability);
	}

	/**
	 * \brief Returns a const reference to the selection probability of the individual.
	 */
	inline const double &probability(tag::SelectionProbability p) const {
		return(m_selectionProbability);
	}

	/**
	 * \brief Returns the level of non-dominance of the individual.
	 */
	inline unsigned int rank() const {
		return(m_rank);
	}

	/**
	 * \brief Returns a reference to the level of non-dominance of the individual. Allows for lvalue()-semantic.
	 */
	inline unsigned int &rank() {
		return(m_rank);
	}

	/**
	 * \brief Returns the quality of an individual with respect to the Pareto-front it belongs to.
	 */
	inline double share() const {
		return(m_share);
	}

	/**
	 * \brief Returns a non-const reference to the quality of an individual with respect to the Pareto-front it belongs to.
	 */
	inline double &share() {
		return(m_share);
	}

	/**
	 * \brief Restores the individual and all of its chromosomes from the archive.
	 */
	template<typename Archive>
	void read(Archive &archive) {
		archive &BOOST_SERIALIZATION_NVP(m_searchPoint);
		archive &BOOST_SERIALIZATION_NVP(m_noObjectives);
		archive &BOOST_SERIALIZATION_NVP(m_age);
		archive &BOOST_SERIALIZATION_NVP(m_matingProbability);
		archive &BOOST_SERIALIZATION_NVP(m_selectionProbability);
		archive &BOOST_SERIALIZATION_NVP(m_penalizedFitness);
		archive &BOOST_SERIALIZATION_NVP(m_unpenalizedFitness);
		archive &BOOST_SERIALIZATION_NVP(m_rank);
		archive &BOOST_SERIALIZATION_NVP(m_share);

		archive &boost::serialization::make_nvp("chromosome_0", boost::get<0>(*this));
		archive &boost::serialization::make_nvp("chromosome_1", boost::get<1>(*this));
		archive &boost::serialization::make_nvp("chromosome_2", boost::get<2>(*this));
		archive &boost::serialization::make_nvp("chromosome_3", boost::get<3>(*this));
		archive &boost::serialization::make_nvp("chromosome_4", boost::get<4>(*this));
		archive &boost::serialization::make_nvp("chromosome_5", boost::get<5>(*this));
		archive &boost::serialization::make_nvp("chromosome_6", boost::get<6>(*this));
		archive &boost::serialization::make_nvp("chromosome_7", boost::get<7>(*this));
		archive &boost::serialization::make_nvp("chromosome_8", boost::get<8>(*this));
		archive &boost::serialization::make_nvp("chromosome_9", boost::get<9>(*this));
	}

	/**
	 * \brief Stores the individual and all of its chromosomes in an archive.
	 */
	template<typename Archive>
	void write(Archive &archive) const {
		archive &BOOST_SERIALIZATION_NVP(m_searchPoint);
		archive &BOOST_SERIALIZATION_NVP(m_noObjectives);
		archive &BOOST_SERIALIZATION_NVP(m_age);
		archive &BOOST_SERIALIZATION_NVP(m_matingProbability);
		archive &BOOST_SERIALIZATION_NVP(m_selectionProbability);
		archive &BOOST_SERIALIZATION_NVP(m_penalizedFitness);
		archive &BOOST_SERIALIZATION_NVP(m_unpenalizedFitness);
		archive &BOOST_SERIALIZATION_NVP(m_rank);
		archive &BOOST_SERIALIZATION_NVP(m_share);

		archive &boost::serialization::make_nvp("chromosome_0", boost::get<0>(*this));
		archive &boost::serialization::make_nvp("chromosome_1", boost::get<1>(*this));
		archive &boost::serialization::make_nvp("chromosome_2", boost::get<2>(*this));
		archive &boost::serialization::make_nvp("chromosome_3", boost::get<3>(*this));
		archive &boost::serialization::make_nvp("chromosome_4", boost::get<4>(*this));
		archive &boost::serialization::make_nvp("chromosome_5", boost::get<5>(*this));
		archive &boost::serialization::make_nvp("chromosome_6", boost::get<6>(*this));
		archive &boost::serialization::make_nvp("chromosome_7", boost::get<7>(*this));
		archive &boost::serialization::make_nvp("chromosome_8", boost::get<8>(*this));
		archive &boost::serialization::make_nvp("chromosome_9", boost::get<9>(*this));
	}

	/**
	 * \brief Stores the individual and all of its chromosomes in an archive.
	 */
	template<typename Archive>
	void serialize(Archive &archive, const unsigned int version) {
		archive &BOOST_SERIALIZATION_NVP(m_searchPoint);
		archive &BOOST_SERIALIZATION_NVP(m_noObjectives);
		archive &BOOST_SERIALIZATION_NVP(m_age);
		archive &BOOST_SERIALIZATION_NVP(m_matingProbability);
		archive &BOOST_SERIALIZATION_NVP(m_selectionProbability);
		archive &BOOST_SERIALIZATION_NVP(m_penalizedFitness);
		archive &BOOST_SERIALIZATION_NVP(m_unpenalizedFitness);
		archive &BOOST_SERIALIZATION_NVP(m_rank);
		archive &BOOST_SERIALIZATION_NVP(m_share);

		archive &boost::serialization::make_nvp("chromosome_0", boost::get<0>(*this));
		archive &boost::serialization::make_nvp("chromosome_1", boost::get<1>(*this));
		archive &boost::serialization::make_nvp("chromosome_2", boost::get<2>(*this));
		archive &boost::serialization::make_nvp("chromosome_3", boost::get<3>(*this));
		archive &boost::serialization::make_nvp("chromosome_4", boost::get<4>(*this));
		archive &boost::serialization::make_nvp("chromosome_5", boost::get<5>(*this));
		archive &boost::serialization::make_nvp("chromosome_6", boost::get<6>(*this));
		archive &boost::serialization::make_nvp("chromosome_7", boost::get<7>(*this));
		archive &boost::serialization::make_nvp("chromosome_8", boost::get<8>(*this));
		archive &boost::serialization::make_nvp("chromosome_9", boost::get<9>(*this));

	}

	/**
	 * \brief Checks for equality
	 */
	bool operator==(const this_type &rhs) const {
		return(
		        m_searchPoint == rhs.m_searchPoint &&
		        m_age == rhs.m_age &&
		        m_noObjectives == rhs.m_noObjectives &&
		        m_matingProbability == rhs.m_matingProbability &&
		        m_selectionProbability == rhs.m_selectionProbability &&
		        m_rank == rhs.m_rank &&
		        m_share == rhs.m_share &&
		        m_penalizedFitness == rhs.m_penalizedFitness &&
		        m_unpenalizedFitness == rhs.m_unpenalizedFitness
		      );
	}

protected:
	SearchSpaceType m_searchPoint; ///< The search point associated with the individual.

	unsigned int m_age;	///< The age of the individual (in generations).
	unsigned int m_noObjectives; ///< The number of objectives.

	double m_matingProbability; ///< The mating probability of this individual.
	double m_selectionProbability; ///< The selection probability of this individual.

	unsigned int m_rank; ///< The level of non-dominance of the individual. The lower the better.
	double m_share; ///< The (multi-objective) share of the individual.

	FitnessType m_penalizedFitness; ///< Penalized fitness of the individual.
	FitnessType m_unpenalizedFitness; ///< Unpenalized fitness of the individual.
	FitnessType m_scaledFitness; ///< Scaled fitness of the individual.

};

/*!
 * Creates an individual of a certain type.
 */
template<typename SearchPointType>
TypedIndividual< SearchPointType > make_individual(const SearchPointType &p,
        const std::vector<double> &fitness,
        const std::vector<double> &unpenalizedFitness) {
	TypedIndividual< SearchPointType > result;
	result.setNoObjectives(fitness.size());
	result.fitness(tag::UnpenalizedFitness()) = unpenalizedFitness;
	result.fitness(tag::PenalizedFitness()) = fitness;

	return(result);
}

/**
 * \brief Allows for extracting the penalized and unpenalized fitness of arbitrary
 *  TypedIndividuals based on fitness traits.
 */
template<
typename T0,
         typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5,
         typename T6,
         typename T7,
         typename T8,
         typename T9,
         typename T10>
struct FitnessTraits< TypedIndividual<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> > {

	typedef TypedIndividual<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> IndividualType;

	typename IndividualType::FitnessType &operator()(IndividualType &t, tag::PenalizedFitness p) const {
		return(t.fitness(p));
	}

	const typename IndividualType::FitnessType &operator()(const IndividualType &t, tag::PenalizedFitness p) const {
		return(t.fitness(p));
	}

	typename IndividualType::FitnessType &operator()(IndividualType &t, tag::UnpenalizedFitness p) const {
		return(t.fitness(p));
	}

	const typename IndividualType::FitnessType &operator()(const IndividualType &t, tag::UnpenalizedFitness p) const {
		return(t.fitness(p));
	}

	typename IndividualType::FitnessType &operator()(IndividualType &t, tag::ScaledFitness p) const {
		return(t.fitness(p));
	}

	const typename IndividualType::FitnessType &operator()(const IndividualType &t, tag::ScaledFitness p) const {
		return(t.fitness(p));
	}

};

}
#endif
