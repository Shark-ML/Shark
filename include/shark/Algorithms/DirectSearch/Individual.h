/*!
 * 
 *
 * \brief       TypedIndividual
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2014
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_TYPED_INDIVIDUAL_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_TYPED_INDIVIDUAL_H

#include <shark/LinAlg/Base.h>
#include <boost/range/adaptor/transformed.hpp>

namespace shark {

/// \brief Individual is a simple templated class modelling
/// an individual that acts as a candidate solution in an evolutionary algorithm.
///
/// The class holds the current search point as well as the penalized and unpenalized fitness,
/// its domination rank with respect to the population, its age, a boolean variable determining
/// whether the individual is selected for the next parent generation and some payload chromosome
/// which is by default a RealVector.
///
/// The states mean the following:
/// - the search point is the point in search space the individual represents.
/// - the penalized and unpenailzed fitness are related by: 
/// if the search point is in the search region of the optimization region, penalized and unpenalized
/// fitness are the same. otherwise the unpenalized fitness is the value of the closest feasible point
/// to the search point. The penalized fitness is the same value plus an penalty term. Usually this
/// is ||s-closestFeasible(s)||^2, the squared distance between the search point and the closest
/// feasible point. 
///  - the domination rank indicates in which front the individual is. a nondominated individual has rank
///    1, individuals that are only dominated by individuals with rank one have rank 2 and so on.
///    In single objective optimization, the rank is simply the number of individuals with better fitness+1.
/// -  the age is the number of generations the individual has survived. 
/// - selection: survival selection schemes never delete or move points, instead they indicate which points
///  are to be deleted.
template< typename PointType, class FitnessTypeT, class Chromosome = RealVector > 
class Individual {
public:

	typedef FitnessTypeT FitnessType;

	typedef PointType SearchPointType;
	
	///\brief Ordering relation by the ranks of the individuals
	struct RankOrdering{
		bool operator()(Individual const& individual1, Individual const& individual2){
			return individual1.rank() < individual2.rank();
		}
	};
	
	///\brief Ordering relation by the fitness of the individuals(only single objective)
	struct FitnessOrdering{
		bool operator()(Individual const& individual1, Individual const& individual2){
			return individual1.unpenalizedFitness()  < individual2.unpenalizedFitness() ;
		}
	};

	/// \brief Default constructor that initializes the individual's attributes to default values.
	Individual() 
	: m_rank(0)
	, m_selected(false)
	{}

	/// \brief Returns a reference to the search point that is associated with the individual.
	SearchPointType& searchPoint() {
		return m_searchPoint;
	}

	/// \brief Returns a const reference to the search point that is associated with the individual.
	SearchPointType const& searchPoint() const {
		return m_searchPoint;
	}
	
	/// \brief Returns a reference to the chromosome that is associated with the individual.
	Chromosome& chromosome() {
		return m_chromosome;
	}

	/// \brief Returns a const reference to the chromosome that is associated with the individual.
	Chromosome const& chromosome() const{
		return m_chromosome;
	}

	/// \brief Returns a reference to the unpenalized fitness of the individual. 
	FitnessType& unpenalizedFitness() {
		return m_unpenalizedFitness;
	}

	/// \brief Returns the unpenalized fitness of the individual. 
	FitnessType const& unpenalizedFitness() const {
		return m_unpenalizedFitness;
	}

	/// \brief Returns a reference to the penalized fitness of the individual. 
	FitnessType& penalizedFitness() {
		return m_penalizedFitness;
	}
	/// \brief Returns the unpenalized fitness of the individual. 
	FitnessType const& penalizedFitness() const {
		return m_penalizedFitness;
	}

	/// \brief Returns the level of non-dominance of the individual.
	unsigned int rank() const {
		return m_rank;
	}

	/// \brief Returns a reference to the level of non-dominance of the individual. Allows for lvalue()-semantic.
	unsigned int& rank() {
		return m_rank;
	}

	/// \brief Returns true if the individual is selected for the next parent generation 
	bool selected() const {
		return m_selected;
	}

	/// \brief Returns true if the individual is selected for the next parent generation 
	bool& selected() {
		return m_selected;
	}

	/// \brief Stores the individual and all of its chromosomes in an archive.
	template<typename Archive>
	void serialize(Archive & archive, const unsigned int version) {
		archive & BOOST_SERIALIZATION_NVP(m_searchPoint);
		archive & BOOST_SERIALIZATION_NVP(m_chromosome);
		archive & BOOST_SERIALIZATION_NVP(m_penalizedFitness);
		archive & BOOST_SERIALIZATION_NVP(m_unpenalizedFitness);
		archive & BOOST_SERIALIZATION_NVP(m_rank);
		archive & BOOST_SERIALIZATION_NVP(m_selected);

	}
	
	friend void swap(Individual& i1, Individual& i2){
		using std::swap;
		swap(i1.m_searchPoint,i2.m_searchPoint);
		swap(i1.m_chromosome,i2.m_chromosome);
		swap(i1.m_rank,i2.m_rank);
		swap(i1.m_selected,i2.m_selected);
		swap(i1.m_unpenalizedFitness,i2.m_unpenalizedFitness);
		swap(i1.m_penalizedFitness,i2.m_penalizedFitness);
	}

protected:
	SearchPointType m_searchPoint; ///< The search point associated with the individual.
	Chromosome m_chromosome; ///< The search point associated with the individual.

	unsigned int m_rank; ///< The level of non-dominance of the individual. The lower the better.
	bool m_selected; ///< Is the individual selected for the next parent set?

	FitnessType m_penalizedFitness; ///< Penalized fitness of the individual.
	FitnessType m_unpenalizedFitness; ///< Unpenalized fitness of the individual.
};

//TODO: pre C++14, this is too hard to implement using lambdas only.
namespace detail{
	struct IndividualPenalizedFitnessFunctor{
		template<class Individual>
		typename Individual::FitnessType& operator()(Individual& ind)const{
			return ind.penalizedFitness();
		}
		template<class Individual>
		typename Individual::FitnessType const& operator()(Individual const& ind)const{
			return ind.penalizedFitness();
		}
	};
	
	struct IndividualUnpenalizedFitnessFunctor{
		template<class Individual>
		typename Individual::FitnessType& operator()(Individual& ind)const{
			return ind.unpenalizedFitness();
		}
		template<class Individual>
		typename Individual::FitnessType const& operator()(Individual const& ind)const{
			return ind.unpenalizedFitness();
		}
	};
	
	struct IndividualSearchPointFunctor{
		template<class Individual>
		typename Individual::SearchPointType& operator()(Individual& ind)const{
			return ind.unpenalizedFitness();
		}
		template<class Individual>
		typename Individual::SearchPointType const& operator()(Individual const& ind)const{
			return ind.searchPoint();
		}
	};
	
	struct IndividualRankFunctor{
		template<class Individual>
		unsigned int& operator()(Individual& ind)const{
			return ind.rank();
		}
		template<class Individual>
		unsigned int operator()(Individual const& ind)const{
			return ind.rank();
		}
	};
}

template<class IndividualRange>
auto penalizedFitness(IndividualRange& range) -> decltype(
		boost::adaptors::transform(range,detail::IndividualPenalizedFitnessFunctor())
){
	return boost::adaptors::transform(range,detail::IndividualPenalizedFitnessFunctor());
}

template<class IndividualRange>
auto unpenalizedFitness(IndividualRange& range) -> decltype(
		boost::adaptors::transform(range,detail::IndividualUnpenalizedFitnessFunctor())
){
	return boost::adaptors::transform(range,detail::IndividualUnpenalizedFitnessFunctor());
}

template<class IndividualRange>
auto ranks(IndividualRange& range) -> decltype(
		boost::adaptors::transform(range,detail::IndividualRankFunctor())
){
	return boost::adaptors::transform(range,detail::IndividualRankFunctor());
}


template<class IndividualRange>
auto searchPoint(IndividualRange& range) -> decltype(
		boost::adaptors::transform(range,detail::IndividualSearchPointFunctor())
){
	return boost::adaptors::transform(range,detail::IndividualSearchPointFunctor());
}

}
#endif
