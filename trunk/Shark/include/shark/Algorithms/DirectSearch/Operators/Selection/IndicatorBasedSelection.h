/*!
 *
 *
 * \brief       Indicator-based selection strategy for multi-objective selection.
 *
 *
 *
 * \author      T.Voss
 * \date        2010
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
//===========================================================================
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_INDICATOR_BASED_SELECTION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_INDICATOR_BASED_SELECTION_H

#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Algorithms/DirectSearch/BoundingBoxCalculator.h>

#include <shark/Core/OpenMP.h>
#include <iterator>
#include <limits>
#include <map>
#include <vector>

namespace shark {
/** \cond */
template<typename T>
struct view_reference {
	T * mep_value;
	bool m_isBoundaryElement;
public:
	view_reference() : mep_value( NULL ),
		m_isBoundaryElement( false )
	{

	}

	view_reference(T & value) : mep_value( &value ),
		m_isBoundaryElement( false )
	{
	}

	operator T & ()
	{
		return( *mep_value );
	}

	operator const T & () const
	{
		return( *mep_value );
	}

	view_reference<T> operator=( const T & rhs )
	{
		*mep_value = rhs;
		return( *this );
	}
};
/** \endcond */

/**
* \brief Implements the well-known indicator-based selection strategy.
*
* See
* Kalyanmoy Deb and Amrit Pratap and Sameer Agarwal and T. Meyarivan,
* A Fast Elitist Multi-Objective Genetic Algorithm: NSGA-II,
* IEEE Transactions on Evolutionary Computation
* Year 2000, Volume 6, p. 182-197
*
* \tparam Indicator The second-level sorting criterion.
*/
template<typename Indicator>
struct IndicatorBasedSelection {

	/**
	* \brief Value that marks the maximum share, i.e., the worst.
	*/
	static double BEST_SHARE()
	{
		return( 1E10 );
	}

	/**
	* \brief Value that marks the minimum share, i.e., the best.
	*/
	static double WORST_SHARE()
	{
		return( -1E10 );
	}

	/**
	* \brief Assigns the maximum share to every member of the population.
	*/
	template<typename PopulationType>
	struct MaxShareOperator {

		template<typename Member>
		void operator()( Member & m )
		{
			static_cast<typename PopulationType::value_type &>( m ).share() = BEST_SHARE();
		}

	};

	/**
	* \brief Assigns the minimum share to every member of the population.
	*/
	template<typename PopulationType>
	struct MinShareOperator {

		template<typename Member>
		void operator()( Member & m )
		{
			static_cast<typename PopulationType::value_type &>( m ).share() = WORST_SHARE();
		}

	};

	/**
	* \brief Default c'tor.
	* \param [in] mu The target size mu of the population.
	* \param [in] noObjectives The number of objectives to consider.
	*/
	IndicatorBasedSelection( unsigned int mu = 0, unsigned int noObjectives = 0 ) 
	: m_mu( mu )
	{
	}

	/**
	* \brief Accesses the target size mu of the population.
	*/
	unsigned int mu() const
	{
		return( m_mu );
	}

	/**
	* \brief Adjusts the target size mu of the population.
	* \param [in] mu The new target size mu.
	*/
	void setMu( unsigned int mu )
	{
		m_mu = mu;
	}

	/**
	* \brief Serializes the state of the selection operator.
	*/
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version )
	{
		archive & BOOST_SERIALIZATION_NVP( m_indicator );
		archive & BOOST_SERIALIZATION_NVP( m_extractor );
		archive & BOOST_SERIALIZATION_NVP( m_mu );
	}

	/**
	* \brief Executes the algorithm and assigns each member of the population
	* its level non-dominance (rank) and its individual contribution to the front
	* it belongs to (share).
	*
	* \param [in,out] population The population to be ranked.
	*/
	template<typename PopulationType>
	void operator()( PopulationType & population )
	{
		if(population.empty()) return;
		
		//get the number of objectives
		std::size_t noObjectives =  population[0].fitness( tag::PenalizedFitness() ).size();
		
		typedef std::vector< view_reference<typename PopulationType::value_type> > View;

		RealVector utopianFitness( noObjectives, std::numeric_limits<double>::max() );
		RealVector nadirFitness( noObjectives, -std::numeric_limits<double>::max() );

		CastingFitnessExtractor<typename PopulationType::value_type> cExtractor;

		BoundingBoxCalculator< CastingFitnessExtractor<typename PopulationType::value_type> > bbCalculator( cExtractor, utopianFitness, nadirFitness );

		unsigned int maxRank = 0;
		std::map< unsigned int, View > fronts;

		for( unsigned int i = 0; i < population.size(); i++ ) {
			maxRank = std::max( maxRank, static_cast<unsigned int>( population[i].rank() ) );

			fronts[population[i].rank()].push_back( population[i] );

			population[i].share() = WORST_SHARE();
		}

		unsigned int popSize = 0;

		for( unsigned int rank = 1; rank <= maxRank; rank++ ) {

			if( popSize >= m_mu-1 )
				break;

			View & front = fronts[rank];
			std::for_each( front.begin(), front.end(), MaxShareOperator<PopulationType>() );

			if( popSize + front.size() <= m_mu ) {
				popSize += front.size();
				continue;
			}

			std::fill( utopianFitness.begin(), utopianFitness.end(), 1E10 );
			std::fill( nadirFitness.begin(), nadirFitness.end(), -1E10 );

			std::vector<double> minCoordinates( noObjectives, std::numeric_limits<double>::max() );
			std::vector<unsigned int> minElements( noObjectives, 0 );

			for( unsigned int i = 0; i < front.size(); i++ ) {
				typename PopulationType::value_type & ind = static_cast<typename PopulationType::value_type &>( front[i] );
				bbCalculator( ind );
				for( unsigned int j = 0; j < noObjectives; j++ ) {
					if( ind.fitness( tag::PenalizedFitness() )[j] <= minCoordinates[j] ) {
						minCoordinates[j] = ind.fitness( tag::PenalizedFitness() )[j];
						minElements[j] = i;
					}
				}
			}

			for( unsigned int i = 0; i < minElements.size(); i++ ) {
				front[ minElements[i] ].m_isBoundaryElement = true;
				front[ minElements[i] ].mep_value->share() = BEST_SHARE();
			}

			for( unsigned int i = 0; i < nadirFitness.size(); i++ )
				nadirFitness[i] += 1.0;

			m_indicator.setUtopianFitness( utopianFitness );
			m_indicator.setNadirFitness( nadirFitness );

			unsigned int size = front.size();

			//std::cout << "Front size: " << rank << " = " << front.size() << std::endl;

			while( front.size() > 1 && popSize + front.size() > m_mu ) {
				unsigned int lc = m_indicator.leastContributor(cExtractor,front);
				front[lc].mep_value->share() = size - front.size();
				front.erase( front.begin() + lc );
			}

			if( front.size() == 1 )
				front[0].mep_value->share() = BEST_SHARE();
			popSize += front.size();

		}
	}

	Indicator m_indicator; ///< Instance of the second level sorting criterion.
	FitnessExtractor m_extractor; ///< Maps individuals to the objective space.

	unsigned int m_mu; ///< Target population size.
};
}

#endif
