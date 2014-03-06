/*!
 *
 *
 * \brief       Steady state (+1) Indicator-based selection strategy for multi-objective selection.
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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_SS_INDICATOR_BASED_SELECTION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_SS_INDICATOR_BASED_SELECTION_H

#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Algorithms/DirectSearch/BoundingBoxCalculator.h>

#include <shark/Core/OpenMP.h>
#include <iterator>
#include <limits>
#include <map>
#include <vector>

namespace shark {

/**
 * \brief Steady state (+1) Indicator-based selection strategy for multi-objective selection.
 */
template<typename Indicator>
struct SteadyStateIndicatorBasedSelection {

	static double BEST_SHARE;
	static double WORST_SHARE;

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
		
		typedef std::vector< view_reference<typename PopulationType::value_type > > View;

		RealVector utopianFitness( noObjectives, std::numeric_limits<double>::max() );
		RealVector nadirFitness( noObjectives, -std::numeric_limits<double>::max() );

		CastingFitnessExtractor<typename PopulationType::value_type> cExtractor;

		BoundingBoxCalculator< CastingFitnessExtractor<typename PopulationType::value_type> > bbCalculator( cExtractor, utopianFitness, nadirFitness );

		unsigned int maxRank = 0;
		std::map<unsigned int, View > fronts;

		for( unsigned int i = 0; i < population.size(); i++ ) {
			maxRank = std::max( maxRank, static_cast<unsigned int>( population[i].rank() ) );
			fronts[population[i].rank()].push_back( population[i] );
			population[i].share() = BEST_SHARE;
		}

		View & front = fronts[maxRank];

		if( front.size() == 0 )
			return;

		if( front.size() == 1 ) {
			front[0].mep_value->share() = WORST_SHARE;
			return;
		}


		for( unsigned int i = 0; i < front.size(); i++ )
			bbCalculator( front[i] );

		m_indicator.setUtopianFitness( utopianFitness );
		m_indicator.setNadirFitness( nadirFitness );

		std::size_t lc = m_indicator.leastContributor(cExtractor,front );

		front[lc].mep_value->share() = WORST_SHARE;
	}

	template <typename Node>
	void configure( const Node & node )
	{
		m_indicator.configure( node );
	}
	
	/**
	* \brief Adjusts the target size mu of the population(unused as SteadyState uses mu=1)
	* \param [in] mu The new target size mu.
	*/
	void setMu( unsigned int mu )
	{
		(void)mu;
	}

	/**
	 * \brief Stores/restores the serializer's state.
	 * \tparam Archive Type of the archive.
	 * \param [in,out] archive The archive to serialize to.
	 * \param [in] version number, currently unused.
	 */
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int version )
	{
		archive & BOOST_SERIALIZATION_NVP( m_indicator );
		archive & BOOST_SERIALIZATION_NVP( m_extractor );
	}

	Indicator m_indicator;
	FitnessExtractor m_extractor;
};

template<class T>
double SteadyStateIndicatorBasedSelection<T>::BEST_SHARE = 1E10;
template<class T>
double SteadyStateIndicatorBasedSelection<T>::WORST_SHARE = -1E10;
}

#endif
