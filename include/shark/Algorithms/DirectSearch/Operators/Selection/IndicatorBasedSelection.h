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
#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_INDICATOR_BASED_SELECTION_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_OPERATORS_SELECTION_INDICATOR_BASED_SELECTION_H

#include <shark/Algorithms/DirectSearch/FitnessExtractor.h>
#include <shark/Algorithms/DirectSearch/FastNonDominatedSort.h>

#include <map>
#include <vector>

namespace shark {

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

	/** \cond */
	template<typename T>
	struct view_reference {
		T * mep_value;
	public:
		view_reference() : mep_value( NULL ){}
		view_reference(T & value) : mep_value( &value ){}

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
		
		template<class U>
		RealVector const& fitness(U tag)const{
			return mep_value->fitness(tag);
		}
		
		bool& selected(){
			return mep_value->selected();
		}
	};
	/** \endcond */

	/**
	* \brief Default c'tor.
	* \param [in] mu The target size mu of the population.
	* \param [in] noObjectives The number of objectives to consider.
	*/
	IndicatorBasedSelection( unsigned int mu = 0) : m_mu( mu ){}

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
		
		//perform a nondominated sort to assign the rank to every element
		FastNonDominatedSort nonDomSort;
		nonDomSort(population);
		
		typedef std::vector< view_reference<typename PopulationType::value_type > > View;

		unsigned int maxRank = 0;
		std::map< unsigned int, View > fronts;

		for( unsigned int i = 0; i < population.size(); i++ ) {
			maxRank = std::max( maxRank, static_cast<unsigned int>( population[i].rank() ) );
			fronts[population[i].rank()].push_back( population[i] );
			population[i].selected() = true;
		}

		//deselect the highest rank fronts until we would end up with less than mu elements
		unsigned int rank = maxRank;
		unsigned int popSize = population.size();
		
		while(popSize-fronts[rank].size() >= m_mu){
			//deselect all elements in this front
			View & front = fronts[rank];
			for(std::size_t i = 0; i != front.size(); ++i){
				front[i].selected() = false;
			}
			popSize -= front.size();
			--rank;
		}
		
		//now use the indicator to deselect the worst approximating elements of the last selected front
		m_indicator.updateInternals(FitnessExtractor(),population);
		View& front = fronts[rank];
		for(; popSize >=m_mu;--popSize) {
			unsigned int lc = m_indicator.leastContributor(FitnessExtractor(),front);
			front[lc].selected() = false;
			front.erase( front.begin() + lc );
		}
	}


	template <typename Node>
	void configure( const Node & node )
	{
		m_indicator.configure( node );
	}

	/**
	* \brief Accesses the target size mu of the population.
	*/
	unsigned int mu() const
	{
		return m_mu;
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
		archive & BOOST_SERIALIZATION_NVP( m_mu );
	}

	Indicator m_indicator; ///< Instance of the second level sorting criterion.
	FitnessExtractor m_extractor; ///< Maps individuals to the objective space.
	unsigned int m_mu; ///< Target population size.
};
}

#endif
